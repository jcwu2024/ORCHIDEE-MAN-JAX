"""Real-input normalized key bank FP32 forward/native-VJP screen.

Capture stops at original coordinate decoder inputs (no physics or loss).
Screen recomputes K/V projections from captured raw banks, so their parameter
and raw-bank derivatives are included. Synthetic cotangents are diagnostics.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
import pickle
from pathlib import Path
import sys
import time
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        with temp.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def capture(args, emit):
    import jax
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    helper = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    _, reference = helper.trusted_profile(args)
    import subprocess
    root = Path(os.environ["ORCHIDEE_REPO_ROOT"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != reference["source_git_head"]:
        raise ValueError("requires immutable source checkout")
    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
    if context["identity"] != identity:
        raise ValueError("prepared context identity mismatch")
    head = context["head"]
    config = head.shared_config
    if (config.resolved_decoder_width, config.resolved_decoder_attention_heads, config.coordinate_blocks) != (384, 8, 2):
        raise ValueError("requires actual decoder384/8heads/2blocks")
    parameters = jax.device_put(load_learning_checkpoint(Path(identity["initial_checkpoint"]), expected_method=reference["method"])["parameters"])
    adapter = helper.load_staged_adapter(args.candidate_module_file)
    assembler = helper.load_backbone_assembler(args.backbone_module_file)
    macro = sys.modules["research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed"]
    expert = sys.modules["research.daily_coarse_graining.daily_operator.shared_decoder_expert_execution"]
    original_cache, original_field = expert._decoder_projection_cache, expert._field_dynamic_projection_cache
    original_expert, original_mapped, original_run = expert.expert_decode_fields, macro._decode_mapped_rows, st._run_coordinate_blocks_reusing_projections
    token_by_projection, expert_tokens, records = {}, {}, []
    current = {"sample": 0, "route": "gpp"}
    def cache(p, c, family, *, carbon_tokens):
        result = original_cache(p, c, family, carbon_tokens=carbon_tokens)
        for index, projections in enumerate(result):
            banks = (family[index], c.process_tokens[index][None, :], carbon_tokens)
            expert_tokens[index] = banks
            token_by_projection[id(projections)] = banks
        return result
    def field(p, projections, tokens):
        result = original_field(p, projections, tokens)
        token_by_projection[id(result)] = token_by_projection[id(projections)] + tokens
        return result
    def save(query, dynamic, kwargs, banks, route, chunk):
        row = {"sample": current["sample"], "route": route, "chunk": chunk,
               "query": query, "dynamic": dynamic, "banks": banks,
               "masks": tuple(kwargs[k] for k in ("family_mask", "dynamic_mask", "process_mask", "carbon_mask")),
               "indices": kwargs["dynamic_projection_indices"]}
        records.append(jax.device_get(row))
    def run(query, dynamic, **kwargs):
        # GPP has no dependency on this decoder's outputs while assembling inputs.
        if not any(r["sample"] == current["sample"] and r["route"] == "gpp" for r in records):
            save(query, dynamic, kwargs, token_by_projection[id(kwargs["projections"])], "gpp", query.shape[0])
        return query
    def grouped(shared, fields, owners, projections, config, *, chunk_size):
        pending = []
        for owner in sorted(set(owners)):
            group = tuple(f for f, e in zip(fields, owners, strict=True) if e == owner)
            field_banks = [token_by_projection[id(f.projections)] for f in group]
            pending.append((owner, expert_tokens[owner][:2] + (None,
                jnp.concatenate(tuple(b[3] for b in field_banks)),
                jnp.concatenate(tuple(b[4] for b in field_banks)))))
        current["pending"] = iter(pending)
        return original_expert(shared, fields, owners, projections, config, chunk_size=chunk_size)
    def mapped(shared, **kwargs):
        owner, banks = next(current["pending"])
        converted = dict(kwargs)
        converted["carbon_mask"] = None
        converted["dynamic_projection_indices"] = (kwargs["axis_indices"], kwargs["local_indices"], kwargs["active"])
        save(kwargs["query"], kwargs["dynamic"], converted, banks, f"expert{owner}", kwargs["chunk_size"])
        return jnp.zeros((kwargs["query"].shape[0],), dtype=jnp.float32)
    expert._decoder_projection_cache, expert._field_dynamic_projection_cache = cache, field
    expert.expert_decode_fields, macro._decode_mapped_rows, st._run_coordinate_blocks_reusing_projections = grouped, mapped, run
    try:
        with jax.disable_jit():
            for sample in range(4):
                current["sample"] = sample
                row = jax.tree_util.tree_map(lambda x: x[sample], context["daily4"])
                inputs = head.input_assembler.assemble(row.inputs)
                neural = assembler(parameters, inputs, normalization=head.input_normalization, layout=head.layout, config=config)
                adapter(parameters, neural, inputs, schema=head.learned_schema, layout=head.layout,
                        input_normalization=head.input_normalization, config=config, carbon_state_layout=head.carbon_state_layout)
                emit("sample_captured", sample=sample, records=len(records))
    finally:
        expert._decoder_projection_cache, expert._field_dynamic_projection_cache = original_cache, original_field
        expert.expert_decode_fields, macro._decode_mapped_rows, st._run_coordinate_blocks_reusing_projections = original_expert, original_mapped, original_run
    for row in records:
        if any(not np.all(np.isfinite(value)) for value in jax.tree_util.tree_leaves((row["query"], row["dynamic"], row["banks"]))):
            raise ValueError("nonfinite real decoder input/bank capture")
    write(args.payload, {"schema": "shared_normalized_keys_real_v1", "identity": identity, "config": config,
                         "parameters": jax.device_get(parameters["shared_transformer"]["coordinate_blocks"]),
                         "records": records, "capture": "decoder_input_boundary_no_decoder_outputs_no_physics_no_loss"})
    return {"status": "captured", "records": len(records), "payload": str(args.payload),
            "routes": [{"route": r["route"], "sample": r["sample"], "rows": r["query"].shape[0], "chunk": r["chunk"],
                        "active_rows": int(np.count_nonzero(r["indices"][2])),
                        "bank_rows": [None if bank is None else bank.shape[0] for bank in r["banks"]]} for r in records]}


def component(blocks, values, masks, indices, config, chunk, *, normalized, native_unroll=1):
    import jax
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    nk = sys.modules.get("shared_normalized_key_execution")
    if nk is None:
        nk = load(Path(__file__).with_name("shared_normalized_key_execution.py"), "shared_normalized_key_execution")
    query, dynamic, family, process, carbon, axis, local = values
    projections = []
    for block in blocks:
        attention = block["attention"]
        pairs = [st._project_key_value(attention, x) if x is not None else (None, None)
                 for x in (family, process, carbon, axis, local)]
        projections.append(st._DecoderBlockProjection(*(x for pair in pairs for x in pair)))
    projections = tuple(projections)
    if normalized:
        projections = nk.normalize_projections(projections, config)
    rows = query.shape[0]
    chunk = min(chunk, rows)
    pad = (-rows) % chunk
    def chunks(value):
        if value is None:
            return None
        padded = jnp.pad(value, ((0, pad),) + ((0, 0),) * (value.ndim - 1))
        return padded.reshape((-1, chunk) + value.shape[1:])
    packed = jax.tree_util.tree_map(chunks, (query, dynamic, masks, indices))
    runner = nk.run_blocks if normalized else st._run_coordinate_blocks_reusing_projections
    def call(items):
        q, d, m, i = items
        return runner(q, d, family_mask=m[0], dynamic_mask=m[1], process_mask=m[2], carbon_mask=m[3],
                      blocks=blocks, projections=projections, config=config, dynamic_projection_indices=i)
    if native_unroll == 1:
        output = jax.lax.map(call, packed)
    else:
        # lax.map's original map-in-scan semantics, explicit bounded unrolling.
        _, output = jax.lax.scan(lambda carry, items: (carry, call(items)), None, packed, unroll=native_unroll)
    return output.reshape((-1,) + query.shape[1:])[:rows]


def screen(args, emit):
    import jax
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    from jax.experimental.serialize_executable import serialize
    helper = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    with args.payload.open("rb") as stream:
        saved = pickle.load(stream)
    if saved["schema"] != "shared_normalized_keys_real_v1":
        raise ValueError("wrong capture schema")
    blocks = jax.device_put(saved["parameters"])
    routes = sorted({r["route"] for r in saved["records"]})
    if args.routes:
        if not set(args.routes) <= set(routes):
            raise ValueError("requested route not present in real capture")
        routes = args.routes
    candidate_name = args.variant
    results = {}
    for route in routes:
        records = sorted((r for r in saved["records"] if r["route"] == route), key=lambda x: x["sample"])
        if [r["sample"] for r in records] != [0, 1, 2, 3]:
            raise ValueError("unaligned four-row capture")
        def values(row):
            return (row["query"], row["dynamic"], *row["banks"])
        selected = records[:1] if args.microbatch == 1 else records
        def stack(*arrays):
            return None if arrays[0] is None else np.stack(arrays)
        x = jax.device_put(jax.tree_util.tree_map(stack, *(values(r) for r in selected)))
        masks = jax.device_put(jax.tree_util.tree_map(stack, *(r["masks"] for r in selected)))
        indices = jax.device_put(jax.tree_util.tree_map(stack, *(r["indices"] for r in selected)))
        shape = x[0].shape
        cot = jnp.sin(jnp.arange(np.prod(shape), dtype=jnp.float32) * 0.11).reshape(shape)
        compiled = {}
        route_report = {"shape": shape, "chunk": records[0]["chunk"], "variants": {}}
        for name, normalized in (("reference", False), (candidate_name, args.variant == "normalized")):
            native_unroll = 2 if name == "unroll2" else 1
            def forward(p, inputs):
                return jax.vmap(lambda v, m, i: component(p, v, m, i, saved["config"], records[0]["chunk"], normalized=normalized, native_unroll=native_unroll))(inputs, masks, indices)
            def combined(p, inputs, cotangent):
                output, pullback = jax.vjp(forward, p, inputs)
                return output, pullback(cotangent)
            began = time.perf_counter()
            fwd_lower = jax.jit(forward).lower(blocks, x)
            reverse_lower = jax.jit(combined).lower(blocks, x, cot)
            lower = time.perf_counter() - began
            emit("compiling", route=route, variant=name, lower_seconds=lower)
            began = time.perf_counter()
            fwd = fwd_lower.compile()
            reverse = reverse_lower.compile()
            compile_seconds = time.perf_counter() - began
            result = jax.device_get(jax.block_until_ready(reverse(blocks, x, cot)))
            jax.block_until_ready(fwd(blocks, x))
            memory = reverse.memory_analysis()
            metric = {k: int(getattr(memory, k)) for k in ("argument_size_in_bytes", "temp_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes")}
            metric["estimated_live_bytes"] = sum(metric[k] for k in ("argument_size_in_bytes", "temp_size_in_bytes", "output_size_in_bytes")) - metric["alias_size_in_bytes"]
            if metric["estimated_live_bytes"] > 12 * 2**30:
                raise ValueError("component exceeds12GiB bound")
            row = {"lower_seconds": lower, "compile_seconds": compile_seconds, "memory": metric,
                   "forward_seconds": [], "forward_native_vjp_seconds": []}
            path = args.output.with_suffix(f".{route}.{name}.pkl")
            write(path, {"identity": saved["identity"], "result": result})
            write(args.output.with_suffix(f".{route}.{name}.executable.pkl"), serialize(reverse))
            if name == "reference":
                reference = result
            row["comparisons"] = {"output": helper.compare_trees(jax, result[0], reference[0]),
                                  "all_decoder_parameter_vjp": helper.compare_trees(jax, result[1][0], reference[1][0]),
                                  "all_captured_input_vjp": helper.compare_trees(jax, result[1][1], reference[1][1])}
            row["result_artifact"] = str(path)
            route_report["variants"][name] = row
            compiled[name] = (fwd, reverse)
            args.output.with_suffix(f".{route}.{name}.hlo.txt").write_text(reverse.as_text(), encoding="utf-8")
            emit("variant_saved", route=route, variant=name, results=results | {route: route_report})
        for repetition in range(args.repeats):
            for name in (("reference", candidate_name) if repetition % 2 == 0 else (candidate_name, "reference")):
                fwd, reverse = compiled[name]
                for key, function, arguments in (("forward_seconds", fwd, (blocks, x)), ("forward_native_vjp_seconds", reverse, (blocks, x, cot))):
                    jax.block_until_ready(arguments)
                    began = time.perf_counter()
                    jax.block_until_ready(function(*arguments))
                    route_report["variants"][name][key].append(time.perf_counter() - began)
        for row in route_report["variants"].values():
            row["forward_median_seconds"] = float(np.median(row["forward_seconds"]))
            row["forward_native_vjp_median_seconds"] = float(np.median(row["forward_native_vjp_seconds"]))
        ref, cand = (route_report["variants"][k] for k in ("reference", candidate_name))
        route_report["forward_speedup"] = ref["forward_median_seconds"] / cand["forward_median_seconds"]
        route_report["vjp_speedup"] = ref["forward_native_vjp_median_seconds"] / cand["forward_native_vjp_median_seconds"]
        route_report["strict_gate_passed"] = all(c["passed"] for c in cand["comparisons"].values())
        results[route] = route_report
        emit("route_completed", results=results)
        del compiled, reference, result
    return {"status": "measured", "results": results, "microbatch": args.microbatch, "variant": args.variant,
            "strict_gate_passed": all(r["strict_gate_passed"] for r in results.values()),
            "cotangent": "synthetic_sin0.11_all_output_coordinates_not_science_loss",
            "full_model_or_complete_update_admission": False, "device_memory": jax.devices()[0].memory_stats()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("capture", "screen"), required=True)
    for key in ("output", "payload", "patch-owner", "source-profile", "prepared-context", "candidate-module-file", "backbone-module-file"):
        p.add_argument("--" + key, type=Path, required=key in ("output", "payload", "patch-owner"))
    p.add_argument("--microbatch", type=int, choices=(1, 4), default=1)
    p.add_argument("--variant", choices=("normalized", "unroll2"), default="normalized")
    p.add_argument("--routes", nargs="+")
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    if len(devices) != 1 or jax.default_backend() != "gpu":
        raise ValueError("requires exactly one GPU")
    report = {"status": "running", "mode": args.mode, "rtol": 1e-5, "atol": 1e-6, "training_updates": 0}
    started = time.perf_counter()
    def emit(stage, **kwargs):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **kwargs)
        write(args.output, report)
        print(stage, flush=True)
    try:
        emit("starting")
        emit("completed", **(capture(args, emit) if args.mode == "capture" else screen(args, emit)))
    except BaseException as error:
        emit("exception", status="error", error=str(error), error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
