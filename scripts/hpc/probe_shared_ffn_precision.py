"""Bounded real decoder FFN arithmetic screen, not training or precision adoption.

Capture uses unchanged update-0 micro4 neural forward only. Native VJP uses a
declared synthetic cotangent, not the science-loss adjoint. Source owners:
shared_transformer.py::_run_coordinate_blocks_reusing_projections:537-607;
strong_baseline_architecture.py::_linear/_ffn:886-898; and
shared_decoder_macro_packed.py::_decode_mapped_rows:25-86/expert_decode_fields:288-371.
No physical process is implemented or modified here.
"""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
import os
import pickle
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_real_daily_helpers(path=None):
    name = "scripts.hpc.probe_shared_expert_real_daily"
    if path is None:
        return __import__(name, fromlist=["atomic_json"])
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load explicit staged real-daily helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_helper_path = None
if __name__ == "__main__":
    _bootstrap_parser = argparse.ArgumentParser(add_help=False)
    _bootstrap_parser.add_argument("--real-daily-probe-file", type=Path)
    _helper_path = _bootstrap_parser.parse_known_args()[0].real_daily_probe_file
_helpers = load_real_daily_helpers(_helper_path)
atomic_json, atomic_pickle, compare_trees = _helpers.atomic_json, _helpers.atomic_pickle, _helpers.compare_trees


_FP32_FFN = None


def initialize_fragment_modules():
    """Initialize package-level physical lookup tables before entering JAX AD."""
    global _FP32_FFN
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _ffn
    _FP32_FFN = _ffn


@contextmanager
def deadline(seconds, label, emit):
    """Hard watchdog: Python signal alarms cannot reliably interrupt XLA compile."""
    def expired():
        emit("timeout", status="timeout", timeout_scope=label, limit_seconds=seconds)
        os._exit(124)
    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def ffn(parameters, value, *, mixed):
    import jax
    import jax.numpy as jnp
    if _FP32_FFN is None:
        initialize_fragment_modules()
    if not mixed:
        return _FP32_FFN(parameters, value)

    def linear(p, x):
        x = jnp.asarray(x, jnp.float16)
        w = jnp.asarray(p["kernel"], jnp.float16)
        y = jax.lax.dot_general(
            x, w, (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return y + jnp.asarray(p["bias"], jnp.float32)
    return linear(parameters["second"], jax.nn.gelu(linear(parameters["first"], value)))


def fragment(parameters, payloads, *, mixed):
    """Preserve each recorded chunk; do not coalesce to a giant GEMM."""
    import jax
    return tuple(
        jax.lax.map(lambda x: ffn(parameters[block], x, mixed=mixed), values)
        for block, values in payloads
    )


def forward_vjp(parameters, inputs, cotangent, block_ids, *, mixed):
    import jax
    def forward(p, x):
        return fragment(p, tuple(zip(block_ids, x, strict=True)), mixed=mixed)
    result, pullback = jax.vjp(forward, parameters, inputs)
    return result, pullback(cotangent)


def comparison(jax, actual, reference):
    result = compare_trees(jax, actual, reference)
    if result["nonfinite_actual"] or result["nonfinite_reference"]:
        result["cosine"] = None
        result["max_relative_to_reference_floor_1e_6"] = None
        return result
    dot = a2 = b2 = 0.0
    max_relative = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(reference), strict=True):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        dot += float(np.sum(a * b))
        a2 += float(np.sum(a * a))
        b2 += float(np.sum(b * b))
        max_relative = max(max_relative, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-6), initial=0)))
    result["cosine"] = float(dot / np.sqrt(a2 * b2)) if a2 * b2 > 0 else (1.0 if a2 == b2 else 0.0)
    result["max_relative_to_reference_floor_1e_6"] = max_relative
    return result


def lowering_evidence(lowered, executable):
    """Bounded metadata only; do not equate casts/cuBLAS calls with Tensor Cores."""
    stable = str(lowered.compiler_ir(dialect="stablehlo"))
    optimized = executable.as_text()
    dots = [line.strip()[:600] for line in stable.splitlines() if "stablehlo.dot_general" in line]
    dot_types = Counter(line.rsplit(" : ", 1)[-1].strip()
                        for line in stable.splitlines() if "stablehlo.dot_general" in line)
    kernels = [line.strip()[:600] for line in optimized.splitlines() if "custom_call_target=" in line]
    return {
        "stablehlo_dot_count": len(dots), "stablehlo_dot_samples": dots[:24],
        "all_forward_and_native_backward_dot_type_signatures": dict(dot_types),
        "optimized_custom_call_count": len(kernels), "optimized_custom_call_samples": kernels[:24],
        "optimized_dtype_tokens": {name: len(re.findall(r"\b" + name + r"\[", optimized)) for name in ("f16", "f32")},
        "tensor_core_execution_proven": False,
        "note": "Lowering evidence only. No GPU instruction/kernel profiling is inferred from source casts or custom-call names.",
    }


def group_records(records):
    """Group by block/shape while retaining chunk order and all inactive rows."""
    groups = {}
    for row in records:
        key = (row["block"], tuple(row["input"].shape))
        groups.setdefault(key, []).append(row)
    return tuple({
        "block": block, "inputs": np.stack([r["input"] for r in rows]),
        "active": np.stack([r["active"] for r in rows]),
        "call_indices": [r["call_index"] for r in rows],
        "sample_indices": [r["sample_index"] for r in rows],
        "routes": [r["route"] for r in rows],
        "expert_ids": [r.get("expert_id") for r in rows],
        "group_real_rows": [r.get("group_real_rows") for r in rows],
    } for (block, _shape), rows in groups.items())


def validate_payload(saved):
    if saved.get("schema_version") != "shared_real_ffn_payload_v1":
        raise ValueError("unknown FFN payload schema")
    identity = saved["identity"]
    if identity.get("artifact_id") != 742 or identity.get("daily_rows") != 4:
        raise ValueError("payload must be actual artifact742 daily micro4")
    parameters = saved["parameters"]
    if len(parameters) != 2:
        raise ValueError("exactly two decoder FFN parameter blocks required")
    for p in parameters:
        for layer, shape in (("first", (384, 1536)), ("second", (1536, 384))):
            if np.shape(p[layer]["kernel"]) != shape:
                raise ValueError("FFN width is not admitted 384/1536")
            if np.shape(p[layer]["bias"]) != (shape[1],):
                raise ValueError("FFN bias shape mismatch")
    calls = []
    for group in saved["groups"]:
        inputs, active = group["inputs"], group["active"]
        if inputs.ndim != 4 or inputs.shape[2:] != (1, 384) or active.shape != inputs.shape[:2]:
            raise ValueError("recorded chunk input/mask shapes differ")
        if inputs.dtype != np.float32 or active.dtype != np.bool_ or not np.all(np.isfinite(inputs)):
            raise ValueError("nonfinite or incorrectly typed capture")
        if inputs.shape[1] > 128 or group["block"] not in (0, 1):
            raise ValueError("invalid expert128 chunk or block")
        calls.extend(zip(group["call_indices"], [group["block"]] * inputs.shape[0], strict=True))
    by_call = {}
    for call, block in calls:
        by_call.setdefault(call, []).append(block)
    if not by_call or any(sorted(blocks) != [0, 1] for blocks in by_call.values()):
        raise ValueError("missing or duplicated coordinate-block FFN calls")
    return saved


def schedule_group(group, microbatch):
    """Restore daily vmap's sample axis without changing expert chunk ordering."""
    if microbatch not in (1, 4):
        raise ValueError("only microbatch1/4 diagnostic schedules are defined")
    indices = {}
    samples = group["sample_indices"]
    for sample in range(4):
        indices[sample] = sorted((i for i, owner in enumerate(samples) if owner == sample),
                                 key=lambda i: group["call_indices"][i])
    if set(samples) != {0, 1, 2, 3} or len({len(rows) for rows in indices.values()}) != 1:
        raise ValueError("each shape/block group must contain aligned calls for all four samples")
    def descriptors(rows):
        return [(group["routes"][i], group.get("expert_ids", [None] * len(samples))[i],
                 group.get("group_real_rows", [None] * len(samples))[i]) for i in rows]
    if any(descriptors(indices[sample]) != descriptors(indices[0]) for sample in range(1, 4)):
        raise ValueError("sample expert/route/group sequence differs; cannot restore micro4 vectorization")
    if microbatch == 4:
        order = np.asarray([indices[sample] for sample in range(4)], dtype=np.int32).T
    else:
        order = np.asarray([i for sample in range(4) for i in indices[sample]], dtype=np.int32)
    return group["inputs"][order], {
        "microbatch": microbatch, "block": group["block"],
        "captured_shape": list(group["inputs"].shape), "scheduled_shape": list(group["inputs"][order].shape),
        "calls_per_sample": len(indices[0]), "record_order_indices": order.tolist(),
        "active_count": int(np.sum(group["active"])),
        "active_counts_per_sample": [int(np.sum(group["active"][indices[sample]])) for sample in range(4)],
    }


@contextmanager
def capture_coordinate_chunks(records, emit):
    """Compile only two-block chunks while outer production lax.map stays eager."""
    import jax
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    from research.daily_coarse_graining.daily_operator import shared_decoder_macro_packed as macro
    original = st._run_coordinate_blocks_reusing_projections
    original_macro = macro._run_coordinate_blocks_reusing_projections
    original_decode = macro._decode_mapped_rows
    current = {"sample": 0, "route": "gpp", "call": 0}
    compiled = {}
    adapter_module = sys.modules.get("research.daily_coarse_graining.daily_operator.shared_decoder_expert_execution")
    original_expert_decode = getattr(adapter_module, "expert_decode_fields", None)

    def expert_decode(*args, **kwargs):
        current["pending_experts"] = iter(sorted(set(args[2])))
        try:
            return original_expert_decode(*args, **kwargs)
        finally:
            current.pop("pending_experts", None)

    def decode(*args, **kwargs):
        old = current["route"]
        old_expert = current.get("expert_id")
        old_rows = current.get("group_real_rows")
        current["route"] = "expert128"
        current["expert_id"] = next(current["pending_experts"]) if "pending_experts" in current else None
        current["group_real_rows"] = int(kwargs["query"].shape[0])
        try:
            return original_decode(*args, **kwargs)
        finally:
            current["route"] = old
            current["expert_id"] = old_expert
            current["group_real_rows"] = old_rows

    def capture(query, dynamic_tokens, **kwargs):
        config = kwargs.pop("config")
        key = repr(config)
        if key not in compiled:
            def chunk(q, dynamic, **dynamic_kwargs):
                captured = []
                original_ffn = st._ffn
                def record(p, x):
                    captured.append(x)
                    return original_ffn(p, x)
                st._ffn = record
                try:
                    output = original(q, dynamic, config=config, **dynamic_kwargs)
                finally:
                    st._ffn = original_ffn
                return output, tuple(captured)
            compiled[key] = jax.jit(chunk)
        function = compiled[key]
        # Re-enable JIT locally: the outer forward has no whole-model tracing.
        with jax.disable_jit(False), deadline(300, "capture_coordinate_chunk_compile", emit):
            output, values = jax.block_until_ready(function(query, dynamic_tokens, **kwargs))
        if len(values) != 2:
            raise ValueError("capture did not expose exactly two FFNs")
        active = np.asarray(kwargs["process_mask"]).reshape(query.shape[0])
        for block, value in enumerate(values):
            records.append({"block": block, "input": np.asarray(value), "active": active.copy(),
                            "sample_index": current["sample"], "route": current["route"], "call_index": current["call"],
                            "expert_id": current.get("expert_id"), "group_real_rows": current.get("group_real_rows")})
        current["call"] += 1
        return output
    st._run_coordinate_blocks_reusing_projections = capture
    macro._run_coordinate_blocks_reusing_projections = capture
    macro._decode_mapped_rows = decode
    if original_expert_decode is not None:
        adapter_module.expert_decode_fields = expert_decode
    try:
        yield current
    finally:
        st._run_coordinate_blocks_reusing_projections = original
        macro._run_coordinate_blocks_reusing_projections = original_macro
        macro._decode_mapped_rows = original_decode
        if original_expert_decode is not None:
            adapter_module.expert_decode_fields = original_expert_decode


def capture_run(args, emit):
    import jax
    from scripts.hpc.probe_shared_expert_real_daily import (
        trusted_profile, initial_checkpoint, context_identity, load_staged_adapter, load_backbone_assembler,
    )
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    profile, reference = trusted_profile(args)
    import subprocess
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_head != reference["source_git_head"]:
        raise ValueError("capture requires declared immutable original checkout")
    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    identity = context_identity(reference, initial_checkpoint(reference), 742)
    if context["identity"] != identity:
        raise ValueError("prepared context identity differs")
    head = context["head"]
    if (head.shared_config.resolved_decoder_width != 384 or head.shared_config.resolved_decoder_ffn_width != 1536
            or head.shared_config.coordinate_blocks != 2):
        raise ValueError("prepared architecture is not width384/FFN1536/two-block")
    checkpoint = load_learning_checkpoint(Path(identity["initial_checkpoint"]), expected_method=reference["method"])
    parameters = jax.device_put(checkpoint["parameters"])
    adapter = load_staged_adapter(args.candidate_module_file)
    assembler = load_backbone_assembler(args.backbone_module_file) if args.backbone_module_file else None
    records = []
    with capture_coordinate_chunks(records, emit) as current, jax.disable_jit():
        for sample_index in range(4):
            current["sample"] = sample_index
            sample = jax.tree_util.tree_map(lambda x: x[sample_index], context["daily4"])
            emit("capture_sample_start", sample_index=sample_index, recorded_ffn_calls=len(records))
            inputs = head.input_assembler.assemble(sample.inputs)
            neural_context = (head.assemble_context(parameters, inputs) if assembler is None else assembler(
                parameters, inputs, normalization=head.input_normalization, layout=head.layout, config=head.shared_config))
            jax.block_until_ready(adapter(
                parameters, neural_context, inputs, schema=head.learned_schema, layout=head.layout,
                input_normalization=head.input_normalization, config=head.shared_config,
                carbon_state_layout=head.carbon_state_layout,
            ))
            emit("capture_sample_complete", sample_index=sample_index, recorded_ffn_calls=len(records))
    saved = {"schema_version": "shared_real_ffn_payload_v1", "identity": identity,
             "parameters": jax.device_get(tuple(b["ffn"] for b in parameters["shared_transformer"]["coordinate_blocks"])),
             "groups": group_records(records), "source_profile": str(args.source_profile),
             "capture_strategy": "eager_neural_forward_with_two_block_chunk_jit_no_physics_no_loss_no_AD",
             "staged_adapter": str(args.candidate_module_file), "staged_backbone": str(args.backbone_module_file)}
    validate_payload(saved)
    atomic_pickle(args.payload, saved)
    return {"status": "captured", "payload": str(args.payload), "ffn_calls": len(records),
            "groups": [{k: v for k, v in g.items() if k != "inputs" and k != "active"} |
                       {"input_shape": list(g["inputs"].shape), "active_count": int(np.sum(g["active"]))} for g in saved["groups"]]}


def fragment_run(args, emit):
    import jax
    import jax.numpy as jnp
    initialize_fragment_modules()
    with args.payload.open("rb") as stream:
        saved = validate_payload(pickle.load(stream))
    parameters = jax.device_put(saved["parameters"])
    scheduled = tuple(schedule_group(g, args.microbatch) for g in saved["groups"])
    inputs = jax.device_put(tuple(values for values, _metadata in scheduled))
    scheduling = [metadata for _values, metadata in scheduled]
    emit("payload_scheduled", scheduling=scheduling,
         schedule_semantics="chunk_map_with_sample_vmap_axis" if args.microbatch == 4 else "sample_serial_chunk_map")
    block_ids = tuple(g["block"] for g in saved["groups"])
    # Include padded/inactive coordinates: they really execute FFNs. This is not
    # a downstream-mask or science cotangent; it is a fixed sensitivity screen.
    cotangent = tuple(
        jnp.sin(jnp.arange(group["inputs"].size, dtype=jnp.float32) * 0.11).reshape(group["inputs"].shape)[
            jnp.asarray(metadata["record_order_indices"], dtype=jnp.int32)]
        for group, metadata in zip(saved["groups"], scheduling, strict=True)
    )
    variants = {}
    reference = None
    for name, mixed in (("fp32", False), ("fp16_operands_fp32_result", True)):
        def call(p, x, cot):
            return forward_vjp(p, x, cot, block_ids, mixed=mixed)
        function = jax.jit(call)
        emit(name + "_before_compile")
        started = time.perf_counter()
        with deadline(300, name + "_lower_compile", emit):
            lowered = function.lower(parameters, inputs, cotangent)
            lower_seconds = time.perf_counter() - started
            began = time.perf_counter()
            executable = lowered.compile()
            reverse_compile_seconds = time.perf_counter() - began
            began = time.perf_counter()
            forward_function = jax.jit(lambda p, x: fragment(p, tuple(zip(block_ids, x, strict=True)), mixed=mixed))
            forward_executable = forward_function.lower(parameters, inputs).compile()
            forward_compile_seconds = time.perf_counter() - began
        row = {"lower_seconds": lower_seconds, "compile_seconds": time.perf_counter() - began,
               "lowering": lowering_evidence(lowered, executable)}
        row.update(compile_seconds=reverse_compile_seconds, forward_compile_seconds=forward_compile_seconds)
        memory = executable.memory_analysis()
        row["memory"] = {key: int(getattr(memory, key)) for key in (
            "argument_size_in_bytes", "temp_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes")}
        row["memory"]["estimated_live_bytes"] = sum(row["memory"][k] for k in (
            "argument_size_in_bytes", "temp_size_in_bytes", "output_size_in_bytes")) - row["memory"]["alias_size_in_bytes"]
        if row["memory"]["estimated_live_bytes"] > args.memory_limit_gib * 2**30:
            raise ValueError("fragment exceeds declared memory bound")
        variants[name] = row
        emit(name + "_compiled", variants=variants)
        jax.block_until_ready(forward_executable(parameters, inputs))
        forward_times = []
        for _ in range(args.repeats):
            began = time.perf_counter()
            jax.block_until_ready(forward_executable(parameters, inputs))
            forward_times.append(time.perf_counter() - began)
        result = jax.block_until_ready(executable(parameters, inputs, cotangent))
        times = []
        for _ in range(args.repeats):
            began = time.perf_counter()
            result = jax.block_until_ready(executable(parameters, inputs, cotangent))
            times.append(time.perf_counter() - began)
        result = jax.device_get(result)
        row.update(warm_forward_plus_native_vjp_seconds=times, warm_median_seconds=float(np.median(times)),
                   warm_forward_seconds=forward_times, warm_forward_median_seconds=float(np.median(forward_times)),
                   device_memory_stats=jax.devices()[0].memory_stats())
        # This commit precedes all candidate compilation, even on a failed gate.
        artifact = args.output.with_suffix("." + name + ".pkl")
        atomic_pickle(artifact, {"identity": saved["identity"], "payload": str(args.payload.resolve()),
                                 "result": result, "measurement": row})
        row["result_artifact"] = str(artifact)
        if reference is None:
            reference = result
        row["comparisons"] = {
            "forward": comparison(jax, result[0], reference[0]),
            "ffn_parameter_vjp": comparison(jax, result[1][0], reference[1][0]),
            "captured_input_vjp": comparison(jax, result[1][1], reference[1][1]),
        }
        emit(name + "_saved", variants=variants)
        del executable, forward_executable, forward_function, lowered, function, result
    candidate = variants["fp16_operands_fp32_result"]
    return {"status": "measured", "variants": variants,
            "identity": saved["identity"],
            "payload": str(args.payload),
            "scheduling": scheduling,
            "exact_route_gate_passed": all(c["passed"] for c in candidate["comparisons"].values()),
            "fragment_speedup": variants["fp32"]["warm_median_seconds"] / candidate["warm_median_seconds"],
            "cotangent": "deterministic_sin_0.11_original_record_order_all_coordinates_synthetic_not_science_loss",
            "physical_validation": "not_executed", "scientific_acceptance": False, "precision_adopted": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("capture", "fragment"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--source-profile", type=Path)
    p.add_argument("--prepared-context", type=Path)
    p.add_argument("--candidate-module-file", type=Path)
    p.add_argument("--backbone-module-file", type=Path)
    p.add_argument("--real-daily-probe-file", type=Path,
                   help="Explicit staged context-loader for immutable checkouts predating this probe")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--microbatch", type=int, choices=(1, 4), default=4)
    p.add_argument("--memory-limit-gib", type=int, default=12)
    return p


def run(args):
    report = {"status": "running", "mode": args.mode, "training_updates": 0,
              "scientific_acceptance": False, "precision_adopted": False, "rtol": 1e-5, "atol": 1e-6}
    started = time.perf_counter()
    write_enabled = False
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        if write_enabled:
            atomic_json(args.output, report)
        print(stage, flush=True)
    try:
        if args.repeats < 1 or args.memory_limit_gib < 1:
            raise ValueError("positive repeats and memory bound required")
        outputs = [args.output.resolve()]
        if args.mode == "capture":
            outputs.append(args.payload.resolve())
            if any(getattr(args, key) is None for key in ("source_profile", "prepared_context", "candidate_module_file")):
                raise ValueError("capture requires source-profile/prepared-context/candidate-module-file")
        inputs = [getattr(args, key) for key in ("source_profile", "prepared_context", "candidate_module_file", "backbone_module_file")]
        if args.mode == "fragment":
            inputs.append(args.payload)
            outputs.extend(args.output.with_suffix("." + name + ".pkl").resolve()
                           for name in ("fp32", "fp16_operands_fp32_result"))
        if any(path is not None and path.resolve() in outputs for path in inputs):
            raise ValueError("refusing output/input collision")
        write_enabled = True
        import jax
        if args.require_gpu and (len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu"):
            raise RuntimeError("requires exactly one visible GPU")
        emit("start", devices=[str(d) for d in jax.devices()], jax_version=jax.__version__)
        with deadline(600, args.mode + "_total", emit):
            outcome = capture_run(args, emit) if args.mode == "capture" else fragment_run(args, emit)
        emit("completed", **outcome)
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    run(parser().parse_args())
