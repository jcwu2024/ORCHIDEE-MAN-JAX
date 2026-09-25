"""Bounded heterogeneous actual-helper probe for non-GPP decoder packing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_coarse_graining.daily_operator.shared_decoder_dense import DenseDecoderField  # noqa: E402
from research.daily_coarse_graining.daily_operator.shared_transformer import (  # noqa: E402
    SharedTransformerConfig,
    _DecoderBlockProjection,
    _project_key_value,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (  # noqa: E402
    _linear_parameters,
    _residual_block_parameters,
)

FIELD_SPECS = (
    # expert, rows, axis slots, field-local axis bank, field-local state bank
    (0, 1, 0, 0, 1),
    (1, 3, 1, 2, 1),
    (2, 6, 2, 4, 6),
    (3, 7, 3, 7, 1),
    (4, 9, 4, 8, 9),
    (5, 11, 5, 44, 896),
    (6, 4, 1, 2, 4),
    (5, 5, 4, 7, 1),
)
EXPERT_COUNT = 7
FAMILY_COUNT = 20
VARIANTS = ("reference64", "packed16", "packed32", "packed64")
ACTUAL_VARIANTS = ("reference64", "packed128", "packed256", "packed512")
LAYOUT_SCHEMA = "shared_decoder_macro_non_gpp_layout_v1"


def export_layout_metadata(path):
    from research.daily_coarse_graining.daily_operator.shared_transformer import _decoder_dynamic_plan
    from scripts.hpc.audit_shared_decoder_cost import _load_static_layout

    current = _load_static_layout()
    plan = _decoder_dynamic_plan(schema=current["schema"], layout=current["layout"])
    rows = []
    for index, source in enumerate(current["fields"]):
        if source["uses_carbon_bank"]:
            if source["field"] != "daily.fast_gpp":
                raise ValueError(f"unexpected carbon field: {source['field']}")
            continue
        axis = np.asarray(plan.field_axis_indices[index], dtype=np.int32)
        local = np.asarray(plan.field_local_indices[index], dtype=np.int32)
        axis_bank = np.asarray(plan.field_axis_bank_indices[index], dtype=np.int32)
        local_bank = np.asarray(plan.field_local_bank_indices[index], dtype=np.int32)
        mapped = np.asarray(plan.local_state_indices[local_bank] >= 0, dtype=bool)
        eligible = int(source["eligible_coordinates"])
        if axis.shape != (eligible, source["dynamic_tokens_per_query"] - 1) or local.shape != (eligible,):
            raise ValueError(f"decoder plan shape changed: {source['field']}")
        rows.append({
            "field_index": index,
            "field_name": source["field"],
            "expert_index": int(source["expert_index"]),
            "eligible_rows": eligible,
            "axis_slots": int(axis.shape[1]),
            "axis_bank_size": int(axis_bank.size),
            "local_bank_size": int(local_bank.size),
            "axis_indices": axis.tolist(),
            "local_indices": local.tolist(),
            "local_bank_mapped": mapped.tolist(),
        })
    result = {
        "schema_version": LAYOUT_SCHEMA,
        "source": "canonical_static_layout_and_decoder_dynamic_plan",
        "family_count": len(current["layout"].source_family_names),
        "expert_count": len(current["layout"].expert_names),
        "field_count": len(rows),
        "eligible_rows": sum(row["eligible_rows"] for row in rows),
        "fields": rows,
    }
    validate_layout_metadata(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size, "fields": len(rows)}


def validate_layout_metadata(payload):
    if (
        payload.get("schema_version") != LAYOUT_SCHEMA
        or payload.get("source") != "canonical_static_layout_and_decoder_dynamic_plan"
        or payload.get("family_count") != FAMILY_COUNT
        or payload.get("expert_count") != EXPERT_COUNT
        or payload.get("field_count") != 160
        or payload.get("eligible_rows") != 8548
        or len(payload.get("fields", ())) != 160
    ):
        raise ValueError("incompatible non-GPP macro layout metadata")
    field_indices = []
    seen_names = set()
    total = 0
    for row in payload["fields"]:
        index, name = row.get("field_index"), row.get("field_name")
        count, slots = row.get("eligible_rows"), row.get("axis_slots")
        axis_bank, local_bank = row.get("axis_bank_size"), row.get("local_bank_size")
        if (
            not isinstance(index, int) or not isinstance(name, str) or not name
            or name == "daily.fast_gpp" or name in seen_names
            or not isinstance(count, int) or count <= 0
            or not isinstance(slots, int) or not 0 <= slots <= 5
            or not isinstance(axis_bank, int) or axis_bank < 0
            or not isinstance(local_bank, int) or local_bank <= 0
            or not isinstance(row.get("expert_index"), int)
            or not 0 <= row["expert_index"] < EXPERT_COUNT
        ):
            raise ValueError(f"invalid macro layout field identity or shape: {name}")
        axis = np.asarray(row["axis_indices"])
        local = np.asarray(row["local_indices"])
        mapped = np.asarray(row["local_bank_mapped"])
        if (
            axis.shape != (count, slots) or local.shape != (count,)
            or mapped.shape != (local_bank,) or mapped.dtype != np.dtype("bool")
            or (slots > 0 and not np.issubdtype(axis.dtype, np.integer))
            or not np.issubdtype(local.dtype, np.integer)
            or np.any(axis < 0) or np.any(axis >= axis_bank)
            or np.any(local < 0) or np.any(local >= local_bank)
        ):
            raise ValueError(f"invalid macro layout indices or bank mask: {name}")
        field_indices.append(index)
        seen_names.add(name)
        total += count
    if field_indices != [index for index in range(161) if index != 30] or total != 8548:
        raise ValueError("macro layout field order or total changed")
    return payload


def load_layout_metadata(path):
    return validate_layout_metadata(json.loads(path.read_text(encoding="utf-8")))


def _layout_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_candidate(path):
    name = "research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed"
    if path is None:
        module = __import__(name, fromlist=["packed_decode_fields"])
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load candidate module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def fixture(width, *, seed=20260925, layout_metadata=None):
    specs = (
        FIELD_SPECS if layout_metadata is None else tuple(
            (row["expert_index"], row["eligible_rows"], row["axis_slots"],
             row["axis_bank_size"], row["local_bank_size"])
            for row in layout_metadata["fields"]
        )
    )
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 2 + len(specs) * 3 + 4))
    blocks = []
    for block_index in range(2):
        block = _residual_block_parameters(next(keys), width, 4 * width)
        for part in ("key", "value"):
            block["attention"][part]["bias"] = jnp.linspace(0.01, 0.02, width, dtype=jnp.float32) * (block_index + 1)
        blocks.append(block)
    shared = {
        "coordinate_blocks": tuple(blocks),
        "readout_first": _linear_parameters(next(keys), width, min(width, 256)),
        "readout_second": _linear_parameters(next(keys), min(width, 256), 1),
    }
    payload = {
        "family": jax.random.normal(next(keys), (EXPERT_COUNT, FAMILY_COUNT, width), dtype=jnp.float32),
        "process": jax.random.normal(next(keys), (EXPERT_COUNT, 1, width), dtype=jnp.float32),
        "fields": tuple(
            {
                "query": jax.random.normal(next(keys), (rows, 1, width), dtype=jnp.float32),
                "axis": jax.random.normal(next(keys), (axis_bank, width), dtype=jnp.float32),
                "local": jax.random.normal(next(keys), (local_bank, width), dtype=jnp.float32),
            }
            for _expert, rows, _slots, axis_bank, local_bank in specs
        ),
    }
    masks = []
    for field_index, (expert, rows, slots, axis_bank, local_bank) in enumerate(specs):
        row = np.arange(rows)
        active = (row + field_index) % 5 != 0
        if not active.any():
            active[-1] = True
        family_valid = np.arange(FAMILY_COUNT) % (expert % 3 + 2) != 0
        if layout_metadata is None:
            axis_index = (
                (row[:, None] * 3 + np.arange(slots)[None, :] * 5 + field_index) % axis_bank
                if slots else np.zeros((rows, 0), dtype=np.int32)
            )
            local_index = (row * 89 + field_index) % local_bank
            local_mapped = np.ones((local_bank,), dtype=bool)
        else:
            source = layout_metadata["fields"][field_index]
            axis_index = np.asarray(source["axis_indices"], dtype=np.int32)
            local_index = np.asarray(source["local_indices"], dtype=np.int32)
            local_mapped = np.asarray(source["local_bank_mapped"], dtype=bool)
        axis_valid = active[:, None] & ((row[:, None] + np.arange(slots)[None, :]) % 3 != 0)
        local_valid = active & local_mapped[local_index] & ((row + field_index) % 4 != 0)
        masks.append(
            {
                "active": jnp.asarray(active),
                "family": jnp.asarray(active[:, None, None] & family_valid[None, None, :]),
                "dynamic": jnp.asarray(np.concatenate((axis_valid, local_valid[:, None]), axis=1)[:, None, :]),
                "process": jnp.asarray(active[:, None, None]),
                "axis_indices": jnp.asarray(axis_index, dtype=jnp.int32),
                "local_indices": jnp.asarray(local_index, dtype=jnp.int32),
            }
        )
    return shared, payload, tuple(masks)


def prepare_fields(shared, payload, masks, *, specs=FIELD_SPECS):
    expert_projections = []
    for expert in range(EXPERT_COUNT):
        blocks = []
        for block in shared["coordinate_blocks"]:
            family_key, family_value = _project_key_value(block["attention"], payload["family"][expert])
            process_key, process_value = _project_key_value(block["attention"], payload["process"][expert])
            blocks.append(
                _DecoderBlockProjection(
                    family_key=family_key, family_value=family_value,
                    process_key=process_key, process_value=process_value,
                    carbon_key=None, carbon_value=None,
                    axis_key=None, axis_value=None, local_key=None, local_value=None,
                )
            )
        expert_projections.append(tuple(blocks))
    fields = []
    for (expert, _rows, _slots, _axis_bank, _local_bank), data, mask in zip(
        specs, payload["fields"], masks, strict=True
    ):
        projections = []
        for block, common in zip(shared["coordinate_blocks"], expert_projections[expert], strict=True):
            axis_key, axis_value = _project_key_value(block["attention"], data["axis"])
            local_key, local_value = _project_key_value(block["attention"], data["local"])
            projections.append(common._replace(
                axis_key=axis_key, axis_value=axis_value,
                local_key=local_key, local_value=local_value,
            ))
        dynamic = jnp.concatenate(
            (data["axis"][mask["axis_indices"]], data["local"][mask["local_indices"]][:, None, :]),
            axis=1,
        )
        fields.append(
            DenseDecoderField(
                query=data["query"], dynamic_tokens=dynamic,
                dynamic_mask=mask["dynamic"], family_mask=mask["family"],
                process_mask=mask["process"], carbon_mask=None,
                projections=tuple(projections),
                dynamic_projection_indices=(mask["axis_indices"], mask["local_indices"], mask["active"]),
                output_active=mask["active"],
            )
        )
    return tuple(fields), tuple(expert_projections)


def compare(actual, expected, *, rtol=1e-5, atol=1e-6):
    failed = coordinates = 0
    maximum = 0.0
    for new, old in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        new, old = np.asarray(new), np.asarray(old)
        failed += int(np.count_nonzero(~np.isclose(new, old, rtol=rtol, atol=atol)))
        coordinates += old.size
        if old.size:
            maximum = max(maximum, float(np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))))
    return {"passed": failed == 0, "coordinates": coordinates, "failed": failed, "max_abs": maximum, "rtol": rtol, "atol": atol}


def run(args, decoder):
    if args.width <= 0 or args.width % 8 or args.repeats <= 0:
        raise ValueError("width must be divisible by eight and repeats positive")
    devices = jax.devices()
    if args.require_gpu and not all(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, found {devices}")
    layout = load_layout_metadata(args.layout_metadata_file) if args.layout_metadata_file else None
    specs = (
        FIELD_SPECS if layout is None else tuple(
            (row["expert_index"], row["eligible_rows"], row["axis_slots"],
             row["axis_bank_size"], row["local_bank_size"])
            for row in layout["fields"]
        )
    )
    variants = args.variants or (ACTUAL_VARIANTS[1:] if layout is not None else VARIANTS[1:])
    modes = ("reference64",) + tuple(variants)
    reference_artifact = args.reference_artifact or args.output.with_suffix(".reference.pkl")
    identity = {"width": args.width, "microbatch": args.microbatch,
                "layout_sha256": _layout_digest(layout) if layout is not None else None}
    if args.resume_reference and layout is None:
        raise ValueError("reference resume requires actual layout metadata")
    shared, payload, masks = fixture(args.width, layout_metadata=layout)
    if args.microbatch == 4:
        samples = [payload]
        samples.extend(
            fixture(args.width, seed=20260925 + index, layout_metadata=layout)[1]
            for index in range(1, 4)
        )
        payload = jax.tree_util.tree_map(lambda *items: jnp.stack(items), *samples)
    config = SharedTransformerConfig(
        width=args.width, ffn_width=4 * args.width,
        coordinate_blocks=2, readout_width=min(args.width, 256),
    )
    experts = tuple(item[0] for item in specs)
    report = {
        "status": "running",
        "scope": (
            "actual non-GPP 160-field shape/inverse-index layout with synthetic FP32 values and masks; "
            "not Teacher data or runtime masks"
            if layout is not None else
            "heterogeneous eight-field non-carbon fragment; not the actual 160-field layout"
        ),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "width": args.width,
        "microbatch": args.microbatch,
        "field_specs": specs,
        "layout_sha256": identity["layout_sha256"],
        "layout_metadata_file": str(args.layout_metadata_file) if layout is not None else None,
        "reference_artifact": str(reference_artifact) if layout is not None else None,
        "variants": {},
        "comparison": {},
        "retained_updates": 0,
        "scientific_acceptance": False,
    }

    def save(stage):
        report["stage"] = stage
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(stage, flush=True)

    reference = None
    if args.resume_reference:
        with reference_artifact.open("rb") as stream:
            saved = pickle.load(stream)
        if saved.get("identity") != identity:
            raise ValueError("reference artifact identity differs from requested probe")
        reference = saved["result"]
        report["variants"]["reference64"] = {"status": "loaded_from_artifact"}
        save("reference64_restored")
    for mode in modes if reference is None else modes[1:]:
        def one(parameters, inputs):
            fields, projections = prepare_fields(parameters, inputs, masks, specs=specs)
            if mode == "reference64":
                values = decoder.reference_decode_fields(parameters, fields, config)
            else:
                values = decoder.packed_decode_fields(
                    parameters, fields, experts, projections, config,
                    chunk_size=int(mode.removeprefix("packed")),
                )
            outputs = jnp.concatenate(values)
            target = jnp.sin(jnp.arange(outputs.size, dtype=jnp.float32) * 0.13)
            return jnp.mean((outputs - target) ** 2), outputs

        if args.microbatch == 1:
            objective = one
        else:
            def objective(parameters, inputs):
                losses, outputs = jax.vmap(one, in_axes=(None, 0))(parameters, inputs)
                return jnp.mean(losses), outputs

        function = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
        save(f"{mode}_trace_lower_compile")
        equations = len(jax.make_jaxpr(lambda p, x: objective(p, x)[0])(shared, payload).jaxpr.eqns)
        start = time.perf_counter()
        lowered = function.lower(shared, payload)
        lower_seconds = time.perf_counter() - start
        start = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter() - start
        memory = executable.memory_analysis()
        row = {
            "primal_jaxpr_equations": equations,
            "lower_seconds": lower_seconds,
            "compile_or_cache_load_seconds": compile_seconds,
            "compiler_temporary_bytes": memory.temp_size_in_bytes,
            "compiler_argument_bytes": memory.argument_size_in_bytes,
            "compiler_output_bytes": memory.output_size_in_bytes,
        }
        report["variants"][mode] = row
        save(f"{mode}_compiled")
        if memory.temp_size_in_bytes > args.max_compiler_temp_gib * (1024**3):
            report["status"] = "resource_limited"
            save(f"{mode}_resource_limited")
            return report
        result = executable(shared, payload)
        jax.block_until_ready(result)
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            current = executable(shared, payload)
            jax.block_until_ready(current)
            timings.append((time.perf_counter() - start) * 1000.0)
        row["median_warm_ms"] = float(np.median(timings))
        if reference is None:
            reference = result
            if layout is not None:
                reference_artifact.parent.mkdir(parents=True, exist_ok=True)
                with reference_artifact.open("wb") as stream:
                    pickle.dump({"identity": identity, "result": jax.device_get(result)}, stream,
                                protocol=pickle.HIGHEST_PROTOCOL)
                row["reference_artifact_bytes"] = reference_artifact.stat().st_size
        else:
            report["comparison"][mode] = {
                "loss_output": compare(result[0], reference[0]),
                "parameter_payload_gradients": compare(result[1], reference[1]),
            }
            if not all(item["passed"] for item in report["comparison"][mode].values()):
                report["status"] = "numerical_rejection"
                save(f"{mode}_numerical_rejection")
                return report
        save(f"{mode}_complete")
    report["status"] = "passed"
    save("complete")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--microbatch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-compiler-temp-gib", type=float, default=4.0)
    parser.add_argument("--candidate-module-file", type=Path)
    parser.add_argument("--export-layout-only", type=Path)
    parser.add_argument("--layout-metadata-file", type=Path)
    parser.add_argument("--variants", nargs="+", choices=tuple(mode for mode in VARIANTS[1:] + ACTUAL_VARIANTS[1:]))
    parser.add_argument("--reference-artifact", type=Path)
    parser.add_argument("--resume-reference", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "shared-decoder-macro-packed-local.json")
    args = parser.parse_args()
    if args.export_layout_only is not None:
        print(json.dumps(export_layout_metadata(args.export_layout_only), indent=2))
        return
    report = run(args, load_candidate(args.candidate_module_file))
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
