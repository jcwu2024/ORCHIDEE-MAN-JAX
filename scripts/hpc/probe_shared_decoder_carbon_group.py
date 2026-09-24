"""Bounded actual-helper screen of five 448-coordinate litter/DOC fields."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from functools import lru_cache
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

AUDITED_FIELD_INDICES = (64, 78, 79, 80, 81)
AUDITED_FIELD_NAMES = (
    "typed.ok_leak.litter_to_doc",
    "typed.ok_leak.doc_run_2_peat",
    "typed.ok_leak.doc_free_to_adsorbed",
    "typed.ok_leak.doc_advective_interface",
    "typed.ok_leak.doc_diffusive_interface",
)
AUDITED_SIGNATURE = {
    "eligible_coordinates": 448,
    "chunk_width": 64,
    "chunk_count": 7,
    "axis_slots": 4,
    "axis_bank": 42,
    "local_bank": 1,
    "uses_carbon_bank": False,
}
FAMILY_COUNT = 20
DEFAULT_VARIANTS = ("grouped64", "grouped128", "grouped256")
OPTIONAL_VARIANTS = ("fieldlocal128", "fieldlocal256")


def load_candidate(path):
    name = "research.daily_coarse_graining.daily_operator.shared_decoder_carbon_group"
    if path is None:
        module = __import__(name, fromlist=["grouped_decode_fields"])
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load candidate module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def audited_dynamic_indices():
    from research.daily_coarse_graining.daily_operator.shared_transformer import _decoder_dynamic_plan
    from scripts.hpc.audit_shared_decoder_cost import _load_static_layout

    current = _load_static_layout()
    plan = _decoder_dynamic_plan(schema=current["schema"], layout=current["layout"])
    if len(current["layout"].source_family_names) != FAMILY_COUNT:
        raise ValueError("audited source-family count changed")
    result = []
    for field_index, name in zip(AUDITED_FIELD_INDICES, AUDITED_FIELD_NAMES, strict=True):
        if current["fields"][field_index]["field"] != name:
            raise ValueError(f"audited field identity changed: {field_index}")
        if int(current["fields"][field_index]["expert_index"]) != 5:
            raise ValueError(f"audited expert changed: {name}")
        axis_index = plan.field_axis_indices[field_index]
        local_index = plan.field_local_indices[field_index]
        axis_bank = plan.field_axis_bank_indices[field_index]
        local_bank = plan.field_local_bank_indices[field_index]
        if axis_index.shape != (448, 4) or axis_bank.size != 42 or local_bank.size != 1:
            raise ValueError(f"audited dynamic-bank shape changed: {field_index}")
        if plan.local_state_indices[local_bank].tolist() != [-1]:
            raise ValueError(f"audited local-state mask changed: {field_index}")
        result.append((axis_index, local_index))
    return tuple(result)


def export_layout_metadata(path):
    fields = [
        {
            "field_index": field_index,
            "field_name": name,
            "expert_index": 5,
            "axis_indices": axis_indices.tolist(),
            "local_indices": local_indices.tolist(),
        }
        for field_index, name, (axis_indices, local_indices) in zip(
            AUDITED_FIELD_INDICES, AUDITED_FIELD_NAMES, audited_dynamic_indices(), strict=True
        )
    ]
    payload = {
        "schema_version": "shared_decoder_carbon_group_layout_v1",
        "source": "canonical_static_layout_and_decoder_dynamic_plan",
        "signature": AUDITED_SIGNATURE,
        "family_count": FAMILY_COUNT,
        "local_state_index": -1,
        "fields": fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size}


@lru_cache(maxsize=4)
def load_layout_metadata(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "shared_decoder_carbon_group_layout_v1"
        or payload.get("signature") != AUDITED_SIGNATURE
        or payload.get("family_count") != FAMILY_COUNT
        or payload.get("local_state_index") != -1
        or len(payload.get("fields", ())) != len(AUDITED_FIELD_INDICES)
    ):
        raise ValueError("incompatible carbon-group layout metadata")
    result = []
    for row, index, name in zip(payload["fields"], AUDITED_FIELD_INDICES, AUDITED_FIELD_NAMES, strict=True):
        if row.get("field_index") != index or row.get("field_name") != name or row.get("expert_index") != 5:
            raise ValueError(f"carbon-group layout field identity changed: {index}")
        axis = np.asarray(row["axis_indices"], dtype=np.int32)
        local = np.asarray(row["local_indices"], dtype=np.int32)
        if axis.shape != (448, 4) or local.shape != (448,):
            raise ValueError(f"carbon-group layout index shape changed: {index}")
        if np.any(axis < 0) or np.any(axis >= 42) or np.any(local != 0):
            raise ValueError(f"carbon-group layout index bounds changed: {index}")
        result.append((axis, local))
    return tuple(result)


def fixture(width, rows=448, *, seed=20260925, layout_metadata_file=None):
    if rows <= 0:
        raise ValueError("rows must be positive")
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 2 + len(AUDITED_FIELD_INDICES) * 3 + 4))
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
        "family": jax.random.normal(next(keys), (FAMILY_COUNT, width), dtype=jnp.float32),
        "process": jax.random.normal(next(keys), (1, width), dtype=jnp.float32),
        "fields": tuple(
            {
                "query": jax.random.normal(next(keys), (rows, 1, width), dtype=jnp.float32),
                "axis": jax.random.normal(next(keys), (AUDITED_SIGNATURE["axis_bank"], width), dtype=jnp.float32),
                "local": jax.random.normal(next(keys), (1, width), dtype=jnp.float32),
            }
            for _ in AUDITED_FIELD_INDICES
        ),
    }
    row = np.arange(rows)
    slot = np.arange(AUDITED_SIGNATURE["axis_slots"])
    family_valid = np.arange(FAMILY_COUNT) % 4 != 0
    actual_plan = (
        load_layout_metadata(Path(layout_metadata_file)) if layout_metadata_file is not None else audited_dynamic_indices()
    ) if rows == 448 else None
    metadata = []
    for field_index in range(len(AUDITED_FIELD_INDICES)):
        active = (row + field_index) % 11 != 0
        if actual_plan is None:
            axis_index = (row[:, None] * 3 + slot[None, :] * 7 + field_index) % AUDITED_SIGNATURE["axis_bank"]
            local_index = np.zeros((rows,), dtype=np.int32)
        else:
            axis_index, local_index = actual_plan[field_index]
        axis_valid = active[:, None] & ((row[:, None] + slot[None, :] + field_index) % 5 != 0)
        local_valid = (
            active & ((row + field_index) % 7 != 0)
            if actual_plan is None
            else np.zeros((rows,), dtype=bool)
        )
        metadata.append(
            {
                "active": jnp.asarray(active),
                "family_mask": jnp.asarray(active[:, None, None] & family_valid[None, None, :]),
                "dynamic_mask": jnp.asarray(np.concatenate((axis_valid, local_valid[:, None]), axis=1)[:, None, :]),
                "process_mask": jnp.asarray(active[:, None, None]),
                "axis_indices": jnp.asarray(axis_index, dtype=jnp.int32),
                "local_indices": jnp.asarray(local_index, dtype=jnp.int32),
            }
        )
    return shared, payload, tuple(metadata)


def prepare_fields(shared, payload, metadata):
    common = []
    for block in shared["coordinate_blocks"]:
        family_key, family_value = _project_key_value(block["attention"], payload["family"])
        process_key, process_value = _project_key_value(block["attention"], payload["process"])
        common.append((family_key, family_value, process_key, process_value, block))
    fields = []
    for data, meta in zip(payload["fields"], metadata, strict=True):
        projections = []
        for family_key, family_value, process_key, process_value, block in common:
            axis_key, axis_value = _project_key_value(block["attention"], data["axis"])
            local_key, local_value = _project_key_value(block["attention"], data["local"])
            projections.append(
                _DecoderBlockProjection(
                    family_key=family_key,
                    family_value=family_value,
                    process_key=process_key,
                    process_value=process_value,
                    carbon_key=None,
                    carbon_value=None,
                    axis_key=axis_key,
                    axis_value=axis_value,
                    local_key=local_key,
                    local_value=local_value,
                )
            )
        dynamic = jnp.concatenate(
            (
                data["axis"][meta["axis_indices"]],
                data["local"][meta["local_indices"]][:, None, :],
            ),
            axis=1,
        )
        fields.append(
            DenseDecoderField(
                query=data["query"],
                dynamic_tokens=dynamic,
                dynamic_mask=meta["dynamic_mask"],
                family_mask=meta["family_mask"],
                process_mask=meta["process_mask"],
                carbon_mask=None,
                projections=tuple(projections),
                dynamic_projection_indices=(meta["axis_indices"], meta["local_indices"], meta["active"]),
                output_active=meta["active"],
            )
        )
    return tuple(fields)


def compare(actual, expected, *, rtol=1e-5, atol=1e-6):
    failed = count = 0
    max_abs = 0.0
    for new, old in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        new, old = np.asarray(new), np.asarray(old)
        failed += int(np.count_nonzero(~np.isclose(new, old, rtol=rtol, atol=atol)))
        count += old.size
        max_abs = max(max_abs, float(np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))))
    return {"passed": failed == 0, "coordinates": count, "failed": failed, "max_abs": max_abs, "rtol": rtol, "atol": atol}


def run(args, decoder):
    if args.width % 8 or args.width <= 0 or args.rows <= 0 or args.repeats <= 0:
        raise ValueError("width divisible by eight, rows and repeats positive")
    devices = jax.devices()
    if args.require_gpu and not all(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, found {devices}")
    layout_metadata_file = getattr(args, "layout_metadata_file", None)
    if args.require_gpu and args.rows == 448 and layout_metadata_file is None:
        raise ValueError("448-row GPU runs require --layout-metadata-file")
    shared, payload, metadata = fixture(args.width, args.rows, layout_metadata_file=layout_metadata_file)
    if args.microbatch == 4:
        samples = [payload]
        samples.extend(
            fixture(args.width, args.rows, seed=20260925 + index, layout_metadata_file=layout_metadata_file)[1]
            for index in range(1, 4)
        )
        payload = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *samples)
    config = SharedTransformerConfig(
        width=args.width,
        ffn_width=4 * args.width,
        coordinate_blocks=2,
        readout_width=min(args.width, 256),
    )
    report = {
        "status": "running",
        "scope": "five same-expert litter/DOC fields; no Teacher, tokenizer, physical updater, recurrence or full model",
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "width": args.width,
        "rows_per_field": args.rows,
        "microbatch": args.microbatch,
        "blocks": 2,
        "ffn_width": 4 * args.width,
        "family_count": FAMILY_COUNT,
        "audited_field_indices": AUDITED_FIELD_INDICES,
        "audited_field_names": AUDITED_FIELD_NAMES,
        "audited_signature": AUDITED_SIGNATURE,
        "axis_index_source": "actual_static_plan" if args.rows == 448 else "synthetic_padding_fixture",
        "layout_metadata_file": str(layout_metadata_file) if layout_metadata_file is not None else None,
        "local_state_index": -1,
        "local_attention_mask": "always_false" if args.rows == 448 else "synthetic_mixed_boolean_values",
        "axis_attention_mask": "synthetic_mixed_boolean_values",
        "variants": {},
        "requested_variants": ("reference64",) + tuple(args.variants),
        "comparison": {},
        "parity_tolerances": {"rtol": 1e-5, "atol": 1e-6},
        "retained_updates": 0,
        "scientific_acceptance": False,
    }

    def save(stage):
        report["stage"] = stage
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(stage, flush=True)

    reference = None
    for mode in ("reference64",) + tuple(args.variants):
        if mode == "reference64":
            decode = decoder.reference_decode_fields_chunk64
        elif mode.startswith("fieldlocal"):
            chunk_size = int(mode.removeprefix("fieldlocal"))
            def decode(parameters, fields, cfg, *, _chunk_size=chunk_size):
                return decoder.reference_decode_fields_chunked(parameters, fields, cfg, chunk_size=_chunk_size)
        else:
            chunk_size = int(mode.removeprefix("grouped"))
            def decode(parameters, fields, cfg, *, _chunk_size=chunk_size):
                return decoder.grouped_decode_fields(parameters, fields, cfg, chunk_size=_chunk_size)

        def one(parameters, inputs):
            fields = prepare_fields(parameters, inputs, metadata)
            outputs = jnp.concatenate(decode(parameters, fields, config))
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
            report["resource_limit"] = f"{mode} compiler temporary bytes exceed {args.max_compiler_temp_gib} GiB"
            save(f"{mode}_resource_limited")
            return report
        result = executable(shared, payload)
        jax.block_until_ready(result)
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = executable(shared, payload)
            jax.block_until_ready(value)
            timings.append((time.perf_counter() - start) * 1000.0)
        row["median_warm_ms"] = float(np.median(timings))
        row["warm_ms"] = timings
        if reference is None:
            reference = result
        else:
            report["comparison"][mode] = {
                "loss_output": compare(result[0], reference[0]),
                "parameter_payload_gradients": compare(result[1], reference[1]),
            }
            if not all(value["passed"] for value in report["comparison"][mode].values()):
                report["status"] = "numerical_rejection"
                save(f"{mode}_numerical_rejection")
                return report
        save(f"{mode}_complete")
    report["status"] = "passed"
    report["numerical_parity_passed"] = True
    save("complete")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--rows", type=int, default=448)
    parser.add_argument("--microbatch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--variants", nargs="+", choices=DEFAULT_VARIANTS + OPTIONAL_VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-compiler-temp-gib", type=float, default=16.0)
    parser.add_argument("--candidate-module-file", type=Path)
    parser.add_argument("--layout-metadata-file", type=Path)
    parser.add_argument("--export-layout-only", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "shared-decoder-carbon-group-local.json")
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
