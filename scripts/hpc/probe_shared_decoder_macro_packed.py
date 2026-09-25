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
EXPERT_VARIANTS = ("expert128", "expert256")
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
    used_experts = tuple(sorted({item[0] for item in specs}))
    expert_projections = []
    for expert in range(EXPERT_COUNT):
        if expert not in used_experts:
            expert_projections.append(None)
            continue
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


def _reference_groups(specs, fields_per_block):
    if fields_per_block <= 0:
        raise ValueError("reference fields per block must be positive")
    return tuple(
        tuple(range(start, min(start + fields_per_block, len(specs))))
        for start in range(0, len(specs), fields_per_block)
    )


def _reference_piece(parameters, inputs, *, masks, specs, indices, config, decoder,
                    output_offset, total_rows, microbatch):
    piece_specs = tuple(specs[index] for index in indices)
    piece_masks = tuple(masks[index] for index in indices)
    piece_inputs = dict(inputs)
    piece_inputs["fields"] = tuple(inputs["fields"][index] for index in indices)

    def one(sample):
        fields, _ = prepare_fields(parameters, sample, piece_masks, specs=piece_specs)
        values = decoder.reference_decode_fields(parameters, fields, config)
        outputs = jnp.concatenate(values)
        target = jnp.sin(
            jnp.arange(output_offset, output_offset + outputs.size, dtype=jnp.float32) * 0.13
        )
        # Each piece uses the full objective denominator; piece losses therefore add exactly.
        return jnp.sum((outputs - target) ** 2) / total_rows, outputs

    if microbatch == 1:
        return one(piece_inputs)
    losses, outputs = jax.vmap(one)(piece_inputs)
    return jnp.mean(losses), outputs


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _write_reference_artifact(path, state):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _build_modular_reference(args, decoder, shared, payload, masks, specs, config,
                             identity, reference_artifact, report, save):
    fields_per_block = getattr(args, "reference_fields_per_block", 8)
    groups = _reference_groups(specs, fields_per_block)
    state = {"identity": identity, "completed_groups": 0, "outputs": [],
             "loss": 0.0, "parameter_gradients": None, "input_gradients": None,
             "group_metrics": []}
    if args.resume_reference:
        with reference_artifact.open("rb") as stream:
            saved = pickle.load(stream)
        if saved.get("identity") != identity:
            raise ValueError("reference artifact identity differs from requested probe")
        if "result" in saved:
            report["variants"]["reference64"] = {"status": "loaded_from_artifact"}
            save("reference64_restored")
            return saved["result"]
        state.update(saved)

    total_rows = sum(spec[1] for spec in specs)
    output_offset = 0
    group_rows = state["group_metrics"]
    trace_and_lower_seconds = sum(row["trace_and_lower_seconds"] for row in group_rows)
    compile_seconds_total = sum(row["compile_seconds"] for row in group_rows)
    first_execution_seconds_total = sum(row["first_execution_seconds"] for row in group_rows)
    for group_index, indices in enumerate(groups):
        group_size = sum(specs[index][1] for index in indices)
        if group_index < state["completed_groups"]:
            output_offset += group_size
            continue
        def objective(p, x):
            return _reference_piece(
                p, x, masks=masks, specs=specs, indices=indices, config=config,
                decoder=decoder, output_offset=output_offset, total_rows=total_rows,
                microbatch=args.microbatch,
            )
        function = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
        report["variants"]["reference64"] = {
            "status": "trace_and_lower_in_progress", "group_index": group_index,
            "group_count": len(groups), "completed_groups": state["completed_groups"],
            "timing_scope": "modular reference only; not a monolithic-reference comparison",
        }
        save(f"reference64_group_{group_index}_trace_lower")
        start = time.perf_counter()
        lowered = function.lower(shared, payload)
        trace_and_lower_seconds_for_group = time.perf_counter() - start
        report["variants"]["reference64"] = {
            "status": "compile_in_progress", "group_index": group_index,
            "group_count": len(groups), "completed_groups": state["completed_groups"],
            "trace_and_lower_seconds": trace_and_lower_seconds_for_group,
            "timing_scope": "modular reference only; not a monolithic-reference comparison",
        }
        save(f"reference64_group_{group_index}_compile")
        start = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter() - start
        trace_and_lower_seconds += trace_and_lower_seconds_for_group
        compile_seconds_total += compile_seconds
        memory = executable.memory_analysis()
        if memory.temp_size_in_bytes > args.max_compiler_temp_gib * (1024**3):
            report["status"] = "resource_limited"
            report["variants"]["reference64"] = {
                "status": "resource_limited", "group_index": group_index,
                "compiler_temporary_bytes": memory.temp_size_in_bytes,
            }
            save(f"reference64_group_{group_index}_resource_limited")
            return None
        start = time.perf_counter()
        result = executable(shared, payload)
        jax.block_until_ready(result)
        first_execution_seconds = time.perf_counter() - start
        first_execution_seconds_total += first_execution_seconds
        (loss, outputs), (parameter_gradients, input_gradients) = result
        state["outputs"].append(jax.device_get(outputs))
        state["loss"] = state["loss"] + jax.device_get(loss)
        state["parameter_gradients"] = (
            jax.device_get(parameter_gradients) if state["parameter_gradients"] is None
            else _tree_add(state["parameter_gradients"], jax.device_get(parameter_gradients))
        )
        state["input_gradients"] = (
            jax.device_get(input_gradients) if state["input_gradients"] is None
            else _tree_add(state["input_gradients"], jax.device_get(input_gradients))
        )
        state["completed_groups"] = group_index + 1
        group_rows.append({
            "group_index": group_index, "field_start": indices[0],
            "field_stop": indices[-1] + 1, "fields": len(indices),
            "rows": group_size,
            "trace_and_lower_seconds": trace_and_lower_seconds_for_group,
            "compile_seconds": compile_seconds,
            "first_execution_seconds": first_execution_seconds,
            "compiler_temporary_bytes": memory.temp_size_in_bytes,
        })
        state["group_metrics"] = group_rows
        checkpoint = dict(state)
        reference_artifact.parent.mkdir(parents=True, exist_ok=True)
        _write_reference_artifact(reference_artifact, checkpoint)
        report["variants"]["reference64"] = {
            "status": "modular_reference_in_progress",
            "completed_groups": state["completed_groups"],
            "group_count": len(groups), "fields_per_group": fields_per_block,
            "group_metrics": group_rows,
            "trace_and_lower_seconds": trace_and_lower_seconds,
            "compile_seconds": compile_seconds_total,
            "first_execution_seconds": first_execution_seconds_total,
            "timing_scope": "modular reference only; not a monolithic-reference comparison",
            "artifact_bytes": reference_artifact.stat().st_size,
        }
        save(f"reference64_group_{group_index}_complete")
        output_offset += group_size

    outputs = jnp.concatenate(tuple(jnp.asarray(item) for item in state["outputs"]), axis=-1)
    result = (
        (jnp.asarray(state["loss"]), outputs),
        (state["parameter_gradients"], state["input_gradients"]),
    )
    state["result"] = result
    state["status"] = "complete"
    reference_artifact.parent.mkdir(parents=True, exist_ok=True)
    _write_reference_artifact(reference_artifact, state)
    report["variants"]["reference64"] = {
        "status": "modular_reference_complete", "group_count": len(groups),
        "fields_per_group": fields_per_block, "group_metrics": group_rows,
        "trace_and_lower_seconds": trace_and_lower_seconds,
        "compile_seconds": compile_seconds_total,
        "first_execution_seconds": first_execution_seconds_total,
        "timing_scope": "modular reference only; not a monolithic-reference comparison",
        "artifact_bytes": reference_artifact.stat().st_size,
    }
    save("reference64_complete")
    return result


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


def output_discrepancies(actual, expected, specs, *, layout=None, limit=8):
    """Locate failing output coordinates without changing the comparison gate."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim not in (1, 2):
        raise ValueError("output comparison requires equal [rows] or [batch, rows] shapes")
    if actual.shape[-1] != sum(item[1] for item in specs):
        raise ValueError("output rows do not match declared field layout")
    if actual.ndim == 1:
        actual, expected = actual[None, :], expected[None, :]
    boundaries = np.cumsum([item[1] for item in specs])
    failed = np.argwhere(~np.isclose(actual, expected, rtol=1e-5, atol=1e-6))
    rows = []
    for batch_index, output_index in failed[:limit]:
        field_index = int(np.searchsorted(boundaries, output_index, side="right"))
        offset = int(boundaries[field_index - 1]) if field_index else 0
        got, want = float(actual[batch_index, output_index]), float(expected[batch_index, output_index])
        rows.append({
            "batch_index": int(batch_index), "output_index": int(output_index),
            "field_index": field_index, "field_row": int(output_index) - offset,
            "expert_index": int(specs[field_index][0]),
            "field_name": None if layout is None else layout["fields"][field_index]["field_name"],
            "candidate": got, "reference": want, "absolute_difference": abs(got - want),
            "allowed_absolute_difference": 1e-6 + 1e-5 * abs(want),
        })
    return {"failed": len(failed), "reported": len(rows), "coordinates": rows}


def evaluate_output_oracle(module_path, details, shared, payload, masks, config, microbatch):
    """Independent high-precision checks for only the bounded failing coordinates."""
    spec = importlib.util.spec_from_file_location("shared_decoder_roundoff_oracle", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load output oracle: {module_path}")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    rows = []
    for coordinate in details["coordinates"]:
        sample = payload if microbatch == 1 else jax.tree_util.tree_map(
            lambda value: value[coordinate["batch_index"]], payload,
        )
        result = oracle.coordinate_oracle(
            shared, sample, masks[coordinate["field_index"]],
            coordinate["field_index"], coordinate["field_row"], coordinate["expert_index"],
            heads=config.resolved_decoder_attention_heads, epsilon=config.normalization_epsilon,
        )
        value = float(result["output"])
        rows.append({
            **coordinate, "oracle": result,
            "candidate_absolute_error_to_oracle": abs(coordinate["candidate"] - value),
            "reference_absolute_error_to_oracle": abs(coordinate["reference"] - value),
        })
    return rows


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
    modes = tuple(variants)
    reference_artifact = args.reference_artifact or args.output.with_suffix(".reference.pkl")
    identity = {"width": args.width, "microbatch": args.microbatch,
                "layout_sha256": _layout_digest(layout) if layout is not None else None,
                "fixture_seed": 20260925,
                "reference_fields_per_block": getattr(args, "reference_fields_per_block", 8),
                "reference_method": "field_group_sum_global_denominator_v1"}
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
        "reference_artifact": str(reference_artifact),
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

    reference = _build_modular_reference(
        args, decoder, shared, payload, masks, specs, config, identity,
        reference_artifact, report, save,
    )
    if reference is None:
        return report
    if getattr(args, "reference_only", False):
        report["status"] = "passed"
        save("reference_only_complete")
        return report
    for mode in modes:
        def one(parameters, inputs):
            fields, projections = prepare_fields(parameters, inputs, masks, specs=specs)
            if mode.startswith("packed"):
                values = decoder.packed_decode_fields(
                    parameters, fields, experts, projections, config,
                    chunk_size=int(mode.removeprefix("packed")),
                )
            else:
                values = decoder.expert_decode_fields(
                    parameters, fields, experts, projections, config,
                    chunk_size=int(mode.removeprefix("expert")),
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
        save(f"{mode}_trace_lower")
        start = time.perf_counter()
        lowered = function.lower(shared, payload)
        trace_lower_seconds = time.perf_counter() - start
        report["variants"][mode] = {"trace_and_lower_seconds": trace_lower_seconds}
        save(f"{mode}_compile")
        start = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter() - start
        memory = executable.memory_analysis()
        row = {
            "trace_and_lower_seconds": trace_lower_seconds,
            "compile_seconds": compile_seconds,
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
        # Preserve cheap forward evidence even if the numerical gate fails.
        output_artifact = args.output.with_suffix(f".{mode}.outputs.npz")
        np.savez(output_artifact, candidate=np.asarray(result[0][1]),
                 reference=np.asarray(reference[0][1]),
                 candidate_loss=np.asarray(result[0][0]), reference_loss=np.asarray(reference[0][0]))
        row["output_artifact"] = str(output_artifact)
        report["comparison"][mode] = {
            "loss_output": compare(result[0], reference[0]),
            "loss": compare(result[0][0], reference[0][0]),
            "output": compare(result[0][1], reference[0][1]),
            "parameter_payload_gradients": compare(result[1], reference[1]),
        }
        details = output_discrepancies(result[0][1], reference[0][1], specs, layout=layout)
        report.setdefault("output_discrepancies", {})[mode] = details
        oracle_file = getattr(args, "roundoff_oracle_module_file", None)
        if oracle_file is not None and details["coordinates"]:
            save(f"{mode}_independent_output_oracle")
            report.setdefault("output_oracle", {})[mode] = evaluate_output_oracle(
                oracle_file, details, shared, payload, masks, config, args.microbatch,
            )
        if not all(item["passed"] for item in report["comparison"][mode].values()):
            report["status"] = "numerical_rejection"
            save(f"{mode}_numerical_rejection")
            continue
        save(f"{mode}_complete")
    if report["status"] == "running":
        report["status"] = "passed"
    save("complete")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--microbatch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-compiler-temp-gib", type=float, default=4.0)
    parser.add_argument("--reference-fields-per-block", type=int, default=8)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--candidate-module-file", type=Path)
    parser.add_argument("--roundoff-oracle-module-file", type=Path)
    parser.add_argument("--export-layout-only", type=Path)
    parser.add_argument("--layout-metadata-file", type=Path)
    parser.add_argument(
        "--variants", nargs="+",
        choices=tuple(mode for mode in VARIANTS[1:] + ACTUAL_VARIANTS[1:] + EXPERT_VARIANTS),
    )
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
