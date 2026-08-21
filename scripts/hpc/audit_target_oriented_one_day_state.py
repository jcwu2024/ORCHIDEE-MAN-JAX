"""Audit real-v6 one-day state residuals before target-oriented training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED, _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import _retained_tail_trace_signature
from scripts.hpc.run_target_oriented_full_step_preflight import (
    PARAMETER_COUNT,
    _git_head,
    _head,
    _prepare_template,
    _slice,
    _validate_prepared_objective_shapes,
)
from scripts.hpc.target_oriented_training_prepared import (
    load_prepared_target_artifact,
    load_prepared_target_manifest,
)


def _owner(contract: Any, group: str, coordinate: int) -> Mapping[str, Any]:
    if group == "other":
        for leaf in contract.state_leaves:
            if leaf.start is not None and int(leaf.start) <= coordinate < int(leaf.stop):
                local = coordinate - int(leaf.start)
                return {
                    "state_leaf": leaf.key,
                    "leaf_coordinate": [
                        int(value) for value in np.unravel_index(local, tuple(leaf.shape))
                    ],
                }
        raise ValueError("other-state coordinate has no Markov owner")
    if group == "carbon":
        from research.daily_coarse_graining.constrained_daily_replay import (
            CARBON_INVENTORY_FIELDS,
        )

        offset = 0
        for name in CARBON_INVENTORY_FIELDS:
            matches = tuple(
                leaf
                for leaf in contract.state_leaves
                if leaf.component == "slowproc_stomate_previous_step_state"
                and leaf.path == (name,)
                and leaf.start is not None
            )
            if len(matches) != 1:
                raise ValueError(f"carbon inventory owner drift for {name}")
            leaf = matches[0]
            size = int(np.prod(leaf.shape, dtype=np.int64))
            if offset <= coordinate < offset + size:
                return {
                    "state_leaf": leaf.key,
                    "leaf_coordinate": [
                        int(value)
                        for value in np.unravel_index(coordinate - offset, tuple(leaf.shape))
                    ],
                }
            offset += size
        raise ValueError("carbon-state coordinate has no inventory owner")
    return {"group_coordinate": int(coordinate)}


def _state_report(
    predicted: Any,
    expected: Any,
    predicted_defined: Any,
    expected_defined: Any,
    scales: Any,
    eligible: Any,
    contract: Any,
) -> Mapping[str, Any]:
    groups = {}
    for name in predicted._fields:
        left = np.asarray(getattr(predicted, name), dtype=np.float64)
        right = np.asarray(getattr(expected, name), dtype=np.float64)
        left_defined = np.asarray(getattr(predicted_defined, name), dtype=np.bool_)
        right_defined = np.asarray(getattr(expected_defined, name), dtype=np.bool_)
        scale = np.broadcast_to(np.asarray(getattr(scales, name), dtype=np.float64), left.shape)
        loss_eligible = np.broadcast_to(
            np.asarray(getattr(eligible, name), dtype=np.bool_),
            left.shape,
        )
        valid = left_defined & right_defined & loss_eligible & np.isfinite(left) & np.isfinite(right)
        residual = np.zeros_like(left)
        np.subtract(left, right, out=residual, where=valid)
        np.divide(residual, scale, out=residual, where=valid)
        squared = residual * residual
        flat_squared = squared.reshape(-1)
        top_count = min(20, int(np.count_nonzero(valid)))
        top_indices = (
            np.argpartition(flat_squared, -top_count)[-top_count:] if top_count else np.asarray([], dtype=np.int64)
        )
        top_indices = top_indices[np.argsort(flat_squared[top_indices])[::-1]]
        coordinate_width = int(np.prod(left.shape[1:], dtype=np.int64))
        top = []
        for flat_index in top_indices:
            sample = int(flat_index // coordinate_width)
            coordinate = int(flat_index % coordinate_width)
            index = np.unravel_index(int(flat_index), left.shape)
            top.append(
                {
                    "sample": sample,
                    "coordinate": coordinate,
                    **_owner(contract, name, coordinate),
                    "predicted": float(left[index]),
                    "expected": float(right[index]),
                    "absolute_error": float(abs(left[index] - right[index])),
                    "scale": float(scale[index]),
                    "absolute_normalized_error": float(abs(residual[index])),
                }
            )
        mismatch = left_defined != right_defined
        mismatch_by_coordinate = np.sum(mismatch.reshape((left.shape[0], -1)), axis=0)
        mismatch_coordinates = np.flatnonzero(mismatch_by_coordinate)
        mismatch_details = [
            {
                "coordinate": int(coordinate),
                "count": int(mismatch_by_coordinate[coordinate]),
                **_owner(contract, name, int(coordinate)),
            }
            for coordinate in mismatch_coordinates
        ]
        groups[name] = {
            "valid_count": int(np.count_nonzero(valid)),
            "loss_eligible_coordinate_count": int(np.count_nonzero(loss_eligible[0])),
            "mse": float(np.sum(squared, dtype=np.float64) / max(np.count_nonzero(valid), 1)),
            "predicted_defined_expected_undefined": int(np.count_nonzero(left_defined & ~right_defined)),
            "predicted_undefined_expected_defined": int(np.count_nonzero(~left_defined & right_defined)),
            "mismatch_coordinates": mismatch_details,
            "top_absolute_normalized_errors": top,
        }
    return groups


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_attention import (
        initialize_target_oriented_attention_parameters,
        target_oriented_attention_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        load_target_oriented_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        derive_target_oriented_scales,
        load_target_oriented_training_contract,
    )

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    target_contract = load_target_oriented_contract()
    training_contract = load_target_oriented_training_contract(args.training_contract)
    learned_schema = build_mechanistic_learned_schema(runtime["schema"], mechanism)
    layout = build_mechanistic_output_layout(
        adapter=runtime["adapter"],
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        owner_contract=runtime["owner_contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        inputs=runtime["probe_assembled"],
    )
    runtime = dict(runtime) | {
        "architecture_contract": architecture,
        "mechanism_contract": mechanism,
        "learned_schema": learned_schema,
        "target_layout": layout,
    }
    head = _head(runtime, target_contract)
    parameters = initialize_target_oriented_attention_parameters(
        jax.random.PRNGKey(PROBE_SEED),
        schema=learned_schema,
        layout=layout,
        inputs=runtime["probe_assembled"],
        architecture_contract=architecture,
        target_contract=target_contract,
    )
    if target_oriented_attention_parameter_count(parameters) != PARAMETER_COUNT:
        raise ValueError("target-oriented state audit parameter count drift")
    references = runtime["index"].select(spatial_split="train", temporal_split="train")
    schedule = build_target_oriented_training_schedule(references, training_contract, updates=8)
    prepared_identity = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))["identity"]
    prepared = load_prepared_target_manifest(
        args.prepared_manifest,
        training_contract_sha256=training_contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        updates=8,
        source_git_head=str(prepared_identity["source_git_head"]),
    )
    template = _prepare_template(runtime, schedule["entries"][0])
    if template.static_owner.include_modelout_diagnostics:
        template = replace(
            template,
            static_owner=replace(template.static_owner, include_modelout_diagnostics=False),
        )
    artifact = load_prepared_target_artifact(
        args.prepared_manifest,
        prepared["entries"][0],
        training_contract_sha256=training_contract.sha256,
        schedule_entry=schedule["entries"][0],
    )
    scales = derive_target_oriented_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    _validate_prepared_objective_shapes(
        artifact,
        complete_schema=runtime["schema"],
        scales=scales,
    )
    if artifact.trace_signature != _retained_tail_trace_signature(template):
        raise ValueError("prepared and live retained structures differ")
    dynamic = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail_template=template,
    )
    complete_names = tuple(field.name for field in runtime["schema"].fields)
    learned_names = {field.name for field in learned_schema.fields}

    def one(parameters, inputs, day_inputs):
        result = dynamic(parameters, inputs, day_inputs)
        field_nonfinite = jnp.stack(
            tuple(
                jnp.sum(~jnp.isfinite(jnp.asarray(result.prediction.named_outputs[name])), dtype=jnp.int32)
                for name in complete_names
            )
        )
        return result.next_state, result.next_state_defined, field_nonfinite

    four = jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))
    results = []
    for start in range(0, 32, 4):
        microbatch = _slice(artifact.batch.one_day, start, start + 4)
        results.append(four(parameters, microbatch.inputs, microbatch.day_inputs))
    jax.block_until_ready(results)
    predicted = jax.tree_util.tree_map(
        lambda *values: np.concatenate(tuple(np.asarray(value) for value in values), axis=0),
        *(value[0] for value in results),
    )
    predicted_defined = jax.tree_util.tree_map(
        lambda *values: np.concatenate(tuple(np.asarray(value) for value in values), axis=0),
        *(value[1] for value in results),
    )
    field_nonfinite = np.sum(
        np.concatenate(tuple(np.asarray(value[2]) for value in results), axis=0),
        axis=0,
    )
    nonfinite_fields = [
        {
            "field": name,
            "count": int(count),
            "learned": name in learned_names,
        }
        for name, count in zip(complete_names, field_nonfinite, strict=True)
        if count
    ]
    report = {
        "schema_version": "target_oriented_one_day_state_audit_v1",
        "identity": {
            "git_head": _git_head(),
            "prepared_source_git_head": str(prepared_identity["source_git_head"]),
            "runtime_year": int(schedule["entries"][0]["runtime_year"]),
        },
        "policy": {
            "training_updates": 0,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
        },
        "state_groups": _state_report(
            predicted,
            artifact.batch.one_day.expected_next_state,
            predicted_defined,
            artifact.batch.one_day.expected_next_state_defined,
            scales.state,
            scales.state_loss_eligible,
            runtime["contract"],
        ),
        "nonfinite_complete_output_fields": nonfinite_fields,
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
