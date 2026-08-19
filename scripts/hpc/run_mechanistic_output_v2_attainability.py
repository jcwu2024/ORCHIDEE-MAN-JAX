"""Audit whether each v2 output coordinate can attain its Teacher target."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_strong_baseline_one_point_qualification import (
    BUDGET_ATOL,
    STOCK_ATOL,
    _git_head,
    _prepare_point_year,
    _sha256_file,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

FIELD_MSE_GATE = 1.0e-8
EXACT_DIAGNOSTIC_MSE_GATE = 1.0e-8
EXACT_ATOL = 1.0e-10
EXACT_RTOL = 1.0e-12
DOC_FIXED_POINT_GATE = 1.0e-10
HARD_NAMES = (
    "prediction_nonfinite_defined_values",
    "fast_state_nonfinite_defined_values",
    "water_budget_failures",
    "carbon_budget_failures",
    "water_negative_inventory_values",
    "carbon_negative_inventory_values",
    "canopy_status_mismatches",
    "rveget_status_mismatches",
    "doc_fixed_point_nonconverged_days",
)


class OracleDayBatch(NamedTuple):
    total: Any
    expert_mse: Any
    expert_count: Any
    label_mse: Any
    label_count: Any
    field_mse: Any
    field_count: Any
    defined_bce: Any
    defined_count: Any
    exact_mse: Any
    exact_count: Any
    exact_maximum_absolute_error: Any
    exact_maximum_tolerance_ratio: Any
    exact_tolerance_failure_count: Any
    hard_counts: Any
    doc_coordinate_residual: Any


def _weighted_records(
    names: tuple[str, ...],
    squared: np.ndarray,
    counts: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result = {}
    for index, name in enumerate(names):
        count = float(counts[index])
        result[name] = {
            "normalized_mse": float(squared[index] / count) if count > 0.0 else None,
            "defined_value_count": count,
        }
    return result


def _attainability_gate_passed(
    *,
    maximum_field_mse: float,
    maximum_exact_mse: float,
    maximum_doc_residual: float,
    hard_counts: Mapping[str, int],
) -> bool:
    return bool(
        maximum_field_mse <= FIELD_MSE_GATE
        and maximum_exact_mse <= EXACT_DIAGNOSTIC_MSE_GATE
        and maximum_doc_residual <= DOC_FIXED_POINT_GATE
        and all(value == 0 for value in hard_counts.values())
    )


def _build_attainability_head(runtime: Mapping[str, Any]) -> Any:
    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        MechanisticStrongBaselineProcessHead,
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
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
    return MechanisticStrongBaselineProcessHead(
        contract=runtime["contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        input_assembler=runtime["adapter"].input_assembler,
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        layout=layout,
        input_normalization=runtime["input_normalization"],
        output_normalization=runtime["output_normalization"],
        forcing_layout=runtime["forcing_layout"],
        forcing_indices=runtime["forcing_indices"],
        message_gate_scale=0.0,
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or args.doc_fixed_point_iterations < 1:
        raise ValueError("attainability batch size and DOC iteration count must be positive")
    started = time.perf_counter()
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.mechanistic_output_attainability import (
        materialize_attainability_oracle,
        target_canopy_endpoint,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_output_loss import (
        masked_mechanistic_output_loss,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_output_reconstruction import (
        CANOPY_STATUS_LOGITS_AUX,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_qualification import (
        padded_epoch_batches,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
        RVEGET_LOGIT_KEY,
        training_representation_outputs,
    )

    head = _build_attainability_head(runtime)
    architecture = head.architecture_contract
    mechanism = head.mechanism_contract
    learned_schema = head.learned_schema
    dataset = _prepare_point_year(runtime)
    sample_count = int(dataset["sample_count"])
    inputs_all = jax.device_put(dataset["assembled_inputs"])
    targets_all = jax.device_put(dataset["targets"])

    expert_names = tuple(architecture.expert_names)
    active_labels = {field.label_index for field in learned_schema.fields}
    label_names = tuple(
        name for index, name in enumerate(learned_schema.labels) if index in active_labels
    )
    canopy_names = set(mechanism.canopy_transfer_fields)
    ordinary_field_names = tuple(
        field.name
        for field in learned_schema.fields
        if field.name not in canopy_names and field.value_representation != "defined_logit"
    )
    field_names = ordinary_field_names + (
        mechanism.canopy_endpoint_field,
        *(mechanism.canopy_status_fields[status] for status in ("storage", "zero", "debt")),
    )
    defined_names = (RVEGET_LOGIT_KEY,)
    exact_names = tuple(mechanism.exact_fields)

    def one_day(inputs: Any, targets: Any) -> OracleDayBatch:
        oracle = materialize_attainability_oracle(
            head,
            targets,
            inputs,
            doc_fixed_point_iterations=args.doc_fixed_point_iterations,
        )
        loss = masked_mechanistic_output_loss(
            oracle.prediction.named_outputs,
            targets,
            inputs=inputs,
            complete_schema=runtime["schema"],
            learned_schema=learned_schema,
            scales=runtime["loss_scales"],
            architecture_contract=architecture,
            mechanism_contract=mechanism,
            auxiliary_outputs=oracle.prediction.auxiliary_outputs,
        )

        prediction_nonfinite = jnp.asarray(0, dtype=jnp.int64)
        vegetation = jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)[:, 0] < 0.5
        for name, value in oracle.prediction.named_outputs.items():
            defined = jnp.asarray(targets[name].defined, dtype=jnp.bool_)
            if name.endswith("roughheight_pft"):
                defined = defined & vegetation[None, :]
            prediction_nonfinite += jnp.sum(
                defined & ~jnp.isfinite(jnp.asarray(value)),
                dtype=jnp.int64,
            )
        state_nonfinite = sum(
            (
                jnp.sum(
                    jnp.asarray(defined, dtype=jnp.bool_) & ~jnp.isfinite(jnp.asarray(value)),
                    dtype=jnp.int64,
                )
                for value, defined in zip(
                    oracle.fast_update.state,
                    oracle.fast_update.state_defined,
                    strict=True,
                )
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        water_budget = (
            ~jnp.isfinite(oracle.fast_update.water.budget_residual)
            | (jnp.abs(oracle.fast_update.water.budget_residual) > BUDGET_ATOL)
        ).astype(jnp.int64)
        carbon_budget = (
            ~jnp.isfinite(oracle.fast_update.carbon.budget_residual)
            | (jnp.abs(oracle.fast_update.carbon.budget_residual) > BUDGET_ATOL)
        ).astype(jnp.int64)
        water_negative = jnp.sum(
            oracle.fast_update.water.defined
            & (oracle.fast_update.water.inventory < -STOCK_ATOL),
            dtype=jnp.int64,
        )
        carbon_negative = jnp.sum(
            oracle.fast_update.carbon.defined
            & (oracle.fast_update.carbon.inventory < -STOCK_ATOL),
            dtype=jnp.int64,
        )

        endpoint = target_canopy_endpoint(targets, inputs)
        canopy_target = jnp.where(
            endpoint > 1.0e-10,
            0,
            jnp.where(endpoint < -1.0e-10, 2, 1),
        )
        canopy_logits = oracle.prediction.auxiliary_outputs[CANOPY_STATUS_LOGITS_AUX]
        canopy_eligible = (
            jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
            & (jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)[:, 0] < 0.5)
        )[None, :]
        canopy_mismatch = jnp.sum(
            canopy_eligible & (jnp.argmax(canopy_logits, axis=-1) != canopy_target),
            dtype=jnp.int64,
        )
        rveget_target = targets[RVEGET_LOGIT_KEY]
        rveget_logits = oracle.prediction.named_outputs[RVEGET_LOGIT_KEY]
        rveget_mismatch = jnp.sum(
            jnp.asarray(rveget_target.defined, dtype=jnp.bool_)
            & ((rveget_logits >= 0.0) != (jnp.asarray(rveget_target.value) >= 0.5)),
            dtype=jnp.int64,
        )
        doc_nonconverged = (
            ~jnp.isfinite(oracle.doc_coordinate_fixed_point_residual)
            | (oracle.doc_coordinate_fixed_point_residual > DOC_FIXED_POINT_GATE)
        ).astype(jnp.int64)

        represented = training_representation_outputs(
            oracle.prediction.named_outputs,
            inputs,
            runtime["schema"],
        )
        exact_maximum_absolute_error = []
        exact_maximum_tolerance_ratio = []
        exact_tolerance_failure_count = []
        for name in exact_names:
            actual = jnp.asarray(represented[name], dtype=jnp.float64)
            wanted = jnp.asarray(targets[name].value, dtype=jnp.float64)
            selected = (
                jnp.asarray(targets[name].defined, dtype=jnp.bool_)
                & jnp.isfinite(actual)
                & jnp.isfinite(wanted)
            )
            difference = jnp.where(selected, jnp.abs(actual - wanted), 0.0)
            tolerance = EXACT_ATOL + EXACT_RTOL * jnp.abs(wanted)
            exact_maximum_absolute_error.append(jnp.max(difference))
            exact_maximum_tolerance_ratio.append(
                jnp.max(jnp.where(selected, difference / tolerance, 0.0))
            )
            exact_tolerance_failure_count.append(
                jnp.sum(selected & (difference > tolerance), dtype=jnp.int64)
            )

        return OracleDayBatch(
            total=loss.total,
            expert_mse=jnp.stack(tuple(loss.expert_mse[name] for name in expert_names)),
            expert_count=jnp.stack(tuple(loss.expert_label_count[name] for name in expert_names)),
            label_mse=jnp.stack(tuple(loss.label_mse[name] for name in label_names)),
            label_count=jnp.stack(tuple(loss.label_count[name] for name in label_names)),
            field_mse=jnp.stack(tuple(loss.field_mse[name] for name in field_names)),
            field_count=jnp.stack(tuple(loss.field_count[name] for name in field_names)),
            defined_bce=jnp.stack(tuple(loss.defined_bce[name] for name in defined_names)),
            defined_count=jnp.stack(tuple(loss.defined_count[name] for name in defined_names)),
            exact_mse=jnp.stack(tuple(loss.exact_diagnostic_mse[name] for name in exact_names)),
            exact_count=jnp.stack(tuple(loss.exact_diagnostic_count[name] for name in exact_names)),
            exact_maximum_absolute_error=jnp.stack(tuple(exact_maximum_absolute_error)),
            exact_maximum_tolerance_ratio=jnp.stack(tuple(exact_maximum_tolerance_ratio)),
            exact_tolerance_failure_count=jnp.stack(tuple(exact_tolerance_failure_count)),
            hard_counts=jnp.stack(
                (
                    prediction_nonfinite,
                    state_nonfinite,
                    water_budget,
                    carbon_budget,
                    water_negative,
                    carbon_negative,
                    canopy_mismatch,
                    rveget_mismatch,
                    doc_nonconverged,
                )
            ),
            doc_coordinate_residual=oracle.doc_coordinate_fixed_point_residual,
        )

    @jax.jit
    def batch(indices: Any) -> OracleDayBatch:
        inputs = jax.tree_util.tree_map(lambda value: value[indices], inputs_all)
        targets = jax.tree_util.tree_map(lambda value: value[indices], targets_all)
        return jax.vmap(one_day)(inputs, targets)

    total_sum = 0.0
    total_days = 0.0
    weighted = {
        "experts": [np.zeros(len(expert_names)), np.zeros(len(expert_names))],
        "labels": [np.zeros(len(label_names)), np.zeros(len(label_names))],
        "fields": [np.zeros(len(field_names)), np.zeros(len(field_names))],
        "defined": [np.zeros(len(defined_names)), np.zeros(len(defined_names))],
        "exact": [np.zeros(len(exact_names)), np.zeros(len(exact_names))],
    }
    hard = np.zeros(len(HARD_NAMES), dtype=np.int64)
    exact_maximum_absolute_error = np.zeros(len(exact_names), dtype=np.float64)
    exact_maximum_tolerance_ratio = np.zeros(len(exact_names), dtype=np.float64)
    exact_tolerance_failure_count = np.zeros(len(exact_names), dtype=np.int64)
    maximum_doc_residual = 0.0
    for indices, weights in padded_epoch_batches(
        sample_count,
        args.batch_size,
        seed=0,
        epoch=1,
    ):
        result = batch(jnp.asarray(indices, dtype=jnp.int32))
        jax.block_until_ready(result)
        admitted = np.asarray(weights, dtype=np.float64)
        total_sum += float(np.sum(np.asarray(result.total) * admitted))
        total_days += float(np.sum(admitted))
        for key, mse, count in (
            ("experts", result.expert_mse, result.expert_count),
            ("labels", result.label_mse, result.label_count),
            ("fields", result.field_mse, result.field_count),
            ("defined", result.defined_bce, result.defined_count),
            ("exact", result.exact_mse, result.exact_count),
        ):
            counts = np.asarray(count, dtype=np.float64) * admitted[:, None]
            weighted[key][0] += np.sum(np.asarray(mse, dtype=np.float64) * counts, axis=0)
            weighted[key][1] += np.sum(counts, axis=0)
        hard += np.sum(
            np.asarray(result.hard_counts, dtype=np.int64) * admitted[:, None].astype(np.int64),
            axis=0,
        )
        exact_maximum_absolute_error = np.maximum(
            exact_maximum_absolute_error,
            np.max(
                np.where(
                    admitted[:, None] > 0.0,
                    np.asarray(result.exact_maximum_absolute_error),
                    0.0,
                ),
                axis=0,
            ),
        )
        exact_maximum_tolerance_ratio = np.maximum(
            exact_maximum_tolerance_ratio,
            np.max(
                np.where(
                    admitted[:, None] > 0.0,
                    np.asarray(result.exact_maximum_tolerance_ratio),
                    0.0,
                ),
                axis=0,
            ),
        )
        exact_tolerance_failure_count += np.sum(
            np.asarray(result.exact_tolerance_failure_count, dtype=np.int64)
            * admitted[:, None].astype(np.int64),
            axis=0,
        )
        maximum_doc_residual = max(
            maximum_doc_residual,
            float(np.max(np.where(admitted > 0.0, np.asarray(result.doc_coordinate_residual), 0.0))),
        )

    records = {
        "experts": _weighted_records(expert_names, *weighted["experts"]),
        "labels": _weighted_records(label_names, *weighted["labels"]),
        "fields": _weighted_records(field_names, *weighted["fields"]),
        "defined_status": _weighted_records(defined_names, *weighted["defined"]),
        "exact_diagnostics": _weighted_records(exact_names, *weighted["exact"]),
    }
    for index, name in enumerate(exact_names):
        records["exact_diagnostics"][name].update(
            maximum_absolute_error=float(exact_maximum_absolute_error[index]),
            maximum_tolerance_ratio=float(exact_maximum_tolerance_ratio[index]),
            tolerance_failure_count=int(exact_tolerance_failure_count[index]),
        )
    active_field_mse = [
        value["normalized_mse"]
        for value in records["fields"].values()
        if value["normalized_mse"] is not None
    ]
    active_defined_mse = [
        value["normalized_mse"]
        for value in records["defined_status"].values()
        if value["normalized_mse"] is not None
    ]
    maximum_field_mse = max(active_field_mse + active_defined_mse, default=0.0)
    maximum_exact_mse = max(
        (
            value["normalized_mse"]
            for value in records["exact_diagnostics"].values()
            if value["normalized_mse"] is not None
        ),
        default=0.0,
    )
    exact_failure_count = int(np.sum(exact_tolerance_failure_count))
    hard_counts = {name: int(hard[index]) for index, name in enumerate(HARD_NAMES)}
    passed = _attainability_gate_passed(
        maximum_field_mse=maximum_field_mse,
        maximum_exact_mse=maximum_exact_mse,
        maximum_doc_residual=maximum_doc_residual,
        hard_counts=hard_counts,
    )
    report = {
        "schema_version": "mechanistic_output_v2_attainability_audit_v2",
        "status": "passed" if passed else "completed_failed_gate",
        "execution_status": "completed",
        "identity": {
            "git_head": _git_head(),
            "dataset_manifest_sha256": _sha256_file(args.dataset_manifest),
            "mechanism_contract_sha256": _sha256_file(mechanism.source_path),
        },
        "data": {
            "landpoint_id": "001.0-071.0",
            "year": 1961,
            "valid_day_count": sample_count,
            "day_index_minimum": int(np.min(dataset["day_indices"])),
            "day_index_maximum": int(np.max(dataset["day_indices"])),
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
        },
        "oracle": {
            "neural_body_executed": False,
            "network_parameter_count": 0,
            "optimizer_updates": 0,
            "doc_fixed_point_iterations": args.doc_fixed_point_iterations,
            "maximum_doc_coordinate_fixed_point_residual": maximum_doc_residual,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
        },
        "gates": {
            "maximum_learned_field_normalized_mse": FIELD_MSE_GATE,
            "maximum_exact_diagnostic_normalized_mse": EXACT_DIAGNOSTIC_MSE_GATE,
            "maximum_doc_coordinate_fixed_point_residual": DOC_FIXED_POINT_GATE,
            "required_hard_failure_count": 0,
        },
        "diagnostics": {
            "teacher_physical_exact_atol": EXACT_ATOL,
            "teacher_physical_exact_rtol": EXACT_RTOL,
            "teacher_physical_exact_tolerance_is_gate": False,
            "interpretation": (
                "derived exact fields are deterministic conditional on materialized sources; "
                "teacher-coordinate error is gated in normalized output space"
            ),
        },
        "result": {
            "mean_complete_loss": total_sum / total_days,
            "maximum_learned_field_normalized_mse": maximum_field_mse,
            "maximum_exact_diagnostic_normalized_mse": maximum_exact_mse,
            "maximum_exact_diagnostic_tolerance_ratio": float(
                np.max(exact_maximum_tolerance_ratio)
            ),
            "exact_diagnostic_tolerance_failure_count": exact_failure_count,
            "hard_counts": hard_counts,
            **records,
        },
        "device": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in runtime["devices"]],
            "memory": [_memory_report(device) for device in runtime["devices"]],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "isolate_litter_and_surface_energy_optimization"
            if passed
            else "repair_unattainable_output_coordinates"
        ),
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--doc-fixed-point-iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["execution_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
