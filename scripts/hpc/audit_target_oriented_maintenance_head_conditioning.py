"""Audit exact finite conditioning of the independent maintenance final head."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from research.daily_coarse_graining.daily_operator.target_oriented_maintenance_head_conditioning import (
    AUDIT_UPDATES,
    HEAD_PARAMETER_COUNT,
    calibrated_step_records,
    classify_maintenance_head_steps,
    load_target_oriented_maintenance_head_conditioning_contract,
    maintenance_head,
    replace_maintenance_head,
    resolve_maintenance_head_index,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    normalized_error_from_sufficient_statistics,
)
from scripts.hpc.audit_target_oriented_component_resolved_geometry import (
    _gradient_report,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    CHECKPOINT_UPDATE,
    _add_gradient,
    _load_context,
    _tree_size,
    _zero_gradient,
)
from scripts.hpc.audit_target_oriented_flux_protected_virtual_step import (
    _evaluate_group,
    _parameter_nonfinite_count,
    _step_group_record,
    _unit_direction,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import PARAMETER_COUNT
from scripts.hpc.run_target_oriented_full_step_preflight import _slice
from scripts.hpc.target_oriented_training_prepared import (
    load_prepared_target_artifact,
)

SCHEMA_VERSION = "target_oriented_maintenance_head_conditioning_audit_report_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _maintenance_score(statistics: Any) -> float:
    error = normalized_error_from_sufficient_statistics(
        statistics.fluxes["maintenance_respiration"]
    )
    return float(np.asarray(0.5 * (error.mse + error.bias_squared)))


def _accumulate_head_gradient(
    jax: Any,
    jnp: Any,
    head: Any,
    parameters: Any,
    step: Any,
    batches: Sequence[Any],
    total: Any,
) -> tuple[Any, float]:
    gradient = _zero_gradient(jax, jnp, head)
    loss = jnp.asarray(0.0, dtype=jnp.float64)
    for batch in batches:
        current_loss, current_gradient = step(head, parameters, batch, total)
        loss = loss + current_loss
        gradient = _add_gradient(jax, jnp, gradient, current_gradient)
    jax.block_until_ready((loss, gradient))
    return gradient, float(np.asarray(loss))


def _head_gram(jax: Any, jnp: Any, gradients: Sequence[Any]) -> np.ndarray:
    from jax.flatten_util import ravel_pytree

    vectors = [ravel_pytree(value)[0].astype(jnp.float64) for value in gradients]
    matrix = jnp.stack(vectors, axis=0)
    gram = jnp.matmul(matrix, matrix.T, precision=jax.lax.Precision.HIGHEST)
    jax.block_until_ready(gram)
    result = np.asarray(jax.device_get(gram), dtype=np.float64)
    if result.shape != (16, 16) or np.any(~np.isfinite(result)):
        raise FloatingPointError("maintenance-head Gram matrix is invalid")
    return result


def _realized_direction_geometry(
    jax: Any,
    jnp: Any,
    gradients: Sequence[Any],
    direction: Any,
) -> tuple[Mapping[str, float], tuple[float, ...]]:
    from jax.flatten_util import ravel_pytree

    vectors = [ravel_pytree(value)[0].astype(jnp.float64) for value in gradients]
    direction_vector = ravel_pytree(direction)[0].astype(jnp.float64)
    matrix = jnp.stack((*vectors, direction_vector), axis=0)
    gram = jnp.matmul(matrix, matrix.T, precision=jax.lax.Precision.HIGHEST)
    jax.block_until_ready(gram)
    host = np.asarray(jax.device_get(gram), dtype=np.float64)
    direction_norm = float(np.sqrt(max(host[-1, -1], 0.0)))
    gradient_norms = np.sqrt(np.maximum(np.diag(host[:-1, :-1]), 0.0))
    derivatives = host[:-1, -1] / max(direction_norm, 1.0e-30)
    cosines = derivatives / np.maximum(gradient_norms, 1.0e-30)
    if (
        not np.isfinite(direction_norm)
        or direction_norm <= 0.0
        or np.any(~np.isfinite(cosines))
        or np.any(~np.isfinite(derivatives))
    ):
        raise FloatingPointError("maintenance-head realized direction is invalid")
    return (
        {
            "unit_norm": direction_norm,
            "minimum_constraint_cosine": float(np.min(cosines)),
            "median_constraint_cosine": float(np.median(cosines)),
            "maximum_constraint_cosine": float(np.max(cosines)),
        },
        tuple(float(value) for value in derivatives),
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
        gradient_direction_diagnostics,
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_target_oriented_maintenance_head_conditioning_contract(
        args.maintenance_head_conditioning_contract
    )
    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = _load_context(
        args,
        science_flux_basis=amendment.flux_basis,
        verify_validation_artifacts=False,
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    parameters = context["parameters"]
    steps = context["steps"]
    schedule = context["schedule"]
    prepared_records = context["prepared_records"]
    training_contract = context["training_contract"]
    target_contract = context["target_contract"]
    if _tree_size(jax, parameters) != PARAMETER_COUNT:
        raise ValueError("maintenance-head parent parameter coverage drift")
    head_index = resolve_maintenance_head_index(runtime["learned_schema"])
    head = maintenance_head(parameters, head_index)
    if _tree_size(jax, head) != HEAD_PARAMETER_COUNT:
        raise ValueError("maintenance-head differentiated parameter coverage drift")
    memory_before = _memory_report(runtime["devices"][0])

    groups = []
    baseline_records = []
    parent_scores = []
    forward_started = time.perf_counter()
    for update in AUDIT_UPDATES:
        artifact = load_prepared_target_artifact(
            args.prepared_manifest,
            prepared_records[update],
            training_contract_sha256=training_contract.sha256,
            schedule_entry=schedule["entries"][update - 1],
        )
        if artifact.trace_signature != context["trace_signature"]:
            raise ValueError("maintenance-head prepared trace drift")
        group = {
            "update": update,
            "runtime_year": int(schedule["entries"][update - 1]["runtime_year"]),
            "one_batches": [
                _slice(artifact.batch.one_day, index, index + 4)
                for index in range(0, 32, 4)
            ],
            "seven_batches": [
                _slice(artifact.batch.seven_day, index, index + 1)
                for index in range(8)
            ],
            "thirty_batches": [
                _slice(artifact.batch.thirty_day, index, index + 1)
                for index in range(4)
            ],
        }
        baseline, totals = _evaluate_group(
            jax,
            jnp,
            parameters,
            steps,
            group,
            target_contract,
            flux_basis=amendment.flux_basis,
        )
        if baseline["hard_failure_count"] != 0:
            raise RuntimeError(f"maintenance-head baseline hard failure: {baseline}")
        groups.append(group | {"totals": totals})
        baseline_records.append(
            {"update": update, "runtime_year": group["runtime_year"]} | baseline
        )
        parent_scores.extend(
            (
                _maintenance_score(totals["seven"]),
                _maintenance_score(totals["thirty"]),
            )
        )
        print(f"maintenance_head baseline update={update} complete", flush=True)
    forward_seconds = time.perf_counter() - forward_started

    labels = []
    gradients = []
    gradient_losses = []
    gradient_reports = []
    gradient_started = time.perf_counter()
    for group in groups:
        for horizon, step_name, batch_name, total_name in (
            (7, "seven_maintenance_head_gradient", "seven_batches", "seven"),
            (30, "thirty_maintenance_head_gradient", "thirty_batches", "thirty"),
        ):
            label = f"update_{group['update']}.{horizon}_day_maintenance_head"
            gradient, loss = _accumulate_head_gradient(
                jax,
                jnp,
                head,
                parameters,
                steps[step_name],
                group[batch_name],
                group["totals"][total_name],
            )
            report = _gradient_report(jax, gradient)
            if (
                report["value_count"] != HEAD_PARAMETER_COUNT
                or report["finite_value_count"] != HEAD_PARAMETER_COUNT
                or report["l2_norm"] <= 0.0
            ):
                raise FloatingPointError(f"invalid maintenance-head gradient: {label}")
            labels.append(label)
            gradients.append(jax.device_get(gradient))
            gradient_losses.append(loss)
            gradient_reports.append(report)
            print(f"maintenance_head gradient={label} complete", flush=True)

    gram = _head_gram(jax, jnp, gradients)
    diagnostics = gradient_direction_diagnostics(gram, labels=labels)
    selected = diagnostics["directions"]["minimum_norm_unit"]
    coefficients = [float(selected["weights"][label]) for label in labels]
    unit_direction, combined_report = _unit_direction(
        jax,
        jnp,
        head,
        gradients,
        coefficients,
    )
    realized, directional_derivatives = _realized_direction_geometry(
        jax,
        jnp,
        gradients,
        unit_direction,
    )
    calibration = calibrated_step_records(
        parent_scores,
        directional_derivatives,
        contract,
    )
    direction_record = {
        "labels": labels,
        "parent_scores": parent_scores,
        "linearized_losses": gradient_losses,
        "gradient_reports": gradient_reports,
        "pairwise_cosines": diagnostics["pairwise_cosines"],
        "minimum_norm_unit": selected,
        "combined_gradient": combined_report,
        "realized": realized,
        "directional_derivatives": list(directional_derivatives),
        "minimum_constraint_cosine": realized["minimum_constraint_cosine"],
    }
    gradient_seconds = time.perf_counter() - gradient_started

    virtual_records = []
    virtual_started = time.perf_counter()
    minimum_cosine = float(contract.payload["direction"]["minimum_constraint_cosine"])
    if realized["minimum_constraint_cosine"] >= minimum_cosine:
        for declared in calibration:
            if not bool(declared["within_parameter_bound"]):
                continue
            proposed_head = jax.tree_util.tree_map(
                lambda value, direction: value
                - jnp.asarray(declared["step_norm"], dtype=value.dtype) * direction,
                head,
                unit_direction,
            )
            proposed = replace_maintenance_head(parameters, head_index, proposed_head)
            nonfinite = _parameter_nonfinite_count(jax, jnp, proposed)
            group_records = []
            for group, before in zip(groups, baseline_records, strict=True):
                after, _totals = _evaluate_group(
                    jax,
                    jnp,
                    proposed,
                    steps,
                    group,
                    target_contract,
                    flux_basis=amendment.flux_basis,
                )
                group_records.append(
                    _step_group_record(before, after, update=int(group["update"]))
                )
            virtual_records.append(
                dict(declared)
                | {
                    "nonfinite_value_count": nonfinite,
                    "groups": group_records,
                }
            )
            print(
                "maintenance_head "
                f"predicted_fraction={declared['predicted_fraction']:.9g} complete",
                flush=True,
            )
            del proposed, proposed_head
    classification = classify_maintenance_head_steps(
        direction_record,
        virtual_records,
        contract,
    )
    virtual_seconds = time.perf_counter() - virtual_started
    memory_after = _memory_report(runtime["devices"][0])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "action": classification["action"],
        "identity": {
            "git_head": _git_head(),
            "maintenance_head_conditioning_contract_sha256": contract.sha256,
            "maintenance_state_recurrence_report_sha256": contract.payload["identity"][
                "maintenance_state_recurrence_report_sha256"
            ],
            "flux_basis_amendment_sha256": amendment.sha256,
            "stage_checkpoint_sha256": args.stage_checkpoint_sha256,
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "training_contract_sha256": training_contract.sha256,
            "target_contract_sha256": target_contract.sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_invoked": False,
            "optimizer_updates_retained": 0,
            "proposed_parameters_retained": False,
            "checkpoint_written": False,
            "recovery_written": False,
            "input_mask_changed": False,
            "source_routing_changed": False,
            "architecture_changed": False,
        },
        "parameter_count": PARAMETER_COUNT,
        "differentiated_head_parameter_count": HEAD_PARAMETER_COUNT,
        "maintenance_head_field_index": head_index,
        "baseline_groups": baseline_records,
        "direction": direction_record,
        "step_calibration": list(calibration),
        "virtual_steps": virtual_records,
        "classification": classification,
        "timing_seconds": {
            "baseline_forward": forward_seconds,
            "head_gradient_and_geometry": gradient_seconds,
            "virtual_steps": virtual_seconds,
            "total": time.perf_counter() - started,
        },
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    _atomic_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--flux-basis-amendment", type=Path, required=True)
    parser.add_argument(
        "--maintenance-head-conditioning-contract", type=Path, required=True
    )
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=CHECKPOINT_UPDATE)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
