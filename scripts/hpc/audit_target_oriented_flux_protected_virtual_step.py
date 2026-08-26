"""Audit a bounded direct flux step while protecting state and stock objectives."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_flux_virtual_step import (
    AUDIT_UPDATES,
    DERIVED_FLUX_KEYS,
    HORIZONS,
    OPTIMIZED_FLUX_KEYS,
    classify_flux_protected_virtual_steps,
    load_target_oriented_flux_virtual_step_contract,
)
from scripts.hpc.audit_target_oriented_component_resolved_geometry import (
    _chunked_tree_gram,
    _gradient_report,
    _merge,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    CHECKPOINT_UPDATE,
    _add_gradient,
    _hard_host,
    _load_context,
    _tree_size,
    _zero_gradient,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import PARAMETER_COUNT
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _hard_failure_count,
    _slice,
    _zero_hard,
)
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_flux_protected_virtual_step_audit_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _relative_change(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before):
        return float("nan")
    if before > 1.0e-15:
        return after / before - 1.0
    return 0.0 if abs(after - before) <= 1.0e-15 else float("inf")


def _acceptance_score(record: Mapping[str, Any]) -> float:
    rmse = float(record["normalized_rmse"])
    bias = float(record["normalized_signed_bias"])
    return 0.5 * (rmse * rmse + bias * bias)


def _science_report(statistics: Any, *, flux_basis: Sequence[str]) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        science_window_from_sufficient_statistics,
    )

    result = science_window_from_sufficient_statistics(
        statistics,
        flux_basis=flux_basis,
    )
    return {
        "flux_loss": float(np.asarray(0.25 * (result.flux_mse + result.flux_bias_squared))),
        "stock_loss": float(np.asarray(0.25 * (result.stock_mse + result.stock_bias_squared))),
        "independent_flux_scores": {
            name: float(np.asarray(0.5 * (error.mse + error.bias_squared)))
            for name, error in result.flux_errors.items()
        },
        "independent_stock_scores": {
            name: float(np.asarray(0.5 * (error.mse + error.bias_squared)))
            for name, error in result.stock_errors.items()
        },
    }


def _evaluate_group(
    jax: Any,
    jnp: Any,
    parameters: Any,
    steps: Mapping[str, Any],
    group: Mapping[str, Any],
    target_contract: Any,
    *,
    flux_basis: Sequence[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        add_science_window_sufficient_statistics,
        add_target_oriented_acceptance_statistics,
        finalize_target_oriented_acceptance_statistics,
        merge_error_sufficient_statistics,
        necessary_process_supervision_from_sufficient_statistics,
        one_day_state_anchor_from_sufficient_statistics,
    )

    one_values = [steps["one_forward"](parameters, batch) for batch in group["one_batches"]]
    seven_values = [
        steps["seven_forward"](parameters, batch) for batch in group["seven_batches"]
    ]
    thirty_values = [
        steps["thirty_forward"](parameters, batch) for batch in group["thirty_batches"]
    ]
    seven_acceptance_values = [
        steps["seven_acceptance_forward"](parameters, batch)
        for batch in group["seven_batches"]
    ]
    thirty_acceptance_values = [
        steps["thirty_acceptance_forward"](parameters, batch)
        for batch in group["thirty_batches"]
    ]
    jax.block_until_ready(
        (
            one_values,
            seven_values,
            thirty_values,
            seven_acceptance_values,
            thirty_acceptance_values,
        )
    )

    total_state = merge_error_sufficient_statistics([value[0] for value in one_values])
    total_process = merge_error_sufficient_statistics([value[1] for value in one_values])
    total_seven = _merge(
        add_science_window_sufficient_statistics,
        [value[0] for value in seven_values],
    )
    total_thirty = _merge(
        add_science_window_sufficient_statistics,
        [value[0] for value in thirty_values],
    )
    total_seven_acceptance = _merge(
        add_target_oriented_acceptance_statistics,
        [value[0] for value in seven_acceptance_values],
    )
    total_thirty_acceptance = _merge(
        add_target_oriented_acceptance_statistics,
        [value[0] for value in thirty_acceptance_values],
    )

    hard = _zero_hard(jnp)
    for values in (one_values, seven_values, thirty_values):
        for value in values:
            hard = _add_hard(jax, hard, value[-1])
    jax.block_until_ready(hard)
    hard_host = _hard_host(jax.device_get(hard))

    state = one_day_state_anchor_from_sufficient_statistics(total_state)
    process = necessary_process_supervision_from_sufficient_statistics(
        total_process,
        target_contract,
    )
    one_day = float(
        np.asarray(
            state.total
            + target_contract.loss_coefficients["necessary_process_supervision"]
            * process.total
        )
    )
    science = {
        "7": _science_report(total_seven, flux_basis=flux_basis),
        "30": _science_report(total_thirty, flux_basis=flux_basis),
    }
    acceptance = {
        "7": finalize_target_oriented_acceptance_statistics(
            total_seven_acceptance,
            horizon=7,
            target_contract=target_contract,
        ),
        "30": finalize_target_oriented_acceptance_statistics(
            total_thirty_acceptance,
            horizon=30,
            target_contract=target_contract,
        ),
    }
    return (
        {
            "one_day_state_and_process": one_day,
            "science": science,
            "acceptance": acceptance,
            "hard_counts": hard_host,
            "hard_failure_count": _hard_failure_count(hard_host),
        },
        {
            "state": total_state,
            "process": total_process,
            "seven": total_seven,
            "thirty": total_thirty,
        },
    )


def _accumulate_gradient(
    jax: Any,
    jnp: Any,
    parameters: Any,
    step: Any,
    batches: Sequence[Any],
    total: Any,
) -> tuple[Any, float]:
    gradient = _zero_gradient(jax, jnp, parameters)
    loss = jnp.asarray(0.0, dtype=jnp.float64)
    for batch in batches:
        current_loss, current_gradient = step(parameters, batch, total)
        loss = loss + current_loss
        gradient = _add_gradient(jax, jnp, gradient, current_gradient)
    jax.block_until_ready((loss, gradient))
    return gradient, float(np.asarray(loss))


def _unit_direction(
    jax: Any,
    jnp: Any,
    parameters: Any,
    gradients: Sequence[Any],
    coefficients: Sequence[float],
) -> tuple[Any, Mapping[str, Any]]:
    combined = _zero_gradient(jax, jnp, parameters)
    for coefficient, gradient in zip(coefficients, gradients, strict=True):
        combined = jax.tree_util.tree_map(
            lambda current, value: current
            + jnp.asarray(coefficient, dtype=jnp.float32)
            * jnp.asarray(value, dtype=jnp.float32),
            combined,
            gradient,
        )
    jax.block_until_ready(combined)
    report = _gradient_report(jax, combined)
    norm = float(report["l2_norm"])
    if report["nonfinite_value_count"] != 0 or norm <= 0.0:
        raise FloatingPointError("protected virtual-step direction is invalid")
    unit = jax.tree_util.tree_map(lambda value: value / norm, combined)
    jax.block_until_ready(unit)
    return unit, report


def _parameter_nonfinite_count(jax: Any, jnp: Any, parameters: Any) -> int:
    count = sum(
        (
            jnp.sum(~jnp.isfinite(jnp.asarray(value)), dtype=jnp.int64)
            for value in jax.tree_util.tree_leaves(parameters)
        ),
        jnp.asarray(0, dtype=jnp.int64),
    )
    jax.block_until_ready(count)
    return int(np.asarray(count))


def _step_group_record(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    update: int,
) -> Mapping[str, Any]:
    independent = {}
    derived = {}
    stock = {}
    parent_passing_retained = True
    failing_stock_increases = []
    flux_keys = set((*OPTIMIZED_FLUX_KEYS, *DERIVED_FLUX_KEYS, "heterotrophic_respiration"))
    for horizon in HORIZONS:
        key = str(horizon)
        before_keys = before["acceptance"][key]["keys"]
        after_keys = after["acceptance"][key]["keys"]
        independent[key] = {
            name: _relative_change(
                _acceptance_score(after_keys[name]),
                _acceptance_score(before_keys[name]),
            )
            for name in OPTIMIZED_FLUX_KEYS
        }
        derived[key] = {
            name: _relative_change(
                _acceptance_score(after_keys[name]),
                _acceptance_score(before_keys[name]),
            )
            for name in DERIVED_FLUX_KEYS
        }
        stock_names = tuple(name for name in before_keys if name not in flux_keys)
        stock[key] = {
            name: _relative_change(
                _acceptance_score(after_keys[name]),
                _acceptance_score(before_keys[name]),
            )
            for name in stock_names
        }
        for name, parent in before_keys.items():
            if bool(parent["passed"]):
                parent_passing_retained = parent_passing_retained and bool(
                    after_keys[name]["passed"]
                )
        for name in stock_names:
            if not bool(before_keys[name]["passed"]):
                failing_stock_increases.append(
                    _relative_change(
                        float(after_keys[name]["normalized_rmse"]),
                        float(before_keys[name]["normalized_rmse"]),
                    )
                )

    source_changes = {
        str(horizon): {
            source: _relative_change(
                float(after["science"][str(horizon)][f"{source}_loss"]),
                float(before["science"][str(horizon)][f"{source}_loss"]),
            )
            for source in ("flux", "stock")
        }
        for horizon in HORIZONS
    }
    return {
        "update": int(update),
        "one_day_relative_change": _relative_change(
            float(after["one_day_state_and_process"]),
            float(before["one_day_state_and_process"]),
        ),
        "independent_flux_relative_changes": independent,
        "derived_flux_relative_changes": derived,
        "stock_relative_changes": stock,
        "source_relative_changes": source_changes,
        "parent_passing_acceptance_keys_retained": parent_passing_retained,
        "parent_failing_stock_rmse_relative_increase_max": (
            max(failing_stock_increases) if failing_stock_increases else 0.0
        ),
        "hard_counts": after["hard_counts"],
        "hard_failure_count": int(after["hard_failure_count"]),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
        gradient_direction_diagnostics,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_target_oriented_flux_virtual_step_contract(args.virtual_step_contract)
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

    parameter_count = _tree_size(jax, parameters)
    if parameter_count != PARAMETER_COUNT:
        raise ValueError("protected virtual-step parameter coverage drift")
    memory_before = _memory_report(runtime["devices"][0])

    groups = []
    baseline_records = []
    forward_started = time.perf_counter()
    for update in AUDIT_UPDATES:
        artifact = load_prepared_target_artifact(
            args.prepared_manifest,
            prepared_records[update],
            training_contract_sha256=training_contract.sha256,
            schedule_entry=schedule["entries"][update - 1],
        )
        if artifact.trace_signature != context["trace_signature"]:
            raise ValueError("protected virtual-step prepared trace drift")
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
            raise RuntimeError(f"protected virtual-step baseline hard failure: {baseline}")
        groups.append(group | {"totals": totals})
        baseline_records.append(
            {"update": update, "runtime_year": group["runtime_year"]} | baseline
        )
        print(f"flux_protected_virtual baseline update={update} complete", flush=True)
    forward_seconds = time.perf_counter() - forward_started

    labels = []
    gradients = []
    gradient_losses = []
    gradient_reports = []
    gradient_started = time.perf_counter()
    for group in groups:
        for horizon, step_name, batch_name, total_name in (
            (7, "seven_flux_gradient", "seven_batches", "seven"),
            (30, "thirty_flux_gradient", "thirty_batches", "thirty"),
        ):
            label = f"update_{group['update']}.{horizon}_day_component_resolved_flux"
            gradient, loss = _accumulate_gradient(
                jax,
                jnp,
                parameters,
                steps[step_name],
                group[batch_name],
                group["totals"][total_name],
            )
            report = _gradient_report(jax, gradient)
            if (
                report["value_count"] != PARAMETER_COUNT
                or report["finite_value_count"] != PARAMETER_COUNT
                or report["l2_norm"] <= 0.0
            ):
                raise FloatingPointError(f"invalid protected flux gradient: {label}")
            labels.append(label)
            gradient_losses.append(loss)
            gradient_reports.append(report)
            gradients.append(jax.device_get(gradient))
            del gradient
            print(f"flux_protected_virtual gradient={label} complete", flush=True)

    chunk_gram = jax.jit(
        lambda matrix: jnp.matmul(
            matrix,
            matrix.T,
            precision=jax.lax.Precision.HIGHEST,
        )
    )
    gram = _chunked_tree_gram(jax, gradients, chunk_gram)
    diagnostics = gradient_direction_diagnostics(gram, labels=labels)
    selected_direction = diagnostics["directions"]["minimum_norm_unit"]
    directional_cosines = selected_direction["normalized_directional_derivatives"]
    ideal_minimum_constraint_cosine = min(
        float(value) for value in directional_cosines.values()
    )
    direction_record = {
        "labels": labels,
        "gradient_losses": gradient_losses,
        "gradient_reports": gradient_reports,
        "gradient_norms": diagnostics["gradient_norms"],
        "pairwise_cosines": diagnostics["pairwise_cosines"],
        "minimum_norm_unit": selected_direction,
        "ideal_minimum_constraint_cosine": ideal_minimum_constraint_cosine,
        "minimum_constraint_cosine": ideal_minimum_constraint_cosine,
    }

    virtual_records = []
    virtual_started = time.perf_counter()
    minimum_required = float(contract.payload["direction"]["minimum_constraint_cosine"])
    if ideal_minimum_constraint_cosine >= minimum_required:
        coefficients = [float(selected_direction["weights"][label]) for label in labels]
        unit_direction, realized_direction_report = _unit_direction(
            jax,
            jnp,
            parameters,
            gradients,
            coefficients,
        )
        direction_record["realized_combined_gradient"] = realized_direction_report
        direction_record["geometry_to_realized_norm_relative_error"] = abs(
            float(selected_direction["direction_norm"])
            - float(realized_direction_report["l2_norm"])
        ) / float(selected_direction["direction_norm"])
        realized_host = jax.device_get(unit_direction)
        realized_gram = _chunked_tree_gram(
            jax,
            (*gradients, realized_host),
            chunk_gram,
        )
        realized_norm = float(np.sqrt(max(realized_gram[-1, -1], 0.0)))
        realized_cosines = {
            label: float(
                realized_gram[index, -1]
                / np.sqrt(realized_gram[index, index])
                / realized_norm
            )
            for index, label in enumerate(labels)
        }
        realized_minimum = min(realized_cosines.values())
        direction_record["realized_constraint_cosines"] = realized_cosines
        direction_record["realized_unit_norm"] = realized_norm
        direction_record["minimum_constraint_cosine"] = realized_minimum
        del gradients, realized_host
        if realized_minimum >= minimum_required:
            for step_norm in contract.step_norms:
                proposed = jax.tree_util.tree_map(
                    lambda value, direction: value
                    - jnp.asarray(step_norm, dtype=value.dtype) * direction,
                    parameters,
                    unit_direction,
                )
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
                        _step_group_record(
                            before,
                            after,
                            update=int(group["update"]),
                        )
                    )
                virtual_records.append(
                    {
                        "step_norm": step_norm,
                        "nonfinite_value_count": nonfinite,
                        "groups": group_records,
                    }
                )
                del proposed
                print(
                    f"flux_protected_virtual step_norm={step_norm:.9g} complete",
                    flush=True,
                )
    else:
        del gradients

    classification = classify_flux_protected_virtual_steps(
        direction_record,
        virtual_records,
        contract,
    )
    virtual_seconds = time.perf_counter() - virtual_started
    gradient_seconds = time.perf_counter() - gradient_started
    memory_after = _memory_report(runtime["devices"][0])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "action": classification["next_action"],
        "identity": {
            "git_head": _git_head(),
            "virtual_step_contract_sha256": contract.sha256,
            "geometry_report_sha256": contract.payload["identity"]["geometry_report_sha256"],
            "flux_basis_amendment_sha256": amendment.sha256,
            "stage_checkpoint": str(args.stage_checkpoint.resolve()),
            "stage_checkpoint_sha256": args.stage_checkpoint_sha256,
            "completed_updates": CHECKPOINT_UPDATE,
            "prepared_manifest": str(args.prepared_manifest.resolve()),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "training_contract_sha256": training_contract.sha256,
            "target_contract_sha256": target_contract.sha256,
            "state_anchor_policy_sha256": context["state_anchor_policy"].sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "validation_artifact_values_read": False,
            "sealed_test_read": False,
            "optimizer_invoked": False,
            "optimizer_updates_retained": 0,
            "proposed_parameters_retained": False,
            "checkpoint_written": False,
            "recovery_written": False,
            "audit_updates": list(AUDIT_UPDATES),
            "step_norm_grid": list(contract.step_norms),
        },
        "parameter_count": parameter_count,
        "baseline_groups": baseline_records,
        "direction": direction_record,
        "virtual_steps": virtual_records,
        "classification": classification,
        "timing_seconds": {
            "baseline_forward": forward_seconds,
            "gradient_and_geometry": gradient_seconds - virtual_seconds,
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
    parser.add_argument("--virtual-step-contract", type=Path, required=True)
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
