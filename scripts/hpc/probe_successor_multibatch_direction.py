"""Audit one accumulated-gradient direction across ten mixed batches."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    GRADIENT_FEASIBILITY_LABELS,
    classify_direct_trust_region,
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from scripts.hpc.probe_successor_batch_diversity import (
    POINTS_PER_MIXED_UPDATE,
    _join_context_anchors,
    _slice_context_batch,
    _stack_contexts,
)
from scripts.hpc.probe_successor_direct_trust_region import (
    _make_combined_direction,
    _make_direct_virtual_step,
)
from scripts.hpc.probe_successor_gradient_feasibility import (
    PARENT_UPDATE,
    _load_contract,
    _make_loss_evaluator,
    _make_measurement,
    _make_objective,
    _sha256_file,
)
from scripts.hpc.probe_successor_realized_step import _relative_changes
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import (
    _device_memory_report,
    _hard_counts_host,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.run_successor_staged_training import _training_identity
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)

AUDIT_GROUPS = 10
SCALARIZATION_SMOKE_UPDATES = 5


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_gradient_accumulator(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def accumulate(current: Any, gradient: Any, scale: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda total, value: total + scale * jnp.asarray(value),
            current,
            gradient,
        )

    return accumulate


def _make_nonfinite_counter(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def count(values: Any) -> Any:
        return sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int64)
                for leaf in jax.tree_util.tree_leaves(values)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )

    return count


def _make_flattened_unit_gradient(jax: Any) -> Any:
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    @jax.jit
    def flatten(gradient: Any) -> Any:
        values, _unravel = ravel_pytree(gradient)
        norm = jnp.linalg.norm(values)
        nonfinite = jnp.sum(~jnp.isfinite(values), dtype=jnp.int64)
        return values / jnp.maximum(norm, 1.0e-30), norm, nonfinite

    return flatten


def _make_flat_gram(jax: Any) -> Any:
    @jax.jit
    def measure(unit_gradients: Any) -> Any:
        return unit_gradients @ unit_gradients.T

    return measure


def _make_flat_direction(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def combine(unit_gradients: Any, weights: Any) -> Any:
        direction = weights @ unit_gradients
        norm = jnp.linalg.norm(direction)
        nonfinite = jnp.sum(~jnp.isfinite(direction), dtype=jnp.int64)
        return direction, norm, nonfinite

    return combine


def _make_loss_directional_evaluator(
    runtime: Mapping[str, Any],
    adapter_template: Any,
    **objective_arguments: Any,
) -> Any:
    """Evaluate all four loss directional derivatives with one group JVP."""

    jax, objective = _make_objective(
        runtime,
        adapter_template,
        **objective_arguments,
    )

    def evaluate(
        parameters: Any,
        direction: Any,
        anchor_batch: Any,
        context_batches: Any,
        day_inputs: Any,
    ) -> Any:
        return jax.jvp(
            lambda current: objective(
                current,
                anchor_batch,
                context_batches,
                day_inputs,
            ),
            (parameters,),
            (direction,),
            has_aux=True,
        )

    return jax.jit(evaluate)


def _empty_accumulators(jax: Any, parameters: Any) -> list[Any]:
    import jax.numpy as jnp

    return [
        jax.tree_util.tree_map(lambda value: jnp.zeros_like(value), parameters)
        for _label in GRADIENT_FEASIBILITY_LABELS
    ]


def _load_group(
    jax: Any,
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    prepared_manifest_path: Path,
    signature: Mapping[str, Any],
    group_index: int,
) -> tuple[Any, Any, Any, list[tuple[Mapping[str, Any], Any]]]:
    offset = group_index * POINTS_PER_MIXED_UPDATE
    group = []
    for point_index in range(POINTS_PER_MIXED_UPDATE):
        record = manifest["entries"][offset + point_index]
        entry = schedule["entries"][int(record["update"]) - 1]
        artifact = load_prepared_artifact(
            prepared_manifest_path,
            record,
            schedule_entry=entry,
        )
        if artifact.trace_signature != signature:
            raise ValueError("multibatch direction retained-tail trace drift")
        group.append((entry, artifact))
    point_years = {
        (str(entry["landpoint_id"]), int(entry["year"]))
        for entry, _artifact in group
    }
    if len(point_years) != POINTS_PER_MIXED_UPDATE:
        raise ValueError("multibatch direction group repeated a point-year")
    contexts = tuple(
        _slice_context_batch(jax, artifact.batch, context_index=index)
        for index, (_entry, artifact) in enumerate(group)
    )
    return (
        _join_context_anchors(jax, contexts),
        _stack_contexts(jax, contexts),
        _stack_contexts(
            jax,
            tuple(artifact.day_inputs for _entry, artifact in group),
        ),
        group,
    )


def _accumulate_group_gradients(
    jax: Any,
    measurement: Any,
    accumulate: Any,
    count_nonfinite: Any,
    accumulators: list[Any],
    parameters: Any,
    anchor_batch: Any,
    context_batch: Any,
    context_inputs: Any,
) -> Mapping[str, Any]:
    before = None
    parent_hard = None
    nonfinite = {}
    scale = np.asarray(1.0 / AUDIT_GROUPS, dtype=np.float64)
    for component_index, seed in enumerate(
        np.eye(len(GRADIENT_FEASIBILITY_LABELS), dtype=np.float64)
    ):
        losses, gradient, hard_counts = measurement(
            parameters,
            seed,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        jax.block_until_ready((losses, gradient, hard_counts))
        current = np.asarray(losses, dtype=np.float64)
        current_hard = _hard_counts_host(hard_counts)
        if before is None:
            before = current
            parent_hard = current_hard
        elif not np.array_equal(before, current) or parent_hard != current_hard:
            raise ValueError("multibatch direction seed changed objective outputs")
        gradient_nonfinite = count_nonfinite(gradient)
        jax.block_until_ready(gradient_nonfinite)
        label = GRADIENT_FEASIBILITY_LABELS[component_index]
        nonfinite[label] = int(np.asarray(gradient_nonfinite))
        accumulators[component_index] = accumulate(
            accumulators[component_index],
            gradient,
            scale,
        )
        jax.block_until_ready(accumulators[component_index])
        del losses, gradient, hard_counts
    if before is None or parent_hard is None:
        raise ValueError("multibatch direction produced no gradients")
    return {
        "losses_before": {
            label: float(before[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "parent_hard_gate_counts": parent_hard,
        "nonfinite_component_gradient_values": nonfinite,
    }


def _collect_group_unit_gradients(
    jax: Any,
    measurement: Any,
    flatten: Any,
    unit_gradients: list[Any],
    parameters: Any,
    anchor_batch: Any,
    context_batch: Any,
    context_inputs: Any,
) -> Mapping[str, Any]:
    before = None
    parent_hard = None
    gradient_norms = {}
    nonfinite = {}
    for component_index, seed in enumerate(
        np.eye(len(GRADIENT_FEASIBILITY_LABELS), dtype=np.float64)
    ):
        losses, gradient, hard_counts = measurement(
            parameters,
            seed,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        unit, norm, gradient_nonfinite = flatten(gradient)
        jax.block_until_ready((losses, unit, norm, gradient_nonfinite, hard_counts))
        current = np.asarray(losses, dtype=np.float64)
        current_hard = _hard_counts_host(hard_counts)
        if before is None:
            before = current
            parent_hard = current_hard
        elif not np.array_equal(before, current) or parent_hard != current_hard:
            raise ValueError("group-aware direction seed changed objective outputs")
        label = GRADIENT_FEASIBILITY_LABELS[component_index]
        gradient_norms[label] = float(np.asarray(norm))
        nonfinite[label] = int(np.asarray(gradient_nonfinite))
        unit_gradients.append(unit)
        del losses, gradient, hard_counts, norm, gradient_nonfinite
    if before is None or parent_hard is None:
        raise ValueError("group-aware direction produced no gradients")
    return {
        "losses_before": {
            label: float(before[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "parent_hard_gate_counts": parent_hard,
        "gradient_norms": gradient_norms,
        "nonfinite_component_gradient_values": nonfinite,
    }


def _collect_selected_unit_gradients(
    jax: Any,
    measurement: Any,
    flatten: Any,
    *,
    selected_labels: Sequence[str],
    unit_by_label: dict[str, Any],
    parameters: Any,
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    prepared_manifest_path: Path,
    signature: Any,
) -> tuple[list[Mapping[str, Any]], int]:
    """Collect individual unit gradients only for selected group constraints."""

    selected = set(selected_labels)
    records = []
    nonfinite_count = 0
    for group_index in range(AUDIT_GROUPS):
        group_labels = tuple(
            f"group_{group_index + 1:02d}.{label}"
            for label in GRADIENT_FEASIBILITY_LABELS
            if f"group_{group_index + 1:02d}.{label}" in selected
        )
        if not group_labels:
            continue
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            prepared_manifest_path,
            signature,
            group_index,
        )
        losses_before = None
        hard_before = None
        norms = {}
        for constraint in group_labels:
            label = constraint.split(".", 1)[1]
            seed = np.zeros(len(GRADIENT_FEASIBILITY_LABELS), dtype=np.float64)
            seed[GRADIENT_FEASIBILITY_LABELS.index(label)] = 1.0
            losses, gradient, hard_counts = measurement(
                parameters,
                seed,
                anchor_batch,
                context_batch,
                context_inputs,
            )
            unit, norm, gradient_nonfinite = flatten(gradient)
            jax.block_until_ready(
                (losses, unit, norm, gradient_nonfinite, hard_counts)
            )
            current_losses = np.asarray(losses, dtype=np.float64)
            current_hard = _hard_counts_host(hard_counts)
            if losses_before is None:
                losses_before = current_losses
                hard_before = current_hard
            elif not np.array_equal(losses_before, current_losses) or (
                hard_before != current_hard
            ):
                raise ValueError("active-set seed changed objective outputs")
            unit_by_label[constraint] = unit
            norms[constraint] = float(np.asarray(norm))
            nonfinite_count += int(np.asarray(gradient_nonfinite))
            del losses, gradient, hard_counts, norm, gradient_nonfinite
        records.append(
            {
                "group": group_index,
                "constraints": list(group_labels),
                "gradient_norms": norms,
                "losses": (
                    losses_before.tolist() if losses_before is not None else None
                ),
                "hard_gate_counts": hard_before,
            }
        )
        del anchor_batch, context_batch, context_inputs, group
        gc.collect()
    if set(selected_labels) - set(unit_by_label):
        raise ValueError("active-set gradient collection incomplete")
    return records, nonfinite_count


def classify_group_aware_direction(
    diagnostics: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    *,
    step_norms: tuple[float, ...],
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    direction = diagnostics["directions"]["minimum_norm_unit"]
    cosines = direction["normalized_directional_derivatives"]
    minimum_cosine = min(float(value) for value in cosines.values())
    direction_passed = bool(direction["all_strictly_positive"]) and minimum_cosine >= float(
        contract["direction_gate"]["minimum_group_component_cosine"]
    )
    if not direction_passed:
        return {
            "status": "group_aware_common_direction_not_supported",
            "passed": False,
            "direction_passed": False,
            "minimum_group_component_cosine": minimum_cosine,
            "selected_step_norm": None,
            "candidates": [],
            "next_action": contract["direction_fail_action"],
        }
    result = dict(
        classify_direct_trust_region(
            records,
            step_norms=step_norms,
            gates=contract["gates"],
        )
    )
    result["direction_passed"] = True
    result["minimum_group_component_cosine"] = minimum_cosine
    result["status"] = (
        "group_aware_finite_step_supported"
        if result["passed"]
        else "group_aware_finite_step_not_supported"
    )
    result["next_action"] = (
        contract["pass_action"] if result["passed"] else contract["step_fail_action"]
    )
    return result


def _frozen_scalarization_seeds(
    predecessor: Mapping[str, Any],
    *,
    expected_step_norm: float,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    classification = predecessor.get("classification", {})
    if (
        predecessor.get("schema_version")
        != "successor_group_aware_step_extension_report_v1"
        or predecessor.get("status") != "completed"
        or classification.get("status") != "group_aware_finite_step_supported"
        or not classification.get("passed", False)
        or float(classification.get("selected_step_norm")) != expected_step_norm
    ):
        raise ValueError("group-aware scalarization predecessor is not admitted")
    records = predecessor.get("records", ())
    if len(records) != AUDIT_GROUPS:
        raise ValueError("group-aware scalarization predecessor group count drift")
    weights = predecessor["aggregate_direction"]["normalized_simplex_weights"]
    seeds = np.zeros(
        (AUDIT_GROUPS, len(GRADIENT_FEASIBILITY_LABELS)), dtype=np.float64
    )
    positive_constraints = 0
    for group_index, record in enumerate(records):
        if int(record["group"]) != group_index:
            raise ValueError("group-aware scalarization predecessor order drift")
        for component_index, label in enumerate(GRADIENT_FEASIBILITY_LABELS):
            constraint = f"group_{group_index + 1:02d}.{label}"
            weight = float(weights[constraint])
            norm = float(record["gradient_norms"][label])
            if not np.isfinite(norm) or norm <= 0.0 or weight < 0.0:
                raise ValueError("invalid frozen scalarization coefficient")
            seeds[group_index, component_index] = weight / norm
            positive_constraints += int(weight > 0.0)
    active_groups = int(np.count_nonzero(np.any(seeds > 0.0, axis=1)))
    return seeds, {
        "positive_constraints": positive_constraints,
        "active_groups": active_groups,
        "seed_sum": float(np.sum(seeds)),
        "expected_direction_norm": float(
            predecessor["aggregate_direction"]["direction_norm"]
        ),
    }


def classify_frozen_scalarization_smoke(
    validation: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    pass_action: str,
    fail_action: str,
) -> Mapping[str, Any]:
    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CANDIDATE as VALIDATION_CANDIDATE,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CONTROL as VALIDATION_BASELINE,
    )
    from scripts.hpc.run_successor_named_biomass_objective_ab import (
        classify_objective_ab,
    )

    total_updates = int(gates["total_updates"])
    all_relative = {
        label: [
            float(group["loss_relative_changes"][label])
            for update in updates
            for group in update["groups"]
        ]
        for label in GRADIENT_FEASIBILITY_LABELS
    }
    component_medians = {
        label: float(np.median(values)) if values else None
        for label, values in all_relative.items()
    }
    maximum_change = max(
        (
            max(float(value) for value in group["loss_relative_changes"].values())
            for update in updates
            for group in update["groups"]
        ),
        default=None,
    )
    hard_failure_count = sum(
        sum(group["parent_hard_gate_counts"].values())
        + sum(group["proposed_hard_gate_counts"].values())
        for update in updates
        for group in update["groups"]
    )
    nonfinite_count = sum(
        sum(update["nonfinite_values"].values()) for update in updates
    )
    first_equivalence = updates[0].get("parent_equivalence", {}) if updates else {}
    training_checks = {
        "all_updates_completed": len(updates) == total_updates
        and all(bool(update["update_applied"]) for update in updates),
        "frozen_support_size_exact": bool(updates)
        and int(updates[0]["scalarization"]["positive_constraints"])
        == int(gates["expected_positive_constraints"])
        and int(updates[0]["scalarization"]["active_groups"])
        == int(gates["expected_active_groups"]),
        "parent_direction_equivalent": bool(first_equivalence)
        and float(first_equivalence["direction_norm_relative_error"])
        <= float(gates["maximum_parent_direction_norm_relative_error"])
        and float(first_equivalence["maximum_virtual_relative_change_error"])
        <= float(gates["maximum_parent_virtual_relative_change_error"]),
        "hard_failures_zero": hard_failure_count
        == int(gates["hard_failure_count"]),
        "nonfinite_values_zero": nonfinite_count
        == int(gates["nonfinite_value_count"]),
        "every_group_component_nonincreasing": maximum_change is not None
        and maximum_change
        <= float(gates["maximum_group_component_relative_increase"]),
        "each_component_median_decreased": all(
            value is not None
            and value <= -float(gates["minimum_each_component_median_relative_decrease"])
            for value in component_medians.values()
        ),
    }
    validation_result = classify_objective_ab(
        {
            "current_objective_control": validation[VALIDATION_BASELINE],
            "named_biomass_candidate": validation[VALIDATION_CANDIDATE],
        },
        gates=gates,
    )
    passed = all(training_checks.values()) and validation_result["passed"]
    return {
        "status": (
            "frozen_group_aware_scalarization_smoke_admitted"
            if passed
            else "frozen_group_aware_scalarization_smoke_rejected"
        ),
        "passed": passed,
        "training_checks": training_checks,
        "component_median_relative_changes": component_medians,
        "maximum_group_component_relative_increase": maximum_change,
        "hard_failure_count": hard_failure_count,
        "nonfinite_value_count": nonfinite_count,
        "validation": validation_result,
        "next_action": pass_action if passed else fail_action,
    }


def classify_second_state_recalibration(
    direction_classification: Mapping[str, Any],
    support: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
    pass_action: str,
    direction_fail_action: str,
    step_fail_action: str,
    support_fail_action: str,
) -> Mapping[str, Any]:
    """Classify the exact second-state direction and its active-set support."""

    direction_passed = bool(direction_classification["direction_passed"])
    finite_step_passed = bool(direction_classification["passed"])
    sparse = int(support["recalibrated_support_size"]) <= int(
        gates["maximum_recalibrated_support_size"]
    )
    overlapping = int(support["overlap_size"]) >= int(
        gates["minimum_parent_support_overlap"]
    )
    active_set_supported = finite_step_passed and sparse and overlapping
    if not direction_passed:
        status = "second_state_common_direction_not_supported"
        next_action = direction_fail_action
    elif not finite_step_passed:
        status = "second_state_finite_step_not_supported"
        next_action = step_fail_action
    elif not (sparse and overlapping):
        status = "second_state_support_not_suitable_for_active_set"
        next_action = support_fail_action
    else:
        status = "second_state_active_set_recalibration_supported"
        next_action = pass_action
    return {
        "status": status,
        "passed": active_set_supported,
        "direction_passed": direction_passed,
        "finite_step_passed": finite_step_passed,
        "selected_step_norm": direction_classification["selected_step_norm"],
        "minimum_group_component_cosine": direction_classification[
            "minimum_group_component_cosine"
        ],
        "support_sparse": sparse,
        "support_overlapping": overlapping,
        "direction_classification": direction_classification,
        "next_action": next_action,
    }


def classify_active_set_recalibration(
    direction_checks: Mapping[str, Any],
    step_classification: Mapping[str, Any] | None,
    efficiency: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
    exact_selected_step_norm: float,
    pass_action: str,
    fail_action: str,
) -> Mapping[str, Any]:
    """Require directional closure, bounded VJPs, and the exact finite-step gate."""

    direction_passed = (
        int(direction_checks["nonpositive_constraint_count"]) == 0
        and float(direction_checks["minimum_relative_directional_derivative"])
        > float(gates["minimum_relative_directional_derivative"])
    )
    step_passed = bool(step_classification and step_classification["passed"])
    selected_step = (
        float(step_classification["selected_step_norm"])
        if step_classification and step_classification["selected_step_norm"] is not None
        else None
    )
    efficiency_passed = (
        int(efficiency["component_reverse_passes"])
        <= int(gates["maximum_component_reverse_passes"])
        and int(efficiency["directional_check_rounds"])
        <= int(gates["maximum_directional_check_rounds"])
        and int(efficiency["added_constraints"])
        <= int(gates["maximum_added_constraints"])
    )
    reference_step_reproduced = selected_step == float(exact_selected_step_norm)
    passed = (
        direction_passed
        and step_passed
        and efficiency_passed
        and reference_step_reproduced
    )
    return {
        "status": (
            "active_set_recalibration_supported"
            if passed
            else "active_set_recalibration_not_supported"
        ),
        "passed": passed,
        "direction_passed": direction_passed,
        "finite_step_passed": step_passed,
        "efficiency_passed": efficiency_passed,
        "reference_step_reproduced": reference_step_reproduced,
        "selected_step_norm": selected_step,
        "direction_checks": direction_checks,
        "step_classification": step_classification,
        "efficiency": efficiency,
        "next_action": pass_action if passed else fail_action,
    }


def classify_repeated_active_set_smoke(
    validation: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    pass_action: str,
    fail_action: str,
) -> Mapping[str, Any]:
    """Classify repeated mechanism closure separately from promotion scale."""

    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CANDIDATE as VALIDATION_CANDIDATE,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CONTROL as VALIDATION_BASELINE,
    )
    from scripts.hpc.run_successor_named_biomass_objective_ab import (
        classify_objective_ab,
    )

    expected_updates = int(gates["training"]["expected_active_set_updates"])
    training_checks = {
        "all_updates_applied": len(updates) == expected_updates
        and all(bool(update["update_applied"]) for update in updates),
        "all_directions_passed": len(updates) == expected_updates
        and all(bool(update["direction_passed"]) for update in updates),
        "reverse_pass_budget_met": len(updates) == expected_updates
        and all(
            int(update["component_reverse_passes"])
            <= int(gates["training"]["maximum_component_reverse_passes_per_update"])
            for update in updates
        ),
        "round_budget_met": len(updates) == expected_updates
        and all(
            len(update["rounds"])
            <= int(gates["training"]["maximum_directional_check_rounds_per_update"])
            for update in updates
        ),
        "hard_failures_zero": sum(
            int(update["hard_failure_count"]) for update in updates
        )
        == int(gates["training"]["hard_failure_count"]),
        "nonfinite_values_zero": sum(
            int(update["nonfinite_value_count"]) for update in updates
        )
        == int(gates["training"]["nonfinite_value_count"]),
    }
    objective_validation = {
        "current_objective_control": validation[VALIDATION_BASELINE],
        "named_biomass_candidate": validation[VALIDATION_CANDIDATE],
    }
    directional_validation = classify_objective_ab(
        objective_validation,
        gates=gates["validation"],
    )
    promotion_gates = dict(gates["validation"])
    promotion_gates["minimum_research_primary_composite_relative_improvement"] = (
        float(gates["promotion"]["minimum_research_primary_composite_relative_improvement"])
    )
    promotion_validation = classify_objective_ab(
        objective_validation,
        gates=promotion_gates,
    )
    mechanism_passed = all(training_checks.values())
    passed = mechanism_passed and bool(directional_validation["passed"])
    return {
        "status": (
            "repeated_active_set_smoke_supported"
            if passed
            else "repeated_active_set_smoke_not_supported"
        ),
        "passed": passed,
        "mechanism_passed": mechanism_passed,
        "training_checks": training_checks,
        "directional_validation": directional_validation,
        "promotion_threshold_met": bool(promotion_validation["passed"]),
        "promotion_validation": promotion_validation,
        "next_action": pass_action if passed else fail_action,
    }


def classify_cumulative_microstep_smoke(
    validation: Mapping[str, Any] | None,
    microsteps: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    maximum_microsteps: int,
    pass_action: str,
    fail_action: str,
) -> Mapping[str, Any]:
    """Classify one cumulative macro-update and optional rollout evidence."""

    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CANDIDATE as VALIDATION_CANDIDATE,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CONTROL as VALIDATION_BASELINE,
    )
    from scripts.hpc.run_successor_named_biomass_objective_ab import (
        classify_objective_ab,
    )

    applied = tuple(step for step in microsteps if step["update_applied"])
    final_cumulative = applied[-1]["cumulative"] if applied else None
    macro_effect_passed = bool(final_cumulative and final_cumulative["effect_gate_passed"])
    training_checks = {
        "bounded_microstep_count": 0 < len(microsteps) <= maximum_microsteps,
        "at_least_one_microstep_applied": bool(applied),
        "all_applied_microsteps_safe": all(bool(step["microstep_safe"]) for step in applied),
        "stopped_before_unsafe_microstep": all(bool(step["update_applied"]) for step in microsteps[:-1]),
        "all_attempted_directions_passed": all(bool(step["direction_passed"]) for step in microsteps),
        "reverse_pass_budget_met": all(
            int(step["component_reverse_passes"])
            <= int(gates["training"]["maximum_component_reverse_passes_per_microstep"])
            for step in microsteps
        ),
        "round_budget_met": all(
            len(step["rounds"]) <= int(gates["training"]["maximum_directional_check_rounds_per_microstep"])
            for step in microsteps
        ),
        "hard_failures_zero": sum(int(step["hard_failure_count"]) for step in microsteps)
        == int(gates["training"]["hard_failure_count"]),
        "nonfinite_values_zero": sum(int(step["nonfinite_value_count"]) for step in microsteps)
        == int(gates["training"]["nonfinite_value_count"]),
        "cumulative_macro_effect_passed": macro_effect_passed,
    }
    mechanism_passed = all(training_checks.values())
    directional_validation = None
    promotion_validation = None
    if validation is not None:
        objective_validation = {
            "current_objective_control": validation[VALIDATION_BASELINE],
            "named_biomass_candidate": validation[VALIDATION_CANDIDATE],
        }
        directional_validation = classify_objective_ab(
            objective_validation,
            gates=gates["validation"],
        )
        promotion_gates = dict(gates["validation"])
        promotion_gates["minimum_research_primary_composite_relative_improvement"] = float(
            gates["promotion"]["minimum_research_primary_composite_relative_improvement"]
        )
        promotion_validation = classify_objective_ab(
            objective_validation,
            gates=promotion_gates,
        )
    validation_consistent = (validation is not None) == mechanism_passed
    passed = bool(
        mechanism_passed and validation_consistent and directional_validation and directional_validation["passed"]
    )
    return {
        "status": ("cumulative_microstep_smoke_supported" if passed else "cumulative_microstep_smoke_not_supported"),
        "passed": passed,
        "mechanism_passed": mechanism_passed,
        "training_checks": training_checks,
        "microsteps_attempted": len(microsteps),
        "microsteps_applied": len(applied),
        "cumulative": final_cumulative,
        "directional_validation": directional_validation,
        "promotion_threshold_met": bool(promotion_validation and promotion_validation["passed"]),
        "promotion_validation": promotion_validation,
        "validation_consistent": validation_consistent,
        "next_action": pass_action if passed else fail_action,
    }


def _run_frozen_scalarization_smoke(
    *,
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    runtime: Mapping[str, Any],
    jax: Any,
    contract: Any,
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    assets: Any,
    adapter_template: Any,
    adapter_timing: Mapping[str, Any],
    signature: Any,
    parameters: Any,
    parameter_count: int,
    long_scales: Any,
    measurement: Any,
    loss_evaluator: Any,
    memory_before: Mapping[str, Any],
    started: float,
    backtracking: bool,
) -> Mapping[str, Any]:
    from scripts.hpc.prepare_successor_rollout_screen import (
        _prepare_adapter as _prepare_validation_adapter,
    )
    from scripts.hpc.prepare_successor_staged_validation import (
        _reference as _validation_reference,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CANDIDATE as VALIDATION_CANDIDATE,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CONTROL as VALIDATION_BASELINE,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import _evaluate_subset
    from scripts.hpc.successor_staged_validation_prepared import (
        load_staged_validation_prepared,
        ordered_validation_entries,
    )

    step_norm = float(audit["update"]["global_step_norm"])
    step_norms = tuple(
        float(value) for value in audit["update"].get("step_norm_grid", [step_norm])
    )
    if not step_norms or step_norms[0] != step_norm or any(
        left <= right for left, right in zip(step_norms, step_norms[1:])
    ):
        raise ValueError("scalarization step grid must start at the admitted norm")
    minimum_admissible_step = float(
        audit["update"].get("minimum_admissible_step_norm", step_norm)
    )
    seeds, scalarization = _frozen_scalarization_seeds(
        predecessor,
        expected_step_norm=step_norm,
    )
    if (
        scalarization["positive_constraints"]
        != int(audit["gates"]["expected_positive_constraints"])
        or scalarization["active_groups"]
        != int(audit["gates"]["expected_active_groups"])
    ):
        raise ValueError("frozen scalarization support drift")
    active_groups = tuple(
        int(index) for index in np.flatnonzero(np.any(seeds > 0.0, axis=1))
    )
    combine = _make_combined_direction(jax)
    direct_step = _make_direct_virtual_step(jax)
    count_nonfinite = _make_nonfinite_counter(jax)
    parent_parameters = parameters
    updates = []

    for short_update in range(1, SCALARIZATION_SMOKE_UPDATES + 1):
        before_records = []
        gradients = []
        gradient_nonfinite = 0
        for group_index in range(AUDIT_GROUPS):
            anchor_batch, context_batch, context_inputs, group = _load_group(
                jax,
                manifest,
                schedule,
                args.prepared_manifest,
                signature,
                group_index,
            )
            if group_index in active_groups:
                losses, gradient, hard_counts = measurement(
                    parameters,
                    seeds[group_index],
                    anchor_batch,
                    context_batch,
                    context_inputs,
                )
                current_nonfinite = count_nonfinite(gradient)
                jax.block_until_ready(
                    (losses, gradient, hard_counts, current_nonfinite)
                )
                gradients.append(gradient)
                gradient_nonfinite += int(np.asarray(current_nonfinite))
            else:
                losses, hard_counts = loss_evaluator(
                    parameters,
                    anchor_batch,
                    context_batch,
                    context_inputs,
                )
                jax.block_until_ready((losses, hard_counts))
            before_records.append(
                {
                    "group": group_index,
                    "point_years": [
                        {
                            "update": int(entry["update"]),
                            "landpoint_id": str(entry["landpoint_id"]),
                            "year": int(entry["year"]),
                        }
                        for entry, _artifact in group
                    ],
                    "losses": np.asarray(losses, dtype=np.float64),
                    "hard_gate_counts": _hard_counts_host(hard_counts),
                }
            )
            del anchor_batch, context_batch, context_inputs, group, losses, hard_counts
            gc.collect()
        direction, direction_norm, direction_nonfinite = combine(
            tuple(gradients),
            np.ones(len(gradients), dtype=np.float64),
        )
        direction_norm_host = float(np.asarray(direction_norm))
        direction_norm_error = abs(
            direction_norm_host - float(scalarization["expected_direction_norm"])
        ) / max(float(scalarization["expected_direction_norm"]), 1.0e-30)
        attempts = []
        selected = None
        parent_equivalence = None
        retained_attempt = None
        for candidate_step_norm in step_norms:
            proposed, proposed_nonfinite = direct_step(
                parameters,
                direction,
                direction_norm,
                np.asarray(candidate_step_norm, dtype=np.float64),
            )
            jax.block_until_ready(
                (
                    direction,
                    direction_norm,
                    direction_nonfinite,
                    proposed,
                    proposed_nonfinite,
                )
            )
            group_records = []
            virtual_error = 0.0
            for group_index in range(AUDIT_GROUPS):
                anchor_batch, context_batch, context_inputs, group = _load_group(
                    jax,
                    manifest,
                    schedule,
                    args.prepared_manifest,
                    signature,
                    group_index,
                )
                losses_after, hard_counts_after = loss_evaluator(
                    proposed,
                    anchor_batch,
                    context_batch,
                    context_inputs,
                )
                jax.block_until_ready((losses_after, hard_counts_after))
                before = before_records[group_index]["losses"]
                after = np.asarray(losses_after, dtype=np.float64)
                relative = _relative_changes(before, after)
                if short_update == 1 and candidate_step_norm == step_norm:
                    expected = predecessor["records"][group_index]["steps"][
                        format(step_norm, ".12g")
                    ]["virtual_loss_relative_changes"]
                    virtual_error = max(
                        virtual_error,
                        max(
                            abs(float(relative[label]) - float(expected[label]))
                            for label in GRADIENT_FEASIBILITY_LABELS
                        ),
                    )
                group_records.append(
                    {
                        "group": group_index,
                        "point_years": before_records[group_index]["point_years"],
                        "losses_before": {
                            label: float(before[index])
                            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
                        },
                        "losses_after": {
                            label: float(after[index])
                            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
                        },
                        "loss_relative_changes": relative,
                        "parent_hard_gate_counts": before_records[group_index][
                            "hard_gate_counts"
                        ],
                        "proposed_hard_gate_counts": _hard_counts_host(
                            hard_counts_after
                        ),
                    }
                )
                del anchor_batch, context_batch, context_inputs, group
                del losses_after, hard_counts_after
                gc.collect()
            losses_nonfinite = sum(
                int(
                    np.count_nonzero(
                        ~np.isfinite(tuple(group_record[loss_key].values()))
                    )
                )
                for group_record in group_records
                for loss_key in (
                    "losses_before",
                    "losses_after",
                    "loss_relative_changes",
                )
            )
            nonfinite = {
                "scalar_group_gradients": gradient_nonfinite,
                "direction": int(np.asarray(direction_nonfinite)),
                "proposed_parameters": int(np.asarray(proposed_nonfinite)),
                "losses_and_relative_changes": losses_nonfinite,
            }
            hard_failures = sum(
                sum(group_record["parent_hard_gate_counts"].values())
                + sum(group_record["proposed_hard_gate_counts"].values())
                for group_record in group_records
            )
            maximum_change = max(
                float(value)
                for group_record in group_records
                for value in group_record["loss_relative_changes"].values()
            )
            candidate_admissible = (
                candidate_step_norm >= minimum_admissible_step
                and hard_failures == 0
                and sum(nonfinite.values()) == 0
                and maximum_change
                <= float(
                    audit["gates"]["maximum_group_component_relative_increase"]
                )
            )
            attempts.append(
                {
                    "step_norm": candidate_step_norm,
                    "admissible": candidate_admissible,
                    "maximum_group_component_relative_increase": maximum_change,
                    "hard_failure_count": hard_failures,
                    "nonfinite_value_count": sum(nonfinite.values()),
                }
            )
            if short_update == 1 and candidate_step_norm == step_norm:
                parent_equivalence = {
                    "direction_norm_relative_error": direction_norm_error,
                    "maximum_virtual_relative_change_error": virtual_error,
                }
                candidate_admissible = candidate_admissible and (
                    direction_norm_error
                    <= float(
                        audit["gates"][
                            "maximum_parent_direction_norm_relative_error"
                        ]
                    )
                    and virtual_error
                    <= float(
                        audit["gates"][
                            "maximum_parent_virtual_relative_change_error"
                        ]
                    )
                )
                attempts[-1]["admissible"] = candidate_admissible
            if retained_attempt is not None:
                del retained_attempt
            retained_attempt = (
                proposed,
                group_records,
                nonfinite,
                hard_failures,
                maximum_change,
                candidate_step_norm,
            )
            if candidate_admissible:
                selected = retained_attempt
                break
            if not backtracking:
                break

        if retained_attempt is None:
            raise ValueError("scalarization smoke evaluated no candidate steps")
        (
            proposed,
            group_records,
            nonfinite,
            hard_failures,
            maximum_change,
            selected_step_norm,
        ) = retained_attempt
        update_applied = selected is not None
        updates.append(
            {
                "short_update": short_update,
                "update_applied": update_applied,
                "step_norm": selected_step_norm if update_applied else None,
                "attempts": attempts,
                "direction_norm": direction_norm_host,
                "scalarization": scalarization,
                "parent_equivalence": parent_equivalence,
                "groups": group_records,
                "maximum_group_component_relative_increase": maximum_change,
                "hard_failure_count": hard_failures,
                "nonfinite_values": nonfinite,
            }
        )
        if update_applied:
            parameters = proposed
        print(
            json.dumps(
                {
                    "event": "frozen_group_aware_scalarization_update",
                    "update": short_update,
                    "total": SCALARIZATION_SMOKE_UPDATES,
                    "applied": update_applied,
                    "step_norm": selected_step_norm if update_applied else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del before_records, gradients, direction, proposed, group_records
        gc.collect()
        if not update_applied:
            break

    validation_prepared, prepared_panel = load_staged_validation_prepared(
        args.validation_prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        validation_panel_sha256=audit["identity"]["validation_panel_sha256"],
    )
    if prepared_panel != assets.validation_panel:
        raise ValueError("frozen scalarization validation panel drift")
    first_slice, first_validation_entry = ordered_validation_entries(
        assets.validation_panel
    )[0]
    validation_reference = _validation_reference(
        runtime["index"],
        first_slice,
        first_validation_entry,
    )
    validation_adapter = _prepare_validation_adapter(
        runtime,
        validation_reference,
        int(first_validation_entry["start_rows"][0]),
    )
    validation = _evaluate_subset(
        runtime,
        contract=contract,
        parameters={
            VALIDATION_BASELINE: parent_parameters,
            VALIDATION_CANDIDATE: parameters,
        },
        long_scales=long_scales,
        adapter_template=validation_adapter,
        prepared_manifest_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=assets.validation_panel,
    )
    classification = classify_frozen_scalarization_smoke(
        validation,
        updates,
        gates=audit["gates"],
        pass_action=audit["pass_action"],
        fail_action=audit["fail_action"],
    )
    if backtracking:
        classification = dict(classification)
        classification["status"] = (
            "group_aware_backtracking_smoke_admitted"
            if classification["passed"]
            else "group_aware_backtracking_smoke_rejected"
        )
    report = {
        "schema_version": (
            "successor_group_aware_backtracking_smoke_report_v1"
            if backtracking
            else "successor_group_aware_scalarization_smoke_report_v1"
        ),
        "status": "completed",
        "git_head": _git_head(),
        "identity": {
            "smoke_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.smoke_report),
            "failed_scalarization_report_sha256": (
                _sha256_file(args.failed_scalarization_report)
                if backtracking
                else None
            ),
            "staged_training_contract_sha256": contract.sha256,
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": parameter_count,
        "parent_update": PARENT_UPDATE,
        "step_norm": step_norm,
        "step_norm_grid": list(step_norms),
        "updates": updates,
        "validation": validation,
        "classification": classification,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_updates": sum(bool(update["update_applied"]) for update in updates),
        "checkpoint_written": False,
        "recovery_written": False,
        "final_parameters_retained": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "next_gate": classification["next_action"],
    }
    _atomic_json(args.output, report)
    return report


def _run_second_state_recalibration_audit(
    *,
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    jax: Any,
    runtime: Mapping[str, Any],
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    signature: Any,
    parameters: Any,
    parameter_count: int,
    measurement: Any,
    loss_evaluator: Any,
    adapter_timing: Mapping[str, Any],
    memory_before: Mapping[str, Any],
    started: float,
) -> Mapping[str, Any]:
    """Recalibrate the exact 40-objective direction after one admitted step."""

    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    first_step_norm = float(audit["first_update"]["global_step_norm"])
    seeds, parent_scalarization = _frozen_scalarization_seeds(
        predecessor,
        expected_step_norm=first_step_norm,
    )
    if (
        parent_scalarization["positive_constraints"]
        != int(audit["gates"]["expected_parent_support_size"])
        or parent_scalarization["active_groups"]
        != int(audit["gates"]["expected_parent_active_groups"])
    ):
        raise ValueError("second-state parent scalarization support drift")

    active_groups = tuple(
        int(index) for index in np.flatnonzero(np.any(seeds > 0.0, axis=1))
    )
    combine = _make_combined_direction(jax)
    direct_step = _make_direct_virtual_step(jax)
    count_nonfinite = _make_nonfinite_counter(jax)
    first_gradients = []
    first_records = []
    first_gradient_nonfinite = 0
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        if group_index in active_groups:
            losses, gradient, hard_counts = measurement(
                parameters,
                seeds[group_index],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            gradient_nonfinite = count_nonfinite(gradient)
            jax.block_until_ready(
                (losses, gradient, hard_counts, gradient_nonfinite)
            )
            first_gradients.append(gradient)
            first_gradient_nonfinite += int(np.asarray(gradient_nonfinite))
        else:
            losses, hard_counts = loss_evaluator(
                parameters,
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses, hard_counts))
        first_records.append(
            {
                "group": group_index,
                "losses": np.asarray(losses, dtype=np.float64),
                "hard_gate_counts": _hard_counts_host(hard_counts),
            }
        )
        del anchor_batch, context_batch, context_inputs, group, losses, hard_counts
        gc.collect()

    first_direction, first_direction_norm, first_direction_nonfinite = combine(
        tuple(first_gradients),
        np.ones(len(first_gradients), dtype=np.float64),
    )
    second_state_parameters, first_proposed_nonfinite = direct_step(
        parameters,
        first_direction,
        first_direction_norm,
        np.asarray(first_step_norm, dtype=np.float64),
    )
    jax.block_until_ready(
        (
            first_direction,
            first_direction_norm,
            first_direction_nonfinite,
            second_state_parameters,
            first_proposed_nonfinite,
        )
    )
    first_direction_norm_host = float(np.asarray(first_direction_norm))
    first_direction_norm_error = abs(
        first_direction_norm_host
        - float(parent_scalarization["expected_direction_norm"])
    ) / max(float(parent_scalarization["expected_direction_norm"]), 1.0e-30)
    del first_gradients, first_direction
    gc.collect()

    flatten = _make_flattened_unit_gradient(jax)
    unit_gradients = []
    records = []
    maximum_first_step_reproduction_error = 0.0
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        record = _collect_group_unit_gradients(
            jax,
            measurement,
            flatten,
            unit_gradients,
            second_state_parameters,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        first_relative = _relative_changes(
            first_records[group_index]["losses"],
            np.asarray(
                [record["losses_before"][label] for label in GRADIENT_FEASIBILITY_LABELS],
                dtype=np.float64,
            ),
        )
        expected_relative = predecessor["records"][group_index]["steps"][
            format(first_step_norm, ".12g")
        ]["virtual_loss_relative_changes"]
        maximum_first_step_reproduction_error = max(
            maximum_first_step_reproduction_error,
            max(
                abs(float(first_relative[label]) - float(expected_relative[label]))
                for label in GRADIENT_FEASIBILITY_LABELS
            ),
        )
        records.append(
            {
                "group": group_index,
                "point_years": [
                    {
                        "update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                **record,
                "steps": {},
            }
        )
        del anchor_batch, context_batch, context_inputs, group
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "second_state_recalibration_gradient_group",
                    "group": group_index + 1,
                    "total": AUDIT_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    constraint_labels = tuple(
        f"group_{group_index + 1:02d}.{label}"
        for group_index in range(AUDIT_GROUPS)
        for label in GRADIENT_FEASIBILITY_LABELS
    )
    unit_matrix = jnp.stack(tuple(unit_gradients))
    gram = _make_flat_gram(jax)(unit_matrix)
    jax.block_until_ready(gram)
    diagnostics = gradient_direction_diagnostics(
        np.asarray(gram, dtype=np.float64),
        labels=constraint_labels,
    )
    selected = diagnostics["directions"]["minimum_norm_unit"]
    weights = np.asarray(
        [selected["weights"][label] for label in constraint_labels],
        dtype=np.float64,
    )
    direction_values, direction_norm, direction_nonfinite = _make_flat_direction(jax)(
        unit_matrix,
        weights,
    )
    _flat_parameters, unravel = ravel_pytree(second_state_parameters)
    direction = unravel(direction_values)
    jax.block_until_ready((direction, direction_norm, direction_nonfinite))
    del unit_gradients, unit_matrix, direction_values
    gc.collect()

    support_tolerance = float(audit["support"]["weight_tolerance"])
    parent_simplex = predecessor["aggregate_direction"][
        "normalized_simplex_weights"
    ]
    recalibrated_simplex = selected["normalized_simplex_weights"]
    parent_support = tuple(
        label for label in constraint_labels if float(parent_simplex[label]) > support_tolerance
    )
    recalibrated_support = tuple(
        label
        for label in constraint_labels
        if float(recalibrated_simplex[label]) > support_tolerance
    )
    parent_support_set = set(parent_support)
    recalibrated_support_set = set(recalibrated_support)
    overlap = tuple(sorted(parent_support_set & recalibrated_support_set))
    support = {
        "weight_tolerance": support_tolerance,
        "parent_support_size": len(parent_support),
        "recalibrated_support_size": len(recalibrated_support),
        "overlap_size": len(overlap),
        "parent_support": list(parent_support),
        "recalibrated_support": list(recalibrated_support),
        "overlap": list(overlap),
        "entered": sorted(recalibrated_support_set - parent_support_set),
        "dropped": sorted(parent_support_set - recalibrated_support_set),
    }

    step_norms = tuple(float(value) for value in audit["step_norm_grid"])
    minimum_direction_cosine = min(
        float(value)
        for value in selected["normalized_directional_derivatives"].values()
    )
    evaluate_steps = bool(selected["all_strictly_positive"]) and (
        minimum_direction_cosine
        >= float(audit["direction_gate"]["minimum_group_component_cosine"])
    )
    proposals = {}
    proposal_nonfinite = {}
    if evaluate_steps:
        for step_norm in step_norms:
            proposed, nonfinite = direct_step(
                second_state_parameters,
                direction,
                direction_norm,
                np.asarray(step_norm, dtype=np.float64),
            )
            jax.block_until_ready((proposed, nonfinite))
            key = format(step_norm, ".12g")
            proposals[key] = proposed
            proposal_nonfinite[key] = int(np.asarray(nonfinite))

    for group_index in range(AUDIT_GROUPS if evaluate_steps else 0):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        before = np.asarray(
            [records[group_index]["losses_before"][label] for label in GRADIENT_FEASIBILITY_LABELS],
            dtype=np.float64,
        )
        for step_norm in step_norms:
            key = format(step_norm, ".12g")
            losses_after, hard_counts = loss_evaluator(
                proposals[key],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses_after, hard_counts))
            after = np.asarray(losses_after, dtype=np.float64)
            records[group_index]["steps"][key] = {
                "step_norm": step_norm,
                "virtual_losses_after": {
                    label: float(after[index])
                    for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
                },
                "virtual_loss_relative_changes": _relative_changes(before, after),
                "hard_gate_counts": _hard_counts_host(hard_counts),
                "nonfinite_proposed_parameter_values": proposal_nonfinite[key],
            }
            del losses_after, hard_counts
        del anchor_batch, context_batch, context_inputs, group
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "second_state_recalibration_virtual_group",
                    "group": group_index + 1,
                    "total": AUDIT_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    direction_classification = classify_group_aware_direction(
        diagnostics,
        records,
        step_norms=step_norms,
        contract=audit,
    )
    classification = classify_second_state_recalibration(
        direction_classification,
        support,
        gates=audit["gates"],
        pass_action=audit["pass_action"],
        direction_fail_action=audit["direction_fail_action"],
        step_fail_action=audit["step_fail_action"],
        support_fail_action=audit["support_fail_action"],
    )
    first_hard_failure_count = sum(
        sum(record["hard_gate_counts"].values()) for record in first_records
    )
    hard_failure_count = first_hard_failure_count + sum(
        sum(record["parent_hard_gate_counts"].values())
        + sum(sum(step["hard_gate_counts"].values()) for step in record["steps"].values())
        for record in records
    )
    nonfinite_count = (
        first_gradient_nonfinite
        + int(np.asarray(first_direction_nonfinite))
        + int(np.asarray(first_proposed_nonfinite))
        + sum(
            sum(record["nonfinite_component_gradient_values"].values())
            for record in records
        )
        + int(np.asarray(direction_nonfinite))
        + sum(proposal_nonfinite.values())
    )
    all_losses_finite = all(
        all(np.isfinite(value) for value in record["losses_before"].values())
        and all(
            all(np.isfinite(value) for value in step["virtual_losses_after"].values())
            for step in record["steps"].values()
        )
        for record in records
    )
    reconstruction_passed = (
        first_direction_norm_error
        <= float(audit["gates"]["maximum_parent_direction_norm_relative_error"])
        and maximum_first_step_reproduction_error
        <= float(audit["gates"]["maximum_first_step_relative_change_error"])
    )
    evidence_valid = (
        reconstruction_passed
        and hard_failure_count == int(audit["gates"]["hard_failure_count"])
        and nonfinite_count == int(audit["gates"]["nonfinite_value_count"])
        and all_losses_finite
    )
    report = {
        "schema_version": "successor_second_state_recalibration_audit_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": _git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.smoke_report),
            "backtracking_report_sha256": _sha256_file(args.backtracking_report),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": parameter_count,
        "parent_update": PARENT_UPDATE,
        "first_update_reconstruction": {
            "step_norm": first_step_norm,
            "scalarization": parent_scalarization,
            "direction_norm": first_direction_norm_host,
            "direction_norm_relative_error": first_direction_norm_error,
            "maximum_relative_change_error": maximum_first_step_reproduction_error,
        },
        "second_state_direction": {
            "direction_norm": float(np.asarray(direction_norm)),
            "minimum_component_cosine": minimum_direction_cosine,
            "component_cosines": selected["normalized_directional_derivatives"],
            "normalized_simplex_weights": selected["normalized_simplex_weights"],
            "support": support,
        },
        "step_norm_grid": list(step_norms),
        "classification": classification,
        "records": records,
        "evidence_gates": {
            "evidence_valid": evidence_valid,
            "first_update_reconstruction_passed": reconstruction_passed,
            "all_losses_finite": all_losses_finite,
            "hard_failure_count": hard_failure_count,
            "nonfinite_value_count": nonfinite_count,
        },
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "reconstructed_direct_updates": 1,
        "new_updates_applied": 0,
        "optimizer_updates": 0,
        "proposed_parameters_retained": False,
        "checkpoint_written": False,
        "recovery_written": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "next_gate": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_second_state_recalibration_evidence"
        ),
    }
    _atomic_json(args.output, report)
    return report


def _recalibrate_active_set_direction(
    *,
    jax: Any,
    measurement: Any,
    directional_evaluator: Any,
    parameters: Any,
    initial_support: Sequence[str],
    active_set: Mapping[str, Any],
    gates: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    prepared_manifest_path: Path,
    signature: Any,
) -> Mapping[str, Any]:
    """Build one active-set direction and check all 40 objectives by JVP."""

    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    constraint_labels = tuple(
        f"group_{group_index + 1:02d}.{label}"
        for group_index in range(AUDIT_GROUPS)
        for label in GRADIENT_FEASIBILITY_LABELS
    )
    if not initial_support or not set(initial_support) <= set(constraint_labels):
        raise ValueError("active-set initial support is invalid")
    flatten = _make_flattened_unit_gradient(jax)
    unit_by_label: dict[str, Any] = {}
    gradient_records, nonfinite_count = _collect_selected_unit_gradients(
        jax,
        measurement,
        flatten,
        selected_labels=initial_support,
        unit_by_label=unit_by_label,
        parameters=parameters,
        manifest=manifest,
        schedule=schedule,
        prepared_manifest_path=prepared_manifest_path,
        signature=signature,
    )
    component_reverse_passes = len(initial_support)
    hard_failure_count = sum(
        sum(record["hard_gate_counts"].values())
        for record in gradient_records
        if record["hard_gate_counts"] is not None
    )
    added_constraints: list[str] = []
    rounds = []
    final = None
    support_tolerance = float(active_set["weight_tolerance"])
    threshold = float(gates["minimum_relative_directional_derivative"])

    for round_index in range(1, int(active_set["maximum_rounds"]) + 1):
        active_labels = tuple(
            label for label in constraint_labels if label in unit_by_label
        )
        unit_matrix = jnp.stack(tuple(unit_by_label[label] for label in active_labels))
        gram = _make_flat_gram(jax)(unit_matrix)
        jax.block_until_ready(gram)
        diagnostics = gradient_direction_diagnostics(
            np.asarray(gram, dtype=np.float64),
            labels=active_labels,
        )
        selected = diagnostics["directions"]["minimum_norm_unit"]
        weights = np.asarray(
            [selected["weights"][label] for label in active_labels],
            dtype=np.float64,
        )
        direction_values, direction_norm, direction_nonfinite = _make_flat_direction(
            jax
        )(unit_matrix, weights)
        _flat_parameters, unravel = ravel_pytree(parameters)
        direction = unravel(direction_values)
        unit_direction = jax.tree_util.tree_map(
            lambda value: value / jnp.maximum(direction_norm, 1.0e-30),
            direction,
        )
        jax.block_until_ready(
            (direction, direction_norm, direction_nonfinite, unit_direction)
        )
        nonfinite_count += int(np.asarray(direction_nonfinite))

        derivatives = {}
        group_losses = []
        group_hard = []
        for group_index in range(AUDIT_GROUPS):
            anchor_batch, context_batch, context_inputs, group = _load_group(
                jax,
                manifest,
                schedule,
                prepared_manifest_path,
                signature,
                group_index,
            )
            losses, directional, hard_counts = directional_evaluator(
                parameters,
                unit_direction,
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses, directional, hard_counts))
            losses_host = np.asarray(losses, dtype=np.float64)
            directional_host = np.asarray(directional, dtype=np.float64)
            nonfinite_count += int(np.sum(~np.isfinite(losses_host)))
            nonfinite_count += int(np.sum(~np.isfinite(directional_host)))
            hard_host = _hard_counts_host(hard_counts)
            hard_failure_count += sum(hard_host.values())
            for component_index, label in enumerate(GRADIENT_FEASIBILITY_LABELS):
                constraint = f"group_{group_index + 1:02d}.{label}"
                derivatives[constraint] = float(
                    directional_host[component_index]
                    / max(abs(losses_host[component_index]), 1.0e-30)
                )
            group_losses.append(losses_host)
            group_hard.append(hard_host)
            del anchor_batch, context_batch, context_inputs, group
            del losses, directional, hard_counts
            gc.collect()

        violations = tuple(
            label for label in constraint_labels if derivatives[label] <= threshold
        )
        new_constraints = tuple(
            label for label in violations if label not in unit_by_label
        )
        simplex_support = tuple(
            label
            for label in active_labels
            if float(selected["normalized_simplex_weights"][label])
            > support_tolerance
        )
        rounds.append(
            {
                "round": round_index,
                "available_gradient_count": len(active_labels),
                "simplex_support_size": len(simplex_support),
                "simplex_support": list(simplex_support),
                "direction_norm": float(np.asarray(direction_norm)),
                "minimum_relative_directional_derivative": min(
                    derivatives.values()
                ),
                "nonpositive_constraints": list(violations),
                "new_constraints": list(new_constraints),
            }
        )
        final = {
            "direction": direction,
            "direction_norm": direction_norm,
            "losses": group_losses,
            "hard_gate_counts": group_hard,
            "relative_directional_derivatives": derivatives,
            "support": simplex_support,
            "available_labels": active_labels,
            "normalized_simplex_weights": selected[
                "normalized_simplex_weights"
            ],
        }
        del unit_matrix, gram, diagnostics, direction_values, unit_direction
        gc.collect()
        if not new_constraints:
            break
        if (
            len(added_constraints) + len(new_constraints)
            > int(active_set["maximum_added_constraints"])
        ):
            break
        new_records, added_nonfinite = _collect_selected_unit_gradients(
            jax,
            measurement,
            flatten,
            selected_labels=new_constraints,
            unit_by_label=unit_by_label,
            parameters=parameters,
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=prepared_manifest_path,
            signature=signature,
        )
        gradient_records.extend(new_records)
        nonfinite_count += added_nonfinite
        hard_failure_count += sum(
            sum(record["hard_gate_counts"].values())
            for record in new_records
            if record["hard_gate_counts"] is not None
        )
        component_reverse_passes += len(new_constraints)
        added_constraints.extend(new_constraints)

    if final is None:
        raise ValueError("active-set recalibration produced no direction")
    nonpositive = tuple(
        label
        for label in constraint_labels
        if final["relative_directional_derivatives"][label] <= threshold
    )
    return {
        **final,
        "rounds": rounds,
        "direction_passed": not nonpositive,
        "minimum_relative_directional_derivative": min(
            final["relative_directional_derivatives"].values()
        ),
        "nonpositive_constraints": nonpositive,
        "component_reverse_passes": component_reverse_passes,
        "group_jvps": len(rounds) * AUDIT_GROUPS,
        "added_constraints": tuple(added_constraints),
        "nonfinite_value_count": nonfinite_count,
        "hard_failure_count": hard_failure_count,
    }


def _run_active_set_recalibration_audit(
    *,
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    exact_reference: Mapping[str, Any],
    jax: Any,
    runtime: Mapping[str, Any],
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    signature: Any,
    parameters: Any,
    parameter_count: int,
    measurement: Any,
    loss_evaluator: Any,
    directional_evaluator: Any,
    adapter_timing: Mapping[str, Any],
    memory_before: Mapping[str, Any],
    started: float,
) -> Mapping[str, Any]:
    """Audit prior-support active-set recalibration at the second state."""

    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    first_step_norm = float(audit["first_update"]["global_step_norm"])
    seeds, parent_scalarization = _frozen_scalarization_seeds(
        predecessor,
        expected_step_norm=first_step_norm,
    )
    active_parent_groups = tuple(
        int(index) for index in np.flatnonzero(np.any(seeds > 0.0, axis=1))
    )
    combine = _make_combined_direction(jax)
    direct_step = _make_direct_virtual_step(jax)
    count_nonfinite = _make_nonfinite_counter(jax)
    first_gradients = []
    first_records = []
    nonfinite_count = 0
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        if group_index in active_parent_groups:
            losses, gradient, hard_counts = measurement(
                parameters,
                seeds[group_index],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            gradient_nonfinite = count_nonfinite(gradient)
            jax.block_until_ready(
                (losses, gradient, hard_counts, gradient_nonfinite)
            )
            first_gradients.append(gradient)
            nonfinite_count += int(np.asarray(gradient_nonfinite))
        else:
            losses, hard_counts = loss_evaluator(
                parameters,
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses, hard_counts))
        first_records.append(
            {
                "group": group_index,
                "losses": np.asarray(losses, dtype=np.float64),
                "hard_gate_counts": _hard_counts_host(hard_counts),
            }
        )
        del anchor_batch, context_batch, context_inputs, group, losses, hard_counts
        gc.collect()

    first_direction, first_direction_norm, first_direction_nonfinite = combine(
        tuple(first_gradients),
        np.ones(len(first_gradients), dtype=np.float64),
    )
    second_state_parameters, first_proposed_nonfinite = direct_step(
        parameters,
        first_direction,
        first_direction_norm,
        np.asarray(first_step_norm, dtype=np.float64),
    )
    jax.block_until_ready(
        (
            first_direction,
            first_direction_norm,
            first_direction_nonfinite,
            second_state_parameters,
            first_proposed_nonfinite,
        )
    )
    nonfinite_count += int(np.asarray(first_direction_nonfinite))
    nonfinite_count += int(np.asarray(first_proposed_nonfinite))
    first_direction_norm_error = abs(
        float(np.asarray(first_direction_norm))
        - float(parent_scalarization["expected_direction_norm"])
    ) / max(float(parent_scalarization["expected_direction_norm"]), 1.0e-30)
    del first_gradients, first_direction
    gc.collect()

    constraint_labels = tuple(
        f"group_{group_index + 1:02d}.{label}"
        for group_index in range(AUDIT_GROUPS)
        for label in GRADIENT_FEASIBILITY_LABELS
    )
    support_tolerance = float(audit["active_set"]["weight_tolerance"])
    parent_weights = predecessor["aggregate_direction"][
        "normalized_simplex_weights"
    ]
    parent_support = tuple(
        label
        for label in constraint_labels
        if float(parent_weights[label]) > support_tolerance
    )
    if len(parent_support) != int(audit["gates"]["expected_parent_support_size"]):
        raise ValueError("active-set parent support drift")

    flatten = _make_flattened_unit_gradient(jax)
    unit_by_label: dict[str, Any] = {}
    gradient_records, gradient_nonfinite = _collect_selected_unit_gradients(
        jax,
        measurement,
        flatten,
        selected_labels=parent_support,
        unit_by_label=unit_by_label,
        parameters=second_state_parameters,
        manifest=manifest,
        schedule=schedule,
        prepared_manifest_path=args.prepared_manifest,
        signature=signature,
    )
    nonfinite_count += gradient_nonfinite
    component_reverse_passes = len(parent_support)
    added_constraints: list[str] = []
    rounds = []
    directional_hard_failure_count = 0
    final_direction = None
    final_direction_norm = None
    final_direction_checks = None
    final_losses = None
    final_hard_counts = None
    final_simplex = None

    for round_index in range(1, int(audit["active_set"]["maximum_rounds"]) + 1):
        active_labels = tuple(
            label for label in constraint_labels if label in unit_by_label
        )
        unit_matrix = jnp.stack(tuple(unit_by_label[label] for label in active_labels))
        gram = _make_flat_gram(jax)(unit_matrix)
        jax.block_until_ready(gram)
        diagnostics = gradient_direction_diagnostics(
            np.asarray(gram, dtype=np.float64),
            labels=active_labels,
        )
        selected = diagnostics["directions"]["minimum_norm_unit"]
        weights = np.asarray(
            [selected["weights"][label] for label in active_labels],
            dtype=np.float64,
        )
        direction_values, direction_norm, direction_nonfinite = _make_flat_direction(
            jax
        )(unit_matrix, weights)
        _flat_parameters, unravel = ravel_pytree(second_state_parameters)
        direction = unravel(direction_values)
        unit_direction = jax.tree_util.tree_map(
            lambda value: value / jnp.maximum(direction_norm, 1.0e-30),
            direction,
        )
        jax.block_until_ready(
            (direction, direction_norm, direction_nonfinite, unit_direction)
        )
        current_nonfinite = int(np.asarray(direction_nonfinite))
        nonfinite_count += current_nonfinite

        relative_derivatives = {}
        losses_by_group = []
        hard_by_group = []
        round_nonfinite = current_nonfinite
        for group_index in range(AUDIT_GROUPS):
            anchor_batch, context_batch, context_inputs, group = _load_group(
                jax,
                manifest,
                schedule,
                args.prepared_manifest,
                signature,
                group_index,
            )
            losses, derivatives, hard_counts = directional_evaluator(
                second_state_parameters,
                unit_direction,
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses, derivatives, hard_counts))
            losses_host = np.asarray(losses, dtype=np.float64)
            derivatives_host = np.asarray(derivatives, dtype=np.float64)
            round_nonfinite += int(np.sum(~np.isfinite(losses_host)))
            round_nonfinite += int(np.sum(~np.isfinite(derivatives_host)))
            for component_index, label in enumerate(GRADIENT_FEASIBILITY_LABELS):
                constraint = f"group_{group_index + 1:02d}.{label}"
                relative_derivatives[constraint] = float(
                    derivatives_host[component_index]
                    / max(abs(losses_host[component_index]), 1.0e-30)
                )
            losses_by_group.append(losses_host)
            hard_host = _hard_counts_host(hard_counts)
            hard_by_group.append(hard_host)
            directional_hard_failure_count += sum(hard_host.values())
            del anchor_batch, context_batch, context_inputs, group
            del losses, derivatives, hard_counts
            gc.collect()
        nonfinite_count += round_nonfinite - current_nonfinite
        threshold = float(audit["gates"]["minimum_relative_directional_derivative"])
        violations = tuple(
            label
            for label in constraint_labels
            if relative_derivatives[label] <= threshold
        )
        new_constraints = tuple(
            label for label in violations if label not in unit_by_label
        )
        simplex_support = tuple(
            label
            for label in active_labels
            if float(selected["normalized_simplex_weights"][label])
            > support_tolerance
        )
        rounds.append(
            {
                "round": round_index,
                "available_gradient_count": len(active_labels),
                "simplex_support_size": len(simplex_support),
                "simplex_support": list(simplex_support),
                "direction_norm": float(np.asarray(direction_norm)),
                "minimum_relative_directional_derivative": min(
                    relative_derivatives.values()
                ),
                "nonpositive_constraints": list(violations),
                "new_constraints": list(new_constraints),
            }
        )
        final_direction = direction
        final_direction_norm = direction_norm
        final_direction_checks = relative_derivatives
        final_losses = losses_by_group
        final_hard_counts = hard_by_group
        final_simplex = {
            "available_labels": list(active_labels),
            "normalized_simplex_weights": selected[
                "normalized_simplex_weights"
            ],
            "support": list(simplex_support),
        }
        del unit_matrix, gram, diagnostics, direction_values, unit_direction
        gc.collect()
        if not new_constraints:
            break
        if (
            len(added_constraints) + len(new_constraints)
            > int(audit["active_set"]["maximum_added_constraints"])
        ):
            break
        new_records, added_nonfinite = _collect_selected_unit_gradients(
            jax,
            measurement,
            flatten,
            selected_labels=new_constraints,
            unit_by_label=unit_by_label,
            parameters=second_state_parameters,
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=args.prepared_manifest,
            signature=signature,
        )
        gradient_records.extend(new_records)
        nonfinite_count += added_nonfinite
        component_reverse_passes += len(new_constraints)
        added_constraints.extend(new_constraints)

    if (
        final_direction is None
        or final_direction_norm is None
        or final_direction_checks is None
        or final_losses is None
        or final_hard_counts is None
        or final_simplex is None
    ):
        raise ValueError("active-set recalibration produced no direction")

    minimum_relative_derivative = min(final_direction_checks.values())
    nonpositive = tuple(
        label
        for label in constraint_labels
        if final_direction_checks[label]
        <= float(audit["gates"]["minimum_relative_directional_derivative"])
    )
    direction_checks = {
        "minimum_relative_directional_derivative": minimum_relative_derivative,
        "nonpositive_constraint_count": len(nonpositive),
        "nonpositive_constraints": list(nonpositive),
        "relative_directional_derivatives": final_direction_checks,
    }
    direction_passed = len(nonpositive) == 0
    step_norms = tuple(float(value) for value in audit["step_norm_grid"])
    records = [
        {
            "group": group_index,
            "losses_before": {
                label: float(final_losses[group_index][component_index])
                for component_index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
            },
            "parent_hard_gate_counts": final_hard_counts[group_index],
            "steps": {},
        }
        for group_index in range(AUDIT_GROUPS)
    ]
    proposal_nonfinite = {}
    proposals = {}
    if direction_passed:
        for step_norm in step_norms:
            proposed, proposed_nonfinite = direct_step(
                second_state_parameters,
                final_direction,
                final_direction_norm,
                np.asarray(step_norm, dtype=np.float64),
            )
            jax.block_until_ready((proposed, proposed_nonfinite))
            key = format(step_norm, ".12g")
            proposals[key] = proposed
            proposal_nonfinite[key] = int(np.asarray(proposed_nonfinite))
            nonfinite_count += proposal_nonfinite[key]

        for group_index in range(AUDIT_GROUPS):
            anchor_batch, context_batch, context_inputs, group = _load_group(
                jax,
                manifest,
                schedule,
                args.prepared_manifest,
                signature,
                group_index,
            )
            before = final_losses[group_index]
            for step_norm in step_norms:
                key = format(step_norm, ".12g")
                losses_after, hard_counts = loss_evaluator(
                    proposals[key],
                    anchor_batch,
                    context_batch,
                    context_inputs,
                )
                jax.block_until_ready((losses_after, hard_counts))
                after = np.asarray(losses_after, dtype=np.float64)
                nonfinite_count += int(np.sum(~np.isfinite(after)))
                records[group_index]["steps"][key] = {
                    "step_norm": step_norm,
                    "virtual_losses_after": {
                        label: float(after[index])
                        for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
                    },
                    "virtual_loss_relative_changes": _relative_changes(before, after),
                    "hard_gate_counts": _hard_counts_host(hard_counts),
                    "nonfinite_proposed_parameter_values": proposal_nonfinite[key],
                }
                del losses_after, hard_counts
            del anchor_batch, context_batch, context_inputs, group
            gc.collect()

    step_classification = (
        classify_direct_trust_region(
            records,
            step_norms=step_norms,
            gates=audit["gates"],
        )
        if direction_passed
        else None
    )
    efficiency = {
        "parent_reconstruction_group_reverse_passes": len(active_parent_groups),
        "component_reverse_passes": component_reverse_passes,
        "directional_check_rounds": len(rounds),
        "group_jvps": len(rounds) * AUDIT_GROUPS,
        "initial_constraints": len(parent_support),
        "added_constraints": len(added_constraints),
        "added_constraint_labels": added_constraints,
        "exact_reference_component_reverse_passes": int(
            audit["exact_reference"]["component_reverse_passes"]
        ),
    }
    exact_selected_step_norm = float(
        exact_reference["classification"]["selected_step_norm"]
    )
    classification = classify_active_set_recalibration(
        direction_checks,
        step_classification,
        efficiency,
        gates=audit["gates"],
        exact_selected_step_norm=exact_selected_step_norm,
        pass_action=audit["pass_action"],
        fail_action=audit["fail_action"],
    )
    hard_failure_count = sum(
        sum(record["hard_gate_counts"].values()) for record in first_records
    ) + directional_hard_failure_count + sum(
        sum(record["parent_hard_gate_counts"].values())
        + sum(sum(step["hard_gate_counts"].values()) for step in record["steps"].values())
        for record in records
    ) + sum(
        sum(record["hard_gate_counts"].values())
        for record in gradient_records
        if record["hard_gate_counts"] is not None
    )
    reconstruction_passed = first_direction_norm_error <= float(
        audit["gates"]["maximum_parent_direction_norm_relative_error"]
    )
    evidence_valid = (
        reconstruction_passed
        and hard_failure_count == int(audit["gates"]["hard_failure_count"])
        and nonfinite_count == int(audit["gates"]["nonfinite_value_count"])
    )
    report = {
        "schema_version": "successor_active_set_recalibration_audit_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": _git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.smoke_report),
            "exact_reference_report_sha256": _sha256_file(
                args.exact_recalibration_report
            ),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": parameter_count,
        "parent_update": PARENT_UPDATE,
        "first_update_reconstruction": {
            "step_norm": first_step_norm,
            "direction_norm_relative_error": first_direction_norm_error,
        },
        "active_set": {
            "parent_support": list(parent_support),
            "rounds": rounds,
            "final_simplex": final_simplex,
            "direction_checks": direction_checks,
            "efficiency": efficiency,
        },
        "exact_reference": {
            "minimum_component_cosine": exact_reference[
                "second_state_direction"
            ]["minimum_component_cosine"],
            "selected_step_norm": exact_selected_step_norm,
            "support": exact_reference["second_state_direction"]["support"],
        },
        "step_norm_grid": list(step_norms),
        "classification": classification,
        "records": records,
        "evidence_gates": {
            "evidence_valid": evidence_valid,
            "first_update_reconstruction_passed": reconstruction_passed,
            "hard_failure_count": hard_failure_count,
            "nonfinite_value_count": nonfinite_count,
        },
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "reconstructed_direct_updates": 1,
        "new_updates_applied": 0,
        "optimizer_updates": 0,
        "proposed_parameters_retained": False,
        "checkpoint_written": False,
        "recovery_written": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "next_gate": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_active_set_evidence"
        ),
    }
    _atomic_json(args.output, report)
    return report


def _run_repeated_active_set_smoke(
    *,
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    active_set_reference: Mapping[str, Any],
    jax: Any,
    runtime: Mapping[str, Any],
    contract: Any,
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    assets: Any,
    adapter_template: Any,
    adapter_timing: Mapping[str, Any],
    signature: Any,
    parameters: Any,
    parameter_count: int,
    long_scales: Any,
    measurement: Any,
    loss_evaluator: Any,
    directional_evaluator: Any,
    memory_before: Mapping[str, Any],
    started: float,
) -> Mapping[str, Any]:
    """Apply four bounded active-set updates after reconstructing update one."""

    from scripts.hpc.prepare_successor_rollout_screen import (
        _prepare_adapter as _prepare_validation_adapter,
    )
    from scripts.hpc.prepare_successor_staged_validation import (
        _reference as _validation_reference,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CANDIDATE as VALIDATION_CANDIDATE,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import (
        CONTROL as VALIDATION_BASELINE,
    )
    from scripts.hpc.run_successor_gradient_balanced_smoke import _evaluate_subset
    from scripts.hpc.successor_staged_validation_prepared import (
        load_staged_validation_prepared,
        ordered_validation_entries,
    )

    first_step_norm = float(audit["first_update"]["global_step_norm"])
    seeds, parent_scalarization = _frozen_scalarization_seeds(
        predecessor,
        expected_step_norm=first_step_norm,
    )
    active_parent_groups = tuple(
        int(index) for index in np.flatnonzero(np.any(seeds > 0.0, axis=1))
    )
    combine = _make_combined_direction(jax)
    direct_step = _make_direct_virtual_step(jax)
    count_nonfinite = _make_nonfinite_counter(jax)
    parent_parameters = parameters
    first_gradients = []
    reconstruction_hard_failures = 0
    reconstruction_nonfinite = 0
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        if group_index in active_parent_groups:
            losses, gradient, hard_counts = measurement(
                parameters,
                seeds[group_index],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            gradient_nonfinite = count_nonfinite(gradient)
            jax.block_until_ready(
                (losses, gradient, hard_counts, gradient_nonfinite)
            )
            first_gradients.append(gradient)
            reconstruction_nonfinite += int(np.asarray(gradient_nonfinite))
        else:
            losses, hard_counts = loss_evaluator(
                parameters,
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses, hard_counts))
        reconstruction_hard_failures += sum(_hard_counts_host(hard_counts).values())
        reconstruction_nonfinite += int(
            np.sum(~np.isfinite(np.asarray(losses, dtype=np.float64)))
        )
        del anchor_batch, context_batch, context_inputs, group, losses, hard_counts
        gc.collect()

    first_direction, first_direction_norm, first_direction_nonfinite = combine(
        tuple(first_gradients),
        np.ones(len(first_gradients), dtype=np.float64),
    )
    parameters, first_proposed_nonfinite = direct_step(
        parameters,
        first_direction,
        first_direction_norm,
        np.asarray(first_step_norm, dtype=np.float64),
    )
    jax.block_until_ready(
        (
            first_direction,
            first_direction_norm,
            first_direction_nonfinite,
            parameters,
            first_proposed_nonfinite,
        )
    )
    reconstruction_nonfinite += int(np.asarray(first_direction_nonfinite))
    reconstruction_nonfinite += int(np.asarray(first_proposed_nonfinite))
    first_direction_norm_error = abs(
        float(np.asarray(first_direction_norm))
        - float(parent_scalarization["expected_direction_norm"])
    ) / max(float(parent_scalarization["expected_direction_norm"]), 1.0e-30)
    del first_gradients, first_direction
    gc.collect()

    constraint_labels = tuple(
        f"group_{group_index + 1:02d}.{label}"
        for group_index in range(AUDIT_GROUPS)
        for label in GRADIENT_FEASIBILITY_LABELS
    )
    support_tolerance = float(audit["active_set"]["weight_tolerance"])
    parent_weights = predecessor["aggregate_direction"][
        "normalized_simplex_weights"
    ]
    support = tuple(
        label
        for label in constraint_labels
        if float(parent_weights[label]) > support_tolerance
    )
    step_norms = tuple(float(value) for value in audit["step_norm_grid"])
    if any(left <= right for left, right in zip(step_norms, step_norms[1:])):
        raise ValueError("repeated active-set step grid must be descending")
    updates = []

    for update_index in range(
        1, int(audit["training"]["active_set_updates"]) + 1
    ):
        update_started = time.perf_counter()
        recalibrated = _recalibrate_active_set_direction(
            jax=jax,
            measurement=measurement,
            directional_evaluator=directional_evaluator,
            parameters=parameters,
            initial_support=support,
            active_set=audit["active_set"],
            gates=audit["gates"]["direction"],
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=args.prepared_manifest,
            signature=signature,
        )
        attempts = []
        selected_parameters = None
        selected_groups = None
        selected_step_norm = None
        attempt_hard_failures = 0
        attempt_nonfinite = 0
        selected_hard_failures = 0
        selected_nonfinite = 0
        if recalibrated["direction_passed"]:
            for candidate_step_norm in step_norms:
                proposed, proposed_nonfinite = direct_step(
                    parameters,
                    recalibrated["direction"],
                    recalibrated["direction_norm"],
                    np.asarray(candidate_step_norm, dtype=np.float64),
                )
                jax.block_until_ready((proposed, proposed_nonfinite))
                current_nonfinite = int(np.asarray(proposed_nonfinite))
                groups = []
                changes = []
                current_hard_failures = 0
                for group_index in range(AUDIT_GROUPS):
                    anchor_batch, context_batch, context_inputs, group = _load_group(
                        jax,
                        manifest,
                        schedule,
                        args.prepared_manifest,
                        signature,
                        group_index,
                    )
                    losses_after, hard_counts = loss_evaluator(
                        proposed,
                        anchor_batch,
                        context_batch,
                        context_inputs,
                    )
                    jax.block_until_ready((losses_after, hard_counts))
                    before = recalibrated["losses"][group_index]
                    after = np.asarray(losses_after, dtype=np.float64)
                    relative = _relative_changes(before, after)
                    relative_values = np.asarray(
                        [relative[label] for label in GRADIENT_FEASIBILITY_LABELS],
                        dtype=np.float64,
                    )
                    hard_host = _hard_counts_host(hard_counts)
                    current_hard_failures += sum(hard_host.values())
                    current_nonfinite += int(np.sum(~np.isfinite(after)))
                    current_nonfinite += int(np.sum(~np.isfinite(relative_values)))
                    changes.append(relative_values)
                    groups.append(
                        {
                            "group": group_index,
                            "loss_relative_changes": relative,
                            "hard_gate_counts": hard_host,
                        }
                    )
                    del anchor_batch, context_batch, context_inputs, group
                    del losses_after, hard_counts
                    gc.collect()
                change_matrix = np.stack(changes)
                component_medians = {
                    label: float(np.median(change_matrix[:, component_index]))
                    for component_index, label in enumerate(
                        GRADIENT_FEASIBILITY_LABELS
                    )
                }
                maximum_change = float(np.max(change_matrix))
                admissible = (
                    current_hard_failures == 0
                    and current_nonfinite == 0
                    and maximum_change
                    <= float(
                        audit["gates"]["step"][
                            "maximum_group_component_relative_increase"
                        ]
                    )
                    and all(
                        value
                        <= -float(
                            audit["gates"]["step"][
                                "minimum_each_component_median_relative_decrease"
                            ]
                        )
                        for value in component_medians.values()
                    )
                )
                attempts.append(
                    {
                        "step_norm": candidate_step_norm,
                        "admissible": admissible,
                        "maximum_group_component_relative_increase": maximum_change,
                        "component_median_relative_changes": component_medians,
                        "hard_failure_count": current_hard_failures,
                        "nonfinite_value_count": current_nonfinite,
                    }
                )
                attempt_hard_failures += current_hard_failures
                attempt_nonfinite += current_nonfinite
                if admissible:
                    selected_parameters = proposed
                    selected_groups = groups
                    selected_step_norm = candidate_step_norm
                    selected_hard_failures = current_hard_failures
                    selected_nonfinite = current_nonfinite
                    break
                del proposed, groups
                gc.collect()

        update_applied = selected_parameters is not None
        updates.append(
            {
                "active_set_update": update_index,
                "total_direct_update": update_index + 1,
                "update_applied": update_applied,
                "direction_passed": bool(recalibrated["direction_passed"]),
                "minimum_relative_directional_derivative": float(
                    recalibrated["minimum_relative_directional_derivative"]
                ),
                "initial_support": list(support),
                "final_support": list(recalibrated["support"]),
                "rounds": recalibrated["rounds"],
                "component_reverse_passes": int(
                    recalibrated["component_reverse_passes"]
                ),
                "group_jvps": int(recalibrated["group_jvps"]),
                "added_constraints": list(recalibrated["added_constraints"]),
                "selected_step_norm": selected_step_norm,
                "attempts": attempts,
                "selected_groups": selected_groups,
                "hard_failure_count": int(recalibrated["hard_failure_count"])
                + selected_hard_failures,
                "nonfinite_value_count": int(
                    recalibrated["nonfinite_value_count"]
                )
                + selected_nonfinite,
                "attempted_hard_failure_count": attempt_hard_failures,
                "attempted_nonfinite_value_count": attempt_nonfinite,
                "elapsed_seconds": time.perf_counter() - update_started,
            }
        )
        print(
            json.dumps(
                {
                    "event": "repeated_active_set_update",
                    "update": update_index,
                    "total": int(audit["training"]["active_set_updates"]),
                    "applied": update_applied,
                    "step_norm": selected_step_norm,
                    "reverse_passes": int(
                        recalibrated["component_reverse_passes"]
                    ),
                    "jvp_rounds": len(recalibrated["rounds"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not update_applied:
            break
        parameters = selected_parameters
        support = tuple(recalibrated["support"])
        del recalibrated, selected_parameters
        gc.collect()

    validation_prepared, prepared_panel = load_staged_validation_prepared(
        args.validation_prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        validation_panel_sha256=audit["identity"]["validation_panel_sha256"],
    )
    if prepared_panel != assets.validation_panel:
        raise ValueError("repeated active-set validation panel drift")
    first_slice, first_validation_entry = ordered_validation_entries(
        assets.validation_panel
    )[0]
    validation_reference = _validation_reference(
        runtime["index"],
        first_slice,
        first_validation_entry,
    )
    validation_adapter = _prepare_validation_adapter(
        runtime,
        validation_reference,
        int(first_validation_entry["start_rows"][0]),
    )
    validation = _evaluate_subset(
        runtime,
        contract=contract,
        parameters={
            VALIDATION_BASELINE: parent_parameters,
            VALIDATION_CANDIDATE: parameters,
        },
        long_scales=long_scales,
        adapter_template=validation_adapter,
        prepared_manifest_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=assets.validation_panel,
    )
    classification = classify_repeated_active_set_smoke(
        validation,
        updates,
        gates=audit["gates"],
        pass_action=audit["pass_action"],
        fail_action=audit["fail_action"],
    )
    reference_minimum = float(
        active_set_reference["active_set"]["direction_checks"][
            "minimum_relative_directional_derivative"
        ]
    )
    reference_support = tuple(
        active_set_reference["active_set"]["final_simplex"]["support"]
    )
    first_active = updates[0] if updates else None
    reference_reproduced = bool(first_active) and (
        abs(
            float(first_active["minimum_relative_directional_derivative"])
            - reference_minimum
        )
        <= float(audit["gates"]["reference"]["maximum_minimum_derivative_error"])
        and tuple(first_active["final_support"]) == reference_support
        and float(first_active["selected_step_norm"])
        == float(active_set_reference["classification"]["selected_step_norm"])
    )
    evidence_valid = (
        first_direction_norm_error
        <= float(
            audit["gates"]["reference"][
                "maximum_parent_direction_norm_relative_error"
            ]
        )
        and reconstruction_hard_failures
        == int(audit["gates"]["training"]["hard_failure_count"])
        and reconstruction_nonfinite
        == int(audit["gates"]["training"]["nonfinite_value_count"])
        and reference_reproduced
    )
    report = {
        "schema_version": "successor_repeated_active_set_smoke_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": _git_head(),
        "identity": {
            "smoke_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.smoke_report),
            "active_set_reference_report_sha256": _sha256_file(
                args.active_set_reference_report
            ),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": parameter_count,
        "parent_update": PARENT_UPDATE,
        "first_update_reconstruction": {
            "step_norm": first_step_norm,
            "direction_norm_relative_error": first_direction_norm_error,
            "hard_failure_count": reconstruction_hard_failures,
            "nonfinite_value_count": reconstruction_nonfinite,
        },
        "updates": updates,
        "validation": validation,
        "classification": classification,
        "evidence_gates": {
            "evidence_valid": evidence_valid,
            "active_set_reference_reproduced": reference_reproduced,
        },
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "reconstructed_direct_updates": 1,
        "new_direct_updates_applied": sum(
            bool(update["update_applied"]) for update in updates
        ),
        "optimizer_updates": 0,
        "checkpoint_written": False,
        "recovery_written": False,
        "final_parameters_retained": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "next_gate": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_repeated_active_set_evidence"
        ),
    }
    _atomic_json(args.output, report)
    return report


def _evaluate_active_set_step(
    *,
    jax: Any,
    loss_evaluator: Any,
    parameters: Any,
    proposal_nonfinite: int,
    before_losses: Sequence[np.ndarray],
    macro_before_losses: Sequence[np.ndarray],
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    prepared_manifest_path: Path,
    signature: Any,
) -> Mapping[str, Any]:
    """Evaluate one proposed state against current and macro baselines."""

    groups = []
    losses_by_group = []
    current_changes = []
    cumulative_changes = []
    hard_failure_count = 0
    nonfinite_value_count = int(proposal_nonfinite)
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            prepared_manifest_path,
            signature,
            group_index,
        )
        losses_after, hard_counts = loss_evaluator(
            parameters,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        jax.block_until_ready((losses_after, hard_counts))
        after = np.asarray(losses_after, dtype=np.float64)
        current = _relative_changes(before_losses[group_index], after)
        cumulative = _relative_changes(macro_before_losses[group_index], after)
        current_values = np.asarray(
            [current[label] for label in GRADIENT_FEASIBILITY_LABELS],
            dtype=np.float64,
        )
        cumulative_values = np.asarray(
            [cumulative[label] for label in GRADIENT_FEASIBILITY_LABELS],
            dtype=np.float64,
        )
        hard_host = _hard_counts_host(hard_counts)
        hard_failure_count += sum(hard_host.values())
        nonfinite_value_count += int(np.sum(~np.isfinite(after)))
        nonfinite_value_count += int(np.sum(~np.isfinite(current_values)))
        nonfinite_value_count += int(np.sum(~np.isfinite(cumulative_values)))
        losses_by_group.append(after)
        current_changes.append(current_values)
        cumulative_changes.append(cumulative_values)
        groups.append(
            {
                "group": group_index,
                "current_loss_relative_changes": current,
                "cumulative_loss_relative_changes": cumulative,
                "hard_gate_counts": hard_host,
            }
        )
        del anchor_batch, context_batch, context_inputs, group
        del losses_after, hard_counts
        gc.collect()
    return {
        "groups": groups,
        "losses": tuple(losses_by_group),
        "current_change_matrix": np.stack(current_changes),
        "cumulative_change_matrix": np.stack(cumulative_changes),
        "hard_failure_count": hard_failure_count,
        "nonfinite_value_count": nonfinite_value_count,
    }


def _change_matrix_summary(
    matrix: np.ndarray,
    *,
    maximum_increase: float,
    minimum_median_decrease: float | None = None,
) -> Mapping[str, Any]:
    medians = {
        label: float(np.median(matrix[:, component_index]))
        for component_index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
    }
    maximum_change = float(np.max(matrix))
    nonincreasing = maximum_change <= maximum_increase
    effect_gate_passed = bool(
        nonincreasing
        and minimum_median_decrease is not None
        and all(value <= -minimum_median_decrease for value in medians.values())
    )
    return {
        "maximum_group_component_relative_increase": maximum_change,
        "component_median_relative_changes": medians,
        "all_group_components_nonincreasing": nonincreasing,
        "effect_gate_passed": effect_gate_passed,
    }


def _reference_attempt(
    update: Mapping[str, Any],
    *,
    step_norm: float,
) -> Mapping[str, Any]:
    matches = [attempt for attempt in update["attempts"] if float(attempt["step_norm"]) == step_norm]
    if len(matches) != 1:
        raise ValueError("repeated active-set reference step drift")
    return matches[0]


def _run_cumulative_microstep_smoke(
    *,
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    repeated_reference: Mapping[str, Any],
    jax: Any,
    runtime: Mapping[str, Any],
    contract: Any,
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    assets: Any,
    adapter_template: Any,
    adapter_timing: Mapping[str, Any],
    signature: Any,
    parameters: Any,
    parameter_count: int,
    long_scales: Any,
    measurement: Any,
    loss_evaluator: Any,
    directional_evaluator: Any,
    memory_before: Mapping[str, Any],
    started: float,
) -> Mapping[str, Any]:
    """Accumulate safe recalibrated microsteps into one useful macro-update."""

    parent_parameters = parameters
    first_step_norm = float(audit["reconstruction"]["first_direct_step_norm"])
    second_step_norm = float(audit["reconstruction"]["second_direct_step_norm"])
    microstep_norm = float(audit["microsteps"]["step_norm"])
    maximum_microsteps = int(audit["microsteps"]["maximum_count"])
    seeds, parent_scalarization = _frozen_scalarization_seeds(
        predecessor,
        expected_step_norm=first_step_norm,
    )
    active_parent_groups = tuple(int(index) for index in np.flatnonzero(np.any(seeds > 0.0, axis=1)))
    combine = _make_combined_direction(jax)
    direct_step = _make_direct_virtual_step(jax)
    count_nonfinite = _make_nonfinite_counter(jax)
    first_gradients = []
    reconstruction_hard_failures = 0
    reconstruction_nonfinite = 0
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        if group_index in active_parent_groups:
            losses, gradient, hard_counts = measurement(
                parameters,
                seeds[group_index],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            gradient_nonfinite = count_nonfinite(gradient)
            jax.block_until_ready((losses, gradient, hard_counts, gradient_nonfinite))
            first_gradients.append(gradient)
            reconstruction_nonfinite += int(np.asarray(gradient_nonfinite))
        else:
            losses, hard_counts = loss_evaluator(
                parameters,
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses, hard_counts))
        reconstruction_hard_failures += sum(_hard_counts_host(hard_counts).values())
        reconstruction_nonfinite += int(np.sum(~np.isfinite(np.asarray(losses, dtype=np.float64))))
        del anchor_batch, context_batch, context_inputs, group, losses, hard_counts
        gc.collect()

    first_direction, first_direction_norm, first_direction_nonfinite = combine(
        tuple(first_gradients),
        np.ones(len(first_gradients), dtype=np.float64),
    )
    first_state, first_proposed_nonfinite = direct_step(
        parameters,
        first_direction,
        first_direction_norm,
        np.asarray(first_step_norm, dtype=np.float64),
    )
    jax.block_until_ready(
        (
            first_direction,
            first_direction_norm,
            first_direction_nonfinite,
            first_state,
            first_proposed_nonfinite,
        )
    )
    reconstruction_nonfinite += int(np.asarray(first_direction_nonfinite))
    reconstruction_nonfinite += int(np.asarray(first_proposed_nonfinite))
    first_direction_norm_error = abs(
        float(np.asarray(first_direction_norm)) - float(parent_scalarization["expected_direction_norm"])
    ) / max(float(parent_scalarization["expected_direction_norm"]), 1.0e-30)
    del first_gradients, first_direction
    gc.collect()

    constraint_labels = tuple(
        f"group_{group_index + 1:02d}.{label}"
        for group_index in range(AUDIT_GROUPS)
        for label in GRADIENT_FEASIBILITY_LABELS
    )
    parent_weights = predecessor["aggregate_direction"]["normalized_simplex_weights"]
    support = tuple(
        label
        for label in constraint_labels
        if float(parent_weights[label]) > float(audit["active_set"]["weight_tolerance"])
    )
    second_recalibrated = _recalibrate_active_set_direction(
        jax=jax,
        measurement=measurement,
        directional_evaluator=directional_evaluator,
        parameters=first_state,
        initial_support=support,
        active_set=audit["active_set"],
        gates=audit["gates"]["direction"],
        manifest=manifest,
        schedule=schedule,
        prepared_manifest_path=args.prepared_manifest,
        signature=signature,
    )
    second_state, second_proposed_nonfinite = direct_step(
        first_state,
        second_recalibrated["direction"],
        second_recalibrated["direction_norm"],
        np.asarray(second_step_norm, dtype=np.float64),
    )
    jax.block_until_ready((second_state, second_proposed_nonfinite))
    second_evaluation = _evaluate_active_set_step(
        jax=jax,
        loss_evaluator=loss_evaluator,
        parameters=second_state,
        proposal_nonfinite=int(np.asarray(second_proposed_nonfinite)),
        before_losses=second_recalibrated["losses"],
        macro_before_losses=second_recalibrated["losses"],
        manifest=manifest,
        schedule=schedule,
        prepared_manifest_path=args.prepared_manifest,
        signature=signature,
    )
    second_summary = _change_matrix_summary(
        second_evaluation["current_change_matrix"],
        maximum_increase=float(audit["gates"]["microstep"]["maximum_group_component_relative_increase"]),
        minimum_median_decrease=float(
            audit["gates"]["macro_effect"]["minimum_each_component_median_relative_decrease"]
        ),
    )
    reconstruction_hard_failures += int(second_recalibrated["hard_failure_count"]) + int(
        second_evaluation["hard_failure_count"]
    )
    reconstruction_nonfinite += int(second_recalibrated["nonfinite_value_count"]) + int(
        second_evaluation["nonfinite_value_count"]
    )

    reference_value_tolerance = float(audit["reconstruction"]["maximum_reference_component_change_error"])
    accepted_reference = repeated_reference["updates"][0]
    accepted_attempt = _reference_attempt(
        accepted_reference,
        step_norm=second_step_norm,
    )
    accepted_component_error = max(
        abs(
            second_summary["component_median_relative_changes"][label]
            - float(accepted_attempt["component_median_relative_changes"][label])
        )
        for label in GRADIENT_FEASIBILITY_LABELS
    )
    second_reference_reproduced = (
        second_recalibrated["direction_passed"]
        and tuple(second_recalibrated["support"]) == tuple(accepted_reference["final_support"])
        and abs(
            float(second_recalibrated["minimum_relative_directional_derivative"])
            - float(accepted_reference["minimum_relative_directional_derivative"])
        )
        <= float(audit["reconstruction"]["maximum_reference_minimum_derivative_error"])
        and accepted_component_error <= reference_value_tolerance
        and abs(
            second_summary["maximum_group_component_relative_increase"]
            - float(accepted_attempt["maximum_group_component_relative_increase"])
        )
        <= reference_value_tolerance
        and bool(accepted_attempt["admissible"])
        and second_summary["effect_gate_passed"]
    )
    macro_before_losses = second_evaluation["losses"]
    parameters = second_state
    support = tuple(second_recalibrated["support"])
    del first_state, second_recalibrated, second_evaluation
    gc.collect()

    reference_stopped = repeated_reference["updates"][1]
    stopped_attempt = _reference_attempt(reference_stopped, step_norm=microstep_norm)
    microsteps = []
    first_microstep_reference_reproduced = False
    macro_effect_passed = False
    for microstep_index in range(1, maximum_microsteps + 1):
        microstep_started = time.perf_counter()
        recalibrated = _recalibrate_active_set_direction(
            jax=jax,
            measurement=measurement,
            directional_evaluator=directional_evaluator,
            parameters=parameters,
            initial_support=support,
            active_set=audit["active_set"],
            gates=audit["gates"]["direction"],
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=args.prepared_manifest,
            signature=signature,
        )
        if not recalibrated["direction_passed"]:
            microsteps.append(
                {
                    "microstep": microstep_index,
                    "update_applied": False,
                    "microstep_safe": False,
                    "direction_passed": False,
                    "minimum_relative_directional_derivative": float(
                        recalibrated["minimum_relative_directional_derivative"]
                    ),
                    "initial_support": list(support),
                    "final_support": list(recalibrated["support"]),
                    "rounds": recalibrated["rounds"],
                    "component_reverse_passes": int(recalibrated["component_reverse_passes"]),
                    "group_jvps": int(recalibrated["group_jvps"]),
                    "added_constraints": list(recalibrated["added_constraints"]),
                    "current": None,
                    "cumulative": None,
                    "groups": None,
                    "hard_failure_count": int(recalibrated["hard_failure_count"]),
                    "nonfinite_value_count": int(recalibrated["nonfinite_value_count"]),
                    "elapsed_seconds": time.perf_counter() - microstep_started,
                }
            )
            break
        proposed, proposed_nonfinite = direct_step(
            parameters,
            recalibrated["direction"],
            recalibrated["direction_norm"],
            np.asarray(microstep_norm, dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposed_nonfinite))
        evaluated = _evaluate_active_set_step(
            jax=jax,
            loss_evaluator=loss_evaluator,
            parameters=proposed,
            proposal_nonfinite=int(np.asarray(proposed_nonfinite)),
            before_losses=recalibrated["losses"],
            macro_before_losses=macro_before_losses,
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=args.prepared_manifest,
            signature=signature,
        )
        current_summary = _change_matrix_summary(
            evaluated["current_change_matrix"],
            maximum_increase=float(audit["gates"]["microstep"]["maximum_group_component_relative_increase"]),
        )
        cumulative_summary = _change_matrix_summary(
            evaluated["cumulative_change_matrix"],
            maximum_increase=float(audit["gates"]["macro_effect"]["maximum_group_component_relative_increase"]),
            minimum_median_decrease=float(
                audit["gates"]["macro_effect"]["minimum_each_component_median_relative_decrease"]
            ),
        )
        hard_failure_count = int(recalibrated["hard_failure_count"]) + int(evaluated["hard_failure_count"])
        nonfinite_value_count = int(recalibrated["nonfinite_value_count"]) + int(evaluated["nonfinite_value_count"])
        microstep_safe = bool(
            current_summary["all_group_components_nonincreasing"]
            and hard_failure_count == int(audit["gates"]["microstep"]["hard_failure_count"])
            and nonfinite_value_count == int(audit["gates"]["microstep"]["nonfinite_value_count"])
        )
        update_applied = microstep_safe
        macro_effect_passed = bool(update_applied and cumulative_summary["effect_gate_passed"])
        record = {
            "microstep": microstep_index,
            "step_norm": microstep_norm,
            "update_applied": update_applied,
            "microstep_safe": microstep_safe,
            "direction_passed": True,
            "minimum_relative_directional_derivative": float(recalibrated["minimum_relative_directional_derivative"]),
            "initial_support": list(support),
            "final_support": list(recalibrated["support"]),
            "rounds": recalibrated["rounds"],
            "component_reverse_passes": int(recalibrated["component_reverse_passes"]),
            "group_jvps": int(recalibrated["group_jvps"]),
            "added_constraints": list(recalibrated["added_constraints"]),
            "current": current_summary,
            "cumulative": cumulative_summary,
            "groups": evaluated["groups"],
            "hard_failure_count": hard_failure_count,
            "nonfinite_value_count": nonfinite_value_count,
            "elapsed_seconds": time.perf_counter() - microstep_started,
        }
        microsteps.append(record)
        if microstep_index == 1:
            component_error = max(
                abs(
                    current_summary["component_median_relative_changes"][label]
                    - float(stopped_attempt["component_median_relative_changes"][label])
                )
                for label in GRADIENT_FEASIBILITY_LABELS
            )
            first_microstep_reference_reproduced = (
                abs(
                    record["minimum_relative_directional_derivative"]
                    - float(reference_stopped["minimum_relative_directional_derivative"])
                )
                <= float(audit["reconstruction"]["maximum_reference_minimum_derivative_error"])
                and tuple(record["final_support"]) == tuple(reference_stopped["final_support"])
                and component_error <= reference_value_tolerance
                and abs(
                    current_summary["maximum_group_component_relative_increase"]
                    - float(stopped_attempt["maximum_group_component_relative_increase"])
                )
                <= reference_value_tolerance
            )
        print(
            json.dumps(
                {
                    "event": "cumulative_microstep",
                    "microstep": microstep_index,
                    "maximum": maximum_microsteps,
                    "applied": update_applied,
                    "macro_effect_passed": macro_effect_passed,
                    "reverse_passes": record["component_reverse_passes"],
                    "jvp_rounds": len(record["rounds"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not update_applied:
            del proposed, evaluated, recalibrated
            gc.collect()
            break
        parameters = proposed
        support = tuple(record["final_support"])
        del evaluated, recalibrated
        gc.collect()
        if macro_effect_passed:
            break

    reconstruction_reference_reproduced = (
        first_direction_norm_error <= float(audit["reconstruction"]["maximum_first_direction_norm_relative_error"])
        and second_reference_reproduced
        and first_microstep_reference_reproduced
    )
    evidence_valid = (
        reconstruction_reference_reproduced
        and reconstruction_hard_failures == int(audit["gates"]["training"]["hard_failure_count"])
        and reconstruction_nonfinite == int(audit["gates"]["training"]["nonfinite_value_count"])
    )
    validation = None
    if macro_effect_passed and evidence_valid:
        from scripts.hpc.prepare_successor_rollout_screen import (
            _prepare_adapter as _prepare_validation_adapter,
        )
        from scripts.hpc.prepare_successor_staged_validation import (
            _reference as _validation_reference,
        )
        from scripts.hpc.run_successor_gradient_balanced_smoke import (
            CANDIDATE as VALIDATION_CANDIDATE,
        )
        from scripts.hpc.run_successor_gradient_balanced_smoke import (
            CONTROL as VALIDATION_BASELINE,
        )
        from scripts.hpc.run_successor_gradient_balanced_smoke import (
            _evaluate_subset,
        )
        from scripts.hpc.successor_staged_validation_prepared import (
            load_staged_validation_prepared,
            ordered_validation_entries,
        )

        validation_prepared, prepared_panel = load_staged_validation_prepared(
            args.validation_prepared_manifest,
            staged_training_contract_sha256=contract.sha256,
            validation_panel_sha256=audit["identity"]["validation_panel_sha256"],
        )
        if prepared_panel != assets.validation_panel:
            raise ValueError("cumulative microstep validation panel drift")
        first_slice, first_validation_entry = ordered_validation_entries(assets.validation_panel)[0]
        validation_reference = _validation_reference(
            runtime["index"],
            first_slice,
            first_validation_entry,
        )
        validation_adapter = _prepare_validation_adapter(
            runtime,
            validation_reference,
            int(first_validation_entry["start_rows"][0]),
        )
        validation = _evaluate_subset(
            runtime,
            contract=contract,
            parameters={
                VALIDATION_BASELINE: parent_parameters,
                VALIDATION_CANDIDATE: parameters,
            },
            long_scales=long_scales,
            adapter_template=validation_adapter,
            prepared_manifest_path=args.validation_prepared_manifest,
            prepared=validation_prepared,
            panel=assets.validation_panel,
        )

    classification = classify_cumulative_microstep_smoke(
        validation,
        microsteps,
        gates=audit["gates"],
        maximum_microsteps=maximum_microsteps,
        pass_action=audit["pass_action"],
        fail_action=audit["fail_action"],
    )
    report = {
        "schema_version": "successor_cumulative_microstep_smoke_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": _git_head(),
        "identity": {
            "smoke_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.smoke_report),
            "repeated_active_set_report_sha256": _sha256_file(args.repeated_active_set_report),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": parameter_count,
        "parent_update": PARENT_UPDATE,
        "reconstruction": {
            "first_direction_norm_relative_error": first_direction_norm_error,
            "second_reference_reproduced": second_reference_reproduced,
            "first_microstep_reference_reproduced": (first_microstep_reference_reproduced),
            "hard_failure_count": reconstruction_hard_failures,
            "nonfinite_value_count": reconstruction_nonfinite,
        },
        "microsteps": microsteps,
        "validation": validation,
        "classification": classification,
        "evidence_gates": {
            "evidence_valid": evidence_valid,
            "reconstruction_reference_reproduced": (reconstruction_reference_reproduced),
        },
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "reconstructed_direct_updates": 2,
        "new_microsteps_applied": sum(bool(step["update_applied"]) for step in microsteps),
        "optimizer_updates": 0,
        "checkpoint_written": False,
        "recovery_written": False,
        "final_parameters_retained": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "next_gate": (
            classification["next_action"] if evidence_valid else "stop_and_repair_cumulative_microstep_evidence"
        ),
    }
    _atomic_json(args.output, report)
    return report


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.geometry_output is not None and args.geometry_output.exists():
        raise FileExistsError(args.geometry_output)
    started = time.perf_counter()
    cumulative_microstep = bool(args.cumulative_microstep_smoke)
    repeated_active_set = bool(args.repeated_active_set_smoke)
    active_set_recalibration = bool(args.active_set_recalibration_audit)
    second_state_recalibration = bool(args.second_state_recalibration_audit)
    backtracking_smoke = bool(args.group_aware_backtracking_scalarization_smoke)
    scalarization_smoke = bool(
        args.frozen_group_aware_scalarization_smoke or backtracking_smoke
    )
    group_aware_extension = bool(args.group_aware_step_extension)
    group_aware = bool(args.group_aware or group_aware_extension)
    if cumulative_microstep:
        audit_schema = "successor_cumulative_microstep_smoke_v1"
        audit_status = "frozen_four_microstep_macro_update_no_checkpoint"
        predecessor_schema = "successor_group_aware_step_extension_v1"
        predecessor_status = "frozen_zero_update_forty_objective_extended_grid"
        predecessor_contract_key = "group_aware_step_extension_contract_sha256"
        predecessor_report_key = "group_aware_step_extension_report_sha256"
    elif repeated_active_set:
        audit_schema = "successor_repeated_active_set_smoke_v1"
        audit_status = "frozen_five_total_direct_updates_no_checkpoint"
        predecessor_schema = "successor_group_aware_step_extension_v1"
        predecessor_status = "frozen_zero_update_forty_objective_extended_grid"
        predecessor_contract_key = "group_aware_step_extension_contract_sha256"
        predecessor_report_key = "group_aware_step_extension_report_sha256"
    elif active_set_recalibration:
        audit_schema = "successor_active_set_recalibration_audit_v1"
        audit_status = "frozen_zero_retention_prior_support_active_set"
        predecessor_schema = "successor_group_aware_step_extension_v1"
        predecessor_status = "frozen_zero_update_forty_objective_extended_grid"
        predecessor_contract_key = "group_aware_step_extension_contract_sha256"
        predecessor_report_key = "group_aware_step_extension_report_sha256"
    elif second_state_recalibration:
        audit_schema = "successor_second_state_recalibration_audit_v1"
        audit_status = "frozen_zero_retention_second_state_exact_recalibration"
        predecessor_schema = "successor_group_aware_step_extension_v1"
        predecessor_status = "frozen_zero_update_forty_objective_extended_grid"
        predecessor_contract_key = "group_aware_step_extension_contract_sha256"
        predecessor_report_key = "group_aware_step_extension_report_sha256"
    elif backtracking_smoke:
        audit_schema = "successor_group_aware_backtracking_smoke_v1"
        audit_status = "frozen_5_update_group_scalarization_backtracking"
        predecessor_schema = "successor_group_aware_step_extension_v1"
        predecessor_status = "frozen_zero_update_forty_objective_extended_grid"
        predecessor_contract_key = "group_aware_step_extension_contract_sha256"
        predecessor_report_key = "group_aware_step_extension_report_sha256"
    elif scalarization_smoke:
        audit_schema = "successor_group_aware_scalarization_smoke_v1"
        audit_status = "frozen_5_update_group_scalarization_no_checkpoint"
        predecessor_schema = "successor_group_aware_step_extension_v1"
        predecessor_status = "frozen_zero_update_forty_objective_extended_grid"
        predecessor_contract_key = "group_aware_step_extension_contract_sha256"
        predecessor_report_key = "group_aware_step_extension_report_sha256"
    elif group_aware_extension:
        audit_schema = "successor_group_aware_step_extension_v1"
        audit_status = "frozen_zero_update_forty_objective_extended_grid"
        predecessor_schema = "successor_group_aware_direction_audit_v1"
        predecessor_status = "frozen_zero_update_forty_objective_grid"
        predecessor_contract_key = "group_aware_direction_contract_sha256"
        predecessor_report_key = "group_aware_direction_report_sha256"
    elif group_aware:
        audit_schema = "successor_group_aware_direction_audit_v1"
        audit_status = "frozen_zero_update_forty_objective_grid"
        predecessor_schema = "successor_multibatch_direction_audit_v1"
        predecessor_status = "frozen_zero_update_accumulated_gradient_grid"
        predecessor_contract_key = "multibatch_direction_contract_sha256"
        predecessor_report_key = "multibatch_direction_report_sha256"
    else:
        audit_schema = "successor_multibatch_direction_audit_v1"
        audit_status = "frozen_zero_update_accumulated_gradient_grid"
        predecessor_schema = "successor_direct_direction_smoke_v1"
        predecessor_status = "frozen_10_update_free_gpu_no_checkpoint"
        predecessor_contract_key = "direct_direction_smoke_contract_sha256"
        predecessor_report_key = "direct_direction_smoke_report_sha256"
    audit = _load_contract(
        args.audit_contract,
        schema_version=audit_schema,
        status=audit_status,
    )
    predecessor = _load_contract(
        args.smoke_contract,
        schema_version=predecessor_schema,
        status=predecessor_status,
    )
    candidate = _load_contract(
        args.candidate_coefficient_contract,
        schema_version="successor_named_biomass_candidate_coefficients_v1",
        status="accepted_zero_update_preflight_no_training_authorized",
    )
    if (
        predecessor["contract_sha256"]
        != audit["identity"][predecessor_contract_key]
        or _sha256_file(args.smoke_report)
        != audit["identity"][predecessor_report_key]
    ):
        raise ValueError("direction-audit predecessor identity drift")
    predecessor_report = (
        json.loads(args.smoke_report.read_text(encoding="utf-8"))
        if scalarization_smoke
        or second_state_recalibration
        or active_set_recalibration
        or repeated_active_set
        or cumulative_microstep
        else None
    )
    repeated_reference = None
    if cumulative_microstep:
        if args.repeated_active_set_report is None:
            raise ValueError("cumulative microstep smoke requires repeated report")
        repeated_reference = json.loads(args.repeated_active_set_report.read_text(encoding="utf-8"))
        if (
            _sha256_file(args.repeated_active_set_report) != audit["identity"]["repeated_active_set_report_sha256"]
            or repeated_reference.get("identity", {}).get("smoke_contract_sha256")
            != audit["identity"]["repeated_active_set_contract_sha256"]
            or repeated_reference.get("classification", {}).get("status") != "repeated_active_set_smoke_not_supported"
            or not repeated_reference.get("evidence_gates", {}).get("evidence_valid", False)
        ):
            raise ValueError("cumulative microstep repeated reference drift")
    active_set_reference = None
    if repeated_active_set:
        if args.active_set_reference_report is None:
            raise ValueError("repeated active-set smoke requires reference report")
        active_set_reference = json.loads(
            args.active_set_reference_report.read_text(encoding="utf-8")
        )
        if (
            _sha256_file(args.active_set_reference_report)
            != audit["identity"]["active_set_reference_report_sha256"]
            or active_set_reference.get("identity", {}).get(
                "audit_contract_sha256"
            )
            != audit["identity"]["active_set_reference_contract_sha256"]
            or active_set_reference.get("classification", {}).get("status")
            != "active_set_recalibration_supported"
        ):
            raise ValueError("repeated active-set reference drift")
    exact_reference = None
    if active_set_recalibration:
        if args.exact_recalibration_report is None:
            raise ValueError("active-set audit requires exact recalibration report")
        exact_reference = json.loads(
            args.exact_recalibration_report.read_text(encoding="utf-8")
        )
        if (
            _sha256_file(args.exact_recalibration_report)
            != audit["identity"]["exact_recalibration_report_sha256"]
            or exact_reference.get("identity", {}).get("audit_contract_sha256")
            != audit["identity"]["exact_recalibration_contract_sha256"]
            or exact_reference.get("classification", {}).get("status")
            != "second_state_active_set_recalibration_supported"
        ):
            raise ValueError("active-set exact reference drift")
    if second_state_recalibration:
        if args.backtracking_report is None:
            raise ValueError("second-state audit requires backtracking report")
        backtracking_report = json.loads(
            args.backtracking_report.read_text(encoding="utf-8")
        )
        if (
            _sha256_file(args.backtracking_report)
            != audit["identity"]["backtracking_report_sha256"]
            or backtracking_report.get("identity", {}).get("smoke_contract_sha256")
            != audit["identity"]["backtracking_contract_sha256"]
            or backtracking_report.get("classification", {}).get("status")
            != "group_aware_backtracking_smoke_rejected"
        ):
            raise ValueError("second-state backtracking predecessor drift")
    if backtracking_smoke:
        if args.failed_scalarization_report is None:
            raise ValueError("backtracking smoke requires failed scalarization report")
        failed_report = json.loads(
            args.failed_scalarization_report.read_text(encoding="utf-8")
        )
        if (
            _sha256_file(args.failed_scalarization_report)
            != audit["identity"]["failed_scalarization_report_sha256"]
            or failed_report.get("identity", {}).get("smoke_contract_sha256")
            != audit["identity"]["failed_scalarization_contract_sha256"]
            or failed_report.get("classification", {}).get("status")
            != "frozen_group_aware_scalarization_smoke_rejected"
        ):
            raise ValueError("backtracking scalarization predecessor drift")

    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        STAGED_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        derive_long_horizon_scales,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        load_named_biomass_tendency_scale_asset,
        load_successor_training_assets,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if (
        contract.payload["schema_version"] != STAGED_SCHEMA_VERSION
        or audit["identity"]["staged_training_contract_sha256"] != contract.sha256
        or audit["identity"]["candidate_coefficient_contract_sha256"]
        != candidate["contract_sha256"]
    ):
        raise ValueError("multibatch direction training identity drift")
    arms = tuple(contract.payload["matched_arms"])
    references = runtime["index"].select(spatial_split="train", temporal_split="train")
    schedule = build_successor_small_fit_schedule(
        references,
        contract,
        updates=int(contract.sampling["maximum_updates_per_arm"]),
    )
    verify_successor_small_fit_schedule(schedule, references, contract)
    start_update, end_update = stage_update_range(
        contract.sampling["stage_milestones"],
        2000,
    )
    manifest = load_staged_prepared_manifest(
        args.prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        full_schedule_sha256=schedule["schedule_sha256"],
        start_update=start_update,
        end_update=end_update,
    )
    if len(manifest["entries"]) < AUDIT_GROUPS * POINTS_PER_MIXED_UPDATE:
        raise ValueError("multibatch direction prepared manifest is too short")
    assets = load_successor_training_assets(
        args.coefficient_contract,
        args.validation_panel,
        coefficient_report_path=args.coefficient_report,
        prepared_manifest_path=args.coefficient_prepared_manifest,
        runner_contract=contract,
        components=LONG_HORIZON_COMPONENTS,
    )
    validation_identity = {
        "manifest_sha256": _sha256_file(args.validation_prepared_manifest)
    }
    scale_asset = load_named_biomass_tendency_scale_asset(
        args.tendency_scale_asset,
        args.tendency_scale_acceptance,
    )
    scale_acceptance = json.loads(
        args.tendency_scale_acceptance.read_text(encoding="utf-8")
    )
    if scale_acceptance["contract_sha256"] != audit["identity"][
        "scale_acceptance_contract_sha256"
    ]:
        raise ValueError("multibatch direction tendency-scale identity drift")

    identity = _training_identity(runtime, contract, schedule, assets, validation_identity)
    first_record = manifest["entries"][0]
    first_entry = schedule["entries"][int(first_record["update"]) - 1]
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        first_record,
        schedule_entry=first_entry,
    )
    _reference, adapter_template, adapter_timing = _prepare_adapter(runtime, first_entry)
    signature = _retained_tail_trace_signature(adapter_template)
    if first_artifact.trace_signature != signature:
        raise ValueError("multibatch direction initial trace drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(lambda value: value[0], first_artifact.batch.anchor_inputs)
    )
    initialized = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(int(contract.sampling["seed"])),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=assembled,
            arm=arm,
        )
        for arm in arms
    }
    counts = {arm: successor_parameter_count(initialized[arm]) for arm in arms}
    if counts.get(SOURCE_ROUTED_ATTENTION) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("multibatch direction Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("multibatch direction requires update-1000 recovery")
    parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    objective_arguments = {
        "runner_contract": contract,
        "long_scales": long_scales,
        "tendency_scales": scale_asset.scales,
        "flux_coefficients": assets.coefficients,
        "tendency_coefficient": float(
            candidate["resolved_coefficients"]["tendency_mse"]
        ),
    }
    measurement = _make_measurement(runtime, adapter_template, **objective_arguments)
    loss_evaluator = _make_loss_evaluator(
        runtime,
        adapter_template,
        **objective_arguments,
    )
    memory_before = _device_memory_report(runtime["devices"])
    if cumulative_microstep:
        directional_evaluator = _make_loss_directional_evaluator(
            runtime,
            adapter_template,
            **objective_arguments,
        )
        return _run_cumulative_microstep_smoke(
            args=args,
            audit=audit,
            predecessor=predecessor_report,
            repeated_reference=repeated_reference,
            jax=jax,
            runtime=runtime,
            contract=contract,
            schedule=schedule,
            manifest=manifest,
            assets=assets,
            adapter_template=adapter_template,
            adapter_timing=adapter_timing,
            signature=signature,
            parameters=parameters,
            parameter_count=counts[SOURCE_ROUTED_ATTENTION],
            long_scales=long_scales,
            measurement=measurement,
            loss_evaluator=loss_evaluator,
            directional_evaluator=directional_evaluator,
            memory_before=memory_before,
            started=started,
        )
    if repeated_active_set:
        directional_evaluator = _make_loss_directional_evaluator(
            runtime,
            adapter_template,
            **objective_arguments,
        )
        return _run_repeated_active_set_smoke(
            args=args,
            audit=audit,
            predecessor=predecessor_report,
            active_set_reference=active_set_reference,
            jax=jax,
            runtime=runtime,
            contract=contract,
            schedule=schedule,
            manifest=manifest,
            assets=assets,
            adapter_template=adapter_template,
            adapter_timing=adapter_timing,
            signature=signature,
            parameters=parameters,
            parameter_count=counts[SOURCE_ROUTED_ATTENTION],
            long_scales=long_scales,
            measurement=measurement,
            loss_evaluator=loss_evaluator,
            directional_evaluator=directional_evaluator,
            memory_before=memory_before,
            started=started,
        )
    if active_set_recalibration:
        directional_evaluator = _make_loss_directional_evaluator(
            runtime,
            adapter_template,
            **objective_arguments,
        )
        return _run_active_set_recalibration_audit(
            args=args,
            audit=audit,
            predecessor=predecessor_report,
            exact_reference=exact_reference,
            jax=jax,
            runtime=runtime,
            schedule=schedule,
            manifest=manifest,
            signature=signature,
            parameters=parameters,
            parameter_count=counts[SOURCE_ROUTED_ATTENTION],
            measurement=measurement,
            loss_evaluator=loss_evaluator,
            directional_evaluator=directional_evaluator,
            adapter_timing=adapter_timing,
            memory_before=memory_before,
            started=started,
        )
    if second_state_recalibration:
        return _run_second_state_recalibration_audit(
            args=args,
            audit=audit,
            predecessor=predecessor_report,
            jax=jax,
            runtime=runtime,
            schedule=schedule,
            manifest=manifest,
            signature=signature,
            parameters=parameters,
            parameter_count=counts[SOURCE_ROUTED_ATTENTION],
            measurement=measurement,
            loss_evaluator=loss_evaluator,
            adapter_timing=adapter_timing,
            memory_before=memory_before,
            started=started,
        )
    if scalarization_smoke:
        return _run_frozen_scalarization_smoke(
            args=args,
            audit=audit,
            predecessor=predecessor_report,
            runtime=runtime,
            jax=jax,
            contract=contract,
            schedule=schedule,
            manifest=manifest,
            assets=assets,
            adapter_template=adapter_template,
            adapter_timing=adapter_timing,
            signature=signature,
            parameters=parameters,
            parameter_count=counts[SOURCE_ROUTED_ATTENTION],
            long_scales=long_scales,
            measurement=measurement,
            loss_evaluator=loss_evaluator,
            memory_before=memory_before,
            started=started,
            backtracking=backtracking_smoke,
        )
    if group_aware:
        flatten = _make_flattened_unit_gradient(jax)
        unit_gradients = []
        accumulators = None
    else:
        accumulate = _make_gradient_accumulator(jax)
        count_nonfinite = _make_nonfinite_counter(jax)
        accumulators = _empty_accumulators(jax, parameters)
    records = []
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        if group_aware:
            record = _collect_group_unit_gradients(
                jax,
                measurement,
                flatten,
                unit_gradients,
                parameters,
                anchor_batch,
                context_batch,
                context_inputs,
            )
        else:
            record = _accumulate_group_gradients(
                jax,
                measurement,
                accumulate,
                count_nonfinite,
                accumulators,
                parameters,
                anchor_batch,
                context_batch,
                context_inputs,
            )
        records.append(
            {
                "group": group_index,
                "point_years": [
                    {
                        "update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                **record,
                "steps": {},
            }
        )
        del anchor_batch, context_batch, context_inputs, group
        gc.collect()
        print(
            json.dumps(
                {
                    "event": (
                        "group_aware_direction_gradient_group"
                        if group_aware
                        else "multibatch_direction_gradient_group"
                    ),
                    "group": group_index + 1,
                    "total": AUDIT_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if group_aware:
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree

        constraint_labels = tuple(
            f"group_{group_index + 1:02d}.{label}"
            for group_index in range(AUDIT_GROUPS)
            for label in GRADIENT_FEASIBILITY_LABELS
        )
        unit_matrix = jnp.stack(tuple(unit_gradients))
        flat_gram = _make_flat_gram(jax)
        gram = flat_gram(unit_matrix)
        jax.block_until_ready(gram)
        gram_host = np.asarray(gram, dtype=np.float64)
        if args.geometry_output is not None:
            _atomic_json(
                args.geometry_output,
                {
                    "schema_version": "successor_group_aware_gradient_geometry_v1",
                    "status": "completed_pre_solver",
                    "git_head": _git_head(),
                    "audit_contract_sha256": audit["contract_sha256"],
                    "constraint_labels": list(constraint_labels),
                    "normalized_gram_matrix": gram_host.tolist(),
                    "groups": [
                        {
                            "group": record["group"],
                            "point_years": record["point_years"],
                            "losses_before": record["losses_before"],
                            "gradient_norms": record["gradient_norms"],
                            "parent_hard_gate_counts": record[
                                "parent_hard_gate_counts"
                            ],
                            "nonfinite_component_gradient_values": record[
                                "nonfinite_component_gradient_values"
                            ],
                        }
                        for record in records
                    ],
                    "optimizer_updates": 0,
                    "parameters_retained": False,
                },
            )
        diagnostics = gradient_direction_diagnostics(
            gram_host,
            labels=constraint_labels,
        )
        selected = diagnostics["directions"]["minimum_norm_unit"]
        weights = np.asarray(
            [selected["weights"][label] for label in constraint_labels],
            dtype=np.float64,
        )
        flat_direction = _make_flat_direction(jax)
        direction_values, direction_norm, direction_nonfinite = flat_direction(
            unit_matrix,
            weights,
        )
        _flat_parameters, unravel = ravel_pytree(parameters)
        direction = unravel(direction_values)
        aggregate_nonfinite = np.asarray((0,), dtype=np.int64)
        del unit_gradients, unit_matrix, direction_values
        gc.collect()
    else:
        gram_measurement = _make_gradient_gram(jax)
        gram, aggregate_nonfinite = gram_measurement(tuple(accumulators))
        jax.block_until_ready((gram, aggregate_nonfinite))
        diagnostics = gradient_direction_diagnostics(np.asarray(gram, dtype=np.float64))
        selected = diagnostics["directions"]["minimum_norm_unit"]
        weights = np.asarray(
            [selected["weights"][label] for label in GRADIENT_FEASIBILITY_LABELS],
            dtype=np.float64,
        )
        combine = _make_combined_direction(jax)
        direction, direction_norm, direction_nonfinite = combine(
            tuple(accumulators),
            weights,
        )
    jax.block_until_ready((direction, direction_norm, direction_nonfinite))
    direct_step = _make_direct_virtual_step(jax)
    step_norms = tuple(float(value) for value in audit["step_norm_grid"])
    minimum_direction_cosine = min(
        selected["normalized_directional_derivatives"].values()
    )
    evaluate_steps = not group_aware or (
        bool(selected["all_strictly_positive"])
        and minimum_direction_cosine
        >= float(audit["direction_gate"]["minimum_group_component_cosine"])
    )
    proposals = {}
    proposal_nonfinite = {}
    if evaluate_steps:
        for step_norm in step_norms:
            proposed, nonfinite = direct_step(
                parameters,
                direction,
                direction_norm,
                np.asarray(step_norm, dtype=np.float64),
            )
            jax.block_until_ready((proposed, nonfinite))
            key = format(step_norm, ".12g")
            proposals[key] = proposed
            proposal_nonfinite[key] = int(np.asarray(nonfinite))

    for group_index in range(AUDIT_GROUPS if evaluate_steps else 0):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        before = np.asarray(
            [
                records[group_index]["losses_before"][label]
                for label in GRADIENT_FEASIBILITY_LABELS
            ],
            dtype=np.float64,
        )
        for step_norm in step_norms:
            key = format(step_norm, ".12g")
            losses_after, hard_counts = loss_evaluator(
                proposals[key],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses_after, hard_counts))
            after = np.asarray(losses_after, dtype=np.float64)
            records[group_index]["steps"][key] = {
                "step_norm": step_norm,
                "virtual_losses_after": {
                    label: float(after[index])
                    for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
                },
                "virtual_loss_relative_changes": _relative_changes(before, after),
                "hard_gate_counts": _hard_counts_host(hard_counts),
                "nonfinite_proposed_parameter_values": proposal_nonfinite[key],
            }
            del losses_after, hard_counts
        del anchor_batch, context_batch, context_inputs, group
        gc.collect()
        print(
            json.dumps(
                {
                    "event": (
                        "group_aware_direction_virtual_group"
                        if group_aware
                        else "multibatch_direction_virtual_group"
                    ),
                    "group": group_index + 1,
                    "total": AUDIT_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    hard_failure_count = sum(
        sum(record["parent_hard_gate_counts"].values())
        + sum(
            sum(step["hard_gate_counts"].values())
            for step in record["steps"].values()
        )
        for record in records
    )
    component_nonfinite_count = sum(
        sum(record["nonfinite_component_gradient_values"].values())
        for record in records
    )
    nonfinite_count = (
        component_nonfinite_count
        + int(np.sum(np.asarray(aggregate_nonfinite, dtype=np.int64)))
        + int(np.asarray(direction_nonfinite))
        + sum(proposal_nonfinite.values())
    )
    all_losses_finite = all(
        all(np.isfinite(value) for value in record["losses_before"].values())
        and all(
            all(np.isfinite(value) for value in step["virtual_losses_after"].values())
            for step in record["steps"].values()
        )
        for record in records
    )
    if group_aware:
        classification = classify_group_aware_direction(
            diagnostics,
            records,
            step_norms=step_norms,
            contract=audit,
        )
    else:
        classification = classify_direct_trust_region(
            records,
            step_norms=step_norms,
            gates=audit["gates"],
        )
        classification = dict(classification)
        classification["next_action"] = (
            audit["pass_action"] if classification["passed"] else audit["fail_action"]
        )
    evidence_valid = (
        hard_failure_count == int(audit["gates"]["hard_failure_count"])
        and nonfinite_count == int(audit["gates"]["nonfinite_value_count"])
        and all_losses_finite
    )
    report = {
        "schema_version": (
            "successor_group_aware_step_extension_report_v1"
            if group_aware_extension
            else "successor_group_aware_direction_audit_report_v1"
            if group_aware
            else "successor_multibatch_direction_audit_report_v1"
        ),
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": _git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.smoke_report),
            "staged_training_contract_sha256": contract.sha256,
            "geometry_sidecar_sha256": (
                _sha256_file(args.geometry_output)
                if args.geometry_output is not None
                else None
            ),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "gradient_aggregation": audit["direction"],
        "group_aware": group_aware,
        "group_aware_step_extension": group_aware_extension,
        "step_norm_grid": list(step_norms),
        "aggregate_direction": {
            "gradient_norms": diagnostics["gradient_norms"],
            "direction_norm": float(np.asarray(direction_norm)),
            "minimum_component_cosine": minimum_direction_cosine,
            "component_cosines": selected["normalized_directional_derivatives"],
            "normalized_simplex_weights": selected["normalized_simplex_weights"],
            "raw_gradient_coefficients": selected["weights"],
        },
        "classification": classification,
        "evidence_gates": {
            "evidence_valid": evidence_valid,
            "all_losses_finite": all_losses_finite,
            "hard_failure_count": hard_failure_count,
            "nonfinite_value_count": nonfinite_count,
        },
        "records": records,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_updates": 0,
        "proposed_parameters_retained": False,
        "checkpoint_written": False,
        "recovery_written": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "next_gate": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_direction_evidence"
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
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--audit-contract", type=Path, required=True)
    parser.add_argument("--smoke-contract", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--candidate-coefficient-contract", type=Path, required=True)
    parser.add_argument("--tendency-scale-asset", type=Path, required=True)
    parser.add_argument("--tendency-scale-acceptance", type=Path, required=True)
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-report", type=Path, required=True)
    parser.add_argument("--coefficient-prepared-manifest", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--recovery-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--group-aware", action="store_true")
    mode.add_argument("--group-aware-step-extension", action="store_true")
    mode.add_argument("--frozen-group-aware-scalarization-smoke", action="store_true")
    mode.add_argument(
        "--group-aware-backtracking-scalarization-smoke", action="store_true"
    )
    mode.add_argument("--second-state-recalibration-audit", action="store_true")
    mode.add_argument("--active-set-recalibration-audit", action="store_true")
    mode.add_argument("--repeated-active-set-smoke", action="store_true")
    mode.add_argument("--cumulative-microstep-smoke", action="store_true")
    parser.add_argument("--failed-scalarization-report", type=Path)
    parser.add_argument("--backtracking-report", type=Path)
    parser.add_argument("--exact-recalibration-report", type=Path)
    parser.add_argument("--active-set-reference-report", type=Path)
    parser.add_argument("--repeated-active-set-report", type=Path)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
