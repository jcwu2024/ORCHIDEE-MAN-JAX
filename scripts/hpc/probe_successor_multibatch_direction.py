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
        proposed, proposed_nonfinite = direct_step(
            parameters,
            direction,
            direction_norm,
            np.asarray(step_norm, dtype=np.float64),
        )
        jax.block_until_ready(
            (direction, direction_norm, direction_nonfinite, proposed, proposed_nonfinite)
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
            if short_update == 1:
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
                    "proposed_hard_gate_counts": _hard_counts_host(hard_counts_after),
                }
            )
            del anchor_batch, context_batch, context_inputs, group
            del losses_after, hard_counts_after
            gc.collect()

        direction_norm_host = float(np.asarray(direction_norm))
        direction_norm_error = abs(
            direction_norm_host - float(scalarization["expected_direction_norm"])
        ) / max(float(scalarization["expected_direction_norm"]), 1.0e-30)
        losses_nonfinite = sum(
            int(
                np.count_nonzero(
                    ~np.isfinite(tuple(group_record[loss_key].values()))
                )
            )
            for group_record in group_records
            for loss_key in ("losses_before", "losses_after", "loss_relative_changes")
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
        parent_equivalence = (
            {
                "direction_norm_relative_error": direction_norm_error,
                "maximum_virtual_relative_change_error": virtual_error,
            }
            if short_update == 1
            else None
        )
        update_applied = (
            hard_failures == 0
            and sum(nonfinite.values()) == 0
            and maximum_change
            <= float(audit["gates"]["maximum_group_component_relative_increase"])
            and (
                short_update != 1
                or (
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
            )
        )
        updates.append(
            {
                "short_update": short_update,
                "update_applied": update_applied,
                "step_norm": step_norm,
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
    report = {
        "schema_version": "successor_group_aware_scalarization_smoke_report_v1",
        "status": "completed",
        "git_head": _git_head(),
        "identity": {
            "smoke_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.smoke_report),
            "staged_training_contract_sha256": contract.sha256,
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": parameter_count,
        "parent_update": PARENT_UPDATE,
        "step_norm": step_norm,
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


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.geometry_output is not None and args.geometry_output.exists():
        raise FileExistsError(args.geometry_output)
    started = time.perf_counter()
    scalarization_smoke = bool(args.frozen_group_aware_scalarization_smoke)
    group_aware_extension = bool(args.group_aware_step_extension)
    group_aware = bool(args.group_aware or group_aware_extension)
    if scalarization_smoke:
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
        else None
    )

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
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
