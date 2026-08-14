"""Run the bounded 100-update current-vs-named-biomass objective A/B."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
)
from research.daily_coarse_graining.daily_operator.successor_rollout_screen import (
    add_rollout_screen_statistics,
    empty_rollout_screen_statistics,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from research.daily_coarse_graining.daily_operator.successor_staged_training import (
    REQUIRED_VALIDATION_HORIZONS,
    REQUIRED_VALIDATION_SLICES,
    finalize_staged_validation_statistics,
    validation_hard_failure_count,
)
from scripts.hpc.prepare_successor_rollout_screen import (
    _prepare_adapter as _prepare_validation_adapter,
)
from scripts.hpc.prepare_successor_staged_validation import (
    _reference as _validation_reference,
)
from scripts.hpc.probe_successor_batch_diversity import (
    POINTS_PER_MIXED_UPDATE,
    _join_context_anchors,
    _slice_context_batch,
    _stack_contexts,
)
from scripts.hpc.probe_successor_named_biomass_gradients import (
    HORIZONS,
    _make_context_trajectory,
)
from scripts.hpc.run_successor_batch_diversity_short_fit import (
    _make_mixed_train_step,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_rollout_screen import _make_evaluator
from scripts.hpc.run_successor_small_fit import (
    ArmUpdateResult,
    _device_memory_report,
    _hard_counts_host,
    _make_arm_objectives,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.run_successor_staged_training import _training_identity
from scripts.hpc.successor_rollout_screen_prepared import (
    load_prepared_rollout_artifact,
)
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)
from scripts.hpc.successor_staged_validation_prepared import (
    load_staged_validation_prepared,
    ordered_validation_entries,
)

PARENT_UPDATE = 1000
SHORT_UPDATES = 100
OBJECTIVES = ("current_objective_control", "named_biomass_candidate")
RESEARCH_PRIMARY_KEYS = (
    "cumulative.npp",
    "terminal.aboveground_biomass_carbon",
    "terminal.belowground_biomass_carbon",
    "terminal.total_live_biomass_carbon",
)
MAINTENANCE_KEY = "cumulative.maintenance_respiration"


class CandidateUpdateResult(NamedTuple):
    parameters: Any
    optimizer: Any
    loss: Any
    anchor: Any
    components: Any
    gradient_norm: Any
    nonfinite_gradient_values: Any
    hard_counts: Any
    update_applied: Any


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("contract_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_contract(
    path: Path,
    *,
    schema_version: str,
    status: str,
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != schema_version
        or payload.get("status") != status
        or payload.get("contract_sha256") != _canonical_sha256(payload)
    ):
        raise ValueError(f"contract drift: {schema_version}")
    return payload


def _make_candidate_train_step(
    runtime: Mapping[str, Any],
    adapter_template: Any,
    *,
    long_scales: Any,
    tendency_scales: Mapping[int, Mapping[str, float]],
    coefficients: Mapping[str, float],
    runner_contract: Any,
) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_adamw_update,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        CANDIDATE_OBJECTIVE_COMPONENTS,
        candidate_successor_training_objective,
    )

    jax = runtime["jax"]
    anchor_objective, _rollout_objective = _make_arm_objectives(
        runtime,
        SOURCE_ROUTED_ATTENTION,
        adapter_template,
        long_scales,
    )
    context_trajectory = _make_context_trajectory(
        runtime,
        adapter_template,
        runner_contract,
    )

    def objective(
        parameters: Any,
        anchor_batch: Any,
        context_batches: Any,
        day_inputs: Any,
    ) -> Any:
        anchor = anchor_objective(parameters, anchor_batch)
        predicted, expected, counts = jax.vmap(
            context_trajectory,
            in_axes=(None, 0, 0),
        )(parameters, context_batches, day_inputs)
        hard_counts = jax.tree_util.tree_map(
            lambda values: jnp.sum(values, axis=0),
            counts,
        )
        actual = jax.tree_util.tree_map(
            lambda value: jnp.swapaxes(value, 0, 1),
            predicted,
        )
        wanted = jax.tree_util.tree_map(
            lambda value: jnp.swapaxes(value, 0, 1),
            expected,
        )
        result = candidate_successor_training_objective(
            anchor,
            actual,
            wanted,
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            tendency_scales=tendency_scales,
            coefficients=coefficients,
            horizons=HORIZONS,
        )
        components = jnp.stack(
            tuple(getattr(result, name) for name in CANDIDATE_OBJECTIVE_COMPONENTS)
        )
        return result.total, (anchor, components, hard_counts)

    @jax.jit
    def train_step(
        parameters: Any,
        optimizer: Any,
        anchor_batch: Any,
        context_batches: Any,
        day_inputs: Any,
        learning_rate: Any,
    ) -> CandidateUpdateResult:
        (loss, auxiliary), gradients = jax.value_and_grad(
            lambda values: objective(
                values,
                anchor_batch,
                context_batches,
                day_inputs,
            ),
            has_aux=True,
        )(parameters)
        anchor, components, hard_counts = auxiliary
        next_parameters, next_optimizer, gradient_norm = successor_adamw_update(
            parameters,
            gradients,
            optimizer,
            learning_rate,
            runner_contract,
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)))
                for leaf in jax.tree_util.tree_leaves(gradients)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        hard_total = sum(
            (jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(hard_counts)),
            jnp.asarray(0, dtype=jnp.int64),
        )
        applied = jnp.isfinite(loss) & (nonfinite == 0) & (hard_total == 0)
        admitted_parameters = jax.tree_util.tree_map(
            lambda proposed, current: jnp.where(applied, proposed, current),
            next_parameters,
            parameters,
        )
        admitted_optimizer = jax.tree_util.tree_map(
            lambda proposed, current: jnp.where(applied, proposed, current),
            next_optimizer,
            optimizer,
        )
        return CandidateUpdateResult(
            admitted_parameters,
            admitted_optimizer,
            loss,
            anchor,
            components,
            gradient_norm,
            nonfinite,
            hard_counts,
            applied,
        )

    return train_step


def _evaluate_objectives(
    runtime: Mapping[str, Any],
    *,
    contract: Any,
    parameters: Mapping[str, Any],
    long_scales: Any,
    adapter_template: Any,
    prepared_manifest_path: Path,
    prepared: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    evaluator = _make_evaluator(
        runtime,
        SOURCE_ROUTED_ATTENTION,
        adapter_template,
        long_scales,
        contract,
    )
    accumulators = {
        objective: {
            slice_id: empty_rollout_screen_statistics()
            for slice_id in REQUIRED_VALIDATION_SLICES
        }
        for objective in OBJECTIVES
    }
    ordered = ordered_validation_entries(panel)
    for ordinal, (record, (slice_id, entry)) in enumerate(
        zip(prepared["entries"], ordered, strict=True),
        start=1,
    ):
        artifact = load_prepared_rollout_artifact(
            prepared_manifest_path,
            record,
            slice_id=slice_id,
            window_entry=entry,
        )
        if artifact.trace_signature != _retained_tail_trace_signature(adapter_template):
            raise ValueError("named-biomass A/B validation retained-tail trace drift")
        for objective in OBJECTIVES:
            statistics = evaluator(
                parameters[objective],
                artifact.batch,
                artifact.day_inputs,
            )
            jax.block_until_ready(statistics)
            accumulators[objective][slice_id] = add_rollout_screen_statistics(
                accumulators[objective][slice_id],
                jax.device_get(statistics),
            )
        print(
            json.dumps(
                {
                    "event": "named_biomass_objective_ab_validation",
                    "ordinal": ordinal,
                    "total": len(ordered),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        objective: {
            "slices": {
                slice_id: finalize_staged_validation_statistics(
                    accumulators[objective][slice_id],
                    primary_outputs=contract.payload["validation"]["primary_outputs"],
                )
                for slice_id in REQUIRED_VALIDATION_SLICES
            }
        }
        for objective in OBJECTIVES
    }


def _rms(values: list[float]) -> float:
    result = float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))
    if not np.isfinite(result):
        raise ValueError("named-biomass A/B metric is nonfinite")
    return result


def _relative_change(candidate: float, control: float) -> float | None:
    if control == 0.0:
        return 0.0 if candidate == 0.0 else None
    return candidate / control - 1.0


def classify_objective_ab(
    validation: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> Mapping[str, Any]:
    control = validation["current_objective_control"]
    candidate = validation["named_biomass_candidate"]
    cells = tuple(
        (slice_id, str(horizon))
        for slice_id in REQUIRED_VALIDATION_SLICES
        for horizon in REQUIRED_VALIDATION_HORIZONS
    )

    key_aggregates = {}
    for key in RESEARCH_PRIMARY_KEYS:
        control_value = _rms(
            [
                float(control["slices"][slice_id]["horizons"][horizon]["keys"][key]["normalized_rmse"])
                for slice_id, horizon in cells
            ]
        )
        candidate_value = _rms(
            [
                float(candidate["slices"][slice_id]["horizons"][horizon]["keys"][key]["normalized_rmse"])
                for slice_id, horizon in cells
            ]
        )
        key_aggregates[key] = {
            "control_rms": control_value,
            "candidate_rms": candidate_value,
            "relative_change": _relative_change(candidate_value, control_value),
            "no_worse": candidate_value <= control_value,
        }

    control_primary_values = []
    candidate_primary_values = []
    cell_records = []
    for slice_id, horizon in cells:
        control_cell = control["slices"][slice_id]["horizons"][horizon]["keys"]
        candidate_cell = candidate["slices"][slice_id]["horizons"][horizon]["keys"]
        control_value = _rms(
            [float(control_cell[key]["normalized_rmse"]) for key in RESEARCH_PRIMARY_KEYS]
        )
        candidate_value = _rms(
            [float(candidate_cell[key]["normalized_rmse"]) for key in RESEARCH_PRIMARY_KEYS]
        )
        control_primary_values.extend(
            float(control_cell[key]["normalized_rmse"])
            for key in RESEARCH_PRIMARY_KEYS
        )
        candidate_primary_values.extend(
            float(candidate_cell[key]["normalized_rmse"])
            for key in RESEARCH_PRIMARY_KEYS
        )
        cell_records.append(
            {
                "slice": slice_id,
                "horizon_days": int(horizon),
                "control_composite": control_value,
                "candidate_composite": candidate_value,
                "relative_change": _relative_change(candidate_value, control_value),
                "improved": candidate_value < control_value,
            }
        )

    control_composite = _rms(control_primary_values)
    candidate_composite = _rms(candidate_primary_values)
    composite_improvement = 1.0 - candidate_composite / control_composite

    maintenance_cells = []
    for slice_id, horizon in cells:
        control_value = float(
            control["slices"][slice_id]["horizons"][horizon]["keys"][MAINTENANCE_KEY]["normalized_rmse"]
        )
        candidate_value = float(
            candidate["slices"][slice_id]["horizons"][horizon]["keys"][MAINTENANCE_KEY]["normalized_rmse"]
        )
        maintenance_cells.append(
            {
                "slice": slice_id,
                "horizon_days": int(horizon),
                "control_rmse": control_value,
                "candidate_rmse": candidate_value,
                "relative_change": _relative_change(candidate_value, control_value),
                "within_regression_limit": candidate_value
                <= control_value
                * (1.0 + float(gates["maximum_maintenance_single_cell_relative_regression"])),
            }
        )
    control_maintenance = _rms([item["control_rmse"] for item in maintenance_cells])
    candidate_maintenance = _rms(
        [item["candidate_rmse"] for item in maintenance_cells]
    )

    hard_failures = {
        objective: validation_hard_failure_count(validation[objective])
        for objective in OBJECTIVES
    }
    checks = {
        "hard_failures_zero": all(value == 0 for value in hard_failures.values()),
        "research_primary_composite_improved": composite_improvement
        >= float(gates["minimum_research_primary_composite_relative_improvement"]),
        "each_research_primary_key_no_worse": all(
            record["no_worse"] for record in key_aggregates.values()
        ),
        "cell_majority_improved": sum(item["improved"] for item in cell_records)
        >= int(gates["minimum_improved_slice_horizon_cells"]),
        "maintenance_aggregate_within_limit": candidate_maintenance
        <= control_maintenance
        * (1.0 + float(gates["maximum_maintenance_aggregate_relative_regression"])),
        "maintenance_cells_within_limit": all(
            item["within_regression_limit"] for item in maintenance_cells
        ),
    }
    passed = all(checks.values())
    return {
        "status": "named_biomass_objective_admitted" if passed else "named_biomass_objective_rejected",
        "passed": passed,
        "checks": checks,
        "hard_failure_counts": hard_failures,
        "research_primary": {
            "control_composite": control_composite,
            "candidate_composite": candidate_composite,
            "relative_improvement": composite_improvement,
            "key_aggregates": key_aggregates,
            "cells": cell_records,
            "improved_cell_count": sum(item["improved"] for item in cell_records),
        },
        "maintenance_guard": {
            "control_aggregate_rms": control_maintenance,
            "candidate_aggregate_rms": candidate_maintenance,
            "relative_change": _relative_change(candidate_maintenance, control_maintenance),
            "cells": maintenance_cells,
        },
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    ab_contract = _load_contract(
        args.ab_contract,
        schema_version="successor_named_biomass_objective_ab_v1",
        status="frozen_user_approved_free_gpu_no_checkpoint",
    )
    candidate_contract = _load_contract(
        args.candidate_coefficient_contract,
        schema_version="successor_named_biomass_candidate_coefficients_v1",
        status="accepted_zero_update_preflight_no_training_authorized",
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
        successor_learning_rate,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        CANDIDATE_OBJECTIVE_COMPONENTS,
        LONG_HORIZON_COMPONENTS,
        derive_long_horizon_scales,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        load_named_biomass_tendency_scale_asset,
        load_successor_training_assets,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if contract.payload["schema_version"] != STAGED_SCHEMA_VERSION:
        raise ValueError("named-biomass A/B requires staged-v3")
    if (
        ab_contract["identity"]["staged_training_contract_sha256"] != contract.sha256
        or ab_contract["identity"]["candidate_coefficient_contract_sha256"]
        != candidate_contract["contract_sha256"]
    ):
        raise ValueError("named-biomass A/B contract identity drift")
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
    if len(manifest["entries"]) < SHORT_UPDATES * POINTS_PER_MIXED_UPDATE:
        raise ValueError("named-biomass A/B prepared manifest is too short")
    validation_prepared, prepared_panel = load_staged_validation_prepared(
        args.validation_prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        validation_panel_sha256=args.validation_panel_sha256,
    )
    assets = load_successor_training_assets(
        args.coefficient_contract,
        args.validation_panel,
        coefficient_report_path=args.coefficient_report,
        prepared_manifest_path=args.coefficient_prepared_manifest,
        runner_contract=contract,
        components=LONG_HORIZON_COMPONENTS,
    )
    if prepared_panel != assets.validation_panel:
        raise ValueError("named-biomass A/B validation panel drift")
    scale_asset = load_named_biomass_tendency_scale_asset(
        args.tendency_scale_asset,
        args.tendency_scale_acceptance,
    )

    identity = _training_identity(runtime, contract, schedule, assets, validation_prepared)
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
        raise ValueError("named-biomass A/B retained-tail trace drift")
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
        raise ValueError("named-biomass A/B Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("named-biomass A/B requires update-1000 recovery")
    parent_parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
    parent_optimizer = recovery["optimizers"][SOURCE_ROUTED_ATTENTION]
    parameters = {objective: parent_parameters for objective in OBJECTIVES}
    optimizers = {objective: parent_optimizer for objective in OBJECTIVES}

    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    control_step = _make_mixed_train_step(
        runtime,
        SOURCE_ROUTED_ATTENTION,
        adapter_template,
        assets.coefficients,
        long_scales,
        contract,
    )
    resolved = candidate_contract["resolved_coefficients"]
    candidate_coefficients = {
        name: float(resolved[name]) for name in CANDIDATE_OBJECTIVE_COMPONENTS
    }
    candidate_step = _make_candidate_train_step(
        runtime,
        adapter_template,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        coefficients=candidate_coefficients,
        runner_contract=contract,
    )
    steps = {
        "current_objective_control": control_step,
        "named_biomass_candidate": candidate_step,
    }
    component_names = {
        "current_objective_control": LONG_HORIZON_COMPONENTS,
        "named_biomass_candidate": CANDIDATE_OBJECTIVE_COMPONENTS,
    }
    clip_counts = {objective: 0 for objective in OBJECTIVES}
    records = []
    memory_before = _device_memory_report(runtime["devices"])
    for short_update in range(1, SHORT_UPDATES + 1):
        offset = (short_update - 1) * POINTS_PER_MIXED_UPDATE
        group = []
        for group_index in range(POINTS_PER_MIXED_UPDATE):
            record = manifest["entries"][offset + group_index]
            entry = schedule["entries"][int(record["update"]) - 1]
            artifact = load_prepared_artifact(
                args.prepared_manifest,
                record,
                schedule_entry=entry,
            )
            if artifact.trace_signature != signature:
                raise ValueError("named-biomass A/B prepared trace drift")
            group.append((entry, artifact))
        point_years = {
            (str(entry["landpoint_id"]), int(entry["year"]))
            for entry, _artifact in group
        }
        if len(point_years) != POINTS_PER_MIXED_UPDATE:
            raise ValueError("named-biomass A/B group repeated a point-year")
        contexts = tuple(
            _slice_context_batch(jax, artifact.batch, context_index=index)
            for index, (_entry, artifact) in enumerate(group)
        )
        anchor_batch = _join_context_anchors(jax, contexts)
        context_batch = _stack_contexts(jax, contexts)
        context_inputs = _stack_contexts(
            jax,
            tuple(artifact.day_inputs for _entry, artifact in group),
        )
        absolute_update = PARENT_UPDATE + short_update
        learning_rate = np.asarray(
            successor_learning_rate(contract, absolute_update),
            dtype=np.float64,
        )
        objective_records = {}
        for objective in OBJECTIVES:
            result: ArmUpdateResult | CandidateUpdateResult = steps[objective](
                parameters[objective],
                optimizers[objective],
                anchor_batch,
                context_batch,
                context_inputs,
                learning_rate,
            )
            jax.block_until_ready(result)
            hard_counts = _hard_counts_host(result.hard_counts)
            if not bool(np.asarray(result.update_applied)):
                raise RuntimeError(
                    f"named-biomass A/B update {short_update} failed for "
                    f"{objective}: {hard_counts}"
                )
            parameters[objective] = result.parameters
            optimizers[objective] = result.optimizer
            gradient_norm = float(np.asarray(result.gradient_norm))
            clip_counts[objective] += int(
                gradient_norm > float(contract.optimizer["global_gradient_norm_clip"])
            )
            objective_records[objective] = {
                "loss": float(np.asarray(result.loss)),
                "one_day_anchor": float(np.asarray(result.anchor)),
                "components": {
                    name: float(value)
                    for name, value in zip(
                        component_names[objective],
                        np.asarray(result.components),
                        strict=True,
                    )
                },
                "gradient_norm": gradient_norm,
                "hard_gate_counts": hard_counts,
            }
        records.append(
            {
                "short_update": short_update,
                "optimizer_update": absolute_update,
                "learning_rate": float(learning_rate),
                "point_years": [
                    {
                        "update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                "objectives": objective_records,
            }
        )
        del group, contexts, anchor_batch, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "named_biomass_objective_ab_update",
                    "update": short_update,
                    "total": SHORT_UPDATES,
                },
                sort_keys=True,
            ),
            flush=True,
        )

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
    validation = _evaluate_objectives(
        runtime,
        contract=contract,
        parameters=parameters,
        long_scales=long_scales,
        adapter_template=validation_adapter,
        prepared_manifest_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=assets.validation_panel,
    )
    classification = classify_objective_ab(
        validation,
        gates=ab_contract["gates"],
    )
    report = {
        "schema_version": "successor_named_biomass_objective_ab_report_v1",
        "status": "completed",
        "git_head": _git_head(),
        "identity": {
            "ab_contract_sha256": ab_contract["contract_sha256"],
            "staged_training_contract_sha256": contract.sha256,
            "candidate_coefficient_contract_sha256": candidate_contract[
                "contract_sha256"
            ],
            "tendency_scale_asset_sha256": scale_asset.asset_sha256,
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "updates_per_objective": SHORT_UPDATES,
        "same_parent_parameters_and_optimizer": True,
        "same_training_groups_and_learning_rates": True,
        "batch": {
            "point_years_per_update": POINTS_PER_MIXED_UPDATE,
            "one_day_anchors": 32,
            "rollout_windows": 4,
            "rollout_days": 30,
        },
        "classification": classification,
        "gradient_clip_counts_report_only": clip_counts,
        "updates": records,
        "validation": validation,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_written": False,
        "recovery_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "reconstructs_48_forcing_records": False,
        "executes_48_step_state_scan": False,
        "next_gate": (
            ab_contract["pass_action"]
            if classification["passed"]
            else ab_contract["fail_action"]
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
    parser.add_argument("--ab-contract", type=Path, required=True)
    parser.add_argument("--candidate-coefficient-contract", type=Path, required=True)
    parser.add_argument("--tendency-scale-asset", type=Path, required=True)
    parser.add_argument("--tendency-scale-acceptance", type=Path, required=True)
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-report", type=Path, required=True)
    parser.add_argument("--coefficient-prepared-manifest", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-panel-sha256", required=True)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--recovery-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
