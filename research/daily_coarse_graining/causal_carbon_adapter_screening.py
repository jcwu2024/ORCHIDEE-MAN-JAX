"""Frozen all-sample screening for the matched Experiment C adapter arms."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining import rollout_stability_screening as shared
from research.daily_coarse_graining.canonical_multistep_training_run import (
    _make_runtime_and_compiled_forcing,
)
from research.daily_coarse_graining.causal_carbon_adapter_daily_model import (
    assemble_causal_carbon_adapter_parameters,
    disable_causal_carbon_adapter_groups,
)
from research.daily_coarse_graining.causal_carbon_adapter_protocol import (
    ARM_IDS,
    CausalCarbonAdapterProtocol,
    load_causal_carbon_adapter_protocol,
)
from research.daily_coarse_graining.causal_carbon_adapter_run import (
    _load_pickle,
    _load_verified_execution,
    _prepare_adapter_experiment,
    _sha256_file,
    _training_identity,
    _verify_training_checkpoint,
)
from research.daily_coarse_graining.markov_dataset import load_markov_shard
from research.daily_coarse_graining.production_training_protocol import (
    load_training_protocol,
)
from research.daily_coarse_graining.rollout_stability_run import (
    _bind_dynamic_transition,
    retained_tail_trace_signature,
)

REPORT_SCHEMA_VERSION = "causal_carbon_adapter_screening_report_v1"
ARM_REPORT_SCHEMA_VERSION = "causal_carbon_adapter_screening_arm_v1"
PROGRESS_SCHEMA_VERSION = "causal_carbon_adapter_screening_progress_v1"
IDENTITY_SCHEMA_VERSION = "causal_carbon_adapter_screening_identity_v1"


def _empty_accumulators() -> dict[str, Any]:
    return {
        arm_id: {
            slice_id: {
                shared._cell_id(horizon, feedback): {
                    "sums": None,
                    "windows": 0,
                    "predicted_days": 0,
                    "batches": 0,
                }
                for horizon in shared.REQUIRED_HORIZONS
                for feedback in shared.REQUIRED_FEEDBACK
            }
            for slice_id in shared.REQUIRED_SLICES
        }
        for arm_id in ARM_IDS
    }


def _field_error_metrics(sums, ids, masks) -> Mapping[str, Any]:
    return {
        field_id: shared._error_metrics(sums, np.asarray(mask, dtype=bool))
        for field_id, mask in zip(ids, masks, strict=True)
    }


def _signed_bias_huber(sums, mask) -> Mapping[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    counts = np.asarray(sums.evaluated_values)[mask]
    total_count = float(np.sum(counts))
    if total_count <= 0:
        raise ValueError("causal screening bias selection has no values")
    mean = float(np.sum(np.asarray(sums.normalized_sum)[mask]) / total_count)
    absolute = abs(mean)
    return {
        "normalized_mean_bias": mean,
        "signed_bias_huber": (0.5 * absolute * absolute if absolute <= 1.0 else absolute - 0.5),
        "evaluated_values": int(round(total_count)),
    }


def _field_bias_metrics(sums, ids, masks) -> Mapping[str, Any]:
    return {field_id: _signed_bias_huber(sums, mask) for field_id, mask in zip(ids, masks, strict=True)}


def _finalize_cell(cell, *, resources, layout) -> Mapping[str, Any]:
    result = dict(shared._finalize_cell(cell, resources=resources))
    sums = cell["sums"]
    result["causal_carbon"] = {
        "interface_terminal": _field_error_metrics(
            sums.fast_terminal,
            layout.fast_field_ids,
            layout.fast_field_masks,
        ),
        "primary_state_all_leads": _field_error_metrics(
            sums.state_all,
            layout.primary_state_field_ids,
            layout.primary_state_field_masks,
        ),
        "flux_state_bias_all_leads": _field_bias_metrics(
            sums.state_all,
            layout.flux_bias_field_ids,
            layout.flux_bias_field_masks,
        ),
        "stock_tendency_bias_all_leads": _field_bias_metrics(
            sums.tendency_all,
            layout.stock_tendency_field_ids,
            layout.stock_tendency_field_masks,
        ),
        "guard_state_all_leads": _field_error_metrics(
            sums.state_all,
            layout.guard_field_ids,
            layout.guard_field_masks,
        ),
    }
    return result


def _require_complete_arm_report(
    report: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> None:
    if (
        report.get("schema_version") != ARM_REPORT_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("sealed_test_used") is not False
    ):
        raise ValueError("causal screening refuses an incomplete arm report")
    if set(report.get("slices", {})) != set(shared.REQUIRED_SLICES):
        raise ValueError("causal screening arm slice inventory is incomplete")
    required_cells = {
        shared._cell_id(horizon, feedback)
        for horizon in shared.REQUIRED_HORIZONS
        for feedback in shared.REQUIRED_FEEDBACK
    }
    for slice_id in shared.REQUIRED_SLICES:
        cells = report["slices"][slice_id]["cells"]
        if set(cells) != required_cells:
            raise ValueError(f"causal screening cell inventory is incomplete for {slice_id}")
        for horizon in shared.REQUIRED_HORIZONS:
            expected = expected_counts[slice_id][horizon]
            for feedback in shared.REQUIRED_FEEDBACK:
                cell = cells[shared._cell_id(horizon, feedback)]
                if (
                    int(cell["windows"]) != expected["windows"]
                    or int(cell["predicted_days"]) != expected["predicted_days"]
                ):
                    raise ValueError(f"causal screening evidence is incomplete for {slice_id}/{horizon}/{feedback}")


def _metric_ratio(
    records: list[Mapping[str, Any]],
    *,
    gate_id: str,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    metric: str,
    threshold: float,
) -> None:
    records.append(
        shared._ratio_record(
            gate_id=gate_id,
            candidate=candidate[metric],
            control=control[metric],
            threshold=threshold,
        )
    )


def classify_screening(
    *,
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: CausalCarbonAdapterProtocol,
    expected_counts: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> Mapping[str, Any]:
    """Apply the frozen Experiment C gates without metric substitution."""

    _require_complete_arm_report(control, expected_counts=expected_counts)
    _require_complete_arm_report(candidate, expected_counts=expected_counts)
    gates = protocol.raw["screening_validation"]["gates_relative_to_control"]
    ratios: list[Mapping[str, Any]] = []
    cells = (
        (1, "teacher_forced"),
        (7, "free"),
        (30, "free"),
    )
    primary_thresholds = {
        1: gates["one_step_primary_state_ratio_max"],
        7: gates["seven_day_primary_state_ratio_max"],
        30: gates["thirty_day_primary_state_ratio_max"],
    }

    for slice_id in shared.REQUIRED_SLICES:
        control_one = control["slices"][slice_id]["cells"][shared._cell_id(1, "teacher_forced")]
        candidate_one = candidate["slices"][slice_id]["cells"][shared._cell_id(1, "teacher_forced")]
        for field_id in sorted(control_one["causal_carbon"]["interface_terminal"]):
            _metric_ratio(
                ratios,
                gate_id=f"interface/{slice_id}/{field_id}",
                candidate=candidate_one["causal_carbon"]["interface_terminal"][field_id],
                control=control_one["causal_carbon"]["interface_terminal"][field_id],
                metric="normalized_huber",
                threshold=gates["one_step_interface_ratio_max"],
            )

        for horizon, feedback in cells:
            cell_id = shared._cell_id(horizon, feedback)
            control_cell = control["slices"][slice_id]["cells"][cell_id]
            candidate_cell = candidate["slices"][slice_id]["cells"][cell_id]
            _metric_ratio(
                ratios,
                gate_id=f"global_state/{slice_id}/horizon_{horizon}",
                candidate=candidate_cell["state_terminal_lead"]["global"],
                control=control_cell["state_terminal_lead"]["global"],
                metric="normalized_rmse",
                threshold=gates["global_state_ratio_max"],
            )
            for field_id in sorted(control_cell["causal_carbon"]["primary_state_all_leads"]):
                _metric_ratio(
                    ratios,
                    gate_id=(f"primary_state/{slice_id}/horizon_{horizon}/{field_id}"),
                    candidate=candidate_cell["causal_carbon"]["primary_state_all_leads"][field_id],
                    control=control_cell["causal_carbon"]["primary_state_all_leads"][field_id],
                    metric="normalized_huber",
                    threshold=primary_thresholds[horizon],
                )
            for family, gate_name in (
                ("flux_state_bias_all_leads", "flux_bias_ratio_max"),
                (
                    "stock_tendency_bias_all_leads",
                    "stock_tendency_bias_ratio_max",
                ),
            ):
                for field_id in sorted(control_cell["causal_carbon"][family]):
                    _metric_ratio(
                        ratios,
                        gate_id=(f"{family}/{slice_id}/horizon_{horizon}/{field_id}"),
                        candidate=candidate_cell["causal_carbon"][family][field_id],
                        control=control_cell["causal_carbon"][family][field_id],
                        metric="signed_bias_huber",
                        threshold=gates[gate_name],
                    )
            for field_id in sorted(control_cell["causal_carbon"]["guard_state_all_leads"]):
                _metric_ratio(
                    ratios,
                    gate_id=f"guard/{slice_id}/horizon_{horizon}/{field_id}",
                    candidate=candidate_cell["causal_carbon"]["guard_state_all_leads"][field_id],
                    control=control_cell["causal_carbon"]["guard_state_all_leads"][field_id],
                    metric="normalized_huber",
                    threshold=gates["guard_field_ratio_max"],
                )

    hard_records = []
    hard_zero_fields = (
        "state_unexpected_defined_status_mismatches",
        "fast_unexpected_defined_status_mismatches",
        "discrete_state_mismatches",
        "nonfinite_defined_state_values",
        "nonfinite_defined_fast_values",
        "negative_source_nonnegative_carbon_stocks",
    )
    for arm_name, arm in (("control", control), ("candidate", candidate)):
        for slice_id in shared.REQUIRED_SLICES:
            for horizon, feedback in cells:
                counts = arm["slices"][slice_id]["cells"][shared._cell_id(horizon, feedback)]["hard_counts"]
                for field in hard_zero_fields:
                    count = int(counts[field])
                    hard_records.append(
                        {
                            "id": (f"{arm_name}/{slice_id}/horizon_{horizon}/{feedback}/{field}"),
                            "count": count,
                            "required": 0,
                            "passed": count == 0,
                        }
                    )

    structural = []
    for arm_name, arm in (("control", control), ("candidate", candidate)):
        structural.extend(
            (
                {
                    "id": f"{arm_name}/restart_split_prediction",
                    "passed": bool(arm["structural_gates"]["restart_split_exact"]),
                },
                {
                    "id": f"{arm_name}/hidden_cross_day_network_memory",
                    "passed": (arm["model_architecture"]["cross_day_memory"] == "canonical_state_only"),
                },
                {
                    "id": f"{arm_name}/source_backed_retained_tail",
                    "passed": (
                        arm["structural_gates"]["retained_tail"] == "canonical_source_backed_dynamic_transition"
                    ),
                },
            )
        )
    passed = (
        all(record["passed"] for record in ratios)
        and all(record["passed"] for record in hard_records)
        and all(record["passed"] for record in structural)
    )
    return {
        "status": "passed" if passed else "rejected",
        "decision": (
            "advance_to_three_seed_confirmation" if passed else "stop_and_attribute_declared_screening_failure"
        ),
        "ratio_gates": ratios,
        "hard_constraints": hard_records,
        "structural_constraints": structural,
        "metric_policy": {
            "interface": "fieldwise_terminal_normalized_huber",
            "primary_and_guard": "fieldwise_all_lead_normalized_huber",
            "global_state": "terminal_normalized_rmse",
            "flux_bias": "fieldwise_huber_of_signed_normalized_mean_state_error",
            "stock_tendency_bias": ("fieldwise_huber_of_signed_normalized_mean_tendency_error"),
        },
    }


def _load_arms(
    experiment_root,
    execution,
    schedule,
    experiment,
    *,
    candidate_disabled_group_ids: tuple[str, ...] = (),
):
    parameters = {}
    identities = {}
    budget = int(schedule["updates"])
    for arm_id in ARM_IDS:
        report_path = experiment_root / arm_id / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != "causal_carbon_adapter_training_report_v1"
            or report.get("status") != "completed"
            or report.get("arm_id") != arm_id
            or int(report.get("updates", -1)) != budget
            or report.get("sealed_test_used") is not False
        ):
            raise ValueError(f"causal screening arm report is incomplete for {arm_id}")
        checkpoint_path = Path(report["checkpoint"]).resolve()
        if _sha256_file(checkpoint_path) != report["checkpoint_sha256"]:
            raise ValueError(f"causal screening checkpoint hash drift for {arm_id}")
        checkpoint = _load_pickle(checkpoint_path)
        identity = _training_identity(execution, experiment, arm_id)
        _verify_training_checkpoint(
            checkpoint,
            arm_id=arm_id,
            identity=identity,
            schedule=schedule,
        )
        if int(checkpoint["next_update"]) != budget:
            raise ValueError(f"causal screening checkpoint did not reach budget for {arm_id}")
        if report.get("identity") != identity:
            raise ValueError(f"causal screening arm-report identity drift for {arm_id}")
        trainable = jax.tree_util.tree_map(
            jnp.asarray,
            checkpoint["parameters"],
        )
        arm_parameters = assemble_causal_carbon_adapter_parameters(
            experiment.base_parameters,
            trainable,
        )
        disabled_group_ids = ()
        if arm_id == ARM_IDS[1]:
            disabled_group_ids = candidate_disabled_group_ids
            arm_parameters = disable_causal_carbon_adapter_groups(
                arm_parameters,
                experiment.adapter_definition.causal_carbon_adapter_spec,
                disabled_group_ids,
            )
        parameters[arm_id] = arm_parameters
        identities[arm_id] = {
            "training_report": {
                "path": str(report_path),
                "sha256": _sha256_file(report_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": report["checkpoint_sha256"],
            },
            "identity": identity,
            "post_training_transform": {
                "id": "exact_zero_adapter_group_ablation_v1",
                "disabled_group_ids": list(disabled_group_ids),
                "training_checkpoint_unchanged": True,
            },
        }
    return parameters, identities


def _evaluation_identity(
    *,
    protocol,
    execution,
    arms,
    inventory_identity,
    batch_sizes,
    max_shards_per_slice,
    verify_dataset_hashes,
):
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "evaluation_git_head": shared._current_git_head(),
        "training_git_head": execution["training_git_head"],
        "protocol_sha256": protocol.sha256,
        "execution_canonical_sha256": execution["canonical_sha256"],
        "arms": arms,
        "inventory_canonical_sha256": inventory_identity["canonical_sha256"],
        "horizons": list(shared.REQUIRED_HORIZONS),
        "feedback": list(shared.REQUIRED_FEEDBACK),
        "batch_sizes": {str(key): int(value) for key, value in batch_sizes.items()},
        "max_shards_per_slice": max_shards_per_slice,
        "dataset_content_hashes_reverified": bool(verify_dataset_hashes),
        "sealed_test_used": False,
    }
    return payload | {"canonical_sha256": shared._canonical_digest(payload)}


def run_screening(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    candidate_disabled_group_ids = tuple(dict.fromkeys(args.candidate_disabled_group))
    experiment_root = args.experiment_root.resolve()
    execution_path = experiment_root / "training_execution.json"
    raw_execution = json.loads(execution_path.read_text(encoding="utf-8"))
    protocol_path = Path(raw_execution["protocol"]["path"]).resolve()
    protocol = load_causal_carbon_adapter_protocol(protocol_path)
    paths = raw_execution["dataset"]
    experiment = _prepare_adapter_experiment(
        protocol_path=protocol_path,
        dataset_path=paths["dataset"]["path"],
        statistics_path=paths["statistics"]["path"],
        acceptance_path=paths["acceptance"]["path"],
        parent_checkpoint_path=paths["parent_checkpoint"]["path"],
        parent_report_path=paths["parent_report"]["path"],
        plan_path=None,
    )
    execution_path, execution, schedule = _load_verified_execution(
        execution_path,
        experiment,
        require_current_training_head=False,
    )
    resources = experiment.resources._replace(model_definition=experiment.adapter_definition)
    if not args.skip_dataset_content_hash_verification:
        shared.verify_training_acceptance(
            experiment.paths["acceptance"],
            experiment.paths["dataset"],
            experiment.paths["statistics"],
            verify_dataset_hashes=True,
        )
    training_protocol_path = Path(__file__).resolve().parents[2] / protocol.raw["artifacts"]["training_protocol"]
    training_protocol = load_training_protocol(training_protocol_path)
    if training_protocol.sha256 != protocol.raw["artifacts"]["training_protocol_sha256"]:
        raise ValueError("causal screening training protocol hash drift")
    inventory = shared.build_model_selection_inventory(
        resources.index,
        training_protocol,
    )
    inventory_identity = shared._inventory_identity(inventory)
    parameters, arm_identities = _load_arms(
        experiment_root,
        execution,
        schedule,
        experiment,
        candidate_disabled_group_ids=candidate_disabled_group_ids,
    )
    batch_sizes = {
        1: int(args.batch_size_1),
        7: int(args.batch_size_7),
        30: int(args.batch_size_30),
    }
    if any(value < 1 for value in batch_sizes.values()):
        raise ValueError("causal screening batch sizes must be positive")
    identity = _evaluation_identity(
        protocol=protocol,
        execution=execution,
        arms=arm_identities,
        inventory_identity=inventory_identity,
        batch_sizes=batch_sizes,
        max_shards_per_slice=args.max_shards_per_slice,
        verify_dataset_hashes=not args.skip_dataset_content_hash_verification,
    )
    output_root = args.output_root.resolve()
    progress_path = output_root / "screening_progress.pkl"
    tasks = shared._selected_tasks(
        inventory,
        max_shards_per_slice=args.max_shards_per_slice,
    )
    if progress_path.exists():
        progress = _load_pickle(progress_path)
        if (
            progress.get("schema_version") != PROGRESS_SCHEMA_VERSION
            or progress.get("identity") != identity
            or progress.get("task_count") != len(tasks)
        ):
            raise ValueError("existing causal screening progress identity drift")
        accumulators = progress["accumulators"]
        next_task = int(progress["next_task"])
        structural_gates = progress["structural_gates"]
        reference_days = progress["reference_days"]
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        accumulators = _empty_accumulators()
        next_task = 0
        structural_gates = {}
        reference_days = {}

    runtime = None
    runtime_landpoint = None
    transition = None
    trace_signature = None
    compiled_steps = {}
    interval_started = time.perf_counter()
    for task_index in range(next_task, len(tasks)):
        slice_id, reference = tasks[task_index]
        shard = load_markov_shard(reference.path)
        reference_key = shared._reference_key(slice_id, reference)
        previous_days = reference_days.setdefault(reference_key, int(shard.days))
        if int(previous_days) != int(shard.days):
            raise ValueError("causal screening shard transition count drift")
        first_batch_for_runtime = runtime_landpoint != reference.landpoint_id

        for raw, valid, starts in shared._padded_window_batches(
            shard,
            horizon=1,
            batch_size=batch_sizes[1],
        ):
            compiled_forcing = None
            if first_batch_for_runtime:
                runtime, compiled_forcing = _make_runtime_and_compiled_forcing(
                    landpoint_id=reference.landpoint_id,
                    entry=resources.plan_entries[reference.landpoint_id],
                    config_path=resources.config_path,
                    contract=resources.contract,
                    batch=raw,
                )
                runtime_landpoint = reference.landpoint_id
                first_batch_for_runtime = False
                observed_signature = retained_tail_trace_signature(runtime.static)
                if transition is None:
                    transition = _bind_dynamic_transition(resources, runtime)
                    trace_signature = observed_signature
                elif observed_signature != trace_signature:
                    raise ValueError("causal screening retained-tail trace signature drift")
            prepared = shared._prepare_batch(
                resources,
                raw,
                runtime,
                compiled_forcing=compiled_forcing,
            )
            if (1, 3) not in compiled_steps:
                compiled_steps[(1, 3)] = shared.make_screening_batch_step(
                    resources=resources,
                    transition=transition,
                )
            step = compiled_steps[(1, 3)]
            all_weights, terminal_weights = shared._teacher_window_weights(
                shard.days,
                starts,
            )
            all_weights *= valid[None, :, None]
            terminal_weights *= valid[None, :, None]
            for arm_id in ARM_IDS:
                batch_sums = shared._host_tree(
                    step(
                        parameters[arm_id],
                        *prepared,
                        jnp.asarray(all_weights),
                        jnp.asarray(terminal_weights),
                    )
                )
                for weight_index, horizon in enumerate(shared.REQUIRED_HORIZONS):
                    selected = shared._select_weight_index(
                        batch_sums,
                        weight_index,
                    )
                    windows = int(round(float(np.sum(terminal_weights[weight_index]))))
                    predicted_days = int(round(float(np.sum(all_weights[weight_index]))))
                    shared._accumulate_cell(
                        accumulators,
                        arm_id=arm_id,
                        slice_id=slice_id,
                        horizon=horizon,
                        feedback="teacher_forced",
                        sums=selected,
                        windows=windows,
                        predicted_days=predicted_days,
                    )
                    if horizon == 1:
                        shared._accumulate_cell(
                            accumulators,
                            arm_id=arm_id,
                            slice_id=slice_id,
                            horizon=1,
                            feedback="free",
                            sums=selected,
                            windows=windows,
                            predicted_days=predicted_days,
                        )

        for horizon in (7, 30):
            for raw, valid, _ in shared._padded_window_batches(
                shard,
                horizon=horizon,
                batch_size=batch_sizes[horizon],
            ):
                prepared = shared._prepare_batch(resources, raw, runtime)
                if (horizon, 1) not in compiled_steps:
                    compiled_steps[(horizon, 1)] = shared.make_screening_batch_step(
                        resources=resources,
                        transition=transition,
                    )
                step = compiled_steps[(horizon, 1)]
                all_weights, terminal_weights = shared._free_window_weights(
                    valid,
                    horizon,
                )
                for arm_id in ARM_IDS:
                    batch_sums = shared._host_tree(
                        step(
                            parameters[arm_id],
                            *prepared,
                            jnp.asarray(all_weights),
                            jnp.asarray(terminal_weights),
                        )
                    )
                    shared._accumulate_cell(
                        accumulators,
                        arm_id=arm_id,
                        slice_id=slice_id,
                        horizon=horizon,
                        feedback="free",
                        sums=shared._select_weight_index(batch_sums, 0),
                        windows=int(np.count_nonzero(valid)),
                        predicted_days=int(np.count_nonzero(valid) * horizon),
                    )
                if not structural_gates and horizon == 30:
                    for arm_id in ARM_IDS:
                        structural_gates[arm_id] = shared._restart_split_probe(
                            parameters=parameters[arm_id],
                            prepared=prepared,
                            resources=resources,
                            transition=transition,
                        )

        next_task = task_index + 1
        if next_task % shared.CHECKPOINT_EVERY_REFERENCES == 0 or next_task == len(tasks):
            shared._atomic_pickle(
                progress_path,
                {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "identity": identity,
                    "task_count": len(tasks),
                    "next_task": next_task,
                    "accumulators": accumulators,
                    "structural_gates": structural_gates,
                    "reference_days": reference_days,
                },
            )
            elapsed = time.perf_counter() - interval_started
            print(
                "causal_adapter_screening "
                f"references={next_task}/{len(tasks)} "
                f"last={reference.landpoint_id}:{reference.year}:{slice_id} "
                f"interval_seconds={elapsed:.3f}",
                flush=True,
            )
            interval_started = time.perf_counter()

    full_evidence = args.max_shards_per_slice is None
    selected_by_slice = {
        slice_id: tuple((task_slice, reference) for task_slice, reference in tasks if task_slice == slice_id)
        for slice_id in shared.REQUIRED_SLICES
    }
    expected_counts = {
        slice_id: {
            horizon: shared._expected_cell_counts(
                selected_by_slice[slice_id],
                horizon,
                reference_days=reference_days,
            )
            for horizon in shared.REQUIRED_HORIZONS
        }
        for slice_id in shared.REQUIRED_SLICES
    }
    arm_reports = {}
    for arm_id in ARM_IDS:
        slices = {}
        for slice_id in shared.REQUIRED_SLICES:
            cells = {
                cell_id: _finalize_cell(
                    cell,
                    resources=resources,
                    layout=experiment.layout,
                )
                for cell_id, cell in accumulators[arm_id][slice_id].items()
            }
            slices[slice_id] = {
                "selection": {
                    "spatial_split": inventory[slice_id]["spatial_split"],
                    "temporal_split": inventory[slice_id]["temporal_split"],
                    "shards": len(selected_by_slice[slice_id]),
                    "complete_population_shards": inventory[slice_id]["shards"],
                },
                "cells": cells,
            }
        arm_report = {
            "schema_version": ARM_REPORT_SCHEMA_VERSION,
            "status": "completed" if full_evidence else "smoke_completed",
            "arm_id": arm_id,
            "identity": arm_identities[arm_id],
            "model_architecture": experiment.adapter_definition.identity(),
            "post_training_transform": arm_identities[arm_id]["post_training_transform"],
            "slices": slices,
            "structural_gates": {
                "restart_split_exact": bool(structural_gates[arm_id]["passed"]),
                "restart_split_probe": structural_gates[arm_id],
                "retained_tail": "canonical_source_backed_dynamic_transition",
            },
            "sealed_test_used": False,
        }
        arm_reports[arm_id] = arm_report
        shared._atomic_json(
            output_root / arm_id / "screening_report.json",
            arm_report,
        )

    classification = None
    if full_evidence:
        classification = classify_screening(
            control=arm_reports[ARM_IDS[0]],
            candidate=arm_reports[ARM_IDS[1]],
            protocol=protocol,
            expected_counts=expected_counts,
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (classification["status"] if classification is not None else "smoke_completed"),
        "promotion_eligible": classification is not None,
        "identity": identity,
        "inventory": inventory_identity,
        "selected_shards": {slice_id: len(selected_by_slice[slice_id]) for slice_id in shared.REQUIRED_SLICES},
        "expected_counts": {
            slice_id: {str(horizon): counts for horizon, counts in horizons.items()}
            for slice_id, horizons in expected_counts.items()
        },
        "arm_reports": {
            arm_id: {
                "path": str(output_root / arm_id / "screening_report.json"),
                "sha256": _sha256_file(output_root / arm_id / "screening_report.json"),
            }
            for arm_id in ARM_IDS
        },
        "classification": classification,
        "elapsed_seconds": time.perf_counter() - started,
        "sealed_test_used": False,
    }
    report_path = output_root / ("matched_screening_report.json" if full_evidence else "screening_smoke_report.json")
    return shared._atomic_json(report_path, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size-1", type=int, default=512)
    parser.add_argument("--batch-size-7", type=int, default=512)
    parser.add_argument("--batch-size-30", type=int, default=256)
    parser.add_argument("--max-shards-per-slice", type=int)
    parser.add_argument(
        "--candidate-disabled-group",
        action="append",
        default=[],
        help="evaluation-only exact-zero ablation applied to the trained candidate",
    )
    parser.add_argument(
        "--skip-dataset-content-hash-verification",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run_screening(build_parser().parse_args(argv))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
