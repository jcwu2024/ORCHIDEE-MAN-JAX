"""Run one matched successor training stage and its fixed validation panel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_rollout_screen import (
    add_rollout_screen_statistics,
    empty_rollout_screen_statistics,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
    write_staged_best_checkpoint,
    write_staged_recovery,
)
from research.daily_coarse_graining.daily_operator.successor_staged_training import (
    REQUIRED_VALIDATION_SLICES,
    advance_staged_validation_state,
    finalize_staged_validation_statistics,
)
from scripts.hpc.prepare_successor_rollout_screen import (
    _prepare_adapter as _prepare_validation_adapter,
)
from scripts.hpc.prepare_successor_staged_validation import (
    _reference as _validation_reference,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_rollout_screen import _make_evaluator
from scripts.hpc.run_successor_small_fit import (
    _component_record,
    _hard_counts_host,
    _make_arm_steps,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
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


def _training_identity(
    runtime: Mapping[str, Any],
    contract: Any,
    schedule: Mapping[str, Any],
    assets: Any,
    validation_prepared: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "staged_training_contract_sha256": contract.sha256,
        "full_schedule_sha256": schedule["schedule_sha256"],
        "frozen_coefficient_contract_sha256": assets.coefficient_contract[
            "contract_sha256"
        ],
        "validation_panel_sha256": assets.validation_panel["panel_sha256"],
        "validation_prepared_manifest_sha256": validation_prepared[
            "manifest_sha256"
        ],
        "dataset_manifest_sha256": contract.payload["identity"][
            "dataset_manifest_sha256"
        ],
        "successor_candidate_contract_sha256": runtime[
            "successor_contract"
        ].sha256,
        "fast_owner_contract_sha256": runtime["owner_contract"].contract_sha256,
    }


def _evaluate_panel(
    runtime: Mapping[str, Any],
    *,
    contract: Any,
    parameters: Mapping[str, Any],
    arms: tuple[str, ...],
    long_scales: Any,
    adapter_template: Any,
    prepared_manifest_path: Path,
    prepared: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    jax = runtime["jax"]
    evaluators = {
        arm: _make_evaluator(runtime, arm, adapter_template, long_scales, contract)
        for arm in arms
    }
    accumulators = {
        arm: {
            slice_id: empty_rollout_screen_statistics()
            for slice_id in REQUIRED_VALIDATION_SLICES
        }
        for arm in arms
    }
    ordered = ordered_validation_entries(panel)
    for record, (slice_id, entry) in zip(prepared["entries"], ordered, strict=True):
        artifact = load_prepared_rollout_artifact(
            prepared_manifest_path,
            record,
            slice_id=slice_id,
            window_entry=entry,
        )
        if artifact.trace_signature != _retained_tail_trace_signature(adapter_template):
            raise ValueError("successor staged validation retained-tail trace drift")
        for arm in arms:
            statistics = evaluators[arm](
                parameters[arm],
                artifact.batch,
                artifact.day_inputs,
            )
            jax.block_until_ready(statistics)
            accumulators[arm][slice_id] = add_rollout_screen_statistics(
                accumulators[arm][slice_id],
                jax.device_get(statistics),
            )
    reports = {}
    for arm in arms:
        reports[arm] = {
            "slices": {
                slice_id: finalize_staged_validation_statistics(
                    accumulators[arm][slice_id],
                    primary_outputs=contract.payload["validation"]["primary_outputs"],
                )
                for slice_id in REQUIRED_VALIDATION_SLICES
            }
        }
    return reports


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if int(args.recovery_interval) <= 0:
        raise ValueError("successor staged recovery interval must be positive")
    started = time.perf_counter()
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        STAGED_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        initialize_successor_adamw,
        load_successor_small_fit_contract,
        successor_learning_rate,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        derive_long_horizon_scales,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        load_successor_training_assets,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if contract.payload["schema_version"] != STAGED_SCHEMA_VERSION:
        raise ValueError("staged runner requires the staged-v3 contract")
    arms = tuple(contract.payload["matched_arms"])
    start_update, end_update = stage_update_range(
        contract.sampling["stage_milestones"],
        args.milestone,
    )
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_successor_small_fit_schedule(
        references,
        contract,
        updates=int(contract.sampling["maximum_updates_per_arm"]),
    )
    verify_successor_small_fit_schedule(schedule, references, contract)
    stage_manifest = load_staged_prepared_manifest(
        args.prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        full_schedule_sha256=schedule["schedule_sha256"],
        start_update=start_update,
        end_update=end_update,
    )
    stage_identity = stage_manifest["identity"]
    if (
        stage_identity.get("dataset_manifest_sha256")
        != contract.payload["identity"]["dataset_manifest_sha256"]
        or stage_identity.get("successor_candidate_contract_sha256")
        != runtime["successor_contract"].sha256
        or stage_identity.get("fast_owner_contract_sha256")
        != runtime["owner_contract"].contract_sha256
    ):
        raise ValueError("successor staged prepared model identity drift")
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
        raise ValueError("successor staged validation prepared panel drift")
    identity = _training_identity(
        runtime,
        contract,
        schedule,
        assets,
        validation_prepared,
    )
    first_record = stage_manifest["entries"][0]
    first_entry = schedule["entries"][start_update - 1]
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        first_record,
        schedule_entry=first_entry,
    )
    _reference, adapter_template, _timing = _prepare_adapter(runtime, first_entry)
    if first_artifact.trace_signature != _retained_tail_trace_signature(adapter_template):
        raise ValueError("successor staged retained-tail trace drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(lambda value: value[0], first_artifact.batch.anchor_inputs)
    )
    parameters = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(int(contract.sampling["seed"])),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=assembled,
            arm=arm,
        )
        for arm in arms
    }
    counts = {arm: successor_parameter_count(parameters[arm]) for arm in arms}
    if set(counts.values()) != {EXPECTED_PARAMETER_COUNT}:
        raise ValueError("successor staged parameter count drift")
    optimizers = {arm: initialize_successor_adamw(parameters[arm]) for arm in arms}
    previous_milestone = start_update - 1
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=previous_milestone,
        maximum_completed_updates=end_update,
    )
    if recovery is None:
        if start_update != 1:
            raise ValueError("successor staged continuation requires recovery")
        completed = 0
        validation_state = None
        best_parameters = {arm: parameters[arm] for arm in arms}
    else:
        completed = int(recovery["completed_updates"])
        parameters = dict(recovery["parameters"])
        optimizers = dict(recovery["optimizers"])
        validation_state = recovery["validation_state"]
        best_parameters = dict(recovery["best_parameters"])
        if validation_state is not None and validation_state.get("stop_training") is True:
            raise ValueError("successor staged recovery has already stopped training")
        completed_milestones = (
            [] if validation_state is None else validation_state["completed_milestones"]
        )
        if completed == end_update and int(args.milestone) in completed_milestones:
            raise ValueError("successor staged milestone is already complete")
        if completed < end_update and completed_milestones != list(
            contract.sampling["stage_milestones"][:
                tuple(contract.sampling["stage_milestones"]).index(args.milestone)
            ]
        ):
            raise ValueError("successor staged recovery validation history drift")
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    first_validation_slice, first_validation_entry = ordered_validation_entries(
        assets.validation_panel
    )[0]
    first_validation_reference = _validation_reference(
        runtime["index"],
        first_validation_slice,
        first_validation_entry,
    )
    validation_adapter_template = _prepare_validation_adapter(
        runtime,
        first_validation_reference,
        int(first_validation_entry["start_rows"][0]),
    )
    steps = {
        arm: _make_arm_steps(
            runtime,
            arm,
            adapter_template,
            assets.coefficients,
            long_scales,
            contract,
        )[0]
        for arm in arms
    }
    records_by_update = {
        int(record["update"]): record for record in stage_manifest["entries"]
    }
    update_records = []
    for update in range(max(completed + 1, start_update), end_update + 1):
        entry = schedule["entries"][update - 1]
        artifact = load_prepared_artifact(
            args.prepared_manifest,
            records_by_update[update],
            schedule_entry=entry,
        )
        if artifact.trace_signature != first_artifact.trace_signature:
            raise ValueError("successor staged prepared trace drift")
        rate = successor_learning_rate(contract, update)
        arm_records = {}
        for arm in arms:
            result = steps[arm](
                parameters[arm],
                optimizers[arm],
                artifact.batch,
                artifact.day_inputs,
                np.asarray(rate, dtype=np.float64),
            )
            jax.block_until_ready(result)
            hard_counts = _hard_counts_host(result.hard_counts)
            if not bool(np.asarray(result.update_applied)):
                raise RuntimeError(
                    f"successor staged update {update} failed for {arm}: {hard_counts}"
                )
            parameters[arm] = result.parameters
            optimizers[arm] = result.optimizer
            arm_records[arm] = {
                "loss": float(np.asarray(result.loss)),
                "one_day_anchor": float(np.asarray(result.anchor)),
                "long_horizon_components": _component_record(
                    LONG_HORIZON_COMPONENTS,
                    result.components,
                ),
                "gradient_norm": float(np.asarray(result.gradient_norm)),
                "hard_gate_counts": hard_counts,
            }
        completed = update
        update_records.append(
            {
                "update": update,
                "landpoint_id": entry["landpoint_id"],
                "year": int(entry["year"]),
                "learning_rate": rate,
                "arms": arm_records,
            }
        )
        if update % int(args.recovery_interval) == 0 and update < end_update:
            write_staged_recovery(
                args.recovery_path,
                identity=identity,
                parameters=jax.device_get(parameters),
                optimizers=jax.device_get(optimizers),
                completed_updates=completed,
                parameter_counts=counts,
                validation_state=validation_state,
                best_parameters=jax.device_get(best_parameters),
                expected_arms=arms,
            )
        print(
            json.dumps(
                {
                    "event": "successor_staged_update",
                    "update": update,
                    "milestone": end_update,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    validation_reports = _evaluate_panel(
        runtime,
        contract=contract,
        parameters=parameters,
        arms=arms,
        long_scales=long_scales,
        adapter_template=validation_adapter_template,
        prepared_manifest_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=assets.validation_panel,
    )
    next_validation_state = advance_staged_validation_state(
        validation_state,
        milestone=end_update,
        arm_reports=validation_reports,
        contract=contract,
    )
    for arm in arms:
        if next_validation_state["history"][-1]["arms"][arm]["improved"]:
            best_parameters[arm] = parameters[arm]
    best_checkpoint = (
        write_staged_best_checkpoint(
            args.best_checkpoint,
            identity=identity,
            parameters=jax.device_get(best_parameters),
            parameter_counts=counts,
            validation_state=next_validation_state,
            expected_arms=arms,
        )
        if all(
            next_validation_state["arms"][arm]["best_milestone"] is not None
            for arm in arms
        )
        else None
    )
    recovery_record = write_staged_recovery(
        args.recovery_path,
        identity=identity,
        parameters=jax.device_get(parameters),
        optimizers=jax.device_get(optimizers),
        completed_updates=end_update,
        parameter_counts=counts,
        validation_state=next_validation_state,
        best_parameters=jax.device_get(best_parameters),
        expected_arms=arms,
    )
    report = {
        "schema_version": "successor_staged_training_stage_report_v1",
        "status": (
            "failed_validation_hard_gate"
            if next_validation_state["stop_reason"] == "validation_hard_failure"
            else "passed"
        ),
        "milestone": end_update,
        "stage_update_range": [start_update, end_update],
        "identity": identity,
        "stage_prepared_manifest_sha256": stage_manifest["manifest_sha256"],
        "training_updates_executed": len(update_records),
        "completed_updates_per_arm": end_update,
        "updates": update_records,
        "validation": validation_reports,
        "validation_state": next_validation_state,
        "best_checkpoint": best_checkpoint,
        "recovery": recovery_record,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "reconstructs_48_forcing_records": False,
        "executes_48_step_state_scan": False,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "repair_validation_hard_failure"
            if next_validation_state["stop_reason"] == "validation_hard_failure"
            else "stop_staged_training_and_review_best_checkpoints"
            if next_validation_state["stop_training"]
            else f"prepare_stage_to_{next_validation_state['next_milestone']}"
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
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-report", type=Path, required=True)
    parser.add_argument("--coefficient-prepared-manifest", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-panel-sha256", required=True)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--recovery-path", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-interval", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
