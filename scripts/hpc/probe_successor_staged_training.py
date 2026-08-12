"""Two-update real-v6 smoke for the staged successor training path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

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
    _hard_counts_host,
    _make_arm_steps,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.successor_rollout_screen_prepared import (
    load_prepared_rollout_artifact,
)
from scripts.hpc.successor_small_fit_prepared import (
    load_prepared_artifact,
    load_prepared_manifest,
)
from scripts.hpc.successor_staged_validation_prepared import (
    load_staged_validation_prepared,
    ordered_validation_entries,
)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
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
        raise ValueError("staged smoke requires the staged-v3 contract")
    arms = tuple(contract.payload["matched_arms"])
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_successor_small_fit_schedule(references, contract, updates=2)
    verify_successor_small_fit_schedule(schedule, references, contract)
    prepared = load_prepared_manifest(
        args.prepared_manifest,
        runner_contract_sha256=contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        updates=2,
        source_git_head=args.prepared_source_git_head,
    )
    validation_prepared, panel = load_staged_validation_prepared(
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
    if panel != assets.validation_panel:
        raise ValueError("staged smoke validation panel drift")
    first = load_prepared_artifact(
        args.prepared_manifest,
        prepared["entries"][0],
        schedule_entry=schedule["entries"][0],
    )
    _reference, training_adapter, _timing = _prepare_adapter(
        runtime,
        schedule["entries"][0],
    )
    if first.trace_signature != _retained_tail_trace_signature(training_adapter):
        raise ValueError("staged smoke retained-tail trace drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(lambda value: value[0], first.batch.anchor_inputs)
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
        raise ValueError("staged smoke parameter count drift")
    optimizers = {arm: initialize_successor_adamw(parameters[arm]) for arm in arms}
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    steps = {
        arm: _make_arm_steps(
            runtime,
            arm,
            training_adapter,
            assets.coefficients,
            long_scales,
            contract,
        )[0]
        for arm in arms
    }
    training_records = []
    for update, (entry, record) in enumerate(
        zip(schedule["entries"], prepared["entries"], strict=True),
        start=1,
    ):
        artifact = load_prepared_artifact(
            args.prepared_manifest,
            record,
            schedule_entry=entry,
        )
        arms_record = {}
        for arm in arms:
            result = steps[arm](
                parameters[arm],
                optimizers[arm],
                artifact.batch,
                artifact.day_inputs,
                np.asarray(successor_learning_rate(contract, update), dtype=np.float64),
            )
            jax.block_until_ready(result)
            parameters[arm] = result.parameters
            optimizers[arm] = result.optimizer
            arms_record[arm] = {
                "loss": float(np.asarray(result.loss)),
                "gradient_norm": float(np.asarray(result.gradient_norm)),
                "nonfinite_gradient_values": int(
                    np.asarray(result.nonfinite_gradient_values)
                ),
                "hard_gate_counts": _hard_counts_host(result.hard_counts),
                "update_applied": bool(np.asarray(result.update_applied)),
            }
        training_records.append({"update": update, "arms": arms_record})
    selected_validation = []
    ordered = ordered_validation_entries(panel)
    for slice_id in ("temporal", "spatial", "joint"):
        selected_validation.append(
            next(
                (record, entry)
                for record, (wanted_slice, entry) in zip(
                    validation_prepared["entries"], ordered, strict=True
                )
                if wanted_slice == slice_id
            )
        )
    first_validation_entry = selected_validation[0][1]
    first_validation_reference = _validation_reference(
        runtime["index"], "temporal", first_validation_entry
    )
    validation_adapter = _prepare_validation_adapter(
        runtime,
        first_validation_reference,
        int(first_validation_entry["start_rows"][0]),
    )
    evaluators = {
        arm: _make_evaluator(runtime, arm, validation_adapter, long_scales, contract)
        for arm in arms
    }
    validation_records = []
    for record, entry in selected_validation:
        artifact = load_prepared_rollout_artifact(
            args.validation_prepared_manifest,
            record,
            slice_id=record["slice_id"],
            window_entry=entry,
        )
        if artifact.trace_signature != _retained_tail_trace_signature(
            validation_adapter
        ):
            raise ValueError("staged smoke validation retained-tail trace drift")
        arm_records = {}
        for arm in arms:
            statistics = evaluators[arm](
                parameters[arm], artifact.batch, artifact.day_inputs
            )
            jax.block_until_ready(statistics)
            hard = jax.device_get(statistics).hard_counts
            arm_records[arm] = _hard_counts_host(hard)
        validation_records.append(
            {"slice_id": record["slice_id"], "arms": arm_records}
        )
    hard_failure_count = sum(
        value
        for update in training_records
        for arm in arms
        for value in update["arms"][arm]["hard_gate_counts"].values()
    ) + sum(
        value
        for record in validation_records
        for arm in arms
        for value in record["arms"][arm].values()
    )
    passed = hard_failure_count == 0 and all(
        update["arms"][arm]["update_applied"]
        and update["arms"][arm]["nonfinite_gradient_values"] == 0
        and np.isfinite(update["arms"][arm]["loss"])
        and np.isfinite(update["arms"][arm]["gradient_norm"])
        for update in training_records
        for arm in arms
    )
    report = {
        "schema_version": "successor_staged_training_gpu_smoke_v1",
        "status": "passed" if passed else "failed",
        "training_updates_per_arm": 2,
        "checkpoint_written": False,
        "recovery_written": False,
        "training": training_records,
        "validation": validation_records,
        "hard_failure_count": hard_failure_count,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "reconstructs_48_forcing_records": False,
        "executes_48_step_state_scan": False,
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
    parser.add_argument("--prepared-source-git-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
