"""Run the frozen 100-update equal-sample point-year diversity A/B."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
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
)
from research.daily_coarse_graining.daily_operator.successor_staged_training import (
    REQUIRED_VALIDATION_SLICES,
    finalize_staged_validation_statistics,
    validation_hard_failure_count,
    validation_primary_composite,
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
    _make_mixed_objective,
    _slice_context_batch,
    _stack_contexts,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_rollout_screen import _make_evaluator
from scripts.hpc.run_successor_small_fit import (
    ArmUpdateResult,
    _component_record,
    _device_memory_report,
    _hard_counts_host,
    _make_arm_steps,
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

SHORT_UPDATES = 100
PARENT_UPDATE = 1000
MATERIAL_MIXED_IMPROVEMENT = 0.005
STRATEGIES = ("single_point_year", "four_point_year")


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_mixed_train_step(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    coefficients: Mapping[str, float],
    long_scales: Any,
    contract: Any,
) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_adamw_update,
    )

    jax = runtime["jax"]
    objective = _make_mixed_objective(
        runtime,
        arm,
        adapter_template,
        coefficients,
        long_scales,
        contract,
    )

    @jax.jit
    def train_step(
        parameters: Any,
        optimizer: Any,
        anchor_batch: Any,
        context_batches: Any,
        day_inputs: Any,
        learning_rate: Any,
    ) -> ArmUpdateResult:
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
            contract,
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
        return ArmUpdateResult(
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


def _evaluate_strategies(
    runtime: Mapping[str, Any],
    *,
    contract: Any,
    parameters: Mapping[str, Mapping[str, Any]],
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
        strategy: {
            arm: {
                slice_id: empty_rollout_screen_statistics()
                for slice_id in REQUIRED_VALIDATION_SLICES
            }
            for arm in arms
        }
        for strategy in STRATEGIES
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
            raise ValueError("batch-diversity validation retained-tail trace drift")
        for arm in arms:
            evaluator = evaluators[arm]
            for strategy in STRATEGIES:
                statistics = evaluator(
                    parameters[strategy][arm],
                    artifact.batch,
                    artifact.day_inputs,
                )
                jax.block_until_ready(statistics)
                accumulators[strategy][arm][slice_id] = add_rollout_screen_statistics(
                    accumulators[strategy][arm][slice_id],
                    jax.device_get(statistics),
                )
        print(
            json.dumps(
                {
                    "event": "successor_batch_diversity_validation",
                    "ordinal": ordinal,
                    "total": len(ordered),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        strategy: {
            arm: {
                "slices": {
                    slice_id: finalize_staged_validation_statistics(
                        accumulators[strategy][arm][slice_id],
                        primary_outputs=contract.payload["validation"]["primary_outputs"],
                    )
                    for slice_id in REQUIRED_VALIDATION_SLICES
                }
            }
            for arm in arms
        }
        for strategy in STRATEGIES
    }


def _baseline_validation(path: Path, arms: tuple[str, ...]) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "passed"
        or int(payload.get("milestone", -1)) != PARENT_UPDATE
        or set(payload.get("validation", {})) != set(arms)
    ):
        raise ValueError("batch-diversity baseline is not the accepted update-1000 report")
    return payload["validation"]


def _classify(
    validation: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    *,
    arms: tuple[str, ...],
    primary_outputs: Any,
) -> Mapping[str, Any]:
    arm_records = {}
    for arm in arms:
        baseline_value = validation_primary_composite(
            baseline[arm],
            primary_outputs=primary_outputs,
        )
        single_value = validation_primary_composite(
            validation["single_point_year"][arm],
            primary_outputs=primary_outputs,
        )
        mixed_value = validation_primary_composite(
            validation["four_point_year"][arm],
            primary_outputs=primary_outputs,
        )
        arm_records[arm] = {
            "baseline_primary_composite": baseline_value,
            "single_primary_composite": single_value,
            "mixed_primary_composite": mixed_value,
            "single_relative_change_from_baseline": single_value / baseline_value - 1.0,
            "mixed_relative_change_from_baseline": mixed_value / baseline_value - 1.0,
            "mixed_relative_improvement_over_single": 1.0 - mixed_value / single_value,
            "single_hard_failure_count": validation_hard_failure_count(
                validation["single_point_year"][arm]
            ),
            "mixed_hard_failure_count": validation_hard_failure_count(
                validation["four_point_year"][arm]
            ),
        }
    hard_passed = all(
        record["single_hard_failure_count"] == 0
        and record["mixed_hard_failure_count"] == 0
        for record in arm_records.values()
    )
    mixed_no_worse_both = all(
        record["mixed_primary_composite"] <= record["single_primary_composite"]
        for record in arm_records.values()
    )
    material_improvement = any(
        record["mixed_relative_improvement_over_single"]
        >= MATERIAL_MIXED_IMPROVEMENT
        for record in arm_records.values()
    )
    admitted = hard_passed and mixed_no_worse_both and material_improvement
    return {
        "status": "mixed_sampling_admitted" if admitted else "mixed_sampling_rejected",
        "hard_gates_passed": hard_passed,
        "mixed_no_worse_for_both_arms": mixed_no_worse_both,
        "at_least_one_arm_improved_by_0_5_percent": material_improvement,
        "arms": arm_records,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
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
        raise ValueError("batch-diversity short fit requires staged-v3")
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
        raise ValueError("batch-diversity validation panel drift")
    identity = _training_identity(runtime, contract, schedule, assets, validation_prepared)
    first_record = manifest["entries"][0]
    first_entry = schedule["entries"][PARENT_UPDATE]
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        first_record,
        schedule_entry=first_entry,
    )
    _reference, adapter_template, adapter_timing = _prepare_adapter(runtime, first_entry)
    signature = _retained_tail_trace_signature(adapter_template)
    if first_artifact.trace_signature != signature:
        raise ValueError("batch-diversity retained-tail trace drift")
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
    if set(counts.values()) != {EXPECTED_PARAMETER_COUNT}:
        raise ValueError("batch-diversity parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("batch-diversity short fit requires update-1000 recovery")
    baseline = _baseline_validation(args.baseline_report, arms)
    parameters = {
        strategy: dict(recovery["parameters"])
        for strategy in STRATEGIES
    }
    optimizers = {
        strategy: dict(recovery["optimizers"])
        for strategy in STRATEGIES
    }
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    single_steps = {
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
    mixed_steps = {
        arm: _make_mixed_train_step(
            runtime,
            arm,
            adapter_template,
            assets.coefficients,
            long_scales,
            contract,
        )
        for arm in arms
    }
    records = []
    clip_counts = {
        strategy: {arm: 0 for arm in arms}
        for strategy in STRATEGIES
    }
    memory_before = _device_memory_report(runtime["devices"])
    for short_update in range(1, SHORT_UPDATES + 1):
        offset = (short_update - 1) * POINTS_PER_MIXED_UPDATE
        group = []
        for group_index in range(POINTS_PER_MIXED_UPDATE):
            record = manifest["entries"][offset + group_index]
            update = int(record["update"])
            entry = schedule["entries"][update - 1]
            artifact = load_prepared_artifact(
                args.prepared_manifest,
                record,
                schedule_entry=entry,
            )
            if artifact.trace_signature != signature:
                raise ValueError("batch-diversity prepared trace drift")
            group.append((entry, artifact))
        point_years = {
            (str(entry["landpoint_id"]), int(entry["year"]))
            for entry, _artifact in group
        }
        if len(point_years) != POINTS_PER_MIXED_UPDATE:
            raise ValueError("batch-diversity group repeated a point-year")
        designated = group[(short_update - 1) % POINTS_PER_MIXED_UPDATE]
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
        rate = np.asarray(
            successor_learning_rate(contract, absolute_update),
            dtype=np.float64,
        )
        arm_records = {}
        for arm in arms:
            single = single_steps[arm](
                parameters["single_point_year"][arm],
                optimizers["single_point_year"][arm],
                designated[1].batch,
                designated[1].day_inputs,
                rate,
            )
            mixed = mixed_steps[arm](
                parameters["four_point_year"][arm],
                optimizers["four_point_year"][arm],
                anchor_batch,
                context_batch,
                context_inputs,
                rate,
            )
            jax.block_until_ready((single, mixed))
            strategy_records = {}
            for strategy, result in zip(STRATEGIES, (single, mixed), strict=True):
                hard_counts = _hard_counts_host(result.hard_counts)
                if not bool(np.asarray(result.update_applied)):
                    raise RuntimeError(
                        f"batch-diversity update {short_update} failed for "
                        f"{strategy}/{arm}: {hard_counts}"
                    )
                parameters[strategy][arm] = result.parameters
                optimizers[strategy][arm] = result.optimizer
                norm = float(np.asarray(result.gradient_norm))
                clip_counts[strategy][arm] += int(
                    norm > float(contract.optimizer["global_gradient_norm_clip"])
                )
                strategy_records[strategy] = {
                    "loss": float(np.asarray(result.loss)),
                    "one_day_anchor": float(np.asarray(result.anchor)),
                    "long_horizon_components": _component_record(
                        LONG_HORIZON_COMPONENTS,
                        result.components,
                    ),
                    "gradient_norm": norm,
                    "hard_gate_counts": hard_counts,
                }
            arm_records[arm] = strategy_records
        records.append(
            {
                "short_update": short_update,
                "optimizer_update": absolute_update,
                "learning_rate": float(rate),
                "point_years": [
                    {
                        "update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                "single_point_year_update": int(designated[0]["update"]),
                "arms": arm_records,
            }
        )
        del group, contexts, anchor_batch, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "successor_batch_diversity_short_update",
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
    validation = _evaluate_strategies(
        runtime,
        contract=contract,
        parameters=parameters,
        arms=arms,
        long_scales=long_scales,
        adapter_template=validation_adapter,
        prepared_manifest_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=assets.validation_panel,
    )
    classification = _classify(
        validation,
        baseline,
        arms=arms,
        primary_outputs=contract.payload["validation"]["primary_outputs"],
    )
    report = {
        "schema_version": "successor_batch_diversity_short_fit_v1",
        "status": "completed",
        "git_head": _git_head(),
        "parent_update": PARENT_UPDATE,
        "short_updates": SHORT_UPDATES,
        "parameter_count_per_arm": counts,
        "batch_comparison": {
            "single_point_year": {
                "point_years": 1,
                "one_day_anchors": 32,
                "rollout_windows": 4,
            },
            "four_point_year": {
                "point_years": 4,
                "one_day_anchors": 32,
                "rollout_windows": 4,
            },
            "rollout_days": 30,
            "same_loss_coefficients": True,
            "same_learning_rates": True,
        },
        "classification": classification,
        "gradient_clip_counts": clip_counts,
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
            "review_mixed_sampling_before_any_paid_proposal"
            if classification["status"] == "mixed_sampling_admitted"
            else "close_mixed_sampling_branch",
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
    parser.add_argument("--recovery-path", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

