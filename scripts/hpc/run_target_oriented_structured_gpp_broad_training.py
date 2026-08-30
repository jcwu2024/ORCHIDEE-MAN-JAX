"""Run the bounded first broad-data structured-GPP training stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_checkpoint import (
    load_broad_recovery,
    write_broad_checkpoint,
    write_broad_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_preflight import (
    build_broad_training_schedule,
    reference_identity,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_training import (
    CHECKPOINT_UPDATES,
    load_structured_gpp_broad_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_checkpoint import (
    load_structured_gpp_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
    structured_gpp_direct_loss,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_error_sufficient_statistics,
    initialize_target_oriented_adamw,
    normalized_error_from_sufficient_statistics,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.audit_target_oriented_structured_gpp_data_coverage import (
    _contract_updates,
    _merged_day_keys,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _load_artifact,
    _replace_adapter,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _hard_failure_count,
    _make_steps,
)
from scripts.hpc.run_target_oriented_source_shaped_branch_training import (
    _make_optimizer_step,
)
from scripts.hpc.run_target_oriented_structured_gpp_broad_preflight import (
    _direct_forward,
    _prepare_batch,
)
from scripts.hpc.run_target_oriented_structured_gpp_short_horizon_screen import (
    _load_screen_contract,
    _load_training_report,
)
from scripts.hpc.run_target_oriented_structured_gpp_streaming_throughput import (
    COVERAGE_REPORT_SHA256,
    _direct_statistics_and_gradient,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
    _make_direct_steps,
    _one_day_hard,
    _retained_parameter_report,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_broad_training_result_v1"
SCHEDULE_REPORT_SHA256 = "7de9e840eb8773b4ab9eaeb136cf97c295a13079b6589eebfea970e83679e252"
BROAD_PREFLIGHT_REPORT_SHA256 = "0962b18d30e5f1b718554897064f7ca8d859234d24b93068f24ed034e7a39b0d"
BROAD_SCHEDULE_SHA256 = "b4295558406fd627837b012897585992ccd4e75042f7e76171b2a286bc7d6b8b"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_gradient_check(jax: Any, jnp: Any) -> Any:
    @jax.jit
    def check(gradient: Any) -> tuple[Any, Any]:
        leaves = jax.tree_util.tree_leaves(gradient)
        nonfinite = sum(
            (jnp.sum(~jnp.isfinite(value), dtype=jnp.int32) for value in leaves),
            jnp.asarray(0, dtype=jnp.int32),
        )
        nonzero_leaves = sum(
            (jnp.any(value != 0.0).astype(jnp.int32) for value in leaves),
            jnp.asarray(0, dtype=jnp.int32),
        )
        return nonfinite, nonzero_leaves

    return check


def _monitor_panel(
    runtime: Mapping[str, Any],
    specialist: Any,
    direct_steps: Mapping[str, Any],
    references: Mapping[tuple[str, int, str, str], Any],
    monitor_entries: Sequence[Mapping[str, Any]],
    *,
    optimizer_update: int,
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    records = []
    total = None
    started = time.perf_counter()
    for entry in monitor_entries:
        identity = (
            str(entry["landpoint_id"]),
            int(entry["year"]),
            str(entry["parent_shard_sha256"]),
            str(entry["typed_shard_sha256"]),
        )
        rows = tuple(int(value) for value in entry["monitor_rows"])
        if identity not in references or len(rows) != 16:
            raise ValueError("structured GPP broad monitor identity drift")
        host_batch, preparation = _prepare_batch(
            runtime,
            ((references[identity], rows),),
            expected_days=16,
        )
        batch = jax.device_put(host_batch)
        statistics = _direct_forward(jax, specialist, batch, direct_steps)
        error = normalized_error_from_sufficient_statistics(statistics)
        loss = float(np.asarray(structured_gpp_direct_loss(statistics)))
        count = int(np.asarray(statistics.count))
        if not np.isfinite(loss) or count != 16:
            raise RuntimeError("structured GPP broad monitor point-year failed")
        total = statistics if total is None else add_error_sufficient_statistics(total, statistics)
        records.append(
            {
                "landpoint_id": identity[0],
                "year": identity[1],
                "parent_shard_sha256": identity[2],
                "typed_shard_sha256": identity[3],
                "rows": list(rows),
                "loss": loss,
                "mse": float(np.asarray(error.mse)),
                "normalized_bias": float(np.asarray(error.mean_bias)),
                "bias_squared": float(np.asarray(error.bias_squared)),
                "count": count,
                "preparation_seconds": preparation["total_seconds"],
            }
        )
    if total is None:
        raise ValueError("structured GPP broad monitor cannot be empty")
    aggregate_error = normalized_error_from_sufficient_statistics(total)
    aggregate = {
        "loss": float(np.asarray(structured_gpp_direct_loss(total))),
        "mse": float(np.asarray(aggregate_error.mse)),
        "normalized_bias": float(np.asarray(aggregate_error.mean_bias)),
        "bias_squared": float(np.asarray(aggregate_error.bias_squared)),
        "count": float(np.asarray(aggregate_error.count)),
    }
    if not all(np.isfinite(value) for value in aggregate.values()) or int(aggregate["count"]) != 2048:
        raise RuntimeError("structured GPP broad aggregate monitor failed")
    return {
        "optimizer_update": int(optimizer_update),
        "aggregate": aggregate,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _monitor_comparison(
    initial: Mapping[str, Any],
    current: Mapping[str, Any],
) -> Mapping[str, Any]:
    before = float(initial["aggregate"]["loss"])
    after = float(current["aggregate"]["loss"])
    initial_records = {
        (value["landpoint_id"], int(value["year"]), value["parent_shard_sha256"], value["typed_shard_sha256"]): value
        for value in initial["records"]
    }
    improved = 0
    for value in current["records"]:
        identity = (
            value["landpoint_id"],
            int(value["year"]),
            value["parent_shard_sha256"],
            value["typed_shard_sha256"],
        )
        if float(value["loss"]) < float(initial_records[identity]["loss"]):
            improved += 1
    return {
        "initial_loss": before,
        "current_loss": after,
        "loss_reduction": 1.0 - after / before,
        "improved_point_years": improved,
        "initial_normalized_bias": float(initial["aggregate"]["normalized_bias"]),
        "current_normalized_bias": float(current["aggregate"]["normalized_bias"]),
    }


def _is_admission_only(maximum_updates: int) -> bool:
    return int(maximum_updates) < min(CHECKPOINT_UPDATES)


def _hard_counts(value: Any) -> Mapping[str, int]:
    return {name: int(np.asarray(item)) for name, item in value._asdict().items()}


def _stage_metadata(
    monitor_panels: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    stopped_early: bool,
) -> Mapping[str, Any]:
    return {
        "monitor_panels": list(monitor_panels),
        "checkpoints": list(checkpoints),
        "stopped_early": bool(stopped_early),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    contract = load_structured_gpp_broad_training_contract(args.broad_training_contract)
    maximum_updates = (
        int(contract.payload["data"]["optimizer_updates_maximum"])
        if args.maximum_updates is None
        else int(args.maximum_updates)
    )
    if not 1 <= maximum_updates <= int(contract.payload["data"]["optimizer_updates_maximum"]):
        raise ValueError("structured GPP broad training update budget drift")
    if sha256_file(args.schedule_report) != SCHEDULE_REPORT_SHA256:
        raise ValueError("structured GPP broad training schedule-report hash drift")
    if sha256_file(args.broad_preflight_report) != BROAD_PREFLIGHT_REPORT_SHA256:
        raise ValueError("structured GPP broad training preflight-report hash drift")
    if sha256_file(args.coverage_report) != COVERAGE_REPORT_SHA256:
        raise ValueError("structured GPP broad training coverage-report hash drift")
    schedule_report = json.loads(args.schedule_report.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
    if (
        schedule_report["identity"]["broad_schedule_sha256"] != BROAD_SCHEDULE_SHA256
        or schedule_report["identity"]["broad_preflight_contract_sha256"]
        != contract.payload["identity"]["broad_preflight_contract_sha256"]
        or coverage.get("status") != "passed"
    ):
        raise ValueError("structured GPP broad training audit identity drift")

    screen = _load_screen_contract(args.screen_contract)
    screen_identity = screen["identity"]
    training_report = _load_training_report(args.joint_training_report, screen)

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
        load_distribution_aware_checkpoint,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
        build_target_oriented_structured_gpp_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        initialize_structured_gpp_parameters,
        load_structured_gpp_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_training import (
        load_structured_gpp_joint_training_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
        load_structured_gpp_warm_start_contract,
    )

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    target_contract = context["target_contract"]
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    distribution_checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=screen_identity["parent_distribution_checkpoint_sha256"],
        identity=distribution_report["identity"],
        completed_updates=250,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(distribution_report["training"]["clip_threshold"]),
    )
    initialized_parent = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=distribution_contract,
    )
    parent_parameters = _replace_adapter(
        initialized_parent, jax.device_put(distribution_checkpoint["adapter"])
    )
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("structured GPP broad training parent parameter count drift")
    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    initialized = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    parent_checkpoint = load_structured_gpp_checkpoint(
        args.selected_checkpoint,
        expected_sha256=screen_identity["selected_checkpoint_sha256"],
        identity=training_report["identity"],
        completed_updates=int(screen_identity["selected_checkpoint_optimizer_step"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(training_report["training"]["clip_threshold"]),
    )
    initial_specialist = jax.device_put(parent_checkpoint["specialist"])
    initial_parameters = replace_specialist(initialized, initial_specialist)
    if (
        specialist_element_count(initial_specialist) != SPECIALIST_PARAMETER_COUNT
        or _tree_size(jax, initial_parameters) != CANDIDATE_PARAMETER_COUNT
    ):
        raise ValueError("structured GPP broad training candidate parameter count drift")

    references = tuple(runtime["index"].select(spatial_split="train", temporal_split="train"))
    reference_map = {reference_identity(value): value for value in references}
    warm_contract = load_structured_gpp_warm_start_contract(args.warm_start_contract)
    joint_contract = load_structured_gpp_joint_training_contract(args.joint_training_contract)
    warm_updates, joint_updates = _contract_updates(warm_contract, joint_contract)
    old_entries = {int(value["update"]): value for value in context["schedule"]["entries"]}
    prior = _merged_day_keys(old_entries[value] for value in (*warm_updates, *joint_updates))["all"]
    monitor_entries = tuple(coverage["proposed_monitor"]["entries"])
    monitor_point_years = {
        (
            str(value["landpoint_id"]),
            int(value["year"]),
            str(value["parent_shard_sha256"]),
            str(value["typed_shard_sha256"]),
        )
        for value in monitor_entries
    }
    schedule_data = contract.payload["data"]
    schedule = build_broad_training_schedule(
        references,
        prior_exposed_days=prior,
        monitor_point_years=monitor_point_years,
        seed=20260830,
        optimizer_updates=5000,
        point_years_per_update=int(schedule_data["point_years_per_update"]),
        days_per_point_year=int(schedule_data["days_per_point_year_per_update"]),
    )
    if schedule.sha256 != BROAD_SCHEDULE_SHA256:
        raise ValueError("structured GPP broad training rebuilt schedule drift")

    direct_steps = _make_direct_steps(runtime, head, initialized)
    full_steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        science_flux_basis=amendment.flux_basis,
    )
    physical_artifact = _load_artifact(args, context, 251)
    gradient_check = _make_gradient_check(jax, jnp)
    optimizer_step = _make_optimizer_step(jax, jnp, contract, contract.clip_threshold)
    learning_rate = float(contract.optimizer["learning_rate"])
    memory_before = _memory_report(runtime["devices"][0])
    identity = {
        "git_head": _git_head(),
        "broad_training_contract_sha256": contract.sha256,
        "schedule_report_sha256": SCHEDULE_REPORT_SHA256,
        "broad_schedule_sha256": BROAD_SCHEDULE_SHA256,
        "broad_preflight_report_sha256": BROAD_PREFLIGHT_REPORT_SHA256,
        "coverage_report_sha256": COVERAGE_REPORT_SHA256,
        "selected_parent_checkpoint_sha256": screen_identity["selected_checkpoint_sha256"],
    }

    recovery_path = output_dir / "recovery.pkl"
    recovery = load_broad_recovery(
        recovery_path,
        identity=identity,
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=contract.clip_threshold,
        maximum_updates=maximum_updates,
    )
    if recovery is None:
        initial_monitor = _monitor_panel(
            runtime,
            initial_specialist,
            direct_steps,
            reference_map,
            monitor_entries,
            optimizer_update=0,
        )
        initial_hard = _hard_counts(
            jax.device_get(
                _one_day_hard(
                    jax,
                    jnp,
                    initial_parameters,
                    physical_artifact.batch,
                    full_steps,
                )
            )
        )
        if _hard_failure_count(initial_hard) != 0:
            raise RuntimeError("structured GPP broad training initial physical gate failed")
        specialist = initial_specialist
        optimizer = initialize_target_oriented_adamw(specialist)
        completed = 0
        update_records = []
        monitor_panels = [
            {
                "optimizer_update": 0,
                "panel": initial_monitor,
                "comparison_to_initial": _monitor_comparison(initial_monitor, initial_monitor),
                "physical_hard_counts": initial_hard,
            }
        ]
        checkpoints = []
        stopped_early = False
        latest_recovery = None
    else:
        specialist, optimizer = jax.device_put((recovery["specialist"], recovery["optimizer"]))
        completed = int(recovery["completed_updates"])
        update_records = list(recovery["update_records"])
        initial_monitor = recovery["initial_monitor"]
        metadata = recovery.get("stage_metadata", {})
        monitor_panels = list(metadata.get("monitor_panels", ()))
        checkpoints = list(metadata.get("checkpoints", ()))
        stopped_early = bool(metadata.get("stopped_early", False))
        latest_recovery = {
            "path": str(recovery_path),
            "sha256": sha256_file(recovery_path),
            "completed_updates": completed,
            "promotion_eligible": False,
        }
        if stopped_early:
            raise ValueError("structured GPP broad training cannot resume after early stop")
    jax.block_until_ready((specialist, optimizer))

    recovery_interval = int(contract.payload["bounded_stage"]["recovery_interval_updates"])
    for optimizer_update in range(completed + 1, maximum_updates + 1):
        update_started = time.perf_counter()
        selections = schedule.batches[optimizer_update - 1].selections
        host_batch, preparation = _prepare_batch(
            runtime,
            tuple((value.reference, value.rows) for value in selections),
        )
        batch = jax.device_put(host_batch)
        statistics, gradient, compute = _direct_statistics_and_gradient(
            jax, jnp, specialist, batch, direct_steps
        )
        loss = float(np.asarray(structured_gpp_direct_loss(statistics)))
        count = int(np.asarray(statistics.count))
        gradient_nonfinite, nonzero_leaves = gradient_check(gradient)
        proposal = optimizer_step(
            specialist,
            gradient,
            optimizer,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        jax.block_until_ready((gradient_nonfinite, nonzero_leaves, proposal))
        updated, next_optimizer, raw_norm, clip_scale, proposal_nonfinite, step_norm = proposal
        record = {
            "optimizer_update": optimizer_update,
            "direct_gpp_loss": loss,
            "evaluated_values": count,
            "learning_rate": learning_rate,
            "raw_gradient_norm": float(np.asarray(raw_norm)),
            "clip_scale": float(np.asarray(clip_scale)),
            "clipped": float(np.asarray(clip_scale)) < 1.0,
            "parameter_step_l2": float(np.asarray(step_norm)),
            "preparation_seconds": preparation,
            "compute_seconds": compute,
            "total_seconds": time.perf_counter() - update_started,
        }
        if (
            not np.isfinite(loss)
            or loss <= 0.0
            or count != 208
            or int(np.asarray(gradient_nonfinite)) != 0
            or int(np.asarray(nonzero_leaves)) != 129
            or int(np.asarray(proposal_nonfinite)) != 0
            or int(np.asarray(next_optimizer.step)) != optimizer_update
            or not np.isfinite(record["raw_gradient_norm"])
            or record["raw_gradient_norm"] <= 0.0
            or not 0.0 < record["clip_scale"] <= 1.0
            or not np.isfinite(record["parameter_step_l2"])
            or record["parameter_step_l2"] <= 0.0
        ):
            raise RuntimeError(f"structured GPP broad training update {optimizer_update} failed")
        specialist, optimizer = updated, next_optimizer
        update_records.append(record)
        print(
            f"structured_gpp broad_training update={optimizer_update}/{maximum_updates} "
            f"loss={loss:.9g} norm={record['raw_gradient_norm']:.9g} "
            f"clip={record['clip_scale']:.6g} seconds={record['total_seconds']:.3f}",
            flush=True,
        )

        current_monitor = monitor_panels[-1]["panel"]
        if optimizer_update in CHECKPOINT_UPDATES and optimizer_update <= maximum_updates:
            current_monitor = _monitor_panel(
                runtime,
                specialist,
                direct_steps,
                reference_map,
                monitor_entries,
                optimizer_update=optimizer_update,
            )
            proposed_parameters = replace_specialist(initialized, specialist)
            hard = _hard_counts(
                jax.device_get(
                    _one_day_hard(
                        jax,
                        jnp,
                        proposed_parameters,
                        physical_artifact.batch,
                        full_steps,
                    )
                )
            )
            comparison = _monitor_comparison(initial_monitor, current_monitor)
            if _hard_failure_count(hard) != 0:
                raise RuntimeError("structured GPP broad training checkpoint physical gate failed")
            monitor_panels.append(
                {
                    "optimizer_update": optimizer_update,
                    "panel": current_monitor,
                    "comparison_to_initial": comparison,
                    "physical_hard_counts": hard,
                }
            )
            host_specialist, host_optimizer = jax.device_get((specialist, optimizer))
            checkpoint_path = output_dir / f"stage-update-{optimizer_update:04d}.pkl"
            checkpoint = write_broad_checkpoint(
                checkpoint_path,
                identity=identity,
                specialist=host_specialist,
                optimizer=host_optimizer,
                completed_updates=optimizer_update,
                parameter_count=SPECIALIST_PARAMETER_COUNT,
                clip_threshold=contract.clip_threshold,
                update_records=update_records,
                initial_monitor=initial_monitor,
                current_monitor=current_monitor,
                stage_metadata=_stage_metadata(
                    monitor_panels,
                    checkpoints,
                    stopped_early=False,
                ),
            )
            checkpoints.append({"optimizer_update": optimizer_update} | checkpoint)
            print(
                f"structured_gpp broad_training monitor={optimizer_update} "
                f"reduction={comparison['loss_reduction']:.9g} "
                f"improved={comparison['improved_point_years']}/128",
                flush=True,
            )
            if optimizer_update == 500 and comparison["loss_reduction"] <= 0.0:
                stopped_early = True

        if (
            optimizer_update % recovery_interval == 0
            or optimizer_update in CHECKPOINT_UPDATES
            or optimizer_update == maximum_updates
        ):
            host_specialist, host_optimizer = jax.device_get((specialist, optimizer))
            latest_recovery = write_broad_recovery(
                recovery_path,
                identity=identity,
                specialist=host_specialist,
                optimizer=host_optimizer,
                completed_updates=optimizer_update,
                parameter_count=SPECIALIST_PARAMETER_COUNT,
                clip_threshold=contract.clip_threshold,
                update_records=update_records,
                initial_monitor=initial_monitor,
                current_monitor=current_monitor,
                stage_metadata=_stage_metadata(
                    monitor_panels,
                    checkpoints,
                    stopped_early=stopped_early,
                ),
            )
        if stopped_early:
            break

    completed_updates = int(np.asarray(optimizer.step))
    retained = _retained_parameter_report(
        jax,
        initial_parameters,
        replace_specialist(initialized, specialist),
    )
    checkpoint_by_update = {int(value["optimizer_update"]): value for value in checkpoints}
    completed_monitor_panels = [value for value in monitor_panels if int(value["optimizer_update"]) > 0]
    if completed_monitor_panels:
        selected_panel = min(
            completed_monitor_panels,
            key=lambda value: float(value["panel"]["aggregate"]["loss"]),
        )
        selected_update = int(selected_panel["optimizer_update"])
        selected_checkpoint = checkpoint_by_update[selected_update]
        selected_comparison = selected_panel["comparison_to_initial"]
        acceptance = (
            float(selected_comparison["loss_reduction"])
            >= float(contract.payload["train_only_monitor"]["minimum_final_aggregate_loss_reduction"])
            and int(selected_comparison["improved_point_years"])
            >= int(contract.payload["train_only_monitor"]["minimum_improved_point_years"])
            and abs(float(selected_comparison["current_normalized_bias"]))
            <= abs(float(selected_comparison["initial_normalized_bias"]))
            and retained["exactly_unchanged"]
        )
    else:
        selected_panel = None
        selected_checkpoint = None
        selected_comparison = None
        acceptance = False
    admission_only = _is_admission_only(maximum_updates)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "admission_passed" if admission_only else "passed",
        "identity": identity,
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_or_sealed_test_values_read": False,
            "consumed_96_window_panel_read": False,
            "unique_training_days": 208 * completed_updates,
            "retained_parent_parameters_trainable": False,
            "checkpoint_promotion_eligible": False,
        },
        "training": {
            "requested_maximum_updates": maximum_updates,
            "completed_updates": completed_updates,
            "stopped_early": stopped_early,
            "learning_rate": learning_rate,
            "clip_threshold": contract.clip_threshold,
            "clipped_update_count": sum(bool(value["clipped"]) for value in update_records),
            "updates": update_records,
        },
        "initial_monitor": initial_monitor,
        "monitor_panels": monitor_panels,
        "checkpoints": checkpoints,
        "selected_monitor_panel": selected_panel,
        "selected_checkpoint": selected_checkpoint,
        "selected_comparison": selected_comparison,
        "latest_recovery": latest_recovery,
        "model": {
            "candidate_parameter_count": CANDIDATE_PARAMETER_COUNT,
            "trainable_specialist_parameter_count": SPECIALIST_PARAMETER_COUNT,
            "retained_parent": retained,
        },
        "acceptance": {
            "admission_only": admission_only,
            "broad_stage_accepted": acceptance,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "request_paid_stage_1_resource_approval"
            if admission_only
            else contract.payload["next_gate"][
                "on_stage_acceptance" if acceptance else "on_stage_failure"
            ]
        ),
    }
    _atomic_json(report_path, report)
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
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--distribution-contract", type=Path, required=True)
    parser.add_argument("--distribution-training-report", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--structured-contract", type=Path, required=True)
    parser.add_argument("--joint-training-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--warm-start-contract", type=Path, required=True)
    parser.add_argument("--joint-training-contract", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--schedule-report", type=Path, required=True)
    parser.add_argument("--broad-preflight-report", type=Path, required=True)
    parser.add_argument("--broad-training-contract", type=Path, required=True)
    parser.add_argument("--maximum-updates", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
