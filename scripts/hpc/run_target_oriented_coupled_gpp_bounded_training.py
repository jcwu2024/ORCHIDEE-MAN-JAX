"""Run the first bounded coupled-GPP training stage or its discarded admission."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_bounded_training import (
    build_coupled_gpp_training_schedule,
    finite_step_accepted,
    load_coupled_gpp_bounded_training_contract,
    monitor_continuation_accepted,
    selected_milestone,
    source_changes,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_checkpoint import (
    load_coupled_gpp_checkpoint,
    load_coupled_gpp_recovery,
    write_coupled_gpp_checkpoint,
    write_coupled_gpp_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    GRADIENT_SOURCES,
    RETAINED_PARAMETER_COUNT,
    TRAINABLE_FAMILIES,
    TRAINABLE_PARAMETER_COUNT,
    build_reachable_state_anchor_layout,
    coupled_gpp_trainable_tree,
    load_coupled_gpp_objective_calibration_contract,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_optimizer_preflight import (
    four_source_macro_objectives,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
    target_oriented_adamw_update,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    _add_gradient,
    _load_context,
    _tree_size,
    _zero_gradient,
)
from scripts.hpc.calibrate_target_oriented_coupled_gpp_objective import (
    _make_direct_steps,
    _make_state_steps,
    _state_gradient,
    _validate_reachability_report,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _load_artifact,
)
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _direct_gradient,
    _horizon_gradient,
    _one_day_hard,
    _tree_cosine,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_optimizer_preflight import (
    _fixed_parameter_report,
    _mean_gradient,
    _source_losses,
    _state_forward,
    _weighted_gradient,
)
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _hard_failure_count,
    _make_steps,
)
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import (
    _make_proposal_check,
    _tree_scalar_report,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _direct_forward,
    _hard_host,
    _horizon_forward,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_bounded_training_result_v1"
ADMISSION_SCHEMA_VERSION = "target_oriented_coupled_gpp_bounded_training_runner_admission_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _validate_backoff_report(path: Path, contract: Any) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = contract.payload["identity"]["optimizer_backoff_report_sha256"]
    acceptance = report.get("acceptance", {})
    if (
        sha256_file(path) != expected
        or report.get("schema_version") != "target_oriented_coupled_gpp_optimizer_backoff_result_v1"
        or report.get("status") != "passed"
        or not all(bool(value) for value in acceptance.values())
        or report.get("policy", {}).get("optimizer_updates_persisted") != 0
    ):
        raise ValueError("coupled GPP bounded training predecessor is not accepted")
    return report


def _mean_losses(records: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    values = tuple(records)
    if not values:
        raise ValueError("coupled GPP mean loss requires records")
    return {name: float(np.mean([float(record[name]) for record in values])) for name in GRADIENT_SOURCES}


def _all_hard_zero(records: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        _hard_failure_count(value) == 0
        for record in records
        for name, value in record.items()
        if name != "artifact_update"
    )


def _evaluate_artifacts(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    *,
    parameters: Any,
    trainable: Any,
    steps: Mapping[str, Any],
    direct_steps: Mapping[str, Any],
    state_steps: Mapping[str, Any],
    layout: Any,
    updates: Sequence[int],
    flux_basis: Any,
    coefficients: Mapping[str, float],
    label: str,
) -> Mapping[str, Any]:
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    records = []
    for ordinal, update in enumerate(updates, start=1):
        started = time.perf_counter()
        artifact = _load_artifact(args, context, int(update))
        direct, direct_seconds = _direct_forward(jax, trainable, artifact.batch, direct_steps)
        state, state_seconds = _state_forward(jax, trainable, artifact.batch, state_steps, layout)
        one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
        seven, seven_hard, seven_seconds = _horizon_forward(
            jax,
            jnp,
            parameters,
            artifact.batch,
            steps,
            horizon=7,
            flux_basis=flux_basis,
        )
        thirty, thirty_hard, thirty_seconds = _horizon_forward(
            jax,
            jnp,
            parameters,
            artifact.batch,
            steps,
            horizon=30,
            flux_basis=flux_basis,
        )
        losses = _source_losses(direct, state, seven, thirty)
        hard = {
            "one_day": _hard_host(jax.device_get(one_hard)),
            "seven_day": _hard_host(jax.device_get(seven_hard)),
            "thirty_day": _hard_host(jax.device_get(thirty_hard)),
        }
        if not all(np.isfinite(value) and value >= 0.0 for value in losses.values()):
            raise RuntimeError(f"coupled GPP {label} artifact {update} has invalid loss")
        records.append(
            {
                "artifact_update": int(update),
                "runtime_year": int(context["schedule"]["entries"][int(update) - 1]["runtime_year"]),
                "source_losses": losses,
                "hard_counts": hard,
                "timing_seconds": {
                    "direct": direct_seconds,
                    "state": state_seconds,
                    "seven_day": seven_seconds,
                    "thirty_day": thirty_seconds,
                    "total": time.perf_counter() - started,
                },
            }
        )
        print(
            f"coupled_gpp {label}={ordinal}/{len(updates)} artifact={update} "
            f"seconds={records[-1]['timing_seconds']['total']:.3f}",
            flush=True,
        )
    hard_records = [{"artifact_update": value["artifact_update"]} | value["hard_counts"] for value in records]
    if not _all_hard_zero(hard_records):
        raise RuntimeError(f"coupled GPP {label} physical hard gate failed")
    means = _mean_losses(tuple(value["source_losses"] for value in records))
    return {
        "label": label,
        "artifact_updates": [int(value) for value in updates],
        "source_mean_losses": means,
        "weighted_objective": four_source_macro_objectives(means, coefficients),
        "records": records,
        "hard_gate_passed": True,
    }


def _training_proposal(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    *,
    parameters: Any,
    trainable: Any,
    optimizer: Any,
    optimizer_step: Any,
    proposal_check: Any,
    steps: Mapping[str, Any],
    gradient_steps: Mapping[str, Any],
    direct_steps: Mapping[str, Any],
    state_steps: Mapping[str, Any],
    layout: Any,
    group: Sequence[int],
    flux_basis: Any,
    coefficients: Mapping[str, float],
    relative_tolerance: float,
    optimizer_update: int,
    return_internal: bool = False,
) -> Mapping[str, Any]:
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    update_started = time.perf_counter()
    gradient_sums = {name: _zero_gradient(jax, jnp, trainable) for name in GRADIENT_SOURCES}
    before_records = []
    pre_hard_records = []
    artifact_records = []
    for ordinal, update in enumerate(group, start=1):
        artifact_started = time.perf_counter()
        artifact = _load_artifact(args, context, int(update))
        direct, direct_gradient, direct_forward, direct_backward = _direct_gradient(
            jax, jnp, trainable, artifact.batch, direct_steps
        )
        state, state_gradient, state_forward, state_backward = _state_gradient(
            jax, jnp, trainable, artifact.batch, state_steps, layout
        )
        one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
        seven, seven_hard, seven_gradient, seven_timing = _horizon_gradient(
            jax,
            jnp,
            parameters,
            trainable,
            artifact.batch,
            steps,
            gradient_steps,
            horizon=7,
            flux_basis=flux_basis,
        )
        thirty, thirty_hard, thirty_gradient, thirty_timing = _horizon_gradient(
            jax,
            jnp,
            parameters,
            trainable,
            artifact.batch,
            steps,
            gradient_steps,
            horizon=30,
            flux_basis=flux_basis,
        )
        current = {
            GRADIENT_SOURCES[0]: direct_gradient,
            GRADIENT_SOURCES[1]: state_gradient,
            GRADIENT_SOURCES[2]: seven_gradient,
            GRADIENT_SOURCES[3]: thirty_gradient,
        }
        gradient_sums = {name: _add_gradient(jax, jnp, gradient_sums[name], current[name]) for name in GRADIENT_SOURCES}
        jax.block_until_ready(gradient_sums)
        losses = _source_losses(direct, state, seven, thirty)
        before_records.append(losses)
        hard = {
            "artifact_update": int(update),
            "one_day": _hard_host(jax.device_get(one_hard)),
            "seven_day": _hard_host(jax.device_get(seven_hard)),
            "thirty_day": _hard_host(jax.device_get(thirty_hard)),
        }
        pre_hard_records.append(hard)
        artifact_records.append(
            {
                "artifact_update": int(update),
                "runtime_year": int(context["schedule"]["entries"][int(update) - 1]["runtime_year"]),
                "source_losses_before": losses,
                "hard_counts_before": hard,
                "timing_seconds": {
                    "direct_forward": direct_forward,
                    "direct_gradient": direct_backward,
                    "state_forward": state_forward,
                    "state_gradient": state_backward,
                    "seven_day": seven_timing,
                    "thirty_day": thirty_timing,
                    "total": time.perf_counter() - artifact_started,
                },
            }
        )
        print(
            f"coupled_gpp training update={optimizer_update} gradient={ordinal}/8 "
            f"artifact={update} seconds={artifact_records[-1]['timing_seconds']['total']:.3f}",
            flush=True,
        )
        del current, direct_gradient, state_gradient, seven_gradient, thirty_gradient

    if not _all_hard_zero(pre_hard_records):
        raise RuntimeError(f"coupled GPP training update {optimizer_update} pre-update hard gate failed")
    source_gradients = {name: _mean_gradient(jax, gradient_sums[name], len(group)) for name in GRADIENT_SOURCES}
    combined_gradient = _weighted_gradient(jax, source_gradients, coefficients)
    source_reports = {name: _tree_scalar_report(jax, jnp, source_gradients[name]) for name in GRADIENT_SOURCES}
    combined_report = _tree_scalar_report(jax, jnp, combined_gradient)
    source_to_combined = {
        name: _tree_cosine(jax, source_gradients[name], combined_gradient) for name in GRADIENT_SOURCES
    }
    updated_trainable, updated_optimizer, raw_norm, clip_scale = optimizer_step(trainable, combined_gradient, optimizer)
    jax.block_until_ready((updated_trainable, updated_optimizer, raw_norm, clip_scale))
    parameter_nonfinite, optimizer_nonfinite, delta_squared = proposal_check(
        trainable, updated_trainable, updated_optimizer
    )
    jax.block_until_ready((parameter_nonfinite, optimizer_nonfinite, delta_squared))
    parameter_step = jax.tree_util.tree_map(lambda after, before: after - before, updated_trainable, trainable)
    family_steps = {name: _tree_scalar_report(jax, jnp, parameter_step[name]) for name in TRAINABLE_FAMILIES}
    proposal_parameters = replace_coupled_gpp_trainable(parameters, updated_trainable)
    post = _evaluate_artifacts(
        args,
        context,
        parameters=proposal_parameters,
        trainable=updated_trainable,
        steps=steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
        layout=layout,
        updates=group,
        flux_basis=flux_basis,
        coefficients=coefficients,
        label=f"update_{optimizer_update}_post",
    )
    before_means = _mean_losses(before_records)
    after_means = post["source_mean_losses"]
    changes = source_changes(before_means, after_means)
    engineering_passed = bool(
        all(
            value["all_finite"]
            and value["element_count"] == TRAINABLE_PARAMETER_COUNT
            and value["l2_norm"] > 0.0
            and value["nonzero_leaf_count"] == value["leaf_count"]
            for value in (*source_reports.values(), combined_report)
        )
        and int(np.asarray(parameter_nonfinite)) == 0
        and int(np.asarray(optimizer_nonfinite)) == 0
        and float(np.asarray(delta_squared)) > 0.0
        and np.isfinite(float(np.asarray(raw_norm)))
        and 0.0 < float(np.asarray(clip_scale)) <= 1.0
        and int(np.asarray(updated_optimizer.step)) == optimizer_update
        and all(report["all_finite"] and report["l2_norm"] > 0.0 for report in family_steps.values())
    )
    if not engineering_passed:
        raise RuntimeError(f"coupled GPP training update {optimizer_update} proposal failed")
    accepted = finite_step_accepted(
        before_means,
        after_means,
        coefficients=coefficients,
        relative_tolerance=relative_tolerance,
    )
    result = {
        "accepted": accepted,
        "updated_trainable": updated_trainable,
        "updated_optimizer": updated_optimizer,
        "record": {
            "optimizer_update": optimizer_update,
            "artifact_updates": [int(value) for value in group],
            "source_mean_losses_before": before_means,
            "source_mean_losses_after": after_means,
            "source_changes": changes,
            "weighted_objective_before": four_source_macro_objectives(before_means, coefficients),
            "weighted_objective_after": four_source_macro_objectives(after_means, coefficients),
            "finite_step_accepted": accepted,
            "gradients": {
                "sources": source_reports,
                "combined": combined_report,
                "source_to_combined_cosines": source_to_combined,
            },
            "optimizer": {
                "learning_rate": context["coupled_learning_rate"],
                "raw_gradient_norm": float(np.asarray(raw_norm)),
                "clip_scale": float(np.asarray(clip_scale)),
                "parameter_delta_l2": float(np.sqrt(float(np.asarray(delta_squared)))),
                "family_parameter_steps": family_steps,
            },
            "artifact_records": artifact_records,
            "post_evaluation": post,
            "timing_seconds": {"total": time.perf_counter() - update_started},
        },
    }
    if return_internal:
        result["internal"] = {
            "source_gradients": source_gradients,
            "combined_gradient": combined_gradient,
            "parameter_step": parameter_step,
        }
    return result


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / ("admission-report.json" if args.admission_only else "report.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()

    contract = load_coupled_gpp_bounded_training_contract(args.bounded_training_contract)
    objective_contract = load_coupled_gpp_objective_calibration_contract(args.objective_calibration_contract)
    identity_contract = contract.payload["identity"]
    if objective_contract.sha256 != identity_contract["objective_calibration_contract_sha256"]:
        raise ValueError("coupled GPP bounded training objective-contract drift")
    _validate_backoff_report(args.optimizer_backoff_report, contract)
    layout = build_reachable_state_anchor_layout(objective_contract)
    _validate_reachability_report(args.reachability_report, objective_contract, layout)

    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )
    from scripts.hpc.audit_target_oriented_gpp_state_reachability import (
        _build_candidate,
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
    if (
        amendment.sha256 != identity_contract["flux_basis_amendment_sha256"]
        or sha256_file(args.prepared_manifest) != identity_contract["prepared_manifest_sha256"]
        or context["training_contract"].sha256 != identity_contract["base_training_contract_sha256"]
        or context["target_contract"].sha256 != identity_contract["target_contract_sha256"]
    ):
        raise ValueError("coupled GPP bounded training data identity drift")
    schedule = build_coupled_gpp_training_schedule()
    required_artifacts = {value for group in schedule.groups for value in group} | set(contract.monitor_updates)
    missing_artifacts = sorted(required_artifacts - set(context["prepared_records"]))
    if missing_artifacts:
        raise ValueError(f"coupled GPP bounded training prepared coverage drift: {missing_artifacts}")

    candidate = _build_candidate(runtime, args, context["template"])
    initialized_parameters = candidate["parameters"]
    head = candidate["head"]
    initialized_trainable = coupled_gpp_trainable_tree(initialized_parameters)
    fixed_parameters = replace_coupled_gpp_trainable(initialized_parameters, {})
    if (
        candidate["coupled_contract"].sha256 != identity_contract["coupled_candidate_contract_sha256"]
        or sha256_file(args.selected_checkpoint) != identity_contract["structured_checkpoint_sha256"]
        or _tree_size(jax, initialized_parameters) != int(identity_contract["candidate_parameter_count"])
        or _tree_size(jax, initialized_trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, fixed_parameters) != RETAINED_PARAMETER_COUNT
    ):
        raise ValueError("coupled GPP bounded training parameter boundary drift")

    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        context["target_contract"],
        science_flux_basis=amendment.flux_basis,
    )
    gradient_steps = steps["make_trainable_gradient_steps"](fixed_parameters, replace_coupled_gpp_trainable)
    direct_steps = _make_direct_steps(runtime, head, initialized_parameters)
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail_template=context["template"],
    )
    state_steps = _make_state_steps(
        runtime,
        transition,
        initialized_parameters,
        context["scales"].state.other,
        layout,
    )
    context["coupled_learning_rate"] = contract.learning_rate
    optimizer_step = jax.jit(
        lambda values, derivatives, state: target_oriented_adamw_update(
            values,
            derivatives,
            state,
            learning_rate=jnp.asarray(contract.learning_rate, dtype=jnp.float32),
            clip_norm=jnp.asarray(contract.clip_threshold, dtype=jnp.float32),
            contract=context["training_contract"],
        )
    )
    proposal_check = _make_proposal_check(jax, jnp)
    memory_before = _memory_report(runtime["devices"][0])
    identity = {
        "git_head": _git_head(),
        "bounded_training_contract_sha256": contract.sha256,
        "objective_calibration_contract_sha256": objective_contract.sha256,
        "optimizer_backoff_report_sha256": sha256_file(args.optimizer_backoff_report),
        "structured_checkpoint_sha256": sha256_file(args.selected_checkpoint),
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "coupled_candidate_contract_sha256": candidate["coupled_contract"].sha256,
        "schedule_sha256": schedule.sha256,
    }
    recovery_path = output_dir / "recovery.pkl"
    update_records: list[Mapping[str, Any]] = []
    milestone_monitors: list[Mapping[str, Any]] = []
    checkpoints: list[Mapping[str, Any]] = []
    trainable = initialized_trainable
    optimizer = initialize_target_oriented_adamw(trainable)
    initial_monitor = None

    if not args.admission_only:
        recovered = load_coupled_gpp_recovery(
            recovery_path,
            identity=identity,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
            clip_threshold=contract.clip_threshold,
            maximum_updates=8,
        )
        if recovered is not None:
            trainable = jax.device_put(recovered["trainable"])
            optimizer = jax.device_put(recovered["optimizer"])
            update_records = list(recovered["update_records"])
            milestone_monitors = list(recovered["milestone_monitors"])
            initial_monitor = recovered["initial_monitor"]
            for monitor in milestone_monitors:
                if bool(monitor["continuation_gate_passed"]):
                    update = int(monitor["optimizer_update"])
                    path = output_dir / f"stage-update-{update:04d}.pkl"
                    digest = sha256_file(path)
                    load_coupled_gpp_checkpoint(
                        path,
                        expected_sha256=digest,
                        identity=identity,
                        completed_updates=update,
                        parameter_count=TRAINABLE_PARAMETER_COUNT,
                        clip_threshold=contract.clip_threshold,
                    )
                    checkpoints.append(
                        {
                            "optimizer_update": update,
                            "path": str(path),
                            "sha256": digest,
                            "completed_updates": update,
                            "promotion_eligible": False,
                        }
                    )
            print(
                f"coupled_gpp bounded_training resumed={len(update_records)}",
                flush=True,
            )

    parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    if initial_monitor is None:
        initial_monitor = _evaluate_artifacts(
            args,
            context,
            parameters=parameters,
            trainable=trainable,
            steps=steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            updates=contract.monitor_updates,
            flux_basis=amendment.flux_basis,
            coefficients=contract.coefficients,
            label="initial_monitor",
        )

    maximum_updates = 1 if args.admission_only else 8
    stopped_reason = None
    if milestone_monitors and not bool(milestone_monitors[-1]["continuation_gate_passed"]):
        stopped_reason = "milestone_monitor_failed"
    relative_tolerance = float(
        contract.payload["finite_step_acceptance"]["all_four_macro_source_losses_must_not_increase_relative_tolerance"]
    )
    for index in range(len(update_records), maximum_updates):
        if stopped_reason is not None:
            break
        optimizer_update = index + 1
        parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
        proposal = _training_proposal(
            args,
            context,
            parameters=parameters,
            trainable=trainable,
            optimizer=optimizer,
            optimizer_step=optimizer_step,
            proposal_check=proposal_check,
            steps=steps,
            gradient_steps=gradient_steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            group=schedule.groups[index],
            flux_basis=amendment.flux_basis,
            coefficients=contract.coefficients,
            relative_tolerance=relative_tolerance,
            optimizer_update=optimizer_update,
        )
        if not proposal["accepted"]:
            stopped_reason = "finite_step_rejected"
            _atomic_json(
                output_dir / "rejected-proposal.json",
                {
                    "schema_version": "target_oriented_coupled_gpp_rejected_training_proposal_v1",
                    "identity": identity,
                    "proposal": proposal["record"],
                    "updates_retained_before_rejection": len(update_records),
                },
            )
            print(
                f"coupled_gpp bounded_training update={optimizer_update} rejected stop=true",
                flush=True,
            )
            break
        trainable = proposal["updated_trainable"]
        optimizer = proposal["updated_optimizer"]
        update_records.append(proposal["record"])
        print(
            f"coupled_gpp bounded_training update={optimizer_update}/{maximum_updates} "
            f"accepted seconds={proposal['record']['timing_seconds']['total']:.3f}",
            flush=True,
        )
        if not args.admission_only and optimizer_update in (4, 8):
            parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
            panel = _evaluate_artifacts(
                args,
                context,
                parameters=parameters,
                trainable=trainable,
                steps=steps,
                direct_steps=direct_steps,
                state_steps=state_steps,
                layout=layout,
                updates=contract.monitor_updates,
                flux_basis=amendment.flux_basis,
                coefficients=contract.coefficients,
                label=f"milestone_{optimizer_update}_monitor",
            )
            previous = initial_monitor if not milestone_monitors else milestone_monitors[-1]
            changes = source_changes(previous["source_mean_losses"], panel["source_mean_losses"])
            passed = monitor_continuation_accepted(
                previous["source_mean_losses"],
                panel["source_mean_losses"],
                coefficients=contract.coefficients,
            )
            panel = dict(
                panel,
                optimizer_update=optimizer_update,
                source_changes_from_previous_monitor=changes,
                continuation_gate_passed=passed,
            )
            milestone_monitors.append(panel)
            if passed:
                host_trainable, host_optimizer = jax.device_get((trainable, optimizer))
                path = output_dir / f"stage-update-{optimizer_update:04d}.pkl"
                checkpoint = write_coupled_gpp_checkpoint(
                    path,
                    identity=identity,
                    trainable=host_trainable,
                    optimizer=host_optimizer,
                    completed_updates=optimizer_update,
                    parameter_count=TRAINABLE_PARAMETER_COUNT,
                    clip_threshold=contract.clip_threshold,
                    update_records=update_records,
                    initial_monitor=initial_monitor,
                    milestone_monitors=milestone_monitors,
                )
                checkpoints.append({"optimizer_update": optimizer_update} | checkpoint)
            else:
                stopped_reason = "milestone_monitor_failed"
            print(
                f"coupled_gpp bounded_training milestone={optimizer_update} passed={passed}",
                flush=True,
            )
        if not args.admission_only:
            host_trainable, host_optimizer = jax.device_get((trainable, optimizer))
            write_coupled_gpp_recovery(
                recovery_path,
                identity=identity,
                trainable=host_trainable,
                optimizer=host_optimizer,
                completed_updates=optimizer_update,
                parameter_count=TRAINABLE_PARAMETER_COUNT,
                clip_threshold=contract.clip_threshold,
                update_records=update_records,
                initial_monitor=initial_monitor,
                milestone_monitors=milestone_monitors,
            )
            _atomic_json(
                output_dir / "progress.json",
                {
                    "schema_version": "target_oriented_coupled_gpp_bounded_training_progress_v1",
                    "identity": identity,
                    "completed_updates": optimizer_update,
                    "stopped_reason": stopped_reason,
                    "milestone_monitors": milestone_monitors,
                    "update_records": update_records,
                },
            )

    if args.admission_only:
        if len(update_records) != 1:
            raise RuntimeError("coupled GPP runner admission did not retain one ephemeral update")
        host_trainable, host_optimizer = jax.device_get((trainable, optimizer))
        temporary = output_dir / "admission-recovery.pkl"
        written = write_coupled_gpp_recovery(
            temporary,
            identity=identity,
            trainable=host_trainable,
            optimizer=host_optimizer,
            completed_updates=1,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
            clip_threshold=contract.clip_threshold,
            update_records=update_records,
            initial_monitor=initial_monitor,
            milestone_monitors=(),
        )
        load_coupled_gpp_recovery(
            temporary,
            identity=identity,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
            clip_threshold=contract.clip_threshold,
            maximum_updates=8,
        )
        temporary.unlink()
        report = {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "status": "passed",
            "identity": identity,
            "policy": {
                "optimizer_updates_proposed": 1,
                "optimizer_updates_persisted": 0,
                "checkpoint_written": False,
                "recovery_serialization_roundtrip_passed": True,
                "teacher_data_regenerated": False,
                "historical_v5_used": False,
                "fresh_validation_values_evaluated": False,
                "sealed_test_read": False,
            },
            "initial_monitor": initial_monitor,
            "discarded_update": update_records[0],
            "temporary_recovery_sha256": written["sha256"],
            "memory_before": memory_before,
            "memory_after": _memory_report(runtime["devices"][0]),
            "elapsed_seconds": time.perf_counter() - started,
            "next_gate": "request_exact_paid_resource_authorization_for_bounded_stage",
        }
        _atomic_json(report_path, report)
        return report

    selected = selected_milestone(milestone_monitors)
    selected_checkpoint = next(
        (value for value in checkpoints if value["optimizer_update"] == selected),
        None,
    )
    final_parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    fixed_report = _fixed_parameter_report(jax, initialized_parameters, final_parameters)
    checkpoint_unchanged = sha256_file(args.selected_checkpoint) == identity_contract["structured_checkpoint_sha256"]
    execution_passed = bool(
        len(update_records) == int(np.asarray(optimizer.step))
        and fixed_report["exactly_unchanged"]
        and fixed_report["element_count"] == RETAINED_PARAMETER_COUNT
        and checkpoint_unchanged
    )
    if not execution_passed:
        raise RuntimeError("coupled GPP bounded training execution gate failed")
    status = "passed" if selected > 0 else "stopped_scientific_gate"
    timings = np.asarray(
        [record["timing_seconds"]["total"] for record in update_records],
        dtype=np.float64,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "identity": identity,
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_updates_executed": len(update_records),
            "artifact_presentations": 8 * len(update_records),
            "checkpoint_promotion_eligible": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "candidate_parameter_count": _tree_size(jax, final_parameters),
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "fixed_parameters": fixed_report,
        },
        "training": {
            "maximum_updates": 8,
            "completed_updates": len(update_records),
            "stopped_reason": stopped_reason,
            "learning_rate": contract.learning_rate,
            "clip_threshold": contract.clip_threshold,
            "clipped_update_count": sum(record["optimizer"]["clip_scale"] < 1.0 for record in update_records),
            "update_seconds": None
            if not len(timings)
            else {
                "minimum": float(np.min(timings)),
                "median": float(np.median(timings)),
                "maximum": float(np.max(timings)),
                "total": float(np.sum(timings)),
            },
            "updates": update_records,
        },
        "initial_monitor": initial_monitor,
        "milestone_monitors": milestone_monitors,
        "checkpoints": checkpoints,
        "selected_milestone": selected,
        "selected_checkpoint": selected_checkpoint,
        "source_checkpoint_unchanged": checkpoint_unchanged,
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"]["on_checkpoint_selected" if selected > 0 else "on_no_checkpoint"],
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
    parser.add_argument("--distribution-checkpoint-sha256", required=True)
    parser.add_argument("--structured-contract", type=Path, required=True)
    parser.add_argument("--broad-training-report", type=Path, required=True)
    parser.add_argument("--coupled-contract", type=Path, required=True)
    parser.add_argument("--reachability-contract", type=Path, required=True)
    parser.add_argument("--gpu-admission-report", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--objective-calibration-contract", type=Path, required=True)
    parser.add_argument("--objective-calibration-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--optimizer-backoff-report", type=Path, required=True)
    parser.add_argument("--bounded-training-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--admission-only", action="store_true")
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
