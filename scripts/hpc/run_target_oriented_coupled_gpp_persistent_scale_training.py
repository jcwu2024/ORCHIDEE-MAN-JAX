"""Run persistent scale-selected coupled-GPP training or its free admission."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    GRADIENT_SOURCES,
    RETAINED_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_optimizer_preflight import (
    four_source_macro_objectives,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_checkpoint import (
    load_persistent_checkpoint,
    load_persistent_recovery,
    write_persistent_checkpoint,
    write_persistent_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    ALL_REPORTED_KEYS,
    MAXIMUM_UPDATES,
    MILESTONES,
    TRANSFER_RESULT_SHA256,
    build_persistent_scale_training_schedule,
    load_coupled_gpp_persistent_scale_training_contract,
    monitor_gate_decision,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_scale_selected_transfer import (
    load_coupled_gpp_scale_selected_transfer_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_science_window_sufficient_statistics,
    add_target_oriented_acceptance_statistics,
    finalize_target_oriented_acceptance_statistics,
    science_window_from_sufficient_statistics,
)
from scripts.hpc.probe_successor_direct_trust_region import _make_combined_direction
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import (
    _all_hard_zero,
    _fixed_parameter_report,
    _mean_losses,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _make_dtype_preserving_direct_virtual_step,
    _measure_source_gradients,
)
from scripts.hpc.run_target_oriented_coupled_gpp_optimizer_preflight import (
    _source_losses,
    _state_forward,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selected_transfer import (
    _parser as _transfer_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selecting_recalibration import (
    _git_head,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selecting_recalibration import (
    run as _run_scale_selecting,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import (
    _direction_record,
    _evaluate_scales,
    _scaled_parameters,
)
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _merge_statistics,
    _slice,
    _zero_hard,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _direct_forward,
    _hard_host,
    _one_day_hard,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_persistent_scale_training_result_v1"
ADMISSION_SCHEMA_VERSION = "target_oriented_coupled_gpp_persistent_scale_training_admission_v1"


def _load_transfer_result(path: Path, persistent_contract: Any) -> Mapping[str, Any]:
    if sha256_file(path) != TRANSFER_RESULT_SHA256:
        raise ValueError("persistent coupled GPP transfer result identity drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    transfer = report.get("transfer", {})
    if (
        report.get("schema_version") != "target_oriented_coupled_gpp_scale_selected_transfer_audit_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "non_test_seven_thirty_day_transfer_accepted"
        or report.get("identity", {}).get("transfer_contract_sha256")
        != persistent_contract.payload["identity"]["transfer_contract_sha256"]
        or transfer.get("accepted") is not True
        or transfer.get("physical_hard_gate_passed") is not True
        or report.get("policy", {}).get("checkpoint_written") is not False
    ):
        raise ValueError("persistent coupled GPP predecessor transfer is not accepted")
    return report


def _error_record(value: Any) -> Mapping[str, float]:
    mse = float(np.asarray(value.mse))
    bias = float(np.asarray(value.mean_bias))
    count = float(np.asarray(value.count))
    return {
        "normalized_rmse": float(np.sqrt(max(mse, 0.0))),
        "normalized_signed_bias": bias,
        "normalized_loss": 0.5 * (mse + bias * bias),
        "evaluated_values": count,
    }


def _science_record(value: Any) -> Mapping[str, Any]:
    return {
        "total": float(np.asarray(value.total)),
        "flux_mse": float(np.asarray(value.flux_mse)),
        "flux_bias_squared": float(np.asarray(value.flux_bias_squared)),
        "stock_mse": float(np.asarray(value.stock_mse)),
        "stock_bias_squared": float(np.asarray(value.stock_bias_squared)),
        "flux_keys": {name: _error_record(error) for name, error in value.flux_errors.items()},
        "stock_keys": {name: _error_record(error) for name, error in value.stock_errors.items()},
    }


def _acceptance_key_scores(report: Mapping[str, Any]) -> Mapping[str, float]:
    keys = report["keys"]
    if set(keys) != set(ALL_REPORTED_KEYS):
        raise ValueError("persistent coupled GPP detailed science registry drift")
    return {
        name: 0.5 * (float(value["normalized_rmse"]) ** 2 + float(value["normalized_signed_bias"]) ** 2)
        for name, value in keys.items()
    }


def _horizon_monitor(
    jax: Any,
    jnp: Any,
    parameters: Any,
    batch: Any,
    steps: Mapping[str, Any],
    *,
    horizon: int,
    flux_basis: Sequence[str],
    target_contract: Any,
) -> tuple[Any, Any, Mapping[str, Any], Any, float]:
    source = batch.seven_day if horizon == 7 else batch.thirty_day
    count = 8 if horizon == 7 else 4
    name = "seven" if horizon == 7 else "thirty"
    started = time.perf_counter()
    values = [steps[f"{name}_monitor_forward"](parameters, _slice(source, index, index + 1)) for index in range(count)]
    jax.block_until_ready(values)
    science_statistics = _merge_statistics(
        add_science_window_sufficient_statistics,
        [value[0] for value in values],
    )
    acceptance_statistics = _merge_statistics(
        add_target_oriented_acceptance_statistics,
        [value[1] for value in values],
    )
    hard = _zero_hard(jnp)
    for value in values:
        hard = _add_hard(jax, hard, value[2])
    science = science_window_from_sufficient_statistics(
        science_statistics,
        flux_basis=flux_basis,
    )
    acceptance = finalize_target_oriented_acceptance_statistics(
        acceptance_statistics,
        horizon=horizon,
        target_contract=target_contract,
    )
    jax.block_until_ready((science, hard))
    return science, acceptance_statistics, acceptance, hard, time.perf_counter() - started


def _evaluate_detailed_monitor(
    args: Any,
    context: Mapping[str, Any],
    *,
    parameters: Any,
    trainable: Any,
    steps: Mapping[str, Any],
    direct_steps: Mapping[str, Any],
    state_steps: Mapping[str, Any],
    layout: Any,
    updates: Sequence[int],
    flux_basis: Sequence[str],
    coefficients: Mapping[str, float],
    label: str,
) -> Mapping[str, Any]:
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    target_contract = context["target_contract"]
    records = []
    acceptance_totals: dict[str, Any] = {}
    for ordinal, update in enumerate(updates, start=1):
        started = time.perf_counter()
        from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact

        artifact = _load_artifact(args, context, int(update))
        direct, direct_seconds = _direct_forward(jax, trainable, artifact.batch, direct_steps)
        state, state_seconds = _state_forward(jax, trainable, artifact.batch, state_steps, layout)
        one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
        horizon_records = {}
        hard = {"one_day": _hard_host(jax.device_get(one_hard))}
        science_values = {}
        for horizon in (7, 30):
            science, acceptance_statistics, acceptance, current_hard, seconds = _horizon_monitor(
                jax,
                jnp,
                parameters,
                artifact.batch,
                steps,
                horizon=horizon,
                flux_basis=flux_basis,
                target_contract=target_contract,
            )
            key = str(horizon)
            if key in acceptance_totals:
                acceptance_totals[key] = add_target_oriented_acceptance_statistics(
                    acceptance_totals[key],
                    acceptance_statistics,
                )
            else:
                acceptance_totals[key] = acceptance_statistics
            science_values[key] = science
            hard[f"{horizon}_day"] = _hard_host(jax.device_get(current_hard))
            horizon_records[key] = {
                "component_resolved_training_basis": _science_record(science),
                "absolute_science": acceptance,
                "absolute_key_scores": _acceptance_key_scores(acceptance),
                "timing_seconds": seconds,
            }
        losses = _source_losses(direct, state, science_values["7"], science_values["30"])
        if not all(np.isfinite(value) and value >= 0.0 for value in losses.values()):
            raise RuntimeError(f"persistent coupled GPP {label} artifact {update} has invalid loss")
        records.append(
            {
                "artifact_update": int(update),
                "runtime_year": int(context["schedule"]["entries"][int(update) - 1]["runtime_year"]),
                "source_losses": losses,
                "horizons": horizon_records,
                "hard_counts": hard,
                "timing_seconds": {
                    "direct": direct_seconds,
                    "state": state_seconds,
                    "total": time.perf_counter() - started,
                },
            }
        )
        print(
            f"persistent coupled_gpp {label}={ordinal}/{len(updates)} artifact={update} "
            f"seconds={records[-1]['timing_seconds']['total']:.3f}",
            flush=True,
        )
    hard_records = [{"artifact_update": value["artifact_update"]} | value["hard_counts"] for value in records]
    if not _all_hard_zero(hard_records):
        raise RuntimeError(f"persistent coupled GPP {label} physical hard gate failed")
    means = _mean_losses(tuple(value["source_losses"] for value in records))
    acceptance = {
        horizon: finalize_target_oriented_acceptance_statistics(
            acceptance_totals[horizon],
            horizon=int(horizon),
            target_contract=target_contract,
        )
        for horizon in ("7", "30")
    }
    key_scores = {horizon: _acceptance_key_scores(value) for horizon, value in acceptance.items()}
    return {
        "label": label,
        "artifact_updates": [int(value) for value in updates],
        "source_mean_losses": means,
        "weighted_objective": four_source_macro_objectives(means, coefficients),
        "acceptance": acceptance,
        "absolute_key_scores": key_scores,
        "cumulative_gpp_scores": {horizon: scores["gpp"] for horizon, scores in key_scores.items()},
        "records": records,
        "hard_gate_passed": True,
    }


def _persistent_update(
    args: Any,
    context: Mapping[str, Any],
    *,
    initialized_parameters: Any,
    trainable: Any,
    steps: Mapping[str, Any],
    gradient_steps: Mapping[str, Any],
    direct_steps: Mapping[str, Any],
    state_steps: Mapping[str, Any],
    layout: Any,
    group: Sequence[int],
    flux_basis: Sequence[str],
    coefficients: Mapping[str, float],
    contract: Any,
    persistent_update: int,
) -> Mapping[str, Any]:
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    started = time.perf_counter()
    parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    measurement = _measure_source_gradients(
        args,
        context,
        parameters=parameters,
        trainable=trainable,
        steps=steps,
        gradient_steps=gradient_steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
        layout=layout,
        group=group,
        flux_basis=flux_basis,
        macro_update=persistent_update,
    )
    source_gradients = measurement["source_gradients"]
    ordered_gradients = tuple(source_gradients[name] for name in GRADIENT_SOURCES)
    gram_measurement = _make_gradient_gram(jax)
    gram, source_nonfinite = gram_measurement(ordered_gradients)
    jax.block_until_ready((gram, source_nonfinite))
    gram_host = np.asarray(gram, dtype=np.float64)
    nonfinite_host = np.asarray(source_nonfinite, dtype=np.int64)
    diagnostics = gradient_direction_diagnostics(gram_host, labels=GRADIENT_SOURCES)
    selected_direction = diagnostics["directions"]["minimum_norm_unit"]
    minimum_cosine = float(contract.payload["direction"]["minimum_source_cosine"])
    common_first_order = bool(
        np.all(nonfinite_host == 0)
        and all(
            float(selected_direction["normalized_directional_derivatives"][name]) >= minimum_cosine
            for name in GRADIENT_SOURCES
        )
    )
    base_record = {
        "persistent_update": persistent_update,
        "artifact_updates": [int(value) for value in group],
        "source_mean_losses_before": measurement["source_mean_losses"],
        "source_gradients": measurement["source_reports"],
        "gradient_gram": gram_host.tolist(),
        "source_nonfinite_counts": {name: int(nonfinite_host[index]) for index, name in enumerate(GRADIENT_SOURCES)},
        "direction_diagnostics": diagnostics,
        "selected_minimum_norm_direction": selected_direction,
        "common_first_order_direction": common_first_order,
        "artifact_records": measurement["artifact_records"],
    }
    if not common_first_order:
        return {
            "accepted": False,
            "classification": "no_common_first_order_direction",
            "record": base_record
            | {
                "selected_scale": None,
                "scale_evaluations": [],
                "finite_step_accepted": False,
                "timing_seconds": {"total": time.perf_counter() - started},
            },
        }

    weights = np.asarray(
        [selected_direction["weights"][name] for name in GRADIENT_SOURCES],
        dtype=np.float64,
    )
    combine_direction = _make_combined_direction(jax)
    common_gradient, common_norm, common_nonfinite = combine_direction(ordered_gradients, weights)
    jax.block_until_ready((common_gradient, common_norm, common_nonfinite))
    if (
        int(np.asarray(common_nonfinite)) != 0
        or not np.isfinite(float(np.asarray(common_norm)))
        or float(np.asarray(common_norm)) <= 0.0
    ):
        raise RuntimeError(f"persistent coupled GPP update {persistent_update} common gradient is invalid")
    parameter_step_l2 = float(contract.payload["direction"]["full_parameter_step_l2"])
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)
    proposed, proposal_nonfinite = direct_virtual_step(
        trainable,
        common_gradient,
        common_norm,
        np.asarray(parameter_step_l2, dtype=np.float64),
    )
    jax.block_until_ready((proposed, proposal_nonfinite))
    if int(np.asarray(proposal_nonfinite)) != 0:
        raise RuntimeError(f"persistent coupled GPP update {persistent_update} proposal is nonfinite")
    direction = _direction_record(jax, jnp, source_gradients, trainable, proposed)
    if direction["common_source_descent_direction"] is not True:
        raise RuntimeError(f"persistent coupled GPP update {persistent_update} realization drift")

    scale_records = []
    selected_scale = None
    relative_tolerance = float(
        contract.payload["scale_selection"]["all_four_training_group_sources_must_not_increase_relative_tolerance"]
    )
    for scale in contract.scales:
        current = _evaluate_scales(
            args,
            context,
            initialized_parameters=initialized_parameters,
            before_trainable=trainable,
            proposed_trainable=proposed,
            before_losses=measurement["source_mean_losses"],
            scales=(scale,),
            scale_one_evaluation=None,
            steps=steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            group=group,
            flux_basis=flux_basis,
            coefficients=coefficients,
            relative_tolerance=relative_tolerance,
            label=f"persistent_update_{persistent_update}",
            direction_norm=float(direction["parameter_descent_direction"]["l2_norm"]),
        )[0]
        scale_records.append(current)
        if current["finite_step_accepted"] is True:
            selected_scale = float(scale)
            break
    accepted = selected_scale is not None
    record = base_record | {
        "combined_gradient_norm": float(np.asarray(common_norm)),
        "direction": direction,
        "full_parameter_delta_l2": parameter_step_l2,
        "scale_evaluations": scale_records,
        "selected_scale": selected_scale,
        "finite_step_accepted": accepted,
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    if not accepted:
        return {
            "accepted": False,
            "classification": "no_accepted_finite_scale",
            "record": record,
        }
    return {
        "accepted": True,
        "classification": "scale_selected_update_accepted",
        "updated_trainable": _scaled_parameters(jax, trainable, proposed, selected_scale),
        "record": record,
    }


def run(args: Any) -> Mapping[str, Any]:
    report_path = args.output.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()
    contract = load_coupled_gpp_persistent_scale_training_contract(args.persistent_contract)
    transfer_contract = load_coupled_gpp_scale_selected_transfer_contract(args.transfer_contract)
    if transfer_contract.sha256 != contract.payload["identity"]["transfer_contract_sha256"]:
        raise ValueError("persistent coupled GPP transfer contract chain drift")
    _load_transfer_result(args.transfer_result, contract)
    reconstructed, internal = _run_scale_selecting(
        args,
        write_output=False,
        return_internal=True,
    )
    if reconstructed.get("status") != "passed" or reconstructed.get("smoke", {}).get("updates_accepted") != [5, 6, 7]:
        raise RuntimeError("persistent coupled GPP update-7 reconstruction failed")

    context = internal["context"]
    runtime = context["runtime"]
    jax = runtime["jax"]
    schedule = build_persistent_scale_training_schedule()
    required = {value for group in schedule.groups for value in group} | set(contract.monitor_updates)
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"persistent coupled GPP prepared coverage drift: {missing}")
    initialized_parameters = internal["initialized_parameters"]
    bootstrap_trainable = internal["final_trainable"]
    fixed_parameters = replace_coupled_gpp_trainable(initialized_parameters, {})
    steps = internal["steps"]
    if not {"seven_monitor_forward", "thirty_monitor_forward"} <= set(steps):
        raise RuntimeError("persistent coupled GPP detailed monitor steps are unavailable")
    gradient_steps = steps["make_trainable_gradient_steps"](
        fixed_parameters,
        replace_coupled_gpp_trainable,
    )
    direct_steps = internal["direct_steps"]
    state_steps = internal["state_steps"]
    layout = internal["layout"]
    flux_basis = internal["flux_basis"]
    coefficients = {str(name): float(value) for name, value in contract.payload["objective"]["coefficients"].items()}
    identity = {
        "git_head": _git_head(),
        "persistent_contract_sha256": contract.sha256,
        "transfer_result_sha256": sha256_file(args.transfer_result),
        "scale_selecting_contract_sha256": reconstructed["identity"]["scale_selecting_contract_sha256"],
        "scale_selecting_result_sha256": sha256_file(args.scale_selecting_result),
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "schedule_sha256": schedule.sha256,
    }
    memory_before = _memory_report(internal["device"])
    recovery_path = output_dir / "recovery.pkl"
    trainable = bootstrap_trainable
    update_records: list[Mapping[str, Any]] = []
    milestone_monitors: list[Mapping[str, Any]] = []
    checkpoints: list[Mapping[str, Any]] = []
    initial_monitor = None

    if not args.admission_only:
        recovered = load_persistent_recovery(
            recovery_path,
            identity=identity,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
            maximum_updates=MAXIMUM_UPDATES,
        )
        if recovered is not None:
            trainable = jax.device_put(recovered["trainable"])
            update_records = list(recovered["update_records"])
            milestone_monitors = list(recovered["milestone_monitors"])
            initial_monitor = recovered["initial_monitor"]
            for monitor in milestone_monitors:
                if bool(monitor["gate"]["checkpoint_passed"]):
                    update = int(monitor["persistent_update"])
                    path = output_dir / f"stage-update-{update:04d}.pkl"
                    digest = sha256_file(path)
                    load_persistent_checkpoint(
                        path,
                        expected_sha256=digest,
                        identity=identity,
                        completed_updates=update,
                        parameter_count=TRAINABLE_PARAMETER_COUNT,
                    )
                    checkpoints.append(
                        {
                            "persistent_update": update,
                            "path": str(path),
                            "sha256": digest,
                            "completed_updates": update,
                            "promotion_eligible": False,
                        }
                    )
            print(f"persistent coupled_gpp resumed={len(update_records)}", flush=True)

    if initial_monitor is None:
        initial_parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
        initial_monitor = _evaluate_detailed_monitor(
            args,
            context,
            parameters=initial_parameters,
            trainable=trainable,
            steps=steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            updates=contract.monitor_updates,
            flux_basis=flux_basis,
            coefficients=coefficients,
            label="initial_after_update_7_monitor",
        )

    maximum_updates = 1 if args.admission_only else MAXIMUM_UPDATES
    stopped_reason = None
    if milestone_monitors and bool(milestone_monitors[-1]["gate"]["stop_stage"]):
        stopped_reason = "milestone_scientific_gate"
    for index in range(len(update_records), maximum_updates):
        if stopped_reason is not None:
            break
        persistent_update = index + 1
        proposal = _persistent_update(
            args,
            context,
            initialized_parameters=initialized_parameters,
            trainable=trainable,
            steps=steps,
            gradient_steps=gradient_steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            group=schedule.groups[index],
            flux_basis=flux_basis,
            coefficients=coefficients,
            contract=contract,
            persistent_update=persistent_update,
        )
        if not proposal["accepted"]:
            stopped_reason = str(proposal["classification"])
            _atomic_json(
                output_dir / "rejected-proposal.json",
                {
                    "schema_version": "target_oriented_coupled_gpp_persistent_rejected_proposal_v1",
                    "identity": identity,
                    "proposal": proposal["record"],
                    "updates_retained_before_rejection": len(update_records),
                },
            )
            print(
                f"persistent coupled_gpp update={persistent_update} rejected reason={stopped_reason}",
                flush=True,
            )
            break
        trainable = proposal["updated_trainable"]
        update_records.append(proposal["record"])
        print(
            f"persistent coupled_gpp update={persistent_update}/{maximum_updates} "
            f"scale={proposal['record']['selected_scale']} "
            f"seconds={proposal['record']['timing_seconds']['total']:.3f}",
            flush=True,
        )
        if not args.admission_only and persistent_update in MILESTONES:
            parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
            panel = _evaluate_detailed_monitor(
                args,
                context,
                parameters=parameters,
                trainable=trainable,
                steps=steps,
                direct_steps=direct_steps,
                state_steps=state_steps,
                layout=layout,
                updates=contract.monitor_updates,
                flux_basis=flux_basis,
                coefficients=coefficients,
                label=f"milestone_{persistent_update}_monitor",
            )
            gate = monitor_gate_decision(
                contract,
                persistent_update=persistent_update,
                current_sources=panel["source_mean_losses"],
                initial_gpp_scores=initial_monitor["cumulative_gpp_scores"],
                current_gpp_scores=panel["cumulative_gpp_scores"],
                physical_gate_passed=bool(panel["hard_gate_passed"]),
                checkpoint_already_selected=any(
                    bool(value["gate"]["checkpoint_passed"]) for value in milestone_monitors
                ),
            )
            panel = dict(panel, persistent_update=persistent_update, gate=gate)
            milestone_monitors.append(panel)
            if gate["checkpoint_passed"]:
                host_trainable = jax.device_get(trainable)
                path = output_dir / f"stage-update-{persistent_update:04d}.pkl"
                if path.exists():
                    digest = sha256_file(path)
                    existing = load_persistent_checkpoint(
                        path,
                        expected_sha256=digest,
                        identity=identity,
                        completed_updates=persistent_update,
                        parameter_count=TRAINABLE_PARAMETER_COUNT,
                    )
                    same_parameters = all(
                        np.array_equal(np.asarray(left), np.asarray(right))
                        for left, right in zip(
                            jax.tree_util.tree_leaves(existing["trainable"]),
                            jax.tree_util.tree_leaves(host_trainable),
                            strict=True,
                        )
                    )
                    if not same_parameters:
                        raise RuntimeError("persistent coupled GPP orphan checkpoint parameter drift")
                    checkpoint = {
                        "path": str(path),
                        "sha256": digest,
                        "completed_updates": persistent_update,
                        "parameter_count": TRAINABLE_PARAMETER_COUNT,
                        "promotion_eligible": False,
                    }
                else:
                    checkpoint = write_persistent_checkpoint(
                        path,
                        identity=identity,
                        trainable=host_trainable,
                        completed_updates=persistent_update,
                        parameter_count=TRAINABLE_PARAMETER_COUNT,
                        update_records=update_records,
                        initial_monitor=initial_monitor,
                        milestone_monitors=milestone_monitors,
                    )
                checkpoints.append({"persistent_update": persistent_update} | checkpoint)
            if gate["stop_stage"]:
                stopped_reason = "milestone_scientific_gate"
            print(
                f"persistent coupled_gpp milestone={persistent_update} "
                f"continue={gate['continuation_passed']} checkpoint={gate['checkpoint_passed']}",
                flush=True,
            )
        if not args.admission_only:
            write_persistent_recovery(
                recovery_path,
                identity=identity,
                trainable=jax.device_get(trainable),
                completed_updates=persistent_update,
                parameter_count=TRAINABLE_PARAMETER_COUNT,
                update_records=update_records,
                initial_monitor=initial_monitor,
                milestone_monitors=milestone_monitors,
            )
            _atomic_json(
                output_dir / "progress.json",
                {
                    "schema_version": "target_oriented_coupled_gpp_persistent_progress_v1",
                    "identity": identity,
                    "completed_updates": persistent_update,
                    "stopped_reason": stopped_reason,
                    "milestone_monitors": milestone_monitors,
                    "selected_scales": [value["selected_scale"] for value in update_records],
                },
            )

    if args.admission_only:
        accepted = len(update_records) == 1
        roundtrip = False
        temporary_sha = None
        if accepted:
            temporary = output_dir / "admission-recovery.pkl"
            written = write_persistent_recovery(
                temporary,
                identity=identity,
                trainable=jax.device_get(trainable),
                completed_updates=1,
                parameter_count=TRAINABLE_PARAMETER_COUNT,
                update_records=update_records,
                initial_monitor=initial_monitor,
                milestone_monitors=(),
            )
            recovered = load_persistent_recovery(
                temporary,
                identity=identity,
                parameter_count=TRAINABLE_PARAMETER_COUNT,
                maximum_updates=MAXIMUM_UPDATES,
            )
            roundtrip = recovered is not None and int(recovered["completed_updates"]) == 1
            temporary_sha = written["sha256"]
            temporary.unlink()
        report = {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "status": "passed" if accepted and roundtrip else "failed_scientific_gate",
            "classification": ("persistent_runner_admission_passed" if accepted and roundtrip else stopped_reason),
            "identity": identity,
            "policy": {
                "persistent_updates_proposed": 1,
                "persistent_updates_persisted": 0,
                "checkpoint_written": False,
                "recovery_retained": False,
                "teacher_data_regenerated": False,
                "historical_v5_used": False,
                "fresh_validation_values_evaluated": False,
                "sealed_test_read": False,
                "forcing_records_reconstructed": 0,
                "subdaily_state_scan_steps": 0,
            },
            "initial_monitor": initial_monitor,
            "discarded_update": None if not accepted else update_records[0],
            "temporary_recovery_roundtrip_passed": roundtrip,
            "temporary_recovery_sha256": temporary_sha,
            "memory_before": memory_before,
            "memory_after": _memory_report(internal["device"]),
            "elapsed_seconds": time.perf_counter() - started,
            "next_gate": (
                contract.payload["next_gate"]["on_admission_pass"]
                if accepted and roundtrip
                else contract.payload["next_gate"]["on_no_common_direction_or_scale"]
            ),
        }
        _atomic_json(report_path, report)
        return report

    final_parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    fixed_report = _fixed_parameter_report(jax, initialized_parameters, final_parameters)
    if not fixed_report["exactly_unchanged"] or int(fixed_report["element_count"]) != RETAINED_PARAMETER_COUNT:
        raise RuntimeError("persistent coupled GPP fixed-parameter boundary failed")
    selected_checkpoint = checkpoints[-1] if checkpoints else None
    status = "passed" if selected_checkpoint is not None else "stopped_scientific_gate"
    classification = (
        "persistent_checkpoint_selected"
        if selected_checkpoint is not None
        else (stopped_reason or "maximum_updates_without_checkpoint")
    )
    timings = np.asarray(
        [record["timing_seconds"]["total"] for record in update_records],
        dtype=np.float64,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "classification": classification,
        "identity": identity,
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "persistent_updates_executed": len(update_records),
            "artifact_presentations": 8 * len(update_records),
            "optimizer_state_used": False,
            "checkpoint_promotion_eligible": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "fixed_parameters": fixed_report,
        },
        "training": {
            "maximum_updates": MAXIMUM_UPDATES,
            "completed_updates": len(update_records),
            "stopped_reason": stopped_reason,
            "selected_scales": [record["selected_scale"] for record in update_records],
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
        "selected_checkpoint": selected_checkpoint,
        "memory_before": memory_before,
        "memory_after": _memory_report(internal["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][
            "on_checkpoint_selected" if selected_checkpoint is not None else "on_no_checkpoint_by_update_25"
        ],
    }
    _atomic_json(report_path, report)
    return report


def _parser() -> Any:
    parser = _transfer_parser()
    parser.add_argument("--persistent-contract", type=Path, required=True)
    parser.add_argument("--transfer-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--admission-only", action="store_true")
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
