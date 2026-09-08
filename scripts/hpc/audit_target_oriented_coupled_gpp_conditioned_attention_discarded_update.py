"""Run one zero-retention artifact-aware step for conditioned GPP attention."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.model import (
    bind_daily_operator_dynamic_retained_inputs,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_direction import (
    SCHEDULED_NON_GPP_SOURCES,
    artifact_aware_scale_passed,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistence import (
    load_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    load_coupled_gpp_artifact_aware_persistent_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention import (
    conditioned_coupled_gpp_trainable_tree,
    coupled_gpp_conditioned_attention_temperatures,
    initialize_coupled_gpp_conditioned_attention_parameters,
    load_coupled_gpp_conditioned_attention_contract,
    replace_conditioned_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    CONDITIONED_ADMISSION_REPORT_SHA256,
    LOGICAL_UPDATE,
    PARENT_RECOVERY_SHA256,
    PARENT_REPORT_SHA256,
    PARENT_TRAINABLE_PARAMETER_COUNT,
    RETAINED_PARAMETER_COUNT,
    TOTAL_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    absolute_science_gate,
    calibrate_full_step_l2,
    classify_discarded_update,
    load_conditioned_attention_discarded_update_contract,
    monitor_transfer_gate,
    science_progress_gate,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_optimizer_preflight import (
    four_source_macro_objectives,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_target_oriented_acceptance_statistics,
    derive_target_oriented_acceptance_scales,
    finalize_target_oriented_acceptance_statistics,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_direction import (
    _combined_panel_mean,
    _measure_direct_artifact_gradients,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_step_transfer import (
    _evaluate_direct_groups,
    _panel_change,
    _relative_change,
)
from scripts.hpc.calibrate_target_oriented_coupled_gpp_objective import (
    _make_direct_steps,
    _make_state_steps,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.probe_successor_multibatch_direction import (
    _make_flat_direction,
    _make_flat_gram,
    _make_flattened_unit_gradient,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _initialization_only,
)
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _parser as _persistent_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import (
    _all_hard_zero,
    _fixed_parameter_report,
    _mean_losses,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
    _make_dtype_preserving_direct_virtual_step,
    _measure_source_gradients,
)
from scripts.hpc.run_target_oriented_coupled_gpp_optimizer_preflight import (
    _source_losses,
    _state_forward,
)
from scripts.hpc.run_target_oriented_coupled_gpp_persistent_scale_training import (
    _acceptance_key_scores,
    _evaluate_detailed_monitor,
    _horizon_monitor,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import (
    _evaluate_scales,
    _scaled_parameters,
)
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps
from scripts.hpc.run_target_oriented_structured_gpp_gpu_admission import _tree_size
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import _tree_scalar_report
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _direct_forward,
    _hard_host,
    _one_day_hard,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_conditioned_attention_discarded_update_result_v1"
PARENT_SCHEMA_VERSION = "target_oriented_coupled_gpp_artifact_aware_persistent_training_result_v1"
CONDITIONED_ADMISSION_SCHEMA_VERSION = "target_oriented_coupled_gpp_conditioned_attention_admission_result_v1"


def _load_parent_state(
    report_path: Path,
    recovery_path: Path,
    *,
    persistent_contract_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if sha256_file(report_path) != PARENT_REPORT_SHA256 or sha256_file(recovery_path) != PARENT_RECOVERY_SHA256:
        raise ValueError("conditioned discarded-update parent file identity drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    training = report.get("training", {})
    persistence = report.get("persistence", {})
    if (
        report.get("schema_version") != PARENT_SCHEMA_VERSION
        or report.get("status") != "failed_scientific_gate"
        or report.get("classification") != "artifact_aware_persistent_stopped_train_only"
        or report.get("identity", {}).get("artifact_aware_persistent_contract_sha256")
        != persistent_contract_sha256
        or int(training.get("accepted_update_count", -1)) != 1
        or int(training.get("completed_logical_update", -1)) != 2
        or training.get("stopped_reason") != "no_common_direction_or_scale"
        or tuple(float(value) for value in training.get("selected_scales", ())) != (0.5,)
        or persistence.get("recovery_sha256") != PARENT_RECOVERY_SHA256
        or persistence.get("checkpoint") is not None
        or int(report.get("model", {}).get("trainable_parameter_count", -1)) != PARENT_TRAINABLE_PARAMETER_COUNT
    ):
        raise ValueError("conditioned discarded-update parent report drift")
    recovered = load_recovery(
        recovery_path,
        identity=report["identity"],
        parameter_count=PARENT_TRAINABLE_PARAMETER_COUNT,
    )
    records = () if recovered is None else tuple(recovered.get("update_records", ()))
    if (
        recovered is None
        or int(recovered.get("completed_logical_update", -1)) != 2
        or len(records) != 1
        or int(records[0].get("logical_update", -1)) != 2
        or records[0].get("selection", {}).get("accepted") is not True
        or float(records[0].get("selection", {}).get("selected_scale", -1.0)) != 0.5
    ):
        raise ValueError("conditioned discarded-update recovery progress drift")
    return report, recovered


def _load_conditioned_admission(path: Path, *, contract_sha256: str) -> Mapping[str, Any]:
    if sha256_file(path) != CONDITIONED_ADMISSION_REPORT_SHA256:
        raise ValueError("conditioned discarded-update admission identity drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != CONDITIONED_ADMISSION_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("classification") != "conditioned_attention_removes_known_qk_scale_jump"
        or report.get("identity", {}).get("conditioning_contract_sha256") != contract_sha256
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
        or report.get("policy", {}).get("training_updates") != 0
        or report.get("policy", {}).get("checkpoint_written") is not False
    ):
        raise ValueError("conditioned discarded-update admission report drift")
    return report


def _conditioned_bootstrap(
    args: Any,
    *,
    recovered: Mapping[str, Any],
    conditioning_contract: Any,
    warm_snow_free_boundary: bool = False,
) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_coupled_gpp_conditioned_attention_head,
    )

    bootstrap = _initialization_only(args)
    context = bootstrap["context"]
    runtime = context["runtime"]
    jax = runtime["jax"]
    parent_trainable = jax.device_put(recovered["trainable"])
    parent_parameters = replace_coupled_gpp_trainable(bootstrap["initialized_parameters"], parent_trainable)
    head = build_target_oriented_coupled_gpp_conditioned_attention_head(
        bootstrap["head"],
        conditioning_contract=conditioning_contract,
    )
    if warm_snow_free_boundary:
        # Set the physical boundary before binding any forward/gradient helper.
        head = replace(head, warm_snow_free_boundary=True)
    parameters = initialize_coupled_gpp_conditioned_attention_parameters(
        parent_parameters,
        conditioning_contract=conditioning_contract,
        structured_config=head.structured_config,
        config=head.conditioning_config,
    )
    trainable = conditioned_coupled_gpp_trainable_tree(parameters)
    fixed = replace_conditioned_coupled_gpp_trainable(parameters, {})
    if (
        _tree_size(jax, parameters) != TOTAL_PARAMETER_COUNT
        or _tree_size(jax, trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, fixed) != RETAINED_PARAMETER_COUNT
    ):
        raise ValueError("conditioned discarded-update parameter boundary drift")

    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        context["target_contract"],
        acceptance_scales=acceptance_scales,
        acceptance_flux_names=("gpp",),
        science_flux_basis=bootstrap["flux_basis"],
    )
    gradient_steps = steps["make_trainable_gradient_steps"](
        fixed,
        replace_conditioned_coupled_gpp_trainable,
    )
    direct_steps = _make_direct_steps(
        runtime,
        head,
        parameters,
        replace_trainable=replace_conditioned_coupled_gpp_trainable,
    )
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail_template=context["template"],
    )
    state_steps = _make_state_steps(
        runtime,
        transition,
        parameters,
        context["scales"].state.other,
        bootstrap["layout"],
        replace_trainable=replace_conditioned_coupled_gpp_trainable,
    )
    return dict(
        bootstrap,
        head=head,
        initialized_parameters=parameters,
        trainable=trainable,
        fixed_parameters=fixed,
        steps=steps,
        gradient_steps=gradient_steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
    )


def _evaluate_loaded_detailed_artifact(
    context: Mapping[str, Any],
    *,
    update: int,
    artifact: Any,
    parameters: Any,
    trainable: Any,
    bootstrap: Mapping[str, Any],
    label: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    started = time.perf_counter()
    direct, direct_seconds = _direct_forward(jax, trainable, artifact.batch, bootstrap["direct_steps"])
    state, state_seconds = _state_forward(
        jax,
        trainable,
        artifact.batch,
        bootstrap["state_steps"],
        bootstrap["layout"],
    )
    one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, bootstrap["steps"])
    horizon_records = {}
    hard = {"one_day": _hard_host(jax.device_get(one_hard))}
    science_values = {}
    acceptance_statistics = {}
    for horizon in (7, 30):
        science, statistics, acceptance, current_hard, seconds = _horizon_monitor(
            jax,
            jnp,
            parameters,
            artifact.batch,
            bootstrap["steps"],
            horizon=horizon,
            flux_basis=bootstrap["flux_basis"],
            target_contract=context["target_contract"],
        )
        key = str(horizon)
        acceptance_statistics[key] = statistics
        science_values[key] = science
        hard[f"{horizon}_day"] = _hard_host(jax.device_get(current_hard))
        horizon_records[key] = {
            "absolute_science": acceptance,
            "absolute_key_scores": _acceptance_key_scores(acceptance),
            "timing_seconds": seconds,
        }
    losses = _source_losses(direct, state, science_values["7"], science_values["30"])
    if not all(np.isfinite(value) and value >= 0.0 for value in losses.values()):
        raise RuntimeError(f"conditioned discarded-update {label} artifact {update} has invalid loss")
    return (
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
        },
        acceptance_statistics,
    )


def _add_acceptance_statistics(
    totals: dict[str, Any],
    statistics: Mapping[str, Any],
) -> None:
    for horizon in ("7", "30"):
        totals[horizon] = (
            statistics[horizon]
            if horizon not in totals
            else add_target_oriented_acceptance_statistics(totals[horizon], statistics[horizon])
        )


def _finalize_loaded_detailed_panel(
    context: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    acceptance_totals: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    if not records:
        raise ValueError(f"conditioned discarded-update {label} panel is empty")
    hard_records = [{"artifact_update": value["artifact_update"]} | value["hard_counts"] for value in records]
    if not _all_hard_zero(hard_records):
        raise RuntimeError(f"conditioned discarded-update {label} physical hard gate failed")
    means = _mean_losses(tuple(value["source_losses"] for value in records))
    acceptance = {
        horizon: finalize_target_oriented_acceptance_statistics(
            acceptance_totals[horizon],
            horizon=int(horizon),
            target_contract=context["target_contract"],
        )
        for horizon in ("7", "30")
    }
    return {
        "label": label,
        "artifact_updates": [int(value["artifact_update"]) for value in records],
        "source_mean_losses": means,
        "weighted_objective": four_source_macro_objectives(means, bootstrap["coefficients"]),
        "acceptance": acceptance,
        "absolute_key_scores": {horizon: _acceptance_key_scores(value) for horizon, value in acceptance.items()},
        "records": list(records),
        "hard_gate_passed": True,
    }


def _paired_monitor(
    args: Any,
    context: Mapping[str, Any],
    *,
    before_trainable: Any,
    after_trainable: Any,
    bootstrap: Mapping[str, Any],
    updates: Sequence[int],
) -> Mapping[str, Any]:
    before_parameters = replace_conditioned_coupled_gpp_trainable(
        bootstrap["initialized_parameters"],
        before_trainable,
    )
    after_parameters = replace_conditioned_coupled_gpp_trainable(
        bootstrap["initialized_parameters"],
        after_trainable,
    )
    before_records = []
    after_records = []
    before_acceptance: dict[str, Any] = {}
    after_acceptance: dict[str, Any] = {}
    load_count = 0
    for ordinal, update_value in enumerate(updates, start=1):
        update = int(update_value)
        artifact = _load_artifact(args, context, update)
        load_count += 1
        before_record, before_statistics = _evaluate_loaded_detailed_artifact(
            context,
            update=update,
            artifact=artifact,
            parameters=before_parameters,
            trainable=before_trainable,
            bootstrap=bootstrap,
            label="fixed_monitor_initial",
        )
        after_record, after_statistics = _evaluate_loaded_detailed_artifact(
            context,
            update=update,
            artifact=artifact,
            parameters=after_parameters,
            trainable=after_trainable,
            bootstrap=bootstrap,
            label="fixed_monitor_candidate",
        )
        before_records.append(before_record)
        after_records.append(after_record)
        _add_acceptance_statistics(before_acceptance, before_statistics)
        _add_acceptance_statistics(after_acceptance, after_statistics)
        print(
            f"conditioned discarded_update fixed_monitor_pair={ordinal}/{len(updates)} artifact={update}",
            flush=True,
        )
        del artifact, before_statistics, after_statistics
        gc.collect()
    before = _finalize_loaded_detailed_panel(
        context,
        records=before_records,
        acceptance_totals=before_acceptance,
        bootstrap=bootstrap,
        label="fixed_monitor_initial",
    )
    after = _finalize_loaded_detailed_panel(
        context,
        records=after_records,
        acceptance_totals=after_acceptance,
        bootstrap=bootstrap,
        label="fixed_monitor_candidate",
    )
    return {
        "artifact_load_count": load_count,
        "each_artifact_loaded_once": load_count == len(set(int(value) for value in updates)),
        "initial": before,
        "candidate": after,
    }


def _train_scale_record(
    *,
    scheduled_after: Mapping[str, Any],
    artifact_before: Sequence[Mapping[str, Any]],
    artifact_after: Sequence[Mapping[str, Any]],
    combined_before: float,
    relative_tolerance: float,
    scale: float,
    full_step_l2: float,
) -> Mapping[str, Any]:
    group_changes = tuple(
        _panel_change(first, second) for first, second in zip(artifact_before, artifact_after, strict=True)
    )
    artifact_changes = [change for group in group_changes for change in group["artifact_changes"]]
    combined_after = _combined_panel_mean(artifact_after)
    record = {
        "scale": float(scale),
        "parameter_delta_l2": float(scale) * float(full_step_l2),
        "scheduled_four_source_finite_step_accepted": bool(scheduled_after["finite_step_accepted"]),
        "scheduled_physical_hard_gate_passed": bool(scheduled_after["hard_gate_passed"]),
        "scheduled_evaluation": scheduled_after,
        "artifact_direct_gpp_relative_changes": [float(item["relative_change"]) for item in artifact_changes],
        "maximum_artifact_direct_gpp_relative_change": max(float(item["relative_change"]) for item in artifact_changes),
        "improved_artifact_count": sum(bool(item["improved"]) for item in artifact_changes),
        "combined_artifact_direct_gpp_before": combined_before,
        "combined_artifact_direct_gpp_after": combined_after,
        "combined_artifact_direct_gpp_relative_change": _relative_change(combined_before, combined_after),
        "artifact_groups": list(group_changes),
    }
    record["artifact_and_scheduled_gate_passed"] = artifact_aware_scale_passed(
        record,
        expected_artifact_count=40,
        relative_tolerance=relative_tolerance,
    )
    return record


def run(args: Any) -> Mapping[str, Any]:
    from jax.flatten_util import ravel_pytree

    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    contract = load_conditioned_attention_discarded_update_contract(args.discarded_update_contract)
    conditioning_contract = load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract)
    persistent_contract = load_coupled_gpp_artifact_aware_persistent_training_contract(
        args.artifact_aware_persistent_contract
    )
    if (
        conditioning_contract.sha256 != contract.payload["identity"]["conditioned_attention_contract_sha256"]
        or persistent_contract.sha256 != contract.payload["identity"]["artifact_aware_persistent_contract_sha256"]
    ):
        raise ValueError("conditioned discarded-update contract chain drift")
    _load_conditioned_admission(
        args.conditioned_admission_report.resolve(),
        contract_sha256=conditioning_contract.sha256,
    )
    parent_report, recovered = _load_parent_state(
        args.parent_persistent_training_report.resolve(),
        args.parent_artifact_aware_recovery.resolve(),
        persistent_contract_sha256=persistent_contract.sha256,
    )
    recovery_before = args.parent_artifact_aware_recovery.stat()
    bootstrap = _conditioned_bootstrap(
        args,
        recovered=recovered,
        conditioning_contract=conditioning_contract,
    )
    context = bootstrap["context"]
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    required = set(contract.train_artifacts) | set(contract.monitor_updates)
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"conditioned discarded-update prepared coverage drift: {missing}")

    trainable = bootstrap["trainable"]
    parameters = bootstrap["initialized_parameters"]
    scheduled_group = contract.train_groups[0]
    initial_science = _evaluate_detailed_monitor(
        args,
        context,
        parameters=parameters,
        trainable=trainable,
        steps=bootstrap["steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        updates=scheduled_group,
        flux_basis=bootstrap["flux_basis"],
        coefficients=bootstrap["coefficients"],
        label="conditioned_update3_train_only_initial_science",
    )
    initial_absolute = absolute_science_gate(contract, initial_science)
    print(
        "conditioned discarded_update initial_train_science "
        f"absolute_passed={initial_absolute['passed']}",
        flush=True,
    )

    scheduled = _measure_source_gradients(
        args,
        context,
        parameters=parameters,
        trainable=trainable,
        steps=bootstrap["steps"],
        gradient_steps=bootstrap["gradient_steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        group=scheduled_group,
        flux_basis=bootstrap["flux_basis"],
        macro_update=LOGICAL_UPDATE,
        expected_parameter_count=TRAINABLE_PARAMETER_COUNT,
    )
    flatten_gradient = _make_flattened_unit_gradient(jax)
    artifact_measurements = tuple(
        _measure_direct_artifact_gradients(
            args,
            context,
            trainable=trainable,
            direct_steps=bootstrap["direct_steps"],
            updates=group,
            flatten_gradient=flatten_gradient,
            label=f"conditioned_update3_panel_{index}",
            expected_parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
        for index, group in enumerate(contract.train_groups, start=1)
    )
    unit_gradients = [unit for measurement in artifact_measurements for unit in measurement["unit_gradients"]]
    gradient_reports = {
        record["source"]: record
        for measurement in artifact_measurements
        for record in measurement["gradient_reports"]
    }
    constraint_labels = [
        record["source"]
        for measurement in artifact_measurements
        for record in measurement["gradient_reports"]
    ]
    for source in SCHEDULED_NON_GPP_SOURCES:
        unit, norm, nonfinite = flatten_gradient(scheduled["source_gradients"][source])
        jax.block_until_ready((unit, norm, nonfinite))
        norm_host = float(np.asarray(norm))
        nonfinite_host = int(np.asarray(nonfinite))
        if (
            nonfinite_host != 0
            or not np.isfinite(norm_host)
            or norm_host <= 0.0
            or int(unit.size) != TRAINABLE_PARAMETER_COUNT
        ):
            raise RuntimeError(f"conditioned discarded-update scheduled gradient {source} is invalid")
        unit_gradients.append(unit)
        constraint_labels.append(source)
        gradient_reports[source] = {
            "source": source,
            "l2_norm": norm_host,
            "nonfinite_count": nonfinite_host,
            "element_count": int(unit.size),
        }
    del scheduled["source_gradients"]
    if tuple(constraint_labels) != contract.source_registry:
        raise RuntimeError("conditioned discarded-update gradient registry drift")

    unit_matrix = jnp.stack(tuple(unit_gradients))
    jax.block_until_ready(unit_matrix)
    for measurement in artifact_measurements:
        del measurement["unit_gradients"]
    del unit_gradients
    gc.collect()
    gram = _make_flat_gram(jax)(unit_matrix)
    jax.block_until_ready(gram)
    gram_host = np.asarray(gram, dtype=np.float64)
    diagnostics = gradient_direction_diagnostics(gram_host, labels=contract.source_registry)
    selected_direction = diagnostics["directions"]["minimum_norm_unit"]
    cosines = {
        name: float(selected_direction["normalized_directional_derivatives"][name])
        for name in contract.source_registry
    }
    common_first_order = all(
        value >= float(contract.payload["direction"]["minimum_constraint_cosine"])
        for value in cosines.values()
    )

    constraint_losses = {}
    for measurement in artifact_measurements:
        for source, loss in zip(
            (record["source"] for record in measurement["gradient_reports"]),
            (record["loss"] for record in measurement["panel"]["artifact_losses"]),
            strict=True,
        ):
            constraint_losses[source] = float(loss)
    constraint_losses.update(
        {source: float(scheduled["source_mean_losses"][source]) for source in SCHEDULED_NON_GPP_SOURCES}
    )
    trainable_report = _tree_scalar_report(jax, jnp, trainable)
    calibration = None
    proposed = None
    combined_norm_host = None
    if common_first_order:
        calibration = calibrate_full_step_l2(
            contract,
            losses=constraint_losses,
            gradient_l2_norms={name: float(gradient_reports[name]["l2_norm"]) for name in contract.source_registry},
            direction_cosines=cosines,
            trainable_l2_norm=float(trainable_report["l2_norm"]),
        )
        if calibration["usable"]:
            weights = jnp.asarray(
                [selected_direction["normalized_simplex_weights"][name] for name in contract.source_registry],
                dtype=jnp.float64,
            )
            flat_direction, common_norm, common_nonfinite = _make_flat_direction(jax)(unit_matrix, weights)
            jax.block_until_ready((flat_direction, common_norm, common_nonfinite))
            combined_norm_host = float(np.asarray(common_norm))
            if int(np.asarray(common_nonfinite)) != 0 or combined_norm_host <= 0.0:
                raise RuntimeError("conditioned discarded-update common direction is invalid")
            flat_template, unravel = ravel_pytree(trainable)
            if int(flat_template.size) != TRAINABLE_PARAMETER_COUNT:
                raise RuntimeError("conditioned discarded-update flattened parameter width drift")
            direction_tree = unravel(flat_direction)
            proposed, proposal_nonfinite = _make_dtype_preserving_direct_virtual_step(jax)(
                trainable,
                direction_tree,
                common_norm,
                np.asarray(calibration["full_parameter_delta_l2"], dtype=np.float64),
            )
            jax.block_until_ready((proposed, proposal_nonfinite))
            if int(np.asarray(proposal_nonfinite)) != 0:
                raise RuntimeError("conditioned discarded-update proposal is nonfinite")
            del flat_template, flat_direction, direction_tree, common_norm, common_nonfinite, proposal_nonfinite
    del unit_matrix, gram
    gc.collect()

    artifact_before = tuple(measurement["panel"] for measurement in artifact_measurements)
    combined_before = _combined_panel_mean(artifact_before)
    relative_tolerance = float(
        contract.payload["train_only_selection"]["each_artifact_direct_gpp_relative_regression_maximum"]
    )
    scale_records = []
    selected_record = None
    selected_trainable = None
    if common_first_order and calibration is not None and calibration["usable"]:
        for scale in contract.scales:
            scheduled_after = _evaluate_scales(
                args,
                context,
                initialized_parameters=parameters,
                before_trainable=trainable,
                proposed_trainable=proposed,
                before_losses=scheduled["source_mean_losses"],
                scales=(scale,),
                scale_one_evaluation=None,
                steps=bootstrap["steps"],
                direct_steps=bootstrap["direct_steps"],
                state_steps=bootstrap["state_steps"],
                layout=bootstrap["layout"],
                group=scheduled_group,
                flux_basis=bootstrap["flux_basis"],
                coefficients=bootstrap["coefficients"],
                relative_tolerance=relative_tolerance,
                label="conditioned_discarded_update3",
                direction_norm=float(calibration["full_parameter_delta_l2"]),
                replace_trainable=replace_conditioned_coupled_gpp_trainable,
            )[0]
            candidate = _scaled_parameters(jax, trainable, proposed, scale)
            artifact_after = _evaluate_direct_groups(
                args,
                context,
                trainable=candidate,
                direct_steps=bootstrap["direct_steps"],
                groups=contract.train_groups,
                label=f"conditioned_discarded_update3_scale_{scale:g}",
            )
            record = _train_scale_record(
                scheduled_after=scheduled_after,
                artifact_before=artifact_before,
                artifact_after=artifact_after,
                combined_before=combined_before,
                relative_tolerance=relative_tolerance,
                scale=scale,
                full_step_l2=float(calibration["full_parameter_delta_l2"]),
            )
            if record["artifact_and_scheduled_gate_passed"]:
                candidate_parameters = replace_conditioned_coupled_gpp_trainable(parameters, candidate)
                candidate_science = _evaluate_detailed_monitor(
                    args,
                    context,
                    parameters=candidate_parameters,
                    trainable=candidate,
                    steps=bootstrap["steps"],
                    direct_steps=bootstrap["direct_steps"],
                    state_steps=bootstrap["state_steps"],
                    layout=bootstrap["layout"],
                    updates=scheduled_group,
                    flux_basis=bootstrap["flux_basis"],
                    coefficients=bootstrap["coefficients"],
                    label=f"conditioned_update3_train_only_scale_{scale:g}_science",
                )
                record["train_only_science"] = candidate_science
                record["science_progress"] = science_progress_gate(
                    contract,
                    initial_science,
                    candidate_science,
                    gate_name="train_only_selection",
                )
                record["absolute_science"] = absolute_science_gate(contract, candidate_science)
            else:
                record["train_only_science"] = None
                record["science_progress"] = None
                record["absolute_science"] = None
            record["train_only_scale_passed"] = bool(
                record["artifact_and_scheduled_gate_passed"]
                and record["science_progress"] is not None
                and record["science_progress"]["passed"]
            )
            scale_records.append(record)
            print(
                f"conditioned discarded_update scale={scale:g} train_only_passed={record['train_only_scale_passed']}",
                flush=True,
            )
            if record["train_only_scale_passed"]:
                selected_record = record
                selected_trainable = candidate
                break
            del candidate

    selected_scale = None if selected_record is None else float(selected_record["scale"])
    paired_monitor = None
    monitor_gate = None
    if selected_trainable is not None:
        paired_monitor = _paired_monitor(
            args,
            context,
            before_trainable=trainable,
            after_trainable=selected_trainable,
            bootstrap=bootstrap,
            updates=contract.monitor_updates,
        )
        monitor_gate = monitor_transfer_gate(
            contract,
            paired_monitor["initial"],
            paired_monitor["candidate"],
        )

    absolute_passed = None
    if monitor_gate is not None and monitor_gate["passed"]:
        absolute_passed = bool(
            selected_record["absolute_science"]["passed"]
            and monitor_gate["absolute_science"]["passed"]
        )
    classification = classify_discarded_update(
        common_first_order_direction=common_first_order,
        calibration_usable=bool(calibration is not None and calibration["usable"]),
        selected_scale=selected_scale,
        monitor_passed=None if monitor_gate is None else bool(monitor_gate["passed"]),
        absolute_science_passed=absolute_passed,
    )

    inspected_trainable = trainable if selected_trainable is None else selected_trainable
    inspected_parameters = replace_conditioned_coupled_gpp_trainable(parameters, inspected_trainable)
    fixed_report = _fixed_parameter_report(
        jax,
        parameters,
        inspected_parameters,
        replace_trainable=replace_conditioned_coupled_gpp_trainable,
    )
    recovery_after = args.parent_artifact_aware_recovery.stat()
    recovery_unchanged = bool(
        recovery_before.st_size == recovery_after.st_size
        and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
        and sha256_file(args.parent_artifact_aware_recovery) == PARENT_RECOVERY_SHA256
    )
    initial_temperatures = np.asarray(
        jax.device_get(coupled_gpp_conditioned_attention_temperatures(parameters, config=bootstrap["head"].conditioning_config)),
        dtype=np.float64,
    )
    inspected_temperatures = np.asarray(
        jax.device_get(
            coupled_gpp_conditioned_attention_temperatures(
                inspected_parameters,
                config=bootstrap["head"].conditioning_config,
            )
        ),
        dtype=np.float64,
    )
    engineering_passed = bool(
        recovery_unchanged
        and fixed_report["exactly_unchanged"]
        and len(gradient_reports) == 43
        and all(int(value["element_count"]) == TRAINABLE_PARAMETER_COUNT for value in gradient_reports.values())
        and all(int(value["nonfinite_count"]) == 0 for value in gradient_reports.values())
        and (paired_monitor is None or paired_monitor["each_artifact_loaded_once"])
    )
    if not engineering_passed:
        raise RuntimeError("conditioned discarded-update engineering gate failed")

    next_key = {
        "conditioned_discarded_update_meets_absolute_science_gate": "on_monitor_transfer_with_absolute_science_pass",
        "conditioned_discarded_update_transfers_but_absolute_science_remains_open": "on_monitor_transfer_with_absolute_science_open",
        "conditioned_discarded_update_fails_monitor": "on_monitor_transfer_failure",
        "no_conditioned_common_direction": "on_no_common_direction",
        "conditioned_step_calibration_unusable": "on_no_train_only_finite_step",
        "no_conditioned_train_only_finite_step": "on_no_train_only_finite_step",
    }[classification]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "discarded_update_contract_sha256": contract.sha256,
            "conditioned_attention_contract_sha256": conditioning_contract.sha256,
            "conditioned_admission_report_sha256": sha256_file(args.conditioned_admission_report),
            "parent_persistent_report_sha256": sha256_file(args.parent_persistent_training_report),
            "parent_update2_recovery_sha256": sha256_file(args.parent_artifact_aware_recovery),
            "parent_completed_logical_update": int(parent_report["training"]["completed_logical_update"]),
        },
        "policy": {
            "logical_optimizer_updates": 0,
            "optimizer_created": False,
            "parameter_proposals_persisted": 0,
            "checkpoint_written": False,
            "recovery_written_or_modified": False,
            "trainable_parameters_written": False,
            "fresh_validation_values_evaluated": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "model": {
            "parameter_count": TOTAL_PARAMETER_COUNT,
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "retained_parameter_count": RETAINED_PARAMETER_COUNT,
            "initial_trainable_l2": trainable_report["l2_norm"],
            "initial_temperatures": initial_temperatures.tolist(),
            "inspected_temperatures": inspected_temperatures.tolist(),
            "maximum_temperature_change": float(np.max(np.abs(inspected_temperatures - initial_temperatures))),
        },
        "train_only_initial_science": {
            "panel": initial_science,
            "absolute_science": initial_absolute,
        },
        "direction": {
            "source_registry": list(contract.source_registry),
            "source_gradients": gradient_reports,
            "constraint_losses": constraint_losses,
            "gradient_gram": gram_host.tolist(),
            "diagnostics": diagnostics,
            "selected_minimum_norm_direction": selected_direction,
            "common_first_order_direction": common_first_order,
            "combined_gradient_norm": combined_norm_host,
            "calibration": calibration,
        },
        "train_only_selection": {
            "monitor_excluded": True,
            "artifact_groups_initial": list(artifact_before),
            "combined_40_artifact_direct_gpp_initial": combined_before,
            "scale_evaluations": scale_records,
            "selected_scale": selected_scale,
            "selected_parameter_delta_l2": None
            if selected_record is None
            else float(selected_record["parameter_delta_l2"]),
        },
        "fixed_monitor_pair": paired_monitor,
        "fixed_monitor_gate": monitor_gate,
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "monitor_artifacts_loaded_at_most_once": paired_monitor is None
            or paired_monitor["each_artifact_loaded_once"],
            "proposal_discarded": True,
            "scientific_model_promotion_implied": False,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in context["runtime"]["devices"]],
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][next_key],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _persistent_parser()
    parser.add_argument("--parent-persistent-training-report", type=Path, required=True)
    parser.add_argument("--parent-artifact-aware-recovery", type=Path, required=True)
    parser.add_argument("--conditioning-contract", type=Path, required=True)
    parser.add_argument("--conditioned-admission-report", type=Path, required=True)
    parser.add_argument("--discarded-update-contract", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
