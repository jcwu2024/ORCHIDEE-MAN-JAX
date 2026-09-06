"""Run the bounded, resumable conditioned-Attention training smoke."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_direction import (
    SCHEDULED_NON_GPP_SOURCES,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    load_coupled_gpp_artifact_aware_persistent_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention import (
    coupled_gpp_conditioned_attention_temperatures,
    load_coupled_gpp_conditioned_attention_contract,
    replace_conditioned_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    RETAINED_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    absolute_science_gate,
    load_conditioned_attention_discarded_update_contract,
    monitor_transfer_gate,
    science_progress_gate,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_persistence import (
    load_checkpoint,
    load_recovery,
    load_start_snapshot,
    parameter_trees_identical,
    write_checkpoint,
    write_recovery,
    write_start_snapshot,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_persistent_training import (
    DISCARDED_UPDATE_RESULT_SHA256,
    FINAL_LOGICAL_UPDATE,
    START_COMPLETED_LOGICAL_UPDATE,
    calibrate_persistent_step_l2,
    classify_persistent_smoke,
    load_conditioned_attention_persistent_training_contract,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_direction import (
    _combined_panel_mean,
    _measure_direct_artifact_gradients,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    SCHEMA_VERSION as DISCARDED_RESULT_SCHEMA_VERSION,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    _conditioned_bootstrap,
    _load_conditioned_admission,
    _load_parent_state,
    _train_scale_record,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    _parser as _discarded_parser,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_step_transfer import (
    _evaluate_direct_groups,
)
from scripts.hpc.probe_successor_multibatch_direction import (
    _make_flat_direction,
    _make_flat_gram,
    _make_flattened_unit_gradient,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import (
    _fixed_parameter_report,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
    _make_dtype_preserving_direct_virtual_step,
    _measure_source_gradients,
)
from scripts.hpc.run_target_oriented_coupled_gpp_persistent_scale_training import (
    _evaluate_detailed_monitor,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import (
    _evaluate_scales,
    _scaled_parameters,
)
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import (
    _tree_scalar_report,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_conditioned_attention_persistent_training_result_v1"


def _load_discarded_result(path: Path, *, contract_sha256: str) -> Mapping[str, Any]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    selection = report.get("train_only_selection", {})
    monitor = report.get("fixed_monitor_gate", {})
    if (
        digest != DISCARDED_UPDATE_RESULT_SHA256
        or report.get("schema_version") != DISCARDED_RESULT_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("classification") != "conditioned_discarded_update_transfers_but_absolute_science_remains_open"
        or report.get("identity", {}).get("discarded_update_contract_sha256") != contract_sha256
        or float(selection.get("selected_scale", -1.0)) != 1.0
        or selection.get("scale_evaluations", [{}])[0].get("improved_artifact_count") != 40
        or monitor.get("passed") is not True
        or monitor.get("nonregressing_direct_gpp_artifact_count") != 8
        or monitor.get("absolute_science", {}).get("passed") is not False
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
        or report.get("integrity", {}).get("proposal_discarded") is not True
    ):
        raise ValueError("conditioned persistent predecessor result drift")
    return report


def _monitor(
    args: Any,
    bootstrap: Mapping[str, Any],
    contract: Any,
    *,
    trainable: Any,
    label: str,
) -> Mapping[str, Any]:
    parameters = replace_conditioned_coupled_gpp_trainable(bootstrap["initialized_parameters"], trainable)
    return _evaluate_detailed_monitor(
        args,
        bootstrap["context"],
        parameters=parameters,
        trainable=trainable,
        steps=bootstrap["steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        updates=contract.monitor_updates,
        flux_basis=bootstrap["flux_basis"],
        coefficients=bootstrap["coefficients"],
        label=label,
    )


def _train_only_update(
    args: Any,
    bootstrap: Mapping[str, Any],
    contract: Any,
    *,
    trainable: Any,
    panel: tuple[tuple[int, ...], ...],
    source_registry: tuple[str, ...],
    logical_update: int,
    flatten_gradient: Any,
    gram_step: Any,
    combine_direction: Any,
    direct_virtual_step: Any,
) -> Mapping[str, Any]:
    from jax.flatten_util import ravel_pytree

    started = time.perf_counter()
    context = bootstrap["context"]
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    parameters = replace_conditioned_coupled_gpp_trainable(bootstrap["initialized_parameters"], trainable)
    initial_science = _evaluate_detailed_monitor(
        args,
        context,
        parameters=parameters,
        trainable=trainable,
        steps=bootstrap["steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        updates=panel[0],
        flux_basis=bootstrap["flux_basis"],
        coefficients=bootstrap["coefficients"],
        label=f"conditioned_persistent_update_{logical_update}_initial_science",
    )
    initial_absolute = absolute_science_gate(contract, initial_science)
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
        group=panel[0],
        flux_basis=bootstrap["flux_basis"],
        macro_update=logical_update,
        expected_parameter_count=TRAINABLE_PARAMETER_COUNT,
    )
    del parameters

    artifact_measurements = tuple(
        _measure_direct_artifact_gradients(
            args,
            context,
            trainable=trainable,
            direct_steps=bootstrap["direct_steps"],
            updates=group,
            flatten_gradient=flatten_gradient,
            label=f"conditioned_persistent_update_{logical_update}_panel_{index}",
            expected_parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
        for index, group in enumerate(panel, start=1)
    )
    unit_gradients = [unit for measurement in artifact_measurements for unit in measurement["unit_gradients"]]
    gradient_reports = {
        record["source"]: record for measurement in artifact_measurements for record in measurement["gradient_reports"]
    }
    constraint_labels = [
        record["source"] for measurement in artifact_measurements for record in measurement["gradient_reports"]
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
            raise RuntimeError(f"conditioned persistent scheduled gradient {source} is invalid")
        unit_gradients.append(unit)
        constraint_labels.append(source)
        gradient_reports[source] = {
            "source": source,
            "l2_norm": norm_host,
            "nonfinite_count": nonfinite_host,
            "element_count": int(unit.size),
        }
    del scheduled["source_gradients"]
    if tuple(constraint_labels) != source_registry or len(gradient_reports) != 43:
        raise RuntimeError("conditioned persistent gradient registry drift")

    unit_matrix = jnp.stack(tuple(unit_gradients))
    jax.block_until_ready(unit_matrix)
    for measurement in artifact_measurements:
        del measurement["unit_gradients"]
    del unit_gradients
    gc.collect()
    gram = gram_step(unit_matrix)
    jax.block_until_ready(gram)
    gram_host = np.asarray(gram, dtype=np.float64)
    diagnostics = gradient_direction_diagnostics(gram_host, labels=source_registry)
    selected_direction = diagnostics["directions"]["minimum_norm_unit"]
    cosines = {name: float(selected_direction["normalized_directional_derivatives"][name]) for name in source_registry}
    common_first_order = all(
        value >= float(contract.payload["direction"]["minimum_constraint_cosine"]) for value in cosines.values()
    )

    constraint_losses: dict[str, float] = {}
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
        calibration = calibrate_persistent_step_l2(
            contract,
            source_registry=source_registry,
            losses=constraint_losses,
            gradient_l2_norms={name: float(gradient_reports[name]["l2_norm"]) for name in source_registry},
            direction_cosines=cosines,
            trainable_l2_norm=float(trainable_report["l2_norm"]),
        )
        if calibration["usable"]:
            weights = jnp.asarray(
                [selected_direction["normalized_simplex_weights"][name] for name in source_registry],
                dtype=jnp.float64,
            )
            flat_direction, common_norm, common_nonfinite = combine_direction(unit_matrix, weights)
            jax.block_until_ready((flat_direction, common_norm, common_nonfinite))
            combined_norm_host = float(np.asarray(common_norm))
            if int(np.asarray(common_nonfinite)) != 0 or combined_norm_host <= 0.0:
                raise RuntimeError("conditioned persistent common direction is invalid")
            flat_template, unravel = ravel_pytree(trainable)
            if int(flat_template.size) != TRAINABLE_PARAMETER_COUNT:
                raise RuntimeError("conditioned persistent flattened parameter width drift")
            direction_tree = unravel(flat_direction)
            proposed, proposal_nonfinite = direct_virtual_step(
                trainable,
                direction_tree,
                common_norm,
                np.asarray(calibration["full_parameter_delta_l2"], dtype=np.float64),
            )
            jax.block_until_ready((proposed, proposal_nonfinite))
            if int(np.asarray(proposal_nonfinite)) != 0:
                raise RuntimeError("conditioned persistent proposal is nonfinite")
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
                initialized_parameters=bootstrap["initialized_parameters"],
                before_trainable=trainable,
                proposed_trainable=proposed,
                before_losses=scheduled["source_mean_losses"],
                scales=(scale,),
                scale_one_evaluation=None,
                steps=bootstrap["steps"],
                direct_steps=bootstrap["direct_steps"],
                state_steps=bootstrap["state_steps"],
                layout=bootstrap["layout"],
                group=panel[0],
                flux_basis=bootstrap["flux_basis"],
                coefficients=bootstrap["coefficients"],
                relative_tolerance=relative_tolerance,
                label=f"conditioned_persistent_update_{logical_update}",
                direction_norm=float(calibration["full_parameter_delta_l2"]),
                replace_trainable=replace_conditioned_coupled_gpp_trainable,
            )[0]
            candidate = _scaled_parameters(jax, trainable, proposed, scale)
            artifact_after = _evaluate_direct_groups(
                args,
                context,
                trainable=candidate,
                direct_steps=bootstrap["direct_steps"],
                groups=panel,
                label=f"conditioned_persistent_update_{logical_update}_scale_{scale:g}",
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
                candidate_parameters = replace_conditioned_coupled_gpp_trainable(
                    bootstrap["initialized_parameters"], candidate
                )
                candidate_science = _evaluate_detailed_monitor(
                    args,
                    context,
                    parameters=candidate_parameters,
                    trainable=candidate,
                    steps=bootstrap["steps"],
                    direct_steps=bootstrap["direct_steps"],
                    state_steps=bootstrap["state_steps"],
                    layout=bootstrap["layout"],
                    updates=panel[0],
                    flux_basis=bootstrap["flux_basis"],
                    coefficients=bootstrap["coefficients"],
                    label=f"conditioned_persistent_update_{logical_update}_scale_{scale:g}_science",
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
                f"conditioned persistent update={logical_update} scale={scale:g} "
                f"train_only_passed={record['train_only_scale_passed']}",
                flush=True,
            )
            if record["train_only_scale_passed"]:
                selected_record = record
                selected_trainable = candidate
                break
            del candidate

    selected_scale = None if selected_record is None else float(selected_record["scale"])
    result = {
        "logical_update": int(logical_update),
        "train_only_panel": {
            "artifact_groups": [list(group) for group in panel],
            "artifact_count": sum(len(group) for group in panel),
        },
        "initial_science": {
            "panel": initial_science,
            "absolute_science": initial_absolute,
        },
        "direction": {
            "source_registry": list(source_registry),
            "source_gradients": gradient_reports,
            "constraint_losses": constraint_losses,
            "gradient_gram": gram_host.tolist(),
            "diagnostics": diagnostics,
            "selected_minimum_norm_direction": selected_direction,
            "common_first_order_direction": common_first_order,
            "combined_gradient_norm": combined_norm_host,
            "calibration": calibration,
        },
        "baseline": {
            "scheduled_source_mean_losses": scheduled["source_mean_losses"],
            "scheduled_artifact_records": scheduled["artifact_records"],
            "artifact_groups": list(artifact_before),
            "combined_40_artifact_direct_gpp_mean": combined_before,
        },
        "scale_evaluations": scale_records,
        "selection": {
            "monitor_excluded": True,
            "stopped_after_first_train_only_passing_scale": selected_scale is not None,
            "selected_scale": selected_scale,
            "selected_parameter_delta_l2": None
            if selected_record is None
            else float(selected_record["parameter_delta_l2"]),
            "absolute_science": None if selected_record is None else selected_record["absolute_science"],
            "accepted": selected_scale is not None,
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    return {
        "updated_trainable": trainable if selected_trainable is None else selected_trainable,
        "record": result,
    }


def _write_progress(
    path: Path,
    *,
    identity: Mapping[str, Any],
    completed_logical_update: int,
    records: Sequence[Mapping[str, Any]],
    monitors: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "target_oriented_coupled_gpp_conditioned_attention_progress_v1",
            "identity": identity,
            "completed_logical_update": int(completed_logical_update),
            "selected_scales": [value["selection"]["selected_scale"] for value in records],
            "full_parameter_delta_l2": [
                value["direction"]["calibration"]["full_parameter_delta_l2"] for value in records
            ],
            "monitor_local_gates": [value["local_gate"] for value in monitors],
            "monitor_cumulative_gates": [value["cumulative_gate"] for value in monitors],
        },
    )


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    contract = load_conditioned_attention_persistent_training_contract(args.conditioned_persistent_contract)
    discarded_contract = load_conditioned_attention_discarded_update_contract(args.discarded_update_contract)
    conditioning_contract = load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract)
    artifact_persistent_contract = load_coupled_gpp_artifact_aware_persistent_training_contract(
        args.artifact_aware_persistent_contract
    )
    if (
        discarded_contract.sha256 != contract.payload["identity"]["discarded_update_contract_sha256"]
        or conditioning_contract.sha256 != contract.payload["identity"]["conditioned_attention_contract_sha256"]
        or artifact_persistent_contract.sha256
        != contract.payload["identity"]["artifact_aware_persistent_contract_sha256"]
        or discarded_contract.train_groups != contract.train_panels[0]
    ):
        raise ValueError("conditioned persistent contract chain drift")
    _load_discarded_result(
        args.conditioned_discarded_update_result.resolve(),
        contract_sha256=discarded_contract.sha256,
    )
    _load_conditioned_admission(
        args.conditioned_admission_report.resolve(),
        contract_sha256=conditioning_contract.sha256,
    )
    _, parent_recovery = _load_parent_state(
        args.parent_persistent_training_report.resolve(),
        args.parent_artifact_aware_recovery.resolve(),
        persistent_contract_sha256=artifact_persistent_contract.sha256,
    )
    bootstrap = _conditioned_bootstrap(
        args,
        recovered=parent_recovery,
        conditioning_contract=conditioning_contract,
    )
    context = bootstrap["context"]
    jax = context["runtime"]["jax"]
    required = {value for panel in contract.train_panels for group in panel for value in group}
    required.update(contract.monitor_updates)
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"conditioned persistent prepared coverage drift: {missing}")

    start_path = output_dir / "start.pkl"
    recovery_path = output_dir / "recovery.pkl"
    checkpoint_path = output_dir / f"checkpoint-update-{FINAL_LOGICAL_UPDATE:04d}.pkl"
    rejected_path = output_dir / "rejected-proposal.json"
    if recovery_path.exists() and not start_path.exists():
        raise ValueError("conditioned persistent recovery exists without immutable start")

    base_identity = {
        "git_head": _git_head(),
        "conditioned_persistent_contract_sha256": contract.sha256,
        "discarded_update_result_sha256": sha256_file(args.conditioned_discarded_update_result),
        "conditioned_attention_admission_sha256": sha256_file(args.conditioned_admission_report),
        "parent_update2_recovery_sha256": sha256_file(args.parent_artifact_aware_recovery),
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
    }
    if not start_path.exists():
        host_start = jax.device_get(bootstrap["trainable"])
        written_start = write_start_snapshot(
            start_path,
            identity=base_identity,
            trainable=host_start,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
        loaded_start = load_start_snapshot(
            start_path,
            identity=base_identity,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
        if written_start["sha256"] != sha256_file(start_path) or not parameter_trees_identical(
            host_start, loaded_start["trainable"]
        ):
            raise RuntimeError("conditioned persistent start roundtrip failed")
        initialization_mode = "conditioned_update2_snapshot_created"
    else:
        loaded_start = load_start_snapshot(
            start_path,
            identity=base_identity,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
        initialization_mode = "identity_exact_resume"
    start_sha256 = sha256_file(start_path)
    persistence_identity = dict(base_identity, start_snapshot_sha256=start_sha256)
    trainable = jax.device_put(loaded_start["trainable"])
    records: list[Mapping[str, Any]] = []
    update_monitors: list[Mapping[str, Any]] = []
    initial_monitor = None
    completed = START_COMPLETED_LOGICAL_UPDATE

    recovered = load_recovery(
        recovery_path,
        identity=persistence_identity,
        parameter_count=TRAINABLE_PARAMETER_COUNT,
    )
    if recovered is not None:
        trainable = jax.device_put(recovered["trainable"])
        records = list(recovered["update_records"])
        update_monitors = list(recovered["update_monitors"])
        initial_monitor = recovered["initial_monitor"]
        completed = int(recovered["completed_logical_update"])
        print(f"conditioned persistent resumed_after_update={completed}", flush=True)

    memory_before = _memory_report(bootstrap["device"])
    if initial_monitor is None:
        initial_monitor = _monitor(
            args,
            bootstrap,
            contract,
            trainable=trainable,
            label="conditioned_persistent_initial_monitor",
        )
        write_recovery(
            recovery_path,
            identity=persistence_identity,
            trainable=jax.device_get(trainable),
            completed_logical_update=completed,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
            update_records=records,
            initial_monitor=initial_monitor,
            update_monitors=update_monitors,
        )
        _write_progress(
            output_dir / "progress.json",
            identity=persistence_identity,
            completed_logical_update=completed,
            records=records,
            monitors=update_monitors,
        )

    stopped_reason = None
    flatten_gradient = _make_flattened_unit_gradient(jax)
    gram_step = _make_flat_gram(jax)
    combine_direction = _make_flat_direction(jax)
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)
    for logical_update, panel, source_registry in zip(
        contract.logical_updates,
        contract.train_panels,
        contract.source_registries,
        strict=True,
    ):
        if logical_update <= completed:
            continue
        measured = _train_only_update(
            args,
            bootstrap,
            contract,
            trainable=trainable,
            panel=panel,
            source_registry=source_registry,
            logical_update=logical_update,
            flatten_gradient=flatten_gradient,
            gram_step=gram_step,
            combine_direction=combine_direction,
            direct_virtual_step=direct_virtual_step,
        )
        record = measured["record"]
        if record["selection"]["accepted"] is not True:
            stopped_reason = "train_only"
            _atomic_json(
                rejected_path,
                {
                    "schema_version": "target_oriented_coupled_gpp_conditioned_attention_rejection_v1",
                    "identity": persistence_identity,
                    "reason": stopped_reason,
                    "retained_through_logical_update": completed,
                    "proposal": record,
                },
            )
            break

        candidate_trainable = measured["updated_trainable"]
        candidate_monitor = _monitor(
            args,
            bootstrap,
            contract,
            trainable=candidate_trainable,
            label=f"conditioned_persistent_update_{logical_update}_monitor",
        )
        previous_monitor = initial_monitor if not update_monitors else update_monitors[-1]["panel"]
        local_gate = monitor_transfer_gate(contract, previous_monitor, candidate_monitor)
        cumulative_gate = monitor_transfer_gate(contract, initial_monitor, candidate_monitor)
        monitor_accepted = bool(local_gate["passed"] and cumulative_gate["passed"])
        print(
            f"conditioned persistent update={logical_update} local_monitor={local_gate['passed']} "
            f"cumulative_monitor={cumulative_gate['passed']}",
            flush=True,
        )
        if not monitor_accepted:
            stopped_reason = "monitor_transfer"
            _atomic_json(
                rejected_path,
                {
                    "schema_version": "target_oriented_coupled_gpp_conditioned_attention_rejection_v1",
                    "identity": persistence_identity,
                    "reason": stopped_reason,
                    "retained_through_logical_update": completed,
                    "proposal": record,
                    "candidate_monitor": candidate_monitor,
                    "local_gate": local_gate,
                    "cumulative_gate": cumulative_gate,
                },
            )
            break

        trainable = candidate_trainable
        records.append(record)
        update_monitors.append(
            {
                "logical_update": int(logical_update),
                "panel": candidate_monitor,
                "local_gate": local_gate,
                "cumulative_gate": cumulative_gate,
            }
        )
        completed = int(logical_update)
        write_recovery(
            recovery_path,
            identity=persistence_identity,
            trainable=jax.device_get(trainable),
            completed_logical_update=completed,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
            update_records=records,
            initial_monitor=initial_monitor,
            update_monitors=update_monitors,
        )
        _write_progress(
            output_dir / "progress.json",
            identity=persistence_identity,
            completed_logical_update=completed,
            records=records,
            monitors=update_monitors,
        )
        print(
            f"conditioned persistent retained_update={completed}/{FINAL_LOGICAL_UPDATE} "
            f"scale={record['selection']['selected_scale']} seconds={record['timing_seconds']['total']:.3f}",
            flush=True,
        )

    final_absolute_passed = None
    if completed == FINAL_LOGICAL_UPDATE:
        final_absolute_passed = bool(
            records[-1]["selection"]["absolute_science"]["passed"]
            and update_monitors[-1]["cumulative_gate"]["absolute_science"]["passed"]
        )
    classification = classify_persistent_smoke(
        completed_logical_update=completed,
        stopped_reason=stopped_reason,
        final_absolute_science_passed=final_absolute_passed,
    )

    checkpoint = None
    if completed == FINAL_LOGICAL_UPDATE:
        host_trainable = jax.device_get(trainable)
        if checkpoint_path.exists():
            digest = sha256_file(checkpoint_path)
            loaded_checkpoint = load_checkpoint(
                checkpoint_path,
                expected_sha256=digest,
                identity=persistence_identity,
                parameter_count=TRAINABLE_PARAMETER_COUNT,
            )
            if not parameter_trees_identical(host_trainable, loaded_checkpoint["trainable"]):
                raise RuntimeError("conditioned persistent checkpoint parameter drift")
            checkpoint = {
                "path": str(checkpoint_path),
                "sha256": digest,
                "completed_logical_update": FINAL_LOGICAL_UPDATE,
                "parameter_count": TRAINABLE_PARAMETER_COUNT,
                "promotion_eligible": False,
            }
        else:
            checkpoint = write_checkpoint(
                checkpoint_path,
                identity=persistence_identity,
                trainable=host_trainable,
                completed_logical_update=completed,
                parameter_count=TRAINABLE_PARAMETER_COUNT,
                update_records=records,
                initial_monitor=initial_monitor,
                update_monitors=update_monitors,
            )

    final_parameters = replace_conditioned_coupled_gpp_trainable(bootstrap["initialized_parameters"], trainable)
    fixed_report = _fixed_parameter_report(
        jax,
        bootstrap["initialized_parameters"],
        final_parameters,
        replace_trainable=replace_conditioned_coupled_gpp_trainable,
    )
    if not fixed_report["exactly_unchanged"] or int(fixed_report["element_count"]) != RETAINED_PARAMETER_COUNT:
        raise RuntimeError("conditioned persistent fixed-parameter boundary failed")
    initial_temperatures = np.asarray(
        jax.device_get(
            coupled_gpp_conditioned_attention_temperatures(
                bootstrap["initialized_parameters"],
                config=bootstrap["head"].conditioning_config,
            )
        ),
        dtype=np.float64,
    )
    final_temperatures = np.asarray(
        jax.device_get(
            coupled_gpp_conditioned_attention_temperatures(
                final_parameters,
                config=bootstrap["head"].conditioning_config,
            )
        ),
        dtype=np.float64,
    )
    engineering_passed = bool(
        fixed_report["exactly_unchanged"]
        and sha256_file(args.parent_artifact_aware_recovery)
        == contract.payload["identity"]["artifact_aware_update2_recovery_sha256"]
        and len(records) == len(update_monitors) == completed - START_COMPLETED_LOGICAL_UPDATE
        and all(value["local_gate"]["passed"] and value["cumulative_gate"]["passed"] for value in update_monitors)
    )
    if not engineering_passed:
        raise RuntimeError("conditioned persistent engineering gate failed")

    if completed == FINAL_LOGICAL_UPDATE:
        status = "passed"
        next_key = (
            "on_three_update_transfer_with_absolute_science_pass"
            if final_absolute_passed
            else "on_three_update_transfer_with_absolute_science_open"
        )
    else:
        status = "failed_scientific_gate"
        next_key = "on_train_only_stop" if stopped_reason == "train_only" else "on_monitor_transfer_stop"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "classification": classification,
        "identity": persistence_identity,
        "policy": {
            "initialization_mode": initialization_mode,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_created": False,
            "optimizer_state_used": False,
            "fixed_parameters_persisted": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
            "monitor_selected_direction_or_scale": False,
            "failed_monitor_candidate_retained": False,
        },
        "starting_state": {
            "completed_logical_update": START_COMPLETED_LOGICAL_UPDATE,
            "path": str(start_path),
            "sha256": start_sha256,
            "roundtrip_passed": True,
        },
        "training": {
            "requested_logical_updates": list(contract.logical_updates),
            "completed_logical_update": completed,
            "accepted_update_count": len(records),
            "stopped_reason": stopped_reason,
            "selected_scales": [value["selection"]["selected_scale"] for value in records],
            "updates": records,
        },
        "initial_monitor": initial_monitor,
        "update_monitors": update_monitors,
        "final_absolute_science_passed": final_absolute_passed,
        "persistence": {
            "recovery_path": str(recovery_path),
            "recovery_sha256": sha256_file(recovery_path),
            "checkpoint": checkpoint,
            "rejected_proposal_path": str(rejected_path) if rejected_path.exists() else None,
        },
        "model": {
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "retained_parameter_count": RETAINED_PARAMETER_COUNT,
            "initial_temperatures": initial_temperatures.tolist(),
            "final_temperatures": final_temperatures.tolist(),
            "maximum_temperature_change": float(np.max(np.abs(final_temperatures - initial_temperatures))),
            "fixed_parameters": fixed_report,
        },
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "parent_update2_recovery_unchanged": True,
            "all_retained_updates_passed_local_and_cumulative_monitor": all(
                value["local_gate"]["passed"] and value["cumulative_gate"]["passed"] for value in update_monitors
            ),
            "scientific_model_promotion_implied": False,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in context["runtime"]["devices"]],
        "memory_before": memory_before,
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][next_key],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _discarded_parser()
    parser.add_argument("--conditioned-discarded-update-result", type=Path, required=True)
    parser.add_argument("--conditioned-persistent-contract", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
