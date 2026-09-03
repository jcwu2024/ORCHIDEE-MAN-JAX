"""Run the frozen three-update artifact-aware direct-GPP continuity smoke."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_continuity import (
    ARTIFACT_AWARE_RESULT_SHA256,
    classify_artifact_aware_continuity,
    load_coupled_gpp_artifact_aware_continuity_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_direction import (
    SCHEDULED_NON_GPP_SOURCES,
    artifact_aware_scale_passed,
    load_coupled_gpp_artifact_aware_direction_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_generalization import (
    load_coupled_gpp_direct_generalization_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    TRAINABLE_PARAMETER_COUNT,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    build_persistent_scale_training_schedule,
    load_coupled_gpp_persistent_scale_training_contract,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_direction import (
    _combined_panel_mean,
    _measure_direct_artifact_gradients,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_direction import (
    _parser as _artifact_aware_parser,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_generalization import (
    _bootstrap_update7,
    _direct_panel,
    _load_persistent_result,
    _replay_direction,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_step_transfer import (
    _evaluate_direct_groups,
    _panel_change,
    _relative_change,
)
from scripts.hpc.probe_successor_direct_trust_region import _make_combined_direction
from scripts.hpc.probe_successor_multibatch_direction import (
    _make_flat_direction,
    _make_flat_gram,
    _make_flattened_unit_gradient,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import _fixed_parameter_report
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
    _make_dtype_preserving_direct_virtual_step,
    _measure_source_gradients,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import (
    _evaluate_scales,
    _scaled_parameters,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_artifact_aware_continuity_smoke_result_v1"


def _load_artifact_aware_result(path: Path, *, contract_sha256: str) -> Mapping[str, Any]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    selection = report.get("selection", {})
    if (
        digest != ARTIFACT_AWARE_RESULT_SHA256
        or report.get("schema_version") != "target_oriented_coupled_gpp_artifact_aware_direction_audit_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "artifact_aware_direction_transfers_to_monitor"
        or report.get("identity", {}).get("artifact_aware_contract_sha256") != contract_sha256
        or selection.get("selected_scale") != 0.5
        or float(selection.get("selected_monitor_relative_change", 0.0)) >= 0.0
        or selection.get("selected_monitor", {}).get("improved_artifact_count") != 8
        or report.get("scale_evaluations", [{}])[0].get("improved_artifact_count") != 40
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
        or report.get("policy", {}).get("checkpoint_written") is not False
        or report.get("policy", {}).get("trainable_parameters_written") is not False
    ):
        raise ValueError("artifact-aware continuity predecessor result drift")
    return report


def _train_only_update(
    args: Any,
    context: Mapping[str, Any],
    *,
    contract: Any,
    initialized_parameters: Any,
    trainable: Any,
    bootstrap: Mapping[str, Any],
    panel: tuple[tuple[int, ...], ...],
    source_registry: tuple[str, ...],
    logical_update: int,
    flatten_gradient: Any,
    direct_virtual_step: Any,
) -> Mapping[str, Any]:
    from jax.flatten_util import ravel_pytree

    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    started = time.perf_counter()
    parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
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
            label=f"continuity_update_{logical_update}_panel_{index}",
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
            raise RuntimeError(f"artifact-aware continuity scheduled gradient {source} is invalid")
        unit_gradients.append(unit)
        constraint_labels.append(source)
        gradient_reports[source] = {
            "source": source,
            "l2_norm": norm_host,
            "nonfinite_count": nonfinite_host,
            "element_count": int(unit.size),
        }
        del norm, nonfinite
    del scheduled["source_gradients"]
    if tuple(constraint_labels) != source_registry or len(gradient_reports) != len(source_registry):
        raise RuntimeError("artifact-aware continuity gradient registry drift")

    unit_matrix = jnp.stack(tuple(unit_gradients))
    jax.block_until_ready(unit_matrix)
    for measurement in artifact_measurements:
        del measurement["unit_gradients"]
    del unit_gradients
    gc.collect()
    gram = _make_flat_gram(jax)(unit_matrix)
    jax.block_until_ready(gram)
    gram_host = np.asarray(gram, dtype=np.float64)
    diagnostics = gradient_direction_diagnostics(gram_host, labels=source_registry)
    selected_direction = diagnostics["directions"]["minimum_norm_unit"]
    minimum_cosine = float(contract.payload["direction"]["minimum_constraint_cosine"])
    common_first_order = all(
        float(selected_direction["normalized_directional_derivatives"][name]) >= minimum_cosine
        for name in source_registry
    )

    proposed = None
    common_norm_host = None
    if common_first_order:
        weights = jnp.asarray(
            [selected_direction["normalized_simplex_weights"][name] for name in source_registry],
            dtype=jnp.float64,
        )
        flat_direction, common_norm, common_nonfinite = _make_flat_direction(jax)(unit_matrix, weights)
        jax.block_until_ready((flat_direction, common_norm, common_nonfinite))
        common_norm_host = float(np.asarray(common_norm))
        if int(np.asarray(common_nonfinite)) != 0 or common_norm_host <= 0.0:
            raise RuntimeError("artifact-aware continuity common direction is invalid")
        flat_template, unravel = ravel_pytree(trainable)
        direction_tree = unravel(flat_direction)
        del flat_template
        proposed, proposal_nonfinite = direct_virtual_step(
            trainable,
            direction_tree,
            common_norm,
            np.asarray(float(contract.payload["direction"]["full_parameter_step_l2"]), dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposal_nonfinite))
        if int(np.asarray(proposal_nonfinite)) != 0:
            raise RuntimeError("artifact-aware continuity direction realization is nonfinite")
        del direction_tree, flat_direction, common_norm, common_nonfinite, proposal_nonfinite
    del unit_matrix, gram
    gc.collect()

    artifact_before = tuple(measurement["panel"] for measurement in artifact_measurements)
    combined_before = _combined_panel_mean(artifact_before)
    relative_tolerance = float(
        contract.payload["scale_selection"]["each_train_only_artifact_direct_gpp_relative_change_maximum"]
    )
    scale_records = []
    selected_record = None
    if common_first_order:
        for scale in contract.scales:
            scheduled_after = _evaluate_scales(
                args,
                context,
                initialized_parameters=initialized_parameters,
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
                label=f"artifact_aware_continuity_update_{logical_update}",
                direction_norm=float(contract.payload["direction"]["full_parameter_step_l2"]),
            )[0]
            candidate = _scaled_parameters(jax, trainable, proposed, scale)
            artifact_after = _evaluate_direct_groups(
                args,
                context,
                trainable=candidate,
                direct_steps=bootstrap["direct_steps"],
                groups=panel,
                label=f"artifact_aware_continuity_update_{logical_update}_scale_{scale:g}",
            )
            group_changes = tuple(
                _panel_change(first, second) for first, second in zip(artifact_before, artifact_after, strict=True)
            )
            artifact_changes = [change for group in group_changes for change in group["artifact_changes"]]
            combined_after = _combined_panel_mean(artifact_after)
            record = {
                "scale": float(scale),
                "parameter_delta_l2": float(scale) * float(contract.payload["direction"]["full_parameter_step_l2"]),
                "scheduled_four_source_finite_step_accepted": bool(scheduled_after["finite_step_accepted"]),
                "scheduled_physical_hard_gate_passed": bool(scheduled_after["hard_gate_passed"]),
                "scheduled_evaluation": scheduled_after,
                "artifact_direct_gpp_relative_changes": [float(item["relative_change"]) for item in artifact_changes],
                "maximum_artifact_direct_gpp_relative_change": max(
                    float(item["relative_change"]) for item in artifact_changes
                ),
                "improved_artifact_count": sum(bool(item["improved"]) for item in artifact_changes),
                "combined_artifact_direct_gpp_before": combined_before,
                "combined_artifact_direct_gpp_after": combined_after,
                "combined_artifact_direct_gpp_relative_change": _relative_change(combined_before, combined_after),
                "artifact_groups": list(group_changes),
            }
            record["train_only_scale_passed"] = artifact_aware_scale_passed(
                record,
                expected_artifact_count=40,
                relative_tolerance=relative_tolerance,
            )
            scale_records.append(record)
            print(
                f"coupled_gpp artifact_aware_continuity update={logical_update} scale={scale:g} "
                f"train_only_passed={record['train_only_scale_passed']}",
                flush=True,
            )
            del candidate
            if record["train_only_scale_passed"]:
                selected_record = record
                break

    selected_scale = None if selected_record is None else float(selected_record["scale"])
    updated_trainable = (
        trainable if selected_scale is None else _scaled_parameters(jax, trainable, proposed, selected_scale)
    )
    result = {
        "logical_update": int(logical_update),
        "train_only_panel": {
            "artifact_groups": [list(group) for group in panel],
            "artifact_count": sum(len(group) for group in panel),
        },
        "direction": {
            "source_registry": list(source_registry),
            "source_gradients": gradient_reports,
            "gradient_gram": gram_host.tolist(),
            "diagnostics": diagnostics,
            "selected_minimum_norm_direction": selected_direction,
            "common_first_order_direction": common_first_order,
            "combined_gradient_norm": common_norm_host,
            "full_parameter_step_l2": float(contract.payload["direction"]["full_parameter_step_l2"]),
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
            "accepted": selected_scale is not None,
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    return {"updated_trainable": updated_trainable, "record": result}


def _monitor_trajectory(
    args: Any,
    context: Mapping[str, Any],
    *,
    trainables: Sequence[Any],
    direct_steps: Mapping[str, Any],
    monitor_updates: Sequence[int],
) -> Mapping[str, Any]:
    panels = tuple(
        _direct_panel(
            args,
            context,
            trainable=trainable,
            direct_steps=direct_steps,
            updates=monitor_updates,
            label=f"artifact_aware_continuity_monitor_state_{index}",
        )
        for index, trainable in enumerate(trainables)
    )
    baseline = panels[0]
    relative_to_initial = tuple(_panel_change(baseline, panel) for panel in panels[1:])
    consecutive = tuple(_panel_change(first, second) for first, second in zip(panels, panels[1:], strict=True))
    return {
        "state_panels": list(panels),
        "relative_to_initial": list(relative_to_initial),
        "consecutive_changes": list(consecutive),
        "relative_changes": [float(record["mean_relative_change"]) for record in relative_to_initial],
    }


def run(args: Any) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_coupled_gpp_artifact_aware_continuity_contract(args.continuity_contract)
    artifact_contract = load_coupled_gpp_artifact_aware_direction_contract(args.artifact_aware_contract)
    if (
        artifact_contract.sha256 != contract.payload["identity"]["artifact_aware_direction_contract_sha256"]
        or artifact_contract.train_groups != contract.train_panels[0]
        or artifact_contract.monitor_updates != contract.monitor_updates
    ):
        raise ValueError("artifact-aware continuity predecessor contract drift")
    predecessor_digest = sha256_file(args.artifact_aware_result)
    predecessor = _load_artifact_aware_result(
        args.artifact_aware_result,
        contract_sha256=artifact_contract.sha256,
    )
    persistent_contract = load_coupled_gpp_persistent_scale_training_contract(args.persistent_contract)
    schedule = build_persistent_scale_training_schedule()
    if (
        persistent_contract.sha256 != contract.payload["identity"]["persistent_contract_sha256"]
        or schedule.sha256 != contract.payload["identity"]["persistent_schedule_sha256"]
    ):
        raise ValueError("artifact-aware continuity persistent identity drift")
    generalization_contract = load_coupled_gpp_direct_generalization_contract(args.generalization_contract)
    persistent_result = _load_persistent_result(
        args.persistent_result,
        audit_contract=generalization_contract,
        persistent_contract=persistent_contract,
        schedule=schedule,
    )

    recovery_before = args.parent_recovery.stat()
    bootstrap = _bootstrap_update7(args, persistent_contract)
    context = bootstrap["context"]
    required = {value for panel in contract.train_panels for group in panel for value in group}
    required.update(contract.monitor_updates)
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"artifact-aware continuity prepared coverage drift: {missing}")

    jax = context["runtime"]["jax"]
    initialized_parameters = bootstrap["initialized_parameters"]
    common = dict(
        initialized_parameters=initialized_parameters,
        steps=bootstrap["steps"],
        gradient_steps=bootstrap["gradient_steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        flux_basis=bootstrap["flux_basis"],
        persistent_contract=persistent_contract,
        gram_measurement=_make_gradient_gram(jax),
        combine_direction=_make_combined_direction(jax),
        direct_virtual_step=_make_dtype_preserving_direct_virtual_step(jax),
    )
    replay_records = persistent_result["training"]["updates"]
    update_one = _replay_direction(
        args,
        context,
        trainable=bootstrap["trainable"],
        group=schedule.groups[0],
        persistent_update=1,
        expected_record=replay_records[0],
        selected_scale=float(replay_records[0]["selected_scale"]),
        **common,
    )
    trainable = update_one["updated_trainable"]
    trainable_trajectory = [trainable]
    flatten_gradient = _make_flattened_unit_gradient(jax)
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)
    update_records = []
    for logical_update, panel, source_registry in zip(
        contract.logical_updates,
        contract.train_panels,
        contract.source_registries,
        strict=True,
    ):
        measured = _train_only_update(
            args,
            context,
            contract=contract,
            initialized_parameters=initialized_parameters,
            trainable=trainable,
            bootstrap=bootstrap,
            panel=panel,
            source_registry=source_registry,
            logical_update=logical_update,
            flatten_gradient=flatten_gradient,
            direct_virtual_step=direct_virtual_step,
        )
        update_records.append(measured["record"])
        if not measured["record"]["selection"]["accepted"]:
            break
        trainable = measured["updated_trainable"]
        trainable_trajectory.append(trainable)

    accepted_count = len(trainable_trajectory) - 1
    monitor = None
    monitor_relative_changes = None
    if accepted_count == len(contract.logical_updates):
        monitor = _monitor_trajectory(
            args,
            context,
            trainables=trainable_trajectory,
            direct_steps=bootstrap["direct_steps"],
            monitor_updates=contract.monitor_updates,
        )
        monitor_relative_changes = monitor["relative_changes"]
    classification = classify_artifact_aware_continuity(
        accepted_update_count=accepted_count,
        required_update_count=len(contract.logical_updates),
        monitor_relative_changes=monitor_relative_changes,
    )

    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    inspected_trainable = trainable_trajectory[-1]
    fixed_report = _fixed_parameter_report(
        jax,
        initialized_parameters,
        replace_coupled_gpp_trainable(initialized_parameters, inspected_trainable),
    )
    predecessor_unchanged = sha256_file(args.artifact_aware_result) == predecessor_digest
    gradients_finite = all(
        int(gradient["nonfinite_count"]) == 0
        for record in update_records
        for gradient in record["direction"]["source_gradients"].values()
    )
    engineering_passed = bool(
        recovery_unchanged
        and fixed_report["exactly_unchanged"]
        and predecessor_unchanged
        and predecessor_digest == ARTIFACT_AWARE_RESULT_SHA256
        and gradients_finite
    )
    if not engineering_passed:
        raise RuntimeError("artifact-aware continuity engineering gate failed")

    next_key = {
        "artifact_aware_three_update_continuity_passes_monitor": "on_three_update_monitor_continuity",
        "artifact_aware_continuity_stops_train_only": "on_train_only_continuity_stop",
        "artifact_aware_three_update_continuity_fails_monitor": "on_monitor_continuity_failure",
    }[classification]
    first_selection = update_records[0]["selection"] if update_records else {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "continuity_contract_sha256": contract.sha256,
            "artifact_aware_direction_contract_sha256": artifact_contract.sha256,
            "artifact_aware_direction_result_sha256": predecessor_digest,
            "persistent_contract_sha256": persistent_contract.sha256,
            "persistent_schedule_sha256": schedule.sha256,
        },
        "policy": {
            "in_memory_direct_parameter_updates_applied": accepted_count,
            "optimizer_updates_persisted": 0,
            "parameter_proposals_persisted": 0,
            "checkpoint_written": False,
            "recovery_written": False,
            "trainable_parameters_written": False,
            "fresh_validation_values_evaluated": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "reconstruction": {
            "bootstrap": {
                "selected_scales": [1.0, 1.0, 0.25, 1.0, 0.5],
                "elapsed_seconds": bootstrap["elapsed_seconds"],
            },
            "persistent_update_1": update_one["record"],
            "predecessor_selected_scale": float(predecessor["selection"]["selected_scale"]),
            "repeated_first_panel": list(contract.panel_group_updates[0]) == [2, 11, 12, 13, 14],
            "first_update_train_only_mechanism_reproduced": bool(first_selection.get("accepted", False)),
            "cross_gpu_bit_identity_required": False,
        },
        "train_only": {
            "required_updates": len(contract.logical_updates),
            "accepted_updates": accepted_count,
            "all_three_updates_selected": accepted_count == len(contract.logical_updates),
            "monitor_excluded_from_every_direction_and_scale_selection": True,
            "updates": update_records,
        },
        "monitor": monitor,
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "artifact_aware_direction_result_unchanged": predecessor_unchanged,
            "all_measured_gradients_finite": gradients_finite,
            "scientific_model_promotion_implied": False,
        },
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][next_key],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _artifact_aware_parser()
    parser.add_argument("--continuity-contract", type=Path, required=True)
    parser.add_argument("--artifact-aware-result", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
