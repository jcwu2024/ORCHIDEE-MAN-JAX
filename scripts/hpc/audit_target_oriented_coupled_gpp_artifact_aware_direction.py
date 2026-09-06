"""Audit a 43-constraint artifact-aware direct-GPP direction without training."""

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
    MULTI_GROUP_RESULT_SHA256,
    SCHEDULED_NON_GPP_SOURCES,
    artifact_aware_scale_passed,
    classify_artifact_aware_direction,
    load_coupled_gpp_artifact_aware_direction_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_generalization import (
    load_coupled_gpp_direct_generalization_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_step_transfer import (
    load_coupled_gpp_direct_step_transfer_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_multi_group_direction import (
    load_coupled_gpp_multi_group_direction_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    TRAINABLE_PARAMETER_COUNT,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    build_persistent_scale_training_schedule,
    load_coupled_gpp_persistent_scale_training_contract,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_generalization import (
    _bootstrap_update7,
    _direct_loss,
    _direct_panel,
    _load_persistent_result,
    _replay_direction,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_step_transfer import (
    _evaluate_direct_groups,
    _panel_change,
    _relative_change,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_multi_group_direction import (
    _load_step_transfer_result,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_multi_group_direction import (
    _parser as _multi_group_parser,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
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
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import _direct_gradient

SCHEMA_VERSION = "target_oriented_coupled_gpp_artifact_aware_direction_audit_result_v1"


def _load_multi_group_result(path: Path, contract: Any) -> Mapping[str, Any]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        digest != MULTI_GROUP_RESULT_SHA256
        or digest != contract.payload["identity"]["multi_group_result_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_multi_group_direction_audit_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "multi_group_direction_fails_monitor"
        or report.get("selection", {}).get("selected_scale") != 0.5
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
        or report.get("policy", {}).get("checkpoint_written") is not False
    ):
        raise ValueError("artifact-aware direction predecessor result drift")
    return report


def _measure_direct_artifact_gradients(
    args: Any,
    context: Mapping[str, Any],
    *,
    trainable: Any,
    direct_steps: Mapping[str, Any],
    updates: Sequence[int],
    flatten_gradient: Any,
    label: str,
    expected_parameter_count: int | None = None,
) -> Mapping[str, Any]:
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    parameter_count = TRAINABLE_PARAMETER_COUNT if expected_parameter_count is None else int(expected_parameter_count)
    started = time.perf_counter()
    unit_gradients = []
    gradient_reports = []
    loss_records = []
    for ordinal, update in enumerate(updates, start=1):
        artifact_started = time.perf_counter()
        artifact = _load_artifact(args, context, int(update))
        direct, gradient, forward_seconds, backward_seconds = _direct_gradient(
            jax,
            jnp,
            trainable,
            artifact.batch,
            direct_steps,
        )
        unit, norm, nonfinite = flatten_gradient(gradient)
        jax.block_until_ready((unit, norm, nonfinite))
        norm_host = float(np.asarray(norm))
        nonfinite_host = int(np.asarray(nonfinite))
        if (
            nonfinite_host != 0
            or not np.isfinite(norm_host)
            or norm_host <= 0.0
            or int(unit.size) != parameter_count
        ):
            raise RuntimeError(f"artifact-aware direct-GPP gradient {update} is invalid")
        source = f"{DIRECT_SOURCE}.artifact_{int(update)}"
        unit_gradients.append(unit)
        gradient_reports.append(
            {
                "source": source,
                "artifact_update": int(update),
                "l2_norm": norm_host,
                "nonfinite_count": nonfinite_host,
                "element_count": int(unit.size),
            }
        )
        loss_records.append(
            {
                "artifact_update": int(update),
                "runtime_year": int(context["schedule"]["entries"][int(update) - 1]["runtime_year"]),
                "loss": _direct_loss(direct),
                "timing_seconds": {
                    "forward": forward_seconds,
                    "gradient": backward_seconds,
                    "total": time.perf_counter() - artifact_started,
                },
            }
        )
        print(
            f"coupled_gpp artifact_aware {label}_gradient={ordinal}/{len(updates)} artifact={update}",
            flush=True,
        )
        del artifact, direct, gradient, norm, nonfinite
    losses = np.asarray([record["loss"] for record in loss_records], dtype=np.float64)
    return {
        "unit_gradients": tuple(unit_gradients),
        "gradient_reports": tuple(gradient_reports),
        "panel": {
            "artifact_updates": [int(value) for value in updates],
            "mean_loss": float(np.mean(losses)),
            "artifact_losses": loss_records,
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }


def _combined_panel_mean(panels: Sequence[Mapping[str, Any]]) -> float:
    losses = [float(record["loss"]) for panel in panels for record in panel["artifact_losses"]]
    return float(np.mean(np.asarray(losses, dtype=np.float64)))


def run(args: Any) -> Mapping[str, Any]:
    from jax.flatten_util import ravel_pytree

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_coupled_gpp_artifact_aware_direction_contract(args.artifact_aware_contract)
    multi_contract = load_coupled_gpp_multi_group_direction_contract(args.multi_group_contract)
    multi_digest = sha256_file(args.multi_group_result)
    _load_multi_group_result(args.multi_group_result, contract)
    step_contract = load_coupled_gpp_direct_step_transfer_contract(args.step_transfer_contract)
    step_digest = sha256_file(args.direct_step_transfer_result)
    _load_step_transfer_result(args.direct_step_transfer_result, multi_contract)
    persistent_contract = load_coupled_gpp_persistent_scale_training_contract(args.persistent_contract)
    if (
        multi_contract.sha256 != contract.payload["identity"]["multi_group_contract_sha256"]
        or persistent_contract.sha256 != contract.payload["identity"]["persistent_contract_sha256"]
        or step_contract.sha256 != multi_contract.payload["identity"]["direct_step_transfer_contract_sha256"]
        or multi_contract.scheduled_group != contract.train_groups[0]
        or multi_contract.guard_groups != contract.train_groups[1:]
        or multi_contract.monitor_updates != contract.monitor_updates
    ):
        raise ValueError("artifact-aware direction contract chain drift")
    schedule = build_persistent_scale_training_schedule()
    if schedule.sha256 != contract.payload["identity"]["persistent_schedule_sha256"]:
        raise ValueError("artifact-aware direction persistent schedule drift")
    generalization_contract = load_coupled_gpp_direct_generalization_contract(args.generalization_contract)
    persistent_result = _load_persistent_result(
        args.persistent_result,
        audit_contract=generalization_contract,
        persistent_contract=persistent_contract,
        schedule=schedule,
    )
    replay_records = persistent_result["training"]["updates"]

    recovery_before = args.parent_recovery.stat()
    bootstrap = _bootstrap_update7(args, persistent_contract)
    context = bootstrap["context"]
    required = set(contract.train_artifacts) | set(contract.monitor_updates)
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"artifact-aware direction prepared coverage drift: {missing}")

    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    initialized_parameters = bootstrap["initialized_parameters"]
    gram_measurement = _make_gradient_gram(jax)
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)
    common = dict(
        initialized_parameters=initialized_parameters,
        steps=bootstrap["steps"],
        gradient_steps=bootstrap["gradient_steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        flux_basis=bootstrap["flux_basis"],
        persistent_contract=persistent_contract,
        gram_measurement=gram_measurement,
        combine_direction=None,
        direct_virtual_step=direct_virtual_step,
    )
    common["combine_direction"] = _make_combined_direction(jax)
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
    before_update_two = update_one["updated_trainable"]
    parameters = replace_coupled_gpp_trainable(initialized_parameters, before_update_two)
    scheduled = _measure_source_gradients(
        args,
        context,
        parameters=parameters,
        trainable=before_update_two,
        steps=bootstrap["steps"],
        gradient_steps=bootstrap["gradient_steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        group=contract.train_groups[0],
        flux_basis=bootstrap["flux_basis"],
        macro_update=2,
    )
    del parameters

    flatten_gradient = _make_flattened_unit_gradient(jax)
    artifact_measurements = tuple(
        _measure_direct_artifact_gradients(
            args,
            context,
            trainable=before_update_two,
            direct_steps=bootstrap["direct_steps"],
            updates=group,
            flatten_gradient=flatten_gradient,
            label=("scheduled" if index == 0 else f"guard_{index}"),
        )
        for index, group in enumerate(contract.train_groups)
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
            raise RuntimeError(f"artifact-aware scheduled gradient {source} is invalid")
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
    if tuple(constraint_labels) != contract.source_registry or len(gradient_reports) != len(contract.source_registry):
        raise RuntimeError("artifact-aware gradient registry drift")

    unit_matrix = jnp.stack(tuple(unit_gradients))
    jax.block_until_ready(unit_matrix)
    for measurement in artifact_measurements:
        del measurement["unit_gradients"]
    del unit_gradients
    gc.collect()
    flat_gram = _make_flat_gram(jax)
    gram = flat_gram(unit_matrix)
    jax.block_until_ready(gram)
    gram_host = np.asarray(gram, dtype=np.float64)
    diagnostics = gradient_direction_diagnostics(gram_host, labels=contract.source_registry)
    selected_direction = diagnostics["directions"]["minimum_norm_unit"]
    minimum_cosine = float(contract.payload["direction"]["minimum_constraint_cosine"])
    common_first_order = all(
        float(selected_direction["normalized_directional_derivatives"][name]) >= minimum_cosine
        for name in contract.source_registry
    )

    proposed = None
    common_norm_host = None
    if common_first_order:
        weights = jnp.asarray(
            [selected_direction["normalized_simplex_weights"][name] for name in contract.source_registry],
            dtype=jnp.float64,
        )
        combine_flat = _make_flat_direction(jax)
        flat_direction, common_norm, common_nonfinite = combine_flat(unit_matrix, weights)
        jax.block_until_ready((flat_direction, common_norm, common_nonfinite))
        common_norm_host = float(np.asarray(common_norm))
        if int(np.asarray(common_nonfinite)) != 0 or common_norm_host <= 0.0:
            raise RuntimeError("artifact-aware common direction is invalid")
        flat_template, unravel = ravel_pytree(before_update_two)
        direction_tree = unravel(flat_direction)
        del flat_template
        proposed, proposal_nonfinite = direct_virtual_step(
            before_update_two,
            direction_tree,
            common_norm,
            np.asarray(float(contract.payload["direction"]["full_parameter_step_l2"]), dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposal_nonfinite))
        if int(np.asarray(proposal_nonfinite)) != 0:
            raise RuntimeError("artifact-aware direction realization is nonfinite")
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
                before_trainable=before_update_two,
                proposed_trainable=proposed,
                before_losses=scheduled["source_mean_losses"],
                scales=(scale,),
                scale_one_evaluation=None,
                steps=bootstrap["steps"],
                direct_steps=bootstrap["direct_steps"],
                state_steps=bootstrap["state_steps"],
                layout=bootstrap["layout"],
                group=contract.train_groups[0],
                flux_basis=bootstrap["flux_basis"],
                coefficients=bootstrap["coefficients"],
                relative_tolerance=relative_tolerance,
                label="artifact_aware_scheduled",
                direction_norm=float(contract.payload["direction"]["full_parameter_step_l2"]),
            )[0]
            candidate = _scaled_parameters(jax, before_update_two, proposed, scale)
            artifact_after = _evaluate_direct_groups(
                args,
                context,
                trainable=candidate,
                direct_steps=bootstrap["direct_steps"],
                groups=contract.train_groups,
                label=f"artifact_aware_scale_{scale:g}",
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
                expected_artifact_count=len(contract.train_artifacts),
                relative_tolerance=relative_tolerance,
            )
            scale_records.append(record)
            print(
                f"coupled_gpp artifact_aware scale={scale:g} train_only_passed={record['train_only_scale_passed']}",
                flush=True,
            )
            del candidate
            if record["train_only_scale_passed"]:
                selected_record = record
                break

    selected_scale = None if selected_record is None else float(selected_record["scale"])
    monitor_before = None
    selected_monitor = None
    if selected_record is None:
        inspected_trainable = before_update_two
        selected_monitor_change = None
    else:
        inspected_trainable = _scaled_parameters(jax, before_update_two, proposed, selected_scale)
        monitor_before = _direct_panel(
            args,
            context,
            trainable=before_update_two,
            direct_steps=bootstrap["direct_steps"],
            updates=contract.monitor_updates,
            label="artifact_aware_selected_monitor_before",
        )
        monitor_after = _direct_panel(
            args,
            context,
            trainable=inspected_trainable,
            direct_steps=bootstrap["direct_steps"],
            updates=contract.monitor_updates,
            label="artifact_aware_selected_monitor_after",
        )
        selected_monitor = _panel_change(monitor_before, monitor_after)
        selected_monitor_change = float(selected_monitor["mean_relative_change"])
    classification = classify_artifact_aware_direction(
        common_first_order_direction=common_first_order,
        selected_scale=selected_scale,
        monitor_relative_change=selected_monitor_change,
    )

    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    fixed_report = _fixed_parameter_report(
        jax,
        initialized_parameters,
        replace_coupled_gpp_trainable(initialized_parameters, inspected_trainable),
    )
    predecessor_unchanged = sha256_file(args.multi_group_result) == multi_digest
    step_result_unchanged = sha256_file(args.direct_step_transfer_result) == step_digest
    engineering_passed = bool(
        recovery_unchanged
        and fixed_report["exactly_unchanged"]
        and predecessor_unchanged
        and step_result_unchanged
        and len(gradient_reports) == len(contract.source_registry)
        and all(int(record["nonfinite_count"]) == 0 for record in gradient_reports.values())
    )
    if not engineering_passed:
        raise RuntimeError("artifact-aware direction engineering gate failed")

    next_key = {
        "artifact_aware_direction_transfers_to_monitor": "on_artifact_aware_direction_transfers_to_monitor",
        "artifact_aware_direction_fails_monitor": "on_artifact_aware_direction_fails_monitor",
        "no_artifact_aware_common_direction": "on_no_artifact_aware_common_direction",
        "no_artifact_aware_finite_step": "on_no_artifact_aware_finite_step",
    }[classification]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "artifact_aware_contract_sha256": contract.sha256,
            "multi_group_contract_sha256": multi_contract.sha256,
            "multi_group_result_sha256": multi_digest,
            "persistent_contract_sha256": persistent_contract.sha256,
            "persistent_schedule_sha256": schedule.sha256,
        },
        "policy": {
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
        },
        "direction": {
            "source_registry": list(contract.source_registry),
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
            "monitor": monitor_before,
        },
        "scale_evaluations": scale_records,
        "selection": {
            "monitor_excluded": True,
            "stopped_after_first_train_only_passing_scale": selected_scale is not None,
            "selected_scale": selected_scale,
            "selected_monitor_relative_change": selected_monitor_change,
            "selected_monitor": selected_monitor,
            "classification": classification,
        },
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "multi_group_result_unchanged": predecessor_unchanged,
            "direct_step_transfer_result_unchanged": step_result_unchanged,
            "scientific_model_promotion_implied": False,
        },
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][next_key],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _multi_group_parser()
    parser.add_argument("--multi-group-result", type=Path, required=True)
    parser.add_argument("--artifact-aware-contract", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
