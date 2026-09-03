"""Audit finite-step GPP transfer at persistent update 2 without retention."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_step_transfer import (
    AUDITED_PERSISTENT_UPDATE,
    DIRECT_GENERALIZATION_RESULT_SHA256,
    classify_step_transfer,
    load_coupled_gpp_direct_step_transfer_contract,
    select_largest_train_only_scale,
    train_only_scale_passed,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    build_persistent_scale_training_schedule,
    load_coupled_gpp_persistent_scale_training_contract,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_generalization import (
    _bootstrap_update7,
    _direct_panel,
    _replay_direction,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_generalization import (
    _parser as _generalization_parser,
)
from scripts.hpc.probe_successor_direct_trust_region import _make_combined_direction
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import _fixed_parameter_report
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
    _make_dtype_preserving_direct_virtual_step,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import (
    _evaluate_scales,
    _scaled_parameters,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_direct_step_transfer_audit_result_v1"


def _load_generalization_result(path: Path, contract: Any) -> Mapping[str, Any]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    acceptance = report.get("acceptance", {})
    attribution = report.get("first_regression_attribution", {})
    replay = tuple(report.get("trajectory", {}).get("replay_updates", ()))
    if (
        digest != DIRECT_GENERALIZATION_RESULT_SHA256
        or digest != contract.payload["identity"]["direct_generalization_result_sha256"]
        or report.get("status") != "passed"
        or report.get("classification") != "direct_gpp_generalization_attribution_completed"
        or report.get("identity", {}).get("audit_contract_sha256")
        != contract.payload["identity"]["direct_generalization_contract_sha256"]
        or report.get("identity", {}).get("persistent_result_sha256")
        != contract.payload["identity"]["persistent_result_sha256"]
        or attribution.get("persistent_update") != AUDITED_PERSISTENT_UPDATE
        or len(replay) != 10
        or acceptance.get("engineering_gate_passed") is not True
        or acceptance.get("scientific_model_promotion_implied") is not False
        or report.get("policy", {}).get("checkpoint_written") is not False
        or report.get("policy", {}).get("recovery_written") is not False
    ):
        raise ValueError("direct-step transfer predecessor result drift")
    return report


def _relative_change(before: float, after: float) -> float:
    first = float(before)
    second = float(after)
    if not np.isfinite(first) or first <= 0.0 or not np.isfinite(second) or second <= 0.0:
        raise ValueError("direct-step transfer loss is invalid")
    return (second - first) / first


def _panel_change(before: Mapping[str, Any], after: Mapping[str, Any]) -> Mapping[str, Any]:
    first_records = tuple(before["artifact_losses"])
    second_records = tuple(after["artifact_losses"])
    if tuple(item["artifact_update"] for item in first_records) != tuple(
        item["artifact_update"] for item in second_records
    ):
        raise ValueError("direct-step transfer panel order drift")
    artifact_changes = []
    for first, second in zip(first_records, second_records, strict=True):
        change = _relative_change(first["loss"], second["loss"])
        artifact_changes.append(
            {
                "artifact_update": int(first["artifact_update"]),
                "before": float(first["loss"]),
                "after": float(second["loss"]),
                "relative_change": change,
                "improved": change < 0.0,
            }
        )
    return {
        "before_mean": float(before["mean_loss"]),
        "after_mean": float(after["mean_loss"]),
        "mean_relative_change": _relative_change(before["mean_loss"], after["mean_loss"]),
        "improved_artifact_count": sum(bool(item["improved"]) for item in artifact_changes),
        "artifact_changes": artifact_changes,
    }


def _evaluate_direct_groups(
    args: Any,
    context: Mapping[str, Any],
    *,
    trainable: Any,
    direct_steps: Mapping[str, Any],
    groups: tuple[tuple[int, ...], ...],
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _direct_panel(
            args,
            context,
            trainable=trainable,
            direct_steps=direct_steps,
            updates=group,
            label=f"{label}_group_{index}",
        )
        for index, group in enumerate(groups, start=1)
    )


def run(args: Any) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_coupled_gpp_direct_step_transfer_contract(args.step_transfer_contract)
    source_digest = sha256_file(args.direct_generalization_result)
    predecessor = _load_generalization_result(args.direct_generalization_result, contract)
    persistent_contract = load_coupled_gpp_persistent_scale_training_contract(args.persistent_contract)
    if persistent_contract.sha256 != contract.payload["identity"]["persistent_contract_sha256"]:
        raise ValueError("direct-step transfer persistent contract chain drift")
    schedule = build_persistent_scale_training_schedule()
    if schedule.sha256 != contract.payload["identity"]["persistent_schedule_sha256"]:
        raise ValueError("direct-step transfer persistent schedule drift")

    recovery_before = args.parent_recovery.stat()
    bootstrap = _bootstrap_update7(args, persistent_contract)
    context = bootstrap["context"]
    required = set(contract.scheduled_group) | set(contract.monitor_updates)
    required |= {value for group in contract.guard_groups for value in group}
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"direct-step transfer prepared coverage drift: {missing}")

    jax = context["runtime"]["jax"]
    initialized_parameters = bootstrap["initialized_parameters"]
    gram_measurement = _make_gradient_gram(jax)
    combine_direction = _make_combined_direction(jax)
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)
    replay_records = predecessor["trajectory"]["replay_updates"]
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
        combine_direction=combine_direction,
        direct_virtual_step=direct_virtual_step,
    )
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
    update_two = _replay_direction(
        args,
        context,
        trainable=before_update_two,
        group=contract.scheduled_group,
        persistent_update=AUDITED_PERSISTENT_UPDATE,
        expected_record=replay_records[1],
        selected_scale=float(contract.payload["direction"]["audited_original_selected_scale"]),
        **common,
    )
    proposed = update_two["proposed_trainable"]
    before_losses = update_two["record"]["source_mean_losses_before"]

    guard_before = _evaluate_direct_groups(
        args,
        context,
        trainable=before_update_two,
        direct_steps=bootstrap["direct_steps"],
        groups=contract.guard_groups,
        label="guard_before",
    )
    combined_guard_before = float(np.mean([float(panel["mean_loss"]) for panel in guard_before]))
    relative_tolerance = float(
        contract.payload["scale_selection"]["each_train_only_guard_group_direct_gpp_relative_change_maximum"]
    )
    scale_records = []
    for scale in contract.scales:
        scheduled = _evaluate_scales(
            args,
            context,
            initialized_parameters=initialized_parameters,
            before_trainable=before_update_two,
            proposed_trainable=proposed,
            before_losses=before_losses,
            scales=(scale,),
            scale_one_evaluation=None,
            steps=bootstrap["steps"],
            direct_steps=bootstrap["direct_steps"],
            state_steps=bootstrap["state_steps"],
            layout=bootstrap["layout"],
            group=contract.scheduled_group,
            flux_basis=bootstrap["flux_basis"],
            coefficients=bootstrap["coefficients"],
            relative_tolerance=relative_tolerance,
            label="direct_step_transfer_scheduled",
            direction_norm=float(contract.payload["direction"]["full_parameter_step_l2"]),
        )[0]
        candidate = _scaled_parameters(jax, before_update_two, proposed, scale)
        guard_after = _evaluate_direct_groups(
            args,
            context,
            trainable=candidate,
            direct_steps=bootstrap["direct_steps"],
            groups=contract.guard_groups,
            label=f"guard_scale_{scale:g}",
        )
        guard_changes = tuple(
            _panel_change(first, second) for first, second in zip(guard_before, guard_after, strict=True)
        )
        combined_guard_after = float(np.mean([float(panel["mean_loss"]) for panel in guard_after]))
        record = {
            "scale": float(scale),
            "parameter_delta_l2": float(scale) * float(contract.payload["direction"]["full_parameter_step_l2"]),
            "scheduled_four_source_finite_step_accepted": bool(scheduled["finite_step_accepted"]),
            "scheduled_physical_hard_gate_passed": bool(scheduled["hard_gate_passed"]),
            "scheduled_evaluation": scheduled,
            "guard_group_direct_gpp_relative_changes": [float(item["mean_relative_change"]) for item in guard_changes],
            "combined_guard_direct_gpp_before": combined_guard_before,
            "combined_guard_direct_gpp_after": combined_guard_after,
            "combined_guard_direct_gpp_relative_change": _relative_change(combined_guard_before, combined_guard_after),
            "guard_groups": list(guard_changes),
        }
        record["train_only_scale_passed"] = train_only_scale_passed(
            record,
            relative_tolerance=relative_tolerance,
        )
        scale_records.append(record)
        print(
            f"coupled_gpp direct_step_transfer scale={scale:g} train_only_passed={record['train_only_scale_passed']}",
            flush=True,
        )
        del candidate

    selected_scale = select_largest_train_only_scale(
        scale_records,
        relative_tolerance=relative_tolerance,
    )
    selected_record = next(
        (record for record in scale_records if selected_scale is not None and record["scale"] == selected_scale),
        None,
    )
    monitor_before = None
    selected_monitor = None
    if selected_record is None:
        inspected_trainable = before_update_two
        selected_monitor_change = None
    else:
        inspected_trainable = _scaled_parameters(
            jax,
            before_update_two,
            proposed,
            selected_record["scale"],
        )
        monitor_before = _direct_panel(
            args,
            context,
            trainable=before_update_two,
            direct_steps=bootstrap["direct_steps"],
            updates=contract.monitor_updates,
            label="selected_monitor_before",
        )
        monitor_after = _direct_panel(
            args,
            context,
            trainable=inspected_trainable,
            direct_steps=bootstrap["direct_steps"],
            updates=contract.monitor_updates,
            label="selected_monitor_after",
        )
        selected_monitor = _panel_change(monitor_before, monitor_after)
        selected_monitor_change = float(selected_monitor["mean_relative_change"])
    classification = classify_step_transfer(
        selected_scale,
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
    source_unchanged = sha256_file(args.direct_generalization_result) == source_digest
    engineering_passed = bool(
        len(scale_records) == len(contract.scales)
        and recovery_unchanged
        and fixed_report["exactly_unchanged"]
        and source_unchanged
    )
    if not engineering_passed:
        raise RuntimeError("direct-step transfer engineering gate failed")

    next_key = {
        "train_only_scale_transfers_to_monitor": "on_train_only_scale_transfers_to_monitor",
        "train_only_scale_fails_monitor": "on_train_only_scale_fails_monitor",
        "no_train_only_shared_finite_scale": "on_no_train_only_shared_finite_scale",
    }[classification]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "step_transfer_contract_sha256": contract.sha256,
            "direct_generalization_result_sha256": source_digest,
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
            "audited_update_2_direction": update_two["record"],
        },
        "selection": {
            "monitor_excluded": True,
            "train_only_guard_persistent_updates": list(
                contract.payload["data"]["train_only_guard_persistent_updates"]
            ),
            "selected_scale": selected_scale,
            "selected_monitor_relative_change": selected_monitor_change,
            "selected_monitor": selected_monitor,
            "classification": classification,
        },
        "baseline": {
            "guard_groups": list(guard_before),
            "combined_guard_direct_gpp_mean": combined_guard_before,
            "monitor": monitor_before,
        },
        "scale_evaluations": scale_records,
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "direct_generalization_result_unchanged": source_unchanged,
            "scientific_model_promotion_implied": False,
        },
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][next_key],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _generalization_parser()
    parser.add_argument("--step-transfer-contract", type=Path, required=True)
    parser.add_argument("--direct-generalization-result", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
