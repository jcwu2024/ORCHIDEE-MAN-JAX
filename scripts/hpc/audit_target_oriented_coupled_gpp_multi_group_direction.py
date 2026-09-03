"""Audit an eight-constraint multi-group direct-GPP direction without training."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_generalization import (
    load_coupled_gpp_direct_generalization_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_step_transfer import (
    load_coupled_gpp_direct_step_transfer_contract,
    select_largest_train_only_scale,
    train_only_scale_passed,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_multi_group_direction import (
    DIRECT_STEP_TRANSFER_RESULT_SHA256,
    MULTI_GROUP_GRADIENT_SOURCES,
    classify_multi_group_direction,
    load_coupled_gpp_multi_group_direction_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
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
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_generalization import (
    _parser as _generalization_parser,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_step_transfer import (
    _evaluate_direct_groups,
    _panel_change,
    _relative_change,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _add_gradient, _zero_gradient
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.probe_successor_direct_trust_region import _make_combined_direction
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import _fixed_parameter_report
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
    _make_dtype_preserving_direct_virtual_step,
    _measure_source_gradients,
)
from scripts.hpc.run_target_oriented_coupled_gpp_optimizer_preflight import _mean_gradient
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import (
    _evaluate_scales,
    _scaled_parameters,
)
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import _tree_scalar_report
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import _direct_gradient

SCHEMA_VERSION = "target_oriented_coupled_gpp_multi_group_direction_audit_result_v1"


def _load_step_transfer_result(path: Path, contract: Any) -> Mapping[str, Any]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        digest != DIRECT_STEP_TRANSFER_RESULT_SHA256
        or digest != contract.payload["identity"]["direct_step_transfer_result_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_direct_step_transfer_audit_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "train_only_scale_fails_monitor"
        or report.get("selection", {}).get("selected_scale") != 0.25
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
        or report.get("policy", {}).get("checkpoint_written") is not False
    ):
        raise ValueError("multi-group direction predecessor result drift")
    return report


def _measure_direct_group_gradient(
    args: Any,
    context: Mapping[str, Any],
    *,
    trainable: Any,
    direct_steps: Mapping[str, Any],
    updates: Sequence[int],
    label: str,
) -> Mapping[str, Any]:
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    started = time.perf_counter()
    gradient_sum = _zero_gradient(jax, jnp, trainable)
    records = []
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
        gradient_sum = _add_gradient(jax, jnp, gradient_sum, gradient)
        jax.block_until_ready(gradient_sum)
        records.append(
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
            f"coupled_gpp multi_group {label}_gradient={ordinal}/{len(updates)} artifact={update}",
            flush=True,
        )
        del direct, gradient, artifact
    mean_gradient = _mean_gradient(jax, gradient_sum, len(updates))
    jax.block_until_ready(mean_gradient)
    gradient_report = _tree_scalar_report(jax, jnp, mean_gradient)
    if (
        not gradient_report["all_finite"]
        or int(gradient_report["element_count"]) != TRAINABLE_PARAMETER_COUNT
        or float(gradient_report["l2_norm"]) <= 0.0
    ):
        raise RuntimeError(f"multi-group direct-GPP gradient {label} is invalid")
    losses = np.asarray([record["loss"] for record in records], dtype=np.float64)
    return {
        "gradient": mean_gradient,
        "gradient_report": gradient_report,
        "panel": {
            "artifact_updates": [int(value) for value in updates],
            "mean_loss": float(np.mean(losses)),
            "artifact_losses": records,
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }


def run(args: Any) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_coupled_gpp_multi_group_direction_contract(args.multi_group_contract)
    step_contract = load_coupled_gpp_direct_step_transfer_contract(args.step_transfer_contract)
    step_digest = sha256_file(args.direct_step_transfer_result)
    _load_step_transfer_result(args.direct_step_transfer_result, contract)
    persistent_contract = load_coupled_gpp_persistent_scale_training_contract(args.persistent_contract)
    if (
        step_contract.sha256 != contract.payload["identity"]["direct_step_transfer_contract_sha256"]
        or persistent_contract.sha256 != contract.payload["identity"]["persistent_contract_sha256"]
        or step_contract.scheduled_group != contract.scheduled_group
        or step_contract.guard_groups != contract.guard_groups
        or step_contract.monitor_updates != contract.monitor_updates
    ):
        raise ValueError("multi-group direction contract chain drift")
    schedule = build_persistent_scale_training_schedule()
    if schedule.sha256 != contract.payload["identity"]["persistent_schedule_sha256"]:
        raise ValueError("multi-group direction persistent schedule drift")
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
    required = set(contract.scheduled_group) | set(contract.monitor_updates)
    required |= {value for group in contract.guard_groups for value in group}
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"multi-group direction prepared coverage drift: {missing}")

    jax = context["runtime"]["jax"]
    initialized_parameters = bootstrap["initialized_parameters"]
    gram_measurement = _make_gradient_gram(jax)
    combine_direction = _make_combined_direction(jax)
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
        group=contract.scheduled_group,
        flux_basis=bootstrap["flux_basis"],
        macro_update=2,
    )
    guard_measurements = tuple(
        _measure_direct_group_gradient(
            args,
            context,
            trainable=before_update_two,
            direct_steps=bootstrap["direct_steps"],
            updates=group,
            label=f"guard_{index}",
        )
        for index, group in enumerate(contract.guard_groups, start=1)
    )

    gradients = {
        MULTI_GROUP_GRADIENT_SOURCES[0]: scheduled["source_gradients"][DIRECT_SOURCE],
        **{MULTI_GROUP_GRADIENT_SOURCES[index]: guard_measurements[index - 1]["gradient"] for index in range(1, 5)},
        MULTI_GROUP_GRADIENT_SOURCES[5]: scheduled["source_gradients"][GRADIENT_SOURCES[1]],
        MULTI_GROUP_GRADIENT_SOURCES[6]: scheduled["source_gradients"][GRADIENT_SOURCES[2]],
        MULTI_GROUP_GRADIENT_SOURCES[7]: scheduled["source_gradients"][GRADIENT_SOURCES[3]],
    }
    gradient_reports = {
        MULTI_GROUP_GRADIENT_SOURCES[0]: scheduled["source_reports"][DIRECT_SOURCE],
        **{
            MULTI_GROUP_GRADIENT_SOURCES[index]: guard_measurements[index - 1]["gradient_report"]
            for index in range(1, 5)
        },
        MULTI_GROUP_GRADIENT_SOURCES[5]: scheduled["source_reports"][GRADIENT_SOURCES[1]],
        MULTI_GROUP_GRADIENT_SOURCES[6]: scheduled["source_reports"][GRADIENT_SOURCES[2]],
        MULTI_GROUP_GRADIENT_SOURCES[7]: scheduled["source_reports"][GRADIENT_SOURCES[3]],
    }
    ordered = tuple(gradients[name] for name in MULTI_GROUP_GRADIENT_SOURCES)
    gram, nonfinite = gram_measurement(ordered)
    jax.block_until_ready((gram, nonfinite))
    gram_host = np.asarray(gram, dtype=np.float64)
    nonfinite_host = np.asarray(nonfinite, dtype=np.int64)
    diagnostics = gradient_direction_diagnostics(gram_host, labels=MULTI_GROUP_GRADIENT_SOURCES)
    selected_direction = diagnostics["directions"]["minimum_norm_unit"]
    minimum_cosine = float(contract.payload["direction"]["minimum_constraint_cosine"])
    common_first_order = bool(
        np.all(nonfinite_host == 0)
        and all(
            float(selected_direction["normalized_directional_derivatives"][name]) >= minimum_cosine
            for name in MULTI_GROUP_GRADIENT_SOURCES
        )
    )

    guard_before = tuple(value["panel"] for value in guard_measurements)
    combined_guard_before = float(np.mean([float(value["mean_loss"]) for value in guard_before]))
    relative_tolerance = float(
        contract.payload["scale_selection"]["each_train_only_guard_group_direct_gpp_relative_change_maximum"]
    )
    proposed = None
    common_norm_host = None
    scale_records = []
    if common_first_order:
        weights = np.asarray(
            [selected_direction["weights"][name] for name in MULTI_GROUP_GRADIENT_SOURCES],
            dtype=np.float64,
        )
        common_gradient, common_norm, common_nonfinite = combine_direction(ordered, weights)
        jax.block_until_ready((common_gradient, common_norm, common_nonfinite))
        common_norm_host = float(np.asarray(common_norm))
        proposed, proposal_nonfinite = direct_virtual_step(
            before_update_two,
            common_gradient,
            common_norm,
            np.asarray(float(contract.payload["direction"]["full_parameter_step_l2"]), dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposal_nonfinite))
        if int(np.asarray(common_nonfinite)) != 0 or int(np.asarray(proposal_nonfinite)) != 0:
            raise RuntimeError("multi-group common direction realization is nonfinite")

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
                group=contract.scheduled_group,
                flux_basis=bootstrap["flux_basis"],
                coefficients=bootstrap["coefficients"],
                relative_tolerance=relative_tolerance,
                label="multi_group_scheduled",
                direction_norm=float(contract.payload["direction"]["full_parameter_step_l2"]),
            )[0]
            candidate = _scaled_parameters(jax, before_update_two, proposed, scale)
            guard_after = _evaluate_direct_groups(
                args,
                context,
                trainable=candidate,
                direct_steps=bootstrap["direct_steps"],
                groups=contract.guard_groups,
                label=f"multi_group_scale_{scale:g}",
            )
            guard_changes = tuple(
                _panel_change(first, second) for first, second in zip(guard_before, guard_after, strict=True)
            )
            combined_guard_after = float(np.mean([float(value["mean_loss"]) for value in guard_after]))
            record = {
                "scale": float(scale),
                "parameter_delta_l2": float(scale) * float(contract.payload["direction"]["full_parameter_step_l2"]),
                "scheduled_four_source_finite_step_accepted": bool(scheduled_after["finite_step_accepted"]),
                "scheduled_physical_hard_gate_passed": bool(scheduled_after["hard_gate_passed"]),
                "scheduled_evaluation": scheduled_after,
                "guard_group_direct_gpp_relative_changes": [
                    float(value["mean_relative_change"]) for value in guard_changes
                ],
                "combined_guard_direct_gpp_before": combined_guard_before,
                "combined_guard_direct_gpp_after": combined_guard_after,
                "combined_guard_direct_gpp_relative_change": _relative_change(
                    combined_guard_before,
                    combined_guard_after,
                ),
                "guard_groups": list(guard_changes),
            }
            record["train_only_scale_passed"] = train_only_scale_passed(
                record,
                relative_tolerance=relative_tolerance,
            )
            scale_records.append(record)
            print(
                f"coupled_gpp multi_group scale={scale:g} train_only_passed={record['train_only_scale_passed']}",
                flush=True,
            )
            del candidate

    selected_scale = (
        select_largest_train_only_scale(
            scale_records,
            relative_tolerance=relative_tolerance,
        )
        if common_first_order
        else None
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
        inspected_trainable = _scaled_parameters(jax, before_update_two, proposed, selected_record["scale"])
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
    classification = classify_multi_group_direction(
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
    predecessor_unchanged = sha256_file(args.direct_step_transfer_result) == step_digest
    engineering_passed = bool(
        recovery_unchanged
        and fixed_report["exactly_unchanged"]
        and predecessor_unchanged
        and len(gradient_reports) == len(MULTI_GROUP_GRADIENT_SOURCES)
        and np.all(nonfinite_host == 0)
    )
    if not engineering_passed:
        raise RuntimeError("multi-group direction engineering gate failed")

    next_key = {
        "multi_group_direction_transfers_to_monitor": "on_multi_group_direction_transfers_to_monitor",
        "multi_group_direction_fails_monitor": "on_multi_group_direction_fails_monitor",
        "no_multi_group_common_direction": "on_no_multi_group_common_direction",
        "no_multi_group_finite_step": "on_no_multi_group_finite_step",
    }[classification]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "multi_group_contract_sha256": contract.sha256,
            "direct_step_transfer_contract_sha256": step_contract.sha256,
            "direct_step_transfer_result_sha256": step_digest,
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
            "source_registry": list(MULTI_GROUP_GRADIENT_SOURCES),
            "source_gradients": gradient_reports,
            "gradient_gram": gram_host.tolist(),
            "source_nonfinite_counts": {
                name: int(nonfinite_host[index]) for index, name in enumerate(MULTI_GROUP_GRADIENT_SOURCES)
            },
            "diagnostics": diagnostics,
            "selected_minimum_norm_direction": selected_direction,
            "common_first_order_direction": common_first_order,
            "combined_gradient_norm": common_norm_host,
            "full_parameter_step_l2": float(contract.payload["direction"]["full_parameter_step_l2"]),
        },
        "baseline": {
            "scheduled_source_mean_losses": scheduled["source_mean_losses"],
            "scheduled_artifact_records": scheduled["artifact_records"],
            "guard_groups": list(guard_before),
            "combined_guard_direct_gpp_mean": combined_guard_before,
            "monitor": monitor_before,
        },
        "scale_evaluations": scale_records,
        "selection": {
            "monitor_excluded": True,
            "selected_scale": selected_scale,
            "selected_monitor_relative_change": selected_monitor_change,
            "selected_monitor": selected_monitor,
            "classification": classification,
        },
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "direct_step_transfer_result_unchanged": predecessor_unchanged,
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
    parser.add_argument("--direct-step-transfer-result", type=Path, required=True)
    parser.add_argument("--multi-group-contract", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
