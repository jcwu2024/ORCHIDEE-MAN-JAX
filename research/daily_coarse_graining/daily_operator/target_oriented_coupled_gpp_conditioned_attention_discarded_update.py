"""Frozen discarded-update preflight for conditioned coupled-GPP attention."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_coupled_gpp_artifact_aware_direction import (
    SCHEDULED_NON_GPP_SOURCES,
)
from .target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    FROZEN_CONTRACT_SHA256 as ARTIFACT_PERSISTENT_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_conditioned_attention import (
    FROZEN_CONTRACT_SHA256 as CONDITIONED_ATTENTION_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_direct_generalization import MONITOR_UPDATES
from .target_oriented_coupled_gpp_objective import DIRECT_SOURCE
from .target_oriented_coupled_gpp_persistent_scale_training import (
    SCHEDULE_SHA256,
    build_persistent_scale_training_schedule,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_conditioned_attention_discarded_update_v1"
FROZEN_CONTRACT_SHA256 = "b29e92b5696b0d4f980b33f1b802eb8f014aa61f7d9e49e1ca82134fcb97d27d"
CONDITIONED_ADMISSION_REPORT_SHA256 = "080f6b9dbe3ba7f74af1311dec4ee72ed1fba11dbf634ddec97bb08019233263"
PARENT_REPORT_SHA256 = "ded2f4a59fbcd8fafd84d7a64e597017911f1222f5f748170fc82dec97d7aa3a"
PARENT_RECOVERY_SHA256 = "da019556821239cbf50bd67372002f4217bb10f70eafef55375c9c0c31de857c"
PARENT_TRAINABLE_PARAMETER_COUNT = 15_960_195
TRAINABLE_PARAMETER_COUNT = 15_960_211
TOTAL_PARAMETER_COUNT = 109_464_375
RETAINED_PARAMETER_COUNT = 93_504_164
LOGICAL_UPDATE = 3
PANEL_GROUP_UPDATES = (3, 15, 16, 17, 18)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_conditioned_attention_discarded_update_v1.json"
)


def _artifact_sources(groups: Sequence[Sequence[int]]) -> tuple[str, ...]:
    return tuple(f"{DIRECT_SOURCE}.artifact_{int(update)}" for group in groups for update in group)


@dataclass(frozen=True)
class ConditionedAttentionDiscardedUpdateContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def train_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in group) for group in self.payload["data"]["train_only_artifact_groups"])

    @property
    def train_artifacts(self) -> tuple[int, ...]:
        return tuple(value for group in self.train_groups for value in group)

    @property
    def source_registry(self) -> tuple[str, ...]:
        return _artifact_sources(self.train_groups) + SCHEDULED_NON_GPP_SOURCES

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["fixed_monitor_artifact_updates"])

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["step_calibration"]["scale_multipliers_descending"])

    @property
    def required_science_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["baseline"]["required_science_keys"])

    @property
    def critical_science_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["baseline"]["critical_long_horizon_keys"])

    @property
    def supporting_science_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["baseline"]["supporting_science_keys"])


def _relative_change(before: float, after: float, *, floor: float = 1.0e-12) -> float:
    first = float(before)
    second = float(after)
    if not np.isfinite(first) or first < 0.0 or not np.isfinite(second) or second < 0.0:
        raise ValueError("conditioned discarded-update science value is invalid")
    return (second - first) / max(first, float(floor))


def _science_key_losses(
    panel: Mapping[str, Any],
    required_keys: Sequence[str],
) -> Mapping[str, float]:
    acceptance = panel.get("acceptance", {})
    if set(acceptance) != {"7", "30"}:
        raise ValueError("conditioned discarded-update science horizons drift")
    result: dict[str, float] = {}
    for horizon in ("7", "30"):
        keys = acceptance[horizon].get("keys", {})
        if not set(required_keys) <= set(keys):
            raise ValueError("conditioned discarded-update science key registry drift")
        for name in required_keys:
            record = keys[name]
            rmse = float(record["normalized_rmse"])
            bias = float(record["normalized_signed_bias"])
            if not np.isfinite(rmse) or rmse < 0.0 or not np.isfinite(bias):
                raise ValueError("conditioned discarded-update science metric is invalid")
            result[f"{horizon}.{name}"] = 0.5 * (rmse * rmse + bias * bias)
    return result


def absolute_science_gate(
    contract: ConditionedAttentionDiscardedUpdateContract,
    panel: Mapping[str, Any],
) -> Mapping[str, Any]:
    spec = contract.payload["absolute_science_gate"]
    acceptance = panel.get("acceptance", {})
    checks: dict[str, bool] = {}
    for horizon, rmse_name, bias_name in (
        ("7", "seven_day_normalized_rmse_maximum", "seven_day_absolute_normalized_signed_bias_maximum"),
        ("30", "thirty_day_normalized_rmse_maximum", "thirty_day_absolute_normalized_signed_bias_maximum"),
    ):
        keys = acceptance.get(horizon, {}).get("keys", {})
        if not set(contract.required_science_keys) <= set(keys):
            raise ValueError("conditioned discarded-update absolute science key registry drift")
        for name in contract.required_science_keys:
            record = keys[name]
            checks[f"{horizon}.{name}"] = bool(
                np.isfinite(float(record["normalized_rmse"]))
                and np.isfinite(float(record["normalized_signed_bias"]))
                and float(record["normalized_rmse"]) <= float(spec[rmse_name])
                and abs(float(record["normalized_signed_bias"])) <= float(spec[bias_name])
            )
    checks["physical_hard_gate"] = panel.get("hard_gate_passed") is True
    return {"passed": all(checks.values()), "checks": checks}


def science_progress_gate(
    contract: ConditionedAttentionDiscardedUpdateContract,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    gate_name: str,
) -> Mapping[str, Any]:
    if gate_name not in ("train_only_selection", "fixed_monitor"):
        raise ValueError("conditioned discarded-update science gate name drift")
    spec = contract.payload[gate_name]
    initial = _science_key_losses(before, contract.required_science_keys)
    current = _science_key_losses(after, contract.required_science_keys)
    changes = {name: _relative_change(initial[name], current[name]) for name in initial}
    critical = tuple(
        changes[f"{horizon}.{name}"] for horizon in ("7", "30") for name in contract.critical_science_keys
    )
    supporting = tuple(
        changes[f"{horizon}.{name}"] for horizon in ("7", "30") for name in contract.supporting_science_keys
    )
    initial_macro = float(np.mean(np.asarray(tuple(initial.values()), dtype=np.float64)))
    current_macro = float(np.mean(np.asarray(tuple(current.values()), dtype=np.float64)))
    macro_change = _relative_change(initial_macro, current_macro)
    checks = {
        "required_science_macro_strictly_decreases": macro_change < 0.0,
        "critical_key_horizons_within_cap": max(critical)
        <= float(spec["critical_key_horizon_relative_regression_maximum"]),
        "supporting_key_horizons_within_cap": max(supporting)
        <= float(spec["supporting_key_horizon_relative_regression_maximum"]),
        "physical_hard_gate": after.get("hard_gate_passed") is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "initial_macro": initial_macro,
        "current_macro": current_macro,
        "macro_relative_change": macro_change,
        "key_horizon_relative_changes": changes,
        "maximum_critical_relative_change": max(critical),
        "maximum_supporting_relative_change": max(supporting),
    }


def calibrate_full_step_l2(
    contract: ConditionedAttentionDiscardedUpdateContract,
    *,
    losses: Mapping[str, float],
    gradient_l2_norms: Mapping[str, float],
    direction_cosines: Mapping[str, float],
    trainable_l2_norm: float,
) -> Mapping[str, Any]:
    labels = contract.source_registry
    if set(losses) != set(labels) or set(gradient_l2_norms) != set(labels) or set(direction_cosines) != set(labels):
        raise ValueError("conditioned discarded-update calibration registry drift")
    spec = contract.payload["step_calibration"]
    floor = float(spec["loss_floor"])
    slopes: dict[str, float] = {}
    for name in labels:
        loss = float(losses[name])
        norm = float(gradient_l2_norms[name])
        cosine = float(direction_cosines[name])
        if (
            not np.isfinite(loss)
            or loss < 0.0
            or not np.isfinite(norm)
            or norm <= 0.0
            or not np.isfinite(cosine)
            or cosine <= 0.0
        ):
            raise ValueError("conditioned discarded-update calibration value is invalid")
        slopes[name] = norm * cosine / max(loss, floor)
    maximum_slope = max(slopes.values())
    trainable_norm = float(trainable_l2_norm)
    if not np.isfinite(trainable_norm) or trainable_norm <= 0.0 or maximum_slope <= 0.0:
        raise ValueError("conditioned discarded-update calibration norm is invalid")
    response_bound = float(spec["target_maximum_linearized_relative_reduction"]) / maximum_slope
    absolute_cap = float(spec["absolute_parameter_delta_l2_cap"])
    relative_cap = float(spec["relative_trainable_parameter_l2_cap"]) * trainable_norm
    full_step = min(response_bound, absolute_cap, relative_cap)
    usable = bool(np.isfinite(full_step) and full_step >= float(spec["minimum_usable_parameter_delta_l2"]))
    limiting_value = min((response_bound, "linearized_response"), (absolute_cap, "absolute_l2"), (relative_cap, "relative_l2"))[1]
    return {
        "usable": usable,
        "full_parameter_delta_l2": full_step,
        "trainable_parameter_l2": trainable_norm,
        "maximum_constraint_relative_slope": maximum_slope,
        "constraint_relative_slopes": slopes,
        "caps": {
            "linearized_response": response_bound,
            "absolute_l2": absolute_cap,
            "relative_l2": relative_cap,
        },
        "limiting_cap": limiting_value,
        "maximum_predicted_relative_reduction": full_step * maximum_slope,
    }


def monitor_transfer_gate(
    contract: ConditionedAttentionDiscardedUpdateContract,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Mapping[str, Any]:
    spec = contract.payload["fixed_monitor"]
    before_records = tuple(before.get("records", ()))
    after_records = tuple(after.get("records", ()))
    before_updates = tuple(int(value.get("artifact_update", -1)) for value in before_records)
    after_updates = tuple(int(value.get("artifact_update", -1)) for value in after_records)
    if before_updates != contract.monitor_updates or after_updates != contract.monitor_updates:
        raise ValueError("conditioned discarded-update monitor artifact order drift")
    artifact_changes = tuple(
        _relative_change(first["source_losses"][DIRECT_SOURCE], second["source_losses"][DIRECT_SOURCE])
        for first, second in zip(before_records, after_records, strict=True)
    )
    direct_change = _relative_change(
        before["source_mean_losses"][DIRECT_SOURCE],
        after["source_mean_losses"][DIRECT_SOURCE],
    )
    tolerance = float(spec["artifact_direct_gpp_relative_regression_tolerance"])
    nonregressing = sum(value <= tolerance for value in artifact_changes)
    science = science_progress_gate(contract, before, after, gate_name="fixed_monitor")
    absolute = absolute_science_gate(contract, after)
    checks = {
        "direct_gpp_mean_strictly_decreases": direct_change < 0.0,
        "minimum_nonregressing_direct_gpp_artifacts": nonregressing
        >= int(spec["minimum_nonregressing_direct_gpp_artifact_count"]),
        "science_progress": science["passed"],
        "physical_hard_gate": after.get("hard_gate_passed") is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "direct_gpp_mean_relative_change": direct_change,
        "artifact_direct_gpp_relative_changes": list(artifact_changes),
        "nonregressing_direct_gpp_artifact_count": nonregressing,
        "science_progress": science,
        "absolute_science": absolute,
    }


def classify_discarded_update(
    *,
    common_first_order_direction: bool,
    calibration_usable: bool,
    selected_scale: float | None,
    monitor_passed: bool | None,
    absolute_science_passed: bool | None,
) -> str:
    if not common_first_order_direction:
        if calibration_usable or selected_scale is not None or monitor_passed is not None:
            raise ValueError("missing conditioned direction cannot have downstream results")
        return "no_conditioned_common_direction"
    if not calibration_usable:
        if selected_scale is not None or monitor_passed is not None:
            raise ValueError("unusable conditioned calibration cannot have a selected step")
        return "conditioned_step_calibration_unusable"
    if selected_scale is None:
        if monitor_passed is not None:
            raise ValueError("missing conditioned train-only step cannot have monitor results")
        return "no_conditioned_train_only_finite_step"
    if monitor_passed is not True:
        if absolute_science_passed is not None:
            raise ValueError("failed conditioned monitor cannot claim an absolute result")
        return "conditioned_discarded_update_fails_monitor"
    if absolute_science_passed is None:
        raise ValueError("passing conditioned monitor requires an absolute science verdict")
    return (
        "conditioned_discarded_update_meets_absolute_science_gate"
        if absolute_science_passed
        else "conditioned_discarded_update_transfers_but_absolute_science_remains_open"
    )


def load_conditioned_attention_discarded_update_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> ConditionedAttentionDiscardedUpdateContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    starting = payload.get("starting_state", {})
    direction = payload.get("direction", {})
    calibration = payload.get("step_calibration", {})
    train_selection = payload.get("train_only_selection", {})
    absolute = payload.get("absolute_science_gate", {})
    monitor = payload.get("fixed_monitor", {})
    retention = payload.get("retention", {})
    execution = payload.get("execution", {})
    schedule = build_persistent_scale_training_schedule()
    expected_groups = tuple(schedule.groups[index - 1] for index in PANEL_GROUP_UPDATES)
    observed_groups = tuple(tuple(int(value) for value in group) for group in data.get("train_only_artifact_groups", ()))
    artifacts = tuple(value for group in observed_groups for value in group)
    required_keys = tuple(str(value) for value in payload.get("baseline", {}).get("required_science_keys", ()))
    critical_keys = tuple(str(value) for value in payload.get("baseline", {}).get("critical_long_horizon_keys", ()))
    supporting_keys = tuple(str(value) for value in payload.get("baseline", {}).get("supporting_science_keys", ()))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_conditioned_update3_gradient_or_monitor_values_are_evaluated"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("conditioned_attention_contract_sha256") != CONDITIONED_ATTENTION_CONTRACT_SHA256
        or identity.get("conditioned_attention_admission_report_sha256") != CONDITIONED_ADMISSION_REPORT_SHA256
        or identity.get("artifact_aware_persistent_contract_sha256") != ARTIFACT_PERSISTENT_CONTRACT_SHA256
        or identity.get("artifact_aware_persistent_schedule_sha256") != SCHEDULE_SHA256
        or identity.get("artifact_aware_persistent_report_sha256") != PARENT_REPORT_SHA256
        or identity.get("artifact_aware_update2_recovery_sha256") != PARENT_RECOVERY_SHA256
        or int(identity.get("parent_trainable_parameter_count", -1)) != PARENT_TRAINABLE_PARAMETER_COUNT
        or int(identity.get("candidate_parameter_count", -1)) != TOTAL_PARAMETER_COUNT
        or int(identity.get("candidate_trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(identity.get("retained_parameter_count", -1)) != RETAINED_PARAMETER_COUNT
        or schedule.sha256 != SCHEDULE_SHA256
        or int(data.get("logical_update", -1)) != LOGICAL_UPDATE
        or int(data.get("scheduled_group_update", -1)) != PANEL_GROUP_UPDATES[0]
        or tuple(int(value) for value in data.get("guard_group_updates", ())) != PANEL_GROUP_UPDATES[1:]
        or observed_groups != expected_groups
        or len(artifacts) != 40
        or len(set(artifacts)) != 40
        or int(data.get("train_only_artifact_count", -1)) != 40
        or int(data.get("teacher_days_per_direct_gpp_artifact", -1)) != 208
        or tuple(int(value) for value in data.get("fixed_monitor_artifact_updates", ())) != MONITOR_UPDATES
        or not set(artifacts).isdisjoint(MONITOR_UPDATES)
        or data.get("fixed_monitor_disjoint_from_train_only_panel") is not True
        or int(starting.get("required_completed_logical_update", -1)) != 2
        or starting.get("accepted_update2_retained") is not True
        or starting.get("rejected_update3_applied") is not False
        or starting.get("earlier_bootstrap_state_used_as_candidate_start") is not False
        or int(direction.get("gradient_constraint_count", -1)) != 43
        or int(direction.get("direct_gpp_artifact_constraint_count", -1)) != 40
        or int(direction.get("scheduled_non_gpp_constraint_count", -1)) != 3
        or direction.get("each_gradient_unit_normalized") is not True
        or float(direction.get("minimum_constraint_cosine", -1.0)) != 1.0e-4
        or direction.get("optimizer_or_momentum_state_used") is not False
        or int(direction.get("temperature_gradient_coordinates_included", -1)) != 16
        or set(required_keys) != set(critical_keys) | set(supporting_keys)
        or set(critical_keys) & set(supporting_keys)
        or tuple(absolute.get("required_keys", ())) != required_keys
        or float(absolute.get("seven_day_normalized_rmse_maximum", -1.0)) != 0.35
        or float(absolute.get("seven_day_absolute_normalized_signed_bias_maximum", -1.0)) != 0.1
        or float(absolute.get("thirty_day_normalized_rmse_maximum", -1.0)) != 0.25
        or float(absolute.get("thirty_day_absolute_normalized_signed_bias_maximum", -1.0)) != 0.07
        or absolute.get("required_for_single_discarded_step_selection") is not False
        or tuple(float(value) for value in calibration.get("scale_multipliers_descending", ()))
        != (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
        or calibration.get("old_parent_full_step_l2_inherited") is not False
        or calibration.get("old_parent_scale_ladder_inherited") is not False
        or float(calibration.get("target_maximum_linearized_relative_reduction", -1.0)) != 0.02
        or float(calibration.get("absolute_parameter_delta_l2_cap", -1.0)) != 0.01
        or float(calibration.get("relative_trainable_parameter_l2_cap", -1.0)) != 1.0e-5
        or float(calibration.get("minimum_usable_parameter_delta_l2", -1.0)) != 1.0e-8
        or float(train_selection.get("each_artifact_direct_gpp_relative_regression_maximum", -1.0)) != 1.0e-6
        or train_selection.get("monitor_values_excluded_from_direction_calibration_and_scale_selection") is not True
        or monitor.get("read_only_after_train_only_scale_is_selected") is not True
        or monitor.get("each_artifact_loaded_at_most_once") is not True
        or monitor.get("initial_and_candidate_evaluated_as_a_pair") is not True
        or monitor.get("monitor_never_selects_direction_or_scale") is not True
    ):
        raise ValueError("conditioned discarded-update contract drift")
    forbidden_data = (
        "fresh_validation_values_allowed",
        "teacher_data_regeneration_allowed",
        "historical_v5_allowed",
        "sealed_test_allowed",
    )
    forbidden_execution = (
        "paid_partition_authorized",
        "fresh_validation_read_authorized",
        "teacher_data_regeneration_authorized",
        "historical_v5_authorized",
        "365_day_rollout_authorized",
        "gate_d2_authorized",
        "sealed_test_authorized",
        "product_acceptance_authorized",
    )
    if (
        any(bool(data[name]) for name in forbidden_data)
        or any(bool(value) for value in retention.values())
        or execution.get("free_gln01_allowed") is not True
        or int(execution.get("timeout_minutes", -1)) != 180
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
        or any(bool(execution[name]) for name in forbidden_execution)
    ):
        raise ValueError("conditioned discarded-update execution boundary drift")
    return ConditionedAttentionDiscardedUpdateContract(payload, source, observed)


__all__ = [
    "CONDITIONED_ADMISSION_REPORT_SHA256",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "LOGICAL_UPDATE",
    "PANEL_GROUP_UPDATES",
    "PARENT_RECOVERY_SHA256",
    "PARENT_REPORT_SHA256",
    "PARENT_TRAINABLE_PARAMETER_COUNT",
    "RETAINED_PARAMETER_COUNT",
    "SCHEMA_VERSION",
    "TOTAL_PARAMETER_COUNT",
    "TRAINABLE_PARAMETER_COUNT",
    "ConditionedAttentionDiscardedUpdateContract",
    "absolute_science_gate",
    "calibrate_full_step_l2",
    "classify_discarded_update",
    "load_conditioned_attention_discarded_update_contract",
    "monitor_transfer_gate",
    "science_progress_gate",
]
