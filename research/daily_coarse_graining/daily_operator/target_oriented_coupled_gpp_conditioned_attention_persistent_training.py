"""Frozen bounded persistent-training smoke for conditioned GPP attention."""

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
from .target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    FROZEN_CONTRACT_SHA256 as DISCARDED_UPDATE_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    PANEL_GROUP_UPDATES as DISCARDED_PANEL_GROUP_UPDATES,
)
from .target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    PARENT_RECOVERY_SHA256,
    PARENT_REPORT_SHA256,
    RETAINED_PARAMETER_COUNT,
    TOTAL_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
)
from .target_oriented_coupled_gpp_direct_generalization import MONITOR_UPDATES
from .target_oriented_coupled_gpp_objective import DIRECT_SOURCE
from .target_oriented_coupled_gpp_persistent_scale_training import (
    SCHEDULE_SHA256,
    build_persistent_scale_training_schedule,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1"
FROZEN_CONTRACT_SHA256 = "ac65c5be2b4202edad81ad8ad371e47ec8cf39443c4e4eedd2e80ab02e5c9aff"
DISCARDED_UPDATE_RESULT_SHA256 = "0aa30a0eb38f4e1f4da8d0a155a325cae35271b2cca60c04bf104c7e9fe9b42a"
LOGICAL_UPDATES = (3, 4, 5)
START_COMPLETED_LOGICAL_UPDATE = 2
FINAL_LOGICAL_UPDATE = LOGICAL_UPDATES[-1]
TRAIN_PANEL_GROUP_UPDATES = (
    (3, 15, 16, 17, 18),
    (4, 19, 20, 21, 22),
    (5, 23, 24, 25, 26),
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1.json"
)


def _artifact_sources(groups: Sequence[Sequence[int]]) -> tuple[str, ...]:
    return tuple(f"{DIRECT_SOURCE}.artifact_{int(update)}" for group in groups for update in group)


@dataclass(frozen=True)
class ConditionedAttentionPersistentTrainingContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def logical_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["logical_updates"])

    @property
    def panel_group_updates(self) -> tuple[tuple[int, ...], ...]:
        configured = self.payload["data"]["train_only_schedule_group_updates_by_logical_update"]
        return tuple(tuple(int(value) for value in configured[str(update)]) for update in self.logical_updates)

    @property
    def train_panels(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        schedule = build_persistent_scale_training_schedule()
        return tuple(
            tuple(tuple(int(value) for value in schedule.groups[index - 1]) for index in group_updates)
            for group_updates in self.panel_group_updates
        )

    @property
    def source_registries(self) -> tuple[tuple[str, ...], ...]:
        return tuple(_artifact_sources(panel) + SCHEDULED_NON_GPP_SOURCES for panel in self.train_panels)

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["fixed_monitor"]["artifact_updates"])

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


def calibrate_persistent_step_l2(
    contract: ConditionedAttentionPersistentTrainingContract,
    *,
    source_registry: Sequence[str],
    losses: Mapping[str, float],
    gradient_l2_norms: Mapping[str, float],
    direction_cosines: Mapping[str, float],
    trainable_l2_norm: float,
) -> Mapping[str, Any]:
    labels = tuple(source_registry)
    if set(losses) != set(labels) or set(gradient_l2_norms) != set(labels) or set(direction_cosines) != set(labels):
        raise ValueError("conditioned persistent calibration registry drift")
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
            raise ValueError("conditioned persistent calibration value is invalid")
        slopes[name] = norm * cosine / max(loss, floor)
    maximum_slope = max(slopes.values())
    trainable_norm = float(trainable_l2_norm)
    if not np.isfinite(trainable_norm) or trainable_norm <= 0.0 or maximum_slope <= 0.0:
        raise ValueError("conditioned persistent calibration norm is invalid")
    response_bound = float(spec["target_maximum_linearized_relative_reduction"]) / maximum_slope
    absolute_cap = float(spec["absolute_parameter_delta_l2_cap"])
    relative_cap = float(spec["relative_trainable_parameter_l2_cap"]) * trainable_norm
    full_step = min(response_bound, absolute_cap, relative_cap)
    usable = bool(np.isfinite(full_step) and full_step >= float(spec["minimum_usable_parameter_delta_l2"]))
    limiting_cap = min(
        (response_bound, "linearized_response"),
        (absolute_cap, "absolute_l2"),
        (relative_cap, "relative_l2"),
    )[1]
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
        "limiting_cap": limiting_cap,
        "maximum_predicted_relative_reduction": full_step * maximum_slope,
    }


def classify_persistent_smoke(
    *,
    completed_logical_update: int,
    stopped_reason: str | None,
    final_absolute_science_passed: bool | None,
) -> str:
    completed = int(completed_logical_update)
    if completed == FINAL_LOGICAL_UPDATE:
        if stopped_reason is not None or final_absolute_science_passed is None:
            raise ValueError("completed conditioned smoke requires a final absolute verdict")
        return (
            "conditioned_attention_three_update_smoke_meets_absolute_science_gate"
            if final_absolute_science_passed
            else "conditioned_attention_three_update_smoke_transfers_absolute_open"
        )
    if final_absolute_science_passed is not None:
        raise ValueError("incomplete conditioned smoke cannot have a final absolute verdict")
    if stopped_reason == "train_only":
        return "conditioned_attention_persistent_smoke_stopped_train_only"
    if stopped_reason == "monitor_transfer":
        return "conditioned_attention_persistent_smoke_stopped_monitor_transfer"
    raise ValueError("incomplete conditioned smoke requires a known stop reason")


def load_conditioned_attention_persistent_training_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> ConditionedAttentionPersistentTrainingContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    starting = payload.get("starting_state", {})
    baseline = payload.get("baseline", {})
    direction = payload.get("direction", {})
    calibration = payload.get("step_calibration", {})
    selection = payload.get("train_only_selection", {})
    absolute = payload.get("absolute_science_gate", {})
    monitor = payload.get("fixed_monitor", {})
    persistence = payload.get("persistence", {})
    execution = payload.get("execution", {})
    schedule = build_persistent_scale_training_schedule()
    configured = data.get("train_only_schedule_group_updates_by_logical_update", {})
    group_updates = tuple(tuple(int(value) for value in configured.get(str(update), ())) for update in LOGICAL_UPDATES)
    panels = tuple(
        tuple(tuple(int(value) for value in schedule.groups[index - 1]) for index in indexes)
        for indexes in group_updates
    )
    flattened = tuple(tuple(value for group in panel for value in group) for panel in panels)
    artifacts = tuple(value for panel in flattened for value in panel)
    monitor_updates = tuple(int(value) for value in monitor.get("artifact_updates", ()))
    required = tuple(str(value) for value in baseline.get("required_science_keys", ()))
    critical = tuple(str(value) for value in baseline.get("critical_long_horizon_keys", ()))
    supporting = tuple(str(value) for value in baseline.get("supporting_science_keys", ()))
    expected_scales = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_conditioned_persistent_training_values_are_evaluated"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("conditioned_attention_contract_sha256") != CONDITIONED_ATTENTION_CONTRACT_SHA256
        or identity.get("discarded_update_contract_sha256") != DISCARDED_UPDATE_CONTRACT_SHA256
        or identity.get("discarded_update_result_sha256") != DISCARDED_UPDATE_RESULT_SHA256
        or identity.get("artifact_aware_persistent_contract_sha256") != ARTIFACT_PERSISTENT_CONTRACT_SHA256
        or identity.get("artifact_aware_persistent_schedule_sha256") != SCHEDULE_SHA256
        or identity.get("artifact_aware_persistent_report_sha256") != PARENT_REPORT_SHA256
        or identity.get("artifact_aware_update2_recovery_sha256") != PARENT_RECOVERY_SHA256
        or int(identity.get("candidate_parameter_count", -1)) != TOTAL_PARAMETER_COUNT
        or int(identity.get("candidate_trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(identity.get("retained_parameter_count", -1)) != RETAINED_PARAMETER_COUNT
        or schedule.sha256 != SCHEDULE_SHA256
        or tuple(int(value) for value in data.get("logical_updates", ())) != LOGICAL_UPDATES
        or group_updates != TRAIN_PANEL_GROUP_UPDATES
        or TRAIN_PANEL_GROUP_UPDATES[0] != DISCARDED_PANEL_GROUP_UPDATES
        or any(len(panel) != 5 for panel in panels)
        or any(len(values) != 40 or len(set(values)) != 40 for values in flattened)
        or len(artifacts) != 120
        or len(set(artifacts)) != 120
        or not set(artifacts).isdisjoint(monitor_updates)
        or int(data.get("train_only_artifact_count_per_logical_update", -1)) != 40
        or int(data.get("unique_train_only_artifact_count", -1)) != 120
        or int(data.get("teacher_days_per_direct_gpp_artifact", -1)) != 208
        or data.get("scheduled_group_is_first_group_for_each_logical_update") is not True
        or data.get("artifact_panels_overlap_across_logical_updates") is not False
        or data.get("fixed_monitor_disjoint_from_all_train_only_panels") is not True
        or int(starting.get("required_completed_logical_update", -1)) != START_COMPLETED_LOGICAL_UPDATE
        or starting.get("accepted_update2_retained") is not True
        or starting.get("rejected_parent_update3_applied") is not False
        or starting.get("discarded_conditioned_update3_applied") is not False
        or starting.get("exact_conditioned_parameter_start_snapshot_written_before_training") is not True
        or starting.get("start_snapshot_roundtrip_required") is not True
        or starting.get("resume_loads_latest_identity_exact_recovery") is not True
        or starting.get("optimizer_state_loaded") is not False
        or baseline.get("initial_fixed_monitor_written_before_first_gradient") is not True
        or baseline.get("scheduled_science_panel_reported_before_each_update_gradient") is not True
        or set(required) != set(critical) | set(supporting)
        or set(critical) & set(supporting)
        or tuple(absolute.get("required_keys", ())) != required
        or float(absolute.get("seven_day_normalized_rmse_maximum", -1.0)) != 0.35
        or float(absolute.get("seven_day_absolute_normalized_signed_bias_maximum", -1.0)) != 0.1
        or float(absolute.get("thirty_day_normalized_rmse_maximum", -1.0)) != 0.25
        or float(absolute.get("thirty_day_absolute_normalized_signed_bias_maximum", -1.0)) != 0.07
        or absolute.get("required_for_smoke_update_retention") is not False
        or int(direction.get("gradient_constraint_count_per_logical_update", -1)) != 43
        or int(direction.get("direct_gpp_artifact_constraint_count_per_logical_update", -1)) != 40
        or int(direction.get("scheduled_non_gpp_constraint_count_per_logical_update", -1)) != 3
        or direction.get("direction_recomputed_after_each_accepted_update") is not True
        or direction.get("previous_direction_or_simplex_weights_reused") is not False
        or direction.get("each_gradient_unit_normalized") is not True
        or float(direction.get("minimum_constraint_cosine", -1.0)) != 1.0e-4
        or direction.get("optimizer_or_momentum_state_used") is not False
        or int(direction.get("temperature_gradient_coordinates_included", -1)) != 16
        or calibration.get("recomputed_after_each_accepted_update") is not True
        or tuple(float(value) for value in calibration.get("scale_multipliers_descending", ())) != expected_scales
        or calibration.get("old_parent_full_step_l2_inherited") is not False
        or calibration.get("previous_conditioned_full_step_l2_inherited") is not False
        or float(calibration.get("target_maximum_linearized_relative_reduction", -1.0)) != 0.02
        or float(calibration.get("absolute_parameter_delta_l2_cap", -1.0)) != 0.01
        or float(calibration.get("relative_trainable_parameter_l2_cap", -1.0)) != 1.0e-5
        or float(calibration.get("minimum_usable_parameter_delta_l2", -1.0)) != 1.0e-8
        or float(selection.get("each_artifact_direct_gpp_relative_regression_maximum", -1.0)) != 1.0e-6
        or selection.get("stop_training_after_first_update_without_passing_scale") is not True
        or selection.get("monitor_values_excluded_from_direction_calibration_and_scale_selection") is not True
        or monitor_updates != MONITOR_UPDATES
        or monitor.get("evaluated_after_each_train_only_scale_is_selected") is not True
        or monitor.get("previous_accepted_monitor_is_step_local_baseline") is not True
        or monitor.get("initial_monitor_is_cumulative_baseline") is not True
        or monitor.get("both_step_local_and_cumulative_gates_required_for_retention") is not True
        or monitor.get("failed_monitor_candidate_discarded") is not True
        or monitor.get("monitor_never_selects_direction_or_scale") is not True
        or persistence.get("start_snapshot_immutable") is not True
        or persistence.get("recovery_written_after_initial_monitor") is not True
        or persistence.get("recovery_written_after_each_monitor_accepted_update") is not True
        or persistence.get("identity_exact_resume_required") is not True
        or persistence.get("atomic_write_required") is not True
        or persistence.get("trainable_parameter_tree_persisted") is not True
        or persistence.get("fixed_parameters_persisted") is not False
        or persistence.get("optimizer_state_persisted") is not False
        or persistence.get("checkpoint_only_after_all_three_updates_pass") is not True
        or persistence.get("checkpoint_promotion_eligible") is not False
        or persistence.get("recovery_promotion_eligible") is not False
    ):
        raise ValueError("conditioned attention persistent-training contract drift")
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
        or execution.get("free_gln01_training_smoke_allowed") is not True
        or int(execution.get("timeout_minutes", -1)) != 240
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
        or any(bool(execution[name]) for name in forbidden_execution)
    ):
        raise ValueError("conditioned attention persistent execution boundary drift")
    return ConditionedAttentionPersistentTrainingContract(payload, source, observed)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "DISCARDED_UPDATE_RESULT_SHA256",
    "FINAL_LOGICAL_UPDATE",
    "FROZEN_CONTRACT_SHA256",
    "LOGICAL_UPDATES",
    "SCHEMA_VERSION",
    "START_COMPLETED_LOGICAL_UPDATE",
    "TRAIN_PANEL_GROUP_UPDATES",
    "ConditionedAttentionPersistentTrainingContract",
    "calibrate_persistent_step_l2",
    "classify_persistent_smoke",
    "load_conditioned_attention_persistent_training_contract",
]
