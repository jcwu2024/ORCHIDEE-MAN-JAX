"""Frozen contract for the persisted maintenance-state recurrence audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_flux_basis import contract_sha256
from .target_oriented_snow_thermal_alias_recurrence import (
    SNOW_THERMAL_ALIAS_STATE_KEYS,
)
from .target_oriented_snowheat_recurrence import resolve_continuous_state_leaves

SCHEMA_VERSION = "target_oriented_maintenance_state_recurrence_intervention_v1"
FROZEN_CONTRACT_SHA256 = "cf4f11670df3fe3dc279802d24c23a5f8ebf6e6744ba47dac32de419cfa4e839"
RESP_MAINT_STATE_KEY = "slowproc_stomate_previous_step_state.resp_maint"
RESP_MAINT_PART_STATE_KEY = "slowproc_stomate_previous_step_state.resp_maint_part"
MAINTENANCE_HEAD_FIELD = "daily.fast_maintenance_respiration"
INTERVENTION_STATE_KEYS = (*SNOW_THERMAL_ALIAS_STATE_KEYS, RESP_MAINT_STATE_KEY)
WINDOW_NAMES = ("thirty_day_extreme", "thirty_day_matched_control")
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_maintenance_state_recurrence_intervention_v1.json"
)


@dataclass(frozen=True)
class TargetOrientedMaintenanceStateRecurrenceContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def windows(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["data"]["windows"])


def _validate(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_zero_training_execution"
    ):
        raise ValueError("maintenance-state recurrence schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("snow_thermal_alias_parent_report_sha256")
        != "5714d30db9a905498a1e31203f7d6a622417fc165328ac47bf93e077e4422168"
        or identity.get("snow_thermal_alias_parent_contract_sha256")
        != "5ab495baf5c140d71e841defaa6e799d8c6419717944e6aa38ab37b5ecd77f63"
        or identity.get("fast_owner_reconstruction_contract_sha256")
        != "4db5f828442212c58903096c36015f9d769c19aeaaa4eb9a4326fbea2fdfc4c9"
        or identity.get("update_1000_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
        or identity.get("prepared_manifest_sha256")
        != "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
        or int(identity.get("parameter_count", 0)) != 83_755_173
    ):
        raise ValueError("maintenance-state recurrence identity drift")
    data = payload.get("data", {})
    windows = tuple(data.get("windows", ()))
    if (
        data.get("release_id") != "pft14-daily-teacher-v6"
        or data.get("split") != "train/train"
        or tuple(record.get("name") for record in windows) != WINDOW_NAMES
        or any(int(record.get("horizon", 0)) != 30 for record in windows)
        or any(
            bool(data.get(name))
            for name in (
                "teacher_regeneration_allowed",
                "historical_v5_allowed",
                "validation_artifact_values_allowed",
                "sealed_test_allowed",
            )
        )
    ):
        raise ValueError("maintenance-state recurrence data boundary drift")
    intervention = payload.get("intervention", {})
    if (
        tuple(intervention.get("parent_state_keys", ()))
        != SNOW_THERMAL_ALIAS_STATE_KEYS
        or intervention.get("added_state_key") != RESP_MAINT_STATE_KEY
        or intervention.get("excluded_reset_accumulator")
        != RESP_MAINT_PART_STATE_KEY
        or intervention.get("diagnostic_only") is not True
        or intervention.get("allowed_in_training") is not False
        or intervention.get("allowed_in_final_inference") is not False
        or any(
            intervention.get(name) is not True
            for name in (
                "other_continuous_state_coordinates_remain_predicted",
                "discrete_state_remains_predicted",
                "forcing_unchanged",
                "parameters_unchanged",
                "operator_unchanged",
                "retained_exact_tail_unchanged",
            )
        )
    ):
        raise ValueError("maintenance-state recurrence intervention drift")
    measurement = payload.get("measurement", {})
    if (
        measurement.get("field") != "maintenance_respiration"
        or measurement.get("learned_head") != MAINTENANCE_HEAD_FIELD
        or measurement.get("mode")
        != "snow_thermal_alias_plus_resp_maint_teacher_carry"
        or int(measurement.get("gradient_count", 0)) != 2
        or int(measurement.get("gradient_coverage", 0)) != 83_755_173
    ):
        raise ValueError("maintenance-state recurrence measurement drift")
    execution = payload.get("execution", {})
    if (
        execution.get("free_gln01_only") is not True
        or execution.get("paid_training_authorized") is not False
        or int(execution.get("optimizer_updates", -1)) != 0
        or int(execution.get("parameter_steps", -1)) != 0
        or any(
            bool(execution.get(name))
            for name in (
                "proposed_parameters_retained",
                "optimizer_state_retained",
                "checkpoint_written",
                "recovery_written",
                "teacher_data_regenerated",
                "historical_v5_used",
                "validation_values_evaluated",
                "sealed_test_used",
                "input_mask_edit_allowed",
                "source_routing_edit_allowed",
                "architecture_edit_allowed",
                "365_day_rollout_allowed",
                "gate_d2_allowed",
            )
        )
    ):
        raise ValueError("maintenance-state recurrence execution drift")


def load_target_oriented_maintenance_state_recurrence_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedMaintenanceStateRecurrenceContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("maintenance-state recurrence contract is not frozen v1")
    return TargetOrientedMaintenanceStateRecurrenceContract(payload, source, observed)


def resolve_maintenance_intervention_state_leaves(markov_contract: Any):
    """Resolve the four accepted snow aliases plus persisted resp_maint."""

    leaves = resolve_continuous_state_leaves(markov_contract, INTERVENTION_STATE_KEYS)
    by_key = {leaf.key: leaf for leaf in leaves}
    resp_maint = by_key[RESP_MAINT_STATE_KEY]
    if resp_maint.classification != "prognostic" or resp_maint.path != ("resp_maint",):
        raise ValueError("persisted resp_maint Markov semantics drift")
    if RESP_MAINT_PART_STATE_KEY in by_key:
        raise ValueError("reset resp_maint_part accumulator entered the intervention")
    return leaves


def classify_maintenance_state_recurrence(
    records: Mapping[str, Mapping[str, Any]],
    contract: TargetOrientedMaintenanceStateRecurrenceContract,
) -> Mapping[str, Any]:
    if set(records) != set(WINDOW_NAMES):
        raise ValueError("maintenance-state result window registry drift")
    policy = contract.payload["classification"]
    extreme = records["thirty_day_extreme"]
    gradient_reduction = float(extreme["parent_to_treatment_gradient_reduction_ratio"])
    head_reduction = float(extreme["parent_to_treatment_maintenance_head_reduction_ratio"])
    confirmed = gradient_reduction >= float(
        policy["extreme_gradient_reduction_ratio_for_confirmation"]
    ) and head_reduction >= float(
        policy["maintenance_head_reduction_ratio_for_confirmation"]
    )
    no_effect = gradient_reduction < float(
        policy["gradient_reduction_ratio_for_no_effect"]
    ) and head_reduction < float(
        policy["maintenance_head_reduction_ratio_for_no_effect"]
    )
    if confirmed:
        classification = "maintenance_state_recurrence_confirmed"
    elif no_effect:
        classification = "maintenance_state_recurrence_not_explanatory"
    else:
        classification = "maintenance_state_recurrence_mixed"
    return {
        "extreme_window": {
            "parent_to_treatment_gradient_reduction_ratio": gradient_reduction,
            "parent_to_treatment_maintenance_head_reduction_ratio": head_reduction,
            "confirmation_threshold_passed": confirmed,
            "no_effect_threshold_met": no_effect,
        },
        "classification": classification,
        "action": str(contract.payload["decision"][classification]),
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "INTERVENTION_STATE_KEYS",
    "MAINTENANCE_HEAD_FIELD",
    "RESP_MAINT_PART_STATE_KEY",
    "RESP_MAINT_STATE_KEY",
    "SCHEMA_VERSION",
    "TargetOrientedMaintenanceStateRecurrenceContract",
    "WINDOW_NAMES",
    "classify_maintenance_state_recurrence",
    "load_target_oriented_maintenance_state_recurrence_contract",
    "resolve_maintenance_intervention_state_leaves",
]
