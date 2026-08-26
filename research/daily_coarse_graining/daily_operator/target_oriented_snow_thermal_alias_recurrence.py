"""Frozen contract and owner checks for mirror-complete snow recurrence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_flux_basis import contract_sha256
from .target_oriented_snowheat_recurrence import resolve_continuous_state_leaves

SCHEMA_VERSION = "target_oriented_snow_thermal_alias_recurrence_intervention_v1"
FROZEN_CONTRACT_SHA256 = "5ab495baf5c140d71e841defaa6e799d8c6419717944e6aa38ab37b5ecd77f63"
SNOW_THERMAL_ALIAS_GROUPS = (
    (
        "hydrol_previous_step_state.snowheat",
        ("diffuco_previous_step_state.snowheat",),
    ),
    (
        "hydrol_previous_step_state.snowtemp",
        ("diffuco_previous_step_state.snowtemp",),
    ),
)
SNOW_THERMAL_ALIAS_STATE_KEYS = tuple(
    key
    for canonical, mirrors in SNOW_THERMAL_ALIAS_GROUPS
    for key in (canonical, *mirrors)
)
SNOW_THERMAL_TENDENCY_FIELDS = (
    "thermal.hydrol_previous_step_state.snowheat.normalized_tendency",
    "thermal.hydrol_previous_step_state.snowtemp.normalized_tendency",
)
WINDOW_NAMES = ("thirty_day_extreme", "thirty_day_matched_control")
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_snow_thermal_alias_recurrence_intervention_v1.json"
)


@dataclass(frozen=True)
class TargetOrientedSnowThermalAliasRecurrenceContract:
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
        raise ValueError("snow thermal alias recurrence schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("snow_thermal_parent_report_sha256")
        != "c4b80543a3a726819af3d6e658603484b7aa233cb485333e689af47cfb303810"
        or identity.get("snow_thermal_parent_contract_sha256")
        != "959413d639a6bd853b15ca2fc896beea4ea5e690463f4a78e8c3cac1ccaf96ca"
        or identity.get("fast_owner_reconstruction_contract_sha256")
        != "4db5f828442212c58903096c36015f9d769c19aeaaa4eb9a4326fbea2fdfc4c9"
        or identity.get("update_1000_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
        or identity.get("prepared_manifest_sha256")
        != "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
        or int(identity.get("parameter_count", 0)) != 83_755_173
    ):
        raise ValueError("snow thermal alias recurrence identity drift")
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
        raise ValueError("snow thermal alias recurrence data boundary drift")
    intervention = payload.get("intervention", {})
    declared_groups = tuple(
        (
            str(group.get("canonical_state_key")),
            tuple(str(key) for key in group.get("mirror_state_keys", ())),
        )
        for group in intervention.get("canonical_alias_groups", ())
    )
    if (
        declared_groups != SNOW_THERMAL_ALIAS_GROUPS
        or tuple(intervention.get("state_keys", ())) != SNOW_THERMAL_ALIAS_STATE_KEYS
        or tuple(intervention.get("learned_tendency_fields", ()))
        != SNOW_THERMAL_TENDENCY_FIELDS
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
        raise ValueError("snow thermal alias recurrence intervention drift")
    measurement = payload.get("measurement", {})
    if (
        measurement.get("field") != "maintenance_respiration"
        or measurement.get("mode") != "snow_thermal_alias_complete_teacher_carry"
        or int(measurement.get("gradient_count", 0)) != 2
        or int(measurement.get("gradient_coverage", 0)) != 83_755_173
    ):
        raise ValueError("snow thermal alias recurrence measurement drift")
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
                "architecture_edit_allowed",
                "365_day_rollout_allowed",
                "gate_d2_allowed",
            )
        )
    ):
        raise ValueError("snow thermal alias recurrence execution drift")


def load_target_oriented_snow_thermal_alias_recurrence_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedSnowThermalAliasRecurrenceContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("snow thermal alias recurrence contract is not frozen v1")
    return TargetOrientedSnowThermalAliasRecurrenceContract(payload, source, observed)


def resolve_snow_thermal_alias_state_leaves(markov_contract: Any, owner_contract: Any):
    """Resolve all state slices and verify their declared fast-owner mirrors."""

    leaves = resolve_continuous_state_leaves(
        markov_contract,
        SNOW_THERMAL_ALIAS_STATE_KEYS,
    )
    owner_by_key = {leaf.key: leaf for leaf in owner_contract.leaves}
    for canonical, mirrors in SNOW_THERMAL_ALIAS_GROUPS:
        if canonical not in owner_by_key:
            raise ValueError(f"missing canonical snow thermal owner {canonical}")
        for mirror in mirrors:
            leaf = owner_by_key.get(mirror)
            if (
                leaf is None
                or leaf.role != "mirror_of_canonical_owner"
                or leaf.canonical_owner != canonical
            ):
                raise ValueError(f"snow thermal mirror ownership drift for {mirror}")
    return leaves


def classify_snow_thermal_alias_recurrence(
    records: Mapping[str, Mapping[str, Any]],
    contract: TargetOrientedSnowThermalAliasRecurrenceContract,
) -> Mapping[str, Any]:
    if set(records) != set(WINDOW_NAMES):
        raise ValueError("snow thermal alias result window registry drift")
    policy = contract.payload["classification"]
    extreme = records["thirty_day_extreme"]
    gradient_reduction = float(extreme["parent_to_treatment_gradient_reduction_ratio"])
    snowheat_reduction = float(extreme["parent_to_treatment_snowheat_head_reduction_ratio"])
    confirmed = gradient_reduction >= float(
        policy["extreme_gradient_reduction_ratio_for_alias_confirmation"]
    ) and snowheat_reduction >= float(
        policy["snowheat_head_reduction_ratio_for_alias_confirmation"]
    )
    no_effect = gradient_reduction < float(
        policy["gradient_reduction_ratio_for_no_effect"]
    ) and snowheat_reduction < float(
        policy["snowheat_head_reduction_ratio_for_no_effect"]
    )
    if confirmed:
        classification = "snow_thermal_alias_path_confirmed"
    elif no_effect:
        classification = "snow_thermal_alias_path_not_explanatory"
    else:
        classification = "snow_thermal_alias_path_mixed"
    return {
        "extreme_window": {
            "parent_to_treatment_gradient_reduction_ratio": gradient_reduction,
            "parent_to_treatment_snowheat_head_reduction_ratio": snowheat_reduction,
            "alias_confirmation_threshold_passed": confirmed,
            "no_effect_threshold_met": no_effect,
        },
        "classification": classification,
        "action": str(contract.payload["decision"][classification]),
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "SNOW_THERMAL_ALIAS_GROUPS",
    "SNOW_THERMAL_ALIAS_STATE_KEYS",
    "SNOW_THERMAL_TENDENCY_FIELDS",
    "TargetOrientedSnowThermalAliasRecurrenceContract",
    "WINDOW_NAMES",
    "classify_snow_thermal_alias_recurrence",
    "load_target_oriented_snow_thermal_alias_recurrence_contract",
    "resolve_snow_thermal_alias_state_leaves",
]
