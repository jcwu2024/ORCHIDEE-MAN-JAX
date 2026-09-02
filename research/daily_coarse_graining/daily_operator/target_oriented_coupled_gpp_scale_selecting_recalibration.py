"""Frozen scale-selecting direct recalibration smoke for coupled GPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_coupled_gpp_bounded_training import SCHEDULE_SHA256
from .target_oriented_coupled_gpp_objective import GRADIENT_SOURCES
from .target_oriented_coupled_gpp_update3_common_direction import (
    MINIMUM_SOURCE_COSINE,
    REFERENCE_DELTA_L2,
)
from .target_oriented_coupled_gpp_update5_scale_audit import (
    DISPLACEMENT_SCALES,
    UPDATE_FIVE_ARTIFACTS,
)
from .target_oriented_coupled_gpp_update5_scale_audit import (
    FROZEN_CONTRACT_SHA256 as UPDATE5_SCALE_AUDIT_SHA256,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_v1"
FROZEN_CONTRACT_SHA256 = "3b7109fcccc5a721c96d1ac49eba2fe92a5a7fa0c8386783d562da33fc1bf8a5"
UPDATE5_SCALE_AUDIT_RESULT_SHA256 = "2199a0b7e8c3bad7b8990e9a06e2d13f361c7fec4acf7f37e78bb4befeef5081"
CONTINUED_UPDATES = (6, 7)
CONTINUED_GROUPS = (
    (332, 994, 933, 930, 699, 916, 387, 427),
    (675, 417, 455, 803, 779, 284, 385, 520),
)
EXPECTED_UPDATE5_ACCEPTED_SCALES = (0.25, 0.125, 0.0625, 0.03125)
EXPECTED_UPDATE5_SELECTED_SCALE = 0.25
REQUIRED_CONSECUTIVE_UPDATES = 3
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPScaleSelectingRecalibrationContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(
            float(value) for value in self.payload["scale_selection"]["parameter_displacement_scales_descending"]
        )


def load_coupled_gpp_scale_selecting_recalibration_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPScaleSelectingRecalibrationContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_zero_retention_three_update_scale_selecting_recalibration_smoke"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("coupled GPP scale-selecting recalibration contract is not frozen")
    identity = payload["identity"]
    data = payload["data"]
    direction = payload["direction"]
    selection = payload["scale_selection"]
    evaluation = payload["evaluation"]
    retention = payload["retention"]
    execution = payload["execution"]
    if (
        identity["update5_scale_audit_contract_sha256"] != UPDATE5_SCALE_AUDIT_SHA256
        or identity["update5_scale_audit_result_sha256"] != UPDATE5_SCALE_AUDIT_RESULT_SHA256
        or identity["schedule_sha256"] != SCHEDULE_SHA256
        or int(identity["reconstructed_selected_update"]) != 5
        or tuple(int(value) for value in identity["continued_updates"]) != CONTINUED_UPDATES
        or tuple(int(value) for value in data["update5_artifact_updates"]) != UPDATE_FIVE_ARTIFACTS
        or tuple(tuple(int(value) for value in group) for group in data["continued_artifact_groups"])
        != CONTINUED_GROUPS
        or tuple(direction["source_registry"]) != GRADIENT_SOURCES
        or direction["solver"] != "minimum_norm_simplex_over_unit_source_gradients"
        or float(direction["minimum_source_cosine"]) != MINIMUM_SOURCE_COSINE
        or float(direction["full_parameter_step_l2"]) != REFERENCE_DELTA_L2
        or direction["parameter_dtype_preserved"] is not True
        or direction["recompute_after_every_accepted_update"] is not True
        or direction["adamw_realization_allowed"] is not False
        or tuple(float(value) for value in selection["parameter_displacement_scales_descending"]) != DISPLACEMENT_SCALES
        or selection["selection_rule"] != "evaluate_descending_and_apply_first_finite_step_accepted_scale"
        or tuple(float(value) for value in selection["update5_expected_accepted_scales"])
        != EXPECTED_UPDATE5_ACCEPTED_SCALES
        or float(selection["update5_expected_selected_scale"]) != EXPECTED_UPDATE5_SELECTED_SCALE
        or selection["evaluate_smaller_scales_after_first_acceptance"] is not False
        or int(evaluation["required_consecutive_scale_selected_updates"]) != REQUIRED_CONSECUTIVE_UPDATES
        or float(evaluation["relative_source_tolerance"]) != 1.0e-6
        or evaluation["stop_on_no_common_direction"] is not True
        or evaluation["stop_on_no_accepted_scale"] is not True
    ):
        raise ValueError("coupled GPP scale-selecting recalibration identity or method drift")
    if (
        any(
            bool(data[name])
            for name in (
                "fresh_validation_values_allowed",
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "sealed_test_allowed",
            )
        )
        or any(
            bool(retention[name])
            for name in (
                "optimizer_updates_persisted",
                "parameter_proposals_persisted",
                "recovery_modified",
                "checkpoint_written",
                "training_continuation_authorized",
            )
        )
        or any(
            bool(execution[name])
            for name in (
                "paid_partition_authorized",
                "fresh_validation_read_authorized",
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
            )
        )
    ):
        raise ValueError("coupled GPP scale-selecting recalibration opens a forbidden path")
    if not bool(execution["free_gln01_allowed"]):
        raise ValueError("coupled GPP scale-selecting recalibration free execution is closed")
    if int(execution["forcing_records_reconstructed"]) != 0 or int(execution["subdaily_state_scan_steps"]) != 0:
        raise ValueError("coupled GPP scale-selecting recalibration inference boundary drift")
    return CoupledGPPScaleSelectingRecalibrationContract(payload, source, observed)


__all__ = [
    "CONTINUED_GROUPS",
    "CONTINUED_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "EXPECTED_UPDATE5_ACCEPTED_SCALES",
    "EXPECTED_UPDATE5_SELECTED_SCALE",
    "FROZEN_CONTRACT_SHA256",
    "REQUIRED_CONSECUTIVE_UPDATES",
    "SCHEMA_VERSION",
    "UPDATE5_SCALE_AUDIT_RESULT_SHA256",
    "CoupledGPPScaleSelectingRecalibrationContract",
    "load_coupled_gpp_scale_selecting_recalibration_contract",
]
