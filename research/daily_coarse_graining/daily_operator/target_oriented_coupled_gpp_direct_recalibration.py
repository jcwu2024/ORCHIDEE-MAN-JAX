"""Frozen direct common-direction recalibration smoke for coupled GPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_coupled_gpp_bounded_training import (
    FROZEN_CONTRACT_SHA256 as BOUNDED_TRAINING_SHA256,
)
from .target_oriented_coupled_gpp_bounded_training import SCHEDULE_SHA256
from .target_oriented_coupled_gpp_fresh_moment import (
    FROZEN_CONTRACT_SHA256 as FRESH_MOMENT_SHA256,
)
from .target_oriented_coupled_gpp_objective import GRADIENT_SOURCES
from .target_oriented_coupled_gpp_update3_common_direction import (
    FROZEN_CONTRACT_SHA256 as COMMON_DIRECTION_AUDIT_SHA256,
)
from .target_oriented_coupled_gpp_update3_common_direction import (
    MINIMUM_SOURCE_COSINE,
    REFERENCE_DELTA_L2,
    UPDATE_THREE_ARTIFACTS,
    UPDATE_TWO_ARTIFACTS,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_direct_recalibration_smoke_v1"
FROZEN_CONTRACT_SHA256 = "f7f0efc11fb716ec8946f70539b06585cce0c106087cd63de79acb5db648e9cf"
COMMON_DIRECTION_RESULT_SHA256 = "642086afe65c08909b162153cf08ab8ca72bf31f5041b47e422592165b95ec8d"
DIRECT_RECALIBRATION_UPDATES = (3, 4, 5)
DIRECT_RECALIBRATION_GROUPS = (
    UPDATE_THREE_ARTIFACTS,
    (481, 672, 900, 294, 998, 668, 433, 535),
    (965, 351, 272, 363, 660, 670, 548, 740),
)
REQUIRED_CONSECUTIVE_UPDATES = 3
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_direct_recalibration_smoke_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPDirectRecalibrationContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


def load_coupled_gpp_direct_recalibration_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPDirectRecalibrationContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_zero_retention_three_update_direct_recalibration_smoke"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("coupled GPP direct recalibration contract is not frozen")
    identity = payload["identity"]
    data = payload["data"]
    direction = payload["direction"]
    evaluation = payload["evaluation"]
    retention = payload["retention"]
    execution = payload["execution"]
    if (
        identity["common_direction_audit_contract_sha256"] != COMMON_DIRECTION_AUDIT_SHA256
        or identity["common_direction_result_sha256"] != COMMON_DIRECTION_RESULT_SHA256
        or identity["fresh_moment_contract_sha256"] != FRESH_MOMENT_SHA256
        or identity["bounded_training_contract_sha256"] != BOUNDED_TRAINING_SHA256
        or identity["schedule_sha256"] != SCHEDULE_SHA256
        or int(identity["parent_completed_updates"]) != 1
        or int(identity["reconstructed_fresh_update"]) != 2
        or tuple(int(value) for value in identity["direct_recalibration_updates"]) != DIRECT_RECALIBRATION_UPDATES
        or tuple(int(value) for value in data["reconstruction_artifact_updates"]) != UPDATE_TWO_ARTIFACTS
        or tuple(tuple(int(value) for value in group) for group in data["direct_recalibration_artifact_groups"])
        != DIRECT_RECALIBRATION_GROUPS
        or tuple(direction["source_registry"]) != GRADIENT_SOURCES
        or direction["solver"] != "minimum_norm_simplex_over_unit_source_gradients"
        or float(direction["minimum_source_cosine"]) != MINIMUM_SOURCE_COSINE
        or float(direction["parameter_step_l2"]) != REFERENCE_DELTA_L2
        or float(direction["parameter_step_scale"]) != 1.0
        or direction["recompute_after_every_accepted_update"] is not True
        or direction["adamw_realization_allowed"] is not False
        or int(evaluation["required_consecutive_direct_updates"]) != REQUIRED_CONSECUTIVE_UPDATES
        or float(evaluation["relative_source_tolerance"]) != 1.0e-6
        or evaluation["stop_on_first_scientific_rejection"] is not True
    ):
        raise ValueError("coupled GPP direct recalibration identity or method drift")
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
        raise ValueError("coupled GPP direct recalibration opens a forbidden path")
    if not bool(execution["free_gln01_allowed"]):
        raise ValueError("coupled GPP direct recalibration free execution is closed")
    if int(execution["forcing_records_reconstructed"]) != 0 or int(execution["subdaily_state_scan_steps"]) != 0:
        raise ValueError("coupled GPP direct recalibration inference boundary drift")
    return CoupledGPPDirectRecalibrationContract(payload, source, observed)


__all__ = [
    "COMMON_DIRECTION_RESULT_SHA256",
    "DEFAULT_CONTRACT_PATH",
    "DIRECT_RECALIBRATION_GROUPS",
    "DIRECT_RECALIBRATION_UPDATES",
    "FROZEN_CONTRACT_SHA256",
    "REQUIRED_CONSECUTIVE_UPDATES",
    "SCHEMA_VERSION",
    "CoupledGPPDirectRecalibrationContract",
    "load_coupled_gpp_direct_recalibration_contract",
]
