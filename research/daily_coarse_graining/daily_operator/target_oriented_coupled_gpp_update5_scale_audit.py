"""Frozen update-5 bounded-scale audit for direct coupled-GPP recalibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_coupled_gpp_bounded_training import SCHEDULE_SHA256
from .target_oriented_coupled_gpp_direct_recalibration import (
    DIRECT_RECALIBRATION_GROUPS,
)
from .target_oriented_coupled_gpp_direct_recalibration import (
    FROZEN_CONTRACT_SHA256 as DIRECT_RECALIBRATION_SHA256,
)
from .target_oriented_coupled_gpp_objective import GRADIENT_SOURCES
from .target_oriented_coupled_gpp_update3_common_direction import (
    MINIMUM_SOURCE_COSINE,
    REFERENCE_DELTA_L2,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_update5_scale_audit_v1"
FROZEN_CONTRACT_SHA256 = "835a8ee86feb6691596552ac4047bc4f02905702d6534282c1f6be190ee69cbd"
DIRECT_RECALIBRATION_RESULT_SHA256 = "0c2d3e219f4dce16be40d15ce6a1ccbdb7c5f4f239e88e83a874e42b23242fd2"
UPDATE_FIVE_ARTIFACTS = DIRECT_RECALIBRATION_GROUPS[-1]
DISPLACEMENT_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_update5_scale_audit_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPUpdate5ScaleAuditContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["direction"]["parameter_displacement_scales"])


def load_coupled_gpp_update5_scale_audit_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPUpdate5ScaleAuditContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_zero_retention_update5_direct_scale_audit"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("coupled GPP update-5 scale audit contract is not frozen")
    identity = payload["identity"]
    data = payload["data"]
    direction = payload["direction"]
    evaluation = payload["evaluation"]
    retention = payload["retention"]
    execution = payload["execution"]
    if (
        identity["direct_recalibration_contract_sha256"] != DIRECT_RECALIBRATION_SHA256
        or identity["direct_recalibration_result_sha256"] != DIRECT_RECALIBRATION_RESULT_SHA256
        or identity["schedule_sha256"] != SCHEDULE_SHA256
        or tuple(int(value) for value in identity["reconstructed_accepted_updates"]) != (3, 4)
        or int(identity["audited_update"]) != 5
        or tuple(int(value) for value in data["audit_artifact_updates"]) != UPDATE_FIVE_ARTIFACTS
        or tuple(direction["source_registry"]) != GRADIENT_SOURCES
        or float(direction["minimum_source_cosine"]) != MINIMUM_SOURCE_COSINE
        or float(direction["full_parameter_step_l2"]) != REFERENCE_DELTA_L2
        or direction["parameter_dtype_preserved"] is not True
        or tuple(float(value) for value in direction["parameter_displacement_scales"]) != DISPLACEMENT_SCALES
        or evaluation["full_scale_evaluation_reused_from_reconstructed_smoke"] is not True
        or evaluation["updates_3_and_4_must_remain_accepted"] is not True
        or evaluation["update_5_full_scale_must_remain_rejected"] is not True
        or float(evaluation["relative_source_tolerance"]) != 1.0e-6
    ):
        raise ValueError("coupled GPP update-5 scale identity or method drift")
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
        raise ValueError("coupled GPP update-5 scale audit opens a forbidden path")
    if not bool(execution["free_gln01_allowed"]):
        raise ValueError("coupled GPP update-5 scale free execution is closed")
    if int(execution["forcing_records_reconstructed"]) != 0 or int(execution["subdaily_state_scan_steps"]) != 0:
        raise ValueError("coupled GPP update-5 scale inference boundary drift")
    return CoupledGPPUpdate5ScaleAuditContract(payload, source, observed)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "DIRECT_RECALIBRATION_RESULT_SHA256",
    "DISPLACEMENT_SCALES",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "UPDATE_FIVE_ARTIFACTS",
    "CoupledGPPUpdate5ScaleAuditContract",
    "load_coupled_gpp_update5_scale_audit_contract",
]
