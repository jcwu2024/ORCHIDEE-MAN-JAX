"""Frozen update-3 common-direction audit for the coupled-GPP candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .target_oriented_coupled_gpp_bounded_training import (
    FROZEN_CONTRACT_SHA256 as BOUNDED_TRAINING_SHA256,
)
from .target_oriented_coupled_gpp_bounded_training import SCHEDULE_SHA256
from .target_oriented_coupled_gpp_fresh_moment import (
    FROZEN_CONTRACT_SHA256 as FRESH_MOMENT_SHA256,
)
from .target_oriented_coupled_gpp_objective import GRADIENT_SOURCES
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_update3_common_direction_audit_v1"
FROZEN_CONTRACT_SHA256 = "956940a1b9cc159fb732980b5bb0f090fc9664adebcf27b22719ec310855cf09"
FRESH_MOMENT_RESULT_SHA256 = "3e07fb8a313c9885e069d44b5266212d4fe12e0b9295876db8a701ecdc15f834"
UPDATE_TWO_ARTIFACTS = (326, 935, 542, 913, 897, 836, 887, 863)
UPDATE_THREE_ARTIFACTS = (968, 750, 987, 728, 953, 253, 646, 937)
DISPLACEMENT_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
REFERENCE_DELTA_L2 = 0.007758973351050004
MINIMUM_SOURCE_COSINE = 1.0e-4
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_update3_common_direction_audit_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPUpdate3CommonDirectionContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["direction"]["parameter_displacement_scales"])


def classify_update3_common_direction(
    *,
    raw_source_cosines: Mapping[str, float],
    direct_accepted_scales: Sequence[float],
    adamw_source_cosines: Mapping[str, float],
    adamw_accepted_scales: Sequence[float],
    minimum_source_cosine: float = MINIMUM_SOURCE_COSINE,
) -> str:
    if set(raw_source_cosines) != set(GRADIENT_SOURCES) or set(adamw_source_cosines) != set(GRADIENT_SOURCES):
        raise ValueError("coupled GPP update-3 common-direction source registry drift")
    raw_common = all(float(raw_source_cosines[name]) >= minimum_source_cosine for name in GRADIENT_SOURCES)
    adamw_common = all(float(adamw_source_cosines[name]) >= minimum_source_cosine for name in GRADIENT_SOURCES)
    if not raw_common:
        return "no_common_first_order_direction"
    if tuple(direct_accepted_scales):
        return "direct_common_finite_step"
    if adamw_common and tuple(adamw_accepted_scales):
        return "fresh_adamw_common_finite_step"
    if tuple(adamw_accepted_scales):
        return "adamw_finite_step_without_common_geometry"
    return "common_direction_no_finite_scale"


def load_coupled_gpp_update3_common_direction_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPUpdate3CommonDirectionContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_zero_retention_update3_common_direction_audit"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("coupled GPP update-3 common-direction contract is not frozen")
    identity = payload["identity"]
    data = payload["data"]
    direction = payload["direction"]
    evaluation = payload["evaluation"]
    retention = payload["retention"]
    execution = payload["execution"]
    if (
        identity["fresh_moment_contract_sha256"] != FRESH_MOMENT_SHA256
        or identity["fresh_moment_result_sha256"] != FRESH_MOMENT_RESULT_SHA256
        or identity["bounded_training_contract_sha256"] != BOUNDED_TRAINING_SHA256
        or identity["schedule_sha256"] != SCHEDULE_SHA256
        or int(identity["parent_completed_updates"]) != 1
        or int(identity["reconstructed_fresh_update"]) != 2
        or int(identity["audited_update"]) != 3
        or tuple(int(value) for value in data["reconstruction_artifact_updates"]) != UPDATE_TWO_ARTIFACTS
        or tuple(int(value) for value in data["audit_artifact_updates"]) != UPDATE_THREE_ARTIFACTS
        or tuple(direction["source_registry"]) != GRADIENT_SOURCES
        or direction["solver"] != "minimum_norm_simplex_over_unit_source_gradients"
        or float(direction["minimum_source_cosine"]) != MINIMUM_SOURCE_COSINE
        or float(direction["learning_rate"]) != 2.0e-6
        or float(direction["global_gradient_norm_clip"]) != 12.808193184117403
        or float(direction["reference_update3_fresh_adamw_parameter_delta_l2"]) != REFERENCE_DELTA_L2
        or tuple(float(value) for value in direction["parameter_displacement_scales"]) != DISPLACEMENT_SCALES
        or float(evaluation["relative_source_tolerance"]) != 1.0e-6
        or evaluation["numeric_reproduction_differences_are_reporting_only"] is not True
    ):
        raise ValueError("coupled GPP update-3 common-direction identity or method drift")
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
        raise ValueError("coupled GPP update-3 common-direction audit opens a forbidden path")
    if not bool(execution["free_gln01_allowed"]):
        raise ValueError("coupled GPP update-3 common-direction free execution is closed")
    if int(execution["forcing_records_reconstructed"]) != 0 or int(execution["subdaily_state_scan_steps"]) != 0:
        raise ValueError("coupled GPP update-3 common-direction inference boundary drift")
    return CoupledGPPUpdate3CommonDirectionContract(payload, source, observed)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "DISPLACEMENT_SCALES",
    "FRESH_MOMENT_RESULT_SHA256",
    "FROZEN_CONTRACT_SHA256",
    "MINIMUM_SOURCE_COSINE",
    "REFERENCE_DELTA_L2",
    "SCHEMA_VERSION",
    "UPDATE_THREE_ARTIFACTS",
    "UPDATE_TWO_ARTIFACTS",
    "CoupledGPPUpdate3CommonDirectionContract",
    "classify_update3_common_direction",
    "load_coupled_gpp_update3_common_direction_contract",
]
