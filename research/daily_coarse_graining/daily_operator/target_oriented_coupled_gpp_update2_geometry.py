"""Frozen update-2 optimizer-geometry audit for the coupled-GPP candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .target_oriented_coupled_gpp_bounded_training import (
    FROZEN_CONTRACT_SHA256 as BOUNDED_TRAINING_SHA256,
)
from .target_oriented_coupled_gpp_bounded_training import (
    SCHEDULE_SHA256,
)
from .target_oriented_coupled_gpp_objective import GRADIENT_SOURCES
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_update2_geometry_audit_v1"
FROZEN_CONTRACT_SHA256 = "7585cc5b46a76fb19d08703c8e88839f8758258888a1f302d5448dbf7e79f198"
PARENT_GIT_HEAD = "6d1864b2d7c2c95b176c1995d8f59ebd03d92d8a"
PARENT_RESULT_SHA256 = "bf83501782f0e6815a5bd408b203f15c8bcceefc3effb9cdda713aefd58be39c"
PARENT_REJECTED_SHA256 = "0dcd1525212a38891ac593be251807a32c6e7785e5156e9f2b6fba49b5f6a9e3"
UPDATE_TWO_ARTIFACTS = (326, 935, 542, 913, 897, 836, 887, 863)
DISPLACEMENT_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_update2_geometry_audit_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPUpdate2GeometryContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["directions"]["parameter_displacement_scales"])


def classify_update2_geometry(
    *,
    inherited_cosines: Mapping[str, float],
    reset_cosines: Mapping[str, float],
    inherited_accepted_scales: Sequence[float],
    reset_accepted_scales: Sequence[float],
) -> str:
    if set(inherited_cosines) != set(GRADIENT_SOURCES) or set(reset_cosines) != set(GRADIENT_SOURCES):
        raise ValueError("coupled GPP update-2 geometry source registry drift")
    inherited_common = all(float(inherited_cosines[name]) > 0.0 for name in GRADIENT_SOURCES)
    reset_common = all(float(reset_cosines[name]) > 0.0 for name in GRADIENT_SOURCES)
    if tuple(inherited_accepted_scales):
        return "inherited_backtracking_admissible"
    if tuple(reset_accepted_scales):
        return "moment_reset_admissible"
    if not inherited_common and reset_common:
        return "inherited_moment_rotation"
    if inherited_common or reset_common:
        return "common_direction_no_finite_scale"
    return "no_common_optimizer_direction"


def load_coupled_gpp_update2_geometry_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPUpdate2GeometryContract:
    import json

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_zero_training_update2_optimizer_geometry_audit"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("coupled GPP update-2 geometry contract is not frozen")
    identity = payload["identity"]
    data = payload["data"]
    directions = payload["directions"]
    evaluation = payload["evaluation"]
    retention = payload["retention"]
    execution = payload["execution"]
    if (
        identity["parent_git_head"] != PARENT_GIT_HEAD
        or identity["parent_result_report_sha256"] != PARENT_RESULT_SHA256
        or identity["parent_rejected_proposal_sha256"] != PARENT_REJECTED_SHA256
        or identity["bounded_training_contract_sha256"] != BOUNDED_TRAINING_SHA256
        or identity["schedule_sha256"] != SCHEDULE_SHA256
        or int(identity["expected_recovery_completed_updates"]) != 1
        or int(identity["expected_rejected_optimizer_update"]) != 2
        or tuple(int(value) for value in data["artifact_updates"]) != UPDATE_TWO_ARTIFACTS
        or tuple(float(value) for value in directions["parameter_displacement_scales"]) != DISPLACEMENT_SCALES
        or float(directions["learning_rate"]) != 2.0e-6
        or float(directions["global_gradient_norm_clip"]) != 12.808193184117403
        or float(evaluation["relative_source_tolerance"]) != 1.0e-6
        or float(evaluation["parent_scale_one_reference_absolute_tolerance"]) != 1.0e-8
        or evaluation["parent_scale_one_numeric_reference_is_reporting_only"] is not True
        or evaluation["parent_scale_one_frozen_gate_outcome_must_remain_rejected"] is not True
    ):
        raise ValueError("coupled GPP update-2 geometry identity or method drift")
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
        raise ValueError("coupled GPP update-2 geometry contract opens a forbidden path")
    if not bool(execution["free_gln01_allowed"]):
        raise ValueError("coupled GPP update-2 geometry free execution is closed")
    if int(execution["forcing_records_reconstructed"]) != 0 or int(execution["subdaily_state_scan_steps"]) != 0:
        raise ValueError("coupled GPP update-2 geometry inference boundary drift")
    return CoupledGPPUpdate2GeometryContract(payload, source, observed)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "DISPLACEMENT_SCALES",
    "FROZEN_CONTRACT_SHA256",
    "PARENT_GIT_HEAD",
    "PARENT_REJECTED_SHA256",
    "PARENT_RESULT_SHA256",
    "SCHEMA_VERSION",
    "UPDATE_TWO_ARTIFACTS",
    "CoupledGPPUpdate2GeometryContract",
    "classify_update2_geometry",
    "load_coupled_gpp_update2_geometry_contract",
]
