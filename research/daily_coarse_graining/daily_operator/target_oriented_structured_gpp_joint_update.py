"""Frozen discarded joint-update preflight for structured GPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .target_oriented_source_shaped_maintenance import contract_sha256
from .target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from .target_oriented_structured_gpp_training import SPECIALIST_PARAMETER_COUNT

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_update_preflight_v1"
FROZEN_CONTRACT_SHA256 = "5d255e49f7add44c4948a3e737fe6edd72999b6fbd6a37439dc787cb38c0c5d5"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_joint_update_preflight_v1.json"
)


@dataclass(frozen=True)
class StructuredGPPJointUpdatePreflightContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def artifact_update(self) -> int:
        return int(self.payload["data"]["ephemeral_update_artifact"])

    @property
    def coefficients(self) -> Mapping[str, float]:
        return {
            str(name): float(value)
            for name, value in self.payload["objective"]["coefficients"].items()
        }

    @property
    def clip_threshold(self) -> float:
        return float(self.payload["objective"]["combined_gradient_p90_clip_threshold"])


def load_structured_gpp_joint_update_preflight_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPJointUpdatePreflightContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structured GPP joint-update preflight schema drift")
    if (
        payload.get("status") != "frozen_before_free_gpu_discarded_joint_update"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP joint-update preflight contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    objective = payload["objective"]
    optimizer = payload["optimizer_proposal"]
    execution = payload["execution"]
    coefficients = {str(name): float(value) for name, value in objective["coefficients"].items()}
    if (
        int(identity["warm_start_checkpoint_update"]) != 250
        or int(identity["candidate_parameter_count"]) != 106_383_527
        or int(identity["trainable_specialist_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(identity["retained_parent_parameter_count"]) != 93_504_164
        or int(data["ephemeral_update_artifact"]) != 512
        or int(data["direct_gpp_teacher_days"]) != 208
        or int(data["seven_day_windows"]) != 8
        or int(data["thirty_day_windows"]) != 4
        or any(
            bool(data[name])
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_values_allowed",
                "sealed_test_allowed",
            )
        )
    ):
        raise ValueError("structured GPP joint-update identity or data boundary drift")
    if (
        tuple(objective["gradient_sources"]) != GRADIENT_SOURCES
        or set(coefficients) != set(GRADIENT_SOURCES)
        or coefficients[DIRECT_SOURCE] != 1.0
        or coefficients[SEVEN_SOURCE] != 5.065197169020285
        or coefficients[THIRTY_SOURCE] != 4.594433600967992
        or float(objective["combined_gradient_p90_clip_threshold"]) != 46.802861793665286
        or objective["coefficients_recalibrated"] is not False
        or objective["validation_weight_tuning_allowed"] is not False
        or optimizer["optimizer"] != "adamw"
        or optimizer["state_source"] != "warm_start_checkpoint_after_250_direct_gpp_updates"
        or int(optimizer["initial_optimizer_step"]) != 250
        or int(optimizer["proposed_optimizer_step"]) != 251
        or float(optimizer["learning_rate"]) != 4.0e-6
        or float(optimizer["beta1"]) != 0.9
        or float(optimizer["beta2"]) != 0.95
        or float(optimizer["epsilon"]) != 1.0e-8
        or float(optimizer["weight_decay"]) != 0.01
        or float(optimizer["gradient_clip_threshold"]) != float(
            objective["combined_gradient_p90_clip_threshold"]
        )
        or int(optimizer["updates_executed"]) != 1
        or int(optimizer["updates_persisted"]) != 0
        or optimizer["checkpoint_written"] is not False
        or optimizer["recovery_written"] is not False
    ):
        raise ValueError("structured GPP joint-update objective or optimizer drift")
    if (
        execution["free_gln01_allowed"] is not True
        or any(
            bool(execution[name])
            for name in (
                "paid_training_authorized",
                "joint_training_authorized",
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_read_authorized",
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
                "cross_day_neural_hidden_state",
            )
        )
        or int(execution["forcing_records_reconstructed"]) != 0
        or int(execution["subdaily_state_scan_steps"]) != 0
    ):
        raise ValueError("structured GPP joint-update execution boundary drift")
    return StructuredGPPJointUpdatePreflightContract(payload, source, observed)


def structured_gpp_joint_objectives(
    source_losses: Mapping[str, Any],
    coefficients: Mapping[str, float],
) -> Mapping[str, float]:
    values = {name: float(np.asarray(source_losses[name])) for name in GRADIENT_SOURCES}
    if set(values) != set(GRADIENT_SOURCES) or not all(
        np.isfinite(value) and value >= 0.0 for value in values.values()
    ):
        raise ValueError("structured GPP joint-update source loss is invalid")
    science = (
        float(coefficients[SEVEN_SOURCE]) * values[SEVEN_SOURCE]
        + float(coefficients[THIRTY_SOURCE]) * values[THIRTY_SOURCE]
    )
    total = float(coefficients[DIRECT_SOURCE]) * values[DIRECT_SOURCE] + science
    return {"direct_gpp": values[DIRECT_SOURCE], "weighted_science": science, "joint": total}


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "StructuredGPPJointUpdatePreflightContract",
    "load_structured_gpp_joint_update_preflight_contract",
    "structured_gpp_joint_objectives",
]
