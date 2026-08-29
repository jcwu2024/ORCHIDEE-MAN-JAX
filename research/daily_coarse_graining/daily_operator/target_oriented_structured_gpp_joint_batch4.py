"""Discarded batch-four AdamW preflight contract for structured GPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_source_shaped_maintenance import contract_sha256
from .target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from .target_oriented_structured_gpp_training import SPECIALIST_PARAMETER_COUNT

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_batch4_preflight_v1"
FROZEN_CONTRACT_SHA256 = "06d8c3f43beeac9c0c64dc8fd679ff57d09dc7b34df64a233df9f9d0dcc19169"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_joint_batch4_preflight_v1.json"
)
ARTIFACT_GROUPS = ((512, 548, 584, 620), (656, 692, 728, 763))
RUNTIME_YEAR_GROUPS = ((1973, 1969, 1989, 1988), (1972, 2003, 2002, 1987))


@dataclass(frozen=True)
class StructuredGPPJointBatch4Contract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def artifact_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(value) for value in group)
            for group in self.payload["data"]["artifact_groups"]
        )

    @property
    def coefficients(self) -> Mapping[str, float]:
        return {
            str(name): float(value)
            for name, value in self.payload["objective"]["coefficients"].items()
        }

    @property
    def clip_threshold(self) -> float:
        return float(self.payload["objective"]["combined_gradient_p90_clip_threshold"])


def mean_source_losses(
    values: tuple[Mapping[str, float], ...],
) -> Mapping[str, float]:
    if not values:
        raise ValueError("structured GPP batch4 mean requires source losses")
    return {
        name: sum(float(record[name]) for record in values) / len(values)
        for name in GRADIENT_SOURCES
    }


def load_structured_gpp_joint_batch4_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPJointBatch4Contract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structured GPP batch4 preflight schema drift")
    if (
        payload.get("status") != "frozen_before_free_gpu_discarded_batch4_joint_updates"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP batch4 preflight contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    objective = payload["objective"]
    optimizer = payload["optimizer_proposals"]
    acceptance = payload["acceptance"]
    execution = payload["execution"]
    coefficients = {str(name): float(value) for name, value in objective["coefficients"].items()}
    artifact_groups = tuple(tuple(int(value) for value in group) for group in data["artifact_groups"])
    runtime_groups = tuple(
        tuple(int(value) for value in group) for group in data["artifact_runtime_year_groups"]
    )
    if (
        identity["heterogeneity_execution_git_head"]
        != "95322f8eb6fca4ff5cf9ac4d722cec11710cd1a5"
        or identity["heterogeneity_report_sha256"]
        != "15d993a2c184918c7b6feb246a246d4d0ee8ecabfcdefb240abfd0659c880eea"
        or int(identity["warm_start_checkpoint_update"]) != 250
        or int(identity["candidate_parameter_count"]) != 106_383_527
        or int(identity["trainable_specialist_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(identity["retained_parent_parameter_count"]) != 93_504_164
        or int(data["effective_batch_size"]) != 4
        or artifact_groups != ARTIFACT_GROUPS
        or runtime_groups != RUNTIME_YEAR_GROUPS
        or int(data["direct_gpp_teacher_days_per_artifact"]) != 208
        or int(data["seven_day_windows_per_artifact"]) != 8
        or int(data["thirty_day_windows_per_artifact"]) != 4
        or data["source_loss_aggregation"] != "arithmetic_mean_across_four_artifacts"
        or data["per_artifact_loss_changes"]
        != "reported_without_individual_acceptance_threshold"
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
        raise ValueError("structured GPP batch4 identity or data boundary drift")
    if (
        tuple(objective["gradient_sources"]) != GRADIENT_SOURCES
        or set(coefficients) != set(GRADIENT_SOURCES)
        or coefficients[DIRECT_SOURCE] != 1.0
        or coefficients[SEVEN_SOURCE] != 5.065197169020285
        or coefficients[THIRTY_SOURCE] != 4.594433600967992
        or float(objective["combined_gradient_p90_clip_threshold"])
        != 46.802861793665286
        or objective["source_gradient_aggregation"]
        != "arithmetic_mean_across_four_artifacts"
        or objective["coefficients_recalibrated"] is not False
        or objective["validation_weight_tuning_allowed"] is not False
    ):
        raise ValueError("structured GPP batch4 objective drift")
    if (
        int(optimizer["proposal_count"]) != 2
        or optimizer["proposals_are_independent"] is not True
        or optimizer["optimizer"] != "adamw"
        or int(optimizer["initial_optimizer_step"]) != 250
        or int(optimizer["proposed_optimizer_step"]) != 251
        or float(optimizer["learning_rate"]) != 4e-6
        or float(optimizer["gradient_clip_threshold"]) != 46.802861793665286
        or int(optimizer["updates_executed"]) != 2
        or int(optimizer["updates_persisted"]) != 0
        or optimizer["checkpoint_written"] is not False
        or optimizer["recovery_written"] is not False
        or acceptance["per_artifact_loss_change_is_not_an_acceptance_gate"] is not True
        or acceptance["both_proposals_start_from_same_parameter_and_optimizer_state"] is not True
    ):
        raise ValueError("structured GPP batch4 optimizer or acceptance drift")
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
        raise ValueError("structured GPP batch4 execution boundary drift")
    return StructuredGPPJointBatch4Contract(payload, source, observed)


__all__ = [
    "ARTIFACT_GROUPS",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "RUNTIME_YEAR_GROUPS",
    "SCHEMA_VERSION",
    "StructuredGPPJointBatch4Contract",
    "load_structured_gpp_joint_batch4_contract",
    "mean_source_losses",
]
