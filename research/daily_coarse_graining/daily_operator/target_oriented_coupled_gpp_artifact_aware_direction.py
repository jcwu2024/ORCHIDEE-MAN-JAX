"""Frozen artifact-aware direct-GPP direction audit contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .target_oriented_coupled_gpp_direct_generalization import MONITOR_UPDATES
from .target_oriented_coupled_gpp_direct_step_transfer import (
    AUDITED_PERSISTENT_UPDATE,
    DISPLACEMENT_SCALES,
)
from .target_oriented_coupled_gpp_multi_group_direction import (
    FROZEN_CONTRACT_SHA256 as MULTI_GROUP_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    TRAINABLE_PARAMETER_COUNT,
)
from .target_oriented_coupled_gpp_persistent_scale_training import (
    FROZEN_CONTRACT_SHA256 as PERSISTENT_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_persistent_scale_training import (
    SCHEDULE_SHA256,
    build_persistent_scale_training_schedule,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_artifact_aware_direction_audit_v1"
FROZEN_CONTRACT_SHA256 = "0606e7e22822a7d01d704929d72056f7a20e95b67299a86d22ea49053f37fc52"
MULTI_GROUP_RESULT_SHA256 = "cf10f5e27da8f0c50f582b14e7513b6ec87a913b2121dd52584a872672663577"
TRAIN_ONLY_GROUP_UPDATES = (2, 11, 12, 13, 14)
SCHEDULED_NON_GPP_SOURCES = GRADIENT_SOURCES[1:]
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.json"
)


def _artifact_sources(groups: tuple[tuple[int, ...], ...]) -> tuple[str, ...]:
    return tuple(f"{DIRECT_SOURCE}.artifact_{update}" for group in groups for update in group)


@dataclass(frozen=True)
class CoupledGPPArtifactAwareDirectionContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["scale_selection"]["displacement_scales_descending"])

    @property
    def train_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(value) for value in group) for group in self.payload["data"]["train_only_artifact_groups"]
        )

    @property
    def train_artifacts(self) -> tuple[int, ...]:
        return tuple(value for group in self.train_groups for value in group)

    @property
    def source_registry(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["direction"]["source_registry"])

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["fixed_monitor_artifact_updates"])


def artifact_aware_scale_passed(
    record: Mapping[str, Any],
    *,
    expected_artifact_count: int,
    relative_tolerance: float,
) -> bool:
    changes = tuple(float(value) for value in record["artifact_direct_gpp_relative_changes"])
    if len(changes) != expected_artifact_count or np.any(~np.isfinite(np.asarray(changes))):
        raise ValueError("artifact-aware finite-step record is incomplete")
    return bool(
        record["scheduled_four_source_finite_step_accepted"]
        and record["scheduled_physical_hard_gate_passed"]
        and max(changes) <= float(relative_tolerance)
        and float(record["combined_artifact_direct_gpp_relative_change"]) < 0.0
    )


def classify_artifact_aware_direction(
    *,
    common_first_order_direction: bool,
    selected_scale: float | None,
    monitor_relative_change: float | None,
) -> str:
    if not common_first_order_direction:
        if selected_scale is not None or monitor_relative_change is not None:
            raise ValueError("a missing artifact-aware direction cannot have a step or monitor result")
        return "no_artifact_aware_common_direction"
    if selected_scale is None:
        if monitor_relative_change is not None:
            raise ValueError("a missing artifact-aware train-only scale cannot have a monitor result")
        return "no_artifact_aware_finite_step"
    if float(selected_scale) not in DISPLACEMENT_SCALES:
        raise ValueError("selected artifact-aware scale is outside the frozen ladder")
    if monitor_relative_change is None or not np.isfinite(float(monitor_relative_change)):
        raise ValueError("a selected artifact-aware scale requires a finite monitor result")
    return (
        "artifact_aware_direction_transfers_to_monitor"
        if float(monitor_relative_change) < 0.0
        else "artifact_aware_direction_fails_monitor"
    )


def load_coupled_gpp_artifact_aware_direction_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPArtifactAwareDirectionContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    direction = payload.get("direction", {})
    selection = payload.get("scale_selection", {})
    retention = payload.get("retention", {})
    execution = payload.get("execution", {})
    schedule = build_persistent_scale_training_schedule()
    expected_groups = tuple(schedule.groups[index - 1] for index in TRAIN_ONLY_GROUP_UPDATES)
    observed_groups = tuple(
        tuple(int(value) for value in group) for group in data.get("train_only_artifact_groups", ())
    )
    artifacts = tuple(value for group in observed_groups for value in group)
    registry = _artifact_sources(observed_groups) + SCHEDULED_NON_GPP_SOURCES
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_artifact_gradient_values_are_evaluated"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("multi_group_contract_sha256") != MULTI_GROUP_CONTRACT_SHA256
        or identity.get("multi_group_result_sha256") != MULTI_GROUP_RESULT_SHA256
        or identity.get("persistent_contract_sha256") != PERSISTENT_CONTRACT_SHA256
        or identity.get("persistent_schedule_sha256") != SCHEDULE_SHA256
        or schedule.sha256 != SCHEDULE_SHA256
        or int(identity.get("trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(data.get("audited_persistent_update", -1)) != AUDITED_PERSISTENT_UPDATE
        or observed_groups != expected_groups
        or len(artifacts) != 40
        or len(set(artifacts)) != 40
        or int(data.get("train_only_artifact_count", -1)) != len(artifacts)
        or int(data.get("teacher_days_per_direct_gpp_artifact", -1)) != 208
        or tuple(int(value) for value in data.get("fixed_monitor_artifact_updates", ())) != MONITOR_UPDATES
        or data.get("fixed_monitor_excluded_from_direction_and_scale_selection") is not True
        or data.get("candidate_direction_monitor_values_evaluated_before_contract_freeze") is not False
        or tuple(direction.get("source_registry", ())) != registry
        or int(direction.get("gradient_constraint_count", -1)) != len(registry)
        or int(direction.get("direct_gpp_artifact_constraint_count", -1)) != len(artifacts)
        or int(direction.get("scheduled_non_gpp_constraint_count", -1)) != len(SCHEDULED_NON_GPP_SOURCES)
        or direction.get("each_gradient_unit_normalized") is not True
        or float(direction.get("minimum_constraint_cosine", -1.0)) != 1.0e-4
        or float(direction.get("full_parameter_step_l2", -1.0)) != 0.007758973351050004
        or direction.get("optimizer_or_momentum_state_used") is not False
        or tuple(float(value) for value in selection.get("displacement_scales_descending", ())) != DISPLACEMENT_SCALES
        or selection.get("stop_after_first_train_only_passing_scale") is not True
        or selection.get("monitor_values_excluded_from_selection") is not True
        or float(selection.get("each_train_only_artifact_direct_gpp_relative_change_maximum", -1.0)) != 1.0e-6
    ):
        raise ValueError("coupled GPP artifact-aware direction contract drift")
    if any(bool(value) for value in retention.values()):
        raise ValueError("coupled GPP artifact-aware direction audit opens a retention path")
    if (
        execution.get("free_gln01_allowed") is not True
        or execution.get("paid_partition_authorized") is not False
        or int(execution.get("timeout_minutes", -1)) != 120
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
        or any(
            bool(execution[name])
            for name in (
                "fresh_validation_read_authorized",
                "teacher_data_regeneration_authorized",
                "historical_v5_authorized",
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
            )
        )
    ):
        raise ValueError("coupled GPP artifact-aware direction execution boundary drift")
    return CoupledGPPArtifactAwareDirectionContract(payload, source, observed)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "MULTI_GROUP_RESULT_SHA256",
    "SCHEDULED_NON_GPP_SOURCES",
    "SCHEMA_VERSION",
    "TRAIN_ONLY_GROUP_UPDATES",
    "CoupledGPPArtifactAwareDirectionContract",
    "artifact_aware_scale_passed",
    "classify_artifact_aware_direction",
    "load_coupled_gpp_artifact_aware_direction_contract",
]
