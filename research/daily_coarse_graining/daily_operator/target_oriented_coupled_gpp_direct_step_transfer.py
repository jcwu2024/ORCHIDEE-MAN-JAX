"""Frozen finite-step transfer audit at the first persistent GPP regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_coupled_gpp_direct_generalization import (
    FROZEN_CONTRACT_SHA256 as DIRECT_GENERALIZATION_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_direct_generalization import (
    MONITOR_UPDATES,
    PERSISTENT_RESULT_SHA256,
)
from .target_oriented_coupled_gpp_objective import (
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

SCHEMA_VERSION = "target_oriented_coupled_gpp_direct_step_transfer_audit_v1"
FROZEN_CONTRACT_SHA256 = "b6f05589af4e8f985acf32b95853561a59ec68299bbd000ddec2438382913a61"
DIRECT_GENERALIZATION_RESULT_SHA256 = "23327c3c4081e283ea934acd03cb2db3c0c04b7ddbf261c430fb17b219e51cd9"
AUDITED_PERSISTENT_UPDATE = 2
GUARD_SCHEDULE_UPDATES = (11, 12, 13, 14)
DISPLACEMENT_SCALES = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_direct_step_transfer_audit_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPDirectStepTransferContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(
            float(value) for value in self.payload["scale_selection"]["parameter_displacement_scales_descending"]
        )

    @property
    def scheduled_group(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["scheduled_artifact_updates"])

    @property
    def guard_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(value) for value in group) for group in self.payload["data"]["train_only_guard_artifact_groups"]
        )

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["fixed_monitor_artifact_updates"])


def train_only_scale_passed(record: Mapping[str, Any], *, relative_tolerance: float) -> bool:
    changes = tuple(float(value) for value in record["guard_group_direct_gpp_relative_changes"])
    aggregate = float(record["combined_guard_direct_gpp_relative_change"])
    tolerance = float(relative_tolerance)
    if len(changes) != len(GUARD_SCHEDULE_UPDATES) or tolerance < 0.0:
        raise ValueError("direct-step transfer guard registry drift")
    if not np.isfinite(aggregate) or not all(np.isfinite(value) for value in changes):
        raise ValueError("direct-step transfer guard change is nonfinite")
    return bool(
        record["scheduled_four_source_finite_step_accepted"]
        and record["scheduled_physical_hard_gate_passed"]
        and all(value <= tolerance for value in changes)
        and aggregate < 0.0
    )


def select_largest_train_only_scale(
    records: Sequence[Mapping[str, Any]],
    *,
    relative_tolerance: float,
) -> float | None:
    values = tuple(records)
    if tuple(float(record["scale"]) for record in values) != DISPLACEMENT_SCALES:
        raise ValueError("direct-step transfer scale order drift")
    passing = [
        float(record["scale"])
        for record in values
        if train_only_scale_passed(record, relative_tolerance=relative_tolerance)
    ]
    return max(passing) if passing else None


def classify_step_transfer(selected_scale: float | None, *, monitor_relative_change: float | None) -> str:
    if selected_scale is None:
        if monitor_relative_change is not None:
            raise ValueError("monitor result cannot select a missing train-only scale")
        return "no_train_only_shared_finite_scale"
    if monitor_relative_change is None or not np.isfinite(float(monitor_relative_change)):
        raise ValueError("selected direct-step transfer scale requires a finite monitor result")
    if float(selected_scale) not in DISPLACEMENT_SCALES:
        raise ValueError("selected direct-step transfer scale is outside the frozen ladder")
    return (
        "train_only_scale_transfers_to_monitor"
        if float(monitor_relative_change) < 0.0
        else "train_only_scale_fails_monitor"
    )


def load_coupled_gpp_direct_step_transfer_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPDirectStepTransferContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    reconstruction = payload.get("reconstruction", {})
    direction = payload.get("direction", {})
    selection = payload.get("scale_selection", {})
    diagnosis = payload.get("diagnosis", {})
    retention = payload.get("retention", {})
    execution = payload.get("execution", {})
    schedule = build_persistent_scale_training_schedule()
    guard_groups = tuple(schedule.groups[index - 1] for index in GUARD_SCHEDULE_UPDATES)
    first_ten = {value for group in schedule.groups[:10] for value in group}
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_direct_step_transfer_values_are_evaluated"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("direct_generalization_contract_sha256") != DIRECT_GENERALIZATION_CONTRACT_SHA256
        or identity.get("direct_generalization_result_sha256") != DIRECT_GENERALIZATION_RESULT_SHA256
        or identity.get("persistent_contract_sha256") != PERSISTENT_CONTRACT_SHA256
        or identity.get("persistent_result_sha256") != PERSISTENT_RESULT_SHA256
        or identity.get("persistent_schedule_sha256") != SCHEDULE_SHA256
        or schedule.sha256 != SCHEDULE_SHA256
        or int(identity.get("trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(data.get("audited_persistent_update", -1)) != AUDITED_PERSISTENT_UPDATE
        or tuple(int(value) for value in data.get("scheduled_artifact_updates", ()))
        != schedule.groups[AUDITED_PERSISTENT_UPDATE - 1]
        or tuple(int(value) for value in data.get("train_only_guard_persistent_updates", ())) != GUARD_SCHEDULE_UPDATES
        or tuple(tuple(int(value) for value in group) for group in data.get("train_only_guard_artifact_groups", ()))
        != guard_groups
        or int(data.get("train_only_guard_artifact_count", -1)) != 32
        or any(value in first_ten for group in guard_groups for value in group)
        or data.get("guard_artifacts_used_by_persistent_updates_1_through_10") is not False
        or data.get("guard_outcomes_evaluated_before_contract_freeze") is not False
        or tuple(int(value) for value in data.get("fixed_monitor_artifact_updates", ())) != MONITOR_UPDATES
        or data.get("fixed_monitor_excluded_from_scale_selection") is not True
        or reconstruction.get("historical_scale_search_repeated") is not False
        or reconstruction.get("persistent_recovery_read") is not False
        or tuple(direction.get("source_registry", ())) != GRADIENT_SOURCES
        or float(direction.get("minimum_source_cosine", -1.0)) != 1.0e-4
        or float(direction.get("full_parameter_step_l2", -1.0)) != 0.007758973351050004
        or float(direction.get("audited_original_selected_scale", -1.0)) != 0.5
        or direction.get("direction_recomputed_once") is not True
        or direction.get("optimizer_or_momentum_state_used") is not False
        or tuple(float(value) for value in selection.get("parameter_displacement_scales_descending", ()))
        != DISPLACEMENT_SCALES
        or selection.get("monitor_values_excluded_from_selection") is not True
        or float(selection.get("each_train_only_guard_group_direct_gpp_relative_change_maximum", -1.0)) != 1.0e-6
        or diagnosis.get("fixed_monitor_evaluated_only_after_train_only_scale_selection") is not True
        or diagnosis.get("fixed_monitor_not_read_when_no_train_only_scale") is not True
        or diagnosis.get("monitor_transfer_requires_strict_mean_direct_gpp_decrease") is not True
        or diagnosis.get("scientific_model_promotion_implied") is not False
    ):
        raise ValueError("coupled GPP direct-step transfer contract drift")
    if any(
        bool(data[name])
        for name in (
            "fresh_validation_values_allowed",
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "sealed_test_allowed",
        )
    ) or any(bool(value) for value in retention.values()):
        raise ValueError("coupled GPP direct-step transfer audit opens a forbidden data or retention path")
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
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
            )
        )
    ):
        raise ValueError("coupled GPP direct-step transfer execution boundary drift")
    return CoupledGPPDirectStepTransferContract(payload, source, observed)


__all__ = [
    "AUDITED_PERSISTENT_UPDATE",
    "DEFAULT_CONTRACT_PATH",
    "DIRECT_GENERALIZATION_RESULT_SHA256",
    "DISPLACEMENT_SCALES",
    "FROZEN_CONTRACT_SHA256",
    "GUARD_SCHEDULE_UPDATES",
    "SCHEMA_VERSION",
    "CoupledGPPDirectStepTransferContract",
    "classify_step_transfer",
    "load_coupled_gpp_direct_step_transfer_contract",
    "select_largest_train_only_scale",
    "train_only_scale_passed",
]
