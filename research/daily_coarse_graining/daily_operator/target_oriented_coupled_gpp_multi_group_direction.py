"""Frozen multi-group direct-GPP direction audit contract."""

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
    GUARD_SCHEDULE_UPDATES,
)
from .target_oriented_coupled_gpp_direct_step_transfer import (
    FROZEN_CONTRACT_SHA256 as DIRECT_STEP_TRANSFER_CONTRACT_SHA256,
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

SCHEMA_VERSION = "target_oriented_coupled_gpp_multi_group_direction_audit_v1"
FROZEN_CONTRACT_SHA256 = "1c103e79663d6da22e7fcc1275432d4caec08bbe4390c695dd17ee1596b82d19"
DIRECT_STEP_TRANSFER_RESULT_SHA256 = "df2593b423c3f380f6af45b7ee3faaeddae0f78a29911371928b54a91a966c94"
MULTI_GROUP_GRADIENT_SOURCES = (
    f"{DIRECT_SOURCE}.scheduled",
    f"{DIRECT_SOURCE}.guard_1",
    f"{DIRECT_SOURCE}.guard_2",
    f"{DIRECT_SOURCE}.guard_3",
    f"{DIRECT_SOURCE}.guard_4",
    GRADIENT_SOURCES[1],
    GRADIENT_SOURCES[2],
    GRADIENT_SOURCES[3],
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_multi_group_direction_audit_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPMultiGroupDirectionContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["scale_selection"]["displacement_scales_descending"])

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


def classify_multi_group_direction(
    *,
    common_first_order_direction: bool,
    selected_scale: float | None,
    monitor_relative_change: float | None,
) -> str:
    if not common_first_order_direction:
        if selected_scale is not None or monitor_relative_change is not None:
            raise ValueError("a missing multi-group direction cannot have a step or monitor result")
        return "no_multi_group_common_direction"
    if selected_scale is None:
        if monitor_relative_change is not None:
            raise ValueError("a missing train-only scale cannot have a monitor result")
        return "no_multi_group_finite_step"
    if float(selected_scale) not in DISPLACEMENT_SCALES:
        raise ValueError("selected multi-group scale is outside the frozen ladder")
    if monitor_relative_change is None or not np.isfinite(float(monitor_relative_change)):
        raise ValueError("a selected multi-group scale requires a finite monitor result")
    return (
        "multi_group_direction_transfers_to_monitor"
        if float(monitor_relative_change) < 0.0
        else "multi_group_direction_fails_monitor"
    )


def load_coupled_gpp_multi_group_direction_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPMultiGroupDirectionContract:
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
    guard_groups = tuple(schedule.groups[index - 1] for index in GUARD_SCHEDULE_UPDATES)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_multi_group_gradient_values_are_evaluated"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("direct_step_transfer_contract_sha256") != DIRECT_STEP_TRANSFER_CONTRACT_SHA256
        or identity.get("direct_step_transfer_result_sha256") != DIRECT_STEP_TRANSFER_RESULT_SHA256
        or identity.get("persistent_contract_sha256") != PERSISTENT_CONTRACT_SHA256
        or identity.get("persistent_schedule_sha256") != SCHEDULE_SHA256
        or schedule.sha256 != SCHEDULE_SHA256
        or int(identity.get("trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(data.get("audited_persistent_update", -1)) != AUDITED_PERSISTENT_UPDATE
        or tuple(int(value) for value in data.get("scheduled_artifact_updates", ()))
        != schedule.groups[AUDITED_PERSISTENT_UPDATE - 1]
        or tuple(int(value) for value in data.get("train_only_guard_persistent_updates", ())) != GUARD_SCHEDULE_UPDATES
        or tuple(tuple(int(value) for value in group) for group in data.get("train_only_guard_artifact_groups", ()))
        != guard_groups
        or tuple(int(value) for value in data.get("fixed_monitor_artifact_updates", ())) != MONITOR_UPDATES
        or data.get("fixed_monitor_excluded_from_direction_and_scale_selection") is not True
        or data.get("candidate_direction_monitor_values_evaluated_before_contract_freeze") is not False
        or tuple(direction.get("source_registry", ())) != MULTI_GROUP_GRADIENT_SOURCES
        or direction.get("each_group_mean_gradient_unit_normalized") is not True
        or float(direction.get("minimum_constraint_cosine", -1.0)) != 1.0e-4
        or float(direction.get("full_parameter_step_l2", -1.0)) != 0.007758973351050004
        or direction.get("optimizer_or_momentum_state_used") is not False
        or tuple(float(value) for value in selection.get("displacement_scales_descending", ())) != DISPLACEMENT_SCALES
        or selection.get("monitor_values_excluded_from_selection") is not True
        or float(selection.get("each_train_only_guard_group_direct_gpp_relative_change_maximum", -1.0)) != 1.0e-6
    ):
        raise ValueError("coupled GPP multi-group direction contract drift")
    if any(bool(value) for value in retention.values()):
        raise ValueError("coupled GPP multi-group direction audit opens a retention path")
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
        raise ValueError("coupled GPP multi-group direction execution boundary drift")
    return CoupledGPPMultiGroupDirectionContract(payload, source, observed)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "DIRECT_STEP_TRANSFER_RESULT_SHA256",
    "FROZEN_CONTRACT_SHA256",
    "MULTI_GROUP_GRADIENT_SOURCES",
    "SCHEMA_VERSION",
    "CoupledGPPMultiGroupDirectionContract",
    "classify_multi_group_direction",
    "load_coupled_gpp_multi_group_direction_contract",
]
