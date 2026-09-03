"""Frozen three-update artifact-aware direct-GPP continuity contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_coupled_gpp_artifact_aware_direction import (
    FROZEN_CONTRACT_SHA256 as ARTIFACT_AWARE_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_artifact_aware_direction import (
    SCHEDULED_NON_GPP_SOURCES,
)
from .target_oriented_coupled_gpp_direct_generalization import MONITOR_UPDATES
from .target_oriented_coupled_gpp_direct_step_transfer import DISPLACEMENT_SCALES
from .target_oriented_coupled_gpp_objective import DIRECT_SOURCE, TRAINABLE_PARAMETER_COUNT
from .target_oriented_coupled_gpp_persistent_scale_training import (
    FROZEN_CONTRACT_SHA256 as PERSISTENT_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_persistent_scale_training import (
    SCHEDULE_SHA256,
    build_persistent_scale_training_schedule,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1"
FROZEN_CONTRACT_SHA256 = "451a1003514c465b1e191f741e07cbe18285bf117eee6ae71020706d577e20f3"
ARTIFACT_AWARE_RESULT_SHA256 = "67bde6d8e57f9c49a31f5a6dbba89f0b24545d92a8a49366054668b62b09cbfe"
LOGICAL_UPDATES = (2, 3, 4)
TRAIN_PANEL_GROUP_UPDATES = (
    (2, 11, 12, 13, 14),
    (3, 15, 16, 17, 18),
    (4, 19, 20, 21, 22),
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.json"
)


def _artifact_sources(groups: Sequence[Sequence[int]]) -> tuple[str, ...]:
    return tuple(f"{DIRECT_SOURCE}.artifact_{update}" for group in groups for update in group)


@dataclass(frozen=True)
class CoupledGPPArtifactAwareContinuityContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["scale_selection"]["displacement_scales_descending"])

    @property
    def logical_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["logical_updates"])

    @property
    def panel_group_updates(self) -> tuple[tuple[int, ...], ...]:
        source = self.payload["data"]["train_only_schedule_group_updates_by_logical_update"]
        return tuple(tuple(int(value) for value in source[str(update)]) for update in self.logical_updates)

    @property
    def train_panels(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        schedule = build_persistent_scale_training_schedule()
        return tuple(
            tuple(tuple(int(value) for value in schedule.groups[index - 1]) for index in group_updates)
            for group_updates in self.panel_group_updates
        )

    @property
    def source_registries(self) -> tuple[tuple[str, ...], ...]:
        return tuple(_artifact_sources(panel) + SCHEDULED_NON_GPP_SOURCES for panel in self.train_panels)

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["fixed_monitor_artifact_updates"])


def monitor_trajectory_passed(relative_changes: Sequence[float], *, expected_updates: int = 3) -> bool:
    values = np.asarray(tuple(float(value) for value in relative_changes), dtype=np.float64)
    if values.size != expected_updates or np.any(~np.isfinite(values)):
        raise ValueError("artifact-aware monitor trajectory is incomplete")
    return bool(np.all(values < 0.0))


def classify_artifact_aware_continuity(
    *,
    accepted_update_count: int,
    required_update_count: int,
    monitor_relative_changes: Sequence[float] | None,
) -> str:
    accepted = int(accepted_update_count)
    required = int(required_update_count)
    if accepted < 0 or required <= 0 or accepted > required:
        raise ValueError("artifact-aware continuity update count is invalid")
    if accepted < required:
        if monitor_relative_changes is not None:
            raise ValueError("a stopped train-only continuity smoke cannot read the monitor")
        return "artifact_aware_continuity_stops_train_only"
    if monitor_relative_changes is None:
        raise ValueError("completed train-only continuity requires a monitor trajectory")
    return (
        "artifact_aware_three_update_continuity_passes_monitor"
        if monitor_trajectory_passed(monitor_relative_changes, expected_updates=required)
        else "artifact_aware_three_update_continuity_fails_monitor"
    )


def load_coupled_gpp_artifact_aware_continuity_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPArtifactAwareContinuityContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    reconstruction = payload.get("reconstruction", {})
    direction = payload.get("direction", {})
    selection = payload.get("scale_selection", {})
    monitor = payload.get("monitor_evaluation", {})
    retention = payload.get("retention", {})
    execution = payload.get("execution", {})
    schedule = build_persistent_scale_training_schedule()
    configured_group_updates = data.get("train_only_schedule_group_updates_by_logical_update", {})
    observed_group_updates = tuple(
        tuple(int(value) for value in configured_group_updates.get(str(update), ())) for update in LOGICAL_UPDATES
    )
    panels = tuple(
        tuple(tuple(int(value) for value in schedule.groups[index - 1]) for index in group_updates)
        for group_updates in observed_group_updates
    )
    flattened_panels = tuple(tuple(value for group in panel for value in group) for panel in panels)
    all_artifacts = tuple(value for panel in flattened_panels for value in panel)
    monitor_updates = tuple(int(value) for value in data.get("fixed_monitor_artifact_updates", ()))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_consecutive_artifact_gradient_values_are_evaluated"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("artifact_aware_direction_contract_sha256") != ARTIFACT_AWARE_CONTRACT_SHA256
        or identity.get("artifact_aware_direction_result_sha256") != ARTIFACT_AWARE_RESULT_SHA256
        or identity.get("persistent_contract_sha256") != PERSISTENT_CONTRACT_SHA256
        or identity.get("persistent_schedule_sha256") != SCHEDULE_SHA256
        or schedule.sha256 != SCHEDULE_SHA256
        or int(identity.get("trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or tuple(int(value) for value in data.get("logical_updates", ())) != LOGICAL_UPDATES
        or observed_group_updates != TRAIN_PANEL_GROUP_UPDATES
        or any(len(panel) != 5 for panel in panels)
        or any(len(artifacts) != 40 or len(set(artifacts)) != 40 for artifacts in flattened_panels)
        or len(all_artifacts) != 120
        or len(set(all_artifacts)) != 120
        or int(data.get("train_only_artifact_count_per_logical_update", -1)) != 40
        or int(data.get("unique_train_only_artifact_count", -1)) != 120
        or int(data.get("teacher_days_per_direct_gpp_artifact", -1)) != 208
        or monitor_updates != MONITOR_UPDATES
        or not set(all_artifacts).isdisjoint(monitor_updates)
        or data.get("scheduled_group_is_first_group_for_each_logical_update") is not True
        or data.get("artifact_panels_overlap_across_logical_updates") is not False
        or data.get("fixed_monitor_disjoint_from_all_train_only_panels") is not True
        or data.get("fixed_monitor_excluded_from_directions_and_scale_selection") is not True
        or reconstruction.get("first_smoke_update_repeats_predecessor_logical_update_and_panel") is not True
        or reconstruction.get("cross_gpu_bit_identity_required") is not False
        or int(direction.get("gradient_constraint_count_per_logical_update", -1)) != 43
        or int(direction.get("direct_gpp_artifact_constraint_count_per_logical_update", -1)) != 40
        or tuple(direction.get("scheduled_non_gpp_source_registry", ())) != SCHEDULED_NON_GPP_SOURCES
        or direction.get("direction_recomputed_after_each_accepted_step") is not True
        or direction.get("previous_simplex_weights_reused") is not False
        or direction.get("each_gradient_unit_normalized") is not True
        or float(direction.get("minimum_constraint_cosine", -1.0)) != 1.0e-4
        or float(direction.get("full_parameter_step_l2", -1.0)) != 0.007758973351050004
        or direction.get("optimizer_or_momentum_state_used") is not False
        or tuple(float(value) for value in selection.get("displacement_scales_descending", ())) != DISPLACEMENT_SCALES
        or selection.get("stop_after_first_train_only_passing_scale") is not True
        or selection.get("stop_smoke_after_first_logical_update_without_a_passing_scale") is not True
        or float(selection.get("each_train_only_artifact_direct_gpp_relative_change_maximum", -1.0)) != 1.0e-6
        or selection.get("monitor_values_excluded_from_selection") is not True
        or monitor.get("begins_only_after_all_three_train_only_updates_are_selected") is not True
        or monitor.get("individual_artifact_changes_reported_but_not_gated") is not True
        or monitor.get("monitor_not_read_when_train_only_continuity_stops") is not True
    ):
        raise ValueError("coupled GPP artifact-aware continuity contract drift")
    if any(bool(value) for value in retention.values()):
        raise ValueError("coupled GPP artifact-aware continuity smoke opens a retention path")
    if (
        int(execution.get("maximum_in_memory_direct_parameter_updates", -1)) != len(LOGICAL_UPDATES)
        or execution.get("free_gln01_allowed") is not True
        or execution.get("paid_partition_authorized") is not False
        or int(execution.get("timeout_minutes", -1)) != 240
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
        raise ValueError("coupled GPP artifact-aware continuity execution boundary drift")
    return CoupledGPPArtifactAwareContinuityContract(payload, source, observed)


__all__ = [
    "ARTIFACT_AWARE_RESULT_SHA256",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "LOGICAL_UPDATES",
    "SCHEMA_VERSION",
    "TRAIN_PANEL_GROUP_UPDATES",
    "CoupledGPPArtifactAwareContinuityContract",
    "classify_artifact_aware_continuity",
    "load_coupled_gpp_artifact_aware_continuity_contract",
    "monitor_trajectory_passed",
]
