"""Contract and monitor gates for bounded artifact-aware persistent training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .target_oriented_coupled_gpp_artifact_aware_continuity import (
    FROZEN_CONTRACT_SHA256 as CONTINUITY_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_artifact_aware_continuity import (
    _artifact_sources,
)
from .target_oriented_coupled_gpp_artifact_aware_direction import SCHEDULED_NON_GPP_SOURCES
from .target_oriented_coupled_gpp_direct_generalization import MONITOR_UPDATES
from .target_oriented_coupled_gpp_direct_step_transfer import DISPLACEMENT_SCALES
from .target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    TRAINABLE_PARAMETER_COUNT,
)
from .target_oriented_coupled_gpp_persistent_scale_training import (
    ALL_REPORTED_KEYS,
    SCHEDULE_SHA256,
    build_persistent_scale_training_schedule,
)
from .target_oriented_coupled_gpp_persistent_scale_training import (
    FROZEN_CONTRACT_SHA256 as PARENT_PERSISTENT_CONTRACT_SHA256,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_artifact_aware_persistent_training_v1"
FROZEN_CONTRACT_SHA256 = "3826d3584cf054bff0d08f0290e3bb20df2df90b3de47da8d9ffa6abc93d6578"
CONTINUITY_RESULT_SHA256 = "aa326bb4cfb237528cb0785e7d03d7f440f91647bcab4959d0334afc0a452d69"
PARENT_PERSISTENT_RESULT_SHA256 = "764b14e169345ba119407b9e3dcf3a7e9141900c2710c8afa2775dd0fe881168"
PREPARED_MANIFEST_SHA256 = "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
LOGICAL_UPDATES = (2, 3, 4, 5, 6, 7)
MILESTONES = (4, 7)
TRAIN_PANEL_GROUP_UPDATES = (
    (2, 11, 12, 13, 14),
    (3, 15, 16, 17, 18),
    (4, 19, 20, 21, 22),
    (5, 23, 24, 25, 26),
    (6, 27, 28, 29, 30),
    (7, 31, 32, 33, 34),
)
PRIMARY_SCIENCE_KEYS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
    "heterotrophic_respiration",
    "autotrophic_respiration",
    "npp",
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "total_live_biomass_carbon",
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPArtifactAwarePersistentTrainingContract:
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
        configured = self.payload["data"]["train_only_schedule_group_updates_by_logical_update"]
        return tuple(tuple(int(value) for value in configured[str(update)]) for update in self.logical_updates)

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
        return tuple(int(value) for value in self.payload["monitor"]["artifact_updates"])


def _relative_change(before: float, after: float) -> float:
    first = float(before)
    second = float(after)
    if not np.isfinite(first) or first <= 0.0 or not np.isfinite(second) or second < 0.0:
        raise ValueError("artifact-aware persistent monitor value is invalid")
    return (second - first) / first


def primary_science_scores(panel: Mapping[str, Any]) -> Mapping[str, float]:
    horizons = panel.get("absolute_key_scores", {})
    if set(horizons) != {"7", "30"}:
        raise ValueError("artifact-aware persistent monitor horizons drift")
    scores = {}
    for horizon in ("7", "30"):
        if set(horizons[horizon]) != set(ALL_REPORTED_KEYS):
            raise ValueError("artifact-aware persistent monitor key registry drift")
        for name in PRIMARY_SCIENCE_KEYS:
            value = float(horizons[horizon][name])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("artifact-aware persistent primary science value is invalid")
            scores[f"{horizon}.{name}"] = value
    return scores


def artifact_aware_monitor_gate(
    contract: CoupledGPPArtifactAwarePersistentTrainingContract,
    *,
    logical_update: int,
    initial: Mapping[str, Any],
    current: Mapping[str, Any],
) -> Mapping[str, Any]:
    update = int(logical_update)
    if update not in MILESTONES:
        raise ValueError("artifact-aware persistent monitor evaluated outside a milestone")
    initial_sources = initial["source_mean_losses"]
    current_sources = current["source_mean_losses"]
    if set(initial_sources) != set(GRADIENT_SOURCES) or set(current_sources) != set(GRADIENT_SOURCES):
        raise ValueError("artifact-aware persistent source registry drift")
    source_changes = {name: _relative_change(initial_sources[name], current_sources[name]) for name in GRADIENT_SOURCES}
    initial_primary = primary_science_scores(initial)
    current_primary = primary_science_scores(current)
    primary_changes = {name: _relative_change(initial_primary[name], current_primary[name]) for name in initial_primary}
    initial_macro = float(np.mean(np.asarray(tuple(initial_primary.values()), dtype=np.float64)))
    current_macro = float(np.mean(np.asarray(tuple(current_primary.values()), dtype=np.float64)))
    primary_macro_change = _relative_change(initial_macro, current_macro)
    maximum_primary_change = max(primary_changes.values())
    monitor = contract.payload["monitor"]
    common_checks = {
        "physical_hard_gate": current.get("hard_gate_passed") is True,
        "all_sources_finite": all(np.isfinite(float(value)) for value in current_sources.values()),
        "all_primary_scores_finite": all(np.isfinite(float(value)) for value in current_primary.values()),
    }
    if update != MILESTONES[-1]:
        checks = common_checks | {
            "direct_gpp_regression_within_intermediate_cap": source_changes[DIRECT_SOURCE]
            <= float(monitor["intermediate_direct_gpp_mean_relative_regression_maximum"]),
            "primary_macro_regression_within_intermediate_cap": primary_macro_change
            <= float(monitor["intermediate_primary_science_macro_relative_regression_maximum"]),
            "every_primary_key_horizon_within_intermediate_cap": maximum_primary_change
            <= float(monitor["intermediate_any_primary_key_horizon_relative_regression_maximum"]),
        }
        passed = all(checks.values())
        final = False
    else:
        non_gpp_sources = tuple(name for name in GRADIENT_SOURCES if name != DIRECT_SOURCE)
        checks = common_checks | {
            "direct_gpp_mean_strictly_improves": source_changes[DIRECT_SOURCE] < 0.0,
            "primary_science_macro_strictly_improves": primary_macro_change < 0.0,
            "seven_day_gpp_not_worse": primary_changes["7.gpp"] <= 0.0,
            "thirty_day_gpp_not_worse": primary_changes["30.gpp"] <= 0.0,
            "non_gpp_source_means_within_final_cap": max(source_changes[name] for name in non_gpp_sources)
            <= float(monitor["final_non_gpp_source_mean_relative_regression_maximum"]),
            "every_primary_key_horizon_within_final_cap": maximum_primary_change
            <= float(monitor["final_any_primary_key_horizon_relative_regression_maximum"]),
        }
        passed = all(checks.values())
        final = True
    return {
        "logical_update": update,
        "final_milestone": final,
        "checks": checks,
        "passed": passed,
        "stop_training": not passed,
        "source_relative_changes": source_changes,
        "primary_science_initial_macro": initial_macro,
        "primary_science_current_macro": current_macro,
        "primary_science_macro_relative_change": primary_macro_change,
        "primary_key_horizon_relative_changes": primary_changes,
        "maximum_primary_key_horizon_relative_change": maximum_primary_change,
    }


def load_coupled_gpp_artifact_aware_persistent_training_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPArtifactAwarePersistentTrainingContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    starting = payload.get("starting_state", {})
    direction = payload.get("direction", {})
    selection = payload.get("scale_selection", {})
    monitor = payload.get("monitor", {})
    persistence = payload.get("persistence", {})
    execution = payload.get("execution", {})
    schedule = build_persistent_scale_training_schedule()
    configured = data.get("train_only_schedule_group_updates_by_logical_update", {})
    group_updates = tuple(tuple(int(value) for value in configured.get(str(update), ())) for update in LOGICAL_UPDATES)
    panels = tuple(
        tuple(tuple(int(value) for value in schedule.groups[index - 1]) for index in indexes)
        for indexes in group_updates
    )
    flattened = tuple(tuple(value for group in panel for value in group) for panel in panels)
    all_artifacts = tuple(value for panel in flattened for value in panel)
    monitor_updates = tuple(int(value) for value in monitor.get("artifact_updates", ()))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_exact_start_state_is_serialized"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("continuity_contract_sha256") != CONTINUITY_CONTRACT_SHA256
        or identity.get("continuity_result_sha256") != CONTINUITY_RESULT_SHA256
        or identity.get("parent_persistent_contract_sha256") != PARENT_PERSISTENT_CONTRACT_SHA256
        or identity.get("parent_persistent_schedule_sha256") != SCHEDULE_SHA256
        or identity.get("parent_persistent_result_sha256") != PARENT_PERSISTENT_RESULT_SHA256
        or identity.get("prepared_manifest_sha256") != PREPARED_MANIFEST_SHA256
        or int(identity.get("candidate_parameter_count", -1)) != 109_464_359
        or int(identity.get("trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(identity.get("fixed_parameter_count", -1)) != 93_504_164
        or schedule.sha256 != SCHEDULE_SHA256
        or tuple(int(value) for value in data.get("logical_updates", ())) != LOGICAL_UPDATES
        or group_updates != TRAIN_PANEL_GROUP_UPDATES
        or any(len(panel) != 5 for panel in panels)
        or any(len(values) != 40 or len(set(values)) != 40 for values in flattened)
        or len(all_artifacts) != 240
        or len(set(all_artifacts)) != 240
        or not set(all_artifacts).isdisjoint(monitor_updates)
        or int(data.get("train_only_artifact_count_per_logical_update", -1)) != 40
        or int(data.get("unique_train_only_artifact_count", -1)) != 240
        or int(data.get("teacher_days_per_direct_gpp_artifact", -1)) != 208
        or data.get("scheduled_group_is_first_group_for_each_logical_update") is not True
        or data.get("artifact_panels_overlap_across_logical_updates") is not False
        or data.get("fixed_monitor_disjoint_from_all_train_only_panels") is not True
        or monitor_updates != MONITOR_UPDATES
        or tuple(int(value) for value in monitor.get("milestone_logical_updates", ())) != MILESTONES
        or tuple(monitor.get("reported_primary_science_keys", ())) != PRIMARY_SCIENCE_KEYS
        or starting.get("fresh_run_reconstructs_once") is not True
        or starting.get("persistent_update_1_replayed_before_snapshot") is not True
        or starting.get("exact_parameter_only_start_snapshot_written_before_monitor") is not True
        or starting.get("snapshot_roundtrip_required") is not True
        or starting.get("resume_uses_initialization_only_context_without_historical_update_replay") is not True
        or starting.get("independent_reconstruction_used_after_snapshot_exists") is not False
        or starting.get("fixed_parameters_serialized") is not False
        or starting.get("optimizer_state_serialized") is not False
        or int(direction.get("gradient_constraint_count_per_logical_update", -1)) != 43
        or int(direction.get("direct_gpp_artifact_constraint_count_per_logical_update", -1)) != 40
        or direction.get("direction_recomputed_after_each_accepted_step") is not True
        or direction.get("previous_simplex_weights_reused") is not False
        or direction.get("each_gradient_unit_normalized") is not True
        or float(direction.get("minimum_constraint_cosine", -1.0)) != 1.0e-4
        or float(direction.get("full_parameter_step_l2", -1.0)) != 0.007758973351050004
        or direction.get("optimizer_or_momentum_state_used") is not False
        or tuple(float(value) for value in selection.get("displacement_scales_descending", ())) != DISPLACEMENT_SCALES
        or selection.get("stop_after_first_train_only_passing_scale") is not True
        or selection.get("stop_training_after_first_logical_update_without_a_passing_scale") is not True
        or float(selection.get("each_train_only_artifact_direct_gpp_relative_change_maximum", -1.0)) != 1.0e-6
        or selection.get("monitor_values_excluded_from_direction_and_scale_selection") is not True
        or monitor.get("initial_monitor_evaluated_only_after_exact_start_snapshot_is_written") is not True
        or monitor.get("monitor_never_selects_direction_or_scale") is not True
        or monitor.get("physical_hard_gate_required") is not True
        or monitor.get("fresh_validation_used") is not False
        or persistence.get("start_snapshot_immutable") is not True
        or persistence.get("recovery_written_after_initial_monitor") is not True
        or persistence.get("recovery_written_after_each_accepted_update") is not True
        or persistence.get("identity_exact_resume_required") is not True
        or persistence.get("atomic_write_required") is not True
        or persistence.get("trainable_parameter_tree_persisted") is not True
        or persistence.get("fixed_parameters_persisted") is not False
        or persistence.get("optimizer_state_persisted") is not False
        or persistence.get("checkpoint_only_after_final_monitor_gate") is not True
        or persistence.get("checkpoint_promotion_eligible") is not False
        or persistence.get("recovery_promotion_eligible") is not False
    ):
        raise ValueError("coupled GPP artifact-aware persistent training contract drift")
    if any(
        bool(data[name])
        for name in (
            "fresh_validation_values_allowed",
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "sealed_test_allowed",
        )
    ) or (
        execution.get("free_gln01_training_smoke_allowed") is not True
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
        raise ValueError("coupled GPP artifact-aware persistent execution boundary drift")
    return CoupledGPPArtifactAwarePersistentTrainingContract(payload, source, observed)


__all__ = [
    "CONTINUITY_RESULT_SHA256",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "LOGICAL_UPDATES",
    "MILESTONES",
    "PREPARED_MANIFEST_SHA256",
    "PRIMARY_SCIENCE_KEYS",
    "SCHEMA_VERSION",
    "TRAIN_PANEL_GROUP_UPDATES",
    "CoupledGPPArtifactAwarePersistentTrainingContract",
    "artifact_aware_monitor_gate",
    "load_coupled_gpp_artifact_aware_persistent_training_contract",
    "primary_science_scores",
]
