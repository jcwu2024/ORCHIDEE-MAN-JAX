"""Frozen schedule and gates for persistent scale-selecting coupled-GPP training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .target_oriented_coupled_gpp_bounded_training import (
    MONITOR_UPDATES,
    build_coupled_gpp_training_schedule,
    eligible_training_artifacts,
)
from .target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
    TRAINABLE_PARAMETER_COUNT,
)
from .target_oriented_coupled_gpp_scale_selected_transfer import (
    FROZEN_CONTRACT_SHA256 as TRANSFER_CONTRACT_SHA256,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_persistent_scale_training_v1"
FROZEN_CONTRACT_SHA256 = "1742d9d19a6b8e8d91331723bf06047a17fa5ce5d31cc528987205a838d8b5e4"
TRANSFER_RESULT_SHA256 = "b77e92fefdce5a427c471118f8b003bf69f01abb23c184ea86ba5eef54e5f625"
SCHEDULE_SHA256 = "c79c5f2e125706d73b306ccd5aa23582a062b6f387d55224edee52666492548d"
MAXIMUM_UPDATES = 100
MACRO_BATCH_SIZE = 8
MILESTONES = (10, 25, 50, 100)
CHECKPOINT_DEADLINE = 25
REPORTED_FLUX_KEYS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
    "heterotrophic_respiration",
)
REPORTED_STOCK_KEYS = (
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "nonliving_ecosystem_carbon",
)
DERIVED_REPORTED_KEYS = (
    "autotrophic_respiration",
    "npp",
    "nep",
    "nee",
    "total_live_biomass_carbon",
    "ecosystem_carbon",
)
ALL_REPORTED_KEYS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
    "autotrophic_respiration",
    "npp",
    "heterotrophic_respiration",
    "nep",
    "nee",
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "total_live_biomass_carbon",
    "nonliving_ecosystem_carbon",
    "ecosystem_carbon",
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_persistent_scale_training_v1.json"
)


@dataclass(frozen=True)
class PersistentScaleTrainingSchedule:
    groups: tuple[tuple[int, ...], ...]
    pool: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class CoupledGPPPersistentScaleTrainingContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["scale_selection"]["scales_descending"])

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["monitor"]["artifact_updates"])


def persistent_training_pool() -> tuple[int, ...]:
    bootstrap = {artifact for group in build_coupled_gpp_training_schedule().groups[:7] for artifact in group}
    return tuple(value for value in eligible_training_artifacts() if value not in bootstrap)


def build_persistent_scale_training_schedule(
    *,
    seed: int = 20260903,
    maximum_updates: int = MAXIMUM_UPDATES,
    macro_batch_size: int = MACRO_BATCH_SIZE,
) -> PersistentScaleTrainingSchedule:
    required = int(maximum_updates) * int(macro_batch_size)
    if required <= 0 or int(macro_batch_size) <= 0:
        raise ValueError("persistent coupled GPP schedule dimensions must be positive")
    pool = persistent_training_pool()
    if not pool:
        raise ValueError("persistent coupled GPP schedule pool is empty")
    selected: list[int] = []
    epoch = 0
    while len(selected) < required:
        order = list(
            np.random.Generator(np.random.PCG64(int(seed) + epoch)).permutation(np.asarray(pool, dtype=np.int64))
        )
        remainder = len(selected) % int(macro_batch_size)
        if remainder:
            blocked = set(selected[-remainder:])
            order = [value for value in order if int(value) not in blocked] + [
                value for value in order if int(value) in blocked
            ]
        selected.extend(int(value) for value in order)
        epoch += 1
    selected = selected[:required]
    groups = tuple(
        tuple(selected[start : start + int(macro_batch_size)]) for start in range(0, required, int(macro_batch_size))
    )
    if any(len(set(group)) != len(group) for group in groups):
        raise RuntimeError("persistent coupled GPP schedule repeats within a macro batch")
    canonical = {"groups": groups, "pool": pool}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PersistentScaleTrainingSchedule(groups=groups, pool=pool, sha256=digest)


def monitor_gate_decision(
    contract: CoupledGPPPersistentScaleTrainingContract,
    *,
    persistent_update: int,
    current_sources: Mapping[str, float],
    initial_gpp_scores: Mapping[str, float],
    current_gpp_scores: Mapping[str, float],
    physical_gate_passed: bool,
    checkpoint_already_selected: bool,
) -> Mapping[str, Any]:
    monitor = contract.payload["monitor"]
    pre5 = {str(name): float(value) for name, value in monitor["pre_update_5_source_baseline"].items()}
    after7 = {str(name): float(value) for name, value in monitor["after_update_7_source_initial"].items()}
    continuation_checks = {
        "physical_gate": bool(physical_gate_passed),
        "direct_gpp_not_worse_than_after_update_7": float(current_sources[DIRECT_SOURCE]) <= after7[DIRECT_SOURCE],
        "state_retains_pre_update_5_improvement": float(current_sources[STATE_SOURCE]) <= pre5[STATE_SOURCE],
        "seven_day_retains_pre_update_5_improvement": float(current_sources[SEVEN_SOURCE]) <= pre5[SEVEN_SOURCE],
        "thirty_day_retains_pre_update_5_improvement": float(current_sources[THIRTY_SOURCE]) <= pre5[THIRTY_SOURCE],
        "seven_day_cumulative_gpp_not_worse": float(current_gpp_scores["7"]) <= float(initial_gpp_scores["7"]),
        "thirty_day_cumulative_gpp_not_worse": float(current_gpp_scores["30"]) <= float(initial_gpp_scores["30"]),
    }
    continuation_passed = all(continuation_checks.values())
    checkpoint_checks = dict(
        continuation_checks,
        direct_gpp_recovered_to_pre_update_5=float(current_sources[DIRECT_SOURCE]) <= pre5[DIRECT_SOURCE],
    )
    checkpoint_passed = all(checkpoint_checks.values())
    deadline_missed = (
        int(persistent_update) >= CHECKPOINT_DEADLINE
        and not bool(checkpoint_already_selected)
        and not checkpoint_passed
    )
    stop = not continuation_passed or deadline_missed or (bool(checkpoint_already_selected) and not checkpoint_passed)
    return {
        "continuation_checks": continuation_checks,
        "continuation_passed": continuation_passed,
        "checkpoint_checks": checkpoint_checks,
        "checkpoint_passed": checkpoint_passed,
        "checkpoint_deadline_missed": deadline_missed,
        "stop_stage": stop,
    }


def load_coupled_gpp_persistent_scale_training_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPPersistentScaleTrainingContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    direction = payload.get("direction", {})
    selection = payload.get("scale_selection", {})
    monitor = payload.get("monitor", {})
    objective = payload.get("objective", {})
    persistence = payload.get("persistence", {})
    admission = payload.get("admission", {})
    execution = payload.get("execution", {})
    schedule = build_persistent_scale_training_schedule(
        seed=int(data.get("schedule_seed", -1)),
        maximum_updates=int(data.get("maximum_persistent_updates", -1)),
        macro_batch_size=int(data.get("macro_batch_artifact_count", -1)),
    )
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_runner_qualification_and_paid_resource_review"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("transfer_contract_sha256") != TRANSFER_CONTRACT_SHA256
        or identity.get("transfer_result_sha256") != TRANSFER_RESULT_SHA256
        or int(identity.get("candidate_parameter_count", -1)) != 109_464_359
        or int(identity.get("trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(identity.get("fixed_parameter_count", -1)) != 93_504_164
        or int(data.get("training_pool_artifact_count", -1)) != len(persistent_training_pool())
        or schedule.sha256 != SCHEDULE_SHA256
        or data.get("materialized_schedule_sha256") != SCHEDULE_SHA256
        or len(schedule.groups) != MAXIMUM_UPDATES
        or tuple(int(value) for value in data.get("bootstrap_selected_updates", ())) != (2, 3, 4, 5, 6, 7)
        or data.get("exclude_all_artifacts_consumed_by_updates_1_through_7") is not True
        or int(data.get("training_artifact_presentations", -1)) != MAXIMUM_UPDATES * MACRO_BATCH_SIZE
        or int(data.get("direct_gpp_teacher_days_per_update", -1)) != 1664
        or int(data.get("one_day_state_transitions_per_update", -1)) != 256
        or int(data.get("seven_day_windows_per_update", -1)) != 64
        or int(data.get("thirty_day_windows_per_update", -1)) != 32
        or tuple(objective.get("gradient_sources", ())) != GRADIENT_SOURCES
        or objective.get("coefficients")
        != {
            DIRECT_SOURCE: 1.0,
            STATE_SOURCE: 0.6524275555272667,
            SEVEN_SOURCE: 0.4072260558556481,
            THIRTY_SOURCE: 0.28249951035415655,
        }
        or objective.get("source_loss_and_gradient_aggregation") != "arithmetic_mean_across_eight_artifacts"
        or objective.get("coefficient_tuning_allowed") is not False
        or direction.get("source_registry") != list(GRADIENT_SOURCES)
        or direction.get("solver") != "minimum_norm_simplex_over_unit_source_gradients"
        or direction.get("recompute_after_every_accepted_update") is not True
        or tuple(float(value) for value in selection.get("scales_descending", ()))
        != (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
        or selection.get("largest_passing_scale_selected") is not True
        or tuple(int(value) for value in monitor.get("milestone_updates", ())) != MILESTONES
        or int(monitor.get("checkpoint_recovery_deadline_update", -1)) != CHECKPOINT_DEADLINE
        or tuple(int(value) for value in monitor.get("artifact_updates", ())) != MONITOR_UPDATES
        or set(monitor.get("pre_update_5_source_baseline", {})) != set(GRADIENT_SOURCES)
        or set(monitor.get("after_update_7_source_initial", {})) != set(GRADIENT_SOURCES)
        or monitor.get("per_key_science_required") is not True
        or tuple(monitor.get("reported_flux_keys", ())) != REPORTED_FLUX_KEYS
        or tuple(monitor.get("reported_stock_keys", ())) != REPORTED_STOCK_KEYS
        or tuple(monitor.get("derived_reported_keys", ())) != DERIVED_REPORTED_KEYS
        or monitor.get("cumulative_gpp_guard_horizons") != [7, 30]
        or monitor.get("stop_if_no_checkpoint_by_update_25") is not True
        or monitor.get("stop_after_selected_checkpoint_if_later_milestone_loses_checkpoint_gate") is not True
        or monitor.get("physical_hard_gate_required") is not True
        or monitor.get("validation_used") is not False
        or persistence.get("recovery_written_after_each_accepted_update") is not True
        or persistence.get("recovery_promotion_eligible") is not False
        or persistence.get("checkpoint_only_at_passing_milestone") is not True
        or tuple(int(value) for value in persistence.get("checkpoint_updates", ())) != MILESTONES
        or persistence.get("checkpoint_promotion_eligible") is not False
        or persistence.get("fixed_parameters_persisted") is not False
        or persistence.get("trainable_parameter_tree_persisted") is not True
        or persistence.get("optimizer_state_persisted") is not False
        or persistence.get("atomic_write_required") is not True
        or persistence.get("identity_exact_resume_required") is not True
        or admission.get("free_gln01_reconstructs_update_7_state") is not True
        or admission.get("detailed_initial_monitor_required") is not True
        or admission.get("one_discarded_persistent_update_required") is not True
        or admission.get("temporary_recovery_roundtrip_required") is not True
        or admission.get("retained_checkpoint_or_recovery") is not False
        or int(admission.get("timeout_minutes", -1)) != 120
    ):
        raise ValueError("persistent coupled GPP training contract drift")
    if any(
        bool(data[name])
        for name in (
            "fresh_validation_values_allowed",
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "sealed_test_allowed",
        )
    ) or (
        execution.get("free_gln01_runner_admission_allowed") is not True
        or execution.get("free_gln01_persistent_training_allowed") is not False
        or execution.get("paid_training_authorized_by_contract_alone") is not False
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
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
    ):
        raise ValueError("persistent coupled GPP training opens a forbidden path")
    return CoupledGPPPersistentScaleTrainingContract(payload, source, observed)


__all__ = [
    "ALL_REPORTED_KEYS",
    "CHECKPOINT_DEADLINE",
    "DEFAULT_CONTRACT_PATH",
    "DERIVED_REPORTED_KEYS",
    "FROZEN_CONTRACT_SHA256",
    "MACRO_BATCH_SIZE",
    "MAXIMUM_UPDATES",
    "MILESTONES",
    "REPORTED_FLUX_KEYS",
    "REPORTED_STOCK_KEYS",
    "SCHEDULE_SHA256",
    "TRANSFER_RESULT_SHA256",
    "CoupledGPPPersistentScaleTrainingContract",
    "PersistentScaleTrainingSchedule",
    "build_persistent_scale_training_schedule",
    "load_coupled_gpp_persistent_scale_training_contract",
    "monitor_gate_decision",
    "persistent_training_pool",
]
