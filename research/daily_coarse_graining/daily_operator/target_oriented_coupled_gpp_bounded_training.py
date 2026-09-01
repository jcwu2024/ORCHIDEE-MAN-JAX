"""Frozen schedule and stop rules for bounded coupled-GPP training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
    TRAINABLE_PARAMETER_COUNT,
)
from .target_oriented_coupled_gpp_optimizer_preflight import (
    four_source_macro_objectives,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_bounded_training_v1"
FROZEN_CONTRACT_SHA256 = "5378fe9908c272b8651d8b2087e56e4d7751ead18c7395794bbe2361f6699cca"
SCHEDULE_SHA256 = "cb5e2ab12beecd558efe1aaf8082da54567b63631ce0b982e03934067b36c04e"
CALIBRATION_UPDATES = (352, 353, 354, 355, 356, 357, 359, 360)
MONITOR_UPDATES = (251, 358, 465, 572, 679, 786, 893, 1000)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_bounded_training_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPTrainingSchedule:
    groups: tuple[tuple[int, ...], ...]
    sha256: str


@dataclass(frozen=True)
class CoupledGPPBoundedTrainingContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def coefficients(self) -> Mapping[str, float]:
        return {str(name): float(value) for name, value in self.payload["objective"]["coefficients"].items()}

    @property
    def clip_threshold(self) -> float:
        return float(self.payload["optimizer"]["global_gradient_norm_clip"]["threshold"])

    @property
    def learning_rate(self) -> float:
        return float(self.payload["optimizer"]["learning_rate"])

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["monitor"]["artifact_updates"])


def eligible_training_artifacts() -> tuple[int, ...]:
    excluded = frozenset(CALIBRATION_UPDATES) | frozenset(MONITOR_UPDATES)
    return tuple(update for update in range(251, 1001) if update not in excluded)


def build_coupled_gpp_training_schedule(
    *,
    seed: int = 20260901,
    maximum_updates: int = 8,
    macro_batch_size: int = 8,
) -> CoupledGPPTrainingSchedule:
    if maximum_updates <= 0 or macro_batch_size <= 0:
        raise ValueError("coupled GPP training schedule dimensions must be positive")
    required = int(maximum_updates) * int(macro_batch_size)
    candidates = np.asarray(eligible_training_artifacts(), dtype=np.int64)
    if required > candidates.size:
        raise ValueError("coupled GPP training schedule exceeds eligible artifacts")
    order = np.random.Generator(np.random.PCG64(int(seed))).permutation(candidates)
    selected = tuple(int(value) for value in order[:required])
    groups = tuple(tuple(selected[start : start + macro_batch_size]) for start in range(0, required, macro_batch_size))
    canonical = {"groups": groups}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CoupledGPPTrainingSchedule(groups=groups, sha256=digest)


def source_changes(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> Mapping[str, Mapping[str, float]]:
    if set(before) != set(GRADIENT_SOURCES) or set(after) != set(GRADIENT_SOURCES):
        raise ValueError("coupled GPP source-loss registry drift")
    result = {}
    for name in GRADIENT_SOURCES:
        first = float(before[name])
        second = float(after[name])
        if not np.isfinite(first) or first <= 0.0 or not np.isfinite(second) or second < 0.0:
            raise ValueError("coupled GPP source loss is invalid")
        result[name] = {
            "before": first,
            "after": second,
            "relative_change": (second - first) / first,
        }
    return result


def finite_step_accepted(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    coefficients: Mapping[str, float],
    relative_tolerance: float = 1.0e-6,
) -> bool:
    changes = source_changes(before, after)
    before_joint = four_source_macro_objectives(before, coefficients)["joint"]
    after_joint = four_source_macro_objectives(after, coefficients)["joint"]
    return bool(
        np.isfinite(before_joint)
        and np.isfinite(after_joint)
        and after_joint < before_joint
        and all(float(changes[name]["relative_change"]) <= float(relative_tolerance) for name in GRADIENT_SOURCES)
    )


def monitor_continuation_accepted(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    coefficients: Mapping[str, float],
) -> bool:
    return finite_step_accepted(
        before,
        after,
        coefficients=coefficients,
        relative_tolerance=0.0,
    )


def selected_milestone(monitors: Sequence[Mapping[str, Any]]) -> int:
    selected = 0
    stopped = False
    for expected_update, monitor in zip((4, 8), monitors, strict=False):
        if int(monitor.get("optimizer_update", -1)) != expected_update:
            raise ValueError("coupled GPP milestone-monitor order drift")
        if stopped:
            raise ValueError("coupled GPP monitor continues after failed milestone")
        if bool(monitor.get("continuation_gate_passed")):
            selected = expected_update
        else:
            stopped = True
    if len(monitors) > 2:
        raise ValueError("coupled GPP milestone-monitor count drift")
    return selected


def load_coupled_gpp_bounded_training_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPBoundedTrainingContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_runner_qualification_and_paid_resource_review"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("coupled GPP bounded-training contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    objective = payload["objective"]
    optimizer = payload["optimizer"]
    finite_step = payload["finite_step_acceptance"]
    stage = payload["bounded_stage"]
    monitor = payload["monitor"]
    execution = payload["execution"]
    schedule = build_coupled_gpp_training_schedule(
        seed=int(data["training_schedule"]["seed"]),
        maximum_updates=int(data["maximum_optimizer_updates"]),
        macro_batch_size=int(data["macro_batch_artifact_count"]),
    )
    coefficients = {str(name): float(value) for name, value in objective["coefficients"].items()}
    if (
        identity["optimizer_backoff_report_sha256"]
        != "6abb472de28e7eee1683fac2bcd52d265f9d0ce1bdafd3c0cd43883978f751c5"
        or int(identity["candidate_parameter_count"]) != 109_464_359
        or int(identity["trainable_parameter_count"]) != TRAINABLE_PARAMETER_COUNT
        or int(identity["fixed_parameter_count"]) != 93_504_164
        or tuple(int(value) for value in data["objective_calibration_artifacts"]) != CALIBRATION_UPDATES
        or tuple(int(value) for value in data["monitor_artifacts"]) != MONITOR_UPDATES
        or int(data["prepared_artifact_start"]) != 251
        or int(data["prepared_artifact_end"]) != 1000
        or int(data["training_candidate_artifact_count"]) != len(eligible_training_artifacts())
        or int(data["macro_batch_artifact_count"]) != 8
        or int(data["maximum_optimizer_updates"]) != 8
        or int(data["training_artifact_presentations"]) != 64
        or data["training_schedule"]["materialized_schedule_sha256"] != schedule.sha256
        or schedule.sha256 != SCHEDULE_SHA256
        or len({value for group in schedule.groups for value in group}) != 64
        or any(value in MONITOR_UPDATES or value in CALIBRATION_UPDATES for group in schedule.groups for value in group)
    ):
        raise ValueError("coupled GPP bounded-training identity or schedule drift")
    if (
        tuple(objective["gradient_sources"]) != GRADIENT_SOURCES
        or coefficients
        != {
            DIRECT_SOURCE: 1.0,
            STATE_SOURCE: 0.6524275555272667,
            SEVEN_SOURCE: 0.4072260558556481,
            THIRTY_SOURCE: 0.28249951035415655,
        }
        or optimizer["state_source"] != "fresh_for_both_trainable_families"
        or int(optimizer["initial_optimizer_step"]) != 0
        or int(optimizer["maximum_optimizer_step"]) != 8
        or float(optimizer["learning_rate"]) != 2.0e-6
        or float(optimizer["global_gradient_norm_clip"]["threshold"]) != 12.808193184117403
        or float(finite_step["all_four_macro_source_losses_must_not_increase_relative_tolerance"]) != 1.0e-6
        or tuple(int(value) for value in stage["milestone_updates"]) != (4, 8)
        or int(stage["recovery_interval_accepted_updates"]) != 1
        or tuple(int(value) for value in monitor["artifact_updates"]) != MONITOR_UPDATES
        or float(monitor["all_four_source_losses_must_not_increase_relative_tolerance"]) != 0.0
    ):
        raise ValueError("coupled GPP bounded-training objective or stop-rule drift")
    if any(
        bool(data[name])
        for name in (
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "fresh_validation_values_allowed",
            "sealed_test_allowed",
        )
    ) or any(
        bool(execution[name])
        for name in (
            "paid_training_authorized_by_contract_alone",
            "free_gln01_persistent_training_allowed",
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "fresh_validation_read_authorized",
            "365_day_rollout_authorized",
            "gate_d2_authorized",
            "sealed_test_authorized",
            "product_acceptance_authorized",
            "cross_day_neural_hidden_state",
        )
    ):
        raise ValueError("coupled GPP bounded-training opens a forbidden path")
    if int(execution["forcing_records_reconstructed"]) != 0 or int(execution["subdaily_state_scan_steps"]) != 0:
        raise ValueError("coupled GPP bounded-training inference boundary drift")
    return CoupledGPPBoundedTrainingContract(payload, source, observed)


__all__ = [
    "CALIBRATION_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "MONITOR_UPDATES",
    "SCHEDULE_SHA256",
    "CoupledGPPBoundedTrainingContract",
    "CoupledGPPTrainingSchedule",
    "build_coupled_gpp_training_schedule",
    "eligible_training_artifacts",
    "finite_step_accepted",
    "load_coupled_gpp_bounded_training_contract",
    "monitor_continuation_accepted",
    "selected_milestone",
    "source_changes",
]
