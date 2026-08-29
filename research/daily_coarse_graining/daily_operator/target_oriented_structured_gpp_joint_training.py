"""Bounded batch-four joint training contract for structured GPP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_source_shaped_maintenance import contract_sha256
from .target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from .target_oriented_structured_gpp_training import SPECIALIST_PARAMETER_COUNT

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_training_v1"
FROZEN_CONTRACT_SHA256 = "168d44bdf8e1094ab4d683edaf8ebf576ff7dd27053d754419956cce30798832"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_joint_training_v1.json"
)
SCHEDULE_SHA256 = "770bae619e427820df2c6ad71c48bab3180a33bc1a1784f37e04680a7a455f70"
TRAIN_ONLY_PANEL_UPDATES = (251, 358, 465, 572, 679, 786, 893, 1000)


@dataclass(frozen=True)
class JointTrainingSchedule:
    groups_by_epoch: tuple[tuple[tuple[int, ...], ...], ...]
    dropped_by_epoch: tuple[tuple[int, ...], ...]
    sha256: str

    @property
    def flat_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(group for epoch in self.groups_by_epoch for group in epoch)


@dataclass(frozen=True)
class StructuredGPPJointTrainingContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def coefficients(self) -> Mapping[str, float]:
        return {
            str(name): float(value)
            for name, value in self.payload["objective"]["coefficients"].items()
        }

    @property
    def clip_threshold(self) -> float:
        return float(self.payload["optimizer"]["global_gradient_norm_clip"]["threshold"])

    @property
    def panel_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["epoch_panel"]["artifact_updates"])


def candidate_artifact_updates() -> tuple[int, ...]:
    return tuple(value for value in range(512, 764) if value not in {572, 679})


def build_joint_training_schedule(
    *,
    seed: int = 20260830,
    epochs: int = 4,
    batch_size: int = 4,
) -> JointTrainingSchedule:
    candidates = candidate_artifact_updates()
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("structured GPP joint schedule dimensions must be positive")
    used_count = len(candidates) - len(candidates) % batch_size
    if used_count <= 0:
        raise ValueError("structured GPP joint schedule has no full batch")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    groups_by_epoch = []
    dropped_by_epoch = []
    for _epoch in range(int(epochs)):
        order = tuple(
            int(value)
            for value in generator.permutation(np.asarray(candidates, dtype=np.int64))
        )
        used = order[:used_count]
        groups_by_epoch.append(
            tuple(
                tuple(used[start : start + batch_size])
                for start in range(0, used_count, batch_size)
            )
        )
        dropped_by_epoch.append(tuple(order[used_count:]))
    canonical = {
        "groups": groups_by_epoch,
        "dropped": dropped_by_epoch,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return JointTrainingSchedule(
        groups_by_epoch=tuple(groups_by_epoch),
        dropped_by_epoch=tuple(dropped_by_epoch),
        sha256=digest,
    )


def panel_source_changes(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> Mapping[str, Mapping[str, float]]:
    result = {}
    for name in GRADIENT_SOURCES:
        first = float(before[name])
        second = float(after[name])
        if not np.isfinite(first) or first <= 0.0 or not np.isfinite(second):
            raise ValueError("structured GPP joint panel loss is invalid")
        result[name] = {
            "before": first,
            "after": second,
            "relative_change": (second - first) / first,
        }
    return result


def epoch_panel_continuation_passed(changes: Mapping[str, Mapping[str, float]]) -> bool:
    return set(changes) == set(GRADIENT_SOURCES) and all(
        float(changes[name]["after"]) < float(changes[name]["before"])
        for name in GRADIENT_SOURCES
    )


def selected_epoch_from_panels(panels: Sequence[Mapping[str, Any]]) -> int:
    selected = 0
    stopped = False
    for expected_epoch, panel in enumerate(panels, start=1):
        if int(panel["epoch"]) != expected_epoch:
            raise ValueError("structured GPP joint epoch panel order drift")
        if stopped:
            raise ValueError("structured GPP joint epoch panel continues after failure")
        if bool(panel["continuation_gate_passed"]):
            selected = expected_epoch
        else:
            stopped = True
    return selected


def validate_joint_training_recovery_metadata(
    metadata: Mapping[str, Any],
    *,
    completed_joint_updates: int,
    updates_per_epoch: int = 62,
    initial_optimizer_step: int = 250,
) -> Mapping[str, Any]:
    completed = int(completed_joint_updates)
    per_epoch = int(updates_per_epoch)
    if completed < 0 or per_epoch <= 0:
        raise ValueError("structured GPP joint recovery progress is invalid")
    initial_panel = metadata.get("initial_joint_panel")
    epoch_panels = tuple(metadata.get("epoch_panels", ()))
    epoch_checkpoints = tuple(metadata.get("epoch_checkpoints", ()))
    if not isinstance(initial_panel, Mapping):
        raise ValueError("structured GPP joint recovery initial panel is missing")
    completed_epochs, partial_epoch_updates = divmod(completed, per_epoch)
    if len(epoch_panels) != completed_epochs:
        raise ValueError("structured GPP joint recovery epoch-panel coverage drift")
    selected_epoch = selected_epoch_from_panels(epoch_panels)
    for epoch, panel in enumerate(epoch_panels, start=1):
        expected_step = int(initial_optimizer_step) + epoch * per_epoch
        changes = panel.get("source_changes_from_previous_panel", {})
        if (
            int(panel.get("absolute_optimizer_step", -1)) != expected_step
            or not isinstance(panel.get("panel"), Mapping)
            or bool(panel.get("continuation_gate_passed"))
            != epoch_panel_continuation_passed(changes)
        ):
            raise ValueError("structured GPP joint recovery epoch-panel identity drift")
    if epoch_panels and not bool(epoch_panels[-1]["continuation_gate_passed"]):
        if partial_epoch_updates != 0:
            raise ValueError("structured GPP joint recovery continued after failed epoch")
    if len(epoch_checkpoints) != selected_epoch:
        raise ValueError("structured GPP joint recovery checkpoint coverage drift")
    for epoch, checkpoint in enumerate(epoch_checkpoints, start=1):
        expected_step = int(initial_optimizer_step) + epoch * per_epoch
        if (
            int(checkpoint.get("epoch", -1)) != epoch
            or int(checkpoint.get("completed_updates", -1)) != expected_step
            or not checkpoint.get("path")
            or not checkpoint.get("sha256")
        ):
            raise ValueError("structured GPP joint recovery checkpoint identity drift")
    return {
        "initial_joint_panel": initial_panel,
        "epoch_panels": epoch_panels,
        "epoch_checkpoints": epoch_checkpoints,
        "selected_epoch": selected_epoch,
    }


def load_structured_gpp_joint_training_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPJointTrainingContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structured GPP joint training schema drift")
    if (
        payload.get("status")
        != "frozen_before_local_runner_qualification_and_paid_resource_approval"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP joint training contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    objective = payload["objective"]
    optimizer = payload["optimizer"]
    bounded = payload["bounded_stage"]
    panel = payload["epoch_panel"]
    execution = payload["execution"]
    schedule = build_joint_training_schedule(
        seed=int(data["shuffle"]["seed"]),
        epochs=int(data["epochs_maximum"]),
        batch_size=int(data["effective_batch_size"]),
    )
    coefficients = {str(name): float(value) for name, value in objective["coefficients"].items()}
    if (
        identity["batch4_preflight_report_sha256"]
        != "6a574c67aa3aaa0883cbe066b3fe4372490163e7c3639cc0a172c3401cea0ac1"
        or int(identity["warm_start_checkpoint_update"]) != 250
        or float(identity["warm_start_clip_threshold"]) != 17.256861560753983
        or int(identity["candidate_parameter_count"]) != 106_383_527
        or int(identity["trainable_specialist_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(identity["retained_parent_parameter_count"]) != 93_504_164
        or int(data["artifact_candidate_start"]) != 512
        or int(data["artifact_candidate_end"]) != 763
        or tuple(int(value) for value in data["excluded_evaluation_artifacts"])
        != (572, 679)
        or int(data["candidate_artifact_count"]) != 250
        or int(data["effective_batch_size"]) != 4
        or int(data["epochs_maximum"]) != 4
        or int(data["optimizer_updates_per_epoch"]) != 62
        or int(data["optimizer_updates_maximum"]) != 248
        or tuple(tuple(int(value) for value in item) for item in data["shuffle"]["dropped_artifacts_by_epoch"])
        != schedule.dropped_by_epoch
        or data["shuffle"]["materialized_schedule_sha256"] != schedule.sha256
        or schedule.sha256 != SCHEDULE_SHA256
        or len({value for group in schedule.flat_groups for value in group}) != 250
        or tuple(int(value) for value in data["train_only_panel_updates"])
        != TRAIN_ONLY_PANEL_UPDATES
        or tuple(int(value) for value in panel["artifact_updates"])
        != TRAIN_ONLY_PANEL_UPDATES
        or data["panel_disjoint_from_used_training_artifacts"] is not True
        or any(
            bool(data[name])
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_values_allowed_during_training_or_checkpoint_selection",
                "sealed_test_allowed",
            )
        )
    ):
        raise ValueError("structured GPP joint training identity or data drift")
    if (
        tuple(objective["gradient_sources"]) != GRADIENT_SOURCES
        or set(coefficients) != set(GRADIENT_SOURCES)
        or coefficients[DIRECT_SOURCE] != 1.0
        or coefficients[SEVEN_SOURCE] != 5.065197169020285
        or coefficients[THIRTY_SOURCE] != 4.594433600967992
        or objective["source_gradient_aggregation"]
        != "arithmetic_mean_across_four_artifacts"
        or optimizer["name"] != "adamw"
        or int(optimizer["initial_optimizer_step"]) != 250
        or int(optimizer["maximum_final_optimizer_step"]) != 498
        or float(optimizer["learning_rate"]) != 4e-6
        or float(optimizer["global_gradient_norm_clip"]["threshold"])
        != 46.802861793665286
        or int(bounded["recovery_interval_joint_updates"]) != 10
        or tuple(int(value) for value in bounded["checkpoint_absolute_optimizer_steps"])
        != (312, 374, 436, 498)
        or panel["continuation_requires_each_source_strictly_decrease_from_previous_panel"]
        is not True
        or panel["checkpoint_selection_uses_validation"] is not False
    ):
        raise ValueError("structured GPP joint training objective or stage drift")
    if (
        execution["paid_training_requires_explicit_resource_authorization"] is not True
        or execution["paid_training_authorized_by_contract_alone"] is not False
        or execution["free_gln01_training_allowed"] is not False
        or any(
            bool(execution[name])
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_read_authorized_during_training",
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
        raise ValueError("structured GPP joint training execution boundary drift")
    return StructuredGPPJointTrainingContract(payload, source, observed)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "SCHEDULE_SHA256",
    "TRAIN_ONLY_PANEL_UPDATES",
    "JointTrainingSchedule",
    "StructuredGPPJointTrainingContract",
    "build_joint_training_schedule",
    "candidate_artifact_updates",
    "epoch_panel_continuation_passed",
    "load_structured_gpp_joint_training_contract",
    "panel_source_changes",
    "selected_epoch_from_panels",
    "validate_joint_training_recovery_metadata",
]
