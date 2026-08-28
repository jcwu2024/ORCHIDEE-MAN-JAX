"""Frozen calibration and bounded-training contracts for distribution-aware maintenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_distribution_aware_guard_calibration_v1"
BOUNDED_TRAINING_SCHEMA_VERSION = "target_oriented_distribution_aware_bounded_training_v1"
BOUNDED_TRAINING_FROZEN_SHA256 = "0c60924ee12d0ea555b082f76990ca55e34f86f67effb052f5eec44c01210051"
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_distribution_aware_guard_calibration_v1.json"
)
_BOUNDED_TRAINING_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_distribution_aware_bounded_training_v1.json"
)


@dataclass(frozen=True)
class DistributionAwareGuardCalibrationContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def calibration_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["calibration_updates"])


@dataclass(frozen=True)
class TeacherGuardCoefficientCalibration:
    free_gradient_norms: tuple[float, ...]
    teacher_guard_gradient_norms: tuple[float, ...]
    median_free_gradient_norm: float
    median_teacher_guard_gradient_norm: float
    unclipped_coefficient: float
    coefficient: float
    clipped: bool


@dataclass(frozen=True)
class DistributionAwareBoundedTrainingContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def optimizer(self) -> Mapping[str, Any]:
        return self.payload["optimizer"]

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["evaluation_updates"])

    @property
    def science_gate(self) -> Mapping[str, Any]:
        return self.payload["science_gate"]


def load_distribution_aware_guard_calibration_contract(
    path: str | Path = _CONTRACT_PATH,
) -> DistributionAwareGuardCalibrationContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("distribution-aware guard calibration schema drift")
    if payload.get("contract_sha256") != observed:
        raise ValueError("distribution-aware guard calibration SHA256 mismatch")
    if payload.get("status") != "frozen_before_zero_update_calibration":
        raise ValueError("distribution-aware guard calibration is not frozen")
    identity = payload["identity"]
    if (
        int(identity["candidate_parameter_count"]) != 93_504_933
        or int(identity["trainable_adapter_parameter_count"]) != 9_750_529
    ):
        raise ValueError("distribution-aware guard calibration parameter ownership drift")
    data = payload["data"]
    updates = tuple(int(value) for value in data["calibration_updates"])
    if len(updates) != 8 or len(set(updates)) != 8:
        raise ValueError("distribution-aware guard calibration panel drift")
    guard = payload["teacher_guard_objective"]
    if (
        tuple(int(value) for value in guard["horizons_days"]) != (7, 30)
        or tuple(guard["keys"]) != ("maintenance_respiration", "growth_respiration", "npp")
        or int(guard["cell_count"]) != 6
        or not np.isclose(float(guard["cell_weight"]), 1.0 / 6.0)
    ):
        raise ValueError("distribution-aware Teacher-guard objective drift")
    return DistributionAwareGuardCalibrationContract(payload, source, observed)


def load_distribution_aware_bounded_training_contract(
    path: str | Path = _BOUNDED_TRAINING_CONTRACT_PATH,
) -> DistributionAwareBoundedTrainingContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != BOUNDED_TRAINING_SCHEMA_VERSION:
        raise ValueError("distribution-aware bounded-training schema drift")
    if (
        payload.get("contract_sha256") != observed
        or observed != BOUNDED_TRAINING_FROZEN_SHA256
        or payload.get("status") != "frozen_before_one_update_admission"
    ):
        raise ValueError("distribution-aware bounded-training contract is not frozen")

    identity = payload["identity"]
    ownership = payload["parameter_ownership"]
    if (
        int(identity["candidate_parameter_count"]) != 93_504_933
        or int(identity["trainable_adapter_parameter_count"]) != 9_750_529
        or int(ownership["trainable_parameter_count"]) != 9_750_529
        or int(ownership["retained_parent_parameter_count"]) != 83_754_404
        or ownership["parent_parameters_frozen"] is not True
    ):
        raise ValueError("distribution-aware bounded-training parameter ownership drift")

    data = payload["data"]
    training = data["training_updates"]
    excluded = tuple(int(value) for value in training["excluded_artifact_updates"])
    excluded_set = set(excluded)
    training_updates = tuple(
        value
        for value in range(
            int(training["artifact_candidate_start"]),
            int(training["artifact_candidate_end"]) + 1,
        )
        if value not in excluded_set
    )
    calibration_updates = tuple(int(value) for value in data["guard_calibration_updates"])
    evaluation_updates = tuple(int(value) for value in data["evaluation_updates"])
    if (
        len(training_updates) != 250
        or training_updates[0] != 252
        or training_updates[-1] != 511
        or excluded != (352, 353, 354, 355, 356, 357, 358, 359, 360, 465)
        or int(training["optimizer_update_count"]) != 250
        or calibration_updates != (352, 353, 354, 355, 356, 357, 359, 360)
        or evaluation_updates != (251, 358, 465, 572, 679, 786, 893, 1000)
        or set(training_updates) & set(calibration_updates)
        or set(training_updates) & set(evaluation_updates)
        or set(calibration_updates) & set(evaluation_updates)
    ):
        raise ValueError("distribution-aware bounded-training data separation drift")

    objective = payload["objective"]
    optimizer = payload["optimizer"]
    clipping = optimizer["global_gradient_norm_clip"]
    if (
        float(objective["free_objective_coefficient"]) != 1.0
        or float(objective["teacher_guard_coefficient"]) != 0.1
        or int(objective["teacher_guard_cell_count"]) != 6
        or not np.isclose(float(objective["teacher_guard_cell_weight"]), 1.0 / 6.0)
        or optimizer["name"] != "adamw"
        or int(optimizer["warmup_updates"]) != 25
        or int(optimizer["maximum_updates"]) != 250
        or float(optimizer["peak_learning_rate"]) != 1.0e-4
        or float(optimizer["final_learning_rate"]) != 3.0e-5
        or float(clipping["threshold"]) != 2.692620468992339
    ):
        raise ValueError("distribution-aware bounded-training objective or optimizer drift")
    return DistributionAwareBoundedTrainingContract(payload, source, observed)


def calibrate_teacher_guard_coefficient(
    free_gradient_norms: Sequence[float],
    teacher_guard_gradient_norms: Sequence[float],
    *,
    minimum: float = 0.1,
    maximum: float = 10.0,
) -> TeacherGuardCoefficientCalibration:
    """Balance median adapter-gradient norms without validation feedback."""

    free = np.asarray(tuple(free_gradient_norms), dtype=np.float64)
    guard = np.asarray(tuple(teacher_guard_gradient_norms), dtype=np.float64)
    if free.shape != (8,) or guard.shape != (8,):
        raise ValueError("Teacher-guard calibration requires eight norm pairs")
    if not np.all(np.isfinite(free)) or not np.all(np.isfinite(guard)):
        raise ValueError("Teacher-guard calibration norms must be finite")
    if np.any(free <= 0.0) or np.any(guard <= 0.0):
        raise ValueError("Teacher-guard calibration norms must be strictly positive")
    if not 0.0 < minimum <= maximum:
        raise ValueError("Teacher-guard coefficient bounds are invalid")
    median_free = float(np.median(free))
    median_guard = float(np.median(guard))
    raw = median_free / median_guard
    coefficient = float(np.clip(raw, minimum, maximum))
    return TeacherGuardCoefficientCalibration(
        free_gradient_norms=tuple(float(value) for value in free),
        teacher_guard_gradient_norms=tuple(float(value) for value in guard),
        median_free_gradient_norm=median_free,
        median_teacher_guard_gradient_norm=median_guard,
        unclipped_coefficient=raw,
        coefficient=coefficient,
        clipped=coefficient != raw,
    )


def combine_adapter_gradients(
    tree_util: Any,
    free_gradient: Any,
    teacher_guard_gradient: Any,
    coefficient: float,
) -> Any:
    """Combine the two gradients over their identical adapter subtree."""

    weight = float(coefficient)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("Teacher-guard gradient coefficient must be finite and nonnegative")
    if tree_util.tree_structure(free_gradient) != tree_util.tree_structure(teacher_guard_gradient):
        raise ValueError("free and Teacher-guard adapter gradients must share a tree")
    return tree_util.tree_map(
        lambda free, guard: free + weight * guard,
        free_gradient,
        teacher_guard_gradient,
    )


def _acceptance_score(record: Mapping[str, Any]) -> float:
    rmse = float(record["normalized_rmse"])
    bias = float(record["normalized_signed_bias"])
    return 0.5 * (rmse * rmse + bias * bias)


def _relative_change(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before):
        return float("nan")
    if before > 1.0e-15:
        return after / before - 1.0
    return 0.0 if abs(after - before) <= 1.0e-15 else float("inf")


def aggregate_target_oriented_acceptance(
    groups: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Aggregate equal-scale group summaries using their evaluated-value counts."""

    if not groups:
        raise ValueError("target-oriented acceptance aggregation requires groups")
    result = {}
    for horizon in ("7", "30"):
        registries = [group["acceptance"][horizon]["keys"] for group in groups]
        names = tuple(registries[0])
        if any(tuple(registry) != names for registry in registries[1:]):
            raise ValueError("target-oriented aggregate key registry drift")
        keys = {}
        for name in names:
            records = [registry[name] for registry in registries]
            counts = np.asarray(
                [float(record["evaluated_values"]) for record in records],
                dtype=np.float64,
            )
            if np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
                raise ValueError("target-oriented aggregate count drift")
            rmse_limits = {float(record["rmse_maximum"]) for record in records}
            bias_limits = {float(record["absolute_bias_maximum"]) for record in records}
            if len(rmse_limits) != 1 or len(bias_limits) != 1:
                raise ValueError("target-oriented aggregate threshold drift")
            rmse = float(
                np.sqrt(
                    np.average(
                        np.square([float(record["normalized_rmse"]) for record in records]),
                        weights=counts,
                    )
                )
            )
            bias = float(
                np.average(
                    [float(record["normalized_signed_bias"]) for record in records],
                    weights=counts,
                )
            )
            rmse_limit = rmse_limits.pop()
            bias_limit = bias_limits.pop()
            keys[name] = {
                "evaluated_values": float(np.sum(counts)),
                "normalized_rmse": rmse,
                "normalized_signed_bias": bias,
                "rmse_maximum": rmse_limit,
                "absolute_bias_maximum": bias_limit,
                "passed": bool(rmse <= rmse_limit and abs(bias) <= bias_limit),
            }
        result[horizon] = {
            "keys": keys,
            "all_key_gates_passed": all(value["passed"] for value in keys.values()),
        }
    return result


def compare_distribution_aware_train_train_groups(
    parent_groups: Sequence[Mapping[str, Any]],
    candidate_groups: Sequence[Mapping[str, Any]],
    contract: DistributionAwareBoundedTrainingContract,
) -> Mapping[str, Any]:
    """Apply the relative mechanism gate and absolute aggregate science gate."""

    expected = contract.evaluation_updates
    parent = {int(value["update"]): value for value in parent_groups}
    candidate = {int(value["update"]): value for value in candidate_groups}
    if tuple(parent) != expected or tuple(candidate) != expected:
        raise ValueError("distribution-aware screen group identity or order drift")

    gate = contract.science_gate
    maintenance_key = str(gate["maintenance_key"])
    protected_keys = tuple(str(value) for value in gate["protected_keys"])
    maintenance_reductions = {"7": [], "30": []}
    protected_changes = {
        horizon: {name: [] for name in protected_keys}
        for horizon in ("7", "30")
    }
    hard_failure_count = 0
    finite = True
    for update in expected:
        before = parent[update]
        after = candidate[update]
        if int(before["runtime_year"]) != int(after["runtime_year"]):
            raise ValueError("distribution-aware screen runtime-year drift")
        hard_failure_count += int(after["hard_failure_count"])
        for horizon in ("7", "30"):
            before_keys = before["acceptance"][horizon]["keys"]
            after_keys = after["acceptance"][horizon]["keys"]
            if set(before_keys) != set(after_keys):
                raise ValueError("distribution-aware screen acceptance registry drift")
            reduction = -_relative_change(
                _acceptance_score(after_keys[maintenance_key]),
                _acceptance_score(before_keys[maintenance_key]),
            )
            maintenance_reductions[horizon].append(reduction)
            finite = finite and np.isfinite(reduction)
            for name in protected_keys:
                change = _relative_change(
                    _acceptance_score(after_keys[name]),
                    _acceptance_score(before_keys[name]),
                )
                protected_changes[horizon][name].append(change)
                finite = finite and np.isfinite(change)

    median_maintenance = {
        horizon: float(np.median(values))
        for horizon, values in maintenance_reductions.items()
    }
    median_protected = {
        horizon: {
            name: float(np.median(values))
            for name, values in by_name.items()
        }
        for horizon, by_name in protected_changes.items()
    }
    minimum_reduction = gate["minimum_median_maintenance_score_reduction_by_horizon"]
    maintenance_passed = all(
        median_maintenance[horizon] >= float(minimum_reduction[horizon])
        for horizon in ("7", "30")
    )
    protected_passed = all(
        value <= float(gate["maximum_median_protected_score_increase"])
        for by_name in median_protected.values()
        for value in by_name.values()
    )
    hard_passed = hard_failure_count <= int(gate["hard_failure_count_maximum"])
    relative_passed = finite and maintenance_passed and protected_passed and hard_passed
    parent_aggregate = aggregate_target_oriented_acceptance(tuple(parent.values()))
    candidate_aggregate = aggregate_target_oriented_acceptance(tuple(candidate.values()))
    absolute_passed = all(
        candidate_aggregate[str(horizon)]["all_key_gates_passed"]
        for horizon in gate["absolute_target_oriented_all_keys_required_at_horizons"]
    )
    promoted = relative_passed and absolute_passed
    return {
        "all_scores_finite": bool(finite),
        "hard_failure_count": hard_failure_count,
        "hard_gate_passed": hard_passed,
        "median_maintenance_score_reduction": median_maintenance,
        "maintenance_reduction_gate_passed": maintenance_passed,
        "median_protected_score_relative_changes": median_protected,
        "protected_score_gate_passed": protected_passed,
        "relative_mechanism_gate_passed": relative_passed,
        "parent_aggregate_acceptance": parent_aggregate,
        "candidate_aggregate_acceptance": candidate_aggregate,
        "absolute_target_oriented_gate_passed": absolute_passed,
        "scientific_promotion_gate_passed": promoted,
        "classification": (
            "scientific_promotion_gate_passed"
            if promoted
            else (
                "distribution_aware_mechanism_supported_not_promoted"
                if relative_passed
                else "distribution_aware_mechanism_not_yet_supported"
            )
        ),
    }


__all__ = [
    "BOUNDED_TRAINING_FROZEN_SHA256",
    "BOUNDED_TRAINING_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "DistributionAwareBoundedTrainingContract",
    "DistributionAwareGuardCalibrationContract",
    "TeacherGuardCoefficientCalibration",
    "aggregate_target_oriented_acceptance",
    "calibrate_teacher_guard_coefficient",
    "combine_adapter_gradients",
    "compare_distribution_aware_train_train_groups",
    "load_distribution_aware_bounded_training_contract",
    "load_distribution_aware_guard_calibration_contract",
]
