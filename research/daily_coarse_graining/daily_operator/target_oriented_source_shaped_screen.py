"""Frozen train/train screen contract and science comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "target_oriented_source_shaped_train_train_screen_v1"
FROZEN_SHA256 = "a06beb815834142f44939022eea6ebd044e0a096b5d3c443754a9d87fb0f8cee"
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_source_shaped_train_train_screen_v1.json"
)


def contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SourceShapedTrainTrainScreenContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["evaluation_updates"])

    @property
    def science_gate(self) -> Mapping[str, Any]:
        return self.payload["science_gate"]

    @property
    def optimizer(self) -> Mapping[str, Any]:
        return self.payload["branch_only_stage"]["optimizer"]


def load_source_shaped_train_train_screen_contract(
    path: str | Path = _CONTRACT_PATH,
) -> SourceShapedTrainTrainScreenContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("source-shaped screen schema version drift")
    if payload.get("contract_sha256") != observed or observed != FROZEN_SHA256:
        raise ValueError("source-shaped screen contract SHA256 drift")

    data = payload["data"]
    evaluation = tuple(int(value) for value in data["evaluation_updates"])
    calibration = tuple(int(value) for value in data["calibration_updates"])
    training = data["branch_training_updates"]
    training_updates = tuple(
        range(
            int(training["artifact_update_start"]),
            int(training["artifact_update_end"]) + 1,
        )
    )
    if evaluation != (251, 358, 465, 572, 679, 786, 893, 1000):
        raise ValueError("source-shaped screen evaluation panel drift")
    if len(calibration) != 8 or len(training_updates) != int(training["optimizer_update_count"]):
        raise ValueError("source-shaped screen calibration/training count drift")
    if set(evaluation) & set(calibration) or set(evaluation) & set(training_updates):
        raise ValueError("source-shaped evaluation groups overlap optimization data")
    if set(calibration) & set(training_updates):
        raise ValueError("source-shaped calibration groups overlap training data")
    if payload["branch_only_stage"]["trainable_parameter_count"] != 394_242:
        raise ValueError("source-shaped screen trainable parameter count drift")
    return SourceShapedTrainTrainScreenContract(payload, source, observed)


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


def compare_source_shaped_train_train_groups(
    parent_groups: Sequence[Mapping[str, Any]],
    candidate_groups: Sequence[Mapping[str, Any]],
    contract: SourceShapedTrainTrainScreenContract,
) -> Mapping[str, Any]:
    """Compare identical train/train groups under the frozen source gate."""

    expected = contract.evaluation_updates
    parent = {int(value["update"]): value for value in parent_groups}
    candidate = {int(value["update"]): value for value in candidate_groups}
    if tuple(parent) != expected or tuple(candidate) != expected:
        raise ValueError("source-shaped screen group identity or order drift")

    gate = contract.science_gate
    maintenance_key = str(gate["maintenance_key"])
    protected_keys = tuple(str(value) for value in gate["protected_keys"])
    maintenance_reductions = {"7": [], "30": []}
    protected_changes = {
        horizon: {name: [] for name in protected_keys}
        for horizon in ("7", "30")
    }
    group_records = []
    finite = True
    hard_failure_count = 0
    absolute = True

    for update in expected:
        before = parent[update]
        after = candidate[update]
        if int(before["runtime_year"]) != int(after["runtime_year"]):
            raise ValueError("source-shaped screen runtime-year drift")
        hard = int(after["hard_failure_count"])
        hard_failure_count += hard
        absolute = absolute and all(
            bool(after["acceptance"][horizon]["all_key_gates_passed"])
            for horizon in ("7", "30")
        )
        horizon_record = {}
        for horizon in ("7", "30"):
            before_keys = before["acceptance"][horizon]["keys"]
            after_keys = after["acceptance"][horizon]["keys"]
            if maintenance_key not in before_keys or any(name not in before_keys for name in protected_keys):
                raise ValueError("source-shaped parent acceptance registry drift")
            if set(before_keys) != set(after_keys):
                raise ValueError("source-shaped candidate acceptance registry drift")
            before_maintenance = _acceptance_score(before_keys[maintenance_key])
            after_maintenance = _acceptance_score(after_keys[maintenance_key])
            reduction = -_relative_change(after_maintenance, before_maintenance)
            maintenance_reductions[horizon].append(reduction)
            changes = {}
            for name in protected_keys:
                change = _relative_change(
                    _acceptance_score(after_keys[name]),
                    _acceptance_score(before_keys[name]),
                )
                protected_changes[horizon][name].append(change)
                changes[name] = change
            values = (reduction, *changes.values())
            finite = finite and all(np.isfinite(value) for value in values)
            horizon_record[horizon] = {
                "maintenance_score_reduction": reduction,
                "protected_score_relative_changes": changes,
            }
        group_records.append(
            {
                "update": update,
                "runtime_year": int(after["runtime_year"]),
                "hard_failure_count": hard,
                "horizons": horizon_record,
            }
        )

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
    maximum_any = max(
        value
        for by_name in protected_changes.values()
        for values in by_name.values()
        for value in values
    )
    maximum_median = max(
        value
        for by_name in median_protected.values()
        for value in by_name.values()
    )
    minimum_reduction = gate["minimum_median_score_reduction_by_horizon"]
    maintenance_passed = all(
        median_maintenance[horizon] >= float(minimum_reduction[horizon])
        for horizon in ("7", "30")
    )
    protected_passed = (
        maximum_median <= float(gate["maximum_median_protected_score_increase"])
        and maximum_any <= float(gate["maximum_any_group_protected_score_increase"])
    )
    hard_passed = hard_failure_count <= int(gate["hard_failure_count_maximum"])
    relative_passed = finite and maintenance_passed and protected_passed and hard_passed
    return {
        "all_scores_finite": finite,
        "hard_failure_count": hard_failure_count,
        "hard_gate_passed": hard_passed,
        "median_maintenance_score_reduction": median_maintenance,
        "maintenance_reduction_gate_passed": maintenance_passed,
        "median_protected_score_relative_changes": median_protected,
        "maximum_median_protected_score_increase": maximum_median,
        "maximum_any_group_protected_score_increase": maximum_any,
        "protected_score_gate_passed": protected_passed,
        "relative_source_gate_passed": relative_passed,
        "absolute_target_oriented_gate_passed_on_all_groups": absolute,
        "scientific_promotion_gate_passed": relative_passed and absolute,
        "classification": (
            "scientific_promotion_gate_passed"
            if relative_passed and absolute
            else (
                "source_mechanism_supported_not_promoted"
                if relative_passed
                else "source_mechanism_not_yet_supported"
            )
        ),
        "groups": group_records,
    }


__all__ = [
    "FROZEN_SHA256",
    "SCHEMA_VERSION",
    "SourceShapedTrainTrainScreenContract",
    "compare_source_shaped_train_train_groups",
    "contract_sha256",
    "load_source_shaped_train_train_screen_contract",
]
