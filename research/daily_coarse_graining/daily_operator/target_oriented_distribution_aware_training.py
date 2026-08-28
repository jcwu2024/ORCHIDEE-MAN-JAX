"""Training-preflight contract for distribution-aware maintenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_distribution_aware_guard_calibration_v1"
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_distribution_aware_guard_calibration_v1.json"
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


__all__ = [
    "SCHEMA_VERSION",
    "DistributionAwareGuardCalibrationContract",
    "TeacherGuardCoefficientCalibration",
    "calibrate_teacher_guard_coefficient",
    "load_distribution_aware_guard_calibration_contract",
]
