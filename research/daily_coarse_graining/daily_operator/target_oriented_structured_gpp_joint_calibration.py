"""Frozen train-only joint-gradient calibration for structured GPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_source_shaped_maintenance import contract_sha256
from .target_oriented_structured_gpp_training import SPECIALIST_PARAMETER_COUNT

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_gradient_calibration_v1"
FROZEN_CONTRACT_SHA256 = "c00ebf0242744c1583c12d1475b20021e325c4816c0391fb78a928ac874868fe"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_joint_gradient_calibration_v1.json"
)
CALIBRATION_UPDATES = (352, 353, 354, 355, 356, 357, 359, 360)
DIRECT_SOURCE = "direct_gpp_208_teacher_days"
SEVEN_SOURCE = "seven_day_component_resolved_science"
THIRTY_SOURCE = "thirty_day_component_resolved_science"
HORIZON_SOURCES = (SEVEN_SOURCE, THIRTY_SOURCE)
GRADIENT_SOURCES = (DIRECT_SOURCE, *HORIZON_SOURCES)


@dataclass(frozen=True)
class StructuredGPPJointGradientCalibrationContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def calibration_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["calibration_updates"])

    @property
    def target_gradient_norm_ratios(self) -> Mapping[str, float]:
        return {
            str(name): float(value)
            for name, value in self.payload["coefficient_calibration"]["target_gradient_norm_ratios"].items()
        }


@dataclass(frozen=True)
class StructuredGPPJointGradientCoefficients:
    resolved_coefficients: Mapping[str, float]
    source_distributions: Mapping[str, Mapping[str, float]]
    record_count: int
    coefficients_inside_interval: bool


def load_structured_gpp_joint_gradient_calibration_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPJointGradientCalibrationContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structured GPP joint-gradient calibration schema drift")
    if (
        payload.get("status") != "frozen_before_free_gpu_train_only_joint_gradient_calibration"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP joint-gradient calibration contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    ownership = payload["parameter_ownership"]
    calibration = tuple(int(value) for value in data["calibration_updates"])
    if (
        calibration != CALIBRATION_UPDATES
        or int(identity["warm_start_checkpoint_update"]) != 250
        or int(identity["candidate_parameter_count"]) != 106_383_527
        or int(identity["trainable_specialist_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(identity["retained_parent_parameter_count"]) != 93_504_164
        or int(ownership["differentiated_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(ownership["retained_parent_parameter_count"]) != 93_504_164
        or ownership["parent_parameters_frozen"] is not True
        or ownership["checkpoint_parameters_frozen"] is not True
        or ownership["optimizer_created"] is not False
        or int(ownership["optimizer_updates_executed"]) != 0
        or ownership["checkpoint_written"] is not False
    ):
        raise ValueError("structured GPP joint-gradient ownership or identity drift")
    if (
        int(data["direct_gpp_teacher_days_per_artifact"]) != 208
        or int(data["seven_day_windows_per_artifact"]) != 8
        or int(data["thirty_day_windows_per_artifact"]) != 4
        or any(
            bool(data[name])
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_values_allowed",
                "sealed_test_allowed",
            )
        )
    ):
        raise ValueError("structured GPP joint-gradient data boundary drift")

    objective = payload["objective"]
    coefficient = payload["coefficient_calibration"]
    execution = payload["execution"]
    if (
        tuple(objective["gradient_sources"]) != GRADIENT_SOURCES
        or float(objective["direct_gpp"]["coefficient"]) != 1.0
        or tuple(int(value) for value in objective["short_horizon_science"]["horizons_days"]) != (7, 30)
        or set(coefficient["target_gradient_norm_ratios"]) != set(HORIZON_SOURCES)
        or sum(float(value) for value in coefficient["target_gradient_norm_ratios"].values()) != 0.5
        or coefficient["aggregation"] != "per_source_median_across_eight_train_train_artifacts"
        or not 0.0 < float(coefficient["coefficient_minimum"]) < float(coefficient["coefficient_maximum"])
        or coefficient["recalibrate_during_training"] is not False
        or coefficient["validation_weight_tuning_allowed"] is not False
        or coefficient["combined_gradient_clip_statistic"]
        != "p90_across_eight_calibrated_combined_gradients"
        or execution["free_gln01_allowed"] is not True
        or any(
            bool(execution[name])
            for name in (
                "paid_training_authorized",
                "joint_training_authorized",
                "optimizer_update_authorized",
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_read_authorized",
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
                "cross_day_neural_hidden_state",
            )
        )
    ):
        raise ValueError("structured GPP joint-gradient objective or execution drift")
    return StructuredGPPJointGradientCalibrationContract(payload, source, observed)


def resolve_structured_gpp_joint_gradient_coefficients(
    records: Sequence[Mapping[str, Any]],
    contract: StructuredGPPJointGradientCalibrationContract,
) -> StructuredGPPJointGradientCoefficients:
    records = tuple(records)
    if tuple(int(record["artifact_update"]) for record in records) != contract.calibration_updates:
        raise ValueError("structured GPP joint-gradient calibration record identity drift")
    policy = contract.payload["coefficient_calibration"]
    minimum = float(policy["coefficient_minimum"])
    maximum = float(policy["coefficient_maximum"])
    targets = contract.target_gradient_norm_ratios
    resolved: dict[str, float] = {}
    distributions: dict[str, Mapping[str, float]] = {}
    inside_interval = True
    for source in HORIZON_SOURCES:
        candidates = np.asarray(
            [
                targets[source]
                * float(record["source_gradients"][DIRECT_SOURCE]["l2_norm"])
                / float(record["source_gradients"][source]["l2_norm"])
                for record in records
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(candidates)) or np.any(candidates <= 0.0):
            raise ValueError(f"structured GPP joint-gradient coefficient candidate is invalid for {source}")
        value = float(np.median(candidates))
        inside_interval = inside_interval and minimum <= value <= maximum
        achieved = np.asarray(
            [
                value
                * float(record["source_gradients"][source]["l2_norm"])
                / float(record["source_gradients"][DIRECT_SOURCE]["l2_norm"])
                for record in records
            ],
            dtype=np.float64,
        )
        resolved[source] = value
        distributions[source] = {
            "candidate_minimum": float(np.min(candidates)),
            "candidate_p10": float(np.quantile(candidates, 0.10)),
            "candidate_median": value,
            "candidate_p90": float(np.quantile(candidates, 0.90)),
            "candidate_maximum": float(np.max(candidates)),
            "achieved_ratio_minimum": float(np.min(achieved)),
            "achieved_ratio_median": float(np.median(achieved)),
            "achieved_ratio_maximum": float(np.max(achieved)),
        }
    return StructuredGPPJointGradientCoefficients(resolved, distributions, len(records), inside_interval)


def combined_gradient_p90_clip(
    norms: Sequence[float],
    *,
    minimum: float,
) -> float:
    values = np.asarray(tuple(norms), dtype=np.float64)
    if values.shape != (8,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("structured GPP combined-gradient clip requires eight finite positive norms")
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("structured GPP combined-gradient clip minimum is invalid")
    return max(float(np.quantile(values, 0.9, method="linear")), float(minimum))


__all__ = [
    "CALIBRATION_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "DIRECT_SOURCE",
    "FROZEN_CONTRACT_SHA256",
    "GRADIENT_SOURCES",
    "HORIZON_SOURCES",
    "SCHEMA_VERSION",
    "SEVEN_SOURCE",
    "StructuredGPPJointGradientCalibrationContract",
    "StructuredGPPJointGradientCoefficients",
    "THIRTY_SOURCE",
    "combined_gradient_p90_clip",
    "load_structured_gpp_joint_gradient_calibration_contract",
    "resolve_structured_gpp_joint_gradient_coefficients",
]
