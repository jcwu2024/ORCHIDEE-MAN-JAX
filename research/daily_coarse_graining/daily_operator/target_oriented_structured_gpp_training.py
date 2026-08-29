"""Frozen preflight contract and direct loss for the structured GPP specialist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .target_oriented_source_shaped_maintenance import contract_sha256
from .target_oriented_training import (
    ErrorSufficientStatistics,
    _linearized_error_loss,
    normalized_error_from_sufficient_statistics,
    normalized_error_sufficient_statistics,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_training_preflight_v1"
FROZEN_CONTRACT_SHA256 = "b2b6c0129f524ad56440633ad5923fbb75bd602e4cd87c655625295ac23ce828"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_training_preflight_v1.json"
)
SPECIALIST_KEY = "structured_gpp"
SPECIALIST_PARAMETER_COUNT = 12_879_363
CANDIDATE_PARAMETER_COUNT = 106_383_527


@dataclass(frozen=True)
class StructuredGPPTrainingPreflightContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def calibration_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["calibration_updates"])

    @property
    def ephemeral_update(self) -> Mapping[str, Any]:
        return self.payload["ephemeral_update"]


@dataclass(frozen=True)
class StructuredGPPGradientCalibration:
    direct_norms: tuple[float, ...]
    seven_day_norms: tuple[float, ...]
    thirty_day_norms: tuple[float, ...]
    direct_p90_clip_threshold: float
    median_norms: Mapping[str, float]
    initial_horizon_to_direct_norm_ratios: Mapping[str, float]


def _stage_updates(stage: Mapping[str, Any]) -> tuple[int, ...]:
    excluded = {int(value) for value in stage["excluded_artifact_updates"]}
    return tuple(
        update
        for update in range(
            int(stage["artifact_candidate_start"]),
            int(stage["artifact_candidate_end"]) + 1,
        )
        if update not in excluded
    )


def load_structured_gpp_training_preflight_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPTrainingPreflightContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structured GPP training-preflight schema drift")
    if (
        payload.get("status") != "frozen_before_free_gpu_gradient_calibration_and_ephemeral_update"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP training-preflight contract is not frozen")

    identity = payload["identity"]
    ownership = payload["parameter_ownership"]
    if (
        int(identity["candidate_parameter_count"]) != CANDIDATE_PARAMETER_COUNT
        or int(identity["trainable_specialist_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(ownership["trainable_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(ownership["retained_parent_parameter_count"]) != 93_504_164
        or ownership["parent_parameters_frozen"] is not True
    ):
        raise ValueError("structured GPP training-preflight parameter ownership drift")

    data = payload["data"]
    calibration = tuple(int(value) for value in data["calibration_updates"])
    evaluation = tuple(int(value) for value in data["evaluation_updates"])
    sources = data["direct_gpp_day_sources"]
    if (
        calibration != (352, 353, 354, 355, 356, 357, 359, 360)
        or evaluation != (251, 358, 465, 572, 679, 786, 893, 1000)
        or set(calibration) & set(evaluation)
        or int(data["direct_gpp_teacher_days_per_artifact"]) != 208
        or sum(int(value) for value in sources.values()) != 208
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
        raise ValueError("structured GPP training-preflight data boundary drift")

    warm = _stage_updates(payload["planned_bounded_training"]["warm_start"])
    joint = _stage_updates(payload["planned_bounded_training"]["joint_stage"])
    if (
        len(warm) != 250
        or len(joint) != 250
        or set(warm) & set(joint)
        or set(warm) & set(calibration)
        or set(warm) & set(evaluation)
        or set(joint) & set(evaluation)
        or payload["planned_bounded_training"]["authorized_by_this_contract"] is not False
    ):
        raise ValueError("structured GPP planned training schedule drift")

    direct = payload["objective"]["direct_gpp"]
    probe = payload["ephemeral_update"]
    if (
        float(direct["coefficient"]) != 1.0
        or probe["optimizer"] != "adamw"
        or float(probe["learning_rate"]) != 4.0e-6
        or int(probe["optimizer_updates_executed"]) != 1
        or int(probe["optimizer_updates_persisted"]) != 0
        or probe["checkpoint_written"] is not False
        or probe["recovery_written"] is not False
        or payload["execution"]["paid_training_authorized"] is not False
    ):
        raise ValueError("structured GPP training-preflight execution drift")
    return StructuredGPPTrainingPreflightContract(payload, source, observed)


def structured_gpp_direct_statistics(
    predicted: Any,
    expected: Any,
    defined: Any,
    scale: Any,
    active_vegetation: Any,
) -> ErrorSufficientStatistics:
    """Accumulate direct GPP error only on defined active vegetation."""

    predicted = jnp.asarray(predicted, dtype=jnp.float64)
    active = jnp.broadcast_to(jnp.asarray(active_vegetation, dtype=jnp.bool_), predicted.shape)
    return normalized_error_sufficient_statistics(
        predicted,
        expected,
        jnp.asarray(defined, dtype=jnp.bool_) & active,
        scale,
    )


def structured_gpp_direct_loss(statistics: ErrorSufficientStatistics) -> Any:
    error = normalized_error_from_sufficient_statistics(statistics)
    return 0.5 * (error.mse + error.bias_squared)


def linearized_structured_gpp_direct_loss(
    current: ErrorSufficientStatistics,
    total: ErrorSufficientStatistics,
) -> Any:
    """Return one microbatch contribution with the exact full-batch gradient."""

    return _linearized_error_loss(
        current,
        total,
        mse_weight=0.5,
        bias_squared_weight=0.5,
    )


def specialist_tree(parameters: Mapping[str, Any]) -> Any:
    return parameters["target_oriented"][SPECIALIST_KEY]


def replace_specialist(parameters: Mapping[str, Any], specialist: Any) -> Mapping[str, Any]:
    target = dict(parameters["target_oriented"])
    target[SPECIALIST_KEY] = specialist
    return dict(parameters, target_oriented=target)


def combine_specialist_gradients(
    tree_util: Any,
    gradients: Sequence[Any],
    coefficients: Sequence[float],
) -> Any:
    gradients = tuple(gradients)
    coefficients = tuple(float(value) for value in coefficients)
    if not gradients or len(gradients) != len(coefficients):
        raise ValueError("structured GPP gradient combination arity drift")
    leaves, structure = tree_util.tree_flatten(gradients[0])
    del leaves
    if any(tree_util.tree_structure(value) != structure for value in gradients[1:]):
        raise ValueError("structured GPP gradients do not share a tree")
    return tree_util.tree_map(
        lambda *values: sum(
            coefficient * jnp.asarray(value, dtype=jnp.float32)
            for coefficient, value in zip(coefficients, values, strict=True)
        ),
        *gradients,
    )


def calibrate_structured_gpp_gradient_norms(
    direct_norms: Sequence[float],
    seven_day_norms: Sequence[float],
    thirty_day_norms: Sequence[float],
    *,
    minimum_clip_threshold: float = 1.0e-8,
) -> StructuredGPPGradientCalibration:
    """Freeze the warm-start clip scale and report initial horizon geometry."""

    arrays = {
        "direct_gpp": np.asarray(tuple(direct_norms), dtype=np.float64),
        "seven_day": np.asarray(tuple(seven_day_norms), dtype=np.float64),
        "thirty_day": np.asarray(tuple(thirty_day_norms), dtype=np.float64),
    }
    if any(value.shape != (8,) for value in arrays.values()):
        raise ValueError("structured GPP gradient calibration requires eight norm triplets")
    if any(not np.all(np.isfinite(value)) or np.any(value <= 0.0) for value in arrays.values()):
        raise ValueError("structured GPP calibration norms must be finite and positive")
    if not np.isfinite(minimum_clip_threshold) or minimum_clip_threshold <= 0.0:
        raise ValueError("structured GPP minimum clip threshold is invalid")
    medians = {name: float(np.median(value)) for name, value in arrays.items()}
    clip = max(
        float(np.quantile(arrays["direct_gpp"], 0.9, method="linear")),
        float(minimum_clip_threshold),
    )
    direct = medians["direct_gpp"]
    ratios = {
        "seven_day": medians["seven_day"] / direct,
        "thirty_day": medians["thirty_day"] / direct,
    }
    return StructuredGPPGradientCalibration(
        direct_norms=tuple(float(value) for value in arrays["direct_gpp"]),
        seven_day_norms=tuple(float(value) for value in arrays["seven_day"]),
        thirty_day_norms=tuple(float(value) for value in arrays["thirty_day"]),
        direct_p90_clip_threshold=clip,
        median_norms=medians,
        initial_horizon_to_direct_norm_ratios=ratios,
    )


def specialist_element_count(tree: Any) -> int:
    return sum(int(np.prod(value.shape, dtype=np.int64)) for value in jax.tree_util.tree_leaves(tree))


__all__ = [
    "CANDIDATE_PARAMETER_COUNT",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "SPECIALIST_PARAMETER_COUNT",
    "StructuredGPPGradientCalibration",
    "StructuredGPPTrainingPreflightContract",
    "calibrate_structured_gpp_gradient_norms",
    "combine_specialist_gradients",
    "linearized_structured_gpp_direct_loss",
    "load_structured_gpp_training_preflight_contract",
    "replace_specialist",
    "specialist_element_count",
    "specialist_tree",
    "structured_gpp_direct_loss",
    "structured_gpp_direct_statistics",
]
