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
FROZEN_CONTRACT_SHA256 = "6fa0253794938c4400b1e2ac028e818f177ca9c9632edd0fbb0f562520562b62"
WARM_START_SCHEMA_VERSION = "target_oriented_structured_gpp_warm_start_v1"
WARM_START_FROZEN_CONTRACT_SHA256 = "1a3b8138b7fcef0cc28f7669d89cdfaae053244b706eebf23b3aa4efcc4818a6"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_training_preflight_v1.json"
)
DEFAULT_WARM_START_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_warm_start_v1.json"
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
    direct_p90_clip_threshold: float
    median_direct_norm: float


@dataclass(frozen=True)
class StructuredGPPWarmStartContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def training_updates(self) -> tuple[int, ...]:
        return _stage_updates(self.payload["data"]["training_updates"])

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["evaluation_updates"])

    @property
    def optimizer(self) -> Mapping[str, Any]:
        return self.payload["optimizer"]


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
    calibration_contract = payload["gradient_calibration"]
    rollout = payload["short_horizon_forward_diagnostic"]
    probe = payload["ephemeral_update"]
    if (
        float(direct["coefficient"]) != 1.0
        or tuple(int(value) for value in calibration_contract["artifact_updates"]) != calibration
        or calibration_contract["gradient_sources"] != ["direct_gpp_208_teacher_days"]
        or int(calibration_contract["required_gradient_element_count"]) != SPECIALIST_PARAMETER_COUNT
        or calibration_contract["optimizer_created_during_calibration"] is not False
        or int(rollout["artifact_update"]) != 352
        or tuple(int(value) for value in rollout["horizons_days"]) != (7, 30)
        or int(rollout["optimizer_updates"]) != 0
        or rollout["gradients_evaluated"] is not False
        or rollout["require_objectives_finite_positive"] is not True
        or rollout["require_physical_hard_failure_count_zero"] is not True
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


def load_structured_gpp_warm_start_contract(
    path: str | Path = DEFAULT_WARM_START_CONTRACT_PATH,
) -> StructuredGPPWarmStartContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != WARM_START_SCHEMA_VERSION:
        raise ValueError("structured GPP warm-start schema drift")
    if (
        payload.get("status") != "frozen_after_free_gpu_preflight_before_paid_warm_start"
        or payload.get("contract_sha256") != observed
        or observed != WARM_START_FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP warm-start contract is not frozen")

    identity = payload["identity"]
    ownership = payload["parameter_ownership"]
    if (
        identity["training_preflight_contract_sha256"] != FROZEN_CONTRACT_SHA256
        or identity["training_preflight_report_sha256"]
        != "d531f8dcc6d684c1cba6c471d28e640a82114d0967849bea5bb5d7f47a2cbccc"
        or int(identity["candidate_parameter_count"]) != CANDIDATE_PARAMETER_COUNT
        or int(identity["trainable_specialist_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(identity["retained_parent_parameter_count"]) != 93_504_164
        or int(ownership["trainable_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(ownership["retained_parent_parameter_count"]) != 93_504_164
        or ownership["parent_parameters_frozen"] is not True
    ):
        raise ValueError("structured GPP warm-start ownership or preflight drift")

    contract = StructuredGPPWarmStartContract(payload, source, observed)
    training = contract.training_updates
    calibration = tuple(int(value) for value in payload["data"]["clip_calibration_updates"])
    evaluation = contract.evaluation_updates
    if (
        len(training) != 250
        or training[0] != 252
        or training[-1] != 511
        or calibration != (352, 353, 354, 355, 356, 357, 359, 360)
        or evaluation != (251, 358, 465, 572, 679, 786, 893, 1000)
        or set(training) & set(calibration)
        or set(training) & set(evaluation)
        or int(payload["data"]["direct_gpp_teacher_days_per_artifact"]) != 208
    ):
        raise ValueError("structured GPP warm-start data separation drift")

    objective = payload["objective"]
    optimizer = payload["optimizer"]
    clipping = optimizer["global_gradient_norm_clip"]
    endpoint = payload["endpoint_evaluation"]
    if (
        objective["name"] != "direct_gpp_208_teacher_days"
        or objective["recursive_candidate_state_used"] is not False
        or optimizer["name"] != "adamw"
        or optimizer["learning_rate_schedule"] != "constant"
        or float(optimizer["learning_rate"]) != 4.0e-6
        or int(optimizer["maximum_updates"]) != 250
        or float(clipping["threshold"]) != 17.256861560753983
        or tuple(int(value) for value in endpoint["direct_gpp_panel_updates"]) != evaluation
        or float(endpoint["minimum_aggregate_direct_gpp_loss_reduction"]) != 0.1
        or int(endpoint["minimum_individual_artifact_improvement_count"]) != 6
        or payload["next_gate"]["joint_training_authorized"] is not False
        or payload["execution"]["seven_or_thirty_day_gradients_authorized"] is not False
    ):
        raise ValueError("structured GPP warm-start objective or endpoint drift")
    return contract


def compare_structured_gpp_direct_panels(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    contract: StructuredGPPWarmStartContract,
) -> Mapping[str, Any]:
    expected = contract.evaluation_updates
    initial_records = tuple(initial["records"])
    final_records = tuple(final["records"])
    if (
        tuple(int(value["artifact_update"]) for value in initial_records) != expected
        or tuple(int(value["artifact_update"]) for value in final_records) != expected
    ):
        raise ValueError("structured GPP direct-panel identity drift")
    initial_loss = float(initial["aggregate_direct_gpp_loss"])
    final_loss = float(final["aggregate_direct_gpp_loss"])
    if not np.isfinite(initial_loss) or not np.isfinite(final_loss) or initial_loss <= 0.0 or final_loss < 0.0:
        raise ValueError("structured GPP direct-panel loss is invalid")
    improvement_count = sum(
        float(after["direct_gpp_loss"]) < float(before["direct_gpp_loss"])
        for before, after in zip(initial_records, final_records, strict=True)
    )
    reduction = 1.0 - final_loss / initial_loss
    endpoint = contract.payload["endpoint_evaluation"]
    passed = reduction >= float(endpoint["minimum_aggregate_direct_gpp_loss_reduction"]) and improvement_count >= int(
        endpoint["minimum_individual_artifact_improvement_count"]
    )
    return {
        "initial_aggregate_direct_gpp_loss": initial_loss,
        "final_aggregate_direct_gpp_loss": final_loss,
        "aggregate_direct_gpp_loss_reduction": reduction,
        "individual_artifact_improvement_count": improvement_count,
        "artifact_count": len(expected),
        "capability_gate_passed": passed,
    }


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
    *,
    minimum_clip_threshold: float = 1.0e-8,
) -> StructuredGPPGradientCalibration:
    """Freeze the direct-GPP warm-start clip scale from train-only gradients."""

    direct = np.asarray(tuple(direct_norms), dtype=np.float64)
    if direct.shape != (8,):
        raise ValueError("structured GPP gradient calibration requires eight direct norms")
    if not np.all(np.isfinite(direct)) or np.any(direct <= 0.0):
        raise ValueError("structured GPP calibration norms must be finite and positive")
    if not np.isfinite(minimum_clip_threshold) or minimum_clip_threshold <= 0.0:
        raise ValueError("structured GPP minimum clip threshold is invalid")
    clip = max(
        float(np.quantile(direct, 0.9, method="linear")),
        float(minimum_clip_threshold),
    )
    return StructuredGPPGradientCalibration(
        direct_norms=tuple(float(value) for value in direct),
        direct_p90_clip_threshold=clip,
        median_direct_norm=float(np.median(direct)),
    )


def specialist_element_count(tree: Any) -> int:
    return sum(int(np.prod(value.shape, dtype=np.int64)) for value in jax.tree_util.tree_leaves(tree))


__all__ = [
    "CANDIDATE_PARAMETER_COUNT",
    "DEFAULT_CONTRACT_PATH",
    "DEFAULT_WARM_START_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "SPECIALIST_PARAMETER_COUNT",
    "StructuredGPPGradientCalibration",
    "StructuredGPPTrainingPreflightContract",
    "StructuredGPPWarmStartContract",
    "WARM_START_FROZEN_CONTRACT_SHA256",
    "WARM_START_SCHEMA_VERSION",
    "calibrate_structured_gpp_gradient_norms",
    "combine_specialist_gradients",
    "compare_structured_gpp_direct_panels",
    "linearized_structured_gpp_direct_loss",
    "load_structured_gpp_training_preflight_contract",
    "load_structured_gpp_warm_start_contract",
    "replace_specialist",
    "specialist_element_count",
    "specialist_tree",
    "structured_gpp_direct_loss",
    "structured_gpp_direct_statistics",
]
