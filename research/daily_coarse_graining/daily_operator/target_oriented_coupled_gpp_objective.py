"""Field-grouped state anchor and calibration policy for coupled GPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

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

SCHEMA_VERSION = "target_oriented_coupled_gpp_objective_calibration_v1"
FROZEN_CONTRACT_SHA256 = "20986e2d4fdefdddf2c89bd5a8ab83f64c680804d5b2b5b497c011bc34dcb825"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_objective_calibration_v1.json"
)

DIRECT_SOURCE = "direct_gpp_208_teacher_days"
STATE_SOURCE = "reachable_state_anchor_32_teacher_days"
SEVEN_SOURCE = "seven_day_component_resolved_science"
THIRTY_SOURCE = "thirty_day_component_resolved_science"
GRADIENT_SOURCES = (DIRECT_SOURCE, STATE_SOURCE, SEVEN_SOURCE, THIRTY_SOURCE)
TRAINABLE_FAMILIES = ("structured_gpp", "coupled_gpp_carbon_feedback")
TRAINABLE_PARAMETER_COUNT = 15_960_195
RETAINED_PARAMETER_COUNT = 93_504_164


@dataclass(frozen=True)
class CoupledGPPObjectiveCalibrationContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def calibration_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["calibration_artifact_updates"])


class ReachableStateAnchorLayout(NamedTuple):
    field_names: tuple[str, ...]
    global_state_indices: np.ndarray
    field_index: np.ndarray


class FieldGroupedStateAnchorResult(NamedTuple):
    total: Any
    field_loss: Mapping[str, Any]
    field_mse: Mapping[str, Any]
    field_bias_squared: Mapping[str, Any]
    field_count: Mapping[str, Any]


class CoupledGPPObjectiveCoefficients(NamedTuple):
    resolved: Mapping[str, float]
    candidates: Mapping[str, tuple[float, ...]]
    inside_interval: bool


def _inventory(payload: Mapping[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            str(record["field"]),
            tuple(int(value) for value in record["global_state_indices"]),
        )
        for record in payload["reachable_state_anchor"]["fields"]
    )


def load_coupled_gpp_objective_calibration_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPObjectiveCalibrationContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_after_reachability_for_zero_update_train_only_calibration"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("coupled GPP objective-calibration contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    anchor = payload["reachable_state_anchor"]
    objective = payload["objective"]
    calibration = payload["coefficient_calibration"]
    ownership = payload["parameter_ownership"]
    execution = payload["execution"]
    inventory = _inventory(payload)
    flat = tuple(index for _name, indices in inventory for index in indices)
    if (
        identity["reachability_report_sha256"] != "dc53765e84f14b0d08e7d9d6bafbd0527a36349f57f012dac1cf8d5e1a36d020"
        or identity["structured_checkpoint_sha256"]
        != "3b173b8980c37b49db3d1547dddad2ab98ec00c733e2f564469490ee382d4ec0"
        or int(identity["candidate_parameter_count"]) != 109_464_359
        or int(identity["trainable_parameter_count"]) != TRAINABLE_PARAMETER_COUNT
        or int(identity["retained_parameter_count"]) != RETAINED_PARAMETER_COUNT
        or tuple(int(value) for value in data["calibration_artifact_updates"])
        != (352, 353, 354, 355, 356, 357, 359, 360)
        or int(data["direct_gpp_teacher_days_per_artifact"]) != 208
        or int(data["one_day_teacher_samples_per_artifact"]) != 32
        or int(data["seven_day_windows_per_artifact"]) != 8
        or int(data["thirty_day_windows_per_artifact"]) != 4
        or int(anchor["coordinate_count"]) != 41
        or int(anchor["field_count"]) != 16
        or len(inventory) != 16
        or len(flat) != 41
        or len(set(flat)) != 41
        or tuple(objective["gradient_sources"]) != GRADIENT_SOURCES
        or calibration["reference_gradient_source"] != DIRECT_SOURCE
        or float(calibration["reference_coefficient"]) != 1.0
        or calibration["target_gradient_norm_ratios"] != {STATE_SOURCE: 0.5, SEVEN_SOURCE: 0.25, THIRTY_SOURCE: 0.25}
        or tuple(ownership["differentiated_families"])
        != tuple(f"target_oriented.{name}" for name in TRAINABLE_FAMILIES)
        or int(ownership["differentiated_parameter_count"]) != TRAINABLE_PARAMETER_COUNT
        or int(ownership["retained_parameter_count"]) != RETAINED_PARAMETER_COUNT
    ):
        raise ValueError("coupled GPP objective-calibration identity drift")
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
            "optimizer_created",
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "fresh_validation_read_authorized",
            "sealed_test_authorized",
            "365_day_rollout_authorized",
            "gate_d2_authorized",
            "product_acceptance_authorized",
            "cross_day_neural_hidden_state",
        )
    ):
        raise ValueError("coupled GPP objective calibration opens a forbidden path")
    if (
        int(execution["optimizer_updates_authorized"]) != 0
        or int(execution["checkpoint_writes_authorized"]) != 0
        or int(execution["forcing_records_reconstructed"]) != 0
        or int(execution["subdaily_state_scan_steps"]) != 0
    ):
        raise ValueError("coupled GPP objective-calibration execution drift")
    return CoupledGPPObjectiveCalibrationContract(payload, source, observed)


def build_reachable_state_anchor_layout(
    contract: CoupledGPPObjectiveCalibrationContract,
) -> ReachableStateAnchorLayout:
    inventory = _inventory(contract.payload)
    names = tuple(name for name, _indices in inventory)
    indices = np.concatenate(tuple(np.asarray(values, dtype=np.int32) for _name, values in inventory))
    fields = np.concatenate(
        tuple(np.full((len(values),), field, dtype=np.int32) for field, (_name, values) in enumerate(inventory))
    )
    return ReachableStateAnchorLayout(names, indices, fields)


def reachable_state_anchor_sufficient_statistics(
    predicted: Any,
    expected: Any,
    predicted_defined: Any,
    expected_defined: Any,
    scale: Any,
    *,
    layout: ReachableStateAnchorLayout,
) -> Mapping[str, ErrorSufficientStatistics]:
    predicted = jnp.asarray(predicted, dtype=jnp.float64)
    expected = jnp.asarray(expected, dtype=jnp.float64)
    predicted_defined = jnp.asarray(predicted_defined, dtype=jnp.bool_)
    expected_defined = jnp.asarray(expected_defined, dtype=jnp.bool_)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    if (
        predicted.shape != expected.shape
        or predicted_defined.shape != predicted.shape
        or expected_defined.shape != predicted.shape
        or scale.ndim != 1
        or predicted.shape[-1] != scale.shape[0]
    ):
        raise ValueError("reachable state-anchor array shape drift")
    indices = jnp.asarray(layout.global_state_indices, dtype=jnp.int32)
    selected_predicted = jnp.take(predicted, indices, axis=-1)
    selected_expected = jnp.take(expected, indices, axis=-1)
    selected_defined = jnp.take(predicted_defined & expected_defined, indices, axis=-1)
    selected_scale = jnp.take(scale, indices, axis=0)
    return {
        name: normalized_error_sufficient_statistics(
            selected_predicted[..., layout.field_index == field],
            selected_expected[..., layout.field_index == field],
            selected_defined[..., layout.field_index == field],
            selected_scale[layout.field_index == field],
        )
        for field, name in enumerate(layout.field_names)
    }


def add_reachable_state_anchor_statistics(
    left: Mapping[str, ErrorSufficientStatistics],
    right: Mapping[str, ErrorSufficientStatistics],
) -> Mapping[str, ErrorSufficientStatistics]:
    if tuple(left) != tuple(right):
        raise ValueError("reachable state-anchor statistic registry drift")
    return {name: jax.tree_util.tree_map(lambda a, b: a + b, left[name], right[name]) for name in left}


def field_grouped_state_anchor_from_sufficient_statistics(
    statistics: Mapping[str, ErrorSufficientStatistics],
    *,
    field_names: Sequence[str],
) -> FieldGroupedStateAnchorResult:
    names = tuple(str(name) for name in field_names)
    if tuple(statistics) != names:
        raise ValueError("reachable state-anchor field registry drift")
    errors = {name: normalized_error_from_sufficient_statistics(statistics[name]) for name in names}
    mse = {name: value.mse for name, value in errors.items()}
    bias = {name: value.bias_squared for name, value in errors.items()}
    count = {name: value.count for name, value in errors.items()}
    loss = {name: 0.5 * (mse[name] + bias[name]) for name in names}
    active = jnp.stack(tuple(count[name] > 0.0 for name in names))
    values = jnp.stack(tuple(loss[name] for name in names))
    active_count = jnp.sum(active, dtype=jnp.float64)
    total = jnp.where(
        active_count > 0.0,
        jnp.sum(jnp.where(active, values, 0.0)) / active_count,
        0.0,
    )
    return FieldGroupedStateAnchorResult(total, loss, mse, bias, count)


def linearized_reachable_state_anchor_loss(
    current: Mapping[str, ErrorSufficientStatistics],
    total: Mapping[str, ErrorSufficientStatistics],
    *,
    field_names: Sequence[str],
) -> Any:
    names = tuple(str(name) for name in field_names)
    if tuple(current) != names or tuple(total) != names:
        raise ValueError("reachable state-anchor linearized registry drift")
    active = {name: jax.lax.stop_gradient(jnp.asarray(total[name].count) > 0.0) for name in names}
    active_count = jnp.sum(jnp.stack(tuple(active.values())), dtype=jnp.float64)
    values = tuple(
        jnp.where(
            active[name],
            _linearized_error_loss(
                current[name],
                total[name],
                mse_weight=0.5,
                bias_squared_weight=0.5,
            ),
            0.0,
        )
        for name in names
    )
    return jnp.where(
        active_count > 0.0,
        jnp.sum(jnp.stack(values)) / active_count,
        0.0,
    )


def coupled_gpp_trainable_tree(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    target = parameters["target_oriented"]
    return {name: target[name] for name in TRAINABLE_FAMILIES}


def replace_coupled_gpp_trainable(
    parameters: Mapping[str, Any],
    trainable: Mapping[str, Any],
) -> Mapping[str, Any]:
    keys = frozenset(trainable)
    if keys not in (frozenset(), frozenset(TRAINABLE_FAMILIES)):
        raise ValueError("coupled GPP trainable family drift")
    target = dict(parameters["target_oriented"])
    for name in TRAINABLE_FAMILIES:
        target.pop(name, None)
    target.update(trainable)
    return dict(parameters, target_oriented=target)


def resolve_coupled_gpp_objective_coefficients(
    records: Sequence[Mapping[str, Any]],
    contract: CoupledGPPObjectiveCalibrationContract,
) -> CoupledGPPObjectiveCoefficients:
    records = tuple(records)
    calibration = contract.payload["coefficient_calibration"]
    if len(records) != len(contract.calibration_updates):
        raise ValueError("coupled GPP objective-calibration record count drift")
    minimum = float(calibration["coefficient_minimum"])
    maximum = float(calibration["coefficient_maximum"])
    targets = {str(name): float(value) for name, value in calibration["target_gradient_norm_ratios"].items()}
    candidates: dict[str, tuple[float, ...]] = {}
    resolved = {DIRECT_SOURCE: float(calibration["reference_coefficient"])}
    inside = True
    for source in GRADIENT_SOURCES[1:]:
        values = []
        for record in records:
            reports = record["source_gradients"]
            direct = float(reports[DIRECT_SOURCE]["l2_norm"])
            current = float(reports[source]["l2_norm"])
            if not np.isfinite(direct) or direct <= 0.0 or not np.isfinite(current) or current <= 0.0:
                raise ValueError("coupled GPP objective-calibration gradient norm is invalid")
            values.append(targets[source] * direct / current)
        candidates[source] = tuple(values)
        raw = float(np.median(np.asarray(values, dtype=np.float64)))
        inside = inside and minimum <= raw <= maximum
        resolved[source] = float(np.clip(raw, minimum, maximum))
    return CoupledGPPObjectiveCoefficients(resolved, candidates, inside)


def combined_gradient_p90_clip(
    norms: Sequence[float],
    *,
    minimum: float,
) -> float:
    values = np.asarray(tuple(norms), dtype=np.float64)
    if values.shape != (8,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("coupled GPP combined-gradient norms are invalid")
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("coupled GPP minimum clip is invalid")
    return max(float(np.quantile(values, 0.9, method="linear")), float(minimum))


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "DIRECT_SOURCE",
    "FROZEN_CONTRACT_SHA256",
    "GRADIENT_SOURCES",
    "RETAINED_PARAMETER_COUNT",
    "SCHEMA_VERSION",
    "SEVEN_SOURCE",
    "STATE_SOURCE",
    "THIRTY_SOURCE",
    "TRAINABLE_FAMILIES",
    "TRAINABLE_PARAMETER_COUNT",
    "CoupledGPPObjectiveCalibrationContract",
    "CoupledGPPObjectiveCoefficients",
    "FieldGroupedStateAnchorResult",
    "ReachableStateAnchorLayout",
    "add_reachable_state_anchor_statistics",
    "build_reachable_state_anchor_layout",
    "combined_gradient_p90_clip",
    "coupled_gpp_trainable_tree",
    "field_grouped_state_anchor_from_sufficient_statistics",
    "linearized_reachable_state_anchor_loss",
    "load_coupled_gpp_objective_calibration_contract",
    "reachable_state_anchor_sufficient_statistics",
    "replace_coupled_gpp_trainable",
    "resolve_coupled_gpp_objective_coefficients",
]
