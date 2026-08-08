"""Architecture-independent process supervision and local one-day metrics."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from .training_data import MaskedProcessTarget
from .types import DailyOperatorResult, StateDefinedMasks, StateFeatureGroups


class ProcessLossResult(NamedTuple):
    total: Any
    label_mse: Mapping[str, Any]
    label_count: Mapping[str, Any]
    field_mse: Mapping[str, Any]
    field_count: Mapping[str, Any]


def unit_process_loss_scales(
    targets: Mapping[str, MaskedProcessTarget],
) -> dict[str, Any]:
    """Build explicit unit scales for plumbing tests, never formal training."""

    return {
        name: np.ones_like(np.asarray(target.value), dtype=np.float64)
        for name, target in targets.items()
    }


def masked_equal_label_macro_mse(
    predicted: Mapping[str, Any],
    targets: Mapping[str, MaskedProcessTarget],
    *,
    scales: Mapping[str, Any],
    field_labels: Mapping[str, str],
) -> ProcessLossResult:
    """Compute finite masked MSE within labels and equal-weight labels.

    Fields within one scientific label are pooled by their valid element
    counts. Labels are then averaged with equal weight, so tensor size cannot
    make the OK_LEAK family dominate a small daily process family.
    """

    expected = set(targets)
    for name, mapping in (
        ("prediction", predicted),
        ("scale", scales),
        ("field-label", field_labels),
    ):
        if set(mapping) != expected:
            missing = sorted(expected - set(mapping))
            extra = sorted(set(mapping) - expected)
            raise ValueError(f"process {name} registry drift; missing={missing}, extra={extra}")

    field_sse: dict[str, Any] = {}
    field_count: dict[str, Any] = {}
    field_mse: dict[str, Any] = {}
    label_sse: dict[str, Any] = {}
    label_count: dict[str, Any] = {}
    for name in targets:
        prediction = jnp.asarray(predicted[name], dtype=jnp.float64)
        target = jnp.asarray(targets[name].value, dtype=jnp.float64)
        defined = jnp.asarray(targets[name].defined, dtype=jnp.bool_)
        scale = jnp.asarray(scales[name], dtype=jnp.float64)
        if prediction.shape != target.shape or defined.shape != target.shape:
            raise ValueError(f"process field shape drift for {name}")
        try:
            scale = jnp.broadcast_to(scale, target.shape)
        except ValueError as error:
            raise ValueError(f"process scale shape drift for {name}") from error
        valid_scale = jnp.isfinite(scale) & (scale > 0.0)
        safe_scale = jnp.where(valid_scale, scale, jnp.nan)
        safe_prediction = jnp.where(defined, prediction, 0.0)
        safe_target = jnp.where(defined, target, 0.0)
        residual = (safe_prediction - safe_target) / safe_scale
        sse = jnp.sum(residual * residual, dtype=jnp.float64)
        count = jnp.sum(defined, dtype=jnp.float64)
        mse = jnp.where(count > 0.0, sse / count, 0.0)
        field_sse[name] = sse
        field_count[name] = count
        field_mse[name] = mse
        label = field_labels[name]
        label_sse[label] = label_sse.get(label, jnp.asarray(0.0, dtype=jnp.float64)) + sse
        label_count[label] = label_count.get(label, jnp.asarray(0.0, dtype=jnp.float64)) + count

    label_mse = {
        name: jnp.where(label_count[name] > 0.0, value / label_count[name], 0.0)
        for name, value in label_sse.items()
    }
    active = jnp.stack(tuple(count > 0.0 for count in label_count.values()))
    values = jnp.stack(tuple(label_mse.values()))
    active_count = jnp.sum(active, dtype=jnp.float64)
    total = jnp.where(
        active_count > 0.0,
        jnp.sum(jnp.where(active, values, 0.0), dtype=jnp.float64) / active_count,
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    return ProcessLossResult(
        total=total,
        label_mse=label_mse,
        label_count=label_count,
        field_mse=field_mse,
        field_count=field_count,
    )


def _group_metrics(
    predicted: StateFeatureGroups,
    expected: StateFeatureGroups,
    defined: StateDefinedMasks,
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name in StateFeatureGroups._fields:
        actual = np.asarray(getattr(predicted, name), dtype=np.float64)
        wanted = np.asarray(getattr(expected, name), dtype=np.float64)
        mask = np.asarray(getattr(defined, name), dtype=np.bool_)
        if actual.shape != wanted.shape or mask.shape != wanted.shape:
            raise ValueError(f"one-day state shape drift for {name}")
        valid = mask & np.isfinite(actual) & np.isfinite(wanted)
        residual = np.where(valid, actual - wanted, 0.0)
        count = int(np.count_nonzero(valid))
        result[name] = {
            "count": count,
            "rmse": float(np.sqrt(np.sum(residual * residual) / count)) if count else 0.0,
            "mean_bias": float(np.sum(residual) / count) if count else 0.0,
            "maximum_absolute_error": float(np.max(np.abs(residual[valid]))) if count else 0.0,
            "nonfinite_prediction_count": int(np.count_nonzero(mask & ~np.isfinite(actual))),
        }
    return result


def one_day_acceptance_metrics(
    result: DailyOperatorResult,
    *,
    expected_state: StateFeatureGroups,
    expected_state_defined: StateDefinedMasks,
    expected_discrete_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return fieldwise state, budget, inventory, mask, and discrete metrics."""

    actual_discrete = result.next_discrete_state
    if set(actual_discrete) != set(expected_discrete_state):
        raise ValueError("one-day discrete-state registry drift")
    discrete_mismatches = sum(
        int(
            np.count_nonzero(
                np.asarray(actual_discrete[name]) != np.asarray(expected_discrete_state[name])
            )
        )
        for name in expected_discrete_state
    )
    mask_mismatches = sum(
        int(
            np.count_nonzero(
                np.asarray(getattr(result.next_state_defined, name), dtype=np.bool_)
                != np.asarray(getattr(expected_state_defined, name), dtype=np.bool_)
            )
        )
        for name in StateDefinedMasks._fields
    )
    invalid_inventory = {}
    for name, domain in (
        ("water", result.fast_update.water),
        ("carbon", result.fast_update.carbon),
    ):
        inventory = np.asarray(domain.inventory, dtype=np.float64)
        defined = np.asarray(domain.defined, dtype=np.bool_)
        invalid_inventory[name] = int(
            np.count_nonzero(defined & (~np.isfinite(inventory) | (inventory < -2.0e-12)))
        )
    return {
        "state": _group_metrics(
            result.next_state,
            expected_state,
            expected_state_defined,
        ),
        "water_budget_residual": float(np.asarray(result.fast_update.water.budget_residual)),
        "carbon_budget_residual": float(np.asarray(result.fast_update.carbon.budget_residual)),
        "invalid_inventory_count": invalid_inventory,
        "defined_mask_mismatch_count": mask_mismatches,
        "discrete_state_mismatch_count": discrete_mismatches,
    }


def validate_local_one_day_hard_gates(
    metrics: Mapping[str, Any],
    *,
    budget_atol: float = 2.0e-12,
) -> None:
    """Fail on hard local gates without imposing an accuracy promotion gate."""

    for domain in ("water", "carbon"):
        residual = abs(float(metrics[f"{domain}_budget_residual"]))
        if not np.isfinite(residual) or residual > budget_atol:
            raise ValueError(f"{domain} one-day budget residual exceeds the local gate")
        if int(metrics["invalid_inventory_count"][domain]) != 0:
            raise ValueError(f"{domain} one-day inventory is invalid")
    if int(metrics["defined_mask_mismatch_count"]) != 0:
        raise ValueError("one-day defined-status masks are not exact")
    if int(metrics["discrete_state_mismatch_count"]) != 0:
        raise ValueError("one-day discrete state is not exact")
    if any(
        int(group["nonfinite_prediction_count"]) != 0
        for group in metrics["state"].values()
    ):
        raise ValueError("one-day state contains a nonfinite defined prediction")
