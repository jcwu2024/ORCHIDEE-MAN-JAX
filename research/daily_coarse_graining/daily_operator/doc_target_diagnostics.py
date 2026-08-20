"""Host-side statistics for the frozen DOC one-point training targets."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _validated_arrays(
    value: Any,
    defined: Any,
    scale: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(value, dtype=np.float64)
    mask = np.asarray(defined, dtype=np.bool_)
    if values.shape != mask.shape or values.ndim < 1:
        raise ValueError("DOC diagnostic values and masks must share a day-first shape")
    scales = np.asarray(scale, dtype=np.float64)
    try:
        scales = np.broadcast_to(scales, values.shape)
    except ValueError as error:
        raise ValueError("DOC diagnostic scale does not broadcast to the target") from error
    valid = mask & np.isfinite(values) & np.isfinite(scales) & (scales > 0.0)
    normalized = np.divide(
        values,
        scales,
        out=np.zeros_like(values),
        where=np.isfinite(scales) & (scales > 0.0),
    )
    return values, mask, valid, normalized


def _quantiles(values: np.ndarray) -> Mapping[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("minimum", "p50", "p90", "p99", "maximum")}
    quantiles = np.quantile(values, (0.0, 0.5, 0.9, 0.99, 1.0))
    return {
        name: float(value)
        for name, value in zip(
            ("minimum", "p50", "p90", "p99", "maximum"),
            quantiles,
            strict=True,
        )
    }


def masked_field_statistics(
    value: Any,
    defined: Any,
    scale: Any,
    *,
    meaningful_normalized_magnitude: float = 1.0e-8,
) -> Mapping[str, Any]:
    """Summarize one day-first target without using model predictions."""

    if meaningful_normalized_magnitude < 0.0:
        raise ValueError("meaningful normalized magnitude must be nonnegative")
    values, mask, valid, normalized = _validated_arrays(value, defined, scale)
    flat_valid = valid.reshape(values.shape[0], -1)
    flat = normalized.reshape(values.shape[0], -1)
    selected = flat[flat_valid]
    count = int(selected.size)

    exact_nonzero = flat_valid & (flat != 0.0)
    meaningful = flat_valid & (np.abs(flat) > meaningful_normalized_magnitude)
    positive = flat_valid & (flat > meaningful_normalized_magnitude)
    negative = flat_valid & (flat < -meaningful_normalized_magnitude)

    coordinate_count = np.sum(flat_valid, axis=0, dtype=np.int64)
    coordinate_sum = np.sum(np.where(flat_valid, flat, 0.0), axis=0, dtype=np.float64)
    coordinate_mean = np.divide(
        coordinate_sum,
        coordinate_count,
        out=np.zeros_like(coordinate_sum),
        where=coordinate_count > 0,
    )
    coordinate_residual = np.where(flat_valid, flat - coordinate_mean[None, :], 0.0)
    coordinate_sse = float(np.sum(coordinate_residual * coordinate_residual, dtype=np.float64))
    zero_sse = float(np.sum(np.where(flat_valid, flat * flat, 0.0), dtype=np.float64))
    global_mean = float(np.mean(selected, dtype=np.float64)) if count else 0.0
    global_residual = np.where(flat_valid, flat - global_mean, 0.0)
    global_sse = float(np.sum(global_residual * global_residual, dtype=np.float64))
    coordinate_nonzero_days = np.sum(meaningful, axis=0, dtype=np.int64)
    admitted_coordinate_counts = coordinate_nonzero_days[coordinate_count > 0]

    return {
        "shape": [int(size) for size in values.shape],
        "declared_defined_count": int(np.count_nonzero(mask)),
        "valid_scaled_count": count,
        "nonfinite_or_invalid_scale_count": int(np.count_nonzero(mask & ~valid)),
        "exact_nonzero_count": int(np.count_nonzero(exact_nonzero)),
        "meaningful_nonzero_count": int(np.count_nonzero(meaningful)),
        "meaningful_nonzero_fraction": (
            float(np.count_nonzero(meaningful) / count) if count else None
        ),
        "positive_count": int(np.count_nonzero(positive)),
        "negative_count": int(np.count_nonzero(negative)),
        "active_day_count": int(np.count_nonzero(np.any(meaningful, axis=1))),
        "active_coordinate_count": int(np.count_nonzero(coordinate_nonzero_days > 0)),
        "defined_coordinate_count": int(np.count_nonzero(coordinate_count > 0)),
        "normalized_signed_quantiles": _quantiles(selected),
        "normalized_absolute_quantiles": _quantiles(np.abs(selected)),
        "meaningful_nonzero_absolute_quantiles": _quantiles(np.abs(flat[meaningful])),
        "per_coordinate_nonzero_day_count_quantiles": _quantiles(
            admitted_coordinate_counts.astype(np.float64)
        ),
        "zero_predictor_mse": zero_sse / count if count else None,
        "global_mean_predictor": global_mean if count else None,
        "global_mean_predictor_mse": global_sse / count if count else None,
        "coordinate_mean_predictor_mse": coordinate_sse / count if count else None,
        "coordinate_mean_residual_fraction_of_zero_mse": (
            coordinate_sse / zero_sse if zero_sse > 0.0 else None
        ),
    }


def paired_normalized_statistics(
    left_value: Any,
    left_defined: Any,
    left_scale: Any,
    right_value: Any,
    right_defined: Any,
    right_scale: Any,
    *,
    meaningful_normalized_magnitude: float = 1.0e-8,
) -> Mapping[str, Any]:
    """Compare same-coordinate normalized targets across all admitted days."""

    _left, _left_mask, left_valid, left = _validated_arrays(
        left_value,
        left_defined,
        left_scale,
    )
    _right, _right_mask, right_valid, right = _validated_arrays(
        right_value,
        right_defined,
        right_scale,
    )
    if left.shape != right.shape:
        raise ValueError("paired DOC fields must have the same day-first shape")
    valid = left_valid & right_valid
    left_selected = left[valid]
    right_selected = right[valid]

    def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
        if first.size < 2 or float(np.std(first)) == 0.0 or float(np.std(second)) == 0.0:
            return None
        return float(np.corrcoef(first, second)[0, 1])

    left_active = valid & (np.abs(left) > meaningful_normalized_magnitude)
    right_active = valid & (np.abs(right) > meaningful_normalized_magnitude)
    union = left_active | right_active
    intersection = left_active & right_active
    return {
        "paired_count": int(left_selected.size),
        "signed_correlation": correlation(left_selected, right_selected),
        "absolute_magnitude_correlation": correlation(
            np.abs(left_selected),
            np.abs(right_selected),
        ),
        "left_active_count": int(np.count_nonzero(left_active)),
        "right_active_count": int(np.count_nonzero(right_active)),
        "coactive_count": int(np.count_nonzero(intersection)),
        "active_union_count": int(np.count_nonzero(union)),
        "active_jaccard": (
            float(np.count_nonzero(intersection) / np.count_nonzero(union))
            if np.any(union)
            else None
        ),
    }


def inventory_partition_statistics(
    incoming_value: Any,
    incoming_defined: Any,
    outgoing_value: Any,
    outgoing_defined: Any,
    *,
    meaningful_magnitude: float = 1.0e-12,
) -> Mapping[str, Any]:
    """Verify the mutually exclusive endpoint partition used for DOC state."""

    incoming = np.asarray(incoming_value, dtype=np.float64)
    outgoing = np.asarray(outgoing_value, dtype=np.float64)
    incoming_mask = np.asarray(incoming_defined, dtype=np.bool_)
    outgoing_mask = np.asarray(outgoing_defined, dtype=np.bool_)
    if (
        incoming.shape != outgoing.shape
        or incoming_mask.shape != incoming.shape
        or outgoing_mask.shape != incoming.shape
    ):
        raise ValueError("DOC incoming/outgoing endpoint fields must have matching shapes")
    valid = incoming_mask & outgoing_mask & np.isfinite(incoming) & np.isfinite(outgoing)
    incoming_active = valid & (incoming > meaningful_magnitude)
    outgoing_active = valid & (outgoing > meaningful_magnitude)
    return {
        "paired_count": int(np.count_nonzero(valid)),
        "incoming_negative_count": int(np.count_nonzero(valid & (incoming < 0.0))),
        "outgoing_out_of_unit_interval_count": int(
            np.count_nonzero(valid & ((outgoing < 0.0) | (outgoing > 1.0)))
        ),
        "simultaneously_active_count": int(np.count_nonzero(incoming_active & outgoing_active)),
        "incoming_only_count": int(np.count_nonzero(incoming_active & ~outgoing_active)),
        "outgoing_only_count": int(np.count_nonzero(outgoing_active & ~incoming_active)),
        "unchanged_count": int(np.count_nonzero(valid & ~incoming_active & ~outgoing_active)),
        "mutually_exclusive": not bool(np.any(incoming_active & outgoing_active)),
    }


__all__ = [
    "inventory_partition_statistics",
    "masked_field_statistics",
    "paired_normalized_statistics",
]
