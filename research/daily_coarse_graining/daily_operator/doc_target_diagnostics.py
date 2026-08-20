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


def paired_daily_aggregate_statistics(
    left_value: Any,
    left_defined: Any,
    left_scale: Any,
    right_value: Any,
    right_defined: Any,
    right_scale: Any,
    *,
    meaningful_normalized_magnitude: float = 1.0e-8,
) -> Mapping[str, Any]:
    """Compare daily activity across targets with different coordinate axes."""

    if meaningful_normalized_magnitude < 0.0:
        raise ValueError("meaningful normalized magnitude must be nonnegative")
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
    if left.shape[0] != right.shape[0]:
        raise ValueError("paired DOC fields must have the same day count")

    def daily_means(
        normalized: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = normalized.reshape(normalized.shape[0], -1)
        flat_valid = valid.reshape(valid.shape[0], -1)
        valid_count = np.sum(flat_valid, axis=1, dtype=np.int64)
        signed_sum = np.sum(np.where(flat_valid, flat, 0.0), axis=1, dtype=np.float64)
        absolute_sum = np.sum(
            np.where(flat_valid, np.abs(flat), 0.0),
            axis=1,
            dtype=np.float64,
        )
        signed_mean = np.divide(
            signed_sum,
            valid_count,
            out=np.zeros_like(signed_sum),
            where=valid_count > 0,
        )
        absolute_mean = np.divide(
            absolute_sum,
            valid_count,
            out=np.zeros_like(absolute_sum),
            where=valid_count > 0,
        )
        active = np.any(
            flat_valid & (np.abs(flat) > meaningful_normalized_magnitude),
            axis=1,
        )
        return signed_mean, absolute_mean, active

    def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
        if first.size < 2 or float(np.std(first)) == 0.0 or float(np.std(second)) == 0.0:
            return None
        return float(np.corrcoef(first, second)[0, 1])

    left_signed, left_absolute, left_active = daily_means(left, left_valid)
    right_signed, right_absolute, right_active = daily_means(right, right_valid)
    left_admitted = np.any(left_valid.reshape(left.shape[0], -1), axis=1)
    right_admitted = np.any(right_valid.reshape(right.shape[0], -1), axis=1)
    paired_days = left_admitted & right_admitted
    union = paired_days & (left_active | right_active)
    intersection = paired_days & left_active & right_active
    return {
        "paired_day_count": int(np.count_nonzero(paired_days)),
        "signed_daily_mean_correlation": correlation(
            left_signed[paired_days],
            right_signed[paired_days],
        ),
        "absolute_daily_mean_correlation": correlation(
            left_absolute[paired_days],
            right_absolute[paired_days],
        ),
        "left_active_day_count": int(np.count_nonzero(paired_days & left_active)),
        "right_active_day_count": int(np.count_nonzero(paired_days & right_active)),
        "coactive_day_count": int(np.count_nonzero(intersection)),
        "active_day_union_count": int(np.count_nonzero(union)),
        "active_day_jaccard": (
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


def _geometry_quantiles(values: np.ndarray) -> Mapping[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            name: None
            for name in ("minimum", "p01", "p10", "p50", "p90", "p99", "maximum")
        }
    quantiles = np.quantile(finite, (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0))
    return {
        name: float(value)
        for name, value in zip(
            ("minimum", "p01", "p10", "p50", "p90", "p99", "maximum"),
            quantiles,
            strict=True,
        )
    }


def _inverse_softplus(value: np.ndarray) -> np.ndarray:
    result = np.full(value.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(value) & (value > 0.0)
    result[positive] = value[positive] + np.log(-np.expm1(-value[positive]))
    result[np.isfinite(value) & (value == 0.0)] = -np.inf
    return result


def _broadcast_geometry_array(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    try:
        return np.broadcast_to(np.asarray(value, dtype=dtype), shape)
    except ValueError as error:
        raise ValueError(f"DOC geometry {name} does not broadcast to the endpoint shape") from error


def _selected_geometry_report(values: np.ndarray, selected: np.ndarray) -> Mapping[str, Any]:
    chosen = np.asarray(values, dtype=np.float64)[selected]
    finite = np.isfinite(chosen)
    return {
        "count": int(chosen.size),
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "quantiles": _geometry_quantiles(chosen[finite]),
    }


def inventory_transition_geometry_statistics(
    incoming_value: Any,
    incoming_defined: Any,
    outgoing_value: Any,
    outgoing_defined: Any,
    day_start_value: Any,
    day_start_defined: Any,
    *,
    incoming_output_mean: Any,
    incoming_output_scale: Any,
    outgoing_output_mean: Any,
    outgoing_output_scale: Any,
    incoming_loss_scale: Any,
    outgoing_loss_scale: Any,
    current_minimum_inventory_ratio: float = 0.05,
    separate_minimum_nonnegative_ratio: float = 0.05,
    meaningful_magnitude: float = 1.0e-12,
    normalized_jacobian_floor: float = 1.0e-6,
) -> Mapping[str, Any]:
    """Audit inverse targets and local loss conditioning for DOC endpoints."""

    if not 0.0 < current_minimum_inventory_ratio < 1.0:
        raise ValueError("current DOC inventory ratio must lie in (0, 1)")
    if not 0.0 < separate_minimum_nonnegative_ratio < 1.0:
        raise ValueError("separate DOC nonnegative ratio must lie in (0, 1)")
    if meaningful_magnitude < 0.0 or normalized_jacobian_floor < 0.0:
        raise ValueError("DOC geometry thresholds must be nonnegative")

    incoming = np.asarray(incoming_value, dtype=np.float64)
    outgoing = np.asarray(outgoing_value, dtype=np.float64)
    if incoming.shape != outgoing.shape or incoming.ndim < 1:
        raise ValueError("DOC geometry endpoints must share a non-scalar shape")
    shape = incoming.shape
    incoming_mask = _broadcast_geometry_array(
        "incoming mask", incoming_defined, shape, dtype=np.bool_
    )
    outgoing_mask = _broadcast_geometry_array(
        "outgoing mask", outgoing_defined, shape, dtype=np.bool_
    )
    day_start = _broadcast_geometry_array("day-start value", day_start_value, shape)
    day_start_mask = _broadcast_geometry_array(
        "day-start mask", day_start_defined, shape, dtype=np.bool_
    )
    incoming_mean = _broadcast_geometry_array(
        "incoming output mean", incoming_output_mean, shape
    )
    incoming_scale = _broadcast_geometry_array(
        "incoming output scale", incoming_output_scale, shape
    )
    outgoing_mean = _broadcast_geometry_array(
        "outgoing output mean", outgoing_output_mean, shape
    )
    outgoing_scale = _broadcast_geometry_array(
        "outgoing output scale", outgoing_output_scale, shape
    )
    incoming_loss = _broadcast_geometry_array(
        "incoming loss scale", incoming_loss_scale, shape
    )
    outgoing_loss = _broadcast_geometry_array(
        "outgoing loss scale", outgoing_loss_scale, shape
    )

    declared = incoming_mask & outgoing_mask & day_start_mask
    finite = (
        np.isfinite(incoming)
        & np.isfinite(outgoing)
        & np.isfinite(day_start)
        & np.isfinite(incoming_mean)
        & np.isfinite(incoming_scale)
        & np.isfinite(outgoing_mean)
        & np.isfinite(outgoing_scale)
        & np.isfinite(incoming_loss)
        & np.isfinite(outgoing_loss)
    )
    positive_scales = (
        (incoming_scale > 0.0)
        & (outgoing_scale > 0.0)
        & (incoming_loss > 0.0)
        & (outgoing_loss > 0.0)
    )
    physical = (
        (incoming >= 0.0)
        & (outgoing >= 0.0)
        & (outgoing <= 1.0)
        & (day_start >= 0.0)
    )
    valid = declared & finite & positive_scales & physical
    incoming_active = valid & (incoming > meaningful_magnitude)
    outgoing_active = valid & (outgoing > meaningful_magnitude)
    simultaneous = incoming_active & outgoing_active
    incoming_only = incoming_active & ~outgoing_active
    outgoing_only = outgoing_active & ~incoming_active
    unchanged = valid & ~incoming_active & ~outgoing_active

    target_change = incoming - day_start * outgoing
    target_end = day_start + target_change
    transition_scale = np.sqrt(
        incoming_scale * incoming_scale + (day_start * outgoing_scale) ** 2
    )
    center_change = incoming_mean - day_start * outgoing_mean
    center_end = np.maximum(
        day_start + center_change,
        transition_scale * current_minimum_inventory_ratio,
    )
    center_ratio = center_end / transition_scale
    center_baseline = _inverse_softplus(center_ratio)
    target_ratio = target_end / transition_scale
    joint_raw = _inverse_softplus(target_ratio) - center_baseline
    finite_joint_inverse = valid & np.isfinite(joint_raw)
    joint_end = np.logaddexp(0.0, center_baseline + joint_raw) * transition_scale
    joint_reconstruction_error = np.abs(joint_end - target_end)
    joint_sigmoid = np.exp(-np.logaddexp(0.0, -(center_baseline + joint_raw)))
    joint_end_jacobian = joint_sigmoid * transition_scale
    joint_incoming_jacobian = joint_end_jacobian / incoming_loss
    safe_start = np.where(day_start > 0.0, day_start, 1.0)
    joint_outgoing_jacobian = joint_end_jacobian / (safe_start * outgoing_loss)

    incoming_center_ratio = np.maximum(
        incoming_mean / incoming_scale,
        separate_minimum_nonnegative_ratio,
    )
    incoming_baseline = _inverse_softplus(incoming_center_ratio)
    incoming_target_ratio = incoming / incoming_scale
    separate_incoming_raw = _inverse_softplus(incoming_target_ratio) - incoming_baseline
    separate_incoming_sigmoid = 1.0 - np.exp(-incoming_target_ratio)
    separate_incoming_jacobian = (
        separate_incoming_sigmoid * incoming_scale / incoming_loss
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        separate_outgoing_raw = np.log(outgoing) - np.log1p(-outgoing)
    separate_outgoing_jacobian = outgoing * (1.0 - outgoing) / outgoing_loss

    center_delta = center_end - day_start
    center_incoming = np.maximum(center_delta, 0.0)
    center_outgoing = np.where(
        day_start > 0.0,
        np.clip(np.maximum(-center_delta, 0.0) / safe_start, 0.0, 1.0),
        0.0,
    )
    incoming_zero_sse = float(
        np.sum(np.where(valid, (incoming / incoming_loss) ** 2, 0.0), dtype=np.float64)
    )
    outgoing_zero_sse = float(
        np.sum(np.where(valid, (outgoing / outgoing_loss) ** 2, 0.0), dtype=np.float64)
    )
    incoming_center_sse = float(
        np.sum(
            np.where(valid, ((center_incoming - incoming) / incoming_loss) ** 2, 0.0),
            dtype=np.float64,
        )
    )
    outgoing_center_sse = float(
        np.sum(
            np.where(valid, ((center_outgoing - outgoing) / outgoing_loss) ** 2, 0.0),
            dtype=np.float64,
        )
    )
    valid_count = int(np.count_nonzero(valid))
    combined_zero_sse = incoming_zero_sse + outgoing_zero_sse

    joint_incoming = _selected_geometry_report(joint_incoming_jacobian, incoming_only)
    joint_outgoing = _selected_geometry_report(joint_outgoing_jacobian, outgoing_only)
    separate_incoming = _selected_geometry_report(
        separate_incoming_jacobian, incoming_only
    )
    separate_outgoing = _selected_geometry_report(
        separate_outgoing_jacobian, outgoing_only
    )
    joint_incoming_median = joint_incoming["quantiles"]["p50"]
    joint_outgoing_median = joint_outgoing["quantiles"]["p50"]
    median_ratio = (
        joint_incoming_median / joint_outgoing_median
        if joint_incoming_median is not None
        and joint_outgoing_median is not None
        and joint_outgoing_median > 0.0
        else None
    )

    current_active = incoming_only | outgoing_only
    current_nonfinite_active_inverse = int(
        np.count_nonzero(current_active & ~np.isfinite(joint_raw))
    )
    current_below_floor = int(
        np.count_nonzero(
            incoming_only & (joint_incoming_jacobian <= normalized_jacobian_floor)
        )
        + np.count_nonzero(
            outgoing_only & (joint_outgoing_jacobian <= normalized_jacobian_floor)
        )
    )
    median_ratio_balanced = median_ratio is not None and 0.1 <= median_ratio <= 10.0

    return {
        "shape": [int(size) for size in shape],
        "declared_count": int(np.count_nonzero(declared)),
        "valid_count": valid_count,
        "invalid_declared_count": int(np.count_nonzero(declared & ~valid)),
        "partition": {
            "incoming_only_count": int(np.count_nonzero(incoming_only)),
            "outgoing_only_count": int(np.count_nonzero(outgoing_only)),
            "unchanged_count": int(np.count_nonzero(unchanged)),
            "simultaneously_active_count": int(np.count_nonzero(simultaneous)),
            "zero_day_start_count": int(np.count_nonzero(valid & (day_start == 0.0))),
            "zero_target_end_count": int(np.count_nonzero(valid & (target_end == 0.0))),
            "unit_outgoing_count": int(np.count_nonzero(valid & (outgoing == 1.0))),
        },
        "target": {
            "day_start": _selected_geometry_report(day_start, valid),
            "day_end": _selected_geometry_report(target_end, valid),
            "signed_change": _selected_geometry_report(target_change, valid),
            "end_over_joint_scale": _selected_geometry_report(target_ratio, valid),
            "incoming_loss_scaled_active_magnitude": _selected_geometry_report(
                incoming / incoming_loss, incoming_only
            ),
            "outgoing_loss_scaled_active_magnitude": _selected_geometry_report(
                outgoing / outgoing_loss, outgoing_only
            ),
        },
        "loss_geometry": {
            "valid_count_per_field": valid_count,
            "incoming_zero_predictor_mse": (
                incoming_zero_sse / valid_count if valid_count else None
            ),
            "outgoing_zero_predictor_mse": (
                outgoing_zero_sse / valid_count if valid_count else None
            ),
            "incoming_fraction_of_combined_zero_predictor_sse": (
                incoming_zero_sse / combined_zero_sse if combined_zero_sse > 0.0 else None
            ),
            "outgoing_fraction_of_combined_zero_predictor_sse": (
                outgoing_zero_sse / combined_zero_sse if combined_zero_sse > 0.0 else None
            ),
            "joint_raw_zero_center_incoming_mse": (
                incoming_center_sse / valid_count if valid_count else None
            ),
            "joint_raw_zero_center_outgoing_mse": (
                outgoing_center_sse / valid_count if valid_count else None
            ),
        },
        "current_joint_signed_day_end": {
            "inverse_raw": _selected_geometry_report(joint_raw, valid),
            "active_nonfinite_inverse_count": current_nonfinite_active_inverse,
            "finite_inverse_maximum_reconstruction_error": (
                float(np.max(joint_reconstruction_error[finite_joint_inverse]))
                if np.any(finite_joint_inverse)
                else None
            ),
            "incoming_active_normalized_jacobian": joint_incoming,
            "outgoing_active_normalized_jacobian": joint_outgoing,
            "active_jacobian_at_or_below_floor_count": current_below_floor,
            "incoming_to_outgoing_median_jacobian_ratio": median_ratio,
            "median_jacobian_ratio_within_0p1_to_10": median_ratio_balanced,
            "conditioning_passed": (
                current_nonfinite_active_inverse == 0
                and current_below_floor == 0
                and median_ratio_balanced
            ),
        },
        "separate_endpoint_transforms": {
            "incoming_active_inverse_raw": _selected_geometry_report(
                separate_incoming_raw, incoming_only
            ),
            "outgoing_active_inverse_raw": _selected_geometry_report(
                separate_outgoing_raw, outgoing_only
            ),
            "incoming_active_normalized_jacobian": separate_incoming,
            "outgoing_active_normalized_jacobian": separate_outgoing,
            "active_nonfinite_inverse_count": int(
                np.count_nonzero(incoming_only & ~np.isfinite(separate_incoming_raw))
                + np.count_nonzero(outgoing_only & ~np.isfinite(separate_outgoing_raw))
            ),
            "active_jacobian_at_or_below_floor_count": int(
                np.count_nonzero(
                    incoming_only
                    & (separate_incoming_jacobian <= normalized_jacobian_floor)
                )
                + np.count_nonzero(
                    outgoing_only
                    & (separate_outgoing_jacobian <= normalized_jacobian_floor)
                )
            ),
        },
        "status_plus_native_magnitude": {
            "status_names": ["incoming", "unchanged", "outgoing"],
            "status_counts": [
                int(np.count_nonzero(incoming_only)),
                int(np.count_nonzero(unchanged)),
                int(np.count_nonzero(outgoing_only)),
            ],
            "incoming_loss_scaled_magnitude": _selected_geometry_report(
                incoming / incoming_loss, incoming_only
            ),
            "outgoing_loss_scaled_magnitude": _selected_geometry_report(
                outgoing / outgoing_loss, outgoing_only
            ),
            "finite_target_count": int(
                np.count_nonzero(
                    (incoming_only & np.isfinite(incoming / incoming_loss))
                    | (outgoing_only & np.isfinite(outgoing / outgoing_loss))
                    | unchanged
                )
            ),
        },
    }


__all__ = [
    "inventory_partition_statistics",
    "inventory_transition_geometry_statistics",
    "masked_field_statistics",
    "paired_daily_aggregate_statistics",
    "paired_normalized_statistics",
]
