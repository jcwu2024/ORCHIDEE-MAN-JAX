"""Shared normalized-head to physical-output transformation semantics."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .neural_architecture import NeuralOutputNormalization, OutputFieldSpec
from .types import AssembledDailyOperatorInput


def field_coordinates(shape: tuple[int, ...]) -> np.ndarray:
    if not shape:
        return np.zeros((1, 0), dtype=np.int32)
    return np.stack(np.unravel_index(np.arange(np.prod(shape)), shape), axis=1).astype(np.int32)


def normalized_output_to_physical(
    raw: Any,
    active_mask: Any,
    *,
    field: OutputFieldSpec,
    normalization: NeuralOutputNormalization,
    inputs: AssembledDailyOperatorInput,
    minimum_nonnegative_ratio: float = 1.0e-12,
    minimum_unit_interval_center: float = 1.0e-6,
    rescale_unit_interval_delta: bool = True,
) -> Any:
    """Apply the accepted constraint/delta transform in float64."""

    raw = jnp.asarray(raw, dtype=jnp.float64).reshape(-1)
    active_mask = jnp.asarray(active_mask, dtype=jnp.bool_).reshape(-1)
    if field.constraint == "logit":
        return jnp.where(active_mask, raw, 0.0).reshape(field.shape)

    mean = jnp.asarray(normalization.mean[field.name], dtype=jnp.float64).reshape(-1)
    scale = jnp.asarray(normalization.scale[field.name], dtype=jnp.float64).reshape(-1)
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    if field.value_representation == "direct":
        if field.constraint == "unit_interval":
            value = jax.nn.sigmoid(raw)
        elif field.constraint == "signed_unit":
            value = jnp.tanh(raw)
        elif field.constraint == "nonnegative":
            ratio = jnp.maximum(mean / scale, minimum_nonnegative_ratio)
            baseline = ratio + jnp.log(-jnp.expm1(-ratio))
            value = jax.nn.softplus(baseline + raw) * scale
        else:
            value = mean + scale * raw
    else:
        if field.value_representation == "normalized_day_end_minus_day_start_owner":
            continuous = jnp.asarray(
                inputs.canonical_input.continuous_state,
                dtype=jnp.float64,
            )
            start = int(field.day_start_state_start)
            stop = int(field.day_start_state_stop)
            day_start = continuous[start:stop]
            day_start_defined = jnp.isfinite(day_start) & (jnp.abs(day_start) < 0.5e20)
            base = jnp.where(day_start_defined, day_start, 0.0)
        else:
            base = jnp.zeros_like(raw)
        center = base + mean
        if field.constraint == "unit_interval":
            bounded = jnp.clip(
                center,
                minimum_unit_interval_center,
                1.0 - minimum_unit_interval_center,
            )
            local_slope = jnp.maximum(bounded * (1.0 - bounded), 1.0e-6)
            baseline = jnp.log(bounded) - jnp.log1p(-bounded)
            raw_scale = (
                scale / local_slope
                if rescale_unit_interval_delta
                else jnp.ones_like(scale)
            )
            value = jax.nn.sigmoid(baseline + raw * raw_scale)
        elif field.constraint == "nonnegative":
            positive_center = jnp.maximum(
                center,
                scale * minimum_nonnegative_ratio,
            )
            ratio = positive_center / scale
            baseline = ratio + jnp.log(-jnp.expm1(-ratio))
            value = jax.nn.softplus(baseline + raw) * scale
        elif field.constraint == "signed_unit":
            bounded = jnp.clip(center, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
            local_slope = jnp.maximum(1.0 - bounded * bounded, 1.0e-6)
            value = jnp.tanh(jnp.arctanh(bounded) + raw * scale / local_slope)
        else:
            value = center + scale * raw
    return jnp.where(active_mask, value, 0.0).reshape(field.shape)


__all__ = ["field_coordinates", "normalized_output_to_physical"]
