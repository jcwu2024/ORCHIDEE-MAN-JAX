"""Opt-in neural feature conditioning; never modifies the carried physical state."""
from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

CONTINUOUS_TAIL_VERSION = "continuous_log_tail_v1"
CONTINUOUS_TAIL_THRESHOLD = 16.0


def standardize_continuous(
    value: Any, mean: Any, scale: Any, threshold: float | None = None,
    *, dtype: Any = jnp.float32, legacy_standardize: Callable | None = None,
    coordinate_mask: Any = None,
) -> Any:
    """Preserve legacy arithmetic by default; optionally compress only tails.

    Threshold is static model configuration. Enabled arithmetic uses float64
    before the neural dtype conversion, so e.g. 0.048 / 8e-12 does not enter
    Attention as a six-billion-valued float32 feature. No stop-gradient or
    surrogate derivative is used.
    """
    if threshold is None:
        if legacy_standardize is not None:
            return legacy_standardize(value, mean, scale)
        value, mean, scale = (jnp.asarray(x, dtype=dtype) for x in (value, mean, scale))
        safe_scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
        return jnp.where(jnp.isfinite(value), (value - mean) / safe_scale, 0.0)
    if not 0.0 < threshold < float("inf"):
        raise ValueError("continuous tail threshold must be positive and finite")
    if coordinate_mask is not None:
        selected = jnp.asarray(coordinate_mask, dtype=jnp.bool_)
        # Non-selected features use their exact historical arithmetic. Hide
        # selected values from the inactive legacy branch before its division.
        legacy = standardize_continuous(
            jnp.where(selected, mean, value), mean, scale, dtype=dtype,
            legacy_standardize=legacy_standardize,
        )
        conditioned = standardize_continuous(
            jnp.where(selected, value, mean), mean, scale, threshold, dtype=dtype,
        )
        return jnp.where(selected, conditioned, legacy)
    value, mean, scale = (jnp.asarray(x, dtype=jnp.float64) for x in (value, mean, scale))
    finite = jnp.isfinite(value)
    safe_scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    z = (jnp.where(finite, value, mean) - mean) / safe_scale
    outside = jnp.abs(z) > threshold
    # Sanitize the inactive log branch before evaluation, including its AD.
    excess = jnp.where(outside, jnp.abs(z) - threshold, 0.0)
    tail = jnp.sign(z) * (threshold + jnp.log1p(excess))
    return jnp.where(finite, jnp.where(outside, tail, z), 0.0).astype(dtype)
