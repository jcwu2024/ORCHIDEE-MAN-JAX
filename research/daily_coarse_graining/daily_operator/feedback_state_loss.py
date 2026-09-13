"""Supervise a diagnosed recurrent state in its actual neural input units.

This is loss geometry, not a physical transition or input-normalization change.
The caller declares the field and supplies its frozen input scales.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp


class FeedbackStateLossResult(NamedTuple):
    loss: Any
    count: Any
    expected_count: Any
    invalid_count: Any
    predicted: Any
    teacher: Any
    physical_residual: Any
    normalized_residual: Any
    valid: Any


def feedback_state_mse(*, predicted, teacher, predicted_defined, teacher_defined,
                       eligible, scale) -> FeedbackStateLossResult:
    """Coordinate MSE after scaling by the frozen neural input standard deviation.

    No PFT cover weighting or coordinate pooling precedes the square. An eligible
    Teacher-defined coordinate with an invalid prediction/scale is explicitly
    counted as invalid; the training caller must reject it. Mask before arithmetic
    so undefined NaNs never contaminate valid gradients.
    """
    predicted = jnp.asarray(predicted, dtype=jnp.float64)
    teacher = jnp.asarray(teacher, dtype=jnp.float64)
    if predicted.shape != teacher.shape:
        raise ValueError('feedback predicted and Teacher shapes differ')
    shape = predicted.shape
    expected = (jnp.broadcast_to(jnp.asarray(eligible, dtype=bool), shape)
                & jnp.broadcast_to(jnp.asarray(teacher_defined, dtype=bool), shape))
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), shape)
    valid = (expected & jnp.broadcast_to(jnp.asarray(predicted_defined, dtype=bool), shape)
             & jnp.isfinite(predicted) & jnp.isfinite(teacher)
             & jnp.isfinite(scale) & (scale > 0.0))
    residual = jnp.where(valid, predicted, 0.0) - jnp.where(valid, teacher, 0.0)
    normalized = residual / jnp.where(valid, scale, 1.0)
    count = jnp.sum(valid)
    return FeedbackStateLossResult(
        loss=jnp.sum(normalized ** 2) / jnp.maximum(count, 1),
        count=count, expected_count=jnp.sum(expected),
        invalid_count=jnp.sum(expected & ~valid),
        predicted=predicted, teacher=teacher, physical_residual=residual,
        normalized_residual=normalized, valid=valid,
    )
