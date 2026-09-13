"""Source-direct canopy organic-carbon stock-change supervision.

The reduction follows ``stomate_soilcarbon.f90::soilcarbon_leak``:1245-1251
and 2326-2332, which adds ``interception_storage`` directly on the land-area
basis.  It is a shared loss geometry helper, not a replacement for the
source-backed TF-DOC transition at lines 1497-1525.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp


class CanopyChangeLossResult(NamedTuple):
    """Masked stock-change loss and source-direct diagnostic quantities."""

    loss: Any
    count: Any
    predicted_delta: Any
    teacher_delta: Any
    physical_residual: Any
    normalized_residual: Any
    row_valid: Any
    scale_valid: Any
    active_slot_count: Any
    bare_active_slot_count: Any
    bare_predicted_delta: Any
    bare_teacher_delta: Any
    bare_physical_residual: Any


def _as_stock(value: Any, *, name: str) -> Any:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim < 1:
        raise ValueError(f"{name} must end in a compact PFT axis")
    return array


def _broadcast_mask(value: Any, shape: tuple[int, ...], *, name: str) -> Any:
    mask = jnp.asarray(value, dtype=jnp.bool_)
    try:
        return jnp.broadcast_to(mask, shape)
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to the compact stock shape") from error


def canopy_carbon_stock_change_loss(
    *,
    start_pred: Any,
    next_pred: Any,
    start_teacher: Any,
    next_teacher: Any,
    start_pred_defined: Any,
    next_pred_defined: Any,
    start_teacher_defined: Any,
    next_teacher_defined: Any,
    active_pft_mask: Any,
    bare_pft_mask: Any,
    scale: Any,
    penalty: str = "mse",
) -> CanopyChangeLossResult:
    """Return source-direct stock-change MSE for compact ``[..., pft]`` vectors.

    ``scale`` is supplied by the frozen train/train state-delta statistics.
    All active slots, including bare soil, enter the stock consequence. Bare
    contributions are returned separately so an applicability issue is visible
    rather than hidden by a clamp. Invalid coordinates are replaced before any
    subtraction, preventing an undefined NaN from entering a JAX gradient.
    """

    if penalty not in ("mse", "pseudo_huber"):
        raise ValueError("unknown canopy stock-change penalty")
    start_pred = _as_stock(start_pred, name="start_pred")
    next_pred = _as_stock(next_pred, name="next_pred")
    start_teacher = _as_stock(start_teacher, name="start_teacher")
    next_teacher = _as_stock(next_teacher, name="next_teacher")
    shape = start_pred.shape
    if any(value.shape != shape for value in (next_pred, start_teacher, next_teacher)):
        raise ValueError("canopy stock vectors must share one compact shape")

    start_pred_defined = _broadcast_mask(start_pred_defined, shape, name="start_pred_defined")
    next_pred_defined = _broadcast_mask(next_pred_defined, shape, name="next_pred_defined")
    start_teacher_defined = _broadcast_mask(start_teacher_defined, shape, name="start_teacher_defined")
    next_teacher_defined = _broadcast_mask(next_teacher_defined, shape, name="next_teacher_defined")
    active = _broadcast_mask(active_pft_mask, shape, name="active_pft_mask")
    bare = _broadcast_mask(bare_pft_mask, shape, name="bare_pft_mask")
    scale = jnp.asarray(scale, dtype=jnp.float64)
    if scale.ndim != 0:
        raise ValueError("canopy stock-change scale must be scalar")

    coordinate_valid = (
        active
        & start_pred_defined
        & next_pred_defined
        & start_teacher_defined
        & next_teacher_defined
        & jnp.isfinite(start_pred)
        & jnp.isfinite(next_pred)
        & jnp.isfinite(start_teacher)
        & jnp.isfinite(next_teacher)
    )
    has_active = jnp.any(active, axis=-1)
    all_active_valid = jnp.all((~active) | coordinate_valid, axis=-1)
    row_valid = has_active & all_active_valid
    scale_valid = jnp.isfinite(scale) & (scale > 0.0)
    loss_valid = row_valid & scale_valid

    # Mask before subtraction: undefined NaNs must not enter primal or tangent arithmetic.
    safe_start_pred = jnp.where(coordinate_valid, start_pred, 0.0)
    safe_next_pred = jnp.where(coordinate_valid, next_pred, 0.0)
    safe_start_teacher = jnp.where(coordinate_valid, start_teacher, 0.0)
    safe_next_teacher = jnp.where(coordinate_valid, next_teacher, 0.0)
    predicted_by_slot = safe_next_pred - safe_start_pred
    teacher_by_slot = safe_next_teacher - safe_start_teacher

    predicted_delta = jnp.where(
        row_valid,
        jnp.sum(jnp.where(active, predicted_by_slot, 0.0), axis=-1),
        0.0,
    )
    teacher_delta = jnp.where(
        row_valid,
        jnp.sum(jnp.where(active, teacher_by_slot, 0.0), axis=-1),
        0.0,
    )
    physical_residual = jnp.where(row_valid, predicted_delta - teacher_delta, 0.0)
    safe_scale = jnp.where(scale_valid, scale, 1.0)
    normalized_residual = jnp.where(loss_valid, physical_residual / safe_scale, 0.0)
    count = jnp.sum(loss_valid, dtype=jnp.float64)
    squared = normalized_residual * normalized_residual
    # Algebraically sqrt(1+r^2)-1, with accurate small-residual values and a
    # bounded large-residual derivative. No clipping or stop_gradient.
    penalties = squared if penalty == "mse" else squared / (jnp.sqrt(1.0 + squared) + 1.0)
    loss = jnp.sum(penalties, dtype=jnp.float64) / jnp.maximum(count, 1.0)

    bare_active = active & bare
    bare_predicted_delta = jnp.where(
        row_valid,
        jnp.sum(jnp.where(bare_active, predicted_by_slot, 0.0), axis=-1),
        0.0,
    )
    bare_teacher_delta = jnp.where(
        row_valid,
        jnp.sum(jnp.where(bare_active, teacher_by_slot, 0.0), axis=-1),
        0.0,
    )
    return CanopyChangeLossResult(
        loss=loss,
        count=count,
        predicted_delta=predicted_delta,
        teacher_delta=teacher_delta,
        physical_residual=physical_residual,
        normalized_residual=normalized_residual,
        row_valid=row_valid,
        scale_valid=scale_valid,
        active_slot_count=jnp.sum(active, axis=-1, dtype=jnp.int32),
        bare_active_slot_count=jnp.sum(bare_active, axis=-1, dtype=jnp.int32),
        bare_predicted_delta=bare_predicted_delta,
        bare_teacher_delta=bare_teacher_delta,
        bare_physical_residual=jnp.where(row_valid, bare_predicted_delta - bare_teacher_delta, 0.0),
    )
