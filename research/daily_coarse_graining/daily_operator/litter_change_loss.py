"""Area-weighted daily litter-carbon stock-change supervision.

The stock extraction matches ``successor_training.py`` and the Fortran mass-
balance inventories in ``stomate_litter.f90::littercalc`` lines 1975-1996 and
2801-2820: select ``ICARBON``, multiply each active PFT by its land fraction,
and sum the litter-type and (for belowground litter) soil-layer axes.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_orchidee.stomate.carbon_kernels import ICARBON

from .long_horizon_objective import derive_area_weighted_pft_carbon_pool

LITTER_OWNER_KEYS = (
    "slowproc_stomate_previous_step_state.litter_above",
    "slowproc_stomate_previous_step_state.litter_below",
)
LITTER_OWNER_NAMES = ("aboveground", "belowground")


class LitterOwnerLayout(NamedTuple):
    """Static compact-state layout for one litter inventory owner."""

    name: str
    leaf: Any
    pft_axis: int
    element_axis: int


class LitterCarbonLayout(NamedTuple):
    """The two source litter owners and their common compact PFT layout."""

    owners: tuple[LitterOwnerLayout, LitterOwnerLayout]
    selected_pft_indices: tuple[int, ...]


class LitterChangeScales(NamedTuple):
    """Auditable train-state-delta scale propagation for both owners."""

    owner_names: tuple[str, str]
    raw_scale: np.ndarray
    effective_scale: np.ndarray
    coordinate_count: np.ndarray
    scale_source_pft_index: int
    scale_compact_pft_index: int
    minimum_scale_gC_m2_day: float
    scale_rule: str


class LitterCarbonPools(NamedTuple):
    """Land-area litter carbon with an explicit final owner axis."""

    value: Any
    defined: Any
    active_pft_count: Any


class LitterChangeLossResult(NamedTuple):
    """Two-owner loss plus physical and validity diagnostics."""

    loss: Any
    owner_loss: Any
    predicted_delta: Any
    teacher_delta: Any
    physical_residual: Any
    normalized_residual: Any
    valid: Any
    scale_valid: Any
    valid_count: Any
    expected_count: Any
    invalid_prediction_count: Any
    invalid_target_count: Any
    invalid_count: Any
    empty_count: Any


def resolve_litter_carbon_layout(contract: Any) -> LitterCarbonLayout:
    """Resolve and validate the two continuous litter state owners."""

    owners = []
    selected_pfts = None
    for name, key in zip(LITTER_OWNER_NAMES, LITTER_OWNER_KEYS, strict=True):
        matches = tuple(leaf for leaf in contract.state_leaves if leaf.key == key)
        if len(matches) != 1:
            raise ValueError(f"litter carbon state owner drift for {name}")
        leaf = matches[0]
        if leaf.start is None or leaf.stop is None:
            raise ValueError(f"litter carbon state owner {name} must be continuous")
        axes = tuple(str(axis) for axis in leaf.axis_names)
        if axes.count("nvm") != 1 or axes.count("nelements") != 1:
            raise ValueError(f"litter carbon state owner {name} requires PFT and element axes")
        pft_axis = axes.index("nvm")
        element_axis = axes.index("nelements")
        if int(leaf.shape[element_axis]) <= ICARBON:
            raise ValueError(f"litter carbon state owner {name} has no carbon coordinate")
        owner_pfts = tuple(int(index) for index in leaf.selected_pft_indices)
        if len(owner_pfts) != int(leaf.shape[pft_axis]):
            raise ValueError(f"litter carbon compact PFT layout drift for {name}")
        if selected_pfts is None:
            selected_pfts = owner_pfts
        elif owner_pfts != selected_pfts:
            raise ValueError("litter carbon owners have different compact PFT layouts")
        owners.append(LitterOwnerLayout(name, leaf, pft_axis, element_axis))
    return LitterCarbonLayout(tuple(owners), selected_pfts)


def derive_litter_change_scales(
    state_delta_variance: Any,
    contract: Any,
    *,
    scale_source_pft_index: int,
    minimum_scale_gC_m2_day: float,
) -> LitterChangeScales:
    """Propagate frozen TRAIN coordinate variances for one selected PFT.

    ``sqrt(sum(variance))`` is applied only to each owner's carbon coordinates
    at the declared source PFT. This is deterministic RSS propagation in the
    absence of stored covariance, not an exact standard deviation of the
    aggregate. The explicit physical floor handles genuinely near-zero TRAIN
    changes; filled generic scales and standing-stock levels never enter it.
    """

    minimum = float(minimum_scale_gC_m2_day)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum litter-change scale must be finite and positive")
    source_pft = int(scale_source_pft_index)
    layout = resolve_litter_carbon_layout(contract)
    try:
        compact_pft = layout.selected_pft_indices.index(source_pft)
    except ValueError as error:
        raise ValueError("litter scale source PFT is absent from the compact layout") from error
    state_variance = np.asarray(state_delta_variance, dtype=np.float64)
    if state_variance.shape != (contract.continuous_state_width,):
        raise ValueError("frozen state-delta variance width drift")

    raw = []
    counts = []
    for owner in layout.owners:
        leaf = owner.leaf
        field = state_variance[leaf.start : leaf.stop].reshape(leaf.shape)
        carbon = np.take(field, ICARBON, axis=owner.element_axis)
        pft_axis = owner.pft_axis - int(owner.element_axis < owner.pft_axis)
        selected = np.take(carbon, compact_pft, axis=pft_axis)
        if selected.size < 1 or not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
            raise ValueError(f"invalid frozen litter state-delta variance for {owner.name}")
        raw_scale = float(np.sqrt(np.sum(selected, dtype=np.float64)))
        if not np.isfinite(raw_scale):
            raise ValueError(f"invalid derived litter-change scale for {owner.name}")
        raw.append(raw_scale)
        counts.append(int(selected.size))
    raw_array = np.asarray(raw, dtype=np.float64)
    return LitterChangeScales(
        owner_names=LITTER_OWNER_NAMES,
        raw_scale=raw_array,
        effective_scale=np.maximum(raw_array, minimum),
        coordinate_count=np.asarray(counts, dtype=np.int32),
        scale_source_pft_index=source_pft,
        scale_compact_pft_index=compact_pft,
        minimum_scale_gC_m2_day=minimum,
        scale_rule="coordinate_variance_sum_sqrt",
    )


def extract_litter_carbon_pools(
    continuous_state: Any,
    continuous_defined: Any,
    *,
    pft_fraction: Any,
    active_pft_mask: Any,
    contract: Any,
    layout: LitterCarbonLayout | None = None,
) -> LitterCarbonPools:
    """Extract separate above/below litter pools for scalar or batched states."""

    layout = resolve_litter_carbon_layout(contract) if layout is None else layout
    state = jnp.asarray(continuous_state, dtype=jnp.float64)
    defined = jnp.asarray(continuous_defined, dtype=jnp.bool_)
    if state.ndim < 1 or state.shape[-1] != contract.continuous_state_width or defined.shape != state.shape:
        raise ValueError("litter extraction requires [..., continuous_state] values and masks")
    fraction = jnp.asarray(pft_fraction, dtype=jnp.float64)
    active = jnp.asarray(active_pft_mask, dtype=jnp.bool_)
    expected_fraction_shape = state.shape[:-1] + (len(layout.selected_pft_indices),)
    if fraction.shape != expected_fraction_shape or active.shape != fraction.shape:
        raise ValueError("litter extraction PFT context shape drift")
    fraction_finite = jnp.isfinite(fraction)
    safe_fraction = jnp.where(active & fraction_finite, fraction, 0.0)
    has_active = jnp.any(active, axis=-1)
    active_count = jnp.sum(active, axis=-1, dtype=jnp.int32)
    leading_ndim = state.ndim - 1

    values = []
    masks = []
    for owner in layout.owners:
        leaf = owner.leaf
        shape = state.shape[:-1] + tuple(int(size) for size in leaf.shape)
        field = state[..., leaf.start : leaf.stop].reshape(shape)
        field_defined = defined[..., leaf.start : leaf.stop].reshape(shape)
        coordinate_valid = field_defined & jnp.isfinite(field)
        safe_field = jnp.where(coordinate_valid, field, 0.0)
        pft_axis = leading_ndim + owner.pft_axis
        element_axis = leading_ndim + owner.element_axis
        values.append(
            derive_area_weighted_pft_carbon_pool(
                safe_field,
                safe_fraction,
                active,
                pft_axis=pft_axis,
                element_axis=element_axis,
            )
        )
        carbon_valid = jnp.take(coordinate_valid, ICARBON, axis=element_axis)
        carbon_pft_axis = pft_axis - int(element_axis < pft_axis)
        relevant_shape = [1] * carbon_valid.ndim
        relevant_shape[:leading_ndim] = state.shape[:-1]
        relevant_shape[carbon_pft_axis] = active.shape[-1]
        relevant = jnp.broadcast_to(jnp.reshape(active, tuple(relevant_shape)), carbon_valid.shape)
        all_coordinates_valid = jnp.all(
            (~relevant) | carbon_valid,
            axis=tuple(range(leading_ndim, carbon_valid.ndim)),
        )
        all_fractions_valid = jnp.all((~active) | fraction_finite, axis=-1)
        masks.append(has_active & all_fractions_valid & all_coordinates_valid)
    owner_count = len(layout.owners)
    return LitterCarbonPools(
        value=jnp.stack(values, axis=-1),
        defined=jnp.stack(masks, axis=-1),
        active_pft_count=jnp.broadcast_to(active_count[..., None], active_count.shape + (owner_count,)),
    )


def litter_carbon_stock_change_loss(
    *,
    initial_teacher: LitterCarbonPools,
    next_predicted: LitterCarbonPools,
    next_teacher: LitterCarbonPools,
    scale: Any,
    penalty: str = "pseudo_huber",
) -> LitterChangeLossResult:
    """Compare daily change for above/below owners from one shared Teacher start."""

    if penalty != "pseudo_huber":
        raise ValueError("litter stock-change penalty must be pseudo_huber")
    shape = jnp.asarray(initial_teacher.value).shape
    if not shape or shape[-1] != len(LITTER_OWNER_NAMES):
        raise ValueError("litter pools must end in the two-owner axis")
    pools = (initial_teacher, next_predicted, next_teacher)
    for pool in pools:
        if (
            jnp.asarray(pool.value).shape != shape
            or jnp.asarray(pool.defined).shape != shape
            or jnp.asarray(pool.active_pft_count).shape != shape
        ):
            raise ValueError("litter pool shapes differ")
    scales = jnp.asarray(scale, dtype=jnp.float64)
    if scales.shape != (len(LITTER_OWNER_NAMES),):
        raise ValueError("litter stock-change scale must have one value per owner")

    initial_value = jnp.asarray(initial_teacher.value, dtype=jnp.float64)
    predicted_value = jnp.asarray(next_predicted.value, dtype=jnp.float64)
    teacher_value = jnp.asarray(next_teacher.value, dtype=jnp.float64)
    expected = jnp.asarray(next_teacher.active_pft_count) > 0
    initial_valid = jnp.asarray(initial_teacher.defined, dtype=jnp.bool_) & jnp.isfinite(initial_value)
    predicted_valid = jnp.asarray(next_predicted.defined, dtype=jnp.bool_) & jnp.isfinite(predicted_value)
    teacher_valid = jnp.asarray(next_teacher.defined, dtype=jnp.bool_) & jnp.isfinite(teacher_value)
    target_valid = initial_valid & teacher_valid
    scale_valid = jnp.isfinite(scales) & (scales > 0.0)
    valid = expected & target_valid & predicted_valid & scale_valid

    # Sanitize each endpoint before subtraction, including the shared Teacher
    # initial state, so undefined NaNs have no primal or tangent path.
    safe_initial = jnp.where(initial_valid, initial_value, 0.0)
    safe_predicted = jnp.where(predicted_valid, predicted_value, 0.0)
    safe_teacher = jnp.where(teacher_valid, teacher_value, 0.0)
    predicted_delta = jnp.where(valid, safe_predicted - safe_initial, 0.0)
    teacher_delta = jnp.where(valid, safe_teacher - safe_initial, 0.0)
    physical_residual = jnp.where(valid, predicted_delta - teacher_delta, 0.0)
    broadcast_scale = jnp.broadcast_to(scales, shape)
    normalized_residual = jnp.where(valid, physical_residual / jnp.where(valid, broadcast_scale, 1.0), 0.0)
    squared = normalized_residual * normalized_residual
    penalties = squared / (jnp.sqrt(1.0 + squared) + 1.0)
    reduction_axes = tuple(range(len(shape) - 1))
    valid_count = jnp.sum(valid, axis=reduction_axes, dtype=jnp.float64)
    owner_loss = jnp.sum(penalties, axis=reduction_axes, dtype=jnp.float64) / jnp.maximum(valid_count, 1.0)
    expected_count = jnp.sum(expected, axis=reduction_axes, dtype=jnp.float64)
    invalid_target_count = jnp.sum(expected & ~target_valid, axis=reduction_axes, dtype=jnp.int32)
    invalid_prediction_count = jnp.sum(expected & ~predicted_valid, axis=reduction_axes, dtype=jnp.int32)
    invalid_scale_count = jnp.sum(~scale_valid, dtype=jnp.int32)
    invalid_count = (
        jnp.sum(invalid_target_count, dtype=jnp.int32)
        + jnp.sum(invalid_prediction_count, dtype=jnp.int32)
        + invalid_scale_count
    )
    empty_count = jnp.sum(expected_count == 0, dtype=jnp.int32)
    return LitterChangeLossResult(
        loss=jnp.mean(owner_loss),
        owner_loss=owner_loss,
        predicted_delta=predicted_delta,
        teacher_delta=teacher_delta,
        physical_residual=physical_residual,
        normalized_residual=normalized_residual,
        valid=valid,
        scale_valid=scale_valid,
        valid_count=valid_count,
        expected_count=expected_count,
        invalid_prediction_count=invalid_prediction_count,
        invalid_target_count=invalid_target_count,
        invalid_count=invalid_count,
        empty_count=empty_count,
    )
