"""Architecture-independent biomass/productivity rollout objectives."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from jax_orchidee.stomate.carbon_kernels import (
    IAGRHRTPN,
    IAGRHRTST,
    IAGRSAPPN,
    IAGRSAPST,
    ICARBON,
    ICARBRES,
    IFRUIT,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    NPARTS,
)

DEFAULT_HORIZONS = (1, 7, 30)
ECOSYSTEM_CARBON_COMPONENTS = (
    "live_biomass",
    "litter",
    "soil_particulate",
    "dissolved_organic",
    "canopy_interception_organic",
)

# This is the exact above/below routing used by littercalc_litter_increments.
# Reserve carbon follows the source's above-litter route, so the two groups
# partition all plant parts without inventing an unowned carbon pool.
ABOVEGROUND_BIOMASS_PARTS = (
    ILEAF,
    ISAPABOVE,
    IHEARTABOVE,
    IFRUIT,
    ICARBRES,
    IAGRSAPST,
    IAGRSAPPN,
    IAGRHRTST,
    IAGRHRTPN,
)
BELOWGROUND_BIOMASS_PARTS = (ISAPBELOW, IHEARTBELOW, IROOT)


class MaskedTrajectory(NamedTuple):
    """Time-leading values and their source-defined status."""

    value: Any
    defined: Any


class BiomassCarbonStocks(NamedTuple):
    """Land-area-weighted plant carbon stocks in gC m-2."""

    aboveground: Any
    belowground: Any
    reserve: Any
    total: Any


class NormalizedError(NamedTuple):
    mse: Any
    mean_bias: Any
    bias_squared: Any
    count: Any


class HorizonObjective(NamedTuple):
    cumulative_flux: Mapping[str, NormalizedError]
    terminal_stock: Mapping[str, NormalizedError]
    cumulative_flux_macro_mse: Any
    cumulative_flux_macro_bias_squared: Any
    terminal_stock_macro_mse: Any
    terminal_stock_macro_bias_squared: Any


class LongHorizonObjectiveResult(NamedTuple):
    total: Any
    cumulative_flux_mse: Any
    cumulative_flux_bias_squared: Any
    terminal_stock_mse: Any
    terminal_stock_bias_squared: Any
    by_horizon: Mapping[int, HorizonObjective]


class RolloutHardGateCounts(NamedTuple):
    nonfinite_defined_prediction: Any
    negative_declared_stock: Any
    defined_mask_mismatch: Any
    discrete_state_mismatch: Any
    budget_residual_failure: Any


def derive_area_weighted_biomass_carbon(
    biomass: Any,
    pft_fraction: Any,
    active_pft_mask: Any,
) -> BiomassCarbonStocks:
    """Derive source-partitioned AGB/BGB from the full biomass state.

    ``biomass`` must end in ``[pft, plant_part, element]``. ORCHIDEE stores
    biomass per PFT-covered area, so the result is weighted by the supplied
    source PFT fraction before the PFT axis is reduced.
    """

    biomass = jnp.asarray(biomass, dtype=jnp.float64)
    fraction = jnp.asarray(pft_fraction, dtype=jnp.float64)
    active = jnp.asarray(active_pft_mask, dtype=jnp.bool_)
    if biomass.ndim < 3 or biomass.shape[-2:] != (NPARTS, ICARBON + 1):
        raise ValueError(
            f"biomass must end in [pft, {NPARTS}, {ICARBON + 1}]"
        )
    if fraction.shape != biomass.shape[:-2] or active.shape != fraction.shape:
        raise ValueError("PFT fraction/mask must match the biomass PFT prefix")
    if biomass.shape[-3] != fraction.shape[-1]:
        raise ValueError("biomass and PFT fraction axes differ")

    carbon = biomass[..., ICARBON]
    weight = jnp.where(active, fraction, 0.0)
    above_by_pft = jnp.sum(
        jnp.take(carbon, jnp.asarray(ABOVEGROUND_BIOMASS_PARTS), axis=-1),
        axis=-1,
    )
    below_by_pft = jnp.sum(
        jnp.take(carbon, jnp.asarray(BELOWGROUND_BIOMASS_PARTS), axis=-1),
        axis=-1,
    )
    reserve_by_pft = carbon[..., ICARBRES]
    total_by_pft = jnp.sum(carbon, axis=-1)
    return BiomassCarbonStocks(
        aboveground=jnp.sum(above_by_pft * weight, axis=-1),
        belowground=jnp.sum(below_by_pft * weight, axis=-1),
        reserve=jnp.sum(reserve_by_pft * weight, axis=-1),
        total=jnp.sum(total_by_pft * weight, axis=-1),
    )


def derive_area_weighted_pft_carbon_pool(
    stock: Any,
    pft_fraction: Any,
    active_pft_mask: Any,
    *,
    pft_axis: int,
    element_axis: int | None = None,
) -> Any:
    """Reduce one source PFT-owned carbon pool to land-area units.

    Leading dimensions are inferred from ``pft_fraction[..., pft]``. All
    remaining scientific pool/layer axes are summed exactly once. When an
    element axis is supplied, only the source carbon element is selected.
    """

    stock = jnp.asarray(stock, dtype=jnp.float64)
    fraction = jnp.asarray(pft_fraction, dtype=jnp.float64)
    active = jnp.asarray(active_pft_mask, dtype=jnp.bool_)
    if fraction.ndim < 1 or active.shape != fraction.shape:
        raise ValueError("PFT fraction and active mask must share [..., pft]")
    leading_ndim = fraction.ndim - 1
    if stock.ndim <= leading_ndim:
        raise ValueError("carbon stock has no scientific axes")

    normalized_pft_axis = pft_axis % stock.ndim
    if element_axis is not None:
        normalized_element_axis = element_axis % stock.ndim
        if normalized_element_axis == normalized_pft_axis:
            raise ValueError("PFT and element axes must differ")
        if stock.shape[normalized_element_axis] <= ICARBON:
            raise ValueError("carbon element is absent")
        stock = jnp.take(stock, ICARBON, axis=normalized_element_axis)
        if normalized_element_axis < normalized_pft_axis:
            normalized_pft_axis -= 1

    if stock.shape[:leading_ndim] != fraction.shape[:-1]:
        raise ValueError("carbon stock leading dimensions differ from PFT fractions")
    if stock.shape[normalized_pft_axis] != fraction.shape[-1]:
        raise ValueError("carbon stock and fraction PFT widths differ")
    if normalized_pft_axis < leading_ndim:
        raise ValueError("PFT axis overlaps leading sample dimensions")

    weight_shape = [1] * stock.ndim
    weight_shape[:leading_ndim] = fraction.shape[:-1]
    weight_shape[normalized_pft_axis] = fraction.shape[-1]
    weight = jnp.reshape(jnp.where(active, fraction, 0.0), tuple(weight_shape))
    reduction_axes = tuple(range(leading_ndim, stock.ndim))
    return jnp.sum(stock * weight, axis=reduction_axes)


def derive_ecosystem_carbon(
    components: Mapping[str, Any],
) -> Any:
    """Sum independent ecosystem carbon inventories without double counting."""

    expected = set(ECOSYSTEM_CARBON_COMPONENTS)
    if set(components) != expected:
        missing = sorted(expected - set(components))
        extra = sorted(set(components) - expected)
        raise ValueError(
            f"ecosystem carbon component drift; missing={missing}, extra={extra}"
        )
    arrays = {
        name: jnp.asarray(components[name], dtype=jnp.float64)
        for name in ECOSYSTEM_CARBON_COMPONENTS
    }
    shape = arrays[ECOSYSTEM_CARBON_COMPONENTS[0]].shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("ecosystem carbon component shapes differ")
    return sum(arrays.values(), jnp.zeros(shape, dtype=jnp.float64))


def _validate_trajectory_registry(
    predicted: Mapping[str, MaskedTrajectory],
    expected: Mapping[str, MaskedTrajectory],
    scales: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    names = set(expected)
    for label, registry in (("prediction", predicted), ("scale", scales)):
        if set(registry) != names:
            missing = sorted(names - set(registry))
            extra = sorted(set(registry) - names)
            raise ValueError(
                f"{kind} {label} registry drift; missing={missing}, extra={extra}"
            )
    if not names:
        raise ValueError(f"{kind} registry is empty")
    for name in expected:
        predicted_value = jnp.asarray(predicted[name].value)
        expected_value = jnp.asarray(expected[name].value)
        predicted_defined = jnp.asarray(predicted[name].defined)
        expected_defined = jnp.asarray(expected[name].defined)
        if predicted_value.shape != expected_value.shape:
            raise ValueError(f"{kind} value shape drift for {name}")
        if (
            predicted_defined.shape != predicted_value.shape
            or expected_defined.shape != expected_value.shape
        ):
            raise ValueError(f"{kind} mask shape drift for {name}")
        if predicted_value.ndim < 1:
            raise ValueError(f"{kind} trajectory {name} has no time axis")
        scale = jnp.asarray(scales[name], dtype=jnp.float64)
        try:
            jnp.broadcast_to(scale, predicted_value.shape[1:])
        except ValueError as error:
            raise ValueError(f"{kind} scale shape drift for {name}") from error


def _normalized_error(
    predicted: Any,
    expected: Any,
    defined: Any,
    scale: Any,
) -> NormalizedError:
    predicted = jnp.asarray(predicted, dtype=jnp.float64)
    expected = jnp.asarray(expected, dtype=jnp.float64)
    defined = jnp.asarray(defined, dtype=jnp.bool_)
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), predicted.shape)
    valid = (
        defined
        & jnp.isfinite(predicted)
        & jnp.isfinite(expected)
        & jnp.isfinite(scale)
        & (scale > 0.0)
    )
    residual = jnp.where(valid, (predicted - expected) / scale, 0.0)
    count = jnp.sum(valid, dtype=jnp.float64)
    mean = jnp.where(count > 0.0, jnp.sum(residual) / count, 0.0)
    mse = jnp.where(
        count > 0.0,
        jnp.sum(residual * residual, dtype=jnp.float64) / count,
        0.0,
    )
    return NormalizedError(mse, mean, mean * mean, count)


def _macro(values: Sequence[NormalizedError], attribute: str) -> Any:
    active = jnp.stack(tuple(value.count > 0.0 for value in values))
    selected = jnp.stack(tuple(getattr(value, attribute) for value in values))
    count = jnp.sum(active, dtype=jnp.float64)
    return jnp.where(
        count > 0.0,
        jnp.sum(jnp.where(active, selected, 0.0), dtype=jnp.float64) / count,
        0.0,
    )


def _horizon_objective(
    predicted_fluxes: Mapping[str, MaskedTrajectory],
    expected_fluxes: Mapping[str, MaskedTrajectory],
    predicted_stocks: Mapping[str, MaskedTrajectory],
    expected_stocks: Mapping[str, MaskedTrajectory],
    *,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    horizon: int,
) -> HorizonObjective:
    flux_errors: dict[str, NormalizedError] = {}
    for name in expected_fluxes:
        predicted = predicted_fluxes[name]
        expected = expected_fluxes[name]
        daily_defined = (
            jnp.asarray(predicted.defined[:horizon], dtype=jnp.bool_)
            & jnp.asarray(expected.defined[:horizon], dtype=jnp.bool_)
        )
        cumulative_defined = jnp.all(daily_defined, axis=0)
        predicted_sum = jnp.sum(
            jnp.where(daily_defined, predicted.value[:horizon], 0.0),
            axis=0,
        )
        expected_sum = jnp.sum(
            jnp.where(daily_defined, expected.value[:horizon], 0.0),
            axis=0,
        )
        flux_errors[name] = _normalized_error(
            predicted_sum,
            expected_sum,
            cumulative_defined,
            horizon * jnp.asarray(flux_scales[name], dtype=jnp.float64),
        )

    stock_errors: dict[str, NormalizedError] = {}
    for name in expected_stocks:
        predicted = predicted_stocks[name]
        expected = expected_stocks[name]
        defined = (
            jnp.asarray(predicted.defined[horizon - 1], dtype=jnp.bool_)
            & jnp.asarray(expected.defined[horizon - 1], dtype=jnp.bool_)
        )
        stock_errors[name] = _normalized_error(
            predicted.value[horizon - 1],
            expected.value[horizon - 1],
            defined,
            stock_scales[name],
        )

    flux_values = tuple(flux_errors.values())
    stock_values = tuple(stock_errors.values())
    return HorizonObjective(
        cumulative_flux=flux_errors,
        terminal_stock=stock_errors,
        cumulative_flux_macro_mse=_macro(flux_values, "mse"),
        cumulative_flux_macro_bias_squared=_macro(flux_values, "bias_squared"),
        terminal_stock_macro_mse=_macro(stock_values, "mse"),
        terminal_stock_macro_bias_squared=_macro(stock_values, "bias_squared"),
    )


def long_horizon_biomass_productivity_objective(
    predicted_fluxes: Mapping[str, MaskedTrajectory],
    expected_fluxes: Mapping[str, MaskedTrajectory],
    predicted_stocks: Mapping[str, MaskedTrajectory],
    expected_stocks: Mapping[str, MaskedTrajectory],
    *,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    coefficients: Mapping[str, Any],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> LongHorizonObjectiveResult:
    """Compute equal-field, equal-horizon cumulative and terminal losses.

    Coefficients are deliberately explicit. This module freezes the objective
    components, masks and normalization, but does not silently choose their
    weight relative to the existing one-day process-supervision anchor.
    """

    _validate_trajectory_registry(
        predicted_fluxes,
        expected_fluxes,
        flux_scales,
        kind="daily flux",
    )
    _validate_trajectory_registry(
        predicted_stocks,
        expected_stocks,
        stock_scales,
        kind="stock",
    )
    horizon_tuple = tuple(int(value) for value in horizons)
    if not horizon_tuple or any(value <= 0 for value in horizon_tuple):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizon_tuple)) != len(horizon_tuple):
        raise ValueError("horizons must be unique")
    available = min(
        min(jnp.asarray(value.value).shape[0] for value in expected_fluxes.values()),
        min(jnp.asarray(value.value).shape[0] for value in expected_stocks.values()),
    )
    if max(horizon_tuple) > available:
        raise ValueError("requested horizon exceeds the available trajectory")
    component_names = {
        "cumulative_flux_mse",
        "cumulative_flux_bias_squared",
        "terminal_stock_mse",
        "terminal_stock_bias_squared",
    }
    if set(coefficients) != component_names:
        raise ValueError("long-horizon coefficient registry drift")

    by_horizon = {
        horizon: _horizon_objective(
            predicted_fluxes,
            expected_fluxes,
            predicted_stocks,
            expected_stocks,
            flux_scales=flux_scales,
            stock_scales=stock_scales,
            horizon=horizon,
        )
        for horizon in horizon_tuple
    }
    denominator = jnp.asarray(len(by_horizon), dtype=jnp.float64)
    components = {
        "cumulative_flux_mse": sum(
            value.cumulative_flux_macro_mse for value in by_horizon.values()
        )
        / denominator,
        "cumulative_flux_bias_squared": sum(
            value.cumulative_flux_macro_bias_squared
            for value in by_horizon.values()
        )
        / denominator,
        "terminal_stock_mse": sum(
            value.terminal_stock_macro_mse for value in by_horizon.values()
        )
        / denominator,
        "terminal_stock_bias_squared": sum(
            value.terminal_stock_macro_bias_squared
            for value in by_horizon.values()
        )
        / denominator,
    }
    total = sum(
        jnp.asarray(coefficients[name], dtype=jnp.float64) * value
        for name, value in components.items()
    )
    return LongHorizonObjectiveResult(
        total=total,
        cumulative_flux_mse=components["cumulative_flux_mse"],
        cumulative_flux_bias_squared=components[
            "cumulative_flux_bias_squared"
        ],
        terminal_stock_mse=components["terminal_stock_mse"],
        terminal_stock_bias_squared=components["terminal_stock_bias_squared"],
        by_horizon=by_horizon,
    )


def rollout_hard_gate_counts(
    predicted_stocks: Mapping[str, MaskedTrajectory],
    expected_stocks: Mapping[str, MaskedTrajectory],
    *,
    nonnegative_stock_fields: Sequence[str],
    budget_residuals: Mapping[str, Any],
    predicted_discrete_state: Mapping[str, Any],
    expected_discrete_state: Mapping[str, Any],
    budget_atol: float = 2.0e-12,
    stock_atol: float = 2.0e-12,
) -> RolloutHardGateCounts:
    """Count non-negotiable failures separately from scientific accuracy."""

    if set(predicted_stocks) != set(expected_stocks):
        raise ValueError("hard-gate stock registry drift")
    if not set(nonnegative_stock_fields) <= set(expected_stocks):
        raise ValueError("unknown nonnegative stock field")
    if set(predicted_discrete_state) != set(expected_discrete_state):
        raise ValueError("hard-gate discrete-state registry drift")

    nonfinite = jnp.asarray(0, dtype=jnp.int32)
    negative = jnp.asarray(0, dtype=jnp.int32)
    mask_mismatch = jnp.asarray(0, dtype=jnp.int32)
    nonnegative = set(nonnegative_stock_fields)
    for name in expected_stocks:
        predicted = jnp.asarray(predicted_stocks[name].value)
        expected = jnp.asarray(expected_stocks[name].value)
        predicted_defined = jnp.asarray(predicted_stocks[name].defined, dtype=jnp.bool_)
        expected_defined = jnp.asarray(expected_stocks[name].defined, dtype=jnp.bool_)
        if (
            predicted.shape != expected.shape
            or predicted_defined.shape != predicted.shape
            or expected_defined.shape != expected.shape
        ):
            raise ValueError(f"hard-gate stock shape drift for {name}")
        nonfinite = nonfinite + jnp.sum(
            expected_defined & ~jnp.isfinite(predicted), dtype=jnp.int32
        )
        mask_mismatch = mask_mismatch + jnp.sum(
            predicted_defined != expected_defined, dtype=jnp.int32
        )
        if name in nonnegative:
            negative = negative + jnp.sum(
                predicted_defined & jnp.isfinite(predicted) & (predicted < -stock_atol),
                dtype=jnp.int32,
            )

    predicted_leaves, predicted_tree = jax.tree_util.tree_flatten(
        predicted_discrete_state
    )
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(
        expected_discrete_state
    )
    if predicted_tree != expected_tree:
        raise ValueError("hard-gate discrete-state trees differ")
    discrete = sum(
        (
            jnp.sum(jnp.asarray(left) != jnp.asarray(right), dtype=jnp.int32)
            for left, right in zip(predicted_leaves, expected_leaves, strict=True)
        ),
        jnp.asarray(0, dtype=jnp.int32),
    )
    budget = sum(
        (
            jnp.sum(
                ~jnp.isfinite(jnp.asarray(value))
                | (jnp.abs(jnp.asarray(value)) > budget_atol),
                dtype=jnp.int32,
            )
            for value in budget_residuals.values()
        ),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return RolloutHardGateCounts(
        nonfinite_defined_prediction=nonfinite,
        negative_declared_stock=negative,
        defined_mask_mismatch=mask_mismatch,
        discrete_state_mismatch=discrete,
        budget_residual_failure=budget,
    )
