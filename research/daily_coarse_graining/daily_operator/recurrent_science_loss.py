"""Shared physically scaled recurrent science objective.

The factory returns a scalar loss with the public ``RolloutLoss`` signature:
``(params, initial_continuous, observations, targets, loss_context)``. Its
``.terms`` attribute exposes structured diagnostics for reporting. It is only
a scalar construction layer; callers remain responsible for selecting TRAIN
windows, physical gates, and native-AD/update admission.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, NamedTuple

import jax.numpy as jnp

from .long_horizon_objective import MaskedTrajectory, derive_area_weighted_pft_carbon_pool
from .successor_training import (
    _normalized_pft_context,
    _reduced_stock_defined,
    _trajectory_field,
    extract_biomass_productivity_trajectories,
)
from .target_oriented_training import _acceptance_trajectories

FLUX_KEYS = (
    "gpp",
    "npp",
    "autotrophic_respiration",
    "heterotrophic_respiration",
)
STOCK_KEYS = (
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "canopy_interception_organic_carbon",
)
WEIGHT_COMPONENTS = (
    "daily_flux",
    "cumulative_flux",
    "stock_endpoint",
    "stock_trajectory",
)
CANOPY_KEY = "canopy_interception_organic_carbon"


class ScienceTrajectories(NamedTuple):
    """Shared extracted trajectories plus the named source-direct canopy stock."""

    fluxes: Mapping[str, MaskedTrajectory]
    stocks: Mapping[str, MaskedTrajectory]


class TermDiagnostics(NamedTuple):
    """Masked scalar loss and validity accounting for one named term."""

    loss: Any
    count: Any
    expected_count: Any
    invalid_prediction_count: Any
    invalid_target_count: Any
    normalized_residual: Any


class RecurrentScienceLossResult(NamedTuple):
    """Loss components retained by the caller for physical/error reporting."""

    total: Any
    weighted_components: Mapping[str, Any]
    daily_flux: Mapping[str, TermDiagnostics]
    cumulative_flux: Mapping[str, TermDiagnostics]
    stock_endpoint: Mapping[str, TermDiagnostics]
    stock_trajectory: Mapping[str, TermDiagnostics]


def _positive_mapping(values: Mapping[str, float], keys: tuple[str, ...], *, name: str) -> dict[str, float]:
    if set(values) != set(keys):
        raise ValueError(f"{name} keys must be exactly {keys}")
    result = {}
    for key in keys:
        value = float(values[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name}.{key} must be finite and positive")
        result[key] = value
    return result


def _weights(weights: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if set(weights) != set(WEIGHT_COMPONENTS):
        raise ValueError(f"weights must contain exactly {WEIGHT_COMPONENTS}")
    result = {}
    for component in WEIGHT_COMPONENTS:
        keys = FLUX_KEYS if "flux" in component else STOCK_KEYS
        values = weights[component]
        if set(values) != set(keys):
            raise ValueError(f"weights.{component} keys must be exactly {keys}")
        result[component] = {}
        for key in keys:
            value = float(values[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"weights.{component}.{key} must be finite and nonnegative")
            result[component][key] = value
    return result


def _trajectory(value: Any, defined: Any, *, name: str) -> MaskedTrajectory:
    value = jnp.asarray(value, dtype=jnp.float64)
    defined = jnp.asarray(defined, dtype=jnp.bool_)
    if value.ndim < 1 or value.shape != defined.shape:
        raise ValueError(f"{name} must provide value/defined arrays with a leading day axis")
    return MaskedTrajectory(value=value, defined=defined)


def _canopy_interception_trajectory(observation: Mapping[str, Any], contract: Any) -> MaskedTrajectory:
    """Project the source-direct TF-DOC canopy carbon field from shared state layout.

    Fortran provenance: ``stomate_soilcarbon.f90::soilcarbon_leak`` lines
    1225, 1245-1251, 1497-1525, and 2326-2332.  Storage is already on the
    land-area basis, so PFT fractions are intentionally replaced by ones.
    """

    state = jnp.asarray(observation["state"], dtype=jnp.float64)
    defined = jnp.asarray(observation["defined"], dtype=jnp.bool_)
    if state.ndim != 2 or defined.shape != state.shape:
        raise ValueError("canopy projection requires [day, state] values and masks")
    pft_width = len(contract.active_pft_indices)
    fraction, active = _normalized_pft_context(
        observation["fraction"], observation["active"], days=state.shape[0], pft_width=pft_width
    )
    interception, interception_defined = _trajectory_field(state, defined, contract, "interception_storage")
    return MaskedTrajectory(
        value=derive_area_weighted_pft_carbon_pool(
            interception, jnp.ones_like(fraction), active, pft_axis=2, element_axis=-1
        ),
        defined=_reduced_stock_defined(interception_defined, active, pft_axis=2),
    )


def shared_science_trajectories(observation: Mapping[str, Any], contract: Any) -> ScienceTrajectories:
    """Use the reporting extractor and expose its source-direct canopy component."""

    extracted = extract_biomass_productivity_trajectories(
        observation["state"],
        observation["defined"],
        daily_fast_gpp=observation["gpp"],
        daily_fast_gpp_defined=observation["gpp_defined"],
        pft_fraction=observation["fraction"],
        active_pft_mask=observation["active"],
        contract=contract,
    )
    return ScienceTrajectories(
        fluxes=extracted.fluxes,
        stocks={**extracted.stocks, CANOPY_KEY: _canopy_interception_trajectory(observation, contract)},
    )


def _masked_term(
    predicted: MaskedTrajectory, target: MaskedTrajectory, *, scale: Any, pseudo_huber: bool
) -> TermDiagnostics:
    predicted = _trajectory(predicted.value, predicted.defined, name="predicted trajectory")
    target = _trajectory(target.value, target.defined, name="target trajectory")
    if predicted.value.shape != target.value.shape:
        raise ValueError("predicted and target trajectory shapes differ")
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), predicted.value.shape)
    target_valid = target.defined & jnp.isfinite(target.value)
    prediction_valid = predicted.defined & jnp.isfinite(predicted.value)
    scale_valid = jnp.isfinite(scale) & (scale > 0.0)
    valid = target_valid & prediction_valid & scale_valid
    # Values are masked before subtraction so undefined NaNs have no primal or AD path.
    residual = (jnp.where(valid, predicted.value, 0.0) - jnp.where(valid, target.value, 0.0)) / jnp.where(
        valid, scale, 1.0
    )
    squared = residual * residual
    penalty = squared / (jnp.sqrt(1.0 + squared) + 1.0) if pseudo_huber else squared
    count = jnp.sum(valid, dtype=jnp.float64)
    expected = target.defined
    return TermDiagnostics(
        loss=jnp.sum(penalty, dtype=jnp.float64) / jnp.maximum(count, 1.0),
        count=count,
        expected_count=jnp.sum(expected, dtype=jnp.float64),
        invalid_prediction_count=jnp.sum(expected & ~prediction_valid, dtype=jnp.float64),
        invalid_target_count=jnp.sum(target.defined & ~jnp.isfinite(target.value), dtype=jnp.float64),
        normalized_residual=residual,
    )


def _cumulative_term(predicted: MaskedTrajectory, target: MaskedTrajectory, *, scale: Any) -> TermDiagnostics:
    predicted = _trajectory(predicted.value, predicted.defined, name="predicted trajectory")
    target = _trajectory(target.value, target.defined, name="target trajectory")
    if predicted.value.shape != target.value.shape:
        raise ValueError("predicted and target trajectory shapes differ")
    if predicted.value.shape[0] < 1:
        raise ValueError("cumulative trajectory cannot be empty")
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), predicted.value.shape[1:])
    target_valid = target.defined & jnp.isfinite(target.value)
    prediction_valid = predicted.defined & jnp.isfinite(predicted.value)
    target_row = jnp.all(target_valid, axis=0)
    prediction_row = jnp.all(prediction_valid, axis=0)
    scale_valid = jnp.isfinite(scale) & (scale > 0.0)
    valid = target_row & prediction_row & scale_valid
    safe_predicted = jnp.where(prediction_valid, predicted.value, 0.0)
    safe_target = jnp.where(target_valid, target.value, 0.0)
    residual = jnp.where(valid, jnp.sum(safe_predicted, axis=0) - jnp.sum(safe_target, axis=0), 0.0) / jnp.where(
        valid, scale * predicted.value.shape[0], 1.0
    )
    count = jnp.sum(valid, dtype=jnp.float64)
    expected = target_row
    return TermDiagnostics(
        loss=jnp.sum(residual * residual, dtype=jnp.float64) / jnp.maximum(count, 1.0),
        count=count,
        expected_count=jnp.sum(expected, dtype=jnp.float64),
        invalid_prediction_count=jnp.sum(expected & ~prediction_row, dtype=jnp.float64),
        invalid_target_count=jnp.sum(target.defined & ~jnp.isfinite(target.value), dtype=jnp.float64),
        normalized_residual=residual,
    )


def _stock_endpoint_term(
    predicted: MaskedTrajectory, target: MaskedTrajectory, *, daily_scale: Any, pseudo_huber: bool
) -> TermDiagnostics:
    if predicted.value.shape[0] < 1 or target.value.shape[0] < 1:
        raise ValueError("stock endpoint trajectory cannot be empty")
    days = predicted.value.shape[0]
    return _masked_term(
        # Retain a day axis so a scalar-per-day stock trajectory remains valid.
        MaskedTrajectory(predicted.value[-1:], predicted.defined[-1:]),
        MaskedTrajectory(target.value[-1:], target.defined[-1:]),
        scale=jnp.asarray(daily_scale) * days,
        pseudo_huber=pseudo_huber,
    )


def _stock_trajectory_term(
    predicted: MaskedTrajectory, target: MaskedTrajectory, *, daily_scale: Any, pseudo_huber: bool
) -> TermDiagnostics:
    elapsed = jnp.arange(1, predicted.value.shape[0] + 1, dtype=jnp.float64)
    shape = (predicted.value.shape[0],) + (1,) * (predicted.value.ndim - 1)
    return _masked_term(predicted, target, scale=jnp.reshape(elapsed, shape) * daily_scale, pseudo_huber=pseudo_huber)


def _accepted_science(
    extracted: ScienceTrajectories,
) -> tuple[Mapping[str, MaskedTrajectory], Mapping[str, MaskedTrajectory]]:
    fluxes, stocks = _acceptance_trajectories(extracted.fluxes, extracted.stocks)
    if CANOPY_KEY not in extracted.stocks:
        raise ValueError(f"trajectory extractor must expose {CANOPY_KEY}")
    return (
        {key: fluxes[key] for key in FLUX_KEYS},
        {
            "aboveground_biomass_carbon": stocks["aboveground_biomass_carbon"],
            "belowground_biomass_carbon": stocks["belowground_biomass_carbon"],
            CANOPY_KEY: extracted.stocks[CANOPY_KEY],
        },
    )


def make_recurrent_science_loss(
    jax: Any,
    jnp_module: Any,
    *,
    contract: Any,
    flux_scales: Mapping[str, float],
    stock_daily_change_scales: Mapping[str, float],
    weights: Mapping[str, Mapping[str, float]],
    trajectory_extractor: Callable[[Mapping[str, Any], Any], ScienceTrajectories] = shared_science_trajectories,
) -> Callable[[Any, Any, Mapping[str, Any], Mapping[str, Any], Any], Any]:
    """Build a no-default recurrent loss compatible with ``RolloutLoss.loss_fn``.

    ``flux_scales`` and ``stock_daily_change_scales`` are supplied physical
    scales.  No standing-stock scale, fraction stop-gradient, or default
    training coefficient is introduced here.
    """

    del jax  # Kept in the factory API alongside existing rollout-loss factories.
    del jnp_module
    flux_scales = _positive_mapping(flux_scales, FLUX_KEYS, name="flux_scales")
    stock_scales = _positive_mapping(stock_daily_change_scales, STOCK_KEYS, name="stock_daily_change_scales")
    weights = _weights(weights)

    def terms(
        params: Any,
        initial_continuous: Any,
        observations: Mapping[str, Any],
        targets: Mapping[str, Any],
        loss_context: Any,
    ) -> RecurrentScienceLossResult:
        del params, initial_continuous, loss_context
        predicted_fluxes, predicted_stocks = _accepted_science(trajectory_extractor(observations, contract))
        target_fluxes, target_stocks = _accepted_science(trajectory_extractor(targets, contract))
        daily_flux = {
            key: _masked_term(predicted_fluxes[key], target_fluxes[key], scale=flux_scales[key], pseudo_huber=False)
            for key in FLUX_KEYS
        }
        cumulative_flux = {
            key: _cumulative_term(predicted_fluxes[key], target_fluxes[key], scale=flux_scales[key])
            for key in FLUX_KEYS
        }
        stock_endpoint = {
            key: _stock_endpoint_term(
                predicted_stocks[key], target_stocks[key], daily_scale=stock_scales[key], pseudo_huber=key == CANOPY_KEY
            )
            for key in STOCK_KEYS
        }
        stock_trajectory = {
            key: _stock_trajectory_term(
                predicted_stocks[key], target_stocks[key], daily_scale=stock_scales[key], pseudo_huber=key == CANOPY_KEY
            )
            for key in STOCK_KEYS
        }
        sections = {
            "daily_flux": daily_flux,
            "cumulative_flux": cumulative_flux,
            "stock_endpoint": stock_endpoint,
            "stock_trajectory": stock_trajectory,
        }
        weighted_components = {
            section: sum(jnp.asarray(weights[section][key]) * value.loss for key, value in values.items())
            for section, values in sections.items()
        }
        return RecurrentScienceLossResult(
            total=sum(weighted_components.values()),
            weighted_components=weighted_components,
            daily_flux=daily_flux,
            cumulative_flux=cumulative_flux,
            stock_endpoint=stock_endpoint,
            stock_trajectory=stock_trajectory,
        )

    def loss_fn(
        params: Any,
        initial_continuous: Any,
        observations: Mapping[str, Any],
        targets: Mapping[str, Any],
        loss_context: Any,
    ) -> Any:
        return terms(params, initial_continuous, observations, targets, loss_context).total

    # Keep the RolloutLoss callable scalar while retaining pure-JAX diagnostics.
    loss_fn.terms = terms
    return loss_fn
