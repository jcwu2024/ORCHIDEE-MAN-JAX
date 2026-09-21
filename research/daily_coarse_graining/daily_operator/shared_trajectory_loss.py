"""Field-balanced trajectory loss for the shared daily process operator.

This module contains no process dynamics.  It projects existing source-owned
daily states into physical carbon quantities, then scores daily errors and
coherent rollout drift.  The returned scalar covers the flux and stock science
groups only; callers combine those groups with the separate process-day loss.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from .long_horizon_objective import MaskedTrajectory, derive_area_weighted_pft_carbon_pool
from .successor_training import (
    _normalized_pft_context,
    _reduced_stock_defined,
    _trajectory_field,
    extract_biomass_productivity_trajectories,
)

FLUX_KEYS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
    "heterotrophic_respiration",
)
LIVING_STOCK_KEYS = (
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
)
NONLIVING_STOCK_KEYS = (
    "aboveground_litter_carbon",
    "belowground_litter_carbon",
    "soil_particulate_organic_carbon",
    "dissolved_organic_carbon",
    "canopy_interception_organic_carbon",
)
STOCK_KEYS = LIVING_STOCK_KEYS + NONLIVING_STOCK_KEYS


class SharedComponentTrajectories(NamedTuple):
    """Independent physical flux and stock trajectories used by the loss."""

    fluxes: Mapping[str, MaskedTrajectory]
    stocks: Mapping[str, MaskedTrajectory]


class TermDiagnostics(NamedTuple):
    """One masked pseudo-Huber term and its validity accounting."""

    loss: Any
    count: Any
    expected_count: Any
    invalid_prediction_count: Any
    invalid_target_count: Any
    normalized_residual: Any


class SharedTrajectoryHardCounts(NamedTuple):
    """Failures that must not be interpreted as a small masked loss."""

    nonfinite_defined_prediction: Any
    nonfinite_defined_target: Any
    defined_mask_mismatch: Any
    missing_valid_term: Any
    total: Any


class SharedTrajectoryLossResult(NamedTuple):
    """Field-level diagnostics plus the two equal science-group macros."""

    total: Any
    flux: Any
    stock: Any
    living_stock: Any
    nonliving_stock: Any
    flux_daily: Mapping[str, TermDiagnostics]
    flux_prefix: Mapping[str, TermDiagnostics]
    stock_change: Mapping[str, TermDiagnostics]
    stock_trajectory: Mapping[str, TermDiagnostics]
    hard_counts: SharedTrajectoryHardCounts


class SharedTrajectoryScales(NamedTuple):
    """Frozen CPU-derived physical scales and their TRAIN coverage counts."""

    flux_scales: Mapping[str, float]
    stock_change_scales: Mapping[str, float]
    flux_counts: Mapping[str, int]
    stock_change_counts: Mapping[str, int]
    flux_nonzero_counts: Mapping[str, int]
    stock_change_nonzero_counts: Mapping[str, int]


def _trajectory(value: Any, defined: Any, *, name: str) -> MaskedTrajectory:
    value = jnp.asarray(value, dtype=jnp.float64)
    defined = jnp.asarray(defined, dtype=jnp.bool_)
    if value.ndim < 1 or value.shape != defined.shape:
        raise ValueError(f"{name} must provide equal time-leading value and defined arrays")
    if value.shape[0] < 1:
        raise ValueError(f"{name} cannot be empty")
    return MaskedTrajectory(value, defined)


def _positive_scales(values: Mapping[str, float], keys: tuple[str, ...], *, name: str) -> dict[str, float]:
    if set(values) != set(keys):
        raise ValueError(f"{name} keys must be exactly {keys}")
    result = {}
    for key in keys:
        value = float(values[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name}.{key} must be finite and positive")
        result[key] = value
    return result


def _prefix_mask(mask: Any) -> Any:
    return jnp.cumprod(jnp.asarray(mask, dtype=jnp.int32), axis=0).astype(jnp.bool_)


def _pseudo_huber(normalized_residual: Any) -> Any:
    squared = normalized_residual * normalized_residual
    return squared / (jnp.sqrt(1.0 + squared) + 1.0)


def _term(
    normalized_residual: Any,
    *,
    expected: Any,
    prediction_valid: Any,
    invalid_target_count: Any,
) -> TermDiagnostics:
    normalized = jnp.asarray(normalized_residual, dtype=jnp.float64)
    expected = jnp.asarray(expected, dtype=jnp.bool_)
    prediction_valid = jnp.asarray(prediction_valid, dtype=jnp.bool_)
    valid = expected & prediction_valid & jnp.isfinite(normalized)
    safe = jnp.where(valid, normalized, 0.0)
    count = jnp.sum(valid, dtype=jnp.float64)
    return TermDiagnostics(
        loss=jnp.sum(_pseudo_huber(safe), dtype=jnp.float64) / jnp.maximum(count, 1.0),
        count=count,
        expected_count=jnp.sum(expected, dtype=jnp.float64),
        invalid_prediction_count=jnp.sum(expected & ~prediction_valid, dtype=jnp.float64),
        invalid_target_count=jnp.asarray(invalid_target_count, dtype=jnp.float64),
        normalized_residual=safe,
    )


def _validate_pair(
    predicted: Mapping[str, MaskedTrajectory],
    target: Mapping[str, MaskedTrajectory],
    keys: tuple[str, ...],
    *,
    kind: str,
) -> tuple[dict[str, MaskedTrajectory], dict[str, MaskedTrajectory]]:
    expected = set(keys)
    for label, registry in (("prediction", predicted), ("target", target)):
        if set(registry) != expected:
            missing = sorted(expected - set(registry))
            extra = sorted(set(registry) - expected)
            raise ValueError(f"{kind} {label} registry drift; missing={missing}, extra={extra}")
    predicted_result = {}
    target_result = {}
    day_counts = set()
    for key in keys:
        predicted_value = _trajectory(predicted[key].value, predicted[key].defined, name=f"predicted {kind}.{key}")
        target_value = _trajectory(target[key].value, target[key].defined, name=f"target {kind}.{key}")
        if predicted_value.value.shape != target_value.value.shape:
            raise ValueError(f"{kind}.{key} prediction and target shapes differ")
        predicted_result[key] = predicted_value
        target_result[key] = target_value
        day_counts.add(predicted_value.value.shape[0])
    if len(day_counts) != 1:
        raise ValueError(f"{kind} trajectories have inconsistent day counts")
    return predicted_result, target_result


def _raw_hard_counts(
    predicted: Mapping[str, MaskedTrajectory], target: Mapping[str, MaskedTrajectory], keys: tuple[str, ...]
) -> tuple[Any, Any, Any]:
    nonfinite_prediction = jnp.asarray(0, dtype=jnp.int32)
    nonfinite_target = jnp.asarray(0, dtype=jnp.int32)
    mask_mismatch = jnp.asarray(0, dtype=jnp.int32)
    for key in keys:
        predicted_value = predicted[key]
        target_value = target[key]
        nonfinite_prediction += jnp.sum(
            predicted_value.defined & ~jnp.isfinite(predicted_value.value), dtype=jnp.int32
        )
        nonfinite_target += jnp.sum(target_value.defined & ~jnp.isfinite(target_value.value), dtype=jnp.int32)
        mask_mismatch += jnp.sum(predicted_value.defined != target_value.defined, dtype=jnp.int32)
    return nonfinite_prediction, nonfinite_target, mask_mismatch


def _mean(values: Iterable[Any]) -> Any:
    values = tuple(values)
    if not values:
        raise ValueError("cannot average an empty registry")
    return sum(values, jnp.asarray(0.0, dtype=jnp.float64)) / len(values)


def shared_component_trajectories(observation: Mapping[str, Any], contract: Any) -> SharedComponentTrajectories:
    """Project the seven independent stocks without changing process logic.

    Litter/SOC/DOC retain the PFT-covered-area convention used by the shared
    successor extractor.  Canopy interception is already on the land-area
    basis (``stomate_soilcarbon.f90::soilcarbon_leak`` lines 1225, 1245-1251,
    1497-1525, and 2326-2332), so it is summed with unit PFT weights.
    """

    base = extract_biomass_productivity_trajectories(
        observation["state"],
        observation["defined"],
        daily_fast_gpp=observation["gpp"],
        daily_fast_gpp_defined=observation["gpp_defined"],
        pft_fraction=observation["fraction"],
        active_pft_mask=observation["active"],
        contract=contract,
    )
    state = jnp.asarray(observation["state"], dtype=jnp.float64)
    defined = jnp.asarray(observation["defined"], dtype=jnp.bool_)
    pft_width = len(contract.active_pft_indices)
    fraction, active = _normalized_pft_context(
        observation["fraction"], observation["active"], days=state.shape[0], pft_width=pft_width
    )

    split_litter = {}
    split_litter_defined = {}
    for output_name, state_name in (
        ("aboveground_litter_carbon", "litter_above"),
        ("belowground_litter_carbon", "litter_below"),
    ):
        value, mask = _trajectory_field(state, defined, contract, state_name)
        split_litter[output_name] = derive_area_weighted_pft_carbon_pool(
            value, fraction, active, pft_axis=3, element_axis=-1
        )
        split_litter_defined[output_name] = _reduced_stock_defined(mask, active, pft_axis=3)

    canopy, canopy_defined = _trajectory_field(state, defined, contract, "interception_storage")
    canopy_value = derive_area_weighted_pft_carbon_pool(
        canopy, jnp.ones_like(fraction), active, pft_axis=2, element_axis=-1
    )
    stocks = {
        "aboveground_biomass_carbon": base.stocks["aboveground_biomass_carbon"],
        "belowground_biomass_carbon": base.stocks["belowground_biomass_carbon"],
        "aboveground_litter_carbon": MaskedTrajectory(
            split_litter["aboveground_litter_carbon"], split_litter_defined["aboveground_litter_carbon"]
        ),
        "belowground_litter_carbon": MaskedTrajectory(
            split_litter["belowground_litter_carbon"], split_litter_defined["belowground_litter_carbon"]
        ),
        "soil_particulate_organic_carbon": base.stocks["soil_particulate_organic_carbon"],
        "dissolved_organic_carbon": base.stocks["dissolved_organic_carbon"],
        "canopy_interception_organic_carbon": MaskedTrajectory(
            canopy_value, _reduced_stock_defined(canopy_defined, active, pft_axis=2)
        ),
    }
    return SharedComponentTrajectories(
        fluxes={key: base.fluxes[key] for key in FLUX_KEYS},
        stocks=stocks,
    )


def make_shared_trajectory_loss(
    contract: Any,
    flux_scales: Mapping[str, float],
    stock_change_scales: Mapping[str, float],
    *,
    trajectory_extractor: Callable[[Mapping[str, Any], Any], SharedComponentTrajectories] = shared_component_trajectories,
) -> Callable[[Any, Any, Mapping[str, Any], Mapping[str, Any], Any], Any]:
    """Build the equal-flux/equal-living-nonliving science loss.

    Prediction and target rollouts share their initial state.  Consequently
    stock residuals are also errors in change from that common initial state,
    with the pre-window residual defined as zero.
    """

    flux_scales = _positive_scales(flux_scales, FLUX_KEYS, name="flux_scales")
    stock_scales = _positive_scales(stock_change_scales, STOCK_KEYS, name="stock_change_scales")

    def terms(
        params: Any,
        initial_continuous: Any,
        observations: Mapping[str, Any],
        targets: Mapping[str, Any],
        loss_context: Any,
    ) -> SharedTrajectoryLossResult:
        del params, initial_continuous, loss_context
        predicted = trajectory_extractor(observations, contract)
        expected = trajectory_extractor(targets, contract)
        predicted_flux, expected_flux = _validate_pair(
            predicted.fluxes, expected.fluxes, FLUX_KEYS, kind="flux"
        )
        predicted_stock, expected_stock = _validate_pair(
            predicted.stocks, expected.stocks, STOCK_KEYS, kind="stock"
        )

        flux_daily = {}
        flux_prefix = {}
        for key in FLUX_KEYS:
            prediction = predicted_flux[key]
            target = expected_flux[key]
            target_valid = target.defined & jnp.isfinite(target.value)
            prediction_valid = prediction.defined & jnp.isfinite(prediction.value)
            pair_valid = target_valid & prediction_valid
            error = jnp.where(pair_valid, prediction.value - target.value, 0.0)
            invalid_target = jnp.sum(target.defined & ~jnp.isfinite(target.value), dtype=jnp.float64)
            flux_daily[key] = _term(
                error / flux_scales[key],
                expected=target_valid,
                prediction_valid=prediction_valid,
                invalid_target_count=invalid_target,
            )

            target_prefix = _prefix_mask(target_valid)
            prediction_prefix = _prefix_mask(prediction_valid)
            elapsed = jnp.arange(1, error.shape[0] + 1, dtype=jnp.float64)
            scale_shape = (error.shape[0],) + (1,) * (error.ndim - 1)
            prefix_residual = jnp.cumsum(error, axis=0) / (
                flux_scales[key] * jnp.sqrt(jnp.reshape(elapsed, scale_shape))
            )
            flux_prefix[key] = _term(
                prefix_residual,
                expected=target_prefix,
                prediction_valid=prediction_prefix,
                invalid_target_count=invalid_target,
            )

        stock_change = {}
        stock_trajectory = {}
        for key in STOCK_KEYS:
            prediction = predicted_stock[key]
            target = expected_stock[key]
            target_valid = target.defined & jnp.isfinite(target.value)
            prediction_valid = prediction.defined & jnp.isfinite(prediction.value)
            pair_valid = target_valid & prediction_valid
            residual = jnp.where(pair_valid, prediction.value - target.value, 0.0)
            previous = jnp.concatenate((jnp.zeros_like(residual[:1]), residual[:-1]), axis=0)
            change_residual = residual - previous
            expected_change = jnp.concatenate(
                (target_valid[:1], target_valid[1:] & target_valid[:-1]), axis=0
            )
            prediction_change = jnp.concatenate(
                (prediction_valid[:1], prediction_valid[1:] & prediction_valid[:-1]), axis=0
            )
            invalid_target = jnp.sum(target.defined & ~jnp.isfinite(target.value), dtype=jnp.float64)
            stock_change[key] = _term(
                change_residual / stock_scales[key],
                expected=expected_change,
                prediction_valid=prediction_change,
                invalid_target_count=invalid_target,
            )

            elapsed = jnp.arange(1, residual.shape[0] + 1, dtype=jnp.float64)
            scale_shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
            stock_trajectory[key] = _term(
                residual / (stock_scales[key] * jnp.sqrt(jnp.reshape(elapsed, scale_shape))),
                expected=target_valid,
                prediction_valid=prediction_valid,
                invalid_target_count=invalid_target,
            )

        flux_field_losses = {
            key: 0.5 * (flux_daily[key].loss + flux_prefix[key].loss) for key in FLUX_KEYS
        }
        stock_field_losses = {
            key: 0.5 * (stock_change[key].loss + stock_trajectory[key].loss) for key in STOCK_KEYS
        }
        flux_macro = _mean(flux_field_losses.values())
        living_macro = _mean(stock_field_losses[key] for key in LIVING_STOCK_KEYS)
        nonliving_macro = _mean(stock_field_losses[key] for key in NONLIVING_STOCK_KEYS)
        stock_macro = 0.5 * (living_macro + nonliving_macro)

        flux_hard = _raw_hard_counts(predicted_flux, expected_flux, FLUX_KEYS)
        stock_hard = _raw_hard_counts(predicted_stock, expected_stock, STOCK_KEYS)
        all_terms = tuple(flux_daily.values()) + tuple(flux_prefix.values()) + tuple(stock_change.values()) + tuple(
            stock_trajectory.values()
        )
        missing_valid = sum(
            (term.expected_count > 0.0) & (term.count <= 0.0) for term in all_terms
        ).astype(jnp.int32)
        nonfinite_prediction = flux_hard[0] + stock_hard[0]
        nonfinite_target = flux_hard[1] + stock_hard[1]
        mask_mismatch = flux_hard[2] + stock_hard[2]
        hard_total = nonfinite_prediction + nonfinite_target + mask_mismatch + missing_valid
        return SharedTrajectoryLossResult(
            total=0.5 * (flux_macro + stock_macro),
            flux=flux_macro,
            stock=stock_macro,
            living_stock=living_macro,
            nonliving_stock=nonliving_macro,
            flux_daily=flux_daily,
            flux_prefix=flux_prefix,
            stock_change=stock_change,
            stock_trajectory=stock_trajectory,
            hard_counts=SharedTrajectoryHardCounts(
                nonfinite_prediction,
                nonfinite_target,
                mask_mismatch,
                missing_valid,
                hard_total,
            ),
        )

    def loss_fn(
        params: Any,
        initial_continuous: Any,
        observations: Mapping[str, Any],
        targets: Mapping[str, Any],
        loss_context: Any,
    ) -> Any:
        return terms(params, initial_continuous, observations, targets, loss_context).total

    loss_fn.terms = terms
    return loss_fn


def build_shared_trajectory_scales(
    samples: Iterable[tuple[SharedComponentTrajectories, Mapping[str, MaskedTrajectory]]],
    *,
    split_label: str,
    flux_floors: Mapping[str, float],
    stock_change_floors: Mapping[str, float],
) -> SharedTrajectoryScales:
    """Stream frozen TRAIN-only RMS scales on extracted physical quantities.

    Each sample is ``(target_trajectory, initial_target_stocks)``.  Initial
    stock values have the trajectory's non-time shape and make the first daily
    change an actual Teacher transition instead of an endpoint magnitude.
    The split label is deliberately strict, and callers must pass explicit
    positive physical-unit floors; validation/test-driven calibration is not
    admitted.
    """

    if split_label != "train/train":
        raise ValueError("shared trajectory scales require split_label='train/train'")
    flux_floors = _positive_scales(flux_floors, FLUX_KEYS, name="flux_floors")
    stock_floors = _positive_scales(stock_change_floors, STOCK_KEYS, name="stock_change_floors")
    flux_sum_squares = {key: 0.0 for key in FLUX_KEYS}
    stock_sum_squares = {key: 0.0 for key in STOCK_KEYS}
    flux_counts = {key: 0 for key in FLUX_KEYS}
    stock_counts = {key: 0 for key in STOCK_KEYS}
    flux_nonzero = {key: 0 for key in FLUX_KEYS}
    stock_nonzero = {key: 0 for key in STOCK_KEYS}

    for sample_index, (trajectories, initial_stocks) in enumerate(samples):
        if set(trajectories.fluxes) != set(FLUX_KEYS) or set(trajectories.stocks) != set(STOCK_KEYS):
            raise ValueError(f"scale sample {sample_index} trajectory registry drift")
        if set(initial_stocks) != set(STOCK_KEYS):
            raise ValueError(f"scale sample {sample_index} initial stock registry drift")
        for key in FLUX_KEYS:
            trajectory = trajectories.fluxes[key]
            values = np.asarray(trajectory.value, dtype=np.float64)
            defined = np.asarray(trajectory.defined, dtype=bool)
            if values.ndim < 1 or values.shape != defined.shape:
                raise ValueError(f"scale sample {sample_index} flux.{key} shape drift")
            if np.any(defined & ~np.isfinite(values)):
                raise ValueError(f"scale sample {sample_index} flux.{key} has nonfinite defined target")
            selected = values[defined]
            flux_sum_squares[key] += float(np.sum(selected * selected, dtype=np.float64))
            flux_counts[key] += int(selected.size)
            flux_nonzero[key] += int(np.count_nonzero(selected))

        for key in STOCK_KEYS:
            trajectory = trajectories.stocks[key]
            values = np.asarray(trajectory.value, dtype=np.float64)
            defined = np.asarray(trajectory.defined, dtype=bool)
            initial = initial_stocks[key]
            initial_value = np.asarray(initial.value, dtype=np.float64)
            initial_defined = np.asarray(initial.defined, dtype=bool)
            if values.ndim < 1 or values.shape != defined.shape:
                raise ValueError(f"scale sample {sample_index} stock.{key} shape drift")
            if initial_value.shape != values.shape[1:] or initial_defined.shape != initial_value.shape:
                raise ValueError(f"scale sample {sample_index} initial stock.{key} shape drift")
            if np.any(defined & ~np.isfinite(values)) or np.any(initial_defined & ~np.isfinite(initial_value)):
                raise ValueError(f"scale sample {sample_index} stock.{key} has nonfinite defined target")
            previous_value = np.concatenate((initial_value[None, ...], values[:-1]), axis=0)
            previous_defined = np.concatenate((initial_defined[None, ...], defined[:-1]), axis=0)
            delta_defined = defined & previous_defined
            delta = values - previous_value
            selected = delta[delta_defined]
            stock_sum_squares[key] += float(np.sum(selected * selected, dtype=np.float64))
            stock_counts[key] += int(selected.size)
            stock_nonzero[key] += int(np.count_nonzero(selected))

    if any(count == 0 for count in flux_counts.values()) or any(count == 0 for count in stock_counts.values()):
        raise ValueError("TRAIN scale coverage is empty for at least one required field")
    flux_scales = {
        key: max(math.sqrt(flux_sum_squares[key] / flux_counts[key]), flux_floors[key]) for key in FLUX_KEYS
    }
    stock_scales = {
        key: max(math.sqrt(stock_sum_squares[key] / stock_counts[key]), stock_floors[key]) for key in STOCK_KEYS
    }
    return SharedTrajectoryScales(
        flux_scales=flux_scales,
        stock_change_scales=stock_scales,
        flux_counts=flux_counts,
        stock_change_counts=stock_counts,
        flux_nonzero_counts=flux_nonzero,
        stock_change_nonzero_counts=stock_nonzero,
    )
