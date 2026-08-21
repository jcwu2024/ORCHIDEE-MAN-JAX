from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator import (
    MaskedTrajectory,
    load_target_oriented_contract,
    one_day_state_anchor,
    science_window_objective,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_science_window_sufficient_statistics,
    linearized_one_day_state_loss,
    linearized_process_supervision_loss,
    linearized_science_window_loss,
    merge_error_sufficient_statistics,
    necessary_process_supervision_from_sufficient_statistics,
    one_day_state_anchor_from_sufficient_statistics,
    one_day_state_sufficient_statistics,
    process_supervision_sufficient_statistics,
    science_window_from_sufficient_statistics,
    science_window_sufficient_statistics,
)
from research.daily_coarse_graining.daily_operator.training_loss import ProcessLossResult
from research.daily_coarse_graining.daily_operator.types import (
    StateDefinedMasks,
    StateFeatureGroups,
)


def _state_groups(value):
    return StateFeatureGroups(**{name: value for name in StateFeatureGroups._fields})


def _state_masks(value):
    return StateDefinedMasks(**{name: value for name in StateFeatureGroups._fields})


def test_state_microbatch_statistics_preserve_full_loss_and_gradient():
    expected = jnp.arange(24, dtype=jnp.float64).reshape(8, 3) / 10.0
    feature = jnp.linspace(-1.0, 2.0, 24, dtype=jnp.float64).reshape(8, 3)
    defined = jnp.ones_like(expected, dtype=jnp.bool_).at[1, 2].set(False)

    def full(parameter):
        return one_day_state_anchor(
            _state_groups(expected + parameter * feature),
            _state_groups(expected),
            _state_masks(defined),
            _state_masks(defined),
            _state_groups(jnp.asarray([1.0, 2.0, 3.0])),
        ).total

    def accumulated(parameter):
        pieces = []
        for start in range(0, 8, 2):
            selected = slice(start, start + 2)
            pieces.append(
                one_day_state_sufficient_statistics(
                    _state_groups(expected[selected] + parameter * feature[selected]),
                    _state_groups(expected[selected]),
                    _state_masks(defined[selected]),
                    _state_masks(defined[selected]),
                    _state_groups(jnp.asarray([1.0, 2.0, 3.0])),
                )
            )
        merged = merge_error_sufficient_statistics(pieces)
        return one_day_state_anchor_from_sufficient_statistics(merged).total

    parameter = jnp.asarray(0.7, dtype=jnp.float64)
    np.testing.assert_allclose(accumulated(parameter), full(parameter), rtol=1e-12)
    np.testing.assert_allclose(
        jax.grad(accumulated)(parameter),
        jax.grad(full)(parameter),
        rtol=1e-12,
    )

    total = merge_error_sufficient_statistics(
        [
            one_day_state_sufficient_statistics(
                _state_groups(expected[selected] + parameter * feature[selected]),
                _state_groups(expected[selected]),
                _state_masks(defined[selected]),
                _state_masks(defined[selected]),
                _state_groups(jnp.asarray([1.0, 2.0, 3.0])),
            )
            for selected in (slice(start, start + 2) for start in range(0, 8, 2))
        ]
    )
    independent_gradient = 0.0
    for start in range(0, 8, 2):
        selected = slice(start, start + 2)

        def local(value):
            current = one_day_state_sufficient_statistics(
                _state_groups(expected[selected] + value * feature[selected]),
                _state_groups(expected[selected]),
                _state_masks(defined[selected]),
                _state_masks(defined[selected]),
                _state_groups(jnp.asarray([1.0, 2.0, 3.0])),
            )
            return linearized_one_day_state_loss(current, total)

        independent_gradient += jax.grad(local)(parameter)
    np.testing.assert_allclose(independent_gradient, jax.grad(full)(parameter), rtol=1e-12)


def _trajectory(value, defined):
    return MaskedTrajectory(value=value, defined=defined)


def _science_trajectories(parameter, selected):
    time = jnp.arange(30, dtype=jnp.float64)[:, None]
    batch = jnp.arange(8, dtype=jnp.float64)[None, :]
    feature = (time + 1.0) * (batch + 2.0) / 100.0
    expected_flux = 1.0 + time * 0.01 + batch * 0.02
    expected_stock = 10.0 + time * 0.1 + batch * 0.2
    defined = jnp.ones((30, 8), dtype=jnp.bool_).at[3, 1].set(False)

    def take(value):
        return value[:, selected]

    predicted_fluxes = {
        "gpp": _trajectory(take(expected_flux + parameter * feature), take(defined)),
        "maintenance_respiration": _trajectory(take(0.2 * expected_flux + 0.3 * parameter * feature), take(defined)),
        "growth_respiration": _trajectory(take(0.1 * expected_flux), take(defined)),
        "heterotrophic_respiration": _trajectory(take(0.15 * expected_flux - 0.2 * parameter * feature), take(defined)),
    }
    expected_fluxes = {
        "gpp": _trajectory(take(expected_flux), take(defined)),
        "maintenance_respiration": _trajectory(take(0.2 * expected_flux), take(defined)),
        "growth_respiration": _trajectory(take(0.1 * expected_flux), take(defined)),
        "heterotrophic_respiration": _trajectory(take(0.15 * expected_flux), take(defined)),
    }
    predicted_stocks = {
        "aboveground_biomass_carbon": _trajectory(take(expected_stock + 0.4 * parameter * feature), take(defined)),
        "belowground_biomass_carbon": _trajectory(
            take(0.5 * expected_stock - 0.1 * parameter * feature), take(defined)
        ),
        "total_live_biomass_carbon": _trajectory(take(1.5 * expected_stock + 0.3 * parameter * feature), take(defined)),
        "ecosystem_carbon": _trajectory(take(2.0 * expected_stock + 0.6 * parameter * feature), take(defined)),
    }
    expected_stocks = {
        "aboveground_biomass_carbon": _trajectory(take(expected_stock), take(defined)),
        "belowground_biomass_carbon": _trajectory(take(0.5 * expected_stock), take(defined)),
        "total_live_biomass_carbon": _trajectory(take(1.5 * expected_stock), take(defined)),
        "ecosystem_carbon": _trajectory(take(2.0 * expected_stock), take(defined)),
    }
    return predicted_fluxes, expected_fluxes, predicted_stocks, expected_stocks


FLUX_SCALES = {
    "gpp": 1.0,
    "autotrophic_respiration": 0.5,
    "heterotrophic_respiration": 0.25,
}
STOCK_SCALES = {
    "aboveground_biomass_carbon": 2.0,
    "belowground_biomass_carbon": 1.0,
    "nonliving_ecosystem_carbon": 3.0,
}


def test_science_microbatch_statistics_preserve_global_bias_and_gradient():
    def full(parameter):
        trajectories = _science_trajectories(parameter, slice(None))
        return science_window_objective(
            *trajectories,
            flux_scales=FLUX_SCALES,
            stock_scales=STOCK_SCALES,
            horizon=30,
        ).total

    def accumulated(parameter):
        statistics = None
        for index in range(8):
            current = science_window_sufficient_statistics(
                *_science_trajectories(parameter, slice(index, index + 1)),
                flux_scales=FLUX_SCALES,
                stock_scales=STOCK_SCALES,
                horizon=30,
            )
            statistics = (
                current if statistics is None else add_science_window_sufficient_statistics(statistics, current)
            )
        return science_window_from_sufficient_statistics(statistics).total

    parameter = jnp.asarray(0.6, dtype=jnp.float64)
    np.testing.assert_allclose(accumulated(parameter), full(parameter), rtol=1e-12)
    np.testing.assert_allclose(
        jax.grad(accumulated)(parameter),
        jax.grad(full)(parameter),
        rtol=1e-12,
    )

    total = None
    for index in range(8):
        current = science_window_sufficient_statistics(
            *_science_trajectories(parameter, slice(index, index + 1)),
            flux_scales=FLUX_SCALES,
            stock_scales=STOCK_SCALES,
            horizon=30,
        )
        total = current if total is None else add_science_window_sufficient_statistics(total, current)
    independent_gradient = 0.0
    for index in range(8):

        def local(value):
            current = science_window_sufficient_statistics(
                *_science_trajectories(value, slice(index, index + 1)),
                flux_scales=FLUX_SCALES,
                stock_scales=STOCK_SCALES,
                horizon=30,
            )
            return linearized_science_window_loss(current, total)

        independent_gradient += jax.grad(local)(parameter)
    np.testing.assert_allclose(independent_gradient, jax.grad(full)(parameter), rtol=1e-12)


def test_process_microbatch_statistics_preserve_label_weighting():
    contract = load_target_oriented_contract()
    labels = tuple(label for category in contract.labels_by_category.values() for label in category)

    def microbatch(parameter, offset, count):
        mse = {label: (parameter - (index % 5) * 0.1 - offset) ** 2 for index, label in enumerate(labels)}
        counts = {label: jnp.asarray(count + index % 3) for index, label in enumerate(labels)}
        return ProcessLossResult(jnp.asarray(0.0), mse, counts, {}, {})

    def accumulated(parameter):
        pieces = [
            process_supervision_sufficient_statistics(microbatch(parameter, 0.0, 2)),
            process_supervision_sufficient_statistics(microbatch(parameter, 0.3, 5)),
        ]
        return necessary_process_supervision_from_sufficient_statistics(
            merge_error_sufficient_statistics(pieces), contract
        ).total

    def direct(parameter):
        weights = {
            category: value
            for category, value in zip(
                contract.labels_by_category,
                (0.0, 1.0, 0.25),
                strict=True,
            )
        }
        numerator = jnp.asarray(0.0, dtype=jnp.float64)
        denominator = jnp.asarray(0.0, dtype=jnp.float64)
        for category, category_labels in contract.labels_by_category.items():
            for label in category_labels:
                index = labels.index(label)
                first_count = 2 + index % 3
                second_count = 5 + index % 3
                first = (parameter - (index % 5) * 0.1) ** 2
                second = (parameter - (index % 5) * 0.1 - 0.3) ** 2
                mse = (first * first_count + second * second_count) / (first_count + second_count)
                numerator = numerator + weights[category] * mse
                denominator = denominator + weights[category]
        return numerator / denominator

    parameter = jnp.asarray(0.8, dtype=jnp.float64)
    np.testing.assert_allclose(accumulated(parameter), direct(parameter), rtol=1e-12)
    np.testing.assert_allclose(
        jax.grad(accumulated)(parameter),
        jax.grad(direct)(parameter),
        rtol=1e-12,
    )

    total = merge_error_sufficient_statistics(
        [
            process_supervision_sufficient_statistics(microbatch(parameter, 0.0, 2)),
            process_supervision_sufficient_statistics(microbatch(parameter, 0.3, 5)),
        ]
    )
    independent_gradient = sum(
        (
            jax.grad(
                lambda value, offset=offset, count=count: linearized_process_supervision_loss(
                    process_supervision_sufficient_statistics(microbatch(value, offset, count)),
                    total,
                    contract,
                )
            )(parameter)
            for offset, count in ((0.0, 2), (0.3, 5))
        ),
        jnp.asarray(0.0),
    )
    np.testing.assert_allclose(independent_gradient, jax.grad(direct)(parameter), rtol=1e-12)
