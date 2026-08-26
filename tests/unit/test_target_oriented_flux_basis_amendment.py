from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator import (
    COMPONENT_RESOLVED_FLUX_BASIS,
    INDEPENDENT_FLUX_BASIS,
    INDEPENDENT_STOCK_BASIS,
    MaskedTrajectory,
    linearized_science_flux_loss,
    linearized_science_stock_loss,
    linearized_science_window_loss,
    science_window_objective,
)
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    FROZEN_CONTRACT_SHA256,
    load_target_oriented_flux_basis_amendment,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    ErrorSufficientStatistics,
    ScienceWindowSufficientStatistics,
    science_window_from_sufficient_statistics,
    science_window_sufficient_statistics,
)


def _trajectory(value: float) -> MaskedTrajectory:
    return MaskedTrajectory(
        value=jnp.full((30,), value, dtype=jnp.float64),
        defined=jnp.ones((30,), dtype=jnp.bool_),
    )


def _fluxes(*, maintenance: float, growth: float):
    return {
        "gpp": _trajectory(10.0),
        "npp": _trajectory(7.0),
        "maintenance_respiration": _trajectory(maintenance),
        "growth_respiration": _trajectory(growth),
        "heterotrophic_respiration": _trajectory(3.0),
    }


def _stocks():
    return {
        "aboveground_biomass_carbon": _trajectory(10.0),
        "belowground_biomass_carbon": _trajectory(5.0),
        "total_live_biomass_carbon": _trajectory(15.0),
        "ecosystem_carbon": _trajectory(25.0),
    }


def test_flux_basis_amendment_is_frozen_without_mutating_the_parent_basis():
    amendment = load_target_oriented_flux_basis_amendment()

    assert amendment.sha256 == FROZEN_CONTRACT_SHA256
    assert INDEPENDENT_FLUX_BASIS == (
        "gpp",
        "autotrophic_respiration",
        "heterotrophic_respiration",
    )
    assert amendment.flux_basis == COMPONENT_RESOLVED_FLUX_BASIS
    assert amendment.stock_basis == INDEPENDENT_STOCK_BASIS
    assert amendment.payload["data"] == {
        "release_id": "pft14-daily-teacher-v6",
        "split": "train/train",
        "teacher_regeneration_allowed": False,
        "historical_v5_allowed": False,
        "sealed_test_allowed": False,
    }


def test_component_resolved_basis_exposes_compensating_respiration_errors():
    expected_fluxes = _fluxes(maintenance=2.0, growth=1.0)
    predicted_fluxes = _fluxes(maintenance=3.0, growth=0.0)
    stocks = _stocks()

    aggregate = science_window_objective(
        predicted_fluxes,
        expected_fluxes,
        stocks,
        stocks,
        flux_scales={name: 1.0 for name in INDEPENDENT_FLUX_BASIS},
        stock_scales={name: 1.0 for name in INDEPENDENT_STOCK_BASIS},
        horizon=7,
    )
    resolved = science_window_objective(
        predicted_fluxes,
        expected_fluxes,
        stocks,
        stocks,
        flux_scales={name: 1.0 for name in COMPONENT_RESOLVED_FLUX_BASIS},
        stock_scales={name: 1.0 for name in INDEPENDENT_STOCK_BASIS},
        horizon=7,
        flux_basis=COMPONENT_RESOLVED_FLUX_BASIS,
    )

    np.testing.assert_allclose(aggregate.total, 0.0)
    np.testing.assert_allclose(resolved.flux_mse, 0.5)
    np.testing.assert_allclose(resolved.flux_bias_squared, 0.5)
    np.testing.assert_allclose(resolved.total, 0.25)
    assert tuple(resolved.flux_errors) == COMPONENT_RESOLVED_FLUX_BASIS

    statistics = science_window_sufficient_statistics(
        predicted_fluxes,
        expected_fluxes,
        stocks,
        stocks,
        flux_scales={name: 1.0 for name in COMPONENT_RESOLVED_FLUX_BASIS},
        stock_scales={name: 1.0 for name in INDEPENDENT_STOCK_BASIS},
        horizon=7,
        flux_basis=COMPONENT_RESOLVED_FLUX_BASIS,
    )
    sufficient = science_window_from_sufficient_statistics(
        statistics,
        flux_basis=COMPONENT_RESOLVED_FLUX_BASIS,
    )
    np.testing.assert_allclose(sufficient.total, resolved.total)
    np.testing.assert_allclose(sufficient.flux_mse, resolved.flux_mse)
    np.testing.assert_allclose(sufficient.flux_bias_squared, resolved.flux_bias_squared)


def _statistics(names: tuple[str, ...], scale: float) -> dict[str, ErrorSufficientStatistics]:
    return {
        name: ErrorSufficientStatistics(
            sum_squared=jnp.asarray(scale * (index + 1), dtype=jnp.float64),
            sum_signed=jnp.asarray(0.1 * scale * (index + 1), dtype=jnp.float64),
            count=jnp.asarray(4.0, dtype=jnp.float64),
        )
        for index, name in enumerate(names)
    }


def test_split_science_linearization_sums_to_the_complete_amended_loss():
    current = ScienceWindowSufficientStatistics(
        fluxes=_statistics(COMPONENT_RESOLVED_FLUX_BASIS, 1.0),
        stocks=_statistics(INDEPENDENT_STOCK_BASIS, 2.0),
    )
    total = ScienceWindowSufficientStatistics(
        fluxes=_statistics(COMPONENT_RESOLVED_FLUX_BASIS, 3.0),
        stocks=_statistics(INDEPENDENT_STOCK_BASIS, 4.0),
    )

    flux = linearized_science_flux_loss(
        current,
        total,
        flux_basis=COMPONENT_RESOLVED_FLUX_BASIS,
    )
    stock = linearized_science_stock_loss(current, total)
    complete = linearized_science_window_loss(
        current,
        total,
        flux_basis=COMPONENT_RESOLVED_FLUX_BASIS,
    )

    np.testing.assert_allclose(complete, flux + stock, rtol=0.0, atol=0.0)
