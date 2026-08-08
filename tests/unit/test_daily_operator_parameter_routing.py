from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.daily_operator.parameter_routing import (
    DEFAULT_PFT_PARAMETER_ROUTING,
    fast_process_parameter_factors,
    retained_exact_pft_parameters,
    route_retained_stomate_parameters,
    select_fast_process_factor_channels,
    select_retained_exact_channels,
)


def _parameters() -> np.ndarray:
    return np.asarray(
        [
            [10.0, 0.01, 0.10, 1.0, 0.20, 50.0],
            [20.0, 0.02, 0.20, 2.0, 0.30, 80.0],
        ],
        dtype=np.float64,
    )


def _stomate_parameters() -> teacher.DriverCompiledStomateParameterValues:
    return teacher.DriverCompiledStomateParameterValues(
        alloc_min=np.linspace(0.01, 0.14, 14, dtype=np.float64),
        residence_time=np.linspace(10.0, 140.0, 14, dtype=np.float64),
        vcmax25=np.linspace(30.0, 43.0, 14, dtype=np.float64),
        maint_resp_slope=np.ones((14, 3), dtype=np.float64),
    )


def test_parameter_contract_builds_complete_disjoint_runtime_views() -> None:
    routing = DEFAULT_PFT_PARAMETER_ROUTING

    assert routing.channel_names == (
        "vcmax25",
        "maint_resp_slope_a",
        "maint_resp_slope_b",
        "maint_resp_slope_c",
        "alloc_min",
        "residence_time",
    )
    assert routing.fast_process_factor_indices == (0, 1, 2, 3)
    assert routing.retained_exact_indices == (4, 5)
    np.testing.assert_array_equal(
        select_fast_process_factor_channels(_parameters()),
        _parameters()[:, :4],
    )
    np.testing.assert_array_equal(
        select_retained_exact_channels(_parameters()),
        _parameters()[:, 4:],
    )


def test_fast_factors_ignore_retained_exact_parameter_perturbations() -> None:
    parameters = _parameters()
    temperature = np.asarray([5.0, 10.0], dtype=np.float64)
    baseline = fast_process_parameter_factors(parameters, temperature)
    perturbed = parameters.copy()
    perturbed[:, 4:] = np.asarray([[0.9, 900.0], [0.8, 800.0]])
    observed = fast_process_parameter_factors(perturbed, temperature)

    np.testing.assert_array_equal(observed.vcmax25, baseline.vcmax25)
    np.testing.assert_array_equal(
        observed.maintenance_temperature_slope,
        baseline.maintenance_temperature_slope,
    )
    np.testing.assert_allclose(
        baseline.maintenance_temperature_slope,
        parameters[:, 3] + parameters[:, 2] * temperature + parameters[:, 1] * temperature**2,
    )


def test_retained_exact_view_ignores_fast_process_parameter_perturbations() -> None:
    parameters = _parameters()
    baseline = retained_exact_pft_parameters(parameters)
    perturbed = parameters.copy()
    perturbed[:, :4] *= 1000.0
    observed = retained_exact_pft_parameters(perturbed)

    np.testing.assert_array_equal(observed.alloc_min, baseline.alloc_min)
    np.testing.assert_array_equal(observed.residence_time, baseline.residence_time)


def test_fast_and_retained_parameter_gradients_are_cross_isolated() -> None:
    parameters = jnp.asarray(_parameters())
    temperature = jnp.asarray([5.0, 10.0], dtype=jnp.float64)

    def fast_objective(values):
        factors = fast_process_parameter_factors(values, temperature)
        return jnp.sum(factors.vcmax25 + factors.maintenance_temperature_slope)

    def retained_objective(values):
        retained = retained_exact_pft_parameters(values)
        return jnp.sum(retained.alloc_min + retained.residence_time)

    fast_gradient = np.asarray(jax.grad(fast_objective)(parameters))
    retained_gradient = np.asarray(jax.grad(retained_objective)(parameters))
    assert np.all(fast_gradient[:, :4] != 0.0)
    np.testing.assert_array_equal(fast_gradient[:, 4:], 0.0)
    np.testing.assert_array_equal(retained_gradient[:, :4], 0.0)
    np.testing.assert_array_equal(retained_gradient[:, 4:], 1.0)


def test_runtime_retained_parameters_replace_only_named_active_full_pft_rows() -> None:
    base = _stomate_parameters()
    parameters = _parameters()
    routed = jax.jit(
        lambda values: route_retained_stomate_parameters(
            base,
            values,
            np.asarray([True, True]),
            (0, 13),
        )
    )(parameters)

    expected_alloc = np.asarray(base.alloc_min).copy()
    expected_alloc[[0, 13]] = parameters[:, 4]
    expected_residence = np.asarray(base.residence_time).copy()
    expected_residence[[0, 13]] = parameters[:, 5]
    np.testing.assert_array_equal(routed.alloc_min, expected_alloc)
    np.testing.assert_array_equal(routed.residence_time, expected_residence)
    expected_vcmax25 = np.asarray(base.vcmax25).copy()
    expected_vcmax25[[0, 13]] = parameters[:, 0]
    np.testing.assert_array_equal(routed.vcmax25, expected_vcmax25)
    np.testing.assert_array_equal(routed.maint_resp_slope, base.maint_resp_slope)

    def objective(values):
        updated = route_retained_stomate_parameters(
            base,
            values,
            np.asarray([True, True]),
            (0, 13),
        )
        return updated.alloc_min[0] + updated.residence_time[13] + updated.vcmax25[13]

    gradient = np.asarray(jax.jit(jax.grad(objective))(parameters))
    expected_gradient = np.zeros_like(parameters)
    expected_gradient[0, 4] = 1.0
    expected_gradient[1, 5] = 1.0
    expected_gradient[1, 0] = 1.0
    np.testing.assert_array_equal(gradient, expected_gradient)


def test_inactive_runtime_pft_preserves_static_retained_parameters() -> None:
    base = _stomate_parameters()
    parameters = _parameters()
    parameters[1, 4:] = np.asarray([-1.0e6, -2.0e6])
    routed = route_retained_stomate_parameters(
        base,
        parameters,
        np.asarray([True, False]),
        (0, 13),
    )

    assert float(routed.alloc_min[0]) == parameters[0, 4]
    assert float(routed.residence_time[0]) == parameters[0, 5]
    assert float(routed.alloc_min[13]) == float(base.alloc_min[13])
    assert float(routed.residence_time[13]) == float(base.residence_time[13])
    assert float(routed.vcmax25[13]) == float(base.vcmax25[13])
