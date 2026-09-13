from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from scripts.hpc.probe_feedback_loss_calibration import _make_batch_objective, _make_evaluator


def test_dynamic_coefficients_select_component_gradients_without_retracing_definition():
    base_weights = {
        "independent_daily_flux_mse": 1.0,
        "one_day_state_anchor": 0.1,
        "necessary_process_supervision": 0.01,
    }

    def objective(parameters, sample):
        prediction = parameters["slope"] * sample + parameters["offset"]
        auxiliary = {
            "flux_loss": (prediction - 1.0) ** 2,
            "state_loss": (prediction + 2.0) ** 2,
            "process_loss": prediction**2,
            "canopy_stock_change": {"loss": (2.0 * prediction - 3.0) ** 2},
            "litterpart_feedback": {"loss": (prediction + 4.0) ** 2},
            "hard": {"synthetic": jnp.asarray(0, dtype=jnp.int32)},
            "physical": {},
        }
        return prediction * 0.0, auxiliary

    parameters = {
        "slope": jnp.asarray(0.25, dtype=jnp.float32),
        "offset": jnp.asarray(-0.5, dtype=jnp.float32),
    }
    samples = jnp.asarray([-2.0, -0.5, 1.0, 3.0], dtype=jnp.float32)
    evaluator = _make_evaluator(jax, jnp, objective, base_weights)

    def terms(current_parameters):
        def one(sample):
            _, auxiliary = objective(current_parameters, sample)
            return jnp.stack(
                (
                    auxiliary["flux_loss"] + 0.1 * auxiliary["state_loss"]
                    + 0.01 * auxiliary["process_loss"],
                    auxiliary["canopy_stock_change"]["loss"],
                    auxiliary["litterpart_feedback"]["loss"],
                )
            )

        return jnp.mean(jax.vmap(one)(samples), axis=0)

    observed = []
    for direction in np.eye(3, dtype=np.float32):
        coefficients = jnp.asarray(direction)
        (value_and_auxiliary, gradient) = evaluator(parameters, samples, coefficients)
        expected_value, expected_gradient = jax.value_and_grad(
            lambda current: jnp.vdot(coefficients, terms(current))
        )(parameters)
        np.testing.assert_allclose(value_and_auxiliary[0], expected_value)
        for name in parameters:
            np.testing.assert_allclose(gradient[name], expected_gradient[name])
        observed.append(tuple(float(gradient[name]) for name in sorted(parameters)))

    assert len(set(observed)) == 3

    compact_value, compact_auxiliary = _make_batch_objective(
        jax, jnp, objective, base_weights
    )(parameters, samples, jnp.asarray([1.0, 3e-9, 2.5e-5]))
    monitor_value, monitor_auxiliary = _make_batch_objective(
        jax, jnp, objective, base_weights, include_base_components=True
    )(parameters, samples, jnp.asarray([1.0, 3e-9, 2.5e-5]))
    np.testing.assert_allclose(monitor_value, compact_value)
    np.testing.assert_allclose(monitor_auxiliary["terms"], compact_auxiliary["terms"])
    assert not {"flux_loss", "state_loss", "process_loss"} & set(compact_auxiliary)
    assert {"flux_loss", "state_loss", "process_loss"} <= set(monitor_auxiliary)
