from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Inputs(NamedTuple):
    continuous_state: object
    forcing: object
    target: object


class Samples(NamedTuple):
    inputs: Inputs


def test_dynamic_daily_samples_change_loss_and_gradients():
    import jax
    import jax.numpy as jnp

    from scripts.hpc.shared_daily_dual_ad import (
        make_daily_dual_objective,
        make_daily_dual_value_and_grad,
    )

    def daily_objective(parameters, samples):
        prediction = parameters * samples.inputs.continuous_state[:, 0] + samples.inputs.forcing
        error = prediction - samples.inputs.target
        return jnp.mean(error**2), {"mean_error": jnp.mean(error)}

    dual = make_daily_dual_objective(jax, daily_objective)
    backward = make_daily_dual_value_and_grad(jax, dual)
    states = jnp.zeros((4, 3854), dtype=jnp.float32).at[:, 0].set(
        jnp.asarray([0.3, -0.4, 0.2, 0.5], dtype=jnp.float32)
    )
    first = Samples(Inputs(
        states,
        jnp.asarray([0.0, 0.2, -0.1, 0.4], dtype=jnp.float32),
        jnp.asarray([0.1, 0.0, 0.2, -0.2], dtype=jnp.float32),
    ))
    second = first._replace(inputs=first.inputs._replace(
        forcing=jnp.asarray([0.8, -0.5, 0.7, -0.9], dtype=jnp.float32),
        target=jnp.asarray([-0.4, 0.6, -0.3, 0.8], dtype=jnp.float32),
    ))

    first_value, first_grads = backward(jnp.asarray(0.7, dtype=jnp.float32), states, first)
    second_value, second_grads = backward(jnp.asarray(0.7, dtype=jnp.float32), states, second)
    jax.block_until_ready((first_value, first_grads, second_value, second_grads))

    assert np.isfinite(float(first_value[0])) and np.isfinite(float(second_value[0]))
    assert not np.isclose(float(first_value[0]), float(second_value[0]))
    assert not np.isclose(float(first_grads[0]), float(second_grads[0]))
    assert first_grads[1].shape == second_grads[1].shape == (4, 3854)
    first_state_grad = np.asarray(first_grads[1])
    second_state_grad = np.asarray(second_grads[1])
    assert np.all(np.isfinite(first_state_grad)) and np.all(np.isfinite(second_state_grad))
    assert np.any(first_state_grad != 0) and np.any(second_state_grad != 0)
    assert not np.allclose(first_state_grad, second_state_grad)
