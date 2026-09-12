from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss

jax.config.update("jax_enable_x64", True)


def toy_step(params, continuous, discrete, day):
    dtype = continuous.dtype
    phase = discrete["phase"]
    gate_offset = jnp.where(discrete["gate"], jnp.asarray(0.025, dtype), jnp.asarray(-0.015, dtype))
    fraction = jax.nn.sigmoid(params["fraction_gain"] * continuous + day["fraction_bias"])
    preactivation = (
        params["matrix"] @ continuous
        + params["bias"]
        + day["forcing"]
        + jnp.asarray(0.035, dtype) * phase.astype(dtype)
        + jnp.asarray(0.08, dtype) * fraction
        + gate_offset
    )
    following = jnp.tanh(preactivation)
    active = day["observe"] & (phase % 2 == 0)
    defined = day["defined"] & jnp.isfinite(following)
    observations = {
        "active": active,
        "defined": defined,
        "fraction": fraction,
        "flux": fraction * (following + params["output_scale"] * day["forcing"]),
        "gate": discrete["gate"],
        "phase": phase,
        "state": jnp.full_like(following, 999.0),  # The component must replace this.
    }
    next_discrete = {
        "gate": jnp.logical_xor(discrete["gate"], day["toggle"]),
        "phase": phase + day["phase_increment"],
    }
    return following, next_discrete, observations


def toy_loss(params, initial, observations, targets, context):
    valid = observations["active"] & observations["defined"]
    flux_error = jnp.where(valid, observations["flux"] - targets["flux"], 0.0)
    fraction_error = jnp.where(
        observations["defined"], observations["fraction"] - targets["fraction"], 0.0
    )
    valid_count = jnp.maximum(jnp.sum(valid), 1)
    defined_count = jnp.maximum(jnp.sum(observations["defined"]), 1)
    return (
        context["flux_weight"] * jnp.sum(flux_error**2) / valid_count
        + context["fraction_weight"] * jnp.sum(fraction_error**2) / defined_count
        + context["path_weight"] * jnp.mean(observations["state"] ** 2)
        + context["terminal_weight"] * jnp.sum((observations["state"][-1] - targets["terminal"]) ** 2)
        + context["direct_parameter_weight"] * jnp.sum(params["regularizer_only"] ** 2)
        + context["direct_initial_weight"] * jnp.sum((initial - targets["initial"]) ** 2)
    )


def case(dtype, horizon):
    dtype = jnp.dtype(dtype)
    index = jnp.arange(horizon, dtype=dtype)
    params = {
        "bias": jnp.asarray([0.02, -0.04], dtype=dtype),
        "fraction_gain": jnp.asarray([0.45, -0.30], dtype=dtype),
        "matrix": jnp.asarray([[0.30, -0.12], [0.08, 0.27]], dtype=dtype),
        "output_scale": jnp.asarray(0.15, dtype=dtype),
        "regularizer_only": jnp.asarray([0.11, -0.07], dtype=dtype),
    }
    initial = jnp.asarray([0.10, -0.20], dtype=dtype)
    initial_discrete = {"gate": jnp.asarray(False), "phase": jnp.asarray(1, dtype=jnp.int32)}
    days = {
        "defined": jnp.stack((index % 4 != 2, index % 5 != 3), axis=1),
        "forcing": jnp.stack((0.12 + 0.025 * index, -0.08 + 0.017 * index), axis=1),
        "fraction_bias": jnp.stack((-0.10 + 0.013 * index, 0.06 - 0.009 * index), axis=1),
        "observe": jnp.stack((index % 3 != 1, index % 4 != 0), axis=1),
        "phase_increment": (1 + (jnp.arange(horizon) % 2)).astype(jnp.int32),
        "toggle": jnp.arange(horizon) % 3 == 0,
    }
    targets = {
        "flux": jnp.stack((0.03 + 0.01 * index, -0.02 + 0.004 * index), axis=1),
        "fraction": jnp.stack((0.48 + 0.002 * index, 0.52 - 0.003 * index), axis=1),
        "initial": jnp.asarray([-0.03, 0.04], dtype=dtype),
        "terminal": jnp.asarray([0.08, -0.06], dtype=dtype),
    }
    context = {
        "direct_initial_weight": jnp.asarray(0.09, dtype=dtype),
        "direct_parameter_weight": jnp.asarray(0.12, dtype=dtype),
        "flux_weight": jnp.asarray(0.70, dtype=dtype),
        "fraction_weight": jnp.asarray(0.15, dtype=dtype),
        "path_weight": jnp.asarray(0.04, dtype=dtype),
        "terminal_weight": jnp.asarray(0.35, dtype=dtype),
    }
    return params, initial, initial_discrete, days, targets, context


def direct_observations(params, initial, initial_discrete, days):
    def transition(carry, day):
        continuous, discrete = carry
        following, next_discrete, observations = toy_step(params, continuous, discrete, day)
        observations = dict(observations, state=following)
        return (following, next_discrete), observations

    return jax.lax.scan(transition, (initial, initial_discrete), days)[1]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_public_rollout_returns_final_carry_and_component_owned_state(checkpointed):
    params, initial, initial_discrete, days, _, _ = case(jnp.float64, 3)
    component = make_rollout_loss(jax, jnp, toy_step, toy_loss)

    def transition(carry, day):
        continuous, discrete = carry
        following, next_discrete, observations = toy_step(params, continuous, discrete, day)
        return (following, next_discrete), dict(observations, state=following)

    expected_final, expected_observations = jax.lax.scan(
        transition, (initial, initial_discrete), days
    )
    final, observations = component.rollout(
        params, initial, initial_discrete, days, checkpointed=checkpointed
    )
    assert_tree_close(final, expected_final, rtol=0.0, atol=0.0)
    assert_tree_close(observations, expected_observations, rtol=0.0, atol=0.0)
    assert np.max(np.abs(np.asarray(observations["state"]))) < 1.0

    with pytest.raises(TypeError, match="one floating array"):
        component.rollout(params, {"state": initial}, initial_discrete, days)


def assert_tree_close(actual, expected, *, rtol, atol):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("horizon", [1, 2, 3, 7])
def test_streamed_matches_reference_and_checkpointed_whole_gradients(dtype, horizon):
    params, initial, initial_discrete, days, targets, context = case(dtype, horizon)
    component = make_rollout_loss(jax, jnp, toy_step, toy_loss)
    arguments = (initial_discrete, days, targets, context)

    direct_value, direct_gradient = jax.value_and_grad(component.loss, argnums=(0, 1))(
        params, initial, *arguments
    )
    checkpoint_value, checkpoint_gradient = jax.value_and_grad(
        component.checkpointed_loss, argnums=(0, 1)
    )(params, initial, *arguments)
    streamed_value, streamed_gradient = component.streamed_value_and_grad(
        params, initial, *arguments
    )

    rtol, atol = ((3e-10, 3e-11) if dtype == jnp.float64 else (3e-5, 3e-6))
    np.testing.assert_allclose(streamed_value, direct_value, rtol=rtol, atol=atol)
    np.testing.assert_allclose(checkpoint_value, direct_value, rtol=rtol, atol=atol)
    assert_tree_close(streamed_gradient, direct_gradient, rtol=rtol, atol=atol)
    assert_tree_close(checkpoint_gradient, direct_gradient, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        streamed_gradient[0]["regularizer_only"],
        2 * context["direct_parameter_weight"] * params["regularizer_only"],
        rtol=rtol,
        atol=atol,
    )
    assert component.reverse_compile_count == 1
    assert component.reverse_memory["estimated_peak_bytes"] > 0


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_reference_jvp_duality_and_finite_difference(dtype):
    params, initial, initial_discrete, days, targets, context = case(dtype, 3)
    component = make_rollout_loss(jax, jnp, toy_step, toy_loss)
    parameter_direction = jax.tree_util.tree_map(lambda value: jnp.full_like(value, 0.07), params)
    initial_direction = jnp.asarray([-0.05, 0.09], dtype=dtype)

    def objective(current_params, current_initial):
        return component.loss(
            current_params, current_initial, initial_discrete, days, targets, context
        )

    value, tangent = jax.jvp(objective, (params, initial), (parameter_direction, initial_direction))
    checkpoint_value, checkpoint_tangent = jax.jvp(
        lambda current_params, current_initial: component.checkpointed_loss(
            current_params, current_initial, initial_discrete, days, targets, context
        ),
        (params, initial),
        (parameter_direction, initial_direction),
    )
    streamed_value, (parameter_gradient, initial_gradient) = component.streamed_value_and_grad(
        params, initial, initial_discrete, days, targets, context
    )
    reverse_projection = sum(
        jnp.vdot(gradient, direction)
        for gradient, direction in zip(
            jax.tree_util.tree_leaves(parameter_gradient),
            jax.tree_util.tree_leaves(parameter_direction),
            strict=True,
        )
    ) + jnp.vdot(initial_gradient, initial_direction)

    rtol, atol = ((5e-10, 5e-11) if dtype == jnp.float64 else (5e-5, 5e-6))
    np.testing.assert_allclose(streamed_value, value, rtol=rtol, atol=atol)
    np.testing.assert_allclose(checkpoint_value, value, rtol=rtol, atol=atol)
    np.testing.assert_allclose(reverse_projection, tangent, rtol=rtol, atol=atol)
    np.testing.assert_allclose(checkpoint_tangent, tangent, rtol=rtol, atol=atol)

    epsilon = 1e-5 if dtype == jnp.float64 else 2e-3
    plus_params = jax.tree_util.tree_map(lambda value, direction: value + epsilon * direction, params, parameter_direction)
    minus_params = jax.tree_util.tree_map(lambda value, direction: value - epsilon * direction, params, parameter_direction)
    plus_initial = initial + epsilon * initial_direction
    minus_initial = initial - epsilon * initial_direction
    secant = (objective(plus_params, plus_initial) - objective(minus_params, minus_initial)) / (2 * epsilon)
    fd_rtol = 2e-7 if dtype == jnp.float64 else 3e-3
    np.testing.assert_allclose(secant, reverse_projection, rtol=fd_rtol, atol=fd_rtol * 0.1)


def test_dynamic_carry_fraction_discrete_masks_and_target_reuse():
    params, initial, initial_discrete, days, targets, context = case(jnp.float64, 7)
    observations = direct_observations(params, initial, initial_discrete, days)
    expected_phases = [1]
    for increment in np.asarray(days["phase_increment"][:-1]):
        expected_phases.append(expected_phases[-1] + int(increment))
    np.testing.assert_array_equal(observations["phase"], expected_phases)
    assert np.max(np.abs(np.asarray(observations["state"]))) < 1.0

    changed_days = dict(days, forcing=days["forcing"].at[0].add(jnp.asarray([0.3, -0.2])))
    changed_observations = direct_observations(params, initial, initial_discrete, changed_days)
    np.testing.assert_allclose(changed_observations["fraction"][0], observations["fraction"][0])
    assert not np.allclose(changed_observations["fraction"][1], observations["fraction"][1])

    valid = np.asarray(observations["active"] & observations["defined"])
    masked_targets = dict(
        targets,
        flux=jnp.where(jnp.asarray(valid), targets["flux"], targets["flux"] + 1000.0),
    )
    component = make_rollout_loss(jax, jnp, toy_step, toy_loss)
    baseline = component.streamed_value_and_grad(
        params, initial, initial_discrete, days, targets, context
    )
    masked = component.streamed_value_and_grad(
        params, initial, initial_discrete, days, masked_targets, context
    )
    np.testing.assert_allclose(masked[0], baseline[0], rtol=0.0, atol=0.0)
    assert_tree_close(masked[1], baseline[1], rtol=0.0, atol=0.0)
    assert component.reverse_compile_count == 1


def test_targets_context_and_days_are_dynamic_and_direct_initial_term_is_retained():
    params, initial, initial_discrete, days, targets, context = case(jnp.float64, 3)
    component = make_rollout_loss(jax, jnp, toy_step, toy_loss)
    without_direct_initial = dict(context, direct_initial_weight=jnp.asarray(0.0, dtype=jnp.float64))
    _, (_, initial_gradient_without) = component.streamed_value_and_grad(
        params, initial, initial_discrete, days, targets, without_direct_initial
    )
    _, (_, initial_gradient_with) = component.streamed_value_and_grad(
        params, initial, initial_discrete, days, targets, context
    )
    expected_direct = 2 * context["direct_initial_weight"] * (initial - targets["initial"])
    np.testing.assert_allclose(initial_gradient_with - initial_gradient_without, expected_direct, rtol=2e-10, atol=2e-11)

    changed_days = dict(days, forcing=days["forcing"] + jnp.asarray([0.015, -0.01]))
    changed_targets = dict(targets, flux=targets["flux"] + 0.025, terminal=targets["terminal"] - 0.01)
    changed_context = dict(context, flux_weight=context["flux_weight"] * 1.2)
    streamed_value, streamed_gradient = component.streamed_value_and_grad(
        params, initial, initial_discrete, changed_days, changed_targets, changed_context
    )
    direct_value, direct_gradient = jax.value_and_grad(component.loss, argnums=(0, 1))(
        params, initial, initial_discrete, changed_days, changed_targets, changed_context
    )
    np.testing.assert_allclose(streamed_value, direct_value, rtol=3e-10, atol=3e-11)
    assert_tree_close(streamed_gradient, direct_gradient, rtol=3e-10, atol=3e-11)
    assert component.reverse_compile_count == 1


def test_daily_scalar_float0_cotangent_keeps_array_type_for_aot_runtime():
    component = make_rollout_loss(jax,jnp,toy_step,toy_loss)
    cotangent = {'count':np.zeros((2,),dtype=jax.dtypes.float0),
                 'mask':np.zeros((2,3),dtype=jax.dtypes.float0),
                 'state':jnp.arange(6.).reshape(2,3)}
    day = component._slice(cotangent,1)
    assert isinstance(day['count'],np.ndarray)
    assert day['count'].shape == ()
    assert day['count'].dtype == jax.dtypes.float0
    assert day['mask'].shape == (3,)
    np.testing.assert_array_equal(day['state'],[3.,4.,5.])
