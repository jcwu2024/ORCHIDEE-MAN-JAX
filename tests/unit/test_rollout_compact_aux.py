from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
from tests.unit.test_rollout_loss import assert_tree_close, case, toy_loss, toy_step

jax.config.update("jax_enable_x64", True)


def loss_with_compact_aux(params, initial, observations, targets, context):
    loss = toy_loss(params, initial, observations, targets, context)
    aux = {
        "nonfinite_diagnostic": jnp.sqrt(-jnp.sum(observations["state"][-1] ** 2)),
        "terminal_state_sum": jnp.sum(observations["state"][-1]),
        "valid_count": jnp.sum(observations["active"] & observations["defined"]),
    }
    return loss, aux


@pytest.mark.parametrize("horizon", [1, 2, 3, 7])
def test_checkpointed_compact_aux_matches_scalar_loss_and_gradients(horizon):
    arguments = case(jnp.float64, horizon)
    component = make_rollout_loss(jax, jnp, toy_step, toy_loss)
    compact_loss = component.bind_checkpointed_loss_with_aux(loss_with_compact_aux)

    (actual_value, auxiliary), actual_gradient = jax.value_and_grad(
        compact_loss, argnums=(0, 1), has_aux=True
    )(*arguments)
    expected_value, expected_gradient = jax.value_and_grad(
        component.checkpointed_loss, argnums=(0, 1)
    )(*arguments)

    np.testing.assert_allclose(actual_value, expected_value, rtol=0.0, atol=0.0)
    assert_tree_close(actual_gradient, expected_gradient, rtol=0.0, atol=0.0)
    assert bool(jnp.isnan(auxiliary["nonfinite_diagnostic"]))
    assert bool(jnp.isfinite(auxiliary["terminal_state_sum"]))
    assert auxiliary["valid_count"].dtype == jnp.dtype(jnp.int64)

    params, _, _, _, _, context = arguments
    np.testing.assert_allclose(
        actual_gradient[0]["regularizer_only"],
        2 * context["direct_parameter_weight"] * params["regularizer_only"],
        rtol=2e-14,
        atol=2e-15,
    )


def test_bound_compact_aux_keeps_dynamic_inputs_and_builds_one_rollout(monkeypatch):
    params, initial, initial_discrete, days, targets, context = case(jnp.float64, 3)
    component = make_rollout_loss(jax, jnp, toy_step, toy_loss)
    rollout_calls = []
    original_rollout = component.rollout

    def counted_rollout(*args, **kwargs):
        rollout_calls.append(kwargs.get("checkpointed"))
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(component, "rollout", counted_rollout)
    compact_loss = component.bind_checkpointed_loss_with_aux(loss_with_compact_aux)
    value_and_grad = jax.jit(jax.value_and_grad(compact_loss, argnums=(0, 1), has_aux=True))

    baseline = value_and_grad(params, initial, initial_discrete, days, targets, context)
    assert rollout_calls == [True]

    changed_discrete = {
        "gate": jnp.logical_not(initial_discrete["gate"]),
        "phase": initial_discrete["phase"] + jnp.asarray(2, dtype=jnp.int32),
    }
    changed_days = dict(
        days,
        defined=jnp.logical_not(days["defined"]),
        observe=jnp.logical_not(days["observe"]),
        phase_increment=days["phase_increment"] + jnp.asarray(1, dtype=jnp.int32),
        toggle=jnp.logical_not(days["toggle"]),
    )
    changed_targets = dict(targets, terminal=targets["terminal"] + 0.07)
    changed_context = dict(context, terminal_weight=context["terminal_weight"] * 1.4)
    changed = value_and_grad(
        params,
        initial,
        changed_discrete,
        changed_days,
        changed_targets,
        changed_context,
    )

    assert rollout_calls == [True]
    reference = make_rollout_loss(jax, jnp, toy_step, toy_loss)
    expected_changed = jax.value_and_grad(
        reference.checkpointed_loss, argnums=(0, 1)
    )(
        params,
        initial,
        changed_discrete,
        changed_days,
        changed_targets,
        changed_context,
    )
    np.testing.assert_allclose(changed[0][0], expected_changed[0], rtol=3e-14, atol=3e-15)
    assert_tree_close(changed[1], expected_changed[1], rtol=3e-14, atol=3e-15)
    assert not np.isclose(np.asarray(changed[0][0]), np.asarray(baseline[0][0]))
    assert not np.allclose(np.asarray(changed[1][1]), np.asarray(baseline[1][1]))
    assert bool(jnp.isnan(changed[0][1]["nonfinite_diagnostic"]))
