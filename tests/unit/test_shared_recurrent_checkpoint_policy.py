"""Check that the diagnostic changes checkpoint policy without changing AD."""
import jax
import jax.numpy as jnp
import pytest

from scripts.hpc.probe_shared_recurrent_checkpoint_policy import PolicyJax, matched_summary


def test_policy_proxy_preserves_value_and_both_gradients():
    def objective(module, weights, initial):
        def step(carry, day):
            next_state = jnp.tanh(carry @ weights + day)
            return next_state, next_state

        _, states = module.lax.scan(module.checkpoint(step), initial, jnp.ones((3, 2)))
        return jnp.sum(states**2)

    weights, initial = jnp.eye(2), jnp.array([0.2, -0.1])
    baseline = jax.value_and_grad(lambda w, x: objective(jax, w, x), argnums=(0, 1))(weights, initial)
    policy = PolicyJax(jax, jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    candidate = jax.value_and_grad(lambda w, x: objective(policy, w, x), argnums=(0, 1))(weights, initial)
    assert jnp.allclose(baseline[0], candidate[0])
    assert all(jnp.allclose(a, b) for a, b in zip(baseline[1], candidate[1]))


def test_matched_summary_requires_three_pairs():
    assert matched_summary([2, 2.1, 1.9], [1, 1.1, 0.9])["speed_ratio"] == pytest.approx(2)
    with pytest.raises(ValueError):
        matched_summary([1, 2], [1, 2])
