from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    _attention,
    _linear_parameters,
)
from scripts.hpc.run_mixed_learning import lowered_precision_summary

jax.config.update('jax_enable_x64', True)


def _case():
    keys = jax.random.split(jax.random.PRNGKey(915), 6)
    parameters = {name: _linear_parameters(key, 32, 32)
                  for name, key in zip(('query', 'key', 'value', 'output'), keys)}
    q = jax.random.normal(keys[4], (2, 5, 32), dtype=jnp.float32)
    kv = jax.random.normal(keys[5], (2, 7, 32), dtype=jnp.float32)
    mask = jnp.ones((2, 5, 7), dtype=bool).at[:, 0, :].set(False).at[:, :, -1].set(False)
    return parameters, q, kv, mask


def test_explicit_fp32_removes_promoted_matrix_products():
    parameters, q, kv, mask = _case()
    legacy = jax.jit(lambda p, q, kv: _attention(p, q, kv, heads=4, mask=mask))
    candidate = jax.jit(lambda p, q, kv: _attention(p, q, kv, heads=4, mask=mask,
                                                  score_dtype=jnp.float32))
    old = lowered_precision_summary(legacy.lower(parameters, q, kv))
    new = lowered_precision_summary(candidate.lower(parameters, q, kv))
    assert old['matrix_operation_result_types']['f64'] == 1
    assert new['matrix_operation_result_types'].get('f64', 0) == 0
    assert new['matrix_operation_result_types']['f32'] == 6
    np.testing.assert_allclose(candidate(parameters, q, kv), legacy(parameters, q, kv),
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('score_dtype', [None, jnp.float32])
def test_masked_precision_path_has_finite_native_gradients_and_jvp(score_dtype):
    p, q, kv, mask = _case()

    def loss(p, q, kv):
        result = _attention(p, q, kv, heads=4, mask=mask, score_dtype=score_dtype)
        return jnp.mean(jnp.square(result))

    value, gradients = jax.value_and_grad(loss, argnums=(0, 1, 2))(p, q, kv)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(gradients))
    # Fully masked queries and masked key/value positions carry no input gradient.
    np.testing.assert_array_equal(gradients[1][:, 0, :], 0.)
    np.testing.assert_array_equal(gradients[2][:, -1, :], 0.)
    direction = jnp.ones_like(q) / 10
    _, tangent = jax.jvp(lambda q: loss(p, q, kv), (q,), (direction,))
    np.testing.assert_allclose(tangent, jnp.vdot(gradients[1], direction), rtol=1e-5, atol=1e-6)
