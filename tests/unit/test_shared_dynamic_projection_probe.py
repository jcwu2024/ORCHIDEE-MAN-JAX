"""Strict FP32 native-VJP and interface regression for the bounded component."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from scripts.hpc.probe_shared_dynamic_projection import forward_vjp, projection_component, scalar_forward_grad


def test_component_preserves_repeated_empty_banks_and_nonzero_bias_gradients():
    key = jax.random.PRNGKey(147)
    keys = jax.random.split(key, 10)
    parameters = tuple({name: {'kernel': jax.random.normal(keys[2 * i + j], (8, 8), dtype=jnp.float32) * 0.1,
                               'bias': jnp.linspace(0.03, 0.09, 8, dtype=jnp.float32)}
                        for j, name in enumerate(('key', 'value'))} for i in range(2))
    tokens = (jax.random.normal(keys[5], (5, 8), dtype=jnp.float32),
              jax.random.normal(keys[6], (4, 8), dtype=jnp.float32))
    axis = tuple(np.asarray(x, dtype=np.int32) for x in ([0, 2, 4], [0, 1, 2], []))
    local = tuple(np.asarray(x, dtype=np.int32) for x in ([0, 2], [0, 1, 3], []))
    outputs = projection_component(parameters, tokens, axis, local, union=False)
    cotangent = jax.tree_util.tree_map(
        lambda value: jnp.sin(jnp.arange(value.size, dtype=value.dtype)).reshape(value.shape), outputs)
    reference = jax.jit(partial(forward_vjp, axis_indices=axis, local_indices=local, union=False))(parameters, tokens, cotangent)
    actual = jax.jit(partial(forward_vjp, axis_indices=axis, local_indices=local, union=True))(parameters, tokens, cotangent)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(reference), strict=True):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert np.linalg.norm(actual[1][0][0]['key']['kernel']) > 1e-4
    assert np.linalg.norm(actual[1][1][0]) > 1e-4
    values, structure = jax.tree_util.tree_flatten(cotangent)
    packed = jnp.concatenate(tuple(value.reshape(-1) for value in values))
    shapes = tuple(value.shape for value in values)
    for union, expected in ((False, reference), (True, actual)):
        scalar, gradients = jax.jit(partial(scalar_forward_grad, axis_indices=axis, local_indices=local,
                                           shapes=shapes, structure=structure, union=union))(parameters, tokens, packed)
        expected_scalar = sum(jnp.vdot(value, c) for value, c in
                              zip(jax.tree_util.tree_leaves(expected[0]), values, strict=True))
        np.testing.assert_allclose(scalar, expected_scalar, rtol=1e-5, atol=1e-6)
        for a, b in zip(jax.tree_util.tree_leaves(gradients), jax.tree_util.tree_leaves(expected[1]), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
