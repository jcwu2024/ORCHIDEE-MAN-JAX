"""Tiny replay/AD diagnostics; no actual-model admission."""
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.probe_shared_union_integrated import forbid_graph_lookup, repeat_row0, tree_summary
from scripts.hpc.shared_daily_microbatch import make_microbatch_objective


def test_repeat_entire_dynamic_sample_preserves_row0_loss_and_parameter_gradient():
    Sample = namedtuple('RepeatSample', 'inputs day_inputs targets defined weights')
    original = Sample(jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
                      jnp.arange(4, dtype=jnp.float32), jnp.ones((4, 2)),
                      jnp.array([[True, False]] * 4), jnp.arange(4, dtype=jnp.float32) + 1)
    repeated = repeat_row0(jax, original)
    for actual, old in zip(repeated, original):
        assert actual.shape == old.shape and actual.dtype == old.dtype
        for row in actual:
            np.testing.assert_array_equal(row, old[0])
    def objective(p, sample):
        value = jnp.sum(jnp.where(sample.defined, (p * sample.inputs - sample.targets) ** 2, 0))
        return sample.weights * value + p * sample.day_inputs, {'hard': {'bad': jnp.array(0)}}
    batch = jax.jit(jax.value_and_grad(make_microbatch_objective(jax, jnp, objective), has_aux=True))
    single = jax.value_and_grad(objective, has_aux=True)
    actual = batch(jnp.float32(0.7), repeated)
    expected = single(jnp.float32(0.7), jax.tree_util.tree_map(lambda value: value[0], original))
    np.testing.assert_allclose(actual[0][0], expected[0][0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match='four leading'):
        repeat_row0(jax, np.ones((3, 2)))


def test_saved_reference_guard_rejects_graph_lookup_and_restores():
    from jax._src import compiler
    original = compiler.compile_or_get_cached
    with forbid_graph_lookup(jax), pytest.raises(RuntimeError, match='new graph'):
        compiler.compile_or_get_cached()
    assert compiler.compile_or_get_cached is original


def test_tree_summary_reports_actual_finite_and_nonzero_gradients():
    result = tree_summary(jax, {'gradient': np.array([0, 3, 4], dtype=np.float32)})
    assert result['coordinates'] == 3 and result['nonzero'] == 2 and result['norm'] == 5
    assert tree_summary(jax, np.array([np.nan]))['nonfinite'] == 1


def test_saved_serialized_tree_and_execution_use_supplied_dynamic_rows():
    from jax.experimental.serialize_executable import deserialize_and_load, serialize
    p = jnp.asarray(0.7, jnp.float32)
    samples = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    executable = jax.jit(lambda weights, rows: jnp.mean(jnp.sum(weights * rows, axis=1))).lower(p, samples).compile()
    serialized = serialize(executable)
    repeated = repeat_row0(jax, samples)
    assert serialized[1] == jax.tree_util.tree_structure(((p, repeated), {}))
    with forbid_graph_lookup(jax):
        restored = deserialize_and_load(*serialized)
        result = restored(p, repeated)
    np.testing.assert_allclose(result, np.sum(np.asarray(p) * np.asarray(samples[0])))
