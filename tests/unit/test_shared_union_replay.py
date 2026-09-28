"""Tiny observer tests, not real-model replay acceptance."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import probe_shared_neural_precision as owner
from scripts.hpc.probe_shared_union_replay import observed_runtime, permitted_helper


def test_helper_admission_excludes_model_and_large_graphs():
    signature = {'result_nominal_bytes': 32, 'argument_nominal_bytes': 64}
    counts = {'builtin.module': 1, 'func.func': 1, 'stablehlo.add': 1, 'func.return': 1}
    assert permitted_helper('jit_add', counts, signature, initialization=False)
    assert not permitted_helper('jit_neural_stage', counts, signature, initialization=False)
    assert not permitted_helper('jit_other', counts | {'stablehlo.dot_general': 1}, signature, initialization=False)
    assert not permitted_helper('jit_add', {'stablehlo.add': 129}, signature, initialization=False)


def test_observer_preserves_native_AD_and_rejects_stage_backend():
    from jax._src import compiler
    def neural_stage(p, state):
        return {'tiny': jnp.exp(p * state)}, {'tiny': state > 0}
    def transition_stage(p, state, outputs):
        return outputs['tiny'] + p + state
    def loss_stage(p, state, result):
        return jnp.sum(result ** 2 + p), {'hard': {'bad': jnp.array(0)}}
    inputs = (jnp.array([0.3, 0.4]), jnp.array([0.5, 0.6]), jnp.asarray(1.0))
    # Populate process-local jit cache first, then observe the same identities.
    function = owner.native_staged_dispatcher(jax, tuple(jax.jit(f) for f in (neural_stage, transition_stage, loss_stage)))
    expected = jax.device_get(jax.block_until_ready(function(*inputs)))
    report = {'phase': 'warm_test', 'synchronize_dispatch': True}
    with observed_runtime(jax, owner, report, lambda *a, **kw: None,
                          SimpleNamespace(cache_seconds=5)):
        actual = jax.device_get(jax.block_until_ready(function(*inputs)))
        # Ready C++ jit fast paths bypass Python. A fresh tiny helper exercises
        # the positive hook without recompiling any differentiated model stage.
        fresh = jax.jit(lambda value: value + jnp.asarray(0.17, value.dtype))
        jax.block_until_ready(fresh(inputs[0]))
        backend_name = 'backend_compile' if hasattr(compiler, 'backend_compile') else 'backend_compile_and_load'
        with pytest.raises(RuntimeError, match='forbids model'):
            getattr(compiler, backend_name)()
    for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)
    assert report['dispatch_events']
    assert all(event['synchronization_changes_schedule'] for event in report['dispatch_events'])
