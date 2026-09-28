"""Small shape/native-AD and no-executable guard checks, never the real model."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.probe_shared_neural_precision import module_signature
from scripts.hpc.probe_shared_neural_transpose_lower import (
    compiler_guard, cotangent_shapes, ir_summary, native_transpose,
)


def test_checkpointed_native_transpose_lowers_with_masks_without_compile():
    def neural(p, state):
        return {'value': p * state + jnp.sin(state)}, {'active': state > 0}
    neural = jax.jit(jax.checkpoint(neural))
    parameter = jax.ShapeDtypeStruct((3,), np.float32)
    state = jax.ShapeDtypeStruct((3,), np.float32)
    evidence = {'phase': 'shape_lowering', 'initialization_lookups': [], 'forbidden_lookups': []}
    with compiler_guard(module_signature, evidence):
        outputs = jax.eval_shape(neural, parameter, state)
        cotangent = cotangent_shapes(jax, outputs)
        assert cotangent[1]['active'].dtype == jax.dtypes.float0
        function = native_transpose(jax, neural)
        lowered = function.lower(parameter, state, cotangent)
        gradients = jax.eval_shape(function, parameter, state, cotangent)
        summary = ir_summary(lowered.compiler_ir('stablehlo'), module_signature)
    assert not evidence['initialization_lookups'] and not evidence['forbidden_lookups']
    assert [value.shape for value in gradients] == [(3,), (3,)]
    assert summary['main_signature']['result_nominal_bytes'] == 24
    assert summary['main_reachable_unique_functions'] >= 1
    assert summary['operations'] > 0
    evidence['phase'] = 'shape_lowering'
    with compiler_guard(module_signature, evidence), pytest.raises(RuntimeError, match='lookup forbidden'):
        lowered.compile()
    assert len(evidence['forbidden_lookups']) == 1


def test_initializer_guard_only_allows_small_identified_tables():
    evidence = {'phase': 'initialization', 'initialization_lookups': [], 'forbidden_lookups': []}
    shape = jax.ShapeDtypeStruct((4,), np.float32)
    lowered = jax.jit(lambda x: x + 1).lower(shape)
    with compiler_guard(module_signature, evidence):
        executable = lowered.compile()
    assert len(evidence['initialization_lookups']) == 1
    assert evidence['initialization_lookups'][0]['signature']['result_nominal_bytes'] == 16
    del executable
