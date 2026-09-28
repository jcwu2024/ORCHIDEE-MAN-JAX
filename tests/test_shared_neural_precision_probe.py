"""Small diagnostic plumbing tests; no synthetic scientific acceptance."""
from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.probe_shared_neural_precision import (
    compare_results, module_signature, native_staged_dispatcher, neural_policy, parameter_groups, staged_daily_functions, state_groups, vector_metrics,
)


def test_scoped_policy_restores_even_on_error():
    old = lambda *args: None
    module = SimpleNamespace(_linear=old, _ffn=old)
    untouched = SimpleNamespace(_linear=old)
    with pytest.raises(RuntimeError), neural_policy([module], mixed=True):
        assert module._linear is not old
        assert untouched._linear is old
        raise RuntimeError('test')
    assert module._linear is old and module._ffn is old


def test_metrics_near_zero_underflow_and_nonfinite():
    zeros = np.zeros(3, dtype=np.float32)
    near = vector_metrics(jax, zeros, zeros)
    assert near['screening_passed'] and near['relative_l2'] is None
    lost = vector_metrics(jax, zeros, np.ones(3, dtype=np.float32))
    assert lost['new_zero_coordinates'] == 3 and not lost['screening_passed']
    bad = vector_metrics(jax, np.array([np.nan], dtype=np.float32), zeros[:1])
    assert bad['nonfinite'] == 1 and not bad['screening_passed']


def test_groups_keep_semantics_and_exclude_discrete():
    leaves = [SimpleNamespace(key='water', start=0, stop=2, discrete=False),
              SimpleNamespace(key='day', start=None, stop=None, discrete=True),
              SimpleNamespace(key='carbon', start=2, stop=3, discrete=False)]
    grouped = state_groups(np.arange(3), SimpleNamespace(state_leaves=leaves))
    assert list(grouped) == ['water', 'carbon']
    assert list(parameter_groups({'shared_transformer': {'backbone': 1}, 'encoder': 2})) == ['shared_transformer.backbone', 'encoder']


def test_trace_local_capture_and_dynamic_loss_multiplier():
    captured = {}
    def measured(p, state, multiplier):
        captured.clear()
        output = p * state
        captured['outputs'] = {'tiny': output}
        return jnp.sum(output ** 2) * multiplier, dict(captured)
    executable = jax.jit(jax.value_and_grad(measured, argnums=(0, 1), has_aux=True))
    p, state = jnp.ones(2), jnp.arange(2, dtype=jnp.float32)
    (loss, diagnostics), grads = executable(p, state, jnp.asarray(1.0))
    (scaled, _), scaled_grads = executable(p, state, jnp.asarray(8.0))
    np.testing.assert_allclose(loss, scaled / 8)
    for a, b in zip(grads, scaled_grads):
        np.testing.assert_allclose(a, b / 8)
    assert diagnostics['outputs']['tiny'].shape == (2,)


def test_existing_scalar_output_scale_and_structural_mask():
    inputs_type = namedtuple('Inputs', ['continuous_state'])
    sample_type = namedtuple('Sample', ['inputs'])
    leaf = SimpleNamespace(key='state', start=0, stop=2, discrete=False)
    head = SimpleNamespace(contract=SimpleNamespace(state_leaves=[leaf]),
        output_normalization=SimpleNamespace(scale={'tiny': np.array(2.0)}),
        persistent_coordinates=None,
        input_assembler=SimpleNamespace(assemble=lambda _: SimpleNamespace(
            state_defined=SimpleNamespace(other=np.array([True, False])))))
    context = {'head': head, 'daily4': sample_type(inputs_type(np.ones((1, 2))))}
    diagnostics = {'outputs': {'tiny': np.array([1.0, 2.0])},
        'raw_masks': {'tiny': np.array([True, True])},
        'auxiliary': {'hard': {'nonfinite': 0}, 'state_diagnostics': {
            'predicted_defined': np.array([True, False]),
            'expected_defined': np.array([True, False])}}}
    result = {'loss': np.array(1.0), 'diagnostics': diagnostics,
        'gradients': ({'shared_transformer': {'head': np.ones(2)}}, np.array([1.0, 0.0]))}
    comparison = compare_results(jax, result, result, context)
    assert comparison['screening_passed'] and comparison['state_diagnostic_masks_exact']
    assert comparison['comparisons']['outputs']['tiny']['normalized_max'] == 0


def test_native_stages_preserve_direct_and_neural_state_paths():
    def neural_stage(p, state):
        return {'tiny': jnp.exp(p * state)}, {'tiny': state > 0}
    def transition_stage(p, state, outputs):
        return outputs['tiny'] + state ** 2 + p * state
    def loss_stage(p, state, result):
        return jnp.sum(result ** 2 + 0.1 * state + p), {'hard': {'nonfinite': jnp.sum(~jnp.isfinite(result))}}
    stages = tuple(jax.jit(f) for f in (neural_stage, transition_stage, loss_stage))
    staged = native_staged_dispatcher(jax, stages)
    def monolithic(p, state, multiplier):
        outputs, masks = neural_stage(p, state)
        result = transition_stage(p, state, outputs)
        loss, auxiliary = loss_stage(p, state, result)
        return loss * multiplier, {'auxiliary': auxiliary, 'outputs': outputs, 'raw_masks': masks}
    direct = jax.jit(jax.value_and_grad(monolithic, argnums=(0, 1), has_aux=True))
    inputs = (jnp.array([0.3, 0.4]), jnp.array([0.5, 0.6]), jnp.asarray(1.0))
    left, right = staged(*inputs), direct(*inputs)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('neural_remat', [False, True])
def test_real_stage_extraction_proxy_and_factory_preserve_all_paths(monkeypatch, neural_remat):
    from research.daily_coarse_graining.daily_operator import model, shared_training
    inputs_type = namedtuple('StageInputs', ['continuous_state'])
    sample_type = namedtuple('StageSample', ['inputs', 'day_inputs'])
    captured, calls = {}, {'assemble_context': 0, 'post_update': 0}
    @dataclass(frozen=True)
    class Tail:
        day_inputs: object
    class Head:
        input_assembler = SimpleNamespace(assemble=lambda value: value.continuous_state)
        def assemble_context(self, p, state):
            calls['assemble_context'] += 1
            return p * state
        def learned_outputs(self, p, encoded, state):
            captured['raw_masks'] = {'tiny': state > 0}
            return {'tiny': jnp.exp(encoded)}
        def prediction_from_learned_outputs(self, outputs, state):
            return outputs['tiny'] + 0.2 * state
        def __call__(self, p, encoded, state):
            return self.prediction_from_learned_outputs(self.learned_outputs(p, encoded, state), state)
        def materialize_post_update(self, p, encoded, state, prediction, updated):
            # Same discarded context/parameter contract as the real shared owner.
            calls['post_update'] += 1
            return updated + 0.3 * state
    def fake_transition(p, inputs, *, input_assembler, process_head, retained_tail):
        state = input_assembler.assemble(inputs)
        encoded = process_head.assemble_context(p, state)
        prediction = process_head(p, encoded, state)
        updated = prediction + state ** 2 + retained_tail.day_inputs
        result = process_head.materialize_post_update(p, encoded, state, prediction, updated)
        return {'value': result, 'mask': state > 0}
    monkeypatch.setattr(model, 'daily_operator_transition', fake_transition)
    def make_objective(runtime, head, template, state_scales, science_scales, **kwargs):
        transition = shared_training.bind_daily_operator_dynamic_retained_inputs(
            input_assembler=head.input_assembler, process_head=head, retained_tail_template=template)
        def objective(p, row):
            result = transition(p, row.inputs, row.day_inputs)
            process = jnp.sum(result['value'] ** 2)
            # Direct parameter/state loss routes are deliberately nonzero.
            loss = process / 3 + jnp.sum(p + row.inputs.continuous_state) / 9
            return loss, {'hard': {'nonfinite': jnp.sum(~jnp.isfinite(result['value']))},
                          'state_diagnostics': result}
        return objective
    head = Head()
    sample = sample_type(inputs_type(jnp.array([0.5, 0.6])), jnp.array([0.1, 0.2]))
    context = {'runtime': {}, 'template': Tail(sample.day_inputs), 'state_scales': None, 'science_scales': None}
    stages = staged_daily_functions(jax, head, sample, context, make_objective, captured, Head.learned_outputs,
                                   neural_remat=neural_remat)
    function = native_staged_dispatcher(jax, stages)
    p = jnp.array([0.3, 0.4])
    actual = function(p, sample.inputs.continuous_state, jnp.asarray(1.0))
    assert calls == {'assemble_context': 1, 'post_update': 1}
    unsplit = make_objective({}, head, context['template'], None, None)
    def direct(p, state):
        return unsplit(p, sample._replace(inputs=inputs_type(state)))
    expected = jax.jit(jax.value_and_grad(direct, argnums=(0, 1), has_aux=True))(p, sample.inputs.continuous_state)
    np.testing.assert_allclose(actual[0][0], expected[0][0], rtol=1e-5, atol=1e-6)
    for a, b in zip(actual[1], expected[1]):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(actual[0][1]['auxiliary']['state_diagnostics']['mask'], expected[0][1]['state_diagnostics']['mask'])
    assert int(actual[0][1]['auxiliary']['hard']['nonfinite']) == 0


def test_recorded_module_result_signature_before_compilation():
    array = jnp.ones(2, dtype=jnp.float32)
    module = jax.jit(lambda value: (value, value > 0)).lower(array).compiler_ir()
    signature = module_signature(module)
    assert signature['argument_leaves'] == 1 and signature['result_leaves'] == 2
    assert signature['argument_nominal_bytes'] == 8 and signature['result_nominal_bytes'] == 10


def test_cache_setup_precedes_helper_or_model_import(monkeypatch):
    from scripts.hpc import probe_shared_neural_precision as probe
    order = []
    monkeypatch.setattr(probe, 'configure_stage_cache', lambda args: order.append('configure'))
    def stop_at_helper(path):
        order.append('helper')
        raise RuntimeError('stop before runtime')
    monkeypatch.setattr(probe, 'load_helper', stop_at_helper)
    with pytest.raises(RuntimeError, match='stop before runtime'):
        probe.run(SimpleNamespace(real_daily_probe_file=None))
    assert order == ['configure', 'helper']
