import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.state_loss_attribution import attribute_state_loss, state_field_names
from research.daily_coarse_graining.daily_operator.training_data import DailyProcessTrainingAdapter
from tests.unit.test_daily_operator_successor_process_head import _markov_contract, _synthetic_parent


def test_field_layout_matches_actual_canonical_assembler_groups():
    adapter = DailyProcessTrainingAdapter.from_contract(_markov_contract())
    parent, values, defined = _synthetic_parent(adapter)
    example = adapter.assemble(parent, supplemental_values=values, supplemental_defined=defined)
    state = adapter.input_assembler.assemble(example.inputs).state
    names = state_field_names(adapter.contract, adapter.input_assembler)
    for group in state._fields:
        assert len(names[group]) == np.asarray(getattr(state, group)).size
    assert any(name.endswith(':debt') for name in names['water'])


def test_attribution_reconstructs_bias_cross_terms_and_excludes_masked_outlier():
    residual = np.array([3., -1., 1e10])
    valid = np.array([True, True, False])
    diagnostic = {name: {'a': value} for name, value in {
        'predicted': residual, 'expected': np.zeros(3), 'scales': np.ones(3),
        'predicted_defined': np.ones(3, bool), 'expected_defined': np.ones(3, bool), 'eligible': valid,
    }.items()}
    diagnostic.update(group_loss={'a': 3.}, group_count={'a': 2})
    result = attribute_state_loss(diagnostic, {'a': ['first', 'second', 'masked']})
    assert result['reconstructed_state_total'] == 3.
    assert len(result['fields']) == 2
    assert result['groups']['a']['mse'] == 5.
    assert result['groups']['a']['signed_bias'] == 1.
    gradient = jax.grad(lambda x: .5 * (jnp.mean(x*x) + jnp.mean(x)**2))(jnp.array([3., -1.]))
    expected = {name: abs(float(value)) for name, value in zip(('first', 'second'), gradient, strict=True)}
    for row in result['fields']:
        np.testing.assert_allclose(row['state_output_gradient_norm'], expected[row['field']])
