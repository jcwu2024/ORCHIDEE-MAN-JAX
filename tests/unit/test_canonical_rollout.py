from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator import canonical_rollout as bridge
from research.daily_coarse_graining.daily_operator.training_data import DailyProcessTrainingAdapter
from tests.unit.test_daily_operator_successor_process_head import _markov_contract, _synthetic_parent


def test_canonical_step_rebuilds_fraction_and_keeps_its_state_gradient(monkeypatch):
    contract = _markov_contract()
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    parent, values, defined = _synthetic_parent(adapter)
    sample = adapter.assemble(parent, supplemental_values=values, supplemental_defined=defined)
    inputs = sample.inputs
    leaf = next(x for x in contract.state_leaves if x.key == 'slowproc_stomate_previous_step_state.veget_max')
    state = jnp.asarray(inputs.continuous_state).at[leaf.start:leaf.stop].set(jnp.array([.25,.75]))
    parameters = {'all_neural_families': jnp.array(2.)}
    retained = {'marker': jnp.array(7)}

    def transition(p, current, day):
        assert set(p) == {'all_neural_families'}
        assert day is retained
        return SimpleNamespace(next_state=SimpleNamespace(other=current.continuous_state),
            next_discrete_state=current.discrete_state,
            next_state_defined=SimpleNamespace(other=jnp.ones_like(current.continuous_state, dtype=bool)),
            prediction=SimpleNamespace(named_outputs={'daily.fast_gpp': (p['all_neural_families']*current.pft.pft_fraction)[None,:]}),
            diagnostics={'water_budget_residual': jnp.array(0.), 'carbon_budget_residual': jnp.array(0.)})

    monkeypatch.setattr(bridge, 'bind_daily_operator_dynamic_retained_inputs', lambda **kw: transition)
    head = SimpleNamespace(contract=contract, input_assembler=adapter.input_assembler)
    step = bridge.make_canonical_step(head, object())

    def loss(s):
        _, _, obs = step(parameters, s, inputs.discrete_state, (inputs, retained))
        return jnp.sum(obs['gpp']*obs['fraction'][None,:])

    np.testing.assert_allclose(loss(state), 1.25)
    gradient = jax.grad(loss)(state)
    np.testing.assert_allclose(gradient[leaf.start:leaf.stop], [1.,3.])


def test_science_probe_loss_uses_dynamic_fraction_and_horizon():
    contract = _markov_contract()
    scales = SimpleNamespace(fluxes={name: 1. for name in bridge.FLUX_FIELDS},
                             stocks={name: 1. for name in bridge.STOCK_FIELDS})
    loss = bridge.make_science_probe_loss(contract, scales)
    states = jnp.zeros((2, contract.continuous_state_width))
    target = {'state': states, 'defined': jnp.ones_like(states, dtype=bool),
              'gpp': jnp.zeros((2,1,2)), 'fraction': jnp.array([[.25,.75],[.25,.75]]),
              'active': jnp.ones((2,2), dtype=bool)}
    predicted = dict(target, gpp=jnp.array([[[1.,2.]],[[1.,2.]]]))

    def f(fraction):
        return loss({}, states[0], dict(predicted, fraction=fraction), target, {'stock_weight': 1.})

    np.testing.assert_allclose(f(target['fraction']), 1.75**2/6)
    gradient = jax.grad(f)(target['fraction'])
    np.testing.assert_allclose(gradient, np.tile([1.75/6, 3.5/6], (2,1)), rtol=1e-6)
