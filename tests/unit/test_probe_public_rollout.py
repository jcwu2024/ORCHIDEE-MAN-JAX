from types import SimpleNamespace

import jax
import numpy as np

from scripts.hpc.probe_public_rollout import gradient_comparison, physical_checks
from tests.unit.test_daily_operator_successor_process_head import _markov_contract


def test_coordinate_error_cannot_hide_in_large_leaf_norm():
    expected = np.full(1000,1000.)
    expected[0] = 0.
    actual = expected.copy()
    actual[0] = 1e-3
    result = gradient_comparison(jax, {'w':actual}, {'w':expected})
    assert result['error_l2'] < 1e-6+1e-4*result['reference_l2']
    assert not result['passed']
    assert result['failed_coordinates'] == 1
    assert gradient_comparison(jax, {'w':expected}, {'w':expected})['passed']


def test_physical_gate_catches_teacher_defined_nan_and_negative_stock():
    contract = _markov_contract()
    width = contract.continuous_state_width
    state = np.zeros((1,width))
    valid = np.ones_like(state,dtype=bool)
    leaf = next(v for v in contract.state_leaves
                if v.key=='slowproc_stomate_previous_step_state.biomass')
    state[0,leaf.start:leaf.stop] = -1.
    gsmean = next(v for v in contract.state_leaves if v.key=='diffuco_previous_step_state.gsmean')
    state[0,gsmean.start] = np.nan
    observed_defined = valid.copy()
    observed_defined[0,gsmean.start] = False
    obs = {'state':state, 'defined':observed_defined, 'gpp':np.zeros((1,1,2)),
           'gpp_defined':np.ones((1,1,2),dtype=bool), 'active':np.ones((1,2),dtype=bool),
           'fraction':np.array([[.25,.75]]), 'budgets':np.zeros((1,2)),
           'discrete':{'flag':np.array([0])}, 'learned_output_nonfinite':np.array([1])}
    targets = {'defined':valid}
    templates = SimpleNamespace(pft=SimpleNamespace(pft_traits=np.zeros((1,2,1))))
    scales = SimpleNamespace(state_declared_dynamic=SimpleNamespace(other=np.ones(width,dtype=bool)))
    hard = physical_checks(obs, targets, obs['discrete'], templates, contract, scales)
    assert hard['state_nonfinite'] == 1
    assert hard['mask_mismatch'] == 0
    assert hard['stock_negative'] > 0
    assert hard['learned_output_nonfinite'] == 1
