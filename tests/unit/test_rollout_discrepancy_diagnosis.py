from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
from scripts.hpc.diagnose_public_rollout_discrepancy import (
    disagreement_directions,
    joint_jvp,
    project,
    recover_saved_projections,
    replay_difference,
    rounded_tangent,
)
from tests.unit.test_rollout_loss import case, toy_loss, toy_step


def test_difference_directions_ignore_none_and_separate_state_from_parameters():
    parameters = {'w':jnp.array([1.,2.]),'unused':None}
    initial = jnp.array([3.,4.])
    reference = {'gradient':({'w':np.array([.1,.2]),'unused':None},np.array([1.,2.]))}
    streamed = {'gradient':({'w':np.array([.4,.6]),'unused':None},np.array([1.,4.]))}
    directions = disagreement_directions(jax,parameters,initial,reference,streamed)
    assert [v[0] for v in directions] == ['parameter_difference','state_difference']
    dp,ds = directions[0][1]
    assert dp['unused'] is None
    np.testing.assert_allclose(dp['w'],[.6,.8])
    np.testing.assert_array_equal(ds,[0.,0.])
    dp,ds = directions[1][1]
    np.testing.assert_array_equal(dp['w'],[0.,0.])
    np.testing.assert_array_equal(ds,[0.,1.])


def test_joint_jvp_matches_full_derivative_projection_on_both_inputs():
    component = make_rollout_loss(jax,jnp,toy_step,toy_loss)
    arguments = case(jnp.float64,2)
    params,state,discrete,days,targets,context = arguments
    _,gradient = jax.value_and_grad(component.loss,argnums=(0,1))(*arguments)
    dp = jax.tree_util.tree_map(lambda p:jnp.ones_like(p)*.05,params)
    ds = jnp.array([.2,-.3])
    result = joint_jvp(component)(params,state,dp,ds,discrete,days,targets,context)
    np.testing.assert_allclose(result,project(jax,gradient,(dp,ds)),rtol=1e-10,atol=1e-12)


def test_replay_observer_reports_real_state_and_discrete_differences():
    a = {'state':np.array([1.,2.]),'mask':np.array([True,False]),'gpp':np.array([3.]),'defined':np.array([True,True])}
    b = {'state':np.array([1.,2.1]),'mask':np.array([True,True]),'gpp':np.array([3.]),'defined':np.array([True,True])}
    labels = [SimpleNamespace(start=0,stop=2,key='test_state')]
    same = replay_difference(jax,a,a,labels=labels)
    assert same['exact_observations']
    different = replay_difference(jax,a,b,labels=labels)
    assert different['nonfloating_mismatch'] == 1
    assert different['state_fields'][0]['field'] == 'test_state'
    np.testing.assert_allclose(different['state_fields'][0]['max_absolute'],.1)


def test_fixed_undefined_primal_has_zero_rounded_tangent_without_hiding_bad_directions():
    state = jnp.array([1.,jnp.nan])
    direction = jnp.array([1.,0.])
    epsilon = 1e-4
    plus,minus = state+epsilon*direction,state-epsilon*direction
    raw = (plus-minus)/(2*epsilon)
    with pytest.raises(ValueError,match='finite'):
        project(jax,jnp.array([2.,0.]),raw)
    fixed = rounded_tangent(jax,jnp,plus,minus,direction,epsilon)
    np.testing.assert_allclose(fixed,[1.,0.],atol=1e-10)
    np.testing.assert_allclose(project(jax,jnp.array([2.,0.]),fixed),2.,atol=1e-10)
    def f(x):
        return jnp.sum(jnp.where(jnp.isfinite(x),x*x,0.))
    np.testing.assert_allclose(jax.jvp(f,(state,),(raw,))[1],jax.jvp(f,(state,),(fixed,))[1])
    bad = rounded_tangent(jax,jnp,plus,minus,jnp.array([1.,1.]),epsilon)
    with pytest.raises(ValueError,match='finite'):
        project(jax,jnp.array([2.,0.]),bad)


def test_equal_undefined_nan_is_not_a_replay_difference_but_defined_nan_is_reported():
    obs = {'state':np.array([1.,np.nan]),'defined':np.array([True,False])}
    labels = [SimpleNamespace(start=0,stop=2,key='test_state')]
    report = replay_difference(jax,obs,obs,labels=labels)
    assert report['exact_observations']
    assert report['nonfinite_defined'] == 0
    invalid = {**obs,'defined':np.array([True,True])}
    report = replay_difference(jax,invalid,invalid,labels=labels)
    assert not report['exact_observations']
    assert report['nonfinite_defined'] == 1


def test_recovery_uses_saved_scalar_results_and_only_recomputes_finite_projections():
    p = {'w':jnp.array([.3],dtype=jnp.float32),'unused':None}
    state = jnp.array([1.,jnp.nan],dtype=jnp.float64)
    ref = {'gradient':({'w':np.array([.4]),'unused':None},np.array([2.,0.]))}
    streamed = {'gradient':({'w':np.array([.401]),'unused':None},np.array([2.1,0.]))}
    prior = {'largest_state_coordinate':{'index':0},'directions':[
        {'name':name,'pairs':[{'epsilon':1e-4,'fd':fd,'jvp':fd,'reference_projection':'NaN'}]}
        for name,fd in [('parameter_difference',.4),('largest_state_coordinate',2.)]]}
    rows = recover_saved_projections(jax,jnp,(p,state),ref,streamed,prior,ref)
    for row in rows:
        pair = row['pairs'][0]
        assert np.isfinite(pair['reference_projection'])
        assert np.isfinite(pair['streamed_projection'])
        assert pair['checkpointed_projection'] == pair['reference_projection']
    assert rows[1]['pairs'][0]['jvp'] == 2.
    np.testing.assert_allclose(rows[1]['pairs'][0]['reference_projection'],2.,atol=1e-10)
    assert prior['directions'][0]['pairs'][0]['reference_projection'] == 'NaN'
