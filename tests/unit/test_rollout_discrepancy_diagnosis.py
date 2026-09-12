from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
from scripts.hpc.diagnose_public_rollout_discrepancy import (
    disagreement_directions,
    joint_jvp,
    project,
    replay_difference,
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
    a = {'state':np.array([1.,2.]),'mask':np.array([True,False]),'gpp':np.array([3.])}
    b = {'state':np.array([1.,2.1]),'mask':np.array([True,True]),'gpp':np.array([3.])}
    labels = [SimpleNamespace(start=0,stop=2,key='test_state')]
    same = replay_difference(jax,a,a,labels=labels)
    assert same['exact_observations']
    different = replay_difference(jax,a,b,labels=labels)
    assert different['nonfloating_mismatch'] == 1
    assert different['state_fields'][0]['field'] == 'test_state'
    np.testing.assert_allclose(different['state_fields'][0]['max_absolute'],.1)
