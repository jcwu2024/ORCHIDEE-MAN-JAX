import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
from scripts.hpc.probe_public_rollout_staged import StageStore, compare_stage, directional_stage, gradient_stage
from tests.unit.test_rollout_loss import case, toy_loss, toy_step

jax.config.update('jax_enable_x64',True)


def test_reference_survives_later_failure_and_is_reused_without_recompilation(tmp_path,monkeypatch):
    arguments = case(jnp.float64,2)
    component = make_rollout_loss(jax,jnp,toy_step,toy_loss)
    store = StageStore(tmp_path,{'case':'fixed-two-day'})
    def emit(*a,**k):
        pass
    reference = gradient_stage('reference',component,arguments,store,emit)
    assert (tmp_path/'reference.pkl').exists()

    def interrupted(*a,**kw):
        raise RuntimeError('simulated later-stage interruption')

    with monkeypatch.context() as patch:
        patch.setattr(component,'streamed_value_and_grad',interrupted)
        with pytest.raises(RuntimeError,match='interruption'):
            gradient_stage('streamed',component,arguments,store,emit)
        patch.setattr(component,'loss',interrupted)
        restored = gradient_stage('reference',component,arguments,store,emit)
    assert restored['value'] == reference['value']
    for a,b in zip(jax.tree_util.tree_leaves(restored['gradient']),jax.tree_util.tree_leaves(reference['gradient']),strict=True):
        np.testing.assert_array_equal(a,b)
    assert not (tmp_path/'streamed.pkl').exists()

    streamed = gradient_stage('streamed',component,arguments,store,emit)
    assert compare_stage(jax,streamed,restored)['passed']
    assert component.reverse_compile_count == 1
    with pytest.raises(ValueError,match='different probe case'):
        StageStore(tmp_path,{'case':'different'}).read('reference')


def test_saved_derivative_pairs_resume_without_repeating_loss_or_jvp(tmp_path,monkeypatch):
    arguments = case(jnp.float64,2)
    component = make_rollout_loss(jax,jnp,toy_step,toy_loss)
    store = StageStore(tmp_path,{'case':'directions'})
    events = []

    def emit(stage,**values):
        events.append(stage)

    gradient_stage('reference',component,arguments,store,emit)
    gradient_stage('streamed',component,arguments,store,emit)
    reference,streamed = store.read('reference'),store.read('streamed')
    result = directional_stage(component,arguments,reference,streamed,store,lambda obs:{},emit)
    assert result['passed']
    assert events.count('derivative_started') == 3

    def forbidden(*a,**kw):
        raise RuntimeError('a completed direction must not be evaluated again')

    monkeypatch.setattr(component,'loss',forbidden)
    restored = directional_stage(component,arguments,reference,streamed,store,lambda obs:{},emit)
    assert restored == result
    assert events.count('derivative_started') == 3
