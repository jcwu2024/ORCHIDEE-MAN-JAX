"""Synthetic scalar fixtures only; they do not establish real-model AD truth."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.rollout_loss import RolloutLoss
from scripts.hpc import shared_current_T7_fd_guard as guard

jax.config.update('jax_enable_x64', True)


STEPS = (0.4, 0.2, 0.1, 0.05, 0.025)


def certificate(raw=2.0, *, lower=0.0, upper=None, status='known', kink=False):
    return {
        'status': status,
        'lower_bound': lower,
        'upper_bound': upper,
        'raw_value': raw,
        'source': 'synthetic test certificate',
        'kink': kink,
    }


def samples_for(function, x, steps=STEPS, *, mode='central', branch='same'):
    result = {}
    for h in steps:
        entry = {'plus': {'loss': function(x + h), 'branch': branch}}
        if mode in {'central', 'left'}:
            entry['minus'] = {'loss': function(x - h), 'branch': branch}
        result[h] = entry
    return result


def analyze(**kwargs):
    defaults = dict(
        target=2382,
        certificate=certificate(),
        raw_value=2.0,
        baseline_loss=10.0,
        baseline_branch='same',
        steps=STEPS,
        samples=samples_for(lambda x: x * x + 3 * x, 2.0),
        state20_raw=0.0,
        state20_defined=True,
    )
    defaults.update(kwargs)
    return guard.analyze_fd(**defaults)


def test_positive_polynomial_fd_converges_before_ad_comparison():
    function = lambda x: x * x + 3 * x
    plan = guard.predeclared_fd_plan(2382, 2.0, 1.0)
    result = analyze(
        certificate=plan['certificate'],
        steps=plan['steps'],
        samples=samples_for(function, 2.0, plan['steps']),
        ad_value=7.0,
    )
    assert result['fd_status'] == 'converged'
    assert result['fd_value'] == pytest.approx(7.0)
    assert result['ad_status'] == 'match'
    assert result['real_model_truth_established'] is False


def test_combined_native_state_gradient_and_jvp_reuse_tiny_checkpointed_rollout_abi():
    def step(params, state, discrete, day):
        following = params * state + day
        return following, discrete, {'state': following}

    def loss_with_aux(params, initial, observations, targets, context):
        residual = observations['state'] - targets
        loss = jnp.sum(residual**2) + context * jnp.sum(initial**2)
        auxiliary = {'trajectory_sum': jnp.sum(observations['state'])}
        return loss, auxiliary

    rollout = RolloutLoss(jax, jnp, step, lambda *_: jnp.asarray(0.0))
    objective = rollout.bind_checkpointed_loss_with_aux(loss_with_aux)
    combined = jax.jit(guard.make_combined(jax, objective))
    params = jnp.asarray(0.7, dtype=jnp.float64)
    initial = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    discrete = jnp.asarray(0, dtype=jnp.int32)
    days = jnp.asarray([[0.1, 0.2], [0.3, -0.2]], dtype=jnp.float64)
    targets = jnp.asarray([[0.0, 0.1], [0.2, -0.1]], dtype=jnp.float64)
    context = jnp.asarray(0.25, dtype=jnp.float64)
    tangent = jnp.asarray([1.0, -0.5], dtype=jnp.float64)
    args = (params, initial, discrete, days, targets, context, tangent)
    (loss, auxiliary), state_gradient, directional_derivative = combined(*args)
    initial_compile_count = combined._cache_size()
    reference_loss, reference_aux = objective(params, initial, discrete, days, targets, context)
    reference_gradient = jax.grad(lambda state: objective(params, state, discrete, days, targets, context)[0])(initial)
    _, reference_jvp = jax.jvp(
        lambda state: objective(params, state, discrete, days, targets, context)[0],
        (initial,),
        (tangent,),
    )
    np.testing.assert_allclose(loss, reference_loss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(auxiliary['trajectory_sum'], reference_aux['trajectory_sum'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state_gradient, reference_gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(directional_derivative, reference_jvp, rtol=1e-12, atol=1e-12)
    assert state_gradient.shape == initial.shape

    changed_params = params + jnp.asarray(0.1, dtype=params.dtype)
    changed_initial = initial + jnp.asarray([0.05, 0.03], dtype=initial.dtype)
    changed_discrete = jnp.asarray(1, dtype=discrete.dtype)
    changed_days = days + jnp.asarray([[0.02, -0.03], [0.04, 0.01]], dtype=days.dtype)
    changed_targets = targets + jnp.asarray([[0.01, 0.02], [-0.01, 0.03]], dtype=targets.dtype)
    changed_context = context * jnp.asarray(1.3, dtype=context.dtype)
    changed_tangent = jnp.asarray([-0.2, 0.8], dtype=initial.dtype)
    changed_args = (
        changed_params,
        changed_initial,
        changed_discrete,
        changed_days,
        changed_targets,
        changed_context,
        changed_tangent,
    )
    changed_output = combined(*changed_args)
    assert combined._cache_size() == initial_compile_count == 1
    (changed_loss, changed_aux), changed_gradient, changed_jvp = changed_output
    expected_changed_loss, changed_aux_reference = objective(
        changed_params, changed_initial, changed_discrete, changed_days, changed_targets, changed_context
    )
    expected_changed_gradient = jax.grad(
        lambda state: objective(
            changed_params, state, changed_discrete, changed_days, changed_targets, changed_context
        )[0]
    )(changed_initial)
    _, expected_changed_jvp = jax.jvp(
        lambda state: objective(
            changed_params, state, changed_discrete, changed_days, changed_targets, changed_context
        )[0],
        (changed_initial,),
        (changed_tangent,),
    )
    np.testing.assert_allclose(changed_loss, expected_changed_loss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(changed_aux['trajectory_sum'], changed_aux_reference['trajectory_sum'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(changed_gradient, expected_changed_gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(changed_jvp, expected_changed_jvp, rtol=1e-12, atol=1e-12)
    assert not np.allclose(np.asarray(directional_derivative), np.asarray(changed_jvp))


def test_predeclared_plans_cover_four_coordinates_and_keep_2621_computational_only():
    plans = {
        coordinate: guard.predeclared_fd_plan(coordinate, value, scale)
        for coordinate, value, scale in [
            (434, 0.4, 1.0),
            (2382, 2.0, 1.0),
            (2621, 2.0, 1.0),
            (200, 0.0, 0.4),
        ]
    }
    assert all(len(plan['steps']) == 6 for plan in plans.values())
    assert all(plan['computational_only'] for plan in plans.values())
    assert plans[434]['mode'] == plans[2382]['mode'] == plans[2621]['mode'] == 'central'
    assert plans[200]['mode'] == 'right'
    assert plans[2621]['domain']['kind'] == 'nonnegative_stock'
    assert plans[2621]['physical_reachability'] == 'not_established'
    assert plans[2621]['internal_branch_limits'].startswith('fingerprints are diagnostic')
    assert plans[2382]['steps'][0] < 2.0
    assert plans[2621]['steps'][0] < 2.0


def test_plan_uses_left_grid_at_fraction_upper_bound_and_keeps_native_b1_separate():
    upper = guard.predeclared_fd_plan(434, 1.0, 0.8)
    native = guard.predeclared_fd_plan(guard.NATIVE_B1_DIRECTION, 0.0, 0.8)
    assert upper['mode'] == 'left'
    assert upper['steps'][0] == pytest.approx(0.1)
    assert native['coordinate_name'] == 'native_B1_unit_3854'
    assert native['native_direction_length'] == 3854
    assert native['native_direction_scaling'] == 1.0
    assert native['steps'][0] == pytest.approx(0.1)


def test_unsupported_physical_domain_is_inconclusive():
    result = analyze(target=2621, certificate=certificate(status='unsupported'))
    assert result['fd_status'] == 'inconclusive'
    assert 'unsupported' in result['reason']


@pytest.mark.parametrize('target,kink', [(434, False), (200, True)])
def test_central_fd_rejects_lower_boundary_or_certified_kink(target, kink):
    cert = certificate(raw=0.0, lower=0.0, kink=kink)
    result = analyze(
        target=target,
        certificate=cert,
        raw_value=0.0,
        baseline_loss=0.0,
        samples=samples_for(lambda x: x + 1.0, 0.0),
    )
    assert result['fd_status'] == 'inconclusive'
    assert 'kink' in result['reason'] or 'outside certified domain' in result['reason']


def test_right_sided_boundary_result_does_not_claim_ad_equivalence():
    hs = (0.4, 0.2, 0.1, 0.05)
    samples = samples_for(lambda x: 2 * x, 0.0, hs, mode='right')
    result = analyze(
        target=434,
        certificate=certificate(raw=0.0),
        raw_value=0.0,
        baseline_loss=0.0,
        steps=hs,
        samples=samples,
        mode='right',
        ad_value=2.0,
    )
    assert result['fd_status'] == 'converged'
    assert result['derivative_scope'] == 'right-sided boundary diagnostic only'
    assert result['ad_status'] == 'diagnostic_difference_only_no_two_sided_claim'
    assert result['one_sided_ad_absolute_difference'] == pytest.approx(0.0)


def test_left_sided_upper_boundary_result_reports_only_one_sided_derivative():
    hs = (0.4, 0.2, 0.1, 0.05)
    result = analyze(
        target=434,
        certificate=certificate(raw=1.0, lower=0.0, upper=1.0),
        raw_value=1.0,
        baseline_loss=2.0,
        steps=hs,
        samples=samples_for(lambda x: 2 * x, 1.0, hs, mode='left'),
        mode='left',
        ad_value=2.0,
    )
    assert result['fd_status'] == 'converged'
    assert result['fd_value'] == pytest.approx(2.0)
    assert result['derivative_scope'] == 'left-sided boundary diagnostic only'
    assert result['ad_status'] == 'diagnostic_difference_only_no_two_sided_claim'


def test_branch_fingerprint_change_is_reported_without_automatic_rejection():
    samples = samples_for(lambda x: x * x + 3 * x, 2.0)
    samples[0.05]['minus']['branch'] = 'changed'
    result = analyze(samples=samples)
    assert result['fd_status'] == 'converged'
    assert result['branch_change_observed'] is True
    assert any(item['fingerprint'] == 'changed' for item in result['branch_fingerprints'])
    assert 'do not establish all internal branch states' in result['branch_interpretation_limit']


def test_missing_branch_fingerprints_allow_fd_but_limit_interpretation():
    samples = samples_for(lambda x: x * x + 3 * x, 2.0)
    samples[0.4]['plus'].pop('branch')
    result = analyze(baseline_branch=None, samples=samples)
    assert result['fd_status'] == 'converged'
    assert result['branch_change_observed'] is None
    assert result['branch_fingerprints'][0]['fingerprint'] is None
    assert 'unexposed branches remain unknown' in result['branch_interpretation_limit']


def test_nonfinite_forward_sample_rejects_grid_and_keeps_state20_failure():
    samples = samples_for(lambda x: x * x + 3 * x, 2.0)
    samples[0.1]['plus']['loss'] = math.nan
    result = analyze(samples=samples, state20_raw=math.nan, state20_defined=False)
    assert result['fd_status'] == 'inconclusive'
    assert 'must be finite' in result['reason']
    assert result['state20_raw'] is math.nan or math.isnan(result['state20_raw'])
    assert result['state20_status'] == 'strict_nonfinite_or_undefined_failure'
    assert result['strict_failure'] == 'state20_strict_nonfinite_or_undefined'


def test_hard_failed_forward_sample_is_inconclusive_and_retains_finite_loss():
    samples = samples_for(lambda x: x * x + 3 * x, 2.0)
    samples[0.4]['plus'].update(loss=17.0, hard_passed=False, AD_passed=True)
    result = analyze(samples=samples)
    assert result['fd_status'] == 'inconclusive'
    assert 'hard gate' in result['reason']
    assert result['forward_gate_failures'] == [
        {'h': 0.4, 'side': 'plus', 'finite_loss': 17.0}]


def test_no_three_scale_plateau_is_not_rescued_by_best_single_step():
    function = lambda x: x**3 + 4 * x
    samples = samples_for(function, 2.0)
    result = analyze(baseline_loss=function(2.0), samples=samples, ad_value=16.0)
    assert result['fd_status'] == 'inconclusive'
    assert result['ad_status'] == 'not_compared'
    assert 'no three-consecutive-scale' in result['reason']


def test_cancellation_limited_quotient_is_rejected():
    hs = (0.4, 0.2, 0.1)
    samples = {h: {'plus': {'loss': 1.0, 'branch': 'same'},
                   'minus': {'loss': 1.0, 'branch': 'same'}} for h in hs}
    result = analyze(steps=hs, samples=samples)
    assert result['fd_status'] == 'inconclusive'
    assert 'cancellation-limited' in result['reason']


def test_nonfinite_quotient_arithmetic_is_inconclusive():
    hs = (0.4, 0.2, 0.1)
    samples = {h: {'plus': {'loss': 1e308, 'branch': 'same'},
                   'minus': {'loss': -1e308, 'branch': 'same'}} for h in hs}
    result = analyze(steps=hs, samples=samples)
    assert result['fd_status'] == 'inconclusive'
    assert 'quotient arithmetic is nonfinite' in result['reason']


def test_nonfinite_denominator_and_slope_are_inconclusive():
    denominator_steps = (1e308, 8e307, 5e307)
    denominator_samples = {
        h: {'plus': {'loss': 1.0, 'branch': 'same'},
            'minus': {'loss': 0.0, 'branch': 'same'}}
        for h in denominator_steps
    }
    denominator_result = analyze(
        raw_value=0.0,
        certificate=certificate(raw=0.0, lower=None),
        baseline_loss=0.5,
        steps=denominator_steps,
        samples=denominator_samples,
    )
    assert denominator_result['fd_status'] == 'inconclusive'
    assert 'quotient arithmetic is nonfinite' in denominator_result['reason']

    slope_steps = (1e-308, 8e-309, 5e-309)
    slope_samples = {
        h: {'plus': {'loss': 1e308, 'branch': 'same'},
            'minus': {'loss': 0.0, 'branch': 'same'}}
        for h in slope_steps
    }
    slope_result = analyze(
        raw_value=0.0,
        certificate=certificate(raw=0.0, lower=None),
        baseline_loss=0.5,
        steps=slope_steps,
        samples=slope_samples,
    )
    assert slope_result['fd_status'] == 'inconclusive'
    assert 'FD slope is nonfinite' in slope_result['reason']


def test_unrepresentable_coordinate_steps_are_rejected():
    x = 1e300
    hs = (0.4, 0.2, 0.1)
    samples = {h: {'plus': {'loss': 2.0, 'branch': 'same'},
                   'minus': {'loss': 1.0, 'branch': 'same'}} for h in hs}
    result = analyze(
        raw_value=x,
        certificate=certificate(raw=x, lower=0.0),
        baseline_loss=1.5,
        steps=hs,
        samples=samples,
    )
    assert result['fd_status'] == 'inconclusive'
    assert 'not representable' in result['reason']


def test_acceptance_tolerances_cannot_be_overridden():
    with pytest.raises(TypeError):
        analyze(rtol=1e-4)
    with pytest.raises(TypeError):
        analyze(atol=1e-5)


def test_native_b1_is_unit_direction_and_dividing_by_four_cannot_hide_failure():
    x = 2.0
    function = lambda value: value * value + 4 * value
    result = analyze(
        target=guard.NATIVE_B1_DIRECTION,
        certificate=certificate(raw=x),
        raw_value=x,
        baseline_loss=function(x),
        samples=samples_for(function, x),
        ad_value=2.0,
    )
    assert result['target_name'] == 'native_B1_unit_3854'
    assert result['fd_status'] == 'converged'
    assert result['fd_value'] == pytest.approx(8.0)
    assert result['ad_status'] == 'mismatch'
    with pytest.raises(ValueError, match='must not be rescaled'):
        analyze(target=guard.NATIVE_B1_DIRECTION, direction_scale=4.0)
    with pytest.raises(ValueError, match='not a flattened coordinate'):
        analyze(target=guard.NATIVE_B1_LENGTH)
