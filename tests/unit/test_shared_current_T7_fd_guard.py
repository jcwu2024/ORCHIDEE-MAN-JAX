"""Synthetic scalar fixtures only; they do not establish real-model AD truth."""
import math

import pytest

from scripts.hpc import shared_current_T7_fd_guard as guard


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


def branch_certificate(fingerprint='same', *, status='complete_for_certified_local_domain'):
    return {
        'status': status,
        'scope': 'certified_local_domain',
        'evidence': 'explicit_scalar_formula',
        'covers_all_internal_branches': True,
        'fingerprint': fingerprint,
        'source': 'explicit synthetic test formula',
    }


def samples_for(function, x, steps=STEPS, *, mode='central', branch='same'):
    result = {}
    for h in steps:
        entry = {'plus': {'loss': function(x + h), 'branch': branch}}
        if mode == 'central':
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
        branch_certificate=branch_certificate(),
        steps=STEPS,
        samples=samples_for(lambda x: x * x + 3 * x, 2.0),
        state20_raw=0.0,
        state20_defined=True,
    )
    defaults.update(kwargs)
    return guard.analyze_fd(**defaults)


def test_positive_polynomial_fd_converges_before_ad_comparison():
    result = analyze(ad_value=7.0)
    assert result['fd_status'] == 'converged'
    assert result['fd_value'] == pytest.approx(7.0)
    assert result['ad_status'] == 'match'
    assert result['real_model_truth_established'] is False


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
    assert result['ad_status'] == 'not_compared_right_derivative_only'


def test_branch_fingerprint_change_rejects_sample_grid():
    samples = samples_for(lambda x: x * x + 3 * x, 2.0)
    samples[0.05]['minus']['branch'] = 'changed'
    result = analyze(samples=samples)
    assert result['fd_status'] == 'inconclusive'
    assert 'branch fingerprint changed' in result['reason']


@pytest.mark.parametrize('baseline,endpoint,cert', [
    (None, None, branch_certificate(fingerprint=None)),
    ('unknown', 'unknown', branch_certificate(fingerprint='unknown')),
    ('same', None, branch_certificate()),
    ('same', 'same', {**branch_certificate(), 'status': 'unknown'}),
    ('same', 'same', {**branch_certificate(), 'evidence': 'output_mask_fingerprint'}),
])
def test_unknown_or_incomplete_branch_evidence_is_inconclusive(baseline, endpoint, cert):
    samples = samples_for(lambda x: x * x + 3 * x, 2.0)
    samples[0.4]['plus']['branch'] = endpoint
    result = analyze(baseline_branch=baseline, branch_certificate=cert, samples=samples)
    assert result['fd_status'] == 'inconclusive'
    assert 'branch' in result['reason']


def test_nonfinite_forward_sample_rejects_grid_and_keeps_state20_failure():
    samples = samples_for(lambda x: x * x + 3 * x, 2.0)
    samples[0.1]['plus']['loss'] = math.nan
    result = analyze(samples=samples, state20_raw=math.nan, state20_defined=False)
    assert result['fd_status'] == 'inconclusive'
    assert 'must be finite' in result['reason']
    assert result['state20_raw'] is math.nan or math.isnan(result['state20_raw'])
    assert result['state20_status'] == 'strict_nonfinite_or_undefined_failure'
    assert result['strict_failure'] == 'state20_strict_nonfinite_or_undefined'


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
