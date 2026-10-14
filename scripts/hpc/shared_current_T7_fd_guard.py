"""Disconnected scalar finite-difference method guard for current T7.

This module does not import or execute the model, JAX, or any saved artifact.
It classifies caller-supplied scalar forward samples against an explicit domain
certificate and predeclared step grid. Passing these method fixtures does not
establish real-model forward or AD truth.
"""
from __future__ import annotations

import math
from typing import Any


FIXED_COORDINATES = {
    434: 'profil_froz_hydro_ns',
    2382: 'carbon_32l',
    2621: 'deepC_peat',
    200: 'canopy2ground',
}
NATIVE_B1_DIRECTION = 'native_B1_unit_3854'
NATIVE_B1_LENGTH = 3854
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-6


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a real scalar')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite')
    return result


def _validate_target(target: int | str, direction_scale: float) -> str:
    if target in FIXED_COORDINATES:
        if direction_scale != 1.0:
            raise ValueError('fixed-coordinate direction scale must remain 1')
        return FIXED_COORDINATES[target]
    if target == NATIVE_B1_DIRECTION:
        if direction_scale != 1.0:
            raise ValueError('native B1 unit-length direction must not be rescaled')
        return NATIVE_B1_DIRECTION
    if target == NATIVE_B1_LENGTH:
        raise ValueError('3854 is a direction length, not a flattened coordinate')
    raise ValueError('unknown fixed coordinate or native B1 direction')


def analyze_fd(
    target: int | str,
    *,
    certificate: dict[str, Any],
    raw_value: float,
    baseline_loss: float,
    baseline_branch: Any,
    branch_certificate: dict[str, Any],
    steps: list[float] | tuple[float, ...],
    samples: dict[float, dict[str, dict[str, Any]]],
    mode: str = 'central',
    ad_value: float | None = None,
    state20_raw: float = float('nan'),
    state20_defined: bool = False,
    direction_scale: float = 1.0,
) -> dict[str, Any]:
    """Classify supplied FD scalar values; no forward function is accepted.

    ``samples[h]`` contains ``plus`` and, for central mode, ``minus`` records
    with finite ``loss`` and an exactly comparable ``branch`` fingerprint.
    The explicit branch certificate must cover every branch of the certified
    local domain for an explicitly specified scalar formula. Public output
    masks and unknown real-model branch state cannot satisfy this contract.
    ``steps`` is an explicit descending grid. At least three consecutive
    smallest scales must form a stable plateau under the fixed acceptance
    tolerances before AD is consulted.
    """
    name = _validate_target(target, direction_scale)
    x = _finite_scalar(raw_value, 'raw_value')
    f0 = _finite_scalar(baseline_loss, 'baseline_loss')

    result: dict[str, Any] = {
        'target': target,
        'target_name': name,
        'mode': mode,
        'fd_status': 'inconclusive',
        'ad_status': 'not_compared',
        'slopes': [],
        'state20_status': 'pass' if state20_defined and _is_finite(state20_raw) else 'strict_nonfinite_or_undefined_failure',
        'state20_raw': state20_raw,
        'state20_defined': bool(state20_defined),
        'real_model_truth_established': False,
    }
    if not result['state20_defined'] or not _is_finite(state20_raw):
        result['strict_failure'] = 'state20_strict_nonfinite_or_undefined'

    if not isinstance(certificate, dict) or certificate.get('status') not in {'known', 'unsupported'}:
        result['reason'] = 'explicit known/unsupported domain certificate required'
        return result
    if certificate['status'] == 'unsupported':
        result['reason'] = 'physical domain is unsupported by the supplied certificate'
        return result
    if (
        not isinstance(branch_certificate, dict)
        or branch_certificate.get('status') != 'complete_for_certified_local_domain'
        or branch_certificate.get('scope') != 'certified_local_domain'
        or branch_certificate.get('evidence') != 'explicit_scalar_formula'
        or branch_certificate.get('covers_all_internal_branches') is not True
        or branch_certificate.get('source') in (None, '')
        or not _known_fingerprint(baseline_branch)
        or branch_certificate.get('fingerprint') != baseline_branch
    ):
        result['reason'] = 'complete explicit branch certificate/fingerprint required; unknown real-model branches are inconclusive'
        return result
    if certificate.get('source') in (None, ''):
        result['reason'] = 'domain certificate source is required'
        return result
    cert_raw = _finite_scalar(certificate.get('raw_value'), 'certificate.raw_value')
    if cert_raw != x:
        result['reason'] = 'raw value differs from domain certificate'
        return result
    lower = certificate.get('lower_bound')
    upper = certificate.get('upper_bound')
    lower = None if lower is None else _finite_scalar(lower, 'certificate.lower_bound')
    upper = None if upper is None else _finite_scalar(upper, 'certificate.upper_bound')
    if lower is not None and upper is not None and lower > upper:
        result['reason'] = 'invalid domain certificate bounds'
        return result
    if (lower is not None and x < lower) or (upper is not None and x > upper):
        result['reason'] = 'raw value lies outside certified domain'
        return result
    if mode not in {'central', 'right'}:
        result['reason'] = 'mode must be central or right'
        return result
    if mode == 'central' and certificate.get('kink', False):
        result['reason'] = 'central difference is invalid at certified kink'
        return result
    if not isinstance(steps, (list, tuple)) or len(steps) < 3:
        result['reason'] = 'at least three predeclared steps are required'
        return result
    try:
        hs = [_finite_scalar(h, 'step') for h in steps]
    except ValueError as error:
        result['reason'] = str(error)
        return result
    if any(h <= 0 for h in hs) or any(a <= b for a, b in zip(hs, hs[1:])):
        result['reason'] = 'steps must be positive, distinct, and strictly descending'
        return result

    slopes: list[float] = []
    for h in hs:
        entry = samples.get(h)
        if not isinstance(entry, dict) or 'plus' not in entry:
            result['reason'] = f'missing sample at predeclared step {h}'
            return result
        plus = entry['plus']
        if not isinstance(plus, dict):
            result['reason'] = f'invalid positive sample at step {h}'
            return result
        try:
            fp = _finite_scalar(plus.get('loss'), f'plus loss at {h}')
        except ValueError as error:
            result['reason'] = str(error)
            return result
        if not _known_fingerprint(plus.get('branch')) or plus['branch'] != baseline_branch:
            result['reason'] = f'branch fingerprint changed at positive endpoint for h={h}'
            return result
        x_plus = x + h
        if not math.isfinite(x_plus):
            result['reason'] = f'positive endpoint is nonfinite at h={h}'
            return result
        if x_plus == x:
            result['reason'] = f'positive step is not representable at h={h}'
            return result
        if lower is not None and x_plus < lower or upper is not None and x_plus > upper:
            result['reason'] = f'positive endpoint outside certified domain at h={h}'
            return result
        if mode == 'right':
            numerator = fp - f0
            denominator = h
            magnitude = abs(fp) + abs(f0)
        else:
            if 'minus' not in entry or not isinstance(entry['minus'], dict):
                result['reason'] = f'missing negative sample at central step {h}'
                return result
            minus = entry['minus']
            try:
                fm = _finite_scalar(minus.get('loss'), f'minus loss at {h}')
            except ValueError as error:
                result['reason'] = str(error)
                return result
            if not _known_fingerprint(minus.get('branch')) or minus['branch'] != baseline_branch:
                result['reason'] = f'branch fingerprint changed at negative endpoint for h={h}'
                return result
            x_minus = x - h
            if not math.isfinite(x_minus):
                result['reason'] = f'negative endpoint is nonfinite at h={h}'
                return result
            if x_minus == x:
                result['reason'] = f'negative step is not representable at h={h}'
                return result
            if lower is not None and x_minus < lower or upper is not None and x_minus > upper:
                result['reason'] = f'negative endpoint outside certified domain at h={h}'
                return result
            numerator = fp - fm
            denominator = 2.0 * h
            magnitude = abs(fp) + abs(fm)
        if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0 or not math.isfinite(magnitude):
            result['reason'] = f'FD quotient arithmetic is nonfinite at h={h}'
            return result
        if abs(numerator) <= math.ulp(1.0) * magnitude:
            result['reason'] = f'loss subtraction is cancellation-limited at h={h}'
            return result
        slope = numerator / denominator
        if not math.isfinite(slope):
            result['reason'] = f'FD slope is nonfinite at h={h}'
            return result
        slopes.append(slope)

    result['slopes'] = slopes
    # The three smallest scales must have two successive stable refinements.
    tail = slopes[-3:]
    refinements = [(abs(a - b), DEFAULT_ATOL + DEFAULT_RTOL * abs(b)) for a, b in zip(tail, tail[1:])]
    if any(not math.isfinite(delta) or not math.isfinite(limit) for delta, limit in refinements):
        result['reason'] = 'FD stability comparison arithmetic is nonfinite'
        return result
    stable = all(delta <= limit for delta, limit in refinements)
    if not stable:
        result['reason'] = 'no three-consecutive-scale FD stability plateau'
        return result
    result['fd_status'] = 'converged'
    result['fd_value'] = tail[-1]
    result['fd_plateau'] = tail
    if mode == 'right':
        result['derivative_scope'] = 'right-sided boundary diagnostic only'
        if ad_value is not None:
            result['ad_status'] = 'not_compared_right_derivative_only'
        return result
    if ad_value is not None:
        try:
            ad = _finite_scalar(ad_value, 'ad_value')
        except ValueError as error:
            result['ad_status'] = 'invalid_nonfinite_or_nonscalar'
            result['reason'] = str(error)
            return result
        result['ad_value'] = ad
        delta = abs(result['fd_value'] - ad)
        limit = DEFAULT_ATOL + DEFAULT_RTOL * abs(ad)
        if not math.isfinite(delta) or not math.isfinite(limit):
            result['ad_status'] = 'inconclusive_nonfinite_comparison'
            return result
        result['ad_status'] = 'match' if delta <= limit else 'mismatch'
    return result


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def _known_fingerprint(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in {'', 'unknown', 'none'}
