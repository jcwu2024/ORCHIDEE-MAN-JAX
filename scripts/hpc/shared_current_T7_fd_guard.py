"""Program-coordinate FD analysis and native combined state-AD ABI helpers.

No model, JAX runtime or saved artifact is imported or executed at import time.
The caller supplies the public objective/runtime and predeclared scalar samples.
Passing small method tests does not establish real-model forward or AD truth.
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
STEP_MULTIPLIERS = (1.0, 1 / 3, 1 / 9, 1 / 27, 1 / 81, 1 / 243)
COORDINATE_DOMAINS = {
    434: {'kind': 'fraction', 'lower_bound': 0.0, 'upper_bound': 1.0},
    2382: {'kind': 'nonnegative_stock', 'lower_bound': 0.0, 'upper_bound': None},
    2621: {'kind': 'nonnegative_stock', 'lower_bound': 0.0, 'upper_bound': None},
    200: {'kind': 'nonnegative_flow', 'lower_bound': 0.0, 'upper_bound': None},
}


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


def predeclared_fd_plan(coordinate: int | str, x: float, scale: float) -> dict[str, Any]:
    """Return a bounded coordinate FD plan without asserting physical reachability.

    Positive interior coordinates use a six-step central grid based on 10% of
    the nearest finite scalar-bound margin. A zero lower bound uses a positive
    one-sided grid based on 1/8 of the caller-declared scale. Coordinate 2621
    is explicitly computational-only: this plan does not perturb its source
    pools or claim that the resulting state is physically reachable.
    """
    _validate_target(coordinate, 1.0)
    value = _finite_scalar(x, 'x')
    declared_scale = _finite_scalar(scale, 'scale')
    if declared_scale <= 0:
        raise ValueError('scale must be positive')
    if coordinate == NATIVE_B1_DIRECTION:
        base = declared_scale * 0.125
        bounds = {'lower_bound': None, 'upper_bound': None}
        mode = 'central'
        domain_kind = 'native_B1_direction'
    else:
        bounds = COORDINATE_DOMAINS[coordinate]
        lower, upper = bounds['lower_bound'], bounds['upper_bound']
        if value < lower or (upper is not None and value > upper):
            raise ValueError('coordinate value lies outside its scalar domain')
        boundary_base = declared_scale * 0.125
        if upper is not None:
            boundary_base = min(boundary_base, (upper - lower) * 0.125)
        if lower is not None and value == lower:
            base, mode = boundary_base, 'right'
        elif upper is not None and value == upper:
            base, mode = boundary_base, 'left'
        else:
            margin = value - lower if upper is None else min(value - lower, upper - value)
            base, mode = margin * 0.1, 'central'
        domain_kind = bounds['kind']
    steps = tuple(base * multiplier for multiplier in STEP_MULTIPLIERS)
    if not all(math.isfinite(h) and h > 0 for h in steps):
        raise ValueError('declared scale/value does not produce finite positive steps')
    if any(a <= b for a, b in zip(steps, steps[1:])):
        raise ValueError('step grid is not strictly descending in floating-point representation')
    certificate = {
        'status': 'known',
        'lower_bound': bounds['lower_bound'],
        'upper_bound': bounds['upper_bound'],
        'raw_value': value,
        'source': f'scalar-coordinate-domain:{domain_kind}',
    }
    return {
        'coordinate': coordinate,
        'coordinate_name': NATIVE_B1_DIRECTION if coordinate == NATIVE_B1_DIRECTION else FIXED_COORDINATES[coordinate],
        'raw_value': value,
        'scale': declared_scale,
        'mode': mode,
        'steps': steps,
        'domain': {'kind': domain_kind, **bounds},
        'certificate': certificate,
        'computational_only': True,
        'physical_reachability': 'not_established',
        'native_direction_length': NATIVE_B1_LENGTH if coordinate == NATIVE_B1_DIRECTION else None,
        'native_direction_scaling': 1.0 if coordinate == NATIVE_B1_DIRECTION else None,
        'internal_branch_limits': 'fingerprints are diagnostic only; unexposed internal branches remain unknown',
    }


def make_combined_native(jax: Any, objective: Any) -> Any:
    """Build native state-only gradient plus directional JVP for a bound loss.

    ``objective`` follows ``RolloutLoss.bind_checkpointed_loss_with_aux``:
    ``(params, initial_continuous, initial_discrete, days, targets,
    loss_context) -> (scalar_loss, auxiliary)``. Parameters and other dynamic
    inputs are ordinary arguments, while differentiation is restricted to the
    complete ``initial_continuous`` array. The returned callable accepts a
    final ``state_tangent`` argument and returns ``((loss, auxiliary),
    full_state_gradient, directional_derivative)``. Callers may JIT this
    callable once and reuse the executable with equal-shaped dynamic inputs.
    """
    state_value_and_grad = jax.value_and_grad(objective, argnums=1, has_aux=True)

    def combined_native(params, initial_continuous, initial_discrete, days, targets, loss_context, state_tangent):
        (loss, auxiliary), state_gradient = state_value_and_grad(
            params,
            initial_continuous,
            initial_discrete,
            days,
            targets,
            loss_context,
        )

        def scalar_for_state(state):
            return objective(params, state, initial_discrete, days, targets, loss_context)[0]

        _, directional_derivative = jax.jvp(
            scalar_for_state,
            (initial_continuous,),
            (state_tangent,),
        )
        return (loss, auxiliary), state_gradient, directional_derivative

    return combined_native


make_combined = make_combined_native


def analyze_fd(
    target: int | str,
    *,
    certificate: dict[str, Any],
    raw_value: float,
    baseline_loss: float,
    baseline_branch: Any,
    steps: list[float] | tuple[float, ...],
    samples: dict[float, dict[str, dict[str, Any]]],
    mode: str = 'central',
    ad_value: float | None = None,
    state20_raw: float = float('nan'),
    state20_defined: bool = False,
    direction_scale: float = 1.0,
) -> dict[str, Any]:
    """Classify supplied FD scalar values; no forward function is accepted.

    ``samples[h]`` contains the required perturbed scalar loss records and may
    include a ``branch`` fingerprint. Fingerprints are diagnostic only: a
    changed known fingerprint is recorded, while missing/unexposed internal
    branch state remains an interpretation limit rather than a rejection.
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
        'branch_fingerprints': [],
        'branch_change_observed': False if _known_fingerprint(baseline_branch) else None,
        'branch_interpretation_limit': 'reported fingerprints do not establish all internal branch states; unexposed branches remain unknown and physical reachability is not established',
        'computational_only': target == 2621 or target == NATIVE_B1_DIRECTION,
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
    if mode not in {'central', 'right', 'left'}:
        result['reason'] = 'mode must be central, right, or left'
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
        if not isinstance(entry, dict):
            result['reason'] = f'missing sample at predeclared step {h}'
            return result

        def record_branch(side: str, sample: dict[str, Any]) -> None:
            fingerprint = sample.get('branch')
            result['branch_fingerprints'].append({'h': h, 'side': side, 'fingerprint': fingerprint})
            if _known_fingerprint(baseline_branch) and _known_fingerprint(fingerprint):
                if fingerprint != baseline_branch:
                    result['branch_change_observed'] = True
            elif result['branch_change_observed'] is False:
                result['branch_change_observed'] = None

        def sample(side: str) -> tuple[float, float] | None:
            selected = entry.get(side)
            if not isinstance(selected, dict):
                result['reason'] = f'missing {side} sample at predeclared step {h}'
                return None
            try:
                loss = _finite_scalar(selected.get('loss'), f'{side} loss at {h}')
            except ValueError as error:
                result['reason'] = str(error)
                return None
            if selected.get('hard_passed') is False:
                result.setdefault('forward_gate_failures', []).append(
                    {'h': h, 'side': side, 'finite_loss': loss})
                result['reason'] = f'{side} forward sample failed hard gate at h={h}'
                return None
            coordinate = x + h if side == 'plus' else x - h
            if not math.isfinite(coordinate):
                result['reason'] = f'{side} endpoint is nonfinite at h={h}'
                return None
            if coordinate == x:
                result['reason'] = f'{side} step is not representable at h={h}'
                return None
            if (lower is not None and coordinate < lower) or (upper is not None and coordinate > upper):
                result['reason'] = f'{side} endpoint outside certified domain at h={h}'
                return None
            record_branch(side, selected)
            return loss, coordinate

        if mode == 'right':
            plus_sample = sample('plus')
            if plus_sample is None:
                return result
            fp, _ = plus_sample
            numerator = fp - f0
            denominator = h
            magnitude = abs(fp) + abs(f0)
        elif mode == 'left':
            minus_sample = sample('minus')
            if minus_sample is None:
                return result
            fm, _ = minus_sample
            numerator = f0 - fm
            denominator = h
            magnitude = abs(f0) + abs(fm)
        else:
            plus_sample = sample('plus')
            minus_sample = sample('minus')
            if plus_sample is None or minus_sample is None:
                return result
            fp, _ = plus_sample
            fm, _ = minus_sample
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
    if mode in {'right', 'left'}:
        result['derivative_scope'] = f'{mode}-sided boundary diagnostic only'
        if ad_value is not None:
            try:
                ad = _finite_scalar(ad_value, 'ad_value')
            except ValueError as error:
                result['ad_status'] = 'invalid_nonfinite_or_nonscalar'
                result['reason'] = str(error)
                return result
            difference = abs(result['fd_value'] - ad)
            if not math.isfinite(difference):
                result['ad_status'] = 'inconclusive_nonfinite_comparison'
                return result
            result['ad_value'] = ad
            result['one_sided_ad_absolute_difference'] = difference
            result['ad_status'] = 'diagnostic_difference_only_no_two_sided_claim'
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
