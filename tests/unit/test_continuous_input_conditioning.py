from __future__ import annotations

import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.input_conditioning import standardize_continuous
from research.daily_coarse_graining.daily_operator.neural_architecture import NeuralInputNormalization
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _standardize
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
    StructuredGPPStateSlice,
    _normalized_state_slice,
)

jax.config.update("jax_enable_x64", True)


def test_default_preserves_legacy_values_and_old_constructor():
    x = jnp.array([0., -32., 1e10, jnp.nan, jnp.inf])
    np.testing.assert_array_equal(standardize_continuous(x, 1., 1e-12), _standardize(x, 1., 1e-12))
    norm = NeuralInputNormalization(*([np.zeros(2)] * 11))
    assert norm.continuous_tail_threshold is None
    assert pickle.loads(pickle.dumps(norm)).continuous_tail_threshold is None


def test_central_linear_signed_tail_and_boundary_derivatives():
    def f(x):
        return standardize_continuous(x, 0., 1., 16., dtype=jnp.float64)
    x = jnp.array([-1e12, -17., -16., -1., 0., 1., 16., 17., 1e12])
    y = jax.jit(f)(x)
    np.testing.assert_allclose(y, -f(-x), rtol=0., atol=0.)
    np.testing.assert_array_equal(y[2:7], x[2:7])
    expected = np.where(abs(x) <= 16, 1., 1. / (1. + np.maximum(abs(x) - 16., 0.)))
    derivative = jax.vmap(jax.grad(f))(x)
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)
    assert np.all(derivative > 0.)
    for edge in (-16., 16.):
        assert float(jax.grad(f)(edge - 1e-7)) == pytest.approx(1., abs=2e-7)
        assert float(jax.grad(f)(edge + 1e-7)) == pytest.approx(1., abs=2e-7)


def test_implicated_doc_pair_and_true_ad_agree():
    a, b = 0.04795535427446734, 0.04795534980054048
    def f(x):
        return standardize_continuous(x, 1.2828389639283212e-12, 7.960271159294281e-12, 16.)
    assert abs(float(f(a) - f(b))) < 1e-5
    forward, tangent = jax.jvp(f, (jnp.array(a),), (jnp.array(1.),))
    _, pullback = jax.vjp(f, jnp.array(a))
    reverse = pullback(jnp.ones_like(forward))[0]
    finite_difference = (float(f(a + 1e-4)) - float(f(a - 1e-4))) / 2e-4
    assert float(tangent) == pytest.approx(float(reverse), rel=1e-6)
    assert float(tangent) == pytest.approx(finite_difference, rel=1e-3)


def test_precast_extreme_and_nonfinite_mask_gradients():
    def f(x):
        return standardize_continuous(x, 0., 1e-30, 16.)
    x = jnp.array([1e30, -1e30, 0., jnp.nan, jnp.inf])
    y = jax.jit(f)(x)
    derivative = jax.grad(lambda v: jnp.sum(f(v)))(x)
    assert np.isfinite(y).all() and np.isfinite(derivative).all()
    np.testing.assert_array_equal(y[3:], 0.)
    np.testing.assert_array_equal(derivative[3:], 0.)


def test_structured_route_preserves_raw_masks_but_conditions_features():
    raw = jnp.array([0.04795535427446734, jnp.nan])
    inputs = SimpleNamespace(state=SimpleNamespace(other=raw),
                             state_defined=SimpleNamespace(other=jnp.array([True, False])))
    norm = NeuralInputNormalization(*([np.zeros(2)] * 11))._replace(
        continuous_scale=np.full(2, 7.960271159294281e-12), continuous_tail_threshold=16.)
    spec = StructuredGPPStateSlice(start=0, stop=2, shape=(2,))
    original, features, mask = _normalized_state_slice(inputs, norm, spec)
    np.testing.assert_array_equal(original, raw.astype(jnp.float32))
    np.testing.assert_array_equal(mask, [True, False])
    assert 16 < float(features[0]) < 50
    assert float(features[1]) == 0.


def test_short_recurrence_reverse_matches_jvp_and_finite_difference():
    def rollout(p):
        def day(x, _):
            y = 0.8 * x + p * standardize_continuous(x, 0., 1e-12, 16., dtype=jnp.float64)
            return y, y
        _, states = jax.lax.scan(day, jnp.array(.048), None, length=3)
        return jnp.sum(states ** 2)
    p = jnp.array(.001)
    reverse = jax.grad(rollout)(p)
    tangent = jax.jvp(rollout, (p,), (jnp.ones_like(p),))[1]
    fd = (rollout(p + 1e-7) - rollout(p - 1e-7)) / 2e-7
    np.testing.assert_allclose(reverse, tangent, rtol=1e-12)
    np.testing.assert_allclose(reverse, fd, rtol=1e-7)


def test_selected_tail_preserves_other_features_and_gradients_bitwise():
    mean = jnp.array([1e-12, .2345678912345, -3.])
    scale = jnp.array([8e-12, .000237654321234, 1e-3])
    mask = jnp.array([True, False, False])
    x = jnp.array([.048, 28.45678123456, 300.543211234])

    def candidate(value):
        return standardize_continuous(value, mean, scale, 16., coordinate_mask=mask)

    baseline = jax.jit(lambda value: _standardize(value, mean, scale))(x)
    output = jax.jit(candidate)(x)
    np.testing.assert_array_equal(output[1:], baseline[1:])
    np.testing.assert_array_equal(candidate(x)[1:], _standardize(x, mean, scale)[1:])
    assert 16. < float(output[0]) < 50.
    old_gradient = jax.grad(lambda v: jnp.sum(_standardize(v, mean, scale)))(x)
    new_gradient = jax.grad(lambda v: jnp.sum(candidate(v)))(x)
    np.testing.assert_array_equal(new_gradient[1:], old_gradient[1:])
    assert np.isfinite(new_gradient).all()


def test_masked_structured_slice_uses_global_coordinate_selection():
    raw = jnp.array([1., .048, 200., jnp.nan])
    inputs = SimpleNamespace(state=SimpleNamespace(other=raw),
                             state_defined=SimpleNamespace(other=jnp.array([True, True, True, False])))
    norm = NeuralInputNormalization(*([np.zeros(4)] * 11))._replace(
        continuous_scale=np.full(4, 8e-12), continuous_tail_threshold=16.,
        continuous_tail_mask=np.array([False, True, False, False]))
    _, features, _ = _normalized_state_slice(inputs, norm, StructuredGPPStateSlice(1, 4, (3,)))
    assert 16. < float(features[0]) < 50.
    assert float(features[1]) == float(_standardize(raw[2], 0., 8e-12))
    assert float(features[2]) == 0.
