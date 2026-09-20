"""The shared mixed-training wrapper must not discard the optional loss."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.probe_feedback_loss_calibration import _make_batch_objective

WEIGHTS = dict(independent_daily_flux_mse=1., one_day_state_anchor=.1,
               necessary_process_supervision=.01)


def objective(enabled):
    def one(parameter, sample):
        value = parameter * sample
        auxiliary = dict(flux_loss=value**2, state_loss=value, process_loss=value,
                         canopy_stock_change={'loss': value**2},
                         litterpart_feedback={'loss': value}, hard={}, physical={})
        if enabled:
            auxiliary['litter_stock_change'] = dict(unweighted_loss=(value-3)**2,
                                                   total=.1*(value-3)**2)
        return 0., auxiliary
    return one


def test_optional_term_reaches_scalar_and_gradient_without_double_weight():
    samples = jnp.array([1., 2.])
    old = _make_batch_objective(jax, jnp, objective(False), WEIGHTS)
    new = _make_batch_objective(jax, jnp, objective(True), WEIGHTS)
    c3 = jnp.array([1., .1, .1])
    c4 = jnp.array([1., .1, .1, .1])
    for coeff in (c3, c4):
        def fn(p):
            return new(p, samples, coeff)[0]

        def expected(p):
            return old(p, samples, c3)[0] + .1*jnp.mean((p*samples-3)**2)
        np.testing.assert_allclose(fn(1.), expected(1.))
        np.testing.assert_allclose(jax.grad(fn)(1.), jax.grad(expected)(1.))
    def disabled(p):
        return new(p, samples, c4.at[3].set(0.))[0]

    def legacy(p):
        return old(p, samples, c3)[0]
    np.testing.assert_array_equal(disabled(1.), legacy(1.))
    np.testing.assert_array_equal(jax.grad(disabled)(1.), jax.grad(legacy)(1.))
    assert 'litter_stock_change' in new(1., samples, c4)[1]


def test_missing_fourth_term_is_rejected():
    batch = _make_batch_objective(jax, jnp, objective(False), WEIGHTS)
    with pytest.raises(ValueError, match='require litter'):
        batch(1., jnp.array([1.]), jnp.ones(4))
