"""Scale and tangent-guard regressions for Fortran turn 569-665.

The active class values are from audited actual artifact275/old-step1/row17
capture call41 (biomass-amplification-cpu-old-row17-first6); the scale law is
checked independently of the historical compiled network and its loss.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.stomate.carbon_kernels import turnover_leaf_age_fall

FRACTIONS = np.array([0.002545695295720209, 0.011313425835462453, 0.0434304139324322, 0.9427104649363851])
AGES = np.array([3080.6829585821606, 3082.6110769428087, 3085.793962047229, 3103.533185648637])


def actual_class_update(mass, fractions=jnp.asarray(FRACTIONS), ages=jnp.asarray(AGES)):
    biomass = jnp.zeros((1, 14, 12, 1), dtype=jnp.float64).at[0, 13, 0, 0].set(mass)
    age = jnp.zeros((1, 14, 4), dtype=jnp.float64).at[0, 13].set(ages)
    frac = jnp.zeros_like(age).at[0, 13].set(fractions)
    # Trees have a temperature-independent critical age of730 at this trace.
    return turnover_leaf_age_fall(
        biomass,
        jnp.zeros_like(biomass),
        age,
        frac,
        jnp.array([296.73314693372953]),
        is_tree=jnp.arange(14) != 0,
        natural=jnp.ones(14, bool),
        ok_laidev=jnp.zeros(14, bool),
        leafagecrit=jnp.full(14, 730.0, dtype=jnp.float64),
    )


def relative_reference(fractions, ages):
    rates = jnp.minimum(0.99, 1.0 / (730.0 * (730.0 / ages) ** 4))
    prefixes = jnp.concatenate((jnp.ones(1), jnp.cumprod(1.0 - fractions * rates)))
    losses = prefixes[:-1] * fractions * rates
    return (fractions - losses) / prefixes[-1], prefixes[-1]


def test_actual_scale_cancels_in_jvp_and_every_reverse_basis_without_stock_gradient_loss():
    mass = jnp.array(5.804965023785553e-139)
    def fraction(value):
        return actual_class_update(value)[2][0, 13]

    expected, survival = relative_reference(jnp.asarray(FRACTIONS), jnp.asarray(AGES))
    np.testing.assert_allclose(fraction(mass), expected, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(jax.jvp(fraction, (mass,), (jnp.ones_like(mass),))[1], np.zeros(4))
    np.testing.assert_array_equal(jax.jacrev(fraction)(mass), np.zeros(4))
    assert float(jax.grad(lambda value: jnp.dot(jnp.array([0.3, -0.2, 0.7, 1.1]), fraction(value)))(mass)) == 0.0
    np.testing.assert_allclose(
        jax.grad(lambda value: actual_class_update(value)[0][0, 13, 0, 0])(mass), survival, rtol=1e-15
    )
    np.testing.assert_allclose(
        jax.grad(lambda value: actual_class_update(value)[1][0, 13, 0, 0])(mass), 1 - survival, rtol=1e-15
    )


@pytest.mark.parametrize("mass", [0.0, -1e-139, 1e-155])
def test_zero_negative_and_below_cutoff_preserve_source_zero_fraction_tangent(mass):
    value = jnp.array(mass)
    def fn(f):
        return actual_class_update(value, fractions=f)[2][0, 13]

    np.testing.assert_array_equal(jax.jacrev(fn)(jnp.asarray(FRACTIONS)), np.zeros((4, 4)))
    if mass <= 0:
        np.testing.assert_array_equal(fn(jnp.asarray(FRACTIONS)), np.zeros(4))


def test_fraction_and_age_parameter_directions_preserve_sequential_dimensionless_law():
    f, ages = jnp.asarray(FRACTIONS), jnp.asarray(AGES)
    weights = jnp.array([0.3, -0.2, 0.7, 1.1])
    def actual(fractions, age):
        return jnp.dot(weights, actual_class_update(jnp.array(5.804965023785553e-139), fractions, age)[2][0, 13])

    def reference(fractions, age):
        return jnp.dot(weights, relative_reference(fractions, age)[0])

    directions = (jnp.array([0.1, -0.2, 0.3, -0.4]), jnp.array([1.0, -2.0, 3.0, -4.0]))
    np.testing.assert_allclose(
        jax.jvp(actual, (f, ages), directions)[1], jax.jvp(reference, (f, ages), directions)[1], rtol=1e-13, atol=1e-15
    )


@pytest.mark.parametrize("offset, represented", [(-1, False), (0, True), (1, True)])
def test_exact_existing_cutoff_and_neighboring_values_keep_original_tangent_policy(offset, represented):
    cutoff = np.sqrt(np.finfo(np.float64).tiny)
    mass = cutoff if offset == 0 else np.nextafter(cutoff, 0.0 if offset < 0 else np.inf)
    # Inactive age turnover has p_N=1, so actual postmass is exactly this
    # supplied boundary value; no inferred threshold from rounded products.
    def fn(f):
        return actual_class_update(jnp.array(mass), fractions=f, ages=jnp.zeros(4))[2][0, 13]

    expected = np.eye(4) if represented else np.zeros((4, 4))
    np.testing.assert_array_equal(jax.jacrev(fn)(jnp.asarray(FRACTIONS)), expected)
