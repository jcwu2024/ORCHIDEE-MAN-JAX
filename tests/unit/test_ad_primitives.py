from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.ad_primitives import (
    source_ratio_with_finite_tangent,
    source_sqrt_with_finite_zero_tangent,
)


def test_source_ratio_lowers_to_native_divide_and_has_finite_tiny_gradient():
    numerator = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    denominator = jnp.asarray([2.0, 0.0], dtype=jnp.float64)
    lowered = jax.jit(source_ratio_with_finite_tangent).lower(
        numerator, denominator
    ).as_text()

    assert "stablehlo.divide" in lowered
    assert "stablehlo.select" not in lowered
    assert "custom_call" not in lowered
    np.testing.assert_array_equal(
        source_ratio_with_finite_tangent(numerator[:1], denominator[:1]),
        numerator[:1] / denominator[:1],
    )
    gradient = jax.grad(
        lambda value: source_ratio_with_finite_tangent(value, value)
    )(jnp.asarray(0.0, dtype=jnp.float64))
    assert np.isfinite(gradient)


def test_source_sqrt_lowers_to_native_sqrt_and_has_finite_zero_gradient():
    values = jnp.asarray([0.0, 4.0], dtype=jnp.float64)
    lowered = jax.jit(source_sqrt_with_finite_zero_tangent).lower(
        values
    ).as_text()

    assert "stablehlo.sqrt" in lowered
    assert "stablehlo.select" not in lowered
    assert "custom_call" not in lowered
    np.testing.assert_array_equal(
        source_sqrt_with_finite_zero_tangent(values),
        jnp.sqrt(values),
    )
    assert jax.grad(source_sqrt_with_finite_zero_tangent)(
        jnp.asarray(0.0, dtype=jnp.float64)
    ) == 0.0
