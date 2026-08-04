from __future__ import annotations

import jax
import numpy as np

from research.daily_coarse_graining.constrained_daily_replay import (
    apply_bounded_inventory_transition,
    apply_bounded_tendency,
    encode_inventory_endpoints,
    encode_signed_inventory_endpoints,
    encode_thermal_endpoints,
)


def test_inventory_endpoint_encoding_is_nonnegative_and_exact_without_clipping():
    start = np.asarray([0.0, 2.0, 4.0, np.nan, -1.0e20])
    end = np.asarray([3.0, 1.0, 4.5, np.nan, -1.0e20])

    encoded = encode_inventory_endpoints(start, end, name="test.inventory")

    np.testing.assert_allclose(encoded.outgoing_fraction[:3], [0.0, 0.5, 0.0])
    np.testing.assert_allclose(encoded.incoming_amount[:3], [3.0, 0.0, 0.5])
    np.testing.assert_allclose(encoded.reconstructed[:3], end[:3])
    assert np.isnan(encoded.reconstructed[3])
    assert encoded.reconstructed[4] == -1.0e20
    np.testing.assert_array_equal(encoded.defined, [True, True, True, False, False])
    assert np.min(encoded.reconstructed[encoded.defined]) >= 0.0


def test_inventory_endpoint_encoding_rejects_negative_teacher_stock():
    with np.testing.assert_raises_regex(ValueError, "negative Teacher endpoint"):
        encode_inventory_endpoints([1.0], [-2.0e-12], name="test.inventory")


def test_signed_canopy_carry_uses_nonnegative_storage_and_debt_components():
    encoded = encode_signed_inventory_endpoints(
        np.asarray([-8.0e-5, 0.2, -1.0e20]),
        np.asarray([0.1, -2.0e-5, -1.0e20]),
        name="water.qsintveg",
    )

    np.testing.assert_allclose(encoded.reconstructed[:2], [0.1, -2.0e-5])
    assert encoded.reconstructed[2] == -1.0e20
    assert np.min(encoded.positive.reconstructed[encoded.positive.defined]) >= 0.0
    assert np.min(encoded.debt.reconstructed[encoded.debt.defined]) >= 0.0


def test_bounded_inventory_and_tendency_updates_have_finite_gradients():
    inventory_gradient = jax.grad(lambda fraction: apply_bounded_inventory_transition(2.0, fraction, 0.5))(0.25)
    tendency_gradient = jax.grad(lambda normalized: apply_bounded_tendency(280.0, normalized, 100.0))(0.05)

    assert np.isfinite(np.asarray(inventory_gradient))
    assert np.isfinite(np.asarray(tendency_gradient))
    np.testing.assert_allclose(inventory_gradient, -2.0)
    np.testing.assert_allclose(tendency_gradient, 100.0)


def test_thermal_endpoint_encoding_preserves_undefined_mask_and_declared_bound():
    encoded = encode_thermal_endpoints(
        np.asarray([280.0, np.nan]),
        np.asarray([285.0, np.nan]),
        name="test.temperature",
        bound=10.0,
    )

    np.testing.assert_allclose(encoded.normalized_tendency, [0.5, 0.0])
    np.testing.assert_allclose(encoded.reconstructed[0], 285.0)
    assert np.isnan(encoded.reconstructed[1])

    with np.testing.assert_raises_regex(ValueError, "exceeds"):
        encode_thermal_endpoints([280.0], [291.0], name="test.temperature", bound=10.0)
