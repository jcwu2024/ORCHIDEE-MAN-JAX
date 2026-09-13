import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.canopy_change_loss import (
    canopy_carbon_stock_change_loss,
)


def _loss(**overrides):
    values = {
        "start_pred": jnp.asarray([[10.0, 20.0, 30.0]]),
        "next_pred": jnp.asarray([[15.0, 23.0, 34.0]]),
        "start_teacher": jnp.asarray([[12.0, 18.0, 32.0]]),
        "next_teacher": jnp.asarray([[13.0, 21.0, 35.0]]),
        "start_pred_defined": jnp.ones((1, 3), dtype=bool),
        "next_pred_defined": jnp.ones((1, 3), dtype=bool),
        "start_teacher_defined": jnp.ones((1, 3), dtype=bool),
        "next_teacher_defined": jnp.ones((1, 3), dtype=bool),
        "active_pft_mask": jnp.asarray([True, True, True]),
        "bare_pft_mask": jnp.asarray([True, False, False]),
        "scale": jnp.asarray(2.0),
    }
    values.update(overrides)
    return canopy_carbon_stock_change_loss(**values)


def test_active_source_direct_sum_includes_bare_and_reports_its_contribution():
    result = _loss()

    # No PFT fraction enters: all three active slots contribute directly.
    np.testing.assert_allclose(result.predicted_delta, [12.0])
    np.testing.assert_allclose(result.teacher_delta, [7.0])
    np.testing.assert_allclose(result.physical_residual, [5.0])
    np.testing.assert_allclose(result.loss, 6.25)
    np.testing.assert_allclose(result.bare_predicted_delta, [5.0])
    np.testing.assert_allclose(result.bare_teacher_delta, [1.0])
    np.testing.assert_allclose(result.bare_physical_residual, [4.0])
    np.testing.assert_array_equal(result.active_slot_count, [3])
    np.testing.assert_array_equal(result.bare_active_slot_count, [1])


def test_pseudo_huber_preserves_direction_with_bounded_large_error_gradient():
    def objective(value):
        # Teacher total increment is7; two fixed predicted increments sum7.
        return _loss(next_pred=jnp.asarray([[10.0 + value, 23.0, 34.0]]),
                     penalty="pseudo_huber").loss

    for value in (-1.e5, -2., 0., 2., 1.e5):
        loss, gradient = jax.value_and_grad(objective)(jnp.asarray(value))
        residual = value / 2.0
        np.testing.assert_allclose(loss, np.hypot(1., residual) - 1., rtol=1.e-6, atol=1.e-8)
        np.testing.assert_allclose(gradient, residual / np.hypot(1., residual) / 2., rtol=1.e-6)
        assert abs(float(gradient)) <= .5
        if value:
            assert np.sign(float(gradient)) == np.sign(value)
    # Source diagnostics retain physical units regardless of optimization penalty.
    np.testing.assert_allclose(_loss(penalty="pseudo_huber").physical_residual,
                               _loss().physical_residual)


def test_recurrent_change_ignores_counteracting_initial_stock_offsets():
    result = _loss(
        start_pred=jnp.asarray([[100.0, 0.0]]),
        next_pred=jnp.asarray([[110.0, 5.0]]),
        start_teacher=jnp.asarray([[1.0, 200.0]]),
        next_teacher=jnp.asarray([[4.0, 212.0]]),
        start_pred_defined=jnp.ones((1, 2), dtype=bool),
        next_pred_defined=jnp.ones((1, 2), dtype=bool),
        start_teacher_defined=jnp.ones((1, 2), dtype=bool),
        next_teacher_defined=jnp.ones((1, 2), dtype=bool),
        active_pft_mask=jnp.asarray([True, True]),
        bare_pft_mask=jnp.asarray([False, False]),
    )

    # Endpoints differ sharply, but the recurrent daily stock changes match.
    np.testing.assert_allclose(result.predicted_delta, [15.0])
    np.testing.assert_allclose(result.teacher_delta, [15.0])
    np.testing.assert_allclose(result.physical_residual, [0.0])
    np.testing.assert_allclose(result.loss, 0.0)


def test_undefined_nan_is_masked_before_arithmetic_and_gradients_stay_finite():
    def objective(next_pred):
        return _loss(
            next_pred=next_pred,
            next_pred_defined=jnp.asarray([[True, False]]),
            start_pred=jnp.asarray([[1.0, 3.0]]),
            start_teacher=jnp.asarray([[1.0, 3.0]]),
            next_teacher=jnp.asarray([[2.0, 4.0]]),
            start_pred_defined=jnp.ones((1, 2), dtype=bool),
            start_teacher_defined=jnp.ones((1, 2), dtype=bool),
            next_teacher_defined=jnp.ones((1, 2), dtype=bool),
            active_pft_mask=jnp.asarray([True, True]),
            bare_pft_mask=jnp.asarray([False, False]),
        ).loss

    next_pred = jnp.asarray([[2.0, jnp.nan]])
    loss, gradient = jax.value_and_grad(objective)(next_pred)
    np.testing.assert_allclose(loss, 0.0)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(next_pred))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_valid_synthetic_transition_has_a_finite_nonzero_gradient():
    def objective(value):
        return _loss(next_pred=jnp.asarray([[value, 23.0, 34.0]])).loss

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(12.0))
    np.testing.assert_allclose(value, 1.0)
    np.testing.assert_allclose(gradient, 1.0)
    assert bool(jnp.isfinite(gradient))
