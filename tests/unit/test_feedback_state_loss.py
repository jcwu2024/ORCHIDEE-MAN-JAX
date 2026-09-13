import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.feedback_state_loss import feedback_state_mse


def test_small_physical_error_is_measured_in_recurrent_input_units():
    def loss(value):
        return feedback_state_mse(
            predicted=value, teacher=jnp.array([0., 1.]),
            predicted_defined=True, teacher_defined=True, eligible=True,
            scale=jnp.array([1., .01])).loss

    value, grad = jax.value_and_grad(loss)(jnp.array([.2, .99]))
    np.testing.assert_allclose(value, .52)
    np.testing.assert_allclose(grad, [.2, -100.])
    direction = jnp.array([.1, -.2])
    _, tangent = jax.jvp(loss, (jnp.array([.2, .99]),), (direction,))
    np.testing.assert_allclose(tangent, jnp.vdot(grad, direction))


def test_inactive_nan_does_not_contaminate_gradient_but_invalid_active_is_reported():
    def run(value):
        return feedback_state_mse(
            predicted=value, teacher=jnp.array([1., jnp.nan, 1.]),
            predicted_defined=jnp.array([True, False, False]),
            teacher_defined=jnp.array([True, False, True]),
            eligible=jnp.array([True, False, True]), scale=jnp.array([.1, jnp.nan, .1]))

    result = run(jnp.array([.9, jnp.nan, jnp.nan]))
    assert int(result.count) == 1
    assert int(result.expected_count) == 2
    assert int(result.invalid_count) == 1
    grad = jax.grad(lambda value: run(value).loss)(jnp.array([.9, jnp.nan, jnp.nan]))
    np.testing.assert_allclose(grad, [-20., 0., 0.])


def test_counteracting_coordinate_errors_cannot_cancel_and_invalid_scale_is_counted():
    result = feedback_state_mse(
        predicted=jnp.array([.8, 1.2, 1.]), teacher=jnp.ones(3),
        predicted_defined=True, teacher_defined=True, eligible=True,
        scale=jnp.array([.1, .1, 0.]))
    np.testing.assert_allclose(result.loss, 4.)
    assert int(result.invalid_count) == 1
