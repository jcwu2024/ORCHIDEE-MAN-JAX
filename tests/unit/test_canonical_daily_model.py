from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalModelConfig,
    canonical_model_apply,
    canonical_one_step_loss,
    equal_group_weights,
    initialize_canonical_model,
    parameter_count,
)


def _config():
    return CanonicalModelConfig(
        state_width=12,
        forcing_width=3,
        parameter_width=4,
        landpoint_static_width=5,
        annual_condition_width=1,
        fast_day_target_width=18,
        dynamic_undefined_width=2,
        state_latent_width=8,
        forcing_latent_width=6,
        condition_latent_width=5,
        hidden_width=10,
    )


def _batch(parameter_shift=0.0):
    batch_size = 3
    state = jnp.arange(batch_size * 12, dtype=jnp.float32).reshape(batch_size, 12) / 20
    forcing = jnp.arange(batch_size * 5 * 3, dtype=jnp.float32).reshape(batch_size, 5, 3) / 10
    return CanonicalDayBatch(
        state=state,
        state_finite=jnp.ones_like(state, dtype=bool),
        normalized_fast_day_baseline=jnp.zeros(
            (batch_size, 18), dtype=jnp.float32
        ),
        forcing_native=forcing,
        forcing_finite=jnp.ones_like(forcing, dtype=bool),
        parameters=jnp.ones((batch_size, 4), dtype=jnp.float32) + parameter_shift,
        parameters_finite=jnp.ones((batch_size, 4), dtype=bool),
        landpoint_static=jnp.ones((batch_size, 5), dtype=jnp.float32),
        landpoint_static_finite=jnp.ones((batch_size, 5), dtype=bool),
        annual_conditions=jnp.ones((batch_size, 1), dtype=jnp.float32),
        annual_conditions_finite=jnp.ones((batch_size, 1), dtype=bool),
        calendar=jnp.tile(jnp.asarray([[0.0, 1.0, 0.0, 1.0]]), (batch_size, 1)),
    )


def test_canonical_model_is_jittable_and_has_fixed_output_shapes():
    config = _config()
    parameters = initialize_canonical_model(config, seed=3)
    prediction = jax.jit(canonical_model_apply)(parameters, _batch())
    assert prediction.normalized_fast_day_target.shape == (
        3,
        config.fast_day_target_width,
    )
    assert parameter_count(parameters) > config.fast_day_target_width
    assert np.all(
        np.isfinite(np.asarray(prediction.normalized_fast_day_target))
    )


def test_dynamic_parameter_conditions_change_predictions():
    parameters = initialize_canonical_model(_config(), seed=4)
    baseline = canonical_model_apply(parameters, _batch())
    shifted = canonical_model_apply(parameters, _batch(parameter_shift=0.25))
    assert not np.array_equal(
        np.asarray(baseline.normalized_fast_day_target),
        np.asarray(shifted.normalized_fast_day_target),
    )


def test_fast_day_prediction_is_centered_on_persistent_baseline():
    parameters = initialize_canonical_model(_config(), seed=4)
    batch = _batch()
    shifted = batch._replace(
        normalized_fast_day_baseline=jnp.full((3, 18), 2.5)
    )
    baseline_prediction = canonical_model_apply(parameters, batch)
    shifted_prediction = canonical_model_apply(parameters, shifted)
    np.testing.assert_allclose(
        np.asarray(shifted_prediction.normalized_fast_day_target)
        - np.asarray(baseline_prediction.normalized_fast_day_target),
        2.5,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_one_step_loss_has_finite_nonzero_gradients():
    config = _config()
    parameters = initialize_canonical_model(config, seed=5)
    batch = _batch()
    target = jnp.full(
        (3, config.fast_day_target_width), 0.2, dtype=jnp.float32
    )
    weights = equal_group_weights(
        config.fast_day_target_width,
        ((0, 6), (6, 18)),
    )

    def loss(value):
        return canonical_one_step_loss(
            value,
            batch,
            normalized_fast_day_target=target,
            fast_day_target_finite=jnp.ones_like(target, dtype=bool),
            fast_day_target_weights=weights,
            dynamic_undefined_target=jnp.zeros((3, 2), dtype=bool),
        )

    value, gradient = jax.value_and_grad(loss)(parameters)
    norm = np.sqrt(
        sum(
            float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
            for leaf in jax.tree_util.tree_leaves(gradient)
        )
    )
    assert np.isfinite(float(value))
    assert norm > 0.0


def test_group_weights_reject_gaps_and_overlaps():
    with pytest.raises(ValueError, match="cover every"):
        equal_group_weights(5, ((0, 2), (3, 5)))
    with pytest.raises(ValueError, match="overlapping"):
        equal_group_weights(5, ((0, 3), (2, 5)))
