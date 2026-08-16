from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.strong_baseline_qualification import (
    initialize_strong_baseline_adamw,
    padded_epoch_batches,
    qualification_learning_rate,
    strong_baseline_adamw_update,
    strong_baseline_weight_decay_mask,
)


def test_padded_epoch_batches_consume_every_sample_once():
    batches = padded_epoch_batches(10, 4, seed=23, epoch=2)
    admitted = np.concatenate(
        [indices[weights.astype(np.bool_)] for indices, weights in batches]
    )
    assert len(batches) == 3
    assert sorted(admitted.tolist()) == list(range(10))
    assert sum(float(np.sum(weights)) for _indices, weights in batches) == 10.0
    assert np.count_nonzero(batches[-1][1] == 0.0) == 2


def test_qualification_learning_rate_has_declared_warmup_and_cosine_end():
    assert qualification_learning_rate(1, 100) == pytest.approx(5.0e-5)
    assert qualification_learning_rate(2, 100) == pytest.approx(1.0e-4)
    assert qualification_learning_rate(100, 100) == pytest.approx(1.0e-5)


def test_adamw_excludes_biases_and_normalization_scales_from_decay():
    parameters = {
        "layer": {
            "kernel": jnp.ones((2,), dtype=jnp.float32),
            "bias": jnp.ones((2,), dtype=jnp.float32),
        },
        "attention_norm_scale": jnp.ones((2,), dtype=jnp.float32),
        "attention_norm_bias": jnp.ones((2,), dtype=jnp.float32),
    }
    gradients = {
        "layer": {
            "kernel": jnp.zeros((2,), dtype=jnp.float32),
            "bias": jnp.zeros((2,), dtype=jnp.float32),
        },
        "attention_norm_scale": jnp.zeros((2,), dtype=jnp.float32),
        "attention_norm_bias": jnp.zeros((2,), dtype=jnp.float32),
    }
    mask = strong_baseline_weight_decay_mask(parameters)
    state = initialize_strong_baseline_adamw(parameters)
    update = jax.jit(
        lambda values, derivatives, optimizer: strong_baseline_adamw_update(
            values,
            derivatives,
            optimizer,
            1.0e-2,
            mask,
            weight_decay=0.1,
        )
    )
    updated, _state = update(parameters, gradients, state)
    assert np.all(np.asarray(updated["layer"]["kernel"]) < 1.0)
    np.testing.assert_array_equal(np.asarray(updated["layer"]["bias"]), np.ones((2,)))
    np.testing.assert_array_equal(np.asarray(updated["attention_norm_scale"]), np.ones((2,)))
    np.testing.assert_array_equal(np.asarray(updated["attention_norm_bias"]), np.ones((2,)))
