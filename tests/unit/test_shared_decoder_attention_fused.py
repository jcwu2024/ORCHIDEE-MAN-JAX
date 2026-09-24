"""Numerical checks for the diagnostic single-contraction decoder attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_decoder_attention_fused import (
    fused_attention_from_projected_segments,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import (
    _attention_from_projected_segments,
    _project_key_value,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _residual_block_parameters


@pytest.mark.parametrize("with_carbon", [False, True])
@pytest.mark.parametrize("preproject_dynamic", [False, True])
def test_fused_attention_matches_production_outputs_and_native_gradients(with_carbon, preproject_dynamic):
    width, rows = 32, 7
    block = _residual_block_parameters(jax.random.PRNGKey(17), width, 4 * width)
    parameters = block["attention"]
    keys = jax.random.split(jax.random.PRNGKey(23), 5)
    payload = {
        "query": jax.random.normal(keys[0], (rows, 1, width)),
        "dynamic": jax.random.normal(keys[1], (rows, 3, width)),
        "family": jax.random.normal(keys[2], (5, width)),
        "process": jax.random.normal(keys[3], (1, width)),
        "carbon": jax.random.normal(keys[4], (9, width)),
    }
    active = jnp.arange(rows) % 3 != 0
    family_mask = jnp.broadcast_to(active[:, None, None], (rows, 1, 5))
    dynamic_mask = jnp.broadcast_to(active[:, None, None], (rows, 1, 3))
    process_mask = active[:, None, None]
    carbon_mask = jnp.broadcast_to(active[:, None, None], (rows, 1, 9))
    carbon_mask = carbon_mask & (jnp.arange(9)[None, None, :] % 2 == 0)

    def apply(helper, trainable, inputs):
        family_key, family_value = _project_key_value(trainable, inputs["family"])
        process_key, process_value = _project_key_value(trainable, inputs["process"])
        dynamic_key = dynamic_value = None
        if preproject_dynamic:
            dynamic_key, dynamic_value = _project_key_value(trainable, inputs["dynamic"])
        carbon_key = carbon_value = None
        if with_carbon:
            carbon_key, carbon_value = _project_key_value(trainable, inputs["carbon"])
        return helper(
            trainable,
            inputs["query"],
            inputs["dynamic"],
            dynamic_key=dynamic_key,
            dynamic_value=dynamic_value,
            family_key=family_key,
            family_value=family_value,
            family_mask=family_mask,
            dynamic_mask=dynamic_mask,
            process_key=process_key,
            process_value=process_value,
            process_mask=process_mask,
            carbon_key=carbon_key,
            carbon_value=carbon_value,
            carbon_mask=carbon_mask if with_carbon else None,
            heads=4,
            normalization_epsilon=1e-6,
        )

    reference = apply(_attention_from_projected_segments, parameters, payload)
    candidate = apply(fused_attention_from_projected_segments, parameters, payload)
    np.testing.assert_allclose(candidate, reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(candidate)[~np.asarray(active)], 0.0)

    def loss(helper, trainable, inputs):
        output = apply(helper, trainable, inputs)
        return jnp.sum(output * jnp.sin(jnp.arange(rows, dtype=jnp.float32))[:, None, None])

    reference_gradient = jax.grad(loss, argnums=(1, 2))(_attention_from_projected_segments, parameters, payload)
    candidate_gradient = jax.grad(loss, argnums=(1, 2))(fused_attention_from_projected_segments, parameters, payload)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(candidate_gradient), jax.tree_util.tree_leaves(reference_gradient), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
