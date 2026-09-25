"""Output and native-AD parity for shared backbone source projections."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_backbone_projection_reuse import (
    run_backbone_with_shared_source_projection,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import SharedTransformerConfig, _run_blocks
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _residual_block_parameters


@pytest.mark.parametrize("batch", [None, 4])
def test_common_source_projection_matches_backbone_and_all_native_gradients(batch):
    width, experts, sources, axes = 32, 3, 5, 4
    config = SharedTransformerConfig(width=width, ffn_width=4 * width)
    prefix = () if batch is None else (batch,)
    keys = jax.random.split(jax.random.PRNGKey(20260925), 5)
    blocks = tuple(_residual_block_parameters(key, width, 4 * width) for key in keys[:2])
    payload = {
        "query": jax.random.normal(keys[2], prefix + (experts, 1, width), dtype=jnp.float32),
        "source": jax.random.normal(keys[3], prefix + (sources, width), dtype=jnp.float32),
        "axis": jax.random.normal(keys[4], prefix + (experts, axes, width), dtype=jnp.float32),
    }
    mask = jnp.ones(prefix + (experts, 1, sources + axes), dtype=jnp.bool_)
    mask = mask.at[..., 0, :, 1::2].set(False)
    mask = mask.at[..., -1, :, :].set(False)
    if batch is not None:
        mask = mask.at[1, 1, :, :sources].set(False)

    def reference(trainable, inputs):
        source = jnp.broadcast_to(inputs["source"][..., None, :, :], prefix + (experts, sources, width))
        bank = jnp.concatenate((source, inputs["axis"]), axis=-2)
        return _run_blocks(inputs["query"], bank, mask, trainable, config)

    def candidate(trainable, inputs):
        return run_backbone_with_shared_source_projection(
            inputs["query"], inputs["source"], inputs["axis"], mask, trainable, config
        )

    expected = reference(blocks, payload)
    actual = candidate(blocks, payload)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert np.isfinite(np.asarray(actual)).all()

    weights = jnp.sin(jnp.arange(actual.size, dtype=jnp.float32)).reshape(actual.shape)

    def loss(function, trainable, inputs):
        return jnp.sum(function(trainable, inputs) * weights)

    expected_grads = jax.grad(loss, argnums=(1, 2))(reference, blocks, payload)
    actual_grads = jax.grad(loss, argnums=(1, 2))(candidate, blocks, payload)
    for got, want in zip(
        jax.tree_util.tree_leaves(actual_grads), jax.tree_util.tree_leaves(expected_grads), strict=True
    ):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    for key in ("query", "source", "axis"):
        assert np.isfinite(np.asarray(actual_grads[1][key])).all()
        assert np.any(np.asarray(actual_grads[1][key]) != 0)


def test_common_source_projection_rejects_mismatched_mask():
    config = SharedTransformerConfig(width=32, ffn_width=128)
    with pytest.raises(ValueError, match="mask shape"):
        run_backbone_with_shared_source_projection(
            jnp.zeros((2, 1, 32)),
            jnp.zeros((3, 32)),
            jnp.zeros((2, 4, 32)),
            jnp.ones((2, 1, 6), dtype=jnp.bool_),
            (),
            config,
        )
