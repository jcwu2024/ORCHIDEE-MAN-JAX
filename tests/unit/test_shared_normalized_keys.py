"""Numerical gates for masks, bias fallback and native derivative placement."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import shared_transformer as st
from scripts.hpc.probe_shared_normalized_keys import component


@pytest.mark.parametrize("zero_bias", (False, True))
@pytest.mark.parametrize("variant", ("normalized", "unroll2"))
def test_execution_preserves_masks_and_full_vjp(zero_bias, variant):
    config = st.SharedTransformerConfig(width=8, attention_heads=2, ffn_width=32,
                                        coordinate_blocks=2, decoder_execution="gather_dynamic_reuse")
    rng = np.random.default_rng(301)
    def array(shape):
        return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    blocks = tuple(st._residual_block_parameters(jax.random.PRNGKey(i), 8, 32) for i in range(2))
    for block in blocks:
        for name in ("key", "value"):
            block["attention"][name]["bias"] = jnp.zeros((8,)) if zero_bias else array((8,))
    # Active row, inactive-but-unmasked bias stress row, and all-masked row.
    values = (array((3, 1, 8)), array((3, 3, 8)), array((4, 8)), array((1, 8)), array((5, 8)), array((2, 8)), array((2, 8)))
    masks = tuple(jnp.asarray([[True], [True], [False]])[:, None, :] * jnp.ones((3, 1, count), dtype=bool)
                  for count in (4, 3, 1, 5))
    indices = (jnp.asarray([[0, 1], [1, 0], [0, 1]]), jnp.asarray([0, 1, 1]), jnp.asarray([True, False, False]))
    cotangent = array((3, 1, 8))
    results = []
    for candidate in (False, True):
        forward = lambda p, x: component(p, x, masks, indices, config, 2,
                                        normalized=candidate and variant == "normalized",
                                        native_unroll=2 if candidate and variant == "unroll2" else 1)
        output, pullback = jax.vjp(forward, blocks, values)
        results.append((output, pullback(cotangent)))
    for actual, expected in zip(jax.tree_util.tree_leaves(results[1]), jax.tree_util.tree_leaves(results[0]), strict=True):
        assert np.all(np.isfinite(actual))
        assert np.all(np.isfinite(expected))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_normalized_key_adapter_restores_patches():
    from scripts.hpc.shared_normalized_key_execution import patches
    original = st._run_coordinate_blocks_reusing_projections
    with pytest.raises(RuntimeError):
        with patches(st.SharedTransformerConfig()):
            assert st._run_coordinate_blocks_reusing_projections is not original
            raise RuntimeError("abort tracing")
    assert st._run_coordinate_blocks_reusing_projections is original
