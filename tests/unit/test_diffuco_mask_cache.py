"""Regression for host cache retaining a JAX tracer between batch shapes."""
import jax
import numpy as np
import pytest

from jax_orchidee.sechiba import diffuco


@pytest.mark.parametrize("pft_index", range(14))
def test_mask_cache_remains_host_only_across_independent_traces(pft_index):
    diffuco._diffuco_pft_column_mask.cache_clear()
    for size in (2, 3):
        backgrounds = [np.arange(size * 14, dtype=np.float64).reshape(size, 14) + k for k in range(5)]
        columns = [np.full((size, 1), 100 + k, dtype=np.float64) for k in range(5)]

        def transition(*args):
            return diffuco._diffuco_insert_pft_output_columns(*args, pft_index=pft_index)

        # checking_leaks rejects a cached tracer even on the first trace,
        # independently of whether this JAX version exposes all_args_info.
        with jax.checking_leaks():
            lowered = jax.jit(transition).lower(*(backgrounds + columns))
        info = lowered._lowering.compile_args.get("all_args_info")
        if info is not None:
            assert len(info.in_avals) == 10
        actual = lowered.compile()(*(backgrounds + columns))
        for output, background, column in zip(actual, backgrounds, columns, strict=True):
            expected = background.copy()
            expected[:, pft_index:pft_index + 1] = column
            np.testing.assert_array_equal(output, expected)
            assert output.dtype == expected.dtype
        mask = diffuco._diffuco_pft_column_mask(14, pft_index)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        assert not mask.flags.writeable
        np.testing.assert_array_equal(mask, np.arange(14) == pft_index)
        jax.clear_caches()
