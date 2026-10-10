from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.shared_daily_microbatch import (
    MicrobatchFailure,
    compile_microbatch_evaluator,
    make_daily_accumulator,
    make_daily_batch_slicer,
    make_microbatch_objective,
)
from tests.unit.test_shared_daily_microbatch import (
    _batch,
    _daily_objective,
    _parameters,
    _tree_assert_allclose,
)


@pytest.mark.parametrize("count", [32, 64])
def test_cached_tree_slicer_preserves_all_590_leaf_slices(count):
    rows = np.arange(count, dtype=np.int32)
    batch = {
        f"leaf_{index:03d}": jnp.asarray(
            (rows[:, None] + index).astype(np.float32)
            if index % 3 == 0
            else (rows[:, None] + index).astype(np.int16)
            if index % 3 == 1
            else ((rows[:, None] + index) % 2 == 0),
        )
        for index in range(590)
    }
    size = 4
    slicer = make_daily_batch_slicer(jax, size)

    for start in range(0, count, size):
        actual = slicer(batch, jnp.asarray(start, dtype=jnp.int32))
        for name, value in batch.items():
            expected = value[start : start + size]
            np.testing.assert_array_equal(actual[name], expected)
            assert actual[name].dtype == value.dtype
            assert actual[name].shape == expected.shape


@pytest.mark.parametrize("count", [32, 64])
def test_tree_slice_accumulator_matches_leaf_loss_aux_and_gradient(count):
    parameters = _parameters()
    batch = _batch(count)
    objective = make_microbatch_objective(jax, jnp, _daily_objective)
    example = jax.tree_util.tree_map(lambda value: value[:4], batch)
    evaluator = compile_microbatch_evaluator(
        jax=jax,
        parameters=parameters,
        microbatch_objective=objective,
        microbatch_example=example,
    )
    leaf = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=4,
        evaluator=evaluator,
        expected_count=count,
    )
    tree = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=4,
        evaluator=evaluator,
        expected_count=count,
        slice_execution="tree",
    )

    expected = leaf(parameters, batch)
    actual = tree(parameters, batch)

    _tree_assert_allclose(actual, expected)


def test_tree_slice_rejects_first_invalid_microbatch_with_same_gate_details():
    parameters = _parameters()
    batch = _batch(64)
    batch["hard"] = batch["hard"].at[13].set(2)
    objective = make_microbatch_objective(jax, jnp, _daily_objective)
    evaluator = compile_microbatch_evaluator(
        jax=jax,
        parameters=parameters,
        microbatch_objective=objective,
        microbatch_example=jax.tree_util.tree_map(lambda value: value[:4], batch),
    )
    accumulator = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=4,
        evaluator=evaluator,
        expected_count=64,
        slice_execution="tree",
    )

    with pytest.raises(MicrobatchFailure) as caught:
        accumulator(parameters, batch)

    assert caught.value.microbatch_index == 3
    assert caught.value.details["row_start"] == 12
    assert caught.value.details["row_stop"] == 16
    assert caught.value.details["hard"] == {"toy_failure": 2}
