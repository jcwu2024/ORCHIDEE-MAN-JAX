from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.one_day_training_state import (
    CHECKPOINT_SCHEMA,
    load_training_checkpoint,
    make_gradient_accumulator,
    save_training_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
    target_oriented_adamw_update,
)

OPTIMIZER = {
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.0,
}


def _loss(jnp, parameters, batch):
    features, targets = batch
    prediction = features @ parameters["weight"] + parameters["bias"]
    return jnp.mean((prediction - targets) ** 2)


def _assert_tree_allclose(jax, left, right, *, exact=False):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for actual, expected in zip(left_leaves, right_leaves, strict=True):
        if exact:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_weighted_microbatch_gradient_matches_full_batch_and_adamw_update():
    import jax
    import jax.numpy as jnp

    parameters = {
        "weight": jnp.asarray([0.4, -0.3], dtype=jnp.float32),
        "bias": jnp.asarray(0.2, dtype=jnp.float32),
    }
    features = jnp.asarray(
        [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 1.0], [5.0, 3.0]],
        dtype=jnp.float32,
    )
    targets = jnp.asarray([2.0, -1.0, 3.0, 0.5, -4.0], dtype=jnp.float32)
    gradient = jax.grad(lambda values, batch: _loss(jnp, values, batch))
    direct = gradient(parameters, (features, targets))
    microgradients = (
        (gradient(parameters, (features[:2], targets[:2])), 2),
        (gradient(parameters, (features[2:], targets[2:])), 3),
    )

    add, finish = make_gradient_accumulator(jax, jnp)
    total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    total_count = 0
    for mean_microgradient, count in microgradients:
        total = add(total, mean_microgradient, count)
        total_count += count
    accumulated = finish(total, total_count)

    _assert_tree_allclose(jax, accumulated, direct)

    state = initialize_target_oriented_adamw(parameters)
    contract = SimpleNamespace(optimizer=OPTIMIZER)
    direct_result = target_oriented_adamw_update(
        parameters,
        direct,
        state,
        learning_rate=1.0e-2,
        clip_norm=0.25,
        contract=contract,
    )
    accumulated_result = target_oriented_adamw_update(
        parameters,
        accumulated,
        state,
        learning_rate=1.0e-2,
        clip_norm=0.25,
        contract=contract,
    )

    _assert_tree_allclose(jax, accumulated_result[0], direct_result[0])
    _assert_tree_allclose(jax, accumulated_result[1], direct_result[1])
    np.testing.assert_allclose(accumulated_result[2], direct_result[2])
    np.testing.assert_allclose(accumulated_result[3], direct_result[3], rtol=1e-6)
    assert float(direct_result[2]) > 0.25
    assert float(direct_result[3]) < 1.0


def test_checkpoint_restores_full_adam_state_cursor_and_exact_two_step_result(tmp_path):
    import jax
    import jax.numpy as jnp

    initial = {
        "weight": jnp.asarray([0.2, -0.1], dtype=jnp.float32),
        "bias": jnp.asarray(0.05, dtype=jnp.float32),
    }
    batches = (
        (
            jnp.asarray([[1.0, 3.0], [-2.0, 0.5]], dtype=jnp.float32),
            jnp.asarray([1.5, -0.5], dtype=jnp.float32),
        ),
        (
            jnp.asarray([[4.0, -1.0], [0.25, 2.0]], dtype=jnp.float32),
            jnp.asarray([-2.0, 0.75], dtype=jnp.float32),
        ),
    )
    contract = SimpleNamespace(optimizer=OPTIMIZER)
    gradient = jax.grad(lambda values, batch: _loss(jnp, values, batch))

    state0 = initialize_target_oriented_adamw(initial)
    gradient1 = gradient(initial, batches[0])
    parameters1, state1, _, _ = target_oriented_adamw_update(
        initial,
        gradient1,
        state0,
        learning_rate=5.0e-3,
        clip_norm=0.5,
        contract=contract,
    )
    gradient2 = gradient(parameters1, batches[1])
    uninterrupted_parameters, uninterrupted_state, _, _ = target_oriented_adamw_update(
        parameters1,
        gradient2,
        state1,
        learning_rate=5.0e-3,
        clip_norm=0.5,
        contract=contract,
    )

    method = {
        "schema_version": "fresh_one_day_training_method_v1",
        "seed": 20260911,
        "sampling": {"seed": 20260912, "batch_size": 32, "microbatch_size": 4},
        "optimizer": {**OPTIMIZER, "learning_rate": 5.0e-3, "clip_norm": 0.5},
    }
    path = tmp_path / "training.checkpoint.pkl"
    saved = save_training_checkpoint(
        path,
        jax=jax,
        parameters=parameters1,
        optimizer_state=state1,
        completed_updates=1,
        cursor={"epoch": 2, "batch_offset": 7},
        method=method,
    )
    restored = load_training_checkpoint(path, expected_method=method)

    assert saved["schema_version"] == CHECKPOINT_SCHEMA
    assert restored["completed_updates"] == 1
    assert restored["cursor"] == {"epoch": 2, "batch_offset": 7}
    _assert_tree_allclose(jax, restored["parameters"], jax.device_get(parameters1), exact=True)
    _assert_tree_allclose(jax, restored["optimizer_state"], jax.device_get(state1), exact=True)
    for loaded, original in zip(
        jax.tree_util.tree_leaves(restored["parameters"]),
        jax.tree_util.tree_leaves(parameters1),
        strict=True,
    ):
        assert loaded.dtype == original.dtype
    for loaded, original in zip(
        jax.tree_util.tree_leaves(restored["optimizer_state"]),
        jax.tree_util.tree_leaves(state1),
        strict=True,
    ):
        assert loaded.dtype == original.dtype

    resumed_parameters1 = jax.device_put(restored["parameters"])
    resumed_state1 = jax.device_put(restored["optimizer_state"])
    resumed_gradient2 = gradient(resumed_parameters1, batches[1])
    resumed_parameters, resumed_state, _, _ = target_oriented_adamw_update(
        resumed_parameters1,
        resumed_gradient2,
        resumed_state1,
        learning_rate=5.0e-3,
        clip_norm=0.5,
        contract=contract,
    )

    _assert_tree_allclose(jax, resumed_parameters, uninterrupted_parameters, exact=True)
    _assert_tree_allclose(jax, resumed_state, uninterrupted_state, exact=True)
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_checkpoint_rejects_method_mismatch(tmp_path):
    import jax
    import jax.numpy as jnp

    parameters = {"weight": jnp.asarray([1.0], dtype=jnp.float32)}
    path = tmp_path / "training.checkpoint.pkl"
    save_training_checkpoint(
        path,
        jax=jax,
        parameters=parameters,
        optimizer_state=initialize_target_oriented_adamw(parameters),
        completed_updates=0,
        cursor={"epoch": 0, "batch_offset": 0},
        method={"schema_version": "method_v1", "seed": 1},
    )

    with pytest.raises(ValueError, match="method mismatch"):
        load_training_checkpoint(
            path,
            expected_method={"schema_version": "method_v1", "seed": 2},
        )
