from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
from research.daily_coarse_graining.daily_operator.one_day_training_state import (
    make_gradient_accumulator,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.probe_mixed_physical_recurrent import parser, select_draws

jax.config.update("jax_enable_x64", True)


def _parameters():
    return {
        "kernel": jnp.asarray([0.4, -0.7, 0.2], dtype=jnp.float64),
        "bias": jnp.asarray(-0.15, dtype=jnp.float64),
    }


def _prediction(parameters, inputs):
    return inputs @ parameters["kernel"] + parameters["bias"]


def _mean_loss(parameters, inputs, targets):
    residual = _prediction(parameters, inputs) - targets
    return jnp.mean(residual * residual)


def _tree_allclose(actual, expected, *, rtol=1e-12, atol=1e-12):
    for observed, wanted in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(observed, wanted, rtol=rtol, atol=atol)


def _accumulated_mean(parameters, inputs, targets, groups):
    add, finish = make_gradient_accumulator(jax, jnp)
    total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    for indices in groups:
        indices = jnp.asarray(indices)
        gradient = jax.grad(_mean_loss)(parameters, inputs[indices], targets[indices])
        total = add(total, gradient, len(indices))
    return finish(total, len(inputs))


def test_mixed_gradient_matches_daily_mean_plus_recurrent_window_mean():
    parameters = _parameters()
    index = jnp.arange(64, dtype=jnp.float64)
    daily_inputs = jnp.stack(
        (0.1 + index / 50.0, jnp.sin(index / 7.0), jnp.cos(index / 11.0)), axis=1
    )
    daily_targets = 0.2 - index / 100.0
    recurrent_inputs = jnp.asarray(
        [[2.0, -1.0, 0.5], [-0.4, 1.7, 0.9]], dtype=jnp.float64
    )
    recurrent_targets = jnp.asarray([-0.6, 1.2], dtype=jnp.float64)

    daily_groups = tuple(tuple(range(start, start + 8)) for start in range(0, 64, 8))
    daily = _accumulated_mean(parameters, daily_inputs, daily_targets, daily_groups)
    recurrent = _accumulated_mean(parameters, recurrent_inputs, recurrent_targets, ([0], [1]))
    mixed = make_mixed_gradient(jax, jnp)(daily, recurrent, 1.0, 1.0)
    expected = jax.grad(
        lambda current: _mean_loss(current, daily_inputs, daily_targets)
        + _mean_loss(current, recurrent_inputs, recurrent_targets)
    )(parameters)
    _tree_allclose(mixed, expected)

    # Pooling 64 daily examples and two recurrent windows under one denominator
    # silently changes the declared component weights.
    pooled = jax.tree_util.tree_map(
        lambda daily_mean, recurrent_mean: (64.0 * daily_mean + 2.0 * recurrent_mean) / 66.0,
        daily,
        recurrent,
    )
    assert any(
        not np.allclose(observed, wanted)
        for observed, wanted in zip(
            jax.tree_util.tree_leaves(pooled),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        )
    )


def test_one_clip_after_merge_matches_direct_mixed_optimizer_update():
    parameters = _parameters()
    daily = {"kernel": jnp.asarray([2.0, 100.0, 0.5]), "bias": jnp.asarray(2.0)}
    recurrent = {"kernel": jnp.asarray([-1.0, -2.0, 2.5]), "bias": jnp.asarray(1.0)}
    combine = make_mixed_gradient(jax, jnp)
    mixed = combine(daily, recurrent, 1.0, 1.0)

    protocol = {
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "learning_rate": 1.0e-5,
            "clip_norm": 1.0,
        }
    }
    update = make_update(protocol)
    state = initialize_target_oriented_adamw(parameters)
    proposed, proposed_state, norm, scale = update(parameters, mixed, state)

    direct = jax.grad(
        lambda current: sum(
            jnp.vdot(value, daily[name] + recurrent[name])
            for name, value in current.items()
        )
    )(parameters)
    expected, expected_state, expected_norm, expected_scale = update(parameters, direct, state)
    _tree_allclose(mixed, direct)
    _tree_allclose(proposed, expected)
    _tree_allclose(proposed_state, expected_state)
    np.testing.assert_allclose(norm, expected_norm)
    np.testing.assert_allclose(scale, expected_scale)
    assert float(scale) < 1.0

    # Clipping components independently before summing is a different update:
    # it changes both direction and the Adam moment state.
    daily_norm = jnp.sqrt(sum(jnp.sum(value.astype(jnp.float32) ** 2) for value in daily.values()))
    recurrent_norm = jnp.sqrt(
        sum(jnp.sum(value.astype(jnp.float32) ** 2) for value in recurrent.values())
    )
    preclipped = combine(
        daily,
        recurrent,
        jnp.minimum(1.0, 1.0 / daily_norm),
        jnp.minimum(1.0, 1.0 / recurrent_norm),
    )
    wrong_proposed, wrong_state, _, _ = update(parameters, preclipped, state)
    assert not np.allclose(np.asarray(wrong_state.first["kernel"]), np.asarray(proposed_state.first["kernel"]))
    assert not np.allclose(np.asarray(wrong_proposed["kernel"]), np.asarray(proposed["kernel"]))


def test_select_draws_uses_successive_daily_batches_and_stored_t7_order():
    batches = []
    for batch_index in range(5):
        rows = tuple(
            SimpleNamespace(artifact_id=batch_index, batch_row=row, day_key=(batch_index, row))
            for row in range(32)
        )
        batches.append(SimpleNamespace(batch_index=batch_index, draw_count=32, rows=rows))
    sampler = SimpleNamespace(iter_epoch=lambda epoch: iter(batches) if epoch == 0 else iter(()))
    windows = []
    t7_identifiers = []
    for index in range(8):
        t7_identifiers.append(f"t7-{index}")
        windows.extend(
            (
                {
                    "identifier": t7_identifiers[-1],
                    "length": 7,
                    "teacher_day_keys": [[index, day] for day in range(7)],
                },
                {
                    "identifier": f"t30-{index}",
                    "length": 30,
                    "teacher_day_keys": [[100 + index, day] for day in range(30)],
                },
            )
        )
    selection = {
        "role": "train_only_bounded_recurrent_admission_not_validation",
        "windows": windows,
    }

    daily, recurrent = select_draws(sampler, selection)

    assert tuple((row.artifact_id, row.batch_row) for row in daily[0]) == tuple(
        (batch, row) for batch in (0, 1) for row in range(32)
    )
    assert tuple((row.artifact_id, row.batch_row) for row in daily[1]) == tuple(
        (batch, row) for batch in (2, 3) for row in range(32)
    )
    assert [[row["identifier"] for row in group] for group in recurrent] == [
        t7_identifiers[:2],
        t7_identifiers[2:4],
    ]


def test_select_draws_rejects_replayed_daily_example():
    batches = []
    for batch_index in range(4):
        rows = [SimpleNamespace(artifact_id=batch_index, batch_row=row) for row in range(32)]
        batches.append(SimpleNamespace(batch_index=batch_index, draw_count=32, rows=tuple(rows)))
    batches[-1].rows = (*batches[-1].rows[:-1], batches[0].rows[0])
    sampler = SimpleNamespace(iter_epoch=lambda epoch: iter(batches))
    selection = {
        "role": "train_only_bounded_recurrent_admission_not_validation",
        "windows": [
            {"length": 7, "teacher_day_keys": [[index, day] for day in range(7)]}
            for index in range(4)
        ],
    }

    with pytest.raises(ValueError, match="128 distinct TRAIN examples"):
        select_draws(sampler, selection)


def test_mixed_runner_cli_exposes_shape_and_forward_only_modes():
    required = [
        "--dataset-manifest", "dataset.json",
        "--acceptance-report", "acceptance.json",
        "--base-statistics", "base.npz",
        "--process-statistics", "process.npz",
        "--fast-statistics", "fast.npz",
        "--prepared-manifest", "prepared.json",
        "--output", "report.json",
        "--mixed-protocol", "mixed.json",
        "--pool-protocol", "pool.json",
        "--admission-config", "admission.json",
        "--catalogue-report", "catalogue.json",
        "--selection", "selection.json",
    ]
    shape = parser().parse_args([*required, "--shape-only"])
    forward = parser().parse_args([*required, "--forward-only"])
    assert shape.shape_only and not shape.forward_only
    assert forward.forward_only and not forward.shape_only
