from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining import first_neural_process_training_run as run
from research.daily_coarse_graining.first_neural_process_training_run import (
    AdamWState,
    checkpoint_identity,
    initialize_adamw,
    iter_process_validation_batches,
    learning_rate_at_update,
    load_process_training_protocol,
    load_training_checkpoint,
    make_adamw_train_step,
    prefetched_process_training_batches,
    write_training_checkpoint,
)


def test_frozen_first_process_training_protocol_and_schedule():
    protocol = load_process_training_protocol()
    assert protocol.payload["status"] == "ready_unsubmitted"
    assert protocol.payload["sampling"]["test_split_allowed"] is False
    assert learning_rate_at_update(protocol, 1) == pytest.approx(6.0e-7)
    assert learning_rate_at_update(protocol, 500) == pytest.approx(3.0e-4)
    assert learning_rate_at_update(protocol, 10_000) == pytest.approx(3.0e-5)
    with pytest.raises(ValueError, match="outside"):
        learning_rate_at_update(protocol, 10_001)


def test_protocol_hash_and_test_policy_fail_closed(tmp_path):
    source = load_process_training_protocol().source_path
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["sampling"]["test_split_allowed"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_process_training_protocol(path)


def test_adamw_step_clips_and_updates_finite_parameters():
    protocol = load_process_training_protocol()

    def loss(parameters, inputs, targets):
        return jnp.mean((parameters["weight"] * inputs - targets) ** 2)

    parameters = {"weight": jnp.asarray(10.0, dtype=jnp.float64)}
    optimizer = initialize_adamw(parameters)
    assert isinstance(optimizer, AdamWState)
    step = make_adamw_train_step(loss, protocol)
    updated, optimizer, value, gradient_norm, clip_scale = step(
        parameters,
        optimizer,
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        jnp.asarray(3.0e-4, dtype=jnp.float64),
    )
    assert np.isfinite(np.asarray(value))
    assert float(gradient_norm) > 1.0
    assert float(clip_scale) < 1.0
    assert float(updated["weight"]) < float(parameters["weight"])
    assert int(optimizer.step) == 1


def test_checkpoint_roundtrip_and_identity_drift(tmp_path):
    class Prepared:
        candidate = type("Candidate", (), {"sha256": "a" * 64})()
        protocol = type(
            "Protocol",
            (),
            {
                "sha256": "b" * 64,
                "training": {
                    "arms": ["axial_process_operator_v1"],
                    "seeds": [20260807],
                },
            },
        )()
        artifact_hashes = {"process_statistics_json_sha256": "c" * 64}

    identity = checkpoint_identity(
        Prepared(),
        arm="axial_process_operator_v1",
        seed=20260807,
        source_git_head="d" * 40,
    )
    path = write_training_checkpoint(
        tmp_path / "checkpoint.pkl",
        identity=identity,
        update=250,
        parameters={"weight": np.asarray([1.0])},
        optimizer=AdamWState(250, {"weight": np.asarray([0.0])}, {"weight": np.asarray([0.0])}),
        history=[{"update": 250, "loss": 1.0}],
    )
    loaded = load_training_checkpoint(path, identity=identity, maximum_updates=10_000)
    assert loaded["update"] == 250
    drifted = dict(identity)
    drifted["seed"] = 20260808
    with pytest.raises(ValueError, match="identity drift"):
        load_training_checkpoint(path, identity=drifted, maximum_updates=10_000)


def test_bounded_prefetch_preserves_batch_order(monkeypatch):
    expected = [(np.asarray([index]), np.asarray([-index])) for index in range(4)]

    def batches(*_args, **_kwargs):
        yield from expected

    monkeypatch.setattr(run, "iter_process_training_batches", batches)
    observed = list(
        prefetched_process_training_batches(
            object(),
            seed=1,
            batch_size=16,
            maximum_batches=4,
            prefetch_batches=2,
        )
    )
    assert len(observed) == len(expected)
    for actual, wanted in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(actual[0], wanted[0])
        np.testing.assert_array_equal(actual[1], wanted[1])


def test_validation_slices_remain_declared_and_test_closed():
    with pytest.raises(ValueError, match="not declared"):
        list(
            iter_process_validation_batches(
                object(),
                spatial_split="test",
                temporal_split="train",
                seed=1,
                batch_size=16,
                maximum_batches=1,
            )
        )
