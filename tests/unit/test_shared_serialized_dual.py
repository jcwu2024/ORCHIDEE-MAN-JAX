"""Fail-closed guards and bounded timing for saved-executable dual replay."""
from __future__ import annotations

from collections import namedtuple
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc.probe_shared_serialized_dual import (
    RecurrentWorker,
    RemoteDualExecutor,
    _compact_learning_run,
    _validate_bundle,
    measure_complete_updates,
    require_distinct_batches,
)


def test_real_64_row_batch_requires_two_distinct_full_artifacts():
    import jax

    left = {"features": np.ones((32, 2), dtype=np.float32),
            "mask": np.ones((32,), dtype=bool)}
    right = {"features": np.zeros((32, 2), dtype=np.float32),
             "mask": np.zeros((32,), dtype=bool)}
    require_distinct_batches(jax, (11, 12), left, right)
    with pytest.raises(ValueError, match="distinct artifact IDs"):
        require_distinct_batches(jax, (11, 11), left, right)
    with pytest.raises(ValueError, match="complete real 32-row"):
        require_distinct_batches(jax, (11, 12), left, {**right, "mask": right["mask"][:16]})
    with pytest.raises(ValueError, match="incompatible trees"):
        require_distinct_batches(jax, (11, 12), left, {"features": right["features"]})


def test_compact_bundle_requires_exact_reference_identity_and_fresh_checkpoint(tmp_path):
    checkpoint = tmp_path / "trusted.update-000000.checkpoint.pkl"
    method = {"training_plan": {"training_artifact_ids": [11, 12, 13]}}
    reference = {
        "source_git_head": "original-commit", "method": method,
        "checkpoints": [{"update": 0, "retained_milestone": True,
                         "path": str(checkpoint)}],
    }
    bundle = {
        "schema_version": "shared_serialized_dual_inputs_v1",
        "source_git_head": "original-commit", "profile_sha256": "profile-digest",
        "method": method, "artifact_ids": (11, 12),
        "initial_checkpoint": str(checkpoint), "payloads": [object(), object()],
    }
    kwargs = dict(profile_sha256="profile-digest", reference=reference,
                  source_head="original-commit")
    assert _validate_bundle(bundle, **kwargs) == (11, 12)
    with pytest.raises(ValueError, match="source/profile/method"):
        _validate_bundle({**bundle, "profile_sha256": "other"}, **kwargs)
    with pytest.raises(ValueError, match="declared artifacts"):
        _validate_bundle({**bundle, "artifact_ids": (11, 11)}, **kwargs)
    with pytest.raises(ValueError, match="checkpoint differs"):
        _validate_bundle({**bundle, "initial_checkpoint": str(tmp_path / "other.pkl")}, **kwargs)
    with pytest.raises(ValueError, match="exactly two prepared payloads"):
        _validate_bundle({**bundle, "payloads": [object()]}, **kwargs)
    with pytest.raises(ValueError, match="exactly one retained update-0"):
        _validate_bundle(bundle, **{**kwargs, "reference": {**reference, "checkpoints": []}})


def test_compact_replay_feeds_two_saved_payloads_and_persists_checkpoint(monkeypatch, tmp_path):
    import jax

    from research.daily_coarse_graining.daily_operator import fresh_start_training
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc import shared_daily_microbatch

    Optimizer = namedtuple("Optimizer", "step")
    device = SimpleNamespace(platform="gpu", process_index=0, id=0,
                             memory_stats=lambda: {"peak_bytes_in_use": 17})
    monkeypatch.setattr(jax, "local_devices", lambda: [device])
    monkeypatch.setattr(jax, "default_device", lambda *_: nullcontext())
    monkeypatch.setattr(jax, "device_put", lambda value, *_: value)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(learning, "load_learning_checkpoint", lambda *_a, **_kw: {
        "completed_updates": 0, "parameters": {"w": np.array([1.], dtype=np.float32)},
        "optimizer": Optimizer(0),
    })
    monkeypatch.setattr(learning, "make_tree_operations", lambda *_a: {"finite": lambda _: True})
    monkeypatch.setattr(learning, "compile_gradient_evaluators", lambda **_kw: ({"daily": "saved"}, {}))
    monkeypatch.setattr(shared_daily_microbatch, "make_daily_accumulator", lambda **_kw: "accumulate")
    monkeypatch.setattr(fresh_start_training, "make_update", lambda *_a: "update")
    observed, saved = [], []

    def execute(**kwargs):
        observed.append((float(kwargs["daily_samples"]["row"][0]),
                         kwargs["seven_case"], kwargs["thirty_case"],
                         kwargs["daily_accumulator"]))
        return kwargs["parameters"], Optimizer(kwargs["optimizer"].step + 1), {
            "loss": {"joint": 1.0}, "gradient_norms": {},
            "optimizer_gradient_norm": 1.0, "clip_scale": 1.0,
        }

    monkeypatch.setattr(learning, "execute_learning_update", execute)
    monkeypatch.setattr(learning, "save_learning_checkpoint",
                        lambda path, **kwargs: saved.append((path, kwargs)))
    output = tmp_path / "replay.updates.json"
    bundle = {
        "initial_checkpoint": str(tmp_path / "original.update-000000.checkpoint.pkl"),
        "source_git_head": "original", "method": {"configuration": {"gradient_memory_limit_gib": 24},
                                                    "daily_protocol": {}},
        "artifact_ids": (11, 12),
        "payloads": [
            {"daily": {"row": np.full(32, value, dtype=np.float32)},
             "cases": {7: f"seven-{value}", 30: f"thirty-{value}"}}
            for value in (1, 2)
        ],
    }
    args = SimpleNamespace(output=output, input_bundle=tmp_path / "bundle.pkl")
    result = _compact_learning_run(args, bundle=bundle, reference={"method": bundle["method"]})
    assert observed == [(1.0, "seven-1", "thirty-1", "accumulate"),
                        (2.0, "seven-2", "thirty-2", "accumulate")]
    assert result["status"] == "passed" and result["completed_updates"] == 2
    assert [row["artifact_id"] for row in result["updates"]] == [11, 12]
    assert saved[0][1]["completed_updates"] == 2
    assert saved[0][1]["optimizer"].step == 2
    assert saved[0][1]["method"] == bundle["method"]


def test_complete_update_timing_includes_proposals_and_three_interleaved_repeats():
    calls = []

    def serial_once():
        calls.append("serial")
        return (1, 2, {"loss": {"joint": 3.0}})

    class Dual:
        def step(self, *, daily_payload, recurrent_payload, persist):
            assert daily_payload == "daily"
            assert recurrent_payload == {7: "seven", 30: "thirty"}
            assert not persist
            calls.append("dual")
            return {"loss": {"joint": 3.0}}, {}

    fake_jax = SimpleNamespace(block_until_ready=lambda _: None)
    result = measure_complete_updates(
        jax=fake_jax, serial_once=serial_once, dual=Dual(), daily="daily",
        secondary_cases={7: "seven", 30: "thirty"}, repeats=3,
    )
    assert calls == ["serial", "dual", "serial", "dual", "dual", "serial", "serial", "dual"]
    assert len(result["serial_seconds"]) == len(result["dual_seconds"]) == 3
    assert result["interleaved"]
    with pytest.raises(ValueError, match="at least three"):
        measure_complete_updates(
            jax=fake_jax, serial_once=serial_once, dual=Dual(), daily="daily",
            secondary_cases={}, repeats=2,
        )


def test_private_worker_pipe_roundtrip_without_jax_or_network(tmp_path):
    worker = RecurrentWorker(
        tmp_path / "unused.json", cuda_device="1", memory_limit_bytes=1,
        echo_test=True,
    )
    try:
        assert worker.metadata == {"status": "ready", "mode": "echo_test"}
        fake_jax = SimpleNamespace(device_get=lambda value: value)
        result, recurrent_seconds, total_seconds = worker.evaluate(
            fake_jax, {"w": np.array([1., 2.], dtype=np.float32)},
            {7: (np.array([3.], dtype=np.float32),)},
        )
        np.testing.assert_array_equal(result["parameters"]["w"], [1., 2.])
        np.testing.assert_array_equal(result["cases"][7][0], [3.])
        assert recurrent_seconds == total_seconds == 0.0
    finally:
        worker.close()
    assert worker.process.returncode == 0


def test_remote_executor_preserves_full_gradient_and_optimizer_step_on_cpu():
    import jax
    import jax.numpy as jnp

    Optimizer = namedtuple("Optimizer", "step")
    parameters = {"w": jnp.array([1., -1.], dtype=jnp.float32)}
    optimizer = Optimizer(jnp.array(0))

    class Worker:
        def evaluate(self, jax_module, sent_parameters, cases):
            assert jax_module is jax
            assert cases == {7: "T7", 30: "T30"}
            return {
                7: {"gradient": {"w": np.array([0.2, 0.3], dtype=np.float32)},
                    "loss": 0.5, "norm": 0.36, "hard": {"violation": 0}},
                30: {"gradient": {"w": np.array([-0.1, 0.4], dtype=np.float32)},
                     "loss": 0.25, "norm": 0.41, "hard": {"violation": 0}},
            }, 0.01, 0.02

    def daily_branch(current, batch):
        assert batch == "daily"
        return {"gradient": {"w": jnp.array([0.4, -0.2])}, "loss": 1.0,
                "norm": 0.45, "hard": {"violation": 0}, "seconds": 0.01}

    def update(current, gradient, state):
        return (jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, current, gradient),
                Optimizer(state.step + 1), jnp.array(0.7), jnp.array(1.0))

    with RemoteDualExecutor(
        jax=jax, parameters=parameters, optimizer=optimizer,
        daily_branch=daily_branch, worker=Worker(),
        combine=lambda d, s, t: jax.tree_util.tree_map(lambda a, b, c: a + b + c, d, s, t),
        update=update, finite=lambda _: True,
    ) as dual:
        for step in (1, 2):
            record, values = dual.step(
                daily_payload="daily", recurrent_payload={7: "T7", 30: "T30"},
                persist=True,
            )
            assert record["optimizer_step"] == step
            np.testing.assert_allclose(values["gradient"]["w"], [0.5, 0.5], atol=1e-6)
            np.testing.assert_allclose(values["parameters"]["w"],
                                       [1. - 0.05 * step, -1. - 0.05 * step], atol=1e-6)
