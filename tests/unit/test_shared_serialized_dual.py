"""Fail-closed guards and bounded timing for saved-executable dual replay."""
from __future__ import annotations

from collections import namedtuple
from contextlib import nullcontext
from copy import deepcopy
import pickle
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc.probe_shared_serialized_dual import (
    RecurrentWorker,
    RemoteDualExecutor,
    _compact_learning_run,
    _validate_bundle,
    compare_schedule_pair,
    make_recurrent_branch,
    measure_complete_updates,
    measure_schedule_pairs,
    measure_changing_schedule_pairs,
    require_distinct_batches,
    run_schedule_comparison,
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


def _schedule_component(gradient):
    return {
        "gradient": {"w": np.asarray(gradient, dtype=np.float32), "static": None},
        "loss": 0.25, "norm": float(np.linalg.norm(gradient)),
        "hard": {"violation": 0}, "seconds": 0.01,
        "auxiliary": {"hard": {"violation": np.array(0)},
                      "state_diagnostic": np.array([2., 3.], dtype=np.float32),
                      "mask_count": np.array(2, dtype=np.int32), "static": None},
    }


@pytest.mark.parametrize("horizons", [(7,), (30,), (7, 30)])
@pytest.mark.parametrize("preserve_auxiliary", [False, True])
def test_recurrent_branch_exact_horizon_set_and_complete_auxiliary(horizons, preserve_auxiliary):
    import jax

    calls = []
    parameters = {"w": np.array([1., -1.], dtype=np.float32)}

    def evaluator(horizon):
        def evaluate(current, case):
            assert current is parameters and case == f"T{horizon}"
            calls.append(horizon)
            component = _schedule_component([horizon / 9., 0.])
            return (component["loss"], component["auxiliary"]), component["gradient"]
        return evaluate

    branch = make_recurrent_branch(
        jax=jax, device=jax.local_devices()[0],
        evaluators={h: evaluator(h) for h in horizons}, horizons=horizons,
        operations={"finite": lambda _: True,
                    "norm": lambda gradient: np.linalg.norm(gradient["w"])},
        preserve_auxiliary=preserve_auxiliary,
    )
    result, seconds = branch(parameters, {h: (f"T{h}",) for h in horizons})
    assert calls == list(horizons) and set(result) == set(horizons) and seconds >= 0
    for h in horizons:
        np.testing.assert_allclose(result[h]["gradient"]["w"], [h / 9., 0.])
        assert ("auxiliary" in result[h]) is preserve_auxiliary
        if preserve_auxiliary:
            assert set(result[h]["auxiliary"]) == {"hard", "state_diagnostic", "mask_count", "static"}
            np.testing.assert_array_equal(result[h]["auxiliary"]["state_diagnostic"], [2., 3.])
    with pytest.raises(ValueError, match="cases must match"):
        branch(parameters, {h: (f"T{h}",) for h in (7, 30, 1)})


@pytest.mark.parametrize("horizons,evaluators", [
    ((), {}), ((30, 7), {7: None, 30: None}), ((7, 7), {7: None}),
    ((7,), {7: None, 30: None}), ((7, 30), {30: None}),
])
def test_recurrent_branch_rejects_undeclared_horizons(horizons, evaluators):
    with pytest.raises(ValueError, match="evaluators must match"):
        make_recurrent_branch(jax=None, device=None, evaluators=evaluators,
                              operations={}, horizons=horizons)


@pytest.mark.parametrize("local_seven", [False, True])
def test_remote_schedule_overlaps_and_discards_full_optimizer_proposals(local_seven):
    import jax

    Optimizer = namedtuple("Optimizer", "step first_moment second_moment")
    parameters = {"w": np.array([1., -1.], dtype=np.float32), "static": None}
    optimizer = Optimizer(np.array(0), np.array([0., 0.], dtype=np.float32),
                          np.array([0., 0.], dtype=np.float32))
    components = {"daily": _schedule_component([0.4, -0.2]),
                  "T7": _schedule_component([1.8 / 9., 2.7 / 9.]),
                  "T30": _schedule_component([-0.1, 0.4])}
    worker_started, release_worker = Event(), Event()
    events = []

    class Worker:
        def evaluate(self, jax_module, sent_parameters, cases):
            assert jax_module is jax and sent_parameters is parameters
            assert cases == ({30: "T30"} if local_seven else {7: "T7", 30: "T30"})
            worker_started.set()
            assert release_worker.wait(timeout=5), "local branch never released worker"
            return {h: deepcopy(components[f"T{h}"]) for h in cases}, 0.01, 0.02

    def daily(current, batch):
        assert current is parameters and batch == "daily32"
        assert worker_started.wait(timeout=5), "worker did not start concurrently"
        events.append("daily")
        if not local_seven:
            release_worker.set()
        return deepcopy(components["daily"])

    def seven(current, cases):
        assert current is parameters and cases == {7: "T7"}
        assert events[-1] == "daily" and worker_started.is_set()
        events.append("T7")
        release_worker.set()
        return {7: deepcopy(components["T7"])}, 0.01

    def combine(d, s, t):
        for actual, name in zip((d, s, t), ("daily", "T7", "T30"), strict=True):
            np.testing.assert_array_equal(actual["w"], components[name]["gradient"]["w"])
        events.append("combine")
        return jax.tree_util.tree_map(lambda a, b, c: a + b + c, d, s, t)

    def update(current, gradient, state):
        assert current is parameters and state is optimizer
        events.append("update")
        proposal = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, current, gradient)
        return proposal, Optimizer(state.step + 1, gradient["w"], gradient["w"] ** 2), 0.7, 1.

    executor = RemoteDualExecutor(
        jax=jax, parameters=parameters, optimizer=optimizer, daily_branch=daily,
        worker=Worker(), combine=combine, update=update, finite=lambda _: True,
        local_recurrent_branch=seven if local_seven else None, preserve_components=True,
    )
    try:
        for _ in range(2):
            worker_started.clear()
            release_worker.clear()
            record, values = executor.step(daily_payload="daily32",
                recurrent_payload={7: "T7", 30: "T30"}, persist=False)
            assert executor.parameters is parameters and executor.optimizer is optimizer
            assert record["optimizer_step"] == 1
            np.testing.assert_allclose(values["gradient"]["w"], [.5, .5], atol=1e-6)
            assert set(values["components"]) == {"daily", "T7", "T30"}
            assert values["components"]["T7"]["auxiliary"]["static"] is None
        assert events == (["daily", "T7", "combine", "update"] if local_seven
                          else ["daily", "combine", "update"]) * 2
        np.testing.assert_array_equal(optimizer.first_moment, [0., 0.])
        np.testing.assert_array_equal(parameters["w"], [1., -1.])
    finally:
        release_worker.set()
        executor.close()


_ScheduleOptimizer = namedtuple("ScheduleOptimizer", "step first_moment second_moment")
ScheduleOptimizer = _ScheduleOptimizer  # pickle resolves the declared type name


def _fixed_schedule_result(seconds=1.):
    components = {name: _schedule_component([.1, .2]) for name in ("daily", "T7", "T30")}
    record = {"optimizer_gradient_norm": .7, "clip_scale": 1., "optimizer_step": 1,
              "loss": {"joint": .75}, "timing_seconds": {"total": seconds}}
    values = {"gradient": {"w": np.array([.3, .6], dtype=np.float32), "static": None},
              "parameters": {"w": np.array([.97, .94], dtype=np.float32), "static": None},
              "optimizer": _ScheduleOptimizer(np.array(1), np.array([.1, .2], dtype=np.float32),
                                     np.array([.01, .04], dtype=np.float32)),
              "components": components}
    return record, values


@pytest.mark.parametrize("mismatch", ["cancelling_gradients", "auxiliary_state", "auxiliary_dtype",
                                    "auxiliary_shape", "auxiliary_none", "optimizer_moment",
                                    "component_loss", "component_norm", "physical_counter"])
def test_schedule_comparison_checks_components_and_full_return_abi(mismatch):
    import jax

    old_record, old_values = _fixed_schedule_result()
    new_record, new_values = deepcopy((old_record, old_values))
    assert compare_schedule_pair(jax, old_record, old_values, new_record, new_values)["status"] == "passed"
    seven = new_values["components"]["T7"]
    if mismatch == "cancelling_gradients":
        seven["gradient"]["w"] += .1
        new_values["components"]["T30"]["gradient"]["w"] -= .1
    elif mismatch == "optimizer_moment":
        new_values["optimizer"].first_moment[0] += .1
    elif mismatch == "component_loss":
        seven["loss"] += .1
    elif mismatch == "component_norm":
        seven["norm"] += .1
    elif mismatch == "physical_counter":
        seven["hard"]["violation"] = 1
    elif mismatch == "auxiliary_state":
        seven["auxiliary"]["state_diagnostic"][0] += .1
    elif mismatch == "auxiliary_dtype":
        seven["auxiliary"]["state_diagnostic"] = seven["auxiliary"]["state_diagnostic"].astype(np.float64)
    elif mismatch == "auxiliary_shape":
        seven["auxiliary"]["state_diagnostic"] = seven["auxiliary"]["state_diagnostic"][:1]
    else:
        seven["auxiliary"]["static"] = np.array(0.)
    result = compare_schedule_pair(jax, old_record, old_values, new_record, new_values)
    assert result["status"] == "rejected"
    if mismatch == "cancelling_gradients":
        assert result["full_tree_comparisons"]["gradient"]["passed"]
        assert not result["components"]["T7"]["gradient"]["passed"]


def test_fixed_schedule_pairs_alternate_discard_and_retain_actual_auxiliary(tmp_path):
    import jax

    calls, emits = [], []
    parameters, optimizer = object(), object()

    class Executor:
        def __init__(self, name, seconds):
            self.name, self.seconds = name, seconds
            self.parameters, self.optimizer = parameters, optimizer

        def step(self, *, daily_payload, recurrent_payload, persist):
            assert daily_payload == "daily32" and recurrent_payload == {7: "T7", 30: "T30"}
            assert persist is False
            calls.append(self.name)
            return _fixed_schedule_result(self.seconds)

    result = measure_schedule_pairs(jax=jax, old=Executor("old", 2.), new=Executor("new", 1.),
        daily="daily32", cases={7: "T7", 30: "T30"}, repeats=3,
        emit=lambda stage, **kwargs: emits.append(stage), output=tmp_path / "comparison.json")
    assert calls == ["old", "new", "old", "new", "new", "old", "old", "new"]
    assert [row["order"] for row in result["paired_rows"]] == [["old", "new"], ["new", "old"], ["old", "new"]]
    assert result["completed_changing_updates"] == 0
    assert result["proposals_per_schedule_including_warmup"] == 4
    assert result["old_over_new_throughput_ratio"] == 2.
    assert result["median_card_seconds_per_proposal"] == {"old": 4., "new": 2.}
    assert emits.count("schedule_pair_compared") == 3
    with (tmp_path / "comparison.auxiliary.pkl").open("rb") as handle:
        retained = pickle.load(handle)
    assert set(retained) == {"old", "new"}
    assert set(retained["old"]["T7"]) == {"hard", "state_diagnostic", "mask_count", "static"}
    np.testing.assert_array_equal(retained["new"]["T30"]["state_diagnostic"], [2., 3.])
    with pytest.raises(ValueError, match="at least three"):
        measure_schedule_pairs(jax=jax, old=None, new=None, daily=None, cases=None,
                               repeats=2, emit=lambda *_a, **_kw: None)


def test_changing_schedule_pairs_use_following_weights_and_optimizer(tmp_path):
    import jax
    calls = []
    class Executor:
        def __init__(self, name):
            self.name = name
            self.parameters = {"w": np.array([1., 2.], np.float32)}
            self.optimizer = _ScheduleOptimizer(np.array(0), np.zeros(2, np.float32), np.zeros(2, np.float32))
        def step(self, *, daily_payload, recurrent_payload, persist):
            assert persist
            version = int(self.optimizer.step)
            calls.append((self.name, version, daily_payload, self.parameters["w"].copy()))
            record, values = _fixed_schedule_result()
            self.parameters = {"w": self.parameters["w"] - .1}
            self.optimizer = _ScheduleOptimizer(np.array(version+1), np.ones(2, np.float32)*(version+1), np.ones(2, np.float32))
            values.update(parameters=self.parameters, optimizer=self.optimizer)
            record.update(optimizer_step=version+1, branch_parameter_versions={name: version for name in ("daily", "T7", "T30")})
            return record, values
    result = measure_changing_schedule_pairs(jax=jax, old=Executor("old"), new=Executor("new"),
        payloads=[{"daily": "A", "cases": {7: "T7", 30: "T30"}}, {"daily": "B", "cases": {7: "T7b", 30: "T30b"}}],
        updates=2, emit=lambda *_a, **_kw: None, output=tmp_path/"changing.json")
    assert [(name, version, daily) for name, version, daily, _ in calls] == [("old",0,"A"),("new",0,"A"),("new",1,"B"),("old",1,"B")]
    np.testing.assert_allclose(calls[-1][3], [0.9, 1.9])
    assert result["completed_changing_updates"] == 2 and result["actual_updates_all_chains"] == 4
    with (tmp_path/"changing.old.diagnostic-chain.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    assert int(saved["optimizer"].step) == 2
    np.testing.assert_allclose(saved["parameters"]["w"], [0.8, 1.8])


@pytest.mark.parametrize("override", [{"input_bundle": None}, {"daily_counts": [64]},
                                    {"repeats": 2}, {"prepare_only": True},
                                    {"resume_checkpoint": "checkpoint"}, {"prepare_input_bundle": True}])
def test_comparison_guards_reject_before_learning_or_file_access(monkeypatch, tmp_path, override):
    from scripts.hpc import run_shared_learning as learning

    def forbidden(*_args, **_kwargs):
        pytest.fail("comparison entered a changing-learning or compile path")

    for name in ("run", "execute_learning_update", "save_learning_checkpoint", "compile_gradient_evaluators"):
        monkeypatch.setattr(learning, name, forbidden)
    args = SimpleNamespace(input_bundle=tmp_path / "missing.pkl", daily_counts=[32], repeats=3,
                           prepare_only=False, resume_checkpoint=None, prepare_input_bundle=False)
    vars(args).update(override)
    with pytest.raises(ValueError, match="comparison-only"):
        run_schedule_comparison(args)


def test_comparison_cli_rejects_conflicting_prepare_flag_before_preparation(tmp_path):
    from pathlib import Path

    output = tmp_path / "comparison.json"
    command = [sys.executable, "-B", "-m", "scripts.hpc.probe_shared_serialized_dual"]
    for name in ("dataset-manifest", "acceptance-report", "base-statistics",
                 "process-statistics", "fast-statistics", "prepared-manifest",
                 "joint-admission-report", "validation-manifest", "source-profile"):
        command.extend((f"--{name}", str(tmp_path / f"missing-{name}.json")))
    command.extend(("--output", str(output), "--input-bundle", str(tmp_path / "missing.pkl"),
                    "--daily-counts", "32", "--comparison-only", "--prepare-input-bundle"))
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert "comparison-only cannot prepare or resume training" in completed.stderr
    assert not output.exists(), "conflicting comparison entered preparation"
