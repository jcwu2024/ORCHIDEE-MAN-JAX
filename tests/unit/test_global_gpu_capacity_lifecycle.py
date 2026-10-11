"""Ownership and persistence with weakref-tracked stand-ins, no JIT or physics."""
from __future__ import annotations

import json
import weakref
from types import SimpleNamespace
from typing import NamedTuple

import jax
import numpy as np
import pytest

from scripts.dev import benchmark_global_batch as actual_base
from scripts.dev import benchmark_global_gpu_capacity as capacity
from scripts.dev.global_batch_stage_store import StageStore


class Arguments(NamedTuple):
    values_by_component: tuple
    forcing_days: object


class DeviceLeaf:
    def __init__(self, array, trace, kind):
        self.array = np.array(array, copy=True)
        trace.refs.append((kind, weakref.ref(self)))


class Trace:
    def __init__(self):
        self.clock = 0.
        self.refs = []
        self.events = []
        self.boundary = None
        self.sample = None
        self.fail_sample = None
        self.snapshots = []

    def event(self, name, *details):
        self.clock += 1
        self.events.append((self.boundary, self.sample, name, *details))

    def live(self, prefix):
        return sum(ref() is not None for kind, ref in self.refs if kind.startswith(prefix))


def arrays(state, forcing):
    next_state = state + forcing.sum(axis=1, keepdims=True)
    return (next_state,), {"mask": next_state % 2 == 0, "discrete": next_state.astype(np.int32),
        "native_float32": next_state.astype(np.float32), "native_float64": next_state.astype(np.float64)}


class Executable:
    def __init__(self, trace):
        self.trace = trace

    def __call__(self, state, forcing):
        trace = self.trace
        state = state[0].array if isinstance(state[0], DeviceLeaf) else state[0]
        forcing = forcing.array if isinstance(forcing, DeviceLeaf) else forcing
        day = int(forcing[0, 0])
        trace.event("block", day, int(state[0, 0]), int(state[-1, 0]))
        if trace.boundary == "wrapper" and trace.sample == trace.fail_sample and day == 9:
            raise MemoryError("injected second-block allocation failure")
        result = arrays(state, forcing)
        return jax.tree_util.tree_map(lambda leaf: DeviceLeaf(leaf, trace, "result_" + str(trace.boundary)), result)


class Recorder:
    def __init__(self, path, trace):
        self.path, self.trace = path, trace
        self.result = {"gates": [], "timing": {}, "server_resources": {}, "environment": {}, "phases": []}
        self.sentinel_paths = {}

    def phase(self, name, **details):
        trace = self.trace
        if name.endswith("_start"):
            trace.boundary = details.get("boundary")
            trace.sample = details.get("sample_index")
        self.result["phases"].append({"phase": name, **details})
        self.path.write_text(json.dumps(self.result), encoding="utf-8")
        # Large IO cost detects persistence accidentally moved inside timers.
        trace.clock += 100
        if name.endswith("_wrapper_start"):
            trace.snapshots.append({"sample": trace.sample, "resident": trace.live("resident"),
                "initial": trace.live("result_initial"), "core": trace.live("result_core"),
                "wrapper": trace.live("result_wrapper")})


def traced_case(monkeypatch, tmp_path, strategy="phased_release", actual_memory=None):
    trace = Trace()
    monkeypatch.setattr(capacity.time, "perf_counter", lambda: trace.clock)
    lanes = [SimpleNamespace(landpoint_id=f"p{i}", arguments=Arguments((np.array([[i]], dtype=np.int32),),
        np.arange(2, 9, dtype=np.int32)[None, :])) for i in range(64)]
    second = [lane.arguments._replace(forcing_days=np.arange(9, 16, dtype=np.int32)[None, :]) for lane in lanes]
    expanded, others, _ = capacity.capacity_layout(lanes, second, 2048)
    state = np.arange(64, dtype=np.int32)[:, None]
    first = arrays(state, np.tile(np.arange(2, 9, dtype=np.int32), (64, 1)))
    reference = first, arrays(first[0][0], np.tile(np.arange(9, 16, dtype=np.int32), (64, 1)))
    recorder = Recorder(tmp_path / "report.json", trace)

    def stack(selected):
        trace.event("stack_first")
        return jax.tree_util.tree_map(lambda *parts: np.concatenate(parts), *(lane.arguments for lane in selected))

    def stack_second(selected, arguments):
        trace.event("stack_second")
        return jax.tree_util.tree_map(lambda *parts: np.concatenate(parts), *arguments)

    def put(value):
        trace.event("H2D")
        kind = "wrapper_input" if trace.boundary == "wrapper" else "resident"
        return jax.tree_util.tree_map(lambda leaf: DeviceLeaf(leaf, trace, kind), value)

    def get(value):
        trace.event("D2H")
        return jax.tree_util.tree_map(lambda leaf: leaf.array.copy(), value)

    def sync(value):
        trace.event("sync")

    def lower(fn, arguments, label, rec):
        rec.result["timing"].update({label + "_lower_seconds": 1, label + "_compile_seconds": 2})
        return Executable(trace)

    # Exercise the real Python run_two state handoff, replacing only its sync.
    monkeypatch.setattr(actual_base, "sync", sync)
    base = SimpleNamespace(batch=SimpleNamespace(stack_lanes=stack, make_batch_transition=lambda lanes: None),
        stack_second=stack_second, lower_compile=lower, run_two=actual_base.run_two, sync=sync,
        compare=actual_base.compare, driver=SimpleNamespace(_COMPILED_SECHIBA_SCAN_CACHE={}),
        rss_bytes=lambda: 100, linux_ru_maxrss_bytes=lambda: 200)
    scale = SimpleNamespace(base=base, np=np,
        jax=SimpleNamespace(tree_util=jax.tree_util, device_put=put, device_get=get),
        tree_bytes=lambda value: sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(value)),
        executable_memory=lambda executable: actual_memory if actual_memory is not None else {
            "available": True, "temp_size_in_bytes": 10, "argument_size_in_bytes": 20, "alias_size_in_bytes": 0},
        visible_gpu_memory=lambda: {"applicable": True, "total_bytes": 32 * 1024**3,
            "free_bytes": 30 * 1024**3, "uuid": "GPU-test"})
    monkeypatch.setattr(capacity, "fixed2048_host_screen", lambda *a: {})
    args = SimpleNamespace(warm_repeats=5, measurement_strategy=strategy)
    memories = {1024: {"available": True, "temp_size_in_bytes": 15}}
    return trace, recorder, lambda: capacity.measure_shape(1024, args, recorder, expanded, others,
        reference, scale, memories, 2048), reference, scale


def assert_wrapper_events(events, chunks):
    assert [event[0] for event in events] == ["stack_first", "stack_second", "H2D", "block", "block", "sync", "D2H"] * chunks + ["sync"]


@pytest.mark.parametrize("strategy", capacity.MEASUREMENT_STRATEGIES)
def test_ownership_and_complete_wrapper_contract(monkeypatch, tmp_path, strategy):
    trace, recorder, measure, reference, _ = traced_case(monkeypatch, tmp_path, strategy)
    value, record = measure()
    starts = [(entry["boundary"], entry["sample_index"]) for entry in recorder.result["phases"]
        if entry["phase"].endswith(("_core_start", "_wrapper_start"))]
    expected = ([(boundary, i) for i in range(1, 6) for boundary in ("core", "wrapper")]
        if strategy == "legacy_interleaved" else [(boundary, i) for boundary in ("core", "wrapper") for i in range(1, 6)])
    assert starts == expected
    assert len(trace.snapshots) == 5
    for snapshot in trace.snapshots:
        assert snapshot["initial"] == snapshot["core"] == snapshot["wrapper"] == 0
        assert (snapshot["resident"] > 0) == (strategy != "phased_release")
    for i in range(1, 6):
        events = [event[2:] for event in trace.events if event[:2] == ("wrapper", i)]
        assert_wrapper_events(events, 2)
        blocks = [event for event in events if event[0] == "block"]
        assert [event[1] for event in blocks] == [2, 9, 2, 9]
        # Both chunks preserve origin order and use the Day8 final state.
        assert [(event[2], event[3]) for event in blocks] == [(0, 63), (35, 98)] * 2
    # Only execution/synchronization (7) and real wrapper operations (15) are timed.
    assert record["timing"]["core"]["samples_seconds"] == [7] * 5
    assert record["timing"]["wrapper"]["samples_seconds"] == [15] * 5
    assert len(record["gates"]) == 2051 and all(gate["passed"] for gate in record["gates"])
    assert record["provenance"]["measurement_strategy"] == strategy
    for actual, expected_leaf in zip(jax.tree_util.tree_leaves(value["outputs"]),
            jax.tree_util.tree_leaves(reference), strict=True):
        assert isinstance(actual, np.ndarray) and actual.dtype == expected_leaf.dtype
        np.testing.assert_array_equal(actual, np.tile(expected_leaf, (32, 1)))


def test_weakrefs_detect_a_stale_executor_owner(monkeypatch, tmp_path):
    trace, _, measure, _, scale = traced_case(monkeypatch, tmp_path)
    lower = scale.base.lower_compile
    held = []

    def stale_lower(fn, arguments, label, rec):
        held.append(arguments)  # Mutant: an additional owner defeats releasing pairs.
        return lower(fn, arguments, label, rec)

    scale.base.lower_compile = stale_lower
    measure()
    assert all(snapshot["resident"] > 0 for snapshot in trace.snapshots)
    held.clear()
    assert trace.live("resident") == 0


@pytest.mark.parametrize("mutant", ["reuse_resident", "omit_D2H", "omit_collection_sync"])
def test_wrapper_trace_detects_contract_mutants(monkeypatch, tmp_path, mutant):
    trace, _, _, _, scale = traced_case(monkeypatch, tmp_path)
    first = [SimpleNamespace(arguments=Arguments((np.array([[i]], dtype=np.int32),),
        np.arange(2, 9, dtype=np.int32)[None, :])) for i in range(2)]
    second = [lane.arguments._replace(forcing_days=np.arange(9, 16, dtype=np.int32)[None, :]) for lane in first]
    if mutant == "reuse_resident":
        resident = scale.jax.device_put((scale.base.batch.stack_lanes(first), scale.base.stack_second(first, second)))
        scale.jax.device_put = lambda value: resident
    elif mutant == "omit_D2H":
        scale.jax.device_get = lambda value: value
    else:
        scale.base.sync = lambda value: None
    trace.events.clear()
    trace.boundary, trace.sample = "wrapper", 1
    capacity._wrapper_collection(Executable(trace), [(first, second)], scale.base, scale.jax)
    with pytest.raises(AssertionError):
        assert_wrapper_events([event[2:] for event in trace.events], 1)


@pytest.mark.parametrize("sample", [1, 3])
def test_second_block_failure_retains_analysis_and_exact_sample_before_execution(monkeypatch, tmp_path, sample):
    trace, recorder, measure, _, _ = traced_case(monkeypatch, tmp_path)
    trace.fail_sample = sample
    store = StageStore(tmp_path / "stages", {"measurement_strategy": "phased_release"})
    with pytest.raises(MemoryError, match="second-block"):
        store.run("B1024", measure)
    saved = json.loads(recorder.path.read_text())
    diagnosis = saved["capacity_diagnostics"]["capacity_B1024"]
    assert diagnosis["executable_memory_analysis"]["temp_size_in_bytes"] == 10
    assert diagnosis["postcompile_memory_budget_validation"]["permitted"] is True
    assert diagnosis["post_transfer_GPU_memory"]["physical_gpu"]["uuid"] == "GPU-test"
    assert diagnosis["phase"] == "wrapper_start" and diagnosis["sample_index"] == sample
    assert saved["phases"][-1]["phase"] == "capacity_B1024_wrapper_start"
    assert saved["phases"][-1]["core_input_roots_retained"] is False
    assert store.manifest["stages"]["B1024"]["status"] == "failed"
    assert not store.checkpoint_path("B1024").exists()


@pytest.mark.parametrize("actual", [{"available": False, "reason": "unknown"},
    {"available": True}, {"available": True, "temp_size_in_bytes": -1},
    {"available": True, "temp_size_in_bytes": "10"}, {"available": True, "temp_size_in_bytes": True},
    {"available": True, "temp_size_in_bytes": 40 * 1024**3}])
def test_postcompile_denial_persists_obtained_analysis_and_guard(monkeypatch, tmp_path, actual):
    trace, recorder, measure, _, _ = traced_case(monkeypatch, tmp_path, actual_memory=actual)
    with pytest.raises(MemoryError, match="postcompile|compiler"):
        measure()
    diagnosis = json.loads(recorder.path.read_text())["capacity_diagnostics"]["capacity_B1024"]
    assert diagnosis["executable_memory_analysis"] == actual
    assert diagnosis["postcompile_memory_budget_validation"]["permitted"] is False
    assert diagnosis["postcompile_memory_budget_validation"]["actual_executable_memory_analysis"] == actual
    assert diagnosis["phase"] == "postcompile_guard"
    assert not any(event[2] == "block" for event in trace.events)


def test_posttransfer_identity_failure_is_persisted_before_compile(monkeypatch, tmp_path):
    trace, recorder, measure, _, scale = traced_case(monkeypatch, tmp_path)
    snapshot = scale.visible_gpu_memory()
    snapshots = iter([snapshot, {**snapshot, "uuid": "GPU-other"}])
    scale.visible_gpu_memory = lambda: next(snapshots)
    with pytest.raises(MemoryError, match="GPU UUID/total changed"):
        measure()
    diagnosis = json.loads(recorder.path.read_text())["capacity_diagnostics"]["capacity_B1024"]
    assert diagnosis["post_transfer_identity_matches"] is False
    assert diagnosis["post_transfer_GPU_memory"]["physical_gpu"]["uuid"] == "GPU-other"
    assert "executable_memory_analysis" not in diagnosis
    assert not any(event[2] == "block" for event in trace.events)


@pytest.mark.parametrize("strategy", ["phased_resident_control", "phased_release"])
@pytest.mark.parametrize("size,total", [(2048, 2048), (1024, 1024), (512, 2048)])
def test_phased_scope_is_fail_closed(strategy, size, total):
    with pytest.raises(ValueError, match="fresh fixed2048 B1024"):
        capacity.measurement_contract(SimpleNamespace(measurement_strategy=strategy), size=size, total_lanes=total)


def test_stage_identity_prevents_cross_strategy_resume(tmp_path):
    identity = capacity.measurement_contract(SimpleNamespace(measurement_strategy="phased_resident_control"),
        size=1024, total_lanes=2048)
    store = StageStore(tmp_path, identity)
    store.save("control", {"outputs": [1]}, {"measurement_status": "complete"})
    release = capacity.measurement_contract(SimpleNamespace(measurement_strategy="phased_release"), size=1024, total_lanes=2048)
    with pytest.raises(ValueError, match="identity changed"):
        StageStore(tmp_path, release)
    assert StageStore(tmp_path, identity).load("control")[0] == {"outputs": [1]}
