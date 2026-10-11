from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
from test_global_gpu_capacity_lifecycle import Arguments, DeviceLeaf, Executable, Recorder, Trace, arrays

from scripts.dev import benchmark_global_batch as actual_base
from scripts.dev import benchmark_global_gpu_capacity as capacity


def make_measure_inputs(monkeypatch, tmp_path, strategy="phased_release"):
    trace = Trace()
    monkeypatch.setattr(capacity.time, "perf_counter", lambda: trace.clock)
    lanes = [SimpleNamespace(landpoint_id=f"p{i}", arguments=Arguments((np.array([[i]], dtype=np.int32),),
        np.arange(2, 9, dtype=np.int32)[None, :])) for i in range(64)]
    second = [lane.arguments._replace(forcing_days=np.arange(9, 16, dtype=np.int32)[None, :]) for lane in lanes]
    expanded, others, _ = capacity.capacity_layout(lanes, second, 2048)
    state = np.arange(64, dtype=np.int32)[:, None]
    first_output = arrays(state, np.tile(np.arange(2, 9, dtype=np.int32), (64, 1)))
    reference = first_output, arrays(first_output[0][0], np.tile(np.arange(9, 16, dtype=np.int32), (64, 1)))
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

    monkeypatch.setattr(actual_base, "sync", sync)
    base = SimpleNamespace(batch=SimpleNamespace(stack_lanes=stack, make_batch_transition=lambda value: None),
        stack_second=stack_second, lower_compile=lower, run_two=actual_base.run_two, sync=sync,
        compare=actual_base.compare, driver=SimpleNamespace(_COMPILED_SECHIBA_SCAN_CACHE={}),
        rss_bytes=lambda: 100, linux_ru_maxrss_bytes=lambda: 200)
    scale = SimpleNamespace(base=base, np=np,
        jax=SimpleNamespace(tree_util=jax.tree_util, device_put=put, device_get=get),
        tree_bytes=lambda value: sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(value)),
        executable_memory=lambda executable: {"available": True, "temp_size_in_bytes": 10,
            "argument_size_in_bytes": 20, "alias_size_in_bytes": 0},
        visible_gpu_memory=lambda: {"applicable": True, "total_bytes": 32 * 1024**3,
            "free_bytes": 30 * 1024**3, "uuid": "GPU-test"})
    monkeypatch.setattr(capacity, "fixed2048_host_screen", lambda *values: {})
    args = SimpleNamespace(warm_repeats=5, measurement_strategy=strategy)
    memories = {1024: {"available": True, "temp_size_in_bytes": 15}}
    return trace, args, recorder, expanded, others, reference, scale, memories


def test_lifecycle_observer_runs_outside_timers_and_reports_root_ownership(monkeypatch, tmp_path):
    for strategy, after_owned in (("phased_resident_control", True), ("phased_release", False)):
        case_dir = tmp_path / strategy
        case_dir.mkdir()
        trace, args, recorder, expanded, others, reference, scale, memories = make_measure_inputs(
            monkeypatch, case_dir, strategy)
        observations = []

        def observe(event_name, state):
            observations.append((event_name, dict(state)))
            trace.clock += 50

        _, record = capacity.measure_shape(1024, args, recorder, expanded, others, reference, scale,
            memories, 2048, memory_observer=observe)
        transitions = [(name, state["logical_root_owned"]) for name, state in observations
            if name.startswith("core_roots_release_")]
        assert transitions == [("core_roots_release_before", True), ("core_roots_release_after", after_owned)]
        samples = [(name, state) for name, state in observations if name.startswith("wrapper_sample_")]
        assert len(samples) == 10
        assert [state["sample_index"] for name, state in samples if name.endswith("start")] == [1, 2, 3, 4, 5]
        assert all(state["boundary"] == "wrapper" for _, state in samples)
        assert all((state["logical_root_bytes"] > 0) is after_owned for _, state in samples)
        assert all(state["logical_root_owned"] is after_owned for _, state in samples)
        assert record["timing"]["core"]["samples_seconds"] == [7] * 5
        assert record["timing"]["wrapper"]["samples_seconds"] == [15] * 5


def test_validation_hooks_wrap_only_non_warm_calls(monkeypatch, tmp_path):
    trace, args, recorder, expanded, others, reference, scale, memories = make_measure_inputs(monkeypatch, tmp_path)
    calls, outputs = [], []

    def factory(executable):
        def checked(*values):
            calls.append((trace.boundary, trace.sample))
            return executable(*values)
        return checked

    result, _ = capacity.measure_shape(1024, args, recorder, expanded, others, reference, scale,
        memories, 2048, validation_sink=lambda event, tree: outputs.append((event, tree)),
        validation_executable_factory=factory)
    assert calls == [("initial", None)] * 4 + [("gate", None)] * 4
    assert [event for event, _ in outputs] == ["initial_output", "frozen_reset_output", "permutation_output", "replacement_output"]
    assert [jax.tree_util.tree_leaves(tree)[0].shape[0] for _, tree in outputs] == [2048, 2048, 1024, 1024]
    assert result["measurements"]["core"]["samples_seconds"] == [7] * 5
    assert result["measurements"]["wrapper"]["samples_seconds"] == [15] * 5


def test_memory_observer_uuid_failure_stops_before_wrapper_execution(monkeypatch, tmp_path):
    trace, args, recorder, expanded, others, reference, scale, memories = make_measure_inputs(
        monkeypatch, tmp_path)
    selected_uuid = "GPU-test"
    seen = []

    def observe(event_name, state):
        nonlocal selected_uuid
        snapshot_uuid = "GPU-test" if event_name == "core_roots_release_before" else "GPU-other"
        seen.append((event_name, snapshot_uuid, state))
        if snapshot_uuid != selected_uuid:
            raise MemoryError("selected GPU UUID changed during lifecycle observation")

    try:
        capacity.measure_shape(1024, args, recorder, expanded, others, reference, scale, memories, 2048,
            memory_observer=observe)
    except MemoryError as exc:
        assert "UUID changed" in str(exc)
    else:
        raise AssertionError("UUID mismatch must stop lifecycle measurement")
    assert [item[0] for item in seen] == ["core_roots_release_before", "core_roots_release_after"]
    assert not any(event[0] == "block" and event[1] == 9 for event in trace.events if event[0] == "wrapper")
