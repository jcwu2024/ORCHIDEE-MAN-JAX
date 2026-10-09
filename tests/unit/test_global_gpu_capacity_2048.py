"""Fresh-process2048 capacity protocol; tiny arrays and subprocess stand-ins only."""
import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.dev import benchmark_global_gpu_capacity as capacity
from scripts.hpc import global_gpu_capacity_2048_coordinator as coordinator


def stage_fixture(directory, shape, *, pid=None, start=None, failure=None):
    directory.mkdir(parents=True)
    identity = {"contract": capacity.FIXED2048_CONTRACT, "process_strategy": capacity.PROCESS_STRATEGY_2048,
        "stage_shape": shape, "total_lanes": 2048, "candidate_shapes": [1024, 2048], "warm_repeats": 5,
        "budget_seconds": 7200, "run_binding": {"Slurm_job_id": "test-job", "hostname": "gpu-node",
            "physical_gpu": {"uuid": "GPU-test"}, "allocator": "default", "preallocate": False,
            "window_start_epoch": float(os.environ.get("ORCHJAX_GPU_WINDOW_START_EPOCH", "1"))}}
    if failure == "common":
        identity["run_binding"]["physical_gpu"]["uuid"] = "different-GPU"
    name = f"capacity_B{shape}_fixed2048_fresh_v1"
    gates = [{"name": f"capacity_B{shape}_lane_{i}_source_{i % 64}_GPU_reference", "passed": True} for i in range(2048)]
    gates += [{"name": f"capacity_B{shape}_{suffix}", "passed": True} for suffix in ("frozen_reset_repeat", "permutation", "replacement_isolation")]
    if failure == "numerical":
        gates[0]["passed"] = False
    data = b"tiny-checkpoint-receipt"
    sha = hashlib.sha256(data).hexdigest()
    origin = {"pid": shape if pid is None else pid, "process_start_epoch": time.time() if start is None else start,
        "environment": {"git_head": "fixed-code", "hostname": "gpu-node"}}
    provenance = {"total_lanes": 2048, "lane_days": 28672, "unique_real_points": 64,
        "chunk_size": shape, "chunks_per_collection": 2048 // shape, "blocks": [[2, 8], [9, 15]],
        "half_hours_per_day": 48, "full_outputs": True, "full_two_block_workload": True,
        "process_strategy": capacity.PROCESS_STRATEGY_2048,
        "full_output_native_schema": [{"shape": [2048, 1], "dtype": "float64"}] * 760,
        "executable_memory_analysis": {"available": True, "temp_size_in_bytes": 100},
        "admission_compiler_memory_records": {"1024": {"available": True, "temp_size_in_bytes": 150}},
        "memory_preflight": {"resident_2048_input_bytes": 100, "full_2048_output_bytes": 200},
        "host_peak_RSS_bytes": 300}
    stage = {"status": "complete", "measurement_status": "complete", "strict_numerical_passed": failure != "numerical",
        "gates": gates, "path": f"stage_{name}.pkl", "sha256": sha, "checkpoint_io": {"bytes": len(data)},
        "provenance": provenance, "execution_origin": origin,
        "timing": {key: capacity.lane_metrics([1, 2, 3, 4, 5], 2048) for key in ("core", "wrapper")}}
    manifest = {"identity": identity, "stages": {name: stage}}
    report = {"status": "completed_exploratory_capacity_screen", "completed_shapes": [shape],
        "stages": manifest["stages"], "environment": origin["environment"]}
    (directory / f"stage_{name}.pkl").write_bytes(data + (b"corrupt" if failure == "corrupt" else b""))
    (directory / "stage_manifest.json").write_text(json.dumps(manifest))
    (directory / "report.json").write_text(json.dumps(report))


def coordinate_args(tmp_path):
    return SimpleNamespace(output_dir=tmp_path / "capacity", budget_seconds=7200, dry_run=False,
        child_command=["fake-srun", "fake-python", "capacity.py", "--fixed-2048", "--output-dir", "replaced"])


@pytest.mark.parametrize("failure", [None, "exit", "missing", "corrupt", "common", "same-pid", "older-start", "same-start", "numerical", "timeout"])
def test_coordinator_sequential_fresh_children_preserve_control_on_later_failure(monkeypatch, tmp_path, failure):
    args = coordinate_args(tmp_path)
    monkeypatch.delenv("ORCHJAX_GPU_WINDOW_START_EPOCH", raising=False)
    events = []
    def child(shape, command, remaining):
        events.append(("start", shape))
        assert command[command.index("--output-dir") + 1] == str(args.output_dir / f"B{shape}")
        if shape == 2048:
            prior = json.loads((args.output_dir / "orchestration.json").read_text())["children"]["1024"]
            assert prior["returncode"] == 0 and "exit_epoch" in prior and "sealed_receipt" in prior
            assert events[:2] == [("start", 1024), ("exit", 1024)]
            assert command[command.index("--control-stage-dir") + 1] == str(args.output_dir / "B1024")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, remaining)
            if failure == "exit":
                return {"returncode": 1, "command_process_pid": 9}
        if shape == 1024 or failure != "missing":
            stage_fixture(args.output_dir / f"B{shape}", shape,
                pid=1024 if shape == 2048 and failure == "same-pid" else shape,
                start=(100 if shape == 2048 and failure == "older-start" else
                    prior["worker_execution_origin"]["process_start_epoch"] if shape == 2048 and failure == "same-start" else None),
                failure=failure if shape == 2048 else None)
        events.append(("exit", shape))
        return {"returncode": 0, "command_process_pid": shape + 1}
    report = coordinator.coordinate(args, run_child=child, snapshot=lambda: {
        "GPU_uuid": "GPU-test", "GPU_free_bytes": 32 * 1024**3, "GPU_total_bytes": 32 * 1024**3, "MemAvailable_bytes": 100 * 1024**3})
    expected = [1024, 2048] if failure is None else [1024]
    assert report["completed_shapes"] == expected == report["comparison_shapes"]
    assert (args.output_dir / "B1024/stage_capacity_B1024_fixed2048_fresh_v1.pkl").is_file()
    assert report["status"] == ("completed_exploratory_capacity_screen" if failure is None else "candidate_failed")
    assert report["B2048_dispatch_decision"]["estimated_device_bytes"] == 1100


@pytest.mark.parametrize("failure", ["numerical", "corrupt", "exit"])
def test_invalid_control_never_dispatches_candidate(monkeypatch, tmp_path, failure):
    args = coordinate_args(tmp_path)
    monkeypatch.delenv("ORCHJAX_GPU_WINDOW_START_EPOCH", raising=False)
    events = []
    def child(shape, command, remaining):
        events.append(shape)
        stage_fixture(args.output_dir / f"B{shape}", shape, failure=failure)
        return {"returncode": 1 if failure == "exit" else 0, "command_process_pid": shape + 1}
    report = coordinator.coordinate(args, run_child=child, snapshot=lambda: pytest.fail("no candidate resource query"))
    assert events == [1024] and report["completed_shapes"] == []
    assert report["status"] == "control_failed"


@pytest.mark.parametrize("resource", ["gpu", "host", "time"])
def test_denied_candidate_preserves_sealed_control(monkeypatch, tmp_path, resource):
    args = coordinate_args(tmp_path)
    monkeypatch.delenv("ORCHJAX_GPU_WINDOW_START_EPOCH", raising=False)
    events = []
    def child(shape, command, remaining):
        events.append(shape)
        stage_fixture(args.output_dir / f"B{shape}", shape)
        if resource == "time":
            monkeypatch.setattr(coordinator.time, "time", lambda: float(os.environ["ORCHJAX_GPU_WINDOW_START_EPOCH"]) + 3601)
        return {"returncode": 0, "command_process_pid": shape + 1}
    snapshot = {"GPU_uuid": "GPU-test", "GPU_total_bytes": 32 * 1024**3, "GPU_free_bytes": 1 if resource == "gpu" else 32 * 1024**3,
        "MemAvailable_bytes": 1 if resource == "host" else 100 * 1024**3}
    report = coordinator.coordinate(args, run_child=child, snapshot=lambda: snapshot)
    assert events == [1024] and report["comparison_shapes"] == [1024]
    assert report["status"] == "candidate_denied"


def test_cpu_coordinator_dry_run_has_no_JAX_imports_or_output_creation(tmp_path):
    command = [sys.executable, "-B", str(Path(coordinator.__file__).resolve()), "--output-dir", str(tmp_path / "new"),
        "--budget-seconds", "7200", "--dry-run", "--", "fake-srun", "python", "capacity.py", "--fixed-2048", "--output-dir", "parent"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["JAX_imported"] is False and plan["model_execution"] is False
    assert plan["contract"] == capacity.FIXED2048_CONTRACT
    assert not (tmp_path / "new").exists()
    assert plan["commands"]["2048"][-2:] == ["--control-stage-dir", str(tmp_path / "new/B1024")]


@pytest.mark.parametrize("shape,chunks", [(1024, 2), (2048, 1)])
def test_shape_execution_uses_complete2048_native_outputs_and_only_own_compile(monkeypatch, shape, chunks):
    events = []
    lanes = [SimpleNamespace(landpoint_id=f"p{i}", arguments=(np.array([[i]], dtype=np.int32),
        np.array([[i % 2]], dtype=np.bool_))) for i in range(64)]
    second = [lane.arguments for lane in lanes]
    reference = tuple(np.concatenate([lane.arguments[i] for lane in lanes]) for i in range(2))
    expanded, others, origins = capacity.capacity_layout(lanes, second, 2048)
    def stack(items):
        return jax.tree_util.tree_map(lambda *parts: np.concatenate(parts), *(lane.arguments for lane in items))
    def lower(fn, arguments, label, recorder):
        events.append(("compile", arguments[0].shape[0]))
        recorder.result["timing"].update({label + "_lower_seconds": 1, label + "_compile_seconds": 2})
        return fn
    def run(executable, first, other):
        events.append(("run", first[0].shape[0]))
        return first
    base = SimpleNamespace(batch=SimpleNamespace(stack_lanes=stack, make_batch_transition=lambda lanes: None),
        stack_second=lambda lanes, other: jax.tree_util.tree_map(lambda *parts: np.concatenate(parts), *other),
        lower_compile=lower, run_two=run, sync=lambda *a: None,
        compare=lambda *a, **kw: {"passed": True, "compared_leaves": 2},
        driver=SimpleNamespace(_COMPILED_SECHIBA_SCAN_CACHE={}), rss_bytes=lambda: 100, linux_ru_maxrss_bytes=lambda: 200)
    fake_jax = SimpleNamespace(tree_util=jax.tree_util, device_put=lambda v: v, device_get=lambda v: v)
    scale = SimpleNamespace(base=base, jax=fake_jax, np=np,
        tree_bytes=lambda v: sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(v)),
        executable_memory=lambda e: {"available": True, "temp_size_in_bytes": 10},
        visible_gpu_memory=lambda: {"applicable": True, "total_bytes": 32 * 1024**3, "free_bytes": 30 * 1024**3, "uuid": "GPU-test"})
    recorder = SimpleNamespace(result={"gates": [], "timing": {}, "server_resources": {}, "environment": {}}, sentinel_paths={})
    monkeypatch.setattr(capacity, "fixed2048_host_screen", lambda *a: {})
    memories = {1024: {"available": True, "temp_size_in_bytes": 15}}
    value, record = capacity.measure_shape(shape, SimpleNamespace(warm_repeats=5), recorder,
        expanded, others, reference, scale, memories, 2048)
    assert [event for event in events if event[0] == "compile"] == [("compile", shape)]
    assert sum(event[0] == "run" for event in events) == 11 * chunks + 2
    for actual, expected in zip(value["outputs"], reference, strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, np.tile(expected, (32, 1)))
    assert len(record["gates"]) == 2051 and len(origins) == 2048 and origins[-1]["repeat_index"] == 31
    assert all(metric["lane_days"] == 28672 for metric in value["measurements"].values())
    assert "first_full2048_call_seconds" in record["startup"]
    assert memories[1024]["temp_size_in_bytes"] == 15
    assert capacity.TOTAL_LANES == 512


def historical_fixture(tmp_path, monkeypatch, failure=None):
    directory = tmp_path / "historical1024"
    directory.mkdir()
    expected = {"contract": capacity.FIXED1024_CONTRACT, "science_sha256": "exact-science",
        "actual_science_source_sha256": "exact-source", "execution_source_compatibility": "exact-certificate"}
    identity = expected.copy()
    if failure == "identity":
        identity["science_sha256"] = "changed"
    outputs = tuple(np.zeros((1024, 1), dtype=np.bool_ if i % 3 == 0 else np.int32 if i % 3 == 1 else np.float64) for i in range(760))
    reference = tuple(value[:64] for value in outputs)
    name = "capacity_B1024_fixed1024_v1"
    timings = {key: capacity.lane_metrics([1, 2, 3, 4, 5], 1024) for key in ("core", "wrapper")}
    proof = {"measurement_status": "complete", "strict_numerical_passed": True, "gates": [{"passed": True}] * 1027,
        "timing": timings, "execution_origin": {"pid": 91, "environment": {
            "git_head": capacity.B1024_CODE_SHA, "loadavg": (1., 2., 3.)}},
        "provenance": {"workload_kind": "capacity_screen_repeated_FULL64", "total_lanes": 1024,
            "chunk_size": 1024, "chunks_per_collection": 1, "lane_days": 14336, "unique_real_points": 64,
            "unique_point_days": 896, "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48,
            "full_outputs": True, "full_two_block_workload": True, "host_peak_RSS_bytes": capacity.B1024_PEAK_RSS_BYTES,
            "executable_memory_analysis": {"available": True, "temp_size_in_bytes": 4571419856}}}
    value = {"outputs": outputs, "measurements": timings}
    if failure == "schema":
        value["outputs"] = list(outputs)
    elif failure == "dtype":
        value["outputs"] = tuple(leaf.astype(np.float32) for leaf in outputs)
    elif failure == "gates":
        proof["gates"] = [{"passed": True}] * 1026
    elif failure == "failed-gate":
        proof["gates"] = [{"passed": True}] * 1026 + [{"passed": False}]
    elif failure == "code":
        proof["execution_origin"]["environment"]["git_head"] = "changed"
    elif failure == "timing":
        timings["core"]["median_seconds"] = 99
    elif failure == "memory":
        proof["provenance"]["executable_memory_analysis"]["available"] = False
    elif failure == "RSS":
        proof["provenance"]["host_peak_RSS_bytes"] = 1
    data = pickle.dumps({"identity": identity, "name": name, "status": "complete", "record": proof, "value": value})
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(capacity, "B1024_SHA256", "changed" if failure == "sha" else sha)
    monkeypatch.setattr(capacity, "B1024_BYTES", len(data) + int(failure == "bytes"))
    stage = {"status": "complete", "sha256": sha, **proof}
    report = {"status": "failed" if failure == "report" else "completed_exploratory_capacity_screen",
        "completed_shapes": [512, 1024], "stages": {name: stage}, "environment": proof["execution_origin"]["environment"]}
    (directory / f"stage_{name}.pkl").write_bytes(data)
    (directory / "stage_manifest.json").write_text(json.dumps({"identity": identity, "stages": {name: stage}}))
    (directory / "report.json").write_text(json.dumps(report))
    return directory, expected, reference


@pytest.mark.parametrize("failure", [None, "identity", "schema", "dtype", "gates", "failed-gate", "code",
    "timing", "memory", "RSS", "sha", "bytes", "report"])
def test_historical1024_evidence_exact_import_without_timing_or_source_write(monkeypatch, tmp_path, failure):
    directory, expected, reference = historical_fixture(tmp_path, monkeypatch, failure)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    if failure is not None:
        with pytest.raises(ValueError):
            capacity.import_b1024_evidence(directory, expected, tmp_path / "new", jax, reference=reference)
    else:
        origin = capacity.import_b1024_evidence(directory, expected, tmp_path / "new", jax, reference=reference)
        assert "timing" not in origin and "measurements" not in origin
        assert origin["historical_process_peak_RSS_bytes"] == 27865956352
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert not (tmp_path / "new").exists()


def runner_fixture(monkeypatch, tmp_path):
    from scripts.dev import global_server_reference as reference_module
    from scripts.dev import global_source_compatibility as source_module

    lanes = [SimpleNamespace(landpoint_id=f"p{i}", arguments=np.array([i], dtype=np.float64)) for i in range(64)]
    second = [lane.arguments for lane in lanes]
    reference = tuple(np.arange(64, dtype=np.int32).reshape(64, 1) for _ in range(760))
    source = {"science_sha256": "science", "actual_science_source_sha256": "actual", "execution_source_compatibility": "certificate"}
    monkeypatch.setattr(source_module, "source_identity", lambda *a: source.copy())
    monkeypatch.setattr(reference_module, "load_reference", lambda *a: (reference, {"sha256": "reference"}))
    monkeypatch.setattr(capacity, "historical_b1024_identity", lambda *a: {"old": "identity"})
    monkeypatch.setattr(capacity, "import_b1024_evidence", lambda *a, **kw: {"provenance": {
        "executable_memory_analysis": {"available": True, "temp_size_in_bytes": 100}}})
    monkeypatch.setattr(capacity, "fixed2048_host_screen", lambda *a: {"conservative_required_bytes": 64321847296})
    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    monkeypatch.setenv("ORCHJAX_GPU_WINDOW_START_EPOCH", str(time.time()))
    monkeypatch.setenv("ORCHJAX_GLOBAL_BATCH_PROCESS_START_EPOCH", str(time.time()))
    args = SimpleNamespace(stage_shape=1024, prepared_input=tmp_path / "prepared/input", full_reference_dir=tmp_path / "reference",
        b1024_evidence_dir=tmp_path / "history", output_dir=tmp_path / "capacity/B1024", budget_seconds=7200,
        control_stage_dir=None, baseline_ref="source", warm_repeats=5)
    environment = {"git_head": "fixed-code", "hostname": "gpu-node", "physical_gpu": {"uuid": "GPU-test"}}
    def recorder():
        resources = {"mode": "GPU_auxiliary", "allocated_CPU_request_count": 4,
            "observed_unique_physical_cores": 2, "original_affinity": [4, 5, 6, 7], "selected_logical_cpus": [4, 5, 6, 7],
            "hostname": "gpu-node", "CUDA_VISIBLE_DEVICES_preserved": "0", "topology": [{"logical_cpu": 4}],
            "Slurm": {"SLURM_JOB_ID": "test-job", "SLURM_JOB_PARTITION": "gnall", "SLURM_CPUS_PER_TASK": "4"},
            "thread_affinity_after_backend": {"expected_logical_cpus": [4, 5, 6, 7],
                "all_observed_thread_masks_within_declared_cores": True, "observed_thread_count": 1,
                "threads": [{"TID": os.getpid(), "allowed_logical_cpus": [4, 5, 6, 7]}]}}
        return SimpleNamespace(result={"environment": environment.copy(), "server_resources": resources,
            "gates": [], "timing": {}}, sentinel_paths={}, flush=lambda: None, phase=lambda *a, **kw: None)
    def stack(selected):
        # Native batching includes one shared scalar, not one copy per lane.
        return np.concatenate([lane.arguments for lane in selected]), np.array([42])
    base = SimpleNamespace(stage_identity=lambda *a: {"science_sha256": "science"},
        batch=SimpleNamespace(stack_lanes=stack), stack_second=lambda selected, other: (np.concatenate(other), np.array([42])),
        roughheight_sentinel_contract=lambda *a: {})
    scale = SimpleNamespace(base=base, jax=SimpleNamespace(tree_util=jax.tree_util), np=np, PRECISION_POLICY="native",
        tree_bytes=lambda value: sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(value)),
        visible_gpu_memory=lambda: {"applicable": True, "total_bytes": 32 * 1024**3, "free_bytes": 30 * 1024**3, "uuid": "GPU-test"})
    calls = []
    def measure(shape, args, rec, expanded, other, ref, scale, memories, total, prepared_host):
        assert total == 2048 and len(prepared_host[0]) == total // shape
        calls.append(shape)
        actual = tuple(np.tile(leaf, (32, 1)) for leaf in ref)
        names = [f"capacity_B{shape}_lane_{i}_source_{i % 64}_GPU_reference" for i in range(2048)]
        names += [f"capacity_B{shape}_{suffix}" for suffix in ("frozen_reset_repeat", "permutation", "replacement_isolation")]
        metrics = {key: capacity.lane_metrics([1] * 5, 2048) for key in ("core", "wrapper")}
        return {"outputs": actual, "measurements": metrics}, {"measurement_status": "complete", "strict_numerical_passed": True,
            "gates": [{"name": name, "passed": True} for name in names], "timing": metrics, "startup": {},
            "provenance": {"total_lanes": 2048, "lane_days": 28672, "unique_real_points": 64, "chunk_size": shape,
                "chunks_per_collection": 2048 // shape, "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48,
                "full_outputs": True, "full_two_block_workload": True, "executable_memory_analysis": {
                    "available": True, "temp_size_in_bytes": 50}, "host_peak_RSS_bytes": 200,
                "memory_preflight": rec.result["fixed2048_stage_decision"]["device_memory"]},
            "execution_origin": {"available": True, "pid": os.getpid(), "environment": rec.result["environment"]}}
    monkeypatch.setattr(capacity, "measure_shape", measure)
    return args, recorder, lanes, second, scale, calls


@pytest.mark.parametrize("denial", [None, "time", "gpu", "allocator"])
def test_fresh_control_admission_uses_actual_stacks_and_compiles_only_own_stage(monkeypatch, tmp_path, denial):
    args, factory, lanes, second, scale, calls = runner_fixture(monkeypatch, tmp_path)
    rec = factory()
    if denial == "time":
        monkeypatch.setenv("ORCHJAX_GPU_WINDOW_START_EPOCH", str(time.time() - 3601))
    elif denial == "gpu":
        scale.visible_gpu_memory = lambda: {"applicable": True, "total_bytes": 32 * 1024**3, "free_bytes": 1}
    elif denial == "allocator":
        monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    if denial is None:
        capacity.benchmark_fixed2048(args, rec, lanes, second, scale)
        assert calls == [1024]
        assert rec.result["completed_shapes"] == [1024]
        decision = rec.result["fixed2048_stage_decision"]
        assert decision["device_memory"]["resident_2048_input_bytes"] == 32768 + 32
        manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
        proof = manifest["stages"]["capacity_B1024_fixed2048_fresh_v1"]
        assert len(proof["provenance"]["full_output_native_schema"]) == 760
        assert proof["execution_origin"]["process_start_epoch"] == float(os.environ["ORCHJAX_GLOBAL_BATCH_PROCESS_START_EPOCH"])
        assert proof["execution_origin"]["process_start_epoch_source"] == "GPU entry main before JAX/backend initialization"
        assert manifest["identity"]["run_binding"]["window_start_epoch"] == float(os.environ["ORCHJAX_GPU_WINDOW_START_EPOCH"])
    else:
        with pytest.raises((ValueError, MemoryError)):
            capacity.benchmark_fixed2048(args, rec, lanes, second, scale)
        assert calls == [] and not (args.output_dir / "stage_manifest.json").exists()


@pytest.mark.parametrize("failure", [None, "not-exited", "same-pid", "same-start", "proof-drift"])
def test_candidate_reloads_and_verifies_sealed_control_in_later_worker(monkeypatch, tmp_path, failure):
    args, factory, lanes, second, scale, calls = runner_fixture(monkeypatch, tmp_path)
    first_start = time.time()
    monkeypatch.setenv("ORCHJAX_GLOBAL_BATCH_PROCESS_START_EPOCH", str(first_start))
    monkeypatch.setattr(capacity.os, "getpid", lambda: 1024)
    first = factory()
    capacity.benchmark_fixed2048(args, first, lanes, second, scale)
    first.result["status"] = "completed_exploratory_capacity_screen"
    (args.output_dir / "report.json").write_text(json.dumps(first.result))
    control_dir = args.output_dir
    manifest = json.loads((control_dir / "stage_manifest.json").read_text())
    sha = manifest["stages"]["capacity_B1024_fixed2048_fresh_v1"]["sha256"]
    (control_dir.parent / "orchestration.json").write_text(json.dumps({"children": {"1024": {
        "returncode": None if failure == "not-exited" else 0, "exit_epoch": first_start + .01,
        "sealed_receipt": {"sha256": sha}}}}))
    if failure == "proof-drift":
        manifest["stages"]["capacity_B1024_fixed2048_fresh_v1"]["provenance"]["host_peak_RSS_bytes"] = 201
        (control_dir / "stage_manifest.json").write_text(json.dumps(manifest))
        first.result["stages"] = manifest["stages"]
        (control_dir / "report.json").write_text(json.dumps(first.result))
    monkeypatch.setattr(capacity.os, "getpid", lambda: 1024 if failure == "same-pid" else 2048)
    monkeypatch.setenv("ORCHJAX_GLOBAL_BATCH_PROCESS_START_EPOCH", str(first_start if failure == "same-start" else first_start + 1))
    args.stage_shape, args.control_stage_dir, args.output_dir = 2048, control_dir, control_dir.parent / "B2048"
    second_recorder = factory()
    if failure is None:
        capacity.benchmark_fixed2048(args, second_recorder, lanes, second, scale)
        assert calls == [1024, 2048]
        assert second_recorder.result["control_stage_origin"]["sha256"] == sha
        assert second_recorder.result["completed_shapes"] == [2048]
        next_manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
        proof = next_manifest["stages"]["capacity_B2048_fixed2048_fresh_v1"]
        assert proof["provenance"]["admission_compiler_memory_records"]["1024"]["temp_size_in_bytes"] == 100
        assert first.result["server_resources"]["thread_affinity_after_backend"]["threads"] != second_recorder.result["server_resources"]["thread_affinity_after_backend"]["threads"]
        first_identity = manifest["identity"]["run_binding"]["server_resources"]
        assert first_identity == next_manifest["identity"]["run_binding"]["server_resources"]
        assert "threads" not in first_identity["backend_thread_affinity_contract"]
    else:
        with pytest.raises(ValueError):
            capacity.benchmark_fixed2048(args, second_recorder, lanes, second, scale)
        assert calls == [1024] and not (args.output_dir / "stage_manifest.json").exists()
    assert (control_dir / "stage_capacity_B1024_fixed2048_fresh_v1.pkl").is_file()


@pytest.mark.parametrize("epoch,pid", [(float("nan"), 1), (float("inf"), 1), (0, 1), (1, 0), (1, True)])
def test_coordinator_rejects_invalid_worker_start_and_pid(tmp_path, epoch, pid):
    stage_fixture(tmp_path / "B1024", 1024, start=epoch, pid=pid)
    with pytest.raises(ValueError, match="PID/process-start"):
        coordinator.validate_stage_receipt(tmp_path / "B1024", 1024)


@pytest.mark.parametrize("shape,extra", [(1024, []), (2048, ["--control-stage-dir", "control"])])
def test_child_python_dry_run_immutable_contract_without_JAX_or_inputs(tmp_path, shape, extra):
    command = [sys.executable, "-B", str(Path(capacity.__file__).resolve()), "--prepared-input", "missing-input",
        "--full-reference-dir", "missing-ref", "--memory-baseline-dir", "unused-B64", "--output-dir", "unused-output",
        "--fixed-2048", "--stage-shape", str(shape), "--b1024-evidence-dir", "missing-history",
        "--budget-seconds", "7200", "--dry-run", *extra]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    assert value["contract"] == capacity.FIXED2048_CONTRACT and value["stage_shape"] == shape
    assert value["lane_days"] == 28672 and value["minimum_remaining_seconds"] == 3600
    assert value["JAX_imported"] is False and value["model_execution"] is False
    assert value["initial_conservative_host_required_bytes"] == 64321847296
    assert not (tmp_path / "unused-output").exists()


@pytest.mark.parametrize("extra", [["--fixed-1024"], ["--continue-512"], ["--optional-b64-baseline"],
    ["--conditional-512"], ["--admit-inputs"], ["--warm-repeats", "4"], ["--budget-seconds", "5400"],
    ["--control-stage-dir", "unexpected-control"]])
def test_child_rejects_mixed_or_incomplete_contract_before_JAX(tmp_path, extra):
    command = [sys.executable, "-B", str(Path(capacity.__file__).resolve()), "--prepared-input", "missing-input",
        "--full-reference-dir", "missing-ref", "--memory-baseline-dir", "unused-B64", "--output-dir", "unused-output",
        "--fixed-2048", "--stage-shape", "1024", "--b1024-evidence-dir", "missing-history",
        "--budget-seconds", "7200", "--dry-run", *extra]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and not (tmp_path / "unused-output").exists()


@pytest.mark.parametrize("changed", ["selected_logical_cpus", "hostname", "Slurm", "CUDA_VISIBLE_DEVICES_preserved"])
def test_stable_allocation_identity_retains_material_boundaries(changed):
    original = {"mode": "GPU_auxiliary", "allocated_CPU_request_count": 4, "selected_logical_cpus": [4, 5, 6, 7],
        "hostname": "node", "Slurm": {"SLURM_JOB_ID": "job"}, "CUDA_VISIBLE_DEVICES_preserved": "0"}
    different = {**original, changed: "changed"}
    assert capacity.stable_resource_identity(original) != capacity.stable_resource_identity(different)


def test_actual_cpu_subprocess_dispatch_waits_and_records_distinct_workers(monkeypatch, tmp_path):
    """Exercise Popen/wait/reap with a stdlib-only receipt worker, no model/JAX."""
    worker = tmp_path / "receipt_worker.py"
    code = (
        "import argparse,hashlib,json,os,sys,time\nfrom pathlib import Path\n"
        f"sys.path.insert(0,{str(capacity.ROOT)!r})\n"
        "from scripts.dev import benchmark_global_gpu_capacity as capacity\n"
        + inspect.getsource(stage_fixture)
        + "\np=argparse.ArgumentParser();p.add_argument('--output-dir');p.add_argument('--stage-shape',type=int)\n"
        "a,unused=p.parse_known_args()\nstage_fixture(Path(a.output_dir),a.stage_shape,pid=os.getpid(),start=time.time())\n"
    )
    worker.write_text(code)
    monkeypatch.delenv("ORCHJAX_GPU_WINDOW_START_EPOCH", raising=False)
    args = coordinate_args(tmp_path)
    args.child_command = [sys.executable, "-B", str(worker), "--fixed-2048", "--output-dir", "replaced"]
    report = coordinator.coordinate(args, snapshot=lambda: {"GPU_uuid": "GPU-test", "GPU_total_bytes": 32 * 1024**3,
        "GPU_free_bytes": 32 * 1024**3, "MemAvailable_bytes": 100 * 1024**3})
    assert report["completed_shapes"] == [1024, 2048]
    first, second = (report["children"][str(shape)] for shape in (1024, 2048))
    assert first["command_process_pid"] == first["worker_execution_origin"]["pid"]
    assert second["command_process_pid"] == second["worker_execution_origin"]["pid"]
    assert first["command_process_pid"] != second["command_process_pid"]
    assert first["exit_epoch"] <= second["worker_execution_origin"]["process_start_epoch"]


def test_postcompile_uses_original_whole_stage_budget_without_double_counting_placed_inputs():
    mib = 1024**2
    memories = {1024: {"available": True, "temp_size_in_bytes": 20 * mib}}
    pre = {"physical_gpu": {"applicable": True, "uuid": "GPU-test", "total_bytes": 32 * 1024**3,
        "free_bytes": 3 * 1024**3}}
    scale = SimpleNamespace(visible_gpu_memory=lambda: {"applicable": True, "uuid": "GPU-test",
        "total_bytes": 32 * 1024**3, "free_bytes": 500 * mib})
    result = capacity.fixed2048_postcompile_screen(2048, 100 * mib, 100 * mib, memories, scale, pre,
        {"available": True, "temp_size_in_bytes": 40 * mib})
    assert result["permitted"] is True
    assert result["estimated_device_bytes"] == 540 * mib
    assert result["original_preallocation_physical_gpu_budget"]["free_bytes"] == 3 * 1024**3
    assert result["postcompile_physical_gpu_snapshot"]["free_bytes"] == 500 * mib
    assert result["required_physical_slack_bytes"] == 2 * 1024**3
    # The legacy residual-free comparison stays unchanged, and would reject.
    with pytest.raises(MemoryError):
        capacity.device_screen(2048, 100 * mib, 100 * mib, memories, scale)


@pytest.mark.parametrize("failure", ["larger-temp", "missing-analysis", "negative-temp", "different-card", "different-total"])
def test_postcompile_fails_closed_for_larger_or_unknown_actual_bound(failure):
    mib = 1024**2
    actual = {"available": True, "temp_size_in_bytes": 40 * mib}
    if failure == "larger-temp":
        actual["temp_size_in_bytes"] = 1024 * mib
    elif failure == "missing-analysis":
        actual = {"available": False}
    elif failure == "negative-temp":
        actual["temp_size_in_bytes"] = -1
    memories = {2048: actual}
    pre = {"physical_gpu": {"applicable": True, "uuid": "GPU-test", "total_bytes": 32 * 1024**3, "free_bytes": 3 * 1024**3}}
    physical = {"applicable": True, "uuid": "other-GPU" if failure == "different-card" else "GPU-test",
        "total_bytes": 31 * 1024**3 if failure == "different-total" else 32 * 1024**3, "free_bytes": 500 * mib}
    with pytest.raises(MemoryError):
        capacity.fixed2048_postcompile_screen(2048, 100 * mib, 100 * mib, memories,
            SimpleNamespace(visible_gpu_memory=lambda: physical), pre, actual)
