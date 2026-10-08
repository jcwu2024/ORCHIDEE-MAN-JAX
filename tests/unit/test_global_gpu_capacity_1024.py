"""Fixed1024 protocol with tiny arrays/mocks; no real-model or GPU compile."""
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.dev import benchmark_global_batch as base
from scripts.dev import benchmark_global_gpu_capacity as capacity


def tiny(monkeypatch):
    lanes = [SimpleNamespace(landpoint_id=f"p{i}", arguments=np.array([[i]], dtype=np.float64)) for i in range(64)]
    second = [lane.arguments + 1 for lane in lanes]
    reference = (np.arange(64, dtype=np.float64).reshape(64, 1), np.arange(64, dtype=np.float64).reshape(64, 1) + 1)
    events = []
    recorder = SimpleNamespace(result={"gates": [], "timing": {}, "server_resources": {}, "environment": {}},
        sentinel_paths={}, phase=lambda *a, **kw: None, flush=lambda: None)
    def lower(fn, first, label, recorder):
        events.append(("compile", len(first)))
        recorder.result["timing"].update({label + "_lower_seconds": 1, label + "_compile_seconds": 1})
        return fn
    def run(fn, first, other):
        events.append(("run", len(first)))
        return first.copy(), other.copy()
    def stack(selected):
        return np.concatenate([lane.arguments for lane in selected])
    current = SimpleNamespace(batch=SimpleNamespace(stack_lanes=stack, make_batch_transition=lambda lanes: None),
        stack_second=lambda lanes, other: np.concatenate(other), lower_compile=lower, run_two=run,
        sync=lambda *a: None, compare=base.compare, driver=SimpleNamespace(_COMPILED_SECHIBA_SCAN_CACHE={}),
        linux_ru_maxrss_bytes=lambda: 1234, rss_bytes=lambda: 100,
        roughheight_sentinel_contract=lambda *a: {}, legacy_factory=lambda *a: None,
        stage_identity=lambda *a: {"science_sha256": "science"})
    scale = SimpleNamespace(base=current, jax=SimpleNamespace(device_put=lambda v: v, device_get=lambda v: v,
        tree_util=jax.tree_util, clear_caches=lambda: None), np=np, PRECISION_POLICY="native",
        tree_bytes=lambda v: sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(v)),
        executable_memory=lambda e: {"available": True, "temp_size_in_bytes": 10}, screen_inputs=lambda *a: {},
        visible_gpu_memory=lambda: {"applicable": True, "total_bytes": 32 * 1024**3, "free_bytes": 30 * 1024**3})
    monkeypatch.setattr(capacity, "fixed1024_host_screen", lambda *a: {"conservative_required_bytes": 47583821824})
    return lanes, second, reference, recorder, scale, events


@pytest.mark.parametrize("size,chunks", [(512, 2), (1024, 1)])
def test_new_shapes_measure_full1024_and_preserve_native_masks(monkeypatch, size, chunks):
    lanes, second, reference, recorder, scale, events = tiny(monkeypatch)
    expanded, others, origins = capacity.capacity_layout(lanes, second, 1024)
    memories = {512: {"available": True, "temp_size_in_bytes": 15}}
    value, record = capacity.measure_shape(size, SimpleNamespace(warm_repeats=5), recorder,
        expanded, others, reference, scale, memories, 1024)
    assert memories[512]["temp_size_in_bytes"] == 15
    assert record["provenance"]["executable_memory_analysis"]["temp_size_in_bytes"] == 10
    assert [event for event in events if event[0] == "compile"] == [("compile", size)]
    assert sum(event[0] == "run" for event in events) == 11 * chunks + 2
    assert len(origins) == 1024 and origins[-1]["repeat_index"] == 15
    for actual, expected in zip(value["outputs"], reference, strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, np.tile(expected, (16, 1)))
    assert len(record["gates"]) == 1027 and all(item["passed"] for item in record["gates"])
    assert record["provenance"]["total_lanes"] == 1024
    assert record["provenance"]["chunks_per_collection"] == chunks
    assert "first_full1024_call_seconds" in record["startup"] and "first_full512_call_seconds" not in record["startup"]
    assert "resident_1024_input_bytes" in record["provenance"]["memory_preflight"]
    assert all(metric["lane_days"] == 14336 for metric in value["measurements"].values())
    assert capacity.TOTAL_LANES == 512 and capacity.lane_metrics([1] * 5)["lane_days"] == 7168


def test_guards_require_time_and_sealed_new_record_without_speed_condition():
    new = {512: {"core": {"median_seconds": 99}}}
    sealed = {"capacity_B512_fixed1024_v1": {"status": "complete", "strict_numerical_passed": True,
        "gates": [{"passed": True}] * 1027}}
    assert capacity.fixed1024_decision(512, {}, {}, 2700, 100, 3 * 1024**3)["permitted"]
    assert capacity.fixed1024_decision(1024, new, sealed, 2700, 100, 3 * 1024**3)["permitted"]
    for records, stages, seconds, estimate, free in [({}, sealed, 2700, 100, 3 * 1024**3),
        (new, {}, 2700, 100, 3 * 1024**3), (new, sealed, 2699, 100, 3 * 1024**3),
        (new, sealed, 2700, None, 3 * 1024**3), (new, sealed, 2700, 100, 2 * 1024**3)]:
        assert not capacity.fixed1024_decision(1024, records, stages, seconds, estimate, free)["permitted"]


@pytest.mark.parametrize("second_policy", ["complete", "resource_denied", "numerical_denied"])
def test_curve_contains_only_new_measurements_and_seals_before_next(monkeypatch, tmp_path, second_policy):
    from scripts.dev import global_server_reference as reference_module
    from scripts.dev import global_source_compatibility as source_module

    lanes, second, reference, recorder, scale, events = tiny(monkeypatch)
    args = SimpleNamespace(output_dir=tmp_path / "new", full_reference_dir=tmp_path / "reference",
        prepared_input=tmp_path / "input/prepared.pkl", b512_evidence_dir=tmp_path / "historical512",
        budget_seconds=5400, baseline_ref="truth", fixed_1024=True, warm_repeats=5)
    monkeypatch.setattr(reference_module, "load_reference", lambda *a: (reference, {"sha256": "ref"}))
    monkeypatch.setattr(capacity, "b512_evidence_identity", lambda *a: {})
    monkeypatch.setattr(source_module, "source_identity", lambda *a: {"science_sha256": "science"})
    monkeypatch.setattr(capacity, "import_b512_evidence", lambda *a, **kw: {"provenance": {
        "executable_memory_analysis": {"available": True, "temp_size_in_bytes": 100}}})
    calls = []
    def measure(size, args, recorder, *other):
        assert other[-1] == 1024
        if size == 1024:
            assert (args.output_dir / "stage_capacity_B512_fixed1024_v1.pkl").is_file()
            assert recorder.result["completed_shapes"] == [512]
            assert other[-2][512]["temp_size_in_bytes"] == 100
            assert other[-2][512]["current_process_peak_RSS_bytes"] == 200
        calls.append(size)
        if second_policy == "resource_denied":
            scale.visible_gpu_memory = lambda: {"applicable": True, "total_bytes": 32 * 1024**3, "free_bytes": 1}
        return {"measurements": {k: capacity.lane_metrics([2] * 5, 1024) for k in ("core", "wrapper")}}, {
            "measurement_status": "complete", "strict_numerical_passed": second_policy != "numerical_denied",
            "gates": [{"passed": True}] * 1026 + [{"passed": second_policy != "numerical_denied"}],
            "provenance": {"executable_memory_analysis": {
                "available": True, "temp_size_in_bytes": 50}, "host_peak_RSS_bytes": 200}}
    monkeypatch.setattr(capacity, "measure_shape", measure)
    capacity.benchmark(args, recorder, lanes, second, scale)
    expected = [512, 1024] if second_policy == "complete" else [512]
    assert calls == expected == recorder.result["completed_shapes"] == recorder.result["comparison_shapes"]
    assert "comparison_shapes_including_imported_baseline" not in recorder.result
    manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
    assert manifest["identity"]["contract"] == capacity.FIXED1024_CONTRACT
    assert set(manifest["stages"]) == {f"capacity_B{size}_fixed1024_v1" for size in expected}
    assert manifest["identity"]["historical_B512_evidence_sha256"] == capacity.B512_SHA256
    decision = recorder.result["fixed1024_B1024_decision"]
    assert decision["compiler_memory_origin"]["current_memory_records"][512]["temp_size_in_bytes"] == 100
    assert "upper_bound" in " ".join(decision)
    if second_policy == "numerical_denied":
        assert any("strict numerical acceptance" in reason for reason in decision["reasons"])
        assert manifest["stages"]["capacity_B512_fixed1024_v1"]["strict_numerical_passed"] is False
    assert not any(event[0] == "compile" for event in events)


@pytest.mark.parametrize("extra", [["--continue-512"], ["--continue-256-512"], ["--optional-b64-baseline"],
    ["--conditional-512"], ["--budget-seconds", "2700"], ["--warm-repeats", "4"], ["--admit-inputs"]])
def test_rejects_mixed_protocol_before_JAX(extra, tmp_path):
    command = [sys.executable, str(Path(capacity.__file__).resolve()), "--prepared-input", "input",
        "--full-reference-dir", "ref", "--memory-baseline-dir", "unused-b64", "--output-dir", "out",
        "--fixed-1024", "--b512-evidence-dir", "old512", "--dry-run", *extra]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0


def test_host_projection_is_conservative_and_uses_actual_new_peak(monkeypatch):
    # Restore actual helper independently of tiny's stand-in; no /proc access.
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: "MemAvailable: 999999999 kB\n")
    first = capacity.fixed1024_host_screen()
    assert first["conservative_required_bytes"] == 47583821824
    next_shape = capacity.fixed1024_host_screen(30 * 1024**3)
    assert next_shape["conservative_required_bytes"] == 68 * 1024**3


def test_python_dry_run_reads_no_prepared_or_b64_files_and_imports_no_JAX(tmp_path):
    command = [sys.executable, str(Path(capacity.__file__).resolve()), "--prepared-input", "nonexistent-input",
        "--full-reference-dir", "nonexistent-ref", "--memory-baseline-dir", "nonexistent-b64",
        "--output-dir", "unused-output", "--fixed-1024", "--b512-evidence-dir", "nonexistent-history",
        "--dry-run"]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report["contract"] == capacity.FIXED1024_CONTRACT
    assert report["candidates"] == [512, 1024] and report["lane_days"] == 14336
    assert report["total_lanes"] == 1024 and report["minimum_remaining_seconds_each_stage"] == 2700
    assert report["initial_conservative_host_required_bytes"] == 47583821824
    assert report["JAX_imported"] is False and report["model_execution"] is False
    assert report["historical_B512_timing_is_curve_origin"] is False
    assert not (tmp_path / "unused-output").exists()


def test_compiler_estimate_uses_max_historical_and_current_shape():
    memories = {512: {"available": True, "temp_size_in_bytes": 30},
        1024: {"available": True, "temp_size_in_bytes": 50}}
    assert capacity.memory_estimate(1024, 100, 200, memories) == 2 * 100 + 3 * 200 + 60


@pytest.mark.parametrize("strict,gates", [(False, [{"passed": True}] * 1026 + [{"passed": False}]),
    (True, []), (True, [{"passed": True}] * 1026), (True, [{"passed": True}] * 1028)])
def test_completed_b512_with_failed_or_incomplete_numerical_gates_cannot_admit_b1024(strict, gates):
    stages = {"capacity_B512_fixed1024_v1": {"status": "complete", "strict_numerical_passed": strict, "gates": gates}}
    decision = capacity.fixed1024_decision(1024, {512: {}}, stages, 5400, 1, 3 * 1024**3)
    assert decision["permitted"] is False
    assert any("strict numerical acceptance" in reason for reason in decision["reasons"])


def historical_fixture(tmp_path, monkeypatch, failure=None):
    source = tmp_path / "history"
    source.mkdir()
    identity = dict.fromkeys(("contract", "baseline_ref", "science_sha256", "lanes", "runtime",
        "compiler_environment", "warm_repeats", "gate", "precision_policy", "validation_mode",
        "workload_kind", "total_lanes", "lane_origins", "unique_real_points", "candidate_shapes",
        "conditional_512", "optional_B64_baseline", "conditional_policy", "numerical_reference_sha256",
        "continuation_B128_sha256", "continuation_B256_sha256", "budget_seconds",
        "actual_science_source_sha256", "execution_source_compatibility"), "exact")
    identity.update(contract=capacity.B512_CONTRACT, actual_science_source_sha256=capacity.B512_ACTUAL_SOURCE_SHA256)
    measurements = {key: capacity.lane_metrics([1, 2, 3, 4, 5]) for key in ("core", "wrapper")}
    provenance = {"workload_kind": "capacity_screen_repeated_FULL64", "total_lanes": 512,
        "unique_real_points": 64, "chunk_size": 512, "chunks_per_collection": 1,
        "lane_days": 7168, "unique_point_days": 896, "blocks": [[2, 8], [9, 15]],
        "half_hours_per_day": 48, "full_outputs": True, "full_two_block_workload": True,
        "executable_memory_analysis": {"available": True, "temp_size_in_bytes": 123},
        "host_peak_RSS_bytes": capacity.B512_PEAK_RSS_BYTES}
    outputs = tuple(np.zeros((512, 1), dtype=np.float32 if i % 2 else np.int32) for i in range(760))
    reference = tuple(value[:64] for value in outputs)
    record = {"measurement_status": "complete", "strict_numerical_passed": True,
        "gates": [{"passed": True}] * 515, "timing": measurements, "provenance": provenance,
        "execution_origin": {"environment": {"git_head": capacity.B512_CODE_SHA, "loadavg": (1., 2., 3.)}}}
    name = "capacity_B512_fixed512_v1"
    envelope = {"name": name, "status": "complete", "identity": identity.copy(),
        "value": {"outputs": outputs, "measurements": measurements}, "record": record}
    if failure == "contract":
        identity["contract"] = "other"
    elif failure == "certificate":
        identity["execution_source_compatibility"] = "other"
    elif failure == "science":
        identity["science_sha256"] = "other"
    elif failure == "origin":
        record["execution_origin"]["environment"]["git_head"] = "other"
    elif failure == "gates":
        record["gates"] = [{"passed": True}] * 514
    elif failure == "native-dtype":
        envelope["value"]["outputs"] = tuple(value.astype(np.float64) for value in outputs)
    elif failure == "schema":
        envelope["value"]["outputs"] = list(outputs)
    elif failure == "timing":
        measurements["core"]["median_seconds"] = 99
    elif failure == "memory":
        provenance["executable_memory_analysis"]["available"] = False
    elif failure == "RSS":
        provenance["host_peak_RSS_bytes"] = 1
    data = pickle.dumps(envelope)
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(capacity, "B512_SHA256", "other" if failure == "checksum" else sha)
    monkeypatch.setattr(capacity, "B512_BYTES", len(data) + int(failure == "bytes"))
    stage = {"status": "complete", "sha256": sha, **record}
    manifest = {"identity": envelope["identity"], "stages": {name: stage}}
    report = {"status": "completed_exploratory_capacity_screen", "completed_shapes": [512],
        "stages": manifest["stages"], "environment": record["execution_origin"]["environment"],
        "gates": record["gates"], "linux_ru_maxrss_peak_bytes": capacity.B512_PEAK_RSS_BYTES}
    if failure == "report":
        report["status"] = "failed"
    (source / f"stage_{name}.pkl").write_bytes(data)
    (source / "stage_manifest.json").write_text(json.dumps(manifest))
    (source / "report.json").write_text(json.dumps(report))
    return source, identity, reference


def test_historical_import_returns_only_origin_and_preserves_read_only_source(monkeypatch, tmp_path):
    source, identity, reference = historical_fixture(tmp_path, monkeypatch)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    origin = capacity.import_b512_evidence(source, identity, tmp_path / "new", jax, reference=reference)
    assert "timing" not in origin and "measurements" not in origin
    assert origin["historical_process_peak_RSS_bytes"] == 19496943616
    assert origin["source_stage"] == "capacity_B512_fixed512_v1"
    assert not (tmp_path / "new").exists()
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}


@pytest.mark.parametrize("failure", ["contract", "certificate", "science", "origin", "gates", "native-dtype",
    "schema", "timing", "memory", "RSS", "checksum", "bytes", "report"])
def test_historical_import_rejects_non_exact_or_incomplete_evidence(monkeypatch, tmp_path, failure):
    source, identity, reference = historical_fixture(tmp_path, monkeypatch, failure)
    with pytest.raises(ValueError):
        capacity.import_b512_evidence(source, identity, tmp_path / "new", jax, reference=reference)
