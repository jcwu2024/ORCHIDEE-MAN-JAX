"""Bounded performance evidence never promotes failed scientific consistency."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dev import summarize_global_server_cost as cost

_spec = importlib.util.spec_from_file_location("server_cost_fixtures", Path(__file__).with_name("test_global_server_benchmark.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
cost_fixture = _fixtures.cost_fixture


def exploratory_fixture(directory):
    directory.mkdir()
    name, manifest, report = cost_fixture(directory)
    manifest["identity"]["validation_mode"] = "exploratory"
    record = manifest["stages"][name]
    record.update(measurement_status="complete", evidence_class="exploratory_performance", strict_numerical_passed=False)
    record["provenance"].update(full_two_block_workload=True, full_two_block_parity=False, cross_mode_scalar_parity=False)
    record["gates"][0].update(passed=False, compared_leaves=1, max_abs=1e-6, max_bound_ratio=2.0,
        atol=1e-8, rtol=1e-10,
        failures=[{"path": "leaf", "reason": "tolerance", "max_bound_ratio": 2.0}])
    report["timing"] = {"server_scalar_once_lower_seconds": 3, "server_scalar_once_compile_seconds": 4,
                        "B16_first_call_seconds": 5}
    return name, manifest, report


def save(directory, manifest, report):
    (directory / "stage_manifest.json").write_text(json.dumps(manifest))
    (directory / "report.json").write_text(json.dumps(report))


def test_failed_GPU_parity_is_separate_complete_performance(tmp_path):
    cpu = tmp_path / "cpu"
    name, manifest, report = exploratory_fixture(cpu)
    with pytest.raises(ValueError, match="cannot be promoted"):
        cost.accept_stage(cpu, name, manifest, report)
    save(cpu, manifest, report)
    result = cost.summarize([cpu], [])
    assert not result["accepted_stages"] and not result["same_workload_comparisons"]
    assert result["status"] == "exploratory_performance_measured"
    assert not result["withheld_stages"]
    stage = result["exploratory_stages"][0]
    assert stage["scientific_acceptance"] is False and stage["strict_numerical_passed"] is False
    assert stage["failed_gates"] == [manifest["stages"][name]["gates"][0]]
    assert stage["core"]["range_seconds"] == [2, 2]
    assert stage["startup_and_compile_timing"] == report["timing"]
    assert result["pricing"]["primary"]["CPU_CNY_per_allocated_count_hour"] == .05
    job = {"directories": [str(cpu.resolve())], "status": "completed", "wall_seconds": 100,
           "allocated_CPU_count": 4, "GPU_count": 0, "provenance": "observed allocation"}
    observed = cost.summarize([cpu], [job])["exploratory_stages"][0]["cost_scenarios"]["primary"]
    assert observed["aggregate_benchmark_job_cost"]["wall_seconds"] == 100
    assert "CNY_per_common_workload" not in observed["aggregate_benchmark_job_cost"]
    assert observed["aggregate_benchmark_job_cost"]["CNY"] > observed["core"]["CNY"]


@pytest.mark.parametrize("failure", ["schema", "nonfinite", "internal", "sample", "incomplete", "status", "SHA"])
def test_hard_failures_have_no_exploratory_measurement(tmp_path, failure):
    cpu = tmp_path / "cpu"
    name, manifest, report = exploratory_fixture(cpu)
    record = manifest["stages"][name]
    gate = record["gates"][0]
    if failure == "schema":
        gate["failures"][0]["reason"] = "schema"
    elif failure == "nonfinite":
        gate["max_abs"] = float("inf")
    elif failure == "internal":
        record["gates"][-1].update(passed=False, failures=[{"reason": "tolerance"}])
    elif failure == "sample":
        record["timing"]["B16_warm_frozen_state_core"]["samples_seconds"][0] = float("nan")
    elif failure == "incomplete":
        record["timing"] = {}
    elif failure == "status":
        record["strict_numerical_passed"] = True
    else:
        record["sha256"] = "b" * 64
    save(cpu, manifest, report)
    result = cost.summarize([cpu], [])
    assert not result["accepted_stages"] and not result["exploratory_stages"]
    assert len(result["withheld_stages"]) == 1


def test_exploratory_ratios_require_exact_workload_and_exclude_winner(tmp_path):
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    name, manifest, report = exploratory_fixture(cpu)
    save(cpu, manifest, report)
    gpu.mkdir()
    gpu_name, gpu_manifest, gpu_report = cost_fixture(gpu, "gpu")
    save(gpu, gpu_manifest, gpu_report)
    result = cost.summarize([cpu, gpu], [])
    assert len(result["accepted_stages"]) == len(result["exploratory_stages"]) == 1
    assert not result["same_workload_comparisons"]
    comparison = result["exploratory_same_workload_comparisons"][0]
    assert comparison["scientific_acceptance"] is False
    assert "GPU_cheaper_per_warm_workload_primary" not in comparison
    assert comparison["measured_throughput_ratio"] == 1
    gpu_manifest["identity"]["lanes"][0]["native_input"] = "different"
    save(gpu, gpu_manifest, gpu_report)
    assert not cost.summarize([cpu, gpu], [])["exploratory_same_workload_comparisons"]


def test_all_passing_exploratory_remains_exploratory(tmp_path):
    cpu = tmp_path / "cpu"
    name, manifest, report = exploratory_fixture(cpu)
    record = manifest["stages"][name]
    record["gates"][0] = {"name": record["gates"][0]["name"], "passed": True}
    record["strict_numerical_passed"] = True
    record["provenance"]["full_two_block_parity"] = True
    save(cpu, manifest, report)
    result = cost.summarize([cpu], [])
    assert not result["accepted_stages"]
    assert result["exploratory_stages"][0]["strict_numerical_passed"] is True
    assert result["exploratory_stages"][0]["scientific_acceptance"] is False


@pytest.mark.parametrize("fault", [None, "role", "source", "cross_mode"])
def test_pool_internal_scalar_reference_role_is_strict(tmp_path, fault):
    import copy

    cpu = tmp_path / "cpu"
    name, manifest, report = exploratory_fixture(cpu)
    identity = manifest["identity"]
    source = copy.deepcopy(identity)
    source.update(runtime=_fixtures.toy_environment(), cpu_reference_role="accepted")
    identity.update(mode="pool", cpu_reference_role="cpu-internal")
    record = manifest["stages"].pop(name)
    name = "B16_server_pool_v1"
    manifest["stages"][name] = record
    for gate in record["gates"]:
        gate["name"] = gate["name"].replace("_scalar_", "_pool_")
    record["gates"].append({"name": "B16_pool_cross_mode_server_scalar", "passed": True})
    proof = record["provenance"]
    proof.update(cross_mode_scalar_parity=True, CPU_scalar_reference_role="cpu_internal_comparison",
        CPU_scalar_reference={"source_identity": source, "role": "cpu_internal_comparison",
            "native_input_identity_verified": True, "fixed_gate": cost.FIXED_GATE,
            "point_ids": proof["point_ids"]})
    if fault == "role":
        identity["cpu_reference_role"] = "accepted"
    elif fault == "source":
        source["mode"] = "batch"
    elif fault == "cross_mode":
        proof["cross_mode_scalar_parity"] = False
    save(cpu, manifest, report)
    result = cost.summarize([cpu], [])
    assert len(result["exploratory_stages"]) == (1 if fault is None else 0)
    assert len(result["withheld_stages"]) == (0 if fault is None else 1)


def test_runner_scalar_complete_record_is_admitted_without_cross_mode(monkeypatch, tmp_path):
    args, recorder, lanes, second = _fixtures.toy_cpu_case(monkeypatch, tmp_path / "runner", gpu_change=_fixtures.change_gpu_x)
    # Keep the toy transition but declare the complete fixed comparator policy.
    identity = _fixtures.toy_identity(lanes)
    identity["gate"] = cost.FIXED_GATE.copy()
    monkeypatch.setattr(_fixtures.scale.base, "stage_identity", lambda *args: identity.copy())
    _fixtures.run_toy_case(args, recorder, lanes, second)
    manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
    name = "B16_server_scalar_v1"
    record = manifest["stages"][name]
    assert record["provenance"]["cross_mode_scalar_parity"] is False
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir()
    _, _, report = cost_fixture(resource_dir)
    record["provenance"]["server_resources"] = report["server_resources"]
    report["timing"] = recorder.result["timing"]
    save(args.output_dir, manifest, report)
    result = cost.summarize([args.output_dir], [])
    assert not result["withheld_stages"]
    assert result["exploratory_stages"][0]["strict_numerical_passed"] is False


def test_strict_stage_cannot_promote_CPU_internal_reference(tmp_path):
    name, manifest, report = cost_fixture(tmp_path)
    manifest["identity"]["cpu_reference_role"] = "cpu-internal"
    with pytest.raises(ValueError, match="cannot be promoted"):
        cost.accept_stage(tmp_path, name, manifest, report)


def test_explicit_checkout_CLI_runs_from_unrelated_cwd(tmp_path):
    cpu = tmp_path / "cpu"
    _, manifest, report = exploratory_fixture(cpu)
    save(cpu, manifest, report)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    output = tmp_path / "summary.json"
    completed = subprocess.run([sys.executable, str(Path(cost.__file__).resolve()),
        "--cpu-dir", str(cpu), "--output", str(output)], cwd=unrelated,
        text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "exploratory_performance_measured"
    assert len(result["exploratory_stages"]) == 1 and not result["accepted_stages"]
