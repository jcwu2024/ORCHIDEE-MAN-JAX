from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.dev import benchmark_global_batch as batch_base
from scripts.dev import summarize_global_server_cost as cost
from scripts.dev.global_batch_stage_store import StageStore

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / "scripts/dev/collect_global_cpu48_b64.py"
SCALAR_STAGE = "B64_server_scalar_v1"
BATCH_STAGE = "B64_server_batch_v1"
REFERENCE_STAGE = "B4_group_references"
BASELINE = "133e35aec7b83b9d79888c46092f64d518e97299"
SCIENCE = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"
PRECISION = cost.PRECISION
GATE = cost.FIXED_GATE.copy()


def outputs():
    leaves = [np.full((64,), index / 10, dtype=np.float64) for index in range(760)]
    block = (tuple(leaves[:190]), tuple(leaves[190:380]))
    other = (tuple(leaves[380:570]), tuple(leaves[570:]))
    return (block, other)


def identity(mode, runtime, *, reference_sha, git_head):
    return {"contract": "server-CPU-same-workload-v1", "mode": mode,
        "CPU_budget": 1 if mode == "scalar" else 48,
        "validation_mode": "exploratory", "cpu_reference_role": "cpu-internal",
        "baseline_ref": BASELINE, "science_sha256": SCIENCE, "gate": GATE,
        "precision_policy": PRECISION, "numerical_reference_sha256": reference_sha,
        "parent_allocated_CPU_request_count": 48, "billing_CPU_request_count": 48,
        "runtime": {**runtime, "git_head": git_head},
        "lanes": [{"landpoint_id": f"point-{index:02d}", "inputs_sha256": f"input-{index:02d}"}
            for index in range(64)]}


def compare_gate(name, actual, reference):
    return {"name": name, **batch_base.compare(actual, reference)}


def write_stage(directory, name, stage_identity, value, record):
    StageStore(directory, stage_identity).save(name, value, record)
    return json.loads((directory / "stage_manifest.json").read_text())


def run_env():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def build_run(root):
    run_root = root / "run"
    scalar_dir, batch_dir, gpu_dir = run_root / "scalar", run_root / "batch", root / "gpu"
    scalar_dir.mkdir(parents=True)
    batch_dir.mkdir(parents=True)
    gpu_dir.mkdir(parents=True)
    raw_outputs = outputs()
    runtime_cpu = {"jax": "0.4.38", "jaxlib": "0.4.38", "numpy": "1.26.4",
        "x64_enabled": True, "device_details": [{"platform": "cpu", "kind": "synthetic CPU", "id": 0}]}
    runtime_gpu = {**runtime_cpu, "device_details": [{"platform": "gpu", "kind": "synthetic GPU", "id": 0}]}
    reference_sha = ""
    scalar_identity = identity("scalar", runtime_cpu, reference_sha=reference_sha, git_head="1" * 40)
    lane_ids = [lane["landpoint_id"] for lane in scalar_identity["lanes"]]
    scalar_gates = [compare_gate(f"B64_scalar_lane_{i}_{point}_GPU_reference",
        batch_base.jax.tree_util.tree_map(lambda leaf: np.asarray(leaf)[i], raw_outputs),
        batch_base.jax.tree_util.tree_map(lambda leaf: np.asarray(leaf)[i], raw_outputs))
        for i, point in enumerate(lane_ids)]
    scalar_timing = {key: {"samples_seconds": [1.0, 1.1, 0.9, 1.0, 1.05],
        "median_seconds": 1.0, "min_seconds": 0.9, "max_seconds": 1.1,
        "valid_point_days": 896} for key in (
            "B64_warm_frozen_state_core", "B64_preprepared_wrapper_stack_transfer_materialize")}
    scalar_timing["B64_warm_frozen_state_core"]["samples_seconds"] = [1.0] * 5
    scalar_timing["B64_preprepared_wrapper_stack_transfer_materialize"]["samples_seconds"] = [1.2] * 5
    scalar_timing["B64_preprepared_wrapper_stack_transfer_materialize"]["median_seconds"] = 1.2
    scalar_gates.append(compare_gate("B64_scalar_frozen_reset_repeat", raw_outputs, raw_outputs))
    scalar_resources = {"step_CPU_budget": 1, "allocated_CPU_request_count": 1,
        "parent_allocated_CPU_request_count": 48, "billing_CPU_request_count": 48,
        "Slurm": {"SLURM_CPUS_PER_TASK": "1", "ORCHJAX_PARENT_CPU_COUNT": "48",
            "ORCHJAX_SLURM_ALLOC_TRES": "cpu=48,mem=192G"}}
    scalar_record = {"status": "complete", "measurement_status": "complete",
        "evidence_class": "exploratory_performance", "strict_numerical_passed": True,
        "gates": scalar_gates, "timing": scalar_timing,
        "provenance": {"unique_real_points": 64, "point_ids": lane_ids,
            "precision_policy": PRECISION, "blocks": [[2, 8], [9, 15]],
            "half_hours_per_day": 48, "full_outputs": True,
            "full_two_block_workload": True, "full_two_block_parity": True,
            "frozen_reset_repeat": True, "five_or_more_synchronized_repeats": True,
            "server_resources": scalar_resources},
        "execution_origin": {"available": True, "environment": {**runtime_cpu, "loadavg": (0.2, 0.1, 0.05)}}}
    write_stage(scalar_dir, SCALAR_STAGE, scalar_identity,
        {"outputs": raw_outputs, "measurements": scalar_timing}, scalar_record)

    gpu_identity = identity("gpu-reference", runtime_gpu, reference_sha="pending", git_head="0" * 40)
    gpu_identity.update(contract="server-candidate-first-full-physics-v1", validation_mode="strict",
        cpu_reference_role="accepted", numerical_reference_sha256=None)
    gpu_record = {"status": "complete", "path": f"stage_{REFERENCE_STAGE}.pkl",
        "evidence_class": "accepted", "gates": [{"name": "synthetic-complete-reference", "passed": True}],
        "provenance": {"own_inputs": True, "point_ids": lane_ids},
        "execution_origin": {"available": True, "environment": runtime_gpu}}
    gpu_manifest = write_stage(gpu_dir, REFERENCE_STAGE, gpu_identity,
        {"outputs": raw_outputs}, gpu_record)
    reference_sha = gpu_manifest["stages"][REFERENCE_STAGE]["sha256"]

    scalar_identity["numerical_reference_sha256"] = reference_sha
    # Re-seal the scalar identity against the actual synthetic FULL64 reference.
    scalar_manifest_path = scalar_dir / "stage_manifest.json"
    scalar_manifest = json.loads(scalar_manifest_path.read_text())
    scalar_manifest["identity"]["numerical_reference_sha256"] = reference_sha
    scalar_payload = (scalar_dir / scalar_manifest["stages"][SCALAR_STAGE]["path"]).read_bytes()
    envelope = pickle.loads(scalar_payload)
    envelope["identity"]["numerical_reference_sha256"] = reference_sha
    scalar_payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    (scalar_dir / scalar_manifest["stages"][SCALAR_STAGE]["path"]).write_bytes(scalar_payload)
    scalar_manifest["stages"][SCALAR_STAGE]["sha256"] = hashlib.sha256(scalar_payload).hexdigest()
    scalar_manifest_path.write_text(json.dumps(scalar_manifest))
    scalar_identity = scalar_manifest["identity"]

    batch_identity = identity("batch", runtime_cpu, reference_sha=reference_sha, git_head="1" * 40)
    batch_identity.update(mode="batch", CPU_budget=48, thread_environment_before_JAX={
        "OMP_NUM_THREADS": "48", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true"})
    from scripts.dev.global_server_reference import load_reference
    _, scalar_source = load_reference(scalar_dir, batch_identity, name=SCALAR_STAGE,
        points=64, cpu_reference_role="cpu-internal")
    gates = [compare_gate(f"B64_batch_lane_{i}_{point}_GPU_reference",
        batch_base.jax.tree_util.tree_map(lambda leaf: np.asarray(leaf)[i], raw_outputs),
        batch_base.jax.tree_util.tree_map(lambda leaf: np.asarray(leaf)[i], raw_outputs))
        for i, point in enumerate(lane_ids)]
    batch_timing = {key: {"samples_seconds": [1.0] * 5, "median_seconds": 1.0,
        "valid_point_days": 896, "process_CPU_samples": [{
            "wall_seconds": 1.0, "process_CPU_user_seconds_delta": 40.0,
            "process_CPU_system_seconds_delta": 1.0, "process_CPU_user_system_seconds_delta": 41.0,
            "busy_core_equivalents": 41.0, "step_utilization_fraction": 41 / 48,
            "parent_utilization_fraction": 41 / 48, "step_CPU_budget": 48,
            "parent_CPU_budget": 48} for _ in range(5)]} for key in (
            "B64_warm_frozen_state_core", "B64_preprepared_wrapper_stack_transfer_materialize")}
    cross_gate = compare_gate("B64_batch_cross_mode_server_scalar", raw_outputs, raw_outputs)
    gates += [compare_gate("B64_batch_frozen_reset_repeat", raw_outputs, raw_outputs), cross_gate,
        compare_gate("B64_server_batch_cross_group_permutation", raw_outputs, raw_outputs),
        compare_gate("B64_server_batch_replacement_isolation", raw_outputs, raw_outputs)]
    observations = {phase: {"available": True, "RSS_bytes": 1024, "peak_RSS_bytes": 2048,
        "MemAvailable_bytes": 100 * 1024**3} for phase in (
            "pre_compile", "post_compile", "first_call_complete", "warm_complete")}
    observations["first_call_process_CPU"] = {"process_CPU_user_system_seconds_delta": 2.0,
        "step_CPU_budget": 48, "parent_CPU_budget": 48}
    observations["thread_affinity_after_compile"] = {"all_observed_thread_masks_within_declared_cores": True}
    observations["thread_affinity_after_warm"] = {"all_observed_thread_masks_within_declared_cores": True}
    topology = [{"package_id": 0, "physical_core_id": index} for index in range(48)]
    selected = list(range(48))
    batch_resources = {"step_CPU_budget": 48, "allocated_CPU_request_count": 48,
        "parent_allocated_CPU_request_count": 48, "billing_CPU_request_count": 48,
        "topology": topology, "selected_logical_cpus": selected,
        "thread_environment_before_JAX": batch_identity["thread_environment_before_JAX"],
        "Slurm": {"SLURM_JOB_ID": "synthetic", "SLURM_JOB_PARTITION": "cnall",
            "SLURM_CPUS_PER_TASK": "48", "ORCHJAX_PARENT_CPU_COUNT": "48",
            "ORCHJAX_SLURM_ALLOC_TRES": "cpu=48,mem=192G",
            "ORCHJAX_SLURM_ALLOC_TRES_SOURCE": "AllocTRES"}}
    batch_record = {"status": "complete", "measurement_status": "complete",
        "evidence_class": "exploratory_performance", "strict_numerical_passed": True,
        "gates": gates, "timing": batch_timing,
        "execution_origin": {"available": True, "environment": {**runtime_cpu, "loadavg": (0.2, 0.1, 0.05)}},
        "provenance": {"unique_real_points": 64, "point_ids": lane_ids,
            "precision_policy": PRECISION, "blocks": [[2, 8], [9, 15]],
            "half_hours_per_day": 48, "full_outputs": True,
            "full_two_block_workload": True, "full_two_block_parity": True,
            "frozen_reset_repeat": True, "cross_mode_scalar_parity": True,
            "five_or_more_synchronized_repeats": True, "cross_group_permutation": True,
            "replacement_isolation": True, "CPU_scalar_reference_role": "cpu_internal_comparison",
            "CPU_scalar_reference": {**scalar_source, "source_identity": scalar_identity,
                "role": "cpu_internal_comparison", "point_ids": lane_ids, "fixed_gate": GATE,
                "native_input_identity_verified": True}, "server_resources": batch_resources,
            "runtime_observations": observations}}
    write_stage(batch_dir, BATCH_STAGE, batch_identity, {"outputs": raw_outputs}, batch_record)
    report = {"runtime_observations": {"B64": observations},
        "timing": {key: value for key, value in batch_timing.items()}}
    (batch_dir / "report.json").write_text(json.dumps(report))
    (scalar_dir / "report.json").write_text(json.dumps({"runtime_observations": {}}))
    run_root.mkdir(exist_ok=True)
    orchestration = {"status": "completed_exploratory_measurement",
        "expected_execution_commit": "1" * 40, "expected_science_sha256": SCIENCE,
        "expected_gpu_reference_sha256": reference_sha,
        "allocation": {"parent_cpu_count": 48, "job_id": "synthetic-job"},
        "wall_seconds": 120, "coordinator_scope_primary_cost_cny_estimate": 0.08,
        "coordinator_scope_conservative_cost_cny_estimate": 0.112,
        "coordinator_scope_cost_limitation": "synthetic test interval",
        "commands": {"batch": {"argv": [sys.executable, str(COLLECTOR), "--gpu-reference-dir", str(gpu_dir),
            "--prepared-input", str(root / "prepared.pkl")]}}}
    (run_root / "orchestration.json").write_text(json.dumps(orchestration))
    (root / "prepared.pkl").write_bytes(b"synthetic unused input marker")
    return run_root, batch_dir


def collect_cli(run_root, result_path, reference_sha, parent_accounting=None):
    bootstrap = ("import sys; from scripts.dev import collect_global_cpu48_b64 as collector; "
        "collector.REFERENCE_SHA = sys.argv.pop(1); collector.main()")
    command = [sys.executable, "-c", bootstrap, reference_sha,
        "--run-root", str(run_root), "--output", str(result_path)]
    if parent_accounting is not None:
        command += ["--parent-accounting", str(parent_accounting)]
    return subprocess.run(command,
        cwd=ROOT, env=run_env(), text=True, capture_output=True, check=False)


def test_collector_cli_round_trip_and_saved_array_cpu_internal_gate(tmp_path):
    run_root, batch_dir = build_run(tmp_path)
    result_path = tmp_path / "collected.json"
    reference_sha = json.loads((run_root / "orchestration.json").read_text())["expected_gpu_reference_sha256"]
    completed = collect_cli(run_root, result_path, reference_sha)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text())
    assert result["status"] == "complete_exploratory_measurement"
    assert result["workload"]["point_days"] == 896
    assert len(result["cpu_batch"]["raw_output_schema"]) == 760
    assert result["cpu_batch"]["runtime_observations"]["process_cpu_samples_per_metric"] == {
        "B64_warm_frozen_state_core": 5,
        "B64_preprepared_wrapper_stack_transfer_materialize": 5}
    assert result["raw_array_gate_recomputation"]["cross_mode_full_output_passed"] is True

    scalar_dir = run_root / "scalar"
    manifest_path = scalar_dir / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["stages"][SCALAR_STAGE]
    checkpoint = scalar_dir / record["path"]
    envelope = pickle.loads(checkpoint.read_bytes())
    changed = envelope["value"]["outputs"]
    first_leaf = np.asarray(changed[0][0][0]).copy()
    first_leaf[0] += 1e-3
    changed_first_group = (first_leaf, *changed[0][0][1:])
    changed_first_block = (changed_first_group, changed[0][1])
    envelope["value"]["outputs"] = (changed_first_block, changed[1])
    payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    checkpoint.write_bytes(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    rejected = collect_cli(run_root, tmp_path / "tampered.json", reference_sha)
    assert rejected.returncode != 0
    assert "candidate saved arrays disagree with the complete CPU scalar reference" in rejected.stderr


def test_collector_cli_rejects_checkpoint_manifest_identity_tamper(tmp_path):
    run_root, batch_dir = build_run(tmp_path)
    path = batch_dir / "stage_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["lanes"][0]["inputs_sha256"] = "tampered-input"
    path.write_text(json.dumps(manifest))
    reference_sha = json.loads((run_root / "orchestration.json").read_text())["expected_gpu_reference_sha256"]
    rejected = collect_cli(run_root, tmp_path / "tampered.json", reference_sha)
    assert rejected.returncode != 0
    assert "saved output envelope identity/status mismatch" in rejected.stderr


def test_collector_cli_rejects_noncanonical_coordinator_reference_sha(tmp_path):
    run_root, _ = build_run(tmp_path)
    orchestration_path = run_root / "orchestration.json"
    orchestration = json.loads(orchestration_path.read_text())
    reference_sha = orchestration["expected_gpu_reference_sha256"]
    orchestration["expected_gpu_reference_sha256"] = "f" * 64
    orchestration_path.write_text(json.dumps(orchestration))
    rejected = collect_cli(run_root, tmp_path / "rejected.json", reference_sha)
    assert rejected.returncode != 0
    assert "differs from accepted FULL64 snapshot" in rejected.stderr


def test_collector_cli_binds_terminal_parent_accounting_to_raw_sacct_row(tmp_path):
    import copy

    run_root, _ = build_run(tmp_path)
    path = tmp_path / "accounting.json"
    values = {"source": "sacct", "scope": "whole_slurm_parent_job",
        "job_id": "synthetic-job", "terminal_status": "COMPLETED",
        "Elapsed": "00:02:00", "wall_seconds": 120,
        "allocated_CPU_count": 48, "GPU_count": 0,
        "AllocTRES": "cpu=48,mem=192G",
        "directories": [str(run_root / "scalar"), str(run_root / "batch")]}

    def seal(raw):
        result = copy.deepcopy(values)
        result.update(raw)
        result["raw_record"] = "|".join((result["job_id"], result["terminal_status"],
            result["Elapsed"], result["AllocTRES"])) + "\n"
        result["raw_record_sha256"] = hashlib.sha256(result["raw_record"].encode()).hexdigest()
        path.write_text(json.dumps(result))

    seal({})
    reference_sha = json.loads((run_root / "orchestration.json").read_text())["expected_gpu_reference_sha256"]
    passed = collect_cli(run_root, tmp_path / "accounted.json", reference_sha, path)
    assert passed.returncode == 0, passed.stderr
    accounting_result = json.loads((tmp_path / "accounted.json").read_text())
    cost_record = accounting_result["whole_parent_allocation_cost"]
    assert cost_record["status"] == "terminal_parent_accounting_loaded"
    assert cost_record["job_id"] == "synthetic-job" and cost_record["AllocTRES"] == values["AllocTRES"]
    assert cost_record["Elapsed"] == "00:02:00" and cost_record["wall_seconds"] == 120

    corruptions = [
        ({"AllocTRES": "cpu=47,mem=192G"}, "cpu=48"),
        ({"AllocTRES": "cpu=48,cpu=48,mem=192G"}, "cpu=48"),
        ({"AllocTRES": "cpu=48,gres/gpu=1"}, "no GPU allocation"),
        ({"job_id": "different-job"}, "job ID differs"),
        ({"Elapsed": "00:03:00", "wall_seconds": 120}, "Elapsed field"),
    ]
    for index, (changes, message) in enumerate(corruptions):
        seal(changes)
        rejected = collect_cli(run_root, tmp_path / f"rejected-{index}.json", reference_sha, path)
        assert rejected.returncode != 0
        if message == "Elapsed field":
            message = "wall_seconds must equal the parent Elapsed field"
        assert message in rejected.stderr
