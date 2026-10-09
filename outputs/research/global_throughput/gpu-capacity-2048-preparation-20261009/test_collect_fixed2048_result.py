"""Synthetic CPU-only tests for the fixed-2048 result collector contract."""
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dev.global_server_reference import _evidence_mismatch

SOURCE = Path(__file__).with_name("collect_fixed2048_result.py")
SPEC = importlib.util.spec_from_file_location("fixed2048_collector_test", SOURCE)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)

CODE_SHA = "a" * 40
LANES = [{"landpoint_id": f"point-{i:02d}", "inputs_sha256": f"{i:064x}"} for i in range(64)]


def make_identity(shape=1024):
    return {"contract": collector.CONTRACT, "process_strategy": collector.PROCESS_STRATEGY,
        "stage_shape": shape, "candidate_shapes": [1024, 2048], "total_lanes": 2048,
        "unique_real_points": 64, "warm_repeats": 5, "budget_seconds": 7200,
        "minimum_remaining_seconds": 3600, "validation_mode": "exploratory",
        "workload_kind": "capacity_screen_repeated_FULL64",
        "historical_B1024_evidence_sha256": "6bea479fcf3fa3d81d334c132106641572ca92a58886b26e764bcd7dee6069d5",
        "lane_origins": [{"lane_index": i, "source_index": i % 64,
            "landpoint_id": LANES[i % 64]["landpoint_id"], "repeat_index": i // 64} for i in range(2048)],
        "numerical_reference_sha256": "r" * 64, "lanes": LANES, "baseline_ref": "ref",
        "science_sha256": "c" * 64, "actual_science_source_sha256": "d" * 64,
        "execution_source_compatibility": {"contract": "source-cert"}, "precision_policy": "native",
        "runtime": {"python": "3.12", "jax": "0.4.38", "jaxlib": "0.4.38", "numpy": "2.4.6",
            "device_details": [{"platform": "gpu", "kind": "Tesla V100"}], "x64_enabled": True},
        "run_binding": {"Slurm_job_id": "999999", "hostname": "node1",
            "physical_gpu": {"uuid": "GPU-test", "driver_version": "550.1"},
            "server_resources": {"allocated_CPU_request_count": 4}, "allocator": "default",
            "preallocate": False, "window_start_epoch": 1000.0}}


def make_approval(**changes):
    values = {"job_id": "999999", "job_name": "orchjax-gpu-cap2048", "node": "node1",
        "gpu_uuid": "GPU-test", "slurm_global_gpu_id": "0", "driver_version": "550.1"}
    values.update(changes)
    return SimpleNamespace(**values)


def make_stage(shape=1024, *, pid=20, process_start=1100.0, window_start=1000.0):
    identity = make_identity(shape)
    identity["run_binding"]["window_start_epoch"] = window_start
    env = {"git_head": CODE_SHA, "python": "3.12", "jax": "0.4.38", "jaxlib": "0.4.38",
        "numpy": "2.4.6", "x64_enabled": True, "hostname": "node1",
        "device_details": [{"platform": "gpu", "kind": "Tesla V100"}],
        "physical_gpu": {"uuid": "GPU-test", "driver_version": "550.1"},
        "command_line": f"worker-{pid}", "pid": pid, "process_start_epoch": process_start,
        "loadavg": [1.0, 2.0, 3.0] if pid == 20 else [3.0, 4.0, 5.0]}
    origin = {"pid": pid, "process_start_epoch": process_start,
        "process_start_epoch_source": "GPU entry main before JAX/backend initialization",
        "process_strategy": collector.PROCESS_STRATEGY, "Slurm_job_id": "999999",
        "hostname": "node1", "physical_gpu": identity["run_binding"]["physical_gpu"], "environment": env}
    return {"identity": identity, "common_identity": {k: v for k, v in identity.items() if k != "stage_shape"},
        "execution_origin": origin, "allocation": {"Slurm": {"SLURM_JOB_ID": "999999"}},
        "checkpoint_sha256": f"{shape:064x}", "manifest_sha256": "m" * 64, "report_sha256": "p" * 64}


def test_identity_binds_exact_2048_cyclic_inputs_reference_and_resource_policy():
    identity = make_identity()
    admission = {"source_identity": {"baseline_ref": "ref", "science_sha256": "c" * 64,
        "lanes": LANES, "precision_policy": "native"}, "FULL64_reference_origin": {"sha256": "r" * 64}}
    common = collector.validate_identity(identity, admission, CODE_SHA, 1024)
    assert common == {k: v for k, v in identity.items() if k != "stage_shape"}
    identity["lane_origins"][2047]["source_index"] = 1
    with pytest.raises(ValueError, match="cyclic 2048"):
        collector.validate_identity(identity, admission, CODE_SHA, 1024)


def test_identity_rejects_allocator_or_window_mutation():
    admission = {"source_identity": {"baseline_ref": "ref", "science_sha256": "c" * 64,
        "lanes": LANES, "precision_policy": "native"}, "FULL64_reference_origin": {"sha256": "r" * 64}}
    identity = make_identity()
    identity["run_binding"]["preallocate"] = True
    with pytest.raises(ValueError, match="default allocator"):
        collector.validate_identity(identity, admission, CODE_SHA, 1024)
    identity = make_identity()
    identity["run_binding"]["window_start_epoch"] = float("nan")
    with pytest.raises(ValueError, match="default allocator"):
        collector.validate_identity(identity, admission, CODE_SHA, 1024)


def test_stage_store_manifest_evidence_allows_only_loadavg_json_sequence_normalization(tmp_path):
    from scripts.dev.global_batch_stage_store import StageStore

    identity = {"contract": "synthetic"}
    store = StageStore(tmp_path, identity)
    record = {"execution_origin": {"environment": {"loadavg": (1.0, 2.0, 3.0), "jax": "0.4.38"}}}
    store.save("fixture", {"value": 1}, record)
    envelope = pickle.loads((tmp_path / "stage_fixture.pkl").read_bytes())
    manifest = json.loads((tmp_path / "stage_manifest.json").read_text())
    collector.validate_manifest_record(envelope["record"], manifest["stages"]["fixture"],
                                       _evidence_mismatch)
    manifest["stages"]["fixture"]["execution_origin"]["environment"]["loadavg"][0] = 9.0
    with pytest.raises(ValueError, match="checkpoint/manifest record disagreement"):
        collector.validate_manifest_record(envelope["record"], manifest["stages"]["fixture"],
                                           _evidence_mismatch)
def test_samples_require_five_positive_finite_full_workload_observations():
    values = [20.0, 21.0, 19.0, 22.0, 20.5]
    metric = {"samples_seconds": values, "median_seconds": 20.5, "range_seconds": [19.0, 22.0],
        "lane_days": 28672, "unique_real_points": 64, "lane_days_per_second": 28672 / 20.5}
    assert collector.validate_samples(metric, "core")["median_seconds"] == 20.5
    metric["samples_seconds"][3] = float("nan")
    with pytest.raises(ValueError, match="five positive finite"):
        collector.validate_samples(metric, "core")


def memory_fixture(shape=1024):
    resident, output, slack = 1000, 200, 2 * 1024**3
    historical = {"available": True, "temp_size_in_bytes": 100}
    control = {"available": True, "temp_size_in_bytes": 150}
    actual = {"available": True, "temp_size_in_bytes": 120 if shape == 1024 else 250}
    sources = {"1024": {"available": True, "temp_size_in_bytes": 120 if shape == 1024 else 150}}
    prior_temp = 100 if shape == 1024 else 150
    pre_estimate = 2 * resident + 3 * output + prior_temp * shape // 1024
    post_temp = 120 if shape == 1024 else max(150 * 2, 250)
    post_estimate = 2 * resident + 3 * output + post_temp
    if shape == 2048:
        sources["2048"] = actual
    gpu = {"applicable": True, "uuid": "GPU-test", "total_bytes": 40 * 1024**3,
        "free_bytes": post_estimate + slack + 100}
    proof = {"memory_preflight": {"phase": "preallocation_whole_stage_budget_before_device_put",
        "resident_2048_input_bytes": resident, "full_2048_output_bytes": output,
        "estimated_device_bytes": pre_estimate, "required_physical_slack_bytes": slack,
        "physical_gpu": dict(gpu)},
        "post_transfer_GPU_memory": {"phase": "post_transfer_physical_free_telemetry_only",
            "physical_gpu": {**gpu, "free_bytes": 1}},
        "executable_memory_analysis": actual, "admission_compiler_memory_records": sources,
        "postcompile_memory_budget_validation": {"phase": "postcompile_updated_whole_stage_bound_against_preallocation_budget",
            "shape": shape, "original_preallocation_physical_gpu_budget": dict(gpu),
            "postcompile_physical_gpu_snapshot": {**gpu, "free_bytes": 2},
            "resident_2048_input_bytes": resident, "full_2048_output_bytes": output,
            "estimated_device_bytes": post_estimate, "required_physical_slack_bytes": slack,
            "actual_executable_memory_analysis": actual, "resident_inputs_already_placed": True,
            "permitted": True}}
    return proof, historical, control


@pytest.mark.parametrize("shape", [1024, 2048])
def test_phase_aware_memory_guard_reuses_original_free_budget_and_actual_compiler_bound(shape):
    proof, historical, control = memory_fixture(shape)
    receipt = collector.validate_memory_guard(proof, shape, historical, control if shape == 2048 else None)
    assert receipt["preallocation_estimate_bytes"] == proof["memory_preflight"]["estimated_device_bytes"]
    assert receipt["postcompile_free_is_budget"] is False


@pytest.mark.parametrize("mutation", ["wrong_phase", "reduced_margin", "larger_estimate", "missing_analysis"])
def test_phase_aware_memory_guard_fails_closed_on_budget_mutations(mutation):
    proof, historical, _ = memory_fixture()
    if mutation == "wrong_phase":
        proof["post_transfer_GPU_memory"]["phase"] = "postcompile_budget"
    elif mutation == "reduced_margin":
        proof["postcompile_memory_budget_validation"]["required_physical_slack_bytes"] = 0
    elif mutation == "larger_estimate":
        proof["postcompile_memory_budget_validation"]["estimated_device_bytes"] += 1
    else:
        proof["executable_memory_analysis"] = {"available": False}
    with pytest.raises(ValueError):
        collector.validate_memory_guard(proof, 1024, historical)


def test_gate_contract_requires_all_2048_reference_and_three_internal_gates():
    gates = [{"name": f"capacity_B1024_lane_{i}_source_{i % 64}_GPU_reference", "atol": 1e-8,
        "rtol": 1e-10, "compared_leaves": 760, "passed": True} for i in range(2048)]
    gates += [{"name": f"capacity_B1024_{suffix}", "atol": 1e-8, "rtol": 1e-10,
        "compared_leaves": 760, "passed": True} for suffix in collector.GATE_NAMES]
    collector.check_gate_set(gates, 1024, None, {})
    gates[-1]["passed"] = False
    with pytest.raises(ValueError, match="gate failed"):
        collector.check_gate_set(gates, 1024, None, {})


def test_raw9_accounts_one_terminal_parent_and_keeps_overrun_cost():
    approval = make_approval()
    row = "|".join((approval.job_id, approval.job_name, "gnall", "TIMEOUT", "7201", "4",
        "cpu=4,node=1,billing=4,gres/gpu=1", approval.node, "124:0"))
    child = "|".join((approval.job_id + ".batch", approval.job_name, "gnall", "TIMEOUT", "7190", "4",
        "cpu=4,node=1,billing=4", approval.node, "124:0"))
    accounting = collector.parse_raw9(row + "\n" + child, approval)
    assert accounting["parent_counted_once"] is True
    assert accounting["whole_job_cost"]["primary"]["GPU_card_hours"] == 7201 / 3600
    result = {"valid_exploratory_stages": [make_stage()], "withheld_stages": [], "capacity_curve_available": True}
    collector.apply_accounting(result, accounting)
    assert result["valid_exploratory_stages"] == []
    assert result["capacity_curve_available"] is False
    assert result["accounting"]["observed_wall_seconds"] == 7201


def test_raw9_requires_one_terminal_parent_and_exact_allocation():
    approval = make_approval()
    row = "|".join((approval.job_id, approval.job_name, "gnall", "COMPLETED", "7000", "4",
        "cpu=4,node=1,billing=4,gres/gpu=1", approval.node, "0:0"))
    with pytest.raises(ValueError, match="exactly one"):
        collector.parse_raw9(row + "\n" + row, approval)
    with pytest.raises(ValueError, match="not terminal"):
        collector.parse_raw9(row.replace("COMPLETED", "RUNNING"), approval)
    bad_tres = row.replace("gres/gpu=1", "gres/gpu=2")
    with pytest.raises(ValueError, match="allocation"):
        collector.parse_raw9(bad_tres, approval)


def test_distinct_fresh_processes_require_same_origin_and_allow_declared_process_fields():
    control, candidate = make_stage(), make_stage(2048, pid=21, process_start=1200.0)
    control["allocation"]["thread_affinity_after_backend"] = {"thread_ids": [1, 2], "observed_count": 2}
    candidate["allocation"]["thread_affinity_after_backend"] = {"thread_ids": [9, 10], "observed_count": 2}
    common = collector.compare_origins(control, candidate)
    assert common["distinct_worker_pids"] == [20, 21]
    candidate["execution_origin"]["environment"]["jax"] = "different"
    with pytest.raises(ValueError, match="normalized worker runtime"):
        collector.compare_origins(control, candidate)


def test_common_identity_rejects_changed_declared_logical_cpu_mask():
    control, candidate = make_stage(), make_stage(2048, pid=21, process_start=1200.0)
    candidate["identity"]["run_binding"]["server_resources"]["selected_logical_cpus"] = [8, 9, 10, 11]
    candidate["common_identity"] = {key: value for key, value in candidate["identity"].items()
                                    if key != "stage_shape"}
    with pytest.raises(ValueError, match="common code/science/input/runtime"):
        collector.compare_origins(control, candidate)


@pytest.mark.parametrize("candidate_start,control_exit", [(1100.0, 1150.0), (float("nan"), 1150.0)])
def test_candidate_cannot_overlap_or_lack_finite_worker_window(candidate_start, control_exit):
    control, candidate = make_stage(), make_stage(2048, pid=21, process_start=candidate_start)
    with pytest.raises(ValueError, match="after the successful B1024 process exit"):
        collector.validate_worker_window(control, candidate, control_exit)


def test_coordinator_receipt_retains_control_when_candidate_does_not_start(tmp_path):
    control = make_stage()
    (tmp_path / "orchestration.json").write_text(json.dumps({"contract": collector.CONTRACT,
        "process_strategy": collector.PROCESS_STRATEGY, "budget_seconds": 7200, "window_start_epoch": 1000.0,
        "status": "candidate_denied", "completed_shapes": [1024], "comparison_shapes": [1024],
        "children": {"1024": {"returncode": 0, "command_process_pid": 15,
            "worker_execution_origin": control["execution_origin"],
            "exit_epoch": 1150.0, "dispatch_start_epoch": 1010.0,
            "sealed_receipt": {"sha256": control["checkpoint_sha256"],
                "manifest_sha256": control["manifest_sha256"], "report_sha256": control["report_sha256"]}}}}))
    receipt = collector.validate_coordinator(tmp_path, {1024: control})
    assert receipt["children"]["1024"]["returncode"] == 0


def test_parent_timeout_during_candidate_keeps_control_dispatch_receipt(tmp_path):
    control = make_stage()
    (tmp_path / "orchestration.json").write_text(json.dumps({"contract": collector.CONTRACT,
        "process_strategy": collector.PROCESS_STRATEGY, "budget_seconds": 7200, "window_start_epoch": 1000.0,
        "status": "running", "completed_shapes": [1024], "comparison_shapes": [1024],
        "children": {"1024": {"returncode": 0, "command_process_pid": 15,
            "worker_execution_origin": control["execution_origin"],
            "exit_epoch": 1150.0, "dispatch_start_epoch": 1010.0,
            "sealed_receipt": {"sha256": control["checkpoint_sha256"],
                "manifest_sha256": control["manifest_sha256"], "report_sha256": control["report_sha256"]}},
            "2048": {"command_process_pid": 16, "dispatch_start_epoch": 1160.0}}}))
    assert collector.validate_coordinator(tmp_path, {1024: control})["status"] == "running"


def test_invalid_candidate_coordinator_receipt_does_not_disqualify_control(tmp_path):
    control, candidate = make_stage(), make_stage(2048, pid=21, process_start=1200.0)
    (tmp_path / "orchestration.json").write_text(json.dumps({"contract": collector.CONTRACT,
        "process_strategy": collector.PROCESS_STRATEGY, "budget_seconds": 7200, "window_start_epoch": 1000.0,
        "status": "completed_exploratory_capacity_screen", "completed_shapes": [1024, 2048],
        "comparison_shapes": [1024, 2048], "children": {
            "1024": {"returncode": 0, "command_process_pid": 15,
                "worker_execution_origin": control["execution_origin"],
                "exit_epoch": 1150.0, "dispatch_start_epoch": 1010.0,
                "sealed_receipt": {"sha256": control["checkpoint_sha256"],
                    "manifest_sha256": control["manifest_sha256"], "report_sha256": control["report_sha256"]}},
            "2048": {"returncode": 0, "command_process_pid": 16,
                "worker_execution_origin": candidate["execution_origin"],
                "dispatch_start_epoch": 1160.0, "exit_epoch": 1300.0,
                "sealed_receipt": {"sha256": "wrong", "manifest_sha256": "wrong", "report_sha256": "wrong"}}}}))
    assert collector.validate_coordinator(tmp_path, {1024: control})["completed_shapes"] == [1024, 2048]
    with pytest.raises(ValueError, match="terminal receipt"):
        collector.validate_coordinator(tmp_path, {1024: control, 2048: candidate})


def test_coordinator_requires_candidate_worker_to_start_after_control_exit(tmp_path):
    control, candidate = make_stage(), make_stage(2048, pid=21, process_start=1140.0)
    def receipt_for(stage):
        return {"sha256": stage["checkpoint_sha256"],
            "manifest_sha256": stage["manifest_sha256"], "report_sha256": stage["report_sha256"]}
    children = {"1024": {"returncode": 0, "command_process_pid": 15,
        "worker_execution_origin": control["execution_origin"],
        "dispatch_start_epoch": 1010.0, "exit_epoch": 1150.0,
        "sealed_receipt": receipt_for(control)},
        "2048": {"returncode": 0, "command_process_pid": 16,
            "worker_execution_origin": candidate["execution_origin"],
            "dispatch_start_epoch": 1160.0, "exit_epoch": 1300.0,
            "sealed_receipt": receipt_for(candidate)}}
    (tmp_path / "orchestration.json").write_text(json.dumps({"contract": collector.CONTRACT,
        "process_strategy": collector.PROCESS_STRATEGY, "budget_seconds": 7200,
        "window_start_epoch": 1000.0, "status": "completed_exploratory_capacity_screen",
        "completed_shapes": [1024, 2048], "comparison_shapes": [1024, 2048], "children": children}))
    with pytest.raises(ValueError, match="B2048 child window binding"):
        collector.validate_coordinator(tmp_path, {1024: control, 2048: candidate})


def test_candidate_common_identity_mismatch_is_rejected():
    control, candidate = make_stage(), make_stage(2048, pid=21, process_start=1200.0)
    candidate["identity"]["run_binding"]["window_start_epoch"] = 999.0
    candidate["common_identity"] = {k: v for k, v in candidate["identity"].items() if k != "stage_shape"}
    with pytest.raises(ValueError, match="common code/science/input/runtime"):
        collector.compare_origins(control, candidate)


def test_candidate_failure_keeps_sealed_B1024_without_curve_or_ratio():
    control = make_stage()
    accepted, valid, common, curve, ratios = collector.stage_outcome({1024: control}, [])
    assert list(accepted) == [1024]
    assert valid == [control]
    assert common is None and curve is False and ratios is None


def test_candidate_without_valid_control_is_withheld():
    candidate = make_stage(2048, pid=21, process_start=1200.0)
    accepted, valid, common, curve, ratios = collector.stage_outcome({2048: candidate}, [])
    assert not accepted and not valid
    assert common is None and curve is False and ratios is None
