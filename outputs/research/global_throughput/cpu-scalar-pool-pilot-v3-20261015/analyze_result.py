"""Analyze this terminal pilot's text/JSON receipts only; never load model arrays."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
PIN = "bae03d9ef78726f00987737cdc58eac705a922ae"
JOB = "15869035"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory = json.loads((HERE / "terminal-inventory.json").read_text())
    for record in inventory["files"]:
        path = HERE / record["name"]
        assert digest(path) == record["sha256"]
        assert path.stat().st_size == record["size_bytes"]
    rows = [line.split("|") for line in (HERE / "terminal-sacct-all.psv").read_text().splitlines()]
    parents = [row for row in rows if row[0] == JOB]
    assert len(parents) == 1
    parent = parents[0]
    assert parent[1:5] == ["FAILED", "1:0", "4", "4"]
    assert (HERE / "terminal-sacct-parent.psv").read_text().strip() == "|".join(parent)
    assert (HERE / "terminal-code-head.txt").read_text().strip() == PIN
    assert not (HERE / "terminal-code-status.txt").read_text().strip()
    context = (HERE / "launch-context.raw.txt").read_text()
    log = (HERE / "job-15869035.raw.log").read_text()
    receipts = [json.loads(line) for line in log.splitlines()
                if line.startswith('{"schema": "global_cpu_child_prejax_v1"')]
    failure = next(item for item in receipts if item["status"] == "resource_admission_failed")
    assert failure["affinity"] == [24, 25, 79, 80]
    assert not failure["JAX_imported"] and not failure["model_execution"]
    assert "allocation affinity must contain exactly the declared unique physical cores" in failure["error"]
    topology = {}
    for line in context.splitlines():
        match = re.fullmatch(r"(\d+)\s+(\d+)\s+(\d+)\s+yes\s*", line)
        if match:
            logical, core, socket = map(int, match.groups())
            topology[logical] = {"logical_cpu": logical, "core": core, "socket": socket}
    mapping = [topology[cpu] for cpu in failure["affinity"]]
    physical_keys = sorted({(item["socket"], item["core"]) for item in mapping})
    assert len(physical_keys) == 3
    source = subprocess.check_output(["git", "show", PIN + ":scripts/dev/global_server_resources.py"])
    assert b'identity = record["package_id"], record["physical_core_id"]' in source
    assert inventory["root_entries"] == ["launch-context.txt"]
    assert not any(inventory["expected_artifact_existence"].values())
    cpu_hours = Decimal(parent[3]) * Decimal(parent[4]) / Decimal(3600)
    totals = {"allocated_core_hours": str(cpu_hours),
              "CNY_at_0.05": str(cpu_hours * Decimal("0.05")),
              "CNY_at_0.07": str(cpu_hours * Decimal("0.07")),
              "rates_verified": False, "billing_parent_count": 1,
              "includes": "full parent allocation: startup, admission, failure, launcher/IO",
              "step_rows_added_to_parent_cost": False,
              "historical_jobs_added_to_this_cost": False}
    result = {"schema": "global_cpu_scalar_pool_pilot_v3_result_v1", "job_id": JOB,
        "execution_commit": PIN, "state": parent[1], "exit_code": parent[2],
        "parent_elapsed_seconds": int(parent[3]), "allocated_CPUs": int(parent[4]),
        "allocated_TRES": parent[5], "node": parent[6],
        "Slurm_CPU_IDs": "47-50", "prejax_affinity": failure["affinity"],
        "observed_lscpu_affinity_mapping": mapping,
        "observed_unique_socket_core_keys": physical_keys,
        "observed_unique_physical_core_count": len(physical_keys),
        "gate_key": ["package_id", "physical_core_id"],
        "gate_source": "scripts/dev/global_server_resources.py:61-67 at execution pin",
        "gate_source_sha256": hashlib.sha256(source).hexdigest(),
        "failure_class": "pre-JAX physical-core resource admission",
        "error": failure["error"], "optional_GPU_telemetry_fix_exercised": True,
        "actual_srun_dispatch_reached": True, "JAX_imported": False,
        "physical_model_executed": False, "workers_started": 0,
        "scientific_arrays_deserialized": False,
        "per_worker_lower_compile_cache_RSS_HWM": "not reached",
        "FULL64_760_leaf_CPU_strict_gate": "not reached",
        "GPU_cross_backend_exploratory_gates": "not reached",
        "frozen_reset_gate": "not reached", "first_call_core_wrapper": "not reached",
        "five_warm_core_wrapper_samples": "not reached",
        "completed_point_days": 0, "intended_unique_points": 64,
        "intended_point_days": 896, "scientific_accepted": False,
        "CPU_utilization": {"model_or_pool_utilization": "not reached",
            "parent_sacct_TotalCPU_seconds": 0.812,
            "parent_sacct_CPU_time_divided_by_wall": 0.203,
            "parent_allocated_CPU_normalized_fraction": 0.05075,
            "step_sacct_TotalCPU_seconds": 0.465, "step_elapsed_seconds": 3,
            "time_command_user_seconds": 0.24, "time_command_system_seconds": 0.21,
            "time_command_wall_seconds": 2.88,
            "time_command_busy_core_equivalents": 0.15625,
            "scope": "short failed pre-JAX startup only; not worker efficiency or model throughput"},
        "prejax_process_time_MaxRSS_kbytes": 18780,
        "prejax_process_time_MaxRSS_scope": "time-command child peak; not a model/worker or aggregate pool peak",
        "launch_host_load_average": [53.85, 54.57, 52.39],
        "terminal_node_snapshot_scope": "later snapshot; cannot establish launch contention or explain affinity mapping",
        "memory_192GB_48worker_conclusion": "no supporting or rejecting residence/throughput evidence",
        "accounting": totals, "paid_submission_count": 1, "paid_submissions_remaining": 0,
        "retry_permitted": False,
        "minimum_next_decision": "separately authorize/define a non-model physical-core binding diagnostic; verify Slurm billing IDs, OS affinity and sysfs topology before any revised pin or paid pilot",
        "unverified_option_candidates": ["hint", "threads-per-core"],
        "prohibited_inference": "four billable Slurm CPUs do not establish four distinct physical cores on this observed configuration",
        "raw_sources": [{"name": name, "sha256": digest(HERE / name)} for name in (
            "job-15869035.raw.log", "launch-context.raw.txt", "terminal-sacct-all.psv",
            "terminal-sacct-parent.psv", "terminal-node-snapshot.txt", "scheduler-config.raw.txt",
            "scheduler-config-selected.txt", "terminal-inventory.json", "paid-submission-attempt.json")]}
    (HERE / "result-analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", newline="\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
