"""Account for this terminal parent allocation using recovered metadata only."""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
verification = json.loads((ROOT / "COLLECTION_VERIFICATION.json").read_bytes())
evidence = ROOT / "terminal-evidence"
report = json.loads((evidence / "run/phased_resident_control/capacity/B1024/report.json").read_bytes())
orchestration = json.loads((evidence / "run/orchestration.json").read_bytes())
parent = verification["parent_slurm"]
seconds, cpus = int(parent["ElapsedRaw"]), int(parent["AllocCPUS"])
phases = {row["phase"]: row for row in report["phases"]}
guard = phases["capacity_B1024_postcompile_guard"]["postcompile_memory_budget_validation"]
log = (evidence / "slurm-log.txt").read_text(encoding="utf-8")
maximum_rss_kib = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)[1])
record = {
    "schema": "global-gpu-lifecycle-terminal-measurements-v1",
    "execution_sha": verification["execution_sha"],
    "parent_allocation": parent,
    "orchestration_status": orchestration["status"],
    "failure": {**report["initial_validation_failure"], "lane_origin": report["identity"]["lane_origins"][45],
                "failed_array_bytes": 880948565,
                "specific_leaf_and_error_magnitude": None,
                "limitation": "original saved report names first failing lane only; later bounded CPU explanation is attached when present"},
    "accepted_arms": [],
    "second_arm_dispatched": False,
    "warm_samples": {"core": [], "wrapper": []},
    "throughput_ratio": None,
    "timing": {
        "prepared_input_reload_seconds": report["prepared_input_reload_seconds"],
        "reference_memory_reload_seconds": report["reference_memory_reload_seconds"],
        "lower_call_seconds": report["timing"]["capacity_B1024_lower_seconds"],
        "compile_call_seconds": report["timing"]["capacity_B1024_compile_seconds"],
        "compile_start_to_compiled_phase_seconds": phases["capacity_B1024_compiled"]["elapsed_seconds"]
            - phases["capacity_B1024_compile_start"]["elapsed_seconds"],
        "input_transfer_boundary_seconds": phases["capacity_B1024_input_transfer_complete"]["elapsed_seconds"]
            - phases["capacity_B1024_input_transfer_start"]["elapsed_seconds"],
        "initial_collection_start_child_elapsed_seconds": phases["capacity_B1024_initial_collection_start"]["elapsed_seconds"],
        "initial_execution_seconds": None,
        "validation_seconds": None,
        "failed_array_serialization_seconds": None,
        "failed_array_io_seconds": None,
        "timing_limitation": "failure precedes initial_collection_complete and completed stage; no separate execution/validation/serialization/IO timers saved",
        "report_elapsed_seconds": report["elapsed_seconds"],
        "coordinator_wall_seconds": orchestration["orchestration_wall_seconds"],
        "child_gnu_time_wall": "18:56.08",
        "child_gnu_time_user_cpu_seconds": float(re.search(r"User time \(seconds\): ([\d.]+)", log)[1]),
        "child_gnu_time_system_cpu_seconds": float(re.search(r"System time \(seconds\): ([\d.]+)", log)[1]),
        "parent_allocation_elapsed_seconds": seconds,
    },
    "memory": {
        "guard_permitted": guard["permitted"],
        "resident_input_bytes": guard["resident_2048_input_bytes"],
        "full_output_bytes": guard["full_2048_output_bytes"],
        "estimated_device_bytes": guard["estimated_device_bytes"],
        "original_preallocation_free_bytes": guard["original_preallocation_physical_gpu_budget"]["free_bytes"],
        "required_physical_slack_bytes": guard["required_physical_slack_bytes"],
        "compiler_memory_analysis": guard["actual_executable_memory_analysis"],
        "host_peak_rss_bytes_ru_maxrss": report["linux_ru_maxrss_peak_bytes"],
        "host_peak_rss_bytes_gnu_time": maximum_rss_kib * 1024,
        "physical_vram_peak_bytes": None,
        "root_release_exercised": False,
        "limitation": "compiler estimates and discrete GPU snapshots do not establish measured VRAM peak or allocator reclamation",
    },
    "compiled_input_signature": report["compilations"]["capacity_B1024"]["compiled_input_signature"],
    "whole_parent_allocation_cost": {
        "allocated_cpu_count_seconds": seconds * cpus,
        "allocated_gpu_card_seconds": seconds,
        "primary_cny": seconds / 3600 * (cpus * 0.05 + 2.20),
        "conservative_cny": seconds / 3600 * (cpus * 0.07 + 2.20),
        "approved_primary_ceiling_cny": 4.80,
        "approved_conservative_ceiling_cny": 4.96,
        "actual_bill_verified": False,
        "basis": "single parent Slurm ElapsedRaw; four allocated CPUs plus one GPU; no batch/extern/srun double count",
    },
    "authorization_status": "closed_after_single_terminal_allocation_no_residual_budget_reuse",
    "evidence": {"collection": "COLLECTION_VERIFICATION.json", "raw": "terminal-evidence/"},
    "new_model_jax_or_array_execution_for_accounting": False,
}
comparison = ROOT / "saved-array-explanation/comparison"
if (comparison / "summary.json").exists():
    summary = json.loads((comparison / "summary.json").read_bytes())
    assert summary["status"] == "complete_saved_initial_failure_comparison"
    assert summary["all760_leaves_and2048_lanes_checked"] is True
    with gzip.open(comparison / "all-failing-coordinates.jsonl.gz", "rt", encoding="utf-8") as stream:
        detail = [json.loads(line) for line in stream]
    coordinates = [row for row in detail if row["record_type"] == "failed_coordinate"]
    record["failure"]["specific_leaf_and_error_magnitude"] = {
        "failing_leaves": summary["failing_leaves"],
        "unique_points": summary["failing_unique_points"],
        "failing_lanes": summary["failing_lanes"],
        "native_failing_coordinates_including_repeats": summary["native_failing_coordinates_including_repeats"],
        "exact_failure_group_count": summary["exact_failure_group_count"],
        "coordinates_per_exact_group": coordinates,
    }
    record["failure"]["limitation"] = "bounded saved-array numerical explanation complete; no cause attribution or scientific/performance acceptance"
    record["saved_array_CPU_explanation"] = {
        "status": summary["status"],
        "comparison_runs": 1,
        "execution_code_pin": summary["execution_code_pin"],
        "gate": summary["gate"],
        "runtime": summary["runtime"],
        "outer_gnu_time_wall_seconds": 29.54,
        "all760_leaves_and2048_lanes_checked": True,
        "scientific_acceptance": False,
        "supervisor_saved_array_localization_accepted": True,
        "supervisor_thread": "01a128d2-3ec2-77a1-8d51-1665c220d2ca",
        "scientific_execution_closed": True,
        "old_ABI_failure_evidence": False,
        "cause_attribution": None,
        "read_only_original_scientific_arrays_on_free_CPU_test_node": True,
        "original_scientific_arrays_downloaded": False,
        "bookkeeping_input_scope": "this generator reads only recovered metadata and scalar differences; the separately authorized CPU explanation read the arrays on test03",
        "evidence": {
            "summary": "saved-array-explanation/comparison/summary.json",
            "coordinates": "saved-array-explanation/comparison/all-failing-coordinates.jsonl.gz",
            "collection": "EXPLANATION_COLLECTION.json",
            "readable": "SAVED_ARRAY_EXPLANATION.md",
        },
    }
assert record["memory"]["host_peak_rss_bytes_ru_maxrss"] == record["memory"]["host_peak_rss_bytes_gnu_time"]
(ROOT / "terminal-measurements.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"failure": record["failure"], "timing": record["timing"],
                  "cost": record["whole_parent_allocation_cost"]}, indent=2))
