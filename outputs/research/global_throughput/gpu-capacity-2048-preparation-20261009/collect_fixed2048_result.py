"""Read-only collector for the fresh-process fixed-2048 capacity screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

CONTRACT = "GPU-fixed2048-repeated-FULL64-B1024-B2048-fresh-process-capacity-v1"
PROCESS_STRATEGY = "sequential_fresh_process_per_shape"
LANES, UNIQUE_POINTS, LANE_DAYS, LEAVES = 2048, 64, 28672, 760
STAGES = {1024: "capacity_B1024_fixed2048_fresh_v1", 2048: "capacity_B2048_fixed2048_fresh_v1"}
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
GATE_NAMES = ("frozen_reset_repeat", "permutation", "replacement_isolation")


def demand(ok, message):
    if not ok:
        raise ValueError(message)


def file_digest(path):
    digest, count = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def allocation_contract(resources):
    slurm = resources.get("Slurm", {})
    tres = dict(part.split("=", 1) for part in slurm.get("ORCHJAX_SLURM_ALLOC_TRES", "").split(",") if "=" in part)
    return {"allocated_CPU_request_count": resources.get("allocated_CPU_request_count"),
        "job_id": str(slurm.get("SLURM_JOB_ID")), "partition": slurm.get("SLURM_JOB_PARTITION"),
        "node_list": slurm.get("SLURM_JOB_NODELIST"), "global_gpu_id": str(slurm.get("SLURM_JOB_GPUS")),
        "tres": {key: tres.get(key) for key in ("node", "cpu", "gres/gpu", "billing")}}


def stable_resource_identity(resources):
    keys = ("mode", "allocated_CPU_request_count", "observed_unique_physical_cores", "original_affinity",
        "selected_logical_cpus", "topology", "hostname", "CUDA_VISIBLE_DEVICES_preserved", "Slurm")
    result = {key: resources[key] for key in keys if key in resources}
    census = resources.get("thread_affinity_after_backend")
    if census is not None:
        result["backend_thread_affinity_contract"] = {key: census[key] for key in (
            "expected_logical_cpus", "all_observed_thread_masks_within_declared_cores")}
    return result


def validate_manifest_record(record, manifest_record, evidence_mismatch):
    for key, value in record.items():
        mismatch = evidence_mismatch(value, manifest_record.get(key), "record." + key)
        demand(mismatch is None, f"checkpoint/manifest record disagreement: {mismatch}")


def parse_raw9(text, approval):
    rows = [line.split("|") for line in text.splitlines() if line.strip()]
    parents = [row for row in rows if row and row[0] == approval.job_id]
    demand(len(parents) == 1, "exactly one raw9 sacct parent row is required")
    row = parents[0]
    demand(len(row) == 9, "raw9 JobIDRaw|JobName|Partition|State|ElapsedRaw|AllocCPUS|AllocTRES|NodeList|ExitCode required")
    demand(row[1] == approval.job_name == "orchjax-gpu-cap2048" and row[2] == "gnall",
           "fixed-2048 parent job name/partition mismatch")
    state = row[3].split()[0].rstrip("+")
    demand(state in TERMINAL, "parent sacct row is not terminal")
    seconds, cpus = int(row[4]), int(row[5])
    tres = dict(part.split("=", 1) for part in row[6].split(",") if "=" in part)
    demand(seconds >= 0 and cpus == 4 and tres.get("node") == "1" and tres.get("cpu") == "4"
           and tres.get("gres/gpu") == "1", "terminal parent allocation must be one node, four CPUs, one GPU")
    demand(row[7] == approval.node, "terminal parent node differs from explicit observed node")
    return {"terminal": True, "state": state, "exit_code": row[8], "observed_wall_seconds": seconds,
        "approved_wall_seconds": 7200, "approved_wall_exceeded": seconds > 7200,
        "node": row[7], "cpus": cpus, "tres": tres, "raw_parent_row": row,
        "raw_schema": "JobIDRaw|JobName|Partition|State|ElapsedRaw|AllocCPUS|AllocTRES|NodeList|ExitCode",
        "parent_counted_once": True,
        "whole_job_cost": {label: {"CNY": (2.20 + 4 * cpu_price) * seconds / 3600,
            "GPU_card_hours": seconds / 3600, "allocated_CPU_count_hours": 4 * seconds / 3600,
            "billing_verified": False, "scope": "one raw9 terminal parent row; child rows are never summed"}
            for label, cpu_price in (("primary", .05), ("conservative", .07))}}


def validate_identity(identity, admission, expected_code_sha, shape):
    demand(identity.get("contract") == CONTRACT and identity.get("process_strategy") == PROCESS_STRATEGY,
           "fixed-2048 contract/process strategy mismatch")
    demand(identity.get("stage_shape") == shape and identity.get("candidate_shapes") == [1024, 2048]
           and identity.get("total_lanes") == LANES and identity.get("unique_real_points") == UNIQUE_POINTS
           and identity.get("warm_repeats") == 5 and identity.get("budget_seconds") == 7200,
           "fixed-2048 stage identity/workload mismatch")
    demand(identity.get("minimum_remaining_seconds") == 3600 and identity.get("validation_mode") == "exploratory"
           and identity.get("workload_kind") == "capacity_screen_repeated_FULL64",
           "fixed-2048 stage guard/evidence scope mismatch")
    demand(identity.get("historical_B1024_evidence_sha256") == "6bea479fcf3fa3d81d334c132106641572ca92a58886b26e764bcd7dee6069d5",
           "exact B1024 historical provenance pin required")
    lanes = identity.get("lanes", [])
    demand(len(lanes) == UNIQUE_POINTS and len({entry.get("landpoint_id") for entry in lanes}) == UNIQUE_POINTS,
           "exactly 64 admitted FULL64 source lanes required")
    origins = [{"lane_index": i, "source_index": i % UNIQUE_POINTS,
        "landpoint_id": lanes[i % UNIQUE_POINTS]["landpoint_id"], "repeat_index": i // UNIQUE_POINTS}
        for i in range(LANES)]
    demand(identity.get("lane_origins") == origins, "exact cyclic 2048 lane origins required")
    reference_sha = identity.get("numerical_reference_sha256")
    admitted = admission.get("FULL64_reference_origin", {})
    demand(reference_sha == admitted.get("sha256"), "original admitted FULL64 reference SHA mismatch")
    source = admission.get("source_identity", {})
    for key in ("baseline_ref", "science_sha256", "lanes", "precision_policy"):
        demand(identity.get(key) == source.get(key), f"fixed-2048 admission source identity mismatch: {key}")
    demand(identity.get("science_sha256") and identity.get("actual_science_source_sha256")
           and identity.get("execution_source_compatibility"), "source identity/certificate required")
    run = identity.get("run_binding", {})
    demand(run.get("allocator") == "default" and run.get("preallocate") is False
           and type(run.get("window_start_epoch")) in (int, float)
           and math.isfinite(run["window_start_epoch"]) and run["window_start_epoch"] > 0,
           "unchanged default allocator with PREALLOCATE=false required")
    return {key: value for key, value in identity.items() if key != "stage_shape"}


def validate_samples(metric, label):
    samples = metric.get("samples_seconds", [])
    demand(len(samples) == 5 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples),
           f"{label}: five positive finite measured samples required")
    median = statistics.median(samples)
    demand(math.isclose(metric.get("median_seconds", -1), median, rel_tol=1e-12)
           and metric.get("range_seconds") == [min(samples), max(samples)], f"{label}: median/range mismatch")
    demand(metric.get("lane_days") == LANE_DAYS and metric.get("unique_real_points") == UNIQUE_POINTS
           and math.isclose(metric.get("lane_days_per_second", -1), LANE_DAYS / median, rel_tol=1e-12),
           f"{label}: full 28672 lane-day denominator required")
    return {"samples_seconds": list(samples), "median_seconds": median,
            "range_seconds": [min(samples), max(samples)], "lane_days": LANE_DAYS,
            "unique_real_points": UNIQUE_POINTS, "lane_days_per_second": LANE_DAYS / median}


def validate_memory_guard(provenance, shape, historical_memory, control_memory=None):
    pre = provenance.get("memory_preflight", {})
    physical = pre.get("physical_gpu", {})
    demand(pre.get("phase") == "preallocation_whole_stage_budget_before_device_put"
           and pre.get("required_physical_slack_bytes") == 2 * 1024**3,
           f"B{shape}: preallocation whole-stage guard phase/slack mismatch")
    resident, full_output = pre.get("resident_2048_input_bytes"), pre.get("full_2048_output_bytes")
    demand(type(resident) is int and resident > 0 and type(full_output) is int and full_output > 0
           and physical.get("applicable") is True and isinstance(physical.get("uuid"), str)
           and physical.get("uuid") and type(physical.get("total_bytes")) is int
           and type(physical.get("free_bytes")) is int and physical["total_bytes"] >= 30 * 1024**3,
           f"B{shape}: selected physical GPU preallocation budget missing")
    prior = [historical_memory]
    if shape == 2048:
        demand(control_memory is not None, "B2048 requires validated new B1024 compiler-memory control")
        prior.append(control_memory)
    temps = [item.get("temp_size_in_bytes") for item in prior]
    demand(all(item.get("available") is True and type(value) is int and value >= 0
               for item, value in zip(prior, temps, strict=True)),
           f"B{shape}: missing exact prior compiler-memory analysis")
    temp = max(value * shape // 1024 for value in temps)
    estimated = 2 * resident + 3 * full_output + temp
    demand(pre.get("estimated_device_bytes") == estimated
           and physical["free_bytes"] >= estimated + pre["required_physical_slack_bytes"],
           f"B{shape}: preallocation estimate/margin does not match the untouched free budget")

    post_transfer = provenance.get("post_transfer_GPU_memory", {})
    transfer_gpu = post_transfer.get("physical_gpu", {})
    demand(post_transfer.get("phase") == "post_transfer_physical_free_telemetry_only"
           and transfer_gpu.get("uuid") == physical["uuid"]
           and transfer_gpu.get("total_bytes") == physical["total_bytes"]
           and type(transfer_gpu.get("free_bytes")) is int and transfer_gpu["free_bytes"] >= 0,
           f"B{shape}: post-transfer snapshot must remain telemetry for the same selected GPU")

    actual = provenance.get("executable_memory_analysis", {})
    records = provenance.get("admission_compiler_memory_records", {})
    demand(actual.get("available") is True and type(actual.get("temp_size_in_bytes")) is int
           and actual["temp_size_in_bytes"] >= 0,
           f"B{shape}: actual postcompile executable-memory analysis required")
    expected_shapes = {"1024"} if shape == 1024 else {"1024", "2048"}
    demand(set(records) == expected_shapes, f"B{shape}: exact compiler-memory admission record set required")
    record_values = [records[str(previous)] for previous in sorted(map(int, records))]
    demand(all(item.get("available") is True and type(item.get("temp_size_in_bytes")) is int
               and item["temp_size_in_bytes"] >= 0 for item in record_values),
           f"B{shape}: invalid compiler-memory admission record")
    if shape == 1024:
        demand(records["1024"].get("temp_size_in_bytes") == max(
            actual["temp_size_in_bytes"], historical_memory["temp_size_in_bytes"]),
            "B1024: postcompile memory bound must retain historical and actual temporaries")
    else:
        demand(records["1024"].get("temp_size_in_bytes") == max(
            control_memory["temp_size_in_bytes"], historical_memory["temp_size_in_bytes"])
            and records["2048"].get("temp_size_in_bytes") == actual["temp_size_in_bytes"],
            "B2048: postcompile bounds must retain historical/control and actual candidate temporaries")
    post_temp = max(item["temp_size_in_bytes"] * shape // previous
                    for previous, item in ((int(key), value) for key, value in records.items()))
    post_estimate = 2 * resident + 3 * full_output + post_temp
    post = provenance.get("postcompile_memory_budget_validation", {})
    after = post.get("postcompile_physical_gpu_snapshot", {})
    demand(post.get("phase") == "postcompile_updated_whole_stage_bound_against_preallocation_budget"
           and post.get("shape") == shape and post.get("original_preallocation_physical_gpu_budget") == physical
           and post.get("resident_2048_input_bytes") == resident
           and post.get("full_2048_output_bytes") == full_output
           and post.get("estimated_device_bytes") == post_estimate
           and post.get("required_physical_slack_bytes") == 2 * 1024**3
           and post.get("actual_executable_memory_analysis") == actual
           and post.get("resident_inputs_already_placed") is True and post.get("permitted") is True,
           f"B{shape}: postcompile updated whole-stage bound/arithmetic mismatch")
    demand(after.get("applicable") is True and after.get("uuid") == physical["uuid"]
           and after.get("total_bytes") == physical["total_bytes"]
           and type(after.get("free_bytes")) is int and after["free_bytes"] >= 0
           and post_estimate + 2 * 1024**3 <= physical["free_bytes"],
           f"B{shape}: updated estimate must fit the original free budget and preserve selected GPU")
    return {"preallocation_estimate_bytes": estimated, "postcompile_estimate_bytes": post_estimate,
            "original_free_budget_bytes": physical["free_bytes"],
            "postcompile_free_snapshot_bytes": after.get("free_bytes"),
            "postcompile_free_is_budget": False}


def check_gate_set(gates, shape, base, sentinel_paths):
    names = [f"capacity_B{shape}_lane_{i}_source_{i % 64}_GPU_reference" for i in range(LANES)]
    names.extend(f"capacity_B{shape}_{suffix}" for suffix in GATE_NAMES)
    demand(len(gates) == 2051 and [gate.get("name") for gate in gates] == names,
           "exact 2048 saved-reference gates plus three internal gates required in canonical order")
    for gate in gates:
        demand(gate.get("atol") == 1e-8 and gate.get("rtol") == 1e-10
               and gate.get("compared_leaves") == LEAVES and gate.get("passed") is True
               and not gate.get("failures"), f"strict full760 gate failed: {gate.get('name')}")
        for sentinel in gate.get("allowed_sentinels", []):
            demand(sentinel_paths.get(sentinel.get("path")) == sentinel.get("contract"),
                   "candidate sentinel differs from immutable original FULL64 source")


def validate_stage(directory, shape, admission, code_sha, approval, base, reference, sentinel_paths,
                   historical_memory, control_memory=None):
    directory = Path(directory)
    manifest_path, report_path = directory / "stage_manifest.json", directory / "report.json"
    demand(manifest_path.is_file() and report_path.is_file(), f"B{shape}: stage manifest/report absent")
    manifest, report = json.loads(manifest_path.read_text()), json.loads(report_path.read_text())
    identity = manifest.get("identity", {})
    common = validate_identity(identity, admission, code_sha, shape)
    name, stage = STAGES[shape], manifest.get("stages", {}).get(STAGES[shape], {})
    demand(stage.get("status") == "complete" and stage.get("measurement_status") == "complete"
           and stage.get("strict_numerical_passed") is True and stage.get("scientific_acceptance") is False,
           f"B{shape}: complete exploratory strict stage required")
    demand(stage.get("path") == f"stage_{name}.pkl", f"B{shape}: canonical checkpoint path required")
    checkpoint = directory / stage["path"]
    demand(checkpoint.resolve().is_relative_to(directory.resolve()) and checkpoint.is_file(),
           f"B{shape}: checkpoint must exist inside its stage directory")
    sha, byte_count = file_digest(checkpoint)
    demand(stage.get("sha256") == sha and stage.get("checkpoint_io", {}).get("bytes") == byte_count,
           f"B{shape}: full checkpoint SHA/byte receipt mismatch")
    with checkpoint.open("rb") as stream:
        envelope = pickle.load(stream)
    demand(envelope.get("identity") == identity and envelope.get("name") == name
           and envelope.get("status") == "complete", f"B{shape}: checkpoint envelope identity/status mismatch")
    record = envelope.get("record", {})
    from scripts.dev.global_server_reference import _evidence_mismatch
    validate_manifest_record(record, stage, _evidence_mismatch)
    origin = record.get("execution_origin", {})
    manifest_origin = stage.get("execution_origin", {})
    demand(_evidence_mismatch(origin, manifest_origin, "record.execution_origin") is None,
           f"B{shape}: manifest worker origin differs from checkpoint")
    origin = manifest_origin
    validate_loadavg(origin.get("environment", {}).get("loadavg"), f"B{shape}")
    demand(report.get("status") == "completed_exploratory_capacity_screen"
           and report.get("completed_shapes") == [shape] and report.get("comparison_shapes") == [shape]
           and report.get("stages", {}).get(name) == stage
           and _evidence_mismatch(report.get("environment"), origin.get("environment"),
                                  "record.execution_origin.environment") is None,
           f"B{shape}: report/manifest/origin mismatch")
    demand(report.get("gates") == record.get("gates"), f"B{shape}: report/manifest gate list mismatch")
    proof = record.get("provenance", {})
    for key, expected in {"total_lanes": LANES, "lane_days": LANE_DAYS, "unique_real_points": UNIQUE_POINTS,
        "chunk_size": shape, "chunks_per_collection": LANES // shape, "blocks": [[2, 8], [9, 15]],
        "half_hours_per_day": 48, "full_outputs": True, "full_two_block_workload": True,
        "process_strategy": PROCESS_STRATEGY}.items():
        demand(proof.get(key) == expected, f"B{shape}: workload proof mismatch: {key}")
    memory_guard = validate_memory_guard(proof, shape, historical_memory, control_memory)
    schema = proof.get("full_output_native_schema", [])
    demand(len(schema) == LEAVES, f"B{shape}: exact full760 native output schema required")
    for item in schema:
        demand(item.get("shape", [None])[0] == LANES and isinstance(item.get("dtype"), str) and item["dtype"],
               f"B{shape}: native output shape/dtype mismatch")
    demand(origin.get("environment", {}).get("git_head") == code_sha,
           f"B{shape}: execution origin code SHA differs from explicit binding")
    demand(origin.get("process_strategy") == PROCESS_STRATEGY and origin.get("Slurm_job_id") == approval.job_id
           and origin.get("hostname") == approval.node and origin.get("physical_gpu", {}).get("uuid") == approval.gpu_uuid
           and origin.get("physical_gpu", {}).get("driver_version") == approval.driver_version,
           f"B{shape}: actual worker process/job/node/GPU origin mismatch")
    demand(origin.get("process_start_epoch_source") == "GPU entry main before JAX/backend initialization",
           f"B{shape}: declared worker process-start source required")
    demand(identity.get("run_binding", {}).get("Slurm_job_id") == approval.job_id
           and identity.get("run_binding", {}).get("hostname") == approval.node
           and identity.get("run_binding", {}).get("physical_gpu", {}).get("uuid") == approval.gpu_uuid
           and identity.get("run_binding", {}).get("physical_gpu", {}).get("driver_version") == approval.driver_version,
           f"B{shape}: common job/node/physical GPU identity differs from observed startup pins")
    demand(type(origin.get("pid")) is int and origin["pid"] > 0
           and type(origin.get("process_start_epoch")) in (int, float) and origin["process_start_epoch"] > 0,
           f"B{shape}: actual worker PID/process start required")
    slurm = proof.get("server_resources", {}).get("Slurm", {})
    tres = dict(part.split("=", 1) for part in slurm.get("ORCHJAX_SLURM_ALLOC_TRES", "").split(",") if "=" in part)
    demand(str(slurm.get("SLURM_JOB_ID")) == approval.job_id and slurm.get("SLURM_JOB_PARTITION") == "gnall"
           and slurm.get("SLURM_JOB_NODELIST") == approval.node
           and str(slurm.get("SLURM_JOB_GPUS")) == approval.slurm_global_gpu_id
           and proof.get("server_resources", {}).get("allocated_CPU_request_count") == 4
           and tres.get("node") == "1" and tres.get("cpu") == "4" and tres.get("gres/gpu") == "1",
           f"B{shape}: explicit parent/node/GPU/CPU allocation mismatch")
    demand(identity.get("run_binding", {}).get("server_resources")
           == stable_resource_identity(proof.get("server_resources", {})),
           f"B{shape}: stable run identity/allocation resource mismatch")
    gates = record.get("gates", [])
    check_gate_set(gates, shape, base, sentinel_paths)
    metrics = {key: validate_samples(record.get("timing", {}).get(key, {}), f"B{shape}.{key}")
        for key in ("core", "wrapper")}
    demand(record.get("timing") == envelope.get("value", {}).get("measurements"),
           f"B{shape}: checkpoint measurements differ from sealed timing")
    actual, actual_tree = base.jax.tree_util.tree_flatten(envelope.get("value", {}).get("outputs"))
    expected, expected_tree = base.jax.tree_util.tree_flatten(reference)
    demand(actual_tree == expected_tree and len(actual) == len(expected) == LEAVES,
           f"B{shape}: native output PyTree differs from admitted FULL64 reference")
    import numpy as np
    for index, (got, ref) in enumerate(zip(actual, expected, strict=True)):
        got, ref = np.asarray(got), np.asarray(ref)
        demand(ref.ndim > 0 and ref.shape[0] == UNIQUE_POINTS and got.shape == (LANES, *ref.shape[1:])
               and got.dtype == ref.dtype, f"B{shape}: full native shape/dtype mismatch at leaf {index}")
    comparisons = []
    outputs = envelope["value"]["outputs"]
    for lane in range(LANES):
        compared = base.compare(base.jax.tree_util.tree_map(lambda value: value[lane], outputs),
            base.jax.tree_util.tree_map(lambda value: value[lane % UNIQUE_POINTS], reference),
            sentinel_paths=sentinel_paths)
        demand(compared.get("passed") is True, f"B{shape}: actual saved reference comparison failed at lane {lane}")
        comparisons.append({"lane": lane, "source_index": lane % UNIQUE_POINTS, **compared})
    io = stage.get("checkpoint_io", {})
    demand(io.get("available") is True and io.get("bytes") == byte_count
           and all(type(io.get(key)) in (int, float) and math.isfinite(io[key]) and io[key] >= 0
               for key in ("serialization_seconds", "atomic_write_seconds")),
           f"B{shape}: actual complete checkpoint IO timings required")
    return {"stage": name, "batch_size": shape, "chunks_per_collection": LANES // shape,
        "checkpoint_sha256": sha, "checkpoint_bytes": byte_count, "lane_days": LANE_DAYS,
        "unique_real_points": UNIQUE_POINTS, "native_output_leaves": LEAVES,
        "strict_numerical_passed": True, "actual_saved_output_reference_passed": True,
        "actual_saved_output_reference_comparison_count": len(comparisons),
        "actual_saved_output_reference_comparisons": comparisons, "core": metrics["core"],
        "wrapper": metrics["wrapper"], "execution_origin": origin,
        "memory_guard": memory_guard, "executable_memory_analysis": proof["executable_memory_analysis"],
        "allocation": proof.get("server_resources"), "identity": identity,
        "common_identity": common, "manifest_sha256": file_digest(manifest_path)[0],
        "report_sha256": file_digest(report_path)[0]}


def validate_coordinator(root, accepted):
    path = Path(root) / "orchestration.json"
    demand(path.is_file(), "authoritative CPU coordinator orchestration receipt required")
    report = json.loads(path.read_text())
    demand(report.get("contract") == CONTRACT and report.get("process_strategy") == PROCESS_STRATEGY
           and report.get("budget_seconds") == 7200 and type(report.get("window_start_epoch")) in (int, float)
           and math.isfinite(report["window_start_epoch"]) and report["window_start_epoch"] > 0,
           "coordinator contract/window mismatch")
    children = report.get("children", {})
    if accepted and 1024 in accepted:
        demand(report.get("status") in {"candidate_denied", "candidate_failed", "completed_exploratory_capacity_screen", "running"},
               "coordinator must be terminal with a valid control outcome")
        demand(report.get("completed_shapes", [])[:1] == [1024]
               and report.get("completed_shapes") in ([1024], [1024, 2048])
               and report.get("comparison_shapes") == report.get("completed_shapes"),
               "coordinator must retain the completed B1024 control before optional candidate")
        child = children.get("1024", {})
        receipt = child.get("sealed_receipt", {})
        control = accepted[1024]
        dispatch, exit_epoch = child.get("dispatch_start_epoch"), child.get("exit_epoch")
        demand(control["identity"]["run_binding"]["window_start_epoch"] == report["window_start_epoch"]
               and type(dispatch) in (int, float) and math.isfinite(dispatch)
               and type(exit_epoch) in (int, float) and math.isfinite(exit_epoch)
               and exit_epoch >= dispatch >= report["window_start_epoch"]
               and type(child.get("command_process_pid")) is int and child["command_process_pid"] > 0
               and child.get("worker_execution_origin") == control["execution_origin"]
               and dispatch <= control["execution_origin"]["process_start_epoch"] <= exit_epoch,
               "B1024 child window binding differs from coordinator start")
        demand(child.get("returncode") == 0 and exit_epoch <= children.get("2048", {}).get("dispatch_start_epoch", float("inf"))
               and receipt.get("sha256") == control["checkpoint_sha256"]
               and receipt.get("manifest_sha256") == control["manifest_sha256"]
               and receipt.get("report_sha256") == control["report_sha256"],
               "coordinator must seal/exit B1024 before candidate dispatch with matching receipts")
    if accepted and 2048 in accepted:
        demand(report.get("status") == "completed_exploratory_capacity_screen",
               "two-stage curve requires completed coordinator status")
        demand(report.get("completed_shapes") == [1024, 2048]
               and report.get("comparison_shapes") == [1024, 2048],
               "two-stage coordinator must record both completed comparison shapes")
        demand(1024 in accepted, "B2048 cannot be promoted without a strictly accepted B1024 control")
        child = children.get("2048", {})
        receipt = child.get("sealed_receipt", {})
        candidate = accepted[2048]
        dispatch, exit_epoch = child.get("dispatch_start_epoch"), child.get("exit_epoch")
        demand(candidate["identity"]["run_binding"]["window_start_epoch"] == report["window_start_epoch"]
               and type(dispatch) in (int, float) and math.isfinite(dispatch)
               and type(exit_epoch) in (int, float) and math.isfinite(exit_epoch)
               and exit_epoch >= dispatch >= report["window_start_epoch"]
               and type(child.get("command_process_pid")) is int and child["command_process_pid"] > 0
               and child.get("worker_execution_origin") == candidate["execution_origin"]
               and dispatch <= candidate["execution_origin"]["process_start_epoch"] <= exit_epoch,
               "B2048 child window binding differs from coordinator start")
        demand(child.get("returncode") == 0 and receipt.get("sha256") == candidate["checkpoint_sha256"]
               and receipt.get("manifest_sha256") == candidate["manifest_sha256"]
               and receipt.get("report_sha256") == candidate["report_sha256"],
               "coordinator B2048 terminal receipt differs from strictly validated stage")
        validate_worker_window(accepted[1024], candidate, children["1024"].get("exit_epoch"))
    return report


def compare_origins(control, candidate):
    left, right = control["execution_origin"], candidate["execution_origin"]
    demand(left["pid"] != right["pid"] and left["process_start_epoch"] < right["process_start_epoch"],
           "B1024/B2048 must be distinct sequential worker processes")
    demand(left.get("process_strategy") == right.get("process_strategy") == PROCESS_STRATEGY,
           "both workers must declare the fresh-process strategy")
    demand(control["common_identity"] == candidate["common_identity"],
           "B1024/B2048 common code/science/input/runtime/job/node/GPU/allocator identity mismatch")
    for key in ("Slurm_job_id", "hostname", "physical_gpu"):
        demand(left.get(key) == right.get(key), f"worker origin differs across stages: {key}")
    demand(allocation_contract(control["allocation"]) == allocation_contract(candidate["allocation"]),
           "stable stage resource/allocation identity differs")
    demand(normalize_environment(left.get("environment", {})) == normalize_environment(right.get("environment", {})),
           "normalized worker runtime environment differs across fresh processes")
    # The coordinator timestamps prove non-overlap; worker process start epochs alone prove ordering.
    return {"common_origin_passed": True, "distinct_worker_pids": [left["pid"], right["pid"]],
            "worker_start_epochs": [left["process_start_epoch"], right["process_start_epoch"]]}


def normalize_environment(environment):
    process_fields = {"pid", "process_id", "process_start_epoch", "process_start_epoch_source",
        "command", "command_line", "argv", "loadavg"}
    if isinstance(environment, dict):
        return {key: normalize_environment_value(value) for key, value in environment.items()
                if key not in process_fields}
    return environment


def validate_loadavg(loadavg, label):
    demand(type(loadavg) in (tuple, list) and len(loadavg) == 3
           and all(type(item) in (int, float) and math.isfinite(item) and item >= 0 for item in loadavg),
           f"{label}: actual three-value host load average required")


def normalize_environment_value(value):
    if isinstance(value, dict):
        return {key: normalize_environment_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_environment_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_environment_value(item) for item in value)
    return value


def validate_worker_window(control, candidate, control_exit_epoch):
    control_start = control["execution_origin"].get("process_start_epoch")
    candidate_start = candidate["execution_origin"].get("process_start_epoch")
    demand(type(control_exit_epoch) in (int, float) and math.isfinite(control_exit_epoch)
           and type(control_start) in (int, float) and math.isfinite(control_start)
           and type(candidate_start) in (int, float) and math.isfinite(candidate_start)
           and candidate_start > control_start and candidate_start >= control_exit_epoch,
           "fresh B2048 worker must start after the successful B1024 process exit")


def stage_outcome(accepted, withheld):
    accepted = dict(accepted)
    common = None
    if 2048 in accepted and 1024 not in accepted:
        withheld.append({"stage": STAGES[2048], "reason": "strict B1024 control was not accepted"})
        accepted.pop(2048)
    if 1024 in accepted and 2048 in accepted:
        try:
            common = compare_origins(accepted[1024], accepted[2048])
        except ValueError as exc:
            withheld.append({"stage": STAGES[2048], "reason": str(exc)})
            accepted.pop(2048)
    valid = [accepted[size] for size in (1024, 2048) if size in accepted]
    curve = len(valid) == 2 and common is not None
    ratios = None
    if curve:
        ratios = {boundary: accepted[1024][boundary]["median_seconds"] / accepted[2048][boundary]["median_seconds"]
                  for boundary in ("core", "wrapper")}
    return accepted, valid, common, curve, ratios


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key in ("code-dir", "stage-dir", "admission", "sacct-file", "b1024-evidence-dir", "output"):
        result.add_argument("--" + key, required=True, type=Path)
    result.add_argument("--expected-code-sha", required=True)
    for key in ("job-id", "job-name", "node", "gpu-uuid", "slurm-global-gpu-id", "driver-version"):
        result.add_argument("--" + key, required=True)
    for key, value in (("approved-nodes", 1), ("approved-tasks", 1), ("approved-cpus", 4),
                       ("approved-gpus", 1), ("approved-wall-seconds", 7200)):
        result.add_argument("--" + key, type=int, choices=[value], required=True)
    return result


def admit_code_checkout(code_dir, expected):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_dir, text=True).strip()
    demand(head == expected, "collector checkout HEAD differs from explicit expected-code-sha")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=code_dir, text=True)
    demand(not status.strip(), "execution code checkout must be clean")


def empty_result(args, reason, accounting=None):
    result = {"job_id": args.job_id, "code_commit": args.expected_code_sha, "contract": CONTRACT,
        "scientific_acceptance": False, "valid_exploratory_stages": [],
        "withheld_stages": [{"stage": STAGES[size], "reason": reason} for size in (1024, 2048)],
        "incomplete_stages": [], "historical_B1024_evidence": None, "accounting": accounting,
        "capacity_curve_available": False, "capacity_curve_scope": "no two-stage curve without both strict fresh-process stages",
        "B2048_over_B1024_throughput_ratio": None,
        "scientific_claims": {"cpu_gpu_ratio": None, "ten_x_improvement": None, "general_scientific_acceptance": False}}
    if accounting is not None:
        apply_accounting(result, accounting)
    return result


def apply_accounting(result, accounting):
    result["accounting"] = accounting
    result["parent_successfully_completed"] = accounting["state"] == "COMPLETED" and accounting["exit_code"] == "0:0"
    result["stage_promotion_withheld_due_to_wall_overrun"] = accounting["approved_wall_exceeded"]
    if accounting["approved_wall_exceeded"]:
        result["valid_exploratory_stages"] = []
        result["withheld_stages"] = [{"stage": STAGES[size],
            "reason": "terminal parent exceeded approved 7200-second wall envelope"} for size in (1024, 2048)]
        result["capacity_curve_available"] = False
        result["B2048_over_B1024_throughput_ratio"] = None
        result["capacity_curve_scope"] = "stages withheld because observed terminal wall exceeded approved envelope"


def collect(args):
    demand(args.job_id.isdecimal(), "explicit numeric parent JobID required")
    approval = args
    accounting = parse_raw9(args.sacct_file.read_text(), approval)
    if accounting["approved_wall_exceeded"]:
        result = empty_result(args, "terminal parent exceeded approved wall; actual cost retained", accounting)
        return result, 1
    admit_code_checkout(args.code_dir, args.expected_code_sha)
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(args.code_dir.resolve()))
    from scripts.dev import benchmark_global_batch as base
    from scripts.dev import benchmark_global_gpu_capacity as capacity
    from scripts.dev.global_server_reference import load_reference
    from scripts.dev.global_source_compatibility import source_identity

    admission = json.loads(args.admission.read_text())
    stage_root = args.stage_dir
    if not (stage_root / "B1024").is_dir():
        return empty_result(args, "B1024 stage directory absent; denied/unfinished run", accounting), 0
    first_manifest = json.loads((stage_root / "B1024" / "stage_manifest.json").read_text())
    first_identity = first_manifest.get("identity", {})
    validate_identity(first_identity, admission, args.expected_code_sha, 1024)
    expected_source = source_identity(args.code_dir)
    for key in ("science_sha256", "actual_science_source_sha256", "execution_source_compatibility"):
        demand(first_identity.get(key) == expected_source.get(key),
               f"fixed-2048 source identity/certificate differs from checked-out source: {key}")
    reference, actual_ref = load_reference(Path(admission["FULL64_reference_origin"]["path"]).parent, first_identity)
    from scripts.dev.global_server_reference import _evidence_mismatch
    admitted_ref = admission["FULL64_reference_origin"]
    proof = {key: value for key, value in admitted_ref.items() if key != "backend_conditions"}
    actual = {key: value for key, value in actual_ref.items() if key != "backend_conditions"}
    demand(_evidence_mismatch(proof, actual, "admitted_reference") is None,
           "original admitted FULL64 reference identity/schema changed")
    demand(_evidence_mismatch(actual_ref, json.loads((stage_root / "B1024" / "report.json").read_text()).get("reference_origin"),
                              "report.reference_origin") is None, "B1024 report reference origin mismatch")
    reference_envelope = pickle.loads(Path(admitted_ref["path"]).read_bytes())
    sentinels = {}
    for gate in reference_envelope.get("record", {}).get("gates", []):
        for allowed in gate.get("allowed_sentinels", []):
            previous = sentinels.setdefault(allowed["path"], allowed["contract"])
            demand(previous == allowed["contract"], "original FULL64 sentinel contract conflict")

    # The admitted old B1024 checkpoint supplies provenance only; strip and never expose its timing.
    history_input = dict(first_identity)
    for key in ("contract", "process_strategy", "stage_shape", "total_lanes", "lane_origins",
        "unique_real_points", "candidate_shapes", "warm_repeats", "budget_seconds",
        "minimum_remaining_seconds", "validation_mode", "workload_kind", "numerical_reference_sha256",
        "historical_B1024_evidence_sha256", "run_binding", "actual_science_source_sha256",
        "execution_source_compatibility"):
        history_input.pop(key, None)
    history_expected = capacity.historical_b1024_identity(history_input,
        first_identity["lane_origins"], admitted_ref["sha256"])
    historical = capacity.import_b1024_evidence(args.b1024_evidence_dir, history_expected,
        stage_root, base.jax, reference=reference)
    demand(historical.get("sha256") == capacity.B1024_SHA256
           and historical.get("bytes") == capacity.B1024_BYTES
           and "timing" not in historical, "exact historical B1024 origin-only evidence required")

    accepted, withheld, incomplete = {}, [], []
    historical_memory = historical["provenance"]["executable_memory_analysis"]
    for shape in (1024, 2048):
        directory = stage_root / f"B{shape}"
        if not directory.is_dir():
            incomplete.append({"stage": STAGES[shape], "status": "absent", "reason": "stage directory absent"})
            continue
        if shape == 2048 and 1024 not in accepted:
            withheld.append({"stage": STAGES[2048], "reason": "strict B1024 control was not accepted"})
            continue
        try:
            accepted[shape] = validate_stage(directory, shape, admission, args.expected_code_sha,
                approval, base, reference, sentinels, historical_memory,
                accepted[1024]["executable_memory_analysis"] if shape == 2048 else None)
        except (ValueError, KeyError, TypeError, AssertionError, OSError) as exc:
            withheld.append({"stage": STAGES[shape], "reason": str(exc)})
    coordinator = None
    if 1024 in accepted:
        try:
            coordinator = validate_coordinator(stage_root, {1024: accepted[1024]})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            withheld.append({"stage": "coordinator_control", "reason": str(exc)})
            accepted.clear()
    if coordinator is not None and 2048 in accepted:
        try:
            validate_coordinator(stage_root, accepted)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            withheld.append({"stage": STAGES[2048], "reason": f"coordinator candidate receipt: {exc}"})
            accepted.pop(2048)
    if coordinator is None and 2048 in accepted:
        withheld.append({"stage": STAGES[2048], "reason": "strict B1024 control/coordinator receipt was not accepted"})
        accepted.pop(2048)
    if 1024 not in accepted and coordinator is None:
        coordinator = None
    accepted, valid, common, curve, ratios = stage_outcome(accepted, withheld)
    result = {"job_id": args.job_id, "code_commit": args.expected_code_sha, "contract": CONTRACT,
        "scientific_acceptance": False, "valid_exploratory_stages": valid,
        "withheld_stages": withheld, "incomplete_stages": incomplete,
        "historical_B1024_evidence": {key: value for key, value in historical.items() if key != "timing"},
        "coordinator": coordinator, "accounting": accounting,
        "capacity_curve_available": curve, "B2048_over_B1024_throughput_ratio": ratios,
        "capacity_curve_scope": "same 2048-lane workload, common origin, two fresh sequential processes" if curve else
            "B2048 absent, denied, invalid or origin-mismatched; retain valid B1024 and report no ratio",
        "common_process_origin": common,
        "scientific_claims": {"cpu_gpu_ratio": None, "ten_x_improvement": None,
            "general_scientific_acceptance": False}}
    apply_accounting(result, accounting)
    return result, 0


def main():
    args = parser().parse_args()
    try:
        result, status = collect(args)
    except (ValueError, KeyError, TypeError, AssertionError, OSError, subprocess.SubprocessError) as exc:
        accounting = None
        try:
            accounting = parse_raw9(args.sacct_file.read_text(), args)
        except (ValueError, OSError, TypeError):
            pass
        result = empty_result(args, str(exc), accounting)
        status = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "valid_stages": len(result["valid_exploratory_stages"]),
                      "capacity_curve_available": result["capacity_curve_available"]}, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
