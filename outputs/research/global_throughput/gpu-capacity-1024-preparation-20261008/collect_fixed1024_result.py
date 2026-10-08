"""Read-only saved-result collector for one approved fixed-1024 GPU job."""
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

CONTRACT = "GPU-fixed1024-repeated-FULL64-B512-B1024-capacity-v1"
SCIENCE_SHA = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"
ACTUAL_SCIENCE_SHA = "b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8"
LEGACY_B512_SHA = "1c44d84ad73f3a04f6d21522a0b35011d157e3a1dbc5b155dc93833a44912a2a"
LEGACY_B512_BYTES = 220430830
STAGE_NAMES = ("capacity_B512_fixed1024_v1", "capacity_B1024_fixed1024_v1")
LANES = 1024
LANE_DAYS = 14336
UNIQUE_POINTS = 64
GATE = {"atol": 1e-8, "rtol": 1e-10, "discrete_schema_dtype": "exact",
        "sentinel": "declared_bare_soil_roughheight_only"}
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_raw9(text, approval):
    rows = [line.split("|") for line in text.splitlines() if line.strip()]
    parents = [row for row in rows if row and row[0] == approval.job_id]
    demand(len(parents) == 1, "exactly one raw9 sacct parent row is required")
    row = parents[0]
    demand(len(row) == 9, "raw9 sacct JobIDRaw|JobName|Partition|State|ElapsedRaw|AllocCPUS|AllocTRES|NodeList|ExitCode required")
    demand(approval.job_name == "orchjax-gpu-cap1024" and row[1] == approval.job_name
           and row[2] == "gnall", "fixed-1024 parent job name/partition mismatch")
    state = row[3].split()[0].rstrip("+")
    demand(state in TERMINAL, "parent sacct row is not terminal; stages remain unpromoted")
    seconds, cpus = int(row[4]), int(row[5])
    tres = dict(part.split("=", 1) for part in row[6].split(",") if "=" in part)
    demand(seconds >= 0 and cpus == 4, "terminal wall or allocated CPU count is invalid")
    demand(tres.get("node") == "1" and tres.get("cpu") == "4" and tres.get("gres/gpu") == "1",
           "terminal parent must record one node, four CPUs, and one GPU")
    demand(row[7] == approval.node, "terminal parent node differs from explicit approval")
    return {"terminal": True, "state": state, "exit_code": row[8], "observed_wall_seconds": seconds,
            "approved_wall_seconds": 5400, "approved_wall_exceeded": seconds > 5400,
            "node": row[7], "cpus": cpus, "tres": tres, "raw_parent_row": row,
            "raw_schema": "JobIDRaw|JobName|Partition|State|ElapsedRaw|AllocCPUS|AllocTRES|NodeList|ExitCode",
            "parent_counted_once": True,
            "whole_job_cost": {label: {"CNY": (2.20 + 4 * cpu_price) * seconds / 3600,
                "GPU_card_hours": seconds / 3600, "allocated_CPU_count_hours": 4 * seconds / 3600,
                "billing_verified": False, "scope": "one raw9 terminal parent row; no child rows summed"}
                for label, cpu_price in (("primary", .05), ("conservative", .07))}}


def identity_checks(identity, admission, expected_code_sha):
    source = admission["source_identity"]
    demand(identity.get("contract") == CONTRACT, "fixed-1024 contract mismatch")
    demand(identity.get("candidate_shapes") == [512, 1024], "fixed candidate order must be [512, 1024]")
    demand(identity.get("total_lanes") == LANES and identity.get("unique_real_points") == UNIQUE_POINTS,
           "fixed workload must be 1024 lanes repeating 64 real points")
    demand(identity.get("warm_repeats") == 5,
           "fixed five-repeat policy mismatch")
    demand(identity.get("science_sha256") == SCIENCE_SHA
           and identity.get("actual_science_source_sha256") == ACTUAL_SCIENCE_SHA,
           "required Linux source identity changed")
    demand(identity.get("baseline_ref") == source.get("baseline_ref")
           and identity.get("science_sha256") == source.get("science_sha256")
           and identity.get("lanes") == source.get("lanes")
           and identity.get("precision_policy") == source.get("precision_policy"),
           "fixed-1024 source identity differs from canonical admission")
    demand(identity.get("historical_B512_evidence_sha256") == LEGACY_B512_SHA,
           "exact sealed historical B512 checkpoint identity required")
    demand(identity.get("conditional_policy") == "fixed1024_resource_only_2700s_2GiB_GPU_8GiB_host_slack",
           "fixed resource-only policy required")
    lanes = identity["lanes"]
    demand(len(lanes) == UNIQUE_POINTS and len({lane["landpoint_id"] for lane in lanes}) == UNIQUE_POINTS,
           "exactly 64 admitted source lanes required")
    expected_origins = [{"lane_index": index, "source_index": index % UNIQUE_POINTS,
        "landpoint_id": lanes[index % UNIQUE_POINTS]["landpoint_id"], "repeat_index": index // UNIQUE_POINTS}
        for index in range(LANES)]
    demand(identity.get("lane_origins") == expected_origins, "exact cyclic 1024-lane source origins required")
    demand(identity.get("numerical_reference_sha256") == admission["FULL64_reference_origin"]["sha256"],
           "canonical original FULL64 reference SHA mismatch")
    demand(identity.get("gate") == GATE, "fixed numerical tolerances/schema/sentinel policy changed")
    # This certificate pins the only admitted source-byte divergence from the baseline.
    certificate = identity.get("execution_source_compatibility", {})
    expected_certificate = {
        "active": True,
        "contract": "exact-DIFFUCO-host-mask-cache-repair-v1",
        "source_path": "jax_orchidee/sechiba/diffuco.py",
        "original_file_sha256": "04fc1193e85d2deca0f97cc06008989053ccaf956db3dc9552dd6b9cc2549d22",
        "actual_file_sha256": "42073dd14f427db8ed78803c90cd5a64b677bf822d1edf6b6fbed7aa97166a14",
        "baseline_full_source_sha256": SCIENCE_SHA,
        "baseline_checkout_bytes_sha256": SCIENCE_SHA,
        "baseline_checkout_representation": "canonical Linux LF source bytes",
        "actual_full_source_sha256": ACTUAL_SCIENCE_SHA,
        "compatibility_scope": "exact helper-only immutable host Boolean mask; all other source bytes unchanged",
        "claim": "audited execution-only equivalence; source bytes differ",
    }
    demand(certificate == expected_certificate, "exact admitted source-compatibility certificate mismatch")


def validate_sample(metric, label):
    values = metric.get("samples_seconds", [])
    demand(len(values) == 5 and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                                    for value in values), f"{label}: five positive finite raw samples required")
    median = statistics.median(values)
    demand(math.isclose(metric.get("median_seconds", -1), median, rel_tol=1e-12), f"{label}: median mismatch")
    demand(metric.get("range_seconds") == [min(values), max(values)], f"{label}: complete range mismatch")
    demand(metric.get("lane_days") == LANE_DAYS and metric.get("unique_real_points") == UNIQUE_POINTS
           and metric.get("unique_point_days") == 896
           and metric.get("unique_point_days_is_throughput_denominator") is False,
           f"{label}: fixed 14336 lane-day denominator required")
    demand(math.isclose(metric.get("lane_days_per_second", -1), LANE_DAYS / median, rel_tol=1e-12),
           f"{label}: lane-day throughput mismatch")
    warm_costs = {label: {"CNY_per_lane_day": ((2.20 + 4 * cpu_price) * median / 3600) / LANE_DAYS,
        "GPU_card_hours_per_lane_day": (median / 3600) / LANE_DAYS,
        "allocated_CPU_count_hours_per_lane_day": (4 * median / 3600) / LANE_DAYS,
        "scope": "measured warm collection only; excludes startup, checkpoint IO, and parent allocation"}
        for label, cpu_price in (("primary", .05), ("conservative", .07))}
    return {"samples_seconds": list(values), "median_seconds": median,
            "range_seconds": [min(values), max(values)], "lane_days": LANE_DAYS,
            "unique_real_points": UNIQUE_POINTS, "unique_point_days": 896,
            "lane_days_per_second": LANE_DAYS / median,
            "warm_cost_per_lane_day": warm_costs}


def validate_checkpoint_io(io, checkpoint_bytes):
    demand(io.get("available") is True and all(type(io.get(key)) in (int, float)
        and math.isfinite(io[key]) and io[key] >= 0
        for key in ("serialization_seconds", "atomic_write_seconds")),
        "actual checkpoint serialization and atomic-write times required")
    demand(io.get("bytes") == checkpoint_bytes,
           "checkpoint IO byte count differs from complete sealed checkpoint")


def stage_gate_names(size):
    references = {f"capacity_B{size}_lane_{index}_source_{index % UNIQUE_POINTS}_GPU_reference"
                  for index in range(LANES)}
    internal = {f"capacity_B{size}_frozen_reset_repeat",
                f"capacity_B{size}_permutation",
                f"capacity_B{size}_replacement_isolation"}
    return references, internal


def validate_stage_header(directory, name, identity, manifest_record, expected_code_sha, approval):
    demand(name in STAGE_NAMES, "only fixed-1024 B512/B1024 stages are admissible")
    size = 512 if name == STAGE_NAMES[0] else 1024
    chunk_count = 2 if size == 512 else 1
    demand(manifest_record.get("status") == "complete", "stage is not atomically complete")
    path = (directory / manifest_record.get("path", "")).resolve()
    demand(path.is_relative_to(directory.resolve()) and path.is_file(), "checkpoint is missing or escapes stage directory")
    digest = sha256(path)
    demand(digest == manifest_record.get("sha256"), "complete checkpoint SHA mismatch")
    with path.open("rb") as stream:
        envelope = pickle.load(stream)
    demand(envelope.get("identity") == identity and envelope.get("name") == name
           and envelope.get("status") == "complete", "checkpoint envelope identity/status mismatch")
    record = envelope.get("record", {})
    from scripts.dev.global_server_reference import _evidence_mismatch
    for key, value in record.items():
        mismatch = _evidence_mismatch(value, manifest_record.get(key), "record." + key)
        demand(mismatch is None, f"checkpoint/manifest record disagreement: {mismatch}")
    demand(record.get("measurement_status") == "complete"
           and record.get("evidence_class") == "exploratory_capacity_performance"
           and record.get("scientific_acceptance") is False,
           "only completed exploratory capacity measurements are collectable")
    proof = record.get("provenance", {})
    expected = {"workload_kind": "capacity_screen_repeated_FULL64", "total_lanes": LANES,
        "unique_real_points": UNIQUE_POINTS, "lane_days": LANE_DAYS, "unique_point_days": 896,
        "chunk_size": size, "chunks_per_collection": chunk_count,
        "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48,
        "full_outputs": True, "full_two_block_workload": True}
    for key, value in expected.items():
        demand(type(proof.get(key)) is type(value) and proof.get(key) == value,
               f"exact fixed-1024 workload proof mismatch: {key}")
    env = record.get("execution_origin", {}).get("environment", {})
    demand(env.get("git_head") == expected_code_sha and env.get("jax") == "0.4.38"
           and env.get("jaxlib") == "0.4.38" and env.get("x64_enabled") is True
           and env.get("gpu_count") == 1, "execution code/native GPU runtime origin mismatch")
    device = env.get("device_details", [])
    demand(len(device) == 1 and device[0].get("platform") == "gpu" and "V100" in device[0].get("kind", ""),
           "actual single Tesla V100 device origin required")
    resources = proof.get("server_resources", {})
    slurm = resources.get("Slurm", {})
    tres = dict(part.split("=", 1) for part in slurm.get("ORCHJAX_SLURM_ALLOC_TRES", "").split(",") if "=" in part)
    demand(str(slurm.get("SLURM_JOB_ID")) == approval.job_id
           and slurm.get("SLURM_JOB_PARTITION") == "gnall"
           and resources.get("allocated_CPU_request_count") == 4,
           "stage did not use the explicitly approved job and CPU/GPU allocation")
    demand(tres.get("cpu") == "4" and tres.get("gres/gpu") == "1" and tres.get("node") == "1",
           "stage allocation TRES must be one node, four CPUs, one GPU")
    demand(env.get("hostname") == resources.get("hostname") == approval.node
           and slurm.get("SLURM_JOB_NODELIST") == approval.node,
           "stage actual node differs from explicit approval")
    physical = env.get("physical_gpu", {})
    demand(physical.get("uuid") == approval.gpu_uuid
           and physical.get("driver_version") == approval.driver_version
           and str(slurm.get("SLURM_JOB_GPUS")) == approval.slurm_global_gpu_id,
           "stage physical GPU identity differs from observed startup pins")
    return envelope, record, digest


def validate_stage(directory, name, identity, manifest_record, expected_code_sha, approval,
                   reference, base, sentinel_paths, decision):
    import numpy as np

    envelope, record, digest = validate_stage_header(directory, name, identity, manifest_record,
                                                     expected_code_sha, approval)
    size = 512 if name == STAGE_NAMES[0] else 1024
    output = envelope["value"]["outputs"]
    demand(isinstance(output, (tuple, list)) and len(output) == 2
           and all(isinstance(block, (tuple, list)) and len(block) == 2 for block in output),
           "two full state/day-output block boundaries required")
    actual, actual_tree = base.jax.tree_util.tree_flatten(output)
    expected, expected_tree = base.jax.tree_util.tree_flatten(reference)
    demand(actual_tree == expected_tree and len(actual) == len(expected) == 760,
           "exact 760-leaf native output PyTree required")
    for index, (got, source) in enumerate(zip(actual, expected, strict=True)):
        got, source = np.asarray(got), np.asarray(source)
        demand(source.ndim > 0 and source.shape[0] == UNIQUE_POINTS
               and got.shape == (LANES, *source.shape[1:]) and got.dtype == source.dtype,
               f"native full-1024 shape/dtype mismatch at leaf {index}")
    for block in output:
        days = base.jax.tree_util.tree_leaves(block[1])
        demand(days and all(np.asarray(leaf).ndim >= 2 and np.asarray(leaf).shape[:2] == (LANES, 7)
                            for leaf in days), "both seven-day daily-output blocks are required")
    gates = record.get("gates", [])
    reference_names, internal_names = stage_gate_names(size)
    gate_map = {gate.get("name"): gate for gate in gates}
    demand(len(gates) == 1027 and len(gate_map) == 1027
           and set(gate_map) == reference_names | internal_names,
           "exactly 1024 independent FULL64 comparisons plus three internal gates required")
    for gate_name, gate in gate_map.items():
        demand(gate.get("atol") == 1e-8 and gate.get("rtol") == 1e-10
               and gate.get("compared_leaves") == 760,
               f"original full-leaf strict gate policy required: {gate_name}")
        demand(gate.get("passed") is True and not gate.get("failures"),
               f"strict fixed-1024 gate failed: {gate_name}")
        for allowed in gate.get("allowed_sentinels", []):
            demand(sentinel_paths.get(allowed["path"]) == allowed["contract"],
                   "candidate sentinel differs from immutable original FULL64 source")
    comparisons = []
    for lane in range(LANES):
        compared = base.compare(base.jax.tree_util.tree_map(lambda value: value[lane], output),
                                base.jax.tree_util.tree_map(lambda value: value[lane % UNIQUE_POINTS], reference),
                                sentinel_paths=sentinel_paths)
        demand(compared.get("passed") is True, f"actual saved-output reference comparison failed at lane {lane}")
        comparisons.append({"lane": lane, "source_index": lane % UNIQUE_POINTS, **compared})
    demand(record.get("strict_numerical_passed") is True, "strict numerical stage summary must pass")
    measurements = envelope["value"].get("measurements", {})
    from scripts.dev.global_server_reference import _evidence_mismatch
    demand(_evidence_mismatch(measurements, record.get("timing"), "record.timing") is None,
           "checkpoint measurements differ from stage record timing")
    metrics = {boundary: validate_sample(measurements[boundary], f"{name}.{boundary}")
               for boundary in ("core", "wrapper")}
    startup = record.get("startup", {})
    for key in ("lower_seconds", "compile_seconds", "host_stack_seconds", "device_transfer_seconds",
                "first_full1024_call_seconds"):
        value = startup.get(key)
        demand(type(value) in (int, float) and math.isfinite(value) and value >= 0,
               f"separate actual startup boundary required: {key}")
    io = manifest_record.get("checkpoint_io", {})
    validate_checkpoint_io(io, (directory / manifest_record["path"]).stat().st_size)
    proof = record["provenance"]
    memory = proof.get("memory_preflight", {})
    physical = memory.get("physical_gpu", {})
    estimate = memory.get("estimated_device_bytes")
    demand(memory.get("shape") == size
           and type(memory.get("resident_1024_input_bytes")) is int and memory["resident_1024_input_bytes"] > 0
           and type(memory.get("full_1024_output_bytes")) is int and memory["full_1024_output_bytes"] > 0
           and type(estimate) is int and type(physical.get("free_bytes")) is int
           and type(physical.get("total_bytes")) is int
           and physical["total_bytes"] >= 30 * 1024**3
           and physical["free_bytes"] >= estimate + 2 * 1024**3,
           "actual stacked-input device memory guard or physical slack failed")
    host = proof.get("host_memory_preflight", {})
    demand(type(host.get("MemAvailable_bytes")) is int
           and type(host.get("conservative_required_bytes")) is int
           and host["MemAvailable_bytes"] >= host["conservative_required_bytes"],
           "actual per-stage host-memory resource guard is missing or failed")
    if size == 1024:
        demand(type(decision.get("resident_1024_input_bytes_upper_bound")) is int
               and decision["resident_1024_input_bytes_upper_bound"] > 0
               and decision.get("input_byte_scope") == "conservative pre-stack raw lane-list upper bound; counts repeated shared-static leaves; actual-stack bytes recorded in stage memory_preflight",
               "B1024 raw input memory screen is missing or has the wrong scope")
    return {"stage": name, "batch_size": size, "chunks_per_collection": 2 if size == 512 else 1,
        "checkpoint_sha256": digest, "checkpoint_bytes": (directory / manifest_record["path"]).stat().st_size,
        "total_lanes": LANES, "unique_real_points": UNIQUE_POINTS, "lane_days": LANE_DAYS,
        "unique_point_days": 896, "native_output_leaves": 760,
        "strict_numerical_passed": True, "actual_saved_output_reference_passed": True,
        "actual_saved_output_reference_comparison_count": len(comparisons),
        "actual_saved_output_reference_comparisons": comparisons,
        "core": metrics["core"], "wrapper": metrics["wrapper"],
        "startup": startup, "checkpoint_IO": io,
        "execution_origin": record["execution_origin"], "allocation": record["provenance"]["server_resources"],
        "memory": {key: value for key, value in record["provenance"].items()
                   if "memory" in key.lower() or "rss" in key.lower() or "vram" in key.lower()}}


def report_stage_consistency(report, name, record):
    from scripts.dev.global_server_reference import _evidence_mismatch

    mismatch = _evidence_mismatch(record, report.get("stages", {}).get(name), "stage")
    demand(mismatch is None, f"report/manifest stage mismatch: {mismatch}")
    gate_map = {entry.get("name"): entry for entry in report.get("gates", [])}
    stage_gates = record.get("gates", [])
    demand(all(gate_map.get(entry.get("name")) == entry for entry in stage_gates),
           "report gate list differs from sealed stage gate records")
    label = name.split("_fixed", 1)[0]
    for boundary in ("lower", "compile"):
        key = f"{label}_{boundary}_seconds"
        demand(report.get("timing", {}).get(key) == record.get("startup", {}).get(f"{boundary}_seconds"),
               f"report/stage {boundary} timing mismatch")


def validate_stage_decision(report, size, historical_origin, b512_result=None):
    key = f"fixed1024_B{size}_decision"
    decision = report.get(key, {})
    demand(decision.get("shape") == size and decision.get("permitted") is True
           and decision.get("minimum_remaining_seconds") == 2700
           and decision.get("remaining_seconds", 0) >= 2700
           and decision.get("requires_performance_improvement") is False
           and decision.get("policy") == "fixed1024_resource_only_2700s_2GiB_GPU_8GiB_host_slack",
           f"{key} did not pass its time/memory-only admission guard")
    estimated = decision.get("estimated_device_bytes")
    free = decision.get("free_device_bytes")
    demand(type(estimated) is int and type(free) is int
           and free >= estimated + 2 * 1024**3,
           f"{key} device memory estimate lacks required 2GiB physical slack")
    memory_origin = decision.get("compiler_memory_origin", {})
    demand(memory_origin.get("historical_B512_sha256") == LEGACY_B512_SHA
           and memory_origin.get("historical_actual_analysis")
           == historical_origin.get("provenance", {}).get("executable_memory_analysis"),
           f"{key} must retain the exact historical B512 compiler-memory origin")
    if size == 1024:
        memory_record = memory_origin.get("current_memory_records", {}).get("512",
            memory_origin.get("current_memory_records", {}).get(512))
        demand(b512_result is not None and memory_record
               and memory_record.get("temp_size_in_bytes") >= b512_result.get("memory", {}).get("executable_memory_analysis", {}).get("temp_size_in_bytes", 0),
               "B1024 guard must include current B512 actual compiler-memory evidence")
    host = decision.get("host_memory", {})
    demand(type(host.get("MemAvailable_bytes")) is int
           and type(host.get("conservative_required_bytes")) is int
           and host["MemAvailable_bytes"] >= host["conservative_required_bytes"],
           f"{key} host-memory resource guard is missing or failed")


def validate_shared_origin(results):
    demand(len(results) >= 1, "no strictly valid fixed-1024 stage")
    first = results[0]["execution_origin"]
    first_env = first["environment"]
    for result in results[1:]:
        origin = result["execution_origin"]
        env = origin["environment"]
        demand(origin.get("pid") == first.get("pid"), "B512/B1024 stages did not share one process PID")
        demand(env == first_env, "B512/B1024 stages did not share the exact recorded execution environment")
        demand(result["allocation"].get("Slurm") == results[0]["allocation"].get("Slurm"),
               "B512/B1024 stages did not share one Slurm job/allocation")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("code-dir", "stage-dir", "admission", "sacct-file", "b512-evidence-dir", "output"):
        result.add_argument("--" + name, required=True, type=Path)
    result.add_argument("--expected-code-sha", required=True)
    for name in ("job-id", "job-name", "node"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--gpu-uuid", required=True,
        help="Observed startup physical GPU identity pin, not an additional approval.")
    result.add_argument("--slurm-global-gpu-id", required=True,
        help="Observed startup Slurm GPU identity pin, not an additional approval.")
    result.add_argument("--driver-version", required=True,
        help="Observed startup GPU driver identity pin, not an additional approval.")
    result.add_argument("--approved-nodes", type=int, choices=[1], required=True)
    result.add_argument("--approved-tasks", type=int, choices=[1], required=True)
    result.add_argument("--approved-cpus", type=int, choices=[4], required=True)
    result.add_argument("--approved-gpus", type=int, choices=[1], required=True)
    result.add_argument("--approved-wall-seconds", type=int, choices=[5400], required=True)
    return result


def admit_code_checkout(code_dir, expected_code_sha):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_dir, text=True).strip()
    demand(head == expected_code_sha, "collector checkout HEAD differs from explicit expected-code-sha")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                     cwd=code_dir, text=True)
    demand(not status.strip(), "code checkout must be clean before execution imports")


def empty_result(args, reason, accounting=None):
    result = {"job_id": args.job_id, "code_commit": args.expected_code_sha, "contract": CONTRACT,
        "scientific_acceptance": False, "valid_exploratory_stages": [],
        "withheld_stages": [{"stage": name, "reason": reason} for name in STAGE_NAMES],
        "incomplete_stages": [], "imported_historical_B512_evidence": None,
        "accounting": accounting, "comparison_scope": "same ordered 64 real FULL64 inputs repeated cyclically across 1024 lanes; common 14336 lane-day denominator",
        "K_amortization": "not reported; complete phase setup attribution is unavailable",
        "scientific_claims": {"cpu_gpu_ratio": None, "ten_x_improvement": None, "general_scientific_acceptance": False}}
    if accounting is not None:
        apply_parent_accounting(result, accounting)
    return result


def retain_shared_stages(accepted, withheld):
    """A mismatch between stages withholds B1024 while preserving a strict B512 seal."""
    if len(accepted) != 2:
        return accepted, withheld
    try:
        validate_shared_origin(accepted)
    except ValueError as error:
        withheld.append({"stage": STAGE_NAMES[1], "reason": str(error)})
        return accepted[:1], withheld
    return accepted, withheld


def apply_parent_accounting(result, accounting):
    result["accounting"] = accounting
    result["parent_successfully_completed"] = accounting["state"] == "COMPLETED" and accounting["exit_code"] == "0:0"
    result["stage_promotion_withheld_due_to_wall_overrun"] = accounting.get("approved_wall_exceeded", False)
    if accounting.get("approved_wall_exceeded"):
        result["valid_exploratory_stages"] = []
        result["withheld_stages"] = [{"stage": name,
            "reason": "terminal parent exceeded the approved 5400-second wall envelope"} for name in STAGE_NAMES]
        result["capacity_curve_available"] = False
        result["capacity_curve_scope"] = "stages withheld because observed terminal wall exceeded approved envelope"
    return result


def main():
    args = parser().parse_args()
    demand(args.job_id.isdecimal(), "explicit numeric parent JobID is required")
    approval = args
    accounting = None
    try:
        accounting = parse_raw9(args.sacct_file.read_text(), approval)
        if accounting.get("approved_wall_exceeded"):
            result = empty_result(args, "terminal parent exceeded approved wall envelope; actual cost retained", accounting)
            write_result(args.output, result)
            return 1
        admit_code_checkout(args.code_dir, args.expected_code_sha)
        os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true", PYTHONDONTWRITEBYTECODE="1")
        sys.path.insert(0, str(args.code_dir.resolve()))
        from scripts.dev import benchmark_global_batch as base
        from scripts.dev import benchmark_global_gpu_capacity as capacity
        from scripts.dev.global_server_reference import load_reference
        from scripts.dev.global_source_compatibility import source_identity

        admission = json.loads(args.admission.read_text())
        manifest_path = args.stage_dir / "stage_manifest.json"
        report_path = args.stage_dir / "report.json"
        if not manifest_path.is_file():
            result = empty_result(args, "stage manifest absent; denied or unfinished stages are withheld", accounting)
            write_result(args.output, result)
            return 0
        manifest = json.loads(manifest_path.read_text())
        identity = manifest.get("identity", {})
        identity_checks(identity, admission, args.expected_code_sha)
        demand(set(manifest.get("stages", {})).issubset(STAGE_NAMES), "unexpected stage names in fixed-1024 manifest")
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        demand(source_identity(args.code_dir) == {
            "science_sha256": SCIENCE_SHA, "actual_science_source_sha256": ACTUAL_SCIENCE_SHA,
            "execution_source_compatibility": identity["execution_source_compatibility"]},
            "executed source certificate differs from required immutable Linux source identity")

        # The B512 checkpoint is only a verified resource/scientific origin; no prior timing enters the curve.
        baseline_identity = capacity.b512_evidence_identity(identity, identity["lane_origins"],
                                                             identity["numerical_reference_sha256"])
        historical_origin = capacity.import_b512_evidence(args.b512_evidence_dir, baseline_identity,
                                                           args.stage_dir, base.jax, reference=None)
        demand(historical_origin.get("sha256") == LEGACY_B512_SHA
               and historical_origin.get("bytes") == LEGACY_B512_BYTES
               and historical_origin.get("source_stage") == "capacity_B512_fixed512_v1"
               and historical_origin.get("role") == "historical numerical/schema/compiler-temp/RSS evidence only; excluded from current timing curve"
               and "timing" not in historical_origin,
               "exact sealed B512 origin-only evidence required")

        admitted_ref = admission["FULL64_reference_origin"]
        reference, actual_ref = load_reference(Path(admitted_ref["path"]).parent, identity)
        from scripts.dev.global_server_reference import _evidence_mismatch
        admitted_proof = {key: value for key, value in admitted_ref.items() if key != "backend_conditions"}
        actual_proof = {key: value for key, value in actual_ref.items() if key != "backend_conditions"}
        demand(_evidence_mismatch(admitted_proof, actual_proof, "admitted_reference") is None,
               "original admitted FULL64 reference identity/schema changed")
        demand(actual_ref.get("backend_conditions", {}).get("source")
               == admitted_ref.get("backend_conditions", {}).get("source"),
               "original FULL64 source backend certificate changed")
        demand(_evidence_mismatch(actual_ref, report.get("reference_origin"), "report.reference_origin") is None,
               "run report FULL64 reference origin mismatch")
        with Path(admitted_ref["path"]).open("rb") as stream:
            reference_envelope = pickle.load(stream)
        sentinel_paths = {}
        for gate in reference_envelope.get("record", {}).get("gates", []):
            for allowed in gate.get("allowed_sentinels", []):
                prior = sentinel_paths.setdefault(allowed["path"], allowed["contract"])
                demand(prior == allowed["contract"], "original FULL64 sentinel contract conflict")

        stages = manifest.get("stages", {})
        accepted, withheld, incomplete = [], [], []
        for name in STAGE_NAMES:
            record = stages.get(name)
            if not isinstance(record, dict) or record.get("status") != "complete":
                decision_key = "fixed1024_B512_decision" if name == STAGE_NAMES[0] else "fixed1024_B1024_decision"
                decision = report.get(decision_key, {})
                reason = decision.get("reasons") or [record.get("reason") if isinstance(record, dict) else "stage absent"]
                incomplete.append({"stage": name, "status": record.get("status") if isinstance(record, dict) else "absent",
                                   "reason": reason})
                continue
            try:
                report_stage_consistency(report, name, record)
                size = 512 if name == STAGE_NAMES[0] else 1024
                prior = accepted[0] if size == 1024 and accepted else None
                validate_stage_decision(report, size, historical_origin, prior)
                result = validate_stage(args.stage_dir, name, identity, record, args.expected_code_sha,
                                        approval, reference, base, sentinel_paths,
                                        report[f"fixed1024_B{size}_decision"])
                accepted.append(result)
            except (ValueError, KeyError, TypeError, AssertionError, OSError) as error:
                withheld.append({"stage": name, "reason": str(error)})
        # B1024 remains conditional on a completed and strictly passing new B512 control.
        accepted, withheld = retain_shared_stages(accepted, withheld)
        complete_sizes = [size for size, name in zip((512, 1024), STAGE_NAMES)
                          if stages.get(name, {}).get("status") == "complete"]
        demand(report.get("completed_shapes", []) == complete_sizes
               and report.get("comparison_shapes", []) == complete_sizes,
               "report completed/comparison shape lists disagree with atomic fixed-1024 stages")
        result = {"job_id": args.job_id, "code_commit": args.expected_code_sha, "contract": CONTRACT,
            "scientific_acceptance": False, "execution_source_identity": source_identity(args.code_dir),
            "valid_exploratory_stages": accepted, "withheld_stages": withheld,
            "incomplete_stages": incomplete,
            "capacity_curve_available": len(accepted) == 2,
            "capacity_curve_scope": "new fixed-1024 B512 versus B1024" if len(accepted) == 2
                else "B1024 unmeasured or withheld; no B512-versus-B1024 curve",
            "imported_historical_B512_evidence": historical_origin,
            "historical_B512_timing_promoted": False,
            "observed_allocation_identity_pins": {"gpu_uuid": args.gpu_uuid,
                "driver_version": args.driver_version, "slurm_global_gpu_id": args.slurm_global_gpu_id},
            "run_report_status": report.get("status"),
            "fixed1024_B512_decision": report.get("fixed1024_B512_decision"),
            "fixed1024_B1024_decision": report.get("fixed1024_B1024_decision"),
            "comparison_scope": "same ordered 64 real FULL64 inputs repeated cyclically across 1024 lanes; B512 two chunks and B1024 one chunk; common 14336 lane-day denominator",
            "cost_boundaries": "core, prepared wrapper, lower, compile, first full call, host stack, device transfer, checkpoint IO, and one terminal parent allocation are reported separately",
            "K_amortization": "not reported; complete phase setup attribution is unavailable",
            "scientific_claims": {"cpu_gpu_ratio": None, "ten_x_improvement": None,
                                  "training_or_ad": False, "general_scientific_acceptance": False},
            "collector_execution": "CPU saved-array validation only; no model invocation, compilation, GPU initialization, or remote access"}
        apply_parent_accounting(result, accounting)
        write_result(args.output, result)
        return 1 if withheld else 0
    except (ValueError, KeyError, TypeError, AssertionError, OSError, subprocess.SubprocessError) as error:
        result = empty_result(args, str(error), accounting)
        write_result(args.output, result)
        return 1


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"job_id": result["job_id"], "valid_stages": len(result["valid_exploratory_stages"]),
                      "withheld_stages": len(result["withheld_stages"]),
                      "incomplete_stages": len(result["incomplete_stages"]), "output": str(path)}))


if __name__ == "__main__":
    raise SystemExit(main())
