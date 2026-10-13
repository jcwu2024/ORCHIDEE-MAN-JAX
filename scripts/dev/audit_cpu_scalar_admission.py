"""Independently recompute the saved FULL64 scalar-to-GPU gates only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if (ROOT / "scripts").is_dir():
    sys.path.insert(0, str(ROOT))


EXPECTED_SOURCE = "b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8"
EXPECTED_TREE = "75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba"
EXPECTED_SCIENCE = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"
EXPECTED_BASELINE = "133e35aec7b83b9d79888c46092f64d518e97299"
EXPECTED_GPU_SHA = "a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6"
EXPECTED_PREPARED_SHA = "be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7"
EXPECTED_EXECUTION_COMMIT = "03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8"
EXPECTED_SCALAR_SHA = "1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa"
EXPECTED_MANIFEST_SHA = "269c4e04b2404177dbb0c2b350673be0f41fa9a63dbfa5055af35f2d9550a641"
EXPECTED_REPORT_SHA = "81bd4ceb413c079eb8f1fa7c00c61f4a8d74ddb75278d2a49bd1e5fd82db4da0"
EXPECTED_PARENT_ACCOUNTING_SHA = "a04c687530f390ad1d5f1d3668293a552b3a9b37f0cd05ef1f264660c2f37113"
EXPECTED_PARENT_RAW_SHA = "e059a19bb41ff88ee685de0e76a13727f9861039c6373832e320dad541321bd6"
STAGE_NAME = "B64_server_scalar_v1"
GPU_STAGE = "B4_group_references"
RESET_LIMITATION = "saved frozen-reset gate is sealed; reset arrays were not independently retained for recomputation"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sealed_stage(directory: Path, name: str):
    manifest_path, report_path = directory / "stage_manifest.json", directory / "report.json"
    manifest_bytes, report_bytes = manifest_path.read_bytes(), report_path.read_bytes()
    manifest, report = json.loads(manifest_bytes), json.loads(report_bytes)
    identity = manifest["identity"]
    record = manifest["stages"][name]
    report_record = report["stages"][name]
    from scripts.dev.global_server_reference import _evidence_mismatch

    mismatch = _evidence_mismatch(record, report_record, f"report.stages.{name}")
    if mismatch is not None:
        raise ValueError(f"report and manifest stage evidence differs at {mismatch}")
    if (record.get("status") != "complete"
            or name == STAGE_NAME and record.get("measurement_status") != "complete"):
        raise ValueError(f"{name}: complete stage required")
    checkpoint = (directory / record["path"]).resolve()
    if not checkpoint.is_relative_to(directory.resolve()) or not checkpoint.is_file():
        raise ValueError(f"{name}: checkpoint escapes stage directory or is missing")
    payload = checkpoint.read_bytes()
    checkpoint_sha = sha256(payload)
    if checkpoint_sha != record.get("sha256"):
        raise ValueError(f"{name}: checkpoint checksum mismatch")
    envelope = pickle.loads(payload)
    if (envelope.get("identity") != identity or envelope.get("name") != name
            or envelope.get("status") != "complete"):
        raise ValueError(f"{name}: checkpoint envelope identity/status mismatch")
    for field in ("gates", "provenance", "timing", "measurement_status", "evidence_class",
            "strict_numerical_passed", "execution_origin"):
        mismatch = _evidence_mismatch(envelope.get("record", {}).get(field), record.get(field), f"record.{field}")
        if mismatch is not None:
            raise ValueError(f"{name}: checkpoint and manifest evidence differs at {mismatch}")
    return {"manifest": manifest, "report": report, "identity": identity, "record": record,
        "value": envelope["value"], "checkpoint": checkpoint,
        "manifest_sha256": sha256(manifest_bytes), "report_sha256": sha256(report_bytes),
        "checkpoint_sha256": checkpoint_sha}


def gate_matches(recorded, measured):
    fields = ("passed", "compared_leaves", "max_abs", "max_bound_ratio", "failures", "atol", "rtol",
        "nonfinite_policy", "allowed_sentinels")
    return all(recorded.get(field) == measured.get(field) for field in fields)


def audit(args):
    scalar_dir, gpu_dir = args.scalar_dir.resolve(), args.gpu_reference_dir.resolve()
    scalar = read_sealed_stage(scalar_dir, STAGE_NAME)
    gpu = read_sealed_stage(gpu_dir, GPU_STAGE)
    if (scalar["checkpoint_sha256"] != EXPECTED_SCALAR_SHA
            or scalar["manifest_sha256"] != EXPECTED_MANIFEST_SHA
            or scalar["report_sha256"] != EXPECTED_REPORT_SHA):
        raise ValueError("saved scalar checkpoint/manifest/report does not match the exact terminal v3 pins")
    identity, record = scalar["identity"], scalar["record"]
    source_identity = record.get("execution_origin", {}).get("environment", {})
    command = source_identity.get("command", [])
    if (identity.get("mode") != "scalar" or identity.get("CPU_budget") != 1
            or identity.get("baseline_ref") != EXPECTED_BASELINE
            or identity.get("science_sha256") != EXPECTED_SCIENCE
            or identity.get("numerical_reference_sha256") != EXPECTED_GPU_SHA
            or len(identity.get("lanes", [])) != 64
            or record.get("strict_numerical_passed") is not False
            or record.get("measurement_status") != "complete"):
        raise ValueError("saved scalar source identity/workload/strict outcome differs from the expected v3 evidence")
    proof = record.get("provenance", {})
    if (proof.get("point_ids") != [lane["landpoint_id"] for lane in identity["lanes"]]
            or proof.get("unique_real_points") != 64 or proof.get("blocks") != [[2, 8], [9, 15]]
            or proof.get("half_hours_per_day") != 48 or proof.get("full_outputs") is not True
            or proof.get("full_two_block_workload") is not True or proof.get("frozen_reset_repeat") is not True):
        raise ValueError("saved scalar source does not seal the full 64-point two-block workload")
    resources = proof.get("server_resources", {})
    slurm = resources.get("Slurm", {})
    if slurm.get("SLURM_JOB_ID") != "15820467":
        raise ValueError("saved scalar source is not the expected parent job 15820467")
    if not command:
        raise ValueError("saved scalar execution command is missing")
    prepared_path = Path(command[command.index("--prepared-input") + 1]).resolve()
    config_path = Path(command[command.index("--config") + 1]).resolve()
    declared_gpu_dir = Path(command[command.index("--gpu-reference-dir") + 1]).resolve()
    if declared_gpu_dir != gpu_dir:
        raise ValueError("saved scalar execution GPU-reference directory mismatch")
    prepared_sha = file_sha256(prepared_path)
    if prepared_sha != EXPECTED_PREPARED_SHA:
        raise ValueError("saved scalar prepared-input artifact differs from the frozen FULL64 input")
    if gpu["record"].get("sha256") != EXPECTED_GPU_SHA:
        raise ValueError("GPU reference stage does not match the certified FULL64 reference SHA")

    execution_commit = source_identity.get("git_head")
    code_dir = Path(source_identity.get("cwd", "")).resolve()
    if execution_commit != EXPECTED_EXECUTION_COMMIT or not code_dir.is_dir():
        raise ValueError("saved scalar execution commit/code directory is not available")
    actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_dir, text=True).strip()
    tree_oid = subprocess.check_output(["git", "rev-parse", "HEAD:jax_orchidee"], cwd=code_dir, text=True).strip()
    if actual_head != execution_commit or tree_oid != EXPECTED_TREE:
        raise ValueError("saved scalar code directory does not match its execution commit/physical science tree")
    sys.path.insert(0, str(code_dir))
    from scripts.dev.global_source_compatibility import source_identity as science_source_identity
    from scripts.dev import benchmark_global_scale as scale

    science = science_source_identity(code_dir)
    if (science["science_sha256"] != EXPECTED_SCIENCE
            or science["actual_science_source_sha256"] != EXPECTED_SOURCE):
        raise ValueError("saved scalar execution source does not match the audited science digest")

    from scripts.dev import benchmark_global_batch as base

    input_lanes, second = scale.read_prepared(prepared_path, config_path)
    point_ids = [lane.landpoint_id for lane in input_lanes]
    gpu_point_ids = [lane["landpoint_id"] for lane in gpu["identity"].get("lanes", [])]
    if (point_ids != [lane["landpoint_id"] for lane in identity["lanes"]]
            or gpu["identity"].get("lanes") != identity.get("lanes")
            or gpu_point_ids != point_ids or len(point_ids) != 64 or len(set(point_ids)) != 64):
        raise ValueError("prepared input, scalar source, and GPU reference must have the same 64 unique points in exact order")
    from types import SimpleNamespace

    semantic_args = SimpleNamespace(baseline_ref=EXPECTED_BASELINE)
    recomputed_inputs = [{"landpoint_id": lane.landpoint_id,
        "inputs_sha256": base.semantic_reference_identity(semantic_args, lane, second_lane)}
        for lane, second_lane in zip(input_lanes, second, strict=True)]
    if recomputed_inputs != identity["lanes"]:
        raise ValueError("prepared-input order/content digests differ from the sealed scalar lane identities")

    scalar_value, gpu_value = scalar["value"], gpu["value"]
    actual_outputs = scalar_value.get("outputs") if isinstance(scalar_value, dict) else scalar_value
    reference_outputs = gpu_value.get("outputs") if isinstance(gpu_value, dict) and "outputs" in gpu_value else gpu_value
    if actual_outputs is None or reference_outputs is None:
        raise ValueError("saved scalar or GPU stage lacks full output arrays")
    actual_leaves = base.jax.tree_util.tree_leaves(actual_outputs)
    reference_leaves = base.jax.tree_util.tree_leaves(reference_outputs)
    if len(actual_leaves) != 760 or len(reference_leaves) != 760:
        raise ValueError("saved scalar and FULL64 GPU reference must each contain 760 output leaves")
    schema = []
    for index, (actual, reference) in enumerate(zip(actual_leaves, reference_leaves, strict=True)):
        a, r = base.np.asarray(actual), base.np.asarray(reference)
        if (a.shape != r.shape or a.dtype != r.dtype or a.ndim == 0
                or a.shape[0] != 64 or a.dtype.kind not in "biuf"):
            raise ValueError(f"saved output schema mismatch at leaf {index}: {a.shape}/{a.dtype} != {r.shape}/{r.dtype}")
        schema.append({"leaf": index, "shape": list(a.shape), "dtype": str(a.dtype)})
    gates = {gate.get("name"): gate for gate in record.get("gates", [])}
    point_ids = [lane["landpoint_id"] for lane in identity["lanes"]]
    results = []
    for index, point in enumerate(point_ids):
        name = f"B64_scalar_lane_{index}_{point}_GPU_reference"
        saved = gates.get(name)
        if saved is None:
            raise ValueError(f"saved GPU gate missing: {name}")
        expected = base.jax.tree_util.tree_map(lambda value: base.np.asarray(value)[index], reference_outputs)
        actual = base.jax.tree_util.tree_map(lambda value: base.np.asarray(value)[index], actual_outputs)
        sentinels = {item["path"]: item["contract"] for item in saved.get("allowed_sentinels", [])}
        measured = base.compare(actual, expected, sentinel_paths=sentinels)
        if not gate_matches(saved, measured):
            raise ValueError(f"saved scalar gate differs from recomputed actual arrays: {name}")
        if not measured["passed"]:
            from scripts.dev.global_server_reference import finite_tolerance_failure
            if not finite_tolerance_failure(saved):
                raise ValueError(f"hard numerical failure in saved scalar arrays: {name}")
        results.append({"name": name, "passed": measured["passed"],
            "compared_leaves": measured["compared_leaves"], "max_abs": measured["max_abs"],
            "max_bound_ratio": measured["max_bound_ratio"], "failure_count": len(measured["failures"])})
    failed = [item for item in results if not item["passed"]]
    if (len(results) != 64 or sum(item["failure_count"] for item in results) != 4
            or len(failed) != 3 or record.get("strict_numerical_passed") is not False):
        raise ValueError("recomputed scalar GPU gate summary differs from the sealed strict-failure evidence")
    reset_name = "B64_scalar_frozen_reset_repeat"
    reset_gate = gates.get(reset_name)
    if reset_gate is None or reset_gate.get("passed") is not True:
        raise ValueError("saved frozen-reset repeat gate is not sealed as passed")

    accounting_path = scalar_dir.parent / "parent-accounting.json"
    accounting_bytes = accounting_path.read_bytes()
    accounting_sha = sha256(accounting_bytes)
    accounting = json.loads(accounting_bytes)
    if accounting_sha != EXPECTED_PARENT_ACCOUNTING_SHA:
        raise ValueError("old parent accounting file differs from its terminal evidence pin")
    raw_record = accounting.get("raw_record", "")
    raw_sha = sha256(raw_record.encode("utf-8"))
    if (accounting.get("job_id") != "15820467" or accounting.get("source") != "sacct"
            or accounting.get("scope") != "whole_slurm_parent_job"
            or accounting.get("terminal_status") != "FAILED" or accounting.get("Elapsed") != "00:36:37"
            or accounting.get("wall_seconds") != 2197 or accounting.get("allocated_CPU_count") != 48
            or accounting.get("GPU_count") != 0 or accounting.get("AllocTRES") != "billing=48,cpu=48,node=1"
            or raw_record != "15820467|FAILED|00:36:37|billing=48,cpu=48,node=1\n"
            or raw_sha != EXPECTED_PARENT_RAW_SHA or accounting.get("raw_record_sha256") != raw_sha):
        raise ValueError("old whole-parent Slurm accounting record differs from the terminal evidence")

    scalar_source = {"job_id": "15820467", "execution_commit": execution_commit,
        "execution_code_dir": str(code_dir), "prepared_input_path": str(prepared_path),
        "prepared_input_sha256": prepared_sha, "gpu_reference_path": str(gpu_dir),
        "gpu_reference_sha256": EXPECTED_GPU_SHA, "certified_science_sha256": EXPECTED_SCIENCE,
        "actual_science_source_sha256": EXPECTED_SOURCE, "physical_science_tree_oid": EXPECTED_TREE,
        "baseline_ref": EXPECTED_BASELINE, "source_identity": identity,
        "stage_name": STAGE_NAME, "stage_path": scalar["checkpoint"].name,
        "stage_sha256": scalar["checkpoint_sha256"], "manifest_sha256": scalar["manifest_sha256"],
        "report_sha256": scalar["report_sha256"], "runtime": identity.get("runtime"),
        "workload": {"unique_points": 64, "point_days": 896, "blocks": [[2, 8], [9, 15]],
            "half_hours_per_day": 48, "output_leaves": 760},
        "point_ids": point_ids,
        "input_identity": identity["lanes"],
        "gpu_reference_stage_sha256": gpu["record"]["sha256"],
        "whole_parent_accounting": {"path": str(accounting_path), "sha256": accounting_sha,
            "raw_record": raw_record, "raw_record_sha256": raw_sha, "job_id": accounting["job_id"],
            "terminal_status": accounting["terminal_status"], "Elapsed": accounting["Elapsed"],
            "wall_seconds": accounting["wall_seconds"], "AllocTRES": accounting["AllocTRES"],
            "allocated_CPU_count": 48, "GPU_count": 0,
            "allocated_parent_CPU_request_core_hours": 48 * 2197 / 3600,
            "primary_cost_cny_estimate": 48 * 2197 / 3600 * 0.05,
            "conservative_cost_cny_estimate": 48 * 2197 / 3600 * 0.07,
            "counted_once_as_source_parent": True}}
    audit = {"strict_numerical_passed": False, "GPU_gate_count": 64,
        "GPU_passed_gate_count": len(results) - len(failed), "GPU_tolerance_failed_gate_count": len(failed),
        "GPU_tolerance_failure_count": sum(item["failure_count"] for item in failed),
        "prepared_lane_identities_recomputed": True, "point_order_verified": True,
        "verified_unique_point_count": 64,
        "recomputed_GPU_gates": results, "output_schema": schema,
        "frozen_reset_gate": {"name": reset_name, "passed": True,
            "independent_array_recomputation": False},
        "frozen_reset_arrays_recomputed": False, "reset_evidence_limitation": RESET_LIMITATION,
        "scientific_acceptance": False}
    return {"schema": "global-cpu48-scalar-audit-v1", "status": "saved_arrays_audited",
        "scientific_acceptance": False, "scalar_source": scalar_source, "audit": audit,
        "audit_inputs": {"scalar_checkpoint_sha256": scalar["checkpoint_sha256"],
            "gpu_reference_checkpoint_sha256": gpu["checkpoint_sha256"],
            "prepared_input_sha256": prepared_sha}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar-dir", type=Path, required=True)
    parser.add_argument("--gpu-reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    try:
        affinity_cpu_ids = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_cpu_ids = None
    result["audit_execution"] = {"script_path": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()), "command": [sys.executable, *sys.argv],
        "scheduler_launch_command": os.environ.get("ORCJAX_AUDIT_LAUNCH_COMMAND"),
        "working_directory": str(Path.cwd()), "hostname": platform.node(),
        "python": sys.version, "platform": platform.platform(),
        "cpu_affinity_ids": affinity_cpu_ids, "cpu_affinity_count": len(affinity_cpu_ids)
            if affinity_cpu_ids is not None else None,
        "os_cpu_count": os.cpu_count(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "GPU_gate_count": result["audit"]["GPU_gate_count"],
        "failed_gate_count": result["audit"]["GPU_tolerance_failed_gate_count"],
        "reset_arrays_recomputed": result["audit"]["frozen_reset_arrays_recomputed"],
        "receipt_sha256": sha256(args.output.read_bytes())}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
