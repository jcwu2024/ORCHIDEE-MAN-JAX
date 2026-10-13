"""Read-only integrity collector for an exploratory CPU48/FULL64/B64 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REFERENCE_SHA = "a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6"
SCIENCE_SHA = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"
BASELINE_REF = "133e35aec7b83b9d79888c46092f64d518e97299"
POINT_DAYS = 64 * 14
CORE_METRIC = "B64_warm_frozen_state_core"
WRAPPER_METRIC = "B64_preprepared_wrapper_stack_transfer_materialize"
SCALAR_STAGE = "B64_server_scalar_v1"
BATCH_STAGE = "B64_server_batch_v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sealed_stage(directory: Path, name: str):
    manifest_path, report_path = directory / "stage_manifest.json", directory / "report.json"
    manifest_bytes, report_bytes = manifest_path.read_bytes(), report_path.read_bytes()
    manifest, report = json.loads(manifest_bytes), json.loads(report_bytes)
    identity = manifest["identity"]
    record = manifest["stages"][name]
    if record.get("status") != "complete" or record.get("measurement_status") != "complete":
        raise ValueError(f"{name}: complete numerical stage required")
    checkpoint = (directory / record["path"]).resolve()
    if not checkpoint.is_relative_to(directory.resolve()) or not checkpoint.is_file():
        raise ValueError(f"{name}: checkpoint path escapes stage directory or is missing")
    payload = checkpoint.read_bytes()
    if sha256(payload) != record.get("sha256"):
        raise ValueError(f"{name}: saved output checkpoint SHA mismatch")
    envelope = pickle.loads(payload)
    if envelope.get("identity") != identity or envelope.get("name") != name or envelope.get("status") != "complete":
        raise ValueError(f"{name}: saved output envelope identity/status mismatch")
    saved_record = envelope.get("record", {})
    from scripts.dev.global_server_reference import _evidence_mismatch
    for key in ("gates", "provenance", "timing", "measurement_status", "evidence_class",
                "strict_numerical_passed", "execution_origin"):
        mismatch = _evidence_mismatch(saved_record.get(key), record.get(key), f"record.{key}")
        if mismatch is not None:
            raise ValueError(f"{name}: checkpoint/manifest evidence differs at {mismatch}")
    return {"identity": identity, "record": record, "value": envelope["value"], "report": report,
        "manifest": manifest, "checkpoint": checkpoint, "checkpoint_sha256": sha256(payload),
        "manifest_sha256": sha256(manifest_bytes), "report_sha256": sha256(report_bytes)}


def require_tree_schema(actual, expected, size=64):
    import jax
    import numpy as np

    left, right = jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    if not left or len(left) != len(right):
        raise ValueError("saved output leaves are missing or differ from reference")
    schema = []
    for index, (candidate, reference) in enumerate(zip(left, right, strict=True)):
        a, r = np.asarray(candidate), np.asarray(reference)
        if a.shape != r.shape or a.dtype != r.dtype or a.ndim == 0 or a.shape[0] != size:
            raise ValueError(f"saved output schema mismatch at leaf {index}: {a.shape}/{a.dtype} != {r.shape}/{r.dtype}")
        schema.append({"leaf": index, "shape": list(a.shape), "dtype": str(a.dtype)})
    return schema


def slice_tree(tree, index):
    import jax
    import numpy as np
    return jax.tree_util.tree_map(lambda leaf: np.asarray(leaf)[index], tree)


def gate_matches(recorded, measured):
    keys = ("passed", "compared_leaves", "max_abs", "max_bound_ratio", "failures", "atol", "rtol",
        "nonfinite_policy", "allowed_sentinels")
    return all(recorded.get(key) == measured.get(key) for key in keys)


def check_gpu_gate_set(record, actual_outputs, reference_outputs):
    from scripts.dev import benchmark_global_batch as base
    from scripts.dev.global_server_reference import finite_tolerance_failure

    identity = record["identity"]
    points = [lane["landpoint_id"] for lane in identity["lanes"]]
    names = {gate["name"]: gate for gate in record["record"]["gates"]}
    results = []
    for index, point in enumerate(points):
        name = f"B64_batch_lane_{index}_{point}_GPU_reference"
        if name not in names:
            raise ValueError(f"required saved-output numerical gate missing: {name}")
        sentinels = {item["path"]: item["contract"] for item in names[name].get("allowed_sentinels", [])}
        measured = base.compare(slice_tree(actual_outputs, index), slice_tree(reference_outputs, index),
            sentinel_paths=sentinels)
        if not gate_matches(names[name], measured):
            raise ValueError(f"saved gate differs from recomputed actual arrays: {name}")
        if not measured["passed"] and not finite_tolerance_failure(names[name]):
            raise ValueError(f"hard numerical failure in saved arrays: {name}")
        results.append({"name": name, "passed": measured["passed"], "compared_leaves": measured["compared_leaves"],
            "max_abs": measured["max_abs"], "max_bound_ratio": measured["max_bound_ratio"],
            "failure_count": len(measured["failures"])})
    return results


def check_candidate_saved_arrays(batch, outputs):
    """Recompute batch isolation/reset gates from the separately sealed arrays."""
    from scripts.dev import benchmark_global_batch as base

    provenance = batch["record"].get("provenance", {})
    seal = provenance.get("validation_arrays")
    if not isinstance(seal, dict) or seal.get("schema") != "global_cpu48_candidate_validation_arrays_v1":
        raise ValueError("v4 candidate validation-array seal missing or malformed")
    directory = batch["checkpoint"].parent.resolve()
    path = Path(seal.get("path", "")).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("candidate validation-array path escapes the batch stage directory")
    raw = path.read_bytes()
    if len(raw) != seal.get("bytes") or sha256(raw) != seal.get("sha256"):
        raise ValueError("candidate validation-array sidecar checksum/size mismatch")
    evidence = pickle.loads(raw)
    identity = batch["identity"]
    if (evidence.get("schema") != seal["schema"] or evidence.get("candidate_identity") != identity
            or seal.get("schema") != "global_cpu48_candidate_validation_arrays_v1"):
        raise ValueError("candidate validation-array sidecar identity/schema mismatch")
    size = len(identity["lanes"])
    permutation_index = list(reversed(range(size)))
    replacement_index = list(range(size - 1)) + [0]
    if (evidence.get("permutation_index") != permutation_index
            or evidence.get("replacement_index") != replacement_index):
        raise ValueError("candidate validation-array permutation/replacement index contract mismatch")
    gates = {gate.get("name"): gate for gate in batch["record"].get("gates", [])}
    required = {
        f"B{size}_server_batch_cross_group_permutation": (evidence["permutation"], permutation_index),
        f"B{size}_server_batch_replacement_isolation": (evidence["replacement"], replacement_index),
        f"B{size}_batch_frozen_reset_repeat": (evidence["frozen_reset"], None),
    }
    results = []
    for name, (actual, indices) in required.items():
        recorded = gates.get(name)
        if recorded is None:
            raise ValueError(f"required candidate saved-array gate missing: {name}")
        expected = outputs if indices is None else base.jax.tree_util.tree_map(
            lambda leaf: base.np.asarray(leaf)[base.np.asarray(indices)], outputs)
        sentinels = {item["path"]: item["contract"] for item in recorded.get("allowed_sentinels", [])}
        measured = base.compare(actual, expected, sentinel_paths=sentinels)
        if not gate_matches(recorded, measured) or measured.get("passed") is not True:
            raise ValueError(f"candidate saved arrays disagree with recomputed output: {name}")
        results.append({"name": name, "passed": True, "compared_leaves": measured["compared_leaves"],
            "max_abs": measured["max_abs"], "max_bound_ratio": measured["max_bound_ratio"]})
    return {"sidecar": {"path": str(path), "sha256": sha256(raw), "bytes": len(raw), "schema": seal["schema"]},
        "checks": results, "all_passed": True}


def validate_telemetry(bundle):
    record, identity, report = bundle["record"], bundle["identity"], bundle["report"]
    proof = record["provenance"]
    resources = proof["server_resources"]
    if (resources.get("step_CPU_budget") != 48 or resources.get("allocated_CPU_request_count") != 48
            or resources.get("parent_allocated_CPU_request_count") != 48
            or resources.get("billing_CPU_request_count") != 48):
        raise ValueError("actual 48-CPU step and parent-billing resource fields required")
    slurm = resources.get("Slurm", {})
    tres = slurm.get("ORCHJAX_SLURM_ALLOC_TRES", "")
    cpu_entries = [part for part in tres.split(",") if part.split("=", 1)[0] == "cpu"]
    if cpu_entries != ["cpu=48"] or slurm.get("ORCHJAX_PARENT_CPU_COUNT") != "48":
        raise ValueError("exact parent AllocTRES cpu=48 proof required")
    topology = resources.get("topology", [])
    physical = {(item.get("package_id"), item.get("physical_core_id")) for item in topology}
    selected = resources.get("selected_logical_cpus", [])
    if len(physical) != 48 or len(selected) != 48 or len(set(selected)) != 48:
        raise ValueError("48 distinct physical-core selections required")
    runtime = identity.get("runtime", {})
    if (runtime.get("jax") != "0.4.38" or runtime.get("jaxlib") != "0.4.38"
            or runtime.get("x64_enabled") is not True
            or {device.get("platform") for device in runtime.get("device_details", [])} != {"cpu"}
            or len(runtime.get("device_details", [])) != 1):
        raise ValueError("actual single CPU JAX device with accepted x64 runtime required")
    env = resources.get("thread_environment_before_JAX", {})
    flags = env.get("XLA_FLAGS") or ""
    if (env.get("OMP_NUM_THREADS") != "48" or env.get("MKL_NUM_THREADS") != "1"
            or env.get("OPENBLAS_NUM_THREADS") != "1" or env.get("NUMEXPR_NUM_THREADS") != "1"
            or "--xla_cpu_multi_thread_eigen=true" not in flags
            or "intra_op_parallelism_threads=1" in flags
            or "--xla_force_host_platform_device_count" in flags):
        raise ValueError("recorded batch48 native-thread policy differs from declared policy")
    for metric_name in (CORE_METRIC, WRAPPER_METRIC):
        metric = record["timing"].get(metric_name, {})
        samples = metric.get("process_CPU_samples", [])
        if len(samples) < 5:
            raise ValueError(f"{metric_name}: five or more process CPU samples required")
        for sample in samples:
            numeric = ("wall_seconds", "process_CPU_user_seconds_delta", "process_CPU_system_seconds_delta",
                "process_CPU_user_system_seconds_delta", "busy_core_equivalents",
                "step_utilization_fraction", "parent_utilization_fraction")
            if any(not isinstance(sample.get(key), (int, float)) or not math.isfinite(sample[key]) for key in numeric):
                raise ValueError(f"{metric_name}: incomplete process CPU sample")
            if sample.get("step_CPU_budget") != 48 or sample.get("parent_CPU_budget") != 48:
                raise ValueError(f"{metric_name}: sample budgets differ from actual 48-core contract")
    observations = report.get("runtime_observations", {}).get("B64", {})
    provenance_observations = proof.get("runtime_observations", {})
    if observations != provenance_observations:
        raise ValueError("runtime phase observations must be identical in report and checkpoint provenance")
    for phase in ("pre_compile", "post_compile", "first_call_complete", "warm_complete"):
        item = observations.get(phase, {})
        if (item.get("available") is not True or item.get("RSS_bytes", 0) <= 0
                or item.get("peak_RSS_bytes", 0) <= 0 or item.get("MemAvailable_bytes", 0) <= 0):
            raise ValueError(f"missing actual host/resource observation at {phase}")
    first_cpu = observations.get("first_call_process_CPU", {})
    if (first_cpu.get("process_CPU_user_system_seconds_delta", 0) <= 0
            or first_cpu.get("step_CPU_budget") != 48 or first_cpu.get("parent_CPU_budget") != 48):
        raise ValueError("first-call process CPU accounting missing or outside the 48-CPU contract")
    for key in ("thread_affinity_after_compile", "thread_affinity_after_warm"):
        census = observations.get(key, {})
        if census.get("all_observed_thread_masks_within_declared_cores") is not True:
            raise ValueError(f"missing bounded thread-affinity census {key}")
    if record["timing"][CORE_METRIC].get("process_CPU_samples") != report["timing"][CORE_METRIC].get("process_CPU_samples"):
        raise ValueError("core process CPU samples differ between stage and report")
    if record["timing"][WRAPPER_METRIC].get("process_CPU_samples") != report["timing"][WRAPPER_METRIC].get("process_CPU_samples"):
        raise ValueError("wrapper process CPU samples differ between stage and report")
    return {"step_cpu_count": 48, "parent_billing_cpu_count": 48, "selected_unique_physical_cores": len(physical),
        "jax_devices": len(runtime["device_details"]), "runtime_observation_phases": [
            "pre_compile", "post_compile", "first_call_complete", "warm_complete"],
        "process_cpu_samples_per_metric": {
            CORE_METRIC: len(record["timing"][CORE_METRIC]["process_CPU_samples"]),
            WRAPPER_METRIC: len(record["timing"][WRAPPER_METRIC]["process_CPU_samples"])},
        "thread_environment": env, "scientific_acceptance": False}


def save_result(output_path: Path, result: dict) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def elapsed_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})", value or "")
    if match is None:
        raise ValueError("terminal sacct Elapsed must use [days-]HH:MM:SS")
    days, hours, minutes, seconds = match.groups()
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    if hours > 23 or minutes > 59 or seconds > 59:
        raise ValueError("terminal sacct Elapsed contains an out-of-range time field")
    return int(days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds


def validate_raw_accounting(accounting: dict, expected_job_id: str, run_root: Path, *, batch_only=False) -> int:
    raw = accounting.get("raw_record")
    if not isinstance(raw, str) or not raw.endswith("\n"):
        raise ValueError("terminal accounting must preserve the exact sacct parsable2 parent row")
    raw_digest = sha256(raw.encode("utf-8"))
    if accounting.get("raw_record_sha256") != raw_digest:
        raise ValueError("terminal accounting raw parent row SHA mismatch")
    fields = raw.rstrip("\n").split("|")
    if len(fields) != 4 or fields != [accounting.get("job_id"), accounting.get("terminal_status"),
            accounting.get("Elapsed"), accounting.get("AllocTRES")]:
        raise ValueError("terminal accounting fields differ from the preserved sacct parent row")
    if accounting.get("job_id") != expected_job_id:
        raise ValueError("terminal accounting job ID differs from the coordinator parent allocation")
    tres = accounting.get("AllocTRES", "")
    parts = [part.strip() for part in tres.split(",") if part.strip()]
    cpu = [part for part in parts if part.split("=", 1)[0] == "cpu"]
    gpus = [part for part in parts if part.split("=", 1)[0].startswith("gres/gpu")]
    if cpu != ["cpu=48"] or any(part != "gres/gpu=0" for part in gpus):
        raise ValueError("terminal accounting AllocTRES must contain exactly cpu=48 and no GPU allocation")
    seconds = elapsed_seconds(accounting["Elapsed"])
    if seconds <= 0 or accounting.get("wall_seconds") != seconds:
        raise ValueError("terminal accounting wall_seconds must equal the parent Elapsed field")
    expected_dirs = [str(run_root / "batch")] if batch_only else [str(run_root / "scalar"), str(run_root / "batch")]
    if accounting.get("directories") != expected_dirs:
        scope = "batch-only" if batch_only else "scalar and batch"
        raise ValueError(f"raw parent accounting must cover this exact {scope} RUN_ROOT")
    if (accounting.get("source") not in {"sacct", "raw9"}
            or accounting.get("scope") != "whole_slurm_parent_job"
            or accounting.get("allocated_CPU_count") != 48 or accounting.get("GPU_count") != 0
            or accounting.get("terminal_status") not in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"}):
        raise ValueError("raw parent accounting must identify a terminal sacct/raw9 whole-job 48-CPU record")
    return seconds


def raw_parent_cost(accounting: dict | None, accounting_path: Path | None, run_root: Path,
                    expected_job_id: str | None, *, batch_only=False) -> dict:
    if accounting is None:
        return {"status": "pending_external_terminal_accounting",
            "required_source": "terminal sacct or raw9 accounting covering the entire Slurm parent allocation",
            "whole_job_cost_cny": None, "billing_verified": False}
    if not expected_job_id:
        raise ValueError("coordinator parent job ID is required to bind terminal accounting")
    wall = validate_raw_accounting(accounting, expected_job_id, run_root, batch_only=batch_only)
    return {"status": "terminal_parent_accounting_loaded", "source": accounting["source"],
        "path": str(accounting_path), "sha256": sha256(accounting_path.read_bytes()),
        "raw_parent_record": accounting["raw_record"], "raw_parent_record_sha256": accounting["raw_record_sha256"],
        "job_id": accounting["job_id"], "AllocTRES": accounting["AllocTRES"],
        "Elapsed": accounting["Elapsed"], "terminal_status": accounting["terminal_status"], "wall_seconds": wall,
        "allocated_parent_CPU_request_core_hours": 48 * wall / 3600,
        "primary_cost_cny_estimate": 48 * wall / 3600 * 0.05,
        "conservative_cost_cny_estimate": 48 * wall / 3600 * 0.07,
        "billing_verified": False, "scope": "whole parent Slurm job; scalar step is charged inside the parent 48-CPU allocation"}


def reused_source_parent_cost(admission: dict, admission_path: Path, candidate_job_id: str | None) -> dict:
    source = admission.get("scalar_source", {})
    parent = source.get("whole_parent_accounting", {})
    raw = parent.get("raw_record")
    if not isinstance(raw, str) or not raw.endswith("\n"):
        raise ValueError("validated scalar admission must preserve the exact old parent accounting row")
    fields = raw.rstrip("\n").split("|")
    if len(fields) != 4 or fields != [parent.get("job_id"), parent.get("terminal_status"),
            parent.get("Elapsed"), parent.get("AllocTRES")]:
        raise ValueError("old scalar parent accounting summary differs from its preserved raw row")
    raw_digest = sha256(raw.encode("utf-8"))
    if parent.get("raw_record_sha256") != raw_digest:
        raise ValueError("old scalar parent accounting raw-row SHA mismatch")
    source_job_id = fields[0]
    if not candidate_job_id or candidate_job_id == source_job_id:
        raise ValueError("new candidate job ID must be present and distinct from the old scalar source job")
    wall = elapsed_seconds(fields[2])
    allocated = 48 * wall / 3600
    return {"status": "terminal_source_parent_accounting_loaded", "path": str(parent.get("path")),
        "admission_path": str(admission_path), "admission_sha256": admission.get("sha256"),
        "sha256": parent.get("sha256"), "job_id": source_job_id,
        "candidate_job_id": candidate_job_id, "terminal_status": fields[1], "Elapsed": fields[2],
        "wall_seconds": wall, "AllocTRES": fields[3], "raw_parent_record": raw,
        "raw_parent_record_sha256": raw_digest, "allocated_CPU_request_core_hours": allocated,
        "primary_cost_cny_estimate": allocated * 0.05,
        "conservative_cost_cny_estimate": allocated * 0.07,
        "billing_verified": False, "counted_once_as_source_parent": True,
        "new_allocation_reuse_charge": False,
        "scope": "completed prior 48-CPU source job; not included in new parent allocation estimate"}


def validated_reused_source_parent_cost(orchestration: dict) -> dict | None:
    if orchestration.get("execution_route") != "batch-only":
        return None
    commands = orchestration.get("commands", {}).get("batch", {}).get("argv", [])
    if "--cpu-reference-dir" not in commands:
        raise ValueError("batch-only orchestration does not name its external scalar directory")
    scalar_dir = Path(commands[commands.index("--cpu-reference-dir") + 1])
    admission_record = orchestration.get("external_scalar_reference_admission")
    candidate = orchestration.get("candidate_provenance")
    if (not isinstance(admission_record, dict) or not isinstance(candidate, dict)
            or not admission_record.get("path") or not admission_record.get("sha256")):
        raise ValueError("batch-only orchestration lacks external scalar admission/candidate provenance")
    from scripts.dev.global_server_reference import validate_cpu_reference_admission

    admission = validate_cpu_reference_admission(admission_record["path"], admission_record["sha256"],
        scalar_dir, expected_source=candidate)
    candidate_job_id = orchestration.get("allocation", {}).get("job_id") or candidate.get("job_id")
    return reused_source_parent_cost(admission, Path(admission_record["path"]), candidate_job_id)


def collect(run_root: Path, output_path: Path, parent_accounting_path: Path | None = None) -> dict:
    run_root = run_root.resolve()
    orchestration = load_json(run_root / "orchestration.json")
    batch_only = orchestration.get("execution_route") == "batch-only"
    accounting = load_json(parent_accounting_path) if parent_accounting_path is not None else None
    parent_job_id = orchestration.get("allocation", {}).get("job_id")
    parent_cost = raw_parent_cost(accounting, parent_accounting_path, run_root, parent_job_id,
        batch_only=batch_only)
    coordinator_cost = {"wall_seconds": orchestration.get("wall_seconds"),
        "primary_cost_cny_estimate": orchestration.get("coordinator_scope_primary_cost_cny_estimate"),
        "conservative_cost_cny_estimate": orchestration.get("coordinator_scope_conservative_cost_cny_estimate"),
        "scope": orchestration.get("coordinator_scope_cost_limitation"),
        "whole_parent_cost": False}
    reused_source_parent = validated_reused_source_parent_cost(orchestration)
    if orchestration.get("status") != "completed_exploratory_measurement":
        return save_result(output_path, {"schema": "global-cpu48-b64-collection-v1",
            "status": "incomplete_run_failure_preserved", "scientific_acceptance": False,
            "coordinator_status": orchestration.get("status"), "steps": orchestration.get("steps", {}),
            "coordinator_scope_cost": coordinator_cost, "whole_parent_allocation_cost": parent_cost,
            "reused_scalar_source_parent_cost": reused_source_parent,
            "orchestration_path": str(run_root / "orchestration.json"),
            "claim_scope": "failure accounting only; no throughput claim"})
    from scripts.dev import summarize_global_server_cost as cost
    from scripts.dev.global_server_reference import load_reference

    expected_reference_sha = orchestration.get("expected_gpu_reference_sha256")
    if expected_reference_sha != REFERENCE_SHA:
        raise ValueError("coordinator-bound GPU reference SHA differs from accepted FULL64 snapshot")
    if orchestration.get("expected_science_sha256") != SCIENCE_SHA:
        raise ValueError("coordinator-bound science SHA mismatch")
    if orchestration.get("allocation", {}).get("parent_cpu_count") != 48:
        raise ValueError("48 CPU parent allocation not sealed in orchestration")
    batch_argv = orchestration["commands"]["batch"]["argv"]
    if batch_only:
        if "--cpu-reference-dir" not in batch_argv:
            raise ValueError("batch-only orchestration does not name its external scalar directory")
        scalar_dir = Path(batch_argv[batch_argv.index("--cpu-reference-dir") + 1])
    else:
        scalar_dir = run_root / "scalar"
    scalar = sealed_stage(scalar_dir, SCALAR_STAGE)
    batch = sealed_stage(run_root / "batch", BATCH_STAGE)
    identity = batch["identity"]
    if (identity.get("science_sha256") != SCIENCE_SHA or identity.get("baseline_ref") != BASELINE_REF
            or identity.get("mode") != "batch" or identity.get("CPU_budget") != 48
            or identity.get("validation_mode") != "exploratory"
            or identity.get("cpu_reference_role") != "cpu-internal"
            or identity.get("numerical_reference_sha256") != expected_reference_sha
            or identity.get("lanes") != scalar["identity"].get("lanes")):
        raise ValueError("CPU batch/scalar full input and scientific identity mismatch")
    if identity.get("runtime", {}).get("git_head") != orchestration.get("expected_execution_commit"):
        raise ValueError("actual candidate immutable execution commit mismatch")
    reference_dir = Path(orchestration["commands"]["batch"]["argv"][
        orchestration["commands"]["batch"]["argv"].index("--gpu-reference-dir") + 1])
    gpu_outputs, gpu_origin = load_reference(reference_dir, identity)
    if gpu_origin.get("sha256") != expected_reference_sha:
        raise ValueError("actual loaded GPU numerical reference SHA mismatch")
    admission_args = {}
    if batch_only:
        admission = orchestration.get("external_scalar_reference_admission")
        candidate_provenance = identity.get("candidate_provenance")
        if not isinstance(admission, dict) or not candidate_provenance:
            raise ValueError("batch-only orchestration lacks its external scalar admission/candidate provenance")
        admission_args = {"cpu_reference_admission_record": admission.get("path"),
            "expected_cpu_reference_admission_sha256": admission.get("sha256"),
            "candidate_provenance": candidate_provenance}
    scalar_outputs, scalar_origin = load_reference(scalar_dir, identity,
        name=SCALAR_STAGE, points=64, cpu_reference_role="cpu-internal", **admission_args)
    if scalar_origin.get("role") != "cpu_internal_comparison" or scalar_origin.get("native_input_identity_verified") is not True:
        raise ValueError("CPU scalar reference was not loaded as an exact internal numerical comparison")
    outputs = batch["value"].get("outputs")
    if outputs is None or len(outputs) != 2:
        raise ValueError("candidate checkpoint must contain both complete seven-day output blocks")
    schema = require_tree_schema(outputs, gpu_outputs)
    gpu_gates = check_gpu_gate_set(batch, outputs, gpu_outputs)
    candidate_validation_arrays = check_candidate_saved_arrays(batch, outputs) if batch_only else None
    recorded_cross = next((gate for gate in batch["record"]["gates"]
        if gate.get("name") == "B64_batch_cross_mode_server_scalar"), None)
    if recorded_cross is None:
        raise ValueError("candidate internal scalar cross-mode gate missing")
    from scripts.dev import benchmark_global_batch as base
    cross = base.compare(outputs, scalar_outputs,
        sentinel_paths={item["path"]: item["contract"] for item in recorded_cross.get("allowed_sentinels", [])})
    if not gate_matches(recorded_cross, cross) or not cross["passed"]:
        raise ValueError("candidate saved arrays disagree with the complete CPU scalar reference")
    cost_stage = cost.exploratory_stage(run_root / "batch", BATCH_STAGE, batch["manifest"], batch["report"])
    telemetry = validate_telemetry(batch)
    point_ids = [lane["landpoint_id"] for lane in identity["lanes"]]
    if len(point_ids) != 64 or len(set(point_ids)) != 64 or cost_stage["core"]["point_days"] != POINT_DAYS:
        raise ValueError("candidate denominator must be 64 unique points x 14 days")
    warm = cost_stage["wrapper"]
    source_parent = (reused_source_parent_cost(scalar_origin["admission"],
        Path(scalar_origin["admission"]["path"]),
        orchestration.get("allocation", {}).get("job_id") or scalar_origin["admission"].get("candidate_identity", {}).get("job_id"))
        if scalar_origin.get("admission") is not None else None)
    jobs = []
    if accounting is not None:
        jobs = [{"directories": accounting["directories"], "status": accounting["terminal_status"],
            "wall_seconds": accounting["wall_seconds"], "allocated_CPU_count": 48, "GPU_count": 0,
            "provenance": f"{accounting['source']}:{parent_accounting_path}"}]
    summary = cost.summarize([run_root / "batch"], jobs)
    result = {"schema": "global-cpu48-b64-collection-v1", "status": "complete_exploratory_measurement",
        "scientific_acceptance": False, "performance_claim_scope": "one 48-CPU parent allocation; one CPU process; native B64 vmap",
        "execution_commit": orchestration["expected_execution_commit"], "science_sha256": SCIENCE_SHA,
        "baseline_ref": BASELINE_REF, "gpu_reference_sha256": expected_reference_sha,
        "canonical_prepared_input": str(orchestration["commands"]["batch"]["argv"][
            orchestration["commands"]["batch"]["argv"].index("--prepared-input") + 1]),
        "workload": {"unique_points": 64, "days_per_point": 14, "point_days": POINT_DAYS,
            "full_outputs": True, "blocks": [[2, 8], [9, 15]]},
        "cpu_batch": {"stage": BATCH_STAGE, "evidence_class": "exploratory_performance",
            "strict_numerical_passed": batch["record"].get("strict_numerical_passed"),
            "core": cost_stage["core"], "wrapper": warm, "runtime_observations": telemetry,
            "raw_output_schema": schema, "saved_output_checkpoint": {
                "path": str(batch["checkpoint"]), "sha256": batch["checkpoint_sha256"]},
            "manifest_sha256": batch["manifest_sha256"], "report_sha256": batch["report_sha256"]},
        "raw_array_gate_recomputation": {"GPU_reference": gpu_gates, "CPU_internal_scalar": {
            "role": "one complete two-block tree comparison; per-point values retained as diagnostic only",
            "passed": cross["passed"], "compared_leaves": cross["compared_leaves"]},
            "cross_mode_full_output_passed": cross["passed"], "cross_mode_compared_leaves": cross["compared_leaves"],
            "GPU_gate_count": len(gpu_gates), "CPU_internal_gate_count": 1,
            "GPU_tolerance_failures": sum(not row["passed"] for row in gpu_gates)},
        "cpu_internal_reference": {"stage": SCALAR_STAGE, "role": scalar_origin["role"],
            "sha256": scalar["record"]["sha256"], "manifest_sha256": scalar["manifest_sha256"],
            "report_sha256": scalar["report_sha256"],
            "directory": str(scalar_dir), "admission": scalar_origin.get("admission")},
        "candidate_validation_arrays": candidate_validation_arrays,
        "coordinator_scope_cost": coordinator_cost,
        "whole_parent_allocation_cost": parent_cost,
        "reused_scalar_source_parent_cost": source_parent,
        "parent_accounting_limitation": "coordinator interval excludes Slurm submission/startup; whole-job cost stays pending until terminal sacct/raw9 evidence is supplied",
        "cost_scenarios": summary, "excluded_claims": [
            "accepted CPU scientific result", "GPU-equivalent scientific acceptance",
            "measured multi-process CPU throughput", "10x improvement", "annual or Fortran speedup"]}
    return save_result(output_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-accounting", type=Path)
    args = parser.parse_args()
    result = collect(args.run_root, args.output, args.parent_accounting)
    print(json.dumps({"output": str(args.output), "status": result["status"],
        "scientific_acceptance": False, "point_days": POINT_DAYS}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
