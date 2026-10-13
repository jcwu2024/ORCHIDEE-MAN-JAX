"""Read complete full-state numerical snapshots across explicitly recorded backends.

CPU/GPU numerical reference use never imports an executable or old timing.
The existing prepared classes and fixed comparison gates remain authoritative.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path


def finite_tolerance_failure(gate):
    """Only the original comparator's finite, tolerance-only failure schema."""
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    failures = gate.get("failures")
    return (gate.get("passed") is False and isinstance(failures, list) and bool(failures)
        and isinstance(gate.get("compared_leaves"), int) and gate["compared_leaves"] > 0
        and finite(gate.get("max_abs")) and gate["max_abs"] >= 0
        and finite(gate.get("max_bound_ratio")) and gate["max_bound_ratio"] > 1
        and all(isinstance(failure, dict) and failure.get("reason") == "tolerance"
                and isinstance(failure.get("path"), str) and bool(failure["path"])
                and finite(failure.get("max_bound_ratio")) and failure["max_bound_ratio"] > 1
                for failure in failures))


def _evidence_mismatch(checkpoint, manifest, path):
    """Keep record evidence exact, allowing only loadavg's JSON tuple/list form."""
    if path == "record.execution_origin.environment.loadavg":
        if type(checkpoint) in (tuple, list) and type(manifest) in (tuple, list):
            # os.getloadavg() returns a tuple; atomic_json writes that same
            # non-scientific telemetry as a list. Preserve all three values.
            if len(checkpoint) != 3 or len(manifest) != 3:
                return path
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in (*checkpoint, *manifest)):
                return path
            checkpoint = list(checkpoint)
            manifest = list(manifest)
    if type(checkpoint) is not type(manifest):
        return path
    if isinstance(checkpoint, dict):
        for key in sorted(set(checkpoint) | set(manifest)):
            child = f"{path}.{key}"
            if key not in checkpoint or key not in manifest:
                return child
            mismatch = _evidence_mismatch(checkpoint[key], manifest[key], child)
            if mismatch is not None:
                return mismatch
    elif isinstance(checkpoint, (tuple, list)):
        if len(checkpoint) != len(manifest):
            return path
        for index, (left, right) in enumerate(zip(checkpoint, manifest, strict=True)):
            mismatch = _evidence_mismatch(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
    elif checkpoint != manifest:
        return path
    return None


def validate_cpu_reference_admission(admission_path, expected_sha256, old_directory, *, expected_source):
    """Verify a sealed scalar-source admission without importing pickle or JAX.

    The external admission binds the old scalar checkpoint to its manifest and
    report, then explicitly matches its scientific/input contract to the new
    candidate. Job IDs and execution commits are retained as separate
    provenance and are therefore allowed to differ.
    """
    admission_path, old_directory = Path(admission_path).resolve(), Path(old_directory).resolve()
    admission_bytes = admission_path.read_bytes()
    digest = hashlib.sha256(admission_bytes).hexdigest()
    if digest != expected_sha256:
        raise ValueError("CPU scalar admission record checksum mismatch")
    admission = json.loads(admission_bytes)
    if (admission.get("schema") != "global-cpu48-scalar-admission-v1"
            or admission.get("status") != "admitted_for_internal_comparison_only"
            or admission.get("scientific_acceptance") is not False):
        raise ValueError("CPU scalar admission record is not an internal-comparison-only admission")
    candidate = admission.get("candidate_identity")
    source = admission.get("scalar_source", {})
    source_identity = source.get("source_identity")
    audit = admission.get("audit", {})
    if not isinstance(candidate, dict) or not isinstance(expected_source, dict):
        raise ValueError("CPU scalar admission candidate identity is missing")
    candidate_job = expected_source.get("job_id")
    for key in ("execution_commit", "execution_code_dir", "prepared_input_path", "prepared_input_sha256",
                "gpu_reference_path", "gpu_reference_sha256", "certified_science_sha256",
                "actual_science_source_sha256", "physical_science_tree_oid", "baseline_ref"):
        if candidate.get(key) != expected_source.get(key):
            raise ValueError(f"CPU scalar admission candidate identity mismatch at {key}")
    if candidate.get("job_id") is not None:
        if candidate_job is not None and candidate.get("job_id") != candidate_job:
            raise ValueError("CPU scalar admission candidate job identity mismatch")
    if not isinstance(source_identity, dict):
        raise ValueError("CPU scalar admission lacks the original scalar source identity")
    if (source.get("job_id") != "15820467"
            or source.get("execution_commit") != "03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8"
            or source.get("stage_sha256") != "1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa"
            or source.get("prepared_input_sha256") != "be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7"):
        raise ValueError("CPU scalar admission is not bound to the exact sealed old source job/checkpoint/input")
    for key, source_key in (("baseline_ref", "baseline_ref"),
            ("science_sha256", "certified_science_sha256"),
            ("numerical_reference_sha256", "gpu_reference_sha256")):
        if source_identity.get(key) != candidate.get(source_key):
            raise ValueError(f"CPU scalar admission source/candidate {source_key} mismatch")
    for key in ("prepared_input_sha256", "gpu_reference_sha256", "certified_science_sha256",
                "physical_science_tree_oid", "baseline_ref", "actual_science_source_sha256"):
        if source.get(key) != candidate.get(key):
            raise ValueError(f"CPU scalar admission source/candidate {key} mismatch")
    if not source.get("job_id") or not source.get("execution_commit") or not source.get("execution_code_dir"):
        raise ValueError("CPU scalar admission must preserve the source job and execution identity")
    if ((candidate.get("job_id") is not None and not candidate.get("job_id"))
            or not expected_source.get("execution_commit") or not expected_source.get("execution_code_dir")):
        raise ValueError("CPU scalar admission must preserve the candidate job and execution identity")
    if candidate_job is not None and source.get("job_id") == candidate_job:
        raise ValueError("CPU scalar source and candidate must retain distinct job identities")
    if audit.get("strict_numerical_passed") is not False:
        raise ValueError("CPU scalar admission must preserve the failed strict numerical status")
    if audit.get("frozen_reset_arrays_recomputed") is not False:
        raise ValueError("CPU scalar admission must disclose that frozen-reset arrays were not recomputed")
    if (audit.get("GPU_gate_count") != 64 or audit.get("GPU_passed_gate_count") != 61
            or audit.get("GPU_tolerance_failed_gate_count") != 3
            or audit.get("GPU_tolerance_failure_count") != 4
            or len(audit.get("recomputed_GPU_gates", [])) != 64):
        raise ValueError("CPU scalar admission does not preserve the recomputed 64-point gate summary")
    parent = source.get("whole_parent_accounting", {})
    raw_parent = parent.get("raw_record", "")
    if (parent.get("sha256") != "a04c687530f390ad1d5f1d3668293a552b3a9b37f0cd05ef1f264660c2f37113"
            or parent.get("job_id") != "15820467" or parent.get("wall_seconds") != 2197
            or parent.get("allocated_CPU_count") != 48 or parent.get("GPU_count") != 0
            or parent.get("AllocTRES") != "billing=48,cpu=48,node=1"
            or parent.get("terminal_status") != "FAILED" or parent.get("Elapsed") != "00:36:37"
            or parent.get("counted_once_as_source_parent") is not True
            or hashlib.sha256(raw_parent.encode("utf-8")).hexdigest() != parent.get("raw_record_sha256")):
        raise ValueError("CPU scalar admission lacks verified old whole-parent accounting counted once")

    manifest_path, report_path = old_directory / "stage_manifest.json", old_directory / "report.json"
    manifest_bytes, report_bytes = manifest_path.read_bytes(), report_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    if source.get("manifest_sha256") != manifest_sha or source.get("report_sha256") != report_sha:
        raise ValueError("CPU scalar admission manifest/report checksum mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("identity") != source_identity:
        raise ValueError("CPU scalar admission source identity differs from the sealed scalar manifest")
    stage_name = source.get("stage_name")
    if stage_name != "B64_server_scalar_v1":
        raise ValueError("CPU scalar admission must name the complete B64 scalar stage")
    record = manifest.get("stages", {}).get(stage_name, {})
    if record.get("status") != "complete" or record.get("measurement_status") != "complete":
        raise ValueError("CPU scalar admission source stage is not complete")
    stage_path = (old_directory / record.get("path", "")).resolve()
    if not stage_path.is_relative_to(old_directory) or not stage_path.is_file():
        raise ValueError("CPU scalar admission stage path escapes its sealed directory")
    stage_bytes = stage_path.read_bytes()
    stage_sha = hashlib.sha256(stage_bytes).hexdigest()
    if (stage_sha != record.get("sha256") or source.get("stage_sha256") != stage_sha
            or source.get("stage_path") != record.get("path")):
        raise ValueError("CPU scalar admission stage checksum/path mismatch")
    return {"path": str(admission_path), "sha256": digest, "source_job_id": source.get("job_id"),
            "source_execution_commit": source.get("execution_commit"), "source_identity": source_identity,
            "scalar_source": source, "candidate_identity": candidate,
            "stage": {"name": stage_name, "path": str(stage_path),
                "sha256": stage_sha}, "manifest_sha256": manifest_sha, "report_sha256": report_sha,
            "audit": audit}


def bind_cpu_reference_admission(receipt_path, output_path, candidate_identity, *, expected_receipt_sha256):
    """Bind an independently audited old scalar receipt to a final immutable candidate.

    This stdlib-only step does not rerun arrays or model code. The candidate
    commit and code directory are supplied after the execution tree is sealed;
    the resulting admission file remains an external runtime input.
    """
    receipt_path, output_path = Path(receipt_path).resolve(), Path(output_path).resolve()
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha != expected_receipt_sha256:
        raise ValueError("CPU scalar audit receipt checksum mismatch")
    receipt = json.loads(receipt_bytes)
    if (receipt.get("schema") != "global-cpu48-scalar-audit-v1"
            or receipt.get("status") != "saved_arrays_audited"
            or receipt.get("scientific_acceptance") is not False):
        raise ValueError("CPU scalar audit receipt is not an internal-comparison-only audit")
    audit, source = receipt.get("audit", {}), receipt.get("scalar_source", {})
    if (audit.get("strict_numerical_passed") is not False
            or audit.get("frozen_reset_arrays_recomputed") is not False
            or audit.get("GPU_gate_count") != 64 or audit.get("GPU_passed_gate_count") != 61
            or audit.get("GPU_tolerance_failed_gate_count") != 3
            or audit.get("GPU_tolerance_failure_count") != 4
            or len(audit.get("recomputed_GPU_gates", [])) != 64):
        raise ValueError("CPU scalar audit receipt does not preserve the required gate/reset limitation")
    candidate = dict(candidate_identity)
    if candidate.get("job_id") is not None:
        raise ValueError("final pre-allocation CPU scalar admission must leave candidate job_id null")
    for key in ("execution_commit", "physical_science_tree_oid"):
        value = candidate.get(key)
        if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"candidate {key} must be a full lowercase Git object ID")
    for key in ("execution_code_dir", "prepared_input_path", "prepared_input_sha256", "gpu_reference_path",
                "gpu_reference_sha256", "certified_science_sha256", "actual_science_source_sha256", "baseline_ref"):
        if not candidate.get(key):
            raise ValueError(f"candidate identity is missing {key}")
    for key in ("prepared_input_sha256", "gpu_reference_sha256", "certified_science_sha256",
                "actual_science_source_sha256", "physical_science_tree_oid", "baseline_ref"):
        if source.get(key) != candidate.get(key):
            raise ValueError(f"CPU scalar source/candidate {key} mismatch")
    source_identity = source.get("source_identity", {})
    if (source_identity.get("science_sha256") != candidate.get("certified_science_sha256")
            or source_identity.get("numerical_reference_sha256") != candidate.get("gpu_reference_sha256")
            or source_identity.get("baseline_ref") != candidate.get("baseline_ref")):
        raise ValueError("CPU scalar source identity differs from the bound candidate science/reference")
    parent = source.get("whole_parent_accounting", {})
    if (parent.get("sha256") != "a04c687530f390ad1d5f1d3668293a552b3a9b37f0cd05ef1f264660c2f37113"
            or parent.get("job_id") != "15820467" or parent.get("wall_seconds") != 2197
            or parent.get("allocated_CPU_count") != 48 or parent.get("GPU_count") != 0
            or parent.get("counted_once_as_source_parent") is not True
            or hashlib.sha256(parent.get("raw_record", "").encode("utf-8")).hexdigest()
                != parent.get("raw_record_sha256")):
        raise ValueError("CPU scalar audit receipt lacks verified old parent accounting")
    admission = {"schema": "global-cpu48-scalar-admission-v1",
        "status": "admitted_for_internal_comparison_only", "scientific_acceptance": False,
        "candidate_identity": candidate, "scalar_source": source, "audit": audit,
        "audit_receipt": {"path": str(receipt_path), "sha256": receipt_sha,
            "schema": receipt["schema"], "status": receipt["status"]}}
    payload = (json.dumps(admission, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(payload)
    return {"path": str(output_path), "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "candidate_identity": candidate, "audit_receipt_sha256": receipt_sha}


def load_reference(directory, identity, *, name="B4_group_references", points=None, cpu_reference_role="accepted",
        cpu_reference_admission_record=None, expected_cpu_reference_admission_sha256=None,
        candidate_provenance=None):
    if cpu_reference_role not in {"accepted", "cpu-internal"}:
        raise ValueError("unknown CPU reference role")
    internal = cpu_reference_role == "cpu-internal"
    if internal and (name == "B4_group_references" or identity.get("validation_mode", "strict") != "exploratory"):
        raise ValueError("cpu-internal role requires an exploratory CPU cross-mode consumer")
    if internal and {device.get("platform") for device in identity.get("runtime", {}).get("device_details", [])} != {"cpu"}:
        raise ValueError("cpu-internal reference requires an actual CPU consumer")
    directory = Path(directory).resolve()
    raw_manifest = (directory / "stage_manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    origin = manifest["identity"]
    for key in ("baseline_ref", "science_sha256", "gate"):
        if origin.get(key) != identity.get(key):
            raise ValueError(f"portable numerical reference {key} mismatch")
    selected = identity["lanes"] if points is None else identity["lanes"][:points]
    if origin["lanes"][:len(selected)] != selected or len(origin["lanes"]) < len(selected):
        raise ValueError("portable numerical reference exact point order/input dtype-byte identity mismatch")
    if origin.get("precision_policy") != identity.get("precision_policy"):
        raise ValueError("portable reference native precision policy mismatch")
    if origin["runtime"].get("x64_enabled") is not True or identity["runtime"].get("x64_enabled") is not True:
        raise ValueError("portable numerical reference requires JAX x64 on both backends")
    for runtime in (origin["runtime"], identity["runtime"]):
        if runtime.get("jax") != "0.4.38" or runtime.get("jaxlib") != "0.4.38":
            raise ValueError("server comparisons require the accepted JAX/jaxlib 0.4.38 runtime")
    expected_backend = "gpu" if name == "B4_group_references" else "cpu"
    if {device.get("platform") for device in origin["runtime"].get("device_details", [])} != {expected_backend}:
        raise ValueError("numerical reference source backend does not match its declared role")
    if name != "B4_group_references" and origin.get("mode") != "scalar":
        raise ValueError("CPU cross-mode reference must originate from server scalar execution")
    admission = None
    external_v4_reuse = (cpu_reference_admission_record is not None
        or expected_cpu_reference_admission_sha256 is not None or candidate_provenance is not None
        or identity.get("candidate_provenance") is not None)
    if internal and name == "B64_server_scalar_v1" and external_v4_reuse:
        if not cpu_reference_admission_record or not expected_cpu_reference_admission_sha256 or not candidate_provenance:
            raise ValueError("FULL64 CPU scalar reuse requires an external saved-scalar admission record")
        admission = validate_cpu_reference_admission(cpu_reference_admission_record,
            expected_cpu_reference_admission_sha256, directory, expected_source=candidate_provenance)
        if (candidate_provenance.get("baseline_ref") != identity.get("baseline_ref")
                or candidate_provenance.get("certified_science_sha256") != identity.get("science_sha256")
                or candidate_provenance.get("gpu_reference_sha256") != identity.get("numerical_reference_sha256")
                or candidate_provenance.get("execution_commit") != identity.get("runtime", {}).get("git_head")):
            raise ValueError("FULL64 CPU scalar admission candidate provenance differs from current execution identity")
        if (not candidate_provenance.get("job_id")
                or candidate_provenance.get("job_id") == admission.get("source_job_id")):
            raise ValueError("FULL64 CPU scalar reuse requires distinct recorded source and candidate job IDs")
    record = manifest["stages"].get(name, {})
    if record.get("status") != "complete" or not record.get("sha256"):
        raise ValueError("reference must be a complete SHA-verified stage")
    path = directory / record.get("path", f"stage_{name}.pkl")
    if not path.resolve().is_relative_to(directory):
        raise ValueError("reference path outside declared directory")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != record["sha256"]:
        raise ValueError("reference stage checksum mismatch")
    envelope = pickle.loads(data)
    if envelope["identity"] != origin or envelope["name"] != name or envelope["status"] != "complete":
        raise ValueError("reference envelope identity/status mismatch")
    proof = record.get("provenance", {})
    if name != "B4_group_references" and "parent_allocated_CPU_request_count" in origin:
        _validate_scalar_parent_allocation(origin, proof)
    ids = [item["landpoint_id"] for item in selected]
    if proof.get("point_ids", [])[:len(ids)] != ids or not record.get("gates"):
        raise ValueError("reference point provenance/numerical gates mismatch")
    if not internal and (any(gate.get("passed") is not True for gate in record["gates"])
            or origin.get("validation_mode", "strict") != "strict"
            or record.get("evidence_class", "accepted") != "accepted"):
        raise ValueError("accepted reference point provenance/numerical gates mismatch")
    if name == "B4_group_references" and proof.get("own_inputs") is not True:
        raise ValueError("FULL64 B4 references must use every point's own inputs")
    value = envelope["value"]
    output = value["outputs"] if isinstance(value, dict) and "outputs" in value else value
    if internal:
        for key in ("gates", "provenance", "timing", "measurement_status", "evidence_class", "strict_numerical_passed", "execution_origin"):
            mismatch = _evidence_mismatch(envelope.get("record", {}).get(key), record.get(key), f"record.{key}")
            if mismatch is not None:
                raise ValueError(f"CPU internal reference checkpoint/manifest evidence changed at {mismatch}")
        _validate_cpu_internal(name, origin, identity, record, value, output, ids)
    return output, {"path": str(path), "sha256": digest, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "source_identity": origin, "execution_origin": record.get("execution_origin"),
                    "role": "cpu_internal_comparison" if internal else "complete numerical evidence only; no prior executable or timing reuse",
                    "backend_conditions": {"source": origin["runtime"], "current": identity["runtime"]},
                    "point_ids": ids, "native_input_identity_verified": True,
                    "fixed_gate": identity["gate"],
                    **({"admission": admission, "candidate_provenance": candidate_provenance}
                       if admission is not None else {})}


def _validate_scalar_parent_allocation(origin, proof):
    """New scalar steps disclose the whole parent charge; legacy references remain readable."""
    from scripts.dev.global_server_resources import allocated_tres_cpu_count

    parent = origin["parent_allocated_CPU_request_count"]
    resources = proof.get("server_resources", {})
    slurm = resources.get("Slurm", {})
    if (type(parent) is not int or parent < 1 or origin.get("CPU_budget") != 1
            or origin.get("billing_CPU_request_count") != parent
            or resources.get("step_CPU_budget") != 1
            or resources.get("allocated_CPU_request_count") != 1
            or resources.get("parent_allocated_CPU_request_count") != parent
            or resources.get("billing_CPU_request_count") != parent
            or slurm.get("ORCHJAX_PARENT_CPU_COUNT") not in (None, str(parent))
            or slurm.get("SLURM_CPUS_PER_TASK") != "1"):
        raise ValueError("CPU scalar reference step/parent billing identity mismatch")
    tres = slurm.get("ORCHJAX_SLURM_ALLOC_TRES", "")
    if not tres and parent == 1 and slurm.get("ORCHJAX_PARENT_CPU_COUNT") is None:
        return  # Ordinary legacy one-core launch without the new enclosing-allocation interface.
    try:
        allocated = allocated_tres_cpu_count(tres)
    except RuntimeError as exc:
        raise ValueError("CPU scalar reference parent allocated TRES mismatch") from exc
    if allocated != parent:
        raise ValueError("CPU scalar reference parent allocated TRES mismatch")


def _validate_cpu_internal(name, origin, identity, record, value, output, ids):
    """A measured scalar CPU snapshot is an internal comparator, not acceptance."""
    size = len(ids)
    proof = record["provenance"]
    gates = record["gates"]
    if (name != f"B{size}_server_scalar_v1" or origin.get("validation_mode", "strict") not in {"strict", "exploratory"}
            or record.get("measurement_status") != "complete"
            or record.get("evidence_class") not in {"accepted", "exploratory_performance"}
            or proof.get("full_two_block_workload") is not True or proof.get("full_outputs") is not True
            or proof.get("blocks") != [[2, 8], [9, 15]] or proof.get("half_hours_per_day") != 48
            or proof.get("unique_real_points") != size or proof.get("point_ids") != ids
            or proof.get("frozen_reset_repeat") is not True or proof.get("five_or_more_synchronized_repeats") is not True
            or origin.get("numerical_reference_sha256") != identity.get("numerical_reference_sha256")):
        raise ValueError("CPU internal reference lacks complete two-block scalar measurement provenance")
    execution = record.get("execution_origin", {})
    if (execution.get("available") is not True or
            {device.get("platform") for device in execution.get("environment", {}).get("device_details", [])} != {"cpu"}):
        raise ValueError("CPU internal reference requires recorded actual CPU scalar execution")
    expected = {f"B{size}_scalar_lane_{i}_{point}_GPU_reference" for i, point in enumerate(ids)}
    gpu_gates = {gate["name"]: gate for gate in gates if gate.get("name") in expected}
    repeats = [gate for gate in gates if gate.get("name") == f"B{size}_scalar_frozen_reset_repeat"]
    if (set(gpu_gates) != expected or len(repeats) != 1 or repeats[0].get("passed") is not True
            or record.get("strict_numerical_passed") != all(gate.get("passed") is True for gate in gates)
            or proof.get("full_two_block_parity") != all(gate.get("passed") is True for gate in gpu_gates.values())):
        raise ValueError("CPU internal reference parity/repeat gate metadata incomplete or inconsistent")
    for gate in gates:
        if gate.get("atol") != 1e-8 or gate.get("rtol") != 1e-10:
            raise ValueError("CPU internal reference changed the fixed numerical tolerance")
        if gate.get("passed") is not True and (gate.get("name") not in expected or not finite_tolerance_failure(gate)):
            raise ValueError("CPU internal reference contains a hard numerical failure")
    if not isinstance(value, dict) or "outputs" not in value or not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError("CPU internal reference requires actual complete two-block outputs")
    if any(not isinstance(block, (tuple, list)) or len(block) != 2 for block in output):
        raise ValueError("CPU internal reference missing a full block boundary")
    for key in (f"B{size}_warm_frozen_state_core", f"B{size}_preprepared_wrapper_stack_transfer_materialize"):
        metric = record.get("timing", {}).get(key, {})
        samples = metric.get("samples_seconds", [])
        if (len(samples) < 5 or any(not isinstance(sample, (int, float)) or not math.isfinite(sample) or sample <= 0 for sample in samples)
                or metric.get("valid_point_days") != size * 14
                or value.get("measurements", {}).get(key) != metric):
            raise ValueError("CPU internal reference requires five synchronized full-workload samples")
    from scripts.dev import benchmark_global_batch as base
    leaves = base.jax.tree_util.tree_leaves(output)
    if not leaves or any(base.np.asarray(leaf).ndim == 0 or base.np.asarray(leaf).shape[0] != size for leaf in leaves):
        raise ValueError("CPU internal reference output lane count/shape mismatch")
    sentinel_paths = {item["path"]: item["contract"] for item in repeats[0].get("allowed_sentinels", [])}
    check = base.compare(output, output, sentinel_paths=sentinel_paths)
    if not check["passed"] or repeats[0].get("compared_leaves") != check["compared_leaves"]:
        raise ValueError("CPU internal reference actual output schema/nonfinite evidence mismatch")
