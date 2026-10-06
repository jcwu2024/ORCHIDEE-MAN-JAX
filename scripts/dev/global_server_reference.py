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


def load_reference(directory, identity, *, name="B4_group_references", points=None, cpu_reference_role="accepted"):
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
            if envelope.get("record", {}).get(key) != record.get(key):
                raise ValueError("CPU internal reference checkpoint/manifest evidence changed")
        _validate_cpu_internal(name, origin, identity, record, value, output, ids)
    return output, {"path": str(path), "sha256": digest, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "source_identity": origin, "execution_origin": record.get("execution_origin"),
                    "role": "cpu_internal_comparison" if internal else "complete numerical evidence only; no prior executable or timing reuse",
                    "backend_conditions": {"source": origin["runtime"], "current": identity["runtime"]},
                    "point_ids": ids, "native_input_identity_verified": True,
                    "fixed_gate": identity["gate"]}


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
