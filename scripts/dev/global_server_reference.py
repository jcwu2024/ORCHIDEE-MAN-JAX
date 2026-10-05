"""Read complete full-state numerical snapshots across explicitly recorded backends.

CPU/GPU numerical reference use never imports an executable or old timing.
The existing prepared classes and fixed comparison gates remain authoritative.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path


def load_reference(directory, identity, *, name="B4_group_references", points=None):
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
    if (proof.get("point_ids", [])[:len(ids)] != ids or not record.get("gates")
            or any(gate.get("passed") is not True for gate in record["gates"])):
        raise ValueError("reference point provenance/numerical gates mismatch")
    if name == "B4_group_references" and proof.get("own_inputs") is not True:
        raise ValueError("FULL64 B4 references must use every point's own inputs")
    value = envelope["value"]
    output = value["outputs"] if isinstance(value, dict) and "outputs" in value else value
    return output, {"path": str(path), "sha256": digest, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "source_identity": origin, "execution_origin": record.get("execution_origin"),
                    "role": "complete numerical evidence only; no prior executable or timing reuse",
                    "backend_conditions": {"source": origin["runtime"], "current": identity["runtime"]},
                    "point_ids": ids, "native_input_identity_verified": True,
                    "fixed_gate": identity["gate"]}
