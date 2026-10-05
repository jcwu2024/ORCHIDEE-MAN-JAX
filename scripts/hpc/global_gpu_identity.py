"""Read-only physical GPU identity, independent of cgroup-local NVML index.

Numeric Slurm global device IDs must match the exact NVIDIA device minor in
both /proc driver records and accessible NVML XML. Unknown mappings fail closed.
CUDA_VISIBLE_DEVICES is recorded and preserved, never used as a physical index.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def pci_address(value):
    match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value.strip())
    if match is None:
        raise ValueError(f"unknown PCI address {value!r}")
    return tuple(int(part, 16) for part in match.groups())


def proc_inventory(root=Path("/proc/driver/nvidia/gpus")):
    records = []
    for path in sorted(root.glob("*/information")):
        fields = {key.strip(): value.strip() for line in path.read_text().splitlines()
                  if ":" in line for key, value in [line.split(":", 1)]}
        records.append({"uuid": fields["GPU UUID"], "minor": int(fields["Device Minor"]),
                        "pci": fields["Bus Location"], "path": str(path)})
    return records


def resolve(global_id, visible, xml, proc):
    if not global_id or "," in global_id or global_id.strip() != global_id:
        raise ValueError("exactly one unmodified Slurm global GPU ID required")
    if not visible or visible == "-1" or "," in visible:
        raise ValueError("exactly one preserved CUDA_VISIBLE_DEVICES identity required")
    tree = ET.fromstring(xml)
    driver = tree.findtext("driver_version")
    accessible = [{"uuid": gpu.findtext("uuid"), "minor": int(gpu.findtext("minor_number", "-1")),
                   "pci": gpu.findtext("pci/pci_bus_id"), "name": gpu.findtext("product_name")}
                  for gpu in tree.findall("gpu")]
    for records in (proc, accessible):
        if len({item["uuid"] for item in records}) != len(records) or len({item["minor"] for item in records}) != len(records):
            raise ValueError("ambiguous duplicate driver/NVML GPU identities")
    if global_id.isdecimal():
        matched = [record for record in proc if record["minor"] == int(global_id)]
        source = "SLURM_JOB_GPUS numeric ID -> exact /proc Device Minor -> NVML UUID/minor/PCI cross-check"
    elif global_id.startswith("GPU-"):
        matched = [record for record in proc if record["uuid"] == global_id]
        source = "SLURM_JOB_GPUS exact UUID -> /proc and NVML cross-check"
    else:
        raise ValueError(f"unsupported Slurm global GPU identity {global_id!r}")
    if len(matched) != 1:
        raise ValueError(f"unknown/ambiguous Slurm device mapping for {global_id!r}; driver records={proc}")
    physical = matched[0]
    selected = [record for record in accessible if record["uuid"] == physical["uuid"]]
    if len(selected) != 1:
        raise ValueError(f"Slurm-selected driver UUID {physical['uuid']} not accessible in NVML; visible metadata={accessible}")
    selected = selected[0]
    if selected["minor"] != physical["minor"] or pci_address(selected["pci"] or "") != pci_address(physical["pci"]):
        raise ValueError("NVML UUID matches but exact driver device minor/PCI identity differs")
    if not driver or not selected["name"] or not selected["uuid"].startswith("GPU-"):
        raise ValueError("required UUID/model/driver metadata missing")
    return {"status": "verified_metadata_only", "slurm_global_gpu_id": global_id,
            "CUDA_VISIBLE_DEVICES_preserved": visible, "uuid": selected["uuid"],
            "name": selected["name"], "driver": driver, "device_minor": selected["minor"],
            "pci_bus_id": selected["pci"], "origin": source, "proc_record": physical,
            "accessible_NVML_devices": accessible, "model_execution": False,
            "CUDA_selector_UUID_resolution_verified": False,
            "CUDA_execution_condition": "single allocated Slurm GPU cgroup and preserved CUDA_VISIBLE_DEVICES; helper verifies NVML physical metadata, not CUDA selector resolution",
            "mapping_constraint": "numeric Slurm ID must equal recorded NVIDIA device minor; arbitrary first-visible GPU is never accepted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slurm-global-id", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result, code = {}, 0
    try:
        command = ["nvidia-smi", "-q", "-x"]
        query = subprocess.run(command, capture_output=True, text=True, timeout=20)
        query_proof = {"command": command, "exit_code": query.returncode, "stderr": query.stderr,
                       "stdout_bytes": len(query.stdout.encode()), "scope": "read-only metadata; no GPU context/compute"}
        if query.returncode:
            raise RuntimeError(f"NVML XML query failed: exit={query.returncode}; stdout={query.stdout!r}; stderr={query.stderr!r}")
        result = resolve(args.slurm_global_id, args.cuda_visible_devices, query.stdout, proc_inventory())
        result["NVML_query"] = query_proof
    except Exception as exc:
        code = 2
        result = {"status": "failed_metadata_only", "error": str(exc), "slurm_global_gpu_id": args.slurm_global_id,
                  "CUDA_VISIBLE_DEVICES_preserved": args.cuda_visible_devices, "model_execution": False}
        print(f"GPU identity metadata failed: {exc}", file=sys.stderr)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if code == 0:
        print("\t".join(result[key] for key in ("uuid", "name", "driver")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
