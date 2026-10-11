"""Verify and recover compact terminal metadata without loading scientific arrays."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
PIN = "3b4ae35e695f85480ecb72461e4ba93f4534bbd8"
ARCHIVE_SHA = "538f4254a9b1d03afc4c010d4942278b9153f5d1cf98d4043c02f650251e07b5"
archive = ROOT / "terminal-evidence-15769234.tar.gz"
destination = ROOT / "terminal-evidence"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA
assert archive.stat().st_size == 87684
assert not destination.exists(), "single-use metadata recovery directory"
with tarfile.open(archive, "r:gz") as stream:
    members = stream.getmembers()
    for member in members:
        relative = PurePosixPath(member.name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert member.isfile() or member.isdir(), "links/devices are not evidence files"
        assert not member.name.endswith(".pkl"), "array retrieval is outside compact recovery"
    stream.extractall(destination, filter="data")

checksums = {}
for row in (destination / "run-sha256.txt").read_text(encoding="utf-8").splitlines():
    digest, name = row.split("  ", 1)
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and ".." not in relative.parts
    checksums[relative.as_posix().removeprefix("./")] = digest
verified = []
for path in sorted((destination / "run").rglob("*")):
    if path.is_file():
        relative = path.relative_to(destination / "run").as_posix()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksums[relative]
        verified.append(relative)

assert (destination / "terminal-code-identity.txt").read_bytes() == (PIN + "\n").encode()
rows = list(csv.DictReader(io.StringIO((destination / "terminal-sacct.txt").read_text()), delimiter="|"))
parent = next(row for row in rows if row["JobIDRaw"] == "15769234")
assert parent["State"] == "FAILED" and parent["ExitCode"] == "1:0"
assert parent["ElapsedRaw"] == "1140" and parent["AllocCPUS"] == "4"
orchestration = json.loads((destination / "run/orchestration.json").read_bytes())
assert orchestration["expected_code_sha"] == PIN
assert orchestration["status"] == "first_arm_failed"
assert set(orchestration["children"]) == {"phased_resident_control"}
assert orchestration["comparison_allowed"] is False and orchestration["ratio_published"] is False
report_path = destination / "run/phased_resident_control/capacity/B1024/report.json"
report = json.loads(report_path.read_bytes())
failure = report["initial_validation_failure"]
array_relative = "phased_resident_control/capacity/B1024/failed_initial_arrays.pkl"
assert failure["sha256"] == checksums[array_relative]
phases = report.get("phases", [])
assert not any("warm_sample" in phase["phase"] for phase in phases)
assert not (destination / "run/comparison.json").exists()

record = {
    "schema": "global-gpu-lifecycle-terminal-metadata-recovery-v1",
    "archive": archive.name,
    "archive_sha256": ARCHIVE_SHA,
    "archive_bytes": archive.stat().st_size,
    "source_sha256_receipt": "15-collect-terminal.stdout.txt",
    "verified_compact_run_files": verified,
    "execution_sha": PIN,
    "terminal_checkout_clean": True,
    "parent_slurm": parent,
    "orchestration_status": orchestration["status"],
    "only_control_dispatched": True,
    "failed_initial_array_sha256": failure["sha256"],
    "failed_array_hash_matches_independent_collection_manifest": True,
    "scientific_arrays_retrieved_or_loaded": False,
    "scientific_arrays_preserved_remote": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-15769234/" + array_relative,
    "array_and_scientific_acceptance": False,
    "throughput_ratio": None,
    "model_or_jax_execution": False,
}
(ROOT / "COLLECTION_VERIFICATION.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"verification": "passed", "verified_compact_run_files": len(verified),
                  "report_status": report.get("status"), "failure": failure,
                  "phases": phases}, indent=2))
