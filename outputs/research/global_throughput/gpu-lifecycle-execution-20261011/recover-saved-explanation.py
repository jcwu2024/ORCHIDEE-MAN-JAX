"""Verify and recover the compact CPU explanation; never load model arrays."""
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

root = Path(__file__).resolve().parent
archive = root / "saved-array-explanation-15769234.tar.gz"
destination = root / "saved-array-explanation"
expected_sha = "64cdcf2537c01897fea0e56da255d63df0d313f7f3daaf7005cc041c85fb9148"
remote = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-15769234-saved-array-explanation/"
raw = archive.read_bytes()
assert len(raw) == 12481 and hashlib.sha256(raw).hexdigest() == expected_sha
assert not destination.exists(), "preserve previous recovery"
destination.mkdir()
with tarfile.open(archive, "r:gz") as stream:
    for member in stream.getmembers():
        relative = PurePosixPath(member.name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert member.isdir() or member.isfile(), "no links/devices in compact recovery"
        target = destination.joinpath(*relative.parts).resolve()
        assert target.is_relative_to(destination.resolve())
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source:
                target.write_bytes(source.read())
inventory = {}
for line in (destination / "saved-array-explanation-inventory.tsv").read_text().splitlines():
    size, name = line.split("\t", 1)
    inventory[name] = int(size)
verified = []
for line in (destination / "saved-array-explanation-sha256.txt").read_text().splitlines():
    digest, absolute = line.split("  ", 1)
    assert absolute.startswith(remote)
    name = absolute[len(remote):]
    data = (destination / name).read_bytes()
    assert len(data) == inventory[name] and hashlib.sha256(data).hexdigest() == digest
    verified.append({"path": name, "bytes": len(data), "sha256": digest})
assert len(verified) == len(inventory) == 10
assert (destination / "exit-status.txt").read_text().strip() == "0"
summary = json.loads((destination / "comparison/summary.json").read_text())
assert summary["status"] == "complete_saved_initial_failure_comparison"
assert summary["all760_leaves_and2048_lanes_checked"] is True
assert summary["field_mapping_complete"] is True
assert summary["scientific_acceptance"] is False
assert all(summary["runtime"][key] is False for key in ("JAX_imported", "jaxlib_imported", "model_imported"))
preflight = root / "cpu-explanation-preflight-old-git.sh"
assert hashlib.sha256(preflight.read_bytes()).hexdigest() == "e31c61efb8ebc14b9e48127afe8123ed30f36806774ad36ffc7da23de0450c24"
record = {
    "status": "verified_compact_CPU_explanation_recovery",
    "archive": {"path": str(archive), "bytes": len(raw), "sha256": expected_sha},
    "verified_files": verified,
    "launcher": {"path": "cpu-explanation-run.sh", "sha256": hashlib.sha256((root / "cpu-explanation-run.sh").read_bytes()).hexdigest()},
    "preflight_before_array_loading": {"command_receipt": "20-compare-saved-arrays.command.json", "archived_stdin": preflight.name, "sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(), "failure": "test03 default old Git does not support -C; stopped before input verification, output creation or array loading"},
    "actual_comparison_command_receipt": "21-compare-saved-arrays.command.json",
    "CPU_array_comparison_runs": 1,
    "model_JAX_GPU_execution": False,
    "original_scientific_arrays_downloaded": False,
    "raw_failed_arrays_retained_remote": True,
    "scope": "verify transferred metadata/scalar differences only; numerical interpretation is separate",
}
(root / "EXPLANATION_COLLECTION.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": record["status"], "verified_files": len(verified), "failing_leaf_count": summary["failing_leaf_count"], "failing_lane_count": summary["failing_lane_count"]}))
