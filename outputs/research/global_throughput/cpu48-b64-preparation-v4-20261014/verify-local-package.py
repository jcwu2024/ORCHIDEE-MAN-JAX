"""Verify the actual v4 artifacts with stdlib and Git; never run the model."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
EXECUTION = "60dccbddde10be4295aab7e7fe55fa7864d3c3f3"
SOURCE = "493252e418f3fd827af4a6f7748a63d3157e680d"
BASE = "03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8"
TREE = "75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba"
RESEARCH_TREE = "62e90547ee8eadab387713397bb111c399c81b6c"
ADMISSION_SHA = "f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d"
AUDIT_SHA = "cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431"


def git(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, check=True
    ).stdout


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    pin_bytes = (OUT / "execution-code-pin.json").read_bytes()
    pin = json.loads(pin_bytes)
    assert pin["execution_code_commit"] == EXECUTION
    assert pin["source_research_commit"] == SOURCE
    assert pin["base_execution_commit"] == BASE
    assert git("rev-parse", pin["execution_ref"]).decode().strip() == EXECUTION
    assert git("rev-parse", EXECUTION + "^").decode().strip() == BASE
    assert git("rev-parse", "HEAD").decode().strip() == SOURCE
    assert git("symbolic-ref", "--short", "HEAD").decode().strip() == "research/global-throughput"
    assert not git("status", "--porcelain", "--untracked-files=all").strip()
    assert git("rev-parse", EXECUTION + ":jax_orchidee").decode().strip() == TREE
    assert git("rev-parse", "HEAD:jax_orchidee").decode().strip() == RESEARCH_TREE
    physical = git("ls-tree", "-r", BASE, "--", "jax_orchidee")
    assert physical == git("ls-tree", "-r", EXECUTION, "--", "jax_orchidee")
    assert len(physical.splitlines()) == 104
    paths = pin["harness_paths"]
    assert len(paths) == len(set(paths)) == 15
    changed = git("diff", "--name-only", BASE, EXECUTION).decode().splitlines()
    assert sorted(changed) == sorted(paths) == sorted(pin["execution_changed_paths"])
    for path in paths:
        content = git("show", EXECUTION + ":" + path)
        assert content == git("show", SOURCE + ":" + path)
        assert sha(content) == pin["harness_blob_sha256"][path]
    reference = pin["external_scalar_reference"]
    artifacts = {
        "pin": {"name": "execution-code-pin.json", "sha256": sha(pin_bytes), "bytes": len(pin_bytes)}
    }
    for label, description in (
        ("patch", pin["patch"]),
        ("bundle", pin["bundle"]),
        ("audit", reference["audit_receipt"]),
        ("admission", reference["admission_record"]),
    ):
        content = (OUT / description["name"]).read_bytes()
        assert sha(content) == description["sha256"]
        assert len(content) == description["bytes"]
        artifacts[label] = {"name": description["name"], "sha256": sha(content), "bytes": len(content)}
    assert artifacts["admission"]["sha256"] == ADMISSION_SHA
    assert artifacts["audit"]["sha256"] == AUDIT_SHA
    admission = json.loads((OUT / artifacts["admission"]["name"]).read_bytes())
    assert admission["schema"] == "global-cpu48-scalar-admission-v1"
    assert admission["status"] == "admitted_for_internal_comparison_only"
    assert admission["candidate_identity"] == reference["candidate_identity"]
    assert admission["candidate_identity"]["execution_commit"] == EXECUTION
    assert admission["candidate_identity"]["job_id"] is None
    assert str(admission["scalar_source"]["job_id"]) == "15820467"
    assert admission["scalar_source"]["execution_commit"] == BASE
    assert reference["scientific_acceptance"] is False
    assert reference["new_scalar_generation"] is False
    allocation = pin["allocation_plan"]
    assert (allocation["partition"], allocation["nodes"], allocation["tasks"],
            allocation["allocated_CPUs"], allocation["GPUs"], allocation["wall_seconds"]) == (
                "cnall", 1, 1, 48, 0, 3600)
    bundle = subprocess.run(
        ["git", "bundle", "verify", str(OUT / pin["bundle"]["name"])],
        cwd=ROOT, capture_output=True, check=True
    )
    receipt = {
        "schema": "cpu48-v4-local-package-verification-v1",
        "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "conda run -n ORCJAX python -B " + Path(__file__).relative_to(ROOT).as_posix(),
        "cwd": ROOT.as_posix(),
        "execution_commit": EXECUTION,
        "source_research_commit": SOURCE,
        "base_execution_commit": BASE,
        "physical_files_compared": 104,
        "physical_tree": TREE,
        "research_tree_preserved": RESEARCH_TREE,
        "source_harness_blobs_compared": len(paths),
        "artifacts": artifacts,
        "bundle_verify": {
            "exit_code": bundle.returncode,
            "stdout": bundle.stdout.decode(),
            "stderr": bundle.stderr.decode(),
            "prerequisite": BASE,
        },
        "candidate_job_id": None,
        "scalar_job_id": "15820467",
        "scientific_acceptance": False,
        "model_or_jax_execution": False,
        "remote_deployment_or_submission": False,
    }
    (OUT / "local-package-verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "execution_commit": EXECUTION,
                      "physical_files": 104, "harness_blobs": len(paths),
                      "admission_sha256": ADMISSION_SHA, "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
