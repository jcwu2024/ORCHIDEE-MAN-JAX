"""Verify saved delivery bytes and claims locally; never execute the model."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
WORKTREE = BASE.parents[3]
PIN = "60dccbddde10be4295aab7e7fe55fa7864d3c3f3"
START = "e762c6f87d9f2b61f8531c3688e1f86d3bbd1bac"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=WORKTREE, text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    assert Path(git("rev-parse", "--show-toplevel")).resolve() == WORKTREE
    assert git("branch", "--show-current") == "research/global-throughput"
    assert git("rev-parse", "HEAD") == START
    assert git("rev-parse", "codex/global-cpu48-b64-v4") == PIN
    assert git("rev-parse", "HEAD:jax_orchidee") == "62e90547ee8eadab387713397bb111c399c81b6c"
    assert git("rev-parse", f"{PIN}:jax_orchidee") == "75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba"
    assert not git("status", "--porcelain", "--", "jax_orchidee", "scripts", "tests", "configs")

    manifest_path = BASE / "runtime-evidence-manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["job_id"] == "15853683"
    assert manifest["terminal_status"] == "FAILED"
    paths = []
    for record in manifest["records"]:
        path = (BASE / record["local_path"]).resolve()
        assert path.is_relative_to(BASE)
        assert path.stat().st_size == record["bytes"], record["local_path"]
        assert sha(path) == record["sha256"], record["local_path"]
        if path.suffix == ".json":
            read_json(path)
        paths.append(record["local_path"])

    operations_checked = 0
    for path in sorted((BASE / "operations").glob("*.json")):
        operation = read_json(path)
        for stream in ("stdout", "stderr"):
            if stream not in operation:
                continue
            receipt = operation[stream]
            raw = Path(receipt["path"]).resolve()
            assert raw.is_relative_to(BASE)
            assert raw.stat().st_size == receipt["bytes"], str(raw)
            assert sha(raw) == receipt["sha256"], str(raw)
        if operation.get("stdin_sha256"):
            source = (WORKTREE / operation["source"]).resolve()
            assert source.is_relative_to(BASE)
            assert sha(source) == operation["stdin_sha256"], str(source)
        operations_checked += 1

    preflight = read_json(BASE / "preflight/15853600/production-preflight.json")
    assert preflight["status"] == "actual_production_identity_and_seal_passed"
    assert preflight["JAX_imported"] is False and preflight["model_execution"] is False
    report = read_json(BASE / "paid-run/batch/report.json")
    assert report["status"] == "failed"
    assert report["phases"] == [] and report["gates"] == [] and report["timing"] == {}
    assert report["message"] == "FULL64 CPU scalar admission candidate provenance differs from current execution identity"
    resources = report["server_resources"]
    child = resources["actual_child_prejax_receipt"]
    assert child["status"] == "resource_admission_passed"
    assert child["allocation_admitted"] is True
    assert len(set(child["selected_physical_cores"])) == 48
    assert child["actual_environment"] == child["thread_policy"]
    assert child["JAX_imported"] is False
    assert report["environment"]["jax"] == "0.4.38"
    assert report["environment"]["jaxlib"] == "0.4.38"
    assert report["environment"]["x64_enabled"] is True
    for key, value in preflight["actual_candidate_identity"].items():
        if key != "job_id":
            assert resources["candidate_provenance"][key] == value, key

    collector = read_json(BASE / "collector-result.json")
    assert collector["status"] == "incomplete_run_failure_preserved"
    assert collector["scientific_acceptance"] is False
    assert collector["claim_scope"] == "failure accounting only; no throughput claim"
    new = collector["whole_parent_allocation_cost"]
    old = collector["reused_scalar_source_parent_cost"]
    assert new["job_id"] == "15853683" and new["wall_seconds"] == 50
    assert math.isclose(new["allocated_parent_CPU_request_core_hours"], 48 * 50 / 3600)
    assert math.isclose(new["primary_cost_cny_estimate"], 48 * 50 / 3600 * 0.05)
    assert math.isclose(new["conservative_cost_cny_estimate"], 48 * 50 / 3600 * 0.07)
    assert old["job_id"] == "15820467" and old["wall_seconds"] == 2197
    assert old["counted_once_as_source_parent"] is True
    assert old["new_allocation_reuse_charge"] is False
    assert new["billing_verified"] is False and old["billing_verified"] is False

    documents = [WORKTREE / "docs/research/global_throughput" / name for name in
                 ("README.md", "SERVER_BENCHMARK.md", "SUPERVISOR_HANDOFF.md")]
    added_links = 0
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8-sig")):
            if "cpu48-b64-execution-v4-20261014" in target:
                assert (document.parent / target).resolve().exists(), target
                added_links += 1

    result = {
        "schema": "cpu48-v4-local-delivery-acceptance-v1",
        "status": "passed_saved_failure_evidence_only",
        "starting_research_commit": START,
        "execution_commit": PIN,
        "manifest_sha256": sha(manifest_path),
        "manifest_records_verified": len(paths),
        "operation_receipts_verified": operations_checked,
        "current_entry_links_verified": added_links,
        "terminal_job_id": "15853683",
        "terminal_seconds": 50,
        "allocated_cpus": 48,
        "model_execution": False,
        "remote_actions": False,
        "source_changes": False,
        "scientific_acceptance": False,
        "throughput_claim": False,
        "specific_failed_identity_field_resolved": False,
        "scope": "local bytes, recorded failure/resource/accounting boundaries and current entry links; no model rerun",
    }
    (BASE / "LOCAL_ACCEPTANCE.json").write_bytes(
        (json.dumps(result, indent=2) + "\n").encode("utf-8")
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
