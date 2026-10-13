"""Version-package admission uses tiny Git trees, never a model or server."""
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from scripts.dev import prepare_global_cpu48_execution as package


def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()
    run("init", "-b", "research/test")
    run("config", "user.name", "Fixture")
    run("config", "user.email", "fixture@example.invalid")
    (root / "jax_orchidee").mkdir()
    (root / "jax_orchidee/science.py").write_text("value = 1\n")
    (root / "scripts/dev").mkdir(parents=True)
    (root / "scripts/dev/runner.py").write_text("budget = 4\n")
    run("add", ".")
    run("commit", "-m", "base")
    base = run("rev-parse", "HEAD")
    (root / "jax_orchidee/science.py").write_text("value = 2\n")
    (root / "scripts/dev/runner.py").write_text("budget = 48\n")
    run("add", ".")
    run("commit", "-m", "research keeps new science")
    return root, run, base


def test_execution_keeps_base_science_and_research_working_tree(tmp_path):
    root, run, base = repository(tmp_path)
    current = run("rev-parse", "HEAD")
    digest = hashlib.sha256(b"jax_orchidee/science.py\0value = 1\n\0").hexdigest()
    output = root / "outputs/version"
    result = package.build(root, output, source=current, base=base, ref="codex/test-execution",
                           harness_paths=["scripts/dev/runner.py"], expected_science=digest)
    execution = result["execution_code_commit"]
    assert run("show", f"{execution}:jax_orchidee/science.py") == "value = 1"
    assert run("show", f"{execution}:scripts/dev/runner.py") == "budget = 48"
    assert run("rev-parse", "HEAD") == current
    assert (root / "jax_orchidee/science.py").read_text() == "value = 2\n"
    assert run("status", "--porcelain") == "?? outputs/"
    assert result["scientific_source"]["every_scientific_blob_identical_to_base"]
    assert result["remote_deployment_or_submission"] is False
    assert result["execution_route"] == "batch-only"
    assert result["allocation_plan"]["new_scalar_generation"] is False
    assert result["allocation_plan"]["wall_seconds"] == 3600
    assert result["allocation_plan"]["conservative_worst_case_CNY"] == pytest.approx(3.36)


@pytest.mark.parametrize("path", ["jax_orchidee/science.py", "scripts/dev/../science.py", "scripts/dev/data.pkl"])
def test_science_and_non_harness_paths_are_refused(path):
    with pytest.raises(ValueError, match="allowlist"):
        package.validate_paths([path])


def test_unapproved_scientific_base_is_refused_before_creating_ref(tmp_path):
    root, run, base = repository(tmp_path)
    with pytest.raises(ValueError, match="scientific identity"):
        package.build(root, root / "outputs/version", source="HEAD", base=base,
                      ref="codex/test-execution", harness_paths=["scripts/dev/runner.py"])
    assert run("branch", "--list", "codex/test-execution") == ""


def test_version_ref_is_reusable_but_never_overwritten(tmp_path):
    root, run, base = repository(tmp_path)
    (root / ".git/info/exclude").write_text("outputs/\n")
    digest = hashlib.sha256(b"jax_orchidee/science.py\0value = 1\n\0").hexdigest()
    arguments = dict(source="HEAD", base=base, ref="codex/test-execution",
                     harness_paths=["scripts/dev/runner.py"], expected_science=digest)
    first = package.build(root, root / "outputs/version", **arguments)
    second = package.build(root, root / "outputs/version", **arguments)
    execution = first["execution_code_commit"]
    assert second["execution_code_commit"] == execution
    (root / "scripts/dev/runner.py").write_text("budget = 49\n")
    run("add", "scripts/dev/runner.py")
    run("commit", "-m", "changed harness requires a new execution version")
    with pytest.raises(ValueError, match="existing execution ref differs"):
        package.build(root, root / "outputs/version", **arguments)
    assert run("rev-parse", "codex/test-execution") == execution
    assert run("show", f"{execution}:scripts/dev/runner.py") == "budget = 48"
    assert run("show", "HEAD:jax_orchidee/science.py") == "value = 2"


def test_dirty_research_is_refused_without_creating_execution_ref(tmp_path):
    root, run, base = repository(tmp_path)
    (root / "scripts/dev/runner.py").write_text("budget = 49\n")
    with pytest.raises(ValueError, match="must be clean"):
        package.build(root, root / "outputs/version", source="HEAD", base=base,
                      ref="codex/test-execution", harness_paths=["scripts/dev/runner.py"])
    assert run("branch", "--list", "codex/test-execution") == ""
    assert (root / "scripts/dev/runner.py").read_text() == "budget = 49\n"


@pytest.mark.parametrize("budget", [True, 1799, 5401, 3600.5])
def test_budget_outside_approved_envelope_creates_no_execution_ref(tmp_path, budget):
    root, run, base = repository(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        package.build(root, root / "outputs/version", source="HEAD", base=base,
                      ref="codex/test-execution", harness_paths=["scripts/dev/runner.py"],
                      budget_seconds=budget)
    assert run("branch", "--list", "codex/test-execution") == ""


def test_tampered_scalar_audit_is_refused_before_creating_ref(tmp_path):
    root, run, base = repository(tmp_path)
    (root / ".git/info/exclude").write_text("outputs/\n")
    receipt = root / "outputs/version/scalar-audit/audit.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"{}\n")
    digest = hashlib.sha256(b"jax_orchidee/science.py\0value = 1\n\0").hexdigest()
    with pytest.raises(ValueError, match="audit receipt checksum"):
        package.build(root, root / "outputs/version", source="HEAD", base=base,
                      ref="codex/test-execution", harness_paths=["scripts/dev/runner.py"],
                      expected_science=digest, scalar_audit_receipt=receipt,
                      expected_scalar_audit_sha256="0" * 64)
    assert run("branch", "--list", "codex/test-execution") == ""


def test_scalar_audit_cannot_escape_preparation_output(tmp_path):
    root, run, base = repository(tmp_path)
    receipt = tmp_path / "audit.json"
    receipt.write_bytes(b"{}\n")
    digest = hashlib.sha256(b"jax_orchidee/science.py\0value = 1\n\0").hexdigest()
    with pytest.raises(ValueError, match="inside the preparation output"):
        package.build(root, root / "outputs/version", source="HEAD", base=base,
                      ref="codex/test-execution", harness_paths=["scripts/dev/runner.py"],
                      expected_science=digest, scalar_audit_receipt=receipt,
                      expected_scalar_audit_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    assert run("branch", "--list", "codex/test-execution") == ""


def test_external_admission_binds_exact_execution_and_preserves_existing_bytes(tmp_path):
    root, run, base = repository(tmp_path)
    (root / ".git/info/exclude").write_text("outputs/\n")
    digest = hashlib.sha256(b"jax_orchidee/science.py\0value = 1\n\0").hexdigest()
    output = root / "outputs/version"
    receipt = output / "scalar-audit/audit.json"
    receipt.parent.mkdir(parents=True)
    # Only a tiny metadata fixture exercises version binding here. The real
    # saved-array audit and runtime validator have their own admission tests.
    raw_parent = "15820467|FAILED|00:36:37|billing=48,cpu=48,node=1\n"
    source = {"job_id": "15820467", "execution_commit": base,
        "stage_sha256": package.SCALAR_CHECKPOINT_SHA,
        "prepared_input_sha256": package.PREPARED_INPUT_SHA,
        "gpu_reference_sha256": package.REFERENCE,
        "certified_science_sha256": digest, "actual_science_source_sha256": digest,
        "physical_science_tree_oid": run("rev-parse", f"{base}:jax_orchidee"),
        "baseline_ref": package.BASELINE,
        "whole_parent_accounting": {
            "sha256": "a04c687530f390ad1d5f1d3668293a552b3a9b37f0cd05ef1f264660c2f37113",
            "job_id": "15820467", "wall_seconds": 2197, "allocated_CPU_count": 48,
            "GPU_count": 0, "counted_once_as_source_parent": True,
            "raw_record": raw_parent, "raw_record_sha256": hashlib.sha256(raw_parent.encode()).hexdigest()},
        "source_identity": {"science_sha256": digest,
            "numerical_reference_sha256": package.REFERENCE, "baseline_ref": package.BASELINE}}
    receipt.write_text(json.dumps({"schema": "global-cpu48-scalar-audit-v1",
        "status": "saved_arrays_audited", "scientific_acceptance": False,
        "scalar_source": source, "audit": {"strict_numerical_passed": False,
            "frozen_reset_arrays_recomputed": False, "GPU_gate_count": 64,
            "GPU_passed_gate_count": 61, "GPU_tolerance_failed_gate_count": 3,
            "GPU_tolerance_failure_count": 4, "recomputed_GPU_gates": [{}] * 64}}))
    arguments = dict(source="HEAD", base=base, ref="codex/test-execution",
        harness_paths=["scripts/dev/runner.py"], expected_science=digest,
        scalar_audit_receipt=receipt,
        expected_scalar_audit_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    result = package.build(root, output, **arguments)
    admission = output / result["external_scalar_reference"]["admission_record"]["name"]
    original = admission.read_bytes()
    bound = json.loads(original)
    assert bound["candidate_identity"]["execution_commit"] == result["execution_code_commit"]
    assert bound["candidate_identity"]["execution_commit"] != run("rev-parse", "HEAD")
    assert bound["candidate_identity"]["job_id"] is None
    assert bound["scalar_source"]["execution_commit"] == base
    assert result["external_scalar_reference"]["admission_record"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert package.build(root, output, **arguments)["execution_code_commit"] == result["execution_code_commit"]
    assert admission.read_bytes() == original
    bound["candidate_identity"]["execution_commit"] = run("rev-parse", "HEAD")
    admission.write_text(json.dumps(bound))
    changed = admission.read_bytes()
    with pytest.raises(ValueError, match="existing scalar admission differs"):
        package.build(root, output, **arguments)
    assert admission.read_bytes() == changed
    assert run("rev-parse", "codex/test-execution") == result["execution_code_commit"]
