"""Version-package admission uses tiny Git trees, never a model or server."""
from __future__ import annotations

import hashlib
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
