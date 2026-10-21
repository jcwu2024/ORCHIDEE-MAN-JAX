from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess

import pytest

from research.daily_coarse_graining import teacher_point_batch as batch
from research.daily_coarse_graining import teacher_point_verification as verification


class LedgerFixture:
    """Stage transitions persisted across independent serial invocations."""
    def __init__(self, work):
        self.work = work
        self.manifest_sha256 = "manifest-hash"
        self.manifest = {"shards": [{"landpoint_id": p} for p in ("A", "B", "C")]}
        self.connection = sqlite3.connect(work / "migration.sqlite")
        self.connection.execute("CREATE TABLE IF NOT EXISTS points(point PRIMARY KEY, record)")
        self.calls = []

    def record(self, point):
        row = self.connection.execute("SELECT record FROM points WHERE point=?", (point,)).fetchone()
        return None if row is None else json.loads(row[0])

    def save(self, point, record):
        self.connection.execute("INSERT OR REPLACE INTO points VALUES(?,?)", (point, json.dumps(record)))
        self.connection.commit()

    def pack(self, point):
        self.calls.append((point, "pack"))
        self.save(point, {"stage": "published"})

    def accept_reader_proof(self, point, path):
        self.calls.append((point, "accept"))
        self.save(point, {"stage": "reader_verified", "reader_proof": json.loads(path.read_text())})

    def retire(self, point, **kwargs):
        self.calls.append((point, "retire"))
        record = self.record(point)
        record["stage"] = "retired"
        self.save(point, record)


def test_serial_finite_batch_resumes_completed_proofs_from_one_durable_ledger(tmp_path, monkeypatch):
    proved = []

    def prove(migration, point, *, scan_full_index):
        assert scan_full_index is False
        assert migration.record(point)["stage"] == "published"
        proved.append(point)
        return {"landpoint_id": point, "full_result": [1, 2, 3]}

    monkeypatch.setattr(verification, "prove_point", prove)
    first = LedgerFixture(tmp_path)
    result = batch.run_batch(first, max_points=1, time_budget_seconds=60)
    assert result["processed"] == ["A"] and result["stop_reason"] == "max_points"
    first.connection.close()
    second = LedgerFixture(tmp_path)
    result = batch.run_batch(second, max_points=1, time_budget_seconds=60)
    assert result["processed"] == ["B"] and result["skipped_complete"] == 1
    assert proved == ["A", "B"]
    assert second.record("A")["reader_proof"]["landpoint_id"] == "A"
    assert second.record("B")["reader_proof"]["full_result"] == [1, 2, 3]
    assert list(tmp_path.glob("*.json")) == [tmp_path / "batch-reader-proof.json"]
    second.connection.close()


def test_soft_deadline_never_starts_another_point(tmp_path, monkeypatch):
    clock = {"now": 0}
    monkeypatch.setattr(batch.time, "monotonic", lambda: clock["now"])

    def prove(*args, **kwargs):
        clock["now"] = 100
        return {"passed": True}

    monkeypatch.setattr(verification, "prove_point", prove)
    migration = LedgerFixture(tmp_path)
    result = batch.run_batch(migration, max_points=3, time_budget_seconds=10)
    assert result["processed"] == ["A"] and result["stop_reason"] == "time_budget"
    assert migration.record("B") is None
    migration.connection.close()


def test_dependency_expiry_rechecked_after_long_proof_before_any_retirement(tmp_path, monkeypatch):
    clock = {"now": 10}
    monkeypatch.setattr(batch.time, "time", lambda: clock["now"])
    dependency = tmp_path / "dependencies.json"
    dependency.write_text(json.dumps({"dataset_manifest_sha256": "manifest-hash",
                                     "active_queued_audit_complete": True,
                                     "expires_unix": 20, "preserve_paths": []}))

    def prove(*args, **kwargs):
        clock["now"] = 30
        return {"passed": True}

    monkeypatch.setattr(verification, "prove_point", prove)
    migration = LedgerFixture(tmp_path)
    with pytest.raises(ValueError, match="dependency audit"):
        batch.run_batch(migration, max_points=1, time_budget_seconds=60,
                        retire=True, review_token="review", dependencies=dependency)
    assert migration.record("A")["stage"] == "reader_verified"
    assert ("A", "retire") not in migration.calls
    migration.connection.close()


@pytest.mark.parametrize("partition,host", [("cnmix", "test01"), ("cnmix", "cln02"),
                                           ("cnmix", "gln01"), ("cnall", "cn001"),
                                           ("", "cn001")])
def test_production_execution_rejects_free_login_and_wrong_partitions(monkeypatch, partition, host):
    monkeypatch.setattr(batch.socket, "gethostname", lambda: host)
    monkeypatch.setenv("SLURM_JOB_PARTITION", partition)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    with pytest.raises(ValueError, match="paid cnmix"):
        batch.validate_execution(Path("/WORK/liwei_work/jcwu/data"), Path("/WORK/liwei_work/jcwu/work"))


@pytest.mark.parametrize("extra", [[], ["--retire"], ["--review-token", "review"],
                                  ["--time-budget-seconds", "inf"]])
def test_cli_requires_explicit_finite_budget_and_complete_retirement_arguments(extra):
    arguments = ["--dataset-root", "/x", "--work-root", "/y", "--expected-commit", "a" * 40]
    if extra:
        arguments += ["--max-points", "1", "--time-budget-seconds", "60"]
    with pytest.raises(SystemExit):
        batch.main(arguments + extra)


def test_checkout_requires_exact_commit_and_clean_tree_without_git_C(monkeypatch):
    monkeypatch.setattr(batch, "CHECKOUT", Path.cwd().resolve())
    observed = []
    dirty = {"value": False}

    def git(arguments, **kwargs):
        observed.append(arguments)
        if arguments == ["git", "rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        return " M changed.py\n" if dirty["value"] else ""

    monkeypatch.setattr(batch.subprocess, "check_output", git)
    assert batch.validate_checkout("a" * 40) == "a" * 40
    with pytest.raises(ValueError, match="expected commit"):
        batch.validate_checkout("b" * 40)
    dirty["value"] = True
    with pytest.raises(ValueError, match="clean"):
        batch.validate_checkout("a" * 40)
    assert all("-C" not in arguments for arguments in observed)


def test_cli_dry_run_never_creates_ledger_or_checks_compute_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "validate_checkout", lambda commit: commit)

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not enter production allocation or create a ledger")

    monkeypatch.setattr(batch, "Migration", forbidden)
    monkeypatch.setattr(batch, "validate_execution", forbidden)
    work = tmp_path / "not-created"
    assert batch.main(["--dataset-root", str(tmp_path), "--work-root", str(work),
                       "--expected-commit", "a" * 40, "--max-points", "1",
                       "--time-budget-seconds", "60", "--dry-run"]) == 0
    assert not work.exists()


def test_launcher_actual_dry_run_argument_expansion_preserves_spaces_and_tokens():
    shell = shutil.which("bash")
    if not shell:
        pytest.skip("bash is unavailable")
    script = Path(__file__).resolve().parents[2] / "scripts/hpc/slurm_teacher_point_storage.sh"
    checkout = "/WORK/liwei_work/jcwu/code checkout"
    args = ["--dataset-root", "/WORK/liwei_work/jcwu/dataset",
            "--work-root", "/WORK/liwei_work/jcwu/work space",
            "--expected-commit", "a" * 40, "--max-points", "2",
            "--time-budget-seconds", "600", "--retire", "--review-token", "review token",
            "--dependencies", "/WORK/liwei_work/jcwu/dependencies.json"]
    syntax = subprocess.run([shell, "-n", str(script)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    result = subprocess.run([shell, str(script), "--dry-run", "--checkout", checkout, *args],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert shlex.split(lines[0]) == ["cwd=" + checkout]
    command = shlex.split(lines[1])
    assert command[:3] == ["nice", "-n", "15"]
    assert command[3] == "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python"
    assert command[4:7] == ["-B", "-m", "research.daily_coarse_graining.teacher_point_batch"]
    assert command[7:] == [*args, "--dry-run"]
