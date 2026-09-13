from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.hpc.merge_expanded_one_day import _expected_schedule
from scripts.hpc.prepare_expanded_one_day import (
    PREPARATION_SCHEMA,
    _preparation_identity,
    _validate_resume_report,
    run,
)


def _plan(tmp_path):
    dataset = (tmp_path / "dataset-manifest.json").resolve()
    dataset.write_text("{}", encoding="utf-8")
    plan_path = (tmp_path / "plan.json").resolve()
    entries = []
    for identifier in (1, 2):
        entries.append(
            {
                "id": identifier,
                "landpoint_id": f"point-{identifier:03d}",
                "year": 1978 + identifier,
                "parent_shard_sha256": f"parent-{identifier}",
                "typed_shard_sha256": f"typed-{identifier}",
                "start_rows": [identifier, identifier + 10],
                "new_start_rows": [identifier + 10],
            }
        )
    plan = {
        "schema_version": "one_day_expansion_plan_v1",
        "checks": {"train_only": True},
        "dataset_manifest": str(dataset),
        "source_identity": {"training_contract_sha256": "contract"},
        "source_manifest": str(tmp_path / "source.json"),
        "seed": 20260913,
        "rows_per_point_year": 2,
        "entries": entries,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan, plan_path, dataset


def _entry(plan, output_dir, identifier):
    path = output_dir / f"point-year-{identifier:06d}.pkl.gz"
    path.write_bytes(f"accepted-{identifier}".encode())
    planned = plan["entries"][identifier - 1]
    return {
        "update": identifier,
        "path": path.name,
        "rows": len(planned["start_rows"]),
        "new_rows": len(planned["new_start_rows"]),
        "landpoint_id": planned["landpoint_id"],
        "year": planned["year"],
        "schedule_entry": _expected_schedule(planned),
        "old_row_exact": True,
        "size_bytes": path.stat().st_size,
    }


def _report(plan, plan_path, entries, *, completed=False):
    return {
        "schema_version": PREPARATION_SCHEMA,
        "status": "passed" if completed else "running",
        "stage": "completed" if completed else "point_year_completed",
        "persisted_reader_exact": completed,
        "plan": str(plan_path),
        "entry_ids": [1, 2],
        "identity": _preparation_identity(plan, plan_path),
        "entries": entries,
        "elapsed_seconds": 12.5,
    }


def test_resume_accepts_only_committed_prefix_and_leaves_incomplete_payload_uncommitted(tmp_path):
    plan, plan_path, _dataset = _plan(tmp_path)
    output_dir = tmp_path / "worker"
    output_dir.mkdir()
    first = _entry(plan, output_dir, 1)
    incomplete = output_dir / "point-year-000002.pkl.gz"
    incomplete.write_bytes(b"interrupted")
    report_path = output_dir / "prepared_manifest.json"

    assert not _validate_resume_report(
        _report(plan, plan_path, [first]),
        report_path=report_path,
        plan=plan,
        plan_path=plan_path,
        ids=[1, 2],
    )
    assert (output_dir / first["path"]).read_bytes() == b"accepted-1"
    assert incomplete.read_bytes() == b"interrupted"


@pytest.mark.parametrize("drift", ["identity", "range", "prefix", "schedule", "size", "path"])
def test_resume_rejects_selection_or_committed_entry_drift(tmp_path, drift):
    plan, plan_path, _dataset = _plan(tmp_path)
    output_dir = tmp_path / "worker"
    output_dir.mkdir()
    report_path = output_dir / "prepared_manifest.json"
    first = _entry(plan, output_dir, 1)
    report = _report(plan, plan_path, [first])
    ids = [1, 2]
    if drift == "identity":
        report["identity"]["expansion_seed"] += 1
    elif drift == "range":
        ids = [1]
    elif drift == "prefix":
        report["entries"][0]["update"] = 2
    elif drift == "schedule":
        report["entries"][0]["schedule_entry"]["runtime_year"] += 1
    elif drift == "size":
        report["entries"][0]["size_bytes"] += 1
    else:
        report["entries"][0]["path"] = "other.pkl.gz"

    with pytest.raises(ValueError, match="resume"):
        _validate_resume_report(
            report,
            report_path=report_path,
            plan=plan,
            plan_path=plan_path,
            ids=ids,
        )


def test_completed_resume_returns_without_bootstrap_or_rewriting_manifest(tmp_path, monkeypatch):
    plan, plan_path, dataset = _plan(tmp_path)
    output_dir = tmp_path / "worker"
    output_dir.mkdir()
    entries = [_entry(plan, output_dir, identifier) for identifier in (1, 2)]
    report = _report(plan, plan_path, entries, completed=True)
    report_path = output_dir / "prepared_manifest.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    original = report_path.read_bytes()

    def unexpected_git(*_args, **_kwargs):
        raise AssertionError("completed resume must return before bootstrap")

    monkeypatch.setattr("scripts.hpc.prepare_expanded_one_day.subprocess.check_output", unexpected_git)
    args = SimpleNamespace(
        plan=plan_path,
        dataset_manifest=dataset,
        output_dir=output_dir,
        entry_ids=None,
        start_entry=1,
        end_entry=2,
        resume=True,
    )

    assert run(args) == deepcopy(report)
    assert report_path.read_bytes() == original


def test_fresh_mode_still_rejects_an_existing_manifest(tmp_path):
    plan, plan_path, dataset = _plan(tmp_path)
    output_dir = tmp_path / "worker"
    output_dir.mkdir()
    (output_dir / "prepared_manifest.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        plan=plan_path,
        dataset_manifest=dataset,
        output_dir=output_dir,
        entry_ids=None,
        start_entry=1,
        end_entry=2,
        resume=False,
    )

    with pytest.raises(FileExistsError, match="new output directory"):
        run(args)
