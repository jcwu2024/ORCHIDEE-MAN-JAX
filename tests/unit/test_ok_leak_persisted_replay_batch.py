from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.dev.probe_ok_leak_driver_capture import (
    _atomic_write_json,
    _sha256_file,
)
from scripts.dev.verify_ok_leak_persisted_replay_batch import run_batch


def _sample(root: Path, day_index: int):
    capture = root / "capture" / f"day-{day_index}"
    capture.mkdir(parents=True)
    arrays = capture / "ok_leak_driver_series.npz"
    arrays.write_bytes(f"driver-{day_index}".encode())
    report = capture / "report.json"
    report.write_text(
        json.dumps({"capture_npz": arrays.name}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        landpoint_id="001.0-071.0",
        year=1961,
        day_index=day_index,
        report_path=report,
        capture_npz_sha256=_sha256_file(arrays),
    )


def _arguments(tmp_path: Path):
    return argparse.Namespace(
        capture_root=tmp_path / "capture",
        capture_plan=tmp_path / "capture-plan.json",
        dataset_manifest=tmp_path / "dataset-manifest.json",
        teacher_plan=tmp_path / "teacher-plan.json",
        expected_git_head="a" * 40,
        output=tmp_path / "output",
        limit=None,
    )


def _install_fakes(monkeypatch, dataset, calls, *, failed_days=()):
    import scripts.dev.verify_ok_leak_persisted_replay_batch as batch

    monkeypatch.setattr(batch, "load_ok_leak_driver_dataset", lambda *_: dataset)

    def fake_probe(arguments):
        calls.append(arguments.day_index)
        arguments.output.mkdir(parents=True, exist_ok=True)
        passed = arguments.day_index not in failed_days
        comparisons = {
            f"ok_leak.field_{index}": {
                "shape_equal": True,
                "exact": True,
                "max_absolute_error": 0.0,
                "defined_status_mismatches": 0,
            }
            for index in range(14)
        }
        report = {
            "schema_version": "ok_leak_driver_capture_probe_v2",
            "passed": passed,
            "landpoint_id": arguments.landpoint_id,
            "year": arguments.year,
            "day_index": arguments.day_index,
            "persisted_driver_npz_sha256": _sha256_file(
                arguments.replay_driver_npz
            ),
            "persisted_ok_leak_endpoint_comparisons": comparisons,
            "persisted_ok_leak_endpoint_within_1e-12": passed,
            "sealed_test_used": False,
        }
        _atomic_write_json(arguments.output / "report.json", report)
        return report

    monkeypatch.setattr(batch, "run_probe", fake_probe)


def test_persisted_replay_batch_is_atomic_and_restartable(tmp_path, monkeypatch):
    samples = (_sample(tmp_path, 2), _sample(tmp_path, 3))
    dataset = SimpleNamespace(
        dataset_id="dataset-v5",
        teacher_git_head="b" * 40,
        contract_sha256="c" * 64,
        capture_plan_sha256="d" * 64,
        samples=samples,
    )
    calls = []
    _install_fakes(monkeypatch, dataset, calls)
    args = _arguments(tmp_path)

    first = run_batch(args)
    assert first["status"] == "complete"
    assert first["completed_record_count"] == 2
    assert first["failed_record_count"] == 0
    assert calls == [2, 3]

    calls.clear()
    second = run_batch(args)
    assert second["status"] == "complete"
    assert calls == []
    assert all(item["resumed"] for item in second["records"])


def test_persisted_replay_batch_collects_failures(tmp_path, monkeypatch):
    samples = (_sample(tmp_path, 2), _sample(tmp_path, 3))
    dataset = SimpleNamespace(
        dataset_id="dataset-v5",
        teacher_git_head="b" * 40,
        contract_sha256="c" * 64,
        capture_plan_sha256="d" * 64,
        samples=samples,
    )
    calls = []
    _install_fakes(monkeypatch, dataset, calls, failed_days={2})

    result = run_batch(_arguments(tmp_path))

    assert result["status"] == "failed"
    assert result["completed_record_count"] == 1
    assert result["failed_record_count"] == 1
    assert calls == [2, 3]
