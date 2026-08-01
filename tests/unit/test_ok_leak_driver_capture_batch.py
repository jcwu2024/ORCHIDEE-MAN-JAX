from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.dev.capture_ok_leak_driver_batch import (
    _capture_directory,
    run_batch,
)
from scripts.dev.probe_ok_leak_driver_capture import _atomic_write_json, _sha256_file


def _record(day_index: int):
    return {
        "landpoint_id": "001.0-071.0",
        "year": 1961,
        "day_index": day_index,
        "row": day_index - 2,
        "source_shard": "shards/001.0-071.0/1961.npz",
        "source_shard_sha256": "a" * 64,
        "day_start_state_sha256": f"{day_index:064x}",
        "fast_day_target_sha256": f"{day_index + 1:064x}",
        "metrics": {},
        "selection_reasons": ["synthetic"],
    }


def _arguments(tmp_path: Path):
    records = [_record(2), _record(3)]
    plan = {
        "plan_sha256": "b" * 64,
        "dataset_id": "synthetic-v5",
        "records": records,
    }
    plan_path = tmp_path / "capture_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps({"plan_sha256": "teacher-plan-hash"}), encoding="utf-8")
    teacher_plan = tmp_path / "teacher_plan.json"
    teacher_plan.write_text("{}", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    return argparse.Namespace(
        capture_plan=plan_path,
        dataset_manifest=manifest_path,
        dataset_root=dataset_root,
        teacher_plan=teacher_plan,
        output=tmp_path / "capture_output",
        limit=None,
    ), records


def _install_fakes(monkeypatch, calls, *, failed_days=()):
    import scripts.dev.capture_ok_leak_driver_batch as batch

    monkeypatch.setattr(
        batch,
        "verify_bounded_capture_plan",
        lambda *_: {"passed": True, "checks": []},
    )
    monkeypatch.setattr(
        batch,
        "load_plan",
        lambda *_args, **_kwargs: SimpleNamespace(
            plan_sha256="teacher-plan-hash",
            entries=(
                SimpleNamespace(landpoint_id="001.0-071.0", year=1961),
            ),
        ),
    )

    def fake_probe(arguments):
        calls.append((arguments.landpoint_id, arguments.year, arguments.day_index))
        arguments.output.mkdir(parents=True, exist_ok=True)
        arrays_path = arguments.output / "ok_leak_driver_series.npz"
        arrays_path.write_bytes(f"capture-{arguments.day_index}".encode())
        passed = arguments.day_index not in failed_days
        report = {
            "passed": passed,
            "landpoint_id": arguments.landpoint_id,
            "year": arguments.year,
            "day_index": arguments.day_index,
            "day_start_state_sha256": arguments.expected_day_start_state_sha256,
            "fast_day_target_sha256": arguments.expected_fast_day_target_sha256,
            "capture_plan_sha256": arguments.capture_plan_sha256,
            "capture_npz_sha256": _sha256_file(arrays_path),
            "next_continuous_state": {
                "within_tolerance": arguments.day_index == 2,
            },
            "source_driver_comparisons": {"soil_mc": {"exact": True}},
            "exact_scan_replay": {"exact": True},
            "ok_leak_endpoint_within_1e-12": passed,
            "next_discrete_state": {"date": {"exact": True}},
        }
        _atomic_write_json(arguments.output / "report.json", report)
        return report

    monkeypatch.setattr(batch, "run_probe", fake_probe)


def test_batch_is_atomic_and_resumes_completed_days(tmp_path, monkeypatch):
    args, records = _arguments(tmp_path)
    calls = []
    _install_fakes(monkeypatch, calls)

    first = run_batch(args)

    assert first["status"] == "complete"
    assert first["completed_record_count"] == 2
    assert first["capture_interface_passed_count"] == 2
    assert first["next_state_diagnostic_passed_count"] == 1
    assert calls == [("001.0-071.0", 1961, 2), ("001.0-071.0", 1961, 3)]
    for record in records:
        output = _capture_directory(args.output.resolve(), record)
        assert output.is_dir()
        assert not output.with_name(output.name + ".incomplete").exists()

    calls.clear()
    second = run_batch(args)
    assert second["status"] == "complete"
    assert second["next_state_diagnostic_passed_count"] == 1
    assert calls == []
    assert all(item["resumed"] for item in second["records"])


def test_batch_promotes_complete_staging_directory_without_rerun(tmp_path, monkeypatch):
    args, records = _arguments(tmp_path)
    args.limit = 1
    calls = []
    _install_fakes(monkeypatch, calls)
    run_batch(args)
    final = _capture_directory(args.output.resolve(), records[0])
    staging = final.with_name(final.name + ".incomplete")
    final.replace(staging)

    calls.clear()
    result = run_batch(args)

    assert result["status"] == "bounded_smoke_complete"
    assert calls == []
    assert final.is_dir()
    assert not staging.exists()


def test_batch_collects_scientific_failures_and_attempts_remaining_days(
    tmp_path,
    monkeypatch,
):
    args, records = _arguments(tmp_path)
    calls = []
    _install_fakes(monkeypatch, calls, failed_days={2})

    result = run_batch(args)

    assert calls == [("001.0-071.0", 1961, 2), ("001.0-071.0", 1961, 3)]
    assert result["status"] == "capture_interface_failed"
    assert result["attempted_record_count"] == 2
    assert result["capture_interface_passed_count"] == 1
    assert result["capture_interface_failed_count"] == 1
    assert result["failed_records"][0]["failure_reasons"] == [
        "ok_leak_endpoint_within_1e-12"
    ]
    failed = _capture_directory(args.output.resolve(), records[0])
    assert failed.with_name(failed.name + ".incomplete").is_dir()
    assert _capture_directory(args.output.resolve(), records[1]).is_dir()
