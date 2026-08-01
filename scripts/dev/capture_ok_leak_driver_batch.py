"""Capture a frozen bounded OK_LEAK driver plan in one restartable process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from research.daily_coarse_graining.ok_leak_capture_selection import (
    verify_bounded_capture_plan,
)
from research.daily_coarse_graining.teacher_shards import load_plan
from scripts.dev.probe_ok_leak_driver_capture import (
    _atomic_write_json,
    _sha256_file,
    run_probe,
)


def _capture_directory(root: Path, record: Mapping[str, Any]) -> Path:
    return (
        root
        / "captures"
        / str(record["landpoint_id"])
        / f"{int(record['year']):04d}_day_{int(record['day_index']):03d}"
    )


def _existing_capture_summary(
    output: Path,
    record: Mapping[str, Any],
    *,
    capture_plan_sha256: str,
) -> Mapping[str, Any] | None:
    report_path = output / "report.json"
    arrays_path = output / "ok_leak_driver_series.npz"
    if not report_path.exists() and not arrays_path.exists():
        return None
    if not report_path.exists() or not arrays_path.exists():
        raise ValueError(f"partial capture output requires inspection: {output}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "landpoint_id": record["landpoint_id"],
        "year": int(record["year"]),
        "day_index": int(record["day_index"]),
        "day_start_state_sha256": record["day_start_state_sha256"],
        "fast_day_target_sha256": record["fast_day_target_sha256"],
        "capture_plan_sha256": capture_plan_sha256,
    }
    drift = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    observed_arrays_hash = _sha256_file(arrays_path)
    if report.get("capture_npz_sha256") != observed_arrays_hash:
        drift["capture_npz_sha256"] = (
            report.get("capture_npz_sha256"),
            observed_arrays_hash,
        )
    if not report.get("passed"):
        drift["passed"] = (report.get("passed"), True)
    if drift:
        raise ValueError(f"existing capture identity/status drift at {output}: {drift}")
    return {
        "landpoint_id": record["landpoint_id"],
        "year": int(record["year"]),
        "day_index": int(record["day_index"]),
        "report": str(report_path.relative_to(output.parents[2])),
        "capture_npz_sha256": observed_arrays_hash,
        "resumed": True,
    }


def run_batch(args: argparse.Namespace) -> Mapping[str, Any]:
    capture_plan_path = args.capture_plan.resolve()
    dataset_manifest_path = args.dataset_manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    teacher_plan_path = args.teacher_plan.resolve()
    output = args.output.resolve()
    capture_plan = json.loads(capture_plan_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    generation_plan = load_plan(teacher_plan_path, require_inputs=True)
    if generation_plan.plan_sha256 != dataset_manifest.get("plan_sha256"):
        raise ValueError("Teacher generation plan hash does not match the dataset manifest")
    generation_entries = {
        (item.landpoint_id, int(item.year)) for item in generation_plan.entries
    }
    missing_generation_entries = [
        (item["landpoint_id"], int(item["year"]))
        for item in capture_plan["records"]
        if (item["landpoint_id"], int(item["year"])) not in generation_entries
    ]
    if missing_generation_entries:
        raise ValueError(
            f"Teacher generation plan is missing capture records: {missing_generation_entries[:8]}"
        )
    verification = verify_bounded_capture_plan(
        capture_plan_path,
        dataset_manifest_path,
        dataset_root,
    )
    if not verification["passed"]:
        raise ValueError("capture plan verification failed")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output / "plan_verification.json", verification)

    all_records = tuple(capture_plan["records"])
    limit = getattr(args, "limit", None)
    records = all_records if limit is None else all_records[: int(limit)]
    if not records:
        raise ValueError("batch capture has no requested records")
    summaries = []
    capture_plan_sha256 = capture_plan["plan_sha256"]
    for record in records:
        day_output = _capture_directory(output, record)
        existing = _existing_capture_summary(
            day_output,
            record,
            capture_plan_sha256=capture_plan_sha256,
        )
        if existing is not None:
            summaries.append(existing)
            continue
        staging_output = day_output.with_name(day_output.name + ".incomplete")
        staged_summary = None
        if (staging_output / "report.json").exists() and (
            staging_output / "ok_leak_driver_series.npz"
        ).exists():
            try:
                staged_summary = _existing_capture_summary(
                    staging_output,
                    record,
                    capture_plan_sha256=capture_plan_sha256,
                )
            except ValueError:
                staged_summary = None
        if staged_summary is not None:
            day_output.parent.mkdir(parents=True, exist_ok=True)
            staging_output.replace(day_output)
            promoted = _existing_capture_summary(
                day_output,
                record,
                capture_plan_sha256=capture_plan_sha256,
            )
            summaries.append(dict(promoted) | {"resumed": True})
            continue
        probe_args = SimpleNamespace(
            dataset_manifest=dataset_manifest_path,
            plan=teacher_plan_path,
            landpoint_id=record["landpoint_id"],
            year=int(record["year"]),
            day_index=int(record["day_index"]),
            output=staging_output,
            capture_plan_sha256=capture_plan_sha256,
            expected_day_start_state_sha256=record["day_start_state_sha256"],
            expected_fast_day_target_sha256=record["fast_day_target_sha256"],
        )
        report = run_probe(probe_args)
        if not report["passed"]:
            raise ValueError(f"capture scientific gate failed: {staging_output}")
        _existing_capture_summary(
            staging_output,
            record,
            capture_plan_sha256=capture_plan_sha256,
        )
        day_output.parent.mkdir(parents=True, exist_ok=True)
        staging_output.replace(day_output)
        summary = _existing_capture_summary(
            day_output,
            record,
            capture_plan_sha256=capture_plan_sha256,
        )
        summaries.append(dict(summary) | {"resumed": False})
        _atomic_write_json(
            output / "capture_manifest.json",
            {
                "schema_version": "ok_leak_auxiliary_capture_manifest_v1",
                "status": "partial",
                "capture_plan": str(capture_plan_path),
                "capture_plan_sha256": capture_plan_sha256,
                "dataset_id": capture_plan["dataset_id"],
                "generation_plan_sha256": generation_plan.plan_sha256,
                "sealed_test_used": False,
                "requested_record_count": len(records),
                "full_plan_record_count": len(all_records),
                "completed_record_count": len(summaries),
                "records": summaries,
            },
        )

    manifest = {
        "schema_version": "ok_leak_auxiliary_capture_manifest_v1",
        "status": "complete" if len(records) == len(all_records) else "bounded_smoke_complete",
        "capture_plan": str(capture_plan_path),
        "capture_plan_sha256": capture_plan_sha256,
        "dataset_id": capture_plan["dataset_id"],
        "generation_plan_sha256": generation_plan.plan_sha256,
        "sealed_test_used": False,
        "requested_record_count": len(records),
        "full_plan_record_count": len(all_records),
        "completed_record_count": len(summaries),
        "records": summaries,
    }
    _atomic_write_json(output / "capture_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    manifest = run_batch(args)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
