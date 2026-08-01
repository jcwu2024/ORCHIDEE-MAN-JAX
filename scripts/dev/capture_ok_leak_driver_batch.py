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
from scripts.dev.probe_ok_leak_driver_capture import _atomic_write_json, _sha256_file
from scripts.dev.probe_ok_leak_outer_block_reentry import (
    run_probe,
    teacher_block_for_day,
)


def _capture_directory(root: Path, record: Mapping[str, Any]) -> Path:
    return (
        root
        / "captures"
        / str(record["landpoint_id"])
        / f"{int(record['year']):04d}_day_{int(record['day_index']):03d}"
    )


def _next_state_diagnostic_passed(report: Mapping[str, Any]) -> bool:
    comparison = report.get("next_continuous_state", {})
    if "within_tolerance" in comparison:
        return bool(comparison["within_tolerance"])
    return bool(
        comparison.get("shape_equal")
        and comparison.get("defined_status_mismatches", 0) == 0
        and comparison.get("max_absolute_error") is not None
        and comparison["max_absolute_error"] <= 1.0e-12
    )


def _capture_interface_passed(report: Mapping[str, Any]) -> bool:
    value = report.get("capture_interface_passed")
    return bool(report.get("passed")) if value is None else bool(value)


def _capture_interface_failure_reasons(report: Mapping[str, Any]) -> list[str]:
    if report.get("capture_interface_passed") is not None:
        if _capture_interface_passed(report):
            return []
        reasons = []
        if report.get("capture") is None:
            reasons.append("compiled_driver_capture")
        if not report.get(
            "target_ok_leak_endpoint_within_1e-12",
            report.get("ok_leak_endpoint_within_1e-12"),
        ):
            reasons.append("target_ok_leak_endpoint_within_1e-12")
        discrete = report.get("next_discrete_state", {})
        if not discrete or not all(item.get("exact") for item in discrete.values()):
            reasons.append("next_discrete_state_exact")
        return reasons or ["outer_block_capture_interface_gate"]

    reasons = []
    drivers = report.get("source_driver_comparisons", {})
    if not drivers or not all(item.get("exact") for item in drivers.values()):
        reasons.append("source_driver_exact")
    if not report.get("exact_scan_replay", {}).get("exact"):
        reasons.append("exact_scan_replay")
    if not report.get("ok_leak_endpoint_within_1e-12"):
        reasons.append("ok_leak_endpoint_within_1e-12")
    discrete = report.get("next_discrete_state", {})
    if not discrete or not all(item.get("exact") for item in discrete.values()):
        reasons.append("next_discrete_state_exact")
    return reasons or ["unspecified_capture_interface_gate"]


def _capture_failure_summary(
    output: Path,
    record: Mapping[str, Any],
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    report_path = output / "report.json"
    arrays_path = output / "ok_leak_driver_series.npz"
    return {
        "landpoint_id": record["landpoint_id"],
        "year": int(record["year"]),
        "day_index": int(record["day_index"]),
        "report": str(report_path.relative_to(output.parents[2])),
        "capture_npz_sha256": _sha256_file(arrays_path),
        "failure_reasons": _capture_interface_failure_reasons(report),
        "next_state_diagnostic_passed": _next_state_diagnostic_passed(report),
    }


def _batch_manifest(
    *,
    status: str,
    capture_plan_path: Path,
    capture_plan_sha256: str,
    dataset_id: str,
    generation_plan_sha256: str,
    requested_record_count: int,
    full_plan_record_count: int,
    summaries: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "schema_version": "ok_leak_auxiliary_capture_manifest_v2",
        "status": status,
        "capture_plan": str(capture_plan_path),
        "capture_plan_sha256": capture_plan_sha256,
        "dataset_id": dataset_id,
        "generation_plan_sha256": generation_plan_sha256,
        "sealed_test_used": False,
        "requested_record_count": requested_record_count,
        "full_plan_record_count": full_plan_record_count,
        "attempted_record_count": len(summaries) + len(failures),
        "completed_record_count": len(summaries),
        "capture_interface_passed_count": len(summaries),
        "capture_interface_failed_count": len(failures),
        "next_state_diagnostic_passed_count": sum(
            bool(item["next_state_diagnostic_passed"]) for item in summaries
        ),
        "records": list(summaries),
        "failed_records": list(failures),
    }


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
        "schema_version": "ok_leak_outer_block_reentry_probe_v2",
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
    if not _capture_interface_passed(report):
        drift["capture_interface_passed"] = (
            report.get("capture_interface_passed"),
            True,
        )
    if drift:
        raise ValueError(f"existing capture identity/status drift at {output}: {drift}")
    return {
        "landpoint_id": record["landpoint_id"],
        "year": int(record["year"]),
        "day_index": int(record["day_index"]),
        "report": str(report_path.relative_to(output.parents[2])),
        "capture_npz_sha256": observed_arrays_hash,
        "next_state_diagnostic_passed": _next_state_diagnostic_passed(report),
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
        (item.landpoint_id, int(item.year)): item
        for item in generation_plan.entries
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
    failures = []
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
        plan_entry = generation_entries[
            (record["landpoint_id"], int(record["year"]))
        ]
        block_start_day, block_days, _target_offset = teacher_block_for_day(
            int(record["day_index"]),
            block_size=generation_plan.block_size,
            days_in_year=plan_entry.days,
        )
        probe_args = SimpleNamespace(
            dataset_manifest=dataset_manifest_path,
            plan=teacher_plan_path,
            landpoint_id=record["landpoint_id"],
            year=int(record["year"]),
            day_index=int(record["day_index"]),
            block_start_day=block_start_day,
            block_days=block_days,
            doc_sqrt_mode="production",
            output=staging_output / "report.json",
            capture_npz=staging_output / "ok_leak_driver_series.npz",
            capture_plan_sha256=capture_plan_sha256,
            expected_day_start_state_sha256=record["day_start_state_sha256"],
            expected_fast_day_target_sha256=record["fast_day_target_sha256"],
        )
        report = run_probe(probe_args)
        if not _capture_interface_passed(report):
            failures.append(_capture_failure_summary(staging_output, record, report))
            _atomic_write_json(
                output / "capture_manifest.json",
                _batch_manifest(
                    status="partial_with_capture_interface_failures",
                    capture_plan_path=capture_plan_path,
                    capture_plan_sha256=capture_plan_sha256,
                    dataset_id=capture_plan["dataset_id"],
                    generation_plan_sha256=generation_plan.plan_sha256,
                    requested_record_count=len(records),
                    full_plan_record_count=len(all_records),
                    summaries=summaries,
                    failures=failures,
                ),
            )
            continue
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
            _batch_manifest(
                status=(
                    "partial_with_capture_interface_failures"
                    if failures
                    else "partial"
                ),
                capture_plan_path=capture_plan_path,
                capture_plan_sha256=capture_plan_sha256,
                dataset_id=capture_plan["dataset_id"],
                generation_plan_sha256=generation_plan.plan_sha256,
                requested_record_count=len(records),
                full_plan_record_count=len(all_records),
                summaries=summaries,
                failures=failures,
            ),
        )

    if failures:
        status = "capture_interface_failed"
    elif len(records) == len(all_records):
        status = "complete"
    else:
        status = "bounded_smoke_complete"
    manifest = _batch_manifest(
        status=status,
        capture_plan_path=capture_plan_path,
        capture_plan_sha256=capture_plan_sha256,
        dataset_id=capture_plan["dataset_id"],
        generation_plan_sha256=generation_plan.plan_sha256,
        requested_record_count=len(records),
        full_plan_record_count=len(all_records),
        summaries=summaries,
        failures=failures,
    )
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
    return 1 if manifest["capture_interface_failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
