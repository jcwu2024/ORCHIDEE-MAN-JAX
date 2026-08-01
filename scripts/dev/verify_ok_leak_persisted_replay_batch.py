"""Verify every hash-bound persisted driver capture through the current scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from research.daily_coarse_graining.ok_leak_driver_dataset import (
    load_ok_leak_driver_dataset,
)
from scripts.dev.probe_ok_leak_driver_capture import (
    _atomic_write_json,
    _sha256_file,
    run_probe,
)

MANIFEST_SCHEMA_VERSION = "ok_leak_persisted_replay_manifest_v1"


def _key(sample) -> tuple[str, int, int]:
    return sample.landpoint_id, int(sample.year), int(sample.day_index)


def _output_directory(root: Path, sample) -> Path:
    return (
        root
        / "replays"
        / sample.landpoint_id
        / f"{int(sample.year):04d}_day_{int(sample.day_index):03d}"
    )


def _driver_npz(sample) -> Path:
    capture_report = json.loads(sample.report_path.read_text(encoding="utf-8"))
    path = (sample.report_path.parent / capture_report["capture_npz"]).resolve()
    if _sha256_file(path) != sample.capture_npz_sha256:
        raise ValueError(f"persisted replay source hash drift: {_key(sample)}")
    return path


def _summary(output: Path, sample) -> Mapping[str, Any] | None:
    report_path = output / "report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "ok_leak_driver_capture_probe_v2",
        "landpoint_id": sample.landpoint_id,
        "year": int(sample.year),
        "day_index": int(sample.day_index),
        "persisted_driver_npz_sha256": sample.capture_npz_sha256,
        "persisted_ok_leak_endpoint_within_1e-12": True,
        "sealed_test_used": False,
    }
    drift = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    if drift:
        raise ValueError(f"persisted replay report drift at {output}: {drift}")
    comparisons = report.get("persisted_ok_leak_endpoint_comparisons") or {}
    if len(comparisons) != 14:
        raise ValueError(f"persisted replay endpoint coverage drift at {output}")
    return {
        "landpoint_id": sample.landpoint_id,
        "year": int(sample.year),
        "day_index": int(sample.day_index),
        "report": str(report_path.relative_to(output.parents[2])),
        "persisted_driver_npz_sha256": sample.capture_npz_sha256,
        "max_absolute_error": max(
            float(item["max_absolute_error"]) for item in comparisons.values()
        ),
        "exact_endpoint_count": sum(
            bool(item["exact"]) for item in comparisons.values()
        ),
        "endpoint_count": len(comparisons),
        "resumed": True,
    }


def _manifest(
    *,
    status: str,
    dataset,
    capture_root: Path,
    evaluation_git_head: str,
    requested_count: int,
    total_count: int,
    summaries: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": status,
        "dataset_id": dataset.dataset_id,
        "teacher_git_head": dataset.teacher_git_head,
        "markov_contract_sha256": dataset.contract_sha256,
        "capture_plan_sha256": dataset.capture_plan_sha256,
        "capture_root": str(capture_root),
        "evaluation_git_head": evaluation_git_head,
        "sealed_test_used": False,
        "requested_record_count": requested_count,
        "full_record_count": total_count,
        "completed_record_count": len(summaries),
        "failed_record_count": len(failures),
        "records": list(summaries),
        "failed_records": list(failures),
    }


def run_batch(args: argparse.Namespace) -> Mapping[str, Any]:
    capture_root = args.capture_root.resolve()
    output = args.output.resolve()
    dataset = load_ok_leak_driver_dataset(
        capture_root,
        args.capture_plan.resolve(),
        args.dataset_manifest.resolve(),
    )
    all_samples = dataset.samples
    samples = (
        all_samples
        if args.limit is None
        else all_samples[: int(args.limit)]
    )
    if not samples:
        raise ValueError("persisted replay batch has no requested records")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = []
    for sample in samples:
        final = _output_directory(output, sample)
        existing = _summary(final, sample)
        if existing is not None:
            summaries.append(existing)
            continue
        staging = final.with_name(final.name + ".incomplete")
        try:
            staged = _summary(staging, sample)
        except ValueError:
            staged = None
        if staged is not None:
            final.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(final)
            summaries.append(dict(_summary(final, sample)) | {"resumed": True})
            continue
        replay_npz = _driver_npz(sample)
        report = run_probe(
            SimpleNamespace(
                dataset_manifest=args.dataset_manifest.resolve(),
                plan=args.teacher_plan.resolve(),
                landpoint_id=sample.landpoint_id,
                year=int(sample.year),
                day_index=int(sample.day_index),
                output=staging,
                replay_driver_npz=replay_npz,
                expected_day_start_state_sha256=None,
                expected_fast_day_target_sha256=None,
                capture_plan_sha256=dataset.capture_plan_sha256,
            )
        )
        if not report.get("persisted_ok_leak_endpoint_within_1e-12"):
            failures.append(
                {
                    "landpoint_id": sample.landpoint_id,
                    "year": int(sample.year),
                    "day_index": int(sample.day_index),
                    "report": str(
                        (staging / "report.json").relative_to(output)
                    ),
                    "persisted_driver_npz_sha256": sample.capture_npz_sha256,
                }
            )
        else:
            final.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(final)
            summaries.append(dict(_summary(final, sample)) | {"resumed": False})
        _atomic_write_json(
            output / "replay_manifest.json",
            _manifest(
                status="partial_with_failures" if failures else "partial",
                dataset=dataset,
                capture_root=capture_root,
                evaluation_git_head=args.expected_git_head,
                requested_count=len(samples),
                total_count=len(all_samples),
                summaries=summaries,
                failures=failures,
            ),
        )
    status = (
        "failed"
        if failures
        else "complete" if len(samples) == len(all_samples) else "bounded_smoke_complete"
    )
    manifest = _manifest(
        status=status,
        dataset=dataset,
        capture_root=capture_root,
        evaluation_git_head=args.expected_git_head,
        requested_count=len(samples),
        total_count=len(all_samples),
        summaries=summaries,
        failures=failures,
    )
    _atomic_write_json(output / "replay_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--teacher-plan", type=Path, required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    manifest = run_batch(args)
    print(
        json.dumps(
            {name: value for name, value in manifest.items() if name != "records"},
            indent=2,
        )
    )
    return 1 if manifest["failed_record_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
