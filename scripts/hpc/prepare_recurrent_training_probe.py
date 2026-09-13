"""Select bounded recurrent TRAIN windows from existing prepared metadata only."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any, Mapping

ROLE = "train_only_bounded_recurrent_admission_not_validation"
SEED = 20260914
WINDOWS_PER_HORIZON = 8
EXPECTED_MONITOR_DAYS = 64
EXPECTED_DEVELOPMENT_WINDOW_DAYS = 296


def _day_key(value: Any) -> tuple[str, int, int]:
    if not isinstance(value, Mapping):
        raise TypeError("day metadata must be a mapping")
    landpoint_id = value.get("landpoint_id")
    year = value.get("year")
    start_row = value.get("start_row")
    if not isinstance(landpoint_id, str) or not landpoint_id:
        raise ValueError("day metadata lacks landpoint_id")
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError("day metadata lacks integer year")
    if isinstance(start_row, bool) or not isinstance(start_row, int) or start_row < 0:
        raise ValueError("day metadata lacks nonnegative integer start_row")
    return landpoint_id, year, start_row


def recurrent_exclusions(
    catalogue: Mapping[str, Any], development_selection: Mapping[str, Any]
) -> tuple[set[tuple[str, int, int]], set[tuple[str, int, int]], set[tuple[str, int, int]]]:
    """Return monitor-only, development-window, and combined exclusions.

    Expanded one-day TRAIN rows are intentionally not read here: the recurrent
    probe is a bounded train-only admission, so overlap with them is allowed.
    """

    from scripts.hpc.trained_screen_common import validate_selection

    monitor_rows = catalogue.get("monitor_rows")
    if not isinstance(monitor_rows, list):
        raise ValueError("expanded catalogue lacks monitor_rows")
    monitor_days = {_day_key(value) for value in monitor_rows}
    validate_selection(development_selection)
    development_days = {
        _day_key({"landpoint_id": key[0], "year": key[1], "start_row": key[2]})
        for window in development_selection["windows"]
        for key in window["teacher_day_keys"]
    }
    return monitor_days, development_days, monitor_days | development_days


def _source_manifest_path(development_selection_path: Path, reference: Any) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError("development selection lacks source_manifest")
    path = Path(reference)
    if not path.is_absolute():
        path = development_selection_path.parent / path
    return path.resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.forward_screen_selection import select_windows
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        load_target_oriented_training_contract,
    )
    from research.daily_coarse_graining.typed_sidecar import load_coherent_typed_sidecar_index
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    catalogue_path = args.catalogue.resolve()
    development_path = args.development_selection.resolve()
    with gzip.open(catalogue_path, "rt", encoding="utf-8") as handle:
        catalogue = json.load(handle)
    development_selection = json.loads(development_path.read_text(encoding="utf-8"))
    if development_selection.get("status") != "passed":
        raise ValueError("development selection is not accepted")

    monitor_days, development_days, excluded_days = recurrent_exclusions(
        catalogue, development_selection
    )
    if len(monitor_days) != EXPECTED_MONITOR_DAYS:
        raise ValueError("expanded catalogue monitor-day count differs from the declared 64")
    if len(development_days) != EXPECTED_DEVELOPMENT_WINDOW_DAYS:
        raise ValueError("development selection day count differs from the declared 296")

    source_path = _source_manifest_path(
        development_path, development_selection.get("source_manifest")
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_identity = development_selection.get("source_identity")
    if not isinstance(source_identity, dict) or source.get("identity") != source_identity:
        raise ValueError("development selection and prepared source identities differ")

    index = load_coherent_typed_sidecar_index(args.dataset_manifest, verify_hashes=False)
    schedule = build_target_oriented_training_schedule(
        index.select(spatial_split="train", temporal_split="train"),
        load_target_oriented_training_contract(),
        updates=source["updates"],
    )
    if schedule["schedule_sha256"] != source_identity.get("schedule_sha256"):
        raise ValueError("prepared schedule identity differs from development source metadata")
    source_updates = {int(record["update"]) for record in source["entries"]}
    entries = [entry for entry in schedule["entries"] if int(entry["update"]) in source_updates]
    if len(entries) != len(source_updates):
        raise ValueError("source prepared artifacts are absent from the reconstructed schedule")

    windows = select_windows(
        entries,
        excluded_days,
        seed=SEED,
        per_horizon=WINDOWS_PER_HORIZON,
    )
    selected_days = {tuple(key) for window in windows for key in window["teacher_day_keys"]}
    if selected_days & excluded_days:
        raise RuntimeError("recurrent selection overlaps its declared exclusions")

    report = {
        "status": "passed",
        "schema_version": "recurrent_training_probe_selection_v1",
        "role": ROLE,
        "seed": SEED,
        "windows_per_horizon": WINDOWS_PER_HORIZON,
        "windows": windows,
        "source_manifest": str(source_path),
        "source_identity": source_identity,
        "candidate_source": {
            "kind": "reconstructed_exact_train_schedule_metadata",
            "dataset_manifest": str(args.dataset_manifest.resolve()),
            "schedule_sha256": schedule["schedule_sha256"],
            "prepared_artifact_count": len(entries),
            "prepared_artifact_updates": sorted(source_updates),
        },
        "exclusions": {
            "expanded_catalogue": str(catalogue_path),
            "monitor_day_count": len(monitor_days),
            "development_selection": str(development_path),
            "development_window_day_count": len(development_days),
            "combined_day_count": len(excluded_days),
            "expanded_one_day_train_rows_excluded": False,
            "expanded_one_day_train_overlap_allowed": True,
        },
        "spatial_temporal_split": "train/train",
        "teacher_payload_reads": 0,
        "teacher_regenerated": False,
        "sealed_test_accessed": False,
        "parameter_updates": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("windows", "source_identity")},
            indent=2,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalogue", "development-selection", "dataset-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
