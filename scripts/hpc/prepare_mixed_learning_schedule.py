"""Prepare a deterministic metadata-only recurrent TRAIN schedule for mixed learning."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from scripts.hpc.prepare_recurrent_training_probe import (
    EXPECTED_DEVELOPMENT_WINDOW_DAYS,
    EXPECTED_MONITOR_DAYS,
    _source_manifest_path,
    recurrent_exclusions,
)

ROLE = "train_only_mixed_learning_schedule"
SEED = 20260914
_COMPONENTS = (("seven_day", 7, "T7"), ("thirty_day", 30, "T30"))


def _window_identity(row: Mapping[str, Any]) -> tuple[Any, str, int]:
    return int(row["artifact_id"]), str(row["component"]), int(row["window_index"])


def _semantic_window_identity(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["landpoint_id"]),
        int(row["year"]),
        int(row["start_row"]),
        int(row["length"]),
    )


def _day_keys(row: Mapping[str, Any]) -> set[tuple[str, int, int]]:
    return {tuple(key) for key in row["teacher_day_keys"]}


def _candidate_rows(
    entries: Iterable[Any], *, component: str, length: int
) -> list[dict[str, Any]]:
    """Reuse the development selector's canonical candidate/window-index semantics."""

    from research.daily_coarse_graining.daily_operator.forward_screen_selection import (
        _artifact_id,
        _component_candidates,
    )

    rows = []
    for entry in entries:
        artifact_id = _artifact_id(entry)
        rows.extend(candidate.as_metadata() for candidate in _component_candidates(
            entry, artifact_id, component, length
        ))
    identities = [_window_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate {component} window identity in reconstructed schedule")
    return sorted(rows, key=_window_identity)


def _deduplicate_semantic_windows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first canonical artifact/window reference for each Teacher trajectory."""

    unique, seen, aliases = [], set(), 0
    for row in rows:
        identity = _semantic_window_identity(row)
        if identity in seen:
            aliases += 1
            continue
        seen.add(identity)
        unique.append(row)
    return unique, aliases


def select_mixed_learning_windows(
    entries: Iterable[Any],
    excluded_days: Iterable[tuple[str, int, int]],
    *,
    updates: int = 200,
    seed: int = SEED,
) -> dict[str, Any]:
    """Select one distinct T7 and T30 metadata row per update without payload reads."""

    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    schedule_entries = tuple(entries)
    if not schedule_entries:
        raise ValueError("mixed-learning schedule has no reconstructed entries")
    exclusions = {tuple(key) for key in excluded_days}
    generator = np.random.Generator(np.random.PCG64(seed))
    candidates: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for component, length, _horizon in _COMPONENTS:
        raw = _candidate_rows(schedule_entries, component=component, length=length)
        eligible = [row for row in raw if not (_day_keys(row) & exclusions)]
        eligible_unique, aliases = _deduplicate_semantic_windows(eligible)
        order = generator.permutation(len(eligible_unique))
        candidates[component] = [eligible_unique[int(index)] for index in order]
        counts[component] = {
            "raw_candidate_count": len(raw),
            "excluded_candidate_count": len(raw) - len(eligible),
            "eligible_candidate_count": len(eligible),
            "eligible_alias_count_removed": aliases,
            "eligible_unique_semantic_candidate_count": len(eligible_unique),
        }

    available_t30 = list(candidates["thirty_day"])
    update_rows, flat_rows = [], []
    t7_cursor = 0
    while len(update_rows) < updates and t7_cursor < len(candidates["seven_day"]):
        t7 = candidates["seven_day"][t7_cursor]
        t7_cursor += 1
        t7_days = _day_keys(t7)
        compatible = next(
            (index for index, t30 in enumerate(available_t30) if t7_days.isdisjoint(_day_keys(t30))),
            None,
        )
        if compatible is None:
            continue
        t30 = available_t30.pop(compatible)
        update = len(update_rows) + 1
        if _window_identity(t7) == _window_identity(t30):
            raise RuntimeError("mixed-learning horizons unexpectedly share a window identity")
        update_rows.append({"update": update, "T7": t7, "T30": t30})
        flat_rows.extend((
            {"update": update, "horizon": "T7", "window": t7},
            {"update": update, "horizon": "T30", "window": t30},
        ))

    if len(update_rows) != updates:
        raise ValueError(
            "metadata-only mixed-learning window shortage after exclusions and same-update "
            f"disjointness: required_updates={updates} selected_updates={len(update_rows)} "
            f"eligible_T7={counts['seven_day']['eligible_candidate_count']} "
            f"eligible_T30={counts['thirty_day']['eligible_candidate_count']}"
        )

    selected = [record["window"] for record in flat_rows]
    identities = [_window_identity(row) for row in selected]
    if len(identities) != len(set(identities)):
        raise RuntimeError("mixed-learning schedule repeated a window identity")
    semantic_identities = [_semantic_window_identity(row) for row in selected]
    if len(semantic_identities) != len(set(semantic_identities)):
        raise RuntimeError("mixed-learning schedule repeated a semantic window")
    if any(_day_keys(record["T7"]) & _day_keys(record["T30"]) for record in update_rows):
        raise RuntimeError("mixed-learning schedule has same-update semantic day overlap")
    day_occurrences = Counter(day for row in selected for day in _day_keys(row))
    for component, _length, horizon in _COMPONENTS:
        counts[component]["selected_window_count"] = sum(
            record["horizon"] == horizon for record in flat_rows
        )
    return {
        "update_rows": update_rows,
        "window_table": flat_rows,
        "candidate_counts": counts,
        "coverage": {
            "selected_window_count": len(selected),
            "selected_unique_window_identity_count": len(set(identities)),
            "selected_unique_semantic_window_count": len(set(semantic_identities)),
            "selected_unique_point_year_count": len({(row["landpoint_id"], row["year"]) for row in selected}),
            "selected_unique_site_count": len({row["landpoint_id"] for row in selected}),
            "selected_unique_year_count": len({row["year"] for row in selected}),
            "selected_recurrent_teacher_day_count": len(day_occurrences),
            "cross_update_teacher_day_overlap_count": sum(count - 1 for count in day_occurrences.values() if count > 1),
            "same_update_day_overlap_count": 0,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
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

    selected = select_mixed_learning_windows(entries, excluded_days, updates=args.updates)
    report = {
        "status": "passed",
        "schema_version": "mixed_learning_recurrent_schedule_v1",
        "role": ROLE,
        "not_validation": True,
        "seed": SEED,
        "updates": args.updates,
        "windows_per_update": {"T7": 1, "T30": 1},
        "update_rows": selected["update_rows"],
        "window_table": selected["window_table"],
        "source_manifest": str(source_path),
        "source_identity": source_identity,
        "candidate_source": {
            "kind": "reconstructed_exact_train_schedule_metadata",
            "dataset_manifest": str(args.dataset_manifest.resolve()),
            "schedule_sha256": schedule["schedule_sha256"],
            "prepared_artifact_count": len(entries),
            "prepared_artifact_updates": sorted(source_updates),
        },
        "candidate_counts": selected["candidate_counts"],
        "coverage": selected["coverage"],
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
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("update_rows", "window_table", "source_identity")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalogue", "development-selection", "dataset-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    run(parser.parse_args())
