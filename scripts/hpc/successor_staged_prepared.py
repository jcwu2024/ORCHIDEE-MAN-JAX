"""Prepared-artifact admission for incremental successor training stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.hpc.successor_small_fit_prepared import (
    load_prepared_artifact,
    sha256_file,
)

STAGED_PREPARED_MANIFEST_SCHEMA = "successor_staged_prepared_manifest_v1"


def stage_update_range(milestones: Sequence[int], milestone: int) -> tuple[int, int]:
    ordered = tuple(int(value) for value in milestones)
    wanted = int(milestone)
    if not ordered or tuple(sorted(set(ordered))) != ordered or wanted not in ordered:
        raise ValueError("successor staged milestone registry drift")
    position = ordered.index(wanted)
    return (1 if position == 0 else ordered[position - 1] + 1, wanted)


def staged_worker_updates(
    start_update: int,
    end_update: int,
    worker_index: int,
    worker_count: int,
) -> tuple[int, ...]:
    if not 1 <= int(start_update) <= int(end_update):
        raise ValueError("successor staged update range is invalid")
    if int(worker_count) < 1 or not 0 <= int(worker_index) < int(worker_count):
        raise ValueError("successor staged worker identity is invalid")
    return tuple(
        range(
            int(start_update) + int(worker_index),
            int(end_update) + 1,
            int(worker_count),
        )
    )


def load_staged_prepared_manifest(
    path: str | Path,
    *,
    staged_training_contract_sha256: str,
    full_schedule_sha256: str,
    start_update: int,
    end_update: int,
) -> Mapping[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    identity = payload.get("identity", {})
    expected_updates = list(range(int(start_update), int(end_update) + 1))
    entries = payload.get("entries", ())
    if (
        payload.get("schema_version") != STAGED_PREPARED_MANIFEST_SCHEMA
        or payload.get("status") != "passed"
        or identity.get("staged_training_contract_sha256")
        != staged_training_contract_sha256
        or identity.get("full_schedule_sha256") != full_schedule_sha256
        or int(payload.get("start_update", 0)) != int(start_update)
        or int(payload.get("end_update", 0)) != int(end_update)
        or int(payload.get("artifact_count", -1)) != len(expected_updates)
        or [int(record.get("update", -1)) for record in entries]
        != expected_updates
        or payload.get("teacher_data_regenerated") is not False
        or payload.get("historical_v5_used") is not False
        or payload.get("sealed_test_used") is not False
    ):
        raise ValueError("successor staged prepared-manifest identity drift")
    if int(payload.get("total_size_bytes", -1)) != sum(
        int(record["size_bytes"]) for record in entries
    ):
        raise ValueError("successor staged prepared-manifest size drift")
    return payload | {"manifest_sha256": sha256_file(source)}


def admit_staged_prepared_artifacts(
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    full_schedule: Mapping[str, Any],
) -> None:
    source = Path(manifest_path).resolve()
    for record in manifest["entries"]:
        update = int(record["update"])
        load_prepared_artifact(
            source,
            record,
            schedule_entry=full_schedule["entries"][update - 1],
        )
