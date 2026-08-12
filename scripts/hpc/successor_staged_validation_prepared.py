"""Prepared artifacts for the fixed successor milestone-validation panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from scripts.hpc.successor_rollout_screen_prepared import (
    load_prepared_rollout_artifact,
)

STAGED_VALIDATION_PREPARED_SCHEMA = "successor_staged_validation_prepared_v1"


def ordered_validation_entries(panel: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (slice_id, entry)
        for slice_id in ("temporal", "spatial", "joint")
        for entry in panel["slices"][slice_id]["entries"]
    ]


def load_staged_validation_prepared(
    manifest_path: str | Path,
    *,
    staged_training_contract_sha256: str,
    validation_panel_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = Path(manifest_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    panel_path = source.parent / str(payload.get("validation_panel", ""))
    if not panel_path.is_file():
        raise ValueError("successor staged validation panel is missing")
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    ordered = ordered_validation_entries(panel)
    if (
        payload.get("schema_version") != STAGED_VALIDATION_PREPARED_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("staged_training_contract_sha256")
        != staged_training_contract_sha256
        or payload.get("validation_panel_sha256") != validation_panel_sha256
        or panel.get("panel_sha256") != validation_panel_sha256
        or payload.get("validation_panel_file_sha256") != sha256_file(panel_path)
        or int(payload.get("artifact_count", -1)) != len(ordered)
        or len(payload.get("entries", ())) != len(ordered)
        or payload.get("sealed_test_used") is not False
        or payload.get("teacher_data_regenerated") is not False
        or payload.get("historical_v5_used") is not False
    ):
        raise ValueError("successor staged validation-prepared identity drift")
    for record, (slice_id, entry) in zip(payload["entries"], ordered, strict=True):
        if record.get("slice_id") != slice_id or record.get("window_entry") != entry:
            raise ValueError("successor staged validation-prepared order drift")
        load_prepared_rollout_artifact(
            source,
            record,
            slice_id=slice_id,
            window_entry=entry,
        )
    return payload | {"manifest_sha256": sha256_file(source)}, panel
