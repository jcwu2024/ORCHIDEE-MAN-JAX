"""Prepared artifacts for the fixed target-oriented milestone panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    ordered_target_oriented_validation_entries,
)
from scripts.hpc.successor_rollout_screen_prepared import load_prepared_rollout_artifact

PREPARED_SCHEMA = "target_oriented_validation_prepared_v1"


def load_target_oriented_validation_prepared(
    path: str | Path,
    *,
    training_contract_sha256: str,
    target_contract_sha256: str,
    verify_artifacts: bool = True,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    panel_path = (source.parent / str(payload.get("validation_panel", ""))).resolve()
    if not panel_path.is_file():
        raise ValueError("target-oriented validation panel is missing")
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    ordered = ordered_target_oriented_validation_entries(panel)
    if (
        payload.get("schema_version") != PREPARED_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("training_contract_sha256") != training_contract_sha256
        or payload.get("target_contract_sha256") != target_contract_sha256
        or payload.get("validation_panel_sha256") != panel.get("panel_sha256")
        or payload.get("validation_panel_file_sha256") != sha256_file(panel_path)
        or int(payload.get("artifact_count", -1)) != len(ordered)
        or len(payload.get("entries", ())) != len(ordered)
        or payload.get("sealed_test_used") is not False
        or payload.get("teacher_data_regenerated") is not False
        or payload.get("historical_v5_used") is not False
    ):
        raise ValueError("target-oriented validation-prepared identity drift")
    for record, (slice_id, horizon, entry) in zip(payload["entries"], ordered, strict=True):
        if (
            record.get("slice_id") != slice_id
            or int(record.get("horizon_days", -1)) != horizon
            or record.get("window_entry") != entry
        ):
            raise ValueError("target-oriented validation-prepared order drift")
        if verify_artifacts:
            load_prepared_rollout_artifact(
                source,
                record,
                slice_id=slice_id,
                window_entry=entry,
            )
    return payload | {"manifest_sha256": sha256_file(source)}, panel


__all__ = ["PREPARED_SCHEMA", "load_target_oriented_validation_prepared"]
