from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from scripts.hpc.successor_rollout_screen_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedRolloutScreenArtifact,
    atomic_json,
    load_prepared_rollout_artifact,
    load_prepared_rollout_manifest,
    write_prepared_rollout_artifact,
)


class _Index:
    def __init__(self, references):
        self.references = tuple(references)

    def select(self, *, spatial_split=None, temporal_split=None):
        return tuple(
            item
            for item in self.references
            if (spatial_split is None or item.parent.spatial_split == spatial_split)
            and (temporal_split is None or item.parent.temporal_split == temporal_split)
        )


def _reference(slice_id: str, ordinal: int, *, days: int = 365):
    splits = {
        "temporal": ("train", "validation"),
        "spatial": ("validation", "train"),
        "joint": ("validation", "validation"),
    }
    spatial, temporal = splits[slice_id]
    parent = SimpleNamespace(
        landpoint_id=f"{slice_id}-{ordinal:02d}",
        year=1961 + ordinal,
        sha256=f"parent-{slice_id}-{ordinal}",
        spatial_split=spatial,
        temporal_split=temporal,
        path=Path(f"{slice_id}-{ordinal}.npz"),
    )
    return SimpleNamespace(
        parent=parent,
        sha256=f"typed-{slice_id}-{ordinal}",
        day_count=days,
        path=Path(f"typed-{slice_id}-{ordinal}.npz"),
    )


































def test_prepared_artifact_roundtrip_and_hash_failure(tmp_path):
    artifact = PreparedRolloutScreenArtifact(
        PREPARED_ARTIFACT_SCHEMA,
        "temporal",
        {"landpoint_id": "one", "start_rows": [1]},
        "trace",
        {"batch": np.asarray([1.0])},
        {"day": np.asarray([2.0])},
    )
    artifact_path = tmp_path / "screen-001.pkl"
    record = write_prepared_rollout_artifact(artifact_path, artifact) | {
        "slice_id": "temporal",
        "window_entry": dict(artifact.window_entry),
        "trace_signature": "trace",
    }
    manifest = {
        "schema_version": PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "screen_contract_sha256": "contract",
        "window_plan_sha256": "plan",
        "source_git_head": "b" * 40,
        "artifact_count": 1,
        "total_size_bytes": record["size_bytes"],
        "entries": [record],
    }
    manifest_path = tmp_path / "prepared_manifest.json"
    atomic_json(manifest_path, manifest)
    loaded_manifest = load_prepared_rollout_manifest(
        manifest_path,
        screen_contract_sha256="contract",
        window_plan_sha256="plan",
        source_git_head="b" * 40,
    )
    loaded = load_prepared_rollout_artifact(
        manifest_path,
        loaded_manifest["entries"][0],
        slice_id="temporal",
        window_entry=artifact.window_entry,
    )
    assert np.array_equal(loaded.batch["batch"], np.asarray([1.0]))

    artifact_path.write_bytes(artifact_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="hash drift"):
        load_prepared_rollout_artifact(
            manifest_path,
            loaded_manifest["entries"][0],
            slice_id="temporal",
            window_entry=artifact.window_entry,
        )
