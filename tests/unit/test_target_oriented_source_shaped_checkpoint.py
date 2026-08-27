from __future__ import annotations

import pickle

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_checkpoint import (
    CHECKPOINT_SCHEMA,
    RECOVERY_SCHEMA,
    load_source_shaped_checkpoint,
    load_source_shaped_recovery,
    write_source_shaped_checkpoint,
    write_source_shaped_recovery,
)


def _state(completed: int):
    branches = {
        "aboveground": {"kernel": np.asarray([1.0, 2.0], dtype=np.float32)},
        "belowground": {"kernel": np.asarray([3.0], dtype=np.float32)},
    }
    optimizer = {
        "step": np.asarray(completed, dtype=np.int32),
        "first": np.zeros(3, dtype=np.float32),
    }
    records = [{"optimizer_update": update, "loss": float(update)} for update in range(1, completed + 1)]
    return branches, optimizer, records


def test_source_shaped_recovery_round_trip(tmp_path):
    identity = {"parent_checkpoint_sha256": "parent", "screen_contract_sha256": "screen"}
    branches, optimizer, records = _state(2)
    path = tmp_path / "recovery.pkl"
    written = write_source_shaped_recovery(
        path,
        identity=identity,
        branches=branches,
        optimizer=optimizer,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=0.25,
        update_records=records,
    )
    loaded = load_source_shaped_recovery(
        path,
        identity=identity,
        parameter_count=3,
        clip_threshold=0.25,
        maximum_updates=100,
    )
    assert loaded is not None
    assert loaded["schema_version"] == RECOVERY_SCHEMA
    assert loaded["completed_updates"] == 2
    assert written["sha256"] == sha256_file(path)


def test_source_shaped_checkpoint_round_trip_and_identity(tmp_path):
    identity = {"parent_checkpoint_sha256": "parent", "screen_contract_sha256": "screen"}
    branches, optimizer, records = _state(2)
    groups = [{"update": value} for value in range(8)]
    comparison = {"classification": "source_mechanism_not_yet_supported"}
    path = tmp_path / "stage.pkl"
    written = write_source_shaped_checkpoint(
        path,
        identity=identity,
        branches=branches,
        optimizer=optimizer,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=0.25,
        update_records=records,
        candidate_groups=groups,
        screen_comparison=comparison,
    )
    loaded = load_source_shaped_checkpoint(
        path,
        expected_sha256=written["sha256"],
        identity=identity,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=0.25,
    )
    assert loaded["schema_version"] == CHECKPOINT_SCHEMA
    assert loaded["screen_comparison"] == comparison

    changed = pickle.loads(path.read_bytes())
    changed["identity"] = {"parent_checkpoint_sha256": "wrong"}
    path.write_bytes(pickle.dumps(changed))
    with pytest.raises(ValueError, match="identity or coverage"):
        load_source_shaped_checkpoint(
            path,
            expected_sha256=sha256_file(path),
            identity=identity,
            completed_updates=2,
            parameter_count=3,
            clip_threshold=0.25,
        )


def test_source_shaped_checkpoint_rejects_incomplete_updates(tmp_path):
    branches, optimizer, _records = _state(2)
    with pytest.raises(ValueError, match="update coverage"):
        write_source_shaped_recovery(
            tmp_path / "recovery.pkl",
            identity={"run": "test"},
            branches=branches,
            optimizer=optimizer,
            completed_updates=2,
            parameter_count=3,
            clip_threshold=0.25,
            update_records=[{"optimizer_update": 1}],
        )
