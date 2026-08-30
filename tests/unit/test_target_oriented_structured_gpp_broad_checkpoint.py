import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_checkpoint import (
    load_broad_checkpoint,
    load_broad_recovery,
    write_broad_checkpoint,
    write_broad_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    TargetOrientedAdamWState,
)


def _monitor(loss=1.0):
    return {
        "aggregate": {
            "loss": loss,
            "mse": 1.5,
            "normalized_bias": -0.5,
            "bias_squared": 0.25,
            "count": 2048.0,
        },
        "records": [
            {
                "landpoint_id": f"p{index}",
                "year": 1961,
                "parent_shard_sha256": f"parent-{index}",
                "typed_shard_sha256": f"typed-{index}",
            }
            for index in range(128)
        ],
    }


def _kwargs():
    specialist = {"weight": np.asarray([1.0, 2.0], dtype=np.float32)}
    optimizer = TargetOrientedAdamWState(
        step=np.asarray(1, dtype=np.int32),
        first={"weight": np.zeros((2,), dtype=np.float32)},
        second={"weight": np.zeros((2,), dtype=np.float32)},
    )
    return {
        "identity": {"contract": "abc"},
        "specialist": specialist,
        "optimizer": optimizer,
        "completed_updates": 1,
        "parameter_count": 2,
        "clip_threshold": 3.0,
        "update_records": [{"optimizer_update": 1}],
        "initial_monitor": _monitor(),
        "current_monitor": _monitor(0.9),
        "stage_metadata": {"monitor_panels": []},
    }


def test_structured_gpp_broad_recovery_round_trip(tmp_path):
    path = tmp_path / "recovery.pkl"
    record = write_broad_recovery(path, **_kwargs())
    loaded = load_broad_recovery(
        path,
        identity={"contract": "abc"},
        parameter_count=2,
        clip_threshold=3.0,
        maximum_updates=1000,
    )
    assert record["completed_updates"] == loaded["completed_updates"] == 1
    with pytest.raises(ValueError, match="identity drift"):
        load_broad_recovery(
            path,
            identity={"contract": "changed"},
            parameter_count=2,
            clip_threshold=3.0,
            maximum_updates=1000,
        )


def test_structured_gpp_broad_checkpoint_round_trip(tmp_path):
    path = tmp_path / "stage-update-0001.pkl"
    record = write_broad_checkpoint(path, **_kwargs())
    loaded = load_broad_checkpoint(
        path,
        expected_sha256=record["sha256"],
        identity={"contract": "abc"},
        completed_updates=1,
        parameter_count=2,
        clip_threshold=3.0,
    )
    assert loaded["current_monitor"]["aggregate"]["loss"] == 0.9


def test_structured_gpp_broad_checkpoint_rejects_incomplete_monitor_count(tmp_path):
    kwargs = _kwargs()
    kwargs["initial_monitor"]["aggregate"]["count"] = 2032.0
    with pytest.raises(ValueError, match="monitor drift"):
        write_broad_recovery(tmp_path / "recovery.pkl", **kwargs)
