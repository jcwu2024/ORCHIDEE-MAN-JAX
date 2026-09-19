"""Report cadence, crash ordering and live-progress semantics on tiny state."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.mixed_training_state import (
    load_mixed_training_checkpoint,
    save_mixed_training_checkpoint,
)
from scripts.hpc.mixed_learning_reporting import MixedLearningReporter
from scripts.hpc.run_mixed_learning import (
    reconcile_resume_checkpoint,
    trim_resume_report,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.trained_screen_common import json_safe


def reporter(path, interval=25):
    return MixedLearningReporter(path, write_json=_atomic_json,
                                 json_safe=json_safe, interval=interval)


def initial():
    return dict(stage="metadata", status="running", completed_updates=0,
                updates=[], monitors=[{"update": 0, "physical": {"gpp": [1., 2.]}}],
                checkpoints=[], scientific_acceptance=False, elapsed_seconds=0.)


def update(record, step):
    record.update(stage="optimizer_completed", completed_updates=step,
                  elapsed_seconds=float(step))
    record["updates"].append(dict(update=step, seconds=.2, mixed_loss=1. / step,
                                  daily_hard={"stock_negative": 0}))


def test_full_history_is_periodic_but_progress_is_current_and_bounded(tmp_path):
    path = tmp_path / "report.json"
    writer = reporter(path)
    record = initial()
    writer.publish(record)
    record["monitors"][0]["large_trajectory"] = [1.] * 10000
    for step in range(1, 25):
        update(record, step)
        assert not writer.publish(record)
    assert json.loads(path.read_text())["completed_updates"] == 0
    progress = json.loads(writer.progress_path.read_text())
    assert progress["completed_updates"] == 24
    assert progress["full_report_completed_updates"] == 0
    assert "monitors" not in progress and "updates" not in progress
    assert writer.progress_path.stat().st_size < 2000
    update(record, 25)
    before = copy.deepcopy(record)
    assert writer.publish(record)
    assert json.loads(path.read_text()) == json_safe(record)
    assert record == before  # Reporting never prunes the in-memory history.
    assert writer.full_write_count == 2
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("stage", ["failed", "stopped", "completed", "monitor_completed"])
def test_boundary_flushes_all_records_and_removes_stale_live_fields(tmp_path, stage):
    path = tmp_path / "report.json"
    writer = reporter(path)
    record = initial()
    writer.publish(record)
    update(record, 1)
    record.update(stage="training", next_update=2, monitor_update=0)
    writer.publish(record)
    assert json.loads(writer.progress_path.read_text())["next_update"] == 2
    record.update(stage=stage, status="failed" if stage == "failed" else "completed")
    if stage == "failed":
        record["error"] = "example failure"
        record["failed_training_window"] = {"budget_failure": 1}
    writer.publish(record)
    assert json.loads(path.read_text()) == json_safe(record)
    progress = json.loads(writer.progress_path.read_text())
    assert "next_update" not in progress and "monitor_update" not in progress
    assert progress["latest_recorded_monitor_update"] == 0


def test_eager_and_periodic_final_reports_are_identical(tmp_path):
    record = initial()
    eager, periodic = reporter(tmp_path / "eager.json", 1), reporter(tmp_path / "periodic.json")
    for writer in (eager, periodic):
        writer.publish(record)
    for step in range(1, 32):
        update(record, step)
        for writer in (eager, periodic):
            writer.publish(record)
    record.update(stage="completed", status="completed")
    for writer in (eager, periodic):
        writer.publish(record)
    assert json.loads(eager.output.read_text()) == json.loads(periodic.output.read_text())
    assert periodic.full_write_count == 3
    assert eager.full_write_count == 33


def test_failed_full_write_does_not_claim_new_persisted_progress(tmp_path):
    writer = reporter(tmp_path / "report.json")
    record = initial()
    writer.publish(record)
    old_progress = writer.progress_path.read_text()
    update(record, 1)
    record["stage"] = "checkpoint_saving"
    def failing_write(path, payload):
        raise OSError("storage unavailable")
    writer.write_json = failing_write
    with pytest.raises(OSError, match="storage unavailable"):
        writer.publish(record)
    assert writer.last_full_update == 0
    assert writer.progress_path.read_text() == old_progress


@pytest.mark.parametrize("checkpoint_replaced", [False, True])
def test_crash_around_checkpoint_rename_leaves_recoverable_record_history(tmp_path, checkpoint_replaced):
    path = tmp_path / "report.json"
    checkpoint = tmp_path / "report.recovery.checkpoint.pkl"
    writer = reporter(path)
    record = initial()
    method = {"updates": 10}
    identity = {"source": "tiny"}
    fake_jax = SimpleNamespace(device_get=lambda value: value)

    def save(step):
        save_mixed_training_checkpoint(
            checkpoint, jax=fake_jax, parameters={"weight": np.array([step * .1])},
            optimizer_state=SimpleNamespace(step=step), completed_updates=step,
            daily_sampler_cursor={"epoch": 0, "batch_offset": step},
            recurrent_schedule_cursor=step, method=method, source_identities=identity,
            schedule_identity=identity, fresh_initialization_identity=identity,
            maximum_updates=10, replace=True)

    for step in range(1, 3):
        update(record, step)
    record["stage"] = "checkpoint_saving"
    writer.publish(record)
    save(2)
    record["checkpoints"].append({"update": 2, "path": str(checkpoint), "seconds": 1.})
    record["stage"] = "checkpoint_saved"
    writer.publish(record)
    update(record, 3)
    record["stage"] = "checkpoint_saving"
    writer.publish(record)
    if checkpoint_replaced:
        save(3)
    # Simulated process death: no checkpoint_saved and no exception handler.
    restored = load_mixed_training_checkpoint(
        checkpoint, expected_method=method, expected_source_identities=identity,
        expected_schedule_identity=identity, expected_fresh_initialization_identity=identity,
        maximum_updates=10)
    resumed = json.loads(path.read_text())
    completed = restored["completed_updates"]
    trim_resume_report(resumed, completed)
    reconcile_resume_checkpoint(resumed, checkpoint, completed)
    assert [row["update"] for row in resumed["updates"]] == list(range(1, completed + 1))
    assert resumed["checkpoints"][-1]["update"] == completed
    assert restored["daily_sampler_cursor"]["batch_offset"] == completed
    np.testing.assert_array_equal(restored["parameters"]["weight"], [completed * .1])
    if checkpoint_replaced:
        assert resumed["checkpoints"][-1]["recovered_metadata"]
        assert "seconds" not in resumed["checkpoints"][-1]  # Not a fabricated timing.
    else:
        assert resumed["checkpoints"][-1]["seconds"] == 1.
