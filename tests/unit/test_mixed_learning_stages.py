from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.daily_coarse_graining.daily_operator.mixed_training_state import (
    load_mixed_training_checkpoint,
    save_mixed_training_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.mixed_learning_stages import validate_learning_schedule

ROOT = Path(__file__).resolve().parents[2]
OUTER_WEIGHTS = {"daily": 1.0, "T7": 0.5, "T30": 0.5}
SOURCES = {"daily": "daily-source", "recurrent": "recurrent-source"}
DATA_SCHEDULE = {"identity": "prepared-5415-schedule"}
FRESH = {"seed": 20260911, "parameter_count": 109_464_375}


def _schedule(name="integration_daily_then_mixed_v1"):
    return {
        "schema_version": "mixed_learning_stages_v1",
        "name": name,
        "total_updates": 2,
        "milestones": [0, 1, 2],
        "phases": [
            {
                "name": "daily_pretraining",
                "objective": "daily",
                "end_update": 1,
                "outer_weights": {"daily": 1.0},
            },
            {
                "name": "mixed_finetuning",
                "objective": "mixed",
                "end_update": 2,
                "outer_weights": OUTER_WEIGHTS,
            },
        ],
    }


def _validated(value=None, *, available=5415):
    return validate_learning_schedule(
        _schedule() if value is None else value,
        outer_weights=OUTER_WEIGHTS,
        available_updates=available,
    )


def _method(schedule):
    return {
        "implementation": "mixed_learning_v1",
        "configuration": {"updates": 5415},
        "scope": "integration_only",
        "integration_updates": 2,
        "learning_schedule": schedule.record,
    }


def _save(
    path,
    *,
    jax,
    parameters,
    optimizer,
    completed,
    recurrent,
    method,
    maximum,
):
    return save_mixed_training_checkpoint(
        path,
        jax=jax,
        parameters=parameters,
        optimizer_state=optimizer._replace(step=jax.numpy.asarray(completed)),
        completed_updates=completed,
        daily_sampler_cursor={"epoch": 0, "batch_offset": completed},
        recurrent_schedule_cursor=recurrent,
        method=method,
        source_identities=SOURCES,
        schedule_identity=DATA_SCHEDULE,
        fresh_initialization_identity=FRESH,
        maximum_updates=maximum,
    )


def _load(path, *, method, maximum):
    return load_mixed_training_checkpoint(
        path,
        expected_method=method,
        expected_source_identities=SOURCES,
        expected_schedule_identity=DATA_SCHEDULE,
        expected_fresh_initialization_identity=FRESH,
        maximum_updates=maximum,
    )


def test_daily_to_mixed_boundary_keeps_independent_recurrent_cursor():
    schedule = _validated()

    assert schedule.phase_for_update(1).objective == "daily"
    assert schedule.phase_for_update(2).objective == "mixed"
    assert [schedule.recurrent_cursor(update) for update in range(3)] == [0, 0, 1]
    assert schedule.progress(1) == {
        "completed_updates": 1,
        "completed_phase": "daily_pretraining",
        "next_phase": "mixed_finetuning",
        "daily_sampler_cursor": {"epoch": 0, "batch_offset": 1},
        "recurrent_schedule_cursor": 0,
        "consumed": {
            "daily_batches": 1,
            "recurrent_updates": 0,
            "recurrent_windows": {"T7": 0, "T30": 0},
        },
    }

    records = [
        {
            "update": 1,
            "phase": "daily_pretraining",
            "objective": "daily",
            "daily_batch_index": 0,
            "recurrent_schedule_index": None,
            "recurrent_schedule_cursor_after": 0,
        },
        {
            "update": 2,
            "phase": "mixed_finetuning",
            "objective": "mixed",
            "daily_batch_index": 1,
            "recurrent_schedule_index": 0,
            "recurrent_schedule_cursor_after": 1,
        },
    ]
    schedule.validate_update_records(records, completed_updates=2)
    records[1]["recurrent_schedule_index"] = 1
    with pytest.raises(ValueError, match="phase/cursor"):
        schedule.validate_update_records(records, completed_updates=2)


def test_enriched_report_keeps_canonical_progress_and_actual_counts_separate():
    schedule = _validated()
    batches = (
        SimpleNamespace(draw_count=64, source_draw_count=64, padding_draw_count=0),
        SimpleNamespace(draw_count=64, source_draw_count=61, padding_draw_count=3),
    )
    progress = schedule.progress(2)
    consumption = schedule.data_consumption(2, batches)

    schedule.validate_progress(progress, completed_updates=2)
    schedule.validate_data_consumption(
        consumption, completed_updates=2, batches=batches
    )
    assert consumption == {
        "daily_batches": 2,
        "recurrent_updates": 1,
        "recurrent_windows": {"T7": 1, "T30": 1},
        "daily_draws": 128,
        "daily_source_draws": 125,
        "daily_padding_draws": 3,
    }
    with pytest.raises(ValueError, match="data consumption"):
        schedule.validate_data_consumption(
            {**consumption, "daily_source_draws": 124},
            completed_updates=2,
            batches=batches,
        )


def test_pilot_manifests_preserve_weights_endpoints_and_recurrent_counts():
    manifest_root = ROOT / "manifests" / "coarse_graining"
    staged = _validated(
        json.loads((manifest_root / "mixed_learning_stages_pilot_v1.json").read_text())
    )
    joint = _validated(
        json.loads((manifest_root / "mixed_learning_joint_pilot_v1.json").read_text())
    )

    assert staged.total_updates == 800
    assert staged.milestones == (0, 200, 400, 600, 800)
    assert staged.phase_for_update(400).objective == "daily"
    assert staged.phase_for_update(401).objective == "mixed"
    assert staged.recurrent_cursor(400) == 0
    assert staged.recurrent_cursor(800) == 400
    assert joint.recurrent_cursor(joint.total_updates) == joint.total_updates
    assert all(
        phase.outer_weights == OUTER_WEIGHTS
        for phase in (*staged.phases, *joint.phases)
        if phase.objective == "mixed"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value.update(total_updates=5416), "available daily batches"),
        (lambda value: value["milestones"].remove(1), "phase endpoint"),
        (
            lambda value: value["phases"][1]["outer_weights"].update(T30=0.25),
            "changes the declared outer weights",
        ),
        (lambda value: value["phases"][1].update(end_update=1), "increasing"),
    ),
)
def test_schedule_drift_fails_closed(mutate, message):
    value = copy.deepcopy(_schedule())
    mutate(value)
    with pytest.raises(ValueError, match=message):
        _validated(value)


def test_checkpoint_round_trip_crosses_stage_boundary_and_rejects_cursor_tamper(
    tmp_path,
):
    import jax
    import jax.numpy as jnp

    schedule = _validated()
    method = _method(schedule)
    parameters = {"weight": jnp.asarray([1.0])}
    optimizer = initialize_target_oriented_adamw(parameters)

    daily_path = tmp_path / "daily-end.pkl"
    _save(
        daily_path,
        jax=jax,
        parameters=parameters,
        optimizer=optimizer,
        completed=1,
        recurrent=0,
        method=method,
        maximum=2,
    )
    assert _load(daily_path, method=method, maximum=2)["recurrent_schedule_cursor"] == 0

    mixed_path = tmp_path / "mixed-end.pkl"
    _save(
        mixed_path,
        jax=jax,
        parameters=parameters,
        optimizer=optimizer,
        completed=2,
        recurrent=1,
        method=method,
        maximum=2,
    )
    restored = _load(mixed_path, method=method, maximum=2)
    assert restored["daily_sampler_cursor"]["batch_offset"] == 2
    assert restored["recurrent_schedule_cursor"] == 1
    assert int(restored["optimizer_state"].step) == 2

    with mixed_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["recurrent_schedule_cursor"] = 2
    with mixed_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="declared phase progress"):
        _load(mixed_path, method=method, maximum=2)

    changed = _method(_validated(_schedule(name="different-schedule")))
    with pytest.raises(ValueError, match="method mismatch"):
        _load(daily_path, method=changed, maximum=2)
    with pytest.raises(ValueError, match="differs from the checkpoint method"):
        _load(daily_path, method=method, maximum=5415)


def test_plain_epoch_checkpoint_can_pass_legacy_200_update_ceiling(tmp_path):
    import jax
    import jax.numpy as jnp

    method = {
        "implementation": "mixed_learning_v1",
        "configuration": {"updates": 5415},
        "scope": "one_pool_pass_learning",
    }
    parameters = {"weight": jnp.asarray([1.0])}
    optimizer = initialize_target_oriented_adamw(parameters)
    path = tmp_path / "epoch-update-002500.pkl"
    _save(
        path,
        jax=jax,
        parameters=parameters,
        optimizer=optimizer,
        completed=2500,
        recurrent=2500,
        method=method,
        maximum=5415,
    )

    restored = _load(path, method=method, maximum=5415)
    assert restored["completed_updates"] == 2500
    assert restored["recurrent_schedule_cursor"] == 2500
