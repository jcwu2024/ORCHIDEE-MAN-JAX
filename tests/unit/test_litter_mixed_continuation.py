from __future__ import annotations

from pathlib import Path

import pytest

from scripts.hpc.probe_litter_mixed_admission import (
    EXPECTED_SELECTED_WEIGHT,
    EXPECTED_SOURCE_ADMISSION_GIT_HEAD,
    _align_continuation_history,
    _continuation_artifact_path,
    _continuation_checkpoint_method,
    _resume_continuation_report,
    _source_admission_method,
    _validate_continuation_protocol,
    run,
)


def continuation_protocol() -> dict:
    return {
        "schema_version": "litter_mixed_continuation_v1",
        "source_relative_update": 20,
        "target_relative_update": 100,
        "source_absolute_update": 4020,
        "milestones": [40, 60, 80, 100],
        "recovery_interval": 5,
        "selected_weight": EXPECTED_SELECTED_WEIGHT,
        "daily_batch": 64,
        "microbatch": 8,
        "outer_weights": {"daily": 1.0, "T7": 0.5, "T30": 0.5},
        "scientific_stop": {
            "consecutive_monitors": 2,
            "rmse_ratio": 2.0,
            "minimum_absolute_increase_gC_m2": {
                "flux": 1.0,
                "biomass": 1.0,
                "canopy": 5.0,
                "nonliving": 10.0,
            },
        },
        "scientific_acceptance": False,
    }


def source_protocol() -> dict:
    return {
        "schema_version": "litter_mixed_admission_v1",
        "source_update": 4000,
        "calibration_batch_indices": [4000, 4001],
        "update_batch_indices": list(range(4000, 4020)),
        "admission_updates": 2,
        "updates_per_arm": 20,
        "daily_batch": 64,
        "microbatch": 8,
        "outer_weights": {"daily": 1.0, "T7": 0.5, "T30": 0.5},
        "weight_selection": {
            "target_median_ratio": 0.5,
            "maximum_ratio": 1.0,
            "maximum_weight": 0.001,
        },
        "litter_carbon_stock_change": {
            "weight": 0.1,
            "penalty": "pseudo_huber",
            "scale_source_pft_index": 13,
            "minimum_scale_gC_m2_day": 0.01,
        },
        "daily_coefficients_old": [1.0, 0.1, 0.1, 0.0],
        "daily_coefficients_litter_only": [0.0, 0.0, 0.0, 1.0],
        "scientific_acceptance": False,
    }


def clean_panel() -> dict:
    return {
        "hard_counts": {"all_zero": True},
        "litter_stock_change": {"hard_invalid_count": 0, "hard_empty_count": 0},
    }


def source_report(tmp_path: Path) -> dict:
    return {
        "schema_version": "litter_mixed_admission_result_v1",
        "status": "completed",
        "technical_admission_passed": True,
        "bounded_continuation_completed": True,
        "full_mixed_admission": True,
        "monitor_gates_passed": True,
        "scientific_acceptance": False,
        "source_git_head": EXPECTED_SOURCE_ADMISSION_GIT_HEAD,
        "source_checkpoint_path": str(tmp_path / "source4000.pkl"),
        "protocol": source_protocol(),
        "selected_weight": EXPECTED_SELECTED_WEIGHT,
        "weight_selection": {"selected_weight": EXPECTED_SELECTED_WEIGHT},
        "arms": {
            "selected": {
                "status": "completed",
                "litter_weight": EXPECTED_SELECTED_WEIGHT,
                "completed_updates": 20,
                "final_optimizer_step": 4020,
                "final_daily_panels": {
                    "fixed_training_daily": clean_panel(),
                    "excluded_train_daily": clean_panel(),
                },
            }
        },
        "development_monitors": {
            "selected": {
                "physical_gate_passed": True,
                "metric_population_complete": True,
            }
        },
    }


def test_protocol_freezes_selected_only_20_to_100_geometry():
    admitted = _validate_continuation_protocol(continuation_protocol())
    assert admitted["update_indices"] == tuple(range(4020, 4100))
    assert admitted["milestones"] == (40, 60, 80, 100)
    assert admitted["selected_weight"] == EXPECTED_SELECTED_WEIGHT

    changed = continuation_protocol()
    changed["selected_weight"] *= 2
    with pytest.raises(ValueError, match="selected weight"):
        _validate_continuation_protocol(changed)

    changed = continuation_protocol()
    changed["milestones"] = [40, 80, 100]
    with pytest.raises(ValueError, match="milestones"):
        _validate_continuation_protocol(changed)


def test_source_method_is_reconstructed_exactly_and_continuation_tracks_parent(tmp_path):
    prior = source_report(tmp_path)
    source_method = _source_admission_method(prior)
    assert source_method["source_update"] == 4000
    assert source_method["arm"] == "selected"
    assert source_method["source_git_head"] == EXPECTED_SOURCE_ADMISSION_GIT_HEAD

    parent = tmp_path / "selected20.pkl"
    method = _continuation_checkpoint_method(
        source_checkpoint=parent,
        source_method=source_method,
        protocol=continuation_protocol(),
        runner_git_head="current-head",
    )
    assert method["parent_checkpoint"] == str(parent.resolve())
    assert method["parent_method"] == source_method
    assert method["runner_git_head"] == "current-head"

    prior["arms"]["selected"]["final_optimizer_step"] = 4019
    with pytest.raises(ValueError, match="selected20 progress"):
        _source_admission_method(prior)


def test_recovery_alignment_requires_monitor_before_advancing():
    updates = [{"relative_update": value} for value in range(21, 46)]
    monitor40 = {"relative_update": 40, "stop_decision": {"stop": False}}
    retained, monitors, due = _align_continuation_history(
        updates, [monitor40], completed=45, milestones=(40, 60, 80, 100)
    )
    assert len(retained) == 25
    assert monitors == [monitor40]
    assert due is None

    retained, monitors, due = _align_continuation_history(
        updates[:20], [], completed=40, milestones=(40, 60, 80, 100)
    )
    assert due == 40
    assert not monitors

    with pytest.raises(ValueError, match="unevaluated milestone"):
        _align_continuation_history(
            updates, [], completed=45, milestones=(40, 60, 80, 100)
        )


def test_resume_merge_preserves_progress_and_artifact_names(tmp_path):
    previous = {
        "updates": [{"relative_update": 21}],
        "milestone_monitors": [{"relative_update": 40}],
        "milestone_checkpoints": [{"relative_update": 40}],
        "recovery_checkpoint": {"relative_update": 40},
        "elapsed_seconds": 11.0,
        "prior_elapsed_seconds": 3.0,
        "status": "running",
    }
    current = {
        "updates": [],
        "milestone_monitors": [],
        "milestone_checkpoints": [],
        "status": "running",
        "source_git_head": "new",
    }
    merged = _resume_continuation_report(previous, current)
    assert merged["updates"] == previous["updates"]
    assert merged["milestone_monitors"] == previous["milestone_monitors"]
    assert merged["recovery_checkpoint"] == {"relative_update": 40}
    assert merged["prior_elapsed_seconds"] == 14.0
    assert merged["resumed_incomplete_report"]

    output = tmp_path / "continuation.json"
    assert _continuation_artifact_path(output, recovery=True).name.endswith(
        ".selected-continuation-recovery.checkpoint.pkl"
    )
    assert _continuation_artifact_path(output, relative_update=40).name.endswith(
        ".selected-continuation-update-000040.checkpoint.pkl"
    )


def test_continuation_options_must_be_paired_before_execution(tmp_path):
    args = type("Args", (), {
        "continuation_protocol": tmp_path / "continuation.json",
        "source_admission_report": None,
    })()
    with pytest.raises(ValueError, match="required together"):
        run(args)
