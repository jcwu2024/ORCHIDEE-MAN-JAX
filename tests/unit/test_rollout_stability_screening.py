from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.markov_dataset import (
    MarkovDatasetIndex,
    MarkovShardRef,
)
from research.daily_coarse_graining.rollout_stability_objective import (
    SCIENCE_FIELDS,
)
from research.daily_coarse_graining.rollout_stability_screening import (
    ARM_REPORT_SCHEMA_VERSION,
    REQUIRED_FEEDBACK,
    REQUIRED_HORIZONS,
    REQUIRED_SLICES,
    _cell_id,
    _evaluation_identity,
    _ratio_record,
    _require_complete_arm_report,
    _teacher_window_weights,
    build_model_selection_inventory,
    classify_screening,
)


def _reference(
    *,
    landpoint: str,
    year: int,
    spatial: str,
    temporal: str,
) -> MarkovShardRef:
    return MarkovShardRef(
        landpoint_id=landpoint,
        year=year,
        spatial_split=spatial,
        temporal_split=temporal,
        path=Path(f"/dataset/{landpoint}-{year}.npz"),
        sha256=f"sha-{landpoint}-{year}",
        contract_sha256="contract",
    )


def _training_protocol():
    return SimpleNamespace(
        raw={
            "validation": {
                "temporal": {
                    "spatial_split": "train",
                    "temporal_split": "validation",
                    "model_selection": True,
                },
                "spatial": {
                    "spatial_split": "validation",
                    "temporal_split": "train",
                    "model_selection": True,
                },
                "joint": {
                    "spatial_split": "validation",
                    "temporal_split": "validation",
                    "model_selection": True,
                },
            }
        }
    )


def test_model_selection_inventory_is_exact_and_never_uses_test():
    references = (
        _reference(
            landpoint="train",
            year=2005,
            spatial="train",
            temporal="validation",
        ),
        _reference(
            landpoint="validation",
            year=1999,
            spatial="validation",
            temporal="train",
        ),
        _reference(
            landpoint="validation",
            year=2005,
            spatial="validation",
            temporal="validation",
        ),
        _reference(
            landpoint="sealed",
            year=2009,
            spatial="test",
            temporal="test",
        ),
    )
    index = MarkovDatasetIndex(
        dataset_id="dataset",
        teacher_git_head="teacher",
        contract_sha256="contract",
        shards=references,
    )

    inventory = build_model_selection_inventory(index, _training_protocol())

    assert {name: value["shards"] for name, value in inventory.items()} == {
        "temporal": 1,
        "spatial": 1,
        "joint": 1,
    }
    assert all(
        "test" not in {item.spatial_split, item.temporal_split}
        for value in inventory.values()
        for item in value["references"]
    )


def test_teacher_forced_weights_cover_every_possible_window_exactly():
    day_indices = np.arange(365)

    all_weights, terminal_weights = _teacher_window_weights(365, day_indices)

    for index, horizon in enumerate(REQUIRED_HORIZONS):
        windows = 365 - horizon + 1
        assert int(np.sum(all_weights[index])) == windows * horizon
        assert int(np.sum(terminal_weights[index])) == windows
    assert np.all(all_weights[0] == 1)
    assert np.all(terminal_weights[0] == 1)


def test_ratio_gate_refuses_nonzero_candidate_over_zero_control():
    record = _ratio_record(
        gate_id="dynamic-status",
        candidate=1.0e-6,
        control=0.0,
        threshold=1.05,
    )

    assert record["ratio"] is None
    assert record["passed"] is False
    assert record["zero_denominator_policy"] == ("failed_nonzero_candidate_over_zero_control")


def _metric(value: float):
    return {
        "global": {"normalized_rmse": value},
        "families": {"family": {"normalized_rmse": value}},
    }


def _cell(value: float, *, windows: int, days: int):
    hard = {
        "state_unexpected_defined_status_mismatches": 0,
        "state_declared_dynamic_status_mismatches": 0,
        "state_dynamic_evaluated_values": days,
        "fast_unexpected_defined_status_mismatches": 0,
        "fast_declared_dynamic_status_mismatches": 0,
        "fast_dynamic_evaluated_values": days,
        "discrete_state_mismatches": 0,
        "nonfinite_defined_state_values": 0,
        "nonfinite_defined_fast_values": 0,
        "negative_source_nonnegative_carbon_stocks": 0,
    }
    return {
        "windows": windows,
        "predicted_days": days,
        "fast_boundary_terminal_lead": _metric(value),
        "state_terminal_lead": _metric(value),
        "tendency_bias_all_leads": {
            "global": {"weighted_huber_mean_bias": value},
            "families": {"family": {"weighted_huber_mean_bias": value}},
        },
        "science_terminal_state_and_all_lead_tendency": {field: {"objective_score": value} for field in SCIENCE_FIELDS},
        "hard_counts": hard,
    }


def _expected_counts():
    return {
        slice_id: {
            horizon: {
                "windows": 10,
                "predicted_days": 10 * horizon,
            }
            for horizon in REQUIRED_HORIZONS
        }
        for slice_id in REQUIRED_SLICES
    }


def _arm(value: float):
    expected = _expected_counts()
    return {
        "schema_version": ARM_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "sealed_test_used": False,
        "model_architecture": {
            "cross_day_memory": "canonical_state_only",
        },
        "structural_gates": {
            "restart_split_exact": True,
            "retained_tail": "canonical_source_backed_dynamic_transition",
        },
        "slices": {
            slice_id: {
                "cells": {
                    _cell_id(horizon, feedback): _cell(
                        value,
                        windows=expected[slice_id][horizon]["windows"],
                        days=expected[slice_id][horizon]["predicted_days"],
                    )
                    for horizon in REQUIRED_HORIZONS
                    for feedback in REQUIRED_FEEDBACK
                }
            }
            for slice_id in REQUIRED_SLICES
        },
    }


def _screening_protocol():
    return SimpleNamespace(
        raw={
            "screening_validation": {
                "gates_relative_to_one_step_continuation_control": {
                    "one_step_global_ratio_max": 1.02,
                    "one_step_family_ratio_max": 1.05,
                    "seven_day_free_rollout_global_ratio_max": 0.95,
                    "thirty_day_free_rollout_global_ratio_max": 0.95,
                    "tendency_bias_ratio_max": 0.90,
                    "science_metric_ratio_max": 1.05,
                    "declared_dynamic_status_error_ratio_max": 1.05,
                }
            }
        }
    )


def test_classification_passes_complete_better_candidate_and_catches_hard_failure():
    control = _arm(1.0)
    candidate = _arm(0.8)

    passed = classify_screening(
        control=control,
        candidate=candidate,
        protocol=_screening_protocol(),
        expected_counts=_expected_counts(),
    )

    assert passed["status"] == "passed"
    assert passed["decision"] == "advance_to_three_seed_confirmation"

    candidate["slices"]["joint"]["cells"][_cell_id(30, "free")]["hard_counts"]["discrete_state_mismatches"] = 1
    rejected = classify_screening(
        control=control,
        candidate=candidate,
        protocol=_screening_protocol(),
        expected_counts=_expected_counts(),
    )
    assert rejected["status"] == "rejected"
    assert any(not item["passed"] for item in rejected["hard_constraints"])


def test_incomplete_arm_report_is_refused_instead_of_classified():
    report = _arm(1.0)
    del report["slices"]["spatial"]["cells"][_cell_id(30, "free")]

    with pytest.raises(ValueError, match="cell inventory is incomplete"):
        _require_complete_arm_report(
            report,
            expected_counts=_expected_counts(),
        )


def test_screening_identity_binds_checkpoint_and_inventory(monkeypatch):
    monkeypatch.setattr(
        "research.daily_coarse_graining.rollout_stability_screening._current_git_head",
        lambda: "evaluation-head",
    )
    protocol = SimpleNamespace(sha256="protocol")
    execution = {
        "training_git_head": "training-head",
        "canonical_sha256": "execution",
        "artifacts": {"dataset": {"sha256": "dataset"}},
    }
    arms = {
        "control": {"checkpoint": {"sha256": "control"}},
        "candidate": {"checkpoint": {"sha256": "candidate"}},
    }
    inventory = {"canonical_sha256": "inventory"}

    first = _evaluation_identity(
        protocol=protocol,
        execution=execution,
        arms=arms,
        inventory_identity=inventory,
        batch_sizes={1: 4, 7: 4, 30: 2},
        max_shards_per_slice=None,
        verify_dataset_hashes=False,
    )
    changed_arms = copy.deepcopy(arms)
    changed_arms["candidate"]["checkpoint"]["sha256"] = "changed"
    second = _evaluation_identity(
        protocol=protocol,
        execution=execution,
        arms=changed_arms,
        inventory_identity=inventory,
        batch_sizes={1: 4, 7: 4, 30: 2},
        max_shards_per_slice=None,
        verify_dataset_hashes=False,
    )

    assert first["canonical_sha256"] != second["canonical_sha256"]
    assert first["sealed_test_used"] is False
