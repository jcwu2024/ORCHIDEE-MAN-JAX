from __future__ import annotations

import numpy as np

from research.daily_coarse_graining.canonical_daily_model import CanonicalDayBatch
from research.daily_coarse_graining.spatial_conditioning_diagnostic import (
    _nearest_training_distances,
    _normalize_feature_group,
    _permutation_indices,
    _replace_condition,
    classify_diagnostic,
)


def test_feature_normalization_uses_training_points_only_and_drops_constants():
    features = {
        "a": np.asarray([1.0, 5.0, 9.0]),
        "b": np.asarray([3.0, 5.0, 9.0]),
        "v": np.asarray([9.0, 5.0, 9.0]),
    }
    normalized, active = _normalize_feature_group(features, ("a", "b"))
    assert active == 1
    np.testing.assert_allclose(normalized["a"], [-1.0])
    np.testing.assert_allclose(normalized["b"], [1.0])
    np.testing.assert_allclose(normalized["v"], [7.0])


def test_nearest_distance_marks_validation_outside_train_envelope():
    features = {
        "a": np.asarray([0.0, 0.0]),
        "b": np.asarray([1.0, 0.0]),
        "c": np.asarray([0.0, 1.0]),
        "v": np.asarray([10.0, 10.0]),
    }
    report = _nearest_training_distances(
        features,
        ("a", "b", "c"),
        {"a": "train", "b": "train", "c": "train", "v": "validation"},
    )
    assert report["points"]["v"]["above_all_train_loo"]
    assert report["points"]["v"]["nearest_train_id"] in {"b", "c"}


def test_cross_point_permutation_preserves_date_blocks():
    ids = ("a", "b", "c", "a", "b", "c")
    np.testing.assert_array_equal(_permutation_indices(ids), [2, 0, 1, 5, 3, 4])


def test_forcing_intervention_replaces_the_explicit_forcing_mask():
    batch = CanonicalDayBatch(
        state=np.ones((1, 2)),
        state_finite=np.ones((1, 2), dtype=bool),
        normalized_fast_day_baseline=np.ones((1, 2)),
        forcing_native=np.ones((1, 4, 2)),
        forcing_finite=np.ones((1, 4, 2), dtype=bool),
        parameters=np.ones((1, 2)),
        parameters_finite=np.ones((1, 2), dtype=bool),
        landpoint_static=np.ones((1, 2)),
        landpoint_static_finite=np.ones((1, 2), dtype=bool),
        annual_conditions=np.ones((1, 2)),
        annual_conditions_finite=np.ones((1, 2), dtype=bool),
        calendar=np.ones((1, 4)),
    )
    values = np.zeros_like(batch.forcing_native)
    finite = np.zeros_like(batch.forcing_finite)
    replaced = _replace_condition(batch, "forcing_native", values, finite)
    np.testing.assert_array_equal(replaced.forcing_native, values)
    np.testing.assert_array_equal(replaced.forcing_finite, finite)


def test_classification_separates_coverage_and_condition_use():
    coverage = {"validation_outside_training_loo": False}
    condition_use = {
        "groups": {
            "landpoint_static": {
                "validation_cross_point_permutation": {
                    "loss_change_from_full": 0.02
                }
            },
            "parameters": {
                "validation_cross_point_permutation": {
                    "loss_change_from_full": 0.01
                }
            },
        }
    }
    result = classify_diagnostic(coverage, condition_use)
    assert result["decision"] == "coverage_not_extreme_architecture_or_representation_remains"
    assert result["spatial_condition_channel_detectably_used"]


def test_classification_reports_joint_coverage_and_condition_use_failure():
    coverage = {"validation_outside_training_loo": True}
    condition_use = {
        "groups": {
            name: {
                "validation_cross_point_permutation": {
                    "loss_change_from_full": 0.0
                }
            }
            for name in ("landpoint_static", "parameters")
        }
    }
    result = classify_diagnostic(coverage, condition_use)
    assert result["decision"] == "spatial_coverage_and_condition_use_both_insufficient"
