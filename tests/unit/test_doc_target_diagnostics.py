from __future__ import annotations

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.doc_target_diagnostics import (
    inventory_partition_statistics,
    masked_field_statistics,
    paired_daily_aggregate_statistics,
    paired_normalized_statistics,
)


def test_masked_field_statistics_exposes_sparse_coordinate_baseline() -> None:
    value = np.asarray(
        [
            [[0.0, 2.0], [0.0, 0.0]],
            [[0.0, 4.0], [0.0, 0.0]],
            [[0.0, 6.0], [0.0, 0.0]],
        ]
    )
    defined = np.ones_like(value, dtype=np.bool_)
    report = masked_field_statistics(value, defined, np.asarray([[1.0, 2.0], [1.0, 1.0]]))

    assert report["valid_scaled_count"] == 12
    assert report["meaningful_nonzero_count"] == 3
    assert report["active_day_count"] == 3
    assert report["active_coordinate_count"] == 1
    assert report["zero_predictor_mse"] == pytest.approx(14.0 / 12.0)
    assert report["coordinate_mean_predictor_mse"] == pytest.approx(2.0 / 12.0)
    assert report["coordinate_mean_residual_fraction_of_zero_mse"] == pytest.approx(1.0 / 7.0)


def test_paired_normalized_statistics_reports_coactivity_and_correlation() -> None:
    left = np.asarray([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    right = np.asarray([[0.0, 2.0, 0.0], [0.0, 4.0, 8.0]])
    defined = np.ones_like(left, dtype=np.bool_)
    report = paired_normalized_statistics(
        left,
        defined,
        np.ones(3),
        right,
        defined,
        np.ones(3),
    )

    assert report["paired_count"] == 6
    assert report["left_active_count"] == 4
    assert report["right_active_count"] == 3
    assert report["coactive_count"] == 3
    assert report["active_union_count"] == 4
    assert report["active_jaccard"] == pytest.approx(0.75)
    assert report["signed_correlation"] is not None


def test_paired_daily_statistics_accepts_different_coordinate_axes() -> None:
    left = np.asarray([[0.0, 2.0], [2.0, 2.0], [0.0, 0.0]])
    right = np.asarray(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    report = paired_daily_aggregate_statistics(
        left,
        np.ones_like(left, dtype=np.bool_),
        np.ones(2),
        right,
        np.ones_like(right, dtype=np.bool_),
        np.ones((2, 2)),
    )

    assert report["paired_day_count"] == 3
    assert report["left_active_day_count"] == 2
    assert report["right_active_day_count"] == 2
    assert report["coactive_day_count"] == 2
    assert report["active_day_union_count"] == 2
    assert report["active_day_jaccard"] == 1.0
    assert report["signed_daily_mean_correlation"] == pytest.approx(1.0)
    assert report["absolute_daily_mean_correlation"] == pytest.approx(1.0)


def test_inventory_partition_statistics_requires_exclusive_endpoint_roles() -> None:
    incoming = np.asarray([[0.0, 2.0, 0.0, 0.0]])
    outgoing = np.asarray([[0.5, 0.0, 0.0, 0.0]])
    defined = np.ones_like(incoming, dtype=np.bool_)
    report = inventory_partition_statistics(
        incoming,
        defined,
        outgoing,
        defined,
    )

    assert report == {
        "paired_count": 4,
        "incoming_negative_count": 0,
        "outgoing_out_of_unit_interval_count": 0,
        "simultaneously_active_count": 0,
        "incoming_only_count": 1,
        "outgoing_only_count": 1,
        "unchanged_count": 2,
        "mutually_exclusive": True,
    }


def test_doc_target_diagnostics_reject_shape_drift() -> None:
    with pytest.raises(ValueError, match="same day-first shape"):
        paired_normalized_statistics(
            np.zeros((2, 3)),
            np.ones((2, 3), dtype=np.bool_),
            np.ones(3),
            np.zeros((2, 4)),
            np.ones((2, 4), dtype=np.bool_),
            np.ones(4),
        )
