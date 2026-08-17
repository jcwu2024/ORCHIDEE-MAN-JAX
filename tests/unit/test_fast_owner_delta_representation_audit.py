from __future__ import annotations

import numpy as np

from research.daily_coarse_graining.fast_owner_delta_representation_audit import (
    _error_summary,
    summarize_endpoint_reachability,
)


def test_nonnegative_endpoint_allows_negative_delta() -> None:
    result = summarize_endpoint_reachability(
        np.asarray([-6.0, -2.0, 0.0, 1.0]),
        np.asarray([10.0, 1.0, 0.0, 2.0]),
        np.asarray([True, True, True, True]),
        mean=np.zeros(4),
        scale=np.ones(4),
        constraint="nonnegative",
    )

    assert result["defined_value_count"] == 4
    assert result["negative_delta_count"] == 2
    assert result["hard_range_impossible_count"] == 1
    assert result["finite_raw_boundary_only_count"] == 1
    assert result["finite_required_raw_count"] == 2


def test_unit_interval_endpoint_checks_absolute_value_not_delta() -> None:
    result = summarize_endpoint_reachability(
        np.asarray([-0.6, -0.2, 0.2, 0.5, 1.2]),
        np.asarray([0.8, 0.1, 0.1, 0.5, 0.0]),
        np.ones(5, dtype=np.bool_),
        mean=np.zeros(5),
        scale=np.ones(5),
        constraint="unit_interval",
    )

    assert result["hard_range_impossible_count"] == 2
    assert result["finite_raw_boundary_only_count"] == 1
    assert result["negative_delta_count"] == 2
    assert result["finite_required_raw_count"] == 2


def test_signed_endpoint_has_no_transform_range_failure() -> None:
    result = summarize_endpoint_reachability(
        np.asarray([-1.0e6, 0.0, 1.0e6]),
        np.asarray([2.0e6, 0.0, -2.0e6]),
        np.ones(3, dtype=np.bool_),
        mean=np.zeros(3),
        scale=np.ones(3),
        constraint="signed",
    )

    assert result["hard_range_impossible_count"] == 0
    assert result["finite_raw_boundary_only_count"] == 0


def test_boundary_only_endpoint_has_no_finite_raw_summary() -> None:
    result = summarize_endpoint_reachability(
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 0.0]),
        np.ones(2, dtype=np.bool_),
        mean=np.zeros(2),
        scale=np.ones(2),
        constraint="nonnegative",
    )

    assert result["finite_required_raw_count"] == 0
    assert result["finite_raw_boundary_only_count"] == 2
    assert result["required_raw_absolute_median"] is None
    assert result["required_raw_absolute_p99"] is None
    assert result["required_raw_absolute_maximum"] is None


def test_source_reconstruction_error_summary_uses_field_aware_tolerance() -> None:
    result = _error_summary(
        np.asarray([1.0, 1.0e6]),
        np.asarray([1.0 + 5.0e-11, 1.0e6 + 5.0e-7]),
    )

    assert result["passed"]
    assert result["within_tolerance_count"] == 2
    assert result["exact_value_count"] == 0


def test_source_reconstruction_error_summary_ignores_undefined_target_slots() -> None:
    result = _error_summary(
        np.asarray([np.nan, 2.0]),
        np.asarray([1.0e20, 2.0]),
    )

    assert result["passed"]
    assert result["value_count"] == 1
    assert result["undefined_target_count"] == 1
    assert result["nonfinite_reconstructed_count"] == 0
