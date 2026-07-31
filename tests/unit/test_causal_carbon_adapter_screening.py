from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.causal_carbon_adapter_screening import (
    ARM_REPORT_SCHEMA_VERSION,
    _require_complete_arm_report,
    _signed_bias_huber,
    classify_screening,
)
from research.daily_coarse_graining.rollout_stability_screening import (
    REQUIRED_FEEDBACK,
    REQUIRED_HORIZONS,
    REQUIRED_SLICES,
    _cell_id,
)

INTERFACE_FIELDS = (
    "daily_interface.gpp_daily",
    "daily_interface.resp_maint_part",
    "ok_leak.carbon_32l",
    "ok_leak.deepC_peat",
)
PRIMARY_FIELDS = (
    "npp_daily",
    "resp_growth",
    "resp_maint",
    "biomass",
    "lai",
    "carbon_32l",
    "deepC_peat",
)
FLUX_FIELDS = ("npp_daily", "resp_growth", "resp_maint")
STOCK_FIELDS = ("biomass", "lai", "carbon_32l", "deepC_peat")
GUARD_FIELDS = ("litter", "DOC")


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


def _hard_counts(days):
    return {
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


def _error(value):
    return {
        "normalized_rmse": value,
        "normalized_huber": value,
    }


def _cell(value, *, windows, days):
    return {
        "windows": windows,
        "predicted_days": days,
        "state_terminal_lead": {"global": {"normalized_rmse": value}},
        "causal_carbon": {
            "interface_terminal": {field: _error(value) for field in INTERFACE_FIELDS},
            "primary_state_all_leads": {field: _error(value) for field in PRIMARY_FIELDS},
            "flux_state_bias_all_leads": {field: {"signed_bias_huber": value} for field in FLUX_FIELDS},
            "stock_tendency_bias_all_leads": {field: {"signed_bias_huber": value} for field in STOCK_FIELDS},
            "guard_state_all_leads": {field: _error(value) for field in GUARD_FIELDS},
        },
        "hard_counts": _hard_counts(days),
    }


def _arm(value):
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


def _protocol():
    return SimpleNamespace(
        raw={
            "screening_validation": {
                "gates_relative_to_control": {
                    "one_step_interface_ratio_max": 1.02,
                    "one_step_primary_state_ratio_max": 0.98,
                    "seven_day_primary_state_ratio_max": 0.95,
                    "thirty_day_primary_state_ratio_max": 0.95,
                    "global_state_ratio_max": 1.02,
                    "flux_bias_ratio_max": 0.90,
                    "stock_tendency_bias_ratio_max": 0.90,
                    "guard_field_ratio_max": 1.05,
                }
            }
        }
    )


def test_causal_screening_passes_better_candidate_and_rejects_guard_regression():
    control = _arm(1.0)
    candidate = _arm(0.8)

    passed = classify_screening(
        control=control,
        candidate=candidate,
        protocol=_protocol(),
        expected_counts=_expected_counts(),
    )

    assert passed["status"] == "passed"
    assert passed["decision"] == "advance_to_three_seed_confirmation"
    assert len(passed["ratio_gates"]) == 165
    assert len(passed["hard_constraints"]) == 108
    assert len(passed["structural_constraints"]) == 6
    assert all(record["passed"] for record in passed["hard_constraints"])
    assert all(record["passed"] for record in passed["structural_constraints"])

    candidate["slices"]["joint"]["cells"][_cell_id(30, "free")]["causal_carbon"]["guard_state_all_leads"]["DOC"][
        "normalized_huber"
    ] = 1.2
    rejected = classify_screening(
        control=control,
        candidate=candidate,
        protocol=_protocol(),
        expected_counts=_expected_counts(),
    )
    assert rejected["status"] == "rejected"
    assert any(
        record["id"] == "guard/joint/horizon_30/DOC" and not record["passed"] for record in rejected["ratio_gates"]
    )


def test_causal_screening_refuses_incomplete_population_evidence():
    report = _arm(1.0)
    del report["slices"]["spatial"]["cells"][_cell_id(30, "free")]

    with pytest.raises(ValueError, match="cell inventory is incomplete"):
        _require_complete_arm_report(
            report,
            expected_counts=_expected_counts(),
        )


def test_signed_bias_huber_uses_aggregate_signed_error_not_mae():
    sums = SimpleNamespace(
        normalized_sum=np.asarray([3.0, -1.0, 100.0]),
        evaluated_values=np.asarray([2.0, 2.0, 0.0]),
    )

    result = _signed_bias_huber(
        sums,
        np.asarray([True, True, False]),
    )

    assert result["normalized_mean_bias"] == pytest.approx(0.5)
    assert result["signed_bias_huber"] == pytest.approx(0.125)
    assert result["evaluated_values"] == 4
