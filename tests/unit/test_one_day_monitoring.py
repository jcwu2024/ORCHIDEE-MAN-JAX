import json

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.one_day_monitoring import summarize_monitor


def _physical(*, fluxes=None, stocks=None):
    return {"fluxes": fluxes or {}, "stocks": stocks or {}}


def test_monitor_uses_joint_defined_finite_mask_and_reports_signed_bias():
    report = summarize_monitor(
        [
            _physical(
                fluxes={
                    "gpp": {
                        "predicted": np.array([2.0, 4.0, np.nan, np.inf, 99.0]),
                        "teacher": np.array([1.0, 6.0, 3.0, 4.0, np.nan]),
                        "defined": np.array([True, True, True, True, False]),
                    }
                }
            )
        ]
    )

    stats = report["fluxes"]["fields"]["gpp"]
    assert report["sample_count"] == 1
    assert report["fluxes"]["unit"] == "gC_m-2_day-1"
    assert stats["n"] == 2
    assert stats["nonfinite_defined"] == 2
    assert stats["missing_records"] == 0
    assert stats["rmse"] == pytest.approx(np.sqrt(2.5))
    assert stats["mae"] == pytest.approx(1.5)
    assert stats["bias"] == pytest.approx(-0.5)
    assert stats["absolute_error_p50"] == pytest.approx(1.5)
    assert stats["absolute_error_p90"] == pytest.approx(1.9)
    assert stats["absolute_error_p99"] == pytest.approx(1.99)
    assert stats["teacher_mean"] == pytest.approx(3.5)
    json.dumps(report, allow_nan=False)


def test_monitor_uses_explicit_stock_changes_and_never_cross_point_levels():
    records = [
        _physical(
            stocks={
                "biomass": {
                    "predicted": np.array([100.0]),
                    "teacher": np.array([90.0]),
                    "defined": np.array([True]),
                    "predicted_change": np.array([2.0]),
                    "teacher_change": np.array([1.0]),
                },
                "level_only": {
                    "predicted": np.array([1000.0]),
                    "teacher": np.array([10.0]),
                    "defined": np.array([True]),
                },
            }
        ),
        _physical(
            stocks={
                "biomass": {
                    "predicted": np.array([20.0]),
                    "teacher": np.array([10.0]),
                    "defined": np.array([True]),
                    "predicted_change": np.array([-3.0]),
                    "teacher_change": np.array([-1.0]),
                },
                "level_only": {
                    "predicted": np.array([1.0]),
                    "teacher": np.array([100.0]),
                    "defined": np.array([True]),
                },
            }
        ),
    ]

    report = summarize_monitor(records)
    biomass = report["stocks"]["fields"]["biomass"]
    assert report["stocks"]["unit"] == "gC_m-2"
    assert report["stocks"]["change_unit"] == "gC_m-2"
    assert biomass["level"]["bias"] == pytest.approx(10.0)
    assert biomass["change"]["n"] == 2
    assert biomass["change"]["rmse"] == pytest.approx(np.sqrt(2.5))
    assert biomass["change"]["mae"] == pytest.approx(1.5)
    assert biomass["change"]["bias"] == pytest.approx(-0.5)
    assert biomass["change"]["teacher_mean"] == pytest.approx(0.0)

    level_only_change = report["stocks"]["fields"]["level_only"]["change"]
    assert level_only_change["n"] == 0
    assert level_only_change["missing_records"] == 2
    assert level_only_change["rmse"] is None
    assert level_only_change["bias"] is None


def test_monitor_preserves_provided_fields_and_reports_missing_npp_without_deriving_it():
    report = summarize_monitor(
        [
            _physical(
                fluxes={
                    "gpp": {"predicted": [10.0], "teacher": [9.0], "defined": [True]},
                    "nee": {"predicted": [3.0], "teacher": [1.0], "defined": [True]},
                }
            )
        ]
    )
    assert tuple(report["fluxes"]["fields"]) == ("gpp", "nee")
    assert "npp" in report["fluxes"]["missing_expected_fields"]
    assert "npp" not in report["fluxes"]["fields"]

    provided = summarize_monitor(
        [
            _physical(
                fluxes={
                    "gpp": {"predicted": [100.0], "teacher": [100.0], "defined": [True]},
                    "npp": {"predicted": [7.0], "teacher": [2.0], "defined": [True]},
                }
            )
        ]
    )
    assert provided["fluxes"]["fields"]["npp"]["bias"] == pytest.approx(5.0)
    assert "npp" not in provided["fluxes"]["missing_expected_fields"]


def test_monitor_empty_valid_fields_and_empty_panel_use_none_not_nan():
    no_valid = summarize_monitor(
        [
            _physical(
                fluxes={
                    "gpp": {
                        "predicted": [np.nan, 5.0],
                        "teacher": [1.0, np.inf],
                        "defined": [True, True],
                    }
                }
            )
        ]
    )
    stats = no_valid["fluxes"]["fields"]["gpp"]
    assert stats["n"] == 0
    assert stats["nonfinite_defined"] == 2
    for name in (
        "rmse",
        "mae",
        "bias",
        "absolute_error_p50",
        "absolute_error_p90",
        "absolute_error_p99",
        "teacher_mean",
    ):
        assert stats[name] is None
    json.dumps(no_valid, allow_nan=False)

    empty = summarize_monitor([])
    assert empty["sample_count"] == 0
    assert empty["fluxes"]["fields"] == {}
    assert empty["stocks"]["fields"] == {}
    assert "npp" in empty["fluxes"]["missing_expected_fields"]
    json.dumps(empty, allow_nan=False)
