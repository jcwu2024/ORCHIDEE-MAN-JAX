"""Role separation tests for the reusable 200-update recurrent monitor."""

import pytest

from scripts.hpc.mixed_learning_monitor import _panel_classification, _validate_panel_rows


def _row(horizon, index):
    return {
        "length": horizon,
        "component": {7: "seven_day", 30: "thirty_day"}[horizon],
        "artifact_id": index,
        "landpoint_id": f"site-{index}",
        "year": 2000,
        "start_row": index * 100,
        "teacher_day_keys": [(f"site-{index}", 2000, index * 100 + day) for day in range(horizon)],
    }


def _horizon(horizon, windows, *, keys_pass=True):
    return {
        "window_count": windows,
        "physical": {
            "fluxes": {
                f"f{index}": {
                    "cumulative_gC_m2": {"count": windows, "nonfinite_defined": 0},
                    "daily_gC_m2_day": {"count": windows * horizon, "nonfinite_defined": 0},
                }
                for index in range(8)
            },
            "stocks": {
                f"s{index}": {
                    "final": {"count": windows, "nonfinite_defined": 0},
                    "change": {"count": windows, "nonfinite_defined": 0},
                }
                for index in range(5)
            },
        },
        "science": {
            "keys": {str(index): {"evaluated_values": windows} for index in range(13)},
            "all_key_gates_passed": keys_pass,
        },
    }


def test_train_panel_never_claims_development_science_acceptance():
    rows = (_row(7, 1), _row(30, 2))
    assert _validate_panel_rows(rows, development=False, apply_science_gates=False) == rows
    report = {
        "windows": [{"hard": {"budget_failure": 0}}, {"hard": {"budget_failure": 0}}],
        "horizons": {"7": _horizon(7, 1), "30": _horizon(30, 1)},
    }
    result = _panel_classification(report, development=False, apply_science_gates=False)
    assert result == {
        "physical_gate_passed": True,
        "metric_population_complete": True,
        "panel_science_gates_applied": False,
        "panel_science_gates_passed": False,
        "preliminary_screen_passed": False,
    }


def test_development_panel_requires_existing_strict_sixteen_window_selection():
    rows = tuple(_row(7, index) for index in range(8)) + tuple(_row(30, 100 + index) for index in range(8))
    assert len(_validate_panel_rows(rows, development=True, apply_science_gates=True)) == 16
    with pytest.raises(ValueError, match="strict development"):
        _validate_panel_rows(rows[:2], development=False, apply_science_gates=True)
    with pytest.raises(ValueError, match="exactly eight"):
        _validate_panel_rows(rows[:15], development=True, apply_science_gates=True)


def test_complete_development_classification_uses_both_horizon_populations():
    report = {
        "windows": [{"hard": {"budget_failure": 0}} for _ in range(16)],
        "horizons": {"7": _horizon(7, 8), "30": _horizon(30, 8)},
    }
    accepted = _panel_classification(report, development=True, apply_science_gates=True)
    assert accepted["metric_population_complete"]
    assert accepted["panel_science_gates_passed"]
    assert accepted["preliminary_screen_passed"]
    report["horizons"]["30"]["science"]["keys"]["0"]["evaluated_values"] = 7
    rejected = _panel_classification(report, development=True, apply_science_gates=True)
    assert not rejected["metric_population_complete"]
    assert not rejected["panel_science_gates_passed"]


def test_train_population_completeness_rejects_missing_or_nonfinite_metrics():
    report = {
        "windows": [{"hard": {"budget_failure": 0}}, {"hard": {"budget_failure": 0}}],
        "horizons": {"7": _horizon(7, 1), "30": _horizon(30, 1)},
    }
    assert _panel_classification(report, development=False, apply_science_gates=False)[
        "metric_population_complete"
    ]
    report["horizons"]["7"]["physical"]["fluxes"]["f0"]["daily_gC_m2_day"][
        "nonfinite_defined"
    ] = 1
    assert not _panel_classification(report, development=False, apply_science_gates=False)[
        "metric_population_complete"
    ]
