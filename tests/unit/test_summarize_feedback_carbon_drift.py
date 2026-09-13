from scripts.dev.summarize_feedback_carbon_drift import summarize_report


def _window(*, horizon, artifact, predicted, teacher, canopy):
    return {
        "selection": {
            "artifact_id": artifact,
            "landpoint_id": "101.0-115.0",
            "year": 1995,
            "start_row": 9,
            "length": horizon,
        },
        "physical": {"stocks": {"nonliving_ecosystem_carbon": {
            "predicted_final": [predicted], "teacher_final": [teacher],
        }}},
        "feedback_diagnostics": {"canopy_stock_change": {
            "physical_residual": canopy,
        }},
    }


def test_summarizer_telescopes_canopy_by_its_selected_horizon():
    report = {
        "checkpoint_update": 1998,
        "windows": [
            _window(horizon=7, artifact=1, predicted=90.0, teacher=100.0, canopy=[-1.0] * 7),
            _window(horizon=30, artifact=2, predicted=70.0, teacher=100.0, canopy=[-2.0] * 30),
            _window(horizon=30, artifact=3, predicted=95.0, teacher=100.0, canopy=[-1.0] * 30),
        ],
    }

    result = summarize_report(report, source="synthetic.json")

    assert set(result["horizons"]) == {"7", "30"}
    t7 = result["horizons"]["7"]
    assert t7["window_count"] == 1
    assert t7["windows"][0]["cumulative_canopy_residual_gC_m2"] == -7.0
    assert t7["windows"][0]["remaining_nonliving_discrepancy_gC_m2"] == -3.0

    t30 = result["horizons"]["30"]
    assert t30["window_count"] == 2
    assert t30["windows"][0]["cumulative_canopy_residual_gC_m2"] == -60.0
    assert t30["windows"][0]["remaining_nonliving_discrepancy_gC_m2"] == 30.0
    assert t30["dominant_window"]["selection"]["artifact_id"] == 2
    assert t30["dominant_window"]["share_of_absolute_terminal_discrepancy"] == 6.0 / 7.0
