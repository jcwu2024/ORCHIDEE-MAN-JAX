from copy import deepcopy

from scripts.hpc.litter_continuation_monitor import should_stop, summarize_progress

POLICY = {
    "consecutive_monitors": 2,
    "rmse_ratio": 2.0,
    "minimum_absolute_increase_gC_m2": {
        "flux": 1.0, "biomass": 1.0, "canopy": 5.0, "nonliving": 10.0,
    },
}


def panel():
    return {"horizons": {
        str(horizon): {
            "physical": {
                "fluxes": {
                    name: {"cumulative_gC_m2": {"rmse": 2.0}, "daily_gC_m2_day": {"rmse": 99.0}}
                    for name in ("gpp", "npp", "autotrophic_respiration")
                },
                "stocks": {
                    name: {"change_gC_m2": {"rmse": 0.2}}
                    for name in ("aboveground_biomass_carbon", "belowground_biomass_carbon", "nonliving_ecosystem_carbon")
                },
            },
            "canopy_physical": {"change_gC_m2": {"rmse": 4.0}},
        } for horizon in (7, 30)
    }}


def test_progress_uses_cumulative_flux_and_separate_canopy_change():
    baseline = panel()
    current = deepcopy(baseline)
    current["horizons"]["30"]["physical"]["fluxes"]["gpp"]["cumulative_gC_m2"]["rmse"] = 1.0
    summary = summarize_progress(baseline, current)
    assert len(summary["metrics"]) == 14
    assert summary["metrics"]["T30/gpp"]["relative_change"] == -0.5
    assert summary["metrics"]["T7/canopy_interception_organic_carbon"]["group"] == "canopy"
    assert not summary["scientific_acceptance"]


def test_transient_or_different_metric_regressions_do_not_stop():
    baseline = panel()
    first = deepcopy(baseline)
    second = deepcopy(baseline)
    first["horizons"]["30"]["physical"]["fluxes"]["gpp"]["cumulative_gC_m2"]["rmse"] = 5.0
    second["horizons"]["30"]["physical"]["fluxes"]["npp"]["cumulative_gC_m2"]["rmse"] = 5.0
    a, b = [summarize_progress(baseline, value) for value in (first, second)]
    assert not should_stop([a], POLICY)["stop"]
    assert not should_stop([a, b], POLICY)["stop"]
    decision = should_stop([a, a], POLICY)
    assert decision["stop"]
    assert decision["triggered_metrics"] == ["T30/gpp"]


def test_large_ratio_of_small_error_does_not_stop_and_zero_baseline_is_serializable():
    baseline = panel()
    current = deepcopy(baseline)
    key = "belowground_biomass_carbon"
    current["horizons"]["7"]["physical"]["stocks"][key]["change_gC_m2"]["rmse"] = 0.9
    summary = summarize_progress(baseline, current)
    assert summary["metrics"][f"T7/{key}"]["ratio"] == 4.5
    assert not should_stop([summary, summary], POLICY)["stop"]
    baseline["horizons"]["7"]["physical"]["stocks"][key]["change_gC_m2"]["rmse"] = 0.0
    summary = summarize_progress(baseline, current)
    assert summary["metrics"][f"T7/{key}"]["ratio"] is None
    assert not should_stop([summary, summary], POLICY)["stop"]
