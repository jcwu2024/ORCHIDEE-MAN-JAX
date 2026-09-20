"""Physical identities and missing-day handling for the saved-result audit."""
import numpy as np

from scripts.hpc.analyze_mixed_carbon_drift import stock_change_series, window_diagnostics


def _stock(predicted, teacher, initial):
    return {"predicted_daily": [[x] for x in predicted], "teacher_daily": [[x] for x in teacher],
            "predicted_final": [predicted[-1]], "teacher_final": [teacher[-1]],
            "predicted_change": [predicted[-1] - initial], "teacher_change": [teacher[-1] - initial],
            "defined": [[True], [True]]}


def _window():
    def flux(p, t):
        return {"predicted_daily": [[x] for x in p], "teacher_daily": [[x] for x in t],
                "defined": [[True], [True]]}
    return {"selection": {"length": 2},
            "canopy_physical": _stock([13, 15], [12, 14], 10),
            "physical": {"stocks": {"nonliving_ecosystem_carbon": _stock([102, 101], [105, 108], 100)},
                         "fluxes": {"gpp": flux([8, 10], [6, 7]), "npp": flux([5, 6], [4, 4]),
                                    "autotrophic_respiration": flux([3, 4], [2, 3]),
                                    "maintenance_respiration": flux([2, 2], [1, 2]),
                                    "growth_respiration": flux([1, 2], [1, 1])}}}


def test_decomposition_respects_day_zero_and_flux_identities():
    window = _window()
    np.testing.assert_allclose(stock_change_series(window["canopy_physical"], "predicted"), [[3], [5]])
    result = window_diagnostics(window)
    np.testing.assert_allclose(result["nonliving_error"], [[-3], [-7]])
    np.testing.assert_allclose(result["canopy_error"], [[1], [1]])
    np.testing.assert_allclose(result["noncanopy_error"], [[-4], [-8]])
    np.testing.assert_allclose(result["npp_cumulative_error"], [[1], [3]])
    assert result["npp_error_identity_max_abs"] == 0


def test_missing_day_does_not_become_valid_cumulative_flux():
    window = _window()
    window["physical"]["fluxes"]["gpp"]["defined"][0] = [False]
    result = window_diagnostics(window)
    assert np.all(np.isnan(result["gpp_cumulative_error"]))
    np.testing.assert_allclose(result["noncanopy_error"], [[-4], [-8]])
