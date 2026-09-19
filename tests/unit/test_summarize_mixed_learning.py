from __future__ import annotations

from scripts.hpc import summarize_mixed_learning as summary


def _stats(rmse, bias):
    return {"rmse": rmse, "bias": bias, "mae": abs(bias), "count": 1, "nonfinite_defined": 0}


def _window(selection_id="same", rh_teacher=0.0):
    fluxes = {}
    for name, teacher, predicted in (
        ("gpp", 10.0, 8.0),
        ("npp", 5.0, 4.0),
        ("autotrophic_respiration", 3.0, 4.0),
        ("heterotrophic_respiration", rh_teacher, 0.0),
    ):
        fluxes[name] = {
            "teacher_total": [teacher],
            "predicted_total": [predicted],
        }
    stocks = {}
    for name in (
        "aboveground_biomass_carbon",
        "belowground_biomass_carbon",
        "nonliving_ecosystem_carbon",
    ):
        stocks[name] = {
            "teacher_change": [1.0],
            "predicted_change": [-2.0],
            "teacher_final": [101.0],
            "predicted_final": [102.0],
        }
    return {
        "selection": {"length": 7, "artifact_id": selection_id, "start_row": 4},
        "physical": {"fluxes": fluxes, "stocks": stocks},
        "canopy_physical": {
            "teacher_change": [2.0],
            "predicted_change": [-1.0],
            "teacher_final": [202.0],
            "predicted_final": [199.0],
        },
    }


def _horizon(rh_nonzero=0):
    fluxes = {}
    for name in ("gpp", "npp", "autotrophic_respiration", "heterotrophic_respiration"):
        fluxes[name] = {
            "cumulative_gC_m2": _stats(2.0, -1.0),
            "teacher_nonzero_totals": rh_nonzero if name == "heterotrophic_respiration" else 1,
        }
    stocks = {}
    for name in (
        "aboveground_biomass_carbon",
        "belowground_biomass_carbon",
        "nonliving_ecosystem_carbon",
    ):
        stocks[name] = {
            "change_gC_m2": _stats(3.0, -3.0),
            "final_gC_m2": _stats(999.0, 999.0),
        }
    return {
        "physical": {"fluxes": fluxes, "stocks": stocks},
        "canopy_physical": {
            "change_gC_m2": _stats(3.0, -3.0),
            "final_gC_m2": _stats(999.0, 999.0),
        },
    }


def _report(selection_id="same", monitors=True, rh_teacher=0.0):
    monitor = {
        "update": 2,
        "development": {
            "role": "development",
            "windows": [_window(selection_id, rh_teacher)],
            "horizons": {"7": _horizon(int(rh_teacher != 0.0))},
        },
        "fixed_training_recurrent": {
            "role": "fixed_train_monitor",
            "windows": [_window("fixed", rh_teacher)],
            "horizons": {"7": _horizon(int(rh_teacher != 0.0))},
        },
    }
    return {
        "schema_version": "mixed_learning_result_v1",
        "status": "running",
        "completed_updates": 2,
        "learning_schedule": {"total_updates": 10},
        "scientific_acceptance": False,
        "updates": [{"update": 1, "seconds": 2.0}, {"update": 2, "seconds": 3.0}],
        "monitors": [monitor] if monitors else [],
    }


def test_stock_growth_uses_change_and_retains_inventory():
    result = summary.summarize_reports({"arm": _report()})
    record = next(row for row in result["records"] if row["panel"] == "development")

    assert record["metrics"]["agb_growth"]["rmse"] == 3.0
    assert record["metrics"]["agb_growth"]["source_field"] == "change_gC_m2"
    values = record["windows"][0]["metrics"]["agb_growth"]
    assert values["teacher"] == 1.0
    assert values["predicted"] == -2.0
    assert values["inventory_teacher"] == 101.0
    assert values["inventory_predicted"] == 102.0
    assert record["measured_update_seconds"] == 5.0


def test_incomplete_snapshot_without_monitors_is_reportable(tmp_path):
    result = summary.summarize_reports({"live": _report(monitors=False)})

    assert result["records"] == []
    assert "live: no recorded monitor snapshots yet" in result["warnings"]
    assert any("no post-monitor learning outcome is inferred" in item for item in result["warnings"])
    outputs = summary.write_bundle(result, tmp_path, "Live snapshot")
    assert {path.name for path in outputs} == {"summary.json", "README.md"}
    assert "Recorded monitors" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_zero_rh_caveat_and_mismatched_populations_are_separated():
    result = summary.summarize_reports(
        {"first": _report("population-a"), "second": _report("population-b")}
    )

    assert result["rh_teacher_all_zero"] is True
    assert any("zero RH error" in limitation for limitation in result["limitations"])
    populations = result["population_groups"]["development"]["7"]
    assert len(populations) == 2
    assert all(len(group["arms"]) == 1 for group in populations.values())
    assert all(not group["overlay_is_population_comparable"] for group in populations.values())


def test_same_selection_with_different_teacher_reference_is_not_overlaid():
    first = _report("same-selection")
    second = _report("same-selection")
    second["monitors"][0]["development"]["windows"][0]["physical"]["fluxes"]["gpp"][
        "teacher_total"
    ] = [11.0]

    result = summary.summarize_reports({"first": first, "second": second})

    populations = result["population_groups"]["development"]["7"]
    assert len(populations) == 2
    assert all(len(group["arms"]) == 1 for group in populations.values())


def test_training_beyond_initial_monitor_is_not_called_learning():
    report = _report()
    report["completed_updates"] = 15
    report["monitors"][0]["update"] = 0

    result = summary.summarize_reports({"active": report})

    assert result["arms"]["active"]["latest_recorded_monitor_update"] == 0
    assert result["arms"]["active"]["unmonitored_completed_updates"] == 15
    assert any("training reached update 15" in item for item in result["warnings"])
