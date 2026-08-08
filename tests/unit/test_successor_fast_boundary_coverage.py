from __future__ import annotations

from scripts.dev.audit_successor_fast_boundary_coverage import build_ledger


def _leaf(
    key: str,
    start: int,
    *,
    component: str | None,
    family: str,
) -> dict[str, object]:
    field = key.rsplit(".", 1)[-1]
    return {
        "key": key,
        "family": family,
        "component": component,
        "path": [field],
        "shape": [1],
        "start": start,
        "stop": start + 1,
    }


def test_successor_fast_boundary_ledger_distinguishes_baseline_and_owner():
    state_leaves = [
        _leaf(
            "driver_previous_step_state.fluxsens",
            0,
            component="driver_previous_step_state",
            family="driver",
        ),
        _leaf(
            "hydrol_previous_step_state.mc",
            1,
            component="hydrol_previous_step_state",
            family="hydrol",
        ),
        _leaf(
            "thermosoil_previous_step_state.ptn",
            2,
            component="thermosoil_previous_step_state",
            family="thermosoil",
        ),
    ]
    fast_leaves = [
        _leaf(
            "driver_previous_step_state.fluxsens",
            0,
            component="driver_previous_step_state",
            family="driver",
        ),
        _leaf(
            "hydrol_previous_step_state.mc",
            1,
            component="hydrol_previous_step_state",
            family="hydrol",
        ),
        _leaf(
            "thermosoil_previous_step_state.ptn",
            2,
            component="thermosoil_previous_step_state",
            family="thermosoil",
        ),
        _leaf(
            "driver_previous_step_state.old_qair",
            3,
            component="driver_previous_step_state",
            family="driver",
        ),
        _leaf(
            "diffuco_previous_step_state.rveget",
            4,
            component="diffuco_previous_step_state",
            family="diffuco_enerbil",
        ),
        _leaf(
            "daily_interface.gpp_daily",
            5,
            component=None,
            family="daily_interface",
        ),
    ]
    contract = {
        "fast_day_target_width": len(fast_leaves),
        "state_leaves": state_leaves,
        "fast_day_target_leaves": fast_leaves,
    }

    report = build_ledger(contract)
    by_key = {item["key"]: item for item in report["leaves"]}

    assert report["residual_representation"] == {
        "same_owner_baseline_columns": 3,
        "mean_centered_columns": 3,
        "note": (
            "A same-owner baseline is a residual representation; it does not imply numeric persistence across the day."
        ),
    }
    assert by_key["driver_previous_step_state.fluxsens"]["coverage_status"] == ("semantic_collision")
    assert by_key["hydrol_previous_step_state.mc"]["source_kind"] == ("conservative_water_endpoint")
    assert by_key["thermosoil_previous_step_state.ptn"]["source_kind"] == ("bounded_thermal_endpoint")
    assert by_key["driver_previous_step_state.old_qair"]["coverage_status"] == ("exact_derivation")
    assert by_key["diffuco_previous_step_state.rveget"]["dynamic_undefined_columns"] == 1
    assert by_key["daily_interface.gpp_daily"]["source_outputs"] == ("daily.fast_gpp",)
