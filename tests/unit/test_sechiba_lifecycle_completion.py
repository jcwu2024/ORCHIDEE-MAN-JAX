from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from jax_orchidee.sechiba.lifecycle_completion import (
    HistorySwitches,
    SechibaFinalizeSwitches,
    intersurf_initialize_2d,
    intersurf_main_2d,
    intersurf_time,
    ioipslctrl_history,
    sechiba_finalize,
)


def test_intersurf_time_gregorian_initializes_once_and_routes_calendar_state():
    state = intersurf_time(
        istp=1,
        date0=2451544.5,
        dt_sechiba=3600,
        calendar="gregorian",
        one_year=365 * 86400,
        one_day=86400,
    )
    assert (state.year, state.month, state.day, state.second) == (2000, 1, 1, 3600)
    assert state.year_spread == 1
    later = intersurf_time(
        istp=24,
        date0=2451544.5,
        dt_sechiba=3600,
        calendar="gregorian",
        one_year=999,
        one_day=86400,
        previous=state,
    )
    assert later.year_length == state.year_length
    assert (later.month, later.day) == (1, 2)


def test_non_gregorian_requires_ioipsl_ymds_primitive_and_computes_derived_values():
    with pytest.raises(ValueError, match="non_gregorian_ymds"):
        intersurf_time(
            istp=1,
            date0=0,
            dt_sechiba=86400,
            calendar="360d",
            one_year=360 * 86400,
            one_day=86400,
        )
    state = intersurf_time(
        istp=31,
        date0=100,
        dt_sechiba=86400,
        calendar="360d",
        one_year=360 * 86400,
        one_day=86400,
        non_gregorian_ymds=lambda _step, _dt: (2, 2, 2, 0),
    )
    assert state.month_length == 30
    assert state.julian_diff == 31
    assert state.year_spread == pytest.approx(360 * 86400 / 365.2425)


def test_history_paper_xios_route_disables_all_ioipsl_streams_before_transport():
    calls = []
    result = ioipslctrl_history(
        lon=np.array([[1.0], [2.0]]),
        lat=np.array([[3.0], [3.0]]),
        dt=1800,
        one_day=86400,
        switches=HistorySwitches(
            xios_orchidee_ok=True,
            ok_stomate=True,
            stomate_hist_days=-1,
            stomate_ipcc_hist_days=30,
        ),
        transport=calls.append,
    )
    assert calls == []
    assert [stream.active for stream in result.streams] == [False] * 4
    assert result.handles == {
        "sechiba": -1,
        "sechiba2": -1,
        "stomate": -1,
        "stomate_ipcc": -1,
    }


def test_history_routes_frequency_levels_grid_and_scientific_selections():
    calls = []
    result = ioipslctrl_history(
        lon=np.array([[10.0, 10.0], [20.0, 20.0]]),
        lat=np.array([[40.0, 50.0], [40.0, 50.0]]),
        dt=3600,
        one_day=86400,
        switches=HistorySwitches(
            xios_orchidee_ok=False,
            write_step=86400,
            sechiba_histfile2=True,
            sechiba_histlevel=3,
            sechiba_histlevel2=2,
            stomate_hist_days=-1,
            stomate_ipcc_hist_days=30,
            river_routing=True,
            do_floodplains=True,
        ),
        transport=lambda request: calls.append(request) or len(calls),
    )
    assert [stream.name for stream in calls] == [
        "sechiba",
        "sechiba2",
        "stomate",
        "stomate_ipcc",
    ]
    assert result.handles == {
        "sechiba": 1,
        "sechiba2": 2,
        "stomate": 3,
        "stomate_ipcc": 4,
    }
    primary = result.streams[0]
    np.testing.assert_array_equal(primary.longitude, [10, 20])
    np.testing.assert_array_equal(primary.latitude, [40, 50])
    assert primary.operations["ave"][2:4] == ("ave(scatter(X))", "never")
    assert primary.selections["river_routing"] and primary.selections["do_floodplains"]


def test_history_level_eleven_is_valid_and_nonroot_never_opens_transport():
    calls = []
    result = ioipslctrl_history(
        lon=np.ones((1, 1)),
        lat=np.ones((1, 1)),
        dt=1800,
        one_day=86400,
        switches=HistorySwitches(
            xios_orchidee_ok=False,
            sechiba_histlevel=11,
            is_omp_root=False,
            stomate_hist_days=0,
        ),
        transport=calls.append,
    )
    assert result.streams[0].active
    assert result.streams[0].operations["ave"][-1] == "ave(scatter(X))"
    assert calls == []
    assert all(handle == -1 for handle in result.handles.values())


def _fields(iim=2, jjm=2):
    base = np.arange(iim * jjm, dtype=float).reshape(iim, jjm)
    names = (
        "u",
        "v",
        "zlev",
        "qair",
        "precip_rain",
        "precip_snow",
        "lwdown",
        "swnet",
        "swdown",
        "temp_air",
        "epot_air",
        "ccanopy",
        "petAcoef",
        "peqAcoef",
        "petBcoef",
        "peqBcoef",
        "cdrag",
        "pb",
        "coszang",
    )
    return {name: base + index for index, name in enumerate(names)}


def _owner(request):
    n = len(request.contfrac)
    return {
        "z0m": np.arange(n) + 1.0,
        "coastalflow": np.full((n, 2), 20.0),
        "riverflow": np.full((n, 2), 40.0),
        "tsol_rad": np.full(n, 280.0),
        "vevapp": np.full(n, 10.0),
        "temp_sol_new": np.full(n, 281.0),
        "qsurf": np.full(n, 0.01),
        "albedo": np.full((n, 2), 0.2),
        "fluxsens": np.full(n, 5.0),
        "fluxlat": np.full(n, 6.0),
        "emis": np.full(n, 0.98),
        "cdrag": np.full(n, 0.1),
    }


@pytest.mark.parametrize("initializer", [True, False])
def test_2d_lifecycle_computes_compression_units_scatter_and_calendar_step(initializer):
    fields = _fields()
    fn = intersurf_initialize_2d if initializer else intersurf_main_2d
    result = fn(
        kjit=5,
        iim=2,
        jjm=2,
        kindex=np.array([1, 4]),
        xrdt=10.0,
        date0_shifted=100.0,
        itau_offset=2,
        lon=np.array([[1, 2], [3, 4]]),
        lat=np.array([[41, 42], [43, 44]]),
        zcontfrac=np.array([[0.1, 0.2], [0.3, 0.4]]),
        fields=fields,
        owner=_owner,
        undef_sechiba=-999.0,
    )
    np.testing.assert_array_equal(
        result.request.forcing["precip_rain"],
        fields["precip_rain"][[0, 1], [0, 1]] * 10,
    )
    np.testing.assert_array_equal(result.request.latitude_longitude, [[41, 1], [44, 4]])
    assert result.outputs["vevapp"][0, 0] == 1
    assert result.outputs["coastalflow"][1, 1, 0] == 2
    assert result.outputs["z0m"][0, 1] == -999
    assert result.calendar_step == (8 if initializer else 7)


def _finalize_state():
    return {
        "rsol": np.ones(2),
        "reinfiltration": np.ones((2, 2)),
        "returnflow": np.ones((2, 2)),
        "sed_deposition": np.ones((2, 3)),
        "poc_deposition": np.ones((2, 3)),
        "sed_deposition_d": np.ones(2),
        "poc_deposition_d": np.ones((2, 3)),
        "flood_res": np.ones(2),
        "flood_frac": np.ones(2),
        "streamfl_frac": np.ones(2),
        "fastr": np.ones(2),
        "irrigation": np.ones((2, 2)),
        "co2_flux": np.array([[9, 2, 3], [8, 4, 5.0]]),
        "veget_max": np.array([[0.5, 0.2, 0.3], [0.1, 0.4, 0.5]]),
    }


def test_finalize_orders_owners_applies_no_routing_writeback_then_serializes():
    calls = []
    owners = {
        name: (lambda state, name=name: calls.append(name) or name)
        for name in (
            "diffuco",
            "enerbil",
            "hydrol",
            "condveg",
            "thermosoil",
            "slowproc",
        )
    }
    result = sechiba_finalize(
        state=_finalize_state(),
        finalizers=owners,
        restart_transport=lambda state: (
            calls.append("write") or state["netco2flux"].copy()
        ),
    )
    assert calls == [
        "diffuco",
        "enerbil",
        "hydrol",
        "condveg",
        "thermosoil",
        "slowproc",
        "write",
    ]
    np.testing.assert_array_equal(result.state["reinfiltration"], 0)
    np.testing.assert_array_equal(result.state["rsol"], -1)
    np.testing.assert_allclose(result.netco2flux, [1.3, 4.1])


def test_finalize_routes_multigrid_to_routing_owner_without_zeroing_state():
    state = _finalize_state()
    owners = {
        name: (lambda _state, name=name: name)
        for name in (
            "diffuco",
            "enerbil",
            "hydrol",
            "condveg",
            "thermosoil",
            "routing",
            "slowproc",
        )
    }
    result = sechiba_finalize(
        state=state,
        finalizers=owners,
        switches=SechibaFinalizeSwitches(river_routing=True, nbp_glo=2),
    )
    assert "routing_finalize" in result.process_order
    np.testing.assert_array_equal(result.state["reinfiltration"], 1)


def test_owned_source_ledger_accounts_for_exactly_68_arms():
    path = Path(
        "docs/source_audits/pft14_source_owners_sechiba_lifecycle_completion.yaml"
    )
    ledger = yaml.safe_load(path.read_text(encoding="ascii"))
    assert ledger["totals"]["arms"] == 68
    assert sum(entry["arm_count"] for entry in ledger["entries"]) == 68
    assert [entry["arm_count"] for entry in ledger["entries"]] == [42, 10, 6, 6, 4]
