from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from jax_orchidee.driver.domain import read_domain_grid, read_forcing_step_cached
from jax_orchidee.driver.forcing_readers import (
    ForcingDispatchContext,
    ForcingFields,
    VerticalForcingOptions,
    cyclic_forcing_record,
    forcing_grid,
    forcing_just_read,
    forcing_landind,
    forcing_read,
    route_forcing_landpoints,
)


def _variables(iim: int = 3, jjm: int = 2, ttm: int = 3) -> dict[str, np.ndarray]:
    base = np.arange(iim * jjm * ttm, dtype=np.float64).reshape(iim, jjm, ttm, order="F")
    names = [
        "Tair", "Snowf", "Rainf", "Tmin", "precip", "SWdown", "LWdown", "PSurf", "Qair",
        "Wind_N", "Wind_E", "Wind", "Levels", "Levels_uv", "levels", "SWnet", "Eair",
        "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy",
    ]
    out = {name: base + 10.0 * number for number, name in enumerate(names)}
    out["Tair"] = out["Tair"] + 280.0
    out["Tmin"] = out["Tmin"] + 270.0
    out["PSurf"] = out["PSurf"] + 100000.0
    return out


def _fixed() -> VerticalForcingOptions:
    return VerticalForcingOptions(zheight=True, zlev_fixed=2.0, zlevuv_fixed=10.0)


def test_cyclic_record_uses_fortran_mod_semantics() -> None:
    assert cyclic_forcing_record(0, 4) == 0
    assert [cyclic_forcing_record(value, 4) for value in (1, 2, 4, 5, 8, 9)] == [1, 2, 4, 1, 4, 1]


def test_just_read_extracts_record_then_fortran_zoom_and_wind_components() -> None:
    variables = _variables()
    result = forcing_just_read(
        variables, ttm=3, itb=2, ite=2, vertical=_fixed(), daily_interpol=False,
        wind_n_exists=True, is_watchout=False, i_index=np.array([2, 0]), j_index=np.array([1]),
    )
    np.testing.assert_array_equal(result.tair, variables["Tair"][np.ix_([2, 0], [1], [1])][:, :, 0])
    np.testing.assert_array_equal(result.u, variables["Wind_N"][np.ix_([2, 0], [1], [1])][:, :, 0])
    np.testing.assert_array_equal(result.v, variables["Wind_E"][np.ix_([2, 0], [1], [1])][:, :, 0])
    np.testing.assert_array_equal(result.zlev, np.full((2, 1), 2.0))
    np.testing.assert_array_equal(result.zlev_uv, np.full((2, 1), 10.0))
    assert result.SWnet is None


def test_just_read_daily_and_single_wind_preserve_source_assignments() -> None:
    result = forcing_just_read(
        _variables(), ttm=3, itb=1, ite=1, vertical=replace(_fixed(), zsamelev_uv=True),
        daily_interpol=True, wind_n_exists=False, is_watchout=False,
    )
    assert result.snowf is None
    np.testing.assert_array_equal(result.v, np.zeros_like(result.u))
    np.testing.assert_array_equal(result.zlev_uv, result.zlev)


def test_hybrid_levels_and_missing_value_branch_match_formula() -> None:
    variables = _variables(iim=2, jjm=1, ttm=1)
    variables["Tair"][:, :, 0] = [[300.0], [1.0e20]]
    variables["PSurf"][:, :, 0] = [[100000.0], [90000.0]]
    vertical = VerticalForcingOptions(
        zhybrid=True, zhybrid_a=1000.0, zhybrid_b=0.1, zhybriduv_a=500.0,
        zhybriduv_b=0.05, cte_molr=287.0, cte_grav=9.81, val_exp=1.0e19,
    )
    result = forcing_just_read(
        variables, ttm=1, itb=1, ite=1, vertical=vertical, daily_interpol=False,
        wind_n_exists=True, is_watchout=False,
    )
    density = 100000.0 / (287.0 * 300.0)
    assert result.zlev[0, 0] == pytest.approx((100000.0 - (1000.0 + 0.1 * 100000.0)) / (density * 9.81))
    assert result.zlev[1, 0] == 0.0
    assert result.zlev_uv[1, 0] == 0.0


def test_levels_and_watchout_overwrite_vertical_fields_and_read_all_payloads() -> None:
    variables = _variables(ttm=1)
    result = forcing_just_read(
        variables, ttm=1, itb=1, ite=1, vertical=VerticalForcingOptions(zlevels=True),
        daily_interpol=False, wind_n_exists=True, is_watchout=True,
    )
    np.testing.assert_array_equal(result.zlev, variables["levels"][:, :, 0])
    np.testing.assert_array_equal(result.zlev_uv, result.zlev)
    for name in ("SWnet", "Eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy"):
        np.testing.assert_array_equal(getattr(result, name), variables[name][:, :, 0])


def test_missing_variable_and_invalid_record_are_rejected_without_defaults() -> None:
    variables = _variables()
    variables.pop("Qair")
    with pytest.raises(KeyError, match="Qair"):
        forcing_just_read(
            variables, ttm=3, itb=1, ite=1, vertical=_fixed(), daily_interpol=False,
            wind_n_exists=True, is_watchout=False,
        )
    with pytest.raises(ValueError, match="itb == ite"):
        forcing_just_read(
            _variables(), ttm=3, itb=1, ite=2, vertical=_fixed(), daily_interpol=False,
            wind_n_exists=True, is_watchout=False,
        )


def test_landind_celsius_kelvin_and_column_major_routes() -> None:
    celsius = np.array([[10.0, 999999.0], [20.0, 30.0], [999999.0, 40.0]])
    selected = forcing_landind(celsius)
    np.testing.assert_array_equal(selected.kindex, [1, 2, 5, 6])
    assert (selected.nbindex, selected.i_test, selected.j_test) == (4, 3, 2)
    np.testing.assert_array_equal(route_forcing_landpoints(celsius, selected.kindex), [10.0, 20.0, 30.0, 40.0])

    kelvin = forcing_landind(np.array([[280.0, 999999.0], [510.0, 290.0]]))
    np.testing.assert_array_equal(kelvin.kindex, [1, 4])


def test_forcing_grid_constructs_weather_and_zooms_interpol_coordinates() -> None:
    lon, lat = forcing_grid(
        iim=2, jjm=2, weathergen=True, interpol=False, init_f=True,
        limit_west=0.0, limit_east=4.0, limit_north=4.0, limit_south=0.0,
        merid_res=2.0, zonal_res=2.0,
    )
    np.testing.assert_array_equal(lon, [[1.0, 1.0], [3.0, 3.0]])
    np.testing.assert_array_equal(lat, [[3.0, 1.0], [3.0, 1.0]])

    full_lon = np.arange(12).reshape(3, 4)
    full_lat = full_lon + 100
    zoom_lon, zoom_lat = forcing_grid(
        iim=2, jjm=2, weathergen=False, interpol=True, init_f=True,
        lon_full=full_lon, lat_full=full_lat, i_index=np.array([2, 0]), j_index=np.array([3, 1]),
    )
    np.testing.assert_array_equal(zoom_lon, full_lon[np.ix_([2, 0], [3, 1])])
    np.testing.assert_array_equal(zoom_lat, full_lat[np.ix_([2, 0], [3, 1])])

    local_lon, local_lat = forcing_grid(
        iim=3, jjm=2, weathergen=True, interpol=False, init_f=False,
        lon_global=full_lon, lat_global=full_lat, local_j_begin=2, local_j_end=3,
    )
    np.testing.assert_array_equal(local_lon, full_lon[:, 1:3])
    np.testing.assert_array_equal(local_lat, full_lat[:, 1:3])


def test_forcing_read_dispatches_owner_and_computes_nonwatchout_eair() -> None:
    source = forcing_just_read(
        _variables(ttm=1), ttm=1, itb=1, ite=1, vertical=_fixed(), daily_interpol=False,
        wind_n_exists=True, is_watchout=False,
    )
    called: list[ForcingDispatchContext] = []

    def weather(context: ForcingDispatchContext) -> ForcingFields:
        called.append(context)
        return source

    context = ForcingDispatchContext(0, 7, 0, True, np.zeros((3, 2)))
    result = forcing_read(
        context=context, interpol=False, weathergen=True, interpol_owner=None,
        weathergen_owner=weather, is_watchout=False, cp_air=1004.0, cte_grav=9.81,
        val_exp=1.0e19,
    )
    assert called[0].itauin == 7
    np.testing.assert_array_equal(called[0].fcontfrac_seed, np.ones((3, 2)))
    np.testing.assert_allclose(result.Eair, 1004.0 * source.tair + 9.81 * source.zlev)


def test_forcing_read_rejects_missing_mode_and_missing_process_owner() -> None:
    context = ForcingDispatchContext(1, 1, 1, False)
    with pytest.raises(ValueError, match="neither"):
        forcing_read(
            context=context, interpol=False, weathergen=False, interpol_owner=None,
            weathergen_owner=None, is_watchout=False, cp_air=1.0, cte_grav=1.0, val_exp=1.0,
        )
    with pytest.raises(ValueError, match="forcing_read_interpol owner"):
        forcing_read(
            context=context, interpol=True, weathergen=False, interpol_owner=None,
            weathergen_owner=None, is_watchout=False, cp_air=1.0, cte_grav=1.0, val_exp=1.0,
        )


def test_forcing_read_requires_prior_eair_at_invalid_tair_cells() -> None:
    source = forcing_just_read(
        _variables(ttm=1), ttm=1, itb=1, ite=1, vertical=_fixed(), daily_interpol=False,
        wind_n_exists=True, is_watchout=False,
    )
    source = replace(source, tair=np.full_like(source.tair, 1.0e20))
    with pytest.raises(ValueError, match="prior Eair state"):
        forcing_read(
            context=ForcingDispatchContext(1, 1, 1, False), interpol=True, weathergen=False,
            interpol_owner=lambda _: source, weathergen_owner=None, is_watchout=False,
            cp_air=1004.0, cte_grav=9.81, val_exp=1.0e19,
        )


def _write_asymmetric_netcdf(path: Path) -> None:
    t, y, x = np.arange(3)[:, None, None], np.arange(2)[None, :, None], np.arange(3)[None, None, :]
    def values(offset: float) -> np.ndarray:
        return offset + 100.0 * t + 10.0 * y + x
    xr.Dataset(
        {
            "nav_lon": (("y", "x"), [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
            "nav_lat": (("y", "x"), [[50.0, 50.0, 50.0], [40.0, 40.0, 40.0]]),
            "Tair": (("tstep", "y", "x"), values(200.0)),
            "Snowf": (("tstep", "y", "x"), values(20.0)),
            "Rainf": (("tstep", "y", "x"), values(30.0)),
            "SWdown": (("tstep", "y", "x"), values(40.0)),
            "LWdown": (("tstep", "y", "x"), values(50.0)),
            "PSurf": (("tstep", "y", "x"), values(100000.0)),
            "Qair": (("tstep", "y", "x"), values(0.01)),
            "Wind_N": (("tstep", "y", "x"), values(60.0)),
            "Wind_E": (("tstep", "y", "x"), values(70.0)),
            "contfrac": (("y", "x"), [[0.5, 0.0, 0.2], [0.0, 0.7, 0.9]]),
            "Areas": (("y", "x"), [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]]),
            "Height_Lev1": ((), 2.0),
            "Height_Levuv": ((), 10.0),
        }
    ).to_netcdf(path)


def test_asymmetric_netcdf_fortran_order_and_domain_cache_crosscheck(tmp_path: Path) -> None:
    forcing_path = tmp_path / "forcing_1961.nc"
    _write_asymmetric_netcdf(forcing_path)
    config_path = tmp_path / "case.yaml"
    pattern = str(tmp_path / "forcing_{year}.nc").replace(chr(92), "/")
    config_path.write_text(
        "drivers:\n  atmospheric_forcing:\n"
        f"    file_pattern: '{pattern}'\n"
        "domain:\n  west: 0\n  east: 40\n  south: 30\n  north: 60\n",
        encoding="ascii",
    )
    domain = read_domain_grid(config_path, year=1961)
    cached = read_forcing_step_cached(config_path, domain=domain, year=1961, tstep=1)
    result = forcing_read(
        forcing_path, itauin=2, itau_split=1, interpol=True, weathergen=False,
        daily_interpol=False, vertical=None, lrstread=False,
    )
    np.testing.assert_array_equal(result.land.kindex, domain.kindex)
    np.testing.assert_array_equal(result.land.ij_zero_based, domain.forcing_indices_zero_based)
    np.testing.assert_allclose(result.land_fields["tair"][:, None], cached.Tair)
    np.testing.assert_allclose(result.land_fields["u"][:, None], cached.Wind_N)
    np.testing.assert_allclose(result.land_fields["v"][:, None], cached.Wind_E)
    np.testing.assert_allclose(result.contfrac_land[:, None], cached.contfrac)
    np.testing.assert_allclose(result.areas_land[:, None], cached.Areas)


def test_source_owner_ledger_records_exact_43_arm_batch() -> None:
    root = Path(__file__).resolve().parents[2]
    ledger = yaml.safe_load(
        (root / "docs/source_audits/pft14_source_owners_forcing_readers.yaml").read_text(encoding="utf-8")
    )
    assert ledger["procedure_count"] == 4
    assert ledger["control_flow_arms"] == 43
    assert {
        entry["fortran_subroutine"]: entry["arm_count"] for entry in ledger["entries"]
    } == {"forcing_read": 6, "forcing_just_read": 23, "forcing_grid": 2, "forcing_landind": 12}
