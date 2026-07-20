from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from jax_orchidee.driver.domain import (
    _fortran_time_zone,
    read_domain_grid,
    read_forcing_model_step_cached,
    read_forcing_step,
    read_forcing_step_cached,
)


def _write_multiland_forcing(path: Path) -> None:
    y = np.asarray([0, 1], dtype=np.int32)
    x = np.asarray([0, 1], dtype=np.int32)
    tstep = np.asarray([0, 1], dtype=np.int32)
    nav_lon = np.asarray([[10.0, 11.0], [10.0, 11.0]], dtype=np.float64)
    nav_lat = np.asarray([[50.0, 50.0], [51.0, 51.0]], dtype=np.float64)
    contfrac = np.asarray([[0.5, 0.0], [0.25, 0.75]], dtype=np.float64)
    areas = np.asarray([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float64)

    def forcing_values(base: float) -> np.ndarray:
        values = np.zeros((2, 2, 2), dtype=np.float64)
        for it in range(2):
            for iy in range(2):
                for ix in range(2):
                    values[it, iy, ix] = base + it * 100.0 + iy * 10.0 + ix
        return values

    ds = xr.Dataset(
        data_vars={
            "nav_lon": (("y", "x"), nav_lon),
            "nav_lat": (("y", "x"), nav_lat),
            "contfrac": (("y", "x"), contfrac),
            "Areas": (("y", "x"), areas),
            "Height_Lev1": ((), 2.0),
            "Height_Levuv": ((), 10.0),
            "Tair": (("tstep", "y", "x"), forcing_values(1000.0)),
            "PSurf": (("tstep", "y", "x"), forcing_values(2000.0)),
            "Qair": (("tstep", "y", "x"), forcing_values(3000.0)),
            "Wind_E": (("tstep", "y", "x"), forcing_values(4000.0)),
            "Wind_N": (("tstep", "y", "x"), forcing_values(5000.0)),
            "Rainf": (("tstep", "y", "x"), forcing_values(6000.0)),
            "Snowf": (("tstep", "y", "x"), forcing_values(7000.0)),
            "SWdown": (("tstep", "y", "x"), forcing_values(8000.0)),
            "LWdown": (("tstep", "y", "x"), forcing_values(9000.0)),
        },
        coords={"tstep": tstep, "y": y, "x": x},
    )
    ds.to_netcdf(path)


def _write_config(path: Path, forcing_pattern: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "drivers:",
                "  atmospheric_forcing:",
                f"    file_pattern: \"{str(forcing_pattern).replace(chr(92), '/')}\"",
                "domain:",
                "  west: 9.5",
                "  east: 11.5",
                "  south: 49.5",
                "  north: 51.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_fortran_time_zone_preserves_first_match_exit_and_integer_truncation():
    lon = np.asarray([[-180.9], [-172.5], [-8.9], [0.0], [173.0], [180.0]])
    zone, local = _fortran_time_zone(lon, gmt=23.5)
    np.testing.assert_array_equal(zone, [1, 2, 12, 13, 1, 1])
    np.testing.assert_allclose(local, [11.5, 12.5, 22.5, 23.5, 11.5, 11.5])


def test_fortran_time_zone_rejects_source_unassigned_longitude():
    with np.testing.assert_raises_regex(ValueError, "outside the source search range"):
        _fortran_time_zone(np.asarray([[188.0]]), gmt=0.0)


def test_read_forcing_step_selects_all_land_points_in_fortran_order(tmp_path):
    """Fortran: readdim2.f90::forcing_landind lines 1899-1968."""

    forcing_path = tmp_path / "forcing_1961.nc"
    config_path = tmp_path / "case.yaml"
    _write_multiland_forcing(forcing_path)
    _write_config(config_path, tmp_path / "forcing_{year}.nc")

    grid = read_domain_grid(config_path, year=1961)
    forcing = read_forcing_step(config_path, domain=grid, year=1961, tstep=1)

    assert grid.nbindex == 3
    np.testing.assert_array_equal(grid.kindex, np.asarray([1, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(
        grid.forcing_indices_zero_based,
        np.asarray([[0, 0], [0, 1], [1, 1]], dtype=np.int32),
    )
    np.testing.assert_allclose(forcing.Tair, np.asarray([[1100.0], [1110.0], [1111.0]]))
    np.testing.assert_allclose(forcing.PSurf, np.asarray([[2100.0], [2110.0], [2111.0]]))
    np.testing.assert_allclose(forcing.Wind_N, np.asarray([[5100.0], [5110.0], [5111.0]]))
    np.testing.assert_allclose(forcing.Wind_E, np.asarray([[4100.0], [4110.0], [4111.0]]))
    np.testing.assert_allclose(forcing.Areas, np.asarray([[1000.0], [3000.0], [4000.0]]))
    np.testing.assert_allclose(forcing.contfrac, np.asarray([[0.5], [0.25], [0.75]]))
    np.testing.assert_allclose(forcing.zlev, np.full((3, 1), 2.0))
    np.testing.assert_allclose(forcing.zlevuv, np.full((3, 1), 10.0))


def test_read_forcing_step_cached_matches_uncached_multiland_selection(tmp_path):
    """Fortran: raw slab values are unchanged by JAX IO cache strategy."""

    forcing_path = tmp_path / "forcing_1961.nc"
    config_path = tmp_path / "case.yaml"
    _write_multiland_forcing(forcing_path)
    _write_config(config_path, tmp_path / "forcing_{year}.nc")

    grid = read_domain_grid(config_path, year=1961)
    uncached = read_forcing_step(config_path, domain=grid, year=1961, tstep=1)
    cached = read_forcing_step_cached(config_path, domain=grid, year=1961, tstep=1)

    for name in ("Tair", "PSurf", "Qair", "Wind_E", "Wind_N", "Rainf", "Snowf", "SWdown", "LWdown", "Areas", "contfrac"):
        np.testing.assert_allclose(getattr(cached, name), getattr(uncached, name))
    np.testing.assert_allclose(cached.zlev, uncached.zlev)
    np.testing.assert_allclose(cached.zlevuv, uncached.zlevuv)


def test_read_forcing_model_step_cached_handles_multiland_no_solar_split(tmp_path):
    """Fortran: readdim2.f90::forcing_read_interpol lines 1450-1661."""

    forcing_path = tmp_path / "forcing_1961.nc"
    config_path = tmp_path / "case.yaml"
    _write_multiland_forcing(forcing_path)
    _write_config(config_path, tmp_path / "forcing_{year}.nc")

    grid = read_domain_grid(config_path, year=1961)
    forcing = read_forcing_model_step_cached(
        config_path,
        domain=grid,
        year=1961,
        model_tstep=0,
        split=2,
        nb_spread=1,
    )

    np.testing.assert_allclose(forcing.Tair, np.asarray([[1050.0], [1060.0], [1061.0]]))
    np.testing.assert_allclose(forcing.PSurf, np.asarray([[2050.0], [2060.0], [2061.0]]))
    np.testing.assert_allclose(forcing.Rainf, np.asarray([[12000.0], [12020.0], [12022.0]]))
    np.testing.assert_allclose(forcing.Snowf, np.asarray([[14000.0], [14020.0], [14022.0]]))
    np.testing.assert_allclose(forcing.SWdown, np.asarray([[8000.0], [8010.0], [8011.0]]))
    np.testing.assert_allclose(forcing.LWdown, np.asarray([[9050.0], [9060.0], [9061.0]]))


def test_read_forcing_model_step_cached_handles_multiland_solarang_compression(tmp_path):
    """SWdown redistribution is computed on the full zoom grid before land selection."""

    forcing_path = tmp_path / "forcing_1961.nc"
    config_path = tmp_path / "case.yaml"
    _write_multiland_forcing(forcing_path)
    _write_config(config_path, tmp_path / "forcing_{year}.nc")

    grid = read_domain_grid(config_path, year=1961)
    forcing = read_forcing_model_step_cached(
        config_path,
        domain=grid,
        year=1961,
        model_tstep=0,
        split=12,
        nb_spread=6,
    )

    assert forcing.SWdown.shape == (3, 1)
    assert np.all(np.isfinite(forcing.SWdown))
    assert np.all(forcing.SWdown <= 2000.0)
