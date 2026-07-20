from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.domain import (
    read_annual_co2,
    read_domain_grid,
    read_forcing_model_step_cached,
    read_forcing_first_step,
    read_forcing_step,
    read_water_table_sequences,
)
from jax_orchidee.trace.server_1961 import find_server_record


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
DIFFUCO_TRACE_ROOT = ROOT / "outputs" / "server_1961_diffuco_active_after_main_trace_20260624_2244" / "traces"


def test_1961_domain_selects_single_nav_lon_lat_point():
    grid = read_domain_grid(CONFIG, year=1961)

    assert grid.iim == 1
    assert grid.jjm == 1
    assert grid.nbindex == 1
    assert np.array_equal(grid.kindex, np.asarray([1], dtype=np.int32))
    assert np.allclose(grid.lon, [[109.0]])
    assert np.allclose(grid.lat, [[21.0]])
    assert np.allclose(grid.lalo, [[21.0, 109.0]])
    assert np.array_equal(grid.forcing_indices_zero_based, np.asarray([[144, 34]], dtype=np.int32))

    # This is the NetCDF nav_lon/nav_lat-selected value. An older local audit
    # reported contfrac=1.0 from dimension indices rather than geolocated nav
    # coordinates; the Fortran path follows nav_lon/nav_lat.
    assert np.allclose(grid.contfrac, [[0.5625]])
    assert np.allclose(grid.contfrac_land, [0.5625])
    assert np.allclose(grid.area, [12345139970.1911])
    assert np.allclose(grid.resolution, [[111106.370838091, 111111.0]])
    assert np.array_equal(grid.neighbours, np.full((1, 8), -1, dtype=np.int32))


def test_reference_run_def_domain_override_selects_reference_history_point():
    grid = read_domain_grid(
        CONFIG,
        year=1961,
        domain_override={"west": -180.0, "east": -178.0, "south": -20.0, "north": -18.0},
    )

    assert grid.iim == 1
    assert grid.jjm == 1
    assert grid.nbindex == 1
    assert np.allclose(grid.lon, [[-179.0]])
    assert np.allclose(grid.lat, [[-19.0]])
    assert np.allclose(grid.lalo, [[-19.0, -179.0]])
    assert np.array_equal(grid.forcing_indices_zero_based, np.asarray([[0, 54]], dtype=np.int32))
    assert np.allclose(grid.contfrac, [[0.0625]])
    assert np.allclose(grid.contfrac_land, [0.0625])
    assert np.allclose(grid.area, [12345148958.8051])
    assert np.allclose(grid.resolution, [[111106.45173569776, 111110.99999999987]])


def test_1961_first_forcing_step_reads_active_variables_at_domain_point():
    grid = read_domain_grid(CONFIG, year=1961)
    forcing = read_forcing_first_step(CONFIG, domain=grid, year=1961)

    assert forcing.tstep == 0
    assert np.allclose(forcing.Tair, [[291.3238830566406]])
    assert np.allclose(forcing.PSurf, [[98422.8671875]])
    assert np.allclose(forcing.Qair, [[0.008104387670755386]])
    assert np.allclose(forcing.Wind_E, [[-3.8484580516815186]])
    assert np.allclose(forcing.Wind_N, [[-0.8622167110443115]])
    assert np.allclose(forcing.Rainf, [[0.0]])
    assert np.allclose(forcing.Snowf, [[0.0]])
    assert np.allclose(forcing.SWdown, [[362.3620910644531]])
    assert np.allclose(forcing.LWdown, [[338.0463562011719]])
    assert np.allclose(forcing.Areas, [[46266785792.0]])
    assert np.allclose(forcing.contfrac, [[0.5625]])

    assert forcing.Height_Lev1 == 2.0
    assert forcing.Height_Levuv == 10.0
    assert np.allclose(forcing.temp_air, forcing.Tair)
    assert np.allclose(forcing.pb, forcing.PSurf)
    assert np.allclose(forcing.qair, forcing.Qair)
    assert np.allclose(forcing.u, forcing.Wind_N)
    assert np.allclose(forcing.v, forcing.Wind_E)
    assert np.allclose(forcing.precip_rain, forcing.Rainf)
    assert np.allclose(forcing.precip_snow, forcing.Snowf)
    assert np.allclose(forcing.swdown, forcing.SWdown)
    assert np.allclose(forcing.lwdown, forcing.LWdown)
    assert np.allclose(forcing.zlev, [[2.0]])
    assert np.allclose(forcing.zlevuv, [[10.0]])


def test_1961_later_forcing_step_uses_same_fortran_land_point_mapping():
    grid = read_domain_grid(CONFIG, year=1961)
    forcing = read_forcing_step(CONFIG, domain=grid, year=1961, tstep=1)

    assert forcing.tstep == 1
    assert np.allclose(forcing.Tair, [[289.68145751953125]])
    assert np.allclose(forcing.PSurf, [[98808.9921875]])
    assert np.allclose(forcing.Qair, [[0.008919534273445606]])
    assert np.allclose(forcing.Wind_E, [[-4.260966777801514]])
    assert np.allclose(forcing.Wind_N, [[-0.2651669383049011]])
    assert np.allclose(forcing.Rainf, [[1.9264459751866525e-06]])
    assert np.allclose(forcing.Snowf, [[0.0]])
    assert np.allclose(forcing.SWdown, [[144.6377716064453]])
    assert np.allclose(forcing.LWdown, [[357.85186767578125]])
    assert np.allclose(forcing.u, forcing.Wind_N)
    assert np.allclose(forcing.v, forcing.Wind_E)


def test_1961_model_step_swdown_matches_active_fortran_solarang_trace():
    grid = read_domain_grid(CONFIG, year=1961)
    forcing = read_forcing_model_step_cached(
        CONFIG,
        domain=grid,
        year=1961,
        model_tstep=644,
        split=12,
        nb_spread=6,
    )
    trace = find_server_record(
        "diffuco_trans_co2",
        tag="after_diffuco_trans_co2_pft14",
        criteria={"kjit": 645, "ji": 1, "jv": 14},
        root=DIFFUCO_TRACE_ROOT,
        scan_limit=None,
    )

    # readdim2.f90::forcing_read_interpol lines 1526-1550 and 1620-1639
    # redistribute SWdown through solar.f90::solarang. The remaining tolerance
    # is the observed cross-runtime trigonometric roundoff, not a process gap.
    np.testing.assert_allclose(forcing.swdown.reshape(-1)[0], trace["swdown"], rtol=0.0, atol=2.3e-7)


def test_1961_forcing_step_rejects_out_of_range_timestep():
    grid = read_domain_grid(CONFIG, year=1961)
    with np.testing.assert_raises_regex(IndexError, "outside forcing range"):
        read_forcing_step(CONFIG, domain=grid, year=1961, tstep=1460)


def test_1961_co2_and_water_table_sequences_match_local_inputs():
    assert read_annual_co2(CONFIG, 1961) == 317.27

    sequences = read_water_table_sequences(CONFIG)
    assert sequences.positive.shape == (17520,)
    assert sequences.differential.shape == (17520,)
    assert np.allclose(sequences.positive[:5], [308.33, 329.00, 349.67, 347.92, 346.17])
    assert np.allclose(sequences.differential[:5], [44.83, 20.67, 20.67, -1.75, -1.75])
