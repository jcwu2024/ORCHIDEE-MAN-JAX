from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.season import (
    SEASON_MEMORY_PROVENANCE,
    SeasonAnnualState,
    SeasonBiometeorologyState,
    SeasonMemoryState,
    SeasonTimeScales,
    season_biometeorology_step,
    season_annual_step,
    season_memory_step,
    season_midwinter_solad,
    season_update_gdd_init_date,
)


def _state(npts=2, nvm=4, nsoil=3):
    base_moist_month = np.asarray([[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]], dtype=np.float64)[:npts, :nvm]
    base_moist_week = np.asarray([[0.3, 0.5, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]], dtype=np.float64)[:npts, :nvm]
    base_tmin = np.asarray([[2.0, 39.0, 0.0, 5.0], [0.0, 3.0, 10.0, 39.0]], dtype=np.float64)[:npts, :nvm]
    base_gdd = np.asarray([[9.0, 10.0, 20.0, 30.0], [7.0, 11.0, 21.0, 31.0]], dtype=np.float64)[:npts, :nvm]
    return SeasonMemoryState(
        moiavail_month=base_moist_month,
        moiavail_week=base_moist_week,
        t2m_longterm=np.asarray([300.0, 250.0], dtype=np.float64),
        tau_longterm=5.0,
        t2m_month=np.asarray([280.0, 260.0], dtype=np.float64),
        t2m_week=np.asarray([274.0, 272.0], dtype=np.float64),
        tseason=np.asarray([281.0, 282.0], dtype=np.float64),
        tseason_length=np.asarray([2.0, 3.0], dtype=np.float64),
        tseason_tmp=np.asarray([550.0, 810.0], dtype=np.float64),
        tmin_spring_time=base_tmin,
        onset_date=np.zeros((npts, nvm), dtype=np.float64),
        tsoil_month=np.arange(npts * nsoil, dtype=np.float64).reshape(npts, nsoil) + 270.0,
        soilhum_month=np.arange(npts * nsoil, dtype=np.float64).reshape(npts, nsoil) / 10.0 + 0.2,
        gdd_from_growthinit=base_gdd,
    )


def test_season_midwinter_gdd_init_date_uses_fortran_downward_solar_flux_branch():
    latitude = np.asarray([21.0], dtype=np.float64)
    solad_day1 = np.asarray(season_midwinter_solad(latitude, 1.0))

    np.testing.assert_allclose(solad_day1, [645.36478912], rtol=1e-10)

    current_lower = np.asarray([[172.0, 620.30564873]], dtype=np.float64)
    unchanged = season_update_gdd_init_date(current_lower, latitude=latitude, julian_diff=1.0)
    np.testing.assert_allclose(np.asarray(unchanged), current_lower)

    current_higher = np.asarray([[172.0, 700.0]], dtype=np.float64)
    updated = season_update_gdd_init_date(current_higher, latitude=latitude, julian_diff=1.0)
    np.testing.assert_allclose(np.asarray(updated[:, 0]), [1.0])
    np.testing.assert_allclose(np.asarray(updated[:, 1]), solad_day1)


def test_season_memory_step_matches_fortran_relaxations_and_ordered_daily_updates():
    state = _state()
    moiavail_daily = np.asarray([[0.8, 0.7, 0.6, 0.5], [0.4, 0.3, 0.2, 0.1]], dtype=np.float64)
    t2m_daily = np.asarray([276.0, 280.0], dtype=np.float64)
    tsoil_daily = np.asarray([[271.0, 272.0, 273.0], [274.0, 275.0, 276.0]], dtype=np.float64)
    soilhum_daily = np.asarray([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float64)
    begin_leaves = np.asarray([[False, True, False, False], [False, False, True, False]])
    when_growthinit = np.asarray([[0.0, 0.0, 2.0, 4.0], [0.0, 5.0, 0.0, 6.0]], dtype=np.float64)
    natural = np.asarray([True, False, True, False])
    leaf_tab = np.asarray([4, 1, 1, 2])
    pheno_type = np.asarray([0, 2, 2, 0])
    pft_to_mtc = np.asarray([1, 8, 6, 3])
    scales = SeasonTimeScales(tau_hum_month=20.0, tau_hum_week=7.0, tau_t2m_month=20.0, tau_t2m_week=7.0)

    result = season_memory_step(
        state,
        dt_days=1.0,
        end_of_year=False,
        moiavail_daily=moiavail_daily,
        t2m_daily=t2m_daily,
        tsoil_daily=tsoil_daily,
        soilhum_daily=soilhum_daily,
        begin_leaves=begin_leaves,
        julian_diff=123.0,
        when_growthinit=when_growthinit,
        natural=natural,
        leaf_tab=leaf_tab,
        pheno_type=pheno_type,
        pft_to_mtc=pft_to_mtc,
        time_scales=scales,
    ).state

    np.testing.assert_allclose(result.moiavail_month, (state.moiavail_month * 19.0 + moiavail_daily) / 20.0)
    np.testing.assert_allclose(result.moiavail_week, (state.moiavail_week * 6.0 + moiavail_daily) / 7.0)
    np.testing.assert_allclose(result.t2m_month, (state.t2m_month * 19.0 + t2m_daily) / 20.0)
    np.testing.assert_allclose(result.t2m_week, (state.t2m_week * 6.0 + t2m_daily) / 7.0)
    np.testing.assert_allclose(result.tsoil_month, (state.tsoil_month * 19.0 + tsoil_daily) / 20.0)
    np.testing.assert_allclose(result.soilhum_month, (state.soilhum_month * 19.0 + soilhum_daily) / 20.0)

    expected_tau = 6.0
    expected_longterm = (state.t2m_longterm * 5.0 + t2m_daily) / expected_tau
    expected_longterm = np.maximum(253.1, np.minimum(303.1, expected_longterm))
    assert result.tau_longterm == expected_tau
    np.testing.assert_allclose(result.t2m_longterm, expected_longterm)

    np.testing.assert_allclose(result.onset_date, np.where(begin_leaves, 123.0, 0.0))
    np.testing.assert_allclose(result.tseason_tmp, np.asarray([1098.0, 810.0]))
    np.testing.assert_allclose(result.tseason_length, np.asarray([4.0, 3.0]))

    expected_tmin = np.asarray([[2.0, 1.0, 0.0, 5.0], [0.0, 5.0, 1.0, 39.0]])
    np.testing.assert_allclose(result.tmin_spring_time, expected_tmin)

    expected_gdd = state.gdd_from_growthinit.copy()
    expected_gdd[:, 1] = [276.0 - 273.15, 11.0 + 280.0 - 273.15]
    expected_gdd[:, 2] = -9999.0
    expected_gdd[:, 3] = [30.0 + 276.0 - 273.15, 31.0 + 280.0 - 273.15]
    np.testing.assert_allclose(result.gdd_from_growthinit, expected_gdd)
    assert any("stomate_season.f90::season lines 577-772" in item for item in SEASON_MEMORY_PROVENANCE)


def test_season_memory_firstcall_initializes_unset_restart_memory_before_relaxation():
    npts, nvm, nsoil = 1, 3, 2
    zero_state = SeasonMemoryState(
        moiavail_month=np.zeros((npts, nvm), dtype=np.float64),
        moiavail_week=np.zeros((npts, nvm), dtype=np.float64),
        t2m_longterm=np.asarray([260.0], dtype=np.float64),
        tau_longterm=0.0,
        t2m_month=np.zeros(npts, dtype=np.float64),
        t2m_week=np.zeros(npts, dtype=np.float64),
        tseason=np.zeros(npts, dtype=np.float64),
        tseason_length=np.zeros(npts, dtype=np.float64),
        tseason_tmp=np.zeros(npts, dtype=np.float64),
        tmin_spring_time=np.zeros((npts, nvm), dtype=np.float64),
        onset_date=np.zeros((npts, nvm), dtype=np.float64),
        tsoil_month=np.zeros((npts, nsoil), dtype=np.float64),
        soilhum_month=np.zeros((npts, nsoil), dtype=np.float64),
        gdd_from_growthinit=np.ones((npts, nvm), dtype=np.float64),
    )
    moiavail_daily = np.asarray([[0.9, 0.8, 0.7]], dtype=np.float64)
    t2m_daily = np.asarray([274.15], dtype=np.float64)
    tsoil_daily = np.asarray([[270.0, 271.0]], dtype=np.float64)
    soilhum_daily = np.asarray([[0.4, 0.5]], dtype=np.float64)

    result = season_memory_step(
        zero_state,
        dt_days=1.0,
        end_of_year=False,
        moiavail_daily=moiavail_daily,
        t2m_daily=t2m_daily,
        tsoil_daily=tsoil_daily,
        soilhum_daily=soilhum_daily,
        begin_leaves=np.zeros((npts, nvm), dtype=bool),
        julian_diff=1.0,
        when_growthinit=np.ones((npts, nvm), dtype=np.float64),
        natural=np.asarray([True, True, False]),
        leaf_tab=np.asarray([4, 1, 3]),
        pheno_type=np.asarray([0, 2, 0]),
        pft_to_mtc=np.asarray([1, 2, 3]),
        time_scales=SeasonTimeScales(),
        firstcall=True,
    ).state

    np.testing.assert_allclose(result.moiavail_month, moiavail_daily)
    np.testing.assert_allclose(result.moiavail_week, moiavail_daily)
    np.testing.assert_allclose(result.t2m_month, t2m_daily)
    np.testing.assert_allclose(result.t2m_week, t2m_daily)
    np.testing.assert_allclose(result.tsoil_month, tsoil_daily)
    np.testing.assert_allclose(result.soilhum_month, soilhum_daily)
    np.testing.assert_allclose(result.tseason, t2m_daily)


def test_season_memory_end_of_year_derives_tseason_and_resets_only_local_year_fields():
    state = _state(npts=2, nvm=3, nsoil=2)

    result = season_memory_step(
        state,
        dt_days=1.0,
        end_of_year=True,
        moiavail_daily=np.ones((2, 3), dtype=np.float64) * 0.5,
        t2m_daily=np.asarray([270.0, 271.0], dtype=np.float64),
        tsoil_daily=np.ones((2, 2), dtype=np.float64) * 270.0,
        soilhum_daily=np.ones((2, 2), dtype=np.float64) * 0.5,
        begin_leaves=np.ones((2, 3), dtype=bool),
        julian_diff=365.0,
        when_growthinit=np.ones((2, 3), dtype=np.float64),
        natural=np.asarray([True, False, True]),
        leaf_tab=np.asarray([4, 1, 1]),
        pheno_type=np.asarray([0, 2, 2]),
        pft_to_mtc=np.asarray([1, 8, 6]),
    ).state

    np.testing.assert_allclose(result.tseason, np.asarray([1098.0 / 4.0, 810.0 / 3.0]))
    np.testing.assert_allclose(result.tseason_tmp, 0.0)
    np.testing.assert_allclose(result.tseason_length, 0.0)
    np.testing.assert_allclose(result.tmin_spring_time, 0.0)
    np.testing.assert_allclose(result.onset_date, 0.0)


def test_season_annual_step_matches_longterm_fluxes_and_paper_case_non_dgvm_statistics():
    npts, nvm, nparts = 2, 4, 3
    annual = SeasonAnnualState(
        npp_longterm=np.asarray([[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]], dtype=np.float64),
        turnover_longterm=np.ones((npts, nvm, nparts, 1), dtype=np.float64) * 5.0,
        gpp_week=np.asarray([[0.0, 7.0, 14.0, 21.0], [0.0, 8.0, 15.0, 22.0]], dtype=np.float64),
        maxmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64) * 0.3,
        maxmoiavail_thisyear=np.ones((npts, nvm), dtype=np.float64) * 0.4,
        minmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64) * 0.7,
        minmoiavail_thisyear=np.ones((npts, nvm), dtype=np.float64) * 0.6,
        maxgppweek_lastyear=np.ones((npts, nvm), dtype=np.float64) * 2.0,
        maxgppweek_thisyear=np.ones((npts, nvm), dtype=np.float64) * 3.0,
        gdd0_lastyear=np.asarray([1.0, 2.0], dtype=np.float64),
        gdd0_thisyear=np.asarray([10.0, 20.0], dtype=np.float64),
        precip_lastyear=np.asarray([3.0, 4.0], dtype=np.float64),
        precip_thisyear=np.asarray([30.0, 40.0], dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64) * 9.0,
        lm_thisyearmax=np.ones((npts, nvm), dtype=np.float64) * 2.0,
        maxfpc_lastyear=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        maxfpc_thisyear=np.ones((npts, nvm), dtype=np.float64) * 0.1,
    )
    biomass = np.zeros((npts, nvm, nparts, 1), dtype=np.float64)
    biomass[:, 1, 0, 0] = [5.0, 1.0]
    biomass[:, 2, 0, 0] = [6.0, 7.0]
    biomass[:, 3, 0, 0] = [1.0, 8.0]
    moiavail_daily = np.asarray([[0.5, 0.2, 0.8, 0.1], [0.6, 0.9, 0.3, 0.2]], dtype=np.float64)
    gpp_daily = np.asarray([[0.0, 70.0, 0.0, 35.0], [0.0, 14.0, 21.0, 0.0]], dtype=np.float64)
    npp_daily = np.ones((npts, nvm), dtype=np.float64) * 2.0
    turnover_daily = np.ones((npts, nvm, nparts, 1), dtype=np.float64) * 0.5
    veget = np.asarray([[0.0, 0.3, 0.4, 0.2], [0.0, 0.1, 0.5, 0.6]], dtype=np.float64)
    veget_max = np.asarray([[0.0, 0.2, 0.0, 0.4], [0.0, 0.3, 0.4, 0.0]], dtype=np.float64)
    scales = SeasonTimeScales(tau_gpp_week=7.0, coeff_tau_longterm=3.0, tau_climatology=20.0)

    step = season_annual_step(
        annual,
        dt_days=1.0,
        tau_longterm=10.0,
        end_of_year=False,
        veget=veget,
        veget_max=veget_max,
        moiavail_daily=moiavail_daily,
        t2m_daily=np.asarray([274.15, 270.0], dtype=np.float64),
        precip_daily=np.asarray([2.0, 3.0], dtype=np.float64),
        biomass=biomass,
        npp_daily=npp_daily,
        turnover_daily=turnover_daily,
        gpp_daily=gpp_daily,
        natural=np.asarray([True, True, False, False]),
        pasture=np.asarray([False, False, False, True]),
        leaflife_tab=np.ones(nvm, dtype=np.float64),
        pheno_model=("none", "none", "cold", "none"),
        time_scales=scales,
    )
    result = step.state

    tau_longterm = 10.0
    expected_npp = (annual.npp_longterm * (tau_longterm - 1.0) + npp_daily * 365.0) / tau_longterm
    np.testing.assert_allclose(result.npp_longterm, expected_npp)
    np.testing.assert_allclose(
        result.turnover_longterm,
        (annual.turnover_longterm * (tau_longterm - 1.0) + turnover_daily * 365.0) / tau_longterm,
    )
    expected_gpp_week = np.where(veget_max > 0.0, (annual.gpp_week * 6.0 + gpp_daily) / 7.0, 0.0)
    np.testing.assert_allclose(result.gpp_week, expected_gpp_week)
    np.testing.assert_allclose(result.maxmoiavail_thisyear, np.maximum(annual.maxmoiavail_thisyear, moiavail_daily))
    np.testing.assert_allclose(result.minmoiavail_thisyear, np.minimum(annual.minmoiavail_thisyear, moiavail_daily))
    np.testing.assert_allclose(result.maxgppweek_thisyear, np.maximum(annual.maxgppweek_thisyear, expected_gpp_week))
    np.testing.assert_allclose(result.gdd0_thisyear, np.asarray([11.0, 20.0]))
    np.testing.assert_allclose(result.precip_thisyear, np.asarray([32.0, 43.0]))
    expected_lm = annual.lm_thisyearmax.copy()
    expected_lm[:, 1:] = np.maximum(expected_lm[:, 1:], biomass[:, 1:, 0, 0])
    np.testing.assert_allclose(result.lm_thisyearmax, expected_lm)
    np.testing.assert_allclose(result.maxfpc_thisyear, np.maximum(annual.maxfpc_thisyear, veget))
    expected_herbivores = np.ones((npts, nvm), dtype=np.float64) * 100000.0
    expected_herbivores[:, 0] = 0.0
    nlflong = expected_npp[:, 1] * 0.33
    expected_herbivores[:, 1] = 365.0 * 2.0 * nlflong / (0.019 * nlflong**1.38)
    np.testing.assert_allclose(step.herbivores, expected_herbivores)


def test_season_annual_step_end_of_year_rolls_and_resets_statistics():
    npts, nvm = 1, 4
    annual = SeasonAnnualState(
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        turnover_longterm=np.ones((npts, nvm, 2, 1), dtype=np.float64),
        gpp_week=np.ones((npts, nvm), dtype=np.float64),
        maxmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        maxmoiavail_thisyear=np.ones((npts, nvm), dtype=np.float64) * 0.6,
        minmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64) * 0.8,
        minmoiavail_thisyear=np.ones((npts, nvm), dtype=np.float64) * 0.3,
        maxgppweek_lastyear=np.ones((npts, nvm), dtype=np.float64) * 4.0,
        maxgppweek_thisyear=np.ones((npts, nvm), dtype=np.float64) * 8.0,
        gdd0_lastyear=np.asarray([10.0], dtype=np.float64),
        gdd0_thisyear=np.asarray([100.0], dtype=np.float64),
        precip_lastyear=np.asarray([20.0], dtype=np.float64),
        precip_thisyear=np.asarray([200.0], dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64) * 1.0,
        lm_thisyearmax=np.ones((npts, nvm), dtype=np.float64) * 5.0,
        maxfpc_lastyear=np.ones((npts, nvm), dtype=np.float64) * 0.1,
        maxfpc_thisyear=np.asarray([[0.0, 0.2, 0.3, 0.4]], dtype=np.float64),
    )

    step = season_annual_step(
        annual,
        dt_days=1.0,
        tau_longterm=10.0,
        end_of_year=True,
        veget=np.zeros((npts, nvm), dtype=np.float64),
        veget_max=np.ones((npts, nvm), dtype=np.float64),
        moiavail_daily=np.ones((npts, nvm), dtype=np.float64) * 0.5,
        t2m_daily=np.asarray([270.0], dtype=np.float64),
        precip_daily=np.asarray([0.0], dtype=np.float64),
        biomass=np.zeros((npts, nvm, 2, 1), dtype=np.float64),
        npp_daily=np.zeros((npts, nvm), dtype=np.float64),
        turnover_daily=np.zeros((npts, nvm, 2, 1), dtype=np.float64),
        gpp_daily=np.zeros((npts, nvm), dtype=np.float64),
        natural=np.asarray([True, True, False, True]),
        pasture=np.asarray([False, False, False, True]),
        leaflife_tab=np.ones(nvm, dtype=np.float64),
        pheno_model=("none", "none", "cold", "none"),
        time_scales=SeasonTimeScales(tau_climatology=20.0),
    )
    result = step.state

    np.testing.assert_allclose(result.maxmoiavail_lastyear, (0.2 * 19.0 + 0.6) / 20.0)
    np.testing.assert_allclose(result.minmoiavail_lastyear, (0.8 * 19.0 + 0.3) / 20.0)
    np.testing.assert_allclose(result.maxgppweek_lastyear, (4.0 * 19.0 + 8.0) / 20.0)
    np.testing.assert_allclose(result.gdd0_lastyear, 100.0)
    np.testing.assert_allclose(result.precip_lastyear, 200.0)
    np.testing.assert_allclose(result.lm_lastyearmax, 5.0)
    np.testing.assert_allclose(result.maxfpc_lastyear, np.asarray([[0.0, 0.2, 0.0, 0.0]]))
    np.testing.assert_allclose(result.maxmoiavail_thisyear, 0.0)
    np.testing.assert_allclose(result.minmoiavail_thisyear, 1.0e33)
    np.testing.assert_allclose(result.maxgppweek_thisyear, 0.0)
    np.testing.assert_allclose(result.gdd0_thisyear, 0.0)
    np.testing.assert_allclose(result.precip_thisyear, 0.0)
    np.testing.assert_allclose(result.lm_thisyearmax, 0.0)
    np.testing.assert_allclose(result.maxfpc_thisyear, 0.0)


def test_season_annual_step_dgvm_updates_maxfpc_and_leaf_mass_relaxation():
    """Fortran: stomate_season.f90::season lines 1281-1367 with ok_dgvm."""

    npts, nvm = 1, 4
    annual = SeasonAnnualState(
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        turnover_longterm=np.ones((npts, nvm, 2, 1), dtype=np.float64),
        gpp_week=np.ones((npts, nvm), dtype=np.float64),
        maxmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64),
        maxmoiavail_thisyear=np.ones((npts, nvm), dtype=np.float64),
        minmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64),
        minmoiavail_thisyear=np.ones((npts, nvm), dtype=np.float64),
        maxgppweek_lastyear=np.ones((npts, nvm), dtype=np.float64),
        maxgppweek_thisyear=np.ones((npts, nvm), dtype=np.float64),
        gdd0_lastyear=np.zeros(npts, dtype=np.float64),
        gdd0_thisyear=np.zeros(npts, dtype=np.float64),
        precip_lastyear=np.zeros(npts, dtype=np.float64),
        precip_thisyear=np.zeros(npts, dtype=np.float64),
        lm_lastyearmax=np.array([[0.0, 4.0, 5.0, 6.0]], dtype=np.float64),
        lm_thisyearmax=np.array([[0.0, 4.0, 0.0, 6.0]], dtype=np.float64),
        maxfpc_lastyear=np.array([[0.0, 0.2, 0.3, 0.4]], dtype=np.float64),
        maxfpc_thisyear=np.array([[0.0, 0.1, 0.1, 0.1]], dtype=np.float64),
    )
    biomass = np.zeros((npts, nvm, 2, 1), dtype=np.float64)
    biomass[0, :, 0, 0] = [0.0, 4.0, 3.0, 10.0]
    veget = np.array([[0.0, 0.5, 0.2, 0.1]], dtype=np.float64)
    veget_max = np.array([[0.0, 0.5, 0.2, 0.1]], dtype=np.float64)
    leaflife = np.array([1.0, 2.0, 1.0, 4.0], dtype=np.float64)

    step = season_annual_step(
        annual,
        dt_days=1.0,
        tau_longterm=10.0,
        end_of_year=False,
        veget=veget,
        veget_max=veget_max,
        moiavail_daily=np.ones((npts, nvm), dtype=np.float64),
        t2m_daily=np.array([270.0], dtype=np.float64),
        precip_daily=np.array([0.0], dtype=np.float64),
        biomass=biomass,
        npp_daily=np.zeros((npts, nvm), dtype=np.float64),
        turnover_daily=np.zeros((npts, nvm, 2, 1), dtype=np.float64),
        gpp_daily=np.zeros((npts, nvm), dtype=np.float64),
        natural=np.array([True, True, True, False]),
        pasture=np.array([False, False, False, False]),
        leaflife_tab=leaflife,
        ok_dgvm=True,
        time_scales=SeasonTimeScales(),
    )
    result = step.state

    fracnat = 1.0 - veget_max[0, 3]
    tau_pft1 = 365.0 / leaflife[1]
    expected_maxfpc_pft1 = (annual.maxfpc_lastyear[0, 1] * (tau_pft1 - 1.0) + veget[0, 1] / fracnat) / tau_pft1
    expected_lm_pft1 = (annual.lm_thisyearmax[0, 1] * (tau_pft1 - 1.0) + biomass[0, 1, 0, 0]) / tau_pft1
    tau_pft3 = 365.0 / leaflife[3]
    expected_lm_pft3 = (annual.lm_thisyearmax[0, 3] * (tau_pft3 - 1.0) + biomass[0, 3, 0, 0]) / tau_pft3

    np.testing.assert_allclose(np.asarray(result.maxfpc_lastyear)[0, 1], expected_maxfpc_pft1)
    np.testing.assert_allclose(np.asarray(result.maxfpc_thisyear)[0, 1], max(expected_maxfpc_pft1, veget[0, 1]))
    np.testing.assert_allclose(np.asarray(result.maxfpc_lastyear)[0, 3], annual.maxfpc_lastyear[0, 3])
    np.testing.assert_allclose(np.asarray(result.lm_thisyearmax)[0, 1], expected_lm_pft1)
    np.testing.assert_allclose(np.asarray(result.lm_thisyearmax)[0, 2], biomass[0, 2, 0, 0])
    np.testing.assert_allclose(np.asarray(result.lm_thisyearmax)[0, 3], expected_lm_pft3)


def test_season_biometeorology_step_matches_gdd_ncd_ngd_and_humidity_memory_sections():
    state = SeasonBiometeorologyState(
        gdd_m5_dormance=np.asarray([[9.0, -9999.0, 10.0, 11.0], [8.0, 4.0, -9999.0, 12.0]], dtype=np.float64),
        gdd_midwinter=np.asarray([[1.0, -9999.0, 2.0, 3.0], [1.0, 5.0, 6.0, 7.0]], dtype=np.float64),
        ncd_dormance=np.asarray([[1.0, -9999.0, 2.0, 3.0], [1.0, 5.0, 6.0, 7.0]], dtype=np.float64),
        ngd_minus5=np.asarray([[0.0, 2.0, 4.0, 6.0], [1.0, 3.0, 5.0, 7.0]], dtype=np.float64),
        time_hum_min=np.asarray([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float64),
        hum_min_dormance=np.asarray([[0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]], dtype=np.float64),
    )
    t2m_daily = np.asarray([280.0, 265.0], dtype=np.float64)
    t2m_month = np.asarray([270.0, 290.0], dtype=np.float64)
    t2m_week = np.asarray([275.0, 280.0], dtype=np.float64)
    t2m_longterm = np.asarray([276.0, 281.0], dtype=np.float64)
    moiavail_month = np.asarray([[0.5, 0.7, 0.6, 0.4], [0.5, 0.9, 0.6, 0.7]], dtype=np.float64)
    when_growthinit = np.asarray([[0.0, 2.0, 0.0, 3.0], [1.0, 0.0, 2.0, 4.0]], dtype=np.float64)
    pheno_gdd_crit = np.asarray(
        [[-9999.0, -9999.0, -9999.0], [1.0, 2.0, 3.0], [-9999.0, -9999.0, -9999.0], [1.0, 2.0, 3.0]],
        dtype=np.float64,
    )
    ncdgdd_temp = np.asarray([-9999.0, 5.0, -9999.0, 0.0], dtype=np.float64)
    hum_min_time = np.asarray([-9999.0, 50.0, -9999.0, 20.0], dtype=np.float64)

    result = season_biometeorology_step(
        state,
        dt_days=1.0,
        julian_diff=10.0,
        t2m_daily=t2m_daily,
        t2m_month=t2m_month,
        t2m_week=t2m_week,
        t2m_longterm=t2m_longterm,
        moiavail_month=moiavail_month,
        when_growthinit=when_growthinit,
        gdd_init_date=np.asarray([[10.0, 0.0], [9.0, 0.0]], dtype=np.float64),
        pheno_gdd_crit=pheno_gdd_crit,
        ncdgdd_temp=ncdgdd_temp,
        hum_min_time=hum_min_time,
        time_scales=SeasonTimeScales(tau_gdd=40.0, tau_ngd=50.0),
    ).state

    expected_gdd_m5 = state.gdd_m5_dormance.copy()
    expected_gdd_m5[0, 1] = (0.0 + (280.0 - 268.15)) * 39.0 / 40.0
    expected_gdd_m5[1, 1] = -9999.0
    expected_gdd_m5[0, 3] = (0.0 + (280.0 - 268.15)) * 39.0 / 40.0
    expected_gdd_m5[1, 3] = 12.0 * 39.0 / 40.0
    np.testing.assert_allclose(result.gdd_m5_dormance, expected_gdd_m5)

    expected_mid = state.gdd_midwinter.copy()
    expected_mid[0, 1] = 0.0 + (280.0 - 278.15)
    expected_mid[1, 1] = -9999.0
    expected_mid[0, 3] = 0.0 + (280.0 - 273.15)
    expected_mid[1, 3] = -9999.0
    np.testing.assert_allclose(result.gdd_midwinter, expected_mid)

    expected_ncd = state.ncd_dormance.copy()
    expected_ncd[0, 1] = 0.0
    expected_ncd[1, 1] = -9999.0
    expected_ncd[0, 3] = 0.0
    expected_ncd[1, 3] = 8.0
    np.testing.assert_allclose(result.ncd_dormance, expected_ncd)

    expected_ngd = state.ngd_minus5.copy()
    expected_ngd[:, 1:] = np.asarray([[1.0, 1.0, 1.0], [3.0, 5.0, 7.0]]) * 49.0 / 50.0
    np.testing.assert_allclose(result.ngd_minus5, expected_ngd)

    expected_time = state.time_hum_min.copy()
    expected_hum = state.hum_min_dormance.copy()
    expected_time[0, 1] = 0.0
    expected_hum[0, 1] = 0.7
    expected_time[1, 1] = 0.0
    expected_hum[1, 1] = 0.9
    expected_time[0, 3] = 0.0
    expected_hum[0, 3] = 0.4
    expected_time[1, 3] = 5.0
    expected_hum[1, 3] = 0.6
    np.testing.assert_allclose(result.time_hum_min, expected_time)
    np.testing.assert_allclose(result.hum_min_dormance, expected_hum)


def test_season_biometeorology_firstcall_initializes_zero_restart_memories_to_undef():
    npts, nvm = 2, 4
    zeros = np.zeros((npts, nvm), dtype=np.float64)
    result = season_biometeorology_step(
        SeasonBiometeorologyState(
            gdd_m5_dormance=zeros,
            gdd_midwinter=zeros,
            ncd_dormance=zeros,
            ngd_minus5=zeros,
            time_hum_min=zeros,
            hum_min_dormance=zeros,
        ),
        dt_days=1.0,
        julian_diff=100.0,
        t2m_daily=np.asarray([260.0, 290.0]),
        t2m_month=np.asarray([260.0, 290.0]),
        t2m_week=np.asarray([260.0, 290.0]),
        t2m_longterm=np.asarray([260.0, 290.0]),
        moiavail_month=np.zeros((npts, nvm)),
        when_growthinit=np.ones((npts, nvm)),
        gdd_init_date=np.zeros((npts, 2)),
        pheno_gdd_crit=np.full((nvm, 4), -9999.0),
        ncdgdd_temp=np.full(nvm, -9999.0),
        hum_min_time=np.full(nvm, -9999.0),
        firstcall=True,
        undef=-9999.0,
    ).state

    np.testing.assert_allclose(result.gdd_m5_dormance, -9999.0)
    np.testing.assert_allclose(result.gdd_midwinter, -9999.0)
    np.testing.assert_allclose(result.ncd_dormance, -9999.0)
    expected_ngd = np.zeros((npts, nvm))
    expected_ngd[1, 1:] = 49.0 / 50.0
    np.testing.assert_allclose(result.ngd_minus5, expected_ngd)
