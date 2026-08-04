from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.init import (  # noqa: E402
    parse_run_def,
    parse_run_def_bool,
    parse_run_def_float,
    parse_run_def_indexed_bool,
)
from jax_orchidee.driver.orchestration import paper_1961_driver_timestep_scaffold  # noqa: E402
from jax_orchidee.driver.trace import read_static_fields_from_fixed_format_trace_dir  # noqa: E402
from jax_orchidee.sechiba.diffuco import diffuco_bare_cwrr_beta  # noqa: E402
from jax_orchidee.sechiba.diffuco_bridge import first_pft14_diffuco_after_main_payload  # noqa: E402
from jax_orchidee.sechiba.enerbil import (  # noqa: E402
    C_STEFAN,
    CHALEV0,
    CHALSU0,
    CP_AIR,
    CTE_GRAV,
    CTE_MOLR,
    MSMLR_AIR,
    MSMLR_H2O,
    PA_PAR_HPA,
    TP_00,
    assemble_enerbil_explicit_local_step_from_server_bridge,
    assemble_enerbil_first_step_precall_payload,
    condveg_initialized_emis,
    dim2_driver_non_watchout_energy_coupling_inputs,
    driver_swnet_from_swdown_albedo,
    enerbil_begin_local_diagnostics,
    enerbil_cold_start_surface_state,
    enerbil_evapveg_grid_fluxes,
    enerbil_evapveg_pft_fluxes,
    enerbil_explicit_local_step,
    enerbil_first_step_input_coverage,
    enerbil_flux_evapot_corr,
    enerbil_flux_explicit_snow_diagnostics,
    enerbil_flux_inputs_contract,
    enerbil_flux_local_diagnostics,
    enerbil_fusion_step,
    enerbil_pottemp_pass_through,
    enerbil_surface_state_input_coverage,
    enerbil_surftemp_explicit_solve,
    enerbil_surftemp_qsol_sat_update,
    enerbil_swnet_input_chain_coverage,
    enerbil_t2mdiag,
    qsat_moisture_dev_qsatcalc,
    qsat_moisture_qsatcalc,
    read_driver_albedo_restart,
    read_enerbil_soil_thermal_state_restart,
    run_enerbil_first_step_local_from_precall,
    sechiba_air_density_from_pb_temp_air,
)
from jax_orchidee.sechiba.enerbil_bridge import (  # noqa: E402
    ENERBIL_ACTIVE_AFTER_TAG,
    ENERBIL_ACTIVE_BEFORE_TAG,
    ENERBIL_POTTEMP_AFTER_TAG,
    ENERBIL_POTTEMP_BEFORE_TAG,
    read_enerbil_active_payload,
    read_enerbil_after_main_payload,  # noqa: E402
    read_enerbil_pottemp_active_payload,
)
from jax_orchidee.trace.server_1961 import BRIDGE_TRACE_ROOT, read_server_records  # noqa: E402

REFERENCE_SECHIBA_START = (
    ROOT
    / "reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0"
    / "I10/S2_63.206_0.0876_0.2019_50.658/sechiba_start.nc"
)
REFERENCE_DRIVER_START = REFERENCE_SECHIBA_START.with_name("driver_start.nc")
REFERENCE_RUN_DEF = REFERENCE_SECHIBA_START.with_name("run.def")
BRIDGE_RUN_DEF = ROOT / "outputs/server_1961_bridge_trace_20260624/configs/run.def"
USED_RUN_DEF = ROOT / "outputs/server_1961_trace_full_20260623/run/used_run.def"
FULL_TRACE_DIR = ROOT / "outputs/server_1961_trace_full_20260623/traces"


def test_enerbil_cold_start_surface_state_matches_initialize_fallbacks():
    qair = np.asarray([0.011, 0.012], dtype=np.float64)
    state = enerbil_cold_start_surface_state(qair=qair, nvm=3)

    np.testing.assert_allclose(np.asarray(state.temp_sol), 280.0)
    np.testing.assert_allclose(np.asarray(state.temp_sol_pft), 280.0)
    np.testing.assert_allclose(np.asarray(state.temp_sol_new), np.asarray(state.temp_sol))
    np.testing.assert_allclose(np.asarray(state.temp_sol_new_pft), np.asarray(state.temp_sol_pft))
    np.testing.assert_allclose(np.asarray(state.qsurf), qair)
    np.testing.assert_allclose(np.asarray(state.evapot), 0.0)
    np.testing.assert_allclose(np.asarray(state.evapot_corr), 0.0)
    np.testing.assert_allclose(np.asarray(state.tsol_rad), np.asarray(state.temp_sol))
    custom = enerbil_cold_start_surface_state(qair=qair[:1], nvm=1, enerbil_tsurf=276.5)
    np.testing.assert_allclose(np.asarray(custom.temp_sol), 276.5)
    with pytest.raises(ValueError, match="qair"):
        enerbil_cold_start_surface_state(qair=np.ones((1, 1)), nvm=1)


def test_enerbil_fusion_bucket_snow_reduces_grid_and_pft_temperature():
    """Fortran: enerbil.f90::enerbil_fusion lines 1933-1948."""

    chalfu0 = CHALSU0 - CHALEV0
    result = enerbil_fusion_step(
        tot_melt=np.asarray([0.5, 0.0], dtype=np.float64),
        soilcap=np.asarray([chalfu0 * 2.0, chalfu0], dtype=np.float64),
        soilcap_pft=np.asarray(
            [
                [chalfu0, chalfu0 * 4.0, chalfu0 * 8.0],
                [chalfu0, chalfu0, chalfu0],
            ],
            dtype=np.float64,
        ),
        snowdz=np.zeros((2, 3), dtype=np.float64),
        temp_sol_new=np.asarray([275.0, 270.0], dtype=np.float64),
        temp_sol_new_pft=np.asarray(
            [
                [276.0, 277.0, 278.0],
                [271.0, 272.0, 273.0],
            ],
            dtype=np.float64,
        ),
        ok_laidev=np.asarray([True, False, True]),
        ok_explicitsnow=False,
        dt_sechiba=1800.0,
    )

    expected_grid = np.asarray([274.75, 270.0], dtype=np.float64)
    expected_pft = np.asarray(
        [
            [275.5, expected_grid[0], 277.9375],
            [271.0, expected_grid[1], 273.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(result.fusion), [0.5 * chalfu0 / 1800.0, 0.0])
    np.testing.assert_allclose(np.asarray(result.temp_sol_new), expected_grid)
    np.testing.assert_allclose(np.asarray(result.temp_sol_new_pft), expected_pft)
    assert any("enerbil_fusion lines 1881-1953" in item for item in result.provenance)


def test_enerbil_fusion_explicit_snow_caps_when_snow_layers_present():
    """Fortran: enerbil.f90::enerbil_fusion lines 1917-1927."""

    result = enerbil_fusion_step(
        tot_melt=np.asarray([2.0, 2.0], dtype=np.float64),
        soilcap=np.asarray([1000.0, 1000.0], dtype=np.float64),
        soilcap_pft=np.ones((2, 2), dtype=np.float64),
        snowdz=np.asarray([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        temp_sol_new=np.asarray([274.0, 274.0], dtype=np.float64),
        temp_sol_new_pft=np.asarray([[274.5, 275.5], [274.5, 275.5]], dtype=np.float64),
        ok_laidev=np.asarray([True, True]),
        ok_explicitsnow=True,
    )

    np.testing.assert_allclose(np.asarray(result.temp_sol_new), [TP_00, 274.0])
    np.testing.assert_allclose(np.asarray(result.temp_sol_new_pft), [[TP_00, TP_00], [274.5, 275.5]])
    np.testing.assert_allclose(np.asarray(result.fusion), 0.0)


def _qsfrict_expected_table():
    temp = np.arange(371, dtype=np.float64)
    safe_temp = np.maximum(temp, 1.0)
    zrapp = MSMLR_H2O / MSMLR_AIR
    zcorr = 0.00320991
    solid = zrapp * 10.0 ** (
        2.07023
        - zcorr * safe_temp
        - 2484.896 / safe_temp
        + 3.56654 * np.log10(safe_temp)
    )
    liquid = zrapp * 10.0 ** (
        23.8319
        - 2948.964 / safe_temp
        - 5.028 * np.log10(safe_temp)
        - 29810.16 * np.exp(-0.0699382 * safe_temp)
        + 25.21935 * np.exp(-2999.924 / safe_temp)
    )
    table = np.where(temp < 273.0, solid, liquid)
    table[:101] = 0.0
    return table


def test_qsat_moisture_qsatcalc_and_derivative_follow_fortran_table_interpolation():
    temp = np.asarray([280.25, 295.75], dtype=np.float64)
    pres = np.asarray([1000.0, 990.0], dtype=np.float64)
    table = _qsfrict_expected_table()

    qsat = np.asarray(qsat_moisture_qsatcalc(temp, pres))
    dev_qsat = np.asarray(qsat_moisture_dev_qsatcalc(temp, pres))

    jt_qsat = temp.astype(np.int32)
    frac_qsat = temp - jt_qsat
    expected_qsat = ((table[jt_qsat + 1] - table[jt_qsat]) * frac_qsat + table[jt_qsat]) / pres

    jt_dev = (temp + 0.5).astype(np.int32)
    frac_dev = temp + 0.5 - jt_dev
    expected_dev = (
        (table[jt_dev + 1] - 2.0 * table[jt_dev] + table[jt_dev - 1]) * (frac_dev - 1.0)
        + table[jt_dev + 1]
        - table[jt_dev]
    ) / pres

    np.testing.assert_allclose(qsat, expected_qsat)
    np.testing.assert_allclose(dev_qsat, expected_dev)


def test_qsat_moisture_clamps_only_fortran_table_index_outside_bounds():
    temp = np.asarray([99.25, 370.25], dtype=np.float64)
    pres = np.asarray([1013.25, 700.0], dtype=np.float64)
    table = _qsfrict_expected_table()

    q_jt = np.asarray([100, 369], dtype=np.int32)
    q_fraction = temp - q_jt
    q_expected = (
        (table[q_jt + 1] - table[q_jt]) * q_fraction + table[q_jt]
    ) / pres
    d_jt = np.asarray([100, 369], dtype=np.int32)
    d_fraction = temp + 0.5 - d_jt
    d_expected = (
        (table[d_jt + 1] - 2.0 * table[d_jt] + table[d_jt - 1])
        * (d_fraction - 1.0)
        + table[d_jt + 1]
        - table[d_jt]
    ) / pres

    np.testing.assert_allclose(
        np.asarray(qsat_moisture_qsatcalc(temp, pres)), q_expected, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(qsat_moisture_dev_qsatcalc(temp, pres)),
        d_expected,
        rtol=0.0,
        atol=0.0,
    )


def test_enerbil_begin_local_diagnostics_matches_explicit_fortran_formulas_and_pft_branch():
    temp_sol = np.asarray([280.0, 290.0], dtype=np.float64)
    temp_sol_pft = np.asarray(
        [[279.0, 281.0, 282.0], [289.0, 291.0, 292.0]],
        dtype=np.float64,
    )
    lwdown = np.asarray([338.0, 350.0], dtype=np.float64)
    swnet = np.asarray([120.0, 80.0], dtype=np.float64)
    pb = np.asarray([1010.0, 995.0], dtype=np.float64)
    emis = np.asarray([0.96, 0.98], dtype=np.float64)
    ok_laidev = np.asarray([False, True, False])

    result = enerbil_begin_local_diagnostics(
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        lwdown=lwdown,
        swnet=swnet,
        pb=pb,
        emis=emis,
        ok_laidev=ok_laidev,
    )

    expected_psold = temp_sol * CP_AIR
    expected_psold_pft = np.where(ok_laidev[None, :], temp_sol_pft * CP_AIR, expected_psold[:, None])
    expected_lwabs = emis * lwdown
    expected_netrad = lwdown + swnet - (
        emis * C_STEFAN * temp_sol**4 + (1.0 - emis) * lwdown
    )
    expected_netrad_pft = lwdown[:, None] + swnet[:, None] - (
        emis[:, None] * C_STEFAN * temp_sol_pft**4
        + (1.0 - emis[:, None]) * lwdown[:, None]
    )

    np.testing.assert_allclose(np.asarray(result.psold), expected_psold)
    np.testing.assert_allclose(np.asarray(result.psold_pft), expected_psold_pft)
    np.testing.assert_allclose(np.asarray(result.lwabs), expected_lwabs)
    np.testing.assert_allclose(np.asarray(result.netrad), expected_netrad)
    np.testing.assert_allclose(np.asarray(result.netrad_pft)[:, 1:], expected_netrad_pft[:, 1:])
    assert np.isnan(np.asarray(result.netrad_pft)[0, 0])
    assert np.isnan(np.asarray(result.qsol_sat_pft)[0, 0])
    assert np.isnan(np.asarray(result.pdqsold_pft)[0, 0])
    np.testing.assert_allclose(
        np.asarray(result.qsol_sat),
        np.asarray(qsat_moisture_qsatcalc(temp_sol, pb)),
    )
    np.testing.assert_allclose(
        np.asarray(result.qsol_sat_pft)[:, 1],
        np.asarray(qsat_moisture_qsatcalc(temp_sol_pft[:, 1], pb)),
    )
    np.testing.assert_allclose(
        np.asarray(result.pdqsold),
        np.asarray(qsat_moisture_dev_qsatcalc(temp_sol, pb)),
    )


def test_enerbil_surftemp_qsol_sat_update_matches_fortran_formula():
    qsol_sat = np.array([0.014, 0.008])
    pdqsold = np.array([2.0e-4, 1.5e-4])
    dtheta = np.array([120.0, -80.0])
    qsol_sat_pft = np.array([[0.015, 0.016, 0.017], [0.009, 0.010, 0.011]])
    pdqsold_pft = np.array([[2.1e-4, 2.2e-4, 2.3e-4], [1.1e-4, 1.2e-4, 1.3e-4]])
    dtheta_pft = np.array([[100.0, 110.0, 120.0], [-70.0, -80.0, -90.0]])
    ok_laidev = np.array([False, True, False])

    qsol_sat_new, qsol_sat_new_pft = enerbil_surftemp_qsol_sat_update(
        qsol_sat=qsol_sat,
        pdqsold=pdqsold,
        dtheta=dtheta,
        qsol_sat_pft=qsol_sat_pft,
        pdqsold_pft=pdqsold_pft,
        dtheta_pft=dtheta_pft,
        ok_laidev=ok_laidev,
    )

    expected_grid = qsol_sat + (1.0 / 1004.675) * pdqsold * dtheta
    expected_pft_formula = qsol_sat_pft + (1.0 / 1004.675) * pdqsold_pft * dtheta_pft
    expected_pft = np.where(ok_laidev[None, :], expected_pft_formula, expected_grid[:, None])

    np.testing.assert_allclose(qsol_sat_new, expected_grid)
    np.testing.assert_allclose(qsol_sat_new_pft, expected_pft)


def test_enerbil_surftemp_explicit_solve_matches_linearized_fortran_update():
    temp_sol = np.asarray([280.0, 290.0], dtype=np.float64)
    temp_sol_pft = np.asarray(
        [[279.0, 281.0, 282.0], [289.0, 291.0, 292.0]],
        dtype=np.float64,
    )
    begin = enerbil_begin_local_diagnostics(
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        lwdown=np.asarray([338.0, 350.0], dtype=np.float64),
        swnet=np.asarray([120.0, 80.0], dtype=np.float64),
        pb=np.asarray([1010.0, 995.0], dtype=np.float64),
        emis=np.asarray([0.96, 0.98], dtype=np.float64),
        ok_laidev=np.asarray([False, True, False]),
    )
    ok_laidev = np.asarray([False, True, False])
    epot_air = np.asarray([282000.0, 292000.0], dtype=np.float64)
    petAcoef = np.asarray([0.01, 0.02], dtype=np.float64)
    petBcoef = np.asarray([281500.0, 291700.0], dtype=np.float64)
    qair = np.asarray([0.006, 0.008], dtype=np.float64)
    peqAcoef = np.asarray([0.1, 0.12], dtype=np.float64)
    peqBcoef = np.asarray([0.012, 0.014], dtype=np.float64)
    soilflx = np.asarray([8.0, -4.0], dtype=np.float64)
    soilflx_pft = np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
    rau = np.asarray([1.2, 1.1], dtype=np.float64)
    u = np.asarray([1.0, 2.0], dtype=np.float64)
    v = np.asarray([0.5, 1.0], dtype=np.float64)
    q_cdrag = np.asarray([0.01, 0.02], dtype=np.float64)
    q_cdrag_pft = np.asarray([[0.011, 0.012, 0.013], [0.021, 0.022, 0.023]], dtype=np.float64)
    vbeta = np.asarray([0.4, 0.3], dtype=np.float64)
    vbeta_pft = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.1, 0.2]], dtype=np.float64)
    valpha = np.asarray([1.0, 0.9], dtype=np.float64)
    vbeta1 = np.asarray([0.2, 0.1], dtype=np.float64)
    vbeta5 = np.asarray([0.05, 0.02], dtype=np.float64)
    soilcap = np.asarray([5.0e5, 6.0e5], dtype=np.float64)
    soilcap_pft = np.asarray([[4.0e5, 4.5e5, 5.0e5], [5.5e5, 5.8e5, 6.2e5]], dtype=np.float64)
    veget_max = np.asarray([[1.0, 0.8, 1.0], [1.0, 0.5, 1.0]], dtype=np.float64)
    emis = np.asarray([0.96, 0.98], dtype=np.float64)
    dt_sechiba = 1800.0

    result = enerbil_surftemp_explicit_solve(
        psold=begin.psold,
        psold_pft=begin.psold_pft,
        qsol_sat=begin.qsol_sat,
        qsol_sat_pft=begin.qsol_sat_pft,
        pdqsold=begin.pdqsold,
        pdqsold_pft=begin.pdqsold_pft,
        netrad=begin.netrad,
        netrad_pft=begin.netrad_pft,
        emis=emis,
        epot_air=epot_air,
        petAcoef=petAcoef,
        petBcoef=petBcoef,
        qair=qair,
        peqAcoef=peqAcoef,
        peqBcoef=peqBcoef,
        soilflx=soilflx,
        soilflx_pft=soilflx_pft,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        vbeta=vbeta,
        vbeta_pft=vbeta_pft,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        soilcap=soilcap,
        soilcap_pft=soilcap_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        dt_sechiba=dt_sechiba,
    )

    speed = np.maximum(0.1, np.sqrt(u * u + v * v))
    zikt = 1.0 / (rau * speed * q_cdrag)
    zikt_pft = 1.0 / (rau[:, None] * speed[:, None] * q_cdrag_pft)
    zicp = 1.0 / CP_AIR
    snow_beta = vbeta1 * (1.0 - vbeta5)
    evap_beta = (1.0 - vbeta1) * (1.0 - vbeta5) * vbeta

    sensfl_old = (petBcoef - np.asarray(begin.psold)) / (zikt - petAcoef)
    larsub_old = (
        CHALSU0
        * snow_beta
        * (peqBcoef - np.asarray(begin.qsol_sat))
        / (zikt - snow_beta * peqAcoef)
    )
    lareva_old = (
        CHALEV0
        * evap_beta
        * (peqBcoef - valpha * np.asarray(begin.qsol_sat))
        / (zikt - evap_beta * peqAcoef)
        + CHALEV0 * vbeta5 * (peqBcoef - np.asarray(begin.qsol_sat)) / (zikt - vbeta5 * peqAcoef)
    )
    netrad_sns = zicp * 4.0 * emis * C_STEFAN * (zicp * np.asarray(begin.psold)) ** 3
    sensfl_sns = 1.0 / (zikt - petAcoef)
    larsub_sns = (
        CHALSU0
        * snow_beta
        * zicp
        * np.asarray(begin.pdqsold)
        / (zikt - snow_beta * peqAcoef)
    )
    evap_sns_beta = evap_beta * valpha + vbeta5
    lareva_sns = (
        CHALEV0
        * evap_sns_beta
        * zicp
        * np.asarray(begin.pdqsold)
        / (zikt - evap_sns_beta * peqAcoef)
    )
    sum_old = np.asarray(begin.netrad) + sensfl_old + larsub_old + lareva_old + soilflx
    sum_sns = netrad_sns + sensfl_sns + larsub_sns + lareva_sns
    expected_dtheta = dt_sechiba * sum_old / (zicp * soilcap + dt_sechiba * sum_sns)
    expected_psnew = np.asarray(begin.psold) + expected_dtheta
    expected_qsat_new = np.asarray(begin.qsol_sat) + zicp * np.asarray(begin.pdqsold) * expected_dtheta
    expected_temp_new = expected_psnew / CP_AIR
    expected_epot_air_new = zikt * (sensfl_old - sensfl_sns * expected_dtheta) + expected_psnew
    fevap = (lareva_old - lareva_sns * expected_dtheta) + (larsub_old - larsub_sns * expected_dtheta)
    expected_qair_new = (
        zikt
        / (CHALSU0 * snow_beta + CHALEV0 * evap_sns_beta)
        * fevap
        + expected_qsat_new
    )

    np.testing.assert_allclose(np.asarray(result.dtheta), expected_dtheta)
    np.testing.assert_allclose(np.asarray(result.psnew), expected_psnew)
    np.testing.assert_allclose(np.asarray(result.qsol_sat_new), expected_qsat_new)
    np.testing.assert_allclose(np.asarray(result.temp_sol_new), expected_temp_new)
    np.testing.assert_allclose(np.asarray(result.epot_air_new), expected_epot_air_new)
    np.testing.assert_allclose(np.asarray(result.qair_new), expected_qair_new)
    np.testing.assert_allclose(
        np.asarray(result.sensfl),
        sensfl_old - sensfl_sns * expected_dtheta,
    )
    np.testing.assert_allclose(
        np.asarray(result.larsub),
        larsub_old - larsub_sns * expected_dtheta,
    )
    np.testing.assert_allclose(
        np.asarray(result.lareva),
        lareva_old - lareva_sns * expected_dtheta,
    )

    jv = 1
    sensfl_old_pft = (petBcoef - np.asarray(begin.psold_pft)[:, jv]) / (zikt_pft[:, jv] - petAcoef)
    vbeta_pft_fraction = vbeta_pft[:, jv] / veget_max[:, jv]
    evap_beta_pft = (1.0 - vbeta1) * (1.0 - vbeta5) * vbeta_pft_fraction
    larsub_old_pft = (
        CHALSU0
        * snow_beta
        * (peqBcoef - np.asarray(begin.qsol_sat_pft)[:, jv])
        / (zikt_pft[:, jv] - snow_beta * peqAcoef)
    )
    lareva_old_pft = (
        CHALEV0
        * evap_beta_pft
        * (peqBcoef - valpha * np.asarray(begin.qsol_sat_pft)[:, jv])
        / (zikt_pft[:, jv] - evap_beta_pft * peqAcoef)
        + CHALEV0
        * vbeta5
        * (peqBcoef - np.asarray(begin.qsol_sat_pft)[:, jv])
        / (zikt_pft[:, jv] - vbeta5 * peqAcoef)
    )
    netrad_sns_pft = zicp * 4.0 * emis * C_STEFAN * (zicp * np.asarray(begin.psold_pft)[:, jv]) ** 3
    sensfl_sns_pft = 1.0 / (zikt_pft[:, jv] - petAcoef)
    larsub_sns_pft = (
        CHALSU0
        * snow_beta
        * zicp
        * np.asarray(begin.pdqsold_pft)[:, jv]
        / (zikt_pft[:, jv] - snow_beta * peqAcoef)
    )
    evap_sns_beta_pft = evap_beta_pft * valpha + vbeta5
    lareva_sns_pft = (
        CHALEV0
        * evap_sns_beta_pft
        * zicp
        * np.asarray(begin.pdqsold_pft)[:, jv]
        / (zikt_pft[:, jv] - evap_sns_beta_pft * peqAcoef)
    )
    sum_old_pft = (
        np.asarray(begin.netrad_pft)[:, jv]
        + sensfl_old_pft
        + larsub_old_pft
        + lareva_old_pft
        + soilflx_pft[:, jv]
    )
    sum_sns_pft = netrad_sns_pft + sensfl_sns_pft + larsub_sns_pft + lareva_sns_pft
    expected_dtheta_pft = dt_sechiba * sum_old_pft / (
        zicp * soilcap_pft[:, jv] + dt_sechiba * sum_sns_pft
    )

    np.testing.assert_allclose(np.asarray(result.dtheta_pft)[:, jv], expected_dtheta_pft)
    np.testing.assert_allclose(
        np.asarray(result.sensfl_pft)[:, jv],
        sensfl_old_pft - sensfl_sns_pft * expected_dtheta_pft,
    )
    np.testing.assert_allclose(
        np.asarray(result.larsub_pft)[:, jv],
        larsub_old_pft - larsub_sns_pft * expected_dtheta_pft,
    )
    np.testing.assert_allclose(
        np.asarray(result.lareva_pft)[:, jv],
        lareva_old_pft - lareva_sns_pft * expected_dtheta_pft,
    )
    np.testing.assert_allclose(
        np.asarray(result.qsol_sat_new_pft)[:, jv],
        np.asarray(begin.qsol_sat_pft)[:, jv] + zicp * np.asarray(begin.pdqsold_pft)[:, jv] * expected_dtheta_pft,
    )
    np.testing.assert_allclose(np.asarray(result.temp_sol_new_pft)[:, 0], expected_temp_new)
    np.testing.assert_allclose(np.asarray(result.qsol_sat_new_pft)[:, 2], expected_qsat_new)


def test_enerbil_zero_vegetation_pft_has_finite_zero_reverse_influence_on_grid_temperature():
    def objective(inactive_vbeta):
        result = enerbil_surftemp_explicit_solve(
            psold=jnp.asarray([281000.0]),
            psold_pft=jnp.asarray([[281000.0, 281000.0]]),
            qsol_sat=jnp.asarray([0.007]),
            qsol_sat_pft=jnp.asarray([[0.007, 0.007]]),
            pdqsold=jnp.asarray([0.001]),
            pdqsold_pft=jnp.asarray([[0.001, 0.001]]),
            netrad=jnp.asarray([100.0]),
            netrad_pft=jnp.asarray([[100.0, 100.0]]),
            emis=jnp.asarray([0.96]),
            epot_air=jnp.asarray([282000.0]),
            petAcoef=jnp.asarray([0.01]),
            petBcoef=jnp.asarray([281500.0]),
            qair=jnp.asarray([0.006]),
            peqAcoef=jnp.asarray([0.1]),
            peqBcoef=jnp.asarray([0.012]),
            soilflx=jnp.asarray([5.0]),
            soilflx_pft=jnp.asarray([[5.0, 5.0]]),
            rau=jnp.asarray([1.2]),
            u=jnp.asarray([2.0]),
            v=jnp.asarray([1.0]),
            q_cdrag=jnp.asarray([0.01]),
            q_cdrag_pft=jnp.asarray([[0.01, 0.01]]),
            vbeta=jnp.asarray([0.4]),
            vbeta_pft=jnp.asarray([[0.4, inactive_vbeta]]),
            valpha=jnp.asarray([1.0]),
            vbeta1=jnp.asarray([0.0]),
            vbeta5=jnp.asarray([0.0]),
            soilcap=jnp.asarray([5.0e5]),
            soilcap_pft=jnp.asarray([[5.0e5, 5.0e5]]),
            veget_max=jnp.asarray([[1.0, 0.0]]),
            ok_laidev=jnp.asarray([False, True]),
            dt_sechiba=1800.0,
        )
        return result.temp_sol_new[0]

    primal, reverse = jax.value_and_grad(objective)(jnp.asarray(0.0, dtype=jnp.float64))
    assert np.isfinite(float(primal))
    assert float(reverse) == 0.0


def test_enerbil_explicit_local_step_orders_closed_kernels_without_sourcing_inputs():
    temp_sol = np.asarray([280.0], dtype=np.float64)
    temp_sol_pft = np.asarray([[279.0, 281.0, 282.0]], dtype=np.float64)
    lwdown = np.asarray([338.0], dtype=np.float64)
    swnet = np.asarray([120.0], dtype=np.float64)
    pb = np.asarray([1010.0], dtype=np.float64)
    emis = np.asarray([0.96], dtype=np.float64)
    ok_laidev = np.asarray([False, True, False])
    epot_air = np.asarray([282000.0], dtype=np.float64)
    petAcoef = np.asarray([0.01], dtype=np.float64)
    petBcoef = np.asarray([281500.0], dtype=np.float64)
    qair = np.asarray([0.006], dtype=np.float64)
    peqAcoef = np.asarray([0.1], dtype=np.float64)
    peqBcoef = np.asarray([0.012], dtype=np.float64)
    soilflx = np.asarray([8.0], dtype=np.float64)
    soilflx_pft = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    rau = np.asarray([1.2], dtype=np.float64)
    u = np.asarray([1.0], dtype=np.float64)
    v = np.asarray([0.5], dtype=np.float64)
    q_cdrag = np.asarray([0.01], dtype=np.float64)
    q_cdrag_pft = np.asarray([[0.011, 0.012, 0.013]], dtype=np.float64)
    vbeta = np.asarray([0.4], dtype=np.float64)
    vbeta_pft = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)
    valpha = np.asarray([1.0], dtype=np.float64)
    vbeta1 = np.asarray([0.2], dtype=np.float64)
    vbeta2 = np.asarray([[0.10, 0.05, 0.02]], dtype=np.float64)
    vbeta3 = np.asarray([[0.30, 0.10, 0.03]], dtype=np.float64)
    vbeta3pot = np.asarray([[0.40, 0.20, 0.04]], dtype=np.float64)
    vbeta4 = np.asarray([0.3], dtype=np.float64)
    vbeta4_pft = np.asarray([[0.30, 0.20, 0.10]], dtype=np.float64)
    vbeta5 = np.asarray([0.05], dtype=np.float64)
    soilcap = np.asarray([5.0e5], dtype=np.float64)
    soilcap_pft = np.asarray([[4.0e5, 4.5e5, 5.0e5]], dtype=np.float64)
    veget_max = np.asarray([[1.0, 0.8, 1.0]], dtype=np.float64)
    dt_sechiba = 1800.0

    step = enerbil_explicit_local_step(
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        lwdown=lwdown,
        swnet=swnet,
        pb=pb,
        emis=emis,
        ok_laidev=ok_laidev,
        epot_air=epot_air,
        petAcoef=petAcoef,
        petBcoef=petBcoef,
        qair=qair,
        peqAcoef=peqAcoef,
        peqBcoef=peqBcoef,
        soilflx=soilflx,
        soilflx_pft=soilflx_pft,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        vbeta=vbeta,
        vbeta_pft=vbeta_pft,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta3pot=vbeta3pot,
        vbeta4=vbeta4,
        vbeta4_pft=vbeta4_pft,
        vbeta5=vbeta5,
        soilcap=soilcap,
        soilcap_pft=soilcap_pft,
        veget_max=veget_max,
        dt_sechiba=dt_sechiba,
    )

    begin = enerbil_begin_local_diagnostics(
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        lwdown=lwdown,
        swnet=swnet,
        pb=pb,
        emis=emis,
        ok_laidev=ok_laidev,
    )
    np.testing.assert_allclose(np.asarray(step.begin.psold), np.asarray(begin.psold))
    np.testing.assert_allclose(np.asarray(step.surftemp.psnew), np.asarray(begin.psold) + np.asarray(step.surftemp.dtheta))

    flux = enerbil_flux_local_diagnostics(
        emis=emis,
        temp_sol=temp_sol,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        vbeta=vbeta,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        qair=step.surftemp.qair_new,
        epot_air=step.surftemp.epot_air_new,
        psnew=step.surftemp.psnew,
        qsol_sat_new=step.surftemp.qsol_sat_new,
        temp_sol_new=step.surftemp.temp_sol_new,
        lwdown=lwdown,
        swnet=swnet,
        dt_sechiba=dt_sechiba,
    )
    np.testing.assert_allclose(np.asarray(step.flux.qsurf), np.asarray(flux.qsurf))
    np.testing.assert_allclose(np.asarray(step.evapveg_grid.vevapnu), np.asarray(enerbil_evapveg_grid_fluxes(
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta4=vbeta4,
        vbeta5=vbeta5,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        qair=step.surftemp.qair_new,
        qsol_sat_new=step.surftemp.qsol_sat_new,
        dt_sechiba=dt_sechiba,
    ).vevapnu))
    assert np.asarray(step.evapveg_pft.transpir).shape == (1, 3)


def test_enerbil_flux_local_diagnostics_match_fortran_formula():
    dt_sechiba = 1800.0
    min_wind = 0.1
    emis = np.array([0.96, 0.98])
    temp_sol = np.array([290.0, 300.0])
    temp_sol_new = np.array([291.0, 299.0])
    rau = np.array([1.2, 1.1])
    u = np.array([0.03, 3.0])
    v = np.array([0.04, 4.0])
    q_cdrag = np.array([0.01, 0.02])
    vbeta = np.array([0.5, 0.3])
    valpha = np.array([0.8, 1.1])
    vbeta1 = np.array([0.2, 0.8])
    vbeta5 = np.array([0.1, 0.0])
    qair = np.array([0.010, 0.009])
    epot_air = np.array([290000.0, 300000.0])
    psnew = np.array([292000.0, 299000.0])
    qsol_sat_new = np.array([0.014, 0.008])
    lwdown = np.array([350.0, 360.0])
    swnet = np.array([120.0, 80.0])

    result = enerbil_flux_local_diagnostics(
        emis=emis,
        temp_sol=temp_sol,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        vbeta=vbeta,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        qair=qair,
        epot_air=epot_air,
        psnew=psnew,
        qsol_sat_new=qsol_sat_new,
        temp_sol_new=temp_sol_new,
        lwdown=lwdown,
        swnet=swnet,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )

    speed = np.maximum(min_wind, np.sqrt(u * u + v * v))
    qc = speed * q_cdrag
    lwup = (
        emis * C_STEFAN * temp_sol**4
        + 4.0 * emis * C_STEFAN * temp_sol**3 * (temp_sol_new - temp_sol)
        + (1.0 - emis) * lwdown
    )
    snow_or_flood_beta = vbeta1 * (1.0 - vbeta5) + vbeta5
    evap_beta = (1.0 - vbeta1) * (1.0 - vbeta5) * vbeta
    qsurf_raw = (snow_or_flood_beta + evap_beta * valpha) * qsol_sat_new
    vevapp = dt_sechiba * rau * qc * (
        snow_or_flood_beta * (qsol_sat_new - qair)
        + evap_beta * (valpha * qsol_sat_new - qair)
    )
    fluxsubli = CHALSU0 * rau * qc * vbeta1 * (1.0 - vbeta5) * (qsol_sat_new - qair)
    fluxlat = (
        fluxsubli
        + CHALEV0 * rau * qc * vbeta5 * (qsol_sat_new - qair)
        + CHALEV0 * rau * qc * evap_beta * (valpha * qsol_sat_new - qair)
    )

    np.testing.assert_allclose(result.lwup, lwup)
    np.testing.assert_allclose(result.tsol_rad, (lwup / (emis * C_STEFAN)) ** 0.25)
    np.testing.assert_allclose(result.qsurf, np.maximum(qsurf_raw, qair))
    np.testing.assert_allclose(result.netrad, lwdown + swnet - lwup)
    np.testing.assert_allclose(result.vevapp, vevapp)
    np.testing.assert_allclose(result.fluxlat, fluxlat)
    np.testing.assert_allclose(result.fluxsubli, fluxsubli)
    np.testing.assert_allclose(result.fluxsens, rau * qc * (psnew - epot_air))
    np.testing.assert_allclose(result.lwnet, lwdown - lwup)
    np.testing.assert_allclose(
        result.evapot,
        np.maximum(0.0, dt_sechiba * rau * qc * (qsol_sat_new - qair)),
    )
    assert result.qsurf[1] == pytest.approx(qair[1])
    assert result.evapot[1] == pytest.approx(0.0)


def test_enerbil_flux_evapot_corr_matches_fortran_milly_correction():
    dt_sechiba = 1800.0
    min_wind = 0.1
    emis = np.array([0.96, 0.98], dtype=np.float64)
    temp_sol = np.array([290.0, 300.0], dtype=np.float64)
    temp_sol_new = np.array([291.0, 299.0], dtype=np.float64)
    rau = np.array([1.2, 1.1], dtype=np.float64)
    u = np.array([0.03, 3.0], dtype=np.float64)
    v = np.array([0.04, 4.0], dtype=np.float64)
    q_cdrag = np.array([0.01, 0.02], dtype=np.float64)
    vbeta = np.array([0.5, 0.3], dtype=np.float64)
    valpha = np.array([0.8, 1.1], dtype=np.float64)
    vbeta1 = np.array([0.2, 0.8], dtype=np.float64)
    vbeta5 = np.array([0.1, 0.0], dtype=np.float64)
    qair = np.array([0.010, 0.009], dtype=np.float64)
    epot_air = np.array([CP_AIR * 289.5, CP_AIR * 299.0], dtype=np.float64)
    psnew = np.array([CP_AIR * 292.0, CP_AIR * 299.0], dtype=np.float64)
    qsol_sat_new = np.array([0.014, 0.008], dtype=np.float64)
    lwdown = np.array([350.0, 360.0], dtype=np.float64)
    swnet = np.array([120.0, 80.0], dtype=np.float64)
    pb = np.array([1000.0, 990.0], dtype=np.float64)

    flux = enerbil_flux_local_diagnostics(
        emis=emis,
        temp_sol=temp_sol,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        vbeta=vbeta,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        qair=qair,
        epot_air=epot_air,
        psnew=psnew,
        qsol_sat_new=qsol_sat_new,
        temp_sol_new=temp_sol_new,
        lwdown=lwdown,
        swnet=swnet,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )
    result = enerbil_flux_evapot_corr(
        flux=flux,
        emis=emis,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        epot_air=epot_air,
        psnew=psnew,
        pb=pb,
        min_wind=min_wind,
    )

    speed = np.maximum(min_wind, np.sqrt(u * u + v * v))
    qc = speed * q_cdrag
    tair = epot_air / CP_AIR
    grad_qsat = np.asarray(qsat_moisture_dev_qsatcalc(tair, pb))
    evapot = np.asarray(flux.evapot)
    vevapp = np.asarray(flux.vevapp)
    raw = (
        4.0 * emis * C_STEFAN * tair**3
        + rau * qc * CP_AIR
        + CHALEV0 * rau * qc * grad_qsat * vevapp / np.where(evapot != 0.0, evapot, 1.0)
    )
    active = (evapot > 0.0) & ((psnew - epot_air) != 0.0)
    correction = np.where(
        active & (np.abs(raw) > 1.0e-8),
        CHALEV0
        * rau
        * qc
        * grad_qsat
        * (1.0 - vevapp / np.where(evapot != 0.0, evapot, 1.0))
        / raw,
        np.where(active, raw, 0.0),
    )
    correction = np.maximum(0.0, correction)
    np.testing.assert_allclose(np.asarray(result.correction), correction)
    np.testing.assert_allclose(np.asarray(result.evapot_corr), evapot / (1.0 + correction))
    assert np.asarray(result.evapot_corr)[0] <= evapot[0]
    assert np.asarray(result.evapot_corr)[1] == pytest.approx(0.0)


def test_enerbil_pottemp_and_t2mdiag_follow_current_fortran_pass_throughs():
    q_sol_pot = np.asarray([0.010, 0.012], dtype=np.float64)
    temp_sol_pot = np.asarray([289.0, 300.0], dtype=np.float64)
    temp_air = np.asarray([288.0, 301.0], dtype=np.float64)

    pot = enerbil_pottemp_pass_through(q_sol_pot=q_sol_pot, temp_sol_pot=temp_sol_pot)

    np.testing.assert_allclose(np.asarray(pot.q_sol_pot), q_sol_pot)
    np.testing.assert_allclose(np.asarray(pot.temp_sol_pot), temp_sol_pot)
    np.testing.assert_allclose(np.asarray(enerbil_t2mdiag(temp_air)), temp_air)


def test_enerbil_flux_explicit_snow_diagnostics_match_fortran_ablation_branch():
    dt_sechiba = 1800.0
    min_wind = 0.1
    emis = np.array([0.97, 0.96], dtype=np.float64)
    temp_sol = np.array([272.5, 271.0], dtype=np.float64)
    temp_sol_new = np.array([274.4, 272.0], dtype=np.float64)
    lwdown = np.array([315.0, 310.0], dtype=np.float64)
    swnet = np.array([45.0, 20.0], dtype=np.float64)
    rau = np.array([1.25, 1.20], dtype=np.float64)
    u = np.array([1.0, 0.03], dtype=np.float64)
    v = np.array([2.0, 0.04], dtype=np.float64)
    q_cdrag = np.array([0.015, 0.012], dtype=np.float64)
    vbeta = np.array([0.55, 0.40], dtype=np.float64)
    valpha = np.array([0.90, 0.95], dtype=np.float64)
    vbeta1 = np.array([0.70, 0.30], dtype=np.float64)
    vbeta5 = np.array([0.10, 0.00], dtype=np.float64)
    qair = np.array([0.0030, 0.0040], dtype=np.float64)
    epot_air = np.array([CP_AIR * 272.0, CP_AIR * 271.5], dtype=np.float64)
    psnew = np.array([CP_AIR * temp_sol_new[0], CP_AIR * temp_sol_new[1]], dtype=np.float64)
    pb = np.array([1000.0, 1005.0], dtype=np.float64)
    qsol_sat_new = np.asarray(qsat_moisture_qsatcalc(temp_sol_new, pb))
    precip_rain = np.array([2.0e-5, 3.0e-5], dtype=np.float64)
    snowdz = np.array([[0.02, 0.01, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    temp_air = np.array([276.0, 270.0], dtype=np.float64)
    pgflux_in = np.array([-999.0, -888.0], dtype=np.float64)
    soilcap = np.array([5.0e5, 6.0e5], dtype=np.float64)

    flux = enerbil_flux_local_diagnostics(
        emis=emis,
        temp_sol=temp_sol,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        vbeta=vbeta,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        qair=qair,
        epot_air=epot_air,
        psnew=psnew,
        qsol_sat_new=qsol_sat_new,
        temp_sol_new=temp_sol_new,
        lwdown=lwdown,
        swnet=swnet,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )
    result = enerbil_flux_explicit_snow_diagnostics(
        flux=flux,
        emis=emis,
        temp_sol=temp_sol,
        temp_sol_new=temp_sol_new,
        lwdown=lwdown,
        swnet=swnet,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        vbeta=vbeta,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        qair=qair,
        epot_air=epot_air,
        qsol_sat_new=qsol_sat_new,
        pb=pb,
        precip_rain=precip_rain,
        snowdz=snowdz,
        temp_air=temp_air,
        pgflux=pgflux_in,
        soilcap=soilcap,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )

    speed = np.maximum(min_wind, np.sqrt(u * u + v * v))
    qc = speed * q_cdrag
    phpsnow = precip_rain * 4.218e3 * (np.maximum(TP_00, temp_air) - TP_00) / dt_sechiba
    pgflux_raw = np.asarray(flux.netrad) - np.asarray(flux.fluxsens) - np.asarray(flux.fluxlat) + phpsnow
    qsol_sat_tmp = np.asarray(qsat_moisture_qsatcalc(np.full_like(pb, TP_00), pb))
    lwup_tmp = (
        emis * C_STEFAN * temp_sol**4
        + 4.0 * emis * C_STEFAN * temp_sol**3 * (TP_00 - temp_sol)
        + (1.0 - emis) * lwdown
    )
    netrad_tmp = lwdown + swnet - lwup_tmp
    fluxsens_tmp = rau * qc * CP_AIR * (TP_00 - epot_air / CP_AIR)
    fluxlat_tmp = (
        CHALSU0 * rau * qc * vbeta1 * (1.0 - vbeta5) * (qsol_sat_tmp - qair)
        + CHALEV0 * rau * qc * vbeta5 * (qsol_sat_tmp - qair)
        + CHALEV0
        * rau
        * qc
        * (1.0 - vbeta1)
        * (1.0 - vbeta5)
        * vbeta
        * (valpha * qsol_sat_tmp - qair)
    )
    zgflux = netrad_tmp - fluxsens_tmp - fluxlat_tmp + phpsnow
    active = (temp_sol_new > TP_00) & (np.sum(snowdz, axis=1) > 0.0) & (soilcap > 0.0)
    expected_pgflux = np.where(active, zgflux, pgflux_raw)
    expected_temp_sol_add = np.where(active, -(pgflux_raw - zgflux) * dt_sechiba / soilcap, 0.0)

    np.testing.assert_allclose(np.asarray(result.phpsnow), phpsnow)
    np.testing.assert_allclose(np.asarray(result.pgflux), expected_pgflux)
    np.testing.assert_allclose(np.asarray(result.temp_sol_add), expected_temp_sol_add)
    assert np.asarray(result.temp_sol_add)[0] != pytest.approx(0.0)
    assert np.asarray(result.temp_sol_add)[1] == pytest.approx(0.0)

    disabled = enerbil_flux_explicit_snow_diagnostics(
        flux=flux,
        emis=emis,
        temp_sol=temp_sol,
        temp_sol_new=temp_sol_new,
        lwdown=lwdown,
        swnet=swnet,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        vbeta=vbeta,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        qair=qair,
        epot_air=epot_air,
        qsol_sat_new=qsol_sat_new,
        pb=pb,
        precip_rain=precip_rain,
        snowdz=snowdz,
        temp_air=temp_air,
        pgflux=pgflux_in,
        soilcap=soilcap,
        dt_sechiba=dt_sechiba,
        ok_explicitsnow=False,
        min_wind=min_wind,
    )
    np.testing.assert_allclose(np.asarray(disabled.pgflux), pgflux_in)
    np.testing.assert_allclose(np.asarray(disabled.temp_sol_add), np.zeros_like(pgflux_in))


def test_enerbil_evapveg_grid_fluxes_match_fortran_formula():
    dt_sechiba = 1800.0
    min_wind = 0.1
    vbeta1 = np.array([0.2, 0.8])
    vbeta5 = np.array([0.1, 0.0])
    vbeta4 = np.array([0.3, 0.4])
    rau = np.array([1.2, 1.1])
    u = np.array([0.03, 3.0])
    v = np.array([0.04, 4.0])
    q_cdrag = np.array([0.01, 0.02])
    qair = np.array([0.010, 0.006])
    qsol_sat_new = np.array([0.014, 0.008])
    vbeta2 = np.array([[0.10, 0.05, 0.00], [0.20, 0.10, 0.05]])
    vbeta3 = np.array([[0.30, 0.10, 0.05], [0.20, 0.15, 0.05]])

    result = enerbil_evapveg_grid_fluxes(
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta4=vbeta4,
        vbeta5=vbeta5,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        qair=qair,
        qsol_sat_new=qsol_sat_new,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )

    speed = np.maximum(min_wind, np.sqrt(u * u + v * v))
    exchange = dt_sechiba * rau * speed * q_cdrag * (qsol_sat_new - qair)

    np.testing.assert_allclose(result.vevapsno, (1.0 - vbeta5) * vbeta1 * exchange)
    np.testing.assert_allclose(
        result.vevapnu,
        (1.0 - vbeta1) * (1.0 - vbeta5) * vbeta4 * exchange,
    )
    np.testing.assert_allclose(
        result.vevapflo,
        vbeta5 * (1.0 - np.sum(vbeta2, axis=1) - np.sum(vbeta3, axis=1)) * exchange,
    )


def test_diffuco_bare_cwrr_beta_feeds_enerbil_evapveg_fluxes():
    dt_sechiba = 1800.0
    min_wind = 0.1
    evap_bare_lim = np.asarray([0.35], dtype=np.float64)
    vbeta1 = np.asarray([0.1], dtype=np.float64)
    vbeta5 = np.asarray([0.0], dtype=np.float64)
    vbeta2 = np.asarray([[0.05, 0.10, 0.00]], dtype=np.float64)
    vbeta3 = np.asarray([[0.20, 0.15, 0.00]], dtype=np.float64)
    vbeta3pot = np.asarray([[0.25, 0.20, 0.00]], dtype=np.float64)
    veget_max = np.asarray([[0.50, 0.30, 0.00]], dtype=np.float64)
    rau = np.asarray([1.2], dtype=np.float64)
    u = np.asarray([0.03], dtype=np.float64)
    v = np.asarray([0.04], dtype=np.float64)
    q_cdrag = np.asarray([0.01], dtype=np.float64)
    qair = np.asarray([0.010], dtype=np.float64)
    qsol_sat_new = np.asarray([0.014], dtype=np.float64)

    bare = diffuco_bare_cwrr_beta(
        evap_bare_lim=evap_bare_lim,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        veget_max=veget_max,
    )
    grid = enerbil_evapveg_grid_fluxes(
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta4=bare.vbeta4,
        vbeta5=vbeta5,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        qair=qair,
        qsol_sat_new=qsol_sat_new,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )
    pft = enerbil_evapveg_pft_fluxes(
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta3pot=vbeta3pot,
        vbeta4_pft=bare.vbeta4_pft,
        vbeta5=vbeta5,
        veget_max=veget_max,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        qair=qair,
        qsol_sat_new=qsol_sat_new,
        ok_laidev=np.asarray([False, False, False]),
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )

    speed = np.maximum(min_wind, np.sqrt(u * u + v * v))
    exchange = dt_sechiba * rau * speed * q_cdrag * (qsol_sat_new - qair)
    expected_vbeta4 = np.minimum(evap_bare_lim, 1.0 - np.sum(vbeta2 + vbeta3, axis=1))
    expected_pft_beta = np.where(
        veget_max > 0.0,
        np.minimum(evap_bare_lim[:, None], veget_max - (vbeta2 + vbeta3)),
        0.0,
    )

    np.testing.assert_allclose(np.asarray(bare.vbeta4), expected_vbeta4)
    np.testing.assert_allclose(np.asarray(bare.vbeta4_pft), expected_pft_beta)
    np.testing.assert_allclose(np.asarray(grid.vevapnu), (1.0 - vbeta1) * expected_vbeta4 * exchange)
    np.testing.assert_allclose(
        np.asarray(pft.vevapnu_pft),
        np.where(veget_max > 0.0, (1.0 - vbeta1[:, None]) * expected_pft_beta * exchange[:, None], 0.0),
    )
    assert np.asarray(pft.vevapnu_pft)[0, 2] == pytest.approx(0.0)


def test_enerbil_evapveg_pft_fluxes_match_fortran_ok_laidev_contract():
    dt_sechiba = 1800.0
    min_wind = 0.1
    vbeta1 = np.array([0.2, 0.6])
    vbeta5 = np.array([0.1, 0.2])
    rau = np.array([1.2, 1.1])
    u = np.array([0.03, 3.0])
    v = np.array([0.04, 4.0])
    q_cdrag = np.array([0.01, 0.02])
    q_cdrag_pft = np.array([[0.011, 0.012, 0.013], [0.021, 0.022, 0.023]])
    qair = np.array([0.010, 0.006])
    qsol_sat_new = np.array([0.014, 0.008])
    qsol_sat_new_pft = np.array([[0.015, 0.016, 0.017], [0.009, 0.010, 0.011]])
    vbeta2 = np.array([[0.10, 0.05, 0.02], [0.20, 0.10, 0.05]])
    vbeta3 = np.array([[0.30, 0.10, 0.03], [0.20, 0.15, 0.05]])
    vbeta3pot = np.array([[0.40, 0.20, 0.04], [0.30, 0.25, 0.06]])
    vbeta4_pft = np.array([[0.30, 0.20, 0.10], [0.25, 0.35, 0.45]])
    veget_max = np.array([[1.0, 1.0, 0.0], [1.0, 0.5, 1.0]])
    ok_laidev = np.array([False, True, False])

    result = enerbil_evapveg_pft_fluxes(
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta3pot=vbeta3pot,
        vbeta4_pft=vbeta4_pft,
        vbeta5=vbeta5,
        veget_max=veget_max,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        qair=qair,
        qsol_sat_new=qsol_sat_new,
        qsol_sat_new_pft=qsol_sat_new_pft,
        ok_laidev=ok_laidev,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )

    speed = np.maximum(min_wind, np.sqrt(u * u + v * v))
    grid_xx = (
        dt_sechiba
        * (1.0 - vbeta1)
        * (qsol_sat_new - qair)
        * rau
        * speed
        * q_cdrag
    )
    pft_xx = (
        dt_sechiba
        * (1.0 - vbeta1[:, None])
        * (qsol_sat_new_pft - qair[:, None])
        * rau[:, None]
        * speed[:, None]
        * q_cdrag_pft
    )
    xxtemp = np.where(ok_laidev[None, :], pft_xx, grid_xx[:, None])
    expected_vevapnu = np.where(
        ok_laidev[None, :],
        (1.0 - vbeta1[:, None])
        * (1.0 - vbeta5[:, None])
        * vbeta4_pft
        * dt_sechiba
        * rau[:, None]
        * speed[:, None]
        * q_cdrag_pft
        * (qsol_sat_new_pft - qair[:, None]),
        (1.0 - vbeta1[:, None])
        * (1.0 - vbeta5[:, None])
        * vbeta4_pft
        * dt_sechiba
        * rau[:, None]
        * speed[:, None]
        * q_cdrag[:, None]
        * (qsol_sat_new[:, None] - qair[:, None]),
    )
    expected_vevapnu = np.where(veget_max > 0.0, expected_vevapnu, 0.0)

    np.testing.assert_allclose(result.vevapnu_pft, expected_vevapnu)
    np.testing.assert_allclose(result.vevapwet, xxtemp * vbeta2)
    np.testing.assert_allclose(result.transpir, xxtemp * vbeta3)
    np.testing.assert_allclose(result.transpot, xxtemp * vbeta3pot)
    assert result.vevapnu_pft[0, 2] == pytest.approx(0.0)


def test_enerbil_evapveg_pft_fluxes_requires_pft_inputs_for_ok_laidev():
    with pytest.raises(ValueError, match="q_cdrag_pft and qsol_sat_new_pft are required"):
        enerbil_evapveg_pft_fluxes(
            vbeta1=np.array([0.2]),
            vbeta2=np.array([[0.1, 0.0]]),
            vbeta3=np.array([[0.2, 0.0]]),
            vbeta3pot=np.array([[0.3, 0.0]]),
            vbeta4_pft=np.array([[0.4, 0.0]]),
            vbeta5=np.array([0.0]),
            veget_max=np.array([[1.0, 1.0]]),
            rau=np.array([1.2]),
            u=np.array([1.0]),
            v=np.array([0.0]),
            q_cdrag=np.array([0.01]),
            qair=np.array([0.010]),
            qsol_sat_new=np.array([0.014]),
            ok_laidev=np.array([False, True]),
            dt_sechiba=1800.0,
        )


def test_enerbil_after_main_payload_remains_readable_but_is_not_formula_input():
    payload = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    contract = enerbil_flux_inputs_contract(payload)

    assert payload["tag"] == "after_enerbil_main"
    assert payload["kjit"] == 1
    assert payload["ji"] == 1
    assert payload["jv"] == 14
    assert payload["transpir"] == pytest.approx(0.0)
    assert payload["transpot"] == pytest.approx(0.0)
    assert payload["vevapnu_pft"] == pytest.approx(0.0)
    assert payload["evapot"] == pytest.approx(5.719239060617407e-2)
    assert payload["temp_sol_new_pft"] == pytest.approx(295.744633030835)
    assert not contract.ok
    assert "valpha" in contract.missing_inputs
    assert "qsol_sat_new" in contract.missing_inputs
    assert "q_cdrag" in contract.missing_inputs
    assert "qsurf" in contract.after_main_outputs_only
    assert "evapot" in contract.after_main_outputs_only


def test_driver_swnet_from_swdown_albedo_requires_exact_two_band_albedo():
    swdown = np.asarray([100.0, 200.0], dtype=np.float64)
    albedo = np.asarray([[0.10, 0.30], [0.20, 0.40]], dtype=np.float64)

    result = driver_swnet_from_swdown_albedo(swdown=swdown, albedo=albedo)

    np.testing.assert_allclose(result, (1.0 - (albedo[:, 0] + albedo[:, 1]) / 2.0) * swdown)
    with pytest.raises(ValueError, match="albedo must have shape"):
        driver_swnet_from_swdown_albedo(swdown=swdown, albedo=np.asarray([[0.2], [0.3]]))


def test_driver_albedo_restart_closes_swnet_formula_only_with_swdown():
    restart = read_driver_albedo_restart(
        ROOT
        / "reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0"
        / "001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/driver_start.nc"
    )
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]

    np.testing.assert_allclose(
        restart.albedo,
        np.asarray([[0.04526853859061869, 0.22753724443710746]], dtype=np.float64),
    )

    without_albedo = enerbil_swnet_input_chain_coverage(driver_or_intersurf_payload=intersurf)
    assert not without_albedo.ok
    assert "swdown" in without_albedo.covered_by_trace
    assert "swnet" in without_albedo.missing
    assert "albedo_vis" in without_albedo.missing_formula_inputs
    assert "albedo_nir" in without_albedo.missing_formula_inputs

    with_albedo = enerbil_swnet_input_chain_coverage(
        driver_or_intersurf_payload=intersurf,
        driver_albedo_payload=restart.as_coverage_payload(),
    )
    assert with_albedo.ok
    assert "swnet" in with_albedo.covered_by_source_kernel
    assert "albedo" in with_albedo.covered_by_source_kernel
    assert with_albedo.missing_formula_inputs == ()

    swnet = driver_swnet_from_swdown_albedo(
        swdown=np.asarray([intersurf["swdown"]], dtype=np.float64),
        albedo=restart.albedo,
    )
    expected = (1.0 - (restart.albedo[:, 0] + restart.albedo[:, 1]) / 2.0) * intersurf["swdown"]
    np.testing.assert_allclose(swnet, expected)


def test_condveg_initialized_emis_follows_initialize_impaze_branches():
    default_branch = condveg_initialized_emis(npts=3, impaze=False)
    np.testing.assert_allclose(np.asarray(default_branch), np.ones(3))

    imposed_branch = condveg_initialized_emis(npts=2, impaze=True, emis_scal=0.98)
    np.testing.assert_allclose(np.asarray(imposed_branch), np.asarray([0.98, 0.98]))

    with pytest.raises(ValueError, match="emis_scal is required"):
        condveg_initialized_emis(npts=1, impaze=True)


def test_sechiba_var_init_air_density_helper_matches_fortran_formula():
    pb = np.asarray([987.963802083333, 1000.0], dtype=np.float64)
    temp_air = np.asarray([289.76685333252, 300.0], dtype=np.float64)

    result = sechiba_air_density_from_pb_temp_air(pb, temp_air)

    expected = PA_PAR_HPA * pb / (CTE_MOLR * temp_air)
    np.testing.assert_allclose(np.asarray(result), expected)


def test_non_watchout_driver_energy_coupling_helper_is_path_specific_formula():
    temp_air = np.asarray([289.76685333252, 300.0], dtype=np.float64)
    zlev = np.asarray([119.534964329292, 2.0], dtype=np.float64)
    qair = np.asarray([0.009809188079088926, 0.012], dtype=np.float64)

    result = dim2_driver_non_watchout_energy_coupling_inputs(temp_air, zlev, qair)

    expected_epot = CP_AIR * temp_air + CTE_GRAV * zlev
    np.testing.assert_allclose(np.asarray(result["epot_air"]), expected_epot)
    np.testing.assert_allclose(np.asarray(result["petBcoef"]), expected_epot)
    np.testing.assert_allclose(np.asarray(result["peqBcoef"]), qair)
    np.testing.assert_allclose(np.asarray(result["petAcoef"]), np.zeros_like(temp_air))
    np.testing.assert_allclose(np.asarray(result["peqAcoef"]), np.zeros_like(qair))


def test_enerbil_surface_state_coverage_keeps_after_enerbil_fields_after_boundary():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    after_enerbil = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]

    coverage = enerbil_surface_state_input_coverage(
        after_diffuco_payload=after_diffuco,
        driver_or_intersurf_payload=intersurf,
        after_enerbil_payload=after_enerbil,
    )

    assert not coverage.ok
    assert "temp_sol" in coverage.covered_by_trace
    assert "temp_sol_pft" in coverage.covered_by_trace
    assert "swnet" in coverage.missing
    assert "emis" in coverage.missing
    assert "soilflx" in coverage.missing
    assert "soilflx_pft" in coverage.missing
    assert "soilcap" in coverage.missing
    assert "soilcap_pft" in coverage.missing
    assert "soilcap" in coverage.after_boundary_outputs_only
    assert "soilcap_pft" in coverage.after_boundary_outputs_only
    assert "temp_sol" not in coverage.after_boundary_outputs_only
    assert "swdown" in intersurf
    assert "swnet" not in coverage.covered_by_trace


def test_enerbil_surface_state_coverage_closes_only_with_advertised_audited_sources():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    coverage = enerbil_surface_state_input_coverage(
        after_diffuco_payload=after_diffuco,
        source_kernel_outputs=(
            "swnet",
            "emis",
            "soilflx",
            "soilflx_pft",
            "soilcap",
            "soilcap_pft",
        ),
    )

    assert coverage.ok
    assert "temp_sol" in coverage.covered_by_trace
    assert "temp_sol_pft" in coverage.covered_by_trace
    assert "swnet" in coverage.covered_by_source_kernel
    assert "emis" in coverage.covered_by_source_kernel
    assert "soilflx" in coverage.covered_by_source_kernel
    assert "soilflx_pft" in coverage.covered_by_source_kernel
    assert "soilcap" in coverage.covered_by_source_kernel
    assert "soilcap_pft" in coverage.covered_by_source_kernel


def test_read_enerbil_soil_thermal_state_restart_reads_first_step_truth():
    thermal = read_enerbil_soil_thermal_state_restart(REFERENCE_SECHIBA_START)

    assert thermal.path == REFERENCE_SECHIBA_START
    assert thermal.soilcap.shape == (1,)
    assert thermal.soilflx.shape == (1,)
    assert thermal.soilcap_pft.shape == (1, 14)
    assert thermal.soilflx_pft.shape == (1, 14)
    np.testing.assert_allclose(thermal.soilcap, [45865.2880547002])
    np.testing.assert_allclose(thermal.soilflx, [-34.047520966054336])
    np.testing.assert_allclose(thermal.soilcap_pft[0, 13], 45865.2880547002)
    np.testing.assert_allclose(thermal.soilflx_pft[0, 13], -34.047520966054336)
    assert "thermosoil_initialize lines 668-684" in thermal.provenance[0]


def test_enerbil_surface_state_coverage_closes_soil_thermal_state_from_restart_not_after_enerbil():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    after_enerbil = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    thermal = read_enerbil_soil_thermal_state_restart(REFERENCE_SECHIBA_START)

    coverage = enerbil_surface_state_input_coverage(
        after_diffuco_payload=after_diffuco,
        after_enerbil_payload=after_enerbil,
        soil_thermal_restart_payload=thermal.as_coverage_payload(),
    )

    assert "soilcap" in coverage.covered_by_source_kernel
    assert "soilcap_pft" in coverage.covered_by_source_kernel
    assert "soilflx" in coverage.covered_by_source_kernel
    assert "soilflx_pft" in coverage.covered_by_source_kernel
    assert "soilcap" not in coverage.after_boundary_outputs_only
    assert "soilcap_pft" not in coverage.after_boundary_outputs_only
    assert coverage.source_kernel_sources["soilcap"].startswith(
        "jax_orchidee.sechiba.enerbil.read_enerbil_soil_thermal_state_restart"
    )
    assert after_enerbil["soilcap_pft"] == pytest.approx(59764.3072135468)
    assert thermal.soilcap_pft[0, 13] == pytest.approx(45865.2880547002)
    assert "swnet" in coverage.missing
    assert "emis" in coverage.missing


def test_enerbil_surface_state_coverage_closes_emis_from_audited_condveg_initialize():
    after_diffuco = first_pft14_diffuco_after_main_payload()

    missing_branch = enerbil_surface_state_input_coverage(
        after_diffuco_payload=after_diffuco,
        condveg_initialize_executed=True,
        condveg_impaze=True,
    )
    assert "emis" in missing_branch.missing

    coverage = enerbil_surface_state_input_coverage(
        after_diffuco_payload=after_diffuco,
        condveg_initialize_executed=True,
        condveg_impaze=False,
    )

    assert not coverage.ok
    assert "emis" in coverage.covered_by_source_kernel
    assert coverage.source_kernel_sources["emis"].startswith(
        "jax_orchidee.sechiba.enerbil.condveg_initialized_emis"
    )
    assert "swnet" in coverage.missing
    assert "soilcap" in coverage.missing


def test_enerbil_first_step_coverage_separates_trace_source_kernel_and_missing_inputs():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    after_enerbil = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]

    coverage = enerbil_first_step_input_coverage(
        after_diffuco_payload=after_diffuco,
        driver_or_intersurf_payload=intersurf,
        after_enerbil_payload=after_enerbil,
        diffuco_comb_executed=True,
        enerbil_begin_executed=True,
    )

    assert not coverage.ok
    assert "vbeta" in coverage.covered_by_trace
    assert "vbeta2" in coverage.covered_by_trace
    assert "vbeta3" in coverage.covered_by_trace
    assert "q_cdrag" in coverage.covered_by_trace
    assert "q_cdrag_pft" in coverage.covered_by_trace
    assert "temp_sol" in coverage.covered_by_trace
    assert "temp_sol_pft" in coverage.covered_by_trace
    assert "u" in coverage.covered_by_trace
    assert "v" in coverage.covered_by_trace
    assert "qair" in coverage.covered_by_trace
    assert "lwdown" in coverage.covered_by_trace
    assert "swdown" in coverage.covered_by_trace
    assert "temp_air" in coverage.covered_by_trace
    assert "pb" in coverage.covered_by_trace
    assert "zlev" not in coverage.covered_by_trace
    assert "soilcap" in coverage.missing
    assert "soilcap_pft" in coverage.missing
    assert "soilflx" in coverage.missing
    assert "soilflx_pft" in coverage.missing
    assert "valpha" in coverage.covered_by_source_kernel
    assert coverage.source_kernel_sources["valpha"].startswith(
        "jax_orchidee.sechiba.diffuco.diffuco_comb_explicit"
    )
    assert "qsol_sat_new" in coverage.missing
    assert "qsol_sat_new_pft" in coverage.missing
    assert "qsol_sat" in coverage.missing
    assert "pdqsold" in coverage.missing
    assert "rau" in coverage.missing
    assert "epot_air" in coverage.missing
    assert "petAcoef" in coverage.missing
    assert "petBcoef" in coverage.missing
    assert "peqAcoef" in coverage.missing
    assert "peqBcoef" in coverage.missing
    assert "psold" in coverage.missing
    assert "netrad" in coverage.missing
    assert "dtheta" in coverage.missing
    assert "psnew" in coverage.missing
    assert "swnet" in coverage.missing
    assert "emis" in coverage.missing
    assert "swdown" not in coverage.missing
    assert "temp_sol_new" in coverage.missing
    assert "temp_sol_new_pft" in coverage.missing
    assert "temp_sol_new" in coverage.after_boundary_outputs_only
    assert "qsurf" not in coverage.after_boundary_outputs_only
    assert after_diffuco["q_cdrag"] == pytest.approx(0.002033465050147852)
    assert after_enerbil["soilcap_pft"] == pytest.approx(59764.3072135468)
    assert "soilcap" not in coverage.covered_by_trace
    assert "soilcap_pft" not in coverage.covered_by_trace


def test_enerbil_first_step_coverage_accepts_restart_soil_thermal_state():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    after_enerbil = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]
    thermal = read_enerbil_soil_thermal_state_restart(REFERENCE_SECHIBA_START)

    coverage = enerbil_first_step_input_coverage(
        after_diffuco_payload=after_diffuco,
        driver_or_intersurf_payload=intersurf,
        after_enerbil_payload=after_enerbil,
        soil_thermal_restart_payload=thermal.as_coverage_payload(),
        diffuco_comb_executed=True,
    )

    assert "soilcap" in coverage.covered_by_source_kernel
    assert "soilcap_pft" in coverage.covered_by_source_kernel
    assert "soilflx" in coverage.covered_by_source_kernel
    assert "soilflx_pft" in coverage.covered_by_source_kernel
    assert "soilcap" not in coverage.missing
    assert "soilflx_pft" not in coverage.missing
    assert "soilcap" not in coverage.after_boundary_outputs_only
    assert coverage.source_kernel_sources["soilflx_pft"].startswith(
        "jax_orchidee.sechiba.enerbil.read_enerbil_soil_thermal_state_restart"
    )
    assert "after_enerbil_main fields" in " ".join(coverage.notes)
    assert "swnet" in coverage.missing
    assert "emis" in coverage.missing


def test_enerbil_first_step_coverage_closes_rau_only_after_var_init_preconditions():
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]

    coverage = enerbil_first_step_input_coverage(
        driver_or_intersurf_payload=intersurf,
        sechiba_var_init_executed=True,
    )

    assert "pb" in coverage.covered_by_trace
    assert "temp_air" in coverage.covered_by_trace
    assert "rau" in coverage.covered_by_source_kernel
    assert coverage.source_kernel_sources["rau"].startswith(
        "jax_orchidee.sechiba.enerbil.sechiba_air_density_from_pb_temp_air"
    )
    assert "epot_air" in coverage.missing
    assert "petAcoef" in coverage.missing


def test_enerbil_first_step_coverage_closes_non_watchout_driver_coupling_only_when_audited():
    driver = read_server_records("driver_forcing", tags="first_forcing", limit=1)[0]

    without_branch = enerbil_first_step_input_coverage(driver_or_intersurf_payload=driver)
    assert "zlev" in without_branch.covered_by_trace
    assert "epot_air" in without_branch.missing
    assert "petAcoef" in without_branch.missing

    with_branch = enerbil_first_step_input_coverage(
        driver_or_intersurf_payload=driver,
        non_watchout_driver_executed=True,
    )

    assert "zlev" in with_branch.covered_by_trace
    assert "epot_air" in with_branch.covered_by_source_kernel
    assert "petAcoef" in with_branch.covered_by_source_kernel
    assert "petBcoef" in with_branch.covered_by_source_kernel
    assert "peqAcoef" in with_branch.covered_by_source_kernel
    assert "peqBcoef" in with_branch.covered_by_source_kernel
    assert "zlev is an enerbil_surftemp interface input" in " ".join(with_branch.notes)
    assert "petAcoef" not in with_branch.missing


def test_enerbil_first_step_coverage_closes_swnet_only_with_driver_albedo_chain():
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]
    restart = read_driver_albedo_restart(
        ROOT
        / "reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0"
        / "001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/driver_start.nc"
    )

    without_albedo = enerbil_first_step_input_coverage(driver_or_intersurf_payload=intersurf)
    assert "swdown" in without_albedo.covered_by_trace
    assert "swnet" in without_albedo.missing

    with_albedo = enerbil_first_step_input_coverage(
        driver_or_intersurf_payload=intersurf,
        driver_albedo_payload=restart.as_coverage_payload(),
    )
    assert "swdown" in with_albedo.covered_by_trace
    assert "swnet" in with_albedo.covered_by_source_kernel
    assert "swnet" not in with_albedo.missing
    assert with_albedo.source_kernel_sources["swnet"].startswith(
        "jax_orchidee.sechiba.enerbil.driver_swnet_from_swdown_albedo"
    )


def test_enerbil_first_step_coverage_marks_begin_outputs_when_exact_begin_inputs_are_present():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]

    coverage = enerbil_first_step_input_coverage(
        after_diffuco_payload=after_diffuco,
        driver_or_intersurf_payload=intersurf,
        source_kernel_outputs=("swnet", "emis"),
        diffuco_comb_executed=True,
        enerbil_begin_executed=True,
    )

    assert "swnet" in coverage.covered_by_source_kernel
    assert "emis" in coverage.covered_by_source_kernel
    assert "psold" in coverage.covered_by_source_kernel
    assert "psold_pft" in coverage.covered_by_source_kernel
    assert "qsol_sat" in coverage.covered_by_source_kernel
    assert "qsol_sat_pft" in coverage.covered_by_source_kernel
    assert "pdqsold" in coverage.covered_by_source_kernel
    assert "pdqsold_pft" in coverage.covered_by_source_kernel
    assert "lwabs" in coverage.covered_by_source_kernel
    assert "netrad" in coverage.covered_by_source_kernel
    assert "netrad_pft" in coverage.covered_by_source_kernel
    assert "qsol_sat_new" in coverage.missing
    assert "dtheta" in coverage.missing
    assert "psnew" in coverage.missing
    assert "swdown" in coverage.covered_by_trace
    assert "swdown" not in coverage.source_kernel_sources


def test_enerbil_first_step_coverage_uses_condveg_initialize_emis_for_begin_inputs():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]

    coverage = enerbil_first_step_input_coverage(
        after_diffuco_payload=after_diffuco,
        driver_or_intersurf_payload=intersurf,
        source_kernel_outputs=("swnet",),
        diffuco_comb_executed=True,
        enerbil_begin_executed=True,
        condveg_initialize_executed=True,
        condveg_impaze=False,
    )

    assert "emis" in coverage.covered_by_source_kernel
    assert coverage.source_kernel_sources["emis"].startswith(
        "jax_orchidee.sechiba.enerbil.condveg_initialized_emis"
    )
    assert "psold" in coverage.covered_by_source_kernel
    assert "netrad" in coverage.covered_by_source_kernel
    assert "soilcap" in coverage.missing
    assert "qsol_sat_new" in coverage.missing


def test_enerbil_first_step_coverage_marks_surftemp_outputs_only_with_all_inputs():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]
    audited_surftemp_inputs = (
        "swnet",
        "emis",
        "epot_air",
        "petAcoef",
        "petBcoef",
        "peqAcoef",
        "peqBcoef",
        "soilflx",
        "soilflx_pft",
        "rau",
        "soilcap",
        "soilcap_pft",
    )

    partial = enerbil_first_step_input_coverage(
        after_diffuco_payload=after_diffuco,
        driver_or_intersurf_payload=intersurf,
        source_kernel_outputs=audited_surftemp_inputs[:-1],
        diffuco_comb_executed=True,
        enerbil_begin_executed=True,
        enerbil_surftemp_executed=True,
    )
    assert "psnew" in partial.missing
    assert "qsol_sat_new" in partial.missing
    assert "temp_sol_new" in partial.missing

    complete = enerbil_first_step_input_coverage(
        after_diffuco_payload=after_diffuco,
        driver_or_intersurf_payload=intersurf,
        source_kernel_outputs=audited_surftemp_inputs,
        diffuco_comb_executed=True,
        enerbil_begin_executed=True,
        enerbil_surftemp_executed=True,
    )

    assert "psnew" in complete.covered_by_source_kernel
    assert "psnew_pft" in complete.covered_by_source_kernel
    assert "dtheta" in complete.covered_by_source_kernel
    assert "dtheta_pft" in complete.covered_by_source_kernel
    assert "qsol_sat_new" in complete.covered_by_source_kernel
    assert "qsol_sat_new_pft" in complete.covered_by_source_kernel
    assert "temp_sol_new" in complete.covered_by_source_kernel
    assert "temp_sol_new_pft" in complete.covered_by_source_kernel
    assert "qair_new" in complete.covered_by_source_kernel
    assert "epot_air_new" in complete.covered_by_source_kernel
    assert "psnew" not in complete.missing


def test_enerbil_first_step_coverage_does_not_default_valpha_without_diffuco_comb_precondition():
    after_diffuco = first_pft14_diffuco_after_main_payload()

    coverage = enerbil_first_step_input_coverage(
        after_diffuco_payload=after_diffuco,
        diffuco_comb_executed=False,
    )

    assert "vbeta2" in coverage.covered_by_trace
    assert "vbeta3" in coverage.covered_by_trace
    assert "vbeta4" in coverage.covered_by_trace
    assert "valpha" not in coverage.covered_by_source_kernel
    assert "valpha" in coverage.missing


def test_enerbil_first_step_local_restart_assembly_remains_non_parity_for_server_bridge():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    after_enerbil = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]
    albedo = read_driver_albedo_restart(REFERENCE_DRIVER_START)
    thermal = read_enerbil_soil_thermal_state_restart(REFERENCE_SECHIBA_START)

    def grid(value):
        return np.asarray([value], dtype=np.float64)

    def pft14(value, *, fill=0.0):
        data = np.full((1, 14), fill, dtype=np.float64)
        data[0, 13] = value
        return data

    swnet = driver_swnet_from_swdown_albedo(
        swdown=grid(intersurf["swdown"]),
        albedo=albedo.albedo,
    )
    energy = dim2_driver_non_watchout_energy_coupling_inputs(
        grid(intersurf["temp_air"]),
        grid(2.0),
        grid(intersurf["qair"]),
    )
    ok_laidev = np.zeros(14, dtype=bool)
    ok_laidev[13] = True

    step = enerbil_explicit_local_step(
        temp_sol=grid(after_diffuco["temp_sol"]),
        temp_sol_pft=pft14(after_diffuco["temp_sol_pft"], fill=after_diffuco["temp_sol"]),
        lwdown=grid(intersurf["lwdown"]),
        swnet=swnet,
        pb=grid(intersurf["pb"]),
        emis=condveg_initialized_emis(npts=1, impaze=False),
        ok_laidev=ok_laidev,
        epot_air=energy["epot_air"],
        petAcoef=energy["petAcoef"],
        petBcoef=energy["petBcoef"],
        qair=grid(intersurf["qair"]),
        peqAcoef=energy["peqAcoef"],
        peqBcoef=energy["peqBcoef"],
        soilflx=thermal.soilflx,
        soilflx_pft=pft14(thermal.soilflx_pft[0, 13], fill=thermal.soilflx[0]),
        rau=sechiba_air_density_from_pb_temp_air(grid(intersurf["pb"]), grid(intersurf["temp_air"])),
        u=grid(intersurf["u"]),
        v=grid(intersurf["v"]),
        q_cdrag=grid(after_diffuco["q_cdrag"]),
        q_cdrag_pft=pft14(after_diffuco["q_cdrag_pft"], fill=after_diffuco["q_cdrag"]),
        vbeta=grid(after_diffuco["vbeta"]),
        vbeta_pft=pft14(after_diffuco["vbeta_pft"]),
        valpha=grid(1.0),
        vbeta1=grid(after_diffuco["vbeta1"]),
        vbeta2=pft14(after_diffuco["vbeta2"]),
        vbeta3=pft14(after_diffuco["vbeta3"]),
        vbeta3pot=pft14(after_diffuco["vbeta3pot"]),
        vbeta4=grid(after_diffuco["vbeta4"]),
        vbeta4_pft=pft14(after_diffuco["vbeta4_pft"]),
        vbeta5=grid(after_diffuco["vbeta5"]),
        soilcap=thermal.soilcap,
        soilcap_pft=pft14(thermal.soilcap_pft[0, 13], fill=thermal.soilcap[0]),
        veget_max=pft14(after_diffuco["veget_max"]),
        dt_sechiba=1800.0,
    )

    np.testing.assert_allclose(np.asarray(step.flux.qsurf), [after_enerbil["qsurf"]])
    np.testing.assert_allclose(np.asarray(step.evapveg_pft.transpir)[0, 13], after_enerbil["transpir"])
    np.testing.assert_allclose(np.asarray(step.evapveg_pft.transpot)[0, 13], after_enerbil["transpot"])
    np.testing.assert_allclose(np.asarray(step.evapveg_pft.vevapnu_pft)[0, 13], after_enerbil["vevapnu_pft"])

    assert np.asarray(step.surftemp.temp_sol_new)[0] == pytest.approx(284.40084613976836)
    assert np.asarray(step.surftemp.temp_sol_new_pft)[0, 13] == pytest.approx(284.40084613976836)
    assert np.asarray(step.flux.evapot)[0] == pytest.approx(0.0)
    assert np.asarray(step.surftemp.temp_sol_new)[0] != pytest.approx(after_enerbil["temp_sol_new"])
    assert np.asarray(step.flux.evapot)[0] != pytest.approx(after_enerbil["evapot"])
    assert thermal.soilcap_pft[0, 13] != pytest.approx(after_enerbil["soilcap_pft"])


def test_enerbil_first_step_precall_builder_materializes_exact_sources_and_keeps_diffuco_gaps():
    after_diffuco = first_pft14_diffuco_after_main_payload()
    intersurf = read_server_records("intersurf_main", tags="main", limit=1)[0]
    albedo = read_driver_albedo_restart(REFERENCE_DRIVER_START)
    thermal = read_enerbil_soil_thermal_state_restart(REFERENCE_SECHIBA_START)

    assembly = assemble_enerbil_first_step_precall_payload(
        driver_or_intersurf_payload={
            "zlev": np.asarray([2.0], dtype=np.float64),
            "lwdown": np.asarray([intersurf["lwdown"]], dtype=np.float64),
            "swdown": np.asarray([intersurf["swdown"]], dtype=np.float64),
            "temp_air": np.asarray([intersurf["temp_air"]], dtype=np.float64),
            "u": np.asarray([intersurf["u"]], dtype=np.float64),
            "v": np.asarray([intersurf["v"]], dtype=np.float64),
            "qair": np.asarray([intersurf["qair"]], dtype=np.float64),
            "pb": np.asarray([intersurf["pb"]], dtype=np.float64),
            "precip_rain": np.asarray([intersurf["precip_rain"]], dtype=np.float64),
        },
        driver_albedo_payload=albedo.as_coverage_payload(),
        soil_thermal_restart_payload=thermal.as_coverage_payload(),
        after_diffuco_payload={
            "temp_sol": np.asarray([after_diffuco["temp_sol"]], dtype=np.float64),
            "temp_sol_pft": np.full((1, 14), after_diffuco["temp_sol"], dtype=np.float64),
            "q_cdrag": np.asarray([after_diffuco["q_cdrag"]], dtype=np.float64),
            "q_cdrag_pft": np.full((1, 14), after_diffuco["q_cdrag"], dtype=np.float64),
            "vbeta": np.asarray([after_diffuco["vbeta"]], dtype=np.float64),
            "vbeta_pft": np.zeros((1, 14), dtype=np.float64),
            "vbeta1": np.asarray([after_diffuco["vbeta1"]], dtype=np.float64),
            "vbeta2": np.zeros((1, 14), dtype=np.float64),
            "vbeta3": np.zeros((1, 14), dtype=np.float64),
            "vbeta3pot": np.zeros((1, 14), dtype=np.float64),
            "vbeta4": np.asarray([after_diffuco["vbeta4"]], dtype=np.float64),
            "vbeta4_pft": np.zeros((1, 14), dtype=np.float64),
            "vbeta5": np.asarray([after_diffuco["vbeta5"]], dtype=np.float64),
            "veget_max": np.zeros((1, 14), dtype=np.float64),
            "humrel": np.zeros((1, 14), dtype=np.float64),
            "qsurf": np.asarray([after_diffuco["qsurf"]], dtype=np.float64),
            "evapot": np.asarray([after_diffuco["evapot"]], dtype=np.float64),
            "evapot_corr": np.asarray([after_diffuco["evapot_corr"]], dtype=np.float64),
        },
        condveg_impaze=False,
    )

    assert not assembly.ok
    assert "swnet" in assembly.source_kernel_inputs
    assert "rau" in assembly.source_kernel_inputs
    assert "epot_air" in assembly.source_kernel_inputs
    assert "emis" in assembly.source_kernel_inputs
    assert set(("soilcap", "soilcap_pft", "soilflx", "soilflx_pft")) <= set(assembly.restart_inputs)
    assert set(("q_cdrag", "vbeta", "temp_sol", "qsurf", "evapot_corr")) <= set(assembly.diffuco_inputs)
    assert "swnet" not in assembly.missing_inputs
    assert "q_cdrag" not in assembly.missing_inputs
    assert "temp_sol" not in assembly.missing_inputs
    assert "psold" in assembly.missing_inputs
    assert "temp_sol_new" in assembly.missing_inputs

    expected_swnet = driver_swnet_from_swdown_albedo(
        swdown=np.asarray([intersurf["swdown"]], dtype=np.float64),
        albedo=albedo.albedo,
    )
    np.testing.assert_allclose(assembly.payload["swnet"], np.asarray(expected_swnet))
    np.testing.assert_allclose(
        assembly.payload["rau"],
        np.asarray(
            sechiba_air_density_from_pb_temp_air(
                np.asarray([intersurf["pb"]], dtype=np.float64),
                np.asarray([intersurf["temp_air"]], dtype=np.float64),
            )
        ),
    )


def test_enerbil_first_step_local_run_closes_begin_and_surftemp_from_precall_payload():
    values = parse_run_def(USED_RUN_DEF)
    static_truth = read_static_fields_from_fixed_format_trace_dir(FULL_TRACE_DIR)
    scaffold = paper_1961_driver_timestep_scaffold(
        ROOT / "configs" / "orchidee_man_250919.yaml",
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    assert scaffold.diffuco_first_step_local_enerbil_precall is not None
    assert scaffold.enerbil_first_step_precall is not None
    assert scaffold.enerbil_first_step_precall.missing_inputs == (
        "psold",
        "psold_pft",
        "qsol_sat",
        "qsol_sat_pft",
        "pdqsold",
        "pdqsold_pft",
        "lwabs",
        "netrad",
        "netrad_pft",
        "dtheta",
        "dtheta_pft",
        "psnew",
        "psnew_pft",
        "qsol_sat_new",
        "qsol_sat_new_pft",
        "temp_sol_new",
        "temp_sol_new_pft",
        "qair_new",
        "epot_air_new",
    )

    nvm = np.asarray(scaffold.enerbil_first_step_precall.payload["veget_max"]).shape[1]
    ok_laidev = np.asarray(
        [parse_run_def_indexed_bool(values, "OK_LAIDEV", index) for index in range(1, nvm + 1)],
        dtype=bool,
    )
    local = run_enerbil_first_step_local_from_precall(
        scaffold.enerbil_first_step_precall,
        ok_laidev=ok_laidev,
        dt_sechiba=parse_run_def_float(values, "DT_SECHIBA"),
        ok_explicitsnow=parse_run_def_bool(values["OK_EXPLICITSNOW"]),
        min_wind=parse_run_def_float(values, "MIN_WIND"),
    )

    for name in scaffold.enerbil_first_step_precall.missing_inputs:
        assert name in local.payload
    assert "qsurf" in local.payload
    assert "evapot_corr" in local.payload
    np.testing.assert_allclose(np.asarray(local.payload["psnew"]), np.asarray(local.step.surftemp.psnew))
    np.testing.assert_allclose(np.asarray(local.payload["temp_sol_new"]), np.asarray(local.step.surftemp.temp_sol_new))
    assert np.asarray(local.payload["temp_sol_new"]).shape == (1,)
    assert np.asarray(local.payload["temp_sol_new_pft"]).shape == (1, 14)
    assert "enerbil.f90::enerbil_main lines 485-545" in " ".join(local.provenance)


def test_enerbil_server_bridge_assembly_reports_missing_trace_inputs_without_restart_mixing():
    after_diffuco = first_pft14_diffuco_after_main_payload(root=BRIDGE_TRACE_ROOT)
    after_enerbil = read_enerbil_after_main_payload(
        kjit=1,
        ji=1,
        jv=14,
        root=BRIDGE_TRACE_ROOT,
    )
    intersurf = read_server_records(
        "intersurf_main",
        tags="main",
        limit=1,
        root=BRIDGE_TRACE_ROOT,
    )[0]
    thermal = read_enerbil_soil_thermal_state_restart(REFERENCE_SECHIBA_START)

    assembly = assemble_enerbil_explicit_local_step_from_server_bridge(
        after_diffuco_payload=after_diffuco,
        driver_or_intersurf_payload=intersurf,
        after_enerbil_payload=after_enerbil,
    )

    assert not assembly.ok
    assert assembly.step is None
    assert set(assembly.missing_inputs) == {"swnet", "soilflx", "soilflx_pft"}
    assert "soilcap" not in assembly.missing_inputs
    assert "soilcap_pft" not in assembly.missing_inputs
    assert "soilcap" in assembly.trace_input_diagnostics
    assert "soilcap_pft" in assembly.trace_input_diagnostics
    assert assembly.after_boundary_input_diagnostics == ("soilcap", "soilcap_pft")
    assert "soilflx" not in assembly.inputs
    assert "soilflx_pft" not in assembly.inputs
    assert "swnet" not in assembly.inputs
    assert assembly.inputs["soilcap_pft"][0, 13] == pytest.approx(after_enerbil["soilcap_pft"])
    assert assembly.inputs["soilcap_pft"][0, 13] != pytest.approx(thermal.soilcap_pft[0, 13])
    assert any("enerbil_main lines 433-438" in item for item in assembly.provenance)
    assert "THERMOSOIL source-kernel closure is not claimed" in " ".join(assembly.notes)
    assert "reference/case_001_071" in " ".join(assembly.notes)


def test_active_enerbil_server_bridge_assembly_matches_fortran_active_trace_outputs():
    before_enerbil = read_enerbil_active_payload(
        tag=ENERBIL_ACTIVE_BEFORE_TAG,
        kjit=49,
        ji=1,
        jv=14,
    )
    after_enerbil = read_enerbil_active_payload(
        tag=ENERBIL_ACTIVE_AFTER_TAG,
        kjit=49,
        ji=1,
        jv=14,
    )
    before_pottemp = read_enerbil_pottemp_active_payload(
        tag=ENERBIL_POTTEMP_BEFORE_TAG,
        kjit=49,
        ji=1,
    )
    after_pottemp = read_enerbil_pottemp_active_payload(
        tag=ENERBIL_POTTEMP_AFTER_TAG,
        kjit=49,
        ji=1,
    )

    assembly = assemble_enerbil_explicit_local_step_from_server_bridge(
        after_diffuco_payload=before_enerbil,
        driver_or_intersurf_payload=before_enerbil,
        after_enerbil_payload=after_enerbil,
        soil_thermal_trace_input_payload=before_enerbil,
        swnet_trace_input_payload=before_enerbil,
        pottemp_trace_input_payload=before_pottemp,
    )

    assert assembly.ok
    assert assembly.missing_inputs == ()
    assert "swnet" in assembly.trace_input_diagnostics
    assert "soilflx" in assembly.trace_input_diagnostics
    assert "soilflx_pft" in assembly.trace_input_diagnostics
    assert "q_sol_pot" in assembly.trace_input_diagnostics
    assert "temp_sol_pot" in assembly.trace_input_diagnostics
    assert "rau" in assembly.trace_input_diagnostics
    assert "petAcoef" in assembly.trace_input_diagnostics
    assert "valpha" in assembly.trace_input_diagnostics
    assert "rau" not in assembly.source_kernel_inputs
    assert "petAcoef" not in assembly.source_kernel_inputs
    assert "valpha" not in assembly.source_kernel_inputs

    step = assembly.step
    assert step is not None
    np.testing.assert_allclose(np.asarray(step.surftemp.temp_sol_new)[0], after_enerbil["temp_sol_new"])
    np.testing.assert_allclose(
        np.asarray(step.surftemp.temp_sol_new_pft)[0, 13],
        after_enerbil["temp_sol_new_pft"],
    )
    np.testing.assert_allclose(np.asarray(step.flux.qsurf)[0], after_enerbil["qsurf"])
    np.testing.assert_allclose(np.asarray(step.flux.fluxsens)[0], after_enerbil["fluxsens"])
    np.testing.assert_allclose(np.asarray(step.flux.fluxlat)[0], after_enerbil["fluxlat"])
    np.testing.assert_allclose(np.asarray(step.flux.evapot)[0], after_enerbil["evapot"])
    np.testing.assert_allclose(np.asarray(step.evapot_corr.evapot_corr)[0], after_enerbil["evapot_corr"])
    np.testing.assert_allclose(np.asarray(step.evapveg_grid.vevapnu)[0], after_enerbil["vevapnu"])
    np.testing.assert_allclose(np.asarray(step.evapveg_grid.vevapsno)[0], after_enerbil["vevapsno"])
    np.testing.assert_allclose(np.asarray(step.evapveg_grid.vevapflo)[0], after_enerbil["vevapflo"])
    np.testing.assert_allclose(np.asarray(step.evapveg_pft.transpir)[0, 13], after_enerbil["transpir"])
    np.testing.assert_allclose(np.asarray(step.evapveg_pft.transpot)[0, 13], after_enerbil["transpot"])
    np.testing.assert_allclose(np.asarray(step.evapveg_pft.vevapnu_pft)[0, 13], after_enerbil["vevapnu_pft"])
    np.testing.assert_allclose(np.asarray(step.evapveg_pft.vevapwet)[0, 13], after_enerbil["vevapwet"])

    assert step.explicit_snow is not None
    np.testing.assert_allclose(np.asarray(step.explicit_snow.pgflux)[0], after_enerbil["pgflux"])
    np.testing.assert_allclose(
        np.asarray(step.explicit_snow.temp_sol_add)[0],
        after_enerbil["temp_sol_add"],
    )
    assert step.t2mdiag is not None
    np.testing.assert_allclose(np.asarray(step.t2mdiag)[0], after_enerbil["t2mdiag"])
    assert step.pottemp is not None
    np.testing.assert_allclose(np.asarray(step.pottemp.q_sol_pot)[0], after_pottemp["q_sol_pot"])
    np.testing.assert_allclose(np.asarray(step.pottemp.temp_sol_pot)[0], after_pottemp["temp_sol_pot"])
    assert after_enerbil["evapot_corr"] != pytest.approx(after_enerbil["evapot"])


def test_enerbil_bridge_trace_is_cold_start_not_reference_restart_case():
    reference_run = REFERENCE_RUN_DEF.read_text(encoding="utf-8")
    bridge_run = BRIDGE_RUN_DEF.read_text(encoding="utf-8")
    thermal = read_enerbil_soil_thermal_state_restart(REFERENCE_SECHIBA_START)
    after_enerbil = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)

    assert "SECHIBA_restart_in=sechiba_start.nc" in reference_run
    assert "SECHIBA_restart_in=NONE" in bridge_run
    assert "RESTART_FILEIN=driver_start.nc" in reference_run
    assert "RESTART_FILEIN=NONE" in bridge_run
    assert thermal.soilcap_pft[0, 13] == pytest.approx(45865.2880547002)
    assert after_enerbil["soilcap_pft"] == pytest.approx(59764.3072135468)
    assert after_enerbil["soilcap_pft"] != pytest.approx(thermal.soilcap_pft[0, 13])
