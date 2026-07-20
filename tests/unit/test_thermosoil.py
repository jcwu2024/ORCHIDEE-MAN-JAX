from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    CAPA_ICE,
    COND_DRY_ORG,
    COND_SOLID_ORG,
    FR_DT,
    LHF,
    MILLE,
    PHIGEOTH,
    PSNOWDZMIN,
    POROS,
    QZ_USDA,
    RHO_ICE,
    RHO_WATER,
    SN_CAPA,
    SN_COND,
    SN_DENS,
    SMCMAX_USDA,
    SO_CAPA_DRY,
    SO_CAPA_DRY_NS_USDA,
    SO_CAPA_DRY_ORG,
    SO_COND_DRY,
    THKICE,
    THKQTZ,
    THKW,
    WATER_CAPA,
    XCI,
    XP00,
    ZERO_CELSIUS,
    ZSNOWTHRMCOND1,
    ZSNOWTHRMCOND2,
    ZSNOWTHRMCOND_AVAP,
    ZSNOWTHRMCOND_BVAP,
    ZSNOWTHRMCOND_CVAP,
    read_thermosoil_restart_state,
    thermosoil_coef_soil_coverage,
    thermosoil_coef_no_explicit_snow,
    thermosoil_coef_soil_no_snow,
    thermosoil_coef_explicit_snow,
    thermosoil_cond,
    thermosoil_cond_pft,
    thermosoil_cold_start_coef_closure,
    thermosoil_diaglev_from_ptn,
    thermosoil_energy_diagnostics,
    thermosoil_explicit_step,
    thermosoil_final_state,
    thermosoil_getdiff_explicit,
    thermosoil_getdiff_old_thermix_with_snow,
    thermosoil_humlev,
    thermosoil_initial_ptn_constant,
    thermosoil_no_explicit_snow_step,
    thermosoil_profile_explicit_snow,
    thermosoil_profile_no_explicit_snow,
    thermosoil_soilc_tempdiff_fraction,
    thermosoil_toporganiclayer_fraction,
    thermosoil_veget_max_bg,
)

REFERENCE_RUN = (
    ROOT
    / "reference"
    / "case_001_071"
    / "OUT"
    / "orc_calibrate_250919_sen"
    / "arg2_1.0"
    / "001.0-071.0"
    / "I10"
    / "S2_63.206_0.0876_0.2019_50.658"
)


def test_thermosoil_initial_ptn_constant_matches_cold_start_setvar_fallback():
    ptn = thermosoil_initial_ptn_constant(kjpindex=2, ngrnd=4, nvm=3)

    assert np.asarray(ptn).shape == (2, 4, 3)
    np.testing.assert_allclose(np.asarray(ptn), 280.0)
    custom = thermosoil_initial_ptn_constant(kjpindex=1, ngrnd=2, nvm=1, thermosoil_tpro=276.5)
    np.testing.assert_allclose(np.asarray(custom), 276.5)
    with pytest.raises(ValueError, match="positive"):
        thermosoil_initial_ptn_constant(kjpindex=0, ngrnd=1, nvm=1)


class _Moisture:
    def __init__(self):
        self.shumdiag_perma = np.asarray([[0.3, 0.3]], dtype=np.float64)
        self.mc_layh = np.asarray([[0.3, 0.3]], dtype=np.float64)
        self.mcl_layh = np.asarray([[0.3, 0.3]], dtype=np.float64)
        self.tmc_layh = np.asarray([[0.3, 0.3]], dtype=np.float64)
        self.mc_layh_pft = np.asarray([[[0.3], [0.3]]], dtype=np.float64)
        self.mcl_layh_pft = np.asarray([[[0.3], [0.3]]], dtype=np.float64)
        self.tmc_layh_pft = np.asarray([[[0.3], [0.3]]], dtype=np.float64)


def test_thermosoil_cold_start_coef_closure_requires_refsoc_on_paper_path():
    ptn = thermosoil_initial_ptn_constant(kjpindex=1, ngrnd=3, nvm=1)
    result = thermosoil_cold_start_coef_closure(
        ptn=ptn,
        moisture=_Moisture(),
        temp_sol_new=np.asarray([280.0], dtype=np.float64),
        temp_sol_new_pft=np.asarray([[280.0]], dtype=np.float64),
        snowdz=np.zeros((1, 3), dtype=np.float64),
        snowrho=np.full((1, 3), 50.0, dtype=np.float64),
        snowtemp=np.full((1, 3), 273.15, dtype=np.float64),
        njsc=np.asarray([2], dtype=np.int32),
        veget_max=np.asarray([[1.0]], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        dlt=np.asarray([0.1, 0.2, 0.4], dtype=np.float64),
        dz1=np.asarray([10.0, 5.0], dtype=np.float64),
        zlt=np.asarray([0.1, 0.3, 0.7], dtype=np.float64),
        znt=np.asarray([0.05, 0.15, 0.35], dtype=np.float64),
        dz5=np.asarray([0.5, 0.5], dtype=np.float64),
        dt_sechiba=1800.0,
        frac_snow_veg=np.zeros((1,), dtype=np.float64),
        frac_snow_nobio=np.zeros((1, 1), dtype=np.float64),
        totfrac_nobio=np.zeros((1,), dtype=np.float64),
        use_refSOC=True,
        use_soilc_tempdiff=True,
    )

    assert not result.ok
    assert result.coef is None
    assert "refSOC_non_restart_initialization" in result.missing_inputs


def test_thermosoil_cold_start_coef_closure_runs_when_organic_fraction_is_explicit():
    ptn = thermosoil_initial_ptn_constant(kjpindex=1, ngrnd=3, nvm=1)
    result = thermosoil_cold_start_coef_closure(
        ptn=ptn,
        moisture=_Moisture(),
        temp_sol_new=np.asarray([280.0], dtype=np.float64),
        temp_sol_new_pft=np.asarray([[280.0]], dtype=np.float64),
        snowdz=np.zeros((1, 3), dtype=np.float64),
        snowrho=np.full((1, 3), 50.0, dtype=np.float64),
        snowtemp=np.full((1, 3), 273.15, dtype=np.float64),
        njsc=np.asarray([2], dtype=np.int32),
        veget_max=np.asarray([[1.0]], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        dlt=np.asarray([0.1, 0.2, 0.4], dtype=np.float64),
        dz1=np.asarray([10.0, 5.0], dtype=np.float64),
        zlt=np.asarray([0.1, 0.3, 0.7], dtype=np.float64),
        znt=np.asarray([0.05, 0.15, 0.35], dtype=np.float64),
        dz5=np.asarray([0.5, 0.5], dtype=np.float64),
        dt_sechiba=1800.0,
        frac_snow_veg=np.zeros((1,), dtype=np.float64),
        frac_snow_nobio=np.zeros((1, 1), dtype=np.float64),
        totfrac_nobio=np.zeros((1,), dtype=np.float64),
        zx1=np.zeros((1, 3, 1), dtype=np.float64),
        use_refSOC=False,
        use_soilc_tempdiff=True,
    )

    assert result.ok
    assert result.coef is not None
    assert np.asarray(result.coef.soil.cgrnd).shape == (1, 2, 1)
    assert np.asarray(result.coef.lambda_snow).shape == (1,)
    assert np.asarray(result.coef.soilcap).shape == (1,)
    assert np.asarray(result.stempdiag).shape == (1, 2)
    assert any("thermosoil_initialize lines 719-735" in item for item in result.provenance)


def _expected_soil_coef(
    *,
    ptn,
    pcapa,
    pkappa,
    temp_sol_new,
    temp_sol_new_pft,
    veget_max,
    ok_laidev,
    dlt,
    dz1,
    dt_sechiba,
    lambda_thermal,
    veget_mask_2d,
    phigeoth=PHIGEOTH,
):
    npts, ngrnd, nvm = ptn.shape
    zdz2 = np.zeros_like(ptn)
    zdz1 = np.zeros((npts, ngrnd - 1, nvm), dtype=np.float64)
    cgrnd = np.zeros((npts, ngrnd - 1, nvm), dtype=np.float64)
    dgrnd = np.zeros((npts, ngrnd - 1, nvm), dtype=np.float64)
    soilcap_pft = np.zeros((npts, nvm), dtype=np.float64)
    soilflx_pft = np.zeros((npts, nvm), dtype=np.float64)
    soilcap_pft_nosnow = np.zeros((npts, nvm), dtype=np.float64)
    soilflx_pft_nosnow = np.zeros((npts, nvm), dtype=np.float64)
    soilcap = np.zeros(npts, dtype=np.float64)
    soilflx = np.zeros(npts, dtype=np.float64)
    cgrnd_soil = np.zeros(npts, dtype=np.float64)
    dgrnd_soil = np.zeros(npts, dtype=np.float64)
    zdz1_soil = np.zeros(npts, dtype=np.float64)
    zdz2_soil = np.zeros(npts, dtype=np.float64)

    for jv in range(nvm):
        for jg in range(ngrnd):
            active = veget_mask_2d[:, jv]
            zdz2[active, jg, jv] = pcapa[active, jg, jv] * dlt[jg] / dt_sechiba
        for jg in range(ngrnd - 1):
            active = veget_mask_2d[:, jv]
            zdz1[active, jg, jv] = dz1[jg] * pkappa[active, jg, jv]

        active = veget_mask_2d[:, jv]
        z1 = zdz2[:, ngrnd - 1, jv] + zdz1[:, ngrnd - 2, jv]
        cgrnd[active, ngrnd - 2, jv] = (
            phigeoth + zdz2[active, ngrnd - 1, jv] * ptn[active, ngrnd - 1, jv]
        ) / z1[active]
        dgrnd[active, ngrnd - 2, jv] = zdz1[active, ngrnd - 2, jv] / z1[active]

        for jg in range(ngrnd - 2, 0, -1):
            denom = (
                zdz2[:, jg, jv]
                + zdz1[:, jg - 1, jv]
                + zdz1[:, jg, jv] * (1.0 - dgrnd[:, jg, jv])
            )
            z1 = np.zeros_like(denom)
            z1[active] = 1.0 / denom[active]
            cgrnd[active, jg - 1, jv] = (
                ptn[active, jg, jv] * zdz2[active, jg, jv]
                + zdz1[active, jg, jv] * cgrnd[active, jg, jv]
            ) * z1[active]
            dgrnd[active, jg - 1, jv] = zdz1[active, jg - 1, jv] * z1[active]

        base = zdz1[:, 0, jv] * (cgrnd[:, 0, jv] + (dgrnd[:, 0, jv] - 1.0) * ptn[:, 0, jv])
        cap = dt_sechiba * (zdz2[:, 0, jv] + (1.0 - dgrnd[:, 0, jv]) * zdz1[:, 0, jv])
        z1 = lambda_thermal * (1.0 - dgrnd[:, 0, jv]) + 1.0
        soilcap_pft[active, jv] = cap[active] / z1[active]
        pft_temp = temp_sol_new_pft[:, jv] if ok_laidev[jv] else temp_sol_new
        soilflx_pft[active, jv] = base[active] + soilcap_pft[active, jv] * (
            ptn[active, 0, jv] * z1[active] - lambda_thermal * cgrnd[active, 0, jv] - pft_temp[active]
        ) / dt_sechiba

        soilcap_pft_nosnow[active, jv] = cap[active] / z1[active]
        soilflx_pft_nosnow[active, jv] = base[active] + soilcap_pft_nosnow[active, jv] * (
            ptn[active, 0, jv] * z1[active]
            - lambda_thermal * cgrnd[active, 0, jv]
            - temp_sol_new[active]
        ) / dt_sechiba

    for ji in range(npts):
        for jv in range(nvm):
            if veget_mask_2d[ji, jv]:
                soilflx[ji] += soilflx_pft_nosnow[ji, jv] * veget_max[ji, jv]
                soilcap[ji] += soilcap_pft_nosnow[ji, jv] * veget_max[ji, jv]
                cgrnd_soil[ji] += cgrnd[ji, 0, jv] * veget_max[ji, jv]
                dgrnd_soil[ji] += dgrnd[ji, 0, jv] * veget_max[ji, jv]
                zdz1_soil[ji] += zdz1[ji, 0, jv] * veget_max[ji, jv]
                zdz2_soil[ji] += zdz2[ji, 0, jv] * veget_max[ji, jv]

    return {
        "zdz1": zdz1,
        "zdz2": zdz2,
        "cgrnd": cgrnd,
        "dgrnd": dgrnd,
        "soilcap": soilcap,
        "soilcap_pft": soilcap_pft,
        "soilflx": soilflx,
        "soilflx_pft": soilflx_pft,
        "soilcap_pft_nosnow": soilcap_pft_nosnow,
        "soilflx_pft_nosnow": soilflx_pft_nosnow,
        "cgrnd_soil": cgrnd_soil,
        "dgrnd_soil": dgrnd_soil,
        "zdz1_soil": zdz1_soil,
        "zdz2_soil": zdz2_soil,
    }


def test_read_thermosoil_restart_state_normalizes_recurrence_axes():
    state = read_thermosoil_restart_state(REFERENCE_RUN / "sechiba_start.nc")

    assert state.path.name == "sechiba_start.nc"
    assert state.ptn.shape == (1, 32, 14)
    assert state.cgrnd.shape == (1, 31, 14)
    assert state.dgrnd.shape == (1, 31, 14)
    assert state.cgrnd_snow.shape == (1, 3)
    assert state.dgrnd_snow.shape == (1, 3)
    assert state.lambda_snow.shape == (1,)
    assert state.gtemp.shape == (1,)
    assert np.isfinite(state.ptn).all()
    assert np.isfinite(state.cgrnd).all()
    assert np.isfinite(state.dgrnd).all()
    assert "ptn" in state.as_profile_payload()
    assert any("thermosoil_initialize lines 668-684" in item for item in state.provenance)


def test_thermosoil_coef_soil_no_snow_matches_fortran_loops():
    ptn = np.array(
        [
            [[281.0, 282.0, 283.0], [282.0, 283.5, 284.0], [284.0, 285.0, 286.5], [286.0, 287.5, 288.0]],
            [[290.0, 291.0, 292.0], [289.0, 290.5, 291.0], [288.0, 289.5, 290.0], [287.0, 288.5, 289.0]],
        ],
        dtype=np.float64,
    )
    pcapa = np.array(
        [
            [[1.9e6, 2.0e6, 2.1e6], [2.0e6, 2.1e6, 2.2e6], [2.2e6, 2.3e6, 2.4e6], [2.4e6, 2.5e6, 2.6e6]],
            [[1.8e6, 1.9e6, 2.0e6], [1.9e6, 2.0e6, 2.1e6], [2.1e6, 2.2e6, 2.3e6], [2.3e6, 2.4e6, 2.5e6]],
        ],
        dtype=np.float64,
    )
    pkappa = np.array(
        [
            [[0.8, 0.9, 1.0], [0.9, 1.0, 1.1], [1.0, 1.1, 1.2], [1.1, 1.2, 1.3]],
            [[0.7, 0.8, 0.9], [0.85, 0.95, 1.05], [0.95, 1.05, 1.15], [1.05, 1.15, 1.25]],
        ],
        dtype=np.float64,
    )
    temp_sol_new = np.array([280.2, 291.3], dtype=np.float64)
    temp_sol_new_pft = np.array([[280.1, 280.4, 280.7], [291.0, 291.5, 291.8]], dtype=np.float64)
    veget_max = np.array([[0.25, 0.55, 0.20], [0.30, 0.00, 0.40]], dtype=np.float64)
    ok_laidev = np.array([False, True, False])
    dlt = np.array([0.1, 0.3, 0.6, 1.2], dtype=np.float64)
    dz1 = np.array([10.0, 2.5, 1.0], dtype=np.float64)
    veget_mask_2d = np.array([[True, True, True], [True, False, True]])
    dt_sechiba = 1800.0
    lambda_thermal = 0.5

    result = thermosoil_coef_soil_no_snow(
        ptn=ptn,
        pcapa=pcapa,
        pkappa=pkappa,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        dlt=dlt,
        dz1=dz1,
        dt_sechiba=dt_sechiba,
        lambda_thermal=lambda_thermal,
        veget_mask_2d=veget_mask_2d,
    )
    expected = _expected_soil_coef(
        ptn=ptn,
        pcapa=pcapa,
        pkappa=pkappa,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        dlt=dlt,
        dz1=dz1,
        dt_sechiba=dt_sechiba,
        lambda_thermal=lambda_thermal,
        veget_mask_2d=veget_mask_2d,
    )

    for field, expected_value in expected.items():
        np.testing.assert_allclose(np.asarray(getattr(result, field)), expected_value, rtol=1e-13, atol=1e-10)


def test_thermosoil_coef_pft_temperature_branch_only_changes_active_laidev_flux():
    ptn = np.full((1, 3, 2), 285.0, dtype=np.float64)
    pcapa = np.full((1, 3, 2), 2.0e6, dtype=np.float64)
    pkappa = np.full((1, 3, 2), 1.0, dtype=np.float64)
    veget_max = np.array([[0.5, 0.5]], dtype=np.float64)
    temp_sol_new = np.array([280.0], dtype=np.float64)
    temp_sol_new_pft = np.array([[280.0, 282.0]], dtype=np.float64)

    result = thermosoil_coef_soil_no_snow(
        ptn=ptn,
        pcapa=pcapa,
        pkappa=pkappa,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=np.array([False, True]),
        dlt=np.array([0.1, 0.2, 0.4]),
        dz1=np.array([10.0, 3.0]),
        dt_sechiba=1800.0,
        lambda_thermal=0.5,
    )

    assert np.asarray(result.soilflx_pft)[0, 0] == pytest.approx(np.asarray(result.soilflx_pft_nosnow)[0, 0])
    assert np.asarray(result.soilflx_pft)[0, 1] != pytest.approx(np.asarray(result.soilflx_pft_nosnow)[0, 1])
    expected_grid = np.sum(np.asarray(result.soilflx_pft_nosnow)[0] * veget_max[0])
    assert np.asarray(result.soilflx)[0] == pytest.approx(expected_grid)


def test_thermosoil_coef_explicit_snow_matches_snow_layer_recurrence_and_blend():
    ptn = np.array(
        [
            [[280.0, 281.0], [282.0, 283.0], [284.0, 285.0]],
            [[286.0, 287.0], [288.0, 289.0], [290.0, 291.0]],
        ],
        dtype=np.float64,
    )
    pcapa = np.full((2, 3, 2), 2.0e6, dtype=np.float64)
    pkappa = np.array(
        [
            [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4]],
        ],
        dtype=np.float64,
    )
    temp_sol_new = np.array([279.5, 285.5], dtype=np.float64)
    temp_sol_new_pft = np.array([[279.0, 280.0], [285.0, 286.0]], dtype=np.float64)
    veget_max = np.array([[0.6, 0.3], [0.5, 0.4]], dtype=np.float64)
    veget_bg = veget_max.copy()
    veget_bg[:, 0] = np.maximum(1.0 - veget_max[:, 1:].sum(axis=1), 0.0)
    ptn_pftmean = np.sum(ptn * veget_bg[:, None, :], axis=2)
    dlt = np.array([0.1, 0.2, 0.4], dtype=np.float64)
    dz1 = np.array([10.0, 3.0], dtype=np.float64)
    pcapa_snow = np.array([[2.0e5, 3.0e5, 4.0e5], [2.2e5, 3.2e5, 4.2e5]], dtype=np.float64)
    pkappa_snow = np.array([[0.20, 0.25, 0.30], [0.22, 0.27, 0.32]], dtype=np.float64)
    snowdz = np.array([[0.01, 0.02, 0.04], [0.015, 0.025, 0.045]], dtype=np.float64)
    snowtemp = np.array([[270.0, 268.0, 266.0], [269.0, 267.0, 265.0]], dtype=np.float64)
    frac_snow_veg = np.array([0.5, 0.2], dtype=np.float64)
    frac_snow_nobio = np.array([[0.1], [0.3]], dtype=np.float64)
    totfrac_nobio = np.array([0.2, 0.1], dtype=np.float64)
    dt = 1800.0
    zlt = np.array([0.1, 0.3, 0.7], dtype=np.float64)

    result = thermosoil_coef_explicit_snow(
        ptn=ptn,
        ptn_pftmean=ptn_pftmean,
        pcapa=pcapa,
        pkappa=pkappa,
        pcapa_snow=pcapa_snow,
        pkappa_snow=pkappa_snow,
        snowdz=snowdz,
        snowtemp=snowtemp,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=np.array([False, True]),
        dlt=dlt,
        dz1=dz1,
        zlt=zlt,
        dt_sechiba=dt,
        lambda_thermal=0.5,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        totfrac_nobio=totfrac_nobio,
    )

    soil = result.soil
    dz2_snow = np.maximum(snowdz, PSNOWDZMIN)
    dz1_snow = np.zeros_like(dz2_snow)
    dz1_snow[:, 0] = 2.0 / (dz2_snow[:, 1] + dz2_snow[:, 0])
    dz1_snow[:, 1] = 2.0 / (dz2_snow[:, 2] + dz2_snow[:, 1])
    lambda_snow = dz2_snow[:, 0] / 2.0 * dz1_snow[:, 0]
    zdz2_snow = pcapa_snow * dz2_snow / dt
    zdz1_snow = np.zeros_like(dz2_snow)
    zdz1_snow[:, 0] = dz1_snow[:, 0] * pkappa_snow[:, 0]
    zdz1_snow[:, 1] = dz1_snow[:, 1] * pkappa_snow[:, 1]
    zdz1_snow[:, 2] = pkappa_snow[:, 2] / (zlt[0] + dz2_snow[:, 2] / 2.0)

    cgrnd_snow = np.zeros_like(dz2_snow)
    dgrnd_snow = np.zeros_like(dz2_snow)
    z1_bottom = (
        np.asarray(soil.zdz2_soil)
        + (1.0 - np.asarray(soil.dgrnd_soil)) * np.asarray(soil.zdz1_soil)
        + zdz1_snow[:, 2]
    )
    cgrnd_snow[:, 2] = (
        np.asarray(soil.zdz2_soil) * ptn_pftmean[:, 0]
        + np.asarray(soil.zdz1_soil) * np.asarray(soil.cgrnd_soil)
    ) / z1_bottom
    dgrnd_snow[:, 2] = zdz1_snow[:, 2] / z1_bottom
    z1_next = zdz2_snow[:, 2] + (1.0 - dgrnd_snow[:, 2]) * zdz1_snow[:, 2] + zdz1_snow[:, 1]
    cgrnd_snow[:, 1] = (zdz2_snow[:, 2] * snowtemp[:, 2] + zdz1_snow[:, 2] * cgrnd_snow[:, 2]) / z1_next
    dgrnd_snow[:, 1] = zdz1_snow[:, 1] / z1_next
    z1_top = 1.0 / (zdz2_snow[:, 1] + zdz1_snow[:, 0] + zdz1_snow[:, 1] * (1.0 - dgrnd_snow[:, 1]))
    cgrnd_snow[:, 0] = (snowtemp[:, 1] * zdz2_snow[:, 1] + zdz1_snow[:, 1] * cgrnd_snow[:, 1]) * z1_top
    dgrnd_snow[:, 0] = zdz1_snow[:, 0] * z1_top

    snowflx = zdz1_snow[:, 0] * (cgrnd_snow[:, 0] + (dgrnd_snow[:, 0] - 1.0) * snowtemp[:, 0])
    snowcap = dt * (zdz2_snow[:, 0] + (1.0 - dgrnd_snow[:, 0]) * zdz1_snow[:, 0])
    z1_surface = lambda_snow * (1.0 - dgrnd_snow[:, 0]) + 1.0
    snowcap = snowcap / z1_surface
    snowflx = snowflx + snowcap * (
        snowtemp[:, 0] * z1_surface - lambda_snow * cgrnd_snow[:, 0] - temp_sol_new
    ) / dt
    snow_veg_weight = frac_snow_veg * (1.0 - totfrac_nobio)
    snow_nobio_weight = np.sum(frac_snow_nobio, axis=1) * totfrac_nobio
    nonsnow_weight = 1.0 - (snow_veg_weight + snow_nobio_weight)
    expected_soilcap = snowcap * snow_veg_weight + np.asarray(soil.soilcap) * snow_nobio_weight + np.asarray(soil.soilcap) * nonsnow_weight
    expected_soilflx = snowflx * snow_veg_weight + np.asarray(soil.soilflx) * snow_nobio_weight + np.asarray(soil.soilflx) * nonsnow_weight

    np.testing.assert_allclose(np.asarray(result.lambda_snow), lambda_snow)
    np.testing.assert_allclose(np.asarray(result.cgrnd_snow), cgrnd_snow, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.dgrnd_snow), dgrnd_snow, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.snowcap), snowcap, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.snowflx), snowflx, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.soilcap), expected_soilcap, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.soilflx), expected_soilflx, rtol=1e-12)


def test_thermosoil_coef_validates_shapes_and_reports_coverage():
    with pytest.raises(ValueError, match="ptn must have shape"):
        thermosoil_coef_soil_no_snow(
            ptn=np.ones((2, 3)),
            pcapa=np.ones((2, 3)),
            pkappa=np.ones((2, 3)),
            temp_sol_new=np.ones(2),
            temp_sol_new_pft=np.ones((2, 1)),
            veget_max=np.ones((2, 1)),
            ok_laidev=np.ones(1, dtype=bool),
            dlt=np.ones(3),
            dz1=np.ones(2),
            dt_sechiba=1800.0,
            lambda_thermal=0.5,
        )

    coverage = thermosoil_coef_soil_coverage()
    assert coverage.ok
    assert "soilcap" in coverage.covered_fields
    assert "pcapa" in coverage.covered_fields
    assert "snowflx" in coverage.covered_fields
    assert any("thermosoil_coef lines 1518-1628" in item for item in coverage.provenance)
    assert any("thermosoil_getdiff lines 2566-2863" in item for item in coverage.provenance)
    assert any("thermosoil_coef lines 1633-1721" in item for item in coverage.provenance)
    assert not any("explicit-snow coefficient" in item for item in coverage.missing_processes)


def test_thermosoil_veget_max_bg_and_diaglev_match_fortran_weighting():
    veget_max = np.array([[0.20, 0.30, 0.10], [0.05, 0.60, 0.10]], dtype=np.float64)
    ptn = np.array(
        [
            [[280.0, 281.0, 282.0], [283.0, 284.0, 285.0], [286.0, 287.0, 288.0]],
            [[290.0, 291.0, 292.0], [293.0, 294.0, 295.0], [296.0, 297.0, 298.0]],
        ],
        dtype=np.float64,
    )

    veget_bg = np.asarray(thermosoil_veget_max_bg(veget_max))
    expected_bg = veget_max.copy()
    expected_bg[:, 0] = np.maximum(1.0 - veget_max[:, 1:].sum(axis=1), 0.0)
    np.testing.assert_allclose(veget_bg, expected_bg)

    stempdiag = np.asarray(thermosoil_diaglev_from_ptn(ptn=ptn, veget_max=veget_max, nslm=2))
    expected = np.sum(ptn[:, :2, :] * expected_bg[:, None, :], axis=2)
    np.testing.assert_allclose(stempdiag, expected)


def test_thermosoil_final_state_matches_main_pft_mean_outputs():
    veget_max = np.array([[0.2, 0.3, 0.1], [0.1, 0.6, 0.1]], dtype=np.float64)
    ptn = np.array(
        [
            [[280.0, 281.0, 282.0], [283.0, 284.0, 285.0]],
            [[290.0, 291.0, 292.0], [293.0, 294.0, 295.0]],
        ],
        dtype=np.float64,
    )
    pkappa = ptn / 100.0
    shum = np.full_like(ptn, 0.5)

    result = thermosoil_final_state(
        ptn=ptn,
        pkappa=pkappa,
        veget_max=veget_max,
        shum_ngrnd_permalong=shum,
    )

    veget_bg = veget_max.copy()
    veget_bg[:, 0] = np.maximum(1.0 - veget_max[:, 1:].sum(axis=1), 0.0)
    expected_ptn = np.sum(ptn * veget_bg[:, None, :], axis=2)
    expected_pkappa = np.sum(pkappa * veget_bg[:, None, :], axis=2)
    np.testing.assert_allclose(np.asarray(result.ptn_pftmean), expected_ptn)
    np.testing.assert_allclose(np.asarray(result.pkappa_pftmean), expected_pkappa)
    np.testing.assert_allclose(np.asarray(result.gtemp), expected_ptn[:, 0])
    np.testing.assert_allclose(np.asarray(result.ptnlev1), expected_ptn[:, 0])
    np.testing.assert_allclose(np.asarray(result.deephum_prof), shum)
    np.testing.assert_allclose(np.asarray(result.deeptemp_prof), ptn)


def test_thermosoil_profile_no_explicit_snow_updates_layers_and_diaglev():
    ptn = np.array(
        [
            [[280.0, 281.0], [282.0, 283.0], [284.0, 285.0]],
            [[290.0, 291.0], [292.0, 293.0], [294.0, 295.0]],
        ],
        dtype=np.float64,
    )
    cgrnd = np.array(
        [
            [[270.0, 271.0], [272.0, 273.0]],
            [[280.0, 281.0], [282.0, 283.0]],
        ],
        dtype=np.float64,
    )
    dgrnd = np.array(
        [
            [[0.20, 0.25], [0.30, 0.35]],
            [[0.40, 0.45], [0.50, 0.55]],
        ],
        dtype=np.float64,
    )
    temp_sol_new = np.array([285.0, 286.0], dtype=np.float64)
    temp_sol_new_pft = np.array([[286.0, 287.0], [288.0, 289.0]], dtype=np.float64)
    veget_max = np.array([[0.70, 0.20], [0.50, 0.00]], dtype=np.float64)
    ok_laidev = np.array([False, True])
    mask = np.array([[True, True], [True, False]])
    lambda_thermal = 0.5

    result = thermosoil_profile_no_explicit_snow(
        ptn=ptn,
        cgrnd=cgrnd,
        dgrnd=dgrnd,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        lambda_thermal=lambda_thermal,
        veget_mask_2d=mask,
        nslm=2,
    )

    expected = ptn.copy()
    for jv in range(2):
        for ji in range(2):
            if not mask[ji, jv]:
                continue
            surface_temp = temp_sol_new_pft[ji, jv] if ok_laidev[jv] and veget_max[ji, jv] > 0.0 else temp_sol_new[ji]
            expected[ji, 0, jv] = (
                lambda_thermal * cgrnd[ji, 0, jv] + surface_temp
            ) / (lambda_thermal * (1.0 - dgrnd[ji, 0, jv]) + 1.0)
            expected[ji, 1, jv] = cgrnd[ji, 0, jv] + dgrnd[ji, 0, jv] * expected[ji, 0, jv]
            expected[ji, 2, jv] = cgrnd[ji, 1, jv] + dgrnd[ji, 1, jv] * expected[ji, 1, jv]
    veget_bg = veget_max.copy()
    veget_bg[:, 0] = np.maximum(1.0 - veget_max[:, 1:].sum(axis=1), 0.0)
    expected_stemp = np.sum(expected[:, :2, :] * veget_bg[:, None, :], axis=2)

    np.testing.assert_allclose(np.asarray(result.ptn), expected)
    np.testing.assert_allclose(np.asarray(result.stempdiag), expected_stemp)


def test_thermosoil_profile_explicit_snow_uses_snow_surface_coefficients():
    ptn = np.array([[[280.0, 281.0], [282.0, 283.0], [284.0, 285.0]]], dtype=np.float64)
    cgrnd = np.array([[[270.0, 271.0], [272.0, 273.0]]], dtype=np.float64)
    dgrnd = np.array([[[0.2, 0.3], [0.4, 0.5]]], dtype=np.float64)
    cgrnd_snow = np.array([[260.0, 261.0, 262.0]], dtype=np.float64)
    dgrnd_snow = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    snowtemp = np.array([[269.0, 268.0, 267.0]], dtype=np.float64)
    temp_sol_new = np.array([280.0], dtype=np.float64)
    veget_max = np.array([[0.6, 0.2]], dtype=np.float64)
    frac_snow_veg = np.array([0.5], dtype=np.float64)
    frac_snow_nobio = np.array([[0.25]], dtype=np.float64)
    totfrac_nobio = np.array([0.2], dtype=np.float64)

    result = thermosoil_profile_explicit_snow(
        ptn=ptn,
        cgrnd=cgrnd,
        dgrnd=dgrnd,
        cgrnd_snow=cgrnd_snow,
        dgrnd_snow=dgrnd_snow,
        temp_sol_new=temp_sol_new,
        snowtemp=snowtemp,
        veget_max=veget_max,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        totfrac_nobio=totfrac_nobio,
        nslm=2,
    )

    snow_weight = frac_snow_veg[0] * (1.0 - totfrac_nobio[0])
    nobio_weight = frac_snow_nobio[0].sum() * totfrac_nobio[0]
    temp_eff = snowtemp[0, -1] * snow_weight + temp_sol_new[0] * nobio_weight + temp_sol_new[0] * (
        1.0 - snow_weight - nobio_weight
    )
    expected = ptn.copy()
    expected[:, 0, :] = cgrnd_snow[:, -1, None] + dgrnd_snow[:, -1, None] * temp_eff
    expected[:, 1, :] = cgrnd[:, 0, :] + dgrnd[:, 0, :] * expected[:, 0, :]
    expected[:, 2, :] = cgrnd[:, 1, :] + dgrnd[:, 1, :] * expected[:, 1, :]
    veget_bg = veget_max.copy()
    veget_bg[:, 0] = np.maximum(1.0 - veget_max[:, 1:].sum(axis=1), 0.0)
    expected_stemp = np.sum(expected[:, :2, :] * veget_bg[:, None, :], axis=2)

    np.testing.assert_allclose(np.asarray(result.ptn), expected)
    np.testing.assert_allclose(np.asarray(result.stempdiag), expected_stemp)


def test_thermosoil_energy_diagnostics_split_snow_and_soil_surface_energy():
    temp_sol_new = np.array([282.0, 276.0], dtype=np.float64)
    temp_sol_beg = np.array([280.0, 278.0], dtype=np.float64)
    soilcap = np.array([1000.0, 2000.0], dtype=np.float64)
    ptn = np.ones((2, 2, 2), dtype=np.float64) * 285.0
    pcapa_en = np.array(
        [
            [[100.0, 150.0], [200.0, 250.0]],
            [[900.0, 1000.0], [1100.0, 1200.0]],
        ],
        dtype=np.float64,
    )
    veget_max = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float64)
    sn_capa = 500.0

    result = thermosoil_energy_diagnostics(
        temp_sol_new=temp_sol_new,
        temp_sol_beg=temp_sol_beg,
        soilcap=soilcap,
        pcapa_en=pcapa_en,
        veget_max=veget_max,
        ptn=ptn,
        sn_capa=sn_capa,
    )

    delta = soilcap * (temp_sol_new - temp_sol_beg)
    np.testing.assert_allclose(np.asarray(result.coldcont_incr), np.array([delta[0], 0.0]))
    np.testing.assert_allclose(np.asarray(result.surfheat_incr), np.array([0.0, delta[1]]))
    np.testing.assert_allclose(np.asarray(result.ptn_beg), ptn)
    np.testing.assert_allclose(np.asarray(result.temp_sol_beg), temp_sol_new)


def test_thermosoil_humlev_interpolates_hydrol_moisture_to_thermal_levels():
    shumdiag_perma = np.array([[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]], dtype=np.float64)
    mc_layh = np.array([[0.11, 0.21, 0.31], [0.41, 0.51, 0.61]], dtype=np.float64)
    mcl_layh = np.array([[0.10, 0.19, 0.28], [0.38, 0.47, 0.56]], dtype=np.float64)
    tmc_layh = np.array([[11.0, 21.0, 31.0], [41.0, 51.0, 61.0]], dtype=np.float64)
    mc_layh_pft = np.array(
        [
            [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
            [[0.07, 0.08], [0.09, 0.10], [0.11, 0.12]],
        ],
        dtype=np.float64,
    )
    mcl_layh_pft = mc_layh_pft - 0.005
    tmc_layh_pft = mc_layh_pft * 1000.0
    znt = np.array([0.05, 0.25, 0.70, 1.50], dtype=np.float64)
    zlt = np.array([0.10, 0.50, 1.00, 2.00], dtype=np.float64)
    dz5 = np.array([0.25, 0.60, 0.75], dtype=np.float64)
    mask = np.array([[True, False], [True, True]])

    result = thermosoil_humlev(
        shumdiag_perma=shumdiag_perma,
        mc_layh=mc_layh,
        mcl_layh=mcl_layh,
        tmc_layh=tmc_layh,
        mc_layh_pft=mc_layh_pft,
        mcl_layh_pft=mcl_layh_pft,
        tmc_layh_pft=tmc_layh_pft,
        znt=znt,
        zlt=zlt,
        dz5=dz5,
        veget_mask_2d=mask,
    )

    expected_mc = np.zeros((2, 4), dtype=np.float64)
    expected_mc[:, 0] = mc_layh[:, 0]
    expected_mc[:, 1] = mc_layh[:, 0] * (znt[1] - zlt[0]) / znt[1] + mc_layh[:, 1] * zlt[0] / znt[1]
    expected_mc[:, 2] = mc_layh[:, 1] * (zlt[2] - zlt[1]) / (zlt[2] - znt[1]) + mc_layh[:, 2] * (
        zlt[1] - znt[1]
    ) / (zlt[2] - znt[1])
    expected_mc[:, 3] = mc_layh[:, 2]
    np.testing.assert_allclose(np.asarray(result.mc_layt), expected_mc)

    expected_mcl = np.zeros((2, 4), dtype=np.float64)
    expected_mcl[:, 0] = mcl_layh[:, 0]
    expected_mcl[:, 1] = mcl_layh[:, 0] * (znt[1] - zlt[0]) / znt[1] + mcl_layh[:, 1] * zlt[0] / znt[1]
    expected_mcl[:, 2] = mcl_layh[:, 1] * (zlt[2] - zlt[1]) / (zlt[2] - znt[1]) + mcl_layh[:, 2] * (
        zlt[1] - znt[1]
    ) / (zlt[2] - znt[1])
    expected_mcl[:, 3] = mcl_layh[:, 2]
    np.testing.assert_allclose(np.asarray(result.mcl_layt), expected_mcl)
    np.testing.assert_allclose(np.asarray(result.tmc_layt), np.column_stack((tmc_layh, tmc_layh[:, -1])))

    expected_shum = np.zeros((2, 4, 2), dtype=np.float64)
    expected_shum[:, 0, :] = np.where(mask, shumdiag_perma[:, 0, None], 0.0)
    expected_shum[:, 1, :] = np.where(mask, shumdiag_perma[:, 1, None], 0.0)
    expected_shum[:, 2, :] = np.where(mask, shumdiag_perma[:, 2, None], 0.0)
    expected_shum[:, 3, :] = np.where(mask, shumdiag_perma[:, 2, None], 0.0)
    np.testing.assert_allclose(np.asarray(result.shum_ngrnd_perma), expected_shum)

    expected_mc_pft_layer1 = np.maximum(mc_layh_pft[:, 0, :], 1.0e-8)
    expected_mc_pft_layer2 = np.maximum(
        mc_layh_pft[:, 0, :] * (znt[1] - zlt[0]) / znt[1] + mc_layh_pft[:, 1, :] * zlt[0] / znt[1],
        1.0e-8,
    )
    expected_mc_pft_layer4 = mc_layh_pft[:, 2, :]
    np.testing.assert_allclose(np.asarray(result.mc_layt_pft)[:, 0, :], expected_mc_pft_layer1)
    np.testing.assert_allclose(np.asarray(result.mc_layt_pft)[:, 1, :], expected_mc_pft_layer2)
    np.testing.assert_allclose(np.asarray(result.mc_layt_pft)[:, 3, :], expected_mc_pft_layer4)


def test_thermosoil_humlev_satsoil_sets_thermal_humidity_to_one():
    shape = (1, 2)
    pft_shape = (1, 2, 1)
    result = thermosoil_humlev(
        shumdiag_perma=np.array([[0.2, 0.4]], dtype=np.float64),
        mc_layh=np.ones(shape),
        mcl_layh=np.ones(shape),
        tmc_layh=np.ones(shape),
        mc_layh_pft=np.ones(pft_shape),
        mcl_layh_pft=np.ones(pft_shape),
        tmc_layh_pft=np.ones(pft_shape),
        znt=np.array([0.05, 0.2, 0.6]),
        zlt=np.array([0.1, 0.4, 1.0]),
        dz5=np.array([0.2, 0.3]),
        satsoil=True,
    )

    np.testing.assert_allclose(np.asarray(result.shum_ngrnd_perma), np.ones((1, 3, 1)))


def _expected_cond(njsc, smc, sh2o):
    expected = np.zeros_like(smc)
    for ji, soil_class in enumerate(njsc):
        idx = soil_class - 1
        smcmax = SMCMAX_USDA[idx]
        qz = QZ_USDA[idx]
        gammd = (1.0 - smcmax) * 2700.0
        thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
        thko = 2.0 if qz > 0.2 else 3.0
        thks = THKQTZ**qz * thko ** (1.0 - qz)
        for jg in range(smc.shape[1]):
            satratio = smc[ji, jg] / smcmax
            if smc[ji, jg] > 1.0e-8:
                xunfroz = sh2o[ji, jg] / smc[ji, jg]
                xu = xunfroz * smcmax
                thksat = thks ** (1.0 - smcmax) * THKICE ** (smcmax - xu) * THKW**xu
            else:
                thksat = 0.0
            if sh2o[ji, jg] + 0.0005 < smc[ji, jg]:
                ake = satratio
            elif satratio > 0.1:
                ake = np.log10(satratio) + 1.0
            elif satratio > 0.05:
                ake = 0.7 * np.log10(satratio) + 1.0
            else:
                ake = 0.0
            expected[ji, jg] = ake * (thksat - thkdry) + thkdry
    return expected


def test_thermosoil_cond_matches_farouki_johansen_formula():
    njsc = np.array([1, 5], dtype=np.int32)
    smc = np.array([[0.30, 0.08, 0.01], [0.35, 0.03, 0.20]], dtype=np.float64)
    sh2o = np.array([[0.30, 0.02, 0.01], [0.10, 0.03, 0.20]], dtype=np.float64)

    result = thermosoil_cond(njsc=njsc, smc=smc, sh2o=sh2o)

    np.testing.assert_allclose(np.asarray(result.cnd), _expected_cond(njsc, smc, sh2o), rtol=1e-13)


def test_thermosoil_cond_pft_applies_organic_fraction_to_dry_and_solid_terms():
    njsc = np.array([3], dtype=np.int32)
    smc = np.array([[0.25, 0.12]], dtype=np.float64)
    sh2o = np.array([[0.25, 0.03]], dtype=np.float64)
    zx1 = np.array([[[0.0, 0.50], [0.25, 0.75]]], dtype=np.float64)
    zx2 = 1.0 - zx1
    porosnet = zx1 * 0.92 + zx2 * SMCMAX_USDA[2]

    result = thermosoil_cond_pft(njsc=njsc, smc=smc, sh2o=sh2o, zx1=zx1, zx2=zx2, porosnet=porosnet)

    qz = QZ_USDA[2]
    gammd = (1.0 - SMCMAX_USDA[2]) * 2700.0
    thkdry_min = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    thko = 2.0 if qz > 0.2 else 3.0
    thks_min = THKQTZ**qz * thko ** (1.0 - qz)
    expected_thkdry = zx1 * COND_DRY_ORG + zx2 * thkdry_min
    expected_thks = zx1 * COND_SOLID_ORG + zx2 * thks_min
    np.testing.assert_allclose(np.asarray(result.thkdry), expected_thkdry)
    np.testing.assert_allclose(np.asarray(result.thks), expected_thks)
    assert np.asarray(result.cnd)[0, 0, 0] != pytest.approx(np.asarray(result.cnd)[0, 0, 1])


def test_thermosoil_getdiff_explicit_computes_freeze_capacity_conductivity_and_snow():
    ptn = np.array(
        [
            [
                [ZERO_CELSIUS - 2.0, ZERO_CELSIUS - 2.0],
                [ZERO_CELSIUS + 2.0, ZERO_CELSIUS + 2.0],
                [ZERO_CELSIUS, ZERO_CELSIUS],
            ]
        ],
        dtype=np.float64,
    )
    njsc = np.array([4], dtype=np.int32)
    veget_max = np.array([[0.7, 0.2]], dtype=np.float64)
    shum = np.array([[[0.3, 0.3], [0.5, 0.5], [0.7, 0.7]]], dtype=np.float64)
    mc_layt = np.array([[0.20, 0.30, 0.40]], dtype=np.float64)
    tmc_layt_pft = np.array([[[20.0, 25.0], [30.0, 35.0], [40.0, 45.0]]], dtype=np.float64)
    dlt = np.array([0.1, 0.2, 0.4], dtype=np.float64)
    zx1 = np.array([[[0.0, 0.0], [0.2, 0.2], [0.4, 0.4]]], dtype=np.float64)
    snowrho = np.array([[100.0, 300.0]], dtype=np.float64)
    snowtemp = np.array([[260.0, 270.0]], dtype=np.float64)
    pb = np.array([1000.0], dtype=np.float64)

    result = thermosoil_getdiff_explicit(
        ptn=ptn,
        njsc=njsc,
        veget_max=veget_max,
        shum_ngrnd_permalong=shum,
        mc_layt=mc_layt,
        tmc_layt_pft=tmc_layt_pft,
        snowrho=snowrho,
        snowtemp=snowtemp,
        pb=pb,
        dlt=dlt,
        zx1=zx1,
        ok_laidev=np.array([False, True]),
    )

    soil_idx = njsc[0] - 1
    so_capa_ice = SO_CAPA_DRY + POROS * CAPA_ICE * RHO_ICE
    so_capa_dry_net = zx1 * SO_CAPA_DRY_ORG + (1.0 - zx1) * SO_CAPA_DRY_NS_USDA[soil_idx]
    poros_net = zx1 * 0.92 + (1.0 - zx1) * SMCMAX_USDA[soil_idx]
    expected_frozen_pft0 = so_capa_dry_net[0, 0, 0] * (1.0 - poros_net[0, 0, 0]) + so_capa_ice * mc_layt[0, 0]
    expected_frozen_pft1 = so_capa_dry_net[0, 0, 1] + so_capa_ice * tmc_layt_pft[0, 0, 1] / MILLE / dlt[0]
    expected_unfrozen_pft0 = so_capa_dry_net[0, 1, 0] * (1.0 - poros_net[0, 1, 0]) + WATER_CAPA * mc_layt[0, 1]
    expected_unfrozen_pft1 = so_capa_dry_net[0, 1, 1] + WATER_CAPA * tmc_layt_pft[0, 1, 1] / MILLE / dlt[1]
    xx = (ptn[0, 2, 0] - (ZERO_CELSIUS - FR_DT / 2.0)) / FR_DT
    expected_supp = shum[0, 2, 0] * LHF * RHO_WATER / FR_DT
    expected_transition_pft0 = (
        so_capa_dry_net[0, 2, 0] * (1.0 - poros_net[0, 2, 0])
        + WATER_CAPA * mc_layt[0, 2] * xx
        + so_capa_ice * mc_layt[0, 2] * (1.0 - xx)
        + expected_supp
    )
    expected_transition_pft1 = (
        so_capa_dry_net[0, 2, 1]
        + WATER_CAPA * tmc_layt_pft[0, 2, 1] / MILLE / dlt[2] * xx
        + so_capa_ice * tmc_layt_pft[0, 2, 1] / MILLE / dlt[2] * (1.0 - xx)
    )

    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 0, 0], expected_frozen_pft0)
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 0, 1], expected_frozen_pft1)
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 1, 0], expected_unfrozen_pft0)
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 1, 1], expected_unfrozen_pft1)
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 2, 0], expected_transition_pft0)
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 2, 1], expected_transition_pft1)
    np.testing.assert_allclose(np.asarray(result.profil_froz)[0, :, 0], np.array([1.0, 0.0, 0.5]))
    np.testing.assert_allclose(np.asarray(result.pcappa_supp)[0, 2, 0], expected_supp)
    np.testing.assert_allclose(np.asarray(result.pcapa_snow), snowrho * XCI)
    expected_pkappa_snow = (ZSNOWTHRMCOND1 + ZSNOWTHRMCOND2 * snowrho * snowrho) + np.maximum(
        0.0,
        (ZSNOWTHRMCOND_AVAP + ZSNOWTHRMCOND_BVAP / (snowtemp + ZSNOWTHRMCOND_CVAP))
        * (XP00 / (pb[:, None] * 100.0)),
    )
    np.testing.assert_allclose(np.asarray(result.pkappa_snow), expected_pkappa_snow)
    assert np.asarray(result.pkappa).shape == ptn.shape


def test_thermosoil_getdiff_old_thermix_with_snow_matches_bucket_snow_layers():
    """Fortran: thermosoil_getdiff_old_thermix_with_snow lines 2932-3015."""

    snow_h = 0.05
    snow = np.array([snow_h * SN_DENS], dtype=np.float64)
    njsc = np.array([1], dtype=np.int32)
    dlt = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
    zlt = np.array([0.1, 0.3, 0.7, 1.5], dtype=np.float64)
    mc_layt = np.array([[0.2, 0.25, 0.3, 0.35]], dtype=np.float64)
    mcl_layt = np.array([[0.18, 0.22, 0.27, 0.30]], dtype=np.float64)
    tmc_layt = np.array([[20.0, 50.0, 120.0, 280.0]], dtype=np.float64)
    mc_layt_pft = np.array([[[0.2, 0.21], [0.25, 0.26], [0.3, 0.31], [0.35, 0.36]]], dtype=np.float64)
    mcl_layt_pft = np.array([[[0.18, 0.19], [0.22, 0.23], [0.27, 0.28], [0.30, 0.31]]], dtype=np.float64)
    tmc_layt_pft = np.array([[[20.0, 21.0], [50.0, 52.0], [120.0, 124.0], [280.0, 288.0]]], dtype=np.float64)

    result = thermosoil_getdiff_old_thermix_with_snow(
        snow=snow,
        njsc=njsc,
        mc_layt=mc_layt,
        mcl_layt=mcl_layt,
        tmc_layt=tmc_layt,
        mc_layt_pft=mc_layt_pft,
        mcl_layt_pft=mcl_layt_pft,
        tmc_layt_pft=tmc_layt_pft,
        dlt=dlt,
        zlt=zlt,
        ok_laidev=np.array([False, True]),
    )

    soil_idx = 0
    pcapa_wet_l1 = SO_CAPA_DRY_NS_USDA[soil_idx] + WATER_CAPA * SMCMAX_USDA[soil_idx]
    wet_cond = np.asarray(
        thermosoil_cond(
            njsc=njsc,
            smc=np.full((1, 4), SMCMAX_USDA[soil_idx], dtype=np.float64),
            sh2o=np.full((1, 4), SMCMAX_USDA[soil_idx], dtype=np.float64),
        ).cnd
    )
    zx1 = snow_h / zlt[0]
    zx2 = (zlt[0] - snow_h) / zlt[0]
    expected_l1_pcapa = zx1 * SN_CAPA + zx2 * pcapa_wet_l1
    expected_l1_pkappa = 1.0 / (zx1 / SN_COND + zx2 / wet_cond[0, 0])
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 0, :], expected_l1_pcapa)
    np.testing.assert_allclose(np.asarray(result.pcapa_en)[0, 0, :], SN_CAPA)
    np.testing.assert_allclose(np.asarray(result.pkappa)[0, 0, :], expected_l1_pkappa)

    expected_l2_grid_pcapa = SO_CAPA_DRY_NS_USDA[soil_idx] + WATER_CAPA * tmc_layt[0, 1] / 1000.0 / dlt[1]
    expected_l2_pft_pcapa = SO_CAPA_DRY_NS_USDA[soil_idx] + WATER_CAPA * tmc_layt_pft[0, 1, 1] / 1000.0 / dlt[1]
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 1, 0], expected_l2_grid_pcapa)
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 1, 1], expected_l2_pft_pcapa)
    np.testing.assert_allclose(np.asarray(result.pcapa)[0, 2:, :], SO_CAPA_DRY)
    np.testing.assert_allclose(np.asarray(result.pkappa)[0, 2:, :], SO_COND_DRY)
    assert any("thermosoil_getdiff_old_thermix_with_snow" in item for item in result.provenance)


def test_thermosoil_coef_no_explicit_snow_clears_snow_coefficients():
    """Fortran: thermosoil_coef lines 1633-1725 when ok_explicitsnow is false."""

    result = thermosoil_coef_no_explicit_snow(
        ptn=np.full((1, 3, 2), 282.0, dtype=np.float64),
        pcapa=np.full((1, 3, 2), 2.0e6, dtype=np.float64),
        pkappa=np.full((1, 3, 2), 1.0, dtype=np.float64),
        temp_sol_new=np.array([280.0], dtype=np.float64),
        temp_sol_new_pft=np.array([[280.0, 281.0]], dtype=np.float64),
        veget_max=np.array([[0.4, 0.6]], dtype=np.float64),
        ok_laidev=np.array([False, True]),
        dlt=np.array([0.1, 0.2, 0.4], dtype=np.float64),
        dz1=np.array([10.0, 3.0], dtype=np.float64),
        dt_sechiba=1800.0,
        lambda_thermal=0.5,
        cgrnd_snow_template=np.ones((1, 3), dtype=np.float64),
        dgrnd_snow_template=np.ones((1, 3), dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.lambda_snow), [0.5])
    np.testing.assert_allclose(np.asarray(result.cgrnd_snow), 0.0)
    np.testing.assert_allclose(np.asarray(result.dgrnd_snow), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowcap), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowflx), 0.0)
    np.testing.assert_allclose(np.asarray(result.soilcap), np.asarray(result.soil.soilcap))
    np.testing.assert_allclose(np.asarray(result.soilflx), np.asarray(result.soil.soilflx))


def test_thermosoil_soilc_tempdiff_fraction_matches_source_clipping():
    soilc_total = np.array([[[0.0, 65.0], [130.0, 260.0]]], dtype=np.float64)
    zx1, zx2 = thermosoil_soilc_tempdiff_fraction(soilc_total=soilc_total, soilc_max=130.0)
    expected = np.minimum(soilc_total / 130.0, 1.0)
    np.testing.assert_allclose(np.asarray(zx1), expected)
    np.testing.assert_allclose(np.asarray(zx2), 1.0 - expected)

    refsoc = np.array([[65.0, 260.0]], dtype=np.float64)
    zx1_ref, _ = thermosoil_soilc_tempdiff_fraction(refsoc=refsoc, soilc_max=130.0)
    expected_ref = np.minimum(refsoc / 130.0, 1.0)[:, :, None]
    np.testing.assert_allclose(np.asarray(zx1_ref), expected_ref)

    zx1_ref_pft, _ = thermosoil_soilc_tempdiff_fraction(refsoc=refsoc, nvm=3, soilc_max=130.0)
    expected_ref_pft = np.broadcast_to(expected_ref, (1, 2, 3))
    np.testing.assert_allclose(np.asarray(zx1_ref_pft), expected_ref_pft)

    with pytest.raises(ValueError, match="exactly one"):
        thermosoil_soilc_tempdiff_fraction(soilc_total=soilc_total, refsoc=refsoc)


def test_thermosoil_toporganiclayer_fraction_matches_layer_overlap_formula():
    organic_layer_thick = np.array([0.0, 0.05, 0.3, 2.0], dtype=np.float64)
    zlt = np.array([0.1, 0.5, 1.0], dtype=np.float64)

    zx1, zx2 = thermosoil_toporganiclayer_fraction(
        organic_layer_thick=organic_layer_thick,
        zlt=zlt,
        nvm=2,
    )

    expected_2d = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected = np.broadcast_to(expected_2d[:, :, None], (4, 3, 2))
    np.testing.assert_allclose(np.asarray(zx1), expected)
    np.testing.assert_allclose(np.asarray(zx2), 1.0 - expected)


def test_thermosoil_explicit_step_composes_profile_getdiff_coef_and_final_state():
    ptn = np.array([[[280.0], [281.0], [282.0]]], dtype=np.float64)
    cgrnd = np.array([[[275.0], [276.0]]], dtype=np.float64)
    dgrnd = np.array([[[0.2], [0.3]]], dtype=np.float64)
    temp_sol_new = np.array([279.0], dtype=np.float64)
    temp_sol_new_pft = np.array([[279.0]], dtype=np.float64)
    snowrho = np.array([[100.0, 200.0, 300.0]], dtype=np.float64)
    snowtemp = np.array([[270.0, 268.0, 266.0]], dtype=np.float64)
    snowdz = np.array([[0.01, 0.02, 0.04]], dtype=np.float64)
    shumdiag_perma = np.array([[0.2, 0.3]], dtype=np.float64)
    mc_layh = np.array([[0.2, 0.3]], dtype=np.float64)
    mcl_layh = np.array([[0.2, 0.3]], dtype=np.float64)
    tmc_layh = np.array([[20.0, 30.0]], dtype=np.float64)
    mc_layh_pft = np.array([[[0.2], [0.3]]], dtype=np.float64)
    mcl_layh_pft = np.array([[[0.2], [0.3]]], dtype=np.float64)
    tmc_layh_pft = np.array([[[20.0], [30.0]]], dtype=np.float64)
    veget_max = np.array([[1.0]], dtype=np.float64)
    dlt = np.array([0.1, 0.2, 0.4], dtype=np.float64)
    dz1 = np.array([10.0, 3.0], dtype=np.float64)
    zlt = np.array([0.1, 0.3, 0.7], dtype=np.float64)
    znt = np.array([0.05, 0.2, 0.5], dtype=np.float64)
    dz5 = np.array([0.2, 0.4], dtype=np.float64)
    refsoc = np.array([[0.0, 10.0, 20.0]], dtype=np.float64)

    result = thermosoil_explicit_step(
        ptn=ptn,
        cgrnd=cgrnd,
        dgrnd=dgrnd,
        cgrnd_snow=np.array([[260.0, 261.0, 262.0]], dtype=np.float64),
        dgrnd_snow=np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        snowrho=snowrho,
        snowtemp=snowtemp,
        snowdz=snowdz,
        shumdiag_perma=shumdiag_perma,
        mc_layh=mc_layh,
        mcl_layh=mcl_layh,
        tmc_layh=tmc_layh,
        mc_layh_pft=mc_layh_pft,
        mcl_layh_pft=mcl_layh_pft,
        tmc_layh_pft=tmc_layh_pft,
        njsc=np.array([1], dtype=np.int32),
        veget_max=veget_max,
        pb=np.array([1000.0], dtype=np.float64),
        dlt=dlt,
        dz1=dz1,
        zlt=zlt,
        znt=znt,
        dz5=dz5,
        dt_sechiba=1800.0,
        lambda_thermal=0.5,
        frac_snow_veg=np.array([0.4], dtype=np.float64),
        frac_snow_nobio=np.array([[0.0]], dtype=np.float64),
        totfrac_nobio=np.array([0.0], dtype=np.float64),
        temp_sol_beg=np.array([278.0], dtype=np.float64),
        soilcap_initial=np.array([1000.0], dtype=np.float64),
        refsoc=refsoc,
        ok_laidev=np.array([False]),
        veget_mask_2d=np.array([[True]]),
        shum_ngrnd_permalong_previous=np.full((1, 3, 1), 0.9, dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.profile.ptn), np.asarray(result.final.deeptemp_prof))
    expected_long = (
        np.asarray(result.humlev.shum_ngrnd_perma) * 1800.0
        + np.full((1, 3, 1), 0.9) * (30.0 * 86400.0 - 1800.0)
    ) / (30.0 * 86400.0)
    np.testing.assert_allclose(np.asarray(result.final.deephum_prof), expected_long)
    np.testing.assert_allclose(np.asarray(result.final.gtemp), np.asarray(result.final.ptn_pftmean)[:, 0])
    np.testing.assert_allclose(np.asarray(result.coef.soilcap), np.asarray(result.coef.soilcap))
    assert np.asarray(result.getdiff.pcapa).shape == ptn.shape
    assert np.asarray(result.coef.cgrnd_snow).shape == snowrho.shape


def test_thermosoil_no_explicit_snow_step_composes_old_thermix_bucket_branch():
    ptn = np.array([[[280.0], [281.0], [282.0], [283.0]]], dtype=np.float64)
    cgrnd = np.array([[[275.0], [276.0], [277.0]]], dtype=np.float64)
    dgrnd = np.array([[[0.2], [0.3], [0.4]]], dtype=np.float64)
    temp_sol_new = np.array([279.0], dtype=np.float64)
    temp_sol_new_pft = np.array([[279.0]], dtype=np.float64)
    shumdiag_perma = np.array([[0.2, 0.3, 0.4]], dtype=np.float64)
    mc_layh = np.array([[0.2, 0.3, 0.4]], dtype=np.float64)
    mcl_layh = np.array([[0.2, 0.3, 0.4]], dtype=np.float64)
    tmc_layh = np.array([[20.0, 30.0, 40.0]], dtype=np.float64)
    mc_layh_pft = np.array([[[0.2], [0.3], [0.4]]], dtype=np.float64)
    mcl_layh_pft = np.array([[[0.2], [0.3], [0.4]]], dtype=np.float64)
    tmc_layh_pft = np.array([[[20.0], [30.0], [40.0]]], dtype=np.float64)

    result = thermosoil_no_explicit_snow_step(
        ptn=ptn,
        cgrnd=cgrnd,
        dgrnd=dgrnd,
        cgrnd_snow=np.ones((1, 3), dtype=np.float64),
        dgrnd_snow=np.ones((1, 3), dtype=np.float64),
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        snow=np.array([0.02 * SN_DENS], dtype=np.float64),
        shumdiag_perma=shumdiag_perma,
        mc_layh=mc_layh,
        mcl_layh=mcl_layh,
        tmc_layh=tmc_layh,
        mc_layh_pft=mc_layh_pft,
        mcl_layh_pft=mcl_layh_pft,
        tmc_layh_pft=tmc_layh_pft,
        njsc=np.array([1], dtype=np.int32),
        veget_max=np.array([[1.0]], dtype=np.float64),
        dlt=np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64),
        dz1=np.array([10.0, 3.0, 1.0], dtype=np.float64),
        zlt=np.array([0.1, 0.3, 0.7, 1.5], dtype=np.float64),
        znt=np.array([0.05, 0.2, 0.5, 1.1], dtype=np.float64),
        dz5=np.array([0.2, 0.4, 0.8], dtype=np.float64),
        dt_sechiba=1800.0,
        lambda_thermal=0.5,
        temp_sol_beg=np.array([278.0], dtype=np.float64),
        soilcap_initial=np.array([1000.0], dtype=np.float64),
        ok_laidev=np.array([False]),
    )

    np.testing.assert_allclose(np.asarray(result.profile.ptn), np.asarray(result.final.deeptemp_prof))
    np.testing.assert_allclose(np.asarray(result.coef.cgrnd_snow), 0.0)
    np.testing.assert_allclose(np.asarray(result.coef.dgrnd_snow), 0.0)
    np.testing.assert_allclose(np.asarray(result.coef.lambda_snow), [0.5])
    assert np.asarray(result.getdiff.pcapa).shape == ptn.shape
    assert any("thermosoil_getdiff_old_thermix_with_snow" in item for item in result.provenance)
