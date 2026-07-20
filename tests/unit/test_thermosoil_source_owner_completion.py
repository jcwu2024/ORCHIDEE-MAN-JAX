from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    COND_DRY_ORG,
    COND_SOLID_ORG,
    QZ_USDA,
    SMCMAX_USDA,
    SN_CAPA,
    SN_COND,
    SO_CAPA_DRY,
    SO_CAPA_WET,
    SO_COND_DRY,
    SO_COND_WET,
    THKICE,
    THKQTZ,
    THKW,
    THERMOSOIL_EXTERNAL_IO_BOUNDARIES,
    ZERO_CELSIUS,
    add_heat_zimov,
    thermosoil_cond_nopft,
    thermosoil_energy_diagnostics,
    thermosoil_getdiff_thinsnow,
    thermosoil_wlupdate,
)


def _cond_nopft_scalar(njsc, smc, sh2o, zx1, zx2, porosnet):
    jst = njsc - 1
    gammd = (1.0 - SMCMAX_USDA[jst]) * 2700.0
    thkdry_min = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    thkdry = zx1 * COND_DRY_ORG + zx2 * thkdry_min
    thko = 2.0 if QZ_USDA[jst] > 0.2 else 3.0
    thks_min = THKQTZ ** QZ_USDA[jst] * thko ** (1.0 - QZ_USDA[jst])
    thks = zx1 * COND_SOLID_ORG + zx2 * thks_min
    satratio = smc / porosnet
    if smc > 1.0e-8:
        xunfroz = sh2o / smc
        xu = xunfroz * porosnet
        thksat = thks ** (1.0 - porosnet) * THKICE ** (porosnet - xu) * THKW**xu
    else:
        thksat = 0.0
    if sh2o + 0.0005 < smc:
        ake = satratio
    elif satratio > 0.1:
        ake = np.log10(satratio) + 1.0
    elif satratio > 0.05:
        ake = 0.7 * np.log10(satratio) + 1.0
    else:
        ake = 0.0
    return ake * (thksat - thkdry) + thkdry


def test_cond_nopft_covers_quartz_moisture_freeze_and_kersten_arms():
    njsc = np.array([1, 5], dtype=np.int32)  # qz > .2 and qz <= .2
    smc = np.array([[0.30, 0.08, 0.03, 1.0e-9], [0.35, 0.04, 0.02, 0.10]])
    sh2o = np.array([[0.10, 0.08, 0.03, 1.0e-9], [0.35, 0.04, 0.02, 0.01]])
    zx1 = np.array([[0.0, 0.25, 0.50, 0.75], [1.0, 0.6, 0.3, 0.1]])
    zx2 = 1.0 - zx1
    porosnet = np.array([[0.43, 0.50, 0.50, 0.50], [0.46, 0.50, 0.50, 0.50]])

    result = thermosoil_cond_nopft(
        njsc=njsc,
        smc=smc,
        sh2o=sh2o,
        zx1=zx1,
        zx2=zx2,
        porosnet=porosnet,
    )

    expected = np.empty_like(smc)
    for ji in range(smc.shape[0]):
        for jg in range(smc.shape[1]):
            expected[ji, jg] = _cond_nopft_scalar(
                njsc[ji], smc[ji, jg], sh2o[ji, jg], zx1[ji, jg], zx2[ji, jg], porosnet[ji, jg]
            )
    np.testing.assert_allclose(result.cnd, expected, rtol=2e-15, atol=0.0)
    assert result.provenance == (
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_cond_nopft lines 2129-2231",
    )


def test_getdiff_thinsnow_overwrites_only_active_first_layer():
    npts, ngrnd, nvm = 4, 3, 2
    ptn = np.full((npts, ngrnd, nvm), ZERO_CELSIUS)
    ptn[0, 0, :] = ZERO_CELSIUS - 2.0
    ptn[1, 0, :] = ZERO_CELSIUS + 2.0
    shum = np.full_like(ptn, 0.5)
    snowdz = np.array([[0.003, 0.003], [0.001, 0.001], [0.0, 0.0], [0.01, 0.01]])
    original = np.arange(npts * ngrnd * nvm, dtype=np.float64).reshape(npts, ngrnd, nvm) + 10.0
    mask = np.array([[True, False], [True, True], [True, True], [True, True]])

    result = thermosoil_getdiff_thinsnow(
        ptn=ptn,
        shum_ngrnd_permalong=shum,
        snowdz=snowdz,
        pcapa=original,
        pcapa_en=original + 100.0,
        pkappa=original + 200.0,
        profil_froz=original + 300.0,
        veget_mask_2d=mask,
        zlt=np.array([0.004, 0.02, 0.08]),
    )

    base_capa = SO_CAPA_DRY + 0.5 * (SO_CAPA_WET - SO_CAPA_DRY)
    base_cond = SO_COND_DRY + 0.5 * (SO_COND_WET - SO_COND_DRY)
    np.testing.assert_allclose(result.pcapa[0, 0, 0], SN_CAPA)
    np.testing.assert_allclose(result.pkappa[0, 0, 0], SN_COND)
    partial_snow = 0.5
    np.testing.assert_allclose(result.pcapa[1, 0, 0], partial_snow * SN_CAPA + partial_snow * base_capa)
    np.testing.assert_allclose(
        result.pkappa[1, 0, 0], 1.0 / (partial_snow / SN_COND + partial_snow / base_cond)
    )
    np.testing.assert_array_equal(result.pcapa[:, 1:, :], original[:, 1:, :])
    np.testing.assert_array_equal(result.pcapa[0, 0, 1], original[0, 0, 1])
    np.testing.assert_array_equal(result.pcapa[2:, :, :], original[2:, :, :])
    np.testing.assert_allclose(result.profil_froz[0, 0, 0], 1.0)
    np.testing.assert_allclose(result.profil_froz[1, 0, :], 0.0)
    np.testing.assert_allclose(result.pcapa_en[0:2, 0, 0], SN_CAPA)


def test_energy_owner_covers_snow_like_and_soil_surface_branches():
    result = thermosoil_energy_diagnostics(
        temp_sol_new=np.array([275.0, 276.0]),
        temp_sol_beg=np.array([274.0, 274.0]),
        soilcap=np.array([10.0, 20.0]),
        pcapa_en=np.array([[[100.0]], [[SN_CAPA + 1.0]]]),
        veget_max=np.ones((2, 1)),
        ptn=np.array([[[271.0]], [[272.0]]]),
        sn_capa=SN_CAPA,
    )
    np.testing.assert_array_equal(result.coldcont_incr, [10.0, 0.0])
    np.testing.assert_array_equal(result.surfheat_incr, [0.0, 40.0])
    np.testing.assert_array_equal(result.temp_sol_beg, [275.0, 276.0])


def test_wlupdate_changes_only_thawed_active_levels_up_to_ndeep():
    ptn = np.array([[[275.0, 275.0], [275.0, 270.0], [275.0, 275.0]]])
    hsd = np.full_like(ptn, 0.8)
    hsdlong = np.full_like(ptn, 0.2)
    result = thermosoil_wlupdate(
        ptn=ptn,
        hsd=hsd,
        hsdlong=hsdlong,
        veget_mask_2d=np.array([[True, False]]),
        dt_sechiba=86400.0,
        ndeep=2,
    )
    np.testing.assert_allclose(result.hsdlong[0, :2, 0], 0.22)
    np.testing.assert_array_equal(result.hsdlong[0, :, 1], 0.2)
    np.testing.assert_array_equal(result.hsdlong[0, 2, 0], 0.2)


def test_add_heat_zimov_updates_active_pfts_and_recomputes_weighted_mean():
    ptn = np.full((1, 2, 2), 270.0)
    result = add_heat_zimov(
        ptn=ptn,
        heat_zimov=np.array([[[2.0, 9.0], [4.0, 9.0]]]),
        pcapa=np.full((1, 2, 2), 2.0),
        dlt=np.array([1.0, 2.0]),
        dt_sechiba=1.0,
        veget_mask_2d=np.array([[True, False]]),
        veget_max_bg=np.array([[0.75, 0.25]]),
    )
    np.testing.assert_array_equal(result.ptn, [[[271.0, 270.0], [271.0, 270.0]]])
    np.testing.assert_array_equal(result.ptn_pftmean, [[270.75, 270.75]])


def test_refsoc_io_boundary_routes_to_structured_process_owner():
    assert THERMOSOIL_EXTERNAL_IO_BOUNDARIES == (
        (
            "read_refsocfile",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::read_refSOCfile lines 3227-3405",
            "structured NetCDF read and aggregate_p callback routed by read_refsocfile",
        ),
    )
