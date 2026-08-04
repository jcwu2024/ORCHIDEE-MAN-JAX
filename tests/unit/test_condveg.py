from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.condveg import (  # noqa: E402
    CANOPY_C1,
    CANOPY_C2,
    CANOPY_C3,
    CDRAG_FOLIAGE,
    CT_KARMAN,
    CT_LEAF,
    HEIGHT_DISPLACEMENT,
    MIN_SECHIBA,
    MIN_WIND,
    PB_STD,
    PRANDTL,
    SN_DENS,
    SNOWCRI_ALB,
    Z0_BARE,
    Z0_ICE,
    ZERO_CELSIUS,
    condveg_albedo_explicit,
    condveg_albedo_snow_mean,
    condveg_frac_snow,
    condveg_main_minimal,
    condveg_main_minimal_coverage,
    condveg_prescribed_roughness,
    condveg_z0cdrag,
    condveg_z0cdrag_dyn,
    run_condveg_first_step_module,
)


def _expected_dyn_roughness(
    veget,
    veget_max,
    frac_nobio,
    totfrac_nobio,
    zlev,
    height,
    temp_air,
    pb,
    u,
    v,
    lai,
    frac_snow_veg,
):
    ztmp = np.maximum(10.0, zlev)
    z0_ground = (1.0 - frac_snow_veg) * Z0_BARE + frac_snow_veg * Z0_BARE / 10.0
    z0m = veget_max[:, 0] * (CT_KARMAN / np.log(ztmp / z0_ground)) ** 2

    wind = np.sqrt(u * u + v * v)
    u_star = CT_KARMAN * np.maximum(MIN_WIND, wind) / np.log(zlev / z0_ground)
    reynolds = z0_ground * u_star / (
        1.327e-5 * (PB_STD / pb) * (temp_air / ZERO_CELSIUS) ** 1.81
    )
    kbs_m1 = 2.46 * reynolds ** 0.25 - np.log(7.4)
    z0h = veget_max[:, 0] * (
        CT_KARMAN / np.log(ztmp / (z0_ground / np.exp(kbs_m1)))
    ) ** 2

    sumveg = veget_max[:, 0].copy()
    ave_height = np.zeros_like(zlev)
    roughheight_pft = np.full_like(height, np.nan)

    for jv in range(1, veget_max.shape[1]):
        active = veget_max[:, jv] > 0.0
        eta = CANOPY_C1 - CANOPY_C2 * np.exp(-CANOPY_C3 * CDRAG_FOLIAGE * lai[:, jv])
        z0m_pft = (
            height[:, jv]
            * (1.0 - HEIGHT_DISPLACEMENT)
            * (np.exp(-CT_KARMAN / eta) - np.exp(-CT_KARMAN / (CANOPY_C1 - CANOPY_C2)))
        ) + z0_ground
        z0m = np.where(
            active,
            z0m + veget_max[:, jv] * (CT_KARMAN / np.log(ztmp / z0m_pft)) ** 2,
            z0m,
        )
        fc = veget[:, jv] / np.where(active, veget_max[:, jv], 1.0)
        fs = 1.0 - fc
        eta_ec = (CDRAG_FOLIAGE * lai[:, jv]) / (2.0 * eta * eta)
        u_star = CT_KARMAN * np.maximum(MIN_WIND, wind) / np.log(
            (zlev + height[:, jv] * (1.0 - HEIGHT_DISPLACEMENT)) / z0m_pft
        )
        reynolds = z0_ground * u_star / (
            1.327e-5 * (PB_STD / pb) * (temp_air / ZERO_CELSIUS) ** 1.81
        )
        kbs_m1 = 2.46 * reynolds ** 0.25 - np.log(7.4)
        ct_star = PRANDTL ** (-2.0 / 3.0) * np.sqrt(1.0 / reynolds)
        soil_kb = kbs_m1 * fs**2.0
        canopy_kb = soil_kb.copy()
        canopy_mask = lai[:, jv] > MIN_SECHIBA
        canopy_kb[canopy_mask] = (
            (CT_KARMAN * CDRAG_FOLIAGE)
            / (4.0 * CT_LEAF * eta[canopy_mask] * (1.0 - np.exp(-eta_ec[canopy_mask] / 2.0)))
            * fc[canopy_mask] ** 2.0
            + 2.0
            * fc[canopy_mask]
            * fs[canopy_mask]
            * (CT_KARMAN * eta[canopy_mask] * z0m_pft[canopy_mask] / height[:, jv][canopy_mask])
            / ct_star[canopy_mask]
            + kbs_m1[canopy_mask] * fs[canopy_mask] ** 2.0
        )
        kb_m1 = np.where(canopy_mask, canopy_kb, soil_kb)
        z0h_pft = z0m_pft / np.exp(kb_m1)
        z0h = np.where(
            active,
            z0h + veget_max[:, jv] * (CT_KARMAN / np.log(ztmp / z0h_pft)) ** 2,
            z0h,
        )
        sumveg = np.where(active, sumveg + veget_max[:, jv], sumveg)
        ave_height = np.where(active, ave_height + veget_max[:, jv] * height[:, jv], ave_height)
        roughheight_pft[:, jv] = height[:, jv] * (1.0 - HEIGHT_DISPLACEMENT)

    z0h = np.where(sumveg > 0.0, z0h / sumveg, z0h)
    z0m = np.where(sumveg > 0.0, z0m / sumveg, z0m)
    z0h = (1.0 - totfrac_nobio) * z0h
    z0m = (1.0 - totfrac_nobio) * z0m

    z0m += frac_nobio[:, 0] * (CT_KARMAN / np.log(ztmp / Z0_ICE)) ** 2
    u_star = CT_KARMAN * np.maximum(MIN_WIND, wind) / np.log(zlev / Z0_ICE)
    reynolds = Z0_ICE * u_star / (
        1.327e-5 * (PB_STD / pb) * (temp_air / ZERO_CELSIUS) ** 1.81
    )
    kbs_m1 = 2.46 * reynolds ** 0.25 - np.log(7.4)
    z0h += frac_nobio[:, 0] * (CT_KARMAN / np.log(ztmp / (Z0_ICE / np.exp(kbs_m1)))) ** 2

    z0h = ztmp / np.exp(CT_KARMAN / np.sqrt(z0h))
    z0m = ztmp / np.exp(CT_KARMAN / np.sqrt(z0m))
    roughheight = ave_height * (1.0 - HEIGHT_DISPLACEMENT)
    return z0m, z0h, roughheight, roughheight_pft


def _expected_static_roughness(
    veget,
    veget_max,
    frac_nobio,
    totfrac_nobio,
    zlev,
    height,
    tot_bare_soil,
    is_tree,
    z0_over_height,
    ratio_z0m_z0h,
):
    ztmp = np.maximum(10.0, zlev)
    z0m = tot_bare_soil * (CT_KARMAN / np.log(ztmp / Z0_BARE)) ** 2
    z0h = tot_bare_soil * (CT_KARMAN / np.log(ztmp / (Z0_BARE / ratio_z0m_z0h[0]))) ** 2
    sumveg = tot_bare_soil.copy()
    ave_height = np.zeros_like(zlev)
    roughheight_pft = np.full_like(height, np.nan)

    for jv in range(1, veget.shape[1]):
        d_veg = veget_max[:, jv] if is_tree[jv] else veget[:, jv]
        z0_pft = np.maximum(height[:, jv] * z0_over_height[jv], Z0_BARE)
        z0m += d_veg * (CT_KARMAN / np.log(ztmp / z0_pft)) ** 2
        z0h += d_veg * (CT_KARMAN / np.log(ztmp / (z0_pft / ratio_z0m_z0h[jv]))) ** 2
        sumveg += d_veg
        ave_height += veget_max[:, jv] * height[:, jv]

    active = sumveg > 0.0
    z0m[active] /= sumveg[active]
    z0h[active] /= sumveg[active]
    z0m *= 1.0 - totfrac_nobio
    z0h *= 1.0 - totfrac_nobio
    z0m += frac_nobio[:, 0] * (CT_KARMAN / np.log(ztmp / Z0_ICE)) ** 2
    z0h += frac_nobio[:, 0] * (CT_KARMAN / np.log(ztmp / Z0_ICE / ratio_z0m_z0h[0])) ** 2

    z0m = ztmp / np.exp(CT_KARMAN / np.sqrt(z0m))
    z0h = ztmp / np.exp(CT_KARMAN / np.sqrt(z0h))
    roughheight = ave_height * (1.0 - HEIGHT_DISPLACEMENT)
    for jv in range(1, veget.shape[1]):
        roughheight_pft[:, jv] = height[:, jv] * (1.0 - HEIGHT_DISPLACEMENT)
    return z0m, z0h, roughheight, roughheight_pft


def _expected_condveg_albedo(
    *,
    veget,
    veget_max,
    drysoil_frac,
    frac_nobio,
    totfrac_nobio,
    snow,
    snow_age,
    snow_nobio_age,
    tot_bare_soil,
    frac_snow_veg,
    frac_snow_nobio,
    alb_bg_modis=False,
    alb_bare_model=False,
    soilalb_bg=None,
    soilalb_wet=None,
    soilalb_dry=None,
    soilalb_moy=None,
    alb_leaf_vis,
    alb_leaf_nir,
    snowa_aged_vis,
    snowa_aged_nir,
    snowa_dec_vis,
    snowa_dec_nir,
    fixed_snow_albedo=1.0e20,
    undef_sechiba=1.0e20,
    tcst_snowa=10.0,
    alb_ice=np.asarray([0.60, 0.20], dtype=np.float64),
):
    npts, nvm = veget.shape
    alb_leaf = np.stack((alb_leaf_vis, alb_leaf_nir), axis=1)
    if alb_bg_modis:
        alb_bare = soilalb_bg.copy()
    elif alb_bare_model:
        alb_bare = soilalb_wet + drysoil_frac[:, None] * (soilalb_dry - soilalb_wet)
    else:
        alb_bare = soilalb_moy.copy()

    albedo = tot_bare_soil[:, None] * alb_bare
    alb_veget = np.zeros((npts, 2), dtype=np.float64)
    for jv in range(1, nvm):
        albedo += veget[:, jv, None] * alb_leaf[jv, :]
        alb_veget += veget[:, jv, None] * alb_leaf[jv, :]

    if abs(fixed_snow_albedo - undef_sechiba) > np.finfo(np.float64).eps:
        snowa_veg = np.full((npts, 2), fixed_snow_albedo, dtype=np.float64)
        snowa_nobio = np.full((npts, 1, 2), fixed_snow_albedo, dtype=np.float64)
    else:
        snowa_aged = np.stack((snowa_aged_vis, snowa_aged_nir), axis=1)
        snowa_dec = np.stack((snowa_dec_vis, snowa_dec_nir), axis=1)
        agefunc_veg = np.exp(-snow_age / tcst_snowa)
        fraction_veg = 1.0 - totfrac_nobio
        snowa_veg = np.zeros((npts, 2), dtype=np.float64)
        for jb in range(2):
            for jv in range(nvm):
                active = fraction_veg > MIN_SECHIBA
                snowa_veg[active, jb] += (
                    veget_max[active, jv]
                    / fraction_veg[active]
                    * (snowa_aged[jv, jb] + snowa_dec[jv, jb] * agefunc_veg[active])
                )
        agefunc_nobio = np.exp(-snow_nobio_age / tcst_snowa)
        snowa_nobio = np.zeros((npts, 1, 2), dtype=np.float64)
        for jb in range(2):
            snowa_nobio[:, 0, jb] = snowa_aged[0, jb] + snowa_dec[0, jb] * agefunc_nobio

    fraction_veg = 1.0 - totfrac_nobio
    out = fraction_veg[:, None] * (
        (1.0 - frac_snow_veg[:, None]) * albedo + frac_snow_veg[:, None] * snowa_veg
    )
    out += frac_nobio[:, 0, None] * (
        (1.0 - frac_snow_nobio[:, 0, None]) * alb_ice[None, :]
        + frac_snow_nobio[:, 0, None] * snowa_nobio[:, 0, :]
    )
    albedo_snow = fraction_veg[:, None] * frac_snow_veg[:, None] * snowa_veg
    albedo_snow += frac_nobio[:, 0, None] * frac_snow_nobio[:, 0, None] * snowa_nobio[:, 0, :]
    albedo_snow_mean = np.where(snow > 0.0, (albedo_snow[:, 0] + albedo_snow[:, 1]) / 2.0, 0.0)
    return out, albedo_snow, alb_bare, alb_veget, snowa_veg, snowa_nobio, albedo_snow_mean


def test_condveg_frac_snow_explicit_and_nobio_follow_fortran_formula():
    snow = np.array([2.0, 5.0])
    snow_nobio = np.array([[0.0], [10.0]])
    snowdz = np.array([[0.0, 0.0], [0.01, 0.04]])
    snowrho = np.array([[100.0, 150.0], [200.0, 400.0]])

    result = condveg_frac_snow(
        snow=snow,
        snow_nobio=snow_nobio,
        snowrho=snowrho,
        snowdz=snowdz,
        ok_explicitsnow=True,
    )

    snowdepth = snowdz.sum(axis=1)
    snowrho_ave = (snowrho * snowdz).sum(axis=1) / np.where(snowdepth < MIN_SECHIBA, 1.0, snowdepth)
    expected_veg = np.zeros_like(snowdepth)
    depth_mask = snowdepth >= MIN_SECHIBA
    expected_veg[depth_mask] = np.tanh(
        snowdepth[depth_mask] / (0.025 * (snowrho_ave[depth_mask] / 50.0))
    )
    expected_nobio = np.minimum(np.maximum(snow_nobio, 0.0) / (np.maximum(snow_nobio, 0.0) + SNOWCRI_ALB), 1.0)

    np.testing.assert_allclose(np.asarray(result.frac_snow_veg), expected_veg)
    np.testing.assert_allclose(np.asarray(result.frac_snow_nobio), expected_nobio)


def test_condveg_frac_snow_zero_depth_has_finite_zero_gradient():
    def objective(depth):
        result = condveg_frac_snow(
            snow=jnp.asarray([0.0]),
            snow_nobio=jnp.asarray([[0.0]]),
            snowrho=jnp.asarray([[100.0, 150.0]]),
            snowdz=jnp.stack((depth, depth))[None, :],
            ok_explicitsnow=True,
        )
        return result.frac_snow_veg[0]

    value = jnp.asarray(0.0, dtype=jnp.float64)
    forward = jax.jacfwd(objective)(value)
    reverse = jax.grad(objective)(value)

    assert np.asarray(forward) == pytest.approx(0.0)
    assert np.asarray(reverse) == pytest.approx(0.0)


def test_condveg_frac_snow_default_branch_uses_snow_mass_constants():
    result = condveg_frac_snow(
        snow=np.array([-1.0, 33.0]),
        snow_nobio=np.array([[5.0], [20.0]]),
        snowrho=np.ones((2, 1)),
        snowdz=np.ones((2, 1)),
        ok_explicitsnow=False,
    )

    positive = np.array([0.0, 33.0])
    expected = positive / (positive + SNOWCRI_ALB * SN_DENS / 100.0)
    np.testing.assert_allclose(np.asarray(result.frac_snow_veg), expected)


def test_condveg_first_step_wrapper_does_not_require_three_layer_snow_for_bucket_branch():
    """Fortran: condveg_frac_snow lines 879-889 branch on ok_explicitsnow."""

    restart = {
        "snow": np.asarray([33.0], dtype=np.float64),
        "snow_nobio": np.asarray([[10.0]], dtype=np.float64),
        "veget": np.asarray([[0.1, 0.3]], dtype=np.float64),
        "veget_max": np.asarray([[0.2, 0.4]], dtype=np.float64),
        "frac_nobio": np.asarray([[0.0]], dtype=np.float64),
        "height": np.asarray([[0.1, 5.0]], dtype=np.float64),
        "lai": np.asarray([[0.0, 3.0]], dtype=np.float64),
    }
    driver = {
        "zlev": np.asarray([20.0], dtype=np.float64),
        "temp_air": np.asarray([285.0], dtype=np.float64),
        "pb": np.asarray([1000.0], dtype=np.float64),
        "u": np.asarray([2.0], dtype=np.float64),
        "v": np.asarray([1.0], dtype=np.float64),
    }
    slowproc = {"totfrac_nobio": np.asarray([0.0], dtype=np.float64)}

    explicit_missing = run_condveg_first_step_module(
        restart_payload=restart,
        driver_payload=driver,
        slowproc_payload=slowproc,
        ok_explicitsnow=True,
    )
    assert explicit_missing.module is None
    assert set(explicit_missing.missing_inputs) == {"snowrho", "snowdz"}

    bucket = run_condveg_first_step_module(
        restart_payload=restart,
        driver_payload=driver,
        slowproc_payload=slowproc,
        ok_explicitsnow=False,
    )
    assert bucket.module is not None
    assert "frac_snow_veg" in bucket.covered_fields
    expected_veg = 33.0 / (33.0 + SNOWCRI_ALB * SN_DENS / 100.0)
    expected_nobio = 10.0 / (10.0 + SNOWCRI_ALB)
    np.testing.assert_allclose(np.asarray(bucket.module.frac_snow_veg), [expected_veg])
    np.testing.assert_allclose(np.asarray(bucket.module.frac_snow_nobio), [[expected_nobio]])


def test_condveg_z0cdrag_dyn_matches_fortran_algebra_on_small_arrays():
    veget = np.array([[0.25, 0.30, 0.0], [0.50, 0.20, 0.10]], dtype=np.float64)
    veget_max = np.array([[0.40, 0.40, 0.0], [0.55, 0.25, 0.15]], dtype=np.float64)
    frac_nobio = np.array([[0.20], [0.05]], dtype=np.float64)
    totfrac_nobio = frac_nobio[:, 0]
    zlev = np.array([20.0, 15.0])
    height = np.array([[0.1, 5.0, 2.0], [0.1, 8.0, 1.5]], dtype=np.float64)
    temp_air = np.array([285.0, 295.0])
    pb = np.array([1000.0, 990.0])
    u = np.array([2.0, 0.05])
    v = np.array([1.0, 0.02])
    lai = np.array([[0.0, 3.0, 0.0], [0.0, 2.5, 0.5]])
    frac_snow_veg = np.array([0.0, 0.5])

    result = condveg_z0cdrag_dyn(
        veget=veget,
        veget_max=veget_max,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        zlev=zlev,
        height=height,
        temp_air=temp_air,
        pb=pb,
        u=u,
        v=v,
        lai=lai,
        frac_snow_veg=frac_snow_veg,
    )
    expected = _expected_dyn_roughness(
        veget,
        veget_max,
        frac_nobio,
        totfrac_nobio,
        zlev,
        height,
        temp_air,
        pb,
        u,
        v,
        lai,
        frac_snow_veg,
    )

    np.testing.assert_allclose(np.asarray(result.z0m), expected[0])
    np.testing.assert_allclose(np.asarray(result.z0h), expected[1])
    np.testing.assert_allclose(np.asarray(result.roughheight), expected[2])
    np.testing.assert_allclose(np.asarray(result.roughheight_pft)[:, 1:], expected[3][:, 1:])
    assert np.isnan(np.asarray(result.roughheight_pft)[0, 0])


def test_condveg_z0cdrag_dyn_inactive_and_zero_lai_branches_have_finite_reverse_gradient():
    def objective(active_lai):
        result = condveg_z0cdrag_dyn(
            veget=jnp.asarray([[0.0, 0.4, 0.0]], dtype=jnp.float64),
            veget_max=jnp.asarray([[0.0, 0.6, 0.0]], dtype=jnp.float64),
            frac_nobio=jnp.asarray([[0.0]], dtype=jnp.float64),
            totfrac_nobio=jnp.asarray([0.0], dtype=jnp.float64),
            zlev=jnp.asarray([20.0], dtype=jnp.float64),
            height=jnp.asarray([[jnp.nan, 5.0, jnp.nan]], dtype=jnp.float64),
            temp_air=jnp.asarray([290.0], dtype=jnp.float64),
            pb=jnp.asarray([1000.0], dtype=jnp.float64),
            u=jnp.asarray([2.0], dtype=jnp.float64),
            v=jnp.asarray([1.0], dtype=jnp.float64),
            lai=jnp.asarray([[jnp.nan, active_lai, 0.0]], dtype=jnp.float64),
            frac_snow_veg=jnp.asarray([0.0], dtype=jnp.float64),
        )
        return result.z0m[0] + result.z0h[0]

    for lai in (0.0, 2.0):
        primal, reverse = jax.value_and_grad(objective)(jnp.asarray(lai, dtype=jnp.float64))
        assert np.isfinite(float(primal))
        assert np.isfinite(float(reverse))


def test_condveg_z0cdrag_static_branch_matches_fortran_algebra_on_small_arrays():
    veget = np.array([[0.25, 0.30, 0.10], [0.50, 0.20, 0.05]], dtype=np.float64)
    veget_max = np.array([[0.40, 0.40, 0.20], [0.55, 0.25, 0.15]], dtype=np.float64)
    frac_nobio = np.array([[0.20], [0.05]], dtype=np.float64)
    totfrac_nobio = frac_nobio[:, 0]
    zlev = np.array([20.0, 15.0], dtype=np.float64)
    height = np.array([[0.1, 5.0, 2.0], [0.1, 8.0, 1.5]], dtype=np.float64)
    tot_bare_soil = np.array([0.15, 0.25], dtype=np.float64)
    is_tree = np.array([False, True, False], dtype=bool)
    z0_over_height = np.array([0.0, 0.0625, 0.0625], dtype=np.float64)
    ratio_z0m_z0h = np.array([1.0, 1.0, 2.0], dtype=np.float64)

    result = condveg_z0cdrag(
        veget=veget,
        veget_max=veget_max,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        zlev=zlev,
        height=height,
        tot_bare_soil=tot_bare_soil,
        is_tree=is_tree,
        z0_over_height=z0_over_height,
        ratio_z0m_z0h=ratio_z0m_z0h,
    )
    expected = _expected_static_roughness(
        veget,
        veget_max,
        frac_nobio,
        totfrac_nobio,
        zlev,
        height,
        tot_bare_soil,
        is_tree,
        z0_over_height,
        ratio_z0m_z0h,
    )

    np.testing.assert_allclose(np.asarray(result.z0m), expected[0])
    np.testing.assert_allclose(np.asarray(result.z0h), expected[1])
    np.testing.assert_allclose(np.asarray(result.roughheight), expected[2])
    np.testing.assert_allclose(np.asarray(result.roughheight_pft)[:, 1:], expected[3][:, 1:])
    assert np.isnan(np.asarray(result.roughheight_pft)[0, 0])


def test_condveg_albedo_explicit_matches_modis_background_soil_branch():
    veget = np.array([[0.20, 0.30, 0.10], [0.10, 0.20, 0.25]], dtype=np.float64)
    veget_max = np.array([[0.30, 0.40, 0.10], [0.20, 0.30, 0.25]], dtype=np.float64)
    frac_nobio = np.array([[0.20], [0.25]], dtype=np.float64)
    totfrac_nobio = frac_nobio[:, 0]
    snow = np.array([0.0, 2.0], dtype=np.float64)
    snow_age = np.array([1.0, 5.0], dtype=np.float64)
    snow_nobio = np.array([[0.0], [1.0]], dtype=np.float64)
    snow_nobio_age = np.array([2.0, 4.0], dtype=np.float64)
    tot_bare_soil = np.array([0.15, 0.10], dtype=np.float64)
    frac_snow_veg = np.array([0.0, 0.40], dtype=np.float64)
    frac_snow_nobio = np.array([[0.0], [0.30]], dtype=np.float64)
    soilalb_bg = np.array([[0.12, 0.25], [0.15, 0.30]], dtype=np.float64)
    alb_leaf_vis = np.array([0.00, 0.04, 0.06], dtype=np.float64)
    alb_leaf_nir = np.array([0.00, 0.20, 0.24], dtype=np.float64)
    snowa_aged_vis = np.array([0.35, 0.14, 0.18], dtype=np.float64)
    snowa_aged_nir = np.array([0.35, 0.14, 0.18], dtype=np.float64)
    snowa_dec_vis = np.array([0.45, 0.10, 0.60], dtype=np.float64)
    snowa_dec_nir = np.array([0.45, 0.06, 0.52], dtype=np.float64)

    result = condveg_albedo_explicit(
        veget=veget,
        veget_max=veget_max,
        drysoil_frac=np.array([0.2, 0.8], dtype=np.float64),
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        snow=snow,
        snow_age=snow_age,
        snow_nobio=snow_nobio,
        snow_nobio_age=snow_nobio_age,
        tot_bare_soil=tot_bare_soil,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        alb_bg_modis=True,
        soilalb_bg=soilalb_bg,
        alb_leaf_vis=alb_leaf_vis,
        alb_leaf_nir=alb_leaf_nir,
        snowa_aged_vis=snowa_aged_vis,
        snowa_aged_nir=snowa_aged_nir,
        snowa_dec_vis=snowa_dec_vis,
        snowa_dec_nir=snowa_dec_nir,
    )
    expected = _expected_condveg_albedo(
        veget=veget,
        veget_max=veget_max,
        drysoil_frac=np.array([0.2, 0.8], dtype=np.float64),
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        snow=snow,
        snow_age=snow_age,
        snow_nobio_age=snow_nobio_age,
        tot_bare_soil=tot_bare_soil,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        alb_bg_modis=True,
        soilalb_bg=soilalb_bg,
        alb_leaf_vis=alb_leaf_vis,
        alb_leaf_nir=alb_leaf_nir,
        snowa_aged_vis=snowa_aged_vis,
        snowa_aged_nir=snowa_aged_nir,
        snowa_dec_vis=snowa_dec_vis,
        snowa_dec_nir=snowa_dec_nir,
    )

    np.testing.assert_allclose(np.asarray(result.albedo), expected[0])
    np.testing.assert_allclose(np.asarray(result.albedo_snow), expected[1])
    np.testing.assert_allclose(np.asarray(result.alb_bare), expected[2])
    np.testing.assert_allclose(np.asarray(result.alb_veget), expected[3])
    np.testing.assert_allclose(np.asarray(result.snowa_veg), expected[4])
    np.testing.assert_allclose(np.asarray(result.snowa_nobio), expected[5])
    np.testing.assert_allclose(np.asarray(result.albedo_snow_mean), expected[6])


def test_condveg_albedo_explicit_covers_wet_soil_and_fixed_snow_branches():
    veget = np.array([[0.20, 0.30]], dtype=np.float64)
    veget_max = np.array([[0.40, 0.40]], dtype=np.float64)
    frac_nobio = np.array([[0.20]], dtype=np.float64)
    result = condveg_albedo_explicit(
        veget=veget,
        veget_max=veget_max,
        drysoil_frac=np.array([0.5], dtype=np.float64),
        frac_nobio=frac_nobio,
        totfrac_nobio=frac_nobio[:, 0],
        snow=np.array([3.0], dtype=np.float64),
        snow_age=np.array([7.0], dtype=np.float64),
        snow_nobio=np.array([[2.0]], dtype=np.float64),
        snow_nobio_age=np.array([6.0], dtype=np.float64),
        tot_bare_soil=np.array([0.20], dtype=np.float64),
        frac_snow_veg=np.array([0.25], dtype=np.float64),
        frac_snow_nobio=np.array([[0.50]], dtype=np.float64),
        alb_bare_model=True,
        soilalb_wet=np.array([[0.10, 0.20]], dtype=np.float64),
        soilalb_dry=np.array([[0.20, 0.40]], dtype=np.float64),
        alb_leaf_vis=np.array([0.00, 0.05], dtype=np.float64),
        alb_leaf_nir=np.array([0.00, 0.25], dtype=np.float64),
        fixed_snow_albedo=0.75,
    )

    alb_bare = np.array([[0.15, 0.30]])
    base = 0.20 * alb_bare + 0.30 * np.array([[0.05, 0.25]])
    expected = 0.80 * ((1.0 - 0.25) * base + 0.25 * 0.75)
    expected += 0.20 * ((1.0 - 0.50) * np.array([[0.60, 0.20]]) + 0.50 * 0.75)
    np.testing.assert_allclose(np.asarray(result.alb_bare), alb_bare)
    np.testing.assert_allclose(np.asarray(result.snowa_veg), np.array([[0.75, 0.75]]))
    np.testing.assert_allclose(np.asarray(result.snowa_nobio), np.array([[[0.75, 0.75]]]))
    np.testing.assert_allclose(np.asarray(result.albedo), expected)


def test_condveg_albedo_explicit_impaze_uses_prescribed_two_band_albedo_then_snow_update():
    result = condveg_albedo_explicit(
        veget=np.array([[0.2, 0.3]], dtype=np.float64),
        veget_max=np.array([[0.5, 0.3]], dtype=np.float64),
        drysoil_frac=np.array([0.0], dtype=np.float64),
        frac_nobio=np.array([[0.0]], dtype=np.float64),
        totfrac_nobio=np.array([0.0], dtype=np.float64),
        snow=np.array([1.0], dtype=np.float64),
        snow_age=np.array([0.0], dtype=np.float64),
        snow_nobio=np.array([[0.0]], dtype=np.float64),
        snow_nobio_age=np.array([0.0], dtype=np.float64),
        tot_bare_soil=np.array([0.2], dtype=np.float64),
        frac_snow_veg=np.array([0.5], dtype=np.float64),
        frac_snow_nobio=np.array([[0.0]], dtype=np.float64),
        impaze=True,
        albedo_scal=np.array([0.25, 0.35], dtype=np.float64),
        fixed_snow_albedo=0.8,
    )

    expected = 0.5 * np.array([[0.25, 0.35]]) + 0.5 * 0.8
    np.testing.assert_allclose(np.asarray(result.albedo), expected)
    np.testing.assert_allclose(np.asarray(result.alb_bare), np.zeros((1, 2)))
    np.testing.assert_allclose(np.asarray(result.alb_veget), np.zeros((1, 2)))


def test_condveg_main_minimal_gates_branches_and_returns_active_fields():
    rough = condveg_prescribed_roughness(
        kjpindex=2,
        nvm=3,
        z0_scal=0.12,
        roughheight_scal=2.5,
    )
    np.testing.assert_allclose(np.asarray(rough.z0m), [0.12, 0.12])
    np.testing.assert_allclose(np.asarray(rough.roughheight_pft), np.full((2, 3), 2.5))

    with pytest.raises(ValueError, match="z0_scal"):
        condveg_main_minimal(
            snow=np.array([1.0]),
            snow_nobio=np.array([[0.0]]),
            snowrho=np.ones((1, 1)),
            snowdz=np.ones((1, 1)),
            veget=np.ones((1, 2)) * 0.5,
            veget_max=np.ones((1, 2)) * 0.5,
            frac_nobio=np.zeros((1, 1)),
            totfrac_nobio=np.zeros(1),
            zlev=np.ones(1) * 20.0,
            height=np.ones((1, 2)),
            temp_air=np.ones(1) * 285.0,
            pb=np.ones(1) * 1000.0,
            u=np.ones(1),
            v=np.ones(1),
            lai=np.ones((1, 2)),
            emis_scal=0.98,
            impaze=True,
        )

    with pytest.raises(ValueError, match="rough_dyn=False requires"):
        condveg_main_minimal(
            snow=np.array([1.0]),
            snow_nobio=np.array([[0.0]]),
            snowrho=np.ones((1, 1)),
            snowdz=np.ones((1, 1)),
            veget=np.ones((1, 2)) * 0.5,
            veget_max=np.ones((1, 2)) * 0.5,
            frac_nobio=np.zeros((1, 1)),
            totfrac_nobio=np.zeros(1),
            zlev=np.ones(1) * 20.0,
            height=np.ones((1, 2)),
            temp_air=np.ones(1) * 285.0,
            pb=np.ones(1) * 1000.0,
            u=np.ones(1),
            v=np.ones(1),
            lai=np.ones((1, 2)),
            emis_scal=0.98,
            rough_dyn=False,
        )

    static = condveg_main_minimal(
        snow=np.array([1.0]),
        snow_nobio=np.array([[0.0]]),
        snowrho=np.ones((1, 1)),
        snowdz=np.ones((1, 1)),
        veget=np.array([[0.2, 0.3]], dtype=np.float64),
        veget_max=np.array([[0.4, 0.4]], dtype=np.float64),
        frac_nobio=np.zeros((1, 1)),
        totfrac_nobio=np.zeros(1),
        zlev=np.ones(1) * 20.0,
        height=np.array([[0.1, 5.0]], dtype=np.float64),
        temp_air=np.ones(1) * 285.0,
        pb=np.ones(1) * 1000.0,
        u=np.ones(1),
        v=np.ones(1),
        lai=np.ones((1, 2)),
        emis_scal=0.98,
        rough_dyn=False,
        tot_bare_soil=np.array([0.2], dtype=np.float64),
        is_tree=np.array([False, True], dtype=bool),
        z0_over_height=np.array([0.0, 0.0625], dtype=np.float64),
        ratio_z0m_z0h=np.ones(2, dtype=np.float64),
    )
    assert static.z0m.shape == (1,)
    assert static.roughheight_pft.shape == (1, 2)


def test_condveg_main_minimal_can_return_albedo_when_explicit_state_is_supplied():
    result = condveg_main_minimal(
        snow=np.array([1.0], dtype=np.float64),
        snow_age=np.array([2.0], dtype=np.float64),
        snow_nobio=np.array([[0.0]], dtype=np.float64),
        snow_nobio_age=np.array([0.0], dtype=np.float64),
        snowrho=np.ones((1, 2), dtype=np.float64) * 200.0,
        snowdz=np.array([[0.01, 0.02]], dtype=np.float64),
        veget=np.array([[0.20, 0.30]], dtype=np.float64),
        veget_max=np.array([[0.40, 0.40]], dtype=np.float64),
        frac_nobio=np.array([[0.20]], dtype=np.float64),
        totfrac_nobio=np.array([0.20], dtype=np.float64),
        zlev=np.array([20.0], dtype=np.float64),
        height=np.array([[0.1, 5.0]], dtype=np.float64),
        temp_air=np.array([285.0], dtype=np.float64),
        pb=np.array([1000.0], dtype=np.float64),
        u=np.array([2.0], dtype=np.float64),
        v=np.array([1.0], dtype=np.float64),
        lai=np.array([[0.0, 3.0]], dtype=np.float64),
        emis_scal=0.98,
        drysoil_frac=np.array([0.4], dtype=np.float64),
        tot_bare_soil=np.array([0.20], dtype=np.float64),
        alb_bg_modis=True,
        soilalb_bg=np.array([[0.12, 0.25]], dtype=np.float64),
        alb_leaf_vis=np.array([0.00, 0.04], dtype=np.float64),
        alb_leaf_nir=np.array([0.00, 0.20], dtype=np.float64),
        snowa_aged_vis=np.array([0.35, 0.14], dtype=np.float64),
        snowa_aged_nir=np.array([0.35, 0.14], dtype=np.float64),
        snowa_dec_vis=np.array([0.45, 0.10], dtype=np.float64),
        snowa_dec_nir=np.array([0.45, 0.06], dtype=np.float64),
    )

    assert result.albedo is not None
    assert result.albedo_snow is not None
    assert result.alb_bare is not None
    assert result.alb_veget is not None
    assert result.albedo_snow_mean is not None
    np.testing.assert_allclose(np.asarray(result.emis), [0.98])


def test_condveg_main_minimal_coverage_reports_albedo_closed_with_explicit_inputs():
    coverage = condveg_main_minimal_coverage(
        impaze=False,
        rough_dyn=True,
        has_albedo_inputs=True,
    )

    assert coverage.ok
    assert coverage.branch == "dynamic_roughness"
    assert "z0m" in coverage.covered_fields
    assert "albedo" in coverage.covered_fields
    assert "albedo" not in coverage.missing_fields
    assert "condveg_albedo explicit constants/state" not in coverage.missing_inputs
    assert any("condveg_main lines 332-480" in item for item in coverage.provenance)
    assert any("condveg_z0cdrag_dyn lines 1519-1720" in item for item in coverage.provenance)

    static = condveg_main_minimal_coverage(
        impaze=False,
        rough_dyn=False,
        has_static_roughness_inputs=True,
        has_albedo_inputs=True,
    )
    assert static.ok
    assert static.branch == "static_roughness"
    assert "condveg_z0cdrag explicit PFT parameters" not in static.missing_inputs
    assert any("condveg_z0cdrag lines 1288-1474" in item for item in static.provenance)


def test_condveg_albedo_snow_mean_matches_diagnostic_loop():
    mean = condveg_albedo_snow_mean(
        snow=np.array([0.0, 2.0]),
        albedo_snow=np.array([[0.8, 0.6], [0.7, 0.3]]),
    )

    np.testing.assert_allclose(np.asarray(mean), [0.0, 0.5])
