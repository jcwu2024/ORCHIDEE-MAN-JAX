from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.carbon_kernels import (
    IABOVE,
    IACT,
    IACTIVE,
    IBELOW,
    ICARBON,
    ILEAF,
    IPASSIVE,
    IPAS,
    IROOT,
    ISLO,
    ISLOW,
    NPARTS,
    NCARB,
    NPOOL,
    IMETABO,
    IMETABOLIC,
    IMETBEL,
    ISTRABO,
    ISTRBEL,
    ISTRUCTURAL,
)
from jax_orchidee.stomate.soilcarbon_kernels import (
    ICO2AQ,
    IDOCL,
    IDOCR,
    IDRAINAGE,
    IFLOODED,
    IRUNOFF,
    NEXP,
    _sqrt_with_finite_zero_tangent,
    altcalc_doc,
    deep_carbon_altcalc_step,
    deep_carbon_cryoturbation_coefficients,
    deep_carbon_cryoturbation_cycle,
    deep_carbon_cryoturbation_diffuse,
    deep_carbon_ebullition_step,
    deep_carbon_gasdiff_properties_step,
    deep_carbon_input_step,
    deep_carbon_nonmethane_core_step,
    deep_carbon_permafrost_decomp_oxic_step,
    deep_carbon_plant_transport_step,
    deep_carbon_root_depth_step,
    deep_carbon_snow_interpol_step,
    deep_carbon_snowlevels_step,
    deep_carbon_soil_gasdiff_coefficients,
    deep_carbon_soil_gasdiff_diffuse,
    deep_carbon_vertical_integral_step,
    deep_carbon_yedoma_reset_step,
    soilcarbon_leak_doc_diffusion,
    soilcarbon_leak_doc_export,
    soilcarbon_leak_doc_export_aggregate,
    soilcarbon_leak_doc_inputs,
    soilcarbon_leak_core_step,
    soilcarbon_leak_tf_doc_ground_fluxes,
    soilcarbon_leak_tf_doc_inputs,
    soilcarbon_leak_water_transport,
    soilcarbon_cryoturbation_cycle,
    soilcarbon_cryoturbation_coefficients,
    soilcarbon_cryoturbation_diffuse,
    soilcarbon_perma_peat_cmax,
    soilcarbon_perma_peat_redistribute,
)


PFT14 = 13
IFREE = 0


def test_soilcarbon_fastr_sqrt_preserves_values_with_finite_zero_tangent():
    values = jnp.asarray([0.0, 4.0, -1.0], dtype=jnp.float64)
    result = _sqrt_with_finite_zero_tangent(values)
    compiled = jax.jit(_sqrt_with_finite_zero_tangent)(values)

    np.testing.assert_array_equal(np.asarray(result[:2]), [0.0, 2.0])
    np.testing.assert_array_equal(
        np.asarray(compiled[:2]), np.asarray(jnp.sqrt(values[:2]))
    )
    assert np.isnan(np.asarray(result[2]))
    assert float(jax.grad(lambda value: _sqrt_with_finite_zero_tangent(value))(0.0)) == 0.0
    assert float(jax.grad(lambda value: _sqrt_with_finite_zero_tangent(value))(4.0)) == 0.25


def test_altcalc_doc_firstcall_initializes_altmax_indices_from_depth():
    altmax = np.asarray([[0.0, 0.25, 0.75]], dtype=np.float64)
    altmax_ind = np.zeros((1, 3), dtype=np.int32)
    zprof = np.asarray([0.1, 0.5, 1.0], dtype=np.float64)
    veget_mask = np.asarray([[False, True, True]])

    result = altcalc_doc(
        tprof=np.zeros((1, 3, 3), dtype=np.float64),
        zprof=zprof,
        altmax=altmax,
        altmax_ind=altmax_ind,
        altmax_lastyear=np.zeros_like(altmax),
        altmax_ind_lastyear=np.zeros_like(altmax_ind),
        veget_mask=veget_mask,
        firstcall=True,
    )

    assert np.allclose(np.asarray(result.altmax_ind), [[0, 1, 2]])
    assert np.allclose(np.asarray(result.altmax_lastyear), altmax)
    assert np.allclose(np.asarray(result.alt), 0.0)


def test_altcalc_doc_standard_branch_finds_first_frozen_layer_and_updates_altmax():
    tprof = np.asarray(
        [
            [
                [272.0, 274.0],
                [271.0, 275.0],
                [270.0, 272.0],
            ]
        ],
        dtype=np.float64,
    )
    zprof = np.asarray([0.1, 0.5, 1.0], dtype=np.float64)
    altmax = np.asarray([[0.2, 0.2]], dtype=np.float64)
    altmax_ind = np.asarray([[1, 1]], dtype=np.int32)

    result = altcalc_doc(
        tprof=tprof,
        zprof=zprof,
        altmax=altmax,
        altmax_ind=altmax_ind,
        altmax_lastyear=np.zeros_like(altmax),
        altmax_ind_lastyear=np.zeros_like(altmax_ind),
        veget_mask=np.ones((1, 2), dtype=bool),
        firstcall=False,
        dayno=1,
    )

    assert np.allclose(np.asarray(result.alt), [[0.0, 0.5]])
    assert np.allclose(np.asarray(result.alt_ind), [[0, 2]])
    assert np.allclose(np.asarray(result.altmax), [[0.2, 0.5]])
    assert np.allclose(np.asarray(result.altmax_ind), [[1, 2]])


def test_deep_carbon_altcalc_wrapper_uses_same_active_layer_source_logic():
    """Fortran: stomate_permafrost_soilcarbon.f90 altcalc lines 1271-1419."""

    tprof = np.asarray([[[274.0, 270.0], [275.0, 271.0], [272.0, 276.0]]], dtype=np.float64)
    zprof = np.asarray([0.1, 0.5, 1.0], dtype=np.float64)
    altmax = np.asarray([[0.2, 0.2]], dtype=np.float64)
    altmax_ind = np.asarray([[1, 1]], dtype=np.int32)
    kwargs = dict(
        tprof=tprof,
        zprof=zprof,
        altmax=altmax,
        altmax_ind=altmax_ind,
        altmax_lastyear=np.zeros_like(altmax),
        altmax_ind_lastyear=np.zeros_like(altmax_ind),
        veget_mask=np.ones((1, 2), dtype=bool),
        firstcall=False,
        dayno=1,
    )

    expected = altcalc_doc(**kwargs)
    actual = deep_carbon_altcalc_step(**kwargs)

    assert np.allclose(np.asarray(actual.alt), np.asarray(expected.alt))
    assert np.allclose(np.asarray(actual.alt_ind), np.asarray(expected.alt_ind))
    assert np.allclose(np.asarray(actual.altmax), np.asarray(expected.altmax))


def test_deep_carbon_root_depth_caps_to_previous_active_layer():
    """Fortran: stomate_permafrost_soilcarbon.f90 deep_carbcycle lines 945-951."""

    altmax_lastyear = np.asarray([[0.4, 3.0, 1.0]], dtype=np.float64)
    altmax_ind_lastyear = np.asarray([[1, 4, 2]], dtype=np.int32)
    veget_mask = np.asarray([[True, True, False]])

    result = deep_carbon_root_depth_step(
        altmax_lastyear,
        altmax_ind_lastyear,
        veget_mask,
        z_root_max=2.0,
    )

    assert np.allclose(np.asarray(result.z_root), [[0.4, 2.0, 0.0]])
    assert np.allclose(np.asarray(result.rootlev), [[1, 4, 0]])


def _deep_cryo_base_arrays(ndeep=6):
    npts, nvm = 1, 3
    layers = np.arange(ndeep, dtype=np.float64)
    deep_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.zeros_like(deep_a)
    deep_p = np.zeros_like(deep_a)
    deep_a[0, :, 1] = 10.0 + 2.0 * layers
    deep_s[0, :, 1] = 20.0 + 3.0 * layers
    deep_p[0, :, 1] = 30.0 + 4.0 * layers
    return deep_a, deep_s, deep_p


def test_deep_carbon_cryoturbation_coefficients_old_scheme_and_bioturbation():
    """Fortran: stomate_permafrost_soilcarbon.f90 cryoturbate lines 3639-3792."""

    deep_a, deep_s, deep_p = _deep_cryo_base_arrays()
    altmax_lastyear = np.asarray([[0.2, 1.0, 4.0]], dtype=np.float64)
    coeff = deep_carbon_cryoturbation_coefficients(
        altmax_ind=np.asarray([[1, 2, 4]], dtype=np.int32),
        deepC_a=deep_a,
        deepC_s=deep_s,
        deepC_p=deep_p,
        altmax_lastyear=altmax_lastyear,
        fixed_cryoturbation_depth=altmax_lastyear,
        veget_mask=np.asarray([[True, True, True]]),
        zi_soil=np.asarray([0.1, 0.3, 0.6, 1.0, 1.5, 2.1], dtype=np.float64),
        zf_soil=np.asarray([0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.7], dtype=np.float64),
        dt_seconds=10.0,
        diff_k_const=100.0,
        bio_diff_k_const=9.0,
    )

    diff_k = np.asarray(coeff.diff_k)
    assert np.allclose(diff_k[0, :, 1], [100.0, 100.0, 10.0, 1.0, 0.0, 0.0])
    assert np.allclose(diff_k[0, :, 2], [9.0, 9.0, 9.0, 9.0, 9.0, 0.0])
    assert np.asarray(coeff.cryoturb_location)[0, 1]
    assert np.asarray(coeff.bioturb_location)[0, 2]


def test_deep_carbon_cryoturbation_new_method_recurrence_matches_source():
    """Fortran: stomate_permafrost_soilcarbon.f90 cryoturbate lines 3720-3898."""

    deep_a, deep_s, deep_p = _deep_cryo_base_arrays(ndeep=6)
    zi = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    coeff = deep_carbon_cryoturbation_coefficients(
        altmax_ind=np.asarray([[0, 2, 0]], dtype=np.int32),
        deepC_a=deep_a,
        deepC_s=deep_s,
        deepC_p=deep_p,
        altmax_lastyear=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        fixed_cryoturbation_depth=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        veget_mask=np.asarray([[False, True, False]]),
        zi_soil=zi,
        zf_soil=zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
        use_new_cryoturbation=True,
        cryoturbation_method=4,
    )

    assert np.allclose(np.asarray(coeff.diff_k)[0, :, 1], [1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    alpha = np.zeros(6, dtype=np.float64)
    beta = np.zeros(6, dtype=np.float64)
    xc = np.asarray(coeff.xc_cryoturb)[0, :, 1]
    xd = np.asarray(coeff.xd_cryoturb)[0, :, 1]
    xe = xc[5] + xd[4]
    alpha[4] = xd[4] / xe
    beta[4] = xc[5] * deep_a[0, 5, 1] / xe
    for layer in range(3, -1, -1):
        xe = xc[layer + 1] + (1.0 - alpha[layer + 1]) * xd[layer + 1] + xd[layer]
        alpha[layer] = xd[layer] / xe
        beta[layer] = (xc[layer + 1] * deep_a[0, layer + 1, 1] + xd[layer + 1] * beta[layer + 1]) / xe
    assert np.allclose(np.asarray(coeff.alpha_a)[0, :, 1], alpha)
    assert np.allclose(np.asarray(coeff.beta_a)[0, :, 1], beta)


def test_deep_carbon_cryoturbation_diffuse_applies_mass_correction():
    """Fortran: stomate_permafrost_soilcarbon.f90 cryoturbate lines 3546-3616."""

    deep_a, deep_s, deep_p = _deep_cryo_base_arrays(ndeep=6)
    zi = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    altmax_ind = np.asarray([[0, 2, 0]], dtype=np.int32)
    coeff = deep_carbon_cryoturbation_coefficients(
        altmax_ind=altmax_ind,
        deepC_a=deep_a,
        deepC_s=deep_s,
        deepC_p=deep_p,
        altmax_lastyear=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        fixed_cryoturbation_depth=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        veget_mask=np.asarray([[False, True, False]]),
        zi_soil=zi,
        zf_soil=zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
    )

    result = deep_carbon_cryoturbation_diffuse(deep_a, deep_s, deep_p, coeff, altmax_ind, zi, zf)
    thickness = np.diff(zf)
    old_total = np.sum(deep_a[0, :, 1] * thickness)
    new_total = np.sum(np.asarray(result.deepC_a)[0, :, 1] * thickness)
    assert np.allclose(new_total, old_total)
    assert not np.allclose(np.asarray(result.deepC_a)[0, :, 1], deep_a[0, :, 1])
    assert np.allclose(np.asarray(result.deepC_a)[0, :, 0], deep_a[0, :, 0])


def test_deep_carbon_cryoturbation_diffuse_negative_fallback_rescales_profile():
    """Fortran: stomate_permafrost_soilcarbon.f90 cryoturbate lines 3617-3628."""

    deep_a, deep_s, deep_p = _deep_cryo_base_arrays(ndeep=4)
    zi = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    altmax_ind = np.asarray([[0, 2, 0]], dtype=np.int32)
    coeff = deep_carbon_cryoturbation_coefficients(
        altmax_ind=altmax_ind,
        deepC_a=deep_a,
        deepC_s=deep_s,
        deepC_p=deep_p,
        altmax_lastyear=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        fixed_cryoturbation_depth=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        veget_mask=np.asarray([[False, True, False]]),
        zi_soil=zi,
        zf_soil=zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
    )
    coeff = coeff._replace(beta_a=coeff.beta_a.at[0, 0, 1].set(1000.0))

    result = deep_carbon_cryoturbation_diffuse(deep_a, deep_s, deep_p, coeff, altmax_ind, zi, zf)

    profile = np.asarray(result.deepC_a)[0, :, 1]
    assert np.all(profile >= 0.0)
    assert np.allclose(np.sum(profile * np.diff(zf)), np.sum(deep_a[0, :, 1] * np.diff(zf)))


def test_deep_carbon_cryoturbation_cycle_recomputes_coefficients_after_optional_diffuse():
    """Fortran: stomate_permafrost_soilcarbon.f90 deep_carbcycle lines 954 and 1060."""

    deep_a, deep_s, deep_p = _deep_cryo_base_arrays(ndeep=6)
    zi = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    altmax_lastyear = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)
    altmax_ind = np.asarray([[0, 2, 0]], dtype=np.int32)
    first_diffuse, first_coeff = deep_carbon_cryoturbation_cycle(
        deep_a,
        deep_s,
        deep_p,
        None,
        altmax_ind,
        altmax_lastyear,
        altmax_lastyear,
        np.asarray([[False, True, False]]),
        zi,
        zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
    )
    second_diffuse, second_coeff = deep_carbon_cryoturbation_cycle(
        deep_a,
        deep_s,
        deep_p,
        first_coeff,
        altmax_ind,
        altmax_lastyear,
        altmax_lastyear,
        np.asarray([[False, True, False]]),
        zi,
        zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
    )
    explicit_diffuse = deep_carbon_cryoturbation_diffuse(deep_a, deep_s, deep_p, first_coeff, altmax_ind, zi, zf)
    explicit_coeff = deep_carbon_cryoturbation_coefficients(
        altmax_ind,
        explicit_diffuse.deepC_a,
        explicit_diffuse.deepC_s,
        explicit_diffuse.deepC_p,
        altmax_lastyear,
        altmax_lastyear,
        np.asarray([[False, True, False]]),
        zi,
        zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
    )

    assert np.allclose(np.asarray(first_diffuse.deepC_a), deep_a)
    assert np.allclose(np.asarray(second_diffuse.deepC_a), np.asarray(explicit_diffuse.deepC_a))
    assert np.allclose(np.asarray(second_coeff.beta_a), np.asarray(explicit_coeff.beta_a))


def test_deep_carbon_snowlevels_copies_snowdz_to_each_pft():
    """Fortran: stomate_permafrost_soilcarbon.f90 snowlevels lines 2758-2789."""

    snowdz = np.asarray([[0.1, 0.2, 0.4]], dtype=np.float64)
    veget_max = np.asarray([[0.0, 0.3]], dtype=np.float64)

    result = deep_carbon_snowlevels_step(snowdz, veget_max)

    assert np.allclose(np.asarray(result.zf_snow)[0, :, 1], [0.0, 0.1, 0.3, 0.7])
    assert np.allclose(np.asarray(result.zi_snow)[0, :, 1], [0.05, 0.2, 0.5])
    assert np.allclose(np.asarray(result.zf_snow)[0, :, 0], [0.0, 0.1, 0.3, 0.7])


def test_deep_carbon_snow_interpol_uses_old_neighbour_layers():
    """Fortran: stomate_permafrost_soilcarbon.f90 snow_interpol lines 2854-2944."""

    snow_o2 = np.asarray([[[10.0, 0.0], [20.0, 0.0], [40.0, 0.0]]], dtype=np.float64)
    snow_ch4 = snow_o2 * 0.1
    old_zi = np.asarray([[[0.1, 0.0], [0.3, 0.0], [0.7, 0.0]]], dtype=np.float64)
    old_zf = np.asarray([[[0.0, 0.0], [0.2, 0.0], [0.5, 0.0], [0.9, 0.0]]], dtype=np.float64)
    snowdz = np.asarray([[0.2, 0.2, 0.2]], dtype=np.float64)
    veget_max = np.asarray([[0.0, 1.0]], dtype=np.float64)
    mask = np.asarray([[True, False]])

    result = deep_carbon_snow_interpol_step(snow_o2, snow_ch4, old_zi, old_zf, veget_max, snowdz, mask)

    assert np.allclose(np.asarray(result.zi_snow)[0, :, 0], [0.1, 0.3, 0.5])
    assert np.allclose(np.asarray(result.snowO2)[0, :, 0], [10.0, 20.0, 30.0])
    assert np.allclose(np.asarray(result.snowCH4)[0, :, 0], [1.0, 2.0, 3.0])
    assert np.allclose(np.asarray(result.snowO2)[0, :, 1], 0.0)


def test_deep_carbon_gasdiff_properties_matches_source_porosity_formula():
    """Fortran: stomate_permafrost_soilcarbon.f90 get_gasdiff lines 2167-2217."""

    hslong = np.asarray([[[0.2, 0.4], [0.8, 0.6]]], dtype=np.float64)
    snowrho = np.asarray([[100.0, 200.0]], dtype=np.float64)
    mask = np.asarray([[True, False]])

    result = deep_carbon_gasdiff_properties_step(hslong, snowrho, mask)

    por_snow = 1.0 - 100.0 / 920.0
    assert np.allclose(np.asarray(result.airvol_snow)[0, 0, 0], por_snow)
    airvol = 0.5 * (1.0 - 0.2)
    assert np.allclose(np.asarray(result.airvol_soil)[0, 0, 0], airvol)
    assert np.allclose(np.asarray(result.totporO2_soil)[0, 0, 0], airvol + 0.5 * 0.038 * 0.2)
    assert np.allclose(np.asarray(result.diffO2_soil)[0, 0, 0], (1.596e-5 * airvol + 1.596e-9 * 0.038 * 0.2 * 0.5) * (2.0 / 3.0))
    assert np.allclose(np.asarray(result.airvol_soil)[0, :, 1], np.finfo(np.float32).eps)


def test_deep_carbon_soil_gasdiff_no_snow_coefficients_and_diffuse():
    """Fortran: stomate_permafrost_soilcarbon.f90 soil_gasdiff_coeff/diff lines 1769-2089."""

    npts, nsnow, ndeep, nvm = 1, 2, 3, 2
    snow = np.zeros((npts, nsnow, nvm), dtype=np.float64)
    soil_o2 = np.asarray([[[10.0, 0.0], [20.0, 0.0], [40.0, 0.0]]], dtype=np.float64)
    soil_ch4 = soil_o2 * 0.1
    diff_soil = np.ones_like(soil_o2)
    totpor_soil = np.ones_like(soil_o2)
    zi = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    zi_snow = np.ones((npts, nsnow, nvm), dtype=np.float64)
    zf_snow = np.ones((npts, nsnow + 1, nvm), dtype=np.float64)
    mask = np.asarray([[True, False]])
    coeff = deep_carbon_soil_gasdiff_coefficients(
        snow,
        snow,
        snow + 1.0,
        snow + 2.0,
        snow + 1.0,
        snow + 1.0,
        soil_o2,
        soil_ch4,
        diff_soil,
        diff_soil * 2.0,
        totpor_soil,
        totpor_soil,
        zi_snow,
        zf_snow,
        zi,
        zf,
        mask,
        np.zeros((npts, nvm), dtype=np.float64),
        dt_seconds=1.0,
    )
    result = deep_carbon_soil_gasdiff_diffuse(
        snow,
        snow,
        soil_o2,
        soil_ch4,
        coeff,
        psol=np.asarray([101325.0], dtype=np.float64),
        tsurf=np.asarray([300.0], dtype=np.float64),
        veget_mask=mask,
    )

    o2sa = 101325.0 / (8.314 * 300.0) * 0.209 * 32.0
    expected_top = (o2sa + coeff.mu_soil * np.asarray(coeff.betaO2_soil)[0, 0, 0]) / (
        1.0 + coeff.mu_soil * (1.0 - np.asarray(coeff.alphaO2_soil)[0, 0, 0])
    )
    assert np.allclose(np.asarray(result.O2_snow)[0, 0, 0], o2sa)
    assert np.allclose(np.asarray(result.O2_soil)[0, 0, 0], expected_top)
    assert np.allclose(np.asarray(result.O2_soil)[0, 1, 0], np.asarray(coeff.alphaO2_soil)[0, 0, 0] * expected_top + np.asarray(coeff.betaO2_soil)[0, 0, 0])
    assert np.allclose(np.asarray(result.O2_soil)[0, :, 1], soil_o2[0, :, 1])


def test_deep_carbon_plant_transport_updates_gases_and_flux():
    """Fortran: stomate_permafrost_soilcarbon.f90 traMplan lines 2307-2356."""

    ch4 = np.ones((1, 3, 2), dtype=np.float64) * 10.0
    o2 = np.ones_like(ch4) * 100.0
    totpor = np.ones_like(ch4)
    zroot = np.asarray([[1.5, 0.0]], dtype=np.float64)
    rootlev = np.asarray([[2, 0]], dtype=np.int32)
    tref = np.zeros((1, 2), dtype=np.float64)
    hslong = np.ones_like(ch4) * 0.5
    zi = np.asarray([0.5, 1.0, 2.0], dtype=np.float64)
    zf = np.asarray([0.0, 0.75, 1.5, 2.5], dtype=np.float64)
    tprof = np.ones_like(ch4) * 288.15
    mask = np.asarray([[True, False]])

    result = deep_carbon_plant_transport_step(
        ch4,
        o2,
        totpor,
        totpor,
        zroot,
        rootlev,
        tref,
        hslong,
        zi,
        zf,
        tprof,
        mask,
        time_step_seconds=10.0,
        Tgr=0.0,
        refdep=0.75,
    )

    assert np.asarray(result.flupmt)[0, 0] > 0.0
    assert np.asarray(result.CH4_soil)[0, 0, 0] < ch4[0, 0, 0]
    assert np.asarray(result.O2_soil)[0, 0, 0] < o2[0, 0, 0]
    assert np.allclose(np.asarray(result.CH4_soil)[0, :, 1], ch4[0, :, 1])


def test_deep_carbon_ebullition_removes_above_threshold_ch4():
    """Fortran: stomate_permafrost_soilcarbon.f90 ebullition lines 2418-2444."""

    ch4 = np.ones((1, 2, 2), dtype=np.float64)
    totpor = np.ones_like(ch4) * 0.5
    hslong = np.ones_like(ch4)
    zi = np.asarray([0.5, 1.5], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    mask = np.asarray([[True, False]])

    result = deep_carbon_ebullition_step(ch4, totpor, hslong, zi, zf, mask, time_step_seconds=10.0)

    threshold = 12.0e-3 / 0.043
    assert np.allclose(np.asarray(result.CH4_soil)[0, :, 0], threshold)
    assert np.allclose(np.asarray(result.febul)[0, 0], 2.0 * (1.0 - threshold) * 0.5 / 10.0)
    assert np.allclose(np.asarray(result.CH4_soil)[0, :, 1], 1.0)


def test_soilcarbon_leak_tf_doc_inputs_follow_ok_tf_doc_and_tree_logic():
    precip2ground = np.ones((1, 14), dtype=np.float64) * 2.0
    precip2canopy = np.ones((1, 14), dtype=np.float64) * 3.0
    veget_max = np.zeros((1, 14), dtype=np.float64)
    veget_max[0, PFT14] = 0.5
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True

    off = soilcarbon_leak_tf_doc_inputs(
        precip2ground,
        precip2canopy,
        veget_max,
        biomass,
        is_tree,
        ok_tf_doc=False,
        dt_days=0.5,
    )
    on = soilcarbon_leak_tf_doc_inputs(
        precip2ground,
        precip2canopy,
        veget_max,
        biomass,
        is_tree,
        ok_tf_doc=True,
        dt_days=0.5,
    )

    assert np.allclose(np.asarray(off.doc_precip2ground), 0.0)
    assert np.allclose(np.asarray(on.doc_precip2ground)[0, PFT14, ICARBON], 2.0 * 3.02e-3)
    assert np.allclose(np.asarray(on.doc_precip2canopy)[0, PFT14, ICARBON], 3.0 * 3.02e-3)
    assert np.allclose(np.asarray(on.dry_dep_canopy)[0, PFT14, ICARBON], 0.00092 * 0.5 * 100.0 * 0.5)
    assert np.allclose(np.asarray(on.bio_frac), [0.5])
    assert np.allclose(np.asarray(on.doc_precip2ground)[0, 0, ICARBON], 0.0)


def test_tf_doc_nonzero_deposition_updates_storage_and_free_doc_pools():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1212-1233 and 1495-1557."""

    npts, nvm, ndeep, nelements = 1, 14, 3, 1
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.5
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[PFT14] = True
    dt_days = 0.5

    tf_inputs = soilcarbon_leak_tf_doc_inputs(
        np.full((npts, nvm), 2.0, dtype=np.float64),
        np.full((npts, nvm), 3.0, dtype=np.float64),
        veget_max,
        biomass,
        is_tree,
        ok_tf_doc=True,
        dt_days=dt_days,
    )
    storage0 = np.zeros((npts, nvm, nelements), dtype=np.float64)
    storage0[0, PFT14, ICARBON] = 0.25
    canopy2ground = np.zeros((npts, nvm), dtype=np.float64)
    canopy2ground[0, PFT14] = 1.5
    flood_frac = np.asarray([0.2], dtype=np.float64)

    ground = soilcarbon_leak_tf_doc_ground_fluxes(
        tf_inputs.doc_precip2ground,
        tf_inputs.doc_precip2canopy,
        tf_inputs.dry_dep_canopy,
        storage0,
        canopy2ground,
        veget_max,
        flood_frac,
    )
    doc = soilcarbon_leak_doc_inputs(
        np.zeros((npts, nvm, ndeep, 2, NPOOL, nelements), dtype=np.float64),
        np.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=np.float64),
        np.zeros((npts, 4), dtype=np.float64),
        np.zeros((npts, 4), dtype=np.float64),
        ground.wet_dep_ground,
        veget_max,
        bio_frac=tf_inputs.bio_frac,
        dt_days=dt_days,
        nslm=2,
    )

    doc_precip_ground = 2.0 * 3.02e-3
    doc_precip_canopy = 3.0 * 3.02e-3
    dry_dep = 0.00092 * dt_days * 100.0 * 0.5
    canopy_to_ground = min(1.5 * 100.0e-3, 0.25)
    wet = doc_precip_ground + canopy_to_ground
    wet_ground = wet * (1.0 - flood_frac[0])
    wet_flood = wet * flood_frac[0]
    doc_pool_increment = 0.5 * wet_ground / veget_max[0, PFT14]

    np.testing.assert_allclose(np.asarray(ground.doc_canopy2ground)[0, PFT14, ICARBON], canopy_to_ground)
    np.testing.assert_allclose(np.asarray(ground.wet_dep_ground)[0, PFT14, ICARBON], wet_ground)
    np.testing.assert_allclose(np.asarray(ground.wet_dep_flood)[0, PFT14, ICARBON], wet_flood)
    np.testing.assert_allclose(
        np.asarray(ground.interception_storage)[0, PFT14, ICARBON],
        0.25 + dry_dep + doc_precip_canopy - canopy_to_ground,
    )
    np.testing.assert_allclose(np.asarray(doc)[0, PFT14, 0, IFREE, IACT, ICARBON], doc_pool_increment)
    np.testing.assert_allclose(np.asarray(doc)[0, PFT14, 0, IFREE, ISLO, ICARBON], doc_pool_increment)
    np.testing.assert_allclose(np.asarray(doc)[0, 0, :, IFREE, :, ICARBON], 0.0)


def _cryoturb_base_arrays(ndeep=6):
    npts, nvm, nelements = 1, 3, 1
    carbon = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    doc = np.zeros((npts, nvm, ndeep, 2, NPOOL, nelements), dtype=np.float64)
    litter = np.zeros((npts, 2, nvm, ndeep, nelements), dtype=np.float64)
    layers = np.arange(ndeep, dtype=np.float64)
    carbon[0, :, 1, :] = np.asarray([10.0, 11.0, 12.0])[:, None] + 10.0 * layers[None, :]
    doc[0, 1, :, IFREE, IACT, ICARBON] = np.arange(1, ndeep + 1, dtype=np.float64)
    litter[0, :, 1, :, ICARBON] = np.asarray([2.0, 3.0])[:, None] + 2.0 * layers[None, :]
    return carbon, doc, litter


def test_soilcarbon_cryoturbation_coefficients_old_scheme_profiles():
    """Fortran: stomate_soilcarbon.f90 cryoturbate_doc_POC lines 3123-3225."""

    carbon, doc, litter = _cryoturb_base_arrays()
    altmax_lastyear = np.asarray([[0.2, 1.0, 1.5]], dtype=np.float64)
    veget = np.asarray([[True, True, True]])
    coeff = soilcarbon_cryoturbation_coefficients(
        altmax_ind=np.asarray([[1, 2, 4]], dtype=np.int32),
        carbon_32l=carbon,
        doc=doc,
        litter_below=litter,
        altmax_lastyear=altmax_lastyear,
        fixed_cryoturbation_depth=altmax_lastyear,
        veget_mask=veget,
        zi_soil=np.asarray([0.1, 0.3, 0.6, 1.0, 1.5, 2.1], dtype=np.float64),
        zf_soil_b=np.asarray([0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.7], dtype=np.float64),
        dt_seconds=10.0,
        diff_k_const=100.0,
        bio_diff_k_const=9.0,
    )

    diff_k = np.asarray(coeff.diff_k)
    assert np.allclose(diff_k[0, :, 1], [100.0, 100.0, 10.0, 1.0, 0.0, 0.0])
    assert np.allclose(diff_k[0, :, 2], [100.0, 10.0, 1.0, 0.0, 0.0, 0.0])
    assert np.all(np.asarray(coeff.cryoturb_location)[0])
    assert not np.any(np.asarray(coeff.bioturb_location))


def test_soilcarbon_cryoturbation_coefficients_new_methods_and_bioturbation():
    """Fortran: stomate_soilcarbon.f90 cryoturbate_doc_POC lines 3143-3225."""

    carbon, doc, litter = _cryoturb_base_arrays()
    zi = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.5], dtype=np.float64)
    zf = np.asarray([0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.7], dtype=np.float64)
    common = dict(
        altmax_ind=np.asarray([[2, 2, 2]], dtype=np.int32),
        carbon_32l=carbon,
        doc=doc,
        litter_below=litter,
        fixed_cryoturbation_depth=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
        veget_mask=np.asarray([[True, True, True]]),
        zi_soil=zi,
        zf_soil_b=zf,
        dt_seconds=10.0,
        diff_k_const=10.0,
        bio_diff_k_const=2.0,
        use_new_cryoturbation=True,
        use_fixed_cryoturbation_depth=True,
    )

    method1 = soilcarbon_cryoturbation_coefficients(
        altmax_lastyear=np.asarray([[1.0, 4.0, 0.0]], dtype=np.float64),
        cryoturbation_method=1,
        **common,
    )
    assert np.allclose(np.asarray(method1.diff_k)[0, :, 0], [10.0, 10.0, 5.0, 0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(method1.diff_k)[0, :, 1], [2.0, 2.0, 2.0, 2.0, 0.0, 0.0])
    assert not np.any(np.asarray(method1.diff_k)[0, :, 2])
    assert np.asarray(method1.bioturb_location)[0, 1]

    method4 = soilcarbon_cryoturbation_coefficients(
        altmax_lastyear=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        cryoturbation_method=4,
        **common,
    )
    assert np.allclose(np.asarray(method4.diff_k)[0, :, 0], [10.0, 10.0, 7.5, 5.0, 2.5, 0.0])

    method5 = soilcarbon_cryoturbation_coefficients(
        altmax_lastyear=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        cryoturbation_method=5,
        **common,
    )
    assert np.allclose(np.asarray(method5.diff_k)[0, :, 0], [10.0, 10.0, 7.5, 5.0, 2.5, 0.0])


def test_soilcarbon_cryoturbation_coefficients_alpha_beta_recurrence_and_diffuse():
    """Fortran: stomate_soilcarbon.f90 cryoturbate_doc_POC lines 3238-3329 and 2938-2975."""

    carbon, doc, litter = _cryoturb_base_arrays(ndeep=6)
    altmax_lastyear = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)
    zi = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    coeff = soilcarbon_cryoturbation_coefficients(
        altmax_ind=np.asarray([[0, 2, 0]], dtype=np.int32),
        carbon_32l=carbon,
        doc=doc,
        litter_below=litter,
        altmax_lastyear=altmax_lastyear,
        fixed_cryoturbation_depth=altmax_lastyear,
        veget_mask=np.asarray([[False, True, False]]),
        zi_soil=zi,
        zf_soil_b=zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
    )

    assert np.allclose(np.asarray(coeff.xc_cryoturb)[0, :, 1], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(np.asarray(coeff.xd_cryoturb)[0, :, 1], [1.0, 1.0, 0.1, 0.01, 0.0, 0.0])

    alpha = np.zeros(6, dtype=np.float64)
    beta_c = np.zeros(6, dtype=np.float64)
    beta_doc = np.zeros(6, dtype=np.float64)
    beta_litter = np.zeros(6, dtype=np.float64)
    xc = np.asarray(coeff.xc_cryoturb)[0, :, 1]
    xd = np.asarray(coeff.xd_cryoturb)[0, :, 1]
    xe = xc[5] + xd[4]
    alpha[4] = xd[4] / xe
    beta_c[4] = xc[5] * carbon[0, 0, 1, 5] / xe
    beta_doc[4] = xc[5] * doc[0, 1, 5, IFREE, IACT, ICARBON] / xe
    beta_litter[4] = xc[5] * litter[0, 0, 1, 5, ICARBON] / xe
    for layer in range(3, -1, -1):
        xe = xc[layer + 1] + (1.0 - alpha[layer + 1]) * xd[layer + 1] + xd[layer]
        alpha[layer] = xd[layer] / xe
        beta_c[layer] = (xc[layer + 1] * carbon[0, 0, 1, layer + 1] + xd[layer + 1] * beta_c[layer + 1]) / xe
        beta_doc[layer] = (xc[layer + 1] * doc[0, 1, layer + 1, IFREE, IACT, ICARBON] + xd[layer + 1] * beta_doc[layer + 1]) / xe
        beta_litter[layer] = (
            xc[layer + 1] * litter[0, 0, 1, layer + 1, ICARBON] + xd[layer + 1] * beta_litter[layer + 1]
        ) / xe
    assert np.allclose(np.asarray(coeff.alpha_c)[0, 0, 1, :], alpha)
    assert np.allclose(np.asarray(coeff.beta_c)[0, 0, 1, :], beta_c)
    assert np.allclose(np.asarray(coeff.beta_doc)[0, 1, :, IFREE, IACT], beta_doc)
    assert np.allclose(np.asarray(coeff.beta_litter_below)[0, 0, 1, :], beta_litter)

    result = soilcarbon_cryoturbation_diffuse(carbon, doc, litter, coeff, zi)
    expected_c = np.zeros(6, dtype=np.float64)
    expected_c[0] = (carbon[0, 0, 1, 0] + beta_c[0]) / (1.0 + (1.0 - alpha[0]))
    for layer in range(1, 6):
        expected_c[layer] = alpha[layer - 1] * expected_c[layer - 1] + beta_c[layer - 1]
    assert np.allclose(np.asarray(result.carbon_32l)[0, 0, 1, :], expected_c)
    assert np.allclose(np.asarray(result.carbon_32l)[0, 0, 0, :], carbon[0, 0, 0, :])
    assert np.allclose(np.asarray(result.doc)[0, 1, :, 1, IACT, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.litter_below)[0, 1, 1, :, ICARBON] > 0.0, True)


def test_soilcarbon_cryoturbation_cycle_converts_stock_to_concentration_and_back():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1348-1368."""

    carbon, doc, litter = _cryoturb_base_arrays(ndeep=6)
    zf = np.asarray([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0], dtype=np.float64)
    zi = np.asarray([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=np.float64)
    thickness = zf[1:] - zf[:-1]
    altmax_lastyear = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)
    result, coeff = soilcarbon_cryoturbation_cycle(
        carbon,
        doc,
        litter,
        None,
        altmax_ind=np.asarray([[0, 1, 0]], dtype=np.int32),
        altmax_lastyear=altmax_lastyear,
        fixed_cryoturbation_depth=altmax_lastyear,
        veget_mask=np.asarray([[False, True, False]]),
        zi_soil=zi,
        zf_soil_b=zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
    )

    assert np.allclose(np.asarray(result.carbon_32l), carbon)
    assert np.allclose(np.asarray(result.doc), doc)
    assert np.allclose(np.asarray(result.litter_below), litter)
    expected_coeff = soilcarbon_cryoturbation_coefficients(
        altmax_ind=np.asarray([[0, 1, 0]], dtype=np.int32),
        carbon_32l=carbon / thickness[None, None, None, :],
        doc=doc / thickness[None, None, :, None, None, None],
        litter_below=litter / thickness[None, None, None, :, None],
        altmax_lastyear=altmax_lastyear,
        fixed_cryoturbation_depth=altmax_lastyear,
        veget_mask=np.asarray([[False, True, False]]),
        zi_soil=zi,
        zf_soil_b=zf,
        dt_seconds=1.0,
        diff_k_const=1.0,
        bio_diff_k_const=0.0,
    )
    assert np.allclose(np.asarray(coeff.beta_c), np.asarray(expected_coeff.beta_c))


def test_soilcarbon_perma_peat_cmax_matches_source_formula():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1078-1087."""

    peat_bd = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    zf = np.asarray([0.0, 0.5, 1.5, 3.0], dtype=np.float64)

    cmax = np.asarray(soilcarbon_perma_peat_cmax(peat_bd, zf))

    peat_soc = (1.0 / ((0.4 * peat_bd + 0.13) ** 2.19)) * 0.01
    expected = peat_bd * 1.0e6 * peat_soc * np.diff(zf)
    assert np.allclose(cmax, expected)


def test_soilcarbon_perma_peat_redistribute_moves_excess_downward_sequentially():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1375-1437."""

    carbon = np.zeros((1, NCARB, 3, 3), dtype=np.float64)
    carbon[0, :, PFT14 % 3, 0] = [50.0, 30.0, 20.0]
    carbon[0, :, PFT14 % 3, 1] = [10.0, 10.0, 10.0]
    carbon[0, :, PFT14 % 3, 2] = [2.0, 2.0, 2.0]
    is_peat = np.zeros(3, dtype=bool)
    is_peat[PFT14 % 3] = True
    veget_mask = np.zeros((1, 3), dtype=bool)
    veget_mask[0, PFT14 % 3] = True
    zf = np.asarray([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
    cmax = np.asarray([80.0, 20.0, 50.0], dtype=np.float64)

    result = soilcarbon_perma_peat_redistribute(
        carbon,
        cmax,
        zf,
        is_peat,
        veget_mask,
        frac1=0.95,
        frac2=0.05,
        min_stomate=0.0,
    )

    # Layer 1: total 100 > 76, remove 5%, keep pool ratios, transfer 5 * 1/2 downward.
    layer0_after = np.asarray([47.5, 28.5, 19.0])
    transfer0 = 5.0 * (1.0 / 2.0)
    layer1_before_check = np.asarray([10.0, 10.0, 10.0]) + layer0_after / 95.0 * transfer0
    deep1_before_check = np.sum(layer1_before_check)
    excess1 = deep1_before_check * 0.05
    layer1_after = layer1_before_check * ((deep1_before_check - excess1) / deep1_before_check)
    transfer1 = excess1 * (2.0 / 3.0)
    layer2_after = np.asarray([2.0, 2.0, 2.0]) + layer1_after / (deep1_before_check - excess1) * transfer1

    updated = np.asarray(result.carbon_32l)[0, :, PFT14 % 3, :]
    assert np.allclose(updated[:, 0], layer0_after)
    assert np.allclose(updated[:, 1], layer1_after)
    assert np.allclose(updated[:, 2], layer2_after)
    deepc = np.asarray(result.deepc_peat)[0, :, PFT14 % 3]
    assert np.allclose(deepc, [95.0, deep1_before_check - excess1, 6.0 + transfer1])
    assert np.allclose(np.asarray(result.deepc_pt)[0, :2, PFT14 % 3], [95.0 / 1.0, (deep1_before_check - excess1) / 2.0])
    expected_olt_layer2 = zf[2] + (zf[3] - zf[2]) * deepc[2] / cmax[2]
    assert np.allclose(np.asarray(result.peat_olt)[0, PFT14 % 3], expected_olt_layer2)
    assert np.allclose(np.asarray(result.carbon_32l)[0, :, 0, :], 0.0)


def test_deep_carbon_vertical_integral_masks_and_clips_surface_depth():
    """Fortran: stomate_permafrost_soilcarbon.f90 calc_vert_int_soil_carbon lines 4425-4463."""

    npts, ndeep, nvm = 1, 3, 14
    deep_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.zeros_like(deep_a)
    deep_p = np.zeros_like(deep_a)
    deep_a[0, :, PFT14] = [1.0, 2.0, 4.0]
    deep_s[0, :, PFT14] = [10.0, 20.0, 40.0]
    deep_p[0, :, PFT14] = [100.0, 200.0, 400.0]
    deep_a[0, :, 3] = 999.0
    deep_s[0, :, 3] = 999.0
    deep_p[0, :, 3] = 999.0
    zf = np.asarray([0.0, 0.5, 1.5, 3.0], dtype=np.float64)
    veget_mask = np.zeros((npts, nvm), dtype=bool)
    veget_mask[0, PFT14] = True

    result = deep_carbon_vertical_integral_step(deep_a, deep_s, deep_p, zf, veget_mask)

    thickness = np.asarray([0.5, 1.0, 1.5], dtype=np.float64)
    surface_thickness = np.asarray([0.5, 1.0, 0.5], dtype=np.float64)
    assert np.allclose(np.asarray(result.carbon)[0, IACTIVE, PFT14], np.dot(deep_a[0, :, PFT14], thickness))
    assert np.allclose(np.asarray(result.carbon)[0, ISLOW, PFT14], np.dot(deep_s[0, :, PFT14], thickness))
    assert np.allclose(np.asarray(result.carbon)[0, IPASSIVE, PFT14], np.dot(deep_p[0, :, PFT14], thickness))
    assert np.allclose(np.asarray(result.carbon_surf)[0, IACTIVE, PFT14], np.dot(deep_a[0, :, PFT14], surface_thickness))
    assert np.allclose(np.asarray(result.carbon_surf)[0, ISLOW, PFT14], np.dot(deep_s[0, :, PFT14], surface_thickness))
    assert np.allclose(np.asarray(result.carbon_surf)[0, IPASSIVE, PFT14], np.dot(deep_p[0, :, PFT14], surface_thickness))
    assert np.allclose(np.asarray(result.carbon)[0, :, 3], 0.0)
    assert np.allclose(np.asarray(result.carbon_surf)[0, :, 3], 0.0)


def test_deep_carbon_vertical_integral_validates_shape_contracts():
    deep = np.zeros((1, 2, 3), dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    mask = np.ones((1, 3), dtype=bool)

    with pytest.raises(ValueError, match="deepC_s and deepC_p"):
        deep_carbon_vertical_integral_step(deep, np.zeros((1, 3, 3)), deep, zf, mask)
    with pytest.raises(ValueError, match="zf_soil"):
        deep_carbon_vertical_integral_step(deep, deep, deep, np.asarray([0.0, 1.0]), mask)
    with pytest.raises(ValueError, match="veget_mask"):
        deep_carbon_vertical_integral_step(deep, deep, deep, zf, np.ones((1, 2), dtype=bool))


def _expected_carbinput_profile(soilc_in_ts, z_lit, intdep, zi, zf, *, correct=True):
    active = zi < intdep
    profile = np.where(active, soilc_in_ts / z_lit / (1.0 - np.exp(-intdep / z_lit)) * np.exp(-zi / z_lit), 0.0)
    if correct:
        total = np.sum(profile * np.diff(zf))
        profile = profile * soilc_in_ts / total
    return profile


def test_deep_carbon_input_default_path_distributes_and_corrects_to_timestep_input():
    """Fortran: stomate_permafrost_soilcarbon.f90 carbinput lines 3314-3383."""

    npts, ndeep, nvm = 1, 3, 14
    deep_a = np.ones((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.ones_like(deep_a) * 2.0
    deep_p = np.ones_like(deep_a) * 3.0
    soilc_in = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    soilc_in[0, IACTIVE, PFT14] = 8.0
    soilc_in[0, ISLOW, PFT14] = 4.0
    soilc_in[0, IPASSIVE, PFT14] = 2.0
    soilc_in[0, IACTIVE, 3] = 999.0
    zroot = np.zeros((npts, nvm), dtype=np.float64)
    zroot[0, PFT14] = 1.0
    altmax = np.ones((npts, nvm), dtype=np.float64) * 3.0
    rprof = np.ones((npts, nvm), dtype=np.float64) * 4.0
    veget_mask = np.zeros((npts, nvm), dtype=bool)
    veget_mask[0, PFT14] = True
    zi = np.asarray([0.25, 0.75, 1.5], dtype=np.float64)
    zf = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64)

    result = deep_carbon_input_step(
        deep_a,
        deep_s,
        deep_p,
        soilc_in,
        zroot,
        altmax,
        veget_mask,
        zi,
        zf,
        rprof,
        time_step_seconds=43200.0,
    )

    expected_active = _expected_carbinput_profile(4.0, 1.0, 1.0, zi, zf)
    expected_slow = _expected_carbinput_profile(2.0, 1.0, 1.0, zi, zf)
    expected_passive = _expected_carbinput_profile(1.0, 1.0, 1.0, zi, zf)
    assert np.allclose(np.asarray(result.dc_litter_z)[0, IACTIVE, :, PFT14], expected_active)
    assert np.allclose(np.asarray(result.dc_litter_z)[0, ISLOW, :, PFT14], expected_slow)
    assert np.allclose(np.asarray(result.dc_litter_z)[0, IPASSIVE, :, PFT14], expected_passive)
    assert np.allclose(np.sum(np.asarray(result.dc_litter_z)[0, :, :, PFT14] * np.diff(zf)[None, :], axis=1), [4.0, 2.0, 1.0])
    assert np.allclose(np.asarray(result.deepC_a)[0, :, PFT14], 1.0 + expected_active)
    assert np.allclose(np.asarray(result.deepC_s)[0, :, PFT14], 2.0 + expected_slow)
    assert np.allclose(np.asarray(result.deepC_p)[0, :, PFT14], 3.0 + expected_passive)
    assert np.allclose(np.asarray(result.dc_litter_z)[0, :, :, 3], 0.0)
    assert np.allclose(np.asarray(result.deepC_a)[0, :, 3], 1.0)


def test_deep_carbon_input_new_intdep_zlit_uses_fineroot_and_altmax_limits():
    """Fortran: stomate_permafrost_soilcarbon.f90 carbinput lines 3314-3327."""

    npts, ndeep, nvm = 1, 3, 14
    deep = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    soilc_in = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    soilc_in[0, IACTIVE, PFT14] = 6.0
    zroot = np.ones((npts, nvm), dtype=np.float64) * 9.0
    altmax = np.ones((npts, nvm), dtype=np.float64) * 1.2
    rprof = np.ones((npts, nvm), dtype=np.float64) * 4.0
    veget_mask = np.zeros((npts, nvm), dtype=bool)
    veget_mask[0, PFT14] = True
    zi = np.asarray([0.25, 0.75, 1.25], dtype=np.float64)
    zf = np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float64)

    result = deep_carbon_input_step(
        deep,
        deep,
        deep,
        soilc_in,
        zroot,
        altmax,
        veget_mask,
        zi,
        zf,
        rprof,
        time_step_seconds=86400.0,
        new_carbinput_intdepzlit=True,
        correct_carboninput_vertprof=False,
    )

    expected = _expected_carbinput_profile(6.0, 0.75, 1.2, zi, zf, correct=False)
    assert np.allclose(np.asarray(result.dc_litter_z)[0, IACTIVE, :, PFT14], expected)
    assert np.allclose(np.asarray(result.dc_litter_z)[0, IACTIVE, 2, PFT14], 0.0)
    assert np.allclose(np.asarray(result.deepC_a)[0, :, PFT14], expected)


def test_deep_carbon_input_no_pfrost_decomp_leaves_state_unchanged():
    deep = np.ones((1, 3, 14), dtype=np.float64)
    soilc_in = np.ones((1, NCARB, 14), dtype=np.float64)
    z2d = np.ones((1, 14), dtype=np.float64)
    mask = np.ones((1, 14), dtype=bool)
    zi = np.asarray([0.25, 0.75, 1.25], dtype=np.float64)
    zf = np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float64)

    result = deep_carbon_input_step(
        deep,
        deep * 2.0,
        deep * 3.0,
        soilc_in,
        z2d,
        z2d,
        mask,
        zi,
        zf,
        z2d,
        time_step_seconds=86400.0,
        no_pfrost_decomp=True,
    )

    assert np.allclose(np.asarray(result.deepC_a), deep)
    assert np.allclose(np.asarray(result.deepC_s), deep * 2.0)
    assert np.allclose(np.asarray(result.deepC_p), deep * 3.0)
    assert np.allclose(np.asarray(result.dc_litter_z), 0.0)


def test_deep_carbon_permafrost_decomp_oxic_updates_pools_and_deltaC1():
    """Fortran: stomate_permafrost_soilcarbon.f90 permafrost_decomp lines 4064-4304."""

    npts, ndeep, nvm = 1, 1, 14
    deep_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.zeros_like(deep_a)
    deep_p = np.zeros_like(deep_a)
    deep_a[0, 0, PFT14] = 100.0
    deep_s[0, 0, PFT14] = 200.0
    deep_p[0, 0, PFT14] = 300.0
    fbact = np.ones_like(deep_a) * 1000.0
    o2 = np.ones_like(deep_a) * 1000.0
    totpor = np.ones_like(deep_a)
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True
    clay = np.asarray([0.2], dtype=np.float64)

    result = deep_carbon_permafrost_decomp_oxic_step(
        deep_a,
        deep_s,
        deep_p,
        fbact,
        o2,
        totpor,
        clay,
        mask,
        np.asarray([0.0, 1.0], dtype=np.float64),
        time_step_seconds=100.0,
    )

    dca = 100.0 * 100.0 / 1000.0 * (1.0 - 0.75 * 0.2)
    dcs = 200.0 * 100.0 / (1000.0 * 37.0)
    dcp = 300.0 * 100.0 / (1000.0 * 1617.45)
    a_to_p = 0.004 * dca
    a_to_s = (1.0 - (0.85 - 0.68 * 0.2) - 0.004) * dca
    s_to_a = 0.42 * dcs
    s_to_p = 0.03 * dcs
    p_to_a = 0.45 * dcp
    expected_a = 100.0 - dca + s_to_a + p_to_a
    expected_s = 200.0 - dcs + a_to_s
    expected_p = 300.0 - dcp + a_to_p + s_to_p
    fr_a = 0.85 - 0.68 * 0.2
    expected_o2 = 1000.0 - (32.0 / 12.0) * (dca * fr_a + dcs * 0.55 + dcp * 0.55)
    assert np.allclose(np.asarray(result.deepC_a)[0, 0, PFT14], expected_a)
    assert np.allclose(np.asarray(result.deepC_s)[0, 0, PFT14], expected_s)
    assert np.allclose(np.asarray(result.deepC_p)[0, 0, PFT14], expected_p)
    assert np.allclose(np.asarray(result.O2_soil)[0, 0, PFT14], expected_o2)
    assert np.allclose(np.asarray(result.deltaC1_a)[0, 0, PFT14], dca * fr_a)
    assert np.allclose(np.asarray(result.deltaC1_s)[0, 0, PFT14], dcs * 0.55)
    assert np.allclose(np.asarray(result.deltaC1_p)[0, 0, PFT14], dcp * 0.55)
    assert np.allclose(np.asarray(result.deepC_a)[0, 0, 3], 0.0)


def test_deep_carbon_permafrost_decomp_oxygen_limit_is_applied_sequentially():
    """Fortran: stomate_permafrost_soilcarbon.f90 permafrost_decomp lines 4134-4189."""

    npts, ndeep, nvm = 1, 1, 14
    deep = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep[0, 0, PFT14] = 100.0
    fbact = np.ones_like(deep) * 10.0
    o2 = np.zeros_like(deep)
    o2[0, 0, PFT14] = 1.0
    totpor = np.ones_like(deep)
    airvol = np.ones_like(deep)
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True

    result = deep_carbon_permafrost_decomp_oxic_step(
        deep,
        deep,
        deep,
        fbact,
        o2,
        totpor,
        np.asarray([0.0], dtype=np.float64),
        mask,
        np.asarray([0.0, 1.0], dtype=np.float64),
        time_step_seconds=100.0,
        airvol_soil=airvol,
        oxlim=True,
    )

    dca = (1.0 * 12.0 / 32.0) * (1.0 - 0.75 * 0.0)
    o2_after_a = max(1.0 - (32.0 / 12.0) * dca * 0.85, 0.0)
    dcs = min(100.0 * 100.0 / (100.0 * 37.0), o2_after_a * 12.0 / 32.0)
    o2_after_s = max(o2_after_a - (32.0 / 12.0) * dcs * 0.55, 0.0)
    dcp = min(100.0 * 100.0 / (100.0 * 1617.45), o2_after_s * 12.0 / 32.0)
    expected_o2 = max(o2_after_s - (32.0 / 12.0) * dcp * 0.55, 0.0)
    assert np.allclose(np.asarray(result.deltaC1_a)[0, 0, PFT14], dca * 0.85)
    assert np.allclose(np.asarray(result.deltaC1_s)[0, 0, PFT14], dcs * 0.55)
    assert np.allclose(np.asarray(result.deltaC1_p)[0, 0, PFT14], dcp * 0.55)
    assert np.allclose(np.asarray(result.O2_soil)[0, 0, PFT14], expected_o2)


def test_deep_carbon_permafrost_decomp_methane_branch_updates_ch4_o2_and_deltas():
    """Fortran: stomate_permafrost_soilcarbon.f90 permafrost_decomp lines 4195-4300."""

    npts, ndeep, nvm = 1, 1, 14
    deep_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.zeros_like(deep_a)
    deep_p = np.zeros_like(deep_a)
    deep_a[0, 0, PFT14] = 100.0
    deep_s[0, 0, PFT14] = 50.0
    deep_p[0, 0, PFT14] = 20.0
    fbact = np.ones_like(deep_a) * 1000.0
    o2 = np.ones_like(deep_a) * 1.0
    ch4 = np.ones_like(deep_a) * 2.0
    totpor = np.ones_like(deep_a)
    hslong = np.zeros_like(deep_a)
    tprof = np.ones_like(deep_a) * 274.15
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True
    clay = np.asarray([0.0], dtype=np.float64)

    result = deep_carbon_permafrost_decomp_oxic_step(
        deep_a,
        deep_s,
        deep_p,
        fbact,
        o2,
        totpor,
        clay,
        mask,
        np.asarray([0.0, 1.0], dtype=np.float64),
        time_step_seconds=100.0,
        ok_methane=True,
        CH4_soil=ch4,
        totporCH4_soil=totpor,
        hslong=hslong,
        tprof=tprof,
        tau_CH4troph=200.0,
        fbactratio=10.0,
        O2m=3.0,
        MG_useallCpools=False,
    )

    dca_oxic = 100.0 * 100.0 / 1000.0
    o2_after_oxic = max(1.0 - (32.0 / 12.0) * dca_oxic * 0.85, 0.0)
    deep_a_after_oxic = 100.0 - dca_oxic
    meth_lim = np.exp(-o2_after_oxic / 3.0)
    dca_mg = deep_a_after_oxic * 100.0 / (1000.0 * 10.0) * meth_lim
    dch4_gen = dca_mg * 0.85 * 16.0 / 12.0
    ch4_after_gen = 2.0 + dch4_gen
    dch4m = o2_after_oxic / 2.0 * 16.0 / 32.0
    dch4_trophy = min(ch4_after_gen * 100.0 / 200.0, dch4m)
    expected_ch4 = ch4_after_gen - dch4_trophy
    expected_o2 = max(o2_after_oxic - 2.0 * dch4_trophy * 32.0 / 16.0, 0.0)
    assert np.allclose(np.asarray(result.deltaCH4g)[0, 0, PFT14], dch4_gen)
    assert np.allclose(np.asarray(result.deltaC2)[0, 0, PFT14], dca_mg * 0.85)
    assert np.allclose(np.asarray(result.deltaCH4)[0, 0, PFT14], dch4_trophy)
    assert np.allclose(np.asarray(result.deltaC3)[0, 0, PFT14], dch4_trophy / 16.0 * 12.0)
    assert np.allclose(np.asarray(result.CH4_soil)[0, 0, PFT14], expected_ch4)
    assert np.allclose(np.asarray(result.O2_soil)[0, 0, PFT14], expected_o2)
    assert np.allclose(np.asarray(result.nadd_soil)[0, 0, PFT14], dch4_gen / 16.0 - 2.0 * dch4_trophy / 16.0)


def test_deep_carbon_permafrost_decomp_can_apply_perma_peat():
    """Fortran: stomate_permafrost_soilcarbon.f90 permafrost_decomp lines 4195-4400."""

    npts, ndeep, nvm = 1, 2, 14
    deep_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.zeros_like(deep_a)
    deep_p = np.zeros_like(deep_a)
    deep_a[0, :, PFT14] = [50.0, 2.0]
    deep_s[0, :, PFT14] = [30.0, 2.0]
    deep_p[0, :, PFT14] = [20.0, 2.0]
    fbact = np.ones_like(deep_a) * 1.0e12
    o2 = np.ones_like(deep_a) * 1000.0
    totpor = np.ones_like(deep_a)
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    zf = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)

    result = deep_carbon_permafrost_decomp_oxic_step(
        deep_a,
        deep_s,
        deep_p,
        fbact,
        o2,
        totpor,
        np.asarray([0.0], dtype=np.float64),
        mask,
        zf,
        time_step_seconds=100.0,
        perma_peat=True,
        cmax_peat=np.asarray([80.0, 50.0], dtype=np.float64),
        is_peat=is_peat,
        frac1=0.95,
        frac2=0.05,
    )

    assert result.perma_peat is not None
    # First layer starts at 100 g m-2 and transfers 5% scaled by 1m/2m downward.
    assert np.allclose(np.asarray(result.perma_peat.deepc_peat)[0, :, PFT14], [95.0, 14.5], atol=1e-9)
    assert np.allclose(np.asarray(result.deepC_a)[0, 0, PFT14], 47.5)
    assert np.allclose(np.asarray(result.deepC_s)[0, 0, PFT14], 28.5)
    assert np.allclose(np.asarray(result.deepC_p)[0, 0, PFT14], 19.0)


def test_deep_carbon_nonmethane_core_composes_source_order_helpers():
    """Fortran: stomate_permafrost_soilcarbon.f90 deep_carbcycle lines 953-1077."""

    npts, ndeep, nvm = 1, 3, 14
    deep_a = np.ones((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.ones_like(deep_a) * 2.0
    deep_p = np.ones_like(deep_a) * 3.0
    soilc_in = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    soilc_in[0, :, PFT14] = [8.0, 4.0, 2.0]
    fbact = np.ones_like(deep_a) * 1000.0
    o2 = np.ones_like(deep_a) * 1000.0
    totpor = np.ones_like(deep_a)
    clay = np.asarray([0.2], dtype=np.float64)
    veget_mask = np.zeros((npts, nvm), dtype=bool)
    veget_mask[0, PFT14] = True
    zi = np.asarray([0.25, 0.75, 1.5], dtype=np.float64)
    zf = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    zroot = np.ones((npts, nvm), dtype=np.float64)
    altmax_lastyear = np.ones((npts, nvm), dtype=np.float64) * 1.5
    rprof = np.ones((npts, nvm), dtype=np.float64) * 2.0

    carbon_input = deep_carbon_input_step(
        deep_a,
        deep_s,
        deep_p,
        soilc_in,
        zroot,
        altmax_lastyear,
        veget_mask,
        zi,
        zf,
        rprof,
        time_step_seconds=43200.0,
    )
    decomp = deep_carbon_permafrost_decomp_oxic_step(
        carbon_input.deepC_a,
        carbon_input.deepC_s,
        carbon_input.deepC_p,
        fbact,
        o2,
        totpor,
        clay,
        veget_mask,
        zf,
        time_step_seconds=43200.0,
    )
    vertical = deep_carbon_vertical_integral_step(decomp.deepC_a, decomp.deepC_s, decomp.deepC_p, zf, veget_mask)
    expected_resp = (
        np.sum(
            (np.asarray(decomp.deltaC1_a) + np.asarray(decomp.deltaC1_s) + np.asarray(decomp.deltaC1_p))
            * np.diff(zf)[None, :, None],
            axis=1,
        )
        * 86400.0
        / 43200.0
    )
    actual = deep_carbon_nonmethane_core_step(
        deep_a,
        deep_s,
        deep_p,
        soilc_in,
        fbact,
        o2,
        totpor,
        clay,
        veget_mask,
        zi,
        zf,
        zroot,
        altmax_lastyear,
        rprof,
        time_step_seconds=43200.0,
    )

    assert np.allclose(np.asarray(actual.deepC_a), np.asarray(decomp.deepC_a))
    assert np.allclose(np.asarray(actual.deepC_s), np.asarray(decomp.deepC_s))
    assert np.allclose(np.asarray(actual.deepC_p), np.asarray(decomp.deepC_p))
    assert np.allclose(np.asarray(actual.carbon), np.asarray(vertical.carbon))
    assert np.allclose(np.asarray(actual.carbon_surf), np.asarray(vertical.carbon_surf))
    assert np.allclose(np.asarray(actual.resp_hetero_soil), expected_resp)
    assert np.allclose(np.asarray(actual.dc_litter_z), np.asarray(carbon_input.dc_litter_z))
    assert actual.cryoturbation_coefficients is None


def test_deep_carbon_nonmethane_core_with_cryoturbation_matches_explicit_source_order():
    """Fortran: stomate_permafrost_soilcarbon.f90 deep_carbcycle lines 953-1060."""

    npts, ndeep, nvm = 1, 4, 14
    deep_a = np.ones((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.ones_like(deep_a) * 2.0
    deep_p = np.ones_like(deep_a) * 3.0
    deep_a[0, :, PFT14] = [10.0, 12.0, 14.0, 16.0]
    deep_s[0, :, PFT14] = [20.0, 23.0, 26.0, 29.0]
    deep_p[0, :, PFT14] = [30.0, 34.0, 38.0, 42.0]
    soilc_in = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    soilc_in[0, :, PFT14] = [1.0, 0.5, 0.25]
    fbact = np.ones_like(deep_a) * 10000.0
    o2 = np.ones_like(deep_a) * 1000.0
    totpor = np.ones_like(deep_a)
    clay = np.asarray([0.1], dtype=np.float64)
    veget_mask = np.zeros((npts, nvm), dtype=bool)
    veget_mask[0, PFT14] = True
    zi = np.asarray([0.2, 0.6, 1.2, 2.0], dtype=np.float64)
    zf = np.asarray([0.0, 0.4, 0.9, 1.6, 2.5], dtype=np.float64)
    altmax_ind = np.zeros((npts, nvm), dtype=np.int32)
    altmax_ind[0, PFT14] = 1
    altmax_lastyear = np.zeros((npts, nvm), dtype=np.float64)
    altmax_lastyear[0, PFT14] = 0.7
    zroot = np.ones((npts, nvm), dtype=np.float64)
    rprof = np.ones((npts, nvm), dtype=np.float64) * 2.0
    previous = deep_carbon_cryoturbation_coefficients(
        altmax_ind,
        deep_a,
        deep_s,
        deep_p,
        altmax_lastyear,
        altmax_lastyear,
        veget_mask,
        zi,
        zf,
        dt_seconds=43200.0,
        diff_k_const=0.2,
        bio_diff_k_const=0.0,
    )
    diffuse = deep_carbon_cryoturbation_diffuse(deep_a, deep_s, deep_p, previous, altmax_ind, zi, zf)
    carbon_input = deep_carbon_input_step(
        diffuse.deepC_a,
        diffuse.deepC_s,
        diffuse.deepC_p,
        soilc_in,
        zroot,
        altmax_lastyear,
        veget_mask,
        zi,
        zf,
        rprof,
        time_step_seconds=43200.0,
    )
    decomp = deep_carbon_permafrost_decomp_oxic_step(
        carbon_input.deepC_a,
        carbon_input.deepC_s,
        carbon_input.deepC_p,
        fbact,
        o2,
        totpor,
        clay,
        veget_mask,
        zf,
        time_step_seconds=43200.0,
    )
    expected_coeff = deep_carbon_cryoturbation_coefficients(
        altmax_ind,
        decomp.deepC_a,
        decomp.deepC_s,
        decomp.deepC_p,
        altmax_lastyear,
        altmax_lastyear,
        veget_mask,
        zi,
        zf,
        dt_seconds=43200.0,
        diff_k_const=0.2,
        bio_diff_k_const=0.0,
    )

    actual = deep_carbon_nonmethane_core_step(
        deep_a,
        deep_s,
        deep_p,
        soilc_in,
        fbact,
        o2,
        totpor,
        clay,
        veget_mask,
        zi,
        zf,
        zroot,
        altmax_lastyear,
        rprof,
        time_step_seconds=43200.0,
        ok_cryoturb=True,
        cryoturbation_coefficients=previous,
        altmax_ind=altmax_ind,
        fixed_cryoturbation_depth=altmax_lastyear,
        cryoturbation_diff_k_const=0.2,
        cryoturbation_bio_diff_k_const=0.0,
    )

    assert np.allclose(np.asarray(actual.deepC_a), np.asarray(decomp.deepC_a))
    assert np.allclose(np.asarray(actual.deepC_s), np.asarray(decomp.deepC_s))
    assert np.allclose(np.asarray(actual.deepC_p), np.asarray(decomp.deepC_p))
    assert actual.cryoturbation_coefficients is not None
    assert np.allclose(np.asarray(actual.cryoturbation_coefficients.beta_a), np.asarray(expected_coeff.beta_a))


def test_deep_carbon_core_methane_flux_aggregation_matches_source_formula():
    """Fortran: stomate_permafrost_soilcarbon.f90 deep_carbcycle lines 1013-1049."""

    npts, ndeep, nvm = 1, 2, 14
    deep = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep[0, :, PFT14] = [100.0, 50.0]
    soilc_in = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    fbact = np.ones_like(deep) * 1000.0
    o2 = np.ones_like(deep) * 100.0
    ch4 = np.ones_like(deep) * 2.0
    totpor = np.ones_like(deep)
    hslong = np.ones_like(deep) * 0.5
    tprof = np.ones_like(deep) * 283.15
    clay = np.asarray([0.0], dtype=np.float64)
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True
    zi = np.asarray([0.5, 1.5], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    zroot = np.ones((npts, nvm), dtype=np.float64)
    altmax_lastyear = np.ones((npts, nvm), dtype=np.float64)
    rprof = np.ones((npts, nvm), dtype=np.float64)

    decomp = deep_carbon_permafrost_decomp_oxic_step(
        deep,
        deep * 2.0,
        deep * 3.0,
        fbact,
        o2,
        totpor,
        clay,
        mask,
        zf,
        time_step_seconds=100.0,
        ok_methane=True,
        CH4_soil=ch4,
        totporCH4_soil=totpor,
        hslong=hslong,
        tprof=tprof,
        tau_CH4troph=200.0,
    )
    thickness = np.diff(zf)
    dC1i = np.sum(
        (np.asarray(decomp.deltaC1_a) + np.asarray(decomp.deltaC1_s) + np.asarray(decomp.deltaC1_p))
        * thickness[None, :, None],
        axis=1,
    )
    mt = np.sum(np.asarray(decomp.deltaCH4) * totpor * thickness[None, :, None], axis=1)
    mg = np.sum(np.asarray(decomp.deltaCH4g) * totpor * thickness[None, :, None], axis=1)
    ch4_final = np.sum(np.asarray(decomp.CH4_soil) * totpor * thickness[None, :, None], axis=1)
    ch4_initial = np.sum(ch4 * totpor * thickness[None, :, None], axis=1)
    expected_resp = (dC1i + mt * (12.0 / 16.0)) * 86400.0 / 100.0
    expected_sflux = (ch4_initial - ch4_final + mg - mt) * 86400.0 / 100.0
    expected_heat = (
        40.0e6 * 1.0e-3 * np.asarray(decomp.deltaC1_a)
        + 30.0e6 * 1.0e-3 * np.asarray(decomp.deltaC1_s)
        + 10.0e6 * 1.0e-3 * np.asarray(decomp.deltaC1_p)
        + 3.1e6 * 1.0e-3 * np.asarray(decomp.deltaC2)
        + 9.4e6 * 1.0e-3 * np.asarray(decomp.deltaCH4) * totpor
    ) / 100.0
    expected_heat = np.where(mask[:, None, :], expected_heat, 0.0)

    actual = deep_carbon_nonmethane_core_step(
        deep,
        deep * 2.0,
        deep * 3.0,
        soilc_in,
        fbact,
        o2,
        totpor,
        clay,
        mask,
        zi,
        zf,
        zroot,
        altmax_lastyear,
        rprof,
        time_step_seconds=100.0,
        ok_methane=True,
        CH4_soil=ch4,
        totporCH4_soil=totpor,
        hslong=hslong,
        tprof=tprof,
        tau_CH4troph=200.0,
        firstcall=True,
        heat_ch4_gen=3.1e6,
        heat_ch4_troph=9.4e6,
    )

    assert np.allclose(np.asarray(actual.resp_hetero_soil), expected_resp)
    assert np.allclose(np.asarray(actual.sfluxCH4), expected_sflux)
    assert np.allclose(np.asarray(actual.heat_Zimov), expected_heat)
    assert np.allclose(np.asarray(actual.CH4_soil), np.asarray(decomp.CH4_soil))


def test_deep_carbon_yedoma_reset_everywhere_sets_below_active_layer_and_clears_above():
    """Fortran: stomate_permafrost_soilcarbon.f90 initialize_yedoma_carbonstocks lines 3135-3165."""

    npts, ndeep, nvm = 1, 4, 14
    deep_a = np.ones((npts, ndeep, nvm), dtype=np.float64) * 7.0
    deep_s = np.ones_like(deep_a) * 8.0
    deep_p = np.ones_like(deep_a) * 9.0
    zz = np.asarray([0.25, 0.75, 1.5, 3.0], dtype=np.float64)
    altmax_ind = np.zeros((npts, nvm), dtype=np.int32)
    altmax_ind[0, PFT14] = 2
    veget_mask = np.zeros((npts, nvm), dtype=bool)
    veget_mask[0, PFT14] = True

    result = deep_carbon_yedoma_reset_step(
        deep_a,
        deep_s,
        deep_p,
        zz,
        altmax_ind,
        veget_mask,
        yedoma_depth=1.6,
        yedoma_cinit_act=10.0,
        yedoma_cinit_slo=20.0,
        yedoma_cinit_pas=30.0,
        yedoma_map_filename="EVERYWHERE",
    )

    assert int(np.asarray(result.yedoma_depth_index)) == 3
    assert np.allclose(np.asarray(result.deepC_a)[0, :, PFT14], [0.0, 10.0, 10.0, 7.0])
    assert np.allclose(np.asarray(result.deepC_s)[0, :, PFT14], [0.0, 20.0, 20.0, 8.0])
    assert np.allclose(np.asarray(result.deepC_p)[0, :, PFT14], [0.0, 30.0, 30.0, 9.0])
    assert np.allclose(np.asarray(result.deepC_a)[0, :, 0], 7.0)
    assert np.allclose(np.asarray(result.deepC_a)[0, :, 3], 7.0)


def test_deep_carbon_yedoma_reset_none_and_explicit_zero_clear_reset_depth_only():
    deep = np.ones((1, 3, 14), dtype=np.float64) * 5.0
    zz = np.asarray([0.25, 0.75, 1.5], dtype=np.float64)
    altmax_ind = np.zeros((1, 14), dtype=np.int32)
    altmax_ind[0, PFT14] = 1
    veget_mask = np.zeros((1, 14), dtype=bool)
    veget_mask[0, PFT14] = True

    result = deep_carbon_yedoma_reset_step(
        deep,
        deep,
        deep,
        zz,
        altmax_ind,
        veget_mask,
        yedoma_depth=0.8,
        yedoma_cinit_act=10.0,
        yedoma_cinit_slo=20.0,
        yedoma_cinit_pas=30.0,
        yedoma_map_filename="NONE",
    )

    assert np.allclose(np.asarray(result.deepC_a)[0, :, PFT14], [0.0, 0.0, 5.0])
    with pytest.raises(ValueError, match="yedoma_values"):
        deep_carbon_yedoma_reset_step(
            deep,
            deep,
            deep,
            zz,
            altmax_ind,
            veget_mask,
            yedoma_depth=0.8,
            yedoma_cinit_act=10.0,
            yedoma_cinit_slo=20.0,
            yedoma_cinit_pas=30.0,
            yedoma_map_filename="some_file.nc",
        )


def test_soilcarbon_leak_doc_export_aggregate_maps_doc_pools_and_dic_terms():
    npts, nvm, nelements = 1, 14, 1
    doc_exp = np.zeros((npts, nvm, NEXP, 7, nelements), dtype=np.float64)
    doc_exp[0, PFT14, IRUNOFF, IACTIVE, ICARBON] = 2.0
    doc_exp[0, PFT14, IRUNOFF, ISLO, ICARBON] = 3.0
    doc_exp[0, PFT14, IRUNOFF, IPAS, ICARBON] = np.nan
    doc_exp[0, 0, IRUNOFF, IACTIVE, ICARBON] = 999.0
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.5
    resp_litter = np.zeros((npts, nvm, 2), dtype=np.float64)
    resp_litter[0, PFT14, IBELOW] = 4.0
    resp_soil = np.zeros((npts, nvm), dtype=np.float64)
    resp_soil[0, PFT14] = 2.0
    resp_maint = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    resp_maint[0, PFT14, IROOT] = 1.0
    resp_flood = np.zeros((npts, nvm), dtype=np.float64)
    resp_flood[0, PFT14] = 0.25
    resp_flood_soil = np.zeros((npts, nvm), dtype=np.float64)
    resp_flood_soil[0, PFT14] = 0.5
    flood_root = np.zeros((npts, nvm), dtype=np.float64)
    flood_root[0, PFT14] = 0.75
    runoff = np.ones((npts, 4), dtype=np.float64) * 10.0
    drainage = np.ones((npts, 4), dtype=np.float64) * 20.0
    pref = np.zeros(nvm, dtype=np.int32)
    pref[PFT14] = 3

    result = soilcarbon_leak_doc_export_aggregate(
        doc_exp,
        veget_max,
        resp_litter,
        resp_soil,
        resp_flood,
        resp_flood_soil,
        resp_maint,
        flood_root,
        runoff,
        drainage,
        pref,
        flood_frac=np.asarray([0.2], dtype=np.float64),
        dt_days=0.5,
    )

    agg = np.asarray(result.doc_exp_agg)
    assert np.allclose(agg[0, IRUNOFF, IDOCL], 2.0 * 0.5 * 0.5)
    assert np.allclose(agg[0, IRUNOFF, IDOCR], 3.0 * 0.5 * 0.5)
    soil_resp_modif = (4.0 + 2.0 + 1.0) / (4.25 * 0.8 * 0.5)
    assert np.allclose(np.asarray(result.soil_resp_modif)[0, PFT14], soil_resp_modif)
    assert np.allclose(agg[0, IRUNOFF, ICO2AQ], 10.0 * 20e-4 * 0.5 * soil_resp_modif)
    assert np.allclose(agg[0, IDRAINAGE, ICO2AQ], 20.0 * 20e-3 * 0.5 * soil_resp_modif)
    assert np.allclose(agg[0, IFLOODED, ICO2AQ], (0.25 + 0.5 + 0.75) * 0.5)
    assert np.allclose(np.asarray(result.doc_exp_b)[0, PFT14, IRUNOFF, IDOCL, ICARBON], 2.0 * 0.5)
    assert np.allclose(np.asarray(result.doc_exp_b)[0, 0, IRUNOFF, IDOCL, ICARBON], 0.0)


def test_soilcarbon_leak_frac_carb_matches_source_table():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1291-1302."""

    from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_frac_carb

    clay = np.asarray([0.2, 0.5], dtype=np.float64)

    frac_carb = np.asarray(soilcarbon_leak_frac_carb(clay))

    assert frac_carb.shape == (2, NCARB, NCARB)
    active_to_passive = 0.004 / (1.0 - 0.85 + 0.68 * clay)
    assert np.allclose(frac_carb[:, IACTIVE, IACTIVE], 0.0)
    assert np.allclose(frac_carb[:, IACTIVE, IPASSIVE], active_to_passive)
    assert np.allclose(frac_carb[:, IACTIVE, ISLOW], 1.0 - active_to_passive)
    assert np.allclose(frac_carb[:, ISLOW, ISLOW], 0.0)
    assert np.allclose(frac_carb[:, ISLOW, IACTIVE], 0.93)
    assert np.allclose(frac_carb[:, ISLOW, IPASSIVE], 0.07)
    assert np.allclose(frac_carb[:, IPASSIVE, IPASSIVE], 0.0)
    assert np.allclose(frac_carb[:, IPASSIVE, IACTIVE], 1.0)
    assert np.allclose(frac_carb[:, IPASSIVE, ISLOW], 0.0)


def test_soilcarbon_leak_activity_factors_build_doc_and_poc_controls():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1176-1193."""

    from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_activity_factors

    fbact_doc = np.asarray([[[2.0], [4.0]]], dtype=np.float64)
    fbact = np.asarray([[[0.2], [0.4]]], dtype=np.float64)

    result = soilcarbon_leak_activity_factors(fbact_doc, fbact)

    fbact_doc_labile = fbact_doc * (0.149 * 365.0 / 1.3)
    fbact_doc_refractory = fbact_doc * (0.149 * 365.0 / 60.4)
    expected_npool = np.zeros((1, 2, 1, NPOOL), dtype=np.float64)
    expected_npool[:, :, :, [IMETABO, IMETBEL, ISTRABO, ISTRBEL, IACT]] = fbact_doc_labile[:, :, :, None]
    expected_npool[:, :, :, [IPAS, ISLO]] = fbact_doc_refractory[:, :, :, None]
    expected_ncarb = np.zeros((1, 2, 1, NCARB), dtype=np.float64)
    expected_ncarb[:, :, :, IACTIVE] = fbact
    expected_ncarb[:, :, :, ISLOW] = fbact * (0.149 / 5.48)
    expected_ncarb[:, :, :, IPASSIVE] = fbact * (0.149 / 241.0)

    assert np.allclose(np.asarray(result.fbact_doc_labile), fbact_doc_labile)
    assert np.allclose(np.asarray(result.fbact_doc_refractory), fbact_doc_refractory)
    assert np.allclose(np.asarray(result.fbact_npool), expected_npool)
    assert np.allclose(np.asarray(result.fbact_ncarb), expected_ncarb)


def test_soilcarbon_leak_apply_doc_inputs_updates_carbon_only():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1528-1557."""

    from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_apply_doc_inputs

    npts, nvm, ndeep, ndoc, nelements = 1, 14, 3, 2, 2
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[..., 1] = 99.0
    soilcarbon_input_doc = np.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=np.float64)
    soilcarbon_input_doc[0, PFT14, :, IMETABO, ICARBON] = [1.0, 2.0, 3.0]
    soilcarbon_input_doc[0, PFT14, :, ISTRABO, ICARBON] = [4.0, 5.0, 6.0]
    soilcarbon_input_doc[0, PFT14, :, IMETBEL, ICARBON] = [7.0, 8.0, 9.0]
    soilcarbon_input_doc[0, PFT14, :, ISTRBEL, ICARBON] = [10.0, 11.0, 12.0]
    soilcarbon_input_doc[0, PFT14, :, IACT, ICARBON] = 100.0
    soilcarbon_input_doc[0, PFT14, :, ISLO, ICARBON] = 200.0
    soilcarbon_input_doc[0, PFT14, :, IPAS, ICARBON] = 300.0
    soilcarbon_input_doc[0, PFT14, :, :, 1] = 777.0
    doc_to_topsoil = np.zeros((npts, 4), dtype=np.float64)
    doc_to_subsoil = np.zeros((npts, 4), dtype=np.float64)
    doc_to_topsoil[0, IDOCL] = 2.0
    doc_to_topsoil[0, IDOCR] = 4.0
    doc_to_subsoil[0, IDOCL] = 6.0
    doc_to_subsoil[0, IDOCR] = 8.0
    wet_dep_ground = np.zeros((npts, nvm, nelements), dtype=np.float64)
    wet_dep_ground[0, PFT14, ICARBON] = 5.0
    wet_dep_ground[0, PFT14, 1] = 500.0
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.5

    updated = np.asarray(
        soilcarbon_leak_apply_doc_inputs(
            doc,
            soilcarbon_input_doc,
            doc_to_topsoil,
            doc_to_subsoil,
            wet_dep_ground,
            veget_max,
            bio_frac=np.asarray([0.5], dtype=np.float64),
            dt_days=0.25,
            nslm=2,
        )
    )

    assert np.allclose(updated[..., 1], 99.0)
    assert np.allclose(updated[0, PFT14, :, IFREE, IMETABO, ICARBON], [0.25, 0.5, 0.75])
    assert np.allclose(updated[0, PFT14, :, IFREE, ISTRABO, ICARBON], [1.0, 1.25, 1.5])
    assert np.allclose(updated[0, PFT14, :, IFREE, IMETBEL, ICARBON], [1.75, 2.0, 2.25])
    assert np.allclose(updated[0, PFT14, :, IFREE, ISTRBEL, ICARBON], [2.5, 2.75, 3.0])
    assert np.allclose(updated[0, PFT14, 0, IFREE, IACT, ICARBON], 2.0 * 0.25 / 0.5 + 5.0 * 0.5 / 0.5)
    assert np.allclose(updated[0, PFT14, 1, IFREE, IACT, ICARBON], 6.0 * 0.25 / 0.5)
    assert np.allclose(updated[0, PFT14, 0, IFREE, ISLO, ICARBON], 4.0 * 0.25 / 0.5 + 5.0 * 0.5 / 0.5)
    assert np.allclose(updated[0, PFT14, 1, IFREE, ISLO, ICARBON], 8.0 * 0.25 / 0.5)
    assert np.allclose(updated[0, PFT14, 2, IFREE, IACT, ICARBON], 0.0)
    assert np.allclose(updated[0, PFT14, 2, IFREE, ISLO, ICARBON], 0.0)
    assert np.allclose(updated[0, PFT14, :, IFREE, IPAS, ICARBON], 0.0)
    assert np.allclose(updated[0, 0, :, IFREE, :, ICARBON], 0.0)


def test_soilcarbon_leak_litter_lom_constructs_pool_specific_lom():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1563-1590."""

    from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_litter_lom

    npts, nvm, ndeep, ndoc, nelements = 1, 14, 2, 2, 1
    litter_above = np.zeros((npts, 2, nvm, nelements), dtype=np.float64)
    litter_below = np.zeros((npts, 2, nvm, ndeep, nelements), dtype=np.float64)
    litter_above[0, IMETABOLIC, PFT14, ICARBON] = 1.0
    litter_above[0, ISTRUCTURAL, PFT14, ICARBON] = 2.0
    litter_below[0, IMETABOLIC, PFT14, :, ICARBON] = [3.0, 5.0]
    litter_below[0, ISTRUCTURAL, PFT14, :, ICARBON] = [4.0, 6.0]
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    carbon_32l[0, IACTIVE, PFT14, :] = [10.0, 20.0]
    carbon_32l[0, ISLOW, PFT14, :] = [30.0, 40.0]
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    for pool, value in {
        IMETABO: 0.1,
        ISTRABO: 0.2,
        IMETBEL: 0.3,
        ISTRBEL: 0.4,
        IACT: 0.5,
        ISLO: 0.6,
    }.items():
        doc[0, PFT14, :, IFREE, pool, ICARBON] = value

    result = soilcarbon_leak_litter_lom(litter_above, litter_below, carbon_32l, doc)
    litter_tot = np.asarray(result.litter_tot)
    lom = np.asarray(result.lom)

    assert np.allclose(litter_tot[0, PFT14, :], [10.0, 11.0])
    assert np.allclose(lom[0, PFT14, :, IACTIVE], [11.0, 12.0])
    assert np.allclose(lom[0, PFT14, :, ISLOW], [21.5, 32.5])
    assert np.allclose(lom[0, PFT14, :, IPASSIVE], [52.1, 73.1])


def test_soilcarbon_leak_litter_lom_limits_belowground_litter_to_nslm():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1564-1570."""

    from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_litter_lom

    npts, nvm, ndeep, nelements = 1, 14, 3, 1
    litter_above = np.zeros((npts, 2, nvm, nelements), dtype=np.float64)
    litter_below = np.zeros((npts, 2, nvm, ndeep, nelements), dtype=np.float64)
    litter_below[0, :, PFT14, :, ICARBON] = [[1.0, 2.0, 30.0], [4.0, 5.0, 60.0]]
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    doc = np.zeros((npts, nvm, ndeep, 2, NPOOL, nelements), dtype=np.float64)

    result = soilcarbon_leak_litter_lom(
        litter_above,
        litter_below,
        carbon_32l,
        doc,
        nslm=2,
    )

    assert np.array_equal(np.asarray(result.litter_tot)[0, PFT14], [5.0, 7.0, 0.0])
    assert np.array_equal(np.asarray(result.lom)[0, PFT14, :, IACTIVE], [5.0, 7.0, 0.0])


def test_soilcarbon_leak_decompose_poc_doc_natural_peat_pft14_branch():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1605-1785."""

    from jax_orchidee.stomate.soilcarbon_kernels import (
        soilcarbon_leak_decompose_update,
    )

    npts, nvm, ndeep, ndoc, nelements = 1, 14, 1, 2, 1
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    carbon_32l[0, :, PFT14, 0] = [10.0, 20.0, 30.0]
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[0, PFT14, 0, IFREE, :, ICARBON] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    fbact_ncarb = np.zeros((npts, ndeep, nvm, NCARB), dtype=np.float64)
    fbact_ncarb[0, 0, PFT14, :] = 0.2
    fbact_npool = np.zeros((npts, ndeep, nvm, NPOOL), dtype=np.float64)
    fbact_npool[0, 0, PFT14, :] = 0.1
    clay = np.asarray([0.2], dtype=np.float64)
    soil_mc_32l = np.ones((npts, ndeep, 4), dtype=np.float64)
    pref_soil_veg = np.zeros(nvm, dtype=np.int32)
    pref_soil_veg[PFT14] = 0
    natural = np.zeros(nvm, dtype=bool)
    is_peat = np.zeros(nvm, dtype=bool)
    natural[PFT14] = True
    is_peat[PFT14] = True

    result = soilcarbon_leak_decompose_update(
        carbon_32l,
        doc,
        np.zeros((npts, 2, nvm, nelements), dtype=np.float64),
        np.zeros((npts, 2, nvm, ndeep, nelements), dtype=np.float64),
        np.full((npts, nvm), 0.25, dtype=np.float64),
        np.full((npts, nvm, ndeep), 0.4, dtype=np.float64),
        fbact_npool,
        fbact_ncarb,
        soil_mc_32l,
        pref_soil_veg,
        np.asarray([0.3], dtype=np.float64),
        clay,
        natural,
        is_peat,
        np.zeros(nvm, dtype=bool),
        dt_days=0.5,
        cue=0.3,
        priming=False,
        nslm=1,
    )

    carbon_after = np.asarray(result.carbon_32l)[0, :, PFT14, 0]
    doc_after = np.asarray(result.doc)[0, PFT14, 0, IFREE, :, ICARBON]
    assert np.allclose(np.asarray(result.fluxtot)[0, :, ICARBON, PFT14, 0], [0.595, 1.4, 2.1])
    assert np.allclose(np.asarray(result.fluxtot_flood)[0, :, ICARBON, PFT14, 0], [0.085, 0.2, 0.3])
    assert np.allclose(np.asarray(result.fluxtot_doc)[0, PFT14, 0, :], 0.035 * np.arange(1.0, 8.0))
    assert np.allclose(np.asarray(result.fluxtot_doc_flood)[0, PFT14, 0, :], 0.005 * np.arange(1.0, 8.0))
    assert np.allclose(carbon_after, [9.56576, 18.48436084, 27.60587972])
    assert np.allclose(doc_after, [0.96, 1.92, 2.88, 3.84, 4.9785, 6.18, 7.35])
    assert np.allclose(np.asarray(result.resp_hetero_soil)[0, PFT14], 7.105)
    assert np.allclose(np.asarray(result.resp_flood_soil)[0, PFT14], 1.015)


def test_soilcarbon_leak_adsorption_desorption_matches_source_equilibrium():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1789-1838."""

    from jax_orchidee.stomate.soilcarbon_kernels import (
        IADSORBED,
        soilcarbon_leak_adsorption_desorption,
    )

    npts, nvm, ndeep, ndoc, nelements = 2, 14, 2, 2, 1
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[0, PFT14, 0, IFREE, IACT, ICARBON] = 8.0
    doc[0, PFT14, 0, IADSORBED, IACT, ICARBON] = 1.0
    doc[1, PFT14, 0, IFREE, IACT, ICARBON] = 1.0
    doc[1, PFT14, 0, IADSORBED, IACT, ICARBON] = 9.0
    clay = np.asarray([0.3, 0.3], dtype=np.float64)
    bulk_dens = np.asarray([20.0, 1.0], dtype=np.float64)
    z_soil = np.asarray([0.0, 0.1, 0.4], dtype=np.float64)

    result = soilcarbon_leak_adsorption_desorption(doc, clay, bulk_dens, z_soil, nslm=2)
    updated = np.asarray(result.doc)
    kd0 = 10.0 ** (-3.1 + 0.2 * np.log10(30.0) + np.log10(0.1 * 100.0 / 2.0))
    kd0 = np.clip(kd0, 10.0**-3.2, 10.0**-1.5)
    kd1 = 10.0 ** (-3.1 + 0.2 * np.log10(30.0) + np.log10((0.4 + 0.1) / 2.0))
    kd1 = np.clip(kd1, 10.0**-3.2, 10.0**-1.5)

    assert np.allclose(np.asarray(result.kd)[:, 0], kd0)
    assert np.allclose(np.asarray(result.kd)[:, 1], kd1)
    normal_re = kd0 * bulk_dens[0] / (kd0 * bulk_dens[0] + 1.0) * 9.0
    delta = normal_re - 1.0
    assert np.allclose(updated[0, PFT14, 0, IFREE, IACT, ICARBON], 8.0 - delta)
    assert np.allclose(updated[0, PFT14, 0, IADSORBED, IACT, ICARBON], 1.0 + delta)
    desorb_re = kd0 * bulk_dens[1] / (kd0 * bulk_dens[1] + 1.0) * 10.0
    assert np.allclose(updated[1, PFT14, 0, IFREE, IACT, ICARBON], 10.0 - desorb_re)
    assert np.allclose(updated[1, PFT14, 0, IADSORBED, IACT, ICARBON], desorb_re)
    assert np.allclose(
        updated[:, PFT14, 0, IFREE, IACT, ICARBON] + updated[:, PFT14, 0, IADSORBED, IACT, ICARBON],
        [9.0, 10.0],
    )


def test_soilcarbon_leak_water_transport_matches_water_flux_branches():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1840-1968."""

    npts, nvm, ndeep, ndoc, nelements, nstm = 4, 14, 3, 2, 1, 4
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[:, PFT14, :, IFREE, IACT, ICARBON] = np.asarray(
        [
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [1.0, 2.0, 30.0],
        ],
        dtype=np.float64,
    )
    doc[:, PFT14, :, 1, IACT, ICARBON] = 99.0
    soil_mc = np.ones((npts, ndeep, nstm), dtype=np.float64)
    soil_mc_32l = np.ones((npts, ndeep, nstm), dtype=np.float64)
    wat_flux = np.zeros((npts, ndeep, nstm), dtype=np.float64)
    wat_flux[:, :, 3] = np.asarray(
        [
            [1000.0, 500.0, 0.0],
            [-1000.0, -500.0, 0.0],
            [1000.0, -500.0, 0.0],
            [5000.0, 5000.0, 0.0],
        ],
        dtype=np.float64,
    )
    pref = np.zeros(nvm, dtype=np.int32)
    pref[PFT14] = 3
    zf_soil_b = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    result = soilcarbon_leak_water_transport(
        doc,
        soil_mc,
        soil_mc_32l,
        wat_flux,
        pref,
        zf_soil_b,
        flux_red=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        nslm=3,
    )
    updated = np.asarray(result.doc)
    flux = np.asarray(result.doc_flux)

    assert np.allclose(updated[0, PFT14, :, IFREE, IACT, ICARBON], [0.0, 20.0, 40.0])
    assert np.allclose(flux[0, PFT14, :, IACT, ICARBON], [10.0, 10.0, 0.0])
    assert np.allclose(updated[1, PFT14, :, IFREE, IACT, ICARBON], [30.0, 15.0, 15.0])
    assert np.allclose(flux[1, PFT14, :, IACT, ICARBON], [20.0, 15.0, 0.0])
    assert np.allclose(updated[2, PFT14, :, IFREE, IACT, ICARBON], [0.0, 45.0, 15.0])
    assert np.allclose(updated[3, PFT14, :, IFREE, IACT, ICARBON], [0.0, 1.0, 32.0])
    assert np.allclose(updated[:, PFT14, :, 1, IACT, ICARBON], 99.0)
    assert np.allclose(updated[:, 0, :, IFREE, IACT, ICARBON], 0.0)


def test_soilcarbon_leak_doc_diffusion_matches_peak_limit_and_temperature_gate():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1970-2050."""

    npts, nvm, ndeep, ndoc, nelements = 2, 14, 3, 2, 1
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[0, PFT14, :, IFREE, IACT, ICARBON] = [1.0, 3.0, 1.0]
    doc[1, PFT14, :, IFREE, IACT, ICARBON] = [1.0, 3.0, 1.0]
    doc[:, PFT14, :, 1, IACT, ICARBON] = 77.0
    tprof = np.full((npts, ndeep, nvm), 273.15, dtype=np.float64)
    tprof[1, 1, PFT14] = 272.15

    result = soilcarbon_leak_doc_diffusion(
        doc,
        tprof,
        zf_soil_b=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        dif_doc=np.asarray([1.0, 1.0], dtype=np.float64),
    )
    updated = np.asarray(result.doc)
    flux = np.asarray(result.doc_flux_diff)

    assert np.allclose(flux[0, PFT14, :2, IACT, ICARBON], [1.5, 1.5])
    assert np.allclose(updated[0, PFT14, :, IFREE, IACT, ICARBON], [2.5, 0.0, 2.5])
    assert np.allclose(flux[1, PFT14, :, IACT, ICARBON], 0.0)
    assert np.allclose(updated[1, PFT14, :, IFREE, IACT, ICARBON], [1.0, 3.0, 1.0])
    assert np.allclose(updated[:, PFT14, :, 1, IACT, ICARBON], 77.0)


def test_soilcarbon_leak_doc_diffusion_moves_all_source_when_flux_exceeds_stock():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 2023-2043."""

    npts, nvm, ndeep, ndoc, nelements = 2, 14, 2, 2, 1
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[0, PFT14, :, IFREE, IACT, ICARBON] = [1.0, 2.0]
    doc[1, PFT14, :, IFREE, IACT, ICARBON] = [2.0, 1.0]
    tprof = np.full((npts, ndeep, nvm), 273.15, dtype=np.float64)

    result = soilcarbon_leak_doc_diffusion(
        doc,
        tprof,
        zf_soil_b=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        dif_doc=np.asarray([10.0, 10.0], dtype=np.float64),
    )
    updated = np.asarray(result.doc)

    assert np.allclose(updated[0, PFT14, :, IFREE, IACT, ICARBON], [3.0, 0.0])
    assert np.allclose(updated[1, PFT14, :, IFREE, IACT, ICARBON], [0.0, 3.0])


def test_soilcarbon_leak_doc_export_maps_runoff_drain_flood_and_subtracts_pools():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 2053-2303."""

    npts, nvm, ndeep, ndoc, nelements, nstm = 1, 14, 2, 2, 1, 4
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[0, PFT14, 0, IFREE, IMETABO, ICARBON] = 10.0
    doc[0, PFT14, 0, IFREE, ISTRABO, ICARBON] = 20.0
    doc[0, PFT14, 0, IFREE, IACT, ICARBON] = 30.0
    doc[0, PFT14, 1, IFREE, ISTRBEL, ICARBON] = 40.0
    doc[0, PFT14, 1, IFREE, ISLO, ICARBON] = 50.0
    doc[0, PFT14, 1, 1, IACT, ICARBON] = 99.0
    soilwater_31mm = np.ones((npts, nstm), dtype=np.float64)
    runoff = np.zeros((npts, nstm), dtype=np.float64)
    drainage = np.zeros((npts, nstm), dtype=np.float64)
    runoff[:, 3] = 100.0
    drainage[:, 3] = 50.0
    pref = np.zeros(nvm, dtype=np.int32)
    pref[PFT14] = 3
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.5
    wet_dep_flood = np.zeros((npts, nvm, nelements), dtype=np.float64)
    wet_dep_flood[0, PFT14, ICARBON] = 2.0
    floodcarbon_input = np.zeros((npts, nvm, NPOOL, nelements), dtype=np.float64)
    floodcarbon_input[0, PFT14, IACT, ICARBON] = 4.0
    floodcarbon_input[0, PFT14, ISLO, ICARBON] = 6.0
    fluxtot_flood = np.zeros((npts, NCARB, nelements, nvm, ndeep), dtype=np.float64)
    fluxtot_flood[0, IACTIVE, ICARBON, PFT14, :] = [1.0, 2.0]
    fluxtot_flood[0, ISLOW, ICARBON, PFT14, :] = [3.0, 4.0]
    fluxtot_flood[0, IPASSIVE, ICARBON, PFT14, :] = [5.0, 6.0]
    lignin_above = np.zeros((npts, nvm), dtype=np.float64)
    lignin_below = np.zeros((npts, nvm, ndeep), dtype=np.float64)
    lignin_above[0, PFT14] = 0.25
    lignin_below[0, PFT14, 1] = 0.4
    soil_mc = np.ones((npts, ndeep, nstm), dtype=np.float64)
    soil_mc_32l = np.ones((npts, ndeep, nstm), dtype=np.float64)

    result = soilcarbon_leak_doc_export(
        doc,
        soilwater_31mm,
        runoff,
        drainage,
        runoff2peat=np.zeros((npts, nstm), dtype=np.float64),
        fastr=np.asarray([25.0], dtype=np.float64),
        flux_red=np.asarray([1.0], dtype=np.float64),
        pref_soil_veg=pref,
        veget_max=veget_max,
        wet_dep_flood=wet_dep_flood,
        floodcarbon_input=floodcarbon_input,
        fluxtot_flood=fluxtot_flood,
        lignin_struc_above=lignin_above,
        lignin_struc_below=lignin_below,
        soil_mc=soil_mc,
        soil_mc_32l=soil_mc_32l,
        zf_soil_b=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        z_soil=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        is_peat=np.zeros(nvm, dtype=bool),
        dt_days=0.5,
        cue_coef=np.asarray([0.3], dtype=np.float64),
        sro_bottom=2,
        nslm=2,
    )
    doc_run = np.asarray(result.doc_run)[0, PFT14, :, ICARBON]
    doc_drain = np.asarray(result.doc_drain)[0, PFT14, :, ICARBON]
    doc_flood = np.asarray(result.doc_flood)[0, PFT14, :, ICARBON]
    doc_after = np.asarray(result.doc)

    assert np.allclose(np.asarray(result.fastr_corr), [1.0])
    corr = 20.0 / 150.0
    runoff_factor = corr * 100.0e-3
    assert np.allclose(np.asarray(result.soil_doc_corr)[0, PFT14], corr)
    assert np.allclose(doc_run[IACT], 10.0 * runoff_factor + 20.0 * runoff_factor * 0.75 + 30.0 * runoff_factor + 40.0 * runoff_factor * 0.6)
    assert np.allclose(doc_run[ISLO], 20.0 * runoff_factor * 0.25 + 40.0 * runoff_factor * 0.4 + 50.0 * runoff_factor)
    assert np.allclose(doc_run[[IMETABO, ISTRABO, ISTRBEL]], 0.0)
    assert np.allclose(doc_drain[IACT], 1.2)
    assert np.allclose(doc_drain[ISLO], 0.8 + 2.5)
    assert np.allclose(doc_flood[IACT], 2.0 + 2.0 + 0.3 * 3.0)
    assert np.allclose(doc_flood[ISLO], 2.0 + 3.0 + 0.3 * 7.0)
    assert np.allclose(doc_flood[IPAS], 0.3 * 11.0)
    # Fortran lines 2260-2262: runoff, flooded, drainage; each amount / dt.
    assert (IRUNOFF, IFLOODED, IDRAINAGE) == (0, 1, 2)
    np.testing.assert_array_equal(
        np.asarray(result.doc_exp)[0, PFT14, IFLOODED, :, ICARBON],
        doc_flood / 0.5,
    )
    np.testing.assert_array_equal(
        np.asarray(result.doc_exp)[0, PFT14, IDRAINAGE, :, ICARBON],
        doc_drain / 0.5,
    )
    assert np.allclose(np.asarray(result.doc_exp)[0, PFT14, IRUNOFF, IACT, ICARBON], doc_run[IACT] / 0.5)
    assert np.allclose(doc_after[0, PFT14, 0, IFREE, IMETABO, ICARBON], 10.0 - 10.0 * runoff_factor)
    assert np.allclose(doc_after[0, PFT14, 0, IFREE, ISTRABO, ICARBON], 20.0 - 20.0 * runoff_factor)
    istrbel_after_runoff = 40.0 - 40.0 * runoff_factor
    assert np.allclose(doc_after[0, PFT14, 1, IFREE, ISTRBEL, ICARBON], istrbel_after_runoff - istrbel_after_runoff * 50.0e-3)
    assert np.allclose(doc_after[0, PFT14, 1, 1, IACT, ICARBON], 99.0)


def test_soilcarbon_leak_doc_export_adds_peat_runoff_back_before_export():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 2102-2141."""

    npts, nvm, ndeep, ndoc, nelements, nstm = 1, 14, 1, 2, 1, 4
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[0, PFT14, 0, IFREE, IACT, ICARBON] = 10.0
    pref = np.zeros(nvm, dtype=np.int32)
    pref[PFT14] = 3
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0
    soil_mc = np.ones((npts, ndeep, nstm), dtype=np.float64)
    soil_mc_32l = np.ones((npts, ndeep, nstm), dtype=np.float64)
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True

    result = soilcarbon_leak_doc_export(
        doc,
        soilwater_31mm=np.ones((npts, nstm), dtype=np.float64),
        runoff_per_soil=np.zeros((npts, nstm), dtype=np.float64),
        drainage_per_soil=np.zeros((npts, nstm), dtype=np.float64),
        runoff2peat=np.asarray([[100.0, 0.0, 0.0, 100.0]], dtype=np.float64),
        fastr=np.asarray([25.0], dtype=np.float64),
        flux_red=np.asarray([1.0], dtype=np.float64),
        pref_soil_veg=pref,
        veget_max=veget_max,
        wet_dep_flood=np.zeros((npts, nvm, nelements), dtype=np.float64),
        floodcarbon_input=np.zeros((npts, nvm, NPOOL, nelements), dtype=np.float64),
        fluxtot_flood=np.zeros((npts, NCARB, nelements, nvm, ndeep), dtype=np.float64),
        lignin_struc_above=np.zeros((npts, nvm), dtype=np.float64),
        lignin_struc_below=np.zeros((npts, nvm, ndeep), dtype=np.float64),
        soil_mc=soil_mc,
        soil_mc_32l=soil_mc_32l,
        zf_soil_b=np.asarray([0.0, 1.0], dtype=np.float64),
        z_soil=np.asarray([0.0, 1.0], dtype=np.float64),
        is_peat=is_peat,
        dt_days=1.0,
        sro_bottom=1,
        nslm=1,
    )

    assert np.allclose(np.asarray(result.doc_run_2_peat)[0, PFT14, IACT, 0, ICARBON], 1.0)
    assert np.allclose(np.asarray(result.doc)[0, PFT14, 0, IFREE, IACT, ICARBON], 11.0)


def test_soilcarbon_leak_core_step_matches_explicit_source_order_composition():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1176-2303."""

    from jax_orchidee.stomate.soilcarbon_kernels import (
        soilcarbon_leak_activity_factors,
        soilcarbon_leak_adsorption_desorption,
        soilcarbon_leak_apply_doc_inputs,
        soilcarbon_leak_decompose_update,
    )

    npts, nvm, ndeep, ndoc, nelements, nstm = 1, 14, 2, 2, 1, 4
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    carbon_32l[0, :, PFT14, :] = [[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]]
    doc = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    doc[0, PFT14, :, IFREE, IACT, ICARBON] = [2.0, 3.0]
    soilcarbon_input_doc = np.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=np.float64)
    soilcarbon_input_doc[0, PFT14, :, IMETABO, ICARBON] = [0.4, 0.5]
    doc_to_topsoil = np.zeros((npts, 4), dtype=np.float64)
    doc_to_subsoil = np.zeros((npts, 4), dtype=np.float64)
    doc_to_topsoil[0, IDOCL] = 0.2
    doc_to_subsoil[0, IDOCR] = 0.3
    wet_dep_ground = np.zeros((npts, nvm, nelements), dtype=np.float64)
    wet_dep_ground[0, PFT14, ICARBON] = 0.1
    wet_dep_flood = np.zeros((npts, nvm, nelements), dtype=np.float64)
    wet_dep_flood[0, PFT14, ICARBON] = 0.2
    litter_above = np.zeros((npts, 2, nvm, nelements), dtype=np.float64)
    litter_below = np.zeros((npts, 2, nvm, ndeep, nelements), dtype=np.float64)
    litter_above[0, :, PFT14, ICARBON] = [0.2, 0.3]
    litter_below[0, :, PFT14, :, ICARBON] = [[0.1, 0.2], [0.2, 0.3]]
    lignin_above = np.full((npts, nvm), 0.25, dtype=np.float64)
    lignin_below = np.full((npts, nvm, ndeep), 0.35, dtype=np.float64)
    fbact_doc = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    fbact = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    fbact_doc[0, :, PFT14] = [0.01, 0.02]
    fbact[0, :, PFT14] = [0.02, 0.03]
    soil_mc = np.ones((npts, ndeep, nstm), dtype=np.float64)
    soil_mc_32l = np.ones((npts, ndeep, nstm), dtype=np.float64)
    wat_flux = np.zeros((npts, ndeep, nstm), dtype=np.float64)
    wat_flux[0, :, 3] = [10.0, 0.0]
    soilwater_31mm = np.ones((npts, nstm), dtype=np.float64)
    runoff = np.zeros((npts, nstm), dtype=np.float64)
    drainage = np.zeros((npts, nstm), dtype=np.float64)
    runoff[0, 3] = 5.0
    drainage[0, 3] = 4.0
    pref = np.zeros(nvm, dtype=np.int32)
    pref[PFT14] = 3
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.5
    flood_frac = np.asarray([0.1], dtype=np.float64)
    floodcarbon_input = np.zeros((npts, nvm, NPOOL, nelements), dtype=np.float64)
    floodcarbon_input[0, PFT14, IACT, ICARBON] = 0.2
    tprof = np.full((npts, ndeep, nvm), 273.15, dtype=np.float64)
    clay = np.asarray([0.3], dtype=np.float64)
    bulk_dens = np.asarray([1.65], dtype=np.float64)
    z_soil = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    zf_soil_b = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    flux_red = np.asarray([1.0], dtype=np.float64)
    natural = np.zeros(nvm, dtype=bool)
    natural[PFT14] = True
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    is_c4 = np.zeros(nvm, dtype=bool)
    dif_doc = np.asarray([1.0e-5], dtype=np.float64)

    controls = soilcarbon_leak_activity_factors(fbact_doc, fbact)
    doc_inputs = soilcarbon_leak_apply_doc_inputs(
        doc,
        soilcarbon_input_doc,
        doc_to_topsoil,
        doc_to_subsoil,
        wet_dep_ground,
        veget_max,
        dt_days=0.5,
        nslm=2,
    )
    decomp = soilcarbon_leak_decompose_update(
        carbon_32l,
        doc_inputs,
        litter_above,
        litter_below,
        lignin_above,
        lignin_below,
        controls.fbact_npool,
        controls.fbact_ncarb,
        soil_mc_32l,
        pref,
        flood_frac,
        clay,
        natural,
        is_peat,
        is_c4,
        dt_days=0.5,
        nslm=2,
    )
    ads = soilcarbon_leak_adsorption_desorption(decomp.doc, clay, bulk_dens, z_soil, nslm=2)
    water = soilcarbon_leak_water_transport(ads.doc, soil_mc, soil_mc_32l, wat_flux, pref, zf_soil_b, flux_red, nslm=2)
    diffusion = soilcarbon_leak_doc_diffusion(water.doc, tprof, zf_soil_b, dif_doc)
    export = soilcarbon_leak_doc_export(
        diffusion.doc,
        soilwater_31mm,
        runoff,
        drainage,
        np.zeros((npts, nstm), dtype=np.float64),
        np.asarray([25.0], dtype=np.float64),
        flux_red,
        pref,
        veget_max,
        wet_dep_flood,
        floodcarbon_input,
        decomp.fluxtot_flood,
        lignin_above,
        lignin_below,
        soil_mc,
        soil_mc_32l,
        zf_soil_b,
        z_soil,
        is_peat,
        dt_days=0.5,
        sro_bottom=2,
        nslm=2,
    )
    composed = soilcarbon_leak_core_step(
        carbon_32l,
        doc,
        soilcarbon_input_doc,
        doc_to_topsoil,
        doc_to_subsoil,
        wet_dep_ground,
        wet_dep_flood,
        litter_above,
        litter_below,
        lignin_above,
        lignin_below,
        fbact_doc,
        fbact,
        soil_mc,
        soil_mc_32l,
        wat_flux,
        soilwater_31mm,
        runoff,
        drainage,
        np.zeros((npts, nstm), dtype=np.float64),
        np.asarray([25.0], dtype=np.float64),
        pref,
        veget_max,
        flood_frac,
        floodcarbon_input,
        tprof,
        clay,
        bulk_dens,
        z_soil,
        zf_soil_b,
        flux_red,
        natural,
        is_peat,
        is_c4,
        dt_days=0.5,
        dif_doc=dif_doc,
        sro_bottom=2,
        nslm=2,
    )

    assert np.allclose(np.asarray(composed.carbon_32l), np.asarray(decomp.carbon_32l))
    assert np.allclose(np.asarray(composed.doc), np.asarray(export.doc))
    assert np.allclose(np.asarray(composed.doc_exp), np.asarray(export.doc_exp))
    assert np.allclose(np.asarray(composed.kd), np.asarray(ads.kd))
    assert np.allclose(np.asarray(composed.doc_flux), np.asarray(water.doc_flux))
    assert np.allclose(np.asarray(composed.doc_flux_diff), np.asarray(diffusion.doc_flux_diff))


def test_soilcarbon_leak_core_step_with_cryoturbation_matches_explicit_cycle_then_core():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1348-1368."""

    npts, nvm, ndeep, ndoc, nelements, nstm = 1, 14, 6, 2, 1, 4
    carbon, doc, litter_below = _cryoturb_base_arrays(ndeep=ndeep)
    carbon_full = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    doc_full = np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64)
    litter_full = np.zeros((npts, 2, nvm, ndeep, nelements), dtype=np.float64)
    carbon_full[:, :, :3, :] = carbon
    doc_full[:, :3, :, :, :, :] = doc
    litter_full[:, :, :3, :, :] = litter_below
    altmax_lastyear = np.zeros((npts, nvm), dtype=np.float64)
    altmax_lastyear[:, PFT14] = 1.0
    altmax_ind = np.zeros((npts, nvm), dtype=np.int32)
    altmax_ind[:, PFT14] = 2
    veget_mask = np.zeros((npts, nvm), dtype=bool)
    veget_mask[:, PFT14] = True
    zi = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    zf = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    previous = soilcarbon_cryoturbation_coefficients(
        altmax_ind,
        carbon_full,
        doc_full,
        litter_full,
        altmax_lastyear,
        altmax_lastyear,
        veget_mask,
        zi,
        zf,
        dt_seconds=86400.0,
        diff_k_const=0.001 / (86400.0 * 365.0),
        bio_diff_k_const=0.0001 / (86400.0 * 365.0),
    )
    cycle, next_coeff = soilcarbon_cryoturbation_cycle(
        carbon_full,
        doc_full,
        litter_full,
        previous,
        altmax_ind,
        altmax_lastyear,
        altmax_lastyear,
        veget_mask,
        zi,
        zf,
        dt_seconds=86400.0,
        diff_k_const=0.001 / (86400.0 * 365.0),
        bio_diff_k_const=0.0001 / (86400.0 * 365.0),
    )
    pref = np.zeros(nvm, dtype=np.int32)
    pref[PFT14] = 3
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[:, PFT14] = 1.0
    args = dict(
        soilcarbon_input_doc=np.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=np.float64),
        doc_to_topsoil=np.zeros((npts, 4), dtype=np.float64),
        doc_to_subsoil=np.zeros((npts, 4), dtype=np.float64),
        wet_dep_ground=np.zeros((npts, nvm, nelements), dtype=np.float64),
        wet_dep_flood=np.zeros((npts, nvm, nelements), dtype=np.float64),
        litter_above=np.zeros((npts, 2, nvm, nelements), dtype=np.float64),
        lignin_struc_above=np.zeros((npts, nvm), dtype=np.float64),
        lignin_struc_below=np.zeros((npts, nvm, ndeep), dtype=np.float64),
        fbact_doc=np.zeros((npts, ndeep, nvm), dtype=np.float64),
        fbact=np.zeros((npts, ndeep, nvm), dtype=np.float64),
        soil_mc=np.ones((npts, ndeep, nstm), dtype=np.float64),
        soil_mc_32l=np.ones((npts, ndeep, nstm), dtype=np.float64),
        wat_flux=np.zeros((npts, ndeep, nstm), dtype=np.float64),
        soilwater_31mm=np.ones((npts, nstm), dtype=np.float64),
        runoff_per_soil=np.zeros((npts, nstm), dtype=np.float64),
        drainage_per_soil=np.zeros((npts, nstm), dtype=np.float64),
        runoff2peat=np.zeros((npts, nstm), dtype=np.float64),
        fastr=np.asarray([25.0], dtype=np.float64),
        pref_soil_veg=pref,
        veget_max=veget_max,
        flood_frac=np.zeros(npts, dtype=np.float64),
        floodcarbon_input=np.zeros((npts, nvm, NPOOL, nelements), dtype=np.float64),
        tprof=np.full((npts, ndeep, nvm), 273.15, dtype=np.float64),
        clay=np.asarray([0.3], dtype=np.float64),
        bulk_dens=np.asarray([1.65], dtype=np.float64),
        z_soil=zf,
        zf_soil_b=zf,
        flux_red=np.asarray([1.0], dtype=np.float64),
        natural=np.ones(nvm, dtype=bool),
        is_peat=np.zeros(nvm, dtype=bool),
        is_c4=np.zeros(nvm, dtype=bool),
        dt_days=1.0,
        dif_doc=np.asarray([1.0e-5], dtype=np.float64),
        nslm=ndeep,
        sro_bottom=2,
    )
    expected = soilcarbon_leak_core_step(
        cycle.carbon_32l,
        cycle.doc,
        litter_below=cycle.litter_below,
        **args,
    )
    actual = soilcarbon_leak_core_step(
        carbon_full,
        doc_full,
        litter_below=litter_full,
        ok_cryoturb=True,
        cryoturbation_coefficients=previous,
        altmax_ind=altmax_ind,
        altmax_lastyear=altmax_lastyear,
        fixed_cryoturbation_depth=altmax_lastyear,
        veget_mask=veget_mask,
        zi_soil=zi,
        **args,
    )

    assert np.allclose(np.asarray(actual.carbon_32l), np.asarray(expected.carbon_32l))
    assert np.allclose(np.asarray(actual.doc), np.asarray(expected.doc))
    assert np.allclose(np.asarray(actual.litter_below), np.asarray(expected.litter_below))
    assert np.allclose(np.asarray(actual.cryoturbation_coefficients.beta_c), np.asarray(next_coeff.beta_c))


def test_soilcarbon_leak_core_step_requires_explicit_cryoturbation_boundaries():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1359-1362."""

    npts, nvm, ndeep, ndoc, nelements, nstm = 1, 14, 2, 2, 1, 4
    pref = np.zeros(nvm, dtype=np.int32)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    with pytest.raises(ValueError, match="altmax_ind"):
        soilcarbon_leak_core_step(
            np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64),
            np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64),
            np.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=np.float64),
            np.zeros((npts, 4), dtype=np.float64),
            np.zeros((npts, 4), dtype=np.float64),
            np.zeros((npts, nvm, nelements), dtype=np.float64),
            np.zeros((npts, nvm, nelements), dtype=np.float64),
            np.zeros((npts, 2, nvm, nelements), dtype=np.float64),
            np.zeros((npts, 2, nvm, ndeep, nelements), dtype=np.float64),
            np.zeros((npts, nvm), dtype=np.float64),
            np.zeros((npts, nvm, ndeep), dtype=np.float64),
            np.zeros((npts, ndeep, nvm), dtype=np.float64),
            np.zeros((npts, ndeep, nvm), dtype=np.float64),
            np.ones((npts, ndeep, nstm), dtype=np.float64),
            np.ones((npts, ndeep, nstm), dtype=np.float64),
            np.zeros((npts, ndeep, nstm), dtype=np.float64),
            np.ones((npts, nstm), dtype=np.float64),
            np.zeros((npts, nstm), dtype=np.float64),
            np.zeros((npts, nstm), dtype=np.float64),
            np.zeros((npts, nstm), dtype=np.float64),
            np.asarray([25.0], dtype=np.float64),
            pref,
            veget_max,
            np.zeros(npts, dtype=np.float64),
            np.zeros((npts, nvm, NPOOL, nelements), dtype=np.float64),
            np.full((npts, ndeep, nvm), 273.15, dtype=np.float64),
            np.asarray([0.3], dtype=np.float64),
            np.asarray([1.65], dtype=np.float64),
            np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            np.zeros(nvm, dtype=bool),
            np.zeros(nvm, dtype=bool),
            np.zeros(nvm, dtype=bool),
            dt_days=1.0,
            ok_cryoturb=True,
            nslm=2,
            sro_bottom=2,
        )


def test_soilcarbon_leak_core_step_requires_explicit_perma_peat_boundaries():
    """Fortran: stomate_soilcarbon.f90 soilcarbon_leak lines 1375-1437."""

    npts, nvm, ndeep, ndoc, nelements, nstm = 1, 14, 2, 2, 1, 4
    pref = np.zeros(nvm, dtype=np.int32)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    with pytest.raises(ValueError, match="cmax_peat"):
        soilcarbon_leak_core_step(
            np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64),
            np.zeros((npts, nvm, ndeep, ndoc, NPOOL, nelements), dtype=np.float64),
            np.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=np.float64),
            np.zeros((npts, 4), dtype=np.float64),
            np.zeros((npts, 4), dtype=np.float64),
            np.zeros((npts, nvm, nelements), dtype=np.float64),
            np.zeros((npts, nvm, nelements), dtype=np.float64),
            np.zeros((npts, 2, nvm, nelements), dtype=np.float64),
            np.zeros((npts, 2, nvm, ndeep, nelements), dtype=np.float64),
            np.zeros((npts, nvm), dtype=np.float64),
            np.zeros((npts, nvm, ndeep), dtype=np.float64),
            np.zeros((npts, ndeep, nvm), dtype=np.float64),
            np.zeros((npts, ndeep, nvm), dtype=np.float64),
            np.ones((npts, ndeep, nstm), dtype=np.float64),
            np.ones((npts, ndeep, nstm), dtype=np.float64),
            np.zeros((npts, ndeep, nstm), dtype=np.float64),
            np.ones((npts, nstm), dtype=np.float64),
            np.zeros((npts, nstm), dtype=np.float64),
            np.zeros((npts, nstm), dtype=np.float64),
            np.zeros((npts, nstm), dtype=np.float64),
            np.asarray([25.0], dtype=np.float64),
            pref,
            veget_max,
            np.zeros(npts, dtype=np.float64),
            np.zeros((npts, nvm, NPOOL, nelements), dtype=np.float64),
            np.full((npts, ndeep, nvm), 273.15, dtype=np.float64),
            np.asarray([0.3], dtype=np.float64),
            np.asarray([1.65], dtype=np.float64),
            np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            np.zeros(nvm, dtype=bool),
            np.zeros(nvm, dtype=bool),
            np.zeros(nvm, dtype=bool),
            dt_days=1.0,
            perma_peat=True,
            nslm=2,
            sro_bottom=2,
        )
