from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    PEAT_MCF,
    PEAT_MCS,
    PEAT_MCW,
    USDA_MCF,
    USDA_MCS,
    USDA_MCW,
    hydrol_layer_moisture_content,
    hydrol_litter_grid_diagnostics,
    hydrol_litter_top_diagnostics,
    hydrol_grid_flux_aggregates,
    hydrol_evap_bare_limit_grid_beta,
    hydrol_peat_water_table_diagnostics,
    hydrol_evap_bare_limit_diagnostic,
    hydrol_shumdiag,
    hydrol_soil_layer_diagnostics,
    hydrol_soilmoist_aggregates,
    hydrol_stress_aggregates,
    hydrol_water_stress_diagnostics,
)


def test_hydrol_soil_layer_diagnostics_matches_source_threshold_formulas():
    mc = np.asarray([[0.20, 0.30, 0.40, 0.50]], dtype=np.float64)
    mcl = np.asarray([[0.18, 0.26, 0.34, 0.42]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0, 8.0], dtype=np.float64)
    njsc = np.asarray([3], dtype=np.int32)
    result = hydrol_soil_layer_diagnostics(mc=mc, mcl=mcl, dz_mm=dz_mm, njsc=njsc)

    tex = 2
    expected_sm = hydrol_layer_moisture_content(mcl, dz_mm)
    expected_smt = hydrol_layer_moisture_content(mc, dz_mm)
    expected_smw = hydrol_layer_moisture_content(np.full_like(mc, USDA_MCW[tex]), dz_mm)
    expected_smf = hydrol_layer_moisture_content(np.full_like(mc, USDA_MCF[tex]), dz_mm)
    expected_sms = hydrol_layer_moisture_content(np.full_like(mc, USDA_MCS[tex]), dz_mm)
    expected_wet = np.clip(
        (np.asarray(expected_sm) - np.asarray(expected_smw))
        * (np.asarray(expected_sms) - np.asarray(expected_smw))
        / (np.asarray(expected_sms) - np.asarray(expected_smw)) ** 2,
        0.0,
        1.0,
    )

    np.testing.assert_allclose(np.asarray(result.sm), np.asarray(expected_sm))
    np.testing.assert_allclose(np.asarray(result.smt), np.asarray(expected_smt))
    np.testing.assert_allclose(np.asarray(result.smw), np.asarray(expected_smw))
    np.testing.assert_allclose(np.asarray(result.smf), np.asarray(expected_smf))
    np.testing.assert_allclose(np.asarray(result.sms), np.asarray(expected_sms))
    np.testing.assert_allclose(np.asarray(result.sm_nostress), np.asarray(expected_smw) + 0.8 * (np.asarray(expected_smf) - np.asarray(expected_smw)))
    np.testing.assert_allclose(np.asarray(result.soil_wet_ns), expected_wet)


def test_hydrol_soil_layer_diagnostics_uses_peat_constants_for_pft14_tiles():
    mc = np.asarray([[0.30, 0.35, 0.42]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 3.0, 6.0], dtype=np.float64)
    result = hydrol_soil_layer_diagnostics(
        mc=mc,
        mcl=mc,
        dz_mm=dz_mm,
        njsc=np.asarray([2], dtype=np.int32),
        peat_hydro=True,
        soil_tile_index=4,
    )

    np.testing.assert_allclose(
        np.asarray(result.smw),
        np.asarray(hydrol_layer_moisture_content(np.full_like(mc, PEAT_MCW), dz_mm)),
    )
    np.testing.assert_allclose(
        np.asarray(result.smf),
        np.asarray(hydrol_layer_moisture_content(np.full_like(mc, PEAT_MCF), dz_mm)),
    )
    np.testing.assert_allclose(
        np.asarray(result.sms),
        np.asarray(hydrol_layer_moisture_content(np.full_like(mc, PEAT_MCS), dz_mm)),
    )


def test_hydrol_water_stress_diagnostics_old_stress_absent_and_under_mcr_gates():
    layer = hydrol_soil_layer_diagnostics(
        mc=np.asarray([[0.20, 0.30, 0.40], [0.20, 0.30, 0.40]], dtype=np.float64),
        mcl=np.asarray([[0.20, 0.30, 0.40], [0.20, 0.30, 0.40]], dtype=np.float64),
        dz_mm=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        njsc=np.asarray([3, 3], dtype=np.int32),
    )
    nroot = np.asarray(
        [
            [[0.0, 0.4, 0.6], [0.0, 0.25, 0.75]],
            [[0.0, 0.4, 0.6], [0.0, 0.25, 0.75]],
        ],
        dtype=np.float64,
    )
    result = hydrol_water_stress_diagnostics(
        layer=layer,
        nroot=nroot,
        vegetmax_soil_tile=np.asarray([[0.0, 0.5], [0.0, 0.5]], dtype=np.float64),
        is_under_mcr=np.asarray([False, True]),
        njsc=np.asarray([3, 3], dtype=np.int32),
    )

    tex = 2
    old_factor = np.clip(
        (np.asarray(layer.sm) - np.asarray(layer.smw_tmp))
        / (0.8 * (np.asarray(layer.smf_tmp) - np.asarray(layer.smw_tmp)))
        * (np.asarray(layer.smf) - np.asarray(layer.smw))
        / (np.asarray(layer.smf_tmp) - np.asarray(layer.smw_tmp)),
        0.0,
        1.0,
    )
    assert tex == 2
    expected_us_pft2 = old_factor[0] * nroot[0, 1]
    expected_us_pft2[0] = 0.0
    np.testing.assert_allclose(np.asarray(result.us[0, 1]), expected_us_pft2)
    np.testing.assert_allclose(np.asarray(result.humrelv[0, 1]), expected_us_pft2.sum())
    np.testing.assert_allclose(np.asarray(result.humrelv[:, 0]), np.zeros(2))
    np.testing.assert_allclose(np.asarray(result.us[1]), np.zeros_like(nroot[1]))
    np.testing.assert_allclose(np.asarray(result.soil_wet_ns[1]), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(result.undermcr_increment), np.asarray([0.0, 1.0]))


def test_hydrol_old_water_stress_has_finite_reverse_gradient_at_source_zero_top_layer():
    base = hydrol_soil_layer_diagnostics(
        mc=np.asarray([[0.20, 0.30, 0.40]], dtype=np.float64),
        mcl=np.asarray([[0.20, 0.30, 0.40]], dtype=np.float64),
        dz_mm=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        njsc=np.asarray([3], dtype=np.int32),
    )

    def objective(offset):
        layer = base._replace(sm=base.sm + offset)
        result = hydrol_water_stress_diagnostics(
            layer=layer,
            nroot=jnp.asarray([[[0.0, 0.4, 0.6], [0.0, 0.25, 0.75]]]),
            vegetmax_soil_tile=jnp.asarray([[0.0, 0.5]]),
            is_under_mcr=jnp.asarray([False]),
            njsc=jnp.asarray([3], dtype=jnp.int32),
            new_watstress=False,
        )
        return result.humrelv[0, 1]

    value = jnp.asarray(0.0, dtype=jnp.float64)
    forward = jax.jacfwd(objective)(value)
    reverse = jax.grad(objective)(value)

    assert np.isfinite(np.asarray(forward))
    assert np.isfinite(np.asarray(reverse))
    np.testing.assert_allclose(np.asarray(reverse), np.asarray(forward), rtol=1.0e-12, atol=1.0e-12)


def test_hydrol_water_stress_dynamic_root_requires_source_inputs():
    layer = hydrol_soil_layer_diagnostics(
        mc=np.asarray([[0.20, 0.30]], dtype=np.float64),
        mcl=np.asarray([[0.20, 0.30]], dtype=np.float64),
        dz_mm=np.asarray([0.0, 2.0], dtype=np.float64),
        njsc=np.asarray([2], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="dyn_nroot_larix"):
        hydrol_water_stress_diagnostics(
            layer=layer,
            nroot=np.asarray([[[0.0, 1.0]]], dtype=np.float64),
            vegetmax_soil_tile=np.asarray([[1.0]], dtype=np.float64),
            is_under_mcr=np.asarray([False]),
            njsc=np.asarray([2], dtype=np.int32),
            dyn_nroot_larix=True,
        )


def test_hydrol_dynamic_tree_tile6_uses_mineral_root_update_then_peat_stress():
    """hydrol.f90:6588-6601 excludes tile 6 only from tree nroot peat thresholds."""

    mc = np.asarray([[0.20, 0.24, 0.80]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    njsc = np.asarray([2], dtype=np.int32)
    layer = hydrol_soil_layer_diagnostics(
        mc=mc,
        mcl=mc,
        dz_mm=dz_mm,
        njsc=njsc,
        peat_hydro=True,
        soil_tile_index=6,
    )
    initial = np.asarray([[[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]]], dtype=np.float64)
    result = hydrol_water_stress_diagnostics(
        layer=layer,
        nroot=initial,
        vegetmax_soil_tile=np.asarray([[0.0, 1.0]], dtype=np.float64),
        is_under_mcr=np.asarray([False]),
        njsc=njsc,
        pcent=np.asarray([0.4], dtype=np.float64),
        peat_hydro=True,
        soil_tile_index=6,
        dyn_nroot_larix=True,
        is_tree=np.asarray([False, True]),
        znt=np.asarray([0.0, 0.5, 1.5], dtype=np.float64),
    )

    mineral_candidate = np.clip(
        (np.asarray(layer.sm[0]) - np.asarray(layer.smw_tmp[0]))
        / (0.4 * (np.asarray(layer.smf_tmp[0]) - np.asarray(layer.smw_tmp[0])))
        * (np.asarray(layer.smf[0]) - np.asarray(layer.smw[0]))
        / (np.asarray(layer.smf_tmp[0]) - np.asarray(layer.smw_tmp[0])),
        0.0,
        1.0,
    )
    mineral_candidate[0] = 0.0
    expected_nroot = mineral_candidate / mineral_candidate.sum()
    np.testing.assert_allclose(np.asarray(result.nroot[0, 1]), expected_nroot)

    peat_factor = np.clip(
        (np.asarray(layer.sm[0]) - np.asarray(layer.smw_tmp[0]))
        / (0.8 * (np.asarray(layer.smf_tmp[0]) - np.asarray(layer.smw_tmp[0])))
        * (np.asarray(layer.smf[0]) - np.asarray(layer.smw[0]))
        / (np.asarray(layer.smf_tmp[0]) - np.asarray(layer.smw_tmp[0])),
        0.0,
        1.0,
    )
    peat_factor[0] = 0.0
    np.testing.assert_allclose(np.asarray(result.us[0, 1]), peat_factor * expected_nroot)


def test_hydrol_dynamic_root_uses_legacy_threshold_when_new_watstress_enabled():
    """hydrol.f90:6584-6601 is independent of the NEW_WATSTRESS branch at 6648."""

    mc = np.asarray([[0.058, 0.062, 0.068]], dtype=np.float64)
    layer = hydrol_soil_layer_diagnostics(
        mc=mc,
        mcl=mc,
        dz_mm=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        njsc=np.asarray([2], dtype=np.int32),
    )
    kwargs = dict(
        layer=layer,
        nroot=np.asarray([[[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]]], dtype=np.float64),
        vegetmax_soil_tile=np.asarray([[0.0, 1.0]], dtype=np.float64),
        is_under_mcr=np.asarray([False]),
        njsc=np.asarray([2], dtype=np.int32),
        dyn_nroot_larix=True,
        is_tree=np.asarray([False, True]),
        znt=np.asarray([0.0, 0.5, 1.5], dtype=np.float64),
    )
    legacy = hydrol_water_stress_diagnostics(**kwargs, new_watstress=False)
    exponential = hydrol_water_stress_diagnostics(**kwargs, new_watstress=True, alpha_watstress=3.0)

    np.testing.assert_allclose(np.asarray(exponential.nroot), np.asarray(legacy.nroot))
    assert not np.allclose(np.asarray(exponential.us), np.asarray(legacy.us))


def test_hydrol_dynamic_root_normalizes_positive_candidate_below_min_sechiba():
    """hydrol.f90:6595-6601 has no min_sechiba gate on nroot_tmp."""

    dz_mm = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    njsc = np.asarray([2], dtype=np.int32)
    mcw = USDA_MCW[1]
    mc = np.asarray([[mcw, mcw + 2.0e-10, mcw + 8.0e-10]], dtype=np.float64)
    layer = hydrol_soil_layer_diagnostics(mc=mc, mcl=mc, dz_mm=dz_mm, njsc=njsc)
    assert np.all((np.asarray(layer.sm[0, 1:]) - np.asarray(layer.smw_tmp[0, 1:])) < 1.0e-8)

    result = hydrol_water_stress_diagnostics(
        layer=layer,
        nroot=np.asarray([[[0.0, 0.5, 0.5], [0.0, 0.9, 0.1]]], dtype=np.float64),
        vegetmax_soil_tile=np.asarray([[0.0, 1.0]], dtype=np.float64),
        is_under_mcr=np.asarray([False]),
        njsc=njsc,
        dyn_nroot_larix=True,
        is_tree=np.asarray([False, True]),
        znt=np.asarray([0.0, 0.5, 1.5], dtype=np.float64),
    )

    candidate = np.clip(
        (np.asarray(layer.sm[0]) - np.asarray(layer.smw_tmp[0]))
        / (0.8 * (np.asarray(layer.smf_tmp[0]) - np.asarray(layer.smw_tmp[0])))
        * (np.asarray(layer.smf[0]) - np.asarray(layer.smw[0]))
        / (np.asarray(layer.smf_tmp[0]) - np.asarray(layer.smw_tmp[0])),
        0.0,
        1.0,
    )
    candidate[0] = 0.0
    expected = candidate / candidate.sum()
    np.testing.assert_allclose(np.asarray(result.nroot[0, 1]), expected, rtol=1.0e-12, atol=0.0)
    assert not np.allclose(np.asarray(result.nroot[0, 1]), np.asarray([0.0, 0.9, 0.1]))


def test_hydrol_soilmoist_aggregates_match_tile_weighted_source_exports():
    mc = np.asarray(
        [
            [
                [0.20, 0.30],
                [0.25, 0.35],
                [0.30, 0.40],
            ]
        ],
        dtype=np.float64,
    )
    mcl = mc - 0.02
    soiltile = np.asarray([[0.25, 0.75]], dtype=np.float64)
    vegtot = np.asarray([0.8], dtype=np.float64)
    vegtot_old = np.asarray([0.6], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)

    result = hydrol_soilmoist_aggregates(
        mc=mc,
        mcl=mcl,
        soiltile=soiltile,
        vegtot=vegtot,
        vegtot_old=vegtot_old,
        dz_mm=dz_mm,
    )

    expected_soilmoist = (
        soiltile[0, 0] * np.asarray(hydrol_layer_moisture_content(mc[:, :, 0], dz_mm))
        + soiltile[0, 1] * np.asarray(hydrol_layer_moisture_content(mc[:, :, 1], dz_mm))
    ) * vegtot[:, None]
    expected_liquid = (
        soiltile[0, 0] * np.asarray(hydrol_layer_moisture_content(mcl[:, :, 0], dz_mm))
        + soiltile[0, 1] * np.asarray(hydrol_layer_moisture_content(mcl[:, :, 1], dz_mm))
    ) * vegtot_old[:, None]
    np.testing.assert_allclose(np.asarray(result.soilmoist), expected_soilmoist)
    np.testing.assert_allclose(np.asarray(result.soilmoist_liquid), expected_liquid)
    np.testing.assert_allclose(np.asarray(result.mc_layh), np.sum(mc * soiltile[:, None, :] * vegtot[:, None, None], axis=2))
    np.testing.assert_allclose(np.asarray(result.mcl_layh), np.sum(mcl * soiltile[:, None, :] * vegtot[:, None, None], axis=2))
    np.testing.assert_allclose(np.asarray(result.mcl_layh_s), mc)


def test_hydrol_shumdiag_matches_mineral_and_peat_tile_rules():
    npts, nslm, nstm = 1, 3, 6
    mc = np.zeros((npts, nslm, nstm), dtype=np.float64)
    for jst in range(nstm):
        mc[:, :, jst] = np.asarray([[0.20, 0.30, 0.40]], dtype=np.float64) + 0.02 * jst
    soiltile = np.asarray([[0.2, 0.2, 0.1, 0.25, 0.05, 0.2]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    dh_mm = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)
    soilmoist = np.asarray([[0.4, 0.9, 1.2]], dtype=np.float64)
    soil_wet_ns = np.full((npts, nslm, nstm), 0.5, dtype=np.float64)
    result = hydrol_shumdiag(
        mc=mc,
        soil_wet_ns=soil_wet_ns,
        soilmoist=soilmoist,
        soiltile=soiltile,
        dh_mm=dh_mm,
        dz_mm=dz_mm,
        njsc=np.asarray([2], dtype=np.int32),
        peat_hydro=True,
        agri_peat=True,
        tides=True,
    )

    mineral_scale = (USDA_MCS[1] - USDA_MCW[1]) / (USDA_MCF[1] - USDA_MCW[1])
    peat_scale = (PEAT_MCS - PEAT_MCW) / (PEAT_MCF - PEAT_MCW)
    expected = np.zeros((npts, nslm), dtype=np.float64)
    for jst in range(nstm):
        scale = peat_scale if jst + 1 in (4, 5, 6) else mineral_scale
        expected = np.clip(expected + soil_wet_ns[:, :, jst] * soiltile[:, jst, None] * scale, 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(result.shumdiag), expected)
    np.testing.assert_allclose(
        np.asarray(result.shumdiag_peat),
        np.clip(np.asarray(hydrol_layer_moisture_content(mc[:, :, 3], dz_mm)) / dh_mm[None, :], 0.0, 1.0),
    )
    np.testing.assert_allclose(
        np.asarray(result.shumdiag_croppeat),
        np.clip(np.asarray(hydrol_layer_moisture_content(mc[:, :, 4], dz_mm)) / dh_mm[None, :], 0.0, 1.0),
    )
    np.testing.assert_allclose(
        np.asarray(result.shumdiag_man),
        np.clip(np.asarray(hydrol_layer_moisture_content(mc[:, :, 5], dz_mm)) / dh_mm[None, :], 0.0, 1.0),
    )
    np.testing.assert_allclose(np.asarray(result.shumdiag_perma), np.clip(soilmoist / dh_mm[None, :], 0.0, 1.0))


def test_hydrol_stress_aggregates_sum_humrel_and_mask_absent_vegstress():
    humrelv = np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float64)
    vegstressv = np.asarray([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float64)
    result = hydrol_stress_aggregates(
        humrelv=humrelv,
        vegstressv=vegstressv,
        veget_max=np.asarray([[0.0, 0.2]], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.humrel), np.asarray([[0.3, 0.7]]))
    np.testing.assert_allclose(np.asarray(result.vegstress), np.asarray([[0.0, 1.5]]))


def _manual_top_partial(moisture, dz_mm, max_layer):
    total = dz_mm[1] * (3.0 * moisture[:, 0] + moisture[:, 1]) / 8.0
    for jsl in range(1, max_layer):
        total += dz_mm[jsl] * (3.0 * moisture[:, jsl] + moisture[:, jsl - 1]) / 8.0
        total += dz_mm[jsl + 1] * (3.0 * moisture[:, jsl] + moisture[:, jsl + 1]) / 8.0
    return total


def test_hydrol_litter_top_diagnostics_match_source_top_layer_sums_and_peat_above():
    npts, nslm, nstm = 1, 7, 6
    dz_mm = np.asarray([0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64)
    dh_mm = np.asarray([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    mc = np.zeros((npts, nslm, nstm), dtype=np.float64)
    for jst in range(nstm):
        mc[:, :, jst] = np.linspace(0.20, 0.50, nslm, dtype=np.float64)[None, :] + 0.01 * jst

    result = hydrol_litter_top_diagnostics(
        mc=mc,
        dz_mm=dz_mm,
        dh_mm=dh_mm,
        njsc=np.asarray([2], dtype=np.int32),
        peat_hydro=True,
        agri_peat=True,
        tides=True,
    )

    expected_litter_tile4 = _manual_top_partial(mc[:, :, 3], dz_mm, 4)
    expected_trampling_tile3 = _manual_top_partial(mc[:, :, 2], dz_mm, 6)
    np.testing.assert_allclose(np.asarray(result.tmc_litter[:, 3]), expected_litter_tile4)
    np.testing.assert_allclose(np.asarray(result.tmc_trampling[:, 2]), expected_trampling_tile3)
    np.testing.assert_allclose(np.asarray(result.tmc_topgrass), expected_trampling_tile3 / (np.sum(dz_mm[:6]) + dz_mm[6] / 2.0))
    depth4 = np.sum(dh_mm[:4])
    np.testing.assert_allclose(np.asarray(result.mc_peat_above), np.asarray(result.tmc_litter[:, 3]) / depth4)
    np.testing.assert_allclose(np.asarray(result.mc_croppeat_above), np.asarray(result.tmc_litter[:, 4]) / depth4)
    np.testing.assert_allclose(np.asarray(result.mc_man_above), np.asarray(result.tmc_litter[:, 5]) / depth4)
    assert np.all(np.asarray(result.soil_wet_litter) >= 0.0)
    assert np.all(np.asarray(result.soil_wet_litter) <= 1.0)


def test_hydrol_peat_water_table_diagnostics_match_source_peat_and_tide_branches():
    npts, nslm, nstm = 1, 5, 6
    dz_mm = np.asarray([0.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    dh_mm = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    mc = np.full((npts, nslm, nstm), 0.35, dtype=np.float64)
    mcl = mc - 0.05
    mc[:, :, 3] = np.asarray([[0.20, 0.30, 0.45, 0.60, 0.75]], dtype=np.float64)
    mcl[:, :, 3] = np.asarray([[0.18, 0.28, 0.42, 0.55, 0.70]], dtype=np.float64)
    mc[:, :, 5] = np.asarray([[0.25, 0.35, 0.50, 0.65, 0.80]], dtype=np.float64)
    soiltile = np.asarray([[0.1, 0.1, 0.1, 0.3, 0.0, 0.4]], dtype=np.float64)

    result = hydrol_peat_water_table_diagnostics(
        mc=mc,
        mcl=mcl,
        soiltile=soiltile,
        dz_mm=dz_mm,
        dh_mm=dh_mm,
        wt_ab=np.asarray([12.0], dtype=np.float64),
        peat_hydro=True,
        tides=False,
        liqlayers=3,
        numlayers=5,
    )

    tmcs_peat = dz_mm[1] * 4.0 * PEAT_MCS / 8.0
    tmcs_peat += dz_mm[1] * 4.0 * PEAT_MCS / 8.0 + dz_mm[2] * 4.0 * PEAT_MCS / 8.0
    tmcs_peat += dz_mm[2] * 4.0 * PEAT_MCS / 8.0 + dz_mm[3] * 4.0 * PEAT_MCS / 8.0
    tile4 = mcl[:, :, 3]
    tmcl_peat = dz_mm[1] * (3.0 * tile4[:, 0] + tile4[:, 1]) / 8.0
    tmcl_peat += dz_mm[1] * (3.0 * tile4[:, 1] + tile4[:, 0]) / 8.0 + dz_mm[2] * (3.0 * tile4[:, 1] + tile4[:, 2]) / 8.0
    tmcl_peat += dz_mm[2] * (3.0 * tile4[:, 2] + tile4[:, 1]) / 8.0 + dz_mm[3] * (3.0 * tile4[:, 2] + tile4[:, 3]) / 8.0
    h_eau = np.clip((mc[:, :, 3] - 0.15) / (PEAT_MCS - 0.15), 0.0, 1.0)
    expected_wtp = 2000.0 - (np.sum(h_eau * dz_mm[None, :], axis=1) + 12.0)

    np.testing.assert_allclose(np.asarray(result.liqwt_ratio), tmcl_peat / tmcs_peat)
    np.testing.assert_allclose(np.asarray(result.wtp), expected_wtp)
    assert result.fwet_new is None

    tide = hydrol_peat_water_table_diagnostics(
        mc=mc,
        mcl=mcl,
        soiltile=soiltile,
        dz_mm=dz_mm,
        dh_mm=dh_mm,
        wt_ab=np.asarray([12.0], dtype=np.float64),
        peat_hydro=True,
        tides=True,
        liqlayers=3,
        numlayers=5,
    )
    tide_h_eau = np.clip((mc[:, :, 5] - 0.15) / (PEAT_MCS - 0.15), 0.0, 1.0)
    expected_tide_wtp = 2000.0 - (np.sum(tide_h_eau * dz_mm[None, :], axis=1) + 12.0)
    np.testing.assert_allclose(np.asarray(tide.wtp), expected_tide_wtp)


def test_hydrol_peat_water_table_diagnostics_topmodel_fwet_requires_parameters_and_matches_source():
    npts, nslm, nstm = 1, 7, 2
    mc = np.full((npts, nslm, nstm), 0.45, dtype=np.float64)
    mcl = mc.copy()
    soiltile = np.asarray([[0.4, 0.6]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64)
    dh_mm = np.asarray([1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64)
    temp_hydro = np.full((npts, nslm), 274.0, dtype=np.float64)

    with pytest.raises(ValueError, match="param_vp"):
        hydrol_peat_water_table_diagnostics(
            mc=mc,
            mcl=mcl,
            soiltile=soiltile,
            dz_mm=dz_mm,
            dh_mm=dh_mm,
            wt_ab=np.zeros(npts, dtype=np.float64),
            topmodel_new=True,
            temp_hydro=temp_hydro,
            mcs=np.asarray([0.9], dtype=np.float64),
        )

    result = hydrol_peat_water_table_diagnostics(
        mc=mc,
        mcl=mcl,
        soiltile=soiltile,
        dz_mm=dz_mm,
        dh_mm=dh_mm,
        wt_ab=np.zeros(npts, dtype=np.float64),
        topmodel_new=True,
        temp_hydro=temp_hydro,
        mcs=np.asarray([0.9], dtype=np.float64),
        param_vp=np.asarray([2.0], dtype=np.float64),
        param_kp=np.asarray([0.7], dtype=np.float64),
        param_qp=np.asarray([-0.3], dtype=np.float64),
        param_fmax=np.asarray([0.8], dtype=np.float64),
    )

    unfrozen_depth = np.sum(dh_mm)
    tile_wtp = np.sum(mc[:, :, 0] / 0.9 * dh_mm[None, :], axis=1) / 1000.0 - unfrozen_depth / 1000.0
    tile_wtp = np.where(tile_wtp > 0.0, tile_wtp * 0.9, tile_wtp)
    expected_meanwt = np.sum(soiltile * np.repeat(tile_wtp[:, None], nstm, axis=1), axis=1)
    expected_fwet = (1.0 + 2.0 * np.exp(-0.7 * (expected_meanwt - (-0.3)))) ** (-1.0 / 2.0)
    expected_fwet = np.maximum(0.0, np.minimum(0.8, expected_fwet))

    np.testing.assert_allclose(np.asarray(result.meanwt), expected_meanwt)
    np.testing.assert_allclose(np.asarray(result.fwet_new), expected_fwet)


def test_hydrol_grid_flux_aggregates_match_hydrol_diag_soil_mask_and_no_veg_fallback():
    result = hydrol_grid_flux_aggregates(
        ae_ns=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        dr_ns=np.asarray([[0.5, 1.0], [2.0, 3.0]], dtype=np.float64),
        ru_ns=np.asarray([[1.5, 2.0], [4.0, 5.0]], dtype=np.float64),
        tmc=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        soiltile=np.asarray([[0.25, 0.75], [0.4, 0.6]], dtype=np.float64),
        vegtot=np.asarray([0.8, 0.0], dtype=np.float64),
        frac_bare_ns=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        vevapnu=np.asarray([0.05, 0.06], dtype=np.float64),
        subsinksoil=np.asarray([0.2, 0.3], dtype=np.float64),
        tot_melt=np.asarray([1.0, 2.0], dtype=np.float64),
        irrigation=np.asarray([0.1, 0.2], dtype=np.float64),
        returnflow=np.asarray([0.3, 0.4], dtype=np.float64),
        reinfiltration=np.asarray([0.5, 0.6], dtype=np.float64),
        mask_soiltile=np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.ae_ns[0]), np.asarray([0.1, 0.0]))
    np.testing.assert_allclose(np.asarray(result.drainage[0]), 0.8 * 0.25 * 0.5)
    np.testing.assert_allclose(np.asarray(result.runoff[0]), 0.8 * 0.25 * 1.5)
    np.testing.assert_allclose(np.asarray(result.humtot[0]), 0.8 * 0.25 * 10.0)
    np.testing.assert_allclose(np.asarray(result.runoff[1]), 2.0 + 0.2 + 0.4 + 0.6)
    np.testing.assert_allclose(np.asarray(result.drainage[1]), 0.0)
    np.testing.assert_allclose(np.asarray(result.vevapnu), np.asarray([0.05 + 0.2 * 0.8, 0.06]))


def test_hydrol_litter_grid_diagnostics_match_table_lookup_and_drysoil_formula():
    tmc_litter = np.asarray([[1.0, 2.0]], dtype=np.float64)
    litter = hydrol_litter_top_diagnostics(
        mc=np.repeat(np.linspace(0.2, 0.4, 7, dtype=np.float64)[None, :, None], 2, axis=2),
        dz_mm=np.asarray([0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64),
        dh_mm=np.ones(7, dtype=np.float64),
        njsc=np.asarray([2], dtype=np.int32),
    )
    litter = litter._replace(
        tmc_litter=tmc_litter,
        tmc_litter_res=np.asarray([[0.5, 1.0]], dtype=np.float64),
        tmc_litter_sat=np.asarray([[2.5, 3.0]], dtype=np.float64),
        soil_wet_litter=np.asarray([[0.25, 0.75]], dtype=np.float64),
        tmc_litter_awet=np.asarray([[3.0, 4.0]], dtype=np.float64),
        tmc_litter_adry=np.asarray([[0.5, 1.0]], dtype=np.float64),
    )
    k_lin = np.tile(np.arange(1, 52, dtype=np.float64)[:, None], (1, 1))
    result = hydrol_litter_grid_diagnostics(
        litter=litter,
        soiltile=np.asarray([[0.4, 0.6]], dtype=np.float64),
        vegtot=np.asarray([0.8], dtype=np.float64),
        njsc=np.asarray([2], dtype=np.int32),
        k_lin_layer1=k_lin,
        ks=np.asarray([10.0], dtype=np.float64),
    )

    idx0 = int((51 - 1) * ((1.0 - 0.5) / (2.5 - 0.5)) + 1)
    idx1 = int((51 - 1) * ((2.0 - 1.0) / (3.0 - 1.0)) + 1)
    idx0 = max(min(idx0, 50), 1)
    idx1 = max(min(idx1, 50), 1)
    expected_k = 0.8 * (0.4 * np.sqrt(k_lin[idx0 - 1, 0] * 10.0) + 0.6 * np.sqrt(k_lin[idx1 - 1, 0] * 10.0))
    wet = 3.0 * 0.4 + 4.0 * 0.6
    dry = 0.5 * 0.4 + 1.0 * 0.6
    mean = 1.0 * 0.4 + 2.0 * 0.6
    expected_drysoil = 1.0 + max(min((dry - mean) / (wet - dry), 0.0), -1.0)
    np.testing.assert_allclose(np.asarray(result.k_litt), np.asarray([expected_k]))
    np.testing.assert_allclose(np.asarray(result.litterhumdiag), np.asarray([0.25 * 0.4 + 0.75 * 0.6]))
    np.testing.assert_allclose(np.asarray(result.drysoil_frac), np.asarray([expected_drysoil]))


def test_hydrol_evap_bare_limit_diagnostic_matches_rsoil_beta_path():
    result = hydrol_evap_bare_limit_diagnostic(
        tmcint=np.asarray([10.0, 8.0], dtype=np.float64),
        tmc_dummy=np.asarray([[7.0, 9.0, 10.0], [6.0, 7.0, 7.5]], dtype=np.float64),
        flux_bottom=np.asarray([[1.0, 0.5, 0.0], [0.5, 0.25, 0.1]], dtype=np.float64),
        mask_soiltile=np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64),
        frac_bare_ns=np.asarray([[0.4, 0.8, 0.2], [0.5, 0.5, 0.5]], dtype=np.float64),
        vegtot=np.asarray([0.7, 0.0], dtype=np.float64),
        evapot=np.asarray([2.0, 1.0], dtype=np.float64),
        tmc_litter=np.ones((2, 3), dtype=np.float64),
        tmc_litter_wilt=np.zeros((2, 3), dtype=np.float64),
        tmc_litter_res=np.zeros((2, 3), dtype=np.float64),
        is_under_mcr=np.asarray([[False, True, False], [False, False, False]]),
        do_rsoil=True,
    )

    water_budget = np.asarray([[2.0, 0.5, -0.0], [1.5, 0.0, 0.4]], dtype=np.float64)
    bare = np.asarray([[0.8, 0.4, -0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    expected = np.asarray([[0.4, 0.0, -0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(result.water_budget_evap), water_budget)
    np.testing.assert_allclose(np.asarray(result.bare_weighted_evap), bare)
    np.testing.assert_allclose(np.asarray(result.evap_bare_lim), np.asarray([0.28, 0.0], dtype=np.float64))
    np.testing.assert_allclose(np.asarray(result.evap_bare_lim_ns), expected)


def test_hydrol_evap_bare_limit_diagnostic_matches_litter_limited_path():
    result = hydrol_evap_bare_limit_diagnostic(
        tmcint=np.asarray([10.0], dtype=np.float64),
        tmc_dummy=np.asarray([[7.0, 7.0, 7.0, 12.0]], dtype=np.float64),
        flux_bottom=np.asarray([[1.0, 1.0, 1.0, 0.0]], dtype=np.float64),
        mask_soiltile=np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        frac_bare_ns=np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        vegtot=np.asarray([1.0], dtype=np.float64),
        evapot=np.asarray([2.0], dtype=np.float64),
        tmc_litter=np.asarray([[3.0, 1.5, 0.5, 3.0]], dtype=np.float64),
        tmc_litter_wilt=np.asarray([[2.0, 2.0, 2.0, 2.0]], dtype=np.float64),
        tmc_litter_res=np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        is_under_mcr=np.asarray([[False, False, False, False]]),
        do_rsoil=False,
    )

    # water budget is [2, 2, 2, -2]; Fortran clips beta to [0, 1].
    np.testing.assert_allclose(
        np.asarray(result.evap_bare_lim_ns),
        np.asarray([[1.0, 0.5, 0.0, 0.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(np.asarray(result.evap_bare_lim), np.asarray([1.5], dtype=np.float64))


def test_hydrol_evap_bare_limit_diagnostic_applies_grid_beta_floor_sequence():
    result = hydrol_evap_bare_limit_diagnostic(
        tmcint=np.asarray([10.0], dtype=np.float64),
        tmc_dummy=np.asarray([[9.999, 9.999]], dtype=np.float64),
        flux_bottom=np.asarray([[0.0, 0.0]], dtype=np.float64),
        mask_soiltile=np.asarray([[0.5, 0.5]], dtype=np.float64),
        frac_bare_ns=np.asarray([[1.0, 1.0]], dtype=np.float64),
        vegtot=np.asarray([1.0], dtype=np.float64),
        evapot=np.asarray([10.0], dtype=np.float64),
        tmc_litter=np.asarray([[2.0, 2.0]], dtype=np.float64),
        tmc_litter_wilt=np.asarray([[1.0, 1.0]], dtype=np.float64),
        tmc_litter_res=np.asarray([[0.5, 0.5]], dtype=np.float64),
        is_under_mcr=np.asarray([[False, False]]),
    )

    np.testing.assert_allclose(np.asarray(result.evap_bare_lim_ns), np.zeros((1, 2), dtype=np.float64))
    np.testing.assert_allclose(np.asarray(result.evap_bare_lim), np.zeros(1, dtype=np.float64))


def test_hydrol_evap_bare_limit_grid_beta_matches_fortran_weighted_sum():
    result = hydrol_evap_bare_limit_grid_beta(
        evap_bare_lim_ns=np.asarray([[0.2, 0.8], [0.5, 0.25]], dtype=np.float64),
        vegtot=np.asarray([0.75, 0.0], dtype=np.float64),
        soiltile=np.asarray([[0.4, 0.6], [0.5, 0.5]], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result), np.asarray([0.75 * (0.2 * 0.4 + 0.8 * 0.6), 0.0]))


def test_hydrol_evap_bare_limit_diagnostic_accepts_tile_tmcint():
    result = hydrol_evap_bare_limit_diagnostic(
        tmcint=np.asarray([[10.0, 8.0]], dtype=np.float64),
        tmc_dummy=np.asarray([[7.0, 6.0]], dtype=np.float64),
        flux_bottom=np.asarray([[1.0, 0.5]], dtype=np.float64),
        mask_soiltile=np.asarray([[1.0, 1.0]], dtype=np.float64),
        frac_bare_ns=np.asarray([[1.0, 1.0]], dtype=np.float64),
        vegtot=np.asarray([1.0], dtype=np.float64),
        evapot=np.asarray([4.0], dtype=np.float64),
        tmc_litter=np.asarray([[2.0, 2.0]], dtype=np.float64),
        tmc_litter_wilt=np.asarray([[1.0, 1.0]], dtype=np.float64),
        tmc_litter_res=np.asarray([[0.5, 0.5]], dtype=np.float64),
        is_under_mcr=np.asarray([[False, False]]),
    )

    np.testing.assert_allclose(np.asarray(result.evap_bare_lim_ns), np.asarray([[0.5, 0.375]], dtype=np.float64))


def test_hydrol_evap_bare_limit_diagnostic_requires_matching_tile_shapes():
    with pytest.raises(ValueError, match="tile diagnostics"):
        hydrol_evap_bare_limit_diagnostic(
            tmcint=np.asarray([1.0], dtype=np.float64),
            tmc_dummy=np.ones((1, 2), dtype=np.float64),
            flux_bottom=np.ones((1, 1), dtype=np.float64),
            mask_soiltile=np.ones((1, 2), dtype=np.float64),
            frac_bare_ns=np.ones((1, 2), dtype=np.float64),
            vegtot=np.asarray([1.0], dtype=np.float64),
            evapot=np.asarray([1.0], dtype=np.float64),
            tmc_litter=np.ones((1, 2), dtype=np.float64),
            tmc_litter_wilt=np.zeros((1, 2), dtype=np.float64),
            tmc_litter_res=np.zeros((1, 2), dtype=np.float64),
            is_under_mcr=np.zeros((1, 2), dtype=bool),
        )
