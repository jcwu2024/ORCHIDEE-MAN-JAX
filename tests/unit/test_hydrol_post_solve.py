from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (
    PEAT_MCS,
    drainage_correction,
    hydrol_effective_water_table_depth,
    hydrol_correct_negative_runoff,
    hydrol_force_water_table_saturation,
    hydrol_layer_moisture_content,
    hydrol_liquid_redistribution_check,
    hydrol_mc_to_mcl,
    hydrol_mcl_to_mc_after_solve,
    hydrol_soil_flux_diagnostics,
    hydrol_route_over_mcs2,
    hydrol_after_under_mcr_runoff_peat_tide_routing,
    hydrol_runoff_peat_post_loop_reinjection,
    hydrol_soil_clip_over_mcs2,
    hydrol_soil_smooth_under_mcr,
    hydrol_total_moisture_content,
)


def test_hydrol_layer_and_total_moisture_content_match_source_weights():
    moisture = np.asarray([[0.20, 0.30, 0.40, 0.50]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float64)

    layer = np.asarray(hydrol_layer_moisture_content(moisture, dz_mm))

    expected = np.asarray(
        [
            [
                2.0 * (3.0 * 0.20 + 0.30) / 8.0,
                2.0 * (3.0 * 0.30 + 0.20) / 8.0 + 4.0 * (3.0 * 0.30 + 0.40) / 8.0,
                4.0 * (3.0 * 0.40 + 0.30) / 8.0 + 6.0 * (3.0 * 0.40 + 0.50) / 8.0,
                6.0 * (3.0 * 0.50 + 0.40) / 8.0,
            ]
        ],
        dtype=np.float64,
    )

    assert np.allclose(layer, expected)
    assert np.allclose(np.asarray(hydrol_total_moisture_content(moisture, dz_mm)), expected.sum(axis=1))


def test_hydrol_mc_to_mcl_and_post_solve_mc_update_match_source_formula():
    mc = np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64)
    profil_froz = np.asarray([[0.0, 0.25, 1.0]], dtype=np.float64)
    mcr = np.asarray([0.05], dtype=np.float64)

    mcl = np.asarray(hydrol_mc_to_mcl(mc, profil_froz, mcr))
    expected_mcl = np.minimum(mc, mcr[:, None] + (1.0 - profil_froz) * (mc - mcr[:, None]))

    assert np.allclose(mcl, expected_mcl)

    mcl_after = np.asarray([[0.11, 0.16, 0.06]], dtype=np.float64)
    updated_mc = np.asarray(hydrol_mcl_to_mc_after_solve(mcl_after, mc, profil_froz, mcr))
    expected_mc = np.maximum(mcl_after, mcl_after + profil_froz * (mc - mcr[:, None]))

    assert np.allclose(updated_mc, expected_mc)
    assert np.all(updated_mc >= mcl_after)


def test_hydrol_liquid_redistribution_check_and_correction_chain():
    tmci = np.asarray([10.0, 8.0], dtype=np.float64)
    tmcf = np.asarray([9.0, 7.5], dtype=np.float64)
    flux_top = np.asarray([0.2, 0.1], dtype=np.float64)
    dr_ns = np.asarray([0.3, 0.4], dtype=np.float64)
    rootsink = np.asarray([[0.1, 0.2], [0.05, 0.15]], dtype=np.float64)

    check = np.asarray(hydrol_liquid_redistribution_check(tmci, tmcf, flux_top, dr_ns, rootsink))
    expected = tmcf - (tmci - flux_top - dr_ns - rootsink.sum(axis=1))

    assert np.allclose(check, expected)


def test_hydrol_soil_flux_diagnostics_matches_source_upward_budget():
    mcl_before = np.asarray([[0.30, 0.32, 0.34, 0.36]], dtype=np.float64)
    mcl_after = np.asarray([[0.29, 0.33, 0.35, 0.37]], dtype=np.float64)
    rootsink = np.asarray([[0.01, 0.02, 0.03, 0.04]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 3.0, 6.0, 8.0], dtype=np.float64)
    dr_ns = np.asarray([0.12], dtype=np.float64)
    flux_top = np.asarray([-0.36], dtype=np.float64)

    result = hydrol_soil_flux_diagnostics(
        mcl_after=mcl_after,
        mcl_before=mcl_before,
        dr_ns=dr_ns,
        rootsink=rootsink,
        dz_mm=dz_mm,
        flux_top=flux_top,
    )

    delta = mcl_after - mcl_before
    expected = np.zeros_like(mcl_after)
    expected[:, 3] = dr_ns
    expected[:, 2] = expected[:, 3] + (delta[:, 2] + 3.0 * delta[:, 3]) * dz_mm[3] / 8.0 + rootsink[:, 3]
    expected[:, 1] = (
        expected[:, 2]
        + (delta[:, 1] + 3.0 * delta[:, 2]) * dz_mm[2] / 8.0
        + rootsink[:, 2]
        + (3.0 * delta[:, 2] + delta[:, 3]) * dz_mm[3] / 8.0
    )
    expected[:, 0] = (
        expected[:, 1]
        + (delta[:, 0] + 3.0 * delta[:, 1]) * dz_mm[1] / 8.0
        + rootsink[:, 1]
        + (3.0 * delta[:, 1] + delta[:, 2]) * dz_mm[2] / 8.0
    )

    np.testing.assert_allclose(np.asarray(result.qflux), expected)
    np.testing.assert_allclose(np.asarray(result.surface_balance), [0.0], atol=1.0e-12)


def test_hydrol_soil_clip_over_mcs2_removes_excess_and_integrates_correction():
    mc = np.asarray([[0.30, 0.50, 0.45]], dtype=np.float64)
    mcs = np.asarray([0.40], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)

    result = hydrol_soil_clip_over_mcs2(mc=mc, mcs=mcs, dz_mm=dz_mm)

    expected_excess = np.asarray([[0.0, 0.10, 0.05]], dtype=np.float64)
    expected_mc = mc - expected_excess
    expected_corr = np.asarray(hydrol_total_moisture_content(expected_excess, dz_mm))

    assert np.allclose(np.asarray(result.mc), expected_mc)
    assert np.allclose(np.asarray(result.excess), expected_excess)
    assert np.allclose(np.asarray(result.rudr_corr), expected_corr)
    assert np.array_equal(np.asarray(result.is_over_mcs), np.asarray([False]))


def test_hydrol_route_over_mcs2_sends_excess_to_runoff_or_drainage_by_source_condition():
    mc = np.asarray([[0.30, 0.50, 0.45], [0.32, 0.42, 0.50]], dtype=np.float64)
    mcs = np.asarray([0.40, 0.41], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    ru_ns = np.asarray([0.10, 0.20], dtype=np.float64)
    dr_ns = np.asarray([0.30, 0.40], dtype=np.float64)
    free_drain_coef = np.asarray([0.4, 0.5], dtype=np.float64)

    result = hydrol_route_over_mcs2(
        mc=mc,
        mcs=mcs,
        dz_mm=dz_mm,
        ru_ns=ru_ns,
        dr_ns=dr_ns,
        free_drain_coef=free_drain_coef,
        ok_freeze_cwrr=False,
        check_cwrr2=True,
    )

    expected_excess = np.maximum(mc - mcs[:, None], 0.0)
    expected_corr = np.asarray(hydrol_total_moisture_content(expected_excess, dz_mm))
    np.testing.assert_allclose(np.asarray(result.excess), expected_excess)
    np.testing.assert_allclose(np.asarray(result.ru_corr_ns), np.asarray([expected_corr[0], 0.0]))
    np.testing.assert_allclose(np.asarray(result.dr_corr_ns), np.asarray([0.0, expected_corr[1]]))
    np.testing.assert_allclose(np.asarray(result.ru_ns), ru_ns + np.asarray([expected_corr[0], 0.0]))
    np.testing.assert_allclose(np.asarray(result.dr_ns), dr_ns + np.asarray([0.0, expected_corr[1]]))
    np.testing.assert_allclose(np.asarray(result.check_over_ns), np.zeros(2), atol=1.0e-15)

    frozen = hydrol_route_over_mcs2(
        mc=mc,
        mcs=mcs,
        dz_mm=dz_mm,
        ru_ns=ru_ns,
        dr_ns=dr_ns,
        free_drain_coef=free_drain_coef,
        ok_freeze_cwrr=True,
    )
    np.testing.assert_allclose(np.asarray(frozen.dr_corr_ns), np.zeros(2))
    np.testing.assert_allclose(np.asarray(frozen.ru_corr_ns), expected_corr)


def test_hydrol_correct_negative_runoff_moves_deficit_to_drainage():
    result = hydrol_correct_negative_runoff(
        ru_ns=np.asarray([-0.2, 0.3], dtype=np.float64),
        dr_ns=np.asarray([1.0, 2.0], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.ru_ns), np.asarray([0.0, 0.3]))
    np.testing.assert_allclose(np.asarray(result.dr_ns), np.asarray([0.8, 2.0]))
    np.testing.assert_allclose(np.asarray(result.ru_corr2_ns), np.asarray([0.2, 0.0]))


def test_hydrol_force_water_table_saturation_updates_mc_and_dr_force():
    mc = np.asarray([[0.20, 0.30, 0.35, 0.38]], dtype=np.float64)
    dr_ns = np.asarray([2.0], dtype=np.float64)
    zz_mm = np.asarray([1.0, 4.0, 10.0, 20.0], dtype=np.float64)
    dz_mm = np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64)
    mcs = np.asarray([0.40], dtype=np.float64)

    result = hydrol_force_water_table_saturation(
        mc=mc,
        dr_ns=dr_ns,
        zwt_force=np.asarray([0.010], dtype=np.float64),
        zz_mm=zz_mm,
        dz_mm=dz_mm,
        zmaxh_m=2.0,
        mcs=mcs,
    )

    expected_mc = mc.copy()
    expected_mc[:, 2:] = mcs[:, None]
    expected_dmc = expected_mc - mc
    expected_dr_force = hydrol_total_moisture_content(expected_dmc, dz_mm)
    np.testing.assert_allclose(np.asarray(result.mc), expected_mc)
    np.testing.assert_allclose(np.asarray(result.dmc), expected_dmc)
    np.testing.assert_allclose(np.asarray(result.dr_force_ns), np.asarray(expected_dr_force))
    np.testing.assert_allclose(np.asarray(result.dr_ns), dr_ns - np.asarray(expected_dr_force))


def test_hydrol_force_water_table_saturation_inactive_uses_first_point_source_gate():
    mc = np.asarray([[0.20, 0.30, 0.35], [0.25, 0.32, 0.36]], dtype=np.float64)
    result = hydrol_force_water_table_saturation(
        mc=mc,
        dr_ns=np.asarray([1.0, 2.0], dtype=np.float64),
        zwt_force=np.asarray([3.0, 0.0], dtype=np.float64),
        zz_mm=np.asarray([1.0, 4.0, 10.0], dtype=np.float64),
        dz_mm=np.asarray([0.0, 3.0, 6.0], dtype=np.float64),
        zmaxh_m=2.0,
        mcs=np.asarray([0.40, 0.41], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.mc), mc)
    np.testing.assert_allclose(np.asarray(result.dmc), np.zeros_like(mc))
    np.testing.assert_allclose(np.asarray(result.dr_force_ns), np.zeros(2))
    np.testing.assert_allclose(np.asarray(result.dr_ns), np.asarray([1.0, 2.0]))


def test_hydrol_force_water_table_saturation_uses_peat_saturation_threshold():
    mc = np.asarray([[0.20, 0.30, 0.35]], dtype=np.float64)
    result = hydrol_force_water_table_saturation(
        mc=mc,
        dr_ns=np.asarray([2.0], dtype=np.float64),
        zwt_force=np.asarray([0.004], dtype=np.float64),
        zz_mm=np.asarray([1.0, 4.0, 10.0], dtype=np.float64),
        dz_mm=np.asarray([0.0, 3.0, 6.0], dtype=np.float64),
        zmaxh_m=2.0,
        mcs=np.asarray([0.40], dtype=np.float64),
        peat_hydro=True,
        soil_tile_index=4,
    )

    np.testing.assert_allclose(np.asarray(result.mc)[0, 1:], np.asarray([PEAT_MCS, PEAT_MCS]))
    assert np.asarray(result.dr_force_ns)[0] > 0.0


def test_hydrol_effective_water_table_depth_walks_up_from_contiguous_saturated_bottom():
    mc = np.asarray(
        [
            [0.30, 0.39, 0.40, 0.40],
            [0.30, 0.40, 0.39, 0.40],
        ],
        dtype=np.float64,
    )
    result = hydrol_effective_water_table_depth(
        mc=mc,
        zz_mm=np.asarray([1.0, 4.0, 10.0, 20.0], dtype=np.float64),
        mcs=np.asarray([0.40, 0.40], dtype=np.float64),
        undef_sechiba=-9999.0,
    )

    np.testing.assert_allclose(np.asarray(result.wtd_ns), np.asarray([0.010, 0.020]))


def test_hydrol_effective_water_table_depth_uses_peat_saturation():
    mc = np.asarray([[0.30, 0.40, PEAT_MCS, PEAT_MCS]], dtype=np.float64)
    result = hydrol_effective_water_table_depth(
        mc=mc,
        zz_mm=np.asarray([1.0, 4.0, 10.0, 20.0], dtype=np.float64),
        mcs=np.asarray([0.40], dtype=np.float64),
        peat_hydro=True,
        soil_tile_index=4,
        undef_sechiba=-9999.0,
    )

    np.testing.assert_allclose(np.asarray(result.wtd_ns), np.asarray([0.010]))


def _routing_inputs():
    return {
        "ru_ns": np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64),
        "runoff2peat": np.zeros((1, 6), dtype=np.float64),
        "soiltile": np.asarray([[0.2, 0.3, 0.1, 0.25, 0.05, 0.10]], dtype=np.float64),
        "water2infilt": np.zeros((1, 6), dtype=np.float64),
        "run2peat": np.asarray([0.0], dtype=np.float64),
        "run2man": np.asarray([0.0], dtype=np.float64),
        "wt_ab": np.asarray([0.0], dtype=np.float64),
        "wt_ab_tide": np.asarray([0.5], dtype=np.float64),
    }


def test_hydrol_after_under_mcr_routing_tides_splits_mineral_runoff_to_tiles_4_and_6():
    inputs = _routing_inputs()
    result = hydrol_after_under_mcr_runoff_peat_tide_routing(
        soil_tile_index=1,
        **inputs,
        peat_hydro=True,
        ok_ru2peat=True,
        ok_wt_ab=True,
        tides=True,
        wtp_tide=0.0,
        dwtp_tide=0.0,
    )

    mineral_sum = 1.0 * 0.2 + 2.0 * 0.3 + 3.0 * 0.1
    np.testing.assert_allclose(np.asarray(result.run2man), np.asarray([0.10 / (0.25 + 0.10) * mineral_sum]))
    np.testing.assert_allclose(np.asarray(result.run2peat), np.asarray([0.25 / (0.25 + 0.10) * mineral_sum]))
    np.testing.assert_allclose(np.asarray(result.runoff2peat[0, :3]), np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(result.ru_ns[0, :3]), np.zeros(3))


def test_hydrol_after_under_mcr_routing_non_tides_routes_to_tile4_only_on_jst4():
    inputs = _routing_inputs()
    result = hydrol_after_under_mcr_runoff_peat_tide_routing(
        soil_tile_index=4,
        **inputs,
        peat_hydro=True,
        ok_ru2peat=True,
        ok_wt_ab=False,
        tides=False,
        wtp_tide=0.0,
        dwtp_tide=0.0,
    )

    mineral_sum = 1.0 * 0.2 + 2.0 * 0.3 + 3.0 * 0.1
    np.testing.assert_allclose(np.asarray(result.run2peat), np.asarray([mineral_sum]))
    np.testing.assert_allclose(np.asarray(result.run2man), np.zeros(1))
    np.testing.assert_allclose(np.asarray(result.ru_ns[0, :3]), np.zeros(3))


def test_hydrol_after_under_mcr_routing_updates_wt_ab_and_tide_reservoir():
    inputs = _routing_inputs()
    result = hydrol_after_under_mcr_runoff_peat_tide_routing(
        soil_tile_index=4,
        **inputs,
        peat_hydro=True,
        ok_ru2peat=False,
        ok_wt_ab=True,
        tides=True,
        wtp_tide=0.7,
        dwtp_tide=0.2,
        max_wt_ab=3.0,
    )

    np.testing.assert_allclose(np.asarray(result.wt_ab), np.asarray([3.0]))
    np.testing.assert_allclose(np.asarray(result.wt_ab_tide), np.asarray([1.7]))
    np.testing.assert_allclose(np.asarray(result.ru_ns[0, 3]), 0.0)
    np.testing.assert_allclose(np.asarray(result.water2infilt[0, 3]), 1.7)

    retreat = hydrol_after_under_mcr_runoff_peat_tide_routing(
        soil_tile_index=4,
        **inputs,
        peat_hydro=False,
        ok_ru2peat=False,
        ok_wt_ab=False,
        tides=True,
        wtp_tide=0.4,
        dwtp_tide=-0.1,
    )
    np.testing.assert_allclose(np.asarray(retreat.ru_ns[0, 3]), 4.1)
    np.testing.assert_allclose(np.asarray(retreat.wt_ab_tide), np.asarray([0.4]))
    np.testing.assert_allclose(np.asarray(retreat.water2infilt[0, 3]), 0.4)


def test_hydrol_runoff_peat_post_loop_reinjection_updates_water2infilt_and_tmc():
    ru_ns = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)
    water2infilt = np.zeros((1, 6), dtype=np.float64)
    tmc = np.asarray([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]], dtype=np.float64)
    soiltile = np.asarray([[0.2, 0.3, 0.1, 0.25, 0.05, 0.10]], dtype=np.float64)

    result = hydrol_runoff_peat_post_loop_reinjection(
        ru_ns=ru_ns,
        water2infilt=water2infilt,
        tmc=tmc,
        soiltile=soiltile,
        run2peat=np.asarray([1.4], dtype=np.float64),
        run2man=np.asarray([0.7], dtype=np.float64),
        wt_ab=np.asarray([0.2], dtype=np.float64),
        peat_hydro=True,
        agri_peat=True,
        ok_ru2peat=True,
        tides=True,
    )

    expected_run2peat = 1.4 + 5.0 * 0.05
    expected_water2infilt = water2infilt.copy()
    expected_water2infilt[:, 3] += expected_run2peat / 0.25 + 0.2
    expected_water2infilt[:, 5] += 0.7 / 0.10 + 0.2
    np.testing.assert_allclose(np.asarray(result.run2peat), np.asarray([expected_run2peat]))
    np.testing.assert_allclose(np.asarray(result.ru_ns[0, 4]), 0.0)
    np.testing.assert_allclose(np.asarray(result.water2infilt), expected_water2infilt)
    np.testing.assert_allclose(np.asarray(result.tmc_soil), tmc)
    np.testing.assert_allclose(np.asarray(result.tmc), tmc + expected_water2infilt)


def _manual_under_mcr(mc, threshold, dz_mm, zmaxh_m, mask):
    out = mc.copy()
    nslm = out.shape[1]
    for jsl in range(0, nslm - 2):
        excess = np.maximum(threshold - out[:, jsl], 0.0)
        out[:, jsl] += excess
        out[:, jsl + 1] -= excess * (dz_mm[jsl] + dz_mm[jsl + 1]) / (dz_mm[jsl + 1] + dz_mm[jsl + 2])
    jsl = nslm - 2
    excess = np.maximum(threshold - out[:, jsl], 0.0)
    out[:, jsl] += excess
    out[:, jsl + 1] -= excess * (dz_mm[jsl] + dz_mm[jsl + 1]) / dz_mm[jsl + 1]
    jsl = nslm - 1
    excess = np.maximum(threshold - out[:, jsl], 0.0)
    out[:, jsl] += excess
    out[:, jsl - 1] -= excess * dz_mm[jsl] / (dz_mm[jsl - 1] + dz_mm[jsl])
    for jsl in range(nslm - 2, 0, -1):
        excess = np.maximum(threshold - out[:, jsl], 0.0)
        out[:, jsl] += excess
        out[:, jsl - 1] -= excess * (dz_mm[jsl] + dz_mm[jsl + 1]) / (dz_mm[jsl - 1] + dz_mm[jsl])
    excess_top = mask * np.maximum(threshold - out[:, 0], 0.0)
    out[:, 0] += excess_top
    out -= excess_top[:, None] * dz_mm[1] / (2.0 * zmaxh_m * 1000.0)
    out *= mask[:, None]
    return out, excess_top


def test_hydrol_soil_smooth_under_mcr_matches_source_redistribution_and_check():
    mc = np.asarray([[0.02, 0.04, 0.03, 0.20]], dtype=np.float64)
    mcr = np.asarray([0.05], dtype=np.float64)
    dz_mm = np.asarray([0.0, 2.0, 4.0, 8.0], dtype=np.float64)
    mask = np.asarray([1.0], dtype=np.float64)

    result = hydrol_soil_smooth_under_mcr(
        mc=mc,
        mcr=mcr,
        dz_mm=dz_mm,
        zmaxh_m=2.0,
        mask_soiltile=mask,
        check_cwrr2=True,
    )

    expected_mc, expected_excess = _manual_under_mcr(mc.copy(), mcr, dz_mm, 2.0, mask)
    np.testing.assert_allclose(np.asarray(result.mc), expected_mc)
    np.testing.assert_allclose(np.asarray(result.excess_top), expected_excess)
    np.testing.assert_array_equal(np.asarray(result.is_under_mcr), expected_excess > 1.0e-8)
    expected_check = hydrol_total_moisture_content(expected_mc, dz_mm) - hydrol_total_moisture_content(mc, dz_mm)
    np.testing.assert_allclose(np.asarray(result.check_under_ns), np.asarray(expected_check), atol=1.0e-15)


def test_hydrol_soil_smooth_under_mcr_masks_inactive_tile():
    mc = np.asarray([[0.02, 0.04, 0.03]], dtype=np.float64)
    result = hydrol_soil_smooth_under_mcr(
        mc=mc,
        mcr=np.asarray([0.05], dtype=np.float64),
        dz_mm=np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
        zmaxh_m=2.0,
        mask_soiltile=np.asarray([0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.mc), np.zeros_like(mc))
    np.testing.assert_array_equal(np.asarray(result.is_under_mcr), np.asarray([False]))
