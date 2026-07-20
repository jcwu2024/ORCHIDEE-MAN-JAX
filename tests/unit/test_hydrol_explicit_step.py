from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (
    bottom_drainage,
    drainage_correction,
    hydrol_liquid_redistribution_check,
    hydrol_mcl_to_mc_after_solve,
    hydrol_soil_explicit_solve_step,
    hydrol_soil_rhs_main,
    hydrol_soil_setup_coefficients,
    hydrol_soil_tridiag_solve,
    hydrol_total_moisture_content,
)


def _sample_setup():
    return hydrol_soil_setup_coefficients(
        a=np.asarray([[0.10, 0.09, 0.08, 0.07]], dtype=np.float64),
        d=np.asarray([[4.0, 4.5, 5.0, 5.5]], dtype=np.float64),
        dz_mm=np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64),
        dt_days=1.0 / 48.0,
        free_drain_coef=np.asarray([1.0], dtype=np.float64),
    )


def test_hydrol_explicit_solve_step_equals_separate_kernel_calls():
    setup = _sample_setup()
    mcl_before = np.asarray([[0.30, 0.31, 0.32, 0.33]], dtype=np.float64)
    mc_before = np.asarray([[0.32, 0.34, 0.36, 0.38]], dtype=np.float64)
    profil_froz = np.asarray([[0.0, 0.1, 0.2, 0.3]], dtype=np.float64)
    b = np.asarray([[0.1, 0.12, 0.14, 0.16]], dtype=np.float64)
    dz_mm = np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64)
    dt_days = 1.0 / 48.0
    free_drain_coef = np.asarray([1.0], dtype=np.float64)
    flux_top = np.asarray([0.02], dtype=np.float64)
    rootsink = np.asarray([[0.0, 0.01, 0.01, 0.0]], dtype=np.float64)
    resolv = np.asarray([True])
    mask_soiltile = np.asarray([1.0], dtype=np.float64)
    k_bottom = np.asarray([5.0], dtype=np.float64)
    mcr = np.asarray([0.057], dtype=np.float64)

    result = hydrol_soil_explicit_solve_step(
        setup=setup,
        mcl_before=mcl_before,
        mc_before=mc_before,
        profil_froz=profil_froz,
        mcr=mcr,
        b=b,
        dz_mm=dz_mm,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        flux_top=flux_top,
        rootsink=rootsink,
        resolv=resolv,
        mask_soiltile=mask_soiltile,
        k_bottom=k_bottom,
    )

    rhs = hydrol_soil_rhs_main(
        setup=setup,
        mcl=mcl_before,
        b=b,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        flux_top=flux_top,
        rootsink=rootsink,
    )
    solved = hydrol_soil_tridiag_solve(
        e=rhs.e,
        f=rhs.f,
        g1=rhs.g1,
        rhs=rhs.rhs,
        resolv=resolv,
        initial_mcl=mcl_before,
    )
    tmci = hydrol_total_moisture_content(mcl_before, dz_mm)
    tmcf = hydrol_total_moisture_content(solved.mcl, dz_mm)
    dr_before = bottom_drainage(k_bottom, free_drain_coef, dt_days, mask_soiltile=mask_soiltile, resolv=resolv)
    check = hydrol_liquid_redistribution_check(tmci, tmcf, flux_top, dr_before, rootsink)
    corrected = drainage_correction(dr_before, check)
    mc_after = hydrol_mcl_to_mc_after_solve(solved.mcl, mc_before, profil_froz, mcr)

    assert np.allclose(np.asarray(result.rhs), np.asarray(rhs.rhs))
    assert np.allclose(np.asarray(result.mcl_after_tridiag), np.asarray(solved.mcl))
    assert np.allclose(np.asarray(result.tmci), np.asarray(tmci))
    assert np.allclose(np.asarray(result.tmcf), np.asarray(tmcf))
    assert np.allclose(np.asarray(result.check_tr_ns), np.asarray(check))
    assert np.allclose(np.asarray(result.dr_before_corr), np.asarray(dr_before))
    assert np.allclose(np.asarray(result.dr_corrnum), np.asarray(corrected.dr_corrnum))
    assert np.allclose(np.asarray(result.dr_after_corr), np.asarray(corrected.dr_after))
    assert np.allclose(np.asarray(result.mc_after_mcl_update), np.asarray(mc_after))
    assert result.over_mcs is None
    assert result.over_mcs_routing is None
    assert result.negative_runoff is None
    assert result.forced_water_table is None
    assert result.water_table_depth is None
    assert result.under_mcr is None
    assert np.allclose(np.asarray(result.mc_final), np.asarray(mc_after))
    assert result.ru_ns_final is None
    assert np.allclose(np.asarray(result.dr_ns_final), np.asarray(corrected.dr_after))


def test_hydrol_explicit_solve_step_accepts_trace_supplied_check_and_optional_clip():
    setup = _sample_setup()
    result = hydrol_soil_explicit_solve_step(
        setup=setup,
        mcl_before=np.asarray([[0.30, 0.31, 0.32, 0.33]], dtype=np.float64),
        mc_before=np.asarray([[0.45, 0.46, 0.47, 0.48]], dtype=np.float64),
        profil_froz=np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        mcr=np.asarray([0.057], dtype=np.float64),
        b=np.asarray([[0.1, 0.12, 0.14, 0.16]], dtype=np.float64),
        dz_mm=np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64),
        dt_days=1.0 / 48.0,
        free_drain_coef=np.asarray([1.0], dtype=np.float64),
        flux_top=np.asarray([0.02], dtype=np.float64),
        rootsink=np.asarray([[0.0, 0.01, 0.01, 0.0]], dtype=np.float64),
        resolv=np.asarray([True]),
        mask_soiltile=np.asarray([1.0], dtype=np.float64),
        k_bottom=np.asarray([5.0], dtype=np.float64),
        check_tr_ns=np.asarray([0.01], dtype=np.float64),
        mcs=np.asarray([0.34], dtype=np.float64),
        clip_over_mcs=True,
    )

    assert np.allclose(np.asarray(result.check_tr_ns), np.asarray([0.01]))
    assert np.allclose(np.asarray(result.dr_corrnum), np.asarray([-0.01]))
    assert result.over_mcs is not None
    assert np.all(np.asarray(result.over_mcs.mc) <= 0.34)
    assert np.asarray(result.over_mcs.rudr_corr).shape == (1,)
    assert np.allclose(np.asarray(result.mc_final), np.asarray(result.over_mcs.mc))


def test_hydrol_explicit_solve_step_can_run_post_solve_routing_chain():
    setup = _sample_setup()
    result = hydrol_soil_explicit_solve_step(
        setup=setup,
        mcl_before=np.asarray([[0.35, 0.36, 0.37, 0.38]], dtype=np.float64),
        mc_before=np.asarray([[0.50, 0.52, 0.53, 0.54]], dtype=np.float64),
        profil_froz=np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        mcr=np.asarray([0.34], dtype=np.float64),
        b=np.asarray([[0.1, 0.12, 0.14, 0.16]], dtype=np.float64),
        dz_mm=np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64),
        dt_days=1.0 / 48.0,
        free_drain_coef=np.asarray([1.0], dtype=np.float64),
        flux_top=np.asarray([0.02], dtype=np.float64),
        rootsink=np.asarray([[0.0, 0.01, 0.01, 0.0]], dtype=np.float64),
        resolv=np.asarray([True]),
        mask_soiltile=np.asarray([1.0], dtype=np.float64),
        k_bottom=np.asarray([5.0], dtype=np.float64),
        check_tr_ns=np.asarray([0.0], dtype=np.float64),
        mcs=np.asarray([0.34], dtype=np.float64),
        ru_ns=np.asarray([-0.20], dtype=np.float64),
        route_over_mcs=True,
        ok_freeze_cwrr=False,
        check_over_mcs=True,
        correct_negative_runoff=True,
        force_water_table=True,
        zwt_force=np.asarray([0.006], dtype=np.float64),
        zz_mm=np.asarray([1.0, 4.0, 10.0, 20.0], dtype=np.float64),
        diagnose_wtd=True,
        undef_sechiba=-9999.0,
        smooth_under_mcr=True,
        zmaxh_m=2.0,
        peat_hydro=True,
        soil_tile_index=4,
        check_under_mcr=True,
    )

    assert result.over_mcs_routing is not None
    assert result.negative_runoff is not None
    assert result.forced_water_table is not None
    assert result.water_table_depth is not None
    assert result.under_mcr is not None
    np.testing.assert_allclose(np.asarray(result.over_mcs_routing.ru_corr_ns), np.zeros(1))
    assert np.asarray(result.over_mcs_routing.dr_corr_ns)[0] > 0.0
    np.testing.assert_allclose(np.asarray(result.negative_runoff.ru_ns), np.zeros(1))
    np.testing.assert_allclose(np.asarray(result.ru_ns_final), np.zeros(1))
    assert np.asarray(result.forced_water_table.dr_force_ns)[0] > 0.0
    np.testing.assert_allclose(np.asarray(result.dr_ns_final), np.asarray(result.forced_water_table.dr_ns))
    np.testing.assert_allclose(np.asarray(result.water_table_depth.wtd_ns), np.asarray([0.010]))
    np.testing.assert_allclose(np.asarray(result.mc_final), np.asarray(result.under_mcr.mc))
