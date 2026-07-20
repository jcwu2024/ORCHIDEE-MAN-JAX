from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.carbon_kernels import (
    ICARBON,
    ICARBRES,
    IABOVE,
    IBELOW,
    IAGRSAPPN,
    IAGRSAPST,
    IAGRHRTPN,
    IAGRHRTST,
    IHEARTABOVE,
    ILEAF,
    IMETABOLIC,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    ISTRUCTURAL,
    IWPLCC,
    NCARB,
    NLEAFAGES,
    NLEVS,
    NLITT,
    NPARTS,
    NPOOL,
    SENESCENCE_DRY,
    allocation_step,
    constraints_step,
    crown_step,
    crown_woodmass_ind,
    littercalc_leak_core_with_controls,
    gap_mortality_step,
    harvest_agri_step,
    kill_pfts_step,
    light_competition_step,
    lcc_gross_glcchange_step,
    lcchange_deffire_step,
    lcchange_main_agripeat_step,
    lcchange_main_leak_step,
    lpj_cover_peat_step,
    maintenance_respiration,
    npp_leaf_age_sla_age_update,
    npp_closed_update,
    phenology_step,
    prescribe_step,
    stomate_littercalc_entry_prep,
    stomate_lpj_output_diagnostics,
    turnover_step,
    vmax_step,
)
from jax_orchidee.stomate.daily import accumulate_resp_maint_part, sum_resp_maint_radia
from jax_orchidee.stomate.daily import stomate_accumulate_daily
from jax_orchidee.stomate.daily_inputs import StomateEntryLocalPrepResult, stomate_gpp_daily_increment
from jax_orchidee.stomate.integration import (
    ExplicitOkPcDeepCarbonSidecarState,
    stomate_daily_alloc_kill_gap_turnover_explicit,
    stomate_daily_carbon_explicit,
    stomate_daily_carbon_gap_turnover_explicit,
    stomate_daily_carbon_kill_gap_turnover_explicit,
    stomate_daily_carbon_turnover_explicit,
    stomate_daily_carbon_with_alloc_explicit,
    stomate_daily_prescribe_alloc_kill_gap_turnover_explicit,
    stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit,
    stomate_daily_scheduled_gpp_maintenance_prescribe_constraints_alloc_kill_gap_turnover_explicit,
    stomate_daily_scheduled_gpp_prescribe_constraints_alloc_kill_gap_turnover_explicit,
    stomate_daily_lcc_gross_glcchange_explicit,
    stomate_lcchange_deffire_from_restart_state,
    stomate_lcchange_main_agripeat_from_restart_state,
    stomate_lcchange_main_leak_from_restart_state,
    stomate_lcc_gross_glcchange_from_restart_state,
    stomate_lpj_cover_peat_from_restart_state,
    stomate_ok_pc_firstcall_from_restart_gas,
    stomate_ok_pc_firstcall_sidecar_from_restart_gas,
    stomate_ok_pc_deep_carbcycle_explicit,
    stomate_ok_pc_deep_carbcycle_from_restart_state,
    stomate_lpj_outputs_from_post_npp_ok_leak_explicit,
    stomate_ok_leak_from_post_npp_explicit,
    stomate_ok_leak_explicit,
    stomate_ok_leak_with_maintenance_explicit,
)
from jax_orchidee.stomate.modelout import compute_modelout_from_fields, stomate_lpj_history_fields_from_state
from jax_orchidee.stomate.reference import StomateOkPcRestartGasState, stomate_cold_start_entry_state, stomate_cold_start_season_state
from jax_orchidee.stomate.soilcarbon_kernels import (
    IDOCL,
    deep_carbon_altcalc_step,
    deep_carbon_cryoturbation_coefficients,
    deep_carbon_gasdiff_properties_step,
    deep_carbon_nonmethane_core_step,
    deep_carbon_root_depth_step,
    deep_carbon_snow_interpol_step,
    deep_carbon_snowlevels_step,
    deep_carbon_soil_gasdiff_coefficients,
    deep_carbon_soil_gasdiff_diffuse,
    soilcarbon_leak_core_step,
    soilcarbon_leak_doc_export_aggregate,
    soilcarbon_leak_tf_doc_ground_fluxes,
)


PFT14 = 13


def _synthetic_inputs(npts=2, nvm=14):
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[:, PFT14, ILEAF, ICARBON] = 100.0
    f_alloc = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    f_alloc[:, PFT14, ILEAF] = 0.6
    f_alloc[:, PFT14, IROOT] = 0.4
    resp_maint_part = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    resp_maint_part[:, PFT14, ILEAF] = 0.7
    resp_maint_part[:, PFT14, IROOT] = 0.3
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[:, PFT14] = True
    frac_growthresp = np.zeros(nvm, dtype=np.float64)
    frac_growthresp[PFT14] = 0.25
    gpp_daily = np.full((npts, nvm), 10.0, dtype=np.float64)
    return biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily


def test_stomate_ok_pc_deep_carbcycle_explicit_matches_source_order_helpers():
    """Fortran: stomate_permafrost_soilcarbon.f90 deep_carbcycle lines 892-1077."""

    npts, nvm, ndeep, nsnow = 1, 14, 3, 2
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0
    zi_soil = np.asarray([0.25, 0.75, 1.5], dtype=np.float64)
    zf_soil = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    old_snowdz = np.asarray([[0.2, 0.2]], dtype=np.float64)
    snowdz = np.asarray([[0.1, 0.2]], dtype=np.float64)
    old_snow = deep_carbon_snowlevels_step(old_snowdz, veget_max)
    snowrho = np.asarray([[150.0, 250.0]], dtype=np.float64)
    heights_snow = np.repeat(np.sum(snowdz, axis=1)[:, None], nvm, axis=1)
    hslong = np.ones((npts, ndeep, nvm), dtype=np.float64) * 0.4
    tprof = np.ones((npts, ndeep, nvm), dtype=np.float64) * 272.0
    tprof[0, 0, PFT14] = 274.0
    deep_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.zeros_like(deep_a)
    deep_p = np.zeros_like(deep_a)
    deep_a[0, :, PFT14] = [10.0, 12.0, 14.0]
    deep_s[0, :, PFT14] = [20.0, 22.0, 24.0]
    deep_p[0, :, PFT14] = [30.0, 32.0, 34.0]
    soilc_in = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    soilc_in[0, :, PFT14] = [0.4, 0.2, 0.1]
    fbact = np.ones_like(deep_a) * 10000.0
    o2_soil = np.ones_like(deep_a) * 100.0
    ch4_soil = np.ones_like(deep_a) * 2.0
    o2_snow = np.ones((npts, nsnow, nvm), dtype=np.float64) * 50.0
    ch4_snow = np.ones_like(o2_snow) * 0.2
    alt = np.zeros((npts, nvm), dtype=np.float64)
    alt_ind = np.zeros((npts, nvm), dtype=np.int32)
    altmax = np.ones((npts, nvm), dtype=np.float64) * 0.6
    altmax_ind = np.ones((npts, nvm), dtype=np.int32)
    altmax_lastyear = altmax.copy()
    altmax_ind_lastyear = altmax_ind.copy()
    fixed_depth = altmax_lastyear.copy()
    rprof = np.ones((npts, nvm), dtype=np.float64)
    current_props = deep_carbon_gasdiff_properties_step(hslong, snowrho, mask)
    previous_gas = deep_carbon_soil_gasdiff_coefficients(
        o2_snow,
        ch4_snow,
        current_props.diffO2_snow,
        current_props.diffCH4_snow,
        current_props.totporO2_snow,
        current_props.totporCH4_snow,
        o2_soil,
        ch4_soil,
        current_props.diffO2_soil,
        current_props.diffCH4_soil,
        current_props.totporO2_soil,
        current_props.totporCH4_soil,
        old_snow.zi_snow,
        old_snow.zf_snow,
        zi_soil,
        zf_soil,
        mask,
        np.repeat(np.sum(old_snowdz, axis=1)[:, None], nvm, axis=1),
        dt_seconds=43200.0,
    )
    previous_cryo = deep_carbon_cryoturbation_coefficients(
        altmax_ind_lastyear,
        deep_a,
        deep_s,
        deep_p,
        altmax_lastyear,
        fixed_depth,
        mask,
        zi_soil,
        zf_soil,
        dt_seconds=43200.0,
        diff_k_const=0.001 / (86400.0 * 365.0),
        bio_diff_k_const=0.0001 / (86400.0 * 365.0),
    )

    gas_state = deep_carbon_soil_gasdiff_diffuse(
        o2_snow,
        ch4_snow,
        o2_soil,
        ch4_soil,
        previous_gas,
        np.asarray([101325.0], dtype=np.float64),
        np.asarray([300.0], dtype=np.float64),
        mask,
    )
    snow_interp = deep_carbon_snow_interpol_step(
        gas_state.O2_snow,
        gas_state.CH4_snow,
        old_snow.zi_snow,
        old_snow.zf_snow,
        veget_max,
        snowdz,
        mask,
    )
    altcalc = deep_carbon_altcalc_step(
        tprof,
        zi_soil,
        altmax,
        altmax_ind,
        altmax_lastyear,
        altmax_ind_lastyear,
        mask,
        firstcall=False,
        dayno=3,
    )
    root = deep_carbon_root_depth_step(altcalc.altmax_lastyear, altcalc.altmax_ind_lastyear, mask, z_root_max=2.0)
    core = deep_carbon_nonmethane_core_step(
        deep_a,
        deep_s,
        deep_p,
        soilc_in,
        fbact,
        gas_state.O2_soil,
        current_props.totporO2_soil,
        np.asarray([0.1], dtype=np.float64),
        mask,
        zi_soil,
        zf_soil,
        root.z_root,
        altcalc.altmax_lastyear,
        rprof,
        time_step_seconds=43200.0,
        airvol_soil=current_props.airvol_soil,
        ok_methane=True,
        CH4_soil=gas_state.CH4_soil,
        totporCH4_soil=current_props.totporCH4_soil,
        hslong=hslong,
        tprof=tprof,
        firstcall=True,
        ok_cryoturb=True,
        cryoturbation_coefficients=previous_cryo,
        altmax_ind=altcalc.altmax_ind_lastyear,
        fixed_cryoturbation_depth=fixed_depth,
        cryoturbation_diff_k_const=0.001 / (86400.0 * 365.0),
        cryoturbation_bio_diff_k_const=0.0001 / (86400.0 * 365.0),
    )
    next_props = deep_carbon_gasdiff_properties_step(hslong, snowrho, mask)
    next_gas = deep_carbon_soil_gasdiff_coefficients(
        snow_interp.snowO2,
        snow_interp.snowCH4,
        next_props.diffO2_snow,
        next_props.diffCH4_snow,
        next_props.totporO2_snow,
        next_props.totporCH4_snow,
        core.O2_soil,
        core.CH4_soil,
        next_props.diffO2_soil,
        next_props.diffCH4_soil,
        next_props.totporO2_soil,
        next_props.totporCH4_soil,
        snow_interp.zi_snow,
        snow_interp.zf_snow,
        zi_soil,
        zf_soil,
        mask,
        heights_snow,
        dt_seconds=43200.0,
    )

    actual = stomate_ok_pc_deep_carbcycle_explicit(
        deepC_a=deep_a,
        deepC_s=deep_s,
        deepC_p=deep_p,
        soilc_in=soilc_in,
        fbact_out=fbact,
        O2_soil=o2_soil,
        CH4_soil=ch4_soil,
        O2_snow=o2_snow,
        CH4_snow=ch4_snow,
        hslong_in=hslong,
        snowdz=snowdz,
        snowrho=snowrho,
        tprof=tprof,
        tsurf=np.asarray([300.0], dtype=np.float64),
        pb=np.asarray([101325.0], dtype=np.float64),
        clay=np.asarray([0.1], dtype=np.float64),
        veget_max=veget_max,
        veget_mask=mask,
        zi_soil=zi_soil,
        zf_soil=zf_soil,
        rprof=rprof,
        alt=alt,
        alt_ind=alt_ind,
        altmax=altmax,
        altmax_ind=altmax_ind,
        altmax_lastyear=altmax_lastyear,
        altmax_ind_lastyear=altmax_ind_lastyear,
        fixed_cryoturbation_depth=fixed_depth,
        time_step_seconds=43200.0,
        dayno=3,
        gasdiff_coefficients=previous_gas,
        cryoturbation_coefficients=previous_cryo,
        zi_snow=old_snow.zi_snow,
        zf_snow=old_snow.zf_snow,
        airvol_soil=current_props.airvol_soil,
        totporO2_soil=current_props.totporO2_soil,
        totporCH4_soil=current_props.totporCH4_soil,
        diffO2_soil=current_props.diffO2_soil,
        diffCH4_soil=current_props.diffCH4_soil,
        airvol_snow=current_props.airvol_snow,
        totporO2_snow=current_props.totporO2_snow,
        totporCH4_snow=current_props.totporCH4_snow,
        diffO2_snow=current_props.diffO2_snow,
        diffCH4_snow=current_props.diffCH4_snow,
        firstcall=True,
        ok_cryoturb=True,
        ok_methane=True,
    )

    np.testing.assert_allclose(np.asarray(actual.core.deepC_a), np.asarray(core.deepC_a))
    np.testing.assert_allclose(np.asarray(actual.core.resp_hetero_soil), np.asarray(core.resp_hetero_soil))
    np.testing.assert_allclose(np.asarray(actual.core.sfluxCH4), np.asarray(core.sfluxCH4))
    np.testing.assert_allclose(np.asarray(actual.O2_snow), np.asarray(snow_interp.snowO2))
    np.testing.assert_allclose(np.asarray(actual.CH4_snow), np.asarray(snow_interp.snowCH4))
    np.testing.assert_allclose(np.asarray(actual.gasdiff_coefficients.betaO2_soil), np.asarray(next_gas.betaO2_soil))
    np.testing.assert_allclose(
        np.asarray(actual.cryoturbation_coefficients.beta_a),
        np.asarray(core.cryoturbation_coefficients.beta_a),
    )


def test_stomate_ok_pc_restart_sidecar_handoff_feeds_next_day():
    """Fortran: stomate.f90 OK_PC calls lines 3336-3383 and 3628-3655."""

    npts, nvm, ndeep, nsnow = 1, 14, 3, 2
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.8
    zi_soil = np.asarray([0.25, 0.75, 1.5], dtype=np.float64)
    zf_soil = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    snowdz0 = np.asarray([[0.2, 0.2]], dtype=np.float64)
    snowdz1 = np.asarray([[0.1, 0.2]], dtype=np.float64)
    snowdz2 = np.asarray([[0.15, 0.1]], dtype=np.float64)
    snowrho1 = np.asarray([[150.0, 250.0]], dtype=np.float64)
    snowrho2 = np.asarray([[170.0, 220.0]], dtype=np.float64)
    hslong1 = np.ones((npts, ndeep, nvm), dtype=np.float64) * 0.35
    hslong2 = np.ones((npts, ndeep, nvm), dtype=np.float64) * 0.42
    tprof1 = np.ones((npts, ndeep, nvm), dtype=np.float64) * 272.0
    tprof2 = np.ones((npts, ndeep, nvm), dtype=np.float64) * 273.0
    tprof1[0, 0, PFT14] = 274.0
    tprof2[0, 1, PFT14] = 274.5
    o2_soil = np.ones((npts, ndeep, nvm), dtype=np.float64) * 100.0
    ch4_soil = np.ones_like(o2_soil) * 2.0
    o2_snow = np.ones((npts, nsnow, nvm), dtype=np.float64) * 50.0
    ch4_snow = np.ones_like(o2_snow) * 0.2
    old_snow = deep_carbon_snowlevels_step(snowdz0, veget_max)
    initial_props = deep_carbon_gasdiff_properties_step(hslong1, snowrho1, mask)
    previous_gas = deep_carbon_soil_gasdiff_coefficients(
        o2_snow,
        ch4_snow,
        initial_props.diffO2_snow,
        initial_props.diffCH4_snow,
        initial_props.totporO2_snow,
        initial_props.totporCH4_snow,
        o2_soil,
        ch4_soil,
        initial_props.diffO2_soil,
        initial_props.diffCH4_soil,
        initial_props.totporO2_soil,
        initial_props.totporCH4_soil,
        old_snow.zi_snow,
        old_snow.zf_snow,
        zi_soil,
        zf_soil,
        mask,
        np.repeat(np.sum(snowdz0, axis=1)[:, None], nvm, axis=1),
        dt_seconds=43200.0,
    )

    state = stomate_cold_start_entry_state(t2m=np.asarray([289.0]), nvm=nvm, nslm=2, ndeep=ndeep, nelements=1)
    deep_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.zeros_like(deep_a)
    deep_p = np.zeros_like(deep_a)
    deep_a[0, :, PFT14] = [10.0, 12.0, 14.0]
    deep_s[0, :, PFT14] = [20.0, 22.0, 24.0]
    deep_p[0, :, PFT14] = [30.0, 32.0, 34.0]
    altmax = np.ones((npts, nvm), dtype=np.float64) * 0.6
    altmax_ind = np.ones((npts, nvm), dtype=np.int32)
    fixed_depth = altmax.copy()
    state = state._replace(
        deepC_a=deep_a,
        deepC_s=deep_s,
        deepC_p=deep_p,
        soilc_total=deep_a + deep_s + deep_p,
        altmax=altmax,
        fixed_cryoturbation_depth=fixed_depth,
    )
    previous_cryo = deep_carbon_cryoturbation_coefficients(
        altmax_ind,
        deep_a,
        deep_s,
        deep_p,
        altmax,
        fixed_depth,
        mask,
        zi_soil,
        zf_soil,
        dt_seconds=43200.0,
        diff_k_const=0.001 / (86400.0 * 365.0),
        bio_diff_k_const=0.0001 / (86400.0 * 365.0),
    )
    sidecar = ExplicitOkPcDeepCarbonSidecarState(
        O2_soil=o2_soil,
        CH4_soil=ch4_soil,
        O2_snow=o2_snow,
        CH4_snow=ch4_snow,
        zi_snow=old_snow.zi_snow,
        zf_snow=old_snow.zf_snow,
        airvol_soil=initial_props.airvol_soil,
        totporO2_soil=initial_props.totporO2_soil,
        totporCH4_soil=initial_props.totporCH4_soil,
        diffO2_soil=initial_props.diffO2_soil,
        diffCH4_soil=initial_props.diffCH4_soil,
        airvol_snow=initial_props.airvol_snow,
        totporO2_snow=initial_props.totporO2_snow,
        totporCH4_snow=initial_props.totporCH4_snow,
        diffO2_snow=initial_props.diffO2_snow,
        diffCH4_snow=initial_props.diffCH4_snow,
        alt=np.zeros((npts, nvm), dtype=np.float64),
        alt_ind=np.zeros((npts, nvm), dtype=np.int32),
        altmax_ind=altmax_ind,
        altmax_lastyear=altmax.copy(),
        altmax_ind_lastyear=altmax_ind.copy(),
        z_root=np.ones((npts, nvm), dtype=np.float64) * 0.6,
        rootlev=altmax_ind.copy(),
        heights_snow=np.repeat(np.sum(snowdz0, axis=1)[:, None], nvm, axis=1),
        gasdiff_coefficients=previous_gas,
        cryoturbation_coefficients=previous_cryo,
        Tref=np.zeros((npts, nvm), dtype=np.float64),
    )
    soilc_in = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    soilc_in[0, :, PFT14] = [0.4, 0.2, 0.1]
    fbact = np.ones((npts, ndeep, nvm), dtype=np.float64) * 10000.0
    common = dict(
        soilc_in=soilc_in,
        fbact_out=fbact,
        tsurf=np.asarray([300.0], dtype=np.float64),
        pb=np.asarray([101325.0], dtype=np.float64),
        clay=np.asarray([0.1], dtype=np.float64),
        veget_max=veget_max,
        veget_mask=mask,
        zi_soil=zi_soil,
        zf_soil=zf_soil,
        rprof=np.ones((npts, nvm), dtype=np.float64),
        time_step_seconds=43200.0,
        ok_cryoturb=True,
        ok_methane=True,
    )

    day1 = stomate_ok_pc_deep_carbcycle_from_restart_state(
        state=state,
        sidecar=sidecar,
        hslong_in=hslong1,
        snowdz=snowdz1,
        snowrho=snowrho1,
        tprof=tprof1,
        dayno=3,
        firstcall=True,
        **common,
    )
    manual_day2 = stomate_ok_pc_deep_carbcycle_explicit(
        deepC_a=day1.state_after.deepC_a,
        deepC_s=day1.state_after.deepC_s,
        deepC_p=day1.state_after.deepC_p,
        O2_soil=day1.sidecar_after.O2_soil,
        CH4_soil=day1.sidecar_after.CH4_soil,
        O2_snow=day1.sidecar_after.O2_snow,
        CH4_snow=day1.sidecar_after.CH4_snow,
        hslong_in=hslong2,
        snowdz=snowdz2,
        snowrho=snowrho2,
        tprof=tprof2,
        alt=day1.sidecar_after.alt,
        alt_ind=day1.sidecar_after.alt_ind,
        altmax=day1.state_after.altmax,
        altmax_ind=day1.sidecar_after.altmax_ind,
        altmax_lastyear=day1.sidecar_after.altmax_lastyear,
        altmax_ind_lastyear=day1.sidecar_after.altmax_ind_lastyear,
        fixed_cryoturbation_depth=day1.state_after.fixed_cryoturbation_depth,
        gasdiff_coefficients=day1.sidecar_after.gasdiff_coefficients,
        cryoturbation_coefficients=day1.sidecar_after.cryoturbation_coefficients,
        zi_snow=day1.sidecar_after.zi_snow,
        zf_snow=day1.sidecar_after.zf_snow,
        airvol_soil=day1.sidecar_after.airvol_soil,
        totporO2_soil=day1.sidecar_after.totporO2_soil,
        totporCH4_soil=day1.sidecar_after.totporCH4_soil,
        diffO2_soil=day1.sidecar_after.diffO2_soil,
        diffCH4_soil=day1.sidecar_after.diffCH4_soil,
        airvol_snow=day1.sidecar_after.airvol_snow,
        totporO2_snow=day1.sidecar_after.totporO2_snow,
        totporCH4_snow=day1.sidecar_after.totporCH4_snow,
        diffO2_snow=day1.sidecar_after.diffO2_snow,
        diffCH4_snow=day1.sidecar_after.diffCH4_snow,
        Tref=day1.sidecar_after.Tref,
        dayno=4,
        firstcall=False,
        **common,
    )
    day2 = stomate_ok_pc_deep_carbcycle_from_restart_state(
        state=day1.state_after,
        sidecar=day1.sidecar_after,
        hslong_in=hslong2,
        snowdz=snowdz2,
        snowrho=snowrho2,
        tprof=tprof2,
        dayno=4,
        firstcall=False,
        **common,
    )

    np.testing.assert_allclose(np.asarray(day2.state_after.deepC_a), np.asarray(manual_day2.core.deepC_a))
    np.testing.assert_allclose(np.asarray(day2.state_after.deepC_s), np.asarray(manual_day2.core.deepC_s))
    np.testing.assert_allclose(np.asarray(day2.state_after.deepC_p), np.asarray(manual_day2.core.deepC_p))
    np.testing.assert_allclose(np.asarray(day2.state_after.carbon), np.asarray(manual_day2.core.carbon))
    np.testing.assert_allclose(np.asarray(day2.state_after.soilc_total), np.asarray(manual_day2.core.deepC_a + manual_day2.core.deepC_s + manual_day2.core.deepC_p))
    np.testing.assert_allclose(np.asarray(day2.heat_Zimov), np.asarray(manual_day2.core.heat_Zimov))
    np.testing.assert_allclose(np.asarray(day2.sidecar_after.O2_soil), np.asarray(manual_day2.O2_soil))
    np.testing.assert_allclose(np.asarray(day2.sidecar_after.CH4_soil), np.asarray(manual_day2.CH4_soil))
    np.testing.assert_allclose(np.asarray(day2.sidecar_after.gasdiff_coefficients.betaO2_soil), np.asarray(manual_day2.gasdiff_coefficients.betaO2_soil))
    np.testing.assert_allclose(np.asarray(day2.sidecar_after.cryoturbation_coefficients.beta_a), np.asarray(manual_day2.cryoturbation_coefficients.beta_a))
    np.testing.assert_allclose(np.asarray(day2.sidecar_after.Tref), np.asarray(manual_day2.core.Tref))
    np.testing.assert_allclose(
        np.asarray(day2.sfluxCO2_deep),
        np.asarray(np.sum(veget_max * manual_day2.core.resp_hetero_soil, axis=1) / 86400.0),
    )
    np.testing.assert_allclose(
        np.asarray(day2.sfluxCH4_deep),
        np.asarray(np.sum(veget_max * manual_day2.core.sfluxCH4, axis=1) / 86400.0),
    )


def test_stomate_ok_pc_firstcall_sidecar_from_restart_gas_matches_source_order_helpers():
    """Fortran: stomate_permafrost_soilcarbon.f90 deep_carbcycle lines 703-797."""

    npts, nvm, ndeep, nsnow = 1, 14, 3, 2
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0
    zi_soil = np.asarray([0.25, 0.75, 1.5], dtype=np.float64)
    zf_soil = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    snowdz = np.asarray([[0.2, 0.1]], dtype=np.float64)
    snowrho = np.asarray([[150.0, 250.0]], dtype=np.float64)
    hslong = np.ones((npts, ndeep, nvm), dtype=np.float64) * 0.4
    tprof = np.ones((npts, ndeep, nvm), dtype=np.float64) * 272.0
    o2_soil = np.ones((npts, ndeep, nvm), dtype=np.float64) * 100.0
    ch4_soil = np.ones_like(o2_soil) * 2.0
    o2_snow = np.ones((npts, nsnow, nvm), dtype=np.float64) * 50.0
    ch4_snow = np.ones_like(o2_snow) * 0.2
    gas_state = StomateOkPcRestartGasState(
        O2_soil=o2_soil,
        CH4_soil=ch4_soil,
        O2_snow=o2_snow,
        CH4_snow=ch4_snow,
    )
    altmax = np.ones((npts, nvm), dtype=np.float64) * 0.6
    snow_geometry = deep_carbon_snowlevels_step(snowdz, veget_max)
    heights_snow = np.repeat(np.sum(snowdz, axis=1)[:, None], nvm, axis=1)
    props = deep_carbon_gasdiff_properties_step(hslong, snowrho, mask)
    gasdiff = deep_carbon_soil_gasdiff_coefficients(
        o2_snow,
        ch4_snow,
        props.diffO2_snow,
        props.diffCH4_snow,
        props.totporO2_snow,
        props.totporCH4_snow,
        o2_soil,
        ch4_soil,
        props.diffO2_soil,
        props.diffCH4_soil,
        props.totporO2_soil,
        props.totporCH4_soil,
        snow_geometry.zi_snow,
        snow_geometry.zf_snow,
        zi_soil,
        zf_soil,
        mask,
        heights_snow,
        dt_seconds=43200.0,
    )
    altcalc = deep_carbon_altcalc_step(
        tprof,
        zi_soil,
        altmax,
        np.zeros_like(altmax, dtype=np.int32),
        np.zeros_like(altmax),
        np.zeros_like(altmax, dtype=np.int32),
        mask,
        firstcall=True,
        dayno=3,
    )
    root = deep_carbon_root_depth_step(altcalc.altmax_lastyear, altcalc.altmax_ind_lastyear, mask, z_root_max=2.0)

    actual = stomate_ok_pc_firstcall_sidecar_from_restart_gas(
        gas_state=gas_state,
        hslong_in=hslong,
        snowdz=snowdz,
        snowrho=snowrho,
        tprof=tprof,
        tsurf=np.asarray([300.0], dtype=np.float64),
        pb=np.asarray([101325.0], dtype=np.float64),
        veget_max=veget_max,
        veget_mask=mask,
        zi_soil=zi_soil,
        zf_soil=zf_soil,
        altmax=altmax,
        time_step_seconds=43200.0,
        dayno=3,
    )

    np.testing.assert_allclose(np.asarray(actual.zi_snow), np.asarray(snow_geometry.zi_snow))
    np.testing.assert_allclose(np.asarray(actual.zf_snow), np.asarray(snow_geometry.zf_snow))
    np.testing.assert_allclose(np.asarray(actual.airvol_soil), np.asarray(props.airvol_soil))
    np.testing.assert_allclose(np.asarray(actual.gasdiff_coefficients.betaO2_soil), np.asarray(gasdiff.betaO2_soil))
    np.testing.assert_allclose(np.asarray(actual.altmax_lastyear), np.asarray(altcalc.altmax_lastyear))
    np.testing.assert_allclose(np.asarray(actual.altmax_ind_lastyear), np.asarray(altcalc.altmax_ind_lastyear))
    np.testing.assert_allclose(np.asarray(actual.z_root), np.asarray(root.z_root))
    np.testing.assert_allclose(np.asarray(actual.rootlev), np.asarray(root.rootlev))
    assert actual.cryoturbation_coefficients is None


def test_stomate_ok_pc_firstcall_from_restart_gas_matches_manual_sidecar_then_day_step():
    """Fortran: readstart gas lines 1177-1199 plus deep_carbcycle lines 703-1077."""

    npts, nvm, ndeep, nsnow = 1, 14, 3, 2
    mask = np.zeros((npts, nvm), dtype=bool)
    mask[0, PFT14] = True
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0
    zi_soil = np.asarray([0.25, 0.75, 1.5], dtype=np.float64)
    zf_soil = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    snowdz = np.asarray([[0.2, 0.1]], dtype=np.float64)
    snowrho = np.asarray([[150.0, 250.0]], dtype=np.float64)
    hslong = np.ones((npts, ndeep, nvm), dtype=np.float64) * 0.4
    tprof = np.ones((npts, ndeep, nvm), dtype=np.float64) * 272.0
    tprof[0, 0, PFT14] = 274.0
    gas_state = StomateOkPcRestartGasState(
        O2_soil=np.ones((npts, ndeep, nvm), dtype=np.float64) * 100.0,
        CH4_soil=np.ones((npts, ndeep, nvm), dtype=np.float64) * 2.0,
        O2_snow=np.ones((npts, nsnow, nvm), dtype=np.float64) * 50.0,
        CH4_snow=np.ones((npts, nsnow, nvm), dtype=np.float64) * 0.2,
    )
    state = stomate_cold_start_entry_state(t2m=np.asarray([289.0]), nvm=nvm, nslm=2, ndeep=ndeep, nelements=1)
    deep_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deep_s = np.zeros_like(deep_a)
    deep_p = np.zeros_like(deep_a)
    deep_a[0, :, PFT14] = [10.0, 12.0, 14.0]
    deep_s[0, :, PFT14] = [20.0, 22.0, 24.0]
    deep_p[0, :, PFT14] = [30.0, 32.0, 34.0]
    state = state._replace(
        deepC_a=deep_a,
        deepC_s=deep_s,
        deepC_p=deep_p,
        soilc_total=deep_a + deep_s + deep_p,
        altmax=np.ones((npts, nvm), dtype=np.float64) * 0.6,
        fixed_cryoturbation_depth=np.ones((npts, nvm), dtype=np.float64) * 0.6,
    )
    soilc_in = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    soilc_in[0, :, PFT14] = [0.4, 0.2, 0.1]
    common = dict(
        soilc_in=soilc_in,
        fbact_out=np.ones((npts, ndeep, nvm), dtype=np.float64) * 10000.0,
        hslong_in=hslong,
        snowdz=snowdz,
        snowrho=snowrho,
        tprof=tprof,
        tsurf=np.asarray([300.0], dtype=np.float64),
        pb=np.asarray([101325.0], dtype=np.float64),
        clay=np.asarray([0.1], dtype=np.float64),
        veget_max=veget_max,
        veget_mask=mask,
        zi_soil=zi_soil,
        zf_soil=zf_soil,
        rprof=np.ones((npts, nvm), dtype=np.float64),
        time_step_seconds=43200.0,
        dayno=3,
        ok_cryoturb=True,
        ok_methane=True,
    )

    sidecar = stomate_ok_pc_firstcall_sidecar_from_restart_gas(
        gas_state=gas_state,
        hslong_in=hslong,
        snowdz=snowdz,
        snowrho=snowrho,
        tprof=tprof,
        tsurf=np.asarray([300.0], dtype=np.float64),
        pb=np.asarray([101325.0], dtype=np.float64),
        veget_max=veget_max,
        veget_mask=mask,
        zi_soil=zi_soil,
        zf_soil=zf_soil,
        altmax=state.altmax,
        time_step_seconds=43200.0,
        dayno=3,
    )
    expected = stomate_ok_pc_deep_carbcycle_from_restart_state(
        state=state,
        sidecar=sidecar,
        firstcall=True,
        **common,
    )
    actual = stomate_ok_pc_firstcall_from_restart_gas(
        state=state,
        gas_state=gas_state,
        **common,
    )

    np.testing.assert_allclose(np.asarray(actual.state_after.deepC_a), np.asarray(expected.state_after.deepC_a))
    np.testing.assert_allclose(np.asarray(actual.state_after.carbon), np.asarray(expected.state_after.carbon))
    np.testing.assert_allclose(np.asarray(actual.sidecar_after.O2_soil), np.asarray(expected.sidecar_after.O2_soil))
    np.testing.assert_allclose(
        np.asarray(actual.sidecar_after.gasdiff_coefficients.betaO2_soil),
        np.asarray(expected.sidecar_after.gasdiff_coefficients.betaO2_soil),
    )
    np.testing.assert_allclose(np.asarray(actual.heat_Zimov), np.asarray(expected.heat_Zimov))
    np.testing.assert_allclose(np.asarray(actual.sfluxCO2_deep), np.asarray(expected.sfluxCO2_deep))


def test_lcc_gross_glcchange_restart_adapter_uses_explicit_legacy_carbon_and_deepc_pools():
    """Adapter must not synthesize LCC carbon/deepC from nearby restart totals."""

    npts, nvm, ndeep, nelements, nlevels = 1, 5, 2, 1, 2
    start_index = [0, 1, 2, 3, 4]
    nagec_pft = [1, 1, 1, 1, 1]
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[2] = True
    is_grassland_manag[3] = True
    pft_to_mtc = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    agec_group = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.asarray([[0.0, 0.5, 0.4, 0.3, 0.2]], dtype=np.float64)
    shifts = np.zeros((npts, 12), dtype=np.float64)
    glcc_primary = shifts.copy()
    glcc_net = shifts.copy()
    glcc_primary[0, 2] = 0.2
    glcc_net[0, 2] = 0.1

    state = stomate_cold_start_entry_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        nvm=nvm,
        nslm=2,
        ndeep=ndeep,
        nelements=nelements,
    )
    season = stomate_cold_start_season_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        dt_days=30.0,
        nvm=nvm,
        nslm=2,
    )

    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    for offset, part in enumerate(
        [ISAPABOVE, ISAPBELOW, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN],
        start=1,
    ):
        biomass[0, 1, part, ICARBON] = offset * 3.0
    carbon = np.arange(npts * NCARB * nvm, dtype=np.float64).reshape(npts, NCARB, nvm) + 10.0
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    carbon_32l[:, :, :, 0] = carbon + 1000.0
    deepC_a = np.arange(npts * ndeep * nvm, dtype=np.float64).reshape(npts, ndeep, nvm) + 20.0
    deepC_s = deepC_a + 200.0
    deepC_p = deepC_a + 400.0
    two_d = np.ones((npts, nvm), dtype=np.float64)
    leaf = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    flux10 = np.zeros((npts, 10, 2), dtype=np.float64)
    flux100 = np.zeros((npts, 100, 2), dtype=np.float64)

    state = state._replace(
        biomass=biomass,
        bm_to_litter=np.ones_like(biomass) * 0.2,
        carbon=carbon,
        carbon_32l=carbon_32l,
        litter=np.ones((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64) * 2.0,
        lignin_struc=np.ones((npts, nvm, nlevels), dtype=np.float64) * 0.3,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        soilc_total=deepC_a + deepC_s + deepC_p,
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 2.0,
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 3.0,
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 4.0,
        co2_to_bm=two_d,
        gpp_daily=two_d + 1.0,
        npp_daily=two_d + 2.0,
        resp_maint=two_d + 3.0,
        resp_growth=two_d + 4.0,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        leaf_frac=leaf,
        leaf_age=leaf,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        when_growthinit=two_d,
    )
    coeff = np.ones(nvm, dtype=np.float64) * 0.1
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    cn_sapl = np.ones(nvm, dtype=np.float64)
    resp_hetero = two_d + 5.0
    co2_fire = two_d + 6.0

    expected = lcc_gross_glcchange_step(
        veget_max_org=veget_max,
        harvest_matrix=shifts,
        glcc_second_shift=shifts,
        glcc_primary_shift=glcc_primary,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=pft_to_mtc,
        agec_group=agec_group,
        biomass=state.biomass,
        bm_to_litter=state.bm_to_litter,
        carbon=state.carbon,
        litter=state.litter,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        lignin_struc=state.lignin_struc,
        co2_to_bm=state.co2_to_bm,
        gpp_daily=state.gpp_daily,
        npp_daily=state.npp_daily,
        resp_maint=state.resp_maint,
        resp_growth=state.resp_growth,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        convflux=np.zeros((npts, 2), dtype=np.float64),
        prod10=state.prod10,
        prod100=state.prod100,
        flux10=state.flux10,
        flux100=state.flux100,
        cflux_prod10=np.zeros((npts, 2), dtype=np.float64),
        cflux_prod100=np.zeros((npts, 2), dtype=np.float64),
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=12.0,
        npp_longterm=state.npp_longterm,
        ind=state.ind,
        lm_lastyearmax=state.lm_lastyearmax,
        age=state.age,
        everywhere=state.everywhere,
        leaf_frac=state.leaf_frac,
        leaf_age=state.leaf_age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        gpp_week=season.gpp_week,
        when_growthinit=state.when_growthinit,
        gdd_from_growthinit=season.gdd_from_growthinit,
        gdd_midwinter=season.gdd_midwinter,
        time_hum_min=season.time_hum_min,
        gdd_m5_dormance=season.gdd_m5_dormance,
        ncd_dormance=season.ncd_dormance,
        moiavail_month=season.moiavail_month,
        moiavail_week=season.moiavail_week,
        ngd_minus5=season.ngd_minus5,
        dt_days=30.0,
        one_year=365.0,
        nagec_tree=1,
        nagec_herb=1,
    )

    result = stomate_lcc_gross_glcchange_from_restart_state(
        state=state,
        season=season,
        veget_max=veget_max,
        harvest_matrix=shifts,
        glcc_second_shift=shifts,
        glcc_primary_shift=glcc_primary,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=pft_to_mtc,
        agec_group=agec_group,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=12.0,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        dt_days=30.0,
        one_year=365.0,
        nagec_tree=1,
        nagec_herb=1,
    )

    assert np.allclose(np.asarray(result.lcc.applied.state.carbon), np.asarray(expected.applied.state.carbon))
    assert np.allclose(np.asarray(result.lcc.applied.state.deepC_a), np.asarray(expected.applied.state.deepC_a))
    assert np.allclose(np.asarray(result.lcc.applied.prod10), np.asarray(expected.applied.prod10))
    assert np.allclose(np.asarray(result.prod10_total), np.sum(np.asarray(expected.applied.prod10), axis=(1, 2)))
    assert np.allclose(np.asarray(result.state_after.carbon), np.asarray(expected.applied.state.carbon))
    assert np.allclose(np.asarray(result.state_after.deepC_a), np.asarray(expected.applied.state.deepC_a))
    assert np.allclose(np.asarray(result.state_after.soilc_total), np.asarray(result.state_after.deepC_a + result.state_after.deepC_s + result.state_after.deepC_p))
    assert np.allclose(np.asarray(result.state_after.prod10), np.asarray(expected.applied.prod10))
    assert np.allclose(np.asarray(result.state_after.prod10_total), np.asarray(result.prod10_total))
    assert np.allclose(np.asarray(result.season_after.gpp_week), np.asarray(expected.applied.gpp_week))


def test_lcc_gross_glcchange_restart_adapter_requires_process_outputs():
    state = stomate_cold_start_entry_state(t2m=np.asarray([289.0]), nvm=2, nslm=2, ndeep=2)
    season = stomate_cold_start_season_state(t2m=np.asarray([289.0]), dt_days=1.0, nvm=2, nslm=2)

    with pytest.raises(ValueError, match="resp_hetero"):
        stomate_lcc_gross_glcchange_from_restart_state(
            state=state,
            season=season,
            veget_max=np.zeros((1, 2), dtype=np.float64),
            harvest_matrix=np.zeros((1, 1), dtype=np.float64),
            glcc_second_shift=np.zeros((1, 1), dtype=np.float64),
            glcc_primary_shift=np.zeros((1, 1), dtype=np.float64),
            glcc_net_lcc=np.zeros((1, 1), dtype=np.float64),
            start_index=[0, 1],
            nagec_pft=[1, 1],
            is_tree=np.zeros(2, dtype=bool),
            natural=np.zeros(2, dtype=bool),
            is_grassland_manag=np.zeros(2, dtype=bool),
            pft_to_mtc=np.asarray([0, 1], dtype=np.int32),
            agec_group=np.asarray([0, 1], dtype=np.int32),
            coeff_lcchange_1=np.zeros(2, dtype=np.float64),
            coeff_lcchange_10=np.zeros(2, dtype=np.float64),
            coeff_lcchange_100=np.zeros(2, dtype=np.float64),
            bm_sapl=np.zeros((2, NPARTS, 1), dtype=np.float64),
            cn_sapl=np.ones(2, dtype=np.float64),
            npp_longterm_init=0.0,
            resp_hetero=None,
            co2_fire=np.zeros((1, 2), dtype=np.float64),
            dt_days=1.0,
        )


def _minimal_gross_lcc_restart_case():
    npts, nvm, ndeep, nelements = 1, 5, 2, 1
    start_index = [0, 1, 2, 3, 4]
    nagec_pft = [1, 1, 1, 1, 1]
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[2] = True
    is_grassland_manag[3] = True
    pft_to_mtc = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    agec_group = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.asarray([[0.0, 0.5, 0.4, 0.3, 0.2]], dtype=np.float64)
    shifts = np.zeros((npts, 12), dtype=np.float64)
    glcc_primary = shifts.copy()
    glcc_net = shifts.copy()
    glcc_primary[0, 2] = 0.2
    glcc_net[0, 2] = 0.1

    state = stomate_cold_start_entry_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        nvm=nvm,
        nslm=2,
        ndeep=ndeep,
        nelements=nelements,
    )
    season = stomate_cold_start_season_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        dt_days=30.0,
        nvm=nvm,
        nslm=2,
    )
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[0, 1, ISAPABOVE, ICARBON] = 12.0
    biomass[0, 1, ISAPBELOW, ICARBON] = 6.0
    biomass[0, 1, IHEARTABOVE, ICARBON] = 18.0
    biomass[0, 2, ILEAF, ICARBON] = 4.0
    two_d = np.ones((npts, nvm), dtype=np.float64)
    state = state._replace(
        biomass=biomass,
        bm_to_litter=np.ones_like(biomass) * 0.2,
        carbon=np.arange(npts * NCARB * nvm, dtype=np.float64).reshape(npts, NCARB, nvm) + 10.0,
        litter=np.ones((npts, NLITT, nvm, 2, nelements), dtype=np.float64) * 2.0,
        lignin_struc=np.ones((npts, nvm, 2), dtype=np.float64) * 0.3,
        deepC_a=np.ones((npts, ndeep, nvm), dtype=np.float64) * 20.0,
        deepC_s=np.ones((npts, ndeep, nvm), dtype=np.float64) * 40.0,
        deepC_p=np.ones((npts, ndeep, nvm), dtype=np.float64) * 60.0,
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 2.0,
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 3.0,
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 4.0,
        co2_to_bm=two_d,
        gpp_daily=two_d + 1.0,
        npp_daily=two_d + 2.0,
        resp_maint=two_d + 3.0,
        resp_growth=two_d + 4.0,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        when_growthinit=two_d,
    )
    coeff = np.ones(nvm, dtype=np.float64) * 0.1
    return {
        "state": state,
        "season": season,
        "veget_max": veget_max,
        "harvest_matrix": shifts,
        "glcc_second_shift": shifts,
        "glcc_primary_shift": glcc_primary,
        "glcc_net_lcc": glcc_net,
        "start_index": start_index,
        "nagec_pft": nagec_pft,
        "is_tree": is_tree,
        "natural": natural,
        "is_grassland_manag": is_grassland_manag,
        "pft_to_mtc": pft_to_mtc,
        "agec_group": agec_group,
        "coeff_lcchange_1": coeff,
        "coeff_lcchange_10": coeff,
        "coeff_lcchange_100": coeff,
        "bm_sapl": np.zeros((nvm, NPARTS, nelements), dtype=np.float64),
        "cn_sapl": np.ones(nvm, dtype=np.float64),
        "npp_longterm_init": 12.0,
        "resp_hetero": two_d + 5.0,
        "co2_fire": two_d + 6.0,
        "dt_days": 30.0,
        "one_year": 365.0,
        "nagec_tree": 1,
        "nagec_herb": 1,
    }


def test_daily_lcc_gross_glcchange_scheduling_matches_restart_adapter_and_normalizes_cover():
    args = _minimal_gross_lcc_restart_case()

    expected = stomate_lcc_gross_glcchange_from_restart_state(**args)
    result = stomate_daily_lcc_gross_glcchange_explicit(
        do_now_stomate_lcchange=True,
        **args,
    )

    assert result.active is True
    assert result.do_now_stomate_lcchange_after is False
    assert result.done_stomate_lcchange is True
    assert result.lcc is not None
    assert np.allclose(np.asarray(result.lcc.lcc.applied.state.carbon), np.asarray(expected.lcc.applied.state.carbon))
    assert np.allclose(np.asarray(result.state_after.carbon), np.asarray(expected.state_after.carbon))
    assert np.allclose(np.asarray(result.state_after.deepC_a), np.asarray(expected.state_after.deepC_a))
    assert np.allclose(np.asarray(result.state_after.prod10), np.asarray(expected.state_after.prod10))
    assert np.allclose(np.asarray(result.prod10_total), np.asarray(expected.prod10_total))
    assert np.allclose(np.asarray(result.season_after.gpp_week), np.asarray(expected.season_after.gpp_week))
    assert np.allclose(np.asarray(result.veget_max_after).sum(axis=1), np.ones(1))


def test_daily_lcc_gross_glcchange_deforest_fire_branch_uses_source_proxy_boundary():
    """Fortran: stomate_lpj.f90 lines 835-859 and 1446-1464."""

    args = _minimal_gross_lcc_restart_case()

    no_fire = stomate_lcc_gross_glcchange_from_restart_state(**args)
    expected = stomate_lcc_gross_glcchange_from_restart_state(**args, allow_deforest_fire=True)
    result = stomate_daily_lcc_gross_glcchange_explicit(
        do_now_stomate_lcchange=True,
        allow_deforest_fire=True,
        **args,
    )

    assert result.active is True
    assert result.done_stomate_lcchange is True
    assert np.any(np.asarray(result.lcc.lcc.allocation.glcc_pftmtc)[0, 1, 2:] > 0.0)
    assert np.allclose(np.asarray(result.state_after.prod10), np.asarray(expected.state_after.prod10))
    assert np.allclose(np.asarray(result.state_after.prod100), np.asarray(expected.state_after.prod100))
    assert np.allclose(np.asarray(result.prod10_total), np.asarray(expected.prod10_total))
    assert np.allclose(np.asarray(result.prod100_total), np.asarray(expected.prod100_total))
    assert np.asarray(result.prod10_total)[0] > np.asarray(no_fire.prod10_total)[0]
    assert np.asarray(result.prod100_total)[0] > np.asarray(no_fire.prod100_total)[0]
    assert np.allclose(np.asarray(result.veget_max_after).sum(axis=1), np.ones(1))


def test_daily_lcc_gross_glcchange_single_age_class_degenerate_branch_matches_gross_boundary():
    """Fortran: stomate_lpj.f90 lines 1423-1428 with one age class per MTC."""

    args = _minimal_gross_lcc_restart_case()

    expected = stomate_lcc_gross_glcchange_from_restart_state(**args)
    result = stomate_daily_lcc_gross_glcchange_explicit(
        do_now_stomate_lcchange=True,
        single_age_class=True,
        **args,
    )

    assert result.active is True
    assert result.done_stomate_lcchange is True
    assert all(int(count) == 1 for count in args["nagec_pft"])
    assert np.allclose(np.asarray(result.state_after.biomass), np.asarray(expected.state_after.biomass))
    assert np.allclose(np.asarray(result.state_after.prod10), np.asarray(expected.state_after.prod10))
    assert np.allclose(np.asarray(result.season_after.gpp_week), np.asarray(expected.season_after.gpp_week))
    assert np.allclose(np.asarray(result.veget_max_after).sum(axis=1), np.ones(1))


def test_daily_lcc_gross_glcchange_single_age_class_multicohort_uses_sinagec_allocation():
    """Fortran: stomate_glcchange_SinAgeC_fh.f90 lines 533-905 and 1358-1950."""

    npts, nvm, ndeep, nelements = 1, 8, 2, 1
    start_index = [0, 1, 5, 6, 7]
    nagec_pft = [1, 4, 1, 1, 1]
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1:5] = True
    natural[5] = True
    is_grassland_manag[6] = True
    agec_group = np.asarray([0, 1, 1, 1, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, 1:5] = [0.05, 0.1, 0.2, 0.4]
    veget_max[0, 7] = 0.2
    shifts = np.zeros((npts, 12), dtype=np.float64)
    glcc_net = shifts.copy()
    glcc_net[0, 2] = 0.5
    state = stomate_cold_start_entry_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        nvm=nvm,
        nslm=2,
        ndeep=ndeep,
        nelements=nelements,
    )
    season = stomate_cold_start_season_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        dt_days=30.0,
        nvm=nvm,
        nslm=2,
    )
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[0, 4, ISAPABOVE, ICARBON] = 12.0
    biomass[0, 4, ISAPBELOW, ICARBON] = 6.0
    biomass[0, 4, IHEARTABOVE, ICARBON] = 18.0
    biomass[0, 3, ILEAF, ICARBON] = 4.0
    two_d = np.ones((npts, nvm), dtype=np.float64)
    state = state._replace(
        biomass=biomass,
        bm_to_litter=np.ones_like(biomass) * 0.2,
        carbon=np.arange(npts * NCARB * nvm, dtype=np.float64).reshape(npts, NCARB, nvm) + 10.0,
        litter=np.ones((npts, NLITT, nvm, 2, nelements), dtype=np.float64) * 2.0,
        lignin_struc=np.ones((npts, nvm, 2), dtype=np.float64) * 0.3,
        deepC_a=np.ones((npts, ndeep, nvm), dtype=np.float64) * 20.0,
        deepC_s=np.ones((npts, ndeep, nvm), dtype=np.float64) * 40.0,
        deepC_p=np.ones((npts, ndeep, nvm), dtype=np.float64) * 60.0,
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 2.0,
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 3.0,
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 4.0,
        co2_to_bm=two_d,
        gpp_daily=two_d + 1.0,
        npp_daily=two_d + 2.0,
        resp_maint=two_d + 3.0,
        resp_growth=two_d + 4.0,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        when_growthinit=two_d,
    )
    args = {
        "state": state,
        "season": season,
        "veget_max": veget_max,
        "harvest_matrix": shifts,
        "glcc_second_shift": shifts,
        "glcc_primary_shift": shifts,
        "glcc_net_lcc": glcc_net,
        "start_index": start_index,
        "nagec_pft": nagec_pft,
        "is_tree": is_tree,
        "natural": natural,
        "is_grassland_manag": is_grassland_manag,
        "pft_to_mtc": agec_group,
        "agec_group": agec_group,
        "coeff_lcchange_1": np.ones(nvm, dtype=np.float64) * 0.1,
        "coeff_lcchange_10": np.ones(nvm, dtype=np.float64) * 0.1,
        "coeff_lcchange_100": np.ones(nvm, dtype=np.float64) * 0.1,
        "bm_sapl": np.zeros((nvm, NPARTS, nelements), dtype=np.float64),
        "cn_sapl": np.ones(nvm, dtype=np.float64),
        "npp_longterm_init": 12.0,
        "resp_hetero": two_d + 5.0,
        "co2_fire": two_d + 6.0,
        "dt_days": 30.0,
        "one_year": 365.0,
        "nagec_tree": 4,
        "nagec_herb": 1,
        "single_age_class": True,
    }

    expected = stomate_lcc_gross_glcchange_from_restart_state(**args)
    result = stomate_daily_lcc_gross_glcchange_explicit(
        do_now_stomate_lcchange=True,
        **args,
    )

    assert result.active is True
    assert result.done_stomate_lcchange is True
    assert hasattr(result.lcc.lcc.allocation, "hmatrix_real")
    assert np.allclose(np.asarray(result.lcc.lcc.allocation.glcc_real)[0, 2], 0.5)
    assert np.allclose(np.asarray(result.lcc.lcc.allocation.glcc_pft)[0, [4, 3]], [0.4, 0.1])
    assert np.allclose(np.asarray(result.state_after.biomass), np.asarray(expected.state_after.biomass))
    assert np.allclose(np.asarray(result.state_after.prod10), np.asarray(expected.state_after.prod10))
    assert np.allclose(np.asarray(result.prod10_total), np.asarray(expected.prod10_total))
    assert np.allclose(np.asarray(result.veget_max_after).sum(axis=1), np.ones(1))

    fire_expected = stomate_lcc_gross_glcchange_from_restart_state(**args, allow_deforest_fire=True)
    fire_result = stomate_daily_lcc_gross_glcchange_explicit(
        do_now_stomate_lcchange=True,
        allow_deforest_fire=True,
        **args,
    )

    assert fire_result.active is True
    assert hasattr(fire_result.lcc.lcc.allocation, "hmatrix_real")
    assert np.allclose(np.asarray(fire_result.state_after.prod10), np.asarray(fire_expected.state_after.prod10))
    assert np.allclose(np.asarray(fire_result.state_after.prod100), np.asarray(fire_expected.state_after.prod100))
    assert np.asarray(fire_result.prod10_total)[0] > np.asarray(result.prod10_total)[0]
    assert np.asarray(fire_result.prod100_total)[0] > np.asarray(result.prod100_total)[0]


def test_daily_lcc_gross_glcchange_scheduling_consumes_local_prep_handoff():
    args = _minimal_gross_lcc_restart_case()
    local_prep = StomateEntryLocalPrepResult(
        precip=np.zeros((1,), dtype=np.float64),
        veget_cov=args["veget_max"],
        veget_cov_max=args["veget_max"],
        gpp_d=np.zeros_like(args["veget_max"]),
        glccNetLCC=args["glcc_net_lcc"],
        glccSecondShift=args["glcc_second_shift"],
        glccPrimaryShift=args["glcc_primary_shift"],
        harvest_matrix=args["harvest_matrix"],
    )
    explicit_args = {
        name: value
        for name, value in args.items()
        if name not in {"veget_max", "harvest_matrix", "glcc_second_shift", "glcc_primary_shift", "glcc_net_lcc"}
    }

    expected = stomate_lcc_gross_glcchange_from_restart_state(**args)
    result = stomate_daily_lcc_gross_glcchange_explicit(
        do_now_stomate_lcchange=True,
        local_prep=local_prep,
        **explicit_args,
    )

    assert result.active is True
    assert np.allclose(np.asarray(result.state_after.carbon), np.asarray(expected.state_after.carbon))
    assert np.allclose(np.asarray(result.prod100_total), np.asarray(expected.prod100_total))
    assert np.allclose(np.asarray(result.veget_max_after).sum(axis=1), np.ones(1))


def test_daily_lcc_gross_glcchange_age_class_ignores_peat_switches_after_cover_boundary():
    """Fortran: stomate_lpj.f90 lines 1418-1465 dispatch age-class gross LCC before peat ordinary-LCC switches."""

    args = _minimal_gross_lcc_restart_case()
    expected = stomate_lcc_gross_glcchange_from_restart_state(**args)
    expected_daily = stomate_daily_lcc_gross_glcchange_explicit(
        do_now_stomate_lcchange=True,
        **args,
    )

    result = stomate_daily_lcc_gross_glcchange_explicit(
        do_now_stomate_lcchange=True,
        dyn_peat=True,
        agri_peat=True,
        agri_peat_prop=True,
        **args,
    )

    assert result.active is True
    assert result.done_stomate_lcchange is True
    np.testing.assert_allclose(np.asarray(result.state_after.biomass), np.asarray(expected.state_after.biomass))
    np.testing.assert_allclose(np.asarray(result.state_after.carbon), np.asarray(expected.state_after.carbon))
    np.testing.assert_allclose(np.asarray(result.prod10_total), np.asarray(expected.prod10_total))
    np.testing.assert_allclose(np.asarray(result.veget_max_after), np.asarray(expected_daily.veget_max_after))


def test_daily_lcc_gross_glcchange_inactive_path_leaves_state_unchanged():
    args = _minimal_gross_lcc_restart_case()
    state = args["state"]
    season = args["season"]

    result = stomate_daily_lcc_gross_glcchange_explicit(
        state=state,
        season=season,
        veget_max=args["veget_max"],
        do_now_stomate_lcchange=False,
    )

    assert result.active is False
    assert result.lcc is None
    assert result.state_after is state
    assert result.season_after is season
    assert result.done_stomate_lcchange is False
    assert np.allclose(np.asarray(result.veget_max_after), args["veget_max"])


def test_daily_lcc_gross_glcchange_rejects_missing_active_inputs_and_unsupported_branches():
    args = _minimal_gross_lcc_restart_case()
    missing = dict(args)
    missing["harvest_matrix"] = None
    with pytest.raises(ValueError, match="harvest_matrix"):
        stomate_daily_lcc_gross_glcchange_explicit(
            do_now_stomate_lcchange=True,
            **missing,
        )

    with pytest.raises(ValueError, match="cn_ind"):
        stomate_daily_lcc_gross_glcchange_explicit(
            do_now_stomate_lcchange=True,
            use_age_class=False,
            **args,
        )
    with pytest.raises(ValueError, match="lcc"):
        stomate_daily_lcc_gross_glcchange_explicit(
            do_now_stomate_lcchange=True,
            use_age_class=False,
            allow_deforest_fire=True,
            cn_ind=np.ones_like(args["veget_max"]),
            veget_max_old=args["veget_max"],
            **args,
        )


def test_daily_lcc_non_age_class_dyn_peat_skips_lcchange_but_consumes_schedule():
    """Fortran: stomate_lpj.f90 lines 1494-1517 skip lcchange_main under dyn_peat."""

    npts, nvm = 1, 4
    state = stomate_cold_start_entry_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        nvm=nvm,
        nslm=2,
        ndeep=2,
        nelements=1,
    )
    prod10 = np.ones((npts, 11, 2), dtype=np.float64)
    prod100 = np.ones((npts, 101, 2), dtype=np.float64) * 2.0
    state = state._replace(prod10=prod10, prod100=prod100)
    season = stomate_cold_start_season_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        dt_days=30.0,
        nvm=nvm,
        nslm=2,
    )
    current_veget = np.asarray([[0.0, 0.2, 0.3, 0.1]], dtype=np.float64)
    target_veget = np.asarray([[0.0, 0.7, 0.2, 0.1]], dtype=np.float64)

    result = stomate_daily_lcc_gross_glcchange_explicit(
        state=state,
        season=season,
        do_now_stomate_lcchange=True,
        use_age_class=False,
        dyn_peat=True,
        veget_max=target_veget,
        veget_max_old=current_veget,
    )

    expected_veget = current_veget / current_veget.sum(axis=1, keepdims=True)
    assert result.active is True
    assert result.lcc is None
    assert result.do_now_stomate_lcchange_after is False
    assert result.done_stomate_lcchange is True
    np.testing.assert_allclose(np.asarray(result.veget_max_after), expected_veget)
    np.testing.assert_allclose(np.asarray(result.state_after.prod10)[:, 0, :], 0.0)
    np.testing.assert_allclose(np.asarray(result.state_after.prod100)[:, 0, :], 0.0)
    np.testing.assert_allclose(np.asarray(result.state_after.prod10)[:, 1:, :], prod10[:, 1:, :])
    np.testing.assert_allclose(np.asarray(result.state_after.biomass), np.asarray(state.biomass))
    np.testing.assert_allclose(np.asarray(result.prod10_total), np.sum(np.asarray(result.state_after.prod10), axis=(1, 2)))


def test_daily_lcc_non_age_class_dispatch_writes_lcchange_main_leak_state():
    """Fortran: stomate_lpj.f90 lines 1485-1511 dispatch lcchange_main."""

    npts, nvm, ndeep, nelements = 1, 4, 2, 1
    state = stomate_cold_start_entry_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        nvm=nvm,
        nslm=2,
        ndeep=ndeep,
        nelements=nelements,
    )
    veget_old = np.asarray([[0.0, 0.4, 0.1, 0.0]], dtype=np.float64)
    veget_new = np.asarray([[0.0, 0.2, 0.3, 0.0]], dtype=np.float64)
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[0, 1, ISAPABOVE, ICARBON] = 10.0
    biomass[0, 1, IHEARTABOVE, ICARBON] = 20.0
    biomass[0, 1, ISAPBELOW, ICARBON] = 5.0
    biomass[0, 2, ILEAF, ICARBON] = 2.0
    biomass[0, 2, IROOT, ICARBON] = 3.0
    bm_to_litter = np.ones_like(biomass) * 0.1
    turnover_daily = np.ones_like(biomass) * 0.05
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    flux10 = np.zeros((npts, 10, 2), dtype=np.float64)
    flux100 = np.zeros((npts, 100, 2), dtype=np.float64)
    prod10[0, 0, 1] = 99.0
    prod100[0, 0, 1] = 77.0
    litter_above = np.ones((npts, NLITT, nvm, nelements), dtype=np.float64)
    litter_above[0, :, 1, ICARBON] = [8.0, 9.0]
    litter_above[0, :, 2, ICARBON] = [1.0, 2.0]
    litter_below = np.ones((npts, NLITT, nvm, ndeep, nelements), dtype=np.float64)
    carbon_32l = np.ones((npts, NCARB, nvm, ndeep), dtype=np.float64)
    carbon_32l[0, :, 1, :] = 4.0
    doc = np.ones((npts, nvm, ndeep, 2, NPOOL, nelements), dtype=np.float64)
    deepC_a = np.ones((npts, ndeep, nvm), dtype=np.float64)
    deepC_s = deepC_a + 2.0
    deepC_p = deepC_a + 4.0
    deepC_a[0, :, 1] = [10.0, 20.0]
    deepC_s[0, :, 1] = [30.0, 40.0]
    deepC_p[0, :, 1] = [50.0, 60.0]
    state = state._replace(
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        turnover_daily=turnover_daily,
        ind=np.ones((npts, nvm), dtype=np.float64),
        age=np.ones((npts, nvm), dtype=np.float64) * 3.0,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        when_growthinit=np.ones((npts, nvm), dtype=np.float64),
        everywhere=np.ones((npts, nvm), dtype=np.float64),
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        carbon=np.ones((npts, NCARB, nvm), dtype=np.float64),
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 2.0,
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 3.0,
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 4.0,
        litter_above=litter_above,
        litter_below=litter_below,
        carbon_32l=carbon_32l,
        DOC=doc,
    )
    season = stomate_cold_start_season_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        dt_days=30.0,
        nvm=nvm,
        nslm=2,
    )
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[2, :, ICARBON] = np.arange(1.0, NPARTS + 1.0)
    cn_ind = np.asarray([[0.0, 0.5, 1.0, 0.0]], dtype=np.float64)
    coeff = np.asarray([0.0, 0.2, 0.3, 0.4], dtype=np.float64)
    is_tree = np.asarray([False, True, False, False])
    is_grass = np.zeros(nvm, dtype=bool)

    expected = lcchange_main_leak_step(
        dt_days=30.0,
        veget_max=veget_new,
        veget_max_old=veget_old,
        biomass=state.biomass,
        ind=state.ind,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        co2_to_bm=state.co2_to_bm,
        bm_to_litter=state.bm_to_litter,
        turnover_daily=state.turnover_daily,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=state.flux10[:, :, IWPLCC],
        flux100=state.flux100[:, :, IWPLCC],
        prod10=state.prod10[:, :, IWPLCC],
        prod100=state.prod100[:, :, IWPLCC],
        leaf_frac=state.leaf_frac,
        npp_longterm=state.npp_longterm,
        lm_lastyearmax=state.lm_lastyearmax,
        litter_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        litter_not_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        carbon=state.carbon,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        litter_above=state.litter_above,
        litter_below=state.litter_below,
        carbon_32l=state.carbon_32l,
        doc=state.DOC,
        cn_sapl=np.ones(nvm, dtype=np.float64),
        is_tree=is_tree,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        is_grassland_manag=is_grass,
    )
    result = stomate_daily_lcc_gross_glcchange_explicit(
        state=state,
        season=season,
        do_now_stomate_lcchange=True,
        use_age_class=False,
        veget_max=veget_new,
        veget_max_old=veget_old,
        is_tree=is_tree,
        is_grassland_manag=is_grass,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=np.ones(nvm, dtype=np.float64),
        cn_ind=cn_ind,
        npp_longterm_init=10.0,
        dt_days=30.0,
    )

    assert result.active is True
    assert result.done_stomate_lcchange is True
    assert np.allclose(np.asarray(result.state_after.biomass), np.asarray(expected.biomass))
    assert np.allclose(np.asarray(result.state_after.litter_above), np.asarray(expected.litter_above))
    assert np.allclose(np.asarray(result.state_after.carbon_32l), np.asarray(expected.carbon_32l))
    assert np.allclose(np.asarray(result.state_after.DOC), np.asarray(expected.doc))
    assert np.allclose(np.asarray(result.state_after.prod10)[:, :, IWPLCC], np.asarray(expected.prod10))
    assert np.allclose(np.asarray(result.state_after.prod100)[:, :, IWPLCC], np.asarray(expected.prod100))
    assert np.allclose(np.asarray(result.state_after.prod10)[:, :, 1], prod10[:, :, 1])
    assert np.allclose(np.asarray(result.veget_max_after).sum(axis=1), np.ones(npts))

    ok_pc_expected = lcchange_main_leak_step(
        dt_days=30.0,
        veget_max=veget_new,
        veget_max_old=veget_old,
        biomass=state.biomass,
        ind=state.ind,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        co2_to_bm=state.co2_to_bm,
        bm_to_litter=state.bm_to_litter,
        turnover_daily=state.turnover_daily,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=state.flux10[:, :, IWPLCC],
        flux100=state.flux100[:, :, IWPLCC],
        prod10=state.prod10[:, :, IWPLCC],
        prod100=state.prod100[:, :, IWPLCC],
        leaf_frac=state.leaf_frac,
        npp_longterm=state.npp_longterm,
        lm_lastyearmax=state.lm_lastyearmax,
        litter_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        litter_not_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        carbon=state.carbon,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        litter_above=state.litter_above,
        litter_below=state.litter_below,
        carbon_32l=state.carbon_32l,
        doc=state.DOC,
        cn_sapl=np.ones(nvm, dtype=np.float64),
        is_tree=is_tree,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        is_grassland_manag=is_grass,
        ok_pc=True,
    )
    ok_pc_result = stomate_daily_lcc_gross_glcchange_explicit(
        state=state,
        season=season,
        do_now_stomate_lcchange=True,
        use_age_class=False,
        veget_max=veget_new,
        veget_max_old=veget_old,
        is_tree=is_tree,
        is_grassland_manag=is_grass,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=np.ones(nvm, dtype=np.float64),
        cn_ind=cn_ind,
        npp_longterm_init=10.0,
        dt_days=30.0,
        ok_pc=True,
    )
    np.testing.assert_allclose(np.asarray(ok_pc_result.state_after.deepC_a), np.asarray(ok_pc_expected.deepC_a))
    np.testing.assert_allclose(np.asarray(ok_pc_result.state_after.deepC_s), np.asarray(ok_pc_expected.deepC_s))
    np.testing.assert_allclose(np.asarray(ok_pc_result.state_after.deepC_p), np.asarray(ok_pc_expected.deepC_p))


def test_daily_lcc_non_age_class_deffire_dispatch_writes_legacy_litter_state():
    """Fortran: stomate_lpj.f90 lines 1469-1483 dispatch lcchange_deffire."""

    npts, nvm, ndeep, nelements = 1, 4, 2, 1
    state = stomate_cold_start_entry_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        nvm=nvm,
        nslm=2,
        ndeep=ndeep,
        nelements=nelements,
    )
    veget_old = np.asarray([[0.0, 0.4, 0.1, 0.0]], dtype=np.float64)
    veget_new = np.asarray([[0.0, 0.2, 0.3, 0.0]], dtype=np.float64)
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[0, 1, ISAPABOVE, ICARBON] = 10.0
    biomass[0, 1, IHEARTABOVE, ICARBON] = 20.0
    biomass[0, 1, ISAPBELOW, ICARBON] = 5.0
    biomass[0, 2, ILEAF, ICARBON] = 2.0
    biomass[0, 2, IROOT, ICARBON] = 3.0
    litter = np.zeros((npts, NLITT, nvm, NLEVS, nelements), dtype=np.float64)
    litter[0, :, 1, IABOVE, ICARBON] = [20.0, 30.0]
    litter[0, :, 1, IBELOW, ICARBON] = [40.0, 50.0]
    litter[0, :, 2, IABOVE, ICARBON] = [2.0, 3.0]
    litter[0, :, 2, IBELOW, ICARBON] = [4.0, 5.0]
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    flux10 = np.zeros((npts, 10, 2), dtype=np.float64)
    flux100 = np.zeros((npts, 100, 2), dtype=np.float64)
    state = state._replace(
        biomass=biomass,
        bm_to_litter=np.ones_like(biomass) * 0.1,
        turnover_daily=np.ones_like(biomass) * 0.01,
        ind=np.ones((npts, nvm), dtype=np.float64),
        age=np.ones((npts, nvm), dtype=np.float64) * 4.0,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        when_growthinit=np.ones((npts, nvm), dtype=np.float64),
        everywhere=np.ones((npts, nvm), dtype=np.float64),
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        litter=litter,
        carbon=np.ones((npts, NCARB, nvm), dtype=np.float64),
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 2.0,
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 3.0,
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 4.0,
    )
    season = stomate_cold_start_season_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        dt_days=30.0,
        nvm=nvm,
        nslm=2,
    )
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[2, :, ICARBON] = np.arange(1.0, NPARTS + 1.0)
    cn_ind = np.asarray([[0.0, 0.5, 1.0, 0.0]], dtype=np.float64)
    is_tree = np.asarray([False, True, False, False])
    is_grass = np.zeros(nvm, dtype=bool)
    emilit = np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64)
    emilit[0, 1, :, ICARBON] = [1.0, 2.0]
    emibio = np.zeros_like(biomass)
    emibio[0, 1, ISAPABOVE, ICARBON] = 1.0
    emibio[0, 1, IHEARTABOVE, ICARBON] = 2.0
    deepC_a = np.ones((npts, ndeep, nvm), dtype=np.float64)
    deepC_s = np.ones((npts, ndeep, nvm), dtype=np.float64) * 2.0
    deepC_p = np.ones((npts, ndeep, nvm), dtype=np.float64) * 3.0
    deepC_a[0, :, 1] = [10.0, 20.0]
    deepC_s[0, :, 1] = [30.0, 40.0]
    deepC_p[0, :, 1] = [50.0, 60.0]
    deepC_a[0, :, 2] = [1.0, 2.0]
    deepC_s[0, :, 2] = [3.0, 4.0]
    deepC_p[0, :, 2] = [5.0, 6.0]
    state = state._replace(
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        soilc_total=deepC_a + deepC_s + deepC_p,
    )

    expected = lcchange_deffire_step(
        dt_days=30.0,
        veget_max=veget_old,
        veget_max_new=veget_new,
        biomass=state.biomass,
        ind=state.ind,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        co2_to_bm=state.co2_to_bm,
        bm_to_litter=state.bm_to_litter,
        turnover_daily=state.turnover_daily,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=state.flux10[:, :, IWPLCC],
        flux100=state.flux100[:, :, IWPLCC],
        prod10=state.prod10[:, :, IWPLCC],
        prod100=state.prod100[:, :, IWPLCC],
        leaf_frac=state.leaf_frac,
        npp_longterm=state.npp_longterm,
        lm_lastyearmax=state.lm_lastyearmax,
        litter=state.litter,
        litter_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        litter_not_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        carbon=state.carbon,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        lcc=veget_old - veget_new,
        bafrac_deforest_accu=np.zeros((npts, nvm), dtype=np.float64),
        emideforest_litter_accu=emilit,
        emideforest_biomass_accu=emibio,
        deflitsup_total=np.zeros((npts, nvm), dtype=np.float64),
        defbiosup_total=np.zeros((npts, nvm), dtype=np.float64),
        cn_sapl=np.ones(nvm, dtype=np.float64),
        is_tree=is_tree,
        is_grassland_manag=is_grass,
    )
    result = stomate_daily_lcc_gross_glcchange_explicit(
        state=state,
        season=season,
        do_now_stomate_lcchange=True,
        use_age_class=False,
        allow_deforest_fire=True,
        veget_max=veget_new,
        veget_max_old=veget_old,
        is_tree=is_tree,
        is_grassland_manag=is_grass,
        coeff_lcchange_1=np.zeros(nvm, dtype=np.float64),
        coeff_lcchange_10=np.zeros(nvm, dtype=np.float64),
        coeff_lcchange_100=np.zeros(nvm, dtype=np.float64),
        bm_sapl=bm_sapl,
        cn_sapl=np.ones(nvm, dtype=np.float64),
        cn_ind=cn_ind,
        npp_longterm_init=10.0,
        lcc=veget_old - veget_new,
        bafrac_deforest_accu=np.zeros((npts, nvm), dtype=np.float64),
        emideforest_litter_accu=emilit,
        emideforest_biomass_accu=emibio,
        deflitsup_total=np.zeros((npts, nvm), dtype=np.float64),
        defbiosup_total=np.zeros((npts, nvm), dtype=np.float64),
        dt_days=30.0,
    )

    assert result.active is True
    assert result.done_stomate_lcchange is True
    assert np.allclose(np.asarray(result.state_after.litter), np.asarray(expected.litter))
    assert np.allclose(np.asarray(result.state_after.biomass), np.asarray(expected.biomass))
    assert np.allclose(np.asarray(result.state_after.prod10)[:, :, IWPLCC], np.asarray(expected.prod10))
    assert np.allclose(np.asarray(result.state_after.prod100)[:, :, IWPLCC], np.asarray(expected.prod100))
    assert np.allclose(np.asarray(result.lcc.lcchange.deflitsup_total), np.asarray(expected.deflitsup_total))
    assert np.allclose(np.asarray(result.lcc.lcchange.defbiosup_total), np.asarray(expected.defbiosup_total))
    assert np.allclose(np.asarray(result.veget_max_after).sum(axis=1), np.ones(npts))

    ok_pc_expected = lcchange_deffire_step(
        dt_days=30.0,
        veget_max=veget_old,
        veget_max_new=veget_new,
        biomass=state.biomass,
        ind=state.ind,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        co2_to_bm=state.co2_to_bm,
        bm_to_litter=state.bm_to_litter,
        turnover_daily=state.turnover_daily,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=state.flux10[:, :, IWPLCC],
        flux100=state.flux100[:, :, IWPLCC],
        prod10=state.prod10[:, :, IWPLCC],
        prod100=state.prod100[:, :, IWPLCC],
        leaf_frac=state.leaf_frac,
        npp_longterm=state.npp_longterm,
        lm_lastyearmax=state.lm_lastyearmax,
        litter=state.litter,
        litter_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        litter_not_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        carbon=state.carbon,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        lcc=veget_old - veget_new,
        bafrac_deforest_accu=np.zeros((npts, nvm), dtype=np.float64),
        emideforest_litter_accu=emilit,
        emideforest_biomass_accu=emibio,
        deflitsup_total=np.zeros((npts, nvm), dtype=np.float64),
        defbiosup_total=np.zeros((npts, nvm), dtype=np.float64),
        cn_sapl=np.ones(nvm, dtype=np.float64),
        is_tree=is_tree,
        is_grassland_manag=is_grass,
        ok_pc=True,
    )
    ok_pc_result = stomate_daily_lcc_gross_glcchange_explicit(
        state=state,
        season=season,
        do_now_stomate_lcchange=True,
        use_age_class=False,
        allow_deforest_fire=True,
        veget_max=veget_new,
        veget_max_old=veget_old,
        is_tree=is_tree,
        is_grassland_manag=is_grass,
        coeff_lcchange_1=np.zeros(nvm, dtype=np.float64),
        coeff_lcchange_10=np.zeros(nvm, dtype=np.float64),
        coeff_lcchange_100=np.zeros(nvm, dtype=np.float64),
        bm_sapl=bm_sapl,
        cn_sapl=np.ones(nvm, dtype=np.float64),
        cn_ind=cn_ind,
        npp_longterm_init=10.0,
        lcc=veget_old - veget_new,
        bafrac_deforest_accu=np.zeros((npts, nvm), dtype=np.float64),
        emideforest_litter_accu=emilit,
        emideforest_biomass_accu=emibio,
        deflitsup_total=np.zeros((npts, nvm), dtype=np.float64),
        defbiosup_total=np.zeros((npts, nvm), dtype=np.float64),
        dt_days=30.0,
        ok_pc=True,
    )
    np.testing.assert_allclose(np.asarray(ok_pc_result.state_after.deepC_a), np.asarray(ok_pc_expected.deepC_a))
    np.testing.assert_allclose(np.asarray(ok_pc_result.state_after.deepC_s), np.asarray(ok_pc_expected.deepC_s))
    np.testing.assert_allclose(np.asarray(ok_pc_result.state_after.deepC_p), np.asarray(ok_pc_expected.deepC_p))
    np.testing.assert_allclose(np.asarray(ok_pc_result.state_after.soilc_total), np.asarray(ok_pc_expected.deepC_a + ok_pc_expected.deepC_s + ok_pc_expected.deepC_p))


@pytest.mark.parametrize("strategy", ["agri_peat_prop", "agri_peat_mincrop", "agri_peat_maxcrop"])
def test_daily_lcc_non_age_class_agripeat_dispatch_writes_hardcoded_peat_state(strategy):
    """Fortran: stomate_lpj.f90 lines 1494-1505 dispatch lcchange_main_agripeat."""

    npts, nvm, ndeep, nelements = 1, 16, 2, 1
    state = stomate_cold_start_entry_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        nvm=nvm,
        nslm=2,
        ndeep=ndeep,
        nelements=nelements,
    )
    veget_old = np.zeros((npts, nvm), dtype=np.float64)
    veget_new = np.zeros((npts, nvm), dtype=np.float64)
    veget_old[0, [0, 11, 12, 13, 14, 15]] = [0.4, 0.1, 0.1, 0.3, 0.05, 0.05]
    veget_new[0, [0, 11, 12, 13, 14, 15]] = [0.35, 0.15, 0.15, 0.2, 0.1, 0.05]

    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[0, 13, ILEAF, ICARBON] = 10.0
    biomass[0, 13, IROOT, ICARBON] = 15.0
    biomass[0, 14, ILEAF, ICARBON] = 1.0
    biomass[0, 15, IROOT, ICARBON] = 2.0
    bm_to_litter = np.ones_like(biomass) * 0.05
    turnover_daily = np.ones_like(biomass) * 0.01
    litter = np.ones((npts, NLITT, nvm, NLEVS, nelements), dtype=np.float64) * 0.5
    litter[0, :, 13, IABOVE, ICARBON] = [20.0, 30.0]
    litter[0, :, 13, IBELOW, ICARBON] = [40.0, 50.0]
    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64)
    carbon[0, :, 13] = [100.0, 200.0, 300.0]
    deepC_a = np.ones((npts, ndeep, nvm), dtype=np.float64) * 2.0
    deepC_s = np.ones((npts, ndeep, nvm), dtype=np.float64) * 3.0
    deepC_p = np.ones((npts, ndeep, nvm), dtype=np.float64) * 4.0
    deepC_a[0, :, 13] = [10.0, 20.0]
    deepC_s[0, :, 13] = [30.0, 40.0]
    deepC_p[0, :, 13] = [50.0, 60.0]
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    flux10 = np.zeros((npts, 10, 2), dtype=np.float64)
    flux100 = np.zeros((npts, 100, 2), dtype=np.float64)
    prod10[0, 0, 1] = 99.0
    prod100[0, 0, 1] = 77.0
    state = state._replace(
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        turnover_daily=turnover_daily,
        ind=np.ones((npts, nvm), dtype=np.float64),
        age=np.ones((npts, nvm), dtype=np.float64) * 3.0,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        when_growthinit=np.ones((npts, nvm), dtype=np.float64),
        everywhere=np.ones((npts, nvm), dtype=np.float64),
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        litter=litter,
        carbon=carbon,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 2.0,
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 3.0,
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 4.0,
    )
    season = stomate_cold_start_season_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        dt_days=30.0,
        nvm=nvm,
        nslm=2,
    )
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[14, ILEAF, ICARBON] = 1.5
    bm_sapl[15, IROOT, ICARBON] = 2.5
    cn_ind = np.ones((npts, nvm), dtype=np.float64)
    cn_sapl = np.ones(nvm, dtype=np.float64)
    coeff = np.ones(nvm, dtype=np.float64) * 0.2
    natural = np.zeros(nvm, dtype=bool)
    natural[13] = True
    pasture = np.zeros(nvm, dtype=bool)
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[13] = True
    pft_to_mtc = np.arange(nvm, dtype=np.int32)
    pft_to_mtc[14] = 16
    pft_to_mtc[15] = 17
    is_tree = np.zeros(nvm, dtype=bool)
    is_grass = np.zeros(nvm, dtype=bool)
    strategy_kwargs = {strategy: True}

    expected = lcchange_main_agripeat_step(
        dt_days=30.0,
        veget_max=veget_new,
        veget_max_old=veget_old,
        biomass=state.biomass,
        ind=state.ind,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        co2_to_bm=state.co2_to_bm,
        bm_to_litter=state.bm_to_litter,
        turnover_daily=state.turnover_daily,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=state.flux10[:, :, IWPLCC],
        flux100=state.flux100[:, :, IWPLCC],
        prod10=state.prod10[:, :, IWPLCC],
        prod100=state.prod100[:, :, IWPLCC],
        leaf_frac=state.leaf_frac,
        npp_longterm=state.npp_longterm,
        lm_lastyearmax=state.lm_lastyearmax,
        litter=state.litter,
        litter_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        litter_not_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        carbon=state.carbon,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        natural=natural,
        pasture=pasture,
        is_peat=is_peat,
        is_tree=is_tree,
        pft_to_mtc=pft_to_mtc,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        cn_sapl=cn_sapl,
        is_grassland_manag=is_grass,
        npp_longterm_init=10.0,
        ok_pc=True,
        **strategy_kwargs,
    )
    result = stomate_daily_lcc_gross_glcchange_explicit(
        state=state,
        season=season,
        do_now_stomate_lcchange=True,
        use_age_class=False,
        agri_peat=True,
        veget_max=veget_new,
        veget_max_old=veget_old,
        natural=natural,
        pasture=pasture,
        is_peat=is_peat,
        pft_to_mtc=pft_to_mtc,
        is_tree=is_tree,
        is_grassland_manag=is_grass,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        cn_ind=cn_ind,
        npp_longterm_init=10.0,
        dt_days=30.0,
        ok_pc=True,
        **strategy_kwargs,
    )

    assert result.active is True
    assert result.done_stomate_lcchange is True
    assert np.allclose(np.asarray(result.state_after.biomass), np.asarray(expected.biomass))
    assert np.allclose(np.asarray(result.state_after.bm_to_litter), np.asarray(expected.bm_to_litter))
    assert np.allclose(np.asarray(result.state_after.litter), np.asarray(expected.litter))
    assert np.allclose(np.asarray(result.state_after.carbon), np.asarray(expected.carbon))
    assert np.allclose(np.asarray(result.state_after.deepC_a), np.asarray(expected.deepC_a))
    assert np.allclose(np.asarray(result.state_after.deepC_s), np.asarray(expected.deepC_s))
    assert np.allclose(np.asarray(result.state_after.deepC_p), np.asarray(expected.deepC_p))
    assert np.allclose(
        np.asarray(result.state_after.soilc_total),
        np.asarray(expected.deepC_a + expected.deepC_s + expected.deepC_p),
    )
    assert np.allclose(np.asarray(result.state_after.fuel_1hr), np.asarray(expected.fuel_1hr))
    assert np.allclose(np.asarray(result.state_after.prod10)[:, :, IWPLCC], np.asarray(expected.prod10))
    assert np.allclose(np.asarray(result.state_after.prod100)[:, :, IWPLCC], np.asarray(expected.prod100))
    assert np.allclose(np.asarray(result.state_after.prod10)[:, :, 1], prod10[:, :, 1])
    assert np.allclose(np.asarray(result.veget_max_after).sum(axis=1), np.ones(npts))


def test_explicit_daily_carbon_adapter_matches_direct_npp_kernel():
    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs()

    result = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        dt_days=1.0,
    )
    expected = npp_closed_update(
        biomass=biomass,
        gpp=gpp_daily,
        f_alloc=f_alloc,
        resp_maint_part=resp_maint_part,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        dt_days=1.0,
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.npp_update.biomass), np.asarray(expected.biomass))
    assert np.allclose(np.asarray(result.npp_update.bm_alloc), np.asarray(expected.bm_alloc))
    assert np.allclose(np.asarray(result.npp_update.resp_growth), np.asarray(expected.resp_growth))
    assert np.allclose(np.asarray(result.npp_update.npp), np.asarray(expected.npp))
    assert result.npp_update.biomass.shape == biomass.shape
    assert result.age_sla is None


def test_explicit_daily_carbon_adapter_requires_f_alloc():
    biomass, _, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)

    with pytest.raises(ValueError, match="f_alloc"):
        stomate_daily_carbon_explicit(
            gpp_daily=gpp_daily,
            biomass_before=biomass,
            resp_maint_part=resp_maint_part,
            f_alloc=None,
            pft_present=pft_present,
            frac_growthresp=frac_growthresp,
        )


def test_explicit_daily_carbon_adapter_can_run_npp_age_sla_bookkeeping():
    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)
    leaf_age = np.zeros((1, 14, 4), dtype=np.float64)
    leaf_age[0, PFT14, :] = [20.0, 40.0, 80.0, 120.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [0.25, 0.25, 0.25, 0.25]
    age = np.zeros((1, 14), dtype=np.float64)
    age[0, PFT14] = 2.0
    is_tree = np.zeros(14, dtype=bool)
    sla_age1 = np.ones((1, 14), dtype=np.float64) * 0.015
    sla_calc = np.ones((1, 14), dtype=np.float64) * 0.02
    sla_max = np.ones(14, dtype=np.float64) * 0.05
    sla_min = np.ones(14, dtype=np.float64) * 0.01

    result = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        dt_days=1.0,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
    )
    expected = npp_leaf_age_sla_age_update(
        result.npp_update.biomass,
        biomass,
        result.npp_update.bm_alloc,
        leaf_age,
        leaf_frac,
        age,
        pft_present,
        is_tree,
        sla_age1,
        sla_calc,
        sla_max,
        sla_min,
        dt_days=1.0,
    )

    assert result.age_sla is not None
    assert np.allclose(np.asarray(result.age_sla.leaf_frac), np.asarray(expected.leaf_frac))
    assert np.allclose(np.asarray(result.age_sla.age), np.asarray(expected.age))
    assert np.allclose(np.asarray(result.age_sla.sla_calc), np.asarray(expected.sla_calc))


def test_npp_age_sla_uses_post_maintenance_leaf_mass_for_leaf_fractions():
    """Fortran `lm_old` is saved after maintenance tissue pumping."""

    npts, nvm = 1, 14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    f_alloc = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    f_alloc[0, PFT14, ILEAF] = 1.0
    resp_maint_part = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    resp_maint_part[0, PFT14, ILEAF] = 1.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, PFT14] = True
    frac_growthresp = np.zeros(nvm, dtype=np.float64)
    gpp_daily = np.zeros((npts, nvm), dtype=np.float64)
    gpp_daily[0, PFT14] = 1.0
    leaf_age = np.zeros((npts, nvm, 4), dtype=np.float64)
    leaf_age[0, PFT14, :] = [20.0, 40.0, 80.0, 120.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [0.25, 0.25, 0.25, 0.25]
    age = np.zeros((npts, nvm), dtype=np.float64)
    is_tree = np.zeros(nvm, dtype=bool)
    sla_age1 = np.ones((npts, nvm), dtype=np.float64) * 0.015
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    sla_max = np.ones(nvm, dtype=np.float64) * 0.05
    sla_min = np.ones(nvm, dtype=np.float64) * 0.01

    result = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        dt_days=1.0,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
    )

    assert np.allclose(np.asarray(result.npp_update.biomass_before_alloc)[0, PFT14, ILEAF, ICARBON], 99.8)
    assert np.allclose(np.asarray(result.npp_update.bm_alloc)[0, PFT14, ILEAF, ICARBON], 0.2)
    assert np.allclose(np.asarray(result.npp_update.biomass)[0, PFT14, ILEAF, ICARBON], 100.0)
    assert np.allclose(np.asarray(result.age_sla.leaf_frac)[0, PFT14].sum(), 1.0)
    assert np.allclose(np.asarray(result.age_sla.leaf_frac)[0, PFT14, 0], 25.15 / 100.0)


def test_daily_carbon_with_alloc_adapter_wires_alloc_into_npp_without_f_alloc_trace():
    npts, nvm, nslm = 1, 14, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 20.0
    biomass[0, PFT14, IROOT, ICARBON] = 10.0
    biomass[0, PFT14, ISAPABOVE, ICARBON] = 5.0
    biomass[0, PFT14, ICARBRES, ICARBON] = 50.0
    resp_maint_part = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    resp_maint_part[0, PFT14, ILEAF] = 0.2
    resp_maint_part[0, PFT14, IROOT] = 0.1
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, PFT14] = True
    frac_growthresp = np.zeros(nvm, dtype=np.float64)
    frac_growthresp[PFT14] = 0.25

    leaf_age = np.zeros((npts, nvm, 4), dtype=np.float64)
    leaf_age[0, PFT14, :] = [10.0, 20.0, 40.0, 80.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [0.4, 0.3, 0.2, 0.1]
    age = np.zeros((npts, nvm), dtype=np.float64)
    lai = np.zeros((npts, nvm), dtype=np.float64)
    lai[0, PFT14] = 0.8
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0
    senescence = np.zeros((npts, nvm), dtype=bool)
    when_growthinit = np.full((npts, nvm), 40.0, dtype=np.float64)
    moiavail_week = np.full((npts, nvm), 0.8, dtype=np.float64)
    tsoil_month = np.full((npts, nslm), 293.15, dtype=np.float64)
    soilhum_month = np.full((npts, nslm), 0.8, dtype=np.float64)
    z_soil = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    sla_calc = np.full((npts, nvm), 0.02, dtype=np.float64)

    natural = np.zeros(nvm, dtype=bool)
    pasture = np.zeros(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    ok_laidev = np.zeros(nvm, dtype=bool)
    r0 = np.full(nvm, 0.35, dtype=np.float64)
    s0 = np.full(nvm, 0.35, dtype=np.float64)
    ext_coeff = np.full(nvm, 0.5, dtype=np.float64)
    lai_max = np.full(nvm, 12.0, dtype=np.float64)
    lai_max_to_happy = np.full(nvm, 0.5, dtype=np.float64)
    tau_leafinit = np.full(nvm, 10.0, dtype=np.float64)
    alloc_min = np.full(nvm, 0.2, dtype=np.float64)
    alloc_max = np.full(nvm, 0.8, dtype=np.float64)
    demi_alloc = np.full(nvm, 100.0, dtype=np.float64)
    alloc_agr_st = np.zeros(nvm, dtype=np.float64)
    alloc_agr_pn = np.zeros(nvm, dtype=np.float64)
    alloc_agr_st[PFT14] = 0.25
    alloc_agr_pn[PFT14] = 0.05
    sla_age1 = np.full((npts, nvm), 0.02, dtype=np.float64)
    sla_max = np.full(nvm, 0.04, dtype=np.float64)
    sla_min = np.full(nvm, 0.01, dtype=np.float64)

    result = stomate_daily_carbon_with_alloc_explicit(
        gpp_daily=np.full((npts, nvm), 5.0, dtype=np.float64),
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        lai=lai,
        veget_max=veget_max,
        senescence=senescence,
        when_growthinit=when_growthinit,
        moiavail_week=moiavail_week,
        tsoil_month=tsoil_month,
        soilhum_month=soilhum_month,
        age=age,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        z_soil=z_soil,
        sla_calc=sla_calc,
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        ok_LAIdev=ok_laidev,
        r0=r0,
        s0=s0,
        ext_coeff=ext_coeff,
        lai_max=lai_max,
        lai_max_to_happy=lai_max_to_happy,
        tau_leafinit=tau_leafinit,
        alloc_min=alloc_min,
        alloc_max=alloc_max,
        demi_alloc=demi_alloc,
        alloc_agr_st=alloc_agr_st,
        alloc_agr_pn=alloc_agr_pn,
        sla_age1=sla_age1,
        sla_max=sla_max,
        sla_min=sla_min,
        f_fruit=0.1,
        ecureuil=np.zeros(nvm, dtype=np.float64),
    )
    expected_alloc = allocation_step(
        lai=lai,
        veget_max=veget_max,
        senescence=senescence,
        when_growthinit=when_growthinit,
        moiavail_week=moiavail_week,
        tsoil_month=tsoil_month,
        soilhum_month=soilhum_month,
        biomass=biomass,
        age=age,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        z_soil=z_soil,
        sla_calc=sla_calc,
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        ok_LAIdev=ok_laidev,
        r0=r0,
        s0=s0,
        ext_coeff=ext_coeff,
        lai_max=lai_max,
        lai_max_to_happy=lai_max_to_happy,
        tau_leafinit=tau_leafinit,
        alloc_min=alloc_min,
        alloc_max=alloc_max,
        demi_alloc=demi_alloc,
        alloc_agr_st=alloc_agr_st,
        alloc_agr_pn=alloc_agr_pn,
        f_fruit=0.1,
        ecureuil=np.zeros(nvm, dtype=np.float64),
    )
    expected_daily = stomate_daily_carbon_explicit(
        gpp_daily=np.full((npts, nvm), 5.0, dtype=np.float64),
        biomass_before=expected_alloc.biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=expected_alloc.f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=expected_alloc.leaf_age,
        leaf_frac=expected_alloc.leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.allocation.f_alloc), np.asarray(expected_alloc.f_alloc))
    assert np.allclose(np.asarray(result.daily_carbon.npp_update.biomass), np.asarray(expected_daily.npp_update.biomass))
    assert np.allclose(np.asarray(result.daily_carbon.age_sla.leaf_frac), np.asarray(expected_daily.age_sla.leaf_frac))
    assert np.allclose(np.asarray(result.lai_after_alloc)[0, PFT14], np.asarray(expected_alloc.biomass)[0, PFT14, ILEAF, ICARBON] * 0.02)


def test_explicit_daily_carbon_adapter_requires_complete_age_sla_inputs():
    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)

    with pytest.raises(ValueError, match="leaf_frac"):
        stomate_daily_carbon_explicit(
            gpp_daily=gpp_daily,
            biomass_before=biomass,
            resp_maint_part=resp_maint_part,
            f_alloc=f_alloc,
            pft_present=pft_present,
            frac_growthresp=frac_growthresp,
            leaf_age=np.zeros((1, 14, 4), dtype=np.float64),
        )


def test_explicit_daily_carbon_adapter_does_not_hardcode_pft14_shape():
    npts, nvm, active = 1, 5, 3
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[:, active, IROOT, ICARBON] = 4.0
    f_alloc = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    f_alloc[:, active, IROOT] = 1.0
    resp_maint_part = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[:, active] = True
    frac_growthresp = np.zeros(nvm, dtype=np.float64)

    result = stomate_daily_carbon_explicit(
        gpp_daily=np.ones((npts, nvm), dtype=np.float64),
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
    )

    assert result.npp_update.biomass.shape == biomass.shape
    assert np.allclose(np.asarray(result.npp_update.npp)[0, active], 1.0)


def test_explicit_daily_carbon_turnover_adapter_matches_direct_composition():
    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)
    biomass[0, PFT14, IROOT, ICARBON] = 40.0
    leaf_age = np.zeros((1, 14, 4), dtype=np.float64)
    leaf_age[0, PFT14, :] = [60.0, 0.0, 0.0, 0.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [1.0, 0.0, 0.0, 0.0]
    age = np.zeros((1, 14), dtype=np.float64)
    age[0, PFT14] = 2.0
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True
    natural = np.ones(14, dtype=bool)
    sla_age1 = np.ones((1, 14), dtype=np.float64) * 0.015
    sla_calc = np.ones((1, 14), dtype=np.float64) * 0.02
    sla_max = np.ones(14, dtype=np.float64) * 0.05
    sla_min = np.ones(14, dtype=np.float64) * 0.01
    turnover_kwargs = {
        "herbivores": np.zeros((1, 14), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((1, 14), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((1, 14), dtype=np.float64),
        "moiavail_week": np.ones((1, 14), dtype=np.float64) * 0.1,
        "t2m_longterm": np.full(1, 293.15, dtype=np.float64),
        "t2m_month": np.full(1, 293.15, dtype=np.float64),
        "t2m_week": np.full(1, 293.15, dtype=np.float64),
        "veget_max": pft_present.astype(np.float64),
        "gdd_from_growthinit": np.zeros((1, 14), dtype=np.float64),
        "lai": np.ones((1, 14), dtype=np.float64) * 2.0,
        "turnover_time": np.zeros((1, 14), dtype=np.float64),
        "nrec": np.zeros((1, 14), dtype=np.int32),
        "senescence_type": np.zeros(14, dtype=np.int32),
        "natural": natural,
        "is_grassland_manag": np.zeros(14, dtype=bool),
        "ok_laidev": np.zeros(14, dtype=bool),
        "min_leaf_age_for_senescence": np.ones(14, dtype=np.float64) * 30.0,
        "gdd_senescence": np.ones(14, dtype=np.float64) * 100.0,
        "senescence_temp": np.zeros((14, 3), dtype=np.float64),
        "hum_frac": np.ones(14, dtype=np.float64) * 0.5,
        "senescence_hum": np.ones(14, dtype=np.float64) * 0.2,
        "nosenescence_hum": np.ones(14, dtype=np.float64) * 0.8,
        "max_turnover_time": np.ones(14, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(14, dtype=np.float64) * 10.0,
        "leaffall": np.ones(14, dtype=np.float64) * 10.0,
        "lai_max": np.ones(14, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(14, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(14, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(14, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(14, dtype=np.float64) * 730.0,
    }
    turnover_kwargs["senescence_type"][PFT14] = SENESCENCE_DRY

    result = stomate_daily_carbon_turnover_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
        **turnover_kwargs,
    )
    expected_daily = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
    )
    expected_turnover = turnover_step(
        pft_present=pft_present,
        leaf_age=expected_daily.age_sla.leaf_age,
        leaf_frac=expected_daily.age_sla.leaf_frac,
        age=expected_daily.age_sla.age,
        biomass=expected_daily.npp_update.biomass,
        sla_calc=expected_daily.age_sla.sla_calc,
        is_tree=is_tree,
        **turnover_kwargs,
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.turnover.turnover), np.asarray(expected_turnover.turnover))
    assert np.allclose(np.asarray(result.turnover.biomass), np.asarray(expected_turnover.biomass))
    assert np.allclose(np.asarray(result.turnover.leaf_frac), np.asarray(expected_turnover.leaf_frac))
    assert bool(np.asarray(result.turnover.senescence)[0, PFT14])


def test_explicit_daily_carbon_turnover_adapter_requires_complete_sla_bookkeeping_inputs():
    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)

    with pytest.raises(ValueError, match="sla_age1"):
        stomate_daily_carbon_turnover_explicit(
            gpp_daily=gpp_daily,
            biomass_before=biomass,
            resp_maint_part=resp_maint_part,
            f_alloc=f_alloc,
            pft_present=pft_present,
            frac_growthresp=frac_growthresp,
            leaf_age=np.zeros((1, 14, 4), dtype=np.float64),
            leaf_frac=np.zeros((1, 14, 4), dtype=np.float64),
            age=np.zeros((1, 14), dtype=np.float64),
            is_tree=np.zeros(14, dtype=bool),
            sla_calc=np.ones((1, 14), dtype=np.float64) * 0.02,
            herbivores=np.zeros((1, 14), dtype=np.float64),
            maxmoiavail_lastyear=np.ones((1, 14), dtype=np.float64),
            minmoiavail_lastyear=np.zeros((1, 14), dtype=np.float64),
            moiavail_week=np.ones((1, 14), dtype=np.float64),
            t2m_longterm=np.full(1, 293.15, dtype=np.float64),
            t2m_month=np.full(1, 293.15, dtype=np.float64),
            t2m_week=np.full(1, 293.15, dtype=np.float64),
            veget_max=pft_present.astype(np.float64),
            gdd_from_growthinit=np.zeros((1, 14), dtype=np.float64),
            lai=np.zeros((1, 14), dtype=np.float64),
            turnover_time=np.zeros((1, 14), dtype=np.float64),
            nrec=np.zeros((1, 14), dtype=np.int32),
            senescence_type=np.zeros(14, dtype=np.int32),
            natural=np.ones(14, dtype=bool),
            is_grassland_manag=np.zeros(14, dtype=bool),
            ok_laidev=np.zeros(14, dtype=bool),
            min_leaf_age_for_senescence=np.ones(14, dtype=np.float64),
            gdd_senescence=np.ones(14, dtype=np.float64),
            senescence_temp=np.zeros((14, 3), dtype=np.float64),
            hum_frac=np.ones(14, dtype=np.float64),
            senescence_hum=np.ones(14, dtype=np.float64),
            nosenescence_hum=np.ones(14, dtype=np.float64),
            max_turnover_time=np.ones(14, dtype=np.float64),
            min_turnover_time=np.ones(14, dtype=np.float64),
            leaffall=np.ones(14, dtype=np.float64),
            lai_max=np.ones(14, dtype=np.float64),
            leafagecrit=np.ones(14, dtype=np.float64),
            lai_initmin=np.ones(14, dtype=np.float64),
            tau_fruit=np.ones(14, dtype=np.float64),
            tau_sap=np.ones(14, dtype=np.float64),
        )


def test_explicit_daily_carbon_gap_turnover_adapter_matches_direct_fortran_order_composition():
    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)
    biomass[0, PFT14, IROOT, ICARBON] = 40.0
    leaf_age = np.zeros((1, 14, 4), dtype=np.float64)
    leaf_age[0, PFT14, :] = [60.0, 0.0, 0.0, 0.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [1.0, 0.0, 0.0, 0.0]
    age = np.zeros((1, 14), dtype=np.float64)
    age[0, PFT14] = 2.0
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True
    natural = np.ones(14, dtype=bool)
    sla_age1 = np.ones((1, 14), dtype=np.float64) * 0.015
    sla_calc = np.ones((1, 14), dtype=np.float64) * 0.02
    sla_max = np.ones(14, dtype=np.float64) * 0.05
    sla_min = np.ones(14, dtype=np.float64) * 0.01
    common = {
        "herbivores": np.zeros((1, 14), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((1, 14), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((1, 14), dtype=np.float64),
        "moiavail_week": np.ones((1, 14), dtype=np.float64) * 0.1,
        "t2m_longterm": np.full(1, 293.15, dtype=np.float64),
        "t2m_month": np.full(1, 293.15, dtype=np.float64),
        "t2m_week": np.full(1, 293.15, dtype=np.float64),
        "veget_max": pft_present.astype(np.float64),
        "gdd_from_growthinit": np.zeros((1, 14), dtype=np.float64),
        "lai": np.ones((1, 14), dtype=np.float64) * 2.0,
        "turnover_time": np.zeros((1, 14), dtype=np.float64),
        "nrec": np.zeros((1, 14), dtype=np.int32),
        "senescence_type": np.zeros(14, dtype=np.int32),
        "natural": natural,
        "is_grassland_manag": np.zeros(14, dtype=bool),
        "ok_laidev": np.zeros(14, dtype=bool),
        "min_leaf_age_for_senescence": np.ones(14, dtype=np.float64) * 30.0,
        "gdd_senescence": np.ones(14, dtype=np.float64) * 100.0,
        "senescence_temp": np.zeros((14, 3), dtype=np.float64),
        "hum_frac": np.ones(14, dtype=np.float64) * 0.5,
        "senescence_hum": np.ones(14, dtype=np.float64) * 0.2,
        "nosenescence_hum": np.ones(14, dtype=np.float64) * 0.8,
        "max_turnover_time": np.ones(14, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(14, dtype=np.float64) * 10.0,
        "leaffall": np.ones(14, dtype=np.float64) * 10.0,
        "lai_max": np.ones(14, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(14, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(14, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(14, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(14, dtype=np.float64) * 730.0,
    }
    common["senescence_type"][PFT14] = SENESCENCE_DRY
    gap_inputs = {
        "npp_longterm": np.ones((1, 14), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        "lm_lastyearmax": np.ones((1, 14), dtype=np.float64) * 50.0,
        "ind": np.ones((1, 14), dtype=np.float64),
        "bm_to_litter": np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        "t2m_min_daily": np.full(1, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((1, 14), dtype=np.float64),
        "pasture": np.zeros(14, dtype=bool),
        "availability_fact": np.ones(14, dtype=np.float64) * 0.14,
        "residence_time": np.ones(14, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(14, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(14, dtype=np.int32),
        "pheno_type": np.zeros(14, dtype=np.int32),
    }

    result = stomate_daily_carbon_gap_turnover_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
        **gap_inputs,
        **common,
    )
    expected_daily = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
    )
    expected_gap = gap_mortality_step(
        pft_present=pft_present,
        biomass=expected_daily.npp_update.biomass,
        sla_calc=expected_daily.age_sla.sla_calc,
        natural=natural,
        is_tree=is_tree,
        **gap_inputs,
    )
    expected_turnover = turnover_step(
        pft_present=pft_present,
        leaf_age=expected_daily.age_sla.leaf_age,
        leaf_frac=expected_daily.age_sla.leaf_frac,
        age=expected_daily.age_sla.age,
        biomass=expected_gap.biomass,
        sla_calc=expected_daily.age_sla.sla_calc,
        is_tree=is_tree,
        **common,
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.gap.bm_to_litter), np.asarray(expected_gap.bm_to_litter))
    assert np.allclose(np.asarray(result.turnover.turnover), np.asarray(expected_turnover.turnover))
    assert np.allclose(np.asarray(result.turnover.biomass), np.asarray(expected_turnover.biomass))
    assert np.allclose(np.asarray(result.bm_to_litter), np.asarray(expected_gap.bm_to_litter))


def test_explicit_daily_carbon_kill_gap_turnover_adapter_matches_direct_order_composition():
    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)
    biomass[0, PFT14, IROOT, ICARBON] = 40.0
    biomass[0, PFT14, ISAPABOVE, ICARBON] = 30.0
    leaf_age = np.zeros((1, 14, 4), dtype=np.float64)
    leaf_age[0, PFT14, :] = [60.0, 0.0, 0.0, 0.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [1.0, 0.0, 0.0, 0.0]
    age = np.zeros((1, 14), dtype=np.float64)
    age[0, PFT14] = 2.0
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True
    natural = np.ones(14, dtype=bool)
    sla_age1 = np.ones((1, 14), dtype=np.float64) * 0.015
    sla_calc = np.ones((1, 14), dtype=np.float64) * 0.02
    sla_max = np.ones(14, dtype=np.float64) * 0.05
    sla_min = np.ones(14, dtype=np.float64) * 0.01
    common = {
        "herbivores": np.zeros((1, 14), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((1, 14), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((1, 14), dtype=np.float64),
        "moiavail_week": np.ones((1, 14), dtype=np.float64) * 0.1,
        "t2m_longterm": np.full(1, 293.15, dtype=np.float64),
        "t2m_month": np.full(1, 293.15, dtype=np.float64),
        "t2m_week": np.full(1, 293.15, dtype=np.float64),
        "gdd_from_growthinit": np.zeros((1, 14), dtype=np.float64),
        "lai": np.ones((1, 14), dtype=np.float64) * 2.0,
        "turnover_time": np.zeros((1, 14), dtype=np.float64),
        "nrec": np.zeros((1, 14), dtype=np.int32),
        "senescence_type": np.zeros(14, dtype=np.int32),
        "is_grassland_manag": np.zeros(14, dtype=bool),
        "ok_laidev": np.zeros(14, dtype=bool),
        "min_leaf_age_for_senescence": np.ones(14, dtype=np.float64) * 30.0,
        "gdd_senescence": np.ones(14, dtype=np.float64) * 100.0,
        "senescence_temp": np.zeros((14, 3), dtype=np.float64),
        "hum_frac": np.ones(14, dtype=np.float64) * 0.5,
        "senescence_hum": np.ones(14, dtype=np.float64) * 0.2,
        "nosenescence_hum": np.ones(14, dtype=np.float64) * 0.8,
        "max_turnover_time": np.ones(14, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(14, dtype=np.float64) * 10.0,
        "leaffall": np.ones(14, dtype=np.float64) * 10.0,
        "lai_max": np.ones(14, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(14, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(14, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(14, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(14, dtype=np.float64) * 730.0,
    }
    common["senescence_type"][PFT14] = SENESCENCE_DRY
    state = {
        "lm_lastyearmax": np.ones((1, 14), dtype=np.float64) * 50.0,
        "ind": np.ones((1, 14), dtype=np.float64),
        "cn_ind": np.ones((1, 14), dtype=np.float64) * 2.0,
        "bm_to_litter": np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((1, 14), dtype=bool),
        "rip_time": np.ones((1, 14), dtype=np.float64),
        "when_growthinit": np.ones((1, 14), dtype=np.float64) * 6.0,
        "everywhere": np.ones((1, 14), dtype=np.float64),
        "veget_max": pft_present.astype(np.float64),
        "height": np.ones((1, 14), dtype=np.float64),
        "pasture": np.zeros(14, dtype=bool),
    }
    gap_inputs = {
        "npp_longterm": np.ones((1, 14), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        "t2m_min_daily": np.full(1, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((1, 14), dtype=np.float64),
        "availability_fact": np.ones(14, dtype=np.float64) * 0.14,
        "residence_time": np.ones(14, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(14, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(14, dtype=np.int32),
        "pheno_type": np.zeros(14, dtype=np.int32),
        "maxdia": np.ones(14, dtype=np.float64) * 10.0,
    }

    result = stomate_daily_carbon_kill_gap_turnover_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        natural=natural,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
        lpj_gap_const_mort=True,
        **state,
        **gap_inputs,
        **common,
    )
    expected_daily = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
    )
    # Fortran stomate_lpj.f90 lines 1137-1176 call kill/crown only when
    # ok_dgvm .OR. .NOT. lpj_gap_const_mort; the paper run skips both.
    expected_gap = gap_mortality_step(
        npp_longterm=gap_inputs["npp_longterm"],
        turnover_longterm=gap_inputs["turnover_longterm"],
        lm_lastyearmax=state["lm_lastyearmax"],
        pft_present=pft_present,
        biomass=expected_daily.npp_update.biomass,
        ind=state["ind"],
        bm_to_litter=state["bm_to_litter"],
        t2m_min_daily=gap_inputs["t2m_min_daily"],
        tmin_spring_time=gap_inputs["tmin_spring_time"],
        sla_calc=expected_daily.age_sla.sla_calc,
        natural=natural,
        is_tree=is_tree,
        pasture=state["pasture"],
        availability_fact=gap_inputs["availability_fact"],
        residence_time=gap_inputs["residence_time"],
        tmin_crit=gap_inputs["tmin_crit"],
        leaf_tab=gap_inputs["leaf_tab"],
        pheno_type=gap_inputs["pheno_type"],
        lpj_gap_const_mort=True,
    )
    expected_turnover = turnover_step(
        pft_present=pft_present,
        veget_max=state["veget_max"],
        leaf_age=expected_daily.age_sla.leaf_age,
        leaf_frac=expected_daily.age_sla.leaf_frac,
        age=expected_daily.age_sla.age,
        biomass=expected_gap.biomass,
        sla_calc=expected_daily.age_sla.sla_calc,
        is_tree=is_tree,
        natural=natural,
        **common,
    )

    assert result.requires_trace == ()
    assert not np.any(np.asarray(result.kill_after_npp.was_killed))
    assert np.allclose(np.asarray(result.kill_after_npp.biomass), np.asarray(expected_daily.npp_update.biomass))
    assert np.allclose(np.asarray(result.kill_after_npp.bm_to_litter), np.asarray(state["bm_to_litter"]))
    assert np.allclose(np.asarray(result.crown_after_npp.cn_ind), np.asarray(state["cn_ind"]))
    assert np.allclose(np.asarray(result.crown_after_npp.veget_max), np.asarray(state["veget_max"]))
    assert np.allclose(np.asarray(result.gap.bm_to_litter), np.asarray(expected_gap.bm_to_litter))
    assert not np.any(np.asarray(result.kill_after_gap.was_killed))
    assert np.allclose(np.asarray(result.kill_after_gap.bm_to_litter), np.asarray(expected_gap.bm_to_litter))
    assert np.allclose(np.asarray(result.turnover.turnover), np.asarray(expected_turnover.turnover))
    assert result.light is None
    assert result.kill_after_light is None
    assert result.establishment_rates is None
    assert result.establishment_biomass is None
    assert result.crown_after_establish is None
    # Fortran stomate_lpj.f90 lines 1552-1553 runs setlai before vmax
    # unconditionally; this must not depend on wiring the vmax process itself.
    np.testing.assert_allclose(
        np.asarray(result.lai_after_setlai),
        np.asarray(result.turnover.biomass[:, :, ILEAF, ICARBON]) * np.asarray(result.daily_carbon.age_sla.sla_calc),
    )
    assert result.vmax is None
    assert np.allclose(np.asarray(result.bm_to_litter), np.asarray(expected_gap.bm_to_litter))


def test_daily_alloc_kill_gap_turnover_wrapper_removes_external_f_alloc_boundary():
    npts, nvm, nslm = 1, 14, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    biomass[0, PFT14, IROOT, ICARBON] = 40.0
    biomass[0, PFT14, ISAPABOVE, ICARBON] = 30.0
    biomass[0, PFT14, ICARBRES, ICARBON] = 80.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, PFT14] = True
    leaf_age = np.zeros((npts, nvm, 4), dtype=np.float64)
    leaf_age[0, PFT14, :] = [60.0, 0.0, 0.0, 0.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [1.0, 0.0, 0.0, 0.0]
    age = np.zeros((npts, nvm), dtype=np.float64)
    age[0, PFT14] = 2.0
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.ones(nvm, dtype=bool)
    pasture = np.zeros(nvm, dtype=bool)
    ok_laidev = np.zeros(nvm, dtype=bool)
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    lai = biomass[:, :, ILEAF, ICARBON] * sla_calc
    veget_max = pft_present.astype(np.float64)

    alloc_inputs = {
        "lai": lai,
        "veget_max": veget_max,
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64) * 40.0,
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64) * 0.8,
        "tsoil_month": np.ones((npts, nslm), dtype=np.float64) * 293.15,
        "soilhum_month": np.ones((npts, nslm), dtype=np.float64) * 0.8,
        "biomass": biomass,
        "age": age,
        "leaf_age": leaf_age,
        "leaf_frac": leaf_frac,
        "z_soil": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        "sla_calc": sla_calc,
        "natural": natural,
        "pasture": pasture,
        "is_tree": is_tree,
        "ok_LAIdev": ok_laidev,
        "r0": np.ones(nvm, dtype=np.float64) * 0.35,
        "s0": np.ones(nvm, dtype=np.float64) * 0.35,
        "ext_coeff": np.ones(nvm, dtype=np.float64) * 0.5,
        "lai_max": np.ones(nvm, dtype=np.float64) * 12.0,
        "lai_max_to_happy": np.ones(nvm, dtype=np.float64) * 0.5,
        "tau_leafinit": np.ones(nvm, dtype=np.float64) * 10.0,
        "alloc_min": np.ones(nvm, dtype=np.float64) * 0.2,
        "alloc_max": np.ones(nvm, dtype=np.float64) * 0.8,
        "demi_alloc": np.ones(nvm, dtype=np.float64) * 100.0,
        "alloc_agr_st": np.zeros(nvm, dtype=np.float64),
        "alloc_agr_pn": np.zeros(nvm, dtype=np.float64),
    }
    alloc_inputs["alloc_agr_st"][PFT14] = 0.25
    alloc_inputs["alloc_agr_pn"][PFT14] = 0.05

    common = {
        "gpp_daily": np.ones((npts, nvm), dtype=np.float64) * 10.0,
        "resp_maint_part": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "pft_present": pft_present,
        "frac_growthresp": np.zeros(nvm, dtype=np.float64),
        "age": age,
        "sla_calc": sla_calc,
        "sla_age1": np.ones((npts, nvm), dtype=np.float64) * 0.015,
        "sla_max": np.ones(nvm, dtype=np.float64) * 0.05,
        "sla_min": np.ones(nvm, dtype=np.float64) * 0.01,
        "is_tree": is_tree,
        "natural": natural,
        "pasture": pasture,
        "herbivores": np.zeros((npts, nvm), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((npts, nvm), dtype=np.float64),
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64) * 0.1,
        "t2m_longterm": np.full(npts, 293.15, dtype=np.float64),
        "t2m_month": np.full(npts, 293.15, dtype=np.float64),
        "t2m_week": np.full(npts, 293.15, dtype=np.float64),
        "veget_max": veget_max,
        "gdd_from_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "turnover_time": np.zeros((npts, nvm), dtype=np.float64),
        "nrec": np.zeros((npts, nvm), dtype=np.int32),
        "senescence_type": np.zeros(nvm, dtype=np.int32),
        "is_grassland_manag": np.zeros(nvm, dtype=bool),
        "ok_laidev": ok_laidev,
        "min_leaf_age_for_senescence": np.ones(nvm, dtype=np.float64) * 30.0,
        "gdd_senescence": np.ones(nvm, dtype=np.float64) * 100.0,
        "senescence_temp": np.zeros((nvm, 3), dtype=np.float64),
        "hum_frac": np.ones(nvm, dtype=np.float64) * 0.5,
        "senescence_hum": np.ones(nvm, dtype=np.float64) * 0.2,
        "nosenescence_hum": np.ones(nvm, dtype=np.float64) * 0.8,
        "max_turnover_time": np.ones(nvm, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(nvm, dtype=np.float64) * 10.0,
        "leaffall": np.ones(nvm, dtype=np.float64) * 10.0,
        "lai_max": np.ones(nvm, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(nvm, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(nvm, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(nvm, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(nvm, dtype=np.float64) * 730.0,
    }
    state = {
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "lm_lastyearmax": np.ones((npts, nvm), dtype=np.float64) * 50.0,
        "ind": np.ones((npts, nvm), dtype=np.float64),
        "cn_ind": np.ones((npts, nvm), dtype=np.float64) * 2.0,
        "bm_to_litter": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64),
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64) * 6.0,
        "everywhere": np.ones((npts, nvm), dtype=np.float64),
        "height": np.ones((npts, nvm), dtype=np.float64),
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((npts, nvm), dtype=np.float64),
        "availability_fact": np.ones(nvm, dtype=np.float64) * 0.14,
        "residence_time": np.ones(nvm, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(nvm, dtype=np.int32),
        "pheno_type": np.zeros(nvm, dtype=np.int32),
        "maxdia": np.ones(nvm, dtype=np.float64) * 10.0,
        "lpj_gap_const_mort": True,
    }
    manual_alloc = allocation_step(**alloc_inputs, f_fruit=0.1, ecureuil=np.zeros(nvm, dtype=np.float64))
    manual = stomate_daily_carbon_kill_gap_turnover_explicit(
        **common,
        **state,
        biomass_before=manual_alloc.biomass,
        f_alloc=manual_alloc.f_alloc,
        leaf_age=manual_alloc.leaf_age,
        leaf_frac=manual_alloc.leaf_frac,
        lai=np.asarray(manual_alloc.biomass)[:, :, ILEAF, ICARBON] * sla_calc,
    )

    result = stomate_daily_alloc_kill_gap_turnover_explicit(
        alloc_inputs=alloc_inputs,
        allocation_kwargs={"f_fruit": 0.1, "ecureuil": np.zeros(nvm, dtype=np.float64)},
        post_npp_inputs={**common, **state},
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.allocation.f_alloc), np.asarray(manual_alloc.f_alloc))
    assert np.allclose(np.asarray(result.post_npp.daily_carbon.npp_update.biomass), np.asarray(manual.daily_carbon.npp_update.biomass))
    assert np.allclose(np.asarray(result.post_npp.turnover.biomass), np.asarray(manual.turnover.biomass))
    assert np.allclose(np.asarray(result.post_npp.bm_to_litter), np.asarray(manual.bm_to_litter))
    with pytest.raises(ValueError, match="f_alloc"):
        stomate_daily_alloc_kill_gap_turnover_explicit(
            alloc_inputs=alloc_inputs,
            post_npp_inputs={**common, **state, "f_alloc": manual_alloc.f_alloc},
        )


def test_daily_prescribe_alloc_kill_gap_turnover_wrapper_wires_static_cold_start_state():
    npts, nvm, nslm = 1, 14, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, 4), dtype=np.float64)
    leaf_age = np.zeros((npts, nvm, 4), dtype=np.float64)
    age = np.zeros((npts, nvm), dtype=np.float64)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.ones(nvm, dtype=bool)
    pasture = np.zeros(nvm, dtype=bool)
    ok_laidev = np.zeros(nvm, dtype=bool)
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[PFT14, ICARBRES, ICARBON] = 5.0

    prescribe_inputs = {
        "veget_max": veget_max,
        "dt_days": 1.0,
        "pft_present": np.zeros((npts, nvm), dtype=bool),
        "everywhere": np.zeros((npts, nvm), dtype=np.float64),
        "when_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "biomass": biomass,
        "leaf_frac": leaf_frac,
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "cn_ind": np.zeros((npts, nvm), dtype=np.float64),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "natural": natural,
        "pasture": pasture,
        "is_tree": is_tree,
        "bm_sapl": bm_sapl,
        "maxdia": np.ones(nvm, dtype=np.float64),
        "pheno_is_none": np.ones(nvm, dtype=bool),
        "ok_dgvm": False,
        "lpj_gap_const_mort": True,
        "firstcall": True,
        "stomate_restart_none": True,
    }
    alloc_inputs = {
        "lai": np.zeros((npts, nvm), dtype=np.float64),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64) * 40.0,
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64) * 0.8,
        "tsoil_month": np.ones((npts, nslm), dtype=np.float64) * 293.15,
        "soilhum_month": np.ones((npts, nslm), dtype=np.float64) * 0.8,
        "age": age,
        "leaf_age": leaf_age,
        "z_soil": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        "sla_calc": sla_calc,
        "natural": natural,
        "pasture": pasture,
        "is_tree": is_tree,
        "ok_LAIdev": ok_laidev,
        "r0": np.ones(nvm, dtype=np.float64) * 0.35,
        "s0": np.ones(nvm, dtype=np.float64) * 0.35,
        "ext_coeff": np.ones(nvm, dtype=np.float64) * 0.5,
        "lai_max": np.ones(nvm, dtype=np.float64) * 12.0,
        "lai_max_to_happy": np.ones(nvm, dtype=np.float64) * 0.5,
        "tau_leafinit": np.ones(nvm, dtype=np.float64) * 10.0,
        "alloc_min": np.ones(nvm, dtype=np.float64) * 0.2,
        "alloc_max": np.ones(nvm, dtype=np.float64) * 0.8,
        "demi_alloc": np.ones(nvm, dtype=np.float64) * 100.0,
        "alloc_agr_st": np.zeros(nvm, dtype=np.float64),
        "alloc_agr_pn": np.zeros(nvm, dtype=np.float64),
    }
    alloc_inputs["alloc_agr_st"][PFT14] = 0.25
    alloc_inputs["alloc_agr_pn"][PFT14] = 0.05
    post_inputs = {
        "gpp_daily": np.ones((npts, nvm), dtype=np.float64) * 10.0,
        "resp_maint_part": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "frac_growthresp": np.zeros(nvm, dtype=np.float64),
        "age": age,
        "sla_calc": sla_calc,
        "sla_age1": np.ones((npts, nvm), dtype=np.float64) * 0.015,
        "sla_max": np.ones(nvm, dtype=np.float64) * 0.05,
        "sla_min": np.ones(nvm, dtype=np.float64) * 0.01,
        "is_tree": is_tree,
        "natural": natural,
        "pasture": pasture,
        "herbivores": np.zeros((npts, nvm), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((npts, nvm), dtype=np.float64),
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64) * 0.1,
        "t2m_longterm": np.full(npts, 293.15, dtype=np.float64),
        "t2m_month": np.full(npts, 293.15, dtype=np.float64),
        "t2m_week": np.full(npts, 293.15, dtype=np.float64),
        "veget_max": veget_max,
        "gdd_from_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "turnover_time": np.zeros((npts, nvm), dtype=np.float64),
        "nrec": np.zeros((npts, nvm), dtype=np.int32),
        "senescence_type": np.zeros(nvm, dtype=np.int32),
        "is_grassland_manag": np.zeros(nvm, dtype=bool),
        "ok_laidev": ok_laidev,
        "min_leaf_age_for_senescence": np.ones(nvm, dtype=np.float64) * 30.0,
        "gdd_senescence": np.ones(nvm, dtype=np.float64) * 100.0,
        "senescence_temp": np.zeros((nvm, 3), dtype=np.float64),
        "hum_frac": np.ones(nvm, dtype=np.float64) * 0.5,
        "senescence_hum": np.ones(nvm, dtype=np.float64) * 0.2,
        "nosenescence_hum": np.ones(nvm, dtype=np.float64) * 0.8,
        "max_turnover_time": np.ones(nvm, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(nvm, dtype=np.float64) * 10.0,
        "leaffall": np.ones(nvm, dtype=np.float64) * 10.0,
        "lai_max": np.ones(nvm, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(nvm, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(nvm, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(nvm, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(nvm, dtype=np.float64) * 730.0,
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "lm_lastyearmax": np.ones((npts, nvm), dtype=np.float64) * 50.0,
        "bm_to_litter": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64),
        "height": np.ones((npts, nvm), dtype=np.float64),
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((npts, nvm), dtype=np.float64),
        "availability_fact": np.ones(nvm, dtype=np.float64) * 0.14,
        "residence_time": np.ones(nvm, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(nvm, dtype=np.int32),
        "pheno_type": np.zeros(nvm, dtype=np.int32),
        "maxdia": np.ones(nvm, dtype=np.float64) * 10.0,
        "lpj_gap_const_mort": True,
    }

    manual_prescribe = prescribe_step(**prescribe_inputs)
    manual_alloc = allocation_step(
        **alloc_inputs,
        veget_max=veget_max,
        biomass=manual_prescribe.biomass,
        leaf_frac=manual_prescribe.leaf_frac,
        f_fruit=0.1,
        ecureuil=np.zeros(nvm, dtype=np.float64),
    )
    manual = stomate_daily_carbon_kill_gap_turnover_explicit(
        **post_inputs,
        biomass_before=manual_alloc.biomass,
        f_alloc=manual_alloc.f_alloc,
        pft_present=manual_prescribe.pft_present,
        leaf_age=manual_alloc.leaf_age,
        leaf_frac=manual_alloc.leaf_frac,
        ind=manual_prescribe.ind,
        cn_ind=manual_prescribe.cn_ind,
        when_growthinit=manual_prescribe.when_growthinit,
        everywhere=manual_prescribe.everywhere,
        lai=np.asarray(manual_alloc.biomass)[:, :, ILEAF, ICARBON] * sla_calc,
    )

    result = stomate_daily_prescribe_alloc_kill_gap_turnover_explicit(
        prescribe_inputs=prescribe_inputs,
        alloc_inputs=alloc_inputs,
        allocation_kwargs={"f_fruit": 0.1, "ecureuil": np.zeros(nvm, dtype=np.float64)},
        post_npp_inputs=post_inputs,
    )

    assert result.requires_trace == ()
    assert bool(np.asarray(result.prescribe.pft_present)[0, PFT14])
    assert np.allclose(np.asarray(result.prescribe.biomass)[0, PFT14, ICARBRES, ICARBON], 5.0)
    assert np.allclose(np.asarray(result.allocation.f_alloc), np.asarray(manual_alloc.f_alloc))
    assert np.allclose(np.asarray(result.post_npp.turnover.biomass), np.asarray(manual.turnover.biomass))

    constraints_inputs = {
        "t2m_month": np.full(npts, 293.15, dtype=np.float64),
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "adapted": np.zeros((npts, nvm), dtype=np.float64),
        "regenerate": np.zeros((npts, nvm), dtype=np.float64),
        "tseason": np.full(npts, 293.15, dtype=np.float64),
        "natural": natural,
        "is_tree": is_tree,
        "is_peat": np.zeros(nvm, dtype=bool),
        "pheno_is_none": np.ones(nvm, dtype=bool),
        "tmin_crit": np.full(nvm, -9999.0, dtype=np.float64),
        "tcm_crit": np.full(nvm, np.inf, dtype=np.float64),
    }
    manual_constraints = constraints_step(
        **constraints_inputs,
        when_growthinit=manual_prescribe.when_growthinit,
    )
    phenology_inputs = {
        "leaf_age": leaf_age,
        "pheno_model": ["none"] * nvm,
    }
    manual_phenology = phenology_step(
        **phenology_inputs,
        pft_present=manual_prescribe.pft_present,
        when_growthinit=manual_prescribe.when_growthinit,
        biomass=manual_prescribe.biomass,
        leaf_frac=manual_prescribe.leaf_frac,
        co2_to_bm=manual_prescribe.co2_to_bm,
        dt_days=1.0,
    )
    alloc_inputs_after_constraints = {
        name: value for name, value in alloc_inputs.items() if name != "when_growthinit"
    }
    manual_alloc_after_constraints = allocation_step(
        **alloc_inputs_after_constraints,
        veget_max=veget_max,
        when_growthinit=manual_phenology.when_growthinit,
        biomass=manual_phenology.biomass,
        leaf_frac=manual_phenology.leaf_frac,
        f_fruit=0.1,
        ecureuil=np.zeros(nvm, dtype=np.float64),
    )
    manual_after_constraints = stomate_daily_carbon_kill_gap_turnover_explicit(
        **post_inputs,
        biomass_before=manual_alloc_after_constraints.biomass,
        f_alloc=manual_alloc_after_constraints.f_alloc,
        pft_present=manual_prescribe.pft_present,
        leaf_age=manual_alloc_after_constraints.leaf_age,
        leaf_frac=manual_alloc_after_constraints.leaf_frac,
        ind=manual_prescribe.ind,
        cn_ind=manual_prescribe.cn_ind,
        when_growthinit=manual_phenology.when_growthinit,
        everywhere=manual_prescribe.everywhere,
        lai=np.asarray(manual_alloc_after_constraints.biomass)[:, :, ILEAF, ICARBON] * sla_calc,
        regenerate=manual_constraints.regenerate,
    )
    with_constraints = stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        prescribe_inputs=prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        phenology_kwargs={"dt_days": 1.0},
        alloc_inputs=alloc_inputs_after_constraints,
        allocation_kwargs={"f_fruit": 0.1, "ecureuil": np.zeros(nvm, dtype=np.float64)},
        post_npp_inputs=post_inputs,
    )

    assert with_constraints.requires_trace == ()
    assert with_constraints.phenology is not None
    assert np.allclose(np.asarray(with_constraints.constraints.regenerate), np.asarray(manual_constraints.regenerate))
    assert np.allclose(np.asarray(with_constraints.phenology.when_growthinit), np.asarray(manual_phenology.when_growthinit))
    assert np.allclose(
        np.asarray(with_constraints.allocation.f_alloc),
        np.asarray(manual_alloc_after_constraints.f_alloc),
    )
    assert np.allclose(
        np.asarray(with_constraints.post_npp.turnover.biomass),
        np.asarray(manual_after_constraints.turnover.biomass),
    )
    with pytest.raises(ValueError, match="regenerate"):
        stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit(
            prescribe_inputs=prescribe_inputs,
            constraints_inputs=constraints_inputs,
            alloc_inputs=alloc_inputs_after_constraints,
            post_npp_inputs={**post_inputs, "regenerate": manual_constraints.regenerate},
        )

    post_inputs_without_gpp = {name: value for name, value in post_inputs.items() if name != "gpp_daily"}
    post_inputs_without_gpp_or_resp = {
        name: value for name, value in post_inputs.items() if name not in {"gpp_daily", "resp_maint_part"}
    }
    gpp_instant = np.zeros((npts, nvm), dtype=np.float64)
    gpp_instant[0, PFT14] = 10.0
    gpp_d = stomate_gpp_daily_increment(
        gpp_instant,
        veget_max,
        np.zeros(npts, dtype=np.float64),
        dt_sechiba=86400.0,
    )
    manual_gpp_daily = stomate_accumulate_daily(
        np.zeros((npts, nvm), dtype=np.float64),
        gpp_d,
        True,
        dt_sechiba=86400.0,
        dt_stomate=86400.0,
    )
    manual_scheduled_chain = stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        prescribe_inputs=prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        phenology_kwargs={"dt_days": 1.0},
        alloc_inputs=alloc_inputs_after_constraints,
        allocation_kwargs={"f_fruit": 0.1, "ecureuil": np.zeros(nvm, dtype=np.float64)},
        post_npp_inputs={**post_inputs_without_gpp, "gpp_daily": manual_gpp_daily},
    )
    scheduled = stomate_daily_scheduled_gpp_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        gpp=gpp_instant,
        gpp_daily_current=np.zeros((npts, nvm), dtype=np.float64),
        veget_max=veget_max,
        totfrac_nobio=np.zeros(npts, dtype=np.float64),
        dt_sechiba=86400.0,
        dt_stomate=86400.0,
        do_slow=True,
        prescribe_inputs=prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        phenology_kwargs={"dt_days": 1.0},
        alloc_inputs=alloc_inputs_after_constraints,
        allocation_kwargs={"f_fruit": 0.1, "ecureuil": np.zeros(nvm, dtype=np.float64)},
        post_npp_inputs=post_inputs_without_gpp,
    )

    assert scheduled.requires_trace == ()
    assert np.allclose(np.asarray(scheduled.gpp_d), np.asarray(gpp_d))
    assert np.allclose(np.asarray(scheduled.gpp_daily), np.asarray(manual_gpp_daily))
    assert np.allclose(
        np.asarray(scheduled.chain.post_npp.daily_carbon.npp_update.npp),
        np.asarray(manual_scheduled_chain.post_npp.daily_carbon.npp_update.npp),
    )
    initial_biomass_for_maint = np.array(biomass, copy=True)
    initial_biomass_for_maint[0, PFT14, ILEAF, ICARBON] = 100.0
    initial_biomass_for_maint[0, PFT14, IROOT, ICARBON] = 50.0
    maintenance_prescribe_inputs = {**prescribe_inputs, "biomass": initial_biomass_for_maint}
    coeff_maint_zero = np.zeros((nvm, NPARTS), dtype=np.float64)
    coeff_maint_zero[PFT14, :] = 0.01
    maint_resp_slope = np.zeros((nvm, 3), dtype=np.float64)
    maint_ext_coeff = np.ones(nvm, dtype=np.float64) * 0.5
    stempdiag = np.ones((npts, nslm), dtype=np.float64) * 283.15
    manual_maintenance = maintenance_respiration(
        initial_biomass_for_maint,
        np.asarray([293.15], dtype=np.float64),
        np.asarray([293.15], dtype=np.float64),
        stempdiag,
        alloc_inputs_after_constraints["z_soil"],
        np.ones(nvm, dtype=np.float64),
        sla_calc,
        coeff_maint_zero,
        maint_resp_slope,
        maint_ext_coeff,
        is_tree,
        dt_sechiba_days=1.0,
    )
    manual_resp_maint_part = accumulate_resp_maint_part(
        np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        manual_maintenance.resp_maint_part,
    )
    manual_resp_chain = stomate_daily_scheduled_gpp_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        gpp=gpp_instant,
        gpp_daily_current=np.zeros((npts, nvm), dtype=np.float64),
        veget_max=veget_max,
        totfrac_nobio=np.zeros(npts, dtype=np.float64),
        dt_sechiba=86400.0,
        dt_stomate=86400.0,
        do_slow=True,
        prescribe_inputs=maintenance_prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        phenology_kwargs={"dt_days": 1.0},
        alloc_inputs=alloc_inputs_after_constraints,
        allocation_kwargs={"f_fruit": 0.1, "ecureuil": np.zeros(nvm, dtype=np.float64)},
        post_npp_inputs={**post_inputs_without_gpp_or_resp, "resp_maint_part": manual_resp_maint_part},
    )
    maintenance_scheduled = stomate_daily_scheduled_gpp_maintenance_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        gpp=gpp_instant,
        gpp_daily_current=np.zeros((npts, nvm), dtype=np.float64),
        veget_max=veget_max,
        totfrac_nobio=np.zeros(npts, dtype=np.float64),
        dt_sechiba=86400.0,
        dt_stomate=86400.0,
        do_slow=True,
        t2m=np.asarray([293.15], dtype=np.float64),
        t2m_longterm=np.asarray([293.15], dtype=np.float64),
        stempdiag=stempdiag,
        z_soil=alloc_inputs_after_constraints["z_soil"],
        rprof=np.ones(nvm, dtype=np.float64),
        sla_calc=sla_calc,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=maint_ext_coeff,
        is_tree=is_tree,
        resp_maint_part_current=np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        prescribe_inputs=maintenance_prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        phenology_kwargs={"dt_days": 1.0},
        alloc_inputs=alloc_inputs_after_constraints,
        allocation_kwargs={"f_fruit": 0.1, "ecureuil": np.zeros(nvm, dtype=np.float64)},
        post_npp_inputs=post_inputs_without_gpp_or_resp,
        flood_frac=np.zeros(npts, dtype=np.float64),
    )

    assert maintenance_scheduled.requires_trace == ()
    assert np.allclose(
        np.asarray(maintenance_scheduled.maintenance.resp_maint_part),
        np.asarray(manual_maintenance.resp_maint_part),
    )
    assert np.allclose(np.asarray(maintenance_scheduled.resp_maint_part), np.asarray(manual_resp_maint_part))
    assert np.allclose(
        np.asarray(maintenance_scheduled.scheduled_gpp_chain.chain.post_npp.daily_carbon.npp_update.npp),
        np.asarray(manual_resp_chain.chain.post_npp.daily_carbon.npp_update.npp),
    )
    with pytest.raises(ValueError, match="resp_maint_part"):
        stomate_daily_scheduled_gpp_maintenance_prescribe_constraints_alloc_kill_gap_turnover_explicit(
            gpp=gpp_instant,
            gpp_daily_current=np.zeros((npts, nvm), dtype=np.float64),
            veget_max=veget_max,
            totfrac_nobio=np.zeros(npts, dtype=np.float64),
            dt_sechiba=86400.0,
            dt_stomate=86400.0,
            do_slow=True,
            t2m=np.asarray([293.15], dtype=np.float64),
            t2m_longterm=np.asarray([293.15], dtype=np.float64),
            stempdiag=stempdiag,
            z_soil=alloc_inputs_after_constraints["z_soil"],
            rprof=np.ones(nvm, dtype=np.float64),
            sla_calc=sla_calc,
            coeff_maint_zero=coeff_maint_zero,
            maint_resp_slope=maint_resp_slope,
            ext_coeff=maint_ext_coeff,
            is_tree=is_tree,
            resp_maint_part_current=np.zeros((npts, nvm, NPARTS), dtype=np.float64),
            prescribe_inputs=maintenance_prescribe_inputs,
            constraints_inputs=constraints_inputs,
            alloc_inputs=alloc_inputs_after_constraints,
            post_npp_inputs={**post_inputs_without_gpp, "resp_maint_part": manual_resp_maint_part},
        )
    with pytest.raises(ValueError, match="gpp_daily"):
        stomate_daily_scheduled_gpp_prescribe_constraints_alloc_kill_gap_turnover_explicit(
            gpp=gpp_instant,
            gpp_daily_current=np.zeros((npts, nvm), dtype=np.float64),
            veget_max=veget_max,
            totfrac_nobio=np.zeros(npts, dtype=np.float64),
            dt_sechiba=86400.0,
            dt_stomate=86400.0,
            do_slow=True,
            prescribe_inputs=prescribe_inputs,
            constraints_inputs=constraints_inputs,
            alloc_inputs=alloc_inputs_after_constraints,
            post_npp_inputs=post_inputs,
        )
    with pytest.raises(ValueError, match="pft_present"):
        stomate_daily_prescribe_alloc_kill_gap_turnover_explicit(
            prescribe_inputs=prescribe_inputs,
            alloc_inputs=alloc_inputs,
            post_npp_inputs={**post_inputs, "pft_present": manual_prescribe.pft_present},
        )


def test_stomate_daily_carbon_kill_gap_turnover_explicit_wires_dgvm_light_after_turnover():
    npts, nvm = 1, 14
    PFT1 = 1
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT1, ILEAF, ICARBON] = 100.0
    f_alloc = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    f_alloc[0, PFT1, ILEAF] = 1.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, PFT1] = True
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[PFT1] = True
    natural = np.ones(nvm, dtype=bool)
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    common = {
        "resp_maint_part": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "frac_growthresp": np.zeros(nvm, dtype=np.float64),
        "leaf_age": np.ones((npts, nvm, 4), dtype=np.float64),
        "leaf_frac": np.zeros((npts, nvm, 4), dtype=np.float64),
        "age": np.ones((npts, nvm), dtype=np.float64),
        "sla_age1": np.ones((npts, nvm), dtype=np.float64),
        "sla_calc": sla_calc,
        "sla_max": np.ones(nvm, dtype=np.float64) * 0.03,
        "sla_min": np.ones(nvm, dtype=np.float64) * 0.01,
        "herbivores": np.zeros((npts, nvm), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "minmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64),
        "t2m_longterm": np.full(npts, 280.0, dtype=np.float64),
        "t2m_month": np.full(npts, 280.0, dtype=np.float64),
        "t2m_week": np.full(npts, 280.0, dtype=np.float64),
        "gdd_from_growthinit": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "lai": np.ones((npts, nvm), dtype=np.float64),
        "turnover_time": np.ones((npts, nvm), dtype=np.float64) * 30.0,
        "nrec": np.array([1], dtype=np.int32),
        "senescence_type": np.zeros(nvm, dtype=np.int32),
        "is_grassland_manag": np.zeros(nvm, dtype=bool),
        "ok_laidev": np.zeros(nvm, dtype=bool),
        "min_leaf_age_for_senescence": np.zeros(nvm, dtype=np.float64),
        "gdd_senescence": np.ones(nvm, dtype=np.float64) * 1.0e6,
        "senescence_temp": np.zeros((nvm, 3), dtype=np.float64),
        "hum_frac": np.ones(nvm, dtype=np.float64) * 0.2,
        "senescence_hum": np.zeros(nvm, dtype=np.float64),
        "nosenescence_hum": np.ones(nvm, dtype=np.float64),
        "max_turnover_time": np.ones(nvm, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(nvm, dtype=np.float64) * 10.0,
        "leaffall": np.ones(nvm, dtype=np.float64) * 10.0,
        "lai_max": np.ones(nvm, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(nvm, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(nvm, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(nvm, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(nvm, dtype=np.float64) * 730.0,
    }
    state = {
        "lm_lastyearmax": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "cn_ind": np.zeros((npts, nvm), dtype=np.float64),
        "bm_to_litter": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64),
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64) * 6.0,
        "everywhere": np.ones((npts, nvm), dtype=np.float64),
        "veget_max": pft_present.astype(np.float64),
        "height": np.ones((npts, nvm), dtype=np.float64),
        "pasture": np.zeros(nvm, dtype=bool),
    }
    state["ind"][0, PFT1] = 1.0
    state["cn_ind"][0, PFT1] = 1.2
    gap_inputs = {
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((npts, nvm), dtype=np.float64),
        "availability_fact": np.ones(nvm, dtype=np.float64) * 0.14,
        "residence_time": np.ones(nvm, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(nvm, dtype=np.int32),
        "pheno_type": np.zeros(nvm, dtype=np.int32),
        "maxdia": np.ones(nvm, dtype=np.float64) * 10.0,
    }
    light_inputs = {
        "fpc_max": np.ones((npts, nvm), dtype=np.float64),
        "maxfpc_lastyear": np.zeros((npts, nvm), dtype=np.float64),
        "veget_lastlight": np.zeros((npts, nvm), dtype=np.float64),
        "ext_coeff": np.ones(nvm, dtype=np.float64) * 0.5,
    }
    establish_inputs = {
        "regenerate": np.zeros((npts, nvm), dtype=np.float64),
        "neighbours": np.zeros((npts, 8), dtype=np.int32),
        "resolution": np.ones((npts, 2), dtype=np.float64),
        "need_adjacent": np.zeros((npts, nvm), dtype=bool),
        "precip_lastyear": np.zeros(npts, dtype=np.float64),
        "gdd0_lastyear": np.zeros(npts, dtype=np.float64),
        "avail_tree": np.ones(npts, dtype=np.float64),
        "avail_grass": np.ones(npts, dtype=np.float64),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "bm_sapl": np.zeros((nvm, NPARTS, 1), dtype=np.float64),
    }

    result = stomate_daily_carbon_kill_gap_turnover_explicit(
        gpp_daily=np.zeros((npts, nvm), dtype=np.float64),
        biomass_before=biomass,
        f_alloc=f_alloc,
        pft_present=pft_present,
        is_tree=is_tree,
        natural=natural,
        ok_dgvm=True,
        **common,
        **state,
        **gap_inputs,
        **light_inputs,
        **establish_inputs,
    )
    expected_light = light_competition_step(
        veget_max=result.kill_after_gap.veget_max,
        pft_present=result.kill_after_gap.pft_present,
        cn_ind=result.kill_after_gap.cn_ind,
        lai=result.turnover.lai,
        lm_lastyearmax=state["lm_lastyearmax"],
        ind=result.kill_after_gap.ind,
        biomass=result.turnover.biomass,
        bm_to_litter=result.kill_after_gap.bm_to_litter,
        mortality=result.gap.mortality,
        sla_calc=result.daily_carbon.age_sla.sla_calc,
        natural=natural,
        pasture=state["pasture"],
        is_tree=is_tree,
        ok_dgvm=True,
        **light_inputs,
    )
    expected_kill_light = kill_pfts_step(
        lm_lastyearmax=state["lm_lastyearmax"],
        ind=expected_light.ind,
        pft_present=result.kill_after_gap.pft_present,
        cn_ind=result.kill_after_gap.cn_ind,
        biomass=expected_light.biomass,
        senescence=result.turnover.senescence,
        rip_time=result.kill_after_gap.rip_time,
        age=result.turnover.age,
        leaf_age=result.turnover.leaf_age,
        leaf_frac=result.turnover.leaf_frac,
        npp_longterm=result.kill_after_gap.npp_longterm,
        when_growthinit=result.kill_after_gap.when_growthinit,
        everywhere=result.kill_after_gap.everywhere,
        veget_max=result.kill_after_gap.veget_max,
        bm_to_litter=expected_light.bm_to_litter,
        natural=natural,
        pasture=state["pasture"],
        is_tree=is_tree,
        ok_dgvm=True,
    )

    assert result.light is not None
    assert result.kill_after_light is not None
    assert result.establishment_rates is not None
    assert result.establishment_biomass is not None
    assert result.crown_after_establish is not None
    assert np.allclose(np.asarray(result.light.light_death), np.asarray(expected_light.light_death))
    assert np.allclose(np.asarray(result.kill_after_light.bm_to_litter), np.asarray(expected_kill_light.bm_to_litter))
    assert np.allclose(np.asarray(result.bm_to_litter), np.asarray(expected_kill_light.bm_to_litter))


def test_stomate_daily_carbon_kill_gap_turnover_explicit_wires_ordinary_cover_when_requested():
    npts, nvm = 1, 14
    pft = PFT14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 10.0
    f_alloc = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    f_alloc[0, pft, ILEAF] = 1.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    natural = np.ones(nvm, dtype=bool)
    natural[pft] = False
    is_tree = np.zeros(nvm, dtype=bool)
    common = {
        "resp_maint_part": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "frac_growthresp": np.zeros(nvm, dtype=np.float64),
        "leaf_age": np.ones((npts, nvm, 4), dtype=np.float64),
        "leaf_frac": np.zeros((npts, nvm, 4), dtype=np.float64),
        "age": np.ones((npts, nvm), dtype=np.float64),
        "sla_age1": np.ones((npts, nvm), dtype=np.float64),
        "sla_calc": np.ones((npts, nvm), dtype=np.float64) * 0.02,
        "sla_max": np.ones(nvm, dtype=np.float64) * 0.03,
        "sla_min": np.ones(nvm, dtype=np.float64) * 0.01,
        "herbivores": np.zeros((npts, nvm), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "minmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64),
        "t2m_longterm": np.full(npts, 280.0, dtype=np.float64),
        "t2m_month": np.full(npts, 280.0, dtype=np.float64),
        "t2m_week": np.full(npts, 280.0, dtype=np.float64),
        "gdd_from_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "lai": np.ones((npts, nvm), dtype=np.float64),
        "turnover_time": np.ones((npts, nvm), dtype=np.float64) * 30.0,
        "nrec": np.array([1], dtype=np.int32),
        "senescence_type": np.zeros(nvm, dtype=np.int32),
        "is_grassland_manag": np.zeros(nvm, dtype=bool),
        "ok_laidev": np.zeros(nvm, dtype=bool),
        "min_leaf_age_for_senescence": np.zeros(nvm, dtype=np.float64),
        "gdd_senescence": np.ones(nvm, dtype=np.float64) * 1.0e6,
        "senescence_temp": np.zeros((nvm, 3), dtype=np.float64),
        "hum_frac": np.ones(nvm, dtype=np.float64) * 0.2,
        "senescence_hum": np.zeros(nvm, dtype=np.float64),
        "nosenescence_hum": np.ones(nvm, dtype=np.float64),
        "max_turnover_time": np.ones(nvm, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(nvm, dtype=np.float64) * 10.0,
        "leaffall": np.ones(nvm, dtype=np.float64) * 10.0,
        "lai_max": np.ones(nvm, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(nvm, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(nvm, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(nvm, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(nvm, dtype=np.float64) * 730.0,
    }
    state = {
        "lm_lastyearmax": np.ones((npts, nvm), dtype=np.float64),
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "cn_ind": np.zeros((npts, nvm), dtype=np.float64),
        "bm_to_litter": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64),
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64),
        "everywhere": np.ones((npts, nvm), dtype=np.float64),
        "veget_max": np.zeros((npts, nvm), dtype=np.float64),
        "height": np.ones((npts, nvm), dtype=np.float64),
        "pasture": np.zeros(nvm, dtype=bool),
    }
    state["ind"][0, pft] = 1.0
    state["cn_ind"][0, pft] = 0.4
    state["veget_max"][0, pft] = 0.2
    gap_inputs = {
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((npts, nvm), dtype=np.float64),
        "availability_fact": np.ones(nvm, dtype=np.float64) * 0.14,
        "residence_time": np.ones(nvm, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(nvm, dtype=np.int32),
        "pheno_type": np.zeros(nvm, dtype=np.int32),
        "maxdia": np.ones(nvm, dtype=np.float64),
    }
    establish_inputs = {
        "regenerate": np.zeros((npts, nvm), dtype=np.float64),
        "neighbours": np.zeros((npts, 8), dtype=np.int32),
        "resolution": np.ones((npts, 2), dtype=np.float64),
        "need_adjacent": np.zeros((npts, nvm), dtype=bool),
        "precip_lastyear": np.zeros(npts, dtype=np.float64),
        "gdd0_lastyear": np.zeros(npts, dtype=np.float64),
        "avail_tree": np.ones(npts, dtype=np.float64),
        "avail_grass": np.ones(npts, dtype=np.float64),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "bm_sapl": np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        "ext_coeff": np.ones(nvm, dtype=np.float64),
    }
    cover_inputs = {
        "cover_litter": np.zeros((npts, NLITT, nvm, NLEVS, 1), dtype=np.float64),
        "cover_litter_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "cover_litter_not_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "cover_carbon": np.zeros((npts, NCARB, nvm), dtype=np.float64),
        "cover_fuel_1hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_10hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_100hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_1000hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_co2_fire": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_hetero": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_maint": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_growth": np.zeros((npts, nvm), dtype=np.float64),
        "cover_gpp_daily": np.zeros((npts, nvm), dtype=np.float64),
        "cover_deepC_a": np.zeros((npts, 2, nvm), dtype=np.float64),
        "cover_deepC_s": np.zeros((npts, 2, nvm), dtype=np.float64),
        "cover_deepC_p": np.zeros((npts, 2, nvm), dtype=np.float64),
        "is_peat": np.zeros(nvm, dtype=bool),
    }

    result = stomate_daily_carbon_kill_gap_turnover_explicit(
        gpp_daily=np.zeros((npts, nvm), dtype=np.float64),
        biomass_before=biomass,
        f_alloc=f_alloc,
        pft_present=pft_present,
        is_tree=is_tree,
        natural=natural,
        ok_dgvm=False,
        lpj_gap_const_mort=False,
        wire_cover=True,
        **common,
        **state,
        **gap_inputs,
        **establish_inputs,
        **cover_inputs,
    )

    assert result.cover is not None
    assert np.allclose(np.asarray(result.cover.veget_max)[0, pft], 0.2)


def test_stomate_daily_carbon_kill_gap_turnover_explicit_wires_peat_cover_dispatch_when_requested():
    """Fortran: stomate_lpj.f90 lines 1358-1380 dispatch lpj_cover_peat."""

    npts, nvm, ndeep = 1, 14, 2
    source_pft = 1
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, source_pft, ILEAF, ICARBON] = 12.0
    biomass[0, source_pft, IROOT, ICARBON] = 18.0
    f_alloc = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    f_alloc[0, source_pft, ILEAF] = 1.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, source_pft] = True
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, 0] = 0.4
    veget_max[0, source_pft] = 0.6
    veget_max_new = veget_max.copy()
    veget_max_new[0, PFT14] = 0.2
    natural = np.zeros(nvm, dtype=bool)
    natural[source_pft] = True
    natural[PFT14] = True
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    is_tree = np.zeros(nvm, dtype=bool)
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[PFT14, ILEAF, ICARBON] = 3.0
    bm_sapl[PFT14, IROOT, ICARBON] = 5.0
    cover_carbon = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    cover_carbon[0, :, source_pft] = [10.0, 20.0, 30.0]

    common = {
        "resp_maint_part": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "frac_growthresp": np.zeros(nvm, dtype=np.float64),
        "leaf_age": np.ones((npts, nvm, NLEAFAGES), dtype=np.float64),
        "leaf_frac": np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        "age": np.ones((npts, nvm), dtype=np.float64) * 7.0,
        "sla_age1": np.ones((npts, nvm), dtype=np.float64),
        "sla_calc": np.ones((npts, nvm), dtype=np.float64) * 0.02,
        "sla_max": np.ones(nvm, dtype=np.float64) * 0.03,
        "sla_min": np.ones(nvm, dtype=np.float64) * 0.01,
        "herbivores": np.zeros((npts, nvm), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "minmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64),
        "t2m_longterm": np.full(npts, 280.0, dtype=np.float64),
        "t2m_month": np.full(npts, 280.0, dtype=np.float64),
        "t2m_week": np.full(npts, 280.0, dtype=np.float64),
        "gdd_from_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "lai": np.ones((npts, nvm), dtype=np.float64),
        "turnover_time": np.ones((npts, nvm), dtype=np.float64) * 30.0,
        "nrec": np.array([1], dtype=np.int32),
        "senescence_type": np.zeros(nvm, dtype=np.int32),
        "is_grassland_manag": np.zeros(nvm, dtype=bool),
        "ok_laidev": np.zeros(nvm, dtype=bool),
        "min_leaf_age_for_senescence": np.zeros(nvm, dtype=np.float64),
        "gdd_senescence": np.ones(nvm, dtype=np.float64) * 1.0e6,
        "senescence_temp": np.zeros((nvm, 3), dtype=np.float64),
        "hum_frac": np.ones(nvm, dtype=np.float64) * 0.2,
        "senescence_hum": np.zeros(nvm, dtype=np.float64),
        "nosenescence_hum": np.ones(nvm, dtype=np.float64),
        "max_turnover_time": np.ones(nvm, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(nvm, dtype=np.float64) * 10.0,
        "leaffall": np.ones(nvm, dtype=np.float64) * 10.0,
        "lai_max": np.ones(nvm, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(nvm, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(nvm, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(nvm, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(nvm, dtype=np.float64) * 730.0,
    }
    state = {
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64) * 10.0,
        "turnover_longterm": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "lm_lastyearmax": np.zeros((npts, nvm), dtype=np.float64),
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "cn_ind": np.ones((npts, nvm), dtype=np.float64),
        "bm_to_litter": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64),
        "when_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "everywhere": np.zeros((npts, nvm), dtype=np.float64),
        "veget_max": veget_max,
        "height": np.ones((npts, nvm), dtype=np.float64),
        "pasture": np.zeros(nvm, dtype=bool),
    }
    state["ind"][0, source_pft] = 1.0
    gap_inputs = {
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((npts, nvm), dtype=np.float64),
        "availability_fact": np.ones(nvm, dtype=np.float64) * 0.14,
        "residence_time": np.ones(nvm, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(nvm, dtype=np.int32),
        "pheno_type": np.zeros(nvm, dtype=np.int32),
        "maxdia": np.ones(nvm, dtype=np.float64),
    }
    cover_inputs = {
        "cover_litter": np.zeros((npts, NLITT, nvm, NLEVS, 1), dtype=np.float64),
        "cover_litter_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "cover_litter_not_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "cover_carbon": cover_carbon,
        "cover_fuel_1hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_10hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_100hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_1000hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_co2_fire": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_hetero": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_maint": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_growth": np.zeros((npts, nvm), dtype=np.float64),
        "cover_gpp_daily": np.zeros((npts, nvm), dtype=np.float64),
        "cover_deepC_a": np.zeros((npts, ndeep, nvm), dtype=np.float64),
        "cover_deepC_s": np.zeros((npts, ndeep, nvm), dtype=np.float64),
        "cover_deepC_p": np.zeros((npts, ndeep, nvm), dtype=np.float64),
        "cover_veget_max_new": veget_max_new,
        "cover_carbon_save": np.zeros((npts, NCARB, nvm), dtype=np.float64),
        "cover_deepC_a_save": np.zeros((npts, ndeep), dtype=np.float64),
        "cover_deepC_s_save": np.zeros((npts, ndeep), dtype=np.float64),
        "cover_deepC_p_save": np.zeros((npts, ndeep), dtype=np.float64),
        "cover_delta_fsave": np.zeros(npts, dtype=np.float64),
        "cover_liqwt_max_lastyear": np.ones(npts, dtype=np.float64),
        "is_peat": is_peat,
    }

    result = stomate_daily_carbon_kill_gap_turnover_explicit(
        gpp_daily=np.zeros((npts, nvm), dtype=np.float64),
        biomass_before=biomass,
        f_alloc=f_alloc,
        pft_present=pft_present,
        is_tree=is_tree,
        natural=natural,
        ok_dgvm=False,
        lpj_gap_const_mort=True,
        wire_cover=True,
        update_peatfrac=True,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        bm_sapl=bm_sapl,
        cn_sapl=np.ones(nvm, dtype=np.float64) * 0.5,
        dt_days=2.0,
        **common,
        **state,
        **gap_inputs,
        **cover_inputs,
    )
    expected = lpj_cover_peat_step(
        cn_ind=result.kill_after_gap.cn_ind,
        ind=result.kill_after_gap.ind,
        biomass=result.turnover.biomass,
        veget_max_new=veget_max_new,
        veget_max=result.kill_after_gap.veget_max,
        veget_max_old=veget_max,
        litter=cover_inputs["cover_litter"],
        litter_avail=cover_inputs["cover_litter_avail"],
        litter_not_avail=cover_inputs["cover_litter_not_avail"],
        carbon=cover_inputs["cover_carbon"],
        fuel_1hr=cover_inputs["cover_fuel_1hr"],
        fuel_10hr=cover_inputs["cover_fuel_10hr"],
        fuel_100hr=cover_inputs["cover_fuel_100hr"],
        fuel_1000hr=cover_inputs["cover_fuel_1000hr"],
        turnover_daily=result.turnover.turnover,
        bm_to_litter=result.kill_after_gap.bm_to_litter,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        co2_fire=cover_inputs["cover_co2_fire"],
        resp_hetero=cover_inputs["cover_resp_hetero"],
        resp_maint=cover_inputs["cover_resp_maint"],
        resp_growth=cover_inputs["cover_resp_growth"],
        gpp_daily=cover_inputs["cover_gpp_daily"],
        deepC_a=cover_inputs["cover_deepC_a"],
        deepC_s=cover_inputs["cover_deepC_s"],
        deepC_p=cover_inputs["cover_deepC_p"],
        dt_days=2.0,
        age=result.kill_after_gap.age,
        pft_present=result.kill_after_gap.pft_present,
        senescence=result.turnover.senescence,
        when_growthinit=result.kill_after_gap.when_growthinit,
        everywhere=result.kill_after_gap.everywhere,
        leaf_frac=result.turnover.leaf_frac,
        lm_lastyearmax=state["lm_lastyearmax"],
        npp_longterm=result.kill_after_gap.npp_longterm,
        carbon_save=cover_inputs["cover_carbon_save"],
        deepC_a_save=cover_inputs["cover_deepC_a_save"],
        deepC_s_save=cover_inputs["cover_deepC_s_save"],
        deepC_p_save=cover_inputs["cover_deepC_p_save"],
        delta_fsave=cover_inputs["cover_delta_fsave"],
        liqwt_max_lastyear=cover_inputs["cover_liqwt_max_lastyear"],
        natural=natural,
        pasture=state["pasture"],
        is_peat=is_peat,
        is_tree=is_tree,
        bm_sapl=bm_sapl,
        cn_sapl=np.ones(nvm, dtype=np.float64) * 0.5,
        is_grassland_manag=common["is_grassland_manag"],
    )

    assert result.cover is not None
    assert np.allclose(np.asarray(result.cover.veget_max), np.asarray(expected.veget_max))
    assert np.allclose(np.asarray(result.cover.biomass), np.asarray(expected.biomass))
    assert np.allclose(np.asarray(result.cover.carbon), np.asarray(expected.carbon))
    assert np.allclose(np.asarray(result.cover.delta_fsave), np.asarray(expected.delta_fsave))
    assert np.allclose(np.asarray(result.bm_to_litter), np.asarray(expected.bm_to_litter))


def test_stomate_lpj_cover_peat_from_restart_state_writes_dynamic_peat_fields():
    """Fortran: stomate_lpj.f90::lpj_cover_peat lines 2583-3337."""

    npts, nvm, ndeep = 1, 14, 2
    source_pft = 1
    state = stomate_cold_start_entry_state(
        t2m=np.asarray([289.0], dtype=np.float64),
        nvm=nvm,
        nslm=2,
        ndeep=ndeep,
        nelements=1,
    )
    veget_current = np.zeros((npts, nvm), dtype=np.float64)
    veget_current[0, 0] = 0.4
    veget_current[0, source_pft] = 0.6
    veget_new = veget_current.copy()
    veget_new[0, PFT14] = 0.2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, source_pft, ILEAF, ICARBON] = 12.0
    biomass[0, source_pft, IROOT, ICARBON] = 18.0
    carbon = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    carbon[0, :, source_pft] = [10.0, 20.0, 30.0]
    state = state._replace(
        biomass=biomass,
        carbon=carbon,
        ind=np.where(veget_current > 0.0, 1.0, 0.0),
        pft_present=veget_current > 0.0,
        leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64) * 10.0,
        lm_lastyearmax=np.zeros((npts, nvm), dtype=np.float64),
        resp_maint=np.ones((npts, nvm), dtype=np.float64) * 0.1,
        resp_growth=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        gpp_daily=np.ones((npts, nvm), dtype=np.float64) * 0.3,
    )
    natural = np.zeros(nvm, dtype=bool)
    natural[source_pft] = True
    natural[PFT14] = True
    pasture = np.zeros(nvm, dtype=bool)
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    is_tree = np.zeros(nvm, dtype=bool)
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[PFT14, ILEAF, ICARBON] = 3.0
    bm_sapl[PFT14, IROOT, ICARBON] = 5.0
    cn_ind = np.ones((npts, nvm), dtype=np.float64)
    cn_sapl = np.ones(nvm, dtype=np.float64) * 0.5
    zero_litter_avail = np.zeros((npts, NLITT, nvm), dtype=np.float64)
    carbon_save = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    deep_save = np.zeros((npts, ndeep), dtype=np.float64)
    delta_fsave = np.zeros(npts, dtype=np.float64)
    liqwt_max_lastyear = np.ones(npts, dtype=np.float64)

    expected = lpj_cover_peat_step(
        cn_ind=cn_ind,
        ind=state.ind,
        biomass=state.biomass,
        veget_max_new=veget_new,
        veget_max=veget_current,
        veget_max_old=veget_current,
        litter=state.litter,
        litter_avail=zero_litter_avail,
        litter_not_avail=zero_litter_avail,
        carbon=state.carbon,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        turnover_daily=state.turnover_daily,
        bm_to_litter=state.bm_to_litter,
        co2_to_bm=state.co2_to_bm,
        co2_fire=np.zeros((npts, nvm), dtype=np.float64),
        resp_hetero=np.zeros((npts, nvm), dtype=np.float64),
        resp_maint=state.resp_maint,
        resp_growth=state.resp_growth,
        gpp_daily=state.gpp_daily,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        dt_days=2.0,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        leaf_frac=state.leaf_frac,
        lm_lastyearmax=state.lm_lastyearmax,
        npp_longterm=state.npp_longterm,
        carbon_save=carbon_save,
        deepC_a_save=deep_save,
        deepC_s_save=deep_save,
        deepC_p_save=deep_save,
        delta_fsave=delta_fsave,
        liqwt_max_lastyear=liqwt_max_lastyear,
        natural=natural,
        pasture=pasture,
        is_peat=is_peat,
        is_tree=is_tree,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
    )
    result = stomate_lpj_cover_peat_from_restart_state(
        state=state,
        veget_max_current=veget_current,
        veget_max_new=veget_new,
        carbon_save=carbon_save,
        deepC_a_save=deep_save,
        deepC_s_save=deep_save,
        deepC_p_save=deep_save,
        delta_fsave=delta_fsave,
        liqwt_max_lastyear=liqwt_max_lastyear,
        natural=natural,
        pasture=pasture,
        is_peat=is_peat,
        is_tree=is_tree,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        cn_ind=cn_ind,
        dt_days=2.0,
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.cover.veget_max), np.asarray(expected.veget_max))
    assert np.allclose(np.asarray(result.state_after.biomass), np.asarray(expected.biomass))
    assert np.allclose(np.asarray(result.state_after.carbon), np.asarray(expected.carbon))
    assert np.allclose(np.asarray(result.state_after.deepC_a), np.asarray(expected.deepC_a))
    assert np.allclose(np.asarray(result.state_after.soilc_total), np.asarray(expected.deepC_a + expected.deepC_s + expected.deepC_p))
    assert np.allclose(np.asarray(result.cover.carbon_save), np.asarray(expected.carbon_save))
    assert np.allclose(np.asarray(result.cover.delta_fsave), np.asarray(expected.delta_fsave))


def test_stomate_daily_carbon_kill_gap_turnover_explicit_wires_setlai_and_vmax_after_cover():
    npts, nvm = 1, 14
    pft = PFT14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 10.0
    f_alloc = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    f_alloc[0, pft, ILEAF] = 1.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    leaf_age = np.ones((npts, nvm, 4), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, 4), dtype=np.float64)
    leaf_frac[0, pft, 0] = 1.0
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    common = {
        "resp_maint_part": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "frac_growthresp": np.zeros(nvm, dtype=np.float64),
        "leaf_age": leaf_age,
        "leaf_frac": leaf_frac,
        "age": np.ones((npts, nvm), dtype=np.float64),
        "sla_age1": np.ones((npts, nvm), dtype=np.float64),
        "sla_calc": sla_calc,
        "sla_max": np.ones(nvm, dtype=np.float64) * 0.03,
        "sla_min": np.ones(nvm, dtype=np.float64) * 0.01,
        "herbivores": np.zeros((npts, nvm), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "minmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64),
        "t2m_longterm": np.full(npts, 280.0, dtype=np.float64),
        "t2m_month": np.full(npts, 280.0, dtype=np.float64),
        "t2m_week": np.full(npts, 280.0, dtype=np.float64),
        "gdd_from_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "lai": np.ones((npts, nvm), dtype=np.float64),
        "turnover_time": np.ones((npts, nvm), dtype=np.float64) * 30.0,
        "nrec": np.array([1], dtype=np.int32),
        "senescence_type": np.zeros(nvm, dtype=np.int32),
        "is_grassland_manag": np.zeros(nvm, dtype=bool),
        "ok_laidev": np.zeros(nvm, dtype=bool),
        "min_leaf_age_for_senescence": np.zeros(nvm, dtype=np.float64),
        "gdd_senescence": np.ones(nvm, dtype=np.float64) * 1.0e6,
        "senescence_temp": np.zeros((nvm, 3), dtype=np.float64),
        "hum_frac": np.ones(nvm, dtype=np.float64) * 0.2,
        "senescence_hum": np.zeros(nvm, dtype=np.float64),
        "nosenescence_hum": np.ones(nvm, dtype=np.float64),
        "max_turnover_time": np.ones(nvm, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(nvm, dtype=np.float64) * 10.0,
        "leaffall": np.ones(nvm, dtype=np.float64) * 10.0,
        "lai_max": np.ones(nvm, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(nvm, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(nvm, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(nvm, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(nvm, dtype=np.float64) * 730.0,
    }
    state = {
        "lm_lastyearmax": np.ones((npts, nvm), dtype=np.float64),
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "cn_ind": np.zeros((npts, nvm), dtype=np.float64),
        "bm_to_litter": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64),
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64),
        "everywhere": np.ones((npts, nvm), dtype=np.float64),
        "veget_max": np.zeros((npts, nvm), dtype=np.float64),
        "height": np.ones((npts, nvm), dtype=np.float64),
        "pasture": np.zeros(nvm, dtype=bool),
    }
    state["ind"][0, pft] = 1.0
    state["cn_ind"][0, pft] = 0.4
    state["veget_max"][0, pft] = 0.2
    gap_inputs = {
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((npts, nvm), dtype=np.float64),
        "availability_fact": np.ones(nvm, dtype=np.float64) * 0.14,
        "residence_time": np.ones(nvm, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(nvm, dtype=np.int32),
        "pheno_type": np.zeros(nvm, dtype=np.int32),
        "maxdia": np.ones(nvm, dtype=np.float64),
    }
    establish_inputs = {
        "regenerate": np.zeros((npts, nvm), dtype=np.float64),
        "neighbours": np.zeros((npts, 8), dtype=np.int32),
        "resolution": np.ones((npts, 2), dtype=np.float64),
        "need_adjacent": np.zeros((npts, nvm), dtype=bool),
        "precip_lastyear": np.zeros(npts, dtype=np.float64),
        "gdd0_lastyear": np.zeros(npts, dtype=np.float64),
        "avail_tree": np.ones(npts, dtype=np.float64),
        "avail_grass": np.ones(npts, dtype=np.float64),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "bm_sapl": np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        "ext_coeff": np.ones(nvm, dtype=np.float64),
    }
    cover_inputs = {
        "cover_litter": np.zeros((npts, NLITT, nvm, NLEVS, 1), dtype=np.float64),
        "cover_litter_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "cover_litter_not_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "cover_carbon": np.zeros((npts, NCARB, nvm), dtype=np.float64),
        "cover_fuel_1hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_10hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_100hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_fuel_1000hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "cover_co2_fire": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_hetero": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_maint": np.zeros((npts, nvm), dtype=np.float64),
        "cover_resp_growth": np.zeros((npts, nvm), dtype=np.float64),
        "cover_gpp_daily": np.zeros((npts, nvm), dtype=np.float64),
        "cover_deepC_a": np.zeros((npts, 2, nvm), dtype=np.float64),
        "cover_deepC_s": np.zeros((npts, 2, nvm), dtype=np.float64),
        "cover_deepC_p": np.zeros((npts, 2, nvm), dtype=np.float64),
        "is_peat": np.zeros(nvm, dtype=bool),
    }
    vmax_inputs = {
        "vcmax25": np.ones(nvm, dtype=np.float64) * 50.0,
        "n_limfert": np.ones((npts, nvm), dtype=np.float64),
        "leaf_timecst": np.ones(nvm, dtype=np.float64) * 100.0,
        "vmax_leafagecrit": np.ones(nvm, dtype=np.float64) * 100.0,
        "vmax_pheno_type": np.zeros(nvm, dtype=np.int32),
        "vmax_leaf_tab": np.zeros(nvm, dtype=np.int32),
        "vmax_ok_laidev": np.zeros(nvm, dtype=bool),
    }
    output_inputs = {
        "output_litter_above": np.zeros((npts, NLITT, nvm, 1), dtype=np.float64),
        "output_litter_below": np.zeros((npts, NLITT, nvm, 2, 1), dtype=np.float64),
        "output_carbon_32l": np.zeros((npts, NCARB, nvm, 2), dtype=np.float64),
        "output_doc": np.zeros((npts, nvm, 2, 2, NPOOL, 1), dtype=np.float64),
        "output_z_soil": np.asarray([1.0, 2.0], dtype=np.float64),
        "output_zf_soil": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        "output_carb_mass_total_old": np.zeros(npts, dtype=np.float64),
        "output_prod10_total": np.zeros(npts, dtype=np.float64),
        "output_prod100_total": np.zeros(npts, dtype=np.float64),
    }
    output_inputs["output_litter_above"][0, IMETABOLIC, pft, ICARBON] = 2.0
    output_inputs["output_litter_below"][0, ISTRUCTURAL, pft, :, ICARBON] = [3.0, 4.0]
    output_inputs["output_carbon_32l"][0, :, pft, :] = [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]

    result = stomate_daily_carbon_kill_gap_turnover_explicit(
        gpp_daily=np.zeros((npts, nvm), dtype=np.float64),
        biomass_before=biomass,
        f_alloc=f_alloc,
        pft_present=pft_present,
        is_tree=is_tree,
        natural=natural,
        ok_dgvm=False,
        lpj_gap_const_mort=False,
        wire_cover=True,
        wire_harvest_agri=True,
        wire_vmax=True,
        wire_output_diagnostics=True,
        wire_modelout=True,
        frac_turnover_daily=0.55,
        **common,
        **state,
        **gap_inputs,
        **establish_inputs,
        **cover_inputs,
        **vmax_inputs,
        **output_inputs,
    )
    expected_vmax = vmax_step(
        leaf_age=result.establishment_biomass.leaf_age,
        leaf_frac=result.establishment_biomass.leaf_frac,
        vcmax25=vmax_inputs["vcmax25"],
        n_limfert=vmax_inputs["n_limfert"],
        leaf_timecst=vmax_inputs["leaf_timecst"],
        leafagecrit=vmax_inputs["vmax_leafagecrit"],
        pheno_type=vmax_inputs["vmax_pheno_type"],
        leaf_tab=vmax_inputs["vmax_leaf_tab"],
        ok_laidev=vmax_inputs["vmax_ok_laidev"],
    )
    expected_diag = stomate_lpj_output_diagnostics(
        biomass=result.cover.biomass,
        turnover_daily=result.harvest_agri.turnover_daily,
        bm_to_litter=result.harvest_agri.bm_to_litter,
        litter_above=output_inputs["output_litter_above"],
        litter_below=output_inputs["output_litter_below"],
        carbon_32l=output_inputs["output_carbon_32l"],
        doc=output_inputs["output_doc"],
        veget_max=result.cover.veget_max,
        z_soil=output_inputs["output_z_soil"],
        zf_soil=output_inputs["output_zf_soil"],
        carb_mass_total_old=output_inputs["output_carb_mass_total_old"],
        prod10_total=output_inputs["output_prod10_total"],
        prod100_total=output_inputs["output_prod100_total"],
    )

    assert result.cover is not None
    assert result.harvest_agri is not None
    assert result.vmax is not None
    assert result.output_diagnostics is not None
    assert result.modelout is not None
    assert np.allclose(np.asarray(result.lai_after_setlai), np.asarray(result.cover.biomass[:, :, ILEAF, ICARBON]) * sla_calc)
    assert np.allclose(np.asarray(result.vmax.vcmax), np.asarray(expected_vmax.vcmax))
    assert np.allclose(np.asarray(result.output_diagnostics.tot_litter_soil_carb), np.asarray(expected_diag.tot_litter_soil_carb))
    assert np.allclose(np.asarray(result.modelout_fields["LEAF_M"]), np.asarray(result.cover.biomass[:, :, ILEAF, ICARBON]))
    assert np.allclose(np.asarray(result.modelout.GPP_model), np.zeros((npts, nvm), dtype=np.float64))
    assert np.allclose(np.asarray(result.modelout.NPP_model), np.asarray(result.daily_carbon.npp_update.npp))


def _ok_leak_inputs():
    npts, nvm, nelements, nslm, ndeep, nstm = 1, 14, 1, 2, 2, 4
    fuel_shape = (npts, nvm, NLITT, nelements)
    pref = np.zeros(nvm, dtype=np.int32)
    pref[PFT14] = 3
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.5
    litter_above = np.zeros((npts, NLITT, nvm, nelements), dtype=np.float64)
    litter_below = np.zeros((npts, NLITT, nvm, ndeep, nelements), dtype=np.float64)
    lignin_above = np.full((npts, nvm), 0.25, dtype=np.float64)
    lignin_below = np.full((npts, nvm, ndeep), 0.35, dtype=np.float64)
    litterpart = np.zeros((npts, nvm, NLITT), dtype=np.float64)
    dead_leaves = np.zeros((npts, nvm, NLITT), dtype=np.float64)
    bm_to_litter = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    turnover = np.zeros_like(bm_to_litter)
    bm_to_litter[0, PFT14, ILEAF, ICARBON] = 1.0
    bm_to_litter[0, PFT14, ISAPABOVE, ICARBON] = 0.5
    bm_to_litter[0, PFT14, IROOT, ICARBON] = 0.7
    fbact = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    fbact[0, :, PFT14] = [0.01, 0.02]
    fbact_doc = np.zeros_like(fbact)
    fbact_doc[0, :, PFT14] = [0.02, 0.03]
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    carbon_32l[0, :, PFT14, :] = [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
    doc = np.zeros((npts, nvm, ndeep, 2, NPOOL, nelements), dtype=np.float64)
    doc[0, PFT14, :, 0, 4, ICARBON] = [1.0, 1.2]
    soil_mc = np.ones((npts, nslm, nstm), dtype=np.float64)
    soil_mc_32l = np.ones((npts, ndeep, nstm), dtype=np.float64)
    wat_flux = np.zeros((npts, nslm, nstm), dtype=np.float64)
    wat_flux[0, :, 3] = [5.0, 0.0]
    natural = np.zeros(nvm, dtype=bool)
    natural[PFT14] = True
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    return {
        "litter_above": litter_above,
        "litter_below": litter_below,
        "lignin_struc_above": lignin_above,
        "lignin_struc_below": lignin_below,
        "litterpart": litterpart,
        "dead_leaves": dead_leaves,
        "fuel_1hr": np.zeros(fuel_shape, dtype=np.float64),
        "fuel_10hr": np.zeros(fuel_shape, dtype=np.float64),
        "fuel_100hr": np.zeros(fuel_shape, dtype=np.float64),
        "fuel_1000hr": np.zeros(fuel_shape, dtype=np.float64),
        "bm_to_litter": bm_to_litter,
        "turnover": turnover,
        "rprof": np.ones((npts, nvm), dtype=np.float64),
        "veget_max": veget_max,
        "sla_calc": np.full((npts, nvm), 0.1, dtype=np.float64),
        "fbact_litter": fbact,
        "poor_soils": np.zeros(npts, dtype=np.float64),
        "flood_frac": np.asarray([0.1], dtype=np.float64),
        "carbon_32l": carbon_32l,
        "doc": doc,
        "doc_precip2ground": np.zeros((npts, nvm, nelements), dtype=np.float64),
        "doc_precip2canopy": np.zeros((npts, nvm, nelements), dtype=np.float64),
        "dry_dep_canopy": np.zeros((npts, nvm, nelements), dtype=np.float64),
        "interception_storage": np.zeros((npts, nvm, nelements), dtype=np.float64),
        "canopy2ground": np.zeros((npts, nvm), dtype=np.float64),
        "doc_to_topsoil": np.zeros((npts, 4), dtype=np.float64),
        "doc_to_subsoil": np.zeros((npts, 4), dtype=np.float64),
        "fbact_doc": fbact_doc,
        "fbact_soilcarbon": fbact,
        "soil_mc": soil_mc,
        "soil_mc_32l": soil_mc_32l,
        "wat_flux": wat_flux,
        "soilwater_31mm": np.ones((npts, nstm), dtype=np.float64),
        "runoff_per_soil": np.zeros((npts, nstm), dtype=np.float64),
        "drainage_per_soil": np.zeros((npts, nstm), dtype=np.float64),
        "runoff2peat": np.zeros((npts, nstm), dtype=np.float64),
        "fastr": np.asarray([25.0], dtype=np.float64),
        "pref_soil_veg": pref,
        "tprof": np.full((npts, ndeep, nvm), 273.15, dtype=np.float64),
        "clay": np.asarray([0.3], dtype=np.float64),
        "bulk_dens": np.asarray([1.65], dtype=np.float64),
        "z_soil": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        "zf_soil_b": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        "flux_red": np.asarray([1.0], dtype=np.float64),
        "natural": natural,
        "is_peat": is_peat,
        "is_c4": np.zeros(nvm, dtype=bool),
        "resp_maint_part_radia": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "flood_root_radia": np.zeros((npts, nvm), dtype=np.float64),
        "dt_days": 0.5,
        "control_temp_above": np.ones((npts, NLITT), dtype=np.float64),
        "control_moist_above": np.ones((npts, nvm), dtype=np.float64),
        "soil_mc_top_by_pft": np.ones((npts, nvm), dtype=np.float64),
        "dif_doc": np.asarray([1.0e-5], dtype=np.float64),
        "nslm": nslm,
        "ndeep": ndeep,
        "sro_bottom": 2,
    }


def test_ok_leak_explicit_adapter_matches_manual_litter_soilcarbon_doc_chain():
    args = _ok_leak_inputs()

    result = stomate_ok_leak_explicit(**args)
    littercalc = littercalc_leak_core_with_controls(
        args["litter_above"],
        args["litter_below"],
        args["lignin_struc_above"],
        args["lignin_struc_below"],
        args["litterpart"],
        args["dead_leaves"],
        args["fuel_1hr"],
        args["fuel_10hr"],
        args["fuel_100hr"],
        args["fuel_1000hr"],
        args["bm_to_litter"],
        args["turnover"],
        args["rprof"],
        args["z_soil"],
        args["veget_max"],
        args["sla_calc"],
        args["fbact_litter"],
        args["poor_soils"],
        args["flood_frac"],
        dt_days=args["dt_days"],
        control_temp_above=args["control_temp_above"],
        control_moist_above=args["control_moist_above"],
        soil_mc_top_by_pft=args["soil_mc_top_by_pft"],
        nslm=args["nslm"],
        ndeep=args["ndeep"],
        sro_bottom=args["sro_bottom"],
    )
    tf_doc = soilcarbon_leak_tf_doc_ground_fluxes(
        args["doc_precip2ground"],
        args["doc_precip2canopy"],
        args["dry_dep_canopy"],
        args["interception_storage"],
        args["canopy2ground"],
        args["veget_max"],
        args["flood_frac"],
    )
    soilcarbon = soilcarbon_leak_core_step(
        args["carbon_32l"],
        args["doc"],
        littercalc.soilcarbon_input_doc,
        args["doc_to_topsoil"],
        args["doc_to_subsoil"],
        tf_doc.wet_dep_ground,
        tf_doc.wet_dep_flood,
        littercalc.litter_above,
        littercalc.litter_below,
        littercalc.lignin_struc_above,
        littercalc.lignin_struc_below,
        args["fbact_doc"],
        args["fbact_soilcarbon"],
        args["soil_mc"],
        args["soil_mc_32l"],
        args["wat_flux"],
        args["soilwater_31mm"],
        args["runoff_per_soil"],
        args["drainage_per_soil"],
        args["runoff2peat"],
        args["fastr"],
        args["pref_soil_veg"],
        args["veget_max"],
        args["flood_frac"],
        littercalc.floodcarbon_input,
        args["tprof"],
        args["clay"],
        args["bulk_dens"],
        args["z_soil"],
        args["zf_soil_b"],
        args["flux_red"],
        args["natural"],
        args["is_peat"],
        args["is_c4"],
        dt_days=args["dt_days"],
        dif_doc=args["dif_doc"],
        nslm=args["nslm"],
        sro_bottom=args["sro_bottom"],
    )
    doc_export = soilcarbon_leak_doc_export_aggregate(
        soilcarbon.doc_exp,
        args["veget_max"],
        littercalc.resp_hetero_litter,
        soilcarbon.resp_hetero_soil,
        littercalc.resp_hetero_flood,
        soilcarbon.resp_flood_soil,
        args["resp_maint_part_radia"],
        args["flood_root_radia"],
        args["runoff_per_soil"],
        args["drainage_per_soil"],
        args["pref_soil_veg"],
        args["flood_frac"],
        dt_days=args["dt_days"],
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.littercalc.soilcarbon_input_doc), np.asarray(littercalc.soilcarbon_input_doc))
    assert np.allclose(np.asarray(result.soilcarbon.doc), np.asarray(soilcarbon.doc))
    assert np.allclose(np.asarray(result.soilcarbon.doc_exp), np.asarray(soilcarbon.doc_exp))
    assert np.allclose(np.asarray(result.doc_export.doc_exp_agg), np.asarray(doc_export.doc_exp_agg))
    assert np.allclose(np.asarray(result.wet_dep_ground), np.asarray(tf_doc.wet_dep_ground))


def test_ok_leak_explicit_adapter_requires_explicit_litter_controls():
    args = _ok_leak_inputs()
    args["control_temp_above"] = None

    with pytest.raises(ValueError, match="control_temp_above"):
        stomate_ok_leak_explicit(**args)


def test_ok_leak_with_maintenance_adapter_matches_manual_stomate_main_local_order():
    """Fortran: stomate.f90 stomate_main lines 3244-3293 then OK_LEAK call."""

    args = _ok_leak_inputs()
    npts, nvm, nslm = 1, 14, args["nslm"]
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    biomass[0, PFT14, IROOT, ICARBON] = 50.0
    t2m = np.asarray([293.15], dtype=np.float64)
    t2m_longterm = np.asarray([293.15], dtype=np.float64)
    stempdiag = np.full((npts, nslm), 283.15, dtype=np.float64)
    coeff_maint_zero = np.zeros((nvm, NPARTS), dtype=np.float64)
    coeff_maint_zero[PFT14, :] = 0.01
    maint_resp_slope = np.zeros((nvm, 3), dtype=np.float64)
    ext_coeff = np.ones(nvm, dtype=np.float64) * 0.5
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[PFT14] = True
    resp_maint_part = np.ones((npts, nvm, NPARTS), dtype=np.float64) * 0.2
    dt_sechiba = 43200.0
    turnover_daily = np.zeros_like(args["turnover"])
    bm_to_litter_daily = np.zeros_like(args["bm_to_litter"])
    turnover_daily[0, PFT14, ILEAF, ICARBON] = 0.4
    bm_to_litter_daily[0, PFT14, IROOT, ICARBON] = 0.6

    manual_maint = maintenance_respiration(
        biomass,
        t2m,
        t2m_longterm,
        stempdiag,
        args["z_soil"],
        args["rprof"],
        args["sla_calc"],
        coeff_maint_zero,
        maint_resp_slope,
        ext_coeff,
        is_tree,
        dt_sechiba_days=dt_sechiba / 86400.0,
    )
    manual_resp_maint_radia, manual_flood_root = sum_resp_maint_radia(
        manual_maint.resp_maint_part,
        flood_frac=args["flood_frac"],
    )
    manual_resp_maint_part = accumulate_resp_maint_part(resp_maint_part, manual_maint.resp_maint_part)
    prep = stomate_littercalc_entry_prep(
        args["soil_mc"],
        turnover_daily,
        bm_to_litter_daily,
        dt_sechiba=dt_sechiba,
        ndeep=args["ndeep"],
        nslm=args["nslm"],
    )
    manual_args = dict(args)
    manual_args.update(
        {
            "bm_to_litter": prep.bm_to_littercalc,
            "turnover": prep.turnover_littercalc,
            "soil_mc_32l": prep.soil_mc_32l,
            "resp_maint_part_radia": manual_maint.resp_maint_part,
            "flood_root_radia": manual_flood_root,
            "dt_days": dt_sechiba / 86400.0,
        }
    )
    manual_ok_leak = stomate_ok_leak_explicit(**manual_args)

    adapter_args = dict(args)
    for local_name in (
        "bm_to_litter",
        "turnover",
        "soil_mc",
        "soil_mc_32l",
        "resp_maint_part_radia",
            "flood_root_radia",
            "dt_days",
            "rprof",
            "z_soil",
            "sla_calc",
        ):
            adapter_args.pop(local_name, None)

    result = stomate_ok_leak_with_maintenance_explicit(
        biomass=biomass,
        t2m=t2m,
        t2m_longterm=t2m_longterm,
        stempdiag=stempdiag,
        z_soil=args["z_soil"],
        rprof=args["rprof"],
        sla_calc=args["sla_calc"],
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=ext_coeff,
        is_tree=is_tree,
        resp_maint_part=resp_maint_part,
        soil_mc=args["soil_mc"],
        turnover_daily=turnover_daily,
        bm_to_litter_daily=bm_to_litter_daily,
        dt_sechiba=dt_sechiba,
        **adapter_args,
    )

    assert np.allclose(np.asarray(result.maintenance.resp_maint_part), np.asarray(manual_maint.resp_maint_part))
    assert np.allclose(np.asarray(result.resp_maint_radia), np.asarray(manual_resp_maint_radia))
    assert np.allclose(np.asarray(result.flood_root_radia), np.asarray(manual_flood_root))
    assert np.allclose(np.asarray(result.resp_maint_part), np.asarray(manual_resp_maint_part))
    assert np.allclose(np.asarray(result.ok_leak.littercalc.litter_below), np.asarray(manual_ok_leak.littercalc.litter_below))
    assert np.allclose(np.asarray(result.ok_leak.soilcarbon.doc), np.asarray(manual_ok_leak.soilcarbon.doc))
    assert np.allclose(np.asarray(result.ok_leak.doc_export.doc_exp_agg), np.asarray(manual_ok_leak.doc_export.doc_exp_agg))


def test_ok_leak_from_post_npp_adapter_matches_manual_litter_soilcarbon_chain():
    """Fortran: stomate_lpj outputs feed stomate.f90 lines 3288-3489."""

    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)
    leaf_age = np.zeros((1, 14, 4), dtype=np.float64)
    leaf_age[0, PFT14, 0] = 60.0
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, 0] = 1.0
    age = np.zeros((1, 14), dtype=np.float64)
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True
    natural = np.ones(14, dtype=bool)
    sla_age1 = np.ones((1, 14), dtype=np.float64) * 0.015
    sla_calc = np.ones((1, 14), dtype=np.float64) * 0.02
    sla_max = np.ones(14, dtype=np.float64) * 0.05
    sla_min = np.ones(14, dtype=np.float64) * 0.01
    common = {
        "herbivores": np.zeros((1, 14), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((1, 14), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((1, 14), dtype=np.float64),
        "moiavail_week": np.ones((1, 14), dtype=np.float64) * 0.1,
        "t2m_longterm": np.full(1, 293.15, dtype=np.float64),
        "t2m_month": np.full(1, 293.15, dtype=np.float64),
        "t2m_week": np.full(1, 293.15, dtype=np.float64),
        "gdd_from_growthinit": np.zeros((1, 14), dtype=np.float64),
        "lai": np.ones((1, 14), dtype=np.float64) * 2.0,
        "turnover_time": np.zeros((1, 14), dtype=np.float64),
        "nrec": np.zeros((1, 14), dtype=np.int32),
        "senescence_type": np.zeros(14, dtype=np.int32),
        "is_grassland_manag": np.zeros(14, dtype=bool),
        "ok_laidev": np.zeros(14, dtype=bool),
        "min_leaf_age_for_senescence": np.ones(14, dtype=np.float64) * 30.0,
        "gdd_senescence": np.ones(14, dtype=np.float64) * 100.0,
        "senescence_temp": np.zeros((14, 3), dtype=np.float64),
        "hum_frac": np.ones(14, dtype=np.float64) * 0.5,
        "senescence_hum": np.ones(14, dtype=np.float64) * 0.2,
        "nosenescence_hum": np.ones(14, dtype=np.float64) * 0.8,
        "max_turnover_time": np.ones(14, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(14, dtype=np.float64) * 10.0,
        "leaffall": np.ones(14, dtype=np.float64) * 10.0,
        "lai_max": np.ones(14, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(14, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(14, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(14, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(14, dtype=np.float64) * 730.0,
    }
    state = {
        "lm_lastyearmax": np.ones((1, 14), dtype=np.float64) * 50.0,
        "ind": np.ones((1, 14), dtype=np.float64),
        "cn_ind": np.ones((1, 14), dtype=np.float64) * 2.0,
        "bm_to_litter": np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((1, 14), dtype=bool),
        "rip_time": np.ones((1, 14), dtype=np.float64),
        "when_growthinit": np.ones((1, 14), dtype=np.float64) * 6.0,
        "everywhere": np.ones((1, 14), dtype=np.float64),
        "veget_max": pft_present.astype(np.float64),
        "height": np.ones((1, 14), dtype=np.float64),
        "pasture": np.zeros(14, dtype=bool),
    }
    gap_inputs = {
        "npp_longterm": np.ones((1, 14), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        "t2m_min_daily": np.full(1, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((1, 14), dtype=np.float64),
        "availability_fact": np.ones(14, dtype=np.float64) * 0.14,
        "residence_time": np.ones(14, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(14, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(14, dtype=np.int32),
        "pheno_type": np.zeros(14, dtype=np.int32),
        "maxdia": np.ones(14, dtype=np.float64) * 10.0,
    }
    post_npp = stomate_daily_carbon_kill_gap_turnover_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        natural=natural,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
        lpj_gap_const_mort=True,
        **state,
        **gap_inputs,
        **common,
    )
    args = _ok_leak_inputs()
    dt_sechiba = 43200.0
    resp_maint_radia = np.ones((1, 14, NPARTS), dtype=np.float64) * 0.2
    flood_root_radia = np.ones((1, 14), dtype=np.float64) * 0.1
    prep = stomate_littercalc_entry_prep(
        args["soil_mc"],
        post_npp.turnover.turnover,
        post_npp.bm_to_litter,
        dt_sechiba=dt_sechiba,
        ndeep=args["ndeep"],
        nslm=args["nslm"],
    )
    manual_args = dict(args)
    manual_args.update(
        {
            "bm_to_litter": prep.bm_to_littercalc,
            "turnover": prep.turnover_littercalc,
            "soil_mc_32l": prep.soil_mc_32l,
            "resp_maint_part_radia": resp_maint_radia,
            "flood_root_radia": flood_root_radia,
            "dt_days": dt_sechiba / 86400.0,
        }
    )
    manual = stomate_ok_leak_explicit(**manual_args)

    adapter_args = dict(args)
    for local_name in (
        "bm_to_litter",
        "turnover",
        "soil_mc",
        "soil_mc_32l",
        "resp_maint_part_radia",
        "flood_root_radia",
        "dt_days",
    ):
        adapter_args.pop(local_name, None)
    result = stomate_ok_leak_from_post_npp_explicit(
        post_npp=post_npp,
        resp_maint_part_radia=resp_maint_radia,
        flood_root_radia=flood_root_radia,
        soil_mc=args["soil_mc"],
        dt_sechiba=dt_sechiba,
        **adapter_args,
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.prep.turnover_littercalc), np.asarray(prep.turnover_littercalc))
    assert np.allclose(np.asarray(result.prep.bm_to_littercalc), np.asarray(prep.bm_to_littercalc))
    assert np.allclose(np.asarray(result.ok_leak.littercalc.litter_below), np.asarray(manual.littercalc.litter_below))
    assert np.allclose(np.asarray(result.ok_leak.soilcarbon.doc), np.asarray(manual.soilcarbon.doc))
    assert np.allclose(np.asarray(result.ok_leak.doc_export.doc_exp_agg), np.asarray(manual.doc_export.doc_exp_agg))


def test_stomate_lpj_outputs_from_post_npp_ok_leak_matches_manual_diagnostics_and_modelout():
    biomass, f_alloc, resp_maint_part, pft_present, frac_growthresp, gpp_daily = _synthetic_inputs(npts=1)
    leaf_age = np.zeros((1, 14, 4), dtype=np.float64)
    leaf_age[0, PFT14, 0] = 60.0
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, 0] = 1.0
    age = np.zeros((1, 14), dtype=np.float64)
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True
    natural = np.ones(14, dtype=bool)
    common = {
        "herbivores": np.zeros((1, 14), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((1, 14), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((1, 14), dtype=np.float64),
        "moiavail_week": np.ones((1, 14), dtype=np.float64) * 0.1,
        "t2m_longterm": np.full(1, 293.15, dtype=np.float64),
        "t2m_month": np.full(1, 293.15, dtype=np.float64),
        "t2m_week": np.full(1, 293.15, dtype=np.float64),
        "gdd_from_growthinit": np.zeros((1, 14), dtype=np.float64),
        "lai": np.ones((1, 14), dtype=np.float64) * 2.0,
        "turnover_time": np.zeros((1, 14), dtype=np.float64),
        "nrec": np.zeros((1, 14), dtype=np.int32),
        "senescence_type": np.zeros(14, dtype=np.int32),
        "is_grassland_manag": np.zeros(14, dtype=bool),
        "ok_laidev": np.zeros(14, dtype=bool),
        "min_leaf_age_for_senescence": np.ones(14, dtype=np.float64) * 30.0,
        "gdd_senescence": np.ones(14, dtype=np.float64) * 100.0,
        "senescence_temp": np.zeros((14, 3), dtype=np.float64),
        "hum_frac": np.ones(14, dtype=np.float64) * 0.5,
        "senescence_hum": np.ones(14, dtype=np.float64) * 0.2,
        "nosenescence_hum": np.ones(14, dtype=np.float64) * 0.8,
        "max_turnover_time": np.ones(14, dtype=np.float64) * 80.0,
        "min_turnover_time": np.ones(14, dtype=np.float64) * 10.0,
        "leaffall": np.ones(14, dtype=np.float64) * 10.0,
        "lai_max": np.ones(14, dtype=np.float64) * 4.0,
        "leafagecrit": np.ones(14, dtype=np.float64) * 100.0,
        "lai_initmin": np.ones(14, dtype=np.float64) * 0.3,
        "tau_fruit": np.ones(14, dtype=np.float64) * 90.0,
        "tau_sap": np.ones(14, dtype=np.float64) * 730.0,
    }
    state = {
        "lm_lastyearmax": np.ones((1, 14), dtype=np.float64) * 50.0,
        "ind": np.ones((1, 14), dtype=np.float64),
        "cn_ind": np.ones((1, 14), dtype=np.float64) * 2.0,
        "bm_to_litter": np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        "senescence": np.zeros((1, 14), dtype=bool),
        "rip_time": np.ones((1, 14), dtype=np.float64),
        "when_growthinit": np.ones((1, 14), dtype=np.float64) * 6.0,
        "everywhere": np.ones((1, 14), dtype=np.float64),
        "veget_max": pft_present.astype(np.float64),
        "height": np.ones((1, 14), dtype=np.float64),
        "pasture": np.zeros(14, dtype=bool),
    }
    gap_inputs = {
        "npp_longterm": np.ones((1, 14), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        "t2m_min_daily": np.full(1, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((1, 14), dtype=np.float64),
        "availability_fact": np.ones(14, dtype=np.float64) * 0.14,
        "residence_time": np.ones(14, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(14, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(14, dtype=np.int32),
        "pheno_type": np.zeros(14, dtype=np.int32),
        "maxdia": np.ones(14, dtype=np.float64) * 10.0,
    }
    post_npp = stomate_daily_carbon_kill_gap_turnover_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        natural=natural,
        sla_age1=np.ones((1, 14), dtype=np.float64) * 0.015,
        sla_calc=np.ones((1, 14), dtype=np.float64) * 0.02,
        sla_max=np.ones(14, dtype=np.float64) * 0.05,
        sla_min=np.ones(14, dtype=np.float64) * 0.01,
        lpj_gap_const_mort=True,
        **state,
        **gap_inputs,
        **common,
    )
    ok_inputs = _ok_leak_inputs()
    ok_leak = stomate_ok_leak_from_post_npp_explicit(
        post_npp=post_npp,
        resp_maint_part_radia=np.zeros((1, 14, NPARTS), dtype=np.float64),
        flood_root_radia=np.zeros((1, 14), dtype=np.float64),
        soil_mc=ok_inputs["soil_mc"],
        dt_sechiba=43200.0,
        **{k: v for k, v in ok_inputs.items() if k not in {"bm_to_litter", "turnover", "soil_mc", "soil_mc_32l", "resp_maint_part_radia", "flood_root_radia", "dt_days"}},
    )
    manual_diag = stomate_lpj_output_diagnostics(
        biomass=post_npp.turnover.biomass,
        turnover_daily=post_npp.turnover.turnover,
        bm_to_litter=post_npp.bm_to_litter,
        litter_above=ok_leak.ok_leak.littercalc.litter_above,
        litter_below=ok_leak.ok_leak.littercalc.litter_below,
        carbon_32l=ok_leak.ok_leak.soilcarbon.carbon_32l,
        doc=ok_leak.ok_leak.soilcarbon.doc,
        veget_max=state["veget_max"],
        z_soil=ok_inputs["z_soil"][1:],
        zf_soil=ok_inputs["zf_soil_b"],
        carb_mass_total_old=np.asarray([5.0], dtype=np.float64),
    )
    manual_fields = stomate_lpj_history_fields_from_state(
        biomass=post_npp.turnover.biomass,
        gpp_daily=gpp_daily,
        npp_daily=post_npp.daily_carbon.npp_update.npp,
    )
    manual_modelout = compute_modelout_from_fields(manual_fields)

    result = stomate_lpj_outputs_from_post_npp_ok_leak_explicit(
        post_npp=post_npp,
        ok_leak=ok_leak,
        veget_max=state["veget_max"],
        z_soil=ok_inputs["z_soil"][1:],
        zf_soil=ok_inputs["zf_soil_b"],
        carb_mass_total_old=np.asarray([5.0], dtype=np.float64),
        gpp_daily=gpp_daily,
    )

    assert result.requires_trace == ()
    assert np.allclose(np.asarray(result.output_diagnostics.tot_litter_soil_carb), np.asarray(manual_diag.tot_litter_soil_carb))
    assert np.allclose(np.asarray(result.modelout_fields["LEAF_M"]), np.asarray(manual_fields["LEAF_M"]))
    assert np.allclose(
        np.asarray(result.modelout_fields["MAINT_RESP"]),
        np.asarray(post_npp.daily_carbon.npp_update.resp_maint),
    )
    assert np.allclose(
        np.asarray(result.modelout_fields["GROWTH_RESP"]),
        np.asarray(post_npp.daily_carbon.npp_update.resp_growth),
    )
    assert np.allclose(np.asarray(result.modelout.AGB_model), np.asarray(manual_modelout.AGB_model))
    assert np.allclose(np.asarray(result.modelout.NPP_model), np.asarray(manual_modelout.NPP_model))
