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
    DEFAULT_LC,
    IABOVE,
    IACTIVE,
    IBELOW,
    ICARBON,
    ICARBRES,
    IACT,
    IFRUIT,
    ILEAF,
    IMETABO,
    IMETBEL,
    IPASSIVE,
    IROOT,
    IAGRHRTST,
    IAGRHRTPN,
    IAGRSAPPN,
    IAGRSAPST,
    IHEARTABOVE,
    IHEARTBELOW,
    ISAPABOVE,
    ISAPBELOW,
    ISLO,
    ISLOW,
    ISTRABO,
    ISTRBEL,
    ISTRUCTURAL,
    IWPHAR,
    IWPLCC,
    IMETABOLIC,
    NLEVS,
    NLEAFAGES,
    NLITT,
    NCARB,
    NPOOL,
    NPARTS,
    SENESCENCE_COLD,
    SENESCENCE_DRY,
    SENESCENCE_MIXED,
    SENESCENCE_NONE,
    agripeat_adjust_fractions_step,
    agr_allocation_split,
    allocation_step,
    add_incoming_proxy_pft_step,
    constraints_step,
    collect_legacy_scalar_pools_step,
    cover_step,
    deadleaf_cover_from_litter,
    decompose_aboveground_litter_leak,
    decompose_belowground_litter_leak,
    establishment_biomass_step,
    establishment_rates_step,
    empty_pft_lcc_step,
    forest_harvest_legacy_step,
    forest_harvest_proxy_pools_step,
    gap_mortality_step,
    grassland_auto_cut_schedule_step,
    grassland_cutting_spa_step,
    grassland_pre_animal_status_step,
    grassland_role_selection_step,
    grassland_user_cut_schedule_step,
    harvest_herb_legacy_step,
    harvest_agri_step,
    kill_pfts_step,
    initialize_proxy_pft_step,
    crown_step,
    crown_woodmass_ind,
    light_competition_step,
    lcchange_agripeat_finalize_step,
    lcchange_agripeat_redistribution_step,
    lcchange_main_agripeat_step,
    lpj_cover_peat_step,
    LitterIncrementResult,
    litter_control_moisture,
    litter_control_moisture_moyano,
    litter_control_temperature,
    littercalc_add_aboveground_fuel,
    littercalc_aboveground_controls,
    littercalc_activity_factors,
    littercalc_apply_pool_increments,
    littercalc_frac_soil,
    littercalc_leak_core_with_controls,
    littercalc_litter_fractions,
    littercalc_litter_increments,
    littercalc_static_factors,
    littercalc_turnover_times,
    maintenance_respiration,
    npp_leaf_age_sla_age_update,
    npp_closed_update,
    phenology_none_step,
    phenology_step,
    pftinout_step,
    prescribe_step,
    litter_availability_fraction_step,
    lcc_age_class_indices,
    lcc_apply_all_receiver_mtcs_step,
    lcc_calc_cover_step,
    lcchange_deffire_step,
    lcc_cross_give_receive_step,
    lcc_forestry_harvest_sequence_step,
    lcc_glcc_compensation_full_step,
    lcc_gross_glcchange_step,
    lcc_gross_firstday_allocation_step,
    lcc_gross_firstday_single_age_allocation_step,
    lcc_harvest_same_mtc_step,
    lcc_apply_collected_proxy_to_target_step,
    lcc_collect_legacy_pft_step,
    lcchange_main_leak_step,
    lcc_primary_net_transition_sequence_step,
    lcc_receiver_mtc_step,
    lcc_secondary_shift_transition_sequence_step,
    lcc_subtract_outgoing_fractions_step,
    lcc_type_conversion_step,
    product_pool_aging_step,
    root_profile_layer_weights,
    root_temperature,
    set_lai_from_biomass,
    stomate_lpj_cover_dispatch_step,
    stomate_lpj_litter_availability_from_fraction,
    stomate_littercalc_entry_prep,
    stomate_lpj_output_diagnostics,
    stomate_littercalc_scaled_inputs,
    stomate_soil_mc_32l,
    sap_take_step,
    sync_aboveground_fuel_after_decomposition,
    turnover_critical_leaf_age,
    turnover_leaf_age_fall,
    turnover_leaf_mean_age,
    turnover_senescence_flags,
    turnover_step,
    turnover_tree_fruit_and_sapwood,
    update_lignin_fraction,
    vmax_step,
)
from jax_orchidee.stomate.reference import encode_restart_pft_bool_field


PFT14 = 13


def test_stomate_soil_mc_32l_copies_hydrology_layers_and_repeats_layer_11():
    soil_mc = np.arange(2 * 11 * 3, dtype=np.float64).reshape(2, 11, 3)

    result = np.asarray(stomate_soil_mc_32l(soil_mc, ndeep=32, nslm=11))

    assert result.shape == (2, 32, 3)
    assert np.allclose(result[:, :11, :], soil_mc)
    assert np.allclose(result[:, 11:, :], soil_mc[:, 10:11, :])


def test_stomate_soil_mc_32l_validates_source_axis_contract():
    with pytest.raises(ValueError, match="layer axis"):
        stomate_soil_mc_32l(np.zeros((1, 10, 2), dtype=np.float64), ndeep=32, nslm=11)
    with pytest.raises(ValueError, match="ndeep"):
        stomate_soil_mc_32l(np.zeros((1, 11, 2), dtype=np.float64), ndeep=10, nslm=11)


def test_littercalc_scaled_inputs_match_dt_sechiba_over_one_day_formula():
    turnover_daily = np.asarray([[[[2.0], [4.0]]]], dtype=np.float64)
    bm_to_litter = np.asarray([[[[8.0], [16.0]]]], dtype=np.float64)

    turnover_littercalc, bm_to_littercalc = stomate_littercalc_scaled_inputs(
        turnover_daily,
        bm_to_litter,
        dt_sechiba=43200.0,
    )

    assert np.allclose(np.asarray(turnover_littercalc), turnover_daily * 0.5)
    assert np.allclose(np.asarray(bm_to_littercalc), bm_to_litter * 0.5)


def test_littercalc_entry_prep_groups_only_pre_littercalc_local_algebra():
    soil_mc = np.arange(11, dtype=np.float64).reshape(1, 11, 1)
    turnover_daily = np.ones((1, 2, 3, 1), dtype=np.float64) * 3.0
    bm_to_litter = np.ones((1, 2, 3, 1), dtype=np.float64) * 6.0

    result = stomate_littercalc_entry_prep(
        soil_mc,
        turnover_daily,
        bm_to_litter,
        dt_sechiba=21600.0,
        ndeep=13,
        nslm=11,
    )

    assert np.allclose(np.asarray(result.soil_mc_32l)[:, :11, :], soil_mc)
    assert np.allclose(np.asarray(result.soil_mc_32l)[:, 11:, :], soil_mc[:, 10:11, :])
    assert np.allclose(np.asarray(result.turnover_littercalc), turnover_daily * 0.25)
    assert np.allclose(np.asarray(result.bm_to_littercalc), bm_to_litter * 0.25)


def test_littercalc_static_tables_match_fortran_defaults():
    litterfrac = np.asarray(littercalc_litter_fractions())
    frac_soil = np.asarray(littercalc_frac_soil())
    litter_tau, carbon_tau = (np.asarray(x) for x in littercalc_turnover_times())
    grouped = littercalc_static_factors()

    expected_leaf_met = 0.85 - 0.018 * 0.22 * 40.0
    expected_sap_met = 0.85 - 0.018 * 0.35 * 40.0
    assert np.allclose(litterfrac[ILEAF, IMETABOLIC], expected_leaf_met)
    assert np.allclose(litterfrac[ISAPABOVE, IMETABOLIC], expected_sap_met)
    assert np.allclose(litterfrac[:, IMETABOLIC] + litterfrac[:, ISTRUCTURAL], 1.0)
    assert np.allclose(frac_soil[ISTRUCTURAL, IACTIVE, IABOVE], 0.55)
    assert np.allclose(frac_soil[ISTRUCTURAL, ISLOW, IBELOW], 0.7)
    assert np.allclose(frac_soil[IMETABOLIC, IACTIVE, IBELOW], 0.45)
    assert np.allclose(litter_tau[IMETABOLIC], 0.066 * 365.0)
    assert np.allclose(litter_tau[ISTRUCTURAL], 0.245 * 365.0)
    assert np.allclose(carbon_tau[IACTIVE], 0.149 * 365.0)
    assert np.allclose(np.asarray(grouped.litterfrac), litterfrac)
    assert np.allclose(np.asarray(grouped.frac_soil), frac_soil)


def test_littercalc_activity_factors_apply_tau_and_lignin_scaling():
    fbact = np.ones((1, 2, 14), dtype=np.float64) * 0.2
    lignin = np.zeros((1, 14, 2), dtype=np.float64)
    lignin[0, PFT14, :] = [0.1, 0.2]

    result = littercalc_activity_factors(fbact, lignin)

    litter_tau, carbon_tau = (np.asarray(x) for x in littercalc_turnover_times())
    assert np.allclose(np.asarray(result.fbact_met), fbact * carbon_tau[IACTIVE] / litter_tau[IMETABOLIC])
    expected_str = fbact[0, :, PFT14] * carbon_tau[IACTIVE] / litter_tau[ISTRUCTURAL] * np.exp(
        -3.0 * lignin[0, PFT14, :]
    )
    assert np.allclose(np.asarray(result.fbact_str)[0, :, PFT14], expected_str)


def test_littercalc_litter_increments_split_above_below_and_skip_bare_soil():
    npts, nvm, nelements, nslm, ndeep = 1, 14, 1, 2, 4
    bm_to_litter = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    turnover = np.zeros_like(bm_to_litter)
    bm_to_litter[0, 0, ILEAF, ICARBON] = 999.0
    bm_to_litter[0, PFT14, ILEAF, ICARBON] = 10.0
    bm_to_litter[0, PFT14, ISAPABOVE, ICARBON] = 20.0
    bm_to_litter[0, PFT14, ISAPBELOW, ICARBON] = 30.0
    turnover[0, PFT14, IROOT, ICARBON] = 5.0
    dead_leaves = np.zeros((npts, nvm, NLITT), dtype=np.float64)
    rprof = np.ones((npts, nvm), dtype=np.float64)
    z_soil = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    litterfrac = np.asarray(littercalc_litter_fractions())

    result = littercalc_litter_increments(
        bm_to_litter,
        turnover,
        rprof,
        z_soil,
        dead_leaves,
        litterfrac=litterfrac,
        nslm=nslm,
        ndeep=ndeep,
    )

    above_met = litterfrac[ILEAF, IMETABOLIC] * 10.0 + litterfrac[ISAPABOVE, IMETABOLIC] * 20.0
    below_met_source = litterfrac[ISAPBELOW, IMETABOLIC] * 30.0 + litterfrac[IROOT, IMETABOLIC] * 5.0
    weights = np.asarray(root_profile_layer_weights(z_soil, rprof, nslm=nslm))
    assert np.allclose(np.asarray(result.litter_inc_above)[0, IMETABOLIC, PFT14, ICARBON], above_met)
    assert np.allclose(np.asarray(result.litter_inc_above)[0, :, 0, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.litter_inc_below)[0, IMETABOLIC, PFT14, :nslm, ICARBON], below_met_source * weights[0, PFT14])
    assert np.allclose(np.asarray(result.litter_inc_below)[0, :, PFT14, nslm:, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.dead_leaves)[0, PFT14, :], litterfrac[ILEAF, :] * 10.0)
    assert np.allclose(np.asarray(result.lignin_struc_inc_above)[0, PFT14], DEFAULT_LC[ILEAF] * 10.0 + DEFAULT_LC[ISAPABOVE] * 20.0)


def test_littercalc_litter_and_fuel_additions_only_update_carbon_element():
    npts, nvm, nelements, nslm = 1, 14, 2, 1
    bm_to_litter = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    turnover = np.zeros_like(bm_to_litter)
    bm_to_litter[0, PFT14, ILEAF, ICARBON] = 10.0
    bm_to_litter[0, PFT14, ILEAF, 1] = 999.0
    dead_leaves = np.zeros((npts, nvm, NLITT), dtype=np.float64)
    rprof = np.ones((npts, nvm), dtype=np.float64)
    z_soil = np.asarray([0.0, 1.0], dtype=np.float64)

    increments = littercalc_litter_increments(
        bm_to_litter,
        turnover,
        rprof,
        z_soil,
        dead_leaves,
        nslm=nslm,
        ndeep=nslm,
    )
    fuel = littercalc_add_aboveground_fuel(
        np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        bm_to_litter,
        turnover,
    )

    assert np.asarray(increments.litter_inc_above)[0, IMETABOLIC, PFT14, ICARBON] > 0.0
    assert np.allclose(np.asarray(increments.litter_inc_above)[0, :, PFT14, 1], 0.0)
    assert np.asarray(fuel.fuel_total)[0, PFT14, IMETABOLIC, ICARBON] > 0.0
    assert np.allclose(np.asarray(fuel.fuel_total)[0, PFT14, :, 1], 0.0)


def test_littercalc_apply_pool_increments_updates_lignin_and_litterpart_order():
    npts, nvm, nelements, ndeep = 1, 14, 1, 2
    litter_above = np.zeros((npts, NLITT, nvm, nelements), dtype=np.float64)
    litter_below = np.zeros((npts, NLITT, nvm, ndeep, nelements), dtype=np.float64)
    litter_above[0, ISTRUCTURAL, PFT14, ICARBON] = 10.0
    litter_below[0, ISTRUCTURAL, PFT14, 0, ICARBON] = 20.0
    lignin_above = np.zeros((npts, nvm), dtype=np.float64)
    lignin_below = np.zeros((npts, nvm, ndeep), dtype=np.float64)
    lignin_above[0, PFT14] = 0.2
    lignin_below[0, PFT14, 0] = 0.3
    litterpart = np.zeros((npts, nvm, NLITT), dtype=np.float64)
    litterpart[0, PFT14, ISTRUCTURAL] = 0.4
    inc_above = np.zeros_like(litter_above)
    inc_below = np.zeros_like(litter_below)
    inc_above[0, ISTRUCTURAL, PFT14, ICARBON] = 5.0
    inc_below[0, ISTRUCTURAL, PFT14, 0, ICARBON] = 5.0
    inc_pft_above = np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64)
    inc_pft_below = np.zeros((npts, nvm, NLITT, ndeep, nelements), dtype=np.float64)
    inc_pft_above[0, PFT14, ISTRUCTURAL, ICARBON] = 5.0
    lignin_inc_above = np.zeros((npts, nvm), dtype=np.float64)
    lignin_inc_below = np.zeros((npts, nvm, ndeep), dtype=np.float64)
    lignin_inc_above[0, PFT14] = 4.0
    lignin_inc_below[0, PFT14, 0] = 10.0
    increments = LitterIncrementResult(
        litter_inc_above=inc_above,
        litter_inc_below=inc_below,
        litter_inc_pft_above=inc_pft_above,
        litter_inc_pft_below=inc_pft_below,
        dead_leaves=np.zeros((npts, nvm, NLITT), dtype=np.float64),
        lignin_struc_inc_above=lignin_inc_above,
        lignin_struc_inc_below=lignin_inc_below,
    )

    result = littercalc_apply_pool_increments(
        litter_above,
        litter_below,
        lignin_above,
        lignin_below,
        litterpart,
        increments,
    )

    assert np.allclose(np.asarray(result.litter_above)[0, ISTRUCTURAL, PFT14, ICARBON], 15.0)
    assert np.allclose(np.asarray(result.lignin_struc_above)[0, PFT14], (0.2 * 10.0 + 4.0) / 15.0)
    assert np.allclose(np.asarray(result.lignin_struc_below)[0, PFT14, 0], (0.3 * 20.0 + 5.0) / 25.0)
    assert np.allclose(np.asarray(result.litterpart)[0, PFT14, ISTRUCTURAL], (0.4 * 10.0 + 5.0) / 15.0)


def test_decompose_aboveground_litter_leak_matches_source_flux_partition():
    npts, nvm, nelements, ndeep = 1, 14, 1, 3
    litter_above = np.zeros((npts, NLITT, nvm, nelements), dtype=np.float64)
    litter_above[0, ISTRUCTURAL, PFT14, ICARBON] = 100.0
    litter_above[0, IMETABOLIC, PFT14, ICARBON] = 50.0
    dead_leaves = np.ones((npts, nvm, NLITT), dtype=np.float64) * 10.0
    lignin = np.zeros((npts, nvm), dtype=np.float64)
    lignin[0, PFT14] = 0.25
    control_temp = np.ones((npts, NLITT), dtype=np.float64)
    control_moist = np.ones((npts, nvm), dtype=np.float64) * 0.5
    soil_mc_top = np.ones((npts, nvm), dtype=np.float64)
    poor_soils = np.zeros(npts, dtype=np.float64)
    flood_frac = np.asarray([0.3], dtype=np.float64)
    litter_tau, _ = (np.asarray(x) for x in littercalc_turnover_times())
    frac_soil = np.asarray(littercalc_frac_soil())

    result = decompose_aboveground_litter_leak(
        litter_above,
        dead_leaves,
        lignin,
        control_temp,
        control_moist,
        soil_mc_top,
        poor_soils,
        flood_frac,
        dt_days=1.0,
        ndeep=ndeep,
    )

    fd_struct = 1.0 / litter_tau[ISTRUCTURAL] * 0.5 * np.exp(-3.0 * 0.25)
    qd_struct = 100.0 * fd_struct * 0.7
    qd_flood_struct = 100.0 * fd_struct * 0.3 / 3.0
    fd_met = 1.0 / litter_tau[IMETABOLIC] * 0.5
    qd_met = 50.0 * fd_met * 0.7
    qd_flood_met = 50.0 * fd_met * 0.3 / 3.0
    expected_resp = (
        (1.0 - frac_soil[ISTRUCTURAL, IACTIVE, IABOVE]) * qd_struct * 0.75
        + (1.0 - frac_soil[ISTRUCTURAL, ISLOW, IABOVE]) * qd_struct * 0.25
        + (1.0 - frac_soil[IMETABOLIC, IACTIVE, IABOVE]) * qd_met
    )
    assert np.allclose(np.asarray(result.qd_structural)[0, PFT14, ICARBON], qd_struct)
    assert np.allclose(np.asarray(result.litter_above)[0, ISTRUCTURAL, PFT14, ICARBON], 100.0 - qd_struct - qd_flood_struct)
    assert np.allclose(np.asarray(result.litter_above)[0, IMETABOLIC, PFT14, ICARBON], 50.0 - qd_met - qd_flood_met)
    assert np.allclose(np.asarray(result.resp_hetero_litter)[0, PFT14, IABOVE], expected_resp)
    assert np.allclose(np.asarray(result.soilcarbon_input_doc)[0, PFT14, 0, ISTRABO, ICARBON], 0.55 * 0.75 * qd_struct + 0.7 * 0.25 * qd_struct)
    assert np.allclose(np.asarray(result.soilcarbon_input_doc)[0, PFT14, 0, IMETABO, ICARBON], 0.45 * qd_met)
    assert np.allclose(np.asarray(result.floodcarbon_input)[0, PFT14, IACT, ICARBON], 0.55 * 0.75 * qd_flood_struct + 0.45 * qd_flood_met)


def test_littercalc_add_aboveground_fuel_uses_only_above_parts_and_saved_litterfrac():
    npts, nvm, nelements = 1, 14, 1
    bm_to_litter = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    turnover = np.zeros_like(bm_to_litter)
    bm_to_litter[0, 0, ILEAF, ICARBON] = 999.0
    bm_to_litter[0, PFT14, ILEAF, ICARBON] = 10.0
    bm_to_litter[0, PFT14, IFRUIT, ICARBON] = 5.0
    bm_to_litter[0, PFT14, ISAPABOVE, ICARBON] = 20.0
    bm_to_litter[0, PFT14, IHEARTABOVE, ICARBON] = 30.0
    bm_to_litter[0, PFT14, ICARBRES, ICARBON] = 7.0
    bm_to_litter[0, PFT14, ISAPBELOW, ICARBON] = 1000.0
    bm_to_litter[0, PFT14, IHEARTBELOW, ICARBON] = 1000.0
    bm_to_litter[0, PFT14, IROOT, ICARBON] = 1000.0
    litterfrac = np.asarray(littercalc_litter_fractions())
    fuel_shape = (npts, nvm, NLITT, nelements)

    result = littercalc_add_aboveground_fuel(
        np.zeros(fuel_shape, dtype=np.float64),
        np.zeros(fuel_shape, dtype=np.float64),
        np.zeros(fuel_shape, dtype=np.float64),
        np.zeros(fuel_shape, dtype=np.float64),
        bm_to_litter,
        turnover,
        litterfrac=litterfrac,
    )

    for litt in (IMETABOLIC, ISTRUCTURAL):
        leaf_fruit = litterfrac[ILEAF, litt] * 10.0 + litterfrac[IFRUIT, litt] * 5.0
        wood = (
            litterfrac[ISAPABOVE, litt] * 20.0
            + litterfrac[IHEARTABOVE, litt] * 30.0
            + litterfrac[ICARBRES, litt] * 7.0
        )
        assert np.allclose(np.asarray(result.fuel_1hr)[0, PFT14, litt, ICARBON], leaf_fruit + 0.045 * wood)
        assert np.allclose(np.asarray(result.fuel_10hr)[0, PFT14, litt, ICARBON], 0.075 * wood)
        assert np.allclose(np.asarray(result.fuel_100hr)[0, PFT14, litt, ICARBON], 0.210 * wood)
        assert np.allclose(np.asarray(result.fuel_1000hr)[0, PFT14, litt, ICARBON], 0.670 * wood)
        assert np.allclose(np.asarray(result.fuel_total)[0, PFT14, litt, ICARBON], leaf_fruit + wood)
    assert np.allclose(np.asarray(result.fuel_total)[0, 0, :, ICARBON], 0.0)


def test_sync_aboveground_fuel_after_decomposition_subtracts_qd_then_matches_litter_stock():
    fuel_1hr = np.asarray([[[10.0]]], dtype=np.float64)
    fuel_10hr = np.asarray([[[20.0]]], dtype=np.float64)
    fuel_100hr = np.asarray([[[30.0]]], dtype=np.float64)
    fuel_1000hr = np.asarray([[[40.0]]], dtype=np.float64)
    qd = np.asarray([[[10.0]]], dtype=np.float64)
    litter_pool = np.asarray([[[72.0]]], dtype=np.float64)

    result = sync_aboveground_fuel_after_decomposition(
        fuel_1hr,
        fuel_10hr,
        fuel_100hr,
        fuel_1000hr,
        litter_pool,
        qd,
    )

    after_qd = np.asarray([9.0, 18.0, 27.0, 36.0], dtype=np.float64)
    expected = after_qd * (72.0 / 90.0)
    assert np.allclose(np.asarray(result.fuel_1hr)[0, 0, 0], expected[0])
    assert np.allclose(np.asarray(result.fuel_10hr)[0, 0, 0], expected[1])
    assert np.allclose(np.asarray(result.fuel_100hr)[0, 0, 0], expected[2])
    assert np.allclose(np.asarray(result.fuel_1000hr)[0, 0, 0], expected[3])
    assert np.allclose(np.asarray(result.fuel_total)[0, 0, 0], 72.0)


def test_zero_litter_and_fuel_branches_have_finite_zero_gradients():
    zero = jnp.zeros((1,), dtype=jnp.float64)

    lignin_gradient = jax.jit(
        jax.grad(
            lambda increment: jnp.sum(
                update_lignin_fraction(zero, zero, increment, increment)
            )
        )
    )(zero)

    zero_fuel = jnp.zeros((1, 1, 1), dtype=jnp.float64)

    def fuel_objective(qd):
        result = sync_aboveground_fuel_after_decomposition(
            zero_fuel,
            zero_fuel,
            zero_fuel,
            zero_fuel,
            zero_fuel,
            qd,
        )
        return jnp.sum(result.fuel_total)

    fuel_gradient = jax.jit(jax.grad(fuel_objective))(zero_fuel)

    np.testing.assert_array_equal(np.asarray(lignin_gradient), np.zeros(1))
    assert np.all(np.isfinite(np.asarray(lignin_gradient)))
    np.testing.assert_array_equal(np.asarray(fuel_gradient), np.zeros((1, 1, 1)))
    assert np.all(np.isfinite(np.asarray(fuel_gradient)))


def test_deadleaf_cover_from_litter_matches_deadleaf_subroutine_formula():
    dead_leaves = np.zeros((2, 14, NLITT), dtype=np.float64)
    dead_leaves[0, PFT14, IMETABOLIC] = 3.0
    dead_leaves[0, PFT14, ISTRUCTURAL] = 7.0
    dead_leaves[0, 1, IMETABOLIC] = 5.0
    dead_leaves[1, PFT14, ISTRUCTURAL] = 2.0
    veget_max = np.zeros((2, 14), dtype=np.float64)
    veget_max[0, PFT14] = 0.5
    veget_max[0, 1] = 0.2
    veget_max[1, PFT14] = 1.0
    sla_calc = np.ones((2, 14), dtype=np.float64) * 0.1
    sla_calc[0, 1] = 0.4
    sla_calc[1, PFT14] = 0.3

    result = np.asarray(deadleaf_cover_from_litter(dead_leaves, veget_max, sla_calc))

    dead_lai0 = (3.0 + 7.0) * 0.1 * 0.5 + 5.0 * 0.4 * 0.2
    dead_lai1 = 2.0 * 0.3 * 1.0
    assert np.allclose(result, 1.0 - np.exp(-0.5 * np.asarray([dead_lai0, dead_lai1])))


def test_litter_control_moisture_matches_quadratic_clip():
    moist = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64)

    result = np.asarray(litter_control_moisture(moist))

    expected = -1.1 * moist * moist + 2.4 * moist - 0.29
    expected = np.maximum(0.25, np.minimum(1.0, expected))
    assert np.allclose(result, expected)


def _expected_moyano_loop(moist_in, zz_coef_deep, bulk_dens, clay, carbon_32l, veget_max):
    zf = np.concatenate([[0.0], zz_coef_deep])
    layer_fraction = (zf[1:] - zf[:-1]) / zf[-1]
    total_soc = np.sum(
        (carbon_32l[:, IACTIVE, :, :] + carbon_32l[:, ISLOW, :, :] + carbon_32l[:, 2, :, :])
        * veget_max[:, :, None]
        * layer_fraction[None, None, :],
        axis=(1, 2),
    ) / bulk_dens
    out = np.zeros_like(moist_in)
    for i in range(moist_in.shape[0]):
        prsr = np.zeros(120, dtype=np.float64)
        sr = np.zeros(120, dtype=np.float64)
        moistfunc = np.zeros(120, dtype=np.float64)
        for k in range(1, 121):
            x = k / 100.0
            prsr[k - 1] = 1.11066 - 0.83344 * x + 1.48095 * x**2 - 1.02959 * x**3 + 0.07995 * clay[i] + 1.27892 * total_soc[i]
        sr[0] = prsr[0]
        for k in range(1, 120):
            sr[k] = sr[k - 1] * prsr[k]
        moistfunc[:] = sr / np.max(sr)
        ind = int(np.argmax(moistfunc)) + 1
        below = moistfunc[:ind].copy()
        moistfunc[:ind] = moistfunc[:ind] - np.min(below)
        below = moistfunc[:ind].copy()
        moistfunc[:ind] = moistfunc[:ind] / np.max(below)
        index = int(np.rint(moist_in[i] * 100.0))
        out[i] = moistfunc[index - 1] if index > 0 else 0.0
    return out


def test_litter_control_moisture_moyano_matches_fortran_table_rescale():
    moist_in = np.asarray([0.0, 0.3, 0.87], dtype=np.float64)
    zz_coef_deep = np.asarray([0.2, 1.0], dtype=np.float64)
    bulk_dens = np.asarray([1000.0, 900.0, 800.0], dtype=np.float64)
    clay = np.asarray([0.2, 0.4, 0.1], dtype=np.float64)
    carbon_32l = np.zeros((3, 3, 14, 2), dtype=np.float64)
    carbon_32l[:, IACTIVE, PFT14, :] = [[10.0, 20.0], [5.0, 15.0], [12.0, 8.0]]
    carbon_32l[:, ISLOW, PFT14, :] = [[3.0, 4.0], [2.0, 6.0], [1.0, 9.0]]
    carbon_32l[:, 2, PFT14, :] = [[1.0, 2.0], [3.0, 1.0], [2.0, 2.0]]
    veget_max = np.zeros((3, 14), dtype=np.float64)
    veget_max[:, PFT14] = [1.0, 0.5, 0.75]

    result = np.asarray(litter_control_moisture_moyano(moist_in, zz_coef_deep, bulk_dens, clay, carbon_32l, veget_max))
    expected = _expected_moyano_loop(moist_in, zz_coef_deep, bulk_dens, clay, carbon_32l, veget_max)

    assert np.allclose(result, expected)
    assert result[0] == 0.0


def test_litter_control_temperature_matches_fortran_frozen_branches():
    temps = np.asarray([272.0, 272.5, 273.15, 283.15, 303.15], dtype=np.float64)

    standard = np.asarray(litter_control_temperature(temps, 0))
    cutoff_1 = np.asarray(litter_control_temperature(temps, 1))
    cutoff_3 = np.asarray(litter_control_temperature(temps, 2))
    q10_100 = np.asarray(litter_control_temperature(temps, 3))
    q10_1000 = np.asarray(litter_control_temperature(temps, 4))

    assert np.allclose(standard, np.minimum(1.0, np.exp(0.69 * (temps - (273.15 + 30.0)) / 10.0)))
    assert cutoff_1[0] == 0.0
    assert 0.0 < cutoff_1[1] < cutoff_1[2] < cutoff_1[3] < 1.0
    assert cutoff_3[0] > 0.0
    assert q10_100[0] > 0.0
    assert q10_1000[0] < q10_100[0]
    with pytest.raises(ValueError, match="frozen_respiration_func"):
        litter_control_temperature(temps, 99)


def test_littercalc_aboveground_controls_use_pref_soil_tile_and_optional_moyano_branch():
    soil_mc = np.zeros((1, 4, 4), dtype=np.float64)
    soil_mc[0, :, 3] = [0.1, 0.2, 0.3, 0.4]
    soil_mc[0, :, 0] = [0.9, 0.9, 0.9, 0.9]
    z_soil = np.asarray([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64)
    pref_soil_veg = np.zeros(14, dtype=np.int32)
    pref_soil_veg[PFT14] = 3

    result = littercalc_aboveground_controls(
        tsurf=np.asarray([303.15], dtype=np.float64),
        soil_mc=soil_mc,
        z_soil=z_soil,
        pref_soil_veg=pref_soil_veg,
        frozen_respiration_func=1,
    )

    expected_soilhum = 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4
    assert np.allclose(np.asarray(result.soilhum_decomp)[0, PFT14], expected_soilhum)
    assert np.allclose(np.asarray(result.soil_mc_top_by_pft)[0, PFT14], 0.1)
    assert np.allclose(np.asarray(result.control_moist_above)[0, PFT14], np.asarray(litter_control_moisture(expected_soilhum)))
    assert np.allclose(np.asarray(result.control_temp_above)[0, :], 1.0)
    with pytest.raises(ValueError, match="MOIST_FUNC_MOYANO requires"):
        littercalc_aboveground_controls(
            tsurf=np.asarray([303.15], dtype=np.float64),
            soil_mc=soil_mc,
            z_soil=z_soil,
            pref_soil_veg=pref_soil_veg,
            frozen_respiration_func=1,
            moist_func_moyano=True,
        )
    carbon_32l = np.zeros((1, 3, 14, 2), dtype=np.float64)
    carbon_32l[:, IACTIVE, PFT14, :] = [10.0, 20.0]
    veget_max = np.zeros((1, 14), dtype=np.float64)
    veget_max[:, PFT14] = 1.0
    moyano = littercalc_aboveground_controls(
        tsurf=np.asarray([303.15], dtype=np.float64),
        soil_mc=soil_mc,
        z_soil=z_soil,
        pref_soil_veg=pref_soil_veg,
        frozen_respiration_func=1,
        moist_func_moyano=True,
        zz_coef_deep=np.asarray([0.2, 1.0], dtype=np.float64),
        bulk_dens=np.asarray([1000.0], dtype=np.float64),
        clay=np.asarray([0.2], dtype=np.float64),
        carbon_32l=carbon_32l,
        veget_max=veget_max,
    )
    expected = litter_control_moisture_moyano(
        np.asarray([expected_soilhum], dtype=np.float64),
        np.asarray([0.2, 1.0], dtype=np.float64),
        np.asarray([1000.0], dtype=np.float64),
        np.asarray([0.2], dtype=np.float64),
        carbon_32l,
        veget_max,
    )
    assert np.allclose(np.asarray(moyano.control_moist_above)[0, PFT14], np.asarray(expected)[0])


def test_decompose_belowground_litter_leak_respects_sro_bottom_flood_partition():
    npts, nvm, nelements, ndeep = 1, 14, 1, 2
    litter_below = np.zeros((npts, NLITT, nvm, ndeep, nelements), dtype=np.float64)
    litter_below[0, ISTRUCTURAL, PFT14, :, ICARBON] = [100.0, 200.0]
    litter_below[0, IMETABOLIC, PFT14, :, ICARBON] = [50.0, 80.0]
    lignin = np.zeros((npts, nvm, ndeep), dtype=np.float64)
    lignin[0, PFT14, :] = [0.25, 0.4]
    fbact_str = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    fbact_met = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    fbact_str[0, :, PFT14] = [0.01, 0.02]
    fbact_met[0, :, PFT14] = [0.03, 0.04]
    flood_frac = np.asarray([0.3], dtype=np.float64)

    result = decompose_belowground_litter_leak(
        litter_below,
        lignin,
        fbact_met,
        fbact_str,
        flood_frac,
        dt_days=1.0,
        sro_bottom=1,
    )

    qd_struct_layer0 = 100.0 * 0.01 * 0.7
    qd_flood_struct_layer0 = 100.0 * 0.01 * 0.3 / 3.0
    qd_struct_layer1 = 200.0 * 0.02 * 0.7
    qd_flood_struct_layer1 = 200.0 * 0.02 * 0.3 / 3.0
    qd_met_layer0 = 50.0 * 0.03 * 0.7
    qd_flood_met_layer0 = 50.0 * 0.03 * 0.3 / 3.0
    qd_met_layer1 = 80.0 * 0.04 * 0.7
    qd_flood_met_layer1 = 80.0 * 0.04 * 0.3 / 3.0
    assert np.allclose(np.asarray(result.qd_structural)[0, PFT14, 0, ICARBON], qd_struct_layer0)
    assert np.allclose(np.asarray(result.qd_flood_structural)[0, PFT14, 1, ICARBON], qd_flood_struct_layer1)
    shallow_struct_doc = 0.45 * 0.75 * qd_struct_layer0 + 0.7 * 0.25 * qd_struct_layer0
    deep_struct_doc = 0.45 * 0.6 * (qd_struct_layer1 + qd_flood_struct_layer1) + 0.7 * 0.4 * (
        qd_struct_layer1 + qd_flood_struct_layer1
    )
    assert np.allclose(np.asarray(result.soilcarbon_input_doc)[0, PFT14, 0, ISTRBEL, ICARBON], shallow_struct_doc)
    assert np.allclose(np.asarray(result.soilcarbon_input_doc)[0, PFT14, 1, ISTRBEL, ICARBON], deep_struct_doc)
    assert np.allclose(np.asarray(result.soilcarbon_input_doc)[0, PFT14, 0, IMETBEL, ICARBON], 0.45 * qd_met_layer0)
    assert np.allclose(
        np.asarray(result.soilcarbon_input_doc)[0, PFT14, 1, IMETBEL, ICARBON],
        0.45 * (qd_met_layer1 + qd_flood_met_layer1),
    )
    assert np.allclose(
        np.asarray(result.floodcarbon_input)[0, PFT14, IACT, ICARBON],
        0.45 * 0.75 * qd_flood_struct_layer0 + 0.45 * qd_flood_met_layer0,
    )
    assert qd_flood_met_layer1 > 0.0


def test_littercalc_leak_core_with_controls_composes_source_closed_sections_for_pft14():
    npts, nvm, nelements, nslm, ndeep = 1, 14, 1, 2, 2
    litter_above = np.zeros((npts, NLITT, nvm, nelements), dtype=np.float64)
    litter_below = np.zeros((npts, NLITT, nvm, ndeep, nelements), dtype=np.float64)
    lignin_above = np.zeros((npts, nvm), dtype=np.float64)
    lignin_below = np.zeros((npts, nvm, ndeep), dtype=np.float64)
    litterpart = np.zeros((npts, nvm, NLITT), dtype=np.float64)
    dead_leaves = np.zeros((npts, nvm, NLITT), dtype=np.float64)
    fuel_shape = (npts, nvm, NLITT, nelements)
    bm_to_litter = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    turnover = np.zeros_like(bm_to_litter)
    bm_to_litter[0, PFT14, ILEAF, ICARBON] = 10.0
    bm_to_litter[0, PFT14, ISAPABOVE, ICARBON] = 4.0
    bm_to_litter[0, PFT14, IROOT, ICARBON] = 6.0
    rprof = np.ones((npts, nvm), dtype=np.float64)
    z_soil = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.1
    fbact = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    fbact[0, :, PFT14] = [0.01, 0.02]
    control_temp_above = np.ones((npts, NLITT), dtype=np.float64)
    control_moist_above = np.ones((npts, nvm), dtype=np.float64)
    soil_mc_top_by_pft = np.ones((npts, nvm), dtype=np.float64)
    poor_soils = np.zeros(npts, dtype=np.float64)
    flood_frac = np.asarray([0.0], dtype=np.float64)

    result = littercalc_leak_core_with_controls(
        litter_above,
        litter_below,
        lignin_above,
        lignin_below,
        litterpart,
        dead_leaves,
        np.zeros(fuel_shape, dtype=np.float64),
        np.zeros(fuel_shape, dtype=np.float64),
        np.zeros(fuel_shape, dtype=np.float64),
        np.zeros(fuel_shape, dtype=np.float64),
        bm_to_litter,
        turnover,
        rprof,
        z_soil,
        veget_max,
        sla_calc,
        fbact,
        poor_soils,
        flood_frac,
        dt_days=1.0,
        control_temp_above=control_temp_above,
        control_moist_above=control_moist_above,
        soil_mc_top_by_pft=soil_mc_top_by_pft,
        nslm=nslm,
        ndeep=ndeep,
    )

    assert np.asarray(result.litter_above).shape == (npts, NLITT, nvm, nelements)
    assert np.asarray(result.litter_below).shape == (npts, NLITT, nvm, ndeep, nelements)
    assert np.asarray(result.soilcarbon_input_doc).shape == (npts, nvm, ndeep, NPOOL, nelements)
    assert np.asarray(result.deadleaf_cover)[0] > 0.0
    assert np.asarray(result.resp_hetero_litter)[0, PFT14, IABOVE] > 0.0
    assert np.asarray(result.resp_hetero_litter)[0, PFT14, IBELOW] > 0.0
    assert np.allclose(np.asarray(result.floodcarbon_input), 0.0)
    assert np.allclose(
        np.asarray(result.fuel.fuel_total)[0, PFT14, :, ICARBON],
        np.asarray(result.litter_above)[0, :, PFT14, ICARBON],
    )


def test_littercalc_leak_core_can_compute_ordinary_aboveground_controls():
    npts, nvm, nelements, nslm, ndeep = 1, 14, 1, 4, 4
    common_shape_above = (npts, NLITT, nvm, nelements)
    common_shape_below = (npts, NLITT, nvm, ndeep, nelements)
    bm_to_litter = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    turnover = np.zeros_like(bm_to_litter)
    bm_to_litter[0, PFT14, ILEAF, ICARBON] = 4.0
    bm_to_litter[0, PFT14, IROOT, ICARBON] = 4.0
    soil_mc = np.ones((npts, nslm, 4), dtype=np.float64) * 0.5
    pref_soil_veg = np.zeros(nvm, dtype=np.int32)
    pref_soil_veg[PFT14] = 3
    fbact = np.ones((npts, ndeep, nvm), dtype=np.float64) * 0.01
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0

    result = littercalc_leak_core_with_controls(
        np.zeros(common_shape_above, dtype=np.float64),
        np.zeros(common_shape_below, dtype=np.float64),
        np.zeros((npts, nvm), dtype=np.float64),
        np.zeros((npts, nvm, ndeep), dtype=np.float64),
        np.zeros((npts, nvm, NLITT), dtype=np.float64),
        np.zeros((npts, nvm, NLITT), dtype=np.float64),
        np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        bm_to_litter,
        turnover,
        np.ones((npts, nvm), dtype=np.float64),
        np.asarray([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64),
        veget_max,
        np.ones((npts, nvm), dtype=np.float64) * 0.1,
        fbact,
        np.zeros(npts, dtype=np.float64),
        np.zeros(npts, dtype=np.float64),
        dt_days=1.0,
        tsurf=np.asarray([303.15], dtype=np.float64),
        soil_mc=soil_mc,
        z_soil_for_controls=np.asarray([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64),
        pref_soil_veg=pref_soil_veg,
        frozen_respiration_func=1,
        nslm=nslm,
        ndeep=ndeep,
    )

    assert np.asarray(result.deadleaf_cover)[0] > 0.0
    assert np.asarray(result.resp_hetero_litter)[0, PFT14, IABOVE] > 0.0


def test_root_temperature_matches_fortran_exponential_profile():
    stempdiag = np.asarray(
        [
            [280.0, 282.0],
            [286.0, 290.0],
        ],
        dtype=np.float64,
    )
    z_soil = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)
    rprof = np.asarray(
        [
            [0.5, 1.0],
            [1.5, 2.0],
        ],
        dtype=np.float64,
    )

    result = np.asarray(root_temperature(stempdiag[:, None, :], z_soil, rprof))

    rpc = 1.0 / (1.0 - np.exp(-z_soil[-1] / rprof))
    weights = rpc[..., None] * (
        np.exp(-z_soil[:-1] / rprof[..., None])
        - np.exp(-z_soil[1:] / rprof[..., None])
    )
    expected = np.sum(stempdiag[:, None, :] * weights, axis=-1)
    assert np.allclose(result, expected)
    assert result.shape == (2, 2)


def test_set_lai_from_biomass_matches_stomate_lai_formula():
    biomass = np.zeros((2, 14, NPARTS, 1), dtype=np.float64)
    biomass[:, 0, ILEAF, ICARBON] = 999.0
    biomass[:, PFT14, ILEAF, ICARBON] = [10.0, 20.0]
    sla_calc = np.ones((2, 14), dtype=np.float64) * 0.01
    sla_calc[:, PFT14] = [0.02, 0.03]

    lai = np.asarray(set_lai_from_biomass(biomass, sla_calc))

    assert lai.shape == (2, 14)
    assert np.allclose(lai[:, 0], 0.0)
    assert np.allclose(lai[:, PFT14], [0.2, 0.6])


def test_maintenance_respiration_matches_closed_pft14_leaf_and_pool_formula():
    npts, nvm, nslm = 1, 14, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    biomass[0, PFT14, ISAPABOVE, ICARBON] = 50.0
    biomass[0, PFT14, IROOT, ICARBON] = 30.0
    biomass[0, 0, ILEAF, ICARBON] = 999.0

    t2m = np.asarray([293.15], dtype=np.float64)
    t2m_longterm = np.asarray([293.15], dtype=np.float64)
    stempdiag = np.full((npts, nslm), 283.15, dtype=np.float64)
    z_soil = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    rprof = np.ones((npts, nvm), dtype=np.float64)
    sla_calc = np.zeros((npts, nvm), dtype=np.float64)
    sla_calc[0, PFT14] = 0.02
    coeff_maint_zero = np.zeros((nvm, NPARTS), dtype=np.float64)
    coeff_maint_zero[PFT14, :] = 0.01
    maint_resp_slope = np.zeros((nvm, 3), dtype=np.float64)
    maint_resp_slope[PFT14, 0] = 0.01
    ext_coeff = np.full(nvm, 0.5, dtype=np.float64)
    is_tree = np.ones(nvm, dtype=bool)

    result = maintenance_respiration(
        biomass=biomass,
        t2m=t2m,
        t2m_longterm=t2m_longterm,
        stempdiag=stempdiag,
        z_soil=z_soil,
        rprof=rprof,
        sla_calc=sla_calc,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=ext_coeff,
        is_tree=is_tree,
    )

    lai = 2.0
    coeff_above = 0.01 * (1.0 + 0.01 * 20.0)
    coeff_below = 0.01 * (1.0 + 0.01 * 10.0)
    leaf_factor = (0.3 * lai + 1.4 * (1.0 - np.exp(-0.5 * lai))) / lai

    resp = np.asarray(result.resp_maint_part)
    assert np.allclose(np.asarray(result.lai)[0, PFT14], lai)
    assert np.allclose(resp[0, PFT14, ILEAF], coeff_above * 100.0 * leaf_factor)
    assert np.allclose(resp[0, PFT14, ISAPABOVE], coeff_above * 50.0)
    assert np.allclose(resp[0, PFT14, IROOT], coeff_below * 30.0)
    assert np.allclose(resp[0, 0, :], 0.0)
    assert resp.shape == (npts, nvm, NPARTS)


def test_agr_allocation_split_matches_fortran_pft14_active_branch():
    leaf_biomass = np.zeros((1, 14), dtype=np.float64)
    reserve_biomass = np.zeros((1, 14), dtype=np.float64)
    leaf_biomass[0, PFT14] = 10.0
    reserve_biomass[0, PFT14] = 100.0
    sla_calc = np.zeros((1, 14), dtype=np.float64)
    sla_calc[0, PFT14] = 0.01
    senescence = np.zeros((1, 14), dtype=bool)

    result = agr_allocation_split(
        leaf_biomass=leaf_biomass,
        reserve_biomass=reserve_biomass,
        sla_calc=sla_calc,
        senescence=senescence,
        l_to_lsr=np.full((1, 14), 0.3, dtype=np.float64),
        s_to_lsr=np.full((1, 14), 0.4, dtype=np.float64),
        r_to_lsr=np.full((1, 14), 0.3, dtype=np.float64),
        alloc_sap_above=np.full((1, 14), 0.5, dtype=np.float64),
        alloc_agr_st=np.full(14, 0.25, dtype=np.float64),
        alloc_agr_pn=np.full(14, 0.05, dtype=np.float64),
        lai_max=np.full(14, 12.0, dtype=np.float64),
        f_fruit=0.025,
        ecureuil=0.0,
    )

    f_alloc = np.asarray(result.f_alloc)
    remaining = 0.975
    assert np.allclose(f_alloc[0, PFT14, ILEAF], 0.3 * remaining)
    assert np.allclose(f_alloc[0, PFT14, ISAPABOVE], 0.4 * 0.5 * remaining * 0.70)
    assert np.allclose(f_alloc[0, PFT14, IAGRSAPST], 0.4 * 0.5 * remaining * 0.25)
    assert np.allclose(f_alloc[0, PFT14, IAGRSAPPN], 0.4 * 0.5 * remaining * 0.05)
    assert np.allclose(f_alloc[0, PFT14, ISAPBELOW], 0.4 * 0.5 * remaining)
    assert np.allclose(f_alloc[0, PFT14, IROOT], 0.3 * remaining)
    assert np.allclose(f_alloc[0, PFT14, IFRUIT], 0.025)
    assert np.allclose(f_alloc[0, PFT14, ICARBRES], 0.0)
    assert f_alloc.shape == (1, 14, NPARTS)


def test_agr_allocation_split_keeps_senescent_active_pft_in_reserve():
    leaf_biomass = np.zeros((1, 14), dtype=np.float64)
    leaf_biomass[0, PFT14] = 10.0
    senescence = np.zeros((1, 14), dtype=bool)
    senescence[0, PFT14] = True

    result = agr_allocation_split(
        leaf_biomass=leaf_biomass,
        reserve_biomass=np.zeros((1, 14), dtype=np.float64),
        sla_calc=np.ones((1, 14), dtype=np.float64),
        senescence=senescence,
        l_to_lsr=np.full((1, 14), 0.3, dtype=np.float64),
        s_to_lsr=np.full((1, 14), 0.4, dtype=np.float64),
        r_to_lsr=np.full((1, 14), 0.3, dtype=np.float64),
        alloc_sap_above=np.full((1, 14), 0.5, dtype=np.float64),
        alloc_agr_st=np.full(14, 0.25, dtype=np.float64),
        alloc_agr_pn=np.full(14, 0.05, dtype=np.float64),
        lai_max=np.full(14, 12.0, dtype=np.float64),
    )

    expected = np.zeros(NPARTS, dtype=np.float64)
    expected[ICARBRES] = 1.0
    assert np.allclose(np.asarray(result.f_alloc)[0, PFT14, :], expected)


def test_allocation_step_matches_fortran_non_crop_pft14_stress_and_reserve_path():
    npts, nvm, nslm = 1, 14, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 10.0
    biomass[0, PFT14, IROOT, ICARBON] = 5.0
    biomass[0, PFT14, ICARBRES, ICARBON] = 100.0
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_age[0, PFT14, :] = [20.0, 30.0, 40.0, 50.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [0.25, 0.25, 0.25, 0.25]
    lai = np.zeros((npts, nvm), dtype=np.float64)
    lai[0, PFT14] = 0.4
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 1.0
    senescence = np.zeros((npts, nvm), dtype=bool)
    when_growthinit = np.full((npts, nvm), 5.0, dtype=np.float64)
    moiavail_week = np.full((npts, nvm), 0.7, dtype=np.float64)
    tsoil_month = np.asarray([[283.15, 293.15]], dtype=np.float64)
    soilhum_month = np.asarray([[0.4, 0.8]], dtype=np.float64)
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

    result = allocation_step(
        lai=lai,
        veget_max=veget_max,
        senescence=senescence,
        when_growthinit=when_growthinit,
        moiavail_week=moiavail_week,
        tsoil_month=tsoil_month,
        soilhum_month=soilhum_month,
        biomass=biomass,
        age=np.zeros((npts, nvm), dtype=np.float64),
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
        min_l_to_lsr=0.2,
        max_l_to_lsr=0.6,
    )

    use_reserve = min(100.0, 2.0 / 10.0 * (12.0 * 0.5) / 0.02)
    transloc_leaf = 0.3 / (0.3 + 0.35) * use_reserve
    leaf_mass = 10.0 + transloc_leaf
    assert np.allclose(np.asarray(result.transloc_leaf)[0, PFT14], transloc_leaf)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, ILEAF, ICARBON], leaf_mass)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, IROOT, ICARBON], 5.0 + use_reserve - transloc_leaf)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, ICARBRES, ICARBON], 100.0 - use_reserve)

    rpc = 1.0 / (1.0 - np.exp(-z_soil[-1] / 0.2))
    weights = rpc * (np.exp(-z_soil[:-1] / 0.2) - np.exp(-z_soil[1:] / 0.2))
    t_nitrogen = np.sum(tsoil_month[0] * weights)
    h_nitrogen = np.sum(soilhum_month[0] * weights)
    limit_l = max(0.1, np.exp(-0.5 * 0.4))
    limit_w = 0.7
    limit_n_hum = max(0.5, min(1.0, h_nitrogen))
    limit_n_temp = max(0.1, min(1.0, 2.0 ** ((t_nitrogen - 273.15 - 25.0) / 10.0)))
    limit_n = max(0.1, min(1.0, limit_n_hum * limit_n_temp))
    limit_w_or_n = min(limit_w, limit_n)
    r_to_lsr = max(0.15, 0.35 * 3.0 * limit_l / (limit_l + 2.0 * limit_w_or_n))
    s_to_lsr = 0.35 * 3.0 * limit_w_or_n / (2.0 * limit_l + limit_w_or_n)
    l_to_lsr = max(0.2, min(0.6, 1.0 - r_to_lsr - s_to_lsr))
    r_to_lsr = 1.0 - l_to_lsr - s_to_lsr

    assert np.allclose(np.asarray(result.limit_l)[0, PFT14], limit_l)
    assert np.allclose(np.asarray(result.limit_w)[0, PFT14], limit_w)
    assert np.allclose(np.asarray(result.limit_n)[0, PFT14], limit_n)
    assert np.allclose(np.asarray(result.l_to_lsr)[0, PFT14], l_to_lsr)
    assert np.allclose(np.asarray(result.s_to_lsr)[0, PFT14], s_to_lsr)
    assert np.allclose(np.asarray(result.r_to_lsr)[0, PFT14], r_to_lsr)

    f_alloc = np.asarray(result.f_alloc)
    remaining = 0.9
    assert np.allclose(f_alloc[0, PFT14, ILEAF], l_to_lsr * remaining)
    assert np.allclose(f_alloc[0, PFT14, ISAPABOVE], s_to_lsr * remaining * 0.70)
    assert np.allclose(f_alloc[0, PFT14, IAGRSAPST], s_to_lsr * remaining * 0.25)
    assert np.allclose(f_alloc[0, PFT14, IAGRSAPPN], s_to_lsr * remaining * 0.05)
    assert np.allclose(f_alloc[0, PFT14, IROOT], r_to_lsr * remaining)
    assert np.allclose(f_alloc[0, PFT14, IFRUIT], 0.1)
    assert np.allclose(np.sum(f_alloc[0, PFT14, :]), 1.0)
    assert np.allclose(f_alloc[0, 0, ICARBRES], 1.0)
    assert np.allclose(f_alloc[0, 0, :ICARBRES], 0.0)


def test_allocation_step_inactive_zero_reserve_denominators_have_finite_gradients():
    npts, nvm, nslm = 1, 14, 2
    zeros_pft = np.zeros((npts, nvm), dtype=np.float64)
    zeros_biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    zeros_leaf_age = np.zeros(
        (npts, nvm, NLEAFAGES),
        dtype=np.float64,
    )

    def objective(sla_calc):
        result = allocation_step(
            lai=zeros_pft,
            veget_max=zeros_pft,
            senescence=np.zeros((npts, nvm), dtype=bool),
            when_growthinit=zeros_pft,
            moiavail_week=zeros_pft,
            tsoil_month=np.full((npts, nslm), 273.15, dtype=np.float64),
            soilhum_month=np.zeros((npts, nslm), dtype=np.float64),
            biomass=zeros_biomass,
            age=zeros_pft,
            leaf_age=zeros_leaf_age,
            leaf_frac=zeros_leaf_age,
            z_soil=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            sla_calc=sla_calc,
            natural=np.zeros(nvm, dtype=bool),
            pasture=np.zeros(nvm, dtype=bool),
            is_tree=np.zeros(nvm, dtype=bool),
            ok_LAIdev=np.zeros(nvm, dtype=bool),
            r0=np.full(nvm, 0.35, dtype=np.float64),
            s0=np.full(nvm, 0.35, dtype=np.float64),
            ext_coeff=np.full(nvm, 0.5, dtype=np.float64),
            lai_max=np.full(nvm, 12.0, dtype=np.float64),
            lai_max_to_happy=np.full(nvm, 0.5, dtype=np.float64),
            tau_leafinit=np.zeros(nvm, dtype=np.float64),
            alloc_min=np.full(nvm, 0.2, dtype=np.float64),
            alloc_max=np.full(nvm, 0.8, dtype=np.float64),
            demi_alloc=np.full(nvm, 100.0, dtype=np.float64),
            alloc_agr_st=np.zeros(nvm, dtype=np.float64),
            alloc_agr_pn=np.zeros(nvm, dtype=np.float64),
        )
        return jnp.sum(result.transloc_leaf)

    gradient = jax.jit(jax.grad(objective))(zeros_pft)

    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_array_equal(np.asarray(gradient), zeros_pft)


def test_npp_closed_update_matches_source_algebra_with_supplied_alloc_and_maintenance():
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    f_alloc = np.zeros((1, 14, NPARTS), dtype=np.float64)
    f_alloc[0, PFT14, ILEAF] = 0.6
    f_alloc[0, PFT14, IROOT] = 0.4
    resp_maint_part = np.zeros((1, 14, NPARTS), dtype=np.float64)
    resp_maint_part[0, PFT14, ILEAF] = 0.7
    resp_maint_part[0, PFT14, IROOT] = 0.3
    pft_present = np.zeros((1, 14), dtype=bool)
    pft_present[0, PFT14] = True
    frac_growthresp = np.zeros(14, dtype=np.float64)
    frac_growthresp[PFT14] = 0.25

    result = npp_closed_update(
        biomass=biomass,
        gpp=np.full((1, 14), 10.0, dtype=np.float64),
        f_alloc=f_alloc,
        resp_maint_part=resp_maint_part,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        dt_days=1.0,
        tax_max=0.8,
    )

    bm_after_maint = 9.0
    expected_leaf_alloc = 0.75 * (0.6 * bm_after_maint)
    expected_root_alloc = 0.75 * (0.4 * bm_after_maint)
    expected_growth = 0.25 * bm_after_maint
    assert np.allclose(np.asarray(result.bm_alloc)[0, PFT14, ILEAF, ICARBON], expected_leaf_alloc)
    assert np.allclose(np.asarray(result.bm_alloc)[0, PFT14, IROOT, ICARBON], expected_root_alloc)
    assert np.allclose(np.asarray(result.resp_maint)[0, PFT14], 1.0)
    assert np.allclose(np.asarray(result.resp_growth)[0, PFT14], expected_growth)
    assert np.allclose(np.asarray(result.npp)[0, PFT14], 10.0 - 1.0 - expected_growth)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, ILEAF, ICARBON], 100.0 + expected_leaf_alloc)
    assert np.allclose(np.asarray(result.npp)[0, 0], 0.0)


def test_npp_closed_update_pumps_only_fortran_explicit_biomass_pools_when_tax_exceeded():
    biomass = np.full((1, 14, NPARTS, 1), 10.0, dtype=np.float64)
    f_alloc = np.zeros((1, 14, NPARTS), dtype=np.float64)
    f_alloc[0, PFT14, ILEAF] = 1.0
    resp_maint_part = np.zeros((1, 14, NPARTS), dtype=np.float64)
    resp_maint_part[0, PFT14, ILEAF] = 0.8
    resp_maint_part[0, PFT14, IHEARTABOVE] = 0.2
    pft_present = np.zeros((1, 14), dtype=bool)
    pft_present[0, PFT14] = True
    frac_growthresp = np.zeros(14, dtype=np.float64)

    result = npp_closed_update(
        biomass=biomass,
        gpp=np.full((1, 14), 1.0, dtype=np.float64),
        f_alloc=f_alloc,
        resp_maint_part=resp_maint_part,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        dt_days=1.0,
        tax_max=0.8,
    )

    updated = np.asarray(result.biomass)
    assert np.allclose(updated[0, PFT14, ILEAF, ICARBON], 10.0 - 0.16 + 0.2)
    assert np.allclose(updated[0, PFT14, IHEARTABOVE, ICARBON], 10.0)
    assert np.allclose(np.asarray(result.resp_maint)[0, PFT14], 1.0)
    assert np.allclose(np.asarray(result.npp)[0, PFT14], 0.0)
    assert np.asarray(result.bm_alloc).shape == biomass.shape


def test_npp_closed_update_absent_pft_has_finite_zero_maintenance_gradient():
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    gpp = np.zeros((1, 14), dtype=np.float64)
    f_alloc = np.zeros((1, 14, NPARTS), dtype=np.float64)
    pft_present = np.zeros((1, 14), dtype=bool)
    frac_growthresp = np.zeros(14, dtype=np.float64)
    resp_maint_part = jnp.zeros((1, 14, NPARTS), dtype=jnp.float64)

    def biomass_objective(maintenance_parts):
        result = npp_closed_update(
            biomass=biomass,
            gpp=gpp,
            f_alloc=f_alloc,
            resp_maint_part=maintenance_parts,
            pft_present=pft_present,
            frac_growthresp=frac_growthresp,
        )
        return jnp.sum(result.biomass)

    result = npp_closed_update(
        biomass=biomass,
        gpp=gpp,
        f_alloc=f_alloc,
        resp_maint_part=resp_maint_part,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
    )
    gradient = jax.jit(jax.grad(biomass_objective))(resp_maint_part)

    np.testing.assert_array_equal(np.asarray(result.biomass), biomass)
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros_like(resp_maint_part))


def test_npp_leaf_age_sla_age_update_matches_source_bookkeeping_for_grass():
    """Fortran: stomate_npp.f90 npp_calc lines 544-672."""

    npts, nvm = 1, 14
    biomass_old = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass_old[0, PFT14, ILEAF, ICARBON] = 10.0
    biomass_old[0, PFT14, IROOT, ICARBON] = 20.0
    biomass = biomass_old.copy()
    biomass[0, PFT14, ILEAF, ICARBON] = 14.0
    biomass[0, PFT14, IROOT, ICARBON] = 22.0
    bm_alloc = np.zeros_like(biomass)
    bm_alloc[0, PFT14, ILEAF, ICARBON] = 4.0
    bm_alloc[0, PFT14, IROOT, ICARBON] = 2.0
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_age[0, PFT14, :] = [20.0, 40.0, 80.0, 120.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, PFT14, :] = [0.25, 0.25, 0.25, 0.25]
    age = np.zeros((npts, nvm), dtype=np.float64)
    age[0, PFT14] = 2.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, PFT14] = True
    is_tree = np.zeros(nvm, dtype=bool)
    sla_age1 = np.ones((npts, nvm), dtype=np.float64) * 0.015
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    sla_max = np.ones(nvm, dtype=np.float64) * 0.05
    sla_min = np.ones(nvm, dtype=np.float64) * 0.01

    result = npp_leaf_age_sla_age_update(
        biomass,
        biomass_old,
        bm_alloc,
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

    leaf_mass_young = 0.25 * 10.0 + 4.0
    expected_age0 = 20.0 * (leaf_mass_young - 4.0) / leaf_mass_young
    expected_frac0 = leaf_mass_young / 14.0
    expected_other_frac = 0.25 * 10.0 / 14.0
    assert np.allclose(np.asarray(result.leaf_age)[0, PFT14, 0], expected_age0)
    assert np.allclose(np.asarray(result.leaf_frac)[0, PFT14, :], [expected_frac0] + [expected_other_frac] * 3)
    expected_sla_age1 = (0.015 * (leaf_mass_young - 4.0) + 0.05 * 4.0) / leaf_mass_young
    assert np.allclose(np.asarray(result.sla_age1)[0, PFT14], expected_sla_age1)
    expected_sla = (
        expected_sla_age1 * expected_frac0
        + 0.05 * 0.9 * expected_other_frac
        + 0.05 * 0.85 * expected_other_frac
        + 0.05 * 0.8 * expected_other_frac
    )
    assert np.allclose(np.asarray(result.sla_calc)[0, PFT14], expected_sla)
    age_after_increment = 2.0 + 1.0 / 365.0
    expected_age = age_after_increment * ((14.0 + 22.0) - (4.0 + 2.0)) / (14.0 + 22.0)
    assert np.allclose(np.asarray(result.age)[0, PFT14], expected_age)
    assert np.allclose(np.asarray(result.age)[0, 0], 0.0)


def test_npp_leaf_age_sla_age_update_sets_sla_max_for_very_young_weighted_leaf_age():
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 2.0
    bm_alloc = np.zeros_like(biomass)
    leaf_age = np.zeros((1, 14, NLEAFAGES), dtype=np.float64)
    leaf_frac = np.zeros((1, 14, NLEAFAGES), dtype=np.float64)
    leaf_frac[0, PFT14, 0] = 1.0
    pft_present = np.zeros((1, 14), dtype=bool)
    pft_present[0, PFT14] = True

    result = npp_leaf_age_sla_age_update(
        biomass,
        biomass,
        bm_alloc,
        leaf_age,
        leaf_frac,
        np.zeros((1, 14), dtype=np.float64),
        pft_present,
        np.zeros(14, dtype=bool),
        np.ones((1, 14), dtype=np.float64) * 0.01,
        np.ones((1, 14), dtype=np.float64) * 0.02,
        np.ones(14, dtype=np.float64) * 0.05,
        np.ones(14, dtype=np.float64) * 0.01,
        dt_days=1.0,
    )

    assert np.allclose(np.asarray(result.sla_calc)[0, PFT14], 0.05)


def test_npp_leaf_age_sla_age_update_keeps_tree_age_after_increment_only():
    biomass_old = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass = biomass_old.copy()
    biomass[0, PFT14, ILEAF, ICARBON] = 10.0
    bm_alloc = np.zeros_like(biomass)
    bm_alloc[0, PFT14, ILEAF, ICARBON] = 5.0
    pft_present = np.zeros((1, 14), dtype=bool)
    pft_present[0, PFT14] = True
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True
    age = np.zeros((1, 14), dtype=np.float64)
    age[0, PFT14] = 3.0

    result = npp_leaf_age_sla_age_update(
        biomass,
        biomass_old,
        bm_alloc,
        np.zeros((1, 14, NLEAFAGES), dtype=np.float64),
        np.zeros((1, 14, NLEAFAGES), dtype=np.float64),
        age,
        pft_present,
        is_tree,
        np.zeros((1, 14), dtype=np.float64),
        np.zeros((1, 14), dtype=np.float64),
        np.ones(14, dtype=np.float64) * 0.03,
        np.ones(14, dtype=np.float64) * 0.01,
        dt_days=2.0,
    )

    assert np.allclose(np.asarray(result.age)[0, PFT14], 3.0 + 2.0 / 365.0)


def _turnover_fixture(npts=1, nvm=14, nelements=1):
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_frac = np.zeros_like(leaf_age)
    age = np.zeros((npts, nvm), dtype=np.float64)
    lai = np.zeros((npts, nvm), dtype=np.float64)
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[:, PFT14] = True
    params = {
        "pft_present": pft_present,
        "herbivores": np.zeros((npts, nvm), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((npts, nvm), dtype=np.float64),
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64),
        "t2m_longterm": np.full(npts, 293.15, dtype=np.float64),
        "t2m_month": np.full(npts, 293.15, dtype=np.float64),
        "t2m_week": np.full(npts, 293.15, dtype=np.float64),
        "veget_max": pft_present.astype(np.float64),
        "gdd_from_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "leaf_age": leaf_age,
        "leaf_frac": leaf_frac,
        "age": age,
        "lai": lai,
        "biomass": biomass,
        "turnover_time": np.zeros((npts, nvm), dtype=np.float64),
        "nrec": np.zeros((npts, nvm), dtype=np.int32),
        "sla_calc": np.ones((npts, nvm), dtype=np.float64) * 0.02,
        "senescence_type": np.zeros(nvm, dtype=np.int32),
        "is_tree": np.zeros(nvm, dtype=bool),
        "natural": np.ones(nvm, dtype=bool),
        "is_grassland_manag": np.zeros(nvm, dtype=bool),
        "ok_laidev": np.zeros(nvm, dtype=bool),
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
    return params


def test_turnover_leaf_mean_age_matches_fortran_class_sum():
    leaf_age = np.zeros((1, 14, NLEAFAGES), dtype=np.float64)
    leaf_frac = np.zeros_like(leaf_age)
    leaf_age[0, PFT14, :] = [10.0, 20.0, 40.0, 80.0]
    leaf_frac[0, PFT14, :] = [0.1, 0.2, 0.3, 0.4]

    result = np.asarray(turnover_leaf_mean_age(leaf_age, leaf_frac))

    assert np.allclose(result[0, PFT14], 10.0 * 0.1 + 20.0 * 0.2 + 40.0 * 0.3 + 80.0 * 0.4)
    assert np.allclose(result[0, 0], 0.0)


def test_turnover_senescence_flags_matches_mixed_grass_water_and_cold_time_rules():
    params = _turnover_fixture()
    params["biomass"][0, PFT14, ILEAF, ICARBON] = 10.0
    leaf_meanage = np.zeros((1, 14), dtype=np.float64)
    leaf_meanage[0, PFT14] = 60.0
    params["senescence_type"][PFT14] = SENESCENCE_MIXED
    params["moiavail_week"][0, PFT14] = 0.25
    params["maxmoiavail_lastyear"][0, PFT14] = 1.0
    params["minmoiavail_lastyear"][0, PFT14] = 0.0
    params["hum_frac"][PFT14] = 0.5
    params["senescence_hum"][PFT14] = 0.2
    params["nosenescence_hum"][PFT14] = 0.8

    senescence, turnover_time, moiavail_crit, _ = turnover_senescence_flags(
        biomass=params["biomass"],
        leaf_meanage=leaf_meanage,
        maxmoiavail_lastyear=params["maxmoiavail_lastyear"],
        minmoiavail_lastyear=params["minmoiavail_lastyear"],
        moiavail_week=params["moiavail_week"],
        t2m_longterm=params["t2m_longterm"],
        t2m_month=params["t2m_month"],
        t2m_week=params["t2m_week"],
        gdd_from_growthinit=params["gdd_from_growthinit"],
        lai=params["lai"],
        nrec=params["nrec"],
        senescence_type=params["senescence_type"],
        is_tree=params["is_tree"],
        is_grassland_manag=params["is_grassland_manag"],
        ok_laidev=params["ok_laidev"],
        min_leaf_age_for_senescence=params["min_leaf_age_for_senescence"],
        gdd_senescence=params["gdd_senescence"],
        senescence_temp=params["senescence_temp"],
        hum_frac=params["hum_frac"],
        senescence_hum=params["senescence_hum"],
        nosenescence_hum=params["nosenescence_hum"],
        max_turnover_time=params["max_turnover_time"],
        min_turnover_time=params["min_turnover_time"],
        leaffall=params["leaffall"],
        lai_max=params["lai_max"],
    )

    expected_crit = 0.5
    expected_time = 80.0 * (1.0 - (1.0 - 0.25 / expected_crit) ** 2)
    assert not bool(np.asarray(senescence)[0, PFT14])
    assert np.allclose(np.asarray(moiavail_crit)[0, PFT14], expected_crit)
    assert np.allclose(np.asarray(turnover_time)[0, PFT14], expected_time)
    assert np.allclose(np.asarray(turnover_time)[0, 0], 0.0)


def test_turnover_critical_leaf_age_uses_temperature_formula_for_natural_grass():
    t2m_longterm = np.asarray([283.15, 303.15], dtype=np.float64)
    is_tree = np.zeros(14, dtype=bool)
    natural = np.ones(14, dtype=bool)
    leafagecrit = np.ones(14, dtype=np.float64) * 100.0

    result = np.asarray(turnover_critical_leaf_age(t2m_longterm, is_tree, natural, leafagecrit))

    cold_expected = min(150.0, max(75.0, 100.0 - 10.0 * (10.0 - 20.0)))
    warm_expected = min(150.0, max(75.0, 100.0 - 10.0 * (30.0 - 20.0)))
    assert np.allclose(result[:, PFT14], [cold_expected, warm_expected])


def test_turnover_leaf_age_fall_preserves_fortran_sequential_biomass_updates():
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    biomass[0, PFT14, IROOT, ICARBON] = 50.0
    biomass[0, PFT14, IFRUIT, ICARBON] = 20.0
    biomass[0, PFT14, ISAPABOVE, ICARBON] = 30.0
    turnover = np.zeros_like(biomass)
    leaf_age = np.zeros((1, 14, NLEAFAGES), dtype=np.float64)
    leaf_frac = np.zeros_like(leaf_age)
    leaf_age[0, PFT14, 0] = 100.0
    leaf_frac[0, PFT14, 0] = 1.0
    is_tree = np.zeros(14, dtype=bool)
    natural = np.ones(14, dtype=bool)
    ok_laidev = np.zeros(14, dtype=bool)
    leafagecrit = np.ones(14, dtype=np.float64) * 100.0

    new_biomass, new_turnover, new_leaf_frac, leaf_age_crit = turnover_leaf_age_fall(
        biomass,
        turnover,
        leaf_age,
        leaf_frac,
        np.asarray([293.15], dtype=np.float64),
        is_tree=is_tree,
        natural=natural,
        ok_laidev=ok_laidev,
        leafagecrit=leafagecrit,
        dt_days=1.0,
    )

    rate = 1.0 / 100.0
    assert np.allclose(np.asarray(leaf_age_crit)[0, PFT14], 100.0)
    assert np.allclose(np.asarray(new_turnover)[0, PFT14, ILEAF, ICARBON], 100.0 * rate)
    assert np.allclose(np.asarray(new_biomass)[0, PFT14, IROOT, ICARBON], 50.0 - 50.0 * rate)
    assert np.allclose(np.asarray(new_biomass)[0, PFT14, IFRUIT, ICARBON], 20.0 - 20.0 * rate)
    assert np.allclose(np.asarray(new_biomass)[0, PFT14, ISAPABOVE, ICARBON], 30.0 - 30.0 * rate)
    assert np.allclose(np.asarray(new_leaf_frac)[0, PFT14, 0], 1.0)


def test_turnover_tree_fruit_and_sapwood_converts_sap_without_turnover():
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, IFRUIT, ICARBON] = 90.0
    biomass[0, PFT14, ISAPABOVE, ICARBON] = 730.0
    biomass[0, PFT14, ISAPBELOW, ICARBON] = 365.0
    biomass[0, PFT14, IAGRSAPST, ICARBON] = 73.0
    biomass[0, PFT14, IAGRSAPPN, ICARBON] = 146.0
    biomass[0, PFT14, IHEARTABOVE, ICARBON] = 10.0
    biomass[0, PFT14, IHEARTBELOW, ICARBON] = 10.0
    biomass[0, PFT14, IAGRHRTST, ICARBON] = 5.0
    biomass[0, PFT14, IAGRHRTPN, ICARBON] = 5.0
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True

    new_biomass, new_turnover, new_age = turnover_tree_fruit_and_sapwood(
        biomass,
        np.zeros_like(biomass),
        np.ones((1, 14), dtype=np.float64) * 10.0,
        is_tree=is_tree,
        tau_fruit=np.ones(14, dtype=np.float64) * 90.0,
        tau_sap=np.ones(14, dtype=np.float64) * 730.0,
        dt_days=1.0,
    )

    assert np.allclose(np.asarray(new_turnover)[0, PFT14, IFRUIT, ICARBON], 1.0)
    assert np.allclose(np.asarray(new_turnover)[0, PFT14, ISAPABOVE, ICARBON], 0.0)
    assert np.allclose(np.asarray(new_biomass)[0, PFT14, ISAPABOVE, ICARBON], 729.0)
    assert np.allclose(np.asarray(new_biomass)[0, PFT14, IHEARTABOVE, ICARBON], 11.0)
    hw_old = 10.0 + 10.0 + 5.0 + 5.0
    hw_new = 11.0 + 10.5 + 5.1 + 5.2
    assert np.allclose(np.asarray(new_age)[0, PFT14], 10.0 * hw_old / hw_new)


def test_turnover_step_closes_tree_mixed_senescence_and_fruit_sapwood_path():
    params = _turnover_fixture()
    params["is_tree"][PFT14] = True
    params["senescence_type"][PFT14] = SENESCENCE_DRY
    params["biomass"][0, PFT14, ILEAF, ICARBON] = 100.0
    params["biomass"][0, PFT14, IROOT, ICARBON] = 40.0
    params["biomass"][0, PFT14, IFRUIT, ICARBON] = 90.0
    params["biomass"][0, PFT14, ISAPABOVE, ICARBON] = 730.0
    params["biomass"][0, PFT14, IHEARTABOVE, ICARBON] = 10.0
    params["leaf_age"][0, PFT14, :] = [60.0, 0.0, 0.0, 0.0]
    params["leaf_frac"][0, PFT14, :] = [1.0, 0.0, 0.0, 0.0]
    params["moiavail_week"][0, PFT14] = 0.1

    result = turnover_step(**params, dt_days=1.0)

    turnover = np.asarray(result.turnover)
    biomass = np.asarray(result.biomass)
    assert bool(np.asarray(result.senescence)[0, PFT14])
    leaf_age_rate = 1.0 / (100.0 * (100.0 / 60.0) ** 4)
    assert np.allclose(turnover[0, PFT14, ILEAF, ICARBON], 10.0 + 90.0 * leaf_age_rate)
    assert np.allclose(turnover[0, PFT14, IROOT, ICARBON], 4.0 + 36.0 * leaf_age_rate)
    fruit_after_leaf_age = 90.0 - 90.0 * leaf_age_rate
    assert np.allclose(turnover[0, PFT14, IFRUIT, ICARBON], 90.0 * leaf_age_rate + fruit_after_leaf_age / 90.0)
    assert np.allclose(biomass[0, PFT14, ISAPABOVE, ICARBON], 729.0)
    assert np.allclose(biomass[0, PFT14, IHEARTABOVE, ICARBON], 11.0)
    assert np.allclose(turnover[0, 0, :, :], 0.0)


def _gap_fixture(npts=1, nvm=14, nelements=1):
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    bm_to_litter = np.zeros_like(biomass)
    npp_longterm = np.zeros((npts, nvm), dtype=np.float64)
    turnover_longterm = np.zeros_like(biomass)
    lm_lastyearmax = np.ones((npts, nvm), dtype=np.float64)
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[:, PFT14] = True
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[PFT14] = True
    params = {
        "npp_longterm": npp_longterm,
        "turnover_longterm": turnover_longterm,
        "lm_lastyearmax": lm_lastyearmax,
        "pft_present": pft_present,
        "biomass": biomass,
        "ind": np.ones((npts, nvm), dtype=np.float64),
        "bm_to_litter": bm_to_litter,
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((npts, nvm), dtype=np.float64),
        "sla_calc": np.ones((npts, nvm), dtype=np.float64) * 0.02,
        "natural": natural,
        "is_tree": is_tree,
        "pasture": np.zeros(nvm, dtype=bool),
        "availability_fact": np.ones(nvm, dtype=np.float64) * 0.14,
        "residence_time": np.ones(nvm, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(nvm, dtype=np.int32),
        "pheno_type": np.zeros(nvm, dtype=np.int32),
    }
    return params


def test_gap_mortality_constant_tree_mortality_updates_biomass_and_reports_daily_rate():
    params = _gap_fixture()
    params["biomass"][0, PFT14, ILEAF, ICARBON] = 365.0
    params["biomass"][0, PFT14, IROOT, ICARBON] = 73.0

    result = gap_mortality_step(**params, dt_days=2.0, lpj_gap_const_mort=True)

    expected_fraction = 2.0 / (30.0 * 365.0)
    assert np.allclose(np.asarray(result.mortality_fraction)[0, PFT14], expected_fraction)
    assert np.allclose(np.asarray(result.mortality)[0, PFT14], expected_fraction / 2.0)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, PFT14, ILEAF, ICARBON], 365.0 * expected_fraction)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, IROOT, ICARBON], 73.0 * (1.0 - expected_fraction))
    assert np.allclose(np.asarray(result.bm_to_litter)[0, 0, :, :], 0.0)


def test_gap_mortality_growth_efficiency_uses_turnover_longterm_and_sla_vigour():
    params = _gap_fixture()
    params["biomass"][0, PFT14, ILEAF, ICARBON] = 100.0
    params["npp_longterm"][0, PFT14] = 100.0
    params["turnover_longterm"][0, PFT14, ILEAF, ICARBON] = 20.0
    params["turnover_longterm"][0, PFT14, IROOT, ICARBON] = 5.0
    params["lm_lastyearmax"][0, PFT14] = 50.0
    params["sla_calc"][0, PFT14] = 0.02
    params["availability_fact"][PFT14] = 0.14

    result = gap_mortality_step(**params, dt_days=1.0, lpj_gap_const_mort=False, ref_greff=0.035)

    delta = 100.0 - 25.0
    vigour = delta / (50.0 * 0.02)
    availability = 0.14 / (1.0 + 0.035 * vigour)
    expected_fraction = max(0.01, availability) / 365.0
    assert np.allclose(np.asarray(result.delta_biomass)[0, PFT14], delta)
    assert np.allclose(np.asarray(result.vigour)[0, PFT14], vigour)
    assert np.allclose(np.asarray(result.availability)[0, PFT14], availability)
    assert np.allclose(np.asarray(result.mortality_fraction)[0, PFT14], expected_fraction)


def test_gap_mortality_dgvm_low_npp_kills_tree_and_updates_individuals():
    params = _gap_fixture()
    params["biomass"][0, PFT14, ILEAF, ICARBON] = 10.0
    params["biomass"][0, PFT14, ISAPABOVE, ICARBON] = 20.0
    params["npp_longterm"][0, PFT14] = 8.0
    params["ind"][0, PFT14] = 3.0

    result = gap_mortality_step(**params, ok_dgvm=True, npp_longterm_init=10.0)

    assert np.allclose(np.asarray(result.mortality_fraction)[0, PFT14], 1.0)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, :, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, PFT14, ILEAF, ICARBON], 10.0)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, PFT14, ISAPABOVE, ICARBON], 20.0)
    assert np.allclose(np.asarray(result.ind)[0, PFT14], 0.0)


def test_gap_mortality_dgvm_grass_low_npp_excludes_pasture():
    params = _gap_fixture()
    params["is_tree"][PFT14] = False
    params["biomass"][0, PFT14, ILEAF, ICARBON] = 10.0
    params["npp_longterm"][0, PFT14] = 8.0

    result = gap_mortality_step(**params, ok_dgvm=True, npp_longterm_init=10.0)
    assert np.allclose(np.asarray(result.mortality_fraction)[0, PFT14], 1.0)

    params["pasture"][PFT14] = True
    result_pasture = gap_mortality_step(**params, ok_dgvm=True, npp_longterm_init=10.0)
    assert np.allclose(np.asarray(result_pasture.mortality_fraction)[0, PFT14], 0.0)


def test_gap_mortality_dgvm_frost_and_spring_damage_follow_fortran_additive_caps():
    params = _gap_fixture()
    params["biomass"][0, PFT14, ILEAF, ICARBON] = 100.0
    params["npp_longterm"][0, PFT14] = 20.0
    params["t2m_min_daily"][0] = 260.0
    params["tmin_crit"][PFT14] = 270.0
    params["leaf_tab"][PFT14] = 1
    params["pheno_type"][PFT14] = 2
    params["tmin_spring_time"][0, PFT14] = 10.0

    result = gap_mortality_step(
        **params,
        ok_dgvm=True,
        coldness_mort=0.01,
        frost_damage_limit=273.15,
        spring_days_max=40,
    )

    growth_fraction = 0.01 / 365.0
    frost_fraction = 0.01 * (270.0 - 260.0) + growth_fraction
    spring_fraction = 0.01 * (273.15 - 260.0) * 10.0 / 40.0 + frost_fraction
    assert np.allclose(np.asarray(result.mortality_fraction)[0, PFT14], spring_fraction)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, ILEAF, ICARBON], 100.0 * (1.0 - spring_fraction))


def _kill_fixture(npts=1, nvm=14, nelements=1):
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    bm_to_litter = np.zeros_like(biomass)
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[:, PFT14] = True
    biomass[0, PFT14, ILEAF, ICARBON] = 10.0
    biomass[0, PFT14, IROOT, ICARBON] = 20.0
    biomass[0, PFT14, ICARBRES, ICARBON] = 5.0
    params = {
        "lm_lastyearmax": np.ones((npts, nvm), dtype=np.float64),
        "ind": np.ones((npts, nvm), dtype=np.float64),
        "pft_present": pft_present,
        "cn_ind": np.ones((npts, nvm), dtype=np.float64) * 2.0,
        "biomass": biomass,
        "senescence": np.ones((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64) * 3.0,
        "age": np.ones((npts, nvm), dtype=np.float64) * 4.0,
        "leaf_age": np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 5.0,
        "leaf_frac": np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25,
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64) * 6.0,
        "everywhere": np.ones((npts, nvm), dtype=np.float64) * 0.7,
        "veget_max": pft_present.astype(np.float64),
        "bm_to_litter": bm_to_litter,
        "natural": np.ones(nvm, dtype=bool),
        "pasture": np.zeros(nvm, dtype=bool),
        "is_tree": np.ones(nvm, dtype=bool),
    }
    return params


def test_kill_pfts_stomate_branch_moves_all_biomass_to_litter_and_resets_state():
    params = _kill_fixture()
    params["biomass"][0, PFT14, ICARBRES, ICARBON] = 0.0

    result = kill_pfts_step(**params, ok_dgvm=False, min_stomate=1.0e-8)

    assert bool(np.asarray(result.was_killed)[0, PFT14])
    assert np.allclose(np.asarray(result.bm_to_litter)[0, PFT14, ILEAF, ICARBON], 10.0)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, PFT14, IROOT, ICARBON], 20.0)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, :, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.ind)[0, PFT14], 0.0)
    assert np.allclose(np.asarray(result.cn_ind)[0, PFT14], 0.0)
    assert not bool(np.asarray(result.senescence)[0, PFT14])
    assert np.allclose(np.asarray(result.age)[0, PFT14], 0.0)
    assert np.allclose(np.asarray(result.when_growthinit)[0, PFT14], 1.0e33)
    assert np.allclose(np.asarray(result.everywhere)[0, PFT14], 0.0)
    assert np.allclose(np.asarray(result.leaf_age)[0, PFT14, :], 0.0)
    assert np.allclose(np.asarray(result.leaf_frac)[0, PFT14, :], 0.0)
    assert bool(np.asarray(result.pft_present)[0, PFT14])
    assert np.allclose(np.asarray(result.veget_max)[0, PFT14], 1.0)


def test_kill_pfts_dgvm_branch_clears_presence_vegetmax_and_rip_time():
    params = _kill_fixture()
    params["ind"][0, PFT14] = 0.0

    result = kill_pfts_step(**params, ok_dgvm=True, min_stomate=1.0e-8)

    assert bool(np.asarray(result.was_killed)[0, PFT14])
    assert not bool(np.asarray(result.pft_present)[0, PFT14])
    assert np.allclose(np.asarray(result.veget_max)[0, PFT14], 0.0)
    assert np.allclose(np.asarray(result.rip_time)[0, PFT14], 0.0)


def test_kill_pfts_grass_non_constant_mortality_overwrites_npp_longterm():
    params = _kill_fixture()
    params["is_tree"][PFT14] = False
    params["biomass"][0, PFT14, ICARBRES, ICARBON] = -1.0

    result = kill_pfts_step(**params, ok_dgvm=False, lpj_gap_const_mort=False)
    result_const = kill_pfts_step(**params, ok_dgvm=False, lpj_gap_const_mort=True)

    assert np.allclose(np.asarray(result.npp_longterm)[0, PFT14], 500.0)
    assert np.allclose(np.asarray(result_const.npp_longterm)[0, PFT14], 100.0)


def test_kill_pfts_skips_pasture_non_natural_and_bare_soil():
    params = _kill_fixture()
    params["biomass"][0, PFT14, ICARBRES, ICARBON] = 0.0
    params["biomass"][0, 0, ICARBRES, ICARBON] = 0.0
    params["pft_present"][0, 0] = True
    params["pasture"][PFT14] = True

    pasture_result = kill_pfts_step(**params, ok_dgvm=False)
    assert not bool(np.asarray(pasture_result.was_killed)[0, PFT14])
    assert not bool(np.asarray(pasture_result.was_killed)[0, 0])

    params["pasture"][PFT14] = False
    params["natural"][PFT14] = False
    non_natural_result = kill_pfts_step(**params, ok_dgvm=False)
    assert not bool(np.asarray(non_natural_result.was_killed)[0, PFT14])


def test_crown_woodmass_ind_matches_stomate_lpj_dgvm_and_static_formulas():
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ISAPABOVE, ICARBON] = 10.0
    biomass[0, PFT14, ISAPBELOW, ICARBON] = 20.0
    biomass[0, PFT14, IHEARTABOVE, ICARBON] = 30.0
    biomass[0, PFT14, IHEARTBELOW, ICARBON] = 40.0
    biomass[0, PFT14, IAGRSAPST, ICARBON] = 5.0
    biomass[0, PFT14, IAGRSAPPN, ICARBON] = 6.0
    biomass[0, PFT14, IAGRHRTST, ICARBON] = 7.0
    biomass[0, PFT14, IAGRHRTPN, ICARBON] = 8.0
    ind = np.ones((1, 14), dtype=np.float64) * 2.0
    veget_max = np.ones((1, 14), dtype=np.float64) * 0.5

    static = np.asarray(crown_woodmass_ind(biomass, ind, veget_max, ok_dgvm=False))
    dgvm = np.asarray(crown_woodmass_ind(biomass, ind, veget_max, ok_dgvm=True))

    wood = 10.0 + 20.0 + 30.0 + 40.0 + 5.0 + 6.0 + 7.0 + 8.0
    assert np.allclose(static[0, PFT14], wood / 2.0)
    assert np.allclose(dgvm[0, PFT14], wood * 0.5 / 2.0)


def test_crown_step_matches_tree_pipe_model_and_does_not_update_vegetmax():
    pft_present = np.zeros((1, 14), dtype=bool)
    pft_present[0, PFT14] = True
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    woodmass_ind = np.zeros((1, 14), dtype=np.float64)
    woodmass_ind[0, PFT14] = 1.0e5
    veget_max = np.ones((1, 14), dtype=np.float64) * 0.4
    height = np.ones((1, 14), dtype=np.float64) * 3.0
    is_tree = np.zeros(14, dtype=bool)
    is_tree[PFT14] = True
    natural = np.ones(14, dtype=bool)
    maxdia = np.ones(14, dtype=np.float64) * 10.0

    result = crown_step(
        pft_present=pft_present,
        ind=np.ones((1, 14), dtype=np.float64),
        biomass=biomass,
        woodmass_ind=woodmass_ind,
        veget_max=veget_max,
        height=height,
        is_tree=is_tree,
        natural=natural,
        maxdia=maxdia,
    )

    dia = (1.0e5 / (2.0e5 * np.pi / 4.0 * 40.0)) ** (1.0 / 2.5)
    expected_height = 40.0 * dia**0.5
    expected_cn_ind = 100.0 * min(dia, 10.0) ** 1.6
    assert np.allclose(np.asarray(result.height)[0, PFT14], expected_height)
    assert np.allclose(np.asarray(result.cn_ind)[0, PFT14], expected_cn_ind)
    assert np.allclose(np.asarray(result.veget_max), veget_max)
    assert np.allclose(np.asarray(result.cn_ind)[0, 0], 0.0)


def test_crown_step_sets_grass_crown_area_to_one_for_present_pfts():
    pft_present = np.zeros((1, 14), dtype=bool)
    pft_present[0, PFT14] = True
    is_tree = np.zeros(14, dtype=bool)

    result = crown_step(
        pft_present=pft_present,
        ind=np.ones((1, 14), dtype=np.float64),
        biomass=np.zeros((1, 14, NPARTS, 1), dtype=np.float64),
        woodmass_ind=np.zeros((1, 14), dtype=np.float64),
        veget_max=np.zeros((1, 14), dtype=np.float64),
        height=np.ones((1, 14), dtype=np.float64),
        is_tree=is_tree,
        natural=np.ones(14, dtype=bool),
        maxdia=np.ones(14, dtype=np.float64),
    )

    assert np.allclose(np.asarray(result.cn_ind)[0, PFT14], 1.0)
    assert np.allclose(np.asarray(result.height)[0, PFT14], 1.0)


def test_light_competition_static_mode_reduces_biomass_when_fpc_exceeds_max():
    npts, nvm = 1, 14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 100.0
    ind = np.ones((npts, nvm), dtype=np.float64)
    cn_ind = np.ones((npts, nvm), dtype=np.float64)
    lm_lastyearmax = np.ones((npts, nvm), dtype=np.float64) * 100.0
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    ext_coeff = np.ones(nvm, dtype=np.float64) * 0.5
    fpc_max = np.ones((npts, nvm), dtype=np.float64) * 0.2
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, PFT14] = True
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.ones(nvm, dtype=bool)

    result = light_competition_step(
        veget_max=np.zeros((npts, nvm), dtype=np.float64),
        fpc_max=fpc_max,
        pft_present=pft_present,
        cn_ind=cn_ind,
        lai=np.zeros((npts, nvm), dtype=np.float64),
        maxfpc_lastyear=np.zeros((npts, nvm), dtype=np.float64),
        lm_lastyearmax=lm_lastyearmax,
        ind=ind,
        biomass=biomass,
        veget_lastlight=np.zeros((npts, nvm), dtype=np.float64),
        bm_to_litter=np.zeros_like(biomass),
        mortality=np.zeros((npts, nvm), dtype=np.float64),
        sla_calc=sla_calc,
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        ext_coeff=ext_coeff,
        ok_dgvm=False,
        dt_days=2.0,
        min_cover=0.05,
    )

    fpc = 1.0 * max(1.0 - np.exp(-0.5 * (0.02 * 100.0)), 0.05)
    death = 1.0 - 0.2 / fpc
    assert np.allclose(np.asarray(result.light_death)[0, PFT14], death / 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, ILEAF, ICARBON], 100.0 * (1.0 - death))
    assert np.allclose(np.asarray(result.bm_to_litter)[0, PFT14, ILEAF, ICARBON], 100.0 * death)
    assert np.allclose(np.asarray(result.ind)[0, PFT14], 1.0 - death)


def test_light_competition_dgvm_dense_canopy_updates_light_death_and_veget_lastlight():
    npts, nvm = 1, 14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, 1, ILEAF, ICARBON] = 100.0
    biomass[0, PFT14, ILEAF, ICARBON] = 50.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, 1] = True
    pft_present[0, PFT14] = True
    cn_ind = np.zeros((npts, nvm), dtype=np.float64)
    cn_ind[0, 1] = 0.8
    cn_ind[0, PFT14] = 1.0
    ind = np.zeros((npts, nvm), dtype=np.float64)
    ind[0, 1] = 1.0
    ind[0, PFT14] = 0.4
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[1:5] = True
    lm_lastyearmax = np.ones((npts, nvm), dtype=np.float64) * 100.0
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    ext_coeff = np.ones(nvm, dtype=np.float64) * 0.5
    maxfpc_lastyear = np.zeros((npts, nvm), dtype=np.float64)

    result = light_competition_step(
        veget_max=np.zeros((npts, nvm), dtype=np.float64),
        fpc_max=np.ones((npts, nvm), dtype=np.float64),
        pft_present=pft_present,
        cn_ind=cn_ind,
        lai=np.zeros((npts, nvm), dtype=np.float64),
        maxfpc_lastyear=maxfpc_lastyear,
        lm_lastyearmax=lm_lastyearmax,
        ind=ind,
        biomass=biomass,
        veget_lastlight=np.zeros((npts, nvm), dtype=np.float64),
        bm_to_litter=np.zeros_like(biomass),
        mortality=np.zeros((npts, nvm), dtype=np.float64),
        sla_calc=sla_calc,
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        ext_coeff=ext_coeff,
        ok_dgvm=True,
        fpc_crit=0.95,
        dt_days=1.0,
    )

    assert np.allclose(np.asarray(result.light_death)[0, 1], 0.0)
    assert np.asarray(result.light_death)[0, PFT14] > 0.0
    assert np.allclose(np.asarray(result.biomass)[0, 1, ILEAF, ICARBON], 100.0)
    assert np.asarray(result.biomass)[0, PFT14, ILEAF, ICARBON] < 50.0
    expected_tree_lastlight = np.asarray(result.ind)[0, 1] * 0.8 * max(1.0 - np.exp(-100.0 * 0.02 * 0.5), 0.05)
    expected_grass_lastlight = np.asarray(result.ind)[0, PFT14] * (1.0 - np.exp(-100.0 * 0.02 * 0.5))
    assert np.allclose(np.asarray(result.veget_lastlight)[0, 1], expected_tree_lastlight)
    assert np.allclose(np.asarray(result.veget_lastlight)[0, PFT14], expected_grass_lastlight)


def test_light_competition_dgvm_reduces_trees_when_wood_fpc_exceeds_critical():
    npts, nvm = 1, 14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, 1, ILEAF, ICARBON] = 100.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, 1] = True
    cn_ind = np.zeros((npts, nvm), dtype=np.float64)
    cn_ind[0, 1] = 1.2
    ind = np.zeros((npts, nvm), dtype=np.float64)
    ind[0, 1] = 1.0
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[1:5] = True
    lm_lastyearmax = np.ones((npts, nvm), dtype=np.float64) * 100.0
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    ext_coeff = np.ones(nvm, dtype=np.float64) * 0.5

    result = light_competition_step(
        veget_max=np.zeros((npts, nvm), dtype=np.float64),
        fpc_max=np.ones((npts, nvm), dtype=np.float64),
        pft_present=pft_present,
        cn_ind=cn_ind,
        lai=np.zeros((npts, nvm), dtype=np.float64),
        maxfpc_lastyear=np.zeros((npts, nvm), dtype=np.float64),
        lm_lastyearmax=lm_lastyearmax,
        ind=ind,
        biomass=biomass,
        veget_lastlight=np.zeros((npts, nvm), dtype=np.float64),
        bm_to_litter=np.zeros_like(biomass),
        mortality=np.ones((npts, nvm), dtype=np.float64) * 0.25,
        sla_calc=sla_calc,
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        ext_coeff=ext_coeff,
        ok_dgvm=True,
        fpc_crit=0.95,
        dt_days=5.0,
    )

    loss = 1.0 - 0.95 / 1.2
    assert np.allclose(np.asarray(result.light_death)[0, 1], loss / 5.0)
    assert np.allclose(np.asarray(result.ind)[0, 1], 1.0 - loss)
    assert np.allclose(np.asarray(result.biomass)[0, 1, ILEAF, ICARBON], 100.0 * (1.0 - loss))
    assert np.allclose(np.asarray(result.bm_to_litter)[0, 1, ILEAF, ICARBON], 100.0 * loss)
    assert np.allclose(np.asarray(result.mortality)[0, 1], 0.25)


def test_establishment_rates_dgvm_tree_matches_climate_space_formula():
    npts, nvm = 1, 14
    pft = 1
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[pft] = True
    cn_ind = np.zeros((npts, nvm), dtype=np.float64)
    cn_ind[0, pft] = 0.2
    ind = np.zeros((npts, nvm), dtype=np.float64)
    ind[0, pft] = 1.0

    result = establishment_rates_step(
        veget_max=np.zeros((npts, nvm), dtype=np.float64),
        pft_present=pft_present,
        regenerate=np.ones((npts, nvm), dtype=np.float64),
        neighbours=np.zeros((npts, 8), dtype=np.int32),
        resolution=np.ones((npts, 2), dtype=np.float64),
        need_adjacent=np.zeros((npts, nvm), dtype=bool),
        herbivores=np.ones((npts, nvm), dtype=np.float64) * 1.0e6,
        precip_annual=np.array([200.0], dtype=np.float64),
        gdd0=np.array([200.0], dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        cn_ind=cn_ind,
        lai=np.ones((npts, nvm), dtype=np.float64),
        avail_tree=np.array([0.5], dtype=np.float64),
        avail_grass=np.array([0.0], dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        ind=ind,
        everywhere=np.ones((npts, nvm), dtype=np.float64),
        sla_calc=np.ones((npts, nvm), dtype=np.float64),
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        ext_coeff=np.ones(nvm, dtype=np.float64),
        ok_dgvm=True,
        dt_days=365.0,
    )

    fpc = 0.2 * (1.0 - np.exp(-1.0))
    factor = (1.0 - np.exp(-5.0 * (1.0 - fpc))) * (1.0 - fpc)
    estab_tree = 0.12 * factor
    expected = estab_tree * 1.0 * 0.5
    assert np.allclose(np.asarray(result.fpc_nat)[0, pft], fpc)
    assert np.allclose(np.asarray(result.estab_rate_max_tree)[0], estab_tree)
    assert np.allclose(np.asarray(result.d_ind)[0, pft], expected)
    assert np.allclose(np.asarray(result.ind_estab)[0, pft], expected / 365.0)


def test_establishment_rates_dgvm_grass_uses_productivity_and_spacefight():
    npts, nvm = 1, 14
    grass_a, grass_b = 12, PFT14
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, grass_a] = True
    pft_present[0, grass_b] = True
    natural = np.zeros(nvm, dtype=bool)
    natural[grass_a] = True
    natural[grass_b] = True
    is_tree = np.zeros(nvm, dtype=bool)
    everywhere = np.zeros((npts, nvm), dtype=np.float64)
    everywhere[0, grass_a] = 0.25
    everywhere[0, grass_b] = 0.75
    npp_longterm = np.ones((npts, nvm), dtype=np.float64)
    npp_longterm[0, grass_a] = 2.0
    npp_longterm[0, grass_b] = 6.0

    result = establishment_rates_step(
        veget_max=np.zeros((npts, nvm), dtype=np.float64),
        pft_present=pft_present,
        regenerate=np.ones((npts, nvm), dtype=np.float64),
        neighbours=np.zeros((npts, 8), dtype=np.int32),
        resolution=np.ones((npts, 2), dtype=np.float64),
        need_adjacent=np.zeros((npts, nvm), dtype=bool),
        herbivores=np.ones((npts, nvm), dtype=np.float64) * 1.0e6,
        precip_annual=np.array([200.0], dtype=np.float64),
        gdd0=np.array([200.0], dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        cn_ind=np.zeros((npts, nvm), dtype=np.float64),
        lai=np.ones((npts, nvm), dtype=np.float64),
        avail_tree=np.array([0.0], dtype=np.float64),
        avail_grass=np.array([1.0], dtype=np.float64),
        npp_longterm=npp_longterm,
        ind=np.zeros((npts, nvm), dtype=np.float64),
        everywhere=everywhere,
        sla_calc=np.ones((npts, nvm), dtype=np.float64),
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        ext_coeff=np.ones(nvm, dtype=np.float64),
        ok_dgvm=True,
        dt_days=365.0,
        min_stomate=0.0,
    )

    factor = 2.0 + 6.0
    estab_grass = 0.12
    expected_a = estab_grass * 0.25 / 1.0 * (2.0 / factor)
    expected_b = estab_grass * 0.75 / 1.0 * (6.0 / factor)
    assert np.allclose(np.asarray(result.spacefight_grass)[0], 1.0)
    assert np.allclose(np.asarray(result.estab_rate_max_grass)[0], estab_grass)
    assert np.allclose(np.asarray(result.d_ind)[0, grass_a], expected_a)
    assert np.allclose(np.asarray(result.d_ind)[0, grass_b], expected_b)


def test_establishment_rates_static_zero_ind_uses_initial_density():
    npts, nvm = 1, 14
    pft = PFT14
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)

    result = establishment_rates_step(
        veget_max=np.eye(1, nvm, pft, dtype=np.float64),
        pft_present=np.eye(1, nvm, pft, dtype=bool),
        regenerate=np.ones((npts, nvm), dtype=np.float64),
        neighbours=np.zeros((npts, 8), dtype=np.int32),
        resolution=np.ones((npts, 2), dtype=np.float64),
        need_adjacent=np.zeros((npts, nvm), dtype=bool),
        herbivores=np.ones((npts, nvm), dtype=np.float64) * 1.0e6,
        precip_annual=np.array([0.0], dtype=np.float64),
        gdd0=np.array([0.0], dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        cn_ind=np.ones((npts, nvm), dtype=np.float64),
        lai=np.ones((npts, nvm), dtype=np.float64),
        avail_tree=np.zeros(npts, dtype=np.float64),
        avail_grass=np.zeros(npts, dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        ind=np.zeros((npts, nvm), dtype=np.float64),
        everywhere=np.ones((npts, nvm), dtype=np.float64),
        sla_calc=np.ones((npts, nvm), dtype=np.float64),
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        ext_coeff=np.ones(nvm, dtype=np.float64),
        ok_dgvm=False,
        dt_days=1.0,
        ind_0_estab=0.2,
    )

    assert np.allclose(np.asarray(result.d_ind)[0, pft], 0.2)


def test_establishment_rates_herbivores_reduce_d_ind_after_rate_calculation():
    npts, nvm = 1, 14
    pft = PFT14
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, pft] = 1.0

    result = establishment_rates_step(
        veget_max=veget_max,
        pft_present=np.eye(1, nvm, pft, dtype=bool),
        regenerate=np.ones((npts, nvm), dtype=np.float64),
        neighbours=np.zeros((npts, 8), dtype=np.int32),
        resolution=np.ones((npts, 2), dtype=np.float64),
        need_adjacent=np.zeros((npts, nvm), dtype=bool),
        herbivores=np.ones((npts, nvm), dtype=np.float64) * 365.0,
        precip_annual=np.array([0.0], dtype=np.float64),
        gdd0=np.array([0.0], dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        cn_ind=np.ones((npts, nvm), dtype=np.float64),
        lai=np.ones((npts, nvm), dtype=np.float64),
        avail_tree=np.zeros(npts, dtype=np.float64),
        avail_grass=np.zeros(npts, dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        ind=np.zeros((npts, nvm), dtype=np.float64),
        everywhere=np.ones((npts, nvm), dtype=np.float64),
        sla_calc=np.ones((npts, nvm), dtype=np.float64),
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        ext_coeff=np.ones(nvm, dtype=np.float64),
        ok_dgvm=False,
        ok_herbivores=True,
        dt_days=1.0,
        ind_0_estab=0.2,
    )

    assert np.allclose(np.asarray(result.d_ind)[0, pft], 0.2 * np.exp(-0.5))


def test_prescribe_step_cold_start_grass_initializes_reserve_and_presence():
    npts, nvm = 1, 14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.6
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[PFT14, ICARBRES, ICARBON] = 5.0
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.ones(nvm, dtype=bool)

    result = prescribe_step(
        veget_max=veget_max,
        dt_days=2.0,
        pft_present=np.zeros((npts, nvm), dtype=bool),
        everywhere=np.zeros((npts, nvm), dtype=np.float64),
        when_growthinit=np.zeros((npts, nvm), dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        ind=np.zeros((npts, nvm), dtype=np.float64),
        cn_ind=np.zeros((npts, nvm), dtype=np.float64),
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        bm_sapl=bm_sapl,
        maxdia=np.ones(nvm, dtype=np.float64),
        pheno_is_none=np.ones(nvm, dtype=bool),
        ok_dgvm=False,
        lpj_gap_const_mort=True,
        firstcall=True,
        stomate_restart_none=True,
    )

    assert np.allclose(np.asarray(result.cn_ind)[0, PFT14], 1.0)
    assert np.allclose(np.asarray(result.ind)[0, PFT14], 0.6)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, ICARBRES, ICARBON], 5.0)
    assert np.allclose(np.asarray(result.leaf_frac)[0, PFT14], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(result.when_growthinit)[0, PFT14], 1.0e33)
    assert np.allclose(np.asarray(result.co2_to_bm)[0, PFT14], 2.5)
    assert bool(np.asarray(result.pft_present)[0, PFT14])
    assert np.allclose(np.asarray(result.everywhere)[0, PFT14], 1.0)


def test_prescribe_step_tree_pipe_density_and_seasonal_sapling_initialization():
    npts, nvm, pft = 2, 14, 1
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[:, pft] = [0.4, 0.2]
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[pft, ILEAF, ICARBON] = 1.0
    bm_sapl[pft, ISAPABOVE, ICARBON] = 2.0
    bm_sapl[pft, ISAPBELOW, ICARBON] = 1.0
    bm_sapl[pft, ICARBRES, ICARBON] = 4.0
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[pft] = True
    natural = np.ones(nvm, dtype=bool)
    maxdia = np.ones(nvm, dtype=np.float64)
    maxdia[pft] = 0.2

    result = prescribe_step(
        veget_max=veget_max,
        dt_days=4.0,
        pft_present=np.zeros((npts, nvm), dtype=bool),
        everywhere=np.zeros((npts, nvm), dtype=np.float64),
        when_growthinit=np.zeros((npts, nvm), dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        ind=np.zeros((npts, nvm), dtype=np.float64),
        cn_ind=np.zeros((npts, nvm), dtype=np.float64),
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        bm_sapl=bm_sapl,
        maxdia=maxdia,
        pheno_is_none=np.zeros(nvm, dtype=bool),
        ok_dgvm=False,
        lpj_gap_const_mort=True,
        firstcall=True,
        stomate_restart_none=True,
        bm_sapl_rescale=2.0,
        pipe_tune1=100.0,
    )

    expected_cn = 100.0 * 0.2**1.6
    expected_ind = 0.4 / expected_cn
    expected_scale = 2.0 * expected_ind / 0.4
    expected_total = (2.0 + 1.0 + 4.0) * expected_scale
    assert np.allclose(np.asarray(result.cn_ind)[0, pft], expected_cn)
    assert np.allclose(np.asarray(result.ind)[0, pft], expected_ind)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ISAPABOVE, ICARBON], 2.0 * expected_scale)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ISAPBELOW, ICARBON], 1.0 * expected_scale)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 4.0 * expected_scale)
    assert np.allclose(np.asarray(result.leaf_frac)[0, pft], [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(result.co2_to_bm)[0, pft], expected_total / 4.0)
    assert bool(np.asarray(result.pft_present)[0, pft])
    assert np.allclose(np.asarray(result.everywhere)[0, pft], 1.0)
    np.testing.assert_allclose(
        np.asarray(result.biomass)[1, pft], np.asarray(result.biomass)[0, pft]
    )


def test_prescribe_step_firstcall_preserves_nonempty_tree_leaf_fractions():
    npts, nvm, pft = 1, 14, PFT14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 2.0
    biomass[0, pft, ISAPABOVE, ICARBON] = 5.0
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_frac[0, pft] = [0.7, 0.2, 0.1, 0.0]
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, pft] = 0.6
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[pft] = True

    result = prescribe_step(
        veget_max=veget_max,
        dt_days=1.0,
        pft_present=np.ones((npts, nvm), dtype=bool),
        everywhere=np.ones((npts, nvm), dtype=np.float64),
        when_growthinit=np.ones((npts, nvm), dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        ind=np.ones((npts, nvm), dtype=np.float64),
        cn_ind=np.ones((npts, nvm), dtype=np.float64),
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        natural=np.ones(nvm, dtype=bool),
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        bm_sapl=np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        maxdia=np.ones(nvm, dtype=np.float64),
        pheno_is_none=np.ones(nvm, dtype=bool),
        ok_dgvm=False,
        lpj_gap_const_mort=True,
        firstcall=True,
        stomate_restart_none=True,
    )

    np.testing.assert_allclose(np.asarray(result.leaf_frac)[0, pft], leaf_frac[0, pft])


def _pftinout_base_inputs(npts=1, nvm=14):
    return {
        "adapted": np.ones((npts, nvm), dtype=np.float64),
        "regenerate": np.ones((npts, nvm), dtype=np.float64),
        "neighbours": np.zeros((npts, 8), dtype=np.int32),
        "veget_max": np.zeros((npts, nvm), dtype=np.float64),
        "biomass": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "cn_ind": np.ones((npts, nvm), dtype=np.float64),
        "age": np.ones((npts, nvm), dtype=np.float64) * 7.0,
        "leaf_frac": np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        "npp_longterm": np.zeros((npts, nvm), dtype=np.float64),
        "lm_lastyearmax": np.zeros((npts, nvm), dtype=np.float64),
        "senescence": np.ones((npts, nvm), dtype=bool),
        "pft_present": np.zeros((npts, nvm), dtype=bool),
        "everywhere": np.zeros((npts, nvm), dtype=np.float64),
        "when_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "need_adjacent": np.zeros((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64) * 2.0,
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "sla_calc": np.ones((npts, nvm), dtype=np.float64) * 0.02,
        "bm_sapl": np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        "natural": np.ones(nvm, dtype=bool),
        "pasture": np.zeros(nvm, dtype=bool),
        "is_tree": np.zeros(nvm, dtype=bool),
        "ext_coeff": np.ones(nvm, dtype=np.float64),
    }


def test_pftinout_agricultural_grass_initializes_prescribed_absent_pft():
    npts, nvm, pft = 1, 14, PFT14
    params = _pftinout_base_inputs(npts, nvm)
    params["natural"][pft] = False
    params["veget_max"][0, pft] = 0.4
    params["bm_sapl"][pft, ILEAF, ICARBON] = 2.0
    params["bm_sapl"][pft, ICARBRES, ICARBON] = 3.0

    result = pftinout_step(**params, dt_days=5.0, min_stomate=1.0e-8)

    assert bool(np.asarray(result.pft_present)[0, pft])
    assert np.allclose(np.asarray(result.ind)[0, pft], 0.4)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 3.0)
    assert np.allclose(np.asarray(result.co2_to_bm)[0, pft], 1.0)
    assert np.allclose(np.asarray(result.everywhere)[0, pft], 1.0)
    assert not bool(np.asarray(result.senescence)[0, pft])
    assert np.allclose(np.asarray(result.age)[0, pft], 0.0)


def test_pftinout_natural_pft_introduction_respects_adjacent_and_rip_gates():
    npts, nvm, pft = 3, 14, 1
    params = _pftinout_base_inputs(npts, nvm)
    params["is_tree"][pft] = True
    params["need_adjacent"][:, pft] = True
    params["neighbours"][0, 0] = 2
    params["neighbours"][1, 0] = 0
    params["neighbours"][2, 0] = 2
    params["everywhere"][1, pft] = 1.0
    params["rip_time"][2, pft] = 0.1
    params["bm_sapl"][pft, ILEAF, ICARBON] = 4.0
    params["bm_sapl"][pft, IROOT, ICARBON] = 6.0

    result = pftinout_step(
        **params,
        dt_days=365.0,
        ind_0=0.02,
        fpc_crit=0.95,
        min_avail=0.01,
        min_stomate=1.0e-8,
        treat_expansion=True,
        everywhere_init=0.05,
    )

    assert bool(np.asarray(result.can_introduce)[0, pft])
    assert not bool(np.asarray(result.can_introduce)[1, pft])
    assert not bool(np.asarray(result.can_introduce)[2, pft])
    assert bool(np.asarray(result.pft_present)[0, pft])
    assert np.allclose(np.asarray(result.ind)[0, pft], 0.02 * 0.95)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 4.0 * 0.02 * 0.95)
    assert np.allclose(np.asarray(result.co2_to_bm)[0, pft], (4.0 + 6.0) * 0.02 * 0.95 / 365.0)
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 1.0e33)
    assert np.allclose(np.asarray(result.everywhere)[0, pft], 0.05)
    assert not bool(np.asarray(result.need_adjacent)[0, pft])
    assert bool(np.asarray(result.need_adjacent)[1, pft])
    assert bool(np.asarray(result.need_adjacent)[2, pft])
    assert np.allclose(np.asarray(result.lm_lastyearmax)[0, pft], 4.0 * 0.02 * 0.95)
    assert np.allclose(np.asarray(result.npp_longterm)[0, pft], 10.0)
    assert np.allclose(np.asarray(result.leaf_frac)[0, pft, 0], 1.0)


def test_pftinout_need_adjacent_restart_roundtrip_preserves_introduction_gate():
    npts, nvm, pft = 2, 14, 1
    params = _pftinout_base_inputs(npts, nvm)
    params["is_tree"][pft] = True
    params["need_adjacent"][:, pft] = True
    params["neighbours"][0, 0] = 2
    params["everywhere"][1, pft] = 1.0
    params["bm_sapl"][pft, ILEAF, ICARBON] = 4.0
    encoded_need_adjacent = encode_restart_pft_bool_field(params["need_adjacent"])
    params["need_adjacent"] = encoded_need_adjacent >= 0.5

    result = pftinout_step(
        **params,
        dt_days=365.0,
        min_stomate=1.0e-8,
    )

    assert bool(np.asarray(result.can_introduce)[0, pft])
    assert bool(np.asarray(result.pft_present)[0, pft])
    assert not bool(np.asarray(result.need_adjacent)[0, pft])
    assert bool(np.asarray(result.need_adjacent)[1, pft])


def test_pftinout_eliminates_unadapted_present_natural_pft_by_zeroing_ind_only():
    npts, nvm, pft = 1, 14, PFT14
    params = _pftinout_base_inputs(npts, nvm)
    params["pft_present"][0, pft] = True
    params["ind"][0, pft] = 7.0
    params["adapted"][0, pft] = 0.1
    params["biomass"][0, pft, ILEAF, ICARBON] = 11.0

    result = pftinout_step(**params, dt_days=10.0, min_stomate=1.0e-8)

    assert np.allclose(np.asarray(result.ind)[0, pft], 0.0)
    assert bool(np.asarray(result.pft_present)[0, pft])
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 11.0)
    assert np.allclose(np.asarray(result.rip_time)[0, pft], 2.0 + 10.0 / 365.0)


def test_constraints_step_no_threshold_sets_adaptation_and_regeneration_memory():
    npts, nvm, pft = 1, 14, PFT14
    result = constraints_step(
        t2m_month=np.array([293.15], dtype=np.float64),
        t2m_min_daily=np.array([280.0], dtype=np.float64),
        when_growthinit=np.zeros((npts, nvm), dtype=np.float64),
        adapted=np.zeros((npts, nvm), dtype=np.float64),
        regenerate=np.zeros((npts, nvm), dtype=np.float64),
        tseason=np.array([293.15], dtype=np.float64),
        natural=np.ones(nvm, dtype=bool),
        is_tree=np.zeros(nvm, dtype=bool),
        is_peat=np.zeros(nvm, dtype=bool),
        pheno_is_none=np.ones(nvm, dtype=bool),
        tmin_crit=np.full(nvm, -9999.0, dtype=np.float64),
        tcm_crit=np.full(nvm, np.inf, dtype=np.float64),
        dt_days=1.0,
    )

    assert np.allclose(np.asarray(result.adapted)[0, pft], 1.0)
    assert np.allclose(np.asarray(result.regenerate)[0, pft], 1.0)
    assert np.allclose(np.asarray(result.adapted)[0, 0], 0.0)


def test_constraints_step_tree_growthinit_and_tseason_zero_then_memory_relaxes():
    npts, nvm, pft = 2, 14, 1
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[pft] = True
    when_growthinit = np.zeros((npts, nvm), dtype=np.float64)
    when_growthinit[0, pft] = 5.0 * 365.0 + 1.0
    when_growthinit[1, pft] = 1.0e33

    result = constraints_step(
        t2m_month=np.full(npts, 293.15, dtype=np.float64),
        t2m_min_daily=np.full(npts, 280.0, dtype=np.float64),
        when_growthinit=when_growthinit,
        adapted=np.ones((npts, nvm), dtype=np.float64),
        regenerate=np.ones((npts, nvm), dtype=np.float64),
        tseason=np.array([293.15, 279.0], dtype=np.float64),
        natural=np.ones(nvm, dtype=bool),
        is_tree=is_tree,
        is_peat=np.zeros(nvm, dtype=bool),
        pheno_is_none=np.zeros(nvm, dtype=bool),
        tmin_crit=np.full(nvm, -10.0, dtype=np.float64),
        tcm_crit=np.full(nvm, np.inf, dtype=np.float64),
        dt_days=1.0,
    )

    expected = 1.0 / 365.0
    assert np.allclose(np.asarray(result.adapted)[:, pft], [expected, expected])


def test_constraints_step_vernalization_sets_then_decays_and_can_kill_adapted():
    npts, nvm, pft = 2, 14, 2
    tcm_crit = np.full(nvm, np.inf, dtype=np.float64)
    tcm_crit[pft] = 280.0
    regenerate = np.zeros((npts, nvm), dtype=np.float64)
    regenerate[1, pft] = np.exp(-5.0) * 0.99

    result = constraints_step(
        t2m_month=np.array([279.0, 285.0], dtype=np.float64),
        t2m_min_daily=np.full(npts, 280.0, dtype=np.float64),
        when_growthinit=np.zeros((npts, nvm), dtype=np.float64),
        adapted=np.ones((npts, nvm), dtype=np.float64),
        regenerate=regenerate,
        tseason=np.full(npts, 293.15, dtype=np.float64),
        natural=np.ones(nvm, dtype=bool),
        is_tree=np.zeros(nvm, dtype=bool),
        is_peat=np.zeros(nvm, dtype=bool),
        pheno_is_none=np.ones(nvm, dtype=bool),
        tmin_crit=np.full(nvm, -9999.0, dtype=np.float64),
        tcm_crit=tcm_crit,
        dt_days=1.0,
    )

    memory = 364.0 / 365.0
    assert np.allclose(np.asarray(result.regenerate)[0, pft], memory)
    assert np.allclose(np.asarray(result.regenerate)[1, pft], np.exp(-5.0) * 0.99 * memory)
    assert np.allclose(np.asarray(result.adapted)[1, pft], 0.0)


def test_constraints_step_non_natural_without_agriculture_or_peat_is_zeroed():
    npts, nvm, pft = 1, 14, PFT14
    natural = np.ones(nvm, dtype=bool)
    natural[pft] = False

    result = constraints_step(
        t2m_month=np.array([293.15], dtype=np.float64),
        t2m_min_daily=np.array([280.0], dtype=np.float64),
        when_growthinit=np.zeros((npts, nvm), dtype=np.float64),
        adapted=np.ones((npts, nvm), dtype=np.float64),
        regenerate=np.ones((npts, nvm), dtype=np.float64),
        tseason=np.array([293.15], dtype=np.float64),
        natural=natural,
        is_tree=np.zeros(nvm, dtype=bool),
        is_peat=np.zeros(nvm, dtype=bool),
        pheno_is_none=np.ones(nvm, dtype=bool),
        tmin_crit=np.full(nvm, -9999.0, dtype=np.float64),
        tcm_crit=np.full(nvm, -9999.0, dtype=np.float64),
        agriculture=False,
    )

    assert np.allclose(np.asarray(result.adapted)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.regenerate)[0, pft], 0.0)


def test_phenology_step_none_model_updates_schedule_without_leaf_onset():
    npts, nvm, pft = 1, 14, PFT14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 5.0
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_frac[0, pft, 0] = 1.0
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 7.0
    when_growthinit = np.zeros((npts, nvm), dtype=np.float64)
    when_growthinit[0, pft] = 350.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=when_growthinit,
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        dt_days=2.0,
    )

    assert np.allclose(np.asarray(result.allow_initpheno)[0, pft], 1.0)
    assert np.allclose(np.asarray(result.allow_initpheno)[0, 0], 0.0)
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 352.0)
    assert not bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.biomass), biomass)
    assert np.allclose(np.asarray(result.leaf_frac), leaf_frac)
    assert np.allclose(np.asarray(result.leaf_age), leaf_age)


def test_phenology_none_step_matches_public_none_model_path():
    npts, nvm, pft = 1, 14, PFT14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 5.0
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_frac[0, pft, 0] = 1.0
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 7.0
    when_growthinit = np.zeros((npts, nvm), dtype=np.float64)
    when_growthinit[0, pft] = 350.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    co2_to_bm = np.zeros((npts, nvm), dtype=np.float64)

    public = phenology_step(
        pft_present=pft_present,
        when_growthinit=when_growthinit,
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=co2_to_bm,
        pheno_model=["none"] * nvm,
        dt_days=2.0,
    )
    numeric = phenology_none_step(
        pft_present=pft_present,
        when_growthinit=when_growthinit,
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=co2_to_bm,
        dt_days=2.0,
    )

    for name in public._fields:
        if getattr(public, name) is None:
            assert getattr(numeric, name) is None
            continue
        np.testing.assert_allclose(np.asarray(getattr(numeric, name)), np.asarray(getattr(public, name)), rtol=0, atol=0)


def test_phenology_step_rejects_active_unimplemented_onset_model_but_allows_inactive_table_entries():
    npts, nvm, pft = 1, 14, PFT14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pheno_model = ["none"] * nvm
    pheno_model[2] = "unknown_onset"
    pheno_model[pft] = "none"

    inactive = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.zeros((npts, nvm), dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
    )
    assert inactive.begin_leaves.shape == (npts, nvm)

    pft_present[0, 2] = True
    with pytest.raises(NotImplementedError, match="PFT3:unknown_onset"):
        phenology_step(
            pft_present=pft_present,
            when_growthinit=np.zeros((npts, nvm), dtype=np.float64),
            biomass=biomass,
            leaf_frac=leaf_frac,
            leaf_age=leaf_age,
            co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
            pheno_model=pheno_model,
        )


def test_phenology_step_hum_onset_uses_lastyear_moisture_threshold():
    """Fortran: stomate_phenology.f90::pheno_hum lines 650-789."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 6.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 9.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "hum"
    daily = np.zeros((npts, nvm), dtype=np.float64)
    maxmoi = daily.copy()
    minmoi = daily.copy()
    month = daily.copy()
    week = daily.copy()
    maxmoi[0, pft] = 0.9
    minmoi[0, pft] = 0.1
    month[0, pft] = 0.35
    week[0, pft] = 0.55
    hum_frac = np.ones(nvm, dtype=np.float64) * 0.5

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        dt_days=1.0,
        maxmoiavail_lastyear=maxmoi,
        minmoiavail_lastyear=minmoi,
        moiavail_month=month,
        moiavail_week=week,
        hum_frac=hum_frac,
        is_tree=np.asarray([False, False, True, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    assert bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.leaf_frac)[0, pft], [1.0, 0.0, 0.0, 0.0])


def test_phenology_step_ngd_onset_uses_temperature_trend_and_ngd_threshold():
    """Fortran: stomate_phenology.f90::pheno_ngd lines 1668-1750."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 4.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 5.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "ngd"
    ngd_minus5 = np.zeros((npts, nvm), dtype=np.float64)
    ngd_minus5[0, pft] = 12.0

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        dt_days=1.0,
        ngd_minus5=ngd_minus5,
        ngd_crit=np.ones(nvm, dtype=np.float64) * 10.0,
        t2m_month=np.asarray([281.0], dtype=np.float64),
        t2m_week=np.asarray([283.0], dtype=np.float64),
        is_tree=np.asarray([False, False, True, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    assert bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.leaf_age)[0, pft], 0.0)


def test_phenology_step_humgdd_combines_gdd_temperature_and_humidity():
    """Fortran: stomate_phenology.f90::pheno_humgdd lines 1029-1197."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 4.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 4.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "humgdd"
    daily = np.zeros((npts, nvm), dtype=np.float64)
    gdd = daily.copy()
    gdd[0, pft] = 60.0
    maxmoi = daily.copy()
    minmoi = daily.copy()
    month = daily.copy()
    week = daily.copy()
    maxmoi[0, pft] = 0.9
    minmoi[0, pft] = 0.1
    month[0, pft] = 0.35
    week[0, pft] = 0.55

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        gdd_m5_dormance=gdd,
        pheno_gdd_crit=np.asarray([[0.0, 0.0, 0.0], [-9999.0, -9999.0, -9999.0], [20.0, 1.0, 0.0], [-9999.0, -9999.0, -9999.0]], dtype=np.float64),
        maxmoiavail_lastyear=maxmoi,
        minmoiavail_lastyear=minmoi,
        moiavail_month=month,
        moiavail_week=week,
        hum_frac=np.ones(nvm, dtype=np.float64) * 0.5,
        t2m_longterm=np.asarray([293.15], dtype=np.float64),
        t2m_month=np.asarray([281.0], dtype=np.float64),
        t2m_week=np.asarray([283.0], dtype=np.float64),
        is_tree=np.asarray([False, False, True, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    assert bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 0.0)


def test_phenology_step_moigdd_respects_optional_c4_temperature_threshold():
    """Fortran: stomate_phenology.f90::pheno_moigdd lines 1261-1456."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 4.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 4.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "moigdd"
    daily = np.zeros((npts, nvm), dtype=np.float64)
    gdd = daily.copy()
    gdd[0, pft] = 60.0
    time_hum_min = daily.copy()
    time_hum_min[0, pft] = 40.0
    month = daily.copy()
    week = daily.copy()
    month[0, pft] = 0.2
    week[0, pft] = 0.4
    common = dict(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        gdd_m5_dormance=gdd,
        pheno_gdd_crit=np.asarray([[0.0, 0.0, 0.0], [-9999.0, -9999.0, -9999.0], [20.0, 1.0, 0.0], [-9999.0, -9999.0, -9999.0]], dtype=np.float64),
        time_hum_min=time_hum_min,
        hum_min_time=np.ones(nvm, dtype=np.float64) * 30.0,
        moiavail_month=month,
        moiavail_week=week,
        t2m_longterm=np.asarray([293.15], dtype=np.float64),
        t2m_month=np.asarray([290.0], dtype=np.float64),
        t2m_week=np.asarray([292.0], dtype=np.float64),
        is_tree=np.asarray([False, False, False, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    blocked = phenology_step(
        **common,
        pheno_moigdd_t_crit=np.asarray([-9999.0, -9999.0, 22.0, -9999.0], dtype=np.float64),
    )
    assert not bool(np.asarray(blocked.begin_leaves)[0, pft])
    np.testing.assert_allclose(np.asarray(blocked.biomass), biomass, rtol=0, atol=0)

    allowed = phenology_step(
        **common,
        pheno_moigdd_t_crit=np.asarray([-9999.0, -9999.0, 15.0, -9999.0], dtype=np.float64),
    )
    assert bool(np.asarray(allowed.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(allowed.biomass)[0, pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(allowed.biomass)[0, pft, IROOT, ICARBON], 2.0)


def test_phenology_step_moigdd_ok_laidev_uses_stics_lai_onset_and_biomass_forcing():
    """Fortran: stomate_phenology.f90::pheno_moigdd lines 1381-1387 and phenology lines 532-548."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 10.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 4.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "moigdd"
    daily = np.zeros((npts, nvm), dtype=np.float64)
    pdlai = daily.copy()
    slai = daily.copy()
    deltai = daily.copy()
    ssla = np.ones((npts, nvm), dtype=np.float64)
    slai[0, pft] = 0.4
    deltai[0, pft] = 0.01
    ssla[0, pft] = 20.0

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        active_pft_mask=np.asarray([False, False, True, False]),
        gdd_m5_dormance=daily,
        pheno_gdd_crit=np.asarray([[0.0, 0.0, 0.0], [-9999.0, -9999.0, -9999.0], [20.0, 1.0, 0.0], [-9999.0, -9999.0, -9999.0]], dtype=np.float64),
        time_hum_min=daily,
        hum_min_time=np.ones(nvm, dtype=np.float64) * 30.0,
        moiavail_month=daily,
        moiavail_week=daily,
        t2m_longterm=np.asarray([273.15], dtype=np.float64),
        t2m_month=np.asarray([273.15], dtype=np.float64),
        t2m_week=np.asarray([273.15], dtype=np.float64),
        pheno_moigdd_t_crit=np.ones(nvm, dtype=np.float64) * -9999.0,
        is_tree=np.asarray([False, False, False, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        ok_laidev=np.asarray([False, False, True, False]),
        pdlai=pdlai,
        slai=slai,
        deltai=deltai,
        ssla=ssla,
    )

    expected_bm_use = 0.01 / 20.0 * 2.0 * 10000.0
    assert bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], expected_bm_use / 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], expected_bm_use / 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 10.0 - expected_bm_use)
    assert np.allclose(np.asarray(result.leaf_frac)[0, pft], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(result.leaf_age)[0, pft], 0.0)


def test_phenology_step_moi_c4_uses_fixed_warm_temperature_threshold():
    """Fortran: stomate_phenology.f90::pheno_moi_C4 lines 1757-1902."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 4.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 4.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "moi_C4"
    daily = np.zeros((npts, nvm), dtype=np.float64)
    gdd = daily.copy()
    gdd[0, pft] = 60.0
    time_hum_min = daily.copy()
    time_hum_min[0, pft] = 40.0
    month = daily.copy()
    week = daily.copy()
    month[0, pft] = 0.2
    week[0, pft] = 0.4
    common = dict(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        gdd_m5_dormance=gdd,
        pheno_gdd_crit=np.asarray([[0.0, 0.0, 0.0], [-9999.0, -9999.0, -9999.0], [20.0, 1.0, 0.0], [-9999.0, -9999.0, -9999.0]], dtype=np.float64),
        time_hum_min=time_hum_min,
        hum_min_time=np.ones(nvm, dtype=np.float64) * 30.0,
        moiavail_month=month,
        moiavail_week=week,
        t2m_longterm=np.asarray([293.15], dtype=np.float64),
        t2m_week=np.asarray([297.0], dtype=np.float64),
        is_tree=np.asarray([False, False, False, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    blocked = phenology_step(**common, t2m_month=np.asarray([294.0], dtype=np.float64))
    assert not bool(np.asarray(blocked.begin_leaves)[0, pft])

    allowed = phenology_step(**common, t2m_month=np.asarray([296.0], dtype=np.float64))
    assert bool(np.asarray(allowed.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(allowed.biomass)[0, pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(allowed.biomass)[0, pft, IROOT, ICARBON], 2.0)


def test_phenology_step_siggdd_uses_fortran_sigmoid_gdd_threshold():
    """Fortran: stomate_phenology.f90::pheno_siggdd lines 1910-2061."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 4.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 4.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "siggdd"
    daily = np.zeros((npts, nvm), dtype=np.float64)
    gdd = daily.copy()
    gdd[0, pft] = 1000.0
    time_hum_min = daily.copy()
    time_hum_min[0, pft] = 40.0
    month = daily.copy()
    week = daily.copy()
    month[0, pft] = 0.2
    week[0, pft] = 0.4

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        gdd_m5_dormance=gdd,
        pheno_gdd_crit=np.asarray([[0.0, 0.0, 0.0], [-9999.0, -9999.0, -9999.0], [20.0, 1.0, 0.0], [-9999.0, -9999.0, -9999.0]], dtype=np.float64),
        time_hum_min=time_hum_min,
        hum_min_time=np.ones(nvm, dtype=np.float64) * 30.0,
        moiavail_month=month,
        moiavail_week=week,
        t2m_longterm=np.asarray([293.15], dtype=np.float64),
        t2m_month=np.asarray([290.0], dtype=np.float64),
        t2m_week=np.asarray([292.0], dtype=np.float64),
        pheno_moigdd_t_crit=np.asarray([-9999.0, -9999.0, -9999.0, -9999.0], dtype=np.float64),
        is_tree=np.asarray([False, False, False, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    assert bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 2.0)


def test_phenology_step_humgdd_rejects_undefined_gdd_coefficients():
    npts, nvm, pft = 1, 4, 2
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "humgdd"
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True

    with pytest.raises(ValueError, match="pheno_gdd_crit is undefined"):
        phenology_step(
            pft_present=pft_present,
            when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
            biomass=np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
            leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
            leaf_age=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
            co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
            pheno_model=pheno_model,
            gdd_m5_dormance=np.zeros((npts, nvm), dtype=np.float64),
            pheno_gdd_crit=np.ones((nvm, 3), dtype=np.float64) * -9999.0,
            maxmoiavail_lastyear=np.zeros((npts, nvm), dtype=np.float64),
            minmoiavail_lastyear=np.zeros((npts, nvm), dtype=np.float64),
            moiavail_month=np.zeros((npts, nvm), dtype=np.float64),
            moiavail_week=np.zeros((npts, nvm), dtype=np.float64),
            hum_frac=np.ones(nvm, dtype=np.float64) * 0.5,
            t2m_longterm=np.asarray([293.15], dtype=np.float64),
            t2m_month=np.asarray([281.0], dtype=np.float64),
            t2m_week=np.asarray([283.0], dtype=np.float64),
            is_tree=np.asarray([False, False, True, False]),
            lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
            sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        )


def test_phenology_step_ncdgdd_onset_resets_gdd_midwinter_and_allocates_reserve():
    """Fortran: stomate_phenology.f90::pheno_ncdgdd lines 1520-1615."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 4.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 6.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "ncdgdd"
    ncd_dormance = np.zeros((npts, nvm), dtype=np.float64)
    ncd_dormance[0, pft] = 100.0
    gdd_midwinter = np.zeros((npts, nvm), dtype=np.float64)
    gdd_midwinter[0, pft] = 200.0

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        dt_days=1.0,
        ncd_dormance=ncd_dormance,
        gdd_midwinter=gdd_midwinter,
        ncdgdd_temp=np.asarray([-9999.0, -9999.0, 5.0, -9999.0], dtype=np.float64),
        t2m_month=np.asarray([281.0], dtype=np.float64),
        t2m_week=np.asarray([283.0], dtype=np.float64),
        is_tree=np.asarray([False, False, True, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    assert bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.leaf_frac)[0, pft], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(result.leaf_age)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.gdd_midwinter)[0, pft], -9999.0)


def test_phenology_step_ncdgdd_below_threshold_preserves_memory_and_schedule():
    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "ncdgdd"
    ncd_dormance = np.zeros((npts, nvm), dtype=np.float64)
    ncd_dormance[0, pft] = 100.0
    gdd_midwinter = np.zeros((npts, nvm), dtype=np.float64)
    gdd_midwinter[0, pft] = 10.0

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        dt_days=2.0,
        ncd_dormance=ncd_dormance,
        gdd_midwinter=gdd_midwinter,
        ncdgdd_temp=np.asarray([-9999.0, -9999.0, 5.0, -9999.0], dtype=np.float64),
        t2m_month=np.asarray([281.0], dtype=np.float64),
        t2m_week=np.asarray([283.0], dtype=np.float64),
        is_tree=np.asarray([False, False, True, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    assert not bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 352.0)
    np.testing.assert_allclose(np.asarray(result.gdd_midwinter), gdd_midwinter, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(result.biomass), biomass, rtol=0, atol=0)


def test_phenology_step_ncdgdd_rejects_undefined_pft_parameter():
    npts, nvm, pft = 1, 4, 2
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "ncdgdd"
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True

    with pytest.raises(ValueError, match="ncdgdd_temp is undefined"):
        phenology_step(
            pft_present=pft_present,
            when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
            biomass=np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
            leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
            leaf_age=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
            co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
            pheno_model=pheno_model,
            ncd_dormance=np.zeros((npts, nvm), dtype=np.float64),
            gdd_midwinter=np.zeros((npts, nvm), dtype=np.float64),
            ncdgdd_temp=np.ones(nvm, dtype=np.float64) * -9999.0,
            t2m_month=np.asarray([281.0], dtype=np.float64),
            t2m_week=np.asarray([283.0], dtype=np.float64),
            is_tree=np.asarray([False, False, True, False]),
            lai_initmin=np.ones(nvm, dtype=np.float64) * 0.4,
            sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        )


def test_phenology_step_moi_onset_allocates_reserve_and_resets_leaf_age():
    """Fortran: stomate_phenology.f90::phenology lines 319-563 and pheno_moi lines 838-958."""

    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 4.0
    biomass[0, pft, ILEAF, ICARBON] = 0.2
    biomass[0, pft, IROOT, ICARBON] = 0.5
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 12.0
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    when_growthinit = np.zeros((npts, nvm), dtype=np.float64)
    when_growthinit[0, pft] = 350.0
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "moi"
    daily = np.zeros((npts, nvm), dtype=np.float64)
    time_hum_min = daily.copy()
    time_hum_min[0, pft] = 45.0
    moiavail_month = daily.copy()
    moiavail_month[0, pft] = 0.2
    moiavail_week = daily.copy()
    moiavail_week[0, pft] = 0.4
    hum_min_time = np.ones(nvm, dtype=np.float64) * 30.0
    lai_initmin = np.ones(nvm, dtype=np.float64) * 0.6
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.2

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=when_growthinit,
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        dt_days=2.0,
        time_hum_min=time_hum_min,
        moiavail_month=moiavail_month,
        moiavail_week=moiavail_week,
        hum_min_time=hum_min_time,
        is_tree=np.asarray([False, False, True, False]),
        lai_initmin=lai_initmin,
        sla_calc=sla_calc,
    )

    assert bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.when_growthinit)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 2.2)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 2.5)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.leaf_frac)[0, pft], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(result.leaf_age)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.co2_to_bm), 0.0)


def test_phenology_step_moi_always_init_fills_reserve_from_atmosphere():
    npts, nvm, pft = 1, 4, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ICARBRES, ICARBON] = 1.0
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, pft] = True
    pheno_model = ["none"] * nvm
    pheno_model[pft] = "moi"
    daily = np.zeros((npts, nvm), dtype=np.float64)

    result = phenology_step(
        pft_present=pft_present,
        when_growthinit=np.asarray([[0.0, 0.0, 350.0, 0.0]], dtype=np.float64),
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        pheno_model=pheno_model,
        dt_days=2.0,
        time_hum_min=daily,
        moiavail_month=np.asarray([[0.0, 0.0, 1.1, 0.0]], dtype=np.float64),
        moiavail_week=daily,
        hum_min_time=np.ones(nvm, dtype=np.float64) * 30.0,
        is_tree=np.asarray([False, False, True, False]),
        lai_initmin=np.ones(nvm, dtype=np.float64) * 0.6,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        always_init=True,
    )

    assert bool(np.asarray(result.begin_leaves)[0, pft])
    assert np.allclose(np.asarray(result.co2_to_bm)[0, pft], 2.5)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 3.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 3.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ICARBRES, ICARBON], 0.0)


def test_establishment_biomass_first_establishment_adds_sapling_and_updates_leaf_state():
    npts, nvm = 1, 14
    pft = PFT14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_age[0, pft, 0] = 10.0
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_frac[0, pft, 0] = 1.0
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[pft, ILEAF, ICARBON] = 2.0
    bm_sapl[pft, IROOT, ICARBON] = 3.0
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, pft] = 0.5
    d_ind = np.zeros((npts, nvm), dtype=np.float64)
    d_ind[0, pft] = 0.1
    natural = np.ones(nvm, dtype=bool)

    result = establishment_biomass_step(
        d_ind=d_ind,
        ind=np.zeros((npts, nvm), dtype=np.float64),
        biomass=biomass,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=np.ones((npts, nvm), dtype=np.float64) * 5.0,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        veget_max=veget_max,
        woodmass_ind=np.zeros((npts, nvm), dtype=np.float64),
        mortality=np.zeros((npts, nvm), dtype=np.float64),
        bm_to_litter=np.zeros_like(biomass),
        bm_sapl=bm_sapl,
        npp_longterm=np.ones((npts, nvm), dtype=np.float64) * 365.0,
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=np.zeros(nvm, dtype=bool),
        maxdia=np.ones(nvm, dtype=np.float64),
        dt_days=2.0,
        ok_dgvm=False,
    )

    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 0.4)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 0.6)
    assert np.allclose(np.asarray(result.co2_to_bm)[0, pft], (0.4 + 0.6) / 2.0)
    assert np.allclose(np.asarray(result.leaf_age)[0, pft, 0], 0.0)
    assert np.allclose(np.asarray(result.leaf_frac)[0, pft, 0], 0.5)
    assert np.allclose(np.asarray(result.ind)[0, pft], 0.1)
    assert np.allclose(np.asarray(result.age)[0, pft], 0.0)


def test_establishment_biomass_existing_population_distributes_saplings_by_old_biomass():
    npts, nvm = 1, 14
    pft = PFT14
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 2.0
    biomass[0, pft, IROOT, ICARBON] = 6.0
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_age[0, pft, 0] = 8.0
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_frac[0, pft, 0] = 0.5
    leaf_frac[0, pft, 1] = 0.5
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[pft, ILEAF, ICARBON] = 2.0
    bm_sapl[pft, IROOT, ICARBON] = 6.0
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, pft] = 0.5
    d_ind = np.zeros((npts, nvm), dtype=np.float64)
    d_ind[0, pft] = 0.25
    ind = np.zeros((npts, nvm), dtype=np.float64)
    ind[0, pft] = 1.0
    natural = np.ones(nvm, dtype=bool)

    result = establishment_biomass_step(
        d_ind=d_ind,
        ind=ind,
        biomass=biomass,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=np.ones((npts, nvm), dtype=np.float64) * 12.0,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        veget_max=veget_max,
        woodmass_ind=np.zeros((npts, nvm), dtype=np.float64),
        mortality=np.zeros((npts, nvm), dtype=np.float64),
        bm_to_litter=np.zeros_like(biomass),
        bm_sapl=bm_sapl,
        npp_longterm=np.ones((npts, nvm), dtype=np.float64) * 365.0,
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=np.zeros(nvm, dtype=bool),
        maxdia=np.ones(nvm, dtype=np.float64),
        dt_days=1.0,
        ok_dgvm=False,
    )

    total_sapling = (2.0 + 6.0) * 0.25 / 0.5
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], 2.0 + total_sapling * 2.0 / 8.0)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IROOT, ICARBON], 6.0 + total_sapling * 6.0 / 8.0)
    assert np.allclose(np.asarray(result.co2_to_bm)[0, pft], total_sapling)
    assert np.allclose(np.asarray(result.ind)[0, pft], 1.25)
    assert np.allclose(np.asarray(result.age)[0, pft], 12.0 / 1.25)
    leaf_add = 0.25 * 2.0
    expected_leaf_age0 = 8.0 * (0.5 * 2.0) / ((0.5 * 2.0) + leaf_add)
    assert np.allclose(np.asarray(result.leaf_age)[0, pft, 0], expected_leaf_age0)


def test_establishment_biomass_tree_updates_woodmass_ind_static_and_dgvm_forms():
    npts, nvm = 1, 14
    pft = 1
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ISAPABOVE, ICARBON] = 10.0
    biomass[0, pft, ISAPBELOW, ICARBON] = 5.0
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[pft, ISAPABOVE, ICARBON] = 2.0
    bm_sapl[pft, ISAPBELOW, ICARBON] = 1.0
    ind = np.zeros((npts, nvm), dtype=np.float64)
    ind[0, pft] = 1.0
    d_ind = np.zeros((npts, nvm), dtype=np.float64)
    d_ind[0, pft] = 0.5
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, pft] = 0.4
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[pft] = True

    kwargs = dict(
        d_ind=d_ind,
        ind=ind,
        biomass=biomass,
        leaf_age=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        age=np.ones((npts, nvm), dtype=np.float64),
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        veget_max=veget_max,
        woodmass_ind=np.zeros((npts, nvm), dtype=np.float64),
        mortality=np.zeros((npts, nvm), dtype=np.float64),
        bm_to_litter=np.zeros_like(biomass),
        bm_sapl=bm_sapl,
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        natural=natural,
        pasture=np.zeros(nvm, dtype=bool),
        is_tree=is_tree,
        maxdia=np.ones(nvm, dtype=np.float64),
    )
    static = establishment_biomass_step(**kwargs, ok_dgvm=False)
    dgvm = establishment_biomass_step(**kwargs, ok_dgvm=True)

    assert np.allclose(np.asarray(static.woodmass_ind)[0, pft], 15.0 / 1.5)
    assert np.allclose(np.asarray(dgvm.woodmass_ind)[0, pft], 15.0 * 0.4 / 1.5)


def _cover_fixture(npts=1, nvm=14):
    return {
        "cn_ind": np.zeros((npts, nvm), dtype=np.float64),
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "biomass": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "veget_max": np.zeros((npts, nvm), dtype=np.float64),
        "veget_max_old": np.zeros((npts, nvm), dtype=np.float64),
        "lai": np.zeros((npts, nvm), dtype=np.float64),
        "litter": np.zeros((npts, NLITT, nvm, NLEVS, 1), dtype=np.float64),
        "litter_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "litter_not_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "carbon": np.zeros((npts, NCARB, nvm), dtype=np.float64),
        "fuel_1hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "fuel_10hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "fuel_100hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "fuel_1000hr": np.zeros((npts, nvm, NLITT, 1), dtype=np.float64),
        "turnover_daily": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "bm_to_litter": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "co2_fire": np.zeros((npts, nvm), dtype=np.float64),
        "resp_hetero": np.zeros((npts, nvm), dtype=np.float64),
        "resp_maint": np.zeros((npts, nvm), dtype=np.float64),
        "resp_growth": np.zeros((npts, nvm), dtype=np.float64),
        "gpp_daily": np.zeros((npts, nvm), dtype=np.float64),
        "deepC_a": np.zeros((npts, 2, nvm), dtype=np.float64),
        "deepC_s": np.zeros((npts, 2, nvm), dtype=np.float64),
        "deepC_p": np.zeros((npts, 2, nvm), dtype=np.float64),
        "natural": np.ones(nvm, dtype=bool),
        "pasture": np.zeros(nvm, dtype=bool),
        "is_peat": np.zeros(nvm, dtype=bool),
    }


def test_cover_step_dgvm_recomputes_natural_cover_and_bare_soil():
    params = _cover_fixture()
    params["cn_ind"][0, 1] = 0.3
    params["ind"][0, 1] = 1.0
    params["cn_ind"][0, PFT14] = 0.4
    params["ind"][0, PFT14] = 0.5
    params["pasture"][2] = True
    params["veget_max"][0, 2] = 0.1
    params["veget_max_old"] = params["veget_max"].copy()

    result = cover_step(**params, ok_dgvm=True)

    assert np.allclose(np.asarray(result.veget_max)[0, 1], 0.3)
    assert np.allclose(np.asarray(result.veget_max)[0, PFT14], 0.2)
    assert np.allclose(np.asarray(result.veget_max)[0, 0], 0.4)


def test_cover_step_dgvm_scales_natural_cover_when_non_natural_limits_space():
    params = _cover_fixture()
    params["cn_ind"][0, 1] = 0.8
    params["ind"][0, 1] = 1.0
    params["cn_ind"][0, PFT14] = 0.4
    params["ind"][0, PFT14] = 1.0
    params["pasture"][2] = True
    params["veget_max"][0, 2] = 0.2
    params["veget_max_old"] = params["veget_max"].copy()

    result = cover_step(**params, ok_dgvm=True)

    assert np.allclose(np.asarray(result.veget_max)[0, 1], 0.8 * 0.8 / 1.2)
    assert np.allclose(np.asarray(result.veget_max)[0, PFT14], 0.4 * 0.8 / 1.2)
    assert np.allclose(np.asarray(result.veget_max)[0, 0], 0.0)


def test_cover_step_dgvm_dilutes_pools_from_shrinking_to_expanding_pft():
    params = _cover_fixture()
    shrink, expand = 1, PFT14
    params["natural"][:] = False
    params["natural"][shrink] = True
    params["natural"][expand] = True
    params["veget_max_old"][0, shrink] = 0.4
    params["veget_max_old"][0, expand] = 0.2
    params["cn_ind"][0, shrink] = 0.2
    params["ind"][0, shrink] = 1.0
    params["cn_ind"][0, expand] = 0.4
    params["ind"][0, expand] = 1.0
    params["biomass"][0, shrink, ILEAF, ICARBON] = 10.0
    params["biomass"][0, expand, ILEAF, ICARBON] = 20.0
    params["litter"][0, IMETABOLIC, shrink, IABOVE, ICARBON] = 12.0
    params["litter"][0, IMETABOLIC, expand, IABOVE, ICARBON] = 4.0
    params["carbon"][0, IACTIVE, shrink] = 30.0
    params["carbon"][0, IACTIVE, expand] = 10.0
    params["turnover_daily"][0, shrink, ILEAF, ICARBON] = 6.0
    params["turnover_daily"][0, expand, ILEAF, ICARBON] = 2.0
    params["bm_to_litter"][0, shrink, ILEAF, ICARBON] = 8.0
    params["bm_to_litter"][0, expand, ILEAF, ICARBON] = 4.0
    params["gpp_daily"][0, shrink] = 5.0
    params["gpp_daily"][0, expand] = 1.0
    params["resp_maint"][0, shrink] = 3.0
    params["resp_maint"][0, expand] = 1.0
    params["co2_to_bm"][0, shrink] = 2.0
    params["co2_to_bm"][0, expand] = 0.5

    result = cover_step(**params, ok_dgvm=True)

    assert np.allclose(np.asarray(result.veget_max)[0, shrink], 0.2)
    assert np.allclose(np.asarray(result.veget_max)[0, expand], 0.4)
    assert np.allclose(np.asarray(result.biomass)[0, shrink, ILEAF, ICARBON], 20.0)
    assert np.allclose(np.asarray(result.biomass)[0, expand, ILEAF, ICARBON], 10.0)
    assert np.allclose(np.asarray(result.litter)[0, IMETABOLIC, expand, IABOVE, ICARBON], (4.0 * 0.2 + 12.0 * 0.2) / 0.4)
    assert np.allclose(np.asarray(result.carbon)[0, IACTIVE, expand], (10.0 * 0.2 + 30.0 * 0.2) / 0.4)
    assert np.allclose(np.asarray(result.turnover_daily)[0, expand, ILEAF, ICARBON], (2.0 * 0.2 + 6.0 * 0.2) / 0.4)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, expand, ILEAF, ICARBON], (4.0 * 0.2 + 8.0 * 0.2) / 0.4)
    assert np.allclose(np.asarray(result.gpp_daily)[0, expand], (1.0 * 0.2 + 5.0 * 0.2) / 0.4)
    assert np.allclose(np.asarray(result.resp_maint)[0, expand], (1.0 * 0.2 + 3.0 * 0.2) / 0.4)
    assert np.allclose(np.asarray(result.co2_to_bm)[0, expand], (0.5 * 0.2 + 2.0 * 0.2) / 0.4)


def test_dgvm_minimal_day_sequence_introduces_pft_and_carries_state_to_cover():
    npts, nvm, pft = 3, 14, 1
    natural = np.ones(nvm, dtype=bool)
    pasture = np.zeros(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[pft] = True
    pheno_is_none = np.ones(nvm, dtype=bool)
    pheno_is_none[pft] = False

    when_growthinit = np.zeros((npts, nvm), dtype=np.float64)
    constraints = constraints_step(
        t2m_month=np.full(npts, 293.15, dtype=np.float64),
        t2m_min_daily=np.full(npts, 280.0, dtype=np.float64),
        when_growthinit=when_growthinit,
        adapted=np.zeros((npts, nvm), dtype=np.float64),
        regenerate=np.zeros((npts, nvm), dtype=np.float64),
        tseason=np.full(npts, 293.15, dtype=np.float64),
        natural=natural,
        is_tree=is_tree,
        is_peat=np.zeros(nvm, dtype=bool),
        pheno_is_none=pheno_is_none,
        tmin_crit=np.full(nvm, -9999.0, dtype=np.float64),
        tcm_crit=np.full(nvm, np.inf, dtype=np.float64),
        dt_days=365.0,
    )

    neighbours = np.zeros((npts, 8), dtype=np.int32)
    neighbours[0, 0] = 2
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[1, pft] = True
    everywhere = np.zeros((npts, nvm), dtype=np.float64)
    everywhere[1, pft] = 1.0
    need_adjacent = np.zeros((npts, nvm), dtype=bool)
    need_adjacent[0, pft] = True
    rip_time = np.ones((npts, nvm), dtype=np.float64) * 2.0
    cn_ind = np.zeros((npts, nvm), dtype=np.float64)
    cn_ind[:, pft] = 0.5
    ind = np.zeros((npts, nvm), dtype=np.float64)
    ind[1, pft] = 0.8
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[1, pft, ILEAF, ICARBON] = 5.0
    biomass[1, pft, IROOT, ICARBON] = 8.0
    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    bm_sapl[pft, ILEAF, ICARBON] = 4.0
    bm_sapl[pft, IROOT, ICARBON] = 6.0

    pftinout = pftinout_step(
        adapted=np.asarray(constraints.adapted),
        regenerate=np.asarray(constraints.regenerate),
        neighbours=neighbours,
        veget_max=np.zeros((npts, nvm), dtype=np.float64),
        biomass=biomass,
        ind=ind,
        cn_ind=cn_ind,
        age=np.ones((npts, nvm), dtype=np.float64) * 5.0,
        leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64) * 20.0,
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64) * 10.0,
        senescence=np.ones((npts, nvm), dtype=bool),
        pft_present=pft_present,
        everywhere=everywhere,
        when_growthinit=when_growthinit,
        need_adjacent=need_adjacent,
        rip_time=rip_time,
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.02,
        bm_sapl=bm_sapl,
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        ext_coeff=np.ones(nvm, dtype=np.float64),
        dt_days=365.0,
        min_stomate=1.0e-8,
    )

    assert bool(np.asarray(pftinout.can_introduce)[0, pft])
    assert bool(np.asarray(pftinout.pft_present)[0, pft])
    assert not bool(np.asarray(pftinout.need_adjacent)[0, pft])
    assert np.allclose(np.asarray(pftinout.rip_time)[0, pft], 3.0)
    assert np.asarray(pftinout.co2_to_bm)[0, pft] > 0.0

    gap = gap_mortality_step(
        npp_longterm=np.asarray(pftinout.npp_longterm),
        turnover_longterm=np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        lm_lastyearmax=np.asarray(pftinout.lm_lastyearmax),
        pft_present=np.asarray(pftinout.pft_present),
        biomass=np.asarray(pftinout.biomass),
        ind=np.asarray(pftinout.ind),
        bm_to_litter=np.zeros_like(biomass),
        t2m_min_daily=np.full(npts, 280.0, dtype=np.float64),
        tmin_spring_time=np.zeros((npts, nvm), dtype=np.float64),
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.02,
        natural=natural,
        is_tree=is_tree,
        pasture=pasture,
        availability_fact=np.ones(nvm, dtype=np.float64) * 0.01,
        residence_time=np.ones(nvm, dtype=np.float64) * 1000.0,
        tmin_crit=np.full(nvm, -9999.0, dtype=np.float64),
        leaf_tab=np.zeros(nvm, dtype=np.int32),
        pheno_type=np.zeros(nvm, dtype=np.int32),
        dt_days=1.0,
        ok_dgvm=True,
        min_stomate=1.0e-8,
    )
    kill = kill_pfts_step(
        lm_lastyearmax=np.asarray(pftinout.lm_lastyearmax),
        ind=np.asarray(gap.ind),
        pft_present=np.asarray(pftinout.pft_present),
        cn_ind=cn_ind,
        biomass=np.asarray(gap.biomass),
        senescence=np.asarray(pftinout.senescence),
        rip_time=np.asarray(pftinout.rip_time),
        age=np.asarray(pftinout.age),
        leaf_age=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        leaf_frac=np.asarray(pftinout.leaf_frac),
        npp_longterm=np.asarray(pftinout.npp_longterm),
        when_growthinit=np.asarray(pftinout.when_growthinit),
        everywhere=np.asarray(pftinout.everywhere),
        veget_max=np.asarray(pftinout.veget_max),
        bm_to_litter=np.asarray(gap.bm_to_litter),
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        ok_dgvm=True,
        min_stomate=1.0e-8,
    )
    assert bool(np.asarray(kill.pft_present)[0, pft])

    light = light_competition_step(
        veget_max=np.asarray(kill.veget_max),
        fpc_max=np.ones((npts, nvm), dtype=np.float64),
        pft_present=np.asarray(kill.pft_present),
        cn_ind=np.asarray(kill.cn_ind),
        lai=np.zeros((npts, nvm), dtype=np.float64),
        maxfpc_lastyear=np.zeros((npts, nvm), dtype=np.float64),
        lm_lastyearmax=np.asarray(pftinout.lm_lastyearmax),
        ind=np.asarray(kill.ind),
        biomass=np.asarray(kill.biomass),
        veget_lastlight=np.zeros((npts, nvm), dtype=np.float64),
        bm_to_litter=np.asarray(kill.bm_to_litter),
        mortality=np.zeros((npts, nvm), dtype=np.float64),
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.02,
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        ext_coeff=np.ones(nvm, dtype=np.float64),
        ok_dgvm=True,
        dt_days=1.0,
        min_stomate=1.0e-8,
    )
    assert np.asarray(light.veget_lastlight)[0, pft] > 0.0

    rates = establishment_rates_step(
        veget_max=np.asarray(kill.veget_max),
        pft_present=np.asarray(kill.pft_present),
        regenerate=np.asarray(constraints.regenerate),
        neighbours=neighbours,
        resolution=np.ones((npts, 2), dtype=np.float64),
        need_adjacent=np.asarray(pftinout.need_adjacent),
        herbivores=np.ones((npts, nvm), dtype=np.float64) * 1.0e6,
        precip_annual=np.full(npts, 200.0, dtype=np.float64),
        gdd0=np.full(npts, 200.0, dtype=np.float64),
        lm_lastyearmax=np.asarray(pftinout.lm_lastyearmax),
        cn_ind=np.asarray(kill.cn_ind),
        lai=np.ones((npts, nvm), dtype=np.float64),
        avail_tree=np.asarray(pftinout.avail_tree),
        avail_grass=np.asarray(pftinout.avail_grass),
        npp_longterm=np.asarray(kill.npp_longterm),
        ind=np.asarray(light.ind),
        everywhere=np.asarray(kill.everywhere),
        sla_calc=np.ones((npts, nvm), dtype=np.float64),
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        ext_coeff=np.ones(nvm, dtype=np.float64),
        ok_dgvm=True,
        dt_days=1.0,
        min_stomate=1.0e-8,
    )
    biomass_estab = establishment_biomass_step(
        d_ind=np.asarray(rates.d_ind),
        ind=np.asarray(light.ind),
        biomass=np.asarray(light.biomass),
        leaf_age=np.asarray(kill.leaf_age),
        leaf_frac=np.asarray(kill.leaf_frac),
        age=np.asarray(kill.age),
        co2_to_bm=np.asarray(pftinout.co2_to_bm),
        veget_max=np.asarray(kill.veget_max),
        woodmass_ind=np.zeros((npts, nvm), dtype=np.float64),
        mortality=np.asarray(light.mortality),
        bm_to_litter=np.asarray(light.bm_to_litter),
        bm_sapl=bm_sapl,
        npp_longterm=np.asarray(kill.npp_longterm),
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        maxdia=np.ones(nvm, dtype=np.float64),
        ok_dgvm=True,
        dt_days=1.0,
        min_stomate=1.0e-8,
    )
    assert np.asarray(biomass_estab.ind)[0, pft] >= np.asarray(light.ind)[0, pft]

    cover_params = _cover_fixture(npts=npts, nvm=nvm)
    cover_params["cn_ind"] = np.asarray(kill.cn_ind)
    cover_params["ind"] = np.asarray(biomass_estab.ind)
    cover_params["biomass"] = np.asarray(biomass_estab.biomass)
    cover_params["veget_max"] = np.asarray(kill.veget_max)
    cover_params["veget_max_old"] = np.asarray(kill.veget_max)
    cover_params["bm_to_litter"] = np.asarray(biomass_estab.bm_to_litter)
    cover_params["co2_to_bm"] = np.asarray(biomass_estab.co2_to_bm)
    cover_params["natural"] = natural
    cover_params["pasture"] = pasture
    cover = cover_step(**cover_params, ok_dgvm=True, min_stomate=1.0e-8)

    assert np.asarray(cover.veget_max)[0, pft] > 0.0
    assert np.allclose(np.asarray(cover.veget_max)[0, pft], np.asarray(kill.cn_ind)[0, pft] * np.asarray(biomass_estab.ind)[0, pft])


def _lpj_cover_peat_fixture():
    npts, nvm, nelements, ndeep = 1, 14, 1, 2
    veget_old = np.zeros((npts, nvm), dtype=np.float64)
    veget_old[0, 0] = 0.4
    veget_old[0, 1] = 0.6
    veget_new = veget_old.copy()
    veget_new[0, 1] = 0.6
    veget_new[0, PFT14] = 0.2
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[0, 1, ILEAF, ICARBON] = 12.0
    biomass[0, 1, IROOT, ICARBON] = 18.0
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[PFT14, ILEAF, ICARBON] = 3.0
    bm_sapl[PFT14, IROOT, ICARBON] = 5.0
    carbon = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    carbon[0, :, 1] = [10.0, 20.0, 30.0]
    cn_ind = np.ones((npts, nvm), dtype=np.float64)
    ind = np.zeros((npts, nvm), dtype=np.float64)
    natural = np.zeros(nvm, dtype=bool)
    natural[1] = True
    natural[PFT14] = True
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    return {
        "cn_ind": cn_ind,
        "ind": ind,
        "biomass": biomass,
        "veget_max_new": veget_new,
        "veget_max": veget_old.copy(),
        "veget_max_old": veget_old,
        "litter": np.zeros((npts, NLITT, nvm, NLEVS, nelements), dtype=np.float64),
        "litter_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "litter_not_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "carbon": carbon,
        "fuel_1hr": np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        "fuel_10hr": np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        "fuel_100hr": np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        "fuel_1000hr": np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64),
        "turnover_daily": np.zeros_like(biomass),
        "bm_to_litter": np.zeros_like(biomass),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "co2_fire": np.zeros((npts, nvm), dtype=np.float64),
        "resp_hetero": np.zeros((npts, nvm), dtype=np.float64),
        "resp_maint": np.zeros((npts, nvm), dtype=np.float64),
        "resp_growth": np.zeros((npts, nvm), dtype=np.float64),
        "gpp_daily": np.zeros((npts, nvm), dtype=np.float64),
        "deepC_a": np.zeros((npts, ndeep, nvm), dtype=np.float64),
        "deepC_s": np.zeros((npts, ndeep, nvm), dtype=np.float64),
        "deepC_p": np.zeros((npts, ndeep, nvm), dtype=np.float64),
        "dt_days": 2.0,
        "age": np.ones((npts, nvm), dtype=np.float64) * 7.0,
        "pft_present": np.zeros((npts, nvm), dtype=bool),
        "senescence": np.ones((npts, nvm), dtype=bool),
        "when_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "everywhere": np.zeros((npts, nvm), dtype=np.float64),
        "leaf_frac": np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        "lm_lastyearmax": np.zeros((npts, nvm), dtype=np.float64),
        "npp_longterm": np.zeros((npts, nvm), dtype=np.float64),
        "carbon_save": np.zeros((npts, NCARB, nvm), dtype=np.float64),
        "deepC_a_save": np.zeros((npts, ndeep), dtype=np.float64),
        "deepC_s_save": np.zeros((npts, ndeep), dtype=np.float64),
        "deepC_p_save": np.zeros((npts, ndeep), dtype=np.float64),
        "delta_fsave": np.zeros(npts, dtype=np.float64),
        "liqwt_max_lastyear": np.ones(npts, dtype=np.float64),
        "natural": natural,
        "pasture": np.zeros(nvm, dtype=bool),
        "is_peat": is_peat,
        "is_tree": np.zeros(nvm, dtype=bool),
        "bm_sapl": bm_sapl,
        "cn_sapl": np.ones(nvm, dtype=np.float64) * 0.5,
    }


def test_lpj_cover_peat_step_establishes_paper_peat_pft_and_moves_source_carbon():
    """Fortran: stomate_lpj.f90::lpj_cover_peat lines 2877-2999 and 3118-3155."""

    args = _lpj_cover_peat_fixture()
    result = lpj_cover_peat_step(**args, ok_dgvm=False, ok_pc=False)

    assert np.allclose(np.asarray(result.veget_max)[0, 0], 0.4)
    assert np.allclose(np.asarray(result.veget_max)[0, 1], 0.4)
    assert np.allclose(np.asarray(result.veget_max)[0, PFT14], 0.2)
    assert bool(np.asarray(result.pft_present)[0, PFT14])
    assert not bool(np.asarray(result.senescence)[0, PFT14])
    assert np.allclose(np.asarray(result.ind)[0, PFT14], 0.2)
    assert np.allclose(np.asarray(result.when_growthinit)[0, PFT14], 1.0e33)
    assert np.allclose(np.asarray(result.leaf_frac)[0, PFT14, 0], 1.0)
    assert np.allclose(np.asarray(result.npp_longterm)[0, PFT14], 10.0)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, ILEAF, ICARBON], 3.0)
    assert np.allclose(np.asarray(result.biomass)[0, PFT14, IROOT, ICARBON], 5.0)
    assert np.allclose(np.asarray(result.biomass)[0, 1, ILEAF, ICARBON], 12.0 * 0.6 / 0.4)
    assert np.allclose(np.asarray(result.carbon)[0, :, PFT14], [10.0, 20.0, 30.0])
    assert np.allclose(np.asarray(result.carbon)[0, :, 1], [10.0, 20.0, 30.0])
    assert np.allclose(np.asarray(result.delta_fsave), [0.0])


def test_lpj_cover_peat_step_contracting_peat_saves_old_peat_carbon():
    """Fortran: stomate_lpj.f90::lpj_cover_peat lines 3082-3098."""

    args = _lpj_cover_peat_fixture()
    args["veget_max_old"] = args["veget_max_new"].copy()
    args["veget_max"] = args["veget_max_new"].copy()
    args["veget_max_new"] = args["veget_max_new"].copy()
    args["veget_max_new"][0, 1] = 0.6
    args["veget_max_new"][0, PFT14] = 0.1
    args["carbon"][0, :, PFT14] = [100.0, 200.0, 300.0]

    result = lpj_cover_peat_step(**args, ok_dgvm=False, ok_pc=False)

    delta_saved = 0.2 - np.asarray(result.veget_max)[0, PFT14]
    assert delta_saved > 0.0
    assert np.allclose(np.asarray(result.carbon_save)[0, :, PFT14], np.asarray([100.0, 200.0, 300.0]) * delta_saved)
    assert np.allclose(np.asarray(result.delta_fsave), [delta_saved])


def test_stomate_lpj_cover_dispatch_uses_peat_branch_and_clears_update_flag():
    """Fortran: stomate_lpj.f90::StomateLpj lines 1358-1380."""

    peat_inputs = _lpj_cover_peat_fixture()
    ordinary_inputs = _cover_fixture(npts=1, nvm=14)
    ordinary_inputs["cn_ind"][0, 1] = 0.9
    ordinary_inputs["ind"][0, 1] = 1.0

    result = stomate_lpj_cover_dispatch_step(
        update_peatfrac=True,
        done_update_peatfrac=False,
        peat_cover_inputs=peat_inputs,
        cover_inputs=ordinary_inputs,
    )

    assert result.cover is None
    assert result.peat_cover is not None
    assert result.used_peat_cover is True
    assert result.update_peatfrac is False
    assert result.done_update_peatfrac is True
    assert np.allclose(np.asarray(result.peat_cover.veget_max)[0, PFT14], 0.2)


def test_lpj_cover_peat_step_ok_pc_saves_deep_carbon_when_peat_contracts():
    """Fortran: stomate_lpj.f90::lpj_cover_peat lines 3093-3097."""

    args = _lpj_cover_peat_fixture()
    args["veget_max_old"] = args["veget_max_new"].copy()
    args["veget_max"] = args["veget_max_new"].copy()
    args["veget_max_new"] = args["veget_max_new"].copy()
    args["veget_max_new"][0, 1] = 0.6
    args["veget_max_new"][0, PFT14] = 0.1
    args["deepC_a"][0, :, PFT14] = [10.0, 20.0]
    args["deepC_s"][0, :, PFT14] = [30.0, 40.0]
    args["deepC_p"][0, :, PFT14] = [50.0, 60.0]

    result = lpj_cover_peat_step(**args, ok_dgvm=False, ok_pc=True)

    delta_saved = 0.2 - np.asarray(result.veget_max)[0, PFT14]
    assert np.allclose(np.asarray(result.deepC_a_save)[0], np.asarray([10.0, 20.0]) * delta_saved)
    assert np.allclose(np.asarray(result.deepC_s_save)[0], np.asarray([30.0, 40.0]) * delta_saved)
    assert np.allclose(np.asarray(result.deepC_p_save)[0], np.asarray([50.0, 60.0]) * delta_saved)


def test_lpj_cover_peat_step_ok_pc_scales_deep_carbon_and_saved_old_peat_on_expansion():
    """Fortran: stomate_lpj.f90::lpj_cover_peat lines 3113-3164 and 3183-3204."""

    args = _lpj_cover_peat_fixture()
    args["veget_max_old"] = args["veget_max_new"].copy()
    args["veget_max_old"][0, 0] = 0.2
    args["veget_max_old"][0, 1] = 0.6
    args["veget_max_old"][0, PFT14] = 0.2
    args["veget_max"] = args["veget_max_new"].copy()
    args["veget_max"][0, 0] = 0.2
    args["veget_max"][0, 1] = 0.6
    args["veget_max"][0, PFT14] = 0.2
    args["veget_max_new"] = args["veget_max_new"].copy()
    args["veget_max_new"][0, 0] = 0.2
    args["veget_max_new"][0, 1] = 0.5
    args["veget_max_new"][0, PFT14] = 0.3
    args["carbon"][0, :, 1] = [1000.0, 2000.0, 3000.0]
    args["carbon"][0, :, PFT14] = [2.0, 4.0, 8.0]
    args["carbon_save"][0, :, PFT14] = [100.0, 200.0, 300.0]
    args["delta_fsave"] = np.asarray([0.5], dtype=np.float64)
    args["deepC_a"][0, :, PFT14] = [10.0, 20.0]
    args["deepC_s"][0, :, PFT14] = [30.0, 40.0]
    args["deepC_p"][0, :, PFT14] = [50.0, 60.0]
    args["deepC_a_save"][0] = [7.0, 9.0]
    args["deepC_s_save"][0] = [11.0, 13.0]
    args["deepC_p_save"][0] = [15.0, 17.0]

    result = lpj_cover_peat_step(**args, ok_dgvm=False, ok_pc=True)

    delta_fpeat = float(np.asarray(result.delta_fpeat)[0])
    peat_cover = float(np.asarray(result.veget_max)[0, PFT14])
    old_save = 0.5
    carbon_obtain = (delta_fpeat / old_save) * args["carbon_save"][0, :, PFT14]
    expected_carbon_peat = (args["carbon"][0, :, PFT14] * 0.2 + carbon_obtain) / peat_cover
    np.testing.assert_allclose(np.asarray(result.carbon)[0, :, PFT14], expected_carbon_peat)
    np.testing.assert_allclose(
        np.asarray(result.deepC_a)[0, :, PFT14],
        args["deepC_a"][0, :, PFT14] * expected_carbon_peat[IACTIVE] / args["carbon"][0, IACTIVE, PFT14],
    )
    np.testing.assert_allclose(
        np.asarray(result.deepC_s)[0, :, PFT14],
        args["deepC_s"][0, :, PFT14] * expected_carbon_peat[ISLOW] / args["carbon"][0, ISLOW, PFT14],
    )
    np.testing.assert_allclose(
        np.asarray(result.deepC_p)[0, :, PFT14],
        args["deepC_p"][0, :, PFT14] * expected_carbon_peat[IPASSIVE] / args["carbon"][0, IPASSIVE, PFT14],
    )
    np.testing.assert_allclose(np.asarray(result.carbon_save)[0, :, PFT14], args["carbon_save"][0, :, PFT14] * (1.0 - delta_fpeat / old_save))
    post_fsave = old_save - delta_fpeat
    source_factor_after_update = max(1.0 - delta_fpeat / post_fsave, 0.0)
    np.testing.assert_allclose(np.asarray(result.deepC_a_save)[0], args["deepC_a_save"][0] * source_factor_after_update)
    np.testing.assert_allclose(np.asarray(result.deepC_s_save)[0], args["deepC_s_save"][0] * source_factor_after_update)
    np.testing.assert_allclose(np.asarray(result.deepC_p_save)[0], args["deepC_p_save"][0] * source_factor_after_update)


def _agripeat_fraction_fixture(npts=1):
    nvm = 16
    veget_max_new = np.zeros((npts, nvm), dtype=np.float64)
    veget_max_old = np.zeros((npts, nvm), dtype=np.float64)
    natural = np.zeros(nvm, dtype=bool)
    natural[0] = True
    natural[1] = True
    pasture = np.zeros(nvm, dtype=bool)
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    pft_to_mtc = np.arange(nvm, dtype=np.int32)
    pft_to_mtc[14] = 16
    pft_to_mtc[15] = 17
    return {
        "veget_max_new": veget_max_new,
        "veget_max_old": veget_max_old,
        "natural": natural,
        "pasture": pasture,
        "is_peat": is_peat,
        "pft_to_mtc": pft_to_mtc,
    }


def test_agripeat_adjust_fractions_prop_first_peat_crop_occurrence_overlays_map():
    """Fortran: stomate_lcchange.f90::agripeat_adjust_fractions lines 1708-1726."""

    args = _agripeat_fraction_fixture()
    args["veget_max_old"][0, PFT14] = 0.2
    args["veget_max_new"][0, 0] = 0.4
    args["veget_max_new"][0, 1] = 0.3
    args["veget_max_new"][0, 11] = 0.1
    args["veget_max_new"][0, 12] = 0.2

    result = agripeat_adjust_fractions_step(**args, agri_peat_prop=True)

    expected = np.zeros(16, dtype=np.float64)
    expected[0:13] = args["veget_max_new"][0, 0:13] * 0.8
    expected[14] = 0.2 * args["veget_max_new"][0, 11]
    expected[15] = 0.2 * args["veget_max_new"][0, 12]
    expected[PFT14] = 0.2 - expected[14] - expected[15]
    np.testing.assert_allclose(np.asarray(result.veget_max_adjusted)[0], expected)
    np.testing.assert_allclose(np.asarray(result.sumpeat_old), [0.2])
    np.testing.assert_allclose(np.asarray(result.mass_error), [0.0])


def test_agripeat_adjust_fractions_mincrop_covers_enough_and_not_enough_natural_area():
    """Fortran: stomate_lcchange.f90::agripeat_adjust_fractions lines 1761-1806."""

    args = _agripeat_fraction_fixture(npts=2)
    args["veget_max_old"][0, PFT14] = 0.2
    args["veget_max_new"][0, 0] = 0.6
    args["veget_max_new"][0, 1] = 0.2
    args["veget_max_new"][0, 11] = 0.1
    args["veget_max_new"][0, 12] = 0.1

    args["veget_max_old"][1, PFT14] = 0.4
    args["veget_max_old"][1, 14] = 0.1
    args["veget_max_new"][1, 0] = 0.2
    args["veget_max_new"][1, 1] = 0.1
    args["veget_max_new"][1, 11] = 0.2
    args["veget_max_new"][1, 12] = 0.1

    result = agripeat_adjust_fractions_step(**args, agri_peat_mincrop=True)
    adjusted = np.asarray(result.veget_max_adjusted)

    expected0 = args["veget_max_new"][0].copy()
    expected0[PFT14] = 0.2
    expected0[0] = 0.6 * (0.8 - 0.2) / 0.8
    expected0[1] = 0.2 * (0.8 - 0.2) / 0.8
    expected0[14] = 0.0
    expected0[15] = 0.0
    np.testing.assert_allclose(adjusted[0], expected0)

    expected1 = args["veget_max_new"][1].copy()
    expected1[PFT14] = 0.3
    expected1[0] = 0.0
    expected1[1] = 0.0
    expected1[14] = 0.2 * 0.2 / 0.3
    expected1[15] = 0.2 * 0.1 / 0.3
    expected1[11] = 0.2 - expected1[14]
    expected1[12] = 0.1 - expected1[15]
    np.testing.assert_allclose(adjusted[1], expected1)
    np.testing.assert_allclose(np.asarray(result.sum_peat_crops_new), [0.0, 0.2])
    np.testing.assert_allclose(np.asarray(result.mass_error), [0.0, 0.0])


def test_agripeat_adjust_fractions_maxcrop_covers_all_disturbed_and_residual_peat():
    """Fortran: stomate_lcchange.f90::agripeat_adjust_fractions lines 1810-1853."""

    args = _agripeat_fraction_fixture(npts=2)
    args["veget_max_old"][0, PFT14] = 0.3
    args["veget_max_new"][0, 0] = 0.4
    args["veget_max_new"][0, 1] = 0.2
    args["veget_max_new"][0, 11] = 0.3
    args["veget_max_new"][0, 12] = 0.1

    args["veget_max_old"][1, PFT14] = 0.5
    args["veget_max_old"][1, 14] = 0.1
    args["veget_max_new"][1, 0] = 0.6
    args["veget_max_new"][1, 1] = 0.2
    args["veget_max_new"][1, 11] = 0.1
    args["veget_max_new"][1, 12] = 0.1

    result = agripeat_adjust_fractions_step(**args, agri_peat_maxcrop=True)
    adjusted = np.asarray(result.veget_max_adjusted)

    expected0 = args["veget_max_new"][0].copy()
    expected0[14] = 0.3 * 0.3 / 0.4
    expected0[15] = 0.3 * 0.1 / 0.4
    expected0[PFT14] = 0.0
    expected0[11] = 0.3 - expected0[14]
    expected0[12] = 0.1 - expected0[15]
    np.testing.assert_allclose(adjusted[0], expected0)

    expected1 = args["veget_max_new"][1].copy()
    expected1[14] = 0.1
    expected1[15] = 0.1
    expected1[11] = 0.0
    expected1[12] = 0.0
    expected1[PFT14] = 0.4
    expected1[0] = 0.6 * (0.8 - 0.4) / 0.8
    expected1[1] = 0.2 * (0.8 - 0.4) / 0.8
    np.testing.assert_allclose(adjusted[1], expected1)
    np.testing.assert_allclose(np.asarray(result.mass_error), [0.0, 0.0])


def _agripeat_redistribution_fixture():
    npts, nvm, nelements, ndeep = 1, 16, 1, 2
    veget_max_old = np.zeros((npts, nvm), dtype=np.float64)
    veget_max_adjusted = np.zeros_like(veget_max_old)
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    litter = np.zeros((npts, NLITT, nvm, NLEVS, nelements), dtype=np.float64)
    litter_avail = np.zeros((npts, NLITT, nvm), dtype=np.float64)
    litter_not_avail = np.zeros_like(litter_avail)
    bm_to_litter = np.zeros_like(biomass)
    carbon = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    deepC_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deepC_s = np.zeros_like(deepC_a)
    deepC_p = np.zeros_like(deepC_a)
    coeff_lcchange_1 = np.ones(nvm, dtype=np.float64) * 0.1
    coeff_lcchange_10 = np.ones(nvm, dtype=np.float64) * 0.2
    coeff_lcchange_100 = np.ones(nvm, dtype=np.float64) * 0.3
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    return {
        "veget_max_adjusted": veget_max_adjusted,
        "veget_max_old": veget_max_old,
        "biomass": biomass,
        "litter": litter,
        "litter_avail": litter_avail,
        "litter_not_avail": litter_not_avail,
        "bm_to_litter": bm_to_litter,
        "carbon": carbon,
        "deepC_a": deepC_a,
        "deepC_s": deepC_s,
        "deepC_p": deepC_p,
        "coeff_lcchange_1": coeff_lcchange_1,
        "coeff_lcchange_10": coeff_lcchange_10,
        "coeff_lcchange_100": coeff_lcchange_100,
        "is_peat": is_peat,
    }


def test_lcchange_agripeat_redistribution_sends_shrinking_natural_peat_to_both_peat_crops():
    """Fortran: stomate_lcchange.f90::lcchange_main_agripeat lines 1342-1389."""

    args = _agripeat_redistribution_fixture()
    args["veget_max_old"][0, PFT14] = 0.4
    args["veget_max_adjusted"][0, PFT14] = 0.2
    args["veget_max_adjusted"][0, 14] = 0.1
    args["veget_max_adjusted"][0, 15] = 0.1
    args["litter"][0, :, PFT14, :, 0] = 100.0
    args["carbon"][0, :, PFT14] = [10.0, 20.0, 30.0]
    args["deepC_a"][0, :, PFT14] = [1.0, 2.0]
    args["deepC_s"][0, :, PFT14] = [3.0, 4.0]
    args["deepC_p"][0, :, PFT14] = [5.0, 6.0]
    for part, value in ((ILEAF, 2.0), (ISAPBELOW, 4.0), (IHEARTBELOW, 6.0), (IROOT, 8.0), (IFRUIT, 10.0), (ICARBRES, 12.0)):
        args["biomass"][0, PFT14, part, ICARBON] = value
    args["biomass"][0, PFT14, ISAPABOVE, ICARBON] = 8.0
    args["biomass"][0, PFT14, IHEARTABOVE, ICARBON] = 12.0

    result = lcchange_agripeat_redistribution_step(**args, ok_pc=True)

    for pft in (14, 15):
        np.testing.assert_allclose(np.asarray(result.litter)[0, :, pft, :, 0], 100.0)
        np.testing.assert_allclose(np.asarray(result.carbon)[0, :, pft], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(np.asarray(result.deepC_a)[0, :, pft], [1.0, 2.0])
        np.testing.assert_allclose(np.asarray(result.deepC_s)[0, :, pft], [3.0, 4.0])
        np.testing.assert_allclose(np.asarray(result.deepC_p)[0, :, pft], [5.0, 6.0])
        np.testing.assert_allclose(np.asarray(result.bm_to_litter)[0, pft, ILEAF, ICARBON], 2.0)
        np.testing.assert_allclose(np.asarray(result.bm_to_litter)[0, pft, ISAPBELOW, ICARBON], 4.0)
        np.testing.assert_allclose(np.asarray(result.bm_to_litter)[0, pft, IHEARTBELOW, ICARBON], 6.0)
        np.testing.assert_allclose(np.asarray(result.bm_to_litter)[0, pft, IROOT, ICARBON], 8.0)
        np.testing.assert_allclose(np.asarray(result.bm_to_litter)[0, pft, IFRUIT, ICARBON], 10.0)
        np.testing.assert_allclose(np.asarray(result.bm_to_litter)[0, pft, ICARBRES, ICARBON], 12.0)

    above = 20.0
    np.testing.assert_allclose(np.asarray(result.convflux), [0.1 * above * 0.2])
    np.testing.assert_allclose(np.asarray(result.prod10_entry), [0.2 * above * 0.2])
    np.testing.assert_allclose(np.asarray(result.prod100_entry), [0.3 * above * 0.2])
    np.testing.assert_allclose(np.asarray(result.delta_nat_peat), [-0.2])


def test_lcchange_agripeat_redistribution_abandoned_peat_crop_feeds_expanding_pft():
    """Fortran: stomate_lcchange.f90::lcchange_main_agripeat lines 1498-1558."""

    args = _agripeat_redistribution_fixture()
    args["veget_max_old"][0, PFT14] = 0.4
    args["veget_max_adjusted"][0, PFT14] = 0.4
    args["veget_max_old"][0, 14] = 0.2
    args["veget_max_adjusted"][0, 14] = 0.1
    args["veget_max_old"][0, 0] = 0.2
    args["veget_max_adjusted"][0, 0] = 0.3
    args["litter"][0, :, 14, :, 0] = 50.0
    args["litter"][0, :, 0, :, 0] = 10.0
    args["carbon"][0, :, 14] = [5.0, 7.0, 9.0]
    args["carbon"][0, :, 0] = [1.0, 2.0, 3.0]
    args["deepC_a"][0, :, 14] = [2.0, 4.0]
    args["deepC_s"][0, :, 14] = [6.0, 8.0]
    args["deepC_p"][0, :, 14] = [10.0, 12.0]
    args["biomass"][0, 14, ILEAF, ICARBON] = 3.0
    args["biomass"][0, 14, ISAPBELOW, ICARBON] = 4.0
    args["biomass"][0, 14, IHEARTBELOW, ICARBON] = 5.0
    args["biomass"][0, 14, IROOT, ICARBON] = 6.0
    args["biomass"][0, 14, IFRUIT, ICARBON] = 7.0
    args["biomass"][0, 14, ICARBRES, ICARBON] = 8.0
    args["biomass"][0, 14, ISAPABOVE, ICARBON] = 4.0
    args["biomass"][0, 14, IHEARTABOVE, ICARBON] = 6.0

    result = lcchange_agripeat_redistribution_step(**args, ok_pc=True)

    np.testing.assert_allclose(np.asarray(result.litter)[0, :, 0, :, 0], (10.0 * 0.2 + 50.0 * 0.1) / 0.3)
    np.testing.assert_allclose(np.asarray(result.carbon)[0, :, 0], (np.asarray([1.0, 2.0, 3.0]) * 0.2 + np.asarray([5.0, 7.0, 9.0]) * 0.1) / 0.3)
    np.testing.assert_allclose(np.asarray(result.deepC_a)[0, :, 0], np.asarray([2.0, 4.0]) * 0.1 / 0.3)
    np.testing.assert_allclose(np.asarray(result.deepC_s)[0, :, 0], np.asarray([6.0, 8.0]) * 0.1 / 0.3)
    np.testing.assert_allclose(np.asarray(result.deepC_p)[0, :, 0], np.asarray([10.0, 12.0]) * 0.1 / 0.3)
    np.testing.assert_allclose(np.asarray(result.bm_to_litter)[0, 0, ILEAF, ICARBON], 3.0 * 0.1 / 0.3)
    np.testing.assert_allclose(np.asarray(result.bm_to_litter)[0, 0, ISAPBELOW, ICARBON], 4.0 * 0.1 / 0.3)
    above = 10.0
    np.testing.assert_allclose(np.asarray(result.convflux), [0.1 * above * 0.1])
    np.testing.assert_allclose(np.asarray(result.prod10_entry), [0.2 * above * 0.1])
    np.testing.assert_allclose(np.asarray(result.prod100_entry), [0.3 * above * 0.1])


def test_lcchange_agripeat_finalize_ages_products_and_rebalances_fuel_from_litter():
    """Fortran: stomate_lcchange.f90::lcchange_main_agripeat lines 1561-1613."""

    npts, nvm, nelements = 1, 2, 1
    prod10 = np.zeros((npts, 11), dtype=np.float64)
    prod100 = np.zeros((npts, 101), dtype=np.float64)
    flux10 = np.zeros((npts, 10), dtype=np.float64)
    flux100 = np.zeros((npts, 100), dtype=np.float64)
    prod10[0, 0] = 20.0
    prod10[0, 1] = 30.0
    flux10[0, 0] = 3.0
    prod100[0, 0] = 40.0
    prod100[0, 1] = 50.0
    flux100[0, 0] = 0.5
    litter = np.zeros((npts, NLITT, nvm, NLEVS, nelements), dtype=np.float64)
    litter[0, :, 0, IABOVE, 0] = [80.0, 40.0]
    litter[0, :, 1, IABOVE, 0] = [20.0, 12.0]
    fuel_1hr = np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64)
    fuel_10hr = np.zeros_like(fuel_1hr)
    fuel_100hr = np.zeros_like(fuel_1hr)
    fuel_1000hr = np.zeros_like(fuel_1hr)
    fuel_1hr[0, 0, :, 0] = [1.0, 2.0]
    fuel_10hr[0, 0, :, 0] = [3.0, 2.0]
    fuel_100hr[0, 0, :, 0] = [2.0, 2.0]
    fuel_1000hr[0, 0, :, 0] = [4.0, 4.0]

    result = lcchange_agripeat_finalize_step(
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        convflux=np.asarray([10.0], dtype=np.float64),
        cflux_prod10=np.asarray([1.0], dtype=np.float64),
        cflux_prod100=np.asarray([2.0], dtype=np.float64),
        litter=litter,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        dt_days=2.0,
        one_year=10.0,
    )

    np.testing.assert_allclose(np.asarray(result.prod10)[0, 1], 20.0)
    np.testing.assert_allclose(np.asarray(result.flux10)[0, 0], 2.0)
    np.testing.assert_allclose(np.asarray(result.prod100)[0, 1], 40.0)
    np.testing.assert_allclose(np.asarray(result.flux100)[0, 0], 0.4)
    np.testing.assert_allclose(np.asarray(result.convflux), [2.0])
    np.testing.assert_allclose(np.asarray(result.cflux_prod10), [(1.0 + 3.0) * 0.2])
    np.testing.assert_allclose(np.asarray(result.cflux_prod100), [(2.0 + 0.5) * 0.2])
    np.testing.assert_allclose(np.asarray(result.fuel_1hr)[0, 0, :, 0], [80.0 * 0.1, 40.0 * 0.2])
    np.testing.assert_allclose(np.asarray(result.fuel_10hr)[0, 0, :, 0], [80.0 * 0.3, 40.0 * 0.2])
    np.testing.assert_allclose(np.asarray(result.fuel_100hr)[0, 0, :, 0], [80.0 * 0.2, 40.0 * 0.2])
    np.testing.assert_allclose(np.asarray(result.fuel_1000hr)[0, 0, :, 0], [80.0 * 0.4, 40.0 * 0.4])
    np.testing.assert_allclose(np.asarray(result.fuel_1hr)[0, 1, :, 0], [5.0, 3.0])
    np.testing.assert_allclose(np.asarray(result.fuel_10hr)[0, 1, :, 0], [5.0, 3.0])
    np.testing.assert_allclose(np.asarray(result.fuel_100hr)[0, 1, :, 0], [5.0, 3.0])
    np.testing.assert_allclose(np.asarray(result.fuel_1000hr)[0, 1, :, 0], [5.0, 3.0])


def _agripeat_wrapper_fixture():
    npts, nvm, nelements, ndeep = 1, 16, 1, 2
    veget_max_old = np.zeros((npts, nvm), dtype=np.float64)
    veget_max_old[0, 0] = 0.6
    veget_max_old[0, PFT14] = 0.4
    veget_max = np.zeros_like(veget_max_old)
    veget_max[0, 0] = 0.5
    veget_max[0, 11] = 0.25
    veget_max[0, 12] = 0.25
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[0, PFT14, ILEAF, ICARBON] = 2.0
    biomass[0, PFT14, ISAPABOVE, ICARBON] = 8.0
    biomass[0, PFT14, IHEARTABOVE, ICARBON] = 12.0
    biomass[0, PFT14, ISAPBELOW, ICARBON] = 4.0
    biomass[0, PFT14, IROOT, ICARBON] = 6.0
    biomass[0, 0, ILEAF, ICARBON] = 3.0
    biomass[0, 0, ISAPABOVE, ICARBON] = 5.0
    biomass[0, 0, IHEARTABOVE, ICARBON] = 5.0
    litter = np.zeros((npts, NLITT, nvm, NLEVS, nelements), dtype=np.float64)
    litter[0, :, PFT14, :, 0] = 100.0
    litter[0, :, 0, :, 0] = 20.0
    bm_to_litter = np.zeros_like(biomass)
    carbon = np.zeros((npts, NCARB, nvm), dtype=np.float64)
    carbon[0, :, PFT14] = [10.0, 20.0, 30.0]
    carbon[0, :, 0] = [1.0, 2.0, 3.0]
    deepC_a = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    deepC_s = np.zeros_like(deepC_a)
    deepC_p = np.zeros_like(deepC_a)
    deepC_a[0, :, PFT14] = [1.0, 2.0]
    deepC_s[0, :, PFT14] = [3.0, 4.0]
    deepC_p[0, :, PFT14] = [5.0, 6.0]
    fuel_1hr = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64)
    fuel_10hr = np.ones_like(fuel_1hr) * 2.0
    fuel_100hr = np.ones_like(fuel_1hr) * 3.0
    fuel_1000hr = np.ones_like(fuel_1hr) * 4.0
    natural = np.zeros(nvm, dtype=bool)
    natural[0] = True
    natural[1] = True
    natural[PFT14] = True
    pasture = np.zeros(nvm, dtype=bool)
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[PFT14] = True
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[14] = True
    pft_to_mtc = np.arange(nvm, dtype=np.int32)
    pft_to_mtc[14] = 16
    pft_to_mtc[15] = 17
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[14, ILEAF, ICARBON] = 0.5
    bm_sapl[14, IROOT, ICARBON] = 0.25
    bm_sapl[15, ILEAF, ICARBON] = 0.4
    bm_sapl[15, IROOT, ICARBON] = 0.2
    cn_sapl = np.ones(nvm, dtype=np.float64)
    cn_sapl[14] = 0.5
    return {
        "dt_days": 2.0,
        "veget_max": veget_max,
        "veget_max_old": veget_max_old,
        "biomass": biomass,
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "age": np.ones((npts, nvm), dtype=np.float64) * 9.0,
        "pft_present": np.zeros((npts, nvm), dtype=bool),
        "senescence": np.ones((npts, nvm), dtype=bool),
        "when_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "everywhere": np.zeros((npts, nvm), dtype=np.float64),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "bm_to_litter": bm_to_litter,
        "turnover_daily": np.ones_like(biomass) * 0.7,
        "bm_sapl": bm_sapl,
        "cn_ind": np.ones((npts, nvm), dtype=np.float64),
        "flux10": np.zeros((npts, 10), dtype=np.float64),
        "flux100": np.zeros((npts, 100), dtype=np.float64),
        "prod10": np.zeros((npts, 11), dtype=np.float64),
        "prod100": np.zeros((npts, 101), dtype=np.float64),
        "leaf_frac": np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        "npp_longterm": np.zeros((npts, nvm), dtype=np.float64),
        "lm_lastyearmax": np.zeros((npts, nvm), dtype=np.float64),
        "litter": litter,
        "litter_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "litter_not_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "carbon": carbon,
        "deepC_a": deepC_a,
        "deepC_s": deepC_s,
        "deepC_p": deepC_p,
        "fuel_1hr": fuel_1hr,
        "fuel_10hr": fuel_10hr,
        "fuel_100hr": fuel_100hr,
        "fuel_1000hr": fuel_1000hr,
        "natural": natural,
        "pasture": pasture,
        "is_peat": is_peat,
        "is_tree": is_tree,
        "pft_to_mtc": pft_to_mtc,
        "coeff_lcchange_1": np.ones(nvm, dtype=np.float64) * 0.1,
        "coeff_lcchange_10": np.ones(nvm, dtype=np.float64) * 0.2,
        "coeff_lcchange_100": np.ones(nvm, dtype=np.float64) * 0.3,
        "cn_sapl": cn_sapl,
        "agri_peat_prop": True,
        "ok_pc": True,
        "one_year": 10.0,
        "npp_longterm_init": 12.0,
    }


def test_lcchange_main_agripeat_step_composes_fraction_init_redistribution_and_finalize():
    """Fortran: stomate_lcchange.f90::lcchange_main_agripeat lines 1253-1613."""

    args = _agripeat_wrapper_fixture()
    result = lcchange_main_agripeat_step(**args)

    expected_adjusted = np.zeros(16, dtype=np.float64)
    expected_adjusted[0] = 0.5 * 0.6
    expected_adjusted[11] = 0.25 * 0.6
    expected_adjusted[12] = 0.25 * 0.6
    expected_adjusted[PFT14] = 0.4 - 0.4 * 0.25 - 0.4 * 0.25
    expected_adjusted[14] = 0.4 * 0.25
    expected_adjusted[15] = 0.4 * 0.25
    np.testing.assert_allclose(np.asarray(result.veget_max_adjusted)[0], expected_adjusted)

    assert bool(np.asarray(result.pft_present)[0, 14])
    assert not bool(np.asarray(result.senescence)[0, 14])
    np.testing.assert_allclose(np.asarray(result.cn_ind)[0, 14], 0.5)
    np.testing.assert_allclose(np.asarray(result.ind)[0, 14], 0.1 / 0.5)
    np.testing.assert_allclose(np.asarray(result.leaf_frac)[0, 14, 0], 1.0)
    np.testing.assert_allclose(np.asarray(result.npp_longterm)[0, 14], 12.0)
    np.testing.assert_allclose(np.asarray(result.lm_lastyearmax)[0, 14], 0.5 * 0.2)
    np.testing.assert_allclose(np.asarray(result.biomass)[0, 14, ILEAF, ICARBON], 1.0)
    np.testing.assert_allclose(np.asarray(result.co2_to_bm)[0, 14], ((0.2 * 0.5) + (0.2 * 0.25)) * 2.0 / (10.0 * 0.1))

    manual_adjusted = agripeat_adjust_fractions_step(
        veget_max_new=args["veget_max"],
        veget_max_old=args["veget_max_old"],
        natural=args["natural"],
        pasture=args["pasture"],
        is_peat=args["is_peat"],
        pft_to_mtc=args["pft_to_mtc"],
        agri_peat_prop=True,
    ).veget_max_adjusted
    manual_biomass = args["biomass"].copy()
    manual_biomass[0, 14, ILEAF, ICARBON] = 1.0
    manual_biomass[0, 14, IROOT, ICARBON] = 0.5
    manual_biomass[0, 15, ILEAF, ICARBON] = 0.4
    manual_biomass[0, 15, IROOT, ICARBON] = 0.2
    redistribution = lcchange_agripeat_redistribution_step(
        veget_max_adjusted=manual_adjusted,
        veget_max_old=args["veget_max_old"],
        biomass=manual_biomass,
        litter=args["litter"],
        litter_avail=args["litter_avail"],
        litter_not_avail=args["litter_not_avail"],
        bm_to_litter=args["bm_to_litter"],
        carbon=args["carbon"],
        deepC_a=args["deepC_a"],
        deepC_s=args["deepC_s"],
        deepC_p=args["deepC_p"],
        coeff_lcchange_1=args["coeff_lcchange_1"],
        coeff_lcchange_10=args["coeff_lcchange_10"],
        coeff_lcchange_100=args["coeff_lcchange_100"],
        is_peat=args["is_peat"],
        ok_pc=True,
    )
    prod10 = args["prod10"].copy()
    prod100 = args["prod100"].copy()
    prod10[:, 0] = np.asarray(redistribution.prod10_entry)
    prod100[:, 0] = np.asarray(redistribution.prod100_entry)
    finalized = lcchange_agripeat_finalize_step(
        prod10=prod10,
        prod100=prod100,
        flux10=args["flux10"],
        flux100=args["flux100"],
        convflux=redistribution.convflux,
        cflux_prod10=np.zeros(1, dtype=np.float64),
        cflux_prod100=np.zeros(1, dtype=np.float64),
        litter=redistribution.litter,
        fuel_1hr=args["fuel_1hr"],
        fuel_10hr=args["fuel_10hr"],
        fuel_100hr=args["fuel_100hr"],
        fuel_1000hr=args["fuel_1000hr"],
        dt_days=2.0,
        one_year=10.0,
    )
    np.testing.assert_allclose(np.asarray(result.prod10), np.asarray(finalized.prod10))
    np.testing.assert_allclose(np.asarray(result.prod100), np.asarray(finalized.prod100))
    np.testing.assert_allclose(np.asarray(result.convflux), np.asarray(finalized.convflux))
    np.testing.assert_allclose(np.asarray(result.fuel_1hr), np.asarray(finalized.fuel_1hr))


def test_vmax_step_advances_leaf_age_classes_and_computes_weighted_vcmax():
    npts, nvm = 1, 3
    leaf_age = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_frac = np.zeros_like(leaf_age)
    leaf_age[0, 1, :] = [1.0, 20.0, 40.0, 80.0]
    leaf_frac[0, 1, :] = [0.4, 0.3, 0.2, 0.1]

    result = vmax_step(
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        vcmax25=np.asarray([0.0, 50.0, 60.0], dtype=np.float64),
        n_limfert=np.ones((npts, nvm), dtype=np.float64),
        leaf_timecst=np.asarray([1.0, 25.0, 25.0], dtype=np.float64),
        leafagecrit=np.asarray([1.0, 100.0, 100.0], dtype=np.float64),
        pheno_type=np.zeros(nvm, dtype=np.int32),
        leaf_tab=np.zeros(nvm, dtype=np.int32),
        ok_laidev=np.zeros(nvm, dtype=bool),
        dt_days=5.0,
    )

    d2 = 0.4 * 5.0 / 25.0
    d3 = 0.3 * 5.0 / 25.0
    d4 = 0.2 * 5.0 / 25.0
    expected_frac = np.asarray([0.4 - d2, 0.3 + d2 - d3, 0.2 + d3 - d4, 0.1 + d4])
    expected_age = np.asarray(
        [
            6.0,
            ((0.3 - d3) * 25.0 + d2 * 6.0) / (0.3 + d2 - d3),
            ((0.2 - d4) * 45.0 + d3 * 25.0) / (0.2 + d3 - d4),
            (0.1 * 85.0 + d4 * 45.0) / (0.1 + d4),
        ]
    )
    rel_age = expected_age / 100.0
    expected_eff = np.maximum(
        0.3,
        np.minimum(
            1.0,
            np.minimum(
                0.3 + 0.7 * rel_age / 0.03,
                1.0 - 0.7 * (rel_age - 0.5) / (1.0 - 0.5),
            ),
        ),
    )

    assert np.allclose(np.asarray(result.leaf_frac)[0, 1, :], expected_frac)
    assert np.allclose(np.asarray(result.leaf_age)[0, 1, :], expected_age)
    assert np.allclose(np.asarray(result.vcmax)[0, 1], 50.0 * np.sum(expected_eff * expected_frac))


def test_vmax_step_applies_n_limfert_for_lai_dev_or_global_nlim_and_dgvm_evergreen_exception():
    npts, nvm = 1, 4
    leaf_age = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 10.0
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[:, :, 0] = 1.0
    vcmax25 = np.asarray([0.0, 40.0, 50.0, 60.0], dtype=np.float64)
    n_limfert = np.ones((npts, nvm), dtype=np.float64)
    n_limfert[0, 1] = 0.5
    n_limfert[0, 2] = 0.25
    ok_laidev = np.zeros(nvm, dtype=bool)
    ok_laidev[1] = True
    pheno_type = np.zeros(nvm, dtype=np.int32)
    leaf_tab = np.zeros(nvm, dtype=np.int32)
    pheno_type[3] = 1
    leaf_tab[3] = 2

    result = vmax_step(
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        vcmax25=vcmax25,
        n_limfert=n_limfert,
        leaf_timecst=np.ones(nvm, dtype=np.float64) * 100.0,
        leafagecrit=np.ones(nvm, dtype=np.float64) * 100.0,
        pheno_type=pheno_type,
        leaf_tab=leaf_tab,
        ok_laidev=ok_laidev,
        ok_dgvm=True,
        dt_days=0.0,
    )

    rel_age = 10.0 / 100.0
    efficiency = max(0.3, min(1.0, 0.3 + 0.7 * rel_age / 0.03, 1.0 - 0.7 * (rel_age - 0.5) / 0.5))
    assert np.allclose(np.asarray(result.vcmax)[0, 1], 40.0 * 0.5 * efficiency)
    assert np.allclose(np.asarray(result.vcmax)[0, 2], 50.0 * efficiency)
    assert np.allclose(np.asarray(result.vcmax)[0, 3], 60.0)

    nlim = vmax_step(
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        vcmax25=vcmax25,
        n_limfert=n_limfert,
        leaf_timecst=np.ones(nvm, dtype=np.float64) * 100.0,
        leafagecrit=np.ones(nvm, dtype=np.float64) * 100.0,
        pheno_type=pheno_type,
        leaf_tab=leaf_tab,
        ok_laidev=np.zeros(nvm, dtype=bool),
        ok_nlim=True,
        dt_days=0.0,
    )
    assert np.allclose(np.asarray(nlim.vcmax)[0, 2], 50.0 * 0.25 * efficiency)


def test_harvest_agri_step_reduces_non_natural_non_peat_turnover_only():
    npts, nvm = 1, 4
    veget_max = np.asarray([[0.0, 0.5, 0.25, 0.75]], dtype=np.float64)
    turnover = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    bm_to_litter = np.ones_like(turnover) * 9.0
    harvest_parts = [
        ILEAF,
        ISAPABOVE,
        IHEARTABOVE,
        IFRUIT,
        ICARBRES,
        ISAPBELOW,
        IHEARTBELOW,
        IROOT,
        IAGRSAPST,
        IAGRSAPPN,
        IAGRHRTST,
        IAGRHRTPN,
    ]
    for idx, part in enumerate(harvest_parts, start=1):
        turnover[0, 1, part, ICARBON] = float(idx)
        turnover[0, 2, part, ICARBON] = float(idx) * 10.0
        turnover[0, 3, part, ICARBON] = float(idx) * 100.0
    turnover[0, 1, ILEAF, 0] = 1.0
    natural = np.asarray([True, False, True, False], dtype=bool)
    is_peat = np.asarray([False, False, False, True], dtype=bool)

    result = harvest_agri_step(
        veget_max=veget_max,
        bm_to_litter=bm_to_litter,
        turnover_daily=turnover,
        natural=natural,
        is_peat=is_peat,
        frac_turnover_daily=0.55,
    )

    expected_old = sum(float(idx) for idx in range(1, len(harvest_parts) + 1))
    assert np.allclose(np.asarray(result.above_old)[0, 1], expected_old)
    assert np.allclose(np.asarray(result.harvest_above), [0.5 * expected_old * 0.45])
    assert np.allclose(np.asarray(result.turnover_daily)[0, 1, harvest_parts, ICARBON], turnover[0, 1, harvest_parts, ICARBON] * 0.55)
    assert np.allclose(np.asarray(result.turnover_daily)[0, 2, harvest_parts, ICARBON], turnover[0, 2, harvest_parts, ICARBON])
    assert np.allclose(np.asarray(result.turnover_daily)[0, 3, harvest_parts, ICARBON], turnover[0, 3, harvest_parts, ICARBON])
    assert np.allclose(np.asarray(result.bm_to_litter), bm_to_litter)


def test_grassland_role_selection_inactive_gate_skips_management_roles():
    """Fortran: stomate_lpj.f90::StomateLpj lines 1321-1353."""

    result = grassland_role_selection_step(
        enable_grazing=False,
        is_grassland_manag=np.zeros(4, dtype=bool),
        is_grassland_cut=np.zeros(4, dtype=bool),
        is_grassland_grazed=np.zeros(4, dtype=bool),
        is_c4=np.zeros(4, dtype=bool),
        is_tree=np.zeros(4, dtype=bool),
        natural=np.zeros(4, dtype=bool),
    )

    assert result.enabled is False
    assert result.mauto_c3 == -1


def test_grassland_role_selection_matches_fortran_c3_c4_role_masks():
    """Fortran: grassland_management.f90::main_grassland_management lines 970-1037."""

    nvm = 10
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_grassland_cut = np.zeros(nvm, dtype=bool)
    is_grassland_grazed = np.zeros(nvm, dtype=bool)
    is_c4 = np.zeros(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag[[1, 3, 5, 7]] = True
    is_grassland_cut[[2, 6]] = True
    is_grassland_grazed[[3, 7]] = True
    natural[[4, 8]] = True
    is_c4[[5, 6, 7, 8]] = True

    result = grassland_role_selection_step(
        enable_grazing=True,
        is_grassland_manag=is_grassland_manag,
        is_grassland_cut=is_grassland_cut,
        is_grassland_grazed=is_grassland_grazed,
        is_c4=is_c4,
        is_tree=is_tree,
        natural=natural,
    )

    assert result.enabled is True
    assert (result.mauto_c3, result.mcut_c3, result.mgraze_c3, result.mnatural_c3) == (1, 2, 3, 4)
    assert (result.mauto_c4, result.mcut_c4, result.mgraze_c4, result.mnatural_c4) == (5, 6, 7, 8)


def test_grassland_role_selection_rejects_missing_role_instead_of_silent_bare_soil_fallback():
    nvm = 5
    masks = {
        "is_grassland_manag": np.asarray([False, True, False, False, False]),
        "is_grassland_cut": np.asarray([False, False, True, False, False]),
        "is_grassland_grazed": np.asarray([False, False, False, True, False]),
        "is_c4": np.zeros(nvm, dtype=bool),
        "is_tree": np.zeros(nvm, dtype=bool),
        "natural": np.asarray([False, False, False, False, True]),
    }

    with pytest.raises(ValueError, match="missing grassland management role"):
        grassland_role_selection_step(enable_grazing=True, **masks)

    fallback = grassland_role_selection_step(enable_grazing=True, reject_missing=False, **masks)
    assert fallback.mauto_c4 == 0
    assert fallback.mcut_c4 == 0


def test_grassland_user_cut_schedule_flags_due_managed_unverified_cut():
    """Fortran: grassland_management.f90::main_grassland_management lines 1872-1897."""

    npts, nvm, nstocking = 2, 5, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[:, :, ILEAF, ICARBON] = np.arange(npts * nvm, dtype=np.float64).reshape(npts, nvm) + 10.0
    tcut = np.ones((npts, nvm, nstocking), dtype=np.float64) * 200.0
    tcut[0, 2, 0] = 119.3
    tcut[1, 2, 0] = 118.0
    tcut[0, 3, 0] = 119.3
    tcut[0, 2, 1] = 119.3
    tcut_verif = np.zeros((npts, nvm, nstocking), dtype=bool)
    tcut_verif[0, 2, 1] = True
    compt_cut = np.ones((npts, nvm), dtype=np.int32)
    when_growthinit_cut = np.ones((npts, nvm), dtype=np.float64) * 4.0
    is_grassland_manag = np.asarray([False, False, True, False, False], dtype=bool)

    result = grassland_user_cut_schedule_step(
        tjulian=120.0,
        dt=0.5,
        tcut=tcut,
        tcut_verif=tcut_verif,
        compt_cut=compt_cut,
        when_growthinit_cut=when_growthinit_cut,
        biomass=biomass,
        is_grassland_manag=is_grassland_manag,
    )

    flags = np.asarray(result.flag_cutting_by_stocking)
    assert flags.shape == (nstocking, npts, nvm)
    assert flags[0, 0, 2] == 1
    assert flags[0, 1, 2] == 0
    assert flags[0, 0, 3] == 0
    assert flags[1, 0, 2] == 0
    assert np.asarray(result.tcut_verif)[0, 2, 0]
    assert np.asarray(result.tcut_verif)[0, 2, 1]
    assert np.asarray(result.compt_cut)[0, 2] == 2
    assert np.asarray(result.compt_cut)[1, 2] == 1
    assert np.allclose(np.asarray(result.when_growthinit_cut)[0, 2], 0.0)
    assert np.allclose(np.asarray(result.when_growthinit_cut)[1, 2], 4.5)
    assert np.allclose(np.asarray(result.lm_before), biomass[:, :, ILEAF, ICARBON])


def test_grassland_user_cut_schedule_inactive_modes_do_not_touch_state():
    """Fortran: grassland_management.f90::main_grassland_management lines 1872-1877."""

    npts, nvm, nstocking = 1, 3, 1
    tcut = np.ones((npts, nvm, nstocking), dtype=np.float64) * 10.0
    tcut_verif = np.zeros((npts, nvm, nstocking), dtype=bool)
    compt_cut = np.ones((npts, nvm), dtype=np.int32) * 3
    when_growthinit_cut = np.ones((npts, nvm), dtype=np.float64) * 7.0
    lm_before = np.ones((npts, nvm), dtype=np.float64) * 11.0
    biomass = np.ones((npts, nvm, NPARTS, 1), dtype=np.float64) * 5.0

    result = grassland_user_cut_schedule_step(
        tjulian=10.0,
        dt=1.0,
        tcut=tcut,
        tcut_verif=tcut_verif,
        compt_cut=compt_cut,
        when_growthinit_cut=when_growthinit_cut,
        biomass=biomass,
        is_grassland_manag=np.asarray([False, True, False], dtype=bool),
        lm_before=lm_before,
        f_autogestion=1,
    )

    assert np.count_nonzero(np.asarray(result.flag_cutting_by_stocking)) == 0
    assert np.array_equal(np.asarray(result.tcut_verif), tcut_verif)
    assert np.array_equal(np.asarray(result.compt_cut), compt_cut)
    assert np.allclose(np.asarray(result.when_growthinit_cut), when_growthinit_cut)
    assert np.allclose(np.asarray(result.lm_before), lm_before)


def _grassland_pre_animal_status_fixture():
    npts, nvm, ncut = 1, 4, 3
    return {
        "dt": 1.0,
        "tjulian": 120,
        "t2m_daily": np.asarray([285.0], dtype=np.float64),
        "tsoil": np.asarray([281.0], dtype=np.float64),
        "new_day": True,
        "new_year": False,
        "regcount": np.asarray([[1, 1, 2, 1]], dtype=np.int32),
        "tcut": np.zeros((npts, nvm, ncut), dtype=np.float64),
        "devstage": np.asarray([[9.0, 0.0, 0.5, 0.0]], dtype=np.float64),
        "tgrowth": np.zeros((npts, nvm), dtype=np.float64),
        "tcut0": np.zeros((npts, nvm), dtype=np.float64),
        "tacumm": np.zeros((npts,), dtype=np.float64),
        "tsoilcumm": np.zeros((npts,), dtype=np.float64),
        "tacummprev": np.zeros((npts,), dtype=np.float64),
        "tsoilcummprev": np.zeros((npts,), dtype=np.float64),
        "tamean1": np.asarray([280.0], dtype=np.float64),
        "tamean2": np.asarray([280.0], dtype=np.float64),
        "tamean3": np.asarray([280.0], dtype=np.float64),
        "tamean4": np.asarray([280.0], dtype=np.float64),
        "tamean5": np.asarray([280.0], dtype=np.float64),
        "tamean6": np.asarray([280.0], dtype=np.float64),
        "tameand": np.asarray([280.0], dtype=np.float64),
        "tameanw": np.zeros((npts,), dtype=np.float64),
        "tsoilmeand": np.zeros((npts,), dtype=np.float64),
        "trep": 279.0,
        "tbase": 278.0,
        "tasumrep": 100.0,
    }


def test_grassland_pre_animal_status_updates_devstage_temperature_memory_and_tgrowth():
    """Fortran: grassland_management.f90::cal_devstage/cal_tgrowth lines 2611-2712."""

    args = _grassland_pre_animal_status_fixture()
    args["tcut"][0, 2, 0] = 100.0

    result = grassland_pre_animal_status_step(**args)

    increment = (285.0 - 278.0) / 100.0
    np.testing.assert_allclose(np.asarray(result.tacumm), [285.0])
    np.testing.assert_allclose(np.asarray(result.tsoilcumm), [281.0])
    np.testing.assert_allclose(np.asarray(result.tamean6), [280.0])
    np.testing.assert_allclose(np.asarray(result.tameand), [285.0])
    np.testing.assert_allclose(np.asarray(result.tameanw), [(280.0 * 6.0 + 285.0) / 7.0])
    assert np.allclose(np.asarray(result.devstage)[0, 0], 9.0)
    assert np.allclose(np.asarray(result.devstage)[0, 1], increment)
    assert np.allclose(np.asarray(result.devstage)[0, 2], 0.5 + increment)
    assert np.allclose(np.asarray(result.tcut0)[0, 1], 120.0)
    assert np.allclose(np.asarray(result.tgrowth)[0, 1], 0.0)
    assert np.allclose(np.asarray(result.tgrowth)[0, 2], 20.0)


def test_grassland_pre_animal_status_new_year_resets_devstage_tgrowth_and_tcut0():
    """Fortran: grassland_management.f90::Main_appl_pre_animal and cal_* new_year branches."""

    args = _grassland_pre_animal_status_fixture()
    args["new_year"] = True
    args["tcut0"][:, :] = 77.0
    args["tgrowth"][:, :] = 3.0

    result = grassland_pre_animal_status_step(**args)

    np.testing.assert_allclose(np.asarray(result.tcut0), np.zeros_like(args["tcut0"]))
    np.testing.assert_allclose(np.asarray(result.devstage), np.zeros_like(args["devstage"]))
    np.testing.assert_allclose(np.asarray(result.tgrowth), np.zeros_like(args["tgrowth"]))


def _auto_cut_schedule_fixture():
    npts, nvm, ngmean = 2, 5, 3
    return {
        "countschedule": np.ones((npts, nvm), dtype=np.int32),
        "compt_cut": np.zeros((npts, nvm), dtype=np.int32),
        "when_growthinit_cut": np.ones((npts, nvm), dtype=np.float64) * 2.0,
        "nanimal": np.zeros((npts, nvm, 1), dtype=np.float64),
        "cuttingend": np.zeros((npts, nvm), dtype=np.int32),
        "tgrowth": np.ones((npts, nvm), dtype=np.float64) * 50.0,
        "gmean": np.ones((npts, nvm, ngmean), dtype=np.float64),
        "lai": np.ones((npts, nvm), dtype=np.float64) * 3.0,
        "devstage": np.ones((npts, nvm), dtype=np.float64) * 1.4,
        "gmeanslope": np.ones((npts, nvm), dtype=np.float64) * 0.01,
        "mugmean": np.ones((npts, nvm), dtype=np.float64),
        "tasum": np.ones((npts, nvm), dtype=np.float64) * 600.0,
        "is_grassland_manag": np.asarray([False, True, False, True, False], dtype=bool),
        "is_grassland_cut": np.asarray([False, False, True, False, False], dtype=bool),
        "is_grassland_grazed": np.asarray([False, False, False, True, False], dtype=bool),
    }


def test_grassland_auto_cut_schedule_growth_phase_flags_automanaged_pfts_only():
    """Fortran: grassland_management.f90::main_grassland_management lines 1975-1993."""

    args = _auto_cut_schedule_fixture()
    args["nanimal"][1, 1, 0] = 1.0
    args["is_grassland_manag"][2] = True

    result = grassland_auto_cut_schedule_step(
        dt=0.5,
        new_day=True,
        phase="growth",
        f_autogestion=1,
        f_postauto=0,
        **args,
    )

    flags = np.asarray(result.flag_cutting)
    assert flags[0, 1] == 1
    assert flags[1, 1] == 0
    assert flags[0, 2] == 0
    assert flags[0, 3] == 0
    assert np.asarray(result.countschedule)[0, 1] == 2
    assert np.asarray(result.compt_cut)[0, 1] == 1
    assert np.allclose(np.asarray(result.when_growthinit_cut)[0, 1], 0.0)
    assert np.allclose(np.asarray(result.when_growthinit_cut)[1, 1], 2.5)


def test_grassland_auto_cut_schedule_tasum_phase_flags_postauto_mcut_roles():
    """Fortran: grassland_management.f90::main_grassland_management lines 2296-2314."""

    args = _auto_cut_schedule_fixture()
    args["devstage"][:, :] = 1.5
    args["tasum"][1, 3] = 100.0
    args["lai"][0, 1] = 2.0

    result = grassland_auto_cut_schedule_step(
        dt=0.5,
        new_day=True,
        phase="tasum",
        f_autogestion=3,
        f_postauto=0,
        mcut_c3=1,
        mcut_c4=3,
        **args,
    )

    flags = np.asarray(result.flag_cutting)
    assert flags[0, 1] == 0
    assert flags[1, 1] == 1
    assert flags[0, 3] == 1
    assert flags[1, 3] == 0
    assert np.asarray(result.countschedule)[1, 1] == 2
    assert np.asarray(result.compt_cut)[0, 3] == 1
    assert np.allclose(np.asarray(result.when_growthinit_cut)[0, 3], 0.0)
    assert np.allclose(np.asarray(result.when_growthinit_cut)[0, 0], 2.0)


def test_grassland_auto_cut_schedule_skips_when_not_new_day():
    """Fortran: grassland_management.f90::main_grassland_management line 1957."""

    args = _auto_cut_schedule_fixture()
    result = grassland_auto_cut_schedule_step(
        dt=0.5,
        new_day=False,
        phase="growth",
        f_autogestion=1,
        f_postauto=0,
        **args,
    )

    assert np.count_nonzero(np.asarray(result.flag_cutting)) == 0
    np.testing.assert_array_equal(np.asarray(result.countschedule), args["countschedule"])
    np.testing.assert_array_equal(np.asarray(result.compt_cut), args["compt_cut"])
    np.testing.assert_allclose(np.asarray(result.when_growthinit_cut), args["when_growthinit_cut"])


def test_grassland_cutting_spa_step_updates_flagged_cut_biomass_yield_and_saved_sum():
    """Fortran: grassland_cutting.f90::cutting_spa lines 153-303."""

    npts, nvm, ncut, ngmean = 1, 3, 3, 4
    pft = 1
    flag = np.zeros((npts, nvm), dtype=np.int32)
    flag[0, pft] = 1
    c = np.ones((npts, nvm), dtype=np.float64) * 0.2
    n = np.ones((npts, nvm), dtype=np.float64) * 0.03
    mass_factor = 1.0 + (28.5 / 12.0) * c[0, pft] + (62.0 / 14.0) * n[0, pft]
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 0.30 * 1000.0 * 0.45 * mass_factor
    biomass[0, pft, ISAPABOVE, ICARBON] = 0.70 * 1000.0 * 0.45 * mass_factor
    biomass[0, pft, IFRUIT, ICARBON] = 0.20 * 1000.0 * 0.45 * mass_factor
    wshtotcutinit = np.zeros((npts, nvm, ncut), dtype=np.float64)
    wshtotcutinit[0, pft, 1] = 0.40 * mass_factor
    regcount = np.zeros((npts, nvm), dtype=np.int32)
    regcount[0, pft] = 1
    wshtotsum = np.zeros((npts, nvm), dtype=np.float64)
    wshtotsumprev = np.zeros((npts, nvm), dtype=np.float64)

    result = grassland_cutting_spa_step(
        tjulian=120,
        flag_cutting=flag,
        wshtotcutinit=wshtotcutinit,
        lcutinit=np.zeros_like(wshtotcutinit),
        wsh=np.ones((npts, nvm), dtype=np.float64),
        wshtot=np.ones((npts, nvm), dtype=np.float64) * 1.4,
        wr=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        c=c,
        n=n,
        napo=np.ones((npts, nvm), dtype=np.float64) * 0.01,
        nsym=np.ones((npts, nvm), dtype=np.float64) * 0.02,
        fn=np.ones((npts, nvm), dtype=np.float64) * 0.03,
        nel=np.zeros((npts, nvm), dtype=np.float64),
        biomass=biomass,
        devstage=np.ones((npts, nvm), dtype=np.float64) * 0.5,
        regcount=regcount,
        gmean=np.ones((npts, nvm, ngmean), dtype=np.float64) * 9.0,
        tasum=np.ones((npts, nvm), dtype=np.float64) * 100.0,
        tgrowth=np.ones((npts, nvm), dtype=np.float64) * 8.0,
        lai=np.ones((npts, nvm), dtype=np.float64),
        tcut=np.zeros((npts, nvm, ncut), dtype=np.float64),
        tcut_modif=np.zeros((npts, nvm, ncut), dtype=np.float64),
        wshtotsum=wshtotsum,
        controle_azote_sum=np.zeros((npts, nvm), dtype=np.float64),
        wshtotsumprev=wshtotsumprev,
        f_autogestion=1,
    )

    cut_init = 0.40
    retained_shoot = cut_init
    expected_leaf = 0.1 * retained_shoot * 1000.0 * 0.45 * mass_factor
    expected_stem = 0.9 * retained_shoot * 1000.0 * 0.45 * mass_factor
    expected_loss = (1.4 - 0.40 * mass_factor) * 0.05
    expected_regrowth_yield = (1.4 - 0.40 * mass_factor) * 0.95

    assert np.allclose(np.asarray(result.wshcutinit)[0, pft], cut_init)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ILEAF, ICARBON], expected_leaf)
    assert np.allclose(np.asarray(result.biomass)[0, pft, ISAPABOVE, ICARBON], expected_stem)
    assert np.allclose(np.asarray(result.biomass)[0, pft, IFRUIT, ICARBON], 0.0)
    assert np.allclose(np.asarray(result.devstage)[0, pft], 0.0)
    assert np.asarray(result.regcount)[0, pft] == 2
    assert np.allclose(np.asarray(result.tasum)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.tgrowth)[0, pft], 0.0)
    assert np.allclose(np.asarray(result.loss)[0, pft], expected_loss)
    assert np.allclose(np.asarray(result.lossc)[0, pft], expected_loss * 0.45)
    assert np.allclose(np.asarray(result.lossn)[0, pft], expected_loss * (0.03 + 0.03))
    assert np.allclose(np.asarray(result.tlossstart)[0, pft], 120.0)
    assert np.allclose(np.asarray(result.tcut)[0, pft, 1], 120.0)
    assert np.allclose(np.asarray(result.gmean)[0, pft, :], 0.0)
    assert np.allclose(np.asarray(result.wshtotsum)[0, pft], expected_regrowth_yield)
    assert np.allclose(np.asarray(result.wshtotsumprev)[0, pft], expected_regrowth_yield)
    assert np.allclose(np.asarray(result.controle_azote_sum)[0, pft], expected_regrowth_yield)


def test_product_pool_aging_step_matches_fortran_rollover_and_flux_scaling():
    prod10 = np.zeros((1, 11, 2), dtype=np.float64)
    prod100 = np.zeros((1, 101, 2), dtype=np.float64)
    flux10 = np.zeros((1, 10, 2), dtype=np.float64)
    flux100 = np.zeros((1, 100, 2), dtype=np.float64)

    prod10[0, :, 0] = np.arange(50.0, 61.0)
    prod10[0, :, 1] = np.arange(0.2, 11.2)
    flux10[0, :, 0] = np.arange(1.0, 11.0)
    flux10[0, :, 1] = np.linspace(0.1, 1.0, 10)

    prod100[0, :, 0] = np.arange(200.0, 301.0)
    prod100[0, :, 1] = np.linspace(0.5, 100.5, 101)
    flux100[0, :, 0] = np.arange(0.25, 25.25, 0.25)
    flux100[0, :, 1] = np.linspace(0.02, 2.0, 100)

    convflux = np.asarray([[12.0, 18.0]], dtype=np.float64)
    cflux_prod10 = np.asarray([[3.0, 4.0]], dtype=np.float64)
    cflux_prod100 = np.asarray([[5.0, 6.0]], dtype=np.float64)

    expected_prod10 = prod10.copy()
    expected_flux10 = flux10.copy()
    expected_cflux_prod10 = cflux_prod10.copy()
    for l in range(9):
        m = 10 - l
        expected_cflux_prod10 += flux10[:, m - 1, :]
        expected_prod10[:, m, :] = prod10[:, m - 1, :] - flux10[:, m - 2, :]
        expected_flux10[:, m - 1, :] = flux10[:, m - 2, :]
        expected_prod10[:, m, :] = np.where(expected_prod10[:, m, :] < 1.0, 0.0, expected_prod10[:, m, :])
    expected_cflux_prod10 += flux10[:, 0, :]
    expected_flux10[:, 0, :] = 0.1 * prod10[:, 0, :]
    expected_prod10[:, 1, :] = prod10[:, 0, :]
    expected_prod10[:, 0, :] = 0.0

    expected_prod100 = prod100.copy()
    expected_flux100 = flux100.copy()
    expected_cflux_prod100 = cflux_prod100.copy()
    for l in range(99):
        m = 100 - l
        expected_cflux_prod100 += flux100[:, m - 1, :]
        expected_prod100[:, m, :] = prod100[:, m - 1, :] - flux100[:, m - 2, :]
        expected_flux100[:, m - 1, :] = flux100[:, m - 2, :]
        expected_prod100[:, m, :] = np.where(expected_prod100[:, m, :] < 1.0, 0.0, expected_prod100[:, m, :])
    expected_cflux_prod100 += flux100[:, 0, :]
    expected_flux100[:, 0, :] = 0.01 * prod100[:, 0, :]
    expected_prod100[:, 1, :] = prod100[:, 0, :]
    expected_prod100[:, 0, :] = 0.0

    scale = 30.0 / 365.0
    result = product_pool_aging_step(
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        convflux=convflux,
        cflux_prod10=cflux_prod10,
        cflux_prod100=cflux_prod100,
        dt_days=30.0,
        one_year=365.0,
    )

    assert np.allclose(np.asarray(result.prod10), expected_prod10)
    assert np.allclose(np.asarray(result.prod100), expected_prod100)
    assert np.allclose(np.asarray(result.flux10), expected_flux10)
    assert np.allclose(np.asarray(result.flux100), expected_flux100)
    assert np.allclose(np.asarray(result.convflux), convflux * scale)
    assert np.allclose(np.asarray(result.cflux_prod10), expected_cflux_prod10 * scale)
    assert np.allclose(np.asarray(result.cflux_prod100), expected_cflux_prod100 * scale)


def _lcchange_main_leak_fixture():
    npts, nvm, nelements, ndeep, ndoc, npool = 1, 4, 1, 2, 2, 3
    veget_old = np.asarray([[0.0, 0.4, 0.1, 0.0]], dtype=np.float64)
    veget_new = np.asarray([[0.0, 0.2, 0.3, 0.0]], dtype=np.float64)
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    for part in range(NPARTS):
        biomass[0, 1, part, ICARBON] = 10.0 + part
        biomass[0, 2, part, ICARBON] = 2.0 + 0.5 * part
    ind = np.asarray([[0.0, 4.0, 0.5, 0.0]], dtype=np.float64)
    age = np.asarray([[0.0, 7.0, 5.0, 0.0]], dtype=np.float64)
    pft_present = np.asarray([[False, True, True, False]])
    senescence = np.asarray([[False, True, True, False]])
    when_growthinit = np.asarray([[0.0, 3.0, 4.0, 0.0]], dtype=np.float64)
    everywhere = np.asarray([[0.0, 1.0, 0.5, 0.0]], dtype=np.float64)
    co2_to_bm = np.asarray([[0.0, 0.0, 0.2, 0.0]], dtype=np.float64)
    bm_to_litter = np.ones_like(biomass) * 0.25
    turnover_daily = np.ones_like(biomass) * 0.05
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[2, :, ICARBON] = np.arange(1.0, NPARTS + 1.0)
    cn_ind = np.asarray([[0.0, 0.5, 1.0, 0.0]], dtype=np.float64)
    flux10 = np.zeros((npts, 10), dtype=np.float64)
    flux100 = np.zeros((npts, 100), dtype=np.float64)
    flux10[0, 0] = 0.5
    flux100[0, 0] = 0.25
    prod10 = np.zeros((npts, 11), dtype=np.float64)
    prod100 = np.zeros((npts, 101), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)
    leaf_frac[0, :, 0] = 1.0
    npp_longterm = np.ones((npts, nvm), dtype=np.float64) * 3.0
    lm_lastyearmax = np.ones((npts, nvm), dtype=np.float64) * 4.0
    litter_avail = np.ones((npts, NLITT, nvm), dtype=np.float64) * 0.1
    litter_not_avail = np.ones_like(litter_avail) * 0.2
    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64) * 6.0
    deep = np.ones((npts, ndeep, nvm), dtype=np.float64)
    fuel_1hr = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 1.0
    fuel_10hr = np.ones_like(fuel_1hr) * 2.0
    fuel_100hr = np.ones_like(fuel_1hr) * 3.0
    fuel_1000hr = np.ones_like(fuel_1hr) * 4.0
    litter_above = np.zeros((npts, NLITT, nvm, nelements), dtype=np.float64)
    litter_below = np.zeros((npts, NLITT, nvm, ndeep, nelements), dtype=np.float64)
    for litt in range(NLITT):
        litter_above[0, litt, 1, ICARBON] = 20.0 + litt
        litter_above[0, litt, 2, ICARBON] = 4.0 + litt
        for layer in range(ndeep):
            litter_below[0, litt, 1, layer, ICARBON] = 30.0 + 10.0 * litt + layer
            litter_below[0, litt, 2, layer, ICARBON] = 5.0 + 10.0 * litt + layer
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    for carb in range(NCARB):
        for layer in range(ndeep):
            carbon_32l[0, carb, 1, layer] = 100.0 + 10.0 * carb + layer
            carbon_32l[0, carb, 2, layer] = 10.0 + 10.0 * carb + layer
    doc = np.zeros((npts, nvm, ndeep, ndoc, npool, nelements), dtype=np.float64)
    for layer in range(ndeep):
        for doc_idx in range(ndoc):
            for pool in range(npool):
                doc[0, 1, layer, doc_idx, pool, ICARBON] = 200.0 + 10.0 * layer + 3.0 * doc_idx + pool
                doc[0, 2, layer, doc_idx, pool, ICARBON] = 20.0 + 10.0 * layer + 3.0 * doc_idx + pool
    return {
        "dt_days": 30.0,
        "veget_max": veget_new,
        "veget_max_old": veget_old,
        "biomass": biomass,
        "ind": ind,
        "age": age,
        "pft_present": pft_present,
        "senescence": senescence,
        "when_growthinit": when_growthinit,
        "everywhere": everywhere,
        "co2_to_bm": co2_to_bm,
        "bm_to_litter": bm_to_litter,
        "turnover_daily": turnover_daily,
        "bm_sapl": bm_sapl,
        "cn_ind": cn_ind,
        "flux10": flux10,
        "flux100": flux100,
        "prod10": prod10,
        "prod100": prod100,
        "leaf_frac": leaf_frac,
        "npp_longterm": npp_longterm,
        "lm_lastyearmax": lm_lastyearmax,
        "litter_avail": litter_avail,
        "litter_not_avail": litter_not_avail,
        "carbon": carbon,
        "deepC_a": deep,
        "deepC_s": deep + 1.0,
        "deepC_p": deep + 2.0,
        "fuel_1hr": fuel_1hr,
        "fuel_10hr": fuel_10hr,
        "fuel_100hr": fuel_100hr,
        "fuel_1000hr": fuel_1000hr,
        "litter_above": litter_above,
        "litter_below": litter_below,
        "carbon_32l": carbon_32l,
        "doc": doc,
        "cn_sapl": np.asarray([1.0, 0.5, 1.0, 1.0], dtype=np.float64),
        "is_tree": np.asarray([False, True, False, False]),
        "coeff_lcchange_1": np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        "coeff_lcchange_10": np.asarray([0.0, 0.4, 0.5, 0.6], dtype=np.float64),
        "coeff_lcchange_100": np.asarray([0.0, 0.7, 0.8, 0.9], dtype=np.float64),
        "is_grassland_manag": np.asarray([False, False, True, False]),
        "is_grassland_grazed": np.asarray([False, False, True, False]),
    }


def test_lcchange_main_leak_step_redistributes_loss_to_increasing_pft_and_products():
    """Fortran: stomate_lcchange.f90::lcchange_main lines 211-494."""

    args = _lcchange_main_leak_fixture()
    result = lcchange_main_leak_step(**args)

    old_cover = args["veget_max_old"][0, 2]
    new_cover = args["veget_max"][0, 2]
    delta = new_cover - old_cover
    scale = args["dt_days"] / 365.0
    dilu_lit_above = args["litter_above"][0, :, 1, :]
    dilu_lit_below = args["litter_below"][0, :, 1, :, :]
    dilu_soil = args["carbon_32l"][0, :, 1, :]
    biomass_loss = args["biomass"][0, 1, :, :]
    expected_biomass_pft2 = args["biomass"][0, 2].copy()
    expected_co2 = args["co2_to_bm"].copy()
    for part in range(NPARTS):
        bm_new = delta * args["bm_sapl"][2, part, ICARBON]
        bm_new = min(bm_new, args["biomass"][0, 2, part, ICARBON] * delta)
        expected_biomass_pft2[part, ICARBON] = (args["biomass"][0, 2, part, ICARBON] * old_cover + bm_new) / new_cover
        expected_co2[0, 2] += bm_new * scale / new_cover
    expected_litter_above = (args["litter_above"][0, :, 2, :] * old_cover + dilu_lit_above * delta) / new_cover
    expected_litter_below = (args["litter_below"][0, :, 2, :, :] * old_cover + dilu_lit_below * delta) / new_cover
    expected_carbon_32l = (args["carbon_32l"][0, :, 2, :] * old_cover + dilu_soil * delta) / new_cover
    expected_doc = args["doc"][0, 2, :, :, :, ICARBON].copy()
    for layer in range(2):
        for doc_idx in range(2):
            for pool in range(3):
                expected_doc[layer, doc_idx, pool] = (
                    args["doc"][0, 2, layer, doc_idx, pool, ICARBON] * new_cover
                    + args["doc"][0, 1, layer, doc_idx, pool, ICARBON] * delta
                ) / new_cover
    expected_bm_to_litter = args["bm_to_litter"][0, 2].copy()
    for part in [ISAPBELOW, IHEARTBELOW, IROOT, IFRUIT, ICARBRES, ILEAF]:
        expected_bm_to_litter[part, ICARBON] = (
            args["bm_to_litter"][0, 2, part, ICARBON] * old_cover + biomass_loss[part, ICARBON] * delta
        ) / new_cover

    above_loss = sum(args["biomass"][0, 1, part, ICARBON] for part in [ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN])
    expected_prod10_input = args["coeff_lcchange_10"][1] * above_loss * 0.2
    expected_prod100_input = args["coeff_lcchange_100"][1] * above_loss * 0.2

    assert np.allclose(np.asarray(result.biomass)[0, 2], expected_biomass_pft2)
    assert np.allclose(np.asarray(result.co2_to_bm), expected_co2)
    assert np.allclose(np.asarray(result.litter_above)[0, :, 2, :], expected_litter_above)
    assert np.allclose(np.asarray(result.litter_below)[0, :, 2, :, :], expected_litter_below)
    assert np.allclose(np.asarray(result.carbon_32l)[0, :, 2, :], expected_carbon_32l)
    assert np.allclose(np.asarray(result.doc)[0, 2, :, :, :, ICARBON], expected_doc)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, 2], expected_bm_to_litter)
    assert np.allclose(np.asarray(result.age)[0, 2], args["age"][0, 2] * old_cover / new_cover)
    assert np.allclose(np.asarray(result.litter_avail)[0, :, 2], args["litter_avail"][0, :, 2] * old_cover / new_cover)
    assert np.allclose(
        np.asarray(result.litter_not_avail)[0, :, 2],
        np.asarray(result.litter_above)[0, :, 2, ICARBON] - np.asarray(result.litter_avail)[0, :, 2],
    )
    assert np.allclose(np.asarray(result.convflux)[0], args["coeff_lcchange_1"][1] * above_loss * 0.2 * scale)
    assert np.allclose(np.asarray(result.prod10)[0, 1], expected_prod10_input)
    assert np.allclose(np.asarray(result.flux10)[0, 0], 0.1 * expected_prod10_input)
    assert np.allclose(np.asarray(result.prod100)[0, 1], expected_prod100_input)
    assert np.allclose(np.asarray(result.flux100)[0, 0], 0.01 * expected_prod100_input)
    assert np.allclose(np.asarray(result.cflux_prod10)[0], args["flux10"][0, 0] * scale)
    assert np.allclose(np.asarray(result.cflux_prod100)[0], args["flux100"][0, 0] * scale)
    assert np.allclose(np.asarray(result.fuel_1hr)[0, 2], np.asarray(result.litter_above)[0, :, 2, :] * 0.1)
    assert np.allclose(np.asarray(result.fuel_1000hr)[0, 2], np.asarray(result.litter_above)[0, :, 2, :] * 0.4)


def test_lcchange_main_leak_step_ok_pc_redistributes_vertical_deep_carbon():
    """Fortran: stomate_lcchange.f90::lcchange_main lines 259-267 and 356-363."""

    args = _lcchange_main_leak_fixture()
    args["deepC_a"][0, :, 1] = [10.0, 20.0]
    args["deepC_s"][0, :, 1] = [30.0, 40.0]
    args["deepC_p"][0, :, 1] = [50.0, 60.0]
    args["deepC_a"][0, :, 2] = [1.0, 2.0]
    args["deepC_s"][0, :, 2] = [3.0, 4.0]
    args["deepC_p"][0, :, 2] = [5.0, 6.0]

    result = lcchange_main_leak_step(**args, ok_pc=True)

    old_cover = args["veget_max_old"][0, 2]
    new_cover = args["veget_max"][0, 2]
    delta = new_cover - old_cover
    expected_a = (args["deepC_a"][0, :, 2] * old_cover + args["deepC_a"][0, :, 1] * delta) / new_cover
    expected_s = (args["deepC_s"][0, :, 2] * old_cover + args["deepC_s"][0, :, 1] * delta) / new_cover
    expected_p = (args["deepC_p"][0, :, 2] * old_cover + args["deepC_p"][0, :, 1] * delta) / new_cover

    np.testing.assert_allclose(np.asarray(result.deepC_a)[0, :, 2], expected_a)
    np.testing.assert_allclose(np.asarray(result.deepC_s)[0, :, 2], expected_s)
    np.testing.assert_allclose(np.asarray(result.deepC_p)[0, :, 2], expected_p)
    np.testing.assert_allclose(np.asarray(result.carbon_32l), args["carbon_32l"])
    np.testing.assert_allclose(np.asarray(result.doc), args["doc"])


def test_lcchange_main_leak_step_empty_pft_preserves_mict_leak_arrays_like_source():
    """Fortran: stomate_lcchange.f90::lcchange_main lines 397-426."""

    args = _lcchange_main_leak_fixture()
    args["veget_max"] = np.asarray([[0.0, 0.0, 0.5, 0.0]], dtype=np.float64)
    result = lcchange_main_leak_step(**args)

    assert np.allclose(np.asarray(result.ind)[0, 1], 0.0)
    assert np.allclose(np.asarray(result.biomass)[0, 1], 0.0)
    assert not bool(np.asarray(result.pft_present)[0, 1])
    assert not bool(np.asarray(result.senescence)[0, 1])
    assert np.allclose(np.asarray(result.when_growthinit)[0, 1], -9999.0)
    assert np.allclose(np.asarray(result.everywhere)[0, 1], 0.0)
    assert np.allclose(np.asarray(result.carbon)[0, :, 1], 0.0)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, 1], 0.0)
    assert np.allclose(np.asarray(result.turnover_daily)[0, 1], 0.0)
    assert np.allclose(np.asarray(result.litter_above)[0, :, 1, :], args["litter_above"][0, :, 1, :])
    assert np.allclose(np.asarray(result.carbon_32l)[0, :, 1, :], args["carbon_32l"][0, :, 1, :])
    assert np.allclose(np.asarray(result.doc)[0, 1, :, :, :, :], args["doc"][0, 1, :, :, :, :])


def test_lcchange_main_leak_step_ok_pc_empty_pft_clears_deep_carbon():
    """Fortran: stomate_lcchange.f90::lcchange_main lines 397-424."""

    args = _lcchange_main_leak_fixture()
    args["veget_max"] = np.asarray([[0.0, 0.0, 0.5, 0.0]], dtype=np.float64)
    args["deepC_a"][0, :, 1] = [10.0, 20.0]
    args["deepC_s"][0, :, 1] = [30.0, 40.0]
    args["deepC_p"][0, :, 1] = [50.0, 60.0]

    result = lcchange_main_leak_step(**args, ok_pc=True)

    np.testing.assert_allclose(np.asarray(result.deepC_a)[0, :, 1], 0.0)
    np.testing.assert_allclose(np.asarray(result.deepC_s)[0, :, 1], 0.0)
    np.testing.assert_allclose(np.asarray(result.deepC_p)[0, :, 1], 0.0)
    old_cover = args["veget_max_old"][0, 2]
    new_cover = args["veget_max"][0, 2]
    delta = new_cover - old_cover
    expected_pft2 = (args["deepC_a"][0, :, 2] * old_cover + args["deepC_a"][0, :, 1] * delta) / new_cover
    np.testing.assert_allclose(np.asarray(result.deepC_a)[0, :, 2], expected_pft2)


def _lcchange_deffire_fixture():
    npts, nvm, nelements = 1, 4, 1
    veget_old = np.asarray([[0.0, 0.4, 0.1, 0.0]], dtype=np.float64)
    veget_new = np.asarray([[0.0, 0.2, 0.3, 0.0]], dtype=np.float64)
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    for part in range(NPARTS):
        biomass[0, 1, part, ICARBON] = 10.0 + part
        biomass[0, 2, part, ICARBON] = 1.0 + part
    litter = np.zeros((npts, NLITT, nvm, NLEVS, nelements), dtype=np.float64)
    litter[0, :, 1, IABOVE, ICARBON] = [20.0, 30.0]
    litter[0, :, 1, IBELOW, ICARBON] = [40.0, 50.0]
    litter[0, :, 2, IABOVE, ICARBON] = [2.0, 3.0]
    litter[0, :, 2, IBELOW, ICARBON] = [4.0, 5.0]
    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64)
    carbon[0, :, 1] = [11.0, 12.0, 13.0]
    carbon[0, :, 2] = [1.0, 2.0, 3.0]
    emilit = np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64)
    emilit[0, 1, :, ICARBON] = [1.0, 2.0]
    emibio = np.zeros_like(biomass)
    emibio[0, 1, ILEAF, ICARBON] = 0.5
    emibio[0, 1, ISAPABOVE, ICARBON] = 1.0
    emibio[0, 1, IHEARTABOVE, ICARBON] = 2.0
    emibio[0, 1, IROOT, ICARBON] = 0.25
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[2, :, ICARBON] = np.arange(1.0, NPARTS + 1.0)
    cn_ind = np.asarray([[0.0, 0.5, 1.0, 0.0]], dtype=np.float64)
    fuel_1hr = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64)
    fuel_10hr = np.ones_like(fuel_1hr) * 2.0
    fuel_100hr = np.ones_like(fuel_1hr) * 3.0
    fuel_1000hr = np.ones_like(fuel_1hr) * 4.0
    return {
        "dt_days": 30.0,
        "veget_max": veget_old,
        "veget_max_new": veget_new,
        "biomass": biomass,
        "ind": np.ones((npts, nvm), dtype=np.float64),
        "age": np.ones((npts, nvm), dtype=np.float64) * 4.0,
        "pft_present": np.ones((npts, nvm), dtype=bool),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64),
        "everywhere": np.ones((npts, nvm), dtype=np.float64),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "bm_to_litter": np.ones_like(biomass) * 0.1,
        "turnover_daily": np.ones_like(biomass) * 0.01,
        "bm_sapl": bm_sapl,
        "cn_ind": cn_ind,
        "flux10": np.zeros((npts, 10), dtype=np.float64),
        "flux100": np.zeros((npts, 100), dtype=np.float64),
        "prod10": np.zeros((npts, 11), dtype=np.float64),
        "prod100": np.zeros((npts, 101), dtype=np.float64),
        "leaf_frac": np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64),
        "lm_lastyearmax": np.ones((npts, nvm), dtype=np.float64),
        "litter": litter,
        "litter_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "litter_not_avail": np.zeros((npts, NLITT, nvm), dtype=np.float64),
        "carbon": carbon,
        "deepC_a": np.ones((npts, 2, nvm), dtype=np.float64),
        "deepC_s": np.ones((npts, 2, nvm), dtype=np.float64) * 2.0,
        "deepC_p": np.ones((npts, 2, nvm), dtype=np.float64) * 3.0,
        "fuel_1hr": fuel_1hr,
        "fuel_10hr": fuel_10hr,
        "fuel_100hr": fuel_100hr,
        "fuel_1000hr": fuel_1000hr,
        "lcc": veget_old - veget_new,
        "bafrac_deforest_accu": np.zeros((npts, nvm), dtype=np.float64),
        "emideforest_litter_accu": emilit,
        "emideforest_biomass_accu": emibio,
        "deflitsup_total": np.zeros((npts, nvm), dtype=np.float64),
        "defbiosup_total": np.zeros((npts, nvm), dtype=np.float64),
        "cn_sapl": np.ones(nvm, dtype=np.float64),
        "is_tree": np.asarray([False, True, False, False]),
    }


def test_lcchange_deffire_step_subtracts_fire_emissions_before_dilution_and_products():
    """Fortran: stomate_lcchange.f90::lcchange_deffire lines 649-981."""

    npts, nvm, nelements = 1, 4, 1
    veget_old = np.asarray([[0.0, 0.4, 0.1, 0.0]], dtype=np.float64)
    veget_new = np.asarray([[0.0, 0.2, 0.3, 0.0]], dtype=np.float64)
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    for part in range(NPARTS):
        biomass[0, 1, part, ICARBON] = 10.0 + part
        biomass[0, 2, part, ICARBON] = 1.0 + part
    litter = np.zeros((npts, NLITT, nvm, NLEVS, nelements), dtype=np.float64)
    litter[0, :, 1, IABOVE, ICARBON] = [20.0, 30.0]
    litter[0, :, 1, IBELOW, ICARBON] = [40.0, 50.0]
    litter[0, :, 2, IABOVE, ICARBON] = [2.0, 3.0]
    litter[0, :, 2, IBELOW, ICARBON] = [4.0, 5.0]
    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64)
    carbon[0, :, 1] = [11.0, 12.0, 13.0]
    carbon[0, :, 2] = [1.0, 2.0, 3.0]
    emilit = np.zeros((npts, nvm, NLITT, nelements), dtype=np.float64)
    emilit[0, 1, :, ICARBON] = [1.0, 2.0]
    emibio = np.zeros_like(biomass)
    emibio[0, 1, ILEAF, ICARBON] = 0.5
    emibio[0, 1, ISAPABOVE, ICARBON] = 1.0
    emibio[0, 1, IHEARTABOVE, ICARBON] = 2.0
    emibio[0, 1, IROOT, ICARBON] = 0.25
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[2, :, ICARBON] = np.arange(1.0, NPARTS + 1.0)
    cn_ind = np.asarray([[0.0, 0.5, 1.0, 0.0]], dtype=np.float64)
    flux10 = np.zeros((npts, 10), dtype=np.float64)
    flux100 = np.zeros((npts, 100), dtype=np.float64)
    prod10 = np.zeros((npts, 11), dtype=np.float64)
    prod100 = np.zeros((npts, 101), dtype=np.float64)
    fuel_1hr = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64)
    fuel_10hr = np.ones_like(fuel_1hr) * 2.0
    fuel_100hr = np.ones_like(fuel_1hr) * 3.0
    fuel_1000hr = np.ones_like(fuel_1hr) * 4.0

    result = lcchange_deffire_step(
        dt_days=30.0,
        veget_max=veget_old,
        veget_max_new=veget_new,
        biomass=biomass,
        ind=np.ones((npts, nvm), dtype=np.float64),
        age=np.ones((npts, nvm), dtype=np.float64) * 4.0,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        when_growthinit=np.ones((npts, nvm), dtype=np.float64),
        everywhere=np.ones((npts, nvm), dtype=np.float64),
        co2_to_bm=np.zeros((npts, nvm), dtype=np.float64),
        bm_to_litter=np.ones_like(biomass) * 0.1,
        turnover_daily=np.ones_like(biomass) * 0.01,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=flux10,
        flux100=flux100,
        prod10=prod10,
        prod100=prod100,
        leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        litter=litter,
        litter_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        litter_not_avail=np.zeros((npts, NLITT, nvm), dtype=np.float64),
        carbon=carbon,
        deepC_a=np.ones((npts, 2, nvm), dtype=np.float64),
        deepC_s=np.ones((npts, 2, nvm), dtype=np.float64) * 2.0,
        deepC_p=np.ones((npts, 2, nvm), dtype=np.float64) * 3.0,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        lcc=veget_old - veget_new,
        bafrac_deforest_accu=np.zeros((npts, nvm), dtype=np.float64),
        emideforest_litter_accu=emilit,
        emideforest_biomass_accu=emibio,
        deflitsup_total=np.zeros((npts, nvm), dtype=np.float64),
        defbiosup_total=np.zeros((npts, nvm), dtype=np.float64),
        cn_sapl=np.ones(nvm, dtype=np.float64),
        is_tree=np.asarray([False, True, False, False]),
    )

    delta = 0.2
    new_cover = veget_new[0, 2]
    old_cover = veget_old[0, 2]
    lit_surplus = delta * litter[0, :, 1, IABOVE, ICARBON] - emilit[0, 1, :, ICARBON]
    dilu_above = lit_surplus / delta
    dilu_below = litter[0, :, 1, IBELOW, ICARBON]
    expected_litter_pft2_above = (litter[0, :, 2, IABOVE, ICARBON] * old_cover + dilu_above * delta) / new_cover
    expected_litter_pft2_below = (litter[0, :, 2, IBELOW, ICARBON] * old_cover + dilu_below * delta) / new_cover
    bio_surplus = delta * biomass[0, 1, :, ICARBON] - emibio[0, 1, :, ICARBON]
    biomass_loss = bio_surplus / delta
    expected_bm_to_litter_leaf = 0.1 + biomass_loss[ILEAF] * delta / new_cover
    expected_bm_to_litter_root = 0.1 + biomass_loss[IROOT] * delta / new_cover
    above_remaining = sum(bio_surplus[part] for part in [ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN])
    scale = 30.0 / 365.0

    assert np.allclose(np.asarray(result.deflitsup_total)[0, 1], np.sum(lit_surplus))
    assert np.allclose(np.asarray(result.defbiosup_total)[0, 1], np.sum(bio_surplus))
    assert np.allclose(np.asarray(result.litter)[0, :, 2, IABOVE, ICARBON], expected_litter_pft2_above)
    assert np.allclose(np.asarray(result.litter)[0, :, 2, IBELOW, ICARBON], expected_litter_pft2_below)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, 2, ILEAF, ICARBON], expected_bm_to_litter_leaf)
    assert np.allclose(np.asarray(result.bm_to_litter)[0, 2, IROOT, ICARBON], expected_bm_to_litter_root)
    assert np.allclose(np.asarray(result.prod10)[0, 1], 0.4 * above_remaining)
    assert np.allclose(np.asarray(result.prod100)[0, 1], 0.6 * above_remaining)
    assert np.allclose(np.asarray(result.flux10)[0, 0], 0.1 * 0.4 * above_remaining)
    assert np.allclose(np.asarray(result.convflux)[0], (emibio[0, 1, ISAPABOVE, ICARBON] + emibio[0, 1, IHEARTABOVE, ICARBON]) * scale)
    assert np.allclose(np.asarray(result.veget_max), veget_new)
    assert np.allclose(np.asarray(result.fuel_1hr)[0, 2], np.asarray(result.litter)[0, :, 2, IABOVE, :] * 0.1)
    assert np.allclose(np.asarray(result.fuel_1000hr)[0, 2], np.asarray(result.litter)[0, :, 2, IABOVE, :] * 0.4)


def test_lcchange_deffire_step_ok_pc_redistributes_vertical_deep_carbon():
    """Fortran: stomate_lcchange.f90::lcchange_deffire lines 744-750 and 834-840."""

    args = _lcchange_deffire_fixture()
    args["deepC_a"][0, :, 1] = [10.0, 20.0]
    args["deepC_s"][0, :, 1] = [30.0, 40.0]
    args["deepC_p"][0, :, 1] = [50.0, 60.0]
    args["deepC_a"][0, :, 2] = [1.0, 2.0]
    args["deepC_s"][0, :, 2] = [3.0, 4.0]
    args["deepC_p"][0, :, 2] = [5.0, 6.0]

    result = lcchange_deffire_step(**args, ok_pc=True)

    old_cover = args["veget_max"][0, 2]
    new_cover = args["veget_max_new"][0, 2]
    delta = new_cover - old_cover
    expected_a = (args["deepC_a"][0, :, 2] * old_cover + args["deepC_a"][0, :, 1] * delta) / new_cover
    expected_s = (args["deepC_s"][0, :, 2] * old_cover + args["deepC_s"][0, :, 1] * delta) / new_cover
    expected_p = (args["deepC_p"][0, :, 2] * old_cover + args["deepC_p"][0, :, 1] * delta) / new_cover

    np.testing.assert_allclose(np.asarray(result.deepC_a)[0, :, 2], expected_a)
    np.testing.assert_allclose(np.asarray(result.deepC_s)[0, :, 2], expected_s)
    np.testing.assert_allclose(np.asarray(result.deepC_p)[0, :, 2], expected_p)
    np.testing.assert_allclose(np.asarray(result.carbon)[0, :, 1], args["carbon"][0, :, 1])
    np.testing.assert_allclose(np.asarray(result.carbon)[0, :, 2], args["carbon"][0, :, 2])


def test_lcchange_deffire_step_ok_pc_empty_pft_clears_deep_carbon():
    """Fortran: stomate_lcchange.f90::lcchange_deffire lines 891-893."""

    args = _lcchange_deffire_fixture()
    args["veget_max_new"] = np.asarray([[0.0, 0.0, 0.5, 0.0]], dtype=np.float64)
    args["deepC_a"][0, :, 1] = [10.0, 20.0]
    args["deepC_s"][0, :, 1] = [30.0, 40.0]
    args["deepC_p"][0, :, 1] = [50.0, 60.0]
    args["deepC_a"][0, :, 2] = [1.0, 2.0]

    result = lcchange_deffire_step(**args, ok_pc=True)

    np.testing.assert_allclose(np.asarray(result.deepC_a)[0, :, 1], 0.0)
    np.testing.assert_allclose(np.asarray(result.deepC_s)[0, :, 1], 0.0)
    np.testing.assert_allclose(np.asarray(result.deepC_p)[0, :, 1], 0.0)
    old_cover = args["veget_max"][0, 2]
    new_cover = args["veget_max_new"][0, 2]
    delta = new_cover - old_cover
    expected_pft2 = (args["deepC_a"][0, :, 2] * old_cover + args["deepC_a"][0, :, 1] * delta) / new_cover
    np.testing.assert_allclose(np.asarray(result.deepC_a)[0, :, 2], expected_pft2)


def test_forest_harvest_legacy_step_allocates_wood_products_and_litter():
    npts, nvm, nelements = 2, 4, 2
    point, pft = 1, 2
    frac = 0.25
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    woody_parts = [ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN]
    litter_parts = [ISAPBELOW, IHEARTBELOW, IROOT, IFRUIT, ICARBRES, ILEAF]

    for offset, part in enumerate(woody_parts, start=1):
        biomass[point, pft, part, ICARBON] = offset * 10.0
        biomass[0, pft, part, ICARBON] = 999.0
    for offset, part in enumerate(litter_parts, start=1):
        biomass[point, pft, part, ICARBON] = offset * 2.0
        biomass[point, pft, part, 1] = offset * 3.0

    bm_to_litter_pro = np.ones((NPARTS, nelements), dtype=np.float64)
    convflux = np.asarray([5.0, 7.0], dtype=np.float64)
    prod10 = np.zeros((npts, 11), dtype=np.float64)
    prod100 = np.zeros((npts, 101), dtype=np.float64)
    prod10[point, 0] = 11.0
    prod100[point, 0] = 13.0
    coeff_1 = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    coeff_10 = np.asarray([0.0, 0.4, 0.5, 0.6], dtype=np.float64)
    coeff_100 = np.asarray([0.0, 0.7, 0.8, 0.9], dtype=np.float64)

    expected_above = sum(offset * 10.0 for offset in range(1, len(woody_parts) + 1)) * frac
    result = forest_harvest_legacy_step(
        biomass=biomass,
        bm_to_litter_pro=bm_to_litter_pro,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        point=point,
        pft=pft,
        frac=frac,
        coeff_lcchange_1=coeff_1,
        coeff_lcchange_10=coeff_10,
        coeff_lcchange_100=coeff_100,
    )

    expected_bm_to_litter = bm_to_litter_pro.copy()
    expected_bm_to_litter[litter_parts, :] += biomass[point, pft, litter_parts, :] * frac
    expected_convflux = convflux.copy()
    expected_convflux[point] += coeff_1[pft] * expected_above
    expected_prod10 = prod10.copy()
    expected_prod10[point, 0] += coeff_10[pft] * expected_above
    expected_prod100 = prod100.copy()
    expected_prod100[point, 0] += coeff_100[pft] * expected_above

    assert np.allclose(np.asarray(result.above), expected_above)
    assert np.allclose(np.asarray(result.bm_to_litter_pro), expected_bm_to_litter)
    assert np.allclose(np.asarray(result.convflux), expected_convflux)
    assert np.allclose(np.asarray(result.prod10), expected_prod10)
    assert np.allclose(np.asarray(result.prod100), expected_prod100)


def test_forest_harvest_legacy_step_deforest_fire_uses_remaining_above_wood():
    npts, nvm, nelements = 1, 3, 1
    point, pft = 0, 2
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    deforest_remain = np.zeros_like(biomass)
    woody_parts = [ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN]
    litter_parts = [ISAPBELOW, IHEARTBELOW, IROOT, IFRUIT, ICARBRES, ILEAF]

    for offset, part in enumerate(woody_parts, start=1):
        biomass[point, pft, part, ICARBON] = 1000.0
        deforest_remain[point, pft, part, ICARBON] = offset * 5.0
    for offset, part in enumerate(litter_parts, start=1):
        biomass[point, pft, part, ICARBON] = offset * 4.0

    convflux = np.asarray([17.0], dtype=np.float64)
    prod10 = np.zeros((npts, 11), dtype=np.float64)
    prod100 = np.zeros((npts, 101), dtype=np.float64)
    bm_to_litter_pro = np.zeros((NPARTS, nelements), dtype=np.float64)
    coeff = np.ones(nvm, dtype=np.float64) * 99.0
    frac = 0.5

    result = forest_harvest_legacy_step(
        biomass=biomass,
        bm_to_litter_pro=bm_to_litter_pro,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        point=point,
        pft=pft,
        frac=frac,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        allow_deforest_fire=True,
        deforest_biomass_remain=deforest_remain,
    )

    expected_above = sum(offset * 5.0 for offset in range(1, len(woody_parts) + 1))
    expected_bm_to_litter = bm_to_litter_pro.copy()
    expected_bm_to_litter[litter_parts, :] += biomass[point, pft, litter_parts, :] * frac

    assert np.allclose(np.asarray(result.above), expected_above)
    assert np.allclose(np.asarray(result.convflux), convflux)
    assert np.allclose(np.asarray(result.prod10)[point, 0], 0.4 * expected_above)
    assert np.allclose(np.asarray(result.prod100)[point, 0], 0.6 * expected_above)
    assert np.allclose(np.asarray(result.bm_to_litter_pro), expected_bm_to_litter)


def test_forest_harvest_proxy_pools_step_transfers_litter_fuel_and_lignin():
    npts, nvm, nlevels, nelements = 2, 3, 4, 2
    point, pft = 1, 2
    frac = 0.3
    litter = np.arange(npts * NLITT * nvm * nlevels * nelements, dtype=np.float64).reshape(
        npts, NLITT, nvm, nlevels, nelements
    )
    fuel_1hr = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 2.0
    fuel_10hr = np.ones_like(fuel_1hr) * 3.0
    fuel_100hr = np.ones_like(fuel_1hr) * 4.0
    fuel_1000hr = np.ones_like(fuel_1hr) * 5.0
    lignin_struc = np.zeros((npts, nvm, nlevels), dtype=np.float64)
    lignin_struc[point, pft, :] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    litter_pro = np.ones((NLITT, nlevels, nelements), dtype=np.float64)
    fuel_1hr_pro = np.ones((NLITT, nelements), dtype=np.float64) * 10.0
    fuel_10hr_pro = np.ones_like(fuel_1hr_pro) * 20.0
    fuel_100hr_pro = np.ones_like(fuel_1hr_pro) * 30.0
    fuel_1000hr_pro = np.ones_like(fuel_1hr_pro) * 40.0
    lignin_content_pro = np.ones(nlevels, dtype=np.float64) * 0.5

    result = forest_harvest_proxy_pools_step(
        litter=litter,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        lignin_struc=lignin_struc,
        litter_pro=litter_pro,
        fuel_1hr_pro=fuel_1hr_pro,
        fuel_10hr_pro=fuel_10hr_pro,
        fuel_100hr_pro=fuel_100hr_pro,
        fuel_1000hr_pro=fuel_1000hr_pro,
        lignin_content_pro=lignin_content_pro,
        point=point,
        pft=pft,
        frac=frac,
    )

    assert np.allclose(np.asarray(result.litter_pro), litter_pro + litter[point, :, pft, :, :] * frac)
    assert np.allclose(np.asarray(result.fuel_1hr_pro), fuel_1hr_pro + fuel_1hr[point, pft, :, :] * frac)
    assert np.allclose(np.asarray(result.fuel_10hr_pro), fuel_10hr_pro + fuel_10hr[point, pft, :, :] * frac)
    assert np.allclose(np.asarray(result.fuel_100hr_pro), fuel_100hr_pro + fuel_100hr[point, pft, :, :] * frac)
    assert np.allclose(np.asarray(result.fuel_1000hr_pro), fuel_1000hr_pro + fuel_1000hr[point, pft, :, :] * frac)
    expected_lignin = lignin_content_pro + litter[point, ISTRUCTURAL, pft, :, ICARBON] * frac * lignin_struc[point, pft, :]
    assert np.allclose(np.asarray(result.lignin_content_pro), expected_lignin)


def test_harvest_herb_legacy_step_moves_all_biomass_to_proxy_litter():
    npts, nvm, nelements = 2, 3, 2
    point, pft = 1, 1
    biomass = np.arange(npts * nvm * NPARTS * nelements, dtype=np.float64).reshape(
        npts, nvm, NPARTS, nelements
    )
    bm_to_litter_pro = np.ones((NPARTS, nelements), dtype=np.float64)
    veget_frac = 0.6

    result = harvest_herb_legacy_step(
        biomass=biomass,
        bm_to_litter_pro=bm_to_litter_pro,
        point=point,
        pft=pft,
        veget_frac=veget_frac,
    )

    assert np.allclose(
        np.asarray(result.bm_to_litter_pro),
        bm_to_litter_pro + biomass[point, pft, :, :] * veget_frac,
    )


def test_initialize_proxy_pft_step_matches_fortran_tree_scaling():
    nvm, nelements = 4, 2
    pft = 2
    veget_max_pro = 0.25
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[pft, :, ICARBON] = np.arange(1.0, NPARTS + 1.0)
    bm_sapl[pft, :, 1] = np.arange(101.0, 101.0 + NPARTS)
    cn_sapl = np.asarray([1.0, 1.0, 0.5, 1.0], dtype=np.float64)
    is_tree = np.asarray([False, False, True, False], dtype=bool)
    co2_to_bm_pro = 3.0
    npp_longterm_init = 12.0

    result = initialize_proxy_pft_step(
        pft=pft,
        veget_max_pro=veget_max_pro,
        co2_to_bm_pro=co2_to_bm_pro,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        is_tree=is_tree,
        npp_longterm_init=npp_longterm_init,
    )

    ind = veget_max_pro / cn_sapl[pft]
    assert np.allclose(np.asarray(result.biomass_pro), ind * bm_sapl[pft])
    assert np.allclose(np.asarray(result.co2_to_bm_pro), co2_to_bm_pro + np.sum(ind * bm_sapl[pft, :, ICARBON]))
    assert np.allclose(np.asarray(result.ind_pro), ind * veget_max_pro)
    assert np.allclose(np.asarray(result.age_pro), 0.0)
    assert bool(np.asarray(result.pft_present_pro))
    assert not bool(np.asarray(result.senescence_pro))
    assert np.allclose(np.asarray(result.everywhere_pro), veget_max_pro)
    assert np.allclose(np.asarray(result.npp_longterm_pro), npp_longterm_init * veget_max_pro)
    assert np.allclose(np.asarray(result.lm_lastyearmax_pro), bm_sapl[pft, ILEAF, ICARBON] * ind * veget_max_pro)
    assert np.allclose(np.asarray(result.leaf_frac_pro), [veget_max_pro, 0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(result.leaf_age_pro), [veget_max_pro, 0.0, 0.0, 0.0])


def test_sap_take_step_draws_available_carbon_from_active_age_classes_only():
    npts, nvm, nelements = 1, 4, 2
    point = 0
    age_class_pfts = [1, 2, 3]
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    veget_max = np.asarray([[0.0, 0.5, 0.25, 0.0]], dtype=np.float64)
    biomass[point, 1, ILEAF, ICARBON] = 20.0
    biomass[point, 2, ILEAF, ICARBON] = 40.0
    biomass[point, 3, ILEAF, ICARBON] = 999.0
    biomass[point, 1, IROOT, ICARBON] = 2.0
    biomass[point, 2, IROOT, ICARBON] = 2.0
    biomass[point, 1, ILEAF, 1] = 77.0

    biomass_pro = np.zeros((NPARTS, nelements), dtype=np.float64)
    biomass_pro[ILEAF, ICARBON] = 5.0
    biomass_pro[IROOT, ICARBON] = 10.0
    co2_to_bm_pro = 20.0

    result = sap_take_step(
        biomass=biomass,
        veget_max=veget_max,
        biomass_pro=biomass_pro,
        co2_to_bm_pro=co2_to_bm_pro,
        point=point,
        age_class_pfts=age_class_pfts,
        min_stomate=1.0e-8,
    )

    expected = biomass.copy()
    total_leaf = biomass[point, 1, ILEAF, ICARBON] * veget_max[point, 1] + biomass[point, 2, ILEAF, ICARBON] * veget_max[point, 2]
    for pft in [1, 2]:
        bm_org = biomass[point, pft, ILEAF, ICARBON] * veget_max[point, pft]
        share = bm_org / total_leaf * biomass_pro[ILEAF, ICARBON]
        expected[point, pft, ILEAF, ICARBON] = (bm_org - share) / veget_max[point, pft]

    expected_total = np.zeros((NPARTS, nelements), dtype=np.float64)
    expected_total[ILEAF, ICARBON] = total_leaf
    expected_total[ILEAF, 1] = biomass[point, 1, ILEAF, 1] * veget_max[point, 1]
    expected_total[IROOT, ICARBON] = biomass[point, 1, IROOT, ICARBON] * veget_max[point, 1] + biomass[point, 2, IROOT, ICARBON] * veget_max[point, 2]

    assert np.allclose(np.asarray(result.biomass), expected)
    assert np.allclose(np.asarray(result.co2_to_bm_pro), co2_to_bm_pro - biomass_pro[ILEAF, ICARBON])
    assert np.allclose(np.asarray(result.biomass_total), expected_total)


def test_collect_legacy_scalar_pools_step_accumulates_positive_glcc_only():
    npts, nvm, nelements, ndeep = 1, 4, 2, 3
    point = 0
    glcc_frac = np.asarray([0.0, 0.2, -0.5, 0.3], dtype=np.float64)
    frac = np.asarray([0.0, 0.2, 0.0, 0.3], dtype=np.float64)
    bm_to_litter = np.arange(npts * nvm * NPARTS * nelements, dtype=np.float64).reshape(
        npts, nvm, NPARTS, nelements
    )
    carbon = np.arange(npts * NCARB * nvm, dtype=np.float64).reshape(npts, NCARB, nvm)
    deepC_a = np.arange(npts * ndeep * nvm, dtype=np.float64).reshape(npts, ndeep, nvm)
    deepC_s = deepC_a + 100.0
    deepC_p = deepC_a + 200.0
    co2_to_bm = np.asarray([[1.0, 2.0, 999.0, 4.0]], dtype=np.float64)
    gpp_daily = co2_to_bm + 10.0
    npp_daily = co2_to_bm + 20.0
    resp_maint = co2_to_bm + 30.0
    resp_growth = co2_to_bm + 40.0
    resp_hetero = co2_to_bm + 50.0
    co2_fire = co2_to_bm + 60.0
    litter_pro = np.ones((NLITT, 2, nelements), dtype=np.float64)
    litter_pro[ISTRUCTURAL, :, ICARBON] = [5.0, 0.0]
    lignin_content_pro = np.asarray([1.5, 9.0], dtype=np.float64)
    initial_bm_to_litter = np.ones((NPARTS, nelements), dtype=np.float64) * 7.0

    result = collect_legacy_scalar_pools_step(
        glcc_frac=glcc_frac,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        co2_to_bm=co2_to_bm,
        gpp_daily=gpp_daily,
        npp_daily=npp_daily,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        litter_pro=litter_pro,
        lignin_content_pro=lignin_content_pro,
        point=point,
        bm_to_litter_pro_initial=initial_bm_to_litter,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.veget_max_pro), np.sum(frac))
    assert np.allclose(
        np.asarray(result.bm_to_litter_pro),
        initial_bm_to_litter + np.sum(bm_to_litter[point] * frac[:, None, None], axis=0),
    )
    assert np.allclose(np.asarray(result.carbon_pro), np.sum(carbon[point] * frac[None, :], axis=1))
    assert np.allclose(np.asarray(result.deepC_a_pro), np.sum(deepC_a[point] * frac[None, :], axis=1))
    assert np.allclose(np.asarray(result.deepC_s_pro), np.sum(deepC_s[point] * frac[None, :], axis=1))
    assert np.allclose(np.asarray(result.deepC_p_pro), np.sum(deepC_p[point] * frac[None, :], axis=1))
    assert np.allclose(np.asarray(result.co2_to_bm_pro), np.sum(co2_to_bm[point] * frac))
    assert np.allclose(np.asarray(result.gpp_daily_pro), np.sum(gpp_daily[point] * frac))
    assert np.allclose(np.asarray(result.npp_daily_pro), np.sum(npp_daily[point] * frac))
    assert np.allclose(np.asarray(result.resp_maint_pro), np.sum(resp_maint[point] * frac))
    assert np.allclose(np.asarray(result.resp_growth_pro), np.sum(resp_growth[point] * frac))
    assert np.allclose(np.asarray(result.resp_hetero_pro), np.sum(resp_hetero[point] * frac))
    assert np.allclose(np.asarray(result.co2_fire_pro), np.sum(co2_fire[point] * frac))
    assert np.allclose(np.asarray(result.lignin_struc_pro), [1.5 / 5.0, 0.0])


def test_empty_pft_lcc_step_zeroes_exhausted_slot_only():
    npts, nvm, nelements, ndeep, nlevels = 1, 3, 2, 2, 2
    point, pft = 0, 1
    two_d = np.ones((npts, nvm), dtype=np.float64) * 5.0
    result = empty_pft_lcc_step(
        point=point,
        pft=pft,
        veget_max=two_d,
        biomass=np.ones((npts, nvm, NPARTS, nelements), dtype=np.float64),
        ind=two_d,
        carbon=np.ones((npts, NCARB, nvm), dtype=np.float64),
        litter=np.ones((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64),
        lignin_struc=np.ones((npts, nvm, nlevels), dtype=np.float64),
        bm_to_litter=np.ones((npts, nvm, NPARTS, nelements), dtype=np.float64),
        deepC_a=np.ones((npts, ndeep, nvm), dtype=np.float64),
        deepC_s=np.ones((npts, ndeep, nvm), dtype=np.float64),
        deepC_p=np.ones((npts, ndeep, nvm), dtype=np.float64),
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64),
        gpp_daily=two_d,
        npp_daily=two_d,
        gpp_week=two_d,
        npp_longterm=two_d,
        co2_to_bm=two_d,
        resp_maint=two_d,
        resp_growth=two_d,
        resp_hetero=two_d,
        lm_lastyearmax=two_d,
        leaf_frac=np.ones((npts, nvm, NLEAFAGES), dtype=np.float64),
        leaf_age=np.ones((npts, nvm, NLEAFAGES), dtype=np.float64),
        age=two_d,
        everywhere=two_d,
        pft_present=np.ones((npts, nvm), dtype=bool),
        when_growthinit=two_d,
        senescence=np.ones((npts, nvm), dtype=bool),
        gdd_from_growthinit=two_d,
        gdd_midwinter=two_d,
        time_hum_min=two_d,
        gdd_m5_dormance=two_d,
        ncd_dormance=two_d,
        moiavail_month=two_d,
        moiavail_week=two_d,
        ngd_minus5=two_d,
    )

    assert np.allclose(np.asarray(result.veget_max)[point, pft], 0.0)
    assert np.allclose(np.asarray(result.biomass)[point, pft], 0.0)
    assert np.allclose(np.asarray(result.carbon)[point, :, pft], 0.0)
    assert np.allclose(np.asarray(result.litter)[point, :, pft], 0.0)
    assert np.allclose(np.asarray(result.deepC_a)[point, :, pft], 0.0)
    assert np.allclose(np.asarray(result.fuel_1hr)[point, pft], 0.0)
    assert np.allclose(np.asarray(result.bm_to_litter)[point, pft], 0.0)
    assert np.allclose(np.asarray(result.leaf_frac)[point, pft], 0.0)
    assert not bool(np.asarray(result.pft_present)[point, pft])
    assert not bool(np.asarray(result.senescence)[point, pft])
    assert np.allclose(np.asarray(result.veget_max)[point, 0], 5.0)
    assert np.allclose(np.asarray(result.biomass)[point, 2], 1.0)
    assert bool(np.asarray(result.pft_present)[point, 2])


def test_add_incoming_proxy_pft_step_merges_area_weighted_state():
    npts, nvm, nelements, ndeep, nlevels = 1, 3, 2, 2, 2
    point, pft = 0, 1
    veget_old = 0.4
    veget_max_pro = 0.2
    veget_total = veget_old + veget_max_pro
    veget_max = np.ones((npts, nvm), dtype=np.float64) * 0.1
    veget_max[point, pft] = veget_old
    two_d = np.ones((npts, nvm), dtype=np.float64) * 10.0

    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64) * 3.0
    deep = np.ones((npts, ndeep, nvm), dtype=np.float64) * 4.0
    litter = np.ones((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64) * 5.0
    litter[point, ISTRUCTURAL, pft, :, ICARBON] = [10.0, 0.0]
    lignin_struc = np.ones((npts, nvm, nlevels), dtype=np.float64) * 0.8
    lignin_struc[point, pft, :] = [0.3, 0.8]
    litter_pro = np.ones((NLITT, nlevels, nelements), dtype=np.float64) * 2.0
    litter_pro[ISTRUCTURAL, :, ICARBON] = [2.0, 0.0]
    lignin_struc_pro = np.asarray([0.6, 0.9], dtype=np.float64)

    result = add_incoming_proxy_pft_step(
        point=point,
        pft=pft,
        veget_max_pro=veget_max_pro,
        carbon_pro=np.ones(NCARB, dtype=np.float64) * 6.0,
        litter_pro=litter_pro,
        lignin_struc_pro=lignin_struc_pro,
        bm_to_litter_pro=np.ones((NPARTS, nelements), dtype=np.float64) * 8.0,
        deepC_a_pro=np.ones(ndeep, dtype=np.float64) * 7.0,
        deepC_s_pro=np.ones(ndeep, dtype=np.float64) * 8.0,
        deepC_p_pro=np.ones(ndeep, dtype=np.float64) * 9.0,
        fuel_1hr_pro=np.ones((NLITT, nelements), dtype=np.float64) * 11.0,
        fuel_10hr_pro=np.ones((NLITT, nelements), dtype=np.float64) * 12.0,
        fuel_100hr_pro=np.ones((NLITT, nelements), dtype=np.float64) * 13.0,
        fuel_1000hr_pro=np.ones((NLITT, nelements), dtype=np.float64) * 14.0,
        biomass_pro=np.ones((NPARTS, nelements), dtype=np.float64) * 15.0,
        co2_to_bm_pro=16.0,
        npp_longterm_pro=17.0,
        ind_pro=18.0,
        lm_lastyearmax_pro=19.0,
        age_pro=20.0,
        everywhere_pro=0.9,
        leaf_frac_pro=np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float64),
        leaf_age_pro=np.asarray([0.4, 0.0, 0.0, 0.0], dtype=np.float64),
        pft_present_pro=True,
        senescence_pro=False,
        gpp_daily_pro=21.0,
        npp_daily_pro=22.0,
        resp_maint_pro=23.0,
        resp_growth_pro=24.0,
        resp_hetero_pro=25.0,
        co2_fire_pro=26.0,
        veget_max=veget_max,
        carbon=carbon,
        litter=litter,
        lignin_struc=lignin_struc,
        bm_to_litter=np.ones((npts, nvm, NPARTS, nelements), dtype=np.float64) * 6.0,
        deepC_a=deep,
        deepC_s=deep + 1.0,
        deepC_p=deep + 2.0,
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 7.0,
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 8.0,
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 9.0,
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 10.0,
        biomass=np.ones((npts, nvm, NPARTS, nelements), dtype=np.float64) * 11.0,
        co2_to_bm=two_d,
        npp_longterm=two_d + 1.0,
        ind=two_d + 2.0,
        lm_lastyearmax=two_d + 3.0,
        age=two_d + 4.0,
        everywhere=np.ones((npts, nvm), dtype=np.float64) * 0.1,
        leaf_frac=np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 0.25,
        leaf_age=np.ones((npts, nvm, NLEAFAGES), dtype=np.float64) * 1.0,
        pft_present=np.zeros((npts, nvm), dtype=bool),
        senescence=np.ones((npts, nvm), dtype=bool),
        gpp_daily=two_d + 5.0,
        npp_daily=two_d + 6.0,
        resp_maint=two_d + 7.0,
        resp_growth=two_d + 8.0,
        resp_hetero=two_d + 9.0,
        co2_fire=two_d + 10.0,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.veget_max)[point, pft], veget_total)
    assert np.allclose(np.asarray(result.carbon)[point, :, pft], (veget_old * 3.0 + 6.0) / veget_total)
    assert np.allclose(np.asarray(result.deepC_a)[point, :, pft], (veget_old * 4.0 + 7.0) / veget_total)
    assert np.allclose(
        np.asarray(result.bm_to_litter)[point, pft],
        (veget_old * 6.0 + 8.0) / veget_total,
    )
    assert np.allclose(
        np.asarray(result.biomass)[point, pft],
        (veget_old * 11.0 + 15.0) / veget_total,
    )
    assert np.allclose(np.asarray(result.co2_to_bm)[point, pft], (veget_old * 10.0 + 16.0) / veget_total)
    assert np.allclose(np.asarray(result.ind)[point, pft], (veget_old * 12.0 + 18.0) / veget_total)
    assert np.allclose(np.asarray(result.leaf_frac)[point, pft], (veget_old * 0.25 + np.asarray([0.2, 0.0, 0.0, 0.0])) / veget_total)
    assert np.allclose(np.asarray(result.leaf_age)[point, pft], (veget_old * 1.0 + np.asarray([0.4, 0.0, 0.0, 0.0])) / veget_total)
    assert np.allclose(np.asarray(result.everywhere)[point, pft], 0.9)
    assert bool(np.asarray(result.pft_present)[point, pft])
    assert not bool(np.asarray(result.senescence)[point, pft])
    assert np.allclose(np.asarray(result.gpp_daily)[point, pft], (veget_old * 15.0 + 21.0) / veget_total)

    expected_litter_structural = (veget_old * np.asarray([10.0, 0.0]) + np.asarray([2.0, 0.0])) / veget_total
    assert np.allclose(np.asarray(result.litter)[point, ISTRUCTURAL, pft, :, ICARBON], expected_litter_structural)
    expected_lignin0 = (veget_old * 10.0 * 0.3 + 2.0 * 0.6) / (veget_total * expected_litter_structural[0])
    assert np.allclose(np.asarray(result.lignin_struc)[point, pft], [expected_lignin0, 0.8])
    assert np.allclose(np.asarray(result.veget_max)[point, 0], 0.1)


def test_lcc_age_class_indices_follow_fortran_category_and_age_order():
    start_index = [0, 1, 3, 4, 6]
    nagec_pft = [1, 2, 1, 2, 1]
    nvm = 7
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[3] = True
    is_grassland_manag[4] = True

    result = lcc_age_class_indices(
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=2,
        nagec_herb=2,
    )

    assert np.array_equal(np.asarray(result.indall_tree), [1, 2])
    assert np.array_equal(np.asarray(result.indold_tree), [2])
    assert np.array_equal(np.asarray(result.indagec_tree), [[1]])
    assert np.array_equal(np.asarray(result.indall_grass), [3])
    assert np.array_equal(np.asarray(result.indold_grass), [3])
    assert np.array_equal(np.asarray(result.indagec_grass), [[3]])
    assert np.array_equal(np.asarray(result.indall_pasture), [4, 5])
    assert np.array_equal(np.asarray(result.indold_pasture), [5])
    assert np.array_equal(np.asarray(result.indagec_pasture), [[4]])
    assert np.array_equal(np.asarray(result.indall_crop), [6])
    assert np.array_equal(np.asarray(result.indold_crop), [6])
    assert np.array_equal(np.asarray(result.indagec_crop), [[6]])


def test_lcc_calc_cover_step_aggregates_mtc_and_age_classes_old_to_young():
    start_index = [0, 1, 3, 4, 6]
    nagec_pft = [1, 2, 1, 2, 1]
    nvm = 7
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[3] = True
    is_grassland_manag[4] = True
    veget_max = np.asarray([[0.01, 0.11, 0.12, 0.21, 0.31, 0.32, 0.41]], dtype=np.float64)

    result = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=2,
        nagec_herb=2,
    )

    assert np.allclose(np.asarray(result.veget_mtc), [[0.01, 0.23, 0.21, 0.63, 0.41]])
    assert np.allclose(np.asarray(result.vegagec_tree), [[0.12, 0.11]])
    assert np.allclose(np.asarray(result.vegagec_grass), [[0.21, 0.0]])
    assert np.allclose(np.asarray(result.vegagec_pasture), [[0.32, 0.31]])
    assert np.allclose(np.asarray(result.vegagec_crop), [[0.41, 0.0]])


def test_lcc_cross_give_receive_step_allocates_by_donor_and_receiver_cover():
    point = 0
    donor_pfts = [1, 2]
    receiver_agec_pfts = np.asarray([[4], [5]], dtype=np.int32)
    agec_group = np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int32)
    veget_max = np.zeros((1, 6), dtype=np.float64)
    veget_max[point, donor_pfts] = [0.6, 0.4]
    veget_mtc = np.asarray([[0.0, 1.0, 0.75, 0.25]], dtype=np.float64)
    glcc_pft = np.zeros((1, 6), dtype=np.float64)
    glcc_pftmtc = np.zeros((1, 6, 4), dtype=np.float64)
    glcc_pft_tmp = np.zeros((1, 6), dtype=np.float64)

    result = lcc_cross_give_receive_step(
        point=point,
        frac_used=0.5,
        veget_mtc=veget_mtc,
        donor_pfts=donor_pfts,
        receiver_agec_pfts=receiver_agec_pfts,
        nagec_receive=1,
        agec_group=agec_group,
        veget_max=veget_max,
        glcc_pft=glcc_pft,
        glcc_pftmtc=glcc_pftmtc,
        glcc_pft_tmp=glcc_pft_tmp,
        min_stomate=1.0e-8,
    )

    donor_tmp = np.asarray([0.3, 0.2], dtype=np.float64)
    assert np.allclose(np.asarray(result.glcc_pft_tmp)[point, donor_pfts], donor_tmp)
    assert np.allclose(np.asarray(result.glcc_pft)[point, donor_pfts], donor_tmp)
    assert np.allclose(np.asarray(result.veget_max)[point, donor_pfts], [0.3, 0.2])
    assert np.allclose(np.asarray(result.glcc_pftmtc)[point, donor_pfts, 2], donor_tmp * 0.75)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[point, donor_pfts, 3], donor_tmp * 0.25)


def test_lcc_cross_give_receive_step_splits_equally_without_receiver_cover():
    point = 0
    donor_pfts = [1, 2]
    receiver_agec_pfts = np.asarray([[4], [5]], dtype=np.int32)
    agec_group = np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int32)
    veget_max = np.zeros((1, 6), dtype=np.float64)
    veget_max[point, donor_pfts] = [0.6, 0.4]
    veget_mtc = np.zeros((1, 4), dtype=np.float64)

    result = lcc_cross_give_receive_step(
        point=point,
        frac_used=0.5,
        veget_mtc=veget_mtc,
        donor_pfts=donor_pfts,
        receiver_agec_pfts=receiver_agec_pfts,
        nagec_receive=1,
        agec_group=agec_group,
        veget_max=veget_max,
        glcc_pft=np.zeros((1, 6), dtype=np.float64),
        glcc_pftmtc=np.zeros((1, 6, 4), dtype=np.float64),
        glcc_pft_tmp=np.zeros((1, 6), dtype=np.float64),
        min_stomate=1.0e-8,
    )

    donor_tmp = np.asarray([0.3, 0.2], dtype=np.float64)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[point, donor_pfts, 2], donor_tmp / 2.0)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[point, donor_pfts, 3], donor_tmp / 2.0)


def test_lcc_type_conversion_step_consumes_donor_age_classes_old_to_young():
    point = 0
    transition_index = 3
    glcc_real = np.zeros((1, 12), dtype=np.float64)
    glcc_real[point, transition_index] = 0.5
    glcc_remain = glcc_real.copy()
    veget_mtc = np.asarray([[0.0, 1.0, 0.75, 0.25]], dtype=np.float64)
    veget_max = np.zeros((1, 6), dtype=np.float64)
    veget_max[point, 2] = 0.3
    veget_max[point, 1] = 0.4
    vegagec_donor = np.asarray([[0.3, 0.4]], dtype=np.float64)
    donor_old_pfts = [2]
    donor_agec_pfts = np.asarray([[1]], dtype=np.int32)
    receiver_agec_pfts = np.asarray([[4], [5]], dtype=np.int32)
    agec_group = np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int32)

    result = lcc_type_conversion_step(
        point=point,
        transition_index=transition_index,
        glcc_real=glcc_real,
        veget_mtc=veget_mtc,
        donor_old_pfts=donor_old_pfts,
        donor_agec_pfts=donor_agec_pfts,
        receiver_agec_pfts=receiver_agec_pfts,
        nagec_receive=1,
        agec_group=agec_group,
        vegagec_donor=vegagec_donor,
        veget_max=veget_max,
        glcc_pft=np.zeros((1, 6), dtype=np.float64),
        glcc_pftmtc=np.zeros((1, 6, 4), dtype=np.float64),
        glcc_pft_tmp=np.zeros((1, 6), dtype=np.float64),
        glcc_remain=glcc_remain,
        iagec_start=0,
        iagec_end=1,
        old_to_young=True,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.glcc_remain)[point, transition_index], 0.0)
    assert np.allclose(np.asarray(result.vegagec_donor)[point], [0.0, 0.2])
    assert np.allclose(np.asarray(result.veget_max)[point, [2, 1]], [0.0, 0.2])
    assert np.allclose(np.asarray(result.glcc_pft)[point, [2, 1]], [0.3, 0.2])
    assert np.allclose(np.asarray(result.glcc_pftmtc)[point, 2, 2], 0.3 * 0.75)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[point, 2, 3], 0.3 * 0.25)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[point, 1, 2], 0.2 * 0.75)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[point, 1, 3], 0.2 * 0.25)


def test_lcc_glcc_compensation_full_step_uses_surplus_sources_for_deficit_target():
    veget_4veg = np.asarray([[0.3, 0.4, 0.2, 0.0]], dtype=np.float64)
    glcc = np.zeros((1, 12), dtype=np.float64)
    glcc[0, 2] = 0.5
    glcc[0, 5] = 0.1
    glcc[0, 8] = 0.1

    result = lcc_glcc_compensation_full_step(
        veget_4veg=veget_4veg,
        glcc=glcc,
        glcc_real=np.zeros_like(glcc),
        glcc_def=np.zeros_like(glcc),
        incre_deficit=np.zeros((1, 4), dtype=np.float64),
        first_transition=8,
        first_source=2,
        second_transition=5,
        second_source=1,
        third_transition=2,
        third_source=0,
        target_slot=3,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.glcc_def)[0, [2, 5, 8]], [-0.2, 0.3, 0.1])
    assert np.allclose(np.asarray(result.glcc_real)[0, [2, 5, 8]], [0.3, 0.2, 0.2])
    assert np.allclose(np.asarray(result.veget_4veg), [[0.0, 0.2, 0.0, 0.0]])
    assert np.allclose(np.asarray(result.incre_deficit), 0.0)


@pytest.mark.parametrize(
    "veget_sources,glcc_values,expected_real,expected_incre",
    [
        ([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.1, 0.2, 0.3], -0.3),
        ([0.5, 0.2, 0.1], [0.1, 0.3, 0.2], [0.3, 0.2, 0.1], 0.0),
        ([0.1, 0.5, 0.1], [0.2, 0.1, 0.2], [0.1, 0.3, 0.1], 0.0),
        ([0.5, 0.5, 0.1], [0.1, 0.1, 0.2], [0.1, 0.2, 0.1], 0.0),
        ([0.1, 0.1, 0.5], [0.2, 0.2, 0.1], [0.1, 0.1, 0.3], 0.0),
        ([0.5, 0.1, 0.5], [0.1, 0.2, 0.1], [0.1, 0.1, 0.2], 0.0),
        ([0.1, 0.5, 0.5], [0.2, 0.1, 0.1], [0.1, 0.1, 0.2], 0.0),
        ([0.5, 0.6, 0.7], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 0.0),
    ],
)
def test_lcc_glcc_compensation_full_step_covers_fortran_sign_cases(
    veget_sources,
    glcc_values,
    expected_real,
    expected_incre,
):
    """Fortran glcc_compensation_full branches 1-8 in source order."""

    tree, grass, pasture = veget_sources
    f2c, g2c, p2c = glcc_values
    veget_4veg = np.asarray([[tree, grass, pasture, 0.0]], dtype=np.float64)
    glcc = np.zeros((1, 12), dtype=np.float64)
    glcc[0, 2] = f2c
    glcc[0, 5] = g2c
    glcc[0, 8] = p2c

    result = lcc_glcc_compensation_full_step(
        veget_4veg=veget_4veg,
        glcc=glcc,
        glcc_real=np.zeros_like(glcc),
        glcc_def=np.zeros_like(glcc),
        incre_deficit=np.zeros((1, 4), dtype=np.float64),
        first_transition=8,
        first_source=2,
        second_transition=5,
        second_source=1,
        third_transition=2,
        third_source=0,
        target_slot=3,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.glcc_real)[0, [2, 5, 8]], expected_real)
    assert np.allclose(np.asarray(result.incre_deficit)[0, 3], expected_incre)
    assert np.all(np.asarray(result.veget_4veg)[0, :3] >= -1.0e-14)


def test_lcc_harvest_same_mtc_step_resets_receiver_slots_to_tree_mtc():
    glcc_pft = np.asarray([[0.0, 0.2, 0.3, 0.4]], dtype=np.float64)
    glcc_pft_tmp = np.ones((1, 4), dtype=np.float64)
    glcc_pftmtc = np.ones((1, 4, 3), dtype=np.float64) * 9.0
    is_tree = np.asarray([False, True, True, False], dtype=bool)
    pft_to_mtc = np.asarray([0, 1, 1, 2], dtype=np.int32)

    result = lcc_harvest_same_mtc_step(
        glcc_pft=glcc_pft,
        glcc_pft_tmp=glcc_pft_tmp,
        glcc_pftmtc=glcc_pftmtc,
        is_tree=is_tree,
        pft_to_mtc=pft_to_mtc,
    )

    expected = np.zeros_like(glcc_pftmtc)
    expected[0, 1, 1] = 0.2
    expected[0, 2, 1] = 0.3
    assert np.allclose(np.asarray(result.glcc_pft_tmp), 0.0)
    assert np.allclose(np.asarray(result.glcc_pftmtc), expected)


def test_lcc_primary_net_transition_sequence_step_applies_fortran_order():
    start_index = [0, 1, 2, 3, 4]
    nagec_pft = [1, 1, 1, 1, 1]
    nvm, nvmap = 5, 5
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[2] = True
    is_grassland_manag[3] = True
    agec_group = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    indices = lcc_age_class_indices(
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=1,
        nagec_herb=1,
    )
    veget_max = np.asarray([[0.0, 0.5, 0.4, 0.3, 0.2]], dtype=np.float64)
    cover = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=1,
        nagec_herb=1,
    )
    glcc_primary = np.zeros((1, 12), dtype=np.float64)
    glcc_net = np.zeros((1, 12), dtype=np.float64)
    glcc_primary[0, 2] = 0.2
    glcc_net[0, 2] = 0.1
    glcc_primary[0, 5] = 0.1
    glcc_primary[0, 8] = 0.4

    result = lcc_primary_net_transition_sequence_step(
        glcc_primary_shift=glcc_primary,
        glcc_net_lcc=glcc_net,
        veget_mtc=np.asarray(cover.veget_mtc),
        indices=indices,
        agec_group=agec_group,
        vegagec_tree=np.asarray(cover.vegagec_tree),
        vegagec_grass=np.asarray(cover.vegagec_grass),
        vegagec_pasture=np.asarray(cover.vegagec_pasture),
        vegagec_crop=np.asarray(cover.vegagec_crop),
        veget_max=veget_max,
        glcc_pft=np.zeros((1, nvm), dtype=np.float64),
        glcc_pftmtc=np.zeros((1, nvm, nvmap), dtype=np.float64),
        glcc_pft_tmp=np.zeros((1, nvm), dtype=np.float64),
        incre_deficit=np.zeros((1, 12), dtype=np.float64),
        nagec_tree=1,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.glcc_real)[0, 2], 0.3)
    assert np.allclose(np.asarray(result.glcc_remain)[0, [2, 5]], [0.0, 0.0])
    assert np.allclose(np.asarray(result.glcc_remain)[0, 8], 0.1)
    assert np.allclose(np.asarray(result.incre_deficit)[0, 8], -0.1)
    assert np.allclose(np.asarray(result.veget_max)[0, [1, 2, 3]], [0.2, 0.3, 0.0])
    assert np.allclose(np.asarray(result.glcc_pft)[0, [1, 2, 3]], [0.3, 0.1, 0.3])
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 1, 4], 0.3)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 2, 4], 0.1)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 3, 4], 0.3)


def test_lcc_secondary_shift_transition_sequence_step_uses_two_forest_passes():
    start_index = [0, 1, 5, 6, 7]
    nagec_pft = [1, 4, 1, 1, 1]
    nvm, nvmap = 8, 5
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[5] = True
    is_grassland_manag[6] = True
    agec_group = np.asarray([0, 1, 1, 1, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.zeros((1, nvm), dtype=np.float64)
    veget_max[0, 1:5] = [0.05, 0.05, 0.2, 0.4]
    veget_max[0, 5:] = [0.1, 0.1, 0.1]
    indices = lcc_age_class_indices(
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=4,
        nagec_herb=1,
    )
    cover = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=4,
        nagec_herb=1,
    )
    glcc_second = np.zeros((1, 12), dtype=np.float64)
    glcc_second[0, 2] = 0.5

    result = lcc_secondary_shift_transition_sequence_step(
        glcc_second_shift=glcc_second,
        veget_mtc=np.asarray(cover.veget_mtc),
        indices=indices,
        agec_group=agec_group,
        vegagec_tree=np.asarray(cover.vegagec_tree),
        vegagec_grass=np.asarray(cover.vegagec_grass),
        vegagec_pasture=np.asarray(cover.vegagec_pasture),
        vegagec_crop=np.asarray(cover.vegagec_crop),
        veget_max=veget_max,
        glcc_pft=np.zeros((1, nvm), dtype=np.float64),
        glcc_pftmtc=np.zeros((1, nvm, nvmap), dtype=np.float64),
        glcc_pft_tmp=np.zeros((1, nvm), dtype=np.float64),
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        incre_deficit=np.zeros((1, 12), dtype=np.float64),
        nagec_tree=4,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.glcc_second_shift_remain)[0, 2], 0.3)
    assert np.allclose(np.asarray(result.glcc_remain)[0, 2], 0.0)
    assert np.allclose(np.asarray(result.incre_deficit), 0.0)
    assert np.allclose(np.asarray(result.veget_max)[0, 3], 0.0)
    assert np.allclose(np.asarray(result.veget_max)[0, 4], 0.1)
    assert np.allclose(np.asarray(result.glcc_pft)[0, [3, 4]], [0.2, 0.3])
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 3, 4], 0.2)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 4, 4], 0.3)


def test_lcc_forestry_harvest_sequence_step_compensates_secondary_with_primary_then_remaps():
    start_index = [0, 1, 5]
    nagec_pft = [1, 4, 1]
    nvm, nvmap = 6, 3
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1:5] = True
    is_grassland_manag[5] = True
    agec_group = np.asarray([0, 1, 1, 1, 1, 2], dtype=np.int32)
    pft_to_mtc = agec_group.copy()
    veget_max = np.zeros((1, nvm), dtype=np.float64)
    veget_max[0, 1:5] = [0.05, 0.1, 0.2, 0.5]
    veget_max[0, 5] = 0.1
    indices = lcc_age_class_indices(
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=4,
        nagec_herb=1,
    )
    cover = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=4,
        nagec_herb=1,
    )
    harvest_matrix = np.zeros((1, 12), dtype=np.float64)
    harvest_matrix[0, 0] = 0.2
    harvest_matrix[0, 1] = 0.5

    result = lcc_forestry_harvest_sequence_step(
        harvest_matrix=harvest_matrix,
        veget_mtc=np.asarray(cover.veget_mtc),
        indices=indices,
        agec_group=agec_group,
        vegagec_tree=np.asarray(cover.vegagec_tree),
        veget_max=veget_max,
        glcc_pft=np.zeros((1, nvm), dtype=np.float64),
        glcc_pftmtc=np.zeros((1, nvm, nvmap), dtype=np.float64),
        glcc_pft_tmp=np.zeros((1, nvm), dtype=np.float64),
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=pft_to_mtc,
        nagec_tree=4,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.fhmatrix_remain_a)[0, 1], 0.2)
    assert np.allclose(np.asarray(result.fhmatrix_remain_b)[0, 1], 0.0)
    assert np.allclose(np.asarray(result.glcc_remain)[0, [0, 1]], [0.0, 0.0])
    assert np.allclose(np.asarray(result.deficit_pf2yf_final), 0.0)
    assert np.allclose(np.asarray(result.deficit_sf2yf_final), 0.0)
    assert np.allclose(np.asarray(result.glcc_pft)[0, [2, 3, 4]], [0.1, 0.2, 0.4])
    assert np.allclose(np.asarray(result.veget_max)[0, [2, 3, 4]], [0.0, 0.0, 0.1])
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 2, 1], 0.1)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 3, 1], 0.2)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 4, 1], 0.4)
    assert np.allclose(np.asarray(result.glcc_pft_tmp), 0.0)


def test_lcc_gross_firstday_allocation_step_composes_primary_net_path():
    start_index = [0, 1, 2, 3, 4]
    nagec_pft = [1, 1, 1, 1, 1]
    nvm, nvmap = 5, 5
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[2] = True
    is_grassland_manag[3] = True
    agec_group = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.asarray([[0.0, 0.5, 0.4, 0.3, 0.2]], dtype=np.float64)
    zeros = np.zeros((1, 12), dtype=np.float64)
    glcc_primary = zeros.copy()
    glcc_net = zeros.copy()
    glcc_primary[0, 2] = 0.2
    glcc_net[0, 2] = 0.1

    result = lcc_gross_firstday_allocation_step(
        veget_max_org=veget_max,
        harvest_matrix=zeros,
        glcc_second_shift=zeros,
        glcc_primary_shift=glcc_primary,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=agec_group,
        agec_group=agec_group,
        nagec_tree=1,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.glcc_real)[0, 2], 0.3)
    assert np.allclose(np.asarray(result.incre_deficit), 0.0)
    assert np.allclose(np.asarray(result.veget_max)[0, 1], 0.2)
    assert np.allclose(np.asarray(result.glcc_pft)[0, 1], 0.3)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 1, 4], 0.3)
    assert np.allclose(np.asarray(result.harvest.glcc_pft), 0.0)
    assert np.allclose(np.asarray(result.secondary_shift.glcc_remain), 0.0)


def test_lcc_gross_firstday_single_age_allocation_caps_harvest_and_remaps_to_tree_mtc():
    start_index = [0, 1, 5, 6, 7]
    nagec_pft = [1, 4, 1, 1, 1]
    nvm, nvmap = 8, 5
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1:5] = True
    natural[5] = True
    is_grassland_manag[6] = True
    agec_group = np.asarray([0, 1, 1, 1, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.zeros((1, nvm), dtype=np.float64)
    veget_max[0, 4] = 0.3
    zeros = np.zeros((1, 12), dtype=np.float64)
    harvest_matrix = zeros.copy()
    harvest_matrix[0, 0] = 0.2
    harvest_matrix[0, 1] = 0.3

    result = lcc_gross_firstday_single_age_allocation_step(
        veget_max_org=veget_max,
        harvest_matrix=harvest_matrix,
        glcc_second_shift=zeros,
        glcc_primary_shift=zeros,
        glcc_net_lcc=zeros,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=agec_group,
        agec_group=agec_group,
        nagec_tree=4,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.hmatrix_real)[0, 0], 0.3)
    assert np.allclose(np.asarray(result.deficit_pf2yf_final), [-0.2])
    assert np.allclose(np.asarray(result.deficit_sf2yf_final), 0.0)
    assert np.allclose(np.asarray(result.glcc_pft)[0, 4], 0.3)
    assert np.allclose(np.asarray(result.veget_max)[0, 4], 0.0)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 4, 1], 0.3)
    assert np.allclose(np.asarray(result.glcc_pft_tmp), 0.0)
    assert np.allclose(np.asarray(result.glcc_real), 0.0)


def test_lcc_gross_firstday_single_age_allocation_applies_compensated_tree_to_crop_old_to_young():
    start_index = [0, 1, 5, 6, 7]
    nagec_pft = [1, 4, 1, 1, 1]
    nvm, nvmap = 8, 5
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1:5] = True
    natural[5] = True
    is_grassland_manag[6] = True
    agec_group = np.asarray([0, 1, 1, 1, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.zeros((1, nvm), dtype=np.float64)
    veget_max[0, 1:5] = [0.05, 0.1, 0.2, 0.4]
    veget_max[0, 7] = 0.2
    zeros = np.zeros((1, 12), dtype=np.float64)
    glcc_net = zeros.copy()
    glcc_net[0, 2] = 0.5

    result = lcc_gross_firstday_single_age_allocation_step(
        veget_max_org=veget_max,
        harvest_matrix=zeros,
        glcc_second_shift=zeros,
        glcc_primary_shift=zeros,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=agec_group,
        agec_group=agec_group,
        nagec_tree=4,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.glcc_real)[0, 2], 0.5)
    assert np.allclose(np.asarray(result.glcc_def)[0, 2], 0.25)
    assert np.allclose(np.asarray(result.glcc_remain)[0, 2], 0.0)
    assert np.allclose(np.asarray(result.incre_deficit), 0.0)
    assert np.allclose(np.asarray(result.glcc_pft)[0, [4, 3]], [0.4, 0.1])
    assert np.allclose(np.asarray(result.veget_max)[0, [4, 3]], [0.0, 0.1])
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 4, 4], 0.4)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 3, 4], 0.1)


def test_lcc_gross_firstday_single_age_allocation_handles_four_way_transition_stress():
    start_index = [0, 1, 5, 6, 7]
    nagec_pft = [1, 4, 1, 1, 1]
    nvm, nvmap = 8, 5
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1:5] = True
    natural[5] = True
    is_grassland_manag[6] = True
    agec_group = np.asarray([0, 1, 1, 1, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.zeros((1, nvm), dtype=np.float64)
    veget_max[0, 1:5] = [0.05, 0.1, 0.2, 0.4]
    veget_max[0, 5:] = [0.2, 0.3, 0.4]
    zeros = np.zeros((1, 12), dtype=np.float64)
    glcc_net = zeros.copy()
    glcc_net[0, 2] = 0.3
    glcc_net[0, 4] = 0.1
    glcc_net[0, 6] = 0.2
    glcc_net[0, 10] = 0.15

    result = lcc_gross_firstday_single_age_allocation_step(
        veget_max_org=veget_max,
        harvest_matrix=zeros,
        glcc_second_shift=zeros,
        glcc_primary_shift=zeros,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=agec_group,
        agec_group=agec_group,
        nagec_tree=4,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.glcc_real)[0, [2, 4, 6, 10]], [0.3, 0.1, 0.2, 0.15])
    assert np.allclose(np.asarray(result.glcc_remain)[0, [2, 4, 6, 10]], 0.0)
    assert np.allclose(np.asarray(result.incre_deficit), 0.0)
    assert np.allclose(np.asarray(result.glcc_pft)[0, [4, 5, 6, 7]], [0.3, 0.1, 0.2, 0.15])
    assert np.allclose(np.asarray(result.veget_max)[0, [4, 5, 6, 7]], [0.1, 0.1, 0.1, 0.25])
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 4, 4], 0.3)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 5, 3], 0.1)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 6, 1], 0.2)
    assert np.allclose(np.asarray(result.glcc_pftmtc)[0, 7, 2], 0.15)


def test_lcc_collect_legacy_pft_step_dispatches_tree_and_herb_donors():
    npts, nvm, nvmap, nelements, nlevels, ndeep = 1, 4, 3, 1, 2, 2
    point, receiver_mtc = 0, 1
    tree_donor, herb_donor = 2, 3
    tree_frac, herb_frac = 0.2, 0.3
    start_index = [0, 1, 3]
    glcc_pftmtc = np.zeros((npts, nvm, nvmap), dtype=np.float64)
    glcc_pftmtc[point, tree_donor, receiver_mtc] = tree_frac
    glcc_pftmtc[point, herb_donor, receiver_mtc] = herb_frac

    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    woody_parts = [ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN]
    litter_parts = [ISAPBELOW, IHEARTBELOW, IROOT, IFRUIT, ICARBRES, ILEAF]
    for offset, part in enumerate(woody_parts, start=1):
        biomass[point, tree_donor, part, ICARBON] = offset * 10.0
    for offset, part in enumerate(litter_parts, start=1):
        biomass[point, tree_donor, part, ICARBON] = offset * 2.0
    biomass[point, herb_donor, :, ICARBON] = np.arange(1.0, NPARTS + 1.0)
    bm_to_litter = np.zeros_like(biomass)
    bm_to_litter[point, tree_donor, :, ICARBON] = 1.0
    bm_to_litter[point, herb_donor, :, ICARBON] = 2.0

    carbon = np.arange(npts * NCARB * nvm, dtype=np.float64).reshape(npts, NCARB, nvm)
    deepC_a = np.arange(npts * ndeep * nvm, dtype=np.float64).reshape(npts, ndeep, nvm)
    deepC_s = deepC_a + 10.0
    deepC_p = deepC_a + 20.0
    litter = np.zeros((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64)
    litter[point, :, tree_donor, :, ICARBON] = np.asarray([[4.0, 6.0], [8.0, 10.0]])
    litter[point, :, herb_donor, :, ICARBON] = np.asarray([[3.0, 5.0], [7.0, 11.0]])
    fuel_1hr = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64)
    fuel_10hr = fuel_1hr * 2.0
    fuel_100hr = fuel_1hr * 3.0
    fuel_1000hr = fuel_1hr * 4.0
    lignin_struc = np.zeros((npts, nvm, nlevels), dtype=np.float64)
    lignin_struc[point, tree_donor, :] = [0.1, 0.2]
    lignin_struc[point, herb_donor, :] = [0.3, 0.4]
    scalars = np.asarray([[0.0, 10.0, 20.0, 30.0]], dtype=np.float64)
    convflux = np.zeros((npts, 2), dtype=np.float64)
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    coeff_1 = np.asarray([0.0, 0.0, 0.1, 0.0], dtype=np.float64)
    coeff_10 = np.asarray([0.0, 0.0, 0.2, 0.0], dtype=np.float64)
    coeff_100 = np.asarray([0.0, 0.0, 0.7, 0.0], dtype=np.float64)
    is_tree = np.asarray([False, True, True, False], dtype=bool)

    result = lcc_collect_legacy_pft_step(
        point=point,
        receiver_mtc=receiver_mtc,
        start_index=start_index,
        glcc_pftmtc=glcc_pftmtc,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        lignin_struc=lignin_struc,
        co2_to_bm=scalars,
        gpp_daily=scalars + 1.0,
        npp_daily=scalars + 2.0,
        resp_maint=scalars + 3.0,
        resp_growth=scalars + 4.0,
        resp_hetero=scalars + 5.0,
        co2_fire=scalars + 6.0,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        is_tree=is_tree,
        coeff_lcchange_1=coeff_1,
        coeff_lcchange_10=coeff_10,
        coeff_lcchange_100=coeff_100,
        min_stomate=1.0e-8,
    )

    tree_above = sum(offset * 10.0 for offset in range(1, len(woody_parts) + 1)) * tree_frac
    expected_bm_to_litter = (
        bm_to_litter[point, tree_donor] * tree_frac
        + bm_to_litter[point, herb_donor] * herb_frac
    )
    expected_bm_to_litter[litter_parts, :] += biomass[point, tree_donor, litter_parts, :] * tree_frac
    expected_bm_to_litter += biomass[point, herb_donor, :, :] * herb_frac
    expected_litter = (
        litter[point, :, tree_donor, :, :] * tree_frac
        + litter[point, :, herb_donor, :, :] * herb_frac
    )
    lignin_content = (
        litter[point, ISTRUCTURAL, tree_donor, :, ICARBON] * lignin_struc[point, tree_donor, :] * tree_frac
        + litter[point, ISTRUCTURAL, herb_donor, :, ICARBON] * lignin_struc[point, herb_donor, :] * herb_frac
    )
    expected_lignin = lignin_content / expected_litter[ISTRUCTURAL, :, ICARBON]
    frac = np.asarray([0.0, 0.0, tree_frac, herb_frac], dtype=np.float64)

    assert np.allclose(np.asarray(result.veget_max_pro), tree_frac + herb_frac)
    assert np.allclose(np.asarray(result.bm_to_litter_pro), expected_bm_to_litter)
    assert np.allclose(np.asarray(result.carbon_pro), np.sum(carbon[point] * frac[None, :], axis=1))
    assert np.allclose(np.asarray(result.deepC_a_pro), np.sum(deepC_a[point] * frac[None, :], axis=1))
    assert np.allclose(np.asarray(result.litter_pro), expected_litter)
    assert np.allclose(np.asarray(result.fuel_100hr_pro), fuel_100hr[point, tree_donor] * tree_frac + fuel_100hr[point, herb_donor] * herb_frac)
    assert np.allclose(np.asarray(result.lignin_struc_pro), expected_lignin)
    assert np.allclose(np.asarray(result.co2_to_bm_pro), np.sum(scalars[point] * frac))
    assert np.allclose(np.asarray(result.gpp_daily_pro), np.sum((scalars[point] + 1.0) * frac))
    assert np.allclose(np.asarray(result.convflux)[point, IWPHAR], coeff_1[tree_donor] * tree_above)
    assert np.allclose(np.asarray(result.prod10)[point, 0, IWPHAR], coeff_10[tree_donor] * tree_above)
    assert np.allclose(np.asarray(result.prod100)[point, 0, IWPHAR], coeff_100[tree_donor] * tree_above)
    assert np.allclose(np.asarray(result.convflux)[point, IWPLCC], 0.0)


def test_lcc_receiver_mtc_step_empties_exhausted_source_before_target_merge():
    npts, nvm, nvmap, nelements, nlevels, ndeep = 1, 3, 2, 1, 2, 2
    point, receiver_mtc, target_pft = 0, 1, 1
    start_index = [0, target_pft]
    nagec_pft = [1, 1]
    veget_max = np.asarray([[0.0, 0.2, 0.4]], dtype=np.float64)
    glcc_pftmtc = np.zeros((npts, nvm, nvmap), dtype=np.float64)
    glcc_pftmtc[point, target_pft, receiver_mtc] = 0.2

    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[point, target_pft, ILEAF, ICARBON] = 8.0
    biomass[point, target_pft, IROOT, ICARBON] = 4.0
    bm_to_litter = np.ones_like(biomass) * 0.5
    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64) * 3.0
    deep = np.ones((npts, ndeep, nvm), dtype=np.float64) * 4.0
    litter = np.ones((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64) * 2.0
    litter[point, ISTRUCTURAL, target_pft, :, ICARBON] = [5.0, 7.0]
    lignin_struc = np.ones((npts, nvm, nlevels), dtype=np.float64) * 0.25
    fuel = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64)
    scalars = np.ones((npts, nvm), dtype=np.float64) * 6.0
    convflux = np.zeros((npts, 2), dtype=np.float64)
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    is_tree = np.asarray([False, True, False], dtype=bool)
    coeff = np.ones(nvm, dtype=np.float64) * 0.1
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[target_pft, ILEAF, ICARBON] = 2.0
    bm_sapl[target_pft, IROOT, ICARBON] = 1.0
    cn_sapl = np.ones(nvm, dtype=np.float64)
    two_d = np.ones((npts, nvm), dtype=np.float64) * 9.0
    leaf_frac = np.ones((npts, nvm, NLEAFAGES), dtype=np.float64)

    result = lcc_receiver_mtc_step(
        point=point,
        receiver_mtc=receiver_mtc,
        start_index=start_index,
        nagec_pft=nagec_pft,
        glcc_pftmtc=glcc_pftmtc,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deep,
        deepC_s=deep + 1.0,
        deepC_p=deep + 2.0,
        fuel_1hr=fuel,
        fuel_10hr=fuel + 1.0,
        fuel_100hr=fuel + 2.0,
        fuel_1000hr=fuel + 3.0,
        lignin_struc=lignin_struc,
        co2_to_bm=scalars,
        gpp_daily=scalars + 1.0,
        npp_daily=scalars + 2.0,
        resp_maint=scalars + 3.0,
        resp_growth=scalars + 4.0,
        resp_hetero=scalars + 5.0,
        co2_fire=scalars + 6.0,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        is_tree=is_tree,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=12.0,
        veget_max=veget_max,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        leaf_frac=leaf_frac,
        leaf_age=leaf_frac,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.ones((npts, nvm), dtype=bool),
        gpp_week=two_d,
        when_growthinit=two_d,
        gdd_from_growthinit=two_d,
        gdd_midwinter=two_d,
        time_hum_min=two_d,
        gdd_m5_dormance=two_d,
        ncd_dormance=two_d,
        moiavail_month=two_d,
        moiavail_week=two_d,
        ngd_minus5=two_d,
        min_stomate=1.0e-8,
    )

    assert np.array_equal(np.asarray(result.exhausted_sources), [False, True, False])
    assert np.allclose(np.asarray(result.collection.veget_max_pro), 0.2)
    assert np.allclose(np.asarray(result.state.veget_max)[point, target_pft], 0.2)
    assert np.allclose(np.asarray(result.state.biomass)[point, target_pft, ILEAF, ICARBON], 2.0)
    assert np.allclose(np.asarray(result.state.biomass)[point, target_pft, IROOT, ICARBON], 1.0)
    assert bool(np.asarray(result.state.pft_present)[point, target_pft])
    assert not bool(np.asarray(result.state.senescence)[point, target_pft])
    assert np.allclose(np.asarray(result.gpp_week)[point, target_pft], 0.0)


def test_lcc_apply_all_receiver_mtcs_step_composes_receiver_loop_and_product_aging():
    npts, nvm, nvmap, nelements, nlevels, ndeep = 1, 3, 2, 1, 2, 2
    point, receiver_mtc, target_pft, donor_pft = 0, 1, 1, 2
    start_index = [0, target_pft]
    nagec_pft = [1, 1]
    veget_max = np.asarray([[0.0, 0.1, 0.5]], dtype=np.float64)
    glcc_pftmtc = np.zeros((npts, nvm, nvmap), dtype=np.float64)
    glcc_pftmtc[point, donor_pft, receiver_mtc] = 0.2

    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    woody_parts = [ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN]
    for offset, part in enumerate(woody_parts, start=1):
        biomass[point, donor_pft, part, ICARBON] = offset * 5.0
    bm_to_litter = np.ones_like(biomass) * 0.2
    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64) * 3.0
    deep = np.ones((npts, ndeep, nvm), dtype=np.float64) * 4.0
    litter = np.ones((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64) * 2.0
    lignin_struc = np.ones((npts, nvm, nlevels), dtype=np.float64) * 0.3
    fuel = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64)
    scalars = np.ones((npts, nvm), dtype=np.float64) * 6.0
    convflux = np.ones((npts, 2), dtype=np.float64) * 99.0
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    prod10[:, 0, :] = 77.0
    prod100[:, 0, :] = 88.0
    flux10 = np.zeros((npts, 10, 2), dtype=np.float64)
    flux100 = np.zeros((npts, 100, 2), dtype=np.float64)
    is_tree = np.asarray([False, True, True], dtype=bool)
    coeff_1 = np.asarray([0.0, 0.0, 0.1], dtype=np.float64)
    coeff_10 = np.asarray([0.0, 0.0, 0.2], dtype=np.float64)
    coeff_100 = np.asarray([0.0, 0.0, 0.7], dtype=np.float64)
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    cn_sapl = np.ones(nvm, dtype=np.float64)
    two_d = np.ones((npts, nvm), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)

    result = lcc_apply_all_receiver_mtcs_step(
        glcc_pftmtc=glcc_pftmtc,
        start_index=start_index,
        nagec_pft=nagec_pft,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deep,
        deepC_s=deep + 1.0,
        deepC_p=deep + 2.0,
        fuel_1hr=fuel,
        fuel_10hr=fuel + 1.0,
        fuel_100hr=fuel + 2.0,
        fuel_1000hr=fuel + 3.0,
        lignin_struc=lignin_struc,
        co2_to_bm=scalars,
        gpp_daily=scalars + 1.0,
        npp_daily=scalars + 2.0,
        resp_maint=scalars + 3.0,
        resp_growth=scalars + 4.0,
        resp_hetero=scalars + 5.0,
        co2_fire=scalars + 6.0,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        cflux_prod10=np.ones((npts, 2), dtype=np.float64) * 55.0,
        cflux_prod100=np.ones((npts, 2), dtype=np.float64) * 66.0,
        is_tree=is_tree,
        coeff_lcchange_1=coeff_1,
        coeff_lcchange_10=coeff_10,
        coeff_lcchange_100=coeff_100,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=12.0,
        veget_max=veget_max,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        leaf_frac=leaf_frac,
        leaf_age=leaf_frac,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        gpp_week=two_d,
        when_growthinit=two_d,
        gdd_from_growthinit=two_d,
        gdd_midwinter=two_d,
        time_hum_min=two_d,
        gdd_m5_dormance=two_d,
        ncd_dormance=two_d,
        moiavail_month=two_d,
        moiavail_week=two_d,
        ngd_minus5=two_d,
        dt_days=30.0,
        one_year=365.0,
        min_stomate=1.0e-8,
    )

    tree_above = sum(offset * 5.0 for offset in range(1, len(woody_parts) + 1)) * 0.2
    scale = 30.0 / 365.0
    assert np.allclose(np.asarray(result.state.veget_max)[point, donor_pft], 0.3)
    assert np.allclose(np.asarray(result.state.veget_max)[point, target_pft], 0.3)
    assert np.allclose(np.asarray(result.convflux)[point, IWPHAR], coeff_1[donor_pft] * tree_above * scale)
    assert np.allclose(np.asarray(result.prod10)[point, 1, IWPHAR], coeff_10[donor_pft] * tree_above)
    assert np.allclose(np.asarray(result.prod10)[point, 0, IWPHAR], 0.0)
    assert np.allclose(np.asarray(result.prod10)[point, 1, IWPLCC], 0.0)
    assert np.allclose(np.asarray(result.prod100)[point, 1, IWPHAR], coeff_100[donor_pft] * tree_above)
    assert np.allclose(np.asarray(result.convflux)[point, IWPLCC], 0.0)
    assert np.allclose(np.asarray(result.cflux_prod10)[point, IWPLCC], 0.0)


def test_lcc_gross_glcchange_step_matches_explicit_allocation_then_apply():
    start_index = [0, 1, 2, 3, 4]
    nagec_pft = [1, 1, 1, 1, 1]
    npts, nvm, nvmap, nelements, nlevels, ndeep = 1, 5, 5, 1, 2, 2
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[2] = True
    is_grassland_manag[3] = True
    agec_group = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.asarray([[0.0, 0.5, 0.4, 0.3, 0.2]], dtype=np.float64)
    zeros = np.zeros((npts, 12), dtype=np.float64)
    glcc_primary = zeros.copy()
    glcc_net = zeros.copy()
    glcc_primary[0, 2] = 0.2
    glcc_net[0, 2] = 0.1

    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    for offset, part in enumerate([ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN], start=1):
        biomass[0, 1, part, ICARBON] = offset * 4.0
    bm_to_litter = np.ones_like(biomass) * 0.2
    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64) * 3.0
    deep = np.ones((npts, ndeep, nvm), dtype=np.float64) * 4.0
    litter = np.ones((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64) * 2.0
    lignin_struc = np.ones((npts, nvm, nlevels), dtype=np.float64) * 0.3
    fuel = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64)
    scalars = np.ones((npts, nvm), dtype=np.float64) * 6.0
    convflux = np.zeros((npts, 2), dtype=np.float64)
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    flux10 = np.zeros((npts, 10, 2), dtype=np.float64)
    flux100 = np.zeros((npts, 100, 2), dtype=np.float64)
    coeff = np.ones(nvm, dtype=np.float64) * 0.1
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    cn_sapl = np.ones(nvm, dtype=np.float64)
    two_d = np.ones((npts, nvm), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)

    allocation = lcc_gross_firstday_allocation_step(
        veget_max_org=veget_max,
        harvest_matrix=zeros,
        glcc_second_shift=zeros,
        glcc_primary_shift=glcc_primary,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=agec_group,
        agec_group=agec_group,
        nagec_tree=1,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )
    manual = lcc_apply_all_receiver_mtcs_step(
        glcc_pftmtc=allocation.glcc_pftmtc,
        start_index=start_index,
        nagec_pft=nagec_pft,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deep,
        deepC_s=deep + 1.0,
        deepC_p=deep + 2.0,
        fuel_1hr=fuel,
        fuel_10hr=fuel + 1.0,
        fuel_100hr=fuel + 2.0,
        fuel_1000hr=fuel + 3.0,
        lignin_struc=lignin_struc,
        co2_to_bm=scalars,
        gpp_daily=scalars + 1.0,
        npp_daily=scalars + 2.0,
        resp_maint=scalars + 3.0,
        resp_growth=scalars + 4.0,
        resp_hetero=scalars + 5.0,
        co2_fire=scalars + 6.0,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        cflux_prod10=np.zeros((npts, 2), dtype=np.float64),
        cflux_prod100=np.zeros((npts, 2), dtype=np.float64),
        is_tree=is_tree,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=12.0,
        veget_max=veget_max,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        leaf_frac=leaf_frac,
        leaf_age=leaf_frac,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        gpp_week=two_d,
        when_growthinit=two_d,
        gdd_from_growthinit=two_d,
        gdd_midwinter=two_d,
        time_hum_min=two_d,
        gdd_m5_dormance=two_d,
        ncd_dormance=two_d,
        moiavail_month=two_d,
        moiavail_week=two_d,
        ngd_minus5=two_d,
        dt_days=30.0,
        one_year=365.0,
        min_stomate=1.0e-8,
    )
    wrapped = lcc_gross_glcchange_step(
        veget_max_org=veget_max,
        harvest_matrix=zeros,
        glcc_second_shift=zeros,
        glcc_primary_shift=glcc_primary,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=agec_group,
        agec_group=agec_group,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deep,
        deepC_s=deep + 1.0,
        deepC_p=deep + 2.0,
        fuel_1hr=fuel,
        fuel_10hr=fuel + 1.0,
        fuel_100hr=fuel + 2.0,
        fuel_1000hr=fuel + 3.0,
        lignin_struc=lignin_struc,
        co2_to_bm=scalars,
        gpp_daily=scalars + 1.0,
        npp_daily=scalars + 2.0,
        resp_maint=scalars + 3.0,
        resp_growth=scalars + 4.0,
        resp_hetero=scalars + 5.0,
        co2_fire=scalars + 6.0,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        cflux_prod10=np.zeros((npts, 2), dtype=np.float64),
        cflux_prod100=np.zeros((npts, 2), dtype=np.float64),
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=12.0,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        leaf_frac=leaf_frac,
        leaf_age=leaf_frac,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        gpp_week=two_d,
        when_growthinit=two_d,
        gdd_from_growthinit=two_d,
        gdd_midwinter=two_d,
        time_hum_min=two_d,
        gdd_m5_dormance=two_d,
        ncd_dormance=two_d,
        moiavail_month=two_d,
        moiavail_week=two_d,
        ngd_minus5=two_d,
        dt_days=30.0,
        one_year=365.0,
        nagec_tree=1,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(wrapped.allocation.glcc_pftmtc), np.asarray(allocation.glcc_pftmtc))
    assert np.allclose(np.asarray(wrapped.applied.state.veget_max), np.asarray(manual.state.veget_max))
    assert np.allclose(np.asarray(wrapped.applied.prod10), np.asarray(manual.prod10))
    assert np.allclose(np.asarray(wrapped.applied.convflux), np.asarray(manual.convflux))


def test_lcc_gross_glcchange_step_handles_multi_transition_stress_like_explicit_composition():
    """Fortran: gross_glcchange_fh lines 1565-1697 after gross_glcc_firstday_fh lines 2193-3059."""

    start_index = [0, 1, 2, 3, 4]
    nagec_pft = [1, 1, 1, 1, 1]
    npts, nvm, nelements, nlevels, ndeep = 1, 5, 1, 2, 2
    is_tree = np.zeros(nvm, dtype=bool)
    natural = np.zeros(nvm, dtype=bool)
    is_grassland_manag = np.zeros(nvm, dtype=bool)
    is_tree[1] = True
    natural[2] = True
    is_grassland_manag[3] = True
    agec_group = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    veget_max = np.asarray([[0.0, 0.6, 0.5, 0.4, 0.3]], dtype=np.float64)
    harvest_matrix = np.zeros((npts, 12), dtype=np.float64)
    glcc_second = np.zeros((npts, 12), dtype=np.float64)
    glcc_primary = np.zeros((npts, 12), dtype=np.float64)
    glcc_net = np.zeros((npts, 12), dtype=np.float64)
    harvest_matrix[0, [0, 1]] = [0.03, 0.02]
    glcc_second[0, [0, 4]] = [0.02, 0.015]
    glcc_primary[0, [2, 8]] = [0.04, 0.01]
    glcc_net[0, [6, 10]] = [0.025, 0.02]

    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    for pft in range(1, nvm):
        biomass[0, pft, :, ICARBON] = np.arange(1.0, NPARTS + 1.0) * (pft + 1)
    bm_to_litter = np.ones_like(biomass) * 0.2
    carbon = np.arange(npts * NCARB * nvm, dtype=np.float64).reshape(npts, NCARB, nvm) + 1.0
    litter = np.ones((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64) * 2.0
    deep = np.ones((npts, ndeep, nvm), dtype=np.float64) * 4.0
    fuel = np.ones((npts, nvm, NLITT, nelements), dtype=np.float64)
    lignin_struc = np.ones((npts, nvm, nlevels), dtype=np.float64) * 0.3
    scalars = np.ones((npts, nvm), dtype=np.float64) * 6.0
    convflux = np.zeros((npts, 2), dtype=np.float64)
    prod10 = np.zeros((npts, 11, 2), dtype=np.float64)
    prod100 = np.zeros((npts, 101, 2), dtype=np.float64)
    flux10 = np.zeros((npts, 10, 2), dtype=np.float64)
    flux100 = np.zeros((npts, 100, 2), dtype=np.float64)
    coeff = np.ones(nvm, dtype=np.float64) * 0.1
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    cn_sapl = np.ones(nvm, dtype=np.float64)
    two_d = np.ones((npts, nvm), dtype=np.float64)
    leaf_frac = np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64)

    allocation = lcc_gross_firstday_allocation_step(
        veget_max_org=veget_max,
        harvest_matrix=harvest_matrix,
        glcc_second_shift=glcc_second,
        glcc_primary_shift=glcc_primary,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=agec_group,
        agec_group=agec_group,
        nagec_tree=1,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )
    manual = lcc_apply_all_receiver_mtcs_step(
        glcc_pftmtc=allocation.glcc_pftmtc,
        start_index=start_index,
        nagec_pft=nagec_pft,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deep,
        deepC_s=deep + 1.0,
        deepC_p=deep + 2.0,
        fuel_1hr=fuel,
        fuel_10hr=fuel + 1.0,
        fuel_100hr=fuel + 2.0,
        fuel_1000hr=fuel + 3.0,
        lignin_struc=lignin_struc,
        co2_to_bm=scalars,
        gpp_daily=scalars + 1.0,
        npp_daily=scalars + 2.0,
        resp_maint=scalars + 3.0,
        resp_growth=scalars + 4.0,
        resp_hetero=scalars + 5.0,
        co2_fire=scalars + 6.0,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        cflux_prod10=np.zeros((npts, 2), dtype=np.float64),
        cflux_prod100=np.zeros((npts, 2), dtype=np.float64),
        is_tree=is_tree,
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=12.0,
        veget_max=veget_max,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        leaf_frac=leaf_frac,
        leaf_age=leaf_frac,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        gpp_week=two_d,
        when_growthinit=two_d,
        gdd_from_growthinit=two_d,
        gdd_midwinter=two_d,
        time_hum_min=two_d,
        gdd_m5_dormance=two_d,
        ncd_dormance=two_d,
        moiavail_month=two_d,
        moiavail_week=two_d,
        ngd_minus5=two_d,
        dt_days=30.0,
        one_year=365.0,
        min_stomate=1.0e-8,
    )
    wrapped = lcc_gross_glcchange_step(
        veget_max_org=veget_max,
        harvest_matrix=harvest_matrix,
        glcc_second_shift=glcc_second,
        glcc_primary_shift=glcc_primary,
        glcc_net_lcc=glcc_net,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=agec_group,
        agec_group=agec_group,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deep,
        deepC_s=deep + 1.0,
        deepC_p=deep + 2.0,
        fuel_1hr=fuel,
        fuel_10hr=fuel + 1.0,
        fuel_100hr=fuel + 2.0,
        fuel_1000hr=fuel + 3.0,
        lignin_struc=lignin_struc,
        co2_to_bm=scalars,
        gpp_daily=scalars + 1.0,
        npp_daily=scalars + 2.0,
        resp_maint=scalars + 3.0,
        resp_growth=scalars + 4.0,
        resp_hetero=scalars + 5.0,
        co2_fire=scalars + 6.0,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        cflux_prod10=np.zeros((npts, 2), dtype=np.float64),
        cflux_prod100=np.zeros((npts, 2), dtype=np.float64),
        coeff_lcchange_1=coeff,
        coeff_lcchange_10=coeff,
        coeff_lcchange_100=coeff,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=12.0,
        npp_longterm=two_d,
        ind=two_d,
        lm_lastyearmax=two_d,
        age=two_d,
        everywhere=np.zeros_like(two_d),
        leaf_frac=leaf_frac,
        leaf_age=leaf_frac,
        pft_present=np.ones((npts, nvm), dtype=bool),
        senescence=np.zeros((npts, nvm), dtype=bool),
        gpp_week=two_d,
        when_growthinit=two_d,
        gdd_from_growthinit=two_d,
        gdd_midwinter=two_d,
        time_hum_min=two_d,
        gdd_m5_dormance=two_d,
        ncd_dormance=two_d,
        moiavail_month=two_d,
        moiavail_week=two_d,
        ngd_minus5=two_d,
        dt_days=30.0,
        one_year=365.0,
        nagec_tree=1,
        nagec_herb=1,
        min_stomate=1.0e-8,
    )

    assert np.count_nonzero(np.asarray(allocation.glcc_pftmtc)) >= 4
    assert np.allclose(np.asarray(wrapped.allocation.glcc_pftmtc), np.asarray(allocation.glcc_pftmtc))
    assert np.allclose(np.asarray(wrapped.applied.state.veget_max), np.asarray(manual.state.veget_max))
    assert np.allclose(np.asarray(wrapped.applied.state.biomass), np.asarray(manual.state.biomass))
    assert np.allclose(np.asarray(wrapped.applied.prod10), np.asarray(manual.prod10))
    assert np.allclose(np.asarray(wrapped.applied.prod100), np.asarray(manual.prod100))
    assert np.allclose(np.asarray(wrapped.applied.convflux), np.asarray(manual.convflux))


def test_lcc_subtract_outgoing_fractions_step_marks_exhausted_sources():
    veget_max = np.asarray([[0.5, 0.2, 0.05, 0.4]], dtype=np.float64)
    glcc_pftmtc = np.zeros((1, 4, 3), dtype=np.float64)
    glcc_pftmtc[0, 1, 2] = 0.19
    glcc_pftmtc[0, 2, 2] = 0.05
    glcc_pftmtc[0, 3, 1] = 0.3
    glcc_pftmtc[0, 0, 2] = 1.0e-10

    result = lcc_subtract_outgoing_fractions_step(
        veget_max=veget_max,
        glcc_pftmtc=glcc_pftmtc,
        point=0,
        receiver_mtc=2,
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.veget_max)[0], [0.5, 0.01, 0.0, 0.4])
    assert np.array_equal(np.asarray(result.exhausted), [False, False, True, False])


def test_lcc_apply_collected_proxy_to_target_step_subtracts_takes_saplings_and_merges():
    npts, nvm, nelements, ndeep, nlevels = 1, 3, 1, 2, 2
    point, target_pft, receiver_mtc = 0, 1, 1
    veget_max = np.asarray([[0.0, 0.3, 0.25]], dtype=np.float64)
    glcc_pftmtc = np.zeros((npts, nvm, 2), dtype=np.float64)
    glcc_pftmtc[point, 2, receiver_mtc] = 0.2

    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    biomass[point, target_pft, ILEAF, ICARBON] = 10.0
    biomass[point, target_pft, IROOT, ICARBON] = 10.0
    biomass[point, 2, ILEAF, ICARBON] = 20.0
    biomass[point, 2, IROOT, ICARBON] = 20.0
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[target_pft, ILEAF, ICARBON] = 1.0
    bm_sapl[target_pft, IROOT, ICARBON] = 1.0
    cn_sapl = np.ones(nvm, dtype=np.float64)
    cn_sapl[target_pft] = 0.5
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[target_pft] = True

    carbon = np.ones((npts, NCARB, nvm), dtype=np.float64) * 3.0
    deep = np.ones((npts, ndeep, nvm), dtype=np.float64) * 4.0
    litter = np.ones((npts, NLITT, nvm, nlevels, nelements), dtype=np.float64) * 5.0
    lignin_struc = np.ones((npts, nvm, nlevels), dtype=np.float64) * 0.2
    litter_pro = np.ones((NLITT, nlevels, nelements), dtype=np.float64)
    litter_pro[ISTRUCTURAL, :, ICARBON] = [2.0, 2.0]

    result = lcc_apply_collected_proxy_to_target_step(
        point=point,
        receiver_mtc=receiver_mtc,
        target_pft=target_pft,
        glcc_pftmtc=glcc_pftmtc,
        age_class_pfts=[target_pft, 2],
        veget_max_pro=0.2,
        carbon_pro=np.ones(NCARB, dtype=np.float64) * 0.6,
        litter_pro=litter_pro,
        lignin_struc_pro=np.ones(nlevels, dtype=np.float64) * 0.3,
        bm_to_litter_pro=np.ones((NPARTS, nelements), dtype=np.float64) * 0.7,
        deepC_a_pro=np.ones(ndeep, dtype=np.float64) * 0.8,
        deepC_s_pro=np.ones(ndeep, dtype=np.float64) * 0.9,
        deepC_p_pro=np.ones(ndeep, dtype=np.float64) * 1.0,
        fuel_1hr_pro=np.ones((NLITT, nelements), dtype=np.float64) * 1.1,
        fuel_10hr_pro=np.ones((NLITT, nelements), dtype=np.float64) * 1.2,
        fuel_100hr_pro=np.ones((NLITT, nelements), dtype=np.float64) * 1.3,
        fuel_1000hr_pro=np.ones((NLITT, nelements), dtype=np.float64) * 1.4,
        co2_to_bm_pro=2.0,
        gpp_daily_pro=2.1,
        npp_daily_pro=2.2,
        resp_maint_pro=2.3,
        resp_growth_pro=2.4,
        resp_hetero_pro=2.5,
        co2_fire_pro=2.6,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        is_tree=is_tree,
        npp_longterm_init=12.0,
        veget_max=veget_max,
        carbon=carbon,
        litter=litter,
        lignin_struc=lignin_struc,
        bm_to_litter=np.ones((npts, nvm, NPARTS, nelements), dtype=np.float64) * 6.0,
        deepC_a=deep,
        deepC_s=deep + 1.0,
        deepC_p=deep + 2.0,
        fuel_1hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 7.0,
        fuel_10hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 8.0,
        fuel_100hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 9.0,
        fuel_1000hr=np.ones((npts, nvm, NLITT, nelements), dtype=np.float64) * 10.0,
        biomass=biomass,
        co2_to_bm=np.ones((npts, nvm), dtype=np.float64),
        npp_longterm=np.ones((npts, nvm), dtype=np.float64),
        ind=np.ones((npts, nvm), dtype=np.float64),
        lm_lastyearmax=np.ones((npts, nvm), dtype=np.float64),
        age=np.ones((npts, nvm), dtype=np.float64),
        everywhere=np.zeros((npts, nvm), dtype=np.float64),
        leaf_frac=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        leaf_age=np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        pft_present=np.zeros((npts, nvm), dtype=bool),
        senescence=np.ones((npts, nvm), dtype=bool),
        gpp_daily=np.ones((npts, nvm), dtype=np.float64),
        npp_daily=np.ones((npts, nvm), dtype=np.float64),
        resp_maint=np.ones((npts, nvm), dtype=np.float64),
        resp_growth=np.ones((npts, nvm), dtype=np.float64),
        resp_hetero=np.ones((npts, nvm), dtype=np.float64),
        co2_fire=np.ones((npts, nvm), dtype=np.float64),
        min_stomate=1.0e-8,
    )

    assert np.allclose(np.asarray(result.veget_max_after_source_subtract)[point], [0.0, 0.3, 0.05])
    assert np.array_equal(np.asarray(result.exhausted_sources), [False, False, False])
    assert np.allclose(np.asarray(result.biomass_pro)[ILEAF, ICARBON], 0.4)
    assert np.allclose(np.asarray(result.co2_to_bm_pro_after_sap_take), 2.0)
    assert np.allclose(np.asarray(result.state.veget_max)[point, target_pft], 0.5)
    assert np.allclose(np.asarray(result.state.biomass)[point, target_pft, ILEAF, ICARBON], 6.2)
    assert bool(np.asarray(result.state.pft_present)[point, target_pft])
    assert not bool(np.asarray(result.state.senescence)[point, target_pft])


def test_litter_availability_fraction_and_writeback_match_grazing_litter_branch():
    litter_above = np.asarray(
        [
            [
                [100.0, 80.0, 40.0, 0.0],
                [50.0, 20.0, 10.0, 12.0],
            ]
        ],
        dtype=np.float64,
    )
    litter_not_avail = np.asarray(
        [
            [
                [999.0, 20.0, 5.0, 0.0],
                [999.0, 25.0, 3.0, 18.0],
            ]
        ],
        dtype=np.float64,
    )
    is_tree = np.asarray([False, False, True, False], dtype=bool)
    natural = np.asarray([False, False, False, True], dtype=bool)
    is_grassland_manag = np.asarray([False, True, False, False], dtype=bool)
    is_grassland_grazed = np.asarray([False, True, False, False], dtype=bool)
    is_grassland_cut = np.asarray([False, False, False, False], dtype=bool)

    frac = litter_availability_fraction_step(
        litter_above_carbon=litter_above,
        litter_not_avail=litter_not_avail,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        is_grassland_grazed=is_grassland_grazed,
        is_grassland_cut=is_grassland_cut,
    )
    frac = np.asarray(frac)

    expected_frac = np.zeros_like(litter_above)
    expected_frac[0, :, 1] = [0.75, 0.0]
    expected_frac[0, :, 2] = 1.0
    expected_frac[0, :, 3] = [0.0, 0.0]
    assert np.allclose(frac, expected_frac)

    result = stomate_lpj_litter_availability_from_fraction(
        litter_above_carbon=litter_above,
        litter_avail_frac=frac,
    )

    assert np.allclose(np.asarray(result.litter_avail), litter_above * expected_frac)
    assert np.allclose(np.asarray(result.litter_not_avail), litter_above * (1.0 - expected_frac))


def test_stomate_lpj_output_diagnostics_matches_fortran_pool_sums_and_mass_loop():
    npts, nvm, ndeep = 1, 3, 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    turnover = np.zeros_like(biomass)
    bm_to_litter = np.zeros_like(biomass)
    for part in range(NPARTS):
        biomass[0, 1, part, ICARBON] = part + 1.0
        turnover[0, 1, part, ICARBON] = (part + 1.0) * 0.1
        bm_to_litter[0, 1, part, ICARBON] = (part + 1.0) * 0.2
    litter_above = np.zeros((npts, NLITT, nvm, 1), dtype=np.float64)
    litter_below = np.zeros((npts, NLITT, nvm, ndeep, 1), dtype=np.float64)
    litter_above[0, ISTRUCTURAL, 1, ICARBON] = 2.0
    litter_above[0, IMETABOLIC, 1, ICARBON] = 3.0
    litter_below[0, ISTRUCTURAL, 1, :, ICARBON] = [5.0, 7.0]
    litter_below[0, IMETABOLIC, 1, :, ICARBON] = [11.0, 13.0]
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    carbon_32l[0, :, 1, :] = [[17.0, 19.0], [23.0, 29.0], [31.0, 37.0]]
    doc = np.zeros((npts, nvm, ndeep, 2, NPOOL, 1), dtype=np.float64)
    doc[0, 1, :, 0, :, ICARBON] = 0.5
    doc[0, 1, :, 1, :, ICARBON] = 0.25
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, 1] = 0.4
    z_soil = np.asarray([1.0], dtype=np.float64)
    zf_soil = np.asarray([0.0, 0.5, 2.0], dtype=np.float64)

    result = stomate_lpj_output_diagnostics(
        biomass=biomass,
        turnover_daily=turnover,
        bm_to_litter=bm_to_litter,
        litter_above=litter_above,
        litter_below=litter_below,
        carbon_32l=carbon_32l,
        doc=doc,
        veget_max=veget_max,
        z_soil=z_soil,
        zf_soil=zf_soil,
        carb_mass_total_old=np.asarray([100.0], dtype=np.float64),
        prod10_total=np.asarray([2.0], dtype=np.float64),
        prod100_total=np.asarray([3.0], dtype=np.float64),
        contfrac=np.asarray([0.75], dtype=np.float64),
        one_day=10.0,
    )

    expected_litter = np.zeros((npts, nvm, ndeep), dtype=np.float64)
    expected_litter[0, 1, 0] = 2.0 + 3.0 + 5.0 + 11.0
    expected_litter[0, 1, 1] = 7.0 + 13.0
    expected_soil = np.zeros_like(expected_litter)
    expected_soil[0, 1, :] = np.sum(carbon_32l[0, :, 1, :], axis=0) + NPOOL * (0.5 + 0.25)
    expected_live = np.sum(biomass[:, :, :, :], axis=2)
    expected_turnover = np.sum(turnover[:, :, :, :], axis=2)
    expected_bm_litter = np.sum(bm_to_litter[:, :, :, :], axis=2)
    expected_mass = 100.0
    for pft in range(nvm):
        for layer in range(z_soil.size):
            expected_mass += (
                expected_live[0, pft, ICARBON] * veget_max[0, pft]
                + (expected_litter[0, pft, layer] + expected_soil[0, pft, layer])
                * veget_max[0, pft]
                * (z_soil[layer] / z_soil[-1])
                + 5.0
            )

    assert np.allclose(np.asarray(result.tot_litter_carb), expected_litter)
    assert np.allclose(np.asarray(result.tot_soil_carb), expected_soil)
    assert np.allclose(np.asarray(result.tot_live_biomass), expected_live)
    assert np.allclose(np.asarray(result.tot_turnover), expected_turnover)
    assert np.allclose(np.asarray(result.tot_bm_to_litter), expected_bm_litter)
    assert np.allclose(np.asarray(result.carb_mass_total), [expected_mass])
    assert np.allclose(np.asarray(result.carb_mass_variation), [expected_mass - 100.0])
    assert np.allclose(np.asarray(result.carbon_32l_pftmean)[0, :, :], carbon_32l[0, :, 1, :] * 0.4)
    assert np.allclose(np.asarray(result.carbon_32l_conct)[0, :, :], carbon_32l[0, :, 1, :] * 0.4 / np.asarray([0.5, 1.5]))
    assert np.allclose(np.asarray(result.free_doc)[0, :], [NPOOL * 0.5 * 0.4 / 1.0, NPOOL * 0.5 * 0.4 / 1.5])
    assert np.allclose(np.asarray(result.free_doc_stock)[0, :], [NPOOL * 0.5 * 0.4, NPOOL * 0.5 * 0.4])
    assert np.allclose(np.asarray(result.adsorbed_doc)[0, :], [NPOOL * 0.25 * 0.4 / 0.5, NPOOL * 0.25 * 0.4 / 1.5])
    expected_f_veg_litter = np.sum((expected_bm_litter[0, :, ICARBON] + expected_turnover[0, :, ICARBON]) * veget_max[0, :]) / 1.0e3 / 10.0 * 0.75
    assert np.allclose(np.asarray(result.f_veg_litter), [expected_f_veg_litter])
