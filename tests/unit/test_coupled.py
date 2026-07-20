from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.coupled import (  # noqa: E402
    StomateOkLeakBoundaryInputs,
    local_diffuco_sechiba_stomate_pft14_explicit_step,
    paper_case_inactive_crop_input_kwargs,
    paper_case_stomate_boundary_input_kwargs,
    paper_case_stomate_bundle_source_kwargs,
    paper_case_stomate_data_input_kwargs,
    paper_case_stomate_pre_step_input_kwargs,
    sechiba_ld_doc_routing_ok_leak_inputs,
    sechiba_stomate_output_explicit_step,
    sechiba_stomate_ok_leak_explicit_step,
    sechiba_stomate_pft14_explicit_step,
    stomate_active_layer_ok_leak_inputs,
    stomate_doc_transport_ok_leak_inputs,
    stomate_litter_controls_ok_leak_inputs,
    stomate_mc_peat_boundary_inputs,
    stomate_ok_leak_boundary_inputs,
    stomate_perma_peat_ok_leak_inputs,
    stomate_permafrost_activity_ok_leak_inputs,
    stomate_pft_static_ok_leak_inputs,
    stomate_pre_step_ok_leak_boundary_inputs,
    paper_case_stomate_static_input_kwargs,
    paper_case_season_time_scales,
    stomate_season_biometeorology_input_kwargs,
    stomate_restart_season_memory_state,
    stomate_restart_season_input_kwargs,
    stomate_season_annual_input_kwargs,
    stomate_season_memory_input_kwargs,
    stomate_same_step_ok_leak_boundary_inputs,
    stomate_soilwater_31mm_ok_leak_inputs,
    stomate_soil_mc_32l_ok_leak_inputs,
    stomate_static_routing_ok_leak_inputs,
    stomate_tf_doc_ok_leak_inputs,
    stomate_vertical_ok_leak_inputs,
    stomate_restart_ok_leak_state_inputs,
    stomate_restart_input_bundles,
)
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state  # noqa: E402
from jax_orchidee.driver.init import read_run_scalars  # noqa: E402
from jax_orchidee.sechiba.sechiba_step import sechiba_explicit_coupled_step  # noqa: E402
from jax_orchidee.stomate.carbon_kernels import (  # noqa: E402
    IACT,
    ICARBON,
    ICARBRES,
    ILEAF,
    IFRUIT,
    IMETABOLIC,
    IROOT,
    ISAPABOVE,
    ISTRUCTURAL,
    NCARB,
    NLEAFAGES,
    NLITT,
    NPARTS,
    NPOOL,
    SENESCENCE_NONE,
    ISLO,
)
from jax_orchidee.stomate.integration import (  # noqa: E402
    stomate_daily_scheduled_gpp_maintenance_prescribe_constraints_alloc_kill_gap_turnover_explicit,
    stomate_lpj_outputs_from_post_npp_ok_leak_explicit,
    stomate_ok_leak_from_post_npp_explicit,
)
from jax_orchidee.stomate.permafrost import stomate_permafrost_decomposition_controls  # noqa: E402
from jax_orchidee.stomate.reference import (  # noqa: E402
    find_stomate_reference_files,
    read_stomate_daily_accumulator_state,
    read_stomate_restart_entry_state,
    read_stomate_restart_season_state,
    stomate_cold_start_daily_accumulator_state,
    stomate_cold_start_entry_state,
    stomate_cold_start_season_state,
)
from tests.unit.test_sechiba_step import (  # noqa: E402
    _MiniDriverPayload,
    _downstream_sechiba_inputs,
    _enerbil_inputs,
    _local_diffuco_boundary_bundle,
    _local_diffuco_payload_for_enerbil,
    _stomate_driver_source,
)


PFT14 = 13
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def _minimal_stomate_chain_inputs(sechiba_result):
    npts, nvm, nslm = 1, 3, 7
    active = 2
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, active, ILEAF, ICARBON] = 20.0
    biomass[0, active, IROOT, ICARBON] = 10.0
    biomass[0, active, ICARBRES, ICARBON] = 50.0
    leaf_age = np.zeros((npts, nvm, 4), dtype=np.float64)
    leaf_age[0, active, :] = [10.0, 20.0, 40.0, 80.0]
    leaf_frac = np.zeros_like(leaf_age)
    leaf_frac[0, active, :] = [0.4, 0.3, 0.2, 0.1]
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[0, active] = True
    natural = np.ones(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[active] = True
    sla_calc = np.full((npts, nvm), 0.02, dtype=np.float64)
    age = np.zeros((npts, nvm), dtype=np.float64)
    ind = np.zeros((npts, nvm), dtype=np.float64)
    cn_ind = np.zeros((npts, nvm), dtype=np.float64)
    co2_to_bm = np.zeros((npts, nvm), dtype=np.float64)
    everywhere = np.ones((npts, nvm), dtype=np.float64)
    ind[0, active] = 1.0
    cn_ind[0, active] = 0.4
    prescribe_inputs = {
        "veget_max": sechiba_result.slowproc.vegetation.veget_max,
        "dt_days": 1.0,
        "pft_present": pft_present,
        "everywhere": everywhere,
        "when_growthinit": np.ones((npts, nvm), dtype=np.float64) * 20.0,
        "biomass": biomass,
        "leaf_frac": leaf_frac,
        "ind": ind,
        "cn_ind": cn_ind,
        "co2_to_bm": co2_to_bm,
        "natural": natural,
        "pasture": np.zeros(nvm, dtype=bool),
        "is_tree": is_tree,
        "bm_sapl": np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        "maxdia": np.ones(nvm, dtype=np.float64),
        "pheno_is_none": np.ones(nvm, dtype=bool),
    }
    constraints_inputs = {
        "adapted": np.ones((npts, nvm), dtype=np.float64),
        "regenerate": np.ones((npts, nvm), dtype=np.float64),
        "t2m_month": np.full(npts, 293.15, dtype=np.float64),
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tseason": np.full(npts, 12.0, dtype=np.float64),
        "natural": natural,
        "agriculture": np.zeros(nvm, dtype=bool),
        "is_peat": is_peat,
        "pheno_is_none": np.ones(nvm, dtype=bool),
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "tcm_crit": np.full(nvm, np.inf, dtype=np.float64),
        "is_tree": is_tree,
    }
    phenology_inputs = {
        "pheno_model": ("none", "none", "none"),
        "leaf_age": leaf_age,
    }
    alloc_inputs = {
        "lai": np.zeros((npts, nvm), dtype=np.float64),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "moiavail_week": np.full((npts, nvm), 0.8, dtype=np.float64),
        "tsoil_month": np.full((npts, nslm), 293.15, dtype=np.float64),
        "soilhum_month": np.full((npts, nslm), 0.8, dtype=np.float64),
        "age": age,
        "leaf_age": leaf_age,
        "z_soil": np.linspace(0.0, 7.0, nslm + 1, dtype=np.float64),
        "sla_calc": sla_calc,
        "natural": natural,
        "pasture": np.zeros(nvm, dtype=bool),
        "is_tree": is_tree,
        "ok_LAIdev": np.zeros(nvm, dtype=bool),
        "r0": np.full(nvm, 0.35, dtype=np.float64),
        "s0": np.full(nvm, 0.35, dtype=np.float64),
        "ext_coeff": np.full(nvm, 0.5, dtype=np.float64),
        "lai_max": np.full(nvm, 12.0, dtype=np.float64),
        "lai_max_to_happy": np.full(nvm, 0.5, dtype=np.float64),
        "tau_leafinit": np.full(nvm, 10.0, dtype=np.float64),
        "alloc_min": np.full(nvm, 0.2, dtype=np.float64),
        "alloc_max": np.full(nvm, 0.8, dtype=np.float64),
        "demi_alloc": np.full(nvm, 100.0, dtype=np.float64),
        "alloc_agr_st": np.zeros(nvm, dtype=np.float64),
        "alloc_agr_pn": np.zeros(nvm, dtype=np.float64),
    }
    alloc_inputs["alloc_agr_st"][active] = 0.25
    alloc_inputs["alloc_agr_pn"][active] = 0.05
    post_npp_inputs = {
        "frac_growthresp": np.zeros(nvm, dtype=np.float64),
        "npp_longterm": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "turnover_longterm": np.zeros_like(biomass),
        "lm_lastyearmax": np.ones((npts, nvm), dtype=np.float64) * 100.0,
        "bm_to_litter": np.zeros_like(biomass),
        "senescence": np.zeros((npts, nvm), dtype=bool),
        "rip_time": np.ones((npts, nvm), dtype=np.float64),
        "height": np.ones((npts, nvm), dtype=np.float64),
        "t2m_min_daily": np.full(npts, 280.0, dtype=np.float64),
        "tmin_spring_time": np.zeros((npts, nvm), dtype=np.float64),
        "herbivores": np.zeros((npts, nvm), dtype=np.float64),
        "maxmoiavail_lastyear": np.ones((npts, nvm), dtype=np.float64),
        "minmoiavail_lastyear": np.zeros((npts, nvm), dtype=np.float64),
        "moiavail_week": np.ones((npts, nvm), dtype=np.float64) * 0.8,
        "t2m_longterm": np.full(npts, 293.15, dtype=np.float64),
        "t2m_month": np.full(npts, 293.15, dtype=np.float64),
        "t2m_week": np.full(npts, 293.15, dtype=np.float64),
        "gdd_from_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "veget_max": sechiba_result.slowproc.vegetation.veget_max,
        "age": age,
        "turnover_time": np.zeros((npts, nvm), dtype=np.float64),
        "nrec": np.zeros((npts, nvm), dtype=np.int32),
        "sla_calc": sla_calc,
        "senescence_type": np.zeros(nvm, dtype=np.int32),
        "is_tree": is_tree,
        "natural": natural,
        "pasture": np.zeros(nvm, dtype=bool),
        "is_grassland_manag": np.zeros(nvm, dtype=bool),
        "ok_laidev": np.zeros(nvm, dtype=bool),
        "availability_fact": np.ones(nvm, dtype=np.float64) * 0.14,
        "residence_time": np.ones(nvm, dtype=np.float64) * 30.0,
        "tmin_crit": np.full(nvm, np.inf, dtype=np.float64),
        "leaf_tab": np.zeros(nvm, dtype=np.int32),
        "pheno_type": np.zeros(nvm, dtype=np.int32),
        "maxdia": np.ones(nvm, dtype=np.float64),
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
        "sla_age1": np.ones((npts, nvm), dtype=np.float64),
        "sla_max": np.ones(nvm, dtype=np.float64) * 0.03,
        "sla_min": np.ones(nvm, dtype=np.float64) * 0.01,
        "ok_dgvm": False,
        "lpj_gap_const_mort": True,
    }
    maintenance_inputs = {
        "gpp_daily_current": np.zeros((npts, nvm), dtype=np.float64),
        "dt_sechiba": 1800.0,
        "dt_stomate": 86400.0,
        "do_slow": True,
        "t2m": np.full(npts, 293.15, dtype=np.float64),
        "t2m_longterm": np.full(npts, 293.15, dtype=np.float64),
        "z_soil": alloc_inputs["z_soil"],
        "rprof": np.ones((npts, nvm), dtype=np.float64),
        "sla_calc": sla_calc,
        "coeff_maint_zero": np.zeros((nvm, NPARTS), dtype=np.float64),
        "maint_resp_slope": np.zeros((nvm, 3), dtype=np.float64),
        "ext_coeff": np.ones(nvm, dtype=np.float64) * 0.5,
        "is_tree": is_tree,
        "resp_maint_part_current": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
    }
    maintenance_inputs["coeff_maint_zero"][active, :] = 0.01
    return {
        "prescribe_inputs": prescribe_inputs,
        "constraints_inputs": constraints_inputs,
        "phenology_inputs": phenology_inputs,
        "alloc_inputs": alloc_inputs,
        "post_npp_inputs": post_npp_inputs,
        "maintenance_inputs": maintenance_inputs,
    }


def _minimal_ok_leak_inputs(npts=1, nvm=3, nslm=7, ndeep=7, nstm=6):
    fuel_shape = (npts, nvm, NLITT, 1)
    pref_soil_veg = np.arange(nvm, dtype=np.int32)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, 2] = 0.4
    litter_above = np.zeros((npts, NLITT, nvm, 1), dtype=np.float64)
    litter_below = np.zeros((npts, NLITT, nvm, ndeep, 1), dtype=np.float64)
    litter_above[0, IMETABOLIC, 2, ICARBON] = 0.1
    litter_below[0, ISTRUCTURAL, 2, :, ICARBON] = np.linspace(0.01, 0.07, ndeep, dtype=np.float64)
    fbact = np.zeros((npts, ndeep, nvm), dtype=np.float64)
    fbact[0, :, 2] = np.linspace(0.01, 0.02, ndeep, dtype=np.float64)
    carbon_32l = np.zeros((npts, NCARB, nvm, ndeep), dtype=np.float64)
    carbon_32l[0, :, 2, :] = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64)
    doc = np.zeros((npts, nvm, ndeep, 2, NPOOL, 1), dtype=np.float64)
    doc[0, 2, :, 0, 4, ICARBON] = np.linspace(0.1, 0.2, ndeep, dtype=np.float64)
    wat_flux = np.zeros((npts, nslm, nstm), dtype=np.float64)
    wat_flux[0, :, 2] = np.linspace(0.05, 0.0, nslm, dtype=np.float64)
    natural = np.zeros(nvm, dtype=bool)
    natural[2] = True
    is_peat = np.zeros(nvm, dtype=bool)
    is_peat[2] = True
    return {
        "litter_above": litter_above,
        "litter_below": litter_below,
        "lignin_struc_above": np.full((npts, nvm), 0.25, dtype=np.float64),
        "lignin_struc_below": np.full((npts, nvm, ndeep), 0.35, dtype=np.float64),
        "litterpart": np.zeros((npts, nvm, NLITT), dtype=np.float64),
        "dead_leaves": np.zeros((npts, nvm, NLITT), dtype=np.float64),
        "fuel_1hr": np.zeros(fuel_shape, dtype=np.float64),
        "fuel_10hr": np.zeros(fuel_shape, dtype=np.float64),
        "fuel_100hr": np.zeros(fuel_shape, dtype=np.float64),
        "fuel_1000hr": np.zeros(fuel_shape, dtype=np.float64),
        "rprof": np.ones((npts, nvm), dtype=np.float64),
        "veget_max": veget_max,
        "sla_calc": np.full((npts, nvm), 0.1, dtype=np.float64),
        "fbact_litter": fbact,
        "poor_soils": np.zeros(npts, dtype=np.float64),
        "flood_frac": np.zeros(npts, dtype=np.float64),
        "carbon_32l": carbon_32l,
        "doc": doc,
        "doc_precip2ground": np.zeros((npts, nvm, 1), dtype=np.float64),
        "doc_precip2canopy": np.zeros((npts, nvm, 1), dtype=np.float64),
        "dry_dep_canopy": np.zeros((npts, nvm, 1), dtype=np.float64),
        "interception_storage": np.zeros((npts, nvm, 1), dtype=np.float64),
        "doc_to_topsoil": np.zeros((npts, 2), dtype=np.float64),
        "doc_to_subsoil": np.zeros((npts, 2), dtype=np.float64),
        "fbact_doc": fbact,
        "fbact_soilcarbon": fbact,
        "soilwater_31mm": np.ones((npts, nstm), dtype=np.float64),
        "fastr": np.zeros(npts, dtype=np.float64),
        "pref_soil_veg": pref_soil_veg,
        "tprof": np.full((npts, ndeep, nvm), 273.15, dtype=np.float64),
        "clay": np.asarray([0.3], dtype=np.float64),
        "bulk_dens": np.asarray([1.65], dtype=np.float64),
        "z_soil": np.linspace(0.0, float(nslm), nslm + 1, dtype=np.float64),
        "zf_soil_b": np.linspace(0.0, float(ndeep), ndeep + 1, dtype=np.float64),
        "flux_red": np.ones(npts, dtype=np.float64),
        "natural": natural,
        "is_peat": is_peat,
        "is_c4": np.zeros(nvm, dtype=bool),
        "control_temp_above": np.ones((npts, NLITT), dtype=np.float64),
        "control_moist_above": np.ones((npts, nvm), dtype=np.float64),
        "soil_mc_top_by_pft": np.ones((npts, nvm), dtype=np.float64),
        "dif_doc": np.asarray([1.0e-5], dtype=np.float64),
        "nslm": nslm,
        "ndeep": ndeep,
        "sro_bottom": nslm,
    }


def test_stomate_pft_static_ok_leak_inputs_converts_fortran_tile_indices():
    mapped = stomate_pft_static_ok_leak_inputs(
        pref_soil_veg=np.asarray([1, 4, 6], dtype=np.int32),
        natural=np.asarray([True, False, True], dtype=bool),
        is_peat=np.asarray([False, True, False], dtype=bool),
        is_c4=np.asarray([False, False, True], dtype=bool),
    )

    np.testing.assert_array_equal(mapped.ok_leak_inputs["pref_soil_veg"], np.asarray([0, 3, 5], dtype=np.int32))
    np.testing.assert_array_equal(mapped.ok_leak_inputs["natural"], np.asarray([True, False, True], dtype=bool))
    np.testing.assert_array_equal(mapped.ok_leak_inputs["is_peat"], np.asarray([False, True, False], dtype=bool))
    np.testing.assert_array_equal(mapped.ok_leak_inputs["is_c4"], np.asarray([False, False, True], dtype=bool))
    assert mapped.provenance[1].endswith("pft_parameters.f90 lines 247-293")


def test_stomate_pft_static_ok_leak_inputs_accepts_run_scalars_for_pft14_case():
    scalars = read_run_scalars(CONFIG)

    mapped = stomate_pft_static_ok_leak_inputs(run_scalars=scalars)

    assert mapped.ok_leak_inputs["pref_soil_veg"].shape == (14,)
    assert mapped.ok_leak_inputs["pref_soil_veg"][PFT14] == 3
    assert bool(mapped.ok_leak_inputs["natural"][PFT14]) is True
    assert bool(mapped.ok_leak_inputs["is_peat"][PFT14]) is True
    assert bool(mapped.ok_leak_inputs["is_c4"][PFT14]) is False


def test_stomate_pft_static_ok_leak_inputs_rejects_bad_static_arrays():
    with np.testing.assert_raises_regex(ValueError, "Fortran one-based"):
        stomate_pft_static_ok_leak_inputs(
            pref_soil_veg=np.asarray([0, 2], dtype=np.int32),
            natural=np.ones(2, dtype=bool),
            is_peat=np.zeros(2, dtype=bool),
            is_c4=np.zeros(2, dtype=bool),
        )
    with np.testing.assert_raises_regex(ValueError, "shape"):
        stomate_pft_static_ok_leak_inputs(
            pref_soil_veg=np.asarray([1, 2], dtype=np.int32),
            natural=np.ones(3, dtype=bool),
            is_peat=np.zeros(2, dtype=bool),
            is_c4=np.zeros(2, dtype=bool),
        )


def test_sechiba_stomate_pft14_explicit_step_matches_manual_stomate_chain():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    extra_entry_sources = (
        {
            "temp_sol": all_enerbil_inputs["temp_sol"],
            "flood_frac": np.zeros(1, dtype=np.float64),
            "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
            "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
            "fastr": np.zeros(1, dtype=np.float64),
        },
    )

    result = sechiba_stomate_pft14_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
    )
    manual = stomate_daily_scheduled_gpp_maintenance_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        gpp=sechiba.diffuco_payload["gpp"],
        veget_max=sechiba.slowproc.vegetation.veget_max,
        totfrac_nobio=sechiba.slowproc.vegetation.totfrac_nobio,
        stempdiag=sechiba.thermosoil_payload["stempdiag"],
        flood_frac=np.zeros(1, dtype=np.float64),
        **chain["maintenance_inputs"],
        prescribe_inputs={
            **chain["prescribe_inputs"],
            "veget_max": sechiba.slowproc.vegetation.veget_max,
        },
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
    )

    assert result.provenance[1].endswith("slowproc.f90::slowproc_main lines 973-1023")
    assert "gpp" in result.entry_payload.covered_inputs
    assert "stempdiag" in result.entry_payload.covered_inputs
    assert "biomass" in result.missing_entry_inputs
    np.testing.assert_allclose(
        np.asarray(result.stomate.resp_maint_part),
        np.asarray(manual.resp_maint_part),
    )
    np.testing.assert_allclose(
        np.asarray(result.stomate.scheduled_gpp_chain.gpp_daily),
        np.asarray(manual.scheduled_gpp_chain.gpp_daily),
    )
    np.testing.assert_allclose(
        np.asarray(result.stomate.scheduled_gpp_chain.chain.allocation.f_alloc),
        np.asarray(manual.scheduled_gpp_chain.chain.allocation.f_alloc),
    )
    np.testing.assert_allclose(
        np.asarray(result.stomate.scheduled_gpp_chain.chain.post_npp.daily_carbon.npp_update.biomass),
        np.asarray(manual.scheduled_gpp_chain.chain.post_npp.daily_carbon.npp_update.biomass),
    )


def test_local_diffuco_sechiba_stomate_pft14_explicit_step_matches_two_stage_chain():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_bundle = _local_diffuco_boundary_bundle(all_enerbil_inputs)
    local_diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, local_diffuco_payload)
    manual_sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=local_diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(manual_sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, manual_sechiba)
    extra_entry_sources = (
        {
            "temp_sol": all_enerbil_inputs["temp_sol"],
            "flood_frac": np.zeros(1, dtype=np.float64),
            "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
            "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
            "fastr": np.zeros(1, dtype=np.float64),
        },
    )
    manual_coupled = sechiba_stomate_pft14_explicit_step(
        sechiba=manual_sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
    )

    combined = local_diffuco_sechiba_stomate_pft14_explicit_step(
        diffuco_inputs=diffuco_bundle["inputs"],
        pft_output_backgrounds=diffuco_bundle["backgrounds"],
        diffuco_passthrough=diffuco_bundle["passthrough"],
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
    )

    assert combined.provenance[1].endswith("diffuco.f90::diffuco_main lines 629-710")
    np.testing.assert_allclose(
        np.asarray(combined.local_sechiba.step.thermosoil_payload["stempdiag"]),
        np.asarray(manual_sechiba.thermosoil_payload["stempdiag"]),
    )
    np.testing.assert_allclose(
        np.asarray(combined.coupled.stomate.scheduled_gpp_chain.chain.post_npp.turnover.biomass),
        np.asarray(manual_coupled.stomate.scheduled_gpp_chain.chain.post_npp.turnover.biomass),
    )


def test_sechiba_stomate_ok_leak_explicit_step_matches_manual_two_stage_chain():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    extra_entry_sources = (
        {
            "temp_sol": all_enerbil_inputs["temp_sol"],
            "flood_frac": np.zeros(1, dtype=np.float64),
            "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
            "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
            "fastr": np.zeros(1, dtype=np.float64),
        },
    )
    ok_leak_inputs = _minimal_ok_leak_inputs()
    manual_coupled = sechiba_stomate_pft14_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
    )
    manual_ok_leak = stomate_ok_leak_from_post_npp_explicit(
        post_npp=manual_coupled.stomate.scheduled_gpp_chain.chain.post_npp,
        resp_maint_part_radia=manual_coupled.stomate.maintenance.resp_maint_part,
        flood_root_radia=manual_coupled.stomate.flood_root_radia,
        soil_mc=sechiba.hydrol_diagnostics.mc_layh_s,
        dt_sechiba=chain["maintenance_inputs"]["dt_sechiba"],
        wat_flux=sechiba.hydrol_outputs.wat_flux,
        runoff_per_soil=sechiba.hydrol_outputs.runoff_per_soil,
        drainage_per_soil=sechiba.hydrol_outputs.drainage_per_soil,
        runoff2peat=sechiba.hydrol_outputs.runoff2peat,
        canopy2ground=sechiba.hydrol_outputs.canopy2ground,
        **ok_leak_inputs,
    )

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
    )

    assert result.provenance[-1].endswith("soilcarbon_leak lines 1176-2303")
    np.testing.assert_allclose(
        np.asarray(result.coupled.stomate.scheduled_gpp_chain.chain.post_npp.turnover.turnover),
        np.asarray(manual_coupled.stomate.scheduled_gpp_chain.chain.post_npp.turnover.turnover),
    )
    np.testing.assert_allclose(
        np.asarray(result.ok_leak.prep.turnover_littercalc),
        np.asarray(manual_ok_leak.prep.turnover_littercalc),
    )
    np.testing.assert_allclose(
        np.asarray(result.ok_leak.ok_leak.soilcarbon.doc),
        np.asarray(manual_ok_leak.ok_leak.soilcarbon.doc),
    )


def test_sechiba_stomate_ok_leak_explicit_step_derives_tf_doc_from_chain_sources():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    chain["maintenance_inputs"]["is_tree"][2] = True
    chain["constraints_inputs"]["is_tree"][2] = True
    chain["post_npp_inputs"]["is_tree"][2] = True
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key not in {"doc_precip2ground", "doc_precip2canopy", "dry_dep_canopy", "interception_storage"}
    }
    extra_entry_sources = (
        {
            "temp_sol": all_enerbil_inputs["temp_sol"],
            "flood_frac": np.zeros(1, dtype=np.float64),
            "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
            "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
            "fastr": np.zeros(1, dtype=np.float64),
        },
    )

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        tf_doc_boundary={
            "interception_storage": np.full((1, 3, 1), 0.5, dtype=np.float64),
            "ok_tf_doc": True,
        },
    )

    canopy_storage = np.asarray(result.ok_leak.ok_leak.interception_storage)
    assert canopy_storage.shape == (1, 3, 1)
    assert np.isfinite(canopy_storage).all()


def test_sechiba_stomate_ok_leak_explicit_step_derives_perma_peat_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = _minimal_ok_leak_inputs()
    ok_leak_inputs["carbon_32l"][0, :, 2, 0] = [90.0, 30.0, 20.0]

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        perma_peat_boundary={
            "peat_bulk_density": np.full(ok_leak_inputs["ndeep"], 0.2, dtype=np.float64),
            "perma_peat": True,
            "frac1": 0.70,
            "frac2": 0.10,
        },
    )

    perma_peat = result.ok_leak.ok_leak.soilcarbon.perma_peat
    assert perma_peat is not None
    assert np.asarray(perma_peat.deepc_peat).shape == (1, ok_leak_inputs["ndeep"], 3)


def test_sechiba_stomate_ok_leak_explicit_step_derives_active_layer_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = _minimal_ok_leak_inputs()
    ok_leak_inputs["zi_soil"] = np.linspace(0.5, float(ok_leak_inputs["ndeep"]) - 0.5, ok_leak_inputs["ndeep"])

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        active_layer_boundary={
            "restart_altmax": np.zeros((1, 3), dtype=np.float64),
            "restart_fixed_cryoturbation_depth": np.zeros((1, 3), dtype=np.float64),
            "dayno": 10,
            "firstcall_soilcarbon": True,
        },
    )

    assert result.ok_leak.ok_leak.soilcarbon.cryoturbation_coefficients is None
    assert np.asarray(result.ok_leak.ok_leak.soilcarbon.carbon_32l).shape == ok_leak_inputs["carbon_32l"].shape


def test_sechiba_stomate_ok_leak_explicit_step_uses_fixed_cryoturbation_depth_branch():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = _minimal_ok_leak_inputs()
    ok_leak_inputs.update(
        {
            "zi_soil": np.linspace(0.5, float(ok_leak_inputs["ndeep"]) - 0.5, ok_leak_inputs["ndeep"]),
            "ok_cryoturb": True,
            "use_new_cryoturbation": True,
            "use_fixed_cryoturbation_depth": True,
            "cryoturbation_method": 4,
            "cryoturbation_diff_k_in": 86400.0 * 365.0,
        }
    )
    restart_altmax = np.zeros((1, 3), dtype=np.float64)
    restart_altmax[0, 2] = 1.0
    fixed = np.zeros((1, 3), dtype=np.float64)
    fixed[0, 2] = 2.0

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        active_layer_boundary={
            "restart_altmax": restart_altmax,
            "restart_fixed_cryoturbation_depth": fixed,
            "dayno": 10,
            "firstcall_soilcarbon": True,
        },
    )

    coeff = result.ok_leak.ok_leak.soilcarbon.cryoturbation_coefficients
    assert coeff is not None
    np.testing.assert_allclose(np.asarray(coeff.cryoturbation_depth)[0, 2], 2.0)
    np.testing.assert_allclose(np.asarray(coeff.diff_k)[0, :, 2], [1.0, 1.0, 0.875, 0.0, 0.0, 0.0, 0.0])


def test_stomate_soilwater_31mm_ok_leak_inputs_sums_to_sro_bottom():
    soil_mc = np.asarray(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ],
        dtype=np.float64,
    )
    z_soil = np.asarray([0.0, 0.1, 0.4, 1.0], dtype=np.float64)

    first_layer = stomate_soilwater_31mm_ok_leak_inputs(
        soil_mc=soil_mc,
        z_soil=z_soil,
        sro_bottom=1,
    )
    multi_layer = stomate_soilwater_31mm_ok_leak_inputs(
        soil_mc=soil_mc,
        z_soil=z_soil,
        sro_bottom=3,
    )

    np.testing.assert_allclose(first_layer.ok_leak_inputs["soilwater_31mm"], [[0.1, 0.2]])
    np.testing.assert_allclose(
        multi_layer.ok_leak_inputs["soilwater_31mm"],
        [[1.0 * 0.1 + 3.0 * 0.3 + 5.0 * 0.6, 2.0 * 0.1 + 4.0 * 0.3 + 6.0 * 0.6]],
    )
    assert first_layer.provenance[0].endswith("soilcarbon_leak lines 1482-1486")


def test_stomate_soilwater_31mm_ok_leak_inputs_rejects_bad_bounds():
    soil_mc = np.ones((1, 2, 3), dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "at least 1"):
        stomate_soilwater_31mm_ok_leak_inputs(
            soil_mc=soil_mc,
            z_soil=np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
            sro_bottom=0,
        )
    with np.testing.assert_raises_regex(ValueError, "cannot exceed"):
        stomate_soilwater_31mm_ok_leak_inputs(
            soil_mc=soil_mc,
            z_soil=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            sro_bottom=3,
        )
    with np.testing.assert_raises_regex(ValueError, "nondecreasing"):
        stomate_soilwater_31mm_ok_leak_inputs(
            soil_mc=soil_mc,
            z_soil=np.asarray([0.0, 0.2, 0.1], dtype=np.float64),
            sro_bottom=2,
        )


def test_stomate_soil_mc_32l_ok_leak_inputs_extends_hydrology_layers():
    soil_mc = np.asarray(
        [
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        ],
        dtype=np.float64,
    )

    mapped = stomate_soil_mc_32l_ok_leak_inputs(
        soil_mc=soil_mc,
        nslm=2,
        ndeep=4,
    )

    np.testing.assert_allclose(
        np.asarray(mapped.ok_leak_inputs["soil_mc_32l"]),
        [[[0.1, 0.2], [0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]],
    )
    assert mapped.provenance[0].endswith("stomate_main lines 3270-3275")


def test_stomate_mc_peat_boundary_inputs_extends_hydrol_shumdiag_peat():
    shumdiag_peat = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)

    mapped = stomate_mc_peat_boundary_inputs(
        shumdiag_peat=shumdiag_peat,
        nslm=3,
        ndeep=5,
    )

    np.testing.assert_allclose(mapped.mc_peat, [[0.1, 0.2, 0.3, 0.3, 0.3]])
    assert mapped.provenance[0].endswith("stomate_main lines 3040-3052")


def test_stomate_mc_peat_boundary_inputs_rejects_bad_shapes():
    with np.testing.assert_raises_regex(ValueError, "shumdiag_peat"):
        stomate_mc_peat_boundary_inputs(
            shumdiag_peat=np.ones((1, 2, 1), dtype=np.float64),
            nslm=2,
            ndeep=3,
        )
    with np.testing.assert_raises_regex(ValueError, "ndeep"):
        stomate_mc_peat_boundary_inputs(
            shumdiag_peat=np.ones((1, 3), dtype=np.float64),
            nslm=3,
            ndeep=2,
        )


def test_stomate_doc_transport_ok_leak_inputs_computes_flux_red_from_poor_soils():
    mapped = stomate_doc_transport_ok_leak_inputs(
        poor_soils=np.asarray([0.0, 0.25, 1.0], dtype=np.float64),
        flux_red_sro=0.2,
    )

    np.testing.assert_allclose(mapped.ok_leak_inputs["flux_red"], [0.2, 0.4, 1.0])
    assert mapped.provenance[-1].endswith("soilcarbon_leak lines 1204-1205")


def test_stomate_doc_transport_ok_leak_inputs_rejects_bad_shape():
    with np.testing.assert_raises_regex(ValueError, "poor_soils"):
        stomate_doc_transport_ok_leak_inputs(
            poor_soils=np.ones((1, 2), dtype=np.float64),
        )


def test_sechiba_ld_doc_routing_ok_leak_inputs_maps_fortran_daily_fluxes():
    reinfiltration = np.asarray([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    irrigation = np.asarray([[0.001, 0.002], [0.003, 0.004]], dtype=np.float64)
    returnflow = np.asarray([[0.05, 0.06], [0.07, 0.08]], dtype=np.float64)

    mapped = sechiba_ld_doc_routing_ok_leak_inputs(
        reinfiltration=reinfiltration,
        irrigation=irrigation,
        returnflow=returnflow,
        river_routing=True,
        dt_sechiba=1800.0,
    )
    scale = 1000.0 * 86400.0 / 1800.0

    np.testing.assert_allclose(mapped.ok_leak_inputs["doc_to_topsoil"], (reinfiltration + irrigation) * scale)
    np.testing.assert_allclose(mapped.ok_leak_inputs["doc_to_subsoil"], returnflow * scale)
    np.testing.assert_allclose(mapped.entry_source["DOC_to_topsoil"], mapped.ok_leak_inputs["doc_to_topsoil"])
    assert mapped.provenance[0].endswith("sechiba_main lines 1251-1259")

    disabled = sechiba_ld_doc_routing_ok_leak_inputs(
        reinfiltration=reinfiltration,
        irrigation=irrigation,
        returnflow=returnflow,
        river_routing=False,
        dt_sechiba=1800.0,
    )
    np.testing.assert_allclose(disabled.ok_leak_inputs["doc_to_topsoil"], 0.0)
    np.testing.assert_allclose(disabled.ok_leak_inputs["doc_to_subsoil"], 0.0)


def test_sechiba_ld_doc_routing_ok_leak_inputs_rejects_mismatched_flow_shapes():
    with np.testing.assert_raises_regex(ValueError, "share shape"):
        sechiba_ld_doc_routing_ok_leak_inputs(
            reinfiltration=np.ones((1, 2), dtype=np.float64),
            irrigation=np.ones((1, 1), dtype=np.float64),
            returnflow=np.ones((1, 2), dtype=np.float64),
            river_routing=True,
            dt_sechiba=1800.0,
        )


def test_ld_doc_microcase_adds_routing_doc_to_active_and_slow_pools():
    from jax_orchidee.stomate.soilcarbon_kernels import IFREE, soilcarbon_leak_doc_inputs

    npts, nvm, ndeep, nelements = 1, 14, 3, 1
    routing = sechiba_ld_doc_routing_ok_leak_inputs(
        reinfiltration=np.asarray([[0.02, 0.03, 0.04, 0.05]], dtype=np.float64),
        irrigation=np.asarray([[0.01, 0.02, 0.03, 0.04]], dtype=np.float64),
        returnflow=np.asarray([[0.05, 0.06, 0.07, 0.08]], dtype=np.float64),
        river_routing=True,
        dt_sechiba=86400.0,
    )
    doc = np.zeros((npts, nvm, ndeep, 2, NPOOL, nelements), dtype=np.float64)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, PFT14] = 0.5

    updated = np.asarray(
        soilcarbon_leak_doc_inputs(
            doc,
            np.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=np.float64),
            routing.ok_leak_inputs["doc_to_topsoil"],
            routing.ok_leak_inputs["doc_to_subsoil"],
            np.zeros((npts, nvm, nelements), dtype=np.float64),
            veget_max,
            bio_frac=np.asarray([0.5], dtype=np.float64),
            dt_days=1.0,
            nslm=2,
        )
    )

    np.testing.assert_allclose(updated[0, PFT14, 0, IFREE, IACT, ICARBON], 50.0 / 0.5)
    np.testing.assert_allclose(updated[0, PFT14, 1, IFREE, IACT, ICARBON], 60.0 / 0.5)
    np.testing.assert_allclose(updated[0, PFT14, 0, IFREE, ISLO, ICARBON], 70.0 / 0.5)
    np.testing.assert_allclose(updated[0, PFT14, 1, IFREE, ISLO, ICARBON], 70.0 / 0.5)
    np.testing.assert_allclose(updated[0, 0, :, IFREE, :, ICARBON], 0.0)


def test_stomate_ok_leak_boundary_inputs_accepts_ld_doc_routing_override():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    npts, nvm = state.sla_calc.shape
    restart = stomate_restart_ok_leak_state_inputs(
        state=state,
        veget_max=np.full((npts, nvm), 0.2, dtype=np.float64),
    )
    driver = _MiniDriverPayload()
    driver.kjpindex = npts
    driver.nbindex = npts
    driver.kindex = np.arange(1, npts + 1, dtype=np.int32)
    driver.lalo = np.zeros((npts, 2), dtype=np.float64)
    driver.contfrac = np.full(npts, 0.5625, dtype=np.float64)
    driver.resolution = np.ones((npts, 2), dtype=np.float64)
    driver.neighbours = np.full((npts, 8), -1, dtype=np.int32)
    driver.u = np.ones(npts, dtype=np.float64)
    driver.v = np.zeros(npts, dtype=np.float64)
    driver.precip_rain = np.zeros(npts, dtype=np.float64)
    driver.precip_snow = np.zeros(npts, dtype=np.float64)
    driver.swdown = np.ones(npts, dtype=np.float64)
    driver.pb = np.ones(npts, dtype=np.float64)
    driver.clay_frac = np.full(npts, 0.1, dtype=np.float64)
    driver.bulk_dens = np.full(npts, 1.3, dtype=np.float64)
    driver.soil_ph = np.full(npts, 6.0, dtype=np.float64)
    driver.poor_soils = np.zeros(npts, dtype=np.float64)
    static_routing = stomate_static_routing_ok_leak_inputs(
        driver_payload=driver,
        kjit=1,
        nflow=4,
        river_routing=False,
        nbp_glo=driver.kjpindex,
        t2mdiag=np.asarray([280.0], dtype=np.float64),
        temp_sol=np.asarray([281.0], dtype=np.float64),
    )
    ld_doc = sechiba_ld_doc_routing_ok_leak_inputs(
        reinfiltration=np.asarray([[0.02, 0.03, 0.04, 0.05]], dtype=np.float64),
        irrigation=np.asarray([[0.01, 0.02, 0.03, 0.04]], dtype=np.float64),
        returnflow=np.asarray([[0.05, 0.06, 0.07, 0.08]], dtype=np.float64),
        river_routing=True,
        dt_sechiba=86400.0,
    )

    merged = stomate_ok_leak_boundary_inputs(
        restart_state=restart,
        static_routing=static_routing,
        ld_doc_routing=ld_doc,
    )

    np.testing.assert_allclose(merged.ok_leak_inputs["doc_to_topsoil"], ld_doc.ok_leak_inputs["doc_to_topsoil"])
    np.testing.assert_allclose(merged.ok_leak_inputs["doc_to_subsoil"], ld_doc.ok_leak_inputs["doc_to_subsoil"])
    np.testing.assert_allclose(merged.ok_leak_inputs["flood_frac"], static_routing.ok_leak_inputs["flood_frac"])
    assert any("sechiba_main lines 1251-1259" in item for item in merged.provenance)


def test_stomate_permafrost_activity_ok_leak_inputs_maps_microactem_controls():
    tdeep = np.asarray(
        [
            [
                [303.15, 303.15, 303.15],
                [283.15, 283.15, 283.15],
            ]
        ],
        dtype=np.float64,
    )
    hsdeep = np.full_like(tdeep, 0.5)
    zz_deep = np.asarray([0.1, 0.4], dtype=np.float64)
    mc_peat = np.full((1, 2), 0.5, dtype=np.float64)
    poor_soils = np.asarray([0.25], dtype=np.float64)

    mapped = stomate_permafrost_activity_ok_leak_inputs(
        tdeep=tdeep,
        hsdeep=hsdeep,
        zz_deep=zz_deep,
        mc_peat=mc_peat,
        poor_soils=poor_soils,
        frozen_respiration_func=0,
        perma_peat=False,
    )
    expected = stomate_permafrost_decomposition_controls(
        tdeep - 273.15,
        hsdeep,
        zz_deep,
        mc_peat,
        poor_soils,
        frozen_respiration_func=0,
        perma_peat=False,
        agri_peat=False,
    )

    np.testing.assert_allclose(mapped.ok_leak_inputs["fbact_litter"], np.asarray(expected.prmfrst_soilc_tempctrl))
    np.testing.assert_allclose(mapped.ok_leak_inputs["fbact_soilcarbon"], np.asarray(expected.prmfrst_soilc_tempctrl))
    np.testing.assert_allclose(mapped.ok_leak_inputs["fbact_doc"], np.asarray(expected.prmfrst_soilc_tempctrl_doc))
    np.testing.assert_allclose(mapped.ok_leak_inputs["tprof"], tdeep)
    np.testing.assert_allclose(mapped.output_inputs["tprof"], tdeep)
    assert mapped.provenance[-1].endswith("microactem lines 2468-2703")


def test_stomate_permafrost_activity_ok_leak_inputs_rejects_bad_shapes_and_missing_peat_mask():
    tdeep = np.ones((1, 2, 3), dtype=np.float64) * 303.15
    with np.testing.assert_raises_regex(ValueError, "hsdeep"):
        stomate_permafrost_activity_ok_leak_inputs(
            tdeep=tdeep,
            hsdeep=np.ones((1, 2), dtype=np.float64),
            zz_deep=np.ones(2, dtype=np.float64),
            mc_peat=np.ones((1, 2), dtype=np.float64),
            poor_soils=np.ones(1, dtype=np.float64),
            frozen_respiration_func=0,
            perma_peat=False,
        )
    with np.testing.assert_raises_regex(ValueError, "is_peat"):
        stomate_permafrost_activity_ok_leak_inputs(
            tdeep=tdeep,
            hsdeep=np.ones_like(tdeep),
            zz_deep=np.ones(2, dtype=np.float64),
            mc_peat=np.ones((1, 2), dtype=np.float64),
            poor_soils=np.ones(1, dtype=np.float64),
            frozen_respiration_func=0,
            perma_peat=True,
        )


def test_sechiba_stomate_ok_leak_explicit_step_derives_soilwater_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key != "soilwater_31mm"
    }

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        soilwater_boundary={},
    )

    assert np.asarray(result.ok_leak.ok_leak.soilcarbon.doc).shape == ok_leak_inputs["doc"].shape


def test_sechiba_stomate_ok_leak_explicit_step_derives_pft_static_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key not in {"pref_soil_veg", "natural", "is_peat", "is_c4"}
    }

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        pft_static_boundary={
            "pref_soil_veg": np.asarray([1, 2, 3], dtype=np.int32),
            "natural": np.asarray([False, False, True], dtype=bool),
            "is_peat": np.asarray([False, False, True], dtype=bool),
            "is_c4": np.zeros(3, dtype=bool),
        },
    )

    assert np.asarray(result.ok_leak.ok_leak.soilcarbon.doc).shape == ok_leak_inputs["doc"].shape


def test_sechiba_stomate_ok_leak_explicit_step_derives_litter_controls_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key not in {"control_temp_above", "control_moist_above", "soil_mc_top_by_pft"}
    }

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        litter_controls_boundary={
            "pref_soil_veg": np.asarray([1, 2, 3], dtype=np.int32),
            "frozen_respiration_func": 1,
        },
    )

    assert np.asarray(result.ok_leak.ok_leak.soilcarbon.doc).shape == ok_leak_inputs["doc"].shape


def test_sechiba_stomate_ok_leak_explicit_step_accepts_premerged_boundary_inputs():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    extra_entry_sources = (
        {
            "temp_sol": all_enerbil_inputs["temp_sol"],
            "flood_frac": np.zeros(1, dtype=np.float64),
            "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
            "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
            "fastr": np.zeros(1, dtype=np.float64),
        },
    )
    ok_leak_inputs = _minimal_ok_leak_inputs()
    boundary = StomateOkLeakBoundaryInputs(
        ok_leak_inputs=ok_leak_inputs,
        output_inputs={},
        provenance=("test ok leak premerged boundary",),
    )

    split_result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
    )
    bundled_result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        boundary_inputs=boundary,
    )

    np.testing.assert_allclose(
        np.asarray(bundled_result.ok_leak.ok_leak.soilcarbon.doc),
        np.asarray(split_result.ok_leak.ok_leak.soilcarbon.doc),
    )
    np.testing.assert_allclose(
        np.asarray(bundled_result.ok_leak.ok_leak.littercalc.litter_above),
        np.asarray(split_result.ok_leak.ok_leak.littercalc.litter_above),
    )


def test_sechiba_stomate_ok_leak_explicit_step_derives_permafrost_activity_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key not in {"fbact_litter", "fbact_soilcarbon", "fbact_doc", "tprof"}
    }
    ok_leak_inputs["zi_soil"] = np.linspace(0.5, float(ok_leak_inputs["ndeep"]) - 0.5, ok_leak_inputs["ndeep"])

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        permafrost_activity_boundary={
            "frozen_respiration_func": 0,
            "perma_peat": True,
            "is_peat": ok_leak_inputs["is_peat"],
        },
    )

    assert np.asarray(result.ok_leak.ok_leak.soilcarbon.doc).shape == ok_leak_inputs["doc"].shape


def test_sechiba_stomate_ok_leak_explicit_step_requires_mc_peat_for_nonpeat_activity_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key not in {"fbact_litter", "fbact_soilcarbon", "fbact_doc", "tprof"}
    }
    ok_leak_inputs["zi_soil"] = np.linspace(0.5, float(ok_leak_inputs["ndeep"]) - 0.5, ok_leak_inputs["ndeep"])

    with np.testing.assert_raises_regex(ValueError, "mc_peat"):
        sechiba_stomate_ok_leak_explicit_step(
            sechiba=sechiba,
            driver_entry_source=driver_entry_source,
            extra_entry_sources=(
                {
                    "temp_sol": all_enerbil_inputs["temp_sol"],
                    "flood_frac": np.zeros(1, dtype=np.float64),
                    "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                    "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                    "fastr": np.zeros(1, dtype=np.float64),
                },
            ),
            **chain["maintenance_inputs"],
            prescribe_inputs=chain["prescribe_inputs"],
            constraints_inputs=chain["constraints_inputs"],
            phenology_inputs=chain["phenology_inputs"],
            alloc_inputs=chain["alloc_inputs"],
            post_npp_inputs=chain["post_npp_inputs"],
            ok_leak_inputs=ok_leak_inputs,
            permafrost_activity_boundary={
                "frozen_respiration_func": 0,
                "perma_peat": False,
            },
        )


def test_sechiba_stomate_ok_leak_explicit_step_derives_soil_mc_32l_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key != "soil_mc_32l"
    }

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        soil_mc_32l_boundary={},
    )

    assert np.asarray(result.ok_leak.ok_leak.soilcarbon.doc).shape == ok_leak_inputs["doc"].shape


def test_sechiba_stomate_ok_leak_explicit_step_derives_doc_transport_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key != "flux_red"
    }

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        doc_transport_boundary={},
    )

    assert np.asarray(result.ok_leak.ok_leak.soilcarbon.doc).shape == ok_leak_inputs["doc"].shape


def test_sechiba_stomate_ok_leak_explicit_step_uses_source_default_dif_doc():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        key: value
        for key, value in _minimal_ok_leak_inputs().items()
        if key != "dif_doc"
    }
    manual_inputs = _minimal_ok_leak_inputs()
    manual_inputs["dif_doc"] = None

    result = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
    )
    manual = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=(
            {
                "temp_sol": all_enerbil_inputs["temp_sol"],
                "flood_frac": np.zeros(1, dtype=np.float64),
                "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                "fastr": np.zeros(1, dtype=np.float64),
            },
        ),
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=manual_inputs,
    )

    np.testing.assert_allclose(
        np.asarray(result.ok_leak.ok_leak.soilcarbon.doc),
        np.asarray(manual.ok_leak.ok_leak.soilcarbon.doc),
    )


def test_stomate_same_step_ok_leak_boundary_inputs_closes_output_chain_boundary():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    extra_entry_sources = (
        {
            "temp_sol": all_enerbil_inputs["temp_sol"],
            "flood_frac": np.zeros(1, dtype=np.float64),
            "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
            "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
            "fastr": np.zeros(1, dtype=np.float64),
        },
    )
    coupled = sechiba_stomate_pft14_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
    )
    base_ok = _minimal_ok_leak_inputs()
    pre_step_ok = {
        key: value
        for key, value in base_ok.items()
        if key
        not in {
            "control_temp_above",
            "control_moist_above",
            "soil_mc_top_by_pft",
            "doc_precip2ground",
            "doc_precip2canopy",
            "dry_dep_canopy",
            "soilwater_31mm",
            "soil_mc_32l",
            "flux_red",
        }
    }
    pre_step = StomateOkLeakBoundaryInputs(
        ok_leak_inputs=pre_step_ok,
        output_inputs={
            "z_soil": base_ok["z_soil"][1:],
            "zf_soil": base_ok["zf_soil_b"],
            "carb_mass_total_old": np.zeros(1, dtype=np.float64),
        },
        provenance=("test pre-step OK_LEAK boundary",),
    )
    same_step = stomate_same_step_ok_leak_boundary_inputs(
        sechiba=sechiba,
        post_npp=coupled.stomate.scheduled_gpp_chain.chain.post_npp,
        dt_sechiba=chain["maintenance_inputs"]["dt_sechiba"],
        entry_payload=coupled.entry_payload.payload,
        pre_step_boundary=pre_step,
        litter_controls_boundary={
            "pref_soil_veg": np.asarray([1, 2, 3], dtype=np.int32),
            "frozen_respiration_func": 1,
        },
        tf_doc_boundary={
            "ok_tf_doc": False,
        },
        soilwater_boundary={},
        soil_mc_32l_boundary={},
        doc_transport_boundary={},
        is_tree=chain["maintenance_inputs"]["is_tree"],
    )
    bundled = sechiba_stomate_output_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        boundary_inputs=same_step.boundary_inputs,
    )
    split = sechiba_stomate_output_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=same_step.boundary_inputs.ok_leak_inputs,
        output_inputs=same_step.boundary_inputs.output_inputs,
    )

    assert same_step.litter_controls is not None
    assert same_step.tf_doc is not None
    assert same_step.soilwater is not None
    assert same_step.soil_mc_32l is not None
    assert same_step.doc_transport is not None
    assert "control_temp_above" in same_step.boundary_inputs.ok_leak_inputs
    assert "soil_mc_32l" in same_step.boundary_inputs.ok_leak_inputs
    np.testing.assert_allclose(
        np.asarray(bundled.ok_leak_chain.ok_leak.ok_leak.soilcarbon.doc),
        np.asarray(split.ok_leak_chain.ok_leak.ok_leak.soilcarbon.doc),
    )
    np.testing.assert_allclose(
        np.asarray(bundled.outputs.modelout.AGB_model),
        np.asarray(split.outputs.modelout.AGB_model),
    )


def test_sechiba_stomate_ok_leak_explicit_step_rejects_duplicate_tf_doc_sources():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)

    with np.testing.assert_raises_regex(ValueError, "duplicate explicit source fields"):
        sechiba_stomate_ok_leak_explicit_step(
            sechiba=sechiba,
            driver_entry_source=driver_entry_source,
            extra_entry_sources=(
                {
                    "temp_sol": all_enerbil_inputs["temp_sol"],
                    "flood_frac": np.zeros(1, dtype=np.float64),
                    "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                    "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                    "fastr": np.zeros(1, dtype=np.float64),
                },
            ),
            **chain["maintenance_inputs"],
            prescribe_inputs=chain["prescribe_inputs"],
            constraints_inputs=chain["constraints_inputs"],
            phenology_inputs=chain["phenology_inputs"],
            alloc_inputs=chain["alloc_inputs"],
            post_npp_inputs=chain["post_npp_inputs"],
            ok_leak_inputs=_minimal_ok_leak_inputs(),
            tf_doc_boundary={
                "interception_storage": np.zeros((1, 3, 1), dtype=np.float64),
                "ok_tf_doc": False,
            },
        )


def test_sechiba_stomate_ok_leak_rejects_duplicate_hydrol_boundary_fields():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    ok_leak_inputs = {
        **_minimal_ok_leak_inputs(),
        "wat_flux": np.zeros_like(np.asarray(sechiba.hydrol_outputs.wat_flux)),
    }

    with np.testing.assert_raises_regex(ValueError, "HYDROL boundary does not match"):
        sechiba_stomate_ok_leak_explicit_step(
            sechiba=sechiba,
            driver_entry_source=driver_entry_source,
            extra_entry_sources=(
                {
                    "temp_sol": all_enerbil_inputs["temp_sol"],
                    "flood_frac": np.zeros(1, dtype=np.float64),
                    "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                    "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                    "fastr": np.zeros(1, dtype=np.float64),
                },
            ),
            **chain["maintenance_inputs"],
            prescribe_inputs=chain["prescribe_inputs"],
            constraints_inputs=chain["constraints_inputs"],
            phenology_inputs=chain["phenology_inputs"],
            alloc_inputs=chain["alloc_inputs"],
            post_npp_inputs=chain["post_npp_inputs"],
            ok_leak_inputs=ok_leak_inputs,
        )


def test_sechiba_stomate_output_explicit_step_matches_manual_three_stage_chain():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    extra_entry_sources = (
        {
            "temp_sol": all_enerbil_inputs["temp_sol"],
            "flood_frac": np.zeros(1, dtype=np.float64),
            "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
            "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
            "fastr": np.zeros(1, dtype=np.float64),
        },
    )
    ok_leak_inputs = _minimal_ok_leak_inputs()
    vertical_inputs = stomate_vertical_ok_leak_inputs(
        diaglev=ok_leak_inputs["z_soil"][1:],
        zz_coef_deep=ok_leak_inputs["zf_soil_b"][1:],
    )
    output_inputs = {
        **vertical_inputs.output_inputs,
        "carb_mass_total_old": np.zeros(1, dtype=np.float64),
    }
    manual_ok_chain = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
    )
    manual_outputs = stomate_lpj_outputs_from_post_npp_ok_leak_explicit(
        post_npp=manual_ok_chain.coupled.stomate.scheduled_gpp_chain.chain.post_npp,
        ok_leak=manual_ok_chain.ok_leak,
        gpp_daily=manual_ok_chain.coupled.stomate.scheduled_gpp_chain.gpp_daily,
        veget_max=sechiba.slowproc.vegetation.veget_max,
        **output_inputs,
    )

    result = sechiba_stomate_output_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        output_inputs=output_inputs,
    )

    assert result.provenance[-1].endswith("lines 50, 60-76, and 115-118")
    np.testing.assert_allclose(
        np.asarray(result.outputs.output_diagnostics.tot_litter_soil_carb),
        np.asarray(manual_outputs.output_diagnostics.tot_litter_soil_carb),
    )
    np.testing.assert_allclose(
        np.asarray(result.outputs.modelout_fields["LEAF_M"]),
        np.asarray(manual_outputs.modelout_fields["LEAF_M"]),
    )
    np.testing.assert_allclose(
        np.asarray(result.outputs.modelout.AGB_model),
        np.asarray(manual_outputs.modelout.AGB_model),
    )


def test_sechiba_stomate_output_explicit_step_accepts_merged_boundary_bundle():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)
    extra_entry_sources = (
        {
            "temp_sol": all_enerbil_inputs["temp_sol"],
            "flood_frac": np.zeros(1, dtype=np.float64),
            "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
            "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
            "fastr": np.zeros(1, dtype=np.float64),
        },
    )
    ok_leak_inputs = _minimal_ok_leak_inputs()
    vertical_inputs = stomate_vertical_ok_leak_inputs(
        diaglev=ok_leak_inputs["z_soil"][1:],
        zz_coef_deep=ok_leak_inputs["zf_soil_b"][1:],
    )
    output_inputs = {
        **vertical_inputs.output_inputs,
        "carb_mass_total_old": np.zeros(1, dtype=np.float64),
    }
    boundary = StomateOkLeakBoundaryInputs(
        ok_leak_inputs=ok_leak_inputs,
        output_inputs=output_inputs,
        provenance=("test merged boundary",),
    )

    split_result = sechiba_stomate_output_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        ok_leak_inputs=ok_leak_inputs,
        output_inputs=output_inputs,
    )
    bundled_result = sechiba_stomate_output_explicit_step(
        sechiba=sechiba,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        **chain["maintenance_inputs"],
        prescribe_inputs=chain["prescribe_inputs"],
        constraints_inputs=chain["constraints_inputs"],
        phenology_inputs=chain["phenology_inputs"],
        alloc_inputs=chain["alloc_inputs"],
        post_npp_inputs=chain["post_npp_inputs"],
        boundary_inputs=boundary,
    )

    np.testing.assert_allclose(
        np.asarray(bundled_result.outputs.output_diagnostics.tot_litter_soil_carb),
        np.asarray(split_result.outputs.output_diagnostics.tot_litter_soil_carb),
    )
    np.testing.assert_allclose(
        np.asarray(bundled_result.outputs.modelout.AGB_model),
        np.asarray(split_result.outputs.modelout.AGB_model),
    )


def test_sechiba_stomate_output_explicit_step_rejects_duplicate_boundary_bundle_fields():
    boundary = StomateOkLeakBoundaryInputs(
        ok_leak_inputs={"litter_above": np.zeros((1, 2, 3, 1), dtype=np.float64)},
        output_inputs={"carb_mass_total_old": np.zeros(1, dtype=np.float64)},
        provenance=("test duplicate boundary",),
    )
    with np.testing.assert_raises_regex(ValueError, "duplicate explicit source fields"):
        sechiba_stomate_output_explicit_step(
            sechiba=None,
            boundary_inputs=boundary,
            ok_leak_inputs={"litter_above": np.zeros((1, 2, 3, 1), dtype=np.float64)},
            output_inputs={"z_soil": np.ones(2, dtype=np.float64)},
        )
    with np.testing.assert_raises_regex(ValueError, "duplicate explicit source fields"):
        sechiba_stomate_output_explicit_step(
            sechiba=None,
            boundary_inputs=boundary,
            ok_leak_inputs={"z_soil": np.ones(2, dtype=np.float64)},
            output_inputs={"carb_mass_total_old": np.zeros(1, dtype=np.float64)},
        )


def test_sechiba_stomate_output_explicit_step_rejects_duplicate_output_veget_max():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)
    sechiba = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    chain = _minimal_stomate_chain_inputs(sechiba)
    driver_entry_source = _stomate_driver_source(all_enerbil_inputs, downstream, sechiba)

    with np.testing.assert_raises_regex(ValueError, "veget_max"):
        sechiba_stomate_output_explicit_step(
            sechiba=sechiba,
            driver_entry_source=driver_entry_source,
            extra_entry_sources=(
                {
                    "temp_sol": all_enerbil_inputs["temp_sol"],
                    "flood_frac": np.zeros(1, dtype=np.float64),
                    "DOC_to_topsoil": np.zeros((1, 2), dtype=np.float64),
                    "DOC_to_subsoil": np.zeros((1, 2), dtype=np.float64),
                    "fastr": np.zeros(1, dtype=np.float64),
                },
            ),
            **chain["maintenance_inputs"],
            prescribe_inputs=chain["prescribe_inputs"],
            constraints_inputs=chain["constraints_inputs"],
            phenology_inputs=chain["phenology_inputs"],
            alloc_inputs=chain["alloc_inputs"],
            post_npp_inputs=chain["post_npp_inputs"],
            ok_leak_inputs=_minimal_ok_leak_inputs(),
            output_inputs={
                "veget_max": np.zeros_like(np.asarray(sechiba.slowproc.vegetation.veget_max)),
                "z_soil": np.asarray([1.0, 2.0], dtype=np.float64),
                "zf_soil": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
                "carb_mass_total_old": np.zeros(1, dtype=np.float64),
            },
        )


def test_stomate_restart_input_bundles_pass_restart_and_explicit_state_without_defaults():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    npts, nvm, nslm = state.sla_calc.shape[0], state.sla_calc.shape[1], 7
    veget_max = np.full((npts, nvm), 0.2, dtype=np.float64)
    lai = np.full((npts, nvm), 0.7, dtype=np.float64)
    cn_ind = np.full((npts, nvm), 3.0, dtype=np.float64)
    natural = np.ones(nvm, dtype=bool)
    pasture = np.zeros(nvm, dtype=bool)
    is_tree = np.zeros(nvm, dtype=bool)
    is_peat = np.zeros(nvm, dtype=bool)
    pheno_is_none = np.ones(nvm, dtype=bool)
    z_soil = np.linspace(0.0, 7.0, nslm + 1, dtype=np.float64)

    bundles = stomate_restart_input_bundles(
        state=state,
        veget_max=veget_max,
        lai=lai,
        cn_ind=cn_ind,
        dt_days=0.25,
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        is_peat=is_peat,
        pheno_is_none=pheno_is_none,
        bm_sapl=np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        maxdia=np.ones(nvm, dtype=np.float64),
        t2m_month=np.full(npts, 293.15, dtype=np.float64),
        t2m_min_daily=np.full(npts, 280.0, dtype=np.float64),
        tseason=np.full(npts, 12.0, dtype=np.float64),
        tmin_crit=np.full(nvm, np.inf, dtype=np.float64),
        tcm_crit=np.full(nvm, np.inf, dtype=np.float64),
        pheno_model=tuple("none" for _ in range(nvm)),
        moiavail_week=np.full((npts, nvm), 0.8, dtype=np.float64),
        tsoil_month=np.full((npts, nslm), 293.15, dtype=np.float64),
        soilhum_month=np.full((npts, nslm), 0.8, dtype=np.float64),
        z_soil=z_soil,
        ok_laidev=np.zeros(nvm, dtype=bool),
        r0=np.full(nvm, 0.35, dtype=np.float64),
        s0=np.full(nvm, 0.35, dtype=np.float64),
        ext_coeff=np.full(nvm, 0.5, dtype=np.float64),
        lai_max=np.full(nvm, 12.0, dtype=np.float64),
        lai_max_to_happy=np.full(nvm, 0.5, dtype=np.float64),
        tau_leafinit=np.full(nvm, 10.0, dtype=np.float64),
        alloc_min=np.full(nvm, 0.2, dtype=np.float64),
        alloc_max=np.full(nvm, 0.8, dtype=np.float64),
        demi_alloc=np.full(nvm, 100.0, dtype=np.float64),
        alloc_agr_st=np.zeros(nvm, dtype=np.float64),
        alloc_agr_pn=np.zeros(nvm, dtype=np.float64),
        frac_growthresp=np.full(nvm, 0.1, dtype=np.float64),
        height=np.ones((npts, nvm), dtype=np.float64),
        tmin_spring_time=np.zeros((npts, nvm), dtype=np.float64),
        herbivores=np.zeros((npts, nvm), dtype=np.float64),
        maxmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64),
        minmoiavail_lastyear=np.zeros((npts, nvm), dtype=np.float64),
        t2m_longterm=np.full(npts, 293.15, dtype=np.float64),
        t2m_week=np.full(npts, 293.15, dtype=np.float64),
        gdd_from_growthinit=np.zeros((npts, nvm), dtype=np.float64),
        nrec=np.zeros((npts, nvm), dtype=np.int32),
        senescence_type=np.zeros(nvm, dtype=np.int32),
        is_grassland_manag=np.zeros(nvm, dtype=bool),
        availability_fact=np.full(nvm, 0.14, dtype=np.float64),
        residence_time=np.full(nvm, 30.0, dtype=np.float64),
        leaf_tab=np.zeros(nvm, dtype=np.int32),
        pheno_type=np.zeros(nvm, dtype=np.int32),
        min_leaf_age_for_senescence=np.zeros(nvm, dtype=np.float64),
        gdd_senescence=np.full(nvm, 1.0e6, dtype=np.float64),
        senescence_temp=np.zeros((nvm, 3), dtype=np.float64),
        hum_frac=np.full(nvm, 0.2, dtype=np.float64),
        senescence_hum=np.zeros(nvm, dtype=np.float64),
        nosenescence_hum=np.ones(nvm, dtype=np.float64),
        max_turnover_time=np.full(nvm, 80.0, dtype=np.float64),
        min_turnover_time=np.full(nvm, 10.0, dtype=np.float64),
        leaffall=np.full(nvm, 10.0, dtype=np.float64),
        leafagecrit=np.full(nvm, 100.0, dtype=np.float64),
        lai_initmin=np.full(nvm, 0.3, dtype=np.float64),
        tau_fruit=np.full(nvm, 90.0, dtype=np.float64),
        tau_sap=np.full(nvm, 730.0, dtype=np.float64),
        sla_age1=np.ones((npts, nvm), dtype=np.float64),
        sla_max=np.full(nvm, 0.03, dtype=np.float64),
        sla_min=np.full(nvm, 0.01, dtype=np.float64),
        vcmax25=np.full(nvm, 50.0, dtype=np.float64),
        leaf_timecst=np.full(nvm, 100.0 / NLEAFAGES, dtype=np.float64),
        grm_n_limitation=False,
        lpj_gap_const_mort=False,
        ok_dgvm=True,
        t2m=np.full(npts, 293.15, dtype=np.float64),
        rprof=np.ones((npts, nvm), dtype=np.float64),
        coeff_maint_zero=np.zeros((nvm, NPARTS), dtype=np.float64),
        maint_resp_slope=np.zeros((nvm, 3), dtype=np.float64),
        dt_sechiba=21600.0,
        dt_stomate=86400.0,
        do_slow=True,
    )

    np.testing.assert_allclose(bundles.prescribe_inputs["biomass"], state.biomass)
    np.testing.assert_array_equal(bundles.prescribe_inputs["pft_present"], state.pft_present)
    np.testing.assert_allclose(bundles.prescribe_inputs["cn_ind"], cn_ind)
    np.testing.assert_allclose(bundles.alloc_inputs["lai"], lai)
    np.testing.assert_array_equal(bundles.alloc_inputs["senescence"], state.senescence)
    np.testing.assert_allclose(bundles.post_npp_inputs["turnover_longterm"], state.turnover_longterm)
    np.testing.assert_allclose(bundles.post_npp_inputs["bm_to_litter"], 0.0)
    np.testing.assert_allclose(bundles.maintenance_inputs["gpp_daily_current"], state.gpp_daily)
    np.testing.assert_allclose(bundles.maintenance_inputs["resp_maint_part_current"], state.resp_maint_part)
    assert bundles.alloc_inputs["dt_days"] == 0.25
    assert bundles.post_npp_inputs["dt_days"] == 0.25
    assert bundles.prescribe_inputs["ok_dgvm"] is True
    assert bundles.prescribe_inputs["lpj_gap_const_mort"] is False
    assert bundles.post_npp_inputs["ok_dgvm"] is True
    assert bundles.post_npp_inputs["lpj_gap_const_mort"] is False
    assert bundles.post_npp_inputs["wire_vmax"] is True
    assert bundles.post_npp_inputs["ok_nlim_vmax"] is False
    np.testing.assert_allclose(bundles.post_npp_inputs["vcmax25"], 50.0)
    np.testing.assert_allclose(bundles.post_npp_inputs["n_limfert"], 1.0)


def test_paper_case_stomate_static_input_kwargs_fill_source_backed_pft_run_parameters():
    static = paper_case_stomate_static_input_kwargs(CONFIG)
    kwargs = static.kwargs
    pft14 = 13

    assert static.parameters.nvm == 14
    assert kwargs["ext_coeff"].shape == (14,)
    assert kwargs["coeff_maint_zero"].shape == (14, NPARTS)
    assert kwargs["vcmax25"].shape == (14,)
    assert kwargs["leaf_timecst"].shape == (14,)
    assert kwargs["grm_n_limitation"] is False
    assert kwargs["maint_resp_slope"].shape == (14, 3)
    assert kwargs["pheno_model"][pft14] == "none"
    assert kwargs["pheno_is_none"][pft14]
    assert kwargs["ok_dgvm"] is False
    assert kwargs["lpj_gap_const_mort"] is True
    assert kwargs["senescence_type"][pft14] == SENESCENCE_NONE
    np.testing.assert_allclose(kwargs["r0"][pft14], 0.35)
    np.testing.assert_allclose(kwargs["s0"][pft14], 0.35)
    np.testing.assert_allclose(kwargs["alloc_min"][pft14], 0.2019211)
    np.testing.assert_allclose(kwargs["residence_time"][pft14], 50.6583452)
    np.testing.assert_allclose(kwargs["maint_resp_slope"][pft14], [0.0876862, 0.0, 0.0])
    np.testing.assert_allclose(kwargs["coeff_maint_zero"][pft14, ILEAF], 2.35e-3)
    np.testing.assert_allclose(kwargs["coeff_maint_zero"][pft14, IROOT], 1.67e-3)
    np.testing.assert_allclose(static.parameters.pheno_gdd_crit[pft14], [-9999.0, -9999.0, -9999.0])
    np.testing.assert_allclose(static.parameters.ncdgdd_temp[pft14], -9999.0)
    np.testing.assert_allclose(static.parameters.hum_min_time[pft14], -9999.0)
    assert any("pft_parameters.f90" in item for item in static.provenance)


def test_paper_case_season_time_scales_read_used_run_def_defaults():
    scales = paper_case_season_time_scales(CONFIG)

    assert scales.tau_hum_month == 20.0
    assert scales.tau_hum_week == 7.0
    assert scales.tau_gpp_week == 7.0
    assert scales.tau_gdd == 40.0
    assert scales.tau_ngd == 50.0
    assert scales.tau_climatology == 20.0
    assert scales.coeff_tau_longterm == 3.0


def test_stomate_restart_input_bundles_accept_paper_case_static_kwargs_without_static_fixtures():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    npts, nvm, nslm = state.sla_calc.shape[0], state.sla_calc.shape[1], 7
    z_soil = np.linspace(0.0, 7.0, nslm + 1, dtype=np.float64)
    static_kwargs = paper_case_stomate_static_input_kwargs(CONFIG).kwargs

    bundles = stomate_restart_input_bundles(
        state=state,
        veget_max=np.full((npts, nvm), 0.2, dtype=np.float64),
        lai=np.full((npts, nvm), 0.7, dtype=np.float64),
        cn_ind=np.full((npts, nvm), 3.0, dtype=np.float64),
        dt_days=0.25,
        bm_sapl=np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        maxdia=np.ones(nvm, dtype=np.float64),
        t2m_month=np.full(npts, 293.15, dtype=np.float64),
        t2m_min_daily=np.full(npts, 280.0, dtype=np.float64),
        tseason=np.full(npts, 12.0, dtype=np.float64),
        moiavail_week=np.full((npts, nvm), 0.8, dtype=np.float64),
        tsoil_month=np.full((npts, nslm), 293.15, dtype=np.float64),
        soilhum_month=np.full((npts, nslm), 0.8, dtype=np.float64),
        z_soil=z_soil,
        height=np.ones((npts, nvm), dtype=np.float64),
        tmin_spring_time=np.zeros((npts, nvm), dtype=np.float64),
        herbivores=np.zeros((npts, nvm), dtype=np.float64),
        maxmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64),
        minmoiavail_lastyear=np.zeros((npts, nvm), dtype=np.float64),
        t2m_longterm=np.full(npts, 293.15, dtype=np.float64),
        t2m_week=np.full(npts, 293.15, dtype=np.float64),
        gdd_from_growthinit=np.zeros((npts, nvm), dtype=np.float64),
        nrec=np.zeros((npts, nvm), dtype=np.int32),
        t2m=np.full(npts, 293.15, dtype=np.float64),
        rprof=np.ones((npts, nvm), dtype=np.float64),
        dt_sechiba=21600.0,
        dt_stomate=86400.0,
        do_slow=True,
        sla_age1=np.ones((npts, nvm), dtype=np.float64),
        **static_kwargs,
    )

    np.testing.assert_allclose(bundles.alloc_inputs["alloc_min"][0], static_kwargs["alloc_min"][0])
    np.testing.assert_allclose(bundles.post_npp_inputs["residence_time"][13], 50.6583452)
    np.testing.assert_allclose(bundles.maintenance_inputs["coeff_maint_zero"][13, ILEAF], 2.35e-3)
    assert bundles.phenology_inputs["pheno_model"][13] == "none"


def test_stomate_restart_season_input_kwargs_map_restart_memory_without_fills():
    files = find_stomate_reference_files(ROOT)
    season = read_stomate_restart_season_state(files.restart)
    mapped = stomate_restart_season_input_kwargs(season)
    kwargs = mapped.kwargs

    assert season.moiavail_week.shape == (1, 14)
    assert season.tsoil_month.shape[0] == 1
    assert season.soilhum_month.shape == season.tsoil_month.shape
    assert season.tmin_spring_time.shape == (1, 14)
    assert season.begin_leaves.shape == (1, 14)
    for name in (
        "t2m_month",
        "tseason",
        "moiavail_week",
        "tsoil_month",
        "soilhum_month",
        "tmin_spring_time",
        "t2m_longterm",
        "t2m_week",
        "gdd_from_growthinit",
    ):
        assert name in kwargs
    assert "maxmoiavail_lastyear" not in kwargs
    assert any("stomate_io.f90::readstart" in item for item in mapped.provenance)


def test_stomate_season_memory_input_kwargs_advance_restart_memory_from_daily_state_without_extra_fills():
    files = find_stomate_reference_files(ROOT)
    entry_state = read_stomate_restart_entry_state(files.restart)
    season = read_stomate_restart_season_state(files.restart)
    daily = read_stomate_daily_accumulator_state(files.start)
    params = paper_case_stomate_static_input_kwargs(CONFIG).parameters
    scales = paper_case_season_time_scales(CONFIG)
    restart_state = stomate_restart_season_memory_state(season, tau_longterm=1.0)

    mapped = stomate_season_memory_input_kwargs(
        season=season,
        daily=daily,
        entry_state=entry_state,
        parameters=params,
        dt_days=1.0,
        tau_longterm=1.0,
        julian_diff=1.0,
        end_of_year=False,
        time_scales=scales,
    )
    kwargs = mapped.kwargs

    expected_week = (season.moiavail_week * (scales.tau_hum_week - 1.0) + daily.humrel_daily) / scales.tau_hum_week
    expected_t2m_week = (season.t2m_week * (scales.tau_t2m_week - 1.0) + daily.t2m_daily) / scales.tau_t2m_week
    np.testing.assert_allclose(kwargs["moiavail_week"], expected_week)
    np.testing.assert_allclose(kwargs["t2m_week"], expected_t2m_week)
    np.testing.assert_allclose(kwargs["tseason"], restart_state.tseason)
    assert mapped.step.state.tau_longterm == 2.0
    assert kwargs["gdd_from_growthinit"].shape == season.gdd_from_growthinit.shape
    assert "maxmoiavail_lastyear" not in kwargs
    assert any("stomate_season.f90::season lines 577-772" in item for item in mapped.provenance)


def test_stomate_season_annual_input_kwargs_advance_longterm_state_from_restart_and_daily_values():
    files = find_stomate_reference_files(ROOT)
    entry_state = read_stomate_restart_entry_state(files.restart)
    season = read_stomate_restart_season_state(files.restart)
    daily = read_stomate_daily_accumulator_state(files.start)
    params = paper_case_stomate_static_input_kwargs(CONFIG).parameters
    scales = paper_case_season_time_scales(CONFIG)
    veget = np.full(entry_state.npp_daily.shape, 0.2, dtype=np.float64)
    veget[:, 0] = 0.0
    veget_max = veget.copy()

    mapped = stomate_season_annual_input_kwargs(
        season=season,
        daily=daily,
        entry_state=entry_state,
        parameters=params,
        veget=veget,
        veget_max=veget_max,
        dt_days=1.0,
        tau_longterm=2.0,
        end_of_year=False,
        time_scales=scales,
    )
    kwargs = mapped.kwargs

    tau_longterm = 2.0
    expected_npp = (entry_state.npp_longterm * (tau_longterm - 1.0) + entry_state.npp_daily * 365.0) / tau_longterm
    expected_turnover = (
        entry_state.turnover_longterm * (tau_longterm - 1.0) + entry_state.turnover_daily * 365.0
    ) / tau_longterm
    expected_gpp_week = np.where(
        veget_max > 0.0,
        (season.gpp_week * (scales.tau_gpp_week - 1.0) + daily.gpp_daily) / scales.tau_gpp_week,
        0.0,
    )
    np.testing.assert_allclose(kwargs["npp_longterm"], expected_npp)
    np.testing.assert_allclose(kwargs["turnover_longterm"], expected_turnover)
    np.testing.assert_allclose(kwargs["gpp_week"], expected_gpp_week)
    np.testing.assert_allclose(kwargs["lm_lastyearmax"], entry_state.lm_lastyearmax)
    np.testing.assert_allclose(kwargs["herbivores"], mapped.step.herbivores)
    assert kwargs["herbivores"][0, 13] > 0.0
    assert "maxmoiavail_lastyear" not in kwargs
    assert "minmoiavail_lastyear" not in kwargs
    assert any("stomate_season.f90::season sections 12-21" in item for item in mapped.provenance)


def test_stomate_season_biometeorology_input_kwargs_preserve_pft14_undef_parameter_path():
    files = find_stomate_reference_files(ROOT)
    entry_state = read_stomate_restart_entry_state(files.restart)
    season = read_stomate_restart_season_state(files.restart)
    daily = read_stomate_daily_accumulator_state(files.start)
    params = paper_case_stomate_static_input_kwargs(CONFIG).parameters
    pft14 = 13

    mapped = stomate_season_biometeorology_input_kwargs(
        season=season,
        daily=daily,
        entry_state=entry_state,
        parameters=params,
        t2m_month=season.t2m_month,
        t2m_week=season.t2m_week,
        t2m_longterm=season.t2m_longterm,
        moiavail_month=season.moiavail_month,
        dt_days=1.0,
        julian_diff=1.0,
        time_scales=paper_case_season_time_scales(CONFIG),
    )
    kwargs = mapped.kwargs

    np.testing.assert_allclose(kwargs["gdd_m5_dormance"][:, pft14], season.gdd_m5_dormance[:, pft14])
    np.testing.assert_allclose(kwargs["gdd_midwinter"][:, pft14], season.gdd_midwinter[:, pft14])
    np.testing.assert_allclose(kwargs["ncd_dormance"][:, pft14], season.ncd_dormance[:, pft14])
    np.testing.assert_allclose(kwargs["time_hum_min"][:, pft14], season.time_hum_min[:, pft14])
    np.testing.assert_allclose(kwargs["hum_min_dormance"][:, pft14], season.hum_min_dormance[:, pft14])
    assert kwargs["ngd_minus5"].shape == season.ngd_minus5.shape
    assert any("stomate_season.f90::season sections 7-11" in item for item in mapped.provenance)


def test_stomate_restart_input_bundles_accept_static_and_restart_season_kwargs_together():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    season = read_stomate_restart_season_state(files.restart)
    npts, nvm, nslm = state.sla_calc.shape[0], state.sla_calc.shape[1], 7
    z_soil = np.linspace(0.0, 7.0, nslm + 1, dtype=np.float64)
    source_kwargs = {
        **paper_case_stomate_static_input_kwargs(CONFIG).kwargs,
        **stomate_restart_season_input_kwargs(season).kwargs,
    }

    bundles = stomate_restart_input_bundles(
        state=state,
        veget_max=np.full((npts, nvm), 0.2, dtype=np.float64),
        lai=np.full((npts, nvm), 0.7, dtype=np.float64),
        cn_ind=np.full((npts, nvm), 3.0, dtype=np.float64),
        dt_days=0.25,
        bm_sapl=np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        maxdia=np.ones(nvm, dtype=np.float64),
        t2m_min_daily=np.full(npts, 280.0, dtype=np.float64),
        z_soil=z_soil,
        height=np.ones((npts, nvm), dtype=np.float64),
        herbivores=np.zeros((npts, nvm), dtype=np.float64),
        maxmoiavail_lastyear=np.ones((npts, nvm), dtype=np.float64),
        minmoiavail_lastyear=np.zeros((npts, nvm), dtype=np.float64),
        nrec=np.zeros((npts, nvm), dtype=np.int32),
        t2m=np.full(npts, 293.15, dtype=np.float64),
        rprof=np.ones((npts, nvm), dtype=np.float64),
        dt_sechiba=21600.0,
        dt_stomate=86400.0,
        do_slow=True,
        sla_age1=np.ones((npts, nvm), dtype=np.float64),
        **source_kwargs,
    )

    np.testing.assert_allclose(bundles.alloc_inputs["moiavail_week"], season.moiavail_week)
    np.testing.assert_allclose(bundles.alloc_inputs["tsoil_month"], season.tsoil_month)
    np.testing.assert_allclose(bundles.post_npp_inputs["t2m_month"], season.t2m_month)
    np.testing.assert_allclose(bundles.post_npp_inputs["gdd_from_growthinit"], season.gdd_from_growthinit)
    np.testing.assert_allclose(bundles.maintenance_inputs["t2m_longterm"], season.t2m_longterm)


def test_paper_case_stomate_pre_step_input_kwargs_feed_bundles_with_stepped_season_values():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    season = read_stomate_restart_season_state(files.restart)
    daily = read_stomate_daily_accumulator_state(files.start)
    npts, nvm, nslm = state.sla_calc.shape[0], state.sla_calc.shape[1], 7
    z_soil = np.linspace(0.0, 7.0, nslm + 1, dtype=np.float64)
    veget_max = np.full((npts, nvm), 0.2, dtype=np.float64)
    veget_max[:, 0] = 0.0
    source = paper_case_stomate_pre_step_input_kwargs(
        config_path=CONFIG,
        season=season,
        daily=daily,
        entry_state=state,
        veget=veget_max,
        veget_max=veget_max,
        dt_days=1.0,
        tau_longterm=1.0,
        julian_diff=1.0,
    )

    bundles = stomate_restart_input_bundles(
        state=state,
        veget_max=veget_max,
        lai=np.full((npts, nvm), 0.7, dtype=np.float64),
        cn_ind=np.full((npts, nvm), 3.0, dtype=np.float64),
        dt_days=1.0,
        bm_sapl=np.zeros((nvm, NPARTS, 1), dtype=np.float64),
        maxdia=np.ones(nvm, dtype=np.float64),
        t2m_min_daily=np.full(npts, 280.0, dtype=np.float64),
        z_soil=z_soil,
        height=np.ones((npts, nvm), dtype=np.float64),
        maxmoiavail_lastyear=season.maxmoiavail_lastyear,
        minmoiavail_lastyear=season.minmoiavail_lastyear,
        nrec=np.zeros((npts, nvm), dtype=np.int32),
        t2m=daily.t2m_daily,
        rprof=np.ones((npts, nvm), dtype=np.float64),
        dt_sechiba=21600.0,
        dt_stomate=86400.0,
        do_slow=True,
        sla_age1=np.ones((npts, nvm), dtype=np.float64),
        **source.kwargs,
    )

    np.testing.assert_allclose(bundles.alloc_inputs["moiavail_week"], source.memory.step.state.moiavail_week)
    np.testing.assert_allclose(bundles.alloc_inputs["tsoil_month"], source.memory.step.state.tsoil_month)
    np.testing.assert_allclose(bundles.post_npp_inputs["npp_longterm"], source.annual.step.state.npp_longterm)
    np.testing.assert_allclose(bundles.post_npp_inputs["turnover_longterm"], source.annual.step.state.turnover_longterm)
    np.testing.assert_allclose(bundles.post_npp_inputs["gdd_from_growthinit"], source.memory.step.state.gdd_from_growthinit)
    np.testing.assert_allclose(bundles.post_npp_inputs["herbivores"], source.annual.step.herbivores)
    np.testing.assert_allclose(bundles.maintenance_inputs["t2m_longterm"], source.memory.step.state.t2m_longterm)


def test_paper_case_stomate_pre_step_firstcall_initializes_cold_start_season_memory():
    npts, nvm, nslm = 1, 14, 11
    t2m = np.asarray([300.0], dtype=np.float64)
    season = stomate_cold_start_season_state(t2m=t2m, dt_days=1.0, nvm=nvm, nslm=nslm)
    entry_state = stomate_cold_start_entry_state(t2m=t2m, nvm=nvm, nslm=nslm)
    daily = stomate_cold_start_daily_accumulator_state(t2m=t2m, nvm=nvm, nslm=nslm)._replace(
        humrel_daily=np.full((npts, nvm), 0.8, dtype=np.float64),
        t2m_daily=t2m,
        tsoil_daily=np.full((npts, nslm), 299.0, dtype=np.float64),
        soilhum_daily=np.full((npts, nslm), 0.6, dtype=np.float64),
    )
    veget_max = np.full((npts, nvm), 0.2, dtype=np.float64)
    veget_max[:, 0] = 0.0

    source = paper_case_stomate_pre_step_input_kwargs(
        config_path=CONFIG,
        season=season,
        daily=daily,
        entry_state=entry_state,
        veget=veget_max,
        veget_max=veget_max,
        dt_days=1.0,
        julian_diff=1.0,
        firstcall_season=True,
    )

    np.testing.assert_allclose(source.memory.step.state.moiavail_week, daily.humrel_daily)
    np.testing.assert_allclose(source.memory.step.state.soilhum_month, daily.soilhum_daily)
    assert any("stomate_season.f90::season lines 386-473" in item for item in source.memory.step.provenance)


def test_paper_case_stomate_bundle_source_kwargs_add_daily_slowproc_and_lastyear_sources():
    files = find_stomate_reference_files(ROOT)
    restart = reference_case_first_step_restart_state(CONFIG, root=ROOT, stomate_filename="stomate_restart.nc")
    state = read_stomate_restart_entry_state(files.restart)
    season = read_stomate_restart_season_state(files.restart)
    daily = read_stomate_daily_accumulator_state(files.start)
    veget_max = restart.slowproc.veget_max

    source = paper_case_stomate_bundle_source_kwargs(
        config_path=CONFIG,
        season=season,
        daily=daily,
        entry_state=state,
        slowproc_state=restart.slowproc,
        veget=restart.slowproc.veget,
        veget_max=veget_max,
        dt_days=1.0,
        tau_longterm=1.0,
        julian_diff=1.0,
    )

    np.testing.assert_allclose(source.kwargs["height"], restart.slowproc.height)
    np.testing.assert_allclose(source.kwargs["t2m_min_daily"], daily.t2m_min_daily)
    np.testing.assert_allclose(source.kwargs["t2m"], daily.t2m_daily)
    np.testing.assert_allclose(source.kwargs["maxmoiavail_lastyear"], season.maxmoiavail_lastyear)
    np.testing.assert_allclose(source.kwargs["minmoiavail_lastyear"], season.minmoiavail_lastyear)
    np.testing.assert_allclose(source.kwargs["z_soil"], source.boundary.kwargs["z_soil"])
    np.testing.assert_allclose(source.kwargs["rprof"], source.boundary.kwargs["rprof"])
    np.testing.assert_allclose(source.kwargs["bm_sapl"], source.data.kwargs["bm_sapl"])
    np.testing.assert_allclose(source.kwargs["bm_sapl_rescale"], source.data.kwargs["bm_sapl_rescale"])
    np.testing.assert_allclose(source.kwargs["maxdia"], source.data.kwargs["maxdia"])
    np.testing.assert_allclose(source.kwargs["sla_age1"], source.data.kwargs["sla_age1"])
    np.testing.assert_allclose(source.kwargs["herbivores"], source.pre_step.annual.step.herbivores)
    np.testing.assert_allclose(source.kwargs["nrec"], source.inactive_crop.kwargs["nrec"])
    assert source.kwargs["z_soil"].shape == (12,)
    assert source.kwargs["rprof"].shape == (state.biomass.shape[0], state.biomass.shape[1])
    assert source.kwargs["bm_sapl"].shape == (state.biomass.shape[1], NPARTS, 1)
    assert source.kwargs["sla_age1"].shape == (state.biomass.shape[0], state.biomass.shape[1])
    assert source.kwargs["nrec"].shape == (state.biomass.shape[0], state.biomass.shape[1])
    assert source.kwargs["rprof"][0, 13] == pytest.approx(1.25)


def test_paper_case_inactive_crop_input_kwargs_closes_nrec_only_when_ok_laidev_is_inactive():
    params = paper_case_stomate_static_input_kwargs(CONFIG).parameters
    mapped = paper_case_inactive_crop_input_kwargs(parameters=params, npts=2)

    assert not np.any(params.ok_laidev)
    np.testing.assert_array_equal(mapped.kwargs["nrec"], np.zeros((2, params.nvm), dtype=np.int32))
    assert any("turn lines 469-477" in item for item in mapped.provenance)


def test_paper_case_stomate_boundary_input_kwargs_build_z_soil_and_rprof_from_used_run_def():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    boundary = paper_case_stomate_boundary_input_kwargs(
        CONFIG,
        npts=state.biomass.shape[0],
    )

    np.testing.assert_allclose(boundary.kwargs["z_soil"][0], 0.0)
    assert boundary.kwargs["z_soil"].shape == (12,)
    assert boundary.kwargs["rprof"].shape == (state.biomass.shape[0], state.biomass.shape[1])
    np.testing.assert_allclose(boundary.humcste_use[0], boundary.humcste)
    assert boundary.humcste[13] == pytest.approx(0.8)
    assert boundary.kwargs["rprof"][0, 13] == pytest.approx(1.25)
    assert any("sechiba.f90::sechiba_main lines 984-990" in item for item in boundary.provenance)


def test_paper_case_stomate_data_input_kwargs_build_sapling_diameter_and_sla_age1_from_sources():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    data = paper_case_stomate_data_input_kwargs(
        CONFIG,
        npts=state.biomass.shape[0],
    )

    bm_sapl = data.kwargs["bm_sapl"]
    pft14 = 13
    sla = 1.53e-2
    expected_leaf = (((4.0 * 100.0 * (3.0 * 4.0 * sla / (np.pi * 8000.0)) ** 0.8) / sla) ** 5.0)
    csa_sap = expected_leaf / (8000.0 / sla)
    dia = (3.0 * csa_sap * 4.0 / np.pi) ** 0.5
    expected_sap_above = 0.5 * 200000.0 * csa_sap * 40.0 * dia**0.5
    expected_maxdia = (0.3 / ((40.0 * 0.5) / (100.0**0.5))) ** (1.0 / (0.5 - 1.0)) * 0.01
    np.testing.assert_allclose(bm_sapl[pft14, ILEAF, ICARBON], expected_leaf)
    np.testing.assert_allclose(bm_sapl[pft14, ISAPABOVE, ICARBON], expected_sap_above)
    np.testing.assert_allclose(bm_sapl[pft14, IROOT, ICARBON], 0.1 * expected_leaf)
    np.testing.assert_allclose(bm_sapl[pft14, IFRUIT, ICARBON], 0.3 * expected_leaf)
    np.testing.assert_allclose(bm_sapl[pft14, ICARBRES, ICARBON], 0.0)
    assert data.kwargs["bm_sapl_rescale"] == pytest.approx(40.0)
    assert data.kwargs["maxdia"][pft14] == pytest.approx(expected_maxdia)
    assert data.kwargs["sla_age1"].shape == (state.biomass.shape[0], state.biomass.shape[1])
    np.testing.assert_allclose(data.kwargs["sla_age1"][0, pft14], 1.53e-2)
    assert any("stomate_data.f90::data lines 261-346" in item for item in data.provenance)


def test_paper_case_stomate_bundle_source_kwargs_reduce_manual_bundle_inputs():
    files = find_stomate_reference_files(ROOT)
    restart = reference_case_first_step_restart_state(CONFIG, root=ROOT, stomate_filename="stomate_restart.nc")
    state = read_stomate_restart_entry_state(files.restart)
    dirty_work_state = state._replace(
        co2_to_bm=np.full_like(state.co2_to_bm, 17.0),
        bm_to_litter=np.full_like(state.bm_to_litter, 23.0),
    )
    season = read_stomate_restart_season_state(files.restart)
    daily = read_stomate_daily_accumulator_state(files.start)
    npts, nvm = state.sla_calc.shape[0], state.sla_calc.shape[1]
    source = paper_case_stomate_bundle_source_kwargs(
        config_path=CONFIG,
        season=season,
        daily=daily,
        entry_state=state,
        slowproc_state=restart.slowproc,
        veget=restart.slowproc.veget,
        veget_max=restart.slowproc.veget_max,
        dt_days=1.0,
        tau_longterm=1.0,
        julian_diff=1.0,
    )

    bundles = stomate_restart_input_bundles(
        state=dirty_work_state,
        veget_max=restart.slowproc.veget_max,
        lai=restart.slowproc.lai,
        cn_ind=np.full((npts, nvm), 3.0, dtype=np.float64),
        dt_days=1.0,
        dt_sechiba=21600.0,
        dt_stomate=86400.0,
        do_slow=True,
        **source.kwargs,
    )

    np.testing.assert_allclose(bundles.prescribe_inputs["co2_to_bm"], 0.0)
    np.testing.assert_allclose(bundles.post_npp_inputs["bm_to_litter"], 0.0)
    np.testing.assert_allclose(bundles.post_npp_inputs["height"], restart.slowproc.height)
    np.testing.assert_allclose(bundles.post_npp_inputs["t2m_min_daily"], daily.t2m_min_daily)
    np.testing.assert_allclose(bundles.post_npp_inputs["maxmoiavail_lastyear"], season.maxmoiavail_lastyear)
    np.testing.assert_allclose(bundles.maintenance_inputs["t2m"], daily.t2m_daily)
    np.testing.assert_allclose(bundles.daily_process_inputs["gpp_daily"], state.gpp_daily)
    np.testing.assert_allclose(bundles.daily_process_inputs["t2m_daily"], daily.t2m_daily)
    np.testing.assert_allclose(bundles.daily_process_inputs["t2m_min_daily"], daily.t2m_min_daily)
    np.testing.assert_allclose(bundles.daily_process_inputs["resp_maint_part"], state.resp_maint_part)
    np.testing.assert_allclose(bundles.alloc_inputs["z_soil"], source.boundary.kwargs["z_soil"])
    np.testing.assert_allclose(bundles.maintenance_inputs["rprof"], source.boundary.kwargs["rprof"])
    np.testing.assert_allclose(bundles.prescribe_inputs["bm_sapl"], source.data.kwargs["bm_sapl"])
    np.testing.assert_allclose(bundles.prescribe_inputs["bm_sapl_rescale"], source.data.kwargs["bm_sapl_rescale"])
    np.testing.assert_allclose(bundles.prescribe_inputs["maxdia"], source.data.kwargs["maxdia"])
    np.testing.assert_allclose(bundles.post_npp_inputs["sla_age1"], source.data.kwargs["sla_age1"])
    np.testing.assert_allclose(bundles.post_npp_inputs["herbivores"], source.pre_step.annual.step.herbivores)
    np.testing.assert_array_equal(bundles.post_npp_inputs["nrec"], source.inactive_crop.kwargs["nrec"])

    folded_daily_fields = {
        "gpp_daily": np.full_like(state.gpp_daily, 7.0),
        "t2m_daily": np.full_like(daily.t2m_daily, 288.0),
        "t2m_min_daily": np.full_like(daily.t2m_min_daily, 277.0),
        "snowfall_daily": np.full_like(daily.t2m_daily, 0.25),
        "snowmass_daily": np.full_like(daily.t2m_daily, 1.5),
        "tmc_topgrass_daily": np.full_like(daily.t2m_daily, 0.4),
        "resp_maint_part": np.full_like(state.resp_maint_part, 0.8),
        "resp_maint_radia": np.full_like(state.resp_maint_part, 0.05),
        "flood_root_radia": np.full_like(daily.t2m_daily, 0.02),
    }
    folded_bundles = stomate_restart_input_bundles(
        state=state,
        veget_max=restart.slowproc.veget_max,
        lai=restart.slowproc.lai,
        cn_ind=np.full((npts, nvm), 3.0, dtype=np.float64),
        dt_days=1.0,
        dt_sechiba=21600.0,
        dt_stomate=86400.0,
        do_slow=True,
        daily_fields=folded_daily_fields,
        **source.kwargs,
    )

    np.testing.assert_allclose(folded_bundles.maintenance_inputs["gpp_daily_current"], state.gpp_daily)
    np.testing.assert_allclose(folded_bundles.maintenance_inputs["t2m"], daily.t2m_daily)
    np.testing.assert_allclose(folded_bundles.constraints_inputs["t2m_min_daily"], folded_daily_fields["t2m_min_daily"])
    np.testing.assert_allclose(folded_bundles.post_npp_inputs["t2m_min_daily"], folded_daily_fields["t2m_min_daily"])
    np.testing.assert_allclose(folded_bundles.maintenance_inputs["resp_maint_part_current"], state.resp_maint_part)
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["gpp_daily"], folded_daily_fields["gpp_daily"])
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["t2m_daily"], folded_daily_fields["t2m_daily"])
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["resp_maint_part"], folded_daily_fields["resp_maint_part"])
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["resp_maint_radia"], folded_daily_fields["resp_maint_radia"])
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["flood_root_radia"], folded_daily_fields["flood_root_radia"])
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["snowfall_daily"], folded_daily_fields["snowfall_daily"])
    assert "snowfall_daily" not in folded_bundles.post_npp_inputs


def test_stomate_restart_ok_leak_state_inputs_pass_restart_state_without_fills():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    z_soil = np.arange(1.0, 33.0, dtype=np.float64)
    zf_soil = np.arange(0.0, 33.0, dtype=np.float64)
    veget_max = np.full(state.sla_calc.shape, 0.2, dtype=np.float64)

    mapped = stomate_restart_ok_leak_state_inputs(
        state=state,
        veget_max=veget_max,
        z_soil=z_soil,
        zf_soil=zf_soil,
        contfrac=np.asarray([0.75], dtype=np.float64),
        one_day=86400.0,
    )

    assert set(mapped.ok_leak_state_inputs) == {
        "litter_above",
        "litter_below",
        "lignin_struc_above",
        "lignin_struc_below",
        "litterpart",
        "dead_leaves",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "carbon_32l",
        "doc",
        "interception_storage",
    }
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["litter_above"], state.litter_above)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["litter_below"], state.litter_below)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["litterpart"], state.litterpart)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["dead_leaves"], state.dead_leaves)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["fuel_1hr"], state.fuel_1hr)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["fuel_10hr"], state.fuel_10hr)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["fuel_100hr"], state.fuel_100hr)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["fuel_1000hr"], state.fuel_1000hr)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["carbon_32l"], state.carbon_32l)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["doc"], state.DOC)
    np.testing.assert_allclose(mapped.output_inputs["veget_max"], veget_max)
    np.testing.assert_allclose(mapped.output_inputs["z_soil"], z_soil)
    np.testing.assert_allclose(mapped.output_inputs["zf_soil"], zf_soil)
    np.testing.assert_allclose(mapped.output_inputs["carb_mass_total_old"], state.carb_mass_total)
    np.testing.assert_allclose(mapped.output_inputs["prod10_total"], state.prod10_total)
    np.testing.assert_allclose(mapped.output_inputs["prod100_total"], state.prod100_total)
    np.testing.assert_allclose(mapped.ok_leak_state_inputs["interception_storage"], state.interception_storage)
    np.testing.assert_allclose(mapped.output_inputs["contfrac"], [0.75])
    assert mapped.output_inputs["one_day"] == 86400.0

    overridden = stomate_restart_ok_leak_state_inputs(
        state=state,
        veget_max=veget_max,
        z_soil=z_soil,
        zf_soil=zf_soil,
        carb_mass_total_old=np.asarray([12.0], dtype=np.float64),
        prod10_total=np.asarray([1.0], dtype=np.float64),
        prod100_total=np.asarray([2.0], dtype=np.float64),
    )
    np.testing.assert_allclose(overridden.output_inputs["carb_mass_total_old"], [12.0])
    np.testing.assert_allclose(overridden.output_inputs["prod10_total"], [1.0])
    np.testing.assert_allclose(overridden.output_inputs["prod100_total"], [2.0])


def test_stomate_static_routing_ok_leak_inputs_maps_driver_and_no_routing_without_fills():
    all_enerbil_inputs = _enerbil_inputs()
    driver = _MiniDriverPayload()
    driver.kjpindex = 1
    driver.nbindex = 1
    driver.kindex = np.asarray([1], dtype=np.int32)
    driver.lalo = np.asarray([[21.0, 109.0]], dtype=np.float64)
    driver.contfrac = np.asarray([1.0], dtype=np.float64)
    driver.resolution = np.asarray([[111106.0, 111111.0]], dtype=np.float64)
    driver.neighbours = np.asarray([[-1, -1, -1, -1, -1, -1, -1, -1]], dtype=np.int32)
    driver.u = all_enerbil_inputs["u"]
    driver.v = all_enerbil_inputs["v"]
    driver.precip_rain = all_enerbil_inputs["precip_rain"]
    driver.precip_snow = np.asarray([0.0], dtype=np.float64)
    driver.swdown = np.asarray([350.0], dtype=np.float64)
    driver.pb = all_enerbil_inputs["pb"]
    driver.clay_frac = np.asarray([0.1], dtype=np.float64)
    driver.bulk_dens = np.asarray([1.3], dtype=np.float64)
    driver.soil_ph = np.asarray([6.0], dtype=np.float64)
    driver.poor_soils = np.asarray([0.0], dtype=np.float64)

    mapped = stomate_static_routing_ok_leak_inputs(
        driver_payload=driver,
        kjit=1,
        nflow=4,
        river_routing=False,
        nbp_glo=1,
    )

    assert set(mapped.ok_leak_inputs) == {
        "clay",
        "bulk_dens",
        "poor_soils",
        "doc_to_topsoil",
        "doc_to_subsoil",
        "flood_frac",
        "fastr",
    }
    np.testing.assert_allclose(mapped.ok_leak_inputs["clay"], driver.clay_frac)
    np.testing.assert_allclose(mapped.ok_leak_inputs["bulk_dens"], driver.bulk_dens)
    np.testing.assert_allclose(mapped.ok_leak_inputs["poor_soils"], driver.poor_soils)
    np.testing.assert_allclose(mapped.ok_leak_inputs["doc_to_topsoil"], np.zeros((1, 4), dtype=np.float64))
    np.testing.assert_allclose(mapped.ok_leak_inputs["doc_to_subsoil"], np.zeros((1, 4), dtype=np.float64))
    np.testing.assert_allclose(mapped.ok_leak_inputs["flood_frac"], np.zeros(1, dtype=np.float64))
    np.testing.assert_allclose(mapped.ok_leak_inputs["fastr"], np.zeros(1, dtype=np.float64))
    np.testing.assert_allclose(mapped.output_inputs["contfrac"], driver.contfrac)
    assert "DOC_to_topsoil" in mapped.entry_source
    assert "clay" in mapped.entry_source


def test_stomate_static_routing_single_point_with_routing_switch_still_takes_no_routing_branch():
    all_enerbil_inputs = _enerbil_inputs()
    driver = _MiniDriverPayload()
    driver.kjpindex = 1
    driver.nbindex = 1
    driver.kindex = np.asarray([1], dtype=np.int32)
    driver.lalo = np.asarray([[21.0, 109.0]], dtype=np.float64)
    driver.contfrac = np.asarray([1.0], dtype=np.float64)
    driver.resolution = np.asarray([[111106.0, 111111.0]], dtype=np.float64)
    driver.neighbours = np.asarray([[-1, -1, -1, -1, -1, -1, -1, -1]], dtype=np.int32)
    driver.u = all_enerbil_inputs["u"]
    driver.v = all_enerbil_inputs["v"]
    driver.precip_rain = all_enerbil_inputs["precip_rain"]
    driver.precip_snow = np.asarray([0.0], dtype=np.float64)
    driver.swdown = np.asarray([350.0], dtype=np.float64)
    driver.pb = all_enerbil_inputs["pb"]
    driver.clay_frac = np.asarray([0.1], dtype=np.float64)
    driver.bulk_dens = np.asarray([1.3], dtype=np.float64)
    driver.soil_ph = np.asarray([6.0], dtype=np.float64)
    driver.poor_soils = np.asarray([0.0], dtype=np.float64)

    mapped = stomate_static_routing_ok_leak_inputs(
        driver_payload=driver,
        kjit=1,
        nflow=4,
        river_routing=True,
        nbp_glo=1,
    )

    np.testing.assert_allclose(mapped.ok_leak_inputs["doc_to_topsoil"], np.zeros((1, 4), dtype=np.float64))
    np.testing.assert_allclose(mapped.ok_leak_inputs["doc_to_subsoil"], np.zeros((1, 4), dtype=np.float64))
    np.testing.assert_allclose(mapped.ok_leak_inputs["flood_frac"], np.zeros(1, dtype=np.float64))
    np.testing.assert_allclose(mapped.ok_leak_inputs["fastr"], np.zeros(1, dtype=np.float64))


def test_stomate_vertical_ok_leak_inputs_maps_diaglev_and_deep_grid_without_fills():
    diaglev = np.asarray([0.1, 0.4, 1.0], dtype=np.float64)
    zz_coef_deep = np.asarray([0.05, 0.2, 0.6, 1.2], dtype=np.float64)
    zz_deep = np.asarray([0.025, 0.1, 0.3, 0.9], dtype=np.float64)

    mapped = stomate_vertical_ok_leak_inputs(
        diaglev=diaglev,
        zz_coef_deep=zz_coef_deep,
        zz_deep=zz_deep,
    )

    np.testing.assert_allclose(mapped.ok_leak_inputs["z_soil"], [0.0, 0.1, 0.4, 1.0])
    np.testing.assert_allclose(mapped.ok_leak_inputs["zf_soil_b"], [0.0, 0.05, 0.2, 0.6, 1.2])
    np.testing.assert_allclose(mapped.ok_leak_inputs["zi_soil"], zz_deep)
    assert mapped.ok_leak_inputs["nslm"] == 3
    assert mapped.ok_leak_inputs["ndeep"] == 4
    np.testing.assert_allclose(mapped.output_inputs["z_soil"], diaglev)
    np.testing.assert_allclose(mapped.output_inputs["zf_soil"], [0.0, 0.05, 0.2, 0.6, 1.2])


def test_stomate_vertical_ok_leak_inputs_rejects_missing_vertical_source():
    with np.testing.assert_raises_regex(ValueError, "diaglev"):
        stomate_vertical_ok_leak_inputs(diaglev=np.zeros((1, 2)), zz_coef_deep=np.ones(2))
    with np.testing.assert_raises_regex(ValueError, "zz_coef_deep"):
        stomate_vertical_ok_leak_inputs(diaglev=np.ones(2), zz_coef_deep=np.zeros((1, 2)))
    with np.testing.assert_raises_regex(ValueError, "zz_deep"):
        stomate_vertical_ok_leak_inputs(diaglev=np.ones(2), zz_coef_deep=np.ones(3), zz_deep=np.ones(2))


def test_stomate_active_layer_ok_leak_inputs_initializes_and_updates_fortran_mask():
    tprof = np.asarray(
        [
            [
                [274.0, 274.0, 274.0],
                [274.0, 270.0, 270.0],
                [270.0, 270.0, 270.0],
            ]
        ],
        dtype=np.float64,
    )
    zi_soil = np.asarray([0.1, 0.3, 0.7], dtype=np.float64)
    restart_altmax = np.asarray([[0.0, 0.2, 0.5]], dtype=np.float64)
    fixed = np.asarray([[0.0, 0.4, 0.6]], dtype=np.float64)

    mapped = stomate_active_layer_ok_leak_inputs(
        tprof=tprof,
        zi_soil=zi_soil,
        restart_altmax=restart_altmax,
        restart_fixed_cryoturbation_depth=fixed,
        dayno=10,
        firstcall_soilcarbon=True,
    )

    np.testing.assert_array_equal(mapped.ok_leak_inputs["veget_mask"], np.ones((1, 3), dtype=bool))
    np.testing.assert_array_equal(np.asarray(mapped.state_updates["alt_ind"]), np.asarray([[2, 1, 1]]))
    np.testing.assert_allclose(np.asarray(mapped.state_updates["alt"]), np.asarray([[0.3, 0.1, 0.1]]))
    np.testing.assert_array_equal(np.asarray(mapped.ok_leak_inputs["altmax_ind"]), np.asarray([[2, 1, 2]]))
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["altmax_lastyear"]), restart_altmax)
    np.testing.assert_allclose(mapped.ok_leak_inputs["fixed_cryoturbation_depth"], fixed)
    np.testing.assert_allclose(mapped.ok_leak_inputs["zi_soil"], zi_soil)
    np.testing.assert_allclose(np.asarray(mapped.output_inputs["altmax"]), np.asarray([[0.3, 0.2, 0.5]]))


def test_stomate_active_layer_ok_leak_inputs_rejects_bad_shapes():
    with np.testing.assert_raises_regex(ValueError, "tprof"):
        stomate_active_layer_ok_leak_inputs(
            tprof=np.ones((2, 3), dtype=np.float64),
            zi_soil=np.ones(3, dtype=np.float64),
            restart_altmax=np.ones((1, 3), dtype=np.float64),
            restart_fixed_cryoturbation_depth=np.ones((1, 3), dtype=np.float64),
            dayno=1,
            firstcall_soilcarbon=True,
        )
    with np.testing.assert_raises_regex(ValueError, "zi_soil"):
        stomate_active_layer_ok_leak_inputs(
            tprof=np.ones((1, 3, 3), dtype=np.float64),
            zi_soil=np.ones(2, dtype=np.float64),
            restart_altmax=np.ones((1, 3), dtype=np.float64),
            restart_fixed_cryoturbation_depth=np.ones((1, 3), dtype=np.float64),
            dayno=1,
            firstcall_soilcarbon=True,
        )


def test_stomate_litter_controls_ok_leak_inputs_computes_fortran_controls_from_boundaries():
    npts, nvm, nslm, nstm = 1, 3, 4, 4
    soil_mc = np.zeros((npts, nslm, nstm), dtype=np.float64)
    soil_mc[0, :, 3] = [0.1, 0.2, 0.3, 0.4]
    soil_mc[0, :, 0] = [0.9, 0.9, 0.9, 0.9]
    z_soil = np.asarray([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64)
    pref_soil_veg = np.asarray([1, 1, 4], dtype=np.int32)

    mapped = stomate_litter_controls_ok_leak_inputs(
        tsurf=np.asarray([303.15], dtype=np.float64),
        soil_mc=soil_mc,
        z_soil=z_soil,
        pref_soil_veg=pref_soil_veg,
        frozen_respiration_func=1,
    )

    assert set(mapped.ok_leak_inputs) == {
        "control_temp_above",
        "control_moist_above",
        "soil_mc_top_by_pft",
    }
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["control_temp_above"]), np.ones((1, NLITT)))
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["soil_mc_top_by_pft"])[0, 2], 0.1)
    assert np.asarray(mapped.ok_leak_inputs["control_moist_above"])[0, 2] > 0.0


def test_stomate_litter_controls_ok_leak_inputs_rejects_bad_pref_and_missing_moyano_sources():
    soil_mc = np.ones((1, 4, 4), dtype=np.float64)
    z_soil = np.asarray([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "one-based"):
        stomate_litter_controls_ok_leak_inputs(
            tsurf=np.asarray([303.15], dtype=np.float64),
            soil_mc=soil_mc,
            z_soil=z_soil,
            pref_soil_veg=np.asarray([0, 1, 4], dtype=np.int32),
            frozen_respiration_func=1,
        )
    with np.testing.assert_raises_regex(ValueError, "MOIST_FUNC_MOYANO requires"):
        stomate_litter_controls_ok_leak_inputs(
            tsurf=np.asarray([303.15], dtype=np.float64),
            soil_mc=soil_mc,
            z_soil=z_soil,
            pref_soil_veg=np.asarray([1, 1, 4], dtype=np.int32),
            frozen_respiration_func=1,
            moist_func_moyano=True,
        )


def test_stomate_tf_doc_ok_leak_inputs_computes_deposition_from_source_boundaries():
    npts, nvm = 1, 4
    precip2ground = np.full((npts, nvm), 2.0, dtype=np.float64)
    precip2canopy = np.full((npts, nvm), 3.0, dtype=np.float64)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[0, 3] = 0.5
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, 3, ILEAF, ICARBON] = 100.0
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[3] = True
    interception_storage = np.zeros((npts, nvm, 1), dtype=np.float64)
    interception_storage[0, 3, ICARBON] = 0.25

    mapped = stomate_tf_doc_ok_leak_inputs(
        precip2ground=precip2ground,
        precip2canopy=precip2canopy,
        biomass=biomass,
        veget_max=veget_max,
        is_tree=is_tree,
        interception_storage=interception_storage,
        ok_tf_doc=True,
        dt_days=0.5,
    )

    assert set(mapped.ok_leak_inputs) == {
        "doc_precip2ground",
        "doc_precip2canopy",
        "dry_dep_canopy",
        "interception_storage",
    }
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["doc_precip2ground"])[0, 3, ICARBON], 2.0 * 3.02e-3)
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["doc_precip2canopy"])[0, 3, ICARBON], 3.0 * 3.02e-3)
    np.testing.assert_allclose(
        np.asarray(mapped.ok_leak_inputs["dry_dep_canopy"])[0, 3, ICARBON],
        0.00092 * 0.5 * 100.0 * 0.5,
    )
    np.testing.assert_allclose(mapped.ok_leak_inputs["interception_storage"], interception_storage)
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["doc_precip2ground"])[0, 0, ICARBON], 0.0)


def test_stomate_tf_doc_ok_leak_inputs_preserves_storage_when_branch_disabled():
    npts, nvm = 1, 3
    storage = np.full((npts, nvm, 1), 0.4, dtype=np.float64)

    mapped = stomate_tf_doc_ok_leak_inputs(
        precip2ground=np.ones((npts, nvm), dtype=np.float64),
        precip2canopy=np.ones((npts, nvm), dtype=np.float64),
        biomass=np.ones((npts, nvm, NPARTS, 1), dtype=np.float64),
        veget_max=np.ones((npts, nvm), dtype=np.float64),
        is_tree=np.ones(nvm, dtype=bool),
        interception_storage=storage,
        ok_tf_doc=False,
        dt_days=1.0,
    )

    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["doc_precip2ground"]), 0.0)
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["doc_precip2canopy"]), 0.0)
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["dry_dep_canopy"]), 0.0)
    np.testing.assert_allclose(mapped.ok_leak_inputs["interception_storage"], storage)


def test_stomate_tf_doc_ok_leak_inputs_rejects_storage_shape_mismatch():
    with np.testing.assert_raises_regex(ValueError, "interception_storage"):
        stomate_tf_doc_ok_leak_inputs(
            precip2ground=np.ones((1, 3), dtype=np.float64),
            precip2canopy=np.ones((1, 3), dtype=np.float64),
            biomass=np.ones((1, 3, NPARTS, 1), dtype=np.float64),
            veget_max=np.ones((1, 3), dtype=np.float64),
            is_tree=np.ones(3, dtype=bool),
            interception_storage=np.ones((1, 3), dtype=np.float64),
            ok_tf_doc=True,
            dt_days=1.0,
        )


def test_stomate_perma_peat_ok_leak_inputs_computes_cmax_and_fortran_mask():
    zf_soil_b = np.asarray([0.0, 0.5, 1.5, 3.0], dtype=np.float64)
    peat_bulk_density = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)

    mapped = stomate_perma_peat_ok_leak_inputs(
        peat_bulk_density=peat_bulk_density,
        zf_soil_b=zf_soil_b,
        npts=2,
        nvm=4,
        perma_peat=True,
        frac1=0.70,
        frac2=0.10,
    )

    expected_soc = (1.0 / ((0.4 * peat_bulk_density + 0.13) ** 2.19)) * 0.01
    expected_cmax = peat_bulk_density * 1.0e6 * expected_soc * np.diff(zf_soil_b)
    np.testing.assert_allclose(np.asarray(mapped.ok_leak_inputs["cmax_peat"]), expected_cmax)
    np.testing.assert_array_equal(mapped.ok_leak_inputs["perma_peat_veget_mask"], np.ones((2, 4), dtype=bool))
    assert mapped.ok_leak_inputs["perma_peat"] is True
    assert mapped.ok_leak_inputs["frac1"] == 0.70
    assert mapped.ok_leak_inputs["frac2"] == 0.10


def test_stomate_perma_peat_ok_leak_inputs_rejects_bad_shape():
    with np.testing.assert_raises_regex(ValueError, "1D"):
        stomate_perma_peat_ok_leak_inputs(
            peat_bulk_density=np.ones((1, 3), dtype=np.float64),
            zf_soil_b=np.ones(4, dtype=np.float64),
            npts=1,
            nvm=3,
            perma_peat=True,
        )
    with np.testing.assert_raises_regex(ValueError, "zf_soil_b"):
        stomate_perma_peat_ok_leak_inputs(
            peat_bulk_density=np.ones(3, dtype=np.float64),
            zf_soil_b=np.ones(3, dtype=np.float64),
            npts=1,
            nvm=3,
            perma_peat=True,
        )


def test_stomate_ok_leak_boundary_inputs_merges_source_bundles_without_duplicates():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    npts, nvm = state.sla_calc.shape
    restart = stomate_restart_ok_leak_state_inputs(
        state=state,
        veget_max=np.full((npts, nvm), 0.2, dtype=np.float64),
    )
    driver = _MiniDriverPayload()
    driver.kjpindex = npts
    driver.nbindex = npts
    driver.kindex = np.arange(1, npts + 1, dtype=np.int32)
    driver.lalo = np.zeros((npts, 2), dtype=np.float64)
    driver.contfrac = np.full(npts, 0.5625, dtype=np.float64)
    driver.resolution = np.ones((npts, 2), dtype=np.float64)
    driver.neighbours = np.full((npts, 8), -1, dtype=np.int32)
    driver.u = np.ones(npts, dtype=np.float64)
    driver.v = np.zeros(npts, dtype=np.float64)
    driver.precip_rain = np.zeros(npts, dtype=np.float64)
    driver.precip_snow = np.zeros(npts, dtype=np.float64)
    driver.swdown = np.ones(npts, dtype=np.float64)
    driver.pb = np.ones(npts, dtype=np.float64)
    driver.clay_frac = np.full(npts, 0.1, dtype=np.float64)
    driver.bulk_dens = np.full(npts, 1.3, dtype=np.float64)
    driver.soil_ph = np.full(npts, 6.0, dtype=np.float64)
    driver.poor_soils = np.zeros(npts, dtype=np.float64)
    static_routing = stomate_static_routing_ok_leak_inputs(
        driver_payload=driver,
        kjit=1,
        nflow=4,
        river_routing=False,
        nbp_glo=npts,
    )
    vertical = stomate_vertical_ok_leak_inputs(
        diaglev=np.asarray([0.1, 0.4], dtype=np.float64),
        zz_coef_deep=np.asarray([0.1, 0.4], dtype=np.float64),
    )
    litter_controls = stomate_litter_controls_ok_leak_inputs(
        tsurf=np.asarray([303.15], dtype=np.float64),
        soil_mc=np.ones((npts, 4, 4), dtype=np.float64),
        z_soil=np.asarray([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64),
        pref_soil_veg=np.asarray([1] * nvm, dtype=np.int32),
        frozen_respiration_func=1,
    )
    tf_doc = stomate_tf_doc_ok_leak_inputs(
        precip2ground=np.ones((npts, nvm), dtype=np.float64),
        precip2canopy=np.ones((npts, nvm), dtype=np.float64),
        biomass=state.biomass,
        veget_max=np.full((npts, nvm), 0.2, dtype=np.float64),
        is_tree=np.ones(nvm, dtype=bool),
        interception_storage=state.interception_storage,
        ok_tf_doc=True,
        dt_days=1.0,
    )
    perma_peat = stomate_perma_peat_ok_leak_inputs(
        peat_bulk_density=np.linspace(0.1, 0.2, 2, dtype=np.float64),
        zf_soil_b=np.asarray([0.0, 0.1, 0.4], dtype=np.float64),
        npts=npts,
        nvm=nvm,
        perma_peat=True,
        frac1=0.70,
        frac2=0.10,
    )
    soilwater = stomate_soilwater_31mm_ok_leak_inputs(
        soil_mc=np.ones((npts, 2, 4), dtype=np.float64),
        z_soil=np.asarray([0.0, 0.1, 0.4], dtype=np.float64),
        sro_bottom=2,
    )
    doc_transport = stomate_doc_transport_ok_leak_inputs(
        poor_soils=driver.poor_soils,
    )

    merged = stomate_ok_leak_boundary_inputs(
        restart_state=restart,
        static_routing=static_routing,
        vertical=vertical,
        litter_controls=litter_controls,
        tf_doc=tf_doc,
        perma_peat=perma_peat,
        soilwater=soilwater,
        doc_transport=doc_transport,
        extra_ok_leak_inputs={"rprof": np.ones((npts, nvm), dtype=np.float64)},
        extra_output_inputs={"one_day": 86400.0},
    )

    for name in (
        "litter_above",
        "carbon_32l",
        "clay",
        "doc_to_topsoil",
        "z_soil",
        "zf_soil_b",
        "control_temp_above",
        "control_moist_above",
        "soil_mc_top_by_pft",
        "doc_precip2ground",
        "doc_precip2canopy",
        "dry_dep_canopy",
        "interception_storage",
        "perma_peat",
        "cmax_peat",
        "perma_peat_veget_mask",
        "frac1",
        "frac2",
        "soilwater_31mm",
        "flux_red",
        "rprof",
    ):
        assert name in merged.ok_leak_inputs
    for name in ("carb_mass_total_old", "prod10_total", "contfrac", "zf_soil", "one_day"):
        assert name in merged.output_inputs
    np.testing.assert_allclose(merged.output_inputs["contfrac"], driver.contfrac)


def test_stomate_ok_leak_boundary_inputs_rejects_duplicate_sources():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    restart = stomate_restart_ok_leak_state_inputs(
        state=state,
        veget_max=np.full(state.sla_calc.shape, 0.2, dtype=np.float64),
        z_soil=np.asarray([0.1, 0.4], dtype=np.float64),
        zf_soil=np.asarray([0.0, 0.1, 0.4], dtype=np.float64),
    )

    with np.testing.assert_raises_regex(ValueError, "duplicate explicit source fields"):
        stomate_ok_leak_boundary_inputs(
            restart_state=restart,
            extra_ok_leak_inputs={"litter_above": np.zeros_like(state.litter_above)},
        )
    with np.testing.assert_raises_regex(ValueError, "duplicate explicit source fields"):
        stomate_ok_leak_boundary_inputs(
            restart_state=restart,
            extra_output_inputs={"carb_mass_total_old": np.zeros_like(state.carb_mass_total)},
        )


def test_stomate_pre_step_ok_leak_boundary_inputs_composes_restart_driver_pft_and_vertical_sources():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    npts, nvm = state.sla_calc.shape
    driver = _MiniDriverPayload()
    driver.kjpindex = npts
    driver.nbindex = npts
    driver.kindex = np.arange(1, npts + 1, dtype=np.int32)
    driver.lalo = np.zeros((npts, 2), dtype=np.float64)
    driver.contfrac = np.full(npts, 0.5625, dtype=np.float64)
    driver.resolution = np.ones((npts, 2), dtype=np.float64)
    driver.neighbours = np.full((npts, 8), -1, dtype=np.int32)
    driver.u = np.ones(npts, dtype=np.float64)
    driver.v = np.zeros(npts, dtype=np.float64)
    driver.precip_rain = np.zeros(npts, dtype=np.float64)
    driver.precip_snow = np.zeros(npts, dtype=np.float64)
    driver.swdown = np.ones(npts, dtype=np.float64)
    driver.pb = np.ones(npts, dtype=np.float64)
    driver.veget_max = np.full((npts, nvm), 0.2, dtype=np.float64)
    driver.clay_frac = np.full(npts, 0.1, dtype=np.float64)
    driver.bulk_dens = np.full(npts, 1.3, dtype=np.float64)
    driver.soil_ph = np.full(npts, 6.0, dtype=np.float64)
    driver.poor_soils = np.zeros(npts, dtype=np.float64)
    run_scalars = read_run_scalars(CONFIG)

    pre_step = stomate_pre_step_ok_leak_boundary_inputs(
        state=state,
        driver_payload=driver,
        run_scalars=run_scalars,
        diaglev=np.asarray([0.1, 0.4], dtype=np.float64),
        zz_coef_deep=np.asarray([0.1, 0.4], dtype=np.float64),
        kjit=1,
        nflow=4,
        river_routing=False,
        nbp_glo=npts,
    )

    merged = pre_step.boundary_inputs
    for name in (
        "litter_above",
        "carbon_32l",
        "doc",
        "pref_soil_veg",
        "natural",
        "is_peat",
        "is_c4",
        "clay",
        "bulk_dens",
        "poor_soils",
        "doc_to_topsoil",
        "doc_to_subsoil",
        "flood_frac",
        "fastr",
        "z_soil",
        "zf_soil_b",
        "flux_red",
    ):
        assert name in merged.ok_leak_inputs
    for name in ("carb_mass_total_old", "prod10_total", "prod100_total", "contfrac", "one_day", "z_soil", "zf_soil"):
        assert name in merged.output_inputs
    assert pre_step.doc_transport is not None
    np.testing.assert_allclose(merged.ok_leak_inputs["pref_soil_veg"][PFT14], 3)
    np.testing.assert_allclose(merged.output_inputs["contfrac"], driver.contfrac)


def test_stomate_static_routing_ok_leak_inputs_rejects_missing_static_and_active_routing():
    all_enerbil_inputs = _enerbil_inputs()
    driver = _MiniDriverPayload()
    driver.kjpindex = 2
    driver.nbindex = 2
    driver.kindex = np.asarray([1, 2], dtype=np.int32)
    driver.lalo = np.asarray([[21.0, 109.0], [22.0, 110.0]], dtype=np.float64)
    driver.contfrac = np.ones(2, dtype=np.float64)
    driver.resolution = None
    driver.neighbours = None
    driver.u = np.repeat(all_enerbil_inputs["u"], 2)
    driver.v = np.repeat(all_enerbil_inputs["v"], 2)
    driver.precip_rain = np.repeat(all_enerbil_inputs["precip_rain"], 2)
    driver.precip_snow = np.zeros(2, dtype=np.float64)
    driver.swdown = np.full(2, 350.0, dtype=np.float64)
    driver.pb = np.repeat(all_enerbil_inputs["pb"], 2)
    driver.clay_frac = None
    driver.bulk_dens = np.ones(2, dtype=np.float64)
    driver.soil_ph = np.ones(2, dtype=np.float64) * 6.0
    driver.poor_soils = np.zeros(2, dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "driver/static OK_LEAK boundary"):
        stomate_static_routing_ok_leak_inputs(
            driver_payload=driver,
            kjit=1,
            nflow=4,
            river_routing=False,
            nbp_glo=1,
        )

    driver.clay_frac = np.ones(2, dtype=np.float64) * 0.1
    with np.testing.assert_raises_regex(ValueError, "routing state must be supplied explicitly"):
        stomate_static_routing_ok_leak_inputs(
            driver_payload=driver,
            kjit=1,
            nflow=4,
            river_routing=True,
            nbp_glo=2,
        )
