from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.sechiba_boundary import IntersurfFirstStepPayload  # noqa: E402
from jax_orchidee.stomate.entry import (
    assemble_stomate_main_payload,
    contract_gaps,
    daily_accumulation_entry_contract,
    daily_scheduling_contract,
    entry_local_prep_contract,
    entry_normalization_contract,
    first_step_pft14_entry_fields,
    helper_entry_contract,
    littercalc_entry_contract,
    permafrost_control_contract,
    read_first_pft14_entry,
    soilcarbon_leak_contract,
    stomate_diffuco_entry_source,
    stomate_main_argument_contract,
    stomate_main_entry_gaps,
    stomate_driver_entry_source,
    stomate_enerbil_entry_source,
    stomate_erosion_disabled_erodepth_entry_source,
    stomate_hydrol_entry_source,
    stomate_no_routing_entry_source,
    stomate_snow_entry_source,
)
from jax_orchidee.trace.server_1961 import read_server_records


# Fortran provenance for source groups that are not closed by the PFT14 bridge
# trace alone. These comments are intentionally test-side so future source
# mappers have a strict contract before they are implemented in entry.py.
THERMOSOIL_ENTRY_SOURCE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1109-1118",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 788-1054",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 973-1023",
)
THERMOSOIL_STATIC_ENTRY_SOURCE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 993-994",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2431-2432",
)
EROSION_ENTRY_SOURCE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1218-1231",
    "fortran_source/ORCHIDEE/src_sechiba/erosion.f90::erosion_main lines 163-230",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1019-1023",
)


def _by_input(contracts):
    return {contract.helper_input: contract for contract in contracts}


def _by_name(fields):
    return {field.name: field for field in fields}


def _driver_payload():
    return IntersurfFirstStepPayload(
        kjpindex=1,
        nbindex=1,
        kindex=np.asarray([1], dtype=np.int32),
        lalo=np.asarray([[21.0, 109.0]], dtype=np.float64),
        lon=np.asarray([[109.0]], dtype=np.float64),
        lat=np.asarray([[21.0]], dtype=np.float64),
        contfrac=np.asarray([0.5625], dtype=np.float64),
        resolution=np.asarray([[111106.0, 111111.0]], dtype=np.float64),
        neighbours=np.asarray([[-1, -1, -1, -1, -1, -1, -1, -1]], dtype=np.int32),
        corners=np.zeros((1, 4, 2), dtype=np.float64),
        seglength=np.zeros((1, 4), dtype=np.float64),
        zlev=np.asarray([2.0], dtype=np.float64),
        zlevuv=np.asarray([10.0], dtype=np.float64),
        u=np.asarray([-0.8], dtype=np.float64),
        v=np.asarray([-3.8], dtype=np.float64),
        qair=np.asarray([0.008], dtype=np.float64),
        temp_air=np.asarray([291.0], dtype=np.float64),
        pb=np.asarray([98422.0], dtype=np.float64),
        precip_rain=np.asarray([0.0], dtype=np.float64),
        precip_snow=np.asarray([0.0], dtype=np.float64),
        lwdown=np.asarray([338.0], dtype=np.float64),
        swdown=np.asarray([362.0], dtype=np.float64),
        ccanopy=np.asarray([317.27], dtype=np.float64),
        veget_max=np.asarray([[0.0, 1.0]], dtype=np.float64),
        veget=np.asarray([[0.0, 1.0]], dtype=np.float64),
        soiltile=np.asarray([[1.0]], dtype=np.float64),
        njsc=np.asarray([5], dtype=np.int32),
        clay_frac=np.asarray([0.1], dtype=np.float64),
        sand_frac=np.asarray([0.06], dtype=np.float64),
        silt_frac=np.asarray([0.84], dtype=np.float64),
        bulk_dens=np.asarray([1.3859], dtype=np.float64),
        soil_ph=np.asarray([5.66], dtype=np.float64),
        poor_soils=np.asarray([0.0], dtype=np.float64),
        soilclass_sub_index=np.asarray([[1]], dtype=np.int32),
        soilclass_sub_area=np.asarray([[1.0]], dtype=np.float64),
        soilclass=np.zeros((1, 12), dtype=np.float64),
        salinity=np.asarray([33.0], dtype=np.float64),
        tide_height=np.asarray([[1.0]], dtype=np.float64),
        missing_fields=(),
    )


def _hydrol_entry_objects():
    npts = 1
    nslm = 3
    nstm = 2
    base_layer_tile = np.arange(npts * nslm * nstm, dtype=np.float64).reshape(npts, nslm, nstm)
    layer = np.ones((npts, nslm), dtype=np.float64)
    by_tile = np.asarray([[0.2, 0.3]], dtype=np.float64)
    grid = np.asarray([0.4], dtype=np.float64)

    from jax_orchidee.sechiba.hydrol import HydrolModuleDiagnostics, HydrolModuleOutputs

    diagnostics = HydrolModuleDiagnostics(
        layer_by_tile=(),
        stress_by_tile=(),
        nroot=np.ones((npts, 2, nslm), dtype=np.float64),
        soil_wet_ns=base_layer_tile + 1.0,
        us=np.ones((npts, 2, nstm, nslm), dtype=np.float64),
        humrelv=np.ones((npts, 2, nstm), dtype=np.float64),
        vegstressv=np.ones((npts, 2, nstm), dtype=np.float64),
        humrel=np.ones((npts, 2), dtype=np.float64),
        vegstress=np.full((npts, 2), 0.75, dtype=np.float64),
        profil_froz_hydro=None,
        soilmoist=layer + 2.0,
        soilmoist_liquid=layer + 3.0,
        mc_layh=layer + 4.0,
        mcl_layh=layer + 5.0,
        mc_layh_s=base_layer_tile + 6.0,
        mcl_layh_s=base_layer_tile + 7.0,
        shumdiag=layer + 8.0,
        shumdiag_perma=layer + 9.0,
        shumdiag_peat=layer + 10.0,
        shumdiag_croppeat=layer + 11.0,
        shumdiag_man=layer + 12.0,
        undermcr=grid + 13.0,
        tmc_litter=by_tile + 14.0,
        soil_wet_litter=by_tile + 15.0,
        tmc_trampling=grid + 16.0,
        tmc_topgrass=grid + 17.0,
        liqwt_ratio=grid + 18.0,
        wtp=grid + 19.0,
        fwet_new=None,
        mc_peat_above=grid + 20.0,
        mc_croppeat_above=grid + 21.0,
        mc_man_above=grid + 22.0,
        k_litt=grid + 23.0,
        litterhumdiag=grid + 24.0,
        drysoil_frac=grid + 25.0,
        runoff=grid + 26.0,
        drainage=grid + 27.0,
        humtot=grid + 28.0,
        vevapnu=grid + 29.0,
        evap_bare_lim=grid + 30.0,
        ae_ns=by_tile + 31.0,
        evap_bare_lim_ns=by_tile + 32.0,
    )
    outputs = HydrolModuleOutputs(
        runoff_per_soil=by_tile + 30.0,
        drainage_per_soil=by_tile + 31.0,
        runoff2peat=by_tile + 32.0,
        tmc=by_tile + 33.0,
        tmc_soil=by_tile + 34.0,
        wat_flux=base_layer_tile + 35.0,
        wtd=grid + 36.0,
        wtd_ns=by_tile + 37.0,
        precip2canopy=grid + 38.0,
        precip2ground=grid + 39.0,
        canopy2ground=np.ones((npts, 2), dtype=np.float64) + 40.0,
        floodout=grid + 41.0,
    )
    return diagnostics, outputs


def _expected_stomate_payload_missing_after_current_sources():
    return {
        "thermosoil": (
            "tdeep",
            "hsdeep",
            "thawed_humidity",
        ),
        "driver_static": (
            "depth_organic_soil",
            "fc_grazing",
            "humcste_use",
        ),
        "thermosoil_static": (
            "zz_deep",
            "zz_coef_deep",
        ),
        "stomate_state": (
            "soilc_total",
        ),
        "hydrol": (
            "fwet_new",
            "sed_deposition_d",
            "poc_deposition_d",
        ),
    }


def test_first_step_pft14_before_stomate_main_payload_matches_bridge_trace():
    payload = read_first_pft14_entry()
    direct = read_server_records("sechiba_bridge_slowproc", tags="before_stomate_main", limit=1)[0]

    assert payload.kjit == 1
    assert payload.ji == 1
    assert payload.jv == 14
    assert payload.record["tag"] == "before_stomate_main"
    for key in (
        "gpp",
        "lai",
        "veget",
        "veget_max",
        "t2m",
        "temp_sol",
        "humrel",
        "litterhumdiag",
        "precip_rain",
        "precip_snow",
        "snow",
        "wtp",
        "fwet_new",
        "soil_mc_top_tile",
        "wat_flux_top_tile",
    ):
        assert payload.record[key] == direct[key]

    assert payload.record["gpp"] == pytest.approx(0.0, abs=0.0)
    assert payload.record["lai"] == pytest.approx(0.0, abs=0.0)
    assert payload.record["veget"] == pytest.approx(1.0, abs=0.0)
    assert payload.record["veget_max"] == pytest.approx(1.0, abs=0.0)
    assert payload.record["t2m"] == pytest.approx(289.76685333252)
    assert payload.record["temp_sol"] == pytest.approx(280.0, abs=0.0)
    assert payload.record["soil_mc_top_tile"] == pytest.approx(0.299441884927089)
    assert payload.record["wat_flux_top_tile"] == pytest.approx(0.0, abs=0.0)


def test_first_step_pft14_entry_field_contract_lists_traced_and_missing_fields():
    payload = read_first_pft14_entry()
    fields = _by_name(first_step_pft14_entry_fields(payload))

    assert fields["gpp"].status == "covered"
    assert fields["lai"].status == "covered"
    assert fields["veget"].status == "covered"
    assert fields["veget_max"].status == "covered"
    assert fields["t2m"].status == "covered"
    assert fields["temp_sol"].status == "covered"
    assert fields["humrel"].status == "covered"
    assert fields["litterhumdiag"].status == "covered"
    assert fields["precip_rain"].status == "covered"
    assert fields["precip_snow"].status == "covered"
    assert fields["snow"].status == "covered"
    assert fields["wtp"].status == "covered"
    assert fields["fwet_new"].status == "covered"
    assert fields["soil_mc_top_tile"].status == "covered"
    assert fields["wat_flux_top_tile"].status == "covered"

    assert fields["precip"].status == "covered"
    assert fields["precip"].value is None
    assert fields["precip"].trace_fields == ("precip_rain", "precip_snow", "dt_sechiba")
    assert "stomate_precip_increment" in fields["precip"].notes
    with pytest.raises(KeyError, match="precip"):
        payload.require("precip")


def test_daily_accumulation_contract_marks_covered_partial_and_missing_inputs():
    contracts = _by_input(daily_accumulation_entry_contract())

    assert contracts["humrel -> humrel_daily"].status == "covered"
    assert contracts["litterhumdiag -> litterhum_daily"].status == "covered"
    assert contracts["t2m -> t2m_daily"].status == "covered"
    assert contracts["temp_sol -> tsurf_daily"].status == "covered"

    assert contracts["stempdiag -> tsoil_daily"].status == "missing"
    assert contracts["shumdiag -> soilhum_daily"].status == "missing"
    assert contracts["precip -> precip_daily"].status == "covered"
    assert contracts["gpp_d -> gpp_daily"].status == "partial"

    assert contracts["precip -> precip_daily"].entry_fields == ("precip_rain", "precip_snow", "dt_sechiba")
    assert contracts["gpp_d -> gpp_daily"].entry_fields == (
        "gpp",
        "veget_max",
        "dt_sechiba",
        "totfrac_nobio",
    )
    assert "totfrac_nobio" in contracts["gpp_d -> gpp_daily"].notes
    assert "tot_bare_soil" in contracts["gpp_d -> gpp_daily"].notes
    assert "full stempdiag" in contracts["stempdiag -> tsoil_daily"].notes


def test_entry_normalization_contract_separates_current_and_new_nobio_inputs():
    contracts = _by_input(entry_normalization_contract())

    current = contracts["veget, veget_max -> veget_cov, veget_cov_max"]
    firstday = contracts["vegetnew_firstday -> normalized vegetnew_firstday"]
    lcchange = contracts["veget_max_new -> veget_cov_max_new"]

    assert current.status == "partial"
    assert current.entry_fields == ("veget", "veget_max", "totfrac_nobio")
    assert "tot_bare_soil" in current.notes

    assert firstday.status == "missing"
    assert firstday.entry_fields == ("vegetnew_firstday", "totfrac_nobio_new", "date")
    assert "totfrac_nobio_new" in firstday.notes

    assert lcchange.status == "partial"
    assert lcchange.entry_fields == ("veget_max_new", "totfrac_nobio_new", "do_now_stomate_lcchange")
    assert "Current totfrac_nobio" in lcchange.notes
    assert "tot_bare_soil" in lcchange.notes


def test_entry_local_prep_contract_groups_stomate_main_section_3_inputs():
    contracts = _by_input(entry_local_prep_contract())

    assert contracts["precip_rain, precip_snow, dt_sechiba -> precip"].status == "covered"
    current = contracts["veget, veget_max, totfrac_nobio -> veget_cov, veget_cov_max"]
    matrices = contracts[
        "glccNetLCC, glccSecondShift, glccPrimaryShift, harvest_matrix, totfrac_nobio"
    ]
    firstday = contracts["date, vegetnew_firstday, totfrac_nobio_new -> vegetnew_firstday"]
    new_cover = contracts[
        "veget_max_new, totfrac_nobio_new, do_now_stomate_lcchange/dyn_peat/update_peatfrac -> veget_cov_max_new"
    ]
    gpp_d = contracts["gpp, veget_cov_max, dt_sechiba -> gpp_d"]

    assert current.status == "partial"
    assert current.entry_fields == ("veget", "veget_max", "totfrac_nobio")
    assert matrices.status == "partial"
    assert matrices.entry_fields == (
        "glccNetLCC",
        "glccSecondShift",
        "glccPrimaryShift",
        "harvest_matrix",
        "totfrac_nobio",
    )
    assert firstday.status == "missing"
    assert "date == 1" in firstday.notes
    assert new_cover.status == "partial"
    assert "dynamic-peat" in new_cover.notes
    assert gpp_d.status == "partial"
    assert "same section-3 prep" in gpp_d.notes


def test_daily_scheduling_contract_tracks_stomate_main_first_daily_block():
    contracts = _by_input(daily_scheduling_contract())

    assert contracts["day, sec, dt_sechiba -> EndOfMonth"].status == "covered"
    crop = contracts["t2m, ok_LAIdev, SP_tdmin, SP_tdmax, swdown -> st2m, ugdh, uphoi"]
    accum = contracts["explicit increments -> daily accumulators"]
    extrema = contracts["t2m_min, t2m_max -> t2m_min_daily, t2m_max_daily"]

    assert crop.status == "partial"
    assert "PFT14 normally has ok_LAIdev false" in crop.notes
    assert accum.status == "partial"
    assert "no missing daily field is fabricated" in accum.notes
    assert "stempdiag" in accum.entry_fields
    assert "gpp_d" in accum.entry_fields
    assert extrema.status == "covered"
    assert "MIN/MAX" in extrema.notes


def test_littercalc_entry_contract_tracks_partial_littercalc_leak_closure():
    contracts = _by_input(littercalc_entry_contract())

    soil_mc = contracts["soil_mc -> soil_mc_32l"]
    scaled = contracts[
        "turnover_daily, bm_to_litter, dt_sechiba -> turnover_littercalc, bm_to_littercalc"
    ]
    process = contracts["prepared littercalc inputs and litter/carbon state"]

    assert soil_mc.status == "partial"
    assert "top-layer sentinel" in soil_mc.notes
    assert scaled.status == "covered"
    assert scaled.entry_fields == ("turnover_daily", "bm_to_litter", "dt_sechiba")
    assert "source-closed" in scaled.notes
    assert process.status == "partial"
    assert "section-2 litter and lignin additions" in process.notes
    assert "deadleaf cover" in process.notes
    assert "littercalc_leak_core_with_controls" in process.notes
    assert "ordinary and Moyano aboveground" in process.notes
    assert "Full littercalc_leak closure still needs" in process.notes


def test_permafrost_control_contract_tracks_microactem_without_claiming_carbon_core():
    contracts = _by_input(permafrost_control_contract())

    composed = contracts[
        "tdeep, hsdeep, shumdiag_peat, is_peat, tau_peat, z_tau -> prmfrst_soilc_tempctrl"
    ]
    kernel = contracts["temp_celsius, moist_in, mc_peat -> fbact_seconds"]
    downstream = contracts[
        "deepC_a/s/p, fbact_out, gas state, soil carbon input, peat mask, soil grid -> updated OK_PC deep carbon"
    ]

    assert composed.status == "partial"
    assert "full deep thermodynamic arrays" in composed.notes
    assert "is_peat" in composed.entry_fields
    assert kernel.status == "covered"
    assert "2468-2703" in kernel.notes
    assert downstream.status == "partial"
    assert "Local kernels also cover OK_PC" in downstream.notes
    assert "explicit OK_PC daily adapter" in downstream.notes
    assert "firstcall SAVE-sidecar initializer" in downstream.notes
    assert "SAVE-sidecar handoff adapter" in downstream.notes
    assert "firstcall restart-gas wrapper" in downstream.notes
    assert "Fortran OK_PC trace" in downstream.notes


def test_soilcarbon_leak_contract_tracks_ok_leak_path_after_littercalc():
    contracts = _by_input(soilcarbon_leak_contract())

    alt = contracts["tprof, zprof, altmax -> alt, alt_ind, altmax"]
    tf_doc = contracts["precip2ground, precip2canopy, biomass, veget_max -> TF-DOC inputs"]
    export = contracts["DOC_EXP and respiration/water terms -> DOC_EXP_agg"]
    partial = contracts[
        "DOC controls, inputs, POC/DOC decomposition, adsorption, water transport, diffusion, export -> updated carbon/DOC pools"
    ]

    assert alt.status == "partial"
    assert "altmax bookkeeping" in alt.notes
    assert tf_doc.status == "covered"
    assert "ok_TF_DOC" in tf_doc.notes
    assert export.status == "covered"
    assert "labile/refractory DOC aggregation" in export.notes
    assert partial.status == "partial"
    assert "1528-2303" in partial.notes
    assert "water-flux transport" in partial.notes


def test_entry_helper_contract_keeps_stomate_process_boundaries_explicit():
    contracts = helper_entry_contract()
    gaps = _by_input(contract_gaps(contracts))

    assert gaps["veget, veget_max, totfrac_nobio -> veget_cov, veget_cov_max"].status == "partial"
    assert gaps["glccNetLCC, glccSecondShift, glccPrimaryShift, harvest_matrix, totfrac_nobio"].status == "partial"
    assert gaps["date, vegetnew_firstday, totfrac_nobio_new -> vegetnew_firstday"].status == "missing"
    assert (
        gaps[
            "veget_max_new, totfrac_nobio_new, do_now_stomate_lcchange/dyn_peat/update_peatfrac -> veget_cov_max_new"
        ].status
        == "partial"
    )
    assert gaps["gpp, veget_cov_max, dt_sechiba -> gpp_d"].status == "partial"
    assert gaps["t2m, ok_LAIdev, SP_tdmin, SP_tdmax, swdown -> st2m, ugdh, uphoi"].status == "partial"
    assert gaps["explicit increments -> daily accumulators"].status == "partial"
    assert (
        gaps[
            "tdeep, hsdeep, shumdiag_peat, is_peat, tau_peat, z_tau -> prmfrst_soilc_tempctrl"
        ].status
        == "partial"
    )
    assert (
        gaps[
            "deepC_a/s/p, fbact_out, gas state, soil carbon input, peat mask, soil grid -> updated OK_PC deep carbon"
        ].status
        == "partial"
    )
    assert gaps["soil_mc -> soil_mc_32l"].status == "partial"
    assert "turnover_daily, bm_to_litter, dt_sechiba -> turnover_littercalc, bm_to_littercalc" not in gaps
    assert gaps["prepared littercalc inputs and litter/carbon state"].status == "partial"
    assert gaps["tprof, zprof, altmax -> alt, alt_ind, altmax"].status == "partial"
    assert "precip2ground, precip2canopy, biomass, veget_max -> TF-DOC inputs" not in gaps
    assert "DOC_EXP and respiration/water terms -> DOC_EXP_agg" not in gaps
    assert (
        gaps[
            "DOC controls, inputs, POC/DOC decomposition, adsorption, water transport, diffusion, export -> updated carbon/DOC pools"
        ].status
        == "partial"
    )
    assert (
        gaps["explicit littercalc + TF-DOC + soilcarbon_leak + DOC aggregation inputs -> OK_LEAK outputs"].status
        == "partial"
    )
    assert "does not derive phenology" in gaps[
        "explicit littercalc + TF-DOC + soilcarbon_leak + DOC aggregation inputs -> OK_LEAK outputs"
    ].notes
    assert gaps["veget, veget_max -> veget_cov, veget_cov_max"].status == "partial"
    assert gaps["veget_max_new -> veget_cov_max_new"].status == "partial"
    assert gaps["gpp_d -> gpp_daily"].status == "partial"
    assert gaps["biomass"].status == "missing"
    assert gaps["resp_maint_part"].status == "partial"
    assert gaps["f_alloc"].status == "missing"
    assert gaps[
        "gpp/maintenance -> prescribe -> constraints -> phenology(none) -> alloc -> f_alloc/regenerate/post-NPP state"
    ].status == "partial"
    assert gaps["pft_present, frac_growthresp, optional leaf_age/leaf_frac/age/SLA state"].status == "missing"
    assert "allocation_step" in gaps["f_alloc"].notes
    assert "without trace" in gaps[
        "gpp/maintenance -> prescribe -> constraints -> phenology(none) -> alloc -> f_alloc/regenerate/post-NPP state"
    ].notes
    assert "biomass, t2m_longterm, stempdiag, rprof, sla_calc, parameters" in gaps


def test_stomate_main_argument_contract_preserves_fortran_call_order_and_sources():
    payload = read_first_pft14_entry()
    contracts = stomate_main_argument_contract(payload)
    by_name = {contract.name: contract for contract in contracts}

    assert len(contracts) == 124
    assert contracts[0].name == "kjit"
    assert contracts[8].name == "totfrac_nobio"
    assert contracts[13].name == "stempdiag"
    assert contracts[33].name == "gpp"
    assert contracts[-1].name == "poc_deposition_d"

    assert by_name["kjit"].position == 1
    assert by_name["kjit"].status == "covered"
    assert by_name["kjit"].trace_fields == ("kjit",)
    assert by_name["totfrac_nobio"].source == "slowproc_main"
    assert by_name["totfrac_nobio"].status == "missing"
    assert by_name["gpp"].source == "diffuco"
    assert by_name["gpp"].status == "covered"
    assert by_name["veget_max_new"].status == "covered"
    assert by_name["swdown"].status == "missing"
    assert by_name["resp_maint"].direction == "output"
    assert by_name["resp_maint"].status == "output"
    assert any("slowproc_main lines 973-1023" in item for item in by_name["gpp"].provenance)


def test_stomate_main_argument_contract_marks_sentinel_arrays_partial_not_closed():
    payload = read_first_pft14_entry()
    by_name = {contract.name: contract for contract in stomate_main_argument_contract(payload)}

    assert by_name["snowdz"].status == "partial"
    assert by_name["snowdz"].trace_fields == ("snowdz_1",)
    assert "not the full nsnow array" in by_name["snowdz"].notes
    assert by_name["shumdiag_peat"].status == "partial"
    assert by_name["shumdiag_peat"].trace_fields == ("shumdiag_peat_1",)
    assert by_name["soil_mc"].status == "partial"
    assert by_name["soil_mc"].trace_fields == ("soil_mc_top_tile",)
    assert by_name["stempdiag"].status == "missing"
    assert by_name["shumdiag"].status == "missing"


def test_stomate_main_entry_gaps_excludes_outputs_and_keeps_pre_entry_inputs_visible():
    payload = read_first_pft14_entry()
    gaps = {contract.name: contract for contract in stomate_main_entry_gaps(stomate_main_argument_contract(payload))}

    assert "resp_maint" not in gaps
    assert "co2_flux" not in gaps
    assert gaps["totfrac_nobio"].status == "missing"
    assert gaps["totfrac_nobio_new"].status == "missing"
    assert gaps["stempdiag"].source == "thermosoil"
    assert gaps["shumdiag"].source == "hydrol"
    assert gaps["biomass"].source == "stomate_state"


def test_assemble_stomate_main_payload_accepts_only_exact_fortran_argument_names():
    assembly = assemble_stomate_main_payload(
        {
            "kjit": 1,
            "gpp": object(),
            "soil_mc_top_tile": object(),
            "snowdz_1": object(),
        },
        {
            "gpp": "latest-source-wins",
            "soil_mc": "full-array",
        },
    )

    assert not assembly.ready
    assert assembly.payload["gpp"] == "latest-source-wins"
    assert assembly.payload["soil_mc"] == "full-array"
    assert "soil_mc_top_tile" not in assembly.payload
    assert "snowdz_1" not in assembly.payload
    assert "soil_mc" in assembly.covered_inputs
    assert "snowdz" in assembly.missing_inputs
    assert "snowdz" in assembly.missing_by_source["hydrol_condveg"]
    assert assembly.partial_inputs == ()
    assert "resp_maint" in assembly.output_arguments
    assert "resp_maint" not in assembly.missing_inputs


def test_assemble_stomate_main_payload_can_close_declared_inputs_with_explicit_sources():
    names = [
        contract.name
        for contract in stomate_main_argument_contract()
        if contract.direction != "output"
    ]
    explicit = {name: object() for name in names}

    assembly = assemble_stomate_main_payload(explicit)

    assert assembly.ready
    assert assembly.missing_inputs == ()
    assert assembly.partial_inputs == ()
    assert set(assembly.covered_inputs) == set(names)
    assert set(assembly.output_arguments) == {
        "co2_flux",
        "fco2_lu",
        "resp_maint",
        "resp_hetero",
        "resp_growth",
        "temp_growth",
        "f_rot_sech",
        "rot_cmd",
        "peatC",
        "veget_max_adjusted",
        "DOC_EXP_agg",
        "heat_Zimov",
        "sfluxCH4_deep",
        "sfluxCO2_deep",
    }


def test_stomate_driver_entry_source_maps_verified_driver_fields_to_exact_entry_names():
    source = stomate_driver_entry_source(
        _driver_payload(),
        kjit=1,
        t2mdiag=np.asarray([289.0], dtype=np.float64),
        temp_sol=np.asarray([280.0], dtype=np.float64),
        hist_id=10,
        rest_id_stom=20,
    )

    assert source["kjit"] == 1
    assert source["kjpindex"] == 1
    assert np.array_equal(source["index"], np.asarray([1], dtype=np.int32))
    np.testing.assert_allclose(source["t2m"], [289.0])
    np.testing.assert_allclose(source["t2m_min"], [289.0])
    np.testing.assert_allclose(source["t2m_max"], [289.0])
    np.testing.assert_allclose(source["temp_sol"], [280.0])
    np.testing.assert_allclose(source["wspeed"], [np.sqrt(0.8 * 0.8 + 3.8 * 3.8)])
    np.testing.assert_allclose(source["clay"], [0.1])
    np.testing.assert_allclose(source["bulk_dens"], [1.3859])
    assert source["hist_id"] == 10
    assert source["rest_id_stom"] == 20


def test_driver_entry_source_reduces_payload_gaps_without_filling_process_state():
    source = stomate_driver_entry_source(
        _driver_payload(),
        kjit=1,
        t2mdiag=np.asarray([289.0], dtype=np.float64),
        temp_sol=np.asarray([280.0], dtype=np.float64),
    )
    assembly = assemble_stomate_main_payload(
        source,
        {
            "gpp": np.asarray([[0.0, 1.0]], dtype=np.float64),
            "humrel": np.asarray([[0.5, 0.6]], dtype=np.float64),
            "veget": np.asarray([[0.0, 1.0]], dtype=np.float64),
            "veget_max": np.asarray([[0.0, 1.0]], dtype=np.float64),
            "totfrac_nobio": np.asarray([0.0], dtype=np.float64),
            "stempdiag": np.ones((1, 7), dtype=np.float64),
            "snow": np.asarray([0.0], dtype=np.float64),
            "snowdz": np.zeros((1, 3), dtype=np.float64),
            "snowrho": np.zeros((1, 3), dtype=np.float64),
        },
    )

    for name in (
        "kjit",
        "kjpindex",
        "index",
        "lalo",
        "resolution",
        "neighbours",
        "contfrac",
        "clay",
        "t2m",
        "t2m_min",
        "t2m_max",
        "temp_sol",
        "precip_rain",
        "precip_snow",
        "swdown",
        "pb",
        "bulk_dens",
        "soil_ph",
        "poor_soils",
        "gpp",
        "stempdiag",
    ):
        assert name in assembly.covered_inputs
    assert set(assembly.missing_by_source["driver_static"]) == {
        "depth_organic_soil",
        "fc_grazing",
        "humcste_use",
    }
    assert "forcing" not in assembly.missing_by_source
    assert "shumdiag" in assembly.missing_by_source["hydrol"]
    assert "biomass" in assembly.missing_by_source["stomate_state"]


def test_stomate_enerbil_entry_source_keeps_same_step_temp_sol_explicit():
    t2mdiag = np.asarray([289.0], dtype=np.float64)
    evapot_corr = np.asarray([0.025], dtype=np.float64)
    temp_sol = np.asarray([280.0], dtype=np.float64)
    temp_sol_new = np.asarray([284.0], dtype=np.float64)

    without_temp = stomate_enerbil_entry_source(
        t2mdiag=t2mdiag,
        evapot_corr=evapot_corr,
    )
    with_temp = stomate_enerbil_entry_source(
        t2mdiag=t2mdiag,
        evapot_corr=evapot_corr,
        temp_sol=temp_sol,
    )

    assert set(without_temp) == {"t2m", "t2m_min", "t2m_max", "t2mdiag", "evapot_corr"}
    assert "temp_sol" not in without_temp
    np.testing.assert_allclose(with_temp["temp_sol"], temp_sol)
    assert not np.array_equal(with_temp["temp_sol"], temp_sol_new)

    assembly = assemble_stomate_main_payload(without_temp)
    assert "evapot_corr" in assembly.covered_inputs
    assert "temp_sol" in assembly.missing_by_source["enerbil"]


def test_stomate_diffuco_entry_source_maps_only_exact_gpp_export():
    gpp = np.asarray([[0.0, 0.25]], dtype=np.float64)

    source = stomate_diffuco_entry_source(gpp=gpp, gsmean=np.ones_like(gpp))
    assembly = assemble_stomate_main_payload(source)

    assert set(source) == {"gpp"}
    np.testing.assert_allclose(source["gpp"], gpp)
    assert "gpp" in assembly.covered_inputs
    assert "gsmean" not in assembly.covered_inputs


def test_stomate_hydrol_entry_source_maps_only_exact_source_backed_exports():
    diagnostics, outputs = _hydrol_entry_objects()

    source = stomate_hydrol_entry_source(diagnostics=diagnostics, outputs=outputs)

    for name in (
        "humrel",
        "shumdiag",
        "litterhumdiag",
        "tmc_topgrass",
        "shumdiag_peat",
        "mc_peat_above",
        "shumdiag_croppeat",
        "mc_croppeat_above",
        "shumdiag_man",
        "mc_man_above",
        "wtp",
        "liqwt_ratio",
        "soil_mc",
        "wat_flux",
        "drainage_per_soil",
        "runoff_per_soil",
        "runoff2peat",
        "precip2canopy",
        "precip2ground",
        "canopy2ground",
    ):
        assert name in source

    np.testing.assert_allclose(source["soil_mc"], diagnostics.mc_layh_s)
    np.testing.assert_allclose(source["humrel"], diagnostics.vegstress)
    np.testing.assert_allclose(source["wat_flux"], outputs.wat_flux)
    np.testing.assert_allclose(source["drainage_per_soil"], outputs.drainage_per_soil)
    np.testing.assert_allclose(source["runoff_per_soil"], outputs.runoff_per_soil)
    np.testing.assert_allclose(source["runoff2peat"], outputs.runoff2peat)

    for name in (
        "flood_frac",
        "fwet_new",
        "DOC_to_topsoil",
        "DOC_to_subsoil",
        "fastr",
    ):
        assert name not in source


def test_stomate_hydrol_entry_source_accepts_explicit_qflux_and_flood_sum_only():
    diagnostics, outputs = _hydrol_entry_objects()
    wat_flux = np.full((1, 3, 2), 0.125, dtype=np.float64)
    flood_frac_stream = np.asarray([0.75], dtype=np.float64)

    source = stomate_hydrol_entry_source(
        diagnostics=diagnostics,
        outputs=outputs,
        wat_flux=wat_flux,
        flood_frac_plus_streamfl_frac=flood_frac_stream,
    )

    np.testing.assert_allclose(source["wat_flux"], wat_flux)
    np.testing.assert_allclose(source["flood_frac"], flood_frac_stream)
    assert not np.array_equal(source["wat_flux"], outputs.tmc_soil)
    assert not np.array_equal(source["flood_frac"], outputs.floodout)


def test_hydrol_entry_source_reduces_stomate_payload_gaps_without_approximate_fills():
    diagnostics, outputs = _hydrol_entry_objects()
    source = stomate_hydrol_entry_source(diagnostics=diagnostics, outputs=outputs)

    assembly = assemble_stomate_main_payload(source)

    for name in (
        "humrel",
        "shumdiag",
        "litterhumdiag",
        "tmc_topgrass",
        "shumdiag_peat",
        "mc_peat_above",
        "shumdiag_croppeat",
        "mc_croppeat_above",
        "shumdiag_man",
        "mc_man_above",
        "wtp",
        "liqwt_ratio",
        "soil_mc",
        "wat_flux",
        "drainage_per_soil",
        "runoff_per_soil",
        "runoff2peat",
        "precip2canopy",
        "precip2ground",
        "canopy2ground",
    ):
        assert name in assembly.covered_inputs

    for name in (
        "flood_frac",
        "fwet_new",
        "DOC_to_topsoil",
        "DOC_to_subsoil",
        "fastr",
    ):
        assert name in assembly.missing_by_source["hydrol"]


def test_stomate_snow_entry_source_maps_only_exact_snow_state():
    snow = np.asarray([0.25], dtype=np.float64)
    snowdz = np.asarray([[0.1, 0.0, 0.0]], dtype=np.float64)
    snowrho = np.asarray([[150.0, 0.0, 0.0]], dtype=np.float64)

    source = stomate_snow_entry_source(
        snow=snow,
        snowdz=snowdz,
        snowrho=snowrho,
        frac_snow_veg=np.asarray([0.9], dtype=np.float64),
    )
    assembly = assemble_stomate_main_payload(source)

    assert set(source) == {"snow", "snowdz", "snowrho"}
    for name in source:
        np.testing.assert_allclose(source[name], locals()[name])
        assert name in assembly.covered_inputs
    assert "frac_snow_veg" not in assembly.covered_inputs


def test_current_source_backed_payload_keeps_remaining_entry_gaps_explicit():
    source = stomate_driver_entry_source(
        _driver_payload(),
        kjit=1,
        t2mdiag=np.asarray([289.0], dtype=np.float64),
        temp_sol=np.asarray([280.0], dtype=np.float64),
        hist_id=10,
        hist2_id=11,
        rest_id_stom=20,
        hist_id_stom=21,
        hist_id_stom_IPCC=22,
    )
    diagnostics, outputs = _hydrol_entry_objects()
    no_routing = stomate_no_routing_entry_source(
        kjpindex=1,
        nflow=2,
        river_routing=False,
        nbp_glo=1,
    )
    explicit = {
        "gpp": np.asarray([[0.0, 1.0]], dtype=np.float64),
        "humrel": np.asarray([[0.5, 0.6]], dtype=np.float64),
        "veget": np.asarray([[0.0, 1.0]], dtype=np.float64),
        "veget_max": np.asarray([[0.0, 1.0]], dtype=np.float64),
        "totfrac_nobio": np.asarray([0.0], dtype=np.float64),
        "totfrac_nobio_new": np.asarray([0.0], dtype=np.float64),
        "evapot_corr": np.asarray([0.0], dtype=np.float64),
        "stempdiag": np.ones((1, 3), dtype=np.float64),
        "snow": np.asarray([0.0], dtype=np.float64),
        "snowdz": np.zeros((1, 3), dtype=np.float64),
        "snowrho": np.zeros((1, 3), dtype=np.float64),
        "lightn": np.asarray([0.0], dtype=np.float64),
        "popd": np.asarray([0.0], dtype=np.float64),
        "read_observed_ba": False,
        "observed_ba": np.asarray([0.0], dtype=np.float64),
        "humign": np.asarray([0.0], dtype=np.float64),
        "read_cf_fine": False,
        "cf_fine": np.asarray([0.0], dtype=np.float64),
        "read_cf_coarse": False,
        "cf_coarse": np.asarray([0.0], dtype=np.float64),
        "read_ratio_flag": False,
        "ratio_flag": np.asarray([0.0], dtype=np.float64),
        "read_ratio": False,
        "ratio": np.asarray([0.0], dtype=np.float64),
        "deadleaf_cover": np.asarray([0.0], dtype=np.float64),
        "assim_param": np.zeros((1, 2, 1), dtype=np.float64),
        "lai": np.zeros((1, 2), dtype=np.float64),
        "frac_age": np.zeros((1, 2, 4), dtype=np.float64),
        "height": np.zeros((1, 2), dtype=np.float64),
        "veget_max_new": np.asarray([[1.0, 0.0]], dtype=np.float64),
        "vegetnew_firstday": np.asarray([[1.0, 0.0]], dtype=np.float64),
        "glccNetLCC": np.zeros((1, 12), dtype=np.float64),
        "glccSecondShift": np.zeros((1, 12), dtype=np.float64),
        "glccPrimaryShift": np.zeros((1, 12), dtype=np.float64),
        "harvest_matrix": np.zeros((1, 12), dtype=np.float64),
        "bound_spa": np.zeros((1, 2), dtype=np.float64),
        "altmax": np.zeros((1, 2), dtype=np.float64),
        "fpeat": np.zeros((1,), dtype=np.float64),
        "sat_duration": np.zeros((1,), dtype=np.int32),
        "biomass": np.zeros((1, 2, 3, 1), dtype=np.float64),
        "litter_above": np.zeros((1, 2, 2, 1), dtype=np.float64),
        "litter_below": np.zeros((1, 2, 2, 3, 1), dtype=np.float64),
        "carbon_32l": np.zeros((1, 3, 2, 3), dtype=np.float64),
        "DOC": np.zeros((1, 2, 3, 2, 7, 1), dtype=np.float64),
        "lignin_struc_above": np.zeros((1, 2), dtype=np.float64),
        "lignin_struc_below": np.zeros((1, 2, 3), dtype=np.float64),
    }

    assembly = assemble_stomate_main_payload(
        source,
        stomate_hydrol_entry_source(diagnostics=diagnostics, outputs=outputs),
        no_routing,
        stomate_erosion_disabled_erodepth_entry_source(
            erosion_module=False,
            kjpindex=1,
            nvm=2,
        ),
        explicit,
    )

    assert assembly.missing_by_source == _expected_stomate_payload_missing_after_current_sources()
    for name in ("soilc_total", "depth_organic_soil", "thawed_humidity"):
        assert name not in assembly.covered_inputs
        assert name not in assembly.payload


def test_fully_sourced_entry_payload_has_no_blocking_process_gaps():
    from jax_orchidee.sechiba.slowproc import (
        slowproc_dyn_peat_disabled_entry_state,
        slowproc_erosion_daily_zero_entry_state,
        slowproc_fire_disabled_entry_state,
        slowproc_no_lcc_entry_state,
        slowproc_static_entry_state,
        slowproc_thermosoil_entry_init_state,
    )
    from jax_orchidee.stomate.entry import (
        stomate_restart_entry_source,
        stomate_slowproc_dyn_peat_disabled_entry_source,
        stomate_slowproc_erosion_daily_zero_entry_source,
        stomate_slowproc_fire_disabled_entry_source,
        stomate_slowproc_no_lcc_entry_source,
        stomate_slowproc_static_entry_source,
        stomate_slowproc_thermosoil_entry_source,
    )
    from jax_orchidee.stomate.reference import stomate_cold_start_entry_state

    npts, nvm, nslm, ncarb = 1, 2, 3, 3
    driver_source = stomate_driver_entry_source(
        _driver_payload(),
        kjit=1,
        t2mdiag=np.asarray([289.0], dtype=np.float64),
        temp_sol=np.asarray([280.0], dtype=np.float64),
    )
    diagnostics, outputs = _hydrol_entry_objects()
    hydrol_source = stomate_hydrol_entry_source(diagnostics=diagnostics, outputs=outputs)
    hydrol_source["fwet_new"] = np.asarray([0.0], dtype=np.float64)
    slowproc_state = {
        "veget": np.asarray([[0.0, 1.0]], dtype=np.float64),
        "veget_max": np.asarray([[0.0, 1.0]], dtype=np.float64),
        "lai": np.zeros((npts, nvm), dtype=np.float64),
        "frac_age": np.zeros((npts, nvm, 4), dtype=np.float64),
        "height": np.zeros((npts, nvm), dtype=np.float64),
        "totfrac_nobio": np.zeros((npts,), dtype=np.float64),
        "deadleaf_cover": np.zeros((npts,), dtype=np.float64),
    }
    no_lcc = slowproc_no_lcc_entry_state(
        veget_max=slowproc_state["veget_max"],
        use_age_class=False,
        veget_update="0Y",
        map_pft_format=True,
        impose_veg=True,
    )
    restart = stomate_cold_start_entry_state(t2m=np.asarray([289.0], dtype=np.float64), nvm=nvm, nslm=nslm)
    thermosoil = slowproc_thermosoil_entry_init_state(
        kjpindex=npts,
        nvm=nvm,
        znt=np.asarray([0.1, 0.3, 1.0], dtype=np.float64),
        zlt=np.asarray([0.05, 0.2, 0.65], dtype=np.float64),
    )
    static = slowproc_static_entry_state(
        njsc=np.asarray([1], dtype=np.int32),
        soil_classif="zobler",
        pft_to_mtc=np.asarray([1, 2], dtype=np.int32),
        zmaxh=2.0,
        hydrol_humcste=np.asarray([0.8, 0.9], dtype=np.float64),
    )
    same_step = {
        "gpp": np.zeros((npts, nvm), dtype=np.float64),
        "humrel": np.ones((npts, nvm), dtype=np.float64),
        "veget": slowproc_state["veget"],
        "veget_max": slowproc_state["veget_max"],
        "t2m": np.asarray([289.0], dtype=np.float64),
        "t2m_min": np.asarray([289.0], dtype=np.float64),
        "t2m_max": np.asarray([289.0], dtype=np.float64),
        "temp_sol": np.asarray([280.0], dtype=np.float64),
        "evapot_corr": np.asarray([0.0], dtype=np.float64),
        "stempdiag": np.ones((npts, nslm), dtype=np.float64),
        "tdeep": np.ones((npts, nslm, nvm), dtype=np.float64) * 250.0,
        "hsdeep": np.ones((npts, nslm, nvm), dtype=np.float64),
        "thawed_humidity": np.asarray([1.0e20], dtype=np.float64),
        "snow": np.zeros((npts,), dtype=np.float64),
        "snowdz": np.zeros((npts, 3), dtype=np.float64),
        "snowrho": np.zeros((npts, 3), dtype=np.float64),
    }

    assembly = assemble_stomate_main_payload(
        driver_source,
        stomate_no_routing_entry_source(kjpindex=npts, nflow=2, river_routing=False, nbp_glo=npts),
        slowproc_state,
        stomate_slowproc_no_lcc_entry_source(no_lcc),
        stomate_slowproc_fire_disabled_entry_source(slowproc_fire_disabled_entry_state(kjpindex=npts, fire_disable=True)),
        stomate_slowproc_dyn_peat_disabled_entry_source(slowproc_dyn_peat_disabled_entry_state(kjpindex=npts, dyn_peat=False)),
        stomate_slowproc_thermosoil_entry_source(thermosoil),
        stomate_slowproc_static_entry_source(static),
        stomate_erosion_disabled_erodepth_entry_source(erosion_module=False, kjpindex=npts, nvm=nvm),
        stomate_slowproc_erosion_daily_zero_entry_source(
            slowproc_erosion_daily_zero_entry_state(kjpindex=npts, ncarb=ncarb)
        ),
        stomate_restart_entry_source(restart),
        hydrol_source,
        same_step,
    )

    assert assembly.blocking_missing_inputs() == ()
    assert assembly.missing_by_source == {
        "io_handle": ("hist_id", "hist2_id", "rest_id_stom", "hist_id_stom", "hist_id_stom_IPCC")
    }
    assert not assembly.ready
    for name in (
        "tdeep",
        "hsdeep",
        "thawed_humidity",
        "depth_organic_soil",
        "fc_grazing",
        "humcste_use",
        "zz_deep",
        "zz_coef_deep",
        "soilc_total",
        "fwet_new",
        "sed_deposition_d",
        "poc_deposition_d",
    ):
        assert name in assembly.covered_inputs


def test_no_routing_entry_source_closes_only_for_fortran_no_routing_branch():
    source = stomate_no_routing_entry_source(
        kjpindex=2,
        nflow=3,
        river_routing=False,
        nbp_glo=1,
    )

    assert set(source) == {"DOC_to_topsoil", "DOC_to_subsoil", "flood_frac", "fastr"}
    np.testing.assert_allclose(source["DOC_to_topsoil"], np.zeros((2, 3), dtype=np.float64))
    np.testing.assert_allclose(source["DOC_to_subsoil"], np.zeros((2, 3), dtype=np.float64))
    np.testing.assert_allclose(source["flood_frac"], np.zeros((2,), dtype=np.float64))
    np.testing.assert_allclose(source["fastr"], np.zeros((2,), dtype=np.float64))

    single_point_routing_flag = stomate_no_routing_entry_source(
        kjpindex=1,
        nflow=2,
        river_routing=True,
        nbp_glo=1,
    )
    np.testing.assert_allclose(single_point_routing_flag["flood_frac"], [0.0])

    with pytest.raises(ValueError, match="routing state must be supplied explicitly"):
        stomate_no_routing_entry_source(
            kjpindex=2,
            nflow=3,
            river_routing=True,
            nbp_glo=2,
        )


def test_no_routing_entry_source_reduces_stomate_payload_routing_gaps():
    diagnostics, outputs = _hydrol_entry_objects()
    assembly = assemble_stomate_main_payload(
        stomate_hydrol_entry_source(diagnostics=diagnostics, outputs=outputs),
        stomate_no_routing_entry_source(
            kjpindex=1,
            nflow=2,
            river_routing=False,
            nbp_glo=1,
        ),
    )

    for name in ("DOC_to_topsoil", "DOC_to_subsoil", "flood_frac", "fastr"):
        assert name in assembly.covered_inputs
    for name in ("fwet_new",):
        assert name in assembly.missing_by_source["hydrol"]


def test_erosion_disabled_erodepth_source_is_trace_backed_and_guarded():
    source = stomate_erosion_disabled_erodepth_entry_source(
        erosion_module=False,
        kjpindex=1,
        nvm=14,
    )
    assembly = assemble_stomate_main_payload(source)

    assert set(source) == {"erodepth"}
    np.testing.assert_allclose(source["erodepth"], np.zeros((1, 14), dtype=np.float64))
    assert "erodepth" in assembly.covered_inputs
    assert "erodepth" not in assembly.missing_by_source.get("hydrol", ())

    with pytest.raises(ValueError, match="erosion_main"):
        stomate_erosion_disabled_erodepth_entry_source(
            erosion_module=True,
            kjpindex=1,
            nvm=14,
        )


def test_thermosoil_entry_source_maps_only_exact_thermosoil_exports():
    try:
        from jax_orchidee.stomate.entry import stomate_thermosoil_entry_source
    except ImportError:
        pytest.xfail("TDD contract for upcoming thermosoil entry source mapper.")

    values = {
        "stempdiag": np.asarray([[280.0, 281.0, 282.0]], dtype=np.float64),
        "tdeep": np.ones((1, 3, 2), dtype=np.float64),
        "hsdeep": np.ones((1, 3, 2), dtype=np.float64) + 1.0,
        "heat_Zimov": np.ones((1, 3, 2), dtype=np.float64) + 2.0,
        "sfluxCH4_deep": np.asarray([0.1], dtype=np.float64),
        "sfluxCO2_deep": np.asarray([0.2], dtype=np.float64),
        "thawed_humidity": np.asarray([0.5], dtype=np.float64),
        "ptnlev1": np.asarray([279.0], dtype=np.float64),
        "gtemp": np.asarray([278.0], dtype=np.float64),
    }

    source = stomate_thermosoil_entry_source(**values)

    assert set(source) == {
        "stempdiag",
        "tdeep",
        "hsdeep",
        "thawed_humidity",
    }
    for name in source:
        np.testing.assert_allclose(source[name], values[name])
    for forbidden in ("ptnlev1", "gtemp", "soilc_total", "depth_organic_soil"):
        assert forbidden not in source
    assembly = assemble_stomate_main_payload(source)
    assert "tdeep" in assembly.covered_inputs
    assert "hsdeep" in assembly.covered_inputs
    assert "thawed_humidity" in assembly.covered_inputs
    assert "heat_Zimov" not in assembly.covered_inputs
    assert "sfluxCH4_deep" in assembly.output_arguments
    assert "sfluxCO2_deep" in assembly.output_arguments
    assert "soilc_total" in assembly.missing_by_source["stomate_state"]
    assert "depth_organic_soil" in assembly.missing_by_source["driver_static"]
    assert any("thermosoil_main lines 788-1054" in item for item in THERMOSOIL_ENTRY_SOURCE_PROVENANCE)


def test_thermosoil_static_entry_source_maps_soil_levels_without_ambiguous_fills():
    try:
        from jax_orchidee.stomate.entry import stomate_thermosoil_static_entry_source
    except ImportError:
        pytest.xfail("TDD contract for upcoming thermosoil static entry source mapper.")

    zz_deep = np.asarray([0.1, 0.3, 1.0], dtype=np.float64)
    zz_coef_deep = np.asarray([0.05, 0.2, 0.65], dtype=np.float64)

    source = stomate_thermosoil_static_entry_source(
        zz_deep=zz_deep,
        zz_coef_deep=zz_coef_deep,
        soilc_total=np.zeros((1, 3, 2), dtype=np.float64),
        thawed_humidity=np.zeros((1,), dtype=np.float64),
        depth_organic_soil=np.zeros((1,), dtype=np.float64),
    )

    assert set(source) == {"zz_deep", "zz_coef_deep"}
    np.testing.assert_allclose(source["zz_deep"], zz_deep)
    np.testing.assert_allclose(source["zz_coef_deep"], zz_coef_deep)
    for forbidden in ("soilc_total", "thawed_humidity", "depth_organic_soil"):
        assert forbidden not in source
    assembly = assemble_stomate_main_payload(source)
    assert "zz_deep" in assembly.covered_inputs
    assert "zz_coef_deep" in assembly.covered_inputs
    assert "soilc_total" in assembly.missing_by_source["stomate_state"]
    assert "thawed_humidity" in assembly.missing_by_source["thermosoil"]
    assert "depth_organic_soil" in assembly.missing_by_source["driver_static"]
    assert any("slowproc_main lines 993-994" in item for item in THERMOSOIL_STATIC_ENTRY_SOURCE_PROVENANCE)


def test_erosion_entry_source_maps_only_post_erosion_stomate_boundary_exports():
    try:
        from jax_orchidee.stomate.entry import stomate_erosion_entry_source
    except ImportError:
        pytest.xfail("TDD contract for upcoming erosion entry source mapper.")

    erodepth = np.asarray([[0.01, 0.02]], dtype=np.float64)
    sed_deposition_d = np.asarray([0.03], dtype=np.float64)
    poc_deposition_d = np.asarray([[0.04, 0.05, 0.06]], dtype=np.float64)

    source = stomate_erosion_entry_source(
        erodepth=erodepth,
        sed_deposition_d=sed_deposition_d,
        poc_deposition_d=poc_deposition_d,
        POC_EXP_agg=np.ones((1, 3), dtype=np.float64),
        DOC_ERO_agg=np.ones((1, 3), dtype=np.float64),
        SED_EXP_agg=np.ones((1, 3), dtype=np.float64),
    )

    assert set(source) == {"erodepth", "sed_deposition_d", "poc_deposition_d"}
    np.testing.assert_allclose(source["erodepth"], erodepth)
    np.testing.assert_allclose(source["sed_deposition_d"], sed_deposition_d)
    np.testing.assert_allclose(source["poc_deposition_d"], poc_deposition_d)
    for forbidden in ("POC_EXP_agg", "DOC_ERO_agg", "SED_EXP_agg", "soilc_total"):
        assert forbidden not in source
    assembly = assemble_stomate_main_payload(source)
    assert set(source) <= set(assembly.covered_inputs)
    assert "soilc_total" in assembly.missing_by_source["stomate_state"]
    assert any("erosion_main lines 163-230" in item for item in EROSION_ENTRY_SOURCE_PROVENANCE)
