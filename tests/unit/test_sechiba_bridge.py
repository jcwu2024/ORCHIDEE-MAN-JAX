from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.bridge import (
    HYDROL_BRIDGE_TRACE_PROVENANCE,
    bridge_group,
    bridge_group_names,
    hydrol_bridge_trace_mappings,
    hydrol_bridge_trace_pairs,
    read_hydrol_bridge_trace_slice,
    required_bridge_fields,
    required_trace_fields,
    validate_bridge_payload,
)


def test_bridge_groups_expose_minimal_sechiba_boundaries():
    assert bridge_group_names() == (
        "hydrol_to_stomate",
        "hydrol_to_thermosoil",
        "enerbil_condveg_to_thermosoil",
        "thermosoil_to_slowproc_enerbil",
        "hydrol_next_sechiba",
        "diffuco_to_stomate",
        "enerbil_to_hydrol_slowproc",
    )

    hydrol_names = required_bridge_fields(("hydrol_to_stomate",))

    assert "humrel" in hydrol_names
    assert "soil_mc" in hydrol_names
    assert "wat_flux" in hydrol_names
    assert "runoff2peat" in hydrol_names
    assert "precip2ground" in hydrol_names


def test_bridge_fields_record_fortran_provenance_and_trace_status():
    group = bridge_group("diffuco_to_stomate")
    gpp = next(field for field in group.fields if field.name == "gpp")

    assert not gpp.requires_trace
    assert gpp.server_1961_trace == "covered"
    assert any("diffuco.f90::diffuco_main" in item for item in gpp.fortran_provenance)
    assert any("stomate.f90::stomate_main" in item for item in gpp.fortran_provenance)

    blocked = bridge_group("hydrol_to_stomate").requires_trace
    assert "soil_mc" not in blocked
    assert "wat_flux" not in blocked
    assert "shumdiag" in blocked


def test_thermosoil_bridge_contract_records_explicit_inputs_and_outputs_without_new_trace():
    hydrol = bridge_group("hydrol_to_thermosoil")
    upstream = bridge_group("enerbil_condveg_to_thermosoil")
    downstream = bridge_group("thermosoil_to_slowproc_enerbil")

    hydrol_names = {field.name for field in hydrol.fields}
    assert {
        "shumdiag_perma",
        "mc_layh",
        "mcl_layh",
        "tmc_layh",
        "mc_layh_pft",
        "mcl_layh_pft",
        "tmc_layh_pft",
    } <= hydrol_names

    tmc_layh = next(field for field in hydrol.fields if field.name == "tmc_layh")
    assert "soilmoist" in tmc_layh.aliases
    assert not tmc_layh.requires_trace
    assert any("thermosoil_humlev" in item for item in tmc_layh.fortran_provenance)

    upstream_names = {field.name for field in upstream.fields}
    assert "temp_sol_new" in upstream_names
    assert "snowdz/snowrho/snowtemp" in upstream_names
    assert upstream.requires_trace == ()
    soilc = next(field for field in upstream.fields if field.name == "soilc_total/refSOC/zx1")
    recurrence = next(
        field for field in upstream.fields if field.name == "ptn/cgrnd/dgrnd/cgrnd_snow/dgrnd_snow/lambda_snow"
    )
    assert soilc.server_1961_trace == "upstream-required"
    assert recurrence.server_1961_trace == "upstream-required"

    output_names = {field.name for field in downstream.fields}
    assert "stempdiag" in output_names
    assert "soilcap/soilcap_pft" in output_names
    assert "soilflx/soilflx_pft" in output_names
    assert "gtemp/ptnlev1" in output_names
    assert downstream.requires_trace == ()
    assert any(
        "thermosoil.f90::thermosoil_main" in item
        for field in downstream.fields
        for item in field.fortran_provenance
    )


def test_required_trace_fields_omits_covered_gpp_but_keeps_missing_energy_fields():
    needs_trace = required_trace_fields(("diffuco_to_stomate", "enerbil_to_hydrol_slowproc"))

    assert "gpp" not in needs_trace
    assert "gsmean/rveget/rstruct/cimean" not in needs_trace
    assert "transpir/transpot/vevapnu/vevapwet/vevapsno/vevapflo" not in needs_trace
    assert "temp_sol/temp_sol_new/temp_sol_pft/qsurf/t2mdiag/evapot_corr" not in needs_trace
    assert needs_trace == ()


def test_required_trace_fields_omits_source_covered_thermosoil_fields_and_restart_state():
    needs_trace = required_trace_fields(
        ("hydrol_to_thermosoil", "enerbil_condveg_to_thermosoil", "thermosoil_to_slowproc_enerbil")
    )

    assert "shumdiag_perma" not in needs_trace
    assert "mc_layh" not in needs_trace
    assert "temp_sol_new" not in needs_trace
    assert "stempdiag" not in needs_trace
    assert "soilcap/soilcap_pft" not in needs_trace
    assert needs_trace == ()

    incomplete = validate_bridge_payload(
        {"temp_sol_new": object()},
        groups=("enerbil_condveg_to_thermosoil",),
    )
    assert not incomplete.ok
    assert "ptn/cgrnd/dgrnd/cgrnd_snow/dgrnd_snow/lambda_snow" in incomplete.missing_by_group[
        "enerbil_condveg_to_thermosoil"
    ]
    assert "soilc_total/refSOC/zx1" in incomplete.missing_by_group["enerbil_condveg_to_thermosoil"]


def test_validate_bridge_payload_reports_missing_without_guessing_values():
    payload = {"gpp": object(), "humrel": object()}

    result = validate_bridge_payload(payload, groups=("hydrol_to_stomate", "diffuco_to_stomate"))

    assert not result.ok
    assert "humrel" not in result.missing_by_group["hydrol_to_stomate"]
    assert "soil_mc" in result.missing_by_group["hydrol_to_stomate"]
    assert "gpp" not in result.missing_by_group["diffuco_to_stomate"]
    assert "gsmean/rveget/rstruct/cimean" in result.missing_by_group["diffuco_to_stomate"]
    assert "soil_mc" not in result.requires_trace_by_group["hydrol_to_stomate"]
    assert result.requires_trace_by_group["diffuco_to_stomate"] == ()


def test_validate_bridge_payload_accepts_thermosoil_source_aliases_without_filling_values():
    payload = {
        "shumdiag_perma": object(),
        "mc_layh": object(),
        "mcl_layh": object(),
        "soilmoist": object(),
        "mc_layh_pft": object(),
        "mcl_layh_pft": object(),
        "soilmoist_pft": object(),
        "temp_sol_new": object(),
        "temp_sol_new_pft": object(),
        "snowdz": object(),
        "snowrho": object(),
        "snowtemp": object(),
        "frac_snow_veg": object(),
        "frac_snow_nobio": object(),
        "totfrac_nobio": object(),
        "ptn": object(),
        "cgrnd": object(),
        "dgrnd": object(),
        "cgrnd_snow": object(),
        "dgrnd_snow": object(),
        "lambda_snow": object(),
        "soilc_total": object(),
        "stempdiag": object(),
        "soilcap": object(),
        "soilcap_pft": object(),
        "soilflx": object(),
        "soilflx_pft": object(),
        "gtemp": object(),
        "ptnlev1": object(),
        "ptn_pftmean": object(),
        "pkappa_pftmean": object(),
        "deephum_prof": object(),
        "deeptemp_prof": object(),
    }

    result = validate_bridge_payload(
        payload,
        groups=("hydrol_to_thermosoil", "enerbil_condveg_to_thermosoil", "thermosoil_to_slowproc_enerbil"),
    )

    assert result.ok
    assert result.missing_by_group["hydrol_to_thermosoil"] == ()
    assert result.missing_by_group["enerbil_condveg_to_thermosoil"] == ()
    assert result.missing_by_group["thermosoil_to_slowproc_enerbil"] == ()


def test_hydrol_bridge_mappings_record_explicit_trace_renames():
    stomate_mappings = hydrol_bridge_trace_mappings(target="stomate")
    slowproc_mappings = hydrol_bridge_trace_mappings(target="slowproc")

    assert any(
        mapping.source_field == "soil_mc" and mapping.target_field == "soil_mc_top_tile"
        for mapping in stomate_mappings
    )
    assert any(
        mapping.source_field == "drainage_per_soil" and mapping.target_field == "drainage_per_soil_tile"
        for mapping in stomate_mappings
    )
    assert any(mapping.name == "vegstress" for mapping in slowproc_mappings)
    assert not any(mapping.name == "vegstress" for mapping in stomate_mappings)
    assert all(mapping.fortran_provenance == HYDROL_BRIDGE_TRACE_PROVENANCE for mapping in stomate_mappings)


def test_read_hydrol_bridge_trace_slice_reads_first_pft14_tile4_records():
    trace_slice = read_hydrol_bridge_trace_slice(kjit=1, ji=1, jv=14, jst=4, jsl=1)

    assert trace_slice.pft["jst_pref"] == 4
    assert trace_slice.layer["jst_pref"] == 4
    assert trace_slice.tile_layer["jst"] == 4
    assert trace_slice.tile_layer["jsl"] == 1
    assert trace_slice.tile["jst"] == 4
    assert trace_slice.before_slowproc["jst_pref"] == 4
    assert trace_slice.before_stomate["jv"] == 14
    assert trace_slice.provenance == HYDROL_BRIDGE_TRACE_PROVENANCE
    assert trace_slice.pft["mc_peat_above"] == pytest.approx(0.299709557382189)
    assert trace_slice.tile_layer["soil_mc"] == pytest.approx(0.299441884927089)


def test_hydrol_bridge_trace_pairs_are_direct_trace_values():
    trace_slice = read_hydrol_bridge_trace_slice(kjit=1, ji=1, jv=14, jst=4, jsl=1)
    pairs = hydrol_bridge_trace_pairs(trace_slice, target="stomate")

    by_name = {pair.mapping.name: pair for pair in pairs}

    assert by_name["soil_mc_top_tile"].source_value == pytest.approx(0.299441884927089)
    assert by_name["soil_mc_top_tile"].target_value == pytest.approx(0.299441884927089)
    assert by_name["drainage_per_soil_tile"].source_value == pytest.approx(8.189820446528955e-7)
    assert by_name["drainage_per_soil_tile"].target_value == pytest.approx(8.189820446528955e-7)
    assert all(pair.source_value == pytest.approx(pair.target_value) for pair in pairs)
