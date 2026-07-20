from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.enerbil_bridge import (
    ENERBIL_TO_HYDROL_BOUNDARY_PROVENANCE,
    ENERBIL_TO_HYDROL_EXPLICIT_SNOW_FIELDS,
    enerbil_to_hydrol_after_main_trace_fields,
    enerbil_to_hydrol_boundary_field,
    enerbil_to_hydrol_boundary_field_names,
    enerbil_to_hydrol_missing_after_main_trace_fields,
    read_enerbil_after_main_payload,
    validate_enerbil_to_hydrol_boundary_payload,
)


TRACED_HYDROL_FIELDS = (
    "temp_sol_new",
    "transpir",
    "transpot",
    "vevapwet",
    "vevapnu",
    "vevapnu_pft",
    "vevapsno",
    "vevapflo",
    "evapot",
    "evapot_corr",
)


def test_enerbil_to_hydrol_contract_names_are_exact_minimal_boundary():
    names = enerbil_to_hydrol_boundary_field_names()

    assert names == TRACED_HYDROL_FIELDS + ("pgflux", "temp_sol_add")
    assert ENERBIL_TO_HYDROL_EXPLICIT_SNOW_FIELDS == ("pgflux", "temp_sol_add")
    assert enerbil_to_hydrol_after_main_trace_fields() == TRACED_HYDROL_FIELDS
    assert enerbil_to_hydrol_missing_after_main_trace_fields() == (
        "pgflux",
        "temp_sol_add",
    )


def test_explicit_snow_fields_are_required_but_not_in_after_enerbil_trace():
    pgflux = enerbil_to_hydrol_boundary_field("pgflux")
    temp_sol_add = enerbil_to_hydrol_boundary_field("temp_sol_add")

    assert not pgflux.covered_by_after_enerbil_trace
    assert not temp_sol_add.covered_by_after_enerbil_trace
    assert pgflux.after_enerbil_trace_status == "missing_from_after_enerbil_main_trace"
    assert temp_sol_add.after_enerbil_trace_status == "missing_from_after_enerbil_main_trace"
    assert "explicit-snow" in pgflux.hydrol_role
    assert "explicit-snow" in temp_sol_add.hydrol_role


def test_current_after_enerbil_payload_covers_traced_10_fields_and_reports_missing_snow_energy():
    payload = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    validation = validate_enerbil_to_hydrol_boundary_payload(payload)

    assert not validation.ok
    assert validation.covered_fields == TRACED_HYDROL_FIELDS
    assert validation.missing_fields == ("pgflux", "temp_sol_add")
    assert validation.missing_explicit_snow_fields == ("pgflux", "temp_sol_add")
    assert "pgflux" not in payload
    assert "temp_sol_add" not in payload


def test_synthetic_complete_payload_passes_name_only_boundary_validation():
    payload = {name: object() for name in enerbil_to_hydrol_boundary_field_names()}
    validation = validate_enerbil_to_hydrol_boundary_payload(payload)

    assert validation.ok
    assert validation.covered_fields == enerbil_to_hydrol_boundary_field_names()
    assert validation.missing_fields == ()
    assert validation.missing_explicit_snow_fields == ()
    assert validation.explicit_snow_required_fields == ("pgflux", "temp_sol_add")


def test_enerbil_to_hydrol_contract_keeps_fortran_provenance_anchors():
    validation = validate_enerbil_to_hydrol_boundary_payload(TRACED_HYDROL_FIELDS)
    provenance = "\n".join(validation.provenance)

    assert validation.provenance == ENERBIL_TO_HYDROL_BOUNDARY_PROVENANCE
    assert "enerbil.f90::enerbil_main" in provenance
    assert "lines 449-472" in provenance
    assert "lines 1541-1588" in provenance
    assert "lines 1756-1833" in provenance
    assert "sechiba.f90::sechiba_main" in provenance
    assert "lines 1049-1064" in provenance
    assert "hydrol.f90::hydrol_main" in provenance
    assert "lines 984-1002" in provenance
    assert "line 1010" in provenance
    assert "lines 1064-1067" in provenance
    assert "line 1090" in provenance
    assert "lines 1182-1196" in provenance
    assert "lines 1232-1238" in provenance
    assert "lines 1278-1284" in provenance
