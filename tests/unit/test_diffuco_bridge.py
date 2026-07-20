from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.diffuco_bridge import (
    ACTIVE_PAPER_CASE_BRANCHES,
    DIFFUCO_PFT14_FIRST_TIMESTEP_CRITICAL_FIELDS,
    DIFFUCO_TRANS_CO2_PARITY_TAG,
    DIFFUCO_TRANS_CO2_TRACE,
    DIFFUCO_TRANS_TO_AFTER_MAIN_OVERLAP_FIELDS,
    DIFFUCO_ACTIVE_AFTER_MAIN_TRACE,
    diffuco_field,
    diffuco_numeric_parity_readiness,
    diffuco_numeric_parity_required_fields,
    first_active_pft14_diffuco_after_main_payload,
    first_active_pft14_diffuco_trans_co2_payload,
    first_pft14_diffuco_after_main_payload,
    first_pft14_diffuco_trans_co2_payload,
    first_pft14_diffuco_critical_values,
    minimal_diffuco_field_names,
    read_diffuco_after_main_records,
    read_diffuco_trans_co2_records,
    required_after_diffuco_trace_fields,
    validate_diffuco_after_main_trace_payload,
    validate_diffuco_boundary,
    validate_diffuco_module_closure,
    validate_diffuco_trans_to_after_main_overlap,
)
from jax_orchidee.trace.server_1961 import BRIDGE_TRACE_ROOT, trace_schema


def test_minimal_diffuco_contract_marks_gpp_and_resistance_fields_as_trace_required():
    names = minimal_diffuco_field_names()

    assert "gpp" in names
    assert "gsmean" in names
    assert "rveget" in names
    assert "rstruct" in names
    assert "vbeta3" in names
    assert "control_salinity/control_inudate" in names

    after_diffuco = required_after_diffuco_trace_fields()
    assert "gpp" in after_diffuco
    assert "q_cdrag/q_cdrag_pft" in after_diffuco
    assert "control_salinity/control_inudate" in after_diffuco


def test_diffuco_fields_carry_fortran_provenance_and_consumers():
    gpp = diffuco_field("gpp")

    assert gpp.requires_trace
    assert gpp.trace_point == "after_diffuco_main"
    assert "stomate_main" in gpp.consumers
    assert any("diffuco_trans_co2 assigns gpp" in item for item in gpp.fortran_provenance)
    assert any("stomate.f90::stomate_main accumulates" in item for item in gpp.fortran_provenance)

    controls = diffuco_field("control_salinity/control_inudate")
    assert any("jv==14" in item for item in controls.fortran_provenance)


def test_active_paper_case_branches_are_source_backed():
    assert ACTIVE_PAPER_CASE_BRANCHES["co2_photosynthesis"][0] is True
    assert "STOMATE_OK_CO2=y" in ACTIVE_PAPER_CASE_BRANCHES["co2_photosynthesis"][1]

    assert ACTIVE_PAPER_CASE_BRANCHES["mangrove_salinity_tide_control"][0] is True
    assert "READ_SALINITY=TRUE" in ACTIVE_PAPER_CASE_BRANCHES["mangrove_salinity_tide_control"][1]

    assert ACTIVE_PAPER_CASE_BRANCHES["bvoc_chemistry"][0] is False
    assert "CHEMISTRY_BVOC=FALSE" in ACTIVE_PAPER_CASE_BRANCHES["bvoc_chemistry"][1]

    assert ACTIVE_PAPER_CASE_BRANCHES["choisnel_hydrology"][0] is False
    assert "hydrolc_main" in ACTIVE_PAPER_CASE_BRANCHES["choisnel_hydrology"][1]


def test_validate_diffuco_boundary_reports_missing_without_guessing():
    result = validate_diffuco_boundary({"gpp": object(), "vbeta3": object()})

    assert not result.ok
    assert "gpp" not in result.missing_fields
    assert "vbeta3" not in result.missing_fields
    assert "gsmean" in result.missing_fields
    assert "control_salinity/control_inudate" in result.missing_fields
    assert "gpp" in result.requires_trace


def test_diffuco_after_main_trace_payload_parses_exact_schema_arity():
    rows = read_diffuco_after_main_records(limit=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["tag"] == "after_diffuco_main"
    assert len([key for key in row if key not in {"tag", "start_line"}]) == 33
    assert row["kjit"] == 1
    assert row["ji"] == 1
    assert row["jv"] == 14
    assert row["jst_pref"] == 4


def test_diffuco_after_main_trace_covers_trace_backed_contract_but_not_internal_controls():
    payload = first_pft14_diffuco_after_main_payload()
    validation = validate_diffuco_after_main_trace_payload(payload)

    assert validation.parse_and_coverage_ok
    assert not validation.complete
    assert validation.missing_contract_fields == ()
    assert validation.uncovered_internal_diagnostics == ("control_salinity/control_inudate",)
    assert "control_salinity" not in validation.available_columns
    assert "control_inudate" not in validation.available_columns
    assert "control_salinity/control_inudate" not in validation.covered_contract_fields
    assert all(validation.active_branch_checks.values())
    assert any("after_diffuco_main" in item for item in validation.provenance)


def test_existing_diffuco_bridge_trace_is_not_strict_numeric_parity_ready():
    readiness = diffuco_numeric_parity_readiness(root=BRIDGE_TRACE_ROOT)

    assert not readiness.ready
    assert "swdown" in readiness.available_fields
    assert "gpp" in readiness.available_fields
    assert "qsatt" in readiness.missing_fields
    assert "ca" in readiness.missing_fields
    assert "vcmax" in readiness.missing_fields
    assert "vbeta23" in readiness.missing_fields
    assert "control_salinity" in readiness.missing_fields
    assert "control_inudate" in readiness.missing_fields
    assert DIFFUCO_TRANS_CO2_PARITY_TAG in readiness.provenance[-1]


def test_enhanced_diffuco_trace_package_is_strict_numeric_parity_ready_by_default():
    readiness = diffuco_numeric_parity_readiness()

    assert readiness.ready
    assert readiness.missing_fields == ()
    assert readiness.trace_sources["qsatt"] == DIFFUCO_TRANS_CO2_PARITY_TAG
    assert readiness.trace_sources["control_salinity"] == DIFFUCO_TRANS_CO2_PARITY_TAG
    assert readiness.trace_sources["evap_bare_lim"] == "after_hydrol_main_pft"


def test_diffuco_trans_to_after_main_overlap_closes_for_available_inactive_steps():
    mismatches = validate_diffuco_trans_to_after_main_overlap()

    assert tuple(mismatches) == (1, 2, 3, 4)
    assert all(fields == () for fields in mismatches.values())
    assert "gpp" in DIFFUCO_TRANS_TO_AFTER_MAIN_OVERLAP_FIELDS
    assert "vbeta3pot" in DIFFUCO_TRANS_TO_AFTER_MAIN_OVERLAP_FIELDS


def test_diffuco_module_closure_reports_active_trans_co2_and_active_after_main_closed():
    active_trans = first_active_pft14_diffuco_trans_co2_payload()
    active_after = first_active_pft14_diffuco_after_main_payload()
    closure = validate_diffuco_module_closure()

    assert active_trans["kjit"] == 49
    assert active_trans["lai"] > 0.01
    assert active_after is not None
    assert active_after["kjit"] == 49
    assert active_after["lai"] == active_trans["lai"]
    assert closure.trans_co2_active_closed
    assert closure.inactive_main_boundary_closed
    assert closure.active_main_boundary_closed
    assert closure.module_closed
    assert closure.missing_active_after_main_fields == ()
    assert all(fields == () for fields in closure.overlap_mismatches.values())
    assert any("diffuco_main lines 665-710" in item for item in closure.provenance)


def test_active_after_main_trace_is_registered_and_copied():
    closure = validate_diffuco_module_closure()

    assert DIFFUCO_ACTIVE_AFTER_MAIN_TRACE == "sechiba_bridge_diffuco_active"
    assert closure.after_main_active_record_found
    assert closure.missing_active_after_main_fields == ()


def test_first_active_after_main_matches_trans_co2_output_boundary_fields():
    active_trans = first_active_pft14_diffuco_trans_co2_payload()
    active_after = first_active_pft14_diffuco_after_main_payload()

    assert active_after is not None
    assert active_after["kjit"] == active_trans["kjit"] == 49
    for field in (
        "gpp",
        "gsmean",
        "rveget",
        "rstruct",
        "cimean",
        "vbeta3",
        "vbeta3pot",
        "q_cdrag",
        "q_cdrag_pft",
        "humrel",
        "qsintveg",
        "qsintmax",
        "veget",
        "veget_max",
        "lai",
        "qsurf",
    ):
        assert active_after[field] == pytest.approx(active_trans[field])

    assert active_after["valpha"] == pytest.approx(1.0)
    assert active_after["vbeta"] == pytest.approx(active_after["vbeta2"] + active_after["vbeta3"] + active_after["vbeta4"])
    assert active_after["evap_bare_lim"] == pytest.approx(0.0)


def test_diffuco_trans_co2_trace_fixture_parses_and_feeds_strict_readiness(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    schema = trace_schema(DIFFUCO_TRANS_CO2_TRACE).tags[DIFFUCO_TRANS_CO2_PARITY_TAG]
    values = {
        "kjit": 1,
        "ji": 1,
        "jv": 14,
        "swdown": 10.0,
        "pb": 987.0,
        "qsurf": 0.01,
        "qsatt": 0.02,
        "t2m": 289.0,
        "temp_growth": 289.0,
        "ca": 317.27,
        "vcmax": 50.0,
        "humrel": 1.0,
        "veget": 1.0,
        "veget_max": 1.0,
        "lai": 0.0,
        "qsintveg": 0.0,
        "qsintmax": 0.1,
        "vbeta23": 1.0,
        "q_cdrag": 0.1,
        "q_cdrag_pft": 0.1,
        "wind": 2.0,
        "control_salinity": 1.0,
        "control_inudate": 1.0,
        "gpp": 0.0,
        "gsmean": 0.0,
        "rveget": 1.0e20,
        "rstruct": 0.0,
        "cimean": 0.0,
        "vbeta3": 0.0,
        "vbeta3pot": 0.0,
        "assimtot": 0.0,
        "rdtot": 0.0,
        "gstot": 0.0,
        "leaf_gs_top": 0.0,
        "laisum": 0.0,
        "cim": 0.0,
        "ilai": 0,
        "gamma_star": 0.0,
        "fvpd": 1.0,
        "g0var": 0.0,
    }
    trace_path = trace_dir / "orchjax_diffuco_trans_co2_trace.txt"
    trace_path.write_text(
        f"{DIFFUCO_TRANS_CO2_PARITY_TAG} "
        + " ".join(str(values[field]) for field in schema.fields)
        + "\n",
        encoding="utf-8",
    )

    rows = read_diffuco_trans_co2_records(root=trace_dir)
    payload = first_pft14_diffuco_trans_co2_payload(root=trace_dir)
    readiness = diffuco_numeric_parity_readiness(
        records=(
            payload,
            {
                "tag": "after_hydrol_main_pft",
                "evap_bare_lim": 0.0,
                "tot_bare_soil": 0.0,
                "vbeta": 0.0,
                "vbeta_pft": 0.0,
                "vbeta1": 0.0,
                "vbeta2": 0.0,
                "vbeta4": 0.0,
                "vbeta4_pft": 0.0,
            },
        )
    )

    assert rows == (payload,)
    assert payload["tag"] == DIFFUCO_TRANS_CO2_PARITY_TAG
    assert payload["start_line"] == 1
    assert payload["control_salinity"] == pytest.approx(1.0)
    assert payload["g0var"] == pytest.approx(0.0)
    assert readiness.ready
    assert readiness.trace_sources["qsatt"] == DIFFUCO_TRANS_CO2_PARITY_TAG
    assert readiness.trace_sources["evap_bare_lim"] == "after_hydrol_main_pft"


def test_diffuco_numeric_parity_required_fields_cover_trans_co2_and_beta_chain():
    required = diffuco_numeric_parity_required_fields()

    assert "qsatt" in required
    assert "vcmax" in required
    assert "vbeta23" in required
    assert "evap_bare_lim" in required
    assert "tot_bare_soil" in required
    assert "vbeta4_pft" in required


def test_diffuco_first_pft14_critical_values_are_read_from_trace_payload():
    payload = first_pft14_diffuco_after_main_payload()
    critical = first_pft14_diffuco_critical_values()

    assert tuple(critical) == DIFFUCO_PFT14_FIRST_TIMESTEP_CRITICAL_FIELDS
    assert critical == {field: payload[field] for field in DIFFUCO_PFT14_FIRST_TIMESTEP_CRITICAL_FIELDS}
    assert critical["gpp"] == pytest.approx(0.0)
    assert critical["salinity"] == pytest.approx(33.227593421936)
    assert critical["tide_height_1"] == pytest.approx(-1.39575001597404)
    assert critical["veget_max"] == pytest.approx(1.0)
    assert critical["lai"] == pytest.approx(0.0)
    assert critical["temp_sol"] == pytest.approx(280.0)
    assert critical["evapot_corr"] == pytest.approx(0.0)
