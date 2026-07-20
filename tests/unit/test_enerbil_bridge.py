from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.enerbil_bridge import (
    ENERBIL_AFTER_MAIN_TRACE_PROVENANCE,
    ENERBIL_ACTIVE_AFTER_TAG,
    ENERBIL_ACTIVE_BEFORE_TAG,
    ENERBIL_POTTEMP_AFTER_TAG,
    ENERBIL_POTTEMP_BEFORE_TAG,
    enerbil_after_main_trace_coverage,
    enerbil_after_main_trace_fields,
    enerbil_active_after_trace_fields,
    enerbil_active_precall_trace_fields,
    enerbil_boundary_field,
    enerbil_boundary_field_names,
    enerbil_required_trace_fields,
    read_enerbil_after_main_payload,
    read_enerbil_after_main_records,
    read_enerbil_pottemp_active_payload,
    validate_enerbil_active_module_closure,
    validate_enerbil_boundary_payload,
    validate_enerbil_to_hydrol_boundary_payload,
)
from jax_orchidee.trace.server_1961 import trace_available, trace_schema


def test_enerbil_boundary_fields_are_individual_minimal_outputs():
    names = enerbil_boundary_field_names()

    assert "transpir" in names
    assert "transpot" in names
    assert "vevapnu" in names
    assert "vevapnu_pft" in names
    assert "vevapwet" in names
    assert "vevapsno" in names
    assert "temp_sol_new" in names
    assert "qsurf" in names
    assert "t2mdiag" in names
    assert "evapot_corr" in names
    assert "transpir/transpot/vevapnu/vevapwet/vevapsno/vevapflo" not in names


def test_enerbil_fields_record_source_consumers_and_missing_trace_status():
    evapot_corr = enerbil_boundary_field("evapot_corr")
    transpir = enerbil_boundary_field("transpir")

    assert evapot_corr.requires_trace
    assert evapot_corr.trace_status == "missing"
    assert any("enerbil.f90::enerbil_flux" in item for item in evapot_corr.fortran_provenance)
    assert any("slowproc_main" in consumer for consumer in evapot_corr.consumers)
    assert any("hydrol_main" in consumer for consumer in evapot_corr.consumers)

    assert transpir.shape == "(npts,nvm)"
    assert any("hydrol_soil" in item for item in transpir.fortran_provenance)


def test_enerbil_required_trace_fields_marks_all_current_boundary_fields():
    needs_trace = enerbil_required_trace_fields()

    assert set(needs_trace) == set(enerbil_boundary_field_names())
    assert "temp_sol_new_pft" in needs_trace
    assert "vevapflo" in needs_trace


def test_validate_enerbil_boundary_payload_does_not_guess_values():
    result = validate_enerbil_boundary_payload({"transpir": object(), "evapot_corr": object()})

    assert not result.ok
    assert "transpir" not in result.missing_fields
    assert "evapot_corr" not in result.missing_fields
    assert "transpot" in result.missing_fields
    assert "temp_sol_new" in result.missing_fields
    assert "transpot" in result.missing_trace_fields


def test_enerbil_after_main_trace_payload_reads_first_pft14_values():
    payload = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)

    assert payload["tag"] == "after_enerbil_main"
    assert payload["kjit"] == 1
    assert payload["ji"] == 1
    assert payload["jv"] == 14
    assert payload["jst_pref"] == 4

    assert payload["transpir"] == pytest.approx(0.0)
    assert payload["transpot"] == pytest.approx(0.0)
    assert payload["vevapwet"] == pytest.approx(0.0)
    assert payload["vevapnu"] == pytest.approx(0.0)
    assert payload["vevapnu_pft"] == pytest.approx(0.0)
    assert payload["vevapsno"] == pytest.approx(0.0)
    assert payload["vevapflo"] == pytest.approx(0.0)
    assert payload["evapot"] == pytest.approx(5.719239060617407e-2)
    assert payload["evapot_corr"] == pytest.approx(2.580207366855209e-2)
    assert payload["temp_sol_new"] == pytest.approx(295.744633030835)
    assert payload["temp_sol_new_pft"] == pytest.approx(295.744633030835)
    assert payload["qsurf"] == pytest.approx(9.809188079088926e-3)
    assert payload["t2mdiag"] == pytest.approx(289.766853332520)
    assert payload["soilcap_pft"] == pytest.approx(59764.3072135468)


def test_enerbil_after_main_trace_coverage_closes_required_fields_without_changing_contract_semantics():
    payload = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    coverage = enerbil_after_main_trace_coverage(payload, kjit=1, ji=1, jv=14)

    assert coverage.ok
    assert coverage.trace_name == "sechiba_bridge_enerbil"
    assert coverage.tag == "after_enerbil_main"
    assert coverage.criteria == {"kjit": 1, "ji": 1, "jv": 14}
    assert coverage.missing_fields == ()
    assert set(coverage.covered_fields) == set(enerbil_boundary_field_names())
    assert set(enerbil_after_main_trace_fields()) == set(enerbil_boundary_field_names())
    assert any("orchjax_sechiba_bridge_enerbil_trace.txt" in item for item in coverage.provenance)
    assert coverage.provenance == ENERBIL_AFTER_MAIN_TRACE_PROVENANCE

    assert set(enerbil_required_trace_fields()) == set(enerbil_boundary_field_names())


def test_enerbil_after_main_records_use_server_trace_parser():
    rows = read_enerbil_after_main_records(limit=1)

    assert len(rows) == 1
    assert rows[0]["tag"] == "after_enerbil_main"
    assert rows[0]["evapot_corr"] == pytest.approx(2.580207366855209e-2)


def test_existing_after_enerbil_trace_does_not_close_explicit_snow_hydrol_boundary():
    payload = read_enerbil_after_main_payload(kjit=1, ji=1, jv=14)
    validation = validate_enerbil_to_hydrol_boundary_payload(payload)

    assert not validation.ok
    assert validation.missing_explicit_snow_fields == ("pgflux", "temp_sol_add")
    assert "pgflux" in validation.missing_fields
    assert "temp_sol_add" in validation.missing_fields


def test_active_enerbil_trace_contract_fields_are_registered():
    schema = trace_schema("sechiba_bridge_enerbil_active")
    before = schema.tags[ENERBIL_ACTIVE_BEFORE_TAG]
    after = schema.tags[ENERBIL_ACTIVE_AFTER_TAG]

    assert before.fields[0:7] == ("kjit", "ji", "jv", "jst_pref", "lai", "gpp", "veget_max")
    assert after.fields[0:7] == ("kjit", "ji", "jv", "jst_pref", "lai", "gpp", "veget_max")
    assert set(enerbil_active_precall_trace_fields()) <= set(before.fields)
    assert set(enerbil_active_after_trace_fields()) <= set(after.fields)
    assert {"swnet", "soilflx", "soilflx_pft", "pgflux"} <= set(before.fields)
    assert {"pgflux", "temp_sol_add", "soilflx", "soilflx_pft"} <= set(after.fields)
    assert len(before.fields) == len(set(before.fields))
    assert len(after.fields) == len(set(after.fields))


def test_active_enerbil_pottemp_trace_schema_is_registered_and_available():
    schema = trace_schema("enerbil_pottemp_active")
    before = schema.tags[ENERBIL_POTTEMP_BEFORE_TAG]
    after = schema.tags[ENERBIL_POTTEMP_AFTER_TAG]

    assert before.fields == ("kjit", "ji", "q_sol_pot", "temp_sol_pot", "qsurf", "temp_sol")
    assert after.fields == ("kjit", "ji", "q_sol_pot", "temp_sol_pot")
    assert any("enerbil_pottemp" in item for item in schema.provenance)
    assert trace_available("enerbil_pottemp_active")

    payload = read_enerbil_pottemp_active_payload(tag=ENERBIL_POTTEMP_BEFORE_TAG, kjit=1, ji=1)
    assert payload["q_sol_pot"] == pytest.approx(9.809188079088926e-3)
    assert payload["temp_sol_pot"] == pytest.approx(280.0)


def test_active_enerbil_module_closure_validator_requires_pre_and_after_fields():
    before_payload = {
        "kjit": 49,
        "ji": 1,
        "jv": 14,
        "jst_pref": 4,
        **{name: object() for name in enerbil_active_precall_trace_fields()},
    }
    after_payload = {
        "kjit": 49,
        "ji": 1,
        "jv": 14,
        "jst_pref": 4,
        **{name: object() for name in enerbil_active_after_trace_fields()},
    }
    closure = validate_enerbil_active_module_closure(
        before_payload=before_payload,
        after_payload=after_payload,
        kjit=49,
        ji=1,
        jv=14,
    )

    assert closure.ok
    assert closure.missing_pre_fields == ()
    assert closure.missing_after_fields == ()
    assert closure.criteria == {"kjit": 49, "ji": 1, "jv": 14}
    assert any("enerbil_main process boundary" in item for item in closure.provenance)

    incomplete = validate_enerbil_active_module_closure(
        before_payload={"swnet": object()},
        after_payload={"temp_sol_new": object()},
        ji=1,
        jv=14,
    )
    assert not incomplete.ok
    assert "soilflx" in incomplete.missing_pre_fields
    assert "soilflx_pft" in incomplete.missing_pre_fields
    assert "pgflux" in incomplete.missing_pre_fields
    assert "temp_sol_add" in incomplete.missing_after_fields
