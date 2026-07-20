from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "audit_sechiba_state_transition_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_sechiba_state_transition_contract", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_current_sechiba_required_fields_have_no_dangling_producers():
    audit = AUDIT.build_audit()
    assert audit["closed"]
    assert audit["errors"]["day1_required_without_producer"] == {}
    assert audit["errors"]["year_required_without_producer"] == {}
    assert audit["errors"]["producer_without_fast_spec_field"] == {}


def test_contract_closes_only_the_audited_component_subset():
    audit = AUDIT.build_audit()
    assert set(audit["closed_components"]) == set(AUDIT.COMPONENTS)
    assert audit["all_components_closed"]
    assert any("slowproc/STOMATE" in limitation for limitation in audit["limitations"])


def test_static_inventory_includes_helpers_carry_and_real_fast_spec():
    audit = AUDIT.build_audit()
    assert set(AUDIT.ADD_HELPERS) < set(audit["producer_functions"])
    assert AUDIT.CARRY_HELPER in audit["producer_functions"]
    assert audit["fast_state_spec_type"] == "DriverFastStateSpec"
    assert (
        "albedo" in audit["components"]["driver_previous_step_state"]["producer_fields"]
    )
    assert (
        "lai" in audit["components"]["diffuco_previous_step_state"]["producer_fields"]
    )
    assert (
        "humrelv"
        not in audit["components"]["diffuco_previous_step_state"]["producer_fields"]
    )
    assert (
        "profil_froz_hydro_ns"
        in audit["components"]["hydrol_previous_step_state"]["producer_fields"]
    )
    assert (
        "profil_froz"
        not in audit["components"]["hydrol_previous_step_state"]["producer_fields"]
    )


def _contract_document():
    return yaml.safe_load(AUDIT.DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def test_field_contract_is_exactly_the_production_inventory():
    audit = AUDIT.build_audit()
    assert audit["contract_count"] == 123
    assert {
        component: len(report["contract_fields"])
        for component, report in audit["components"].items()
    } == {
        "driver_previous_step_state": 10,
        "diffuco_previous_step_state": 35,
        "enerbil_previous_step_state": 13,
        "hydrol_previous_step_state": 49,
        "thermosoil_previous_step_state": 16,
    }
    assert audit["errors"]["producer_without_contract"] == []
    assert audit["errors"]["contract_without_producer"] == []
    assert audit["errors"]["duplicate_contracts"] == []


@pytest.mark.parametrize("missing_key", sorted(AUDIT.CONTRACT_REQUIRED_KEYS))
def test_every_field_contract_attribute_is_mandatory(missing_key):
    document = _contract_document()
    del document["contracts"][0][missing_key]
    audit = AUDIT.build_audit(contract_document=document)
    assert not audit["closed"]
    assert not audit["all_components_closed"]
    incomplete = audit["errors"]["incomplete_contracts"]
    assert incomplete
    assert missing_key in next(iter(incomplete.values()))


def test_missing_duplicate_and_extra_contracts_fail_closed():
    base = _contract_document()

    missing = copy.deepcopy(base)
    missing["contracts"].pop()
    missing_audit = AUDIT.build_audit(contract_document=missing)
    assert not missing_audit["closed"]
    assert missing_audit["errors"]["producer_without_contract"]

    duplicate = copy.deepcopy(base)
    duplicate["contracts"].append(copy.deepcopy(duplicate["contracts"][0]))
    duplicate_audit = AUDIT.build_audit(contract_document=duplicate)
    assert not duplicate_audit["closed"]
    assert duplicate_audit["errors"]["duplicate_contracts"] == [
        "driver_previous_step_state.albedo"
    ]

    extra = copy.deepcopy(base)
    extra_entry = copy.deepcopy(extra["contracts"][0])
    extra_entry["field"] = "not_a_production_field"
    extra["contracts"].append(extra_entry)
    extra_audit = AUDIT.build_audit(contract_document=extra)
    assert not extra_audit["closed"]
    assert extra_audit["errors"]["contract_without_producer"] == [
        "driver_previous_step_state.not_a_production_field"
    ]


def test_owner_shape_dtype_and_fortran_provenance_are_strict_gates():
    cases = (
        (
            "jax_owner",
            "jax_orchidee.driver.orchestration.not_an_owner",
            "invalid_owners",
        ),
        ("shape_group", "not_an_axis_group", "unknown_shape_groups"),
        ("dtype_kind", "numberish", "invalid_dtype_kinds"),
        ("source_ref", "not_a_source", "invalid_source_refs"),
    )
    for key, value, error_key in cases:
        document = _contract_document()
        document["contracts"][0][key] = value
        audit = AUDIT.build_audit(contract_document=document)
        assert not audit["closed"]
        assert "driver_previous_step_state.albedo" in audit["errors"][error_key]


def test_callable_owner_must_be_the_extracted_producer_for_that_exact_field():
    document = _contract_document()
    document["contracts"][0]["jax_owner"] = (
        "jax_orchidee.driver.orchestration._add_hydrol_to_previous_fields"
    )

    audit = AUDIT.build_audit(contract_document=document)

    assert not audit["closed"]
    mismatch = audit["errors"]["owner_field_mismatches"][
        "driver_previous_step_state.albedo"
    ]
    assert mismatch["declared"].endswith("_add_hydrol_to_previous_fields")
    assert mismatch["extracted_production_owners"] == [
        "jax_orchidee.driver.orchestration._add_condveg_to_previous_fields",
        "jax_orchidee.driver.orchestration._runtime_next_state_from_components",
    ]
