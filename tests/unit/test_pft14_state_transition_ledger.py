from __future__ import annotations

import copy

from scripts.dev.audit_driver_state_transition_contract import build_audit as build_driver
from scripts.dev.audit_pft14_state_transition_ledger import build_audit
from scripts.dev.audit_restart_state_lifecycle import build_audit as build_lifecycle
from scripts.dev.audit_sechiba_state_transition_contract import build_audit as build_sechiba


def test_pft14_state_transition_gate_combines_all_state_denominators():
    audit = build_audit()

    assert audit["closed"] is True
    assert audit["all_components_closed"] is True
    assert all(audit["gates"].values())
    assert audit["counts"]["sechiba_components"] == 5
    assert audit["counts"]["stomate_lifecycle_fields"] == 181
    assert audit["errors"] == {
        "component_mismatch": {"missing": [], "extra": []},
        "boundary_missing_from_lifecycle": {},
        "open_lifecycle_fields": [],
    }
    assert audit["restart_file_phase"]["closed"] is True
    assert audit["restart_file_phase"]["deferred_to_phase_two"] is False


def test_gate_fails_when_a_boundary_field_has_no_lifecycle_contract():
    driver = build_driver()
    lifecycle = build_lifecycle()
    lifecycle["records"] = [
        record for record in lifecycle["records"] if record["field"] != "biomass"
    ]

    audit = build_audit(
        driver_audit=driver,
        sechiba_audit=build_sechiba(),
        lifecycle_audit=lifecycle,
    )

    assert audit["closed"] is False
    assert "biomass" in audit["errors"]["boundary_missing_from_lifecycle"]["half_hour_carry"]


def test_gate_fails_when_a_component_or_field_lifecycle_reopens():
    sechiba = build_sechiba()
    sechiba["all_components_closed"] = False
    lifecycle = build_lifecycle()
    lifecycle = copy.deepcopy(lifecycle)
    lifecycle["records"][0]["lifecycle"]["closed"] = False

    audit = build_audit(
        driver_audit=build_driver(),
        sechiba_audit=sechiba,
        lifecycle_audit=lifecycle,
    )

    assert audit["closed"] is False
    assert audit["gates"]["sechiba_component_contracts_closed"] is False
    assert audit["errors"]["open_lifecycle_fields"]


def test_gate_fails_when_real_restart_alias_roundtrip_reopens():
    alias_audit = {
        "closed": False,
        "summary": {"real_restart_writer_roundtrips": 0},
        "production_roundtrip": {"passed": False},
    }

    audit = build_audit(restart_alias_audit=alias_audit)

    assert audit["closed"] is False
    assert audit["gates"]["restart_alias_roundtrip_closed"] is False
