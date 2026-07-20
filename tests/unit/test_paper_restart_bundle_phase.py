from __future__ import annotations

from scripts.dev.audit_paper_restart_bundle_phase import build_audit


def test_paper_restart_bundle_phase_is_closed() -> None:
    audit = build_audit()

    assert audit["closed"] is True
    assert all(audit["gates"].values())
    assert audit["counts"]["driver_state_fields"] == 13
    assert audit["counts"]["sechiba_state_fields"] == 93
    assert audit["counts"]["stomate_serialized_state_fields"] == 161
    assert audit["counts"]["stomate_validated_fields"] == 6
    assert audit["counts"]["paper_landpoint_schemas"] == 669
    assert audit["counts"]["paper_restart_schema_components"] == 3
