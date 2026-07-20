from __future__ import annotations

import copy
import json

from scripts.dev.audit_stomate_restart_file_phase import build_audit


def test_stomate_restart_file_phase_gate_is_closed() -> None:
    audit = build_audit()
    assert audit["closed"] is True
    assert all(audit["gates"].values())
    assert audit["counts"] == {
        "dimensions": 26,
        "netcdf_variables": 169,
        "serialized_state_fields": 161,
        "validated_not_serialized_fields": 6,
        "paper_mosaic_templates": 669,
        "paper_mosaic_schema_variants": 1,
    }


def test_schema_gate_fails_for_incomplete_file_definition(tmp_path) -> None:
    from jax_orchidee.stomate.restart_io import DEFAULT_STOMATE_RESTART_SCHEMA

    schema = json.loads(DEFAULT_STOMATE_RESTART_SCHEMA.read_text(encoding="utf-8"))
    schema = copy.deepcopy(schema)
    schema["variables"].pop("time_steps")
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    audit = build_audit(path)
    assert audit["closed"] is False
    assert audit["gates"]["complete_file_definition"] is False
    assert audit["gates"]["audited_schema_exact"] is False
