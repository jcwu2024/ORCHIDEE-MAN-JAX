from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "pft14_state_transition_ledger.json"
DEFAULT_RESTART_ALIAS_AUDIT = (
    ROOT / "outputs" / "reference_mode" / "restart_alias_roundtrip.json"
)


def _field_set(report: dict[str, Any], name: str) -> set[str]:
    values = report.get("sets", {}).get(name)
    if not isinstance(values, list):
        raise ValueError(f"driver state audit is missing set {name}")
    return {str(value) for value in values}


def build_audit(
    *,
    driver_audit: dict[str, Any] | None = None,
    sechiba_audit: dict[str, Any] | None = None,
    lifecycle_audit: dict[str, Any] | None = None,
    restart_file_audit: dict[str, Any] | None = None,
    restart_alias_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the independent state audits into the PFT14 state gate.

    This gate covers executable state production and transfer together with
    independent NetCDF construction and the packed restart-alias roundtrip.
    """

    if driver_audit is None:
        from scripts.dev.audit_driver_state_transition_contract import (
            build_audit as build_driver_audit,
        )

        driver_audit = build_driver_audit()
    if sechiba_audit is None:
        from scripts.dev.audit_sechiba_state_transition_contract import (
            build_audit as build_sechiba_audit,
        )

        sechiba_audit = build_sechiba_audit()
    if lifecycle_audit is None:
        from scripts.dev.audit_restart_state_lifecycle import (
            build_audit as build_lifecycle_audit,
        )

        lifecycle_audit = build_lifecycle_audit()
    if restart_file_audit is None:
        from scripts.dev.audit_stomate_restart_file_phase import (
            build_audit as build_restart_file_audit,
        )

        restart_file_audit = build_restart_file_audit()
    if restart_alias_audit is None:
        restart_alias_audit = json.loads(
            DEFAULT_RESTART_ALIAS_AUDIT.read_text(encoding="utf-8")
        )

    lifecycle_records = lifecycle_audit.get("records", [])
    lifecycle_fields = {
        str(record["field"])
        for record in lifecycle_records
        if isinstance(record, dict) and "field" in record
    }
    if len(lifecycle_fields) != len(lifecycle_records):
        raise ValueError("restart lifecycle records contain duplicate or missing field names")

    boundary_sets = {
        name: _field_set(driver_audit, name)
        for name in (
            "half_hour_carry",
            "day_end_writeback",
            "later_day_entry_required",
            "year_handoff_required",
        )
    }
    boundary_missing_from_lifecycle = {
        name: sorted(fields - lifecycle_fields)
        for name, fields in boundary_sets.items()
        if fields - lifecycle_fields
    }

    expected_components = {
        "driver_previous_step_state",
        "diffuco_previous_step_state",
        "enerbil_previous_step_state",
        "hydrol_previous_step_state",
        "thermosoil_previous_step_state",
    }
    component_names = set(sechiba_audit.get("components", {}))
    component_mismatch = {
        "missing": sorted(expected_components - component_names),
        "extra": sorted(component_names - expected_components),
    }
    open_lifecycle_fields = sorted(
        str(record["field"])
        for record in lifecycle_records
        if not record.get("lifecycle", {}).get("closed")
    )

    gates = {
        "sechiba_component_contracts_closed": bool(
            sechiba_audit.get("closed") and sechiba_audit.get("all_components_closed")
        ),
        "slowproc_stomate_boundaries_closed": bool(driver_audit.get("closed")),
        "stomate_field_lifecycle_closed": bool(
            lifecycle_audit.get("state_transition_closed") and not open_lifecycle_fields
        ),
        "boundary_fields_in_lifecycle_ledger": not boundary_missing_from_lifecycle,
        "component_denominator_exact": not component_mismatch["missing"]
        and not component_mismatch["extra"],
        "restart_alias_roundtrip_closed": bool(
            restart_alias_audit.get("closed")
            and restart_alias_audit.get("production_roundtrip", {}).get("passed")
        ),
    }
    closed = all(gates.values())
    restart_file_closed = bool(
        lifecycle_audit.get("restart_lifecycle_closed")
        and restart_file_audit.get("closed")
    )
    return {
        "schema_version": 1,
        "scope": "PFT14 production state transitions and restart alias serialization",
        "closed": closed,
        "all_components_closed": closed,
        "gates": gates,
        "counts": {
            "sechiba_components": len(component_names),
            "sechiba_component_fields": sum(
                len(component.get("producer_fields", ()))
                for component in sechiba_audit.get("components", {}).values()
            ),
            "stomate_lifecycle_fields": len(lifecycle_fields),
            **{name: len(fields) for name, fields in boundary_sets.items()},
        },
        "errors": {
            "component_mismatch": component_mismatch,
            "boundary_missing_from_lifecycle": boundary_missing_from_lifecycle,
            "open_lifecycle_fields": open_lifecycle_fields,
        },
        "restart_file_phase": {
            "closed": restart_file_closed,
            "deferred_to_phase_two": not restart_file_closed,
            "limitations": list(lifecycle_audit.get("limitations", ())),
            "gates": dict(restart_file_audit.get("gates", {})),
        },
        "restart_alias_roundtrip": {
            "closed": restart_alias_audit.get("closed"),
            "production_roundtrip_passed": restart_alias_audit.get(
                "production_roundtrip", {}
            ).get("passed"),
            "real_restart_writer_roundtrips": restart_alias_audit.get(
                "summary", {}
            ).get("real_restart_writer_roundtrips"),
        },
        "inputs": {
            "sechiba": {
                "closed": sechiba_audit.get("closed"),
                "all_components_closed": sechiba_audit.get("all_components_closed"),
            },
            "slowproc_stomate": {"closed": driver_audit.get("closed")},
            "stomate_lifecycle": {
                "state_transition_closed": lifecycle_audit.get("state_transition_closed"),
                "restart_lifecycle_closed": lifecycle_audit.get("restart_lifecycle_closed"),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args(argv)
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.resolve()} closed={audit['closed']} gates={audit['gates']}")
    return int(args.require_closed and not audit["closed"])


if __name__ == "__main__":
    raise SystemExit(main())
