from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (
    LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES,
    STOMATE_DAY_END_WRITEBACK_FIELDS,
    STOMATE_HALF_HOUR_CARRY_FIELDS,
    YEAR_HANDOFF_STATE_GAPS,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "driver_state_transition_contract.json"


def build_audit() -> dict[str, object]:
    carry = set(STOMATE_HALF_HOUR_CARRY_FIELDS)
    day_end = set(STOMATE_DAY_END_WRITEBACK_FIELDS)
    later_entry = set(LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES)
    year_handoff = {
        field
        for gap in YEAR_HANDOFF_STATE_GAPS
        if gap.component == "slowproc_stomate_previous_step_state"
        for field in gap.fields
    }
    missing_later_producers = sorted(later_entry - (carry | day_end))
    missing_year_producers = sorted(year_handoff - (carry | day_end))
    day_only = sorted(day_end - carry)
    return {
        "schema_version": 1,
        "scope": "slowproc/STOMATE half-hour, day-end, and year-handoff boundaries",
        "counts": {
            "half_hour_carry": len(carry),
            "day_end_writeback": len(day_end),
            "later_day_entry_required": len(later_entry),
            "year_handoff_required": len(year_handoff),
        },
        "sets": {
            "half_hour_carry": sorted(carry),
            "day_end_writeback": sorted(day_end),
            "day_boundary_only": day_only,
            "later_day_entry_required": sorted(later_entry),
            "year_handoff_required": sorted(year_handoff),
        },
        "errors": {
            "later_day_fields_without_producer": missing_later_producers,
            "year_handoff_fields_without_producer": missing_year_producers,
        },
        "closed": not missing_later_producers and not missing_year_producers,
        "all_components_closed": False,
        "limitations": [
            "This audit closes only the slowproc/STOMATE boundary subset.",
            "SECHIBA component fields, restart aliases, shapes, masks, and lifecycle FSM remain separate gates.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit driver state boundary producer sets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args(argv)
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.resolve()} closed={audit['closed']} counts={audit['counts']}")
    return int(args.require_closed and not audit["closed"])


if __name__ == "__main__":
    raise SystemExit(main())
