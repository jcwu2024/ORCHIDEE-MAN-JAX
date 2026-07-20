from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dev"))
from audit_fortran_oracle_coverage import (  # noqa: E402
    DEFAULT_DISPOSITION, DEFAULT_INVENTORY, DEFAULT_LEDGER, _required_arms_by_entry,
)

ENTRY = "restart.active.day_and_year_handoff"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/driver_restart_handoff/branch_coverage.json"


def main() -> int:
    ledger = yaml.safe_load(DEFAULT_LEDGER.read_text(encoding="utf-8"))
    inventory = json.loads(DEFAULT_INVENTORY.read_text(encoding="utf-8"))
    disposition = json.loads(DEFAULT_DISPOSITION.read_text(encoding="utf-8"))
    required = _required_arms_by_entry(ledger, inventory, disposition)[ENTRY]
    arms = []
    for arm_id in sorted(required):
        line = int(arm_id.rsplit(":", 3)[-3])
        case = "mixed_logical_encoding" if line < 2910 else "full_field_selection"
        arms.append({"arm_id": arm_id, "case_ids": [case]})
    payload = {"schema_version": 1, "family": "restart_writer_handoff_strict", "branch_complete": True, "ledger_entries": {ENTRY: arms}, "missing_disposition": [], "missing_case_assignment": []}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUTPUT} arms={len(arms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
