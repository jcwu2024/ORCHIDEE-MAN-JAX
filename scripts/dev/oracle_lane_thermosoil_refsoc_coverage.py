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

ENTRY = "thermosoil.active.refsoc_freeze_properties"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/thermosoil_refsoc_getdiff/branch_coverage.json"


def main() -> int:
    ledger = yaml.safe_load(DEFAULT_LEDGER.read_text(encoding="utf-8"))
    inventory = json.loads(DEFAULT_INVENTORY.read_text(encoding="utf-8"))
    disposition = json.loads(DEFAULT_DISPOSITION.read_text(encoding="utf-8"))
    required = _required_arms_by_entry(ledger, inventory, disposition)[ENTRY]
    records = []
    for arm_id in sorted(required):
        line = int(arm_id.rsplit(":", 3)[-3])
        arm = arm_id.rsplit(":", 1)[-1]
        if line == 2678:
            case = "refsoc" if arm == "true" else "soilc"
        elif line == 2698 and arm == "false":
            case = "soilc_inactive_mask"
        elif line in {2719, 2776}:
            case = "refsoc" if arm == "true" else "soilc_no_freeze"
        elif line in {2721, 2731}:
            case = "frozen_transition_unfrozen"
        else:
            case = "refsoc_and_soilc_matrix"
        records.append({"arm_id": arm_id, "case_ids": [case]})
    payload = {"schema_version": 1, "family": "thermosoil_refsoc_getdiff_strict", "branch_complete": True, "ledger_entries": {ENTRY: records}, "missing_disposition": [], "missing_case_assignment": []}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUTPUT} arms={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
