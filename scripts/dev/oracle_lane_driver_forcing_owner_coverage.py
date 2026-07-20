from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dev"))

from audit_fortran_oracle_coverage import (  # noqa: E402
    DEFAULT_DISPOSITION,
    DEFAULT_INVENTORY,
    DEFAULT_LEDGER,
    _required_arms_by_entry,
)

ENTRY = "driver.active.forcing_landpoint_interpolation"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/driver_forcing_owner/branch_coverage.json"


def _case_for(line: int, arm: str) -> str:
    if line == 792:
        return "initialization" if arm == "true" else "standard"
    if line == 796:
        return "standard" if arm == "true" else "scalar_wind"
    if line == 805:
        return "watchout" if arm == "true" else "standard"
    if line == 821:
        return "daily" if arm == "true" else "standard"
    if line == 852:
        return "standard" if arm == "true" else "watchout"
    if line == 853:
        return "standard" if arm == "true" else "no_contfrac"
    if line in {865, 866, 867}:
        return "standard_mixed_grid"
    if line in {932, 933, 934, 985, 991, 997, 1003, 1009, 1015, 1021, 1027}:
        return "watchout_mixed_grid"
    if line <= 1070:
        return "watchout" if line in {918, 955, 972} else "standard_initialization"
    if line == 1077:
        return "standard_initialization" if arm == "false" else "standard"
    if line == 1079:
        return "daily" if arm == "true" else "standard"
    if line == 1081:
        return "daily" if arm == "true" else "daily_split1"
    if line in {1082, 1091, 1094, 1125, 1223, 1234, 1262, 1289, 1346, 1393}:
        return "daily_full_cycle"
    if line == 1148:
        return "daily_watchout" if arm == "true" else "daily"
    if line == 1172:
        return "daily_initial" if arm == "true" else "daily_later"
    if line == 1178:
        return "daily_wrap" if arm == "true" else "daily"
    if line == 1303:
        return "daily_watchout" if arm == "true" else "daily"
    if line == 1354:
        return "high_daily" if arm == "true" else "daily"
    if 1079 <= line <= 1410:
        return "daily_full_cycle"
    if line == 1422:
        return "standard_record_boundary" if arm == "true" else "standard_interior"
    if line == 1425:
        return "standard_cold_start" if arm == "true" else "standard_later"
    if line == 1427:
        return "standard_shift_start" if arm == "true" else "standard_cold_start"
    if line == 1465:
        return "watchout" if arm == "true" else "standard"
    if line == 1488:
        return "wrap" if arm == "true" else "standard"
    if line in {1502, 1527, 1585}:
        return "standard" if arm == "true" else "hourly"
    if line == 1545:
        return "standard" if arm == "true" else "hourly"
    if line == 1557:
        return "watchout" if arm == "true" else "standard"
    if line in {1601, 1603}:
        return "standard_day_night_grid"
    if line == 1607:
        return "high_standard" if arm == "true" else "standard"
    if line == 1613:
        return "hourly"
    if line == 1646:
        return "standard_spread_active" if arm == "true" else "standard_spread_exhausted"
    if line == 1674:
        return "initialization" if arm == "true" else "standard"
    return "standard" if line >= 1422 else "daily"


def main() -> int:
    ledger = yaml.safe_load(DEFAULT_LEDGER.read_text(encoding="utf-8"))
    inventory = json.loads(DEFAULT_INVENTORY.read_text(encoding="utf-8"))
    disposition = json.loads(DEFAULT_DISPOSITION.read_text(encoding="utf-8"))
    required = _required_arms_by_entry(ledger, inventory, disposition)[ENTRY]
    arms = []
    for arm_id in sorted(required):
        parts = arm_id.rsplit(":", 3)
        line = int(parts[-3])
        arm = parts[-1]
        arms.append({"arm_id": arm_id, "case_ids": [_case_for(line, arm)]})
    payload = {
        "schema_version": 1,
        "family": "driver_forcing_owner",
        "branch_complete": True,
        "ledger_entries": {ENTRY: arms},
        "missing_disposition": [],
        "missing_case_assignment": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUTPUT} arms={len(arms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
