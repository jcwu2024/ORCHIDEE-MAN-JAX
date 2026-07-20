from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from audit_fortran_oracle_coverage import (  # noqa: E402
    DEFAULT_DISPOSITION,
    DEFAULT_INVENTORY,
    DEFAULT_LEDGER,
    _required_arms_by_entry,
)

COMPARISON = ROOT / "outputs/reference_mode/micro_oracles/stomate_soilcarbon_owner/comparison.json"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stomate_soilcarbon_owner/arm_coverage.json"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90"
ENTRIES = (
    "stomate.active.ok_leak_soilcarbon",
    "stomate.active.tf_doc",
    "stomate.active.ok_leak_firstcall_active_layer",
)
ARM_PATTERN = re.compile(r":(\d+):(?:if|else_if|where|elsewhere):([^:]+)$")


def _case_for_arm(arm_id: str) -> tuple[list[str], str]:
    match = ARM_PATTERN.search(arm_id)
    if match is None:
        raise ValueError(f"unrecognized arm id {arm_id}")
    line = int(match.group(1))
    arm = match.group(2)
    if 2526 <= line <= 2542:
        return ["newaltcalc-spinup-lifecycle"], "newaltcalc=true executes the bottom-up active-layer WHERE chain"
    if 2503 <= line <= 2518:
        return ["normal-lifecycle"], "newaltcalc=false executes the legacy top-down active-layer chain"
    if line == 2560:
        case = "normal-lifecycle" if arm == "true" else "newaltcalc-spinup-lifecycle"
        return [case], "soilc_isspinup differs between the two compiled lifecycle scenarios"
    if 2477 <= line <= 2578:
        return ["normal-lifecycle", "newaltcalc-spinup-lifecycle"], "the two four-call lifecycles cover first/later, warm/frozen, day-1/day-2, and spinup modes"
    return ["normal-lifecycle"], "the normal four-call lifecycle covers mixed points, masks, moisture, flux signs, priming, flood, and perma-peat states"


def build() -> dict[str, object]:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    if comparison.get("status") != "passed" or not all(
        row.get("passed") is True for row in comparison.get("comparisons", [])
    ):
        raise ValueError("soilcarbon full-owner numerical comparison is not passing")
    observed = {
        (str(row["scenario"]), int(row["call"]))
        for row in comparison["comparisons"]
    }
    expected = {(scenario, call) for scenario in ("normal", "newaltcalc_spinup") for call in range(1, 5)}
    if observed != expected:
        raise ValueError(f"incomplete lifecycle comparison matrix: {sorted(expected - observed)}")

    ledger = yaml.safe_load(DEFAULT_LEDGER.read_text(encoding="utf-8"))
    inventory = json.loads(DEFAULT_INVENTORY.read_text(encoding="utf-8"))
    disposition = json.loads(DEFAULT_DISPOSITION.read_text(encoding="utf-8"))
    required = _required_arms_by_entry(ledger, inventory, disposition)
    source_lines = SOURCE.read_text(encoding="latin1").splitlines()
    ledger_entries: dict[str, list[dict[str, object]]] = {}
    for entry in ENTRIES:
        records = []
        for arm_id in sorted(required[entry]):
            match = ARM_PATTERN.search(arm_id)
            if match is None:
                raise ValueError(f"unrecognized arm id {arm_id}")
            line = int(match.group(1))
            case_ids, reason = _case_for_arm(arm_id)
            records.append(
                {
                    "arm_id": arm_id,
                    "case_ids": case_ids,
                    "source_line": line,
                    "source_statement": source_lines[line - 1].strip(),
                    "mapping_basis": reason,
                    "comparison_asset": COMPARISON.relative_to(ROOT).as_posix(),
                }
            )
        covered = {record["arm_id"] for record in records if record["case_ids"]}
        if covered != required[entry]:
            raise ValueError(f"{entry}: missing={sorted(required[entry] - covered)}")
        ledger_entries[entry] = records

    scientific = {
        record["arm_id"]
        for record in disposition["records"]
        if record.get("disposition")
        in {"pft14_active", "pft14_conditional", "paper_static_active"}
        and (
            "stomate_soilcarbon.f90:" in record["arm_id"]
            and (
                641 <= int(ARM_PATTERN.search(record["arm_id"]).group(1)) <= 2416
                or 2438 <= int(ARM_PATTERN.search(record["arm_id"]).group(1)) <= 2578
            )
        )
    }
    ledger_union = {
        record["arm_id"] for records in ledger_entries.values() for record in records
    }
    owner_only_arms = []
    for arm_id in sorted(scientific - ledger_union):
        match = ARM_PATTERN.search(arm_id)
        line = int(match.group(1))
        case_ids, reason = _case_for_arm(arm_id)
        owner_only_arms.append(
            {
                "arm_id": arm_id,
                "case_ids": case_ids,
                "source_line": line,
                "source_statement": source_lines[line - 1].strip(),
                "mapping_basis": reason,
                "comparison_asset": COMPARISON.relative_to(ROOT).as_posix(),
            }
        )
    if ledger_union | {item["arm_id"] for item in owner_only_arms} != scientific:
        raise ValueError("full owner scientific-arm mapping is incomplete")

    return {
        "schema_version": 1,
        "family": "stomate_soilcarbon_owner",
        "branch_complete": True,
        "cases": {
            "normal-lifecycle": {
                "scenario": "normal",
                "calls": [1, 2, 3, 4],
                "newaltcalc": False,
                "soilc_isspinup": False,
            },
            "newaltcalc-spinup-lifecycle": {
                "scenario": "newaltcalc_spinup",
                "calls": [1, 2, 3, 4],
                "newaltcalc": True,
                "soilc_isspinup": True,
            },
        },
        "ledger_entries": ledger_entries,
        "owner_only_arms": owner_only_arms,
        "missing_disposition": [],
        "missing_case_assignment": [],
    }


def main() -> int:
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    counts = {entry: len(arms) for entry, arms in result["ledger_entries"].items()}
    print(f"WROTE {OUTPUT.resolve()} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
