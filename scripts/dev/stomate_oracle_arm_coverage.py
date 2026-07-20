from __future__ import annotations

import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REQUIRED_DISPOSITIONS = {
    "paper_static_active",
    "pft14_active",
    "pft14_conditional",
}


def load_disposition_records(root: Path) -> list[dict[str, Any]]:
    path = root / "outputs/reference_mode/pft14_arm_disposition.json"
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def arms_for_ledger(
    records: list[dict[str, Any]], ledger_entry: str
) -> tuple[set[str], set[str]]:
    owned = [
        record
        for record in records
        if ledger_entry in record.get("process_ledger_ids", [])
    ]
    required = {
        record["arm_id"]
        for record in owned
        if record["disposition"] in REQUIRED_DISPOSITIONS
    }
    excluded = {record["arm_id"] for record in owned} - required
    return required, excluded


def audit_document(document: dict[str, Any], root: Path) -> dict[str, Any]:
    records = load_disposition_records(root)
    known = {record["arm_id"]: record for record in records}
    reports: list[dict[str, Any]] = []
    seen_entries: set[str] = set()

    for section in ("families", "pending_families"):
        for family in document.get(section, []):
            covered: dict[str, set[str]] = defaultdict(set)
            for case in family.get("branch_cases", []):
                entry = case["ledger_entry"]
                for arm_id in case.get("control_flow_arm_ids", []):
                    if arm_id not in known:
                        raise ValueError(f"{family['id']}:{case['id']}: unknown arm {arm_id}")
                    record = known[arm_id]
                    if entry not in record.get("process_ledger_ids", []):
                        raise ValueError(
                            f"{family['id']}:{case['id']}: {arm_id} is not owned by {entry}"
                        )
                    if record["disposition"] not in REQUIRED_DISPOSITIONS:
                        raise ValueError(
                            f"{family['id']}:{case['id']}: excluded arm claimed as run: {arm_id}"
                        )
                    covered[entry].add(arm_id)

            for entry in family["ledger_entries"]:
                if entry in seen_entries:
                    raise ValueError(f"duplicate lane-C ledger entry: {entry}")
                seen_entries.add(entry)
                required, excluded = arms_for_ledger(records, entry)
                missing = required - covered[entry]
                branch_complete = bool(required) and not missing
                if section == "families" and not branch_complete:
                    raise ValueError(
                        f"{family['id']}:{entry}: verified family misses "
                        f"{len(missing)}/{len(required)} authoritative arms"
                    )
                reports.append(
                    {
                        "family": family["id"],
                        "ledger_entry": entry,
                        "section": section,
                        "required_count": len(required),
                        "covered_count": len(covered[entry]),
                        "excluded_count": len(excluded),
                        "missing_arm_ids": sorted(missing),
                        "branch_complete": branch_complete,
                    }
                )

    return {
        "schema_version": 1,
        "required_dispositions": sorted(REQUIRED_DISPOSITIONS),
        "entries": reports,
        "totals": {
            "entries": len(reports),
            "branch_complete": sum(item["branch_complete"] for item in reports),
            "required_arms": sum(item["required_count"] for item in reports),
            "covered_arms": sum(item["covered_count"] for item in reports),
            "missing_arms": sum(len(item["missing_arm_ids"]) for item in reports),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fragments", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    combined: list[dict[str, Any]] = []
    for fragment in args.fragments:
        document = yaml.safe_load(fragment.read_text(encoding="utf-8"))
        report = audit_document(document, args.root)
        combined.extend(report["entries"])
    result = {
        "schema_version": 1,
        "entries": combined,
        "totals": {
            "entries": len(combined),
            "branch_complete": sum(item["branch_complete"] for item in combined),
            "required_arms": sum(item["required_count"] for item in combined),
            "covered_arms": sum(item["covered_count"] for item in combined),
            "missing_arms": sum(len(item["missing_arm_ids"]) for item in combined),
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
