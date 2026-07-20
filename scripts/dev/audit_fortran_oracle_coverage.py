from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from extract_fortran_micro_oracle import DEFAULT_MANIFEST, ROOT, load_manifest, validate_and_extract


DEFAULT_LEDGER = ROOT / "docs" / "source_audits" / "pft14_reachable_ledger.yaml"
DEFAULT_INVENTORY = ROOT / "outputs" / "reference_mode" / "fortran_control_flow_inventory.json"
DEFAULT_DISPOSITION = ROOT / "outputs" / "reference_mode" / "pft14_arm_disposition.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "fortran_oracle_coverage.json"
SCIENTIFIC_DISPOSITIONS = {"pft14_active", "pft14_conditional", "paper_static_active"}


def _comparison_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="ascii"))
    return bool(
        result.get("status") == "passed"
        and result.get("comparisons")
        and all(item.get("passed") is True for item in result["comparisons"])
    )


def _line_numbers(value: str | int) -> set[int]:
    result: set[int] = set()
    for item in str(value).split(","):
        bounds = [int(part) for part in item.strip().split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _required_arms_by_entry(
    ledger: dict[str, Any], inventory: dict[str, Any], disposition: dict[str, Any]
) -> dict[str, set[str]]:
    arms = {item["id"]: item for item in inventory.get("control_flow_arms", [])}
    procedures = {item["id"]: item for item in inventory.get("procedures", [])}
    scientific_ids = {
        record["arm_id"]
        for record in disposition.get("records", [])
        if record.get("disposition") in SCIENTIFIC_DISPOSITIONS
    }
    result: dict[str, set[str]] = {}
    for entry in ledger.get("entries", []):
        entry_id = str(entry["id"])
        source_file = str(entry.get("fortran_file", "")).replace("\\", "/")
        procedure_name = str(entry.get("fortran_subroutine", "")).lower()
        lines = _line_numbers(entry.get("fortran_lines", ""))
        result[entry_id] = {
            arm_id
            for arm_id in scientific_ids
            if arm_id in arms
            and arms[arm_id].get("file") == source_file
            and arms[arm_id].get("line") in lines
            and procedures.get(arms[arm_id].get("procedure_id"), {}).get("name", "").lower()
            == procedure_name
        }
    return result


def build_report(
    manifest: dict[str, Any],
    ledger: dict[str, Any],
    *,
    inventory: dict[str, Any] | None = None,
    disposition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extracted = validate_and_extract(manifest, root=ROOT)
    verified_procedures = {
        entry["id"]
        for entry, _ in extracted
        if entry.get("numerical_oracle_status") == "verified"
        and entry.get("compilation", {}).get("status") == "passed"
    }
    reachable = {
        entry["id"]: entry
        for entry in ledger.get("entries", [])
        if entry.get("reachability") in {"active", "conditional"}
    }
    required_by_entry = (
        _required_arms_by_entry(ledger, inventory, disposition)
        if inventory is not None and disposition is not None
        else {entry_id: set(entry.get("control_flow_arm_ids", [])) for entry_id, entry in reachable.items()}
    )
    records: list[dict[str, Any]] = []
    duplicate_owners: list[str] = []
    owned: set[str] = set()
    for family in manifest.get("families", []):
        family_id = str(family["id"])
        result_asset = family.get("result_asset") or f"outputs/reference_mode/micro_oracles/{family_id}/comparison.json"
        result_passed = _comparison_passed(ROOT / result_asset)
        procedure_ids = {
            str(item.get("id") or f"{family_id}.{item.get('procedure', 'unknown')}")
            for item in family.get("procedures", [])
        }
        procedures_verified = bool(procedure_ids) and procedure_ids <= verified_procedures
        cases = family.get("branch_cases", [])
        coverage_by_entry: dict[str, set[str]] = {}
        coverage_asset_valid = False
        if family.get("branch_coverage_asset"):
            coverage_path = ROOT / family["branch_coverage_asset"]
            if coverage_path.is_file():
                coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
                raw_entries = coverage.get("ledger_entries", {})
                coverage_asset_valid = bool(coverage.get("branch_complete")) and not coverage.get(
                    "missing_disposition", []
                ) and not coverage.get("missing_case_assignment", [])
                if isinstance(raw_entries, dict) and raw_entries:
                    coverage_by_entry = {
                        str(entry_id): {
                            str(item["arm_id"])
                            for item in arms
                            if item.get("case_ids") or item.get("input_cases")
                        }
                        for entry_id, arms in raw_entries.items()
                    }
                elif coverage.get("ledger_entry") and isinstance(coverage.get("arms"), list):
                    coverage_by_entry = {
                        str(coverage["ledger_entry"]): {
                            str(item["arm_id"])
                            for item in coverage["arms"]
                            if item.get("case_ids") or item.get("input_cases")
                        }
                    }
        for entry_id in family.get("ledger_entries", []):
            if entry_id in owned:
                duplicate_owners.append(entry_id)
            owned.add(entry_id)
            ledger_entry = reachable.get(entry_id)
            entry_cases = [case for case in cases if case.get("ledger_entry") == entry_id]
            covered_arms = coverage_by_entry.get(
                entry_id,
                {arm_id for case in entry_cases for arm_id in case.get("control_flow_arm_ids", [])},
            )
            required_arms = set() if ledger_entry is None else required_by_entry.get(entry_id, set())
            has_coverage = bool(entry_cases) or (
                coverage_asset_valid and entry_id in coverage_by_entry
            )
            branch_coverage_passed = (
                has_coverage
                and required_arms == covered_arms
            )
            missing_arms = required_arms - covered_arms
            unknown_covered_arms = covered_arms - required_arms
            records.append(
                {
                    "ledger_entry": entry_id,
                    "family": family_id,
                    "known_reachable_entry": ledger_entry is not None,
                    "result_passed": result_passed,
                    "procedures_verified": procedures_verified,
                    "branch_coverage_passed": branch_coverage_passed,
                    "required_arm_count": len(required_arms),
                    "covered_arm_count": len(required_arms & covered_arms),
                    "required_arm_ids": sorted(required_arms),
                    "covered_arm_ids": sorted(required_arms & covered_arms),
                    "missing_arm_ids": sorted(missing_arms),
                    "unknown_covered_arm_ids": sorted(unknown_covered_arms),
                    "passed": bool(
                        ledger_entry is not None
                        and result_passed
                        and procedures_verified
                        and branch_coverage_passed
                    ),
                }
            )
    passed_entries = {record["ledger_entry"] for record in records if record["passed"]}
    return {
        "schema_version": 1,
        "reachable_entries": len(reachable),
        "passed_entries": len(passed_entries),
        "complete": len(passed_entries) == len(reachable) and not duplicate_owners,
        "duplicate_owners": sorted(set(duplicate_owners)),
        "missing_entries": sorted(set(reachable) - passed_entries),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit branch-complete Fortran oracle evidence.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--disposition", type=Path, default=DEFAULT_DISPOSITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        load_manifest(args.manifest),
        yaml.safe_load(args.ledger.read_text(encoding="utf-8")),
        inventory=json.loads(args.inventory.read_text(encoding="utf-8")),
        disposition=json.loads(args.disposition.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.resolve()} passed={report['passed_entries']}/{report['reachable_entries']}")
    return int(args.require_complete and not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
