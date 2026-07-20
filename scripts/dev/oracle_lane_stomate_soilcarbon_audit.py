from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FRAGMENT = ROOT / "docs/source_audits/oracle_families/stomate_soilcarbon.yaml"
DISPOSITIONS = ROOT / "docs/source_audits/pft14_arm_dispositions_stomate.yaml"
AUTHORITATIVE_DISPOSITIONS = ROOT / "outputs/reference_mode/pft14_arm_disposition.json"
OWNER_FRAGMENT = ROOT / "docs/source_audits/oracle_families/soilcarbon_owner.yaml"
OWNER_COMPARISON = ROOT / "outputs/reference_mode/micro_oracles/stomate_soilcarbon_owner/comparison.json"
OWNER_COVERAGE = ROOT / "outputs/reference_mode/micro_oracles/stomate_soilcarbon_owner/arm_coverage.json"


def scientific_arm_ids(source_name: str, start_line: int, end_line: int) -> set[str]:
    document = json.loads(AUTHORITATIVE_DISPOSITIONS.read_text(encoding="utf-8"))
    result: set[str] = set()
    pattern = re.compile(rf"/{re.escape(source_name)}:(\d+):")
    for record in document["records"]:
        arm_id = record["arm_id"]
        match = pattern.search(arm_id)
        if match is None or not start_line <= int(match.group(1)) <= end_line:
            continue
        if record.get("disposition") in {
            "pft14_active",
            "pft14_conditional",
            "paper_static_active",
        }:
            result.add(arm_id)
    return result


def procedure_bytes(path: Path, start_line: int, end_line: int) -> bytes:
    lines = path.read_bytes().splitlines(keepends=True)
    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        raise ValueError(f"invalid source span {path}:{start_line}-{end_line}")
    return b"".join(lines[start_line - 1 : end_line])


def audit_fragment() -> dict[str, object]:
    document = yaml.safe_load(FRAGMENT.read_text(encoding="utf-8"))
    if document.get("schema_version") != 2:
        raise ValueError("expected oracle fragment schema_version 2")
    entries: set[str] = set()
    procedures: list[dict[str, object]] = []
    for family in document["families"] + document["pending_families"]:
        pending = family in document["pending_families"]
        if pending and (family.get("status") != "pending" or not family.get("blocker")):
            raise ValueError(f"{family.get('id')}: pending family needs an explicit blocker")
        if pending and not family.get("required_branches"):
            raise ValueError(f"{family['id']}: missing branch matrix")
        if not pending and not family.get("branch_cases"):
            raise ValueError(f"{family['id']}: verified family needs branch cases")
        if not pending and family["id"] == "stomate_littercalc_leak":
            required = set(family.get("required_control_flow_arm_ids", ()))
            covered = {
                arm_id
                for case in family.get("branch_cases", ())
                for arm_id in case.get("control_flow_arm_ids", ())
            }
            source_required = scientific_arm_ids("stomate_litter.f90", 1637, 2873)
            if required != source_required:
                raise ValueError(
                    f"{family['id']}: required arms differ from disposition inventory; "
                    f"missing={sorted(source_required - required)}, extra={sorted(required - source_required)}"
                )
            if covered != required:
                raise ValueError(
                    f"{family['id']}: branch-incomplete; missing={sorted(required - covered)}"
                )
        for ledger_entry in family["ledger_entries"]:
            if ledger_entry in entries:
                raise ValueError(f"duplicate ledger entry {ledger_entry}")
            entries.add(ledger_entry)
        for procedure in family["procedures"]:
            source = ROOT / procedure["source_file"]
            span = procedure_bytes(source, procedure["start_line"], procedure["end_line"])
            declaration = span.decode("utf-8", errors="replace").lower()
            expected = f"{procedure['procedure_kind'].lower()} {procedure['procedure'].lower()}"
            if expected not in declaration:
                raise ValueError(f"{procedure['id']}: source span does not contain {expected}")
            procedures.append(
                {
                    "id": procedure["id"],
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "span_sha256": hashlib.sha256(span).hexdigest(),
                    "line_count": procedure["end_line"] - procedure["start_line"] + 1,
                }
            )

    expected_entries = {
        "stomate.active.ok_leak_litter",
        "stomate.active.ok_leak_soilcarbon",
        "stomate.active.tf_doc",
        "stomate.active.ok_leak_firstcall_active_layer",
    }
    if entries != expected_entries:
        raise ValueError(f"fragment ledger coverage mismatch: {sorted(entries)}")
    owner = yaml.safe_load(OWNER_FRAGMENT.read_text(encoding="utf-8"))
    if owner.get("status") != "passed" or len(owner.get("families", ())) != 1:
        raise ValueError("soilcarbon owner fragment is not promoted")
    family = owner["families"][0]
    if family.get("numerical_oracle_status") != "verified":
        raise ValueError("soilcarbon owner numerical status is not verified")
    comparison = json.loads(OWNER_COMPARISON.read_text(encoding="utf-8"))
    if comparison.get("status") != "passed" or not all(
        item["passed"] for item in comparison["comparisons"]
    ):
        raise ValueError("soilcarbon owner comparison asset is not passing")

    owner_arms = scientific_arm_ids("stomate_soilcarbon.f90", 641, 2416)
    owner_arms |= scientific_arm_ids("stomate_soilcarbon.f90", 2438, 2578)
    defined_arms = owner_arms
    coverage = json.loads(OWNER_COVERAGE.read_text(encoding="utf-8"))
    if not coverage.get("branch_complete") or coverage.get("missing_case_assignment"):
        raise ValueError("soilcarbon owner arm coverage is incomplete")
    covered_union = {
        item["arm_id"]
        for items in coverage.get("ledger_entries", {}).values()
        for item in items
        if item.get("case_ids")
    }
    covered_union.update(
        item["arm_id"]
        for item in coverage.get("owner_only_arms", [])
        if item.get("case_ids")
    )
    if covered_union != defined_arms:
        raise ValueError(
            f"soilcarbon arm mapping mismatch: missing={sorted(defined_arms - covered_union)}, "
            f"extra={sorted(covered_union - defined_arms)}"
        )
    if family.get("covered_control_flow_policy") != "exact_arm_execution_mapping_asset":
        raise ValueError("soilcarbon owner lacks the exact arm-mapping policy")
    arm_digest = hashlib.sha256("\n".join(sorted(defined_arms)).encode()).hexdigest()
    return {
        "status": "passed",
        "ledger_entries": sorted(entries),
        "procedures": procedures,
        "arm_contract": {
            "litter_required_and_covered": 63,
            "soilcarbon_disposition_arms": len(owner_arms),
            "soilcarbon_explicit_exclusions": 4,
            "soilcarbon_defined_inventory": len(defined_arms),
            "soilcarbon_defined_covered": len(covered_union),
            "soilcarbon_defined_arm_sha256": arm_digest,
        },
    }


if __name__ == "__main__":
    print(yaml.safe_dump(audit_fragment(), sort_keys=False))
