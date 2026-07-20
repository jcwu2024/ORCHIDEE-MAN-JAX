from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISPOSITION = ROOT / "outputs/reference_mode/pft14_arm_disposition.json"
DEFAULT_ORACLE_COVERAGE = ROOT / "outputs/reference_mode/fortran_oracle_coverage.json"
DEFAULT_PRODUCTION_COVERAGE = ROOT / "outputs/reference_mode/pft14_production_coverage.json"
DEFAULT_CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
DEFAULT_SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
DEFAULT_OWNER_EVIDENCE = ROOT / "outputs/reference_mode/pft14_owner_region_evidence.json"
DEFAULT_FRAGMENT_DIR = ROOT / "docs/source_audits/oracle_families"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/pft14_arm_evidence_gap.json"
SCIENTIFIC_DISPOSITIONS = {"pft14_active", "pft14_conditional", "paper_static_active"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _covered_oracle_arms(coverage: dict[str, Any]) -> tuple[set[str], dict[str, str]]:
    covered: set[str] = set()
    owner_by_arm: dict[str, str] = {}
    for record in coverage.get("records", []):
        if record.get("passed") is not True:
            continue
        entry_id = str(record["ledger_entry"])
        for arm_id in record.get("covered_arm_ids", []):
            arm = str(arm_id)
            covered.add(arm)
            owner_by_arm[arm] = entry_id
    return covered, owner_by_arm


def _passed_production_entries(coverage: dict[str, Any]) -> set[str]:
    return {
        str(record["ledger_entry"])
        for record in coverage.get("records", [])
        if record.get("passed") is True
    }


def load_pending_evidence(fragment_dir: Path) -> list[dict[str, str]]:
    pending: list[dict[str, str]] = []
    for path in sorted(fragment_dir.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        for item in document.get("pending_families", []):
            if not isinstance(item, dict) or not item.get("pending_reason"):
                continue
            if str(item.get("certification_status", "pending")) != "pending":
                continue
            pending.append(
                {
                    "id": str(item.get("id", "unknown")),
                    "source": path.relative_to(ROOT).as_posix(),
                    "certification_status": str(item.get("certification_status", "pending")),
                    "pending_reason": str(item["pending_reason"]),
                }
            )
    return pending


def _arm_contracts(contract_classes: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not contract_classes:
        return {}
    return {
        str(record["arm_id"]): record
        for record in contract_classes.get("arm_classifications", [])
    }


def _passed_source_proofs(source_proofs: dict[str, Any] | None) -> set[str]:
    if not source_proofs:
        return set()
    return {
        str(record["arm_id"])
        for record in source_proofs.get("records", [])
        if record.get("passed") is True
    }


def _passed_owner_arms(
    owner_evidence: dict[str, Any] | None,
) -> tuple[dict[str, str], list[str]]:
    if not owner_evidence:
        return {}, []
    by_arm: dict[str, str] = {}
    errors: list[str] = []
    for record in owner_evidence.get("records", []):
        if record.get("passed") is not True:
            continue
        required = {str(value) for value in record.get("required_arm_ids", [])}
        covered = {str(value) for value in record.get("covered_arm_ids", [])}
        if not required or required != covered:
            errors.append(f"invalid owner record {record.get('owner_region_id')}")
            continue
        route = str(record.get("evidence_route") or "owner_region_evidence")
        for arm_id in covered:
            if arm_id in by_arm:
                errors.append(f"duplicate owner evidence for {arm_id}")
            else:
                by_arm[arm_id] = route
    return by_arm, errors


def build_report(
    disposition: dict[str, Any],
    oracle_coverage: dict[str, Any],
    production_coverage: dict[str, Any],
    *,
    pending_evidence: list[dict[str, str]] | None = None,
    contract_classes: dict[str, Any] | None = None,
    source_proofs: dict[str, Any] | None = None,
    owner_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    oracle_arms, oracle_owner_by_arm = _covered_oracle_arms(oracle_coverage)
    production_entries = _passed_production_entries(production_coverage)
    contracts = _arm_contracts(contract_classes)
    passed_source_proofs = _passed_source_proofs(source_proofs)
    passed_owner_arms, owner_evidence_errors = _passed_owner_arms(owner_evidence)
    scientific = [
        record
        for record in disposition.get("records", [])
        if record.get("disposition") in SCIENTIFIC_DISPOSITIONS
    ]
    records: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    status_counts: Counter[str] = Counter()

    for source in scientific:
        arm_id = str(source["arm_id"])
        oracle_entry = oracle_owner_by_arm.get(arm_id)
        source_owner_present = source.get("scientific_source_owner_present") is True
        if oracle_entry is not None and oracle_entry in production_entries:
            status = "fortran_oracle_and_production"
        elif oracle_entry is not None:
            status = "fortran_oracle_only"
        elif arm_id in contracts and contracts[arm_id].get("evidence_route") == "source_proof":
            status = "source_proof" if arm_id in passed_source_proofs else "missing_source_proof"
        elif arm_id in contracts and arm_id in passed_owner_arms:
            route = str(
                contracts[arm_id].get("owner_evidence_route")
                or contracts[arm_id].get("evidence_route")
            )
            status = {
                "fortran_transition_oracle_and_production": "owner_transition_and_production",
                "fortran_state_io_oracle_and_lifecycle": "owner_state_io_and_lifecycle",
                "output_acceptance_and_source_contract": "owner_output_acceptance",
            }.get(route, "owner_required_evidence")
        elif source_owner_present:
            status = "source_owner_only"
        else:
            status = "missing_source_owner"
        status_counts[status] += 1
        file = str(source.get("file", "unknown"))
        procedure = str(source.get("procedure") or "unknown")
        groups[(status, file, procedure)].append(arm_id)
        records.append(
            {
                "arm_id": arm_id,
                "branch_id": source.get("branch_id"),
                "arm": source.get("arm"),
                "file": file,
                "procedure": procedure,
                "line": source.get("line"),
                "source": source.get("source"),
                "disposition": source.get("disposition"),
                "evidence_status": status,
                "source_owner_present": source_owner_present,
                "oracle_ledger_entry": oracle_entry,
                "production_entry_passed": bool(
                    oracle_entry is not None and oracle_entry in production_entries
                ),
                "contract_evidence_route": contracts.get(arm_id, {}).get("evidence_route"),
                "owner_evidence_route": contracts.get(arm_id, {}).get("owner_evidence_route"),
                "owner_region_evidence_passed": arm_id in passed_owner_arms,
                "source_proof_passed": arm_id in passed_source_proofs,
            }
        )

    group_rows = [
        {
            "evidence_status": status,
            "file": file,
            "procedure": procedure,
            "arm_count": len(arm_ids),
            "arm_ids": sorted(arm_ids),
        }
        for (status, file, procedure), arm_ids in groups.items()
    ]
    group_rows.sort(
        key=lambda row: (
            row["evidence_status"],
            -int(row["arm_count"]),
            row["file"],
            row["procedure"],
        )
    )
    pending = list(pending_evidence or [])
    unresolved = (
        status_counts["source_owner_only"]
        + status_counts["fortran_oracle_only"]
        + status_counts["missing_source_owner"]
        + status_counts["missing_source_proof"]
    )
    disposition_arms = {str(item["arm_id"]) for item in disposition.get("records", [])}
    scientific_arms = {str(item["arm_id"]) for item in scientific}
    unknown_oracle_arms = sorted(oracle_arms - disposition_arms)
    non_scientific_oracle_arms = sorted((oracle_arms & disposition_arms) - scientific_arms)
    complete = (
        bool(scientific)
        and unresolved == 0
        and not pending
        and not unknown_oracle_arms
        and not owner_evidence_errors
    )
    return {
        "schema_version": 1,
        "scope": "all source-classified PFT14 scientific reachable control-flow arms",
        "complete": complete,
        "counts": {
            "scientific_reachable_arms": len(scientific),
            "fortran_oracle_and_production": status_counts[
                "fortran_oracle_and_production"
            ],
            "fortran_oracle_only": status_counts["fortran_oracle_only"],
            "owner_transition_and_production": status_counts["owner_transition_and_production"],
            "owner_state_io_and_lifecycle": status_counts["owner_state_io_and_lifecycle"],
            "owner_output_acceptance": status_counts["owner_output_acceptance"],
            "source_proof": status_counts["source_proof"],
            "missing_source_proof": status_counts["missing_source_proof"],
            "contract_routed_arms": sum(
                status_counts[name]
                for name in (
                    "owner_transition_and_production",
                    "owner_state_io_and_lifecycle",
                    "owner_output_acceptance",
                    "source_proof",
                    "missing_source_proof",
                    "owner_required_evidence",
                    "source_owner_only",
                )
            ),
            "source_owner_only": status_counts["source_owner_only"],
            "missing_source_owner": status_counts["missing_source_owner"],
            "unresolved_executable_evidence": unresolved,
            "pending_evidence_items": len(pending),
            "unknown_oracle_arms": len(unknown_oracle_arms),
            "non_scientific_oracle_arms": len(non_scientific_oracle_arms),
        },
        "pending_evidence": pending,
        "unknown_oracle_arm_ids": unknown_oracle_arms,
        "non_scientific_oracle_arm_ids": non_scientific_oracle_arms,
        "owner_evidence_errors": owner_evidence_errors,
        "groups": group_rows,
        "records": sorted(records, key=lambda item: str(item["arm_id"])),
        "note": (
            "A scientific source owner is process attribution only. Each scientific arm must "
            "have its contract-selected evidence route: a ledger Fortran oracle plus production, "
            "a passing owner-region transition/lifecycle record, or a pinned source proof."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit executable evidence for every PFT14 scientific reachable arm."
    )
    parser.add_argument("--disposition", type=Path, default=DEFAULT_DISPOSITION)
    parser.add_argument("--oracle-coverage", type=Path, default=DEFAULT_ORACLE_COVERAGE)
    parser.add_argument(
        "--production-coverage", type=Path, default=DEFAULT_PRODUCTION_COVERAGE
    )
    parser.add_argument("--contract-classes", type=Path, default=DEFAULT_CONTRACT_CLASSES)
    parser.add_argument("--source-proofs", type=Path, default=DEFAULT_SOURCE_PROOFS)
    parser.add_argument("--owner-evidence", type=Path, default=DEFAULT_OWNER_EVIDENCE)
    parser.add_argument("--fragment-dir", type=Path, default=DEFAULT_FRAGMENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        json.loads(args.disposition.read_text(encoding="utf-8")),
        json.loads(args.oracle_coverage.read_text(encoding="utf-8")),
        json.loads(args.production_coverage.read_text(encoding="utf-8")),
        pending_evidence=load_pending_evidence(args.fragment_dir),
        contract_classes=json.loads(args.contract_classes.read_text(encoding="utf-8")),
        source_proofs=json.loads(args.source_proofs.read_text(encoding="utf-8")),
        owner_evidence=json.loads(args.owner_evidence.read_text(encoding="utf-8")),
    )
    report["inputs"] = {
        "disposition": {
            "path": args.disposition.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.disposition),
        },
        "oracle_coverage": {
            "path": args.oracle_coverage.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.oracle_coverage),
        },
        "production_coverage": {
            "path": args.production_coverage.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.production_coverage),
        },
        "contract_classes": {
            "path": args.contract_classes.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.contract_classes),
        },
        "source_proofs": {
            "path": args.source_proofs.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.source_proofs),
        },
        "owner_evidence": {
            "path": args.owner_evidence.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.owner_evidence),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} complete={report['complete']} "
        f"covered={report['counts']['scientific_reachable_arms'] - report['counts']['unresolved_executable_evidence']}/"
        f"{report['counts']['scientific_reachable_arms']} "
        f"unresolved={report['counts']['unresolved_executable_evidence']} "
        f"pending={report['counts']['pending_evidence_items']}"
    )
    return int(args.require_complete and not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
