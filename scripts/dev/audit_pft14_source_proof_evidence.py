from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
DEFAULT_DISPOSITION = ROOT / "outputs/reference_mode/pft14_arm_disposition.json"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
EXCLUDED_DISPOSITIONS = {
    "outside_declared_workflow",
    "fortran_undefined_contract",
    "source_unreachable",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def build_report(
    contract_classes: dict[str, Any],
    arm_disposition: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if contract_classes.get("classification_complete") is not True:
        raise ValueError("arm contract classification is incomplete")
    dispositions = {
        str(record["arm_id"]): record for record in arm_disposition.get("records", [])
    }
    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in arm_disposition.get("records", []):
        by_branch[str(record["branch_id"])].append(record)

    expected = [
        record
        for record in contract_classes.get("arm_classifications", [])
        if record.get("evidence_route") == "source_proof"
    ]
    source_cache: dict[str, tuple[list[bytes], str]] = {}
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for contract in expected:
        arm_id = str(contract["arm_id"])
        disposition = dispositions.get(arm_id)
        if disposition is None:
            errors.append(f"{arm_id}: missing arm disposition")
            continue
        source_name = str(contract["fortran_file"]).replace("\\", "/")
        source_path = root / source_name
        if not source_path.is_file():
            errors.append(f"{arm_id}: missing Fortran source {source_name}")
            continue
        if source_name not in source_cache:
            raw = source_path.read_bytes()
            source_cache[source_name] = (raw.splitlines(), _sha256_bytes(raw))
        lines, source_sha = source_cache[source_name]
        line_number = int(contract["line"])
        if not 1 <= line_number <= len(lines):
            errors.append(f"{arm_id}: source line is outside file")
            continue
        physical_line = lines[line_number - 1]
        edge_class = contract.get("edge_class")
        policy_id = str(contract["policy_id"])
        proof_checks: list[dict[str, Any]] = []
        proof_checks.append(
            {
                "name": "disposition_identity",
                "passed": (
                    disposition.get("branch_id") == contract.get("branch_id")
                    and int(disposition.get("line")) == line_number
                    and disposition.get("source") == contract.get("source")
                ),
            }
        )
        if edge_class == "paper_static_dispatch":
            if policy_id == "audited_fixed_source_dispatch":
                siblings = [
                    sibling
                    for sibling in by_branch[str(contract["branch_id"])]
                    if sibling["arm_id"] != arm_id
                ]
                proof_checks.extend(
                    [
                        {
                            "name": "manual_static_source_backing",
                            "passed": disposition.get("classification_source")
                            != "automatic",
                        },
                        {
                            "name": "paper_static_inactive_sibling",
                            "passed": any(
                                sibling.get("disposition") == "paper_static_inactive"
                                for sibling in siblings
                            ),
                        },
                    ]
                )
            else:
                proof_checks.append(
                    {
                        "name": "paper_static_selection",
                        "passed": disposition.get("automatic_disposition")
                        == "paper_static_active",
                    }
                )
        elif edge_class == "valid_input_continuation":
            siblings = [
                sibling
                for sibling in by_branch[str(contract["branch_id"])]
                if sibling["arm_id"] != arm_id
            ]
            proof_checks.extend(
                [
                    {
                        "name": "manual_source_backing",
                        "passed": disposition.get("classification_source") != "automatic",
                    },
                    {
                        "name": "excluded_error_sibling",
                        "passed": any(
                            sibling.get("disposition") in EXCLUDED_DISPOSITIONS
                            for sibling in siblings
                        ),
                    },
                ]
            )
        elif edge_class == "inactive_process":
            proof_checks.append(
                {
                    "name": "inactive_ledger_rationale",
                    "passed": policy_id == "source_ledger_inactive_process"
                    and bool(str(contract.get("rationale") or "").strip()),
                }
            )
        elif contract.get("contract_class") == "runtime_infrastructure":
            proof_checks.append(
                {
                    "name": "runtime_infrastructure_policy",
                    "passed": policy_id
                    in {
                        "mpi_single_process_runtime",
                        "netcdf_status_guard",
                        "print_level_configuration",
                        "fixed_paper_dispatch",
                    },
                }
            )
        else:
            errors.append(f"{arm_id}: unsupported source-proof class")
            continue
        proof_checks.extend(
            [
                {
                    "name": "fortran_source_line_present",
                    "passed": bool(physical_line.strip()),
                },
                {
                    "name": "rationale_present",
                    "passed": bool(str(contract.get("rationale") or "").strip()),
                },
            ]
        )
        passed = all(check["passed"] for check in proof_checks)
        if not passed:
            errors.append(
                f"{arm_id}: failed checks "
                + ", ".join(check["name"] for check in proof_checks if not check["passed"])
            )
        records.append(
            {
                "arm_id": arm_id,
                "branch_id": contract["branch_id"],
                "contract_class": contract["contract_class"],
                "edge_class": edge_class,
                "policy_id": policy_id,
                "fortran_file": source_name,
                "fortran_line": line_number,
                "fortran_source_sha256": source_sha,
                "fortran_physical_line_sha256": _sha256_bytes(physical_line),
                "disposition_source": disposition["classification_source"],
                "rationale": contract["rationale"],
                "checks": proof_checks,
                "passed": passed,
            }
        )

    expected_ids = [str(record["arm_id"]) for record in expected]
    actual_ids = [str(record["arm_id"]) for record in records]
    duplicates = sorted(
        arm_id for arm_id, count in Counter(actual_ids).items() if count != 1
    )
    missing = sorted(set(expected_ids) - set(actual_ids))
    unknown = sorted(set(actual_ids) - set(expected_ids))
    complete = (
        not errors
        and not duplicates
        and not missing
        and not unknown
        and len(actual_ids) == len(expected_ids)
        and all(record["passed"] for record in records)
    )
    return {
        "schema_version": 1,
        "scope": "source-backed evidence for PFT14 arms routed to source_proof",
        "complete": complete,
        "counts": {
            "expected_source_proof_arms": len(expected_ids),
            "passed_source_proof_arms": sum(record["passed"] for record in records),
            "edge_classes": dict(
                sorted(Counter(str(record["edge_class"] or "none") for record in records).items())
            ),
            "contract_classes": dict(
                sorted(Counter(record["contract_class"] for record in records).items())
            ),
            "source_files": len(source_cache),
        },
        "errors": {
            "proof_errors": errors,
            "missing_arms": missing,
            "duplicate_arms": duplicates,
            "unknown_arms": unknown,
        },
        "records": sorted(records, key=lambda record: record["arm_id"]),
        "note": (
            "Source proof closes only the classified branch edge. The enclosing owner-region "
            "Fortran transition/state/output evidence remains independently required."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit PFT14 source-proof edge evidence.")
    parser.add_argument("--contract-classes", type=Path, default=DEFAULT_CONTRACT_CLASSES)
    parser.add_argument("--disposition", type=Path, default=DEFAULT_DISPOSITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        json.loads(args.contract_classes.read_text(encoding="utf-8")),
        json.loads(args.disposition.read_text(encoding="utf-8")),
    )
    report["inputs"] = {
        "contract_classes": {
            "path": args.contract_classes.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.contract_classes),
        },
        "arm_disposition": {
            "path": args.disposition.relative_to(ROOT).as_posix(),
            "sha256": _sha256(args.disposition),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} complete={report['complete']} "
        f"passed={report['counts']['passed_source_proof_arms']}/"
        f"{report['counts']['expected_source_proof_arms']}"
    )
    return int(args.require_complete and not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
