from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
DEFAULT_SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
DEFAULT_EVIDENCE_ROOT = ROOT / "outputs/reference_mode/micro_oracles"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/pft14_owner_region_evidence.json"


def build_report(
    contract_classes: dict[str, Any],
    source_proofs: dict[str, Any],
    evidence_documents: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    required = {
        str(record["owner_region_id"]): record
        for record in contract_classes.get("owner_regions", [])
    }
    passed_source_arms = {
        str(record["arm_id"])
        for record in source_proofs.get("records", [])
        if record.get("passed") is True
    }
    evidence_by_region: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    invalid_records: list[str] = []
    for path, document in evidence_documents:
        if document.get("complete") is not True:
            invalid_records.append(f"{path}: family evidence is incomplete")
        for record in document.get("records", []):
            region_id = str(record.get("owner_region_id"))
            if region_id in evidence_by_region:
                duplicates.append(region_id)
                continue
            required_arms = {str(value) for value in record.get("required_arm_ids", [])}
            covered_arms = {str(value) for value in record.get("covered_arm_ids", [])}
            if record.get("passed") is not True or not required_arms or required_arms != covered_arms:
                invalid_records.append(f"{path}: invalid owner evidence for {region_id}")
                continue
            evidence_by_region[region_id] = {**record, "evidence_asset": path}

    source_owner_records: dict[str, dict[str, Any]] = {}
    for region_id, owner in required.items():
        if owner.get("evidence_route") != "source_proof":
            continue
        arm_ids = {str(value) for value in owner.get("arm_ids", [])}
        missing = arm_ids - passed_source_arms
        if missing:
            invalid_records.append(
                f"source-proof owner {region_id} lacks arm proofs: {sorted(missing)}"
            )
            continue
        source_owner_records[region_id] = {
            "owner_region_id": region_id,
            "base_region_id": owner["base_region_id"],
            "fortran_procedure": owner["fortran_procedure"],
            "required_arm_ids": sorted(arm_ids),
            "covered_arm_ids": sorted(arm_ids),
            "evidence_asset": "outputs/reference_mode/pft14_source_proof_evidence.json",
            "passed": True,
        }

    overlap = sorted(set(evidence_by_region) & set(source_owner_records))
    duplicates.extend(overlap)
    all_evidence = {**source_owner_records, **evidence_by_region}
    unknown = sorted(set(all_evidence) - set(required))
    passed_ids = set(all_evidence) & set(required)
    missing_ids = sorted(set(required) - passed_ids)
    route_required = Counter(str(record["evidence_route"]) for record in required.values())
    route_passed = Counter(
        str(required[region_id]["evidence_route"])
        for region_id in passed_ids
        if region_id in required
    )
    complete = (
        bool(required)
        and not duplicates
        and not invalid_records
        and not unknown
        and not missing_ids
    )
    return {
        "schema_version": 1,
        "scope": "owner-region executable/source evidence for PFT14 equivalence",
        "complete": complete,
        "counts": {
            "required_owner_regions": len(required),
            "passed_owner_regions": len(passed_ids),
            "missing_owner_regions": len(missing_ids),
            "required_by_route": dict(sorted(route_required.items())),
            "passed_by_route": dict(sorted(route_passed.items())),
        },
        "errors": {
            "duplicate_owner_regions": sorted(set(duplicates)),
            "invalid_evidence_records": invalid_records,
            "unknown_owner_regions": unknown,
        },
        "missing_owner_region_ids": missing_ids,
        "records": [all_evidence[region_id] for region_id in sorted(passed_ids)],
        "note": (
            "A region passes only when all of its classified arms are covered. Source-proof "
            "owner regions are derived from the pinned source-proof asset; numerical/state "
            "regions require a family owner_region_evidence.json asset."
        ),
    }


def load_evidence_documents(root: Path) -> list[tuple[str, dict[str, Any]]]:
    documents: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.glob("*/owner_region_evidence.json")):
        documents.append(
            (
                path.relative_to(ROOT).as_posix(),
                json.loads(path.read_text(encoding="utf-8")),
            )
        )
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit PFT14 owner-region evidence closure.")
    parser.add_argument("--contract-classes", type=Path, default=DEFAULT_CONTRACT_CLASSES)
    parser.add_argument("--source-proofs", type=Path, default=DEFAULT_SOURCE_PROOFS)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        json.loads(args.contract_classes.read_text(encoding="utf-8")),
        json.loads(args.source_proofs.read_text(encoding="utf-8")),
        load_evidence_documents(args.evidence_root),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} complete={report['complete']} "
        f"passed={report['counts']['passed_owner_regions']}/"
        f"{report['counts']['required_owner_regions']} "
        f"routes={report['counts']['passed_by_route']}"
    )
    return int(args.require_complete and not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
