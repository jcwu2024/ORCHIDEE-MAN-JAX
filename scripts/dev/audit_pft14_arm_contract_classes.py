from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIONS = ROOT / "outputs/reference_mode/pft14_arm_equivalence_regions.json"
DEFAULT_DISPOSITION = ROOT / "outputs/reference_mode/pft14_arm_disposition.json"
DEFAULT_POLICY = ROOT / "docs/source_audits/pft14_arm_contract_policy.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
EXCLUDED_SIBLING_DISPOSITIONS = {
    "outside_declared_workflow",
    "fortran_undefined_contract",
    "source_unreachable",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return document


def _policy_indexes(
    policy: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    allowed_classes = set(policy.get("allowed_contract_classes", []))
    allowed_routes = set(policy.get("allowed_evidence_routes", []))
    file_rules: dict[str, dict[str, str]] = {}
    for group in policy.get("source_groups", []):
        contract_class = str(group["contract_class"])
        evidence_route = str(group["evidence_route"])
        if contract_class not in allowed_classes or evidence_route not in allowed_routes:
            raise ValueError(f"{group['id']}: invalid class or evidence route")
        reason = str(group.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"{group['id']}: source group requires a reason")
        for file in group.get("fortran_files", []):
            normalized = str(file).replace("\\", "/")
            if normalized in file_rules:
                raise ValueError(f"duplicate source-group rule for {normalized}")
            file_rules[normalized] = {
                "policy_id": str(group["id"]),
                "contract_class": contract_class,
                "evidence_route": evidence_route,
                "reason": reason,
            }

    procedure_rules: dict[tuple[str, str], dict[str, str]] = {}
    for rule in policy.get("procedure_overrides", []):
        contract_class = str(rule["contract_class"])
        evidence_route = str(rule["evidence_route"])
        if contract_class not in allowed_classes or evidence_route not in allowed_routes:
            raise ValueError(f"{rule['id']}: invalid class or evidence route")
        reason = str(rule.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"{rule['id']}: procedure override requires a reason")
        key = (
            str(rule["fortran_file"]).replace("\\", "/"),
            str(rule["fortran_procedure"]).lower(),
        )
        if key in procedure_rules:
            raise ValueError(f"duplicate procedure override for {key}")
        procedure_rules[key] = {
            "policy_id": str(rule["id"]),
            "contract_class": contract_class,
            "evidence_route": evidence_route,
            "reason": reason,
        }
    return file_rules, procedure_rules


def _source_proof_rules(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rules = {
        str(rule["selector"]): rule for rule in policy.get("source_proof_rules", [])
    }
    required = {
        "automatic_paper_static_active",
        "manual_reason_marker_with_excluded_sibling",
        "manual_selected_with_paper_static_inactive_sibling",
        "equivalence_region_source_reclassification",
    }
    missing = sorted(required - set(rules))
    if missing:
        raise ValueError(f"missing source-proof selectors: {missing}")
    return rules


def _proof_classification(
    arm: dict[str, Any],
    disposition: dict[str, Any],
    siblings: list[dict[str, Any]],
    base_region: dict[str, Any],
    proof_rules: dict[str, dict[str, Any]],
) -> dict[str, str] | None:
    if not base_region["requires_executable_evidence"]:
        rule = proof_rules["equivalence_region_source_reclassification"]
        rationale = base_region.get("source_proof_rationale")
        if not rationale:
            raise ValueError(f"{arm['arm_id']}: source-reclassification lacks rationale")
        return {
            "policy_id": str(rule["id"]),
            "contract_class": str(rule["contract_class"]),
            "evidence_route": str(rule["evidence_route"]),
            "reason": f"{rule['reason']} {rationale}",
        }

    if disposition.get("automatic_disposition") == "paper_static_active":
        rule = proof_rules["automatic_paper_static_active"]
        return {
            "policy_id": str(rule["id"]),
            "contract_class": str(rule["contract_class"]),
            "evidence_route": str(rule["evidence_route"]),
            "reason": f"{rule['reason']} {disposition['reason']}",
        }

    static_siblings = [
        sibling
        for sibling in siblings
        if sibling.get("disposition") == "paper_static_inactive"
    ]
    if static_siblings and disposition.get("classification_source") != "automatic":
        rule = proof_rules["manual_selected_with_paper_static_inactive_sibling"]
        return {
            "policy_id": str(rule["id"]),
            "contract_class": str(rule["contract_class"]),
            "evidence_route": str(rule["evidence_route"]),
            "reason": (
                f"{rule['reason']} Selected arm audit: {disposition['reason']} "
                f"Inactive sibling: {static_siblings[0]['reason']}"
            ),
        }

    rule = proof_rules["manual_reason_marker_with_excluded_sibling"]
    lowered_reason = str(disposition.get("reason") or "").lower()
    marker = next(
        (
            str(value)
            for value in rule.get("reason_markers", [])
            if str(value).lower() in lowered_reason
        ),
        None,
    )
    excluded_siblings = [
        sibling
        for sibling in siblings
        if sibling.get("disposition") in EXCLUDED_SIBLING_DISPOSITIONS
    ]
    if marker is not None and excluded_siblings:
        if disposition.get("classification_source") == "automatic":
            raise ValueError(f"{arm['arm_id']}: manual source-proof marker came from automatic rule")
        return {
            "policy_id": str(rule["id"]),
            "contract_class": str(rule["contract_class"]),
            "evidence_route": str(rule["evidence_route"]),
            "reason": (
                f"{rule['reason']} Arm audit: {disposition['reason']} "
                f"Excluded sibling: {excluded_siblings[0]['reason']}"
            ),
        }
    return None


def build_report(
    equivalence_regions: dict[str, Any],
    arm_disposition: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    if equivalence_regions.get("partition_complete") is not True:
        raise ValueError("equivalence-region input is not a complete partition")
    file_rules, procedure_rules = _policy_indexes(policy)
    proof_rules = _source_proof_rules(policy)
    allowed_classes = set(policy.get("allowed_contract_classes", []))
    allowed_routes = set(policy.get("allowed_evidence_routes", []))

    base_regions = {
        str(region["region_id"]): region for region in equivalence_regions.get("regions", [])
    }
    disposition_by_arm = {
        str(record["arm_id"]): record for record in arm_disposition.get("records", [])
    }
    dispositions_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in arm_disposition.get("records", []):
        dispositions_by_branch[str(record["branch_id"])].append(record)

    classified: list[dict[str, Any]] = []
    errors: list[str] = []
    for assignment in equivalence_regions.get("arm_assignments", []):
        arm_id = str(assignment["arm_id"])
        disposition = disposition_by_arm.get(arm_id)
        base_region = base_regions.get(str(assignment["region_id"]))
        if disposition is None or base_region is None:
            errors.append(f"{arm_id}: missing disposition or base region")
            continue
        siblings = [
            item
            for item in dispositions_by_branch[str(disposition["branch_id"])]
            if item["arm_id"] != arm_id
        ]
        key = (
            str(base_region["fortran_file"]).replace("\\", "/"),
            str(base_region["fortran_procedure"]).lower(),
        )
        owner_classification = procedure_rules.get(key) or file_rules.get(key[0])
        if owner_classification is None:
            errors.append(f"{arm_id}: no explicit source-contract policy for {key}")
            continue
        proof = _proof_classification(
            disposition, disposition, siblings, base_region, proof_rules
        )
        contract_class = owner_classification["contract_class"]
        evidence_route = (
            owner_classification["evidence_route"]
            if proof is None
            else proof["evidence_route"]
        )
        policy_id = owner_classification["policy_id"] if proof is None else proof["policy_id"]
        rationale = owner_classification["reason"] if proof is None else proof["reason"]
        edge_class = None if proof is None else proof["contract_class"]
        if contract_class not in allowed_classes:
            errors.append(f"{arm_id}: invalid contract class")
            continue
        if edge_class is not None and edge_class not in allowed_classes:
            errors.append(f"{arm_id}: invalid edge contract class")
            continue
        if evidence_route not in allowed_routes:
            errors.append(f"{arm_id}: invalid evidence route")
            continue
        classified.append(
            {
                "arm_id": arm_id,
                "branch_id": disposition["branch_id"],
                "base_region_id": base_region["region_id"],
                "fortran_file": base_region["fortran_file"],
                "fortran_procedure": base_region["fortran_procedure"],
                "line": disposition["line"],
                "source": disposition["source"],
                "disposition": disposition["disposition"],
                "disposition_source": disposition["classification_source"],
                "contract_class": contract_class,
                "edge_class": edge_class,
                "evidence_route": evidence_route,
                "owner_evidence_route": owner_classification["evidence_route"],
                "policy_id": policy_id,
                "owner_policy_id": owner_classification["policy_id"],
                "owner_rationale": owner_classification["reason"],
                "rationale": rationale,
                "implemented_source_owner_ids": base_region["implemented_source_owner_ids"],
                "jax_owners": base_region["jax_owners"],
                "closed": False,
            }
        )

    expected_ids = [
        str(assignment["arm_id"])
        for assignment in equivalence_regions.get("arm_assignments", [])
    ]
    classified_ids = [record["arm_id"] for record in classified]
    duplicates = sorted(
        arm_id for arm_id, count in Counter(classified_ids).items() if count != 1
    )
    missing = sorted(set(expected_ids) - set(classified_ids))
    unknown = sorted(set(classified_ids) - set(expected_ids))
    without_rationale = sorted(
        record["arm_id"] for record in classified if not record["rationale"].strip()
    )

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in classified:
        grouped[
            (
                record["base_region_id"],
                record["contract_class"],
                record["evidence_route"],
                record["policy_id"],
                str(record["edge_class"] or ""),
            )
        ].append(record)
    contract_regions: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        digest = hashlib.sha256("\x1f".join(key).encode()).hexdigest()[:12]
        first = members[0]
        contract_regions.append(
            {
                "contract_region_id": f"pft14-contract-{digest}",
                "base_region_id": key[0],
                "contract_class": key[1],
                "evidence_route": key[2],
                "policy_id": key[3],
                "edge_class": key[4] or None,
                "fortran_file": first["fortran_file"],
                "fortran_procedure": first["fortran_procedure"],
                "implemented_source_owner_ids": first["implemented_source_owner_ids"],
                "jax_owners": first["jax_owners"],
                "arm_count": len(members),
                "arm_ids": sorted(member["arm_id"] for member in members),
                "source_lines": sorted({int(member["line"]) for member in members}),
                "rationales": sorted({member["rationale"] for member in members}),
                "closed": False,
            }
        )

    route_arm_counts = Counter(record["evidence_route"] for record in classified)
    route_region_counts = Counter(region["evidence_route"] for region in contract_regions)
    class_arm_counts = Counter(record["contract_class"] for record in classified)
    class_region_counts = Counter(region["contract_class"] for region in contract_regions)
    edge_arm_counts = Counter(
        str(record["edge_class"] or "none") for record in classified
    )
    edge_region_counts = Counter(
        str(region["edge_class"] or "none") for region in contract_regions
    )
    classified_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in classified:
        classified_by_base[record["base_region_id"]].append(record)
    owner_regions: list[dict[str, Any]] = []
    for base_region_id, members in sorted(classified_by_base.items()):
        first = members[0]
        contract_classes = {member["contract_class"] for member in members}
        owner_routes = {member["owner_evidence_route"] for member in members}
        owner_policy_ids = {member["owner_policy_id"] for member in members}
        if len(contract_classes) != 1 or len(owner_routes) != 1 or len(owner_policy_ids) != 1:
            errors.append(f"{base_region_id}: inconsistent owner contract classification")
            continue
        all_inactive = all(member["edge_class"] == "inactive_process" for member in members)
        owner_route = "source_proof" if all_inactive else next(iter(owner_routes))
        owner_reason = (
            "Every arm assigned to this owner region is source-ledger inactive under the "
            "declared PFT14 workflow."
            if all_inactive
            else first["owner_rationale"]
        )
        digest = hashlib.sha256(
            f"{base_region_id}\x1f{first['contract_class']}\x1f{owner_route}".encode()
        ).hexdigest()[:12]
        owner_regions.append(
            {
                "owner_region_id": f"pft14-owner-contract-{digest}",
                "base_region_id": base_region_id,
                "contract_class": first["contract_class"],
                "evidence_route": owner_route,
                "policy_id": next(iter(owner_policy_ids)),
                "fortran_file": first["fortran_file"],
                "fortran_procedure": first["fortran_procedure"],
                "implemented_source_owner_ids": first["implemented_source_owner_ids"],
                "jax_owners": first["jax_owners"],
                "arm_count": len(members),
                "arm_ids": sorted(member["arm_id"] for member in members),
                "rationale": owner_reason,
                "closed": False,
            }
        )
    owner_route_region_counts = Counter(
        region["evidence_route"] for region in owner_regions
    )
    owner_route_arm_counts = Counter()
    for region in owner_regions:
        owner_route_arm_counts[region["evidence_route"]] += int(region["arm_count"])
    complete = (
        not errors
        and not duplicates
        and not missing
        and not unknown
        and not without_rationale
        and len(classified_ids) == len(expected_ids)
    )
    return {
        "schema_version": 1,
        "scope": "evidence-route classification of every unresolved PFT14 owner-only arm",
        "classification_complete": complete,
        "counts": {
            "input_arms": len(expected_ids),
            "classified_arms": len(classified_ids),
            "contract_regions": len(contract_regions),
            "contract_class_arms": dict(sorted(class_arm_counts.items())),
            "contract_class_regions": dict(sorted(class_region_counts.items())),
            "edge_class_arms": dict(sorted(edge_arm_counts.items())),
            "edge_class_regions": dict(sorted(edge_region_counts.items())),
            "evidence_route_arms": dict(sorted(route_arm_counts.items())),
            "evidence_route_regions": dict(sorted(route_region_counts.items())),
            "owner_evidence_route_regions": dict(
                sorted(owner_route_region_counts.items())
            ),
            "owner_evidence_route_arms": dict(sorted(owner_route_arm_counts.items())),
        },
        "classification_errors": {
            "policy_errors": errors,
            "missing_assignments": missing,
            "duplicate_assignments": duplicates,
            "unknown_assignments": unknown,
            "missing_rationale": without_rationale,
        },
        "contract_regions": contract_regions,
        "owner_regions": owner_regions,
        "arm_classifications": sorted(classified, key=lambda record: record["arm_id"]),
        "note": (
            "Edge evidence and owner-region evidence are independent. A static dispatch may "
            "use source proof while its scientific owner still requires a Fortran transition "
            "Oracle. Classification does not close an arm; all records remain closed=false "
            "until Stage 4 evidence assets are generated."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify unresolved PFT14 arms by source contract and evidence route."
    )
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--disposition", type=Path, default=DEFAULT_DISPOSITION)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        json.loads(args.regions.read_text(encoding="utf-8")),
        json.loads(args.disposition.read_text(encoding="utf-8")),
        _load_yaml(args.policy),
    )
    report["inputs"] = {
        name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
        for name, path in {
            "equivalence_regions": args.regions,
            "arm_disposition": args.disposition,
            "contract_policy": args.policy,
        }.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    routes = report["counts"]["evidence_route_arms"]
    print(
        f"WROTE {args.output.resolve()} complete={report['classification_complete']} "
        f"arms={report['counts']['classified_arms']}/{report['counts']['input_arms']} "
        f"regions={report['counts']['contract_regions']} routes={routes}"
    )
    return int(args.require_complete and not report["classification_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
