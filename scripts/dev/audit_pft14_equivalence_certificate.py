from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "outputs" / "reference_mode" / "fortran_control_flow_inventory.json"
DEFAULT_LEDGER = ROOT / "docs" / "source_audits" / "pft14_reachable_ledger.yaml"
DEFAULT_EVIDENCE_SCOPE = ROOT / "docs" / "source_audits" / "pft14_evidence_scope.yaml"
DEFAULT_STATE_AUDIT = ROOT / "outputs" / "reference_mode" / "pft14_state_transition_ledger.json"
DEFAULT_SECHIBA_STATE_AUDIT = ROOT / "outputs" / "reference_mode" / "sechiba_state_transition_contract.json"
DEFAULT_RESTART_ALIAS_AUDIT = ROOT / "outputs" / "reference_mode" / "restart_alias_roundtrip.json"
DEFAULT_RESTART_LIFECYCLE_AUDIT = ROOT / "outputs" / "reference_mode" / "restart_state_lifecycle.json"
DEFAULT_STAGE5_LIFECYCLE_ACCEPTANCE = (
    ROOT / "outputs" / "reference_mode" / "stage5_lifecycle_acceptance.json"
)
DEFAULT_REPRESENTATIVE_ACCEPTANCE = (
    ROOT / "outputs" / "acceptance" / "stage6_representative_20260716" / "summary.json"
)
DEFAULT_ARM_DISPOSITION = ROOT / "outputs" / "reference_mode" / "pft14_arm_disposition.json"
DEFAULT_ARM_EVIDENCE = ROOT / "outputs" / "reference_mode" / "pft14_arm_evidence_gap.json"
DEFAULT_ARM_CONTRACT_CLASSES = (
    ROOT / "outputs" / "reference_mode" / "pft14_arm_contract_classes.json"
)
DEFAULT_SOURCE_PROOF_EVIDENCE = (
    ROOT / "outputs" / "reference_mode" / "pft14_source_proof_evidence.json"
)
DEFAULT_OWNER_REGION_EVIDENCE = (
    ROOT / "outputs" / "reference_mode" / "pft14_owner_region_evidence.json"
)
DEFAULT_MICRO_ORACLE_INDEX = ROOT / "outputs" / "reference_mode" / "micro_oracles" / "index.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "pft14_equivalence_certificate.json"
ORACLE_LEVELS = {"fortran_oracle", "production_verified"}
GENERAL_ORACLE_KINDS = {"fortran_micro_oracle", "production_verified"}
SCIENTIFIC_ARM_DISPOSITIONS = {"pft14_active", "pft14_conditional", "paper_static_active"}


def _line_numbers(value: str | int) -> set[int]:
    result: set[int] = set()
    for item in str(value).split(","):
        bounds = [int(part) for part in item.strip().split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _process_span_index(entries: list[dict[str, Any]]) -> dict[tuple[str, str], set[int]]:
    index: dict[tuple[str, str], set[int]] = {}
    for entry in entries:
        file = entry.get("fortran_file")
        procedure = entry.get("fortran_subroutine")
        lines = entry.get("fortran_lines")
        if not file or not procedure or not lines:
            continue
        index.setdefault((str(file).replace("\\", "/"), str(procedure).lower()), set()).update(
            _line_numbers(lines)
        )
    return index


def build_certificate(
    inventory: dict[str, Any],
    ledger: dict[str, Any],
    *,
    evidence_scope: dict[str, Any] | None = None,
    state_audit: dict[str, Any] | None = None,
    sechiba_state_audit: dict[str, Any] | None = None,
    restart_alias_audit: dict[str, Any] | None = None,
    restart_lifecycle_audit: dict[str, Any] | None = None,
    stage5_lifecycle_acceptance: dict[str, Any] | None = None,
    representative_acceptance: dict[str, Any] | None = None,
    arm_disposition: dict[str, Any] | None = None,
    arm_evidence: dict[str, Any] | None = None,
    arm_contract_classes: dict[str, Any] | None = None,
    source_proof_evidence: dict[str, Any] | None = None,
    owner_region_evidence: dict[str, Any] | None = None,
    micro_oracle_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entries = ledger.get("entries", [])
    branches_by_id = {branch["id"]: branch for branch in inventory.get("branches", [])}
    procedures_by_id = {
        procedure["id"]: procedure for procedure in inventory.get("procedures", [])
    }
    reachable_ids = set(inventory.get("reachable_branch_ids", []))
    arms_by_id = {arm["id"]: arm for arm in inventory.get("control_flow_arms", [])}
    reachable_arm_ids = set(inventory.get("target_reachable_arm_ids", []))
    process_spans = _process_span_index(entries)
    span_mapped_ids: set[str] = set()
    for branch_id in reachable_ids:
        branch = branches_by_id[branch_id]
        procedure = procedures_by_id.get(branch.get("procedure_id"))
        if procedure is None:
            continue
        key = (branch["file"], procedure["name"].lower())
        if branch["line"] in process_spans.get(key, set()):
            span_mapped_ids.add(branch_id)

    explicitly_classified: dict[str, str] = {}
    duplicate_classifications: list[str] = []
    for entry in entries:
        for branch_id in entry.get("control_flow_ids", []):
            if branch_id in explicitly_classified:
                duplicate_classifications.append(branch_id)
            explicitly_classified[branch_id] = str(entry.get("id"))
    classified_reachable = reachable_ids.intersection(explicitly_classified)
    unknown_classified = sorted(set(explicitly_classified) - set(branches_by_id))

    explicitly_classified_arms: dict[str, str] = {}
    duplicate_arm_classifications: list[str] = []
    for entry in entries:
        for arm_id in entry.get("control_flow_arm_ids", []):
            if arm_id in explicitly_classified_arms:
                duplicate_arm_classifications.append(arm_id)
            explicitly_classified_arms[arm_id] = str(entry.get("id"))
    generated_arm_status_counts = Counter()
    generated_open_arm_ids: list[str] = []
    generated_candidate_arm_ids: list[str] = []
    scientific_arm_ids: list[str] = []
    scientific_arms_missing_source_owner: list[str] = []
    for record in (arm_disposition or {}).get("records", []):
        arm_id = record.get("arm_id")
        status = str(record.get("status"))
        generated_arm_status_counts[status] += 1
        if record.get("disposition") in SCIENTIFIC_ARM_DISPOSITIONS and isinstance(arm_id, str):
            scientific_arm_ids.append(arm_id)
            if not record.get("scientific_source_owner_present"):
                scientific_arms_missing_source_owner.append(arm_id)
        if status == "classified" and isinstance(arm_id, str):
            if arm_id in explicitly_classified_arms:
                duplicate_arm_classifications.append(arm_id)
            explicitly_classified_arms[arm_id] = f"generated:{record.get('disposition')}"
        elif status == "candidate" and isinstance(arm_id, str):
            generated_candidate_arm_ids.append(arm_id)
        elif status == "open" and isinstance(arm_id, str):
            generated_open_arm_ids.append(arm_id)
    classified_reachable_arms = reachable_arm_ids.intersection(explicitly_classified_arms)
    unknown_classified_arms = sorted(set(explicitly_classified_arms) - set(arms_by_id))
    reachable_arms_by_branch: dict[str, set[str]] = {}
    for arm_id in reachable_arm_ids:
        branch_id = arms_by_id.get(arm_id, {}).get("branch_id")
        if isinstance(branch_id, str):
            reachable_arms_by_branch.setdefault(branch_id, set()).add(arm_id)
    arm_complete_branch_ids = {
        branch_id
        for branch_id, arm_ids in reachable_arms_by_branch.items()
        if arm_ids and arm_ids.issubset(classified_reachable_arms)
    }
    effectively_classified_branches = classified_reachable.union(arm_complete_branch_ids)
    span_mapped_arm_ids = {
        arm_id
        for arm_id in reachable_arm_ids
        if arms_by_id.get(arm_id, {}).get("branch_id") in span_mapped_ids
    }

    reachable_entries = [
        entry for entry in entries if entry.get("reachability") in {"active", "conditional"}
    ]
    scoped_evidence = (evidence_scope or {}).get("entries", {})
    evidence_kind = {
        entry["id"]: scoped_evidence.get(entry["id"], {}).get(
            "kind", entry.get("evidence_level")
        )
        for entry in reachable_entries
    }
    oracle_entries = [
        entry for entry in reachable_entries if evidence_kind[entry["id"]] in GENERAL_ORACLE_KINDS
    ]
    production_entries = [
        entry for entry in reachable_entries if evidence_kind[entry["id"]] == "production_verified"
    ]
    point_trace_entries = [
        entry for entry in reachable_entries if evidence_kind[entry["id"]] == "point_trace_anchor"
    ]
    unmapped = reachable_ids - span_mapped_ids
    unmapped_by_file = Counter(branches_by_id[branch_id]["file"] for branch_id in unmapped)
    unresolved = inventory.get("reachable_unresolved_project_callee_names", [])
    call_graph_complete = not inventory.get("call_graph_limitations")

    gates = {
        "control_flow_fully_classified": (
            len(effectively_classified_branches) == len(reachable_ids)
            and bool(reachable_arm_ids)
            and len(classified_reachable_arms) == len(reachable_arm_ids)
            and not unknown_classified
            and not unknown_classified_arms
            and not duplicate_classifications
            and not duplicate_arm_classifications
            and not unresolved
            and call_graph_complete
        ),
        "all_scientific_arms_source_mapped": (
            bool(scientific_arm_ids) and not scientific_arms_missing_source_owner
        ),
        "all_scientific_arms_have_executable_evidence": bool(
            arm_evidence
            and arm_evidence.get("complete") is True
            and arm_evidence.get("counts", {}).get("scientific_reachable_arms")
            == len(scientific_arm_ids)
            and arm_evidence.get("counts", {}).get("unresolved_executable_evidence") == 0
            and arm_evidence.get("counts", {}).get("pending_evidence_items") == 0
        ),
        "arm_contract_evidence_routes_fully_classified": bool(
            arm_contract_classes
            and arm_contract_classes.get("classification_complete") is True
            and arm_contract_classes.get("counts", {}).get("input_arms")
            == arm_contract_classes.get("counts", {}).get("classified_arms")
            and arm_contract_classes.get("counts", {}).get("input_arms")
            == (arm_evidence or {}).get("counts", {}).get(
                "contract_routed_arms",
                (arm_evidence or {}).get("counts", {}).get("source_owner_only"),
            )
        ),
        "source_proof_edges_closed": bool(
            source_proof_evidence
            and source_proof_evidence.get("complete") is True
            and source_proof_evidence.get("counts", {}).get("expected_source_proof_arms")
            == source_proof_evidence.get("counts", {}).get("passed_source_proof_arms")
            and source_proof_evidence.get("counts", {}).get("expected_source_proof_arms")
            == (arm_contract_classes or {})
            .get("counts", {})
            .get("evidence_route_arms", {})
            .get("source_proof")
        ),
        "all_owner_regions_have_required_evidence": bool(
            owner_region_evidence
            and owner_region_evidence.get("complete") is True
            and owner_region_evidence.get("counts", {}).get("required_owner_regions")
            == owner_region_evidence.get("counts", {}).get("passed_owner_regions")
            and owner_region_evidence.get("counts", {}).get("required_owner_regions")
            == len((arm_contract_classes or {}).get("owner_regions", []))
        ),
        "all_reachable_entries_have_fortran_oracle": len(oracle_entries) == len(reachable_entries),
        "all_reachable_entries_production_verified": len(production_entries) == len(reachable_entries),
        "state_transition_ledger_closed": bool(
            state_audit
            and state_audit.get("all_components_closed")
            and sechiba_state_audit
            and sechiba_state_audit.get("all_components_closed")
            and restart_alias_audit
            and restart_alias_audit.get("closed")
            and restart_lifecycle_audit
            and restart_lifecycle_audit.get("closed")
        ),
        "cold_start_restart_year_handoff_pass": bool(
            stage5_lifecycle_acceptance
            and stage5_lifecycle_acceptance.get("status") == "passed"
            and stage5_lifecycle_acceptance.get("production_execution") is True
            and stage5_lifecycle_acceptance.get("checks")
            and all(
                check.get("passed") is True
                for check in stage5_lifecycle_acceptance["checks"]
            )
        ),
        "representative_paper_scientific_acceptance_pass": bool(
            representative_acceptance
            and representative_acceptance.get("initial_state") == "cold-start"
            and representative_acceptance.get("landpoint_count", 0) > 0
            and representative_acceptance.get("scientific_acceptance_count")
            == representative_acceptance.get("landpoint_count")
            and representative_acceptance.get("results")
            and all(
                result.get("scientific_acceptance_pass") is True
                for result in representative_acceptance["results"]
            )
        ),
        "paper_669_scientific_acceptance_pass": False,
    }
    gates["equivalent"] = all(gates.values())
    extracted_oracles = (micro_oracle_index or {}).get("oracles", [])
    numerical_oracle_status_counts = Counter(
        str(record.get("numerical_oracle_status")) for record in extracted_oracles
    )
    return {
        "schema_version": 1,
        "claim": "PFT14 Fortran-equivalent JAX production implementation",
        "equivalent": gates["equivalent"],
        "gates": gates,
        "control_flow": {
            "reachable_branches": len(reachable_ids),
            "explicitly_classified_reachable_branches": len(classified_reachable),
            "arm_complete_reachable_branches": len(arm_complete_branch_ids),
            "effectively_classified_reachable_branches": len(effectively_classified_branches),
            "process_span_mapped_reachable_branches": len(span_mapped_ids),
            "process_span_unmapped_reachable_branches": len(unmapped),
            "reachable_unresolved_callee_names": unresolved,
            "external_boundary_calls": inventory.get("reachable_external_calls", {}),
            "call_graph_complete": call_graph_complete,
            "call_graph_limitations": inventory.get("call_graph_limitations", []),
            "unknown_classified_ids": unknown_classified,
            "duplicate_classification_ids": sorted(set(duplicate_classifications)),
            "reachable_arms": len(reachable_arm_ids),
            "explicitly_classified_reachable_arms": len(classified_reachable_arms),
            "process_span_mapped_reachable_arms": len(span_mapped_arm_ids),
            "unknown_classified_arm_ids": unknown_classified_arms,
            "duplicate_arm_classification_ids": sorted(set(duplicate_arm_classifications)),
            "arm_disposition_status_counts": dict(generated_arm_status_counts),
            "candidate_arm_ids": generated_candidate_arm_ids,
            "open_arm_ids": generated_open_arm_ids,
            "scientific_reachable_arms": len(scientific_arm_ids),
            "scientific_arms_missing_source_owner": scientific_arms_missing_source_owner,
            "largest_unmapped_files": [
                {"file": file, "branches": count}
                for file, count in unmapped_by_file.most_common(25)
            ],
            "note": (
                "Process-span mapping is diagnostic only. Equivalence requires every stable "
                "control-flow site and every reachable arm ID to be explicitly classified."
            ),
        },
        "evidence": {
            "reachable_ledger_entries": len(reachable_entries),
            "fortran_oracle_entries": len(oracle_entries),
            "production_verified_entries": len(production_entries),
            "point_trace_anchor_entries": len(point_trace_entries),
            "evidence_kind_counts": dict(Counter(evidence_kind.values())),
            "entries_missing_fortran_oracle": [
                entry.get("id") for entry in reachable_entries if entry not in oracle_entries
            ],
            "scientific_arm_executable_evidence": (
                {}
                if arm_evidence is None
                else {
                    "complete": arm_evidence.get("complete"),
                    "counts": arm_evidence.get("counts", {}),
                    "pending_evidence": arm_evidence.get("pending_evidence", []),
                }
            ),
            "arm_contract_evidence_routes": (
                {}
                if arm_contract_classes is None
                else {
                    "classification_complete": arm_contract_classes.get(
                        "classification_complete"
                    ),
                    "counts": arm_contract_classes.get("counts", {}),
                    "classification_errors": arm_contract_classes.get(
                        "classification_errors", {}
                    ),
                }
            ),
            "source_proof_edges": (
                {}
                if source_proof_evidence is None
                else {
                    "complete": source_proof_evidence.get("complete"),
                    "counts": source_proof_evidence.get("counts", {}),
                    "errors": source_proof_evidence.get("errors", {}),
                }
            ),
            "owner_region_evidence": (
                {}
                if owner_region_evidence is None
                else {
                    "complete": owner_region_evidence.get("complete"),
                    "counts": owner_region_evidence.get("counts", {}),
                    "errors": owner_region_evidence.get("errors", {}),
                }
            ),
            "byte_exact_fortran_extracts": len(extracted_oracles),
            "extracted_numerical_status_counts": dict(numerical_oracle_status_counts),
            "extract_note": (
                "Byte-exact source extraction is preparation only and does not count as a "
                "Fortran numerical oracle until independently compiled and executed."
            ),
        },
        "state_transition": {
            "slowproc_stomate_subset_closed": bool(state_audit and state_audit.get("closed")),
            "sechiba_component_subset_closed": bool(
                sechiba_state_audit and sechiba_state_audit.get("closed")
            ),
            "all_components_closed": gates["state_transition_ledger_closed"],
            "restart_alias_roundtrip_closed": bool(
                restart_alias_audit and restart_alias_audit.get("closed")
            ),
            "restart_state_lifecycle_closed": bool(
                restart_lifecycle_audit and restart_lifecycle_audit.get("closed")
            ),
            "slowproc_stomate_limitations": [] if state_audit is None else state_audit.get("limitations", []),
            "sechiba_limitations": (
                [] if sechiba_state_audit is None else sechiba_state_audit.get("limitations", [])
            ),
            "restart_alias_limitations": (
                [] if restart_alias_audit is None else restart_alias_audit.get("limitations", [])
            ),
            "restart_lifecycle_summary": (
                {} if restart_lifecycle_audit is None else restart_lifecycle_audit.get("summary", {})
            ),
            "restart_lifecycle_limitations": (
                []
                if restart_lifecycle_audit is None
                else restart_lifecycle_audit.get("limitations", [])
            ),
            "cold_start_restart_year_handoff": (
                {}
                if stage5_lifecycle_acceptance is None
                else {
                    "status": stage5_lifecycle_acceptance.get("status"),
                    "scope": stage5_lifecycle_acceptance.get("scope"),
                    "checks": len(stage5_lifecycle_acceptance.get("checks", [])),
                }
            ),
            "representative_paper_scientific_acceptance": (
                {}
                if representative_acceptance is None
                else {
                    "initial_state": representative_acceptance.get("initial_state"),
                    "landpoint_count": representative_acceptance.get("landpoint_count"),
                    "scientific_acceptance_count": representative_acceptance.get(
                        "scientific_acceptance_count"
                    ),
                }
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the PFT14 equivalence evidence certificate.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--evidence-scope", type=Path, default=DEFAULT_EVIDENCE_SCOPE)
    parser.add_argument("--state-audit", type=Path, default=DEFAULT_STATE_AUDIT)
    parser.add_argument("--sechiba-state-audit", type=Path, default=DEFAULT_SECHIBA_STATE_AUDIT)
    parser.add_argument("--restart-alias-audit", type=Path, default=DEFAULT_RESTART_ALIAS_AUDIT)
    parser.add_argument(
        "--restart-lifecycle-audit", type=Path, default=DEFAULT_RESTART_LIFECYCLE_AUDIT
    )
    parser.add_argument(
        "--stage5-lifecycle-acceptance",
        type=Path,
        default=DEFAULT_STAGE5_LIFECYCLE_ACCEPTANCE,
    )
    parser.add_argument(
        "--representative-acceptance",
        type=Path,
        default=DEFAULT_REPRESENTATIVE_ACCEPTANCE,
    )
    parser.add_argument("--arm-disposition", type=Path, default=DEFAULT_ARM_DISPOSITION)
    parser.add_argument("--arm-evidence", type=Path, default=DEFAULT_ARM_EVIDENCE)
    parser.add_argument(
        "--arm-contract-classes", type=Path, default=DEFAULT_ARM_CONTRACT_CLASSES
    )
    parser.add_argument(
        "--source-proof-evidence", type=Path, default=DEFAULT_SOURCE_PROOF_EVIDENCE
    )
    parser.add_argument(
        "--owner-region-evidence", type=Path, default=DEFAULT_OWNER_REGION_EVIDENCE
    )
    parser.add_argument("--micro-oracle-index", type=Path, default=DEFAULT_MICRO_ORACLE_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-equivalent", action="store_true")
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    ledger = yaml.safe_load(args.ledger.read_text(encoding="utf-8"))
    evidence_scope = yaml.safe_load(args.evidence_scope.read_text(encoding="utf-8"))
    state_audit = json.loads(args.state_audit.read_text(encoding="utf-8")) if args.state_audit.exists() else None
    sechiba_state_audit = (
        json.loads(args.sechiba_state_audit.read_text(encoding="utf-8"))
        if args.sechiba_state_audit.exists()
        else None
    )
    restart_alias_audit = (
        json.loads(args.restart_alias_audit.read_text(encoding="utf-8"))
        if args.restart_alias_audit.exists()
        else None
    )
    restart_lifecycle_audit = (
        json.loads(args.restart_lifecycle_audit.read_text(encoding="utf-8"))
        if args.restart_lifecycle_audit.exists()
        else None
    )
    stage5_lifecycle_acceptance = (
        json.loads(args.stage5_lifecycle_acceptance.read_text(encoding="utf-8"))
        if args.stage5_lifecycle_acceptance.exists()
        else None
    )
    representative_acceptance = (
        json.loads(args.representative_acceptance.read_text(encoding="utf-8"))
        if args.representative_acceptance.exists()
        else None
    )
    arm_disposition = (
        json.loads(args.arm_disposition.read_text(encoding="utf-8"))
        if args.arm_disposition.exists()
        else None
    )
    arm_evidence = (
        json.loads(args.arm_evidence.read_text(encoding="utf-8"))
        if args.arm_evidence.exists()
        else None
    )
    arm_contract_classes = (
        json.loads(args.arm_contract_classes.read_text(encoding="utf-8"))
        if args.arm_contract_classes.exists()
        else None
    )
    source_proof_evidence = (
        json.loads(args.source_proof_evidence.read_text(encoding="utf-8"))
        if args.source_proof_evidence.exists()
        else None
    )
    owner_region_evidence = (
        json.loads(args.owner_region_evidence.read_text(encoding="utf-8"))
        if args.owner_region_evidence.exists()
        else None
    )
    micro_oracle_index = (
        json.loads(args.micro_oracle_index.read_text(encoding="utf-8"))
        if args.micro_oracle_index.exists()
        else None
    )
    certificate = build_certificate(
        inventory,
        ledger,
        evidence_scope=evidence_scope,
        state_audit=state_audit,
        sechiba_state_audit=sechiba_state_audit,
        restart_alias_audit=restart_alias_audit,
        restart_lifecycle_audit=restart_lifecycle_audit,
        stage5_lifecycle_acceptance=stage5_lifecycle_acceptance,
        representative_acceptance=representative_acceptance,
        arm_disposition=arm_disposition,
        arm_evidence=arm_evidence,
        arm_contract_classes=arm_contract_classes,
        source_proof_evidence=source_proof_evidence,
        owner_region_evidence=owner_region_evidence,
        micro_oracle_index=micro_oracle_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    flow = certificate["control_flow"]
    evidence = certificate["evidence"]
    print(
        f"WROTE {args.output.resolve()} equivalent={certificate['equivalent']} "
        f"classified={flow['effectively_classified_reachable_branches']}/{flow['reachable_branches']} "
        f"span_mapped={flow['process_span_mapped_reachable_branches']}/{flow['reachable_branches']} "
        f"oracle={evidence['fortran_oracle_entries']}/{evidence['reachable_ledger_entries']} "
        f"production={evidence['production_verified_entries']}/{evidence['reachable_ledger_entries']}"
    )
    return int(args.require_equivalent and not certificate["equivalent"])


if __name__ == "__main__":
    raise SystemExit(main())
