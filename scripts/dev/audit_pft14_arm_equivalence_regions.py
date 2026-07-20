from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARM_EVIDENCE = ROOT / "outputs/reference_mode/pft14_arm_evidence_gap.json"
DEFAULT_CONTROL_FLOW = ROOT / "outputs/reference_mode/pft14_control_flow_resolution.json"
DEFAULT_EXTENSIONS = ROOT / "docs/source_audits/pft14_process_inventory_extensions.yaml"
DEFAULT_REACHABLE_LEDGER = ROOT / "docs/source_audits/pft14_reachable_ledger.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/pft14_arm_equivalence_regions.json"

OWNER_ONLY_STATUS = "source_owner_only"
REGION_POLICIES = {
    "scientific_transition_required": {
        "state_impact": "scientific_state_transition",
        "required_evidence": "fortran_transition_oracle_and_production",
        "requires_executable_evidence": True,
    },
    "state_io_transition_required": {
        "state_impact": "restart_or_history_state_io",
        "required_evidence": "fortran_state_io_oracle_and_production",
        "requires_executable_evidence": True,
    },
    "driver_regridding_required": {
        "state_impact": "forcing_or_static_input_mapping",
        "required_evidence": "fortran_regridding_oracle_and_production",
        "requires_executable_evidence": True,
    },
    "implemented_owner_arm_evidence_required": {
        "state_impact": "owner_declared_state_contract",
        "required_evidence": "fortran_arm_oracle_and_production",
        "requires_executable_evidence": True,
    },
    "valid_input_infrastructure_source_proof": {
        "state_impact": "no_scientific_state_mutation_on_valid_inputs",
        "required_evidence": "source_proof_and_denominator_reclassification",
        "requires_executable_evidence": False,
    },
    "paper_static_inactive_reclassification": {
        "state_impact": "unreachable_under_fixed_pft14_workflow",
        "required_evidence": "static_source_proof_and_denominator_reclassification",
        "requires_executable_evidence": False,
    },
}

EXTENSION_REGION_KIND = {
    "scientific_process": "scientific_transition_required",
    "state_io": "state_io_transition_required",
    "driver_regridding": "driver_regridding_required",
    "valid_input_infrastructure": "valid_input_infrastructure_source_proof",
    "paper_static_inactive": "paper_static_inactive_reclassification",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _extension_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record["fortran_file"]).replace("\\", "/").lower(),
        str(record["fortran_subroutine"]).lower(),
    )


def _arm_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record["file"]).replace("\\", "/").lower(),
        str(record.get("procedure") or "unknown").lower(),
    )


def _ledger_entries_by_id(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry["id"]): entry for entry in ledger.get("entries", [])}


def _classify_arm(
    arm: dict[str, Any],
    resolution: dict[str, Any],
    extension: dict[str, Any] | None,
    ledger_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    if extension is not None:
        category = str(extension["category"])
        try:
            return EXTENSION_REGION_KIND[category], "process_inventory_extension"
        except KeyError as exc:
            raise ValueError(f"unknown process inventory category {category!r}") from exc

    process_ids = [str(value) for value in resolution.get("process_ledger_ids", [])]
    if process_ids:
        entries = [ledger_by_id.get(value) for value in process_ids]
        missing = [value for value, entry in zip(process_ids, entries, strict=True) if entry is None]
        if missing:
            raise ValueError(f"{arm['arm_id']}: unknown process ledger IDs {missing}")
        reachability = {str(entry["reachability"]) for entry in entries if entry is not None}
        if reachability == {"inactive"}:
            return "paper_static_inactive_reclassification", "reachable_process_ledger"

    implemented = [
        owner_id
        for owner_id, status in zip(
            resolution.get("source_owner_ids", []),
            resolution.get("source_owner_statuses", []),
            strict=True,
        )
        if status == "implemented"
    ]
    if implemented:
        return "implemented_owner_arm_evidence_required", "implemented_source_owner"

    raise ValueError(
        f"{arm['arm_id']}: no extension, inactive ledger proof, or implemented source owner"
    )


def _region_identity(
    kind: str,
    arm: dict[str, Any],
    resolution: dict[str, Any],
    extension: dict[str, Any] | None,
) -> tuple[str, ...]:
    implemented_owners = sorted(
        str(owner_id)
        for owner_id, status in zip(
            resolution.get("source_owner_ids", []),
            resolution.get("source_owner_statuses", []),
            strict=True,
        )
        if status == "implemented"
    )
    process_ids = sorted(str(value) for value in resolution.get("process_ledger_ids", []))
    return (
        kind,
        str(arm["file"]),
        str(arm.get("procedure") or "unknown"),
        str(extension.get("id")) if extension else "",
        "|".join(implemented_owners),
        "|".join(process_ids),
    )


def build_report(
    arm_evidence: dict[str, Any],
    control_flow: dict[str, Any],
    extensions: dict[str, Any],
    reachable_ledger: dict[str, Any],
) -> dict[str, Any]:
    unresolved = [
        record
        for record in arm_evidence.get("records", [])
        if record.get("evidence_status") == OWNER_ONLY_STATUS
    ]
    resolution_by_id = {
        str(row["control_flow_id"]): row for row in control_flow.get("rows", [])
    }
    extension_by_key = {
        _extension_key(entry): entry for entry in extensions.get("entries", [])
    }
    ledger_by_id = _ledger_entries_by_id(reachable_ledger)

    region_members: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    classifications: dict[str, tuple[str, str]] = {}
    missing_resolutions: list[str] = []
    for arm in unresolved:
        branch_id = str(arm["branch_id"])
        resolution = resolution_by_id.get(branch_id)
        if resolution is None:
            missing_resolutions.append(str(arm["arm_id"]))
            continue
        extension = extension_by_key.get(_arm_key(arm))
        kind, basis = _classify_arm(arm, resolution, extension, ledger_by_id)
        identity = _region_identity(kind, arm, resolution, extension)
        classifications[str(arm["arm_id"])] = (kind, basis)
        region_members[identity].append(
            {
                "arm": arm,
                "resolution": resolution,
                "extension": extension,
            }
        )

    ordered_keys = sorted(region_members)
    regions: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for identity in ordered_keys:
        members = region_members[identity]
        kind = identity[0]
        policy = REGION_POLICIES[kind]
        first = members[0]
        extension = first["extension"]
        arm_ids = sorted(str(item["arm"]["arm_id"]) for item in members)
        branch_ids = sorted({str(item["arm"]["branch_id"]) for item in members})
        owner_ids = sorted(
            {
                str(owner_id)
                for item in members
                for owner_id in item["resolution"].get("source_owner_ids", [])
            }
        )
        owner_statuses = sorted(
            {
                str(status)
                for item in members
                for status in item["resolution"].get("source_owner_statuses", [])
            }
        )
        implemented_owner_ids = sorted(
            {
                str(owner_id)
                for item in members
                for owner_id, status in zip(
                    item["resolution"].get("source_owner_ids", []),
                    item["resolution"].get("source_owner_statuses", []),
                    strict=True,
                )
                if status == "implemented"
            }
        )
        jax_owners = sorted(
            {
                str(owner)
                for item in members
                for owner in item["resolution"].get("jax_owners", [])
                if owner
            }
        )
        process_ledger_ids = sorted(
            {
                str(entry_id)
                for item in members
                for entry_id in item["resolution"].get("process_ledger_ids", [])
            }
        )
        process_ledger_entries = [ledger_by_id[entry_id] for entry_id in process_ledger_ids]
        process_ledger_reachability = sorted(
            {str(entry["reachability"]) for entry in process_ledger_entries}
        )
        process_ledger_conditions = [
            str(entry["condition"])
            for entry in process_ledger_entries
            if entry.get("condition")
        ]
        sources: dict[tuple[int, str], list[str]] = defaultdict(list)
        for item in members:
            arm = item["arm"]
            sources[(int(arm["line"]), str(arm["source"]))].append(str(arm["arm_id"]))
        source_sites = [
            {"line": line, "source": source, "arm_ids": sorted(ids)}
            for (line, source), ids in sorted(sources.items())
        ]
        basis = classifications[arm_ids[0]][1]
        identity_digest = hashlib.sha256("\x1f".join(identity).encode()).hexdigest()[:12]
        region_id = f"pft14-region-{identity_digest}"
        procedure_gap = None if extension is None else extension.get("gap")
        source_proof_rationale = None
        if not policy["requires_executable_evidence"]:
            rationale_parts = []
            if extension is not None and extension.get("reason"):
                rationale_parts.append(str(extension["reason"]))
            rationale_parts.extend(process_ledger_conditions)
            source_proof_rationale = " ".join(rationale_parts) or None
        known_implementation_gap = not implemented_owner_ids and bool(
            policy["requires_executable_evidence"]
        )
        region = {
            "region_id": region_id,
            "region_kind": kind,
            "classification_basis": basis,
            "fortran_file": first["arm"]["file"],
            "fortran_procedure": first["arm"].get("procedure"),
            "source_lines": sorted({int(item["arm"]["line"]) for item in members}),
            "source_sites": source_sites,
            "arm_count": len(arm_ids),
            "branch_count": len(branch_ids),
            "arm_ids": arm_ids,
            "branch_ids": branch_ids,
            "process_inventory_id": None if extension is None else extension.get("id"),
            "process_inventory_category": None
            if extension is None
            else extension.get("category"),
            "process_inventory_reason": None
            if extension is None
            else extension.get("reason"),
            "procedure_level_gap_context": procedure_gap,
            "source_owner_ids": owner_ids,
            "source_owner_statuses": owner_statuses,
            "implemented_source_owner_ids": implemented_owner_ids,
            "jax_owners": jax_owners,
            "process_ledger_ids": process_ledger_ids,
            "process_ledger_reachability": process_ledger_reachability,
            "state_impact": policy["state_impact"],
            "required_evidence": policy["required_evidence"],
            "requires_executable_evidence": policy["requires_executable_evidence"],
            "source_proof_rationale": source_proof_rationale,
            "known_implementation_gap": known_implementation_gap,
            "can_share_transition_oracle": len(arm_ids) > 1,
            "closed": False,
        }
        regions.append(region)
        assignments.extend(
            {
                "arm_id": arm_id,
                "region_id": region_id,
                "region_kind": kind,
                "classification_basis": classifications[arm_id][1],
            }
            for arm_id in arm_ids
        )

    unresolved_ids = [str(record["arm_id"]) for record in unresolved]
    assigned_ids = [str(record["arm_id"]) for record in assignments]
    duplicate_assignments = sorted(
        arm_id for arm_id, count in Counter(assigned_ids).items() if count != 1
    )
    missing_assignments = sorted(set(unresolved_ids) - set(assigned_ids))
    unknown_assignments = sorted(set(assigned_ids) - set(unresolved_ids))
    source_proof_without_rationale = sorted(
        region["region_id"]
        for region in regions
        if not region["requires_executable_evidence"]
        and not region["source_proof_rationale"]
    )
    partition_complete = (
        not missing_resolutions
        and not duplicate_assignments
        and not missing_assignments
        and not unknown_assignments
        and not source_proof_without_rationale
        and len(assigned_ids) == len(unresolved_ids)
    )
    region_kind_counts = Counter(region["region_kind"] for region in regions)
    arm_kind_counts = Counter(
        assignment["region_kind"] for assignment in assignments
    )
    executable_regions = [
        region for region in regions if region["requires_executable_evidence"]
    ]
    source_proof_regions = [
        region for region in regions if not region["requires_executable_evidence"]
    ]
    known_gap_regions = [region for region in regions if region["known_implementation_gap"]]
    partial_context_regions = [
        region
        for region in regions
        if any(status in {"partial", "missing"} for status in region["source_owner_statuses"])
    ]
    return {
        "schema_version": 1,
        "scope": "partition of every PFT14 scientific arm that has only source-owner evidence",
        "partition_complete": partition_complete,
        "counts": {
            "owner_only_arms": len(unresolved_ids),
            "assigned_arms": len(assigned_ids),
            "equivalence_regions": len(regions),
            "transition_oracle_regions": len(executable_regions),
            "transition_oracle_arms": sum(
                int(region["arm_count"]) for region in executable_regions
            ),
            "source_reclassification_regions": len(source_proof_regions),
            "source_reclassification_arms": sum(
                int(region["arm_count"]) for region in source_proof_regions
            ),
            "known_implementation_gap_regions": len(known_gap_regions),
            "partial_or_missing_procedure_context_regions": len(partial_context_regions),
            "region_kinds": dict(sorted(region_kind_counts.items())),
            "arm_kinds": dict(sorted(arm_kind_counts.items())),
        },
        "partition_errors": {
            "missing_control_flow_resolution": sorted(missing_resolutions),
            "missing_assignments": missing_assignments,
            "duplicate_assignments": duplicate_assignments,
            "unknown_assignments": unknown_assignments,
            "source_proof_without_rationale": source_proof_without_rationale,
        },
        "regions": regions,
        "arm_assignments": sorted(assignments, key=lambda item: item["arm_id"]),
        "note": (
            "This asset is a work partition, not equivalence evidence. Every region remains "
            "open until its declared executable evidence or source reclassification is "
            "completed and the arm-evidence denominator is regenerated. Partial procedure "
            "owners are retained as context and never treated as closure."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Partition unresolved PFT14 arm evidence into finite equivalence regions."
    )
    parser.add_argument("--arm-evidence", type=Path, default=DEFAULT_ARM_EVIDENCE)
    parser.add_argument("--control-flow", type=Path, default=DEFAULT_CONTROL_FLOW)
    parser.add_argument("--extensions", type=Path, default=DEFAULT_EXTENSIONS)
    parser.add_argument("--reachable-ledger", type=Path, default=DEFAULT_REACHABLE_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        json.loads(args.arm_evidence.read_text(encoding="utf-8")),
        json.loads(args.control_flow.read_text(encoding="utf-8")),
        _load_yaml(args.extensions),
        _load_yaml(args.reachable_ledger),
    )
    report["inputs"] = {
        name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
        for name, path in {
            "arm_evidence": args.arm_evidence,
            "control_flow": args.control_flow,
            "process_inventory_extensions": args.extensions,
            "reachable_ledger": args.reachable_ledger,
        }.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    counts = report["counts"]
    print(
        f"WROTE {args.output.resolve()} complete={report['partition_complete']} "
        f"arms={counts['assigned_arms']}/{counts['owner_only_arms']} "
        f"regions={counts['equivalence_regions']} "
        f"oracle={counts['transition_oracle_regions']} "
        f"source_reclass={counts['source_reclassification_regions']} "
        f"known_model_gaps={counts['known_implementation_gap_regions']}"
    )
    return int(args.require_complete and not report["partition_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
