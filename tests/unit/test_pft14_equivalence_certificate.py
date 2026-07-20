from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "audit_pft14_equivalence_certificate.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_equivalence_certificate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _inventory():
    procedure_id = "fortran_source/ORCHIDEE/src_sechiba/a.f90:1:step"
    branch_id = "fortran_source/ORCHIDEE/src_sechiba/a.f90:2:if"
    return {
        "procedures": [
            {
                "id": procedure_id,
                "name": "step",
                "file": "fortran_source/ORCHIDEE/src_sechiba/a.f90",
            }
        ],
        "branches": [
            {
                "id": branch_id,
                "file": "fortran_source/ORCHIDEE/src_sechiba/a.f90",
                "line": 2,
                "procedure_id": procedure_id,
            }
        ],
        "reachable_branch_ids": [branch_id],
        "control_flow_arms": [
            {"id": f"{branch_id}:true", "branch_id": branch_id, "arm": "true"},
            {"id": f"{branch_id}:false", "branch_id": branch_id, "arm": "false"},
        ],
        "target_reachable_arm_ids": [f"{branch_id}:true", f"{branch_id}:false"],
        "reachable_unresolved_callee_names": [],
    }


def _ledger(*, classify: bool, evidence_level: str | None):
    entry = {
        "id": "a.active.step",
        "reachability": "active",
        "fortran_file": "fortran_source/ORCHIDEE/src_sechiba/a.f90",
        "fortran_subroutine": "step",
        "fortran_lines": "1-3",
    }
    if classify:
        entry["control_flow_ids"] = ["fortran_source/ORCHIDEE/src_sechiba/a.f90:2:if"]
    if evidence_level is not None:
        entry["evidence_level"] = evidence_level
    return {"entries": [entry]}


def test_process_span_does_not_count_as_explicit_branch_classification():
    certificate = AUDIT.build_certificate(_inventory(), _ledger(classify=False, evidence_level=None))
    assert certificate["control_flow"]["process_span_mapped_reachable_branches"] == 1
    assert certificate["control_flow"]["explicitly_classified_reachable_branches"] == 0
    assert not certificate["gates"]["control_flow_fully_classified"]
    assert not certificate["equivalent"]


def test_explicit_classification_and_production_evidence_close_local_gates_only():
    scope = {
        "entries": {"a.active.step": {"kind": "production_verified", "scope": "micro oracle"}}
    }
    ledger = _ledger(classify=True, evidence_level=None)
    branch_id = "fortran_source/ORCHIDEE/src_sechiba/a.f90:2:if"
    ledger["entries"][0]["control_flow_arm_ids"] = [
        f"{branch_id}:true",
        f"{branch_id}:false",
    ]
    certificate = AUDIT.build_certificate(_inventory(), ledger, evidence_scope=scope)
    assert certificate["gates"]["control_flow_fully_classified"]
    assert certificate["gates"]["all_reachable_entries_have_fortran_oracle"]
    assert certificate["gates"]["all_reachable_entries_production_verified"]
    assert not certificate["equivalent"]


def test_point_trace_anchor_is_not_counted_as_general_fortran_oracle():
    scope = {
        "entries": {"a.active.step": {"kind": "point_trace_anchor", "scope": "one point"}}
    }
    certificate = AUDIT.build_certificate(
        _inventory(), _ledger(classify=True, evidence_level=None), evidence_scope=scope
    )
    assert certificate["evidence"]["point_trace_anchor_entries"] == 1
    assert certificate["evidence"]["fortran_oracle_entries"] == 0
    assert not certificate["gates"]["all_reachable_entries_have_fortran_oracle"]


def test_site_classification_does_not_implicitly_classify_both_branch_arms():
    certificate = AUDIT.build_certificate(
        _inventory(), _ledger(classify=True, evidence_level=None)
    )

    assert certificate["control_flow"]["explicitly_classified_reachable_branches"] == 1
    assert certificate["control_flow"]["explicitly_classified_reachable_arms"] == 0
    assert not certificate["gates"]["control_flow_fully_classified"]


def test_generated_structural_arm_dispositions_count_as_explicit_classification():
    branch_id = "fortran_source/ORCHIDEE/src_sechiba/a.f90:2:if"
    arm_disposition = {
        "records": [
            {"arm_id": f"{branch_id}:true", "status": "classified", "disposition": "diagnostic_taken"},
            {"arm_id": f"{branch_id}:false", "status": "classified", "disposition": "diagnostic_not_taken"},
        ]
    }
    certificate = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=True, evidence_level=None),
        arm_disposition=arm_disposition,
    )

    assert certificate["control_flow"]["explicitly_classified_reachable_arms"] == 2
    assert certificate["control_flow"]["arm_complete_reachable_branches"] == 1
    assert certificate["control_flow"]["effectively_classified_reachable_branches"] == 1
    assert certificate["control_flow"]["arm_disposition_status_counts"] == {"classified": 2}


def test_byte_exact_extract_does_not_count_as_executed_fortran_oracle():
    certificate = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        micro_oracle_index={
            "oracles": [
                {"id": "q.qsat", "numerical_oracle_status": "not_run"},
            ]
        },
    )

    assert certificate["evidence"]["byte_exact_fortran_extracts"] == 1
    assert certificate["evidence"]["extracted_numerical_status_counts"] == {"not_run": 1}
    assert certificate["evidence"]["fortran_oracle_entries"] == 0


def test_reachability_classification_does_not_replace_scientific_source_owner():
    branch_id = "fortran_source/ORCHIDEE/src_sechiba/a.f90:2:if"
    arm_disposition = {
        "records": [
            {
                "arm_id": f"{branch_id}:true",
                "status": "classified",
                "disposition": "pft14_conditional",
                "scientific_source_owner_present": False,
            },
            {
                "arm_id": f"{branch_id}:false",
                "status": "classified",
                "disposition": "pft14_conditional",
                "scientific_source_owner_present": False,
            },
        ]
    }
    certificate = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=True, evidence_level=None),
        arm_disposition=arm_disposition,
    )

    assert certificate["gates"]["control_flow_fully_classified"]
    assert not certificate["gates"]["all_scientific_arms_source_mapped"]
    assert len(certificate["control_flow"]["scientific_arms_missing_source_owner"]) == 2


def test_source_owner_mapping_does_not_replace_executable_arm_evidence():
    branch_id = "fortran_source/ORCHIDEE/src_sechiba/a.f90:2:if"
    arm_disposition = {
        "records": [
            {
                "arm_id": f"{branch_id}:true",
                "status": "classified",
                "disposition": "pft14_conditional",
                "scientific_source_owner_present": True,
            },
            {
                "arm_id": f"{branch_id}:false",
                "status": "classified",
                "disposition": "pft14_conditional",
                "scientific_source_owner_present": True,
            },
        ]
    }
    incomplete = {
        "complete": False,
        "counts": {
            "scientific_reachable_arms": 2,
            "unresolved_executable_evidence": 1,
            "pending_evidence_items": 0,
        },
    }
    certificate = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=True, evidence_level=None),
        arm_disposition=arm_disposition,
        arm_evidence=incomplete,
    )

    assert certificate["gates"]["all_scientific_arms_source_mapped"]
    assert not certificate["gates"]["all_scientific_arms_have_executable_evidence"]

    complete = {
        "complete": True,
        "counts": {
            "scientific_reachable_arms": 2,
            "unresolved_executable_evidence": 0,
            "pending_evidence_items": 0,
        },
    }
    closed = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=True, evidence_level=None),
        arm_disposition=arm_disposition,
        arm_evidence=complete,
    )
    assert closed["gates"]["all_scientific_arms_have_executable_evidence"]


def test_arm_contract_route_classification_is_an_independent_gate() -> None:
    arm_evidence = {"counts": {"source_owner_only": 2}}
    incomplete = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        arm_evidence=arm_evidence,
        arm_contract_classes={
            "classification_complete": False,
            "counts": {"input_arms": 2, "classified_arms": 1},
        },
    )
    assert not incomplete["gates"]["arm_contract_evidence_routes_fully_classified"]

    complete = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        arm_evidence=arm_evidence,
        arm_contract_classes={
            "classification_complete": True,
            "counts": {"input_arms": 2, "classified_arms": 2},
        },
    )
    assert complete["gates"]["arm_contract_evidence_routes_fully_classified"]


def test_source_proof_edges_require_complete_pinned_evidence() -> None:
    contract = {
        "counts": {"evidence_route_arms": {"source_proof": 2}},
    }
    incomplete = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        arm_contract_classes=contract,
        source_proof_evidence={
            "complete": False,
            "counts": {"expected_source_proof_arms": 2, "passed_source_proof_arms": 1},
        },
    )
    assert not incomplete["gates"]["source_proof_edges_closed"]

    complete = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        arm_contract_classes=contract,
        source_proof_evidence={
            "complete": True,
            "counts": {"expected_source_proof_arms": 2, "passed_source_proof_arms": 2},
        },
    )
    assert complete["gates"]["source_proof_edges_closed"]


def test_owner_region_evidence_is_complete_only_when_every_region_passes() -> None:
    classes = {"owner_regions": [{"owner_region_id": "one"}, {"owner_region_id": "two"}]}
    incomplete = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        arm_contract_classes=classes,
        owner_region_evidence={
            "complete": False,
            "counts": {"required_owner_regions": 2, "passed_owner_regions": 1},
        },
    )
    assert not incomplete["gates"]["all_owner_regions_have_required_evidence"]

    complete = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        arm_contract_classes=classes,
        owner_region_evidence={
            "complete": True,
            "counts": {"required_owner_regions": 2, "passed_owner_regions": 2},
        },
    )
    assert complete["gates"]["all_owner_regions_have_required_evidence"]


def test_closed_state_subsets_do_not_close_incomplete_all_component_gate():
    certificate = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        state_audit={"closed": True, "all_components_closed": False},
        sechiba_state_audit={"closed": True, "all_components_closed": False},
    )
    assert certificate["state_transition"]["slowproc_stomate_subset_closed"]
    assert certificate["state_transition"]["sechiba_component_subset_closed"]
    assert not certificate["gates"]["state_transition_ledger_closed"]


def test_restart_lifecycle_is_an_independent_state_gate():
    certificate = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        state_audit={"closed": True, "all_components_closed": True},
        sechiba_state_audit={"closed": True, "all_components_closed": True},
        restart_alias_audit={"closed": True},
        restart_lifecycle_audit={"closed": False, "summary": {"contract_fields": 123}},
    )

    assert not certificate["gates"]["state_transition_ledger_closed"]
    assert not certificate["state_transition"]["restart_state_lifecycle_closed"]
    assert certificate["state_transition"]["restart_lifecycle_summary"]["contract_fields"] == 123


def test_stage5_lifecycle_gate_requires_executed_passing_checks():
    base = {
        "status": "passed",
        "production_execution": True,
        "scope": "cold start and split restart",
        "checks": [{"name": "roundtrip", "passed": True}],
    }
    certificate = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        stage5_lifecycle_acceptance=base,
    )
    assert certificate["gates"]["cold_start_restart_year_handoff_pass"]
    assert certificate["state_transition"]["cold_start_restart_year_handoff"]["checks"] == 1

    base["checks"][0]["passed"] = False
    rejected = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        stage5_lifecycle_acceptance=base,
    )
    assert not rejected["gates"]["cold_start_restart_year_handoff_pass"]


def test_representative_acceptance_gate_requires_every_point_and_cold_start():
    acceptance = {
        "initial_state": "cold-start",
        "landpoint_count": 2,
        "scientific_acceptance_count": 2,
        "results": [
            {"landpoint_id": "001.0-071.0", "scientific_acceptance_pass": True},
            {"landpoint_id": "319.0-057.0", "scientific_acceptance_pass": True},
        ],
    }
    certificate = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        representative_acceptance=acceptance,
    )
    assert certificate["gates"]["representative_paper_scientific_acceptance_pass"]

    acceptance["results"][1]["scientific_acceptance_pass"] = False
    acceptance["scientific_acceptance_count"] = 1
    rejected = AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        representative_acceptance=acceptance,
    )
    assert not rejected["gates"]["representative_paper_scientific_acceptance_pass"]

    acceptance["initial_state"] = "restart-backed"
    assert not AUDIT.build_certificate(
        _inventory(),
        _ledger(classify=False, evidence_level=None),
        representative_acceptance=acceptance,
    )["gates"]["representative_paper_scientific_acceptance_pass"]
