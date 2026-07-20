from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/audit_pft14_arm_evidence.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_arm_evidence", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def _arm(name: str, *, owner: bool = True) -> dict[str, object]:
    return {
        "arm_id": f"source.f90:{name}:if:true",
        "branch_id": f"source.f90:{name}:if",
        "arm": "true",
        "file": "source.f90",
        "procedure": "step",
        "line": int(name),
        "source": "IF (condition) THEN",
        "disposition": "pft14_conditional",
        "scientific_source_owner_present": owner,
    }


def test_report_distinguishes_executable_and_owner_only_evidence() -> None:
    arms = [_arm("1"), _arm("2"), _arm("3"), _arm("4", owner=False)]
    oracle = {
        "records": [
            {
                "ledger_entry": "entry.one",
                "passed": True,
                "covered_arm_ids": [arms[0]["arm_id"]],
            },
            {
                "ledger_entry": "entry.two",
                "passed": True,
                "covered_arm_ids": [arms[1]["arm_id"]],
            },
        ]
    }
    production = {"records": [{"ledger_entry": "entry.one", "passed": True}]}

    report = AUDIT.build_report(
        {"records": arms},
        oracle,
        production,
        pending_evidence=[{"id": "pending", "pending_reason": "not complete"}],
    )

    assert report["counts"] == {
        "scientific_reachable_arms": 4,
        "fortran_oracle_and_production": 1,
        "fortran_oracle_only": 1,
        "owner_transition_and_production": 0,
        "owner_state_io_and_lifecycle": 0,
        "owner_output_acceptance": 0,
        "source_proof": 0,
        "missing_source_proof": 0,
        "contract_routed_arms": 1,
        "source_owner_only": 1,
        "missing_source_owner": 1,
        "unresolved_executable_evidence": 3,
        "pending_evidence_items": 1,
        "unknown_oracle_arms": 0,
        "non_scientific_oracle_arms": 0,
    }
    assert not report["complete"]


def test_report_closes_only_when_every_scientific_arm_has_both_evidence_layers() -> None:
    arms = [_arm("1"), _arm("2")]
    oracle = {
        "records": [
            {
                "ledger_entry": "entry.one",
                "passed": True,
                "covered_arm_ids": [item["arm_id"] for item in arms],
            }
        ]
    }
    production = {"records": [{"ledger_entry": "entry.one", "passed": True}]}

    report = AUDIT.build_report({"records": arms}, oracle, production)

    assert report["counts"]["fortran_oracle_and_production"] == 2
    assert report["counts"]["unresolved_executable_evidence"] == 0
    assert report["complete"]


def test_contract_routes_require_their_actual_owner_and_source_evidence() -> None:
    arms = [_arm("1"), _arm("2"), _arm("3")]
    ids = [item["arm_id"] for item in arms]
    contracts = {
        "arm_classifications": [
            {
                "arm_id": ids[0],
                "evidence_route": "fortran_transition_oracle_and_production",
                "owner_evidence_route": "fortran_transition_oracle_and_production",
            },
            {
                "arm_id": ids[1],
                "evidence_route": "fortran_state_io_oracle_and_lifecycle",
                "owner_evidence_route": "fortran_state_io_oracle_and_lifecycle",
            },
            {"arm_id": ids[2], "evidence_route": "source_proof"},
        ]
    }
    owners = {
        "records": [
            {
                "owner_region_id": "transition",
                "evidence_route": "fortran_transition_oracle_and_production",
                "required_arm_ids": [ids[0]],
                "covered_arm_ids": [ids[0]],
                "passed": True,
            },
            {
                "owner_region_id": "state",
                "evidence_route": "fortran_state_io_oracle_and_lifecycle",
                "required_arm_ids": [ids[1]],
                "covered_arm_ids": [ids[1]],
                "passed": True,
            },
        ]
    }
    proofs = {"records": [{"arm_id": ids[2], "passed": True}]}
    report = AUDIT.build_report(
        {"records": arms},
        {"records": []},
        {"records": []},
        contract_classes=contracts,
        source_proofs=proofs,
        owner_evidence=owners,
    )
    assert report["complete"] is True
    assert report["counts"]["owner_transition_and_production"] == 1
    assert report["counts"]["owner_state_io_and_lifecycle"] == 1
    assert report["counts"]["source_proof"] == 1

    owners["records"][0]["covered_arm_ids"] = []
    rejected = AUDIT.build_report(
        {"records": arms},
        {"records": []},
        {"records": []},
        contract_classes=contracts,
        source_proofs=proofs,
        owner_evidence=owners,
    )
    assert rejected["complete"] is False
    assert rejected["counts"]["source_owner_only"] == 1
    assert rejected["owner_evidence_errors"]


def test_oracle_for_classified_inactive_arm_is_not_unknown() -> None:
    active = _arm("1")
    inactive = {**_arm("2"), "disposition": "paper_static_inactive"}
    oracle = {
        "records": [
            {
                "ledger_entry": "entry.one",
                "passed": True,
                "covered_arm_ids": [active["arm_id"], inactive["arm_id"]],
            }
        ]
    }
    production = {"records": [{"ledger_entry": "entry.one", "passed": True}]}
    report = AUDIT.build_report({"records": [active, inactive]}, oracle, production)
    assert report["complete"] is True
    assert report["counts"]["unknown_oracle_arms"] == 0
    assert report["counts"]["non_scientific_oracle_arms"] == 1
