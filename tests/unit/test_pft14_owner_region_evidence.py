from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/audit_pft14_owner_region_evidence.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_owner_region_evidence", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_repository_owner_evidence_has_no_invalid_or_duplicate_records() -> None:
    report = AUDIT.build_report(
        json.loads(
            (ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json").read_text()
        ),
        json.loads(
            (ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json").read_text()
        ),
        AUDIT.load_evidence_documents(ROOT / "outputs/reference_mode/micro_oracles"),
    )
    assert not any(report["errors"].values())
    assert report["counts"]["passed_owner_regions"] >= 12
    assert report["counts"]["passed_by_route"]["source_proof"] == 9
    assert report["counts"]["passed_by_route"]["fortran_transition_oracle_and_production"] >= 3


def test_owner_region_requires_exact_arm_set() -> None:
    contract = {
        "owner_regions": [
            {
                "owner_region_id": "owner.one",
                "base_region_id": "base.one",
                "fortran_procedure": "step",
                "evidence_route": "fortran_transition_oracle_and_production",
                "arm_ids": ["a", "b"],
            }
        ]
    }
    evidence = {
        "complete": True,
        "records": [
            {
                "owner_region_id": "owner.one",
                "required_arm_ids": ["a", "b"],
                "covered_arm_ids": ["a"],
                "passed": True,
            }
        ],
    }
    report = AUDIT.build_report(contract, {"records": []}, [("evidence.json", evidence)])
    assert not report["complete"]
    assert report["errors"]["invalid_evidence_records"]
