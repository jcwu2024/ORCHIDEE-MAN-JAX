from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/audit_pft14_source_proof_evidence.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_source_proof_evidence", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def _repository_report() -> dict[str, object]:
    return AUDIT.build_report(
        json.loads(
            (ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json").read_text()
        ),
        json.loads((ROOT / "outputs/reference_mode/pft14_arm_disposition.json").read_text()),
    )


def test_repository_source_proofs_are_complete_and_unique() -> None:
    report = _repository_report()
    assert report["complete"]
    assert report["counts"]["expected_source_proof_arms"] == report["counts"][
        "passed_source_proof_arms"
    ]
    assert not any(report["errors"].values())
    assert len({record["arm_id"] for record in report["records"]}) == report["counts"][
        "expected_source_proof_arms"
    ]


def test_source_proof_records_pin_fortran_source_and_line_hashes() -> None:
    report = _repository_report()
    assert all(len(record["fortran_source_sha256"]) == 64 for record in report["records"])
    assert all(
        len(record["fortran_physical_line_sha256"]) == 64 for record in report["records"]
    )
    assert all(record["passed"] for record in report["records"])


def test_source_proof_never_claims_to_close_the_owner_region() -> None:
    report = _repository_report()
    assert "owner-region" in report["note"]
    assert all("owner_region_id" not in record for record in report["records"])
