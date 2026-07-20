from __future__ import annotations

import json
from pathlib import Path

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/reference_mode/micro_oracles"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"


def _document(family: str, name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / family / name).read_text(encoding="ascii"))


def test_hydrol_residual_owner_evidence_is_complete() -> None:
    coverage = _document("hydrol_residual_owners", "branch_coverage.json")
    owners = _document("hydrol_residual_owners", "owner_region_evidence.json")
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 37
    assert owners["complete"] and len(owners["records"]) == 3
    assert all(record["passed"] for record in owners["records"])


def test_hydrol_state_update_owner_evidence_is_complete() -> None:
    coverage = _document("hydrol_state_updates", "branch_coverage.json")
    owners = _document("hydrol_state_updates", "owner_region_evidence.json")
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 38
    assert owners["complete"] and len(owners["records"]) == 3
    assert all(record["passed"] for record in owners["records"])


def test_hydrol_lifecycle_owner_evidence_binds_exact_source_and_execution() -> None:
    comparison = _document("hydrol_lifecycle_owners", "comparison.json")
    owners = _document("hydrol_lifecycle_owners", "owner_region_evidence.json")
    assert comparison["status"] == "passed"
    assert len(comparison["dependency_fortran_oracles"]) == 7
    assert owners["complete"] and len(owners["records"]) == 6
    required_arms = {
        arm_id for record in owners["records"] for arm_id in record["required_arm_ids"]
    }
    assert {record["arm_id"] for record in comparison["arm_evidence"]} == required_arms
    assert all(record["passed"] for record in comparison["arm_evidence"])
    for procedure, metadata in comparison["source_spans"].items():
        span = extract_procedure_bytes(SOURCE, procedure)
        assert metadata["sha256"] == span.span_sha256
        extracted = EVIDENCE / "hydrol_lifecycle_owners" / metadata["asset"]
        assert extracted.read_bytes() == span.span_bytes


def test_hydrol_batch_owns_exactly_twelve_regions() -> None:
    families = (
        "hydrol_residual_owners",
        "hydrol_state_updates",
        "hydrol_lifecycle_owners",
    )
    records = [
        record
        for family in families
        for record in _document(family, "owner_region_evidence.json")["records"]
    ]
    assert len(records) == 12
    assert len({record["owner_region_id"] for record in records}) == 12
