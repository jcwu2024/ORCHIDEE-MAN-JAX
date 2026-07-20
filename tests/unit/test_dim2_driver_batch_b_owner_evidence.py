from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/dim2_driver_batch_b_formal"
OWNER_IDS = {
    "pft14-owner-contract-8b2f034e7b0a",
    "pft14-owner-contract-9d83d36c3d1b",
    "pft14-owner-contract-2f2f83908ced",
    "pft14-owner-contract-f6d7f1a4e631",
}


def test_dim2_driver_batch_b_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="ascii"))
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text(encoding="ascii"))
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text(encoding="ascii"))

    assert comparison["status"] == "passed"
    assert comparison["formal_audit"]["complete"] is True
    assert coverage["branch_complete"] is True
    assert coverage["covered_arm_count"] == coverage["required_arm_count"] == 76
    assert {record["owner_region_id"] for record in owners["records"]} == OWNER_IDS
    assert all(record["passed"] for record in owners["records"])
