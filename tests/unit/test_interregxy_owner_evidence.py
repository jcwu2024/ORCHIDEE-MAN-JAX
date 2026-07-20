from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/interregxy_owners"
OWNER_IDS = {
    "pft14-owner-contract-84349e3f0552",
    "pft14-owner-contract-124f8a37065f",
    "pft14-owner-contract-80e01aaac94b",
}


def test_interregxy_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="ascii"))
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text(encoding="ascii"))
    owners = json.loads(
        (OUTPUT / "owner_region_evidence.json").read_text(encoding="ascii")
    )

    assert comparison["status"] == "passed"
    assert coverage["branch_complete"] is True
    assert coverage["covered_arm_count"] == coverage["required_arm_count"] == 36
    assert owners["complete"] is True
    assert {record["owner_region_id"] for record in owners["records"]} == OWNER_IDS
    assert all(record["passed"] for record in owners["records"])
