from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/readdim2_geometry_owners"


def test_readdim2_geometry_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text())
    assert comparison["status"] == "passed"
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 33
    assert owners["complete"]
    assert len(owners["records"]) == 3
    assert sum(len(record["required_arm_ids"]) for record in owners["records"]) == 38
    assert sum(len(record["source_proof_arm_ids"]) for record in owners["records"]) == 5
