from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/readdim2_flinget_buffer"


def test_readdim2_flinget_buffer_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text())
    assert comparison["status"] == "passed"
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 13
    assert owners["complete"]
    assert len(owners["records"]) == 1
    assert len(owners["records"][0]["required_arm_ids"]) == 14
    assert len(owners["records"][0]["source_proof_arm_ids"]) == 1
