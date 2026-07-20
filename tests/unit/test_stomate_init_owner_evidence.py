from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stomate_init_owner"


def test_stomate_init_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="ascii"))
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text(encoding="ascii"))
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text(encoding="ascii"))
    assert comparison["status"] == "passed"
    assert coverage["branch_complete"] is True
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 5
    assert owners["complete"] is True
    assert len(owners["records"]) == 1
    assert owners["records"][0]["owner_region_id"] == "pft14-owner-contract-b2c84609d0d1"
