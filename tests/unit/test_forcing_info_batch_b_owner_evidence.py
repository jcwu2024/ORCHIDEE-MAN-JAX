from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/forcing_info_batch_b_formal"
OWNER_ID = "pft14-owner-contract-d6769359ae0a"


def test_forcing_info_batch_b_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="ascii"))
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text(encoding="ascii"))
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text(encoding="ascii"))

    assert comparison["status"] == "passed"
    assert coverage["branch_complete"] is True
    assert coverage["covered_arm_count"] == coverage["required_arm_count"] == 31
    assert owners["complete"] is True
    assert len(owners["records"]) == 1
    assert owners["records"][0]["owner_region_id"] == OWNER_ID
    assert owners["records"][0]["missing_arm_ids"] == []
