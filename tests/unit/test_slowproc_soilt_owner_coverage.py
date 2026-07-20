from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_slowproc_soilt_owner_evidence_is_complete() -> None:
    output = ROOT / "outputs/reference_mode/micro_oracles" / "slowproc_batch_d"
    coverage = json.loads((output / "branch_coverage.json").read_text())
    owners = json.loads((output / "owner_region_evidence.json").read_text())
    assert coverage["branch_complete"]
    assert coverage["required_by_owner"]["pft14-owner-contract-463396c23861"] == 51
    assert owners["complete"]
    record = next(
        item
        for item in owners["records"]
        if item["owner_region_id"] == "pft14-owner-contract-463396c23861"
    )
    assert len(record["covered_arm_ids"]) == 51
    assert record["passed"]
