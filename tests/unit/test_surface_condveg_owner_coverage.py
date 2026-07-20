from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_surface_condveg_owner_evidence_is_complete() -> None:
    output = ROOT / "outputs/reference_mode/micro_oracles/surface_condveg_processes"
    coverage = json.loads((output / "branch_coverage.json").read_text())
    owners = json.loads((output / "owner_region_evidence.json").read_text())
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 6
    assert owners["complete"]
    assert len(owners["records"]) == 2
    assert all(record["passed"] for record in owners["records"])
