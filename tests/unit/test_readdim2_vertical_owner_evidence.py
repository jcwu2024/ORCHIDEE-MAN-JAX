from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/readdim2_vertical"


def test_readdim2_vertical_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text())
    dispositions = json.loads(
        (ROOT / "outputs/reference_mode/pft14_arm_disposition.json").read_text()
    )
    assert comparison["status"] == "passed"
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 56
    assert owners["complete"]
    assert len(owners["records"]) == 1
    assert len(owners["records"][0]["required_arm_ids"]) == 56
    unreachable = {
        record["arm_id"]
        for record in dispositions["records"]
        if record["disposition"] == "source_unreachable"
    }
    assert {
        "fortran_source/ORCHIDEE/src_driver/readdim2.f90:2088:if:false",
        "fortran_source/ORCHIDEE/src_driver/readdim2.f90:2184:else_if:false",
    } <= unreachable
