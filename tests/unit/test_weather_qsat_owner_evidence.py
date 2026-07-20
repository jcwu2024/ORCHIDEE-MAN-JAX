from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/weather_qsat"


def test_weather_qsat_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text())
    assert comparison["status"] == "passed"
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 3
    assert owners["complete"]
    assert len(owners["records"]) == 1
