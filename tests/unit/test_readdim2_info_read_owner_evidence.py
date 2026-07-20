from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/readdim2_info_read_owners"


def test_forcing_read_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text())

    assert comparison["status"] == "passed"
    comparison_names = {record["name"] for record in comparison["comparisons"]}
    for case in ("interp", "watchout", "weather_init", "weather_later"):
        assert {
            f"{case}_contfrac",
            f"{case}_meta",
            f"{case}_neighbours",
            f"{case}_resolution",
        } <= comparison_names

    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 6
    assert owners["complete"]
    assert len(owners["records"]) == 1
    assert owners["records"][0]["fortran_procedure"] == "forcing_read"
    assert not owners["records"][0]["missing_arm_ids"]
