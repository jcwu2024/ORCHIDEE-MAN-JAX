from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_surface_enerbil_main_owner_evidence_is_complete() -> None:
    output = ROOT / "outputs/reference_mode/micro_oracles/surface_enerbil_main"
    coverage = json.loads((output / "branch_coverage.json").read_text())
    owners = json.loads((output / "owner_region_evidence.json").read_text())
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 16
    assert owners["complete"]
    assert len(owners["records"]) == 5
    assert all(record["passed"] for record in owners["records"])


def test_surface_enerbil_compound_snow_condition_uses_terminal_line() -> None:
    output = ROOT / "outputs/reference_mode/micro_oracles/surface_enerbil_main"
    coverage = json.loads((output / "branch_coverage.json").read_text())
    snow_arms = [
        record for record in coverage["arms"] if record["original_line"] == 1561
    ]
    assert len(snow_arms) == 2
    assert {record["condition_terminal_original_line"] for record in snow_arms} == {
        1562
    }
    assert all(record["passed"] for record in snow_arms)
