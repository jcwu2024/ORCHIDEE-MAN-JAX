from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_hydrol_explicit_snow_owner_evidence_is_complete() -> None:
    output = ROOT / "outputs/reference_mode/micro_oracles/hydrol_explicit_snow_nonzero"
    coverage = json.loads((output / "branch_coverage.json").read_text())
    owners = json.loads((output / "owner_region_evidence.json").read_text())
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 39
    assert owners["complete"]
    assert len(owners["records"]) == 8
    assert all(record["passed"] for record in owners["records"])


def test_hydrol_explicit_snow_compound_where_uses_terminal_line() -> None:
    output = ROOT / "outputs/reference_mode/micro_oracles/hydrol_explicit_snow_nonzero"
    coverage = json.loads((output / "branch_coverage.json").read_text())
    level_arms = [
        record for record in coverage["arms"] if record["original_line"] == 1613
    ]
    assert len(level_arms) == 2
    assert {record["condition_terminal_original_line"] for record in level_arms} == {
        1614
    }
    assert all(record["passed"] for record in level_arms)
