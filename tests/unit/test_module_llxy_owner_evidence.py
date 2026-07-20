from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/module_llxy_owner"


def _load(name: str) -> dict[str, object]:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def test_module_llxy_numerical_and_owner_evidence_is_complete() -> None:
    comparison = _load("comparison.json")
    coverage = _load("branch_coverage.json")
    owners = _load("owner_region_evidence.json")

    assert comparison["status"] == "passed"
    assert coverage["branch_complete"] is True
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 141
    assert coverage["missing_arm_ids"] == []
    assert owners["complete"] is True
    assert len(owners["records"]) == 15
    assert all(record["passed"] for record in owners["records"])
    assert all(
        set(record["required_arm_ids"]) == set(record["covered_arm_ids"])
        for record in owners["records"]
    )


def test_select_case_arms_use_dynamic_statement_counts() -> None:
    coverage = _load("branch_coverage.json")
    select_arms = [
        arm
        for arm in coverage["arms"]
        if arm["arm_id"].endswith(("select_case:fallthrough", "case:case"))
    ]

    assert len(select_arms) == 23
    assert all(
        arm.get("coverage_kind") == "gcov_case_body_statement_execution"
        and arm.get("case_body_execution_count", 0) > 0
        for arm in select_arms
    )
