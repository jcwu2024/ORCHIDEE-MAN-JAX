from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dev"))

import build_stage4_stomate_parent_owner_evidence as builder  # noqa: E402


EVIDENCE = ROOT / (
    "outputs/reference_mode/micro_oracles/stage4_stomate_parent_orchestration/"
    "owner_region_evidence.json"
)
BRANCHES = EVIDENCE.with_name("branch_coverage.json")
COMPARISON = EVIDENCE.with_name("comparison.json")
OWNER_COUNTS = {
    "pft14-owner-contract-1201d85bae52": 6,
    "pft14-owner-contract-4619c18c114b": 21,
    "pft14-owner-contract-c9db8d90c5e7": 16,
    "pft14-owner-contract-e1608cfb5f71": 34,
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def test_parent_owner_evidence_closes_exact_contract_arms() -> None:
    document = _load(EVIDENCE)
    assert document["complete"] is True
    assert {record["owner_region_id"] for record in document["records"]} == set(OWNER_COUNTS)
    for record in document["records"]:
        assert len(record["required_arm_ids"]) == OWNER_COUNTS[record["owner_region_id"]]
        assert record["covered_arm_ids"] == record["required_arm_ids"]
        assert record["missing_arm_ids"] == []
        assert record["evidence_route"] == "fortran_transition_oracle_and_production"
        assert record["passed"] is True


def test_every_arm_has_source_strict_oracle_and_production_witness() -> None:
    document = _load(EVIDENCE)
    branches = _load(BRANCHES)
    assert branches["branch_complete"] is True
    assert branches["required_arm_count"] == 77
    assert branches["covered_arm_count"] == 77
    assert set(branches["arm_witnesses"]) == set(document["arm_witnesses"])
    for arm_id, witness in document["arm_witnesses"].items():
        assert arm_id.startswith("fortran_source/ORCHIDEE/src_stomate/")
        assert len(witness["source_line_sha256"]) == 64
        assert witness["strict_fortran_oracle_assets"]
        assert witness["production_transition_assets"]


def test_linked_evidence_hashes_and_production_boundaries_are_pinned() -> None:
    document = _load(EVIDENCE)
    for item in document["strict_fortran_oracles"]:
        path = ROOT / item["asset"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        linked = _load(path)
        builder._validate_comparison_document(linked)

    production = document["production_witness"]
    assert production["daily_trace"]["accepted"] is True
    assert production["daily_trace"]["last_itime"] == 17520
    assert production["lpj_trace"]["accepted"] is True
    assert production["lpj_trace"]["inferred_daily_calls"] == 365
    assert production["annual_modelout"]["accepted"] is True
    assert production["annual_modelout"]["max_relative_error"] <= 1.0e-6
    assert all(item["status"] == "passed" for item in production["semantic_windows"])
    assert all(item["status"] == "passed" for item in production["carbon_windows"])


def test_comparison_asset_contains_strict_and_production_results() -> None:
    comparison = _load(COMPARISON)
    assert comparison["status"] == "passed"
    assert len(comparison["comparisons"]) == len(builder.STRICT_ORACLE_FAMILIES) + 4
    assert all(item["passed"] for item in comparison["comparisons"])


def test_failed_or_empty_linked_oracle_cannot_receive_credit() -> None:
    with pytest.raises(ValueError, match="failed comparison"):
        builder._validate_comparison_document(
            {"status": "passed", "source_sha256": "x", "comparisons": [{"passed": False}]}
        )
    with pytest.raises(ValueError, match="non-empty passed"):
        builder._validate_comparison_document(
            {"status": "passed", "source_sha256": "x", "comparisons": []}
        )
