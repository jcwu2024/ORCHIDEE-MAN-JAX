from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stage4_batch_d_thermal"
OWNER_COUNTS = {
    "pft14-owner-contract-da695e536e76": 17,
    "pft14-owner-contract-7cedec1972ee": 2,
    "pft14-owner-contract-d63791b5a816": 40,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage4_batch_d_thermal_comparison_and_original_bytes_are_pinned() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    assert comparison["status"] == "passed"
    assert comparison["comparisons"]
    assert all(record["passed"] for record in comparison["comparisons"])
    assert comparison["minimal_real_format_input"]["format"] == "NetCDF3 64-bit offset"
    assert _sha256(ROOT / comparison["minimal_real_format_input"]["asset"]) == comparison[
        "minimal_real_format_input"
    ]["sha256"]

    span_groups = (
        comparison["reader_source_spans"],
        comparison["initialize_source_spans"],
    )
    for group in span_groups:
        for record in group.values():
            if not isinstance(record, dict):
                continue
            source = ROOT / record["source_file"]
            extracted = ROOT / record["extracted_file"]
            assert _sha256(source) == record["source_sha256"]
            assert _sha256(extracted) == record["span_sha256"]


def test_stage4_batch_d_thermal_arm_routes_and_owner_evidence_are_complete() -> None:
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    evidence = json.loads((OUTPUT / "owner_region_evidence.json").read_text())
    inputs = json.loads((OUTPUT / "inputs.json").read_text())

    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 59
    assert coverage["route_counts"] == {
        "gcov": 47,
        "fatal_proof": 0,
        "pinned_proof": 12,
    }
    assert coverage["fatal_proof_arm_ids"] == []
    assert not coverage["missing_arm_ids"]
    assert all(record["passed"] for record in coverage["arms"])

    records = {record["owner_region_id"]: record for record in evidence["records"]}
    assert evidence["complete"]
    assert set(records) == set(OWNER_COUNTS)
    for owner_id, arm_count in OWNER_COUNTS.items():
        record = records[owner_id]
        assert record["passed"]
        assert len(record["required_arm_ids"]) == arm_count
        assert record["covered_arm_ids"] == record["required_arm_ids"]
        assert not record["missing_arm_ids"]

    assert inputs["owner_ids"] == sorted(OWNER_COUNTS)
    assert inputs["float_policy"] == {"rtol": 1.0e-12, "atol": 1.0e-12}
    assert inputs["discrete_policy"] == "exact"
