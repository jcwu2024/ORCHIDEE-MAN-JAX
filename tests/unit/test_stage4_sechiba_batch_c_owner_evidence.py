from __future__ import annotations

import json
from pathlib import Path

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/sechiba_batch_c"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"


def test_batch_c_closes_sechiba_family_with_exact_source_and_real_witnesses() -> None:
    inputs = json.loads((OUTPUT / "inputs.json").read_text(encoding="ascii"))
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text(encoding="ascii"))
    evidence = json.loads((OUTPUT / "owner_region_evidence.json").read_text(encoding="ascii"))
    comparison = json.loads(
        (OUTPUT / "sechiba_main_fixed_tail_comparison.json").read_text(encoding="ascii")
    )
    aggregate = json.loads(
        (ROOT / "outputs/reference_mode/pft14_owner_region_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert inputs["owner_ids"] == ["pft14-owner-contract-beb73c2fad63"]
    assert len(inputs["case_matrix"]) == 7
    assert len(evidence["records"]) == 6
    for procedure, metadata in inputs["owner_procedure_spans"].items():
        span = extract_procedure_bytes(SOURCE, procedure)
        assert metadata["sha256"] == span.span_sha256
        assert (OUTPUT / metadata["asset"]).read_bytes() == span.span_bytes
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 105
    assert coverage["missing_arm_ids"] == []
    assert coverage["branch_complete"] is True
    assert {record["support"] for record in coverage["arms"]} == {
        "pinned_source_proof",
        "real_gcov",
        "exact_fatal_witness",
    }
    records = {
        item["owner_region_id"]: item
        for item in evidence["records"]
    }
    assert set(records) == {
        "pft14-owner-contract-d5ed3bbbb641",
        "pft14-owner-contract-bee72b6b6bf0",
        "pft14-owner-contract-8437dcaf9cc2",
        "pft14-owner-contract-79f78d322bec",
        "pft14-owner-contract-198a52d8c6f6",
        "pft14-owner-contract-beb73c2fad63",
    }
    assert evidence["complete"] is True
    assert all(item["passed"] is True for item in records.values())
    record = records["pft14-owner-contract-beb73c2fad63"]
    assert len(record["required_arm_ids"]) == 29
    assert record["required_arm_ids"] == record["covered_arm_ids"]
    assert record["missing_arm_ids"] == []
    assert record["comparison_asset"].endswith("sechiba_main_fixed_tail_comparison.json")
    assert comparison["status"] == "passed"
    assert comparison["comparison_passed"] is True
    assert comparison["tolerance_policy"]["changed"] is False
    assert len(comparison["gcov_mapping"]) == 18
    assert all(mapping["passed"] for mapping in comparison["gcov_mapping"])
    assert comparison["fatal_witness"]["matched"] is True
    harness = (OUTPUT / "sechiba_main_fixed_tail_oracle.f90").read_bytes()
    source_lines = SOURCE.read_bytes().splitlines(keepends=True)
    for metadata in (
        comparison["source_fragments"]["rotation_fragment"],
        comparison["source_fragments"]["control_fragment"],
    ):
        start, end = metadata["lines"]
        fragment = b"".join(source_lines[start - 1 : end])
        assert harness.count(fragment.rstrip()) == 1
    for name, metadata in comparison["source_fragments"]["procedures"].items():
        span = extract_procedure_bytes(ROOT / metadata["source"], name)
        assert span.span_sha256 == metadata["sha256"]
        assert harness.count(span.span_bytes.rstrip()) == 1

    accepted = [
        item
        for item in aggregate["records"]
        if item["owner_region_id"] == "pft14-owner-contract-beb73c2fad63"
    ]
    assert len(accepted) == 1
    assert accepted[0]["passed"] is True
    assert len(accepted[0]["covered_arm_ids"]) == 29
    assert not any(aggregate["errors"].values())
