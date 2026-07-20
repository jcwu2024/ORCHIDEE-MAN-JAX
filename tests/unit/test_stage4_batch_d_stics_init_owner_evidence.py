from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stics_init_batch_d"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sticslai/Stics_init.f90"
OWNER_REGION_ID = "pft14-owner-contract-5499b4b53358"
ARMS = {
    "fortran_source/ORCHIDEE/src_sticslai/Stics_init.f90:328:if:true",
    "fortran_source/ORCHIDEE/src_sticslai/Stics_init.f90:441:if:false",
    "fortran_source/ORCHIDEE/src_sticslai/Stics_init.f90:492:if:false",
    "fortran_source/ORCHIDEE/src_sticslai/Stics_init.f90:505:if:false",
}


def _load(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="ascii"))


def test_stics_init_batch_d_owner_evidence_is_complete() -> None:
    comparison = _load("comparison.json")
    coverage = _load("branch_coverage.json")
    owners = _load("owner_region_evidence.json")
    record = owners["records"][0]

    assert comparison["status"] == "passed"
    assert comparison["case_count"] == 4
    assert comparison["writeback_field_count"] > 100
    assert all(item["passed"] for item in comparison["comparisons"])
    assert coverage["mapping"] == "parsed_gcov_plus_canonical_source_proof"
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 4
    assert owners["complete"] and record["passed"]
    assert record["owner_region_id"] == OWNER_REGION_ID
    assert set(record["required_arm_ids"]) == ARMS
    assert set(record["gcov_arm_ids"]) == ARMS
    assert set(record["source_proof_arm_ids"]) == ARMS
    assert record["fatal_boundary_arm_ids"] == []
    central = _load("central_audit_report.json")
    assert central["assigned_owner_acceptance"]["accepted"]
    assert not central["errors"]["duplicate_owner_regions"]
    assert not central["errors"]["invalid_evidence_records"]
    assert not central["errors"]["unknown_owner_regions"]


def test_stics_init_batch_d_binds_exact_source_and_all_route_assets() -> None:
    comparison = _load("comparison.json")
    extracted = extract_procedure_bytes(SOURCE, "stics_init")
    extracted_asset = OUTPUT / comparison["source"]["extracted_asset"]

    assert comparison["source"]["exact_extracted_bytes"]
    assert extracted_asset.read_bytes() == extracted.span_bytes
    assert comparison["source"]["span_sha256"] == hashlib.sha256(extracted.span_bytes).hexdigest()
    assert (OUTPUT / "oracle.f90.gcov").is_file()
    assert _load("source_proof_evidence.json")["complete"]
    fatal = _load("fatal_boundary_evidence.json")
    assert fatal["complete"] and fatal["required_fatal_arm_ids"] == []
    for case_id in range(1, 5):
        assert (OUTPUT / f"case_{case_id}.bin").stat().st_size > 0
