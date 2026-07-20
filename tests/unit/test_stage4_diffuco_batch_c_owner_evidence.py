from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/diffuco_batch_c"
OWNER_IDS = {
    "pft14-owner-contract-8915be0a92c4",
    "pft14-owner-contract-92c46e7e0a73",
    "pft14-owner-contract-ca049a6eb718",
    "pft14-owner-contract-a36633fa671c",
    "pft14-owner-contract-e23a3a9a3599",
}


def test_diffuco_batch_c_owner_evidence_is_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="ascii"))
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text(encoding="ascii"))
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text(encoding="ascii"))
    assert comparison["status"] == "passed"
    assert comparison["focused_tests"]["passed"]
    assert coverage["mapping"] == "gcov_plus_exact_trace_source_proof"
    assert coverage["covered_arm_count"] == 22
    assert coverage["branch_complete"] and owners["complete"]
    assert {record["owner_region_id"] for record in owners["records"]} == OWNER_IDS
    trans = next(record for record in owners["records"] if record["fortran_procedure"] == "diffuco_trans")
    assert trans["passed"] and not trans["missing_arm_ids"]
    finalize = next(record for record in owners["records"] if record["fortran_procedure"] == "diffuco_finalize")
    assert finalize["passed"] and not finalize["missing_arm_ids"]
    initialize = next(record for record in owners["records"] if record["fortran_procedure"] == "diffuco_initialize")
    assert initialize["passed"] and not initialize["missing_arm_ids"]
    closed_ids = {trans["owner_region_id"], finalize["owner_region_id"], initialize["owner_region_id"]}
    assert all(record["passed"] for record in owners["records"])


def test_diffuco_batch_c_fatals_bind_exact_extracted_sources() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="ascii"))
    for name, metadata in comparison["source_spans"].items():
        assert (OUTPUT / metadata["asset"]).is_file(), name
    for fatal in comparison["exact_compile_fatals"].values():
        assert fatal["kind"] == "exact_extracted_source_compile"
        assert fatal["returncode"] != 0
        assert fatal["stderr"]
