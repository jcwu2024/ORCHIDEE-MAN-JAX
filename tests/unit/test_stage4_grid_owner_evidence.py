from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/reference_mode/micro_oracles/grid_owners"


def _load(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="ascii"))


def test_grid_owner_family_closes_all_nine_regions_and_arms() -> None:
    comparison = _load("comparison.json")
    coverage = _load("branch_coverage.json")
    owners = _load("owner_region_evidence.json")

    assert comparison["status"] == "passed"
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == 68
    assert coverage["covered_arm_count"] == 68
    assert coverage["runtime_covered_arm_count"] == 62
    assert coverage["execution_witness_arm_count"] == 4
    assert coverage["source_proof_arm_count"] == 2
    assert coverage["missing_arm_ids"] == []
    assert owners["complete"]
    assert len(owners["records"]) == 9
    assert all(record["passed"] for record in owners["records"])
    assert all(record["missing_arm_ids"] == [] for record in owners["records"])


def test_grid_local_non_gcov_evidence_is_explicit_and_source_hashed() -> None:
    proofs = _load("family_source_proofs.json")
    records = proofs["proofs"]

    assert len(proofs["source_file_sha256"]) == 64
    assert {record["evidence_kind"] for record in records} == {
        "structural_source_proof",
        "fortran_execution_witness",
    }
    assert (
        sum(record["evidence_kind"] == "structural_source_proof" for record in records)
        == 2
    )
    assert (
        sum(
            record["evidence_kind"] == "fortran_execution_witness" for record in records
        )
        == 4
    )
    assert all(record["passed"] and record["proof"] for record in records)


def test_grid_oracle_compiles_real_scientific_dependencies() -> None:
    comparison = _load("comparison.json")
    dependencies = comparison["scientific_dependency_sha256"]

    assert set(dependencies) == {"module_llxy.f90", "haversine.f90"}
    assert all(len(digest) == 64 for digest in dependencies.values())
    assert len(comparison["procedure_span_sha256"]) == 9
    assert all(item["passed"] for item in comparison["comparisons"])
