from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/reference_mode/micro_oracles/interpweight_helpers"


def _load(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_interpweight_helpers_have_complete_owner_and_arm_evidence() -> None:
    comparison = _load("comparison.json")
    coverage = _load("branch_coverage.json")
    owners = _load("owner_region_evidence.json")

    assert comparison["status"] == "passed"
    assert len(comparison["procedure_span_sha256"]) == 14
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == 130
    assert coverage["covered_arm_count"] == 130
    assert owners["complete"]
    assert len(owners["records"]) == 14
    assert all(record["passed"] for record in owners["records"])


def test_interpweight_helper_evidence_excludes_large_and_file_dimension_owners() -> (
    None
):
    owners = _load("owner_region_evidence.json")
    procedures = {record["fortran_procedure"] for record in owners["records"]}

    assert procedures == {
        "interpweight_calc_resolution_in",
        "interpweight_modifying_input1d",
        "interpweight_modifying_input2d",
        "interpweight_modifying_input3d",
        "interpweight_modifying_input4d",
        "interpweight_masking_input1d",
        "interpweight_masking_input2d",
        "interpweight_masking_input3d",
        "interpweight_masking_input4d",
        "interpweight_provide_fractions1d",
        "interpweight_provide_fractions2d",
        "interpweight_provide_fractions4d",
        "interpweight_provide_interpolation2d",
        "interpweight_valvecr",
    }
