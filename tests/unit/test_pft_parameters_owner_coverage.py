from __future__ import annotations

import json
from pathlib import Path

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes
from scripts.dev.oracle_lane_pft_parameters_owners import (
    ERROR_MODES,
    MTC_SOURCE,
    PROCEDURES,
    SOURCE,
    VALID_MODES,
    compose,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/pft_parameters_owners"
EXPECTED_OWNER_IDS = {
    "pft14-owner-contract-0191c23be34f",
    "pft14-owner-contract-2c88b333ac1d",
    "pft14-owner-contract-534d24c44347",
    "pft14-owner-contract-6de98b965834",
    "pft14-owner-contract-768ebd80367d",
    "pft14-owner-contract-84b71e44290b",
    "pft14-owner-contract-ad061fdebb6a",
    "pft14-owner-contract-d55a06ac0681",
    "pft14-owner-contract-e2860a45c9de",
    "pft14-owner-contract-eb0f276e4b27",
}


def test_pft_parameters_owner_evidence_is_complete() -> None:
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text())

    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 78
    assert not coverage["missing_arm_ids"]
    assert owners["complete"]
    assert {record["owner_region_id"] for record in owners["records"]} == EXPECTED_OWNER_IDS
    assert all(record["passed"] for record in owners["records"])


def test_pft_parameters_numerical_and_error_cases_pass_at_strict_tolerance() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    inputs = json.loads((OUTPUT / "inputs.json").read_text())

    assert comparison["status"] == "passed"
    assert len(comparison["comparisons"]) == 121
    assert all(record["passed"] for record in comparison["comparisons"])
    assert comparison["tolerance"]["float_rtol"] <= 1e-12
    assert comparison["tolerance"]["discrete"] == "exact"
    assert set(inputs["valid_modes"]) == set(VALID_MODES)
    assert inputs["error_modes"] == ERROR_MODES


def test_pft_parameters_lane_embeds_real_spans_and_minimal_mtc_host(tmp_path: Path) -> None:
    generated_path = tmp_path / "oracle.f90"
    hashes = compose(generated_path)
    generated = generated_path.read_bytes()

    for procedure in PROCEDURES:
        span = extract_procedure_bytes(SOURCE, procedure)
        assert hashes[procedure] == span.span_sha256
        assert generated.count(span.span_bytes) == 1
    assert MTC_SOURCE.read_bytes() not in generated
    assert b"character(len=34), parameter :: MTC_name(nvmc)" in generated
    assert b"getin_total=getin_total+1" in generated
    assert b"writeback_total=writeback_total+1" in generated
