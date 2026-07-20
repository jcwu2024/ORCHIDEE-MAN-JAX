from __future__ import annotations

import json
from pathlib import Path

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes
from scripts.dev.oracle_lane_soil_parameter_owner import (
    ERROR_MODES,
    PROCEDURE,
    SOURCE,
    VAR_SOURCE,
    VALID_MODES,
    compose,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/soil_parameter_owner"
OWNER_ID = "pft14-owner-contract-13771122554f"


def test_soil_parameter_owner_evidence_is_complete() -> None:
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text())

    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 30
    assert not coverage["missing_arm_ids"]
    assert owners["complete"]
    assert [record["owner_region_id"] for record in owners["records"]] == [OWNER_ID]
    assert owners["records"][0]["passed"]
    assert len(owners["records"][0]["covered_arm_ids"]) == 38


def test_soil_parameter_numerical_and_ipslerr_cases_pass_strictly() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    inputs = json.loads((OUTPUT / "inputs.json").read_text())

    assert comparison["status"] == "passed"
    assert all(record["passed"] for record in comparison["comparisons"])
    assert comparison["tolerance"]["float_rtol"] <= 1e-13
    assert comparison["tolerance"]["discrete"] == "exact"
    assert set(inputs["valid_modes"]) == set(VALID_MODES)
    assert inputs["error_modes"] == ERROR_MODES


def test_soil_parameter_lane_embeds_original_procedure_and_real_state(tmp_path: Path) -> None:
    generated = tmp_path / "oracle.f90"
    hashes = compose(generated)
    payload = generated.read_bytes()
    span = extract_procedure_bytes(SOURCE, PROCEDURE)

    assert hashes[PROCEDURE] == span.span_sha256
    assert payload.count(span.span_bytes) == 1
    assert payload.count(VAR_SOURCE.read_bytes()) == 1
    assert b"interface getin_p" in payload
    assert b"error stop 73" in payload
