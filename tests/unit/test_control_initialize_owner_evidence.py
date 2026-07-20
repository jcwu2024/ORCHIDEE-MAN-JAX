from __future__ import annotations

import json
from pathlib import Path

from dataclasses import fields

from jax_orchidee.parameters.control import ControlInitializationState
from scripts.dev.oracle_lane_control_initialize_owner import FAMILY, OWNER_ID, SOURCE


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def test_control_owner_uses_byte_exact_original_module_and_all_state_fields():
    comparison = json.loads((EVIDENCE / "comparison.json").read_text(encoding="ascii"))
    assert comparison["status"] == "passed"
    assert comparison["byte_exact"] is True
    assert (EVIDENCE / "control.f90").read_bytes() == SOURCE.read_bytes()
    expected = {field.name for field in fields(ControlInitializationState)} - {"provenance"}
    assert set(comparison["state_contract_fields"]) == expected
    compared = {item["name"].split(".", 1)[1] for item in comparison["comparisons"] if "." in item["name"]}
    assert expected <= compared


def test_control_owner_covers_all_reachable_contract_arms():
    coverage = json.loads((EVIDENCE / "branch_coverage.json").read_text(encoding="ascii"))
    assert coverage["branch_complete"] is True
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 36
    assert coverage["missing_arm_ids"] == []
    assert coverage["classification_errors"] == []
    assert (EVIDENCE / "control_owner_oracle.gcov").is_file()


def test_control_owner_evidence_binds_requested_owner_and_call_order():
    owners = json.loads((EVIDENCE / "owner_region_evidence.json").read_text(encoding="ascii"))
    assert owners["complete"] is True
    assert owners["records"][0]["passed"] is True
    assert owners["records"][0]["owner_region_id"] == OWNER_ID
    comparison = json.loads((EVIDENCE / "comparison.json").read_text(encoding="ascii"))
    call_checks = [item for item in comparison["comparisons"] if item["name"].endswith(".parameter_initializers")]
    assert len(call_checks) == 6 and all(item["passed"] for item in call_checks)
