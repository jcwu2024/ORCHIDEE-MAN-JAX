from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "outputs/reference_mode/stage5_lifecycle_acceptance.json"


def test_stage5_lifecycle_evidence_is_real_and_passing():
    result = json.loads(ASSET.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["production_execution"] is True
    assert result["execution"]["trace_inputs"] is False
    assert result["execution"]["server_used"] is False
    assert all(item["passed"] for item in result["checks"])
    checks = {item["name"]: item for item in result["checks"]}
    assert checks["driver_file_roundtrip_exact"]["compared_fields"] == 13
    assert checks["sechiba_file_roundtrip_exact"]["compared_fields"] == 93
    assert checks["stomate_file_roundtrip_exact"]["compared_fields"] >= 161
    assert checks["owner_region_evidence_260_of_260"]["passed"] is True
    assert checks["scientific_arm_evidence_4580_of_4580"]["passed"] is True
    assert checks["hydrol_thermosoil_production_execution_passed"]["comparisons"] == 69
    assert checks["stomate_production_execution_passed"]["comparisons"] > 0
    assert checks["split_vs_memory_modelout_exact"]["passed"] is True
    assert checks["split_vs_memory_state_exact"]["passed"] is True
