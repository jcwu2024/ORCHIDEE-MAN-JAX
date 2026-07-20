from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/oracle_lane_stomate_soilcarbon_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("oracle_lane_stomate_soilcarbon_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_soilcarbon_fragment_has_four_verified_entries():
    result = _module().audit_fragment()
    assert result["status"] == "passed"
    assert result["ledger_entries"] == [
        "stomate.active.ok_leak_firstcall_active_layer",
        "stomate.active.ok_leak_litter",
        "stomate.active.ok_leak_soilcarbon",
        "stomate.active.tf_doc",
    ]


def test_soilcarbon_fragment_hashes_complete_original_owners_and_callees():
    result = _module().audit_fragment()
    procedures = {item["id"]: item for item in result["procedures"]}
    assert procedures["littercalc_leak"]["line_count"] == 1237
    assert procedures["soilcarbon_leak"]["line_count"] == 1776
    assert procedures["deadleaf"]["line_count"] == 51
    assert procedures["altcalc_DOC"]["line_count"] == 141
    assert all(len(item["span_sha256"]) == 64 for item in procedures.values())


def test_soilcarbon_arm_gate_closes_litter_and_full_owner():
    contract = _module().audit_fragment()["arm_contract"]
    assert contract["litter_required_and_covered"] == 63
    assert contract["soilcarbon_explicit_exclusions"] == 4
    assert contract["soilcarbon_defined_inventory"] == 237
    assert contract["soilcarbon_defined_covered"] == 237


def test_littercalc_leak_full_fortran_oracle(tmp_path):
    script = ROOT / "scripts/dev/oracle_lane_stomate_soilcarbon_litter.py"
    spec = importlib.util.spec_from_file_location("oracle_lane_stomate_soilcarbon_litter", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.run_oracle(tmp_path)
    assert result["status"] == "passed"
    assert len(result["comparisons"]) == 75
    assert all(item["passed"] for item in result["comparisons"])
