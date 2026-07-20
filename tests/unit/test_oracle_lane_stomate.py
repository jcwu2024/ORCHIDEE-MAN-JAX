from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/oracle_lane_stomate_carbon.py"


def _module():
    spec = importlib.util.spec_from_file_location("oracle_lane_stomate_carbon", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stomate_fragment_contract_separates_verified_and_pending():
    document = _module().validate_fragment()
    assert [family["id"] for family in document["families"]] == [
        "stomate_prescribe_restart_gate",
        "stomate_constraints",
        "stomate_gap_mortality",
        "stomate_final_lai_vmax",
        "stomate_turnover",
        "stomate_allocation",
        "stomate_npp_growth",
    ]
    families = document["pending_families"]
    assert families == []
    for family in document["families"] + families:
        assert all("control_flow_arm_ids" in case for case in family["branch_cases"])


def test_only_numerically_executed_procedures_are_marked_verified():
    document = yaml.safe_load(
        (ROOT / "docs/source_audits/oracle_families/stomate_carbon.yaml").read_text(
            encoding="utf-8"
        )
    )
    procedures = [
        procedure
        for family in document["families"] + document["pending_families"]
        for procedure in family["procedures"]
    ]
    assert [
        procedure["id"]
        for procedure in procedures
        if procedure["numerical_oracle_status"] == "verified"
    ] == [
        "prescribe",
        "constraints",
        "gap",
        "vmax",
        "turn",
        "alloc",
        "npp_calc",
        "crop_bmalloc",
    ]


def test_constraints_fortran_oracle(tmp_path):
    result = _module().run_oracle(tmp_path)
    assert result["status"] == "passed"
    assert {item["name"] for item in result["comparisons"]} == {"adapted", "regenerate"}


def test_gap_fortran_oracle(tmp_path):
    result = _module().run_gap_oracle(tmp_path)
    assert result["status"] == "passed"
    assert {item["name"] for item in result["comparisons"]} == {
        "mortality",
        "ind",
        "biomass",
        "bm_to_litter",
    }


def test_vmax_fortran_oracle(tmp_path):
    result = _module().run_vmax_oracle(tmp_path)
    assert result["status"] == "passed"
    assert {item["name"] for item in result["comparisons"]} == {
        "vcmax",
        "leaf_age",
        "leaf_frac",
    }


def test_prescribe_fortran_oracle(tmp_path):
    result = _module().run_prescribe_oracle(tmp_path)
    assert result["status"] == "passed"
    assert "cold_biomass" in {item["name"] for item in result["comparisons"]}
    assert "restart_biomass" in {item["name"] for item in result["comparisons"]}
    assert {"tree_cn", "grass_cn"} <= {item["name"] for item in result["comparisons"]}


def test_turnover_fortran_oracle(tmp_path):
    result = _module().run_turnover_oracle(tmp_path)
    assert result["status"] == "passed"
    assert all(item["passed"] for item in result["comparisons"])
    assert max(item["max_ulp_error"] for item in result["comparisons"]) <= 1


def test_allocation_fortran_oracle(tmp_path):
    result = _module().run_allocation_oracle(tmp_path)
    assert result["status"] == "passed"


def test_npp_fortran_oracle(tmp_path):
    result = _module().run_npp_oracle(tmp_path)
    assert result["status"] == "passed"
    assert all(item["max_abs_error"] == 0.0 for item in result["comparisons"])
