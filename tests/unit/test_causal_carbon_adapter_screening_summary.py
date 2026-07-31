from __future__ import annotations

from copy import deepcopy

import pytest

from research.daily_coarse_graining.causal_carbon_adapter_screening_summary import (
    GATE_FAMILIES,
    HORIZONS,
    SELECTION_SLICES,
    summarize_screening,
)


def _report():
    ratios = []
    for slice_id in SELECTION_SLICES:
        for field in ("gpp", "maintenance", "carbon_32l", "deepC_peat"):
            ratios.append(
                {
                    "id": f"interface/{slice_id}/{field}",
                    "ratio": 0.8,
                    "threshold": 1.02,
                    "candidate": 0.8,
                    "control": 1.0,
                    "passed": True,
                }
            )
        for horizon in HORIZONS:
            fields = {
                "global_state": ("global",),
                "primary_state": tuple(f"primary_{index}" for index in range(7)),
                "flux_state_bias_all_leads": tuple(f"flux_{index}" for index in range(3)),
                "stock_tendency_bias_all_leads": tuple(f"stock_{index}" for index in range(4)),
                "guard": ("litter", "DOC"),
            }
            for family, family_fields in fields.items():
                for field in family_fields:
                    ratios.append(
                        {
                            "id": f"{family}/{slice_id}/horizon_{horizon}/{field}",
                            "ratio": 0.8,
                            "threshold": 0.95,
                            "candidate": 0.8,
                            "control": 1.0,
                            "passed": True,
                        }
                    )
    assert len(ratios) == 165
    return {
        "schema_version": "causal_carbon_adapter_screening_report_v1",
        "promotion_eligible": True,
        "sealed_test_used": False,
        "classification": {
            "status": "passed",
            "decision": "advance_to_three_seed_confirmation",
            "ratio_gates": ratios,
            "hard_constraints": [{"id": f"hard_{index}", "passed": True} for index in range(108)],
            "structural_constraints": [{"id": f"structural_{index}", "passed": True} for index in range(6)],
        },
    }


def test_summary_accepts_complete_passing_screen():
    summary = summarize_screening(_report())

    assert summary["screening_status"] == "passed"
    assert summary["ratio_gates"]["total"] == 165
    assert summary["ratio_gates"]["failed"] == 0
    assert summary["hard_gates"]["failed"] == 0
    assert summary["structural_gates"]["failed"] == 0
    assert not any(summary["interpretation_flags"].values())
    assert {record["id"] for record in summary["ratio_gates"]["by_family"]} == set(GATE_FAMILIES)


def test_summary_attributes_predeclared_failure_modes():
    report = deepcopy(_report())
    failed_ids = {
        "interface/joint/gpp",
        "primary_state/spatial/horizon_7/primary_0",
        "stock_tendency_bias_all_leads/temporal/horizon_30/stock_0",
        "guard/joint/horizon_30/DOC",
    }
    for record in report["classification"]["ratio_gates"]:
        if record["id"] in failed_ids:
            record["ratio"] = 1.2
            record["passed"] = False
    report["classification"]["hard_constraints"][0]["passed"] = False
    report["classification"]["status"] = "rejected"
    report["classification"]["decision"] = "stop_and_attribute_declared_screening_failure"

    summary = summarize_screening(report)

    assert summary["ratio_gates"]["failed"] == 4
    assert summary["hard_gates"]["failed"] == 1
    flags = summary["interpretation_flags"]
    assert flags["execution_or_state_integrity_failure"]
    assert flags["one_day_causal_interface_failure"]
    assert flags["seven_day_recursive_failure"]
    assert flags["thirty_day_recursive_failure"]
    assert flags["spatial_generalization_failure"]
    assert flags["joint_generalization_failure"]
    assert flags["carbon_flux_or_stock_bias_failure"]
    assert flags["litter_or_doc_guard_regression"]


def test_summary_refuses_smoke_and_incomplete_gate_inventory():
    smoke = _report()
    smoke["promotion_eligible"] = False
    with pytest.raises(ValueError, match="refuses smoke"):
        summarize_screening(smoke)

    incomplete = _report()
    incomplete["classification"]["ratio_gates"].pop()
    with pytest.raises(ValueError, match="165 ratio gates"):
        summarize_screening(incomplete)
