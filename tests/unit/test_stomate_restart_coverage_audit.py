from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev.audit_stomate_restart_state_coverage import DEFAULT_SCAN, build_audit  # noqa: E402


def test_stomate_restart_state_coverage_audit_classifies_all_scan_fields():
    audit = build_audit(DEFAULT_SCAN)
    summary = audit["summary"]

    assert summary["scan_fields"] == 167
    assert summary["unclassified"] == 0
    assert summary["covered_by_jax_restart_state"] == 110
    assert summary["active_output_diagnostic_not_later_day_entry_state"] == 1
    assert summary["inactive_fire_lcc_grazing_branch"] == 19
    assert summary["inactive_wetland_ch4_branch"] == 19
    assert summary["non_leak_or_dgvm_branch_state"] == 1
    assert summary["spinup_equilibrium_state"] == 8

    statuses = audit["by_status"]
    assert {"field": "resp_hetero", "mapped_to": "resp_hetero"} in statuses[
        "active_output_diagnostic_not_later_day_entry_state"
    ]
    assert {"field": "CH4_soil", "mapped_to": "CH4_soil"} in statuses["inactive_wetland_ch4_branch"]
    assert {"field": "MatrixV", "mapped_to": "MatrixV"} in statuses["spinup_equilibrium_state"]
