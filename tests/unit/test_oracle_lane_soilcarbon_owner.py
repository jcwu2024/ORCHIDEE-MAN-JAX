from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_soilcarbon_owner_fragment_records_branch_complete_gate():
    path = ROOT / "docs/source_audits/oracle_families/soilcarbon_owner.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["status"] == "passed"
    family = document["families"][0]
    states = {item["ledger_entry"]: item["status"] for item in family["components"]}
    assert states == {
        "stomate.active.tf_doc": "branch_complete_fortran_oracle",
        "stomate.active.ok_leak_firstcall_active_layer": "branch_complete_fortran_oracle",
        "stomate.active.ok_leak_soilcarbon": "branch_complete_fortran_oracle",
    }


def test_soilcarbon_owner_arm_asset_has_cases_for_every_owner_arm():
    path = ROOT / "outputs/reference_mode/micro_oracles/stomate_soilcarbon_owner/arm_coverage.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    arms = [
        item
        for records in document["ledger_entries"].values()
        for item in records
    ] + document["owner_only_arms"]
    assert document["branch_complete"] is True
    assert len({item["arm_id"] for item in arms}) == 237
    assert all(item["case_ids"] and item["comparison_asset"] for item in arms)


def test_soilcarbon_owner_comparison_closes_first_and_later_calls():
    path = ROOT / "outputs/reference_mode/micro_oracles/stomate_soilcarbon_owner/comparison.json"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["status"] == "passed"
    assert {item["scenario"] for item in document["comparisons"]} == {
        "normal",
        "newaltcalc_spinup",
    }
    assert {item["call"] for item in document["comparisons"]} == {1, 2, 3, 4}
    assert all(item["passed"] for item in document["comparisons"])
