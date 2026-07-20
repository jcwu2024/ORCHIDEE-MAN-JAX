from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dev"))


def _manifest():
    return yaml.safe_load(
        (
            ROOT / "docs/source_audits/oracle_families/surface_energy_condveg.yaml"
        ).read_text(encoding="utf-8")
    )


def test_surface_lane_fragment_contract_and_case_gate():
    document = _manifest()
    assert document["schema_version"] == 2
    for family in document["families"]:
        covered = set()
        assert {
            "id",
            "runner",
            "ledger_entries",
            "procedures",
            "input_asset",
            "comparison_asset",
            "output_contract",
            "branch_cases",
        } <= family.keys()
        cases = family["branch_cases"]
        assert cases
        for case in cases:
            assert {
                "id",
                "ledger_entry",
                "fortran_lines",
                "condition",
                "expected_arm",
                "input_case",
                "control_flow_arm_ids",
            } <= case.keys()
            covered.add(case["ledger_entry"])
        assert set(family["ledger_entries"]) <= covered
        assert family["certification_status"] == "verified"


def test_surface_lane_source_hashes_are_current():
    from extract_fortran_micro_oracle import extract_procedure_bytes

    for family in _manifest()["families"]:
        for record in family["procedures"]:
            span = extract_procedure_bytes(
                ROOT / record["source_file"], record["procedure"]
            )
            assert span.start_line == record["start_line"]
            assert span.end_line == record["end_line"]
            assert span.source_sha256 == record["expected_source_sha256"]
            assert span.span_sha256 == record["expected_span_sha256"]


def test_surface_lane_verified_assets_actually_passed():
    for family in _manifest()["families"]:
        assert all(
            item["numerical_oracle_status"] == "verified"
            for item in family["procedures"]
        )
        result = json.loads(
            (ROOT / family["procedures"][0]["compilation"]["result_asset"]).read_text(
                encoding="ascii"
            )
        )
        assert result["status"] == "passed"
        assert set(result["ledger_entries"]) == set(family["ledger_entries"])
        assert all(item["passed"] for item in result["comparisons"])


def test_surface_lane_pending_work_is_outside_families():
    document = _manifest()
    active_entries = {
        entry for family in document["families"] for entry in family["ledger_entries"]
    }
    pending_entries = {
        entry
        for family in document["pending_families"]
        for entry in family["ledger_entries"]
    }
    assert active_entries.isdisjoint(pending_entries)
    assert pending_entries == set()
    assert all(
        family["certification_status"] == "verified"
        for family in document["families"]
    )
    assert all(
        family["certification_status"] == "pending"
        and family.get("pending_reason")
        for family in document["pending_families"]
    )


def test_surface_formal_families_are_source_arm_complete():
    from audit_fortran_oracle_coverage import _required_arms_by_entry

    document = _manifest()
    ledger = yaml.safe_load(
        (ROOT / "docs/source_audits/pft14_reachable_ledger.yaml").read_text(
            encoding="utf-8"
        )
    )
    inventory = json.loads(
        (ROOT / "outputs/reference_mode/fortran_control_flow_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    disposition = json.loads(
        (ROOT / "outputs/reference_mode/pft14_arm_disposition.json").read_text(
            encoding="utf-8"
        )
    )
    required = _required_arms_by_entry(ledger, inventory, disposition)
    for family in document["families"]:
        for entry in family["ledger_entries"]:
            covered = {
                arm
                for case in family["branch_cases"]
                if case["ledger_entry"] == entry
                for arm in case["control_flow_arm_ids"]
            }
            assert covered == required[entry]


def test_surface_fvcb_comparison_covers_all_declared_outputs():
    records = _manifest()["families"] + _manifest()["pending_families"]
    family = next(
        item for item in records if item["id"] == "surface_diffuco_co2_fvcb"
    )
    result = json.loads(
        (
            ROOT
            / "outputs/reference_mode/micro_oracles/surface_diffuco_co2_fvcb/comparison.json"
        ).read_text(encoding="ascii")
    )
    assert result["status"] == "passed"
    assert {item["name"] for item in result["comparisons"]} == set(
        family["output_contract"]
    )


def test_surface_enerbil_main_comparison_covers_all_declared_outputs():
    records = _manifest()["families"] + _manifest()["pending_families"]
    family = next(
        item for item in records if item["id"] == "surface_enerbil_main"
    )
    result = json.loads(
        (
            ROOT
            / "outputs/reference_mode/micro_oracles/surface_enerbil_main/comparison.json"
        ).read_text(encoding="ascii")
    )
    assert result["status"] == "passed"
    assert {item["name"] for item in result["comparisons"]} == set(
        family["output_contract"]
    )
