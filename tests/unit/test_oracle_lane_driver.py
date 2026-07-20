from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dev"))
SCRIPT = ROOT / "scripts/dev/oracle_lane_driver_run.py"
FRAGMENT = ROOT / "docs/source_audits/oracle_families/driver_lifecycle.yaml"


def _module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("oracle_lane_driver_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_driver_fragment_contract_and_lane_coverage():
    document = yaml.safe_load(FRAGMENT.read_text(encoding="ascii"))
    assert document["schema_version"] == 2
    required = {
        "id",
        "runner",
        "ledger_entries",
        "procedures",
        "output_contract",
    }
    entries = set()
    for family in document["families"]:
        assert required <= family.keys()
        entries.update(family["ledger_entries"])
        assert ("branch_cases" in family) != ("branch_coverage_asset" in family)
        assert "result_asset" in family or {
            "input_asset", "comparison_asset"
        } <= family.keys()
        for case in family.get("branch_cases", []):
            assert {
                "id",
                "ledger_entry",
                "fortran_lines",
                "condition",
                "expected_arm",
                "input_case",
                "control_flow_arm_ids",
            } <= case.keys()
        assert family["procedures"]
        for procedure in family["procedures"]:
            assert len(procedure["expected_source_sha256"]) == 64
            assert len(procedure["expected_span_sha256"]) == 64
            assert "use_modules" in procedure["dependencies"]
            assert procedure["compilation"]["status"] == "passed"
            assert procedure["compilation"]["standalone_compilable"] is False
            assert procedure["compilation"]["compiler_available"] is True
            assert procedure["numerical_oracle_status"] == "verified"
    assert entries == {
        "routing.active.paper_single_landpoint_zero_branch",
        "modelout.active.paper_history_formula",
        "slowproc.active.daily_surface_writeback",
        "driver.active.salinity_bbox",
        "driver.active.tide_bbox",
        "sechiba.active.module_order",
        "driver.active.forcing_landpoint_interpolation",
    }
    pending_entries = {
        entry
        for family in document["pending_families"]
        for entry in family["ledger_entries"]
    }
    assert pending_entries == {
        "driver.active.forcing_landpoint_interpolation",
        "restart.active.day_and_year_handoff",
    }
    pending = {family["id"]: family for family in document["pending_families"]}
    assert pending["driver_restart_handoff"]["evidence_kind"] == "schema_and_encoding_partial"
    assert {family["certification_status"] for family in pending.values()} == {"superseded"}
    assert all((ROOT / family["superseded_by"]).is_file() for family in pending.values())


def test_driver_source_hashes_are_current():
    module = _module()
    for family_id in (
        "driver_routing_zero",
        "driver_forcing_transition",
        "driver_slowproc_daily",
        "driver_static_bbox",
        "driver_modelout_history",
        "driver_sechiba_module_order",
        "driver_restart_handoff",
    ):
        assert module._verify_procedures(module._manifest_family(family_id))


def test_cached_driver_completed_family_result_is_passing_and_complete():
    document = yaml.safe_load(FRAGMENT.read_text(encoding="ascii"))
    for family in document["families"]:
        path = ROOT / family.get(
            "result_asset",
            f"outputs/reference_mode/micro_oracles/{family['id']}/comparison.json",
        )
        result = json.loads(path.read_text(encoding="ascii"))
        assert result["status"] == "passed"
        assert result["comparisons"] and all(
            item["passed"] for item in result["comparisons"]
        )


def test_driver_formal_families_are_source_arm_complete():
    from audit_fortran_oracle_coverage import _required_arms_by_entry

    document = yaml.safe_load(FRAGMENT.read_text(encoding="ascii"))
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
            if "branch_coverage_asset" in family:
                coverage = json.loads(
                    (ROOT / family["branch_coverage_asset"]).read_text(encoding="utf-8")
                )
                covered = {
                    item["arm_id"]
                    for item in coverage["ledger_entries"][entry]
                    if item.get("case_ids") or item.get("input_cases")
                }
            else:
                covered = {
                    arm
                    for case in family["branch_cases"]
                    if case["ledger_entry"] == entry
                    for arm in case["control_flow_arm_ids"]
                }
            assert covered == required[entry]
