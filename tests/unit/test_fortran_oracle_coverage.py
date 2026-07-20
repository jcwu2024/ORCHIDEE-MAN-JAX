from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "audit_fortran_oracle_coverage.py"


def _module():
    scripts = str(ROOT / "scripts" / "dev")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("oracle_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_family_is_not_misreported_as_stage_three_complete():
    module = _module()
    manifest = deepcopy(module.load_manifest())
    manifest["families"] = [
        family
        for family in manifest["families"]
        if family["id"] != "diffuco_surface_exchange"
    ]
    ledger = yaml.safe_load(module.DEFAULT_LEDGER.read_text(encoding="utf-8"))
    inventory = json.loads(module.DEFAULT_INVENTORY.read_text(encoding="utf-8"))
    disposition = json.loads(module.DEFAULT_DISPOSITION.read_text(encoding="utf-8"))
    report = module.build_report(
        manifest, ledger, inventory=inventory, disposition=disposition
    )
    assert report["reachable_entries"] == 48
    assert "diffuco.active.aerodynamic_boundary" in report["missing_entries"]
    assert not report["complete"]


def test_inventory_mapping_requires_real_scientific_arm_ids():
    module = _module()
    manifest = module.load_manifest(fragment_dir=None)
    ledger = yaml.safe_load(module.DEFAULT_LEDGER.read_text(encoding="utf-8"))
    inventory = json.loads(module.DEFAULT_INVENTORY.read_text(encoding="utf-8"))
    disposition = json.loads(module.DEFAULT_DISPOSITION.read_text(encoding="utf-8"))
    report = module.build_report(
        manifest, ledger, inventory=inventory, disposition=disposition
    )
    by_entry = {item["ledger_entry"]: item for item in report["records"]}
    aero = by_entry["diffuco.active.aerodynamic_boundary"]
    assert aero["required_arm_count"] > 0
    assert aero["covered_arm_count"] == aero["required_arm_count"]
    assert not aero["missing_arm_ids"]
    assert aero["branch_coverage_passed"]

    aero_family = next(
        family for family in manifest["families"] if family["id"] == aero["family"]
    )
    aero_cases = [
        case
        for case in aero_family["branch_cases"]
        if case.get("ledger_entry") == "diffuco.active.aerodynamic_boundary"
    ]
    original_arm_ids = [case.get("control_flow_arm_ids", []) for case in aero_cases]
    try:
        for case in aero_cases:
            case["control_flow_arm_ids"] = ["synthetic:not-an-inventory-arm"]
        rejected = module.build_report(
            manifest, ledger, inventory=inventory, disposition=disposition
        )
    finally:
        for case, arm_ids in zip(aero_cases, original_arm_ids, strict=True):
            case["control_flow_arm_ids"] = arm_ids

    rejected_aero = next(
        item
        for item in rejected["records"]
        if item["ledger_entry"] == "diffuco.active.aerodynamic_boundary"
    )
    assert rejected_aero["missing_arm_ids"]
    assert rejected_aero["unknown_covered_arm_ids"] == [
        "synthetic:not-an-inventory-arm"
    ]
    assert not rejected_aero["branch_coverage_passed"]
