from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "build_pft14_arm_disposition.py"
SPEC = importlib.util.spec_from_file_location("build_pft14_arm_disposition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def _inventory():
    branches = [
        {"id": "f.f90:2:if"},
        {"id": "f.f90:3:if"},
        {"id": "f.f90:4:if"},
    ]
    arms = []
    for branch in branches:
        for arm_name in ("true", "false"):
            arms.append(
                {
                    "id": f"{branch['id']}:{arm_name}",
                    "branch_id": branch["id"],
                    "arm": arm_name,
                    "file": "f.f90",
                    "line": int(branch["id"].split(":")[1]),
                    "procedure_id": "f.f90:1:step",
                    "source": "if (x)",
                }
            )
    return {
        "branches": branches,
        "control_flow_arms": arms,
        "target_reachable_arm_ids": [arm["id"] for arm in arms],
    }


def test_only_narrow_structural_rules_are_automatically_classified():
    resolution = {
        "rows": [
            {
                "control_flow_id": "f.f90:2:if",
                "procedure": "step",
                "resolution": "diagnostic_only",
            },
            {
                "control_flow_id": "f.f90:3:if",
                "procedure": "step",
                "resolution": "valid_input_error_guard",
            },
            {
                "control_flow_id": "f.f90:4:if",
                "procedure": "step",
                "resolution": "process_span_mapped",
                "process_ledger_ids": ["step.active"],
            },
        ]
    }

    result = BUILD.build_arm_disposition(_inventory(), resolution)

    assert result["status_counts"] == {"classified": 4, "candidate": 2}
    assert result["disposition_counts"]["invalid_input_error"] == 1
    assert result["disposition_counts"]["valid_input_continuation"] == 1
    assert not result["closed"]


def test_unmapped_and_inventory_gap_arms_remain_open():
    inventory = _inventory()
    resolution = {
        "rows": [
            {
                "control_flow_id": "f.f90:2:if",
                "procedure": "step",
                "resolution": "process_inventory_open_gap",
                "process_inventory_gap": "missing fallback",
                "process_inventory_id": "step.init",
            },
            {
                "control_flow_id": "f.f90:3:if",
                "procedure": "step",
                "resolution": "unmapped",
            },
            {
                "control_flow_id": "f.f90:4:if",
                "procedure": "step",
                "resolution": "process_inventory_classified",
                "process_inventory_id": "step.alloc",
            },
        ]
    }

    result = BUILD.build_arm_disposition(inventory, resolution)

    assert result["status_counts"] == {"open": 4, "classified": 2}
    assert any(record["reason"] == "missing fallback" for record in result["records"])
    assert not result["closed"]


def test_static_boolean_evaluator_accepts_only_pure_fixed_flag_conditions():
    flags = {"ok_leak": True, "ok_pc": False}

    assert BUILD.evaluate_static_boolean_condition("IF (ok_leak) THEN", flags) is True
    assert BUILD.evaluate_static_boolean_condition("IF (.NOT. ok_pc .AND. ok_leak) THEN", flags) is True
    assert BUILD.evaluate_static_boolean_condition("IF (ok_leak .AND. soil_moisture > 0) THEN", flags) is None
    assert BUILD.evaluate_static_boolean_condition("WHERE (ok_leak)", flags) is None


def test_fixed_config_classifies_taken_and_inactive_arms_even_when_process_is_open():
    inventory = _inventory()
    for arm in inventory["control_flow_arms"]:
        if arm["branch_id"] == "f.f90:2:if":
            arm["source"] = "IF (ok_leak) THEN"
    resolution = {
        "rows": [
            {"control_flow_id": "f.f90:2:if", "procedure": "step", "resolution": "unmapped"},
            {"control_flow_id": "f.f90:3:if", "procedure": "step", "resolution": "unmapped"},
            {"control_flow_id": "f.f90:4:if", "procedure": "step", "resolution": "unmapped"},
        ]
    }

    result = BUILD.build_arm_disposition(inventory, resolution, static_flags={"ok_leak": True})
    records = {record["arm_id"]: record for record in result["records"]}

    assert records["f.f90:2:if:true"]["disposition"] == "paper_static_active"
    assert records["f.f90:2:if:false"]["disposition"] == "paper_static_inactive"
    assert result["status_counts"] == {"classified": 2, "open": 4}


def test_project_static_flags_include_materialized_run_def_booleans():
    flags = BUILD.load_static_boolean_flags()

    assert flags["ok_leak"] is True
    assert flags["river_routing"] is True
    assert flags["land_cover_change"] is False
    assert flags["chemistry_bvoc"] is False
    assert flags["tides"] is True

    _, conflicts = BUILD.load_static_boolean_context()
    assert "tides" not in conflicts


def test_manual_rules_require_real_arm_ids_and_override_candidate_status(tmp_path):
    inventory = _inventory()
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """schema_version: 1
rules:
  - branch_id: f.f90:2:if
    decisions:
      true:
        disposition: pft14_conditional
        reason: Forcing can make x true for PFT14.
      false:
        disposition: pft14_conditional
        reason: Forcing can make x false for PFT14.
""",
        encoding="utf-8",
    )
    manual = BUILD.load_manual_arm_rules(inventory, [rules_path])
    resolution = {
        "rows": [
            {"control_flow_id": "f.f90:2:if", "procedure": "step", "resolution": "process_span_mapped"},
            {"control_flow_id": "f.f90:3:if", "procedure": "step", "resolution": "unmapped"},
            {"control_flow_id": "f.f90:4:if", "procedure": "step", "resolution": "unmapped"},
        ]
    }

    result = BUILD.build_arm_disposition(inventory, resolution, manual_rules=manual)

    assert result["status_counts"] == {"classified": 2, "open": 4}
    assert result["manual_rule_arms"] == 2
    assert result["manual_rule_original_status_counts"] == {"candidate": 2}
    assert result["records"][0]["classification_source"].endswith("rules.yaml")


def test_manual_rule_rejects_unknown_arm(tmp_path):
    rules_path = tmp_path / "bad.yaml"
    rules_path.write_text(
        """schema_version: 1
rules:
  - branch_id: f.f90:2:if
    decisions:
      case:
        disposition: pft14_active
        reason: Invalid arm for this IF.
""",
        encoding="utf-8",
    )

    try:
        BUILD.load_manual_arm_rules(_inventory(), [rules_path])
    except ValueError as exc:
        assert "unknown arm" in str(exc)
    else:
        raise AssertionError("unknown manual arm should fail")
