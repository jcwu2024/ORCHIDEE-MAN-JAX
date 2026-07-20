from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "build_pft14_control_flow_resolution.py"
SPEC = importlib.util.spec_from_file_location("build_pft14_control_flow_resolution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


def test_resolution_keeps_per_branch_denominator_and_does_not_overclaim_process_mapping():
    procedure_id = "f.f90:1:step"
    inventory = {
        "procedures": [{"id": procedure_id, "name": "step"}],
        "branches": [
            {"id": "f.f90:2:if", "file": "f.f90", "line": 2, "kind": "if", "source": "if (x)", "procedure_id": procedure_id},
            {"id": "f.f90:5:if", "file": "f.f90", "line": 5, "kind": "if", "source": "if (y)", "procedure_id": procedure_id},
        ],
        "reachable_branch_ids": ["f.f90:2:if", "f.f90:5:if"],
    }
    ledger = {
        "entries": [
            {"id": "step.active", "fortran_file": "f.f90", "fortran_subroutine": "step", "fortran_lines": "1-3"}
        ]
    }
    scope = {"entries": {"step.active": {"kind": "point_trace_anchor", "scope": "one arm"}}}

    result = BUILD.build_resolution(inventory, ledger, scope)

    assert result["reachable_denominator"] == 2
    assert result["resolution_counts"] == {"process_span_mapped": 1, "unmapped": 1}
    assert result["rows"][0]["evidence_kinds"] == ["point_trace_anchor"]
    assert not result["closed"]


def test_structural_resolution_classifies_only_narrow_error_and_diagnostic_guards():
    assert BUILD._structural_resolution("IF (ier /= 0) CALL ipslerr_p(...) ") == "valid_input_error_guard"
    assert BUILD._structural_resolution("IF (l_error) THEN") == "valid_input_error_guard"
    assert BUILD._structural_resolution("IF (printlev >= 3) WRITE(*,*) x") == "diagnostic_only"
    assert BUILD._structural_resolution("IF (soil_moisture > threshold) THEN") is None


def test_extension_inventory_organizes_but_does_not_hide_open_process_gaps():
    procedure_id = "f.f90:1:init"
    inventory = {
        "procedures": [{"id": procedure_id, "name": "init"}],
        "branches": [
            {
                "id": "f.f90:2:if",
                "file": "f.f90",
                "line": 2,
                "kind": "if",
                "source": "if (x)",
                "procedure_id": procedure_id,
            }
        ],
        "reachable_branch_ids": ["f.f90:2:if"],
    }
    extension = {
        "entries": [
            {
                "id": "init.active",
                "category": "scientific_process",
                "fortran_file": "f.f90",
                "fortran_subroutine": "init",
                "gap": "one arm is not mapped",
            }
        ]
    }

    result = BUILD.build_resolution(inventory, {"entries": []}, {"entries": {}}, extension)

    assert result["resolution_counts"] == {"process_inventory_open_gap": 1}
    assert result["open_denominator"] == 1
    assert result["rows"][0]["process_inventory_id"] == "init.active"
    assert not result["closed"]


def test_only_implemented_exact_source_owner_closes_resolution():
    procedure_id = "f.f90:1:step"
    inventory = {
        "procedures": [{"id": procedure_id, "name": "step"}],
        "branches": [
            {"id": "f.f90:2:if", "file": "f.f90", "line": 2, "kind": "if", "source": "if (x)", "procedure_id": procedure_id},
            {"id": "f.f90:5:if", "file": "f.f90", "line": 5, "kind": "if", "source": "if (y)", "procedure_id": procedure_id},
        ],
        "reachable_branch_ids": ["f.f90:2:if", "f.f90:5:if"],
    }
    owners = {
        "entries": [
            {"id": "step.done", "fortran_file": "f.f90", "fortran_subroutine": "step", "fortran_lines": "1-3", "status": "implemented", "jax_owner": "pkg.step"},
            {"id": "step.partial", "fortran_file": "f.f90", "fortran_subroutine": "step", "fortran_lines": "4-6", "status": "partial", "jax_owner": "pkg.partial", "gap": "false arm"},
        ]
    }

    result = BUILD.build_resolution(
        inventory, {"entries": []}, {"entries": {}}, source_owners=owners
    )

    assert result["resolution_counts"] == {
        "source_owner_mapped": 1,
        "source_owner_open_gap": 1,
    }
    assert result["open_denominator"] == 1
    assert not result["closed"]
