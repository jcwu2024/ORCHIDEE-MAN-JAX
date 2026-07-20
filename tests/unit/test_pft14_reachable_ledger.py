from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "audit_pft14_reachable_ledger.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_reachable_ledger", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_current_pft14_reachable_ledger_passes_audit():
    document = AUDIT.load_ledger()
    assert AUDIT.validate_ledger(document) == []


def test_audit_rejects_missing_required_field_on_active_entry():
    document = copy.deepcopy(AUDIT.load_ledger())
    active = next(entry for entry in document["entries"] if entry["reachability"] == "active")
    active["same_step_outputs"] = []

    errors = AUDIT.validate_ledger(document)

    assert any(active["id"] in error and "same_step_outputs" in error for error in errors)


def test_audit_rejects_unsupported_active_or_conditional_entries():
    for reachability in ("active", "conditional"):
        document = copy.deepcopy(AUDIT.load_ledger())
        entry = next(item for item in document["entries"] if item["reachability"] == reachability)
        entry["status"] = "unsupported"

        errors = AUDIT.validate_ledger(document)

        assert any(entry["id"] in error and "cannot have status unsupported" in error for error in errors)


def test_audit_allows_explicit_inactive_unsupported_entry():
    document = copy.deepcopy(AUDIT.load_ledger())
    entry = next(item for item in document["entries"] if item["id"] == "enerbil.conditional.potential_temperature")
    entry["reachability"] = "inactive"
    entry["status"] = "unsupported"
    assert AUDIT.validate_ledger(document) == []


def test_audit_rejects_missing_priority_module():
    document = copy.deepcopy(AUDIT.load_ledger())
    document["entries"] = [entry for entry in document["entries"] if entry["module"] != "THERMOSOIL"]

    errors = AUDIT.validate_ledger(document)

    assert "priority modules without entries: THERMOSOIL" in errors


def test_ledger_covers_complete_forcing_to_modelout_production_chain():
    document = AUDIT.load_ledger()
    represented = {entry["module"] for entry in document["entries"]}

    assert represented >= AUDIT.REQUIRED_MODULES
    assert {
        "driver.active.forcing_landpoint_interpolation",
        "slowproc.active.daily_surface_writeback",
        "routing.active.paper_single_landpoint_zero_branch",
        "restart.active.day_and_year_handoff",
        "modelout.active.paper_history_formula",
    } <= {entry["id"] for entry in document["entries"]}


def test_audit_rejects_missing_required_branch_id():
    document = copy.deepcopy(AUDIT.load_ledger())
    required_id = "stomate.active.turnover"
    document["entries"] = [entry for entry in document["entries"] if entry["id"] != required_id]

    errors = AUDIT.validate_ledger(document)

    assert any("required branch ids without entries" in error and required_id in error for error in errors)


def test_audit_rejects_nonexistent_python_symbols_and_test_nodes():
    document = copy.deepcopy(AUDIT.load_ledger())
    active = next(entry for entry in document["entries"] if entry["reachability"] == "active")
    active["kernel"] = "jax_orchidee.sechiba.hydrol.not_a_real_kernel"
    active["validation_evidence"] = ["tests/unit/test_hydrol_diagnostics.py::test_not_real"]

    errors = AUDIT.validate_ledger(document)

    assert any(active["id"] in error and "kernel symbol does not exist" in error for error in errors)
    assert any(active["id"] in error and "validation_evidence target does not exist" in error for error in errors)


def test_audit_rejects_nonexistent_fortran_subroutine():
    document = copy.deepcopy(AUDIT.load_ledger())
    active = next(entry for entry in document["entries"] if entry["reachability"] == "active")
    active["fortran_subroutine"] = "not_a_real_fortran_subroutine"

    errors = AUDIT.validate_ledger(document)

    assert any(active["id"] in error and "Fortran subroutine does not exist" in error for error in errors)
