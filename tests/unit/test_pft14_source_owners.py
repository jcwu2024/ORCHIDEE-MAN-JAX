from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "audit_pft14_source_owners.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_source_owners", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _inventory():
    return {
        "procedures": [
            {"file": "f.f90", "name": "step", "start_line": 1, "end_line": 10},
        ]
    }


def test_missing_owner_must_remain_explicit_gap():
    entry = {
        "id": "step.missing",
        "fortran_file": "f.f90",
        "fortran_subroutine": "step",
        "fortran_lines": "2-5",
        "status": "missing",
        "jax_owner": None,
        "gap": "No JAX kernel owns this process yet.",
        "state_contract": [],
        "validation": [],
    }

    assert AUDIT.validate_entries([entry], _inventory()) == []


def test_implemented_owner_requires_real_symbol_state_and_validation():
    entry = {
        "id": "step.bad",
        "fortran_file": "f.f90",
        "fortran_subroutine": "step",
        "fortran_lines": "2-5",
        "status": "implemented",
        "jax_owner": "jax_orchidee.missing.nope",
        "gap": None,
        "state_contract": [],
        "validation": [],
    }

    errors = AUDIT.validate_entries([entry], _inventory())

    assert any("owner does not exist" in error for error in errors)
    assert any("state_contract" in error for error in errors)
    assert any("validation" in error for error in errors)


def test_owner_evidence_paths_must_exist():
    entry = {
        "id": "step.partial",
        "fortran_file": "f.f90",
        "fortran_subroutine": "step",
        "fortran_lines": "2-5",
        "status": "partial",
        "jax_owner": "jax_orchidee.driver.init.RunScalars",
        "gap": "Only a subset is implemented.",
        "state_contract": ["docs/source_audits/does_not_exist.md"],
        "validation": ["tests/unit/does_not_exist.py"],
    }

    errors = AUDIT.validate_entries([entry], _inventory())

    assert any("validation path does not exist" in error for error in errors)
    assert any("state-contract path does not exist" in error for error in errors)
