from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "audit_pft14_evidence_scope.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_evidence_scope", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_current_evidence_scope_covers_every_ledger_entry():
    ledger = yaml.safe_load(AUDIT.DEFAULT_LEDGER.read_text(encoding="utf-8"))
    scope = yaml.safe_load(AUDIT.DEFAULT_SCOPE.read_text(encoding="utf-8"))
    assert AUDIT.validate_scope(ledger, scope) == []
    oracle_entries = [
        item
        for item in scope["entries"].values()
        if item["kind"] == "fortran_micro_oracle"
    ]
    assert len(oracle_entries) == 48
    assert all(item.get("evidence") for item in oracle_entries)
