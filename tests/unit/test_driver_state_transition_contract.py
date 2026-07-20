from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "audit_driver_state_transition_contract.py"
SPEC = importlib.util.spec_from_file_location("audit_driver_state_transition_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_stomate_boundary_required_fields_have_declared_producers():
    audit = AUDIT.build_audit()
    assert audit["closed"]
    assert audit["errors"]["later_day_fields_without_producer"] == []
    assert audit["errors"]["year_handoff_fields_without_producer"] == []


def test_known_daily_memories_are_day_boundary_only_not_half_hour_carry():
    audit = AUDIT.build_audit()
    day_only = set(audit["sets"]["day_boundary_only"])
    assert {"turnover_longterm", "lm_lastyearmax", "fixed_cryoturbation_depth"} <= day_only
