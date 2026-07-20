"""Regression gates for the live STOMATE restart owner oracle."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/oracle_stage4_restart_io.py"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stage4_restart_io"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_restart_io_oracle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_fortran_restart_assets_recompute_without_mismatch() -> None:
    module = _load_module()
    assert module.main() == 0
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["passed"] is True
    assert comparison["comparison_count"] == 335
    assert comparison["mismatch_count"] == 0
    assert (OUTPUT / "cold_stomate_restart.nc").is_file()
    assert (OUTPUT / "distinct_stomate_restart.nc").is_file()


def test_restart_owner_evidence_closes_only_the_two_real_io_owners() -> None:
    evidence = json.loads((OUTPUT / "owner_region_evidence.json").read_text(encoding="utf-8"))
    assert evidence["complete"] is True
    records = {record["owner_region_id"]: record for record in evidence["records"]}
    assert set(records) == {
        "pft14-owner-contract-91683a83999b",
        "pft14-owner-contract-e5b66af5d494",
    }
    for record in records.values():
        assert record["passed"] is True
        assert record["required_arm_ids"] == record["covered_arm_ids"]
        assert record["required_arm_ids"]
