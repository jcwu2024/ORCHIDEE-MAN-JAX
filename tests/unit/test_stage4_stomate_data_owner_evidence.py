"""Fail-closed evidence checks for the Stage4 ``stomate_data::data`` owner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/build_stage4_stomate_data_owner_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage4_stomate_data_owner_evidence", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema3_capture_derives_the_complete_pft14_owner_arm_set():
    module = _module()
    evidence = module.build()
    record = evidence["records"][0]

    assert evidence["complete"] is True
    assert set(record["required_arm_ids"]) == set(record["covered_arm_ids"])
    assert len(record["covered_arm_ids"]) == 16
    assert record["evidence"]["original_procedure_blocks"]["covered"] == 428
    assert record["evidence"]["original_procedure_blocks"]["total"] == 446
    assert record["evidence"]["pft14_predicates"]["408"] == "true"
    assert record["evidence"]["pft14_predicates"]["422"] == "false"


def test_committed_owner_evidence_is_regenerated_from_current_assets():
    module = _module()
    committed = json.loads((module.FAMILY / "owner_region_evidence.json").read_text(encoding="utf-8"))
    assert committed == module.build()
