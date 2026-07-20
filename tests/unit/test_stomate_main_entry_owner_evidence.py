from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/reference_mode/micro_oracles/stomate_main_entry_owner/owner_region_evidence.json"


def test_stomate_main_entry_owner_evidence_is_complete() -> None:
    document = json.loads(EVIDENCE.read_text(encoding="ascii"))
    assert document["complete"] is True
    assert len(document["records"]) == 1
    record = document["records"][0]
    assert record["owner_region_id"] == "pft14-owner-contract-5aa15ce8a73e"
    assert len(record["required_arm_ids"]) == 25
    assert record["required_arm_ids"] == record["covered_arm_ids"]
