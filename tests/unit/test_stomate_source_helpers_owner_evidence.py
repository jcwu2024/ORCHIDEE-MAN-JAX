from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "outputs/reference_mode/micro_oracles/stomate_source_helpers_owner"
    / "owner_region_evidence.json"
)


def test_stomate_source_helpers_owner_evidence_is_complete() -> None:
    document = json.loads(EVIDENCE.read_text(encoding="ascii"))
    assert document["complete"] is True
    assert len(document["records"]) == 2
    records = {record["fortran_procedure"]: record for record in document["records"]}
    assert set(records) == {"sort_ascending", "stomate_var_init"}
    for record in records.values():
        assert len(record["required_arm_ids"]) == 2
        assert record["required_arm_ids"] == record["covered_arm_ids"]
        assert record["passed"] is True
