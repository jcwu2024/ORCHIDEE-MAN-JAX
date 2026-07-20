from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stage4_microactem_snowlevels"
OWNER_COUNTS = {"pft14-owner-contract-1211208a2520": 25, "pft14-owner-contract-6a1aba752eed": 4}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_microactem_snowlevels_stage4_evidence_is_pinned_and_complete() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text())
    evidence = json.loads((OUTPUT / "owner_region_evidence.json").read_text())
    fatal = json.loads((OUTPUT / "fatal_boundary_evidence.json").read_text())
    assert comparison["status"] == "passed"
    assert all(item["passed"] for item in comparison["comparisons"])
    assert _sha256(ROOT / comparison["source"]["file"]) == comparison["source"]["sha256"]
    for name, record in comparison["source"]["procedures"].items():
        assert record["exact_extracted_bytes"]
        assert _sha256(OUTPUT / f"{name}.f90") == record["sha256"]
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 29
    assert not coverage["missing_arm_ids"]
    assert fatal["complete"] and all(record["passed"] for record in fatal["records"])
    records = {record["owner_region_id"]: record for record in evidence["records"]}
    assert evidence["complete"] and set(records) == set(OWNER_COUNTS)
    for owner_id, arm_count in OWNER_COUNTS.items():
        assert records[owner_id]["passed"]
        assert len(records[owner_id]["required_arm_ids"]) == arm_count
        assert records[owner_id]["covered_arm_ids"] == records[owner_id]["required_arm_ids"]
