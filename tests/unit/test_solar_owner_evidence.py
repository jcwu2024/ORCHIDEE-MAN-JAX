from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/reference_mode/micro_oracles/solar_owners"


def _load(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_solar_owner_evidence_closes_all_three_regions_and_thirty_arms() -> None:
    comparison = _load("comparison.json")
    coverage = _load("branch_coverage.json")
    owners = _load("owner_region_evidence.json")

    assert comparison["status"] == "passed"
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == 30
    assert coverage["covered_arm_count"] == 30
    assert owners["complete"]
    assert len(owners["records"]) == 3
    assert all(record["passed"] for record in owners["records"])


def test_solar_evidence_is_source_hashed_and_strict() -> None:
    comparison = _load("comparison.json")
    assert comparison["source_file_sha256"]
    assert set(comparison["procedure_span_sha256"]) == {
        "solarang",
        "time_zone",
        "downward_solar_flux",
    }
    for record in comparison["comparisons"]:
        if record.get("comparison") == "exact":
            continue
        assert record["rtol"] <= 1e-12
        assert record["atol"] <= 1e-12
