from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORACLES = ROOT / "outputs/reference_mode/micro_oracles"


def _load(family: str, asset: str) -> dict[str, object]:
    return json.loads((ORACLES / family / asset).read_text(encoding="utf-8"))


def test_remaining_polygon_and_haversine_owners_have_complete_evidence() -> None:
    expected = {
        "polygon_advanced_owners": (58, 5),
        "haversine_grid_owners": (62, 3),
    }

    for family, (required_arms, owner_count) in expected.items():
        comparison = _load(family, "comparison.json")
        coverage = _load(family, "branch_coverage.json")
        owners = _load(family, "owner_region_evidence.json")

        assert comparison["status"] == "passed"
        assert coverage["branch_complete"]
        assert coverage["required_arm_count"] == required_arms
        assert coverage["covered_arm_count"] == required_arms
        assert owners["complete"]
        assert len(owners["records"]) == owner_count
        assert all(record["passed"] for record in owners["records"])


def test_remaining_geometry_evidence_retains_source_hashes_and_strict_policy() -> None:
    for family in ("polygon_advanced_owners", "haversine_grid_owners"):
        comparison = _load(family, "comparison.json")
        assert comparison["source_file_sha256"]
        assert comparison["procedure_span_sha256"]
        for record in comparison["comparisons"]:
            if record.get("comparison") == "exact":
                continue
            assert record["rtol"] <= 1e-12
            assert record["atol"] <= 1e-12
