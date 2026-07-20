from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORACLES = ROOT / "outputs/reference_mode/micro_oracles"


def _load(family: str, asset: str) -> dict[str, object]:
    return json.loads((ORACLES / family / asset).read_text(encoding="utf-8"))


def test_latest_stage4_owner_families_have_complete_evidence() -> None:
    expected = {
        "xios_ranked_send": (0, 5),
        "gauss_jordan_owner": (17, 2),
        "haversine_primitives": (12, 4),
        "polygon_primitives": (8, 3),
        "condveg_finalize_state_io": (0, 1),
    }

    for family, (dynamic_arms, owner_count) in expected.items():
        comparison = _load(family, "comparison.json")
        coverage = _load(family, "branch_coverage.json")
        owners = _load(family, "owner_region_evidence.json")

        assert comparison["status"] == "passed"
        assert coverage["branch_complete"]
        assert coverage["required_arm_count"] == dynamic_arms
        assert coverage["covered_arm_count"] == dynamic_arms
        assert owners["complete"]
        assert len(owners["records"]) == owner_count
        assert all(record["passed"] for record in owners["records"])


def test_state_and_transport_source_proofs_still_have_numerical_oracles() -> None:
    for family in ("xios_ranked_send", "condveg_finalize_state_io"):
        owners = _load(family, "owner_region_evidence.json")
        assert all(record["source_proof_arm_ids"] for record in owners["records"])
        assert all(record["comparison_asset"] for record in owners["records"])

