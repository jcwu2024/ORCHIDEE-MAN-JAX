import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/reference_mode/micro_oracles/interpolation_aggregate_owners"


def test_interpolation_aggregate_owner_closure_is_strict_and_complete():
    comparison = json.loads((OUT / "comparison.json").read_text(encoding="ascii"))
    coverage = json.loads((OUT / "branch_coverage.json").read_text(encoding="ascii"))
    owners = json.loads(
        (OUT / "owner_region_evidence.json").read_text(encoding="ascii")
    )
    assert comparison["status"] == "passed"
    assert all(
        x.get("comparison") == "exact" or (x["rtol"] <= 1e-12 and x["atol"] <= 1e-12)
        for x in comparison["comparisons"]
    )
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 99
    assert coverage["branch_complete"] is True
    assert owners["complete"] is True
    assert {x["owner_region_id"] for x in owners["records"]} == {
        "pft14-owner-contract-3479c5bbdcd5",
        "pft14-owner-contract-9561ebb9666b",
        "pft14-owner-contract-633db67a9e88",
        "pft14-owner-contract-740908287844",
    }
    assert all(
        set(record["covered_arm_ids"])
        == set(record["gcov_arm_ids"]) | set(record["source_proof_arm_ids"])
        for record in owners["records"]
    )
