from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/reference_mode/micro_oracles/stage4_output_lifecycle/owner_region_evidence.json"
OWNERS = {
    "pft14-owner-contract-6fef46ca099f": 42,
    "pft14-owner-contract-ca05f39256c2": 11,
}
FIELDS = (
    "LEAF_M", "SAP_M_AB", "HEART_M_AB", "AGR_SAP_ST_M", "AGR_HRT_ST_M",
    "AGR_SAP_PN_M", "AGR_HRT_PN_M", "SAP_M_BE", "HEART_M_BE", "ROOT_M",
    "GPP", "NPP",
)
REFERENCE_ORDER = (
    "NPP", "GPP", "LEAF_M", "SAP_M_AB", "SAP_M_BE", "HEART_M_AB",
    "HEART_M_BE", "ROOT_M", "AGR_SAP_ST_M", "AGR_SAP_PN_M",
    "AGR_HRT_ST_M", "AGR_HRT_PN_M",
)


def test_output_lifecycle_owners_have_source_proofs_and_accepted_history_contract() -> None:
    document = json.loads(EVIDENCE.read_text(encoding="ascii"))
    assert document["policy"] == "output_acceptance_and_source_contract"
    assert document["complete"] is True
    assert {record["owner_region_id"] for record in document["records"]} == set(OWNERS)
    assert all(record["passed"] for record in document["records"])
    for record in document["records"]:
        assert len(record["required_arm_ids"]) == OWNERS[record["owner_region_id"]]
        assert record["covered_arm_ids"] == record["required_arm_ids"]
        assert all(proof["source_control_flow_verified"] for proof in document["source_arm_proofs"][record["owner_region_id"]])

    history = document["history_acceptance"]
    assert history["accepted"] is True
    assert set(history["field_order"]) == set(FIELDS)
    assert history["field_order"] == list(REFERENCE_ORDER)
    assert all(value == ["time_counter", "veget", "lat", "lon"] for value in history["dimensions"].values())
    assert all(value == [1, 14, 1, 1] for value in history["shapes"].values())
    assert all(value == "average" for value in (attrs["online_operation"] for attrs in history["attributes"].values()))
    assert all(value == "1800 s" for value in (attrs["interval_operation"] for attrs in history["attributes"].values()))
    assert all(value == "1 yr" for value in (attrs["interval_write"] for attrs in history["attributes"].values()))
    assert all(value != "None" for value in history["fill_values"].values())
    assert document["jax_contracts"]["accepted"] is True
    assert document["jax_contracts"]["modelout_formula_order"] == list(FIELDS)
    assert document["jax_contracts"]["xios_switch_disables_ioipsl_streams"] == [False] * 4
