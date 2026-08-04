from __future__ import annotations

import json

import pytest

from research.daily_coarse_graining.daily_flux_label_inventory import (
    DEFAULT_INVENTORY_PATH,
    audit_daily_flux_label_inventory,
    load_daily_flux_label_inventory,
)


def _synthetic_contract(inventory):
    collections = {
        "state_leaves": {},
        "fast_day_target_leaves": {},
        "diagnostic_leaves": {},
        "native_forcing": set(),
    }
    for label in inventory.labels:
        for ref in label.contract_refs:
            if ref.collection == "diagnostic_leaves":
                collections[ref.collection][ref.key] = {"name": ref.key}
            elif ref.collection == "native_forcing":
                collections[ref.collection].add(ref.key)
            else:
                collections[ref.collection][ref.key] = {"key": ref.key}
    return {
        "schema_version": inventory.teacher_contract_schema,
        "state_leaves": list(collections["state_leaves"].values()),
        "fast_day_target_leaves": list(collections["fast_day_target_leaves"].values()),
        "diagnostic_leaves": list(collections["diagnostic_leaves"].values()),
        "native_forcing": {"fields": sorted(collections["native_forcing"])},
    }


def test_inventory_is_complete_three_way_classification() -> None:
    inventory = load_daily_flux_label_inventory()

    assert inventory.status == "frozen_inventory"
    assert {label.domain for label in inventory.labels} == {"water", "carbon", "energy"}
    assert {label.availability for label in inventory.labels} == {
        "present",
        "exactly_derivable",
        "missing_non_identifiable",
    }
    assert inventory.capture_family_ids == (
        "water_transfer_daily_v1",
        "ok_leak_transfer_daily_v1",
        "energy_flux_daily_v1",
    )
    assert all(label.source_owner for label in inventory.labels)


def test_inventory_records_endpoint_non_identifiability_instead_of_fake_derivation() -> None:
    inventory = load_daily_flux_label_inventory()
    labels = inventory.labels_by_id

    for label_id in (
        "water.transpiration",
        "water.runoff",
        "water.drainage",
        "carbon.litter_respiration",
        "carbon.doc_export",
        "energy.net_radiation_integral",
    ):
        label = labels[label_id]
        assert label.availability == "missing_non_identifiable"
        assert label.capture is not None
        assert label.reason
    assert labels["carbon.litter_input"].availability == "exactly_derivable"
    assert labels["energy.surface_temperature_tendency"].availability == "exactly_derivable"


def test_audit_proves_existing_shards_are_insufficient_but_capture_is_bounded() -> None:
    inventory = load_daily_flux_label_inventory()
    report = audit_daily_flux_label_inventory(
        inventory,
        _synthetic_contract(inventory),
        contract_sha256=inventory.teacher_contract_sha256,
    )

    assert report["label_count"] == len(inventory.labels)
    assert report["missing_label_count"] > 0
    assert report["existing_shards_state_update_ready"] is False
    assert report["existing_shards_budget_ready"] is False
    assert report["ready_after_declared_capture"] is True
    assert set(report["minimal_supplemental_capture"]) == set(inventory.capture_family_ids)
    assert all(report["minimal_supplemental_capture"].values())


def test_audit_fails_on_contract_hash_or_leaf_drift() -> None:
    inventory = load_daily_flux_label_inventory()
    contract = _synthetic_contract(inventory)
    with pytest.raises(ValueError, match="hash"):
        audit_daily_flux_label_inventory(inventory, contract, contract_sha256="wrong")

    contract["fast_day_target_leaves"] = [
        item
        for item in contract["fast_day_target_leaves"]
        if item["key"] != "daily_interface.gpp_daily"
    ]
    with pytest.raises(ValueError, match="absent Teacher fields"):
        audit_daily_flux_label_inventory(
            inventory,
            contract,
            contract_sha256=inventory.teacher_contract_sha256,
        )


def test_loader_rejects_unknown_capture_family(tmp_path) -> None:
    payload = json.loads(DEFAULT_INVENTORY_PATH.read_text(encoding="utf-8"))
    missing = next(
        label for label in payload["labels"] if label["availability"] == "missing_non_identifiable"
    )
    missing["capture"]["family"] = "invented_family"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown capture family"):
        load_daily_flux_label_inventory(path)
