from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import pytest

from research.daily_coarse_graining.daily_operator.fast_owner_reconstruction import (
    load_fast_owner_reconstruction_contract,
)
from research.daily_coarse_graining.daily_operator.neural_architecture import (
    NeuralOutputSchema,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
    FROZEN_CONTRACT_SHA256,
    contract_sha256,
    ledger_by_name,
    load_strong_baseline_contract,
    resolve_strong_baseline_field_ledger,
)
from tests.unit.test_daily_operator_successor_process_head import (
    extended_successor_setup,
    successor_setup,
)


@pytest.fixture(scope="module")
def strong_baseline_resources():
    base = successor_setup.__wrapped__()
    extended = extended_successor_setup.__wrapped__(base)
    contract = load_strong_baseline_contract()
    owner_contract = load_fast_owner_reconstruction_contract(
        markov_contract=base["adapter"].contract,
    )
    ledger = resolve_strong_baseline_field_ledger(
        contract,
        schema=extended["schema"],
        adapter=base["adapter"],
        owner_contract=owner_contract,
    )
    return {
        "adapter": base["adapter"],
        "contract": contract,
        "ledger": ledger,
        "owner_contract": owner_contract,
        "schema": extended["schema"],
    }


def test_strong_baseline_contract_freezes_architecture_and_data_identity(
    strong_baseline_resources,
):
    contract = strong_baseline_resources["contract"]
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.expert_names == (
        "canopy_gpp",
        "surface_energy",
        "hydrology_snow",
        "soil_thermal",
        "maintenance_respiration",
        "litter_poc_doc",
        "daily_interface_finalize",
    )
    assert contract.payload["data_identity"]["historical_v5_allowed"] is False
    assert contract.payload["data_identity"]["teacher_regeneration_allowed"] is False
    assert contract.payload["data_identity"]["sealed_test_allowed"] is False
    assert contract.payload["precision"]["neural_parameters"] == "float32"
    assert contract.payload["precision"]["conservative_update"] == "float64"
    assert contract.payload["capacity"]["parameter_count_hard_maximum"] == 60_000_000


def test_strong_baseline_ledger_covers_every_field_once(strong_baseline_resources):
    ledger = strong_baseline_resources["ledger"]
    schema = strong_baseline_resources["schema"]
    assert len(ledger) == len({record.name for record in ledger}) == 181
    assert len({record.label for record in ledger}) == len(schema.labels) == 48
    assert sum(record.scalar_count for record in ledger) == 39_371
    assert Counter(record.expert for record in ledger) == {
        "canopy_gpp": 17,
        "surface_energy": 18,
        "hydrology_snow": 77,
        "soil_thermal": 16,
        "maintenance_respiration": 1,
        "litter_poc_doc": 42,
        "daily_interface_finalize": 10,
    }


def test_strong_baseline_ledger_preserves_large_tensor_local_axes(
    strong_baseline_resources,
):
    fields = ledger_by_name(strong_baseline_resources["ledger"])
    doc = fields["typed.ok_leak.cryoturbation_doc_transfer"]
    assert doc.expert == "litter_poc_doc"
    assert doc.shape == (14, 32, 2, 7, 1)
    assert doc.axes == (
        "pft",
        "deep_layer",
        "transfer_direction",
        "doc_pool",
        "element",
    )
    assert doc.local_axes == ("pft", "deep_layer", "transfer_direction", "doc_pool")

    hydrology = fields["endpoint.fast_state.hydrol_previous_step_state.us"]
    assert hydrology.expert == "hydrology_snow"
    assert hydrology.axes == ("landpoint", "pft", "soil_tile", "soil_layer")
    assert hydrology.local_axes == ("pft", "soil_tile", "soil_layer")
    assert hydrology.day_start_owner == "hydrol_previous_step_state.us"

    maintenance = fields["daily.fast_maintenance_respiration"]
    assert maintenance.expert == "maintenance_respiration"
    assert maintenance.axes == ("landpoint", "pft", "plant_part")
    assert maintenance.local_axes == ("pft", "plant_part")


def test_strong_baseline_pft_override_recovers_daily_summary_axis(
    strong_baseline_resources,
):
    fields = ledger_by_name(strong_baseline_resources["ledger"])
    summary = fields["endpoint.daily_summary.resp_maint_radia"]
    assert summary.expert == "daily_interface_finalize"
    assert summary.shape == (1, 14)
    assert summary.axes == ("landpoint", "pft")
    assert summary.pft_axis == 1


def test_strong_baseline_ledger_rejects_missing_field(strong_baseline_resources):
    schema = strong_baseline_resources["schema"]
    shortened = NeuralOutputSchema(
        fields=schema.fields[:-1],
        labels=schema.labels,
        active_pft_indices=schema.active_pft_indices,
    )
    with pytest.raises(ValueError, match="field ledger count drift"):
        resolve_strong_baseline_field_ledger(
            strong_baseline_resources["contract"],
            schema=shortened,
            adapter=strong_baseline_resources["adapter"],
            owner_contract=strong_baseline_resources["owner_contract"],
        )


def test_strong_baseline_ledger_rejects_axis_rank_drift(strong_baseline_resources):
    schema = strong_baseline_resources["schema"]
    target = next(index for index, field in enumerate(schema.fields) if field.name == "daily.fast_gpp")
    fields = list(schema.fields)
    fields[target] = replace(fields[target], shape=(1, 2, 1))
    drifted = NeuralOutputSchema(
        fields=tuple(fields),
        labels=schema.labels,
        active_pft_indices=schema.active_pft_indices,
    )
    with pytest.raises(ValueError, match="field axis rank drift"):
        resolve_strong_baseline_field_ledger(
            strong_baseline_resources["contract"],
            schema=drifted,
            adapter=strong_baseline_resources["adapter"],
            owner_contract=strong_baseline_resources["owner_contract"],
        )


def test_strong_baseline_contract_rejects_rehashed_architecture_drift(
    tmp_path,
    strong_baseline_resources,
):
    payload = json.loads(strong_baseline_resources["contract"].source_path.read_text(encoding="utf-8"))
    payload["capacity"]["expert_width"] = 320
    payload["contract_sha256"] = contract_sha256(payload)
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not the approved v1"):
        load_strong_baseline_contract(path)
