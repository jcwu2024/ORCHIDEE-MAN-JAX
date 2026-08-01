from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.carbon_budget_ownership import (
    AGGREGATE_TRANSFER_LABEL_OWNERSHIP,
    AGGREGATE_TRANSFER_LABELS,
    FORTRAN_MIN_STOMATE,
    FULL_STATE_INTERNAL_TRANSFER_REQUIREMENTS,
    OK_LEAK_CARBON_FIELD_OWNERSHIP,
    OK_LEAK_DRIVER_SERIES,
    audit_aggregate_transfer_source_map,
    audit_carbon_budget_contract,
    extract_compiled_ok_leak_driver_steps,
    extract_ok_leak_driver_series,
    ok_leak_driver_capture_metadata,
)

ROOT = Path(__file__).resolve().parents[2]


def _contract_metadata():
    manifest = json.loads(
        (
            ROOT
            / "outputs"
            / "research"
            / "daily_coarse_graining"
            / "dataset-manifest-v5"
            / "dataset_manifest.json"
        ).read_text(encoding="utf-8")
    )
    return manifest["markov_contract"]


def test_v5_contract_has_endpoints_but_not_transfer_or_driver_labels():
    report = audit_carbon_budget_contract(_contract_metadata())

    assert report["ok_leak_target_width"] == 1376
    assert report["independent_inventory_target_width"] == 1222
    assert report["derived_diagnostic_target_width"] == 64
    assert report["overlapping_or_ratio_target_width"] == 90
    assert report["missing_ok_leak_fields"] == []
    assert report["capabilities"] == {
        "endpoint_stock_supervision": True,
        "exact_scan_driver_supervision": False,
        "aggregate_transfer_supervision": False,
    }
    assert report["decision"] == "capture_bounded_ok_leak_driver_auxiliary_dataset"


def test_deep_peat_is_the_only_derived_carbon_diagnostic():
    derived = tuple(
        item.field
        for item in OK_LEAK_CARBON_FIELD_OWNERSHIP
        if item.classification == "derived_diagnostic"
    )
    assert derived == ("deepC_peat",)
    assert not next(
        item for item in OK_LEAK_CARBON_FIELD_OWNERSHIP if item.field == "deepC_peat"
    ).counts_in_carbon_inventory


def test_driver_capture_requires_complete_48_step_series():
    stacks = {
        item.field: np.zeros((48, 1), dtype=np.float64)
        for item in OK_LEAK_DRIVER_SERIES
        if item.field != "resp_maint_part_radia"
    }
    record = SimpleNamespace(
        half_hour_transition=SimpleNamespace(compiled_entry_stacks=stacks),
        maintenance_resp_parts=np.zeros((48, 1, 2), dtype=np.float64),
    )

    arrays = extract_ok_leak_driver_series(record)
    metadata = ok_leak_driver_capture_metadata(arrays)

    assert tuple(arrays) == tuple(item.field for item in OK_LEAK_DRIVER_SERIES)
    assert metadata["step_count"] == 48
    assert metadata["uncompressed_bytes_per_day"] == sum(
        value.nbytes for value in arrays.values()
    )

    stacks["soil_mc"] = np.zeros((47, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="must contain 48 steps"):
        extract_ok_leak_driver_series(record)


def test_driver_capture_rejects_missing_source_array():
    stacks = {
        item.field: np.zeros((48, 1), dtype=np.float64)
        for item in OK_LEAK_DRIVER_SERIES
        if item.field not in {"resp_maint_part_radia", "wat_flux"}
    }
    record = SimpleNamespace(
        half_hour_transition=SimpleNamespace(compiled_entry_stacks=stacks),
        maintenance_resp_parts=np.zeros((48, 1), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="wat_flux"):
        extract_ok_leak_driver_series(record)


def test_outer_compiled_driver_steps_use_the_same_capture_schema():
    steps = SimpleNamespace(
        **{
            item.field: np.zeros((48, 1), dtype=np.float64)
            for item in OK_LEAK_DRIVER_SERIES
        }
    )

    arrays = extract_compiled_ok_leak_driver_steps(steps)

    assert tuple(arrays) == tuple(item.field for item in OK_LEAK_DRIVER_SERIES)
    assert all(value.shape[0] == 48 for value in arrays.values())


def test_aggregate_transfer_source_map_is_complete_but_not_state_sufficient():
    report = audit_aggregate_transfer_source_map()

    assert tuple(item.field for item in AGGREGATE_TRANSFER_LABEL_OWNERSHIP) == (
        AGGREGATE_TRANSFER_LABELS
    )
    assert report["all_labels_source_backed"]
    assert report["aggregate_conservation_diagnostic_ready"]
    assert not report["full_resolved_state_update_ready"]
    assert set(report["additional_internal_transfer_requirements"]) == set(
        FULL_STATE_INTERNAL_TRANSFER_REQUIREMENTS
    )
    assert report["decision"] == (
        "capture_resolved_source_terms_before_daily_tendency_training"
    )


def test_external_doc_input_does_not_double_count_canopy_drip():
    owner = next(
        item
        for item in AGGREGATE_TRANSFER_LABEL_OWNERSHIP
        if item.field == "doc_external_input"
    )

    assert "canopy2ground" not in owner.source_terms
    assert "doc_precip2canopy" in owner.source_terms
    assert "dry_dep_canopy" in owner.source_terms


def test_conservation_absolute_threshold_matches_fortran_min_stomate():
    assert FORTRAN_MIN_STOMATE == 1.0e-8
