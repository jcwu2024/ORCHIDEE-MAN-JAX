from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.daily_coarse_graining import daily_markov_contract as markov
from research.daily_coarse_graining import teacher_data_product_admission as admission
from research.daily_coarse_graining import teacher_production


def _canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _population(tmp_path: Path) -> Path:
    ids = [f"{2 * index + 1:03d}.0-{71 + 2 * index:03d}.0" for index in range(8)]
    rows = [
        {
            "landpoint_id": landpoint_id,
            "grid_x": index,
            "grid_y": index * index,
            "vcmax25": 30.0 + index,
            "maint_resp_slope_c": 0.01 + index / 1000.0,
            "alloc_min": 0.1 + index / 100.0,
            "residence_time": 20.0 + index,
        }
        for index, landpoint_id in enumerate(ids)
    ]
    path = tmp_path / "population.json"
    path.write_text(
        json.dumps({"selected_landpoint_ids": ids, "selected": rows}),
        encoding="utf-8",
    )
    return path


def _condition(name: str, width: int, role: str) -> markov.ConditionLeafSpec:
    return markov.ConditionLeafSpec(
        name=name,
        shape=(width,),
        dtype="float64",
        start=0,
        stop=width,
        temporal_role=role,
        source="test",
    )


def _contract() -> markov.DailyMarkovContract:
    return markov.DailyMarkovContract(
        schema_version=markov.CONTRACT_SCHEMA_VERSION,
        state_leaves=(
            markov.StateLeafSpec(
                component="slow",
                path=("stock",),
                shape=(2,),
                full_shape=(2,),
                dtype="float64",
                start=0,
                stop=2,
                classification="prognostic",
                source_ref="test",
            ),
            markov.StateLeafSpec(
                component="slow",
                path=("flag",),
                shape=(1,),
                full_shape=(1,),
                dtype="bool",
                start=None,
                stop=None,
                classification="discrete",
                source_ref="test",
            ),
        ),
        fast_day_target_leaves=(
            markov.FastDayTargetLeafSpec(
                family="daily_interface",
                component=None,
                path=("gpp_daily",),
                shape=(1,),
                full_shape=(1,),
                dtype="float64",
                start=0,
                stop=1,
                owner="test",
            ),
        ),
        diagnostic_leaves=(
            markov.DiagnosticSpec(
                name="daily_fold.gpp_daily",
                shape=(1,),
                dtype="float64",
                start=0,
                stop=1,
                owner="test",
            ),
        ),
        native_forcing=markov.NativeForcingSpec(
            fields=("Tair",),
            field_shape=(1, 1),
            source_records_per_day=5,
            source_interval_seconds=21600.0,
            model_interval_seconds=1800.0,
            split=12,
            precipitation_spread_steps=6,
        ),
        static_conditions=markov.StaticConditionSpec(
            parameter_leaves=(
                _condition("parameter", 1, "landpoint_parameter"),
            ),
            landpoint_static_leaves=(
                _condition("static", 2, "landpoint_static"),
            ),
            annual_condition_leaves=(
                _condition("co2", 1, "annual_exogenous"),
            ),
            includes=("test",),
        ),
        source_hashes=(("source", "abc"),),
        active_pft_indices=(0, 13),
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    population = _population(tmp_path)
    spec_path = teacher_production.freeze_spec(
        population,
        tmp_path / "spec.json",
        dataset_id="admission-test",
        validation_landpoints=2,
        test_landpoints=2,
    )
    contract = _contract()
    manifest = {
        "schema_version": "daily_teacher_dataset_manifest_v4",
        "status": "complete",
        "dataset_id": "contract-evidence",
        "markov_contract_sha256": contract.sha256,
        "markov_contract": contract.metadata(),
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = teacher_production.load_production_spec(spec_path)
    policy = {
        "schema_version": admission.POLICY_SCHEMA_VERSION,
        "policy_id": "test-policy",
        "production_spec": str(spec_path),
        "production_spec_sha256": _canonical_hash(
            json.loads(spec_path.read_text(encoding="utf-8"))
        ),
        "population_manifest_sha256": spec.population_manifest_sha256,
        "expected_population": {
            "landpoints": 8,
            "first_year": 1961,
            "last_year": 2010,
            "block_size": 7,
            "spatial_split_counts": {"train": 4, "validation": 2, "test": 2},
            "temporal_splits": {
                name: list(bounds) for name, bounds in spec.temporal_splits.items()
            },
        },
        "expected_contract": {
            "dataset_manifest_schema": "daily_teacher_dataset_manifest_v4",
            "schema_version": markov.CONTRACT_SCHEMA_VERSION,
            "sha256": contract.sha256,
            "continuous_state_width": 2,
            "discrete_state_width": 1,
            "fast_day_target_width": 1,
            "diagnostic_width": 1,
            "active_pft_indices": [0, 13],
            "native_forcing_fields": ["Tair"],
            "native_forcing_records_per_day": 5,
            "condition_widths": {
                "parameters": 1,
                "landpoint_static": 2,
                "annual": 1,
            },
            "required_state_components": ["slow"],
            "required_state_keys": ["slow.stock"],
            "required_target_families": ["daily_interface"],
            "required_target_keys": ["daily_interface.gpp_daily"],
            "required_diagnostics": ["daily_fold.gpp_daily"],
        },
        "cold_start_policy": {
            "initialization_mode": "cold_start_bootstrap",
            "first_supervised_day": 2,
            "first_year_transition_count": 364,
            "ordinary_year_transition_count": 365,
        },
        "parameter_extension_policy": {
            "independent_response_claim": False,
            "must_not_modify_baseline_shards": True,
        },
        "resource_basis": {
            "cpu_cost_cny_per_core_hour": 0.07,
            "hot_capture_seconds_per_point_year": 100.0,
            "complete_point_year_seconds_upper": 150.0,
            "compressed_bytes_per_point_year_lower": 3_000_000,
            "compressed_bytes_per_point_year_upper": 5_000_000,
            "accepted_concurrent_workers": 2,
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path, manifest_path, spec_path


def test_admission_freezes_contract_splits_scale_and_scope(tmp_path):
    policy, manifest, _spec = _inputs(tmp_path)
    output = admission.admit_data_product(
        policy_path=policy,
        contract_manifest_path=manifest,
        output_path=tmp_path / "admission.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["admission_phase"] == "pre_generation_contract"
    assert report["contract_summary"] == {
        "continuous_state_width": 2,
        "discrete_state_width": 1,
        "fast_day_target_width": 1,
        "diagnostic_width": 1,
        "condition_widths": {
            "parameters": 1,
            "landpoint_static": 2,
            "annual": 1,
        },
        "active_pft_indices": [0, 13],
    }
    inventory = report["production_inventory"]
    assert inventory["point_years"] == 400
    assert inventory["total_transitions"] == 8 * (364 + 49 * 365)
    assert inventory["train_train_transitions"] == 4 * (364 + 43 * 365)
    assert report["resource_estimate"]["accepted_concurrent_workers"] == 2


def test_admission_rejects_a_missing_required_scientific_diagnostic(tmp_path):
    policy_path, manifest, _spec = _inputs(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["expected_contract"]["required_diagnostics"].append(
        "daily_fold.missing_flux"
    )
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostic inventory is missing"):
        admission.admit_data_product(
            policy_path=policy_path,
            contract_manifest_path=manifest,
            output_path=tmp_path / "admission.json",
        )


def test_admission_rejects_production_spec_drift(tmp_path):
    policy, manifest, spec = _inputs(tmp_path)
    raw = json.loads(spec.read_text(encoding="utf-8"))
    raw["block_size"] = 28
    spec.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="production spec hash mismatch"):
        admission.admit_data_product(
            policy_path=policy,
            contract_manifest_path=manifest,
            output_path=tmp_path / "admission.json",
        )


def test_final_admission_requires_every_frozen_point_year_and_split(tmp_path):
    policy, manifest_path, spec_path = _inputs(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = teacher_production.load_production_spec(spec_path)
    shards = [
        {
            "landpoint_id": item.landpoint_id,
            "year": year,
            "spatial_split": item.spatial_split,
            "temporal_split": spec.temporal_split(year),
            "markov_contract_sha256": manifest["markov_contract_sha256"],
        }
        for item in spec.landpoints
        for year in range(spec.first_year, spec.last_year + 1)
    ]
    manifest.update(
        {
            "dataset_id": spec.dataset_id,
            "landpoint_count": len(spec.landpoints),
            "year_count": spec.last_year - spec.first_year + 1,
            "shard_count": len(shards),
            "shards": shards,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = admission.admit_data_product(
        policy_path=policy,
        contract_manifest_path=manifest_path,
        output_path=tmp_path / "final.json",
        require_production_dataset=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["admission_phase"] == "complete_669_dataset"

    manifest["shards"][-1]["spatial_split"] = "test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="production shard split mismatch"):
        admission.admit_data_product(
            policy_path=policy,
            contract_manifest_path=manifest_path,
            output_path=tmp_path / "final.json",
            require_production_dataset=True,
        )
