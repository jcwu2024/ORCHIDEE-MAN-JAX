from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import research.daily_coarse_graining.causal_carbon_adapter_run as adapter_run
from research.daily_coarse_graining.causal_carbon_adapter_protocol import (
    load_causal_carbon_adapter_protocol,
)
from research.daily_coarse_graining.causal_carbon_adapter_run import (
    PreparedAdapterExperiment,
    _build_training_checkpoint,
    _calibration_reference_schedule,
    _completed_horizon_counts,
    _resolve_gradient_coefficients,
    _verify_training_checkpoint,
    build_training_schedule,
    create_training_execution,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "manifests" / "coarse_graining" / "canonical_669_causal_carbon_adapter_experiment.json"


def test_calibration_schedule_is_deterministic_and_balances_four_landpoints():
    references = [
        SimpleNamespace(
            landpoint_id=f"{landpoint:03d}.0-071.0",
            year=year,
            path=Path(f"{landpoint}-{year}.npz"),
        )
        for landpoint in range(8)
        for year in range(1961, 1966)
    ]

    first = _calibration_reference_schedule(
        references,
        records_per_horizon=16,
        seed=20260801,
    )
    second = _calibration_reference_schedule(
        references,
        records_per_horizon=16,
        seed=20260801,
    )

    assert [(item.landpoint_id, item.year) for item in first] == [(item.landpoint_id, item.year) for item in second]
    assert len({item.landpoint_id for item in first}) == 4
    assert all(
        sum(item.landpoint_id == landpoint for item in first) == 4
        for landpoint in {item.landpoint_id for item in first}
    )


def test_gradient_calibration_resolves_declared_target_ratios():
    protocol = load_causal_carbon_adapter_protocol(PROTOCOL)
    records = [
        {
            "gradient_norms": {
                "L_interface": 2.0,
                "L_next": 4.0,
                "L_rollout": rollout,
                "L_flux_bias": 1.0,
                "L_stock_tendency_bias": 0.5,
            }
        }
        for rollout in (0.0, 2.0, 2.0)
    ]

    resolved = _resolve_gradient_coefficients(records, protocol)

    np.testing.assert_allclose(
        resolved["resolved_coefficients"]["L_next"],
        0.5,
    )
    np.testing.assert_allclose(
        resolved["resolved_coefficients"]["L_rollout"],
        1.0,
    )
    np.testing.assert_allclose(
        resolved["resolved_coefficients"]["L_flux_bias"],
        1.0,
    )
    np.testing.assert_allclose(
        resolved["resolved_coefficients"]["L_stock_tendency_bias"],
        2.0,
    )
    assert resolved["component_audit"]["L_rollout"]["positive_records"] == 2


def test_formal_schedule_is_exact_balanced_and_checkpoint_bound():
    protocol = load_causal_carbon_adapter_protocol(PROTOCOL)
    references = [
        SimpleNamespace(
            landpoint_id=f"{landpoint:03d}.0-071.0",
            year=year,
            spatial_split="train",
            temporal_split="train",
            path=Path(f"{landpoint}-{year}.npz"),
            sha256=f"sha-{landpoint}-{year}",
        )
        for landpoint in range(8)
        for year in range(1961, 1966)
    ]

    class Index:
        dataset_id = "dataset"
        contract_sha256 = "contract"

        @staticmethod
        def select(*, spatial_split, temporal_split):
            assert spatial_split == temporal_split == "train"
            return tuple(references)

    experiment = PreparedAdapterExperiment(
        protocol=protocol,
        paths={},
        resources=SimpleNamespace(index=Index()),
        layout=None,
        adapter_definition=None,
        base_parameters=None,
        initial_trainable=None,
    )
    schedule = build_training_schedule(experiment)

    assert schedule["updates"] == 4096
    assert schedule["horizon_counts"] == {"1": 2048, "3": 1229, "7": 819}
    assert sum(schedule["horizon_counts"].values()) == schedule["updates"]
    assert schedule["sealed_test_used"] is False

    parameters = {"weight": np.ones((2,))}
    optimizer = {"step": np.asarray(17)}
    identity = {"id": "test"}
    checkpoint = _build_training_checkpoint(
        arm_id="causal_interface_one_step_control",
        identity=identity,
        parameters=parameters,
        optimizer=optimizer,
        next_update=17,
        schedule=schedule,
        history=(),
    )
    _verify_training_checkpoint(
        checkpoint,
        arm_id="causal_interface_one_step_control",
        identity=identity,
        schedule=schedule,
    )

    assert checkpoint["completed_horizon_counts"] == _completed_horizon_counts(
        schedule,
        17,
    )

    drifted = copy.deepcopy(checkpoint)
    drifted["identity"]["id"] = "other"
    with pytest.raises(ValueError, match="identity drift"):
        _verify_training_checkpoint(
            drifted,
            arm_id="causal_interface_one_step_control",
            identity=identity,
            schedule=schedule,
        )

    drifted = copy.deepcopy(checkpoint)
    drifted["completed_horizon_counts"] = {"1": 17}
    with pytest.raises(ValueError, match="horizon-count drift"):
        _verify_training_checkpoint(
            drifted,
            arm_id="causal_interface_one_step_control",
            identity=identity,
            schedule=schedule,
        )


def test_execution_manifest_is_reproducible_and_fails_on_existing_drift(
    tmp_path,
    monkeypatch,
):
    protocol = load_causal_carbon_adapter_protocol(PROTOCOL)
    references = [
        SimpleNamespace(
            landpoint_id=f"{landpoint:03d}.0-071.0",
            year=1961,
            spatial_split="train",
            temporal_split="train",
            path=Path(f"{landpoint}-1961.npz"),
            sha256=f"sha-{landpoint}-1961",
        )
        for landpoint in range(4)
    ]

    class Index:
        dataset_id = "dataset"
        contract_sha256 = "contract"

        @staticmethod
        def select(*, spatial_split, temporal_split):
            assert spatial_split == temporal_split == "train"
            return tuple(references)

    paths = {}
    for name in (
        "dataset",
        "statistics",
        "acceptance",
        "parent_checkpoint",
        "parent_report",
    ):
        path = tmp_path / f"{name}.dat"
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    feasibility_path = tmp_path / "feasibility.json"
    feasibility_path.write_text("feasibility", encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text("calibration", encoding="utf-8")
    experiment = PreparedAdapterExperiment(
        protocol=protocol,
        paths=paths,
        resources=SimpleNamespace(index=Index()),
        layout=None,
        adapter_definition=SimpleNamespace(identity=lambda: {"id": "causal_carbon_adapter_v1"}),
        base_parameters=None,
        initial_trainable=None,
    )
    calibration = {
        "training_git_head": "head",
        "canonical_sha256": "calibration-id",
        "resolved_coefficients": {
            "L_next": 1.0,
            "L_rollout": 1.0,
            "L_flux_bias": 0.5,
            "L_stock_tendency_bias": 0.5,
        },
    }
    monkeypatch.setattr(
        adapter_run,
        "_prepare_adapter_experiment",
        lambda **_: experiment,
    )
    monkeypatch.setattr(
        adapter_run,
        "_verified_feasibility_report",
        lambda *_: (feasibility_path, {}),
    )
    monkeypatch.setattr(
        adapter_run,
        "_load_verified_calibration",
        lambda *_: (calibration_path, calibration),
    )
    monkeypatch.setattr(adapter_run, "_current_git_head", lambda: "head")

    output_root = tmp_path / "formal"
    kwargs = {
        "protocol_path": PROTOCOL,
        "dataset_path": paths["dataset"],
        "statistics_path": paths["statistics"],
        "acceptance_path": paths["acceptance"],
        "parent_checkpoint_path": paths["parent_checkpoint"],
        "parent_report_path": paths["parent_report"],
        "feasibility_report_path": feasibility_path,
        "calibration_path": calibration_path,
        "output_root": output_root,
    }
    first = create_training_execution(**kwargs)
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second = create_training_execution(**kwargs)

    assert first == second
    assert json.loads(second.read_text(encoding="utf-8")) == first_payload

    first_payload["same_anchor_batches"] = False
    first.write_text(json.dumps(first_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="execution identity drift"):
        create_training_execution(**kwargs)


def test_cli_dispatches_formal_arm(monkeypatch, tmp_path):
    observed = {}

    def fake_run_training_arm(**kwargs):
        observed.update(kwargs)
        return tmp_path / "report.json"

    monkeypatch.setattr(adapter_run, "run_training_arm", fake_run_training_arm)
    required = [
        "--protocol",
        "protocol.json",
        "--dataset",
        "dataset.json",
        "--statistics",
        "statistics.json",
        "--acceptance",
        "acceptance.json",
        "--parent-checkpoint",
        "parent.pkl",
        "--parent-report",
        "parent.json",
        "--output-root",
        str(tmp_path),
    ]
    result = adapter_run.main(
        [
            "--phase",
            "arm",
            "--execution",
            "execution.json",
            "--arm",
            "causal_interface_rollout_candidate",
            "--stop-after-updates",
            "16",
            *required,
        ]
    )

    assert result == 0
    assert observed["arm_id"] == "causal_interface_rollout_candidate"
    assert observed["stop_after_updates"] == 16
