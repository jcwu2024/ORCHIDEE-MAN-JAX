from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_direction import (
    FROZEN_CONTRACT_SHA256,
    MULTI_GROUP_RESULT_SHA256,
    artifact_aware_scale_passed,
    classify_artifact_aware_direction,
    load_coupled_gpp_artifact_aware_direction_contract,
)
from scripts.hpc import audit_target_oriented_coupled_gpp_artifact_aware_direction as runner


class _FakeJax:
    @staticmethod
    def block_until_ready(value):
        return value


def test_artifact_aware_contract_freezes_40_batches_and_43_constraints():
    contract = load_coupled_gpp_artifact_aware_direction_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.payload["identity"]["multi_group_result_sha256"] == MULTI_GROUP_RESULT_SHA256
    assert len(contract.train_groups) == 5
    assert len(contract.train_artifacts) == 40
    assert len(set(contract.train_artifacts)) == 40
    assert len(contract.source_registry) == 43
    assert contract.payload["data"]["teacher_days_per_direct_gpp_artifact"] == 208
    assert contract.payload["scale_selection"]["stop_after_first_train_only_passing_scale"] is True
    assert contract.payload["data"]["fixed_monitor_excluded_from_direction_and_scale_selection"] is True
    assert not any(contract.payload["retention"].values())
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_large_minimum_norm_solver_supports_43_constraints():
    labels = tuple(f"constraint_{index}" for index in range(43))
    result = gradient_direction_diagnostics(np.eye(43), labels=labels)
    selected = result["directions"]["minimum_norm_unit"]
    assert selected["all_strictly_positive"] is True
    assert min(selected["normalized_directional_derivatives"].values()) == pytest.approx(1.0 / np.sqrt(43.0))
    assert sum(selected["normalized_simplex_weights"].values()) == pytest.approx(1.0)


def test_artifact_aware_scale_and_monitor_classification():
    passing = {
        "scheduled_four_source_finite_step_accepted": True,
        "scheduled_physical_hard_gate_passed": True,
        "artifact_direct_gpp_relative_changes": [-0.1, 1.0e-7],
        "combined_artifact_direct_gpp_relative_change": -0.02,
    }
    assert artifact_aware_scale_passed(passing, expected_artifact_count=2, relative_tolerance=1.0e-6)
    assert not artifact_aware_scale_passed(
        passing | {"artifact_direct_gpp_relative_changes": [-0.1, 2.0e-6]},
        expected_artifact_count=2,
        relative_tolerance=1.0e-6,
    )
    assert (
        classify_artifact_aware_direction(
            common_first_order_direction=True,
            selected_scale=0.25,
            monitor_relative_change=-0.01,
        )
        == "artifact_aware_direction_transfers_to_monitor"
    )
    assert (
        classify_artifact_aware_direction(
            common_first_order_direction=True,
            selected_scale=0.25,
            monitor_relative_change=0.01,
        )
        == "artifact_aware_direction_fails_monitor"
    )
    assert (
        classify_artifact_aware_direction(
            common_first_order_direction=False,
            selected_scale=None,
            monitor_relative_change=None,
        )
        == "no_artifact_aware_common_direction"
    )


def test_artifact_gradient_measurement_keeps_ordered_unit_vectors(monkeypatch):
    monkeypatch.setattr(runner, "TRAINABLE_PARAMETER_COUNT", 2)
    monkeypatch.setattr(
        runner,
        "_load_artifact",
        lambda _args, _context, update: SimpleNamespace(batch=int(update)),
    )
    monkeypatch.setattr(
        runner,
        "_direct_gradient",
        lambda _jax, _jnp, _trainable, batch, _steps: (
            float(batch),
            np.asarray([batch, batch + 1], dtype=np.float64),
            0.1,
            0.2,
        ),
    )
    monkeypatch.setattr(runner, "_direct_loss", float)

    def flatten(gradient):
        norm = np.linalg.norm(gradient)
        return gradient / norm, norm, np.asarray(0)

    context = {
        "runtime": {"jax": _FakeJax(), "jnp": np},
        "schedule": {"entries": [{"runtime_year": 1961}, {"runtime_year": 1962}]},
    }
    result = runner._measure_direct_artifact_gradients(
        None,
        context,
        trainable=np.zeros(2),
        direct_steps={},
        updates=(1, 2),
        flatten_gradient=flatten,
        label="test",
    )
    assert [record["artifact_update"] for record in result["gradient_reports"]] == [1, 2]
    assert result["panel"]["mean_loss"] == 1.5
    np.testing.assert_allclose(np.linalg.norm(result["unit_gradients"][0]), 1.0)


def test_artifact_aware_runner_is_zero_retention_and_monitor_last():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/hpc/audit_target_oriented_coupled_gpp_artifact_aware_direction.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_artifact_aware_direction.sh").read_text(
        encoding="utf-8"
    )
    assert "gradient_direction_diagnostics" in source
    assert "_make_flat_gram" in source
    assert "write_coupled_gpp_recovery" not in source
    assert "write_coupled_gpp_checkpoint" not in source
    selection = source.index("selected_scale = None if selected_record is None")
    monitor = source.index('label="artifact_aware_selected_monitor_before"')
    assert selection < monitor
    assert 'if record["train_only_scale_passed"]:' in source
    assert "PYTHON_MODULE=scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_direction" in launcher
    assert "sbatch" not in launcher


def test_artifact_aware_launcher_supplies_every_required_cli_argument():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(
        encoding="utf-8"
    )
    required_options = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    missing = sorted(option for option in required_options if option not in launcher)
    assert missing == []
