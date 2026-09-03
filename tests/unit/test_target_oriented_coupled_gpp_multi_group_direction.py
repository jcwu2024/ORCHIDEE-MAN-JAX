from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_step_transfer import (
    DISPLACEMENT_SCALES,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_multi_group_direction import (
    DIRECT_STEP_TRANSFER_RESULT_SHA256,
    FROZEN_CONTRACT_SHA256,
    MULTI_GROUP_GRADIENT_SOURCES,
    classify_multi_group_direction,
    load_coupled_gpp_multi_group_direction_contract,
)
from scripts.hpc import audit_target_oriented_coupled_gpp_multi_group_direction as runner


class _TreeUtil:
    @staticmethod
    def tree_map(function, *trees):
        return {name: function(*(tree[name] for tree in trees)) for name in trees[0]}

    @staticmethod
    def tree_leaves(tree):
        return list(tree.values())


class _FakeJax:
    tree_util = _TreeUtil()

    @staticmethod
    def block_until_ready(value):
        return value


def test_multi_group_direction_contract_is_frozen_before_gradients():
    contract = load_coupled_gpp_multi_group_direction_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.payload["identity"]["direct_step_transfer_result_sha256"] == DIRECT_STEP_TRANSFER_RESULT_SHA256
    assert contract.scales == DISPLACEMENT_SCALES
    assert len(MULTI_GROUP_GRADIENT_SOURCES) == 8
    assert len(contract.guard_groups) == 4
    assert contract.payload["data"]["fixed_monitor_excluded_from_direction_and_scale_selection"] is True
    assert contract.payload["data"]["candidate_direction_monitor_values_evaluated_before_contract_freeze"] is False
    assert contract.payload["diagnosis"]["predecessor_direction_recomputed_as_control"] is False
    assert not any(contract.payload["retention"].values())
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_multi_group_direction_classification_keeps_monitor_out_of_missing_steps():
    assert (
        classify_multi_group_direction(
            common_first_order_direction=False,
            selected_scale=None,
            monitor_relative_change=None,
        )
        == "no_multi_group_common_direction"
    )
    assert (
        classify_multi_group_direction(
            common_first_order_direction=True,
            selected_scale=None,
            monitor_relative_change=None,
        )
        == "no_multi_group_finite_step"
    )
    assert (
        classify_multi_group_direction(
            common_first_order_direction=True,
            selected_scale=0.25,
            monitor_relative_change=-0.01,
        )
        == "multi_group_direction_transfers_to_monitor"
    )
    assert (
        classify_multi_group_direction(
            common_first_order_direction=True,
            selected_scale=0.25,
            monitor_relative_change=0.01,
        )
        == "multi_group_direction_fails_monitor"
    )
    with pytest.raises(ValueError, match="cannot have"):
        classify_multi_group_direction(
            common_first_order_direction=False,
            selected_scale=0.25,
            monitor_relative_change=None,
        )


def test_direct_group_gradient_is_the_ordered_artifact_mean(monkeypatch):
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
            {"weights": np.asarray([batch, batch + 1], dtype=np.float32)},
            0.1,
            0.2,
        ),
    )
    monkeypatch.setattr(runner, "_direct_loss", float)
    context = {
        "runtime": {"jax": _FakeJax(), "jnp": np},
        "schedule": {"entries": [{"runtime_year": 1961}, {"runtime_year": 1962}]},
    }
    result = runner._measure_direct_group_gradient(
        None,
        context,
        trainable={"weights": np.zeros(2, dtype=np.float32)},
        direct_steps={},
        updates=(1, 2),
        label="test",
    )
    np.testing.assert_array_equal(
        result["gradient"]["weights"],
        np.asarray([1.5, 2.5], dtype=np.float32),
    )
    assert result["panel"]["artifact_updates"] == [1, 2]
    assert result["panel"]["mean_loss"] == 1.5
    assert [record["runtime_year"] for record in result["panel"]["artifact_losses"]] == [1961, 1962]


def test_multi_group_direction_runner_is_zero_retention_and_monitor_last():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/audit_target_oriented_coupled_gpp_multi_group_direction.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(
        encoding="utf-8"
    )
    assert "gradient_direction_diagnostics" in runner
    assert "MULTI_GROUP_GRADIENT_SOURCES" in runner
    assert "select_largest_train_only_scale" in runner
    assert "write_coupled_gpp_recovery" not in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    selection = runner.index("selected_scale = (")
    monitor = runner.index('label="selected_monitor_before"')
    assert selection < monitor
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
    assert "sbatch" not in launcher
