from __future__ import annotations

import math
from pathlib import Path

import jax
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    calibrate_teacher_guard_coefficient,
    combine_adapter_gradients,
    compare_distribution_aware_train_train_groups,
    load_distribution_aware_bounded_training_contract,
    load_distribution_aware_guard_calibration_contract,
)


def test_distribution_aware_guard_calibration_contract_is_exact():
    contract = load_distribution_aware_guard_calibration_contract()
    assert contract.sha256 == "c101940963465df1e25878f596645037e7d2de22dcefcf21f622fcd1de7179d7"
    assert contract.calibration_updates == (352, 353, 354, 355, 356, 357, 359, 360)
    assert contract.payload["teacher_guard_objective"]["cell_weight"] == 1.0 / 6.0
    assert contract.payload["execution"]["optimizer_created"] is False
    assert contract.payload["execution"]["training_updates"] == 0


def test_teacher_guard_coefficient_uses_medians_and_frozen_clipping():
    free = (1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0)
    guard = (2.0, 4.0, 6.0, 8.0, 12.0, 14.0, 16.0, 18.0)
    result = calibrate_teacher_guard_coefficient(free, guard)
    assert result.median_free_gradient_norm == 5.0
    assert result.median_teacher_guard_gradient_norm == 10.0
    assert result.unclipped_coefficient == 0.5
    assert result.coefficient == 0.5
    assert result.clipped is False

    lower = calibrate_teacher_guard_coefficient(np.ones(8), np.full(8, 100.0))
    upper = calibrate_teacher_guard_coefficient(np.full(8, 100.0), np.ones(8))
    assert lower.coefficient == 0.1 and lower.clipped
    assert upper.coefficient == 10.0 and upper.clipped


@pytest.mark.parametrize(
    ("free", "guard"),
    [
        ((1.0,) * 7, (1.0,) * 8),
        ((1.0,) * 8, (0.0,) * 8),
        ((1.0,) * 8, (float("nan"),) * 8),
    ],
)
def test_teacher_guard_coefficient_rejects_invalid_panels(free, guard):
    with pytest.raises(ValueError):
        calibrate_teacher_guard_coefficient(free, guard)


def test_distribution_aware_bounded_training_contract_is_exact_and_disjoint():
    contract = load_distribution_aware_bounded_training_contract()
    assert contract.sha256 == "0c60924ee12d0ea555b082f76990ca55e34f86f67effb052f5eec44c01210051"
    assert contract.evaluation_updates == (251, 358, 465, 572, 679, 786, 893, 1000)
    assert contract.payload["objective"]["teacher_guard_coefficient"] == 0.1
    assert contract.optimizer["maximum_updates"] == 250
    assert contract.optimizer["global_gradient_norm_clip"]["threshold"] == pytest.approx(
        2.692620468992339
    )
    training = contract.payload["data"]["training_updates"]
    assert training == {
        "artifact_candidate_start": 252,
        "artifact_candidate_end": 511,
        "excluded_artifact_updates": [352, 353, 354, 355, 356, 357, 358, 359, 360, 465],
        "selection": "ascending_candidate_updates_after_exclusion",
        "optimizer_update_count": 250,
    }


def test_distribution_aware_gradient_combination_uses_same_adapter_tree():
    free = {"a": np.asarray([1.0, 2.0]), "b": (np.asarray(3.0),)}
    guard = {"a": np.asarray([4.0, 5.0]), "b": (np.asarray(6.0),)}
    combined = combine_adapter_gradients(jax.tree_util, free, guard, 0.1)
    np.testing.assert_allclose(combined["a"], [1.4, 2.5])
    np.testing.assert_allclose(combined["b"][0], 3.6)
    with pytest.raises(ValueError, match="share a tree"):
        combine_adapter_gradients(jax.tree_util, free, {"different": np.asarray(1.0)}, 0.1)


def _acceptance_key(score: float, *, count: float) -> dict[str, float | bool]:
    return {
        "normalized_rmse": math.sqrt(2.0 * score),
        "normalized_signed_bias": 0.0,
        "evaluated_values": count,
        "rmse_maximum": 1.0,
        "absolute_bias_maximum": 0.1,
        "passed": math.sqrt(2.0 * score) <= 1.0,
    }


def _screen_groups(contract, factor: float):
    names = (contract.science_gate["maintenance_key"], *contract.science_gate["protected_keys"])
    groups = []
    for index, update in enumerate(contract.evaluation_updates):
        acceptance = {}
        for horizon, count in (("7", 8.0), ("30", 4.0)):
            keys = {name: _acceptance_key(factor, count=count) for name in names}
            acceptance[horizon] = {
                "keys": keys,
                "all_key_gates_passed": all(value["passed"] for value in keys.values()),
            }
        groups.append(
            {
                "update": update,
                "runtime_year": 1961 + index,
                "hard_failure_count": 0,
                "acceptance": acceptance,
            }
        )
    return groups


def test_distribution_aware_screen_requires_relative_and_absolute_science_gates():
    contract = load_distribution_aware_bounded_training_contract()
    parent = _screen_groups(contract, 1.0)
    candidate = _screen_groups(contract, 0.25)
    result = compare_distribution_aware_train_train_groups(parent, candidate, contract)
    assert result["median_maintenance_score_reduction"] == {"7": 0.75, "30": 0.75}
    assert result["relative_mechanism_gate_passed"]
    assert result["absolute_target_oriented_gate_passed"]
    assert result["scientific_promotion_gate_passed"]

    unchanged = compare_distribution_aware_train_train_groups(parent, parent, contract)
    assert not unchanged["maintenance_reduction_gate_passed"]
    assert not unchanged["scientific_promotion_gate_passed"]
