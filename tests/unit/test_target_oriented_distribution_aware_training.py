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
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
    load_distribution_aware_checkpoint,
    load_distribution_aware_recovery,
    write_distribution_aware_checkpoint,
    write_distribution_aware_recovery,
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


def test_guard_calibration_runner_is_zero_update_and_free_node_only():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/calibrate_target_oriented_distribution_aware_guard.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/hpc/calibrate_target_oriented_distribution_aware_guard.sh").read_text(encoding="utf-8")
    step_factory = (root / "scripts/hpc/run_target_oriented_full_step_preflight.py").read_text(encoding="utf-8")
    assert '"optimizer_created": False' in runner
    assert '"training_updates": 0' in runner
    assert "initialize_target_oriented_adamw" not in runner
    assert "make_trainable_gradient_steps" in runner
    assert "make_trainable_teacher_state_acceptance_gradient_steps" in runner
    assert "make_trainable_teacher_state_acceptance_gradient_steps" in step_factory
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher


def test_distribution_aware_one_update_admission_is_ephemeral_and_adapter_only():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root / "scripts/hpc/run_target_oriented_distribution_aware_one_update_admission.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root / "scripts/hpc/run_target_oriented_distribution_aware_one_update_admission.sh"
    ).read_text(encoding="utf-8")
    assert "combine_adapter_gradients" in runner
    assert "gradient_parameters=adapter" in runner
    assert "gradient_steps=free_gradient_steps" in runner
    assert '"training_updates_executed": 1' in runner
    assert '"training_updates_persisted": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert '"recovery_written": False' in runner
    assert "write_source_shaped_checkpoint" not in runner
    assert "write_target_oriented_stage_checkpoint" not in runner
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-150m}" in launcher


def test_distribution_aware_adapter_checkpoint_round_trip(tmp_path):
    identity = {"contract": "frozen"}
    adapter = {"kernel": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)}
    optimizer = {
        "step": np.asarray(2, dtype=np.int32),
        "first": {"kernel": np.zeros(3, dtype=np.float32)},
        "second": {"kernel": np.ones(3, dtype=np.float32)},
    }
    records = [{"optimizer_update": 1}, {"optimizer_update": 2}]
    screen = {
        "candidate_groups": [{"update": value} for value in range(8)],
        "comparison_to_fixed_parent": {"hard_gate_passed": True},
    }
    recovery_path = tmp_path / "recovery.pkl"
    write_distribution_aware_recovery(
        recovery_path,
        identity=identity,
        adapter=adapter,
        optimizer=optimizer,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=2.5,
        update_records=records,
        initial_screen=screen,
    )
    recovery = load_distribution_aware_recovery(
        recovery_path,
        identity=identity,
        parameter_count=3,
        clip_threshold=2.5,
        maximum_updates=250,
    )
    assert recovery is not None
    np.testing.assert_array_equal(recovery["adapter"]["kernel"], adapter["kernel"])

    checkpoint_path = tmp_path / "stage-update-0250.pkl"
    metadata = write_distribution_aware_checkpoint(
        checkpoint_path,
        identity=identity,
        adapter=adapter,
        optimizer=optimizer,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=2.5,
        update_records=records,
        initial_screen=screen,
        final_screen=screen,
    )
    checkpoint = load_distribution_aware_checkpoint(
        checkpoint_path,
        expected_sha256=metadata["sha256"],
        identity=identity,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=2.5,
    )
    assert checkpoint["promotion_eligible"] is False
    assert checkpoint["completed_updates"] == 2


def test_distribution_aware_paid_runner_is_bounded_adapter_only_and_unsubmitted():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root / "scripts/hpc/run_target_oriented_distribution_aware_training.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root / "scripts/hpc/run_target_oriented_distribution_aware_training.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        root / "scripts/hpc/slurm_target_oriented_distribution_aware_training_250.sh"
    ).read_text(encoding="utf-8")
    assert "gradient_parameters=adapter" in runner
    assert "gradient_steps=free_gradient_steps" in runner
    assert "combine_adapter_gradients" in runner
    assert "write_distribution_aware_recovery" in runner
    assert "write_distribution_aware_checkpoint" in runner
    assert '"validation_values_evaluated": False' in runner
    assert '"retained_parent_parameters_trainable": False' in runner
    assert "stage-update-0250.pkl" in runner
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-290m}" in launcher
    assert "sbatch" not in launcher
    assert "#SBATCH --partition=gnall" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --cpus-per-task=8" in slurm
    assert "#SBATCH --time=05:00:00" in slurm
    assert "CNY 13.80 worst case" in slurm
