from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    calibrate_teacher_guard_coefficient,
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
