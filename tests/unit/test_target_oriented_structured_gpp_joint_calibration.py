from pathlib import Path

import jax
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    FROZEN_CONTRACT_SHA256,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
    combined_gradient_p90_clip,
    load_structured_gpp_joint_gradient_calibration_contract,
    resolve_structured_gpp_joint_gradient_coefficients,
)
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _combine_host_gradients,
    _tree_cosine,
)


def _records(seven_norms=range(1, 9), thirty_norms=range(2, 10)):
    updates = (352, 353, 354, 355, 356, 357, 359, 360)
    return [
        {
            "artifact_update": update,
            "source_gradients": {
                DIRECT_SOURCE: {"l2_norm": 10.0},
                SEVEN_SOURCE: {"l2_norm": float(seven)},
                THIRTY_SOURCE: {"l2_norm": float(thirty)},
            },
        }
        for update, seven, thirty in zip(updates, seven_norms, thirty_norms, strict=True)
    ]


def test_structured_gpp_joint_calibration_contract_is_exact_and_train_only():
    contract = load_structured_gpp_joint_gradient_calibration_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.calibration_updates == (352, 353, 354, 355, 356, 357, 359, 360)
    assert contract.target_gradient_norm_ratios == {
        SEVEN_SOURCE: 0.25,
        THIRTY_SOURCE: 0.25,
    }
    assert contract.payload["parameter_ownership"]["optimizer_created"] is False
    assert contract.payload["parameter_ownership"]["optimizer_updates_executed"] == 0
    assert contract.payload["execution"]["joint_training_authorized"] is False
    assert contract.payload["execution"]["validation_read_authorized"] is False


def test_structured_gpp_joint_coefficients_use_per_source_train_medians():
    contract = load_structured_gpp_joint_gradient_calibration_contract()
    result = resolve_structured_gpp_joint_gradient_coefficients(_records(), contract)
    seven_candidates = [2.5 / value for value in range(1, 9)]
    thirty_candidates = [2.5 / value for value in range(2, 10)]
    assert result.resolved_coefficients[SEVEN_SOURCE] == pytest.approx(np.median(seven_candidates))
    assert result.resolved_coefficients[THIRTY_SOURCE] == pytest.approx(np.median(thirty_candidates))
    assert result.record_count == 8
    assert result.coefficients_inside_interval
    expected_achieved = np.median(
        [result.resolved_coefficients[SEVEN_SOURCE] * value / 10.0 for value in range(1, 9)]
    )
    assert result.source_distributions[SEVEN_SOURCE]["achieved_ratio_median"] == pytest.approx(
        expected_achieved
    )


def test_structured_gpp_joint_coefficient_interval_failure_remains_reportable():
    contract = load_structured_gpp_joint_gradient_calibration_contract()
    result = resolve_structured_gpp_joint_gradient_coefficients(
        _records(seven_norms=(1.0e8,) * 8, thirty_norms=(1.0e8,) * 8),
        contract,
    )
    assert not result.coefficients_inside_interval
    assert result.resolved_coefficients[SEVEN_SOURCE] == pytest.approx(2.5e-8)


def test_structured_gpp_combined_clip_requires_eight_positive_norms():
    assert combined_gradient_p90_clip(range(1, 9), minimum=1.0e-8) == pytest.approx(7.3)
    with pytest.raises(ValueError, match="eight finite positive norms"):
        combined_gradient_p90_clip((1.0,) * 7, minimum=1.0e-8)


def test_structured_gpp_host_gradient_combination_and_cosines():
    gradients = {
        DIRECT_SOURCE: {"a": np.asarray([1.0, 0.0], dtype=np.float32)},
        SEVEN_SOURCE: {"a": np.asarray([0.0, 2.0], dtype=np.float32)},
        THIRTY_SOURCE: {"a": np.asarray([1.0, 1.0], dtype=np.float32)},
    }
    combined = _combine_host_gradients(
        jax,
        gradients,
        {SEVEN_SOURCE: 0.5, THIRTY_SOURCE: 0.25},
    )
    np.testing.assert_allclose(combined["a"], [1.25, 1.25])
    assert _tree_cosine(jax, gradients[DIRECT_SOURCE], gradients[SEVEN_SOURCE]) == pytest.approx(0.0)
    assert _tree_cosine(jax, gradients[DIRECT_SOURCE], gradients[THIRTY_SOURCE]) == pytest.approx(2**-0.5)


def test_structured_gpp_joint_calibration_runner_is_zero_update_and_free_gpu_only():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root / "scripts" / "hpc" / "calibrate_target_oriented_structured_gpp_joint_gradients.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root / "scripts" / "hpc" / "run_target_oriented_structured_gpp_joint_gradient_calibration.sh"
    ).read_text(encoding="utf-8")
    assert "make_trainable_gradient_steps" in runner
    assert "load_structured_gpp_checkpoint" in runner
    assert '"optimizer_created": False' in runner
    assert '"optimizer_updates_executed": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert "initialize_target_oriented_adamw" not in runner
    assert "write_structured_gpp_checkpoint" not in runner
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-240m}" in launcher
    assert "free gln01" in launcher
