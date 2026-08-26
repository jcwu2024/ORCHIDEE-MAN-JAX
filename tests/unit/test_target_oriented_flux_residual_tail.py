from __future__ import annotations

from pathlib import Path

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_flux_residual_tail import (
    FROZEN_CONTRACT_SHA256,
    bounded_influence_weights,
    classify_tail_mechanism,
    contribution_summary,
    gram_direction_summary,
    influence_candidates,
    load_target_oriented_flux_residual_tail_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def test_target_oriented_flux_residual_tail_contract_is_frozen():
    contract = load_target_oriented_flux_residual_tail_contract()

    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.top_counts == (1, 5, 10)
    assert contract.scale_multipliers == (1.0, 4.0, 16.0)
    assert contract.payload["data"]["total_seven_day_windows"] == 64
    assert contract.payload["execution"]["optimizer_updates"] == 0


def test_contribution_summary_reports_tail_share_and_effective_size():
    summary = contribution_summary([8.0, 1.0, 1.0], top_counts=(1, 2, 3))

    assert summary["top_shares"] == {"1": 0.8, "2": 0.9, "3": 1.0}
    assert np.isclose(summary["effective_sample_size"], 100.0 / 66.0)
    assert np.isclose(summary["positive_median"], 1.0)


def test_bounded_influence_weights_downweight_high_loss_smoothly():
    scores = [1.0, 4.0, 16.0]
    equal = bounded_influence_weights(scores, family="equal_window")
    soft = bounded_influence_weights(scores, family="soft_l1", tau=1.0)
    logarithmic = bounded_influence_weights(scores, family="log1p", tau=1.0)

    assert np.allclose(equal, np.full(3, 1.0 / 3.0))
    assert np.isclose(np.sum(soft), 1.0)
    assert np.isclose(np.sum(logarithmic), 1.0)
    assert soft[0] > soft[1] > soft[2]
    assert logarithmic[0] > logarithmic[1] > logarithmic[2]
    assert logarithmic[2] < soft[2]


def test_influence_candidates_are_fixed_by_positive_median():
    contract = load_target_oriented_flux_residual_tail_contract()
    candidates = influence_candidates([0.0, 2.0, 4.0, 8.0], contract)

    assert tuple(candidates) == (
        "equal_window",
        "soft_l1_x1",
        "soft_l1_x4",
        "soft_l1_x16",
        "log1p_x1",
        "log1p_x4",
        "log1p_x16",
    )
    assert candidates["soft_l1_x1"]["tau"] == 4.0
    assert np.isclose(np.sum(candidates["log1p_x16"]["weights"]), 1.0)


def test_gram_direction_summary_uses_realized_weighted_geometry():
    gram = np.asarray([[4.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    summary = gram_direction_summary(gram, [1.0, 1.0])

    assert np.isclose(summary["direction_norm"], np.sqrt(1.25))
    assert np.allclose(
        summary["constraint_cosines"],
        [1.0 / np.sqrt(1.25), 0.5 / np.sqrt(1.25)],
    )
    assert summary["negative_constraint_fraction"] == 0.0


def test_tail_classifier_distinguishes_loss_and_gradient_concentration():
    contract = load_target_oriented_flux_residual_tail_contract()
    result = classify_tail_mechanism(
        [100.0, 0.1, *([1.0] * 18)],
        [1.0, 90.0, *([1.0] * 18)],
        [0] * 10 + [1] * 10,
        contract,
    )

    assert result["localized_loss_tail"] is True
    assert result["gradient_norm_tail"] is True
    assert 1 in result["jacobian_amplified_window_indices"]


def test_residual_tail_runner_is_zero_training_and_uses_standalone_gradient():
    runner = (ROOT / "scripts" / "hpc" / "audit_target_oriented_flux_residual_tail.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "hpc" / "run_target_oriented_flux_residual_tail_audit.sh").read_text(
        encoding="utf-8"
    )
    preflight = (ROOT / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py").read_text(encoding="utf-8")

    assert "target_oriented_adamw_update" not in runner
    assert '"optimizer_updates_retained": 0' in runner
    assert '"parameter_steps": 0' in runner
    assert "verify_validation_artifacts=False" in runner
    assert "seven_flux_window_gradient" in preflight
    assert "thirty_flux_window_gradient" in preflight
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "gnall" not in launcher
