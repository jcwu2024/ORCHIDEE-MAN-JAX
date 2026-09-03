from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_generalization import (
    AUDITED_PERSISTENT_UPDATES,
    FROZEN_CONTRACT_SHA256,
    PERSISTENT_RESULT_SHA256,
    SELECTED_SCALES,
    cosine_matrix_from_gram,
    directional_attribution,
    finite_loss_changes,
    first_adjacent_regression,
    load_coupled_gpp_direct_generalization_contract,
)


def test_direct_generalization_contract_is_frozen_and_zero_retention():
    contract = load_coupled_gpp_direct_generalization_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.payload["identity"]["persistent_result_sha256"] == PERSISTENT_RESULT_SHA256
    assert contract.audited_updates == AUDITED_PERSISTENT_UPDATES
    assert tuple(contract.payload["trajectory"]["selected_scales_from_result"]) == SELECTED_SCALES
    assert contract.payload["trajectory"]["persistent_recovery_read"] is False
    assert contract.payload["trajectory"]["repeat_scale_search"] is False
    assert contract.payload["trajectory"]["bootstrap_persistent_result_relative_difference_maximum"] == 0.01
    assert contract.payload["trajectory"]["per_update_and_endpoint_persistent_result_loss_comparison"] == "report_only"
    assert not any(contract.payload["retention"].values())
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_first_adjacent_regression_uses_strict_transition_order():
    assert first_adjacent_regression((1.0, 0.9, 0.91, 0.8)) == 2
    assert first_adjacent_regression((1.0, 1.0, 0.9)) is None
    assert first_adjacent_regression((1.0, 1.0000001), relative_tolerance=1.0e-6) is None
    with pytest.raises(ValueError, match="finite and positive"):
        first_adjacent_regression((1.0, np.nan))


def test_direct_generalization_cosines_and_finite_changes_are_exact():
    gram = np.asarray([[4.0, -3.0], [-3.0, 9.0]])
    cosines = cosine_matrix_from_gram(gram)
    assert np.allclose(cosines, [[1.0, -0.5], [-0.5, 1.0]])
    changes = finite_loss_changes((2.0, 4.0), (1.0, 5.0))
    assert changes[0]["relative_change"] == -0.5
    assert changes[0]["improved"] is True
    assert changes[1]["relative_change"] == 0.25
    assert changes[1]["improved"] is False


def test_directional_attribution_separates_sign_and_finite_curvature():
    mismatch = directional_attribution(
        training_monitor_mean_cosine=-0.2,
        training_applied_descent_cosine=0.4,
        monitor_applied_descent_cosine=-0.1,
        monitor_mean_relative_change=0.05,
        negative_within_training_pairs=1,
        negative_within_monitor_pairs=2,
        negative_cross_pairs=20,
    )
    assert mismatch["primary"] == "scheduled_batch_monitor_direction_mismatch"
    assert mismatch["artifact_direction_conflict_observed"] is True
    assert mismatch["sampling_variance_separable_from_population_heterogeneity"] is False

    curvature = directional_attribution(
        training_monitor_mean_cosine=0.2,
        training_applied_descent_cosine=0.4,
        monitor_applied_descent_cosine=0.1,
        monitor_mean_relative_change=0.05,
        negative_within_training_pairs=0,
        negative_within_monitor_pairs=0,
        negative_cross_pairs=0,
    )
    assert curvature["primary"] == "finite_step_curvature_or_nonlinearity"


def test_direct_generalization_runner_is_read_only_and_direct_only_at_attribution():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/audit_target_oriented_coupled_gpp_direct_generalization.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_direct_generalization.sh").read_text(
        encoding="utf-8"
    )
    assert "_run_direct_recalibration(args, write_output=False, return_internal=True)" in runner
    assert "_run_scale_selecting" not in runner
    assert "BOOTSTRAP_SELECTED_SCALES = (1.0, 1.0, 0.25, 1.0, 0.5)" in runner
    assert "_measure_source_gradients" in runner
    assert "_direct_gradient" in runner
    assert "first_adjacent_regression" in runner
    assert "write_persistent_recovery" not in runner
    assert "write_persistent_checkpoint" not in runner
    assert "load_persistent_recovery" not in runner
    assert "load_persistent_checkpoint" not in runner
    assert '"training_updates_persisted": 0' in runner
    assert '"forcing_records_reconstructed": 0' in runner
    assert '"subdaily_state_scan_steps": 0' in runner
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-180m}" in launcher
    assert "target-oriented-coupled-gpp-persistent-14911228/report.json" in launcher
    assert "audit_target_oriented_coupled_gpp_direct_generalization" in launcher
    assert "sbatch" not in launcher
