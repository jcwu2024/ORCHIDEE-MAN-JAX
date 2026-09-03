from __future__ import annotations

from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_step_transfer import (
    DIRECT_GENERALIZATION_RESULT_SHA256,
    DISPLACEMENT_SCALES,
    FROZEN_CONTRACT_SHA256,
    GUARD_SCHEDULE_UPDATES,
    classify_step_transfer,
    load_coupled_gpp_direct_step_transfer_contract,
    select_largest_train_only_scale,
)


def _scale_records(passing_scale: float | None):
    records = []
    for scale in DISPLACEMENT_SCALES:
        passed = scale == passing_scale
        records.append(
            {
                "scale": scale,
                "scheduled_four_source_finite_step_accepted": passed,
                "scheduled_physical_hard_gate_passed": True,
                "guard_group_direct_gpp_relative_changes": [-0.01] * 4 if passed else [0.01] * 4,
                "combined_guard_direct_gpp_relative_change": -0.01 if passed else 0.01,
            }
        )
    return records


def test_direct_step_transfer_contract_is_frozen_before_guard_values():
    contract = load_coupled_gpp_direct_step_transfer_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.payload["identity"]["direct_generalization_result_sha256"] == DIRECT_GENERALIZATION_RESULT_SHA256
    assert contract.scales == DISPLACEMENT_SCALES
    assert tuple(contract.payload["data"]["train_only_guard_persistent_updates"]) == GUARD_SCHEDULE_UPDATES
    assert contract.payload["data"]["guard_outcomes_evaluated_before_contract_freeze"] is False
    assert contract.payload["data"]["fixed_monitor_excluded_from_scale_selection"] is True
    assert contract.payload["diagnosis"]["fixed_monitor_evaluated_only_after_train_only_scale_selection"] is True
    assert contract.payload["diagnosis"]["fixed_monitor_not_read_when_no_train_only_scale"] is True
    assert not any(contract.payload["retention"].values())
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_train_only_scale_selection_never_reads_monitor_change():
    records = _scale_records(0.125)
    assert select_largest_train_only_scale(records, relative_tolerance=1.0e-6) == 0.125
    assert select_largest_train_only_scale(_scale_records(None), relative_tolerance=1.0e-6) is None


def test_direct_step_transfer_classification_separates_selection_and_monitor():
    assert classify_step_transfer(0.125, monitor_relative_change=-0.01) == "train_only_scale_transfers_to_monitor"
    assert classify_step_transfer(0.125, monitor_relative_change=0.01) == "train_only_scale_fails_monitor"
    assert classify_step_transfer(None, monitor_relative_change=None) == "no_train_only_shared_finite_scale"
    with pytest.raises(ValueError, match="cannot select"):
        classify_step_transfer(None, monitor_relative_change=0.01)


def test_direct_step_transfer_runner_is_zero_retention_and_avoids_large_gram():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/audit_target_oriented_coupled_gpp_direct_step_transfer.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_direct_step_transfer.sh").read_text(
        encoding="utf-8"
    )
    assert "_bootstrap_update7" in runner
    assert "_replay_direction" in runner
    assert "_evaluate_scales" in runner
    assert "select_largest_train_only_scale" in runner
    assert "monitor_relative_change" not in runner.split("select_largest_train_only_scale(", 1)[1].split(")", 1)[0]
    assert "write_coupled_gpp_recovery" not in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    assert "cosine_matrix_from_gram" not in runner
    selection = runner.index("selected_scale = select_largest_train_only_scale(")
    monitor = runner.index('label="selected_monitor_before"')
    assert selection < monitor
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
    assert "sbatch" not in launcher
