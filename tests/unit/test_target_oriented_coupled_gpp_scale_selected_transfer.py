from __future__ import annotations

from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_bounded_training import (
    MONITOR_UPDATES,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_scale_selected_transfer import (
    FROZEN_CONTRACT_SHA256,
    PRIMARY_SOURCES,
    load_coupled_gpp_scale_selected_transfer_contract,
    transfer_accepted,
    transfer_changes,
)


def test_scale_selected_transfer_contract_is_frozen_and_independent():
    contract = load_coupled_gpp_scale_selected_transfer_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.monitor_updates == MONITOR_UPDATES
    assert contract.payload["data"]["gradient_or_scale_selection_use_of_monitor_artifacts"] is False
    assert contract.payload["evaluation"]["per_artifact_unanimity_required"] is False
    assert contract.payload["retention"]["checkpoint_written"] is False
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_transfer_gate_requires_both_horizon_means_to_improve():
    before = {
        "seven_day_component_resolved_science": 2.0,
        "thirty_day_component_resolved_science": 4.0,
        "direct_gpp_208_teacher_days": 1.0,
        "reachable_state_anchor_32_teacher_days": 3.0,
    }
    passing = transfer_changes(before, before | {PRIMARY_SOURCES[0]: 1.9, PRIMARY_SOURCES[1]: 3.9})
    failing = transfer_changes(before, before | {PRIMARY_SOURCES[0]: 1.9, PRIMARY_SOURCES[1]: 4.1})
    assert transfer_accepted(passing) is True
    assert transfer_accepted(failing) is False


def test_transfer_runner_reconstructs_without_checkpoint_or_paid_submission():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_scale_selected_transfer.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_scale_selected_transfer.sh").read_text(
        encoding="utf-8"
    )
    assert "write_output=False" in runner
    assert "return_internal=True" in runner
    assert "_evaluate_artifacts" in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    assert "write_coupled_gpp_recovery" not in runner
    assert "RUN_KIND=transfer" in launcher
    assert "sbatch" not in launcher
