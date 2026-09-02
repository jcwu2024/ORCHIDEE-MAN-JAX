from __future__ import annotations

from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_recalibration import (
    DIRECT_RECALIBRATION_GROUPS,
    DIRECT_RECALIBRATION_UPDATES,
    FROZEN_CONTRACT_SHA256,
    REQUIRED_CONSECUTIVE_UPDATES,
    load_coupled_gpp_direct_recalibration_contract,
)


def test_direct_recalibration_contract_is_frozen_and_zero_retention():
    contract = load_coupled_gpp_direct_recalibration_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert tuple(contract.payload["identity"]["direct_recalibration_updates"]) == DIRECT_RECALIBRATION_UPDATES
    assert tuple(tuple(group) for group in contract.payload["data"]["direct_recalibration_artifact_groups"]) == (
        DIRECT_RECALIBRATION_GROUPS
    )
    assert contract.payload["evaluation"]["required_consecutive_direct_updates"] == REQUIRED_CONSECUTIVE_UPDATES
    assert contract.payload["direction"]["adamw_realization_allowed"] is False
    assert contract.payload["retention"]["checkpoint_written"] is False
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_direct_recalibration_runner_is_free_direct_and_zero_retention():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_direct_recalibration.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_direct_recalibration.sh").read_text(
        encoding="utf-8"
    )
    assert "gradient_direction_diagnostics" in runner
    assert "direct_virtual_step" in runner
    assert "DIRECT_RECALIBRATION_UPDATES" in runner
    assert "write_coupled_gpp_recovery" not in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in launcher
