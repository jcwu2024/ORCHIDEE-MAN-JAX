from __future__ import annotations

from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_update5_scale_audit import (
    DISPLACEMENT_SCALES,
    FROZEN_CONTRACT_SHA256,
    UPDATE_FIVE_ARTIFACTS,
    load_coupled_gpp_update5_scale_audit_contract,
)


def test_update5_scale_audit_contract_is_frozen_and_bounded():
    contract = load_coupled_gpp_update5_scale_audit_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.scales == DISPLACEMENT_SCALES
    assert tuple(contract.payload["data"]["audit_artifact_updates"]) == UPDATE_FIVE_ARTIFACTS
    assert contract.payload["direction"]["parameter_dtype_preserved"] is True
    assert contract.payload["retention"]["checkpoint_written"] is False
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_update5_scale_audit_runner_is_free_and_zero_retention():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_update5_scale_audit.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_update5_scale_audit.sh").read_text(encoding="utf-8")
    assert "_run_smoke(args, write_output=False, return_internal=True)" in runner
    assert "_evaluate_scales" in runner
    assert "write_coupled_gpp_recovery" not in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in launcher
