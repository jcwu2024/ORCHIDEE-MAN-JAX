from __future__ import annotations

from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_fresh_moment import (
    ADMISSION_UPDATES,
    FROZEN_CONTRACT_SHA256,
    fresh_moment_admission_passed,
    load_coupled_gpp_fresh_moment_admission_contract,
)


def test_fresh_moment_admission_contract_is_frozen_and_bounded():
    contract = load_coupled_gpp_fresh_moment_admission_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.updates == ADMISSION_UPDATES
    assert contract.groups == (
        (326, 935, 542, 913, 897, 836, 887, 863),
        (968, 750, 987, 728, 953, 253, 646, 937),
    )
    assert contract.payload["retention"]["optimizer_updates_persisted"] == 0
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_fresh_moment_admission_requires_two_consecutive_updates():
    accepted = [
        {"optimizer_update": 2, "finite_step_accepted": True},
        {"optimizer_update": 3, "finite_step_accepted": True},
    ]
    assert fresh_moment_admission_passed(accepted)
    assert not fresh_moment_admission_passed(accepted[:1])
    assert not fresh_moment_admission_passed(accepted[:-1] + [{"optimizer_update": 3, "finite_step_accepted": False}])


def test_fresh_moment_admission_launcher_is_free_and_discarded():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_fresh_moment_admission.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_fresh_moment_admission.sh").read_text(
        encoding="utf-8"
    )
    assert "initialize_target_oriented_adamw(trainable)" in runner
    assert "expected_optimizer_state_step=1" in runner
    assert "write_coupled_gpp_recovery" not in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in launcher
