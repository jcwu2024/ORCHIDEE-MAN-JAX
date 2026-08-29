from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_batch4 import (
    ARTIFACT_GROUPS,
    FROZEN_CONTRACT_SHA256,
    RUNTIME_YEAR_GROUPS,
    load_structured_gpp_joint_batch4_contract,
    mean_source_losses,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)


def test_structured_gpp_batch4_contract_binds_positive_heterogeneity_result():
    contract = load_structured_gpp_joint_batch4_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.artifact_groups == ARTIFACT_GROUPS
    assert RUNTIME_YEAR_GROUPS == ((1973, 1969, 1989, 1988), (1972, 2003, 2002, 1987))
    assert contract.payload["identity"]["heterogeneity_report_sha256"] == (
        "15d993a2c184918c7b6feb246a246d4d0ee8ecabfcdefb240abfd0659c880eea"
    )
    assert contract.clip_threshold == 46.802861793665286


def test_structured_gpp_batch4_contract_freezes_two_independent_discarded_proposals():
    payload = load_structured_gpp_joint_batch4_contract().payload
    optimizer = payload["optimizer_proposals"]
    assert optimizer["proposal_count"] == 2
    assert optimizer["proposals_are_independent"] is True
    assert optimizer["initial_optimizer_step"] == 250
    assert optimizer["proposed_optimizer_step"] == 251
    assert optimizer["updates_persisted"] == 0
    assert optimizer["checkpoint_written"] is False
    assert optimizer["recovery_written"] is False
    assert payload["acceptance"]["per_artifact_loss_change_is_not_an_acceptance_gate"] is True
    assert payload["execution"]["paid_training_authorized"] is False


def test_structured_gpp_batch4_mean_source_losses_is_unweighted_arithmetic_mean():
    result = mean_source_losses(
        (
            {DIRECT_SOURCE: 1.0, SEVEN_SOURCE: 2.0, THIRTY_SOURCE: 3.0},
            {DIRECT_SOURCE: 3.0, SEVEN_SOURCE: 4.0, THIRTY_SOURCE: 5.0},
        )
    )
    assert result == pytest.approx(
        {DIRECT_SOURCE: 2.0, SEVEN_SOURCE: 3.0, THIRTY_SOURCE: 4.0}
    )
    with pytest.raises(ValueError, match="requires source losses"):
        mean_source_losses(())


def test_structured_gpp_batch4_runner_is_discarded_and_free_gpu_only():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_structured_gpp_joint_batch4_preflight.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_structured_gpp_joint_batch4_preflight.sh"
    ).read_text(encoding="utf-8")
    assert "for group_index, artifact_updates in enumerate(contract.artifact_groups)" in runner
    assert "target_oriented_adamw_update" in runner
    assert "optimizer_updates_persisted\": 0" in runner
    assert "per_artifact_changes_reported_not_thresholded" in runner
    assert "write_structured_gpp_checkpoint" not in runner
    assert "write_structured_gpp_recovery" not in runner
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
