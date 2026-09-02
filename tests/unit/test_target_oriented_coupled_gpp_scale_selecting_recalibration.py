from __future__ import annotations

from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_scale_selecting_recalibration import (
    CONTINUED_GROUPS,
    CONTINUED_UPDATES,
    EXPECTED_UPDATE5_SELECTED_SCALE,
    FROZEN_CONTRACT_SHA256,
    REQUIRED_CONSECUTIVE_UPDATES,
    load_coupled_gpp_scale_selecting_recalibration_contract,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selecting_recalibration import (
    _selected_update5_record,
)


def test_scale_selecting_recalibration_contract_is_frozen_and_bounded():
    contract = load_coupled_gpp_scale_selecting_recalibration_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert tuple(contract.payload["identity"]["continued_updates"]) == CONTINUED_UPDATES
    assert tuple(tuple(group) for group in contract.payload["data"]["continued_artifact_groups"]) == CONTINUED_GROUPS
    assert contract.payload["evaluation"]["required_consecutive_scale_selected_updates"] == (
        REQUIRED_CONSECUTIVE_UPDATES
    )
    assert contract.scales == (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    assert contract.payload["scale_selection"]["evaluate_smaller_scales_after_first_acceptance"] is False
    assert contract.payload["retention"]["checkpoint_written"] is False
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_scale_selecting_runner_is_free_and_zero_retention():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_scale_selecting_recalibration.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_scale_selecting_recalibration.sh").read_text(
        encoding="utf-8"
    )
    assert "_run_scale_audit(args, write_output=False, return_internal=True)" in runner
    assert "for scale in contract.scales" in runner
    assert 'if current["finite_step_accepted"] is True' in runner
    assert "write_coupled_gpp_recovery" not in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher


def test_selected_update5_record_requires_unique_passing_quarter_scale():
    report = {
        "update_5": {
            "scales": [
                {"scale": 0.5, "finite_step_accepted": False},
                {"scale": EXPECTED_UPDATE5_SELECTED_SCALE, "finite_step_accepted": True},
                {"scale": 0.125, "finite_step_accepted": True},
            ]
        }
    }
    selected = _selected_update5_record(report)
    assert selected["scale"] == EXPECTED_UPDATE5_SELECTED_SCALE
