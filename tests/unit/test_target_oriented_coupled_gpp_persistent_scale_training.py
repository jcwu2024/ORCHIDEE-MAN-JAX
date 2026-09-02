from __future__ import annotations

from pathlib import Path

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_checkpoint import (
    load_persistent_checkpoint,
    load_persistent_recovery,
    write_persistent_checkpoint,
    write_persistent_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    ALL_REPORTED_KEYS,
    FROZEN_CONTRACT_SHA256,
    SCHEDULE_SHA256,
    build_persistent_scale_training_schedule,
    load_coupled_gpp_persistent_scale_training_contract,
    monitor_gate_decision,
    persistent_training_pool,
)


def _sources(direct: float = 0.06) -> dict[str, float]:
    return {
        DIRECT_SOURCE: direct,
        STATE_SOURCE: 0.025,
        SEVEN_SOURCE: 0.08,
        THIRTY_SOURCE: 0.09,
    }


def _monitor(*, update: int | None = None, checkpoint: bool = False):
    acceptance = {
        horizon: {
            "horizon_days": int(horizon),
            "keys": {
                name: {
                    "normalized_rmse": 0.1,
                    "normalized_signed_bias": 0.01,
                    "evaluated_values": 8.0,
                }
                for name in ALL_REPORTED_KEYS
            },
        }
        for horizon in ("7", "30")
    }
    panel = {
        "records": [{"artifact_update": value} for value in range(8)],
        "source_mean_losses": _sources(),
        "acceptance": acceptance,
        "cumulative_gpp_scores": {"7": 0.00505, "30": 0.00505},
        "hard_gate_passed": True,
    }
    if update is not None:
        panel.update(
            persistent_update=update,
            gate={"checkpoint_passed": checkpoint, "stop_stage": False},
        )
    return panel


def test_persistent_contract_and_schedule_are_frozen():
    contract = load_coupled_gpp_persistent_scale_training_contract()
    schedule = build_persistent_scale_training_schedule()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert schedule.sha256 == SCHEDULE_SHA256
    assert len(schedule.groups) == 100
    assert len(schedule.pool) == len(persistent_training_pool()) == 678
    assert schedule.groups[0] == (552, 744, 692, 706, 763, 949, 846, 292)
    assert all(len(group) == len(set(group)) == 8 for group in schedule.groups)
    assert not ({value for group in schedule.groups for value in group} - set(schedule.pool))
    assert contract.scales == (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    assert contract.monitor_updates == (251, 358, 465, 572, 679, 786, 893, 1000)


def test_persistent_monitor_gate_recovers_direct_gpp_by_deadline():
    contract = load_coupled_gpp_persistent_scale_training_contract()
    initial_gpp = {"7": 0.02, "30": 0.03}
    current_gpp = {"7": 0.019, "30": 0.029}

    update_10 = monitor_gate_decision(
        contract,
        persistent_update=10,
        current_sources=_sources(),
        initial_gpp_scores=initial_gpp,
        current_gpp_scores=current_gpp,
        physical_gate_passed=True,
        checkpoint_already_selected=False,
    )
    assert update_10["continuation_passed"] is True
    assert update_10["checkpoint_passed"] is False
    assert update_10["stop_stage"] is False

    update_25 = monitor_gate_decision(
        contract,
        persistent_update=25,
        current_sources=_sources(),
        initial_gpp_scores=initial_gpp,
        current_gpp_scores=current_gpp,
        physical_gate_passed=True,
        checkpoint_already_selected=False,
    )
    assert update_25["checkpoint_deadline_missed"] is True
    assert update_25["stop_stage"] is True

    recovered = monitor_gate_decision(
        contract,
        persistent_update=25,
        current_sources=_sources(direct=0.045),
        initial_gpp_scores=initial_gpp,
        current_gpp_scores=current_gpp,
        physical_gate_passed=True,
        checkpoint_already_selected=False,
    )
    assert recovered["checkpoint_passed"] is True
    assert recovered["stop_stage"] is False


def test_persistent_parameter_only_recovery_and_checkpoint_roundtrip(tmp_path: Path):
    identity = {"contract": "frozen"}
    trainable = {"structured_gpp": np.asarray([1.0, 2.0], dtype=np.float32)}
    records = [
        {
            "persistent_update": value,
            "selected_scale": 0.5,
            "finite_step_accepted": True,
        }
        for value in range(1, 11)
    ]
    milestone = _monitor(update=10, checkpoint=True)
    recovery = tmp_path / "recovery.pkl"
    write_persistent_recovery(
        recovery,
        identity=identity,
        trainable=trainable,
        completed_updates=10,
        parameter_count=2,
        update_records=records,
        initial_monitor=_monitor(),
        milestone_monitors=(milestone,),
    )
    loaded = load_persistent_recovery(
        recovery,
        identity=identity,
        parameter_count=2,
        maximum_updates=100,
    )
    assert loaded is not None
    assert loaded["completed_updates"] == 10
    assert "optimizer" not in loaded

    checkpoint = tmp_path / "stage-update-0010.pkl"
    write_persistent_checkpoint(
        checkpoint,
        identity=identity,
        trainable=trainable,
        completed_updates=10,
        parameter_count=2,
        update_records=records,
        initial_monitor=_monitor(),
        milestone_monitors=(milestone,),
    )
    selected = load_persistent_checkpoint(
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        identity=identity,
        completed_updates=10,
        parameter_count=2,
    )
    assert selected["completed_updates"] == 10
    assert selected["promotion_eligible"] is False


def test_persistent_runner_reuses_rollout_for_detailed_monitor_and_has_admission():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_full_step_preflight.py").read_text(encoding="utf-8")
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_persistent_scale_training.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_persistent_scale_training.sh").read_text(
        encoding="utf-8"
    )
    slurm = (root / "scripts/hpc/slurm_target_oriented_coupled_gpp_persistent_scale_training.sh").read_text(
        encoding="utf-8"
    )
    assert "window_monitor_statistics" in shared
    assert '"seven_monitor_forward"' in shared
    assert "add_target_oriented_acceptance_statistics" in runner
    assert "for scale in contract.scales" in runner
    assert "write_persistent_recovery" in runner
    assert "write_persistent_checkpoint" in runner
    assert "--admission-only" in runner
    assert 'optimizer_state_used": False' in runner
    assert "sbatch" not in launcher
    assert "ADMISSION_ONLY=${ADMISSION_ONLY:-1}" in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
    assert "#SBATCH --partition=gnall" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=16:00:00" in slurm
    assert "CNY 44.16 worst case" in slurm
