from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_bounded_training import (
    CALIBRATION_UPDATES,
    FROZEN_CONTRACT_SHA256,
    MONITOR_UPDATES,
    SCHEDULE_SHA256,
    build_coupled_gpp_training_schedule,
    eligible_training_artifacts,
    finite_step_accepted,
    load_coupled_gpp_bounded_training_contract,
    monitor_continuation_accepted,
    selected_milestone,
    source_changes,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_checkpoint import (
    load_coupled_gpp_checkpoint,
    load_coupled_gpp_recovery,
    write_coupled_gpp_checkpoint,
    write_coupled_gpp_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
)


def _losses(value: float) -> dict[str, float]:
    return {
        DIRECT_SOURCE: value,
        STATE_SOURCE: 2.0 * value,
        SEVEN_SOURCE: 3.0 * value,
        THIRTY_SOURCE: 4.0 * value,
    }


def _monitor(value: float, *, update: int | None = None, passed: bool = True):
    panel = {
        "records": [{"artifact_update": index} for index in range(8)],
        "source_mean_losses": _losses(value),
        "weighted_objective": {"joint": 10.0 * value},
    }
    if update is not None:
        panel.update(
            optimizer_update=update,
            continuation_gate_passed=passed,
        )
    return panel


def test_coupled_gpp_bounded_training_contract_and_schedule_are_exact():
    contract = load_coupled_gpp_bounded_training_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.learning_rate == 2.0e-6
    assert contract.clip_threshold == 12.808193184117403
    assert contract.monitor_updates == MONITOR_UPDATES
    schedule = build_coupled_gpp_training_schedule()
    assert schedule.sha256 == SCHEDULE_SHA256
    assert schedule.groups == (
        (389, 922, 967, 910, 850, 885, 489, 891),
        (326, 935, 542, 913, 897, 836, 887, 863),
        (968, 750, 987, 728, 953, 253, 646, 937),
        (481, 672, 900, 294, 998, 668, 433, 535),
        (965, 351, 272, 363, 660, 670, 548, 740),
        (332, 994, 933, 930, 699, 916, 387, 427),
        (675, 417, 455, 803, 779, 284, 385, 520),
        (293, 576, 290, 784, 743, 963, 414, 727),
    )
    flattened = [value for group in schedule.groups for value in group]
    assert len(flattened) == len(set(flattened)) == 64
    assert set(flattened).isdisjoint(MONITOR_UPDATES)
    assert set(flattened).isdisjoint(CALIBRATION_UPDATES)
    assert min(flattened) >= 251
    assert len(eligible_training_artifacts()) == 734


def test_coupled_gpp_finite_step_and_monitor_rules_are_bounded():
    contract = load_coupled_gpp_bounded_training_contract()
    before = _losses(1.0)
    assert finite_step_accepted(
        before,
        _losses(0.99),
        coefficients=contract.coefficients,
    )
    tolerated = dict(_losses(0.99), **{SEVEN_SOURCE: before[SEVEN_SOURCE] * 1.0000005})
    assert finite_step_accepted(
        before,
        tolerated,
        coefficients=contract.coefficients,
    )
    assert not monitor_continuation_accepted(
        before,
        tolerated,
        coefficients=contract.coefficients,
    )
    rejected = dict(_losses(0.99), **{SEVEN_SOURCE: before[SEVEN_SOURCE] * 1.001})
    assert not finite_step_accepted(
        before,
        rejected,
        coefficients=contract.coefficients,
    )
    changes = source_changes(before, _losses(0.9))
    assert all(value["relative_change"] == pytest.approx(-0.1) for value in changes.values())


def test_coupled_gpp_milestone_selection_stops_consecutively():
    assert selected_milestone([_monitor(0.9, update=4)]) == 4
    assert selected_milestone([_monitor(0.9, update=4), _monitor(0.8, update=8)]) == 8
    assert selected_milestone([_monitor(1.1, update=4, passed=False)]) == 0
    with pytest.raises(ValueError, match="continues after failed"):
        selected_milestone(
            [
                _monitor(1.1, update=4, passed=False),
                _monitor(0.8, update=8),
            ]
        )


def _state(step: int):
    trainable = {
        "structured_gpp": {"value": np.asarray([1.0, 2.0], dtype=np.float32)},
        "coupled_gpp_carbon_feedback": {"value": np.asarray([3.0], dtype=np.float32)},
    }
    optimizer = {
        "step": np.asarray(step, dtype=np.int32),
        "first": np.asarray([0.1], dtype=np.float32),
        "second": np.asarray([0.2], dtype=np.float32),
    }
    records = [{"optimizer_update": value} for value in range(1, step + 1)]
    return trainable, optimizer, records


def test_coupled_gpp_recovery_and_checkpoint_roundtrip(tmp_path: Path):
    identity = {"contract": "frozen"}
    trainable, optimizer, records = _state(1)
    recovery_path = tmp_path / "recovery.pkl"
    write_coupled_gpp_recovery(
        recovery_path,
        identity=identity,
        trainable=trainable,
        optimizer=optimizer,
        completed_updates=1,
        parameter_count=3,
        clip_threshold=12.0,
        update_records=records,
        initial_monitor=_monitor(1.0),
        milestone_monitors=(),
    )
    recovered = load_coupled_gpp_recovery(
        recovery_path,
        identity=identity,
        parameter_count=3,
        clip_threshold=12.0,
        maximum_updates=8,
    )
    assert recovered is not None
    assert recovered["completed_updates"] == 1

    trainable, optimizer, records = _state(4)
    checkpoint_path = tmp_path / "stage-update-0004.pkl"
    write_coupled_gpp_checkpoint(
        checkpoint_path,
        identity=identity,
        trainable=trainable,
        optimizer=optimizer,
        completed_updates=4,
        parameter_count=3,
        clip_threshold=12.0,
        update_records=records,
        initial_monitor=_monitor(1.0),
        milestone_monitors=(_monitor(0.9, update=4),),
    )
    loaded = load_coupled_gpp_checkpoint(
        checkpoint_path,
        expected_sha256=sha256_file(checkpoint_path),
        identity=identity,
        completed_updates=4,
        parameter_count=3,
        clip_threshold=12.0,
    )
    assert loaded["completed_updates"] == 4


def test_coupled_gpp_training_launchers_are_bounded_and_admissible():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_bounded_training.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_bounded_training.sh").read_text(encoding="utf-8")
    slurm = (root / "scripts/hpc/slurm_target_oriented_coupled_gpp_bounded_training.sh").read_text(encoding="utf-8")
    assert "--admission-only" in runner
    assert "finite_step_accepted" in runner
    assert "write_coupled_gpp_recovery" in runner
    assert "write_coupled_gpp_checkpoint" in runner
    assert "sbatch" not in launcher
    assert "ADMISSION_ONLY=${ADMISSION_ONLY:-0}" in launcher
    assert "ADMISSION_ARG=" in launcher
    assert "ADMISSION_ARGS" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-290m}" in launcher
    assert "#SBATCH --partition=gnall" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --cpus-per-task=8" in slurm
    assert "#SBATCH --time=05:00:00" in slurm
    assert "CNY 13.80 worst case" in slurm
