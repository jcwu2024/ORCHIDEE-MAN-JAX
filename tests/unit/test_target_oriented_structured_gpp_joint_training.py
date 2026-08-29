import hashlib
from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_training import (
    FROZEN_CONTRACT_SHA256,
    SCHEDULE_SHA256,
    build_joint_training_schedule,
    candidate_artifact_updates,
    epoch_panel_continuation_passed,
    load_structured_gpp_joint_training_contract,
    panel_source_changes,
    selected_epoch_from_panels,
    validate_joint_training_recovery_metadata,
)
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import (
    _make_proposal_check,
    _validate_recovered_checkpoint_files,
)


def _losses(direct, seven, thirty):
    return {DIRECT_SOURCE: direct, SEVEN_SOURCE: seven, THIRTY_SOURCE: thirty}


def test_structured_gpp_joint_training_contract_freezes_batch4_epochs():
    contract = load_structured_gpp_joint_training_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.payload["data"]["effective_batch_size"] == 4
    assert contract.payload["data"]["epochs_maximum"] == 4
    assert contract.payload["data"]["optimizer_updates_per_epoch"] == 62
    assert contract.payload["data"]["optimizer_updates_maximum"] == 248
    assert contract.payload["optimizer"]["initial_optimizer_step"] == 250
    assert contract.payload["optimizer"]["maximum_final_optimizer_step"] == 498
    assert contract.payload["execution"]["paid_training_authorized_by_contract_alone"] is False


def test_structured_gpp_joint_training_schedule_is_exact_and_full_coverage():
    schedule = build_joint_training_schedule()
    assert schedule.sha256 == SCHEDULE_SHA256
    assert len(schedule.groups_by_epoch) == 4
    assert all(len(epoch) == 62 for epoch in schedule.groups_by_epoch)
    assert all(len(group) == 4 for group in schedule.flat_groups)
    assert schedule.dropped_by_epoch == (
        (760, 714),
        (642, 686),
        (723, 671),
        (536, 578),
    )
    candidates = set(candidate_artifact_updates())
    assert len(candidates) == 250
    assert {value for group in schedule.flat_groups for value in group} == candidates
    for epoch, dropped in zip(schedule.groups_by_epoch, schedule.dropped_by_epoch, strict=True):
        used = [value for group in epoch for value in group]
        assert len(used) == len(set(used)) == 248
        assert set(used).isdisjoint(dropped)
        assert set(used) | set(dropped) == candidates


def test_structured_gpp_joint_training_panel_gate_requires_all_sources():
    before = _losses(2.0, 1.0, 0.5)
    passing = panel_source_changes(before, _losses(1.9, 0.9, 0.49))
    assert epoch_panel_continuation_passed(passing)
    failing = panel_source_changes(before, _losses(1.9, 1.01, 0.49))
    assert not epoch_panel_continuation_passed(failing)
    with pytest.raises(ValueError, match="panel loss is invalid"):
        panel_source_changes(before, _losses(1.9, float("nan"), 0.49))


def test_structured_gpp_joint_training_selects_last_consecutive_passing_epoch():
    panels = [
        {"epoch": 1, "continuation_gate_passed": True},
        {"epoch": 2, "continuation_gate_passed": True},
        {"epoch": 3, "continuation_gate_passed": False},
    ]
    assert selected_epoch_from_panels(panels) == 2
    assert selected_epoch_from_panels([{"epoch": 1, "continuation_gate_passed": False}]) == 0
    with pytest.raises(ValueError, match="order drift"):
        selected_epoch_from_panels([{"epoch": 2, "continuation_gate_passed": True}])
    with pytest.raises(ValueError, match="continues after failure"):
        selected_epoch_from_panels(
            [
                {"epoch": 1, "continuation_gate_passed": False},
                {"epoch": 2, "continuation_gate_passed": True},
            ]
        )


def _epoch_panel(epoch: int, passed: bool = True) -> dict:
    before = _losses(2.0, 1.0, 0.5)
    after = _losses(1.9, 0.9, 0.49) if passed else _losses(1.9, 1.01, 0.49)
    return {
        "epoch": epoch,
        "absolute_optimizer_step": 250 + 62 * epoch,
        "source_changes_from_previous_panel": panel_source_changes(before, after),
        "continuation_gate_passed": passed,
        "panel": {"source_mean_losses": after},
    }


def _epoch_checkpoint(epoch: int) -> dict:
    return {
        "epoch": epoch,
        "completed_updates": 250 + 62 * epoch,
        "path": f"stage-update-{250 + 62 * epoch:04d}.pkl",
        "sha256": "a" * 64,
    }


def test_structured_gpp_joint_training_recovery_metadata_matches_progress():
    metadata = {
        "initial_joint_panel": {"source_mean_losses": _losses(2.0, 1.0, 0.5)},
        "epoch_panels": [_epoch_panel(1), _epoch_panel(2)],
        "epoch_checkpoints": [_epoch_checkpoint(1), _epoch_checkpoint(2)],
    }
    result = validate_joint_training_recovery_metadata(
        metadata, completed_joint_updates=130
    )
    assert result["selected_epoch"] == 2

    with pytest.raises(ValueError, match="epoch-panel coverage drift"):
        validate_joint_training_recovery_metadata(
            metadata, completed_joint_updates=61
        )

    failed = dict(
        metadata,
        epoch_panels=[_epoch_panel(1), _epoch_panel(2, passed=False)],
        epoch_checkpoints=[_epoch_checkpoint(1)],
    )
    stopped = validate_joint_training_recovery_metadata(
        failed, completed_joint_updates=124
    )
    assert stopped["selected_epoch"] == 1
    with pytest.raises(ValueError, match="continued after failed epoch"):
        validate_joint_training_recovery_metadata(
            failed, completed_joint_updates=125
        )


def test_structured_gpp_joint_training_recovery_rejects_checkpoint_mismatch():
    metadata = {
        "initial_joint_panel": {"source_mean_losses": _losses(2.0, 1.0, 0.5)},
        "epoch_panels": [_epoch_panel(1)],
        "epoch_checkpoints": [],
    }
    with pytest.raises(ValueError, match="checkpoint coverage drift"):
        validate_joint_training_recovery_metadata(
            metadata, completed_joint_updates=62
        )


def test_structured_gpp_joint_training_recovery_validates_checkpoint_file(tmp_path):
    checkpoint = tmp_path / "stage-update-0312.pkl"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    record = {"path": str(checkpoint), "sha256": digest}
    _validate_recovered_checkpoint_files([record], tmp_path.resolve())
    with pytest.raises(ValueError, match="checkpoint file drift"):
        _validate_recovered_checkpoint_files(
            [dict(record, sha256="0" * 64)], tmp_path.resolve()
        )
    with pytest.raises(ValueError, match="checkpoint file drift"):
        _validate_recovered_checkpoint_files([record], (tmp_path / "other").resolve())


def test_structured_gpp_joint_training_runner_checks_optimizer_proposals():
    runner = (
        Path(__file__).resolve().parents[2]
        / "scripts/hpc/run_target_oriented_structured_gpp_joint_training.py"
    ).read_text(encoding="utf-8")
    assert "_make_proposal_check" in runner
    assert '"parameter_nonfinite_count"' in runner
    assert '"optimizer_nonfinite_count"' in runner
    assert '"parameter_delta_l2"' in runner
    assert "_validate_recovered_checkpoint_files" in runner


def test_structured_gpp_joint_training_proposal_check_detects_invalid_state():
    import jax
    import jax.numpy as jnp

    check = _make_proposal_check(jax, jnp)
    before = {"kernel": jnp.asarray([1.0, 2.0], dtype=jnp.float32)}
    after = {"kernel": jnp.asarray([1.0, 2.5], dtype=jnp.float32)}
    optimizer = {
        "step": jnp.asarray(251, dtype=jnp.int32),
        "first": jnp.asarray([0.0, 0.1], dtype=jnp.float32),
    }
    parameter_nonfinite, optimizer_nonfinite, delta_squared = check(
        before, after, optimizer
    )
    assert int(parameter_nonfinite) == 0
    assert int(optimizer_nonfinite) == 0
    assert float(delta_squared) == pytest.approx(0.25)

    invalid_after = {
        "kernel": jnp.asarray([1.0, float("nan")], dtype=jnp.float32)
    }
    invalid_optimizer = dict(
        optimizer, first=jnp.asarray([0.0, float("inf")], dtype=jnp.float32)
    )
    parameter_nonfinite, optimizer_nonfinite, _ = check(
        before, invalid_after, invalid_optimizer
    )
    assert int(parameter_nonfinite) == 1
    assert int(optimizer_nonfinite) == 1


def test_structured_gpp_joint_training_launchers_are_paid_bounded_and_resumable():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts/hpc/run_target_oriented_structured_gpp_joint_training.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        root / "scripts/hpc/slurm_target_oriented_structured_gpp_joint_training.sh"
    ).read_text(encoding="utf-8")
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-590m}" in launcher
    assert "target-oriented-structured-gpp-joint-calibration" in launcher
    assert "--joint-training-contract" in launcher
    assert "--output-dir" in launcher
    assert "#SBATCH --partition=gnall" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --cpus-per-task=8" in slurm
    assert "#SBATCH --time=10:00:00" in slurm
    assert "CNY 27.60 worst case" in slurm
    assert "OUTPUT_DIR=${OUTPUT_DIR:-" in slurm
