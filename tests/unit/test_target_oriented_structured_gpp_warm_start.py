from pathlib import Path

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_checkpoint import (
    load_structured_gpp_checkpoint,
    load_structured_gpp_recovery,
    write_structured_gpp_checkpoint,
    write_structured_gpp_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    WARM_START_FROZEN_CONTRACT_SHA256,
    compare_structured_gpp_direct_panels,
    load_structured_gpp_warm_start_contract,
)


def _panel(loss: float, artifact_losses: tuple[float, ...]) -> dict:
    updates = (251, 358, 465, 572, 679, 786, 893, 1000)
    return {
        "updates": list(updates),
        "aggregate_direct_gpp_loss": loss,
        "evaluated_values": 1664,
        "records": [
            {
                "artifact_update": update,
                "direct_gpp_loss": artifact_loss,
                "evaluated_values": 208,
            }
            for update, artifact_loss in zip(updates, artifact_losses, strict=True)
        ],
    }


def test_structured_gpp_warm_start_contract_is_exact_and_disjoint():
    contract = load_structured_gpp_warm_start_contract()
    assert contract.sha256 == WARM_START_FROZEN_CONTRACT_SHA256
    assert len(contract.training_updates) == 250
    assert contract.training_updates[0] == 252
    assert contract.training_updates[-1] == 511
    assert contract.evaluation_updates == (251, 358, 465, 572, 679, 786, 893, 1000)
    assert not set(contract.training_updates) & set(contract.evaluation_updates)
    assert contract.optimizer["learning_rate"] == 4.0e-6
    assert contract.optimizer["global_gradient_norm_clip"]["threshold"] == 17.256861560753983
    assert contract.payload["execution"]["seven_or_thirty_day_gradients_authorized"] is False


def test_structured_gpp_direct_panel_gate_requires_large_and_broad_improvement():
    contract = load_structured_gpp_warm_start_contract()
    initial = _panel(2.0, (2.0,) * 8)
    final = _panel(1.5, (1.0,) * 6 + (3.0,) * 2)
    result = compare_structured_gpp_direct_panels(initial, final, contract)
    assert result["aggregate_direct_gpp_loss_reduction"] == 0.25
    assert result["individual_artifact_improvement_count"] == 6
    assert result["capability_gate_passed"]

    insufficient = compare_structured_gpp_direct_panels(initial, _panel(1.9, (1.0,) * 8), contract)
    assert not insufficient["capability_gate_passed"]


def test_structured_gpp_checkpoint_and_recovery_round_trip(tmp_path):
    identity = {"contract": "frozen"}
    specialist = {"kernel": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)}
    optimizer = {
        "step": np.asarray(2, dtype=np.int32),
        "first": {"kernel": np.zeros(3, dtype=np.float32)},
        "second": {"kernel": np.ones(3, dtype=np.float32)},
    }
    records = [{"optimizer_update": 1}, {"optimizer_update": 2}]
    initial = _panel(2.0, (2.0,) * 8)
    final = _panel(1.5, (1.0,) * 8)
    recovery_path = tmp_path / "recovery.pkl"
    recovery_metadata = write_structured_gpp_recovery(
        recovery_path,
        identity=identity,
        specialist=specialist,
        optimizer=optimizer,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=17.0,
        update_records=records,
        initial_panel=initial,
    )
    recovery = load_structured_gpp_recovery(
        recovery_path,
        identity=identity,
        parameter_count=3,
        clip_threshold=17.0,
        maximum_updates=250,
    )
    assert recovery_metadata["completed_updates"] == 2
    np.testing.assert_array_equal(recovery["specialist"]["kernel"], specialist["kernel"])

    checkpoint_path = tmp_path / "stage-update-0250.pkl"
    checkpoint_metadata = write_structured_gpp_checkpoint(
        checkpoint_path,
        identity=identity,
        specialist=specialist,
        optimizer=optimizer,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=17.0,
        update_records=records,
        initial_panel=initial,
        final_panel=final,
    )
    checkpoint = load_structured_gpp_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_metadata["sha256"],
        identity=identity,
        completed_updates=2,
        parameter_count=3,
        clip_threshold=17.0,
    )
    assert checkpoint["promotion_eligible"] is False
    assert checkpoint["final_direct_panel"]["aggregate_direct_gpp_loss"] == 1.5


def test_structured_gpp_warm_start_runner_is_bounded_direct_only_and_unsubmitted():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_structured_gpp_warm_start.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/hpc/run_target_oriented_structured_gpp_warm_start.sh").read_text(encoding="utf-8")
    slurm = (root / "scripts/hpc/slurm_target_oriented_structured_gpp_warm_start_250.sh").read_text(encoding="utf-8")
    assert "_direct_gradient" in runner
    assert "make_trainable_gradient_steps" not in runner
    assert "_horizon_gradient" not in runner
    assert "write_structured_gpp_recovery" in runner
    assert "write_structured_gpp_checkpoint" in runner
    assert '"seven_or_thirty_day_gradients_evaluated": False' in runner
    assert '"retained_parent_parameters_trainable": False' in runner
    assert "stage-update-0250.pkl" in runner
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-70m}" in launcher
    assert "sbatch" not in launcher
    assert "#SBATCH --partition=gnall" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --cpus-per-task=8" in slurm
    assert "#SBATCH --time=01:15:00" in slurm
    assert "CNY 3.45 worst case" in slurm
