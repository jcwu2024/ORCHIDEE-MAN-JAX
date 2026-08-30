from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_training import (
    CHECKPOINT_UPDATES,
    FROZEN_CONTRACT_SHA256,
    MONITOR_UPDATES,
    load_structured_gpp_broad_training_contract,
)
from scripts.hpc.run_target_oriented_structured_gpp_broad_training import (
    _is_admission_only,
    _make_gradient_check,
    _monitor_comparison,
)


def test_structured_gpp_broad_training_contract_freezes_stage_one():
    contract = load_structured_gpp_broad_training_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.payload["data"]["optimizer_updates_maximum"] == 1000
    assert contract.payload["data"]["unique_training_days"] == 208_000
    assert contract.payload["optimizer"]["learning_rate"] == 4e-6
    assert contract.clip_threshold == 20.98786557796449
    assert MONITOR_UPDATES == (0, 250, 500, 1000)
    assert CHECKPOINT_UPDATES == (250, 500, 1000)
    assert contract.payload["execution"]["paid_training_authorized_by_contract_alone"] is False


def _monitor(loss: float, bias: float, improved: int | None = None):
    records = []
    for index in range(128):
        records.append(
            {
                "landpoint_id": f"p{index}",
                "year": 1961,
                "parent_shard_sha256": f"parent-{index}",
                "typed_shard_sha256": f"typed-{index}",
                "loss": (
                    loss
                    if improved is None
                    else loss - 0.1
                    if index < improved
                    else loss + 0.1
                ),
            }
        )
    return {
        "aggregate": {"loss": loss, "normalized_bias": bias},
        "records": records,
    }


def test_structured_gpp_broad_monitor_comparison_and_admission_boundary():
    initial = _monitor(1.0, -0.2)
    current = _monitor(0.9, -0.1, improved=96)
    comparison = _monitor_comparison(initial, current)
    assert comparison["loss_reduction"] == pytest.approx(0.1)
    assert comparison["improved_point_years"] == 96
    assert _is_admission_only(2)
    assert not _is_admission_only(250)


def test_structured_gpp_broad_gradient_check_reports_finite_nonzero_leaves():
    check = _make_gradient_check(jax, jnp)
    nonfinite, nonzero = check(
        {
            "active": jnp.asarray([1.0, -2.0]),
            "zero": jnp.asarray([0.0]),
        }
    )
    assert int(nonfinite) == 0
    assert int(nonzero) == 1


def test_structured_gpp_broad_launchers_keep_free_and_paid_boundaries_separate():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_structured_gpp_broad_training.py").read_text(
        encoding="utf-8"
    )
    worker = (root / "scripts/hpc/run_target_oriented_structured_gpp_broad_training.sh").read_text(
        encoding="utf-8"
    )
    admission = (
        root / "scripts/hpc/run_target_oriented_structured_gpp_broad_training_admission.sh"
    ).read_text(encoding="utf-8")
    paid = (root / "scripts/hpc/slurm_target_oriented_structured_gpp_broad_training.sh").read_text(
        encoding="utf-8"
    )
    assert "assemble_direct_gpp" not in runner
    assert "_prepare_batch" in runner
    assert 'EXECUTED_UPDATES=${MAXIMUM_UPDATES:-1000}' in worker
    assert '--maximum-updates "$EXECUTED_UPDATES"' in worker
    assert "EXTRA_ARGS" not in worker
    assert "MAXIMUM_UPDATES=2" in admission
    assert "sbatch" not in admission
    assert "#SBATCH --partition=gnall" in paid
    assert "#SBATCH --gres=gpu:1" in paid
    assert "#SBATCH --time=05:00:00" in paid
