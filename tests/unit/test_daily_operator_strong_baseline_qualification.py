from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.strong_baseline_qualification import (
    accumulate_exact_gradient_microbatch,
    finalize_exact_gradient_accumulator,
    initialize_exact_gradient_accumulator,
    initialize_strong_baseline_adamw,
    padded_epoch_batches,
    qualification_learning_rate,
    strong_baseline_adamw_update,
    strong_baseline_weight_decay_mask,
)
from scripts.hpc.run_strong_baseline_one_point_qualification import (
    _compare_loss_breakdowns,
    _compare_loss_snapshots,
    _one_epoch_training_smoke_passed,
)


def test_padded_epoch_batches_consume_every_sample_once():
    batches = padded_epoch_batches(10, 4, seed=23, epoch=2)
    admitted = np.concatenate([indices[weights.astype(np.bool_)] for indices, weights in batches])
    assert len(batches) == 3
    assert sorted(admitted.tolist()) == list(range(10))
    assert sum(float(np.sum(weights)) for _indices, weights in batches) == 10.0
    assert np.count_nonzero(batches[-1][1] == 0.0) == 2


def test_qualification_learning_rate_has_declared_warmup_and_cosine_end():
    assert qualification_learning_rate(1, 100) == pytest.approx(5.0e-5)
    assert qualification_learning_rate(2, 100) == pytest.approx(1.0e-4)
    assert qualification_learning_rate(100, 100) == pytest.approx(1.0e-5)


def test_adamw_excludes_biases_and_normalization_scales_from_decay():
    parameters = {
        "layer": {
            "kernel": jnp.ones((2,), dtype=jnp.float32),
            "bias": jnp.ones((2,), dtype=jnp.float32),
        },
        "attention_norm_scale": jnp.ones((2,), dtype=jnp.float32),
        "attention_norm_bias": jnp.ones((2,), dtype=jnp.float32),
    }
    gradients = {
        "layer": {
            "kernel": jnp.zeros((2,), dtype=jnp.float32),
            "bias": jnp.zeros((2,), dtype=jnp.float32),
        },
        "attention_norm_scale": jnp.zeros((2,), dtype=jnp.float32),
        "attention_norm_bias": jnp.zeros((2,), dtype=jnp.float32),
    }
    mask = strong_baseline_weight_decay_mask(parameters)
    state = initialize_strong_baseline_adamw(parameters)
    update = jax.jit(
        lambda values, derivatives, optimizer: strong_baseline_adamw_update(
            values,
            derivatives,
            optimizer,
            1.0e-2,
            mask,
            weight_decay=0.1,
        )
    )
    updated, _state = update(parameters, gradients, state)
    assert np.all(np.asarray(updated["layer"]["kernel"]) < 1.0)
    np.testing.assert_array_equal(np.asarray(updated["layer"]["bias"]), np.ones((2,)))
    np.testing.assert_array_equal(np.asarray(updated["attention_norm_scale"]), np.ones((2,)))
    np.testing.assert_array_equal(np.asarray(updated["attention_norm_bias"]), np.ones((2,)))


def test_exact_gradient_accumulator_uses_real_sample_counts():
    parameters = {"weight": jnp.zeros((2,), dtype=jnp.float32)}
    first = {"weight": jnp.asarray([1.0, 3.0], dtype=jnp.float32)}
    second = {"weight": jnp.asarray([5.0, 7.0], dtype=jnp.float32)}

    @jax.jit
    def accumulate():
        state = initialize_exact_gradient_accumulator(parameters)
        state = accumulate_exact_gradient_microbatch(state, first, 2.0)
        state = accumulate_exact_gradient_microbatch(state, second, 1.0)
        return finalize_exact_gradient_accumulator(state)

    result = accumulate()
    np.testing.assert_allclose(
        np.asarray(result.gradients["weight"]),
        np.asarray([7.0 / 3.0, 13.0 / 3.0]),
    )
    assert float(result.admitted_samples) == 3.0
    assert int(result.microbatch_count) == 2
    assert int(result.nonfinite_gradient_values) == 0
    assert 0.0 < float(result.cancellation_ratio) <= 1.0


def test_loss_floor_breakdown_compares_all_counts_and_stages():
    def snapshot(loss: float, count: int = 3):
        record = {
            "mean_daily_loss": loss,
            "active_day_count": count,
            "defined_value_count": float(count * 2),
        }
        return {
            "experts": {"expert": record},
            "labels": {"label": record},
            "fields": {"field": record},
            "defined_status": {"mask": record},
        }

    result = _compare_loss_breakdowns(snapshot(4.0), snapshot(2.0), snapshot(1.0))
    assert result["counts_identical"]
    for section in ("experts", "labels", "fields", "defined_status"):
        record = next(iter(result[section].values()))
        assert record["reduction_from_initialization"] == pytest.approx(0.75)
        assert record["reduction_during_exact_tail"] == pytest.approx(0.5)


def test_mechanistic_loss_snapshot_compares_exact_diagnostics():
    def snapshot(loss: float, count: int = 3):
        record = {
            "mean_daily_loss": loss,
            "active_day_count": count,
            "defined_value_count": float(count * 2),
        }
        return {
            "experts": {"expert": record},
            "labels": {"label": record},
            "fields": {"field": record},
            "defined_status": {"mask": record},
            "exact_diagnostics": {"exact": record},
        }

    result = _compare_loss_snapshots(snapshot(4.0), snapshot(1.0))
    assert result["counts_identical"]
    for section in ("experts", "labels", "fields", "defined_status", "exact_diagnostics"):
        record = next(iter(result[section].values()))
        assert record["reduction_from_initialization"] == pytest.approx(0.75)


def test_one_epoch_training_smoke_uses_execution_and_absolute_hard_gates():
    initial = {"total": 4.0, "expert_mse": {"expert": 2.0}}
    final = {"total": 3.0, "expert_mse": {"expert": 1.5}}
    science = {
        "primary": {
            "npp": {
                "normalized_rmse": 0.8,
                "normalized_signed_bias": 0.1,
                "normalized_mean_absolute_error": 0.5,
                "evaluated_values": 364,
            }
        },
        "hard_counts": {"water_budget_failures": 0, "carbon_budget_failures": 0},
    }
    common = {
        "failure_reason": None,
        "completed_epochs": 1,
        "completed_updates": 46,
        "expected_updates": 46,
        "all_gradients_finite": True,
        "initial_process": initial,
        "final_process": final,
        "final_science": science,
        "counts_identical": True,
    }
    assert _one_epoch_training_smoke_passed(**common)
    assert not _one_epoch_training_smoke_passed(
        **(common | {"final_science": science | {"hard_counts": {"water_budget_failures": 1}}})
    )
    assert not _one_epoch_training_smoke_passed(
        **(common | {"final_process": final | {"total": 4.5}})
    )


def test_one_point_launcher_has_explicit_mechanistic_training_smoke_mode():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_strong_baseline_one_point_qualification.sh").read_text(
        encoding="utf-8"
    )
    assert "TRAINING_SMOKE=${TRAINING_SMOKE:-0}" in launcher
    assert "MODEL_ARGUMENTS+=(--training-smoke)" in launcher
    assert "MAXIMUM_EPOCHS=1" in launcher
    assert "EVALUATION_INTERVAL=1" in launcher
    assert "sbatch" not in launcher


def test_mechanistic_one_point_slurm_envelope_is_single_gpu_and_bounded():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "slurm_mechanistic_output_v2_one_point.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=gnall" in launcher
    assert "#SBATCH --cpus-per-task=4" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "#SBATCH --time=02:00:00" in launcher
    assert "CNY 4.96" in launcher
    assert "MECHANISTIC_OUTPUT=1" in launcher
    assert "MAXIMUM_EPOCHS=100" in launcher
    assert "TRAINING_SMOKE=0" in launcher
    assert "mechanistic-output-one-epoch-smoke-ba7f5d9" in launcher


def test_loss_floor_launcher_is_free_gln01_and_uses_frozen_tail():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_strong_baseline_one_point_loss_floor_diagnostic.sh").read_text(
        encoding="utf-8"
    )
    assert "one-point loss-floor diagnostic requires free gln01" in launcher
    assert "TAIL_UPDATES=${TAIL_UPDATES:-25}" in launcher
    assert "TAIL_LEARNING_RATE=${TAIL_LEARNING_RATE:-1e-5}" in launcher
    assert '--diagnostic-tail-updates "$TAIL_UPDATES"' in launcher
    assert '--diagnostic-tail-learning-rate "$TAIL_LEARNING_RATE"' in launcher
    assert "--named-biomass-tendency-scale-asset" in launcher
    assert "--diagnostic-state-output" in launcher
    assert "sbatch" not in launcher
