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
from scripts.hpc.run_mechanistic_output_v2_attainability import (
    _attainability_gate_passed,
    _weighted_records,
)
from scripts.hpc.run_mechanistic_output_v2_thermal_source_alignment import (
    _error_summary,
)
from scripts.hpc.run_strong_baseline_one_point_qualification import (
    _compare_loss_breakdowns,
    _compare_loss_snapshots,
    _doc_status_early_screen_review,
    _isolated_field_gate,
    _one_epoch_training_smoke_passed,
    _report_exit_code,
    _report_status,
    _split_gradient_microbatches,
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


def test_gradient_microbatches_preserve_logical_batch_and_skip_zero_padding():
    indices = np.arange(8, dtype=np.int32)
    weights = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    microbatches = _split_gradient_microbatches(indices, weights, 2)

    assert len(microbatches) == 2
    np.testing.assert_array_equal(
        np.concatenate([item[0] for item in microbatches]),
        np.arange(4, dtype=np.int32),
    )
    assert sum(float(np.sum(item[1])) for item in microbatches) == 4.0


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


def test_completed_scientific_rejection_is_not_an_execution_failure():
    assert _report_status(gate_passed=False, execution_status="completed") == "completed_failed_gate"
    assert _report_exit_code({"execution_status": "completed"}) == 0


def test_execution_failure_retains_failed_status_and_nonzero_exit():
    assert _report_status(gate_passed=False, execution_status="failed") == "execution_failed"
    assert _report_exit_code({"execution_status": "failed"}) == 1


def test_attainability_records_preserve_field_local_floors():
    records = _weighted_records(
        ("first", "second"),
        np.asarray([2.0, 9.0]),
        np.asarray([4.0, 3.0]),
    )
    assert records["first"]["normalized_mse"] == pytest.approx(0.5)
    assert records["second"]["normalized_mse"] == pytest.approx(3.0)


def test_thermal_alignment_distinguishes_bitwise_and_physical_tolerance():
    target = np.asarray([1.0, 2.0], dtype=np.float64)
    actual = target.copy()
    actual[1] = np.nextafter(actual[1], np.inf)
    summary = _error_summary(actual, target)
    assert summary["bitwise_unequal_value_count"] == 1
    assert summary["tolerance_failure_count"] == 0
    assert not summary["bitwise_exact"]
    assert summary["within_tolerance"]


def test_isolated_field_gate_cannot_hide_a_dominant_field_floor():
    breakdown = {
        "fields": {
            "passed_by_reduction": {
                "initial_mean_daily_loss": 1.0,
                "final_mean_daily_loss": 0.04,
                "reduction_from_initialization": 0.96,
            },
            "passed_by_floor": {
                "initial_mean_daily_loss": 0.2,
                "final_mean_daily_loss": 9.0e-5,
                "reduction_from_initialization": 0.5,
            },
            "failed": {
                "initial_mean_daily_loss": 0.5,
                "final_mean_daily_loss": 0.1,
                "reduction_from_initialization": 0.8,
            },
        }
    }
    result = _isolated_field_gate(
        breakdown,
        ("passed_by_reduction", "passed_by_floor", "failed"),
    )
    assert result["dominant_field_count"] == 3
    assert result["fields"]["passed_by_reduction"]["passed"]
    assert result["fields"]["passed_by_floor"]["passed"]
    assert not result["fields"]["failed"]["passed"]
    assert not result["passed"]


def test_doc_status_early_screen_uses_frozen_absolute_epoch_5_and_10_gates():
    incoming = "inventory.carbon.DOC.incoming_amount"
    outgoing = "inventory.carbon.DOC.outgoing_fraction"
    adsorbed = "typed.ok_leak.doc_free_to_adsorbed"
    advective = "typed.ok_leak.doc_advective_interface"

    def breakdown(reductions: dict[str, float]) -> dict[str, object]:
        return {
            "counts_identical": True,
            "fields": {
                name: {
                    "initial_mean_daily_loss": 1.0,
                    "final_mean_daily_loss": 1.0 - reduction,
                    "reduction_from_initialization": reduction,
                }
                for name, reduction in reductions.items()
            },
        }

    epoch_5 = _doc_status_early_screen_review(
        epoch=5,
        litter_expert_reduction=0.2,
        loss_breakdown=breakdown(
            {incoming: -0.9, outgoing: 0.1, adsorbed: -0.2, advective: -0.1}
        ),
        incoming_field=incoming,
        outgoing_field=outgoing,
        free_to_adsorbed_field=adsorbed,
        advective_field=advective,
        hard_counts={"failure": 0},
        all_gradients_finite=True,
    )
    assert epoch_5["passed"]
    assert epoch_5["decision"] == "continue_to_epoch_10"

    epoch_10 = _doc_status_early_screen_review(
        epoch=10,
        litter_expert_reduction=0.91,
        loss_breakdown=breakdown(
            {incoming: 0.6, outgoing: 0.5, adsorbed: 0.0, advective: 0.1}
        ),
        incoming_field=incoming,
        outgoing_field=outgoing,
        free_to_adsorbed_field=adsorbed,
        advective_field=advective,
        hard_counts={"failure": 0},
        all_gradients_finite=True,
    )
    assert epoch_10["passed"]
    assert epoch_10["decision"] == "open_formal_100_epoch_gate"

    rejected = _doc_status_early_screen_review(
        epoch=10,
        litter_expert_reduction=0.89,
        loss_breakdown=breakdown(
            {incoming: 0.6, outgoing: 0.49, adsorbed: -0.01, advective: 0.1}
        ),
        incoming_field=incoming,
        outgoing_field=outgoing,
        free_to_adsorbed_field=adsorbed,
        advective_field=advective,
        hard_counts={"failure": 0},
        all_gradients_finite=True,
    )
    assert not rejected["passed"]
    assert rejected["decision"] == "stop_rejected"


def test_attainability_gate_uses_normalized_error_for_derived_exact_fields():
    assert _attainability_gate_passed(
        maximum_field_mse=1.0e-9,
        maximum_exact_mse=1.0e-9,
        maximum_doc_residual=1.0e-11,
        hard_counts={"failure": 0},
    )
    assert not _attainability_gate_passed(
        maximum_field_mse=1.0e-9,
        maximum_exact_mse=1.0e-7,
        maximum_doc_residual=1.0e-11,
        hard_counts={"failure": 0},
    )


def test_attainability_runner_uses_a_valid_epoch_for_deterministic_batches():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts" / "hpc" / "run_mechanistic_output_v2_attainability.py").read_text(
        encoding="utf-8"
    )
    assert "seed=0,\n        epoch=1," in runner


def test_one_point_launcher_has_explicit_mechanistic_training_smoke_mode():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_strong_baseline_one_point_qualification.sh").read_text(
        encoding="utf-8"
    )
    assert "TRAINING_SMOKE=${TRAINING_SMOKE:-0}" in launcher
    assert "MODEL_ARGUMENTS+=(--training-smoke)" in launcher
    assert "MAXIMUM_EPOCHS=1" in launcher
    assert "EVALUATION_INTERVAL=1" in launcher


def test_one_point_launcher_selects_hierarchical_litter_candidate_explicitly():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts" / "hpc" / "run_strong_baseline_one_point_qualification.sh"
    ).read_text(encoding="utf-8")
    runner = (
        root / "scripts" / "hpc" / "run_strong_baseline_one_point_qualification.py"
    ).read_text(encoding="utf-8")
    assert "HIERARCHICAL_COORDINATE_ATTENTION=${HIERARCHICAL_COORDINATE_ATTENTION:-0}" in launcher
    assert "MODEL_ARGUMENTS+=(--hierarchical-coordinate-attention)" in launcher
    assert 'test "$ISOLATED_EXPERT" = litter_poc_doc' in launcher
    assert 'parser.add_argument("--hierarchical-coordinate-attention"' in runner
    assert '"selected_coordinate_blocks"' in runner
    assert '"selected_field_heads"' in runner
    assert "58_736_698" in runner
    assert "HIERARCHICAL_GRADIENT_MICROBATCH_SIZE = 2" in runner
    assert "gradient_accumulation_preserves_logical_batch" in runner
    assert 'snapshot_name = "initial_probe" if update == 1 else "final_probe"' in runner
    assert 'ISOLATED_EXPERT=${ISOLATED_EXPERT:-}' in launcher
    assert 'MODEL_ARGUMENTS+=(--isolated-expert "$ISOLATED_EXPERT")' in launcher
    assert "sbatch" not in launcher


def test_one_point_launcher_selects_coordinate_coupled_doc_candidate_explicitly():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts" / "hpc" / "run_strong_baseline_one_point_qualification.sh"
    ).read_text(encoding="utf-8")
    runner = (
        root / "scripts" / "hpc" / "run_strong_baseline_one_point_qualification.py"
    ).read_text(encoding="utf-8")
    assert (
        "COORDINATE_COUPLED_DOC_TRANSITION=${COORDINATE_COUPLED_DOC_TRANSITION:-0}"
        in launcher
    )
    assert "MODEL_ARGUMENTS+=(--coordinate-coupled-doc-transition)" in launcher
    assert 'test "$HIERARCHICAL_COORDINATE_ATTENTION" = 0' in launcher
    assert 'test "$ISOLATED_EXPERT" = litter_poc_doc' in launcher
    assert 'parser.add_argument("--coordinate-coupled-doc-transition"' in runner
    assert "59_971_257" in runner
    assert "coordinate_coupled_doc_transition_v4_one_epoch_smoke_v1" in runner
    assert "sbatch" not in launcher


def test_one_point_launcher_selects_doc_status_branch_magnitude_v5_explicitly():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts" / "hpc" / "run_strong_baseline_one_point_qualification.sh"
    ).read_text(encoding="utf-8")
    runner = (
        root / "scripts" / "hpc" / "run_strong_baseline_one_point_qualification.py"
    ).read_text(encoding="utf-8")
    assert "DOC_STATUS_BRANCH_MAGNITUDE=${DOC_STATUS_BRANCH_MAGNITUDE:-0}" in launcher
    assert "MODEL_ARGUMENTS+=(--doc-status-branch-magnitude)" in launcher
    assert 'test "$HIERARCHICAL_COORDINATE_ATTENTION" = 0' in launcher
    assert 'test "$COORDINATE_COUPLED_DOC_TRANSITION" = 0' in launcher
    assert 'test "$ISOLATED_EXPERT" = litter_poc_doc' in launcher
    assert 'parser.add_argument("--doc-status-branch-magnitude"' in runner
    assert "59_973_309" in runner
    assert "doc_status_branch_magnitude_attention_v5_early_screen_v1" in runner
    assert "masked_doc_status_branch_magnitude_loss" in runner
    assert "sbatch" not in launcher


def test_doc_status_v5_early_screen_slurm_envelope_is_bounded_and_no_checkpoint():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts" / "hpc" / "slurm_doc_status_branch_magnitude_v5_early_screen.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gnall" in launcher
    assert "#SBATCH --cpus-per-task=4" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "#SBATCH --time=02:00:00" in launcher
    assert "CNY 4.96" in launcher
    assert "DOC_STATUS_BRANCH_MAGNITUDE=1" in launcher
    assert "MAXIMUM_EPOCHS=10" in launcher
    assert "EVALUATION_INTERVAL=5" in launcher
    assert "CHECKPOINT" not in launcher


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


def test_mechanistic_attainability_launcher_is_free_zero_training_gpu_audit():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_mechanistic_output_v2_attainability.sh").read_text(
        encoding="utf-8"
    )
    assert "run_mechanistic_output_v2_attainability" in launcher
    assert "--doc-fixed-point-iterations 8" in launcher
    assert "--batch-size 8" in launcher
    assert "sbatch" not in launcher


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
