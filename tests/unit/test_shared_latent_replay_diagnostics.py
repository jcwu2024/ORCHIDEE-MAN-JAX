"""Pure NumPy failure evidence; no JAX initialization, model or GPU execution."""
import json

import numpy as np
import pytest

from scripts.hpc.shared_latent_replay_diagnostics import summarize_state_replay, state_replay_execution_metadata


@pytest.mark.parametrize("architecture,identity", [
    ("field_original", {"source_git_head": "original", "method": {"architecture": "original"}}),
    ("shared_latent32x384_reader64x2", {"source_head": "candidate", "candidate_config": {"latents": 32}}),
])
def test_execution_metadata_retains_the_explicit_original_or_candidate_identity(architecture, identity):
    metadata = state_replay_execution_metadata(execution_architecture=architecture, execution_identity=identity)
    assert metadata["identity"] == identity
    assert metadata["identity"] is not identity
    assert metadata["execution_architecture"] == architecture
    assert metadata["execution_scope"] == "same_architecture_full_initial_state_AD_replay"


def test_near_zero_cancellation_keeps_failure_and_exact_coordinate_context():
    reference = np.array([[1e7, 1e-9, 3.], [7., -2e-9, 6.]], dtype=np.float64)
    actual = reference.copy()
    actual[0, 0] += .08  # Large absolute change remains inside relative tolerance.
    actual[1, 1] += .08  # Near-zero reference from cancellation genuinely fails.
    initial = np.arange(6, dtype=np.float64).reshape(2, 3) + 10
    summary = summarize_state_replay(actual, reference, input_state=initial,
        state_scales={"TRAIN_RMS": np.array([2., 4., 8.]), "floor": .01},
        loss_scales={"daily_weight": 1/3, "flux": {"gpp": 2.8}})
    assert summary["passed"] is False and summary["failed_coordinates"] == 1
    row = summary["failed_rows"][0]
    assert row["index"] == [1, 1] and row["flat_index"] == 4
    assert row["sample_row"] == 1 and row["state_coordinate"] == 1
    assert row["actual"] == actual[1, 1] and row["reference"] == reference[1, 1]
    assert row["input_state"] == 14
    assert row["state_scales"] == {"TRAIN_RMS": 4., "floor": .01}
    assert row["effective_tolerance"] == 1e-6 + 1e-5 * abs(reference[1, 1])
    assert row["tolerance_excess"] == abs(actual[1, 1] - reference[1, 1]) - row["effective_tolerance"]
    assert summary["loss_scale_context"] == {"daily_weight": 1/3, "flux": {"gpp": 2.8}}
    json.dumps(summary, allow_nan=False)


def test_large_relative_error_at_tiny_reference_can_pass_absolute_tolerance():
    reference = np.array([1e-12, 10000.], dtype=np.float64)
    actual = np.array([5e-7, 10000.09], dtype=np.float64)
    summary = summarize_state_replay(actual, reference, input_state=np.ones(2))
    assert summary["passed"] is True
    assert summary["failed_coordinates"] == 0
    assert summary["cause_attribution"] is None


def test_nonfinite_gradients_always_fail_even_matching_infinities_and_json_is_strict():
    reference = np.array([np.inf, np.nan, 0.])
    actual = np.array([np.inf, 0., np.nan])
    summary = summarize_state_replay(actual, reference, input_state=np.ones(3), max_failed_rows=2)
    assert summary["failed_coordinates"] == 3
    assert summary["actual_nonfinite_coordinates"] == 2
    assert summary["reference_nonfinite_coordinates"] == 2
    assert summary["failed_rows_omitted"] == 1
    assert summary["failed_rows"][0]["actual"] == "+Infinity"
    assert summary["failed_rows"][0]["tolerance_excess"] is None
    assert all(row["reason"] == "nonfinite_gradient" for row in summary["failed_rows"])
    json.dumps(summary, allow_nan=False)


def test_bounded_failed_rows_preserve_global_counts_and_full_shape_scale_mapping():
    reference = np.zeros((4, 3854), dtype=np.float64)
    actual = reference.copy()
    actual[0, 19] = .01
    actual[3, 200] = -.08
    scales = np.arange(reference.size, dtype=np.float64).reshape(reference.shape)
    summary = summarize_state_replay(actual, reference, input_state=np.ones_like(reference),
        state_scales=scales, max_failed_rows=1)
    assert summary["coordinates"] == 15416 and summary["failed_coordinates"] == 2
    assert summary["failed_rows_omitted"] == 1
    assert summary["failed_rows"][0]["state_scales"] == 19.
    assert summary["state_scale_context"]["shape"] == [4, 3854]
    assert summary["maximum_absolute_finite_error"] == .08


@pytest.mark.parametrize("which", ["actual", "reference", "input_state", "scale"])
def test_rejects_shape_mismatch_without_reinterpreting_indices(which):
    values = {"actual": np.ones((4, 3)), "reference": np.ones((4, 3)), "input_state": np.ones((4, 3))}
    scale = None
    if which == "scale": scale = np.ones((2, 3))
    else: values[which] = np.ones((2, 3))
    with pytest.raises(ValueError, match="shape"):
        summarize_state_replay(values["actual"], values["reference"], input_state=values["input_state"], state_scales=scale)
