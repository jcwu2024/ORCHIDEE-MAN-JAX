import jax
import numpy as np
import pytest

from scripts.hpc.audit_conditioned_attention_update4_derivative import SCALES, summarize_artifact, tree_dot


def test_central_secant_removes_quadratic_term_and_exposes_wrong_gradient():
    record = {
        "artifact_update": 1,
        "historical_baseline": 2.0,
        "historical_full_step": 2.4,
        "evaluations": [
            {"scale": scale, "loss": 2.0 + 0.1 * scale + 0.3 * scale**2, "gradient_dot_actual_delta": -0.2 * scale}
            for scale in SCALES
        ],
    }
    result = summarize_artifact(record)
    assert result["historical_baseline_error"] == 0
    assert result["historical_full_step_error"] == pytest.approx(0)
    for pair in result["pairs"]:
        assert pair["central_secant"] == pytest.approx(0.1)
        assert pair["gradient_dot_actual_delta"] < 0 < pair["forward_change"]


def test_tree_dot_preserves_mixed_precision_cancellation():
    left = {"a": np.asarray([1e8, 1, -1e8], dtype=np.float32), "b": np.asarray([2.0])}
    right = {"a": np.ones(3, dtype=np.float32), "b": np.asarray([-0.25])}
    assert tree_dot(jax, left, right) == 0.5
