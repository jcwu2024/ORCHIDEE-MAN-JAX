from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import audit_conditioned_attention_window442 as audit

jax.config.update("jax_enable_x64", True)


def test_comparison_finds_coordinate_without_quadratic_contamination():
    slope = np.array([2.0, 7.0])
    epsilon = 1e-3
    plus = 3 + epsilon * slope + epsilon**2 * 9
    minus = 3 - epsilon * slope + epsilon**2 * 9
    row = audit.derivative_comparison([2.0, -7.0], plus, minus, epsilon, [1.0, 2.0])
    assert row["worst_flat_index"] == 1
    assert row["worst_secant"] == pytest.approx(7)
    assert row["worst_scaled_error"] == pytest.approx(-7)
    scalar = audit.derivative_comparison(2.0, 1.002, 0.998, 0.001, 1.0)
    assert scalar["error_scaled_l2"] < 1e-10


def test_displacement_preserves_tree_dtypes_and_actual_rounding():
    values = {"single": jnp.array([1.0], dtype=jnp.float32), "double": jnp.array([1.0], dtype=jnp.float64)}
    direction = jax.tree_util.tree_map(jnp.ones_like, values)
    result = audit.displaced(jax, values, direction, 1e-9)
    assert result["single"].dtype == jnp.float32
    assert result["double"].dtype == jnp.float64
    assert float(result["single"][0]) == 1.0
    assert float(result["double"][0]) > 1.0


def test_day_selection_includes_first_disagreement_predecessor_and_peak():
    rows = [{"relative_error": 0.0, "error_scaled_l2": 0.0} for _ in range(30)]
    rows[3] = {"relative_error": 0.5, "error_scaled_l2": 0.1}
    rows[19] = {"relative_error": 0.7, "error_scaled_l2": 8.0}
    assert audit.select_local_days(rows) == [2, 3, 19]


def test_compiled_local_probe_includes_parameter_and_carry_tangents(monkeypatch):
    from research.daily_coarse_graining.daily_operator import successor_training

    monkeypatch.setattr(
        successor_training, "rollout_input_from_template", lambda template, state, discrete, contract: state
    )
    monkeypatch.setattr(audit, "replace_conditioned_coupled_gpp_trainable", lambda fixed, values: values)

    def dynamic(values, state, day_inputs):
        return SimpleNamespace(
            next_state=SimpleNamespace(other=state * values["weight"]),
            prediction=SimpleNamespace(named_outputs={"flux": state + values["weight"], "mask": jnp.array(True)}),
        )

    bootstrap = {
        "context": {"runtime": {"jax": jax, "jnp": jnp, "contract": None}, "template": None},
        "steps": {"transition_builder": lambda template: dynamic},
    }
    forward, tangent = audit.make_local_payload(bootstrap)
    values, direction = {"weight": jnp.array(2.0)}, {"weight": jnp.array(3.0)}
    state, state_direction = jnp.array([5.0]), jnp.array([7.0])
    base, derivative = tangent(values, state, direction, state_direction, {}, None, {}, None)
    assert float(base["state"][0]) == 10.0
    assert float(derivative["state"][0]) == 29.0
    assert float(derivative["process"]["flux"][0]) == 10.0
    assert "mask" not in derivative["process"]
    epsilon = 1e-4
    plus = forward(
        audit.displaced(jax, values, direction, epsilon), state + epsilon * state_direction, {}, None, {}, None
    )
    minus = forward(
        audit.displaced(jax, values, direction, -epsilon), state - epsilon * state_direction, {}, None, {}, None
    )
    assert (
        audit.derivative_comparison(derivative["state"], plus["state"], minus["state"], epsilon, 1)["error_scaled_l2"]
        < 1e-9
    )


def test_launcher_uses_the_actual_required_parser_branch():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text()
    start = shared.index(
        'elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"'
    )
    branch = shared[start : shared.index("elif [[", start + 8)]
    assert '"scripts.hpc.audit_conditioned_attention_window442" ]]; then' in branch
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"') :]
    required = {option for action in audit._parser()._actions if action.required for option in action.option_strings}
    assert all(option in branch or option in command for option in required)
    wrapper = (root / "scripts/hpc/audit_conditioned_attention_window442.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in wrapper
    assert "sbatch" not in wrapper + shared
