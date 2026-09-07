import json
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
    assert '"scripts.hpc.audit_conditioned_attention_window442"' in branch
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"') :]
    required = {option for action in audit._parser()._actions if action.required for option in action.option_strings}
    assert all(option in branch or option in command for option in required)
    wrapper = (root / "scripts/hpc/audit_conditioned_attention_window442.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in wrapper
    assert "sbatch" not in wrapper + shared


def test_report_is_not_created_before_inherited_bootstrap(tmp_path, monkeypatch):
    (tmp_path / "report.json").write_text(json.dumps({"identity": {}}))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"contract_sha256": "test"}))
    args = SimpleNamespace(
        conditioned_run_dir=tmp_path,
        output=tmp_path / "diagnostic.json",
        parent_persistent_training_report=None,
        parent_artifact_aware_recovery=None,
        artifact_aware_persistent_contract=contract,
        conditioning_contract=None,
    )
    monkeypatch.setattr(audit, "load_recovery", lambda *a, **kw: {"completed_logical_update": 3})
    monkeypatch.setattr(audit, "_load_parent_state", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(audit, "load_coupled_gpp_conditioned_attention_contract", lambda path: None)

    def bootstrap(*a, **kw):
        assert not args.output.exists()
        raise RuntimeError("bootstrap boundary reached")

    monkeypatch.setattr(audit, "_conditioned_bootstrap", bootstrap)
    with pytest.raises(RuntimeError, match="bootstrap boundary reached"):
        audit.run(args)
    assert json.loads(args.output.read_text())["status"] == "failed"


def test_masked_comparison_excludes_only_declared_undefined_coordinates():
    row = audit.derivative_comparison([2.0, 0.0], [3.0, np.nan], [1.0, np.nan], 0.5, [1.0, 1.0], defined=[True, False])
    assert row["excluded_coordinate_count"] == 1
    assert row["nonfinite_count"] == 0
    assert row["error_scaled_l2"] == 0
    assert row["worst_flat_index"] == 0
    bad = audit.derivative_comparison([2.0, 0.0], [np.nan, 0.0], [1.0, 0.0], 0.5, [1.0, 1.0], defined=[True, False])
    assert bad["nonfinite_count"] == 1
