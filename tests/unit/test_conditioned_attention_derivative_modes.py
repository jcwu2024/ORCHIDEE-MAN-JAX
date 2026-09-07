import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from scripts.hpc import audit_conditioned_attention_derivative_modes as audit

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("rematerialize", [False, True])
def test_same_scalar_forward_reverse_and_finite_difference_through_scan(rematerialize):
    def score(values, fixed, batch, total):
        def step(state, x):
            following = jnp.tanh(state * fixed + values["weight"] * x)
            return following, following

        _, trace = jax.lax.scan(jax.checkpoint(step) if rematerialize else step, jnp.array(0.2), batch)
        return jnp.sum(trace**2) + total * jnp.sum(trace)

    kernels = audit.make_kernels(jax, score)
    values, direction = {"weight": jnp.array(0.3)}, {"weight": jnp.array(0.7)}
    args = (jnp.array(0.8), jnp.linspace(0.1, 1.0, 30), jnp.array(0.4))
    primal, gradient = kernels["reverse"](values, *args)
    tangent_primal, tangent = kernels["tangent"](values, direction, *args)
    dot = float(gradient["weight"] * direction["weight"])
    assert float(primal) == pytest.approx(float(tangent_primal), rel=1e-12)
    assert float(tangent) == pytest.approx(dot, rel=1e-12)
    eps = 1e-5
    plus = kernels["forward"](audit.window.displaced(jax, values, direction, eps), *args)
    minus = kernels["forward"](audit.window.displaced(jax, values, direction, -eps), *args)
    assert float((plus - minus) / (2 * eps)) == pytest.approx(dot, rel=1e-8)


def test_score_factory_binds_each_mode_without_changing_fixed_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(audit.window, "replace_conditioned_coupled_gpp_trainable", lambda f, v: (f, v))

    def scalar(parameters, batch, total, **kwargs):
        calls.append((parameters, batch, total, kwargs))
        return 1.0

    bootstrap = {"steps": {"window_linearized_score": scalar}}
    scores = [audit.make_score(bootstrap, mode) for mode in (True, False)]
    for score in scores:
        score("trainable", "fixed", "batch", "total")
    assert [c[3]["rematerialize"] for c in calls] == [True, False]
    assert all(c[:3] == (("fixed", "trainable"), "batch", "total") for c in calls)
    assert all(c[3]["horizon"] == 30 for c in calls)


def test_disagreement_summary_does_not_call_opposite_signs_consistent():
    summary = audit.summarize_mode(
        {"reverse_dot_probe": -1098.0, "tangent": 2.63, "reverse_primal": 0.05, "tangent_primal": 0.05}
    )
    assert not summary["same_sign"]
    assert summary["reverse_tangent_relative_disagreement"] > 1


def test_actual_shell_module_branch_supplies_required_parser_options():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text()
    start = shared.index(
        'elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"'
    )
    branch = shared[start : shared.index("elif [[", start + 8)]
    assert '"scripts.hpc.audit_conditioned_attention_derivative_modes" ]]; then' in branch
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"') :]
    required = {o for a in audit.window._parser()._actions if a.required for o in a.option_strings}
    assert all(o in branch or o in command for o in required)
    wrapper = (root / "scripts/hpc/audit_conditioned_attention_derivative_modes.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in wrapper
    assert "sbatch" not in wrapper + shared


def test_complete_runner_orchestration_and_probe_serialization(tmp_path, monkeypatch):
    (tmp_path / "report.json").write_text(json.dumps({"identity": {"git_head": "test"}}))
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
    values = {"weight": jnp.array(0.3)}
    monkeypatch.setattr(
        audit.window, "load_recovery", lambda *a, **kw: {"completed_logical_update": 3, "trainable": values}
    )
    monkeypatch.setattr(audit.window, "_load_parent_state", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(audit.window, "load_coupled_gpp_conditioned_attention_contract", lambda path: None)
    monkeypatch.setattr(audit.window, "replace_conditioned_coupled_gpp_trainable", lambda f, v: v)
    monkeypatch.setattr(audit.window, "_hard_host", lambda v: v)
    monkeypatch.setattr(audit.window, "_hard_failure_count", lambda v: 0)
    monkeypatch.setattr(audit.window, "add_science_window_sufficient_statistics", lambda a, b: a + b)
    artifact = SimpleNamespace(batch=SimpleNamespace(thirty_day=jnp.ones((4, 30))))
    monkeypatch.setattr(audit.window, "_load_artifact", lambda *a: artifact)

    def score(parameters, batch, total, *, horizon, rematerialize):
        def step(carry, x):
            carry = jnp.tanh(carry * 0.8 + parameters["weight"] * x)
            return carry, carry

        _, trace = jax.lax.scan(jax.checkpoint(step) if rematerialize else step, jnp.array(0.1), batch[0])
        return jnp.mean(trace**2) / total

    def bootstrap(*a, **kw):
        assert not args.output.exists()
        return {
            "context": {
                "runtime": {"jax": jax, "jnp": jnp},
                "schedule": {"entries": [{"components": {"thirty_day": [{"point": "toy"}]}}] * 442},
            },
            "fixed_parameters": {},
            "steps": {"thirty_forward": lambda p, b: (jnp.array(1.0), {}), "window_linearized_score": score},
        }

    monkeypatch.setattr(audit.window, "_conditioned_bootstrap", bootstrap)
    result = audit.run(args)
    assert result["status"] == "completed"
    assert [r["rematerialize"] for r in result["modes"]] == [True, False]
    assert all(r["comparison"]["reverse_tangent_relative_disagreement"] < 1e-10 for r in result["modes"])
    with args.output.with_suffix(".probe.pkl").open("rb") as stream:
        probe = pickle.load(stream)
    assert set(probe) == {
        "schema_version",
        "parent_identity",
        "completed_logical_update",
        "artifact_update",
        "window",
        "direction",
        "total",
    }
    assert float(probe["direction"]["weight"]) == pytest.approx(-1.0)
