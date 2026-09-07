import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import audit_conditioned_attention_streamed_adjoint as audit

jax.config.update("jax_enable_x64", True)


class ToyInputs(NamedTuple):
    continuous_state: object
    discrete_state: object
    forcing: object


class ToyMicro(NamedTuple):
    inputs: object
    day_inputs: object


class ToyGroup(NamedTuple):
    other: object


class ToyPrediction(NamedTuple):
    named_outputs: object


class ToyResult(NamedTuple):
    next_state: object
    next_state_defined: object
    prediction: object
    next_discrete_state: object


class ToyTrace(NamedTuple):
    result: object
    pft_fraction: object
    active_pft_mask: object


def toy_step(values, state, fixed, day):
    following = jnp.tanh(fixed * state + values["weight"] * day + values["bias"])
    return {"state": following, "gpp": jnp.sum(following**2)[None], "fraction": jax.nn.sigmoid(state[:1])}


def toy_score(obs):
    return jnp.sum(obs["gpp"]) ** 2 / 900 + jnp.sum(obs["state"][-1] ** 2) + jnp.mean(obs["gpp"] * obs["fraction"]) ** 2


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_streamed_full_gradient_matches_scan_without_truncation(dtype):
    values = {"weight": jnp.array([0.2, -0.1], dtype=dtype), "bias": jnp.array(0.03, dtype=dtype)}
    direction = jax.tree_util.tree_map(lambda v: jnp.ones_like(v) * 0.2, values)
    initial, fixed = jnp.array([0.1, -0.2]), jnp.array([0.7, 0.6])
    days = jnp.linspace(0.3, 1.0, 30)

    def rollout(parameters):
        def step(state, day):
            obs = toy_step(parameters, state, fixed, day)
            return obs["state"], obs

        return jax.lax.scan(step, initial, days)[1]

    obs = rollout(values)
    starts = [initial] + list(obs["state"][:-1])
    cotangents = jax.grad(toy_score)(obs)
    cotangent_days = [jax.tree_util.tree_map(lambda v: v[i], cotangents) for i in range(30)]
    kernels = audit.make_day_kernels(jax, toy_step)
    streamed = audit.streamed_derivatives(
        jax, jnp, kernels, values, direction, fixed, starts, list(days), cotangent_days
    )
    full = jax.grad(lambda v: toy_score(rollout(v)))(values)
    for a, b in zip(jax.tree_util.tree_leaves(streamed["gradient"]), jax.tree_util.tree_leaves(full), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-9)
    assert streamed["reverse_projection"] == pytest.approx(streamed["forward_projection"], rel=2e-6)
    assert [r["day"] for r in streamed["records"]] == list(range(30, 0, -1))
    assert max(abs(r["duality_gap"]) for r in streamed["records"]) < 1e-7
    assert abs(streamed["records"][-1]["parameter_projection"]) > 1e-4
    epsilon = 1e-4
    plus = toy_score(rollout(audit.window.displaced(jax, values, direction, epsilon)))
    minus = toy_score(rollout(audit.window.displaced(jax, values, direction, -epsilon)))
    assert float((plus - minus) / (2 * epsilon)) == pytest.approx(streamed["reverse_projection"], rel=2e-4)


def test_memory_requirement_rejects_oversized_graph_before_execution():
    executable = SimpleNamespace(
        memory_analysis=lambda: SimpleNamespace(
            argument_size_in_bytes=3, output_size_in_bytes=5, temp_size_in_bytes=100, alias_size_in_bytes=1
        )
    )
    assert audit.memory_requirement(executable, 107)["estimated_peak_bytes"] == 107
    with pytest.raises(RuntimeError, match="exceeds"):
        audit.memory_requirement(executable, 106)


def test_zero_cotangent_handles_undefined_constant_but_not_active_nan():
    assert audit.dot(jax, {"x": np.array([2.0, 0.0])}, {"x": np.array([3.0, np.nan])}) == 6.0
    assert np.isnan(audit.dot(jax, {"x": np.array([2.0, 1.0])}, {"x": np.array([3.0, np.nan])}))


def test_replay_difference_separates_absolute_scaled_and_downstream_errors():
    result = audit.replay_difference(
        np.array([1e8, 1.0, np.nan, 2.0]), np.array([1e8 + 5, 1.01, np.nan, np.nan]),
        np.array([1e8, 0.01, 1.0, 1.0]), np.array([0.0, 3.0, 0.0, 2.0]), lambda i: {"index": i},
    )
    assert result["top_absolute"][0]["index"] == 0
    assert result["top_scaled"][0]["index"] == 1
    assert result["top_downstream"][0]["index"] == 1
    assert result["linearized_downstream_effect"] == pytest.approx(0.03)
    assert result["finite_mask_changes"] == 1
    assert result["nonfinite_active_coordinates"] == 1


def test_actual_launcher_branch_supplies_required_options():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text()
    start = shared.index(
        'elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"'
    )
    branch = shared[start : shared.index("elif [[", start + 8)]
    assert '"scripts.hpc.audit_conditioned_attention_streamed_adjoint" ]]; then' in branch
    assert 'EXTRA_AUDIT_ARGS+=(--replay-trace "$REPLAY_TRACE")' in branch
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"') :]
    required = {o for a in audit._parser()._actions if a.required for o in a.option_strings}
    assert all(o in branch or o in command for o in required)
    wrapper = (root / "scripts/hpc/audit_conditioned_attention_streamed_adjoint.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in wrapper
    assert "sbatch" not in wrapper + shared


def test_full_runner_bootstrap_replay_and_diagnostic_output(tmp_path, monkeypatch):
    identity = {"git_head": "test"}
    (tmp_path / "report.json").write_text(json.dumps({"identity": identity}))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"contract_sha256": "test"}))
    args = SimpleNamespace(
        conditioned_run_dir=tmp_path,
        output=tmp_path / "diagnostic.json",
        probe=tmp_path / "probe.pkl",
        parent_persistent_training_report=None,
        parent_artifact_aware_recovery=None,
        artifact_aware_persistent_contract=contract,
        conditioning_contract=None,
    )
    values = {"weight": jnp.array([0.2, -0.1]), "bias": jnp.array(0.03)}
    direction = jax.tree_util.tree_map(lambda v: jnp.ones_like(v) * 0.2, values)
    with args.probe.open("wb") as stream:
        pickle.dump(
            {
                "parent_identity": identity,
                "completed_logical_update": 3,
                "artifact_update": 442,
                "window": 0,
                "direction": jax.device_get(direction),
                "total": np.array(2.0),
            },
            stream,
        )
    fixed = jnp.array([0.7, 0.6])
    initial = jnp.array([0.1, -0.2])
    batch = ToyMicro(
        ToyInputs(
            jnp.broadcast_to(initial, (4, 30, 2)),
            {"flag": jnp.zeros((4, 30), dtype=bool)},
            jnp.broadcast_to(jnp.linspace(0.3, 1.0, 30), (4, 30)),
        ),
        jnp.zeros((4, 30)),
    )
    artifact = SimpleNamespace(batch=SimpleNamespace(thirty_day=batch))

    def evaluate(parameters, micro):
        def step(state, day):
            obs = toy_step(parameters, state, fixed, day)
            return obs["state"], obs

        obs = jax.lax.scan(step, initial, micro.inputs.forcing[0])[1]
        trace = ToyTrace(
            ToyResult(
                ToyGroup(obs["state"]),
                ToyGroup(jnp.ones_like(obs["state"], dtype=bool)),
                ToyPrediction({"daily.fast_gpp": obs["gpp"]}),
                {"flag": jnp.zeros(30, dtype=bool)},
            ),
            obs["fraction"],
            jnp.ones((30, 1), dtype=bool),
        )
        return trace, None

    def score_factory(bootstrap, micro):
        def score(obs, defined, active, total):
            return toy_score(obs) / total

        return jax.jit(score), jax.jit(jax.value_and_grad(score))

    def bootstrap(*a, **kw):
        assert not args.output.exists()
        return {
            "context": {
                "runtime": {"jax": jax, "jnp": jnp, "contract": None},
                "scales": SimpleNamespace(state=SimpleNamespace(other=np.ones(2))),
            },
            "fixed_parameters": fixed,
            "flux_basis": None,
            "steps": {"thirty_forward": lambda p, b: (toy_score(audit.observations(evaluate(p, b)[0])) / 2, {})},
        }

    monkeypatch.setattr(
        audit.window, "load_recovery", lambda *a, **kw: {"completed_logical_update": 3, "trainable": values}
    )
    monkeypatch.setattr(audit.window, "_load_parent_state", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(audit.window, "load_coupled_gpp_conditioned_attention_contract", lambda path: None)
    monkeypatch.setattr(audit.window, "_conditioned_bootstrap", bootstrap)
    monkeypatch.setattr(audit.window, "_load_artifact", lambda *a: artifact)
    monkeypatch.setattr(audit.window, "replace_conditioned_coupled_gpp_trainable", lambda f, v: v)
    monkeypatch.setattr(audit.window, "_make_trace_evaluator", lambda b: evaluate)
    monkeypatch.setattr(audit.window, "_hard_host", lambda h: h)
    monkeypatch.setattr(audit.window, "_hard_failure_count", lambda h: 0)
    monkeypatch.setattr(audit.window, "linearized_science_window_loss", lambda stats, total, **kw: stats)
    monkeypatch.setattr(audit.window, "_state_location", lambda c, i: {"index": i})
    monkeypatch.setattr(audit, "make_observation_score", score_factory)
    monkeypatch.setattr(audit, "make_step", lambda b: lambda v, s, f, day: toy_step(v, s, f, day[0].forcing))
    result = audit.run(args)
    assert result["status"] == "completed"
    assert len(result["reverse_days"]) == 30
    assert result["forward_projection"] == pytest.approx(result["reverse_projection"], rel=1e-12)
    assert result["baseline_score"] == pytest.approx(result["production_score"], rel=1e-12)
    assert args.output.with_suffix(".trace.pkl").exists()
    args.replay_trace = args.output.with_suffix(".trace.pkl")
    args.output = tmp_path / "replay.json"

    def forbidden(*a, **kw):
        raise AssertionError("saved replay must not load artifacts or run a full trajectory")

    monkeypatch.setattr(audit.window, "_load_artifact", forbidden)
    monkeypatch.setattr(audit.window, "_make_trace_evaluator", forbidden)
    replay = audit.run(args)
    assert replay["status"] == "completed"
    assert [r["day"] for r in replay["replay_days"]] == [2, 15, 16]
    assert max(abs(r["duality_gap"]) for r in replay["replay_days"]) < 1e-10
    for row in replay["replay_days"]:
        assert row["stored_adjoint_maximum_error"] < 1e-10
        assert row["comparisons"]["jvp_to_vjp"]["state"]["maximum_absolute_error"] < 1e-12
    assert not args.output.with_suffix(".trace.pkl").exists()
