import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.streamed_adjoint import StreamedAdjoint
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    COMPONENT_RESOLVED_FLUX_BASIS,
    INDEPENDENT_STOCK_BASIS,
    ScienceWindowSufficientStatistics,
    linearized_science_window_loss,
    normalized_error_sufficient_statistics,
    science_window_from_sufficient_statistics,
)
from scripts.hpc import audit_conditioned_attention_streamed_step as audit

jax.config.update("jax_enable_x64", True)


class Inputs(NamedTuple):
    continuous_state: object
    discrete_state: object
    forcing: object


class Micro(NamedTuple):
    inputs: object
    day_inputs: object


def step(values, state, fixed, day):
    following = jnp.tanh(fixed * state + values["weight"] * day + values["bias"])
    return {"state": following, "gpp": following**2, "fraction": jax.nn.sigmoid(state)}


def statistics(obs, index):
    flux = jnp.mean(obs["gpp"] * obs["fraction"], axis=0)
    stock = obs["state"][-1]

    def group(vector, names):
        return {
            name: normalized_error_sufficient_statistics(
                vector * (k + 1), jnp.full_like(vector, 0.01 * (index + 1)),
                jnp.array([True, (index + k) % 2 == 0]), 1.0,
            ) for k, name in enumerate(names)
        }

    return ScienceWindowSufficientStatistics(group(flux, COMPONENT_RESOLVED_FLUX_BASIS), group(stock, INDEPENDENT_STOCK_BASIS))


def setup_case(dtype, horizon):
    values = {"weight": jnp.array([0.2, -0.1], dtype=dtype), "bias": jnp.array(0.03, dtype=dtype)}
    fixed = jnp.array([0.7, 0.6])
    initial = [jnp.array([0.1, -0.2]) * (i + 1) for i in range(4)]
    days = [jnp.linspace(0.3 + i * 0.03, 1.0, horizon) for i in range(4)]

    def rollout(v, index):
        def transition(s, d):
            obs = step(v, s, fixed, d)
            return obs["state"], obs
        return jax.lax.scan(transition, initial[index], days[index])[1]

    def full_loss(v):
        total = audit.merge_statistics([statistics(rollout(v, i), i) for i in range(4)])
        return science_window_from_sufficient_statistics(total).total

    return values, fixed, initial, days, rollout, full_loss


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("horizon", [7, 30])
def test_full_gradient_matches_four_window_global_bias_and_reuses_compilation(dtype, horizon):
    values, fixed, initial, days, rollout, full_loss = setup_case(dtype, horizon)
    observations = [rollout(values, i) for i in range(4)]
    total = audit.merge_statistics([statistics(obs, i) for i, obs in enumerate(observations)])
    engine = StreamedAdjoint(jax, jnp, step)
    accumulated = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p, dtype=jnp.float64), values)
    first_day_gradients = []
    for i, obs in enumerate(observations):
        cotangents = jax.grad(lambda o: linearized_science_window_loss(statistics(o, i), total, flux_basis=COMPONENT_RESOLVED_FLUX_BASIS))(obs)
        states = [initial[i]] + list(obs["state"][:-1])
        cotangent_days = [jax.tree_util.tree_map(lambda v: v[k], cotangents) for k in range(horizon)]
        gradient = engine.gradient(values, fixed, states, list(days[i]), cotangent_days)
        accumulated = engine.add(accumulated, gradient)
        first_day_gradients.append(engine.gradient(values, fixed, states[:1], list(days[i])[:1], cotangent_days[:1]))
    expected = jax.grad(full_loss)(values)
    for a, b in zip(jax.tree_util.tree_leaves(accumulated), jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-9)
    assert engine.compile_count == 1
    assert audit.streamed.dot(jax, first_day_gradients[0], first_day_gradients[0]) > 1e-12
    norm = np.sqrt(audit.streamed.dot(jax, accumulated, accumulated))
    direction = jax.tree_util.tree_map(lambda g, v: (-g / norm).astype(v.dtype), accumulated, values)
    plus = audit.window.displaced(jax, values, direction, 1e-4)
    minus = audit.window.displaced(jax, values, direction, -1e-4)
    actual_direction = jax.tree_util.tree_map(lambda a, b: (a.astype(jnp.float64) - b.astype(jnp.float64)) / 2e-4, plus, minus)
    assert float((full_loss(plus) - full_loss(minus)) / 2e-4) == pytest.approx(audit.streamed.dot(jax, accumulated, actual_direction), rel=2e-4)
    assert float(full_loss(plus)) < float(full_loss(values))


def test_rejects_oversize_before_reverse_and_propagates_active_nonfinite():
    values, fixed, initial, days, rollout, _ = setup_case(jnp.float64, 7)
    obs = rollout(values, 0)
    states = [initial[0]] + list(obs["state"][:-1])
    cotangents = [{"state": jnp.ones(2), "gpp": jnp.zeros(2), "fraction": jnp.zeros(2)} for _ in states]
    with pytest.raises(RuntimeError, match="memory envelope"):
        StreamedAdjoint(jax, jnp, step, limit_bytes=1).gradient(values, fixed, states, list(days[0]), cotangents)
    cotangents[-1]["state"] = jnp.full(2, jnp.nan)
    with pytest.raises(RuntimeError, match="nonfinite"):
        StreamedAdjoint(jax, jnp, step).gradient(values, fixed, states, list(days[0]), cotangents)


def test_real_runner_four_windows_true_loss_and_no_checkpoint(tmp_path, monkeypatch):
    values, fixed, initial, days, rollout, full_loss = setup_case(jnp.float32, 30)
    args = SimpleNamespace(
        conditioned_run_dir=tmp_path, output=tmp_path / "result.json", artifact_aware_persistent_contract=tmp_path / "contract.json",
        parent_persistent_training_report=None, parent_artifact_aware_recovery=None, conditioning_contract=None,
    )
    (tmp_path / "report.json").write_text(json.dumps({"identity": {"test": True}}))
    args.artifact_aware_persistent_contract.write_text(json.dumps({"contract_sha256": "test"}))
    micro = Micro(Inputs(
        jnp.stack([jnp.broadcast_to(v, (30, 2)) for v in initial]),
        {"index": jnp.stack([jnp.full(30, i, dtype=jnp.int32) for i in range(4)])}, jnp.stack(days),
    ), jnp.zeros((4, 30)))
    artifact = SimpleNamespace(batch=SimpleNamespace(thirty_day=micro))

    def index(m):
        return int(m.inputs.discrete_state["index"][0, 0])

    def evaluator(v, m):
        i = index(m)
        obs = rollout(v, i)
        return SimpleNamespace(
            result=SimpleNamespace(
                next_state=SimpleNamespace(other=obs["state"]),
                next_state_defined=SimpleNamespace(other=jnp.ones_like(obs["state"], dtype=bool)),
                next_discrete_state={"index": jnp.full(30, i, dtype=jnp.int32)},
                prediction=SimpleNamespace(named_outputs={"daily.fast_gpp": obs["gpp"]}),
            ), pft_fraction=obs["fraction"], active_pft_mask=jnp.ones((30, 2), dtype=bool),
        ), None

    def observation_score(b, m):
        i = index(m)
        def score(obs, defined, active, total):
            return linearized_science_window_loss(statistics(obs, i), total, flux_basis=COMPONENT_RESOLVED_FLUX_BASIS)
        return score, jax.value_and_grad(score)

    bootstrap = {"context": {"runtime": {"jax": jax, "jnp": jnp}}, "fixed_parameters": fixed,
                 "flux_basis": COMPONENT_RESOLVED_FLUX_BASIS,
                 "steps": {"thirty_forward": lambda v, m: (statistics(rollout(v, index(m)), index(m)), {})}}
    def initialize(*a, **kw):
        assert not args.output.exists()
        return bootstrap

    monkeypatch.setattr(audit.window, "load_recovery", lambda *a, **kw: {"completed_logical_update": 3, "trainable": values})
    monkeypatch.setattr(audit.window, "_load_parent_state", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(audit.window, "load_coupled_gpp_conditioned_attention_contract", lambda p: None)
    monkeypatch.setattr(audit.window, "_conditioned_bootstrap", initialize)
    monkeypatch.setattr(audit.window, "_load_artifact", lambda *a: artifact)
    monkeypatch.setattr(audit.window, "replace_conditioned_coupled_gpp_trainable", lambda f, v: v)
    monkeypatch.setattr(audit.window, "_make_trace_evaluator", lambda b: evaluator)
    monkeypatch.setattr(audit.window, "_hard_host", lambda h: h)
    monkeypatch.setattr(audit.window, "_hard_failure_count", lambda h: 0)
    monkeypatch.setattr(audit.streamed, "make_observation_score", observation_score)
    monkeypatch.setattr(audit.streamed, "make_step", lambda b: lambda v, s, f, d: step(v, s, f, d[0].forcing))
    result = audit.run(args)
    assert result["status"] == "completed"
    assert result["admission"]["passed"]
    assert result["baseline"]["loss"] == pytest.approx(float(full_loss(values)))
    assert result["reverse_compile_count"] == 1
    assert len(result["evaluations"]) == 4
    assert not list(tmp_path.glob("*.pkl"))
    assert {p.name for p in tmp_path.iterdir()} == {"report.json", "contract.json", "result.json"}
    incomplete = copy.deepcopy(result)
    incomplete["evaluations"] = incomplete["evaluations"][:1]
    assert not audit.classify(incomplete)["passed"]
    bad = copy.deepcopy(result)
    bad["finite_differences"][0]["relative_derivative_error"] = 0.1
    assert not audit.classify(bad)["passed"]
    bad = copy.deepcopy(result)
    bad["evaluations"][0]["hard_failure_count"] = 1
    assert not audit.classify(bad)["passed"]


def test_shared_launcher_supplies_required_args():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text()
    start = shared.index('elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"')
    branch = shared[start:shared.index("elif [[", start + 8)]
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"'):]
    assert '"scripts.hpc.audit_conditioned_attention_streamed_step" ||' in branch
    required = {o for a in audit._parser()._actions if a.required for o in a.option_strings}
    assert all(o in branch or o in command for o in required)
    wrapper = (root / "scripts/hpc/audit_conditioned_attention_streamed_step.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in wrapper
    assert "sbatch" not in shared + wrapper
