import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    COMPONENT_RESOLVED_FLUX_BASIS,
    INDEPENDENT_STOCK_BASIS,
    MaskedTrajectory,
    science_window_from_sufficient_statistics,
    science_window_sufficient_statistics,
)
from scripts.hpc import audit_conditioned_attention_streamed_integration as audit
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import _horizon_gradient
from scripts.hpc.run_target_oriented_full_step_preflight import _zero_hard

jax.config.update("jax_enable_x64", True)


class PFT(NamedTuple):
    pft_fraction: object
    active_pft_mask: object


class Inputs(NamedTuple):
    continuous_state: object
    discrete_state: object
    forcing: object
    pft: object


class Group(NamedTuple):
    other: object


class Micro(NamedTuple):
    inputs: object
    expected_next_state: object
    expected_next_state_defined: object
    expected_fast_gpp: object
    expected_fast_gpp_defined: object
    day_inputs: object


class Prediction(NamedTuple):
    named_outputs: object


class Result(NamedTuple):
    next_state: object
    next_state_defined: object
    prediction: object
    next_discrete_state: object


class Trace(NamedTuple):
    result: object
    pft_fraction: object
    active_pft_mask: object


def transition(v, state, fixed, day):
    following = jnp.tanh(fixed * state + v["weight"] * day[0].forcing + v["bias"])
    return {"state": following, "gpp": (following**2)[None, :], "fraction": jax.nn.sigmoid(state)}


def extract(state, defined, *, daily_fast_gpp, pft_fraction, **kwargs):
    def trajectory(value):
        return MaskedTrajectory(value, jnp.all(defined, axis=-1))
    gpp = trajectory(jnp.sum(daily_fast_gpp[:, 0, :] * pft_fraction, axis=-1))
    return SimpleNamespace(
        fluxes={"gpp": gpp, "maintenance_respiration": trajectory(state[:, 0] * 0.1),
                "growth_respiration": trajectory(state[:, 1] * 0.2), "heterotrophic_respiration": trajectory(state[:, 0] * 0.03)},
        stocks={"aboveground_biomass_carbon": trajectory(state[:, 0]),
                "belowground_biomass_carbon": trajectory(state[:, 1]),
                "total_live_biomass_carbon": trajectory(jnp.sum(state, axis=-1)),
                "ecosystem_carbon": trajectory(jnp.sum(state, axis=-1) * 1.5)},
    )


def make_batch(horizon, shift):
    count = 8 if horizon == 7 else 4
    starts = jnp.stack([jnp.array([0.1, -0.2]) + shift + i * 0.01 for i in range(count)])
    state = jnp.broadcast_to(starts[:, None, :], (count, horizon, 2))
    expected = jnp.full_like(state, 0.03 + shift)
    mask = jnp.ones_like(state, dtype=bool)
    # A completely masked window makes global valid counts differ across batches.
    if shift > 0:
        mask = mask.at[0].set(False)
    return Micro(
        Inputs(state, {"flag": jnp.zeros((count, horizon), dtype=bool)},
               jnp.broadcast_to(jnp.linspace(0.3 + shift, 1.0, horizon), (count, horizon)),
               PFT(jnp.full_like(state, 0.5), jnp.ones_like(state, dtype=bool))),
        Group(expected), Group(mask), jnp.full((count, horizon, 1, 2), 0.02 + shift),
        jnp.ones((count, horizon, 1, 2), dtype=bool), jnp.zeros((count, 2)),
    )


def setup_bootstrap(monkeypatch):
    import research.daily_coarse_graining.daily_operator.successor_training as training
    monkeypatch.setattr(training, "extract_biomass_productivity_trajectories", extract)
    fixed = jnp.array([0.7, 0.6])
    scales = SimpleNamespace(fluxes={k: 1.0 for k in COMPONENT_RESOLVED_FLUX_BASIS}, stocks={k: 1.0 for k in INDEPENDENT_STOCK_BASIS})

    def evaluate(values, micro):
        templates = jax.tree_util.tree_map(lambda v: v[0], micro.inputs)
        def scan(state, template):
            obs = transition(values, state, fixed, (template, {}, micro.day_inputs[0]))
            return obs["state"], obs
        obs = jax.lax.scan(scan, templates.continuous_state[0], templates)[1]
        return Trace(Result(Group(obs["state"]), Group(jnp.ones_like(obs["state"], dtype=bool)),
                            Prediction({"daily.fast_gpp": obs["gpp"]}), templates.discrete_state),
                     obs["fraction"], templates.pft.active_pft_mask), None

    def forward(values, micro):
        trace, _ = evaluate(values, micro)
        predicted = extract(trace.result.next_state.other, trace.result.next_state_defined.other,
                            daily_fast_gpp=trace.result.prediction.named_outputs["daily.fast_gpp"], pft_fraction=trace.pft_fraction)
        expected = extract(micro.expected_next_state.other[0], micro.expected_next_state_defined.other[0],
                           daily_fast_gpp=micro.expected_fast_gpp[0], pft_fraction=micro.inputs.pft.pft_fraction[0])
        statistics = science_window_sufficient_statistics(predicted.fluxes, expected.fluxes, predicted.stocks, expected.stocks,
                    flux_scales=scales.fluxes, stock_scales=scales.stocks, horizon=micro.inputs.forcing.shape[1], flux_basis=COMPONENT_RESOLVED_FLUX_BASIS)
        return statistics, _zero_hard(jnp)

    sentinel = object()
    bootstrap = {"context": {"runtime": {"jax": jax, "jnp": jnp, "contract": None}, "scales": scales},
                 "fixed_parameters": fixed, "flux_basis": COMPONENT_RESOLVED_FLUX_BASIS,
                 "steps": {"seven_forward": forward, "thirty_forward": forward},
                 "gradient_steps": {"unrelated": sentinel, "seven_gradient": sentinel, "thirty_gradient": sentinel},
                 "direct_steps": sentinel, "state_steps": sentinel}
    monkeypatch.setattr(audit.window, "replace_conditioned_coupled_gpp_trainable", lambda f, v: v)
    monkeypatch.setattr(audit.window, "_make_trace_evaluator", lambda b: evaluate)
    monkeypatch.setattr(audit.streamed, "make_step", lambda b: transition)
    return bootstrap


def test_actual_horizon_caller_matches_monolithic_with_changing_targets_and_horizons(monkeypatch):
    bootstrap = setup_bootstrap(monkeypatch)
    callbacks = audit.StreamedHorizonCallbacks(bootstrap)
    attached = callbacks.attach()
    assert attached["gradient_steps"]["unrelated"] is bootstrap["gradient_steps"]["unrelated"]
    assert attached["direct_steps"] is bootstrap["direct_steps"]
    assert attached["state_steps"] is bootstrap["state_steps"]
    for shift in (0.0, 0.04):
        values = {"weight": jnp.array([0.2 + shift, -0.1], dtype=jnp.float32), "bias": jnp.array(0.03, dtype=jnp.float32)}
        batch = SimpleNamespace(seven_day=make_batch(7, shift), thirty_day=make_batch(30, shift))
        for horizon in (7, 30):
            result, hard, gradient, _ = _horizon_gradient(jax, jnp, values, values, batch, bootstrap["steps"], attached["gradient_steps"], horizon=horizon, flux_basis=COMPONENT_RESOLVED_FLUX_BASIS)
            source = batch.seven_day if horizon == 7 else batch.thirty_day
            count = 8 if horizon == 7 else 4
            def objective(v):
                stats = [bootstrap["steps"]["seven_forward"](v, audit.window._slice(source, i, i + 1))[0] for i in range(count)]
                return science_window_from_sufficient_statistics(audit.step_audit.merge_statistics(stats)).total
            expected = jax.grad(objective)(values)
            assert float(result.total) == pytest.approx(float(objective(values)))
            for a, b in zip(jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(expected), strict=True):
                np.testing.assert_allclose(a, b, rtol=3e-6, atol=1e-8)
            assert all(int(v) == 0 for v in hard)
    assert callbacks.calls == {7: 16, 30: 8}
    assert callbacks.reverse.compile_count == 1
    assert all(r["linearized_score_agrees"] for r in callbacks.records)


def test_complete_runner_routes_original_panel_and_discards_parameters(tmp_path, monkeypatch):
    bootstrap = setup_bootstrap(monkeypatch)
    contract = audit.persistent.load_conditioned_attention_persistent_training_contract()
    args = SimpleNamespace(conditioned_run_dir=tmp_path, output=tmp_path / "result.json",
                           artifact_aware_persistent_contract=tmp_path / "contract.json", parent_persistent_training_report=None,
                           parent_artifact_aware_recovery=None, conditioning_contract=None, conditioned_persistent_contract=contract.source_path)
    (tmp_path / "report.json").write_text(json.dumps({"identity": {"test": True}}))
    args.artifact_aware_persistent_contract.write_text(json.dumps({"contract_sha256": "test"}))
    values = {"weight": jnp.array([0.2, -0.1], dtype=jnp.float32), "bias": jnp.array(0.03, dtype=jnp.float32)}
    monkeypatch.setattr(audit.window, "load_recovery", lambda *a, **kw: {"completed_logical_update": 3, "trainable": values})
    monkeypatch.setattr(audit.window, "_load_parent_state", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(audit.window, "load_coupled_gpp_conditioned_attention_contract", lambda p: None)
    def initialize(*a, **kw):
        assert not args.output.exists()
        return bootstrap
    monkeypatch.setattr(audit.window, "_conditioned_bootstrap", initialize)

    def proposed(a, b, c, **kw):
        assert kw["logical_update"] == 4
        assert kw["panel"] == contract.train_panels[1]
        assert kw["source_registry"] == contract.source_registries[1]
        assert len(kw["source_registry"]) == 43
        # Exercise the actual injected callbacks across all eight scheduled artifacts.
        for _ in kw["panel"][0]:
            for horizon in (7, 30):
                source = make_batch(horizon, 0.0)
                count = 8 if horizon == 7 else 4
                stats = [b["steps"]["seven_forward"](values, audit.window._slice(source, i, i + 1))[0] for i in range(count)]
                total = audit.step_audit.merge_statistics(stats)
                name = "seven_gradient" if horizon == 7 else "thirty_gradient"
                for i in range(count):
                    b["gradient_steps"][name](values, audit.window._slice(source, i, i + 1), total)
        return {"updated_trainable": {"must_not_serialize": object()}, "record": {"selection": {"accepted": True, "selected_scale": 0.5}}}

    monkeypatch.setattr(audit.persistent, "_train_only_update", proposed)
    result = audit.run(args)
    assert result["status"] == "completed"
    assert result["admission"]["passed"]
    assert len(result["horizon_records"]) == 96
    assert result["horizon_callback_counts"] == {"7": 64, "30": 32}
    assert result["reverse_compile_count"] == 1
    assert {p.name for p in tmp_path.iterdir()} == {"report.json", "contract.json", "result.json"}


def test_classification_needs_original_gates_complete_gradients_and_primal_agreement():
    record = {"selection": {"accepted": True, "selected_scale": 1.0}}
    callbacks = SimpleNamespace(calls={7: 64, 30: 32}, records=[{"linearized_score_agrees": True} for _ in range(96)])
    assert audit.classify(record, callbacks)["passed"]
    callbacks.records[0]["linearized_score_agrees"] = False
    assert not audit.classify(record, callbacks)["passed"]
    callbacks.records[0]["linearized_score_agrees"] = True
    callbacks.calls[30] = 31
    assert not audit.classify(record, callbacks)["passed"]
    callbacks.calls[30] = 32
    record["selection"]["accepted"] = False
    assert not audit.classify(record, callbacks)["passed"]


def test_launcher_argument_branch():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text()
    start = shared.index('elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"')
    branch = shared[start:shared.index("elif [[", start + 8)]
    assert '"scripts.hpc.audit_conditioned_attention_streamed_integration" ||' in branch
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"'):]
    assert all(o in branch or o in command for a in audit._parser()._actions if a.required for o in a.option_strings)
    wrapper = (root / "scripts/hpc/audit_conditioned_attention_streamed_integration.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in wrapper
    assert "sbatch" not in shared + wrapper
