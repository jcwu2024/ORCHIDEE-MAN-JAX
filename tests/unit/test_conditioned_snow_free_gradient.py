"""Corrected-head binding and the full mixed-precision diagnostic lifecycle."""
import copy
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_conditioned_attention_streamed_step import (
    COMPONENT_RESOLVED_FLUX_BASIS,
    Inputs,
    Micro,
    setup_case,
    statistics,
    step,
)

from research.daily_coarse_graining.daily_operator import target_oriented_process_head as heads
from research.daily_coarse_graining.daily_operator.streamed_adjoint import (
    AdjointPrediction,
    AdjointResult,
    AdjointState,
    AdjointTrace,
)
from scripts.hpc import audit_conditioned_snow_free_gradient as audit
from scripts.hpc import audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update as bootstrap_module
from scripts.hpc.run_target_oriented_full_step_preflight import HardCounts


@pytest.mark.parametrize("enabled", [False, True])
def test_bootstrap_binds_identical_head_everywhere(monkeypatch, enabled):
    @dataclass(frozen=True)
    class Head:
        warm_snow_free_boundary: bool = False
        structured_config: object = None
        conditioning_config: object = None

    runtime = {"jax": jax, "base_statistics": NS(arrays={"next_state": NS(scale=1.)}),
               "process_statistics": NS(outputs={"daily.fast_gpp": NS(scale=1.)}),
               "contract": None, "adapter": NS(input_assembler=None)}
    base = {"context": {"runtime": runtime, "template": None, "scales": NS(state=NS(other=1.)),
                        "target_contract": None}, "head": Head(), "initialized_parameters": {},
            "flux_basis": (), "layout": None}
    seen = []
    monkeypatch.setattr(bootstrap_module, "_initialization_only", lambda a: base)
    monkeypatch.setattr(heads, "build_target_oriented_coupled_gpp_conditioned_attention_head", lambda h, **kw: h)
    for name in ("replace_coupled_gpp_trainable", "initialize_coupled_gpp_conditioned_attention_parameters",
                 "conditioned_coupled_gpp_trainable_tree", "replace_conditioned_coupled_gpp_trainable"):
        monkeypatch.setattr(bootstrap_module, name, lambda *a, **kw: {})
    sizes = iter((bootstrap_module.TOTAL_PARAMETER_COUNT, bootstrap_module.TRAINABLE_PARAMETER_COUNT,
                  bootstrap_module.RETAINED_PARAMETER_COUNT))
    monkeypatch.setattr(bootstrap_module, "_tree_size", lambda *a: next(sizes))
    monkeypatch.setattr(bootstrap_module, "derive_target_oriented_acceptance_scales", lambda *a: None)

    def factory(runtime, head, *a, **kw):
        seen.append(head)
        return {"make_trainable_gradient_steps": lambda *a: seen.append(head)}

    def bind(**kw):
        seen.append(kw["process_head"])
        return kw["process_head"]

    monkeypatch.setattr(bootstrap_module, "_make_steps", factory)
    monkeypatch.setattr(bootstrap_module, "_make_direct_steps", factory)
    monkeypatch.setattr(bootstrap_module, "bind_daily_operator_dynamic_retained_inputs", bind)
    monkeypatch.setattr(bootstrap_module, "_make_state_steps", factory)
    options = {"warm_snow_free_boundary": True} if enabled else {}
    result = bootstrap_module._conditioned_bootstrap(None, recovered={"trainable": {}}, conditioning_contract=None, **options)
    assert len(seen) == 5
    assert all(h is result["head"] and h.warm_snow_free_boundary == enabled for h in seen)
    assert not base["head"].warm_snow_free_boundary


def test_full_three_window_runner_with_real_reverse_jvp_fd_and_hard_counts(tmp_path, monkeypatch):
    identity = {"fixture": True}
    args = NS(conditioned_run_dir=tmp_path, output=tmp_path / "result.json", probe=tmp_path / "probe.pkl",
              artifact_aware_persistent_contract=tmp_path / "contract.json", parent_persistent_training_report=None,
              parent_artifact_aware_recovery=None, conditioning_contract=None)
    (tmp_path / "report.json").write_text(json.dumps({"identity": identity}))
    args.artifact_aware_persistent_contract.write_text(json.dumps({"contract_sha256": "fixture"}))
    cases = {h: setup_case(jnp.float32, h) for h in (7, 30)}
    values, fixed = cases[7][:2]
    with args.probe.open("wb") as handle:
        pickle.dump({"parent_identity": identity, "completed_logical_update": 3, "artifact_update": 442,
                     "window": 0, "direction": jax.device_get(jax.tree_util.tree_map(jnp.ones_like, values)),
                     "total": "must not be reused"}, handle)
    artifacts = {}
    for update, horizon, count in ((322, 7, 8), (442, 30, 4)):
        _, _, initial, days, _, _ = cases[horizon]
        batch = Micro(Inputs(jnp.stack([jnp.broadcast_to(initial[i % 4], (horizon, 2)) for i in range(count)]),
                             {"index": jnp.stack([jnp.full(horizon, i % 4, dtype=jnp.int32) for i in range(count)])},
                             jnp.stack([days[i % 4] for i in range(count)])), jnp.zeros(count))
        artifacts[update] = NS(batch=NS(**{"seven_day" if horizon == 7 else "thirty_day": batch}))

    def coords(micro):
        return micro.inputs.continuous_state.shape[1], int(micro.inputs.discrete_state["index"][0, 0])

    def trace(v, micro):
        horizon, index = coords(micro)
        obs = cases[horizon][4](v, index)
        return AdjointTrace(AdjointResult(AdjointState(obs["state"]), AdjointState(jnp.ones_like(obs["state"], dtype=bool)),
            {"index": jnp.full(horizon, index, dtype=jnp.int32)}, AdjointPrediction({"daily.fast_gpp": obs["gpp"]})),
            obs["fraction"], jnp.ones_like(obs["state"], dtype=bool)), None

    def forward(v, micro):
        horizon, index = coords(micro)
        return statistics(cases[horizon][4](v, index), index), HardCounts(0, 0, 0, horizon, 0, 0, 0)

    def make_score(b, h):
        def evaluate(obs, defined, active, total, micro):
            index = coords(micro)[1]
            return jax.value_and_grad(lambda o: audit.window.linearized_science_window_loss(
                statistics(o, index), total, flux_basis=b["flux_basis"]))(obs)
        return evaluate

    bootstrap = {"context": {"runtime": {"jax": jax, "jnp": jnp}, "scales": NS(state=NS(other=np.ones(2)))},
                 "head": NS(warm_snow_free_boundary=True), "fixed_parameters": fixed,
                 "flux_basis": COMPONENT_RESOLVED_FLUX_BASIS,
                 "steps": {"seven_forward": forward, "thirty_forward": forward}}

    def initialize(*a, **kw):
        assert kw["warm_snow_free_boundary"] and not args.output.exists()
        return bootstrap

    monkeypatch.setattr(audit.window, "load_recovery", lambda *a, **kw: {"completed_logical_update": 3, "trainable": values})
    monkeypatch.setattr(audit.window, "_load_parent_state", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(audit.window, "load_coupled_gpp_conditioned_attention_contract", lambda *a: None)
    monkeypatch.setattr(audit.window, "_conditioned_bootstrap", initialize)
    monkeypatch.setattr(audit.window, "_load_artifact", lambda a, c, u: artifacts[u])
    monkeypatch.setattr(audit.window, "replace_conditioned_coupled_gpp_trainable", lambda f, v: v)
    monkeypatch.setattr(audit.integration, "make_compact_trace_evaluator", lambda b: trace)
    monkeypatch.setattr(audit.integration, "make_observation_gradient", make_score)
    monkeypatch.setattr(audit.streamed, "make_step", lambda b: lambda v, s, f, d: step(v, s, f, d[0].forcing))
    monkeypatch.setattr(audit, "selected_days", lambda b, m, t: np.ones(coords(m)[0], dtype=bool))
    result = audit.run(args)
    json.dumps(result, allow_nan=False)
    assert result["status"] == "completed" and result["passed"]
    assert result["reverse_compile_count"] == 1
    assert len(result["cases"]) == 3
    assert sum(len(d["pairs"]) for r in result["cases"] for d in r["directions"]) == 8
    assert len(list(tmp_path.glob("*.pkl"))) == 1  # Only the input probe; no new weights.
    for key, value in (("hard_failure_count", 1), ("all_days_selected", False), ("score_agrees", False), ("gradient_finite", False)):
        row = copy.deepcopy(result["cases"][0])
        row[key] = value
        assert not audit.classify_case(row)
    row = copy.deepcopy(result["cases"][0])
    row["directions"][0]["pairs"].pop()
    assert not audit.classify_case(row)
    assert not audit.comparison(float("nan"), 0.)["passed"]
    assert not audit.same_masks(jax, {"a": np.array([True])}, {"a": np.array([False])})


def test_launcher_supplies_required_arguments():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text()
    start = shared.index('elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"')
    branch = shared[start:shared.index("elif [[", start + 8)]
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"'):]
    assert '"scripts.hpc.audit_conditioned_snow_free_gradient" ||' in branch
    required = {o for a in audit._parser()._actions if a.required for o in a.option_strings}
    assert all(o in branch or o in command for o in required)
    wrapper = (root / "scripts/hpc/audit_conditioned_snow_free_gradient.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-60m}" in wrapper
    assert "sbatch" not in shared + wrapper
