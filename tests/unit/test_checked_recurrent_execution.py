from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
from scripts.hpc.checked_recurrent_execution import (
    bind_checked_backward,
    decode_counts,
    wrap_separate_checked,
)

jax.config.update("jax_enable_x64", True)


def _step(parameters, continuous, discrete, day):
    template, forcing = day
    following = jnp.tanh(parameters["gain"] * continuous + forcing)
    next_discrete = {"phase": discrete["phase"] + template["increment"]}
    observations = {
        "defined": template["defined"],
        "discrete": next_discrete,
    }
    return following, next_discrete, observations


def _arguments():
    parameters = {
        "gain": jnp.asarray([0.35, -0.20], dtype=jnp.float64),
        "regularizer": jnp.asarray([0.11, -0.07], dtype=jnp.float64),
    }
    initial = jnp.asarray([0.10, -0.15], dtype=jnp.float64)
    discrete = {"phase": jnp.asarray([1, 2], dtype=jnp.int32)}
    templates = {
        "defined": jnp.asarray([[True, True], [True, False], [False, True]]),
        "increment": jnp.asarray([[1, 1], [2, 1], [1, 2]], dtype=jnp.int32),
    }
    forcing = jnp.asarray([[0.04, -0.02], [0.03, 0.01], [-0.01, 0.02]])
    targets = {
        "defined": jnp.asarray([[True, True], [False, False], [False, True]]),
        "state": jnp.asarray([[0.08, -0.03], [0.04, 0.02], [-0.02, 0.05]]),
    }
    expected_discrete = {
        "phase": jnp.asarray([[2, 3], [4, 4], [6, 6]], dtype=jnp.int32)
    }
    context = {"direct_weight": jnp.asarray(0.17, dtype=jnp.float64)}
    return parameters, initial, discrete, (templates, forcing), targets, context, expected_discrete


class _RecurrentLoss:
    def terms(self, parameters, initial, observations, targets, context):
        del initial
        valid = observations["defined"] & targets["defined"]
        residual = jnp.where(valid, observations["state"] - targets["state"], 0.0)
        total = (
            jnp.sum(residual**2)
            + context["direct_weight"] * jnp.sum(parameters["regularizer"] ** 2)
        )
        term = SimpleNamespace(
            count=jnp.sum(valid),
            invalid_prediction_count=jnp.sum(~observations["defined"]),
            invalid_target_count=jnp.sum(~targets["defined"]),
        )
        return SimpleNamespace(
            total=total,
            daily_flux={"flux": term},
            stock_endpoint={"stock": term},
        )

    def __call__(self, parameters, initial, observations, targets, context):
        return self.terms(parameters, initial, observations, targets, context).total


class _TrackingJax:
    tree_util = jax.tree_util

    def __init__(self):
        self.device_put_calls = 0

    def device_put(self, value):
        self.device_put_calls += 1
        return jax.device_put(value)

    def device_get(self, value):
        raise AssertionError("the split wrapper must not copy observations to host")

    def block_until_ready(self, value):
        return jax.block_until_ready(value)


def _patch_dependencies(monkeypatch):
    from research.daily_coarse_graining.daily_operator import (
        fresh_start_training,
        recurrent_science_loss,
        recurrent_validation,
    )
    from scripts.hpc import run_trained_forward_screen

    monkeypatch.setattr(fresh_start_training, "_canopy_change_plan", lambda *args: None)
    monkeypatch.setattr(fresh_start_training, "_litterpart_feedback_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(recurrent_science_loss, "FLUX_KEYS", ("flux",))
    monkeypatch.setattr(recurrent_science_loss, "STOCK_KEYS", ("stock",))
    monkeypatch.setattr(
        recurrent_science_loss, "WEIGHT_COMPONENTS", ("daily_flux", "stock_endpoint")
    )
    monkeypatch.setattr(
        recurrent_validation, "PHYSICAL_COUNT_NAMES", ("discrete_mismatch", "state_nonfinite")
    )

    def physical_counts(observations, targets, expected_discrete, templates, contract, scales):
        del targets, templates, contract, scales
        return {
            "discrete_mismatch": jnp.sum(
                observations["discrete"]["phase"] != expected_discrete["phase"],
                dtype=jnp.int32,
            ),
            "state_nonfinite": jnp.sum(~jnp.isfinite(observations["state"]), dtype=jnp.int32),
        }

    monkeypatch.setattr(recurrent_validation, "physical_counts", physical_counts)
    monkeypatch.setattr(
        run_trained_forward_screen,
        "feedback_diagnostics",
        lambda *args, **kwargs: {"enabled": False},
    )


def _bind(monkeypatch, *, separate_validation):
    _patch_dependencies(monkeypatch)
    recurrent_loss = _RecurrentLoss()
    component = make_rollout_loss(jax, jnp, _step, recurrent_loss)
    head = SimpleNamespace(
        contract=object(),
        input_assembler=SimpleNamespace(assemble=lambda template: template),
    )
    backward, names, validator = bind_checked_backward(
        jax,
        jnp,
        component,
        recurrent_loss,
        {},
        head,
        object(),
        {"loss": {}},
        separate_validation=separate_validation,
    )
    return component, backward, names, validator


def test_split_wrapper_matches_inline_and_reuses_one_primal_without_host_copy(monkeypatch):
    arguments = _arguments()
    _, inline, inline_names, inline_validator = _bind(
        monkeypatch, separate_validation=False
    )
    inline_result = inline(*arguments)
    assert inline_validator is None

    component, raw, split_names, validator = _bind(monkeypatch, separate_validation=True)
    rollout_calls = []
    original_rollout = component.rollout

    def counted_rollout(*args, **kwargs):
        rollout_calls.append(kwargs.get("checkpointed"))
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(component, "rollout", counted_rollout)
    raw_executable = raw.lower(*arguments[:-1]).compile()
    tracking_jax = _TrackingJax()
    events = []
    checked = wrap_separate_checked(
        tracking_jax, raw_executable, validator, emit=events.append
    )
    split_result = checked(*arguments)

    assert inline_names == split_names
    np.testing.assert_allclose(split_result[0][0], inline_result[0][0], rtol=0.0, atol=0.0)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(split_result[1]),
        jax.tree_util.tree_leaves(inline_result[1]),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert decode_counts(split_names, split_result[0][1]) == decode_counts(
        inline_names, inline_result[0][1]
    )
    assert rollout_calls == [True]
    assert tracking_jax.device_put_calls == 1
    assert checked.statistics["observation_bytes"] == 78
    assert checked.statistics["raw_first_execution_seconds"] >= 0.0
    assert checked.statistics["validator_compile_count"] == 1
    assert checked.statistics["validator_compile_seconds"] >= 0.0
    assert checked.statistics["validator_memory"]["estimated_live_bytes"] > 0
    assert events == ["raw_executed", "validator_compile", "validator_compiled"]


def test_split_wrapper_compiles_validator_once_per_horizon_wrapper(monkeypatch):
    arguments = _arguments()
    _, raw, names, validator = _bind(monkeypatch, separate_validation=True)
    raw_executable = raw.lower(*arguments[:-1]).compile()
    tracking_jax = _TrackingJax()
    events = []
    checked = wrap_separate_checked(
        tracking_jax, raw_executable, validator, emit=events.append
    )

    first = checked(*arguments)
    changed_expected = {
        "phase": arguments[-1]["phase"].at[-1, -1].add(jnp.asarray(1, dtype=jnp.int32))
    }
    second = checked(*arguments[:-1], changed_expected)

    assert checked.statistics["validator_compile_count"] == 1
    assert tracking_jax.device_put_calls == 2
    assert events == ["raw_executed", "validator_compile", "validator_compiled"]
    assert decode_counts(names, first[0][1])["discrete_mismatch"] == 1
    assert decode_counts(names, second[0][1])["discrete_mismatch"] == 2
