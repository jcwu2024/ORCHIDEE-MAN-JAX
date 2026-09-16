from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
from research.daily_coarse_graining.daily_operator.one_day_training_state import (
    make_gradient_accumulator,
)
from scripts.hpc.mixed_learning_profiling import (
    equal_compute_joint_updates,
    profile_components,
    profile_learning_budget,
)

jax.config.update("jax_enable_x64", True)


class _WindowPool:
    def case(self, runtime, row):
        horizon = int(row["length"])
        days = jnp.arange(horizon, dtype=jnp.float64)
        return (
            jnp.asarray(row["initial"], dtype=jnp.float64),
            jnp.asarray(0, dtype=jnp.int32),
            days,
            jnp.asarray(row["target"], dtype=jnp.float64),
            jnp.zeros((horizon,), dtype=jnp.int32),
        )


def _callbacks():
    parameters = {
        "weight": jnp.asarray([0.5, -0.25], dtype=jnp.float64),
        "bias": jnp.asarray(0.1, dtype=jnp.float32),
    }
    optimizer = {"step": jnp.asarray(200), "moment": jnp.zeros((2,))}

    def sample(rows):
        return np.asarray([[row["x"], row["target"]] for row in rows], dtype=np.float64)

    def daily_executable(parameters, batch, coefficients):
        prediction = parameters["weight"][0] * batch[:, 0] + parameters["weight"][1]
        residual = prediction - batch[:, 1]
        loss = jnp.mean(residual**2) * coefficients[0]
        gradient = {
            "weight": jnp.asarray(
                [2 * jnp.mean(residual * batch[:, 0]), 2 * jnp.mean(residual)],
                dtype=jnp.float64,
            ),
            "bias": jnp.asarray(jnp.mean(residual), dtype=jnp.float32),
        }
        return (loss, {"hard": {"physical": jnp.zeros((8,), dtype=jnp.int32)}}), gradient

    def forward(parameters, initial, discrete, days):
        observations = initial + days * parameters["weight"][0]
        return observations[-1], observations

    def validate(case, observations):
        return {
            "physical": jnp.sum(~jnp.isfinite(observations)),
            "terms": jnp.asarray(0, dtype=jnp.int32),
            "feedback": jnp.asarray(0, dtype=jnp.int32),
        }

    def recurrent(parameters, initial, discrete, days, targets, context):
        observations = initial + days * parameters["weight"][0]
        residual = observations[-1] - targets
        loss = residual**2 * context["weight"]
        gradient = {
            "weight": jnp.asarray(
                [2 * residual * days[-1], residual], dtype=jnp.float64
            ),
            "bias": jnp.asarray(residual, dtype=jnp.float32),
        }
        return loss, gradient

    def update(parameters, gradient, optimizer):
        norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(value.astype(jnp.float64))) for value in jax.tree_util.tree_leaves(gradient))
        )
        clip = jnp.minimum(1.0, 10.0 / norm)
        proposed = jax.tree_util.tree_map(
            lambda parameter, value: parameter - 1e-3 * clip.astype(value.dtype) * value,
            parameters,
            gradient,
        )
        return proposed, {**optimizer, "step": optimizer["step"] + 1}, norm, clip

    add, finish = make_gradient_accumulator(jax, jnp)
    combine = make_mixed_gradient(jax, jnp)
    finite = jax.jit(
        lambda tree: jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(tree)])
        )
    )
    norm_fn = jax.jit(
        lambda tree: jnp.sqrt(
            sum(jnp.sum(jnp.square(value.astype(jnp.float64))) for value in jax.tree_util.tree_leaves(tree))
        )
    )
    rows = [{"x": index / 10, "target": 0.2 + index / 20} for index in range(64)]
    recurrent_rows = {
        "T7": {"length": 7, "initial": 0.3, "target": 2.0},
        "T30": {"length": 30, "initial": -0.2, "target": 9.0},
    }
    return dict(
        jax=jax,
        jnp=jnp,
        parameters=parameters,
        optimizer=optimizer,
        sample=sample,
        daily_rows=rows,
        recurrent_rows=recurrent_rows,
        window_pool=_WindowPool(),
        daily_executable=daily_executable,
        recurrent_executables={7: recurrent, 30: recurrent},
        coefficients=jnp.asarray([1.0]),
        context={"weight": jnp.asarray(1.0)},
        forward=forward,
        validate_recurrent=validate,
        update=update,
        add=add,
        finish=finish,
        combine=combine,
        finite=finite,
        norm_fn=norm_fn,
    )


def test_profile_components_has_declared_keyword_only_api_and_replays_without_retention():
    signature = inspect.signature(profile_components)
    assert list(signature.parameters) == [
        "jax", "jnp", "parameters", "optimizer", "sample", "daily_rows",
        "recurrent_rows", "window_pool", "daily_executable", "recurrent_executables",
        "coefficients", "context", "forward", "validate_recurrent", "update", "add",
        "finish", "combine", "finite", "norm_fn", "repeats", "trace_dir", "emit",
        "checked_recurrent_executables", "checked_counter_names",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    callbacks = _callbacks()
    original_parameters = jax.tree_util.tree_map(np.asarray, callbacks["parameters"])
    original_step = int(callbacks["optimizer"]["step"])
    events = []
    result = profile_components(
        **callbacks,
        repeats=2,
        emit=lambda stage, **info: events.append((stage, info)),
    )

    assert result["status"] == "completed"
    assert result["parity"]["passed"]
    assert result["retained_parameter_or_optimizer_state"] is False
    assert result["parameters"]["element_count"] == 3
    assert result["parameters"]["by_dtype"]["float64"]["element_count"] == 2
    assert result["parameters"]["by_dtype"]["float32"]["element_count"] == 1
    assert len(result["baseline"]["samples_seconds"]) == 2
    assert len(result["instrumented"]["overall"]["samples_seconds"]) == 2
    assert set(result["instrumented"]["components"]) == {
        "daily_read_pack", "daily_transfer", "daily_backward", "daily_checks",
        "daily_accumulate", "T7_read_pack_transfer", "T7_forward",
        "T7_validity_loss_checks", "T7_backward", "T7_norm_merge",
        "T30_read_pack_transfer", "T30_forward", "T30_validity_loss_checks",
        "T30_backward", "T30_norm_merge", "combined_finite", "adamw",
        "proposed_finite",
    }
    assert result["profile_to_baseline_inflation"]["ratio"] > 0
    assert events[0][0] == "profiling_warmup"
    assert events[-1][0] == "profiling_completed"
    for actual, expected in zip(
        jax.tree_util.tree_leaves(callbacks["parameters"]),
        jax.tree_util.tree_leaves(original_parameters),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert int(callbacks["optimizer"]["step"]) == original_step


def test_trace_failure_is_reported_once_without_repeating_trace_replay(tmp_path):
    callbacks = _callbacks()
    calls = []

    class _FailingTrace:
        def __call__(self, path):
            calls.append(path)
            raise RuntimeError("profiler unavailable")

    callbacks["jax"] = SimpleNamespace(
        tree_util=jax.tree_util,
        block_until_ready=jax.block_until_ready,
        device_get=jax.device_get,
        device_put=jax.device_put,
        profiler=SimpleNamespace(trace=_FailingTrace()),
    )
    result = profile_components(**callbacks, repeats=1, trace_dir=tmp_path / "trace")
    assert result["status"] == "completed"
    assert result["trace"]["status"] == "failed"
    assert "profiler unavailable" in result["trace"]["error"]
    assert len(calls) == 1


def test_checked_replay_uses_compact_aux_without_a_separate_forward():
    callbacks = _callbacks()
    reference = profile_components(**callbacks, repeats=1)

    def checked(parameters, initial, discrete, days, targets, context, expected):
        value, gradient = callbacks['recurrent_executables'][7](
            parameters, initial, discrete, days, targets, context)
        return (value, jnp.zeros((1,), dtype=jnp.int64)), gradient

    def forbidden(*args):
        raise AssertionError('checked replay must not invoke a separate forward/check')

    callbacks.update(forward=forbidden, validate_recurrent=forbidden)
    observed = profile_components(**callbacks, repeats=1,
        checked_recurrent_executables={7: checked, 30: checked},
        checked_counter_names=('physical',))
    assert observed['parity']['reference'] == reference['parity']['reference']
    assert observed['recurrent_execution'] == 'compact_checked'
    keys = observed['instrumented']['components']
    assert 'T30_checked_backward' in keys
    assert 'T30_forward' not in keys


def _budget_callbacks():
    callbacks = _callbacks()

    def checked(horizon):
        def execute(parameters, initial, discrete, days, targets, context, expected):
            value, gradient = callbacks['recurrent_executables'][horizon](
                parameters, initial, discrete, days, targets, context)
            return (value, jnp.zeros((1,), dtype=jnp.int64)), gradient

        return execute

    return {
        key: value
        for key, value in callbacks.items()
        if key not in ('forward', 'validate_recurrent')
    } | {
        'recurrent_executables': {7: checked(7), 30: checked(30)},
        'checked_counter_names': ('physical',),
    }


def test_learning_budget_interleaves_complete_discarded_objectives():
    callbacks = _budget_callbacks()
    original_parameters = jax.tree_util.tree_map(np.asarray, callbacks['parameters'])
    original_step = int(callbacks['optimizer']['step'])
    events = []

    result = profile_learning_budget(
        **callbacks,
        repeats=3,
        emit=lambda stage, **info: events.append((stage, info)),
    )

    assert result['status'] == 'completed'
    assert result['schema_version'] == 'mixed_learning_budget_profile_v1'
    assert result['retained_parameter_or_optimizer_state'] is False
    assert result['rotating_data_throughput_accepted'] is False
    assert result['fixed_checkpoint'] and result['fixed_daily_and_recurrent_inputs']
    assert len(result['daily_only']['samples_seconds']) == 3
    assert len(result['mixed']['samples_seconds']) == 3
    assert [row['objective'] for row in result['replay_order']] == [
        'daily_only', 'mixed', 'mixed', 'daily_only', 'daily_only', 'mixed'
    ]
    assert all(set(row['losses']) == {'daily'} for row in result['daily_only']['records'])
    assert all(
        set(row['losses']) == {'daily', 'T7', 'T30'}
        for row in result['mixed']['records']
    )
    expected = equal_compute_joint_updates(
        result['daily_only']['median_seconds'], result['mixed']['median_seconds'])
    assert result['equal_compute'] == expected
    assert events[:2] == [
        ('learning_budget_warmup', {'objective': 'daily_only', 'discarded': True}),
        ('learning_budget_warmup', {'objective': 'mixed', 'discarded': True}),
    ]
    assert events[-1][0] == 'learning_budget_completed'
    for actual, expected_value in zip(
        jax.tree_util.tree_leaves(callbacks['parameters']),
        jax.tree_util.tree_leaves(original_parameters),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected_value)
    assert int(callbacks['optimizer']['step']) == original_step


def test_equal_compute_joint_budget_uses_declared_nearest_ten_formula():
    result = equal_compute_joint_updates(2.0, 4.0)
    assert result['unrounded_joint_updates'] == 600.0
    assert result['joint_updates_nearest_10'] == 600
    assert result['formula'] == (
        'nearest10(400*daily_median_seconds/mixed_median_seconds+400)'
    )
    assert equal_compute_joint_updates(1.04, 4.0)['joint_updates_nearest_10'] == 500
    for daily, mixed in ((0.0, 1.0), (1.0, float('nan'))):
        with np.testing.assert_raises(ValueError):
            equal_compute_joint_updates(daily, mixed)
