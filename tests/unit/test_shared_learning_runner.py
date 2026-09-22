import json
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.run_shared_learning import (
    ROOT,
    LearningFailure,
    compile_gradient_evaluators,
    execute_learning_update,
    load_learning_checkpoint,
    make_tree_operations,
    parser,
    save_learning_checkpoint,
)


def _daily_objective(parameters, sample):
    prediction = parameters["weight"] * sample + parameters["bias"]
    process = (prediction - 0.25) ** 2
    flux = 0.5 * (prediction + 0.4) ** 2
    stock = 0.25 * (prediction - 0.8) ** 2
    value = process / 3.0 + (flux + stock) / 9.0
    return value, {
        "process_day": process,
        "flux": flux,
        "stock": stock,
        "joint_daily_contribution": value,
        "hard": {"physical": jnp.asarray(0, dtype=jnp.int32)},
    }


def _recurrent_objective(parameters, sample):
    prediction = parameters["weight"] * sample + parameters["bias"]
    flux = (prediction + 0.1) ** 2
    stock = 0.5 * (prediction - 0.2) ** 2
    value = (flux + stock) / 9.0
    return value, {
        "flux": flux,
        "stock": stock,
        "living_stock": stock / 2.0,
        "nonliving_stock": stock / 2.0,
        "hard": {"physical": jnp.asarray(0, dtype=jnp.int32)},
    }


def _evaluators():
    return {
        "daily": jax.jit(jax.value_and_grad(_daily_objective, has_aux=True)),
        "T7": jax.jit(jax.value_and_grad(_recurrent_objective, has_aux=True)),
        "T30": jax.jit(jax.value_and_grad(_recurrent_objective, has_aux=True)),
    }


def _parameters():
    return {
        "weight": jnp.asarray(0.6, dtype=jnp.float32),
        "bias": jnp.asarray(-0.15, dtype=jnp.float32),
    }


def _toy_update(counter=None):
    def update(parameters, gradient, optimizer):
        if counter is not None:
            counter.append(1)
        updated = jax.tree_util.tree_map(lambda value, grad: value - 0.02 * grad, parameters, gradient)
        norm = jnp.sqrt(sum(jnp.square(value) for value in jax.tree_util.tree_leaves(gradient)))
        return updated, optimizer + 1, norm, jnp.asarray(1.0)

    return update


def test_32_daily_gradients_are_averaged_then_recurrent_gradients_added_once():
    parameters = _parameters()
    daily = tuple(jnp.asarray(value, dtype=jnp.float32) for value in np.linspace(-1.0, 1.0, 32))
    seven = jnp.asarray(0.3, dtype=jnp.float32)
    thirty = jnp.asarray(-0.7, dtype=jnp.float32)
    counter = []
    updated, optimizer, record = execute_learning_update(
        jax=jax,
        parameters=parameters,
        optimizer=jnp.asarray(0),
        daily_samples=daily,
        seven_case=(seven,),
        thirty_case=(thirty,),
        evaluators=_evaluators(),
        operations=make_tree_operations(jax, jnp, 32),
        update=_toy_update(counter),
    )

    def direct(current):
        daily_mean = sum(_daily_objective(current, sample)[0] for sample in daily) / 32.0
        return (
            daily_mean
            + _recurrent_objective(current, seven)[0]
            + _recurrent_objective(current, thirty)[0]
        )

    expected_gradient = jax.grad(direct)(parameters)
    expected = jax.tree_util.tree_map(
        lambda value, gradient: value - 0.02 * gradient, parameters, expected_gradient
    )
    for actual, reference in zip(
        jax.tree_util.tree_leaves(updated),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)
    assert counter == [1]
    assert int(optimizer) == 1
    assert record["loss"]["joint"] == pytest.approx(float(direct(parameters)), rel=1e-6)
    assert record["gradient_norms"]["combined"] == pytest.approx(
        float(jnp.sqrt(sum(jnp.square(value) for value in expected_gradient.values()))), rel=1e-6
    )


def test_daily_sum_is_not_silently_used_in_place_of_mean():
    parameters = _parameters()
    sample = jnp.asarray(0.9, dtype=jnp.float32)
    daily = (sample,) * 32
    updated, _optimizer, _record = execute_learning_update(
        jax=jax,
        parameters=parameters,
        optimizer=jnp.asarray(0),
        daily_samples=daily,
        seven_case=(jnp.asarray(0.0),),
        thirty_case=(jnp.asarray(0.0),),
        evaluators=_evaluators(),
        operations=make_tree_operations(jax, jnp, 32),
        update=_toy_update(),
    )
    expected_gradient = jax.grad(
        lambda current: _daily_objective(current, sample)[0]
        + _recurrent_objective(current, jnp.asarray(0.0))[0] * 2.0
    )(parameters)
    expected = jax.tree_util.tree_map(
        lambda value, gradient: value - 0.02 * gradient, parameters, expected_gradient
    )
    for actual, reference in zip(updated.values(), expected.values(), strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)


def _daily_protocol():
    return json.loads(
        (ROOT / "manifests/coarse_graining/shared_transformer_admission_v3.json").read_text()
    )


def _adam_step(parameters, optimizer, batch):
    return execute_learning_update(
        jax=jax,
        parameters=parameters,
        optimizer=optimizer,
        daily_samples=batch,
        seven_case=(jnp.asarray(0.2),),
        thirty_case=(jnp.asarray(-0.4),),
        evaluators=_evaluators(),
        operations=make_tree_operations(jax, jnp, 32),
        update=make_update(_daily_protocol()),
    )


def test_checkpoint_resume_matches_continuous_parameters_and_adamw(tmp_path):
    initial = _parameters()
    first_batch = tuple(jnp.asarray(value) for value in np.linspace(-1.0, 0.0, 32))
    second_batch = tuple(jnp.asarray(value) for value in np.linspace(0.0, 1.0, 32))
    state = initialize_target_oriented_adamw(initial)
    after_one, state_one, row_one = _adam_step(initial, state, first_batch)
    continuous, continuous_state, _row_two = _adam_step(after_one, state_one, second_batch)

    method = {"configuration": {"max_updates": 2}, "identity": "toy"}
    path = tmp_path / "learning.checkpoint.pkl"
    save_learning_checkpoint(
        path,
        jax=jax,
        parameters=after_one,
        optimizer=state_one,
        completed_updates=1,
        method=method,
        updates=[{"update": 1, **row_one}],
        monitors=[],
    )
    saved = load_learning_checkpoint(path, expected_method=method)
    resumed, resumed_state, _record = _adam_step(
        jax.device_put(saved["parameters"]),
        jax.device_put(saved["optimizer"]),
        second_batch,
    )
    for actual, reference in zip(
        jax.tree_util.tree_leaves((resumed, resumed_state)),
        jax.tree_util.tree_leaves((continuous, continuous_state)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, reference)

    changed = deepcopy(method)
    changed["identity"] = "different-data"
    with pytest.raises(ValueError, match="method, data, or cursor"):
        load_learning_checkpoint(path, expected_method=changed)


@pytest.mark.parametrize("failure", ["gradient", "physical"])
def test_nonfinite_or_physical_component_is_rejected_before_update(failure):
    parameters = _parameters()
    calls = []

    def bad(parameters, sample):
        value, auxiliary = _daily_objective(parameters, sample)
        if failure == "gradient":
            broken = jnp.sqrt((parameters["weight"] - parameters["weight"]) ** 2)
            value = value + broken
        else:
            auxiliary["hard"] = {"physical": jnp.asarray(1, dtype=jnp.int32)}
        return value, auxiliary

    evaluators = _evaluators()
    evaluators["daily"] = jax.jit(jax.value_and_grad(bad, has_aux=True))
    with pytest.raises(LearningFailure, match="daily_numerical_or_physical"):
        execute_learning_update(
            jax=jax,
            parameters=parameters,
            optimizer=jnp.asarray(0),
            daily_samples=(jnp.asarray(0.1),) * 32,
            seven_case=(jnp.asarray(0.2),),
            thirty_case=(jnp.asarray(0.3),),
            evaluators=evaluators,
            operations=make_tree_operations(jax, jnp, 32),
            update=_toy_update(calls),
        )
    assert calls == []


def test_three_scalar_graphs_compile_once_and_accept_new_same_shape_values():
    traces = {"daily": 0, "recurrent": 0}

    def daily(parameters, sample):
        traces["daily"] += 1
        value = jnp.sum(parameters * sample)
        return value, {"hard": {"physical": jnp.asarray(0)}}

    def recurrent(parameters, sample):
        traces["recurrent"] += 1
        value = parameters * jnp.sum(sample)
        return value, {"hard": {"physical": jnp.asarray(0)}}

    parameters = jnp.asarray(0.5, dtype=jnp.float32)
    seven = (jnp.ones((7,), dtype=jnp.float32),)
    thirty = (jnp.ones((30,), dtype=jnp.float32),)
    events = []
    evaluators, metadata = compile_gradient_evaluators(
        jax=jax,
        parameters=parameters,
        daily_objective=daily,
        daily_sample=jnp.asarray(2.0, dtype=jnp.float32),
        recurrent_objective=recurrent,
        seven_case=seven,
        thirty_case=thirty,
        memory_limit_bytes=2**40,
        emit=lambda stage, **values: events.append((stage, values)),
    )
    assert traces == {"daily": 1, "recurrent": 2}
    for offset in (1.0, 2.0):
        evaluators["daily"](parameters, jnp.asarray(offset, dtype=jnp.float32))
        evaluators["T7"](parameters, jnp.full((7,), offset, dtype=jnp.float32))
        evaluators["T30"](parameters, jnp.full((30,), offset, dtype=jnp.float32))
    assert traces == {"daily": 1, "recurrent": 2}
    assert set(metadata) == {"daily", "T7", "T30"}
    assert all(row["memory"]["estimated_live_bytes"] < 2**40 for row in metadata.values())
    assert sum(stage == "gradient_graph_compiled" for stage, _ in events) == 3


def test_parser_exposes_prepare_and_same_run_resume_modes():
    required = (
        "dataset-manifest",
        "acceptance-report",
        "base-statistics",
        "process-statistics",
        "fast-statistics",
        "prepared-manifest",
        "joint-admission-report",
        "validation-manifest",
        "output",
    )
    argv = [value for name in required for value in (f"--{name}", f"{name}.json")]
    argv += ["--prepare-only", "--resume-checkpoint", "resume.pkl"]
    args = parser().parse_args(argv)
    assert args.prepare_only is True
    assert args.resume_checkpoint.name == "resume.pkl"
