from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.probe_shared_execution import compare_trees
from scripts.hpc.probe_shared_projection_reuse import compare_component
from scripts.hpc.run_shared_learning import (
    execute_learning_update,
    make_tree_operations,
    persist_tree,
)
from scripts.hpc.shared_daily_microbatch import (
    make_daily_accumulator,
    make_microbatch_objective,
)


def _parameters():
    return {
        "weight": jnp.asarray([0.3, -0.2, 0.1], dtype=jnp.float32),
        "bias": jnp.asarray(0.04, dtype=jnp.float32),
    }


def _daily_objective(parameters, sample):
    prediction = jnp.dot(parameters["weight"], sample["features"]) + parameters["bias"]
    loss = jnp.square(prediction - sample["target"])
    return loss, {
        "process_day": 0.4 * loss,
        "flux": 0.35 * loss,
        "stock": 0.25 * loss,
        "joint_daily_contribution": loss,
        "hard": {"invalid": jnp.asarray(0, dtype=jnp.int32)},
    }


def _recurrent_objective(parameters, forcing):
    prediction = jnp.dot(parameters["weight"], forcing) + parameters["bias"]
    loss = 0.2 * jnp.square(prediction)
    return loss, {
        "flux": 0.6 * loss,
        "stock": 0.4 * loss,
        "living_stock": 0.25 * loss,
        "nonliving_stock": 0.15 * loss,
        "hard": {"invalid": jnp.asarray(0, dtype=jnp.int32)},
    }


def _batch():
    values = np.arange(96, dtype=np.float32).reshape(32, 3)
    return {
        "features": jnp.asarray(np.sin(values / 11.0), dtype=jnp.float32),
        "target": jnp.asarray(np.cos(np.arange(32) / 9.0), dtype=jnp.float32),
    }


def _roundtrip(compiled, path):
    from jax.experimental.serialize_executable import deserialize_and_load, serialize

    package = {"identity": {"test": True}, "component": path.stem, "executable": serialize(compiled)}
    persist_tree(path, jax, package)
    with path.open("rb") as handle:
        restored = pickle.load(handle)
    assert restored["identity"] == package["identity"]
    return deserialize_and_load(*restored["executable"])


def test_compare_component_checks_value_hard_counts_and_complete_gradient():
    parameters = _parameters()
    sample = {"features": jnp.asarray([0.2, -0.1, 0.4]), "target": jnp.asarray(0.3)}
    result = jax.value_and_grad(_daily_objective, has_aux=True)(parameters, sample)
    reference = {
        "value": float(result[0][0]),
        "auxiliary": {"hard": {"invalid": 0}},
        "gradient": jax.device_get(result[1]),
    }

    assert compare_component(jax, result, reference)["passed"]

    bad_hard = ((result[0][0], {**result[0][1], "hard": {"invalid": jnp.asarray(1)}}), result[1])
    assert not compare_component(jax, bad_hard, reference)["passed"]
    bad_gradient = jax.tree_util.tree_map(lambda value: value + 0.1, result[1])
    assert not compare_component(jax, (result[0], bad_gradient), reference)["passed"]


def test_compiled_executable_survives_persist_tree_roundtrip(tmp_path):
    parameters = _parameters()
    forcing = jnp.asarray([0.1, -0.3, 0.2], dtype=jnp.float32)
    compiled = jax.jit(jax.value_and_grad(_recurrent_objective, has_aux=True)).lower(parameters, forcing).compile()
    restored = _roundtrip(compiled, tmp_path / "T7.executable.pkl")

    expected = compiled(parameters, forcing)
    actual = restored(parameters, forcing)
    assert compare_trees(jax, actual, expected, rtol=0.0, atol=0.0)["passed"]


def test_serialized_components_execute_daily_mean_t7_t30_and_one_adamw(tmp_path):
    parameters = _parameters()
    batch = _batch()
    micro_objective = make_microbatch_objective(jax, jnp, _daily_objective)
    sample = jax.tree_util.tree_map(lambda value: value[:4], batch)
    seven = (jnp.asarray([0.15, -0.3, 0.25], dtype=jnp.float32),)
    thirty = (jnp.asarray([-0.2, 0.18, 0.35], dtype=jnp.float32),)

    daily_compiled = jax.jit(jax.value_and_grad(micro_objective, has_aux=True)).lower(parameters, sample).compile()
    recurrent_backward = jax.jit(jax.value_and_grad(_recurrent_objective, has_aux=True))
    seven_compiled = recurrent_backward.lower(parameters, *seven).compile()
    thirty_compiled = recurrent_backward.lower(parameters, *thirty).compile()
    loaded = {
        "daily": _roundtrip(daily_compiled, tmp_path / "daily.executable.pkl"),
        "T7": _roundtrip(seven_compiled, tmp_path / "T7.executable.pkl"),
        "T30": _roundtrip(thirty_compiled, tmp_path / "T30.executable.pkl"),
    }
    loaded["daily"] = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=4,
        evaluator=loaded["daily"],
        expected_count=32,
    )
    direct = {
        "daily": make_daily_accumulator(
            jax=jax,
            jnp=jnp,
            microbatch_size=4,
            evaluator=daily_compiled,
            expected_count=32,
        ),
        "T7": seven_compiled,
        "T30": thirty_compiled,
    }
    protocol = {
        "optimizer": {
            "learning_rate": 2e-3,
            "clip_norm": 0.05,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 1e-2,
        }
    }
    optimizer = initialize_target_oriented_adamw(parameters)
    operations = make_tree_operations(jax, jnp, 1)
    update = make_update(protocol)

    def execute(evaluators):
        return execute_learning_update(
            jax=jax,
            parameters=parameters,
            optimizer=optimizer,
            daily_samples=[batch],
            seven_case=seven,
            thirty_case=thirty,
            evaluators=evaluators,
            operations=operations,
            update=update,
        )

    expected = execute(direct)
    actual = execute(loaded)

    assert int(np.asarray(actual[1].step)) == 1
    assert compare_trees(jax, actual[0], expected[0], rtol=0.0, atol=0.0)["passed"]
    assert compare_trees(jax, actual[1], expected[1], rtol=0.0, atol=0.0)["passed"]
    np.testing.assert_allclose(
        actual[2]["loss"]["joint"],
        actual[2]["loss"]["daily_mean"] + actual[2]["loss"]["T7"]["loss"] + actual[2]["loss"]["T30"]["loss"],
        rtol=0.0,
        atol=0.0,
    )
