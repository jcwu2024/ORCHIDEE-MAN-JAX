from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.run_shared_learning import execute_learning_update, make_tree_operations
from scripts.hpc.shared_daily_microbatch import (
    MicrobatchFailure,
    accumulate_daily_gradients,
    compile_microbatch_evaluator,
    make_daily_accumulator,
    make_microbatch_objective,
)


class ToyTermDiagnostics(NamedTuple):
    loss: object
    count: object
    expected_count: object
    invalid_prediction_count: object
    normalized_residual: object


class ToyHardCounts(NamedTuple):
    nonfinite_prediction: object
    total: object


class ToyScience(NamedTuple):
    total: object
    term: ToyTermDiagnostics
    hard_counts: ToyHardCounts


def _daily_objective(parameters, sample):
    prediction = jnp.dot(parameters["weight"], sample["features"]) + parameters["bias"]
    residual = prediction - sample["target"]
    mask = sample["mask"].astype(prediction.dtype)
    loss = mask * residual * residual + 0.03 * prediction * prediction
    hard = sample["hard"].astype(jnp.int32)
    count = sample["mask"].astype(jnp.float32)
    auxiliary = {
        "process_day": 0.4 * loss,
        "flux": 0.35 * loss,
        "stock": 0.25 * loss,
        "joint_daily_contribution": loss,
        "hard": {"toy_failure": hard},
        "declared_mask_change": sample["declared_mask_change"].astype(jnp.int32),
        "science": ToyScience(
            total=loss,
            term=ToyTermDiagnostics(
                loss=loss,
                count=count,
                expected_count=jnp.asarray(1.0, dtype=jnp.float32),
                invalid_prediction_count=sample["invalid_count"].astype(jnp.float32),
                normalized_residual=residual,
            ),
            hard_counts=ToyHardCounts(nonfinite_prediction=hard, total=hard),
        ),
    }
    return loss, auxiliary


def _batch(count=32):
    values = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    features = np.sin(values / 9.0).astype(np.float32)
    return {
        "features": jnp.asarray(features),
        "target": jnp.asarray(np.cos(np.arange(count) / 7.0), dtype=jnp.float32),
        "mask": jnp.asarray((np.arange(count) % 5) != 0),
        "hard": jnp.zeros((count,), dtype=jnp.int32),
        "declared_mask_change": jnp.asarray(np.arange(count) % 3 == 0, dtype=jnp.int32),
        "invalid_count": jnp.asarray(np.arange(count) % 7 == 0, dtype=jnp.int32),
    }


def _parameters():
    return {
        "weight": jnp.asarray([0.35, -0.2, 0.15], dtype=jnp.float32),
        "bias": jnp.asarray(-0.08, dtype=jnp.float32),
    }


def _tree_assert_allclose(left, right, *, rtol=2.0e-6, atol=2.0e-6):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(left_leaf, right_leaf, rtol=rtol, atol=atol)


def _serial_mean_gradient(parameters, batch):
    evaluator = jax.value_and_grad(_daily_objective, has_aux=True)
    count = int(batch["target"].shape[0])
    values = []
    gradients = []
    for row in range(count):
        sample = jax.tree_util.tree_map(lambda value: value[row], batch)
        (value, _auxiliary), gradient = evaluator(parameters, sample)
        values.append(value)
        gradients.append(gradient)
    gradient = jax.tree_util.tree_map(
        lambda *leaves: sum(leaves, jnp.zeros_like(leaves[0])) / count,
        *gradients,
    )
    return sum(values, jnp.asarray(0.0, dtype=values[0].dtype)) / count, gradient


def test_microbatch_objective_averages_values_and_sums_declared_counts():
    batch = _batch()
    batch["hard"] = batch["hard"].at[jnp.asarray([3, 14, 27])].set(1)
    objective = make_microbatch_objective(jax, jnp, _daily_objective)

    value, auxiliary = objective(_parameters(), batch)

    row_values = jax.vmap(lambda sample: _daily_objective(_parameters(), sample)[0])(batch)
    np.testing.assert_allclose(value, jnp.mean(row_values), rtol=1.0e-6)
    assert int(auxiliary["hard"]["toy_failure"]) == 3
    assert int(auxiliary["declared_mask_change"]) == 11
    assert float(auxiliary["science"].term.count) == float(jnp.sum(batch["mask"]))
    assert float(auxiliary["science"].term.expected_count) == 32.0
    assert float(auxiliary["science"].term.invalid_prediction_count) == 5.0
    assert int(auxiliary["science"].hard_counts.total) == 3
    np.testing.assert_allclose(auxiliary["joint_daily_contribution"], value, rtol=1.0e-6)


@pytest.mark.parametrize("microbatch_size", [2, 4, 8])
def test_accumulated_mean_gradient_matches_direct_and_serial(microbatch_size):
    parameters = _parameters()
    batch = _batch()
    objective = make_microbatch_objective(jax, jnp, _daily_objective)
    direct = jax.value_and_grad(objective, has_aux=True)(parameters, batch)
    example = jax.tree_util.tree_map(lambda value: value[:microbatch_size], batch)
    evaluator = compile_microbatch_evaluator(
        jax=jax,
        parameters=parameters,
        microbatch_objective=objective,
        microbatch_example=example,
    )
    accumulator = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=microbatch_size,
        evaluator=evaluator,
        expected_count=32,
    )

    accumulated = accumulator(parameters, batch)
    serial_value, serial_gradient = _serial_mean_gradient(parameters, batch)

    np.testing.assert_allclose(accumulated[0][0], direct[0][0], rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(accumulated[0][0], serial_value, rtol=2.0e-6, atol=2.0e-6)
    _tree_assert_allclose(accumulated[1], direct[1])
    _tree_assert_allclose(accumulated[1], serial_gradient)
    _tree_assert_allclose(accumulated[0][1], direct[0][1])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unequal_leaves", "unequal row counts"),
        ("wrong_declared_count", "differs from the declared mean"),
        ("unequal_chunks", "equal-sized microbatches"),
    ],
)
def test_accumulator_rejects_unequal_batches(mutation, message):
    batch = _batch()
    expected_count = 32
    microbatch_size = 4
    if mutation == "unequal_leaves":
        batch["target"] = batch["target"][:-1]
    elif mutation == "wrong_declared_count":
        expected_count = 31
    else:
        microbatch_size = 6

    with pytest.raises(ValueError, match=message):
        accumulate_daily_gradients(
            jax=jax,
            jnp=jnp,
            parameters=_parameters(),
            daily_batch=batch,
            microbatch_size=microbatch_size,
            evaluator=lambda *_arguments: pytest.fail("invalid batch reached evaluator"),
            expected_count=expected_count,
        )


def test_accumulator_checks_each_microbatch_hard_gate():
    parameters = _parameters()
    batch = _batch()
    batch["hard"] = batch["hard"].at[5].set(2)
    objective = make_microbatch_objective(jax, jnp, _daily_objective)
    example = jax.tree_util.tree_map(lambda value: value[:4], batch)
    evaluator = compile_microbatch_evaluator(
        jax=jax,
        parameters=parameters,
        microbatch_objective=objective,
        microbatch_example=example,
    )
    accumulator = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=4,
        evaluator=evaluator,
        expected_count=32,
    )

    with pytest.raises(MicrobatchFailure) as caught:
        accumulator(parameters, batch)

    assert caught.value.reason == "daily_numerical_or_physical"
    assert caught.value.microbatch_index == 1
    assert caught.value.details["row_start"] == 4
    assert caught.value.details["hard"] == {"toy_failure": 2}


def test_full_clipped_adamw_proposal_matches_serial_daily_gradient():
    parameters = _parameters()
    batch = _batch()
    objective = make_microbatch_objective(jax, jnp, _daily_objective)
    example = jax.tree_util.tree_map(lambda value: value[:4], batch)
    evaluator = compile_microbatch_evaluator(
        jax=jax,
        parameters=parameters,
        microbatch_objective=objective,
        microbatch_example=example,
    )
    accumulator = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=4,
        evaluator=evaluator,
        expected_count=32,
    )
    (_micro_value, _micro_auxiliary), micro_gradient = accumulator(parameters, batch)
    _serial_value, serial_gradient = _serial_mean_gradient(parameters, batch)
    seven_gradient = {
        "weight": jnp.asarray([0.08, -0.03, 0.05], dtype=jnp.float32),
        "bias": jnp.asarray(0.02, dtype=jnp.float32),
    }
    thirty_gradient = {
        "weight": jnp.asarray([-0.04, 0.07, 0.01], dtype=jnp.float32),
        "bias": jnp.asarray(-0.01, dtype=jnp.float32),
    }

    def combined(daily):
        return jax.tree_util.tree_map(
            lambda one, seven, thirty: one + seven + thirty,
            daily,
            seven_gradient,
            thirty_gradient,
        )

    protocol = {
        "optimizer": {
            "learning_rate": 2.0e-3,
            "clip_norm": 0.05,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 1.0e-2,
        }
    }
    update = make_update(protocol)
    state = initialize_target_oriented_adamw(parameters)
    serial_proposal = update(parameters, combined(serial_gradient), state)
    micro_proposal = update(parameters, combined(micro_gradient), state)

    _tree_assert_allclose(micro_proposal[0], serial_proposal[0], rtol=3.0e-6, atol=3.0e-7)
    _tree_assert_allclose(micro_proposal[1], serial_proposal[1], rtol=3.0e-6, atol=3.0e-7)
    np.testing.assert_allclose(micro_proposal[2], serial_proposal[2], rtol=3.0e-6)
    np.testing.assert_allclose(micro_proposal[3], serial_proposal[3], rtol=3.0e-6)
    assert float(micro_proposal[3]) < 1.0


def test_execute_learning_update_matches_serial_for_two_successive_batches():
    first_batch = _batch()
    second_batch = jax.tree_util.tree_map(lambda value: value, first_batch)
    second_batch["features"] = 0.85 * second_batch["features"] + 0.07
    second_batch["target"] = second_batch["target"] - 0.12
    parameters = _parameters()
    microbatch_objective = make_microbatch_objective(jax, jnp, _daily_objective)
    example = jax.tree_util.tree_map(lambda value: value[:4], first_batch)
    microbatch_evaluator = compile_microbatch_evaluator(
        jax=jax,
        parameters=parameters,
        microbatch_objective=microbatch_objective,
        microbatch_example=example,
    )
    accumulated_daily = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=4,
        evaluator=microbatch_evaluator,
        expected_count=32,
    )
    scalar_daily = jax.jit(jax.value_and_grad(_daily_objective, has_aux=True))

    def recurrent_objective(current_parameters, forcing):
        projection = jnp.dot(current_parameters["weight"], forcing) + current_parameters["bias"]
        loss = 0.2 * projection * projection
        return loss, {
            "flux": 0.6 * loss,
            "stock": 0.4 * loss,
            "living_stock": 0.25 * loss,
            "nonliving_stock": 0.15 * loss,
            "hard": {"toy_failure": jnp.asarray(0, dtype=jnp.int32)},
        }

    recurrent = jax.jit(jax.value_and_grad(recurrent_objective, has_aux=True))
    serial_evaluators = {"daily": scalar_daily, "T7": recurrent, "T30": recurrent}
    micro_evaluators = {"daily": accumulated_daily, "T7": recurrent, "T30": recurrent}
    serial_operations = make_tree_operations(jax, jnp, 32)
    micro_operations = make_tree_operations(jax, jnp, 1)
    protocol = {
        "optimizer": {
            "learning_rate": 2.0e-3,
            "clip_norm": 0.05,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 1.0e-2,
        }
    }
    update = make_update(protocol)
    serial_parameters = parameters
    micro_parameters = parameters
    serial_optimizer = initialize_target_oriented_adamw(parameters)
    micro_optimizer = initialize_target_oriented_adamw(parameters)
    seven_case = (jnp.asarray([0.15, -0.3, 0.25], dtype=jnp.float32),)
    thirty_case = (jnp.asarray([-0.2, 0.18, 0.35], dtype=jnp.float32),)

    for batch in (first_batch, second_batch):
        serial_rows = [
            jax.tree_util.tree_map(lambda value, row=row: value[row], batch)
            for row in range(32)
        ]
        serial_parameters, serial_optimizer, serial_report = execute_learning_update(
            jax=jax,
            parameters=serial_parameters,
            optimizer=serial_optimizer,
            daily_samples=serial_rows,
            seven_case=seven_case,
            thirty_case=thirty_case,
            evaluators=serial_evaluators,
            operations=serial_operations,
            update=update,
        )
        micro_parameters, micro_optimizer, micro_report = execute_learning_update(
            jax=jax,
            parameters=micro_parameters,
            optimizer=micro_optimizer,
            daily_samples=[batch],
            seven_case=seven_case,
            thirty_case=thirty_case,
            evaluators=micro_evaluators,
            operations=micro_operations,
            update=update,
        )

        _tree_assert_allclose(micro_parameters, serial_parameters, rtol=5.0e-6, atol=5.0e-7)
        _tree_assert_allclose(micro_optimizer, serial_optimizer, rtol=5.0e-6, atol=5.0e-7)
        np.testing.assert_allclose(
            micro_report["loss"]["joint"], serial_report["loss"]["joint"], rtol=3.0e-6
        )
        np.testing.assert_allclose(
            micro_report["optimizer_gradient_norm"],
            serial_report["optimizer_gradient_norm"],
            rtol=5.0e-6,
        )
