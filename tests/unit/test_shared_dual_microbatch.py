from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


DUAL_MICROBATCH_INTEGRATION = r'''
import numpy as np
import jax
import jax.numpy as jnp

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.probe_shared_dual import (
    SharedDualExecutor,
    _serial_mixed,
    compile_evaluator_on_device,
    make_component_combiner,
    make_microbatch_daily_branch,
    make_recurrent_branch,
    make_scalar_daily_branch,
)
from scripts.hpc.run_shared_learning import execute_learning_update, make_tree_operations
from scripts.hpc.shared_daily_microbatch import (
    compile_microbatch_evaluator,
    make_microbatch_objective,
)


devices = tuple(jax.local_devices())
assert len(devices) == 2
assert all(device.platform == "cpu" for device in devices)
primary, secondary = devices
COUNT = 64
MICROBATCH = 4


def daily_objective(parameters, sample):
    prediction = jnp.dot(parameters["weight"], sample["features"]) + parameters["bias"]
    residual = prediction - sample["target"]
    loss = sample["mask"].astype(prediction.dtype) * residual * residual
    return loss, {
        "process_day": loss,
        "flux": 0.6 * loss,
        "stock": 0.4 * loss,
        "joint_daily_contribution": loss,
        "hard": {"toy_failure": sample["hard"].astype(jnp.int32)},
    }


def recurrent_objective(parameters, forcing):
    projection = jnp.dot(parameters["weight"], forcing) + parameters["bias"]
    loss = 0.2 * projection * projection
    return loss, {
        "flux": 0.6 * loss,
        "stock": 0.4 * loss,
        "living_stock": 0.25 * loss,
        "nonliving_stock": 0.15 * loss,
        "hard": {"toy_failure": jnp.asarray(0, dtype=jnp.int32)},
    }


def batch(offset):
    index = np.arange(COUNT, dtype=np.float32)
    features = np.stack(
        (
            0.2 + offset + index * 0.003,
            -0.15 + index * 0.002,
            np.full_like(index, 0.4 - offset),
        ),
        axis=1,
    )
    return {
        "features": jnp.asarray(features, dtype=jnp.float32),
        "target": jnp.asarray(0.1 + offset - index * 0.004, dtype=jnp.float32),
        "mask": jnp.asarray((index.astype(np.int32) % 5) != 2),
        "hard": jnp.zeros((COUNT,), dtype=jnp.int32),
    }


def put(tree, device):
    with jax.default_device(device):
        result = jax.device_put(tree, device)
        jax.block_until_ready(result)
        return result


def rows_from_batch(value):
    return [jax.tree_util.tree_map(lambda leaf, row=row: leaf[row], value) for row in range(COUNT)]


def assert_tree_close(left, right, *, rtol=1e-5, atol=1e-6):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(left_leaf, right_leaf, rtol=rtol, atol=atol)


parameters = put({
    "weight": jnp.asarray([0.35, -0.25, 0.18], dtype=jnp.float32),
    "bias": jnp.asarray(-0.04, dtype=jnp.float32),
}, primary)
optimizer = initialize_target_oriented_adamw(parameters)
protocol = {"optimizer": {
    "learning_rate": 2e-3,
    "clip_norm": 0.025,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 1e-2,
}}
update = make_update(protocol)
operations0 = make_tree_operations(jax, jnp, COUNT)
operations1 = make_tree_operations(jax, jnp, COUNT)
with jax.default_device(primary):
    combine = make_component_combiner(jax)

batch0 = put(batch(0.0), primary)
batch1 = put(batch(0.17), primary)
cases0_primary = {
    7: (put(jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32), primary),),
    30: (put(jnp.asarray([0.3, -0.2, 0.1], dtype=jnp.float32), primary),),
}
cases1_primary = {
    7: (put(jnp.asarray([-0.2, 0.3, 0.1], dtype=jnp.float32), primary),),
    30: (put(jnp.asarray([-0.1, 0.25, 0.2], dtype=jnp.float32), primary),),
}
cases0_secondary = put(cases0_primary, secondary)
cases1_secondary = put(cases1_primary, secondary)
peer_parameters = put(parameters, secondary)

scalar_daily_evaluator = compile_evaluator_on_device(
    jax=jax,
    objective=daily_objective,
    arguments=(parameters, rows_from_batch(batch0)[0]),
    device=primary,
)
micro_objective = make_microbatch_objective(jax, jnp, daily_objective)
micro_example = jax.tree_util.tree_map(lambda value: value[:MICROBATCH], batch0)
micro_daily_evaluator = compile_microbatch_evaluator(
    jax=jax,
    parameters=parameters,
    microbatch_objective=micro_objective,
    microbatch_example=micro_example,
)
daily_branch = make_microbatch_daily_branch(
    jax=jax,
    jnp=jnp,
    device=primary,
    evaluator=micro_daily_evaluator,
    operations=operations0,
    microbatch_size=MICROBATCH,
    expected_count=COUNT,
)
recurrent_primary = {
    horizon: compile_evaluator_on_device(
        jax=jax,
        objective=recurrent_objective,
        arguments=(parameters, *cases0_primary[horizon]),
        device=primary,
    )
    for horizon in (7, 30)
}
recurrent_secondary = {
    horizon: compile_evaluator_on_device(
        jax=jax,
        objective=recurrent_objective,
        arguments=(peer_parameters, *put(cases0_primary[horizon], secondary)),
        device=secondary,
    )
    for horizon in (7, 30)
}
recurrent_branch0 = make_recurrent_branch(
    jax=jax, device=primary, evaluators=recurrent_primary, operations=operations0
)
recurrent_branch1 = make_recurrent_branch(
    jax=jax, device=secondary, evaluators=recurrent_secondary, operations=operations1
)
serial_evaluators = {
    "daily": scalar_daily_evaluator,
    "T7": recurrent_primary[7],
    "T30": recurrent_primary[30],
}
serial_daily_branch = make_scalar_daily_branch(
    jax=jax,
    device=primary,
    evaluator=scalar_daily_evaluator,
    operations=operations0,
    expected_count=COUNT,
)


serial_parameters, serial_optimizer = parameters, optimizer
with SharedDualExecutor(
    jax=jax,
    parameters=parameters,
    optimizer=optimizer,
    daily_branch=daily_branch,
    recurrent_branch=recurrent_branch1,
    combine=combine,
    update=update,
    finite=operations0["finite"],
    devices=devices,
    require_gpus=False,
) as dual:
    for current_batch, primary_cases, secondary_cases in (
        (batch0, cases0_primary, cases0_secondary),
        (batch1, cases1_primary, cases1_secondary),
    ):
        expected_gradient, _serial_components = _serial_mixed(
            parameters=dual.parameters,
            daily_payload=rows_from_batch(current_batch),
            recurrent_payload=primary_cases,
            daily_branch=serial_daily_branch,
            recurrent_branch=recurrent_branch0,
            combine=combine,
        )
        serial_parameters, serial_optimizer, serial_report = execute_learning_update(
            jax=jax,
            parameters=serial_parameters,
            optimizer=serial_optimizer,
            daily_samples=rows_from_batch(current_batch),
            seven_case=primary_cases[7],
            thirty_case=primary_cases[30],
            evaluators=serial_evaluators,
            operations=operations0,
            update=update,
        )
        daily_result = daily_branch(dual.parameters, current_batch)
        assert set(daily_result) == {"gradient", "loss", "norm", "hard", "seconds"}
        assert all(leaf.device == primary for leaf in jax.tree_util.tree_leaves(daily_result["gradient"]))
        report, values = dual.step(
            daily_payload=current_batch,
            recurrent_payload=secondary_cases,
            persist=True,
        )
        assert_tree_close(values["parameters"], serial_parameters)
        assert_tree_close(values["optimizer"], serial_optimizer)
        assert_tree_close(values["peer_parameters"], serial_parameters)
        assert_tree_close(values["gradient"], expected_gradient)
        assert report["optimizer_step"] == int(serial_optimizer.step)
        assert report["all_hard_counts_zero"]
        np.testing.assert_allclose(report["loss"]["joint"], serial_report["loss"]["joint"], rtol=2e-5)
        assert all(leaf.device == secondary for leaf in jax.tree_util.tree_leaves(values["peer_parameters"]))

    bad_batch = dict(batch1)
    bad_batch["target"] = bad_batch["target"].at[0].set(jnp.nan)
    try:
        dual.step(daily_payload=bad_batch, recurrent_payload=cases1_secondary, persist=False)
    except ValueError as error:
        assert "daily_numerical_or_physical" in str(error)
    else:
        raise AssertionError("nonfinite daily branch was accepted")

    hard_batch = dict(batch1)
    hard_batch["hard"] = hard_batch["hard"].at[5].set(1)
    try:
        dual.step(daily_payload=hard_batch, recurrent_payload=cases1_secondary, persist=False)
    except ValueError as error:
        assert "daily_numerical_or_physical" in str(error)
    else:
        raise AssertionError("nonzero hard count was accepted")

assert int(serial_optimizer.step) == 2
print("shared dual microbatch integration passed")
'''


def test_shared_dual_microbatch_matches_serial_mean_and_two_updates():
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", DUAL_MICROBATCH_INTEGRATION],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "shared dual microbatch integration passed" in result.stdout
