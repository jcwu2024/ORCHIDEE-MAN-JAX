from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import scripts.hpc.probe_shared_dual as dual_probe
import scripts.hpc.shared_learning_plan as shared_learning_plan

ROOT = Path(__file__).resolve().parents[2]


DUAL_INTEGRATION = r'''
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
    make_recurrent_branch,
    make_scalar_daily_branch,
)
from scripts.hpc.run_shared_learning import execute_learning_update, make_tree_operations


devices = tuple(jax.local_devices())
assert len(devices) == 2
assert all(device.platform == "cpu" for device in devices)
primary, secondary = devices


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
    projection = parameters["weight"] * forcing
    loss = jnp.sum(projection * projection)
    return loss, {
        "flux": 0.6 * loss,
        "stock": 0.4 * loss,
        "living_stock": 0.25 * loss,
        "nonliving_stock": 0.15 * loss,
        "hard": {"toy_failure": jnp.asarray(0, dtype=jnp.int32)},
    }


def batch(offset):
    rows = []
    for index in range(4):
        rows.append({
            "features": jnp.asarray(
                [0.2 + offset + index * 0.03, -0.15 + index * 0.02, 0.4 - offset],
                dtype=jnp.float32,
            ),
            "target": jnp.asarray(0.1 + offset - index * 0.04, dtype=jnp.float32),
            "mask": jnp.asarray(index != 2),
            "hard": jnp.asarray(0, dtype=jnp.int32),
        })
    return rows


def put(tree, device):
    with jax.default_device(device):
        result = jax.device_put(tree, device)
        jax.block_until_ready(result)
        return result


def assert_tree_close(left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(left_leaf, right_leaf, rtol=1e-5, atol=1e-6)


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
operations0 = make_tree_operations(jax, jnp, 4)
operations1 = make_tree_operations(jax, jnp, 4)
with jax.default_device(primary):
    combine = make_component_combiner(jax)

rows0 = put(batch(0.0), primary)
rows1 = put(batch(0.17), primary)
cases0_primary = {
    7: (put(jnp.zeros((3,), dtype=jnp.float32), primary),),
    30: (put(jnp.asarray([0.3, -0.2, 0.1], dtype=jnp.float32), primary),),
}
cases1_primary = {
    7: (put(jnp.zeros((3,), dtype=jnp.float32), primary),),
    30: (put(jnp.asarray([-0.1, 0.25, 0.2], dtype=jnp.float32), primary),),
}
cases0_secondary = put(cases0_primary, secondary)
cases1_secondary = put(cases1_primary, secondary)
peer_parameters = put(parameters, secondary)

daily_evaluator = compile_evaluator_on_device(
    jax=jax,
    objective=daily_objective,
    arguments=(parameters, rows0[0]),
    device=primary,
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
        arguments=(peer_parameters, *cases0_secondary[horizon]),
        device=secondary,
    )
    for horizon in (7, 30)
}
daily_branch = make_scalar_daily_branch(
    jax=jax,
    device=primary,
    evaluator=daily_evaluator,
    operations=operations0,
    expected_count=4,
)
recurrent_branch0 = make_recurrent_branch(
    jax=jax, device=primary, evaluators=recurrent_primary, operations=operations0
)
recurrent_branch1 = make_recurrent_branch(
    jax=jax, device=secondary, evaluators=recurrent_secondary, operations=operations1
)
serial_evaluators = {
    "daily": daily_evaluator,
    "T7": recurrent_primary[7],
    "T30": recurrent_primary[30],
}


def serial_step(current_parameters, current_optimizer, rows, cases):
    return execute_learning_update(
        jax=jax,
        parameters=current_parameters,
        optimizer=current_optimizer,
        daily_samples=rows,
        seven_case=cases[7],
        thirty_case=cases[30],
        evaluators=serial_evaluators,
        operations=operations0,
        update=update,
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
    # Fixed-weight timing replays must not advance either device's weights.
    replay_reference = serial_step(parameters, optimizer, rows0, cases0_primary)
    for _ in range(2):
        _, replay = dual.step(
            daily_payload=rows0, recurrent_payload=cases0_secondary, persist=False
        )
        assert_tree_close(replay["parameters"], replay_reference[0])
        assert_tree_close(replay["optimizer"], replay_reference[1])
        assert_tree_close(dual.parameters, parameters)
        assert_tree_close(dual.peer_parameters, parameters)
        assert int(dual.optimizer.step) == 0
    for rows, primary_cases, secondary_cases in (
        (rows0, cases0_primary, cases0_secondary),
        (rows1, cases1_primary, cases1_secondary),
    ):
        expected_gradient, _ = _serial_mixed(
            parameters=serial_parameters,
            daily_payload=rows,
            recurrent_payload=primary_cases,
            daily_branch=daily_branch,
            recurrent_branch=recurrent_branch0,
            combine=combine,
        )
        serial_parameters, serial_optimizer, serial_report = serial_step(
            serial_parameters, serial_optimizer, rows, primary_cases
        )
        dual_report, dual_values = dual.step(
            daily_payload=rows,
            recurrent_payload=secondary_cases,
            persist=True,
        )
        assert_tree_close(dual_values["gradient"], expected_gradient)
        assert_tree_close(dual_values["parameters"], serial_parameters)
        assert_tree_close(dual_values["optimizer"], serial_optimizer)
        assert_tree_close(dual_values["peer_parameters"], serial_parameters)
        assert dual_report["gradient_norms"]["T7"] == 0.0
        assert dual_report["clip_scale"] < 1.0
        assert dual_report["optimizer_step"] == int(serial_optimizer.step)
        assert dual_report["all_hard_counts_zero"]
        assert np.isclose(
            dual_report["loss"]["joint"], serial_report["loss"]["joint"], rtol=2e-5
        )
        assert all(
            leaf.device == secondary
            for leaf in jax.tree_util.tree_leaves(dual.peer_parameters)
        )

    bad_rows = list(rows1)
    bad_rows[0] = dict(bad_rows[0])
    bad_rows[0]["target"] = put(jnp.asarray(jnp.nan, dtype=jnp.float32), primary)
    try:
        dual.step(
            daily_payload=bad_rows,
            recurrent_payload=cases1_secondary,
            persist=False,
        )
    except ValueError as error:
        assert "daily_numerical_or_physical" in str(error)
    else:
        raise AssertionError("nonfinite daily branch was accepted")

    hard_rows = list(rows1)
    hard_rows[1] = dict(hard_rows[1])
    hard_rows[1]["hard"] = put(jnp.asarray(1, dtype=jnp.int32), primary)
    try:
        dual.step(
            daily_payload=hard_rows,
            recurrent_payload=cases1_secondary,
            persist=False,
        )
    except ValueError as error:
        assert "daily_numerical_or_physical" in str(error)
    else:
        raise AssertionError("nonzero hard count was accepted")

assert int(serial_optimizer.step) == 2
print("shared dual integration passed")
'''


def test_shared_dual_two_host_device_integration():
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", DUAL_INTEGRATION],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "shared dual integration passed" in result.stdout


def test_prepare_only_validates_checkpoint_without_loading_runtime(monkeypatch, tmp_path):
    protocol = tmp_path / "dual.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": "shared_dual_execution_v1",
                "source_update": 100,
                "artifact_ids": [742, 275],
                "daily_count": 32,
                "warmups": 1,
                "timed_replays": 3,
                "continuous_updates": 2,
                "compiled_memory_limit_gib": 24,
                "comparison_rtol": 1e-5,
                "comparison_atol": 1e-6,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        shared_learning_plan,
        "load_learning_configuration",
        lambda *_arguments: {"method": "declared"},
    )
    monkeypatch.setattr(dual_probe, "_method_identity", lambda _loaded: {"identity": 1})
    monkeypatch.setattr(
        dual_probe,
        "load_learning_checkpoint",
        lambda path, *, expected_method: {
            "completed_updates": 100,
            "path": str(path),
            "method": expected_method,
        },
    )
    output = tmp_path / "report.json"
    args = SimpleNamespace(
        benchmark_protocol=protocol,
        resume_checkpoint=tmp_path / "source.pkl",
        protocol=tmp_path / "learning.json",
        joint_admission_report=tmp_path / "admission.json",
        prepared_manifest=tmp_path / "prepared.json",
        validation_manifest=tmp_path / "validation.json",
        output=output,
        prepare_only=True,
    )

    result = dual_probe.run(args)

    assert result["status"] == "passed"
    assert result["scope"] == "checkpoint_and_metadata_only"
    assert json.loads(output.read_text(encoding="utf-8"))["source_update"] == 100
