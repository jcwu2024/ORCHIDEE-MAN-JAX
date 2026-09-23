import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.run_shared_learning import (
    compile_gradient_evaluators,
    execute_learning_update,
    load_learning_checkpoint,
    make_tree_operations,
    save_learning_checkpoint,
)
from scripts.hpc.shared_daily_microbatch import make_daily_accumulator
from scripts.hpc.shared_learning_plan import build_learning_plan, load_learning_configuration

ROOT = Path(__file__).resolve().parents[2]


def _daily(parameters, sample):
    prediction = parameters["weight"] * sample + parameters["bias"]
    value = (prediction - 0.3) ** 2 + 0.2 * prediction ** 2
    return value, {
        "process_day": value,
        "flux": value * 0.5,
        "stock": value * 0.25,
        "joint_daily_contribution": value,
        "hard": {"physical": jnp.asarray(0, dtype=jnp.int32)},
    }


def _recurrent(parameters, sample):
    prediction = parameters["weight"] * sample + parameters["bias"]
    value = 0.1 * (prediction + 0.4) ** 2
    return value, {
        "flux": value,
        "stock": value * 0.5,
        "living_stock": value * 0.25,
        "nonliving_stock": value * 0.25,
        "hard": {"physical": jnp.asarray(0, dtype=jnp.int32)},
    }


def _assert_tree_close(actual, expected):
    for left, right in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=5e-6, atol=5e-7)


def test_split_protocol_requires_declared_microbatch_before_runtime_load(tmp_path):
    source = json.loads(
        (ROOT / "manifests/coarse_graining/shared_split_learning_v1.json").read_text()
    )
    for mutation in (None, 2):
        config = dict(source)
        if mutation is None:
            del config["daily_microbatch_size"]
        else:
            config["daily_microbatch_size"] = mutation
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="four-row daily microbatches"):
            load_learning_configuration(path, tmp_path / "not-loaded.json", path, path)

    manifest = {
        "identity": {"pool": "train"},
        "entries": [{"update": i, "runtime_year": 1961 + i % 44} for i in range(520)],
    }
    plan = build_learning_plan(manifest, source)
    assert len(plan["training_artifact_ids"]) == 500
    assert len(set(plan["training_artifact_ids"])) == 500
    assert plan["summary"]["daily_draws"] == 16000


def test_split_configuration_loads_only_matching_admitted_method(tmp_path, monkeypatch):
    import scripts.hpc.fresh_method_validation_pool as validation_pool

    class ToyPool:
        def __init__(self, _path):
            self.rows = [{"row": row} for row in range(48)]
            self.panel = {"windows_per_slice_per_horizon": 8, "panel_sha256": "toy-panel"}

    monkeypatch.setattr(validation_pool, "FreshMethodValidationPool", ToyPool)
    configuration = json.loads(
        (ROOT / "manifests/coarse_graining/shared_split_learning_v1.json").read_text()
    )
    daily = json.loads(
        (ROOT / "manifests/coarse_graining/shared_transformer_split_width_v1.json").read_text()
    )
    joint = json.loads(
        (ROOT / "manifests/coarse_graining/shared_split_recurrent_admission_v1.json").read_text()
    )
    for filename, data in (
        ("learning.json", configuration),
        (configuration["model_protocol"], daily),
        (configuration["joint_protocol"], joint),
    ):
        (tmp_path / filename).write_text(json.dumps(data))
    manifest = {
        "identity": {"pool": "train"},
        "entries": [{"update": i, "runtime_year": 1961 + i % 44} for i in range(520)],
    }
    manifest_path = tmp_path / "prepared.json"
    manifest_path.write_text(json.dumps(manifest))
    admission = {
        "status": "passed",
        "classification": "shared_daily_T7_T30_two_update_technical_admission_only",
        "completed_updates": 2,
        "protocol": joint,
        "daily_protocol": daily,
        "checkpoint_inherited": False,
        "trainable_parameter_count": daily["architecture"]["parameter_count"],
        "evidence_identity": {"prepared_manifest_identity": manifest["identity"]},
    }
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps(admission))
    loaded = load_learning_configuration(
        tmp_path / "learning.json", admission_path, manifest_path, tmp_path / "panel.json"
    )
    assert loaded["config"]["schema_version"] == "shared_learning_v2"
    assert loaded["plan"]["summary"]["daily_draws"] == 16000

    configuration["model_protocol"] = "shared_transformer_admission_v3.json"
    (tmp_path / "learning.json").write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="declared split method"):
        load_learning_configuration(
            tmp_path / "learning.json", admission_path, manifest_path, tmp_path / "panel.json"
        )


def test_micro4_runner_matches_scalar_updates_and_resume(tmp_path):
    parameters = {
        "weight": jnp.asarray(0.45, dtype=jnp.float32),
        "bias": jnp.asarray(-0.12, dtype=jnp.float32),
    }
    first = jnp.asarray(np.linspace(-0.9, 0.8, 32), dtype=jnp.float32)
    second = jnp.asarray(np.linspace(0.7, -0.5, 32), dtype=jnp.float32)
    seven = (jnp.asarray(0.25, dtype=jnp.float32),)
    thirty = (jnp.asarray(-0.4, dtype=jnp.float32),)
    events = []
    micro_evaluators, metadata = compile_gradient_evaluators(
        jax=jax,
        parameters=parameters,
        daily_objective=_daily,
        daily_sample=first[:4],
        recurrent_objective=_recurrent,
        seven_case=seven,
        thirty_case=thirty,
        memory_limit_bytes=2**40,
        emit=lambda stage, **details: events.append((stage, details)),
        daily_microbatch_size=4,
    )
    assert set(metadata) == {"daily", "T7", "T30"}
    assert sum(stage == "gradient_graph_compiled" for stage, _ in events) == 3
    accumulator = make_daily_accumulator(
        jax=jax, jnp=jnp, microbatch_size=4,
        evaluator=micro_evaluators["daily"], expected_count=32,
    )
    scalar_evaluators = {
        **micro_evaluators,
        "daily": jax.jit(jax.value_and_grad(_daily, has_aux=True)),
    }
    protocol = {
        "optimizer": {
            "name": "AdamW", "learning_rate": 2e-3, "clip_norm": 0.05,
            "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
            "weight_decay": 1e-2,
        }
    }
    update = make_update(protocol)
    serial_parameters = micro_parameters = parameters
    serial_optimizer = micro_optimizer = initialize_target_oriented_adamw(parameters)
    method = {"configuration": {"max_updates": 500}, "identity": "split-toy"}
    for index, batch in enumerate((first, second), start=1):
        rows = [batch[row] for row in range(32)]
        serial_parameters, serial_optimizer, serial_row = execute_learning_update(
            jax=jax, parameters=serial_parameters, optimizer=serial_optimizer,
            daily_samples=rows, seven_case=seven, thirty_case=thirty,
            evaluators=scalar_evaluators, operations=make_tree_operations(jax, jnp, 32),
            update=update,
        )
        micro_parameters, micro_optimizer, micro_row = execute_learning_update(
            jax=jax, parameters=micro_parameters, optimizer=micro_optimizer,
            daily_samples=batch, seven_case=seven, thirty_case=thirty,
            evaluators=micro_evaluators, operations=make_tree_operations(jax, jnp, 32),
            update=update, daily_accumulator=accumulator,
        )
        _assert_tree_close(micro_parameters, serial_parameters)
        _assert_tree_close(micro_optimizer, serial_optimizer)
        assert micro_row["loss"]["joint"] == pytest.approx(serial_row["loss"]["joint"], rel=5e-6)
        assert "daily_32_microbatch_4" in micro_row["timing_seconds"]
        assert "daily_32_sequential" not in micro_row["timing_seconds"]
        if index == 1:
            checkpoint = tmp_path / "split.checkpoint.pkl"
            save_learning_checkpoint(
                checkpoint, jax=jax, parameters=micro_parameters, optimizer=micro_optimizer,
                completed_updates=1, method=method, updates=[{"update": 1, **micro_row}],
                monitors=[],
            )

    saved = load_learning_checkpoint(checkpoint, expected_method=method)
    resumed_parameters, resumed_optimizer, _ = execute_learning_update(
        jax=jax, parameters=jax.device_put(saved["parameters"]),
        optimizer=jax.device_put(saved["optimizer"]), daily_samples=second,
        seven_case=seven, thirty_case=thirty, evaluators=micro_evaluators,
        operations=make_tree_operations(jax, jnp, 32), update=update,
        daily_accumulator=accumulator,
    )
    _assert_tree_close(resumed_parameters, micro_parameters)
    _assert_tree_close(resumed_optimizer, micro_optimizer)
    assert int(np.asarray(resumed_optimizer.step)) == 2
    with pytest.raises(ValueError, match="method, data, or cursor"):
        load_learning_checkpoint(
            checkpoint,
            expected_method={"configuration": {"max_updates": 500}, "identity": "other"},
        )
