import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.run_shared_learning import (
    _method_identity,
    compile_gradient_evaluators,
    execute_learning_update,
    load_learning_checkpoint,
    make_tree_operations,
    parser,
    run,
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


def test_technical_mode_rejects_v1_prepare_and_resume_before_runtime(tmp_path, monkeypatch):
    import scripts.hpc.shared_learning_plan as learning_plan

    required = (
        "dataset-manifest", "acceptance-report", "base-statistics",
        "process-statistics", "fast-statistics", "prepared-manifest",
        "joint-admission-report", "validation-manifest", "output",
    )
    argv = [value for name in required for value in (f"--{name}", str(tmp_path / "x.json"))]
    with pytest.raises(SystemExit):
        parser().parse_args([*argv, "--technical-updates", "1"])

    config = {"schema_version": "shared_learning_v1"}
    monkeypatch.setattr(learning_plan, "load_learning_configuration", lambda *_: {"config": config})
    args = parser().parse_args([*argv, "--technical-updates", "2"])
    with pytest.raises(ValueError, match="requires the shared learning v2"):
        run(args)

    config["schema_version"] = "shared_learning_v2"
    args.resume_checkpoint = tmp_path / "old.pkl"
    with pytest.raises(ValueError, match="fresh execution"):
        run(args)
    args.resume_checkpoint = None
    args.prepare_only = True
    with pytest.raises(ValueError, match="fresh execution"):
        run(args)


def test_technical_mode_runs_two_changing_updates_without_monitors(tmp_path, monkeypatch):
    import research.daily_coarse_graining.daily_operator.shared_persistent_outputs as persistent
    import research.daily_coarse_graining.daily_operator.shared_process_head as process_head
    import research.daily_coarse_graining.daily_operator.shared_recurrent_training as recurrent
    import research.daily_coarse_graining.daily_operator.shared_training as shared_training
    import scripts.hpc.run_shared_learning as learning_runner
    import scripts.hpc.shared_learning_monitor as learning_monitor
    import scripts.hpc.shared_learning_plan as learning_plan
    import scripts.hpc.shared_transformer_runtime as shared_runtime

    class Inputs(NamedTuple):
        continuous_state: object

    class OneDay(NamedTuple):
        inputs: Inputs

    def artifact(values, seven, thirty):
        return SimpleNamespace(batch=SimpleNamespace(
            one_day=OneDay(Inputs(jnp.asarray(values, dtype=jnp.float32))),
            seven_day=jnp.asarray(seven, dtype=jnp.float32),
            thirty_day=jnp.asarray(thirty, dtype=jnp.float32),
        ))

    first = artifact(np.linspace(-1.0, 0.4, 32), 0.2, -0.3)
    second = artifact(np.linspace(0.7, -0.5, 32), -0.2, 0.5)
    identity = {"pool": "toy"}
    optimizer = {
        "name": "AdamW", "learning_rate": 2e-3, "clip_norm": 0.05,
        "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
        "weight_decay": 0.0,
    }
    daily_protocol = {
        "seed": 42,
        "architecture": {"parameter_count": 2, "thermal_transform": "toy"},
        "optimizer": optimizer,
    }
    config = {
        "schema_version": "shared_learning_v2", "max_updates": 500,
        "daily_microbatch_size": 4, "daily_rows_per_update": 32,
        "recurrent_window_index": 0, "gradient_memory_limit_gib": 24,
        "milestones": [0, 100, 250, 500], "checkpoint_interval": 25,
        "checkpoint_selection": "declared_final_update_500_no_best_panel_selection",
    }
    loaded = {
        "config": config,
        "daily_protocol": daily_protocol,
        "admission": {"protocol": {"joint": "toy"}, "scales": {"scale": 1}},
        "plan": {"training_artifact_ids": [11, 12], "fit_artifact_ids": [11],
                 "prepared_manifest_identity": identity},
        "validation_panel": {"panel_sha256": "toy-panel", "rows": []},
    }
    monkeypatch.setattr(learning_plan, "load_learning_configuration", lambda *_: loaded)
    monkeypatch.setattr(learning_runner, "shared_config", lambda architecture: architecture)
    monkeypatch.setattr(persistent, "build_shared_persistent_coordinates", lambda *_: None)
    monkeypatch.setattr(process_head, "build_shared_process_head", lambda *_args, **_kwargs: SimpleNamespace(
        initialize=lambda _assembled, *, seed: {
            "weight": jnp.asarray(0.45, dtype=jnp.float32),
            "bias": jnp.asarray(-0.12, dtype=jnp.float32),
        }
    ))
    monkeypatch.setattr(shared_training, "derive_shared_state_change_scales", lambda *_: None)
    monkeypatch.setattr(
        recurrent, "make_shared_joint_daily_objective",
        lambda *_: lambda parameters, sample: _daily(parameters, sample.inputs.continuous_state),
    )
    monkeypatch.setattr(recurrent, "make_shared_recurrent_objective", lambda *_: (None, _recurrent))
    monkeypatch.setattr(recurrent, "prepare_shared_window", lambda _head, value, **_kwargs: (value,))
    monkeypatch.setattr(
        learning_monitor, "build_shared_learning_monitor",
        lambda *_args, **_kwargs: pytest.fail("technical mode constructed a science monitor"),
    )

    class Device:
        def memory_stats(self):
            return {"peak_bytes_in_use": 2048}

    runtime = {
        "jax": jax, "base_statistics": SimpleNamespace(arrays={"state_delta": None}),
        "probe_assembled": None, "devices": [Device()], "index": None,
    }
    monkeypatch.setattr(
        shared_runtime, "load_shared_runtime",
        lambda _args, _ids: (runtime, SimpleNamespace(learned_schema=None, contract=None),
                             None, [first], {}),
    )
    monkeypatch.setattr(
        shared_runtime, "_load_prepared_artifacts",
        lambda _path, _ids: ([second], {"identity": identity}),
    )
    monkeypatch.setattr(shared_runtime, "validate_shared_artifact_metadata", lambda *_: None)

    args = Namespace(
        output=tmp_path / "technical.json", protocol=tmp_path / "protocol.json",
        joint_admission_report=tmp_path / "joint.json",
        prepared_manifest=tmp_path / "prepared.json",
        validation_manifest=tmp_path / "validation.json",
        prepare_only=False, resume_checkpoint=None, technical_updates=2,
    )
    result = run(args)
    assert result["status"] == "passed"
    assert result["classification"] == "shared_split_two_update_technical_timing_only"
    assert result["completed_updates"] == 2
    assert [row["artifact_id"] for row in result["updates"]] == [11, 12]
    assert result["monitors"] == []
    assert result["science_monitors_executed"] is False
    assert result["formal_learning_completed"] is False
    assert result["declared_formal_max_updates"] == 500
    assert result["device_peak_bytes_in_use"] == 2048
    assert set(result["compiled_gradient_graphs"]) == {"daily", "T7", "T30"}
    assert result["first_update_timing_seconds"] == result["updates"][0]["timing_seconds"]
    assert result["warm_second_update_timing_seconds"] == result["updates"][1]["timing_seconds"]
    assert int(load_learning_checkpoint(
        Path(result["final_checkpoint"]), expected_method=result["method"]
    )["optimizer"].step) == 2
    with pytest.raises(ValueError, match="method, data, or cursor"):
        load_learning_checkpoint(
            Path(result["final_checkpoint"]), expected_method=_method_identity(loaded)
        )
