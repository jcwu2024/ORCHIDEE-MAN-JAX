import json
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.run_shared_recurrent_admission import (
    ROOT,
    AdmissionFailure,
    Component,
    execute_bounded_joint,
    load_prerequisites,
    parser,
    sum_component_gradients,
)
from scripts.hpc.shared_transformer_runtime import select_artifact_ids


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _prerequisite_args(tmp_path):
    protocol_path = ROOT / "manifests/coarse_graining/shared_recurrent_admission_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    daily = json.loads((protocol_path.parent / protocol["daily_protocol"]).read_text(encoding="utf-8"))
    identity = {"dataset": "accepted-v6", "schedule": "fixed"}
    entries = [
        {"update": index + 10, "runtime_year": 1961 + index % 4}
        for index in range(20)
    ]
    artifact_ids = select_artifact_ids(entries, daily["calibration"]["artifact_count"])
    prepared = _write(
        tmp_path / "prepared.json", {"entries": entries, "identity": identity}
    )
    calibration_block = {
        "split": "train/train",
        "artifact_ids": artifact_ids,
        "row_count": 32,
    }
    scales = {
        "flux_scales": {"gpp": 1.0},
        "stock_change_scales": {"carbon": 2.0},
    }
    calibration = {
        "status": "passed",
        "protocol": daily,
        "calibration": calibration_block,
        "scales": scales,
        "data": {"prepared_manifest_identity": identity},
    }
    calibration_path = _write(tmp_path / "calibration.json", calibration)
    admission = {
        **calibration,
        "classification": "one_day_two_update_technical_admission_only",
        "completed_updates": 2,
        "checkpoint_inherited": False,
        "parameter_count": daily["architecture"]["parameter_count"],
        "trainable_parameter_count": daily["architecture"]["parameter_count"],
    }
    admission_path = _write(tmp_path / "daily.json", admission)
    windows = [
        {"artifact_id": artifact_ids[index // 2], "horizon": (7, 30)[index % 2]}
        for index in range(4)
    ]
    cpu = {
        "status": "passed",
        "classification": "CPU_preparation_only",
        "completed_updates": 0,
        "protocol": protocol,
        "daily_protocol": daily,
        "artifact_ids": artifact_ids[:2],
        "scales": scales,
        "data": {"prepared_manifest_identity": identity},
        "parameter_count": daily["architecture"]["parameter_count"],
        "windows": windows,
    }
    cpu_path = _write(tmp_path / "cpu.json", cpu)
    args = SimpleNamespace(
        protocol=protocol_path,
        calibration_report=calibration_path,
        daily_admission_report=admission_path,
        cpu_preparation_report=cpu_path,
        prepared_manifest=prepared,
    )
    return args, admission, cpu


def test_validate_only_prerequisites_bind_daily_calibration_data_and_cpu_evidence(tmp_path):
    args, _admission, _cpu = _prerequisite_args(tmp_path)
    actual = load_prerequisites(args)
    assert actual["protocol"]["complete_joint_updates"] == 2
    assert actual["artifact_ids"] == tuple(actual["calibration_report"]["calibration"]["artifact_ids"][:2])
    assert actual["identity"]["scales"] == actual["daily_admission_report"]["scales"]


@pytest.mark.parametrize(
    ("source", "field", "value"),
    [
        ("daily", "status", "running"),
        ("daily", "classification", "canonical_one_day_forward_only"),
        ("daily", "completed_updates", 1),
        ("daily", "checkpoint_inherited", True),
        ("cpu", "status", "running"),
        ("cpu", "artifact_ids", [999, 998]),
    ],
)
def test_rejects_bad_daily_or_cpu_prerequisite(tmp_path, source, field, value):
    args, admission, cpu = _prerequisite_args(tmp_path)
    payload = admission if source == "daily" else cpu
    payload[field] = value
    _write(args.daily_admission_report if source == "daily" else args.cpu_preparation_report, payload)
    with pytest.raises(ValueError):
        load_prerequisites(args)


def _numeric_report(_jax, tree):
    leaves = [np.asarray(value, dtype=np.float64) for value in jax.tree_util.tree_leaves(tree)]
    squared = sum(float(np.sum(value * value)) for value in leaves)
    return {
        "all_finite": all(np.all(np.isfinite(value)) for value in leaves),
        "l2_norm": float(np.sqrt(squared)),
        "element_count": sum(value.size for value in leaves),
        "leaf_count": len(leaves),
        "nonzero_leaf_count": sum(bool(np.any(value != 0.0)) for value in leaves),
        "maximum_absolute_value": max(float(np.max(np.abs(value))) for value in leaves),
    }


def _objective(weight):
    def objective(parameters, target):
        tokenizer = parameters["tokenizers"][0]
        shared = parameters["shared_transformer"][0]
        flux = weight * (tokenizer - target) ** 2
        stock = weight * (shared + 0.5 * target) ** 2
        value = flux + stock
        return value, {
            "hard": {"physical": jnp.asarray(0, dtype=jnp.int32)},
            "flux": flux,
            "stock": stock,
            "process_day": value,
        }

    return objective


def _toy_protocol():
    recurrent = json.loads(
        (ROOT / "manifests/coarse_graining/shared_recurrent_admission_v1.json").read_text()
    )
    daily = json.loads(
        (ROOT / "manifests/coarse_graining/shared_transformer_admission_v3.json").read_text()
    )
    return recurrent, daily


def _toy_batches(parameters):
    objectives = {"daily": _objective(1.0), "T7": _objective(2.0), "T30": _objective(3.0)}
    targets = ((0.1, -0.2, 0.3), (-0.4, 0.5, -0.6))
    batches = []
    replay = {}
    for artifact, row in zip((819, 982), targets, strict=True):
        batch = {
            name: Component(objectives[name], (parameters, target), name != "daily")
            for name, target in zip(("daily", "T7", "T30"), row, strict=True)
        }
        batches.append(batch)
        replay[f"artifact{artifact}.daily"] = batch["daily"]
        replay[f"artifact{artifact}.T7"] = batch["T7"]
        replay[f"artifact{artifact}.T30"] = batch["T30"]
    return batches, replay


def test_two_update_orchestration_sums_exact_gradients_at_current_parameters(tmp_path):
    recurrent, daily = _toy_protocol()
    parameters = {
        "tokenizers": jnp.asarray([0.7], dtype=jnp.float32),
        "shared_transformer": jnp.asarray([-0.2], dtype=jnp.float32),
    }
    batches, replay = _toy_batches(parameters)
    update = make_update(daily)
    events = []
    actual_parameters, actual_optimizer, rows, final = execute_bounded_joint(
        jax=jax,
        parameters=parameters,
        optimizer=initialize_target_oriented_adamw(parameters),
        update=update,
        batches=batches,
        replay=replay,
        protocol=recurrent,
        evidence_identity={"toy": True},
        output=tmp_path / "result.json",
        emit=lambda stage, **values: events.append((stage, values)),
        numeric_report=_numeric_report,
    )

    expected_parameters = parameters
    expected_optimizer = initialize_target_oriented_adamw(parameters)
    for batch in batches:
        gradients = [
            jax.grad(lambda current, item=item: item.objective(current, *item.arguments[1:])[0])(
                expected_parameters
            )
            for item in batch.values()
        ]
        expected_gradient = sum_component_gradients(jax, gradients)
        expected_parameters, expected_optimizer, _norm, _clip = update(
            expected_parameters, expected_gradient, expected_optimizer
        )

    for left, right in zip(
        jax.tree_util.tree_leaves((actual_parameters, actual_optimizer)),
        jax.tree_util.tree_leaves((expected_parameters, expected_optimizer)),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    assert len(rows) == 2
    for row in rows:
        assert row["loss_before"] == pytest.approx(sum(row["component_loss_before"].values()))
        assert set(row["gradient_groups"]) == {"tokenizers", "shared_transformer"}
        assert all(group["l2_norm"] > 0.0 for group in row["gradient_groups"].values())
    assert len(final["final"]) == 6
    gradient_events = [values for stage, values in events if stage == "component_gradient_compiled"]
    assert len(gradient_events) == 6
    assert [row["gradient_compiled_reused"] for row in gradient_events] == [False] * 3 + [True] * 3
    assert sum(stage == "component_direction_compiled" for stage, _ in events) == 2
    with (tmp_path / "result.checkpoint.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    assert checkpoint["completed_updates"] == 2


def test_resume_reuses_component_saved_before_later_gradient_failure(tmp_path):
    recurrent, daily = _toy_protocol()
    parameters = {
        "tokenizers": jnp.asarray([0.7], dtype=jnp.float32),
        "shared_transformer": jnp.asarray([-0.2], dtype=jnp.float32),
    }
    batches, replay = _toy_batches(parameters)

    def bad_gradient(parameters, target):
        del target
        token = parameters["tokenizers"][0]
        shared = parameters["shared_transformer"][0]
        value = jnp.sqrt((token - token) ** 2) + shared**2
        return value, {"hard": {"physical": jnp.asarray(0)}, "flux": value, "stock": value}

    interrupted = [dict(batch) for batch in batches]
    interrupted[0]["T7"] = Component(bad_gradient, (parameters, 0.0), True)
    output = tmp_path / "resume.json"
    with pytest.raises(AdmissionFailure, match="T7_full_parameter_gradient"):
        execute_bounded_joint(
            jax=jax,
            parameters=parameters,
            optimizer=initialize_target_oriented_adamw(parameters),
            update=make_update(daily),
            batches=interrupted,
            replay=replay,
            protocol=recurrent,
            evidence_identity={"toy": True},
            output=output,
            emit=lambda *_args, **_kwargs: None,
            numeric_report=_numeric_report,
        )
    with output.with_suffix(".gradients.pkl").open("rb") as handle:
        recovered = pickle.load(handle)
    assert tuple(recovered["components"]) == ("daily",)

    events = []
    resumed_parameters, resumed_optimizer, rows, _final = execute_bounded_joint(
        jax=jax,
        parameters=jax.device_put(recovered["parameters"]),
        optimizer=initialize_target_oriented_adamw(parameters),
        update=make_update(daily),
        batches=batches,
        replay=replay,
        protocol=recurrent,
        evidence_identity={"toy": True},
        output=output,
        emit=lambda stage, **values: events.append((stage, values)),
        numeric_report=_numeric_report,
        recovered_gradients=recovered,
    )
    first_update_compile_names = [
        values["component"]
        for stage, values in events
        if stage == "compile_component_gradient" and values["update"] == 1
    ]
    assert first_update_compile_names == ["T7", "T30"]
    assert len(rows) == 2

    clean_output = tmp_path / "clean.json"
    clean_parameters, clean_optimizer, _rows, _final = execute_bounded_joint(
        jax=jax,
        parameters=parameters,
        optimizer=initialize_target_oriented_adamw(parameters),
        update=make_update(daily),
        batches=batches,
        replay=replay,
        protocol=recurrent,
        evidence_identity={"toy": True},
        output=clean_output,
        emit=lambda *_args, **_kwargs: None,
        numeric_report=_numeric_report,
    )
    for left, right in zip(
        jax.tree_util.tree_leaves((resumed_parameters, resumed_optimizer)),
        jax.tree_util.tree_leaves((clean_parameters, clean_optimizer)),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)


def test_physical_failure_stops_before_gradient(tmp_path):
    recurrent, daily = _toy_protocol()
    parameters = {
        "tokenizers": jnp.asarray([0.7], dtype=jnp.float32),
        "shared_transformer": jnp.asarray([-0.2], dtype=jnp.float32),
    }

    def bad(parameters, target):
        value, auxiliary = _objective(1.0)(parameters, target)
        auxiliary["hard"] = {"physical": jnp.asarray(1, dtype=jnp.int32)}
        return value, auxiliary

    batches, replay = _toy_batches(parameters)
    replay["artifact819.T7"] = Component(bad, (parameters, 0.1), True)
    with pytest.raises(AdmissionFailure, match="initial_recurrent_forward_physical"):
        execute_bounded_joint(
            jax=jax,
            parameters=parameters,
            optimizer=initialize_target_oriented_adamw(parameters),
            update=make_update(daily),
            batches=batches,
            replay=replay,
            protocol=recurrent,
            evidence_identity={"toy": True},
            output=tmp_path / "bad.json",
            emit=lambda *_args, **_kwargs: None,
            numeric_report=_numeric_report,
        )


def test_nonfinite_component_gradient_is_rejected(tmp_path):
    recurrent, daily = _toy_protocol()
    parameters = {
        "tokenizers": jnp.asarray([0.7], dtype=jnp.float32),
        "shared_transformer": jnp.asarray([-0.2], dtype=jnp.float32),
    }

    def bad_gradient(parameters, target):
        del target
        token = parameters["tokenizers"][0]
        shared = parameters["shared_transformer"][0]
        value = jnp.sqrt((token - token) ** 2) + shared**2
        return value, {"hard": {"physical": jnp.asarray(0)}, "flux": value, "stock": value}

    batches, replay = _toy_batches(parameters)
    batches[0] = dict(batches[0])
    batches[0]["daily"] = Component(bad_gradient, (parameters, 0.0))
    with pytest.raises(AdmissionFailure, match="daily_full_parameter_gradient"):
        execute_bounded_joint(
            jax=jax,
            parameters=parameters,
            optimizer=initialize_target_oriented_adamw(parameters),
            update=make_update(daily),
            batches=batches,
            replay=replay,
            protocol=recurrent,
            evidence_identity={"toy": True},
            output=tmp_path / "nan.json",
            emit=lambda *_args, **_kwargs: None,
            numeric_report=_numeric_report,
        )


def test_parser_exposes_validation_and_evidence_inputs():
    required = [
        "dataset-manifest",
        "acceptance-report",
        "base-statistics",
        "process-statistics",
        "fast-statistics",
        "prepared-manifest",
        "calibration-report",
        "daily-admission-report",
        "output",
    ]
    argv = [value for name in required for value in (f"--{name}", f"{name}.json")]
    argv += ["--cpu-preparation-report", "cpu.json", "--validate-only"]
    args = parser().parse_args(argv)
    assert args.validate_only is True
    assert args.cpu_preparation_report.name == "cpu.json"
    assert args.resume_report is None
