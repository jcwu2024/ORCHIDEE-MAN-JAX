import json
import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import initialize_target_oriented_adamw
from scripts.hpc.run_shared_transformer_admission import ROOT, run
from scripts.hpc.shared_admission_recovery import load_resume_report, restore_gradient


def objective(parameters, sample):
    value = jnp.sum((parameters["shared_transformer"] * sample - 0.5)**2)
    return value, {"hard": {"finite": jnp.asarray(0)}, "process_day": value,
                   "flux": value, "stock": value, "state_change_anchor": value}


def evidence(tmp_path):
    protocol = json.loads((ROOT / "manifests/coarse_graining/shared_transformer_admission_v1.json").read_text())
    protocol["architecture"]["parameter_count"] = 2
    parameters = {"shared_transformer": jnp.asarray([1., 2.], dtype=jnp.float32)}
    samples = [jnp.asarray(1.), jnp.asarray(2.)]
    (value, _aux), gradient = jax.value_and_grad(objective, has_aux=True)(parameters, samples[0])
    saved_path = tmp_path / "source.gradient.pkl"
    with saved_path.open("wb") as handle:
        pickle.dump(jax.device_get(dict(parameters=parameters, gradient=gradient, value=value,
                                        protocol=protocol, source_git_head="source")), handle)
    initial = [(float(objective(parameters, sample)[0]),
                {"hard": {"finite": 0}, "process_day": float(objective(parameters, sample)[0]),
                 "flux": 1., "stock": 1., "state_change_anchor": 1.}) for sample in samples]
    source = dict(protocol=protocol, completed_updates=0, gradient_artifact=str(saved_path),
                  gradient={"all_finite": True}, initial=initial, source_git_head="source",
                  calibration={"artifact_ids": [1, 2]}, scales={},
                  admission_rows=[{"artifact_id": 1}, {"artifact_id": 2}],
                  state_anchor_attribution=[], forward_device_memory={},
                  compiled_gradient_memory={"estimated_live_bytes": 1024})
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    return path, source, parameters, samples


def test_recovery_rejects_changed_loss_and_preserves_source(tmp_path):
    path, source, _parameters, _samples = evidence(tmp_path)
    with pytest.raises(ValueError, match="preserve"):
        load_resume_report(path, path, source["protocol"])
    changed = deepcopy(source["protocol"])
    changed["objective"]["group_weights"] = [0., 0.5, 0.5]
    with pytest.raises(ValueError, match="changed admission protocol"):
        load_resume_report(path, tmp_path / "result.json", changed)


def test_saved_gradient_must_match_current_tree(tmp_path):
    _path, source, parameters, _samples = evidence(tmp_path)
    shapes = jax.eval_shape(lambda: parameters)
    actual, gradient, _value = restore_gradient(jax, source, shapes)
    np.testing.assert_array_equal(actual["shared_transformer"], parameters["shared_transformer"])
    assert np.all(np.isfinite(gradient["shared_transformer"]))
    with pytest.raises(ValueError, match="shape/dtype"):
        restore_gradient(jax, source, {"shared_transformer": jax.ShapeDtypeStruct((3,), jnp.float32)})


def test_resumed_runner_matches_two_uninterrupted_updates_without_recalibration(tmp_path, monkeypatch):
    import research.daily_coarse_graining.daily_operator.shared_process_head as head_module
    import research.daily_coarse_graining.daily_operator.shared_training as training
    import research.daily_coarse_graining.daily_operator.shared_transformer as architecture
    import research.daily_coarse_graining.daily_operator.target_oriented_training as target
    import scripts.hpc.shared_transformer_runtime as runtime_module

    path, source, parameters, samples = evidence(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(source["protocol"]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": []}))
    artifacts = [SimpleNamespace(batch=SimpleNamespace(one_day=sample[None])) for sample in samples]
    runtime = {"jax": jax, "devices": jax.devices("cpu"), "probe_assembled": None, "contract": None,
               "base_statistics": SimpleNamespace(arrays={"state_delta": SimpleNamespace(scale=None)}),
               "process_statistics": SimpleNamespace(outputs={"daily.fast_gpp": SimpleNamespace(scale=None)})}
    head = SimpleNamespace(contract=None, learned_schema=SimpleNamespace(fields=[SimpleNamespace(name="x", shape=(2,))]),
                           layout=SimpleNamespace(active_pft_indices=[0, 13]),
                           shared_config=SimpleNamespace(decoder_execution="gather_reuse"),
                           initialize=lambda inputs, seed: parameters)
    monkeypatch.setattr(runtime_module, "select_artifact_ids", lambda entries, count: (1, 2))
    monkeypatch.setattr(runtime_module, "load_shared_runtime", lambda args, ids:
                        (runtime, None, None, artifacts, {"calibration_daily_rows": source["admission_rows"]}))
    def forbidden_calibration(*args, **kwargs):
        raise AssertionError("recovery must reuse completed TRAIN calibration")
    monkeypatch.setattr(runtime_module, "prepare_shared_trajectory_scales", forbidden_calibration)
    monkeypatch.setattr(head_module, "build_shared_process_head", lambda *args, **kwargs: head)
    monkeypatch.setattr(training, "make_shared_daily_objective", lambda *args, **kwargs: objective)
    monkeypatch.setattr(target, "derive_target_oriented_scales", lambda *args, **kwargs: None)
    monkeypatch.setattr(architecture, "shared_transformer_static_coordinate_indices", lambda **kwargs: {"x": [0, 1]})
    output = tmp_path / "recovered.json"
    result = run(SimpleNamespace(protocol=protocol_path, resume_report=path, output=output,
                                 prepared_manifest=manifest, shape_only=False))
    assert result["status"] == "passed"
    assert result["completed_updates"] == 2
    assert result["derivative"]["passed"]
    assert result["gradient_artifact"] == source["gradient_artifact"]
    assert not output.with_suffix(".gradient.pkl").exists()
    with Path(result["checkpoint"]).open("rb") as handle:
        actual = pickle.load(handle)
    state = initialize_target_oriented_adamw(parameters)
    update = make_update(source["protocol"])
    for sample in samples:
        (_value, _aux), gradient = jax.value_and_grad(objective, has_aux=True)(parameters, sample)
        parameters, state, _norm, _clip = update(parameters, gradient, state)
    for left, right in zip(jax.tree_util.tree_leaves((actual["parameters"], actual["optimizer"])),
                           jax.tree_util.tree_leaves((parameters, state)), strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-8)
