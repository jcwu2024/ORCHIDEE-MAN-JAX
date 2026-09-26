"""Saved graphs must remain callable as parameters and data change."""
import pickle
import json

import numpy as np
import pytest

from scripts.hpc.probe_shared_serialized_updates import (
    compare_update_reports,
    load_evaluators,
    require_reference,
    require_same_method,
    admission_report_paths,
    validate_admissions,
)


def test_serialized_reverse_preserves_two_updates_and_changed_data(tmp_path):
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load, serialize

    def objective(parameters, data):
        residual = parameters["w"] * data["x"] - data["y"]
        loss = jnp.mean(residual**2)
        return loss, {"hard": {"invalid": jnp.array(0)}, "mean_residual": jnp.mean(residual)}

    initial = {"w": jnp.array([0.5, -0.3], dtype=jnp.float32)}
    samples = [
        {"x": jnp.array([1., 2.]), "y": jnp.array([2., -1.])},
        {"x": jnp.array([-3., 0.5]), "y": jnp.array([0.3, 4.])},
    ]
    compiled = jax.jit(jax.value_and_grad(objective, has_aux=True)).lower(initial, samples[0]).compile()
    source = {}
    for name in ("daily", "T7", "T30"):
        path = tmp_path / f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(serialize(compiled), handle)
        source[name] = {"executable": str(path)}
    events = []
    loaded, metadata = load_evaluators(
        {"executables": source}, deserialize=deserialize_and_load,
        memory_sizes=lambda _: {"estimated_live_bytes": 1}, memory_limit_bytes=2,
        emit=lambda stage, **values: events.append(stage),
    )
    current = initial
    for data in samples:
        expected = compiled(current, data)
        for executable in loaded.values():
            actual = executable(current, data)
            for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        current = jax.tree_util.tree_map(lambda p, g: p - 0.01*g, current, expected[1])
    assert set(metadata) == {"daily", "T7", "T30"}
    assert events == [
        "load_saved_gradient", "deserialize_saved_gradient", "saved_gradient_loaded"
    ] * 3


def test_method_and_source_guard_rejects_incompatible_saved_graphs():
    reference = {"method": {"loss": "original", "seed": 8}, "source_git_head": "original"}
    require_same_method(dict(reference), reference)
    with pytest.raises(ValueError, match="method/data"):
        require_same_method({**reference, "method": {"loss": "changed"}}, reference)
    with pytest.raises(ValueError, match="source checkout"):
        require_same_method({**reference, "source_git_head": "changed"}, reference)


def test_incomplete_profile_not_treated_as_reusable():
    profile = {"status": "passed", "executables": {name: {} for name in ("daily", "T7", "T30")}}
    reference = {"status": "passed", "completed_updates": 2, "technical_updates": 2}
    require_reference(profile, reference)
    with pytest.raises(ValueError, match="three-executable"):
        require_reference({**profile, "executables": {"daily": {}}}, reference)
    with pytest.raises(ValueError, match="two-update"):
        require_reference(profile, {**reference, "completed_updates": 1})


def test_report_scalar_dtypes_are_not_checkpoint_tensor_dtypes():
    import jax

    row = dict(update=1, artifact_id=742, loss={"joint": 0.25}, gradient_norms={"combined": 2.},
               optimizer_gradient_norm=2., clip_scale=0.5)
    reference = {"updates": [row]}
    current = {"updates": [{**row, "optimizer_gradient_norm": np.float32(2.), "clip_scale": np.float32(0.5)}]}
    assert compare_update_reports(jax, current, reference, rtol=1e-5, atol=1e-6)["passed"]
    current["updates"][0]["optimizer_gradient_norm"] = np.float32(3.)
    assert not compare_update_reports(jax, current, reference, rtol=1e-5, atol=1e-6)["passed"]


def test_combined_admissions_preserve_original_profile_identity_and_gate_every_component(tmp_path):
    from types import SimpleNamespace

    reference = {"source_git_head": "original", "method": {"training_plan": {
        "training_artifact_ids": [742, 275]}}, "checkpoints": [{
            "update": 0, "retained_milestone": True, "path": str(tmp_path / "initial.pkl")}]}
    profile = {"executables": {name: {"executable": str(tmp_path / f"original-{name}.pkl")}
                               for name in ("daily", "T7", "T30")}}
    identity = {"schema_version": "shared_expert_real_daily_context_v1", "source_git_head": "original",
                "method": reference["method"], "initial_checkpoint": reference["checkpoints"][0]["path"],
                "artifact_id": 742, "daily_rows": 4, "jax_version": "0.4.38"}
    comparison = {"passed": True, "failed": 0, "nonfinite_actual": 0, "nonfinite_reference": 0,
                  "coordinates": 64884481}
    paths, reports = {}, {}
    for name in ("daily", "T7", "T30"):
        binary = tmp_path / f"candidate-{name}.pkl"
        with binary.open("wb") as stream:
            pickle.dump((name,), stream)
        current_identity = {**identity, "executable": profile["executables"][name]["executable"]}
        report = {"status": "passed", "scientific_acceptance": False, "rtol": 1e-5, "atol": 1e-6,
                  "all_parameter_gradient_coordinates": 64884481,
                  "comparisons": {key: dict(comparison) for key in ("loss", "auxiliary", "all_parameter_gradients")},
                  "reference_hard": {"failure": 0}, "candidate_hard": {"failure": 0},
                  "candidate_executable": str(binary)}
        if name == "daily":
            report.update(scope="actual_canonical_daily_micro4_only_no_T7_T30_no_optimizer",
                          variant="expert128+backbone_source_reuse", identity=current_identity)
        else:
            horizon = int(name[1:])
            artifact = tmp_path / f"result-{name}.pkl"
            with artifact.open("wb") as stream:
                pickle.dump({"identity": {**current_identity, "horizon": horizon,
                                          "source_profile_sha256": "original-digest"}}, stream)
            report.update(scope="actual_canonical_recurrent_single_horizon_no_daily_no_optimizer",
                          horizon=horizon, artifact_id=742, candidate_artifact=str(artifact))
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(report), encoding="utf-8")
        reports[name] = report
    original_profile = json.loads(json.dumps(profile))
    admitted = validate_admissions(profile, reference, paths, profile_sha256="original-digest")
    assert profile == original_profile
    assert admitted["T30"]["original_executable"] == profile["executables"]["T30"]["executable"]
    loaded, metadata = load_evaluators(profile, deserialize=lambda name: name, overrides=admitted,
                                      memory_sizes=lambda _: {"estimated_live_bytes": 1},
                                      memory_limit_bytes=2, emit=lambda *_a, **_kw: None)
    assert loaded == {name: name for name in paths}
    assert metadata["daily"]["admission"] == admitted["daily"]
    with pytest.raises(ValueError, match="profile/horizon"):
        validate_admissions(profile, reference, paths, profile_sha256="replacement-digest")
    reports["daily"]["identity"]["method"] = {"replacement": True}
    paths["daily"].write_text(json.dumps(reports["daily"]), encoding="utf-8")
    with pytest.raises(ValueError, match="original method"):
        validate_admissions(profile, reference, paths, profile_sha256="original-digest")
    reports["daily"]["identity"]["method"] = reference["method"]
    reports["daily"]["comparisons"]["all_parameter_gradients"]["failed"] = 1
    paths["daily"].write_text(json.dumps(reports["daily"]), encoding="utf-8")
    with pytest.raises(ValueError, match="gradient/physical"):
        validate_admissions(profile, reference, paths, profile_sha256="original-digest")
    with pytest.raises(ValueError, match="all three"):
        admission_report_paths(SimpleNamespace(daily_admission_report=paths["daily"]))
