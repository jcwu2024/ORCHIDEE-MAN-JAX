"""Saved graphs must remain callable as parameters and data change."""
import pickle

import numpy as np
import pytest

from scripts.hpc.probe_shared_serialized_updates import (
    compare_update_reports,
    load_evaluators,
    require_reference,
    require_same_method,
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
    assert events == ["load_saved_gradient", "saved_gradient_loaded"] * 3


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
