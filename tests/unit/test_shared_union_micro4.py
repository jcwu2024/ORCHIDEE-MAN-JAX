"""Lightweight contract tests; no real-model compilation or GPU execution."""

import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import probe_shared_union_micro4 as probe
from scripts.hpc.probe_shared_expert_real_daily import candidate_process_head_patches, check_artifact_output_collisions
from scripts.hpc.shared_daily_microbatch import make_microbatch_objective


def test_original_rows_are_exact_distinct_and_never_repeated():
    source = {"values": np.arange(64, dtype=np.float32).reshape(32, 2), "mask": np.ones((32, 2), dtype=bool)}
    daily4 = jax.tree_util.tree_map(lambda value: value[:4].copy(), source)
    identity = probe.validate_original_rows(jax, daily4, source)
    assert identity["row_indices"] == [0, 1, 2, 3]
    assert len(set(identity["row_sha256"])) == 4
    repeated = jax.tree_util.tree_map(lambda value: np.repeat(value[:1], 4, axis=0), daily4)
    with pytest.raises(ValueError, match="original first four"):
        probe.validate_original_rows(jax, repeated, source)
    with pytest.raises(ValueError, match="four distinct"):
        probe.validate_original_rows(
            jax, repeated, jax.tree_util.tree_map(lambda value: np.repeat(value[:1], 32, axis=0), source)
        )


def test_public_mean_full_gradient_and_counts_contract():
    def daily(parameters, sample):
        value = parameters["w"] * sample["x"]
        return jnp.square(value), {
            "hard": {"invalid": jnp.asarray(0, jnp.int32)},
            "count": jnp.asarray(1, jnp.int32),
            "prediction": value,
        }

    objective = make_microbatch_objective(jax, jnp, daily)
    parameters = {"w": jnp.asarray(2.0, jnp.float32)}
    samples = {"x": jnp.arange(1, 5, dtype=jnp.float32)}
    result = jax.jit(jax.value_and_grad(objective, has_aux=True))(parameters, samples)
    assert float(result[0][0]) == 30
    assert float(result[1]["w"]) == 30
    assert int(result[0][1]["count"]) == 4
    assert float(result[0][1]["prediction"]) == 5
    assert probe.validate_result(jax, result, parameters, count=1, leaves=1)["gradient_nonzero"] == 1
    bad = ((result[0][0], dict(result[0][1], hard={"invalid": jnp.asarray(1)})), result[1])
    with pytest.raises(probe.NumericalFailure, match="physical hard gate"):
        probe.validate_result(jax, bad, parameters, count=1, leaves=1)
    with pytest.raises(probe.NumericalFailure, match="meaningful"):
        probe.validate_result(jax, (result[0], {"w": jnp.asarray(0.0, jnp.float32)}), parameters, count=1, leaves=1)


def test_serialized_binary_checks_dynamic_tree_shape_dtype():
    from jax.experimental.serialize_executable import deserialize_and_load, serialize

    parameters = {"w": jnp.asarray(2.0, jnp.float32)}
    samples = {"x": jnp.arange(4, dtype=jnp.float32)}
    executable = jax.jit(lambda p, s: p["w"] * s["x"]).lower(parameters, samples).compile()
    serialized = serialize(executable)
    loaded = deserialize_and_load(*serialized)
    with probe.forbid_field_compile():
        np.testing.assert_array_equal(np.asarray(loaded(parameters, samples)), np.arange(4) * 2)
        with pytest.raises(RuntimeError, match="new graph executable lookup"):
            jax.jit(lambda x: x + 987).lower(jnp.zeros((7,))).compile()
    assert probe.validate_binary_inputs(jax, serialized, loaded, parameters, samples)["shapes_dtypes_match"]
    with pytest.raises(ValueError, match="treedef"):
        probe.validate_binary_inputs(jax, serialized, loaded, {"different": parameters["w"]}, samples)
    with pytest.raises(ValueError, match="shape/dtype"):
        probe.validate_binary_inputs(jax, serialized, loaded, parameters, {"x": jnp.arange(3, dtype=jnp.float32)})
    with pytest.raises(ValueError, match="shape/dtype"):
        probe.validate_binary_inputs(jax, serialized, loaded, parameters, {"x": jnp.arange(4, dtype=jnp.int32)})


def test_union_binding_is_scoped_even_on_failure():
    def original():
        return None

    def original_backbone():
        return None

    module = SimpleNamespace(
        shared_transformer_raw_outputs=original, assemble_shared_transformer_context=original_backbone
    )
    adapter = partial(
        lambda *, dynamic_projection_execution: dynamic_projection_execution, dynamic_projection_execution="union"
    )
    with pytest.raises(RuntimeError):
        with candidate_process_head_patches(module, adapter, lambda: None):
            assert module.shared_transformer_raw_outputs() == "union"
            raise RuntimeError()
    assert module.shared_transformer_raw_outputs is original
    assert module.assemble_shared_transformer_context is original_backbone


def test_rejects_row0_and_unadmitted_reference(tmp_path):
    identity = {"daily_rows": 4}
    path = tmp_path / "ref.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            {
                "identity": identity | {"diagnostic_sample": "row0_repeated_four_times_complete_sample_pytree"},
                "variant": probe.VARIANT,
                "result": (),
            },
            stream,
        )
    with pytest.raises(ValueError, match="original-four"):
        probe.load_reference(path, identity, variant=probe.VARIANT)
    report = {
        "status": "passed",
        "identity": identity,
        "variant": probe.VARIANT,
        "rtol": probe.RTOL,
        "atol": probe.ATOL,
        "all_parameter_gradient_coordinates": probe.PARAMETER_COUNT,
        "comparisons": {name: {"passed": True} for name in ("loss", "auxiliary", "all_parameter_gradients")},
        "candidate_executable": "binary.pkl",
        "candidate_artifact": "result.pkl",
    }
    assert probe.validate_admission(report, identity) == (Path("binary.pkl"), Path("result.pkl"))
    with pytest.raises(ValueError, match="strict gate"):
        probe.validate_admission(report | {"atol": 1e-4}, identity)


def test_warmup_and_three_interleaved_synchronized_pairs(monkeypatch):
    calls = []

    def call(_jax, executable, _parameters, _samples):
        calls.append(executable)
        return (), 0.25

    monkeypatch.setattr(probe, "synchronized_call", call)
    stages = []
    times = probe.matched_warm(jax, "f", "u", {}, {}, lambda stage, **kw: stages.append(stage), 5)
    assert calls == ["f", "u", "f", "u", "u", "f", "f", "u"]
    assert times == {"field": [0.25] * 3, "union": [0.25] * 3}
    assert stages[:2] == ["field_explicit_warmup", "union_explicit_warmup"]
    with pytest.raises(probe.NumericalFailure):
        probe.require_parity({"loss": {"passed": False}})


def test_actual_legacy_canonical_missing_tag_requires_admitted_path_and_original_four(tmp_path):
    identity = {
        "schema_version": "shared_expert_real_daily_context_v1",
        "artifact_id": 742,
        "daily_rows": 4,
        "jax_version": "0.4.38",
        "executable": "canonical.executable.pkl",
    }
    result = ((np.asarray(0.5, dtype=np.float32), {"hard": {"invalid": 0}}), {"w": np.ones(1, dtype=np.float32)})
    admitted_path = tmp_path / "shared-expert-real-daily-20260925.reference.pkl"
    original = {"identity": identity, "result": result}
    with admitted_path.open("wb") as stream:
        pickle.dump(original, stream)
    loaded, provenance = probe.load_reference(
        admitted_path, identity, variant="canonical_reference", legacy_canonical_path=admitted_path
    )
    assert float(loaded[0][0]) == 0.5
    assert provenance["variant_tag"] is None
    assert provenance["provenance"] == "legacy_untagged_canonical_reference_from_admission"
    assert provenance["admitted_reference_artifact"] == str(admitted_path.resolve())
    with pytest.raises(ValueError, match="original-four"):
        probe.load_reference(
            admitted_path,
            identity,
            variant="canonical_reference",
            legacy_canonical_path=tmp_path / "different-reference.pkl",
        )
    with pytest.raises(ValueError, match="original-four"):
        probe.load_reference(admitted_path, identity, variant=probe.VARIANT, legacy_canonical_path=admitted_path)
    for wrong_tag in (probe.VARIANT, "union", None):
        with admitted_path.open("wb") as stream:
            pickle.dump(original | {"variant": wrong_tag}, stream)
        with pytest.raises(ValueError, match="original-four"):
            probe.load_reference(
                admitted_path, identity, variant="canonical_reference", legacy_canonical_path=admitted_path
            )
    row0_identity = identity | {"diagnostic_sample": "row0_repeated_four_times_complete_sample_pytree"}
    with admitted_path.open("wb") as stream:
        pickle.dump(original | {"identity": row0_identity}, stream)
    with pytest.raises(ValueError, match="original-four"):
        probe.load_reference(
            admitted_path, row0_identity, variant="canonical_reference", legacy_canonical_path=admitted_path
        )


def test_preflight_and_output_preservation(tmp_path):
    import json

    path = tmp_path / "runtime.json"
    valid = {
        "status": "passed",
        "passed": True,
        "actual_backend": "gpu",
        "requested_backend": "gpu",
        "jax_version": "0.4.38",
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert probe.validate_runtime_report(path, "0.4.38")["status"] == "passed"
    with pytest.raises(ValueError, match="incompatible"):
        probe.validate_runtime_report(path, "0.10.1")
    with pytest.raises(ValueError, match="overwrite"):
        check_artifact_output_collisions((path,), (path,))
    with pytest.raises(ValueError, match="output already exists"):
        probe.run(
            SimpleNamespace(real_daily_probe_file=Path("scripts/hpc/probe_shared_expert_real_daily.py"), output=path)
        )


def test_time_budgets_are_finite_and_positive():
    import argparse

    assert probe.positive_seconds("180") == 180
    for value in ("-1", "0", "nan", "inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            probe.positive_seconds(value)
