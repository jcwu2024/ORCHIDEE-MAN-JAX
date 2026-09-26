"""Lightweight functional tests, not deployed CUDA or scientific admission."""
from types import SimpleNamespace
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import probe_shared_ffn_precision as probe
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _ffn


def parameters(width=8):
    rng = np.random.default_rng(3)
    return tuple({name: {"kernel": jnp.asarray(rng.normal(0, 0.1, shape), dtype=jnp.float32),
                         "bias": jnp.linspace(0.01, 0.02, shape[1], dtype=jnp.float32)}
                  for name, shape in (("first", (width, width * 4)), ("second", (width * 4, width)))}
                 for _ in range(2))


def test_reference_is_existing_ffn_and_mixed_keeps_fp32_outputs():
    p = parameters()[0]
    x = jnp.arange(24, dtype=jnp.float32).reshape(3, 1, 8) * 0.03
    np.testing.assert_array_equal(probe.ffn(p, x, mixed=False), _ffn(p, x))
    got = probe.ffn(p, x, mixed=True)
    assert got.dtype == jnp.float32
    assert np.isfinite(got).all()
    assert not np.array_equal(got, _ffn(p, x))


def test_native_vjp_reaches_every_parameter_and_each_captured_input():
    p = parameters()
    x = (jnp.ones((2, 3, 1, 8), dtype=jnp.float32), jnp.ones((1, 2, 1, 8), dtype=jnp.float32) * 0.7)
    cot = tuple(jnp.ones_like(value) for value in x)
    for mixed in (False, True):
        output, (dp, dx) = jax.jit(lambda p, x, c: probe.forward_vjp(p, x, c, (0, 1), mixed=mixed))(p, x, cot)
        assert tuple(y.shape for y in output) == tuple(y.shape for y in x)
        for leaf in jax.tree_util.tree_leaves((dp, dx)):
            assert leaf.dtype == jnp.float32
            assert np.isfinite(leaf).all()
            assert np.any(np.asarray(leaf) != 0)


def test_grouping_retains_all_padding_masks_and_call_order():
    rows = [{"block": b, "input": np.full((3, 1, 384), call + b, np.float32),
             "active": np.array([True, False, False]), "call_index": call,
             "sample_index": 0, "route": "expert128"} for call in range(2) for b in range(2)]
    groups = probe.group_records(rows)
    saved = {"schema_version": "shared_real_ffn_payload_v1", "identity": {"artifact_id": 742, "daily_rows": 4},
             "parameters": jax.device_get(parameters(384)), "groups": groups}
    assert probe.validate_payload(saved) is saved
    assert len(groups) == 2
    assert groups[0]["call_indices"] == [0, 1]
    assert groups[0]["inputs"].shape == (2, 3, 1, 384)
    assert np.sum(groups[0]["active"]) == 2
    saved["groups"] = groups[:1]
    with pytest.raises(ValueError, match="missing or duplicated"):
        probe.validate_payload(saved)


def test_comparison_does_not_loosen_exact_gate_and_flags_nonfinite():
    want = np.array([1.0, 2.0], dtype=np.float32)
    assert probe.comparison(jax, want, want)["passed"]
    row = probe.comparison(jax, want + 0.001, want)
    assert not row["passed"]
    assert row["cosine"] > 0.99
    row = probe.comparison(jax, np.array([np.nan, 2], np.float32), want)
    assert not row["passed"] and row["nonfinite_actual"] == 1
    assert row["cosine"] is None
    json.dumps(row, allow_nan=False)


def test_lowering_reports_forward_and_backward_without_tensor_core_claim():
    p = parameters()
    x = (jnp.ones((1, 3, 1, 8), jnp.float32),)
    function = jax.jit(lambda p, x, cot: probe.forward_vjp(p, x, cot, (0,), mixed=True))
    lowered = function.lower(p, x, x)
    evidence = probe.lowering_evidence(lowered, lowered.compile())
    assert evidence["stablehlo_dot_count"] >= 6
    assert any("f16" in line and "f32" in line for line in evidence["stablehlo_dot_samples"])
    assert evidence["tensor_core_execution_proven"] is False


def test_capture_small_coordinate_chunk_returns_exact_output_and_restores_hooks():
    from scripts.hpc.probe_shared_dense_stages import fixture, prepare_fields, load_decoder
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    from research.daily_coarse_graining.daily_operator import shared_decoder_macro_packed as macro
    shared, data, metadata = fixture(width=8, blocks=2, microbatch=1, field_count=2, queries=6)
    data = jax.tree_util.tree_map(lambda x: x[0], data)
    decoder = load_decoder(None)
    fields = prepare_fields(decoder, shared, data, metadata)[1:]
    config = st.SharedTransformerConfig(width=8, ffn_width=32, attention_heads=2, coordinate_blocks=2, readout_width=8)
    reference = macro.reference_decode_fields(shared, fields, config)
    original = st._run_coordinate_blocks_reusing_projections
    records = []
    with probe.capture_coordinate_chunks(records, lambda *a, **k: None), jax.disable_jit():
        output = macro.reference_decode_fields(shared, fields, config)
    np.testing.assert_allclose(output[0], reference[0], rtol=1e-5, atol=1e-6)
    assert st._run_coordinate_blocks_reusing_projections is original
    assert len(records) == 2 and [r["block"] for r in records] == [0, 1]
    assert records[0]["input"].shape == (3, 1, 8)
    assert records[0]["active"].tolist() == [True, True, False]


def test_input_collision_is_rejected_before_jax_execution(tmp_path):
    source = tmp_path / "data.pkl"
    source.write_bytes(b"preserve-input")
    args = SimpleNamespace(mode="fragment", output=tmp_path / "data.pkl", payload=tmp_path / "data.pkl",
                           source_profile=None, prepared_context=None, candidate_module_file=None,
                           backbone_module_file=None, repeats=1, memory_limit_gib=1, require_gpu=False)
    with pytest.raises(ValueError, match="collision"):
        probe.run(args)
    assert source.read_bytes() == b"preserve-input"


def test_fresh_process_fragment_cli_initializes_modules_before_native_tracing(tmp_path):
    """A fresh interpreter reproduces the deployed lazy-import boundary."""
    p = jax.device_get(parameters(384))
    group = {"block": 0, "inputs": np.ones((4, 2, 1, 384), np.float32),
             "active": np.tile(np.array([[True, False]]), (4, 1)), "call_indices": [0, 1, 2, 3],
             "sample_indices": [0, 1, 2, 3], "routes": ["expert128"] * 4}
    saved = {"schema_version": "shared_real_ffn_payload_v1",
             "identity": {"artifact_id": 742, "daily_rows": 4}, "parameters": p,
             "groups": (group, dict(group, block=1))}
    payload = tmp_path / "payload.pkl"
    with payload.open("wb") as stream:
        pickle.dump(saved, stream)
    output = tmp_path / "fragment.json"
    script = Path(probe.__file__).resolve()
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1", JAX_ENABLE_X64="true")
    child = subprocess.run([sys.executable, str(script), "--mode", "fragment", "--payload", str(payload),
                            "--output", str(output), "--repeats", "1"],
                           cwd=script.parents[2], env=env, capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "measured"
    assert report["variants"]["fp32"]["comparisons"]["ffn_parameter_vjp"]["passed"]
    assert report["training_updates"] == 0
    assert report["scheduling"][0]["scheduled_shape"] == [1, 4, 2, 1, 384]


def test_micro4_schedule_matches_sequential_fp32_forward_and_native_vjp():
    p = parameters()
    rng = np.random.default_rng(7)
    values = rng.normal(size=(8, 3, 1, 8)).astype(np.float32)
    group = {"block": 0, "inputs": values, "active": np.ones((8, 3), bool),
             "call_indices": list(range(8)), "sample_indices": np.repeat(np.arange(4), 2).tolist(),
             "routes": ["expert128"] * 8, "expert_ids": [0, 1] * 4,
             "group_real_rows": [250, 3] * 4}
    sequential, _ = probe.schedule_group(group, 1)
    vectorized, metadata = probe.schedule_group(group, 4)
    order = np.asarray(metadata["record_order_indices"])
    assert vectorized.shape == (2, 4, 3, 1, 8)
    seed = rng.normal(size=values.shape).astype(np.float32)
    probe.initialize_fragment_modules()
    one = jax.jit(lambda p, x, c: probe.forward_vjp(p, (x,), (c,), (0,), mixed=False))
    seq_output, (seq_dp, seq_dx) = one(p, sequential, seed)
    vec_output, (vec_dp, vec_dx) = one(p, vectorized, seed[order])
    np.testing.assert_allclose(vec_output[0], np.asarray(seq_output[0])[order], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(vec_dx[0], np.asarray(seq_dx[0])[order], rtol=1e-5, atol=1e-6)
    assert probe.comparison(jax, vec_dp, seq_dp)["passed"]
    malformed = dict(group, expert_ids=[0, 1, 0, 1, 1, 0, 0, 1])
    with pytest.raises(ValueError, match="sequence differs"):
        probe.schedule_group(malformed, 4)


def test_algorithm_capability_fresh_cli_preserves_native_ad_and_reports_cpu_support(tmp_path):
    from scripts.hpc import probe_shared_ffn_algorithm as algorithm
    output = tmp_path / "capability.json"
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.run([sys.executable, algorithm.__file__, "--output", str(output)],
                           env=env, capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "capability_measured"
    assert report["variants"]["fp32"]["status"] == "supported"
    assert report["variants"]["cast_default"]["status"] == "supported"
    assert report["source_evidence"]["preset_available"]
    assert report["variants"]["fp32"]["result_dtypes"] == ["float32"] * 6
    for policy in ("cast_preset", "fp32_storage_preset", "fp32_storage_algorithm"):
        assert report["variants"][policy]["status"] == "unsupported_or_error"
        assert "not supported" in report["variants"][policy]["error"]
    assert report["training_updates"] == 0
