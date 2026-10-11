"""Canonical thin-capture method tests; no ignored scientific assets required."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import jax
import numpy as np
import pytest

from scripts.hpc import capture_shared_latent_T7_summary_operands as capture
from scripts.hpc import probe_shared_latent_T7_batch_numerics as probe
from research.daily_coarse_graining.daily_operator.successor_training import rollout_input_from_template


@pytest.fixture(scope="module")
def assets():
    return capture.synthetic_assets()


@pytest.fixture(scope="module")
def captured(assets):
    return capture.capture_inputs(assets)


def test_canonical_order_pft_refresh_and_independent_masked_summary(assets, captured):
    bundle, detail = captured
    pool = assets["layout"].state_field_pool
    owner = assets["layout"].state_element_pft_index
    # Canonical physical boundary is rebuilt from each sample's own raw state.
    for sample, template in enumerate(assets["templates"]):
        rebuilt = rollout_input_from_template(template, template.continuous_state,
            template.discrete_state, assets["assembler"].contract)
        assembled = assets["assembler"].assemble(rebuilt)
        np.testing.assert_array_equal(assembled.state.other, template.continuous_state)
        assert np.asarray(assembled.state.other).dtype == np.dtype("float64")
        assert not np.any(np.asarray(rebuilt.pft.pft_state) == -999.)
        if sample < 3:
            np.testing.assert_array_equal(rebuilt.pft.pft_state[1], 0.)
            assert rebuilt.pft.pft_fraction[1] == 0.
        raw = template.continuous_state
        # Independent host mask: -2 invalid, -1 unassigned, >=0 belongs to PFT.
        defined = np.isfinite(raw) & (np.abs(raw) < .5e20)
        active = template.pft.active_pft_mask
        selected_defined = defined & ((owner == -1) | ((owner >= 0) & active[np.maximum(owner, 0)]))
        assert not selected_defined[200]  # structurally invalid
        assert selected_defined[201]     # assigned to active PFT0
        assert selected_defined[202] == (sample == 3)
        assert not selected_defined[203]  # NaN physical state
        mean, scale = assets["normalization"].continuous_mean, assets["normalization"].continuous_scale
        values = ((raw.astype(np.float32) - mean.astype(np.float32)) / scale.astype(np.float32)).astype(np.float32)
        z = raw[435] / scale[435]
        values[435] = np.float32(16. + np.log1p(z - 16.))
        expected = []
        expected_valid = []
        for row in pool:
            members = values[row & selected_defined].astype(np.float64)
            expected_valid.append(members.size > 0)
            expected.append([np.mean(members), np.std(members), np.min(members), np.max(members)]
                            if members.size else [0., 0., 0., 0.])
        np.testing.assert_allclose(bundle["x"][sample], expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(bundle["mask"][sample], np.broadcast_to(
            np.asarray(expected_valid)[:, None], bundle["mask"][sample].shape))
    # Distinct ordered inputs retain their distinct summaries, not copied cases.
    assert len({capture.array_identity(row)["sha256_bytes_C_order"] for row in bundle["x"]}) == 4
    assert detail["no_neural_forward_or_full_model_AD"]
    assert bundle["metadata"]["origin"] == "synthetic_method_fixture"
    assert "artifact_ids" not in bundle["metadata"]


def test_raw434_direction_analytic_chain_and_exact_isolation(assets, captured):
    bundle, detail = captured
    np.testing.assert_array_equal(bundle["direction"][:3], 0.)
    assert detail["zero_other_sample_tangents"]
    assert bundle["metadata"]["direction_coordinate"] == 434
    assert bundle["metadata"]["direction_sample"] == 3
    # Coordinate434 appears only in the final padding field. Derive the mean,
    # population-sd and unique-minimum chain rule independently of JAX AD.
    raw = assets["templates"][3].continuous_state
    owner = assets["layout"].state_element_pft_index
    defined = np.isfinite(raw) & (np.abs(raw) < .5e20) & (owner != -2)
    selected = assets["layout"].state_field_pool[-1] & defined
    normalized434 = np.float32(raw[434]) / np.float32(assets["normalization"].continuous_scale[434])
    mean, sd, minimum, _ = bundle["x"][3, -1].astype(np.float64)
    assert normalized434 == minimum
    dz = 1. / float(np.float32(assets["normalization"].continuous_scale[434]))
    expected = [dz / np.count_nonzero(selected),
        (normalized434 - mean) * dz / (np.count_nonzero(selected) * sd), dz, 0.]
    np.testing.assert_allclose(bundle["direction"][3, -1], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(bundle["direction"][3, :-1], 0.)
    invalid = ~bundle["mask"].any(axis=2)
    assert np.any(invalid)
    np.testing.assert_array_equal(bundle["x"][invalid], 0.)
    np.testing.assert_array_equal(bundle["direction"][invalid], 0.)


def test_capture_receipt_seven_key_roundtrip_and_probe_admission(tmp_path, assets):
    receipt, path = capture.write_capture(tmp_path / "fixture", assets)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {*probe.ARRAY_KEYS, "metadata"}
    bundle = probe.load_operands(path)
    assert bundle["x"].shape == (4, len(assets["assembler"].contract.state_leaves), 4)
    assert bundle["kernel"].shape == (4, 384)
    assert bundle["bias"].shape == (384,)
    assert bundle["mask"].dtype == np.dtype("bool")
    assert bundle["metadata"]["capture_receipt_sha256"] == capture.sha256(tmp_path / "fixture.capture.json")
    assert bundle["metadata"]["capture_source_sha256"] == capture.sha256(capture.__file__)
    assert receipt["origin"] == "synthetic_method_fixture"
    assert receipt["current_execution"]["git_head"]
    assert "historical_source_head" not in receipt["provenance"]
    for name in probe.ARRAY_KEYS:
        assert capture.array_identity(bundle[name]) == receipt["array_identities"][name]
    assert probe.affine_oracle(bundle)["primal"].shape == (4, bundle["x"].shape[1], 384)
    with pytest.raises(FileExistsError):
        capture.write_capture(tmp_path / "fixture", assets)


def test_projection_layout_normalization_and_template_rejections(assets):
    wrong = {"tokenizer": {"summary_input": {"kernel": np.ones((4, 384), np.float64), "bias": np.zeros(384, np.float32)}}}
    with pytest.raises(ValueError, match="finite FP32"):
        capture.summary_projection(wrong)
    with pytest.raises(ValueError, match="leaf required"):
        capture.summary_projection({})
    bad = dict(assets, normalization=assets["normalization"]._replace(continuous_scale=np.ones(1)))
    with pytest.raises(ValueError, match="normalization width"):
        capture.capture_inputs(bad)
    bad = copy.deepcopy(assets)
    bad["layout"].state_element_pft_index[434] = 2
    with pytest.raises(ValueError, match="PFT-owner"):
        capture.capture_inputs(bad)
    template = assets["templates"][0]
    prepared = jax.tree_util.tree_map(lambda value: np.broadcast_to(np.asarray(value), (1, 7, *np.shape(value))), template)
    prepared = prepared._replace(day_index=np.arange(1, 8)[None, :])
    first, indices = capture.first_day_template(prepared)
    np.testing.assert_array_equal(first.continuous_state, template.continuous_state)
    np.testing.assert_array_equal(indices, np.arange(1, 8))
    with pytest.raises(ValueError, match="FP64"):
        capture.first_day_template(prepared._replace(continuous_state=prepared.continuous_state.astype(np.float32)))


def test_missing_and_wrong_identity_rejected_before_pickle(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unverified saved source must not be deserialized")
    monkeypatch.setattr(capture.pickle, "load", forbidden)
    path = tmp_path / "source.pkl"
    with pytest.raises(FileNotFoundError, match="required saved source unavailable"):
        capture.read_verified_pickle(path, probe.PARAMETERS_SHA256)
    path.write_bytes(b"not an admitted scientific source")
    with pytest.raises(ValueError, match="identity differs"):
        capture.read_verified_pickle(path, probe.PARAMETERS_SHA256)
    paths = capture.AssetPaths(path, path, path, path, path)
    with pytest.raises(ValueError, match="identity differs"):
        capture.load_real_assets(paths)


@pytest.mark.parametrize("name,index,value", [
    ("x", (0, 0, 0), np.nan),
    ("direction", (3, 0, 0), np.nan),
    ("direction", (0, 0, 0), 1.),
])
def test_failed_operand_admission_preserves_full_raw_arrays(tmp_path, assets, captured, monkeypatch, name, index, value):
    bundle, detail = copy.deepcopy(captured)
    bundle[name][index] = value
    monkeypatch.setattr(capture, "capture_inputs", lambda unused: (bundle, detail))
    with pytest.raises(ValueError, match="failure preserved"):
        capture.write_capture(tmp_path / "failure", assets)
    receipt = json.loads((tmp_path / "failure.capture.json").read_text())
    assert receipt["status"] == "operand_admission_failed"
    assert not (tmp_path / "failure.operands.npz").exists()
    with np.load(tmp_path / "failure.failed-arrays.npz", allow_pickle=False) as preserved:
        assert set(preserved.files) == {*probe.ARRAY_KEYS, "metadata"}
        if np.isnan(value):
            assert np.isnan(preserved[name][index])
        else:
            assert preserved[name][index] == value
        for name in probe.ARRAY_KEYS:
            np.testing.assert_array_equal(preserved[name], bundle[name])


def test_cpu_platform_forced_before_jax_import_under_cuda_environment():
    environment = dict(os.environ, JAX_PLATFORMS="cuda", PYTHONDONTWRITEBYTECODE="1")
    code = "from scripts.hpc import capture_shared_latent_T7_summary_operands as c; import json,os; print(json.dumps({'requested':os.environ['JAX_PLATFORMS'],'platforms':[d.platform for d in c.jax.devices()]}))"
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=capture.ROOT, env=environment,
                            text=True, capture_output=True, timeout=30, check=True)
    assert json.loads(result.stdout.strip()) == {"requested": "cpu", "platforms": ["cpu"]}


def test_historical_identity_order_and_heldout_gates():
    # These are pure validator fixtures; they are never admitted as real files
    # or emitted with captured_component_operands provenance.
    method = {"training_plan": {"training_artifact_ids": probe.ARTIFACT_IDS,
        "prepared_manifest_identity": capture.PREPARED_IDENTITY}, "configuration": {"recurrent_window_index": 0}}
    saved = {"identity": {"source_head": capture.HISTORICAL_HEAD, "artifact_id": 742,
        "candidate_source_sha256": capture.HISTORICAL_OVERLAYS, "input_source_method": method}}
    context = {"identity": {"schema_version": "shared_expert_real_daily_context_v1",
        "source_git_head": capture.HISTORICAL_HEAD, "method": method, "artifact_id": 742,
        "daily_rows": 4, "initial_checkpoint": "historical-checkpoint"}}
    manifest = {"original_checkpoint": "historical-checkpoint"}
    assert capture.validate_historical_method(saved, context, manifest) is method
    bad = copy.deepcopy(saved)
    bad["identity"]["input_source_method"]["training_plan"]["training_artifact_ids"] = [299, 742, 275, 482]
    with pytest.raises(ValueError):
        capture.validate_historical_method(bad, context, manifest)
    bad = copy.deepcopy(saved)
    bad["identity"]["candidate_source_sha256"]["shared_transformer.py"] = "wrong"
    with pytest.raises(ValueError, match="overlay"):
        capture.validate_historical_method(bad, context, manifest)
    record = {"landpoint_id": "fixture", "year": 1961, "parent_shard_sha256": "parent", "typed_shard_sha256": "typed"}
    shard = {"landpoint_id": "fixture", "year": 1961, "spatial_split": "train", "temporal_split": "test",
        "day_count": 365, "shard_sha256": "parent", "typed_shard_sha256": "typed"}
    with pytest.raises(ValueError, match="held-out"):
        capture._training_provenance({"shards": [shard]}, record, 0, 1961)
    shard["temporal_split"] = "train"
    with pytest.raises(ValueError, match="range"):
        capture._training_provenance({"shards": [shard]}, record, 360, 1961)
    assert capture._training_provenance({"shards": [shard]}, record, 0, 1961) is shard
