"""Fail closed on horizon/schema changes without constructing a JAX graph."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hpc import probe_shared_latent_reader_T7_pair as probe


def manifest(tmp_path, horizon):
    artifact = tmp_path / "asset.json"
    artifact.write_text("{}")
    value = {
        "schema_version": ("shared_corrected_latent_reader_T7_pair_manifest_v1" if horizon == 7
                           else "shared_corrected_latent_reader_recurrent_pair_manifest_v1"),
        **{key: str(artifact) for key in ("source_profile", "input_bundle", "original_checkpoint",
                                        "candidate_parameters", "historical_original_admission")},
        "prepared_artifact_index": 0, "horizon": horizon, "native_AD_argnums": [0, 1],
        "precision": "FP32_neural_FP64_physics", "turnover_base_sha256": probe.BASE_SHA,
        "turnover_sha256": probe.CORRECTED_SHA, "training_updates": 0,
        "historical_original_is_corrected_pair_reference": False,
    }
    value.update(({"paid_jobs": False, "T30_graph": False, "independent_gpu_seconds": 14400}
                  if horizon == 7 else {"T30_graph": True, "prepared_artifact_id": 742,
                                       "execution_authorization_is_external": True}))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path, artifact, value


def test_legacy_T7_default_and_manifest_are_retained(tmp_path):
    path, artifact, value = manifest(tmp_path, 7)
    assert probe.read_manifest(SimpleNamespace(manifest=path, parameters=artifact)) == value


def test_T30_requires_its_explicit_schema_and_horizon(tmp_path):
    path, artifact, value = manifest(tmp_path, 30)
    args = SimpleNamespace(manifest=path, parameters=artifact, horizon=30)
    assert probe.read_manifest(args) == value
    with pytest.raises(ValueError, match="schema"):
        probe.read_manifest(SimpleNamespace(manifest=path, parameters=artifact, horizon=7))


@pytest.mark.parametrize("key,bad", [("prepared_artifact_id", 743), ("horizon", 7),
                                   ("turnover_sha256", probe.BASE_SHA),
                                   ("native_AD_argnums", [0]), ("execution_authorization_is_external", False)])
def test_T30_rejects_incompatible_scientific_contract(tmp_path, key, bad):
    path, artifact, value = manifest(tmp_path, 30)
    value[key] = bad
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protocol differs"):
        probe.read_manifest(SimpleNamespace(manifest=path, parameters=artifact, horizon=30))
