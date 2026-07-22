from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining import markov_dataset


def _write_shard(path: Path, *, year: int, offset: float = 0.0) -> str:
    days = 2
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        state_trajectory=np.arange(12, dtype=np.float64).reshape(3, 4) + offset,
        forcing_native=np.full((days, 5, 9), offset),
        forcing_record_indices=np.arange(10, dtype=np.int32).reshape(days, 5),
        parameters=np.asarray([1.0, 2.0]),
        landpoint_static=np.asarray([3.0, 4.0, 5.0]),
        annual_conditions=np.asarray([317.27]),
        diagnostics=np.full((days, 2), offset),
        year=np.asarray(year, dtype=np.int32),
        day_index=np.asarray([1, 2], dtype=np.int32),
        state_discrete__flag=np.asarray([[True], [False], [True]]),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    contract = "contract-sha256"
    first = tmp_path / "workers/first.npz"
    second = tmp_path / "workers/second.npz"
    first_hash = _write_shard(first, year=1961)
    second_hash = _write_shard(second, year=1962, offset=10.0)
    payload = {
        "schema_version": markov_dataset.DATASET_SCHEMA_VERSION,
        "dataset_id": "test-dataset",
        "teacher_git_head": "abc123",
        "markov_contract_sha256": contract,
        "shards": [
            {
                "landpoint_id": "001.0-071.0",
                "year": 1961,
                "spatial_split": "train",
                "temporal_split": "train",
                "markov_contract_sha256": contract,
                "shard": "workers/first.npz",
                "shard_sha256": first_hash,
            },
            {
                "landpoint_id": "003.0-071.0",
                "year": 1962,
                "spatial_split": "validation",
                "temporal_split": "validation",
                "markov_contract_sha256": contract,
                "shard": "workers/second.npz",
                "shard_sha256": second_hash,
            },
        ],
    }
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dataset_index_verifies_hashes_and_selects_frozen_splits(tmp_path):
    index = markov_dataset.load_dataset_index(_manifest(tmp_path))
    assert index.dataset_id == "test-dataset"
    assert len(index.select(spatial_split="train")) == 1
    samples = list(index.samples(spatial_split="validation"))
    assert len(samples) == 2
    assert samples[0]["sample_landpoint_id"] == "003.0-071.0"
    assert samples[0]["year"] == 1962


def test_collation_stacks_model_values_but_not_landpoint_identity(tmp_path):
    index = markov_dataset.load_dataset_index(_manifest(tmp_path))
    samples = list(index.samples())
    batch = markov_dataset.collate_samples((samples[0], samples[2]))
    assert batch["state"].shape == (2, 4)
    assert batch["forcing_native"].shape == (2, 5, 9)
    assert batch["parameters"].shape == (2, 2)
    assert batch["annual_conditions"].shape == (2, 1)
    assert batch["discrete_state"]["flag"].shape == (2, 1)
    assert "sample_landpoint_id" not in batch
    assert batch["state_finite"].all()


def test_dataset_index_rejects_corruption_and_contract_drift(tmp_path):
    manifest = _manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["shards"][0]["markov_contract_sha256"] = "different"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="contract drift"):
        markov_dataset.load_dataset_index(manifest)

    manifest = _manifest(tmp_path)
    (tmp_path / "workers/first.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        markov_dataset.load_dataset_index(manifest)
