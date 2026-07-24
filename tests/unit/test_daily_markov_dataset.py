from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining import markov_dataset
from research.daily_coarse_graining.benchmark_markov_dataset import benchmark


def _write_shard(path: Path, *, year: int, offset: float = 0.0) -> str:
    days = 2
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        state_trajectory=np.arange(12, dtype=np.float64).reshape(3, 4) + offset,
        fast_day_target=np.full((days, 6), offset),
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


def _manifest(
    tmp_path: Path,
    *,
    schema_version: str = markov_dataset.DATASET_SCHEMA_VERSION,
) -> Path:
    contract = "contract-sha256"
    first = tmp_path / "workers/first.npz"
    second = tmp_path / "workers/second.npz"
    first_hash = _write_shard(first, year=1961)
    second_hash = _write_shard(second, year=1962, offset=10.0)
    payload = {
        "schema_version": schema_version,
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


def test_dataset_index_remains_compatible_with_v3_manifests(tmp_path):
    index = markov_dataset.load_dataset_index(
        _manifest(
            tmp_path,
            schema_version=markov_dataset.LEGACY_DATASET_SCHEMA_VERSION,
        )
    )
    assert len(index.shards) == 2


def test_collation_stacks_model_values_but_not_landpoint_identity(tmp_path):
    index = markov_dataset.load_dataset_index(_manifest(tmp_path))
    samples = list(index.samples())
    batch = markov_dataset.collate_samples((samples[0], samples[2]))
    assert batch["state"].shape == (2, 4)
    assert batch["forcing_native"].shape == (2, 5, 9)
    assert batch["fast_day_target"].shape == (2, 6)
    assert batch["parameters"].shape == (2, 2)
    assert batch["annual_conditions"].shape == (2, 1)
    assert batch["discrete_state"]["flag"].shape == (2, 1)
    assert "sample_landpoint_id" not in batch
    assert batch["state_finite"].all()
    assert batch["state_delta"].shape == (2, 4)
    for name in markov_dataset.CONTINUOUS_ARRAY_NAMES:
        assert batch[f"{name}_finite"].shape == batch[name].shape


def test_contiguous_windows_never_cross_landpoint_or_year_boundaries(tmp_path):
    index = markov_dataset.load_dataset_index(_manifest(tmp_path))
    windows = list(index.windows(2))

    assert len(windows) == 2
    assert {window["sample_landpoint_id"] for window in windows} == {
        "001.0-071.0",
        "003.0-071.0",
    }
    for window in windows:
        assert window["state_trajectory"].shape == (3, 4)
        assert window["teacher_next_state"].shape == (2, 4)
        assert window["teacher_fast_day_target"].shape == (2, 6)
        assert window["forcing_native"].shape == (2, 5, 9)
        assert window["discrete_trajectory"]["flag"].shape == (3, 1)
        np.testing.assert_array_equal(window["day_index"], [1, 2])
        assert np.unique(window["year"]).size == 1
    assert list(index.windows(3)) == []
    with pytest.raises(ValueError, match="positive"):
        list(index.windows(0))


def test_collate_windows_stacks_equal_horizon_trajectories(tmp_path):
    index = markov_dataset.load_dataset_index(_manifest(tmp_path))
    windows = list(index.windows(2))
    batch = markov_dataset.collate_windows(windows)

    assert batch["initial_state"].shape == (2, 4)
    assert batch["state_trajectory"].shape == (2, 3, 4)
    assert batch["teacher_next_state"].shape == (2, 2, 4)
    assert batch["teacher_fast_day_target"].shape == (2, 2, 6)
    assert batch["forcing_native"].shape == (2, 2, 5, 9)
    assert batch["initial_discrete_state"]["flag"].shape == (2, 1)
    assert batch["discrete_trajectory"]["flag"].shape == (2, 3, 1)
    assert "sample_landpoint_id" not in batch

    shorter = {**windows[1], "day_index": windows[1]["day_index"][:1]}
    with pytest.raises(ValueError, match="fixed horizon"):
        markov_dataset.collate_windows((windows[0], shorter))


def test_markov_window_rejects_noncontiguous_day_indices(tmp_path):
    manifest = _manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    shard_path = tmp_path / raw["shards"][0]["shard"]
    with np.load(shard_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["day_index"] = np.asarray([1, 3], dtype=np.int32)
    np.savez(shard_path, **arrays)
    raw["shards"][0]["shard_sha256"] = hashlib.sha256(
        shard_path.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    index = markov_dataset.load_dataset_index(manifest)
    with pytest.raises(ValueError, match="not contiguous"):
        list(index.windows(2, spatial_split="train"))


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

    manifest = _manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["markov_contract"] = {"unexpected": True}
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="contract metadata hash mismatch"):
        markov_dataset.load_dataset_index(manifest)


def test_streaming_statistics_use_only_train_split_and_handle_nonfinite_columns(tmp_path):
    manifest = _manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    shard_path = tmp_path / raw["shards"][0]["shard"]
    with np.load(shard_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["state_trajectory"][:, 0] = np.nan
    arrays["state_trajectory"][1:, 1] = np.nan
    np.savez(shard_path, **arrays)
    raw["shards"][0]["shard_sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    index = markov_dataset.load_dataset_index(manifest)
    statistics = markov_dataset.fit_training_statistics(index, chunk_rows=1)
    assert statistics.sample_count == 2
    assert statistics.source_shards[0].startswith("001.0-071.0:1961:")
    state = statistics.arrays["state"]
    np.testing.assert_array_equal(state.count, [0, 1, 2, 2])
    np.testing.assert_allclose(state.mean, [0.0, 1.0, 4.0, 5.0])
    np.testing.assert_allclose(state.variance, [0.0, 0.0, 4.0, 4.0])
    np.testing.assert_allclose(state.scale, [1.0, 1.0, 2.0, 2.0])
    assert statistics.arrays["forcing_native"].count[0] == 10
    assert statistics.arrays["forcing_native"].mean[0] == 0.0
    assert statistics.arrays["parameters"].count[0] == 2
    assert statistics.arrays["state_delta"].count[0] == 0


def test_statistics_asset_roundtrip_and_safe_normalization(tmp_path):
    index = markov_dataset.load_dataset_index(_manifest(tmp_path))
    statistics = markov_dataset.fit_training_statistics(index)
    metadata = markov_dataset.write_training_statistics(statistics, tmp_path / "training_statistics.json")
    loaded = markov_dataset.load_training_statistics(metadata, index=index)
    np.testing.assert_array_equal(loaded.arrays["state"].count, statistics.arrays["state"].count)
    np.testing.assert_allclose(loaded.arrays["state"].variance, statistics.arrays["state"].variance)
    normalized, finite = markov_dataset.normalize_finite(np.asarray([[np.nan, 1.0, 6.0, 7.0]]), loaded.arrays["state"])
    np.testing.assert_array_equal(finite, [[False, True, True, True]])
    assert normalized[0, 0] == 0.0
    assert np.isfinite(normalized).all()
    restored = markov_dataset.denormalize(normalized, loaded.arrays["state"])
    np.testing.assert_allclose(restored[0, 1:], [1.0, 6.0, 7.0])

    arrays_path = metadata.with_suffix(".npz")
    arrays_path.write_bytes(arrays_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="array hash mismatch"):
        markov_dataset.load_training_statistics(metadata)


def test_statistics_and_normalization_exclude_orchidee_sentinels(tmp_path):
    manifest = _manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    shard_path = tmp_path / raw["shards"][0]["shard"]
    with np.load(shard_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["state_trajectory"][:, 0] = 1.0e20
    arrays["fast_day_target"][:, 0] = -1.0e20
    np.savez(shard_path, **arrays)
    raw["shards"][0]["shard_sha256"] = hashlib.sha256(
        shard_path.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    statistics = markov_dataset.fit_training_statistics(
        markov_dataset.load_dataset_index(manifest)
    )
    assert statistics.arrays["state"].count[0] == 0
    assert statistics.arrays["fast_day_target"].count[0] == 0
    normalized, valid = markov_dataset.normalize_finite(
        np.asarray([[1.0e20, -1.0e20, np.nan, 2.0]]),
        statistics.arrays["state"],
    )
    np.testing.assert_array_equal(valid, [[False, False, False, True]])
    np.testing.assert_allclose(normalized, [[0.0, 0.0, 0.0, -1.5]])


def test_prefetched_batches_preserve_order_and_propagate_errors(tmp_path):
    index = markov_dataset.load_dataset_index(_manifest(tmp_path))
    batches = list(markov_dataset.prefetched_batches(index.samples(), 3, prefetch=1))
    assert [batch["state"].shape[0] for batch in batches] == [3, 1]
    np.testing.assert_array_equal(batches[0]["day_index"], [1, 2, 1])

    def broken_samples():
        yield next(index.samples())
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        list(markov_dataset.prefetched_batches(broken_samples(), 1))


def test_memory_benchmark_replays_verified_shards_without_dataset_materialization(tmp_path):
    result = benchmark(_manifest(tmp_path), repeats=4, batch_size=3, prefetch=1)
    assert result["samples"] == 8
    assert result["batches"] == 3
    assert result["largest_collated_batch_bytes"] > 0
    assert np.isfinite(result["checksum"])
