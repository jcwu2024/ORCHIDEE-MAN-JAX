from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research.daily_coarse_graining import canonical_training_run as training
from research.daily_coarse_graining.markov_dataset import DATASET_SCHEMA_VERSION


def _write_shard(path: Path, *, year: int, offset: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    days = 2
    np.savez(
        path,
        state_trajectory=np.arange(12, dtype=np.float64).reshape(3, 4) + offset,
        fast_day_target=np.arange(12, dtype=np.float64).reshape(2, 6) / 10 + offset,
        forcing_native=np.full((days, 5, 3), offset),
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
    contract = {
        "fast_day_target_width": 6,
        "fast_day_target_leaves": [
            {"family": "hydrol", "start": 0, "stop": 3},
            {"family": "ok_leak", "start": 3, "stop": 6},
        ],
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    definitions = (
        ("train-train", "001.0-071.0", 1961, "train", "train", 0.0),
        ("train-validation", "001.0-071.0", 1962, "train", "validation", 0.5),
        ("validation-train", "003.0-071.0", 1961, "validation", "train", 1.0),
        (
            "validation-validation",
            "003.0-071.0",
            1962,
            "validation",
            "validation",
            1.5,
        ),
    )
    shards = []
    for name, landpoint, year, spatial, temporal, offset in definitions:
        path = tmp_path / "shards" / f"{name}.npz"
        shards.append(
            {
                "landpoint_id": landpoint,
                "year": year,
                "spatial_split": spatial,
                "temporal_split": temporal,
                "shard": str(path.relative_to(tmp_path)),
                "shard_sha256": _write_shard(path, year=year, offset=offset),
            }
        )
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "dataset_id": "canonical-training-test",
                "teacher_git_head": "teacher-commit",
                "markov_contract_sha256": contract_hash,
                "markov_contract": contract,
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_streamed_fast_day_training_and_resume(tmp_path):
    manifest = _manifest(tmp_path)
    statistics = training.fit_statistics_asset(
        manifest,
        tmp_path / "statistics.json",
        chunk_rows=1,
    )
    output = tmp_path / "training"
    first = training.train_experiment(
        manifest,
        statistics,
        output,
        epochs=1,
        batch_size=2,
        learning_rate=1.0e-3,
        seed=7,
        max_eval_batches=1,
        architecture={
            "state_latent_width": 4,
            "forcing_latent_width": 3,
            "condition_latent_width": 3,
            "hidden_width": 8,
        },
    )

    assert first["epochs"] == 1
    assert first["test_split_evaluated"] is False
    assert set(first["history"][0]["validation"]) == {
        "joint",
        "spatial",
        "temporal",
    }
    assert (output / "checkpoint.pkl").is_file()
    assert (output / "training_report.json").is_file()

    resumed = training.train_experiment(
        manifest,
        statistics,
        output,
        epochs=2,
        batch_size=2,
        learning_rate=1.0e-3,
        seed=7,
        max_eval_batches=1,
        architecture={
            "state_latent_width": 4,
            "forcing_latent_width": 3,
            "condition_latent_width": 3,
            "hidden_width": 8,
        },
    )
    assert [item["epoch"] for item in resumed["history"]] == [1, 2]


def test_contract_loader_supports_legacy_v3_shard_metadata(tmp_path):
    manifest = _manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    contract = raw.pop("markov_contract")
    raw["schema_version"] = "daily_teacher_dataset_manifest_v3"
    for shard in raw["shards"]:
        shard["markov_contract_sha256"] = raw["markov_contract_sha256"]
        shard["markov_contract"] = contract
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    assert training.load_contract_metadata(manifest) == contract
