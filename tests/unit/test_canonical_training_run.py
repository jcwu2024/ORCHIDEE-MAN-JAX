from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
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
        "state_leaves": [
            {
                "component": "diffuco_previous_step_state",
                "path": ["rveget"],
                "shape": [1],
                "start": 0,
                "stop": 1,
            },
            {
                "component": "hydrol_previous_step_state",
                "path": ["water"],
                "shape": [2],
                "start": 1,
                "stop": 3,
            }
        ],
        "fast_day_target_leaves": [
            {
                "family": "diffuco_enerbil",
                "component": "diffuco_previous_step_state",
                "path": ["rveget"],
                "key": "diffuco_previous_step_state.rveget",
                "start": 0,
                "stop": 1,
            },
            {
                "family": "hydrol",
                "component": "hydrol_previous_step_state",
                "path": ["water"],
                "key": "hydrol_previous_step_state.water",
                "start": 1,
                "stop": 3,
            },
            {
                "family": "ok_leak",
                "component": None,
                "path": ["DOC"],
                "key": "ok_leak.DOC",
                "start": 3,
                "stop": 6,
            },
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
        ("test-test", "005.0-071.0", 1963, "test", "test", 2.0),
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
                "status": "complete",
                "provisional_teacher": True,
                "plan_sha256": "test-plan-sha256",
                "teacher_git_head": "teacher-commit",
                "landpoint_count": 3,
                "year_count": 3,
                "shard_count": len(shards),
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
    acceptance_dir = tmp_path / "acceptance"
    training.accept_training_dataset(
        manifest,
        acceptance_dir,
        chunk_rows=1,
        batch_size=2,
        seed=7,
        architecture={
            "state_latent_width": 4,
            "forcing_latent_width": 3,
            "condition_latent_width": 3,
            "hidden_width": 8,
        },
    )
    statistics = acceptance_dir / "training_statistics.json"
    acceptance = acceptance_dir / "acceptance_report.json"
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
        acceptance_path=acceptance,
    )

    assert first["epochs"] == 1
    assert first["test_split_evaluated"] is False
    assert set(first["history"][0]["validation"]) == {
        "joint",
        "spatial",
        "temporal",
    }
    assert first["history"][0]["validation"]["temporal"]["physical_leaves"]
    assert (output / "checkpoint.pkl").is_file()
    assert (output / "best_checkpoint.pkl").is_file()
    assert (output / "training_report.json").is_file()
    assert first["best_epoch"] == 1
    assert np.isfinite(first["best_validation_score"])
    assert first["target_representation_audit"]["status"] == "passed"
    assert first["identity"]["acceptance_sha256"] == hashlib.sha256(
        acceptance.read_bytes()
    ).hexdigest()
    with (output / "checkpoint.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    assert {
        np.asarray(value).dtype
        for value in jax.tree_util.tree_leaves(checkpoint["parameters"])
    } == {np.dtype("float32")}

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
        acceptance_path=acceptance,
    )
    assert [item["epoch"] for item in resumed["history"]] == [1, 2]


def test_production_protocol_training_consumes_partial_final_batch(
    tmp_path,
    monkeypatch,
):
    manifest = _manifest(tmp_path)
    acceptance_dir = tmp_path / "acceptance"
    training.accept_training_dataset(
        manifest,
        acceptance_dir,
        chunk_rows=1,
        batch_size=2,
        seed=7,
        architecture={
            "state_latent_width": 4,
            "forcing_latent_width": 3,
            "condition_latent_width": 3,
            "hidden_width": 8,
        },
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        training,
        "load_training_protocol",
        lambda _path: SimpleNamespace(
            path=protocol_path,
            sha256="protocol-sha256",
            max_open_shards=1,
            raw={
                "parent_data_product": {
                    "teacher_git_head": "teacher-commit",
                    "contract_sha256": json.loads(
                        manifest.read_text(encoding="utf-8")
                    )["markov_contract_sha256"],
                    "generation_plan_sha256": "test-plan-sha256",
                },
                "sampling": {
                    "strategy": "exact_once_hierarchical_shard_pool_v1"
                },
            },
        ),
    )

    report = training.train_experiment(
        manifest,
        acceptance_dir / "training_statistics.json",
        tmp_path / "protocol-training",
        epochs=1,
        batch_size=3,
        learning_rate=1.0e-3,
        seed=7,
        max_eval_batches=1,
        architecture={
            "state_latent_width": 4,
            "forcing_latent_width": 3,
            "condition_latent_width": 3,
            "hidden_width": 8,
        },
        acceptance_path=acceptance_dir / "acceptance_report.json",
        training_protocol_path=protocol_path,
    )

    assert report["history"][0]["training_samples"] == 2
    assert report["history"][0]["training_batches"] == 1
    assert report["identity"]["training_protocol_sha256"] == "protocol-sha256"


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


def test_dataset_acceptance_builds_hash_bound_statistics_and_gradient_gate(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "acceptance"

    report = training.accept_training_dataset(
        manifest,
        output,
        chunk_rows=1,
        batch_size=2,
        seed=11,
        architecture={
            "state_latent_width": 4,
            "forcing_latent_width": 3,
            "condition_latent_width": 3,
            "hidden_width": 8,
        },
    )

    assert report["status"] == "passed"
    assert report["dataset_validation"]["shards"] == 5
    assert report["dataset_validation"]["split_pair_shards"]["test/test"] == 1
    assert report["target_representation_audit"]["status"] == "passed"
    assert report["training_plumbing_smoke"]["status"] == "passed"
    assert report["training_plumbing_smoke"]["gradient_norm"] > 0.0
    statistics = output / "training_statistics.json"
    acceptance = output / "acceptance_report.json"
    assert statistics.is_file()
    assert statistics.with_suffix(".npz").is_file()
    assert acceptance.is_file()
    verified = training.verify_training_acceptance(
        acceptance,
        manifest,
        statistics,
    )
    assert verified["identity"] == report["identity"]

    statistics.write_text(
        statistics.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(ValueError, "statistics hash mismatch"):
        training.verify_training_acceptance(
            acceptance,
            manifest,
            statistics,
        )


def test_dataset_acceptance_rejects_nonaggregate_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["status"] = "incomplete"
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "complete aggregate"):
        training.accept_training_dataset(manifest, tmp_path / "acceptance")


def test_training_cli_requires_an_acceptance_asset():
    with np.testing.assert_raises(SystemExit):
        training._parser().parse_args(
            [
                "train",
                "--dataset",
                "dataset.json",
                "--statistics",
                "statistics.json",
                "--output-dir",
                "training",
            ]
        )
