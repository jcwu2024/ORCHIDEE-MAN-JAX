import hashlib
import json
from pathlib import Path

import numpy as np

from research.daily_coarse_graining import production_training_protocol as protocol
from research.daily_coarse_graining.markov_dataset import load_dataset_index

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "manifests"
    / "coarse_graining"
    / "daily_teacher_669_training_protocol.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(tmp_path: Path) -> Path:
    records = []
    contract_sha256 = "contract"
    for point in ("point-a", "point-b"):
        for year in (1961, 1962):
            days = 3
            path = tmp_path / f"{point}-{year}.npz"
            offset = 1000 * (point == "point-b") + year
            np.savez_compressed(
                path,
                state_trajectory=np.arange(2 * (days + 1), dtype=np.float64).reshape(days + 1, 2)
                + offset,
                fast_day_target=np.arange(days, dtype=np.float64).reshape(days, 1),
                forcing_native=np.ones((days, 5, 1), dtype=np.float64),
                forcing_record_indices=np.arange(days * 5, dtype=np.int64).reshape(days, 5),
                parameters=np.asarray([1.0]),
                landpoint_static=np.asarray([2.0]),
                annual_conditions=np.asarray([3.0]),
                diagnostics=np.ones((days, 1), dtype=np.float64),
                year=np.asarray(year, dtype=np.int32),
                day_index=np.arange(1, days + 1, dtype=np.int32),
                state_discrete__flag=np.zeros((days + 1, 1), dtype=np.bool_),
            )
            records.append(
                {
                    "landpoint_id": point,
                    "year": year,
                    "spatial_split": "train",
                    "temporal_split": "train",
                    "markov_contract_sha256": contract_sha256,
                    "shard": path.name,
                    "shard_sha256": _sha256(path),
                }
            )
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "daily_teacher_dataset_manifest_v4",
                "dataset_id": "synthetic",
                "teacher_git_head": "teacher",
                "markov_contract_sha256": contract_sha256,
                "shards": records,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_frozen_production_training_protocol_is_valid():
    loaded = protocol.load_training_protocol(PROTOCOL)

    assert loaded.max_open_shards == 8
    assert loaded.raw["validation"]["sealed_final"]["model_selection"] is False


def test_balanced_sampler_uses_every_landpoint_year_day_once(tmp_path):
    index = load_dataset_index(_dataset(tmp_path))
    samples = list(
        protocol.balanced_samples(
            index,
            seed=7,
            max_open_shards=2,
        )
    )

    observed = {
        (
            sample["sample_landpoint_id"],
            sample["sample_year"],
            int(sample["day_index"]),
        )
        for sample in samples
    }
    assert len(samples) == 12
    assert len(observed) == 12
    assert {sample["sample_landpoint_id"] for sample in samples[:2]} == {
        "point-a",
        "point-b",
    }


def test_streaming_preflight_writes_hash_bound_resource_report(tmp_path):
    manifest = _dataset(tmp_path)
    output = protocol.streaming_io_preflight(
        manifest_path=manifest,
        protocol_path=PROTOCOL,
        output_path=tmp_path / "preflight.json",
        max_shards=4,
        batch_size=5,
        samples_per_shard=2,
        seed=4,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["reader"]["sample_count"] == 8
    assert report["reader"]["batch_count"] == 2
    assert report["reader"]["estimated_uncompressed_pool_bytes"] > 0
    assert report["reader"]["max_collated_batch_bytes"] > 0
