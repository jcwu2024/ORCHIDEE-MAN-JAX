from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining.typed_sidecar import (
    DEFAULT_CONTRACT_PATH,
    audit_typed_sidecar_contract,
    collate_typed_sidecar_samples,
    fit_typed_sidecar_statistics,
    load_typed_sidecar_contract,
    load_typed_sidecar_index,
    load_typed_sidecar_statistics,
    read_typed_sidecar_shard,
    write_typed_sidecar_shard,
    write_typed_sidecar_statistics,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parent(path: Path, *, year: int, offset: float) -> str:
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
    return _sha256(path)


def _arrays(contract, value: float):
    values = {
        field.path: np.full((2, *field.feature_shape), value, dtype=np.float64)
        for field in contract.fields
    }
    defined = {
        field.path: np.ones((2, *field.feature_shape), dtype=np.bool_)
        for field in contract.fields
    }
    return values, defined


def _dataset(tmp_path: Path):
    contract = load_typed_sidecar_contract()
    parent_dir = tmp_path / "parent"
    train_path = parent_dir / "shards/train.npz"
    validation_path = parent_dir / "shards/validation.npz"
    train_hash = _write_parent(train_path, year=1961, offset=1.0)
    validation_hash = _write_parent(validation_path, year=2005, offset=100.0)
    parent_payload = {
        "schema_version": "daily_teacher_dataset_manifest_v4",
        "dataset_id": contract.parent_dataset_id,
        "teacher_git_head": "test-head",
        "markov_contract_sha256": contract.parent_contract_sha256,
        "shards": [
            {
                "landpoint_id": "001.0-071.0",
                "year": 1961,
                "spatial_split": "train",
                "temporal_split": "train",
                "markov_contract_sha256": contract.parent_contract_sha256,
                "shard": "shards/train.npz",
                "shard_sha256": train_hash,
            },
            {
                "landpoint_id": "003.0-071.0",
                "year": 2005,
                "spatial_split": "validation",
                "temporal_split": "validation",
                "markov_contract_sha256": contract.parent_contract_sha256,
                "shard": "shards/validation.npz",
                "shard_sha256": validation_hash,
            },
        ],
    }
    parent_manifest = parent_dir / "dataset_manifest.json"
    parent_manifest.write_text(json.dumps(parent_payload), encoding="utf-8")

    sidecar_dir = tmp_path / "sidecar"
    records = []
    for item, value in zip(parent_payload["shards"], (1.0, 100.0), strict=True):
        values, defined = _arrays(contract, value)
        sidecar_path = sidecar_dir / f"{item['year']}.npz"
        write_typed_sidecar_shard(
            sidecar_path,
            contract=contract,
            day_index=np.asarray([1, 2], dtype=np.int32),
            values=values,
            defined=defined,
        )
        records.append(
            {
                "landpoint_id": item["landpoint_id"],
                "year": item["year"],
                "spatial_split": item["spatial_split"],
                "temporal_split": item["temporal_split"],
                "parent_shard_sha256": item["shard_sha256"],
                "sidecar": sidecar_path.name,
                "sidecar_sha256": _sha256(sidecar_path),
                "day_count": 2,
            }
        )
    sidecar_payload = {
        "schema_version": "daily_typed_sidecar_dataset_v1",
        "sidecar_contract_sha256": contract.sha256,
        "parent": {
            "dataset_id": contract.parent_dataset_id,
            "dataset_manifest_sha256": _sha256(parent_manifest),
            "markov_contract_sha256": contract.parent_contract_sha256,
        },
        "shards": records,
    }
    sidecar_manifest = sidecar_dir / "dataset_manifest.json"
    sidecar_manifest.write_text(json.dumps(sidecar_payload), encoding="utf-8")
    return contract, parent_manifest, sidecar_manifest


def test_frozen_typed_sidecar_contract_covers_29_labels_and_66_fields():
    report = audit_typed_sidecar_contract()

    assert report["passed"]
    assert report["label_count"] == 29
    assert report["field_count"] == 66
    assert report["all_fields_have_defined_mask"]


def test_lossless_dense_coo_and_constant_zero_roundtrip(tmp_path):
    contract = load_typed_sidecar_contract()
    values, defined = _arrays(contract, 0.0)
    dense_field, coo_field, zero_field = contract.fields[:3]
    values[dense_field.path][0, 0] = 2.5
    values[coo_field.path][0, 0] = -0.0
    values[coo_field.path][1, 0] = 3.5
    layouts = {
        dense_field.path: "dense",
        coo_field.path: "coo",
        zero_field.path: "constant_zero",
    }
    path = tmp_path / "sidecar.npz"

    write_typed_sidecar_shard(
        path,
        contract=contract,
        day_index=np.asarray([1, 2], dtype=np.int32),
        values=values,
        defined=defined,
        layouts=layouts,
    )
    restored, masks, days = read_typed_sidecar_shard(path, contract=contract)

    np.testing.assert_array_equal(days, [1, 2])
    for field in contract.fields:
        assert restored[field.path].tobytes() == values[field.path].tobytes()
        np.testing.assert_array_equal(masks[field.path], defined[field.path])


def test_hash_join_and_typed_collation_hide_anonymous_parent_target(tmp_path):
    contract, parent_manifest, sidecar_manifest = _dataset(tmp_path)
    index = load_typed_sidecar_index(
        sidecar_manifest,
        parent_manifest_path=parent_manifest,
    )
    samples = list(index.samples())
    batch = collate_typed_sidecar_samples((samples[0], samples[2]))

    assert "fast_day_target" not in batch["inputs"]
    assert "fast_day_target" not in batch["targets"]
    assert "sample_landpoint_id" not in batch
    field = contract.fields[0]
    _, label_name = field.label_id.split(".")
    field_name = field.path.split(".", maxsplit=1)[1]
    captured = batch["targets"].get("supplemental").values.water[label_name][
        field_name
    ]
    assert captured.shape == (2, *field.feature_shape)
    np.testing.assert_array_equal(captured[:, 0], [1.0, 100.0])


def test_statistics_are_explicit_mask_finite_and_train_train_only(tmp_path):
    contract, parent_manifest, sidecar_manifest = _dataset(tmp_path)
    raw = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
    train_path = sidecar_manifest.parent / raw["shards"][0]["sidecar"]
    values, defined, days = read_typed_sidecar_shard(train_path, contract=contract)
    selected = contract.fields[0]
    defined[selected.path][:] = False
    defined[selected.path][0, 0] = True
    write_typed_sidecar_shard(
        train_path,
        contract=contract,
        day_index=days,
        values=values,
        defined=defined,
    )
    raw["shards"][0]["sidecar_sha256"] = _sha256(train_path)
    sidecar_manifest.write_text(json.dumps(raw), encoding="utf-8")
    index = load_typed_sidecar_index(
        sidecar_manifest,
        parent_manifest_path=parent_manifest,
    )

    statistics = fit_typed_sidecar_statistics(index)

    assert statistics.selection == {
        "spatial_split": "train",
        "temporal_split": "train",
    }
    assert statistics.sample_count == 2
    assert statistics.source_sidecars == (
        f"001.0-071.0:1961:{raw['shards'][0]['sidecar_sha256']}",
    )
    assert statistics.fields[selected.path].count[0] == 1
    assert statistics.fields[selected.path].mean[0] == 1.0
    other = contract.fields[-1]
    assert np.all(statistics.fields[other.path].mean == 1.0)
    metadata = write_typed_sidecar_statistics(statistics, tmp_path / "stats.json")
    stored = json.loads(metadata.read_text(encoding="utf-8"))
    assert stored["selection"] == statistics.selection
    assert stored["sample_count"] == 2
    loaded = load_typed_sidecar_statistics(metadata, index=index)
    assert loaded.source_sidecars == statistics.source_sidecars
    np.testing.assert_array_equal(
        loaded.fields[selected.path].count,
        statistics.fields[selected.path].count,
    )

    sidecar_manifest.write_text(
        sidecar_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    changed_index = load_typed_sidecar_index(
        sidecar_manifest,
        parent_manifest_path=parent_manifest,
    )
    with pytest.raises(ValueError, match="statistics dataset identity mismatch"):
        load_typed_sidecar_statistics(metadata, index=changed_index)


@pytest.mark.parametrize(
    ("day_index", "message"),
    [
        (np.asarray([1, 3], dtype=np.int32), "missing days"),
        (np.asarray([1, 1], dtype=np.int32), "duplicate days"),
    ],
)
def test_writer_rejects_missing_or_duplicate_days(tmp_path, day_index, message):
    contract = load_typed_sidecar_contract()
    values, defined = _arrays(contract, 1.0)

    with pytest.raises(ValueError, match=message):
        write_typed_sidecar_shard(
            tmp_path / "bad.npz",
            contract=contract,
            day_index=day_index,
            values=values,
            defined=defined,
        )


def test_reader_rejects_shape_and_mask_drift(tmp_path):
    contract = load_typed_sidecar_contract()
    values, defined = _arrays(contract, 1.0)
    field = contract.fields[0]
    values[field.path] = values[field.path][:, :-1]
    with pytest.raises(ValueError, match="shape/dtype drift"):
        write_typed_sidecar_shard(
            tmp_path / "shape.npz",
            contract=contract,
            day_index=np.asarray([1, 2], dtype=np.int32),
            values=values,
            defined=defined,
        )
    values, defined = _arrays(contract, 1.0)
    defined[field.path] = defined[field.path].astype(np.uint8)
    with pytest.raises(ValueError, match="defined-mask drift"):
        write_typed_sidecar_shard(
            tmp_path / "mask.npz",
            contract=contract,
            day_index=np.asarray([1, 2], dtype=np.int32),
            values=values,
            defined=defined,
        )


def test_contract_rejects_unit_drift_even_with_recomputed_self_hash(tmp_path):
    raw = json.loads(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))
    raw["labels"][0]["unit"] = "wrong unit"
    unhashed = dict(raw)
    unhashed.pop("contract_sha256")
    raw["contract_sha256"] = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unit drift"):
        load_typed_sidecar_contract(path)


def test_index_rejects_duplicate_missing_and_parent_hash_drift(tmp_path):
    _, parent_manifest, sidecar_manifest = _dataset(tmp_path)
    raw = json.loads(sidecar_manifest.read_text(encoding="utf-8"))

    raw["shards"][0]["parent_shard_sha256"] = "0" * 64
    sidecar_manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="parent hash mismatch"):
        load_typed_sidecar_index(
            sidecar_manifest,
            parent_manifest_path=parent_manifest,
        )

    _, parent_manifest, sidecar_manifest = _dataset(tmp_path / "sidecar-hash")
    raw = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
    raw["shards"][0]["sidecar_sha256"] = "0" * 64
    sidecar_manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="typed-sidecar hash mismatch"):
        load_typed_sidecar_index(
            sidecar_manifest,
            parent_manifest_path=parent_manifest,
        )

    _, parent_manifest, sidecar_manifest = _dataset(tmp_path / "duplicate")
    raw = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
    raw["shards"].append(dict(raw["shards"][0]))
    sidecar_manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate typed-sidecar shard"):
        load_typed_sidecar_index(
            sidecar_manifest,
            parent_manifest_path=parent_manifest,
        )

    _, parent_manifest, sidecar_manifest = _dataset(tmp_path / "missing")
    raw = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
    raw["shards"] = raw["shards"][:1]
    sidecar_manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing parent shards"):
        load_typed_sidecar_index(
            sidecar_manifest,
            parent_manifest_path=parent_manifest,
        )
