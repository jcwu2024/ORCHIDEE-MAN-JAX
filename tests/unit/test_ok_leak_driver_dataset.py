from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.carbon_budget_ownership import (
    OK_LEAK_DRIVER_SERIES,
    ok_leak_driver_capture_metadata,
)
from research.daily_coarse_graining.ok_leak_driver_dataset import (
    _canonical_sha256,
    _sha256_array,
    _sha256_file,
    load_ok_leak_driver_dataset,
)


def _write_fixture(tmp_path: Path, monkeypatch):
    import research.daily_coarse_graining.ok_leak_driver_dataset as reader

    dataset_root = tmp_path / "dataset"
    capture_root = tmp_path / "capture"
    dataset_root.mkdir()
    capture_day = capture_root / "captures" / "001.0-071.0" / "1961_day_002"
    capture_day.mkdir(parents=True)
    state = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    target = np.asarray([[5.0, 6.0]])
    source_path = dataset_root / "1961.npz"
    source_path.write_bytes(b"synthetic-source")
    source_hash = _sha256_file(source_path)
    dataset_manifest = {
        "dataset_id": "synthetic-v5",
        "teacher_git_head": "1" * 40,
        "markov_contract_sha256": "2" * 64,
        "markov_contract": {},
    }
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")
    record = {
        "landpoint_id": "001.0-071.0",
        "year": 1961,
        "day_index": 2,
        "row": 0,
        "source_shard": "1961.npz",
        "source_shard_sha256": source_hash,
        "day_start_state_sha256": _sha256_array(state[0]),
        "fast_day_target_sha256": _sha256_array(target[0]),
    }
    unsigned_plan = {
        "schema_version": "ok_leak_auxiliary_capture_plan_v1",
        "dataset_id": dataset_manifest["dataset_id"],
        "teacher_git_head": dataset_manifest["teacher_git_head"],
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "markov_contract_sha256": dataset_manifest["markov_contract_sha256"],
        "sealed_test_used": False,
        "records": [record],
    }
    plan = unsigned_plan | {"plan_sha256": _canonical_sha256(unsigned_plan)}
    plan_path = tmp_path / "capture_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    arrays = {
        item.field: np.full((48, 1), index, dtype=np.float64)
        for index, item in enumerate(OK_LEAK_DRIVER_SERIES)
    }
    arrays_path = capture_day / "ok_leak_driver_series.npz"
    np.savez_compressed(arrays_path, **arrays)
    arrays_hash = _sha256_file(arrays_path)
    report = {
        "schema_version": "ok_leak_outer_block_reentry_probe_v2",
        "capture_interface_passed": True,
        "sealed_test_used": False,
        "landpoint_id": record["landpoint_id"],
        "year": record["year"],
        "day_index": record["day_index"],
        "day_start_state_sha256": record["day_start_state_sha256"],
        "fast_day_target_sha256": record["fast_day_target_sha256"],
        "capture_plan_sha256": plan["plan_sha256"],
        "source_shard_sha256": source_hash,
        "capture_npz": arrays_path.name,
        "capture_npz_sha256": arrays_hash,
        "capture": ok_leak_driver_capture_metadata(arrays)
        | {"path": str(arrays_path), "sha256": arrays_hash},
    }
    report_path = capture_day / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest = {
        "schema_version": "ok_leak_auxiliary_capture_manifest_v2",
        "status": "complete",
        "capture_plan_sha256": plan["plan_sha256"],
        "dataset_id": dataset_manifest["dataset_id"],
        "sealed_test_used": False,
        "capture_interface_failed_count": 0,
        "records": [
            {
                "landpoint_id": record["landpoint_id"],
                "year": record["year"],
                "day_index": record["day_index"],
                "report": str(report_path.relative_to(capture_root)),
                "capture_npz_sha256": arrays_hash,
            }
        ],
    }
    (capture_root / "capture_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    shard = SimpleNamespace(
        days=1,
        state_trajectory=state,
        fast_day_target=target,
        year=np.asarray(1961),
        day_index=np.asarray([2]),
        sample=lambda row: {
            "state": state[row],
            "next_state": state[row + 1],
            "fast_day_target": target[row],
            "forcing_native": np.zeros((4, 2)),
            "parameters": np.zeros(3),
            "landpoint_static": np.zeros(4),
            "annual_conditions": np.zeros(2),
            "diagnostics": np.zeros(5),
            "discrete_state": {"date": np.asarray(1)},
            "next_discrete_state": {"date": np.asarray(2)},
        },
    )
    reference = SimpleNamespace(
        landpoint_id=record["landpoint_id"],
        year=record["year"],
        path=source_path,
        sha256=source_hash,
    )
    monkeypatch.setattr(
        reader,
        "load_dataset_index",
        lambda *_args, **_kwargs: SimpleNamespace(
            dataset_id=dataset_manifest["dataset_id"],
            teacher_git_head=dataset_manifest["teacher_git_head"],
            contract_sha256=dataset_manifest["markov_contract_sha256"],
            shards=(reference,),
        ),
    )
    monkeypatch.setattr(reader, "daily_markov_contract_from_metadata", lambda _: object())
    monkeypatch.setattr(reader, "load_markov_shard", lambda *_args, **_kwargs: shard)
    return capture_root, plan_path, dataset_manifest_path, arrays_path


def test_hash_bound_reader_joins_capture_to_exact_teacher_row(tmp_path, monkeypatch):
    capture_root, plan_path, manifest_path, _ = _write_fixture(tmp_path, monkeypatch)

    dataset = load_ok_leak_driver_dataset(capture_root, plan_path, manifest_path)

    assert dataset.dataset_id == "synthetic-v5"
    assert len(dataset.samples) == 1
    sample = dataset.samples[0]
    assert (sample.landpoint_id, sample.year, sample.day_index) == (
        "001.0-071.0",
        1961,
        2,
    )
    assert tuple(sample.driver_steps) == tuple(
        item.field for item in OK_LEAK_DRIVER_SERIES
    )
    assert all(value.shape == (48, 1) for value in sample.driver_steps.values())
    np.testing.assert_array_equal(sample.state, [1.0, 2.0])
    np.testing.assert_array_equal(sample.next_state, [3.0, 4.0])


def test_hash_bound_reader_rejects_capture_npz_corruption(tmp_path, monkeypatch):
    capture_root, plan_path, manifest_path, arrays_path = _write_fixture(
        tmp_path, monkeypatch
    )
    with arrays_path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ValueError, match="capture NPZ hash mismatch"):
        load_ok_leak_driver_dataset(capture_root, plan_path, manifest_path)
