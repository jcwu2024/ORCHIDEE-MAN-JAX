"""Exercise public Teacher readers with original paths backed only by a pack."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining import teacher_assets, teacher_shards
from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
from research.daily_coarse_graining.daily_operator.training_data import (
    coherent_process_training_examples,
)
from research.daily_coarse_graining.markov_dataset import load_dataset_index
from research.daily_coarse_graining.typed_sidecar import (
    load_coherent_typed_sidecar_index,
    load_typed_sidecar_contract,
    read_typed_sidecar_shard,
    write_typed_sidecar_shard,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(root: Path, point: str, paths: tuple[Path, ...], *, retire=True):
    """Minimal real storage publication; no reader functions are patched."""
    manifest_hash = _hash(root / "dataset_manifest.json")
    container = root / "points" / f"{point}.sqlite"
    container.parent.mkdir()
    rows = {}
    with sqlite3.connect(container) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
        connection.execute(
            "CREATE TABLE assets(path TEXT PRIMARY KEY,size INTEGER,mode INTEGER,"
            "mtime_ns INTEGER,sha256 TEXT,data BLOB)"
        )
        for path in paths:
            name = path.relative_to(root).as_posix()
            info = path.stat()
            row = dict(size=info.st_size, mode=info.st_mode,
                       mtime_ns=info.st_mtime_ns, sha256=_hash(path))
            rows[name] = row
            connection.execute("INSERT INTO assets VALUES(?,?,?,?,?,?)", (
                name, row["size"], row["mode"], row["mtime_ns"],
                row["sha256"], path.read_bytes(),
            ))
        snapshot = hashlib.sha256(teacher_assets.canonical(rows)).hexdigest()
        metadata = dict(schema_version=teacher_assets.SCHEMA, landpoint_id=point,
                        dataset_manifest_sha256=manifest_hash,
                        source_rows=json.dumps(rows), snapshot_sha256=snapshot,
                        complete="true")
        connection.executemany("INSERT INTO metadata VALUES(?,?)", metadata.items())
    (root / teacher_assets.MAP_NAME).write_text(json.dumps({
        "schema_version": teacher_assets.SCHEMA,
        "dataset_manifest_sha256": manifest_hash,
        "points": {point: {
            "container": container.relative_to(root).as_posix(),
            "bytes": container.stat().st_size, "sha256": _hash(container),
            "snapshot_sha256": snapshot, "member_count": len(rows),
            "members": list(rows),
        }},
    }), encoding="utf-8")
    if retire:
        for path in paths:
            path.unlink()


@pytest.fixture
def teacher_fixture(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    contract = load_typed_sidecar_contract(
        Path(__file__).resolve().parents[2]
        / "manifests/coarse_graining/daily_typed_supplement_v3.json"
    )
    entry = teacher_shards.PlanEntry(
        landpoint_id="001.0-071.0", year=1961, days=2,
        spatial_split="train", temporal_split="train",
        run_def=tmp_path / "run.def", reference_run_dir=tmp_path,
        state_cache=tmp_path / "initial.pkl", initialization_mode="state_cache",
        acceptance_checkpoint=None,
    )
    worker = root / "workers/worker-000-of-001"
    base, metadata_path, checkpoint = teacher_shards._entry_paths(worker, entry)
    typed = teacher_shards._entry_typed_path(worker, entry)
    base.parent.mkdir(parents=True)
    np.savez(base,
        state_trajectory=np.arange(12, dtype=np.float64).reshape(3, 4),
        fast_day_target=np.arange(12, dtype=np.float64).reshape(2, 6),
        forcing_native=np.zeros((2, 5, 9)),
        forcing_record_indices=np.arange(10, dtype=np.int32).reshape(2, 5),
        parameters=np.asarray([1.0, 2.0]),
        landpoint_static=np.asarray([3.0]), annual_conditions=np.asarray([317.27]),
        diagnostics=np.zeros((2, 2)), year=np.asarray(1961, dtype=np.int32),
        day_index=np.asarray([1, 2], dtype=np.int32),
        state_discrete__flag=np.asarray([[True], [False], [True]]),
    )
    values = {field.path: np.zeros((2, *field.feature_shape), dtype=np.float64)
              for field in contract.fields}
    masks = {name: np.ones(value.shape, dtype=np.bool_) for name, value in values.items()}
    write_typed_sidecar_shard(typed, contract=contract,
                             day_index=np.asarray([1, 2], dtype=np.int32),
                             values=values, defined=masks)
    release = SimpleNamespace(release_id="release", sha256="release-hash",
                              teacher_source_sha256="source-hash", typed_contract=contract)
    plan = SimpleNamespace(data_release=release, production_scope="pilot",
                           plan_sha256="plan-hash", entries=(entry,),
                           dataset_id=contract.parent_dataset_id, path=tmp_path / "plan.json")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(pickle.dumps(dict(
        teacher_git_head="head", data_release_id=release.release_id,
        data_release_sha256=release.sha256,
        teacher_source_sha256=release.teacher_source_sha256,
        state={"counter": np.asarray([7, 8], dtype=np.int32)},
    )))
    metadata = dict(
        schema_version=teacher_shards.SHARD_SCHEMA_VERSION,
        plan_sha256=plan.plan_sha256, teacher_git_head="head",
        data_release_id=release.release_id, data_release_sha256=release.sha256,
        teacher_source_sha256=release.teacher_source_sha256,
        typed_contract_sha256=contract.sha256, production_scope="pilot",
        landpoint_id=entry.landpoint_id, year=1961, days=2,
        spatial_split="train", temporal_split="train",
        initialization_mode=entry.initialization_mode,
        bootstrap_day=None, transition_start_day=1, transition_count=2,
        preceding_checkpoint_sha256=None, input_hashes={},
        markov_contract_sha256=contract.parent_contract_sha256,
        shard=base.relative_to(worker).as_posix(), shard_sha256=_hash(base),
        checkpoint=checkpoint.relative_to(worker).as_posix(), checkpoint_sha256=_hash(checkpoint),
        typed_shard=typed.relative_to(worker).as_posix(), typed_shard_sha256=_hash(typed),
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    record = teacher_shards._compact_worker_shard_record(worker, entry, metadata)
    (worker / "manifest.json").write_text(json.dumps(dict(
        schema_version=teacher_shards.MANIFEST_SCHEMA_VERSION,
        plan_sha256=plan.plan_sha256, worker_index=0, worker_count=1,
        worker_assignment_strategy=teacher_shards.WORKER_ASSIGNMENT_STRATEGY,
        assigned_entries=[entry.key], completed_entries=[entry.key], complete=True,
        shards=[record],
    )), encoding="utf-8")
    manifest = root / "dataset_manifest.json"
    manifest.write_text(json.dumps(dict(
        schema_version=contract.dataset_manifest_schema,
        dataset_id=contract.parent_dataset_id, status="complete", provisional_teacher=False,
        teacher_git_head="head", markov_contract_sha256=contract.parent_contract_sha256,
        data_release=dict(typed_contract_sha256=contract.sha256,
                          transition_policy="single_pass_continuous_teacher"),
        shards=[dict(landpoint_id=entry.landpoint_id, year=1961,
                     spatial_split="train", temporal_split="train", day_count=2,
                     shard=base.relative_to(root).as_posix(), shard_sha256=_hash(base),
                     typed_shard=typed.relative_to(root).as_posix(), typed_shard_sha256=_hash(typed))],
    )), encoding="utf-8")
    yield SimpleNamespace(root=root, manifest=manifest, base=base, typed=typed,
                          metadata_path=metadata_path, checkpoint=checkpoint,
                          contract=contract, plan=plan, entry=entry, metadata=metadata,
                          worker=worker, paths=(base, typed, metadata_path, checkpoint))
    teacher_assets.close_asset_readers()


def test_public_indices_samples_windows_and_training_stream_use_packed_assets(teacher_fixture):
    f = teacher_fixture
    before = load_markov_shard(f.base)
    before_typed = read_typed_sidecar_shard(f.typed, contract=f.contract)
    original_manifest = f.manifest.read_bytes()
    _publish(f.root, f.entry.landpoint_id, f.paths)
    assert all(not path.exists() for path in f.paths)
    index = load_dataset_index(f.manifest)
    coherent = load_coherent_typed_sidecar_index(f.manifest)
    assert index.shards[0].path == f.base
    assert coherent.shards[0].path == f.typed
    samples = list(index.samples())
    windows = list(index.windows(2))
    assert len(samples) == 2 and len(windows) == 1
    np.testing.assert_array_equal(windows[0]["state_trajectory"], before.state_trajectory)
    np.testing.assert_array_equal(samples[1]["state"], before.sample(1)["state"])
    restored = read_typed_sidecar_shard(f.typed, contract=f.contract)
    for observed, expected in zip(restored[:2], before_typed[:2], strict=True):
        for name in expected:
            assert observed[name].tobytes() == expected[name].tobytes()
    assert len(list(coherent.samples())) == 2
    # The adapter here records exactly what the production generator supplies;
    # all base/typed assembly IO and day alignment remain production code.
    adapter = SimpleNamespace(typed_contract=f.contract,
                              contract=SimpleNamespace(sha256=f.contract.parent_contract_sha256),
                              assemble=lambda sample, **kwargs: (sample, kwargs))
    examples = list(coherent_process_training_examples(coherent, adapter))
    assert len(examples) == 2
    np.testing.assert_array_equal(examples[1][0]["state"], before.sample(1)["state"])
    assert set(examples[0][1]["supplemental_values"]) == set(before_typed[0])
    assert f.manifest.read_bytes() == original_manifest


def test_completion_checkpoint_and_status_restore_without_loose_objects(teacher_fixture, monkeypatch):
    f = teacher_fixture
    monkeypatch.setattr(teacher_shards, "_input_hashes", lambda *_: {})
    _publish(f.root, f.entry.landpoint_id, f.paths)
    restored = teacher_shards._completed_metadata(
        f.plan, f.worker, f.entry, plan_sha256=f.plan.plan_sha256,
        git_head="head", preceding_checkpoint_sha256=None,
    )
    assert restored == f.metadata
    state = teacher_shards._load_checkpoint(f.worker, restored)
    np.testing.assert_array_equal(state["counter"], [7, 8])
    progress = teacher_shards.audit_worker_progress(f.plan, output_root=f.root,
                                                   worker_count=1, verify_hashes=True)
    assert progress["complete"] is True
    assert progress["workers"][0]["errors"] == []


def test_post_acceptance_coherent_index_can_skip_only_file_scan(teacher_fixture):
    f = teacher_fixture
    f.typed.unlink()
    with pytest.raises(FileNotFoundError):
        load_coherent_typed_sidecar_index(f.manifest, verify_hashes=False)
    index = load_coherent_typed_sidecar_index(
        f.manifest, verify_files=False, verify_hashes=False,
    )
    assert index.shards[0].path == f.typed
    with pytest.raises(FileNotFoundError):
        list(index.samples())
    with pytest.raises(FileNotFoundError):
        load_coherent_typed_sidecar_index(f.manifest, verify_files=False)
    manifest = json.loads(f.manifest.read_text(encoding="utf-8"))
    manifest["shards"][0]["typed_shard"] = "../escape.npz"
    f.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes dataset root"):
        load_coherent_typed_sidecar_index(
            f.manifest, verify_files=False, verify_hashes=False,
        )


def test_packed_worker_aggregation_reads_original_metadata_and_day_inventory(
    teacher_fixture, monkeypatch,
):
    f = teacher_fixture
    monkeypatch.setattr(teacher_shards, "_input_hashes", lambda *_: {})
    # Aggregation validates the canonical metadata hash independently. This
    # compact metadata is deliberately a unit fixture, not a scientific contract.
    contract_metadata = {"schema_version": "fixture-markov", "state_width": 4}
    contract_hash = hashlib.sha256(teacher_assets.canonical(contract_metadata)).hexdigest()
    f.plan.data_release.markov_contract_sha256 = contract_hash
    f.metadata.update(markov_contract=contract_metadata, markov_contract_sha256=contract_hash)
    f.metadata_path.write_text(json.dumps(f.metadata), encoding="utf-8")
    record = teacher_shards._compact_worker_shard_record(f.worker, f.entry, f.metadata)
    worker_manifest = teacher_shards._worker_manifest(
        f.plan, worker_index=0, worker_count=1, assigned=(f.entry,),
        completed=(record,), git_head="head",
    )
    (f.worker / "manifest.json").write_text(json.dumps(worker_manifest), encoding="utf-8")
    _publish(f.root, f.entry.landpoint_id, f.paths)
    root_manifest_before = f.manifest.read_bytes()
    worker_manifest_before = (f.worker / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        teacher_shards.aggregate_workers(f.plan, output_root=f.root, worker_count=1)
    aggregated = teacher_shards.aggregate_workers(
        f.plan, output_root=f.root, worker_count=1, manifest_name="aggregation-check.json",
    )
    assert aggregated["shard_count"] == 1
    assert aggregated["markov_contract_sha256"] == contract_hash
    assert all(not path.exists() for path in f.paths)
    assert f.manifest.read_bytes() == root_manifest_before
    assert (f.worker / "manifest.json").read_bytes() == worker_manifest_before


def test_published_release_rejects_generation_before_runtime_lock_or_model(
    teacher_fixture, monkeypatch,
):
    f = teacher_fixture
    _publish(f.root, f.entry.landpoint_id, f.paths)
    before = {path: path.read_bytes() for path in (
        f.manifest, f.worker / "manifest.json",
    )}
    before_paths = set(f.root.rglob("*"))

    def unexpected_runtime(*args, **kwargs):
        pytest.fail("published release reached runtime/git/lock/model entry")

    monkeypatch.setattr(teacher_shards, "_clean_git_head", unexpected_runtime)
    monkeypatch.setattr(teacher_shards, "_worker_lock", unexpected_runtime)
    monkeypatch.setattr(teacher_shards, "_write_entry", unexpected_runtime)
    with pytest.raises(ValueError, match="immutable"):
        teacher_shards.generate_worker(f.plan, output_root=f.root,
                                       worker_index=0, worker_count=1)
    assert set(f.root.rglob("*")) == before_paths
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("worker_manifest", [False, True])
def test_low_level_json_writer_preserves_published_scientific_manifests(
    teacher_fixture, worker_manifest,
):
    f = teacher_fixture
    _publish(f.root, f.entry.landpoint_id, f.paths)
    target = f.worker / "manifest.json" if worker_manifest else f.manifest
    before = target.read_bytes()
    before_paths = set(f.root.rglob("*"))
    with pytest.raises(ValueError, match="immutable"):
        teacher_shards._atomic_json(target, {"overwrite": True})
    assert target.read_bytes() == before
    assert set(f.root.rglob("*")) == before_paths


@pytest.mark.parametrize("retire", [True, False])
def test_mapped_originals_cannot_be_regenerated(teacher_fixture, retire):
    f = teacher_fixture
    original = {path: path.read_bytes() for path in f.paths}
    _publish(f.root, f.entry.landpoint_id, f.paths, retire=retire)
    with pytest.raises(ValueError, match="immutable"):
        teacher_shards._write_entry(f.plan, f.entry, f.worker, git_head="head",
                                   previous_state=None, preceding_checkpoint_sha256=None)
    with pytest.raises(ValueError, match="immutable"):
        write_typed_sidecar_shard(f.typed, contract=f.contract,
                                 day_index=np.asarray([1, 2], dtype=np.int32),
                                 values={}, defined={})
    for writer, target in (
        (teacher_shards._atomic_npz, f.base),
        (teacher_shards._atomic_pickle, f.checkpoint),
        (teacher_shards._atomic_json, f.metadata_path),
    ):
        with pytest.raises(ValueError, match="immutable"):
            writer(target, {})
    if retire:
        assert all(not path.exists() for path in f.paths)
    else:
        assert all(path.read_bytes() == original[path] for path in f.paths)
