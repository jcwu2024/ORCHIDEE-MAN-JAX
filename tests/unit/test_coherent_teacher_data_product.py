from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining import teacher_shards
from research.daily_coarse_graining.supervised_learnability_pilot import (
    CompactTrainingCaptureBlock,
)
from research.daily_coarse_graining.teacher_data_release import (
    SCHEMA_VERSION as RELEASE_SCHEMA_VERSION,
)
from research.daily_coarse_graining.teacher_data_release import (
    canonical_sha256,
    freeze_teacher_data_release,
    load_teacher_data_release,
    python_source_sha256,
    python_tree_sha256,
    sha256_file,
)
from research.daily_coarse_graining.typed_sidecar import (
    COHERENT_CONTRACT_PATH,
    LEGACY_COHERENT_CONTRACT_PATH,
    compiled_daily_flux_fields,
    load_coherent_typed_sidecar_index,
    load_typed_sidecar_contract,
    write_typed_sidecar_shard,
)


def _release_manifest(tmp_path: Path) -> Path:
    contract = load_typed_sidecar_contract(COHERENT_CONTRACT_PATH)
    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()
    (teacher_root / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("model: test\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}\n", encoding="utf-8")
    producer = tmp_path / "producer.py"
    producer.write_text("def produce():\n    return 1\n", encoding="utf-8")
    raw = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "frozen_pre_production",
        "release_id": "test-release-v1",
        "dataset_id": contract.parent_dataset_id,
        "teacher_source": {
            "path": str(teacher_root),
            "python_tree_sha256": python_tree_sha256(teacher_root),
            "hash_mode": "utf8_lf",
        },
        "runtime": {
            "config": {"path": str(config), "sha256": sha256_file(config)},
            "pft_catalog": {
                "path": str(catalog),
                "sha256": sha256_file(catalog),
            },
        },
        "contracts": {
            "markov_contract_sha256": contract.parent_contract_sha256,
            "typed_supplement": {
                "path": str(contract.path),
                "sha256": sha256_file(contract.path),
                "contract_sha256": contract.sha256,
            },
        },
        "producer_source_files": [
            {
                "path": str(producer),
                "sha256": python_source_sha256(producer),
                "hash_mode": "utf8_lf",
            }
        ],
        "generation": {
            "transition_policy": "single_pass_continuous_teacher",
            "artifacts": ["base_shard", "typed_shard", "year_end_checkpoint"],
            "dataset_manifest_schema": contract.dataset_manifest_schema,
        },
    }
    raw["release_sha256"] = canonical_sha256(raw)
    path = tmp_path / "release.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_data_release_binds_teacher_config_contract_and_producer_sources(tmp_path):
    path = _release_manifest(tmp_path)
    release = load_teacher_data_release(path)

    assert release.release_id == "test-release-v1"
    assert release.typed_contract.labels

    raw = json.loads(path.read_text(encoding="utf-8"))
    producer = Path(raw["producer_source_files"][0]["path"])
    producer.write_text("def produce():\n    return 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="producer source hash drift"):
        load_teacher_data_release(path)


def test_release_freezer_is_deterministic_and_rejects_historical_contract(tmp_path):
    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()
    (teacher_root / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("model: test\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}\n", encoding="utf-8")
    producer = tmp_path / "producer.py"
    producer.write_text("def produce():\n    return 1\n", encoding="utf-8")
    first = freeze_teacher_data_release(
        tmp_path / "first.json",
        release_id="test-release-v1",
        teacher_source=teacher_root,
        teacher_config=config,
        pft_catalog=catalog,
        producer_sources=(producer,),
    )
    second = freeze_teacher_data_release(
        tmp_path / "second.json",
        release_id="test-release-v1",
        teacher_source=teacher_root,
        teacher_config=config,
        pft_catalog=catalog,
        producer_sources=(producer,),
    )

    assert json.loads(first.read_text()) == json.loads(second.read_text())
    with pytest.raises(ValueError, match="coherent typed contract"):
        freeze_teacher_data_release(
            tmp_path / "legacy.json",
            release_id="invalid-release",
            teacher_source=teacher_root,
            teacher_config=config,
            pft_catalog=catalog,
            typed_contract=(
                Path(__file__).resolve().parents[2] / "manifests" / "coarse_graining" / "daily_typed_sidecar_v1.json"
            ),
            producer_sources=(producer,),
        )


def test_python_source_identity_is_portable_across_git_line_endings(tmp_path):
    windows = tmp_path / "windows.py"
    linux = tmp_path / "linux.py"
    windows.write_bytes(b"VALUE = 1\r\n\r\n")
    linux.write_bytes(b"VALUE = 1\n\n")

    assert python_source_sha256(windows) == python_source_sha256(linux)


def test_compiled_daily_flux_projection_uses_frozen_field_order():
    contract = load_typed_sidecar_contract(COHERENT_CONTRACT_PATH)
    roots = {name: {} for name in ("water", "energy", "ok_leak")}
    for index, field in enumerate(contract.fields):
        family, name = field.path.split(".", maxsplit=1)
        roots[family][name] = np.asarray([index], dtype=np.float64)
    labels = SimpleNamespace(
        sechiba=SimpleNamespace(
            water=SimpleNamespace(**roots["water"]),
            energy=SimpleNamespace(**roots["energy"]),
        ),
        ok_leak=SimpleNamespace(**roots["ok_leak"]),
    )

    values = compiled_daily_flux_fields(labels, contract=contract)

    assert [float(value) for value in values] == list(range(len(contract.fields)))


def test_compact_assembly_keeps_typed_rows_out_of_the_base_contract(
    monkeypatch,
):
    typed_contract = load_typed_sidecar_contract(COHERENT_CONTRACT_PATH)
    markov_contract = SimpleNamespace(
        discrete_leaves=(),
        native_forcing="native-spec",
    )
    typed_rows = {field.path: np.zeros((2, *field.feature_shape), dtype=np.float64) for field in typed_contract.fields}
    block = CompactTrainingCaptureBlock(
        year=1961,
        day_indices=np.asarray([2, 3], dtype=np.int32),
        state_rows=np.zeros((2, 3), dtype=np.float64),
        discrete_rows={},
        fast_day_targets=np.zeros((2, 2), dtype=np.float64),
        diagnostics=np.zeros((2, 1), dtype=np.float64),
        final_state="final",
        contract=markov_contract,
        typed_capture_rows=typed_rows,
    )
    monkeypatch.setattr(
        teacher_shards,
        "_pack_condition_groups",
        lambda *_args, **_kwargs: (np.asarray([1.0]), ()),
    )
    monkeypatch.setattr(teacher_shards, "_parameter_groups", lambda _context: ())
    monkeypatch.setattr(teacher_shards, "_landpoint_static_groups", lambda _context: ())
    monkeypatch.setattr(teacher_shards, "_annual_condition_groups", lambda _context, **_kwargs: ())
    monkeypatch.setattr(
        teacher_shards,
        "extract_state",
        lambda _state, _contract: (np.ones(3, dtype=np.float64), {}),
    )
    monkeypatch.setattr(
        teacher_shards,
        "native_forcing_days",
        lambda *_args, **_kwargs: (
            np.zeros((2, 5, 1), dtype=np.float64),
            np.zeros((2, 5), dtype=np.int32),
            "native-spec",
        ),
    )

    arrays, observed_contract, final_state = teacher_shards.build_shard_arrays_from_compact_blocks(
        (block,),
        SimpleNamespace(),
        typed_capture_contract=typed_contract,
    )

    assert observed_contract is markov_contract
    assert final_state == "final"
    assert not any(name.startswith("__typed_capture__.") for name in arrays if name == "day_index")
    assert all(f"__typed_capture__.{field.path}" in arrays for field in typed_contract.fields)

    incomplete = block._replace(
        typed_capture_rows={name: value for name, value in typed_rows.items() if name != next(iter(typed_rows))}
    )
    with pytest.raises(ValueError, match="missing typed capture fields"):
        teacher_shards.build_shard_arrays_from_compact_blocks(
            (incomplete,),
            SimpleNamespace(),
            typed_capture_contract=typed_contract,
        )


@pytest.mark.parametrize(
    "contract_path",
    (LEGACY_COHERENT_CONTRACT_PATH, COHERENT_CONTRACT_PATH),
)
def test_coherent_reader_uses_one_manifest_for_base_and_typed_shards(
    tmp_path,
    contract_path,
):
    contract = load_typed_sidecar_contract(contract_path)
    base = tmp_path / "base.npz"
    np.savez(base, marker=np.asarray(1, dtype=np.int32))
    day_index = np.asarray([2, 3], dtype=np.int32)
    values = {field.path: np.zeros((2, *field.feature_shape), dtype=np.float64) for field in contract.fields}
    defined = {name: np.ones(value.shape, dtype=np.bool_) for name, value in values.items()}
    typed = tmp_path / "typed.npz"
    write_typed_sidecar_shard(
        typed,
        contract=contract,
        day_index=day_index,
        values=values,
        defined=defined,
        layouts={field.path: "dense" for field in contract.fields},
    )
    manifest = {
        "schema_version": contract.dataset_manifest_schema,
        "dataset_id": contract.parent_dataset_id,
        "status": "complete",
        "provisional_teacher": False,
        "teacher_git_head": "head",
        "markov_contract_sha256": contract.parent_contract_sha256,
        "data_release": {
            "typed_contract_sha256": contract.sha256,
            "transition_policy": "single_pass_continuous_teacher",
        },
        "shards": [
            {
                "landpoint_id": "001.0-071.0",
                "year": 1961,
                "spatial_split": "train",
                "temporal_split": "train",
                "shard": base.name,
                "shard_sha256": sha256_file(base),
                "typed_shard": typed.name,
                "typed_shard_sha256": sha256_file(typed),
                "day_count": 2,
            }
        ],
    }
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    index = load_coherent_typed_sidecar_index(path, contract_path=contract_path)

    assert index.parent_manifest_sha256 == index.sidecar_manifest_sha256
    assert index.shards[0].parent.path == base
    assert index.shards[0].path == typed


def test_coherent_worker_transaction_and_restart_chain_aggregate(tmp_path):
    release_path = _release_manifest(tmp_path)
    release_raw = json.loads(release_path.read_text(encoding="utf-8"))
    config = Path(release_raw["runtime"]["config"]["path"])
    run_def = tmp_path / "used_run.def"
    run_def.write_text(
        "LIMIT_WEST=70\nLIMIT_EAST=72\nLIMIT_SOUTH=0\nLIMIT_NORTH=2\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    for name in teacher_shards.REFERENCE_INPUT_NAMES:
        (reference / name).write_bytes(name.encode())
    state_cache = tmp_path / "state.pkl"
    state_cache.write_bytes(b"state")
    entries = [
        {
            "landpoint_id": "001.0-071.0",
            "year": 1961,
            "days": 1,
            "spatial_split": "train",
            "temporal_split": "train",
            "run_def": str(run_def),
            "reference_run_dir": str(reference),
            "state_cache": str(state_cache),
        },
        {
            "landpoint_id": "001.0-071.0",
            "year": 1962,
            "days": 1,
            "spatial_split": "train",
            "temporal_split": "train",
            "run_def": str(run_def),
            "reference_run_dir": str(reference),
        },
    ]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": teacher_shards.COHERENT_PLAN_SCHEMA_VERSION,
                "dataset_id": release_raw["dataset_id"],
                "teacher_config": str(config),
                "output_root": str(tmp_path / "dataset"),
                "block_size": 7,
                "data_release": str(release_path),
                "production_scope": "pilot",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    plan = teacher_shards.load_plan(plan_path)
    contract_metadata = {"schema_version": "test_markov_contract", "state_width": 1}
    contract_hash = hashlib.sha256(teacher_shards._canonical_json(contract_metadata)).hexdigest()
    plan = replace(
        plan,
        data_release=replace(
            plan.data_release,
            markov_contract_sha256=contract_hash,
        ),
    )
    worker_root = plan.output_root / "workers" / "worker-000-of-001"
    records = []
    preceding = None
    contract = plan.data_release.typed_contract
    for entry in plan.entries:
        shard, metadata_path, checkpoint = teacher_shards._entry_paths(worker_root, entry)
        typed = teacher_shards._entry_typed_path(worker_root, entry)
        teacher_shards._atomic_npz(
            shard,
            {
                "day_index": np.asarray([1], dtype=np.int32),
                "state_trajectory": np.zeros((2, 1), dtype=np.float64),
            },
        )
        values = {field.path: np.zeros((1, *field.feature_shape), dtype=np.float64) for field in contract.fields}
        write_typed_sidecar_shard(
            typed,
            contract=contract,
            day_index=np.asarray([1], dtype=np.int32),
            values=values,
            defined={name: np.ones(value.shape, dtype=np.bool_) for name, value in values.items()},
            layouts={field.path: "dense" for field in contract.fields},
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{entry.year}".encode())
        metadata = {
            "schema_version": teacher_shards.SHARD_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "teacher_git_head": "teacher-head",
            "data_release_id": plan.data_release.release_id,
            "data_release_sha256": plan.data_release.sha256,
            "teacher_source_sha256": plan.data_release.teacher_source_sha256,
            "typed_contract_sha256": contract.sha256,
            "production_scope": "pilot",
            "landpoint_id": entry.landpoint_id,
            "year": entry.year,
            "days": entry.days,
            "initialization_mode": entry.initialization_mode,
            "bootstrap_day": None,
            "transition_start_day": 1,
            "transition_count": 1,
            "spatial_split": entry.spatial_split,
            "temporal_split": entry.temporal_split,
            "preceding_checkpoint_sha256": preceding,
            "input_hashes": teacher_shards._input_hashes(plan, entry),
            "markov_contract": contract_metadata,
            "markov_contract_sha256": contract_hash,
            "shard": teacher_shards._relative(shard, worker_root),
            "shard_sha256": teacher_shards._sha256_file(shard),
            "checkpoint": teacher_shards._relative(checkpoint, worker_root),
            "checkpoint_sha256": teacher_shards._sha256_file(checkpoint),
            "typed_shard": teacher_shards._relative(typed, worker_root),
            "typed_shard_sha256": teacher_shards._sha256_file(typed),
        }
        teacher_shards._atomic_json(metadata_path, metadata)
        assert (
            teacher_shards._completed_metadata(
                plan,
                worker_root,
                entry,
                plan_sha256=plan.plan_sha256,
                git_head="teacher-head",
                preceding_checkpoint_sha256=preceding,
            )
            == metadata
        )
        records.append(teacher_shards._compact_worker_shard_record(worker_root, entry, metadata))
        preceding = metadata["checkpoint_sha256"]

    typed_last = teacher_shards._entry_typed_path(worker_root, plan.entries[-1])
    hidden_typed = typed_last.with_suffix(".missing")
    typed_last.replace(hidden_typed)
    assert (
        teacher_shards._completed_metadata(
            plan,
            worker_root,
            plan.entries[-1],
            plan_sha256=plan.plan_sha256,
            git_head="teacher-head",
            preceding_checkpoint_sha256=records[0]["checkpoint_sha256"],
        )
        is None
    )
    hidden_typed.replace(typed_last)

    teacher_shards._atomic_json(
        worker_root / "manifest.json",
        teacher_shards._worker_manifest(
            plan,
            worker_index=0,
            worker_count=1,
            assigned=plan.entries,
            completed=records,
            git_head="teacher-head",
        ),
    )
    manifest = teacher_shards.aggregate_workers(
        plan,
        output_root=plan.output_root,
        worker_count=1,
    )

    assert manifest["schema_version"] == teacher_shards.COHERENT_DATASET_SCHEMA_VERSION
    assert manifest["provisional_teacher"] is False
    assert manifest["data_release"]["transition_policy"] == "single_pass_continuous_teacher"
    assert manifest["shards"][1]["preceding_checkpoint_sha256"] == (manifest["shards"][0]["checkpoint_sha256"])
