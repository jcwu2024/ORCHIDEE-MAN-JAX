from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining import supervised_learnability_pilot as capture
from research.daily_coarse_graining import teacher_shards as shards


def _write_plan(tmp_path: Path, entries: list[dict], **overrides) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text("test: true\n", encoding="utf-8")
    payload = {
        "schema_version": shards.SCHEMA_VERSION,
        "dataset_id": "test-dataset",
        "teacher_config": str(config),
        "output_root": str(tmp_path / "dataset"),
        "block_size": 28,
        "entries": entries,
        **overrides,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(tmp_path: Path, landpoint: str, year: int, **overrides) -> dict:
    run_def = tmp_path / f"{landpoint}-{year}.def"
    run_def.write_text("test\n", encoding="utf-8")
    reference = tmp_path / f"reference-{landpoint}"
    reference.mkdir(exist_ok=True)
    cache = tmp_path / f"cache-{landpoint}-{year}.pkl"
    cache.write_bytes(b"cache")
    return {
        "landpoint_id": landpoint,
        "year": year,
        "spatial_split": "train",
        "temporal_split": "train",
        "run_def": str(run_def),
        "reference_run_dir": str(reference),
        "state_cache": str(cache),
        **overrides,
    }


def test_plan_freezes_spatial_and_temporal_splits(tmp_path):
    first = _entry(tmp_path, "001.0-071.0", 1961)
    second = _entry(
        tmp_path,
        "001.0-071.0",
        1962,
        spatial_split="validation",
        temporal_split="validation",
    )
    with pytest.raises(ValueError, match="leaks across spatial splits"):
        shards.load_plan(_write_plan(tmp_path, [first, second]))

    second = _entry(
        tmp_path,
        "003.0-071.0",
        1961,
        spatial_split="validation",
        temporal_split="test",
    )
    with pytest.raises(ValueError, match="leaks across temporal splits"):
        shards.load_plan(_write_plan(tmp_path, [first, second]))


def test_plan_requires_explicit_state_for_each_new_or_gapped_chain(tmp_path):
    first = _entry(tmp_path, "001.0-071.0", 1961)
    first.pop("state_cache")
    with pytest.raises(ValueError, match="requires state_cache"):
        shards.load_plan(_write_plan(tmp_path, [first]))

    first = _entry(tmp_path, "001.0-071.0", 1961)
    third = _entry(tmp_path, "001.0-071.0", 1963)
    third.pop("state_cache")
    with pytest.raises(ValueError, match="does not follow"):
        shards.load_plan(_write_plan(tmp_path, [first, third]))


def test_worker_assignment_keeps_complete_landpoint_chain_together(tmp_path):
    second_year = _entry(tmp_path, "001.0-071.0", 1962)
    second_year.pop("state_cache")
    entries = [
        _entry(tmp_path, "001.0-071.0", 1961),
        second_year,
        _entry(tmp_path, "003.0-071.0", 1961),
    ]
    plan = shards.load_plan(_write_plan(tmp_path, entries))
    selected = [
        shards.assigned_entries(plan, index, 4)
        for index in range(4)
    ]
    owners = {
        entry.landpoint_id: index
        for index, values in enumerate(selected)
        for entry in values
    }
    assert sum(len(values) for values in selected) == 3
    assert owners["001.0-071.0"] == shards.worker_for_landpoint("001.0-071.0", 4)
    assert [
        entry.year
        for entry in selected[owners["001.0-071.0"]]
        if entry.landpoint_id == "001.0-071.0"
    ] == [1961, 1962]


def test_generation_rejects_an_uncommitted_teacher_identity(monkeypatch):
    def output(command, **_kwargs):
        return "abc123\n" if command[1:3] == ["rev-parse", "HEAD"] else " M teacher.py\n"

    monkeypatch.setattr(shards.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        shards._clean_git_head()


def test_aggregate_requires_complete_hash_verified_worker_outputs(tmp_path):
    entry = _entry(tmp_path, "001.0-071.0", 1961)
    plan = shards.load_plan(_write_plan(tmp_path, [entry]))
    output_root = tmp_path / "dataset"
    worker_root = output_root / "workers" / "worker-000-of-001"
    shard_path = worker_root / "shards" / "sample.npz"
    checkpoint_path = worker_root / "checkpoints" / "sample.pkl"
    shards._atomic_npz(shard_path, {"value": np.asarray([1.0])})
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    shard = {
        "landpoint_id": "001.0-071.0",
        "year": 1961,
        "boundary_spec_sha256": "boundary",
        "preceding_checkpoint_sha256": None,
        "input_hashes": shards._input_hashes(plan, plan.entries[0]),
        "shard": "shards/sample.npz",
        "shard_sha256": shards._sha256_file(shard_path),
        "checkpoint": "checkpoints/sample.pkl",
        "checkpoint_sha256": shards._sha256_file(checkpoint_path),
    }
    shards._atomic_json(
        worker_root / "manifest.json",
        {
            "schema_version": shards.MANIFEST_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "teacher_git_head": "teacher",
            "worker_index": 0,
            "worker_count": 1,
            "shards": [shard],
        },
    )
    result = shards.aggregate_workers(plan, output_root=output_root, worker_count=1)
    assert result["status"] == "complete"
    assert result["shard_count"] == 1

    shard_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        shards.aggregate_workers(plan, output_root=output_root, worker_count=1)


def test_compiled_capture_uses_a_short_tail_block(monkeypatch):
    first_record = SimpleNamespace(
        half_hour_transition=SimpleNamespace(current_state="boundary-state")
    )
    monkeypatch.setattr(
        capture,
        "_capture_days",
        lambda **_kwargs: (("start",), ("forcing-1",), (first_record,), "state-1"),
    )
    monkeypatch.setattr(capture.jax, "device_get", lambda value: value)
    monkeypatch.setattr(
        capture,
        "_indexed_tree",
        lambda tree, index: tree[index],
    )
    monkeypatch.setattr(
        capture,
        "_compiled_training_record",
        lambda **kwargs: SimpleNamespace(
            day_index=kwargs["day_index"],
            expected_result=SimpleNamespace(day_end_state=kwargs["day_end_state"]),
        ),
    )
    monkeypatch.setattr(
        capture.teacher,
        "fast_state_from_previous_packet",
        lambda value: SimpleNamespace(spec="state-spec", values_by_component=f"values:{value}"),
    )
    monkeypatch.setattr(
        capture.teacher,
        "previous_packet_from_fast_state",
        lambda value: value.values_by_component,
    )
    monkeypatch.setattr(
        capture.teacher,
        "DriverFastStateBundle",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        capture.teacher,
        "_paper_1961_later_day_transition_inputs",
        lambda **_kwargs: SimpleNamespace(
            hydrol_runtime_static_tables="tables",
            compiled_base_payload_template="payload",
        ),
    )
    for name, value in {
        "_paper_daily_carbon_static_dispatch": "daily",
        "_compiled_stomate_parameter_values": "stomate-parameters",
        "_compiled_landpoint_payload": "landpoint",
        "_compiled_diffuco_parameter_values": "diffuco-parameters",
        "_compiled_hydrol_table_arrays": "hydrol-arrays",
    }.items():
        monkeypatch.setattr(capture.teacher, name, lambda *_args, _value=value, **_kwargs: _value)
    monkeypatch.setattr(
        capture.teacher,
        "read_stomate_restart_season_state",
        lambda *_args, **_kwargs: SimpleNamespace(_asdict=lambda: {"provenance": "test"}),
    )
    monkeypatch.setattr(
        capture.teacher,
        "_paper_compiled_forcing_day",
        lambda _context, **kwargs: np.asarray(kwargs["start_tstep"]),
    )
    compiled_sizes = []

    def compile_block(_config, **kwargs):
        size = int(np.asarray(kwargs["block_day_numbers"]).size)
        compiled_sizes.append(size)

        def executable(_initial, block_forcing, block_days, *_args):
            count = int(np.asarray(block_days).size)
            return (
                f"final-{int(block_days[-1])}",
                ([f"boundary-{index}" for index in range(count)], [f"state-{index}" for index in range(count)]),
            )

        return executable, "state-spec"

    monkeypatch.setattr(
        capture.teacher,
        "_paper_compiled_later_day_block_executable",
        compile_block,
    )
    context = SimpleNamespace(
        runtime=SimpleNamespace(dt_stomate=86400.0, dt_sechiba=1800.0),
        first_step_restart_state=SimpleNamespace(stomate="stomate", stomate_input="input"),
    )
    states, forcings, records, final_state = capture._capture_days_compiled_blocks(
        config_path=Path("config"),
        context=context,
        previous_state="initial",
        year=1964,
        start_day=1,
        days=5,
        block_size=3,
    )
    assert compiled_sizes == [3, 1]
    assert len(states) == len(forcings) == len(records) == 5
    assert final_state == "final-5"
