from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.daily_coarse_graining import teacher_pilot, teacher_shards


def _write_spec(tmp_path: Path) -> Path:
    payload = {
        "schema_version": teacher_pilot.SPEC_SCHEMA_VERSION,
        "pilot_id": "test-pilot",
        "first_year": 1961,
        "last_year": 1963,
        "block_size": 7,
        "temporal_splits": {
            "train": [1961, 1961],
            "validation": [1962, 1962],
            "test": [1963, 1963],
        },
        "landpoints": [
            {"id": "001.0-071.0", "spatial_split": "train", "role": "wet"},
            {"id": "319.0-057.0", "spatial_split": "test", "role": "snow"},
        ],
        "required_reference_files": [
            "driver_start.nc",
            "sechiba_start.nc",
            "stomate_start.nc",
            "stomate_restart.nc",
        ],
        "checkpoint_name": "paper_driver_1961_year_end_state.pkl",
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_pilot_has_disjoint_space_and_complete_time_partitions():
    root = Path(__file__).resolve().parents[2]
    spec = teacher_pilot.load_pilot_spec(root / "manifests/coarse_graining/daily_teacher_pilot_v2.json")
    counts = {split: sum(item.spatial_split == split for item in spec.landpoints) for split in teacher_shards.SPLITS}
    assert counts == {"train": 8, "validation": 2, "test": 2}
    assert spec.last_year - spec.first_year + 1 == 50
    assert [spec.temporal_split(year) for year in (1961, 2005, 2008)] == [
        "train",
        "validation",
        "test",
    ]


def test_stage_verify_and_plan_roundtrip(monkeypatch, tmp_path):
    spec = teacher_pilot.load_pilot_spec(_write_spec(tmp_path))
    source_root = tmp_path / "sources"
    checkpoint_root = tmp_path / "checkpoints"
    references = {}
    for item in spec.landpoints:
        output = source_root / item.landpoint_id / "reference"
        output.mkdir(parents=True)
        for name in spec.required_reference_files:
            (output / name).write_bytes(f"{item.landpoint_id}:{name}".encode())
        used_run_def = source_root / item.landpoint_id / "used_run.def"
        used_run_def.write_text("DT_SECHIBA = 1800\n", encoding="utf-8")
        checkpoint = checkpoint_root / item.landpoint_id / "compiled_checkpoints"
        checkpoint.mkdir(parents=True)
        (checkpoint / spec.checkpoint_name).write_bytes(b"checkpoint")
        references[item.landpoint_id] = SimpleNamespace(
            output_dir=output,
            used_run_def=used_run_def,
        )
    monkeypatch.setattr(
        teacher_pilot,
        "resolve_paper_landpoint_reference",
        lambda _root, landpoint_id: references[landpoint_id],
    )

    staged = tmp_path / "staged"
    manifest = teacher_pilot.stage_assets(
        spec,
        reference_root=source_root,
        checkpoint_root=checkpoint_root,
        destination=staged,
    )
    assert manifest.is_file()
    verified = teacher_pilot.verify_staged_assets(spec, staged)
    assert verified["landpoint_count"] == 2
    assert verified["verified_files"] == 12

    config = tmp_path / "teacher.yaml"
    config.write_text("test: true\n", encoding="utf-8")
    plan_path = teacher_pilot.build_generation_plan(
        spec,
        asset_root=staged,
        teacher_config=config,
        output_root=tmp_path / "output",
        plan_path=tmp_path / "plan.json",
    )
    plan = teacher_shards.load_plan(plan_path, require_inputs=True)
    assert len(plan.entries) == 6
    assert sum(entry.state_cache is not None for entry in plan.entries) == 0
    cold_entries = [
        entry
        for entry in plan.entries
        if entry.initialization_mode == teacher_shards.COLD_START_BOOTSTRAP
    ]
    assert len(cold_entries) == 2
    assert all(entry.year == 1961 for entry in cold_entries)
    assert all(entry.acceptance_checkpoint is not None for entry in cold_entries)
    assert {(entry.year, entry.temporal_split) for entry in plan.entries} == {
        (1961, "train"),
        (1962, "validation"),
        (1963, "test"),
    }

    copied = staged / "landpoints/001.0-071.0/checkpoint.pkl"
    copied.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        teacher_pilot.verify_staged_assets(spec, staged)
