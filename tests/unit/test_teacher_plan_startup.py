"""Repeated annual paths retain ordinary resolution and validation semantics."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from research.daily_coarse_graining import teacher_shards as teacher


def _plan(tmp_path, **overrides):
    raw = {
        "schema_version": teacher.SCHEMA_VERSION,
        "dataset_id": "startup-test",
        "teacher_config": "config.yaml",
        "output_root": "dataset",
        "entries": [{
            "landpoint_id": "001.0-071.0", "year": year,
            "spatial_split": "train", "temporal_split": "train",
            "run_def": "point/run.def", "reference_run_dir": "point/reference",
            "initialization_mode": (teacher.COLD_START_BOOTSTRAP if year == 1961
                                    else teacher.YEAR_START_CHECKPOINT),
        } for year in range(1961, 2011)],
        **overrides,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, raw


def test_plan_resolves_each_unique_input_once_and_keeps_plan_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher, "ROOT", tmp_path)
    path, raw = _plan(tmp_path)
    original = Path.resolve
    calls = Counter()

    def counted(value, *args, **kwargs):
        calls[str(value)] += 1
        return original(value, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted)
    plan = teacher.load_plan(path, require_inputs=False, verify_release_sources=False)
    assert len(plan.entries) == 50
    assert calls == Counter({str(path): 1, **{
        str(tmp_path / name): 1
        for name in ("config.yaml", "dataset", "point/run.def", "point/reference")
    }})
    assert plan.plan_sha256 == hashlib.sha256(teacher._canonical_json(raw)).hexdigest()
    assert all(entry.run_def == tmp_path / "point/run.def" for entry in plan.entries)
    # A subsequent call must resolve anew; there is no persistent path cache.
    teacher.load_plan(path)
    assert all(count == 2 for count in calls.values())


def test_plan_preserves_leaf_symlink_absolute_and_root_relative_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher, "ROOT", tmp_path)
    first, second = tmp_path / "first.def", tmp_path / "second.def"
    first.write_text("first")
    second.write_text("second")
    link = tmp_path / "run.def"
    try:
        link.symlink_to(first)
    except OSError:
        pytest.skip("file symlink creation is unavailable")
    path, raw = _plan(tmp_path)
    for entry in raw["entries"]:
        entry["run_def"] = str(link)  # absolute leaf symlink
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = teacher.load_plan(path)
    assert all(entry.run_def == first.resolve() for entry in plan.entries)
    assert all(entry.reference_run_dir == (tmp_path / "point/reference").resolve()
               for entry in plan.entries)
    link.unlink()
    link.symlink_to(second)
    assert teacher.load_plan(path).entries[0].run_def == second.resolve()


def test_plan_require_inputs_still_rejects_missing_repeated_input(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher, "ROOT", tmp_path)
    path, _ = _plan(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing generation inputs"):
        teacher.load_plan(path, require_inputs=True)


@pytest.mark.parametrize("field,value,message", [
    ("spatial_split", "validation", "leaks across spatial splits"),
    ("year", 1961, "duplicate plan entry"),
    ("year", 1963, "does not follow"),
])
def test_plan_repeated_paths_keep_entry_validation(tmp_path, monkeypatch, field, value, message):
    monkeypatch.setattr(teacher, "ROOT", tmp_path)
    path, raw = _plan(tmp_path)
    raw["entries"][1][field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        teacher.load_plan(path)


def test_plan_default_source_gate_is_forwarded_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher, "ROOT", tmp_path)
    path, _ = _plan(tmp_path, schema_version=teacher.COHERENT_PLAN_SCHEMA_VERSION,
                     data_release="release.json", production_scope="full")
    calls = []

    def reject_release(path, *, verify_sources):
        calls.append((path, verify_sources))
        raise ValueError("release source gate")

    monkeypatch.setattr(teacher, "load_teacher_data_release", reject_release)
    with pytest.raises(ValueError, match="release source gate"):
        teacher.load_plan(path)
    with pytest.raises(ValueError, match="release source gate"):
        teacher.load_plan(path, verify_release_sources=False)
    assert calls == [(tmp_path / "release.json", True), (tmp_path / "release.json", False)]
