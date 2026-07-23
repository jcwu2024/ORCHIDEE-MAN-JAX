from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from jax_orchidee.driver.paper_binding import expected_paper_domain_limits
from research.daily_coarse_graining import teacher_production, teacher_shards


def _population(tmp_path: Path, count: int = 8) -> Path:
    ids = [f"{2 * index + 1:03d}.0-{71 + 2 * index:03d}.0" for index in range(count)]
    rows = [
        {
            "landpoint_id": landpoint_id,
            "grid_x": float(index),
            "grid_y": float(index * index),
            "vcmax25": 40.0 + index,
            "maint_resp_slope_c": 0.01 * index,
            "alloc_min": 0.1 + 0.01 * index,
            "residence_time": 20.0 + 3.0 * index,
        }
        for index, landpoint_id in enumerate(ids)
    ]
    path = tmp_path / "population.json"
    path.write_text(
        json.dumps({"selected_landpoint_ids": ids, "selected": rows}),
        encoding="utf-8",
    )
    return path


def test_freeze_production_spec_is_deterministic_and_disjoint(tmp_path):
    population = _population(tmp_path)
    first = teacher_production.freeze_spec(
        population,
        tmp_path / "first.json",
        dataset_id="production-test",
        last_year=2010,
        validation_landpoints=2,
        test_landpoints=2,
    )
    second = teacher_production.freeze_spec(
        population,
        tmp_path / "second.json",
        dataset_id="production-test",
        last_year=2010,
        validation_landpoints=2,
        test_landpoints=2,
    )
    first_raw = json.loads(first.read_text(encoding="utf-8"))
    second_raw = json.loads(second.read_text(encoding="utf-8"))
    assert first_raw == second_raw
    assert first_raw["spatial_split_counts"] == {
        "test": 2,
        "train": 4,
        "validation": 2,
    }
    spec = teacher_production.load_production_spec(first)
    assert spec.temporal_split(1961) == "train"
    assert spec.temporal_split(2005) == "validation"
    assert spec.temporal_split(2008) == "test"


def test_freeze_can_bind_a_probe_subset_to_the_full_population(tmp_path):
    population = _population(tmp_path)
    raw = json.loads(population.read_text(encoding="utf-8"))
    selected = raw["selected_landpoint_ids"][:2]
    path = teacher_production.freeze_spec(
        population,
        tmp_path / "probe.json",
        dataset_id="probe",
        last_year=1961,
        validation_landpoints=0,
        test_landpoints=0,
        landpoint_ids=selected,
    )
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen["population_count"] == 8
    assert frozen["spatial_split_counts"] == {"test": 0, "train": 2, "validation": 0}
    assert [item["id"] for item in frozen["landpoints"]] == selected


def test_stage_and_plan_use_cold_start_without_prebuilt_checkpoints(
    monkeypatch,
    tmp_path,
):
    population = _population(tmp_path, count=3)
    spec_path = teacher_production.freeze_spec(
        population,
        tmp_path / "spec.json",
        dataset_id="two-point-probe",
        last_year=1961,
        validation_landpoints=1,
        test_landpoints=1,
    )
    spec = teacher_production.load_production_spec(spec_path)
    source_root = tmp_path / "source"
    references = {}
    for item in spec.landpoints:
        reference_dir = source_root / item.landpoint_id / "reference"
        reference_dir.mkdir(parents=True)
        for name in teacher_production.REQUIRED_REFERENCE_FILES:
            (reference_dir / name).write_bytes(f"{item.landpoint_id}:{name}".encode())
        used_run_def = source_root / item.landpoint_id / "used_run.def"
        limits = expected_paper_domain_limits(item.landpoint_id)
        used_run_def.write_text(
            "DT_SECHIBA = 1800\n"
            + "".join(f"{name} = {value}\n" for name, value in limits.items()),
            encoding="utf-8",
        )
        references[item.landpoint_id] = SimpleNamespace(
            output_dir=reference_dir,
            used_run_def=used_run_def,
        )
    monkeypatch.setattr(
        teacher_production,
        "resolve_paper_landpoint_reference",
        lambda _root, landpoint_id: references[landpoint_id],
    )
    staged = tmp_path / "staged"
    manifest = teacher_production.stage_assets(
        spec,
        reference_root=source_root,
        destination=staged,
    )
    assert manifest.is_file()
    assert teacher_production.verify_staged_assets(spec, staged) == {
        "dataset_id": "two-point-probe",
        "landpoint_count": 3,
        "verified_files": 18,
        "verified_bytes": sum(path.stat().st_size for path in staged.rglob("*") if path.is_file())
        - manifest.stat().st_size,
    }
    config = tmp_path / "teacher.yaml"
    config.write_text("test: true\n", encoding="utf-8")
    plan_path = teacher_production.build_generation_plan(
        spec,
        asset_root=staged,
        teacher_config=config,
        output_root=tmp_path / "output",
        plan_path=tmp_path / "plan.json",
        days=8,
    )
    plan = teacher_shards.load_plan(plan_path, require_inputs=True)
    assert len(plan.entries) == 3
    assert all(entry.initialization_mode == teacher_shards.COLD_START_BOOTSTRAP for entry in plan.entries)
    assert all(entry.acceptance_checkpoint is None for entry in plan.entries)
    assert all(entry.days == 8 for entry in plan.entries)
    assert len({entry.run_def for entry in plan.entries}) == 3
