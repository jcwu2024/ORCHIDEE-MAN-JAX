import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "outputs/shared-split-1c70fa4-full-timing.json"
SPEC = importlib.util.spec_from_file_location(
    "estimate_shared_training_scale", ROOT / "scripts/dev/estimate_shared_training_scale.py"
)
assert SPEC is not None and SPEC.loader is not None
SCALE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCALE)
build_estimate = SCALE.build_estimate
load_measurements = SCALE.load_measurements


def test_estimate_keeps_measured_baseline_and_exposure_accounting_distinct():
    measurements = load_measurements(RAW)
    result = build_estimate(
        measurements, train_rows=8_591_565, total_rows=12_208_581,
        ready_daily_pool=346_539, planned_updates=500,
    )
    observed = result["observed_warm_second_update"]
    assert observed["full_wall_seconds"] == pytest.approx(16.473257306963205)
    assert observed["updates_per_hour"] == pytest.approx(218.55, rel=0.001)
    assert observed["daily_draws_per_hour_repeated_including_repeats"] == pytest.approx(6993.6, rel=0.001)
    counts = result["data_counts"]
    assert counts["train_daily_rows"] == 8_591_565
    assert counts["total_daily_rows_including_holdouts"] == 12_208_581
    assert counts["unique_dates_points_years"] is None
    base = next(s for s in result["scenarios"] if s["name"] == "single_gpu_joint_daily32_recurrent_each_update")
    assert base["daily_draws_over_planned_updates"] == 16_000
    assert base["pool_draw_fraction_not_unique_coverage"] == pytest.approx(16_000 / 346_539)
    assert base["estimated_serial_warm_seconds_per_update"] == pytest.approx(observed["full_wall_seconds"])


def test_ideal_parallel_bounds_do_not_claim_measured_speed_or_reduce_gpu_hours():
    result = build_estimate(load_measurements(RAW), train_rows=100, total_rows=120,
                            ready_daily_pool=90, planned_updates=10)
    dual = next(s for s in result["scenarios"] if s["name"] == "ideal_dual_gpu_daily32_vs_recurrent_overlap")
    triple = next(s for s in result["scenarios"] if s["name"] == "ideal_three_plus_gpu_daily32_T7_T30_overlap")
    assert dual["classification"] == "same per-update objectives; ideal execution-placement projection"
    assert dual["ideal_overlap_lower_bound_seconds_per_update"] == pytest.approx(10.17891046, abs=0.01)
    assert triple["ideal_overlap_lower_bound_seconds_per_update"] < dual["ideal_overlap_lower_bound_seconds_per_update"]
    assert triple["allocated_gpu_hours_excluding_startup"] > triple["planned_update_hours_excluding_startup"]
    assert "ideal" in dual["placement_assumption"]
    assert any(s["name"] == "ideal_dual_gpu_daily64_vs_recurrent_overlap" for s in result["scenarios"])
    assert any(s["name"] == "ideal_three_plus_gpu_daily128_T7_T30_overlap" for s in result["scenarios"])


def test_batch_accumulation_and_schedule_changes_are_not_mislabeled_equivalent():
    result = build_estimate(load_measurements(RAW), train_rows=100, total_rows=120,
                            ready_daily_pool=90, planned_updates=10)
    batch64 = next(s for s in result["scenarios"] if s["name"] == "single_gpu_joint_daily64_recurrent_each_update")
    recurrent2 = next(s for s in result["scenarios"] if s["name"].endswith("recurrent_every_2_updates_average_work"))
    daily_only = next(s for s in result["scenarios"] if s["name"] == "daily_only_stage_example_batch32")
    assert "daily batch/sample objective changes" in batch64["classification"]
    assert "recurrent cadence changes" in recurrent2["classification"]
    assert "recurrent terms omitted" in daily_only["classification"]
    assert batch64["daily_draws_over_planned_updates"] == 640
    combo = next(
        s for s in result["scenarios"]
        if s["name"] == "single_gpu_daily64_recurrent_every_2_updates_average_work"
    )
    assert "daily batch/sample objective changes" in combo["classification"]
    assert "recurrent cadence changes" in combo["classification"]
    assert len(combo["method_changes"]) == 2
    assert combo["train_draw_equivalent_days_at_selected_scenario"] == pytest.approx(
        100 / combo["daily_draws_per_hour"] / 24
    )
    assert "every 2 updates" in recurrent2["classification"]
    assert "terms omitted" in daily_only["classification"]


def test_rejects_holdout_count_inversion_and_invalid_raw_timing(tmp_path):
    with pytest.raises(ValueError, match="cannot exceed"):
        build_estimate(load_measurements(RAW), train_rows=11, total_rows=10, ready_daily_pool=5)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"warm_second_update_timing_seconds": {"daily_32_microbatch_4": 0}}))
    with pytest.raises(ValueError, match="missing warm"):
        load_measurements(bad)
