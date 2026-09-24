"""Arithmetic-only throughput scenarios for the shared daily/recurrent trainer.

No model execution or speedup measurement is performed here. Multi-device
results are ideal overlap lower bounds, and non-baseline schedules are not
training-equivalent to the measured joint update.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def load_measurements(path: Path) -> dict[str, float]:
    """Read the observed warm second-update timings from a raw runner report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    timings = report.get("warm_second_update_timing_seconds")
    wall = report.get("warm_second_update_wall_seconds")
    if not isinstance(timings, dict) or wall is None:
        raise ValueError("raw report is missing warm second-update timing fields")
    return {
        "daily_32": _finite_positive("daily_32", timings.get("daily_32_microbatch_4")),
        "T7": _finite_positive("T7", timings.get("T7")),
        "T30": _finite_positive("T30", timings.get("T30")),
        "optimizer": _finite_positive("optimizer", timings.get("optimizer")),
        "wall": _finite_positive("wall", wall),
    }


def build_estimate(
    measurements: dict[str, float],
    *,
    train_rows: int,
    total_rows: int,
    ready_daily_pool: int,
    planned_updates: int = 500,
    daily_batches: tuple[int, ...] = (32, 64, 128),
    recurrent_frequencies: tuple[int, ...] = (1, 2, 4),
) -> dict[str, Any]:
    """Build labeled measured, serial-arithmetic, and ideal-overlap cases."""
    train_rows = _positive_int("train_rows", train_rows)
    total_rows = _positive_int("total_rows", total_rows)
    ready_daily_pool = _positive_int("ready_daily_pool", ready_daily_pool)
    planned_updates = _positive_int("planned_updates", planned_updates)
    if train_rows > total_rows:
        raise ValueError("train_rows cannot exceed total_rows")
    if not daily_batches or any(_positive_int("daily batch", b) != b for b in daily_batches):
        raise ValueError("daily_batches must contain positive integers")
    if not recurrent_frequencies or any(_positive_int("recurrent frequency", f) != f for f in recurrent_frequencies):
        raise ValueError("recurrent_frequencies must contain positive integers")

    measurements = {
        key: _finite_positive(key, measurements.get(key))
        for key in ("daily_32", "T7", "T30", "optimizer", "wall")
    }

    daily32 = measurements["daily_32"]
    recurrent = measurements["T7"] + measurements["T30"]
    base_internal = daily32 + recurrent + measurements["optimizer"]
    base_unattributed = max(0.0, measurements["wall"] - base_internal)
    base_rate = 3600.0 / measurements["wall"]
    base_daily_draw_rate = 32 * base_rate
    ready_exposure_updates = math.ceil(ready_daily_pool / 32)
    train_exposure_updates = math.ceil(train_rows / 32)
    scenarios: list[dict[str, Any]] = []

    def add(name: str, batch: int, frequency: int, devices: int, daily_only: bool = False) -> None:
        daily_seconds = daily32 * batch / 32.0
        recurrent_seconds = 0.0 if daily_only else recurrent / frequency
        serial_internal = daily_seconds + recurrent_seconds + measurements["optimizer"]
        serial_wall = serial_internal + (base_unattributed if frequency == 1 and not daily_only else 0.0)
        if devices == 1:
            ideal_wall = serial_internal
            placement = "serial component arithmetic; not a new measurement"
        elif devices == 2:
            ideal_wall = max(daily_seconds, recurrent_seconds) + measurements["optimizer"]
            placement = "ideal daily-vs-recurrent overlap lower bound; excludes transfer/sync/contention"
        else:
            ideal_wall = max(daily_seconds, measurements["T7"] / frequency, measurements["T30"] / frequency) + measurements["optimizer"]
            placement = "ideal daily/T7/T30 component overlap lower bound; excludes transfer/sync/contention"
        # At reduced recurrent cadence the per-update wall is an average-work
        # scenario; actual recurrent steps remain full-cost spikes.
        if frequency > 1 and not daily_only:
            serial_wall = serial_internal
        chosen_seconds = measurements["wall"] if batch == 32 and frequency == 1 and devices == 1 and not daily_only else serial_wall
        selected_seconds = chosen_seconds if devices == 1 else ideal_wall
        method_changes = []
        if batch != 32:
            method_changes.append("daily batch/sample objective changes from measured batch32")
        if daily_only:
            method_changes.append("recurrent terms omitted; schedule/objective changes")
        elif frequency > 1:
            method_changes.append(f"recurrent cadence changes to every {frequency} updates")
        classification = "; ".join(method_changes) if method_changes else "same per-update objectives"
        if devices > 1:
            classification += "; ideal execution-placement projection"
        scenarios.append({
            "name": name,
            "daily_rows_per_update": batch,
            "recurrent_every_updates": None if daily_only else frequency,
            "devices": devices,
            "classification": classification,
            "method_changes": method_changes,
            "placement_assumption": placement,
            "estimated_serial_warm_seconds_per_update": serial_wall,
            "ideal_overlap_lower_bound_seconds_per_update": ideal_wall,
            "selected_seconds_per_update_for_exposure_arithmetic": selected_seconds,
            "updates_per_hour_at_selected_scenario": 3600.0 / selected_seconds,
            "daily_draws_per_hour": batch * 3600.0 / selected_seconds,
            "daily_draws_over_planned_updates": batch * planned_updates,
            "train_draw_equivalent_days_at_selected_scenario": train_rows / (batch * 3600.0 / selected_seconds) / 24.0,
            "planned_update_hours_excluding_startup": planned_updates * selected_seconds / 3600.0,
            "allocated_gpu_hours_excluding_startup": planned_updates * selected_seconds * devices / 3600.0,
            "pool_draw_fraction_not_unique_coverage": batch * planned_updates / ready_daily_pool,
        })

    for batch in daily_batches:
        add(f"single_gpu_joint_daily{batch}_recurrent_each_update", batch, 1, 1)
        add(f"ideal_dual_gpu_daily{batch}_vs_recurrent_overlap", batch, 1, 2)
        add(f"ideal_three_plus_gpu_daily{batch}_T7_T30_overlap", batch, 1, 3)
        for frequency in recurrent_frequencies:
            if frequency > 1:
                add(f"single_gpu_daily{batch}_recurrent_every_{frequency}_updates_average_work", batch, frequency, 1)
    add("daily_only_stage_example_batch32", 32, 1, 1, daily_only=True)

    return {
        "schema_version": "shared_training_throughput_scale_v1",
        "classification": "arithmetic scenarios, not training results",
        "observed_warm_second_update": {
            "daily_rows": 32,
            "daily_microbatch": 4,
            "component_seconds": {k: measurements[k] for k in ("daily_32", "T7", "T30", "optimizer")},
            "full_wall_seconds": measurements["wall"],
            "updates_per_hour": base_rate,
            "daily_draws_per_hour_repeated_including_repeats": 32 * base_rate,
            "five_hundred_warm_updates_hours_excluding_startup": 500 * measurements["wall"] / 3600.0,
            "interpretation": "one warm observation; not sustained throughput or a full-run wall-time estimate",
        },
        "data_counts": {
            "train_daily_rows": train_rows,
            "total_daily_rows_including_holdouts": total_rows,
            "ready_daily_pool_rows": ready_daily_pool,
            "planned_updates": planned_updates,
            "train_rows_for_one_count_equivalent_pass": train_rows,
            "ready_pool_rows_for_one_count_equivalent_pass": ready_daily_pool,
            "unique_dates_points_years": None,
            "coverage_warning": "draw counts include possible repeats and do not establish unique date/point/year coverage",
        },
        "exposure_equivalent_at_observed_warm_base": {
            "daily_draws_per_hour": base_daily_draw_rate,
            "ready_pool_draws": ready_daily_pool,
            "updates_to_equal_ready_pool_row_count_at_32_draws": ready_exposure_updates,
            "warm_hours_to_equal_ready_pool_row_count": ready_exposure_updates * measurements["wall"] / 3600.0,
            "train_row_draws": train_rows,
            "updates_to_equal_train_row_count_at_32_draws": train_exposure_updates,
            "warm_hours_to_equal_train_row_count": train_exposure_updates * measurements["wall"] / 3600.0,
            "meaning": "exposure-equivalent arithmetic only; repeats may occur, so this is not time to unique coverage",
        },
        "scenarios": scenarios,
        "limits": [
            "Only the single-GPU daily32/micro4 joint timing is observed; all other timings are arithmetic projections.",
            "Ideal multi-GPU bounds ignore communication, synchronization, contention, imbalance, compilation, IO, and launch overhead.",
            "Batch sizes 64/128 change the sampled daily minibatch and are not equivalent to daily32 updates; batch accumulation is not counted as a speed gain.",
            "Recurrent cadence changes and daily-only stages change the optimization schedule/objective and are not equivalent to joint training.",
            "Repeated exposures are not unique TRAIN coverage; counts alone cannot infer point/year/date coverage or sampling replacement behavior.",
            "Warm update arithmetic excludes cold compilation, data preparation, validation panels, checkpoint IO, and runtime variability.",
        ],
    }


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--total-rows", type=int, required=True)
    parser.add_argument("--ready-daily-pool", type=int, required=True)
    parser.add_argument("--planned-updates", type=int, default=500)
    parser.add_argument("--daily-batches", type=_csv_ints, default=(32, 64, 128))
    parser.add_argument("--recurrent-frequencies", type=_csv_ints, default=(1, 2, 4))
    args = parser.parse_args()
    result = build_estimate(
        load_measurements(args.raw_report), train_rows=args.train_rows,
        total_rows=args.total_rows, ready_daily_pool=args.ready_daily_pool,
        planned_updates=args.planned_updates, daily_batches=args.daily_batches,
        recurrent_frequencies=args.recurrent_frequencies,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
