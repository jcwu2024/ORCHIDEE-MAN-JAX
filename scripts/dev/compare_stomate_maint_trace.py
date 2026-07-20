from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_day_scaffold,
    paper_1961_driver_cold_start_runtime_day_result,
    paper_1961_driver_day_scaffold,
)
from jax_orchidee.stomate.carbon_kernels import NPARTS  # noqa: E402
from jax_orchidee.trace.server_1961 import read_server_records  # noqa: E402


def _pft14_parts(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[0, 13, :NPARTS]


def _pft14_total(value) -> float:
    return float(np.asarray(value, dtype=np.float64)[0, 13])


def _trace_groups(*, start_itime: int, steps: int) -> dict[int, list[dict[str, object]]]:
    groups: dict[int, list[dict[str, object]]] = {}
    # The trace is ordered by itime/PFT/part for the single-point paper case.
    # Bound the read to the exact day-window record count instead of loading
    # the full year.
    end_itime = int(start_itime) + int(steps) - 1
    scan_limit = end_itime * 14 * NPARTS
    for row in read_server_records("stomate_maint", tags="maint_after", limit=scan_limit):
        itime = int(row["itime"])
        if itime < start_itime or itime > end_itime:
            continue
        if int(row["jv"]) != 14:
            continue
        groups.setdefault(itime, []).append(row)
    return groups


def _compare_array(jax_value: np.ndarray, trace_value: np.ndarray) -> dict[str, object]:
    delta = jax_value - trace_value
    return {
        "jax": jax_value.tolist(),
        "reference": trace_value.tolist(),
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare local STOMATE maintenance respiration fold against Fortran trace."
    )
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument(
        "--start-itime",
        type=int,
        default=1,
        help="Fortran 1-based itime to compare. Use 49 for the first step of day 2.",
    )
    parser.add_argument("--year", type=int, default=1961)
    parser.add_argument(
        "--initial-state",
        choices=("reference-start", "cold-start"),
        default="cold-start",
        help=(
            "cold-start matches server_1961_trace_full RESTART_FILEIN=NONE traces; "
            "reference-start is diagnostic only unless traces were generated from reference starts."
        ),
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument(
        "--run-def",
        type=Path,
        default=ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-12,
        help="Absolute tolerance for trace_matches_runner.",
    )
    args = parser.parse_args()

    if args.start_itime < 1:
        parser.error("--start-itime must be positive")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.initial_state != "cold-start" and args.start_itime != 1:
        parser.error("later-day trace comparison is currently implemented only for --initial-state=cold-start")

    if args.initial_state == "cold-start":
        if args.start_itime % 48 != 1:
            parser.error("--start-itime must be aligned to a daily boundary plus one")
        day_index = (args.start_itime - 1) // 48 + 1
        day = paper_1961_driver_cold_start_runtime_day_result(
            args.config,
            day_index=day_index,
            year=args.year,
            used_run_def_path=args.run_def,
            single_pass_daily_fold=False,
            retain_stomate_step_results=True,
        )
    else:
        day = paper_1961_driver_day_scaffold(
            args.config,
            year=args.year,
            used_run_def_path=args.run_def,
            root=ROOT,
        )

    if day.daily_process_fold is None:
        summary = {
            "ok": False,
            "initial_state": args.initial_state,
            "missing_components": list(day.missing_components),
            "reason": "daily_process_fold is not available",
        }
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 1

    trace_groups = _trace_groups(start_itime=args.start_itime, steps=args.steps)
    rows = []
    current = np.zeros_like(_pft14_parts(day.daily_process_fold.maintenance.step_results[0].resp_maint_part))
    max_abs = 0.0

    for offset, step_result in enumerate(day.daily_process_fold.maintenance.step_results[: args.steps], start=1):
        itime = int(args.start_itime + offset - 1)
        group = sorted(trace_groups.get(itime, ()), key=lambda row: int(row["part"]))
        if len(group) != NPARTS:
            rows.append({"itime": itime, "reference_available": False, "trace_records": len(group)})
            continue

        jax_radia = _pft14_parts(step_result.resp_maint_part)
        current = current + jax_radia
        jax_sum = _pft14_total(day.daily_process_fold.maintenance.step_results[offset - 1].resp_maint_part.sum(axis=2))
        trace_radia = np.asarray([row["resp_maint_part_radia"] for row in group], dtype=np.float64)
        trace_after_accum = np.asarray([row["resp_maint_part_after_accum"] for row in group], dtype=np.float64)
        trace_sum_values = {float(row["resp_maint_radia_after_sum"]) for row in group}
        if len(trace_sum_values) != 1:
            raise ValueError(f"inconsistent resp_maint_radia_after_sum for itime={itime}")
        trace_sum = trace_sum_values.pop()

        radia_compare = _compare_array(jax_radia, trace_radia)
        accum_compare = _compare_array(current, trace_after_accum)
        sum_abs_error = abs(jax_sum - trace_sum)
        max_abs = max(max_abs, radia_compare["max_abs_error"], accum_compare["max_abs_error"], sum_abs_error)
        rows.append(
            {
                "itime": itime,
                "reference_available": True,
                "resp_maint_part_radia": radia_compare,
                "resp_maint_part_after_accum": accum_compare,
                "resp_maint_radia_after_sum": {
                    "jax": jax_sum,
                    "reference": trace_sum,
                    "abs_error": sum_abs_error,
                },
            }
        )

    all_reference_available = all(row.get("reference_available") for row in rows)
    summary = {
        "ok": bool(getattr(day, "ready_for_day_boundary", getattr(day, "ready_for_day_end_state", False))) and day.daily_process_fold is not None,
        "initial_state": args.initial_state,
        "start_itime": args.start_itime,
        "steps_requested": args.steps,
        "steps_compared": len(rows),
        "missing_components": list(day.missing_components),
        "atol": float(args.atol),
        "max_abs_error": max_abs,
        "trace_matches_runner": all_reference_available and max_abs <= float(args.atol),
        "rows": rows,
        "notes": (
            "server_1961_trace_full was generated with RESTART_FILEIN=NONE. Use --initial-state=cold-start "
            "for parity checks against that package. Day1 PFT14 maintenance is expected to be zero in the "
            "cold-start paper path because cold-start biomass/LAI are zero."
        ),
        "provenance": (
            "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_maint_trace.txt:maint_after",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3244-3267",
            "fortran_source/ORCHIDEE/src_stomate/stomate_resp.f90::maint_respiration lines 122-376",
        ),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["ok"] and summary["trace_matches_runner"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
