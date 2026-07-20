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
    paper_1961_driver_later_day_runtime_result,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.stomate.carbon_kernels import NPARTS  # noqa: E402
from jax_orchidee.trace.server_1961 import read_server_records  # noqa: E402


def _read_gpp_after_accu(path: Path, *, start_itime: int, steps: int) -> dict[int, float]:
    values: dict[int, float] = {}
    tags = {"gpp_before_accu", "gpp_after_accu"}
    end_itime = int(start_itime) + int(steps) - 1
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        parts = lines[index].split()
        index += 1
        if not parts or parts[0] not in tags:
            continue
        tag = parts[0]
        itime = int(parts[1])
        ik = int(parts[2])
        pft = int(parts[3])
        data: list[str] = []
        while index < len(lines):
            next_parts = lines[index].split()
            if next_parts and next_parts[0] in tags:
                break
            data.extend(next_parts)
            index += 1
        if itime < start_itime or itime > end_itime:
            continue
        if tag != "gpp_after_accu" or ik != 1 or pft != 14 or len(data) < 4:
            continue
        values[itime] = float(data[3])
    return values


def _trace_maint_groups(*, start_itime: int, steps: int) -> dict[int, list[dict[str, object]]]:
    groups: dict[int, list[dict[str, object]]] = {}
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


def _pft14(value) -> float:
    return float(np.asarray(value).reshape(-1)[13])


def _pft14_parts(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[0, 13, :NPARTS]


def _pft14_total(value) -> float:
    return float(np.asarray(value, dtype=np.float64)[0, 13])


def _max_abs_array(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(delta))) if delta.size else 0.0


def _compare_gpp(day, *, start_itime: int, steps: int, trace_path: Path, atol: float, rtol: float) -> dict[str, object]:
    trace = _read_gpp_after_accu(trace_path, start_itime=start_itime, steps=steps)
    rows: list[dict[str, object]] = []
    max_abs = 0.0
    max_rel = 0.0
    all_within_tolerance = True
    for offset, step_result in enumerate(day.daily_process_fold.accumulator.step_results[:steps], start=1):
        itime = int(start_itime + offset - 1)
        jax_value = _pft14(step_result.fields["gpp_daily"])
        reference = trace.get(itime)
        if reference is None:
            rows.append({"itime": itime, "reference_available": False, "jax": jax_value})
            continue
        abs_error = abs(jax_value - reference)
        rel_error = abs_error / max(abs(reference), 1.0)
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        within = abs_error <= float(atol) + float(rtol) * abs(reference)
        all_within_tolerance = all_within_tolerance and within
        rows.append(
            {
                "itime": itime,
                "reference_available": True,
                "jax": jax_value,
                "reference": reference,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "within_tolerance": within,
            }
        )
    return {
        "trace_matches_runner": all(row.get("reference_available") for row in rows) and all_within_tolerance,
        "steps_compared": len(rows),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "atol": float(atol),
        "rtol": float(rtol),
        "rows": rows,
        "provenance": (
            "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_daily_trace.txt:gpp_after_accu",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3198-3208",
        ),
    }


def _compare_maint(day, *, start_itime: int, steps: int, atol: float) -> dict[str, object]:
    trace_groups = _trace_maint_groups(start_itime=start_itime, steps=steps)
    rows: list[dict[str, object]] = []
    current = np.zeros_like(_pft14_parts(day.daily_process_fold.maintenance.step_results[0].resp_maint_part))
    max_abs = 0.0
    for offset, step_result in enumerate(day.daily_process_fold.maintenance.step_results[:steps], start=1):
        itime = int(start_itime + offset - 1)
        group = sorted(trace_groups.get(itime, ()), key=lambda row: int(row["part"]))
        if len(group) != NPARTS:
            rows.append({"itime": itime, "reference_available": False, "trace_records": len(group)})
            continue

        jax_radia = _pft14_parts(step_result.resp_maint_part)
        current = current + jax_radia
        jax_sum = _pft14_total(step_result.resp_maint_part.sum(axis=2))
        trace_radia = np.asarray([row["resp_maint_part_radia"] for row in group], dtype=np.float64)
        trace_after_accum = np.asarray([row["resp_maint_part_after_accum"] for row in group], dtype=np.float64)
        trace_sum_values = {float(row["resp_maint_radia_after_sum"]) for row in group}
        if len(trace_sum_values) != 1:
            raise ValueError(f"inconsistent resp_maint_radia_after_sum for itime={itime}")
        trace_sum = trace_sum_values.pop()

        radia_error = _max_abs_array(jax_radia, trace_radia)
        accum_error = _max_abs_array(current, trace_after_accum)
        sum_error = abs(jax_sum - trace_sum)
        max_abs = max(max_abs, radia_error, accum_error, sum_error)
        rows.append(
            {
                "itime": itime,
                "reference_available": True,
                "resp_maint_part_radia_max_abs_error": radia_error,
                "resp_maint_part_after_accum_max_abs_error": accum_error,
                "resp_maint_radia_after_sum_abs_error": sum_error,
            }
        )
    return {
        "trace_matches_runner": all(row.get("reference_available") for row in rows) and max_abs <= float(atol),
        "steps_compared": len(rows),
        "max_abs_error": max_abs,
        "atol": float(atol),
        "rows": rows,
        "provenance": (
            "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_maint_trace.txt:maint_after",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3244-3267",
            "fortran_source/ORCHIDEE/src_stomate/stomate_resp.f90::maint_respiration lines 122-376",
        ),
    }


def _drop_rows(section: dict[str, object]) -> dict[str, object]:
    compact = dict(section)
    compact.pop("rows", None)
    return compact


def _summarize_day(day, *, day_index: int, steps: int, args: argparse.Namespace) -> dict[str, object]:
    start_itime = (int(day_index) - 1) * 48 + 1
    ready = bool(getattr(day, "ready_for_day_boundary", getattr(day, "ready_for_day_end_state", False)))
    if day.daily_process_fold is None:
        return {
            "ok": False,
            "day_index": int(day_index),
            "start_itime": start_itime,
            "missing_components": list(day.missing_components),
            "reason": "daily_process_fold is not available",
        }

    gpp = _compare_gpp(
        day,
        start_itime=start_itime,
        steps=steps,
        trace_path=args.gpp_trace,
        atol=args.gpp_atol,
        rtol=args.gpp_rtol,
    )
    maint = _compare_maint(day, start_itime=start_itime, steps=steps, atol=args.maint_atol)
    if not args.include_rows:
        gpp = _drop_rows(gpp)
        maint = _drop_rows(maint)
    return {
        "ok": ready and bool(gpp["trace_matches_runner"]) and bool(maint["trace_matches_runner"]),
        "initial_state": "cold-start",
        "day_index": int(day_index),
        "start_itime": start_itime,
        "steps_requested": int(steps),
        "missing_components": list(day.missing_components),
        "gpp_after_accu": gpp,
        "maintenance": maint,
        "notes": (
            "This combines the existing daily GPP accumulator and maintenance respiration process-cut checks. "
            "GPP uses the audited whole-driver SWdown roundoff budget; maintenance uses the audited "
            "whole-driver floating accumulation budget from docs/source_audits/numeric_tolerance_ledger.md."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare one cold-start daily STOMATE process window against Fortran traces."
    )
    parser.add_argument(
        "--day-index",
        type=int,
        nargs="+",
        required=True,
        help="1-based paper-case day(s) to validate. Multiple days are advanced in one sequential run.",
    )
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--year", type=int, default=1961)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument(
        "--run-def",
        type=Path,
        default=ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def",
    )
    parser.add_argument(
        "--gpp-trace",
        type=Path,
        default=ROOT / "outputs" / "server_1961_trace_full_20260623" / "traces" / "orchjax_stomate_daily_trace.txt",
    )
    parser.add_argument("--gpp-atol", type=float, default=1.0e-5)
    parser.add_argument("--gpp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--maint-atol", type=float, default=1.0e-11)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-rows", action="store_true")
    args = parser.parse_args()

    day_indices = tuple(sorted(dict.fromkeys(int(day) for day in args.day_index)))
    if any(day < 1 for day in day_indices):
        parser.error("--day-index values must be positive")
    if args.steps < 1:
        parser.error("--steps must be positive")

    context = prepare_paper_1961_driver_context(args.config, used_run_def_path=args.run_def)
    steps_per_stomate = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    max_day = max(day_indices)
    target_days = set(day_indices)
    windows: list[dict[str, object]] = []

    first = paper_1961_driver_cold_start_day_scaffold(
        args.config,
        year=args.year,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
    )
    if 1 in target_days:
        windows.append(_summarize_day(first, day_index=1, steps=args.steps, args=args))
    previous_state = first.first_day_end_state
    stopped_day_index = None
    missing_components: tuple[str, ...] = ()
    if previous_state is None or not first.ready_for_first_day_end_state:
        stopped_day_index = 1
        missing_components = first.missing_components
    else:
        for current_day in range(2, max_day + 1):
            day = paper_1961_driver_later_day_runtime_result(
                args.config,
                previous_state=previous_state,
                day_index=current_day,
                year=args.year,
                start_tstep=(current_day - 1) * steps_per_stomate,
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                single_pass_daily_fold=False,
                retain_stomate_step_results=current_day in target_days,
            )
            if current_day in target_days:
                windows.append(_summarize_day(day, day_index=current_day, steps=args.steps, args=args))
            if not day.ready_for_day_end_state:
                stopped_day_index = current_day
                missing_components = day.missing_components
                break
            previous_state = day.day_end_state

    summary = {
        "ok": stopped_day_index is None and all(bool(window["ok"]) for window in windows),
        "initial_state": "cold-start",
        "requested_day_indices": list(day_indices),
        "validated_day_indices": [int(window["day_index"]) for window in windows],
        "max_day_advanced": max_day if stopped_day_index is None else int(stopped_day_index),
        "stopped_day_index": stopped_day_index,
        "missing_components": list(missing_components),
        "steps_requested": int(args.steps),
        "windows": windows,
    }

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
