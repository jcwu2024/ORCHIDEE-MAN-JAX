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


def _read_gpp_after_accu(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    tags = {"gpp_before_accu", "gpp_after_accu"}
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
        if tag != "gpp_after_accu" or ik != 1 or pft != 14 or len(data) < 4:
            continue
        values[itime] = float(data[3])
    return values


def _pft14(value) -> float:
    return float(np.asarray(value).reshape(-1)[13])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local STOMATE GPP daily accumulation against Fortran trace.")
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
    parser.add_argument(
        "--trace",
        type=Path,
        default=ROOT / "outputs" / "server_1961_trace_full_20260623" / "traces" / "orchjax_stomate_daily_trace.txt",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-12,
        help="Absolute tolerance for trace_matches_runner. Use 1e-5 for whole-driver windows with audited SWdown roundoff.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for trace_matches_runner; default preserves the historical absolute-only check.",
    )
    args = parser.parse_args()

    if args.start_itime < 1:
        parser.error("--start-itime must be positive")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.start_itime % 48 != 1:
        parser.error("--start-itime must be aligned to a daily boundary plus one")

    if args.initial_state == "cold-start":
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
        if args.start_itime != 1:
            parser.error("reference-start comparison currently supports only --start-itime=1")
        day = paper_1961_driver_day_scaffold(
            args.config,
            year=args.year,
            used_run_def_path=args.run_def,
            root=ROOT,
        )
    trace = _read_gpp_after_accu(args.trace)
    rows = []
    max_abs = 0.0
    max_rel = 0.0
    all_within_tolerance = True
    for offset, step_result in enumerate(day.daily_process_fold.accumulator.step_results[: args.steps], start=1):
        itime = int(args.start_itime + offset - 1)
        jax_value = _pft14(step_result.fields["gpp_daily"])
        reference = trace.get(itime)
        if reference is None:
            rows.append({"itime": itime, "reference_available": False, "jax": jax_value})
            continue
        abs_error = abs(jax_value - reference)
        rel_error = abs_error / max(abs(reference), 1.0)
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        within = abs_error <= float(args.atol) + float(args.rtol) * abs(reference)
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

    all_reference_available = all(row.get("reference_available") for row in rows)
    ready = getattr(day, "ready_for_day_boundary", getattr(day, "ready_for_day_end_state", False))
    summary = {
        "ok": bool(ready) and day.daily_process_fold is not None,
        "initial_state": args.initial_state,
        "start_itime": args.start_itime,
        "steps_requested": args.steps,
        "steps_compared": len(rows),
        "missing_components": list(day.missing_components),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "trace_matches_runner": all_reference_available and all_within_tolerance,
        "rows": rows,
        "notes": (
            "server_1961_trace_full was generated with RESTART_FILEIN=NONE. Use --initial-state=cold-start "
            "for parity checks against that package. --initial-state=reference-start is diagnostic only unless "
            "the trace was regenerated from reference/case_001_071 start files."
        ),
        "provenance": (
            "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_daily_trace.txt:gpp_after_accu",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3198-3208",
        ),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["ok"] and all_reference_available else 1


if __name__ == "__main__":
    raise SystemExit(main())
