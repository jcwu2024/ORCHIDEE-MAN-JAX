from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_later_day_runtime_result,
    paper_1961_driver_restart_year_start_day_result,
    prepare_paper_1961_driver_context,
)


def _block_until_ready(value) -> None:
    try:
        import jax

        leaves = jax.tree_util.tree_leaves(value)
    except Exception:
        leaves = (value,)
    for leaf in leaves:
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    try:
        jax.effects_barrier()
    except Exception:
        pass


def _pft14(value) -> float:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[13])


def _read_state_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(f"{path} is not a driver year-end state cache")
    return payload


def _read_daily_trace(path: Path, *, start_itime: int, steps: int) -> dict[int, dict[str, float]]:
    tags = {"gpp_before_accu", "gpp_after_accu"}
    end_itime = int(start_itime) + int(steps) - 1
    values: dict[int, dict[str, float]] = {}
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
        do_slow = parts[4].upper().startswith("T")
        dt_sechiba = float(parts[5])
        data: list[str] = []
        while index < len(lines):
            next_parts = lines[index].split()
            if next_parts and next_parts[0] in tags:
                break
            data.extend(next_parts)
            index += 1
        if itime < start_itime or itime > end_itime or ik != 1 or pft != 14 or len(data) < 4:
            continue
        row = values.setdefault(
            itime,
            {
                "itime": float(itime),
                "do_slow": float(do_slow),
                "dt_sechiba": dt_sechiba,
                "dt_stomate": float(data[0]),
                "veget_cov_max": float(data[1]),
                "gpp_d": float(data[2]),
            },
        )
        if tag == "gpp_before_accu":
            row["gpp_daily_before_accu"] = float(data[3])
        else:
            row["gpp_daily_after_accu"] = float(data[3])
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect restart-year STOMATE GPP scheduling inputs for one target day."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs" / "reference_mode" / "used_run.def")
    parser.add_argument("--restart-year", type=int, default=1963)
    parser.add_argument("--previous-state-cache", type=Path, required=True)
    parser.add_argument("--gpp-trace", type=Path, required=True)
    parser.add_argument("--trace-start-itime", type=int, required=True)
    parser.add_argument("--day-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.day_index < 1:
        parser.error("--day-index must be positive")

    started = time.perf_counter()
    cache = _read_state_cache(args.previous_state_cache)
    if int(cache.get("end_year", -9999)) != int(args.restart_year) - 1:
        raise ValueError("--previous-state-cache must end at restart_year - 1")
    context = prepare_paper_1961_driver_context(args.config, used_run_def_path=args.run_def)
    steps_per_stomate = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    previous_state = cache["state"]

    for day_index in range(1, int(args.day_index) + 1):
        if day_index == 1:
            day = paper_1961_driver_restart_year_start_day_result(
                args.config,
                previous_year_end_state=previous_state,
                year=int(args.restart_year),
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                retain_stomate_step_results=day_index == int(args.day_index),
            )
        else:
            day = paper_1961_driver_later_day_runtime_result(
                args.config,
                previous_state=previous_state,
                day_index=day_index,
                year=int(args.restart_year),
                start_tstep=(day_index - 1) * steps_per_stomate,
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                retain_stomate_step_results=day_index == int(args.day_index),
            )
        _block_until_ready(day)
        if not day.ready_for_day_end_state:
            raise RuntimeError(f"day {day_index} did not complete: {day.missing_components}")
        previous_state = day.day_end_state

    if day.daily_process_fold is None:
        raise RuntimeError("target day did not retain daily_process_fold")

    start_itime = int(args.trace_start_itime) + (int(args.day_index) - 1) * steps_per_stomate
    trace = _read_daily_trace(args.gpp_trace, start_itime=start_itime, steps=steps_per_stomate)
    rows: list[dict[str, object]] = []
    scale = float(context.runtime.dt_stomate / context.runtime.dt_sechiba)
    for offset, step_result in enumerate(day.daily_process_fold.accumulator.step_results, start=1):
        itime = start_itime + offset - 1
        local = step_result.local_prep
        fields = step_result.fields
        jax_gpp_d = _pft14(local.gpp_d)
        jax_veget_cov_max = _pft14(local.veget_cov_max)
        trace_row = trace.get(itime, {})
        row = {
            "step_in_day": offset,
            "itime": itime,
            "jax_gpp_d": jax_gpp_d,
            "jax_veget_cov_max": jax_veget_cov_max,
            "jax_inferred_raw_gpp": jax_gpp_d * jax_veget_cov_max / scale,
            "jax_gpp_daily_after_accu": _pft14(fields["gpp_daily"]),
            "reference_available": bool(trace_row),
        }
        if trace_row:
            ref_gpp_d = float(trace_row["gpp_d"])
            ref_veget_cov_max = float(trace_row["veget_cov_max"])
            row.update(
                {
                    "reference_gpp_d": ref_gpp_d,
                    "reference_veget_cov_max": ref_veget_cov_max,
                    "reference_inferred_raw_gpp": ref_gpp_d * ref_veget_cov_max / scale,
                    "reference_gpp_daily_after_accu": trace_row.get("gpp_daily_after_accu"),
                    "gpp_d_abs_error": abs(jax_gpp_d - ref_gpp_d),
                    "veget_cov_max_abs_error": abs(jax_veget_cov_max - ref_veget_cov_max),
                    "raw_gpp_abs_error": abs((jax_gpp_d * jax_veget_cov_max / scale) - (ref_gpp_d * ref_veget_cov_max / scale)),
                    "gpp_daily_after_accu_abs_error": abs(_pft14(fields["gpp_daily"]) - float(trace_row.get("gpp_daily_after_accu", np.nan))),
                }
            )
        rows.append(row)

    payload = {
        "elapsed_seconds": time.perf_counter() - started,
        "restart_year": int(args.restart_year),
        "day_index": int(args.day_index),
        "trace_start_itime": int(args.trace_start_itime),
        "target_day_start_itime": start_itime,
        "steps_per_stomate": steps_per_stomate,
        "previous_state_cache": str(args.previous_state_cache),
        "gpp_trace": str(args.gpp_trace),
        "max_gpp_d_abs_error": max((float(row.get("gpp_d_abs_error", 0.0)) for row in rows), default=0.0),
        "first_gpp_d_abs_error_gt_1e-6": next((row for row in rows if float(row.get("gpp_d_abs_error", 0.0)) > 1.0e-6), None),
        "rows": rows,
        "provenance": (
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2921-2931 and 3020-3032",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_accu_r1d lines 9341-9411",
            "orchjax_stomate_daily_trace.txt:gpp_before_accu/gpp_after_accu",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "elapsed_seconds": payload["elapsed_seconds"],
                "max_gpp_d_abs_error": payload["max_gpp_d_abs_error"],
                "first_gpp_d_abs_error_gt_1e-6": payload["first_gpp_d_abs_error_gt_1e-6"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
