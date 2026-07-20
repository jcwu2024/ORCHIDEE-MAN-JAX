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
    paper_1961_driver_cold_start_multiday_modelout_lite_run,
    paper_1961_driver_later_day_runtime_result,
    paper_1961_next_step_runtime_result,
    paper_1961_driver_restart_year_multiday_modelout_lite_run,
    paper_1961_driver_restart_year_start_day_result,
    prepare_paper_1961_driver_context,
    rebase_driver_state_for_year_start,
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


def _read_gpp_after_accu(path: Path) -> dict[int, float]:
    tags = {"gpp_before_accu", "gpp_after_accu"}
    values: dict[int, float] = {}
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
        if tag == "gpp_after_accu" and ik == 1 and pft == 14 and len(data) >= 4:
            values[itime] = float(data[3])
    return values


def _read_year_end_state_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(f"{path} is not a driver year-end state cache")
    return payload


def _day_gpp_rows(day, *, day_index: int, trace: dict[int, float], trace_start_itime: int) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    fold = day.daily_process_fold
    if fold is None:
        return rows
    offset0 = (int(day_index) - 1) * 48
    for offset, step_result in enumerate(fold.accumulator.step_results[:48], start=1):
        trace_itime = int(trace_start_itime + offset0 + offset - 1)
        jax_value = _pft14(step_result.fields["gpp_daily"])
        ref_value = trace.get(trace_itime)
        row: dict[str, float | int | bool] = {
            "day_index": int(day_index),
            "step_in_day": int(offset),
            "trace_itime": int(trace_itime),
            "jax": float(jax_value),
            "reference_available": ref_value is not None,
        }
        if ref_value is not None:
            row["reference"] = float(ref_value)
            row["abs_error"] = float(abs(jax_value - ref_value))
            row["rel_error"] = float(abs(jax_value - ref_value) / max(abs(ref_value), 1.0))
        rows.append(row)
    return rows


def _day_end_gpp_row(day, *, day_index: int, trace: dict[int, float], trace_start_itime: int) -> dict[str, float | int | bool]:
    offset0 = (int(day_index) - 1) * 48
    trace_itime = int(trace_start_itime + offset0 + 47)
    if day.daily_modelout is None or "GPP" not in day.daily_modelout.modelout_fields:
        return {
            "day_index": int(day_index),
            "step_in_day": 48,
            "trace_itime": trace_itime,
            "reference_available": trace_itime in trace,
            "modelout_available": False,
        }
    jax_value = _pft14(day.daily_modelout.modelout_fields["GPP"])
    ref_value = trace.get(trace_itime)
    row: dict[str, float | int | bool] = {
        "day_index": int(day_index),
        "step_in_day": 48,
        "trace_itime": trace_itime,
        "jax": float(jax_value),
        "reference_available": ref_value is not None,
        "modelout_available": True,
    }
    if ref_value is not None:
        row["reference"] = float(ref_value)
        row["abs_error"] = float(abs(jax_value - ref_value))
        row["rel_error"] = float(abs(jax_value - ref_value) / max(abs(ref_value), 1.0))
    return row


def _payload_scalar(payload: dict[str, object], name: str, *, pft14: bool = False) -> float | None:
    if name not in payload:
        return None
    value = payload[name]
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    arr = np.asarray(value, dtype=np.float64)
    if pft14:
        return float(arr.reshape(-1)[13])
    return float(arr.reshape(-1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare JAX restart-year GPP accumulator against Fortran trace.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs" / "reference_mode" / "used_run.def")
    parser.add_argument(
        "--gpp-trace",
        type=Path,
        default=ROOT / "outputs" / "server_year_handoff_reference_domain_20260705_1005" / "orchjax_stomate_daily_trace.txt",
    )
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--restart-year", type=int, default=1962)
    parser.add_argument("--trace-start-itime", type=int, default=17521)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-10,
        help="Relative tolerance for large accumulated GPP traces; a row is bad only if both abs and rel tolerances fail.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reference_mode" / "restart_year_1962_gpp_trace_compare.json")
    parser.add_argument(
        "--previous-state-cache",
        type=Path,
        default=None,
        help="Pickle created by cache_driver_year_end_state.py for restart_year - 1; skips warm-up years.",
    )
    parser.add_argument(
        "--day-end-only",
        action="store_true",
        help="Compare only daily modelout GPP against trace day-end gpp_after_accu without retaining 48 step results.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    context = prepare_paper_1961_driver_context(args.config, used_run_def_path=args.run_def)
    trace = _read_gpp_after_accu(args.gpp_trace)

    cache_metadata = None
    if args.previous_state_cache is not None:
        cache_payload = _read_year_end_state_cache(args.previous_state_cache)
        cache_metadata = {
            key: cache_payload.get(key)
            for key in ("description", "config", "run_def", "start_year", "end_year", "days_per_year", "created_elapsed_seconds")
        }
        if int(cache_payload.get("end_year", -9999)) != int(args.restart_year) - 1:
            raise ValueError("--previous-state-cache must end at restart_year - 1")
        previous_state = cache_payload["state"]
    else:
        year1 = paper_1961_driver_cold_start_multiday_modelout_lite_run(
            args.config,
            year=1961,
            ndays=365,
            used_run_def_path=context.run_def_path,
            module_jit=True,
            diffuco_local_jit=True,
            single_pass_daily_fold=False,
        )
        _block_until_ready(year1)
        previous_state = year1.last_day_end_state
        if previous_state is None or not year1.ready_for_requested_days:
            raise RuntimeError(f"1961 did not complete: {year1.missing_components}")
        for warm_year in range(1962, int(args.restart_year)):
            warm = paper_1961_driver_restart_year_multiday_modelout_lite_run(
                args.config,
                previous_year_end_state=previous_state,
                year=warm_year,
                ndays=365,
                used_run_def_path=context.run_def_path,
                module_jit=True,
                diffuco_local_jit=True,
                single_pass_daily_fold=False,
            )
            _block_until_ready(warm)
            previous_state = warm.last_day_end_state
            if previous_state is None or not warm.ready_for_requested_days:
                raise RuntimeError(f"{warm_year} did not complete: {warm.missing_components}")

    first_probe = paper_1961_next_step_runtime_result(
        args.config,
        previous_state=rebase_driver_state_for_year_start(previous_state),
        year=int(args.restart_year),
        tstep=0,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
    )
    _block_until_ready(first_probe)
    first_payload_probe = None
    if first_probe.entry_payload is not None:
        payload = first_probe.entry_payload
        first_payload_probe = {
            "ok": bool(first_probe.ok),
            "missing_components": list(first_probe.missing_components),
            "gpp_pft14": _payload_scalar(payload, "gpp", pft14=True),
            "swdown": _payload_scalar(payload, "swdown"),
            "ccanopy": _payload_scalar(payload, "ccanopy"),
            "t2m": _payload_scalar(payload, "t2m"),
            "tair": _payload_scalar(payload, "temp_air"),
            "qair": _payload_scalar(payload, "qair"),
            "pb": _payload_scalar(payload, "pb"),
            "control_salinity_pft14": _payload_scalar(payload, "control_salinity", pft14=True),
            "control_inudate_pft14": _payload_scalar(payload, "control_inudate", pft14=True),
        }

    steps_per_stomate = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    rows: list[dict[str, float | int | bool]] = []
    day_summaries: list[dict[str, object]] = []
    for day_index in range(1, int(args.days) + 1):
        if day_index == 1:
            day = paper_1961_driver_restart_year_start_day_result(
                args.config,
                previous_year_end_state=previous_state,
                year=int(args.restart_year),
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                module_jit=True,
                diffuco_local_jit=True,
                single_pass_daily_fold=False,
                retain_stomate_step_results=not args.day_end_only,
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
                module_jit=True,
                diffuco_local_jit=True,
                single_pass_daily_fold=False,
                retain_stomate_step_results=not args.day_end_only,
            )
        _block_until_ready(day)
        if args.day_end_only:
            day_rows = [_day_end_gpp_row(day, day_index=day_index, trace=trace, trace_start_itime=args.trace_start_itime)]
        else:
            day_rows = _day_gpp_rows(day, day_index=day_index, trace=trace, trace_start_itime=args.trace_start_itime)
        rows.extend(day_rows)
        available = [row for row in day_rows if row.get("reference_available")]
        day_summaries.append(
            {
                "day_index": int(day_index),
                "ready": bool(day.ready_for_day_end_state),
                "missing_components": list(day.missing_components),
                "steps_compared": len(available),
                "day_end": available[-1] if available else None,
                "max_abs_error": max((float(row.get("abs_error", 0.0)) for row in available), default=0.0),
            }
        )
        if not day.ready_for_day_end_state:
            previous_state = None
            break
        previous_state = day.day_end_state

    compared = [row for row in rows if row.get("reference_available")]
    first_bad = None
    for row in compared:
        if float(row.get("abs_error", 0.0)) > float(args.atol) and float(row.get("rel_error", 0.0)) > float(args.rtol):
            first_bad = row
            break
    expected_compared = int(args.days) if args.day_end_only else int(args.days) * 48
    payload = {
        "ok": first_bad is None and len(compared) == expected_compared,
        "elapsed_seconds": time.perf_counter() - started,
        "run_def": str(args.run_def),
        "gpp_trace": str(args.gpp_trace),
        "trace_start_itime": int(args.trace_start_itime),
        "restart_year": int(args.restart_year),
        "days": int(args.days),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "day_end_only": bool(args.day_end_only),
        "expected_compared_rows": expected_compared,
        "compared_rows": len(compared),
        "previous_state_cache": None if args.previous_state_cache is None else str(args.previous_state_cache),
        "previous_state_cache_metadata": cache_metadata,
        "day_summaries": day_summaries,
        "first_payload_probe": first_payload_probe,
        "first_bad": first_bad,
        "max_abs_error": max((float(row.get("abs_error", 0.0)) for row in compared), default=0.0),
        "max_rel_error": max((float(row.get("rel_error", 0.0)) for row in compared), default=0.0),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "ok",
                    "elapsed_seconds",
                    "max_abs_error",
                    "max_rel_error",
                    "first_bad",
                    "compared_rows",
                    "expected_compared_rows",
                    "first_payload_probe",
                    "day_summaries",
                )
            },
            indent=2,
        )
    )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
