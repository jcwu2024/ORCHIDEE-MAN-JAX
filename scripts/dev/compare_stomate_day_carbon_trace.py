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
    paper_1961_driver_later_day_scaffold,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.stomate.carbon_kernels import ICARBON, NPARTS  # noqa: E402
from jax_orchidee.trace.server_1961 import read_server_records  # noqa: E402


def _pft14_parts(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[0, 13, :NPARTS]


def _pft14_part_carbon(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[0, 13, :NPARTS, ICARBON]


def _pft14_scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64)[0, 13])


def _day_group(schema: str, tag: str, *, day_index: int) -> list[dict[str, object]]:
    groups_seen = 0
    group: list[dict[str, object]] = []
    for row in read_server_records(schema, tags=tag):
        if int(row["jv"]) != 14:
            continue
        group.append(row)
        if len(group) != NPARTS:
            continue
        groups_seen += 1
        if groups_seen == int(day_index):
            return sorted(group, key=lambda item: int(item["part"]))
        group = []
    raise ValueError(f"no complete {schema}:{tag} group for day {day_index}")


def _array_compare(jax_value, reference) -> dict[str, object]:
    jax_array = np.asarray(jax_value, dtype=np.float64)
    ref_array = np.asarray(reference, dtype=np.float64)
    delta = jax_array - ref_array
    abs_delta = np.abs(delta)
    index = int(np.argmax(abs_delta)) if abs_delta.size else 0
    return {
        "max_abs_error": float(abs_delta.reshape(-1)[index]) if abs_delta.size else 0.0,
        "max_index": index,
        "jax_at_max": float(jax_array.reshape(-1)[index]) if abs_delta.size else 0.0,
        "reference_at_max": float(ref_array.reshape(-1)[index]) if abs_delta.size else 0.0,
    }


def _after_alloc_summary(chain, rows: list[dict[str, object]]) -> dict[str, object]:
    trace_f_alloc = np.asarray([row["f_alloc"] for row in rows], dtype=np.float64)
    trace_biomass = np.asarray([row["biomass_after_alloc"] for row in rows], dtype=np.float64)
    trace_lai = np.asarray([row["lai"] for row in rows], dtype=np.float64)
    jax_f_alloc = _pft14_parts(chain.allocation.f_alloc)
    jax_alloc_sap_above = float(np.asarray(chain.allocation.alloc_sap_above, dtype=np.float64)[0, 13])
    trace_sap_total = trace_f_alloc[1] + trace_f_alloc[2] + trace_f_alloc[8] + trace_f_alloc[9]
    trace_alloc_sap_above = (trace_f_alloc[1] + trace_f_alloc[8] + trace_f_alloc[9]) / trace_sap_total
    return {
        "f_alloc": _array_compare(jax_f_alloc, trace_f_alloc),
        "f_alloc_values": {
            "jax": jax_f_alloc.tolist(),
            "reference": trace_f_alloc.tolist(),
            "delta": (jax_f_alloc - trace_f_alloc).tolist(),
        },
        "biomass_after_alloc": _array_compare(_pft14_part_carbon(chain.allocation.biomass), trace_biomass),
        "lai_after_alloc": _array_compare(np.repeat(_pft14_scalar(chain.lai_after_alloc), NPARTS), trace_lai),
        "jax_intermediates": {
            "l_to_lsr": float(np.asarray(chain.allocation.l_to_lsr, dtype=np.float64)[0, 13]),
            "s_to_lsr": float(np.asarray(chain.allocation.s_to_lsr, dtype=np.float64)[0, 13]),
            "r_to_lsr": float(np.asarray(chain.allocation.r_to_lsr, dtype=np.float64)[0, 13]),
            "alloc_sap_above": jax_alloc_sap_above,
            "carb_rescale": float(np.asarray(chain.allocation.carb_rescale, dtype=np.float64)[0, 13]),
            "limit_l": float(np.asarray(chain.allocation.limit_l, dtype=np.float64)[0, 13]),
            "limit_w": float(np.asarray(chain.allocation.limit_w, dtype=np.float64)[0, 13]),
            "limit_n": float(np.asarray(chain.allocation.limit_n, dtype=np.float64)[0, 13]),
            "limit_w_or_n": float(np.asarray(chain.allocation.limit_w_or_n, dtype=np.float64)[0, 13]),
        },
        "trace_inferred": {
            "sap_total_fraction": float(trace_sap_total),
            "alloc_sap_above": float(trace_alloc_sap_above),
            "carb_rescale_if_no_reserve": 1.0,
        },
    }


def _after_npp_summary(chain, rows: list[dict[str, object]]) -> dict[str, object]:
    daily = chain.post_npp.daily_carbon
    age_sla = daily.age_sla
    trace_gpp = np.asarray([row["gpp_daily"] for row in rows], dtype=np.float64)
    trace_f_alloc = np.asarray([row["f_alloc"] for row in rows], dtype=np.float64)
    trace_bm_alloc = np.asarray([row["bm_alloc"] for row in rows], dtype=np.float64)
    trace_biomass = np.asarray([row["biomass_after_npp"] for row in rows], dtype=np.float64)
    trace_resp_part = np.asarray([row["resp_maint_part"] for row in rows], dtype=np.float64)
    trace_resp_maint = np.asarray([row["resp_maint"] for row in rows], dtype=np.float64)
    trace_resp_growth = np.asarray([row["resp_growth"] for row in rows], dtype=np.float64)
    trace_npp = np.asarray([row["npp_daily"] for row in rows], dtype=np.float64)
    trace_pft_present = np.asarray([row["pft_present"] for row in rows], dtype=np.float64)
    trace_age = np.asarray([row["age"] for row in rows], dtype=np.float64)
    pft_present = daily.boundary.pft_present
    return {
        "gpp_daily": _array_compare(np.repeat(_pft14_scalar(daily.boundary.gpp_daily), NPARTS), trace_gpp),
        "f_alloc": _array_compare(_pft14_parts(daily.boundary.f_alloc), trace_f_alloc),
        "bm_alloc": _array_compare(_pft14_part_carbon(daily.npp_update.bm_alloc), trace_bm_alloc),
        "biomass_after_npp": _array_compare(_pft14_part_carbon(daily.npp_update.biomass), trace_biomass),
        "resp_maint_part": _array_compare(_pft14_parts(daily.boundary.resp_maint_part), trace_resp_part),
        "resp_maint": _array_compare(np.repeat(_pft14_scalar(daily.npp_update.resp_maint), NPARTS), trace_resp_maint),
        "resp_growth": _array_compare(np.repeat(_pft14_scalar(daily.npp_update.resp_growth), NPARTS), trace_resp_growth),
        "npp_daily": _array_compare(np.repeat(_pft14_scalar(daily.npp_update.npp), NPARTS), trace_npp),
        "pft_present": _array_compare(np.repeat(_pft14_scalar(pft_present), NPARTS), trace_pft_present),
        "age": _array_compare(np.repeat(_pft14_scalar(age_sla.age), NPARTS), trace_age) if age_sla is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare one paper-case STOMATE daily carbon day to LPJ/NPP traces.")
    parser.add_argument("--day-index", type=int, required=True)
    parser.add_argument("--year", type=int, default=1961)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument(
        "--run-def",
        type=Path,
        default=ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.day_index < 1:
        parser.error("--day-index must be positive")

    context = prepare_paper_1961_driver_context(args.config, used_run_def_path=args.run_def)
    steps_per_stomate = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    first = paper_1961_driver_cold_start_day_scaffold(
        args.config,
        year=args.year,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
    )
    if args.day_index == 1:
        day = first
    else:
        previous_state = first.first_day_end_state
        if previous_state is None or not first.ready_for_first_day_end_state:
            raise RuntimeError(f"day 1 did not close: {first.missing_components}")
        day = None
        for current_day in range(2, args.day_index + 1):
            if current_day == args.day_index:
                day = paper_1961_driver_later_day_scaffold(
                    args.config,
                    previous_state=previous_state,
                    year=args.year,
                    start_tstep=(current_day - 1) * steps_per_stomate,
                    used_run_def_path=context.run_def_path,
                    prepared_context=context,
                    retain_stomate_step_results=False,
                    runtime_entry_payloads=True,
                )
            else:
                runtime_day = paper_1961_driver_later_day_runtime_result(
                    args.config,
                    previous_state=previous_state,
                    day_index=current_day,
                    year=args.year,
                    start_tstep=(current_day - 1) * steps_per_stomate,
                    used_run_def_path=context.run_def_path,
                    prepared_context=context,
                    retain_stomate_step_results=False,
                )
                if not runtime_day.ready_for_day_end_state:
                    raise RuntimeError(f"day {current_day} did not close: {runtime_day.missing_components}")
                previous_state = runtime_day.day_end_state
        if day is None:
            raise RuntimeError("target day was not reached")

    if day.stomate_daily_carbon is None:
        raise RuntimeError(f"target day did not produce stomate_daily_carbon: {day.missing_components}")
    after_alloc_rows = _day_group("stomate_lpj", "after_alloc", day_index=args.day_index)
    after_npp_rows = _day_group("stomate_npp", "after_npp", day_index=args.day_index)
    summary = {
        "ok": bool(day.ready_for_day_end_state),
        "day_index": int(args.day_index),
        "missing_components": list(day.missing_components),
        "after_alloc": _after_alloc_summary(day.stomate_daily_carbon, after_alloc_rows),
        "after_npp": _after_npp_summary(day.stomate_daily_carbon, after_npp_rows),
        "provenance": (
            "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_lpj_trace.txt:after_alloc",
            "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_npp_trace.txt:after_npp",
            "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1093-1131",
            "fortran_source/ORCHIDEE/src_stomate/stomate_alloc.f90::alloc",
            "fortran_source/ORCHIDEE/src_stomate/stomate_npp.f90::npp_calc",
        ),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
