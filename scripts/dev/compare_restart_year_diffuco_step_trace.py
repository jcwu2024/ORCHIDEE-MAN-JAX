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
    _paper_diffuco_day_static_cache_from_previous_state,
    driver_previous_step_state_from_next_step_hydrol_scaffold,
    paper_1961_driver_later_day_runtime_result,
    paper_1961_next_step_hydrol_precall_scaffold,
    paper_1961_driver_restart_year_start_day_result,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.sechiba.diffuco import diffuco_trans_co2_qsatt_explicit  # noqa: E402
from jax_orchidee.trace.server_1961 import trace_schema  # noqa: E402


TAG = "after_diffuco_trans_co2_pft14"


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


def _scalar(value, *, pft14: bool = False) -> float | None:
    if value is None:
        return None
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    if pft14:
        return float(arr.reshape(-1)[13])
    return float(arr.reshape(-1)[0])


def _read_state_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(f"{path} is not a driver year-end state cache")
    return payload


def _read_trans_trace(path: Path, *, kjit: int) -> dict[str, float | int | str] | None:
    fields = trace_schema("diffuco_trans_co2").tags[TAG].fields
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        parts = lines[index].split()
        index += 1
        if not parts or parts[0] != TAG:
            continue
        data: list[str] = parts[1:]
        while len(data) < len(fields) and index < len(lines):
            next_parts = lines[index].split()
            if next_parts and next_parts[0] == TAG:
                break
            data.extend(next_parts)
            index += 1
        if len(data) < len(fields):
            continue
        row: dict[str, float | int | str] = {"tag": TAG}
        for name, text in zip(fields, data, strict=False):
            if name in {"kjit", "ji", "jv", "ilai"}:
                row[name] = int(float(text))
            else:
                row[name] = float(text)
        if int(row["kjit"]) == int(kjit):
            return row
    return None


def _entry_values(payload: dict[str, object]) -> dict[str, float | None]:
    mapping = {
        "gpp": ("gpp", True),
        "qsurf": ("qsurf", False),
        "humrel": ("humrel", True),
        "veget": ("veget", True),
        "veget_max": ("veget_max", True),
        "lai": ("lai", True),
        "qsintveg": ("qsintveg", True),
        "qsintmax": ("qsintmax", True),
        "t2m": ("t2m", False),
        "temp_growth": ("temp_growth", True),
        "pb": ("pb", False),
        "qair": ("qair", False),
    }
    result = {}
    for output_name, (payload_name, pft14) in mapping.items():
        result[output_name] = _scalar(payload.get(payload_name), pft14=pft14)
    return result


def _diffuco_values(diffuco_local) -> dict[str, float | None]:
    if diffuco_local is None:
        return {}
    boundary = diffuco_local.result.boundary
    chain = boundary.process_chain
    trans = chain.trans_co2
    closure = chain.closure
    payload = diffuco_local.result.payload.payload
    kwargs = diffuco_local.kwargs
    trans_inputs = kwargs.get("trans_co2_inputs") or {}
    trans_qsatt = None
    if trans_inputs.get("t2m") is not None and trans_inputs.get("pb") is not None:
        trans_qsatt = diffuco_trans_co2_qsatt_explicit(t2m=trans_inputs["t2m"], pb=trans_inputs["pb"])
    values = {
        "swdown": _scalar(trans_inputs.get("swdown")),
        "pb": _scalar(kwargs.get("pb")),
        "qsurf": _scalar(kwargs.get("qsurf")),
        "qsatt": _scalar(trans_qsatt),
        "surface_qsatt": _scalar(boundary.qsatt),
        "t2m": _scalar(trans_inputs.get("t2m")),
        "temp_growth": _scalar(trans_inputs.get("temp_growth")),
        "ca": _scalar(trans_inputs.get("ca")),
        "vcmax": _scalar(trans_inputs.get("vcmax")),
        "humrel": _scalar(kwargs.get("humrel"), pft14=True),
        "veget": _scalar(kwargs.get("veget"), pft14=True),
        "veget_max": _scalar(kwargs.get("veget_max"), pft14=True),
        "lai": _scalar(kwargs.get("lai"), pft14=True),
        "qsintveg": _scalar(kwargs.get("qsintveg"), pft14=True),
        "qsintmax": _scalar(kwargs.get("qsintmax"), pft14=True),
        "vbeta23": _scalar(kwargs.get("vbeta23"), pft14=True),
        "q_cdrag": _scalar(boundary.drag.q_cdrag),
        "q_cdrag_pft": _scalar(boundary.drag.q_cdrag_pft, pft14=True),
        "wind": _scalar(trans_inputs.get("wind")),
        "control_salinity": _scalar(trans_inputs.get("control_salinity")),
        "control_inudate": _scalar(trans_inputs.get("control_inudate")),
        "gpp": _scalar(payload.get("gpp"), pft14=True),
        "gsmean": _scalar(payload.get("gsmean"), pft14=True),
        "rveget": _scalar(payload.get("rveget"), pft14=True),
        "rstruct": _scalar(payload.get("rstruct"), pft14=True),
        "cimean": _scalar(payload.get("cimean"), pft14=True),
        "vbeta3": _scalar(closure.vbeta3, pft14=True),
        "vbeta3pot": _scalar(closure.vbeta3pot, pft14=True),
        "assimtot": _scalar(trans.controlled_assimtot),
        "rdtot": _scalar(trans.canopy.rdtot),
        "gstot": _scalar(trans.output.gstot_m_per_s),
        "leaf_gs_top": _scalar(trans.canopy.leaf_gs_top),
        "laisum": _scalar(trans.canopy.laisum),
        "cim": _scalar(trans.canopy.cim),
        "ilai": _scalar(trans.canopy.ilai),
        "gamma_star": _scalar(trans.photo_temperature.gamma_star),
        "fvpd": _scalar(trans.vpd_boundary.fvpd),
        "g0var": _scalar(trans.photo_temperature.g0var),
    }
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare one restart-year JAX DIFFUCO/STOMATE entry step to Fortran trace.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs" / "reference_mode" / "used_run.def")
    parser.add_argument("--restart-year", type=int, default=1963)
    parser.add_argument("--previous-state-cache", type=Path, required=True)
    parser.add_argument("--diffuco-trace", type=Path, required=True)
    parser.add_argument("--trace-start-kjit", type=int, required=True)
    parser.add_argument("--day-index", type=int, required=True)
    parser.add_argument("--step-in-day", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.day_index < 1:
        parser.error("--day-index must be positive")
    if not 1 <= args.step_in_day <= 48:
        parser.error("--step-in-day must be in 1..48")

    started = time.perf_counter()
    cache = _read_state_cache(args.previous_state_cache)
    if int(cache.get("end_year", -9999)) != int(args.restart_year) - 1:
        raise ValueError("--previous-state-cache must end at restart_year - 1")
    context = prepare_paper_1961_driver_context(args.config, used_run_def_path=args.run_def)
    steps_per_stomate = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    previous_state = cache["state"]

    for day_index in range(1, int(args.day_index)):
        if day_index == 1:
            day = paper_1961_driver_restart_year_start_day_result(
                args.config,
                previous_year_end_state=previous_state,
                year=int(args.restart_year),
                used_run_def_path=context.run_def_path,
                prepared_context=context,
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
            )
        _block_until_ready(day)
        if not day.ready_for_day_end_state:
            raise RuntimeError(f"day {day_index} did not complete: {day.missing_components}")
        previous_state = day.day_end_state

    day_start_tstep = (int(args.day_index) - 1) * steps_per_stomate
    diffuco_day_static_cache = _paper_diffuco_day_static_cache_from_previous_state(
        context,
        previous_state,
        salinity=None,
        tide_height=None,
    )
    target_scaffold = None
    payload = None
    current_state = previous_state
    for offset in range(1, int(args.step_in_day) + 1):
        tstep = day_start_tstep + offset - 1
        step_scaffold = paper_1961_next_step_hydrol_precall_scaffold(
            args.config,
            previous_state=current_state,
            year=int(args.restart_year),
            tstep=tstep,
            used_run_def_path=context.run_def_path,
            prepared_context=context,
            diffuco_day_static_cache=diffuco_day_static_cache,
        )
        _block_until_ready(step_scaffold)
        if not step_scaffold.ready_for_next_state:
            raise RuntimeError(f"target day step {offset} did not complete: {step_scaffold.missing_previous_state_fields}")
        if offset == int(args.step_in_day):
            target_scaffold = step_scaffold
            payload = step_scaffold.enerbil.diffuco_local_enerbil_precall.result.payload.payload
            break
        current_state = driver_previous_step_state_from_next_step_hydrol_scaffold(step_scaffold)
    if target_scaffold is None or payload is None:
        raise RuntimeError("target scaffold not reached")
    kjit = int(args.trace_start_kjit) + (int(args.day_index) - 1) * steps_per_stomate + int(args.step_in_day) - 1
    trace = _read_trans_trace(args.diffuco_trace, kjit=kjit)
    if trace is None:
        raise RuntimeError(f"no {TAG} trace row for kjit={kjit}")

    diffuco_local = target_scaffold.enerbil.diffuco_local_enerbil_precall
    jax_values = _diffuco_values(diffuco_local)
    comparisons = {}
    for name, jax_value in jax_values.items():
        ref_value = trace.get(name)
        if jax_value is None or ref_value is None:
            comparisons[name] = {"jax": jax_value, "reference": ref_value, "available": False}
        else:
            abs_error = abs(float(jax_value) - float(ref_value))
            comparisons[name] = {
                "jax": float(jax_value),
                "reference": float(ref_value),
                "abs_error": abs_error,
                "rel_error": abs_error / max(abs(float(ref_value)), 1.0),
                "available": True,
            }

    payload_out = {
        "elapsed_seconds": time.perf_counter() - started,
        "restart_year": int(args.restart_year),
        "day_index": int(args.day_index),
        "step_in_day": int(args.step_in_day),
        "kjit": kjit,
        "previous_state_cache": str(args.previous_state_cache),
        "diffuco_trace": str(args.diffuco_trace),
        "comparisons": comparisons,
        "fortran_trace_row": trace,
        "jax_diffuco_payload_keys": sorted(payload.keys()),
        "jax_diffuco_kwargs_keys": sorted(diffuco_local.kwargs.keys()) if diffuco_local is not None else [],
        "provenance": (
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_trans_co2 lines 2338-2969",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90 after diffuco_main bridge to slowproc/STOMATE",
            "orchjax_diffuco_trans_co2_trace.txt:after_diffuco_trans_co2_pft14",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": payload_out["elapsed_seconds"], "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
