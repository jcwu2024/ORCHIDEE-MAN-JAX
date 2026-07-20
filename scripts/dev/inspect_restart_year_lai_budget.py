from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_later_day_scaffold,
    paper_1961_driver_later_day_runtime_result,
    paper_1961_driver_restart_year_start_day_result,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.stomate.carbon_kernels import (  # noqa: E402
    ICARBON,
    ILEAF,
    IROOT,
    ISAPABOVE,
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


def _read_state_cache(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(f"{path} is not a driver year-end state cache")
    return payload


def _arr(value) -> np.ndarray:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    return np.asarray(value, dtype=np.float64)


def _pft14(value) -> float:
    return float(_arr(value).reshape(-1)[13])


def _pft14_part(value, part: int) -> float:
    return float(_arr(value)[0, 13, part, ICARBON])


def _pft14_leaf_age_class(value, leaf_class: int) -> float:
    return float(_arr(value)[0, 13, leaf_class])


def _component_state(packet, component: str):
    fields = packet.fields_by_component[component]
    return type("ComponentState", (), fields)()


def _history_rows(path: Path, days: tuple[int, ...]) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with nc.Dataset(path) as ds:
        for day in days:
            idx = int(day) - 1
            rows[int(day)] = {
                "LAI": float(ds.variables["LAI"][idx, 13, 0, 0]),
                "GPP": float(ds.variables["GPP"][idx, 13, 0, 0]),
                "LEAF_M": float(ds.variables["LEAF_M"][idx, 13, 0, 0]),
                "LEAF_TURN": float(ds.variables["LEAF_TURN"][idx, 13, 0, 0]),
                "LEAF_BM_LITTER": float(ds.variables["LEAF_BM_LITTER"][idx, 13, 0, 0]),
                "BM_ALLOC_LEAF": float(ds.variables["BM_ALLOC_LEAF"][idx, 13, 0, 0]),
                "LEAF_AGE": float(ds.variables["LEAF_AGE"][idx, 13, 0, 0]),
            }
    return rows


def _day_state_summary(day) -> dict[str, object]:
    daily = getattr(day, "stomate_daily_carbon", None)
    state = _component_state(day.day_end_state, "slowproc_stomate_previous_step_state")
    daily_modelout = getattr(day, "daily_modelout", None)
    stomate_outputs = getattr(day, "stomate_outputs", None)
    modelout_fields = (
        daily_modelout.modelout_fields
        if daily_modelout is not None
        else stomate_outputs.modelout_fields
        if stomate_outputs is not None
        else {}
    )
    summary: dict[str, object] = {
        "ready": bool(day.ready_for_day_end_state),
        "start_tstep": int(day.start_tstep),
        "end_tstep": int(getattr(day, "end_tstep", int(day.start_tstep + day.steps_per_stomate - 1))),
        "modelout_GPP": _pft14(modelout_fields["GPP"]) if "GPP" in modelout_fields else None,
        "modelout_LEAF_M": _pft14(modelout_fields["LEAF_M"]) if "LEAF_M" in modelout_fields else None,
        "state_lai": _pft14(state.lai),
        "state_sla_calc": _pft14(state.sla_calc),
        "state_leaf_biomass": _pft14_part(state.biomass, ILEAF),
        "state_root_biomass": _pft14_part(state.biomass, IROOT),
        "state_sapabove_biomass": _pft14_part(state.biomass, ISAPABOVE),
        "state_leaf_age_weighted": sum(
            _pft14_leaf_age_class(state.leaf_age, i) * _pft14_leaf_age_class(state.leaf_frac, i)
            for i in range(4)
        ),
        "state_leaf_age": [_pft14_leaf_age_class(state.leaf_age, i) for i in range(4)],
        "state_leaf_frac": [_pft14_leaf_age_class(state.leaf_frac, i) for i in range(4)],
    }
    if daily is not None:
        post_npp = daily.post_npp
        daily_carbon = post_npp.daily_carbon
        summary.update(
            {
                "after_npp_leaf_biomass": _pft14_part(daily_carbon.npp_update.biomass, ILEAF),
                "after_npp_bm_alloc_leaf": _pft14_part(daily_carbon.npp_update.bm_alloc, ILEAF),
                "after_npp_sla_calc": _pft14(daily_carbon.age_sla.sla_calc),
                "after_npp_leaf_age_weighted": _pft14(daily_carbon.age_sla.leaf_age_weighted),
                "after_npp_leaf_age": [
                    _pft14_leaf_age_class(daily_carbon.age_sla.leaf_age, i) for i in range(4)
                ],
                "after_npp_leaf_frac": [
                    _pft14_leaf_age_class(daily_carbon.age_sla.leaf_frac, i) for i in range(4)
                ],
                "after_turnover_leaf_biomass": _pft14_part(post_npp.turnover.biomass, ILEAF)
                if post_npp.turnover is not None
                else None,
                "turnover_leaf": _pft14_part(post_npp.turnover.turnover, ILEAF) if post_npp.turnover is not None else None,
                "turnover_leaf_meanage": _pft14(post_npp.turnover.leaf_meanage) if post_npp.turnover is not None else None,
                "turnover_leaf_age_crit": _pft14(post_npp.turnover.leaf_age_crit)
                if post_npp.turnover is not None
                else None,
                "turnover_leaf_age": [
                    _pft14_leaf_age_class(post_npp.turnover.leaf_age, i) for i in range(4)
                ]
                if post_npp.turnover is not None
                else None,
                "turnover_leaf_frac": [
                    _pft14_leaf_age_class(post_npp.turnover.leaf_frac, i) for i in range(4)
                ]
                if post_npp.turnover is not None
                else None,
                "bm_to_litter_leaf": _pft14_part(post_npp.bm_to_litter, ILEAF),
                "lai_after_setlai": _pft14(post_npp.lai_after_setlai),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect restart-year PFT14 LAI budget around a target day window.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs" / "reference_mode" / "used_run.def")
    parser.add_argument("--restart-year", type=int, default=1964)
    parser.add_argument("--previous-state-cache", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--start-day", type=int, default=233)
    parser.add_argument("--end-day", type=int, default=236)
    parser.add_argument(
        "--sample-days",
        default=None,
        help="Optional comma-separated day indices to retain/summarize; otherwise all days in start/end window.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.start_day < 1 or args.end_day < args.start_day:
        parser.error("invalid day window")
    cache = _read_state_cache(args.previous_state_cache)
    if int(cache.get("end_year", -9999)) != int(args.restart_year) - 1:
        raise ValueError("--previous-state-cache must end at restart_year - 1")
    context = prepare_paper_1961_driver_context(args.config, used_run_def_path=args.run_def)
    steps_per_stomate = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    previous_state = cache["state"]
    sample_days = (
        set(range(int(args.start_day), int(args.end_day) + 1))
        if args.sample_days is None
        else {int(item.strip()) for item in str(args.sample_days).split(",") if item.strip()}
    )
    if not sample_days:
        parser.error("--sample-days did not contain any day indices")
    if min(sample_days) < int(args.start_day) or max(sample_days) > int(args.end_day):
        parser.error("--sample-days must fall inside --start-day/--end-day")

    rows: list[dict[str, object]] = []
    for day_index in range(1, int(args.end_day) + 1):
        if day_index == 1:
            day = paper_1961_driver_restart_year_start_day_result(
                args.config,
                previous_year_end_state=previous_state,
                year=int(args.restart_year),
                used_run_def_path=context.run_def_path,
                prepared_context=context,
            )
        elif day_index in sample_days:
            day = paper_1961_driver_later_day_scaffold(
                args.config,
                previous_state=previous_state,
                year=int(args.restart_year),
                start_tstep=(day_index - 1) * steps_per_stomate,
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                retain_stomate_step_results=False,
                runtime_entry_payloads=True,
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
        if day_index in sample_days:
            rows.append({"day": int(day_index), "jax": _day_state_summary(day)})

    history = _history_rows(args.history, tuple(sorted(sample_days)))
    for row in rows:
        day_index = int(row["day"])
        ref = history[day_index]
        jax = row["jax"]
        row["fortran_history"] = ref
        if isinstance(jax, dict):
            row["delta"] = {
                "LAI": float(jax["state_lai"]) - ref["LAI"],
                "LEAF_M": float(jax["state_leaf_biomass"]) - ref["LEAF_M"],
                "GPP": float(jax["modelout_GPP"]) - ref["GPP"],
                "LEAF_TURN": (float(jax["turnover_leaf"]) if jax.get("turnover_leaf") is not None else np.nan)
                - ref["LEAF_TURN"],
                "BM_ALLOC_LEAF": (
                    float(jax["after_npp_bm_alloc_leaf"]) if jax.get("after_npp_bm_alloc_leaf") is not None else np.nan
                )
                - ref["BM_ALLOC_LEAF"],
            }
    payload = {
        "elapsed_seconds": time.perf_counter() - started,
        "restart_year": int(args.restart_year),
        "day_window": [int(args.start_day), int(args.end_day)],
        "sample_days": sorted(sample_days),
        "previous_state_cache": str(args.previous_state_cache),
        "history": str(args.history),
        "rows": rows,
        "provenance": (
            "fortran_source/ORCHIDEE/src_stomate/stomate_lai.f90::setlai lines 58-88",
            "fortran_source/ORCHIDEE/src_stomate/stomate_npp.f90::npp_calc lines 544-672",
            "fortran_source/ORCHIDEE/src_stomate/stomate_turnover.f90::turn lines 569-665",
            "stomate_history_1964.nc daily fields LAI/LEAF_M/LEAF_TURN/BM_ALLOC_LEAF",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": payload["elapsed_seconds"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
