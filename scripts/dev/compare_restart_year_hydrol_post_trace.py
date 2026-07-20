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
    _driver_wind_z0_from_previous_state,
    _paper_1961_step_bundle_from_context,
    _paper_diffuco_day_static_cache_from_previous_state,
    _paper_diffuco_day_static_cache_from_firstcall_inputs,
    build_intersurf_first_step_payload,
    driver_previous_step_state_from_next_step_hydrol_scaffold,
    paper_1961_driver_later_day_runtime_result,
    paper_1961_driver_restart_year_start_day_result,
    paper_1961_next_step_hydrol_precall_scaffold,
    prepare_paper_1961_driver_context,
    rebase_driver_state_for_year_start,
)
from jax_orchidee.trace.server_1961 import trace_schema  # noqa: E402


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


def _scalar(value, *index) -> float:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    arr = np.asarray(value, dtype=np.float64)
    if index:
        return float(arr[index])
    return float(arr.reshape(-1)[0])


def _vector(value, *prefix) -> list[float]:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    arr = np.asarray(value, dtype=np.float64)
    if prefix:
        arr = arr[prefix]
    return [float(item) for item in arr.reshape(-1)]


def _read_state_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(f"{path} is not a driver year-end state cache")
    return payload


def _read_fixed_trace_rows(path: Path, *, tag: str, fields: tuple[str, ...], match: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    known_tags = {
        "pre",
        "post_tile",
        "post_layer",
        "mc_after_update",
        "alt_first_solve",
        "alt_residual",
    }
    while index < len(lines):
        parts = lines[index].split()
        index += 1
        if not parts or parts[0] != tag:
            continue
        data: list[str] = parts[1:]
        while len(data) < len(fields) and index < len(lines):
            next_parts = lines[index].split()
            if next_parts and next_parts[0] in known_tags:
                break
            data.extend(next_parts)
            index += 1
        if len(data) < len(fields):
            continue
        row: dict[str, object] = {"tag": tag}
        for name, text in zip(fields, data, strict=False):
            if name in {"kjit", "ji", "jst", "jsl"}:
                row[name] = int(float(text))
            elif name in {"branch_peat", "resolv"}:
                row[name] = text.upper().startswith("T")
            else:
                row[name] = float(text)
        if all(row.get(key) == value for key, value in match.items()):
            rows.append(row)
    return rows


def _comparison(jax_value: float, ref_value: object) -> dict[str, object]:
    if not isinstance(ref_value, (float, int)):
        return {"jax": jax_value, "reference": ref_value, "available": False}
    error = abs(float(jax_value) - float(ref_value))
    return {
        "jax": float(jax_value),
        "reference": float(ref_value),
        "abs_error": error,
        "rel_error": error / max(abs(float(ref_value)), 1.0),
        "available": True,
    }


def _read_alt_residual_rows(path: Path, *, match: dict[str, int]) -> list[dict[str, object]]:
    fields = (
        "kjit",
        "ji",
        "jst",
        "jsl",
        "resolv",
        "rhs",
        "tmat_e",
        "tmat_f",
        "tmat_g1",
        "mcl_after_residual_solve",
    )
    rows: list[dict[str, object]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        parts = lines[index].split()
        index += 1
        if not parts or parts[0] != "alt_residual":
            continue
        data: list[str] = parts[1:]
        while len(data) < len(fields) and index < len(lines):
            next_parts = lines[index].split()
            if next_parts and next_parts[0] == "alt_residual":
                break
            data.extend(next_parts)
            index += 1
        if len(data) < len(fields):
            continue
        row: dict[str, object] = {"tag": "alt_residual"}
        for name, text in zip(fields, data, strict=False):
            if name in {"kjit", "ji", "jst", "jsl"}:
                row[name] = int(float(text))
            elif name == "resolv":
                row[name] = text.upper().startswith("T")
            else:
                row[name] = float(text)
        if all(row.get(key) == value for key, value in match.items()):
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare one restart-year JAX HYDROL post step to Fortran post trace.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs" / "reference_mode" / "used_run.def")
    parser.add_argument("--restart-year", type=int, default=1963)
    parser.add_argument("--previous-state-cache", type=Path, required=True)
    parser.add_argument("--hydrol-main-trace", type=Path)
    parser.add_argument("--hydrol-post-trace", type=Path, required=True)
    parser.add_argument("--hydrol-update-trace", type=Path)
    parser.add_argument("--hydrol-alt-residual-trace", type=Path)
    parser.add_argument("--trace-start-kjit", type=int, required=True)
    parser.add_argument("--day-index", type=int, required=True)
    parser.add_argument("--step-in-day", type=int, required=True)
    parser.add_argument("--scan-through-step", action="store_true")
    parser.add_argument("--soil-tile", type=int, default=4)
    parser.add_argument("--pft-index", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-module-jit", action="store_true")
    args = parser.parse_args()

    if args.day_index < 1:
        parser.error("--day-index must be positive")
    if not 1 <= args.step_in_day <= 48:
        parser.error("--step-in-day must be in 1..48")
    if args.soil_tile < 1:
        parser.error("--soil-tile must be one-based")
    if args.pft_index < 1:
        parser.error("--pft-index must be one-based")

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
                module_jit=not args.no_module_jit,
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
                module_jit=not args.no_module_jit,
            )
        _block_until_ready(day)
        if not day.ready_for_day_end_state:
            raise RuntimeError(f"day {day_index} did not complete: {day.missing_components}")
        previous_state = day.day_end_state

    target_day_initial_state = previous_state
    day_start_tstep = (int(args.day_index) - 1) * steps_per_stomate
    if int(args.day_index) == 1:
        target_day_initial_state = rebase_driver_state_for_year_start(previous_state)
        day_start_bundle = _paper_1961_step_bundle_from_context(
            context,
            year=int(args.restart_year),
            tstep=0,
        )
        day_start_payload = build_intersurf_first_step_payload(
            day_start_bundle,
            driver_z0_for_wind=_driver_wind_z0_from_previous_state(target_day_initial_state),
            dt_sechiba=context.dt_sechiba,
        )
        diffuco_day_static_cache = _paper_diffuco_day_static_cache_from_firstcall_inputs(
            context,
            target_day_initial_state,
            salinity=day_start_payload.salinity,
            tide_height=day_start_payload.tide_height,
        )
    else:
        diffuco_day_static_cache = _paper_diffuco_day_static_cache_from_previous_state(
            context,
            previous_state,
            salinity=None,
            tide_height=None,
        )
    target_scaffold = None
    scanned_steps: list[dict[str, object]] = []
    current_state = target_day_initial_state
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
            module_jit=not args.no_module_jit,
        )
        _block_until_ready(step_scaffold)
        if not step_scaffold.ready_for_next_state:
            raise RuntimeError(f"target day step {offset} did not complete: {step_scaffold.missing_previous_state_fields}")
        if args.scan_through_step:
            step_kjit = int(args.trace_start_kjit) + (int(args.day_index) - 1) * steps_per_stomate + offset - 1
            if step_scaffold.hydrol_module is None or step_scaffold.hydrol_module.module is None:
                raise RuntimeError(f"HYDROL module missing at scanned step {offset}")
            step_module = step_scaffold.hydrol_module.module
            step_tile = step_module.soil.tile_results[int(args.soil_tile) - 1]
            step_rows = _read_fixed_trace_rows(
                args.hydrol_post_trace,
                tag="post_tile",
                fields=trace_schema("hydrol_post").tags["post_tile"].fields,
                match={"kjit": step_kjit, "ji": 1, "jst": int(args.soil_tile)},
            )
            if len(step_rows) == 1:
                scanned_steps.append(
                    {
                        "step_in_day": offset,
                        "kjit": step_kjit,
                        "tmci": _comparison(_scalar(step_tile.solve.tmci), step_rows[0].get("tmci")),
                        "tmcf": _comparison(_scalar(step_tile.solve.tmcf), step_rows[0].get("tmcf")),
                        "flux_top": _comparison(_scalar(step_tile.surface.flux_top), step_rows[0].get("flux_top")),
                        "rootsink_sum": _comparison(
                            float(np.sum(np.asarray(step_module.split.rootsink)[0, :, int(args.soil_tile) - 1])),
                            step_rows[0].get("rootsink_sum"),
                        ),
                        "dr_ns": _comparison(_scalar(step_tile.solve.dr_ns_final), step_rows[0].get("dr_ns")),
                        "k_bottom": _comparison(_scalar(step_tile.k_bottom), step_rows[0].get("k_bottom")),
                        "evap_bare_lim_out": None
                        if step_scaffold.hydrol_module.diagnostics.evap_bare_lim is None
                        else _scalar(step_scaffold.hydrol_module.diagnostics.evap_bare_lim),
                    }
                )
        if offset == int(args.step_in_day):
            target_scaffold = step_scaffold
            break
        current_state = driver_previous_step_state_from_next_step_hydrol_scaffold(step_scaffold)
    if target_scaffold is None or target_scaffold.hydrol_module is None or target_scaffold.hydrol_module.module is None:
        raise RuntimeError("target HYDROL module not reached")

    kjit = int(args.trace_start_kjit) + (int(args.day_index) - 1) * steps_per_stomate + int(args.step_in_day) - 1
    schema = trace_schema("hydrol_post")
    tile_fields = schema.tags["post_tile"].fields
    layer_fields = schema.tags["post_layer"].fields
    tile_rows = _read_fixed_trace_rows(
        args.hydrol_post_trace,
        tag="post_tile",
        fields=tile_fields,
        match={"kjit": kjit, "ji": 1, "jst": int(args.soil_tile)},
    )
    layer_rows = _read_fixed_trace_rows(
        args.hydrol_post_trace,
        tag="post_layer",
        fields=layer_fields,
        match={"kjit": kjit, "ji": 1, "jst": int(args.soil_tile)},
    )
    if len(tile_rows) != 1:
        raise RuntimeError(f"expected one post_tile row for kjit={kjit}, jst={args.soil_tile}; found {len(tile_rows)}")
    if not layer_rows:
        raise RuntimeError(f"no post_layer rows for kjit={kjit}, jst={args.soil_tile}")
    layer_rows = sorted(layer_rows, key=lambda row: int(row["jsl"]))

    hydrol = target_scaffold.hydrol_module
    module = hydrol.module
    diagnostics = hydrol.diagnostics
    module_inputs = hydrol.module_inputs
    tile_index = int(args.soil_tile) - 1
    pft_index = int(args.pft_index) - 1
    tile = module.soil.tile_results[tile_index]
    tile_ref = tile_rows[0]
    jax_tile_values = {
        "tmci": _scalar(tile.solve.tmci),
        "tmcf": _scalar(tile.solve.tmcf),
        "flux_top": _scalar(tile.surface.flux_top),
        "rootsink_sum": float(np.sum(np.asarray(module.split.rootsink)[0, :, tile_index])),
        "dr_ns": _scalar(tile.solve.dr_ns_final),
        "dr_corrnum_ns": _scalar(tile.solve.dr_corrnum),
        "check_tr_ns": _scalar(tile.solve.check_tr_ns),
        "k_bottom": _scalar(tile.k_bottom),
    }
    tile_comparisons = {
        name: _comparison(value, tile_ref.get(name))
        for name, value in jax_tile_values.items()
    }

    jax_mc = _vector(module.soil.mc, 0, slice(None), tile_index)
    jax_mcl = _vector(module.soil.mcl, 0, slice(None), tile_index)
    layer_comparisons = []
    for offset, row in enumerate(layer_rows):
        layer_comparisons.append(
            {
                "jsl": int(row["jsl"]),
                "mcl": _comparison(jax_mcl[offset], row.get("mcl")),
                "mc": _comparison(jax_mc[offset], row.get("mc")),
            }
        )

    jax_layer_states = {
        "input_mc_layers": _vector(module_inputs["mc"], 0, slice(None), tile_index)
        if "mc" in module_inputs
        else None,
        "input_mcl_layers": _vector(module_inputs["mcl"], 0, slice(None), tile_index)
        if "mcl" in module_inputs
        else None,
        "mcl_after_tridiag_layers": _vector(tile.solve.mcl_after_tridiag, 0, slice(None)),
        "mc_after_mcl_update_layers": _vector(tile.solve.mc_after_mcl_update, 0, slice(None)),
        "mc_final_layers": _vector(tile.solve.mc_final, 0, slice(None)),
        "mcl_final_layers": _vector(module.soil.mcl, 0, slice(None), tile_index),
    }
    pre_layer_comparisons = None
    if args.hydrol_main_trace is not None:
        pre_rows = _read_fixed_trace_rows(
            args.hydrol_main_trace,
            tag="pre",
            fields=trace_schema("hydrol_main").tags["pre"].fields,
            match={"kjit": kjit, "ji": 1, "jst": int(args.soil_tile)},
        )
        pre_rows = sorted(pre_rows, key=lambda row: int(row["jsl"]))
        pre_layer_comparisons = [
            {
                "jsl": int(row["jsl"]),
                "input_mc": _comparison(jax_layer_states["input_mc_layers"][offset], row.get("mc")),
                "input_mcl": _comparison(jax_layer_states["input_mcl_layers"][offset], row.get("mcl")),
                "mclint": _comparison(jax_layer_states["input_mcl_layers"][offset], row.get("mclint")),
                "rootsink": _comparison(_scalar(module.split.rootsink, 0, offset, tile_index), row.get("rootsink")),
                "b": _comparison(_scalar(tile.b_for_rhs, 0, offset), row.get("b"))
                if hasattr(tile, "b_for_rhs")
                else {"available": False},
                "setup_e": _comparison(_scalar(tile.setup.e, 0, offset), row.get("e")),
                "setup_f": _comparison(_scalar(tile.setup.f, 0, offset), row.get("f")),
                "setup_g1": _comparison(_scalar(tile.setup.g1, 0, offset), row.get("g1")),
                "setup_ep": _comparison(_scalar(tile.setup.ep, 0, offset), row.get("ep")),
                "setup_fp": _comparison(_scalar(tile.setup.fp, 0, offset), row.get("fp")),
                "setup_gp": _comparison(_scalar(tile.setup.gp, 0, offset), row.get("gp")),
                "rhs": _comparison(_scalar(tile.solve.rhs, 0, offset), row.get("rhs")),
                "tmat_e": _comparison(_scalar(tile.solve.rhs.e, 0, offset), row.get("tmat_e"))
                if hasattr(tile.solve.rhs, "e")
                else _comparison(_scalar(tile.setup.e, 0, offset), row.get("tmat_e")),
                "tmat_f": _comparison(_scalar(tile.setup.f, 0, offset), row.get("tmat_f")),
                "tmat_g1": _comparison(_scalar(tile.setup.g1, 0, offset), row.get("tmat_g1")),
            }
            for offset, row in enumerate(pre_rows)
        ]
    update_layer_comparisons = None
    if args.hydrol_update_trace is not None:
        update_rows = _read_fixed_trace_rows(
            args.hydrol_update_trace,
            tag="mc_after_update",
            fields=trace_schema("hydrol_update").tags["mc_after_update"].fields,
            match={"kjit": kjit, "ji": 1, "jst": int(args.soil_tile)},
        )
        update_rows = sorted(update_rows, key=lambda row: int(row["jsl"]))
        update_layer_comparisons = [
            {
                "jsl": int(row["jsl"]),
                "mc_after_mcl_update": _comparison(jax_layer_states["mc_after_mcl_update_layers"][offset], row.get("mc")),
                "mcl_after_update": _comparison(jax_layer_states["mcl_final_layers"][offset], row.get("mcl")),
            }
            for offset, row in enumerate(update_rows)
        ]

    jax_diagnostics = {
        "humrel": _scalar(diagnostics.humrel, 0, pft_index),
        "humrelv_tile": _scalar(diagnostics.humrelv, 0, pft_index, tile_index),
        "vegstressv_tile": _scalar(diagnostics.vegstressv, 0, pft_index, tile_index),
        "us_tile_layers": _vector(diagnostics.us, 0, pft_index, tile_index, slice(None)),
        "nroot_layers": _vector(diagnostics.nroot, 0, pft_index, slice(None)),
        "soil_wet_ns_tile_layers": _vector(diagnostics.soil_wet_ns, 0, slice(None), tile_index),
        "profil_froz_tile_layers": _vector(module_inputs["profil_froz"], 0, slice(None), tile_index)
        if "profil_froz" in module_inputs
        else None,
        "temp_hydro_layers": _vector(module_inputs["temp_hydro"], 0, slice(None))
        if "temp_hydro" in module_inputs
        else None,
        "stempdiag_layers": _vector(module_inputs["stempdiag"], 0, slice(None))
        if "stempdiag" in module_inputs
        else None,
        "coef_after_infilt_bottom": None
        if tile.coef_after_infilt is None
        else {
            "mc": _scalar(tile.infilt.mc, 0, -1),
            "mc_used": _scalar(tile.coef_after_infilt.mc_used, 0, -1),
            "bin_i": int(np.asarray(tile.coef_after_infilt.bin_i)[0, -1]),
            "bin_offset": int(np.asarray(tile.coef_after_infilt.bin_offset)[0, -1]),
            "k": _scalar(tile.coef_after_infilt.k, 0, -1),
            "k_eval": _scalar(tile.coef_after_infilt.k_eval, 0, -1),
            "k_floor_raw": _scalar(tile.coef_after_infilt.k_floor_raw, 0, -1),
            "a_raw": _scalar(tile.coef_after_infilt.a_raw, 0, -1),
            "b_raw": _scalar(tile.coef_after_infilt.b_raw, 0, -1),
            "d_raw": _scalar(tile.coef_after_infilt.d_raw, 0, -1),
            "kfact_root": _scalar(module_inputs["kfact_root"], 0, -1, tile_index)
            if "kfact_root" in module_inputs
            else None,
        },
        "post_corrections": {
            "over_mcs_has_value": tile.solve.over_mcs is not None,
            "over_mcs_routing_has_value": tile.solve.over_mcs_routing is not None,
            "over_mcs_ru_corr_sum": None
            if tile.solve.over_mcs_routing is None
            else float(np.sum(np.asarray(tile.solve.over_mcs_routing.ru_corr_ns))),
            "over_mcs_dr_corr": None
            if tile.solve.over_mcs_routing is None
            else _scalar(tile.solve.over_mcs_routing.dr_corr_ns),
            "forced_water_table_has_value": tile.solve.forced_water_table is not None,
            "forced_water_table_dr_force": None
            if tile.solve.forced_water_table is None
            else _scalar(tile.solve.forced_water_table.dr_force_ns),
            "forced_water_table_dmc_sum": None
            if tile.solve.forced_water_table is None
            else float(np.sum(np.asarray(tile.solve.forced_water_table.dmc))),
            "negative_runoff_has_value": tile.solve.negative_runoff is not None,
            "negative_runoff_corr": None
            if tile.solve.negative_runoff is None
            else _scalar(tile.solve.negative_runoff.ru_corr2_ns),
            "under_mcr_has_value": tile.solve.under_mcr is not None,
            "under_mcr_excess_top": None
            if tile.solve.under_mcr is None
            else _scalar(tile.solve.under_mcr.excess_top),
        },
        "is_under_mcr_tile": bool(np.asarray(module.soil.is_under_mcr)[0, tile_index]),
        "ae_ns_tile": _scalar(module.split.ae_ns, 0, tile_index),
        "tr_ns_tile": _scalar(module.split.tr_ns, 0, tile_index),
        "precisol_ns_tile": _scalar(module.split.precisol_ns, 0, tile_index),
        "vevapnu_ns_tile": _scalar(module.split.vevapnu_ns, 0, tile_index),
        "evap_bare_lim_out": None if diagnostics.evap_bare_lim is None else _scalar(diagnostics.evap_bare_lim),
        "evap_bare_lim_ns_out_tile": None
        if diagnostics.evap_bare_lim_ns is None
        else _scalar(diagnostics.evap_bare_lim_ns, 0, tile_index),
    }
    selected_inputs: dict[str, object] = {}
    for name in (
        "vevapnu",
        "evap_bare_lim",
        "tot_bare_soil",
        "humrel",
        "transpir",
        "veget_max",
    ):
        value = module_inputs.get(name)
        if value is None:
            selected_inputs[name] = None
        elif name in {"humrel", "transpir", "veget_max"}:
            selected_inputs[f"{name}_pft"] = _scalar(value, 0, pft_index)
        else:
            selected_inputs[name] = _scalar(value)
    for name in (
        "ae_ns",
        "evap_bare_lim_ns",
        "water2infilt",
        "soiltile",
        "precisol_ns",
    ):
        value = module_inputs.get(name)
        if value is None:
            selected_inputs[name] = None
        else:
            selected_inputs[f"{name}_tile"] = _scalar(value, 0, tile_index)
    selected_inputs["frac_bare_ns_tile"] = _scalar(module.vegupd.frac_bare_ns, 0, tile_index)
    selected_inputs["vegetmax_soil_pft_tile"] = _scalar(module.vegupd.vegetmax_soil, 0, pft_index, tile_index)
    selected_inputs["vevapnu_old"] = _scalar(module.split.vevapnu_old)
    evap_alt = module.soil.evap_bare_limit_alt
    alt_diagnostics = None
    if evap_alt is not None:
        tmcint = _scalar(module.soil.tmcint_for_evap_bare_limit, 0, tile_index)
        tmc_dummy = _scalar(evap_alt.tmc, 0, tile_index)
        flux_bottom = _scalar(evap_alt.flux_bottom, 0, tile_index)
        frac_bare = _scalar(module.vegupd.frac_bare_ns, 0, tile_index)
        evapot_corr = _scalar(module_inputs["evapot_corr"])
        bare_mask = 1.0 if frac_bare + 1.0 - 1.0e-8 >= 1.0 else 0.0
        water_budget = tmcint - tmc_dummy - flux_bottom
        alt_diagnostics = {
            "first_mcl_top": _scalar(evap_alt.first.mcl, 0, 0, tile_index),
            "potential_flux_top": evapot_corr * bare_mask,
            "tmcint": tmcint,
            "tmc_dummy": tmc_dummy,
            "flux_bottom": flux_bottom,
            "water_budget_evap": water_budget,
            "bare_weighted_evap": water_budget * frac_bare,
            "evapot": _scalar(module_inputs["evapot"]),
            "evapot_corr": evapot_corr,
            "frac_bare_ns_tile": frac_bare,
        }
        if args.hydrol_alt_residual_trace is not None:
            ref_alt_rows = _read_alt_residual_rows(
                args.hydrol_alt_residual_trace,
                match={"kjit": kjit, "ji": 1, "jst": int(args.soil_tile)},
            )
            ref_alt_rows = sorted(ref_alt_rows, key=lambda row: int(row["jsl"]))
            alt_diagnostics["resolv_alt"] = bool(np.asarray(evap_alt.resolv_alt)[0, tile_index])
            alt_diagnostics["saved_rhs_layers"] = _vector(evap_alt.saved.rhs, 0, slice(None), tile_index)
            alt_diagnostics["saved_e_layers"] = _vector(evap_alt.saved.e, 0, slice(None), tile_index)
            alt_diagnostics["saved_f_layers"] = _vector(evap_alt.saved.f, 0, slice(None), tile_index)
            alt_diagnostics["saved_g1_layers"] = _vector(evap_alt.saved.g1, 0, slice(None), tile_index)
            alt_diagnostics["first_mcl_layers"] = _vector(evap_alt.first.mcl, 0, slice(None), tile_index)
            alt_diagnostics["residual_rhs_layers"] = _vector(evap_alt.residual_equations.rhs, 0, slice(None), tile_index)
            alt_diagnostics["residual_e_layers"] = _vector(evap_alt.residual_equations.e, 0, slice(None), tile_index)
            alt_diagnostics["residual_f_layers"] = _vector(evap_alt.residual_equations.f, 0, slice(None), tile_index)
            alt_diagnostics["residual_g1_layers"] = _vector(evap_alt.residual_equations.g1, 0, slice(None), tile_index)
            alt_diagnostics["second_mcl_layers"] = _vector(evap_alt.second.mcl, 0, slice(None), tile_index)
            alt_diagnostics["fortran_alt_residual_rows"] = ref_alt_rows
            alt_diagnostics["residual_layer_comparisons"] = [
                {
                    "jsl": int(row["jsl"]),
                    "rhs": _comparison(
                        _scalar(evap_alt.residual_equations.rhs, 0, int(row["jsl"]) - 1, tile_index),
                        row["rhs"],
                    ),
                    "tmat_e": _comparison(
                        _scalar(evap_alt.residual_equations.e, 0, int(row["jsl"]) - 1, tile_index),
                        row["tmat_e"],
                    ),
                    "tmat_f": _comparison(
                        _scalar(evap_alt.residual_equations.f, 0, int(row["jsl"]) - 1, tile_index),
                        row["tmat_f"],
                    ),
                    "tmat_g1": _comparison(
                        _scalar(evap_alt.residual_equations.g1, 0, int(row["jsl"]) - 1, tile_index),
                        row["tmat_g1"],
                    ),
                    "mcl_after_residual_solve": _comparison(
                        _scalar(evap_alt.second.mcl, 0, int(row["jsl"]) - 1, tile_index),
                        row["mcl_after_residual_solve"],
                    ),
                }
                for row in ref_alt_rows
            ]

    payload = {
        "elapsed_seconds": time.perf_counter() - started,
        "restart_year": int(args.restart_year),
        "day_index": int(args.day_index),
        "step_in_day": int(args.step_in_day),
        "kjit": kjit,
        "soil_tile": int(args.soil_tile),
        "pft_index": int(args.pft_index),
        "previous_state_cache": str(args.previous_state_cache),
        "hydrol_post_trace": str(args.hydrol_post_trace),
        "fortran_post_tile": tile_ref,
        "tile_comparisons": tile_comparisons,
        "layer_comparisons": layer_comparisons,
        "pre_layer_comparisons": pre_layer_comparisons,
        "update_layer_comparisons": update_layer_comparisons,
        "jax_layer_states": jax_layer_states,
        "scanned_steps": scanned_steps,
        "jax_diagnostics": jax_diagnostics,
        "jax_split_inputs": selected_inputs,
        "jax_alt_diagnostics": alt_diagnostics,
        "provenance": (
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 6007-6052 and 6565-6725",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_diag_soil lines 9043-9107",
            "orchjax_hydrol_post_trace.txt:post_tile/post_layer",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": payload["elapsed_seconds"],
                "tile_comparisons": tile_comparisons,
                "jax_diagnostics": jax_diagnostics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
