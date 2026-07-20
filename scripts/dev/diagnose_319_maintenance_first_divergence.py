from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_day_scaffold,
    paper_1961_driver_later_day_runtime_result,
    paper_1961_next_step_hydrol_precall_scaffold,
    paper_1961_next_step_runtime_result,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference  # noqa: E402
from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    explicitsnow_compactn_step,
    explicitsnow_fall_step,
    explicitsnow_gone_step,
    explicitsnow_levels_step,
    explicitsnow_melt_refrz_step,
    explicitsnow_profile_step,
    explicitsnow_transf_step,
    snow3lheat_explicit,
    snow3lliq_explicit,
    snow3ltemp_explicit,
)
from jax_orchidee.stomate.carbon_kernels import ILEAF, IROOT  # noqa: E402


LANDPOINT_ID = "319.0-057.0"
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
TRACE = ROOT / "outputs/reference_mode/trace_319_dayend_1961_20260718/orchjax_319_maint_dayend.txt"
THERMOSOIL_TRACE = ROOT / "outputs/reference_mode/trace_319_dayend_1961_20260718/orchjax_319_thermosoil_dayend.txt"
FORTRAN_HISTORY = ROOT / "outputs/reference_mode/trace_319_dayend_1961_20260718/fortran_sechiba_history_1961.nc"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/diagnose_319_maintenance_first_divergence.json"


def _run_def() -> Path:
    reference = resolve_paper_landpoint_reference(ROOT, LANDPOINT_ID)
    if reference.iteration_id is None or reference.param_set is None:
        raise ValueError(f"incomplete reference metadata for {LANDPOINT_ID}")
    return (
        ROOT
        / "outputs/paper_250919_materialized_run_defs/arg2_1.0"
        / LANDPOINT_ID
        / reference.iteration_id
        / reference.param_set
        / "used_run.def"
    )


def _fortran_records(path: Path) -> dict[int, dict[str, float]]:
    tokens = path.read_text(encoding="ascii").split()
    records: dict[int, dict[str, float]] = {}
    index = 0
    while index < len(tokens):
        if tokens[index] != "maint_dayend":
            index += 1
            continue
        values = tokens[index + 1 : index + 8]
        if len(values) != 7:
            raise ValueError(f"truncated maint_dayend record at token {index}")
        itime = int(values[0])
        if itime % 48:
            raise ValueError(f"unexpected non-day-end itime {itime}")
        records[itime // 48] = {
            "lai": float(values[1]),
            "t2m": float(values[2]),
            "t2m_longterm": float(values[3]),
            "resp_maint_radia": float(values[4]),
            "resp_maint_leaf": float(values[5]),
            "resp_maint_root": float(values[6]),
        }
        index += 8
    if not records:
        raise ValueError(f"no maint_dayend records in {path}")
    return records


def _fortran_soil_records(path: Path, tag: str) -> dict[int, dict[str, np.ndarray]]:
    """Read the existing day-end per-layer trace without changing its scope.

    ``apply_319_dayend_state_trace.py`` writes four real-valued fields per
    layer after ``tag, kjit, jsl``.  The field names differ only in that
    HYDROL's first field is the soil moisture used by THERMOSOIL.
    """

    tokens = path.read_text(encoding="ascii").split()
    rows: dict[int, dict[str, list[float]]] = {}
    index = 0
    while index < len(tokens):
        if tokens[index] != tag:
            index += 1
            continue
        values = tokens[index + 1 : index + 7]
        if len(values) != 6:
            raise ValueError(f"truncated {tag} record at token {index}")
        itime, layer = int(values[0]), int(values[1])
        if itime % 48:
            raise ValueError(f"unexpected non-day-end {tag} itime {itime}")
        day = itime // 48
        record = rows.setdefault(day, {"first": [], "second": [], "third": [], "fourth": []})
        if layer != len(record["first"]) + 1:
            raise ValueError(f"unexpected {tag} layer order at day {day}: {layer}")
        for key, value in zip(("first", "second", "third", "fourth"), values[2:], strict=True):
            record[key].append(float(value))
        index += 7
    if not rows:
        raise ValueError(f"no {tag} records in {path}")
    return {day: {key: np.asarray(value, dtype=np.float64) for key, value in row.items()} for day, row in rows.items()}


def _fortran_snow_history_metadata(path: Path) -> dict[str, object]:
    """Describe the temporal semantics of the existing Fortran snow history.

    These XIOS fields are daily means, not day-end restart state.  They must
    never be compared to ``hydrol_previous_step_state`` directly.
    """

    with xr.open_dataset(path, decode_times=False) as dataset:
        required = ("snow", "snowtemp", "snowdz", "snowrho")
        missing = tuple(name for name in required if name not in dataset)
        if missing:
            raise ValueError(f"Fortran history missing snow fields: {missing}")
        semantics = {
            name: {
                "online_operation": dataset[name].attrs.get("online_operation"),
                "interval_operation": dataset[name].attrs.get("interval_operation"),
                "interval_write": dataset[name].attrs.get("interval_write"),
                "cell_methods": dataset[name].attrs.get("cell_methods"),
            }
            for name in required
        }
    if any(item["online_operation"] != "average" for item in semantics.values()):
        raise ValueError("expected daily-average Fortran snow history fields")
    return {
        "file": str(path.relative_to(ROOT)),
        "comparison_status": "not_compared_to_day_end_state",
        "reason": "XIOS daily means require JAX half-hour history aggregation before numerical comparison.",
        "fields": semantics,
    }


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64)[0, 13])


def _record(day, *, day_index: int) -> dict[str, float]:
    fold = day.daily_process_fold
    if fold is None:
        raise RuntimeError(f"day {day_index}: no daily process fold")
    accumulated = np.asarray(fold.maintenance.resp_maint_part, dtype=np.float64)[0, 13]
    final_step = fold.maintenance.step_results[-1]
    radia = np.asarray(final_step.resp_maint_part, dtype=np.float64)[0, 13]
    fields = (
        day.first_day_stomate_state.fields
        if day_index == 1
        else day.day_end_state.fields_by_component["slowproc_stomate_previous_step_state"]
    )
    result = {
        "resp_maint_radia": float(radia.sum()),
        "resp_maint_leaf": float(accumulated[ILEAF]),
        "resp_maint_root": float(accumulated[IROOT]),
        "resp_maint_total": float(accumulated.sum()),
        "gpp_daily": _scalar(fields["gpp_daily"]),
        "leaf_biomass_after_turnover": float(np.asarray(fields["biomass"], dtype=np.float64)[0, 13, ILEAF, 0]),
        "root_biomass_after_turnover": float(np.asarray(fields["biomass"], dtype=np.float64)[0, 13, IROOT, 0]),
    }
    if day_index == 1:
        npp = day.stomate_daily_carbon.post_npp.daily_carbon.npp_update
        result.update(npp=_scalar(npp.npp), resp_growth=_scalar(npp.resp_growth))
    else:
        modelout = day.daily_modelout.modelout_fields
        result.update(npp=_scalar(modelout["NPP"]), resp_growth=_scalar(modelout["GROWTH_RESP"]))
    return result


def _jax_soil_state(day) -> dict[str, np.ndarray]:
    fields = day.day_end_state.fields_by_component
    therm = fields["thermosoil_previous_step_state"]
    hydrol = fields["hydrol_previous_step_state"]
    return {
        "stempdiag": np.asarray(therm["stempdiag"], dtype=np.float64)[0],
        "shum_ngrnd_permalong": np.asarray(therm["shum_ngrnd_permalong"], dtype=np.float64)[0, :, 13],
        "snow": np.asarray(hydrol["snow"], dtype=np.float64).reshape(-1),
        "snowtemp": np.asarray(hydrol["snowtemp"], dtype=np.float64).reshape(-1),
        "snowdz": np.asarray(hydrol["snowdz"], dtype=np.float64).reshape(-1),
        "snowrho": np.asarray(hydrol["snowrho"], dtype=np.float64).reshape(-1),
    }


def _soil_comparison(jax: dict[str, np.ndarray], therm_fortran: dict[str, np.ndarray]) -> dict[str, object]:
    fortran = {
        "stempdiag": therm_fortran["first"],
    }
    fields = {}
    for name, reference in fortran.items():
        delta = jax[name] - reference
        layer = int(np.argmax(np.abs(delta)))
        fields[name] = {
            "max_abs_error": float(np.max(np.abs(delta))),
            "layer": layer + 1,
            "jax": float(jax[name][layer]),
            "fortran": float(reference[layer]),
            "delta": float(delta[layer]),
        }
    return fields


def _array_comparison(jax: np.ndarray, fortran: np.ndarray) -> dict[str, object]:
    if jax.shape != fortran.shape:
        return {"shape_match": False, "jax_shape": list(jax.shape), "fortran_shape": list(fortran.shape)}
    delta = jax - fortran
    index = int(np.argmax(np.abs(delta)))
    return {
        "shape_match": True,
        "max_abs_error": float(np.max(np.abs(delta))),
        "index": index + 1,
        "jax": float(jax[index]),
        "fortran": float(fortran[index]),
        "delta": float(delta[index]),
    }


def _comparison(jax: dict[str, float], fortran: dict[str, float]) -> dict[str, object]:
    fields = ("resp_maint_radia", "resp_maint_leaf", "resp_maint_root")
    deltas = {name: float(jax[name] - fortran[name]) for name in fields}
    return {
        "fortran": fortran,
        "jax": jax,
        "deltas": deltas,
        "max_abs_error": max(abs(value) for value in deltas.values()),
    }


def _float_array(value) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _snow_snapshot(*, snowrho, snowdz, snowtemp, snowheat, snowliq) -> dict[str, object]:
    """Serialize only explicit-snow state, preserving the production kernels."""

    return {
        "snow": _float_array(np.sum(np.asarray(snowrho) * np.asarray(snowdz), axis=1)),
        "snowrho": _float_array(snowrho),
        "snowdz": _float_array(snowdz),
        "snowtemp": _float_array(snowtemp),
        "snowheat": _float_array(snowheat),
        "snowliq": _float_array(snowliq),
    }


def _explicit_snow_substep_trace(precall, *, temp_air, pb, u, v) -> dict[str, object]:
    """Trace one production explicit-snow call without changing its equations.

    The inputs are exactly the HYDROL pre-call payload assembled by the driver.
    Each record is the output of the same JAX kernel used by
    ``explicitsnow_main_step``; this diagnostic is deliberately not a separate
    implementation of the snow physics.
    """

    values = precall.payload
    dtype = np.asarray(values["snowdz"]).dtype
    totfrac_nobio = values.get("totfrac_nobio", 1.0 - values["vegtot"])
    fall = explicitsnow_fall_step(
        precip_snow=values["precip_snow"], temp_air=temp_air, u=u, v=v,
        totfrac_nobio=totfrac_nobio, snowrho=values["snowrho"],
        snowdz=values["snowdz"], snowheat=values["snowheat"],
        snowgrain=values["snowgrain"], snowtemp=values["snowtemp"],
    )
    stages = {
        "input": _snow_snapshot(
            snowrho=values["snowrho"], snowdz=values["snowdz"], snowtemp=values["snowtemp"],
            snowheat=values["snowheat"], snowliq=values["snowliq"],
        ),
        "fall": _snow_snapshot(
            snowrho=fall.snowrho, snowdz=fall.snowdz, snowtemp=values["snowtemp"],
            snowheat=fall.snowheat, snowliq=values["snowliq"],
        ),
    }
    levels = explicitsnow_levels_step(np.sum(np.asarray(fall.snowdz), axis=1), dtype=dtype)
    transf = explicitsnow_transf_step(
        snowdz_old=fall.snowdz, snowdz=levels.snowdz, snowrho=fall.snowrho,
        snowheat=fall.snowheat, snowgrain=fall.snowgrain,
    )
    has_snow = np.sum(np.asarray(transf.snowdz), axis=1) > 0.0
    snowtemp = np.where(
        has_snow[:, None],
        np.asarray(snow3ltemp_explicit(transf.snowheat, transf.snowrho, transf.snowdz)),
        273.15,
    )
    snowliq = np.where(
        has_snow[:, None],
        np.asarray(snow3lliq_explicit(transf.snowheat, transf.snowrho, transf.snowdz, snowtemp)),
        0.0,
    )
    stages["transf_and_diagnose"] = _snow_snapshot(
        snowrho=transf.snowrho, snowdz=transf.snowdz, snowtemp=snowtemp,
        snowheat=transf.snowheat, snowliq=snowliq,
    )
    compact = explicitsnow_compactn_step(snowtemp=snowtemp, snowrho=transf.snowrho, snowdz=transf.snowdz)
    snowheat = snow3lheat_explicit(snowliq, compact.snowrho, compact.snowdz, snowtemp)
    stages["compact"] = _snow_snapshot(
        snowrho=compact.snowrho, snowdz=compact.snowdz, snowtemp=snowtemp,
        snowheat=snowheat, snowliq=snowliq,
    )
    profile = explicitsnow_profile_step(
        cgrnd_snow=values["cgrnd_snow"], dgrnd_snow=values["dgrnd_snow"],
        lambda_snow=values["lambda_snow"], temp_sol_new=values["temp_sol_new"],
        snowtemp=snowtemp, snowdz=compact.snowdz, temp_sol_add=values["temp_sol_add"],
    )
    stages["profile"] = _snow_snapshot(
        snowrho=compact.snowrho, snowdz=compact.snowdz, snowtemp=profile.snowtemp,
        snowheat=snowheat, snowliq=snowliq,
    )
    gone = explicitsnow_gone_step(
        pgflux=values["pgflux"], snowheat=snowheat, snowtemp=profile.snowtemp,
        snowdz=compact.snowdz, snowrho=compact.snowrho, snowliq=snowliq,
        grndflux=np.zeros_like(np.asarray(values["pgflux"])), snowmelt=np.zeros_like(np.asarray(values["snow"])),
        soilflxresid=values["soilflxresid"],
    )
    stages["gone"] = _snow_snapshot(
        snowrho=gone.snowrho, snowdz=gone.snowdz, snowtemp=gone.snowtemp,
        snowheat=snowheat, snowliq=gone.snowliq,
    )
    melt = explicitsnow_melt_refrz_step(
        precip_rain=values["precip_rain"], pgflux=values["pgflux"], soilcap=values["soilcap"],
        snowtemp=gone.snowtemp, snowdz=gone.snowdz, snowrho=gone.snowrho, snowliq=gone.snowliq,
        snowmelt=gone.snowmelt, grndflux=gone.grndflux, temp_air=temp_air,
        soilflxresid=values["soilflxresid"],
    )
    stages["melt_refrz"] = _snow_snapshot(
        snowrho=melt.snowrho, snowdz=melt.snowdz, snowtemp=melt.snowtemp,
        snowheat=snowheat, snowliq=melt.snowliq,
    )
    totsnowheat = np.sum(np.asarray(snowheat), axis=1)
    pgflux = np.asarray(values["pgflux"], dtype=np.float64)
    soilflxresid = np.asarray(values["soilflxresid"], dtype=np.float64)
    return {
        "inputs": {
            "precip_snow": _float_array(values["precip_snow"]),
            "precip_rain": _float_array(values["precip_rain"]),
            "temp_air": _float_array(temp_air),
            "pgflux": _float_array(pgflux),
            "soilflxresid": _float_array(soilflxresid),
            "gone_energy_threshold": _float_array(-totsnowheat / 1800.0),
            "gone_energy_available": _float_array(pgflux + soilflxresid),
        },
        "stages": stages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compiled-sechiba-day",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exercise the production 48-step compiled SECHIBA transition",
    )
    parser.add_argument(
        "--capture-first-snow",
        action="store_true",
        help="capture only the production explicit-snow substeps at the known first-snow tstep",
    )
    parser.add_argument(
        "--capture-tstep",
        type=int,
        default=9528,
        help="zero-based first half-hour step to inspect with --capture-first-snow",
    )
    parser.add_argument(
        "--capture-count",
        type=int,
        default=24,
        help="number of consecutive half-hour steps to retain with --capture-first-snow",
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be positive")

    fortran = _fortran_records(TRACE)
    thermosoil_fortran = _fortran_soil_records(THERMOSOIL_TRACE, "thermosoil_dayend")
    snow_history_metadata = _fortran_snow_history_metadata(FORTRAN_HISTORY)
    run_def = _run_def()
    reference = resolve_paper_landpoint_reference(ROOT, LANDPOINT_ID)
    if reference.output_dir is None:
        raise FileNotFoundError(f"reference run unavailable for {LANDPOINT_ID}")
    context = prepare_paper_1961_driver_context(
        CONFIG,
        used_run_def_path=run_def,
        reference_run_dir=reference.output_dir,
    )
    first = paper_1961_driver_cold_start_day_scaffold(
        CONFIG,
        year=1961,
        used_run_def_path=run_def,
        reference_run_dir=reference.output_dir,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
    )
    if not first.ready_for_first_day_end_state:
        raise RuntimeError(f"day 1 incomplete: {first.missing_components}")

    rows = []
    previous = first.first_day_end_state
    current = first
    first_bad_day = None
    for day_index in range(1, args.days + 1):
        if day_index > 1:
            capture_day = args.capture_tstep // 48 + 1
            if args.capture_first_snow and day_index == capture_day:
                # Day 199 begins at t=9504.  Advance only the needed 24
                # production half-hour states, then retain HYDROL's real snow
                # closure through the remainder of the snowy day.
                first_day_tstep = (day_index - 1) * 48
                for tstep in range(first_day_tstep, args.capture_tstep):
                    current_step = paper_1961_next_step_runtime_result(
                        CONFIG,
                        previous_state=previous,
                        year=1961,
                        tstep=tstep,
                        used_run_def_path=run_def,
                        prepared_context=context,
                        module_jit=True,
                        diffuco_local_jit=True,
                    )
                    previous = current_step.next_state
                traces = []
                for tstep in range(args.capture_tstep, args.capture_tstep + args.capture_count):
                    scaffold = paper_1961_next_step_hydrol_precall_scaffold(
                        CONFIG,
                        previous_state=previous,
                        year=1961,
                        tstep=tstep,
                        used_run_def_path=run_def,
                        prepared_context=context,
                        module_jit=True,
                        diffuco_local_jit=True,
                    )
                    if scaffold.hydrol_precall is None or scaffold.hydrol_module is None:
                        raise RuntimeError(f"tstep {tstep}: HYDROL snow scaffold unavailable: {scaffold.missing_previous_state_fields}")
                    snow_step = scaffold.hydrol_module.snow_step
                    traces.append({
                        "tstep": tstep,
                        "trace": _explicit_snow_substep_trace(
                            scaffold.hydrol_precall,
                            temp_air=scaffold.enerbil.payload.temp_air,
                            pb=scaffold.enerbil.payload.pb,
                            u=scaffold.enerbil.payload.u,
                            v=scaffold.enerbil.payload.v,
                        ),
                        "production_main": _snow_snapshot(
                            snowrho=snow_step.snow_state.snowrho,
                            snowdz=snow_step.snow_state.snowdz,
                            snowtemp=snow_step.snow_state.snowtemp,
                            snowheat=snow_step.snow_state.snowheat,
                            snowliq=snow_step.snowliq,
                        ),
                    })
                    current_step = paper_1961_next_step_runtime_result(
                        CONFIG,
                        previous_state=previous,
                        year=1961,
                        tstep=tstep,
                        used_run_def_path=run_def,
                        prepared_context=context,
                        module_jit=True,
                        diffuco_local_jit=True,
                    )
                    previous = current_step.next_state
                payload = {
                    "scope": "319.0-057.0 first-snow explicit-snow substep diagnostic",
                    "fortran_provenance": "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 108-553",
                    "first_tstep": args.capture_tstep,
                    "day": day_index,
                    "compiled_sechiba_day_for_prior_state": bool(args.compiled_sechiba_day),
                    "traces": traces,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"day": day_index, "first_tstep": args.capture_tstep, "output": str(args.output)}, indent=2))
                return 0
            current = paper_1961_driver_later_day_runtime_result(
                CONFIG,
                previous_state=previous,
                day_index=day_index,
                year=1961,
                start_tstep=(day_index - 1) * 48,
                used_run_def_path=run_def,
                reference_run_dir=reference.output_dir,
                prepared_context=context,
                module_jit=True,
                diffuco_local_jit=True,
                use_static_jit_daily_carbon=True,
                prebuild_day_payloads=True,
                use_compiled_sechiba_day=args.compiled_sechiba_day,
                retain_stomate_step_results=True,
            )
            if not current.ready_for_day_end_state:
                raise RuntimeError(f"day {day_index} incomplete: {current.missing_components}")
            previous = current.day_end_state
        else:
            previous = first.first_day_end_state
        if day_index not in fortran:
            continue
        row = {
            "day": day_index,
            **_comparison(_record(current, day_index=day_index), fortran[day_index]),
        }
        if day_index > 1:
            jax_soil = _jax_soil_state(current)
            row["soil_state"] = _soil_comparison(jax_soil, thermosoil_fortran[day_index])
            row["snow_history"] = snow_history_metadata
        rows.append(row)
        if not args.capture_first_snow and first_bad_day is None and row["max_abs_error"] > args.atol:
            first_bad_day = day_index
            break

    payload = {
        "scope": "319.0-057.0 cold-start 1961 STOMATE maintenance first-divergence diagnostic",
        "fortran_provenance": (
            "outputs/server_trace_patch/apply_319_dayend_state_trace.py inserts the trace after "
            "stomate.f90 maintenance accumulation; it is not a post-turnover biomass trace."
        ),
        "jax_provenance": (
            "jax_orchidee.stomate.daily.stomate_daily_process_fold_from_entries; "
            "jax_orchidee.driver.orchestration paper day transitions."
        ),
        "tolerance": float(args.atol),
        "days_requested": int(args.days),
        "compiled_sechiba_day": bool(args.compiled_sechiba_day),
        "days_compared": len(rows),
        "first_bad_day": first_bad_day,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("days_compared", "first_bad_day")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
