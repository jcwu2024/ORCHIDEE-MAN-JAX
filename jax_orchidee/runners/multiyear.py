from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(
    os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = Path(os.environ.get("ORCHIDEE_OUTPUT_ROOT", ROOT / "outputs")).expanduser()

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.dim2 import Dim2RestartTime, resolve_dim2_time_control  # noqa: E402
from jax_orchidee.driver.domain import forcing_file_for_year, load_case_config  # noqa: E402
from jax_orchidee.driver.init import parse_run_def, read_run_scalars  # noqa: E402
from jax_orchidee.driver.orchestration import (  # noqa: E402
    DriverPreviousStepStatePacket,
    driver_year_handoff_state_gaps,
    paper_1961_driver_cold_start_multiday_modelout_lite_run,
    paper_1961_driver_multiday_modelout_lite_run,
    paper_1961_driver_restart_year_multiday_modelout_lite_run,
    release_driver_jit_caches,
)
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference  # noqa: E402
from jax_orchidee.driver.restart import reference_restart_path  # noqa: E402
from jax_orchidee.driver.run_def_materialization import materialize_case_run_def_values  # noqa: E402
from jax_orchidee.driver.source_clock import (  # noqa: E402
    LEGACY_NATIVE_METADATA,
    _dim2_time_options,
    read_forcing_clock_metadata,
    read_source_clock_restart_bundle,
    validate_source_clock_packet_cursor,
)
from jax_orchidee.parameters.pft_catalog import load_pft_catalog  # noqa: E402
from jax_orchidee.runners.run_identity import (  # noqa: E402
    digest,
    file_identity,
    runtime_identity,
    write_atomic,
    write_json,
)
from jax_orchidee.stomate.modelout import (  # noqa: E402
    MODEL_OUTPUT_FIELD_NAMES,
    annual_history_mean_fields_from_daily_modelout,
    compute_modelout_from_fields,
    modelout_pft_selection,
    select_history_point_fields,
)
from jax_orchidee.stomate.reference import (  # noqa: E402
    pack_history_modelout_fields,
    paper_modelout_csv_targets_by_year,
    read_variables,
)

DEFAULT_BASE_USED_RUN_DEF = OUTPUT_ROOT / "server_1961_trace_full_20260623" / "run" / "used_run.def"
DEFAULT_MATERIALIZED_RUN_DEF_ROOT = OUTPUT_ROOT / "paper_250919_materialized_run_defs"
DIAGNOSTIC_HISTORY_FIELDS = (
    "LAI",
    "MAINT_RESP",
    "GROWTH_RESP",
    "VCMAX",
    "MAINT_RESP_AGRSAPST",
    "MAINT_RESP_AGRSAPPN",
    "MAINT_RESP_AGRHRTST",
    "MAINT_RESP_AGRHRTPN",
    "BM_ALLOC_LEAF",
    "BM_ALLOC_SAP_AB",
    "BM_ALLOC_SAP_BE",
    "BM_ALLOC_ROOT",
)
PRODUCTION_COMPILED_SECHIBA_DAY_DEFAULT = "on"
PRODUCTION_NUMPY_ACCUMULATOR_DEFAULT = "off"
PAPER_REFERENCE_PFT_LAYOUT_ID = "paper_250919_legacy14"
PAPER_MANGROVE_PFT_ID = "mangrove_pft14"
EXECUTION_OPTIONS = (
    "module_jit", "diffuco_local_jit", "single_pass_daily_fold", "use_static_jit_daily_carbon",
    "prebuild_day_payloads", "compiled_sechiba_day", "compiled_day_block_size", "numpy_accumulator",
    "release_jit_caches",
)
RESTART_KEYS = ("RESTART_FILEIN", "SECHIBA_RESTART_IN", "STOMATE_RESTART_FILEIN")
START_FILES = ("driver_start.nc", "sechiba_start.nc", "stomate_start.nc")


def _paper_reference_pft_index(config_path: str | Path) -> int:
    """Resolve the archived Fortran-history slot from its named stable layout."""

    config = load_case_config(config_path)
    catalog = load_pft_catalog(config["pft_catalog"]["path"])
    return catalog.layouts[PAPER_REFERENCE_PFT_LAYOUT_ID].index(PAPER_MANGROVE_PFT_ID)


def _materialized_runtime_run_def(reference, *, output_root: Path = DEFAULT_MATERIALIZED_RUN_DEF_ROOT) -> Path:
    if reference.run_def is None:
        raise FileNotFoundError(f"selected landpoint {reference.landpoint_id} has no archived output run.def")
    if reference.iteration_id is None or reference.param_set is None:
        raise ValueError(f"selected landpoint {reference.landpoint_id} is missing iteration/parameter metadata")
    base_used_run_def = reference.used_run_def or DEFAULT_BASE_USED_RUN_DEF
    values = materialize_case_run_def_values(
        base_used_run_def=base_used_run_def,
        case_run_def=reference.run_def,
        base_is_fortran_used_truth=reference.used_run_def is not None,
    )
    output = output_root / "arg2_1.0" / reference.landpoint_id / reference.iteration_id / reference.param_set
    # The prepared driver caches by path. A new deck must use a new path even
    # when another worker is materializing the same point at the same time.
    output = output / f"used_run.{digest(values)}.def"
    if not output.exists() or parse_run_def(output) != values:
        text = "".join(f"{key} = {value}\n" for key, value in sorted(values.items()))
        write_atomic(output, text.encode("utf-8"))
    return output


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


def _pft_scalar(value, *, pft_index: int) -> float:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[int(pft_index)])


def _scalar(value) -> float:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _field_series(run, field: str, *, pft_index: int) -> list[float]:
    series: list[float] = []
    for day in run.daily_modelout:
        if field in day.modelout_fields:
            series.append(_pft_scalar(day.modelout_fields[field], pft_index=pft_index))
    return series


def _delta(jax_value: float, ref_value: float) -> dict[str, float]:
    abs_error = abs(jax_value - ref_value)
    denom = max(abs(ref_value), 1.0)
    return {
        "jax": float(jax_value),
        "reference": float(ref_value),
        "abs_error": float(abs_error),
        "rel_error": float(abs_error / denom),
    }


def _paper_csv_delta(annual_modelout: dict[str, float] | None, csv_row) -> dict[str, object] | None:
    if annual_modelout is None:
        return None
    return {
        "Igrid": csv_row.i_grid,
        "age": int(csv_row.age),
        "target_year": int(csv_row.target_year),
        "reference": csv_row.values,
        "delta": {
            name: _delta(annual_modelout[name], reference)
            for name, reference in csv_row.values.items()
            if name in annual_modelout
        },
        "provenance": (
            "fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py "
            "lines 48-76 and 115-118 select age -> year, PFT14 index 13, and modelout formula"
        ),
    }


def _reference_annual_modelout(fields) -> dict[str, float]:
    modelout = compute_modelout_from_fields(fields)
    return {
        "AGB_model": _scalar(modelout.AGB_model),
        "BGB_model": _scalar(modelout.BGB_model),
        "GPP_model": _scalar(modelout.GPP_model),
        "NPP_model": _scalar(modelout.NPP_model),
    }


def _reference_annual_diagnostics(path: Path, *, reference_pft_index: int) -> dict[str, float]:
    fields = read_variables(path, DIAGNOSTIC_HISTORY_FIELDS)
    return {
        name: float(np.asarray(value, dtype=np.float64)[0, reference_pft_index, 0, 0])
        for name, value in fields.items()
    }


def _summarize_year(
    run,
    *,
    fields: tuple[str, ...],
    elapsed: float,
    handoff_gap_count: int | None,
    paper_csv_targets: dict[int, tuple[object, ...]],
    reference_history: Path | None,
    pft_index: int,
    pft_metadata: dict[str, object],
    reference_pft_index: int,
) -> dict[str, object]:
    field_summary: dict[str, object] = {}
    for field in fields:
        series = _field_series(run, field, pft_index=pft_index)
        if series:
            values = np.asarray(series, dtype=np.float64)
            field_summary[field] = {
                "count": int(values.size),
                "first": float(values[0]),
                "last": float(values[-1]),
                "mean_of_daily_values": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        else:
            field_summary[field] = {"count": 0}
    annual_modelout = None
    if run.daily_modelout:
        annual_fields = annual_history_mean_fields_from_daily_modelout(
            [day.modelout_fields for day in run.daily_modelout],
            field_names=MODEL_OUTPUT_FIELD_NAMES,
        )
        annual_selected = select_history_point_fields(annual_fields, pft_index=pft_index)
        annual_result = compute_modelout_from_fields(annual_selected)
        annual_modelout = {
            "aggregation": "arithmetic mean of completed daily STOMATE output sends",
            "completed_days": len(run.daily_modelout),
            "AGB_model": _scalar(annual_result.AGB_model),
            "BGB_model": _scalar(annual_result.BGB_model),
            "GPP_model": _scalar(annual_result.GPP_model),
            "NPP_model": _scalar(annual_result.NPP_model),
        }
    completed_days = len(run.daily_modelout)
    reference_annual_modelout = None
    reference_annual_fields = None
    reference_annual_diagnostics = None
    if reference_history is not None:
        reference_selected = select_history_point_fields(
            pack_history_modelout_fields(reference_history), pft_index=reference_pft_index,
        )
        reference_annual_modelout = _reference_annual_modelout(reference_selected)
        if completed_days == 365:
            reference_annual_fields = {name: _scalar(reference_selected[name]) for name in MODEL_OUTPUT_FIELD_NAMES}
            reference_annual_diagnostics = _reference_annual_diagnostics(
                reference_history, reference_pft_index=reference_pft_index,
            )
    annual_reference_delta = None
    annual_reference_field_delta = None
    annual_reference_diagnostic_delta = None
    annual_reference_scope = (
        "full-year annual comparison"
        if completed_days == 365
        else "not compared: yearly reference requires 365 completed daily sends"
    )
    if completed_days == 365 and annual_modelout is not None and reference_annual_modelout is not None:
        annual_reference_delta = {
            name: _delta(annual_modelout[name], reference_annual_modelout[name])
            for name in sorted(reference_annual_modelout)
            if name in annual_modelout
        }
    if completed_days == 365 and run.daily_modelout and reference_annual_fields is not None:
        annual_reference_field_delta = {
            name: _delta(_scalar(annual_selected[name]), reference_annual_fields[name])
            for name in MODEL_OUTPUT_FIELD_NAMES
        }
    if completed_days == 365 and reference_annual_diagnostics is not None:
        annual_reference_diagnostic_delta = {
            name: _delta(float(field_summary[name]["mean_of_daily_values"]), reference)
            for name, reference in reference_annual_diagnostics.items()
            if name in field_summary and int(field_summary[name].get("count", 0)) == completed_days
        }
    paper_csv_delta = None
    if completed_days == 365 and annual_modelout is not None:
        paper_csv_delta = tuple(
            item
            for item in (
                _paper_csv_delta(annual_modelout, row)
                for row in paper_csv_targets.get(int(run.year), ())
            )
            if item is not None
        )
    return {
        "year": int(run.year),
        "pft_selection": pft_metadata,
        "requested_days": int(run.requested_days),
        "closed_modelout_days": len(run.daily_modelout),
        "ready_for_requested_days": bool(run.ready_for_requested_days),
        "stopped_day_index": run.stopped_day_index,
        "missing_components": list(run.missing_components),
        "last_tstep": None if run.last_day_end_state is None else int(run.last_day_end_state.tstep),
        "source_clock": _completed_clock(run.last_day_end_state, completed_days * run.steps_per_stomate),
        "handoff_gap_count_before_year": handoff_gap_count,
        "elapsed_seconds": elapsed,
        "fields": field_summary,
        "annual_history_modelout": annual_modelout,
        "reference_annual_modelout": reference_annual_modelout,
        "annual_reference_scope": annual_reference_scope,
        "annual_reference_delta": annual_reference_delta,
        "annual_reference_field_delta": annual_reference_field_delta,
        "annual_reference_diagnostic_delta": annual_reference_diagnostic_delta,
        "paper_modelout_csv_delta": paper_csv_delta,
    }


def _days_for_years(start_year: int, years: int, days_per_year: int | list[int]) -> list[int]:
    if isinstance(days_per_year, int):
        return [int(days_per_year)] * int(years)
    if len(days_per_year) != int(years):
        raise ValueError("--days-per-year list length must match --years")
    return [int(value) for value in days_per_year]


@dataclass
class RunRequest:
    args: argparse.Namespace
    days: list[int]
    fields: tuple[str, ...]
    reference_run_dir: Path | None
    histories: dict[int, Path | None]
    identity: dict
    clocks: list[dict]
    steps_per_day: int
    predecessor: dict | None


def _declared_restart_paths(values: dict[str, str], config_path: Path, reference_run_dir: Path | None) -> tuple[Path, ...]:
    names = [values.get(key, "NONE") for key in RESTART_KEYS]
    absent = [name.upper() == "NONE" for name in names]
    if all(absent):
        return ()
    if any(absent):
        raise ValueError("declare all three restart files or all NONE")
    root = reference_run_dir
    return tuple(
        Path(name).resolve() if Path(name).is_absolute()
        else (root / name).resolve() if root is not None
        else reference_restart_path(config_path, name).resolve()
        for name in names
    )


def _file_restart_directory(paths: tuple[Path, ...], reference_run_dir: Path | None) -> Path:
    # The physical reader uses the three conventional filenames. Bind those
    # files to the very same triplet as DIM2 instead of mixing two archives.
    for directory in (paths[0].parent, reference_run_dir):
        if directory is None:
            continue
        conventional = tuple(directory / name for name in START_FILES)
        if all(candidate.is_file() and source.is_file() and candidate.samefile(source)
               for candidate, source in zip(conventional, paths, strict=True)):
            return directory
    raise ValueError("file restart requires the named driver_start.nc, sechiba_start.nc and stomate_start.nc of one case")


@lru_cache(maxsize=128)
def _forcing_clock(path: str, identity_key: str):
    return read_forcing_clock_metadata(path)


def _expected_clocks(values, forcing, day_counts, steps_per_day, restart, predecessor):
    dt = float(values["DT_SECHIBA"])
    reset, skip, length = _dim2_time_options(values)
    runtime = None if predecessor is None else validate_source_clock_packet_cursor(predecessor["state"])
    if restart:
        start = read_source_clock_restart_bundle(restart, legacy_policy=LEGACY_NATIVE_METADATA).driver
        saved = Dim2RestartTime(start.itau, start.date0, start.dt)
        if runtime is not None:
            clock = runtime.binding
            file_time = start.date0 + start.itau * start.dt / 86400.0
            cache_time = clock.date0 + int(runtime.last_applied_istp) * clock.dt / 86400.0
            if start.calendar != clock.calendar or abs(file_time - cache_time) > np.spacing(max(abs(file_time), abs(cache_time), 1.0)):
                raise ValueError("declared restart files and explicit predecessor refer to different times")
    elif runtime is not None:
        saved = Dim2RestartTime(int(runtime.last_applied_istp), runtime.binding.date0, runtime.binding.dt)
    else:
        saved = None
    clocks = []
    for asset, ndays in zip(forcing, day_counts, strict=True):
        info = _forcing_clock(asset["path"], digest(asset))
        control = resolve_dim2_time_control(
            dt_force=info.dt_force, dt_sechiba=dt, forcing_length=info.count, date0=info.date0,
            restart=Dim2RestartTime(0, info.date0, dt, reset) if reset or saved is None else saved,
            time_skip=skip, time_length=length, calendar=info.calendar,
        )
        terminal = control.istp + ndays * steps_per_day - 1
        clocks.append({
            "calendar": info.calendar, "date0": control.date0_rest, "dt": dt,
            "first_istp": control.istp, "last_istp": terminal,
        })
        saved = Dim2RestartTime(terminal, control.date0_rest, dt)
    return clocks


def _completed_clock(state, nsteps: int) -> dict | None:
    if state is None or nsteps < 1:
        return None
    runtime = validate_source_clock_packet_cursor(state)
    if runtime is None:
        return None
    binding = runtime.binding
    terminal = int(runtime.last_applied_istp)
    return {
        "calendar": binding.calendar, "date0": binding.date0, "dt": binding.dt,
        "first_istp": terminal - nsteps + 1, "last_istp": terminal,
    }


def _configuration_inputs(config: dict) -> dict:
    drivers = config["drivers"]
    return {
        "pft_catalog": file_identity(config["pft_catalog"]["path"]),
        "co2": file_identity(drivers["co2"]["file"]),
        "water_table_text": {key: file_identity(drivers["water_table_text"][key])
                             for key in ("positive_file", "differential_file")},
        "salinity": file_identity(drivers["salinity"]["file"]),
        "tide": file_identity(drivers["tide"]["file"]),
        "static_inputs": {key: file_identity(path) for key, path in config["static_inputs"].items()},
    }


def prepare_run(args: argparse.Namespace) -> RunRequest:
    """Bind one point and its requested years before reading reusable output."""
    if args.years < 1:
        raise ValueError("--years must be positive")
    days = [int(part) for part in args.days_per_year.split(",") if part.strip()]
    day_counts = _days_for_years(args.start_year, args.years, days if "," in args.days_per_year else int(args.days_per_year))
    if any(days < 1 for days in day_counts):
        raise ValueError("all day counts must be positive")
    fields = tuple(args.fields or ("GPP", "NPP", "LEAF_M"))
    reference = resolve_paper_landpoint_reference(ROOT, args.landpoint_id)
    reference_run_dir = reference.output_dir
    if args.run_def is None:
        args.run_def = _materialized_runtime_run_def(reference) if reference_run_dir is not None else DEFAULT_BASE_USED_RUN_DEF
    args.config = args.config.resolve()
    args.run_def = args.run_def.resolve()
    values = parse_run_def(args.run_def)
    declared = _declared_restart_paths({key.upper(): value for key, value in values.items()}, args.config, reference_run_dir)
    if declared and args.initial_state == "restart-backed" and args.previous_state_cache is None:
        reference_run_dir = _file_restart_directory(declared, reference_run_dir)
    predecessor = None if args.previous_state_cache is None else _read_year_end_state_cache(args.previous_state_cache)
    config = load_case_config(args.config)
    steps_per_day = int(round(float(values["DT_STOMATE"]) / float(values["DT_SECHIBA"])))
    years = list(range(args.start_year, args.start_year + args.years))
    histories_by_name = {path.name: path for path in reference.histories}
    histories = {year: histories_by_name.get(f"stomate_history_{year}.nc") for year in years}
    forcing = [file_identity(forcing_file_for_year(config, year)) for year in years]
    clocks = _expected_clocks(values, forcing, day_counts, steps_per_day, declared, predecessor)
    csv = args.paper_modelout_csv or reference.modelout_csv
    csv_zip = reference.modelout_csv_zip if csv is None else None
    source_paths = (reference_run_dir / name for name in START_FILES) if reference_run_dir is not None else (
        reference_restart_path(args.config, name) for name in START_FILES
    )
    initial_source = "previous-state-cache" if predecessor is not None else (
        "cold-start" if args.initial_state == "cold-start" else "restart-files" if declared else "reference-state"
    )
    shared_inputs = _configuration_inputs(config)
    runtime = runtime_identity()
    identity = {
        "version": 1,
        "runtime": runtime,
        "batch": digest({
            "runtime": runtime, "config": file_identity(args.config), "inputs": shared_inputs,
            "forcing": forcing, "start_year": args.start_year, "days": day_counts, "fields": fields,
            "release_jit_caches": args.release_jit_caches,
        }),
        "common": digest({
            "landpoint_id": args.landpoint_id, "start_year": args.start_year,
            "initial_state": args.initial_state, "initial_source": initial_source,
            "config": file_identity(args.config), "inputs": shared_inputs,
            "effective_run_def": values,
            "tide_source_run_def": file_identity(args.run_def.with_name("run.def")),
            "source_start": [file_identity(path) for path in source_paths],
            "declared_restart": [file_identity(path) for path in declared],
            "reference_dir": str(reference.output_dir),
            "reference_csv": file_identity(csv) if csv_zip is None else [file_identity(csv_zip.archive), csv_zip.member],
            "previous_state": file_identity(args.previous_state_cache),
            "fields": fields, "options": {key: getattr(args, key) for key in EXECUTION_OPTIONS},
            "runtime": runtime,
        }),
        "initial_source": initial_source,
        "years": [digest({
            "year": year, "days": ndays, "clock": clock, "forcing": weather, "history": file_identity(histories[year]),
        }) for year, ndays, clock, weather in zip(years, day_counts, clocks, forcing, strict=True)],
    }
    return RunRequest(
        args, day_counts, fields, reference_run_dir, histories, identity, clocks, steps_per_day, predecessor,
    )


def completed_years_match(rows, request: RunRequest, count: int) -> bool:
    return isinstance(rows, list) and len(rows) == count and all(
        isinstance(row, dict)
        and row.get("year") == int(request.args.start_year) + offset
        and row.get("requested_days") == row.get("closed_modelout_days") == request.days[offset]
        and row.get("ready_for_requested_days") is True
        and row.get("source_clock") == request.clocks[offset]
        and row.get("last_tstep") == request.days[offset] * request.steps_per_day - 1
        for offset, row in enumerate(rows)
    )


def _read_year_end_state_cache(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(f"{path} is not a driver year-end state cache")
    return payload


def _year_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("paper_driver_*_year_end_state.pkl"):
        try:
            year = int(path.name.removeprefix("paper_driver_").removesuffix("_year_end_state.pkl"))
        except ValueError:
            continue
        candidates.append((year, path))
    return sorted(candidates, reverse=True)


def _latest_year_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = _year_checkpoints(checkpoint_dir)
    return checkpoints[0][1] if checkpoints else None


def _checkpoint_identity(request: RunRequest, count: int) -> dict:
    return {
        key: request.identity[key] for key in ("version", "common", "runtime", "initial_source")
    } | {"years": request.identity["years"][:count]}


def _resume_checkpoint(request: RunRequest, pft_layout: dict):
    checkpoint_dir = request.args.year_checkpoint_dir
    for year, state_path in _year_checkpoints(checkpoint_dir):
        count = year - request.args.start_year + 1
        if not 1 <= count <= len(request.days):
            continue
        summary_path = checkpoint_dir / f"multiyear_partial_through_{year}.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict) or (
                summary.get("run_identity") != _checkpoint_identity(request, count)
                or summary.get("start_year") != request.args.start_year
                or summary.get("end_year") != year
                or summary.get("pft_layout") != pft_layout
                or not completed_years_match(summary.get("year_summaries"), request, count)
            ):
                raise ValueError("checkpoint identity or completed year sequence does not match")
            state_bytes = state_path.read_bytes()
            if hashlib.sha256(state_bytes).hexdigest() != summary.get("state_sha256"):
                raise ValueError("checkpoint state and summary are not a committed pair")
            payload = pickle.loads(state_bytes)
            stored_years = payload["year_summaries"]
            if (
                payload["run_identity"] != summary["run_identity"]
                or payload["end_year"] != year
                or json.dumps(stored_years, sort_keys=True) != json.dumps(summary["year_summaries"], sort_keys=True)
            ):
                raise ValueError("checkpoint state identity does not match its summary")
            state = payload["state"]
            if (
                _completed_clock(state, request.days[count - 1] * request.steps_per_day) != request.clocks[count - 1]
                or int(state.tstep) != summary["year_summaries"][-1]["last_tstep"]
                or driver_year_handoff_state_gaps(state)
            ):
                raise ValueError("checkpoint state does not support the recorded year handoff")
        except (OSError, ValueError, TypeError, AttributeError, KeyError, ImportError, EOFError, pickle.UnpicklingError) as exc:
            print(f"Ignoring checkpoint {state_path}: {exc}")
            continue
        return count, state, summary["year_summaries"], state_path
    return None


def _write_year_checkpoint(
    *,
    checkpoint_dir: Path,
    request: RunRequest,
    year: int,
    ndays: int,
    state: DriverPreviousStepStatePacket,
    summaries: list[dict[str, object]],
    started_at: float,
    previous_state_cache_metadata: dict[str, object] | None,
    pft_layout_metadata: dict[str, object],
) -> dict[str, str]:
    args = request.args
    state_path = checkpoint_dir / f"paper_driver_{int(year)}_year_end_state.pkl"
    summary_path = checkpoint_dir / f"multiyear_partial_through_{int(year)}.json"
    cache_payload = {
        "description": "Cached paper-case JAX driver year-end state written by run_multiyear_modelout_lite.py.",
        "config": str(args.config),
        "run_def": str(args.run_def),
        "start_year": int(args.start_year),
        "end_year": int(year),
        "days_per_year": int(ndays),
        "previous_state_cache": None if args.previous_state_cache is None else str(args.previous_state_cache),
        "previous_state_cache_metadata": previous_state_cache_metadata,
        "pft_layout": pft_layout_metadata,
        "landpoint_id": args.landpoint_id,
        "run_identity": _checkpoint_identity(request, len(summaries)),
        "created_elapsed_seconds": time.perf_counter() - started_at,
        "year_summaries": summaries,
        "state": state,
    }
    state_bytes = pickle.dumps(cache_payload, protocol=pickle.HIGHEST_PROTOCOL)
    summary_payload = {
        key: value
        for key, value in cache_payload.items()
        if key != "state"
    }
    summary_payload["state_sha256"] = hashlib.sha256(state_bytes).hexdigest()
    write_atomic(state_path, state_bytes)
    write_json(summary_path, summary_payload)
    return {"state_cache": str(state_path), "summary": str(summary_path)}


def parse_run_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="orchidee-jax run",
        description=(
            "Run consecutive PFT14 years with the production driver. "
            "Year handoffs use in-memory state packets and optional resumable "
            "checkpoints rather than NetCDF restart files."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument(
        "--run-def",
        type=Path,
        default=None,
        help=(
            "Runtime run.def for the JAX run. Defaults to a materialized selected-landpoint "
            "run.def; pass outputs/reference_mode/used_run.def for the retained cold-start "
            "paper reference-mode gate."
        ),
    )
    parser.add_argument("--start-year", type=int, default=1961)
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument(
        "--days-per-year",
        default="365",
        help="Either one integer for every year, or a comma-separated list with one value per year.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "multiyear_modelout_lite.json",
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        default=None,
        help="PFT14 modelout field to summarize. May be repeated.",
    )
    parser.add_argument("--single-pass-daily-fold", choices=("on", "off"), default="off")
    parser.add_argument("--use-static-jit-daily-carbon", choices=("on", "off"), default="on")
    parser.add_argument("--prebuild-day-payloads", choices=("on", "off"), default="on")
    parser.add_argument(
        "--compiled-sechiba-day",
        choices=("on", "off"),
        default=PRODUCTION_COMPILED_SECHIBA_DAY_DEFAULT,
    )
    parser.add_argument(
        "--compiled-day-block-size",
        type=int,
        default=0,
        help="Compile complete later-day transitions in fixed blocks; zero disables it.",
    )
    parser.add_argument(
        "--release-jit-caches", choices=("on", "off"), default="off",
        help="Release idle in-process JIT executables before later-day blocks and after each completed year.",
    )
    parser.add_argument(
        "--numpy-accumulator",
        choices=("on", "off"),
        default=PRODUCTION_NUMPY_ACCUMULATOR_DEFAULT,
    )
    parser.add_argument("--module-jit", choices=("on", "off"), default="on")
    parser.add_argument("--diffuco-local-jit", choices=("on", "off"), default="on")
    parser.add_argument("--initial-state", choices=("restart-backed", "cold-start"), default="restart-backed")
    parser.add_argument(
        "--landpoint-id",
        default="001.0-071.0",
        help=(
            "Paper landpoint used to bind forcing, run.def parameters, and optional "
            "annual reference truth."
        ),
    )
    parser.add_argument(
        "--paper-modelout-csv",
        type=Path,
        default=None,
        help="Archived paper modelout CSV. Matching target years get an additional annual CSV delta.",
    )
    parser.add_argument(
        "--previous-state-cache",
        type=Path,
        default=None,
        help="Pickle created by cache_driver_year_end_state.py for start_year - 1; starts with a restart-year handoff.",
    )
    parser.add_argument(
        "--year-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for per-year state caches and partial summaries. "
            "Use this for long paper-target runs so interrupted jobs can resume from the last completed year."
        ),
    )
    parser.add_argument(
        "--resume-checkpoints",
        choices=("on", "off"),
        default="off",
        help="Resume automatically from the latest complete year in --year-checkpoint-dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, request: RunRequest | None = None) -> int:
    if request is None:
        request = prepare_run(parse_run_args(argv))
    args = request.args
    day_counts, fields = request.days, request.fields
    reference_run_dir = request.reference_run_dir
    if args.paper_modelout_csv is None:
        paper_csv_targets = paper_modelout_csv_targets_by_year(root=ROOT, landpoint_id=args.landpoint_id)
        paper_modelout_csv_label = f"resolver:{args.landpoint_id}"
    else:
        paper_csv_targets = paper_modelout_csv_targets_by_year(args.paper_modelout_csv, landpoint_id=args.landpoint_id)
        paper_modelout_csv_label = str(args.paper_modelout_csv)

    run_scalars = read_run_scalars(args.config, run_def_path=args.run_def)
    pft_selection = modelout_pft_selection(run_scalars.pft_layout, PAPER_MANGROVE_PFT_ID)
    reference_pft_index = _paper_reference_pft_index(args.config)
    pft_layout_metadata = run_scalars.pft_layout.metadata()
    pft_selection_metadata = pft_selection.metadata()

    overall_start = time.perf_counter()
    previous_state: DriverPreviousStepStatePacket | None = None
    previous_state_cache_metadata = None
    summaries: list[dict[str, object]] = []
    start_offset = 0
    if request.predecessor is not None:
        cache_payload = request.predecessor
        if int(cache_payload.get("end_year", -9999)) != int(args.start_year) - 1:
            raise ValueError("--previous-state-cache must end at start_year - 1")
        cached_layout = cache_payload.get("pft_layout")
        if cached_layout is not None and cached_layout != pft_layout_metadata:
            raise ValueError("--previous-state-cache PFT layout does not match the configured stable IDs")
        if cached_layout is None and run_scalars.pft_layout.layout_id != "paper_250919_legacy14":
            raise ValueError("legacy state cache without stable PFT metadata is valid only for paper_250919_legacy14")
        previous_state = cache_payload["state"]
        previous_state_cache_metadata = {
            key: cache_payload.get(key)
            for key in ("description", "config", "run_def", "start_year", "end_year", "days_per_year", "created_elapsed_seconds")
        }
    if args.resume_checkpoints == "on" and args.year_checkpoint_dir is not None:
        resumed = _resume_checkpoint(request, pft_layout_metadata)
        if resumed is not None:
            start_offset, previous_state, summaries, checkpoint = resumed
            previous_state_cache_metadata = {
                "resume_checkpoint": str(checkpoint),
                "start_year": int(args.start_year),
                "end_year": int(summaries[-1]["year"]),
                "completed_years": start_offset,
            }
    checkpoint_outputs: list[dict[str, str]] = []
    failed = False
    release_jit_caches = args.release_jit_caches == "on"

    for offset, ndays in enumerate(day_counts[start_offset:], start=start_offset):
        year = int(args.start_year) + offset
        handoff_gap_count: int | None = None
        year_start = time.perf_counter()
        if offset == 0 and previous_state is not None:
            gaps = driver_year_handoff_state_gaps(previous_state)
            handoff_gap_count = len(gaps)
            run = paper_1961_driver_restart_year_multiday_modelout_lite_run(
                args.config,
                previous_year_end_state=previous_state,
                year=year,
                ndays=ndays,
                used_run_def_path=args.run_def,
                reference_run_dir=reference_run_dir,
                module_jit=args.module_jit == "on",
                diffuco_local_jit=args.diffuco_local_jit == "on",
                single_pass_daily_fold=args.single_pass_daily_fold == "on",
                use_static_jit_daily_carbon=args.use_static_jit_daily_carbon == "on",
                prebuild_day_payloads=args.prebuild_day_payloads == "on",
                use_compiled_sechiba_day=args.compiled_sechiba_day == "on",
                compiled_day_block_size=args.compiled_day_block_size,
                release_jit_caches=release_jit_caches,
                use_numpy_accumulator=args.numpy_accumulator == "on",
            )
        elif offset == 0:
            if args.initial_state == "cold-start":
                run = paper_1961_driver_cold_start_multiday_modelout_lite_run(
                    args.config,
                    year=year,
                    ndays=ndays,
                    used_run_def_path=args.run_def,
                    reference_run_dir=reference_run_dir,
                    module_jit=args.module_jit == "on",
                    diffuco_local_jit=args.diffuco_local_jit == "on",
                    single_pass_daily_fold=args.single_pass_daily_fold == "on",
                    use_static_jit_daily_carbon=args.use_static_jit_daily_carbon == "on",
                    prebuild_day_payloads=args.prebuild_day_payloads == "on",
                    use_compiled_sechiba_day=args.compiled_sechiba_day == "on",
                    compiled_day_block_size=args.compiled_day_block_size,
                    release_jit_caches=release_jit_caches,
                    use_numpy_accumulator=args.numpy_accumulator == "on",
                )
            else:
                run = paper_1961_driver_multiday_modelout_lite_run(
                    args.config,
                    year=year,
                    ndays=ndays,
                    used_run_def_path=args.run_def,
                    reference_run_dir=reference_run_dir,
                    root=ROOT,
                    compact_later_days=True,
                    module_jit=args.module_jit == "on",
                    diffuco_local_jit=args.diffuco_local_jit == "on",
                    single_pass_daily_fold=args.single_pass_daily_fold == "on",
                    use_static_jit_daily_carbon=args.use_static_jit_daily_carbon == "on",
                    prebuild_day_payloads=args.prebuild_day_payloads == "on",
                    use_compiled_sechiba_day=args.compiled_sechiba_day == "on",
                    compiled_day_block_size=args.compiled_day_block_size,
                    release_jit_caches=release_jit_caches,
                    use_numpy_accumulator=args.numpy_accumulator == "on",
                )
        else:
            gaps = driver_year_handoff_state_gaps(previous_state)
            handoff_gap_count = len(gaps)
            run = paper_1961_driver_restart_year_multiday_modelout_lite_run(
                args.config,
                previous_year_end_state=previous_state,
                year=year,
                ndays=ndays,
                used_run_def_path=args.run_def,
                reference_run_dir=reference_run_dir,
                module_jit=args.module_jit == "on",
                diffuco_local_jit=args.diffuco_local_jit == "on",
                single_pass_daily_fold=args.single_pass_daily_fold == "on",
                use_static_jit_daily_carbon=args.use_static_jit_daily_carbon == "on",
                prebuild_day_payloads=args.prebuild_day_payloads == "on",
                use_compiled_sechiba_day=args.compiled_sechiba_day == "on",
                compiled_day_block_size=args.compiled_day_block_size,
                release_jit_caches=release_jit_caches,
                use_numpy_accumulator=args.numpy_accumulator == "on",
            )
        _block_until_ready(run)
        elapsed = time.perf_counter() - year_start
        summaries.append(
            _summarize_year(
                run,
                fields=fields,
                elapsed=elapsed,
                handoff_gap_count=handoff_gap_count,
                paper_csv_targets=paper_csv_targets,
                reference_history=request.histories[year],
                pft_index=pft_selection.pft_index,
                pft_metadata=pft_selection_metadata,
                reference_pft_index=reference_pft_index,
            )
        )
        previous_state = run.last_day_end_state
        if not run.ready_for_requested_days:
            failed = True
            break
        if not completed_years_match(summaries, request, offset + 1):
            raise ValueError(f"completed year {year} disagrees with the requested day count or source clock")
        if args.year_checkpoint_dir is not None:
            checkpoint_outputs.append(
                _write_year_checkpoint(
                    checkpoint_dir=args.year_checkpoint_dir,
                    request=request,
                    year=year,
                    ndays=ndays,
                    state=previous_state,
                    summaries=summaries,
                    started_at=overall_start,
                    previous_state_cache_metadata=previous_state_cache_metadata,
                    pft_layout_metadata=pft_layout_metadata,
                )
            )
        if release_jit_caches:
            _block_until_ready(previous_state.fields_by_component)
            release_driver_jit_caches(year=year, boundary="after_year")

    payload = {
        "description": (
            "Consecutive in-memory paper-case lite modelout run. The first "
            f"year uses initial_state={args.initial_state!r}"
            + (
                " from the supplied previous-state cache"
                if args.previous_state_cache is not None
                else ""
            )
            + "; later years use the restart-year handoff runner and carried "
            "dynamic state."
        ),
        "config": str(args.config),
        "run_def": str(args.run_def),
        "start_year": int(args.start_year),
        "requested_years": int(args.years),
        "days_per_year": day_counts,
        "initial_state": args.initial_state,
        "previous_state_cache": None if args.previous_state_cache is None else str(args.previous_state_cache),
        "previous_state_cache_metadata": previous_state_cache_metadata,
        "landpoint_id": args.landpoint_id,
        "run_identity": request.identity,
        "pft_layout": pft_layout_metadata,
        "pft_selection": pft_selection_metadata,
        "module_jit": args.module_jit,
        "diffuco_local_jit": args.diffuco_local_jit,
        "single_pass_daily_fold": args.single_pass_daily_fold,
        "use_static_jit_daily_carbon": args.use_static_jit_daily_carbon,
        "prebuild_day_payloads": args.prebuild_day_payloads,
        "compiled_sechiba_day": args.compiled_sechiba_day,
        "compiled_day_block_size": args.compiled_day_block_size,
        "release_jit_caches": args.release_jit_caches,
        "numpy_accumulator": args.numpy_accumulator,
        "paper_modelout_csv": paper_modelout_csv_label,
        "paper_modelout_target_years": sorted(int(year) for year in paper_csv_targets),
        "year_checkpoint_dir": None if args.year_checkpoint_dir is None else str(args.year_checkpoint_dir),
        "year_checkpoint_outputs": checkpoint_outputs,
        "fields": list(fields),
        "ready_for_requested_years": not failed and len(summaries) == int(args.years),
        "elapsed_seconds": time.perf_counter() - overall_start,
        "years": summaries,
    }

    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ready_for_requested_years": payload["ready_for_requested_years"],
                "completed_years": len(summaries),
                "elapsed_seconds": payload["elapsed_seconds"],
                "last_year": summaries[-1] if summaries else None,
            },
            indent=2,
        )
    )
    return 0 if payload["ready_for_requested_years"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
