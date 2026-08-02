from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(
    os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = Path(os.environ.get("ORCHIDEE_OUTPUT_ROOT", ROOT / "outputs")).expanduser()

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.init import read_run_scalars  # noqa: E402
from jax_orchidee.driver.orchestration import (  # noqa: E402
    DriverPreviousStepStatePacket,
    driver_year_handoff_state_gaps,
    paper_1961_driver_cold_start_multiday_modelout_lite_run,
    paper_1961_driver_multiday_modelout_lite_run,
    paper_1961_driver_restart_year_multiday_modelout_lite_run,
)
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference  # noqa: E402
from jax_orchidee.driver.run_def_materialization import (  # noqa: E402
    materialize_case_run_def_values,
    write_materialized_run_def,
)
from jax_orchidee.stomate.modelout import (  # noqa: E402
    MODEL_OUTPUT_FIELD_NAMES,
    annual_history_mean_fields_from_daily_modelout,
    compute_modelout_from_fields,
    modelout_pft_selection,
    select_history_point_fields,
)
from jax_orchidee.stomate.reference import (  # noqa: E402
    find_stomate_reference_files,
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


def _materialized_runtime_run_def(reference, *, output_root: Path = DEFAULT_MATERIALIZED_RUN_DEF_ROOT) -> Path:
    if reference.run_def is None:
        raise FileNotFoundError(f"selected landpoint {reference.landpoint_id} has no archived output run.def")
    if reference.iteration_id is None or reference.param_set is None:
        raise ValueError(f"selected landpoint {reference.landpoint_id} is missing iteration/parameter metadata")
    output = output_root / "arg2_1.0" / reference.landpoint_id / reference.iteration_id / reference.param_set / "used_run.def"
    base_used_run_def = reference.used_run_def or DEFAULT_BASE_USED_RUN_DEF
    values = materialize_case_run_def_values(
        base_used_run_def=base_used_run_def,
        case_run_def=reference.run_def,
        base_is_fortran_used_truth=reference.used_run_def is not None,
    )
    return write_materialized_run_def(
        values,
        output,
        header_lines=(
            "# Materialized runtime run.def for local JAX paper landpoint validation.",
            f"# Base materialized defaults: {base_used_run_def}",
            f"# Archived landpoint run.def static overrides: {reference.run_def}",
            "# Restart state is bound separately through reference_run_dir.",
        ),
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


def _reference_annual_modelout(year: int, *, landpoint_id: str) -> dict[str, float] | None:
    histories = {path.name: path for path in find_stomate_reference_files(ROOT, landpoint_id=landpoint_id).histories}
    path = histories.get(f"stomate_history_{int(year)}.nc")
    if path is None:
        return None
    fields = select_history_point_fields(pack_history_modelout_fields(path))
    modelout = compute_modelout_from_fields(fields)
    return {
        "AGB_model": _scalar(modelout.AGB_model),
        "BGB_model": _scalar(modelout.BGB_model),
        "GPP_model": _scalar(modelout.GPP_model),
        "NPP_model": _scalar(modelout.NPP_model),
    }


def _reference_annual_fields(year: int, *, landpoint_id: str) -> dict[str, float] | None:
    histories = {path.name: path for path in find_stomate_reference_files(ROOT, landpoint_id=landpoint_id).histories}
    path = histories.get(f"stomate_history_{int(year)}.nc")
    if path is None:
        return None
    selected = select_history_point_fields(pack_history_modelout_fields(path))
    return {name: _scalar(selected[name]) for name in MODEL_OUTPUT_FIELD_NAMES}


def _reference_annual_diagnostics(year: int, *, landpoint_id: str) -> dict[str, float] | None:
    histories = {path.name: path for path in find_stomate_reference_files(ROOT, landpoint_id=landpoint_id).histories}
    path = histories.get(f"stomate_history_{int(year)}.nc")
    if path is None:
        return None
    fields = read_variables(path, DIAGNOSTIC_HISTORY_FIELDS)
    return {
        name: float(np.asarray(value, dtype=np.float64)[0, 13, 0, 0])
        for name, value in fields.items()
    }


def _summarize_year(
    run,
    *,
    fields: tuple[str, ...],
    elapsed: float,
    handoff_gap_count: int | None,
    paper_csv_targets: dict[int, tuple[object, ...]],
    landpoint_id: str,
    pft_index: int,
    pft_metadata: dict[str, object],
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
    reference_annual_modelout = _reference_annual_modelout(int(run.year), landpoint_id=landpoint_id)
    reference_annual_fields = (
        _reference_annual_fields(int(run.year), landpoint_id=landpoint_id)
        if completed_days == 365
        else None
    )
    reference_annual_diagnostics = (
        _reference_annual_diagnostics(int(run.year), landpoint_id=landpoint_id)
        if completed_days == 365
        else None
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


def _read_year_end_state_cache(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(f"{path} is not a driver year-end state cache")
    return payload


def _latest_year_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("paper_driver_*_year_end_state.pkl"):
        try:
            year = int(path.name.removeprefix("paper_driver_").removesuffix("_year_end_state.pkl"))
        except ValueError:
            continue
        candidates.append((year, path))
    return None if not candidates else max(candidates, key=lambda item: item[0])[1]


def _write_year_checkpoint(
    *,
    checkpoint_dir: Path,
    args: argparse.Namespace,
    year: int,
    ndays: int,
    state: DriverPreviousStepStatePacket,
    summaries: list[dict[str, object]],
    started_at: float,
    previous_state_cache_metadata: dict[str, object] | None,
    pft_layout_metadata: dict[str, object],
) -> dict[str, str]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        "created_elapsed_seconds": time.perf_counter() - started_at,
        "year_summaries": summaries,
        "state": state,
    }
    with state_path.open("wb") as handle:
        pickle.dump(cache_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary_payload = {
        key: value
        for key, value in cache_payload.items()
        if key != "state"
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return {"state_cache": str(state_path), "summary": str(summary_path)}


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

    if args.years < 1:
        raise ValueError("--years must be positive")
    if "," in str(args.days_per_year):
        day_counts = _days_for_years(
            args.start_year,
            args.years,
            [int(part.strip()) for part in str(args.days_per_year).split(",") if part.strip()],
        )
    else:
        day_counts = _days_for_years(args.start_year, args.years, int(args.days_per_year))
    if any(days < 1 for days in day_counts):
        raise ValueError("all day counts must be positive")
    fields = tuple(args.fields or ("GPP", "NPP", "LEAF_M"))
    reference = resolve_paper_landpoint_reference(ROOT, args.landpoint_id)
    reference_run_dir = reference.output_dir
    if args.run_def is None:
        args.run_def = (
            _materialized_runtime_run_def(reference)
            if reference.output_dir is not None
            else DEFAULT_BASE_USED_RUN_DEF
        )
    if args.paper_modelout_csv is None:
        paper_csv_targets = paper_modelout_csv_targets_by_year(root=ROOT, landpoint_id=args.landpoint_id)
        paper_modelout_csv_label = f"resolver:{args.landpoint_id}"
    else:
        paper_csv_targets = paper_modelout_csv_targets_by_year(args.paper_modelout_csv, landpoint_id=args.landpoint_id)
        paper_modelout_csv_label = str(args.paper_modelout_csv)

    run_scalars = read_run_scalars(args.config, run_def_path=args.run_def)
    pft_selection = modelout_pft_selection(run_scalars.pft_layout, "mangrove_pft14")
    pft_layout_metadata = run_scalars.pft_layout.metadata()
    pft_selection_metadata = pft_selection.metadata()

    overall_start = time.perf_counter()
    previous_state: DriverPreviousStepStatePacket | None = None
    previous_state_cache_metadata = None
    summaries: list[dict[str, object]] = []
    start_offset = 0
    if args.previous_state_cache is not None:
        cache_payload = _read_year_end_state_cache(args.previous_state_cache)
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
    elif args.resume_checkpoints == "on" and args.year_checkpoint_dir is not None:
        latest_checkpoint = _latest_year_checkpoint(args.year_checkpoint_dir)
        if latest_checkpoint is not None:
            cache_payload = _read_year_end_state_cache(latest_checkpoint)
            if int(cache_payload.get("start_year", -9999)) != int(args.start_year):
                raise ValueError("checkpoint start_year does not match --start-year")
            if cache_payload.get("pft_layout") != pft_layout_metadata:
                raise ValueError("checkpoint PFT layout does not match the configured stable IDs")
            cached_end_year = int(cache_payload.get("end_year", -9999))
            start_offset = cached_end_year - int(args.start_year) + 1
            if start_offset < 0 or start_offset > int(args.years):
                raise ValueError("checkpoint end_year lies outside the requested year range")
            cached_summaries = cache_payload.get("year_summaries")
            if not isinstance(cached_summaries, list) or len(cached_summaries) != start_offset:
                raise ValueError("checkpoint year summaries do not match its end_year")
            previous_state = cache_payload["state"]
            summaries = list(cached_summaries)
            previous_state_cache_metadata = {
                "resume_checkpoint": str(latest_checkpoint),
                "start_year": int(args.start_year),
                "end_year": cached_end_year,
                "completed_years": start_offset,
            }
    checkpoint_outputs: list[dict[str, str]] = []
    failed = False

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
                landpoint_id=args.landpoint_id,
                pft_index=pft_selection.pft_index,
                pft_metadata=pft_selection_metadata,
            )
        )
        previous_state = run.last_day_end_state
        if args.year_checkpoint_dir is not None and previous_state is not None and run.ready_for_requested_days:
            checkpoint_outputs.append(
                _write_year_checkpoint(
                    checkpoint_dir=args.year_checkpoint_dir,
                    args=args,
                    year=year,
                    ndays=ndays,
                    state=previous_state,
                    summaries=summaries,
                    started_at=overall_start,
                    previous_state_cache_metadata=previous_state_cache_metadata,
                    pft_layout_metadata=pft_layout_metadata,
                )
            )
        if not run.ready_for_requested_days:
            failed = True
            break

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
        "pft_layout": pft_layout_metadata,
        "pft_selection": pft_selection_metadata,
        "module_jit": args.module_jit,
        "diffuco_local_jit": args.diffuco_local_jit,
        "single_pass_daily_fold": args.single_pass_daily_fold,
        "use_static_jit_daily_carbon": args.use_static_jit_daily_carbon,
        "prebuild_day_payloads": args.prebuild_day_payloads,
        "compiled_sechiba_day": args.compiled_sechiba_day,
        "compiled_day_block_size": args.compiled_day_block_size,
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
