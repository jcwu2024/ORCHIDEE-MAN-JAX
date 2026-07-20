from __future__ import annotations

import argparse
import json
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
    paper_1961_driver_multiday_modelout_lite_run,
    paper_1961_driver_multiday_modelout_run,
)
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference  # noqa: E402
from jax_orchidee.driver.run_def_materialization import (  # noqa: E402
    materialize_case_run_def_values,
    write_materialized_run_def,
)
from jax_orchidee.stomate.modelout import (  # noqa: E402
    annual_history_mean_fields_from_daily_modelout,
    compute_modelout_from_fields,
    select_history_point_fields,
)
from jax_orchidee.stomate.reference import find_stomate_reference_files, pack_history_modelout_fields  # noqa: E402


DEFAULT_BASE_USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
DEFAULT_MATERIALIZED_RUN_DEF_ROOT = ROOT / "outputs" / "paper_250919_materialized_run_defs"


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


def _pft14_scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[13])


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _history_day_modelout(fields: dict[str, object], day_index: int):
    day_offset = int(day_index) - 1
    time_len = int(np.asarray(next(iter(fields.values()))).shape[0])
    if day_offset >= time_len:
        return None, None
    day_fields = {name: np.asarray(value)[day_offset : day_offset + 1] for name, value in fields.items()}
    selected = select_history_point_fields(day_fields)
    return selected, compute_modelout_from_fields(selected)


def _delta(jax_value: float, ref_value: float) -> dict[str, float]:
    abs_error = abs(jax_value - ref_value)
    denom = max(abs(ref_value), 1.0)
    return {
        "jax": jax_value,
        "reference": ref_value,
        "abs_error": abs_error,
        "rel_error": abs_error / denom,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare local JAX multiday modelout fields against the paper-case Fortran STOMATE history."
    )
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--year", type=int, default=1961)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument(
        "--run-def",
        type=Path,
        default=None,
        help=(
            "Runtime run.def for the JAX run. Defaults to a materialized selected-landpoint "
            "run.def; pass outputs/reference_mode/used_run.def only for the retained "
            "001.0-071.0 cold-start reference-mode gate."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("lite", "scaffold"), default="lite")
    parser.add_argument("--initial-state", choices=("restart-backed", "cold-start"), default="restart-backed")
    parser.add_argument("--compact-later-days", choices=("on", "off"), default="on")
    parser.add_argument("--single-pass-daily-fold", choices=("on", "off"), default="off")
    parser.add_argument("--use-static-jit-daily-carbon", choices=("on", "off"), default="on")
    parser.add_argument("--prebuild-day-payloads", choices=("on", "off"), default="on")
    parser.add_argument("--compiled-sechiba-day", choices=("on", "off"), default="off")
    parser.add_argument(
        "--landpoint-id",
        default="001.0-071.0",
        help=(
            "Paper reference landpoint used for STOMATE history lookup. "
            "The JAX driver inputs must separately match this landpoint for parity assertions."
        ),
    )
    args = parser.parse_args()

    reference = resolve_paper_landpoint_reference(ROOT, args.landpoint_id)
    reference_run_dir = reference.output_dir
    if args.run_def is None:
        args.run_def = (
            _materialized_runtime_run_def(reference)
            if reference.output_dir is not None
            else ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
        )
    histories = {path.name: path for path in find_stomate_reference_files(ROOT, landpoint_id=args.landpoint_id).histories}
    history_path = histories[f"stomate_history_{int(args.year)}.nc"]
    history_fields = pack_history_modelout_fields(history_path)

    start = time.perf_counter()
    if args.mode == "lite" and args.initial_state == "cold-start":
        run = paper_1961_driver_cold_start_multiday_modelout_lite_run(
            args.config,
            year=args.year,
            ndays=args.days,
            used_run_def_path=args.run_def,
            reference_run_dir=reference_run_dir,
            single_pass_daily_fold=args.single_pass_daily_fold == "on",
            use_static_jit_daily_carbon=args.use_static_jit_daily_carbon == "on",
            prebuild_day_payloads=args.prebuild_day_payloads == "on",
            use_compiled_sechiba_day=args.compiled_sechiba_day == "on",
        )
    elif args.mode == "lite":
        run = paper_1961_driver_multiday_modelout_lite_run(
            args.config,
            year=args.year,
            ndays=args.days,
            used_run_def_path=args.run_def,
            reference_run_dir=reference_run_dir,
            root=ROOT,
            compact_later_days=args.compact_later_days == "on",
            single_pass_daily_fold=args.single_pass_daily_fold == "on",
            use_static_jit_daily_carbon=args.use_static_jit_daily_carbon == "on",
            prebuild_day_payloads=args.prebuild_day_payloads == "on",
            use_compiled_sechiba_day=args.compiled_sechiba_day == "on",
        )
    else:
        if args.initial_state == "cold-start":
            raise ValueError("--initial-state cold-start is currently available only with --mode lite")
        run = paper_1961_driver_multiday_modelout_run(
            args.config,
            year=args.year,
            ndays=args.days,
            used_run_def_path=args.run_def,
            reference_run_dir=reference_run_dir,
            root=ROOT,
        )
    elapsed = time.perf_counter() - start

    comparisons = []
    yearly_record = None
    for daily in run.daily_modelout:
        reference_fields, reference_modelout = _history_day_modelout(history_fields, daily.day_index)
        if daily.day_index == 1 and reference_fields is not None and reference_modelout is not None:
            yearly_record = {
                "history_record_index": 1,
                "fields": {
                    "GPP": _delta(_pft14_scalar(daily.modelout_fields["GPP"]), _scalar(reference_fields["GPP"])),
                    "NPP_model": _delta(_pft14_scalar(daily.modelout.NPP_model), _scalar(reference_modelout.NPP_model)),
                    "AGB_model": _delta(_pft14_scalar(daily.modelout.AGB_model), _scalar(reference_modelout.AGB_model)),
                    "LEAF_M": _delta(_pft14_scalar(daily.modelout_fields["LEAF_M"]), _scalar(reference_fields["LEAF_M"])),
                },
                "notes": (
                    "This records the numeric distance between JAX day 1 and the single yearly/sparse history "
                    "record only as an alignment diagnostic. It is not a daily parity assertion."
                ),
            }
        comparisons.append(
            {
                "day": daily.day_index,
                "start_tstep": daily.start_tstep,
                "end_tstep": daily.end_tstep,
                "daily_reference_available": False,
                "history_record_available": reference_fields is not None and reference_modelout is not None,
                "notes": (
                    "The paper-case STOMATE history file is yearly/output-frequency data with one time record; "
                    "daily reference comparison for this day requires an aligned daily trace, not stomate_history_1961.nc."
                ),
            }
        )

    annual_history = None
    annual_reference_fields, annual_reference_modelout = _history_day_modelout(history_fields, 1)
    if run.daily_modelout:
        daily_fields = [daily.modelout_fields for daily in run.daily_modelout]
        annual_fields = annual_history_mean_fields_from_daily_modelout(daily_fields)
        annual_selected = select_history_point_fields(annual_fields)
        annual_modelout = compute_modelout_from_fields(annual_selected)
        if annual_reference_fields is not None and annual_reference_modelout is not None:
            annual_history = {
                "aggregation": "arithmetic mean of completed local daily STOMATE output sends",
                "completed_days": len(run.daily_modelout),
                "requested_days": args.days,
                "history_axes": "(time, vegetation, lat, lon)",
                "reference_history": str(history_path),
                "comparison_scope": (
                    "Diagnostic only unless the local run configuration, forcing year, restart mode, "
                    "and paper sensitivity parameters match the referenced stomate_history file."
                ),
                "fields": {
                    "GPP": _delta(_scalar(annual_selected["GPP"]), _scalar(annual_reference_fields["GPP"])),
                    "NPP_model": _delta(
                        _scalar(annual_modelout.NPP_model),
                        _scalar(annual_reference_modelout.NPP_model),
                    ),
                    "AGB_model": _delta(
                        _scalar(annual_modelout.AGB_model),
                        _scalar(annual_reference_modelout.AGB_model),
                    ),
                    "BGB_model": _delta(
                        _scalar(annual_modelout.BGB_model),
                        _scalar(annual_reference_modelout.BGB_model),
                    ),
                    "LEAF_M": _delta(_scalar(annual_selected["LEAF_M"]), _scalar(annual_reference_fields["LEAF_M"])),
                },
            }

    summary = {
        "ready_for_requested_days": run.ready_for_requested_days,
        "requested_days": args.days,
        "closed_modelout_days": len(run.daily_modelout),
        "missing_components": list(run.missing_components),
        "reference_history": str(history_path),
        "landpoint_id": args.landpoint_id,
        "run_def": str(args.run_def),
        "reference_history_daily_truth": False,
        "mode": args.mode,
        "initial_state": args.initial_state,
        "compact_later_days": args.compact_later_days if args.mode == "lite" else "n/a",
        "single_pass_daily_fold": args.single_pass_daily_fold if args.mode == "lite" else "n/a",
        "use_static_jit_daily_carbon": args.use_static_jit_daily_carbon if args.mode == "lite" else "n/a",
        "prebuild_day_payloads": args.prebuild_day_payloads if args.mode == "lite" else "n/a",
        "compiled_sechiba_day": args.compiled_sechiba_day if args.mode == "lite" else "n/a",
        "yearly_or_sparse_history_record_diagnostic": yearly_record,
        "annual_history_mean_diagnostic": annual_history,
        "elapsed_seconds": elapsed,
        "days": comparisons,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if run.ready_for_requested_days and len(run.daily_modelout) == args.days else 1


if __name__ == "__main__":
    raise SystemExit(main())
