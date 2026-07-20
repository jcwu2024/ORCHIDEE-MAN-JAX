from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_multiday_modelout_lite_run,
    paper_1961_driver_restart_year_multiday_modelout_lite_run,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache an in-memory paper-case driver year-end state for targeted "
            "restart-year diagnostics. This writes only a local Python pickle "
            "under outputs/ and does not alter model semantics."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs" / "reference_mode" / "used_run.def")
    parser.add_argument("--start-year", type=int, default=1961)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--days-per-year", type=int, default=365)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "cache" / "paper_driver_year_end_state.pkl",
    )
    parser.add_argument("--single-pass-daily-fold", choices=("on", "off"), default="off")
    parser.add_argument("--prebuild-day-payloads", choices=("on", "off"), default="on")
    parser.add_argument("--module-jit", choices=("on", "off"), default="on")
    parser.add_argument("--diffuco-local-jit", choices=("on", "off"), default="on")
    parser.add_argument(
        "--previous-state-cache",
        type=Path,
        default=None,
        help="Pickle created by this script for start_year - 1; starts with a restart-year handoff.",
    )
    args = parser.parse_args()

    if args.end_year < args.start_year:
        parser.error("--end-year must be >= --start-year")
    if args.days_per_year < 1:
        parser.error("--days-per-year must be positive")

    started = time.perf_counter()
    previous_state = None
    previous_state_cache_metadata = None
    if args.previous_state_cache is not None:
        with args.previous_state_cache.open("rb") as handle:
            cache_payload = pickle.load(handle)
        if not isinstance(cache_payload, dict) or "state" not in cache_payload:
            raise ValueError(f"{args.previous_state_cache} is not a driver year-end state cache")
        if int(cache_payload.get("end_year", -9999)) != int(args.start_year) - 1:
            raise ValueError("--previous-state-cache must end at start_year - 1")
        previous_state = cache_payload["state"]
        previous_state_cache_metadata = {
            key: cache_payload.get(key)
            for key in ("description", "config", "run_def", "start_year", "end_year", "days_per_year", "created_elapsed_seconds")
        }
    year_summaries: list[dict[str, object]] = []
    for year in range(int(args.start_year), int(args.end_year) + 1):
        year_started = time.perf_counter()
        if year == int(args.start_year) and previous_state is not None:
            run = paper_1961_driver_restart_year_multiday_modelout_lite_run(
                args.config,
                previous_year_end_state=previous_state,
                year=year,
                ndays=int(args.days_per_year),
                used_run_def_path=args.run_def,
                module_jit=args.module_jit == "on",
                diffuco_local_jit=args.diffuco_local_jit == "on",
                single_pass_daily_fold=args.single_pass_daily_fold == "on",
                prebuild_day_payloads=args.prebuild_day_payloads == "on",
            )
        elif year == int(args.start_year):
            run = paper_1961_driver_cold_start_multiday_modelout_lite_run(
                args.config,
                year=year,
                ndays=int(args.days_per_year),
                used_run_def_path=args.run_def,
                module_jit=args.module_jit == "on",
                diffuco_local_jit=args.diffuco_local_jit == "on",
                single_pass_daily_fold=args.single_pass_daily_fold == "on",
                prebuild_day_payloads=args.prebuild_day_payloads == "on",
            )
        else:
            run = paper_1961_driver_restart_year_multiday_modelout_lite_run(
                args.config,
                previous_year_end_state=previous_state,
                year=year,
                ndays=int(args.days_per_year),
                used_run_def_path=args.run_def,
                module_jit=args.module_jit == "on",
                diffuco_local_jit=args.diffuco_local_jit == "on",
                single_pass_daily_fold=args.single_pass_daily_fold == "on",
                prebuild_day_payloads=args.prebuild_day_payloads == "on",
            )
        _block_until_ready(run)
        previous_state = run.last_day_end_state
        year_summaries.append(
            {
                "year": year,
                "ready_for_requested_days": bool(run.ready_for_requested_days),
                "requested_days": int(run.requested_days),
                "completed_days": len(run.daily_modelout),
                "missing_components": list(run.missing_components),
                "last_tstep": None if previous_state is None else int(previous_state.tstep),
                "elapsed_seconds": time.perf_counter() - year_started,
            }
        )
        if previous_state is None or not run.ready_for_requested_days:
            break

    ok = previous_state is not None and year_summaries[-1]["year"] == int(args.end_year) and all(
        bool(item["ready_for_requested_days"]) for item in year_summaries
    )
    payload = {
        "description": "Cached paper-case JAX driver year-end state for local diagnostics.",
        "config": str(args.config),
        "run_def": str(args.run_def),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "days_per_year": int(args.days_per_year),
        "previous_state_cache": None if args.previous_state_cache is None else str(args.previous_state_cache),
        "previous_state_cache_metadata": previous_state_cache_metadata,
        "single_pass_daily_fold": args.single_pass_daily_fold,
        "prebuild_day_payloads": args.prebuild_day_payloads,
        "created_elapsed_seconds": time.perf_counter() - started,
        "year_summaries": year_summaries,
        "state": previous_state,
    }
    if ok:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        {
            "ok": ok,
            "output": str(args.output) if ok else None,
            "elapsed_seconds": payload["created_elapsed_seconds"],
            "year_summaries": year_summaries,
        }
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
