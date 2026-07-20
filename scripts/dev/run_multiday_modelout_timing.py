from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_multiday_modelout_lite_run,
    paper_1961_driver_multiday_modelout_run,
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


def _as_float(value) -> float:
    try:
        import numpy as np

        return float(np.asarray(value).reshape(-1)[13])
    except Exception:
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local paper-case multiday modelout chain with minimal timing overhead."
    )
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--year", type=int, default=1961)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument(
        "--run-def",
        type=Path,
        default=ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def",
    )
    parser.add_argument("--profile-out", type=Path, default=None)
    parser.add_argument("--profile-top", type=int, default=0)
    parser.add_argument("--profile-run", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=None,
        help="Days to run for each warmup pass; defaults to --days. Use a short value to compile kernels before long timing runs.",
    )
    parser.add_argument("--mode", choices=("scaffold", "lite"), default="scaffold")
    parser.add_argument("--module-jit", choices=("on", "off"), default="on")
    parser.add_argument("--diffuco-local-jit", choices=("on", "off"), default="on")
    parser.add_argument("--single-pass-daily-fold", choices=("on", "off"), default="off")
    parser.add_argument("--use-compiled-accumulator", choices=("on", "off"), default="off")
    parser.add_argument("--use-numpy-accumulator", choices=("on", "off"), default="off")
    parser.add_argument("--compact-later-days", choices=("on", "off"), default="off")
    parser.add_argument("--use-fast-state-loop", choices=("on", "off"), default="off")
    parser.add_argument("--use-static-jit-daily-carbon", choices=("on", "off"), default="off")
    parser.add_argument("--prebuild-day-payloads", choices=("on", "off"), default="on")
    parser.add_argument("--compiled-sechiba-day", choices=("on", "off"), default="off")
    parser.add_argument(
        "--compiled-day-block-size",
        type=int,
        default=0,
        help="Compile complete later-day transitions in fixed blocks; zero disables it.",
    )
    parser.add_argument(
        "--show-days",
        type=int,
        default=10,
        help="Number of leading daily records to include in JSON output; use -1 for all days.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.warmup_days is not None and args.warmup_days < 1:
        raise ValueError("--warmup-days must be positive when supplied")

    def _run_model(days: int | None = None):
        ndays = args.days if days is None else int(days)
        runner = (
            paper_1961_driver_multiday_modelout_lite_run
            if args.mode == "lite"
            else paper_1961_driver_multiday_modelout_run
        )
        return runner(
            args.config,
            year=args.year,
            ndays=ndays,
            used_run_def_path=args.run_def,
            root=ROOT,
            module_jit=args.module_jit == "on",
            diffuco_local_jit=args.diffuco_local_jit == "on",
            **(
                {
                    "single_pass_daily_fold": args.single_pass_daily_fold == "on",
                    "use_compiled_accumulator": args.use_compiled_accumulator == "on",
                    "use_numpy_accumulator": args.use_numpy_accumulator == "on",
                    "compact_later_days": args.compact_later_days == "on",
                    "use_fast_state_loop": args.use_fast_state_loop == "on",
                    "use_static_jit_daily_carbon": args.use_static_jit_daily_carbon == "on",
                    "prebuild_day_payloads": args.prebuild_day_payloads == "on",
                    "use_compiled_sechiba_day": args.compiled_sechiba_day == "on",
                    "compiled_day_block_size": args.compiled_day_block_size,
                }
                if args.mode == "lite"
                else {}
            ),
        )

    run = None
    warmup_runs = []
    warmup_days = args.days if args.warmup_days is None else args.warmup_days
    for _ in range(args.warmup):
        start = time.perf_counter()
        warmed = _run_model(days=warmup_days)
        _block_until_ready(warmed)
        warmup_runs.append(time.perf_counter() - start)

    elapsed_runs = []
    profiler = cProfile.Profile() if args.profile_out is not None or args.profile_top else None
    for run_index in range(1, args.repeat + 1):
        start = time.perf_counter()
        profile_this_run = profiler is not None and (args.profile_run in (0, run_index))
        if profile_this_run:
            run = profiler.runcall(_run_model)
        else:
            run = _run_model()
        _block_until_ready(run)
        elapsed_runs.append(time.perf_counter() - start)
    elapsed = elapsed_runs[-1]
    if profiler is not None:
        if args.profile_out is not None:
            args.profile_out.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(args.profile_out)
        if args.profile_top:
            stats = pstats.Stats(profiler, stream=sys.stderr)
            stats.strip_dirs().sort_stats("cumtime").print_stats(args.profile_top)

    if args.show_days < -1:
        raise ValueError("--show-days must be -1 or non-negative")
    visible_days = run.daily_modelout if args.show_days < 0 else run.daily_modelout[: args.show_days]
    days = []
    for daily in visible_days:
        days.append(
            {
                "day": daily.day_index,
                "start_tstep": daily.start_tstep,
                "end_tstep": daily.end_tstep,
                "GPP_pft14": _as_float(daily.modelout_fields["GPP"]),
                "NPP_model_pft14": _as_float(daily.modelout.NPP_model),
                "AGB_model_pft14": _as_float(daily.modelout.AGB_model),
                "LEAF_M_pft14": _as_float(daily.modelout_fields["LEAF_M"]),
            }
        )

    summary = {
        "mode": args.mode,
        "module_jit": args.module_jit,
        "diffuco_local_jit": args.diffuco_local_jit,
        "single_pass_daily_fold": args.single_pass_daily_fold if args.mode == "lite" else "n/a",
        "use_compiled_accumulator": args.use_compiled_accumulator if args.mode == "lite" else "n/a",
        "use_numpy_accumulator": args.use_numpy_accumulator if args.mode == "lite" else "n/a",
        "compact_later_days": args.compact_later_days if args.mode == "lite" else "n/a",
        "use_fast_state_loop": args.use_fast_state_loop if args.mode == "lite" else "n/a",
        "use_static_jit_daily_carbon": args.use_static_jit_daily_carbon if args.mode == "lite" else "n/a",
        "prebuild_day_payloads": args.prebuild_day_payloads if args.mode == "lite" else "n/a",
        "compiled_sechiba_day": args.compiled_sechiba_day if args.mode == "lite" else "n/a",
        "compiled_day_block_size": args.compiled_day_block_size if args.mode == "lite" else "n/a",
        "ready_for_requested_days": run.ready_for_requested_days,
        "requested_days": args.days,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "warmup_days": warmup_days if args.warmup else 0,
        "warmup_runs_seconds": warmup_runs,
        "closed_modelout_days": len(run.daily_modelout),
        "stopped_day_index": run.stopped_day_index if args.mode == "lite" else run.scaffold.stopped_day_index,
        "missing_components": list(run.missing_components),
        "elapsed_seconds": elapsed,
        "elapsed_runs_seconds": elapsed_runs,
        "seconds_per_closed_day": elapsed / max(len(run.daily_modelout), 1),
        "shown_days": len(days),
        "omitted_days": max(len(run.daily_modelout) - len(days), 0),
        "days": days,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if run.ready_for_requested_days and len(run.daily_modelout) == args.days else 1


if __name__ == "__main__":
    raise SystemExit(main())
