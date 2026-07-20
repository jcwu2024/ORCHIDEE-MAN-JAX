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
    driver_year_handoff_state_gaps,
    paper_1961_driver_multiday_modelout_lite_run,
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


def _array_summary(value) -> dict[str, object]:
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    arr = np.asarray(value, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "shape": list(arr.shape),
        "min": None if finite.size == 0 else float(finite.min()),
        "max": None if finite.size == 0 else float(finite.max()),
        "sum": float(np.nansum(arr)),
        "first_value": None if arr.size == 0 else float(arr.reshape(-1)[0]),
    }


def _summarize_run(run, *, fields: tuple[str, ...]) -> dict[str, object]:
    return {
        "year": int(run.year),
        "requested_days": int(run.requested_days),
        "ready_for_requested_days": bool(run.ready_for_requested_days),
        "stopped_day_index": run.stopped_day_index,
        "missing_components": list(run.missing_components),
        "last_tstep": None if run.last_day_end_state is None else int(run.last_day_end_state.tstep),
        "daily": [
            {
                "day_index": int(day.day_index),
                "start_tstep": int(day.start_tstep),
                "end_tstep": int(day.end_tstep),
                "fields": {
                    field: _array_summary(day.modelout_fields[field])
                    for field in fields
                    if field in day.modelout_fields
                },
            }
            for day in run.daily_modelout
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the local paper-case year handoff: carry a dynamic "
            "state from one forcing year into the next while rebasing the "
            "driver timestep counter."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument(
        "--run-def",
        type=Path,
        default=ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def",
    )
    parser.add_argument("--source-year", type=int, default=1961)
    parser.add_argument("--source-days", type=int, default=3)
    parser.add_argument("--restart-year", type=int, default=1962)
    parser.add_argument("--restart-days", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "restart_year_handoff_smoke.json",
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        default=None,
        help="Modelout field to summarize in the JSON output. May be repeated.",
    )
    args = parser.parse_args()

    if args.source_days < 1:
        raise ValueError("--source-days must be positive")
    if args.restart_days < 1:
        raise ValueError("--restart-days must be positive")
    fields = tuple(args.fields or ("GPP", "NPP", "LEAF_M"))

    started = time.perf_counter()
    source = paper_1961_driver_multiday_modelout_lite_run(
        args.config,
        year=args.source_year,
        ndays=args.source_days,
        used_run_def_path=args.run_def,
        root=ROOT,
        compact_later_days=True,
    )
    _block_until_ready(source)
    after_source = time.perf_counter()

    gaps = driver_year_handoff_state_gaps(source.last_day_end_state)
    restart = paper_1961_driver_restart_year_multiday_modelout_lite_run(
        args.config,
        previous_year_end_state=source.last_day_end_state,
        year=args.restart_year,
        ndays=args.restart_days,
        used_run_def_path=args.run_def,
    )
    _block_until_ready(restart)
    ended = time.perf_counter()

    payload = {
        "description": (
            "Local year-handoff smoke: source-year dynamic state is carried "
            "into restart-year day 1 with the driver counter rebased to the "
            "new forcing year."
        ),
        "config": str(args.config),
        "run_def": str(args.run_def),
        "timing_seconds": {
            "source_run": after_source - started,
            "restart_run": ended - after_source,
            "total": ended - started,
        },
        "handoff_gaps_after_source": [
            {
                "component": gap.component,
                "fields": list(gap.fields),
                "provenance": list(gap.provenance),
                "notes": list(gap.notes),
            }
            for gap in gaps
        ],
        "source": _summarize_run(source, fields=fields),
        "restart": _summarize_run(restart, fields=fields),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_ready": payload["source"]["ready_for_requested_days"],
                "handoff_gap_count": len(gaps),
                "restart_ready": payload["restart"]["ready_for_requested_days"],
                "restart_missing_components": payload["restart"]["missing_components"],
                "restart_days": len(payload["restart"]["daily"]),
                "restart_last_tstep": payload["restart"]["last_tstep"],
                "timing_seconds": payload["timing_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
