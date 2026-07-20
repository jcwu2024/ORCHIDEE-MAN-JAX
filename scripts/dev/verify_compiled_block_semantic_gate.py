from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_multiday_modelout_lite_run,
    paper_1961_driver_multiday_modelout_lite_run,
    paper_1961_driver_restart_year_multiday_modelout_lite_run,
)
from jax_orchidee.driver.reference_layout import (  # noqa: E402
    resolve_paper_landpoint_reference,
)
from scripts.dev.audit_compiled_fast_path_acceptance import (  # noqa: E402
    ComparisonAccumulator,
    _materialized_run_def,
)
from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402


configure_jax_compilation_cache(ROOT)
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def _block_until_ready(value) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    jax.effects_barrier()


def _compare(actual, expected, *, days: int, atol: float, rtol: float) -> dict[str, object]:
    comparison = ComparisonAccumulator(atol=atol, rtol=rtol)
    comparison.compare("ready", actual.ready_for_requested_days, expected.ready_for_requested_days)
    comparison.compare("missing", actual.missing_components, expected.missing_components)
    for index, (left, right) in enumerate(
        zip(actual.daily_modelout, expected.daily_modelout, strict=False), start=1
    ):
        comparison.compare(f"day_{index}.modelout_fields", left.modelout_fields, right.modelout_fields)
        comparison.compare(f"day_{index}.modelout", left.modelout, right.modelout)
    if len(actual.daily_modelout) != int(days) or len(expected.daily_modelout) != int(days):
        comparison._fail(
            "daily_modelout",
            "day_count_mismatch",
            actual=len(actual.daily_modelout),
            expected=len(expected.daily_modelout),
            requested=int(days),
        )
    comparison.compare("final_state.tstep", actual.last_day_end_state.tstep, expected.last_day_end_state.tstep)
    comparison.compare(
        "final_state.fields_by_component",
        actual.last_day_end_state.fields_by_component,
        expected.last_day_end_state.fields_by_component,
    )
    return comparison.report()


def _cold(reference, run_def: Path, *, days: int, block_size: int):
    common = dict(
        year=1961,
        ndays=days,
        used_run_def_path=run_def,
        reference_run_dir=reference.output_dir,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    baseline = paper_1961_driver_cold_start_multiday_modelout_lite_run(
        CONFIG,
        **{**common, "use_compiled_sechiba_day": False},
    )
    candidate = paper_1961_driver_cold_start_multiday_modelout_lite_run(
        CONFIG,
        **common,
        compiled_day_block_size=block_size,
    )
    return candidate, baseline


def _annual(reference, run_def: Path, *, days: int, block_size: int):
    common = dict(
        year=1961,
        ndays=days,
        used_run_def_path=run_def,
        reference_run_dir=reference.output_dir,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    baseline = paper_1961_driver_cold_start_multiday_modelout_lite_run(
        CONFIG,
        **common,
        compiled_day_block_size=0,
    )
    candidate = paper_1961_driver_cold_start_multiday_modelout_lite_run(
        CONFIG,
        **common,
        compiled_day_block_size=block_size,
    )
    return candidate, baseline


def _restart(reference, run_def: Path, *, days: int, block_size: int):
    source = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=14,
        used_run_def_path=run_def,
        reference_run_dir=reference.output_dir,
        root=ROOT,
        compact_later_days=True,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    _block_until_ready(source)
    common = dict(
        previous_year_end_state=source.last_day_end_state,
        year=1962,
        ndays=days,
        used_run_def_path=run_def,
        reference_run_dir=reference.output_dir,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    baseline = paper_1961_driver_restart_year_multiday_modelout_lite_run(
        CONFIG,
        **common,
        compiled_day_block_size=0,
    )
    candidate = paper_1961_driver_restart_year_multiday_modelout_lite_run(
        CONFIG,
        **common,
        compiled_day_block_size=block_size,
    )
    return candidate, baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify one compiled-block semantic gate in an isolated process.")
    parser.add_argument("--scenario", choices=("cold", "annual", "restart"), required=True)
    parser.add_argument("--landpoint-id", default="001.0-071.0")
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.days < 1 or args.block_size < 2:
        raise ValueError("days must be positive and block-size must be at least two")

    reference = resolve_paper_landpoint_reference(ROOT, args.landpoint_id)
    if reference.output_dir is None:
        raise FileNotFoundError(f"reference output missing for {args.landpoint_id}")
    run_def = _materialized_run_def(reference)
    started = time.perf_counter()
    runner = {"cold": _cold, "annual": _annual, "restart": _restart}[args.scenario]
    candidate, baseline = runner(reference, run_def, days=args.days, block_size=args.block_size)
    _block_until_ready(candidate)
    _block_until_ready(baseline)
    comparison = _compare(candidate, baseline, days=args.days, atol=args.atol, rtol=args.rtol)
    result = {
        "status": "passed" if comparison["passed"] else "failed",
        "scenario": args.scenario,
        "landpoint_id": args.landpoint_id,
        "days": args.days,
        "block_size": args.block_size,
        "baseline": (
            "strict_source_order" if args.scenario == "cold" else "compiled_daily_no_block"
        ),
        "candidate": "compiled_complete_day_block",
        "comparison": comparison,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "scenario", "days", "elapsed_seconds")}, indent=2))
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
