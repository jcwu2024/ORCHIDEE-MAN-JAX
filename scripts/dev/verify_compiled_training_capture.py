from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration as teacher  # noqa: E402
from research.daily_coarse_graining.daily_markov_contract import (  # noqa: E402
    extract_diagnostics,
    extract_fast_day_target,
    extract_state,
)
from research.daily_coarse_graining.replay_ceiling import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_RUN_DEF,
    _compare_trees,
    _load_state_cache,
)
from research.daily_coarse_graining.supervised_learnability_pilot import (  # noqa: E402
    _capture_days,
    _capture_days_compiled_blocks,
    _iter_capture_days_compiled_blocks,
    _teacher_target_vector,
    build_boundary_vector_spec,
)
from research.daily_coarse_graining.teacher_shards import (  # noqa: E402
    _compact_contract_factory,
    build_shard_arrays_from_compact_blocks,
)


def _target_metrics(expected, actual, *, atol: float, rtol: float):
    finite = np.isfinite(expected) & np.isfinite(actual)
    difference = np.abs(actual[finite] - expected[finite])
    denominator = np.maximum(
        np.abs(expected[finite]), np.finfo(np.float64).tiny
    )
    with np.errstate(over="ignore", invalid="ignore"):
        relative = difference / denominator
    return {
        "passed": bool(
            np.allclose(expected, actual, atol=atol, rtol=rtol, equal_nan=True)
        ),
        "max_absolute_error": float(np.max(difference)) if difference.size else 0.0,
        "max_relative_error": float(np.max(relative)) if relative.size else 0.0,
        "nonfinite_pattern_equal": bool(
            np.array_equal(np.isnan(expected), np.isnan(actual))
            and np.array_equal(np.isposinf(expected), np.isposinf(actual))
            and np.array_equal(np.isneginf(expected), np.isneginf(actual))
        ),
    }


def run(args):
    context = teacher.prepare_paper_1961_driver_context(
        args.config,
        used_run_def_path=args.run_def,
        reference_run_dir=args.reference_run_dir,
    )
    if args.cold_start:
        bootstrap = teacher.paper_1961_driver_cold_start_day_scaffold(
            args.config,
            year=args.year,
            used_run_def_path=args.run_def,
            prepared_context=context,
        )
        if not bootstrap.ready_for_first_day_end_state:
            raise RuntimeError(f"cold-start Day 1 is incomplete: {bootstrap.state_gaps}")
        initial = bootstrap.first_day_end_state
        start_day = 2
    else:
        cache = _load_state_cache(args.state_cache)
        initial = teacher.rebase_driver_state_for_year_start(cache["state"])
        start_day = 1
    started = time.perf_counter()
    daily = _capture_days(
        config_path=args.config,
        context=context,
        previous_state=initial,
        year=args.year,
        start_day=start_day,
        days=args.days,
    )
    daily_seconds = time.perf_counter() - started
    daily_states, daily_forcings, daily_records, daily_final = daily
    hot_capture_seconds = []
    hot_capture_parity = []
    if args.capture_mode == "compact":
        def compact_capture():
            started = time.perf_counter()
            blocks = _iter_capture_days_compiled_blocks(
                config_path=args.config,
                context=context,
                previous_state=initial,
                year=args.year,
                start_day=start_day,
                days=args.days,
                block_size=args.block_size,
                compact_contract_factory=_compact_contract_factory(
                    context,
                    year=args.year,
                    first_day_index=start_day,
                ),
            )
            result = build_shard_arrays_from_compact_blocks(blocks, context)
            return (*result, time.perf_counter() - started)

        arrays, contract, compiled_final, compiled_seconds = compact_capture()
        for _repeat in range(args.hot_repeats):
            hot_arrays, hot_contract, hot_final, hot_seconds = compact_capture()
            hot_capture_seconds.append(hot_seconds)
            hot_capture_parity.append(
                bool(
                    hot_contract.sha256 == contract.sha256
                    and all(
                        np.allclose(
                            arrays[name],
                            hot_arrays[name],
                            atol=args.atol,
                            rtol=args.rtol,
                            equal_nan=True,
                        )
                        for name in arrays
                        if arrays[name].dtype.kind == "f"
                    )
                    and all(
                        np.array_equal(arrays[name], hot_arrays[name])
                        for name in arrays
                        if arrays[name].dtype.kind != "f"
                    )
                    and _compare_trees(
                        compiled_final.fields_by_component,
                        hot_final.fields_by_component,
                        atol=args.atol,
                        rtol=args.rtol,
                    )["passed"]
                )
            )
        comparisons = []
        for offset, (daily_state, daily_record) in enumerate(
            zip(daily_states, daily_records, strict=True)
        ):
            expected_state, expected_discrete = extract_state(
                daily_state,
                contract,
                allow_year_start_missing=offset == 0,
            )
            target = extract_fast_day_target(
                daily_record, contract.fast_day_target_leaves
            )
            diagnostics = extract_diagnostics(daily_record, contract)
            discrete = {
                name.removeprefix("state_discrete__"): value[offset]
                for name, value in arrays.items()
                if name.startswith("state_discrete__")
            }
            comparisons.append(
                {
                    "day_index": int(arrays["day_index"][offset]),
                    "day_start_state": _target_metrics(
                        expected_state,
                        arrays["state_trajectory"][offset],
                        atol=args.atol,
                        rtol=args.rtol,
                    ),
                    "discrete_state_exact": all(
                        np.array_equal(expected_discrete[name], discrete[name])
                        for name in expected_discrete
                    ),
                    "target": _target_metrics(
                        target,
                        arrays["fast_day_target"][offset],
                        atol=args.atol,
                        rtol=args.rtol,
                    ),
                    "diagnostics": _target_metrics(
                        diagnostics,
                        arrays["diagnostics"][offset],
                        atol=args.atol,
                        rtol=args.rtol,
                    ),
                }
            )
        spec_size = contract.fast_day_target_width
    else:
        started = time.perf_counter()
        compiled = _capture_days_compiled_blocks(
            config_path=args.config,
            context=context,
            previous_state=initial,
            year=args.year,
            start_day=start_day,
            days=args.days,
            block_size=args.block_size,
        )
        compiled_seconds = time.perf_counter() - started
        compiled_states, compiled_forcings, compiled_records, compiled_final = compiled
        spec = build_boundary_vector_spec(daily_records[0])
        comparisons = []
        for day_index, values in enumerate(
            zip(
                daily_states,
                compiled_states,
                daily_forcings,
                compiled_forcings,
                daily_records,
                compiled_records,
                strict=True,
            ),
            start=1,
        ):
            daily_state, compiled_state, daily_forcing, compiled_forcing, daily_record, compiled_record = values
            expected = _teacher_target_vector(daily_record, spec)
            actual = _teacher_target_vector(compiled_record, spec)
            comparisons.append(
                {
                    "day_index": day_index,
                    "day_start_state": _compare_trees(
                        daily_state.fields_by_component,
                        compiled_state.fields_by_component,
                        atol=args.atol,
                        rtol=args.rtol,
                    ),
                    "forcing": _compare_trees(
                        daily_forcing,
                        compiled_forcing,
                        atol=0.0,
                        rtol=0.0,
                    ),
                    "target": _target_metrics(
                        expected, actual, atol=args.atol, rtol=args.rtol
                    ),
                }
            )
        spec_size = spec.total_size
    final_state = _compare_trees(
        daily_final.fields_by_component,
        compiled_final.fields_by_component,
        atol=args.atol,
        rtol=args.rtol,
    )
    passed = final_state["passed"] and all(hot_capture_parity) and all(
        item["day_start_state"]["passed"]
        and item.get("discrete_state_exact", True)
        and item.get("forcing", {"passed": True})["passed"]
        and item["target"]["passed"]
        and item.get("diagnostics", {"passed": True})["passed"]
        for item in comparisons
    )
    return {
        "schema_version": "compiled_training_capture_parity_v1",
        "passed": passed,
        "year": args.year,
        "initialization_mode": "cold_start_day1" if args.cold_start else "year_start_checkpoint",
        "transition_start_day": start_day,
        "days": args.days,
        "block_size": args.block_size,
        "capture_mode": args.capture_mode,
        "target_elements": spec_size,
        "timing_seconds": {
            "daily_capture": daily_seconds,
            "compiled_capture": compiled_seconds,
            "compact_hot_capture": hot_capture_seconds,
        },
        "hot_capture_parity": hot_capture_parity,
        "comparisons": comparisons,
        "final_state": final_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-def", type=Path, default=DEFAULT_RUN_DEF)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=1962)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument(
        "--capture-mode", choices=("compact", "legacy"), default="compact"
    )
    parser.add_argument("--hot-repeats", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cold_start == (args.state_cache is not None):
        parser.error("select exactly one of --cold-start or --state-cache")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "target_elements": report["target_elements"],
                "timing_seconds": report["timing_seconds"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
