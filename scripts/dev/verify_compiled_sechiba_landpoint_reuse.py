from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration  # noqa: E402
from jax_orchidee.driver.orchestration import paper_1961_driver_multiday_modelout_lite_run  # noqa: E402
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference  # noqa: E402
from jax_orchidee.driver.run_def_materialization import (  # noqa: E402
    materialize_case_run_def_values,
    write_materialized_run_def,
)
from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402


configure_jax_compilation_cache(ROOT)

DEFAULT_CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
DEFAULT_BASE_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
DEFAULT_RUN_DEF_ROOT = ROOT / "outputs" / "paper_250919_materialized_run_defs" / "arg2_1.0"
DEFAULT_OUTPUT = ROOT / "outputs" / "performance" / "compiled_sechiba_landpoint_reuse.json"


def _materialized_run_def(landpoint_id: str) -> tuple[Path, Path]:
    reference = resolve_paper_landpoint_reference(ROOT, landpoint_id)
    if reference.output_dir is None or reference.run_def is None:
        raise FileNotFoundError(f"incomplete reference package for {landpoint_id}")
    if reference.iteration_id is None or reference.param_set is None:
        raise ValueError(f"missing iteration metadata for {landpoint_id}")
    destination = (
        DEFAULT_RUN_DEF_ROOT
        / landpoint_id
        / reference.iteration_id
        / reference.param_set
        / "used_run.def"
    )
    base = reference.used_run_def or DEFAULT_BASE_RUN_DEF
    values = materialize_case_run_def_values(
        base_used_run_def=base,
        case_run_def=reference.run_def,
        base_is_fortran_used_truth=reference.used_run_def is not None,
    )
    path = write_materialized_run_def(
        values,
        destination,
        header_lines=(
            "# Materialized runtime run.def for compiled landpoint reuse verification.",
            f"# Base defaults: {base}",
            f"# Landpoint overrides: {reference.run_def}",
        ),
    )
    return path, reference.output_dir


def _error_metrics(actual, expected) -> tuple[float, float]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError("compiled and strict nested state schemas differ")
        metrics = [_error_metrics(actual[name], expected[name]) for name in expected]
        return max((item[0] for item in metrics), default=0.0), max(
            (item[1] for item in metrics), default=0.0
        )
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError("compiled and strict nested state sequences differ")
        metrics = [_error_metrics(left, right) for left, right in zip(actual, expected, strict=True)]
        return max((item[0] for item in metrics), default=0.0), max(
            (item[1] for item in metrics), default=0.0
        )
    try:
        actual_array = np.asarray(actual, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError):
        if actual != expected:
            raise AssertionError(f"compiled value {actual!r} differs from strict value {expected!r}")
        return 0.0, 0.0
    if actual_array.shape != expected_array.shape:
        raise AssertionError("compiled and strict state array shapes differ")
    absolute = np.abs(actual_array - expected_array)
    absolute = np.where(np.isnan(actual_array) & np.isnan(expected_array), 0.0, absolute)
    denominator = np.maximum(np.abs(expected_array), 1.0)
    return float(np.max(absolute, initial=0.0)), float(np.max(absolute / denominator, initial=0.0))


def _compare_runs(compiled, strict) -> dict[str, object]:
    if not compiled.ready_for_requested_days or not strict.ready_for_requested_days:
        raise RuntimeError("strict or compiled run stopped before the requested day")
    max_abs = 0.0
    max_rel = 0.0
    max_field = None
    for compiled_day, strict_day in zip(compiled.daily_modelout, strict.daily_modelout, strict=True):
        if compiled_day.modelout_fields.keys() != strict_day.modelout_fields.keys():
            raise AssertionError("compiled and strict modelout schemas differ")
        for name, expected in strict_day.modelout_fields.items():
            absolute, relative = _error_metrics(compiled_day.modelout_fields[name], expected)
            if absolute > max_abs:
                max_abs = absolute
                max_field = f"day{compiled_day.day_index}.modelout.{name}"
            max_rel = max(max_rel, relative)

    compiled_state = compiled.last_day_end_state
    strict_state = strict.last_day_end_state
    if compiled_state.fields_by_component.keys() != strict_state.fields_by_component.keys():
        raise AssertionError("compiled and strict final-state component schemas differ")
    for component, expected_fields in strict_state.fields_by_component.items():
        actual_fields = compiled_state.fields_by_component[component]
        if actual_fields.keys() != expected_fields.keys():
            raise AssertionError(f"compiled and strict {component} state schemas differ")
        for name, expected in expected_fields.items():
            absolute, relative = _error_metrics(actual_fields[name], expected)
            if absolute > max_abs:
                max_abs = absolute
                max_field = f"final_state.{component}.{name}"
            max_rel = max(max_rel, relative)
    return {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "max_abs_field": max_field,
        "tolerance_pass": bool(max_abs <= 1.0e-8 or max_rel <= 1.0e-10),
    }


def _run(landpoint_id: str, *, days: int, compiled: bool):
    run_def, reference_run_dir = _materialized_run_def(landpoint_id)
    start = time.perf_counter()
    result = paper_1961_driver_multiday_modelout_lite_run(
        DEFAULT_CONFIG,
        year=1961,
        ndays=days,
        used_run_def_path=run_def,
        reference_run_dir=reference_run_dir,
        root=ROOT,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=compiled,
    )
    return result, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify one compiled SECHIBA day executable across landpoints.")
    parser.add_argument("--landpoint-id", action="append", default=None)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    landpoint_ids = tuple(args.landpoint_id or ("001.0-071.0", "003.0-077.0"))
    if len(landpoint_ids) < 2:
        raise ValueError("at least two landpoints are required")

    orchestration._COMPILED_SECHIBA_SCAN_CACHE.clear()
    records = []
    callable_identity = None
    for landpoint_id in landpoint_ids:
        executable_count_before = (
            int(next(iter(orchestration._COMPILED_SECHIBA_SCAN_CACHE.values()))._cache_size())
            if orchestration._COMPILED_SECHIBA_SCAN_CACHE
            else 0
        )
        compiled, compiled_seconds = _run(landpoint_id, days=args.days, compiled=True)
        if len(orchestration._COMPILED_SECHIBA_SCAN_CACHE) != 1:
            raise AssertionError("landpoints did not share one structural scan cache entry")
        compiled_callable = next(iter(orchestration._COMPILED_SECHIBA_SCAN_CACHE.values()))
        current_identity = id(compiled_callable)
        if callable_identity is None:
            callable_identity = current_identity
        elif current_identity != callable_identity:
            raise AssertionError("landpoints selected different compiled scan callables")
        executable_count = int(compiled_callable._cache_size())

        strict, strict_seconds = _run(landpoint_id, days=args.days, compiled=False)
        comparison = _compare_runs(compiled, strict)
        _, compiled_repeat_seconds = _run(landpoint_id, days=args.days, compiled=True)
        executable_count_after_repeat = int(compiled_callable._cache_size())
        records.append(
            {
                "landpoint_id": landpoint_id,
                "compiled_seconds": compiled_seconds,
                "compiled_repeat_seconds": compiled_repeat_seconds,
                "strict_seconds": strict_seconds,
                "scan_callable_identity": current_identity,
                "scan_executable_cache_size_before": executable_count_before,
                "scan_executable_cache_size": executable_count,
                "scan_executable_cache_size_after_repeat": executable_count_after_repeat,
                "new_scan_executables": executable_count - executable_count_before,
                "strict_parity": comparison,
            }
        )

    payload = {
        "days": args.days,
        "landpoints": records,
        "structural_scan_cache_entries": len(orchestration._COMPILED_SECHIBA_SCAN_CACHE),
        "same_scan_callable": len({record["scan_callable_identity"] for record in records}) == 1,
        "first_landpoint_scan_executable_variants": records[0]["scan_executable_cache_size"],
        "later_landpoints_added_no_executables": all(
            record["new_scan_executables"] == 0 for record in records[1:]
        ),
        "repeats_added_no_executables": all(
            record["scan_executable_cache_size_after_repeat"] == record["scan_executable_cache_size"]
            for record in records
        ),
        "all_strict_parity_pass": all(record["strict_parity"]["tolerance_pass"] for record in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    accepted = (
        payload["same_scan_callable"]
        and payload["later_landpoints_added_no_executables"]
        and payload["repeats_added_no_executables"]
        and payload["all_strict_parity_pass"]
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
