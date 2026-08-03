from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration  # noqa: E402
from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_multiday_modelout_lite_run,
    paper_1961_driver_restart_year_multiday_modelout_lite_run,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.driver.reference_layout import (  # noqa: E402
    PaperLandpointReference,
    resolve_paper_landpoint_reference,
)
from jax_orchidee.driver.restart_bundle import (  # noqa: E402
    PaperRestartBundlePhysicalState,
    paper_restart_bundle_state_from_day_end_packet,
    read_restart_physical_state,
    write_paper_restart_start_bundle,
)
from jax_orchidee.driver.restart_file_io import read_dim2_driver_restart  # noqa: E402
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state  # noqa: E402
from jax_orchidee.driver.run_def_materialization import (  # noqa: E402
    materialize_case_run_def_values,
    write_materialized_run_def,
)
from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402
from jax_orchidee.stomate.modelout import (  # noqa: E402
    MODEL_OUTPUT_FIELD_NAMES,
    annual_history_mean_fields_from_daily_modelout,
    compute_modelout_from_fields,
)
from scripts.dev.audit_stage5_lifecycle import _packet_from_restart_readers  # noqa: E402


configure_jax_compilation_cache(ROOT)

CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
DEFAULT_BASE_RUN_DEF = ROOT / "outputs/server_1961_trace_full_20260623/run/used_run.def"
RUN_DEF_ROOT = ROOT / "outputs/paper_250919_materialized_run_defs/arg2_1.0"
OUTPUT = ROOT / "outputs/performance/compiled_fast_path_acceptance/comparison.json"
LANDPOINTS = ("001.0-071.0", "069.0-119.0", "319.0-057.0")


class ComparisonAccumulator:
    """Aggregate recursive strict/compiled comparisons without storing arrays."""

    def __init__(self, *, atol: float, rtol: float) -> None:
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.passed = True
        self.float_leaves = 0
        self.discrete_leaves = 0
        self.element_count = 0
        self.max_abs_error = 0.0
        self.max_rel_error = 0.0
        self.max_ulp_error = 0
        self.max_abs_path: str | None = None
        self.max_rel_path: str | None = None
        self.max_ulp_path: str | None = None
        self.failures: list[dict[str, object]] = []

    def _fail(self, path: str, reason: str, **details: object) -> None:
        self.passed = False
        if len(self.failures) < 100:
            self.failures.append({"path": path, "reason": reason, **details})

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, Mapping):
            return dict(value)
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: getattr(value, field.name) for field in fields(value)}
        if hasattr(value, "_asdict"):
            return dict(value._asdict())
        return None

    @staticmethod
    def _ulp_distance(left: np.ndarray, right: np.ndarray) -> int:
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        finite = np.isfinite(left) & np.isfinite(right)
        if not np.any(finite):
            return 0
        left_bits = left[finite].view(np.uint64)
        right_bits = right[finite].view(np.uint64)
        sign = np.uint64(1 << 63)
        left_ordered = np.where(left_bits & sign, ~left_bits, left_bits | sign)
        right_ordered = np.where(right_bits & sign, ~right_bits, right_bits | sign)
        distance = np.where(
            left_ordered >= right_ordered,
            left_ordered - right_ordered,
            right_ordered - left_ordered,
        )
        return int(distance.max(initial=np.uint64(0)))

    def compare(self, path: str, actual: Any, expected: Any) -> None:
        actual_mapping = self._mapping(actual)
        expected_mapping = self._mapping(expected)
        if actual_mapping is not None or expected_mapping is not None:
            if actual_mapping is None or expected_mapping is None:
                self._fail(path, "container_type_mismatch")
                return
            if actual_mapping.keys() != expected_mapping.keys():
                self._fail(
                    path,
                    "mapping_schema_mismatch",
                    missing_actual=sorted(expected_mapping.keys() - actual_mapping.keys()),
                    missing_expected=sorted(actual_mapping.keys() - expected_mapping.keys()),
                )
                return
            for name in expected_mapping:
                self.compare(f"{path}.{name}", actual_mapping[name], expected_mapping[name])
            return
        if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
            if type(actual) is not type(expected) or len(actual) != len(expected):
                self._fail(path, "sequence_schema_mismatch")
                return
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                self.compare(f"{path}[{index}]", left, right)
            return

        try:
            left = np.asarray(actual)
            right = np.asarray(expected)
        except (TypeError, ValueError):
            self.discrete_leaves += 1
            if actual != expected:
                self._fail(path, "non_array_value_mismatch", actual=repr(actual), expected=repr(expected))
            return
        if left.shape != right.shape:
            self._fail(path, "shape_mismatch", actual_shape=list(left.shape), expected_shape=list(right.shape))
            return
        if left.dtype.kind not in "f" or right.dtype.kind not in "f":
            self.discrete_leaves += 1
            self.element_count += int(left.size)
            try:
                equal = np.array_equal(left, right, equal_nan=True)
            except TypeError:
                equal = np.array_equal(left, right)
            if not equal:
                self._fail(path, "discrete_not_exact", actual_dtype=str(left.dtype), expected_dtype=str(right.dtype))
            return

        self.float_leaves += 1
        self.element_count += int(left.size)
        left64 = left.astype(np.float64, copy=False)
        right64 = right.astype(np.float64, copy=False)
        same_nonfinite = (np.isnan(left64) & np.isnan(right64)) | (left64 == right64)
        if not np.array_equal(np.isfinite(left64), np.isfinite(right64)) or not np.all(
            np.isfinite(left64) | same_nonfinite
        ):
            self._fail(path, "nonfinite_mismatch")
            return
        absolute = np.where(same_nonfinite, 0.0, np.abs(left64 - right64))
        relative = absolute / np.maximum(np.abs(right64), np.finfo(np.float64).tiny)
        max_abs = float(absolute.max(initial=0.0))
        max_rel = float(relative.max(initial=0.0))
        max_ulp = self._ulp_distance(left64, right64)
        if max_abs > self.max_abs_error:
            self.max_abs_error = max_abs
            self.max_abs_path = path
        if max_rel > self.max_rel_error:
            self.max_rel_error = max_rel
            self.max_rel_path = path
        if max_ulp > self.max_ulp_error:
            self.max_ulp_error = max_ulp
            self.max_ulp_path = path
        close = np.allclose(left64, right64, atol=self.atol, rtol=self.rtol, equal_nan=True)
        if not close:
            self._fail(path, "float_tolerance_failure", max_abs=max_abs, max_rel=max_rel, max_ulp=max_ulp)

    def report(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "atol": self.atol,
            "rtol": self.rtol,
            "float_leaves": self.float_leaves,
            "discrete_leaves": self.discrete_leaves,
            "element_count": self.element_count,
            "max_abs_error": self.max_abs_error,
            "max_abs_path": self.max_abs_path,
            "max_rel_error": self.max_rel_error,
            "max_rel_path": self.max_rel_path,
            "max_ulp_error": self.max_ulp_error,
            "max_ulp_path": self.max_ulp_path,
            "failures": self.failures,
        }


def _block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    jax.effects_barrier()


def _materialized_run_def(reference: PaperLandpointReference) -> Path:
    if reference.output_dir is None or reference.run_def is None:
        raise FileNotFoundError(f"incomplete reference package for {reference.landpoint_id}")
    if reference.iteration_id is None or reference.param_set is None:
        raise ValueError(f"missing iteration metadata for {reference.landpoint_id}")
    output = RUN_DEF_ROOT / reference.landpoint_id / reference.iteration_id / reference.param_set / "used_run.def"
    base = reference.used_run_def or DEFAULT_BASE_RUN_DEF
    values = materialize_case_run_def_values(
        base_used_run_def=base,
        case_run_def=reference.run_def,
        base_is_fortran_used_truth=reference.used_run_def is not None,
    )
    return write_materialized_run_def(
        values,
        output,
        header_lines=(
            "# Materialized runtime run.def for compiled fast-path acceptance.",
            f"# Base defaults: {base}",
            f"# Landpoint overrides: {reference.run_def}",
        ),
    )


def _run_cold_start(
    reference: PaperLandpointReference,
    run_def: Path,
    *,
    days: int,
    compiled: bool,
    compiled_day_block_size: int = 0,
    numpy_accumulator: bool = False,
):
    started = time.perf_counter()
    run = paper_1961_driver_cold_start_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=days,
        used_run_def_path=run_def,
        reference_run_dir=reference.output_dir,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=compiled,
        compiled_day_block_size=(int(compiled_day_block_size) if compiled else 0),
        use_numpy_accumulator=bool(compiled and numpy_accumulator),
    )
    _block_until_ready(run)
    return run, time.perf_counter() - started


def _run_restart_year(
    reference: PaperLandpointReference,
    run_def: Path,
    previous_state: Any,
    *,
    days: int,
    compiled: bool,
    compiled_day_block_size: int = 0,
    numpy_accumulator: bool = False,
    reference_run_dir: Path | None = None,
):
    started = time.perf_counter()
    run = paper_1961_driver_restart_year_multiday_modelout_lite_run(
        CONFIG,
        previous_year_end_state=previous_state,
        year=1962,
        ndays=days,
        used_run_def_path=run_def,
        reference_run_dir=reference_run_dir or reference.output_dir,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=compiled,
        compiled_day_block_size=(int(compiled_day_block_size) if compiled else 0),
        use_numpy_accumulator=bool(compiled and numpy_accumulator),
    )
    _block_until_ready(run)
    return run, time.perf_counter() - started


def _compare_runs(actual: Any, expected: Any, *, days: int, atol: float, rtol: float) -> dict[str, object]:
    comparison = ComparisonAccumulator(atol=atol, rtol=rtol)
    comparison.compare("ready_for_requested_days", actual.ready_for_requested_days, expected.ready_for_requested_days)
    comparison.compare("missing_components", actual.missing_components, expected.missing_components)
    if len(actual.daily_modelout) != days or len(expected.daily_modelout) != days:
        comparison._fail(
            "daily_modelout",
            "day_count_mismatch",
            actual=len(actual.daily_modelout),
            expected=len(expected.daily_modelout),
            requested=days,
        )
    for index, (actual_day, expected_day) in enumerate(
        zip(actual.daily_modelout, expected.daily_modelout, strict=False), start=1
    ):
        comparison.compare(f"day{index}.day_index", actual_day.day_index, expected_day.day_index)
        comparison.compare(f"day{index}.start_tstep", actual_day.start_tstep, expected_day.start_tstep)
        comparison.compare(f"day{index}.end_tstep", actual_day.end_tstep, expected_day.end_tstep)
        comparison.compare(f"day{index}.modelout_fields", actual_day.modelout_fields, expected_day.modelout_fields)
    comparison.compare("last_day_end_state", actual.last_day_end_state, expected.last_day_end_state)
    return comparison.report()


def _annual_outputs(run: Any) -> dict[str, Any]:
    daily = [day.modelout_fields for day in run.daily_modelout]
    fields = annual_history_mean_fields_from_daily_modelout(
        daily, field_names=MODEL_OUTPUT_FIELD_NAMES, as_history_axes=False
    )
    return {"history_fields": fields, "modelout": compute_modelout_from_fields(fields)}


def _scan_cache_snapshot() -> dict[str, object]:
    callables = tuple(orchestration._COMPILED_SECHIBA_SCAN_CACHE.values())
    block_executables = tuple(orchestration._COMPILED_LATER_DAY_BLOCK_CACHE.values())
    return {
        "structural_entries": len(callables),
        "callable_ids": [id(value) for value in callables],
        "executable_counts": [int(value._cache_size()) for value in callables],
        "total_executables": sum(int(value._cache_size()) for value in callables),
        "block_entries": len(block_executables),
        "block_executable_ids": [id(value) for value in block_executables],
    }


def _split_restart_packet(reference: PaperLandpointReference, state: Any, output: Path) -> tuple[Any, Path]:
    if reference.output_dir is None:
        raise FileNotFoundError(reference.landpoint_id)
    context = prepare_paper_1961_driver_context(
        CONFIG, reference_run_dir=reference.output_dir
    )
    bundle = paper_restart_bundle_state_from_day_end_packet(
        state,
        base_stomate=context.first_step_restart_state.stomate_readstart,
        kjit=int(state.tstep) + 1,
    )
    physical = PaperRestartBundlePhysicalState(
        driver=read_restart_physical_state(reference.output_dir / "driver_start.nc"),
        sechiba=read_restart_physical_state(reference.output_dir / "sechiba_start.nc"),
        stomate=read_restart_physical_state(reference.output_dir / "stomate_start.nc"),
    )
    report = write_paper_restart_start_bundle(
        output,
        state=bundle,
        physical_state=physical,
        pft_layout=context.first_step_restart_state.pft_layout,
    )
    shutil.copyfile(output / "stomate_start.nc", output / "stomate_restart.nc")
    histories = sorted(reference.output_dir.glob("stomate_history_*.nc"))
    if histories:
        shutil.copyfile(histories[0], output / histories[0].name)
    restored = reference_case_first_step_restart_state(
        CONFIG, root=ROOT, run_dir=report.output_directory, stomate_filename="stomate_start.nc"
    )
    restored_packet = _packet_from_restart_readers(
        state,
        driver=read_dim2_driver_restart(output / "driver_start.nc"),
        sechiba=restored.sechiba_restart_state,
        stomate=restored.stomate_readstart,
    )
    return restored_packet, report.output_directory


def _scientific_tolerance_pass(report: dict[str, object]) -> bool:
    failures = report.get("failures", [])
    if any(item.get("reason") != "float_tolerance_failure" for item in failures):
        return False
    return float(report["max_abs_error"]) <= 1.0e-5 or float(report["max_rel_error"]) <= 1.0e-3


def run(
    *,
    landpoints: tuple[str, ...],
    short_days: int,
    annual_days: int,
    restart_days: int,
    atol: float,
    rtol: float,
    numpy_accumulator: bool = False,
    compiled_day_block_size: int = 7,
) -> dict[str, object]:
    orchestration._COMPILED_SECHIBA_SCAN_CACHE.clear()
    orchestration._COMPILED_LATER_DAY_BLOCK_CACHE.clear()
    initial_cache = _scan_cache_snapshot()
    rows: list[dict[str, object]] = []
    for landpoint_id in landpoints:
        reference = resolve_paper_landpoint_reference(ROOT, landpoint_id)
        if reference.output_dir is None:
            raise FileNotFoundError(f"reference output missing for {landpoint_id}")
        run_def = _materialized_run_def(reference)
        cache_before = _scan_cache_snapshot()

        compiled_short, compiled_short_seconds = _run_cold_start(
            reference,
            run_def,
            days=short_days,
            compiled=True,
            compiled_day_block_size=compiled_day_block_size,
            numpy_accumulator=numpy_accumulator,
        )
        cache_after_short = _scan_cache_snapshot()
        strict_short, strict_short_seconds = _run_cold_start(
            reference, run_def, days=short_days, compiled=False
        )
        short_comparison = _compare_runs(
            compiled_short, strict_short, days=short_days, atol=atol, rtol=rtol
        )
        del compiled_short, strict_short
        gc.collect()

        baseline_annual, baseline_annual_seconds = _run_cold_start(
            reference, run_def, days=annual_days, compiled=True
        )
        compiled_annual, compiled_annual_seconds = _run_cold_start(
            reference,
            run_def,
            days=annual_days,
            compiled=True,
            compiled_day_block_size=compiled_day_block_size,
            numpy_accumulator=numpy_accumulator,
        )
        annual_run_comparison = _compare_runs(
            compiled_annual, baseline_annual, days=annual_days, atol=atol, rtol=rtol
        )
        annual_output_comparison = ComparisonAccumulator(atol=atol, rtol=rtol)
        annual_output_comparison.compare(
            "annual", _annual_outputs(compiled_annual), _annual_outputs(baseline_annual)
        )
        annual_output_report = annual_output_comparison.report()
        baseline_annual_state = baseline_annual.last_day_end_state
        compiled_annual_state = compiled_annual.last_day_end_state
        del baseline_annual, compiled_annual
        gc.collect()

        strict_direct, strict_direct_seconds = _run_restart_year(
            reference,
            run_def,
            baseline_annual_state,
            days=restart_days,
            compiled=False,
        )
        compiled_direct, compiled_direct_seconds = _run_restart_year(
            reference,
            run_def,
            compiled_annual_state,
            days=restart_days,
            compiled=True,
            compiled_day_block_size=compiled_day_block_size,
            numpy_accumulator=numpy_accumulator,
        )
        direct_restart_comparison = _compare_runs(
            compiled_direct, strict_direct, days=restart_days, atol=atol, rtol=rtol
        )

        with tempfile.TemporaryDirectory(prefix=f"orchjax_fast_{landpoint_id.replace('.', '_')}_") as temporary:
            temporary_root = Path(temporary)
            strict_packet, strict_restart_dir = _split_restart_packet(
                reference, baseline_annual_state, temporary_root / "strict"
            )
            compiled_packet, compiled_restart_dir = _split_restart_packet(
                reference, compiled_annual_state, temporary_root / "compiled"
            )
            strict_split, strict_split_seconds = _run_restart_year(
                reference,
                run_def,
                strict_packet,
                days=restart_days,
                compiled=False,
                reference_run_dir=strict_restart_dir,
            )
            compiled_split, compiled_split_seconds = _run_restart_year(
                reference,
                run_def,
                compiled_packet,
                days=restart_days,
                compiled=True,
                compiled_day_block_size=compiled_day_block_size,
                numpy_accumulator=numpy_accumulator,
                reference_run_dir=compiled_restart_dir,
            )
            strict_split_roundtrip = _compare_runs(
                strict_split, strict_direct, days=restart_days, atol=atol, rtol=rtol
            )
            compiled_split_roundtrip = _compare_runs(
                compiled_split, compiled_direct, days=restart_days, atol=atol, rtol=rtol
            )
            split_mode_comparison = _compare_runs(
                compiled_split, strict_split, days=restart_days, atol=atol, rtol=rtol
            )

        cache_after = _scan_cache_snapshot()
        comparisons = {
            "short_window": short_comparison,
            "annual_block_vs_compiled_daily": annual_run_comparison,
            "annual_outputs": annual_output_report,
            "cross_year_direct": direct_restart_comparison,
            "strict_restart_roundtrip": strict_split_roundtrip,
            "compiled_restart_roundtrip": compiled_split_roundtrip,
            "cross_year_split": split_mode_comparison,
        }
        row_passed = all(bool(report["passed"]) for report in comparisons.values())
        rows.append(
            {
                "landpoint_id": landpoint_id,
                "passed": row_passed,
                "run_def": str(run_def),
                "reference_run_dir": str(reference.output_dir),
                "timing_seconds": {
                    "short_strict": strict_short_seconds,
                    "short_compiled": compiled_short_seconds,
                    "annual_compiled_daily": baseline_annual_seconds,
                    "annual_compiled": compiled_annual_seconds,
                    "cross_year_direct_strict": strict_direct_seconds,
                    "cross_year_direct_compiled": compiled_direct_seconds,
                    "cross_year_split_strict": strict_split_seconds,
                    "cross_year_split_compiled": compiled_split_seconds,
                },
                "cache_before": cache_before,
                "cache_after_short": cache_after_short,
                "cache_after": cache_after,
                "comparisons": comparisons,
                "annual_scientific_tolerance_pass": _scientific_tolerance_pass(
                    annual_output_report
                ),
            }
        )

    final_cache = _scan_cache_snapshot()
    same_callable = final_cache["structural_entries"] == 1
    later_landpoints_added_no_executable = all(
        row["cache_after"]["total_executables"]
        == row["cache_before"]["total_executables"]
        for row in rows[1:]
    )
    executable_reuse = bool(same_callable and later_landpoints_added_no_executable)
    passed = bool(rows and all(row["passed"] for row in rows) and executable_reuse)
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scope": (
            "strict versus compiled block: cold-start/cross-day, compiled-daily annual output and full "
            "day-end state, cross-year direct and three-file split restart, cross-landpoint executable reuse"
        ),
        "policy": {
            "float_atol": atol,
            "float_rtol": rtol,
            "discrete": "exact",
            "schema": "exact",
            "annual_scientific_atol": 1.0e-5,
            "annual_scientific_rtol": 1.0e-3,
            "fast_path_gate_uses_stricter_numerical_tolerance": True,
        },
        "short_days": short_days,
        "annual_days": annual_days,
        "restart_days": restart_days,
        "compiled_day_block_size": int(compiled_day_block_size),
        "numpy_accumulator": bool(numpy_accumulator),
        "initial_cache": initial_cache,
        "final_cache": final_cache,
        "same_scan_callable": same_callable,
        "later_landpoints_added_no_executable": later_landpoints_added_no_executable,
        "executable_reuse_pass": executable_reuse,
        "landpoints": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the compiled SECHIBA day production fast path.")
    parser.add_argument("--landpoint-id", action="append", default=[])
    parser.add_argument("--short-days", type=int, default=3)
    parser.add_argument("--annual-days", type=int, default=365)
    parser.add_argument("--restart-days", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--numpy-accumulator", choices=("on", "off"), default="off")
    parser.add_argument("--compiled-day-block-size", type=int, default=7)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if min(args.short_days, args.annual_days, args.restart_days) < 1:
        raise ValueError("all day counts must be positive")
    result = run(
        landpoints=tuple(args.landpoint_id or LANDPOINTS),
        short_days=args.short_days,
        annual_days=args.annual_days,
        restart_days=args.restart_days,
        atol=args.atol,
        rtol=args.rtol,
        numpy_accumulator=args.numpy_accumulator == "on",
        compiled_day_block_size=args.compiled_day_block_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "executable_reuse_pass": result["executable_reuse_pass"],
                "landpoints": [
                    {"landpoint_id": row["landpoint_id"], "passed": row["passed"]}
                    for row in result["landpoints"]
                ],
            },
            indent=2,
        )
    )
    return int(args.verify and result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
