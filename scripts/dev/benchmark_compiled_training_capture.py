from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration as teacher  # noqa: E402
from research.daily_coarse_graining.gradient_training_ready import (  # noqa: E402
    _load_state_cache,
)
from research.daily_coarse_graining.supervised_learnability_pilot import (  # noqa: E402
    _capture_days_compiled_blocks,
    _teacher_target_vector,
    build_boundary_vector_spec,
)

DEFAULT_CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "performance"
    / "daily_coarse_graining"
    / "compiled_training_capture_in_process.json"
)


def _cpu_affinity() -> list[int] | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    return sorted(int(value) for value in getter(0))


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _git_metadata() -> dict[str, object]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "clean": None}
    return {"head": head, "clean": not bool(status)}


def _timing_statistics(values: list[float], *, days: int) -> dict[str, object]:
    return {
        "runs_seconds": values,
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.fmean(values),
        "maximum_seconds": max(values),
        "median_seconds_per_requested_day": statistics.median(values) / days,
    }


def _capture(*, args, context, initial_state):
    started = time.perf_counter()
    _, _, records, final_state = _capture_days_compiled_blocks(
        config_path=args.teacher_config,
        context=context,
        previous_state=initial_state,
        year=args.year,
        start_day=1,
        days=args.days,
        block_size=args.block_size,
    )
    seconds = time.perf_counter() - started
    spec = build_boundary_vector_spec(records[0])
    targets = np.stack([_teacher_target_vector(record, spec) for record in records])
    state = teacher.fast_state_from_previous_packet(final_state).values_by_component
    return seconds, targets, jax.device_get(state)


def _array_metrics(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype.kind != expected.dtype.kind:
        return {"passed": False, "schema_equal": False}
    if actual.dtype.kind in "biu":
        exact = bool(np.array_equal(actual, expected))
        return {"passed": exact, "schema_equal": True, "exact": exact}
    finite = np.isfinite(actual) & np.isfinite(expected)
    nonfinite_equal = bool(np.array_equal(np.isfinite(actual), np.isfinite(expected)))
    if np.any(finite):
        delta = np.abs(actual[finite] - expected[finite])
        denominator = np.maximum(np.abs(expected[finite]), 1.0)
        max_abs = float(np.max(delta))
        max_rel = float(np.max(delta / denominator))
    else:
        max_abs = 0.0
        max_rel = 0.0
    passed = bool(
        nonfinite_equal
        and np.allclose(actual, expected, atol=1.0e-8, rtol=1.0e-10, equal_nan=True)
    )
    return {
        "passed": passed,
        "schema_equal": True,
        "nonfinite_pattern_equal": nonfinite_equal,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    }


def _tree_metrics(actual, expected):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    if actual_tree != expected_tree:
        return {"passed": False, "tree_equal": False}
    leaves = [_array_metrics(left, right) for left, right in zip(actual_leaves, expected_leaves, strict=True)]
    return {
        "passed": all(item["passed"] for item in leaves),
        "tree_equal": True,
        "leaf_count": len(leaves),
        "max_abs_error": max((item.get("max_abs_error", 0.0) for item in leaves), default=0.0),
        "max_rel_error": max((item.get("max_rel_error", 0.0) for item in leaves), default=0.0),
        "failed_leaves": [index for index, item in enumerate(leaves) if not item["passed"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure cold and in-process hot compiled Teacher capture."
    )
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-def", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=1962)
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--hot-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (args.days - 1) % args.block_size:
        raise ValueError("benchmark days minus the audited first day must fill complete blocks")
    if args.hot_repeats < 3:
        raise ValueError("accepted performance benchmarks require at least three hot repeats")

    setup_started = time.perf_counter()
    cache = _load_state_cache(args.state_cache)
    context = teacher.prepare_paper_1961_driver_context(
        args.teacher_config,
        used_run_def_path=args.run_def,
        reference_run_dir=args.reference_run_dir,
    )
    initial_state = teacher.rebase_driver_state_for_year_start(cache["state"])
    setup_seconds = time.perf_counter() - setup_started
    first = _capture(args=args, context=context, initial_state=initial_state)
    hot_seconds = []
    comparisons = []
    for repeat in range(args.hot_repeats):
        hot = _capture(args=args, context=context, initial_state=initial_state)
        hot_seconds.append(hot[0])
        target_metrics = _array_metrics(hot[1], first[1])
        state_metrics = _tree_metrics(hot[2], first[2])
        comparisons.append(
            {
                "repeat": repeat + 1,
                "targets": target_metrics,
                "final_state": state_metrics,
                "passed": bool(target_metrics["passed"] and state_metrics["passed"]),
            }
        )
    timing = _timing_statistics(hot_seconds, days=args.days)
    affinity = _cpu_affinity()
    threading_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "XLA_FLAGS",
            "JAX_PLATFORMS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_ID",
        )
    }
    payload = {
        "schema_version": "compiled_training_capture_in_process_v2",
        "year": args.year,
        "days": args.days,
        "block_size": args.block_size,
        "hot_repeats": args.hot_repeats,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "timing_seconds": {
            "setup_not_in_capture": setup_seconds,
            "cold_capture": first[0],
            "cold_seconds_per_requested_day": first[0] / args.days,
            "hot": timing,
        },
        "comparison": {
            "repeats": comparisons,
            "passed": all(item["passed"] for item in comparisons),
        },
        "cache": {
            "later_day_block_entries": len(teacher._COMPILED_LATER_DAY_BLOCK_CACHE),
            "sechiba_scan_entries": len(teacher._COMPILED_SECHIBA_SCAN_CACHE),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_affinity": affinity,
            "cpu_affinity_count": None if affinity is None else len(affinity),
            "devices": [str(device) for device in jax.devices()],
            "threading_environment": threading_environment,
            "peak_rss_bytes": _peak_rss_bytes(),
            "git": _git_metadata(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
