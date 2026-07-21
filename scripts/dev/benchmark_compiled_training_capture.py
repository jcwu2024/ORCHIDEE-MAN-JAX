from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (args.days - 1) % args.block_size:
        raise ValueError("benchmark days minus the audited first day must fill complete blocks")

    cache = _load_state_cache(args.state_cache)
    context = teacher.prepare_paper_1961_driver_context(
        args.teacher_config,
        used_run_def_path=args.run_def,
        reference_run_dir=args.reference_run_dir,
    )
    initial_state = teacher.rebase_driver_state_for_year_start(cache["state"])
    first = _capture(args=args, context=context, initial_state=initial_state)
    second = _capture(args=args, context=context, initial_state=initial_state)
    target_metrics = _array_metrics(second[1], first[1])
    state_metrics = _tree_metrics(second[2], first[2])
    payload = {
        "schema_version": "compiled_training_capture_in_process_v1",
        "year": args.year,
        "days": args.days,
        "block_size": args.block_size,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "timing_seconds": {
            "cold_capture": first[0],
            "hot_capture": second[0],
            "cold_seconds_per_requested_day": first[0] / args.days,
            "hot_seconds_per_requested_day": second[0] / args.days,
        },
        "comparison": {
            "targets": target_metrics,
            "final_state": state_metrics,
            "passed": bool(target_metrics["passed"] and state_metrics["passed"]),
        },
        "cache": {
            "later_day_block_entries": len(teacher._COMPILED_LATER_DAY_BLOCK_CACHE),
            "sechiba_scan_entries": len(teacher._COMPILED_SECHIBA_SCAN_CACHE),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
