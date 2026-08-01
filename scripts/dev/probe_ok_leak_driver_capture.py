"""Capture and replay one real 48-step OK_LEAK driver sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.canonical_teacher_reentry import (
    teacher_reentry_packet,
    teacher_reentry_templates,
)
from research.daily_coarse_graining.carbon_budget_ownership import (
    extract_ok_leak_driver_series,
    ok_leak_driver_capture_metadata,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    extract_fast_day_target,
    extract_state,
    load_markov_shard,
)
from research.daily_coarse_graining.replay_ceiling import (
    capture_pre_daily_stomate_record,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _max_tree_difference(actual: Any, expected: Any) -> Mapping[str, Any]:
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    if actual_tree != expected_tree or len(actual_leaves) != len(expected_leaves):
        return {
            "schema_equal": False,
            "exact": False,
            "max_absolute_error": None,
            "failed_leaf_count": None,
        }
    maximum = 0.0
    failed = 0
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        if actual_array.shape != expected_array.shape or actual_array.dtype != expected_array.dtype:
            failed += 1
            continue
        equal = np.array_equal(actual_array, expected_array, equal_nan=True)
        if not equal:
            failed += 1
        if actual_array.dtype.kind in "fc" and actual_array.size:
            difference = np.abs(actual_array - expected_array)
            finite = np.isfinite(difference)
            if np.any(finite):
                maximum = max(maximum, float(np.max(difference[finite])))
    return {
        "schema_equal": True,
        "exact": failed == 0,
        "max_absolute_error": maximum,
        "failed_leaf_count": failed,
    }


def _array_comparison(
    actual: Any,
    expected: Any,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> Mapping[str, Any]:
    if (atol is None) != (rtol is None):
        raise ValueError("atol and rtol must be declared together")
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        result = {
            "shape_equal": False,
            "exact": False,
            "max_absolute_error": None,
        }
        if atol is not None and rtol is not None:
            result.update(
                {
                    "atol": atol,
                    "rtol": rtol,
                    "within_tolerance": False,
                    "max_tolerance_ratio": None,
                }
            )
        return result
    exact = bool(np.array_equal(actual_array, expected_array, equal_nan=True))
    defined_status_mismatches = int(
        np.count_nonzero(np.isfinite(actual_array) != np.isfinite(expected_array))
    )
    if actual_array.dtype.kind == "b" or expected_array.dtype.kind == "b":
        maximum = 0.0 if exact else 1.0
    else:
        difference = np.abs(
            actual_array.astype(np.float64) - expected_array.astype(np.float64)
        )
        finite = np.isfinite(difference)
        maximum = float(np.max(difference[finite])) if np.any(finite) else 0.0
    result = {
        "shape_equal": True,
        "exact": exact,
        "max_absolute_error": maximum,
        "defined_status_mismatches": defined_status_mismatches,
    }
    if atol is not None and rtol is not None:
        close = np.isclose(
            actual_array,
            expected_array,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
        difference = np.abs(
            actual_array.astype(np.float64) - expected_array.astype(np.float64)
        )
        allowed = atol + rtol * np.abs(expected_array.astype(np.float64))
        finite = np.isfinite(difference) & np.isfinite(allowed)
        ratios = np.divide(
            difference,
            allowed,
            out=np.zeros_like(difference),
            where=finite & (allowed > 0.0),
        )
        max_ratio = float(np.max(ratios[finite])) if np.any(finite) else 0.0
        result.update(
            {
                "atol": atol,
                "rtol": rtol,
                "within_tolerance": bool(np.all(close)),
                "max_tolerance_ratio": max_ratio,
            }
        )
    return result


def _target_leaf_comparisons(actual, expected, leaves) -> Mapping[str, Any]:
    result = {}
    for leaf in leaves:
        key = f"{leaf.family}.{leaf.path[0]}"
        result[key] = _array_comparison(
            np.asarray(actual)[leaf.start : leaf.stop],
            np.asarray(expected)[leaf.start : leaf.stop],
        )
    return result


def _state_leaf_comparisons(
    actual,
    expected,
    leaves,
    *,
    atol: float,
    rtol: float,
) -> Mapping[str, Any]:
    return {
        leaf.key: _array_comparison(
            np.asarray(actual)[leaf.start : leaf.stop],
            np.asarray(expected)[leaf.start : leaf.stop],
            atol=atol,
            rtol=rtol,
        )
        for leaf in leaves
        if not leaf.discrete
    }


def _within_tolerance(comparison: Mapping[str, Any], tolerance: float) -> bool:
    return bool(
        comparison["shape_equal"]
        and comparison.get("defined_status_mismatches", 0) == 0
        and comparison["max_absolute_error"] is not None
        and comparison["max_absolute_error"] <= tolerance
    )


def _ok_leak_endpoint_comparisons(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    return {
        key: value
        for key, value in comparisons.items()
        if key.startswith("ok_leak.")
    }


def _capture_probe_passes(
    source_driver_comparisons: Mapping[str, Mapping[str, Any]],
    replay_tree: Mapping[str, Any],
    ok_leak_comparisons: Mapping[str, Mapping[str, Any]],
    discrete_comparison: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    return bool(
        source_driver_comparisons
        and all(item["exact"] for item in source_driver_comparisons.values())
        and replay_tree["exact"]
        and ok_leak_comparisons
        and all(
            _within_tolerance(item, tolerance)
            for item in ok_leak_comparisons.values()
        )
        and all(item["exact"] for item in discrete_comparison.values())
    )


def run_probe(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest_path = args.dataset_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    references = tuple(
        item
        for item in manifest["shards"]
        if item["landpoint_id"] == args.landpoint_id and int(item["year"]) == args.year
    )
    if len(references) != 1:
        raise ValueError("capture probe requires exactly one matching shard")
    reference = references[0]
    shard_path = (manifest_path.parent / reference["shard"]).resolve()
    shard = load_markov_shard(shard_path)
    matches = np.flatnonzero(np.asarray(shard.day_index) == args.day_index)
    if matches.size != 1:
        raise ValueError("capture probe day is absent or duplicated in the shard")
    row = int(matches[0])
    day_start_state_sha256 = _sha256_array(shard.state_trajectory[row])
    fast_day_target_sha256 = _sha256_array(shard.fast_day_target[row])
    expected_state_hash = getattr(args, "expected_day_start_state_sha256", None)
    expected_target_hash = getattr(args, "expected_fast_day_target_sha256", None)
    if expected_state_hash is not None and day_start_state_sha256 != expected_state_hash:
        raise ValueError("capture plan day-start state hash mismatch")
    if expected_target_hash is not None and fast_day_target_sha256 != expected_target_hash:
        raise ValueError("capture plan fast-day target hash mismatch")

    plan_path = args.plan.resolve() if args.plan is not None else _resolve(manifest_path.parents[3], manifest["plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = tuple(
        item
        for item in plan["entries"]
        if item["landpoint_id"] == args.landpoint_id and int(item["year"]) == args.year
    )
    if len(entries) != 1:
        raise ValueError("capture probe requires exactly one matching plan entry")
    entry = entries[0]
    root = Path(__file__).resolve().parents[2]
    config_path = _resolve(root, plan["teacher_config"])
    context = teacher.prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=_resolve(root, entry["run_def"]),
        reference_run_dir=_resolve(root, entry["reference_run_dir"]),
    )
    templates = teacher_reentry_templates(context)
    discrete = {
        name: np.asarray(values[row])
        for name, values in shard.discrete_trajectories.items()
    }
    packet = teacher_reentry_packet(
        np.asarray(shard.state_trajectory[row]),
        discrete,
        contract,
        tstep=(args.day_index - 1) * templates.steps_per_day - 1,
        templates=templates,
    )

    original_scan = teacher._paper_compiled_ok_leak_fold_jit
    captured_call: dict[str, Any] = {}
    captured_return: dict[str, Any] = {}

    def capture_scan(*scan_args, **scan_kwargs):
        if captured_call:
            raise RuntimeError("compiled OK_LEAK scan executed more than once")
        captured_call.update(scan_kwargs)
        value = original_scan(*scan_args, **scan_kwargs)
        captured_return["value"] = value
        return value

    common = {
        "used_run_def_path": context.run_def_path,
        "prepared_context": context,
        "module_jit": True,
        "diffuco_local_jit": True,
        "use_static_jit_daily_carbon": True,
        "use_compiled_sechiba_day": True,
        "retain_stomate_step_results": False,
        "prebuild_day_payloads": True,
    }
    with patch.object(teacher, "_paper_compiled_ok_leak_fold_jit", capture_scan):
        record = capture_pre_daily_stomate_record(
            config_path,
            previous_state=packet,
            year=args.year,
            day_index=args.day_index,
            start_tstep=(args.day_index - 1) * templates.steps_per_day,
            **common,
        )
    if not captured_call or "value" not in captured_return:
        raise RuntimeError("capture probe did not observe the compiled OK_LEAK scan")

    arrays = extract_ok_leak_driver_series(record)
    original_steps = captured_call["steps"]
    source_driver_comparisons = {
        name: _array_comparison(arrays[name], getattr(original_steps, name))
        for name in arrays
    }
    replay_call = dict(captured_call)
    replay_call["steps"] = teacher.DriverCompiledOkLeakStepInputs(**arrays)
    replayed = original_scan(**replay_call)
    replay_tree = _max_tree_difference(replayed, captured_return["value"])

    observed_target = np.asarray(
        extract_fast_day_target(record, contract.fast_day_target_leaves)
    )
    target_comparison = _array_comparison(observed_target, shard.fast_day_target[row])
    target_leaf_comparisons = _target_leaf_comparisons(
        observed_target,
        shard.fast_day_target[row],
        contract.fast_day_target_leaves,
    )
    ok_leak_comparisons = _ok_leak_endpoint_comparisons(target_leaf_comparisons)
    observed_state, observed_discrete = extract_state(
        record.expected_result.day_end_state,
        contract,
    )
    state_atol = 1.0e-12
    state_rtol = 1.0e-12
    expected_state = shard.state_trajectory[row + 1]
    state_comparison = _array_comparison(
        observed_state,
        expected_state,
        atol=state_atol,
        rtol=state_rtol,
    )
    state_leaf_comparisons = _state_leaf_comparisons(
        observed_state,
        expected_state,
        contract.state_leaves,
        atol=state_atol,
        rtol=state_rtol,
    )
    discrete_comparison = {
        name: _array_comparison(observed_discrete[name], values[row + 1])
        for name, values in shard.discrete_trajectories.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output / "ok_leak_driver_series.npz"
    arrays_temporary = arrays_path.with_suffix(".npz.tmp")
    with arrays_temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    arrays_temporary.replace(arrays_path)
    capture_metadata = ok_leak_driver_capture_metadata(arrays)
    passed = _capture_probe_passes(
        source_driver_comparisons,
        replay_tree,
        ok_leak_comparisons,
        discrete_comparison,
    )
    report = {
        "schema_version": "ok_leak_driver_capture_probe_v2",
        "passed": passed,
        "capture_interface_passed": passed,
        "dataset_manifest": str(manifest_path),
        "dataset_id": manifest["dataset_id"],
        "teacher_git_head": manifest["teacher_git_head"],
        "markov_contract_sha256": manifest["markov_contract_sha256"],
        "landpoint_id": args.landpoint_id,
        "year": args.year,
        "day_index": args.day_index,
        "source_shard": str(shard_path),
        "source_shard_sha256": reference["shard_sha256"],
        "day_start_state_sha256": day_start_state_sha256,
        "fast_day_target_sha256": fast_day_target_sha256,
        "capture_plan_sha256": getattr(args, "capture_plan_sha256", None),
        "capture": capture_metadata,
        "capture_npz": arrays_path.name,
        "capture_npz_sha256": _sha256_file(arrays_path),
        "source_driver_comparisons": source_driver_comparisons,
        "exact_scan_replay": replay_tree,
        "fast_day_target": target_comparison,
        "fast_day_target_leaves": target_leaf_comparisons,
        "ok_leak_endpoint_within_1e-12": all(
            _within_tolerance(item, 1.0e-12)
            for item in ok_leak_comparisons.values()
        ),
        "next_continuous_state": state_comparison,
        "next_state_diagnostic_passed": state_comparison["within_tolerance"],
        "next_continuous_state_leaves": state_leaf_comparisons,
        "next_discrete_state": discrete_comparison,
        "sealed_test_used": False,
    }
    report_path = args.output / "report.json"
    _atomic_write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--landpoint-id", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--day-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_probe(args)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
