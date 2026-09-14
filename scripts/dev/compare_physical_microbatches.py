"""Compare isolated physical-v2 microbatch-8 and microbatch-16 probes."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "physical_microbatch_comparison_v1"
PROBE_SCHEMA_VERSION = "physical_microbatch_probe_v1"
STAGE_NAME = "first_update"
EXPECTED_PARENT_BATCH_SIZE = 64
EXPECTED_TIMED_UPDATES = 3
CHUNK_ELEMENTS = 1_000_000
MINIMUM_SPEED_GAIN_FRACTION = 0.05
RESOURCE_ENVELOPE_KEYS = ("compiled_memory_limit_gib", "gpu_memory_fraction")

# Frozen before the benchmark results are inspected. The coordinate floors
# cover reduction-order noise near zero; the relative L2 gates constrain the
# scientifically consequential aggregate direction and realized Adam step.
TOLERANCES = {
    "loss": {"relative": 1.0e-6, "absolute": 1.0e-8},
    "gradient": {"relative_l2": 1.0e-4, "coordinate_relative": 1.0e-4, "coordinate_absolute": 1.0e-6},
    "parameter_delta": {
        "relative_l2": 1.0e-3,
        "coordinate_relative": 1.0e-3,
        "coordinate_absolute": 1.0e-7,
    },
}


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _flatten_tree(value: Any, path: str = "root") -> tuple[Any, list[tuple[str, np.ndarray]]]:
    """Return a strict, JSON-compatible structure descriptor and array leaves."""

    # JAX treats None as an empty pytree node, used by inactive axis embeddings.
    if value is None:
        return ("none",), []
    if isinstance(value, Mapping):
        children = []
        leaves: list[tuple[str, np.ndarray]] = []
        for key, child in value.items():
            child_structure, child_leaves = _flatten_tree(child, f"{path}[{key!r}]")
            children.append((repr(key), child_structure))
            leaves.extend(child_leaves)
        return ("mapping", _type_name(value), tuple(children)), leaves
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        children = []
        leaves = []
        for name, child in zip(value._fields, value, strict=True):
            child_structure, child_leaves = _flatten_tree(child, f"{path}.{name}")
            children.append((name, child_structure))
            leaves.extend(child_leaves)
        return ("namedtuple", _type_name(value), tuple(children)), leaves
    if isinstance(value, (tuple, list)):
        children = []
        leaves = []
        for index, child in enumerate(value):
            child_structure, child_leaves = _flatten_tree(child, f"{path}[{index}]")
            children.append(child_structure)
            leaves.extend(child_leaves)
        return ("sequence", _type_name(value), tuple(children)), leaves

    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"{path} is not a numeric array leaf: {array.dtype}")
    return ("leaf",), [(path, array)]


def _leaf_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    coordinate_relative: float,
    coordinate_absolute: float,
) -> dict[str, Any]:
    count = int(left.size)
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_squared = right_squared = error_squared = 0.0
    maximum_absolute_error = 0.0
    failed_coordinates = 0
    all_finite = True
    for start in range(0, count, CHUNK_ELEMENTS):
        stop = min(start + CHUNK_ELEMENTS, count)
        a = np.asarray(left_flat[start:stop], dtype=np.float64)
        b = np.asarray(right_flat[start:stop], dtype=np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        all_finite = all_finite and bool(np.all(finite))
        difference = a - b
        absolute = np.abs(difference)
        limit = coordinate_absolute + coordinate_relative * np.maximum(np.abs(a), np.abs(b))
        failed_coordinates += int(np.count_nonzero(~finite | ~np.isfinite(absolute) | (absolute > limit)))
        if absolute.size:
            observed_maximum = float(np.max(absolute))
            maximum_absolute_error = max(maximum_absolute_error, observed_maximum)
        if all_finite:
            left_squared += float(np.sum(a * a, dtype=np.float64))
            right_squared += float(np.sum(b * b, dtype=np.float64))
            error_squared += float(np.sum(difference * difference, dtype=np.float64))
    if not all_finite:
        left_squared = right_squared = error_squared = math.inf
        maximum_absolute_error = math.inf
    return {
        "coordinate_count": count,
        "left_squared_l2": left_squared,
        "right_squared_l2": right_squared,
        "error_squared_l2": error_squared,
        "maximum_absolute_error": maximum_absolute_error,
        "failed_coordinates": failed_coordinates,
        "all_finite": all_finite,
    }


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def compare_numeric_trees(left: Any, right: Any, *, tolerance: Mapping[str, float]) -> dict[str, Any]:
    """Compare two host pytrees without concatenating their large parameter leaves."""

    left_structure, left_leaves = _flatten_tree(left)
    right_structure, right_leaves = _flatten_tree(right)
    structure_matches = left_structure == right_structure
    if not structure_matches:
        return {
            "passed": False,
            "structure_matches": False,
            "shapes_match": False,
            "dtypes_match": False,
            "reason": "tree structures differ",
        }

    shapes_match = all(a.shape == b.shape for (_, a), (_, b) in zip(left_leaves, right_leaves, strict=True))
    dtypes_match = all(a.dtype == b.dtype for (_, a), (_, b) in zip(left_leaves, right_leaves, strict=True))
    if not shapes_match or not dtypes_match:
        return {
            "passed": False,
            "structure_matches": True,
            "shapes_match": shapes_match,
            "dtypes_match": dtypes_match,
            "reason": "tree leaf shapes or dtypes differ",
        }

    totals = {
        "coordinate_count": 0,
        "left_squared_l2": 0.0,
        "right_squared_l2": 0.0,
        "error_squared_l2": 0.0,
        "maximum_absolute_error": 0.0,
        "failed_coordinates": 0,
        "all_finite": True,
    }
    leaf_records = []
    for (path, a), (_, b) in zip(left_leaves, right_leaves, strict=True):
        metrics = _leaf_metrics(
            a,
            b,
            coordinate_relative=float(tolerance["coordinate_relative"]),
            coordinate_absolute=float(tolerance["coordinate_absolute"]),
        )
        leaf_error = math.sqrt(metrics["error_squared_l2"])
        leaf_scale = max(math.sqrt(metrics["left_squared_l2"]), math.sqrt(metrics["right_squared_l2"]))
        leaf_relative = leaf_error / max(leaf_scale, np.finfo(np.float64).tiny)
        leaf_records.append(
            {
                "path": path,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "error_l2": _finite_number(leaf_error),
                "symmetric_scale_l2": _finite_number(leaf_scale),
                "relative_l2_error": _finite_number(leaf_relative),
                "maximum_absolute_error": _finite_number(metrics["maximum_absolute_error"]),
                "failed_coordinates": metrics["failed_coordinates"],
            }
        )
        for name in ("coordinate_count", "left_squared_l2", "right_squared_l2", "error_squared_l2", "failed_coordinates"):
            totals[name] += metrics[name]
        totals["maximum_absolute_error"] = max(
            totals["maximum_absolute_error"], metrics["maximum_absolute_error"]
        )
        totals["all_finite"] = totals["all_finite"] and metrics["all_finite"]

    error_l2 = math.sqrt(totals["error_squared_l2"])
    left_l2 = math.sqrt(totals["left_squared_l2"])
    right_l2 = math.sqrt(totals["right_squared_l2"])
    scale_l2 = max(left_l2, right_l2)
    relative_l2 = error_l2 / max(scale_l2, np.finfo(np.float64).tiny)
    global_limit = float(tolerance["coordinate_absolute"]) + float(tolerance["relative_l2"]) * scale_l2
    leaf_records.sort(
        key=lambda row: math.inf if row["relative_l2_error"] is None else row["relative_l2_error"],
        reverse=True,
    )
    passed = bool(
        totals["all_finite"]
        and totals["failed_coordinates"] == 0
        and error_l2 <= global_limit
    )
    return {
        "passed": passed,
        "structure_matches": True,
        "shapes_match": True,
        "dtypes_match": True,
        "all_values_finite": totals["all_finite"],
        "coordinate_count": totals["coordinate_count"],
        "left_l2": _finite_number(left_l2),
        "right_l2": _finite_number(right_l2),
        "error_l2": _finite_number(error_l2),
        "relative_l2_error": _finite_number(relative_l2),
        "relative_l2_tolerance": float(tolerance["relative_l2"]),
        "global_l2_limit": _finite_number(global_limit),
        "maximum_absolute_error": _finite_number(totals["maximum_absolute_error"]),
        "coordinate_relative_tolerance": float(tolerance["coordinate_relative"]),
        "coordinate_absolute_tolerance": float(tolerance["coordinate_absolute"]),
        "failed_coordinates": totals["failed_coordinates"],
        "largest_relative_leaf_errors": leaf_records[:12],
    }


def _loss_comparison(left: float, right: float) -> dict[str, Any]:
    relative = TOLERANCES["loss"]["relative"]
    absolute = TOLERANCES["loss"]["absolute"]
    finite = bool(np.isfinite(left) and np.isfinite(right))
    error = abs(left - right)
    limit = absolute + relative * max(abs(left), abs(right))
    return {
        "passed": bool(finite and error <= limit),
        "microbatch_8": _finite_number(left),
        "microbatch_16": _finite_number(right),
        "absolute_error": _finite_number(error),
        "symmetric_limit": _finite_number(limit),
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
    }


def _validate_probe(report: Mapping[str, Any], *, microbatch_size: int) -> dict[str, Any]:
    if report.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise ValueError(f"microbatch-{microbatch_size} report schema drift")
    if report.get("microbatch_size") != microbatch_size:
        raise ValueError(f"expected microbatch-{microbatch_size} report")
    if report.get("parent_batch_size") != EXPECTED_PARENT_BATCH_SIZE:
        raise ValueError("benchmark parent batch is not 64")
    if not isinstance(report.get("identity"), dict):
        raise ValueError("benchmark report has no shared identity")
    protocol = report["identity"].get("benchmark_protocol")
    if not isinstance(protocol, dict) or protocol.get("schema_version") != "physical_microbatch_benchmark_v1":
        raise ValueError("benchmark protocol identity drift")
    if protocol.get("timed_updates") != EXPECTED_TIMED_UPDATES:
        raise ValueError("benchmark must declare three timed updates")
    declared_memory_fraction = float(protocol.get("gpu_memory_fraction", math.nan))
    observed_memory_fraction = float(report.get("memory_fraction", math.nan))
    memory_fraction_consistent = bool(
        np.isfinite(declared_memory_fraction)
        and 0.0 < declared_memory_fraction <= 1.0
        and np.isclose(declared_memory_fraction, observed_memory_fraction, rtol=0.0, atol=1.0e-12)
    )

    updates = report.get("updates")
    if not isinstance(updates, list) or len(updates) != EXPECTED_TIMED_UPDATES:
        raise ValueError("benchmark report must contain three timed updates")
    update_gates = []
    for expected_index, update in enumerate(updates, start=1):
        hard = update.get("hard")
        update_gates.append(
            bool(
                update.get("update") == expected_index
                and update.get("parameters_optimizer_finite") is True
                and isinstance(hard, dict)
                and all(value == 0 for value in hard.values())
                and np.isfinite(float(update.get("loss", math.nan)))
                and np.isfinite(float(update.get("compute_seconds", math.nan)))
                and float(update.get("compute_seconds", 0.0)) > 0.0
            )
        )
    seconds = [float(update["compute_seconds"]) for update in updates]
    observed_median = float(np.median(seconds))
    declared_median = float(report.get("median_compute_seconds", math.nan))
    median_consistent = bool(np.isclose(observed_median, declared_median, rtol=1e-12, atol=1e-12))

    compiled = report.get("compiled_memory")
    memory = report.get("memory")
    compiled_bytes = int(compiled.get("estimated_live_bytes", -1)) if isinstance(compiled, dict) else -1
    peak_bytes = int(memory.get("peak_bytes_in_use", -1)) if isinstance(memory, dict) else -1
    bytes_limit = int(memory.get("bytes_limit", -1)) if isinstance(memory, dict) else -1
    compiled_limit = float(protocol["compiled_memory_limit_gib"]) * 1024**3
    memory_available = bool(isinstance(memory, dict) and memory.get("available") is True)
    resource_feasible = bool(
        compiled_bytes >= 0
        and compiled_bytes <= compiled_limit
        and memory_available
        and peak_bytes >= 0
        and bytes_limit > 0
        and peak_bytes <= bytes_limit
    )
    execution_passed = bool(
        report.get("status") == "passed"
        and report.get("completed_updates") == EXPECTED_TIMED_UPDATES
        and report.get("warmup_updates_discarded") == 1
        and report.get("same_batch_repeated") is True
        and report.get("all_hard_counts_zero") is True
        and report.get("all_updates_finite") is True
        and report.get("retained_model") is False
        and report.get("checkpoint_written") is False
        and report.get("checkpoint_inherited") is False
        and all(update_gates)
        and median_consistent
        and memory_fraction_consistent
    )
    return {
        "passed": bool(execution_passed and resource_feasible),
        "execution_passed": execution_passed,
        "resource_feasible": resource_feasible,
        "update_gates": update_gates,
        "median_consistent": median_consistent,
        "declared_gpu_memory_fraction": declared_memory_fraction,
        "observed_gpu_memory_fraction": observed_memory_fraction,
        "gpu_memory_fraction_consistent": memory_fraction_consistent,
        "median_compute_seconds": observed_median,
        "samples_per_second": EXPECTED_PARENT_BATCH_SIZE / observed_median,
        "compiled_estimated_live_bytes": compiled_bytes,
        "compiled_memory_limit_bytes": int(compiled_limit),
        "peak_bytes_in_use": peak_bytes,
        "bytes_limit": bytes_limit,
        "peak_headroom_bytes": bytes_limit - peak_bytes if bytes_limit > 0 and peak_bytes >= 0 else None,
    }


def _split_identity(identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Separate the two permitted runtime-envelope fields from method identity."""

    if not isinstance(identity, dict) or not isinstance(identity.get("benchmark_protocol"), dict):
        raise ValueError("benchmark identity is malformed")
    protocol = dict(identity["benchmark_protocol"])
    envelope = {}
    for key in RESOURCE_ENVELOPE_KEYS:
        value = float(protocol.pop(key, math.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"benchmark identity has invalid {key}")
        envelope[key] = value
    method = dict(identity)
    method["benchmark_protocol"] = protocol
    return method, envelope


def _read_stage(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        stage = pickle.load(handle)
    if not isinstance(stage, dict) or set(stage) != {"identity", "stage", "result"}:
        raise ValueError(f"malformed StageStore artifact: {path}")
    if stage["stage"] != STAGE_NAME:
        raise ValueError(f"comparison artifact is not the {STAGE_NAME} stage")
    result = stage["result"]
    if not isinstance(result, dict) or not {"loss", "gradient", "parameter_delta"} <= set(result):
        raise ValueError(f"first-update result is incomplete: {path}")
    return stage


def compare_physical_microbatches(
    report8: Mapping[str, Any],
    stage8: Mapping[str, Any],
    report16: Mapping[str, Any],
    stage16: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare complete-update equivalence and select the faster feasible mode."""

    checks = {"8": _validate_probe(report8, microbatch_size=8), "16": _validate_probe(report16, microbatch_size=16)}
    method8, envelope8 = _split_identity(report8["identity"])
    method16, envelope16 = _split_identity(report16["identity"])
    stage_identities_match_reports = bool(
        stage8.get("identity") == report8["identity"] and stage16.get("identity") == report16["identity"]
    )
    method_identity_matches = method8 == method16
    identity_matches = bool(stage_identities_match_reports and method_identity_matches)
    envelope_differences = {
        key: {
            "microbatch_8": envelope8[key],
            "microbatch_16": envelope16[key],
            "changed": envelope8[key] != envelope16[key],
        }
        for key in RESOURCE_ENVELOPE_KEYS
    }
    source_git_head_matches = report8.get("source_git_head") == report16.get("source_git_head")
    stage_names_match = stage8.get("stage") == STAGE_NAME and stage16.get("stage") == STAGE_NAME
    first8, first16 = stage8["result"], stage16["result"]
    first_loss = _loss_comparison(float(first8["loss"]), float(first16["loss"]))
    gradient = compare_numeric_trees(first8["gradient"], first16["gradient"], tolerance=TOLERANCES["gradient"])
    parameter_delta = compare_numeric_trees(
        first8["parameter_delta"], first16["parameter_delta"], tolerance=TOLERANCES["parameter_delta"]
    )
    timed_losses = [
        {"update": index + 1, **_loss_comparison(float(left["loss"]), float(right["loss"]))}
        for index, (left, right) in enumerate(zip(report8["updates"], report16["updates"], strict=True))
    ]
    equivalence_passed = bool(
        identity_matches
        and source_git_head_matches
        and stage_names_match
        and first_loss["passed"]
        and gradient["passed"]
        and parameter_delta["passed"]
        and all(row["passed"] for row in timed_losses)
    )
    both_feasible = checks["8"]["passed"] and checks["16"]["passed"]
    passed = bool(equivalence_passed and both_feasible)
    selected = None
    faster = min(checks, key=lambda size: checks[size]["median_compute_seconds"])
    slower = "16" if faster == "8" else "8"
    speedup = checks[slower]["median_compute_seconds"] / checks[faster]["median_compute_seconds"]
    speed_gain_fraction = 1.0 - 1.0 / speedup
    selection_basis = "no recommendation because equivalence or feasibility failed"
    if passed:
        if speed_gain_fraction >= MINIMUM_SPEED_GAIN_FRACTION:
            selected = int(faster)
            selection_basis = "at least 5% lower median complete-update time"
        else:
            selected = min((8, 16), key=lambda size: (checks[str(size)]["peak_bytes_in_use"], size))
            selection_basis = "median speed gain below 5%; lower benchmark peak memory preferred"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "rejected",
        "comparison_scope": "microbatch_8_vs_16_same_batch64_physical_v2",
        "microbatch4_baseline_measured": False,
        "scientific_acceptance_claim": False,
        "tolerances_declared_before_results": TOLERANCES,
        "probe_checks": checks,
        "identity_matches": identity_matches,
        "identity_comparison": {
            "passed": identity_matches,
            "stage_identities_match_own_reports": stage_identities_match_reports,
            "all_non_resource_envelope_fields_match": method_identity_matches,
            "model_data_objective_identity_matches": method_identity_matches,
            "permitted_cross_run_difference_keys": list(RESOURCE_ENVELOPE_KEYS),
            "resource_envelope_differences": envelope_differences,
            "model_or_loss_change_inferred": False if method_identity_matches else None,
        },
        "source_git_head_matches": source_git_head_matches,
        "first_update": {
            "loss": first_loss,
            "gradient": gradient,
            "parameter_delta": parameter_delta,
            "passed": bool(first_loss["passed"] and gradient["passed"] and parameter_delta["passed"]),
        },
        "timed_update_losses": timed_losses,
        "equivalence_passed": equivalence_passed,
        "both_modes_feasible": both_feasible,
        "performance": {
            "faster_observed_microbatch_size": int(faster),
            "observed_median_compute_speedup": speedup,
            "observed_speed_gain_fraction": speed_gain_fraction,
            "minimum_speed_gain_fraction_for_speed_selection": MINIMUM_SPEED_GAIN_FRACTION,
            "microbatch_8_peak_bytes": checks["8"]["peak_bytes_in_use"],
            "microbatch_16_peak_bytes": checks["16"]["peak_bytes_in_use"],
        },
        "recommended_microbatch_size": selected,
        "recommendation_basis": selection_basis,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--microbatch-8-report", type=Path, required=True)
    result.add_argument("--microbatch-16-report", type=Path, required=True)
    result.add_argument("--microbatch-8-stage", type=Path)
    result.add_argument("--microbatch-16-stage", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def run(args: Any) -> Mapping[str, Any]:
    report_paths = {8: args.microbatch_8_report.resolve(), 16: args.microbatch_16_report.resolve()}
    stage_paths = {
        8: (args.microbatch_8_stage or report_paths[8].with_suffix(".stages") / "first_update.pkl").resolve(),
        16: (args.microbatch_16_stage or report_paths[16].with_suffix(".stages") / "first_update.pkl").resolve(),
    }
    reports = {size: json.loads(path.read_text(encoding="utf-8")) for size, path in report_paths.items()}
    stages = {size: _read_stage(path) for size, path in stage_paths.items()}
    comparison = compare_physical_microbatches(reports[8], stages[8], reports[16], stages[16])
    comparison["inputs"] = {
        "microbatch_8_report": str(report_paths[8]),
        "microbatch_16_report": str(report_paths[16]),
        "microbatch_8_stage": str(stage_paths[8]),
        "microbatch_16_stage": str(stage_paths[16]),
    }
    _atomic_json(args.output.resolve(), comparison)
    return comparison


if __name__ == "__main__":
    run(parser().parse_args())
