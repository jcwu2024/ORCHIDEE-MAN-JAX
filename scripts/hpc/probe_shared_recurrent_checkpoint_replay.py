"""Replay saved dual-gradient T7/T30 checkpoint executables without compilation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

RTOL, ATOL = 1e-5, 1e-6
MEMORY_LIMIT = 24 * 2**30


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _validate_assets(args, reference_report, candidate_report, reference_saved, candidate_saved,
                     bundle, checkpoint, profile_sha256, jax_version):
    identity = reference_report.get("identity")
    if (not isinstance(identity, dict) or candidate_report.get("identity") != identity
            or reference_saved.get("identity") != identity
            or candidate_saved.get("identity") != identity):
        raise ValueError("reference/candidate saved identities differ")
    if (reference_report.get("status") != "passed"
            or reference_report.get("stage") != "completed"
            or reference_report.get("mode") != "reference"
            or not reference_report.get("initial_state_gradient_finite")
            or any(reference_report.get("hard_counts", {}).values())
            or not all(row.get("passed") for row in reference_report.get("comparisons", {}).values())
            or set(reference_report.get("comparisons", {}))
            != {"loss_and_auxiliary", "all_parameter_gradients"}):
        raise ValueError("saved default-policy reference lacks full admission")
    if (candidate_report.get("mode") != "candidate"
            or not candidate_report.get("parameter_and_initial_state_gradients")
            or candidate_report.get("candidate_policy") != reference_report.get("candidate_policy")
            or not candidate_report.get("candidate_result")
            or not candidate_report.get("candidate_executable")):
        raise ValueError("candidate report lacks saved full dual-gradient artifacts")
    if (Path(reference_report.get("reference_result", "")).resolve() != args.reference_result.resolve()
            or Path(reference_report.get("reference_executable", "")).resolve()
            != args.reference_executable.resolve()
            or Path(candidate_report["candidate_result"]).resolve() != args.candidate_result.resolve()
            or Path(candidate_report["candidate_executable"]).resolve()
            != args.candidate_executable.resolve()):
        raise ValueError("CLI artifact paths differ from their producing reports")
    if (identity.get("horizon") != args.horizon
            or reference_report.get("horizon") != args.horizon
            or candidate_report.get("horizon") != args.horizon
            or identity.get("source_profile_sha256") != profile_sha256
            or identity.get("jax_version") != jax_version
            or identity.get("method") != checkpoint.get("method")
            or bundle.get("method") != identity.get("method")
            or bundle.get("profile_sha256") != profile_sha256
            or bundle.get("source_git_head") != identity.get("source_head")
            or int(bundle["artifact_ids"][0]) != identity.get("artifact_id")
            or Path(bundle["initial_checkpoint"]).resolve() != args.checkpoint.resolve()
            or Path(identity.get("checkpoint", "")).resolve() != args.checkpoint.resolve()
            or int(checkpoint.get("completed_updates", -1)) != 0):
        raise ValueError("profile, bundle, checkpoint, artifact or horizon differs")
    case = bundle["payloads"][0]["cases"].get(args.horizon)
    if (case is None or len(case) != 5
            or np.shape(case.days[0].continuous_state)[0] != args.horizon):
        raise ValueError("saved recurrent case has wrong horizon")
    return case, identity


def _hard(result):
    return {name: int(np.asarray(value)) for name, value in result[0][1]["hard"].items()}


def _checked_comparison(jax, actual, expected, compare_trees):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    if actual_tree != expected_tree:
        raise ValueError("replay result tree structure differs")
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_array, expected_array = np.asarray(actual_leaf), np.asarray(expected_leaf)
        if actual_array.shape != expected_array.shape or actual_array.dtype != expected_array.dtype:
            raise ValueError("replay result shape or dtype differs")
        if not np.all(np.isfinite(actual_array)) or not np.all(np.isfinite(expected_array)):
            raise ValueError("replay result contains nonfinite values")
    return compare_trees(jax, actual, expected, rtol=RTOL, atol=ATOL)


def _require_physical_zero(actual, expected=None):
    hard = _hard(actual)
    if any(hard.values()) or (expected is not None and hard != _hard(expected)):
        raise ValueError("replay result has nonzero or changed physical hard counts")
    return hard


def _require_memory_peak(memory):
    peak = (memory or {}).get("peak_bytes_in_use")
    if peak is None or int(peak) > MEMORY_LIMIT:
        raise MemoryError("observed device peak is absent or exceeds 24 GiB")


def _orders(repeats: int):
    if repeats < 3:
        raise ValueError("at least three matched pairs are required")
    return [("reference", "candidate") if index % 2 == 0
            else ("candidate", "reference") for index in range(repeats)]


def run(args, report):
    import jax
    from jax.experimental.serialize_executable import deserialize_and_load

    from scripts.hpc.probe_shared_execution import compare_trees
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.run_shared_transformer_admission import memory_sizes

    if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu":
        raise ValueError("replay requires exactly one visible GPU")
    profile_sha256 = hashlib.sha256(args.source_profile.read_bytes()).hexdigest()
    reference_report = json.loads(args.reference_report.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    reference_saved = _read_pickle(args.reference_result)
    candidate_saved = _read_pickle(args.candidate_result)
    bundle = _read_pickle(args.input_bundle)
    checkpoint = load_learning_checkpoint(
        args.checkpoint, expected_method=reference_report["identity"]["method"])
    case, identity = _validate_assets(
        args, reference_report, candidate_report, reference_saved, candidate_saved,
        bundle, checkpoint, profile_sha256, jax.__version__)
    report.update(stage="assets_validated", identity={
        key: identity[key] for key in ("source_head", "source_profile_sha256", "artifact_id",
                                      "horizon", "checkpoint", "jax_version")},
        reference_report=str(args.reference_report), candidate_report=str(args.candidate_report),
        reference_result=str(args.reference_result), candidate_result=str(args.candidate_result),
        reference_executable=str(args.reference_executable),
        candidate_executable=str(args.candidate_executable))
    _write(args.output, report)

    old, new = reference_saved["result"], candidate_saved["result"]
    if (len(old) != 2 or len(new) != 2 or len(old[1]) != 2 or len(new[1]) != 2):
        raise ValueError("saved results are not full dual-gradient outputs")
    comparisons = {
        "loss_and_auxiliary": _checked_comparison(jax, new[0], old[0], compare_trees),
        "all_parameter_gradients": _checked_comparison(
            jax, new[1][0], old[1][0], compare_trees),
        "initial_state_gradient": _checked_comparison(
            jax, new[1][1], old[1][1], compare_trees),
    }
    hard = {"reference": _require_physical_zero(old),
            "candidate": _require_physical_zero(new, old)}
    parity_passed = all(row["passed"] for row in comparisons.values())
    report.update(stage="saved_results_compared", comparisons=comparisons,
                  hard_counts=hard, physical_passed=True,
                  strict_parity_passed=parity_passed, production_adoption=False,
                  scientific_acceptance=False, retained_updates=0)
    _write(args.output, report)

    device = jax.local_devices()[0]
    call_args = (jax.device_put(checkpoint["parameters"]), *jax.device_put(case))
    jax.block_until_ready(call_args)
    executables = {}
    load_seconds = {}
    compiled_memory = {}
    for name, path in (("reference", args.reference_executable),
                       ("candidate", args.candidate_executable)):
        started = time.perf_counter()
        executable = deserialize_and_load(*_read_pickle(path))
        load_seconds[name] = time.perf_counter() - started
        compiled_memory[name] = memory_sizes(executable)
        if compiled_memory[name]["estimated_live_bytes"] > MEMORY_LIMIT:
            raise MemoryError(f"{name} saved executable exceeds 24 GiB estimated live")
        executables[name] = executable
    report.update(stage="executables_loaded", load_seconds=load_seconds,
                  compiled_memory=compiled_memory)
    _write(args.output, report)

    saved_results = {"reference": old, "candidate": new}
    warm_seconds = {}
    replay_comparisons = {}
    for name in ("reference", "candidate"):
        started = time.perf_counter()
        result = executables[name](*call_args)
        jax.block_until_ready(result)
        warm_seconds[name] = time.perf_counter() - started
        host_result = jax.device_get(result)
        _require_physical_zero(host_result, saved_results[name])
        replay_comparisons[name] = _checked_comparison(
            jax, host_result, saved_results[name], compare_trees)
        del result
    report.update(stage="warmed", warm_seconds=warm_seconds,
                  replay_comparisons=replay_comparisons,
                  replay_strict_stable=all(row["passed"] for row in replay_comparisons.values()),
                  device_memory_after_warm=device.memory_stats())
    _write(args.output, report)
    _require_memory_peak(report["device_memory_after_warm"])

    times = {"reference": [], "candidate": []}
    pairs = []
    for order in _orders(args.repeats):
        pair = {"order": list(order)}
        results = {}
        for name in order:
            jax.block_until_ready(call_args)
            started = time.perf_counter()
            result = executables[name](*call_args)
            jax.block_until_ready(result)
            seconds = time.perf_counter() - started
            times[name].append(seconds)
            pair[name + "_seconds"] = seconds
            results[name] = result
        pair["device_memory"] = device.memory_stats()
        _require_memory_peak(pair["device_memory"])
        pair["saved_result_comparisons"] = {}
        pair["hard_counts"] = {}
        for name in ("reference", "candidate"):
            host_result = jax.device_get(results[name])
            pair["hard_counts"][name] = _require_physical_zero(
                host_result, saved_results[name])
            pair["saved_result_comparisons"][name] = _checked_comparison(
                jax, host_result, saved_results[name], compare_trees)
        pair["replay_strict_stable"] = all(
            row["passed"] for row in pair["saved_result_comparisons"].values())
        pairs.append(pair)
        report.update(stage="timing", pairs=pairs,
                      replay_strict_stable=(report["replay_strict_stable"]
                                            and all(row["replay_strict_stable"] for row in pairs)))
        _write(args.output, report)
        del results
    medians = {name: float(np.median(values)) for name, values in times.items()}
    report.update(stage="completed", pairs=pairs,
                  median_seconds=medians,
                  reference_over_candidate_ratio=medians["reference"] / medians["candidate"],
                  device_memory_after_pairs=device.memory_stats(),
                  memory_peak_semantics="process cumulative; not isolated by executable",
                  strict_parity_passed=parity_passed,
                  admitted=False)
    _write(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=(7, 30), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    for name in ("source_profile", "input_bundle", "checkpoint", "reference_report",
                 "reference_result", "reference_executable", "candidate_report",
                 "candidate_result", "candidate_executable", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    paths = [args.output, args.source_profile, args.input_bundle, args.checkpoint,
             args.reference_report, args.reference_result, args.reference_executable,
             args.candidate_report, args.candidate_result, args.candidate_executable]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("output and input artifact paths must all differ")
    report = {"status": "diagnostic_only", "stage": "starting", "scope": "saved_dual_gradient_replay",
              "horizon": args.horizon, "repeats": args.repeats,
              "comparison_rtol": RTOL, "comparison_atol": ATOL,
              "memory_limit_bytes": MEMORY_LIMIT, "admitted": False}
    _write(args.output, report)
    try:
        run(args, report)
    except Exception as error:
        report.update(stage="error", error_type=type(error).__name__, error=str(error),
                      admitted=False)
        _write(args.output, report)
        raise


if __name__ == "__main__":
    main()
