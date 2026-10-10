"""Compare exact daily pytree slicing with the saved micro4 gradient executable."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load diagnostic module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _tree_max_error(jax: Any, left: Any, right: Any) -> dict[str, Any]:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure or len(left_leaves) != len(right_leaves):
        raise ValueError("slice variants returned different pytree structures")
    max_abs = 0.0
    max_rel = 0.0
    elements = 0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        a, b = np.asarray(left_leaf), np.asarray(right_leaf)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("slice variants returned different leaf shape/dtype")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("slice variants returned nonfinite values")
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        denominator = np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
        max_abs = max(max_abs, float(delta.max(initial=0.0)))
        max_rel = max(max_rel, float((delta / np.maximum(denominator, 1.0e-30)).max(initial=0.0)))
        elements += int(a.size)
    return {"leaf_count": len(left_leaves), "element_count": elements, "max_abs": max_abs, "max_rel": max_rel}


def _tree_assert_close(jax: Any, left: Any, right: Any) -> dict[str, Any]:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure:
        raise ValueError("slice variants returned different pytree structures")
    for a, b in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(a, b, rtol=1.0e-5, atol=1.0e-6)
    return _tree_max_error(jax, left, right)


def _synchronized(jax: Any, function: Any, parameters: Any, batch: Any) -> tuple[Any, float]:
    began = time.perf_counter()
    result = function(parameters, batch)
    jax.block_until_ready(result)
    return result, time.perf_counter() - began


def _daily_batch(jax: Any, first: Any, second: Any | None = None) -> Any:
    if second is None:
        return first
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    if first_tree != second_tree:
        raise ValueError("the two original compact daily payloads differ structurally")
    return jax.tree_util.tree_unflatten(
        first_tree,
        [jax.numpy.concatenate((a, b), axis=0) for a, b in zip(first_leaves, second_leaves, strict=True)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro4-probe", type=Path, required=True)
    parser.add_argument("--slice-helper-file", type=Path, required=True)
    parser.add_argument("--real-daily-probe-file", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--prepared-context", type=Path, required=True)
    parser.add_argument("--baseline-admission-report", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--saved-executable", type=Path, required=True)
    parser.add_argument("--runtime-preflight-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists() or args.output.with_suffix(".full.pkl").exists():
        raise ValueError("output exists; preserve old assets and choose a new stem")
    report: dict[str, Any] = {
        "schema": "shared_daily_slice_execution_v1",
        "status": "running",
        "comparison": "daily_execution_only",
        "rtol": 1.0e-5,
        "atol": 1.0e-6,
        "runs": {},
    }

    def emit(stage: str, **fields: Any) -> None:
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
        print(f"daily_slice {stage}", flush=True)

    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental.serialize_executable import deserialize_and_load

        from scripts.hpc.run_shared_learning import load_learning_checkpoint
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        micro4 = _module(args.micro4_probe, "shared_slice_micro4_probe")
        slice_helper = _module(args.slice_helper_file, "shared_daily_slice_helper")
        helper = micro4.load_owner(args.real_daily_probe_file)
        if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu":
            raise ValueError("requires one visible GPU for the admitted saved executable")
        runtime_preflight = micro4.validate_runtime_report(args.runtime_preflight_report, jax.__version__)
        admission = json.loads(args.baseline_admission_report.read_text(encoding="utf-8"))
        if (
            admission.get("status") != "passed"
            or admission.get("union_admission", {}).get("gradient_coordinates") != 64_884_481
            or admission.get("comparisons", {}).get("all_parameter_gradients", {}).get("passed") is not True
        ):
            raise ValueError("completed micro4 report lacks full-gradient admission")
        if Path(admission.get("baseline_binary", "")).resolve() != args.saved_executable.resolve():
            raise ValueError("saved field executable differs from completed micro4 baseline binary")
        with args.reference_result.open("rb") as stream:
            saved_four = pickle.load(stream)
        if saved_four.get("variant") != micro4.VARIANT or "diagnostic_sample" in saved_four.get("identity", {}):
            raise ValueError("four-row result is not the admitted original-four candidate")
        reference = saved_four["result"]
        reference_provenance = {
            "artifact": str(args.reference_result.resolve()),
            "variant": saved_four["variant"],
            "identity": saved_four["identity"],
        }
        profile, method_reference = helper.trusted_profile(args)
        identity = helper.context_identity(method_reference, helper.initial_checkpoint(method_reference), 742)
        saved_identity = saved_four.get("identity", {})
        if {key: value for key, value in saved_identity.items() if key in identity} != identity:
            raise ValueError("admitted result identity differs from trusted artifact742 method")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != identity["source_git_head"]:
            raise ValueError("immutable source checkout differs from the saved micro4 method")
        with args.prepared_context.open("rb") as stream:
            context = pickle.load(stream)
        if context.get("identity") != identity:
            raise ValueError("prepared context identity differs from original artifact742")
        with args.input_bundle.open("rb") as stream:
            bundle = pickle.load(stream)
        if (
            bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
            or bundle.get("method") != method_reference["method"]
            or tuple(bundle.get("artifact_ids", ())) != tuple(method_reference["method"]["training_plan"]["training_artifact_ids"][:2])
            or len(bundle.get("payloads", ())) != 2
        ):
            raise ValueError("expected original compact artifact742/275 daily payloads")
        rows4 = micro4.validate_original_rows(jax, context["daily4"], bundle["payloads"][0]["daily"])
        if saved_identity.get("original_rows") != rows4:
            raise ValueError("saved field replay does not contain the declared original four rows")
        daily32 = bundle["payloads"][0]["daily"]
        daily64 = _daily_batch(jax, daily32, bundle["payloads"][1]["daily"])
        if int(jax.tree_util.tree_leaves(daily32)[0].shape[0]) != 32 or int(jax.tree_util.tree_leaves(daily64)[0].shape[0]) != 64:
            raise ValueError("original daily payload sizes are not 32 and 64")
        checkpoint = load_learning_checkpoint(
            Path(identity["initial_checkpoint"]), expected_method=method_reference["method"]
        )
        parameters = jax.device_put(checkpoint["parameters"], jax.local_devices()[0])
        daily32, daily64 = jax.device_put((daily32, daily64), jax.local_devices()[0])
        with args.saved_executable.open("rb") as stream:
            serialized = pickle.load(stream)
        evaluator = deserialize_and_load(*serialized)
        micro4.validate_binary_inputs(jax, serialized, evaluator, parameters, context["daily4"])
        saved_four_actual = evaluator(parameters, jax.device_put(context["daily4"], jax.local_devices()[0]))
        jax.block_until_ready(saved_four_actual)
        first_four_parity = _tree_assert_close(jax, saved_four_actual, reference)
        del saved_four_actual
        if memory_sizes(evaluator)["estimated_live_bytes"] > 24 * 2**30:
            raise MemoryError("saved evaluator exceeds the unchanged 24 GiB estimate")
        parameter_leaves = jax.tree_util.tree_leaves(parameters)
        if len(parameter_leaves) != 300 or sum(np.size(value) for value in parameter_leaves) != 64_884_481:
            raise ValueError("update0 parameters differ from the admitted full model")
        jax.block_until_ready(parameters)
        report.update(
            identity=identity,
            rows4=rows4,
            daily_batch_rows={"32": 32, "64": 64},
            sample_leaf_count=len(jax.tree_util.tree_leaves(daily32)),
            parameter_leaves=len(parameter_leaves),
            parameter_coordinates=sum(np.size(value) for value in parameter_leaves),
            saved_executable=str(args.saved_executable.resolve()),
            saved_four_row_reference_provenance=reference_provenance,
            runtime_preflight=runtime_preflight,
            estimated_evaluator_memory=memory_sizes(evaluator),
            model_gradient_compiles=0,
            auxiliary_and_full_gradient_checked=True,
            first_four_saved_result_parity=first_four_parity,
            startup_and_reference_load_seconds=time.perf_counter() - started,
        )
        results: dict[str, Any] = {}
        for count, batch in ((32, daily32), (64, daily64)):
            tree_slicer = slice_helper.make_daily_batch_slicer(jax, 4)
            slice_started = time.perf_counter()
            tree_slicer = tree_slicer.lower(batch, jnp.asarray(0, dtype=jnp.int32)).compile()
            slice_compile_seconds = time.perf_counter() - slice_started
            functions = {
                mode: slice_helper.make_daily_accumulator(
                    jax=jax,
                    jnp=jnp,
                    microbatch_size=4,
                    evaluator=evaluator,
                    expected_count=count,
                    slice_execution=mode,
                    _batch_slicer=tree_slicer if mode == "tree" else None,
                )
                for mode in ("leaf", "tree")
            }
            warm = {}
            for mode in ("leaf", "tree"):
                value, elapsed = _synchronized(jax, functions[mode], parameters, batch)
                warm[mode] = {"seconds": elapsed, "result": value}
                emit("explicit_warmup", count=count, mode=mode, seconds=elapsed)
            warm_parity = _tree_assert_close(jax, warm["leaf"]["result"], warm["tree"]["result"])
            timings = {"leaf": [], "tree": []}
            pair_records = []
            latest = {"leaf": warm["leaf"]["result"], "tree": warm["tree"]["result"]}
            for repeat in range(3):
                order = ("leaf", "tree") if repeat % 2 == 0 else ("tree", "leaf")
                pair_seconds = {}
                for mode in order:
                    result, elapsed = _synchronized(jax, functions[mode], parameters, batch)
                    timings[mode].append(elapsed)
                    pair_seconds[mode] = elapsed
                    latest[mode] = result
                errors = _tree_assert_close(jax, latest["leaf"], latest["tree"])
                pair_record = {
                    "repeat": repeat + 1,
                    "order": list(order),
                    "timings_seconds": pair_seconds,
                    "parity_error_summary": errors,
                    "parity_passed": True,
                }
                pair_records.append(pair_record)
                report["runs"][str(count)] = {
                    "tree_slicer_compile_seconds": slice_compile_seconds,
                    "warmup_seconds": {mode: warm[mode]["seconds"] for mode in ("leaf", "tree")},
                    "warmup_parity": warm_parity,
                    "matched_pairs": pair_records,
                    "timings_seconds": timings,
                    "full_result_in_pickle": True,
                }
                emit("matched_pair", count=count, **pair_record)
                device_stats = jax.local_devices()[0].memory_stats() or {}
                if device_stats.get("peak_bytes_in_use", 0) > 24 * 2**30:
                    raise MemoryError("observed active-buffer peak exceeds unchanged 24 GiB admission")
            hard_counts = {
                mode: {
                    key: int(np.asarray(value))
                    for key, value in latest[mode][0][1]["hard"].items()
                }
                for mode in ("leaf", "tree")
            }
            if any(value != 0 for mode in hard_counts.values() for value in mode.values()):
                raise ValueError("daily physical hard-count gate failed")
            medians = {mode: statistics.median(values) for mode, values in timings.items()}
            results[str(count)] = {
                "tree_slicer_compile_seconds": slice_compile_seconds,
                "warmup_seconds": {mode: warm[mode]["seconds"] for mode in ("leaf", "tree")},
                "warmup_parity": warm_parity,
                "matched_pairs": pair_records,
                "timings_seconds": timings,
                "median_seconds": medians,
                "leaf_over_tree_ratio": medians["leaf"] / medians["tree"],
                "hard_counts": hard_counts,
                "peak_device_memory_bytes": device_stats.get("peak_bytes_in_use"),
                "result": latest,
            }
            report["runs"][str(count)] = {
                "tree_slicer_compile_seconds": slice_compile_seconds,
                "warmup_seconds": {mode: warm[mode]["seconds"] for mode in ("leaf", "tree")},
                "warmup_parity": warm_parity,
                "matched_pairs": pair_records,
                "timings_seconds": timings,
                "median_seconds": medians,
                "leaf_over_tree_ratio": medians["leaf"] / medians["tree"],
                "hard_counts": hard_counts,
                "peak_device_memory_bytes": device_stats.get("peak_bytes_in_use"),
                "full_result_in_pickle": True,
            }
            emit("batch_complete", count=count, median_seconds=medians, leaf_over_tree_ratio=medians["leaf"] / medians["tree"])
            del functions, warm, latest
        full_result = {
            "schema": "shared_daily_slice_execution_full_v1",
            "identity": identity,
            "rtol": 1.0e-5,
            "atol": 1.0e-6,
            "batch_results": results,
            "four_row_reference": reference,
        }
        _save_pickle(args.output.with_suffix(".full.pkl"), full_result)
        report["status"] = "passed"
        report["timing_scope"] = "daily_slicing_only_not_full_update_or_T7_T30_throughput"
        emit("passed", full_result_artifact=str(args.output.with_suffix(".full.pkl").resolve()))
    except Exception as exc:
        emit("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
