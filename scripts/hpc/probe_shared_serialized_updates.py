"""Replay two unchanged technical updates using trusted saved executables.

No model graph is lowered here. The original learning runner still owns data,
fresh initialization, physical checks, gradient composition and AdamW updates.
This bounded probe requires the same source checkout and method as its saved
reference; it is not a general cross-version executable cache.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from scripts.hpc.probe_shared_execution import compare_trees
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json


def require_reference(profile, reference):
    if profile.get("status") != "passed" or set(profile.get("executables", {})) != {"daily", "T7", "T30"}:
        raise ValueError("requires a completed three-executable profile")
    if reference.get("status") != "passed" or reference.get("completed_updates") != 2:
        raise ValueError("requires a completed two-update reference")
    if reference.get("technical_updates") != 2:
        raise ValueError("reference must be a technical run, not formal learning")


def require_same_method(current, reference):
    if current["method"] != reference["method"]:
        raise ValueError("saved executables belong to a different method/data/initialization")
    if current["source_git_head"] != reference["source_git_head"]:
        raise ValueError("bounded serialized replay requires the original source checkout")


def load_evaluators(profile, *, deserialize, memory_sizes, memory_limit_bytes, emit):
    """Load one existing executable per component, without a compile fallback."""
    evaluators, metadata = {}, {}
    for name in ("daily", "T7", "T30"):
        emit("load_saved_gradient", component=name)
        began = time.perf_counter()
        cpu_began = time.process_time()
        path = Path(profile["executables"][name]["executable"])
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        read_seconds = time.perf_counter() - began
        read_cpu_seconds = time.process_time() - cpu_began
        emit("deserialize_saved_gradient", component=name,
             executable_read_seconds=read_seconds,
             executable_read_cpu_seconds=read_cpu_seconds)
        began = time.perf_counter()
        cpu_began = time.process_time()
        executable = deserialize(*payload)
        load_seconds = time.perf_counter() - began
        load_cpu_seconds = time.process_time() - cpu_began
        del payload
        memory = memory_sizes(executable)
        if memory["estimated_live_bytes"] > memory_limit_bytes:
            raise ValueError(f"saved {name} executable exceeds declared memory limit")
        evaluators[name] = executable
        metadata[name] = {
            "source_executable": str(path),
            "read_seconds": read_seconds,
            "read_cpu_seconds": read_cpu_seconds,
            "deserialize_seconds": load_seconds,
            "deserialize_cpu_seconds": load_cpu_seconds,
            "memory": memory,
            "cache_policy": "trusted_serialized_executable_no_trace_or_lower",
        }
        emit("saved_gradient_loaded", component=name, compiled_gradient_graphs=metadata)
    return evaluators, metadata


def numerical_update_rows(outcome):
    return [
        {key: row[key] for key in (
            "update", "artifact_id", "loss", "gradient_norms", "optimizer_gradient_norm", "clip_scale"
        )}
        for row in outcome["updates"]
    ]


def compare_update_reports(jax, current, reference, *, rtol, atol):
    # JSON does not retain NumPy scalar dtypes. Compare scalar record values
    # in one host dtype, while checkpoint tensor comparisons remain strict.
    def normalize(rows):
        return jax.tree_util.tree_map(lambda value: np.asarray(value, dtype=np.float64), rows)
    return compare_trees(
        jax, normalize(numerical_update_rows(current)), normalize(numerical_update_rows(reference)),
        rtol=rtol, atol=atol,
    )


def run(args):
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax

    if args.technical_updates != 2 or args.prepare_only or args.resume_checkpoint:
        raise ValueError("serialized replay requires fresh --technical-updates 2")
    profile = json.loads(args.source_profile.read_text(encoding="utf-8"))
    reference = json.loads(Path(profile["technical_update_report"]).read_text(encoding="utf-8"))
    require_reference(profile, reference)
    output = args.output
    update_output = output.with_suffix(".updates.json")
    if update_output.resolve() == Path(profile["technical_update_report"]).resolve():
        raise ValueError("replay output must preserve the reference report/checkpoints")
    report = {
        "status": "running", "scope": "same_method_serialized_two_update_replay",
        "source_profile": str(args.source_profile), "scientific_acceptance": False,
        "comparison_rtol": 1e-5, "comparison_atol": 1e-6,
        "comparison_tolerance_source": "shared_execution_benchmark_v1.json",
    }
    started = time.perf_counter()
    original_compile = learning.compile_gradient_evaluators
    context = {}

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(output, report)
        print(f"serialized_updates stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    def load_instead_of_compile(**kwargs):
        jax = kwargs["jax"]
        _require_supported_jax(jax)
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise ValueError("select one visible GPU matching saved executable layout")
        if kwargs["daily_microbatch_size"] != 4:
            raise ValueError("saved profile is daily micro4")
        current = json.loads(update_output.read_text(encoding="utf-8"))
        require_same_method(current, reference)
        from jax.experimental.serialize_executable import deserialize_and_load

        context["jax"] = jax
        evaluators, metadata = load_evaluators(
            profile, deserialize=deserialize_and_load, memory_sizes=learning.memory_sizes,
            memory_limit_bytes=kwargs["memory_limit_bytes"], emit=emit,
        )
        emit("gradients_ready", gradient_ready_seconds=time.perf_counter() - started)
        return evaluators, metadata

    learning.compile_gradient_evaluators = load_instead_of_compile
    args.output = update_output
    try:
        emit("run_unchanged_training")
        outcome = learning.run(args)
        report["technical_update_report"] = str(update_output)
        if outcome["status"] != "passed":
            emit("rejected", status="rejected", reason="technical_updates_failed")
            return report
        emit("compare_reference", update_run_seconds=time.perf_counter() - started)
        jax = context["jax"]
        saved = learning.load_learning_checkpoint(Path(reference["checkpoint"]), expected_method=reference["method"])
        replay = learning.load_learning_checkpoint(Path(outcome["checkpoint"]), expected_method=reference["method"])
        tolerance = dict(rtol=report["comparison_rtol"], atol=report["comparison_atol"])
        comparisons = {
            "update_losses_gradient_norms": compare_update_reports(jax, outcome, reference, **tolerance),
            "all_parameters": compare_trees(jax, replay["parameters"], saved["parameters"], **tolerance),
            "full_adamw_state": compare_trees(jax, replay["optimizer"], saved["optimizer"], **tolerance),
        }
        emit(
            "completed", status="passed" if all(row["passed"] for row in comparisons.values()) else "rejected",
            comparisons=comparisons, completed_updates=outcome["completed_updates"],
            warm_second_update_wall_seconds=outcome["warm_second_update_wall_seconds"],
            device_peak_bytes_in_use=outcome["device_peak_bytes_in_use"],
            checkpoint_selection="diagnostic_replay_not_formal_training",
        )
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        learning.compile_gradient_evaluators = original_compile
        args.output = output


def parser():
    from scripts.hpc.run_shared_learning import parser as base_parser

    result = base_parser()
    result.add_argument("--source-profile", type=Path, required=True)
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    raise SystemExit(0 if result["status"] == "passed" else 1)
