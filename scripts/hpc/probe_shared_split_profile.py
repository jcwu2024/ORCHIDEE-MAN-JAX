"""Profile admitted split executables without permitting another cold compile.

The unchanged learning runner constructs and loads its three cached graphs.
After its two technical updates, fixed-weight replays measure accumulation and
device kernels. Saved executables support later profiling without retracing.
No result from this diagnostic is a formal learning checkpoint.
"""
from __future__ import annotations

import gzip
import pickle
import time

import numpy as np

from scripts.hpc.probe_shared_split_timing_cache import observe_compiler
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json


def benchmark_replays(invoke, synchronize, repeats):
    """Exclude one warmup and block on each timed result; never update weights."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    synchronize(invoke())
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        synchronize(invoke())
        seconds.append(time.perf_counter() - started)
    return {"seconds": seconds, "median_seconds": float(np.median(seconds))}


def concatenate_daily(jax, left, right):
    """Join two real artifact batches on the host, preserving every sample leaf."""
    return jax.tree_util.tree_map(
        lambda a, b: np.concatenate((np.asarray(a), np.asarray(b)), axis=0), left, right
    )


def save_executable(directory, name, executable, serialize):
    """Consolidate each executable and its trees into one local trusted pickle."""
    directory.mkdir(parents=True, exist_ok=True)
    binary_path = directory / f"{name}.executable.pkl"
    with binary_path.open("wb") as handle:
        pickle.dump(serialize(executable), handle, protocol=pickle.HIGHEST_PROTOCOL)
    hlo = executable.as_text()
    hlo_path = directory / f"{name}.optimized-hlo.txt.gz"
    with gzip.open(hlo_path, "wt", encoding="utf-8") as handle:
        handle.write(hlo)
    return {
        "executable": str(binary_path),
        "serialized_bytes": binary_path.stat().st_size,
        "optimized_hlo": str(hlo_path),
        "optimized_hlo_text_bytes": len(hlo.encode("utf-8")),
        "optimized_hlo_lines": hlo.count("\n"),
    }


def run(args):
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax

    if args.technical_updates != 2 or args.prepare_only or args.resume_checkpoint:
        raise ValueError("profile requires fresh --technical-updates 2")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    output = args.output
    assets = output.with_suffix(".assets")
    report = {
        "status": "running",
        "scope": "fixed_weight_cache_reused_execution_profile_not_learning",
        "scientific_acceptance": False,
        "cache_events": [],
        "executables": {},
        "timings": {},
        "traces": {},
    }
    original_compile = learning.compile_gradient_evaluators
    original_execute = learning.execute_learning_update
    captured = {}
    started = time.perf_counter()

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(output, report)
        print(f"split_profile stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    def compile_observed(**kwargs):
        jax = kwargs["jax"]
        _require_supported_jax(jax)
        from jax._src import compiler
        from jax.experimental.serialize_executable import serialize

        component = None
        original_emit = kwargs["emit"]

        def compile_emit(stage, **values):
            nonlocal component
            if stage == "compile_gradient_graph":
                component = values["component"]
            original_emit(stage, **values)

        def publish(event):
            report["cache_events"].append(event)
            emit("cache_event")

        with observe_compiler(
            compiler, lambda: component, publish, required_cached=("daily", "T7", "T30")
        ):
            evaluators, metadata = original_compile(**{**kwargs, "emit": compile_emit})
        for name, executable in evaluators.items():
            report["executables"][name] = save_executable(assets, name, executable, serialize)
            emit("executable_saved", component=name)
        return evaluators, metadata

    def execute_capture(**kwargs):
        # Keep the first two actual batches; no duplicate rows are added for a
        # nominal larger batch. Overlap between artifacts is still possible.
        captured.setdefault("daily_batches", []).append(kwargs["daily_samples"])
        result = original_execute(**kwargs)
        captured["kwargs"] = kwargs
        captured["parameters"] = result[0]
        return result

    learning.compile_gradient_evaluators = compile_observed
    learning.execute_learning_update = execute_capture
    args.output = output.with_suffix(".updates.json")
    try:
        emit("load_unchanged_learning_graphs")
        outcome = learning.run(args)
        if outcome["status"] != "passed":
            raise RuntimeError("unchanged technical updates did not pass")
        report["technical_update_report"] = str(args.output)
        context = captured["kwargs"]
        parameters = captured["parameters"]
        jax = context["jax"]
        import jax.numpy as jnp

        from scripts.hpc.shared_daily_microbatch import make_daily_accumulator

        batch32 = captured["daily_batches"][1]
        batch64 = concatenate_daily(jax, *captured["daily_batches"])
        evaluators = context["evaluators"]
        rows = jax.tree_util.tree_map(lambda value: value[:4], batch32)
        invocations = {
            "daily_micro4": lambda: evaluators["daily"](parameters, rows),
            "T7": lambda: evaluators["T7"](parameters, *context["seven_case"]),
            "T30": lambda: evaluators["T30"](parameters, *context["thirty_case"]),
        }
        for count, batch in ((32, batch32), (64, batch64)):
            accumulate = make_daily_accumulator(
                jax=jax, jnp=jnp, microbatch_size=4,
                evaluator=evaluators["daily"], expected_count=count,
            )
            invocations[f"daily{count}_micro4"] = (
                lambda accumulate=accumulate, batch=batch: accumulate(parameters, batch)
            )
        for name, invoke in invocations.items():
            emit("warm_replays", component=name)
            report["timings"][name] = benchmark_replays(invoke, jax.block_until_ready, args.repeats)
            emit("replays_completed", component=name)
        report["sampling"] = {
            "artifact_ids": [row["artifact_id"] for row in outcome["updates"]],
            "daily64_source": "concatenated_two_real_32_row_artifacts_no_padding",
            "unique_date_count": "not_audited_do_not_equate_draws_with_unique_coverage",
            "weights": "fixed_technical_update2_parameters_no_update_during_replays",
            "data_timing": "preloaded_host_batches_not_end_to_end_dataset_IO",
        }
        # One daily microbatch and one seven-day reverse are sufficient to
        # compare kernel/source categories without tracing all thirty days.
        for name in args.trace_components:
            trace_path = assets / f"trace-{name}"
            emit("trace_started", component=name)
            with jax.profiler.trace(str(trace_path), create_perfetto_link=False):
                with jax.profiler.TraceAnnotation(f"split_profile_{name}"):
                    jax.block_until_ready(invocations[name]())
            report["traces"][name] = str(trace_path)
            emit("trace_completed", component=name)
        emit("completed", status="passed", device_memory=jax.devices()[0].memory_stats())
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output = output
        learning.compile_gradient_evaluators = original_compile
        learning.execute_learning_update = original_execute


def parser():
    from scripts.hpc.run_shared_learning import parser as base_parser

    result = base_parser()
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--trace-components", nargs="*", choices=("daily_micro4", "T7"), default=("daily_micro4", "T7"))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
