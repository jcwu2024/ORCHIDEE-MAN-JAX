"""Profile one unchanged cached shared gradient; never train or cold-compile it."""
from __future__ import annotations

import gzip
import json
import platform
import time

import numpy as np

from scripts.hpc import probe_shared_execution as probe
from scripts.hpc.probe_shared_compile_cache import (
    DiagnosticComplete,
    _host_context,
    _require_supported_jax,
    _validate_args,
)
from scripts.hpc.probe_shared_compile_cache import parser as base_parser
from scripts.hpc.run_shared_learning import memory_sizes
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json


def run(args):
    import jax
    from jax._src import compiler

    _require_supported_jax(jax)
    began = time.perf_counter()
    report = {
        "status": "running", "scope": "execution_profile_not_scientific_acceptance",
        "component": args.component, "self_test": args.self_test,
        "retained_updates": 0, "jax_version": jax.__version__,
        "host_context": _host_context(), "timing_seconds": {},
    }

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter()-began)
        _atomic_json(args.output, report)
        print(json.dumps({"stage": stage, "seconds": report["elapsed_seconds"]}), flush=True)

    def measure(function, arguments, memory_limit):
        emit("lower_started")
        started = time.perf_counter()
        lowered = function.lower(*arguments)
        report["timing_seconds"]["lower"] = time.perf_counter()-started
        emit("cache_load_started")
        started = time.perf_counter()
        if args.self_test:
            # Tiny no-data profiler plumbing check; no model graph is compiled.
            executable = lowered.compile()
            report["cache"] = {"policy": "tiny_self_test_compile"}
        else:
            executable, report["cache"] = probe.compile_cached_lowered(lowered, compiler)
        report["timing_seconds"]["cache_load"] = time.perf_counter()-started
        if executable is None:
            emit("cache_miss", status="cache_miss")
            raise DiagnosticComplete()
        report["compiled_memory"] = memory_sizes(executable)
        if report["compiled_memory"]["estimated_live_bytes"] > memory_limit:
            emit("resource_limited", status="resource_limited")
            raise DiagnosticComplete()
        emit("warmup")
        result = jax.block_until_ready(executable(*arguments))
        times = []
        for _ in range(3):
            started = time.perf_counter()
            result = jax.block_until_ready(executable(*arguments))
            times.append(time.perf_counter()-started)
        report["timing_seconds"]["warm_replays"] = times
        report["timing_seconds"]["warm_median"] = float(np.median(times))
        # Compiled FLOPs/bytes are optimizer estimates, not a hardware counter.
        try:
            report["compiler_cost_estimate"] = executable.cost_analysis()
        except (AttributeError, RuntimeError, NotImplementedError) as error:
            report["compiler_cost_error"] = str(error)
        trace_dir = args.output.with_suffix(".trace")
        emit("profile_started", trace_directory=str(trace_dir))
        started = time.perf_counter()
        with jax.profiler.trace(trace_dir, create_perfetto_link=False):
            with jax.profiler.TraceAnnotation("shared_cached_gradient", component=args.component):
                result = jax.block_until_ready(executable(*arguments))
        report["timing_seconds"]["profile_including_export"] = time.perf_counter()-started
        report["trace_files"] = [str(p) for p in sorted(trace_dir.rglob("*")) if p.is_file()]
        # One compressed file permits offline kernel/HLO/source correlation.
        hlo_path = args.output.with_suffix(".hlo.txt.gz")
        with gzip.open(hlo_path, "wt", encoding="utf-8") as handle:
            handle.write(executable.as_text())
        report["optimized_hlo"] = str(hlo_path)
        report["device_memory"] = jax.devices()[0].memory_stats()
        (value, auxiliary), _gradient = result
        report["loss"] = float(value)
        report["hard"] = {key: int(value) for key, value in jax.device_get(auxiliary["hard"]).items()}
        emit("completed", status="completed")
        raise DiagnosticComplete()

    original = probe.compile_gradient_evaluators

    def selected_only(**values):
        report["timing_seconds"]["bootstrap"] = time.perf_counter()-began
        if args.component == "daily":
            objective = values["daily_objective"]
            arguments = (values["parameters"], values["daily_sample"])
        else:
            objective = values["recurrent_objective"]
            case = values["seven_case"] if args.component == "T7" else values["thirty_case"]
            arguments = (values["parameters"], *case)
        measure(jax.jit(jax.value_and_grad(objective, has_aux=True)),
                arguments, values["memory_limit_bytes"])

    emit("initialized", host=platform.node())
    try:
        if args.self_test:
            import jax.numpy as jnp

            def objective(weights, sample):
                return jnp.sum((sample @ weights)**2), {"hard": {"invalid": jnp.asarray(0)}}

            arguments = (jnp.ones((64, 64), dtype=jnp.float32),
                         jnp.ones((64, 64), dtype=jnp.float32))
            measure(jax.jit(jax.value_and_grad(objective, has_aux=True)), arguments, 2**30)
        else:
            probe.compile_gradient_evaluators = selected_only
            probe.run(args)
        raise RuntimeError("bootstrap did not reach selected profile boundary")
    except DiagnosticComplete:
        return report
    except Exception as error:
        emit("error", status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        probe.compile_gradient_evaluators = original


def parser():
    result = base_parser()
    result.description = __doc__
    result.set_defaults(diagnostic_mode="cached-only")
    result.add_argument("--component", choices=("daily", "T7", "T30"), default="T7")
    return result


if __name__ == "__main__":
    argument_parser = parser()
    arguments = argument_parser.parse_args()
    _validate_args(arguments, argument_parser)
    if arguments.diagnostic_mode != "cached-only":
        argument_parser.error("this profiler only permits --diagnostic-mode cached-only")
    if arguments.output.with_suffix(".trace").exists():
        argument_parser.error("choose a new --output path; trace directory already exists")
    outcome = run(arguments)
    raise SystemExit(0 if outcome["status"] == "completed" else 2)
