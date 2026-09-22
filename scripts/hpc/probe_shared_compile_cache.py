"""Bounded persistent-cache diagnostic for the shared daily gradient graph."""
from __future__ import annotations

import inspect
import json
import logging
import os
import platform
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from scripts.hpc import probe_shared_execution as shared_probe
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

SUPPORTED_JAX_VERSION = "0.4.38"
_CACHE_SUFFIX = "-cache"


class DiagnosticComplete(BaseException):
    """Escape the inherited runner without entering its ``except Exception``."""


class _KeyOnlyStop(BaseException):
    pass


class _BackendCompileRejected(BaseException):
    pass


def _require_supported_jax(jax: Any) -> None:
    if jax.__version__ != SUPPORTED_JAX_VERSION:
        raise RuntimeError(
            f"shared cache diagnostic requires JAX {SUPPORTED_JAX_VERSION}; "
            f"found {jax.__version__}"
        )
    from jax._src import compiler

    expected = (
        "backend",
        "computation",
        "devices",
        "compile_options",
        "host_callbacks",
        "pgle_profiler",
    )
    actual = tuple(inspect.signature(compiler.compile_or_get_cached).parameters)
    if actual != expected or not hasattr(compiler, "backend_compile"):
        raise RuntimeError(f"unsupported JAX compiler internals: {actual}")


def _internals() -> SimpleNamespace:
    from jax._src import cache_key, compilation_cache, compiler, config

    return SimpleNamespace(
        cache_key=cache_key,
        compilation_cache=compilation_cache,
        compiler=compiler,
        config=config,
    )


def _host_context() -> Mapping[str, Any]:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    env_names = (
        "CUDA_VISIBLE_DEVICES",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_COMPILATION_CACHE_MAX_SIZE",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY",
        "JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_ID",
    )
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None
    return {
        "host": platform.node(),
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "load_average": load_average,
        "environment": {name: os.environ.get(name) for name in env_names},
    }


def _cache_record(
    *,
    internals: Any,
    backend: Any,
    computation: Any,
    devices: Any,
    compile_options: Any,
) -> Mapping[str, Any]:
    began = time.perf_counter()
    config = internals.config
    enabled = bool(internals.compilation_cache.is_cache_used(backend))
    remove_pointer = bool(config.remove_custom_partitioning_ptr_from_cache_key.value)
    ignored = (
        internals.cache_key.IgnoreCallbacks.CUSTOM_PARTITIONING
        if remove_pointer
        else internals.cache_key.IgnoreCallbacks.NO
    )
    messages: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger = logging.getLogger(internals.cache_key.__name__)
    handler = Capture()
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        key = internals.compilation_cache.get_cache_key(
            computation,
            devices,
            compile_options,
            backend,
            ignore_callbacks=ignored,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
    component_hashes = {}
    cumulative_hashes = {}
    for message in messages:
        match = re.fullmatch(
            r"get_cache_key hash of serialized (.+): ([0-9a-f]+)", message
        )
        if match:
            component_hashes[match.group(1)] = match.group(2)
        match = re.fullmatch(
            r"get_cache_key hash after serializing (.+): ([0-9a-f]+)", message
        )
        if match:
            cumulative_hashes[match.group(1)] = match.group(2)
    cache_dir = config.compilation_cache_dir.value
    cache_file = None if cache_dir is None else Path(cache_dir) / f"{key}{_CACHE_SUFFIX}"
    exists = bool(cache_file is not None and cache_file.is_file())
    readable = False
    size = None
    error = None
    if exists:
        try:
            size = cache_file.stat().st_size
            with cache_file.open("rb") as handle:
                handle.read(1)
            readable = True
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "enabled_for_backend": enabled,
        "configured_directory": None if cache_dir is None else str(cache_dir),
        "key": key,
        "component_hashes": component_hashes,
        "cumulative_hashes": cumulative_hashes,
        "ignore_callbacks": getattr(ignored, "name", str(ignored)),
        "include_metadata_in_key": bool(
            config.compilation_cache_include_metadata_in_key.value
        ),
        "custom_hook": internals.cache_key.custom_hook(),
        "file": None if cache_file is None else str(cache_file),
        "file_exists": exists,
        "file_bytes": size,
        "file_readable": readable,
        "file_error": error,
        "lookup_seconds": time.perf_counter() - began,
    }


def diagnose_lowered(
    *,
    lowered: Any,
    mode: str,
    report: dict[str, Any],
    emit: Callable[[str], None],
    internals: Any | None = None,
) -> None:
    """Inspect the real compile boundary, always restoring both JAX hooks."""
    api = _internals() if internals is None else internals
    compiler = api.compiler
    original_compile = compiler.compile_or_get_cached
    original_backend_compile = compiler.backend_compile

    def reject_or_compile(*args: Any, **kwargs: Any) -> Any:
        report["counters"]["backend_compile_calls"] += 1
        emit("backend_compile_requested")
        if mode == "cached-only":
            raise _BackendCompileRejected()
        return original_backend_compile(*args, **kwargs)

    def inspect_cache(
        backend: Any,
        computation: Any,
        devices: Any,
        compile_options: Any,
        host_callbacks: Any,
        pgle_profiler: Any = None,
    ) -> Any:
        report["counters"]["compile_or_get_cached_calls"] += 1
        report["backend"] = {
            "platform": backend.platform,
            "platform_version": backend.platform_version,
            "devices": [str(device) for device in devices.flatten()],
        }
        report["cache"] = _cache_record(
            internals=api,
            backend=backend,
            computation=computation,
            devices=devices,
            compile_options=compile_options,
        )
        emit("cache_lookup_completed")
        if mode == "key-only":
            raise _KeyOnlyStop()
        backend_calls_before = report["counters"]["backend_compile_calls"]
        result = original_compile(
            backend,
            computation,
            devices,
            compile_options,
            host_callbacks,
            pgle_profiler,
        )
        if report["counters"]["backend_compile_calls"] == backend_calls_before:
            report["counters"]["persistent_cache_hits"] += 1
            emit("cached_executable_loaded")
        else:
            emit("backend_compile_returned")
        return result

    compiler.compile_or_get_cached = inspect_cache
    compiler.backend_compile = reject_or_compile
    try:
        began = time.perf_counter()
        try:
            lowered.compile()
        except _KeyOnlyStop:
            report["status"] = "key_observed"
        except _BackendCompileRejected:
            report["status"] = "cache_miss_backend_compile_rejected"
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        else:
            if report["counters"]["compile_or_get_cached_calls"] == 0:
                report.update(
                    status="failed",
                    error="lowered.compile bypassed the intercepted compile boundary",
                )
            else:
                report["status"] = (
                    "backend_compile_completed"
                    if report["counters"]["backend_compile_calls"]
                    else "cached_executable_loaded"
                )
        report["timing_seconds"]["compile_boundary"] = time.perf_counter() - began
        emit("diagnostic_completed")
    finally:
        compiler.compile_or_get_cached = original_compile
        compiler.backend_compile = original_backend_compile
    raise DiagnosticComplete()


def _base_report(jax: Any, mode: str, self_test: bool) -> dict[str, Any]:
    return {
        "schema_version": "shared_compile_cache_diagnostic_v1",
        "status": "running",
        "diagnostic_mode": mode,
        "self_test": self_test,
        "jax_version": jax.__version__,
        "teacher_data_read": False if self_test else None,
        "retained_training_updates": 0,
        "scientific_acceptance": False,
        "host_context": _host_context(),
        "counters": {
            "compile_or_get_cached_calls": 0,
            "backend_compile_calls": 0,
            "persistent_cache_hits": 0,
        },
        "timing_seconds": {},
    }


def _run_self_test(args: Any, jax: Any, report: dict[str, Any], emit: Callable[[str], None]) -> None:
    import jax.numpy as jnp

    def objective(parameters: Any, sample: Any) -> tuple[Any, Mapping[str, Any]]:
        return jnp.sum(jnp.square(parameters * sample)), {"hard": jnp.asarray(0)}

    began = time.perf_counter()
    lowered = jax.jit(jax.value_and_grad(objective, has_aux=True)).lower(
        jnp.ones((4,), dtype=jnp.float32), jnp.arange(4, dtype=jnp.float32)
    )
    report["timing_seconds"]["lower"] = time.perf_counter() - began
    emit("daily_lowered")
    diagnose_lowered(lowered=lowered, mode=args.diagnostic_mode, report=report, emit=emit)


def run(args: Any) -> Mapping[str, Any]:
    import jax

    _require_supported_jax(jax)
    began = time.perf_counter()
    report = _base_report(jax, args.diagnostic_mode, args.self_test)

    def emit(stage: str) -> None:
        report["stage"] = stage
        report["elapsed_seconds"] = time.perf_counter() - began
        _atomic_json(args.output, report)
        print(json.dumps({"stage": stage, "seconds": report["elapsed_seconds"]}), flush=True)

    emit("initialized")
    if args.self_test:
        try:
            _run_self_test(args, jax, report, emit)
        except DiagnosticComplete:
            return report
        raise AssertionError("self-test diagnostic boundary returned")

    original = shared_probe.compile_gradient_evaluators

    def daily_only(**values: Any) -> None:
        daily_backward = jax.jit(jax.value_and_grad(values["daily_objective"], has_aux=True))
        report["timing_seconds"]["bootstrap"] = time.perf_counter() - began
        emit("daily_lower_started")
        lower_started = time.perf_counter()
        lowered = daily_backward.lower(values["parameters"], values["daily_sample"])
        report["timing_seconds"]["daily_lower"] = time.perf_counter() - lower_started
        report["teacher_data_read"] = True
        emit("daily_lowered")
        diagnose_lowered(
            lowered=lowered,
            mode=args.diagnostic_mode,
            report=report,
            emit=emit,
        )

    shared_probe.compile_gradient_evaluators = daily_only
    try:
        try:
            shared_probe.run(args)
        except DiagnosticComplete:
            return report
    finally:
        shared_probe.compile_gradient_evaluators = original
    raise AssertionError("inherited runner passed the daily diagnostic boundary")


def parser():
    result = shared_probe.parser()
    result.description = __doc__
    # Self-test intentionally reads no project data; enforce these conditionally
    # in main after argparse has identified that mode.
    conditionally_required = {
        "dataset_manifest",
        "acceptance_report",
        "base_statistics",
        "process_statistics",
        "fast_statistics",
        "prepared_manifest",
        "joint_admission_report",
        "validation_manifest",
        "benchmark_protocol",
    }
    for action in result._actions:
        if action.dest in conditionally_required:
            action.required = False
    result.add_argument(
        "--diagnostic-mode",
        choices=("key-only", "cached-only", "compile"),
        default="key-only",
    )
    result.add_argument("--self-test", action="store_true")
    return result


def _validate_args(args: Any, argument_parser: Any) -> None:
    if args.self_test:
        return
    required = (
        "dataset_manifest",
        "acceptance_report",
        "base_statistics",
        "process_statistics",
        "fast_statistics",
        "prepared_manifest",
        "joint_admission_report",
        "validation_manifest",
        "benchmark_protocol",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        argument_parser.error(
            "real diagnostic requires " + ", ".join("--" + name.replace("_", "-") for name in missing)
        )


if __name__ == "__main__":
    argument_parser = parser()
    arguments = argument_parser.parse_args()
    _validate_args(arguments, argument_parser)
    outcome = run(arguments)
    raise SystemExit(0 if outcome["status"] != "failed" else 1)
