"""Run unchanged split timing, requiring reuse of the completed daily/T7 caches.

Only compiler observability changes. T30 may compile once; the two completed
graphs must load from cache, avoiding another unobserved expensive restart.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def observe_compiler(compiler, component, publish):
    original_lookup = compiler.compile_or_get_cached
    original_backend = compiler.backend_compile
    active = None

    def backend(*args, **kwargs):
        if active is not None:
            active["backend_compile_calls"] += 1
            if active["component"] in ("daily", "T7"):
                active["status"] = "completed_graph_cache_miss"
                publish(dict(active))
                raise RuntimeError(
                    f"{active['component']} completed graph cache missed; stopped before backend compilation"
                )
        return original_backend(*args, **kwargs)

    def lookup(*args, **kwargs):
        nonlocal active
        computation = args[1] if len(args) > 1 else kwargs["computation"]
        name = str(computation.operation.attributes["sym_name"]).strip('"')
        if name not in ("jit_objective", "jit_checkpointed_loss_with_aux"):
            return original_lookup(*args, **kwargs)
        began = time.perf_counter()
        active = dict(component=component(), module=name, backend_compile_calls=0, status="cache_lookup")
        publish(dict(active))
        try:
            result = original_lookup(*args, **kwargs)
            active.update(
                status="cache_hit" if active["backend_compile_calls"] == 0 else "backend_compiled",
                seconds=time.perf_counter() - began,
            )
            publish(dict(active))
            return result
        finally:
            active = None

    compiler.compile_or_get_cached = lookup
    compiler.backend_compile = backend
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original_lookup
        compiler.backend_compile = original_backend


def main():
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    args = learning.parser().parse_args()
    if args.technical_updates != 2:
        raise ValueError("cache-observed split timing requires --technical-updates 2")
    original = learning.compile_gradient_evaluators
    report = {"events": [], "scope": "unchanged_full_update_timing_compile_boundary"}
    cache_output = args.output.with_suffix(".cache.json")

    def observed(**kwargs):
        jax = kwargs["jax"]
        _require_supported_jax(jax)
        from jax._src import compiler

        component_name = None
        original_emit = kwargs["emit"]

        def emit(stage, **values):
            nonlocal component_name
            if stage == "compile_gradient_graph":
                component_name = values["component"]
            original_emit(stage, **values)

        def publish(event):
            report["events"].append(event)
            _atomic_json(cache_output, report)
            print(f"split_cache component={event['component']} status={event['status']}", flush=True)

        with observe_compiler(compiler, lambda: component_name, publish):
            return original(**{**kwargs, "emit": emit})

    learning.compile_gradient_evaluators = observed
    try:
        result = learning.run(args)
    finally:
        learning.compile_gradient_evaluators = original
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
