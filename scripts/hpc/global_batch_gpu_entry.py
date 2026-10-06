"""Enter the physical batch diagnostic through the accepted Explore1000 CUDA runtime.

The plugin initialization follows scripts/hpc/orcjax_gpu_runtime.py in the
canonical deployment checkout. It changes only backend discovery, not physics.
"""
from __future__ import annotations

import importlib
import os
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    os.environ["ORCHJAX_GLOBAL_BATCH_PROCESS_START"] = str(time.perf_counter())
    candidate_first = "--execution-order=candidate-first" in sys.argv or any(
        left == "--execution-order" and right == "candidate-first" for left, right in zip(sys.argv, sys.argv[1:]))
    capacity = os.environ.get("ORCHJAX_GLOBAL_BATCH_ENTRY") == "benchmark_global_gpu_capacity.py"
    if candidate_first or capacity:
        import json

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.dev.global_server_resources import admit_gpu

        resources = admit_gpu()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        os.environ["ORCHJAX_SERVER_RESOURCE_PLAN"] = json.dumps(resources)
    import jax
    from jax._src import xla_bridge

    # Canonical CentOS 7 / Singularity profile is pinned to JAX 0.4.38.
    xla_bridge._discover_and_register_pjrt_plugins()
    if "cuda" not in xla_bridge._backend_factories:
        importlib.import_module("jax_plugins.xla_cuda12").initialize()
    with xla_bridge._backend_lock:
        backend = xla_bridge._backends.get("cuda")
        if backend is None:
            backend = xla_bridge._init_backend("cuda")
            xla_bridge._backends["cuda"] = backend
        xla_bridge._default_backend = backend
    devices = tuple(backend.devices())
    if len(devices) != 1 or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"diagnostic requires exactly one GPU: {devices}")
    jax.config.update("jax_enable_x64", True)
    if candidate_first or capacity:
        from scripts.dev.global_server_resources import thread_affinity

        resources["thread_affinity_after_backend"] = thread_affinity(resources["original_affinity"])
        os.environ["ORCHJAX_SERVER_RESOURCE_PLAN"] = json.dumps(resources)
    entry = os.environ.get("ORCHJAX_GLOBAL_BATCH_ENTRY", "benchmark_global_batch.py")
    if entry not in {"benchmark_global_batch.py", "benchmark_global_scale.py", "benchmark_global_gpu_capacity.py", "probe_global_batch_cache.py"}:
        raise ValueError(f"unsupported diagnostic entry: {entry}")
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "dev" / entry),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
