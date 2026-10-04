"""Enter the physical batch diagnostic through the accepted Explore1000 CUDA runtime.

The plugin initialization follows scripts/hpc/orcjax_gpu_runtime.py in the
canonical deployment checkout. It changes only backend discovery, not physics.
"""
from __future__ import annotations

import importlib
import os
import runpy
import time
from pathlib import Path


def main() -> None:
    os.environ["ORCHJAX_GLOBAL_BATCH_PROCESS_START"] = str(time.perf_counter())
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
        raise RuntimeError(f"free-node diagnostic requires exactly one GPU: {devices}")
    jax.config.update("jax_enable_x64", True)
    entry = os.environ.get("ORCHJAX_GLOBAL_BATCH_ENTRY", "benchmark_global_batch.py")
    if entry not in {"benchmark_global_batch.py", "probe_global_batch_cache.py"}:
        raise ValueError(f"unsupported diagnostic entry: {entry}")
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "dev" / entry),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
