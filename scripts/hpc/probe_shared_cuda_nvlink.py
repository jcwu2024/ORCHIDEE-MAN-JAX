"""Small no-Teacher CUDA parallel-module linking smoke, not model admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    import jax.numpy as jnp
    if len(devices) != 1:
        raise ValueError("one free-node GPU required")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"status": "running", "scope": "small_cuda_parallel_module_linking_only",
              "teacher_data": False, "model_admission": False,
              "jax_version": jax.__version__, "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
              "TF_CPP_VMODULE": os.environ.get("TF_CPP_VMODULE", ""),
              "compilation_cache_enabled": bool(jax.config.jax_enable_compilation_cache)}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **fields)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(stage, flush=True)
    try:
        parameters = jnp.asarray(np.linspace(-0.2, 0.2, 64*32, dtype=np.float32).reshape(64, 32))
        state = jnp.asarray(np.linspace(-0.3, 0.3, 32*64, dtype=np.float32).reshape(32, 64))
        def objective(p, x):
            hidden = jnp.tanh(x @ p)
            projected = jnp.sin(hidden)
            gram = hidden @ hidden.T
            return jnp.sum(projected**2) + jnp.float32(0.001)*jnp.sum(gram**2)
        options = {"xla_gpu_enable_llvm_module_compilation_parallelism": True,
                   "xla_gpu_force_compilation_parallelism": 4}
        emit("compiling", compiler_options=options)
        began = time.perf_counter()
        executable = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)),
                             compiler_options=options).lower(parameters, state).compile()
        emit("compiled", compile_seconds=time.perf_counter()-began)
        actual = jax.device_get(jax.block_until_ready(executable(parameters, state)))
        replay = jax.device_get(jax.block_until_ready(executable(parameters, state)))
        for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(replay), strict=True):
            if not np.all(np.isfinite(left)) or not np.allclose(left, right, rtol=1e-5, atol=1e-6):
                raise ValueError("small compiled native AD finite/replay failure")
        if not all(np.any(np.asarray(gradient) != 0) for gradient in actual[1]):
            raise ValueError("small parameter/state AD is zero")
        emit("completed", status="passed", loss=float(actual[0]),
             gradient_shapes=[list(np.shape(value)) for value in actual[1]],
             gradient_nonzero=[int(np.count_nonzero(value)) for value in actual[1]],
             device_memory=devices[0].memory_stats())
    except BaseException as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
