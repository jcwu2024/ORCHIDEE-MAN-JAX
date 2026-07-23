"""Fail fast unless the canonical Explore1000 environment executes on GPU."""

from __future__ import annotations

import json
import logging

import numpy as np

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    jax, devices = initialize_gpu_backend()

    # jax.numpy must be imported only after explicit CUDA plugin discovery.
    import jax.numpy as jnp

    @jax.jit
    def objective(value):
        transformed = value @ value.T
        return jnp.mean(jnp.sin(transformed))

    value = jnp.arange(4096, dtype=jnp.float32).reshape(64, 64) / 4096.0
    loss, gradient = jax.value_and_grad(objective)(value)
    jax.block_until_ready((loss, gradient))
    if not np.isfinite(np.asarray(loss)) or not np.all(
        np.isfinite(np.asarray(gradient))
    ):
        raise RuntimeError("orcjax_gpu JIT/gradient smoke produced non-finite values")
    print(
        json.dumps(
            {
                "jax_version": jax.__version__,
                "backend": jax.default_backend(),
                "devices": [str(device) for device in devices],
                "loss": float(loss),
                "gradient_norm": float(jnp.linalg.norm(gradient)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
