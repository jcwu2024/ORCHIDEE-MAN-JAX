"""Fail fast unless the canonical Explore1000 environment executes on GPU."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import xla_bridge


def main() -> int:
    devices = tuple(jax.devices())
    if not devices or any(device.platform != "gpu" for device in devices):
        backend_errors = dict(getattr(xla_bridge, "_backend_errors", {}))
        raise RuntimeError(
            "orcjax_gpu did not select only GPU devices: "
            f"devices={devices}, backend_errors={backend_errors}"
        )

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
