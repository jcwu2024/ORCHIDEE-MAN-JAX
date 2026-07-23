"""Initialize the accepted Explore1000 JAX GPU runtime before JAX use."""

from __future__ import annotations

from typing import Any


def initialize_gpu_backend() -> tuple[Any, tuple[Any, ...]]:
    """Register the CUDA PJRT plugin and fail rather than falling back to CPU."""
    import jax
    from jax._src import xla_bridge

    # JAX 0.4.38 can miss namespace-package plugin discovery in the
    # Explore1000 container. This must run before jax.numpy or model imports.
    xla_bridge._discover_and_register_pjrt_plugins()
    devices = tuple(jax.devices())
    if not devices or any(device.platform != "gpu" for device in devices):
        backend_errors = dict(getattr(xla_bridge, "_backend_errors", {}))
        raise RuntimeError(
            "orcjax_gpu did not select only GPU devices: "
            f"devices={devices}, backend_errors={backend_errors}"
        )
    return jax, devices
