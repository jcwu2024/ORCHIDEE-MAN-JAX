from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def test_gpu_runtime_instantiates_registered_cuda_backend_before_devices(monkeypatch):
    from jax._src import xla_bridge

    calls = []
    device = SimpleNamespace(platform="gpu")
    backend = SimpleNamespace(devices=lambda: (device,))
    monkeypatch.setitem(xla_bridge._backend_factories, "cuda", object())
    monkeypatch.setattr(xla_bridge, "_backends", {})
    monkeypatch.setattr(xla_bridge, "_default_backend", None)
    monkeypatch.setattr(
        xla_bridge,
        "_discover_and_register_pjrt_plugins",
        lambda: calls.append("discover"),
    )
    monkeypatch.setattr(
        xla_bridge, "_init_backend", lambda name: calls.append(name) or backend
    )

    _jax, devices = initialize_gpu_backend()

    assert calls == ["discover", "cuda"]
    assert devices == (device,)


def test_gpu_runtime_rejects_missing_cuda_without_cpu_fallback(monkeypatch):
    from jax._src import xla_bridge

    monkeypatch.setitem(xla_bridge._backend_factories, "cuda", object())
    monkeypatch.setattr(xla_bridge, "_backends", {})
    monkeypatch.setattr(xla_bridge, "_default_backend", None)
    monkeypatch.setattr(xla_bridge, "_discover_and_register_pjrt_plugins", lambda: None)
    monkeypatch.setattr(
        xla_bridge,
        "_init_backend",
        lambda _name: SimpleNamespace(
            devices=lambda: (SimpleNamespace(platform="cpu"),)
        ),
    )
    monkeypatch.setattr(xla_bridge, "_backend_errors", {"cuda": "failed"})

    with pytest.raises(RuntimeError, match="did not select only GPU"):
        initialize_gpu_backend()


def test_gpu_runtime_registers_requested_cpu_without_changing_cuda_default(monkeypatch):
    from jax._src import xla_bridge

    calls = []
    gpu = SimpleNamespace(platform="gpu")
    cpu = SimpleNamespace(platform="cpu")
    cuda_backend = SimpleNamespace(devices=lambda: (gpu,))
    cpu_backend = SimpleNamespace(devices=lambda: (cpu,))
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    monkeypatch.setitem(xla_bridge._backend_factories, "cuda", object())
    monkeypatch.setitem(xla_bridge._backend_factories, "cpu", object())
    monkeypatch.setattr(xla_bridge, "_backends", {})
    monkeypatch.setattr(xla_bridge, "_default_backend", None)
    monkeypatch.setattr(xla_bridge, "_discover_and_register_pjrt_plugins", lambda: None)
    monkeypatch.setattr(
        xla_bridge,
        "_init_backend",
        lambda name: calls.append(name) or {"cuda": cuda_backend, "cpu": cpu_backend}[name],
    )

    _jax, devices = initialize_gpu_backend()

    assert calls == ["cuda", "cpu"]
    assert devices == (gpu,)
    assert xla_bridge._default_backend is cuda_backend
    assert xla_bridge._backends == {
        "cuda": cuda_backend,
        "cpu": cpu_backend,
    }
