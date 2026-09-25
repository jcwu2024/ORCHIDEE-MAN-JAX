"""Tiny diagnostic of attempted single-device executable retargeting.

No Teacher or full model is loaded. This diagnostic targets JAX 0.4.38's
private serialization protocol, never a generic cross-version cache.
The September25 V100 run rejected this approach: the native binary retained
cuda0 despite the requested load options. Do not use this as a loader.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np


def deserialize_on_device(payload, original, target):
    import jax
    from jax._src.lib import xla_client as xc
    from jax.experimental.serialize_executable import _JaxPjrtUnpickler

    if jax.__version__ != "0.4.38":
        raise ValueError("retarget probe is specific to deployed JAX 0.4.38")
    serialized, in_tree, out_tree = payload
    # GPU PJRT 0.4.38 exposes compile_options() but raises UNIMPLEMENTED.
    # Attempt explicit placement through deserialization options. The deployed
    # backend ignores this placement for the native binary; retain as evidence.
    options = xc.CompileOptions()
    options.num_replicas = 1
    options.num_partitions = 1
    options.device_assignment = xc.DeviceAssignment.create(np.asarray([[target.id]], dtype=np.int32))

    class RetargetUnpickler(_JaxPjrtUnpickler):
        def persistent_load(self, pid):
            if pid[0] == "exec":
                return self.backend.deserialize_executable(pid[1], compile_options=options)
            if pid[0] == "device":
                return target
            return super().persistent_load(pid)

    unloaded, arguments, no_kwargs = RetargetUnpickler(io.BytesIO(serialized), target.client).load()
    return jax.stages.Compiled(unloaded.load(), in_tree.unflatten(arguments), out_tree, no_kwargs=no_kwargs)


def run(output):
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import serialize

    devices = jax.local_devices()
    report = {"jax_version": jax.__version__, "devices": [str(d) for d in devices], "status": "running"}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        if len(devices) != 2 or any(d.platform != "gpu" for d in devices):
            raise ValueError("requires two visible GPUs")
        primary, secondary = devices
        p = {"w": np.arange(128, dtype=np.float32).reshape(16, 8) / 128, "b": np.ones(8, dtype=np.float32)}
        x = np.linspace(-1, 1, 64, dtype=np.float32).reshape(4, 16)

        def loss(parameters, data):
            values = jnp.tanh(data @ parameters["w"] + parameters["b"])
            return jnp.mean(values ** 2), values

        args0 = jax.device_put((p, x), primary)
        args1 = jax.device_put((p, x), secondary)
        with jax.default_device(primary):
            original = jax.jit(jax.value_and_grad(loss, has_aux=True)).lower(*args0).compile()
        report["deserialize_api"] = primary.client.deserialize_executable.__doc__
        report["runtime_methods"] = [name for name in dir(original.runtime_executable()) if "compile" in name or "device" in name]
        save()
        restored = deserialize_on_device(serialize(original), original, secondary)
        expected = original(*args0)
        actual = restored(*args1)
        jax.block_until_ready((expected, actual))
        differences = []
        for left, right in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual), strict=True):
            if right.devices() != {secondary}:
                raise ValueError("loaded executable did not execute on the second GPU")
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-5, atol=1e-6)
            differences.append(float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
        report.update(status="passed", maximum_absolute_difference=max(differences), output_device=str(secondary))
        save()
        return report
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps(result, indent=2))
