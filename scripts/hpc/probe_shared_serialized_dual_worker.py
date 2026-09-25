"""Private pipe worker for trusted T7/T30 binaries on one visible GPU.

The parent launches this module with a CUDA_VISIBLE_DEVICES value before this
module imports JAX. Standard output is exclusively the pickle message stream;
diagnostics and exceptions go to standard error.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import sys
import time
from pathlib import Path

PROTOCOL_OUTPUT = sys.stdout.buffer


def serve(profile_path, memory_limit_bytes):
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load

    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.probe_shared_dual import make_recurrent_branch
    from scripts.hpc.run_shared_learning import make_tree_operations, memory_sizes

    def send(message):
        pickle.dump(message, PROTOCOL_OUTPUT, protocol=pickle.HIGHEST_PROTOCOL)
        PROTOCOL_OUTPUT.flush()

    def receive():
        return pickle.load(sys.stdin.buffer)

    _require_supported_jax(jax)
    devices = tuple(jax.local_devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("serialized recurrent worker requires exactly one logical GPU")
    device = devices[0]
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    evaluators, metadata = {}, {}
    for name in ("T7", "T30"):
        path = Path(profile["executables"][name]["executable"])
        began = time.perf_counter()
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        executable = deserialize_and_load(*payload)
        memory = memory_sizes(executable)
        if memory["estimated_live_bytes"] > memory_limit_bytes:
            raise ValueError(f"saved worker {name} executable exceeds memory limit")
        evaluators[int(name[1:])] = executable
        metadata[name] = {"source_executable": str(path), "memory": memory,
                          "load_seconds": time.perf_counter() - began}
    operations = make_tree_operations(jax, jnp, 32)
    branch = make_recurrent_branch(
        jax=jax, device=device, evaluators=evaluators, operations=operations
    )
    send({"status": "ready", "jax_version": jax.__version__,
          "device": str(device), "executables": metadata})
    while True:
        command = receive()
        if command["kind"] == "close":
            send({"status": "closed"})
            return
        if command["kind"] != "evaluate":
            send({"status": "error", "error": "unknown worker command"})
            continue
        try:
            began = time.perf_counter()
            parameters = jax.device_put(command["parameters"], device)
            cases = jax.device_put(command["cases"], device)
            result, recurrent_seconds = branch(parameters, cases)
            result = jax.device_get(result)
            send({"status": "passed", "result": result,
                  "recurrent_seconds": recurrent_seconds,
                  "worker_total_seconds": time.perf_counter() - began})
        except Exception as error:
            send({"status": "error", "error_type": type(error).__name__,
                  "error": str(error)})


def serve_echo_test():
    """Exercise the same private pipe framing without importing CUDA/JAX."""
    def send(message):
        pickle.dump(message, PROTOCOL_OUTPUT, protocol=pickle.HIGHEST_PROTOCOL)
        PROTOCOL_OUTPUT.flush()

    send({"status": "ready", "mode": "echo_test"})
    while True:
        command = pickle.load(sys.stdin.buffer)
        if command["kind"] == "close":
            send({"status": "closed"})
            return
        if command["kind"] != "evaluate":
            send({"status": "error", "error": "unknown worker command"})
            continue
        send({"status": "passed", "result": command,
              "recurrent_seconds": 0.0, "worker_total_seconds": 0.0})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-profile", type=Path, required=True)
    result.add_argument("--memory-limit-bytes", type=int, required=True)
    result.add_argument("--echo-test", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        if args.echo_test:
            serve_echo_test()
        else:
            serve(args.source_profile, args.memory_limit_bytes)
