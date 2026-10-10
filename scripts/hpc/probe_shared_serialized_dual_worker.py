"""Private pipe worker for trusted T7/T30 binaries on one visible GPU.

The parent launches this module with a CUDA_VISIBLE_DEVICES value before this
module imports JAX. Standard output is exclusively the pickle message stream;
diagnostics and exceptions go to standard error.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import contextlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

PROTOCOL_OUTPUT = sys.stdout.buffer

# The immutable checkout can contain older diagnostic helpers. Select the
# staged overlays explicitly, just as the parent does.
def load_overlays():
    root = os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    if root not in sys.path:
        sys.path.insert(0, root)
    for name in ("training_artifact_io", "probe_shared_dual", "probe_shared_serialized_updates"):
        module_name = f"scripts.hpc.{name}"
        spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def serve(profile_path, memory_limit_bytes, admission_reports=None):
    load_overlays()
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load

    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.probe_shared_dual import make_recurrent_branch
    from scripts.hpc.run_shared_learning import make_tree_operations, memory_sizes
    from scripts.hpc.probe_shared_serialized_updates import validate_admissions

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
    admissions = {}
    if admission_reports:
        reference = json.loads(Path(profile["technical_update_report"]).read_text(encoding="utf-8"))
        admissions = validate_admissions(
            profile, reference, admission_reports,
            profile_sha256=hashlib.sha256(Path(profile_path).read_bytes()).hexdigest(),
        )
    evaluators, metadata = {}, {}
    for name in ("T7", "T30"):
        path = Path(admissions.get(name, profile["executables"][name])["executable"])
        print(f"recurrent_worker stage=read_saved_gradient component={name}", file=sys.stderr, flush=True)
        began = time.perf_counter()
        cpu_began = time.process_time()
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        read_seconds = time.perf_counter() - began
        read_cpu_seconds = time.process_time() - cpu_began
        print(f"recurrent_worker stage=deserialize_saved_gradient component={name} "
              f"read_seconds={read_seconds:.3f} read_cpu_seconds={read_cpu_seconds:.3f}",
              file=sys.stderr, flush=True)
        load_began = time.perf_counter()
        cpu_began = time.process_time()
        executable = deserialize_and_load(*payload)
        deserialize_seconds = time.perf_counter() - load_began
        deserialize_cpu_seconds = time.process_time() - cpu_began
        del payload
        memory = memory_sizes(executable)
        if memory["estimated_live_bytes"] > memory_limit_bytes:
            raise ValueError(f"saved worker {name} executable exceeds memory limit")
        evaluators[int(name[1:])] = executable
        metadata[name] = {"source_executable": str(path), "memory": memory,
                          "admission": admissions.get(name),
                          "load_seconds": time.perf_counter() - began,
                          "read_seconds": read_seconds,
                          "read_cpu_seconds": read_cpu_seconds,
                          "deserialize_seconds": deserialize_seconds,
                          "deserialize_cpu_seconds": deserialize_cpu_seconds}
        print(f"recurrent_worker stage=saved_gradient_loaded component={name} "
              f"deserialize_seconds={deserialize_seconds:.3f} "
              f"deserialize_cpu_seconds={deserialize_cpu_seconds:.3f}",
              file=sys.stderr, flush=True)
    operations = make_tree_operations(jax, jnp, 32)
    branches = {
        horizons: make_recurrent_branch(
            jax=jax, device=device,
            evaluators={h: evaluators[h] for h in horizons}, operations=operations,
            horizons=horizons, preserve_auxiliary=True,
        )
        for horizons in ((7, 30), (30,))
    }
    send({"status": "ready", "jax_version": jax.__version__,
          "device": str(device), "executables": metadata,
          "source_profile": str(profile_path), "admitted_overrides": admissions})
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
            horizons = tuple(sorted(command["cases"]))
            if horizons not in branches:
                raise ValueError("worker supports exactly T7+T30 or T30")
            parameters = jax.device_put(command["parameters"], device)
            cases = jax.device_put(command["cases"], device)
            jax.block_until_ready((parameters, cases))
            input_seconds = time.perf_counter() - began
            result, recurrent_seconds = branches[horizons](parameters, cases)
            download_started = time.perf_counter()
            result = jax.device_get(result)
            download_seconds = time.perf_counter() - download_started
            send({"status": "passed", "result": result,
                  "executed_parameter_version": command.get("parameter_version"),
                  "recurrent_seconds": recurrent_seconds,
                  "worker_total_seconds": time.perf_counter() - began,
                  "timing_seconds": {"worker_input_device_put_and_sync": input_seconds,
                                     "worker_result_device_get": download_seconds},
                  "device_memory": device.memory_stats()})
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
    for flag in ("daily", "seven", "thirty"):
        result.add_argument(f"--{flag}-admission-report", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        if args.echo_test:
            serve_echo_test()
        else:
            paths = {name: getattr(args, attribute) for name, attribute in (
                ("daily", "daily_admission_report"), ("T7", "seven_admission_report"),
                ("T30", "thirty_admission_report"))}
            if any(paths.values()) and not all(paths.values()):
                raise ValueError("worker requires all three admission reports")
            serve(args.source_profile, args.memory_limit_bytes, paths if all(paths.values()) else {})
