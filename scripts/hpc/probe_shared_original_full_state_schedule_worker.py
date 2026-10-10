"""Original T7/T30 full-state AD worker; stdout is private pickle framing."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
from pathlib import Path
import pickle
import sys
import time


def load_parent():
    root = os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    if root not in sys.path:
        sys.path.insert(0, root)
    name = "scripts.hpc.probe_shared_original_full_state_schedule"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("probe_shared_original_full_state_schedule.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_command(jax, device, branches, command):
    """Execute received current weights and full cases at the echoed version."""
    horizons = frozenset(command.get("cases", {}))
    if command.get("kind") != "evaluate" or horizons not in (frozenset((7, 30)), frozenset((30,))):
        raise ValueError("original worker accepts exactly full T7+T30 or T30")
    version = command.get("parameter_version")
    if type(version) is not int or version < 0:
        raise ValueError("explicit nonnegative integer current parameter version required")
    began = time.perf_counter()
    parameters = jax.device_put(command["parameters"], device)
    cases = jax.device_put(command["cases"], device)
    jax.block_until_ready((parameters, cases))
    upload = time.perf_counter()-began
    result, seconds = branches[horizons](parameters, cases)
    started = time.perf_counter()
    result = jax.device_get(result)
    download = time.perf_counter()-started
    return {"status": "passed", "result": result, "executed_parameter_version": version,
            "recurrent_seconds": seconds, "worker_total_seconds": time.perf_counter()-began,
            "timing_seconds": {"worker_input_device_put_and_sync": upload,
                               "worker_result_device_get": download},
            "device_memory": device.memory_stats()}


def serve(args, receive, send):
    parent = load_parent()
    # Prepared-case unpickling can import training modules: register CUDA first.
    jax, devices = parent.load_adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("original full-state worker requires one visible GPU")
    import jax.numpy as jnp
    parent.load_overlays()
    from scripts.hpc.probe_shared_dual import make_recurrent_branch
    from scripts.hpc.run_shared_learning import make_tree_operations
    manifest = parent.read_manifest(args.manifest)
    _profile, reference = parent.read_reference(manifest)
    parent.validate_original_assets(manifest, reference)
    initial = receive()
    if initial.get("kind") != "initialize" or set(initial.get("cases", {})) != {7, 30}:
        raise ValueError("original worker requires both full initial ABI cases")
    expected_version = manifest.get("resume", {}).get("completed_pairs", 0)
    if type(initial.get("parameter_version")) is not int or initial["parameter_version"] != expected_version:
        raise ValueError("worker initial parameter version differs from declared resume step")
    device = devices[0]
    parameters, cases = jax.device_put((initial["parameters"], initial["cases"]), device)
    operations = make_tree_operations(jax, jnp, 32)
    evaluators, metadata = parent.load_evaluators(jax, manifest, reference, parameters,
        {"cases": cases}, ("T7", "T30"))
    branches = {}
    for horizons in ((7, 30), (30,)):
        selected = {f"T{h}": evaluators[f"T{h}"] for h in horizons}
        branches[frozenset(horizons)] = parent.full_state_branch(make_recurrent_branch(
            jax=jax, device=device, evaluators={h: evaluators[f"T{h}"] for h in horizons},
            operations=operations, horizons=horizons, preserve_auxiliary=True), selected)
    del initial, parameters, cases
    memory = device.memory_stats()
    parent.require_memory(memory, "worker both recurrent executables resident")
    send({"status": "ready", "device": str(device), "executables": metadata,
          "device_memory": memory, "model_gradient_compilations": 0,
          "resident_components": ["T7", "T30"], "initial_parameter_version": expected_version})
    while True:
        command = receive()
        if command.get("kind") == "close":
            send({"status": "closed"})
            return
        try:
            response = evaluate_command(jax, device, branches, command)
            parent.require_memory(response["device_memory"], "worker aggregate both binaries resident")
            send(response)
            del response
        except Exception as error:
            send({"status": "error", "error_type": type(error).__name__, "error": str(error)})
        finally:
            del command


def serve_echo(receive, send):
    """Exercise actual subprocess framing without JAX/CUDA/artifact loading."""
    initial = receive()
    if initial.get("kind") != "initialize":
        raise ValueError("echo initialization required")
    send({"status": "ready", "mode": "echo_test"})
    while True:
        command = receive()
        if command.get("kind") == "close":
            send({"status": "closed"})
            return
        send({"status": "passed", "result": command,
              "executed_parameter_version": command.get("parameter_version"),
              "recurrent_seconds": 0., "worker_total_seconds": 0.})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--echo-test", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    output = sys.stdout.buffer
    def send(message):
        pickle.dump(message, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.flush()
    def receive():
        return pickle.load(sys.stdin.buffer)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            (serve_echo(receive, send) if args.echo_test else serve(args, receive, send))
    except Exception as error:
        send({"status": "error", "error_type": type(error).__name__, "error": str(error)})
        raise
