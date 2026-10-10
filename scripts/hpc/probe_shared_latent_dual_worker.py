"""Private pipe T30 full-state AD worker; stdout is exclusively pickle framing."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time


def load_parent():
    root = os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    if root not in sys.path:
        sys.path.insert(0, root)
    name = "scripts.hpc.probe_shared_latent_dual"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("probe_shared_latent_dual.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_command(jax, device, branch, command):
    """Use exactly the received current tree/case and return its version token."""
    if command.get("kind") != "evaluate" or set(command.get("cases", {})) != {30}:
        raise ValueError("worker evaluates exactly the full current T30 case")
    if not isinstance(command.get("parameter_version"), int) or command["parameter_version"] < 0:
        raise ValueError("worker requires explicit nonnegative current parameter version")
    began = time.perf_counter()
    parameters = jax.device_put(command["parameters"], device)
    cases = jax.device_put(command["cases"], device)
    jax.block_until_ready((parameters, cases))
    upload = time.perf_counter() - began
    result, seconds = branch(parameters, cases)
    download_started = time.perf_counter()
    result = jax.device_get(result)
    download = time.perf_counter() - download_started
    return {"status": "passed", "result": result,
            "executed_parameter_version": command["parameter_version"],
            "recurrent_seconds": seconds, "worker_total_seconds": time.perf_counter() - began,
            "timing_seconds": {"worker_input_device_put_and_sync": upload, "worker_result_device_get": download},
            "device_memory": device.memory_stats()}


def serve(args, receive, send):
    parent = load_parent()
    manifest = parent.read_manifest(args.manifest)
    readiness = parent.validate_manifest_asset_readiness(manifest)
    if readiness["status"] != "ready_for_saved_graph_load":
        raise FileNotFoundError("worker rejected full-state assets before GPU setup: "
                                + json.dumps(readiness, sort_keys=True))
    # Backend registration precedes JAX NumPy/training/diagnostic imports.
    # Receiving a pickled prepared case can itself import training modules,
    # so even the first receive must follow backend initialization.
    runtime = parent.load_adjacent("orcjax_gpu_runtime")
    jax, devices = runtime.initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("full-state T30 worker requires one visible GPU")
    import jax.numpy as jnp
    parent.load_runtime_overlays()
    from scripts.hpc.probe_shared_dual import make_recurrent_branch
    from scripts.hpc.run_shared_learning import make_tree_operations
    initial = receive()
    if initial.get("kind") != "initialize":
        raise ValueError("worker initial ABI arguments required before load")
    device = devices[0]
    operations = make_tree_operations(jax, jnp, 32)
    cache = {}
    def select(architecture, command):
        if architecture not in ("original", "candidate"):
            raise ValueError("worker explicit architecture selection required")
        if architecture not in cache:
            parameters = jax.device_put(command["parameters"], device)
            case = jax.device_put(command["case"], device)
            evaluators, metadata = parent.load_components(jax, manifest, architecture, parameters,
                {"cases": {30: case}}, command.get("identity"), components=("T30",))
            evaluator = evaluators["T30"]
            branch = parent.stateful_branch(make_recurrent_branch(jax=jax, device=device,
                evaluators={30: evaluator}, operations=operations, horizons=(30,),
                preserve_auxiliary=True), evaluator)
            cache[architecture] = branch, metadata
        return cache[architecture]
    active = args.architecture
    branch, metadata = select(active, initial)
    del initial
    send({"status": "ready", "architecture": active, "device": str(device),
          "executables": metadata, "model_gradient_compilations": 0})
    while True:
        command = receive()
        if command.get("kind") == "close":
            send({"status": "closed"})
            return
        try:
            if command.get("kind") == "select":
                active = command["architecture"]
                branch, metadata = select(active, command)
                send({"status": "ready", "architecture": active, "device": str(device),
                      "executables": metadata, "model_gradient_compilations": 0})
            else:
                response = evaluate_command(jax, device, branch, command)
                parent.require_memory(response["device_memory"], label="worker")
                response["executed_architecture"] = active
                send(response)
                del response
        except Exception as error:
            send({"status": "error", "error_type": type(error).__name__, "error": str(error)})
        finally:
            del command


def serve_echo(receive, send):
    """Real subprocess framing test without JAX, CUDA or model artifacts."""
    initial = receive()
    if initial.get("kind") != "initialize":
        raise ValueError("echo worker initialization required")
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
    result.add_argument("--architecture", choices=("original", "candidate"), required=True)
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
