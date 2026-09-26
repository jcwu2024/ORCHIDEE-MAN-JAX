"""Measure direct loading of saved split executables, without tracing a model."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from scripts.hpc.training_artifact_io import _atomic_json


def run(args):
    import jax
    from jax.experimental.serialize_executable import deserialize_and_load

    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.run_shared_transformer_admission import memory_sizes

    _require_supported_jax(jax)
    source = json.loads(args.profile.read_text(encoding="utf-8"))
    if source["status"] != "passed" or set(source["executables"]) != {"daily", "T7", "T30"}:
        raise ValueError("requires the completed three-component split profile")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("select one visible GPU matching the source executable layout")
    report = {
        "status": "running", "source_profile": str(args.profile),
        "scope": "deserialize_only_no_Teacher_read_no_forward_no_gradient_validation",
        "device": str(devices[0]), "jax_version": jax.__version__, "components": {},
        "scientific_acceptance": False,
    }
    started = time.perf_counter()
    executables = {}

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(args.output, report)
        print(f"serialized_load stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        for name in ("daily", "T7", "T30"):
            emit("read_serialized", component=name)
            began = time.perf_counter()
            path = Path(source["executables"][name]["executable"])
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            read_seconds = time.perf_counter() - began
            emit("deserialize", component=name)
            began = time.perf_counter()
            executable = deserialize_and_load(*payload)
            load_seconds = time.perf_counter() - began
            executables[name] = executable
            report["components"][name] = {
                "read_seconds": read_seconds, "deserialize_seconds": load_seconds,
                "memory": memory_sizes(executable), "device_memory": devices[0].memory_stats(),
            }
            del payload
            emit("loaded", component=name)
        emit("completed", status="passed", resident_executable_count=len(executables))
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
