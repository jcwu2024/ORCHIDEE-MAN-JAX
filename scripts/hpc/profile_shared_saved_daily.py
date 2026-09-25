"""Profile admitted saved daily executables without compiling another graph."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(args):
    import jax
    from jax.experimental.serialize_executable import deserialize_and_load

    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.run_shared_learning import load_learning_checkpoint

    _require_supported_jax(jax)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("select one GPU for saved logical-cuda0 executables")
    admission = json.loads(args.admission_report.read_text(encoding="utf-8"))
    if admission["status"] != "passed":
        raise ValueError("profile requires passed actual daily admission")
    with args.prepared_context.open("rb") as handle:
        context = pickle.load(handle)
    identity = context["identity"]
    checkpoint = load_learning_checkpoint(
        Path(identity["initial_checkpoint"]),
        expected_method=identity["method"],
    )
    parameters = jax.device_put(checkpoint["parameters"])
    daily = jax.device_put(context["daily4"])
    del checkpoint
    report = {
        "status": "running",
        "scope": "saved_daily_micro4_profile_no_compile_no_update",
        "identity": identity,
        "jax_version": jax.__version__,
        "variants": {},
    }
    started = time.perf_counter()

    def emit(stage):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"saved_daily_profile stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    for name, path in (
        ("original", context["daily_executable"]),
        ("expert128", admission["candidate_executable"]),
    ):
        emit(f"load_{name}")
        with Path(path).open("rb") as handle:
            serialized = pickle.load(handle)
        began = time.perf_counter()
        executable = deserialize_and_load(*serialized)
        metadata = {"load_seconds": time.perf_counter() - began, "executable": str(path)}
        del serialized
        result = jax.block_until_ready(executable(parameters, daily))
        (loss, auxiliary), gradient = jax.device_get(result)
        hard = {key: int(np.asarray(value)) for key, value in auxiliary["hard"].items()}
        if (
            not np.isfinite(loss)
            or any(hard.values())
            or not all(np.all(np.isfinite(value)) for value in jax.tree_util.tree_leaves(gradient))
        ):
            raise ValueError(f"saved {name} daily result failed finite/physical checks")
        metadata.update(loss=float(loss), hard_counts=hard)
        del result, gradient
        warm = []
        for _ in range(3):
            began = time.perf_counter()
            jax.block_until_ready(executable(parameters, daily))
            warm.append(time.perf_counter() - began)
        metadata["warm_seconds"] = warm
        report["variants"][name] = metadata
        emit(f"warm_{name}_done")
        trace_dir = args.output.with_suffix(".traces") / name
        with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
            with jax.profiler.TraceAnnotation(f"saved_daily_{name}"):
                jax.block_until_ready(executable(parameters, daily))
        metadata["trace_directory"] = str(trace_dir)
        emit(f"trace_{name}_done")
        del executable
    report["status"] = "passed"
    emit("completed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-context", type=Path, required=True)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
