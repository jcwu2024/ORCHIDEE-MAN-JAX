"""Bounded same-executable native VJP stability and strict leaf replay."""
from __future__ import annotations
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--screen-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--routes", nargs="+", default=("expert0", "expert5"))
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load
    jax.config.update("jax_enable_x64", True)
    if len(devices) != 1 or jax.default_backend() != "gpu":
        raise ValueError("requires singleGPU")
    from research.daily_coarse_graining.daily_operator import shared_transformer
    owner = load(Path(__file__).with_name("probe_shared_normalized_keys.py"), "normalized_screen_owner")
    analyzer = load(Path(__file__).with_name("analyze_shared_normalized_key_results.py"), "normalized_analyzer")
    screen = json.loads(args.screen_report.read_text())
    micro = screen["microbatch"]
    with args.payload.open("rb") as stream:
        saved = pickle.load(stream)
    blocks = jax.device_put(saved["parameters"])
    report = {"status": "running", "microbatch": micro, "routes": {}, "rtol": 1e-5, "atol": 1e-6,
              "new_model_gradient_compilations": 0, "screen": str(args.screen_report)}
    started = time.perf_counter()
    for route in args.routes:
        records = sorted((r for r in saved["records"] if r["route"] == route), key=lambda r: r["sample"])
        selected = records[:1] if micro == 1 else records
        def stack(*values):
            return None if values[0] is None else np.stack(values)
        values = jax.device_put(jax.tree_util.tree_map(stack, *((r["query"], r["dynamic"], *r["banks"]) for r in selected)))
        shape = values[0].shape
        cot = jnp.sin(jnp.arange(np.prod(shape), dtype=jnp.float32) * 0.11).reshape(shape)
        reference = {}
        for name in ("reference", "normalized"):
            row = screen["results"][route]["variants"][name]
            with Path(row["result_artifact"]).open("rb") as stream:
                artifact = pickle.load(stream)
            if artifact["identity"] != saved["identity"]:
                raise ValueError("result identity drift")
            reference[name] = artifact["result"]
        results = {}
        for name in ("reference", "normalized"):
            path = args.screen_report.with_suffix(f".{route}.{name}.executable.pkl")
            with path.open("rb") as stream:
                executable = deserialize_and_load(*pickle.load(stream))
            first = None
            comparisons = []
            for repeat in range(args.repeats):
                actual = jax.device_get(jax.block_until_ready(executable(blocks, values, cot)))
                if first is None:
                    first = actual
                    owner.write(args.output.with_suffix(f".{route}.{name}.result.pkl"), {"identity": saved["identity"], "result": actual})
                comparisons.append({"repeat": repeat,
                                    "against_same_process_first_result": analyzer.compare(actual, first),
                                    "against_screen_saved_result": analyzer.compare(actual, reference[name]),
                                    "candidate_against_reference_saved_result": analyzer.compare(actual, reference["reference"])})
            results[name] = comparisons
            del executable
        report["routes"][route] = results
        report["elapsed_seconds"] = time.perf_counter() - started
        owner.write(args.output, report)
        print(route + "_replayed", flush=True)
    report.update(status="completed", elapsed_seconds=time.perf_counter() - started)
    owner.write(args.output, report)


if __name__ == "__main__":
    main()
