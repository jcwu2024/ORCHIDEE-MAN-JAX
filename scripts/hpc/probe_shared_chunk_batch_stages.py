"""Real expert5 causal screens of changed chunk-batch arithmetic; no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

from probe_shared_command_buffer_loops import comparison, deadline, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing diagnostic")
    started = time.perf_counter()
    report = {"status": "running", "scope": "real_expert5_first_query_stages_fixed_original_inputs",
              "rtol": 1e-5, "atol": 1e-6, "full_model_admission": False}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **fields)
        write(args.output, report)
        print(stage, flush=True)
    emit("initializing")
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, unused_devices = initialize_gpu_backend()
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    with args.payload.open("rb") as stream:
        saved = pickle.load(stream)
    row = next(r for r in saved["records"] if r["route"] == "expert5" and r["sample"] == 0)
    if row["query"].shape != (7542, 1, 384) or row["chunk"] != 128:
        raise ValueError("actual expert5 scope differs")
    p = jax.device_put(saved["parameters"][0]["attention"])
    query = jax.device_put(jnp.pad(row["query"], ((0,10),(0,0),(0,0))).reshape(59,128,1,384))
    config = saved["config"]
    key = st._project_key_value(p, jax.device_put(row["banks"][0]))[0]
    key = key.reshape(20,8,48).swapaxes(-3,-2)
    key = st._stable_head_l2_normalize(key, config.normalization_epsilon)
    stages = {}
    def screen(name, function, values, source):
        outputs = {}
        stages[name] = {"source": source, "arms": {}}
        for arm in ("map", "vmap"):
            mapper = jax.lax.map if arm == "map" else lambda f,x: jax.vmap(f)(x)
            def apply(x):
                return mapper(function, x)
            lower = jax.jit(apply).lower(values)
            began = time.perf_counter()
            with deadline(180, f"stage_backend_{name}_{arm}", emit):
                executable = lower.compile(compiler_options={"xla_gpu_graph_min_graph_size":5,
                    "xla_gpu_enable_llvm_module_compilation_parallelism":False})
            outputs[arm] = jax.device_get(jax.block_until_ready(executable(values)))
            stages[name]["arms"][arm] = {"backend_seconds": time.perf_counter()-began,
                "finite": bool(np.all(np.isfinite(outputs[arm])))}
            args.output.with_suffix(f".{name}.{arm}.hlo.txt").write_text(executable.as_text())
            write(args.output.with_suffix(f".{name}.{arm}.result.pkl"), outputs[arm])
        stages[name]["comparison"] = comparison(jax, outputs["vmap"], outputs["map"])
        emit("stage_measured", stages=stages)
        return jax.device_put(outputs["map"])
    projected = screen("query_projection", lambda q: st._linear(p["query"], q), query,
        "shared_transformer._attention_from_projected_segments:465; strong_baseline_architecture._linear:887")
    def head_norm(q):
        return st._stable_head_l2_normalize(q.reshape(q.shape[:-1]+(8,48)).swapaxes(-3,-2), config.normalization_epsilon)
    normalized = screen("head_normalization", head_norm, projected,
        "shared_transformer._attention_from_projected_segments:481; strong_baseline_architecture._stable_head_l2_normalize")
    screen("family_score", lambda q: jnp.clip(jnp.einsum("...hqd,hkd->...hqk",q,key),-1.0,1.0)*jnp.sqrt(jnp.asarray(48,dtype=jnp.float32)), normalized,
        "shared_transformer._attention_from_projected_segments.scores:497-499")
    emit("completed", status="first_query_stages_measured", stages=stages, identity=saved["identity"],
         input_rows=7542, padded_rows=7552, chunk_size=128, chunk_count=59,
         interpretation="each stage compares identical original-map inputs; diagnostic outputs may change compiler fusion")
    print(json.dumps({k:v["comparison"] for k,v in stages.items()}),flush=True)


if __name__ == "__main__":
    main()
