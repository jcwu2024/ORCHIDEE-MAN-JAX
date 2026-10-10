"""Cheap original head-layout normalization hypothesis on captured key banks."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import re
import time

import numpy as np

from probe_shared_normalized_keys_recurrent import load


def canonical_hlo(source):
    source = re.sub(r", metadata=\{[^}]*\}", "", source)
    source = re.sub(r"^HloModule \S+", "HloModule MODULE", source)
    names = {}
    def replace(match):
        name = match.group(0)
        if name not in names:
            names[name] = f"%node{len(names)}"
        return names[name]
    return re.sub(r"%[A-Za-z0-9_.$-]+", replace, source)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--patch-owner", type=Path, required=True)
    p.add_argument("--mode", choices=("screen",), default="screen")
    p.add_argument("--microbatch", type=int, choices=(1,), default=1)
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    nk = load(Path(__file__).with_name("shared_normalized_key_execution.py"), "shared_normalized_key_execution")
    owner = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    component = load(Path(__file__).with_name("probe_shared_normalized_keys.py"), "normalized_head_order_component_owner")
    if len(devices) != 1 or args.output.exists():
        raise ValueError("oneGPU and preserved new output required")
    with args.payload.open("rb") as stream:
        saved = pickle.load(stream)
    config = saved["config"]
    def head_first(value, unused_config):
        if value is None:
            return None
        original_shape = value.shape
        if value.ndim == 1:
            value = value[None, :]
        heads = config.resolved_decoder_attention_heads
        shaped = value.reshape(value.shape[:-1] + (heads, value.shape[-1] // heads)).swapaxes(-3, -2)
        return st._stable_head_l2_normalize(shaped, config.normalization_epsilon).swapaxes(-3, -2).reshape(original_shape)
    current = nk.normalize_key
    report = {"status": "running", "scope": "captured_key_projection_and_normalization_native_VJP_layout_hypothesis",
              "compiler_options": {}, "source": "shared_transformer._attention_from_projected_segments per_head then stable L2",
              "rtol": 1e-5, "atol": 1e-6, "full_model_admission": False, "banks": [], "identity": saved["identity"]}
    seen = set()
    for record in saved["records"]:
        if record["sample"] != 0 or record["route"] not in ("expert0", "expert4"):
            continue
        for bank_index, host_bank in enumerate(record["banks"]):
            if host_bank is None or tuple(host_bank.shape) in seen:
                continue
            seen.add(tuple(host_bank.shape))
            bank = jax.device_put(host_bank)
            parameters = jax.device_put(saved["parameters"][0]["attention"]["key"])
            cot = jnp.sin(jnp.arange(host_bank.shape[0] * config.resolved_decoder_width, dtype=jnp.float32) * 0.11).reshape((host_bank.shape[0], config.resolved_decoder_width))
            results, hlo = {}, {}
            for name, normalizer in (("current", current), ("head_first", head_first)):
                def kernel(parameters, inputs, cotangent):
                    values, pullback = jax.vjp(lambda p, x: normalizer(st._linear(p, x), config), parameters, inputs)
                    return values, pullback(cotangent)
                executable = jax.jit(kernel).lower(parameters, bank, cot).compile()
                results[name] = jax.device_get(jax.block_until_ready(executable(parameters, bank, cot)))
                hlo[name] = executable.as_text()
                args.output.with_suffix(f".bank{len(report['banks'])}.{name}.hlo.txt").write_text(hlo[name])
            canonical = {name: canonical_hlo(value) for name, value in hlo.items()}
            report["banks"].append({"route": record["route"], "bank_index": bank_index, "raw_bank_shape": host_bank.shape,
                                    "canonical_optimized_hlo_identical": canonical["current"] == canonical["head_first"],
                                    "hlo_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in canonical.items()},
                                    "strict_comparison": owner.compare_trees(jax, results["head_first"], results["current"]),
                                    "bitwise_values_and_gradients_identical": all(np.array_equal(a, b) for a,b in zip(jax.tree_util.tree_leaves(results["current"]), jax.tree_util.tree_leaves(results["head_first"]), strict=True))})
            owner.atomic_json(args.output, report)
    same = all(row["canonical_optimized_hlo_identical"] and row["bitwise_values_and_gradients_identical"] for row in report["banks"])
    if same:
        report.update(status="head_order_canonicalized_no_component_expansion", conclusion="Explicit head-first normalization cancels to the same optimized key-bank value/VJP graph; no evidence to recompile T7.")
    else:
        nk.normalize_key = head_first
        args.variant, args.routes = "normalized", ["expert0", "expert4"]
        def emit(stage, **fields):
            report.update(stage=stage, **fields)
            owner.atomic_json(args.output, report)
            print(stage, flush=True)
        try:
            report["component_screen"] = component.screen(args, emit)
        finally:
            nk.normalize_key = current
        report.update(status="bounded_components_measured", normalization_layout="head_first_then_inverse_transpose")
    report.update(elapsed_seconds=time.perf_counter() - started)
    owner.atomic_json(args.output, report)


if __name__ == "__main__":
    main()
