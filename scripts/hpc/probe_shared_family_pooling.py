"""Bounded FP32 source-family pooling comparison; no Teacher or model weights.

This isolates the broadcast/reduction used by shared_transformer._source_family_bank.
Compiled fragment memory and timings are not full-model performance estimates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def pool(tokens, family, process_mask, *, family_count, contraction):
    selected = family[None, None, :] == jnp.arange(family_count)[None, :, None]
    selected &= process_mask[:, None, :]
    count = jnp.sum(selected, axis=2, dtype=jnp.float32)
    if contraction:
        total = jnp.einsum(
            "efs,sw->efw",
            selected.astype(jnp.float32),
            tokens,
            precision=jax.lax.Precision.HIGHEST,
        )
    else:
        total = jnp.sum(
            selected[..., None].astype(jnp.float32) * tokens[None, None, :, :],
            axis=2,
            dtype=jnp.float32,
        )
    valid = count > 0
    return jnp.where(valid[..., None], total / jnp.maximum(count[..., None], 1), 0), valid


def run_case(*, microbatch, width, sources, experts, families, repetitions):
    rng = np.random.default_rng(20260923)
    tokens = jnp.asarray(rng.normal(size=(microbatch, sources, width)), dtype=jnp.float32)
    # Last family is empty; one expert and selected individual sources are masked.
    family = jnp.asarray(np.arange(sources) % (families - 1), dtype=jnp.int32)
    mask = rng.random((microbatch, experts, sources)) > 0.3
    mask[:, -1, :] = False
    mask[:, :, ::13] = False
    mask = jnp.asarray(mask)
    weights = jnp.asarray(rng.normal(size=(microbatch, experts, families, width)), dtype=jnp.float32)
    report = {
        "microbatch": microbatch,
        "width": width,
        "sources": sources,
        "experts": experts,
        "families": families,
        "variants": {},
    }
    outputs = {}
    for name, contraction in (("broadcast", False), ("contraction", True)):

        def loss(x, m, w):
            y, valid = pool(x, family, m, family_count=families, contraction=contraction)
            return jnp.sum(jnp.sin(y) * w), (y, valid)

        function = jax.jit(jax.vmap(jax.value_and_grad(loss, has_aux=True)))
        started = time.perf_counter()
        executable = function.lower(tokens, mask, weights).compile()
        compile_seconds = time.perf_counter() - started
        result = jax.block_until_ready(executable(tokens, mask, weights))
        memory = executable.memory_analysis()
        times = []
        for _ in range(repetitions):
            started = time.perf_counter()
            jax.block_until_ready(executable(tokens, mask, weights))
            times.append(time.perf_counter() - started)
        outputs[name] = jax.device_get(result)
        report["variants"][name] = {
            "compile_seconds": compile_seconds,
            "warm_median_seconds": float(np.median(times)),
            "warm_seconds": times,
            "argument_bytes": memory.argument_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes,
            "temporary_bytes": memory.temp_size_in_bytes,
        }
    ((ref_loss, (ref_y, ref_valid)), ref_grad) = outputs["broadcast"]
    ((new_loss, (new_y, new_valid)), new_grad) = outputs["contraction"]
    parity = {}
    for label, ref, new in (
        ("loss", ref_loss, new_loss),
        ("forward", ref_y, new_y),
        ("token_gradient", ref_grad, new_grad),
    ):
        difference = np.asarray(new, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        parity[label] = {
            "max_abs": float(np.max(np.abs(difference))),
            "relative_l2": float(
                np.linalg.norm(difference.ravel())
                / max(np.linalg.norm(np.asarray(ref, dtype=np.float64).ravel()), 1e-30)
            ),
            "passed": bool(np.allclose(ref, new, rtol=2e-5, atol=2e-6)),
        }
    parity["valid_mask_equal"] = bool(np.array_equal(ref_valid, new_valid))
    parity["masked_source_gradient_zero"] = bool(np.all(new_grad[:, ::13, :] == 0))
    parity["empty_expert_zero"] = bool(np.all(new_y[:, -1, :, :] == 0))
    parity["empty_family_zero"] = bool(np.all(new_y[:, :, -1, :] == 0))
    report["parity"] = parity
    report["passed"] = all(v["passed"] if isinstance(v, dict) else v for v in parity.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--sources", type=int, default=393)
    parser.add_argument("--experts", type=int, default=7)
    parser.add_argument("--families", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    report = {
        "jax_version": jax.__version__,
        "devices": [str(x) for x in jax.devices()],
        "scope": "synthetic pooling forward+native token VJP only; not full model",
        "cases": [],
    }
    for microbatch in args.microbatches:
        case = run_case(
            microbatch=microbatch,
            width=args.width,
            sources=args.sources,
            experts=args.experts,
            families=args.families,
            repetitions=args.repetitions,
        )
        report["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(case), flush=True)
    if not all(case["passed"] for case in report["cases"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
