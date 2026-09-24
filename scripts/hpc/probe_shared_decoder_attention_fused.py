"""Bounded CPU screen of one production-shaped split decoder attention chunk."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_coarse_graining.daily_operator.shared_transformer import (  # noqa: E402
    _attention_from_projected_segments,
    _project_key_value,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (  # noqa: E402
    _residual_block_parameters,
)


def load_candidate(path):
    name = "research.daily_coarse_graining.daily_operator.shared_decoder_attention_fused"
    if path is None:
        module = __import__(name, fromlist=["fused_attention_from_projected_segments"])
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load candidate module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.fused_attention_from_projected_segments


def compare(actual, expected, *, rtol, atol):
    leaves = zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True)
    failed = count = 0
    maximum = 0.0
    for new, old in leaves:
        new, old = np.asarray(new), np.asarray(old)
        failed += int(np.count_nonzero(~np.isclose(new, old, rtol=rtol, atol=atol)))
        count += old.size
        maximum = max(maximum, float(np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))))
    return {"passed": failed == 0, "coordinates": count, "failed": failed, "max_abs": maximum}


def run(width: int, rows: int, repeats: int, candidate, *, require_gpu: bool):
    if width % 8 or min(width, rows, repeats) <= 0:
        raise ValueError("width must be divisible by eight and all dimensions positive")
    devices = jax.devices()
    if require_gpu and not all(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, found {devices}")
    parameters = _residual_block_parameters(jax.random.PRNGKey(20260925), width, 4 * width)["attention"]
    keys = jax.random.split(jax.random.PRNGKey(11), 4)
    payload = {
        name: jax.random.normal(key, shape, dtype=jnp.float32)
        for (name, shape), key in zip(
            (
                ("query", (rows, 1, width)),
                ("dynamic", (rows, 3, width)),
                ("family", (20, width)),
                ("process", (1, width)),
            ),
            keys,
            strict=True,
        )
    }
    active = jnp.arange(rows) % 7 != 0
    masks = {
        "family": jnp.broadcast_to(active[:, None, None], (rows, 1, 20)),
        "dynamic": jnp.broadcast_to(active[:, None, None], (rows, 1, 3)),
        "process": active[:, None, None],
    }

    def objective(helper, trainable, inputs):
        family_key, family_value = _project_key_value(trainable, inputs["family"])
        dynamic_key, dynamic_value = _project_key_value(trainable, inputs["dynamic"])
        process_key, process_value = _project_key_value(trainable, inputs["process"])
        output = helper(
            trainable,
            inputs["query"],
            inputs["dynamic"],
            dynamic_key=dynamic_key,
            dynamic_value=dynamic_value,
            family_key=family_key,
            family_value=family_value,
            family_mask=masks["family"],
            dynamic_mask=masks["dynamic"],
            process_key=process_key,
            process_value=process_value,
            process_mask=masks["process"],
            carbon_key=None,
            carbon_value=None,
            carbon_mask=None,
            heads=8,
            normalization_epsilon=1e-6,
        )
        target = jnp.sin(jnp.arange(rows, dtype=jnp.float32)[:, None, None] * 0.13)
        return jnp.mean((output - target) ** 2), output

    report = {
        "scope": "one non-carbon attention chunk; no full model, Teacher or recurrence",
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "width": width,
        "rows": rows,
        "family_tokens": 20,
        "dynamic_tokens": 3,
        "heads": 8,
        "variants": {},
    }
    results = {}
    for name, helper in (("reference", _attention_from_projected_segments), ("fused", candidate)):
        function = jax.jit(jax.value_and_grad(lambda p, x: objective(helper, p, x), argnums=(0, 1), has_aux=True))
        equations = len(jax.make_jaxpr(lambda p, x: objective(helper, p, x)[0])(parameters, payload).jaxpr.eqns)
        start = time.perf_counter()
        lowered = function.lower(parameters, payload)
        lower_seconds = time.perf_counter() - start
        start = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter() - start
        results[name] = executable(parameters, payload)
        jax.block_until_ready(results[name])
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = executable(parameters, payload)
            jax.block_until_ready(result)
            timings.append((time.perf_counter() - start) * 1000.0)
        memory = executable.memory_analysis()
        report["variants"][name] = {
            "primal_jaxpr_equations": equations,
            "lower_seconds": lower_seconds,
            "compile_or_cache_load_seconds": compile_seconds,
            "median_warm_ms": float(np.median(timings)),
            "compiler_temporary_bytes": memory.temp_size_in_bytes,
            "compiler_argument_bytes": memory.argument_size_in_bytes,
        }
    ref, fused = results["reference"], results["fused"]
    report["comparison"] = {
        "loss_output": compare(fused[0], ref[0], rtol=1e-5, atol=1e-6),
        "parameter_payload_gradients": compare(fused[1], ref[1], rtol=1e-5, atol=1e-6),
    }
    report["parity_tolerances"] = {"rtol": 1e-5, "atol": 1e-6}
    report["numerical_parity_passed"] = all(value["passed"] for value in report["comparison"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate-module-file", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "shared-decoder-attention-fused-local.json")
    args = parser.parse_args()
    candidate = load_candidate(args.candidate_module_file)
    report = run(args.width, args.rows, args.repeats, candidate, require_gpu=args.require_gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["numerical_parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
