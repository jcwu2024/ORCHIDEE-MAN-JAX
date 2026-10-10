"""Small native linear forward/VJP capability boundaries, not model accuracy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-policy", action="store_true")
    parser.add_argument("--shape-report", type=Path)
    parser.add_argument("--only-vjp", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior probe")
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, _devices = initialize_gpu_backend()
    import jax.numpy as jnp
    if args.matrix_policy:
        from shared_daily_precision_matrix_policy import algorithm_linear
    else:
        from probe_shared_neural_precision import algorithm_linear
    cases = [((384,), (384, 384)), ((128, 2), (2, 384)), ((128, 384), (384, 1)),
             ((128, 384), (384, 192)), ((128, 192), (192, 1)),
             ((1, 2), (2, 384)), ((1, 384), (384, 384)),
             ((4, 128, 1, 384), (384, 384))]
    if args.shape_report is not None:
        previous = json.loads(args.shape_report.read_text())
        cases = [(tuple(row["lhs"]), tuple(row["kernel"]))
                 for row in previous["precision_policy_evidence"]["observed_trace_shapes"]]
    report = {"status": "running", "scope": "bounded_shape_capability_not_public_loss", "variants": [],
              "training_updates": 0, "architecture_changed": False, "synthetic_cotangent": True,
              "rank_one_matrix_policy": args.matrix_policy,
              "actual_shape_report": str(args.shape_report) if args.shape_report else None}
    def emit():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    for shape, kernel_shape in cases:
        p = {"kernel": jnp.full(kernel_shape, .01, jnp.float32), "bias": jnp.full(kernel_shape[1], .01, jnp.float32)}
        x = jnp.full(shape, .1, jnp.float32)
        cotangent = jnp.ones(shape[:-1] + (kernel_shape[1],), jnp.float32)
        def vjp(p, x, cotangent):
            value, pullback = jax.vjp(algorithm_linear, p, x)
            return value, pullback(cotangent)
        row = {"input_shape": list(shape), "kernel_shape": list(kernel_shape)}
        functions = (("native_vjp", vjp, (p, x, cotangent)),) if args.only_vjp else (("forward", algorithm_linear, (p, x)), ("native_vjp", vjp, (p, x, cotangent)))
        for name, function, arguments in functions:
            began = time.perf_counter()
            try:
                result = jax.device_get(jax.block_until_ready(jax.jit(function)(*arguments)))
                row[name] = {"status": "supported", "elapsed_seconds": time.perf_counter() - began,
                             "all_finite": all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(result))}
            except Exception as error:
                row[name] = {"status": "unsupported_or_error", "error": str(error), "error_type": type(error).__name__,
                             "elapsed_seconds": time.perf_counter() - began}
        report["variants"].append(row)
        emit()
        print(json.dumps(row), flush=True)
    report["status"] = "capability_measured"
    emit()


if __name__ == "__main__":
    main()
