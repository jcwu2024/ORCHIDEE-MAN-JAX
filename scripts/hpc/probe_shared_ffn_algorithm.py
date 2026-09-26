"""Native JAX dot-algorithm capability screen; never a custom adjoint.

This explores forward and native-transpose arithmetic only, not training or
precision adoption. Algorithm policies round both multiply operands to FP16,
accumulate/output FP32, and preserve FP32 master/bias/GELU storage. FP32-storage
policies avoid a JAX-visible half-gradient return boundary, but also quantize
native backward multiply operands: this is a distinct numerical treatment.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

POLICIES = ("fp32", "cast_default", "cast_preset", "fp32_storage_preset", "fp32_storage_algorithm")


def atomic_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def algorithm_ffn(parameters, value, *, policy):
    import jax
    import jax.numpy as jnp
    if policy not in POLICIES:
        raise ValueError("unknown native dot policy")
    def linear(p, x):
        storage = jnp.float16 if policy.startswith("cast_") else jnp.float32
        x, w = jnp.asarray(x, storage), jnp.asarray(p["kernel"], storage)
        precision = None
        if policy.endswith("preset"):
            precision = jax.lax.DotAlgorithmPreset.F16_F16_F32
        elif policy.endswith("algorithm"):
            precision = jax.lax.DotAlgorithm(np.float16, np.float16, np.float32)
        return jax.lax.dot_general(x, w, (((x.ndim - 1,), (0,)), ((), ())),
                                   precision=precision, preferred_element_type=jnp.float32) + p["bias"]
    return linear(parameters["second"], jax.nn.gelu(linear(parameters["first"], value)))


def native_vjp(parameters, value, cotangent, *, policy):
    import jax
    result, pullback = jax.vjp(lambda p, x: algorithm_ffn(p, x, policy=policy), parameters, value)
    return result, pullback(cotangent)


def source_evidence():
    import jax
    from jax._src.lax import lax as internal
    evidence = {}
    for name in ("_dot_general_transpose_lhs", "get_algorithm_compute_types", "_dot_general_lower"):
        function = getattr(internal, name)
        lines, first = inspect.getsourcelines(function)
        selected = [{"line": first + i, "text": line.strip()[:240]} for i, line in enumerate(lines)
                    if any(token in line for token in ("precision=precision", "isinstance(algorithm", "precision_type",
                        "always cast", "get_algorithm_compute_types", "lhs = maybe_convert", "rhs = maybe_convert",
                        "accumulation_aval", "not supported", "_convert_element_type(x_bar"))]
        evidence[name] = {"file": inspect.getsourcefile(function), "first_line": first, "selected_lines": selected}
    return {"jax_version": jax.__version__, "functions": evidence,
            "preset_available": hasattr(jax.lax.DotAlgorithmPreset, "F16_F16_F32"),
            "algorithm_fields": list(jax.lax.DotAlgorithm._fields)}


def ir_evidence(lowered, executable):
    stable = str(lowered.compiler_ir(dialect="stablehlo"))
    lines = [line.strip() for line in stable.splitlines() if "stablehlo.dot_general" in line]
    optimized = executable.as_text()
    kernel_lines = [line.strip()[:700] for line in optimized.splitlines() if "custom_call_target=" in line]
    return {"all_dot_type_signatures": dict(Counter(line.rsplit(" : ", 1)[-1] for line in lines)),
            "dot_count": len(lines), "algorithm_dot_count": sum("algorithm =" in line for line in lines),
            "dot_samples": [line[:1100] for line in lines[:12]],
            "optimized_custom_call_count": len(kernel_lines), "optimized_custom_call_samples": kernel_lines[:12],
            "tensor_core_execution_proven": False,
            "note": "Algorithm/lowering evidence is not an instruction-level Tensor-Core execution proof."}


def error_metrics(jax, actual, reference):
    failed = finite_failures = 0
    delta2 = reference2 = 0.0
    maximum = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(reference), strict=True):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        finite_failures += int(np.count_nonzero(~finite))
        failed += int(np.count_nonzero(~(finite & np.isclose(a, b, rtol=1e-5, atol=1e-6))))
        delta = a[finite] - b[finite]
        delta2 += float(np.sum(delta * delta))
        reference2 += float(np.sum(b[finite] ** 2))
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
    return {"exact_route_passed": failed == 0, "failed_coordinates": failed, "nonfinite_coordinates": finite_failures,
            "max_abs": maximum if finite_failures == 0 else None,
            "relative_l2": float(np.sqrt(delta2 / max(reference2, 1e-300))) if finite_failures == 0 else None}


def capability(args, emit):
    import jax
    import jax.numpy as jnp
    # Tiny actual-width FFN, not a performance result or synthetic giant GEMM.
    rng = np.random.default_rng(20260926)
    p = {name: {"kernel": jnp.asarray(rng.normal(0, 0.02, shape), jnp.float32),
                "bias": jnp.linspace(0.01, 0.02, shape[1], dtype=jnp.float32)}
         for name, shape in (("first", (384, 1536)), ("second", (1536, 384)))}
    x = jnp.asarray(rng.normal(size=(4, 2, 1, 384)), jnp.float32)
    cot = jnp.sin(jnp.arange(x.size, dtype=jnp.float32) * 0.11).reshape(x.shape)
    emit("source_inspected", source_evidence=source_evidence(), input_shape=list(x.shape))
    rows, results = {}, {}
    for policy in POLICIES:
        emit("before_compile", policy=policy, variants=rows)
        began = time.perf_counter()
        try:
            function = jax.jit(lambda p, x, cot: native_vjp(p, x, cot, policy=policy))
            lowered = function.lower(p, x, cot)
            executable = lowered.compile()
            compile_seconds = time.perf_counter() - began
            result = jax.device_get(jax.block_until_ready(executable(p, x, cot)))
            results[policy] = result
            rows[policy] = {"status": "supported", "compile_seconds": compile_seconds,
                            "result_dtypes": [str(np.asarray(v).dtype) for v in jax.tree_util.tree_leaves(result)],
                            "ir": ir_evidence(lowered, executable),
                            "vs_fp32": error_metrics(jax, result, results["fp32"]),
                            "vs_cast_default": error_metrics(jax, result, results.get("cast_default", result)),
                            "forward_vs_cast_default": error_metrics(jax, result[0], results.get("cast_default", result)[0])}
            if args.trace_dir is not None and policy == args.policy:
                trace_path = args.trace_dir / policy
                try:
                    jax.profiler.start_trace(str(trace_path))
                    try:
                        jax.block_until_ready(executable(p, x, cot))
                    finally:
                        jax.profiler.stop_trace()
                    rows[policy]["device_trace"] = {"status": "captured", "path": str(trace_path),
                        "note": "Separate instrumented execution, not a warm timing. Kernel review still required."}
                except Exception as error:
                    rows[policy]["device_trace"] = {"status": "unavailable", "error": str(error)[:1000]}
        except Exception as error:
            rows[policy] = {"status": "unsupported_or_error", "elapsed_seconds": time.perf_counter() - began,
                            "error_type": type(error).__name__, "error": str(error)[:2000]}
        emit("policy_completed", policy=policy, variants=rows)
    return {"status": "capability_measured", "variants": rows, "synthetic_cotangent": True,
            "scope": "tiny_actual_width_FFN_native_VJP_capability_only_not_a_speed_measurement",
            "numerical_treatment": "native_backward_products_also_quantized_by_algorithm_policy"}


def real_fragment(args, emit):
    """Reuse saved producer and unchanged scheduling with only a scoped FFN hook."""
    import jax
    capability_report = json.loads(args.capability_report.read_text(encoding="utf-8"))
    if capability_report.get("status") != "capability_measured" or capability_report.get("jax_version") != jax.__version__:
        raise ValueError("completed capability report from this exact JAX runtime required")
    if capability_report["variants"][args.policy]["status"] != "supported":
        raise ValueError("selected native algorithm was not admitted by capability compile")
    helper_name = "scripts.hpc.probe_shared_expert_real_daily"
    helper_spec = importlib.util.spec_from_file_location(helper_name, args.real_daily_probe_file)
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_name] = helper
    helper_spec.loader.exec_module(helper)
    spec = importlib.util.spec_from_file_location("shared_ffn_precision_diagnostic", args.ffn_probe_file)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    original = probe.ffn
    def candidate(p, x, *, mixed):
        return algorithm_ffn(p, x, policy=args.policy) if mixed else original(p, x, mixed=False)
    probe.ffn = candidate
    try:
        result = probe.fragment_run(args, emit)
    finally:
        probe.ffn = original
    result.update(native_algorithm_policy=args.policy,
                  numerical_treatment="forward_and_native_backward_operand_rounding_to_FP16_with_FP32_accumulation",
                  gradients="native_AD_no_custom_rule_no_STE_FP32_storage_boundary")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("capability", "fragment"), default="capability")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--policy", choices=("fp32_storage_preset", "fp32_storage_algorithm"), default="fp32_storage_preset")
    p.add_argument("--capability-report", type=Path)
    p.add_argument("--payload", type=Path)
    p.add_argument("--ffn-probe-file", type=Path)
    p.add_argument("--real-daily-probe-file", type=Path)
    p.add_argument("--microbatch", type=int, choices=(1, 4), default=4)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--memory-limit-gib", type=int, default=12)
    p.add_argument("--trace-dir", type=Path,
                   help="Optional tiny device trace for the selected supported algorithm policy")
    return p


def run(args):
    import jax
    report = {"status": "running", "jax_version": jax.__version__, "training_updates": 0,
              "scientific_acceptance": False, "precision_adopted": False,
              "devices": [str(d) for d in jax.devices()], "mode": args.mode}
    started = time.perf_counter()
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        atomic_json(args.output, report)
        print(stage, flush=True)
    def expired():
        emit("timeout", status="timeout")
        os._exit(124)
    timeout = threading.Timer(120 if args.mode == "capability" else 600, expired)
    timeout.daemon = True
    timeout.start()
    try:
        if args.require_gpu and (len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu"):
            raise RuntimeError("one visible GPU required")
        if args.mode == "fragment" and any(getattr(args, key) is None for key in (
                "capability_report", "payload", "ffn_probe_file", "real_daily_probe_file")):
            raise ValueError("fragment requires explicit capability-report/payload/ffn-probe-file/real-daily-probe-file")
        inputs = [getattr(args, key) for key in ("capability_report", "payload", "ffn_probe_file", "real_daily_probe_file")]
        if args.output.resolve() in [path.resolve() for path in inputs if path is not None]:
            raise ValueError("output collides with input")
        result = capability(args, emit) if args.mode == "capability" else real_fragment(args, emit)
        emit("completed", **result)
        return report
    finally:
        timeout.cancel()


if __name__ == "__main__":
    run(parser().parse_args())
