"""Bounded FP32 affine-operation discriminator, not a neural/T7 model runner.

The default fixture is artificial and matches the saved 128x64 / 512x64
contraction geometry. A captured operand bundle can instead exercise an early
tokenizer projection. No head, prepared-data loader, recurrence or physics is
constructed. The independent FP64 affine oracle is diagnostic, not a change to
the production precision or reference. B1 is one comparison route, not truth.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

SCHEMA = "T7_affine_operand_bundle_v1"
PARAMETERS_SHA256 = "4f95d18d97d2feb4dcab8e1dda6caeb242a4869f044a00b6a7ae6bc689806c14"
CONTEXT_SHA256 = "ff3fdabe755d599bc2ceee72087c03983d9a629add93166519dcda1f56be9026"
ARTIFACT_IDS = [742, 275, 482, 299]
RTOL, ATOL = 1.0e-5, 1.0e-6
ARRAY_KEYS = ("x", "kernel", "bias", "direction", "cotangent", "mask")


def synthetic_operands(seed=20261011):
    """One fixed, nondegenerate method fixture; no search over errors/shapes."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.25, (4, 128, 64)).astype(np.float32)
    direction = np.zeros_like(x)
    direction[3] = rng.normal(0.0, 0.125, x[3].shape).astype(np.float32)
    mask = rng.random(x.shape) > 0.15
    mask[:, 0, :] = False  # Preserve raw outputs, omit these objective terms only.
    return {
        "x": x, "kernel": rng.normal(0.0, 0.125, (64, 64)).astype(np.float32),
        "bias": rng.normal(0.0, 0.05, (64,)).astype(np.float32),
        "direction": direction, "cotangent": rng.normal(size=x.shape).astype(np.float32),
        "mask": mask,
        "metadata": {"schema_version": SCHEMA, "origin": "synthetic_method_fixture",
                     "seed": int(seed), "scope": "shape-matched affine operator only"},
    }


def validate_operands(bundle):
    if set(bundle) != {*ARRAY_KEYS, "metadata"}:
        raise ValueError("exact operand keys plus metadata required")
    meta = bundle["metadata"]
    if not isinstance(meta, dict) or meta.get("schema_version") != SCHEMA:
        raise ValueError("operand schema differs")
    if meta.get("origin") not in ("synthetic_method_fixture", "captured_component_operands"):
        raise ValueError("explicit synthetic/captured provenance required")
    x, kernel, bias = (np.asarray(bundle[k]) for k in ("x", "kernel", "bias"))
    if (x.ndim != 3 or x.shape[0] != 4 or kernel.ndim != 2 or bias.ndim != 1
            or x.shape[-1] != kernel.shape[0] or bias.shape != (kernel.shape[1],)
            or min(*x.shape, *kernel.shape) <= 0):
        raise ValueError("four sample operands and compatible affine geometry required")
    expected = (4, x.shape[1], kernel.shape[1])
    for name in ARRAY_KEYS:
        value = np.asarray(bundle[name])
        if name == "mask":
            if value.dtype != np.dtype("bool") or value.shape != expected:
                raise ValueError("full Boolean output mask required")
        elif value.dtype != np.dtype("float32") or not np.all(np.isfinite(value)):
            raise ValueError("finite FP32 operation operands required: " + name)
    if bundle["direction"].shape != x.shape or bundle["cotangent"].shape != expected:
        raise ValueError("direction/cotangent geometry differs")
    if np.count_nonzero(bundle["direction"][:3]) or not np.count_nonzero(bundle["direction"][3]):
        raise ValueError("one nonzero sample-3 direction and exact other-sample zeros required")
    if not np.any(bundle["mask"]) or not np.any(bundle["cotangent"][bundle["mask"]]):
        raise ValueError("nontrivial active objective required")
    if sum(np.asarray(bundle[k]).size for k in ARRAY_KEYS) > 2_000_000:
        raise ValueError("operand bundle exceeds the bounded small-component scope")
    if meta["origin"] == "captured_component_operands":
        required = {"parameter_file_sha256": PARAMETERS_SHA256,
                    "prepared_context_sha256": CONTEXT_SHA256, "artifact_ids": ARTIFACT_IDS}
        if any(meta.get(k) != v for k, v in required.items()):
            raise ValueError("captured operand source identities differ")
        for key in ("parameter_path", "capture_boundary", "capture_source_sha256",
                    "capture_receipt_sha256", "direction_coordinate", "mask_meaning"):
            if key not in meta or not str(meta[key]).strip():
                raise ValueError("captured provenance field missing: " + key)
        if meta["direction_coordinate"] not in (434, 2382, 2621, 200):
            raise ValueError("captured state direction differs")
    return bundle


def save_operands(path, bundle):
    validate_operands(bundle)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **{k: bundle[k] for k in ARRAY_KEYS},
                        metadata=np.asarray(json.dumps(bundle["metadata"], sort_keys=True)))


def load_operands(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {*ARRAY_KEYS, "metadata"}:
            raise ValueError("operand NPZ members differ")
        bundle = {key: archive[key].copy() for key in ARRAY_KEYS}
        bundle["metadata"] = json.loads(str(archive["metadata"].item()))
    return validate_operands(bundle)


def affine_oracle(bundle):
    """Real-number affine value/JVP and all mean-objective gradients in FP64."""
    x, kernel, bias, direction, cotangent = (
        np.asarray(bundle[k], dtype=np.float64)
        for k in ("x", "kernel", "bias", "direction", "cotangent"))
    cotangent = np.where(bundle["mask"], cotangent, 0.0)
    denominator = cotangent.size
    primal = x @ kernel + bias
    return {"primal": primal, "tangent": direction @ kernel,
            "loss": np.asarray(np.sum(primal * cotangent) / denominator),
            "parameter_gradient": {"kernel": sum(a.T @ b for a, b in zip(x, cotangent)) / denominator,
                                   "bias": np.sum(cotangent, axis=(0, 1)) / denominator},
            "input_gradient": cotangent @ kernel.T / denominator,
            "mask": bundle["mask"]}


def finite_scalar(value):
    """JSON cannot represent NaN/Inf; comparison gates still reject them."""
    value = float(value)
    return value if np.isfinite(value) else None


def compare_arrays(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape:
        raise ValueError("comparison shape differs")
    if a.dtype.kind == "b" or b.dtype.kind == "b":
        exact = bool(np.array_equal(a, b))
        return {"passed": exact, "exact": exact, "shape": list(a.shape)}
    a, b = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    difference = np.full(a.shape, np.nan, dtype=np.float64)
    ratio = np.full(a.shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.subtract(a, b, out=difference, where=finite)
        np.abs(difference, out=difference)
        allowance = ATOL + RTOL * np.abs(b)
        np.divide(difference, allowance, out=ratio, where=finite)
    finite_metrics = np.isfinite(difference) & np.isfinite(allowance) & np.isfinite(ratio)
    failed = ~finite | ~finite_metrics | (difference > allowance)
    return {"passed": not bool(np.any(failed)), "shape": list(a.shape),
            "exact": bool(np.array_equal(a, b, equal_nan=True)),
            "failed_coordinates": int(np.count_nonzero(failed)),
            "nonfinite_coordinates": int(np.count_nonzero(~finite)),
            "actual_nonfinite_coordinates": int(np.count_nonzero(~np.isfinite(a))),
            "reference_nonfinite_coordinates": int(np.count_nonzero(~np.isfinite(b))),
            "derived_nonfinite_coordinates": int(np.count_nonzero(finite & ~finite_metrics)),
            "max_absolute_error": (finite_scalar(np.max(difference, initial=0))
                                   if np.all(finite) else None),
            "max_error_to_allowance": (finite_scalar(np.max(ratio, initial=0))
                                       if np.all(finite) else None)}


def compare_results(actual, reference):
    comparisons = {k: compare_arrays(actual[k], reference[k])
                   for k in ("primal", "tangent", "loss", "input_gradient", "mask")}
    comparisons.update({"parameter_gradient." + k: compare_arrays(actual["parameter_gradient"][k],
                        reference["parameter_gradient"][k]) for k in ("kernel", "bias")})
    return {"passed": all(row["passed"] for row in comparisons.values()), "comparisons": comparisons}


def make_routes(jax, linear):
    """Native JAX AD on an unchanged affine source, with shared parameters."""
    import jax.numpy as jnp

    def evaluate(operator, parameters, x, direction, cotangent, mask):
        primal, tangent = jax.jvp(lambda values: operator(parameters, values), (x,), (direction,))
        def objective(p, values):
            return jnp.mean(jnp.where(mask, operator(p, values) * cotangent, 0.0))
        loss, (parameter_gradient, input_gradient) = jax.value_and_grad(
            objective, argnums=(0, 1))(parameters, x)
        return {"primal": primal, "tangent": tangent, "loss": loss,
                "parameter_gradient": parameter_gradient, "input_gradient": input_gradient,
                "mask": mask}

    def single(*args):
        return evaluate(linear, *args)

    def batched(*args):
        return evaluate(jax.vmap(linear, in_axes=(None, 0)), *args)

    def mapped(*args):
        return evaluate(lambda p, x: jax.lax.map(lambda row: linear(p, row), x), *args)

    def merge(*rows):
        stack = lambda key: jnp.stack([row[key] for row in rows])
        return {"primal": stack("primal"), "tangent": stack("tangent"),
                "loss": jnp.mean(stack("loss")), "mask": stack("mask"),
                "input_gradient": stack("input_gradient") / 4,
                "parameter_gradient": jax.tree_util.tree_map(
                    lambda *values: jnp.mean(jnp.stack(values), axis=0),
                    *(row["parameter_gradient"] for row in rows))}
    return single, batched, mapped, merge


def independent_direction_checks(bundle, result):
    oracle = affine_oracle(bundle)
    cotangent = np.where(bundle["mask"], bundle["cotangent"], 0).astype(np.float64)
    expected = float(np.sum(oracle["tangent"] * cotangent)
                     / oracle["primal"].size)
    with np.errstate(over="ignore", invalid="ignore"):
        jvp_end = float(np.sum(np.asarray(result["tangent"], np.float64) * cotangent)
                        / oracle["primal"].size)
        vjp_end = float(np.sum(np.asarray(result["input_gradient"], np.float64) * bundle["direction"]))
    steps = []
    for step in (1.0e-2, 1.0e-3):
        x = bundle["x"].astype(np.float64)
        direction = bundle["direction"].astype(np.float64)
        kernel, bias = (bundle[k].astype(np.float64) for k in ("kernel", "bias"))
        cot = np.where(bundle["mask"], bundle["cotangent"], 0).astype(np.float64)
        plus, minus = (np.mean(((x + sign * step * direction) @ kernel + bias) * cot)
                       for sign in (1, -1))
        fd = (plus - minus) / (2 * step)
        steps.append({"step": step, "value": finite_scalar(fd), "comparison": compare_arrays(fd, expected)})
    return {"independent_oracle_pairing": finite_scalar(expected),
            "actual_JVP_end": finite_scalar(jvp_end),
            "actual_VJP_end": finite_scalar(vjp_end),
            "actual_pairing_nonfinite_endpoints": int(not np.isfinite(jvp_end)) + int(not np.isfinite(vjp_end)),
            "actual_JVP_VJP_comparison": compare_arrays(jvp_end, vjp_end),
            "JVP_oracle_comparison": compare_arrays(jvp_end, expected),
            "VJP_oracle_comparison": compare_arrays(vjp_end, expected),
            "FP64_finite_differences": steps,
            "scope": "independent affine real-number extension; no full-T7 derivative reference"}


def evaluate_measurements(bundle, measurements):
    """Check every preserved round; a later pass never replaces an earlier failure."""
    oracle = affine_oracle(bundle)
    oracle_checks, directions, isolation, pairs, repeats = [], [], [], [], []
    grouped, baseline = {}, {}
    for row in measurements:
        name, cycle, result = row["route"], row["cycle"], row["result"]
        if cycle not in grouped:
            grouped[cycle] = {}
        if name in grouped[cycle]:
            raise ValueError("duplicate route measurement in a cycle")
        grouped[cycle][name] = result
        key = {"route": name, "cycle": cycle}
        oracle_checks.append({**key, **compare_results(result, oracle)})
        directions.append({**key, **independent_direction_checks(bundle, result)})
        isolation.append({**key, "passed": bool(np.all(result["tangent"][:3] == 0))})
        if cycle == 0:
            baseline[name] = result
        else:
            if name not in baseline:
                raise ValueError("untimed baseline required before warm repeats")
            repeats.append({**key, **compare_results(result, baseline[name])})
    if set(baseline) != {"b1", "vmap", "map"}:
        raise ValueError("three route baselines required")
    for cycle, rows in grouped.items():
        if set(rows) != set(baseline):
            raise ValueError("every cycle requires all three routes")
        pairs.extend({"route": name, "cycle": cycle, **compare_results(rows[name], rows["b1"])}
                     for name in ("vmap", "map"))
    passed = (all(row["passed"] for row in oracle_checks + isolation + pairs + repeats)
              and all(row[key]["passed"] for row in directions
                      for key in ("actual_JVP_VJP_comparison", "JVP_oracle_comparison", "VJP_oracle_comparison"))
              and all(step["comparison"]["passed"] for row in directions for step in row["FP64_finite_differences"]))
    return {"passed": passed, "measurement_count": len(measurements),
            "oracle_comparisons": oracle_checks, "route_comparisons": pairs,
            "within_route_repeat_comparisons": repeats, "independent_direction_checks": directions,
            "zero_other_sample_tangents": isolation}


def result_arrays(result):
    return {**{k: v for k, v in result.items() if k != "parameter_gradient"},
            **{"parameter_gradient." + k: v for k, v in result["parameter_gradient"].items()}}


def save_measurement_evidence(path, bundle, measurements):
    """Preserve every raw round before reporting numerical acceptance/failure."""
    arrays = {f"cycle{row['cycle']}.{row['route']}.{key}": value for row in measurements
              for key, value in result_arrays(row["result"]).items()}
    arrays.update({"oracle." + key: value for key, value in result_arrays(affine_oracle(bundle)).items()})
    np.savez_compressed(path, **arrays)
    acceptance = evaluate_measurements(bundle, measurements)
    return {"status": "passed" if acceptance["passed"] else "numerical_failed", **acceptance,
            "results_container": str(path), "preserved_array_count": len(arrays)}


def save_partial_measurements(path, bundle, measurements):
    """Atomic snapshot of completed calls, retained if a supervisor stops us."""
    arrays = {f"cycle{row['cycle']}.{row['route']}.{key}": value
              for row in measurements for key, value in result_arrays(row["result"]).items()}
    arrays.update({"oracle." + key: value for key, value in result_arrays(affine_oracle(bundle)).items()})
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def initialize_probe_runtime(backend):
    """Use the accepted deployed bootstrap before numpy/model JAX imports."""
    os.environ["JAX_PLATFORMS"] = "cpu" if backend == "cpu" else "cuda"
    if backend == "gpu":
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
        jax, devices = initialize_gpu_backend()
    else:
        import jax
        devices = tuple(jax.devices("cpu"))
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    return jax, devices


def hlo_summary(text):
    rows = [line.strip() for line in text.splitlines()
            if "custom-call(" in line and "gemm" in line or " = " in line and " dot(" in line]
    return {"line_count": len(text.splitlines()), "contraction_count": len(rows),
            "bounded_contractions": rows[:16],
            "selected_algorithms": sorted(set(re.findall(r'"selected_algorithm":"?(-?\d+)', text))),
            "scope": "this small compiled operation only; not the original neural/T7 HLO"}


def run(args):
    if not 1 <= args.repeats <= 5:
        raise ValueError("one to five warm repeats required; no search")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    companion = lambda suffix: output.with_suffix(suffix)
    destinations = [output, companion(".results.npz"), companion(".partial.npz")]
    destinations += [companion("." + name + ".optimized-hlo.txt.gz") for name in ("b1", "vmap", "map")]
    if not args.operands:
        destinations.append(companion(".operands.npz"))
    if any(path.exists() for path in destinations):
        raise FileExistsError("use a new output stem; existing evidence is read-only")
    bundle = load_operands(args.operands) if args.operands else validate_operands(synthetic_operands(args.seed))
    if not args.operands:
        save_operands(companion(".operands.npz"), bundle)
    operand_path = Path(args.operands).resolve() if args.operands else companion(".operands.npz")
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    jax, devices = initialize_probe_runtime(args.backend)
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _linear

    device = devices[0]
    if device.platform != args.backend:
        raise RuntimeError("requested execution backend unavailable")
    parameters = jax.device_put({k: bundle[k] for k in ("kernel", "bias")})
    dynamic = jax.device_put(tuple(bundle[k] for k in ("x", "direction", "cotangent", "mask")))
    single, batched, mapped, merge = make_routes(jax, _linear)
    compile_rows, binaries = {}, {}
    one_args = (parameters, *(array[3] for array in dynamic))
    batch_args = (parameters, *dynamic)
    for name, function, values in (("b1", single, one_args), ("vmap", batched, batch_args),
                                   ("map", mapped, batch_args)):
        start = time.perf_counter()
        lowered = jax.jit(function).lower(*values)
        lower_seconds = time.perf_counter() - start
        start = time.perf_counter()
        binary = lowered.compile()
        backend_seconds = time.perf_counter() - start
        binaries[name] = binary
        text = binary.as_text()
        with gzip.open(companion("." + name + ".optimized-hlo.txt.gz"), "wt", encoding="utf-8") as stream:
            stream.write(text)
        compile_rows[name] = {"lower_seconds": lower_seconds, "backend_seconds": backend_seconds,
                              "HLO": hlo_summary(text)}
    single_arguments = [(parameters, *(array[i] for array in dynamic)) for i in range(4)]
    initial = [binaries["b1"](*values) for values in single_arguments]
    start = time.perf_counter()
    lowered_merge = jax.jit(merge).lower(*initial)
    merge_lower_seconds = time.perf_counter() - start
    start = time.perf_counter()
    merger = lowered_merge.compile()
    merge_backend_seconds = time.perf_counter() - start
    call = {"b1": lambda: merger(*(binaries["b1"](*values) for values in single_arguments)),
            "vmap": lambda: binaries["vmap"](*batch_args), "map": lambda: binaries["map"](*batch_args)}
    block = lambda tree: jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), tree)
    measurements, timings = [], {name: [] for name in call}
    for name in call:
        measurements.append({"route": name, "cycle": 0,
                             "result": jax.device_get(block(call[name]()))})
        save_partial_measurements(companion(".partial.npz"), bundle, measurements)
    for cycle in range(args.repeats):
        for name in tuple(call)[cycle % 3:] + tuple(call)[:cycle % 3]:
            start = time.perf_counter()
            result = block(call[name]())
            timings[name].append(time.perf_counter() - start)
            measurements.append({"route": name, "cycle": cycle + 1,
                                 "result": jax.device_get(result)})
            save_partial_measurements(companion(".partial.npz"), bundle, measurements)
    evidence = save_measurement_evidence(companion(".results.npz"), bundle, measurements)
    source = inspect.getsource(_linear)
    report = {"schema_version": "T7_affine_numerics_probe_v1",
              "backend": device.platform, "device": str(device), "jax_version": jax.__version__,
              "python": sys.executable, "operand_path": str(operand_path),
              "operand_sha256": hashlib.sha256(operand_path.read_bytes()).hexdigest(),
              "provenance": bundle["metadata"], "geometry": {k: list(bundle[k].shape) for k in ARRAY_KEYS},
              "source": {"file": inspect.getsourcefile(_linear), "function": _linear.__qualname__,
                         "sha256": hashlib.sha256(source.encode()).hexdigest()},
              "tolerances": {"rtol": RTOL, "atol": ATOL}, **evidence, "compilation": compile_rows,
              "model_operator_compiles": 3, "additional_small_serial_merge_compiles": 1,
              "total_small_graph_compiles": 4,
              "serial_merge_compilation": {"lower_seconds": merge_lower_seconds,
                                           "backend_seconds": merge_backend_seconds},
              "warm_seconds": {name: {"samples": values, "median": float(np.median(values))}
                               for name, values in timings.items()},
              "precision": "unchanged source FP32 affine operands/results/AD; FP64 host oracle only",
              "source_provenance_verification": "declared capture identities; full source files not rehashed by this probe",
              "inference_limit": "One affine component, with masks retained. Artificial operands verify the discriminator only; "
              "CPU parity cannot exclude a GPU kernel issue. Captured operand provenance and original T7 gates remain separate. "
              "No B1 truth, production repair, physics, recurrence, optimizer update or training speed acceptance follows."}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operands", type=Path)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "backend": report["backend"],
                      "origin": report["provenance"]["origin"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
