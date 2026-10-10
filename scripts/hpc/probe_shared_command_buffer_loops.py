"""Bounded original-FP32 command-buffer capability and real decoder checks.

WHILE support and FUSION minimum-group-size are separate treatments. Preserve
normalization, arithmetic, loop unrolling, native AD and LLVM parallelism false.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import sys
import threading
import time

import numpy as np

BASE_TYPES = "FUSION,CUBLAS,CUSTOM_CALL,CUDNN"
WHILE_TYPES = BASE_TYPES + ",WHILE"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        with temp.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


@contextmanager
def deadline(seconds, label, emit):
    def expired():
        emit("timeout", status="timeout", timeout_scope=label, limit_seconds=seconds)
        os._exit(124)
    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


@contextmanager
def forbid_component_compile(enabled):
    if not enabled:
        yield
        return
    from jax._src import compiler
    original = compiler.compile_or_get_cached
    def forbidden(*unused_args, **unused_kwargs):
        raise RuntimeError("paired saved-binary replay attempted new compilation")
    compiler.compile_or_get_cached = forbidden
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original


def compiler_options(name):
    return {"xla_gpu_enable_command_buffer": BASE_TYPES if name == "default" else WHILE_TYPES,
            "xla_gpu_enable_llvm_module_compilation_parallelism": False}


def comparison(jax, candidate, reference):
    failures, count, maximum = 0, 0, 0.0
    bitwise = True
    for a, b in zip(jax.tree_util.tree_leaves(candidate), jax.tree_util.tree_leaves(reference), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("comparison shape/dtype mismatch")
        error = np.abs(a.astype(np.float64) - b.astype(np.float64))
        valid = np.isfinite(a) & np.isfinite(b) & (error <= 1e-6 + 1e-5 * np.abs(b))
        failures += int(np.count_nonzero(~valid))
        count += a.size
        maximum = max(maximum, float(np.max(error, initial=0)))
        bitwise &= np.array_equal(a, b)
    return {"passed": failures == 0, "failed_coordinates": failures, "coordinates": count,
            "max_abs_error": maximum, "bitwise_identical": bool(bitwise)}


def runtime_library_evidence(args, *, before_initialization=False):
    """Verify actually loaded library paths/API values without cuInit calls."""
    manifest = None
    if args.runtime_profile == "compat":
        if args.runtime_manifest is None:
            raise ValueError("candidate runtime requires its verified manifest")
        manifest = json.loads(args.runtime_manifest.read_text())
        if manifest.get("schema") != "shared_cuda_capture_runtime_v1" or manifest.get("status") != "cpu_versions_verified":
            raise ValueError("candidate runtime lacks verified CPU-only library manifest")
    result = {}
    for label, basename, function in (("libcuda", "libcuda.so", "cuDriverGetVersion"), ("libcudart", "libcudart.so", "cudaRuntimeGetVersion")):
        expected = None if manifest is None else manifest["libraries"][label]
        if expected is not None:
            path = Path(expected["path"]).resolve()
            root = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/toolchains").resolve()
            if not path.is_relative_to(root) or not path.is_file() or expected["api_version"] < 12030:
                raise ValueError("candidate library outside owned toolchains or below12.3")
        else:
            paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if line.split() and Path(line.split()[-1]).name.startswith(basename)}
            if len(paths) != 1:
                raise ValueError(f"ambiguous loaded legacy {label}: {sorted(paths)}")
            path = Path(paths.pop()).resolve()
        library = ctypes.CDLL(str(path))
        entry = getattr(library, function)
        entry.argtypes, entry.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        version = ctypes.c_int()
        status = entry(ctypes.byref(version))
        maps = {str(Path(line.split()[-1]).resolve()) for line in Path("/proc/self/maps").read_text().splitlines()
                if line.split() and Path(line.split()[-1]).name.startswith(basename)}
        if status != 0 or maps != {str(path)} or (expected is not None and version.value != expected["api_version"]):
            raise ValueError(f"runtime library API/source mismatch {label}: status{status},version{version.value},maps{maps}")
        result[label] = {"path": str(path), "api_version": version.value, "loaded_paths": sorted(maps)}
    return {"profile": args.runtime_profile, "before_gpu_initialization": before_initialization,
            "manifest": None if args.runtime_manifest is None else str(args.runtime_manifest), "libraries": result}


def support_replay(args, jax, emit):
    from jax.experimental.serialize_executable import deserialize_and_load
    from jax._src import compiler
    x = jax.device_put(np.arange(59 * 128 * 32, dtype=np.float32).reshape(59, 128, 32) * 1e-5)
    p = jax.device_put(np.full((32,), 0.7, dtype=np.float32))
    cot = jax.device_put(np.sin(np.arange(x.size, dtype=np.float32) * 0.11).reshape(x.shape))
    compiled, first, rows, hlos = {}, {}, {}, {}
    for name, report_path in (("default", args.default_report), ("while", args.while_report)):
        report = json.loads(report_path.read_text())
        if report["status"] != "support_arm_measured" or report["jax_version"] != jax.__version__:
            raise ValueError("support arm absent or incompatible")
        with report_path.with_suffix(f".support.{name}.executable.pkl").open("rb") as stream:
            compiled[name] = deserialize_and_load(*pickle.load(stream))
        hlos[name] = report_path.with_suffix(f".support.{name}.hlo.txt").read_text()
        memory = compiled[name].memory_analysis()
        rows[name] = {"warm_seconds": [], "self_comparisons": [], "memory": {
            key: int(getattr(memory, key)) for key in ("argument_size_in_bytes", "temp_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes")}}
    emit("support_replay_loaded", zero_compile=True,
         optimized_hlo_identical=hlos["default"] == hlos["while"],
         optimized_hlo_sha256={name: hashlib.sha256(hlo.encode()).hexdigest() for name,hlo in hlos.items()})
    original = compiler.compile_or_get_cached
    def forbidden(*unused_args, **unused_kwargs):
        raise RuntimeError("zero-compile replay attempted compilation")
    compiler.compile_or_get_cached = forbidden
    pairs = []
    try:
        for name in ("default", "while"):
            first[name] = jax.device_get(jax.block_until_ready(compiled[name](p, x, cot)))
        for repeat in range(7):
            results = {}
            for name in (("default", "while") if repeat % 2 == 0 else ("while", "default")):
                began = time.perf_counter()
                value = jax.block_until_ready(compiled[name](p, x, cot))
                rows[name]["warm_seconds"].append(time.perf_counter() - began)
                results[name] = jax.device_get(value)
                rows[name]["self_comparisons"].append(comparison(jax, results[name], first[name]))
            pairs.append(comparison(jax, results["while"], results["default"]))
    finally:
        compiler.compile_or_get_cached = original
    for row in rows.values():
        row["warm_median_seconds"] = float(np.median(row["warm_seconds"]))
    emit("support_replay_completed", status="unsupported_WHILE_capture_on_current_runtime", variants=rows,
         pair_comparisons=pairs, strict_passed=all(row["passed"] for row in pairs),
         warm_speedup=rows["default"]["warm_median_seconds"] / rows["while"]["warm_median_seconds"],
         full_model_admission=False, device_memory=jax.devices()[0].memory_stats(),
         stop_reason="Current binary removes WHILE capture for toolkit12.1 / driver12.0; HLO remains unchanged. No real-model compile justified.")


def support(args, jax, emit):
    import jax.numpy as jnp
    import jaxlib
    from jax._src import compiler
    from jax.experimental.serialize_executable import serialize
    # Keep the nanobind parent alive while reading its borrowed DebugOptions.
    metadata_options = compiler.get_compile_options(1, 1)
    debug = metadata_options.executable_build_options.debug_options
    emit("runtime_metadata", jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
         debug_options=str(debug), debug_attributes=[name for name in dir(debug) if "command_buffer" in name or "llvm_module" in name])
    # 59 iterations match expert5's original chunk128 loop count. The nonlinear
    # loop also admits native reverse AD; this is support evidence, not a model.
    x = jax.device_put(np.arange(59 * 128 * 32, dtype=np.float32).reshape(59, 128, 32) * 1e-5)
    p = jax.device_put(np.full((32,), 0.7, dtype=np.float32))
    cot = jax.device_put(np.sin(np.arange(x.size, dtype=np.float32) * 0.11).reshape(x.shape))
    def forward(p, x):
        return jax.lax.map(lambda value: jnp.sin(value * p) + value * 0.03, x)
    def combined(p, x, cot):
        value, pullback = jax.vjp(forward, p, x)
        return value, pullback(cot)
    lower_started = time.perf_counter()
    lowered = jax.jit(combined).lower(p, x, cot)
    emit("support_lowered", lower_seconds=time.perf_counter() - lower_started)
    results, compiled, rows = {}, {}, {}
    names = ("default", "fusion2") if args.mode == "threshold-support" else (args.arm,)
    for name in names:
        options = {"xla_gpu_enable_llvm_module_compilation_parallelism": False}
        if args.mode == "threshold-support":
            options["xla_gpu_graph_min_graph_size"] = 5 if name == "default" else 2
        emit("support_compiling", variant=name, compiler_options=options)
        began = time.perf_counter()
        try:
            with deadline(60, f"support_backend_{name}", emit):
                executable = lowered.compile(compiler_options=options)
        except Exception as error:
            emit("support_compile_rejected", status="unsupported", variant=name,
                 error_type=type(error).__name__, error=str(error), backend_seconds=time.perf_counter() - began)
            return
        backend = time.perf_counter() - began
        began = time.perf_counter()
        try:
            result = jax.device_get(jax.block_until_ready(executable(p, x, cot)))
        except Exception as error:
            emit("support_execute_rejected", status="unsupported", variant=name,
                 error_type=type(error).__name__, error=str(error), backend_seconds=backend)
            return
        cold = time.perf_counter() - began
        hlo = executable.as_text()
        args.output.with_suffix(f".support.{name}.hlo.txt").write_text(hlo, encoding="utf-8")
        write(args.output.with_suffix(f".support.{name}.executable.pkl"), serialize(executable))
        write(args.output.with_suffix(f".support.{name}.result.pkl"), result)
        results[name], compiled[name] = result, executable
        rows[name] = {"compiler_options": options, "backend_seconds": backend, "cold_seconds": cold,
                      "hlo_while_count": hlo.count(" while("),
                      "hlo_command_buffer_call_count": len(re.findall(r"call\([^\n]*to_apply=%?command_buffer", hlo)), "warm_seconds": []}
        emit("support_variant_saved", variants=rows)
    for repeat in range(7):
        for name in (names if repeat % 2 == 0 else tuple(reversed(names))):
            began = time.perf_counter()
            jax.block_until_ready(compiled[name](p, x, cot))
            rows[name]["warm_seconds"].append(time.perf_counter() - began)
    fields = {}
    if args.mode == "threshold-support":
        fields = {"strict_comparison": comparison(jax, results["fusion2"], results["default"]),
                  "optimized_hlo_identical": compiled["fusion2"].as_text() == compiled["default"].as_text()}
    emit("support_completed", status="threshold_support_measured" if args.mode == "threshold-support" else "support_arm_measured", variants=rows,
         all_finite=all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(results)),
         full_model_admission=False, **fields)


def original_component(blocks, values, masks, indices, config, chunk, *, batch_chunks=False,
                       preserve_learned_shapes=False, chunk_batch_size=None):
    """Source reuse: shared_transformer original projected-segment block runner.

    Diagnostic copies the original expert128 packing and native lax.map from
    shared_decoder_macro_packed._decode_mapped_rows; process logic stays in st.
    """
    import jax
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    query, dynamic, family, process, carbon, axis, local = values
    projections = []
    for block in blocks:
        pairs = [st._project_key_value(block["attention"], x) if x is not None else (None, None)
                 for x in (family, process, carbon, axis, local)]
        projections.append(st._DecoderBlockProjection(*(x for pair in pairs for x in pair)))
    rows = query.shape[0]
    chunk = min(chunk, rows)
    pad = (-rows) % chunk
    def chunks(value):
        if value is None:
            return None
        padded = jnp.pad(value, ((0, pad),) + ((0, 0),) * (value.ndim - 1))
        return padded.reshape((-1, chunk) + value.shape[1:])
    packed = jax.tree_util.tree_map(chunks, (query, dynamic, masks, indices))
    def call(items):
        q, d, m, i = items
        return st._run_coordinate_blocks_reusing_projections(q, d, family_mask=m[0], dynamic_mask=m[1],
            process_mask=m[2], carbon_mask=m[3], blocks=blocks, projections=tuple(projections),
            config=config, dynamic_projection_indices=i)
    if preserve_learned_shapes:
        # Source-backed execution repair: retain each canonical learned
        # projection/parameterized normalization on its128-row shape, including
        # native reverse accumulation, and batch the remaining coordinate work.
        # No parameter broadcast/copy and no custom differentiation is added.
        from research.daily_coarse_graining.daily_operator import strong_baseline_architecture as sba
        padded_rows = rows + pad
        flat = jax.tree_util.tree_map(lambda v: None if v is None else v.reshape((-1,) + v.shape[2:]), packed)
        original_linear, original_norm = st._linear, st._layer_norm
        sba_linear, sba_norm = sba._linear, sba._layer_norm
        def linear(parameters, tokens):
            if tokens.shape[0] != padded_rows:
                return original_linear(parameters, tokens)
            shaped = tokens.reshape((-1, chunk) + tokens.shape[1:])
            result = jax.lax.map(lambda v: original_linear(parameters, v), shaped)
            return result.reshape((-1,) + result.shape[2:])
        def norm(tokens, scale, bias):
            if tokens.shape[0] != padded_rows:
                return original_norm(tokens, scale, bias)
            shaped = tokens.reshape((-1, chunk) + tokens.shape[1:])
            result = jax.lax.map(lambda v: original_norm(v, scale, bias), shaped)
            return result.reshape((-1,) + result.shape[2:])
        try:
            # Patches exist only during construction of this diagnostic graph.
            st._linear = sba._linear = linear
            st._layer_norm = sba._layer_norm = norm
            return call(flat)[:rows]
        finally:
            st._linear, st._layer_norm = original_linear, original_norm
            sba._linear, sba._layer_norm = sba_linear, sba_norm
    mapped = (jax.vmap(call)(packed) if batch_chunks else
              jax.lax.map(call, packed, batch_size=chunk_batch_size))
    return mapped.reshape((-1,) + query.shape[1:])[:rows]


def threshold_component(args, jax, emit):
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    from jax.experimental.serialize_executable import serialize
    # Import canonical source eagerly; never create physical lookup tables under
    # a tracing context. No normalized-bank adapter is imported or patched.
    with args.payload.open("rb") as stream:
        saved = pickle.load(stream)
    if saved["schema"] != "shared_normalized_keys_real_v1":
        raise ValueError("requires preserved actual raw-bank capture")
    config = saved["config"]
    if (config.resolved_decoder_width, config.resolved_decoder_attention_heads, config.coordinate_blocks) != (384, 8, 2):
        raise ValueError("requires original 384/eight-head/two-block architecture")
    blocks = jax.device_put(saved["parameters"])
    parameter_count = sum(np.size(a) for a in jax.tree_util.tree_leaves(blocks))
    if parameter_count != 3_548_928:
        raise ValueError("decoder parameter scope drift")
    records = sorted((row for row in saved["records"] if row["route"] == args.route), key=lambda row: row["sample"])
    if [row["sample"] for row in records] != [0, 1, 2, 3] or records[0]["chunk"] != 128:
        raise ValueError("requires original distinct four-sample expert128 capture")
    selected = records[:args.microbatch]
    if args.mode in ("chunk-vmap", "shape-preserved") and (args.route != "expert5" or args.microbatch != 1 or records[0]["query"].shape != (7542, 1, 384)):
        raise ValueError("chunk-vmap first-stage scope is exact expert5/micro1")
    def stack(*values):
        return None if values[0] is None else np.stack(values)
    x = jax.device_put(jax.tree_util.tree_map(stack, *((r["query"], r["dynamic"], *r["banks"]) for r in selected)))
    masks = jax.device_put(jax.tree_util.tree_map(stack, *(r["masks"] for r in selected)))
    indices = jax.device_put(jax.tree_util.tree_map(stack, *(r["indices"] for r in selected)))
    # Use JAX sin exactly as in the original component diagnostic. This is the
    # unchanged synthetic cotangent, never an actual training/science loss.
    cot = jnp.sin(jnp.arange(np.prod(x[0].shape), dtype=jnp.float32) * 0.11).reshape(x[0].shape)
    def forward(p, values):
        return jax.vmap(lambda v,m,i: original_component(p, v, m, i, config, 128,
            batch_chunks=args.mode == "chunk-vmap", preserve_learned_shapes=args.mode == "shape-preserved"))(values, masks, indices)
    def combined(p, values, cotangent):
        output, pullback = jax.vjp(forward, p, values)
        return output, pullback(cotangent)
    def compare_scope(candidate, reference):
        return {"output": comparison(jax, candidate[0], reference[0]),
                "all_decoder_parameter_vjp": comparison(jax, candidate[1][0], reference[1][0]),
                "all_captured_raw_bank_and_input_vjp": comparison(jax, candidate[1][1], reference[1][1])}
    rows, compiled, initial = {}, {}, {}
    emit("component_inputs_ready", identity=saved["identity"], route=args.route, microbatch=args.microbatch,
         decoder_parameter_coordinates=parameter_count, input_shape=x[0].shape, chunk=128,
         cotangent="original_jax_sin0.11_all_output_coordinates_unscaled", full_model_admission=False)
    treatment_names = (("default", 5), ("vmap" if args.mode == "chunk-vmap" else "preserve128", 5)) if args.mode in ("chunk-vmap", "shape-preserved") else (((args.arm, 5),) if args.mode == "runtime-component" else ((("default", 5), ("while", 5)) if args.mode == "runtime-replay" else (("default", 5), ("fusion2", 2))))
    for name, minimum in treatment_names:
        options = {"xla_gpu_graph_min_graph_size": minimum, "xla_gpu_enable_llvm_module_compilation_parallelism": False}
        backend_times = {}
        binaries = []
        if args.mode in ("chunk-vmap", "shape-preserved") and name == "default":
            from jax.experimental.serialize_executable import deserialize_and_load
            path = args.default_report
            source = json.loads(path.read_text())
            if (args.runtime_profile != "legacy" or args.arm != "default"
                    or source.get("status") != "threshold_component_measured"
                    or source.get("identity") != saved["identity"]
                    or source.get("input_shape") != list(x[0].shape)
                    or source.get("route") != args.route or source.get("microbatch") != args.microbatch
                    or source["variants"]["default"]["compiler_options"] != options
                    or source.get("cotangent") != "original_jax_sin0.11_all_output_coordinates_unscaled"):
                raise ValueError("saved original baseline differs in capture/scope/flags/cotangent")
            binary_identity = {}
            for label in ("forward", "native_vjp"):
                binary = path.with_suffix(f".default.{label}.executable.pkl")
                binary_identity[label] = {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
                with binary.open("rb") as stream:
                    binaries.append(deserialize_and_load(*pickle.load(stream)))
            lower_seconds = None
            emit("original_legacy_baseline_binary_loaded", source_compile_report=str(path),
                 saved_baseline_binaries=binary_identity, baseline_component_recompilations=0)
        elif args.mode == "runtime-replay":
            from jax.experimental.serialize_executable import deserialize_and_load
            path = args.default_report if name == "default" else args.while_report
            source = json.loads(path.read_text())
            if (source.get("status") != "runtime_component_arm_measured" or source.get("arm") != name
                    or source.get("runtime_library_evidence", {}).get("libraries") != runtime_library_evidence(args)["libraries"]
                    or source.get("identity") != saved["identity"] or source.get("input_shape") != list(x[0].shape)):
                raise ValueError("saved binaries require identical capture/runtime/input scope")
            for label in ("forward", "native_vjp"):
                with path.with_suffix(f".{name}.{label}.executable.pkl").open("rb") as stream:
                    binaries.append(deserialize_and_load(*pickle.load(stream)))
            lower_seconds = None
            emit("same_runtime_binary_loaded", variant=name, source_compile_report=str(path), zero_component_compile=True)
        else:
            began = time.perf_counter()
            lower_f = jax.jit(forward).lower(blocks, x)
            lower_v = jax.jit(combined).lower(blocks, x, cot)
            lower_seconds = time.perf_counter() - began
            emit("component_compiling", variant=name, lower_seconds=lower_seconds, compiler_options=options)
            for label, lowered in (("forward", lower_f), ("native_vjp", lower_v)):
                began = time.perf_counter()
                with deadline(180, f"component_backend_{name}_{label}", emit):
                    executable = lowered.compile(compiler_options=options)
                backend_times[label] = time.perf_counter() - began
                binaries.append(executable)
                write(args.output.with_suffix(f".{name}.{label}.executable.pkl"), serialize(executable))
                args.output.with_suffix(f".{name}.{label}.hlo.txt").write_text(executable.as_text(), encoding="utf-8")
        memory = binaries[1].memory_analysis()
        metric = {key: int(getattr(memory, key)) for key in ("argument_size_in_bytes", "temp_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes")}
        metric["estimated_live_bytes"] = sum(metric[key] for key in ("argument_size_in_bytes", "temp_size_in_bytes", "output_size_in_bytes")) - metric["alias_size_in_bytes"]
        if metric["estimated_live_bytes"] > 12 * 2**30:
            raise ValueError("component exceeds12GiB bound")
        compiled[name] = tuple(binaries)
        began = time.perf_counter()
        initial[name] = jax.device_get(jax.block_until_ready(binaries[1](blocks, x, cot)))
        cold_vjp_seconds = time.perf_counter() - began
        jax.block_until_ready(binaries[0](blocks, x))
        write(args.output.with_suffix(f".{name}.result.pkl"), {"identity": saved["identity"], "result": initial[name]})
        if args.mode in ("chunk-vmap", "shape-preserved") and name == "default":
            with args.default_report.with_suffix(".default.result.pkl").open("rb") as stream:
                preserved = pickle.load(stream)
            if preserved["identity"] != saved["identity"]:
                raise ValueError("preserved baseline result identity mismatch")
            emit("baseline_result_replay_checked", preserved_baseline_comparison=compare_scope(initial[name], preserved["result"]))
        hlo = binaries[1].as_text()
        rows[name] = {"compiler_options": options, "lower_seconds": lower_seconds, "backend_seconds": backend_times,
                      "cold_vjp_seconds": cold_vjp_seconds, "memory": metric, "warm_forward_seconds": [], "warm_vjp_seconds": [],
                      "self_comparisons": [], "hlo_command_buffer_call_lines": [line for line in hlo.splitlines() if " call(" in line and "command_buffer" in line]}
        emit("component_arm_saved", variants=rows)
    pair_comparisons = []
    names = tuple(name for name, unused_minimum in treatment_names)
    with forbid_component_compile(args.mode in ("runtime-replay", "chunk-vmap", "shape-preserved")):
        for repeat in range(7):
            results = {}
            for name in (names if repeat % 2 == 0 else tuple(reversed(names))):
                fwd, reverse = compiled[name]
                began = time.perf_counter()
                jax.block_until_ready(fwd(blocks, x))
                rows[name]["warm_forward_seconds"].append(time.perf_counter() - began)
                began = time.perf_counter()
                value = jax.block_until_ready(reverse(blocks, x, cot))
                rows[name]["warm_vjp_seconds"].append(time.perf_counter() - began)
                results[name] = jax.device_get(value)
                rows[name]["self_comparisons"].append(compare_scope(results[name], initial[name]))
                # Preserve raw repeated parameter derivatives for scatter stability
                # diagnosis; complete output/input trees remain in initial results.
                if args.mode in ("runtime-component", "runtime-replay", "chunk-vmap", "shape-preserved"):
                    write(args.output.with_suffix(f".{name}.repeat{repeat}.parameter_vjp.pkl"), results[name][1][0])
            if len(names) == 2:
                pair_comparisons.append(compare_scope(results[names[1]], results[names[0]]))
    for row in rows.values():
        row["warm_forward_median_seconds"] = float(np.median(row["warm_forward_seconds"]))
        row["warm_vjp_median_seconds"] = float(np.median(row["warm_vjp_seconds"]))
    if args.mode == "runtime-component":
        emit("runtime_component_arm_completed", status="runtime_component_arm_measured", variants=rows,
             environment_timing_scope="single_process_under_declared_runtime_not_cross_environment_alternating",
             strict_self_gate_passed=all(c["passed"] for pair in rows[args.arm]["self_comparisons"] for c in pair.values()),
             real_component_admission=False, device_memory=jax.devices()[0].memory_stats())
        return
    reference_name, candidate_name = names
    finite_and_nonzero = {name: {scope: {"finite": bool(all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(tree))),
        "coordinates": sum(np.size(a) for a in jax.tree_util.tree_leaves(tree)),
        "nonzero_coordinates": sum(int(np.count_nonzero(a)) for a in jax.tree_util.tree_leaves(tree))}
        for scope, tree in (("output", value[0]), ("parameters", value[1][0]), ("inputs", value[1][1]))}
        for name, value in initial.items()}
    emit("component_completed", status="chunk_vmap_component_measured" if args.mode in ("chunk-vmap", "shape-preserved") else ("runtime_component_paired_measured" if args.mode == "runtime-replay" else "threshold_component_measured"), variants=rows, pair_comparisons=pair_comparisons,
         initial_finite_and_nonzero=finite_and_nonzero,
         strict_self_gate_passed=all(c["passed"] for row in rows.values() for pair in row["self_comparisons"] for c in pair.values()),
         initial_comparisons=compare_scope(initial[candidate_name], initial[reference_name]),
         strict_gate_passed=all(c["passed"] for pair in pair_comparisons for c in pair.values()),
         forward_speedup=rows[reference_name]["warm_forward_median_seconds"] / rows[candidate_name]["warm_forward_median_seconds"],
         vjp_speedup=rows[reference_name]["warm_vjp_median_seconds"] / rows[candidate_name]["warm_vjp_median_seconds"],
         optimized_hlo_identical=compiled[reference_name][1].as_text() == compiled[candidate_name][1].as_text(),
         device_memory=jax.devices()[0].memory_stats())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("support", "support-replay", "threshold-support", "threshold-component", "runtime-support", "runtime-component", "runtime-preflight", "runtime-replay", "chunk-vmap", "shape-preserved"), default="support")
    parser.add_argument("--arm", choices=("default", "while"), default="default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-report", type=Path)
    parser.add_argument("--while-report", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--microbatch", type=int, choices=(1,4), default=1)
    parser.add_argument("--route", choices=("expert2", "expert5"), default="expert5")
    parser.add_argument("--runtime-profile", choices=("legacy", "compat"), default="legacy")
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--runtime-preflight", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite preserved diagnostic")
    started = time.perf_counter()
    scope = "captured_original_expert128_forward_and_native_parameter_raw_input_VJP" if args.mode in ("threshold-component", "runtime-component", "runtime-replay", "chunk-vmap", "shape-preserved") else "synthetic_native_map_and_reverse_support"
    report = {"status": "running", "arm": args.arm, "runtime_profile": args.runtime_profile, "XLA_FLAGS": os.environ.get("XLA_FLAGS"), "scope": scope, "rtol": 1e-5, "atol": 1e-6}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        write(args.output, report)
        print(stage, flush=True)
    emit("initializing_runtime")
    if args.runtime_profile == "compat":
        emit("candidate_cpu_library_sources_checked", runtime_library_evidence=runtime_library_evidence(args, before_initialization=True))
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    emit("runtime_initialized", devices=[str(device) for device in devices])
    emit("jax_versions", jax_version=jax.__version__)
    if args.mode.startswith("runtime-") or args.mode in ("chunk-vmap", "shape-preserved"):
        emit("gpu_runtime_library_sources_checked", runtime_library_evidence=runtime_library_evidence(args))
    if len(devices) != 1:
        raise ValueError("requires exactly one visible GPU")
    with deadline(890 if args.mode in ("threshold-component", "runtime-component", "runtime-replay", "chunk-vmap", "shape-preserved") else 175, "bounded_process", emit):
        if args.mode == "runtime-preflight":
            from scripts.hpc.check_daily_training_runtime import main as public_runtime_main
            path = args.output.with_suffix(".public-preflight.json")
            status = public_runtime_main(["--backend", "gpu", "--output", str(path)])
            emit("actual_gpu_public_runtime_preflight", status="passed" if status == 0 else "failed", public_runtime_preflight=str(path))
            if status != 0:
                raise RuntimeError("changed runtime failed public deployed GPU preflight")
            return
        if args.runtime_profile == "compat":
            if args.runtime_preflight is None:
                raise ValueError("changed runtime requires its actual GPU preflight")
            preflight = json.loads(args.runtime_preflight.read_text())
            current = runtime_library_evidence(args)
            if preflight.get("status") != "passed" or preflight.get("runtime_library_evidence", {}).get("libraries") != current["libraries"]:
                raise ValueError("changed runtime preflight absent/failed/different library sources")
        if args.mode == "support-replay":
            support_replay(args, jax, emit)
        elif args.mode in ("threshold-component", "runtime-component", "runtime-replay", "chunk-vmap", "shape-preserved"):
            threshold_component(args, jax, emit)
        else:
            support(args, jax, emit)


if __name__ == "__main__":
    main()
