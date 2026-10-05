"""B16/B64 full-physics forward screening against the same real-point B4 groups.

Two seven-day blocks with JAX x64 enabled and baseline native dtypes retained.
The geographical payload and solar subcalculation contain FP32. Exact schema,
discrete masks and existing numerical/sentinel gates apply; annual/AD acceptance
requires separate evidence.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from scripts.dev import benchmark_global_batch as base
from scripts.dev.global_batch_stage_store import StageStore
from scripts.dev.probe_global_batch_cache import CacheEvents, atomic_bytes, configure_cache

SIZES = (16, 64)
GPU_SLACK_BYTES = 2 * 1024**3
PRECISION_POLICY = "JAX x64 enabled; baseline native dtypes retained; geographical payload and solar subcalculation contain FP32"
# Exact inherited NetCDF payload schema, not a general low-precision allowance.
# domain.py:786-795 -> sechiba_boundary.py:178-181 -> orchestration.py:10208-10209.
# lalo also feeds orchestration.py:11110-11115 -> season.py:227-287: the solar
# subcalculation contains FP32 before its final recurrent-state FP64 promotion.
NATIVE_GROUP5_FLOAT32 = {
    "['contfrac']": (1,), "['lalo']": (1, 2), "['lat']": (1, 1), "['lon']": (1, 1),
}


def restore_path(*parts):
    text = str(PurePosixPath(*(str(part).replace("\\", "/") for part in parts)))
    if os.name != "nt" and len(text) > 1 and text[1] == ":":
        raise ValueError("prepared metadata contains an unmapped Windows drive path on Linux")
    return Path(text)


class PreparedUnpickler(pickle.Unpickler):
    """Represent approved serialized paths on this host; never remap science arrays."""
    def find_class(self, module, name):
        if module == "pathlib" and name in {"Path", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"}:
            return restore_path
        return super().find_class(module, name)


def read_prepared(path, config):
    data = Path(path).read_bytes()
    lanes, second = PreparedUnpickler(io.BytesIO(data)).load()
    config_bytes = Path(config).read_bytes()
    for lane in lanes:
        if lane.context.config_path.read_bytes() != config_bytes:
            raise ValueError(f"prepared configuration differs from declared config: {lane.landpoint_id}")
        if base.read_run_def_values(lane.context.run_def_path) != lane.context.run_def_values:
            raise ValueError(f"prepared run.def changed since preparation: {lane.landpoint_id}")
    lanes = [replace(lane, static_signature=base.batch.static_signature(
        lane.context, lane.initial_state, lane.tables, lane.dispatch)) for lane in lanes]
    return lanes, second


def load_local_cpu_anchors(args, lanes, second, environment):
    """Migrate accepted pre-stage CPU evidence without a new scalar compilation."""
    if any(device["platform"] != "cpu" for device in environment["device_details"]):
        raise ValueError("local-cpu reference format is restricted to CPU execution")
    if args.anchor_prepared_input is None:
        raise ValueError("local-cpu references require explicit --anchor-prepared-input")
    directory = args.reference_dir.resolve()
    reports = {name: json.loads((directory / name).read_text(encoding="utf-8-sig"))
               for name in ("report.json", "reference_origin_report.json")}
    current, origin = reports["report.json"], reports["reference_origin_report.json"]
    if current["status"] != "passed_bounded_forward_diagnostic":
        raise ValueError("local CPU reference did not complete its bounded acceptance")
    critical = ("python", "jax", "jaxlib", "numpy", "device_details", "x64_enabled", "baseline_ref")
    compiler = ("XLA_FLAGS", "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    missing = []
    for name, report in reports.items():
        previous = report["environment"]
        for key in critical:
            if key not in previous or previous[key] != environment[key]:
                raise ValueError(f"local CPU reference runtime mismatch: {name}/{key}")
        for key in compiler:
            if key not in previous["env"] or previous["env"][key] != environment["env"].get(key):
                raise ValueError(f"local CPU reference compiler environment mismatch: {name}/{key}")
        if previous.get("backend_platform_version") is None:
            missing.append(name + ": backend_platform_version not recorded")
        elif previous["backend_platform_version"] != environment["backend_platform_version"]:
            raise ValueError("local CPU reference backend version mismatch")
        scientific_diff = subprocess.check_output(
            ["git", "diff", "--name-only", previous["git_head"], "--", "jax_orchidee"], cwd=ROOT, text=True)
        if scientific_diff.strip():
            raise ValueError("local CPU reference scientific source changed")
    cleared = {phase.get("landpoint_id") for phase in origin["phases"]
               if phase["phase"] == "independent_scalar_nested_cache_cleared" and phase.get("entries_after") == 0}
    if not set(base.POINTS).issubset(cleared):
        raise ValueError("local CPU independently bound reference evidence missing")
    for name in ("B4_permutation", "B1_original_AOT_vs_extracted_full_state_day_outputs"):
        if not any(gate["name"] == name and gate["passed"] for gate in current["gates"]):
            raise ValueError(f"local CPU source acceptance gate missing: {name}")
    declared = [phase["path"] for phase in current["phases"] if phase["phase"] == "prepared_input_reused"]
    source = args.anchor_prepared_input.resolve()
    if len(declared) != 1 or (ROOT / declared[0].replace("\\", "/")).resolve() != source:
        raise ValueError("local CPU original prepared input does not match source report")
    source_lanes, source_second = read_prepared(source, args.config)
    if tuple(lane.landpoint_id for lane in source_lanes) != base.POINTS:
        raise ValueError("local CPU original prepared anchors differ")
    identities = {}
    for lane, other, old_lane, old_other in zip(lanes[:4], second[:4], source_lanes, source_second, strict=True):
        canonical = base.semantic_reference_identity(args, lane, other)
        if canonical != base.semantic_reference_identity(args, old_lane, old_other):
            raise ValueError(f"local CPU full canonical anchor inputs changed: {lane.landpoint_id}")
        identities[lane.landpoint_id] = base.reference_identity(args, old_lane, old_other)
    values, sources = {}, {}
    for point in base.POINTS:
        path = directory / ("scalar_" + point + "_reference.pkl")
        data = path.read_bytes()
        saved_identity, value = pickle.loads(data)
        if saved_identity != identities[point]:
            raise ValueError(f"local CPU recorded original reference identity changed: {point}")
        values["scalar_" + point] = value
        sources["scalar_" + point] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                                     "execution_origin": {"available": True, "report": str(directory / "reference_origin_report.json"),
                                                          "environment": origin["environment"]},
                                     "role": "independent own-context CPU numerical output; performance excluded"}
    candidate_path = directory / "B4_candidate_before_gate.pkl"
    candidate_identity, candidate = pickle.loads(candidate_path.read_bytes())
    if candidate_identity != identities[base.POINTS[0]]:
        raise ValueError("local CPU B4 candidate recorded identity changed")
    full_path = directory / "B4_full_state_outputs.pkl"
    full_data = full_path.read_bytes()
    full = pickle.loads(full_data)
    if not base.compare(full, candidate, sentinel_paths=base.roughheight_sentinel_contract(lanes[0]))["passed"]:
        raise ValueError("local CPU final B4 output differs from accepted candidate")
    values["B4_batch_only_v1"] = full
    sources["B4_batch_only_v1"] = {"path": str(full_path), "sha256": hashlib.sha256(full_data).hexdigest(),
                                  "execution_origin": {"available": True, "report": str(directory / "report.json"),
                                                       "environment": current["environment"]},
                                  "role": "accepted pre-stage CPU B4 numerical output; performance excluded"}
    evidence = {"format": "local-cpu", "source_prepared_input": str(source),
                "source_prepared_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reports": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in reports},
                "references": sources, "missing_origin_metadata": missing + ["scientific content hash at execution; checked recorded Git source correspondence"]}
    evidence["manifest_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    evidence["identity_hash_scope"] = "CPU evidence contract digest; there is no historical stage manifest"
    return values, evidence


def tree_bytes(value):
    return sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(value))


def screen_inputs(lanes, second):
    """Re-audit all closures, complete shapes and source-supported input sentinels."""
    if not jax.config.jax_enable_x64:
        raise ValueError("JAX x64 must be enabled for baseline native dtype admission")
    ids = [lane.landpoint_id for lane in lanes]
    if len(lanes) not in SIZES or len(second) != len(lanes):
        raise ValueError("prepared input requires 16 or 64 aligned real lanes")
    if tuple(ids[:4]) != base.POINTS or len(set(ids)) != len(ids):
        raise ValueError("four declared anchors first and all real point IDs unique are required")
    audit = base.batch.audit_compatible_lanes(lanes)
    base.batch.audit_compatible_lanes([replace(lane, arguments=other) for lane, other in zip(lanes, second, strict=True)])
    sentinels = base.roughheight_sentinel_contract(lanes[0])
    # Output paths [block][0] prefix the identical values_by_component input tree.
    state_contract = {path[len("[0][0]"):]: contract for path, contract in sentinels.items()
                      if path.startswith("[0][0]")}
    native_leaf_count = 0
    for lane, other in zip(lanes, second, strict=True):
        if base.roughheight_sentinel_contract(lane) != sentinels:
            raise ValueError(f"sentinel layout differs: {lane.landpoint_id}")
        for block, arguments in enumerate((lane.arguments, other)):
            days = np.arange(2 + block * 7, 9 + block * 7, dtype=np.int32)
            if not np.array_equal(arguments.science_day_numbers, days):
                raise ValueError(f"two declared seven-day blocks required: {lane.landpoint_id}")
            result = base.compare(arguments.values_by_component, arguments.values_by_component,
                                  sentinel_paths=state_contract)
            if not result["passed"]:
                raise ValueError(f"unexpected initial-state nonfinite: {lane.landpoint_id}")
            for index, group in enumerate(arguments):
                for path, value in jax.tree_util.tree_flatten_with_path(group)[0]:
                    array = np.asarray(value)
                    native_shape = NATIVE_GROUP5_FLOAT32.get(jax.tree_util.keystr(path)) if index == 5 else None
                    if native_shape is not None:
                        if array.dtype != np.dtype("float32") or array.shape != native_shape:
                            raise ValueError(f"exact baseline native float32 payload schema required: {lane.landpoint_id}/{index}/{path}")
                        native_leaf_count += 1
                    elif array.dtype.kind in "fc" and array.dtype != np.dtype("float64"):
                        raise ValueError(f"float64 dynamic input required outside exact native payload: {lane.landpoint_id}/{index}/{path}")
                    if index != 0 and array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                        raise ValueError(f"nonfinite non-state input: {lane.landpoint_id}/{index}/{path}")
            for forcing in jax.tree_util.tree_leaves(arguments.forcing_days):
                if np.asarray(forcing).shape[:2] != (7, 48):
                    raise ValueError(f"forcing must retain seven days by 48 steps: {lane.landpoint_id}")
    return {**audit, "unique_real_points": len(ids), "blocks": [[2, 8], [9, 15]],
            "input_bytes": tree_bytes(([lane.arguments for lane in lanes], second)),
            "precision_policy": PRECISION_POLICY,
            "native_float32_leaf_count": native_leaf_count,
            "native_float32_schema": [{"group": 5, "path": path, "shape": list(shape), "dtype": "float32"}
                                      for path, shape in NATIVE_GROUP5_FLOAT32.items()],
            "nonfinite_policy": "only audited initial-state bare-soil roughheight sentinels"}


def load_anchor_references(args, lanes, second, environment):
    """Read numerical evidence only; no source StageStore mutation or old timing reuse."""
    if getattr(args, "reference_format", "staged") == "local-cpu":
        return load_local_cpu_anchors(args, lanes, second, environment)
    directory = args.reference_dir.resolve()
    manifest_data = (directory / "stage_manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    expected = base.stage_identity(args, lanes[:4], second[:4], environment)
    if manifest["identity"] != expected:
        raise ValueError("anchor reference input/science/baseline/runtime/device/compiler identity mismatch")
    names = ["B4_batch_only_v1"] + ["scalar_" + point for point in base.POINTS]
    values, origins = {}, {}
    for name in names:
        record = manifest["stages"].get(name, {})
        if record.get("status") != "complete":
            raise ValueError(f"anchor reference incomplete: {name}")
        path = directory / ("stage_" + name + ".pkl")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if record.get("sha256") != digest:
            raise ValueError(f"anchor reference checksum changed: {name}")
        envelope = pickle.loads(data)
        if (envelope["identity"] != expected or envelope["name"] != name or envelope["status"] != "complete"):
            raise ValueError(f"anchor reference envelope changed: {name}")
        if name.startswith("scalar_"):
            proof = record.get("provenance", {})
            if not proof.get("own_context_bound") or not proof.get("nested_cache_cleared_before_lower"):
                raise ValueError(f"independent own-context reference proof missing: {name}")
        values[name] = envelope["value"]["outputs"] if name == "B4_batch_only_v1" else envelope["value"]
        origins[name] = {"path": str(path), "sha256": digest,
                         "execution_origin": record.get("execution_origin", {"available": False}),
                         "imported_from": record.get("imported_from"),
                         "role": "prior numerical evidence only; prior performance measurements excluded"}
    return values, {"manifest": str(directory / "stage_manifest.json"),
                    "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(), "references": origins}


def executable_memory(executable):
    try:
        analysis = executable.memory_analysis()
        if analysis is None:
            return {"available": False, "reason": "backend returned no memory analysis"}
        fields = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
                  "alias_size_in_bytes", "generated_code_size_in_bytes")
        return {"available": True, **{field: int(getattr(analysis, field, 0)) for field in fields},
                "scope": "compiler device buffer estimate; not host RSS or measured VRAM peak"}
    except (AttributeError, NotImplementedError) as exc:
        return {"available": False, "reason": str(exc)}


def visible_gpu_memory():
    if jax.devices()[0].platform != "gpu":
        return {"applicable": False, "reason": "CPU execution; no GPU memory admission claim"}
    uuid = os.environ.get("ORCHJAX_PHYSICAL_GPU_UUID")
    if not uuid:
        raise ValueError("GPU preflight requires launcher-selected physical GPU UUID")
    text = subprocess.check_output(["nvidia-smi", "-i", uuid,
                                    "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
    total, free = (int(value.strip()) * 1024**2 for value in text.strip().split(","))
    return {"applicable": True, "uuid": uuid, "total_bytes": total, "free_bytes": free,
            "scope": "selected physical GPU snapshot immediately before prospective compilation"}


def memory_preflight(size, inputs, b4_output, memory_records, snapshot=None, recorder=None):
    """Conservative resource admission estimate, never a physics approximation."""
    physical = visible_gpu_memory() if snapshot is None else snapshot
    input_bytes = tree_bytes(inputs)
    output_bytes = tree_bytes(b4_output) // 4 * size
    estimates = []
    for batch_size, record in memory_records.items():
        if record.get("available"):
            estimates.append(max(0, record["temp_size_in_bytes"]) * size // batch_size)
    # Both block outputs, working state and host wrapper inputs can overlap.
    projected = 2 * input_bytes + 3 * output_bytes + max(estimates, default=0)
    result = {"size": size, "actual_stacked_input_bytes": input_bytes,
              "output_bytes_from_actual_B4_leaf_shapes": output_bytes,
              "compiler_memory_analysis": {str(k): v for k, v in memory_records.items()},
              "projected_device_buffer_bytes": projected, "required_physical_slack_bytes": GPU_SLACK_BYTES,
              "physical_gpu": physical,
              "limitation": "scaled compiler buffer estimate is a preflight screen, not an OOM guarantee; RSS is not VRAM"}
    if physical["applicable"]:
        result["passed"] = (physical["total_bytes"] >= 30 * 1024**3
                            and physical["free_bytes"] >= projected + GPU_SLACK_BYTES)
        if recorder is not None:
            recorder.result.setdefault("memory_preflight", {})[str(size)] = result
            recorder.flush()
        if physical["total_bytes"] < 30 * 1024**3:
            raise RuntimeError("scale GPU admission requires the declared visible 32GB card")
        if not result["passed"]:
            raise RuntimeError("insufficient selected GPU free-memory slack before batch compilation")
    else:
        result["passed"] = True
    return result


def compile_batch(lanes, second, label, recorder):
    base.driver._COMPILED_SECHIBA_SCAN_CACHE.clear()
    started = time.perf_counter()
    host = base.batch.stack_lanes(lanes), base.stack_second(lanes, second)
    recorder.result["timing"][label + "_host_stack_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    inputs = jax.device_put(host)
    base.sync(inputs)
    recorder.result["timing"][label + "_device_transfer_seconds"] = time.perf_counter() - started
    return base.lower_compile(base.batch.make_batch_transition(lanes), inputs[0], label, recorder), inputs


def scale_benchmark(args, recorder, lanes, second):
    screen = screen_inputs(lanes, second)
    if max(args.sizes) > len(lanes):
        raise ValueError("requested size exceeds audited unique prepared points")
    base.legacy_factory(args.baseline_ref)
    recorder.sentinel_paths = base.roughheight_sentinel_contract(lanes[0])
    recorder.result["compatibility_audit"] = screen
    old_values, origins = load_anchor_references(args, lanes, second, recorder.result["environment"])
    recorder.result["anchor_numerical_origins"] = origins
    identity = base.stage_identity(args, lanes, second, recorder.result["environment"])
    identity.update(contract="unique-real-points-two-seven-day-blocks-B16-B64-v1",
                    reference_manifest_sha256=origins["manifest_sha256"], precision_policy=PRECISION_POLICY)
    store = StageStore(args.output_dir, identity)
    sizes = sorted(set(args.sizes) | ({16} if 64 in args.sizes else set()))
    required = ["B4_group_references"] + [f"B{size}_scale" for size in sizes]
    recorder.result.update(required_stages=required, stages=store.manifest["stages"],
                           requested_sizes=list(args.sizes), execution_sizes=sizes,
                           comparison_policy="actual same-point B4 group serial; no old scalar extrapolation")
    for name in required:
        loaded = store.load(name)
        if loaded is not None:
            recorder.result["gates"].extend(loaded[1].get("gates", []))
            recorder.result["timing"].update(loaded[1].get("timing", {}))
            base.record_timing_origin(recorder.result, name, loaded[1], reused=True)

    def run_stage(name, operation):
        def execute():
            gate_start, phase_start = len(recorder.result["gates"]), len(recorder.result["phases"])
            old_timing = dict(recorder.result["timing"])
            old_compilations = dict(recorder.result.get("compilations", {}))
            started = time.perf_counter()
            recorder.phase(name + "_stage_start")
            value, proof = operation()
            return value, {
                "provenance": proof, "stage_seconds": time.perf_counter() - started,
                "execution_origin": {"available": True, "environment": recorder.result["environment"],
                                     "pid": os.getpid(), "process_start_monotonic": base.PROCESS_STARTED,
                                     "window_identity": os.environ.get("ORCHJAX_GLOBAL_BATCH_WINDOW_ID"),
                                     "report": str(recorder.window_report) if recorder.window_report else None},
                "gates": recorder.result["gates"][gate_start:], "phases": recorder.result["phases"][phase_start:],
                "timing": {k: v for k, v in recorder.result["timing"].items() if old_timing.get(k) != v},
                "compilations": {k: v for k, v in recorder.result.get("compilations", {}).items()
                                 if old_compilations.get(k) != v},
            }
        (value, record), reused = store.run(name, execute)
        base.record_timing_origin(recorder.result, name, record, reused=reused)
        recorder.phase(name + ("_stage_reused" if reused else "_stage_complete"))
        return value

    b4, anchor_inputs, anchor_output = None, None, None
    memory_records = {}
    for size in SIZES:
        record = store.manifest["stages"].get(f"B{size}_scale", {})
        if record.get("status") == "complete":
            analysis = record.get("provenance", {}).get("executable_memory_analysis")
            if analysis is not None:
                memory_records[size] = {**analysis, "origin": record.get("execution_origin"),
                                        "measurement_reused": True}
    resident = {}

    def ensure_b4():
        nonlocal b4, anchor_inputs, anchor_output
        if b4 is None:
            b4, anchor_inputs = compile_batch(lanes[:4], second[:4], "B4_resident", recorder)
            started = time.perf_counter()
            anchor_output = base.run_two(b4, *anchor_inputs)
            recorder.result["timing"]["B4_resident_first_call_seconds"] = time.perf_counter() - started
            recorder.gate("B4_current_anchor_vs_third_round_full_output", anchor_output, old_values["B4_batch_only_v1"])
            for index, point in enumerate(base.POINTS):
                recorder.gate("B4_current_anchor_vs_independent_" + point,
                              jax.tree_util.tree_map(lambda v: v[index], anchor_output), old_values["scalar_" + point])
            memory_records[4] = executable_memory(b4)
            recorder.result["memory_analysis"] = {"4": memory_records[4]}
        return b4

    def group_references():
        executable = ensure_b4()
        outputs = [jax.device_get(anchor_output)]
        for start in range(4, len(lanes), 4):
            group = lanes[start:start + 4]
            inputs = base.batch.stack_lanes(group), base.stack_second(group, second[start:start + 4])
            output = base.run_two(executable, *inputs)
            recorder.gate(f"B4_reference_group_{start // 4}_nonfinite_schema", output, output)
            outputs.append(jax.device_get(output))
        combined = jax.tree_util.tree_map(lambda *items: np.concatenate(items, axis=0), *outputs)
        return combined, {"role": "actual B4 complete-block group outputs for every unique prepared lane",
                          "point_ids": [lane.landpoint_id for lane in lanes], "own_inputs": True,
                          "scalar_recompilations": 0, "anchor_numerical_origins": origins}

    reference = run_stage("B4_group_references", group_references)

    for size in sizes:
        def operation(size=size):
            executable4 = ensure_b4()
            chosen, others = lanes[:size], second[:size]
            host_inputs = base.batch.stack_lanes(chosen), base.stack_second(chosen, others)
            preflight = memory_preflight(size, host_inputs, anchor_output, memory_records, recorder=recorder)
            recorder.result.setdefault("memory_preflight", {})[str(size)] = preflight
            recorder.phase(f"B{size}_memory_preflight_passed", preflight=preflight)
            executable, inputs = compile_batch(chosen, others, f"B{size}_candidate", recorder)
            resident[size] = executable  # Keep accepted B16 and B4 while considering B64.
            memory_records[size] = executable_memory(executable)
            recorder.result["memory_analysis"] = {str(k): v for k, v in memory_records.items()}
            started = time.perf_counter()
            output = base.run_two(executable, *inputs)
            recorder.result["timing"][f"B{size}_first_call_seconds"] = time.perf_counter() - started
            expected = jax.tree_util.tree_map(lambda value: value[:size], reference)
            for index, lane in enumerate(chosen):
                recorder.gate(f"B{size}_lane_{index}_{lane.landpoint_id}_both_blocks",
                              jax.tree_util.tree_map(lambda v: v[index], output),
                              jax.tree_util.tree_map(lambda v: v[index], expected))
            order = list(reversed(range(size)))  # Includes reorder across B4 reference-group boundaries.
            shuffled = [chosen[index] for index in order]
            permuted = base.run_two(executable, base.batch.stack_lanes(shuffled),
                                    base.stack_second(shuffled, [others[index] for index in order]))
            recorder.gate(f"B{size}_cross_group_permutation", permuted,
                          jax.tree_util.tree_map(lambda v: v[np.asarray(order)], expected))
            changed = chosen[:-1] + [chosen[0]]
            isolated = base.run_two(executable, base.batch.stack_lanes(changed),
                                    base.stack_second(changed, others[:-1] + [others[0]]))
            replacement = list(range(size - 1)) + [0]
            recorder.gate(f"B{size}_replacement_all_lanes_isolation", isolated,
                          jax.tree_util.tree_map(lambda v: v[np.asarray(replacement)], expected))
            started = time.perf_counter()
            pairs = jax.device_put([(base.batch.stack_lanes(chosen[start:start + 4]),
                                     base.stack_second(chosen[start:start + 4], others[start:start + 4]))
                                    for start in range(0, size, 4)])
            base.sync(pairs)
            recorder.result["timing"][f"B{size}_B4_groups_host_stack_device_transfer_seconds"] = time.perf_counter() - started
            # Gate the exact baseline inputs/program that will be timed.
            for index, pair in enumerate(pairs):
                recorder.gate(f"B{size}_timing_B4_group_{index}", base.run_two(executable4, *pair),
                              jax.tree_util.tree_map(lambda v: v[index * 4:(index + 1) * 4], expected))
            warm, wrapper, groups = [], [], []
            for _ in range(args.warm_repeats):
                started = time.perf_counter()
                base.run_two(executable, *inputs)
                warm.append(time.perf_counter() - started)
                started = time.perf_counter()
                wrapped = base.run_two(executable, base.batch.stack_lanes(chosen), base.stack_second(chosen, others))
                jax.device_get(wrapped)
                wrapper.append(time.perf_counter() - started)
                started = time.perf_counter()
                for pair in pairs:
                    base.run_two(executable4, *pair)
                groups.append(time.perf_counter() - started)
            point_days = size * 14
            measured = {f"B{size}_warm_frozen_state_core": base.metrics(warm, point_days),
                        f"B{size}_preprepared_wrapper_stack_transfer_materialize": base.metrics(wrapper, point_days),
                        f"B{size}_B4_group_serial": base.metrics(groups, point_days),
                        f"B{size}_core_speedup_vs_B4_groups": statistics.median(groups) / statistics.median(warm)}
            recorder.result["timing"].update(measured)
            return {"outputs": jax.device_get(output), "measurements": measured}, {
                "point_ids": [lane.landpoint_id for lane in chosen], "unique_real_points": size,
                "full_two_block_parity": True, "cross_group_permutation": True, "replacement_isolation": True,
                "baseline": f"{size // 4} actual resident B4 groups over identical {point_days} point-days",
                "five_or_more_synchronized_repeats": True, "memory_preflight": preflight,
                "executable_memory_analysis": memory_records[size],
                "precision_policy": PRECISION_POLICY,
                "accepted_scope": "bounded complete forward with baseline native dtypes; not scalar/Fortran/annual/AD acceptance"}
        run_stage(f"B{size}_scale", operation)
    recorder.result["all_required_stages_complete"] = all(
        store.manifest["stages"].get(name, {}).get("status") == "complete" for name in required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-input", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--execution-order", choices=["baseline-first", "candidate-first"], default="baseline-first")
    parser.add_argument("--full-reference-dir", type=Path, help="Sealed FULL64 B4 group numerical reference for candidate-first.")
    parser.add_argument("--measure-b4-baseline", action="store_true", help="Compile a new actual B4 baseline only after candidate checkpoints.")
    parser.add_argument("--reference-format", choices=["staged", "local-cpu"], default="staged")
    parser.add_argument("--anchor-prepared-input", type=Path,
                        help="Original declared local CPU four-anchor pickle for pre-stage evidence migration.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--baseline-ref", default=base.BASELINE)
    parser.add_argument("--cache-mode", choices=["off", "persistent"], default="off")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--asset-root", type=Path, help="Launcher provenance only; prepared inputs are reused read-only.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    parser.add_argument("--truth-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.warm_repeats < 5 or len(args.sizes) != len(set(args.sizes)):
        parser.error("at least five repeats and unique sizes required")
    if args.execution_order == "baseline-first" and (args.reference_dir is None or args.sizes != sorted(args.sizes)):
        parser.error("baseline-first requires --reference-dir and ascending sizes")
    if args.execution_order == "candidate-first" and args.full_reference_dir is None:
        parser.error("candidate-first requires --full-reference-dir")
    args.output_dir = args.output_dir.resolve()
    if args.reference_dir is not None:
        StageStore.validate_source_destination(args.reference_dir, args.output_dir)
    if args.full_reference_dir is not None:
        StageStore.validate_source_destination(args.full_reference_dir, args.output_dir)
    StageStore.validate_source_destination(args.prepared_input.resolve().parent, args.output_dir)
    if args.anchor_prepared_input is not None:
        StageStore.validate_source_destination(args.anchor_prepared_input.resolve().parent, args.output_dir)
    os.environ.setdefault("ORCHIDEE_REPO_ROOT", str(ROOT))
    os.environ.setdefault("ORCHIDEE_DATA_ROOT", str(args.truth_root / "data"))
    os.environ.setdefault("ORCHIDEE_REFERENCE_ROOT", str(args.truth_root / "reference"))
    os.environ["ORCHIDEE_OUTPUT_ROOT"] = str(args.output_dir)
    policy = configure_cache(args.cache_mode, args.cache_root)
    if (args.output_dir / "report.json").exists():
        atomic_bytes(args.output_dir / "windows" / f"prior_report_{time.time_ns()}.json",
                     (args.output_dir / "report.json").read_bytes())
    recorder = base.Recorder(args.output_dir)
    recorder.cache_observer = CacheEvents()
    previous = {}

    def stopped(number, frame):
        raise InterruptedError(f"signal {number}; completed atomic scale checkpoints retained")

    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, stopped)
    try:
        recorder.result["environment"] = base.metadata(args)
        if args.execution_order == "candidate-first":
            if not os.environ.get("ORCHJAX_SERVER_RESOURCE_PLAN"):
                raise RuntimeError("candidate-first GPU benchmark requires the allocated GPU entry resource admission")
            recorder.result["server_resources"] = json.loads(os.environ["ORCHJAX_SERVER_RESOURCE_PLAN"])
        recorder.result["environment"]["explicit_cache_policy"] = policy
        recorder.result["timing"]["import_setup_seconds"] = time.perf_counter() - base.PROCESS_STARTED
        if not jax.config.jax_enable_x64:
            raise RuntimeError("JAX x64 must be enabled; exact baseline native input dtypes are retained")
        recorder.result["precision_policy"] = PRECISION_POLICY
        started = time.perf_counter()
        lanes, second = read_prepared(args.prepared_input, args.config)
        recorder.phase("prepared_input_reused", path=str(args.prepared_input),
                       policy="portable metadata paths; numeric trees unchanged; on-disk config/run.def verified; static signatures rederived")
        recorder.result["timing"]["prepared_input_read_audit_seconds"] = time.perf_counter() - started
        if args.execution_order == "candidate-first":
            from scripts.dev.global_scale_candidate_first import candidate_first_benchmark

            candidate_first_benchmark(args, recorder, lanes, second)
        else:
            scale_benchmark(args, recorder, lanes, second)
        recorder.result["status"] = "passed_bounded_scale_forward_diagnostic"
    except (InterruptedError, KeyboardInterrupt) as exc:
        recorder.result.update(status="stopped", message=str(exc), exception_type=type(exc).__name__)
        recorder.phase("stopped", message=str(exc))
        return 2
    except Exception as exc:
        recorder.result.update(status="failed", message=str(exc), exception_type=type(exc).__name__,
                               traceback=traceback.format_exc())
        recorder.phase("failed", message=str(exc))
        return 1
    finally:
        recorder.stop.set()
        recorder.result["process_elapsed_seconds_including_import_setup"] = time.perf_counter() - base.PROCESS_STARTED
        recorder.flush()
        recorder.cache_observer.close()
        for number, handler in previous.items():
            signal.signal(number, handler)
    recorder.phase("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
