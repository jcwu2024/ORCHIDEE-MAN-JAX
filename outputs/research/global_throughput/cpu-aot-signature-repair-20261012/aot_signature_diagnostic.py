from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

EXPECTED_PAYLOAD_SHA256 = "be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7"


def tree_equal(left, right, jax, np):
    comparisons = jax.tree_util.tree_map(
        lambda a, b: np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), left, right)
    return all(jax.tree_util.tree_leaves(comparisons))


def raw_device_metadata(arguments, candidate, jax):
    tree, raw = candidate._argument_signature(arguments)
    leaves = jax.tree_util.tree_leaves(arguments)
    device_leaves = [
        jax.device_put(value)
        if not (hasattr(value, "shape") and hasattr(value, "dtype")) else value
        for value in leaves
    ]
    device_arguments = jax.tree_util.tree_structure(arguments).unflatten(device_leaves)
    device_tree, device = candidate._argument_signature(device_arguments)
    return {
        "tree_equal": tree == device_tree,
        "leaf_count": len(raw),
        "raw_device_metadata_equal": raw == device,
        "raw_weak_leaf_count": sum(item[2] for item in raw),
        "device_weak_leaf_count": sum(item[2] for item in device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.code_dir)))

    import jax
    import jaxlib
    import numpy as np

    from scripts.dev import benchmark_global_batch as old_base
    from scripts.dev import benchmark_global_scale as scale
    from scripts.dev import benchmark_global_server_cpu as server

    candidate_path = Path(args.candidate_source)
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if candidate_sha != args.expected_candidate_sha256:
        raise ValueError(f"staged candidate SHA mismatch: {candidate_sha}")
    module_name = "scripts.dev.benchmark_global_batch_candidate"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate module from {candidate_path}")
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = candidate
    spec.loader.exec_module(candidate)
    scale.base = candidate
    if scale.base is not candidate:
        raise AssertionError("scale runner is not using the exact candidate module")

    payload_path = Path(args.payload)
    payload_bytes = payload_path.read_bytes()
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    if payload_sha != EXPECTED_PAYLOAD_SHA256:
        raise ValueError(f"frozen payload SHA mismatch: {payload_sha}")
    lanes, second = scale.PreparedUnpickler(io.BytesIO(payload_bytes)).load()
    selected_indices = [0, 4]
    selected_lanes = [lanes[index] for index in selected_indices]
    selected_second = [second[index] for index in selected_indices]
    supporting_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.code_dir, text=True).strip()

    non_array = {}
    leaf_types = Counter()
    for group_name, group in (("scalar", [lane.arguments for lane in lanes]), ("second", second)):
        for arguments in group:
            pairs, _ = jax.tree_util.tree_flatten_with_path(arguments)
            for path, value in pairs:
                leaf_types[type(value).__module__ + "." + type(value).__qualname__] += 1
                if not (hasattr(value, "shape") and hasattr(value, "dtype")):
                    key = (group_name, jax.tree_util.keystr(path),
                           type(value).__module__ + "." + type(value).__qualname__,
                           candidate._signature_leaf(value))
                    if key not in non_array:
                        non_array[key] = {"group": group_name, "path": key[1], "python_type": key[2],
                                          "value_signature": key[3], "occurrences": 0}
                    non_array[key]["occurrences"] += 1

    first_lane_pairs, _ = jax.tree_util.tree_flatten_with_path(lanes[0].arguments)
    first_lane_args, _ = jax.tree_util.tree_flatten_with_path((tuple(lanes[0].arguments), {}))
    first_non_abstract_leaf = next(
        ((path, value) for path, value in first_lane_args
         if not (hasattr(value, "shape") and hasattr(value, "dtype"))), None)
    raw_float_entry = next(((path, value) for path, value in first_lane_pairs
                            if isinstance(value, float)), None)
    raw_float = None if raw_float_entry is None else raw_float_entry[1]
    raw_float_path = None if raw_float_entry is None else jax.tree_util.keystr(raw_float_entry[0])
    old_helper_error = None
    if first_non_abstract_leaf is not None:
        try:
            old_base._aval_signature([value for _, value in first_lane_args])
        except Exception as error:
            old_helper_error = {"type": type(error).__name__, "message": str(error)}

    scalar_signatures = []
    selected_leaf_types = []
    for index in selected_indices:
        raw = candidate._argument_signature(lanes[index].arguments)[1]
        device = candidate._argument_signature(jax.device_put(lanes[index].arguments))[1]
        scalar_signatures.append({"lane_index": index, "raw_device_exact_equal": raw == device,
                                  "raw_weak_leaves": sum(v[2] for v in raw),
                                  "device_weak_leaves": sum(v[2] for v in device)})
        lane_leaves = jax.tree_util.tree_leaves(lanes[index].arguments)
        selected_leaf_types.append({"lane_index": index,
                                    "landpoint_id": lanes[index].landpoint_id,
                                    "non_array_leaf_count": sum(not (hasattr(value, "shape")
                                                                     and hasattr(value, "dtype"))
                                                                 for value in lane_leaves),
                                    "non_array_types": dict(sorted(Counter(
                                        type(value).__module__ + "." + type(value).__qualname__
                                        for value in lane_leaves
                                        if not (hasattr(value, "shape") and hasattr(value, "dtype")))
                                        .items()))})
    all_lane_metadata = []
    all_second_metadata = []
    for index, lane in enumerate(lanes):
        all_lane_metadata.append({"lane_index": index,
                                  **raw_device_metadata(lane.arguments, candidate, jax)})
        all_second_metadata.append({"lane_index": index,
                                    **raw_device_metadata(second[index], candidate, jax)})
    selected_signatures = [candidate._argument_signature(lane.arguments)[1] for lane in selected_lanes]
    scalar_cross_lane_shape_dtype_equal = tuple((v[0], v[1]) for v in selected_signatures[0]) == tuple(
        (v[0], v[1]) for v in selected_signatures[1])
    scalar_cross_lane_weak_differences = sum(a[2] != b[2] for a, b in zip(
        selected_signatures[0], selected_signatures[1], strict=True))

    same_dtype_strong = np.asarray(1, dtype=np.dtype(candidate._shaped_abstractify(1).dtype))
    weak_lowered = jax.jit(lambda value: value).lower(1)
    weak_audit = candidate.audit_aot_signature(weak_lowered, (1,), lowered=True)
    try:
        candidate.audit_aot_signature(weak_lowered, (same_dtype_strong,), lowered=True)
        weak_audit_rejected = False
    except TypeError:
        weak_audit_rejected = True
    weak_recorder = SimpleNamespace(result={"timing": {}}, phase=lambda name: None)
    weak_executable = candidate.lower_compile(lambda value: value + 1, (1,),
                                              "weak_call_behavior", weak_recorder)
    try:
        weak_only_call = {"accepted": True, "value": int(weak_executable(same_dtype_strong))}
    except Exception as error:
        weak_only_call = {"accepted": False, "error_type": type(error).__name__, "error": str(error)}

    raw_scalar_recorder = SimpleNamespace(result={"timing": {}}, phase=lambda name: None)
    scalar_executable = candidate.lower_compile(lambda value: value + 1.25, (2.5,),
                                                 "raw_float_interface", raw_scalar_recorder)
    scalar_raw = float(scalar_executable(2.5))
    scalar_device = float(scalar_executable(jax.device_put(2.5)))

    def rejected(call):
        try:
            call()
            return False
        except (TypeError, ValueError):
            return True

    scalar_rejections = {
        "wrong_tree": rejected(lambda: scalar_executable((2.5,))),
        "wrong_shape": rejected(lambda: scalar_executable(np.asarray([2.5], dtype=np.float64))),
        "wrong_dtype": rejected(lambda: scalar_executable(np.asarray(2.5, dtype=np.float32))),
    }

    original_scalar_transition = candidate.batch.make_scalar_transition
    original_batch_transition = candidate.batch.make_batch_transition

    def scalar_toy(*values):
        probe = jax.numpy.ravel(jax.tree_util.tree_leaves(values[1])[0])[0]
        return values[0], {"diagnostic": probe}

    def batch_toy(selected):
        def transition(*values):
            probe = jax.numpy.ravel(jax.tree_util.tree_leaves(values[1])[0])[0]
            return values[0], {"diagnostic": probe}
        return transition

    candidate.batch.make_scalar_transition = lambda lane: scalar_toy
    candidate.batch.make_batch_transition = batch_toy
    def recorder_factory():
        return SimpleNamespace(result={"timing": {}, "runtime_observations": {}},
                               phase=lambda *args, **kwargs: None)
    runner_results = {}
    try:
        scalar_recorder = recorder_factory()
        scalar_runner = server.SingleRunner(SimpleNamespace(mode="scalar"), scalar_recorder,
                                            selected_lanes, selected_second, scale)
        scalar_runner.run(2, wrapper=False)
        scalar_core = scalar_runner.collect(2)
        scalar_wrapper = scalar_runner.run(2, wrapper=True)
        runner_results["scalar"] = {
            "core_wrapper_outputs_equal": tree_equal(scalar_core, scalar_wrapper, jax, np),
            "lowered_input_leaves": scalar_recorder.result["compilations"]["server_scalar_once"]
                ["lowered_input_signature"]["caller_leaves"],
            "raw_device_signature_checks": scalar_signatures,
            "cross_lane_shape_dtype_equal": scalar_cross_lane_shape_dtype_equal,
            "cross_lane_weak_type_difference_count": scalar_cross_lane_weak_differences,
        }

        batch_recorder = recorder_factory()
        batch_runner = server.SingleRunner(SimpleNamespace(mode="batch"), batch_recorder,
                                           selected_lanes, selected_second, scale)
        batch_runner.prepare(2)
        batch_runner.run(2, wrapper=False)
        batch_core = batch_runner.collect(2)
        batch_wrapper = batch_runner.run(2, wrapper=True)
        stacked_host = candidate.batch.stack_lanes(selected_lanes)
        stacked_device = jax.device_put(stacked_host)
        host_signature = candidate._argument_signature(stacked_host)[1]
        device_signature = candidate._argument_signature(stacked_device)[1]
        runner_results["batch"] = {
            "core_wrapper_outputs_equal": tree_equal(batch_core, batch_wrapper, jax, np),
            "lowered_input_leaves": batch_recorder.result["compilations"]["server_batch_B2"]
                ["lowered_input_signature"]["caller_leaves"],
            "output_leaf_count": len(jax.tree_util.tree_leaves(batch_core)),
            "raw_device_signature_exact_equal": host_signature == device_signature,
        }
    finally:
        candidate.batch.make_scalar_transition = original_scalar_transition
        candidate.batch.make_batch_transition = original_batch_transition

    report = {
        "schema": "global_cpu_aot_signature_diagnostic_v3",
        "payload": {"path": str(payload_path), "sha256": payload_sha,
                    "lane_count": len(lanes), "second_block_count": len(second),
                    "selected_lane_indices": selected_indices,
                    "selected_landpoint_ids": [lane.landpoint_id for lane in selected_lanes]},
        "candidate": {"path": str(candidate_path), "sha256": candidate_sha,
                      "module_import": candidate.__file__,
                      "abstractifier_module": candidate._shaped_abstractify.__module__,
                      "scale_base_is_candidate": scale.base is candidate,
                      "supporting_code_dir": str(args.code_dir),
                      "supporting_code_commit": supporting_commit,
                      "immutable_v2_module_import": old_base.__file__},
        "environment": {"python": sys.version.split()[0], "python_executable": sys.executable,
                        "jax": jax.__version__,
                        "jaxlib": jaxlib.__version__, "x64_enabled": jax.config.x64_enabled,
                        "node": platform.node(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "launcher": os.environ.get("ORCHJAX_AOT_DIAGNOSTIC_LAUNCHER"),
                        "command": sys.argv},
        "payload_metadata": {"leaf_type_counts": dict(sorted(leaf_types.items())),
                             "unique_non_array_paths_per_group": len(non_array) // 2,
                             "non_array_leaves": [non_array[key] for key in sorted(non_array)],
                             "selected_lane_leaf_types": selected_leaf_types,
                             "all64_scalar_raw_device_metadata": all_lane_metadata,
                             "all64_second_raw_device_metadata": all_second_metadata,
                             "scalar_signature_leaf_count": len(selected_signatures[0]),
                             "scalar_signature_weak_count": sum(v[2] for v in selected_signatures[0])},
        "old_raw_float_helper": {"present_in_payload": raw_float is not None,
                                 "first_raw_float_path": raw_float_path,
                                 "first_non_abstract_leaf_path": jax.tree_util.keystr(first_non_abstract_leaf[0]),
                                 "old_failure_without_physical_lower": old_helper_error},
        "candidate_scalar_interface": {"same_value_raw_device_equal": scalar_raw == scalar_device,
                                       "raw_result": scalar_raw, "device_result": scalar_device,
                                       "standard_rejections": scalar_rejections,
                                       "lowered_input_leaves": raw_scalar_recorder.result["compilations"]
                                           ["raw_float_interface"]["lowered_input_signature"]["caller_leaves"]},
        "strict_weak_type_audit": {
            "weak_construction_signature_passed": weak_audit["public_tree_shape_dtype_weak_type_equal"],
            "same_shape_dtype_strong_input_rejected": weak_audit_rejected,
            "native_compiled_call_with_weak_only_difference": weak_only_call},
        "single_runner": runner_results,
        "scope": "two frozen lanes; physical scalar and batch transitions stubbed; no physical model lowering/compile, GPU, Slurm submission or deployment",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
