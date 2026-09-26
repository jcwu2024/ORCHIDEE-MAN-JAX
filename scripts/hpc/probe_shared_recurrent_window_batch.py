"""Two independent T7 windows: serial versus vmap, without optimizer updates.

This is a two-window execution diagnostic, not a production sampling change.
The public recurrent objective retains native checkpointed AD through all days.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def load_helpers():
    path = Path(__file__).with_name("probe_shared_combined_recurrent.py")
    name = "scripts.hpc.probe_shared_combined_recurrent"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load staged recurrent helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, module.load_daily_probe(path.with_name("probe_shared_expert_real_daily.py"))


def stack_cases(jax, cases):
    """Require identical pytree/leaf geometry; stack only independent windows."""
    import jax.numpy as jnp

    if len(cases) != 2 or any(len(case) != 5 for case in cases):
        raise ValueError("exactly two five-part recurrent cases are required")
    first, first_tree = jax.tree_util.tree_flatten(cases[0])
    second, second_tree = jax.tree_util.tree_flatten(cases[1])
    if first_tree != second_tree or not first:
        raise ValueError("two recurrent case pytrees differ or are empty")
    for left, right in zip(first, second, strict=True):
        if np.shape(left) != np.shape(right) or np.asarray(left).dtype != np.asarray(right).dtype:
            raise ValueError("two recurrent case leaf shapes/dtypes differ")
        if np.asarray(left).dtype.kind not in "biufc":
            raise ValueError("recurrent case leaves must be numeric arrays")
    return jax.tree_util.tree_map(lambda a, b: jnp.stack((a, b), axis=0), *cases)


def make_batched_objective(jax, jnp, recurrent_objective):
    from scripts.hpc.shared_daily_microbatch import make_microbatch_objective

    def single(parameters, case):
        return recurrent_objective(parameters, *case)

    return make_microbatch_objective(jax, jnp, single)


def aggregate_results(jax, jnp, left, right):
    from scripts.hpc.shared_daily_microbatch import _aggregate_auxiliary

    auxiliary = jax.tree_util.tree_map(lambda a, b: jnp.stack((a, b)), left[0][1], right[0][1])
    gradients = jax.tree_util.tree_map(lambda a, b: (a + b) / 2, left[1], right[1])
    return (((left[0][0] + right[0][0]) / 2, _aggregate_auxiliary(jax, jnp, auxiliary)), gradients)


def physical_counts(daily, result):
    return daily.hard_counts(result[0][1])


def inspect_assets(args, helper, daily):
    import jax

    args.horizon = 7
    assets = helper.inspect_assets(args, daily)
    with args.input_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    cases = tuple(payload["cases"][7] for payload in bundle["payloads"])
    for case in cases:
        if np.shape(case.days[0].continuous_state)[0] != 7 or np.shape(case.targets["state"])[0] != 7:
            raise ValueError("both windows must have exactly seven days")
    stacked = stack_cases(jax, cases)
    with args.combined_report.open(encoding="utf-8") as stream:
        combined = json.load(stream)
    if (
        combined.get("status") != "passed" or combined.get("horizon") != 7
        or combined.get("artifact_id") != assets["artifact_id"]
        or combined.get("rtol") != daily.RTOL or combined.get("atol") != daily.ATOL
        or not all(combined.get("comparisons", {}).get(key, {}).get("passed") is True
                   for key in ("loss", "auxiliary", "all_parameter_gradients"))
        or any(combined.get("reference_hard", {"missing": 1}).values())
        or any(combined.get("candidate_hard", {"missing": 1}).values())
    ):
        raise ValueError("reference must be the passed strict combined T7 diagnostic")
    executable = Path(combined["candidate_executable"]).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"combined T7 executable absent: {executable}")
    adapter = daily.load_staged_adapter(args.candidate_module_file)
    backbone = daily.load_backbone_assembler(args.backbone_module_file)
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective

    context = assets["context"]
    _, objective = make_shared_recurrent_objective(
        context["head"], context["template"], context["state_scales"], context["science_scales"]
    )
    return assets | {"cases": cases, "stacked": stacked, "combined": combined,
                     "combined_executable": executable, "adapter": adapter,
                     "backbone": backbone, "objective": objective,
                     "artifact_ids": tuple(bundle["artifact_ids"])}


def benchmark(call, jax, repeats, preserve=None):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    result = call()
    jax.block_until_ready(result)
    if preserve is not None:
        preserve(result)
    times = []
    for _ in range(repeats):
        began = time.perf_counter()
        timed = call()
        jax.block_until_ready(timed)
        times.append(time.perf_counter() - began)
    return result, times


def compile_candidate(backward, parameters, stacked, reference_path, patches, emit):
    if not reference_path.is_file():
        raise FileNotFoundError("full two-window reference must be saved before candidate compilation")
    with patches:
        began = time.perf_counter()
        lowered = backward.lower(parameters, stacked)
        lower_seconds = time.perf_counter() - began
        emit("candidate_before_backend_compile", lower_seconds=lower_seconds)
        began = time.perf_counter()
        candidate = lowered.compile()
        compile_seconds = time.perf_counter() - began
    return candidate, lower_seconds, compile_seconds


def inspect_abstract_forward(jax, daily, assets):
    """Trace the real two-window forward on CPU, without AD/backend compile."""
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator import shared_process_head
    from scripts.hpc.run_shared_learning import load_learning_checkpoint

    began = time.perf_counter()
    checkpoint = load_learning_checkpoint(assets["checkpoint"], expected_method=assets["reference"]["method"])
    if checkpoint["completed_updates"] != 0:
        raise ValueError("abstract forward requires the fresh update-0 checkpoint")

    def abstract(value):
        return jax.ShapeDtypeStruct(np.shape(value), np.asarray(value).dtype)

    parameters = jax.tree_util.tree_map(abstract, checkpoint["parameters"])
    stacked = jax.tree_util.tree_map(abstract, assets["stacked"])
    load_seconds = time.perf_counter() - began
    del checkpoint
    objective = make_batched_objective(jax, jnp, assets["objective"])
    began = time.perf_counter()
    with daily.candidate_process_head_patches(shared_process_head, assets["adapter"], assets["backbone"]):
        loss, auxiliary = jax.eval_shape(objective, parameters, stacked)
    return {"abstract_forward_validated": True, "abstract_parameter_load_seconds": load_seconds,
            "abstract_forward_seconds": time.perf_counter() - began,
            "abstract_loss_shape": list(loss.shape), "abstract_loss_dtype": str(loss.dtype),
            "abstract_auxiliary_leaf_shapes": [list(v.shape) for v in jax.tree_util.tree_leaves(auxiliary)],
            "gpu_gradient_compiled": False}


def gpu_run(args, helper, daily, assets, emit):
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load, serialize

    from research.daily_coarse_graining.daily_operator import shared_process_head
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.run_shared_transformer_admission import memory_sizes

    devices = tuple(jax.local_devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("select exactly one visible GPU")
    _require_supported_jax(jax)
    began = time.perf_counter()
    checkpoint = load_learning_checkpoint(assets["checkpoint"], expected_method=assets["reference"]["method"])
    if checkpoint["completed_updates"] != 0:
        raise ValueError("window batching requires fresh update-0 parameters")
    parameters = jax.device_put(checkpoint["parameters"])
    expected_count = int(assets["reference"]["method"]["daily_protocol"]["architecture"]["parameter_count"])
    if sum(np.size(v) for v in jax.tree_util.tree_leaves(parameters)) != expected_count:
        raise ValueError("parameter count differs from architecture")
    cases = tuple(jax.device_put(case) for case in assets["cases"])
    stacked = jax.device_put(assets["stacked"])
    limit = min(helper.MEMORY_LIMIT_BYTES,
                int(assets["reference"]["method"]["configuration"]["gradient_memory_limit_gib"]) * 2**30)
    identity = helper.result_identity(assets, 7, jax.__version__) | {
        "schema_version": "shared_recurrent_window_batch_v1", "artifact_ids": assets["artifact_ids"],
        "windows": 2, "combined_executable": str(assets["combined_executable"]),
        "combined_report_sha256": helper.hashlib.sha256(args.combined_report.read_bytes()).hexdigest(),
    }
    reference_path = args.reference_result or args.output.with_suffix(".reference.pkl")
    candidate_path = args.output.with_suffix(".candidate.pkl")
    executable_path = args.output.with_suffix(".candidate.executable.pkl")
    paths = (reference_path, candidate_path, executable_path, args.output, args.combined_report,
             assets["combined_executable"], args.input_bundle, args.prepared_context, assets["checkpoint"])
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("input and output artifact paths must be distinct")

    def preserve(path, result):
        daily.atomic_pickle(path, {"identity": identity, "result": jax.device_get(result), "warm_seconds": []})
        emit("result_saved", artifact=str(path))

    reference_memory = None
    aggregation_memory = None
    aggregation_compile_seconds = 0.0
    if args.reference_result:
        reference_result, _ = helper.load_reference_result(reference_path, identity)
        reference_times = []
        emit("reference_recovered_without_serial_timing", artifact=str(reference_path))
    else:
        # The single-window executable's saved identity binds it to this exact context.
        with Path(assets["combined"]["candidate_artifact"]).open("rb") as stream:
            saved = pickle.load(stream)
        if saved.get("identity") != helper.result_identity(assets, 7, jax.__version__):
            raise ValueError("combined executable result identity differs from replay inputs")
        del saved
        with assets["combined_executable"].open("rb") as stream:
            trusted = deserialize_and_load(*pickle.load(stream))
        reference_memory = memory_sizes(trusted)
        if reference_memory["estimated_live_bytes"] > limit:
            raise ValueError("single-window reference exceeds memory bound")
        emit("reference_loaded", reference_load_seconds=time.perf_counter() - began,
             reference_memory=reference_memory)
        left = trusted(parameters, *cases[0])
        jax.block_until_ready(left)
        if any(physical_counts(daily, left).values()):
            raise ValueError("first reference window physical counters are nonzero")
        right = trusted(parameters, *cases[1])
        jax.block_until_ready(right)
        if any(physical_counts(daily, right).values()):
            raise ValueError("second reference window physical counters are nonzero")
        began = time.perf_counter()
        aggregation = jax.jit(lambda a, b: aggregate_results(jax, jnp, a, b)).lower(left, right).compile()
        aggregation_compile_seconds = time.perf_counter() - began
        aggregation_memory = memory_sizes(aggregation)
        if reference_memory["estimated_live_bytes"] + aggregation_memory["estimated_live_bytes"] > limit:
            raise ValueError("conservative serial-plus-aggregation live estimate exceeds memory bound")
        reference_result = aggregation(left, right)
        jax.block_until_ready(reference_result)
        preserve(reference_path, reference_result)
        del left, right, reference_result

        def serial():
            return aggregation(trusted(parameters, *cases[0]), trusted(parameters, *cases[1]))

        _, reference_times = benchmark(serial, jax, args.repeats)
        emit("serial_two_window_timing_completed", reference_warm_seconds=reference_times,
             reference_memory=reference_memory, aggregation_memory=aggregation_memory,
             aggregation_compile_seconds=aggregation_compile_seconds,
             serial_two_window_windows_per_second=2 / float(np.mean(reference_times)))
        reference_result, _ = helper.load_reference_result(reference_path, identity)
        del trusted, aggregation
    reference_hard = physical_counts(daily, reference_result)
    if any(reference_hard.values()):
        raise ValueError("saved two-window reference physical counters are nonzero")
    backward = jax.jit(jax.value_and_grad(make_batched_objective(jax, jnp, assets["objective"]), has_aux=True))
    emit("candidate_before_lower")
    candidate, lower_seconds, compile_seconds = compile_candidate(
        backward, parameters, stacked, reference_path,
        daily.candidate_process_head_patches(shared_process_head, assets["adapter"], assets["backbone"]), emit,
    )
    candidate_memory = memory_sizes(candidate)
    emit("candidate_compiled", candidate_memory=candidate_memory, compile_seconds=compile_seconds)
    if candidate_memory["estimated_live_bytes"] > limit:
        raise ValueError("batched candidate exceeds memory bound")
    daily.atomic_pickle(executable_path, serialize(candidate))
    emit("candidate_executable_saved", artifact=str(executable_path))
    candidate_result, candidate_times = benchmark(
        lambda: candidate(parameters, stacked), jax, args.repeats,
        lambda result: preserve(candidate_path, result),
    )
    comparisons, _, passed = helper.compare_candidate(daily, jax, candidate_result, reference_result)
    candidate_hard = physical_counts(daily, candidate_result)
    passed = passed and not any(candidate_hard.values())
    return {"status": "passed" if passed else "parity_failed", "comparisons": comparisons,
            "reference_hard": reference_hard, "candidate_hard": candidate_hard,
            "reference_artifact": str(reference_path), "candidate_artifact": str(candidate_path),
            "candidate_executable": str(executable_path), "reference_memory": reference_memory,
            "aggregation_memory": aggregation_memory, "candidate_memory": candidate_memory,
            "memory_limit_bytes": limit, "aggregation_compile_seconds": aggregation_compile_seconds,
            "lower_seconds": lower_seconds, "compile_seconds": compile_seconds,
            "reference_warm_seconds": reference_times, "candidate_warm_seconds": candidate_times,
            "serial_two_window_windows_per_second": 2 / float(np.mean(reference_times)) if reference_times else None,
            "batched_two_window_windows_per_second": 2 / float(np.mean(candidate_times)),
            "paired_two_window_speedup": float(np.mean(reference_times) / np.mean(candidate_times)) if reference_times else None,
            "all_parameter_gradient_coordinates": comparisons["all_parameter_gradients"]["coordinates"]}


def run(args):
    protected = (args.source_profile, args.input_bundle, args.prepared_context, args.combined_report,
                 args.candidate_module_file, args.backbone_module_file, args.reference_result)
    if args.output.resolve() in {path.resolve() for path in protected if path is not None}:
        raise ValueError("output must not overwrite an input or recovered reference")
    helper, daily = load_helpers()
    started = time.perf_counter()
    report = {"status": "running", "scope": "two_independent_T7_windows_serial_vs_vmap_no_optimizer",
              "horizon": 7, "windows": 2, "scientific_acceptance": False,
              "production_sampling_change": False, "rtol": daily.RTOL, "atol": daily.ATOL}

    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        daily.atomic_json(args.output, report)
        print(f"shared_recurrent_window_batch stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        emit("inspect_assets")
        assets = inspect_assets(args, helper, daily)
        emit("assets_inspected", artifact_ids=assets["artifact_ids"], asset_load_seconds=time.perf_counter() - started)
        if args.inspect_only:
            import jax

            emit("inspect_abstract_forward")
            result = {"status": "inspected", "two_case_trees_shapes_dtypes_validated": True,
                      "staged_adapters_loaded": True, "recurrent_objective_bound": True,
                      **inspect_abstract_forward(jax, daily, assets)}
        else:
            result = gpu_run(args, helper, daily, assets, emit)
        emit("completed", **result)
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("source-profile", "input-bundle", "prepared-context", "combined-report",
                 "candidate-module-file", "backbone-module-file", "output"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--reference-result", type=Path)
    result.add_argument("--inspect-only", action="store_true")
    result.add_argument("--repeats", type=int, default=3)
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    raise SystemExit(0 if outcome["status"] in {"passed", "inspected"} else 1)
