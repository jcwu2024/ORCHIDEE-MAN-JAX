"""Matched shared-model microbatch speed/gradient/update probe; no retained updates."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc.run_shared_learning import (
    _check_gradient_component,
    _method_identity,
    compile_gradient_evaluators,
    execute_learning_update,
    load_learning_checkpoint,
    make_tree_operations,
    memory_sizes,
    persist_tree,
    shared_config,
)
from scripts.hpc.run_shared_learning import (
    parser as learning_parser,
)


def compare_trees(jax, actual, reference, *, rtol, atol):
    left, left_tree = jax.tree_util.tree_flatten(actual)
    right, right_tree = jax.tree_util.tree_flatten(reference)
    if left_tree != right_tree:
        return dict(passed=False, reason="tree_structure")
    failed = total = 0
    delta_sq = reference_sq = maximum = 0.0
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return dict(passed=False, reason="shape_or_dtype")
        a, b = a.astype(np.float64), b.astype(np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        delta = a-b
        failed += int(np.sum(~finite | (abs(delta) > atol+rtol*abs(b))))
        total += a.size
        # Invalid trees remain rejected without introducing NaN into JSON.
        valid_delta = np.where(finite, delta, 0.)
        delta_sq += float(np.sum(valid_delta**2))
        reference_sq += float(np.sum(np.where(finite, b, 0.)**2))
        maximum = max(maximum, float(np.max(abs(valid_delta), initial=0.)))
    return dict(passed=failed == 0, failed_coordinates=failed, coordinates=total,
                maximum_absolute=maximum,
                relative_l2=float(np.sqrt(delta_sq/max(reference_sq, 1e-300))),
                rtol=rtol, atol=atol)


def serial_daily(jax, parameters, samples, evaluator, operations):
    """Reference the admitted scalar graph and sequential sum, without vmap."""
    gradients = operations["zero"](parameters)
    values, auxiliaries = [], []
    for index, sample in enumerate(samples):
        (value, auxiliary), gradient = evaluator(parameters, sample)
        _check_gradient_component(name="reference_daily", value=value,
                                  auxiliary=auxiliary, gradient=gradient,
                                  finite=operations["finite"], row=index)
        gradients = operations["add"](gradients, gradient)
        values.append(float(value))
        auxiliaries.append(jax.device_get(auxiliary))
    hard = {key: sum(int(aux["hard"][key]) for aux in auxiliaries)
            for key in auxiliaries[0]["hard"]}
    return ((float(np.mean(values)), hard), operations["daily_mean"](gradients))


def run(args):
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.shared_learning_plan import load_learning_configuration

    began = time.perf_counter()
    report = dict(status="running", retained_training_updates=0,
                  scientific_acceptance=False, variants={}, source_checkpoint=str(args.resume_checkpoint),
                  source_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  host=platform.node(), python_version=platform.python_version())

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-began, **values)
        _atomic_json(args.output, report)
        print(f"shared_execution stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        config = json.loads(args.benchmark_protocol.read_text(encoding="utf-8"))
        if config["schema_version"] != "shared_execution_benchmark_v1":
            raise ValueError("unsupported execution benchmark")
        if args.resume_checkpoint is None:
            raise ValueError("trained source checkpoint required")
        loaded = load_learning_configuration(args.protocol, args.joint_admission_report,
                                            args.prepared_manifest, args.validation_manifest)
        saved = load_learning_checkpoint(args.resume_checkpoint, expected_method=_method_identity(loaded))
        if saved["completed_updates"] != config["source_update"]:
            raise ValueError("benchmark requires the declared update100 source")
        emit("source_verified", configuration=config, source_update=saved["completed_updates"])
        if args.prepare_only:
            emit("completed", status="passed", scope="checkpoint_and_metadata_only")
            return report

        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
        from research.daily_coarse_graining.daily_operator.shared_persistent_outputs import (
            build_shared_persistent_coordinates,
        )
        from research.daily_coarse_graining.daily_operator.shared_process_head import build_shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
            make_shared_joint_daily_objective,
            make_shared_recurrent_objective,
            prepare_shared_window,
        )
        from research.daily_coarse_graining.daily_operator.shared_training import derive_shared_state_change_scales
        from scripts.hpc.shared_daily_microbatch import (
            compile_microbatch_evaluator,
            make_daily_accumulator,
            make_microbatch_objective,
        )
        from scripts.hpc.shared_transformer_runtime import load_shared_runtime

        emit("load_runtime")
        runtime, parent, template, artifacts, _metadata = load_shared_runtime(args, (config["artifact_id"],))
        jax = runtime["jax"]
        import jax.numpy as jnp

        report.update(jax_version=jax.__version__, devices=[str(d) for d in jax.devices()])

        daily_protocol = loaded["daily_protocol"]
        coordinates = build_shared_persistent_coordinates(
            parent.learned_schema, parent.contract, runtime["base_statistics"].arrays["state_delta"])
        head = build_shared_process_head(
            parent, template, config=shared_config(daily_protocol["architecture"]),
            thermal_transform=daily_protocol["architecture"]["thermal_transform"],
            persistent_coordinates=coordinates, source_bare_cimean=True)
        state_scales = derive_shared_state_change_scales(runtime, head)
        daily_objective = make_shared_joint_daily_objective(
            runtime, head, template, state_scales, loaded["admission"]["scales"])
        _, recurrent_objective = make_shared_recurrent_objective(
            head, template, state_scales, loaded["admission"]["scales"])
        parameters, optimizer = jax.device_put((saved["parameters"], saved["optimizer"]))
        del saved
        batch = artifacts[0].batch
        daily_batch = batch.one_day
        samples = [jax.tree_util.tree_map(lambda v, i=i: v[i], daily_batch)
                   for i in range(config["daily_count"])]
        seven = prepare_shared_window(head, batch.seven_day, window_index=0, horizon=7)
        thirty = prepare_shared_window(head, batch.thirty_day, window_index=0, horizon=30)
        memory_limit = int(config["compiled_memory_limit_gib"])*2**30
        evaluators, graphs = compile_gradient_evaluators(
            jax=jax, parameters=parameters, daily_objective=daily_objective,
            daily_sample=samples[0], recurrent_objective=recurrent_objective,
            seven_case=seven, thirty_case=thirty, memory_limit_bytes=memory_limit, emit=emit)
        report["reference_graphs"] = graphs
        operations = make_tree_operations(jax, jnp, config["daily_count"])
        mean_operations = make_tree_operations(jax, jnp, 1)
        update = make_update(daily_protocol)

        def execute(daily_evaluator, inputs, ops):
            return execute_learning_update(
                jax=jax, parameters=parameters, optimizer=optimizer, daily_samples=inputs,
                seven_case=seven, thirty_case=thirty,
                evaluators={**evaluators, "daily": daily_evaluator}, operations=ops, update=update)

        def timed(daily_evaluator, inputs, ops):
            for _ in range(config["warmups"]):
                result = execute(daily_evaluator, inputs, ops)
            rows = []
            for _ in range(config["timed_replays"]):
                start = time.perf_counter()
                result = execute(daily_evaluator, inputs, ops)
                jax.block_until_ready(result[:2])
                rows.append(dict(wall_seconds=time.perf_counter()-start, **result[2]))
            return result, dict(replays=rows, median_seconds=float(np.median(
                [r["wall_seconds"] for r in rows])))

        emit("reference_measurement")
        (reference_result, reference_timing) = timed(evaluators["daily"], samples, operations)
        reference_parameters, reference_optimizer, reference_update = jax.device_get(reference_result)
        (reference_value, reference_hard), reference_gradient = serial_daily(
            jax, parameters, samples, evaluators["daily"], operations)
        reference_gradient = jax.device_get(reference_gradient)
        reference_path = args.output.with_suffix(".reference.pkl")
        persist_tree(reference_path, jax, dict(
            parameters=reference_parameters, optimizer=reference_optimizer,
            daily_gradient=reference_gradient, daily_loss=reference_value, hard=reference_hard,
            update=reference_update, config=config, source_checkpoint=str(args.resume_checkpoint)))
        del reference_result
        report["reference"] = dict(**reference_timing, saved=str(reference_path),
                                   process_memory=jax.devices()[0].memory_stats())
        emit("reference_saved")
        micro_objective = make_microbatch_objective(jax, jnp, daily_objective)
        comparisons = dict(rtol=config["comparison_rtol"], atol=config["comparison_atol"])
        for size in config["microbatch_sizes"]:
            name = f"micro{size}"
            emit("compile_microbatch", current_variant=name)
            start = time.perf_counter()
            example = jax.tree_util.tree_map(lambda v: v[:size], daily_batch)
            compiled = compile_microbatch_evaluator(
                jax=jax, parameters=parameters, microbatch_objective=micro_objective,
                microbatch_example=example)
            memory = memory_sizes(compiled)
            result = dict(compile_seconds=time.perf_counter()-start, compiled_memory=memory)
            report["variants"][name] = result
            if memory["estimated_live_bytes"] > memory_limit:
                result.update(status="resource_limited", reason="compiled_memory_over_24GiB")
                emit("microbatch_resource_limited")
                del compiled
                continue

            daily_evaluator = make_daily_accumulator(
                jax=jax, jnp=jnp, microbatch_size=size, evaluator=compiled,
                expected_count=config["daily_count"])

            (value, auxiliary), gradient = daily_evaluator(parameters, daily_batch)
            gradient_comparison = compare_trees(jax, gradient, reference_gradient, **comparisons)
            scalar_pass = bool(np.isclose(float(value), reference_value, **comparisons))
            hard_pass = all(int(auxiliary["hard"][k]) == v for k, v in reference_hard.items())
            result.update(gradient_comparison=gradient_comparison,
                          scalar_pass=scalar_pass, hard_pass=hard_pass)
            del gradient
            if not (gradient_comparison["passed"] and scalar_pass and hard_pass):
                result.update(status="rejected", reason="daily_numerical_comparison")
                emit("microbatch_rejected")
                del daily_evaluator, compiled
                continue
            proposal, timing = timed(daily_evaluator, [daily_batch], mean_operations)
            parameter_comparison = compare_trees(jax, proposal[0], reference_parameters, **comparisons)
            optimizer_comparison = compare_trees(jax, proposal[1], reference_optimizer, **comparisons)
            joint_pass = bool(np.isclose(proposal[2]["loss"]["joint"],
                                        reference_update["loss"]["joint"], **comparisons))
            passed = parameter_comparison["passed"] and optimizer_comparison["passed"] and joint_pass
            result.update(status="passed" if passed else "rejected", timing=timing,
                          parameters=parameter_comparison, optimizer=optimizer_comparison,
                          joint_scalar_pass=joint_pass,
                          speedup=reference_timing["median_seconds"]/timing["median_seconds"],
                          process_memory_cumulative=jax.devices()[0].memory_stats())
            emit("microbatch_completed")
            del proposal, daily_evaluator, compiled
        admitted = {name: r for name, r in report["variants"].items() if r["status"] == "passed"}
        faster = {name: r for name, r in admitted.items() if r["speedup"] > 1}
        selected = min(faster, key=lambda name: faster[name]["timing"]["median_seconds"]) if faster else None
        emit("completed", status="completed", selected_faster_variant=selected,
             scope="fixed_checkpoint_fixed_batch_execution_only",
             next_gate="measured_dual_device_split_after_daily_microbatch_result")
        return report
    except Exception as error:
        emit("error", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser():
    result = learning_parser()
    result.description = __doc__
    result.add_argument("--benchmark-protocol", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
