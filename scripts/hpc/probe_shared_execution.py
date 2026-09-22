"""Matched shared-model microbatch speed/gradient/update probe; no retained updates."""
from __future__ import annotations

import json
import pickle
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
    component = serial_daily_component(jax, parameters, samples, evaluator, operations)
    return (
        (component["value"], component["auxiliary"]["hard"]),
        component["gradient"],
    )


def serial_daily_component(jax, parameters, samples, evaluator, operations):
    """Evaluate and retain the complete validated scalar-row reference result."""
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
    return {
        "value": float(np.mean(values)),
        "auxiliary": {"hard": hard, "rows": tuple(auxiliaries)},
        "gradient": operations["daily_mean"](gradients),
    }


class ReferenceCacheMiss(RuntimeError):
    """A staged reference executable was absent from the persistent cache."""

    def __init__(self, component):
        super().__init__(f"persistent executable missing for reference {component}")
        self.component = component


class _BackendCompileRejected(BaseException):
    pass


def compile_cached_lowered(lowered, compiler):
    """Load an executable while proving that no cold backend compile occurred."""
    original_boundary = compiler.compile_or_get_cached
    original_backend = compiler.backend_compile
    counts = {"compile_or_get_cached_calls": 0, "backend_compile_calls": 0}

    def boundary(*args, **kwargs):
        counts["compile_or_get_cached_calls"] += 1
        return original_boundary(*args, **kwargs)

    def reject(*args, **kwargs):
        counts["backend_compile_calls"] += 1
        raise _BackendCompileRejected()

    compiler.compile_or_get_cached = boundary
    compiler.backend_compile = reject
    try:
        try:
            executable = lowered.compile()
        except _BackendCompileRejected:
            return None, {**counts, "persistent_cache_hit": False, "in_process_hit": False}
        return executable, {
            **counts,
            "persistent_cache_hit": counts["compile_or_get_cached_calls"] > 0,
            "in_process_hit": counts["compile_or_get_cached_calls"] == 0,
        }
    finally:
        compiler.compile_or_get_cached = original_boundary
        compiler.backend_compile = original_backend


def selected_microbatch_sizes(stage, config):
    declared = tuple(int(value) for value in config["microbatch_sizes"])
    if stage == "reference":
        return ()
    if stage == "all":
        return declared
    size = int(stage.removeprefix("micro"))
    # Explicit larger stages extend execution coverage without changing the
    # fixed-data reference identity or the original protocol's default sweep.
    if size not in (*declared, 8, 16):
        raise ValueError(f"{stage} is absent from the benchmark protocol")
    return (size,)


def state_identity(*, source_checkpoint, config, artifact_id, jax_version, method):
    return {
        "schema_version": "shared_execution_state_identity_v1",
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_update": int(config["source_update"]),
        "artifact_id": int(artifact_id),
        "configuration": dict(config),
        "jax_version": str(jax_version),
        "method": method,
    }


def _load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def initialize_state_store(path, jax, identity):
    path.mkdir(parents=True, exist_ok=True)
    identity_path = path / "identity.pkl"
    if identity_path.exists():
        saved = _load_pickle(identity_path)
        if saved != identity:
            raise ValueError("saved reference state identity differs from this benchmark")
    else:
        persist_tree(identity_path, jax, identity)
    return path


def _load_stage_record(state_dir, name, identity, *, required=False):
    path = state_dir / f"{name}.pkl"
    if not path.exists():
        if required:
            raise ValueError(f"staged benchmark requires completed {name} state")
        return None
    record = _load_pickle(path)
    if (
        record.get("schema_version") != "shared_execution_stage_state_v1"
        or record.get("identity") != identity
        or record.get("name") != name
    ):
        raise ValueError(f"saved {name} state identity or schema differs")
    return record["payload"]


def _save_stage_record(state_dir, name, identity, payload, jax):
    path = state_dir / f"{name}.pkl"
    persist_tree(path, jax, {
        "schema_version": "shared_execution_stage_state_v1",
        "identity": identity,
        "name": name,
        "payload": payload,
    })
    return path


def pending_reference_components(state_dir, identity):
    return tuple(
        name for name in ("daily", "T7", "T30")
        if _load_stage_record(state_dir, name, identity) is None
    )


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
        method_identity = _method_identity(loaded)
        parameters, optimizer = jax.device_put((saved["parameters"], saved["optimizer"]))
        del saved
        batch = artifacts[0].batch
        daily_batch = batch.one_day
        samples = [jax.tree_util.tree_map(lambda v, i=i: v[i], daily_batch)
                   for i in range(config["daily_count"])]
        seven = prepare_shared_window(head, batch.seven_day, window_index=0, horizon=7)
        thirty = prepare_shared_window(head, batch.thirty_day, window_index=0, horizon=30)
        memory_limit = int(config["compiled_memory_limit_gib"])*2**30

        # The cache-key overlay deliberately replaces this public boundary.
        if hasattr(args, "diagnostic_mode"):
            compile_gradient_evaluators(
                jax=jax, parameters=parameters, daily_objective=daily_objective,
                daily_sample=samples[0], recurrent_objective=recurrent_objective,
                seven_case=seven, thirty_case=thirty,
                memory_limit_bytes=memory_limit, emit=emit)

        state_dir = args.state_dir or args.output.with_suffix(".state")
        identity = state_identity(
            source_checkpoint=args.resume_checkpoint,
            config=config,
            artifact_id=config["artifact_id"],
            jax_version=jax.__version__,
            method=method_identity,
        )
        initialize_state_store(state_dir, jax, identity)
        report.update(stage_request=args.stage, state_dir=str(state_dir))
        selected_sizes = selected_microbatch_sizes(args.stage, config)
        report["selected_microbatch_sizes"] = list(selected_sizes)
        reference = _load_stage_record(
            state_dir, "reference", identity,
            required=args.stage.startswith("micro"),
        )
        require_components = reference is not None and bool(selected_sizes)
        daily_component = _load_stage_record(
            state_dir, "daily", identity,
            required=require_components,
        )
        if require_components:
            for component_name in ("T7", "T30"):
                _load_stage_record(state_dir, component_name, identity, required=True)
        if args.stage == "reference" and reference is not None:
            report["reference"] = reference["report"]
            emit("completed", status="completed", reused_reference=True)
            return report

        operations = make_tree_operations(jax, jnp, config["daily_count"])
        mean_operations = make_tree_operations(jax, jnp, 1)
        update = make_update(daily_protocol)

        daily_backward = jax.jit(jax.value_and_grad(daily_objective, has_aux=True))
        recurrent_backward = jax.jit(jax.value_and_grad(recurrent_objective, has_aux=True))
        definitions = {
            "daily": (daily_backward, (parameters, samples[0])),
            "T7": (recurrent_backward, (parameters, *seven)),
            "T30": (recurrent_backward, (parameters, *thirty)),
        }
        evaluators, graphs = {}, {}

        def evaluator(name):
            if name in evaluators:
                return evaluators[name]
            from jax._src import compiler

            function, arguments = definitions[name]
            emit("reference_lower", component=name)
            lower_started = time.perf_counter()
            lowered = function.lower(*arguments)
            lower_seconds = time.perf_counter() - lower_started
            emit("reference_cache_load", component=name, lower_seconds=lower_seconds)
            load_started = time.perf_counter()
            compiled, cache = compile_cached_lowered(lowered, compiler)
            load_seconds = time.perf_counter() - load_started
            if compiled is None:
                report["reference_graphs"] = graphs
                emit(
                    "reference_cache_miss",
                    status="cache_miss",
                    component=name,
                    cache=cache,
                    lower_seconds=lower_seconds,
                    load_seconds=load_seconds,
                    next_action="populate_or_diagnose_the_named_reference_cache_entry",
                )
                raise ReferenceCacheMiss(name)
            memory = memory_sizes(compiled)
            if memory["estimated_live_bytes"] > memory_limit:
                raise ValueError(f"cached reference {name} exceeds compiled-memory limit")
            graphs[name] = {
                "lower_seconds": lower_seconds,
                "cache_load_seconds": load_seconds,
                "cache": cache,
                "memory": memory,
            }
            report["reference_graphs"] = graphs
            evaluators[name] = compiled
            emit(
                "reference_graph_loaded",
                component=name,
                process_memory=jax.devices()[0].memory_stats(),
            )
            return compiled

        def evaluate_component(name):
            if name == "daily":
                component = serial_daily_component(
                    jax, parameters, samples, evaluator(name), operations
                )
                return jax.device_get(component)
            case = seven if name == "T7" else thirty
            (value, auxiliary), gradient = evaluator(name)(parameters, *case)
            _check_gradient_component(
                name=f"reference_{name}", value=value, auxiliary=auxiliary,
                gradient=gradient, finite=operations["finite"])
            return jax.device_get({
                "value": value, "auxiliary": auxiliary, "gradient": gradient,
            })

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

        if reference is None:
            for name in pending_reference_components(state_dir, identity):
                emit("reference_component_started", component=name)
                component = evaluate_component(name)
                path = _save_stage_record(state_dir, name, identity, component, jax)
                if name == "daily":
                    daily_component = component
                emit(
                    "reference_component_saved", component=name, saved=str(path),
                    process_memory=jax.devices()[0].memory_stats(),
                )
            daily_component = _load_stage_record(state_dir, "daily", identity, required=True)
            for name in ("daily", "T7", "T30"):
                evaluator(name)
            emit("reference_measurement")
            reference_result, reference_timing = timed(
                evaluators["daily"], samples, operations
            )
            reference_parameters, reference_optimizer, reference_update = jax.device_get(
                reference_result
            )
            reference_report = {
                **reference_timing,
                "process_memory": jax.devices()[0].memory_stats(),
                "full_update_wall_seconds": reference_timing["median_seconds"],
                "saved": str(state_dir / "reference.pkl"),
            }
            reference = {
                "parameters": reference_parameters,
                "optimizer": reference_optimizer,
                "update": reference_update,
                "report": reference_report,
            }
            reference_path = _save_stage_record(
                state_dir, "reference", identity, reference, jax
            )
            reference_report["saved"] = str(reference_path)
            report["reference"] = reference_report
            emit("reference_saved")
            del reference_result
        else:
            report["reference"] = reference["report"]

        if args.stage == "reference":
            emit("completed", status="completed", reused_reference=False)
            return report

        micro_objective = make_microbatch_objective(jax, jnp, daily_objective)
        comparisons = dict(rtol=config["comparison_rtol"], atol=config["comparison_atol"])
        for size in selected_sizes:
            name = f"micro{size}"
            previous = _load_stage_record(state_dir, name, identity)
            if previous is not None and previous.get("status") == "passed":
                report["variants"][name] = previous
                emit("microbatch_reused", current_variant=name)
                continue

            for component_name in ("daily", "T7", "T30"):
                evaluator(component_name)
            emit("matched_scalar_measurement", current_variant=name)
            scalar_result, scalar_timing = timed(
                evaluators["daily"], samples, operations
            )
            scalar_parameters, scalar_optimizer, scalar_update = jax.device_get(scalar_result)
            scalar_parameters_match = compare_trees(
                jax, scalar_parameters, reference["parameters"], **comparisons
            )
            scalar_optimizer_match = compare_trees(
                jax, scalar_optimizer, reference["optimizer"], **comparisons
            )
            scalar_joint_match = bool(np.isclose(
                scalar_update["loss"]["joint"], reference["update"]["loss"]["joint"],
                **comparisons))
            if not (scalar_parameters_match["passed"] and scalar_optimizer_match["passed"]
                    and scalar_joint_match):
                raise ValueError("same-allocation scalar reference differs from saved reference")

            pending = {
                "status": "compile_pending",
                "matched_scalar_timing": scalar_timing,
            }
            _save_stage_record(state_dir, name, identity, pending, jax)
            del scalar_result, scalar_parameters, scalar_optimizer, scalar_update
            emit("compile_microbatch", current_variant=name)
            start = time.perf_counter()
            example = jax.tree_util.tree_map(lambda v: v[:size], daily_batch)
            compiled = compile_microbatch_evaluator(
                jax=jax, parameters=parameters, microbatch_objective=micro_objective,
                microbatch_example=example)
            memory = memory_sizes(compiled)
            result = dict(
                compile_seconds=time.perf_counter()-start,
                compiled_memory=memory,
                matched_scalar_timing=scalar_timing,
                matched_scalar_full_update_wall_seconds=scalar_timing["median_seconds"],
                scalar_reference_parameters=scalar_parameters_match,
                scalar_reference_optimizer=scalar_optimizer_match,
                scalar_reference_joint_scalar_pass=scalar_joint_match,
            )
            report["variants"][name] = result
            if memory["estimated_live_bytes"] > memory_limit:
                result.update(status="resource_limited", reason="compiled_memory_over_24GiB")
                _save_stage_record(state_dir, name, identity, result, jax)
                emit("microbatch_resource_limited")
                del compiled
                continue

            daily_evaluator = make_daily_accumulator(
                jax=jax, jnp=jnp, microbatch_size=size, evaluator=compiled,
                expected_count=config["daily_count"])

            (value, auxiliary), gradient = daily_evaluator(parameters, daily_batch)
            gradient_comparison = compare_trees(
                jax, gradient, daily_component["gradient"], **comparisons)
            scalar_pass = bool(np.isclose(float(value), daily_component["value"], **comparisons))
            hard_pass = all(
                int(auxiliary["hard"][key]) == value
                for key, value in daily_component["auxiliary"]["hard"].items())
            result.update(gradient_comparison=gradient_comparison,
                          scalar_pass=scalar_pass, hard_pass=hard_pass)
            del gradient
            if not (gradient_comparison["passed"] and scalar_pass and hard_pass):
                result.update(status="rejected", reason="daily_numerical_comparison")
                _save_stage_record(state_dir, name, identity, result, jax)
                emit("microbatch_rejected")
                del daily_evaluator, compiled
                continue
            proposal, timing = timed(daily_evaluator, [daily_batch], mean_operations)
            parameter_comparison = compare_trees(
                jax, proposal[0], reference["parameters"], **comparisons)
            optimizer_comparison = compare_trees(
                jax, proposal[1], reference["optimizer"], **comparisons)
            joint_pass = bool(np.isclose(proposal[2]["loss"]["joint"],
                                        reference["update"]["loss"]["joint"], **comparisons))
            passed = parameter_comparison["passed"] and optimizer_comparison["passed"] and joint_pass
            result.update(status="passed" if passed else "rejected", timing=timing,
                          full_update_wall_seconds=timing["median_seconds"],
                          parameters=parameter_comparison, optimizer=optimizer_comparison,
                          joint_scalar_pass=joint_pass,
                          speedup=scalar_timing["median_seconds"]/timing["median_seconds"],
                          process_memory_cumulative=jax.devices()[0].memory_stats())
            _save_stage_record(state_dir, name, identity, result, jax)
            emit("microbatch_completed")
            del proposal, daily_evaluator, compiled
        admitted = {name: r for name, r in report["variants"].items() if r["status"] == "passed"}
        faster = {name: r for name, r in admitted.items() if r["speedup"] > 1}
        selected = min(faster, key=lambda name: faster[name]["timing"]["median_seconds"]) if faster else None
        emit("completed", status="completed", selected_faster_variant=selected,
             scope="fixed_checkpoint_fixed_batch_execution_only",
             next_gate="measured_dual_device_split_after_daily_microbatch_result")
        return report
    except ReferenceCacheMiss:
        return report
    except Exception as error:
        emit("error", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser():
    result = learning_parser()
    result.description = __doc__
    result.add_argument("--benchmark-protocol", type=Path, required=True)
    result.add_argument(
        "--stage", choices=("reference", "micro2", "micro4", "micro8", "micro16", "all"), default="all"
    )
    result.add_argument("--state-dir", type=Path)
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    raise SystemExit(0 if outcome["status"] in ("passed", "completed") else 2)
