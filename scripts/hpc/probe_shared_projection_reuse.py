"""Bounded dynamic-projection reuse admission against saved real-model gradients."""

from __future__ import annotations

import gzip
import json
import pickle
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.hpc.probe_shared_execution import (
    _load_stage_record,
    compare_trees,
    state_identity,
)
from scripts.hpc.probe_shared_execution import parser as base_parser
from scripts.hpc.run_shared_learning import (
    _check_gradient_component,
    _method_identity,
    execute_learning_update,
    load_learning_checkpoint,
    make_tree_operations,
    memory_sizes,
    persist_tree,
    shared_config,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

EXECUTION = "gather_dynamic_reuse"
GRADIENT_TOLERANCE = {"rtol": 5e-4, "atol": 2e-5}
VALUE_TOLERANCE = {"rtol": 2e-5, "atol": 2e-6}
UPDATE_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}


def compare_component(jax, result, reference):
    (value, auxiliary), gradient = result
    hard = {key: int(value) for key, value in jax.device_get(auxiliary["hard"]).items()}
    expected = {key: int(value) for key, value in reference["auxiliary"]["hard"].items()}
    gradient_match = compare_trees(jax, gradient, reference["gradient"], **GRADIENT_TOLERANCE)
    value_match = bool(np.isfinite(float(value)) and np.isclose(float(value), reference["value"], **VALUE_TOLERANCE))
    hard_match = hard == expected and not any(hard.values())
    return {
        "passed": value_match and hard_match and gradient_match["passed"],
        "value": float(value),
        "reference_value": float(reference["value"]),
        "value_passed": value_match,
        "hard": hard,
        "hard_passed": hard_match,
        "gradient": gradient_match,
    }


def _candidate_record(path, identity):
    with path.open("rb") as handle:
        record = pickle.load(handle)
    if record.get("identity") != identity or record.get("status") != "passed" or record.get("component") != path.stem:
        raise ValueError(f"candidate component identity/admission mismatch: {path}")
    return record


def run(args):
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
    from scripts.hpc.shared_daily_microbatch import make_daily_accumulator, make_microbatch_objective
    from scripts.hpc.shared_learning_plan import load_learning_configuration
    from scripts.hpc.shared_transformer_runtime import load_shared_runtime

    began = time.perf_counter()
    report = {
        "status": "running",
        "retained_training_updates": 0,
        "scientific_acceptance": False,
        "execution": EXECUTION,
        "host": platform.node(),
        "components": {},
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "value_tolerance": VALUE_TOLERANCE,
        "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - began)
        _atomic_json(args.output, report)
        print(f"projection_reuse stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        emit("source_metadata")
        config = json.loads(args.benchmark_protocol.read_text(encoding="utf-8"))
        loaded = load_learning_configuration(
            args.protocol, args.joint_admission_report, args.prepared_manifest, args.validation_manifest
        )
        method = _method_identity(loaded)
        saved = load_learning_checkpoint(args.resume_checkpoint, expected_method=method)
        if saved["completed_updates"] != config["source_update"] or config["daily_count"] != 32:
            raise ValueError("source checkpoint/daily count differs from the saved reference")
        if args.prepare_only:
            emit("completed", status="passed", scope="checkpoint_and_method_metadata_only")
            return report
        emit("load_runtime")
        runtime, parent, template, artifacts, _ = load_shared_runtime(args, (config["artifact_id"],))
        jax = runtime["jax"]
        import jax.numpy as jnp
        from jax.experimental.serialize_executable import deserialize_and_load, serialize

        source_identity = state_identity(
            source_checkpoint=args.resume_checkpoint,
            config=config,
            artifact_id=config["artifact_id"],
            jax_version=jax.__version__,
            method=method,
        )
        identity = {
            "source": source_identity,
            "execution": EXECUTION,
            "microbatch_size": args.microbatch_size,
            "candidate_source": report["source_git_head"],
        }
        report["jax_version"] = jax.__version__
        reference_dir = args.state_dir
        reference_names = ("daily", "T7", "T30", "reference") if "joint" in args.components else tuple(args.components)
        references = {
            name: _load_stage_record(reference_dir, name, source_identity, required=True) for name in reference_names
        }
        parameters, optimizer = jax.device_put((saved["parameters"], saved["optimizer"]))
        del saved
        architecture = replace(shared_config(loaded["daily_protocol"]["architecture"]), decoder_execution=EXECUTION)
        coordinates = build_shared_persistent_coordinates(
            parent.learned_schema, parent.contract, runtime["base_statistics"].arrays["state_delta"]
        )
        head = build_shared_process_head(
            parent,
            template,
            config=architecture,
            thermal_transform=loaded["daily_protocol"]["architecture"]["thermal_transform"],
            persistent_coordinates=coordinates,
            source_bare_cimean=True,
        )
        state_scales = derive_shared_state_change_scales(runtime, head)
        daily = make_shared_joint_daily_objective(runtime, head, template, state_scales, loaded["admission"]["scales"])
        _, recurrent = make_shared_recurrent_objective(head, template, state_scales, loaded["admission"]["scales"])
        batch = artifacts[0].batch
        cases = {
            "T7": prepare_shared_window(head, batch.seven_day, window_index=0, horizon=7),
            "T30": prepare_shared_window(head, batch.thirty_day, window_index=0, horizon=30),
        }
        sample = jax.tree_util.tree_map(lambda value: value[: args.microbatch_size], batch.one_day)
        definitions = {
            "daily": (make_microbatch_objective(jax, jnp, daily), (parameters, sample)),
            **{name: (recurrent, (parameters, *case)) for name, case in cases.items()},
        }
        operations = make_tree_operations(jax, jnp, 1)
        limit = int(config["compiled_memory_limit_gib"]) * 2**30
        args.candidate_dir.mkdir(parents=True, exist_ok=True)

        def evaluate_component(name, executable):
            if name == "daily":
                return make_daily_accumulator(
                    jax=jax, jnp=jnp, microbatch_size=args.microbatch_size, evaluator=executable, expected_count=32
                )
            return executable

        def invocation(name, evaluator):
            return evaluator(parameters, batch.one_day) if name == "daily" else evaluator(parameters, *cases[name])

        for name in args.components:
            if name == "joint":
                evaluators = {}
                loaded_memory = {}
                resident_memory = {}
                emit("load_saved_candidate_executables")
                for component in ("daily", "T7", "T30"):
                    record = _candidate_record(args.candidate_dir / f"{component}.pkl", identity)
                    with (args.candidate_dir / f"{component}.executable.pkl").open("rb") as handle:
                        package = pickle.load(handle)
                    if package["identity"] != identity or package["component"] != component:
                        raise ValueError("serialized executable source/component mismatch")
                    compiled = deserialize_and_load(*package["executable"])
                    loaded_memory[component] = memory_sizes(compiled)
                    if loaded_memory[component]["estimated_live_bytes"] > limit:
                        raise ValueError("loaded candidate executable exceeds memory envelope")
                    evaluators[component] = evaluate_component(component, compiled)
                    del record
                    resident_memory[component] = jax.devices()[0].memory_stats()
                # Sequential components reuse workspace; do not sum their peaks.
                resident = resident_memory["T30"] or {}
                projected = int(resident.get("bytes_in_use", 0)) + max(
                    sizes["temp_size_in_bytes"] + sizes["output_size_in_bytes"] for sizes in loaded_memory.values()
                )
                emit(
                    "joint_memory_checked",
                    joint_memory={
                        "after_each_load": resident_memory,
                        "resident_plus_largest_workspace_and_output_bytes": projected,
                    },
                )
                if projected > limit:
                    emit("resource_limited", status="resource_limited")
                    return report
                update = make_update(loaded["daily_protocol"])

                def joint():
                    return execute_learning_update(
                        jax=jax,
                        parameters=parameters,
                        optimizer=optimizer,
                        daily_samples=[batch.one_day],
                        seven_case=cases["T7"],
                        thirty_case=cases["T30"],
                        evaluators=evaluators,
                        operations=operations,
                        update=update,
                    )

                result = joint()
                timings = []
                for _ in range(3):
                    started = time.perf_counter()
                    result = joint()
                    jax.block_until_ready(result[:2])
                    timings.append({"wall_seconds": time.perf_counter() - started, **result[2]})
                expected = references["reference"]
                p = compare_trees(jax, result[0], expected["parameters"], **UPDATE_TOLERANCE)
                o = compare_trees(jax, result[1], expected["optimizer"], **UPDATE_TOLERANCE)
                loss = bool(
                    np.isclose(result[2]["loss"]["joint"], expected["update"]["loss"]["joint"], **VALUE_TOLERANCE)
                )
                report["components"][name] = {
                    "status": "passed" if p["passed"] and o["passed"] and loss else "rejected",
                    "parameters": p,
                    "optimizer": o,
                    "loss_passed": loss,
                    "timings": timings,
                    "memory": jax.devices()[0].memory_stats(),
                }
                emit("joint_completed")
                continue
            objective, arguments = definitions[name]
            if args.profile_saved:
                # Load the admitted exact-runtime artifact without tracing or
                # lowering the model again. Profile one microbatch/window only.
                emit("load_saved_profile_executable", current_component=name)
                with (args.candidate_dir / f"{name}.executable.pkl").open("rb") as handle:
                    package = pickle.load(handle)
                if package["identity"] != identity or package["component"] != name:
                    raise ValueError("profile executable source/component mismatch")
                compiled = deserialize_and_load(*package["executable"])
                memory = memory_sizes(compiled)
                if memory["estimated_live_bytes"] > limit:
                    raise ValueError("profile executable exceeds admitted memory envelope")
                row = {
                    "status": "running",
                    "compiled_memory": memory,
                    "profile_sample_count": args.microbatch_size if name == "daily" else 1,
                    "model_lowering_or_compilation": False,
                }
                report["components"][name] = row
                emit("profile_warmup")
                result = jax.block_until_ready(compiled(*arguments))
                times = []
                for _ in range(3):
                    started = time.perf_counter()
                    result = jax.block_until_ready(compiled(*arguments))
                    times.append(time.perf_counter() - started)
                row["warm_seconds"] = times
                row["warm_median_seconds"] = float(np.median(times))
                row["value"] = float(result[0][0])
                trace_dir = args.output.with_suffix(f".{name}.trace")
                emit("profile_started", trace_directory=str(trace_dir))
                with jax.profiler.trace(trace_dir, create_perfetto_link=False):
                    with jax.profiler.TraceAnnotation("shared_cached_gradient", component=name):
                        result = jax.block_until_ready(compiled(*arguments))
                row["trace_files"] = [str(p) for p in trace_dir.rglob("*") if p.is_file()]
                hlo_path = args.output.with_suffix(f".{name}.hlo.txt.gz")
                with gzip.open(hlo_path, "wt", encoding="utf-8") as handle:
                    handle.write(compiled.as_text())
                row.update(status="passed", optimized_hlo=str(hlo_path), process_memory=jax.devices()[0].memory_stats())
                emit("profile_completed")
                del result, compiled, package
                continue
            emit("compile_started", current_component=name)
            started = time.perf_counter()
            compiled = jax.jit(jax.value_and_grad(objective, has_aux=True)).lower(*arguments).compile()
            memory = memory_sizes(compiled)
            row = {"compile_seconds": time.perf_counter() - started, "compiled_memory": memory}
            report["components"][name] = row
            emit("compiled")
            if memory["estimated_live_bytes"] > limit:
                # Preserve an expensive rejected compile for memory inspection.
                # It has no passed component record and cannot enter joint replay.
                package = {"identity": identity, "component": name, "executable": serialize(compiled)}
                persist_tree(args.candidate_dir / f"{name}.resource_limited.executable.pkl", jax, package)
                row.update(status="resource_limited")
                emit("resource_limited", status="resource_limited")
                return report
            evaluator = evaluate_component(name, compiled)
            emit("compare_saved_reference")
            result = jax.block_until_ready(invocation(name, evaluator))
            _check_gradient_component(
                name=name, value=result[0][0], auxiliary=result[0][1], gradient=result[1], finite=operations["finite"]
            )
            comparison = compare_component(jax, result, references[name])
            row["comparison"] = comparison
            if not comparison["passed"]:
                row.update(status="rejected")
                emit("numerical_rejected", status="rejected")
                return report
            # The completed gradient survives any later timing or serialization failure.
            persist_tree(
                args.candidate_dir / f"{name}.pkl",
                jax,
                {
                    "status": "passed",
                    "identity": identity,
                    "component": name,
                    "result": result,
                    "comparison": comparison,
                },
            )
            emit("gradient_saved")
            times = []
            for _ in range(3):
                started = time.perf_counter()
                result = jax.block_until_ready(invocation(name, evaluator))
                times.append(time.perf_counter() - started)
            row.update(
                status="passed",
                warm_seconds=times,
                warm_median_seconds=float(np.median(times)),
                process_memory=jax.devices()[0].memory_stats(),
            )
            emit("timing_saved")
            # Keep an exact-runtime executable for the later full-update check.
            # This avoids rebuilding the graph solely to combine admitted components.
            package = {"identity": identity, "component": name, "executable": serialize(compiled)}
            persist_tree(args.candidate_dir / f"{name}.executable.pkl", jax, package)
            emit("executable_saved")
            del result, evaluator, compiled, package
        passed = all(row.get("status") == "passed" for row in report["components"].values())
        emit(
            "completed",
            status="passed" if passed else "rejected",
            scope=(
                "saved_executable_profile_not_new_parity_or_scientific_acceptance"
                if args.profile_saved
                else "execution_parity_and_free_node_diagnostic_timing_not_scientific_acceptance"
            ),
        )
        return report
    except Exception as error:
        emit("error", status="error", error=f"{type(error).__name__}: {error}")
        raise


def parser():
    result = base_parser()
    result.description = __doc__
    result.add_argument("--components", choices=("daily", "T7", "T30", "joint"), nargs="+", default=["daily"])
    result.add_argument("--candidate-dir", required=True, type=Path)
    result.add_argument("--microbatch-size", type=int, choices=(2, 4), default=4)
    result.add_argument(
        "--profile-saved",
        action="store_true",
        help="profile one saved exact-runtime executable per component; no model compile",
    )
    return result


if __name__ == "__main__":
    argument_parser = parser()
    arguments = argument_parser.parse_args()
    if arguments.state_dir is None or arguments.resume_checkpoint is None:
        argument_parser.error("saved --state-dir and --resume-checkpoint are required")
    if arguments.profile_saved and "joint" in arguments.components:
        argument_parser.error("--profile-saved profiles individual components only")
    outcome = run(arguments)
    raise SystemExit(0 if outcome["status"] == "passed" else 2)
