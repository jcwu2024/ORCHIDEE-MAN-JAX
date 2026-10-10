"""Bounded fresh latent32/reader64 full parameter and initial-state AD probe.

Uses the public opt-in decoder, unchanged public physical/loss boundaries and
native checkpointed AD. Original architecture timing is comparative evidence,
never a cross-architecture numerical-parity or scientific accuracy gate.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def validate_inputs(jax, serialized, executable, arguments):
    supplied = (arguments, {})
    if serialized[1] != jax.tree_util.tree_structure(supplied):
        raise ValueError("original full AD input treedef differs")
    infos, tree = jax.tree_util.tree_flatten(executable.args_info)
    values, other = jax.tree_util.tree_flatten(supplied)
    if tree != other: raise ValueError("original full AD args_info tree differs")
    for info, value in zip(infos, values, strict=True):
        if tuple(info._aval.shape) != np.shape(value) or np.dtype(info._aval.dtype) != np.asarray(value).dtype:
            raise ValueError("original full AD input shape/dtype differs")

def run(args):
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    from jax.experimental.serialize_executable import serialize, deserialize_and_load
    overlay = Path(__file__).parent
    # Keep the serialized module/class identities while replacing only declared
    # source modules. Install the transformer before importing the head owner.
    load(overlay / "training_artifact_io.py", "scripts.hpc.training_artifact_io")
    package = "research.daily_coarse_graining.daily_operator."
    load(overlay / "shared_transformer.py", package + "shared_transformer")
    latent = load(overlay / "shared_latent_reader.py", package + "shared_latent_reader")
    owner = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    gates = load(args.gate_owner, "latent_reader_admission_utilities")
    forward = load(args.forward_probe, "scripts.hpc.probe_shared_forward_reverse")
    compiler_options = {"xla_gpu_enable_llvm_module_compilation_parallelism": True}
    if args.force_compilation_parallelism:
        compiler_options["xla_gpu_force_compilation_parallelism"] = args.force_compilation_parallelism
    report = {"status": "running", "component": args.component,
        "architecture": "shared_latent32x384_reader64x2", "training_updates": 0,
        "precision": "FP32_neural_FP64_physics", "native_AD_argnums": [0, 1],
        "initialization": "fresh_public_whole_tree", "production_adoption": False,
        "rtol": 1e-5, "atol": 1e-6, "gpu_count": len(devices),
        "compiler_options": compiler_options}
    replay_failures = []
    if args.diagnostic_continue_replay_failures:
        prior = json.loads(args.diagnostic_prior_failure_report.read_text())
        if prior.get("status") != "numerical_failed" or prior.get("executable") != str(args.saved_executable):
            raise ValueError("diagnostic recovery requires the preserved failed candidate report")
        replay_failures.append({"scope": "preserved_prior_strict_replay_failure",
            "artifact": str(args.diagnostic_prior_failure_report), "error": prior.get("error"),
            "comparisons": prior.get("same_architecture_replays"),
            "raw_failed_replay_was_not_saved": True})
        report.update(diagnostic_recovery=True, numerical_admission_blocked=True,
            preserved_numerical_failures=replay_failures)
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **fields)
        owner.atomic_json(args.output, report)
        print(stage, flush=True)
    if args.output.exists(): raise FileExistsError(args.output)
    try:
        emit("starting")
        if len(devices) != 1 or jax.default_backend() != "gpu":
            raise ValueError("one actual free-node GPU required")
        preflight = gates.validate_runtime_report(args.runtime_preflight, jax.__version__)
        profile, reference, bundle, context, checkpoint_path, admissions = forward._assets(args)
        from research.daily_coarse_graining.daily_operator import shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
            make_shared_recurrent_objective, make_shared_joint_daily_objective)
        from scripts.hpc.shared_daily_microbatch import make_microbatch_objective
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        import jax.numpy as jnp
        head = replace(context["head"], shared_config=latent.shared_latent_reader_config(context["head"].shared_config))
        if head.shared_config.latent_reader_execution != "field":
            raise ValueError("this admitted prototype requires canonical field execution")
        seed = int(reference["method"]["daily_protocol"]["seed"])
        inputs = jax.tree_util.tree_map(lambda value: value[0], context["daily4"].inputs)
        inputs = head.input_assembler.assemble(inputs)
        identity = {"source_head": reference["source_git_head"], "artifact_id": int(bundle["artifact_ids"][0]),
            "input_source_method": reference["method"], "seed": seed, "candidate_config": vars(head.shared_config),
            "candidate_source_sha256": {name: hashlib.sha256((overlay/name).read_bytes()).hexdigest()
                for name in ("shared_transformer.py", "shared_latent_reader.py")},
            "source_reuse_assembler_sha256": hashlib.sha256(args.backbone_module_file.read_bytes()).hexdigest(),
            "source_reuse_projection_sha256": hashlib.sha256(args.backbone_module_file.with_name("shared_backbone_projection_reuse.py").read_bytes()).hexdigest()}
        parameter_path = args.parameters or args.output.with_suffix(".parameters.pkl")
        if args.parameters:
            with parameter_path.open("rb") as stream: saved = pickle.load(stream)
            if saved["identity"] != identity: raise ValueError("fresh parameter identity differs")
            host_parameters = saved["parameters"]
            parameters = jax.device_put(host_parameters)
        else:
            parameters = head.initialize(inputs, seed=seed)
            jax.block_until_ready(parameters)
            host_parameters = jax.device_get(parameters)
            owner.atomic_pickle(parameter_path, {"identity": identity, "parameters": host_parameters})
        leaves = jax.tree_util.tree_leaves(host_parameters)
        count = sum(np.size(x) for x in leaves)
        if count != 66123713 or any(np.asarray(x).dtype != np.dtype("float32") or not np.all(np.isfinite(x)) for x in leaves):
            raise ValueError("fresh canonical parameter tree/count/precision invalid")
        payload = bundle["payloads"][0]
        if args.component == "daily":
            load(overlay / "shared_daily_dual_ad.py", "scripts.hpc.shared_daily_dual_ad")
            from scripts.hpc.shared_daily_dual_ad import make_daily_dual_objective
            objective = make_shared_joint_daily_objective(context["runtime"], head, context["template"], context["state_scales"], context["science_scales"])
            objective = make_daily_dual_objective(jax, make_microbatch_objective(jax, jnp, objective))
            daily = jax.device_put(context["daily4"])
            arguments = (parameters, daily.inputs.continuous_state, daily)
            expected_state_shape = (4, 3854)
        else:
            horizon = int(args.component[1:])
            case = payload["cases"][horizon]
            if len(case) != 5 or np.shape(case.days[0].continuous_state)[0] != horizon:
                raise ValueError("recurrent case horizon differs")
            _, objective = make_shared_recurrent_objective(head, context["template"], context["state_scales"], context["science_scales"])
            arguments = (parameters, *jax.device_put(case))
            expected_state_shape = (3854,)
        jax.block_until_ready(arguments)
        original = original_arguments = accepted = None
        if args.original_full_reference:
            from scripts.hpc.run_shared_learning import load_learning_checkpoint
            admission_path = args.original_full_admission or args.original_full_reference.with_name(args.original_full_reference.name.replace(".reference.executable.pkl", "-reference.json"))
            accepted = json.loads(admission_path.read_text())
            valid = accepted.get("validation", accepted)
            full_state_evidence = (accepted.get("parameter_and_initial_state_gradients") is True
                or (accepted.get("scope") in (
                    "public_recurrent_loss_all_parameters_and_initial_state_AD",
                    "original_four_daily_micro4_public_loss_full_parameter_and_initial_state_AD")
                    and valid.get("initial_state_gradient_finite") is True
                    and valid.get("initial_state_gradient_coordinates") == int(np.prod(expected_state_shape))))
            horizon_matches = args.component == "daily" or accepted.get("horizon") == int(args.component[1:])
            if (accepted.get("status") != "passed" or not horizon_matches or not full_state_evidence
                    or accepted.get("identity", {}).get("source_head", accepted.get("identity", {}).get("source_git_head")) != reference["source_git_head"]
                    or accepted["identity"].get("method") != reference["method"]):
                raise ValueError("original full-state reference admission differs")
            original_state = load_learning_checkpoint(checkpoint_path, expected_method=reference["method"])
            if original_state["completed_updates"] != 0: raise ValueError("original baseline not update0")
            original_arguments = ((jax.device_put(original_state["parameters"]), daily.inputs.continuous_state, daily)
                if args.component == "daily" else (jax.device_put(original_state["parameters"]), *jax.device_put(case)))
            with gates.forbid_field_compile():
                with args.original_full_reference.open("rb") as stream: serialized = pickle.load(stream)
                original = deserialize_and_load(*serialized)
                validate_inputs(jax, serialized, original, original_arguments)
            original_memory = memory_sizes(original)
            if original_memory["estimated_live_bytes"] > 24*2**30:
                raise MemoryError("original full-state reference exceeds24GiB")
            del serialized
            emit("original_full_reference_prevalidated", original_reference_admission=str(admission_path),
                original_reference_identity=accepted["identity"], original_compiled_memory=original_memory,
                original_executable_abi_validated=True, original_compiles=0)
        emit("inputs_ready", identity=identity, runtime_preflight=preflight, parameter_count=count,
            parameter_leaves=len(leaves), parameter_artifact=str(parameter_path),
            decoder_dispatch="public_latent_reader_opt_in", candidate_output_chunk_size=head.shared_config.output_chunk_size,
            candidate_reader_execution=head.shared_config.latent_reader_execution,
            expert_native_map_chunk_available=latent.EXPERT_CHUNK_SIZE,
            fresh_candidate_loss_is_not_original_parity=True)
        if args.saved_executable:
            executable_path = args.saved_executable
            saved_identity = json.loads(executable_path.with_suffix(executable_path.suffix + ".identity.json").read_text())
            expected_identity = json.loads(json.dumps({"identity": identity, "component": args.component}, default=str))
            if saved_identity != expected_identity: raise ValueError("saved candidate executable identity differs")
            with gates.forbid_field_compile():
                with executable_path.open("rb") as stream: serialized = pickle.load(stream)
                executable = deserialize_and_load(*serialized)
                validate_inputs(jax, serialized, executable, arguments)
            compile_seconds = 0
            emit("candidate_executable_recovered", saved_executable=str(executable_path), candidate_compiles=0,
                compiler_options_scope="unused_new_compile_options; saved binary retained",
                recovered_compiler_options=prior.get("compiler_options") if args.diagnostic_continue_replay_failures else None)
        else:
            backbone = owner.load_backbone_assembler(args.backbone_module_file)
            original_assembler = shared_process_head.assemble_shared_transformer_context
            try:
                shared_process_head.assemble_shared_transformer_context = backbone
                backward = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True),
                    compiler_options=compiler_options)
                emit("lowering")
                with gates.deadline(args.lower_seconds, "lower", emit):
                    began = time.perf_counter(); lowered = backward.lower(*arguments)
                    lower_seconds = time.perf_counter()-began
                if args.save_lowered_ir:
                    began = time.perf_counter()
                    ir_path = args.output.with_suffix(".lowered.stablehlo.mlir.gz")
                    temporary_ir = ir_path.with_suffix(ir_path.suffix + ".tmp")
                    try:
                        with gzip.open(temporary_ir, "wt", encoding="utf-8", compresslevel=3) as stream:
                            stream.write(lowered.as_text(dialect="stablehlo"))
                        temporary_ir.replace(ir_path)
                    except Exception as error:
                        emit("lowered_ir_save_failed", lower_seconds=lower_seconds,
                            lowered_ir_save_seconds=time.perf_counter()-began,
                            lowered_ir_save_error={"type": type(error).__name__, "message": str(error)},
                            lowered_ir_scope="optional_compiler_diagnostic_not_an_admission_gate")
                    else:
                        emit("lowered_ir_saved", lower_seconds=lower_seconds,
                            lowered_ir_artifact=str(ir_path), lowered_ir_bytes=ir_path.stat().st_size,
                            lowered_ir_save_seconds=time.perf_counter()-began,
                            lowered_ir_scope="text_of_already_lowered_graph_diagnostic_only")
                    finally:
                        temporary_ir.unlink(missing_ok=True)
                emit("compiling", lower_seconds=lower_seconds)
                with gates.deadline(args.backend_seconds, "backend", emit):
                    began = time.perf_counter(); executable = lowered.compile()
                    compile_seconds = time.perf_counter()-began
            finally: shared_process_head.assemble_shared_transformer_context = original_assembler
            executable_path = args.output.with_suffix(".executable.pkl")
            owner.atomic_pickle(executable_path, serialize(executable))
            owner.atomic_json(executable_path.with_suffix(executable_path.suffix + ".identity.json"), {"identity": identity, "component": args.component})
        memory = memory_sizes(executable)
        emit("executable_saved", compile_seconds=compile_seconds, compiled_memory=memory, executable=str(executable_path))
        if memory["estimated_live_bytes"] > 24*2**30: raise MemoryError("candidate executable exceeds24GiB")
        with gates.deadline(180, "first_result", emit):
            actual = jax.device_get(jax.block_until_ready(executable(*arguments)))
        result_path = args.output.with_suffix(".result.pkl")
        owner.atomic_pickle(result_path, {"identity": identity, "result": actual})
        def validate(value):
            parameter_gate = gates.validate_result(jax, (value[0], value[1][0]), host_parameters, count=count, leaves=len(leaves))
            state = np.asarray(value[1][1])
            if (state.shape != expected_state_shape or state.dtype != np.asarray(arguments[1]).dtype
                    or not np.all(np.isfinite(state)) or not np.any(state != 0)):
                raise gates.NumericalFailure("full initial-state gradient invalid/zero")
            return parameter_gate | {"initial_state_shape": list(state.shape), "initial_state_nonzero": int(np.count_nonzero(state)), "initial_state_finite": True}
        validation = validate(actual)
        emit("result_saved", validation=validation, result_artifact=str(result_path),
            candidate_loss=float(actual[0][0]))
        reference_actual = actual
        diagnostic_records = []
        if args.diagnostic_continue_replay_failures:
            diagnostic = load(overlay / "shared_latent_replay_diagnostics.py", "shared_latent_replay_diagnostics")
            with args.diagnostic_reference_result.open("rb") as stream:
                saved_reference = pickle.load(stream)
            if saved_reference["identity"] != identity:
                raise ValueError("diagnostic saved result identity differs")
            reference_actual = saved_reference["result"]
            validate(reference_actual)
            base_arrays = context["runtime"]["base_statistics"].arrays
            scale_arrays = {name: np.asarray(value.scale) for name, value in base_arrays.items()
                if np.shape(value.scale) == (expected_state_shape[-1],)}
            def preserve_state_replay(label, host, reference_value, comparison, *,
                                      execution_architecture, execution_identity):
                execution = diagnostic.state_replay_execution_metadata(
                    execution_architecture=execution_architecture, execution_identity=execution_identity)
                summary = diagnostic.summarize_state_replay(host[1][1], reference_value[1][1],
                    input_state=np.asarray(arguments[1]), state_scales=scale_arrays,
                    loss_scales=context["science_scales"], max_failed_rows=int(np.prod(expected_state_shape)))
                evidence_path = args.output.with_suffix("." + label + ".state-replay.pkl")
                owner.atomic_pickle(evidence_path, {**execution, "label": label,
                    "actual_state_gradient": np.asarray(host[1][1]),
                    "reference_state_gradient": np.asarray(reference_value[1][1]),
                    "input_state": np.asarray(arguments[1]), "state_scale_arrays": scale_arrays,
                    "grouped_state_scales": context["state_scales"], "science_scales": context["science_scales"],
                    "loss_and_auxiliary": host[0], "reference_loss_and_auxiliary": reference_value[0],
                    "comparison": comparison, "summary": summary})
                row = {"label": label, "artifact": str(evidence_path), "comparison": comparison,
                    "initial_state": summary, **execution}
                diagnostic_records.append(row)
                if not all(check["passed"] for check in comparison.values()):
                    replay_failures.append({"scope": label, "artifact": str(evidence_path), "comparisons": comparison,
                        "execution_architecture": execution_architecture})
                    if not comparison["parameters"]["passed"]:
                        owner.atomic_pickle(args.output.with_suffix("." + label + ".full-result.pkl"),
                            {**execution, "result": host})
                emit("diagnostic_state_replay_saved", diagnostic_state_replays=diagnostic_records,
                    preserved_numerical_failures=replay_failures)
            initial_comparison = {"loss_and_auxiliary": owner.compare_trees(jax, actual[0], reference_actual[0]),
                "parameters": owner.compare_trees(jax, actual[1][0], reference_actual[1][0]),
                "initial_state": owner.compare_trees(jax, actual[1][1], reference_actual[1][1])}
            preserve_state_replay("recovered-first-vs-preserved-first", actual, reference_actual, initial_comparison,
                execution_architecture=report["architecture"], execution_identity=identity)
        timings = []; replays = []
        for repeat in range(3):
            with gates.deadline(180, "warm", emit), gates.forbid_field_compile():
                began = time.perf_counter(); result = executable(*arguments); jax.block_until_ready(result)
                timings.append(time.perf_counter()-began)
            host = jax.device_get(result); validate(host)
            comparison = {"loss_and_auxiliary": owner.compare_trees(jax, host[0], reference_actual[0]),
                "parameters": owner.compare_trees(jax, host[1][0], reference_actual[1][0]),
                "initial_state": owner.compare_trees(jax, host[1][1], reference_actual[1][1])}
            replays.append(comparison)
            emit("warm_result", synchronized_warm_seconds=timings, same_architecture_replays=replays)
            if args.diagnostic_continue_replay_failures:
                preserve_state_replay("candidate-warm-" + str(repeat), host, reference_actual, comparison,
                    execution_architecture=report["architecture"], execution_identity=identity)
            else:
                gates.require_parity(comparison)
        stats = jax.local_devices()[0].memory_stats() or {}
        if "peak_bytes_in_use" not in stats or stats["peak_bytes_in_use"] > 24*2**30:
            raise MemoryError("observed candidate peak unavailable/exceeds24GiB")
        if original is not None:
            with gates.forbid_field_compile():
                original_result = jax.device_get(jax.block_until_ready(original(*original_arguments)))
                original_parameters = original_state["parameters"]
                original_leaves = jax.tree_util.tree_leaves(original_parameters)
                original_validation = gates.validate_result(jax, (original_result[0], original_result[1][0]),
                    original_parameters, count=sum(np.size(x) for x in original_leaves), leaves=len(original_leaves))
                original_state_gradient = np.asarray(original_result[1][1])
                if (original_state_gradient.shape != expected_state_shape
                        or original_state_gradient.dtype != np.asarray(original_arguments[1]).dtype
                        or not np.all(np.isfinite(original_state_gradient)) or not np.any(original_state_gradient != 0)):
                    raise gates.NumericalFailure("original full-state replay invalid")
                original_result_path = args.output.with_suffix(".original-result.pkl")
                owner.atomic_pickle(original_result_path, {"identity": accepted["identity"], "result": original_result})
                emit("original_full_reference_replayed", original_validation=original_validation,
                    original_loss=float(original_result[0][0]),
                    original_initial_state_gradient_finite=True,
                    original_initial_state_gradient_nonzero=int(np.count_nonzero(original_state_gradient)),
                    original_result_artifact=str(original_result_path))
                original_times = []; candidate_times = []
                matched_validations = []; matched_replays = []
                for repeat in range(3):
                    for name in (("original", "candidate") if repeat % 2 == 0 else ("candidate", "original")):
                        began = time.perf_counter()
                        value = (original if name == "original" else executable)(*(original_arguments if name == "original" else arguments))
                        jax.block_until_ready(value)
                        (original_times if name == "original" else candidate_times).append(time.perf_counter()-began)
                        if args.diagnostic_continue_replay_failures:
                            host_value = jax.device_get(value)
                            if name == "candidate":
                                checked = validate(host_value)
                                replay_reference = reference_actual
                            else:
                                checked = gates.validate_result(jax, (host_value[0], host_value[1][0]),
                                    original_parameters, count=sum(np.size(x) for x in original_leaves), leaves=len(original_leaves))
                                state_value = np.asarray(host_value[1][1])
                                if (state_value.shape != expected_state_shape or state_value.dtype != original_state_gradient.dtype
                                        or not np.all(np.isfinite(state_value)) or not np.any(state_value != 0)):
                                    raise gates.NumericalFailure("original timed full-state result invalid")
                                replay_reference = original_result
                            checked.update(initial_state_shape=list(np.shape(host_value[1][1])),
                                initial_state_gradient_finite=True,
                                initial_state_gradient_nonzero=int(np.count_nonzero(host_value[1][1])),
                                loss=float(host_value[0][0]))
                            comparison = {"loss_and_auxiliary": owner.compare_trees(jax, host_value[0], replay_reference[0]),
                                "parameters": owner.compare_trees(jax, host_value[1][0], replay_reference[1][0]),
                                "initial_state": owner.compare_trees(jax, host_value[1][1], replay_reference[1][1])}
                            label = name + "-matched-" + str(repeat)
                            preserve_state_replay(label, host_value, replay_reference, comparison,
                                execution_architecture=report["architecture"] if name == "candidate" else
                                    (accepted.get("variant") or accepted["identity"]["method"]["daily_protocol"]["schema_version"]),
                                execution_identity=identity if name == "candidate" else accepted["identity"])
                            matched_validations.append({"label": label, "validation": checked})
                            matched_replays.append({"label": label, "comparison": comparison})
                emit("same_AD_workload_timed", original_full_state_reference=str(args.original_full_reference),
                    matched_warm_seconds={"original": original_times, "candidate": candidate_times},
                    original_over_candidate_median_ratio=float(np.median(original_times)/np.median(candidate_times)),
                    matched_original_over_candidate_ratios=[old/new for old, new in zip(original_times, candidate_times, strict=True)],
                    timing_scope=f"same_{args.component}_all_parameter_and_initial_state_AD_different_architectures_fixed_inputs",
                    cross_architecture_loss_parity_required=False)
                if args.diagnostic_continue_replay_failures:
                    emit("matched_diagnostic_results_checked", matched_result_validations=matched_validations,
                        matched_same_architecture_replays=matched_replays)
        final_stats = jax.local_devices()[0].memory_stats() or {}
        if "peak_bytes_in_use" not in final_stats or final_stats["peak_bytes_in_use"] > 24*2**30:
            raise MemoryError("observed final both-executable peak unavailable/exceeds24GiB")
        emit("completed", status="numerical_failed" if replay_failures else "passed",
            diagnostic_complete=bool(args.diagnostic_continue_replay_failures),
            preserved_numerical_failures=replay_failures, warm_median_seconds=float(np.median(timings)), candidate_result_phase_device_memory=stats,
            device_memory_after_both_executables=final_stats,
            peak_memory_semantics="process cumulative; original executable preloaded for ABI before candidate compile",
            complete_update_admitted=False, scientific_accuracy_admitted=False)
        if replay_failures:
            raise gates.NumericalFailure("preserved strict self-replay failure blocks numerical admission after bounded diagnostics")
    except BaseException as error:
        emit("exception", status="resource_failed" if isinstance(error, MemoryError) else "numerical_failed" if isinstance(error, gates.NumericalFailure) else "error", error=str(error), error_type=type(error).__name__)
        raise

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--component", choices=("daily", "T7", "T30"), required=True)
    for name in ("output", "patch-owner", "gate-owner", "forward-probe", "source-profile", "input-bundle", "prepared-context", "daily-admission-report", "seven-admission-report", "thirty-admission-report", "support-owner-file", "runtime-preflight", "backbone-module-file"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--parameters", type=Path)
    p.add_argument("--original-full-reference", type=Path)
    p.add_argument("--original-full-admission", type=Path)
    p.add_argument("--saved-executable", type=Path)
    p.add_argument("--diagnostic-continue-replay-failures", action="store_true")
    p.add_argument("--diagnostic-reference-result", type=Path)
    p.add_argument("--diagnostic-prior-failure-report", type=Path)
    p.add_argument("--save-lowered-ir", action="store_true",
        help="Save compressed text from the already-lowered graph for bounded compiler diagnosis")
    p.add_argument("--force-compilation-parallelism", type=int, choices=(0, 4), default=0,
        help="0 retains default compiler partitioning; 4 requests compiler parallelism 4")
    p.add_argument("--lower-seconds", type=int, default=480)
    p.add_argument("--backend-seconds", type=int, default=3400)
    args = p.parse_args()
    if args.saved_executable and not args.parameters:
        p.error("saved executable recovery requires its saved fresh parameters")
    if args.saved_executable and args.force_compilation_parallelism:
        p.error("saved executable recovery cannot override compiler parallelism")
    if args.diagnostic_continue_replay_failures and not (args.saved_executable
            and args.diagnostic_reference_result and args.diagnostic_prior_failure_report):
        p.error("replay diagnostics require a saved executable, preserved result and prior numerical-failure report")
    run(args)

if __name__ == "__main__": main()
