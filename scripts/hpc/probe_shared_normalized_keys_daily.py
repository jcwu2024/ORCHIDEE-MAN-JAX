"""Original-four public daily loss/all64,884,481 parameter execution gate.

The treatment is original FP32 field expert128 + source-reuse backbone +
normalized key banks by default, or an explicit scoped execution adapter.
Reuses admission/validation utilities; no union,
mixed precision, new scientific loss, state-gradient certification or updates.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import serialize, deserialize_and_load
    jax.config.update("jax_enable_x64", True)
    owner = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    gates = load(args.gate_owner, "normalized_daily_admission_utilities")
    report = {"status": "running", "scope": "original_four_daily_micro4_parameter_AD_only",
              "variant": "field_expert128+backbone_source_reuse+normalized_key_banks",
              "precision": "unchanged_FP32_neural_FP64_physical", "gpu_count": 1,
              "rtol": 1e-5, "atol": 1e-6, "training_updates": 0, "state_gradient_parity": "not_established",
              "prior_synthetic_component_gate": "failed_parameter_VJP_preserved_separately",
              "budgets_seconds": {"process": args.process_seconds, "lower": args.lower_seconds, "backend": args.backend_seconds}}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        owner.atomic_json(args.output, report)
        print(stage, flush=True)
    outputs = [args.output, args.output.with_suffix(".reference.pkl"), args.output.with_suffix(".candidate.pkl"), args.output.with_suffix(".candidate.executable.pkl")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("preserve prior diagnostic artifacts; choose new output stem")
    try:
        emit("starting")
        with gates.deadline(max(1, args.process_seconds - (time.perf_counter() - started)), "whole_process", emit):
            from research.daily_coarse_graining.daily_operator import shared_process_head
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
            from scripts.hpc.shared_daily_microbatch import make_microbatch_objective
            from scripts.hpc.run_shared_learning import load_learning_checkpoint
            from scripts.hpc.run_shared_transformer_admission import memory_sizes
            execution_file = args.execution_module_file or Path(__file__).with_name("shared_normalized_key_execution.py")
            normalized = load(execution_file, "daily_execution_diagnostic_adapter")
            if args.execution_module_file:
                report.update(variant="field_expert128+backbone_source_reuse+expert5_preserve128_learned_shapes",
                    execution_adapter=str(execution_file),
                    prior_synthetic_component_gate="shape_preserved_output_pass_parameter_and_raw_input_fail_not_waived")
            if len(devices) != 1 or jax.default_backend() != "gpu":
                raise ValueError("requires single actualGPU")
            preflight = gates.validate_runtime_report(args.runtime_preflight, jax.__version__)
            profile, reference = owner.trusted_profile(args)
            root = Path(os.environ["ORCHIDEE_REPO_ROOT"])
            if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != reference["source_git_head"]:
                raise ValueError("requires immutable source checkout")
            identity = owner.context_identity(reference, owner.initial_checkpoint(reference), 742)
            canonical_binary = Path(profile["executables"]["daily"]["executable"]).resolve()
            legacy_identity = identity | {"jax_version": jax.__version__, "executable": str(canonical_binary)}
            with args.prepared_context.open("rb") as stream:
                context = pickle.load(stream)
            if context["identity"] != identity or Path(context["daily_executable"]).resolve() != canonical_binary:
                raise ValueError("prepared context identity drift")
            with args.input_bundle.open("rb") as stream:
                bundle = pickle.load(stream)
            if (bundle["schema_version"] != "shared_serialized_dual_inputs_v1" or bundle["source_git_head"] != identity["source_git_head"]
                    or bundle["method"] != reference["method"] or Path(bundle["initial_checkpoint"]).resolve() != Path(identity["initial_checkpoint"])
                    or tuple(bundle["artifact_ids"]) != tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])):
                raise ValueError("original saved input/method identity drift")
            rows = gates.validate_original_rows(jax, context["daily4"], bundle["payloads"][0]["daily"])
            del bundle
            admission = json.loads(args.baseline_admission.read_text())
            binary_path, result_path = gates.validate_admission(admission, legacy_identity)
            baseline_saved, provenance = gates.load_reference(result_path, legacy_identity, variant=gates.VARIANT)
            checkpoint = load_learning_checkpoint(Path(identity["initial_checkpoint"]), expected_method=reference["method"])
            if checkpoint["completed_updates"] != 0:
                raise ValueError("requires update0")
            host_parameters = checkpoint["parameters"]
            parameter_leaves = jax.tree_util.tree_leaves(host_parameters)
            if len(parameter_leaves) != 300 or sum(np.size(x) for x in parameter_leaves) != 64_884_481 or any(np.asarray(x).dtype != np.float32 for x in parameter_leaves):
                raise ValueError("parameter architecture/precision drift")
            parameters, daily4 = jax.device_put((host_parameters, context["daily4"]))
            jax.block_until_ready((parameters, daily4))
            identity = legacy_identity | {"comparison_schema": "shared_chunk_batch_original_micro4_v1" if args.execution_module_file else "shared_normalized_keys_original_micro4_v1", "original_rows": rows, "baseline_binary": str(binary_path)}
            emit("inputs_ready", identity=identity, runtime_preflight=preflight, baseline_reference=provenance)
            with gates.forbid_field_compile():
                with binary_path.open("rb") as stream:
                    serialized = pickle.load(stream)
                field = deserialize_and_load(*serialized)
                gates.validate_binary_inputs(jax, serialized, field, parameters, daily4)
                del serialized
                memory = memory_sizes(field)
                if memory["estimated_live_bytes"] > 24 * 2**30:
                    raise MemoryError("field executable exceeds24GiB")
                with gates.deadline(100, "field_first_result", emit):
                    value, seconds = gates.synchronized_call(jax, field, parameters, daily4)
                    field_result = jax.device_get(value)
                    del value
                owner.atomic_pickle(outputs[1], {"identity": identity, "variant": gates.VARIANT, "result": field_result})
                field_valid = gates.validate_result(jax, field_result, host_parameters)
                field_parity = gates.comparisons(owner, jax, field_result, baseline_saved)
                emit("field_reference_saved", field_admission=field_valid, saved_reference_comparisons=field_parity,
                     field_memory=memory, field_first_seconds=seconds, baseline_compiles=0)
                gates.require_parity(field_parity)
            del baseline_saved
            adapter = owner.load_staged_adapter(args.candidate_module_file)
            backbone = owner.load_backbone_assembler(args.backbone_module_file)
            objective = make_shared_joint_daily_objective(context["runtime"], context["head"], context["template"], context["state_scales"], context["science_scales"])
            objective = make_microbatch_objective(jax, jnp, objective)
            backward = jax.jit(jax.value_and_grad(objective, has_aux=True))
            with owner.candidate_process_head_patches(shared_process_head, adapter, backbone), normalized.patches(context["head"].shared_config):
                emit("candidate_lowering")
                with gates.deadline(args.lower_seconds, "candidate_lower", emit):
                    began = time.perf_counter()
                    lowered = backward.lower(parameters, daily4)
                    lower_seconds = time.perf_counter() - began
                emit("candidate_compiling", lower_seconds=lower_seconds)
                with gates.deadline(args.backend_seconds, "candidate_backend", emit):
                    began = time.perf_counter()
                    candidate = lowered.compile()
                    compile_seconds = time.perf_counter() - began
            owner.atomic_pickle(outputs[3], serialize(candidate))
            if args.execution_module_file:
                emit("adapter_scope_recorded", changed_chunk_routes=getattr(normalized,"PATCHED_ROWS",None))
            memory = memory_sizes(candidate)
            emit("candidate_executable_saved", compile_seconds=compile_seconds, candidate_memory=memory, candidate_executable=str(outputs[3]))
            if memory["estimated_live_bytes"] > 24 * 2**30:
                raise MemoryError("candidate executable exceeds24GiB")
            with gates.deadline(100, "candidate_first_result", emit):
                value, seconds = gates.synchronized_call(jax, candidate, parameters, daily4)
                actual = jax.device_get(value)
                del value
            owner.atomic_pickle(outputs[2], {"identity": identity, "variant": report["variant"], "result": actual})
            valid = gates.validate_result(jax, actual, host_parameters)
            parity = gates.comparisons(owner, jax, actual, field_result)
            memory_stats = jax.local_devices()[0].memory_stats() or {}
            emit("candidate_result_saved", candidate_artifact=str(outputs[2]), comparisons=parity,
                 candidate_admission=valid, candidate_physical_and_finite_validation=valid,
                 device_memory=memory_stats, candidate_first_seconds=seconds)
            if "peak_bytes_in_use" not in memory_stats or memory_stats["peak_bytes_in_use"] > 24 * 2**30:
                raise MemoryError("observed peak unavailable or exceeds24GiB")
            numerical_passed = True
            try:
                gates.require_parity(parity)
            except gates.NumericalFailure:
                if not args.execution_module_file:
                    raise
                numerical_passed = False
                emit("strict_numerical_failure_preserved", strict_numerical_passed=False,
                     timing_continues_as_diagnostic_only=True, candidate_admission=False)
            del actual, field_result
            candidate_label = "candidate" if args.execution_module_file else "normalized"
            times = {"field": [], candidate_label: []}
            for name, executable in (("field", field), (candidate_label, candidate)):
                jax.block_until_ready(executable(parameters, daily4))
            for repeat in range(7):
                order = ("field", candidate_label) if repeat % 2 == 0 else (candidate_label, "field")
                for name in order:
                    with gates.deadline(100, "warm_" + name, emit), gates.forbid_field_compile():
                        value, seconds = gates.synchronized_call(jax, field if name == "field" else candidate, parameters, daily4)
                        del value
                    times[name].append(seconds)
            ratio = float(np.median(times["field"]) / np.median(times[candidate_label]))
            emit("completed", status="passed" if numerical_passed else "numerical_failed", matched_warm_seconds=times,
                 strict_numerical_passed=numerical_passed,
                 **{"field_over_candidate_median_ratio" if args.execution_module_file else "field_over_normalized_median_ratio":ratio},
                 device_memory=jax.local_devices()[0].memory_stats(), timing_scope="daily_micro4_public_loss_and_full_parameter_AD_only")
    except BaseException as error:
        emit("exception", status="resource_failed" if isinstance(error, MemoryError) else "numerical_failed" if isinstance(error, gates.NumericalFailure) else "error", error=str(error), error_type=type(error).__name__)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "patch-owner", "gate-owner", "source-profile", "input-bundle", "prepared-context", "baseline-admission", "runtime-preflight", "candidate-module-file", "backbone-module-file"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--lower-seconds", type=int, default=240)
    p.add_argument("--backend-seconds", type=int, default=1200)
    p.add_argument("--process-seconds", type=int, default=1800)
    p.add_argument("--execution-module-file", type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
