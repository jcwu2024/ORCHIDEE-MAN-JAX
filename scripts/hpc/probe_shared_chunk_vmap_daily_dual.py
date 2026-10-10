"""Compare original-field and whole-chunk-vmap daily loss plus dual AD."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _three_outputs(helper, jax, result, reference):
    return {
        "loss": helper.compare_trees(jax, result[0][0], reference[0][0]),
        "auxiliary": helper.compare_trees(jax, result[0][1], reference[0][1]),
        "all_parameter_gradients": helper.compare_trees(jax, result[1][0], reference[1][0]),
    }


def _all_outputs(helper, jax, result, reference):
    return _three_outputs(helper, jax, result, reference) | {
        "all_initial_state_gradients": helper.compare_trees(
            jax, result[1][1], reference[1][1]
        )
    }


def _validate_dual(gates, jax, result, parameters, states):
    parameter_admission = gates.validate_result(jax, (result[0], result[1][0]), parameters)
    gradient = np.asarray(result[1][1])
    if gradient.shape != (4, 3854) or not np.all(np.isfinite(gradient)):
        raise gates.NumericalFailure("daily initial-state gradient is nonfinite or has wrong shape")
    nonzero = int(np.count_nonzero(gradient))
    if nonzero == 0:
        raise gates.NumericalFailure("daily initial-state gradient is all zero")
    if tuple(np.shape(states)) != (4, 3854):
        raise ValueError("four initial states changed shape")
    hard = parameter_admission["hard_counts"]
    if len(hard) != 11 or any(hard.values()):
        raise gates.NumericalFailure(f"expected eleven zero physical hard counts, got {hard}")
    return parameter_admission | {
        "initial_state_gradient_shape": list(gradient.shape),
        "initial_state_gradient_coordinates": int(gradient.size),
        "initial_state_gradient_nonzero": nonzero,
        "initial_state_gradient_finite": True,
        "hard_count_total": len(hard),
    }


def _validate_binary_inputs(jax, serialized, executable, arguments):
    if not isinstance(serialized, tuple) or len(serialized) != 3:
        raise ValueError("saved dual executable is not a JAX serialized triple")
    supplied = (arguments, {})
    if serialized[1] != jax.tree_util.tree_structure(supplied):
        raise ValueError("dual executable input treedef differs")
    infos, info_tree = jax.tree_util.tree_flatten(executable.args_info)
    values, supplied_tree = jax.tree_util.tree_flatten(supplied)
    if info_tree != supplied_tree or len(infos) != len(values):
        raise ValueError("loaded dual executable argument tree differs")
    for info, value in zip(infos, values, strict=True):
        if tuple(info._aval.shape) != np.shape(value) or np.dtype(info._aval.dtype) != np.asarray(value).dtype:
            raise ValueError("dual executable input shape/dtype differs")


def _tree_abi(jax, value):
    leaves, treedef = jax.tree_util.tree_flatten(value)
    return {
        "treedef": str(treedef),
        "leaf_count": len(leaves),
        "leaves": [
            {"shape": list(np.shape(leaf)), "dtype": str(np.asarray(leaf).dtype)}
            for leaf in leaves
        ],
    }


def run(args):
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

    jax, devices = initialize_gpu_backend()
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import serialize, deserialize_and_load

    jax.config.update("jax_enable_x64", True)
    owner = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    gates = load(args.gate_owner, "daily_dual_admission_utilities")
    adapter = load(args.execution_module_file, "daily_v4_whole_chunk_adapter")
    dual = load(Path(__file__).with_name("shared_daily_dual_ad.py"), "shared_daily_dual_ad")
    report = {
        "status": "running",
        "stage": "starting",
        "mode": args.stage,
        "scope": "original_four_daily_micro4_public_loss_full_parameter_and_initial_state_AD",
        "variant": "field_original" if args.stage == "reference" else "field_expert128+backbone_source_reuse+expert5_whole_chunk_axis_vmap",
        "dual_input_abi": "(parameters, initial_states[4,3854], daily_samples); dynamic batch with inputs.continuous_state replaced",
        "state_is_optimizer_parameter": False,
        "precision": "unchanged_FP32_neural_FP64_physical",
        "rtol": 1e-5,
        "atol": 1e-6,
        "training_updates": 0,
        "optimizer": False,
        "T7_T30": False,
        "production_adoption": False,
        "scientific_acceptance": False,
        "budgets_seconds": {"lower": args.lower_seconds, "backend": args.backend_seconds, "process": args.process_seconds},
        "source_artifacts": {
            "dual_helper": {"path": str(Path(__file__).with_name("shared_daily_dual_ad.py")),
                            "sha256": hashlib.sha256(Path(__file__).with_name("shared_daily_dual_ad.py").read_bytes()).hexdigest()},
            "dual_probe": {"path": str(Path(__file__).resolve()),
                           "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "field_expert_adapter": {"path": str(args.candidate_module_file),
                                     "sha256": hashlib.sha256(args.candidate_module_file.read_bytes()).hexdigest()},
            "backbone_adapter": {"path": str(args.backbone_module_file),
                                 "sha256": hashlib.sha256(args.backbone_module_file.read_bytes()).hexdigest()},
            "v4_execution_adapter": {"path": str(args.execution_module_file),
                                     "sha256": hashlib.sha256(args.execution_module_file.read_bytes()).hexdigest()},
        },
        "reference_execution": "field_expert128+backbone_source_reuse",
        "candidate_execution": "field_expert128+backbone_source_reuse+expert5_whole_chunk_axis_vmap",
    }

    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        owner.atomic_json(args.output, report)
        print(f"daily_dual stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    outputs = [args.output, args.result_output, args.executable_output]
    if args.stage == "candidate":
        outputs.extend((args.output.with_suffix(".field-replay.pkl"), args.output.with_suffix(".state-replays.pkl")))
    if any(path.exists() for path in outputs):
        raise FileExistsError("preserve previous artifacts; choose a new unique output stem")
    try:
        emit("starting")
        with gates.deadline(max(1, args.process_seconds - (time.perf_counter() - started)), "whole_process", emit):
            from research.daily_coarse_graining.daily_operator import shared_process_head
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
            from scripts.hpc.shared_daily_microbatch import make_microbatch_objective
            from scripts.hpc.run_shared_learning import load_learning_checkpoint
            from scripts.hpc.run_shared_transformer_admission import memory_sizes

            if len(devices) != 1 or jax.default_backend() != "gpu":
                raise ValueError("requires one actual GPU")
            preflight = gates.validate_runtime_report(args.runtime_preflight, jax.__version__)
            profile, reference = owner.trusted_profile(args)
            root = Path(os.environ["ORCHIDEE_REPO_ROOT"])
            if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != reference["source_git_head"]:
                raise ValueError("requires immutable source checkout")
            context_identity = owner.context_identity(reference, owner.initial_checkpoint(reference), 742)
            canonical_binary = Path(profile["executables"]["daily"]["executable"]).resolve()
            parameter_identity = context_identity | {"jax_version": jax.__version__, "executable": str(canonical_binary)}
            with args.prepared_context.open("rb") as stream:
                context = pickle.load(stream)
            if context.get("identity") != context_identity or Path(context["daily_executable"]).resolve() != canonical_binary:
                raise ValueError("prepared context identity drift")
            with args.input_bundle.open("rb") as stream:
                bundle = pickle.load(stream)
            if (
                bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
                or bundle.get("source_git_head") != context_identity["source_git_head"]
                or bundle.get("method") != reference["method"]
                or Path(bundle["initial_checkpoint"]).resolve() != Path(context_identity["initial_checkpoint"])
                or tuple(bundle["artifact_ids"]) != tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
            ):
                raise ValueError("original saved input/method identity drift")
            original_daily = bundle["payloads"][0]["daily"]
            rows = gates.validate_original_rows(jax, context["daily4"], original_daily)
            del bundle
            daily4 = context["daily4"]
            states = daily4.inputs.continuous_state
            if tuple(states.shape) != (4, 3854):
                raise ValueError(f"expected daily4 input state shape (4,3854), got {tuple(states.shape)}")
            checkpoint = load_learning_checkpoint(Path(context_identity["initial_checkpoint"]), expected_method=reference["method"])
            if checkpoint["completed_updates"] != 0:
                raise ValueError("requires update-0 parameters")
            host_parameters = checkpoint["parameters"]
            if sum(np.size(value) for value in jax.tree_util.tree_leaves(host_parameters)) != 64_884_481:
                raise ValueError("parameter architecture drift")
            parameters, daily4, states = jax.device_put((host_parameters, daily4, states))
            jax.block_until_ready((parameters, daily4, states))
            identity = parameter_identity | {
                "comparison_schema": "shared_chunk_vmap_daily_dual_micro4_v1",
                "original_rows": rows,
                "state_gradient_shape": [4, 3854],
                "state_gradient_coordinates": 15_416,
                "v4_parameter_report": str(args.parameter_report),
            }
            emit("inputs_ready", identity=identity, runtime_preflight=preflight, original_rows=rows,
                 daily_state_shape=list(states.shape), daily_state_coordinates=int(states.size),
                 parameter_only_baseline_binary=str(args.baseline_admission))

            daily_objective = make_shared_joint_daily_objective(
                context["runtime"], context["head"], context["template"], context["state_scales"], context["science_scales"]
            )
            daily_objective = make_microbatch_objective(jax, jnp, daily_objective)
            objective = dual.make_daily_dual_objective(jax, daily_objective)
            initial_states = states
            backward = dual.make_daily_dual_value_and_grad(jax, objective)
            arguments = (parameters, initial_states, daily4)

            if args.stage == "reference":
                admission = json.loads(args.baseline_admission.read_text())
                field_binary, _ = gates.validate_admission(admission, parameter_identity)
                with gates.forbid_field_compile():
                    with field_binary.open("rb") as stream:
                        serialized_field = pickle.load(stream)
                    field = deserialize_and_load(*serialized_field)
                    gates.validate_binary_inputs(jax, serialized_field, field, parameters, daily4)
                    del serialized_field
                    field_memory = memory_sizes(field)
                    if field_memory["estimated_live_bytes"] > 24 * 2**30:
                        raise MemoryError("saved field parameter-only binary exceeds 24 GiB")
                    field_value = jax.device_get(jax.block_until_ready(field(parameters, daily4)))
                saved_parameter_report = json.loads(args.parameter_report.read_text())
                with args.parameter_result.open("rb") as stream:
                    saved_parameter = pickle.load(stream)
                if (
                    saved_parameter_report.get("status") != "passed"
                    or saved_parameter_report.get("strict_numerical_passed") is not True
                    or saved_parameter_report.get("identity") != saved_parameter.get("identity")
                    or saved_parameter_report.get("variant") != "field_expert128+backbone_source_reuse+expert5_whole_chunk_axis_vmap"
                    or saved_parameter_report.get("identity", {}).get("source_git_head") != parameter_identity["source_git_head"]
                    or saved_parameter_report.get("identity", {}).get("method") != parameter_identity["method"]
                    or saved_parameter_report.get("identity", {}).get("initial_checkpoint") != parameter_identity["initial_checkpoint"]
                    or saved_parameter_report.get("identity", {}).get("original_rows") != rows
                    or saved_parameter_report.get("execution_adapter_sha256") != hashlib.sha256(args.execution_module_file.read_bytes()).hexdigest()
                    or saved_parameter.get("identity", {}).get("original_rows") != rows
                ):
                    raise ValueError("saved V4 parameter-only reference identity/admission drift")
                parameter_bridge = owner.compare_trees(jax, field_value, saved_parameter["result"])
                if not parameter_bridge["passed"]:
                    raise gates.NumericalFailure("saved parameter-only field binary/result bridge failed")
                emit("field_parameter_bridge_passed", saved_parameter_reference=str(args.parameter_result),
                     parameter_bridge=parameter_bridge, field_memory=field_memory)
                emit("reference_lowering")
                original_adapter = owner.load_staged_adapter(args.candidate_module_file)
                original_backbone = owner.load_backbone_assembler(args.backbone_module_file)
                with owner.candidate_process_head_patches(shared_process_head, original_adapter, original_backbone):
                    with gates.deadline(args.lower_seconds, "reference_lower", emit):
                        began = time.perf_counter()
                        lowered = backward.lower(*arguments)
                        lower_seconds = time.perf_counter() - began
                    emit("reference_compiling", lower_seconds=lower_seconds)
                    with gates.deadline(args.backend_seconds, "reference_backend", emit):
                        began = time.perf_counter()
                        executable = lowered.compile()
                        compile_seconds = time.perf_counter() - began
                serialized_executable = serialize(executable)
                _validate_binary_inputs(jax, serialized_executable, executable, arguments)
                owner.atomic_pickle(args.executable_output, serialized_executable)
                memory = memory_sizes(executable)
                if memory["estimated_live_bytes"] > 24 * 2**30:
                    raise MemoryError("field dual executable exceeds 24 GiB")
                with gates.deadline(120, "reference_first_result", emit):
                    value = jax.device_get(jax.block_until_ready(executable(*arguments)))
                validation = _validate_dual(gates, jax, value, host_parameters, initial_states)
                projection = (value[0], value[1][0])
                bridge = owner.compare_trees(jax, projection, field_value)
                saved_result_bridge = owner.compare_trees(jax, projection, saved_parameter["result"])
                if not bridge["passed"] or not saved_result_bridge["passed"]:
                    raise gates.NumericalFailure("field dual result failed parameter-only loss/aux/gradient bridge")
                binary_abi = {"inputs": _tree_abi(jax, arguments), "outputs": _tree_abi(jax, value)}
                owner.atomic_pickle(args.result_output, {"identity": identity, "result": value, "binary_abi": binary_abi})
                stats = jax.local_devices()[0].memory_stats() or {}
                if "peak_bytes_in_use" not in stats or stats["peak_bytes_in_use"] > 24 * 2**30:
                    raise MemoryError("observed field dual peak unavailable or exceeds 24 GiB")
                emit("completed", status="passed", validation=validation, executable=str(args.executable_output),
                     result=str(args.result_output), estimated_memory=memory, device_memory=stats,
                     bridge_to_parameter_only_binary=bridge, bridge_to_saved_parameter_result=saved_result_bridge,
                     binary_abi=binary_abi,
                     compile_seconds=compile_seconds, production_adoption=False)
                return report

            reference_report = json.loads(args.dual_reference_report.read_text())
            with args.dual_reference_result.open("rb") as stream:
                saved_reference = pickle.load(stream)
            if (
                reference_report.get("status") != "passed"
                or saved_reference.get("identity") != reference_report.get("identity")
                or saved_reference.get("identity", {}).get("source_git_head") != parameter_identity["source_git_head"]
                or saved_reference.get("identity", {}).get("original_rows") != rows
                or saved_reference.get("identity", {}).get("state_gradient_shape") != [4, 3854]
            ):
                raise ValueError("canonical field dual reference identity/admission drift")
            saved_reference_result = saved_reference["result"]
            _validate_dual(gates, jax, saved_reference_result, host_parameters, initial_states)
            with gates.forbid_field_compile():
                with args.dual_reference_executable.open("rb") as stream:
                    serialized_reference = pickle.load(stream)
                field_dual = deserialize_and_load(*serialized_reference)
                _validate_binary_inputs(jax, serialized_reference, field_dual, arguments)
                del serialized_reference
                reference_memory = memory_sizes(field_dual)
                if reference_memory["estimated_live_bytes"] > 24 * 2**30:
                    raise MemoryError("field dual reference exceeds 24 GiB")
                replay = jax.device_get(jax.block_until_ready(field_dual(*arguments)))
            reference_replay = _three_outputs(owner, jax, replay, saved_reference_result)
            reference_state_drift = owner.compare_trees(jax, replay[1][1], saved_reference_result[1][1])
            if any(not row["passed"] for row in reference_replay.values()):
                raise gates.NumericalFailure("saved dual reference loss/auxiliary/parameter replay failed")
            owner.atomic_pickle(args.output.with_suffix(".field-replay.pkl"), {
                "identity": identity,
                "result": replay,
                "comparison_to_saved_reference": reference_replay,
                "state_self_drift": reference_state_drift,
            })
            reference_result = replay

            candidate_adapter = owner.load_staged_adapter(args.candidate_module_file)
            backbone = owner.load_backbone_assembler(args.backbone_module_file)
            patch_context = owner.candidate_process_head_patches(shared_process_head, candidate_adapter, backbone)
            emit("candidate_lowering")
            with patch_context, adapter.patches(context["head"].shared_config):
                with gates.deadline(args.lower_seconds, "candidate_lower", emit):
                    began = time.perf_counter()
                    lowered = backward.lower(*arguments)
                    lower_seconds = time.perf_counter() - began
                emit("candidate_compiling", lower_seconds=lower_seconds)
                with gates.deadline(args.backend_seconds, "candidate_backend", emit):
                    began = time.perf_counter()
                    candidate = lowered.compile()
                    compile_seconds = time.perf_counter() - began
            serialized_candidate = serialize(candidate)
            _validate_binary_inputs(jax, serialized_candidate, candidate, arguments)
            owner.atomic_pickle(args.executable_output, serialized_candidate)
            memory = memory_sizes(candidate)
            if memory["estimated_live_bytes"] > 24 * 2**30:
                raise MemoryError("V4 daily dual executable exceeds 24 GiB")
            with gates.deadline(120, "candidate_first_result", emit):
                first = jax.device_get(jax.block_until_ready(candidate(*arguments)))
            binary_abi = {"inputs": _tree_abi(jax, arguments), "outputs": _tree_abi(jax, first)}
            owner.atomic_pickle(args.result_output, {"identity": identity, "result": first, "binary_abi": binary_abi})
            validation = _validate_dual(gates, jax, first, host_parameters, initial_states)
            initial_comparisons = _all_outputs(owner, jax, first, reference_result)
            stats = jax.local_devices()[0].memory_stats() or {}
            if "peak_bytes_in_use" not in stats or stats["peak_bytes_in_use"] > 24 * 2**30:
                raise MemoryError("observed candidate peak unavailable or exceeds 24 GiB")
            initial_passed = all(row["passed"] for row in initial_comparisons.values())
            emit("strict_initial_comparison_passed" if initial_passed else "strict_initial_comparison_failed",
                 comparisons=initial_comparisons, initial_strict_passed=initial_passed,
                 validation=validation, candidate_memory=memory, device_memory=stats,
                 saved_field_state_self_drift=reference_state_drift,
                 binary_abi=binary_abi, execution_routes=getattr(adapter, "PATCHED_ROWS", []),
                 seven_pair_diagnostics_will_run=True)

            times = {"field": [], "candidate": []}
            state_self_drift = {"field": [], "candidate": []}
            pairs = []
            state_replays = {"field": [], "candidate": []}
            jax.block_until_ready((field_dual(*arguments), candidate(*arguments)))
            for repeat in range(7):
                order = ("field", "candidate") if repeat % 2 == 0 else ("candidate", "field")
                results = {}
                for name in order:
                    executable = field_dual if name == "field" else candidate
                    with gates.deadline(120, f"{name}_warm_{repeat}", emit), gates.forbid_field_compile():
                        began = time.perf_counter()
                        value = executable(*arguments)
                        jax.block_until_ready(value)
                        times[name].append(time.perf_counter() - began)
                    results[name] = jax.device_get(value)
                    state_self_drift[name].append(
                        owner.compare_trees(jax, results[name][1][1], replay[1][1] if name == "field" else first[1][1])
                    )
                    state_replays[name].append(np.asarray(results[name][1][1]).copy())
                    del value
                pair = _all_outputs(owner, jax, results["candidate"], results["field"])
                pairs.append(pair)
                if any(not row["passed"] for row in pair.values()):
                    owner.atomic_pickle(args.output.with_suffix(f".failed-pair-{repeat}.pkl"), {
                        "identity": identity,
                        "field_result": results["field"],
                        "candidate_result": results["candidate"],
                        "comparisons": pair,
                    })
                del results
            passed = initial_passed and all(all(row["passed"] for row in pair.values()) for pair in pairs)
            owner.atomic_pickle(args.output.with_suffix(".state-replays.pkl"), {
                "identity": identity,
                "state_gradient_shape": [4, 3854],
                "field": state_replays["field"],
                "candidate": state_replays["candidate"],
            })
            emit("completed", status="passed" if passed else "parity_failed", validation=validation,
                 comparisons=initial_comparisons, full_derivative_pair_comparisons=pairs,
                 field_state_self_drift=state_self_drift["field"], candidate_state_self_drift=state_self_drift["candidate"],
                 saved_field_state_self_drift=reference_state_drift,
                 self_drift_is_diagnostic=True, matched_warm_seconds=times,
                 field_over_candidate_median_ratio=float(np.median(times["field"]) / np.median(times["candidate"])),
                 candidate_memory=memory, reference_memory=reference_memory, device_memory=jax.local_devices()[0].memory_stats(),
                 strict_rtol=1e-5, strict_atol=1e-6, all_parameter_and_state_coordinates_compared=True,
                 state_replays=str(args.output.with_suffix(".state-replays.pkl")),
                 binary_abi=binary_abi,
                 optimizer_state_gradient_use=False, execution_routes=getattr(adapter, "PATCHED_ROWS", []),
                 compile_seconds=compile_seconds, production_adoption=False)
            return report
    except BaseException as error:
        emit("exception", status="resource_failed" if isinstance(error, MemoryError) else "numerical_failed" if isinstance(error, gates.NumericalFailure) else "error",
             error=str(error), error_type=type(error).__name__)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("reference", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--executable-output", type=Path, required=True)
    parser.add_argument("--patch-owner", type=Path, required=True)
    parser.add_argument("--gate-owner", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--prepared-context", type=Path, required=True)
    parser.add_argument("--baseline-admission", type=Path, required=True)
    parser.add_argument("--runtime-preflight", type=Path, required=True)
    parser.add_argument("--candidate-module-file", type=Path, required=True)
    parser.add_argument("--backbone-module-file", type=Path, required=True)
    parser.add_argument("--execution-module-file", type=Path, required=True)
    parser.add_argument("--parameter-report", type=Path, required=True)
    parser.add_argument("--parameter-result", type=Path, required=True)
    parser.add_argument("--dual-reference-report", type=Path)
    parser.add_argument("--dual-reference-result", type=Path)
    parser.add_argument("--dual-reference-executable", type=Path)
    parser.add_argument("--lower-seconds", type=int, default=240)
    parser.add_argument("--backend-seconds", type=int, default=1500)
    parser.add_argument("--process-seconds", type=int, default=4200)
    args = parser.parse_args()
    required = (args.dual_reference_report, args.dual_reference_result, args.dual_reference_executable)
    if args.stage == "candidate" and any(path is None for path in required):
        parser.error("candidate stage requires all dual-reference artifacts")
    outcome = run(args)
    raise SystemExit(0 if outcome.get("status") == "passed" else 1)


if __name__ == "__main__":
    main()
