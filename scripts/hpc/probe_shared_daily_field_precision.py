"""Isolated four-field-sample F16_F16_F32 neural alias diagnostic; no adoption."""
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
    precision = load(args.execution_module_file, "daily_four_precision_policy")
    metrics = load(Path(__file__).with_name("shared_daily_precision_metrics.py"), "daily_four_precision_metrics")
    dual = load(Path(__file__).with_name("shared_daily_dual_ad.py"), "shared_daily_dual_ad")
    engineering_contract = dict(precision.CONTRACT)
    engineering_contract.pop("inactive_reference_gradient_coordinates_must_remain_zero", None)
    engineering_contract["actual_undefined_initial_state_gradients_must_remain_zero"] = True
    engineering_contract["source_public_mask_invariance_required"] = True
    report = {
        "status": "running",
        "stage": "starting",
        "mode": args.stage,
        "scope": "original_four_daily_micro4_public_loss_full_parameter_and_initial_state_AD",
        "variant": "field_original" if args.stage == "reference" else "field_expert128+backbone_source_reuse+F16_F16_F32_neural_aliases",
        "dual_input_abi": "(parameters, initial_states[4,3854], daily_samples); dynamic batch with inputs.continuous_state replaced",
        "state_is_optimizer_parameter": False,
        "precision": "F16_F16_F32_selected_neural_aliases_FP32_storage_accumulation_FP64_physics",
        "engineering_contract": engineering_contract,
        "compiler_execution": {
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "repair": "disable_exact_operand_upcaster_pass_to_preserve_explicit_F16_algorithm_operands",
            "native_algorithm_unchanged": True,
            "FP32_neural_fallback": False,
            "custom_AD": False,
        },
        "precision_adopted": False,
        "loss_conditioning": {"enabled": args.loss_scale is not None, "dynamic_scale": args.loss_scale,
            "scale_dtype": "float64", "native_scaled_loss_AD": args.loss_scale is not None,
            "public_loss_aux_returned_original": True, "full_tree_unscale_in_timed_binary": args.loss_scale is not None,
            "gradient_divisor_cast_to_each_original_dtype": True, "custom_AD": False},
        "inactive_gradient_gate_scope": {"structural_inactive_parameters_and_outputs": "not_completely_audited", "arbitrary_reference_zeros": "descriptive_only"},
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
            "precision_policy": {"path": str(args.execution_module_file),
                                     "sha256": hashlib.sha256(args.execution_module_file.read_bytes()).hexdigest()},
        },
        "reference_execution": "field_expert128+backbone_source_reuse",
        "candidate_execution": "field_expert128+backbone_source_reuse+F16_F16_F32_neural_aliases",
    }

    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        owner.atomic_json(args.output, report)
        print(f"daily_dual stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    outputs = [args.output, args.result_output, args.executable_output]
    if args.stage == "candidate":
        outputs.extend((args.output.with_suffix(".field-replay.pkl"), args.output.with_suffix(".state-replays.pkl")))
        if args.loss_scale is not None:
            outputs.append(args.output.with_suffix(".dynamic-S1.pkl"))
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
            masks, mask_evidence = metrics.source_mask_evidence(jax, context, args.candidate_module_file)
            emit("source_masks_audited", mask_evidence=mask_evidence)
            checkpoint = load_learning_checkpoint(Path(context_identity["initial_checkpoint"]), expected_method=reference["method"])
            if checkpoint["completed_updates"] != 0:
                raise ValueError("requires update-0 parameters")
            host_parameters = checkpoint["parameters"]
            if sum(np.size(value) for value in jax.tree_util.tree_leaves(host_parameters)) != 64_884_481:
                raise ValueError("parameter architecture drift")
            parameters, daily4, states = jax.device_put((host_parameters, daily4, states))
            jax.block_until_ready((parameters, daily4, states))
            identity = parameter_identity | {
                "comparison_schema": "shared_daily_field_precision_dual_micro4_v1",
                "original_rows": rows,
                "state_gradient_shape": [4, 3854],
                "state_gradient_coordinates": 15_416,
                "original_reference_parameter_bridge_report": str(args.parameter_report),
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
            input_abi = _tree_abi(jax, arguments)
            if input_abi["leaf_count"] != 891:
                raise ValueError("original dynamic 891-leaf input ABI changed")
            candidate_arguments = arguments
            if args.loss_scale is not None:
                backward = dual.make_scaled_daily_dual_value_and_grad(jax, objective)
                candidate_arguments = arguments + (jnp.asarray(args.loss_scale, dtype=jnp.float64),)
                identity = identity | {"comparison_schema": "shared_daily_field_precision_scaled_dual_micro4_v1",
                                       "dynamic_loss_scale": args.loss_scale, "candidate_argument_leaf_count": 892}
                emit("scaled_candidate_892_input_abi_verified", identity=identity,
                     candidate_input_abi=_tree_abi(jax, candidate_arguments))
            emit("dynamic_891_input_abi_verified", input_abi=input_abi)

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
            package = "research.daily_coarse_graining.daily_operator."
            policy_modules = [sys.modules[package + name] for name in ("shared_transformer", "shared_backbone_execution", "shared_backbone_projection_reuse", "shared_decoder_macro_packed")]
            emit("precision_aliases_ready", policy_aliases=metrics.alias_metadata(policy_modules))
            with patch_context, precision.neural_policy(policy_modules, mixed=True):
                with gates.deadline(args.lower_seconds, "candidate_lower", emit):
                    began = time.perf_counter()
                    lowered = backward.lower(*candidate_arguments)
                    lower_seconds = time.perf_counter() - began
                emit("candidate_compiling", lower_seconds=lower_seconds)
                if hasattr(precision, "policy_evidence"):
                    emit("lowered_precision_policy_evidence", precision_policy_evidence=precision.policy_evidence())
                if args.lowering_only:
                    collector = load(Path(__file__).with_name("collect_shared_precision_lowering.py"), "full_ad_dot_lowering_collector")
                    evidence = collector.collect(lowered)
                    owner.atomic_json(args.output.with_suffix(".dot-signatures.json"), evidence)
                    emit("completed", status="lowering_evidence_saved", lower_seconds=lower_seconds,
                         dot_evidence=str(args.output.with_suffix(".dot-signatures.json")),
                         candidate_backend_compilations=0, candidate_model_values_measured=False)
                    return report
                with gates.deadline(args.backend_seconds, "candidate_backend", emit):
                    began = time.perf_counter()
                    compile_options = {}
                    if args.failure_dump is not None:
                        args.failure_dump.mkdir(parents=True, exist_ok=False)
                        compile_options = {
                            "xla_dump_to": str(args.failure_dump),
                            "xla_dump_hlo_as_proto": True,
                            "xla_dump_compress_protos": True,
                            "xla_dump_disable_metadata": True,
                            "xla_dump_hlo_module_re": "jit_objective",
                            "xla_dump_hlo_pass_re": "gemm",
                            "xla_dump_max_hlo_modules": 1,
                        }
                        emit("failure_gpu_pass_capture_configured", failure_dump_options=compile_options)
                    candidate = lowered.compile(compiler_options=compile_options)
                    compile_seconds = time.perf_counter() - began
            serialized_candidate = serialize(candidate)
            _validate_binary_inputs(jax, serialized_candidate, candidate, candidate_arguments)
            owner.atomic_pickle(args.executable_output, serialized_candidate)
            memory = memory_sizes(candidate)
            if memory["estimated_live_bytes"] > 24 * 2**30:
                raise MemoryError("V4 daily dual executable exceeds 24 GiB")
            scaled_s1_result = None
            if args.loss_scale is not None:
                with args.previous_precision_result.open("rb") as stream:
                    previous_precision = pickle.load(stream)
                for key in ("source_git_head", "method", "initial_checkpoint", "artifact_id", "original_rows"):
                    if previous_precision["identity"].get(key) != identity.get(key):
                        raise ValueError("previous precision identity differs")
                s1_arguments = arguments + (jnp.asarray(1.0, dtype=jnp.float64),)
                scaled_s1_result = jax.device_get(jax.block_until_ready(candidate(*s1_arguments)))
                s1_previous = _all_outputs(owner, jax, scaled_s1_result, previous_precision["result"])
                s1_engineering = metrics.engineering_comparison(jax, precision, scaled_s1_result, previous_precision["result"], context["head"].contract, masks, mask_evidence)
                owner.atomic_pickle(args.output.with_suffix(".dynamic-S1.pkl"), {"identity": identity,
                    "result": scaled_s1_result, "dynamic_scale": 1.0, "comparison_to_previous_precision": s1_previous})
                emit("dynamic_S1_wrapper_diagnostic", dynamic_S1_previous_precision_comparisons=s1_previous,
                     dynamic_S1_previous_precision_engineering=s1_engineering,
                     dynamic_S1_is_diagnostic_not_admission=True)
                if not s1_engineering["all_values_finite"] or not s1_engineering["physical_hard_passed"]:
                    raise gates.NumericalFailure("dynamic S1 wrapper failed finite/physical gate; result preserved")
                del previous_precision
            with gates.deadline(120, "candidate_first_result", emit):
                first = jax.device_get(jax.block_until_ready(candidate(*candidate_arguments)))
            binary_abi = {"inputs": _tree_abi(jax, candidate_arguments), "outputs": _tree_abi(jax, first)}
            owner.atomic_pickle(args.result_output, {"identity": identity, "result": first, "binary_abi": binary_abi})
            validation = metrics.observed_validation(jax, first, host_parameters)
            initial_comparisons = _all_outputs(owner, jax, first, reference_result)
            initial_engineering = metrics.engineering_comparison(jax, precision, first, reference_result, context["head"].contract, masks, mask_evidence)
            if args.loss_scale is not None and (not initial_engineering["all_values_finite"] or not initial_engineering["physical_hard_passed"]
                    or not all(initial_engineering["undefined_input_state_gradients_zero_by_sample"]["candidate"])):
                emit("scaled_initial_hard_gate_failed", initial_engineering_comparison=initial_engineering, validation=validation)
                raise gates.NumericalFailure("scaled initial finite/physical/actual undefined gate failed; result preserved")
            if scaled_s1_result is not None:
                emit("scale_vs_dynamic_S1_diagnostic", scale_vs_dynamic_S1_comparisons=_all_outputs(owner, jax, first, scaled_s1_result),
                     scale_vs_dynamic_S1_engineering=metrics.engineering_comparison(jax, precision, first, scaled_s1_result, context["head"].contract, masks, mask_evidence))
                del scaled_s1_result
            stats = jax.local_devices()[0].memory_stats() or {}
            if "peak_bytes_in_use" not in stats or stats["peak_bytes_in_use"] > 24 * 2**30:
                raise MemoryError("observed candidate peak unavailable or exceeds 24 GiB")
            initial_passed = all(row["passed"] for row in initial_comparisons.values())
            emit("strict_initial_comparison_passed" if initial_passed else "strict_initial_comparison_failed",
                 comparisons=initial_comparisons, initial_strict_passed=initial_passed,
                 validation=validation, candidate_memory=memory, device_memory=stats,
                 initial_engineering_comparison=initial_engineering,
                 saved_field_state_self_drift=reference_state_drift,
                 binary_abi=binary_abi, execution_routes="unchanged_original_field128_source_reuse",
                 seven_pair_diagnostics_will_run=True)

            times = {"field": [], "candidate": []}
            state_self_drift = {"field": [], "candidate": []}
            pairs = []
            engineering_pairs = []
            state_replays = {"field": [], "candidate": []}
            jax.block_until_ready((field_dual(*arguments), candidate(*candidate_arguments)))
            for repeat in range(7):
                order = ("field", "candidate") if repeat % 2 == 0 else ("candidate", "field")
                results = {}
                for name in order:
                    executable = field_dual if name == "field" else candidate
                    with gates.deadline(120, f"{name}_warm_{repeat}", emit), gates.forbid_field_compile():
                        began = time.perf_counter()
                        value = executable(*(arguments if name == "field" else candidate_arguments))
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
                engineering_pair = metrics.engineering_comparison(jax, precision, results["candidate"], results["field"], context["head"].contract, masks, mask_evidence)
                engineering_pairs.append(engineering_pair)
                if any(not row["passed"] for row in pair.values()) or not engineering_pair["screening_passed"]:
                    owner.atomic_pickle(args.output.with_suffix(f".failed-pair-{repeat}.pkl"), {
                        "identity": identity,
                        "field_result": results["field"],
                        "candidate_result": results["candidate"],
                        "comparisons": pair,
                        "engineering_comparison": engineering_pair,
                    })
                emit("paired_repeat_saved", repeat=repeat, full_derivative_pair_comparisons=pairs, engineering_pair_comparisons=engineering_pairs)
                del results
            passed = initial_passed and all(all(row["passed"] for row in pair.values()) for pair in pairs)
            owner.atomic_pickle(args.output.with_suffix(".state-replays.pkl"), {
                "identity": identity,
                "state_gradient_shape": [4, 3854],
                "field": state_replays["field"],
                "candidate": state_replays["candidate"],
            })
            engineering_passed = initial_engineering["screening_passed"] and all(row["screening_passed"] for row in engineering_pairs)
            emit("completed", status="screening_passed" if engineering_passed else "screening_failed", validation=validation,
                 strict_numerical_passed=passed, engineering_screening_passed=engineering_passed,
                 initial_engineering_comparison=initial_engineering, engineering_pair_comparisons=engineering_pairs,
                 comparisons=initial_comparisons, full_derivative_pair_comparisons=pairs,
                 field_state_self_drift=state_self_drift["field"], candidate_state_self_drift=state_self_drift["candidate"],
                 saved_field_state_self_drift=reference_state_drift,
                 self_drift_is_diagnostic=True, matched_warm_seconds=times,
                 field_over_candidate_median_ratio=float(np.median(times["field"]) / np.median(times["candidate"])),
                 candidate_memory=memory, reference_memory=reference_memory, device_memory=jax.local_devices()[0].memory_stats(),
                 strict_rtol=1e-5, strict_atol=1e-6, all_parameter_and_state_coordinates_compared=True,
                 state_replays=str(args.output.with_suffix(".state-replays.pkl")),
                 binary_abi=binary_abi,
                 optimizer_state_gradient_use=False, execution_routes="unchanged_original_field128_source_reuse",
                 compile_seconds=compile_seconds, production_adoption=False)
            return report
    except BaseException as error:
        if args.failure_dump is not None and args.failure_dump.exists():
            report["failure_dump_manifest"] = [{"name": path.name, "bytes": path.stat().st_size}
                                               for path in sorted(args.failure_dump.iterdir()) if path.is_file()]
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
    parser.add_argument("--loss-scale", type=float)
    parser.add_argument("--previous-precision-result", type=Path)
    parser.add_argument("--process-seconds", type=int, default=4200)
    parser.add_argument("--lowering-only", action="store_true")
    parser.add_argument("--failure-dump", type=Path)
    args = parser.parse_args()
    required = (args.dual_reference_report, args.dual_reference_result, args.dual_reference_executable)
    if args.stage == "candidate" and any(path is None for path in required):
        parser.error("candidate stage requires all dual-reference artifacts")
    if args.stage != "candidate":
        parser.error("isolated candidate only; original saved reference must be reused")
    if args.loss_scale is not None and (args.loss_scale != 256 or args.previous_precision_result is None):
        parser.error("bounded conditioning route requires S256 and the saved previous precision result")
    outcome = run(args)
    raise SystemExit(0 if outcome.get("status") in ("screening_passed", "lowering_evidence_saved") else 1)


if __name__ == "__main__":
    main()
