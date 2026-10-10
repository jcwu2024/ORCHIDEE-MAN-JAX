"""Proposed fixed-local-S4096 weight-gradient public F4S diagnostic.

Independent draft using the existing original-input/loss/admission support.
No global loss scaling or forward precision change; no current GPU approval.
"""
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



def parameter_engineering_comparison(owner, metrics, jax, precision, actual, reference, contract, masks, mask_evidence):
    row = metrics.engineering_comparison(jax, precision, actual, reference, contract, masks, mask_evidence)
    reference_hard = {name:int(np.asarray(value)) for name,value in reference[0][1]['hard'].items()}
    row['reference_physical_hard_passed'] = len(reference_hard)==11 and not any(reference_hard.values())
    strict = _all_outputs(owner,jax,actual,reference)
    row['parameter_engineering_passed'] = len(row['parameters']) == 16 and all(item['screening_passed'] for item in row['parameters'].values())
    row['strict_primal_aux_state_passed'] = all(strict[name]['passed'] for name in
        ('loss','auxiliary','all_initial_state_gradients'))
    row['screening_passed'] = bool(row['parameter_engineering_passed'] and row['strict_primal_aux_state_passed']
        and row['loss_screening_passed'] and row['all_values_finite'] and row['physical_hard_passed']
        and row['reference_physical_hard_passed']
        and row['raw_masks_source_invariance_passed']
        and all(all(v) for v in row['undefined_input_state_gradients_zero_by_sample'].values()))
    row['state_engineering_metrics_are_descriptive_only'] = True
    return row


def hard_admission_passed(comparison):
    return bool(comparison['all_values_finite'] and comparison['physical_hard_passed']
        and comparison['reference_physical_hard_passed']
        and comparison['raw_masks_source_invariance_passed']
        and all(all(value) for value in comparison['undefined_input_state_gradients_zero_by_sample'].values()))

def run(args):
    if args.stage != 'candidate':
        raise ValueError('this independent draft supports one candidate only')
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

    jax, devices = initialize_gpu_backend()
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import serialize, deserialize_and_load

    jax.config.update("jax_enable_x64", True)
    owner = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    gates = load(args.gate_owner, "daily_dual_admission_utilities")
    precision = load(args.execution_module_file, "daily_four_precision_policy")
    interface = json.loads(args.interface_report.read_text())
    interface_correctness = bool(interface.get('alias_restoration_fixture_passed') is True
        and interface.get('native_full_jvp_algorithm_present') is True
        and all(check.get('parameter_engineering_passed') is True if name=='kernel' else check.get('passed') is True
            for case in interface.get('interface_checks',[]) for name,check in case.get('checks',{}).items())
        and all(
        row.get('native_FP16_gemm_instructions')
        and row.get('observed_device_memory',{}).get('peak_bytes_in_use',25*2**30)<=24*2**30
        and all(row.get('checks',{}).get(name,{}).get('passed') is True for name in ('loss','bias','input'))
        and row.get('checks',{}).get('kernel',{}).get('parameter_engineering_passed') is True
        for row in interface.get('actual_shape_GPU_pairs',[])))
    diagnostic_speed_continue = bool(args.component_speed_screen_failed_diagnostic_continue
        and interface.get('status')=='numerical_or_component_speed_failure'
        and interface.get('component_speed_gate') is False and interface_correctness)
    if ((interface.get('status') != 'passed' and not diagnostic_speed_continue) or interface.get('mode') != 'gpu-interface'
        or interface.get('jax_version') != jax.__version__
        or interface.get('policy') != precision.policy_evidence()
        or interface.get('policy_sha256') != hashlib.sha256(args.execution_module_file.read_bytes()).hexdigest()
        or interface.get('xla_flags') != os.environ.get('XLA_FLAGS','')
        or (interface.get('component_speed_gate') is not True and not diagnostic_speed_continue)
        or not interface_correctness
        or interface.get('representative_shape_set')!='source_F4S_decoder_and_backbone'
        or [(row.get('shape'),row.get('width')) for row in interface.get('actual_shape_GPU_pairs',[])]
            != [([4,128,1,384],1536),([4,7,1,768],3072)]):
        raise ValueError('passed actual GPU interface/policy/compiler identity is required')
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
        "variant": "field_original" if args.stage == "reference" else "field_expert128+backbone_source_reuse+parameter_gradient_only_local_S4096_F16_F16_F32",
        "dual_input_abi": "(parameters, initial_states[4,3854], daily_samples); dynamic batch with inputs.continuous_state replaced",
        "state_is_optimizer_parameter": False,
        "precision": "original_FP32_forward_state_bias_parameter_grad_F16_F16_F32_local_S4096_FP64_physics",
        "engineering_contract": engineering_contract,
        "compiler_execution": {
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "repair": "operand_upcaster_disabled_after_known_rejection" if '--xla_disable_hlo_passes=operand_upcaster' in os.environ.get('XLA_FLAGS','') else "none; original compiler flags",
            "combination": "parameter_precision_plus_operand_upcaster_compatibility" if '--xla_disable_hlo_passes=operand_upcaster' in os.environ.get('XLA_FLAGS','') else "parameter_precision_original_compiler_flags",
            "native_algorithm_unchanged": True,
            "FP32_neural_fallback": False,
            "custom_AD": "native_custom_jvp_transpose_approximate_parameter_derivative",
        },
        "precision_adopted": False,
        "precision_scope": "active_four_linear_plus_three_FFN_parameter_gradient_only",
        "local_parameter_tangent_scale": 4096, "public_loss_scaling": False,
        "loss_conditioning": {"enabled": False, "public_loss_aux_returned_original": True,
            "full_tree_unscale_in_timed_binary": False},
        "interface_report": str(args.interface_report),
        "component_speed_screen_failed_diagnostic_continue": diagnostic_speed_continue,
        "component_speed_gate": interface.get('component_speed_gate'),
        "component_interface_correctness_passed": interface_correctness,
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
        "candidate_execution": "field_expert128+backbone_source_reuse+parameter_gradient_only_local_S4096_F16_F16_F32",
    }

    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        owner.atomic_json(args.output, report)
        print(f"daily_dual stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    outputs = [args.output, args.result_output, args.executable_output]
    outputs.extend(args.output.with_suffix(f".failed-pair-{repeat}.pkl") for repeat in range(3))
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
            reference_validity = _validate_dual(gates, jax, replay, host_parameters, initial_states)
            reference_undefined_zero = metrics.undefined_states_zero(replay[1][1], masks)
            if not all(reference_undefined_zero):
                raise gates.NumericalFailure('original replay undefined state gradients must remain exactly zero')
            emit('reference_validity_checked', reference_validity=reference_validity,
                 reference_undefined_input_state_gradients_zero_by_sample=reference_undefined_zero,
                 saved_field_state_self_drift=reference_state_drift)

            candidate_adapter = owner.load_staged_adapter(args.candidate_module_file)
            backbone = owner.load_backbone_assembler(args.backbone_module_file)
            patch_context = owner.candidate_process_head_patches(shared_process_head, candidate_adapter, backbone)
            emit("candidate_lowering")
            package = "research.daily_coarse_graining.daily_operator."
            policy_modules = [sys.modules[package + name] for name in ("shared_transformer", "shared_backbone_execution", "shared_backbone_projection_reuse", "shared_decoder_macro_packed")]
            emit("precision_aliases_ready", policy_aliases=precision.alias_metadata(policy_modules))
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
                raise MemoryError("parameter-only-gradient daily dual executable exceeds 24 GiB")
            with gates.deadline(120, "candidate_first_result", emit):
                first = jax.device_get(jax.block_until_ready(candidate(*candidate_arguments)))
            binary_abi = {"inputs": _tree_abi(jax, candidate_arguments), "outputs": _tree_abi(jax, first)}
            owner.atomic_pickle(args.result_output, {"identity": identity, "result": first, "binary_abi": binary_abi})
            validation = metrics.observed_validation(jax, first, host_parameters)
            initial_comparisons = _all_outputs(owner, jax, first, reference_result)
            initial_engineering = parameter_engineering_comparison(owner, metrics, jax, precision, first, reference_result, context["head"].contract, masks, mask_evidence)
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
                 three_pairs_run_after_hard_admission_even_if_numerical_gates_fail=True)

            if not hard_admission_passed(initial_engineering):
                emit('completed', status='screening_failed', initial_strict_passed=initial_passed,
                     comparisons=initial_comparisons, initial_engineering_comparison=initial_engineering,
                     stop_reason='nonfinite, physical, source-mask or undefined-zero admission failed; immediate stop',
                     saved_field_state_self_drift=reference_state_drift, compile_seconds=compile_seconds)
                return report
            times = {"field": [], "candidate": []}
            state_self_drift = {"field": [], "candidate": []}
            pairs = []
            engineering_pairs = []
            state_replays = {"field": [], "candidate": []}
            jax.block_until_ready((field_dual(*arguments), candidate(*candidate_arguments)))
            for repeat in range(3):
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
                engineering_pair = parameter_engineering_comparison(owner, metrics, jax, precision, results["candidate"], results["field"], context["head"].contract, masks, mask_evidence)
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
                if not hard_admission_passed(engineering_pair):
                    emit('completed', status='screening_failed', full_derivative_pair_comparisons=pairs,
                         engineering_pair_comparisons=engineering_pairs,
                         stop_reason='paired hard admission failed; immediate stop')
                    return report
            passed = initial_passed and all(all(row["passed"] for row in pair.values()) for pair in pairs)
            owner.atomic_pickle(args.output.with_suffix(".state-replays.pkl"), {
                "identity": identity,
                "state_gradient_shape": [4, 3854],
                "field": state_replays["field"],
                "candidate": state_replays["candidate"],
            })
            engineering_passed = initial_engineering["screening_passed"] and all(row["screening_passed"] for row in engineering_pairs)
            speed_ratio = float(np.median(times["field"]) / np.median(times["candidate"]))
            speed_passed = speed_ratio >= 1.05
            emit("completed", status="screening_passed" if engineering_passed and speed_passed else "screening_failed", validation=validation,
                 strict_numerical_passed=passed, engineering_screening_passed=engineering_passed,
                 F4S_speed_gate_passed=speed_passed, F4S_speed_gate_minimum=1.05,
                 numerical_gate_failure_prevents_expansion=True,
                 timing_is_diagnostic_until_all_gates_pass=True,
                 initial_engineering_comparison=initial_engineering, engineering_pair_comparisons=engineering_pairs,
                 comparisons=initial_comparisons, full_derivative_pair_comparisons=pairs,
                 field_state_self_drift=state_self_drift["field"], candidate_state_self_drift=state_self_drift["candidate"],
                 saved_field_state_self_drift=reference_state_drift,
                 self_drift_is_diagnostic=True, matched_warm_seconds=times,
                 field_over_candidate_median_ratio=speed_ratio,
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
    parser.add_argument("--stage", choices=("candidate",), required=True)
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
    parser.add_argument("--interface-report", type=Path, required=True)
    parser.add_argument("--component-speed-screen-failed-diagnostic-continue",action="store_true")
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
    parser.add_argument("--process-seconds", type=int, default=2380)
    parser.add_argument("--cli-check-only", action="store_true", help="validate expanded arguments without backend or model access")
    parser.add_argument("--lowering-only", action="store_true")
    parser.add_argument("--failure-dump", type=Path)
    args = parser.parse_args()
    required = (args.dual_reference_report, args.dual_reference_result, args.dual_reference_executable)
    if args.stage == "candidate" and any(path is None for path in required):
        parser.error("candidate stage requires all dual-reference artifacts")
    if args.stage != "candidate":
        parser.error("isolated candidate only; original saved reference must be reused")
    if args.cli_check_only:
        print(json.dumps({'status':'CLI_checked_without_backend_or_model','arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}},indent=2))
        return
    outcome = run(args)
    raise SystemExit(0 if outcome.get("status") in ("screening_passed", "lowering_evidence_saved") else 1)


if __name__ == "__main__":
    main()
