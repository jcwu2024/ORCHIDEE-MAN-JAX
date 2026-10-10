"""Public recurrent/default native checkpoint scoped execution gate.

Complete parameter and initial-state gradients; baseline references are saved
before any new candidate compilation. No checkpoint policy or training change.
"""
from __future__ import annotations
import argparse
import hashlib
import gzip
import importlib.util
import json
import os
import pickle
from pathlib import Path
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
        raise ValueError("recurrent executable input treedef drift")
    infos, tree = jax.tree_util.tree_flatten(executable.args_info)
    values, other = jax.tree_util.tree_flatten(supplied)
    if tree != other:
        raise ValueError("recurrent args_info treedef drift")
    for info, value in zip(infos, values, strict=True):
        if tuple(info._aval.shape) != np.shape(value) or np.dtype(info._aval.dtype) != np.asarray(value).dtype:
            raise ValueError("recurrent executable input shape/dtype drift")


def run(args):
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    from jax.experimental.serialize_executable import serialize, deserialize_and_load
    jax.config.update("jax_enable_x64", True)
    owner = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    gates = load(args.gate_owner, "recurrent_normalized_admission_utilities")
    forward_owner = load(args.forward_probe, "scripts.hpc.probe_shared_forward_reverse")
    report = {"status": "running", "stage": "starting", "mode": args.stage, "horizon": args.horizon,
              "scope": "public_recurrent_loss_all_parameters_and_initial_state_AD",
              "checkpoint_policy": "unchanged_native_default", "dynamic_projection_execution": "original_field_only",
              "precision": "unchanged_FP32_neural_FP64_physical", "rtol": 1e-5, "atol": 1e-6,
              "gpu_count": 1, "training_updates": 0, "production_adoption": False,
              "compiler_options": {"xla_gpu_enable_llvm_module_compilation_parallelism": True} if args.llvm_parallelism else {},
              "budgets_seconds": {"lower": args.lower_seconds, "backend": args.backend_seconds, "process": args.process_seconds}}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        owner.atomic_json(args.output, report)
        print(stage, flush=True)
    if args.output.exists():
        raise FileExistsError("preserve prior output; choose new stem")
    try:
        emit("starting")
        with gates.deadline(max(1, args.process_seconds - (time.perf_counter() - started)), "whole_process", emit):
            from research.daily_coarse_graining.daily_operator import shared_process_head
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
            from scripts.hpc.run_shared_learning import load_learning_checkpoint
            from scripts.hpc.run_shared_transformer_admission import memory_sizes
            execution_file = args.execution_module_file or Path(__file__).with_name("shared_normalized_key_execution.py")
            normalized = load(execution_file, "scoped_recurrent_execution_adapter")
            if args.execution_module_file:
                daily = json.loads(args.required_daily_report.read_text())
                whole_vmap = args.execution_variant == "whole-chunk-vmap"
                execution_variant = "expert5_whole_chunk_axis_vmap" if whole_vmap else "expert5_preserve128_learned_shapes"
                expected_routes = ([{"route_shape":7542,"padded_rows":7552,"padding_rows":10,"chunk":128,
                                     "chunk_count":59,"chunk_axis_primitive":"jax.vmap"}] if whole_vmap else
                                   [{"rows":7542,"padded_rows":7552,"chunk":128,"chunk_count":59}])
                if (daily.get("status")!="passed" or daily.get("strict_numerical_passed") is not True
                        or daily.get("variant") != "field_expert128+backbone_source_reuse+"+execution_variant
                        or daily.get("changed_chunk_routes") != expected_routes
                        or (whole_vmap and daily.get("execution_adapter_sha256") != hashlib.sha256(execution_file.read_bytes()).hexdigest())):
                    raise ValueError("requires passed scoped actual original-four daily-loss evidence")
                emit("daily_gate_verified", execution_variant=execution_variant,
                    accepted_daily_report=str(args.required_daily_report),
                    failed_synthetic_component_gates="preserved_separately_not_waived")
            if args.required_seven_report:
                seven = json.loads(args.required_seven_report.read_text())
                comparisons = seven.get("comparisons", {})
                self_replays = seven.get("initial_state_self_replay", {})
                paired_replays = seven.get("initial_state_pair_replay", [])
                full_replay = (json.loads(args.required_seven_replay_report.read_text())
                               if args.required_seven_replay_report else seven)
                full_pairs = full_replay.get("full_derivative_pair_replay", [])
                if (args.horizon != 30 or seven.get("horizon") != 7
                        or seven.get("status") != "passed"
                        or set(comparisons) != {"loss_and_auxiliary", "all_parameter_gradients", "initial_state_gradient"}
                        or any(not row.get("passed") or row.get("failed") for row in comparisons.values())
                        or comparisons["all_parameter_gradients"].get("coordinates") != 64884481
                        or comparisons["initial_state_gradient"].get("coordinates") != 3854
                        or set(self_replays) != {"default", "candidate"}
                        or any(len(rows) != 7 or any(row.get("coordinates") != 3854 for row in rows)
                               for rows in self_replays.values())
                        or len(paired_replays) != 7
                        or any(not row.get("passed") or row.get("failed") for row in paired_replays)
                        or full_replay.get("status") != "passed"
                        or full_replay.get("identity") != seven.get("identity")
                        or len(full_pairs) != 7
                        or any(set(rows) != set(comparisons) or any(not row.get("passed") or row.get("failed") for row in rows.values())
                               for rows in full_pairs)):
                    raise ValueError("requires complete strict T7 initial and seven full derivative paired gates; self records remain diagnostic")
                emit("seven_paired_gates_verified", accepted_seven_report=str(args.required_seven_report),
                     accepted_seven_replay_report=str(args.required_seven_replay_report) if args.required_seven_replay_report else None)
            if len(devices) != 1 or jax.default_backend() != "gpu":
                raise ValueError("requires one actualGPU")
            preflight = gates.validate_runtime_report(args.runtime_preflight, jax.__version__)
            assets = forward_owner._assets(args)
            profile, reference, bundle, context, checkpoint_path, admissions = assets
            base_identity = {"source_head": reference["source_git_head"],
                             "source_profile_sha256": hashlib.sha256(args.source_profile.read_bytes()).hexdigest(),
                             "method": reference["method"], "artifact_id": int(bundle["artifact_ids"][0]),
                             "horizon": args.horizon, "checkpoint": str(checkpoint_path.resolve()), "jax_version": jax.__version__,
                             "admitted_executable": admissions[f"T{args.horizon}"]["executable"],
                             "candidate_decoder_sha256": hashlib.sha256(args.candidate_module_file.read_bytes()).hexdigest(),
                             "candidate_backbone_sha256": hashlib.sha256(args.backbone_module_file.read_bytes()).hexdigest()}
            identity = base_identity | {"execution_adapter_sha256" if args.execution_module_file else "normalized_key_sha256": hashlib.sha256(Path(normalized.__file__).read_bytes()).hexdigest(),
                                       "variant": (args.execution_variant if args.execution_module_file else "normalized") if args.stage == "candidate" else "original_default_reference"}
            checkpoint = load_learning_checkpoint(checkpoint_path, expected_method=reference["method"])
            if checkpoint["completed_updates"] != 0:
                raise ValueError("requires update0")
            host_parameters = checkpoint["parameters"]
            parameters = jax.device_put(host_parameters)
            case = bundle["payloads"][0]["cases"][args.horizon]
            if len(case) != 5 or np.shape(case.days[0].continuous_state)[0] != args.horizon:
                raise ValueError("wrong recurrent horizon")
            arguments = (parameters, *jax.device_put(case))
            jax.block_until_ready(arguments)
            emit("inputs_ready", identity=identity, baseline_identity=base_identity, runtime_preflight=preflight)

            def checks(actual, expected):
                return {"loss_and_auxiliary": owner.compare_trees(jax, actual[0], expected[0]),
                        "all_parameter_gradients": owner.compare_trees(jax, actual[1][0], expected[1][0]),
                        "initial_state_gradient": owner.compare_trees(jax, actual[1][1], expected[1][1])}
            def validation(actual):
                values = gates.validate_result(jax, (actual[0], actual[1][0]), host_parameters)
                state = np.asarray(actual[1][1])
                if state.shape != np.shape(case[0]) or state.size != 3854 or not np.all(np.isfinite(state)) or not np.any(state != 0):
                    raise gates.NumericalFailure("invalid/meaningless full initial-state gradient")
                return values | {"initial_state_gradient_coordinates": state.size,
                                 "initial_state_gradient_nonzero": int(np.count_nonzero(state)),
                                 "initial_state_gradient_finite": True}

            baseline = baseline_result = saved_baseline = None
            if args.stage == "candidate":
                accepted = json.loads(args.reference_report.read_text())
                # The old default T7 dual reference is already admitted against
                # the parameter-only baseline. The new T30 reference has the
                # same bridge, with its own explicit original-default identity.
                accepted_identity = accepted.get("baseline_identity", accepted.get("identity"))
                if accepted.get("status") != "passed" or accepted_identity != base_identity:
                    raise ValueError("default dual reference identity/admission mismatch")
                with args.reference_result.open("rb") as stream:
                    artifact = pickle.load(stream)
                if artifact.get("identity") not in (base_identity, accepted.get("identity")):
                    raise ValueError("saved default dual result identity mismatch")
                saved_baseline = artifact["result"]
                validation(saved_baseline)
                with gates.forbid_field_compile():
                    with args.reference_executable.open("rb") as stream:
                        serialized = pickle.load(stream)
                    baseline = deserialize_and_load(*serialized)
                    validate_inputs(jax, serialized, baseline, arguments)
                    del serialized
                    baseline_memory = memory_sizes(baseline)
                    if baseline_memory["estimated_live_bytes"] > 24 * 2**30:
                        raise MemoryError("default dual executable exceeds24GiB")
                    baseline_result = jax.device_get(jax.block_until_ready(baseline(*arguments)))
                validation(baseline_result)
                replay_checks = checks(baseline_result, saved_baseline)
                owner.atomic_pickle(args.output.with_suffix(".reference-replay.pkl"), {"identity": identity, "result": baseline_result})
                emit("default_reference_replayed", baseline_compiles=0, baseline_memory=baseline_memory,
                     baseline_saved_replay_comparisons=replay_checks,
                     baseline_state_replay_drift=not replay_checks["initial_state_gradient"]["passed"])
                # Retain/report baseline state drift independently; the direct
                # strict candidate comparison uses this same-process replay.
                if not all(replay_checks[k]["passed"] for k in ("loss_and_auxiliary", "all_parameter_gradients")):
                    raise gates.NumericalFailure("default reference parameter/loss bridge failed")
                del saved_baseline

            adapter = owner.load_staged_adapter(args.candidate_module_file)
            backbone = owner.load_backbone_assembler(args.backbone_module_file)
            _rollout, objective = make_shared_recurrent_objective(context["head"], context["template"], context["state_scales"], context["science_scales"])
            options = {"compiler_options": report["compiler_options"]} if args.llvm_parallelism else {}
            backward = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True), **options)
            from contextlib import nullcontext
            key_policy = normalized.patches(context["head"].shared_config) if args.stage == "candidate" else nullcontext()
            with owner.candidate_process_head_patches(shared_process_head, adapter, backbone), key_policy:
                emit("lowering")
                with gates.deadline(args.lower_seconds, "lower", emit):
                    began = time.perf_counter()
                    lowered = backward.lower(*arguments)
                    lower_seconds = time.perf_counter() - began
                emit("compiling", lower_seconds=lower_seconds)
                with gates.deadline(args.backend_seconds, "backend", emit):
                    began = time.perf_counter()
                    executable = lowered.compile()
                    compile_seconds = time.perf_counter() - began
            exe_path = args.reference_executable if args.stage == "reference" else args.output.with_suffix(".candidate.executable.pkl")
            result_path = args.reference_result if args.stage == "reference" else args.output.with_suffix(".candidate.pkl")
            owner.atomic_pickle(exe_path, serialize(executable))
            def export_hlo():
                if not args.export_hlo:
                    return {"hlo_export_requested": False}
                path = args.output.with_suffix(".optimized.hlo.txt.gz")
                with gzip.open(path, "wt", compresslevel=1) as stream:
                    stream.write(executable.as_text())
                return {"hlo_export_requested": True, "optimized_hlo_artifact": str(path)}
            if args.execution_module_file:
                emit("adapter_scope_recorded", changed_chunk_routes=getattr(normalized,"PATCHED_ROWS",None))
            memory = memory_sizes(executable)
            emit("executable_saved", compile_seconds=compile_seconds, compiled_memory=memory, executable=str(exe_path))
            if memory["estimated_live_bytes"] > 24 * 2**30:
                raise MemoryError("new recurrent dual executable exceeds24GiB")
            with gates.deadline(120, "first_result", emit):
                actual = jax.device_get(jax.block_until_ready(executable(*arguments)))
            owner.atomic_pickle(result_path, {"identity": identity, "result": actual})
            valid = validation(actual)
            stats = jax.local_devices()[0].memory_stats() or {}
            emit("result_saved", validation=valid, device_memory=stats, result_artifact=str(result_path))
            if "peak_bytes_in_use" not in stats or stats["peak_bytes_in_use"] > 24 * 2**30:
                raise MemoryError("observed process peak unavailable/exceeds24GiB")
            if args.stage == "reference":
                with gates.forbid_field_compile():
                    with Path(admissions[f"T{args.horizon}"]["executable"]).open("rb") as stream:
                        admitted = deserialize_and_load(*pickle.load(stream))
                    trusted = jax.device_get(jax.block_until_ready(admitted(*arguments)))
                bridge = {"loss_and_auxiliary": owner.compare_trees(jax, actual[0], trusted[0]),
                          "all_parameter_gradients": owner.compare_trees(jax, actual[1][0], trusted[1])}
                emit("reference_bridge_compared", comparisons=bridge)
                gates.require_parity(bridge)
                emit("completed", status="passed", initial_state_gradient_finite=True, **export_hlo())
                return
            parity = checks(actual, baseline_result)
            emit("strict_compared", comparisons=parity)
            candidate_label = "candidate" if args.execution_module_file else "normalized"
            times = {"default": [], candidate_label: []}
            self_checks = {"default": [], candidate_label: []}
            state_pairs = []
            for repeat in range(7):
                order = ("default", candidate_label) if repeat % 2 == 0 else (candidate_label, "default")
                states = {}
                for name in order:
                    with gates.deadline(120, "warm_" + name, emit), gates.forbid_field_compile():
                        began = time.perf_counter()
                        value = (baseline if name == "default" else executable)(*arguments)
                        jax.block_until_ready(value)
                        times[name].append(time.perf_counter() - began)
                    # Initial-state self replay is small and diagnoses known
                    # within-arm state-coordinate instability independently.
                    state = jax.device_get(value[1][1])
                    states[name] = state
                    if args.execution_module_file:
                        owner.atomic_pickle(args.output.with_suffix(f".{name}.repeat{repeat}.initial_state_vjp.pkl"),state)
                    reference_state = baseline_result[1][1] if name == "default" else actual[1][1]
                    self_checks[name].append(owner.compare_trees(jax, state, reference_state))
                    del value
                state_pairs.append(owner.compare_trees(jax,states[candidate_label],states["default"]))
            initial_passed = all(row["passed"] for row in parity.values())
            replay_passed = all(row["passed"] for row in state_pairs)
            passed = initial_passed and replay_passed
            emit("completed", status="passed" if passed else "parity_failed", matched_warm_seconds=times,
                 strict_initial_parity_passed=initial_passed, strict_state_pair_replay_passed=replay_passed,
                 initial_state_self_replay=self_checks,
                 initial_state_pair_replay=state_pairs,
                 full_parameter_pair_replay_required=True, self_replays_are_diagnostic=True,
                 **{"default_over_candidate_median_ratio" if args.execution_module_file else "default_over_normalized_median_ratio":float(np.median(times["default"]) / np.median(times[candidate_label]))},
                 device_memory=jax.local_devices()[0].memory_stats(), complete_update_admitted=False,
                 **export_hlo())
            if not passed:
                raise gates.NumericalFailure("strict complete recurrent derivative gate failed")
    except BaseException as error:
        emit("exception", status="resource_failed" if isinstance(error, MemoryError) else "numerical_failed" if isinstance(error, gates.NumericalFailure) else "error", error=str(error), error_type=type(error).__name__)
        raise


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, choices=(7, 30), required=True)
    p.add_argument("--stage", choices=("reference", "candidate"), required=True)
    for key in ("output", "patch-owner", "gate-owner", "forward-probe", "source-profile", "input-bundle", "prepared-context",
                "daily-admission-report", "seven-admission-report", "thirty-admission-report", "support-owner-file",
                "runtime-preflight", "candidate-module-file", "backbone-module-file", "reference-report", "reference-result", "reference-executable"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--lower-seconds", type=int, default=360)
    p.add_argument("--backend-seconds", type=int, default=5400)
    p.add_argument("--process-seconds", type=int, default=7200)
    p.add_argument("--llvm-parallelism", action="store_true")
    p.add_argument("--export-hlo",action="store_true",help="After numerical results/timings, explicitly export gzip HLO for a named diagnostic")
    p.add_argument("--execution-module-file",type=Path)
    p.add_argument("--execution-variant",choices=("preserve128","whole-chunk-vmap"),default="preserve128")
    p.add_argument("--required-daily-report",type=Path)
    p.add_argument("--required-seven-report",type=Path)
    p.add_argument("--required-seven-replay-report",type=Path)
    return p


def main():
    args = make_parser().parse_args()
    args.component = f"T{args.horizon}"
    run(args)


if __name__ == "__main__":
    main()
