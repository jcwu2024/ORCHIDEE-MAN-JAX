"""Matched changing-weight updates using admitted daily and recurrent binaries.

Both routes execute parameter+initial-state AD for T7/T30 and synchronize the
complete result before extracting parameter gradients for the public runner.
No training checkpoint is retained and no model is lowered or compiled.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import time

import numpy as np

from probe_shared_normalized_keys_recurrent import load, validate_inputs


def admission(path, *, horizon=None, execution_variant="whole-chunk-vmap", paired_replay_path=None):
    report = json.loads(path.read_text())
    if report.get("status") != "passed" or report.get("rtol") != 1e-5 or report.get("atol") != 1e-6:
        raise ValueError("candidate gate status/tolerance not admitted")
    checks = report.get("comparisons", {})
    if not checks or any(not row["passed"] or row["failed"] for row in checks.values()):
        raise ValueError("candidate strict comparisons not admitted")
    if checks["all_parameter_gradients"]["coordinates"] != 64884481:
        raise ValueError("requires complete parameter-gradient admission")
    if horizon is None:
        valid = report["candidate_admission"]
        daily_variant = ("expert5_whole_chunk_axis_vmap" if execution_variant == "whole-chunk-vmap" else "normalized_key_banks")
        if (len(valid["hard_counts"]) != 11 or report.get("scope") != "original_four_daily_micro4_parameter_AD_only"
                or report.get("variant") != "field_expert128+backbone_source_reuse+"+daily_variant
                or report.get("precision") != "unchanged_FP32_neural_FP64_physical"):
            raise ValueError("daily physical gate scope differs")
        binary = Path(report["candidate_executable"])
    else:
        valid = report["validation"]
        if (report.get("horizon") != horizon or report.get("checkpoint_policy") != "unchanged_native_default"
                or report.get("dynamic_projection_execution") != "original_field_only"
                or report.get("precision") != "unchanged_FP32_neural_FP64_physical"
                or len(valid["hard_counts"]) != 14
                or valid.get("initial_state_gradient_coordinates") != 3854
                or valid.get("initial_state_gradient_finite") is not True):
            raise ValueError("recurrent scientific/gradient scope differs")
        candidate_name = "candidate" if execution_variant == "whole-chunk-vmap" else "normalized"
        if report["identity"].get("variant") != ("whole-chunk-vmap" if execution_variant == "whole-chunk-vmap" else "normalized"):
            raise ValueError("recurrent execution variant differs")
        for name in ("default", candidate_name):
            rows = report["initial_state_self_replay"][name]
            if len(rows) != 7 or any(row.get("coordinates") != 3854 for row in rows):
                raise ValueError("recurrent state self diagnostics incomplete")
        paired_states = report.get("initial_state_pair_replay", [])
        if len(paired_states) != 7 or any(not row["passed"] or row["failed"] for row in paired_states):
            raise ValueError("recurrent seven strict paired state gates failed")
        if execution_variant == "whole-chunk-vmap":
            if paired_replay_path is None:
                raise ValueError("V4 requires seven full derivative paired replay evidence")
            replay = json.loads(paired_replay_path.read_text())
            full_pairs = replay.get("full_derivative_pair_replay", [])
            if (replay.get("status") != "passed" or replay.get("identity") != report["identity"]
                    or replay.get("rtol") != 1e-5 or replay.get("atol") != 1e-6
                    or len(full_pairs) != 7
                    or any(set(rows) != {"loss_and_auxiliary", "all_parameter_gradients", "initial_state_gradient"}
                           or any(not row["passed"] or row["failed"] for row in rows.values())
                           or rows["all_parameter_gradients"]["coordinates"] != 64884481
                           or rows["initial_state_gradient"]["coordinates"] != 3854 for rows in full_pairs)):
                raise ValueError("V4 complete strict paired derivative gates incomplete")
        binary = Path(report["executable"])
    if valid["gradient_coordinates"] != 64884481 or valid["gradient_leaves"] != 300 or any(valid["hard_counts"].values()):
        raise ValueError("gradient/physical admission incomplete")
    return report, binary


def daily_dual_admission(path, reference_path, adapter_path):
    """Admit G2 by complete public daily derivatives, never its component."""
    import hashlib
    report = json.loads(path.read_text())
    reference = json.loads(reference_path.read_text())
    scopes = {"loss", "auxiliary", "all_parameter_gradients", "all_initial_state_gradients"}
    if (report.get("status") != "passed" or reference.get("status") != "passed"
            or report.get("variant") != "field_expert128+backbone_source_reuse+expert5_native_map_batch_size2"
            or report.get("scope") != "original_four_daily_micro4_public_loss_full_parameter_and_initial_state_AD"
            or report.get("precision") != "unchanged_FP32_neural_FP64_physical"
            or report.get("rtol") != 1e-5 or report.get("atol") != 1e-6
            or report.get("state_is_optimizer_parameter") is not False
            or report.get("binary_abi", {}).get("inputs", {}).get("leaf_count") != 891
            or report.get("source_artifacts", {}).get("group2_execution_adapter", {}).get("sha256")
                != hashlib.sha256(adapter_path.read_bytes()).hexdigest()):
        raise ValueError("G2 complete daily dual status/precision/dynamic ABI/adapter not admitted")
    pairs = [report.get("comparisons", {})] + report.get("full_derivative_pair_comparisons", [])
    if (len(pairs) != 8 or any(set(rows) != scopes or any(not row.get("passed") or row.get("failed") for row in rows.values())
                             or rows["all_parameter_gradients"].get("coordinates") != 64884481
                             or rows["all_initial_state_gradients"].get("coordinates") != 15416 for rows in pairs)):
        raise ValueError("G2 initial/seven complete strict derivative pairs failed/incomplete")
    for key in ("field_state_self_drift", "candidate_state_self_drift"):
        if len(report.get(key, [])) != 7:
            raise ValueError("daily state self diagnostics incomplete")
    valid = report["validation"]
    if (valid.get("gradient_coordinates") != 64884481 or valid.get("gradient_leaves") != 300
            or valid.get("gradient_nonzero", 0) <= 0 or valid.get("gradient_norm", 0) <= 0
            or valid.get("initial_state_gradient_coordinates") != 15416
            or valid.get("initial_state_gradient_finite") is not True
            or valid.get("initial_state_gradient_nonzero", 0) <= 0
            or len(valid.get("hard_counts", {})) != 11 or any(valid["hard_counts"].values())):
        raise ValueError("daily full gradient/physical validation incomplete")
    if (report.get("device_memory", {}).get("peak_bytes_in_use", 24*2**30+1) > 24*2**30
            or report.get("candidate_memory", {}).get("estimated_live_bytes", 24*2**30+1) > 24*2**30):
        raise ValueError("daily full dual observed/estimated memory not admitted")
    for key in ("source_git_head", "method", "initial_checkpoint", "artifact_id", "original_rows", "state_gradient_shape"):
        if report["identity"].get(key) != reference["identity"].get(key):
            raise ValueError("daily original reference identity differs")
    return report, reference


def daily_precision_admission(path, reference_path, *, ffn_only=False, policy_path=None):
    """Require complete engineering evidence without declaring strict admission."""
    report, reference = (json.loads(p.read_text()) for p in (path, reference_path))
    expected_precision = ("F16_F16_F32_three_active_FFN_aliases_all_linear_FP32_storage_accumulation_FP64_physics"
                          if ffn_only else "F16_F16_F32_selected_neural_aliases_FP32_storage_accumulation_FP64_physics")
    if ffn_only:
        import hashlib
        identity, conditioning = report.get("identity", {}), report.get("loss_conditioning", {})
        if (identity.get("comparison_schema") != "shared_daily_ffn_precision_scaled_dual_micro4_v1"
                or identity.get("precision_scope") != "FFN_only" or identity.get("dynamic_loss_scale") != 4096
                or identity.get("candidate_argument_leaf_count") != 892
                or conditioning.get("dynamic_scale") != 4096
                or any(conditioning.get(key) is not True for key in ("native_scaled_loss_AD", "public_loss_aux_returned_original",
                    "full_tree_unscale_in_timed_binary", "gradient_divisor_cast_to_each_original_dtype"))
                or conditioning.get("custom_AD") is not False or policy_path is None
                or report.get("source_artifacts", {}).get("precision_policy", {}).get("sha256")
                    != hashlib.sha256(policy_path.read_bytes()).hexdigest()):
            raise ValueError("FFN-only native scale4096 schema/conditioning/frozen policy differs")
    rows = [report.get("initial_engineering_comparison", {})] + report.get("engineering_pair_comparisons", [])
    if (report.get("status") != "screening_passed" or reference.get("status") != "passed"
            or report.get("engineering_screening_passed") is not True
            or report.get("scope") != "original_four_daily_micro4_public_loss_full_parameter_and_initial_state_AD"
            or report.get("precision") != expected_precision
            or report.get("binary_abi", {}).get("inputs", {}).get("leaf_count") != (892 if ffn_only else 891)
            or report.get("state_is_optimizer_parameter") is not False
            or len(rows) != 8 or any(row.get("screening_passed") is not True
                or row.get("parameter_groups") != 16 or row.get("parameter_coordinates") != 64884481
                or row.get("state_groups") != 1028 for row in rows)):
        raise ValueError("precision requires initial/seven complete per-sample engineering gates")
    valid = report["validation"]
    if (valid.get("all_values_finite") is not True or valid.get("gradient_leaves") != 300
            or valid.get("gradient_coordinates") != 64884481 or valid.get("gradient_nonzero", 0) <= 0
            or valid.get("initial_state_gradient_coordinates") != 15416
            or valid.get("initial_state_gradient_nonzero", 0) <= 0
            or len(valid.get("hard_counts", {})) != 11 or any(valid["hard_counts"].values())
            or report.get("device_memory", {}).get("peak_bytes_in_use", 24*2**30+1) > 24*2**30
            or report.get("candidate_memory", {}).get("estimated_live_bytes", 24*2**30+1) > 24*2**30):
        raise ValueError("precision finite/physical/full-gradient/memory gates invalid")
    for key in ("source_git_head", "method", "initial_checkpoint", "artifact_id", "original_rows", "state_gradient_shape"):
        if report["identity"].get(key) != reference["identity"].get(key):
            raise ValueError("precision original reference identity differs")
    return report, reference


def run(args):
    started = time.perf_counter()
    tree = load(args.tree_owner_file, "normalized_full_update_tree_owner")
    profile, reference, bundle, digest, admitted = tree._read_identity(args)
    import jax
    import jax.numpy as jnp
    from jax._src import compiler
    from jax.experimental.serialize_executable import deserialize_and_load
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.checked_recurrent_execution import device_helper_cache
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
    from scripts.hpc.shared_daily_microbatch import make_daily_batch_slicer

    ffn_precision = args.execution_variant == "daily-only-ffn-precision-diagnostic"
    precision_diagnostic = ffn_precision or args.execution_variant == "daily-only-precision-diagnostic"
    daily_only = precision_diagnostic or args.execution_variant == "daily-only-group2"
    if daily_only:
        daily_report, original_daily = (daily_precision_admission(args.candidate_daily_report, args.daily_dual_reference_report,
            ffn_only=ffn_precision, policy_path=args.execution_module_file)
            if precision_diagnostic else daily_dual_admission(args.candidate_daily_report, args.daily_dual_reference_report, args.execution_module_file))
        daily_binary = args.daily_dual_candidate_executable
        if precision_diagnostic and daily_binary.resolve() != args.candidate_daily_report.with_suffix(".candidate.executable.pkl").resolve():
            raise ValueError("precision diagnostic binary must be the exact screened candidate artifact")
        recurrent = {}
    else:
        daily_report, daily_binary = admission(args.candidate_daily_report, execution_variant=args.execution_variant)
        recurrent = {h: admission(getattr(args, f"candidate_t{h}_report"), horizon=h, execution_variant=args.execution_variant,
                                 paired_replay_path=getattr(args,f"t{h}_paired_replay_report")) for h in (7, 30)}
    if args.output.exists():
        raise FileExistsError("preserve prior complete-update result")
    daily_identity = daily_report["identity"]
    if (daily_identity["source_git_head"] != reference["source_git_head"]
            or daily_identity["method"] != reference["method"] or daily_identity["artifact_id"] != bundle["artifact_ids"][0]
            or (not daily_only and daily_identity["daily_rows"] != 4) or daily_identity["initial_checkpoint"] != bundle["initial_checkpoint"]
            or (not daily_only and daily_identity["baseline_binary"] != admitted["daily"]["executable"])
            or daily_identity["executable"] != profile["executables"]["daily"]["executable"]):
        raise ValueError("daily candidate admission method/data/source drift")
    for report, _binary in recurrent.values():
        identity = report["identity"]
        if (identity["source_profile_sha256"] != digest or identity["method"] != reference["method"]
                or identity["source_head"] != reference["source_git_head"] or identity["artifact_id"] != bundle["artifact_ids"][0]):
            raise ValueError("candidate admission method/data/source drift")
    device = jax.local_devices()[0]
    loss_scale = jax.device_put(jnp.asarray(4096, dtype=jnp.float64)) if ffn_precision else None
    checkpoint = learning.load_learning_checkpoint(Path(bundle["initial_checkpoint"]), expected_method=reference["method"])
    if checkpoint["completed_updates"] != 0:
        raise ValueError("requires declared update0 checkpoint")
    parameters, optimizer = jax.device_put((checkpoint["parameters"], checkpoint["optimizer"]))
    daily = jax.device_put(bundle["payloads"][0]["daily"])
    cases = {h: jax.device_put(bundle["payloads"][0]["cases"][h]) for h in (7, 30)}
    jax.block_until_ready((parameters, optimizer, daily, cases))
    # Saved daily model binaries accept micro4, while the unchanged public
    # accumulator owns all32 rows and evaluates eight distinct micro4 slices.
    with jax.default_device(device), device_helper_cache(device):
        operations = dict(learning.make_tree_operations(jax, jnp, 32))
        slicer = make_daily_batch_slicer(jax, 4).lower(daily, jnp.asarray(0, dtype=jnp.int32)).compile()
        daily4_example = jax.block_until_ready(slicer(daily, jnp.asarray(0, dtype=jnp.int32)))
    precision_context, mask_rows, mask_evidence_rows = None, [], []
    if precision_diagnostic:
        precision_metrics = load(Path(__file__).with_name("shared_daily_precision_metrics.py"), "precision_full_update_metrics")
        precision_policy = load(Path(__file__).with_name("probe_shared_neural_precision.py"), "precision_full_update_policy")
        with args.prepared_context.open("rb") as stream:
            precision_context = pickle.load(stream)
        if precision_context["identity"]["method"] != reference["method"]:
            raise ValueError("precision mask context method differs")
        for microbatch in range(8):
            sample4 = jax.block_until_ready(slicer(daily, jnp.asarray(microbatch * 4, dtype=jnp.int32)))
            mask, mask_evidence = precision_metrics.source_mask_evidence(
                jax, precision_context | {"daily4": sample4}, args.candidate_module_file)
            mask_rows.append(mask)
            mask_evidence_rows.append(mask_evidence)
    paths = {"default": {"daily": args.daily_dual_reference_executable if daily_only else Path(admitted["daily"]["executable"])},
             "candidate": {"daily": daily_binary}}
    for h in (7, 30):
        ref_report = json.loads(getattr(args, f"t{h}_reference_report").read_text())
        accepted_identity = ref_report.get("baseline_identity", ref_report.get("identity"))
        if ref_report.get("status") != "passed" or (not daily_only and accepted_identity != recurrent[h][0]["baseline_identity"]):
            raise ValueError("default dual reference identity/admission drift")
        if daily_only:
            checks = ref_report["comparisons"]
            valid = ref_report.get("validation", {
                "initial_state_gradient_coordinates": ref_report.get("initial_state_gradient_coordinates"),
                "initial_state_gradient_finite": ref_report.get("initial_state_gradient_finite"),
                "gradient_coordinates": checks.get("all_parameter_gradients", {}).get("coordinates"),
                "hard_counts": ref_report.get("hard_counts", {})})
            if (accepted_identity.get("source_head") != reference["source_git_head"]
                    or accepted_identity.get("source_profile_sha256") != digest
                    or accepted_identity.get("method") != reference["method"]
                    or accepted_identity.get("artifact_id") != bundle["artifact_ids"][0]
                    or accepted_identity.get("horizon") != h
                    or ref_report.get("checkpoint_policy", "unchanged_native_default") != "unchanged_native_default"
                    or ("candidate_policy" in ref_report and ref_report["candidate_policy"] != "dots_with_no_batch_dims_saveable")
                    or valid.get("initial_state_gradient_coordinates") != 3854
                    or valid.get("initial_state_gradient_finite") is not True
                    or valid.get("gradient_coordinates") != 64884481
                    or len(valid.get("hard_counts", {})) != 14 or any(valid["hard_counts"].values())
                    or any(not row.get("passed") or row.get("failed") for row in checks.values())):
                raise ValueError("original dual reference scope/parameter-only bridge invalid")
        paths["default"][f"T{h}"] = getattr(args, f"t{h}_reference_executable")
        paths["candidate"][f"T{h}"] = paths["default"][f"T{h}"] if daily_only else recurrent[h][1]
    counters = {"compile_or_get_cached_calls": 0, "backend_compile_calls": 0}
    evaluators, binary_records = {}, {}
    for route, components in paths.items():
        evaluators[route], binary_records[route] = {}, {}
        for name, path in components.items():
            if daily_only and route == "candidate" and name != "daily":
                evaluators[route][name] = evaluators["default"][name]
                binary_records[route][name] = dict(binary_records["default"][name], reused_default_program=True)
                continue
            with path.open("rb") as stream:
                serialized = pickle.load(stream)
            restore = tree.reject_compilation_before_entry(compiler, counters)
            try:
                executable = deserialize_and_load(*serialized)
            finally:
                restore()
            arguments = ((parameters, daily4_example.inputs.continuous_state, daily4_example) if daily_only else
                         (parameters, daily4_example)) if name == "daily" else (parameters, *cases[int(name[1:])])
            if ffn_precision and route == "candidate" and name == "daily":
                arguments += (loss_scale,)
            validate_inputs(jax, serialized, executable, arguments)
            del serialized
            memory = learning.memory_sizes(executable)
            if memory["estimated_live_bytes"] > 24 * 2**30:
                raise MemoryError("loaded model exceeds24GiB live estimate")
            evaluators[route][name] = executable
            binary_records[route][name] = {"path": str(path), "memory": memory}
    update = make_update(reference["method"]["daily_protocol"])
    daily_states = {route: [] for route in evaluators}
    daily_derivative_results = {route: [] for route in evaluators}
    def daily_evaluator(route):
        def call(p, samples):
            arguments = (p, samples.inputs.continuous_state, samples)
            if ffn_precision and route == "candidate":
                arguments += (loss_scale,)
            result = evaluators[route]["daily"](*arguments)
            jax.block_until_ready(result)
            daily_states[route].append(result[1][1])
            if precision_diagnostic:
                daily_derivative_results[route].append(result)
            return result[0], result[1][0]
        return call
    accumulators = {route: learning.make_learning_daily_accumulator(
        jax=jax, jnp=jnp, microbatch_size=4,
        evaluator=daily_evaluator(route) if daily_only else evaluators[route]["daily"], expected_count=32,
        slice_execution="tree", batch_slicer=slicer) for route in evaluators}
    report = {"status": "running", "scope": "matched_daily_only_group2_full_dual_AD_daily32_T7_T30_changing_updates" if daily_only else "matched_recurrent_dual_AD_daily32_T7_T30_changing_updates",
              "execution_variant": args.execution_variant, "self_replays_are_diagnostic": True,
              "daily_derivative_interface": "parameter_and_micro4_initial_state_dynamic_samples" if daily_only else "parameter_only_micro4",
              "recurrent_derivative_interface": "parameter_and_complete_initial_state", "state_gradients_enter_optimizer": False,
              "source_head": reference["source_git_head"], "source_profile_sha256": digest,
              "daily_rows": 32, "daily_microbatch_size": 4, "recurrent_horizons": [7, 30],
              "gpu_count": 1, "rtol": 1e-5, "atol": 1e-6, "model_gradient_compilations": 0,
              "retained_training_updates": 0, "production_adoption": False, "scientific_acceptance": False,
              "saved_binaries": binary_records, "method": reference["method"],
              "timing_scope": "full_public_update_both_routes_include_initial_state_AD_then_synchronized_parameter_extraction",
              "previous_parameter_only_reference_seconds_not_directly_comparable": True,
              "timings": [], "startup_seconds": time.perf_counter() - started}
    if precision_diagnostic:
        report.update(scope="bounded_precision_plus_compatibility_daily32_original_T7_T30_changing_update_diagnostic",
                      diagnostic_only=True, strict_admitted=False,
                      precision_screen=str(args.candidate_daily_report),
                      precision_compatibility=daily_report.get("compiler_execution"),
                      recurrent_precision="unchanged_FP32_neural_FP64_physical",
                      source_masks_by_microbatch=mask_evidence_rows,
                      daily_precision=daily_report["precision"],
                      dynamic_loss_scale=4096 if ffn_precision else None,
                      pair_interpretation="warm_and_update1_same_inputs_updates2_3_separate_parameter_trajectories",
                      gate="unchanged_strict_optimizer_and_updated_parameters_plus_declared_loss_semantic_gradient_engineering_and_hard_gates")
    tree._atomic_write(args.output, report)

    def execute(route, current_parameters, current_optimizer):
        daily_states[route].clear()
        daily_derivative_results[route].clear()
        captured, auxiliary, states = {}, {}, {}
        op = dict(operations)
        def combine(day, seven, thirty):
            value = operations["combine_daily_mean"](day, seven, thirty)
            captured["gradient"] = value
            return value
        op["combine_daily_mean"] = combine
        def daily_accumulator(p, samples):
            result = accumulators[route](p, samples)
            auxiliary["daily"] = result[0][1]
            return result
        wrapped = {}
        for h in (7, 30):
            name = f"T{h}"
            def recurrent_call(*arguments, _name=name):
                result = evaluators[route][_name](*arguments)
                jax.block_until_ready(result)
                auxiliary[_name] = result[0][1]
                states[_name] = result[1][1]
                return result[0], result[1][0]
            wrapped[name] = recurrent_call
        began = time.perf_counter()
        with jax.default_device(device), device_helper_cache(device):
            new_parameters, new_optimizer, row = learning.execute_learning_update(
                jax=jax, parameters=current_parameters, optimizer=current_optimizer,
                daily_samples=daily, seven_case=cases[7], thirty_case=cases[30],
                evaluators=wrapped, operations=op, update=update, daily_accumulator=daily_accumulator)
            jax.block_until_ready((captured["gradient"], new_parameters, new_optimizer, states))
            if daily_only:
                if len(daily_states[route]) != 8:
                    raise ValueError("requires eight distinct dynamic micro4 state derivatives")
                states["daily32_micro4"] = tuple(daily_states[route])
        seconds = time.perf_counter() - began
        hard = {name: tree._hard_counts(value) for name, value in auxiliary.items()}
        memory = device.memory_stats() or {}
        result = {"gradient": captured["gradient"], "parameters": new_parameters, "optimizer": new_optimizer,
                "update": row, "auxiliary": auxiliary, "initial_state_gradients": states,
                "hard": hard, "all_hard_counts_zero": not any(count for value in hard.values() for count in value.values()),
                "memory_gate_passed": memory.get("peak_bytes_in_use", 24 * 2**30 + 1) <= 24 * 2**30,
                "wall_seconds": seconds, "device_memory": memory}
        if precision_diagnostic:
            if len(daily_derivative_results[route]) != 8:
                raise ValueError("requires eight complete daily derivative trees")
            result["daily_derivative_results"] = tuple(daily_derivative_results[route])
        return result

    def compare(left, right):
        from scripts.hpc.probe_shared_execution import compare_trees
        checks = tree._compare(jax, left, right)
        passed = tree._passed(jax, checks, left, right)
        state = compare_trees(jax, left["initial_state_gradients"], right["initial_state_gradients"], rtol=1e-5, atol=1e-6)
        checks["initial_state_gradients"] = state
        hard_memory_passed = all(value["all_hard_counts_zero"] and value["memory_gate_passed"] for value in (left, right))
        strict_passed = passed and state["passed"] and hard_memory_passed
        if not precision_diagnostic:
            return checks, strict_passed
        gradient_groups = precision_metrics.parameter_groups(left["gradient"])
        reference_groups = precision_metrics.parameter_groups(right["gradient"])
        engineering = {"mixed_parameter_groups": {name: precision_policy.vector_metrics(jax, value, reference_groups[name])
                        for name, value in gradient_groups.items()}, "daily_microbatches": []}
        # Compare each micro4 loss/parameter/state tree independently. These
        # actual derivative results are captured before the public accumulator.
        for microbatch, (actual, expected) in enumerate(zip(left["daily_derivative_results"], right["daily_derivative_results"], strict=True)):
            engineering["daily_microbatches"].append(precision_metrics.engineering_comparison(
                jax, precision_policy, actual, expected, precision_context["head"].contract,
                mask_rows[microbatch], mask_evidence_rows[microbatch]))
        loss_checks = []
        for actual, expected in zip(jax.tree_util.tree_leaves(left["update"]["loss"]),
                                    jax.tree_util.tree_leaves(right["update"]["loss"]), strict=True):
            actual, expected = float(actual), float(expected)
            difference = abs(actual-expected)
            relative = None if abs(expected) <= 1e-12 else difference / abs(expected)
            loss_checks.append({"absolute_difference": difference, "relative_difference": relative,
                "passed": bool(np.isfinite(actual) and np.isfinite(expected) and
                    (difference <= 1e-6 if relative is None else relative <= 1e-3))})
        engineering["complete_loss_checks"] = loss_checks
        engineering["screening_passed"] = (all(row["screening_passed"] for row in engineering["mixed_parameter_groups"].values())
            and all(row["screening_passed"] for row in engineering["daily_microbatches"])
            and all(row["passed"] for row in loss_checks))
        recurrent_states = {name: compare_trees(jax, left["initial_state_gradients"][name], right["initial_state_gradients"][name],rtol=1e-5,atol=1e-6)
                            for name in ("T7", "T30")}
        checks.update(strict_complete_passed=strict_passed, engineering=engineering, recurrent_state_strict=recurrent_states)
        diagnostic_gate = (engineering["screening_passed"] and checks["parameters"]["passed"]
            and checks["optimizer"]["passed"] and all(row["passed"] for row in recurrent_states.values())
            and left["all_hard_counts_zero"] and right["all_hard_counts_zero"] and left["hard"] == right["hard"]
            and hard_memory_passed
            and all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves((left,right))))
        return checks, diagnostic_gate

    def preserve_failure(results, phase):
        paths = {}
        for route, result in results.items():
            path = args.output.with_suffix(f".{phase}.{route}.pkl")
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as stream:
                pickle.dump({"method": reference["method"], "source_profile_sha256": digest,
                             "scope": "diagnostic_proposal_not_retained_training_checkpoint", "result": result},
                            stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
            paths[route] = str(path)
        report["failure_artifacts"] = paths

    warm = {}
    try:
        for route in ("default", "candidate"):
            warm[route] = jax.device_get(execute(route, parameters, optimizer))
        checks, passed = compare(warm["candidate"], warm["default"])
    except BaseException as error:
        preserve_failure(warm, "warm-runtime")
        report.update(status="runtime_error", stage="warm", error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter() - started)
        tree._atomic_write(args.output, report)
        raise
    report.update(warm_comparisons=checks, warm_passed=passed, startup_seconds=time.perf_counter() - started,
                  warm_routes={route: tree._timing_record(value) for route, value in warm.items()})
    tree._atomic_write(args.output, report)
    if not passed:
        preserve_failure(warm, "warm")
        report.update(status="diagnostic_hard_or_optimizer_parameter_gate_failed" if precision_diagnostic else "parity_failed", stage="warm")
        tree._atomic_write(args.output, report)
        raise ValueError("warm complete update strict gate failed")
    del warm
    current = {route: (parameters, optimizer) for route in evaluators}
    timings = {route: [] for route in evaluators}
    for repeat in range(3):
        order = ("default", "candidate") if repeat % 2 == 0 else ("candidate", "default")
        results = {}
        restore = tree.reject_compilation_before_entry(compiler, counters)
        try:
            for route in order:
                value = execute(route, *current[route])
                current[route] = value["parameters"], value["optimizer"]
                timings[route].append(value["wall_seconds"])
                results[route] = jax.device_get(value)
                del value
        except BaseException as error:
            preserve_failure(results, f"update{repeat + 1}-runtime")
            report.update(status="runtime_error", stage=f"changing_update_{repeat}",
                          error_type=type(error).__name__, error=str(error), compile_counts=counters,
                          elapsed_seconds=time.perf_counter() - started)
            tree._atomic_write(args.output, report)
            raise
        finally:
            restore()
        try:
            checks, passed = compare(results["candidate"], results["default"])
        except BaseException as error:
            preserve_failure(results, f"update{repeat + 1}-comparison")
            report.update(status="comparison_error", stage=f"changing_update_{repeat}",
                          error_type=type(error).__name__, error=str(error), compile_counts=counters,
                          elapsed_seconds=time.perf_counter() - started)
            tree._atomic_write(args.output, report)
            raise
        steps = {route: int(np.asarray(result["optimizer"].step)) for route, result in results.items()}
        passed = passed and all(step == repeat + 1 for step in steps.values()) and not any(counters.values())
        report["timings"].append({"repeat": repeat, "order": order, "steps": steps, "comparisons": checks,
                                  "input_relation": "same_parameters_and_optimizer" if repeat == 0 else "separate_parameter_and_optimizer_trajectories",
                                  "passed": passed, "routes": {route: tree._timing_record(value) for route, value in results.items()}})
        report.update(elapsed_seconds=time.perf_counter() - started, compile_counts=counters)
        tree._atomic_write(args.output, report)
        print(f"complete_update_pair_{repeat}", flush=True)
        if not passed:
            preserve_failure(results, f"update{repeat + 1}")
            report.update(status="diagnostic_hard_or_optimizer_parameter_gate_failed" if precision_diagnostic else "parity_failed", stage=f"changing_update_{repeat}")
            tree._atomic_write(args.output, report)
            raise ValueError("changing-weight complete update strict gate failed")
        del results
    report.update(status="diagnostic_passed_strict_not_admitted" if precision_diagnostic else "passed", matched_medians_seconds={name: float(np.median(values)) for name, values in timings.items()},
                  default_over_candidate_median_ratio=float(np.median(timings["default"]) / np.median(timings["candidate"])),
                  device_memory=device.memory_stats(), elapsed_seconds=time.perf_counter() - started)
    tree._atomic_write(args.output, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("tree-owner-file", "slice-helper-file", "support-owner-file", "learning-runner-file", "source-profile", "input-bundle",
                 "daily-admission-report", "seven-admission-report", "thirty-admission-report", "t7-reference-report", "t30-reference-report",
                 "t7-reference-executable", "t30-reference-executable", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("daily", "t7", "t30"):
        p.add_argument("--candidate-"+name+"-report", "--normalized-"+name+"-report", dest="candidate_"+name+"_report",type=Path,required=name=="daily")
    p.add_argument("--execution-variant",choices=("whole-chunk-vmap","normalized","daily-only-group2","daily-only-precision-diagnostic","daily-only-ffn-precision-diagnostic"),default="whole-chunk-vmap")
    for name in ("daily-dual-reference-report", "daily-dual-reference-executable", "daily-dual-candidate-executable", "execution-module-file"):
        p.add_argument("--"+name,type=Path)
    for horizon in (7,30):
        p.add_argument(f"--t{horizon}-paired-replay-report",type=Path)
    p.add_argument("--prepared-context",type=Path)
    p.add_argument("--candidate-module-file",type=Path)
    args = p.parse_args()
    if args.execution_variant in ("daily-only-group2", "daily-only-precision-diagnostic", "daily-only-ffn-precision-diagnostic"):
        if any(getattr(args,name) is None for name in ("daily_dual_reference_report", "daily_dual_reference_executable",
                                                     "daily_dual_candidate_executable", "execution_module_file")):
            p.error("daily-only G2 requires its complete dual reference/candidate binaries and adapter")
        if "precision-diagnostic" in args.execution_variant and (args.prepared_context is None or args.candidate_module_file is None):
            p.error("precision diagnostic requires the declared source-mask context and adapter")
    elif args.candidate_t7_report is None or args.candidate_t30_report is None:
        p.error("recurrent candidate routes require T7/T30 admission reports")
    args.inspect_only = False
    run(args)


if __name__ == "__main__":
    main()
