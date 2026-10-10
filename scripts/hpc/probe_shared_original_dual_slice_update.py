"""Original saved dual-AD binaries, leaf/tree slicing, three discarded updates.

No process loss is rebuilt. Only the existing public accumulator's slicing
option differs; parameters and explicit initial states use the same original
F4S/T7/T30 executables in both arms. State gradients are diagnostic outputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import pickle
import time

import numpy as np

from probe_shared_normalized_keys_recurrent import load, validate_inputs

RTOL, ATOL = 1e-5, 1e-6
EXPECTED_BINARIES = {
    "daily": "shared-chunk-vmap-daily-dual-v4-20261001.reference.executable.pkl",
    "T7": "shared-recurrent-checkpoint-policy-20260930-v2-T7.reference.executable.pkl",
    "T30": "shared-original-t30-dual-reference-20261001.reference.executable.pkl",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def admit_original(report_path, binary_path, name, *, reference, bundle, digest):
    report = json.loads(report_path.read_text())
    identity = report["identity"]
    recorded = report.get("executable", report.get("reference_executable"))
    if (report.get("status") != "passed" or report.get("mode") != "reference"
            or binary_path.name != EXPECTED_BINARIES[name] or recorded != str(binary_path)
            or identity.get("method") != reference["method"]
            or identity.get("artifact_id") != bundle["artifact_ids"][0]
            or identity.get("source_git_head", identity.get("source_head")) != reference["source_git_head"]):
        raise ValueError(f"{name} must be the exact original dual reference")
    if name == "daily":
        checks = {key: report[key] for key in (
            "bridge_to_parameter_only_binary", "bridge_to_saved_parameter_result")}
        valid = report["validation"]
        if (report.get("variant") != "field_original"
                or report.get("precision") != "unchanged_FP32_neural_FP64_physical"
                or report.get("state_is_optimizer_parameter") is not False
                or report["binary_abi"]["inputs"]["leaf_count"] != 891
                or valid.get("initial_state_gradient_coordinates") != 15416
                or valid.get("initial_state_gradient_finite") is not True
                or identity.get("initial_checkpoint") != bundle["initial_checkpoint"]):
            raise ValueError("original F4S dual scope/ABI/state evidence differs")
        hard_count = 11
    else:
        checks = report["comparisons"]
        valid = report.get("validation", report)
        if (identity.get("source_profile_sha256") != digest or identity.get("horizon") != int(name[1:])
                or valid.get("initial_state_gradient_coordinates") != 3854
                or valid.get("initial_state_gradient_finite") is not True
                or report.get("checkpoint_policy", "unchanged_native_default") != "unchanged_native_default"
                or report.get("precision", "unchanged_FP32_neural_FP64_physical") != "unchanged_FP32_neural_FP64_physical"):
            raise ValueError(f"{name} original recurrent scope differs")
        hard_count = 14
    hard = valid.get("hard_counts", report.get("hard_counts", {}))
    gradient_coordinates = (valid.get("gradient_coordinates") if name == "daily"
                            else checks.get("all_parameter_gradients", {}).get("coordinates"))
    if (gradient_coordinates != 64884481 or len(hard) != hard_count or any(hard.values())
            or any(row.get("passed") is not True or row.get("failed", row.get("failed_coordinates", 0))
                   or row.get("rtol", RTOL) != RTOL or row.get("atol", ATOL) != ATOL for row in checks.values())):
        raise ValueError(f"{name} original strict parameter bridge/physical gates incomplete")
    return report


def state_comparison(jax, actual, expected):
    """Keep every failed state coordinate and its actual strict allowance."""
    from scripts.hpc.probe_shared_execution import compare_trees
    rows = {}
    for name in ("daily32_micro4", "T7", "T30"):
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("state shape/dtype differs")
        # Match the shared comparator's float64 arithmetic exactly.
        a, b = a.astype(np.float64), b.astype(np.float64)
        difference = np.abs(a-b)
        allowance = ATOL + RTOL*np.abs(b)
        bad = ~np.isfinite(a) | ~np.isfinite(b) | (difference > allowance)
        rows[name] = compare_trees(jax, a, b, rtol=RTOL, atol=ATOL)
        rows[name]["shape"] = list(a.shape)
        rows[name]["failed_coordinate_evidence"] = [
            {"index": [int(i) for i in index], "actual": float(a[index]), "reference": float(b[index]),
             "absolute_difference": float(difference[index]), "allowance": float(allowance[index])}
            for index in zip(*np.nonzero(bad), strict=True)]
        if len(rows[name]["failed_coordinate_evidence"]) != rows[name]["failed_coordinates"]:
            raise ValueError("state diagnostic does not match strict comparator")
    return rows


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("preserve previous diagnostic")
    report = {"status": "running", "stage": "identity", "scope": "original_FP32_leaf_vs_tree_daily32_T7_T30_full_parameter_state_AD_AdamW_three_discarded_changing_updates",
              "rtol": RTOL, "atol": ATOL, "gpu_count": 1, "retained_training_updates": 0,
              "production_adoption": False, "default_changed": False, "state_gradients_enter_optimizer": False,
              "model_gradient_compilations": 0, "timings": [], "result_artifacts": [],
              "budget_seconds": 1200, "precision": "unchanged_FP32_neural_FP64_physical"}
    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **values)
        write(args.output, report)
        print(stage, flush=True)
    emit("identity")
    current_results = {}
    phase = "bootstrap"
    try:
        tree = load(args.tree_owner_file, "original_dual_slice_tree_owner")
        profile, reference, bundle, digest, admitted = tree._read_identity(args)
        import jax
        import jax.numpy as jnp
        from jax._src import compiler
        from jax.experimental.serialize_executable import deserialize_and_load
        # Resolve the freshly registered canonical overlay from sys.modules;
        # a package attribute may still point at the older support import.
        learning = importlib.import_module("scripts.hpc.run_shared_learning")
        from scripts.hpc.checked_recurrent_execution import device_helper_cache
        from scripts.hpc.shared_daily_microbatch import make_daily_batch_slicer
        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
        if not callable(getattr(learning,"make_learning_daily_accumulator",None)):
            raise ValueError("canonical learning overlay lacks shared daily wrapper")
        for function,positional,keywords in (
            (learning.make_learning_daily_accumulator,(),dict(jax=jax,jnp=jnp,microbatch_size=4,
                evaluator=lambda:None,expected_count=32,slice_execution="tree",batch_slicer=lambda:None)),
            (learning.execute_learning_update,(),dict(jax=jax,parameters=None,optimizer=None,
                daily_samples=None,seven_case=None,thirty_case=None,evaluators={},operations={},
                update=lambda:None,daily_accumulator=lambda:None)),
            (learning.make_tree_operations,(jax,jnp,32),{}),
            (learning.load_learning_checkpoint,(Path(bundle["initial_checkpoint"]),),dict(expected_method=reference["method"])),
            (make_daily_batch_slicer,(jax,4),{}),
            (make_update,(reference["method"]["daily_protocol"],),{}),
            (tree._compare,(jax,None,None),{}),
            (tree._passed,(jax,None,None,None),{}),
            (tree._timing_record,(None,),{}),
        ):
            inspect.signature(function).bind(*positional,**keywords)
        emit("overlay_entry_signatures_verified",learning_overlay=learning.__file__)
        reports, paths = {}, {}
        for name in EXPECTED_BINARIES:
            key = name.lower()
            paths[name] = getattr(args, f"{key}_reference_executable")
            reports[name] = admit_original(getattr(args, f"{key}_reference_report"), paths[name], name,
                                           reference=reference, bundle=bundle, digest=digest)
        preflight = json.loads(args.runtime_preflight.read_text())
        if (preflight.get("status") != "passed" or preflight.get("passed") is not True
                or preflight.get("actual_backend") != "gpu" or preflight.get("jax_version") != jax.__version__):
            raise ValueError("requires compatible completed actual-runtime preflight")
        device = jax.local_devices()[0]
        if len(jax.local_devices()) != 1 or jax.__version__ != "0.4.38" or device.platform != "gpu":
            raise ValueError("requires one canonical JAX0.4.38 GPU")
        checkpoint = learning.load_learning_checkpoint(Path(bundle["initial_checkpoint"]), expected_method=reference["method"])
        if checkpoint["completed_updates"] != 0:
            raise ValueError("requires declared fresh update0")
        parameters, optimizer = jax.device_put((checkpoint["parameters"], checkpoint["optimizer"]))
        daily = jax.device_put(bundle["payloads"][0]["daily"])
        cases = {h: jax.device_put(bundle["payloads"][0]["cases"][h]) for h in (7,30)}
        jax.block_until_ready((parameters, optimizer, daily, cases))
        leaves = jax.tree_util.tree_leaves(daily)
        if len(jax.tree_util.tree_leaves(parameters)) != 300 or sum(np.size(x) for x in jax.tree_util.tree_leaves(parameters)) != 64884481:
            raise ValueError("requires full300/64884481 parameter scope")
        if any(np.shape(x)[0] != 32 for x in leaves):
            raise ValueError("requires32 daily rows")
        with jax.default_device(device), device_helper_cache(device):
            operations = dict(learning.make_tree_operations(jax,jnp,32))
            slicer = make_daily_batch_slicer(jax,4).lower(daily,jnp.asarray(0,dtype=jnp.int32)).compile()
            sample4 = jax.block_until_ready(slicer(daily,jnp.asarray(0,dtype=jnp.int32)))
        # Compare every actual sliced leaf and record distinct starts, outside
        # timed updates. Neither route can silently reuse row0 or a microbatch.
        slice_checks = []
        for start in range(0,32,4):
            leaf = jax.tree_util.tree_map(lambda x: x[start:start+4],daily)
            whole = slicer(daily,jnp.asarray(start,dtype=jnp.int32))
            # Prepared masked placeholders can contain NaNs. Byte equality
            # proves exact slicing even for those unchanged placeholders.
            equal = all(np.asarray(a).dtype == np.asarray(b).dtype
                        and np.asarray(a).shape == np.asarray(b).shape
                        and np.asarray(a).tobytes() == np.asarray(b).tobytes()
                        for a,b in zip(jax.tree_util.tree_leaves(leaf),jax.tree_util.tree_leaves(whole),strict=True))
            if not equal:
                raise ValueError("leaf/tree prepared microbatch differs")
            slice_checks.append({"start":start,"stop":start+4,"leaves":len(leaves),"all_leaves_exact_equal":equal})
        model_counts = {"compile_or_get_cached_calls":0,"backend_compile_calls":0}
        timed_counts = {"compile_or_get_cached_calls":0,"backend_compile_calls":0}
        @contextmanager
        def no_model_compile():
            restore = tree.reject_compilation_before_entry(compiler,model_counts)
            try:
                yield
            finally:
                restore()
        executables, binaries = {}, {}
        for name,path in paths.items():
            load_began = time.perf_counter()
            with path.open("rb") as stream:
                serialized = pickle.load(stream)
            read_seconds = time.perf_counter()-load_began
            deserialize_began = time.perf_counter()
            with no_model_compile():
                binary = deserialize_and_load(*serialized)
            deserialize_seconds = time.perf_counter()-deserialize_began
            arguments = (parameters,sample4.inputs.continuous_state,sample4) if name=="daily" else (parameters,*cases[int(name[1:])])
            validate_inputs(jax,serialized,binary,arguments)
            memory = learning.memory_sizes(binary)
            if memory["estimated_live_bytes"] > 24*2**30:
                raise MemoryError("original executable exceeds24GiB")
            executables[name] = binary
            binaries[name] = {"path":str(path),"memory":memory,"loaded_once_shared_by_both_routes":True,
                              "pickle_read_seconds":read_seconds,"deserialize_seconds":deserialize_seconds,
                              "load_and_abi_check_seconds":time.perf_counter()-load_began}
            del serialized
            emit("original_binary_loaded",loaded_binaries=binaries)
        daily_states = {route:[] for route in ("leaf","tree")}
        def model(name,*values):
            with no_model_compile():
                return jax.block_until_ready(executables[name](*values))
        def daily_evaluator(route):
            def evaluate(p,samples):
                result = model("daily",p,samples.inputs.continuous_state,samples)
                daily_states[route].append(result[1][1])
                return result[0],result[1][0]
            return evaluate
        accumulators = {route:learning.make_learning_daily_accumulator(jax=jax,jnp=jnp,microbatch_size=4,
            evaluator=daily_evaluator(route),expected_count=32,slice_execution=route,
            batch_slicer=slicer if route=="tree" else None) for route in daily_states}
        update = make_update(reference["method"]["daily_protocol"])
        def execute(route,p,opt):
            daily_states[route].clear()
            captured, auxiliary, states = {}, {}, {}
            op = dict(operations)
            def combine(day,seven,thirty):
                value = operations["combine_daily_mean"](day,seven,thirty)
                captured["gradient"] = value
                return value
            op["combine_daily_mean"] = combine
            def accumulator(p,samples):
                result = accumulators[route](p,samples)
                auxiliary["daily"] = result[0][1]
                return result
            wrapped = {}
            for h in (7,30):
                name = f"T{h}"
                def recurrent(*values,_name=name):
                    result = model(_name,*values)
                    auxiliary[_name], states[_name] = result[0][1], result[1][1]
                    return result[0],result[1][0]
                wrapped[name] = recurrent
            began = time.perf_counter()
            with jax.default_device(device),device_helper_cache(device):
                new_p,new_opt,row = learning.execute_learning_update(jax=jax,parameters=p,optimizer=opt,
                    daily_samples=daily,seven_case=cases[7],thirty_case=cases[30],evaluators=wrapped,
                    operations=op,update=update,daily_accumulator=accumulator)
                if len(daily_states[route]) != 8:
                    raise ValueError("requires eight complete micro4 state derivatives")
                states["daily32_micro4"] = tuple(daily_states[route])
                jax.block_until_ready((captured["gradient"],new_p,new_opt,states))
            seconds = time.perf_counter()-began
            hard = {name:tree._hard_counts(value) for name,value in auxiliary.items()}
            memory = device.memory_stats() or {}
            return {"gradient":captured["gradient"],"parameters":new_p,"optimizer":new_opt,
                "update":row,"auxiliary":auxiliary,"initial_state_gradients":states,
                "wall_seconds":seconds,"hard":hard,"all_hard_counts_zero":not any(v for counts in hard.values() for v in counts.values()),
                "device_memory":memory,"memory_gate_passed":memory.get("peak_bytes_in_use",24*2**30+1)<=24*2**30}
        def compare(a,b):
            checks = tree._compare(jax,a,b)
            passed = tree._passed(jax,checks,a,b)
            checks["initial_state_gradients"] = state_comparison(jax,a["initial_state_gradients"],b["initial_state_gradients"])
            finite = all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves((a,b)))
            hard_passed = finite and all(x["all_hard_counts_zero"] and x["memory_gate_passed"] for x in (a,b))
            return checks, passed and hard_passed and all(x["passed"] for x in checks["initial_state_gradients"].values()), hard_passed
        def preserve(results,label,*,full):
            artifacts = {}
            for route,result in results.items():
                state_path = args.output.with_suffix(f".{label}.{route}.states.pkl")
                save(state_path,{"scope":report["scope"],"state_gradients_enter_optimizer":False,"initial_state_gradients":result["initial_state_gradients"],"timing":tree._timing_record(result)})
                artifacts[route] = {"states":str(state_path)}
                if full:
                    raw_path = args.output.with_suffix(f".{label}.{route}.proposal.pkl")
                    save(raw_path,{"scope":"discarded_diagnostic_proposal_not_training_checkpoint","result":result})
                    artifacts[route]["full_proposal"] = str(raw_path)
            report["result_artifacts"].append({"phase":label,"routes":artifacts})
        emit("bootstrap_helpers",source_head=reference["source_git_head"],source_profile_sha256=digest,
             method=reference["method"],saved_binaries=binaries,microbatch_slice_checks=slice_checks,
             daily_rows=32,daily_microbatch_size=4,recurrent_horizons=[7,30],
             source_files={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        for route in ("leaf","tree"):
            current_results[route] = jax.device_get(execute(route,parameters,optimizer))
            emit("bootstrap_route_completed",bootstrap_routes={name:tree._timing_record(value) for name,value in current_results.items()})
        checks,passed,hard_passed = compare(current_results["tree"],current_results["leaf"])
        preserve(current_results,"bootstrap",full=not passed)
        report.update(bootstrap_comparisons=checks,bootstrap_passed=passed,bootstrap_seconds=time.perf_counter()-started)
        warm_states = {route:value["initial_state_gradients"] for route,value in current_results.items()}
        all_passed = passed
        if not hard_passed:
            raise ValueError("bootstrap finite/physical/memory gate failed")
        current_results = {}
        current = {route:(parameters,optimizer) for route in ("leaf","tree")}
        timings = {route:[] for route in current}
        for repeat in range(3):
            phase = f"update{repeat+1}"
            order = ("leaf","tree") if repeat%2==0 else ("tree","leaf")
            restore = tree.reject_compilation_before_entry(compiler,timed_counts)
            try:
                for route in order:
                    value = execute(route,*current[route])
                    current[route] = value["parameters"],value["optimizer"]
                    current_results[route] = jax.device_get(value)
                    timings[route].append(value["wall_seconds"])
            finally:
                restore()
            checks,passed,hard_passed = compare(current_results["tree"],current_results["leaf"])
            steps = {route:int(np.asarray(value["optimizer"].step)) for route,value in current_results.items()}
            passed = passed and all(step==repeat+1 for step in steps.values()) and not any(timed_counts.values()) and not any(model_counts.values())
            self_states = {route:state_comparison(jax,current_results[route]["initial_state_gradients"],warm_states[route]) for route in current} if repeat==0 else None
            preserve(current_results,phase,full=not passed)
            report["timings"].append({"repeat":repeat,"order":order,"steps":steps,"passed":passed,
                "input_relation":"same_update0_parameters_optimizer" if repeat==0 else "separate_parameter_optimizer_trajectories",
                "comparisons":checks,"update0_within_arm_state_self_replay":self_states,
                "routes":{route:tree._timing_record(value) for route,value in current_results.items()}})
            all_passed &= passed
            emit("complete_update_pair_saved",model_compile_counts=model_counts,timed_compile_counts=timed_counts)
            if not hard_passed:
                raise ValueError("finite/physical/memory gate failed; proposal preserved")
            current_results = {}
        emit("completed",status="passed" if all_passed else "strict_parity_failed",completed_pairs=3,
             matched_medians_seconds={name:float(np.median(values)) for name,values in timings.items()},
             leaf_over_tree_median_ratio=float(np.median(timings["leaf"])/np.median(timings["tree"])),
             model_compile_counts=model_counts,timed_compile_counts=timed_counts,device_memory=device.memory_stats())
        return 0 if all_passed else 1
    except BaseException as error:
        if current_results and "preserve" in locals():
            preserve(current_results,phase+"-exception",full=True)
        emit("exception",status="runtime_or_hard_failure",error_type=type(error).__name__,error=str(error),failure_phase=phase)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tree-owner-file","slice-helper-file","support-owner-file","learning-runner-file",
                 "source-profile","input-bundle","daily-admission-report","seven-admission-report","thirty-admission-report",
                 "runtime-preflight","daily-reference-report","daily-reference-executable","t7-reference-report",
                 "t7-reference-executable","t30-reference-report","t30-reference-executable","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    args.inspect_only = False
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
