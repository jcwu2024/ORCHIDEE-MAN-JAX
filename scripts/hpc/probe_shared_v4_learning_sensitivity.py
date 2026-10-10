"""Prepared, unapproved three-arm learning sensitivity diagnostic.

Dry-run and protocol-check perform no model calls or optimizer updates.
Execution is reserved for a later explicit scientific approval.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import threading
import time

ARMS = ("reference_a", "candidate", "reference_b")
STEPS = 16
MILESTONES = (0, 4, 8, 16)
SAVED_STEPS = (1, 4, 8, 16)
ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
FAMILY = ROOT / "runtime/outputs/diagnostics/shared-v4-learning-sensitivity-20261004"
INITIAL = ROOT / "runtime/outputs/diagnostics/shared-v4-same-input-update-20261003/model/pair-1-reference.full-proposal.pkl"


def plan():
    return dict(status="prepared_unapproved_learning_diagnostic", arms=list(ARMS),
        steps_per_arm=STEPS, optimizer_updates=48, initial_optimizer_step=2,
        initial=str(INITIAL), initialization="identical saved diagnostic parameters and AdamW state",
        workload="fixed artifact742 daily32/eight distinct micro4 + public T7 + public T30",
        independent_trajectories=True, original_evaluator_milestones=list(MILESTONES),
        initial_monitor_shared_once=True, planned_model_calls=580,
        model_AD_compiles=0, precision="original neural FP32/physical FP64",
        strict_rtol=1e-5, strict_atol=1e-6, acceptance_threshold_changed=False,
        production_adoption=False, formal_training=False,
        free_node="gln01", GPU_count=1, physical_GPU=1, GPU_memory_limit_GiB=24,
        OMP_threads=4, OpenBLAS_threads=1, total_seconds=1800, kill_grace_seconds=10,
        max_card_hours=.5, paid_jobs=0, CNY=0, output=str(FAMILY),
        required_free_disk_GiB=20, snapshot_steps=list(SAVED_STEPS),
        learning_execution_requires_new_explicit_scientific_approval=True,
        limitations=["one diagnostic initialization and fixed data, not generalization",
            "16 updates and at most 30-day targets do not establish long-horizon training safety",
            "reference repeat variation does not waive strict parity; no automatic admission"])


def protocol(initial, execute, monitor, capture, record):
    """Separate trajectories; common original evaluator; no extra warm calls."""
    current = {arm: initial for arm in ARMS}
    first = monitor("shared_initial", initial, 0)
    record("monitor", 0, {arm: first for arm in ARMS})
    for step in range(1, STEPS + 1):
        offset = (step - 1) % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        for arm in order:
            current[arm], evidence = execute(arm, current[arm], step)
            if step in SAVED_STEPS:
                capture(arm, current[arm], evidence, step)
            record("update", step, dict(arm=arm, evidence=evidence))
        if step in MILESTONES:
            record("monitor", step, {arm: monitor(arm, current[arm], step) for arm in ARMS})
    return current


def protocol_check():
    calls, monitors, snapshots = [], [], []
    def execute(arm, state, step):
        calls.append((arm, state, step))
        return state + (2 if arm == "candidate" else 1), dict(fake=True)
    final = protocol(0, execute, lambda arm, state, step: monitors.append((arm, state, step)),
        lambda arm, state, evidence, step: snapshots.append((arm, step)), lambda *args: None)
    if len(calls) != 48 or len(monitors) != 10 or len(snapshots) != 12:
        raise ValueError("bounded three-arm protocol counts differ")
    if final != dict(reference_a=16, candidate=32, reference_b=16):
        raise ValueError("independent trajectories were merged")
    return dict(status="passed", protocol_calls=48, monitor_calls=10,
        snapshots=12, model_calls=0, optimizer_updates=0, GPU_calls=0, Teacher_reads=0)


def write_json(path, value):
    def scalar(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        raise TypeError("unsupported report value: " + type(x).__name__)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=scalar, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def monitored_effect(monitors):
    """Descriptive learning differences; no new acceptance threshold."""
    rows = []
    initial = monitors[0]["arms"]["reference_a"]["metrics"]
    for point in monitors[1:]:
        arms = point["arms"]
        for key in initial:
            a = arms["reference_a"]["metrics"][key]
            b = arms["reference_b"]["metrics"][key]
            c = arms["candidate"]["metrics"][key]
            rows.append(dict(step=point["step"], metric=key, initial=initial[key],
                reference_a=a, reference_b=b, candidate=c,
                reference_repeat_difference=b-a, candidate_minus_reference_a=c-a,
                candidate_minus_reference_b=c-b,
                reference_a_change=a-initial[key], reference_b_change=b-initial[key],
                candidate_change=c-initial[key]))
    return dict(status="descriptive_not_admission", rows=rows,
        new_scientific_thresholds=False, reference_drift_waives_strict_parity=False,
        generalization_measured=False, production_adoption=False)


def metadata_check(args):
    """Read saved identities on CPU; never evaluate models or optimizers."""
    import probe_shared_v4_same_input_update as base
    from probe_shared_normalized_keys_recurrent import load
    for name, value in base.defaults().items(): setattr(args, name, value)
    result = base.metadata_check(args)
    tree = load(args.tree_owner_file, "sensitivity_metadata_owner")
    _, reference, _, _, _ = tree._read_identity(args)
    with INITIAL.open("rb") as stream: initial = pickle.load(stream)
    if (initial["source_head"] != reference["source_git_head"] or initial["method"] != reference["method"]
            or initial["artifact_id"] != 742 or initial["result"]["pair"] != 1
            or initial["result"]["arm"] != "reference" or int(initial["result"]["optimizer"].step) != 2):
        raise ValueError("declared learning diagnostic initialization differs")
    result.update(initial=str(INITIAL), initial_optimizer_step=2,
        learning_model_calls=0, optimizer_updates=0, scientific_approval=False)
    return result


def execute_approved(args):
    # This function must not be called before the human scientific approval.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(plan(), status="running_approved_diagnostic", model_calls=0,
        model_calls_started=0, completed_optimizer_updates=0, updates=[], monitors=[])
    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **values)
        write_json(args.output_dir/"model.json", report)
    timer = threading.Timer(args.process_seconds, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    try:
        emit("entry_before_runtime_imports")
        import numpy as np
        import probe_shared_v4_same_input_update as base
        from probe_shared_normalized_keys_recurrent import load, validate_inputs
        import jax
        import jax.numpy as jnp
        from jax._src import compiler
        from jax.experimental.serialize_executable import deserialize_and_load
        from scripts.hpc.checked_recurrent_execution import device_helper_cache
        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
        from analyze_shared_v4_ab_offline import compare

        for name, value in base.defaults().items(): setattr(args, name, value)
        args.inspect_only = False
        original = load(args.original_owner_file, "sensitivity_original_owner")
        tree = load(args.tree_owner_file, "sensitivity_tree_owner")
        _, reference, bundle, digest, _ = tree._read_identity(args)
        learning, interface = base.learning_interface(args)
        base.candidate_metadata(args, reference, bundle, digest)
        for name in ("daily", "T7", "T30"):
            original.admit_original(getattr(args,name.lower()+"_reference_report"),
                getattr(args,name.lower()+"_reference_executable"),name,
                reference=reference,bundle=bundle,digest=digest)
        if jax.__version__ != "0.4.38" or len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu":
            raise ValueError("canonical single-GPU JAX0.4.38 required")
        preflight = json.loads(args.runtime_preflight.read_text())
        if (preflight.get("status") != "passed" or not preflight.get("passed")
                or preflight.get("actual_backend") != "gpu" or preflight.get("jax_version") != "0.4.38"):
            raise ValueError("unchanged GPU runtime preflight missing")
        with INITIAL.open("rb") as stream: initial_raw = pickle.load(stream)
        if (initial_raw["source_head"] != reference["source_git_head"] or initial_raw["method"] != reference["method"]
                or initial_raw["artifact_id"] != 742 or initial_raw["result"]["pair"] != 1
                or initial_raw["result"]["arm"] != "reference" or int(initial_raw["result"]["optimizer"].step) != 2):
            raise ValueError("declared diagnostic initialization differs")
        initial = jax.device_put((initial_raw["result"]["parameters"], initial_raw["result"]["optimizer"]))
        del initial_raw
        device = jax.local_devices()[0]
        daily = jax.device_put(bundle["payloads"][0]["daily"])
        cases = jax.device_put(bundle["payloads"][0]["cases"])
        if any(np.shape(x)[0] != 32 for x in jax.tree_util.tree_leaves(daily)):
            raise ValueError("all original32 prepared daily rows required")
        input_states = dict(daily=np.asarray(daily.inputs.continuous_state),
            **{f"T{h}":np.asarray(cases[h][0]) for h in (7,30)})
        defined = {name:np.isfinite(value)&(np.abs(value)<.5e20) for name,value in input_states.items()}
        counts = dict(compile_or_get_cached_calls=0, backend_compile_calls=0)
        @contextmanager
        def no_model_compile():
            restore = tree.reject_compilation_before_entry(compiler, counts)
            try: yield
            finally: restore()
        binaries = {}
        sample4 = jax.tree_util.tree_map(lambda x:x[:4], daily)
        for arm,name in (("reference","daily"),("reference","T7"),("reference","T30"),("candidate","daily"),("candidate","T7")):
            path = getattr(args,name.lower()+"_"+arm+"_executable")
            with path.open("rb") as stream: serialized = pickle.load(stream)
            with no_model_compile(): binary = deserialize_and_load(*serialized)
            values = (initial[0],sample4.inputs.continuous_state,sample4) if name=="daily" else (initial[0],*cases[int(name[1:])])
            validate_inputs(jax, serialized, binary, values)
            if learning.memory_sizes(binary)["estimated_live_bytes"] > 24*2**30:
                raise MemoryError("binary memory gate")
            binaries[arm,name] = binary
            del serialized
        binaries["candidate","T30"] = binaries["reference","T30"]
        with jax.default_device(device),device_helper_cache(device): operations=learning.make_tree_operations(jax,jnp,32)
        update = make_update(reference["method"]["daily_protocol"])
        context, states, auxiliary, captured = {}, {}, {}, {}

        def model(name, *values):
            arm = context["execution_arm"]
            marker=dict(context, component=name)
            report["model_calls_started"] += 1
            emit("model_call_started", current_model_call=marker)
            with no_model_compile(): result=jax.block_until_ready(binaries[arm,name](*values))
            report["model_calls"] += 1
            captured["last_model_result"]=result
            state=np.asarray(jax.device_get(result[1][1]))
            mask=defined[name]
            if name=="daily":
                start=len(states["daily"])*4
                mask=mask[start:start+4]
                expected=(4,3854)
            else: expected=(3854,)
            hard=tree._hard_counts(result[0][1])
            if (state.shape!=expected or not np.isfinite(state).all() or np.count_nonzero(state[~mask])
                    or any(hard.values()) or len(hard)!=(11 if name=="daily" else 14)
                    or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(result[0]))
                    or not bool(np.asarray(operations["finite"](result[1][0])))):
                base.save(args.output_dir/f"failed-model-call-{report['model_calls']}.pkl",jax.device_get(result))
                raise ValueError("state/physical hard gate after preserved output")
            if name=="daily": states["daily"].append(result[1][1])
            else: states[name]=result[1][1]
            auxiliary[name]=result[0][1]
            return result

        def daily_evaluator(p,samples):
            result=model("daily",p,samples.inputs.continuous_state,samples)
            return result[0],result[1][0]
        accumulator=learning.make_learning_daily_accumulator(jax=jax,jnp=jnp,microbatch_size=4,
            evaluator=daily_evaluator,expected_count=32,slice_execution="leaf")
        def daily_accumulator(p,samples):
            result=accumulator(p,samples)
            auxiliary["daily"]=result[0][1]
            return result
        def recurrent(name):
            def evaluate(*values):
                result=model(name,*values)
                return result[0],result[1][0]
            return evaluate
        evaluators={name:recurrent(name) for name in ("T7","T30")}
        op=dict(operations)
        def combine(d,s,t):
            value=operations["combine_daily_mean"](d,s,t)
            captured["gradient"]=value
            return value
        op["combine_daily_mean"]=combine

        def reset(arm,step,phase):
            context.clear();context.update(logical_arm=arm,step=step,phase=phase,
                execution_arm="candidate" if arm=="candidate" and phase=="update" else "reference")
            states.clear();states["daily"]=[];auxiliary.clear();captured.clear()
        def memory_gate():
            memory=device.memory_stats() or {}
            if memory.get("peak_bytes_in_use",24*2**30+1)>24*2**30 or any(counts.values()):
                raise MemoryError("memory/model-compilation hard gate")
            return memory
        def execute(arm,current,step):
            reset(arm,step,"update")
            began=time.perf_counter()
            def preserved_update(*values):
                result=update(*values)
                captured["optimizer_proposal"]=result
                return result
            try:
                with jax.default_device(device),device_helper_cache(device):
                    p,opt,row=learning.execute_learning_update(jax=jax,parameters=current[0],optimizer=current[1],
                        daily_samples=daily,seven_case=cases[7],thirty_case=cases[30],evaluators=evaluators,
                        operations=op,update=preserved_update,daily_accumulator=daily_accumulator)
                    jax.block_until_ready((p,opt,captured["gradient"],states))
                if len(states["daily"])!=8 or int(opt.step)!=2+step: raise ValueError("workload/step differs")
                evidence=dict(row=row,wall_seconds=time.perf_counter()-began,
                    memory=memory_gate(),initial_state_gradients=states.copy(),gradient=captured["gradient"],
                    auxiliary=auxiliary.copy())
            except BaseException:
                # Official checks may raise before returning. Preserve all available
                # inputs, mixed gradient and actual optimizer proposal before rethrow.
                base.save(args.output_dir/f"failed-update-{step:02d}-{arm}.pkl",jax.device_get(dict(
                    context=context.copy(),input_parameters=current[0],input_optimizer=current[1],
                    captured=captured,initial_state_gradients=states.copy(),auxiliary=auxiliary.copy())))
                raise
            report["completed_optimizer_updates"]+=1
            return (p,opt),evidence
        def monitor(arm,current,step):
            reset(arm,step,"original_evaluator")
            began=time.perf_counter()
            with jax.default_device(device),device_helper_cache(device):
                day=daily_accumulator(current[0],daily)
                seven=model("T7",current[0],*cases[7])
                thirty=model("T30",current[0],*cases[30])
            if len(states["daily"])!=8: raise ValueError("monitor daily workload differs")
            result=jax.device_get(dict(daily_value=day[0][0],auxiliary=auxiliary,
                T7_value=seven[0][0],T30_value=thirty[0][0],initial_state_gradients=states))
            result["joint"]=float(result["daily_value"])+float(result["T7_value"])+float(result["T30_value"])
            result["wall_seconds"]=time.perf_counter()-began
            result["memory"]=device.memory_stats() or {}
            base.save(args.output_dir/f"monitor-{step:02d}-{arm}.pkl",result)
            memory_gate()
            metrics=dict(joint=result["joint"],daily=float(result["daily_value"]),
                T7=float(result["T7_value"]),T30=float(result["T30_value"]))
            for name, aux in result["auxiliary"].items():
                for key in ("process_day", "joint_daily_contribution", "flux", "stock", "living_stock", "nonliving_stock"):
                    if key in aux: metrics[name+"/"+key]=float(aux[key])
            return dict(metrics=metrics,joint=result["joint"],daily=float(result["daily_value"]),T7=float(result["T7_value"]),
                T30=float(result["T30_value"]),physical_passed=True,
                original_evaluator=True,raw_path=str(args.output_dir/f"monitor-{step:02d}-{arm}.pkl"))
        def capture(arm,current,evidence,step):
            raw=jax.device_get(dict(parameters=current[0],optimizer=current[1],**evidence))
            base.save(args.output_dir/f"update-{step:02d}-{arm}.pkl",dict(result=raw,initial=str(INITIAL),
                diagnostic_only=True,source_head=reference["source_git_head"],artifact_id=742))
        def first_strict():
            with (args.output_dir/"update-01-reference_a.pkl").open("rb") as stream:
                expected=pickle.load(stream)["result"]
            strict={}
            for arm in ("candidate","reference_b"):
                with (args.output_dir/f"update-01-{arm}.pkl").open("rb") as stream:
                    actual=pickle.load(stream)["result"]
                strict[arm]={key:compare(actual[key],expected[key]) for key in
                    ("gradient","parameters","optimizer","initial_state_gradients","auxiliary")}
                for key in ("loss", "optimizer_gradient_norm", "clip_scale"):
                    strict[arm][key]=compare(actual["row"][key],expected["row"][key])
            write_json(args.output_dir/"common-input-first-update-strict.json",strict)
        def record(kind,step,value):
            if kind=="update":
                evidence=value["evidence"]
                report["updates"].append(dict(step=step,arm=value["arm"],wall_seconds=evidence["wall_seconds"],
                    row=jax.device_get(evidence["row"])))
                if step==1 and all((args.output_dir/f"update-01-{arm}.pkl").exists() for arm in ARMS):
                    first_strict()
            else:
                report["monitors"].append(dict(step=step,arms=value))
                write_json(args.output_dir/"monitored-learning-effect.json",monitored_effect(report["monitors"]))
            emit("record_saved")
        emit("ready",learning_interface=interface,source_head=reference["source_git_head"])
        protocol(initial,execute,monitor,capture,record)
        if report["model_calls"]!=580 or report["completed_optimizer_updates"]!=48:
            raise ValueError("bounded protocol totals differ")
        emit("completed",status="learning_sensitivity_diagnostic_completed_not_adopted",model_compile_counts=counts)
        return 0
    except BaseException as error:
        emit("exception",status="runtime_or_hard_failure",error_type=type(error).__name__,error=str(error))
        return 1
    finally:timer.cancel()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--protocol-check",action="store_true")
    parser.add_argument("--metadata-check",action="store_true")
    parser.add_argument("--execute-approved-diagnostic",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=FAMILY/"model")
    parser.add_argument("--process-seconds",type=float,default=1790)
    args=parser.parse_args()
    if args.dry_run:print(json.dumps(plan(),indent=2));return 0
    if args.protocol_check:print(json.dumps(protocol_check(),indent=2));return 0
    if args.metadata_check:print(json.dumps(metadata_check(args),indent=2));return 0
    if not args.execute_approved_diagnostic:
        parser.error("learning execution is unapproved; use --dry-run or --protocol-check")
    if not 0<args.process_seconds<=1790:parser.error("process budget exceeds proposed allocation")
    return execute_approved(args)


if __name__=="__main__":raise SystemExit(main())
