"""Original T7 forward-mode state basis derivative; one dynamic tangent graph.

This diagnoses the saved original reverse reference. No candidate execution,
precision/loss/policy change, physical perturbation or training update occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
import numpy as np
from probe_shared_command_buffer_loops import comparison, deadline, forbid_component_compile, write

COORDINATES=(33,199,428,200)


def host_scalars(tree):
    if isinstance(tree,dict):
        return {key:host_scalars(value) for key,value in tree.items()}
    value=np.asarray(tree)
    return value.item() if value.shape==() else value.tolist()


def validate_result(jax,result,baseline):
    (loss,aux),derivative=result
    hard={key:int(np.asarray(value)) for key,value in aux["hard"].items()}
    if len(hard)!=14 or any(hard.values()):
        raise ValueError("original JVP physical hard gate failed")
    if np.shape(loss)!=() or np.shape(derivative)!=() or np.asarray(loss).dtype!=np.float64 or np.asarray(derivative).dtype!=np.float64:
        raise ValueError("original JVP scalar ABI/physical FP64 differs")
    if not all(np.isfinite(value).all() for value in jax.tree_util.tree_leaves(result)):
        raise ValueError("original JVP nonfinite result")
    bridge=comparison(jax,result[0],baseline[0])
    if not bridge["passed"]:
        raise ValueError("public loss/aux strict bridge failed")
    return dict(hard_counts=hard,loss_auxiliary_bridge=bridge,finite=True,
                declared_mask_change=int(np.asarray(aux["declared_mask_change"])))


def scalar_metric(value,reference):
    absolute=abs(value-reference)
    allowance=1e-6+1e-5*abs(reference)
    return dict(value=value,reference=reference,absolute_error=absolute,strict_allowance=allowance,
                passed=bool(np.isfinite(value) and np.isfinite(reference) and absolute<=allowance),
                relative_error=absolute/max(abs(reference),np.finfo(np.float64).tiny))


def run(args,emit,report):
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax,devices=initialize_gpu_backend()
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load, serialize
    import probe_shared_forward_reverse as forward
    from probe_shared_normalized_keys_recurrent import validate_inputs
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.run_shared_transformer_admission import memory_sizes
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
    from research.daily_coarse_graining.daily_operator import shared_process_head
    if jax.__version__!="0.4.38" or len(devices)!=1:
        raise ValueError("requires one canonical JAX0.4.38 GPU")
    preflight=json.loads(args.runtime_preflight.read_text())
    if preflight.get("status")!="passed" or not preflight.get("passed") or preflight.get("jax_version")!=jax.__version__ or preflight.get("actual_backend")!="gpu":
        raise ValueError("new actual-environment runtime preflight failed/differs")
    # Tiny actual-node native checkpointed scan JVP/VJP smoke. No real model.
    def synthetic(p,state):
        def step(s,day):
            next_state=jnp.tanh(p*s+day)
            return next_state,next_state
        _,trajectory=jax.lax.scan(jax.checkpoint(step),state,jnp.asarray([.1,.2,.3],jnp.float64))
        return jnp.sum(trajectory,dtype=jnp.float64)
    synthetic_params=jax.device_put(np.asarray([.7,.8],np.float64))
    synthetic_state=jax.device_put(np.asarray([.1,.2],np.float64))
    synthetic_direction=jax.device_put(np.asarray([1.,0.],np.float64))
    primal,tangent=jax.jvp(lambda x:synthetic(synthetic_params,x),(synthetic_state,),(synthetic_direction,))
    gradient=jax.grad(lambda x:synthetic(synthetic_params,x))(synthetic_state)
    smoke=scalar_metric(float(jax.device_get(tangent)),float(np.dot(jax.device_get(gradient),np.asarray([1.,0.]))))
    if not smoke["passed"]:
        raise ValueError("synthetic actual JVP/VJP runtime smoke failed")
    emit("native_jvp_smoke_passed",synthetic_smoke=smoke,runtime_preflight=str(args.runtime_preflight),jax_version=jax.__version__)
    root=Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
    diag=root/"runtime/outputs/diagnostics"
    source=SimpleNamespace(source_profile=diag/"shared-split-profile-c599236.json",
        input_bundle=diag/"shared-split-dual-inputs-c599236.pkl",
        prepared_context=diag/"shared-expert-real-daily-context-c599236.pkl",
        daily_admission_report=diag/"shared-combined-real-daily-20260925.json",
        seven_admission_report=diag/"shared-combined-recurrent-T7-20260926.json",
        thirty_admission_report=diag/"shared-combined-recurrent-T30-20260926.json",
        support_owner_file=root/"runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py")
    profile,ref,bundle,context,checkpoint_path,admissions=forward._assets(source)
    checkpoint=load_learning_checkpoint(checkpoint_path,expected_method=ref["method"])
    if checkpoint["completed_updates"]!=0:
        raise ValueError("requires update0 original parameters")
    parameters=jax.device_put(checkpoint["parameters"])
    if sum(np.size(x) for x in jax.tree_util.tree_leaves(parameters))!=64884481 or len(jax.tree_util.tree_leaves(parameters))!=300:
        raise ValueError("full model parameter scope differs")
    case=bundle["payloads"][0]["cases"][7]
    initial=np.asarray(case[0])
    if initial.shape!=(3854,) or initial.dtype!=np.float64 or len(case)!=5 or np.shape(case.days[0].continuous_state)[0]!=7:
        raise ValueError("original initialstate/T7 ABI differs")
    arguments=(parameters,*jax.device_put(case))
    directions={coordinate:np.eye(1,3854,coordinate,dtype=np.float64)[0] for coordinate in COORDINATES}
    device_directions={coordinate:jax.device_put(value) for coordinate,value in directions.items()}
    stem=diag/"shared-recurrent-checkpoint-policy-20260930-v2-T7"
    accepted=json.loads(Path(str(stem)+"-reference.json").read_text())
    decoder=root/"runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
    backbone=root/"runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
    identity=dict(source_head=ref["source_git_head"],source_profile_sha256=hashlib.sha256(source.source_profile.read_bytes()).hexdigest(),
        method=ref["method"],artifact_id=int(bundle["artifact_ids"][0]),horizon=7,checkpoint=str(checkpoint_path.resolve()),
        jax_version=jax.__version__,admitted_executable=admissions["T7"]["executable"],
        candidate_decoder_sha256=hashlib.sha256(decoder.read_bytes()).hexdigest(),candidate_backbone_sha256=hashlib.sha256(backbone.read_bytes()).hexdigest())
    if accepted.get("status")!="passed" or accepted.get("identity")!=identity or accepted.get("mode")!="reference":
        raise ValueError("original dual reference identity differs")
    with Path(str(stem)+".reference.pkl").open("rb") as stream:
        saved=pickle.load(stream)
    if saved["identity"]!=identity:
        raise ValueError("saved original dual result identity differs")
    with forbid_component_compile(True):
        with Path(str(stem)+".reference.executable.pkl").open("rb") as stream:
            serialized=pickle.load(stream)
        original=deserialize_and_load(*serialized)
        validate_inputs(jax,serialized,original,arguments)
        del serialized
        initial_original=jax.device_get(jax.block_until_ready(original(*arguments)))
    bridges={"loss_and_auxiliary":comparison(jax,initial_original[0],saved["result"][0]),
             "all_parameter_gradients":comparison(jax,initial_original[1][0],saved["result"][1][0]),
             "all_initial_state_gradients":comparison(jax,initial_original[1][1],saved["result"][1][1])}
    if not bridges["loss_and_auxiliary"]["passed"] or not bridges["all_parameter_gradients"]["passed"]:
        raise ValueError("fresh original dual loss/parameter bridge failed")
    if not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(initial_original)):
        raise ValueError("fresh original dual nonfinite")
    baseline=(initial_original[0],np.asarray(initial_original[1][1]))
    saved_state_gradient=np.asarray(saved["result"][1][1]).copy()
    del saved,initial_original
    baselines=[baseline]
    write(args.output.with_suffix(".original-dual-baseline.pkl"),dict(identity=identity,loss_aux=baseline[0],state_gradient=baseline[1]))
    historical={name:[] for name in ("default","candidate")}
    v4stem=diag/"shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4"
    for name in historical:
        for repeat in range(7):
            with Path(str(v4stem)+f".{name}.repeat{repeat}.initial_state_vjp.pkl").open("rb") as stream:
                value=np.asarray(pickle.load(stream))
            if value.shape!=(3854,) or value.dtype!=np.float64:
                raise ValueError("historical state vector ABI differs")
            historical[name].append(value)
        historical[name]=np.stack(historical[name])
    defined=np.isfinite(initial)&(np.abs(initial)<0.5e20)
    coordinate_info=[]
    leaves=[leaf for leaf in context["head"].contract.state_leaves if not leaf.discrete]
    for coordinate in COORDINATES:
        leaf=next(leaf for leaf in leaves if leaf.start<=coordinate<leaf.stop)
        if not defined[coordinate]:
            raise ValueError("basis coordinate undefined")
        coordinate_info.append(dict(coordinate=coordinate,key=leaf.key,shape=list(leaf.shape),
            index=list(np.unravel_index(coordinate-int(leaf.start),leaf.shape)),initial_value=float(initial[coordinate]),
            baseline_vjp=float(baseline[1][coordinate]),saved_vjp=float(saved_state_gradient[coordinate]),
            historical_original=historical["default"][:,coordinate].tolist(),historical_candidate=historical["candidate"][:,coordinate].tolist()))
    emit("original_dual_reference_replayed",identity=identity,original_dual_calls=1,original_bridge=bridges,coordinates=coordinate_info,
         native_policy="unchanged_default_checkpoint",precision="FP32_neural_FP64_physical",
         mask_provenance="input_assembly._defined52–54 finite and abs<0.5e20; no physical perturbation")
    owner=forward._load_module(root/"runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py","scripts.hpc.probe_shared_expert_real_daily")
    adapter=owner.load_staged_adapter(decoder)
    backbone_adapter=owner.load_backbone_assembler(backbone)
    _rollout,objective=make_shared_recurrent_objective(context["head"],context["template"],context["state_scales"],context["science_scales"])
    def directional(parameters,state,discrete,days,targets,loss_context,direction):
        primal,tangent,aux=jax.jvp(lambda x:objective(parameters,x,discrete,days,targets,loss_context),
                                  (state,),(direction,),has_aux=True)
        return (primal,aux),tangent
    jvp_args=(*arguments,device_directions[COORDINATES[0]])
    emit("lowering",dynamic_tangent_shape=[3854],dynamic_tangent_dtype="float64",model_compiles=0)
    began=time.perf_counter()
    with owner.candidate_process_head_patches(shared_process_head,adapter,backbone_adapter),deadline(240,"lower",emit):
        lowered=jax.jit(directional).lower(*jvp_args)
    lower_seconds=time.perf_counter()-began
    stable=str(lowered.compiler_ir(dialect="stablehlo"))
    args.output.with_suffix(".lowered.stablehlo.txt").write_text(stable)
    argument_infos=[dict(shape=list(info._aval.shape),dtype=str(info._aval.dtype)) for info in jax.tree_util.tree_leaves(lowered.args_info)]
    if argument_infos[-1]!={"shape":[3854],"dtype":"float64"}:
        raise ValueError("lowered dynamic direction ABI differs")
    emit("compiling",lower_seconds=lower_seconds,argument_shapes_dtypes=argument_infos,
         stablehlo_optimization_barriers=stable.count("stablehlo.optimization_barrier"),model_compiles=1,
         compiler_options={"xla_gpu_enable_llvm_module_compilation_parallelism":False})
    began=time.perf_counter()
    with deadline(900,"backend",emit):
        executable=lowered.compile(compiler_options={"xla_gpu_enable_llvm_module_compilation_parallelism":False})
    backend_seconds=time.perf_counter()-began
    serialized=serialize(executable)
    validate_inputs(jax,serialized,executable,jvp_args)
    write(args.output.with_suffix(".executable.pkl"),serialized)
    del serialized
    memory=memory_sizes(executable)
    if memory["estimated_live_bytes"]>24*2**30:
        raise MemoryError("JVP estimated live exceeds24GiB")
    emit("compiled_and_saved",backend_seconds=backend_seconds,compiled_memory=memory,executable=str(args.output.with_suffix(".executable.pkl")),dynamic_direction_reuse=True)
    results=[];first={};times=[]
    def evaluate(coordinate,repeat):
        began=time.perf_counter()
        with forbid_component_compile(True):
            value=jax.device_get(jax.block_until_ready(executable(*arguments,device_directions[coordinate])))
        duration=time.perf_counter()-began
        hard=validate_result(jax,value,baseline)
        derivative=float(value[1])
        record=dict(coordinate=coordinate,repeat=repeat,loss_aux=host_scalars(value[0][1]),loss=float(value[0][0]),
                    jvp=derivative,seconds=duration,validation=hard,
                    vs_fresh_original_vjp=scalar_metric(derivative,float(np.dot(baseline[1],directions[coordinate]))),
                    vs_saved_original_vjp=scalar_metric(derivative,float(np.dot(saved_state_gradient,directions[coordinate]))))
        if coordinate in first:
            record["vs_jvp_self"]=scalar_metric(derivative,first[coordinate])
        else:
            first[coordinate]=derivative
        results.append(record);times.append(duration)
        stats=devices[0].memory_stats() or {}
        if "peak_bytes_in_use" not in stats or stats["peak_bytes_in_use"]>24*2**30:
            raise MemoryError("JVP observed active peak unavailable/exceeds24GiB")
        write(args.output.with_suffix(".directional-results.pkl"),dict(identity=identity,rows=results,coordinates=coordinate_info))
        emit("directional_result_saved",jvp_calls=len(results),latest=record,device_memory=stats)
        return record
    for coordinate in COORDINATES:
        evaluate(coordinate,-1)
    emit("initial_comparisons_saved",initial_results=results[:4])
    for repeat in range(7):
        for coordinate in (COORDINATES if repeat%2==0 else tuple(reversed(COORDINATES))):
            evaluate(coordinate,repeat)
        emit("repeat_completed",repeat=repeat,jvp_calls=len(results))
    with forbid_component_compile(True):
        end=jax.device_get(jax.block_until_ready(original(*arguments)))
    end_baseline=(end[0],np.asarray(end[1][1]))
    if not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(end)):
        raise ValueError("end original dual nonfinite")
    write(args.output.with_suffix(".original-dual-end.pkl"),dict(identity=identity,loss_aux=end_baseline[0],state_gradient=end_baseline[1]))
    end_checks={"loss_aux":comparison(jax,end_baseline[0],baseline[0]),"all_initial_state_gradients":comparison(jax,end_baseline[1],baseline[1])}
    del end
    for record in results:
        record["vs_end_original_vjp"]=scalar_metric(record["jvp"],float(end_baseline[1][record["coordinate"]]))
    summary={}
    for coordinate in COORDINATES:
        rows=[row for row in results if row["coordinate"]==coordinate]
        summary[str(coordinate)]=dict(fresh_vjp=float(baseline[1][coordinate]),end_vjp=float(end_baseline[1][coordinate]),
            jvp_range=[min(r["jvp"] for r in rows),max(r["jvp"] for r in rows)],
            native_jvp_vjp_all_strict_pass=all(r["vs_fresh_original_vjp"]["passed"] and r["vs_end_original_vjp"]["passed"] for r in rows),
            strict_failures_against_fresh=sum(not r["vs_fresh_original_vjp"]["passed"] for r in rows),
            strict_failures_against_end=sum(not r["vs_end_original_vjp"]["passed"] for r in rows),
            self_strict_failures=sum(not r.get("vs_jvp_self",{"passed":True})["passed"] for r in rows),
            jvp_median_seconds=float(np.median([r["seconds"] for r in rows if r["repeat"]>=0])),
            candidate_derivative_not_certified=True)
    passed=all(row["native_jvp_vjp_all_strict_pass"] for row in summary.values())
    emit("completed",status="native_jvp_vjp_consistent" if passed else "native_jvp_vjp_discrepancy",
         original_dual_calls=2,jvp_calls=len(results),summary=summary,all_results=results,end_original_self_comparisons=end_checks,
         physics_checked=True,source_masks_unchanged=True,training_throughput_claim=False,default_checkpoint_policy_unchanged=True,
         device_memory=devices[0].memory_stats(),model_compiles=1,candidate_executions=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-preflight",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve completed result")
    started=time.perf_counter()
    report=dict(status="running",scope="original_public_T7_native_dynamic_state_basis_JVP_reference",
                coordinates=list(COORDINATES),gpu_count=1,paid_jobs=0,training_updates=0,candidate_executions=0,
                model_compiles=0,budgets_seconds=dict(lower=240,backend=900,process=1500),rtol=1e-5,atol=1e-6)
    def emit(stage,**fields):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-started,**fields)
        write(args.output,report)
        print(stage,flush=True)
    emit("starting")
    try:
        with deadline(1485,"whole_process",emit):
            run(args,emit,report)
    except BaseException as error:
        emit("exception",status="error",error_type=type(error).__name__,error=str(error))
        raise


if __name__=="__main__":
    main()
