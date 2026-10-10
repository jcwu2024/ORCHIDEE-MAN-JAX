"""Independent bounded FD of saved original T7 scalar; no candidate execution.

Uses original default checkpoint dual-AD executable only, at most59 calls.
No forward lowering, gradient construction or tolerance change is permitted.
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


def write(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    if path.suffix==".json":
        temporary.write_text(json.dumps(value,indent=2,default=str)+"\n")
    else:
        with temporary.open("wb") as stream:
            pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary,path)


def host_scalars(tree):
    if isinstance(tree,dict):
        return {k:host_scalars(v) for k,v in tree.items()}
    a=np.asarray(tree)
    return a.item() if a.shape==() else a.tolist()


def analyze_saved(path):
    """Only preserved arrays; physical right side, no new execution/calls."""
    result=json.loads(path.read_text())
    if result.get("status")!="completed_diagnostic" or result["binary_calls"]!=59:
        raise ValueError("requires completed bounded original replay")
    f0=float(np.median([r["loss"] for r in result["baselines"]]))
    provenance={33:"explicitsnow.f90::explicitsnow_initialize56–86: snow grain size(m), zero initialization",
                199:"hydrol.f90::hydrol_canop4992–5094: canopy2ground=MAX(precisol-precip2ground,zero)",
                200:"hydrol.f90::hydrol_canop4992–5094: canopy2ground=MAX(precisol-precip2ground,zero)",
                428:"hydrol.f90::hydrol_soil_froz8332–8426: frozen water fraction=1-x, x in[0,1]"}
    rows=[]
    for info in result["coordinates"]:
        h=np.asarray([p["actual_h_plus"] for p in info["perturbations"]])
        delta=np.asarray([p["plus"]["loss"]-f0 for p in info["perturbations"]])
        right=delta/h
        # Nonuniform right-sided three-point derivative: f(0),f(hsmall),
        # f(hbig). Uses already saved h's, not additional2h calls.
        second=[float((h[i]/h[i+1]*delta[i+1]-h[i+1]/h[i]*delta[i])/(h[i]-h[i+1])) for i in range(len(h)-1)]
        row=dict(coordinate=info["coordinate"],key=info["key"],local_index=info["local_index"],
            initial_value=info["initial_value"],input_scale=info["input_scale"],control=info["control"],
            original_ad_baseline=info["baseline_ad_values"],original_ad_saved=info["original_saved_ad"],
            historical_original_ad=info["historical_original_ad"],historical_candidate_ad=info["historical_candidate_ad"],
            native_strict_allowance=info["strict_reference_allowance"],historical_candidate_gap=info["historical_candidate_gap_from_current_baseline"],
            domain_provenance="fortran_source/ORCHIDEE/src_sechiba/"+provenance[info["coordinate"]],
            physical_domain="nonnegative; initial point on lower boundary",central_minus_domain_valid=False,
            positive_initial_domain_valid=True,internal_branch_invariance_verified=False,
            exposed_hard_counts_valid=all(p["plus"]["physics_valid"] for p in info["perturbations"]),
            declared_mask_counts_match_baseline=all(p["mask_counts_match_baseline"] for p in info["perturbations"]),
            actual_h_plus=h.tolist(),positive_loss_difference=delta.tolist(),right_first_order=right.tolist(),
            right_second_order_nonuniform=second,adjacent_first_order_changes=np.abs(np.diff(right)).tolist(),
            stable_resolution_band=None,strict_cross_arm_gap_independently_resolved=False,
            assessment="inconclusive: no stable right-sided slope platform; boundary right derivative is not default two-sided AD")
        rows.append(row)
    output=path.with_name("right-side-analysis.json")
    write(output,dict(status="inconclusive",source_results=str(path),new_binary_calls=0,new_model_compiles=0,
        baseline_loss=f0,baseline_loss_range=result["baseline_loss_range"],coordinates=rows,
        cause_not_isolated="neural FP32 quantization, local nonlinearity and hidden branches are not independently separated",
        scalar_ulp_is_not_neural_quantization_bound=True,central_negative_values_discarded_as_feasible_derivative_evidence=True,
        original_or_candidate_correctness_not_adjudicated=True,strict_gate_unchanged=True))
    print(json.dumps(dict(status="inconclusive",output=str(output),new_binary_calls=0,coordinates=rows),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--runtime-preflight",type=Path,required=True)
    parser.add_argument("--analyze-saved",action="store_true")
    args=parser.parse_args()
    if args.analyze_saved:
        analyze_saved(args.output)
        return
    if args.output.exists():
        raise FileExistsError("preserve completed diagnostic")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    root=Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
    diag=root/"runtime/outputs/diagnostics"
    report=dict(status="running",stage="starting",scope="original_T7_public_scalar_local_initial_state_FD",
        model_compiles=0,forward_compiles=0,candidate_executions=0,training_updates=0,paid_jobs=0,
        tolerance_change=False,strict_gate_waiver=False,max_binary_calls=59,binary_calls=0,gpu_count=1,
        hidden_branch_stability_not_claimed=True)
    def emit(stage,**fields):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-started,**fields)
        write(args.output,report)
        print(stage,flush=True)
    emit("starting")
    try:
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
        jax,devices=initialize_gpu_backend()
        from jax._src import compiler
        from jax.experimental.serialize_executable import deserialize_and_load
        import probe_shared_forward_reverse as forward
        from probe_shared_command_buffer_loops import deadline, comparison
        from probe_shared_normalized_keys_recurrent import validate_inputs
        from scripts.hpc.run_shared_learning import load_learning_checkpoint
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        if jax.__version__!="0.4.38" or len(devices)!=1:
            raise ValueError("requires one canonical GPU/JAX0.4.38")
        preflight=json.loads(args.runtime_preflight.read_text())
        if preflight.get("status")!="passed" or not preflight.get("passed") or preflight.get("jax_version")!=jax.__version__ or preflight.get("actual_backend")!="gpu":
            raise ValueError("same runtime preflight differs/failed")
        def forbidden(*unused,**kwargs):
            report["forbidden_compile_attempts"]=report.get("forbidden_compile_attempts",0)+1
            raise RuntimeError("zero-model-compile FD attempted JAX compilation")
        with deadline(410,"whole_process",emit):
            source=SimpleNamespace(source_profile=diag/"shared-split-profile-c599236.json",
                input_bundle=diag/"shared-split-dual-inputs-c599236.pkl",
                prepared_context=diag/"shared-expert-real-daily-context-c599236.pkl",
                daily_admission_report=diag/"shared-combined-real-daily-20260925.json",
                seven_admission_report=diag/"shared-combined-recurrent-T7-20260926.json",
                thirty_admission_report=diag/"shared-combined-recurrent-T30-20260926.json",
                support_owner_file=root/"runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py")
            profile,ref,bundle,context,checkpoint_path,admissions=forward._assets(source)
            checkpoint=load_learning_checkpoint(checkpoint_path,expected_method=ref["method"])
            # Existing import-time physical lookup tables can use tiny eager
            # primitives. Bootstrap is unchanged; reject all compilation from
            # this point, before any executable load or model evaluation.
            compiler.compile_or_get_cached=forbidden
            compiler.backend_compile=forbidden
            if checkpoint["completed_updates"]!=0:
                raise ValueError("requires fixed update0")
            parameters=jax.device_put(checkpoint["parameters"])
            if sum(np.size(v) for v in jax.tree_util.tree_leaves(parameters))!=64884481:
                raise ValueError("parameter scope differs")
            case=bundle["payloads"][0]["cases"][7]
            initial=np.asarray(case[0]).copy()
            if initial.shape!=(3854,) or initial.dtype!=np.float64 or len(case)!=5 or np.shape(case.days[0].continuous_state)[0]!=7:
                raise ValueError("original physical state/horizon ABI differs")
            call_args=(parameters,*jax.device_put(case))
            stem=diag/"shared-recurrent-checkpoint-policy-20260930-v2-T7"
            accepted=json.loads(Path(str(stem)+"-reference.json").read_text())
            identity=dict(source_head=ref["source_git_head"],source_profile_sha256=hashlib.sha256(source.source_profile.read_bytes()).hexdigest(),
                method=ref["method"],artifact_id=int(bundle["artifact_ids"][0]),horizon=7,checkpoint=str(checkpoint_path.resolve()),
                jax_version=jax.__version__,admitted_executable=admissions["T7"]["executable"],
                candidate_decoder_sha256=hashlib.sha256((root/"runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py").read_bytes()).hexdigest(),
                candidate_backbone_sha256=hashlib.sha256((root/"runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py").read_bytes()).hexdigest())
            if accepted.get("status")!="passed" or accepted.get("identity")!=identity or accepted.get("mode")!="reference":
                raise ValueError("requires admitted original default dual reference identity")
            with Path(str(stem)+".reference.pkl").open("rb") as stream:
                saved=pickle.load(stream)
            if saved["identity"]!=identity:
                raise ValueError("original saved result identity differs")
            with Path(str(stem)+".reference.executable.pkl").open("rb") as stream:
                serialized=pickle.load(stream)
            executable=deserialize_and_load(*serialized)
            validate_inputs(jax,serialized,executable,call_args)
            del serialized
            memory=memory_sizes(executable)
            if memory["estimated_live_bytes"]>24*2**30:
                raise MemoryError("original binary exceeds24GiB")
            historical={name:[] for name in ("default","candidate")}
            v4stem=diag/"shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4"
            for name in historical:
                for repeat in range(7):
                    with Path(str(v4stem)+f".{name}.repeat{repeat}.initial_state_vjp.pkl").open("rb") as stream:
                        gradient=np.asarray(pickle.load(stream))
                    if gradient.shape!=(3854,) or gradient.dtype!=np.float64:
                        raise ValueError("saved gradient vector differs")
                    historical[name].append(gradient)
                historical[name]=np.stack(historical[name])
            selected=[33,199,428]
            good=np.all(np.abs(historical["candidate"]-historical["default"])<=1e-6+1e-5*np.abs(historical["default"]),axis=0)
            continuous_leaves=[leaf for leaf in context["head"].contract.state_leaves if not leaf.discrete]
            # Prepared masks stay fixed; save all available original masks.
            templates=case.days[0]
            mask_fields={name:np.asarray(getattr(templates,name)) for name in getattr(templates,"_fields",()) if "defined" in name or "mask" in name}
            # Canonical DailyOperatorInput has no redundant defined mask.
            # Exact host evaluation of input_assembly._defined51–54 for the
            # assembler's other=continuous state (assemble165–185).
            original_state=np.asarray(templates.continuous_state)
            mask_fields["canonical_continuous_defined"]=(np.isfinite(original_state)&(np.abs(original_state)<0.5e20))
            initial_defined=mask_fields["canonical_continuous_defined"][0]
            for name in templates.pft._fields:
                if "mask" in name or "fraction" in name:
                    mask_fields["pft."+name]=np.asarray(getattr(templates.pft,name))
            control=next(i for i in (200,198,427,429,1) if good[i] and initial_defined[i] and np.any(historical["default"][:,i]!=0) and i not in selected)
            selected.append(control)
            for i in selected:
                if not initial_defined[i]:
                    raise ValueError("requested derivative coordinate is undefined")
            normalization=context["head"].input_normalization
            continuous_scale=np.asarray(normalization.continuous_scale)
            coordinate_info=[]
            factors=np.asarray([.03,.01,.003,.001,.0003,.0001,.00003])
            for coordinate in selected:
                leaf=next(leaf for leaf in continuous_leaves if leaf.start<=coordinate<leaf.stop)
                offset=coordinate-int(leaf.start)
                local=tuple(int(i) for i in np.unravel_index(offset,leaf.shape)) if leaf.shape else ()
                value=float(initial[coordinate]); scale=float(continuous_scale[coordinate])
                magnitude=max(abs(value),abs(scale),1e-8)
                # Conservative local radius, avoiding crossing zero and obvious
                # fraction domains. This does not certify internal branch masks.
                domain_radius=.1*abs(value) if value>0 else float("inf")
                fraction=any(word in leaf.key.lower() for word in ("fraction","vegetfrac","veget_max","veget","rveget")) and 0<value<1
                if fraction:
                    domain_radius=min(domain_radius,.1*(1-value))
                max_h=min(.03*magnitude,domain_radius)
                if not np.isfinite(max_h) or max_h<=0:
                    raise ValueError("coordinate lacks finite two-sided physical perturbation")
                h_values=max_h*factors/factors[0]
                info=dict(coordinate=coordinate,key=leaf.key,component=leaf.component,field_shape=list(leaf.shape),
                    local_index=list(local),axis_names=list(leaf.axis_names),initial_value=value,input_scale=scale,
                    initial_defined=True,state_dtype=str(initial.dtype),control=coordinate==control,
                    step_sizes=h_values.tolist(),two_sided_domain_radius=None if np.isinf(domain_radius) else domain_radius,
                    original_saved_ad=float(saved["result"][1][1][coordinate]),
                    historical_original_ad=historical["default"][:,coordinate].tolist(),
                    historical_candidate_ad=historical["candidate"][:,coordinate].tolist())
                coordinate_info.append(info)
            write(args.output.with_suffix(".fixed-masks.pkl"),dict(template_masks=mask_fields,targets_defined=np.asarray(case.targets["defined"]),
                state_declared_dynamic=np.asarray(context["state_scales"].state_declared_dynamic.other),initial_stock_defined=case.loss_context.initial_stock_defined,
                target_masks={k:np.asarray(v) for k,v in case.targets.items() if "defined" in k or "active" in k or "fraction" in k}))
            emit("assets_validated",identity=identity,source_inputs=str(source.input_bundle),original_executable=str(Path(str(stem)+".reference.executable.pkl")),
                 coordinates=coordinate_info,planned_binary_calls=59,memory=memory,mask_fields={k:dict(shape=list(v.shape),dtype=str(v.dtype),true_count=int(np.count_nonzero(v))) for k,v in mask_fields.items()},
                 initial_mask_provenance="input_assembly.py::_defined51–54; assemble165–185, other=continuous",
                 fd_resolution_limit="FP64 output ULP/baseline loss repeat do not cover neural FP32 intermediate quantization",
                 source_branch_limits="saved executable exposes aggregate hard/declared_mask_change only, no full internal branch trace")
            baselines=[];rows=[]
            def evaluate(state,label,full=False):
                if report["binary_calls"]>=59:
                    raise RuntimeError("binary call cap exceeded")
                arguments=(call_args[0],jax.device_put(state),*call_args[2:])
                jax.block_until_ready(arguments)
                began=time.perf_counter()
                result=jax.block_until_ready(executable(*arguments))
                report["binary_calls"]+=1
                duration=time.perf_counter()-began
                compact=jax.device_get((result[0],result[1][1]))
                value,aux=compact[0]
                grad=np.asarray(compact[1])
                hard={k:int(v) for k,v in aux["hard"].items()}
                record=dict(label=label,loss=float(value),loss_dtype=str(value.dtype),loss_ulp=float(abs(np.spacing(value))),
                    auxiliary=host_scalars(aux),hard_counts=hard,physics_valid=not any(hard.values()),
                    declared_mask_change=int(aux["declared_mask_change"]),state_gradients={str(i):float(grad[i]) for i in selected},
                    state_gradient_finite=bool(np.isfinite(grad).all()),elapsed_seconds=duration)
                rows.append(record)
                if full:
                    host_result=jax.device_get(result)
                    bridges={"loss_aux":comparison(jax,host_result[0],saved["result"][0]),
                             "all_parameters":comparison(jax,host_result[1][0],saved["result"][1][0]),
                             "all_initial_states":comparison(jax,host_result[1][1],saved["result"][1][1])}
                    if not bridges["loss_aux"]["passed"] or not bridges["all_parameters"]["passed"]:
                        raise ValueError("original saved loss/parameter bridge failed")
                    emit("baseline_bridge",saved_bridge=bridges,all_parameters_finite=all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(host_result[1][0])))
                    del host_result
                if not np.isfinite(record["loss"]) or not record["state_gradient_finite"]:
                    raise ValueError("nonfinite original replay result")
                write(args.output.with_suffix(".losses-and-state-gradients.pkl"),dict(rows=rows,coordinates=coordinate_info))
                emit("result_saved",latest=record,binary_calls=report["binary_calls"])
                return record
            baselines.append(evaluate(initial,"baseline0",full=True))
            baselines.append(evaluate(initial,"baseline1"))
            if not all(row["physics_valid"] for row in baselines):
                raise ValueError("unperturbed original physics hard gates failed")
            del saved
            for info in coordinate_info:
                coordinate=info["coordinate"]
                info["perturbations"]=[]
                for index,h in enumerate(info["step_sizes"]):
                    plus=initial.copy();minus=initial.copy()
                    plus[coordinate]+=h;minus[coordinate]-=h
                    hp=float(plus[coordinate]-initial[coordinate]);hm=float(initial[coordinate]-minus[coordinate])
                    if hp<=0 or hm<=0:
                        raise ValueError("perturbation lost to input quantization")
                    # Alternate plus/minus order; no candidate is evaluated.
                    if index%2:
                        m=evaluate(minus,f"c{coordinate}-h{index}");p=evaluate(plus,f"c{coordinate}+h{index}")
                    else:
                        p=evaluate(plus,f"c{coordinate}+h{index}");m=evaluate(minus,f"c{coordinate}-h{index}")
                    pair=dict(requested_h=h,actual_h_plus=hp,actual_h_minus=hm,asymmetry=hp-hm,plus=p,minus=m,
                              fd=(p["loss"]-m["loss"])/(hp+hm),
                              physics_valid=p["physics_valid"] and m["physics_valid"],
                              mask_counts_match_baseline=p["declared_mask_change"]==m["declared_mask_change"]==baselines[0]["declared_mask_change"])
                    info["perturbations"].append(pair)
                    emit("finite_difference_saved",coordinates=coordinate_info)
            baselines.append(evaluate(initial,"baseline2_end"))
            loss_noise=max(r["loss"] for r in baselines)-min(r["loss"] for r in baselines)
            for info in coordinate_info:
                for pair in info["perturbations"]:
                    # An empirical resolution floor, not a guaranteed bound:
                    # self loss noise + scalar representational half-ULPs.
                    pair["baseline_noise_slope_floor"]=(2*loss_noise+pair["plus"]["loss_ulp"]+pair["minus"]["loss_ulp"])/(pair["actual_h_plus"]+pair["actual_h_minus"])
                info["baseline_ad_values"]=[r["state_gradients"][str(info["coordinate"])] for r in baselines]
                info["strict_reference_allowance"]=1e-6+1e-5*abs(float(np.median(info["baseline_ad_values"])))
                info["historical_candidate_gap_from_current_baseline"]=abs(float(np.median(info["historical_candidate_ad"]))-float(np.median(info["baseline_ad_values"])))
                differences=[abs(b["fd"]-a["fd"]) for a,b in zip(info["perturbations"][:-1],info["perturbations"][1:])]
                info["adjacent_fd_changes"]=differences
                info["assessment"]="inconclusive_until_convergence_and_resolution_assessed; no strict gate waiver"
            emit("completed",status="completed_diagnostic",baselines=baselines,baseline_loss_range=loss_noise,
                 coordinates=coordinate_info,all_results=rows,device_memory=devices[0].memory_stats(),
                 forbidden_compile_attempts=report.get("forbidden_compile_attempts",0),candidate_executions=0)
    except BaseException as error:
        emit("exception",status="error",error_type=type(error).__name__,error=str(error))
        raise


if __name__=="__main__":
    main()
