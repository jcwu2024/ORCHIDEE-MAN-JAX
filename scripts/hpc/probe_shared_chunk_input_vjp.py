"""Native input-only derivative cost with dynamic original decoder weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np

from probe_shared_command_buffer_loops import (
    comparison, deadline, forbid_component_compile, original_component, write)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("payload", "default-report", "runtime-preflight", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior diagnostic")
    started=time.perf_counter()
    report={"status":"running", "scope":"original_expert5_component_native_input_only_VJP_cost",
        "rtol":1e-5,"atol":1e-6,"parameters_dynamic":True,"inputs_dynamic":True,
        "cotangent_dynamic":True,"native_AD":True,"custom_AD":False,"full_model_admission":False,
        "parameter_gradient_returned":False,"self_replays_are_diagnostic":True,"gpu_count":1,"paid_jobs":0}
    def emit(stage,**fields):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-started,**fields)
        write(args.output,report);print(stage,flush=True)
    emit("initializing")
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax,devices=initialize_gpu_backend()
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import serialize,deserialize_and_load
    if len(devices)!=1 or jax.__version__!="0.4.38":
        raise ValueError("requires original one-GPU JAX0.4.38")
    preflight=json.loads(args.runtime_preflight.read_text())
    if preflight.get("passed") is not True or preflight.get("status")!="passed" or preflight.get("actual_backend")!="gpu":
        raise ValueError("requires actual deployed runtime preflight")
    emit("preflight_verified",runtime_preflight=str(args.runtime_preflight),jax_version=jax.__version__)
    with deadline(890,"bounded_component",emit):
        with args.payload.open("rb") as stream:
            saved=pickle.load(stream)
        if saved["schema"]!="shared_normalized_keys_real_v1":
            raise ValueError("raw capture schema differs")
        selected=next(r for r in saved["records"] if r["route"]=="expert5" and r["sample"]==0)
        if selected["query"].shape!=(7542,1,384) or selected["chunk"]!=128:
            raise ValueError("original expert5 component differs")
        blocks=jax.device_put(saved["parameters"])
        config=saved["config"]
        if (config.resolved_decoder_width,config.resolved_decoder_attention_heads,config.coordinate_blocks)!=(384,8,2):
            raise ValueError("architecture changed")
        expand=lambda a: None if a is None else np.expand_dims(a,0)
        x=jax.device_put(jax.tree_util.tree_map(expand,(selected["query"],selected["dynamic"],*selected["banks"])))
        masks,indices=jax.device_put(jax.tree_util.tree_map(expand,(selected["masks"],selected["indices"])))
        cot=jnp.sin(jnp.arange(np.prod(x[0].shape),dtype=jnp.float32)*0.11).reshape(x[0].shape)
        parameter_count=sum(np.size(a) for a in jax.tree_util.tree_leaves(blocks))
        if parameter_count!=3548928:
            raise ValueError("requires all original decoder parameters dynamic")
        source=json.loads(args.default_report.read_text())
        options={"xla_gpu_graph_min_graph_size":5,"xla_gpu_enable_llvm_module_compilation_parallelism":False}
        if (source.get("status")!="threshold_component_measured" or source.get("identity")!=saved["identity"]
                or source.get("input_shape")!=list(x[0].shape) or source.get("route")!="expert5"
                or source.get("microbatch")!=1 or source["variants"]["default"]["compiler_options"]!=options
                or source.get("cotangent")!="original_jax_sin0.11_all_output_coordinates_unscaled"):
            raise ValueError("reference inputs/flags/cotangent differ")
        path=args.default_report.with_suffix(".default.native_vjp.executable.pkl")
        with forbid_component_compile(True):
            with path.open("rb") as stream:
                original=deserialize_and_load(*pickle.load(stream))
        emit("original_loaded",identity=saved["identity"],baseline_compilations=0,
            baseline_binary={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()},
            parameter_coordinates=parameter_count,input_coordinates=sum(np.size(a) for a in jax.tree_util.tree_leaves(x)),
            input_shape=list(x[0].shape),chunk=128,chunk_count=59,padded_rows=7552,padding_rows=10,
            cotangent="original_jax_sin0.11_all_output_coordinates_unscaled")
        def forward(p,values):
            return jax.vmap(lambda v,m,i: original_component(p,v,m,i,config,128))(values,masks,indices)
        def input_only(p,values,cotangent):
            # p remains a dynamic JIT argument. Native vjp differentiates only
            # values; this is a newly evaluated derivative, never a replayed
            # state gradient or a stopped gradient in a public training loss.
            output,pullback=jax.vjp(lambda v: forward(p,v),values)
            return output,pullback(cotangent)[0]
        began=time.perf_counter()
        lowered=jax.jit(input_only).lower(blocks,x,cot)
        emit("candidate_lowered",lower_seconds=time.perf_counter()-began,compiler_options=options)
        with deadline(180,"candidate_backend",emit):
            began=time.perf_counter();candidate=lowered.compile(compiler_options=options)
        backend=time.perf_counter()-began
        serialized=serialize(candidate)
        expected=((blocks,x,cot),{})
        if serialized[1]!=jax.tree_util.tree_structure(expected):
            raise ValueError("input-only executable lost dynamic argument tree")
        infos,tree=jax.tree_util.tree_flatten(candidate.args_info)
        values,other=jax.tree_util.tree_flatten(expected)
        if tree!=other or len(infos)!=len(values):
            raise ValueError("dynamic ABI differs")
        for info,value in zip(infos,values,strict=True):
            if tuple(info._aval.shape)!=np.shape(value) or np.dtype(info._aval.dtype)!=np.asarray(value).dtype:
                raise ValueError("dynamic argument shape/dtype differs")
        write(args.output.with_suffix(".input_only.executable.pkl"),serialized)
        args.output.with_suffix(".input_only.hlo.txt").write_text(candidate.as_text())
        mem=candidate.memory_analysis()
        memory={k:int(getattr(mem,k)) for k in ("argument_size_in_bytes","temp_size_in_bytes","output_size_in_bytes","alias_size_in_bytes")}
        memory["estimated_live_bytes"]=sum(memory[k] for k in ("argument_size_in_bytes","temp_size_in_bytes","output_size_in_bytes"))-memory["alias_size_in_bytes"]
        if memory["estimated_live_bytes"]>12*2**30:
            raise MemoryError("component exceeds12GiB")
        def project(result):
            return result[0],result[1][1]
        def compare(a,b):
            return {"output":comparison(jax,a[0],b[0]),"all_captured_raw_bank_and_input_vjp":comparison(jax,a[1],b[1])}
        first,times,self_checks={}, {"original_full":[],"native_input_only":[]},{"original_full":[],"native_input_only":[]}
        with forbid_component_compile(True):
            first["original_full"]=jax.device_get(jax.block_until_ready(original(blocks,x,cot)))
            first["native_input_only"]=jax.device_get(jax.block_until_ready(candidate(blocks,x,cot)))
        initial_checks=compare(first["native_input_only"],project(first["original_full"]))
        for name,value in first.items():
            write(args.output.with_suffix(f".{name}.result.pkl"),{"identity":saved["identity"],"result":value})
        emit("initial_compared",initial_comparisons=initial_checks,backend_seconds=backend,
            candidate_memory=memory,dynamic_ABI={"argument_tree":str(serialized[1]),"argument_leaf_count":len(values)})
        pairs=[]
        with forbid_component_compile(True):
            for repeat in range(7):
                results={}
                for name in (("original_full","native_input_only") if repeat%2==0 else ("native_input_only","original_full")):
                    began=time.perf_counter()
                    value=jax.block_until_ready((original if name=="original_full" else candidate)(blocks,x,cot))
                    times[name].append(time.perf_counter()-began)
                    results[name]=jax.device_get(value)
                    actual=project(results[name]) if name=="original_full" else results[name]
                    expected=project(first[name]) if name=="original_full" else first[name]
                    self_checks[name].append(compare(actual,expected))
                    write(args.output.with_suffix(f".{name}.repeat{repeat}.full_result.pkl"),results[name])
                pairs.append(compare(results["native_input_only"],project(results["original_full"])))
                emit("paired_repeat_saved",repeat=repeat,pair_comparisons=pairs)
        medians={k:float(np.median(v)) for k,v in times.items()}
        gradient=first["native_input_only"][1]
        leaves=jax.tree_util.tree_leaves(gradient)
        valid={"finite":all(np.all(np.isfinite(a)) for a in leaves),
            "coordinates":sum(np.size(a) for a in leaves),"nonzero_coordinates":sum(int(np.count_nonzero(a)) for a in leaves)}
        stats=devices[0].memory_stats() or {}
        if not valid["finite"] or valid["nonzero_coordinates"]==0 or stats.get("peak_bytes_in_use",12*2**30+1)>12*2**30:
            raise ValueError("input-gradient finite/meaningful/peak gate failed")
        passed=all(r["passed"] for r in initial_checks.values()) and all(r["passed"] for pair in pairs for r in pair.values())
        emit("completed",status="passed" if passed else "parity_failed",initial_comparisons=initial_checks,
            pair_comparisons=pairs,self_comparisons=self_checks,matched_warm_seconds=times,
            matched_medians_seconds=medians,original_full_over_input_only_ratio=medians["original_full"]/medians["native_input_only"],
            optimized_parameter_route_total_time_budget_seconds=medians["original_full"]-medians["native_input_only"],
            cost_interpretation="input-only evaluation includes its own complete original forward and native backward; an independent parameter evaluator adds its complete forward/backward cost",
            input_gradient_validation=valid,device_memory=stats,public_loss_admission=False)


if __name__=="__main__":
    main()
