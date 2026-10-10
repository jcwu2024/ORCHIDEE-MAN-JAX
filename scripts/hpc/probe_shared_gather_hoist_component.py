"""Gather-only hoist component on original real expert5 raw-bank native VJP."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--default-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-preflight", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior diagnostic")
    started = time.perf_counter()
    candidate_name = "gather_hoist"
    batch_size = None
    report = {"status": "running", "scope": "captured_original_expert128_forward_and_native_parameter_raw_input_VJP",
        "variant": "expert5_raw_gathers_outside_original_128_map_FP32", "rtol": 1e-5, "atol": 1e-6,
        "full_model_admission": False, "training_updates": 0, "paid_jobs": 0,
        "gpu_count": 1, "self_replays_are_diagnostic": True}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **fields)
        write(args.output, report)
        print(stage, flush=True)
    emit("initializing")
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import serialize, deserialize_and_load
    from shared_chunk_gather_hoist_execution import component, validate_original_helpers
    if len(devices) != 1 or jax.__version__ != "0.4.38":
        raise ValueError("requires one canonical JAX0.4.38 GPU")
    preflight=json.loads(args.runtime_preflight.read_text())
    if preflight.get('status')!='passed' or preflight.get('passed') is not True or preflight.get('actual_backend')!='gpu' or preflight.get('requested_backend')!='gpu' or preflight.get('jax_version')!=jax.__version__:
        raise ValueError('missing/failed/incompatible actual runtime preflight')
    emit('runtime_verified',runtime_preflight=str(args.runtime_preflight),precision='original_FP32',
         active_helper_source_provenance=validate_original_helpers())
    with deadline(890, "bounded_process", emit):
        with args.payload.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["schema"] != "shared_normalized_keys_real_v1":
            raise ValueError("requires original raw-bank payload")
        config = saved["config"]
        if (config.resolved_decoder_width, config.resolved_decoder_attention_heads, config.coordinate_blocks) != (384,8,2):
            raise ValueError("architecture differs")
        selected = next(r for r in saved["records"] if r["route"] == "expert5" and r["sample"] == 0)
        if selected["query"].shape != (7542,1,384) or selected["chunk"] != 128:
            raise ValueError("original component scope differs")
        blocks = jax.device_put(saved["parameters"])
        x = jax.device_put(jax.tree_util.tree_map(lambda a: None if a is None else np.expand_dims(a,0),
            (selected["query"],selected["dynamic"],*selected["banks"])))
        masks, indices = jax.device_put(jax.tree_util.tree_map(lambda a: None if a is None else np.expand_dims(a,0),
            (selected["masks"],selected["indices"])))
        cot = jnp.sin(jnp.arange(np.prod(x[0].shape),dtype=jnp.float32)*0.11).reshape(x[0].shape)
        parameters = sum(np.size(a) for a in jax.tree_util.tree_leaves(blocks))
        if parameters != 3548928:
            raise ValueError("decoder parameter scope differs")
        options = {"xla_gpu_graph_min_graph_size":5, "xla_gpu_enable_llvm_module_compilation_parallelism":False}
        source = json.loads(args.default_report.read_text())
        if (source.get("status") != "threshold_component_measured" or source.get("identity") != saved["identity"]
                or source.get("input_shape") != list(x[0].shape) or source.get("route") != "expert5"
                or source.get("microbatch") != 1 or source["variants"]["default"]["compiler_options"] != options
                or source.get("cotangent") != "original_jax_sin0.11_all_output_coordinates_unscaled"):
            raise ValueError("saved original capture/flags/cotangent differ")
        emit("inputs_ready", identity=saved["identity"], input_shape=list(x[0].shape), route="expert5",microbatch=1,
            decoder_parameter_coordinates=parameters,chunk=128,chunk_count=59,padded_rows=7552,padding_rows=10,
            batch_size=batch_size,full_groups=29 if batch_size else None,remainder_chunks=1 if batch_size else None,
            cotangent="original_jax_sin0.11_all_output_coordinates_unscaled", baseline_component_recompilations=0)
        compiled, binary_records = {}, {}
        with forbid_component_compile(True):
            binaries = []
            for label in ("forward","native_vjp"):
                path = args.default_report.with_suffix(f".default.{label}.executable.pkl")
                with path.open("rb") as stream:
                    binaries.append(deserialize_and_load(*pickle.load(stream)))
                binary_records[label] = {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            compiled["default"] = tuple(binaries)
        emit("original_loaded", saved_original_binaries=binary_records)
        candidate_options = dict(options)
        def forward(p,values):
            return jax.vmap(lambda v,m,i: component(p,v,m,i,config,128))(values,masks,indices)
        def combined(p,values,cotangent):
            output,pullback = jax.vjp(forward,p,values)
            return output,pullback(cotangent)
        began=time.perf_counter()
        lower_f=jax.jit(forward).lower(blocks,x)
        lower_v=jax.jit(combined).lower(blocks,x,cot)
        emit("candidate_lowered", lower_seconds=time.perf_counter()-began, compiler_options=candidate_options)
        binaries,backend=[],{}
        for label,lowered in (("forward",lower_f),("native_vjp",lower_v)):
            began=time.perf_counter()
            with deadline(180,"component_backend_"+label,emit):
                executable=lowered.compile(compiler_options=candidate_options)
            backend[label]=time.perf_counter()-began
            binaries.append(executable)
            write(args.output.with_suffix(f".{candidate_name}.{label}.executable.pkl"),serialize(executable))
            args.output.with_suffix(f".{candidate_name}.{label}.hlo.txt").write_text(executable.as_text())
        compiled[candidate_name]=tuple(binaries)
        rows,initial={},{}
        def scopes(a,b):
            return {"output":comparison(jax,a[0],b[0]),
                "all_decoder_parameter_vjp":comparison(jax,a[1][0],b[1][0]),
                "all_captured_raw_bank_and_input_vjp":comparison(jax,a[1][1],b[1][1])}
        def finite(value):
            return {name:{"finite":all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(tree)),
                "coordinates":sum(np.size(a) for a in jax.tree_util.tree_leaves(tree)),
                "nonzero_coordinates":sum(np.count_nonzero(a).item() if hasattr(np.count_nonzero(a),"item") else int(np.count_nonzero(a)) for a in jax.tree_util.tree_leaves(tree))}
                for name,tree in (("output",value[0]),("parameters",value[1][0]),("inputs",value[1][1]))}
        for name,(fwd,reverse) in compiled.items():
            mem=reverse.memory_analysis()
            metric={k:int(getattr(mem,k)) for k in ("argument_size_in_bytes","temp_size_in_bytes","output_size_in_bytes","alias_size_in_bytes")}
            metric["estimated_live_bytes"]=sum(metric[k] for k in ("argument_size_in_bytes","temp_size_in_bytes","output_size_in_bytes"))-metric["alias_size_in_bytes"]
            if metric["estimated_live_bytes"]>12*2**30:
                raise MemoryError("component exceeds12GiB")
            with forbid_component_compile(True):
                initial[name]=jax.device_get(jax.block_until_ready(reverse(blocks,x,cot)))
                jax.block_until_ready(fwd(blocks,x))
            write(args.output.with_suffix(f".{name}.result.pkl"),{"identity":saved["identity"],"result":initial[name]})
            rows[name]={"memory":metric,"warm_forward_seconds":[],"warm_vjp_seconds":[],"self_comparisons":[]}
        finite_rows={key:finite(value) for key,value in initial.items()}
        emit('initial_artifacts_preserved',initial_finite_and_nonzero=finite_rows)
        if any(not scope['finite'] for arm in finite_rows.values() for scope in arm.values()):
            raise ValueError('nonfinite initial component output/VJP; preserved')
        stats=devices[0].memory_stats() or {}
        if 'peak_bytes_in_use' not in stats or stats['peak_bytes_in_use']>12*2**30:
            raise MemoryError('component observed peak unavailable/exceeds12GiB; preserved')
        with args.default_report.with_suffix(".default.result.pkl").open("rb") as stream:
            previous=pickle.load(stream)
        if previous["identity"]!=saved["identity"]:
            raise ValueError("saved baseline identity differs")
        emit("initial_compared",initial_comparisons=scopes(initial[candidate_name],initial["default"]),
            preserved_baseline_comparison=scopes(initial["default"],previous["result"]),backend_seconds=backend)
        pairs=[]
        with forbid_component_compile(True):
            for repeat in range(7):
                results={}
                for name in (("default",candidate_name) if repeat%2==0 else (candidate_name,"default")):
                    fwd,reverse=compiled[name]
                    began=time.perf_counter();jax.block_until_ready(fwd(blocks,x))
                    rows[name]["warm_forward_seconds"].append(time.perf_counter()-began)
                    began=time.perf_counter();value=jax.block_until_ready(reverse(blocks,x,cot))
                    rows[name]["warm_vjp_seconds"].append(time.perf_counter()-began)
                    results[name]=jax.device_get(value)
                    rows[name]["self_comparisons"].append(scopes(results[name],initial[name]))
                    write(args.output.with_suffix(f".{name}.repeat{repeat}.full_vjp.pkl"),results[name])
                pairs.append(scopes(results[candidate_name],results["default"]))
                emit("paired_repeat_saved",repeat=repeat,pair_comparisons=pairs)
        for row in rows.values():
            row["warm_forward_median_seconds"]=float(np.median(row["warm_forward_seconds"]))
            row["warm_vjp_median_seconds"]=float(np.median(row["warm_vjp_seconds"]))
        initial_checks=scopes(initial[candidate_name],initial["default"])
        passed=all(c["passed"] for c in initial_checks.values()) and all(c["passed"] for p in pairs for c in p.values())
        emit("completed",status="passed" if passed else "parity_failed",initial_comparisons=initial_checks,
            variants=rows,pair_comparisons=pairs,initial_finite_and_nonzero={k:finite(v) for k,v in initial.items()},
            forward_speedup=rows["default"]["warm_forward_median_seconds"]/rows[candidate_name]["warm_forward_median_seconds"],
            vjp_speedup=rows["default"]["warm_vjp_median_seconds"]/rows[candidate_name]["warm_vjp_median_seconds"],
            device_memory=devices[0].memory_stats(),model_admission=False)
        raise SystemExit(0 if passed else 1)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        path=Path(__import__('sys').argv[__import__('sys').argv.index('--output')+1])
        if path.exists():
            report=json.loads(path.read_text())
            report.update(status='resource_failed' if isinstance(error,MemoryError) else 'error',stage='exception',error_type=type(error).__name__,error=str(error))
            write(path,report)
        raise
