"""Bounded FP32 original expert5 T7 scan: only outer remat prevent_cse changes.

The decoder body is original_component (original captured 59 chunks128).
RolloutLoss._scan is reused unchanged. The representative recurrence and its
synthetic loss are diagnostic, not public daily physics/science admission.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import re
import time
import numpy as np
from probe_shared_command_buffer_loops import comparison, deadline, forbid_component_compile, original_component, write


class ScanCheckpointJax:
    """Change exactly RolloutLoss._scan's transition checkpoint argument."""
    def __init__(self, jax, prevent_cse):
        self.jax, self.prevent_cse = jax, prevent_cse
    def __getattr__(self, name):
        return getattr(self.jax, name)
    def checkpoint(self, function, **kwargs):
        if function.__name__ != "transition" or kwargs:
            raise ValueError("unexpected checkpoint boundary or changed save policy")
        return self.jax.checkpoint(function, prevent_cse=self.prevent_cse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--runtime-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve completed diagnostic")
    started = time.perf_counter()
    report = dict(status="running", scope="original_full_expert5_FP32_representative_T7_RolloutLoss_scan",
                  public_loss_admission=False, model_admission=False, training_updates=0, paid_jobs=0,
                  gpu_count=1, rtol=1e-5, atol=1e-6, horizon=7, save_policy="default_None_both_arms")
    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **values)
        write(args.output, report)
        print(stage, flush=True)
    emit("initializing")
    preflight = json.loads(args.runtime_preflight.read_text())
    if preflight.get("status") not in ("passed", "ok"):
        raise ValueError("actual GPU runtime preflight did not pass")
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    import jax.numpy as jnp
    from jax._src import ad_checkpoint, config as jax_config
    from jax.experimental.serialize_executable import serialize
    from research.daily_coarse_graining.daily_operator.rollout_loss import RolloutLoss
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    if jax.__version__ != "0.4.38" or len(devices) != 1:
        raise ValueError("requires canonical JAX0.4.38 single GPU")
    emit("runtime_verified", jax_version=jax.__version__, device=str(devices[0]),
         remat_opt_barrier=bool(jax_config.remat_opt_barrier.value),
         native_remat_lowering_source=inspect.getsource(ad_checkpoint._remat_lowering),
         checkpoint_signature=str(inspect.signature(jax.checkpoint)),
         runtime_preflight=str(args.runtime_preflight), rollout_scan_source=inspect.getsource(RolloutLoss._scan))
    with deadline(585, "bounded_process", emit):
        with args.payload.open("rb") as stream:
            saved = pickle.load(stream)
        selected = next(r for r in saved["records"] if r["route"] == "expert5" and r["sample"] == 0)
        config = saved["config"]
        if saved["schema"] != "shared_normalized_keys_real_v1" or selected["query"].shape != (7542,1,384) or selected["chunk"] != 128:
            raise ValueError("requires original captured expert5 full59chunks128")
        if (config.resolved_decoder_width,config.resolved_decoder_attention_heads,config.coordinate_blocks) != (384,8,2):
            raise ValueError("architecture differs")
        p = jax.device_put((saved["parameters"],selected["dynamic"],selected["banks"]))
        q = jax.device_put(selected["query"].reshape(-1))
        masks, indices = jax.device_put((selected["masks"],selected["indices"]))
        days = jax.device_put(np.arange(7,dtype=np.int32))
        discrete = jax.device_put(np.int32(0))
        cot = jnp.sin(jnp.arange(q.size,dtype=jnp.float32)*0.11).reshape(selected["query"].shape)
        blocks_count = sum(np.size(v) for v in jax.tree_util.tree_leaves(p[0]))
        if blocks_count != 3548928 or any(v.dtype != np.float32 for v in jax.tree_util.tree_leaves((p,q))):
            raise ValueError("decoder count/FP32 inputs differ")
        def step(parameters, state, discrete_state, day):
            blocks, dynamic, banks = parameters
            query = state.reshape(selected["query"].shape)
            decoded = original_component(blocks,(query,dynamic,*banks),masks,indices,config,128)
            next_state = (query + jnp.float32(0.001)*decoded).reshape(-1)
            return next_state, discrete_state+1, {"decoded":decoded}
        def loss_with_aux(parameters, state, observations, targets, context):
            loss = jnp.sum(observations["decoded"]*cot[None,...],dtype=jnp.float32)
            return loss, {"final_state":observations["state"][-1], "decoded":observations["decoded"]}
        def unused_loss(*args):
            raise RuntimeError("diagnostic uses bound auxiliary native loss")
        counts = {"decoder_parameters":blocks_count,
                  "raw_dynamic_and_banks":sum(np.size(v) for v in jax.tree_util.tree_leaves(p[1:])),
                  "initial_query":q.size}
        emit("inputs_ready",identity=saved["identity"],chunk_count=59,chunk=128,query_shape=list(selected["query"].shape),
             coordinates=counts, cotangent="original_jax_sin0.11_all_output_coordinates_unscaled_each_of7_steps",
             representative_recurrence="q_next=q+0.001*original_decoder(q,original_dynamic,original_raw_banks)",
             limitations="no physical updater/backbone/public science loss; full expert5 body and all raw-input AD retained")
        compiled, rows, initial, jaxprs = {}, {}, {}, {}
        options = {"xla_gpu_enable_llvm_module_compilation_parallelism":False}
        for name,prevent in (("true",True),("false",False)):
            rollout = RolloutLoss(ScanCheckpointJax(jax,prevent),jnp,step,unused_loss)
            objective = rollout.bind_checkpointed_loss_with_aux(loss_with_aux)
            def function(parameters,state, objective=objective):
                return objective(parameters,state,discrete,days,None,None)
            native = jax.value_and_grad(function,argnums=(0,1),has_aux=True)
            began=time.perf_counter()
            jaxpr = str(jax.make_jaxpr(native)(p,q))
            args.output.with_suffix(f".{name}.jaxpr.txt").write_text(jaxpr)
            lower = jax.jit(native).lower(p,q)
            stable = str(lower.compiler_ir(dialect="stablehlo"))
            args.output.with_suffix(f".{name}.stablehlo.txt").write_text(stable)
            emit("lowered",variant=name,lower_seconds=time.perf_counter()-began)
            began=time.perf_counter()
            with deadline(240,"backend_"+name,emit):
                binary=lower.compile(compiler_options=options)
            compiled[name]=binary
            hlo=binary.as_text()
            args.output.with_suffix(f".{name}.compiledhlo.txt").write_text(hlo)
            write(args.output.with_suffix(f".{name}.executable.pkl"),serialize(binary))
            memory=binary.memory_analysis()
            metric={k:int(getattr(memory,k)) for k in ("argument_size_in_bytes","temp_size_in_bytes","output_size_in_bytes","alias_size_in_bytes")}
            metric["estimated_live_bytes"]=sum(metric[k] for k in ("argument_size_in_bytes","temp_size_in_bytes","output_size_in_bytes"))-metric["alias_size_in_bytes"]
            if metric["estimated_live_bytes"]>24*2**30:
                raise MemoryError("exceeds24GiB")
            rows[name]=dict(backend_seconds=time.perf_counter()-began,compiler_options=options,memory=metric,warm_seconds=[],
                 stablehlo_barriers=stable.count("stablehlo.optimization_barrier"),compiled_barriers=hlo.count("opt-barrier("),
                 stablehlo_whiles=stable.count("stablehlo.while"),compiled_whiles=len(re.findall(r"\bwhile\(",hlo)),
                 compiled_dots=len(re.findall(r"\bdot\(",hlo)),compiled_hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                 remat_jaxpr_count=jaxpr.count("remat2["),default_policy_count=jaxpr.count("policy=None"))
            jaxprs[name]=jaxpr
            with forbid_component_compile(True):
                initial[name]=jax.device_get(jax.block_until_ready(binary(p,q)))
            write(args.output.with_suffix(f".{name}.initial.pkl"),initial[name])
            emit("arm_saved",variants=rows)
        def scopes(a,b):
            return {"loss_and_aux":comparison(jax,a[0],b[0]),"decoder_parameter_vjp":comparison(jax,a[1][0][0],b[1][0][0]),
                    "captured_dynamic_raw_bank_vjp":comparison(jax,a[1][0][1:],b[1][0][1:]),"initial_query_vjp":comparison(jax,a[1][1],b[1][1])}
        first=scopes(initial["false"],initial["true"])
        shapes={n:[dict(shape=list(v.shape),dtype=str(v.dtype)) for v in jax.tree_util.tree_leaves(a)] for n,a in initial.items()}
        finite={n:all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(a)) for n,a in initial.items()}
        emit("initial_compared",initial_comparisons=first,gradient_and_output_shapes=shapes,all_finite=finite)
        pairs=[]
        with forbid_component_compile(True):
            for repeat in range(7):
                results={}
                for name in (("true","false") if repeat%2==0 else ("false","true")):
                    began=time.perf_counter()
                    result=jax.block_until_ready(compiled[name](p,q))
                    rows[name]["warm_seconds"].append(time.perf_counter()-began)
                    results[name]=jax.device_get(result)
                pair=scopes(results["false"],results["true"])
                pairs.append(pair)
                write(args.output.with_suffix(f".pair{repeat}.pkl"),results)
                emit("pair_saved",repeat=repeat,pair_comparisons=pairs)
        for row in rows.values():
            row["warm_median_seconds"]=float(np.median(row["warm_seconds"]))
        strict=all(c["passed"] for c in first.values()) and all(c["passed"] for pair in pairs for c in pair.values())
        emit("completed",status="passed" if strict and all(finite.values()) else "parity_failed",variants=rows,
             pair_comparisons=pairs,initial_comparisons=first,
             warm_speedup=rows["true"]["warm_median_seconds"]/rows["false"]["warm_median_seconds"],
             default_policy_preserved=rows["true"]["default_policy_count"]==rows["false"]["default_policy_count"],
             self_comparisons_are_not_tolerance_waivers=True,device_memory=devices[0].memory_stats())


if __name__ == "__main__":
    main()
