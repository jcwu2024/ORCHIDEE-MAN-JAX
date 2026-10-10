"""Original expert128 real capture: saved native VJP vs deterministic compilation.

No changed decoder, public recurrent loss, training or scientific admission.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
from probe_shared_command_buffer_loops import comparison, deadline, forbid_component_compile, original_component, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--runtime-preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous results')
    started = time.perf_counter()
    report = dict(status='running', stage='initializing', scope='original_expert128_real_micro1_native_vjp',
                  process_xla_flags=os.environ.get('XLA_FLAGS'), gpu_count=1, paid_jobs=0,
                  training_updates=0, full_model_admission=False, public_loss_admission=False,
                  rtol=1e-5, atol=1e-6, reference_compiles=0, deterministic_compiles=0,
                  original_component_source=inspect.getsource(original_component), pairs=[])

    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **fields)
        write(args.output, report)
        print(stage, flush=True)

    emit('initializing')
    try:
        with deadline(590, 'process', emit):
            preflight=json.loads(args.runtime_preflight.read_text())
            if preflight.get('status')!='passed' or preflight.get('actual_backend')!='gpu':
                raise ValueError('new actual deterministic runtime preflight required')
            from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
            jax, devices=initialize_gpu_backend()
            import jax.numpy as jnp
            from jax.experimental.serialize_executable import deserialize_and_load, serialize
            from research.daily_coarse_graining.daily_operator import shared_transformer as st
            if jax.__version__!='0.4.38' or len(devices)!=1:
                raise ValueError('requires canonical JAX0.4.38 single GPU')
            with args.payload.open('rb') as stream:
                saved=pickle.load(stream)
            config=saved['config']
            record=next(r for r in saved['records'] if r['route']=='expert5' and r['sample']==0)
            if saved['schema']!='shared_normalized_keys_real_v1' or record['query'].shape!=(7542,1,384) or record['chunk']!=128:
                raise ValueError('requires original real expert5 full59chunks128')
            if (config.resolved_decoder_width,config.resolved_decoder_attention_heads,config.coordinate_blocks)!=(384,8,2):
                raise ValueError('architecture mismatch')
            reference=json.loads(args.reference_report.read_text())
            base_options={'xla_gpu_graph_min_graph_size':5,'xla_gpu_enable_llvm_module_compilation_parallelism':False}
            if (reference.get('status')!='threshold_component_measured' or reference['identity']!=saved['identity']
                    or reference['route']!='expert5' or reference['microbatch']!=1
                    or reference['variants']['default']['compiler_options']!=base_options
                    or reference['cotangent']!='original_jax_sin0.11_all_output_coordinates_unscaled'):
                raise ValueError('saved original scope/identity/options mismatch')
            # Exact original micro1 outer sample packing; original component's
            # inner chunk execution remains lax.map, never chunk-vmap.
            def outer(value):
                return None if value is None else value[None,...]
            host_x=jax.tree_util.tree_map(outer,(record['query'],record['dynamic'],*record['banks']))
            x=jax.device_put(host_x)
            blocks=jax.device_put(saved['parameters'])
            masks=jax.device_put(jax.tree_util.tree_map(outer,record['masks']))
            indices=jax.device_put(jax.tree_util.tree_map(outer,record['indices']))
            if sum(np.size(v) for v in jax.tree_util.tree_leaves(blocks))!=3548928:
                raise ValueError('parameter scope mismatch')
            if any(v.dtype!=np.float32 for v in jax.tree_util.tree_leaves((blocks,x))):
                raise ValueError('all neural captured inputs/parameters must remain FP32')
            if list(x[0].shape)!=reference['input_shape']:
                raise ValueError('saved baseline ABI mismatch')
            cot=jnp.sin(jnp.arange(np.prod(x[0].shape),dtype=jnp.float32)*0.11).reshape(x[0].shape)

            def forward(parameters, values):
                return jax.vmap(lambda v,m,i:original_component(parameters,v,m,i,config,128))(values,masks,indices)

            def combined(parameters,values,cotangent):
                output,pullback=jax.vjp(forward,parameters,values)
                return output,pullback(cotangent)

            def scopes(actual,expected):
                return {name:comparison(jax,a,b) for name,a,b in (
                    ('output',actual[0],expected[0]),
                    ('all_decoder_parameter_vjp',actual[1][0],expected[1][0]),
                    ('query_vjp',actual[1][1][0],expected[1][1][0]),
                    ('dynamic_vjp',actual[1][1][1],expected[1][1][1]),
                    ('all_raw_bank_vjp',actual[1][1][2:],expected[1][1][2:]))}

            def finite(value):
                return all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(value))

            def memory(binary):
                m=binary.memory_analysis()
                d={key:int(getattr(m,key)) for key in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
                d['estimated_live_bytes']=d['argument_size_in_bytes']+d['output_size_in_bytes']+d['temp_size_in_bytes']-d['alias_size_in_bytes']
                if d['estimated_live_bytes']>24*2**30:
                    raise MemoryError('estimated memory exceeds24GiB')
                return d

            original_path=args.reference_report.with_suffix('.default.native_vjp.executable.pkl')
            began=time.perf_counter()
            with forbid_component_compile(True):
                with original_path.open('rb') as stream:
                    original=deserialize_and_load(*pickle.load(stream))
            binaries={'original':original}
            options={**base_options,'xla_gpu_deterministic_ops':True}
            emit('inputs_and_original_ready',jax_version=jax.__version__,identity=saved['identity'],
                 source_head=saved['identity']['source_git_head'],reference_executable=str(original_path),
                 reference_compile_xla_flags=reference['XLA_FLAGS'],candidate_compile_options=options,
                 original_load_seconds=time.perf_counter()-began,original_memory=memory(original),
                 query_shape=list(record['query'].shape),chunk=128,chunk_count=59,
                 parameter_coordinates=3548928,input_coordinates=sum(np.size(v) for v in jax.tree_util.tree_leaves(x)),
                 cotangent='original_jax_sin0.11_all_output_coordinates_unscaled')
            began=time.perf_counter()
            with deadline(120,'lower',emit):
                lowered=jax.jit(combined).lower(blocks,x,cot)
            lower_seconds=time.perf_counter()-began
            args.output.with_suffix('.stablehlo.txt').write_text(str(lowered.compiler_ir(dialect='stablehlo')))
            emit('lowered',lower_seconds=lower_seconds)
            began=time.perf_counter()
            with deadline(240,'backend',emit):
                candidate=lowered.compile(compiler_options=options)
            backend_seconds=time.perf_counter()-began
            emit('compiled',backend_seconds=backend_seconds,compiler_option_supported=True,deterministic_compiles=1,
                 deterministic_memory=memory(candidate))
            write(args.output.with_suffix('.deterministic.executable.pkl'),serialize(candidate))
            args.output.with_suffix('.deterministic.hlo.txt').write_text(candidate.as_text())
            emit('executable_saved')
            binaries['deterministic']=candidate
            initial={};timings={name:[] for name in binaries};self_checks={name:[] for name in binaries}
            with forbid_component_compile(True):
                for name,binary in binaries.items():
                    initial[name]=jax.device_get(jax.block_until_ready(binary(blocks,x,cot)))
                    if not finite(initial[name]):
                        raise ValueError(name+' initial nonfinite')
                    write(args.output.with_suffix('.'+name+'.initial.pkl'),dict(identity=saved['identity'],result=initial[name]))
                    emit(name+'_initial_saved')
                with args.reference_report.with_suffix('.default.result.pkl').open('rb') as stream:
                    historical=pickle.load(stream)
                if historical['identity']!=saved['identity']:
                    raise ValueError('saved baseline result identity mismatch')
                bridge=scopes(initial['original'],historical['result'])
                del historical
                emit('initial_compared',initial_comparison=scopes(initial['deterministic'],initial['original']),
                     original_saved_bridge=bridge)
                for repeat in range(7):
                    results={}
                    for name in (('original','deterministic') if repeat%2==0 else ('deterministic','original')):
                        began=time.perf_counter()
                        value=jax.block_until_ready(binaries[name](blocks,x,cot))
                        timings[name].append(time.perf_counter()-began)
                        results[name]=jax.device_get(value)
                        if not finite(results[name]):
                            raise ValueError(name+' repeat nonfinite')
                        self_checks[name].append(scopes(results[name],initial[name]))
                    report['pairs'].append(scopes(results['deterministic'],results['original']))
                    write(args.output.with_suffix('.pair'+str(repeat)+'.pkl'),results)
                    emit('pair_saved',repeat=repeat,timings=timings,self_comparisons=self_checks,
                         device_memory=devices[0].memory_stats())
                    if devices[0].memory_stats()['peak_bytes_in_use']>24*2**30:
                        raise MemoryError('active process peak exceeds24GiB')
            medians={name:float(np.median(values)) for name,values in timings.items()}
            strict=all(c['passed'] for c in report['initial_comparison'].values()) and all(c['passed'] for p in report['pairs'] for c in p.values())
            stable={name:all(c['bitwise_identical'] for p in checks for c in p.values()) for name,checks in self_checks.items()}
            emit('completed',status='strict_passed' if strict else 'strict_parity_failed',strict_parity_passed=strict,
                 bitwise_self_stable=stable,warm_median_seconds=medians,
                 diagnostic_timing_ratio=medians['original']/medians['deterministic'],
                 all_results_finite=True,training_throughput_claim=False,model_compiles=1,
                 default_changed=False,device_memory=devices[0].memory_stats())
    except BaseException as error:
        emit('exception',status='error',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':
    main()
