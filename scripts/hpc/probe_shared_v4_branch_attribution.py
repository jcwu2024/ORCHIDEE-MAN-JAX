"""Fixed pair2 input, saved-binary branch evidence: exactly38 model calls."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import threading
import time
import numpy as np
import probe_shared_v4_same_input_update as base

FAMILY=base.DIAG/'shared-v4-same-input-update-20261003'
ORDERS=(('reference','candidate'),('candidate','reference'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,value in base.defaults().items():parser.add_argument('--'+name.replace('_','-'),type=Path,default=value)
    parser.add_argument('--fixed-proposal',type=Path,default=FAMILY/'model/pair-1-reference.full-proposal.pkl')
    parser.add_argument('--old-family',type=Path,default=FAMILY)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--process-seconds',type=float,default=890)
    parser.add_argument('--cli-check-only',action='store_true')
    parser.add_argument('--metadata-check',action='store_true')
    args=parser.parse_args()
    if args.cli_check_only:
        print(json.dumps(dict(status='passed',orders=ORDERS,model_calls=38,optimizer_updates=0,new_model_AD_compiles=0)));return 0
    if args.metadata_check:
        verification=json.loads((args.old_family/'independent-cpu/independent-verification.json').read_text())
        if verification['status']!='independently_verified_raws' or verification['pairs'][2]['A_failures']['gradient']!=977:
            raise ValueError('independent pair2 raw verification missing')
        for path in [args.fixed_proposal,*[getattr(args,name+'_executable') for name in ('daily_reference','daily_candidate','t7_reference','t7_candidate','t30_reference')]]:
            if not path.is_file():raise FileNotFoundError(str(path))
        print(json.dumps(dict(status='passed',fixed_input=str(args.fixed_proposal),independent_977=True,
            saved_binaries_loaded=0,model_calls=0,GPU_calls=0)));return 0
    if args.output_dir is None or not 0<args.process_seconds<=890:parser.error('requires output and≤890s allowance')
    return run(args)

def run(args):
    args.output_dir.mkdir(exist_ok=False,parents=True)
    started=time.perf_counter();timer=threading.Timer(args.process_seconds,lambda:os._exit(124));timer.daemon=True;timer.start()
    report=dict(status='running',scope='fixed_old_pair2_parameters_two_branch_repeats_no_updates',
        model_calls_started=0,model_calls=0,model_call_markers=[],model_AD_compiles=0,optimizer_updates=0,
        retained_training_updates=0,production_adoption=False,raw_model_results=[],component_index=[])
    def emit(stage,**values):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-started,**values)
        base.write(args.output_dir/'model.json',report);print(stage,flush=True)
    try:
        from probe_shared_normalized_keys_recurrent import load,validate_inputs
        tree=load(args.tree_owner_file,'v4_attribution_tree_owner');args.inspect_only=False
        _profile,reference,bundle,digest,_admissions=tree._read_identity(args)
        original=load(args.original_owner_file,'v4_attribution_original_owner')
        learning,signature=base.learning_interface(args)
        for name in ('daily','T7','T30'):
            original.admit_original(getattr(args,name.lower()+'_reference_report'),getattr(args,name.lower()+'_reference_executable'),name,
                reference=reference,bundle=bundle,digest=digest)
        candidate=base.candidate_metadata(args,reference,bundle,digest)
        import jax
        import jax.numpy as jnp
        from jax._src import compiler
        from jax.experimental.serialize_executable import deserialize_and_load
        from scripts.hpc.checked_recurrent_execution import device_helper_cache
        from analyze_shared_v4_ab_offline import compare,entries
        if jax.__version__!='0.4.38' or len(jax.local_devices())!=1 or jax.local_devices()[0].platform!='gpu':
            raise ValueError('canonical one GPU JAX0.4.38 required')
        device=jax.local_devices()[0]
        preflight=json.loads(args.runtime_preflight.read_text())
        if preflight['status']!='passed' or not preflight['passed'] or preflight['actual_backend']!='gpu' or preflight['jax_version']!=jax.__version__:
            raise ValueError('unchanged runtime evidence differs')
        with args.fixed_proposal.open('rb') as stream:fixed=pickle.load(stream)
        if fixed['source_head']!=reference['source_git_head'] or fixed['method']!=reference['method'] or fixed['artifact_id']!=742:
            raise ValueError('fixed input identity differs')
        if fixed['result']['pair']!=1 or fixed['result']['arm']!='reference' or int(np.asarray(fixed['result']['optimizer'].step))!=2:
            raise ValueError('must use actual pair1 reference updated parameters')
        host_parameters=fixed['result']['parameters'];parameters=jax.device_put(host_parameters);del fixed
        daily=jax.device_put(bundle['payloads'][0]['daily']);cases=jax.device_put(bundle['payloads'][0]['cases'])
        sample4=jax.tree_util.tree_map(lambda value:value[:4],daily)
        if any(np.shape(value)[0]!=32 for value in jax.tree_util.tree_leaves(daily)):raise ValueError('requires original daily32')
        defined={'daily':np.isfinite(np.asarray(daily.inputs.continuous_state))&(np.abs(np.asarray(daily.inputs.continuous_state))<.5e20),
            **{f'T{h}':np.isfinite(np.asarray(cases[h][0]))&(np.abs(np.asarray(cases[h][0]))<.5e20) for h in (7,30)}}
        old_indices=json.loads((args.old_family/'independent-cpu/pair-2-failed-parameter-indices.json').read_text())['source_parameter_blocks']
        emit('inputs_ready',fixed_input=str(args.fixed_proposal),fixed_parameter_ABI=compare(host_parameters,host_parameters),
            candidate_metadata=candidate,learning_interface=signature,immutable_source_head=reference['source_git_head'],
            daily_mean_normalization='sum of eight micro4 means divided by eight, canonical accumulator',
            mixed_gradient_operation='canonical make_tree_operations(32).combine_daily_mean: (d+s)+t',
            no_input_donation_or_mutation=True,undefined_input_counts={k:int((~v).sum()) for k,v in defined.items()})
        model_compile_counts={'compile_or_get_cached_calls':0,'backend_compile_calls':0}
        @contextmanager
        def no_model_compile():
            restore=tree.reject_compilation_before_entry(compiler,model_compile_counts)
            try:yield
            finally:restore()
        executables={};loaded=[]
        for arm,name in (('reference','daily'),('reference','T7'),('reference','T30'),('candidate','daily'),('candidate','T7')):
            path=getattr(args,name.lower()+'_'+arm+'_executable');began=time.perf_counter()
            with path.open('rb') as stream:serialized=pickle.load(stream)
            with no_model_compile():binary=deserialize_and_load(*serialized)
            arguments=(parameters,sample4.inputs.continuous_state,sample4) if name=='daily' else (parameters,*cases[int(name[1:])])
            validate_inputs(jax,serialized,binary,arguments)
            memory=learning.memory_sizes(binary)
            if memory['estimated_live_bytes']>24*2**30:raise MemoryError('saved binary memory gate')
            executables[arm,name]=binary;loaded.append(dict(arm=arm,component=name,path=str(path),seconds=time.perf_counter()-began,memory=memory))
            del serialized;emit('saved_binary_loaded',loaded_binaries=loaded)
        helper_count={'backend':0};original_backend=compiler.backend_compile
        def counted(*values,**keywords):
            helper_count['backend']+=1;return original_backend(*values,**keywords)
        compiler.backend_compile=counted
        try:
            with jax.default_device(device),device_helper_cache(device):operations=learning.make_tree_operations(jax,jnp,32)
            components={};daily_states=[];context={};calls={}
            def hard_gate(value,name,mask):
                finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(value))
                counts=tree._hard_counts(value['auxiliary']);state=np.asarray(value['initial_state_gradients'])
                abi=compare(value['gradient'],host_parameters)
                if abi['numeric_leaves']!=300 or abi['coordinates']!=64884481 or abi['static_None_nodes']!=2:
                    raise ValueError('full parameter/static None ABI changed after raw save')
                expected_state_shape=((8,4,3854) if mask.shape==(32,3854) else (4,3854)) if name=='daily' else (3854,)
                if state.shape!=expected_state_shape or len(counts)!=(11 if name=='daily' else 14):
                    raise ValueError('complete state/physical counter ABI changed after raw save')
                zero=int(np.count_nonzero(state.reshape(mask.shape)[~mask]))
                memory=device.memory_stats() or {}
                record=dict(finite=bool(finite),hard_counts=counts,undefined_derivative_nonzero=zero,memory=memory,
                    gradient_ABI={key:abi[key] for key in ('numeric_leaves','coordinates','static_None_nodes')},state_shape=list(state.shape))
                record['passed']=bool(finite and not any(counts.values()) and not zero and
                    memory.get('peak_bytes_in_use',24*2**30+1)<=24*2**30 and not any(model_compile_counts.values()))
                if not record['passed']:raise ValueError('hard branch failure after raw save '+name)
                return record
            def model(arm,name,*values):
                key=arm+':'+name;index=calls.get(key,0);marker=dict(**context,arm=arm,component=name,executable_call_index=index,
                    actual_executable_first_call=index==0)
                report['model_calls_started']+=1;emit('model_call_started',current_model_call=marker)
                began=time.perf_counter()
                with no_model_compile():result=jax.block_until_ready(executables[arm,name](*values))
                marker['model_seconds']=time.perf_counter()-began;host=jax.device_get(result)
                path=args.output_dir/f"call-{report['model_calls_started']:02d}-{arm}-{name}.pkl"
                save_start=time.perf_counter();base.save(path,dict(scope=report['scope'],fixed_input=str(args.fixed_proposal),marker=marker,result=host,
                    saved_before_hard_or_strict_classification=True));marker['raw_save_seconds']=time.perf_counter()-save_start
                calls[key]=index+1;report['model_calls']+=1;report['model_call_markers'].append(marker)
                report['raw_model_results'].append(str(path));emit('raw_model_saved')
                value=dict(loss=host[0][0],auxiliary=host[0][1],gradient=host[1][0],initial_state_gradients=host[1][1])
                mask=defined[name]
                if name=='daily':mask=mask[context['daily_start']:context['daily_start']+4]
                hard=hard_gate(value,name,mask);base.write(path.with_suffix('.hard.json'),hard)
                if name!='daily':
                    components[context['repeat'],arm,name]=value
                    report['component_index'].append(dict(repeat=context['repeat'],arm=arm,component=name,path=str(path)))
                return result
            def daily_evaluator(arm):
                def evaluate(p,samples):
                    context['daily_start']=len(daily_states)*4
                    result=model(arm,'daily',p,samples.inputs.continuous_state,samples);daily_states.append(result[1][1])
                    return result[0],result[1][0]
                return evaluate
            accumulators={arm:learning.make_learning_daily_accumulator(jax=jax,jnp=jnp,microbatch_size=4,
                evaluator=daily_evaluator(arm),expected_count=32,slice_execution='leaf') for arm in ('reference','candidate')}
            for repeat,order in enumerate(ORDERS):
                context.clear();context['repeat']=repeat
                for arm in order:
                    daily_states.clear()
                    with jax.default_device(device),device_helper_cache(device):result=accumulators[arm](parameters,daily)
                    if len(daily_states)!=8:raise ValueError('requires eight distinct daily slices')
                    value=jax.device_get(dict(loss=result[0][0],auxiliary=result[0][1],gradient=result[1],initial_state_gradients=tuple(daily_states)))
                    path=args.output_dir/f'repeat-{repeat}-{arm}-daily32.pkl';base.save(path,dict(scope=report['scope'],result=value,
                        fixed_input=str(args.fixed_proposal),saved_before_hard_or_strict_classification=True))
                    base.write(path.with_suffix('.hard.json'),hard_gate(value,'daily',defined['daily']))
                    components[repeat,arm,'daily']=value;report['component_index'].append(dict(repeat=repeat,arm=arm,component='daily32',path=str(path)))
                    context.pop('daily_start',None);model(arm,'T7',parameters,*cases[7])
                model('reference','T30',parameters,*cases[30])
            if report['model_calls']!=38:raise ValueError('exact38 call contract failed')
            def old_coordinate_evidence(actual,expected):
                aa={ '/'.join(path):value for path,kind,value in entries(actual) if kind=='numeric'}
                bb={ '/'.join(path):value for path,kind,value in entries(expected) if kind=='numeric'}
                rows=[]
                for leaf in old_indices:
                    indices=np.asarray([row['index'] for row in leaf['failed_indices']],dtype=int)
                    index=tuple(indices[:,axis] for axis in range(indices.shape[1]))
                    a,b=aa[leaf['path']][index],bb[leaf['path']][index]
                    row=compare(a,b);row.update(path=leaf['path'],indices=indices.tolist(),actual=a.tolist(),reference=b.tolist())
                    rows.append(row)
                return dict(scope='restricted to retained old977 parameter coordinates',failed=sum(row['failed'] for row in rows),per_leaf=rows)
            def comparison(actual,expected):
                state=compare(np.asarray(actual['initial_state_gradients'],dtype=np.float64),np.asarray(expected['initial_state_gradients'],dtype=np.float64))
                return dict(gradient=compare(actual['gradient'],expected['gradient'],per_leaf=True),
                    loss_auxiliary=compare((actual['loss'],actual['auxiliary']),(expected['loss'],expected['auxiliary'])),state=state,
                    old977=old_coordinate_evidence(actual['gradient'],expected['gradient']))
            comparisons={'cross':{},'self':{}}
            for repeat in range(2):
                for name in ('daily','T7'):
                    comparisons['cross'][f'repeat{repeat}:{name}']=comparison(components[repeat,'candidate',name],components[repeat,'reference',name])
            for arm in ('reference','candidate'):
                for name in ('daily','T7'):
                    comparisons['self'][arm+':'+name]=comparison(components[1,arm,name],components[0,arm,name])
            comparisons['T30_repeat']=comparison(components[1,'reference','T30'],components[0,'reference','T30'])
            base.write(args.output_dir/'branch-comparisons.json',comparisons)
            mixed={};mixed_records={}
            def mix(label,repeat,arm,thirty_repeat):
                with jax.default_device(device),device_helper_cache(device):
                    value=jax.device_get(jax.block_until_ready(operations['combine_daily_mean'](
                        jax.device_put(components[repeat,arm,'daily']['gradient']),jax.device_put(components[repeat,arm,'T7']['gradient']),
                        jax.device_put(components[thirty_repeat,'reference','T30']['gradient']))))
                path=args.output_dir/(label+'.mixed.pkl');base.save(path,dict(scope='exact canonical device combined gradient, no update',
                    repeat=repeat,arm=arm,T30_repeat=thirty_repeat,gradient=value,saved_before_classification=True))
                if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(value)):raise ValueError('mixed nonfinite after save')
                mixed[label]=value;return value
            for repeat in range(2):
                for arm in ('reference','candidate'):mix(f'sharedT30-repeat{repeat}-{arm}',repeat,arm,repeat)
                a=mixed[f'sharedT30-repeat{repeat}-candidate'];b=mixed[f'sharedT30-repeat{repeat}-reference']
                mixed_records[f'sharedT30-cross-repeat{repeat}']=dict(full=compare(a,b,per_leaf=True),old977=old_coordinate_evidence(a,b))
            # Assign distinct T30 repeats to mimic an arm-specific control, using fixed repeat0 daily/T7.
            mix('assignedT30-reference',0,'reference',0);mix('assignedT30-candidate',0,'candidate',1)
            a,b=mixed['assignedT30-candidate'],mixed['assignedT30-reference']
            mixed_records['assignedT30-cross']=dict(full=compare(a,b,per_leaf=True),old977=old_coordinate_evidence(a,b))
            for arm in ('reference','candidate'):
                with (args.old_family/'model'/f'pair-2-{arm}.full-proposal.pkl').open('rb') as stream:old=pickle.load(stream)
                old_gradient=old['result']['gradient']
                for repeat in range(2):
                    fresh=mixed[f'sharedT30-repeat{repeat}-{arm}']
                    mixed_records[f'fresh-repeat{repeat}-{arm}-vs-retained']=dict(full=compare(fresh,old_gradient,per_leaf=True),old977=old_coordinate_evidence(fresh,old_gradient))
                del old,old_gradient
            base.write(args.output_dir/'mixed-comparisons.json',mixed_records)
            final_memory=device.memory_stats() or {}
            if final_memory.get('peak_bytes_in_use',24*2**30+1)>24*2**30:raise MemoryError('post-helper mixed peak exceeds24GiB')
            emit('completed',status='branch_diagnostic_completed_not_adopted',model_compile_counts=model_compile_counts,
                final_device_memory=final_memory,post_helper_memory_gate_passed=True,
                helper_backend_calls=helper_count['backend'],cross_parameter_failures={k:v['gradient']['failed'] for k,v in comparisons['cross'].items()},
                self_parameter_failures={k:v['gradient']['failed'] for k,v in comparisons['self'].items()},
                T30_repeat_parameter_failures=comparisons['T30_repeat']['gradient']['failed'],
                mixed_parameter_failures={k:v['full']['failed'] for k,v in mixed_records.items()},
                causal_limit='fresh component evidence is not a forensic decomposition of unavailable retained component gradients; original A/B rejection remains')
            return 0
        finally:compiler.backend_compile=original_backend
    except BaseException as error:
        emit('exception',status='runtime_or_hard_failure',error_type=type(error).__name__,error=str(error));return 1
    finally:timer.cancel()

if __name__=='__main__':raise SystemExit(main())
