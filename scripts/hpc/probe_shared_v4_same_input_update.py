"""Six saved-binary full proposals: same inputs per pair, reference trajectory.

Uses the official daily32 mean + T7 + T30 update without new model AD graphs.
Strict A/B discrepancies remain visible and do not suppress subsequent pairs
when hard ABI/finite/physical/memory/budget checks pass. No training/adoption.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time
import numpy as np

ROOT=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
DIAG=ROOT/'runtime/outputs/diagnostics'
ORDERS=(('reference','candidate'),('candidate','reference'),('reference','candidate'))
RTOL,ATOL=1e-5,1e-6

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def save(path,value):
    if path.exists():raise FileExistsError('preserve raw proposal: '+str(path))
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)

def paired_protocol(initial,execute,preserve,classify,advance,emit):
    """Exactly three pairs; only reference advances their common input."""
    current=initial;records=[]
    for index,order in enumerate(ORDERS):
        results={}
        for arm in order:
            result=execute(arm,current,index)
            preserve(index,arm,result)
            results[arm]=result
            if not result['hard_passed']:raise ValueError('hard failure after raw save; following arm not run')
        evidence=classify(results['candidate'],results['reference'])
        records.append(dict(pair=index,order=list(order),input_relation='same parameters and AdamW state within pair',evidence=evidence))
        emit(index,records,results)
        current=advance(results['reference'])
    return records,current

def contract_check():
    calls=[];saved=[];emitted=[]
    def execute(arm,current,pair):
        calls.append((pair,arm,current));return dict(hard_passed=True,next=current+1+(100 if arm=='candidate' else 0))
    records,last=paired_protocol(0,execute,lambda i,a,r:saved.append((i,a)),
        lambda a,b:dict(A_passed=False,B_passed=False),lambda r:r['next'],lambda i,*_:emitted.append(i))
    if len(calls)!=6 or last!=3 or [x[2] for x in calls]!=[0,0,1,1,2,2] or len(saved)!=6 or emitted!=[0,1,2]:
        raise ValueError('same-input/ref-only/six-proposal contract failed')
    stopped=[]
    try:
        paired_protocol(0,lambda a,c,i:dict(hard_passed=False),lambda i,a,r:stopped.append((i,a)),
            lambda *x:None,lambda r:0,lambda *x:None)
    except ValueError:pass
    else:raise ValueError('hard failure negative control accepted')
    if stopped!=[(0,'reference')]:raise ValueError('hard failed proposal not saved first')
    return dict(status='passed',same_input_each_pair=True,reference_only_advance=True,
        strict_A_and_B_failures_continue_all_six=True,hard_failure_saves_then_stops=True,
        model_calls=0,GPU_calls=0,Teacher_reads=0)

def defaults():
    update=ROOT/'runtime/transfer/shared-update-changing-slice-20260930'
    original=ROOT/'runtime/transfer/shared-original-dual-slice-update-20261002'
    return dict(tree_owner_file=update/'probe_shared_update_tree_slice.py',
        slice_helper_file=update/'shared_daily_microbatch.py',learning_runner_file=update/'run_shared_learning.py',
        support_owner_file=ROOT/'runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py',
        original_owner_file=original/'probe_shared_original_dual_slice_update.py',
        source_profile=DIAG/'shared-split-profile-c599236.json',input_bundle=DIAG/'shared-split-dual-inputs-c599236.pkl',
        daily_admission_report=DIAG/'shared-combined-real-daily-20260925.json',
        seven_admission_report=DIAG/'shared-combined-recurrent-T7-20260926.json',
        thirty_admission_report=DIAG/'shared-combined-recurrent-T30-20260926.json',
        runtime_preflight=DIAG/'shared-daily-ffn-precision-scale4096-20261001-preflight.json',
        daily_reference_report=DIAG/'shared-chunk-vmap-daily-dual-v4-20261001-reference.json',
        daily_reference_executable=DIAG/'shared-chunk-vmap-daily-dual-v4-20261001.reference.executable.pkl',
        t7_reference_report=DIAG/'shared-recurrent-checkpoint-policy-20260930-v2-T7-reference.json',
        t7_reference_executable=DIAG/'shared-recurrent-checkpoint-policy-20260930-v2-T7.reference.executable.pkl',
        t30_reference_report=DIAG/'shared-original-t30-dual-reference-20261001.json',
        t30_reference_executable=DIAG/'shared-original-t30-dual-reference-20261001.reference.executable.pkl',
        daily_candidate_report=DIAG/'shared-chunk-vmap-daily-dual-v4-20261001-candidate.json',
        daily_candidate_executable=DIAG/'shared-chunk-vmap-daily-dual-v4-20261001.candidate.executable.pkl',
        t7_candidate_report=DIAG/'shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4.json',
        t7_candidate_executable=DIAG/'shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4.candidate.executable.pkl')

def candidate_metadata(args,reference,bundle,digest):
    records={}
    for component in ('daily','t7'):
        report=json.loads(getattr(args,component+'_candidate_report').read_text());identity=report['identity']
        checkpoint=identity.get('initial_checkpoint',identity.get('checkpoint'))
        if (identity['method']!=reference['method'] or checkpoint!=bundle['initial_checkpoint']
                or identity.get('artifact_id')!=742 or identity.get('source_git_head',identity.get('source_head'))!=reference['source_git_head']
                or report['rtol']!=RTOL or report['atol']!=ATOL):raise ValueError('failed candidate provenance differs')
        valid=report['validation'];hard=valid['hard_counts']
        expected_names={'daily':'shared-chunk-vmap-daily-dual-v4-20261001-candidate.json',
            't7':'shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4.json'}
        if getattr(args,component+'_candidate_report').name!=expected_names[component]:raise ValueError('candidate report identity differs')
        comparisons=report['comparisons']
        A_keys=('loss','auxiliary','all_parameter_gradients') if component=='daily' else ('loss_and_auxiliary','all_parameter_gradients')
        expected_coordinates={'loss':1,'auxiliary':160,'loss_and_auxiliary':20,'all_parameter_gradients':64884481}
        if any(comparisons[key]['passed'] is not True or comparisons[key]['failed']!=0
                or comparisons[key]['coordinates']!=expected_coordinates[key]
                or comparisons[key]['nonfinite_actual']!=0 or comparisons[key]['nonfinite_reference']!=0 for key in A_keys):
            raise ValueError('historical candidate public loss/aux/full parameter checks differ')
        if (valid['gradient_coordinates']!=64884481 or valid['gradient_leaves']!=300
                or not valid['initial_state_gradient_finite'] or any(hard.values())):raise ValueError('candidate hard scope differs')
        if component=='daily':
            if identity['original_rows']['row_indices']!=[0,1,2,3]:raise ValueError('candidate daily saved row identity differs')
            if 'v4' not in str(report['variant']).lower() and 'whole' not in str(report['variant']).lower():raise ValueError('candidate daily variant differs')
            if report['state_is_optimizer_parameter'] is not False or valid['initial_state_gradient_coordinates']!=15416:
                raise ValueError('candidate daily state ABI differs')
        else:
            if identity['horizon']!=7 or identity['source_profile_sha256']!=digest or identity['execution_adapter_sha256']!='8a3cc2940a23ced4b5e1c3f0ed7093a29005851771db2f95f784d35e4c399a51':
                raise ValueError('candidate T7 adapter/profile differs')
        records[component]=dict(path=str(getattr(args,component+'_candidate_report')),historical_status=report['status'],
            historical_A_checks={key:comparisons[key] for key in A_keys},prior_state_strict_failure_preserved=True,new_admission=False)
    return records

def learning_interface(args):
    learning=importlib.import_module('scripts.hpc.run_shared_learning')
    signature=inspect.signature(learning.make_learning_daily_accumulator)
    if 'slice_execution' not in signature.parameters or Path(learning.__file__).resolve()!=args.learning_runner_file.resolve():
        raise ValueError('actual imported learning overlay/slice_execution signature differs')
    return learning,dict(path=str(Path(learning.__file__).resolve()),signature=str(signature),slice_execution_present=True)

def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();report=dict(status='running',scope='three_same_input_pairs_reference_only_trajectory_daily32_T7_originalT30_AdamW',
        completed_pairs=0,model_calls=0,model_calls_started=0,model_AD_compiles=0,retained_training_updates=0,production_adoption=False,
        rtol=RTOL,atol=ATOL,state_gradients_enter_optimizer=False,pairs=[],model_call_markers=[])
    def emit(stage,**values):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-started,**values);write(args.output_dir/'model.json',report);print(stage,flush=True)
    def expired():os._exit(124)
    timer=threading.Timer(args.process_seconds,expired);timer.daemon=True;timer.start()
    try:
        from probe_shared_normalized_keys_recurrent import load,validate_inputs
        original=load(args.original_owner_file,'v4_same_input_original_owner')
        tree=load(args.tree_owner_file,'v4_same_input_tree_owner')
        args.inspect_only=False
        profile,reference,bundle,digest,_admissions=tree._read_identity(args)
        import jax
        import jax.numpy as jnp
        from jax._src import compiler
        from jax.experimental.serialize_executable import deserialize_and_load
        from scripts.hpc.checked_recurrent_execution import device_helper_cache
        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
        from analyze_shared_v4_ab_offline import compare as host_compare
        learning,learning_signature=learning_interface(args)
        if jax.__version__!='0.4.38' or len(jax.local_devices())!=1 or jax.local_devices()[0].platform!='gpu':raise ValueError('canonical one GPU JAX0.4.38 required')
        device=jax.local_devices()[0]
        preflight=json.loads(args.runtime_preflight.read_text())
        if preflight['status']!='passed' or not preflight['passed'] or preflight['actual_backend']!='gpu' or preflight['jax_version']!=jax.__version__:
            raise ValueError('unchanged runtime evidence differs')
        old={}
        for name in ('daily','T7','T30'):
            old[name]=original.admit_original(getattr(args,name.lower()+'_reference_report'),getattr(args,name.lower()+'_reference_executable'),name,
                reference=reference,bundle=bundle,digest=digest)
        candidate_records=candidate_metadata(args,reference,bundle,digest)
        checkpoint=learning.load_learning_checkpoint(Path(bundle['initial_checkpoint']),expected_method=reference['method'])
        if checkpoint['completed_updates']!=0:raise ValueError('requires real update0')
        abi=host_compare(checkpoint['parameters'],checkpoint['parameters'])
        if abi['numeric_leaves']!=300 or abi['coordinates']!=64884481:raise ValueError('parameter numeric/static ABI differs')
        parameters,optimizer=jax.device_put((checkpoint['parameters'],checkpoint['optimizer']))
        daily=jax.device_put(bundle['payloads'][0]['daily']);cases=jax.device_put(bundle['payloads'][0]['cases'])
        sample4=jax.tree_util.tree_map(lambda value:value[:4],daily)
        if any(np.shape(value)[0]!=32 for value in jax.tree_util.tree_leaves(daily)):raise ValueError('all32 distinct prepared rows required')
        initial_states=dict(daily32=np.asarray(daily.inputs.continuous_state),T7=np.asarray(cases[7][0]),T30=np.asarray(cases[30][0]))
        defined={name:np.isfinite(value)&(np.abs(value)<.5e20) for name,value in initial_states.items()}
        np.savez(args.output_dir/'declared-input-state-masks.npz',**initial_states,**{name+'_defined':mask for name,mask in defined.items()})
        emit('inputs_ready',source_head=reference['source_git_head'],source_profile_sha256=digest,
            method=reference['method'],checkpoint=bundle['initial_checkpoint'],artifact_id=742,parameter_ABI=abi,
            prepared_inputs_reused=True,original_admission_reports={k:str(getattr(args,k.lower()+'_reference_report')) for k in old},
            candidate_metadata=candidate_records,learning_interface=learning_signature,runtime_preflight_reused=str(args.runtime_preflight),
            masks=dict(provenance='input_assembly._defined52–54 unchanged finite & abs<0.5e20',
                undefined_counts={name:int((~mask).sum()) for name,mask in defined.items()},
                physical_source_masks='unchanged prepared bundle and saved binaries; exposed hard mask counters checked'),
            daily_slice_execution='leaf',daily_micro4_starts=list(range(0,32,4)))
        model_counts=dict(compile_or_get_cached_calls=0,backend_compile_calls=0)
        @contextmanager
        def no_model_compile():
            restore=tree.reject_compilation_before_entry(compiler,model_counts)
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
            if memory['estimated_live_bytes']>24*2**30:raise MemoryError('saved executable estimated memory exceeds24GiB')
            executables[arm,name]=binary;loaded.append(dict(arm=arm,component=name,path=str(path),
                load_ABI_seconds=time.perf_counter()-began,memory=memory));del serialized
            emit('saved_binary_loaded',loaded_binaries=loaded)
        executables['candidate','T30']=executables['reference','T30']
        call_counts={name:0 for name in ('reference:daily','candidate:daily','reference:T7','candidate:T7','shared:T30')}
        helper_counts=dict(backend=0);original_backend=compiler.backend_compile
        def counted_backend(*values,**keywords):
            helper_counts['backend']+=1;return original_backend(*values,**keywords)
        compiler.backend_compile=counted_backend
        try:
            with jax.default_device(device),device_helper_cache(device):operations=learning.make_tree_operations(jax,jnp,32)
            update=make_update(reference['method']['daily_protocol'])
            daily_states={arm:[] for arm in ('reference','candidate')}
            def model(arm,name,*values):
                key=('shared' if name=='T30' else arm)+':'+name
                marker=dict(arm=arm,component=name,executable_call_index=call_counts[key],
                    actual_executable_first_call=call_counts[key]==0,shared_original_T30=name=='T30')
                report['model_calls_started']+=1
                emit('model_call_started',current_model_call=marker)
                began=time.perf_counter()
                with no_model_compile():result=jax.block_until_ready(executables[arm,name](*values))
                marker['seconds']=time.perf_counter()-began;call_counts[key]+=1
                report['model_call_markers'].append(marker);report['model_calls']+=1
                return result
            def daily_evaluator(arm):
                def evaluate(p,samples):
                    result=model(arm,'daily',p,samples.inputs.continuous_state,samples)
                    daily_states[arm].append(result[1][1]);return result[0],result[1][0]
                return evaluate
            accumulators={arm:learning.make_learning_daily_accumulator(jax=jax,jnp=jnp,microbatch_size=4,
                evaluator=daily_evaluator(arm),expected_count=32,slice_execution='leaf') for arm in daily_states}
            raw_times={arm:[] for arm in ('reference','candidate')}
            def execute(arm,current,index):
                p,opt=current;daily_states[arm].clear();captured={};auxiliary={};states={};op=dict(operations)
                def combine(day,seven,thirty):
                    value=operations['combine_daily_mean'](day,seven,thirty);captured['gradient']=value;return value
                op['combine_daily_mean']=combine
                def accumulator(p,samples):
                    result=accumulators[arm](p,samples);auxiliary['daily']=result[0][1];return result
                wrapped={}
                for h in (7,30):
                    name=f'T{h}'
                    def recurrent(*values,_name=name):
                        result=model(arm,_name,*values);auxiliary[_name]=result[0][1];states[_name]=result[1][1];return result[0],result[1][0]
                    wrapped[name]=recurrent
                helpers_before=helper_counts['backend'];began=time.perf_counter()
                with jax.default_device(device),device_helper_cache(device):
                    new_p,new_opt,row=learning.execute_learning_update(jax=jax,parameters=p,optimizer=opt,
                        daily_samples=daily,seven_case=cases[7],thirty_case=cases[30],evaluators=wrapped,
                        operations=op,update=update,daily_accumulator=accumulator)
                    if len(daily_states[arm])!=8:raise ValueError('requires eight distinct complete daily micro4 derivatives')
                    states['daily32_micro4']=tuple(daily_states[arm])
                    jax.block_until_ready((captured['gradient'],new_p,new_opt,states))
                wall=time.perf_counter()-began;transfer_start=time.perf_counter()
                value=jax.device_get(dict(gradient=captured['gradient'],parameters=new_p,optimizer=new_opt,
                    update=row,auxiliary=auxiliary,initial_state_gradients=states))
                value.update(wall_seconds=wall,host_transfer_seconds=time.perf_counter()-transfer_start,
                    helper_backend_calls=helper_counts['backend']-helpers_before,pair=index,arm=arm)
                save(args.output_dir/f'pair-{index}-{arm}.full-proposal.pkl',dict(scope='discarded diagnostic proposal; not a training checkpoint',
                    source_head=reference['source_git_head'],method=reference['method'],artifact_id=742,
                    input_optimizer_step=index,same_inputs_within_pair=True,result=value,
                    saved_before_ABI_finite_physical_memory_and_strict_classification=True))
                # Compute hard checks now, classify strict A/B only after raw save.
                hard={name:tree._hard_counts(aux) for name,aux in auxiliary.items()}
                finite=all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves((value['gradient'],value['parameters'],value['optimizer'],value['initial_state_gradients'],value['auxiliary'],value['update']['loss'],value['update']['optimizer_gradient_norm'],value['update']['clip_scale'])))
                numeric=host_compare(value['gradient'],checkpoint['parameters'])
                host_compare(value['parameters'],checkpoint['parameters']);host_compare(value['optimizer'],checkpoint['optimizer'])
                if numeric['numeric_leaves']!=300 or numeric['coordinates']!=64884481:raise ValueError('proposal parameter/static ABI changed')
                zero_checks={}
                for name,mask_key in (('daily32_micro4','daily32'),('T7','T7'),('T30','T30')):
                    array=np.asarray(value['initial_state_gradients'][name])
                    expected_shape=(8,4,3854) if name=='daily32_micro4' else (3854,)
                    if array.shape!=expected_shape:raise ValueError('complete state derivative ABI changed')
                    zero_checks[name]=int(np.count_nonzero(array.reshape(defined[mask_key].shape)[~defined[mask_key]]))
                memory=device.memory_stats() or {};step=int(np.asarray(value['optimizer'].step))
                hard_passed=(finite and all(not any(v.values()) for v in hard.values())
                    and len(hard['daily'])==11 and len(hard['T7'])==len(hard['T30'])==14
                    and not any(zero_checks.values()) and memory.get('peak_bytes_in_use',24*2**30+1)<=24*2**30
                    and step==index+1 and not any(model_counts.values()))
                value.update(hard=hard,finite=finite,undefined_derivative_nonzero=zero_checks,
                    device_memory=memory,optimizer_step=step,hard_passed=hard_passed)
                return value
            def preserve(index,arm,result):
                raw_times[arm].append(result['wall_seconds'])
                write(args.output_dir/f'pair-{index}-{arm}.hard.json',{key:result[key] for key in
                    ('pair','arm','hard','finite','undefined_derivative_nonzero','device_memory','optimizer_step','hard_passed','wall_seconds','host_transfer_seconds','helper_backend_calls')})
                emit('raw_proposal_saved',last_raw_pair=index,last_raw_arm=arm)
            def classify(actual,expected):
                A={name:host_compare(actual[name],expected[name],per_leaf=True) for name in ('gradient','parameters','optimizer')}
                A['loss_auxiliary']=host_compare((actual['update']['loss'],actual['auxiliary']),
                    (expected['update']['loss'],expected['auxiliary']))
                A['clip_and_norm']=host_compare((actual['update']['optimizer_gradient_norm'],actual['update']['clip_scale']),
                    (expected['update']['optimizer_gradient_norm'],expected['update']['clip_scale']))
                B=original.state_comparison(jax,actual['initial_state_gradients'],expected['initial_state_gradients'])
                return dict(A=A,B=B,A_passed=all(row['passed'] for row in A.values()),B_passed=all(row['passed'] for row in B.values()),
                    tolerance_waiver=False,T30_interpretation='within-pair same-input shared-binary repeat control; cross-pair weights change')
            def pair_saved(index,records,results):
                write(args.output_dir/f'pair-{index}.json',records[-1])
                emit('complete_pair_saved',completed_pairs=index+1,pairs=records,
                    pair_times={arm:results[arm]['wall_seconds'] for arm in results},
                    helper_backend_calls=helper_counts['backend'],model_compile_counts=model_counts)
            emit('proposal_stage_ready',startup_before_first_proposal_seconds=time.perf_counter()-started,
                no_bootstrap_or_warm_model_calls=True,exact_full_proposals=6,exact_model_calls=60,
                first_pair_cold_effects_separate=True,model_compile_counts=model_counts)
            records,_final=paired_protocol((parameters,optimizer),execute,preserve,classify,
                lambda result:jax.device_put((result['parameters'],result['optimizer'])),pair_saved)
            # Raw times are already preserved. No extra measurements follow.
            warm={arm:values[1:] for arm,values in raw_times.items()}
            emit('completed',status='diagnostic_all_A_B_strict_passed_not_adopted' if all(r['evidence']['A_passed'] and r['evidence']['B_passed'] for r in records) else 'diagnostic_strict_discrepancy_not_adopted',
                completed_pairs=3,matched_seconds=raw_times,last_two_warm_seconds=warm,
                last_two_reference_over_candidate_ratio=float(np.median(warm['reference'])/np.median(warm['candidate'])),
                last_two_ranges={arm:[min(values),max(values)] for arm,values in warm.items()},
                speed_stability='unresolved: only two predeclared warm pairs; times/ranges reported without an extra stability gate',
                last_two_paired_ratios=[a/b for a,b in zip(warm['reference'],warm['candidate'],strict=True)],
                first_pair_speed_stability_claim=False,model_compile_counts=model_counts,helper_backend_calls=helper_counts['backend'])
            return 0
        finally:compiler.backend_compile=original_backend
    except BaseException as error:
        emit('exception',status='runtime_or_hard_failure',error_type=type(error).__name__,error=str(error));return 1
    finally:timer.cancel()

def metadata_check(args):
    from probe_shared_normalized_keys_recurrent import load
    tree=load(args.tree_owner_file,'v4_same_input_cpu_tree_owner')
    original=load(args.original_owner_file,'v4_same_input_cpu_original_owner')
    args.inspect_only=True
    _profile,reference,bundle,digest,_admissions=tree._read_identity(args)
    for name in ('daily','T7','T30'):
        original.admit_original(getattr(args,name.lower()+'_reference_report'),getattr(args,name.lower()+'_reference_executable'),name,
            reference=reference,bundle=bundle,digest=digest)
    records=candidate_metadata(args,reference,bundle,digest)
    _learning,learning_signature=learning_interface(args)
    for component in ('daily','t7'):
        if not getattr(args,component+'_candidate_executable').is_file():raise FileNotFoundError('saved candidate binary missing')
    preflight=json.loads(args.runtime_preflight.read_text())
    if preflight['status']!='passed' or not preflight['passed'] or preflight['actual_backend']!='gpu' or preflight['jax_version']!='0.4.38':
        raise ValueError('reused canonical runtime evidence differs')
    return dict(status='passed',candidate_records=records,learning_interface=learning_signature,artifact_id=int(bundle['artifact_ids'][0]),
        source_head=reference['source_git_head'],daily_rows=32,saved_binaries_loaded=0,
        GPU_calls=0,Teacher_reads=0,new_model_evaluations=0,new_model_AD_compiles=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,value in defaults().items():parser.add_argument('--'+name.replace('_','-'),type=Path,default=value)
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--process-seconds',type=float,default=1790)
    parser.add_argument('--cli-check-only',action='store_true');parser.add_argument('--contract-check',action='store_true')
    parser.add_argument('--metadata-check',action='store_true')
    args=parser.parse_args()
    if args.cli_check_only:print(json.dumps(dict(status='passed',JAX_imports=int('jax' in sys.modules),GPU_calls=0)));return 0
    if args.contract_check:print(json.dumps(contract_check(),indent=2));return 0
    if args.metadata_check:print(json.dumps(metadata_check(args),indent=2));return 0
    if args.output_dir is None or not 0<args.process_seconds<=1790:parser.error('requires output and process allowance<=1790')
    return run(args)

if __name__=='__main__':raise SystemExit(main())
