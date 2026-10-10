"""Unapproved bank-adjoint T7 full native parameter/state AD diagnostic.

Preparation modes inspect JSON/file metadata only. Actual execution requires
the separately approved GPU route; no reference compilation is implemented.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import dataclasses
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
HEAD='c599236bcf57b745e9bafed678298010f4c3b58f'
MEMORY_LIMIT=24*2**30
COORDINATES=(33,199,428,200)
RTOL,ATOL=1e-5,1e-6

def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n')
    os.replace(temporary,path)

def write_pickle(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary,path)

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def paths(root):
    diag=root/'runtime/outputs/diagnostics'
    stem=diag/'shared-recurrent-checkpoint-policy-20260930-v2-T7'
    return {'reference_report':Path(str(stem)+'-reference.json'),
        'reference_result':Path(str(stem)+'.reference.pkl'),
        'reference_executable':Path(str(stem)+'.reference.executable.pkl'),
        'JVP_report':diag/'shared-original-state-jvp-20261002/model.json',
        'JVP_results':diag/'shared-original-state-jvp-20261002/model.directional-results.pkl',
        'source_profile':diag/'shared-split-profile-c599236.json',
        'input_bundle':diag/'shared-split-dual-inputs-c599236.pkl',
        'prepared_context':diag/'shared-expert-real-daily-context-c599236.pkl',
        'daily_admission_report':diag/'shared-combined-real-daily-20260925.json',
        'seven_admission_report':diag/'shared-combined-recurrent-T7-20260926.json',
        'thirty_admission_report':diag/'shared-combined-recurrent-T30-20260926.json',
        'support_owner_file':root/'runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py',
        'mask_inventory_source':root/'runtime/transfer/shared-state-reference-fp64-20261002/probe_shared_state_reference_fp64.py',
        'decoder':root/'runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py',
        'backbone':root/'runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py'}

def load_mask_inventory(path):
    """Reuse only the audited pure helper; do not import its precision runner.

    Provenance: probe_shared_state_reference_fp64.py, mask_inventory, lines49–74.
    The complete source hash pins this extraction to the previously staged helper.
    """
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!='34da692a6d10f42a19a1ca63cd38b2a089a3db7af3237484f629ccdbc43ce843':
        raise ValueError('existing pure mask-inventory helper source differs')
    tree=ast.parse(source,filename=str(path))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='mask_inventory']
    if len(functions)!=1:raise ValueError('expected one original mask_inventory function')
    namespace={'np':np,'dataclasses':dataclasses,'hashlib':hashlib}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['mask_inventory']

def inspect_assets(root):
    """No pickle, JAX import, GPU init or real-model input read."""
    files=paths(root)
    missing=[str(value) for value in files.values() if not value.is_file()]
    if missing:
        raise FileNotFoundError('required existing assets missing; no reference fallback: '+', '.join(missing))
    reference=json.loads(files['reference_report'].read_text())
    if reference.get('status')!='passed' or reference.get('mode')!='reference' or reference.get('horizon')!=7:
        raise ValueError('original full-AD T7 reference report is not admitted')
    identity=reference['identity']
    if identity['source_head']!=HEAD or identity['horizon']!=7 or identity['artifact_id']!=742:
        raise ValueError('original full-AD T7 identity differs')
    if reference.get('initial_state_gradient_coordinates')!=3854 or not reference.get('parameter_and_initial_state_gradients'):
        raise ValueError('reference full parameter/state scope differs')
    jvp=json.loads(files['JVP_report'].read_text())
    if jvp.get('status')!='native_jvp_vjp_discrepancy' or jvp.get('identity')!=identity or jvp.get('jvp_calls')!=32:
        raise ValueError('saved original JVP evidence differs/incomplete')
    return {'status':'metadata_ready','files':{key:{'path':str(value),'bytes':value.stat().st_size} for key,value in files.items()},
        'reference_identity':identity,'original_lower_seconds':reference['lower_seconds'],
        'original_backend_seconds':reference['compile_seconds'],
        'pickle_loads':0,'JAX_imports':0,'GPU_calls':0,'real_model_input_reads':0,
        'binary_input_ABI':'validated only on authorized actual-runtime deserialize; metadata inspection is not ABI acceptance'}

def strict_comparison(jax,actual,reference):
    # Same strict coordinate allowance as the existing original JVP/VJP gate.
    if jax.tree_util.tree_structure(actual)!=jax.tree_util.tree_structure(reference):
        raise ValueError('comparison tree differs')
    count=failed=0; maximum=0.0
    for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(reference),strict=True):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or a.dtype!=b.dtype:
            raise ValueError('comparison shape/dtype differs')
        error=np.abs(a.astype(np.float64)-b.astype(np.float64))
        valid=np.isfinite(a)&np.isfinite(b)&(error<=ATOL+RTOL*np.abs(b))
        failed+=int(np.count_nonzero(~valid));count+=a.size
        maximum=max(maximum,float(np.max(error,initial=0)))
    return {'passed':failed==0,'failed_coordinates':failed,'coordinates':count,'max_abs_error':maximum}

def compare_full(jax,actual,reference):
    return {'loss':strict_comparison(jax,actual[0][0],reference[0][0]),
        'auxiliary':strict_comparison(jax,actual[0][1],reference[0][1]),
        'all_parameters':strict_comparison(jax,actual[1][0],reference[1][0]),
        'all_initial_states':strict_comparison(jax,actual[1][1],reference[1][1])}

def hard_admission(jax,result,defined,parameters):
    finite=all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(result))
    hard={key:int(np.asarray(value)) for key,value in result[0][1]['hard'].items()}
    state=np.asarray(result[1][1])
    if state.shape!=(3854,) or state.dtype!=np.float64:
        raise ValueError('full initial-state derivative ABI differs')
    gradients=jax.tree_util.tree_leaves(result[1][0]); supplied=jax.tree_util.tree_leaves(parameters)
    if jax.tree_util.tree_structure(result[1][0])!=jax.tree_util.tree_structure(parameters):
        raise ValueError('parameter gradient tree differs')
    if any(np.shape(a)!=np.shape(b) or np.asarray(a).dtype!=np.asarray(b).dtype for a,b in zip(gradients,supplied,strict=True)):
        raise ValueError('parameter gradient shape/dtype differs')
    physical=len(hard)==14 and not any(hard.values())
    mask=int(np.asarray(result[0][1]['declared_mask_change']))==14
    undefined=np.count_nonzero(~defined)==21 and bool(np.all(state[~defined]==0))
    return {'passed':bool(finite and physical and mask and undefined),'finite':bool(finite),'hard_counts':hard,
        'physical_passed':physical,'declared_mask_change':int(np.asarray(result[0][1]['declared_mask_change'])),
        'declared_mask_count_unchanged':mask,'undefined_coordinates':int(np.count_nonzero(~defined)),
        'undefined_gradients_exact_zero':undefined,'parameter_leaves':len(gradients),
        'parameter_coordinates':sum(np.size(value) for value in gradients),'state_coordinates':state.size}

def jvp_supplement(result,rows):
    state=np.asarray(result[1][1])
    checks=[]
    for coordinate in COORDINATES:
        saved=[row for row in rows if row['coordinate']==coordinate]
        if len(saved)!=8 or not all(np.isfinite(row['jvp']) for row in saved):
            raise ValueError('saved four-coordinate JVP scope differs')
        for row in saved:
            value,reference=float(state[coordinate]),float(row['jvp'])
            error=abs(value-reference);allowance=ATOL+RTOL*abs(reference)
            checks.append({'coordinate':coordinate,'repeat':row['repeat'],'candidate_VJP':value,'original_FP32_JVP':reference,
                'absolute_error':error,'allowance':allowance,'passed':error<=allowance})
    return {'checks':checks,'all_passed':all(row['passed'] for row in checks),
        'role':'supplemental_only_no_strict_fullAD_waiver_no_reference_adoption'}

def save_first_candidate(jax,output_dir,identity,first,admission,baseline,saved_rows,emit):
    write_pickle(output_dir/'candidate.result.pkl',{'identity':identity,'result':first,'hard_admission':admission})
    if not admission['passed']:
        emit('completed',status='hard_admission_failed',initial_hard_admission=admission,
            stop_reason='candidate first hard-admission failure saved; strict/supplemental/pairs not evaluated')
        return None
    return compare_full(jax,first,baseline),[jvp_supplement(first,saved_rows)]

def collect_T7_pairs(jax,evaluate,original,candidate,baseline,first,saved_rows,identity,output_dir,emit,supplement):
    times={'reference':[],'candidate':[]};pairs=[];hard_pairs=[];self_drift={'reference':[],'candidate':[]}
    for repeat in range(3):
        results={};admissions_pair={}
        order=('reference','candidate') if repeat%2==0 else ('candidate','reference')
        for label in order:
            value,seconds,hard=evaluate(original if label=='reference' else candidate,f'{label}_pair_{repeat}')
            results[label]=value;times[label].append(seconds);admissions_pair[label]=hard
            if not hard['passed']:
                write_pickle(output_dir/f'pair-{repeat}.partial-{label}.pkl',{'identity':identity,'result':value,
                    'hard_admission':hard,'order':order,'evaluated_arms':list(results),'other_arm_evaluated':len(results)>1})
                emit('completed',status='hard_admission_failed',failed_pair=repeat,failed_arm=label,
                    paired_hard_admission=admissions_pair,matched_warm_seconds=times,
                    stop_reason='hard failure saved immediately; no following arm or pair evaluated')
                return None
            self_drift[label].append(strict_comparison(jax,value[1][1],baseline[1][1] if label=='reference' else first[1][1]))
        pair=compare_full(jax,results['candidate'],results['reference']);pairs.append(pair);hard_pairs.append(admissions_pair)
        supplement.append(jvp_supplement(results['candidate'],saved_rows))
        write_pickle(output_dir/f'pair-{repeat}.pkl',{'identity':identity,'reference_result':results['reference'],
            'candidate_result':results['candidate'],'comparisons':pair,'hard_admission':admissions_pair})
        emit('pair_saved',repeat=repeat,pair_comparisons=pairs,paired_hard_admission=hard_pairs,matched_warm_seconds=times,JVP_supplement=supplement)
    return times,pairs,hard_pairs,self_drift

def run(args,emit,report):
    # Imports and real-model assets remain behind the actual execution mode.
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax,devices=initialize_gpu_backend()
    from jax.experimental.serialize_executable import deserialize_and_load,serialize
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.run_shared_transformer_admission import memory_sizes
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
    from research.daily_coarse_graining.daily_operator import shared_process_head,shared_transformer as st
    import probe_shared_forward_reverse as forward
    from probe_shared_normalized_keys_recurrent import validate_inputs
    from probe_shared_command_buffer_loops import forbid_component_compile
    import shared_bank_adjoint_accumulation as bank_policy
    if jax.__version__!='0.4.38' or len(devices)!=1:
        raise ValueError('requires one canonical JAX0.4.38 GPU')
    preflight=json.loads(args.runtime_preflight.read_text())
    if preflight.get('status')!='passed' or not preflight.get('passed') or preflight.get('actual_backend')!='gpu' or preflight.get('jax_version')!=jax.__version__:
        raise ValueError('actual runtime preflight failed/differs')
    metadata=inspect_assets(args.root);files=paths(args.root)
    source=SimpleNamespace(**{key:files[key] for key in ('source_profile','input_bundle','prepared_context',
        'daily_admission_report','seven_admission_report','thirty_admission_report','support_owner_file')})
    _profile,ref,bundle,context,checkpoint_path,admissions=forward._assets(source)
    checkpoint=load_learning_checkpoint(checkpoint_path,expected_method=ref['method'])
    if checkpoint['completed_updates']!=0:
        raise ValueError('requires original update0 weights')
    host_parameters=checkpoint['parameters']
    leaves=jax.tree_util.tree_leaves(host_parameters)
    if len(leaves)!=300 or sum(np.size(value) for value in leaves)!=64884481:
        raise ValueError('original full parameter scope differs')
    case=bundle['payloads'][0]['cases'][7];state=np.asarray(case[0])
    if len(case)!=5 or state.shape!=(3854,) or state.dtype!=np.float64 or np.shape(case.days[0].continuous_state)[0]!=7:
        raise ValueError('original T7 input ABI differs')
    defined=np.isfinite(state)&(np.abs(state)<.5e20)
    if np.count_nonzero(~defined)!=21:
        raise ValueError('actual source-defined mask differs')
    mask_inventory=load_mask_inventory(files['mask_inventory_source'])
    mask_evidence,input_masks=mask_inventory(dict(case=case,layout=context['head'].layout,
        normalization=context['head'].input_normalization,state_scales=context['state_scales'],
        template=context['template'],contract=context['head'].contract))
    if len(input_masks)!=112:
        raise ValueError('original source/science mask leaf count differs')
    parameters=jax.device_put(host_parameters);arguments=(parameters,*jax.device_put(case))
    identity=dict(source_head=ref['source_git_head'],source_profile_sha256=hashlib.sha256(files['source_profile'].read_bytes()).hexdigest(),
        method=ref['method'],artifact_id=int(bundle['artifact_ids'][0]),horizon=7,checkpoint=str(checkpoint_path.resolve()),
        jax_version=jax.__version__,admitted_executable=admissions['T7']['executable'],
        candidate_decoder_sha256=hashlib.sha256(files['decoder'].read_bytes()).hexdigest(),
        candidate_backbone_sha256=hashlib.sha256(files['backbone'].read_bytes()).hexdigest())
    if identity!=metadata['reference_identity']:
        raise ValueError('full-AD original identity differs')
    with files['reference_result'].open('rb') as stream:saved=pickle.load(stream)
    with files['JVP_results'].open('rb') as stream:saved_jvp=pickle.load(stream)
    if saved['identity']!=identity or saved_jvp['identity']!=identity:
        raise ValueError('saved result/JVP identity differs')
    with forbid_component_compile(True):
        with files['reference_executable'].open('rb') as stream:serialized=pickle.load(stream)
        original=deserialize_and_load(*serialized)
        validate_inputs(jax,serialized,original,arguments)
        reference_memory=memory_sizes(original)
        if reference_memory['estimated_live_bytes']>MEMORY_LIMIT:raise MemoryError('original estimated memory exceeds24GiB')
        with report['budget'].stage(120,'original_first_replay',emit):
            baseline=jax.device_get(jax.block_until_ready(original(*arguments)))
    baseline_hard=hard_admission(jax,baseline,defined,host_parameters)
    original_peak=devices[0].memory_stats() or {}
    baseline_hard['observed_memory_passed']='peak_bytes_in_use' in original_peak and original_peak['peak_bytes_in_use']<=MEMORY_LIMIT
    baseline_hard['observed_whole_process_memory']=original_peak
    baseline_hard['passed']=bool(baseline_hard['passed'] and baseline_hard['observed_memory_passed'])
    if not baseline_hard['passed']:
        write_pickle(args.output_dir/'original-replay.pkl',{'identity':identity,'result':baseline,'hard_admission':baseline_hard})
        emit('completed',status='hard_admission_failed',stop_reason='original first replay hard admission failed');return
    bridge=compare_full(jax,baseline,saved['result'])
    if any(not bridge[key]['passed'] for key in ('loss','auxiliary','all_parameters')):
        raise ValueError('original full-AD loss/aux/parameter/hard bridge failed; state self-drift retained diagnostically')
    write_pickle(args.output_dir/'original-replay.pkl',{'identity':identity,'result':baseline,'bridge':bridge})
    write_pickle(args.output_dir/'source-defined-mask.pkl',{'identity':identity,'defined':defined,'input_boolean_masks':input_masks})
    emit('reference_replayed',identity=identity,reference_memory=reference_memory,original_saved_bridge=bridge,
        original_hard_admission=baseline_hard,input_masks=mask_evidence,source_mask_invariance='same112booleaninputs; unchanged source masks and only bank dtype boundaries',
        binary_input_ABI='original deserialize input treedef/shapes/dtypes validated',candidate_backend_attempts=0)
    owner=forward._load_module(args.root/'runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py','scripts.hpc.probe_shared_expert_real_daily')
    adapter=owner.load_staged_adapter(files['decoder']);backbone=owner.load_backbone_assembler(files['backbone'])
    macro=sys.modules['research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed']
    expert_decoder=adapter.__globals__.get('expert_decode_fields') if callable(adapter) else getattr(adapter,'expert_decode_fields',None)
    if expert_decoder is None or expert_decoder.__globals__ is not macro.__dict__:
        raise ValueError('actual expert adapter does not use audited macro globals')
    _rollout,objective=make_shared_recurrent_objective(context['head'],context['template'],context['state_scales'],context['science_scales'])
    backward=jax.jit(jax.value_and_grad(objective,argnums=(0,1),has_aux=True))
    with owner.candidate_process_head_patches(shared_process_head,adapter,backbone),bank_policy.patches(st,macro,expert=expert_decoder) as policy_evidence:
        with report['budget'].stage(args.lower_seconds,'candidate_lower',emit):
            began=time.perf_counter();lowered=backward.lower(*arguments);lower_seconds=time.perf_counter()-began
        emit('candidate_lowered',lower_seconds=lower_seconds,policy_evidence=policy_evidence,candidate_backend_attempts=0)
        with report['budget'].stage(args.backend_seconds,'candidate_backend',emit):
            emit('candidate_backend_start',candidate_backend_attempts=1)
            began=time.perf_counter();candidate=lowered.compile();compile_seconds=time.perf_counter()-began
            emit('candidate_backend_completed',compile_seconds=compile_seconds,policy_evidence=policy_evidence,candidate_backend_attempts=1)
    from probe_shared_bank_saved_binary_T7 import ReplayBudget,capture_actual_hlo
    remaining=lambda:report['budget'].limit-(time.perf_counter()-report['budget'].started)
    with ReplayBudget(remaining(),args.output_dir).stage(90,'candidate_serialize',emit):
        emit('candidate_serialize_start')
        serialized=serialize(candidate)
        validate_inputs(jax,serialized,candidate,arguments)
        write_pickle(args.output_dir/'candidate.executable.pkl',serialized)
        emit('candidate_binary_saved',serialized_bytes=(args.output_dir/'candidate.executable.pkl').stat().st_size)
    with ReplayBudget(remaining(),args.output_dir).stage(75,'compact_actual_HLO_export',emit):
        emit('compact_actual_HLO_export_start')
        proof=capture_actual_hlo(candidate,args.output_dir,policy_evidence)
    emit('candidate_compiled',compile_seconds=compile_seconds,policy_evidence=policy_evidence,optimized_bank_accumulation_evidence=proof,
        helper_aliases_restored=True,candidate_backend_attempts=1)
    if not proof['passed']:
        emit('completed',status='bank_accumulation_proof_incomplete',stop_reason='automatic scoped reverse/outer-carry proof incomplete; preserve binary/HLO for targeted offline review before numerical evaluation; no removal claim')
        return
    memory=memory_sizes(candidate)
    if memory['estimated_live_bytes']>MEMORY_LIMIT:raise MemoryError('candidate estimated memory exceeds24GiB')
    def evaluate(executable,label):
        with report['budget'].stage(120,label,emit),forbid_component_compile(True):
            began=time.perf_counter();value=executable(*arguments);jax.block_until_ready(value);seconds=time.perf_counter()-began
        value=jax.device_get(value)
        stats=devices[0].memory_stats() or {}
        hard=hard_admission(jax,value,defined,host_parameters)
        hard['observed_memory_passed']='peak_bytes_in_use' in stats and stats['peak_bytes_in_use']<=MEMORY_LIMIT
        hard['observed_whole_process_memory']=stats
        hard['passed']=bool(hard['passed'] and hard['observed_memory_passed'])
        return value,seconds,hard
    first,_seconds,admission=evaluate(candidate,'candidate_first_result')
    initial_saved=save_first_candidate(jax,args.output_dir,identity,first,admission,baseline,saved_jvp['rows'],emit)
    if initial_saved is None:return
    initial,supplement=initial_saved
    emit('candidate_first_result_saved',initial_comparisons=initial,initial_hard_admission=admission,candidate_memory=memory,
        JVP_supplement=supplement,observed_whole_process_memory=devices[0].memory_stats())
    # One untimed paired warmup, still under hard gates and shared budget.
    for label,executable in (('reference_warmup',original),('candidate_warmup',candidate)):
        value,_seconds,hard=evaluate(executable,label)
        if not hard['passed']:
            write_pickle(args.output_dir/(label+'.partial.pkl'),{'identity':identity,'result':value,'hard_admission':hard})
            emit('completed',status='hard_admission_failed',stop_reason=label+' hard admission failed; partial result saved');return
        del value
    collected=collect_T7_pairs(jax,evaluate,original,candidate,baseline,first,saved_jvp['rows'],identity,args.output_dir,emit,supplement)
    if collected is None:return
    times,pairs,hard_pairs,self_drift=collected
    strict=all(row['passed'] for row in initial.values()) and all(all(row['passed'] for row in pair.values()) for pair in pairs)
    ratio=float(np.median(times['reference'])/np.median(times['candidate']))
    emit('completed',status='diagnostic_passed_no_adoption' if strict and ratio>=1.05 else 'screening_failed',
        strict_numerical_passed=strict,speed_gate_passed=ratio>=1.05,speed_gate_minimum=1.05,median_ratio=ratio,
        initial_comparisons=initial,pair_comparisons=pairs,matched_warm_seconds=times,paired_hard_admission=hard_pairs,
        state_self_drift=self_drift,self_drift_waiver=False,JVP_supplement=supplement,
        candidate_memory=memory,reference_memory=reference_memory,observed_whole_process_memory=devices[0].memory_stats(),
        full_optimizer_updates=0,production_adoption=False)

class Budget:
    def __init__(self,seconds):self.started=time.perf_counter();self.limit=seconds
    @contextmanager
    def stage(self,cap,label,emit):
        remaining=self.limit-(time.perf_counter()-self.started)
        allowed=min(cap,remaining)
        if allowed<=0:raise TimeoutError('global model budget exhausted')
        began=time.perf_counter()
        emit('stage_start',stage_label=label,stage_cap_seconds=cap,effective_timeout_seconds=allowed)
        def expired():
            try:getattr(emit,'timeout',emit)('stage_timeout',status='timeout',stage_label=label,effective_timeout_seconds=allowed,stage_elapsed_seconds=time.perf_counter()-began)
            finally:os._exit(124)
        timer=threading.Timer(allowed,expired);timer.daemon=True;timer.start()
        try:yield
        finally:
            timer.cancel();emit('stage_end',stage_label=label,stage_elapsed_seconds=time.perf_counter()-began)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--runtime-preflight',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--lower-seconds',type=float,default=300)
    parser.add_argument('--backend-seconds',type=float,default=2800)
    parser.add_argument('--process-seconds',type=float,default=3590)
    parser.add_argument('--cli-check-only',action='store_true')
    parser.add_argument('--inspect-assets-only',action='store_true')
    args=parser.parse_args()
    if not (0<args.lower_seconds<=300 and 0<args.backend_seconds<=2800 and 0<args.process_seconds<=3590):
        parser.error('budgets must be positive and within300/2800/3590s')
    if args.cli_check_only:
        print('CLI checked; no JAX/GPU/model/data imports');return 0
    if args.inspect_assets_only:
        print(json.dumps(inspect_assets(args.root),indent=2));return 0
    if args.output_dir is None or args.runtime_preflight is None:parser.error('actual execution requires output-dir and runtime-preflight')
    args.output_dir.mkdir(exist_ok=False,parents=True)
    report={'status':'running','precision':'FP32 projection/attention/FFN; only4axis/local banks F64 across native map transpose',
        'scope':'T7_original_public_full64884481_parameter_and3854_state_native_AD','candidate_backend_attempts':0,
        'training_updates':0,'production_adoption':False,'strict_rtol':RTOL,'strict_atol':ATOL,
        'stage_events':[],'budget':Budget(args.process_seconds)}
    emit_lock=threading.RLock()
    def emit(stage,**fields):
        with emit_lock:
            event={'event':stage,'elapsed_seconds':time.perf_counter()-report['budget'].started,**fields}
            report['stage_events'].append(event);report.update(stage=stage,**fields)
            write_json(args.output_dir/'model.json',{key:value for key,value in report.items() if key!='budget'})
            print(f'bank_adjoint stage={stage} elapsed={event["elapsed_seconds"]:.1f}s',flush=True)
    def timeout_emit(stage,**fields):
        # A busy main-thread writer cannot delay forced termination.
        if emit_lock.acquire(blocking=False):
            try:emit(stage,**fields)
            finally:emit_lock.release()
    emit.timeout=timeout_emit
    # Global timer also covers setup/serialization/host comparisons, not just stages.
    def global_expired():
        try:timeout_emit('process_timeout',status='timeout')
        finally:os._exit(124)
    timer=threading.Timer(args.process_seconds,global_expired);timer.daemon=True;timer.start()
    try:
        emit('starting');run(args,emit,report)
    except BaseException as error:
        emit('exception',status='resource_failed' if isinstance(error,MemoryError) else 'runtime_or_admission_failed',error=str(error));raise
    finally:timer.cancel()
    return 0 if report['status']=='diagnostic_passed_no_adoption' else 1

if __name__=='__main__':raise SystemExit(main())
