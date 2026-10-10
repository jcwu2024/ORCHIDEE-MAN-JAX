"""Additional-budget-only T7 acceptance of the exact saved bank64 binary.

No model objective is constructed, lowered or compiled. All strict numerical
and hard-admission routines are shared with the original approved probe.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace
import probe_shared_bank_adjoint_T7 as original
import shared_bank_adjoint_accumulation as policy

ROOT=original.ROOT
SOURCE_FAMILY=ROOT/'runtime/outputs/diagnostics/shared-bank-adjoint-accumulation-T7-20261002'
CANDIDATE=SOURCE_FAMILY/'model/candidate.executable.pkl'
COMPILE_REPORT_SHA256='95357c6d5acaee22fd0440bf629ee9add20aaa4243f16539d175c108bf1bcddf'

def validate_policy_evidence(evidence):
    """Validate the complete compile-time bank policy before any GPU setup.

    The pinned report was captured inside patches(), so its policy status is
    active; every recorded application must already be complete.
    """
    if evidence['status']!='active' or evidence['overlay']!='native_ad_projected_bank64_outer_map_closure':
        raise ValueError('compile-time bank policy status/overlay differs')
    for key in ('baseline_helper_identity_passed','macro_imported_helper_alias_passed',
                'expert_decode_fields_globals_is_macro_dict','x64_enabled'):
        if evidence[key] is not True:raise ValueError('compile-time policy identity flag differs: '+key)
    for name,relative,digest in (('shared_transformer',policy.ST_RELATIVE_PATH,policy.ST_SHA256),
                                ('shared_decoder_macro_packed',policy.MACRO_RELATIVE_PATH,policy.MACRO_SHA256)):
        source=evidence['source_files'][name]
        if source['path']!=policy.DEPLOYED_SOURCE_PATHS[relative] or source['sha256'].lower()!=digest.lower():
            raise ValueError('compile-time source identity differs: '+name)
    clone=evidence['source_clone'];gathers=['bank_adjoint/gather_'+field for field in policy.BANK_FIELDS]
    if (clone['source_path']!=policy.DEPLOYED_SOURCE_PATHS[policy.ST_RELATIVE_PATH]
        or clone['source_sha256'].lower()!=policy.ST_SHA256.lower() or clone['transformed'] is not True
        or clone['gather_rewrite_counts']!={expression:1 for expression in policy.GATHER_EXPRESSIONS}
        or clone['gather_scope_markers']!=gathers):
        raise ValueError('compile-time source clone/gather rewrite differs')
    if evidence['scope_markers']!={'macro_map':policy.MACRO_SCOPE,
        'bank_widen':['bank_adjoint/widen_'+field for field in policy.BANK_FIELDS],'bank_gather':gathers}:
        raise ValueError('compile-time named scopes differ')
    applications=evidence['applications']
    if not isinstance(applications,list) or not applications:raise ValueError('missing policy applications')
    def positive(value):return type(value) is int and value>0
    bank_total=0
    for application in applications:
        if application['status']!='complete' or application['float32_narrowing_before_concat'] is not True:
            raise ValueError('incomplete bank policy application')
        if any(not positive(application[key]) for key in ('rows','chunk_size','effective_chunk_size',
                    'chunk_count','padded_rows','projection_count','bank_count')):
            raise ValueError('invalid application counts')
        rows=application['rows'];chunk=min(rows,128);count=(rows+chunk-1)//chunk
        if (application['chunk_size']!=128 or application['effective_chunk_size']!=chunk
            or application['chunk_count']!=count or application['padded_rows']!=count*chunk
            or type(application['padding_rows']) is not int or application['padding_rows']!=count*chunk-rows):
            raise ValueError('original chunk/padding metadata differs')
        banks=application['banks'];blocks=application['projection_count']
        if not isinstance(banks,list) or len(banks)!=application['bank_count'] or len(banks)!=blocks*4:
            raise ValueError('incomplete projection banks')
        expected={(block,field) for block in range(blocks) for field in policy.BANK_FIELDS};seen=set()
        for bank in banks:
            pair=(bank['block_index'],bank['field']);shape=bank['shape']
            if type(bank['block_index']) is not int or pair not in expected or pair in seen:
                raise ValueError('duplicate/missing/unexpected bank field or block')
            seen.add(pair)
            if (bank['input_dtype']!='float32' or bank['closure_dtype']!='float64'
                or not isinstance(shape,list) or len(shape)!=2 or any(not positive(value) for value in shape)
                or not positive(bank['elements']) or bank['elements']!=math.prod(shape)):
                raise ValueError('bank shape/elements/precision differs')
        if seen!=expected:raise ValueError('missing bank fields')
        bank_total+=len(banks)
    return {'passed':True,'application_count':len(applications),'bank_records':bank_total,
        'captured_policy_status':'active','all_application_statuses':'complete'}

def load_compile_report(path):
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=COMPILE_REPORT_SHA256:raise ValueError('immutable original compile report SHA256 differs')
    report=json.loads(raw);validation=validate_policy_evidence(report['policy_evidence'])
    return report,{'compile_report_sha256':digest,'policy_evidence_validation':validation}

def inspect_only(root):
    if root.resolve()!=ROOT.resolve():raise ValueError('saved binary source requires exact declared project root')
    report,provenance=load_compile_report(SOURCE_FAMILY/'model/model.json')
    metadata=original.inspect_assets(root)
    ledger=json.loads((SOURCE_FAMILY/'actual-exit.json').read_text())
    ends=[event for event in report['stage_events'] if event['event']=='stage_end' and event.get('stage_label')=='candidate_backend']
    if len(ends)!=1 or ledger['candidate_backend_attempts']!=1 or ledger['exit_status']!=124:
        raise ValueError('expected exact one-backend completed/export-timeout provenance')
    if CANDIDATE.stat().st_size!=220794649:raise ValueError('saved candidate byte size differs')
    if report['identity']!=metadata['reference_identity']:raise ValueError('saved binary source identity differs')
    return {'status':'saved_binary_metadata_ready','candidate':str(CANDIDATE),'candidate_bytes':CANDIDATE.stat().st_size,
        'candidate_sha256':hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),'source_identity':report['identity'],
        'completed_backend_stage_seconds':ends[0]['stage_elapsed_seconds'],'original_runtime_preflight_reused':True,**provenance,
        'GPU_calls':0,'pickle_reads':0,'model_compilations':0,'actual_runtime_ABI':'pending authorized deserialize'}

class ReplayBudget:
    def __init__(self,seconds,destination):self.started=time.perf_counter();self.limit=seconds;self.destination=destination
    @contextmanager
    def stage(self,cap,label,emit):
        remaining=self.limit-(time.perf_counter()-self.started);allowed=min(cap,remaining)
        if allowed<=0:raise TimeoutError('480s family remaining exhausted')
        emit('stage_start',stage_label=label,stage_cap_seconds=cap,effective_timeout_seconds=allowed)
        began=time.perf_counter();timeout_path=self.destination/(label+'.timeout.json')
        # Separate stdlib-only process enforces native C++/gzip/GIL deadlines.
        code=('import json,os,signal,time;from pathlib import Path;'
              'time.sleep(float(__import__("sys").argv[1]));'
              'Path(__import__("sys").argv[3]).write_text(json.dumps({"stage":__import__("sys").argv[4],"exit":124,"reason":"external stage deadline"}));'
              'os.kill(int(__import__("sys").argv[2]),signal.SIGTERM)')
        watchdog=subprocess.Popen([sys.executable,'-B','-c',code,str(allowed),str(os.getpid()),str(timeout_path),label],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:yield
        finally:
            watchdog.terminate();watchdog.wait(timeout=5)
            emit('stage_end',stage_label=label,stage_elapsed_seconds=time.perf_counter()-began)

@contextmanager
def no_model_compilation(compiler,evidence):
    names=('compile_or_get_cached','backend_compile')
    saved={name:getattr(compiler,name) for name in names}
    def forbidden(*args,**kwargs):
        evidence['forbidden_compile_calls']+=1
        raise RuntimeError('saved-binary-only acceptance forbids every new compilation')
    for name in names:setattr(compiler,name,forbidden)
    try:yield
    finally:
        for name,value in saved.items():setattr(compiler,name,value)

def capture_actual_hlo(candidate,destination,policy_evidence,*,defer_audit=False):
    """Compact actual optimized runtime modules; preserve scopes and SSA names."""
    from jaxlib import xla_client
    options=xla_client._xla.HloPrintOptions.short_parsable()
    options.print_large_constants=False
    options.print_backend_config=False
    options.print_program_shape=False
    options.print_metadata=True
    options.print_operand_shape=False
    options.print_operand_names=True
    options.print_percent=True
    options.print_ids=True
    modules=candidate.runtime_executable().hlo_modules()
    if not modules or len(modules)>8:raise ValueError('expected1..8 actual optimized HLO modules')
    reports=[];total=0
    for index,module in enumerate(modules):
        text=module.to_string(options);encoded=text.encode();raw_bytes=len(encoded);total+=raw_bytes
        path=destination/f'candidate.compact-{index}.hlo'
        path.write_bytes(encoded)
        original.write_json(destination/f'candidate.compact-{index}.size.json',{'raw_bytes':raw_bytes,
            'saved_bytes':path.stat().st_size,'sha256':hashlib.sha256(encoded).hexdigest(),
            'cumulative_bytes':total,'soft_review_threshold_bytes':64*2**20,
            'above_soft_threshold':raw_bytes>64*2**20,'hard_text_size_cap':None,
            'all_metadata_retained':True,'truncated':False})
        reports.append({'path':str(path),'raw_printed_bytes':raw_bytes,'bytes':len(encoded),'sha256':hashlib.sha256(encoded).hexdigest(),
            'proof':{'passed':False,'status':'deferred_CPU_audit'} if defer_audit else policy.audit_gpu_hlo(text,policy_evidence)})
    return {'passed':all(row['proof']['passed'] for row in reports),'modules':reports,'audit_deferred':defer_audit,
        'large_constants_printed':False,'backend_config_printed':False,'scope_metadata_printed':True,
        'operand_names_printed':True,'full_as_text_calls':0,'compressed_fulltext_exports':0,
        'claim_limit':'field/deduplicated-bank-shape structural proof; not same-shaped occurrence-exhaustive'}

def resident_CPU_proof(captured,args,compiled_report,report,emit):
    """Audit saved actual IR on CPU; retain GPU objects during bounded repairs.

    A request file only asks to rerun the strict analyzer; it never supplies a
    manual passing verdict. Every child independently checks actual IR hashes.
    """
    request=args.output_dir/'proof-retry.request'
    manifest=args.output_dir/'actual-IR-manifest.json'
    original.write_json(manifest,{'modules':captured['modules'],'policy_evidence':compiled_report['policy_evidence'],
        'compile_report_sha256':COMPILE_REPORT_SHA256,'compile_report_path':str(SOURCE_FAMILY/'model/model.json')})
    attempts=[];attempt=0
    review_started=time.perf_counter()
    review_limit=min(220,max(0,report['budget'].limit-(review_started-report['budget'].started)-90))
    if review_limit<=5:return {'passed':False,'status':'insufficient_remaining_CPU_review','captured':captured}
    with report['budget'].stage(review_limit,'resident_CPU_structural_review',emit):
        while True:
            attempt+=1;output=args.output_dir/f'CPU-proof-{attempt}.json'
            remaining=report['budget'].limit-(time.perf_counter()-report['budget'].started)
            environment=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('audit_saved_bank_hlo_cpu.py')),
                str(manifest),str(output)],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                _stdout,stderr=process.communicate(timeout=max(.1,min(90,remaining,
                    review_limit-(time.perf_counter()-review_started)-5)))
            except subprocess.TimeoutExpired:
                process.kill();_stdout,stderr=process.communicate()
                emit('CPU_proof_timeout',CPU_proof_attempt=attempt,actual_IR_preserved=True)
                return {'passed':False,'status':'CPU_analyzer_timeout','attempts':attempts,'captured':captured}
            if process.returncode==0 and output.exists():
                proof=json.loads(output.read_text());attempts.append(proof)
                emit('CPU_structural_proof_result',CPU_proof_attempt=attempt,proof=proof)
                if proof['passed']:return {**proof,'attempts':attempts,'captured':captured}
            else:
                failed={'passed':False,'CPU_exit':process.returncode,'stderr':stderr.decode(errors='replace')[-4000:]}
                original.write_json(output,failed);attempts.append(failed)
                emit('CPU_structural_proof_result',CPU_proof_attempt=attempt,proof=failed)
            emit('resident_proof_repair_pending',actual_IR_manifest=str(manifest),retry_request=str(request),
                GPU_executables_retained=True,candidate_values=0,CPU_proof_attempt=attempt)
            # The outer watchdog bounds this wait by the same family deadline;
            # retain at least90s for original/candidate values and complete pairs.
            while not request.exists():
                remaining=report['budget'].limit-(time.perf_counter()-report['budget'].started)
                if remaining<=90 or time.perf_counter()-review_started>=review_limit-5:
                    return {'passed':False,'status':'proof_incomplete_remaining_reserved',
                    'attempts':attempts,'captured':captured,'remaining_seconds':remaining}
                time.sleep(.5)
            request.unlink()

def run(args,emit,report):
    metadata=inspect_only(args.root);files=original.paths(args.root)
    if metadata['candidate_sha256']!=args.candidate_sha256:raise ValueError('exact saved candidate digest differs')
    compiled_report,provenance=load_compile_report(SOURCE_FAMILY/'model/model.json')
    emit('compile_policy_verified_before_GPU_initialization',**provenance)
    saved_proof=None
    if args.saved_IR_proof_sha256:
        from saved_bank_ir_admission import validate_saved_IR_certificate
        saved_proof=validate_saved_IR_certificate(args.root,args.saved_IR_proof_sha256,args.candidate_sha256,metadata['source_identity'])
        emit('reviewed_actual_IR_proof_verified_before_GPU_initialization',optimized_bank_accumulation_evidence=saved_proof)
    preflight=json.loads((SOURCE_FAMILY/'preflight.json').read_text())
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax,devices=initialize_gpu_backend()
    from jax.experimental.serialize_executable import deserialize_and_load
    from jax._src import compiler
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.run_shared_transformer_admission import memory_sizes
    import probe_shared_forward_reverse as forward
    from probe_shared_normalized_keys_recurrent import validate_inputs
    if jax.__version__!='0.4.38' or len(devices)!=1 or not preflight['passed'] or preflight['actual_backend']!='gpu':
        raise ValueError('exact canonical GPU runtime/reused preflight differs')
    if args.artifact_only_repair:
        # Unpickling the input treedef imports these serialized class modules.
        # Load their ordinary module constants before the strict deserialize
        # guard, as the full path's asset loader already does. No objective.
        importlib.import_module('research.daily_coarse_graining.daily_operator.shared_transformer')
        compile_evidence={'forbidden_compile_calls':0,'model_lowerings':0,'model_backend_attempts':0}
        with no_model_compilation(compiler,compile_evidence):
            with report['budget'].stage(120,'candidate_deserialize_artifact_repair',emit):
                with CANDIDATE.open('rb') as stream:serialized=pickle.load(stream)
                candidate=deserialize_and_load(*serialized)
            with report['budget'].stage(75,'compact_actual_HLO_proof',emit):
                captured=capture_actual_hlo(candidate,args.output_dir,compiled_report['policy_evidence'],defer_audit=True)
                emit('actual_compact_IR_saved',actual_IR=captured)
            proof=resident_CPU_proof(captured,args,compiled_report,report,emit)
        emit('completed',status='artifact_repair_only_no_model_values',optimized_bank_accumulation_evidence=proof,
            compile_evidence=compile_evidence,candidate_calls=0,
            prior_actual_ABI_validation='preserved-attempt-1/model/model.json; both binaries validated')
        return
    source=SimpleNamespace(**{key:files[key] for key in ('source_profile','input_bundle','prepared_context',
        'daily_admission_report','seven_admission_report','thirty_admission_report','support_owner_file')})
    with report['budget'].stage(120,'assets',emit):
        _profile,ref,bundle,context,checkpoint_path,admissions=forward._assets(source)
        checkpoint=load_learning_checkpoint(checkpoint_path,expected_method=ref['method'])
        if checkpoint['completed_updates']!=0:raise ValueError('requires original update0')
        host_parameters=checkpoint['parameters'];leaves=jax.tree_util.tree_leaves(host_parameters)
        if len(leaves)!=300 or sum(original.np.size(value) for value in leaves)!=64884481:raise ValueError('full parameter scope differs')
        case=bundle['payloads'][0]['cases'][7];state=original.np.asarray(case[0])
        if len(case)!=5 or state.shape!=(3854,) or state.dtype!=original.np.float64 or original.np.shape(case.days[0].continuous_state)[0]!=7:
            raise ValueError('T7 initial-state/day ABI differs')
        defined=original.np.isfinite(state)&(original.np.abs(state)<.5e20)
        inventory=original.load_mask_inventory(files['mask_inventory_source'])
        mask_evidence,masks=inventory(dict(case=case,layout=context['head'].layout,normalization=context['head'].input_normalization,
            state_scales=context['state_scales'],template=context['template'],contract=context['head'].contract))
        if len(masks)!=112 or original.np.count_nonzero(~defined)!=21:raise ValueError('source mask scope differs')
        arguments=(jax.device_put(host_parameters),*jax.device_put(case))
        identity=dict(source_head=ref['source_git_head'],source_profile_sha256=hashlib.sha256(files['source_profile'].read_bytes()).hexdigest(),
            method=ref['method'],artifact_id=int(bundle['artifact_ids'][0]),horizon=7,checkpoint=str(checkpoint_path.resolve()),
            jax_version=jax.__version__,admitted_executable=admissions['T7']['executable'],
            candidate_decoder_sha256=hashlib.sha256(files['decoder'].read_bytes()).hexdigest(),candidate_backbone_sha256=hashlib.sha256(files['backbone'].read_bytes()).hexdigest())
        if identity!=metadata['source_identity']:raise ValueError('exact source/method/assets identity differs')
        for relative,digest in ((policy.ST_RELATIVE_PATH,policy.ST_SHA256),(policy.MACRO_RELATIVE_PATH,policy.MACRO_SHA256)):
            policy._pinned_source(SimpleNamespace(__file__=policy.DEPLOYED_SOURCE_PATHS[relative],__name__=str(relative)),relative,digest)
        with files['reference_result'].open('rb') as stream:saved=pickle.load(stream)
        with files['JVP_results'].open('rb') as stream:saved_jvp=pickle.load(stream)
        if saved['identity']!=identity or saved_jvp['identity']!=identity:raise ValueError('original fullAD/JVP identities differ')
    compile_evidence={'model_lowerings':0,'model_backend_attempts':0,'forbidden_compile_calls':0,'graph_constructed':False}
    emit('inputs_ready',identity=identity,compile_evidence=compile_evidence,input_masks=mask_evidence)
    with no_model_compilation(compiler,compile_evidence):
        with report['budget'].stage(120,'original_deserialize',emit):
            with files['reference_executable'].open('rb') as stream:serialized=pickle.load(stream)
            reference=deserialize_and_load(*serialized);validate_inputs(jax,serialized,reference,arguments)
        with report['budget'].stage(120,'candidate_deserialize',emit):
            with CANDIDATE.open('rb') as stream:serialized=pickle.load(stream)
            candidate=deserialize_and_load(*serialized);validate_inputs(jax,serialized,candidate,arguments)
        reference_memory=memory_sizes(reference);candidate_memory=memory_sizes(candidate)
        if max(reference_memory['estimated_live_bytes'],candidate_memory['estimated_live_bytes'])>original.MEMORY_LIMIT:
            raise MemoryError('estimated active memory exceeds24GiB')
        emit('binaries_loaded',candidate_memory=candidate_memory,reference_memory=reference_memory,compile_evidence=compile_evidence,
            candidate_sha256=metadata['candidate_sha256'],binary_input_ABI='actual treedef/shapes/dtypes validated')
        if saved_proof is None:
            with report['budget'].stage(75,'compact_actual_HLO_proof',emit):
                captured=capture_actual_hlo(candidate,args.output_dir,compiled_report['policy_evidence'],defer_audit=True)
                emit('actual_compact_IR_saved',actual_IR=captured)
            proof=resident_CPU_proof(captured,args,compiled_report,report,emit)
        else:proof=saved_proof
        emit('actual_HLO_proof_saved',optimized_bank_accumulation_evidence=proof)
        if not proof['passed']:
            emit('completed',status='bank_accumulation_proof_incomplete',stop_reason='compact actual proof incomplete; no candidate values; preserve actual HLO for offline review');return
        def evaluate(executable,label):
            with report['budget'].stage(30,label,emit):
                began=time.perf_counter();value=jax.device_get(jax.block_until_ready(executable(*arguments)));seconds=time.perf_counter()-began
            hard=original.hard_admission(jax,value,defined,host_parameters);stats=devices[0].memory_stats() or {}
            hard.update(observed_memory_passed='peak_bytes_in_use' in stats and stats['peak_bytes_in_use']<=original.MEMORY_LIMIT,
                observed_whole_process_memory=stats)
            hard['passed']=bool(hard['passed'] and hard['observed_memory_passed'])
            return value,seconds,hard
        baseline,_seconds,admission=evaluate(reference,'original_first_replay')
        original.write_pickle(args.output_dir/'original-replay.pkl',{'identity':identity,'result':baseline,'hard_admission':admission})
        original.write_pickle(args.output_dir/'source-defined-mask.pkl',{'identity':identity,'defined':defined,'input_boolean_masks':masks})
        if not admission['passed']:emit('completed',status='hard_admission_failed',stop_reason='original hard failure saved');return
        bridge=original.compare_full(jax,baseline,saved['result'])
        original.write_pickle(args.output_dir/'original-replay.pkl',{'identity':identity,'result':baseline,'bridge':bridge,'hard_admission':admission})
        if any(not bridge[key]['passed'] for key in ('loss','auxiliary','all_parameters')):raise ValueError('original saved full bridge failed; no state waiver')
        emit('original_replayed',original_saved_bridge=bridge,original_hard_admission=admission)
        first,_seconds,hard=evaluate(candidate,'candidate_first_result')
        accepted=original.save_first_candidate(jax,args.output_dir,identity,first,hard,baseline,saved_jvp['rows'],emit)
        if accepted is None:return
        initial,supplement=accepted
        emit('candidate_first_result_saved',initial_comparisons=initial,initial_hard_admission=hard,JVP_supplement=supplement)
        for label,executable in (('reference_warmup',reference),('candidate_warmup',candidate)):
            value,_seconds,hard=evaluate(executable,label)
            if not hard['passed']:
                original.write_pickle(args.output_dir/(label+'.partial.pkl'),{'identity':identity,'result':value,'hard_admission':hard})
                emit('completed',status='hard_admission_failed',stop_reason=label+' hard failure saved');return
        collected=original.collect_T7_pairs(jax,evaluate,reference,candidate,baseline,first,saved_jvp['rows'],identity,args.output_dir,emit,supplement)
        if collected is None:return
        times,pairs,hard_pairs,self_drift=collected
        strict=all(row['passed'] for row in initial.values()) and all(all(row['passed'] for row in pair.values()) for pair in pairs)
        ratio=float(original.np.median(times['reference'])/original.np.median(times['candidate']))
        emit('completed',status='diagnostic_passed_no_adoption' if strict and ratio>=1.05 else 'screening_failed',
            strict_numerical_passed=strict,speed_gate_passed=ratio>=1.05,median_ratio=ratio,matched_warm_seconds=times,
            pair_comparisons=pairs,paired_hard_admission=hard_pairs,state_self_drift=self_drift,self_drift_waiver=False,
            JVP_supplement=supplement,compile_evidence=compile_evidence,observed_whole_process_memory=devices[0].memory_stats(),
            full_optimizer_updates=0,production_adoption=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--candidate-sha256')
    parser.add_argument('--process-seconds',type=float,default=470)
    parser.add_argument('--inspect-assets-only',action='store_true')
    parser.add_argument('--cli-check-only',action='store_true')
    parser.add_argument('--artifact-only-repair',action='store_true',help='same-deadline ordinary export repair; no model values')
    parser.add_argument('--saved-IR-proof-sha256',help='exact reviewed actual-IR certificate; reuse without printing/review wait')
    args=parser.parse_args()
    if not 0<args.process_seconds<=590:parser.error('process cap must be1..590s')
    if args.cli_check_only:print('saved-binary CLI checked; no GPU/input/compile');return 0
    if args.inspect_assets_only:print(json.dumps(inspect_only(args.root),indent=2));return 0
    if args.output_dir is None or args.candidate_sha256 is None:parser.error('actual replay requires output-dir/exact candidate digest')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={'status':'running','stage_events':[],'candidate_backend_attempts':0,'training_updates':0,
        'production_adoption':False,'mode':'same_saved_candidate_binary_T7_no_compile','rtol':1e-5,'atol':1e-6,
        'budget':ReplayBudget(args.process_seconds,args.output_dir)}
    def emit(stage,**fields):
        event={'event':stage,'elapsed_seconds':time.perf_counter()-report['budget'].started,'UTC_seconds':time.time(),**fields}
        report['stage_events'].append(event);report.update(stage=stage,**fields)
        original.write_json(args.output_dir/'model.json',{key:value for key,value in report.items() if key!='budget'})
        print('saved_bank stage='+stage+' elapsed='+str(round(event['elapsed_seconds'],1)),flush=True)
    try:emit('starting');run(args,emit,report)
    except BaseException as error:
        emit('exception',status='resource_failed' if isinstance(error,MemoryError) else 'runtime_or_admission_failed',
            failure_kind='GPU_24GiB_gate' if isinstance(error,MemoryError) and '24GiB' in str(error) else 'host_or_runtime_failure',error=repr(error))
        raise
    return 0 if report['status']=='diagnostic_passed_no_adoption' else 1

if __name__=='__main__':raise SystemExit(main())
