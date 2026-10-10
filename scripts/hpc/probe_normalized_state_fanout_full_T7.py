"""Preparation-first full public T7 native normalized-state fan-in candidate.

Actual GPU execution requires separate user authorization. No reference graph
compilation, delta instrumentation, custom VJP, training or adoption is present.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from types import SimpleNamespace
import normalized_state_fanout_full_T7 as policy

ROOT=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
CHECKOUT=ROOT/'runtime/checkouts/shared-split-width-v1'
MEMORY_LIMIT=24*2**30


class Budget:
    def __init__(self,seconds):
        self.started=time.perf_counter(); self.limit=seconds
    @contextmanager
    def stage(self,cap,label,emit):
        allowed=min(cap,self.limit-(time.perf_counter()-self.started))
        if allowed<=0:
            raise TimeoutError('original dispatch allowance exhausted')
        emit('stage_start',stage_label=label,effective_timeout_seconds=allowed)
        def expire():
            try: emit('timeout',status='timeout',stage_label=label)
            finally: os._exit(124)
        timer=threading.Timer(allowed,expire); timer.daemon=True; timer.start()
        try: yield
        finally: timer.cancel()


def synthetic_interface_check():
    """CPU-only native cast/reverse interface; no real model or Teacher reads."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update('jax_enable_x64',True)
    weight_host=np.zeros((8,3854),np.float32)
    weight_host[:,:2]=[[1e7,2.],[1.,3.],[-1e7,4.],[2.,5.],[3.,6.],[4.,7.],[5.,8.],[6.,9.]]
    weights=jnp.asarray(weight_host)
    def normalization(raw):
        with jax.named_scope('original_normalization'):
            return raw.astype(jnp.float32)/2.
    def original(raw,weight_arg):
        state=normalization(raw)
        return sum(jnp.sum(state*weight_arg[i],dtype=jnp.float32) for i in range(8))
    def candidate(raw,weight_arg):
        state=normalization(raw)
        with jax.named_scope(policy.MARKER):
            state=jnp.asarray(state,dtype=jnp.float64)
        contributions=[]
        for i in range(8):
            scope=policy.CONSUMER_MARKER+('field' if i==0 else f'axis_{i-1}')
            with jax.named_scope(scope):
                contributions.append(jnp.sum(state.astype(jnp.float32)*weight_arg[i],dtype=jnp.float32))
        return sum(contributions)
    raw_host=np.zeros(3854,np.float64); raw_host[:2]=[1.,2.]
    raw=jnp.asarray(raw_host)
    a=jax.jit(jax.value_and_grad(original)).lower(raw,weights).compile()
    b=jax.jit(jax.value_and_grad(candidate)).lower(raw,weights).compile()
    value_a,gradient_a=a(raw,weights); value_b,gradient_b=b(raw,weights)
    expected=(np.sum(np.asarray(weights).astype(np.float64),axis=0).astype(np.float32)/np.float32(2)).astype(np.float64)
    if not np.array_equal(np.asarray(value_a),np.asarray(value_b)) or not np.array_equal(np.asarray(gradient_b),expected):
        raise ValueError('native F32 forward/F64 fan-in interface failed')
    module=b.runtime_executable().hlo_modules()[0]
    proto=module.as_serialized_hlo_module_proto()
    import normalized_state_fanout_proto as proto_reader
    _entry,computations,instructions=proto_reader.parse_module(proto)
    import inspect
    candidate_lines,candidate_start=inspect.getsourcelines(candidate)
    normalization_lines,normalization_start=inspect.getsourcelines(normalization)
    profile=dict(source_file=Path(__file__).name,
        source_span=(candidate_start,candidate_start+len(candidate_lines)-1),
        consumer_span=(candidate_start,candidate_start+len(candidate_lines)-1),
        normalization_file=Path(__file__).name,
        normalization_span=(normalization_start,normalization_start+len(normalization_lines)-1))
    proof=proto_reader.extract(proto,**profile)
    if not proof['passed']:
        raise ValueError('tiny native proto F64 fan-in proof did not resolve: '+json.dumps(dict(
            certificates=proof['certificates'],unsupported=proof['unsupported_root_boundaries'],
            profile=profile)))
    # Mutate the actual CPU module's parsed SSA to challenge only this scoped
    # certificate. These controls add no compilation and do not touch real data.
    import copy
    parsed=(_entry,computations,instructions); original_parser=proto_reader.parse_module
    controls={}
    def reject(name,mutate):
        changed=copy.deepcopy(parsed); mutate(*changed)
        proto_reader.parse_module=lambda _data:changed
        try:
            try: admitted=proto_reader.extract(proto,**profile)['passed']
            except ValueError: admitted=False
            if admitted: raise ValueError('negative native proto control accepted: '+name)
            controls[name]='rejected'
        finally: proto_reader.parse_module=original_parser
    certificate=proof['certificates'][0]
    def duplicate(e,c,rows):
        add=rows[certificate['f64_adds'][0][0]]; add['operands'][1]=add['operands'][0]
    reject('duplicate_partial_identity',duplicate)
    def dynamic(e,c,rows): rows[c[e]['root']]['shape']['tuples'][-1]['dynamic_dimensions']=[1]
    reject('dynamic_state_output',dynamic)
    def missing_consumer(e,c,rows): rows[certificate['incoming_f32_partial_converts'][0][0]]['metadata']['op_name']='missing_consumer'
    reject('missing_consumer_metadata',missing_consumer)
    def wrong_source(e,c,rows): rows[certificate['narrow']]['metadata']['source_file']='wrong_source.py'
    reject('wrong_target_source',wrong_source)
    def wrong_normalization(e,c,rows):
        for use in certificate['original_normalization_transpose_immediate_uses']:
            rows[use['id']]['metadata']['source_file']='wrong_normalization.py'
    reject('wrong_normalization_transpose_source',wrong_normalization)
    def boundary(e,c,rows,unused=False):
        root=rows[c[e]['root']]; previous=root['operands'][-1]; number=max(rows)+1
        leaf=copy.deepcopy(rows[previous]); leaf.update(id=number,opcode='constant',operands=[],calls=[],metadata={})
        rows[number]=leaf
        if unused:
            tuple_row=copy.deepcopy(root); tuple_row.update(id=number+1,operands=[number,previous],shape=dict(dtype=13,dimensions=[],dynamic_dimensions=[],tuples=[leaf['shape'],leaf['shape']]))
            rows[number+1]=tuple_row
            gte=copy.deepcopy(leaf); gte.update(id=number+2,opcode='get-tuple-element',operands=[number+1],tuple_index=0)
            rows[number+2]=gte; root['operands'][-1]=number+2
        else:
            leaf.update(opcode='custom-call',operands=[previous]); root['operands'][-1]=number
    reject('unreturned_target_in_unused_GTE',lambda e,c,r:boundary(e,c,r,True))
    reject('unknown_custom_call_boundary',boundary)
    def unknown(e,c,rows):
        boundary(e,c,rows); rows[rows[c[e]['root']]['operands'][-1]]['opcode']='unknown-dataflow-operation'
    reject('unknown_opcode_boundary',unknown)
    def cycle(e,c,rows):
        boundary(e,c,rows); number=rows[c[e]['root']]['operands'][-1]
        rows[number].update(opcode='copy',operands=[number])
    reject('active_SSA_cycle',cycle)
    return dict(status='passed',backend=jax.default_backend(),jax_version=jax.__version__,
        raw_shape=[3854],original_gradient_first_two=np.asarray(gradient_a)[:2].tolist(),
        candidate_gradient_first_two=np.asarray(gradient_b)[:2].tolist(),
        native_F64_sum_then_F32_expected_first_two=expected[:2].tolist(),forward_exact=True,
        native_proto_interface=True,proto_bytes=len(proto),proto_computations=len(computations),
        proto_instructions=len(instructions),scoped_proto_certificate_passed=proof['passed'],
        incoming_partials=[len(row['incoming_f32_partial_converts']) for row in proof['certificates']],
        original_normalization_transpose_source_bound=all(bool(row['original_normalization_transpose_immediate_uses']) for row in proof['certificates']),
        negative_scoped_proto_controls=controls,
        full_HLO_text_rendered=False,
        real_model_evaluations=0,Teacher_reads=0)


def run(args,emit,report):
    with report['budget'].stage(300,'setup_assets_original_replay',emit):
        import hashlib
        import numpy as np
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
        jax,devices=initialize_gpu_backend()
        from jax._src import compiler
        from jax.experimental.serialize_executable import deserialize_and_load,serialize
        from scripts.hpc.run_shared_learning import load_learning_checkpoint
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
        from research.daily_coarse_graining.daily_operator import shared_process_head
        import probe_shared_forward_reverse as forward
        import probe_shared_bank_adjoint_T7 as assets
        from probe_shared_normalized_keys_recurrent import validate_inputs
        from probe_shared_command_buffer_loops import forbid_component_compile
        from probe_normalized_state_fanout_component import backend_guard
        if jax.__version__!='0.4.38' or len(devices)!=1:
            raise ValueError('requires canonical one-GPU JAX0.4.38')
        preflight=json.loads(args.runtime_preflight.read_text())
        if preflight.get('status')!='passed' or not preflight.get('passed') or preflight.get('actual_backend')!='gpu' or preflight.get('jax_version')!=jax.__version__:
            raise ValueError('deployed runtime preflight differs or failed')
        metadata=assets.inspect_assets(args.root); files=assets.paths(args.root)
        source=SimpleNamespace(**{key:files[key] for key in ('source_profile','input_bundle','prepared_context',
            'daily_admission_report','seven_admission_report','thirty_admission_report','support_owner_file')})
        _profile,ref,bundle,context,checkpoint_path,admissions=forward._assets(source)
        checkpoint=load_learning_checkpoint(checkpoint_path,expected_method=ref['method'])
        host_parameters=checkpoint['parameters']
        if checkpoint['completed_updates']!=0 or len(jax.tree_util.tree_leaves(host_parameters))!=300 or sum(np.size(x) for x in jax.tree_util.tree_leaves(host_parameters))!=64884481:
            raise ValueError('original update0/full-network scope differs')
        case=bundle['payloads'][0]['cases'][7]; state=np.asarray(case[0])
        if len(case)!=5 or state.shape!=(3854,) or state.dtype!=np.float64 or np.shape(case.days[0].continuous_state)[0]!=7:
            raise ValueError('original T7 ABI differs')
        if len(context['head'].layout.expert_names)!=7:
            raise ValueError('requires seven original expert state consumers')
        defined=np.isfinite(state)&(np.abs(state)<.5e20)
        mask_inventory=assets.load_mask_inventory(files['mask_inventory_source'])
        mask_evidence,input_masks=mask_inventory(dict(case=case,layout=context['head'].layout,
            normalization=context['head'].input_normalization,state_scales=context['state_scales'],
            template=context['template'],contract=context['head'].contract))
        if len(input_masks)!=112 or np.count_nonzero(~defined)!=21:
            raise ValueError('original source mask scope differs')
        assets.write_pickle(args.output_dir/'candidate-input-snapshot.pkl',dict(case=case,
            normalization=context['head'].input_normalization,layout=context['head'].layout,
            checkpoint=str(checkpoint_path),completed_updates=0,prepared_inputs_reused=True))
        assets.write_json(args.output_dir/'asset-metadata.json',metadata)
        parameters=jax.device_put(host_parameters); arguments=(parameters,*jax.device_put(case))
        identity=dict(source_head=ref['source_git_head'],source_profile_sha256=hashlib.sha256(files['source_profile'].read_bytes()).hexdigest(),
            method=ref['method'],artifact_id=int(bundle['artifact_ids'][0]),horizon=7,checkpoint=str(checkpoint_path.resolve()),
            jax_version=jax.__version__,admitted_executable=admissions['T7']['executable'],
            candidate_decoder_sha256=hashlib.sha256(files['decoder'].read_bytes()).hexdigest(),
            candidate_backbone_sha256=hashlib.sha256(files['backbone'].read_bytes()).hexdigest())
        if identity!=metadata['reference_identity']:
            raise ValueError('original full-AD identity differs')
        with files['reference_result'].open('rb') as stream: saved=pickle.load(stream)
        if saved['identity']!=identity: raise ValueError('historical original result identity differs')
        with forbid_component_compile(True):
            with files['reference_executable'].open('rb') as stream: serialized=pickle.load(stream)
            original=deserialize_and_load(*serialized)
            validate_inputs(jax,serialized,original,arguments)
            original_memory=memory_sizes(original)
            if original_memory['estimated_live_bytes']>MEMORY_LIMIT: raise MemoryError('original memory exceeds24GiB')
            baseline=jax.device_get(jax.block_until_ready(original(*arguments)))
        assets.write_pickle(args.output_dir/'original-first.pkl',dict(identity=identity,result=baseline))
        baseline_hard=assets.hard_admission(jax,baseline,defined,host_parameters)
        baseline_stats=devices[0].memory_stats() or {}
        baseline_hard['observed_memory_passed']='peak_bytes_in_use' in baseline_stats and baseline_stats['peak_bytes_in_use']<=MEMORY_LIMIT
        baseline_hard['passed']=baseline_hard['passed'] and baseline_hard['observed_memory_passed']
        assets.write_pickle(args.output_dir/'source-masks.pkl',dict(identity=identity,defined=defined,input_boolean_masks=input_masks,mask_evidence=mask_evidence))
        saved_comparison=assets.compare_full(jax,baseline,saved['result'])
        emit('original_first_saved',identity=identity,original_hard=baseline_hard,
            historical_saved_result_comparison=saved_comparison,historical_comparison_is_gate=False,
            original_memory=original_memory,mask_evidence=mask_evidence)
        if not baseline_hard['passed']:
            emit('completed',status='hard_admission_failed',stop_reason='original ABI/finite/physical/memory failed'); return
        owner=forward._load_module(args.root/'runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py',
                                  'scripts.hpc.probe_shared_expert_real_daily')
        adapter=owner.load_staged_adapter(files['decoder']); backbone=owner.load_backbone_assembler(files['backbone'])
        _rollout,objective=make_shared_recurrent_objective(context['head'],context['template'],context['state_scales'],context['science_scales'])
        backward=jax.jit(jax.value_and_grad(objective,argnums=(0,1),has_aux=True))
    with report['budget'].stage(3100,'sole_candidate_lower_backend',emit):
        emit('candidate_graph_start',candidate_compile_attempts=1)
        with owner.candidate_process_head_patches(shared_process_head,adapter,backbone),policy.install(backbone,CHECKOUT) as policy_evidence:
            began=time.perf_counter(); lowered=backward.lower(*arguments)
            emit('candidate_lowered',lower_seconds=time.perf_counter()-began,policy_evidence=policy_evidence)
            with backend_guard(compiler,report,emit,'candidate'):
                began=time.perf_counter(); candidate=lowered.compile()
                emit('candidate_compiled',backend_seconds=time.perf_counter()-began)
    candidate_memory=memory_sizes(candidate)
    if candidate_memory['estimated_live_bytes']>MEMORY_LIMIT: raise MemoryError('candidate memory exceeds24GiB')
    def evaluate(executable,label):
        with forbid_component_compile(True),backend_guard(compiler,report,emit):
            began=time.perf_counter(); device_value=executable(*arguments); jax.block_until_ready(device_value)
            seconds=time.perf_counter()-began
            began=time.perf_counter(); value=jax.device_get(device_value); host_seconds=time.perf_counter()-began
        hard=assets.hard_admission(jax,value,defined,host_parameters)
        stats=devices[0].memory_stats() or {}
        hard['observed_memory_passed']='peak_bytes_in_use' in stats and stats['peak_bytes_in_use']<=MEMORY_LIMIT
        hard['observed_memory']=stats; hard['passed']=hard['passed'] and hard['observed_memory_passed']
        return value,seconds,host_seconds,hard
    with report['budget'].stage(90,'candidate_first_raw_and_binary',emit):
        # Full raw candidate result precedes every hard/strict gate.
        with forbid_component_compile(True),backend_guard(compiler,report,emit):
            began=time.perf_counter(); value=candidate(*arguments); jax.block_until_ready(value)
            first_seconds=time.perf_counter()-began; first=jax.device_get(value)
        assets.write_pickle(args.output_dir/'candidate-first.pkl',dict(identity=identity,result=first))
        first_hard=assets.hard_admission(jax,first,defined,host_parameters)
        first_stats=devices[0].memory_stats() or {}
        first_hard['observed_memory_passed']='peak_bytes_in_use' in first_stats and first_stats['peak_bytes_in_use']<=MEMORY_LIMIT
        first_hard['passed']=first_hard['passed'] and first_hard['observed_memory_passed']
        initial=assets.compare_full(jax,first,baseline)
        emit('candidate_first_saved',initial_comparisons=initial,initial_hard_admission=first_hard,
            first_call_seconds=first_seconds,candidate_memory=candidate_memory,observed_memory=first_stats,
            strict_failure_waiver=False,strict_failure_does_not_suppress_pairs=True)
        if not first_hard['passed']:
            emit('completed',status='hard_admission_failed',stop_reason='candidate ABI/finite/physical/memory failed'); return
        # Binary/proto retained before extra timings. Native proto avoids full
        # HLO text rendering; bounded source/marker closure is reviewed offline.
        with forbid_component_compile(True),backend_guard(compiler,report,emit):
            assets.write_pickle(args.output_dir/'candidate.executable.pkl',serialize(candidate))
            modules=[]
            for index,module in enumerate(candidate.runtime_executable().hlo_modules()):
                path=args.output_dir/f'candidate.optimized-{index}.hlo.pb'
                raw=module.as_serialized_hlo_module_proto(); path.write_bytes(raw)
                modules.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        assets.write_json(args.output_dir/'candidate.native-proto.json',dict(modules=modules,
            public_state_output_index=len(jax.tree_util.tree_leaves(first))-1,
            public_state_output_shape=[3854],public_state_output_dtype='float64',
            full_HLO_text_rendered=False,mechanism_status='pending_scoped_CPU_proto_review',new_compilations=0))
        emit('candidate_binary_proto_saved',native_proto=modules,full_HLO_text_rendered=False)
    pairs=[]; timings=dict(reference=[],candidate=[]); completed=False
    with report['budget'].stage(180,'three_interleaved_complete_T7_pairs',emit):
        pair_started=time.perf_counter()
        for repeat in range(3):
            remaining=min(180-(time.perf_counter()-pair_started),report['budget'].limit-(time.perf_counter()-report['budget'].started))
            if remaining<max(15,4*first_seconds+8):
                emit('pair_budget_insufficient',completed_pairs=len(pairs),speed_stability_verified=False); break
            order=('reference','candidate') if repeat%2==0 else ('candidate','reference')
            values={}; self_checks={}; hard_checks={}
            for label in order:
                value,seconds,host_seconds,hard=evaluate(original if label=='reference' else candidate,label)
                path=args.output_dir/f'pair-{repeat}-{label}.pkl'
                compact=dict(identity=identity,arm=label,loss_aux=value[0],state_gradient=value[1][1],
                    synchronous_complete_T7_AD_seconds=seconds,host_transfer_seconds=host_seconds,
                    parameter_self_comparison=dict(passed=None,status='raw_saved_before_comparison'))
                assets.write_pickle(path,compact)
                own_first=baseline if label=='reference' else first
                check=assets.compare_full(jax,value,own_first)
                compact['parameter_self_comparison']=check['all_parameters']
                compact['hard_admission']=hard; assets.write_pickle(path,compact)
                self_checks[label]=check; hard_checks[label]=hard; values[label]=value
                timings[label].append(seconds)
                if not hard['passed']:
                    emit('completed',status='hard_admission_failed',failed_arm=label,failed_pair=repeat,
                        matched_seconds=timings,completed_pairs=len(pairs),stop_reason='hard/memory failure; no following arm'); return
            comparison=assets.compare_full(jax,values['candidate'],values['reference'])
            pair=dict(repeat=repeat,order=order,comparison=comparison,self_comparisons=self_checks,
                      hard_admission=hard_checks)
            pairs.append(pair); assets.write_json(args.output_dir/f'pair-{repeat}.json',pair)
            emit('pair_saved',matched_pairs=pairs,matched_seconds=timings)
            del values
        completed=len(pairs)==3
    strict=all(v['passed'] for v in initial.values()) and completed and all(
        all(v['passed'] for v in row['comparison'].values()) and all(
            all(v['passed'] for v in check.values()) for check in row['self_comparisons'].values()) for row in pairs)
    ratio=float(np.median(timings['reference'])/np.median(timings['candidate'])) if completed else None
    emit('completed',status='strict_numerical_passed_mechanism_review_pending' if strict else
        'strict_numerical_failed' if completed else 'paired_measurement_incomplete',
        strict_numerical_passed=strict,completed_pairs=len(pairs),matched_timing_complete=completed,
        within_arm_strict_stability=completed and all(all(all(v['passed'] for v in check.values())
            for check in row['self_comparisons'].values()) for row in pairs),speed_stability_acceptance=False,
        median_T7_AD_ratio=ratio,paired_T7_AD_ratios=[a/b for a,b in zip(timings['reference'],timings['candidate'])],
        speed_pass_threshold=None,reference_waiver=False,
        full_optimizer_update_speed_claim=False,training_updates=0,reference_adoption=False,production_adoption=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--runtime-preflight',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--process-seconds',type=float,default=3590)
    parser.add_argument('--cli-check-only',action='store_true')
    parser.add_argument('--source-check',type=Path)
    parser.add_argument('--synthetic-interface-check',action='store_true')
    args=parser.parse_args(argv)
    if not 0<args.process_seconds<=3590: parser.error('process cap must be within3590s')
    if args.cli_check_only:
        print(json.dumps(dict(status='CLI_passed',JAX_imports=int('jax' in sys.modules),GPU_calls=0,Teacher_reads=0))); return 0
    if args.source_check:
        print(json.dumps(policy.source_check(args.source_check),indent=2)); return 0
    if args.synthetic_interface_check:
        print(json.dumps(synthetic_interface_check(),indent=2)); return 0
    if args.output_dir is None or args.runtime_preflight is None: parser.error('actual mode requires output-dir/preflight')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',candidate_compile_attempts=0,stage_events=[],budget=Budget(args.process_seconds),
        backend_attempts=dict(candidate=0),setup_backend_attempts=0,forbidden_backend_attempts=0,
        precision='original_FP32_forward_native_F64_normalized_state_fanin_FP64_physical',
        full_parameter_coordinates=64884481,full_state_coordinates=3854,delta_inputs=False,custom_VJP=False)
    lock=threading.RLock()
    def emit(event,**fields):
        with lock:
            report['stage_events'].append(dict(event=event,elapsed_seconds=time.perf_counter()-report['budget'].started,**fields))
            report.update(stage=event,**fields)
            temporary=args.output_dir/'model.json.tmp'
            temporary.write_text(json.dumps({k:v for k,v in report.items() if k!='budget'},indent=2,default=str,allow_nan=False)+'\n')
            os.replace(temporary,args.output_dir/'model.json'); print(event,flush=True)
    def expire():
        if lock.acquire(blocking=False):
            try: emit('process_timeout',status='timeout')
            finally: lock.release()
        os._exit(124)
    timer=threading.Timer(args.process_seconds,expire); timer.daemon=True; timer.start()
    try: emit('starting'); run(args,emit,report)
    except BaseException as error:
        emit('exception',status='resource_failed' if isinstance(error,MemoryError) else 'runtime_or_admission_failed',error=repr(error)); raise
    finally: timer.cancel()
    return 0 if report['status']=='strict_numerical_passed_mechanism_review_pending' else 1


if __name__=='__main__':
    raise SystemExit(main())
