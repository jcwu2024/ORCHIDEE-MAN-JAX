"""Frozen small held-out development panel, original canonical forward only.

CPU mode reads saved parameters/prepared metadata only. GPU mode reuses public
shared_learning_monitor and limits canonical rollout compilation to T7/T30.
No training, optimizer, model derivative graph or new scientific threshold.
"""
from __future__ import annotations
import argparse
from collections import namedtuple
from contextlib import contextmanager
import gc
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import threading
import time
import numpy as np

ROOT=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
DIAG=ROOT/'runtime/outputs/diagnostics'
FAMILY=DIAG/'shared-v4-heldout-science-20261004'
CHECKOUT=ROOT/'runtime/checkouts/shared-split-width-v1'
OVERLAY=ROOT/'runtime/transfer/shared-v4-heldout-science-20261004'
VALIDATION=ROOT/'runtime/outputs/training/fresh-method-validation-20260921/prepared_manifest.json'
CONTEXT=DIAG/'shared-expert-real-daily-context-c599236.pkl'
BUNDLE=DIAG/'shared-split-dual-inputs-c599236.pkl'
INITIAL=DIAG/'shared-v4-same-input-update-20261003/model/pair-1-reference.full-proposal.pkl'
ARMS=('reference_a','candidate','reference_b')
STEPS=(8,16)
ORDINALS=(1,9,33,41)
HEAD='c599236bcf57b745e9bafed678298010f4c3b58f'
ADAM=namedtuple('InertAdam','step first second')
INERT={
 'SharedTrajectoryLossResult':namedtuple('SharedTrajectoryLossResult','total flux stock living_stock nonliving_stock flux_daily flux_prefix stock_change stock_trajectory hard_counts'),
 'TermDiagnostics':namedtuple('TermDiagnostics','loss count expected_count invalid_prediction_count invalid_target_count normalized_residual'),
 'SharedTrajectoryHardCounts':namedtuple('SharedTrajectoryHardCounts','nonfinite_defined_prediction nonfinite_defined_target defined_mask_mismatch missing_valid_term total'),
}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)

def load(path):
    with path.open('rb') as stream:return pickle.load(stream)

class ArraysOnly(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='research.daily_coarse_graining.daily_operator.target_oriented_training' and name=='TargetOrientedAdamWState':return ADAM
        if module=='research.daily_coarse_graining.daily_operator.shared_trajectory_loss' and name in INERT:return INERT[name]
        if module.startswith(('research.','jax','jaxlib','scripts.')):raise ValueError('Non-array weight payload class '+module+'.'+name)
        return super().find_class(module,name)

def arrays_load(path):
    with path.open('rb') as stream:return ArraysOnly(stream).load()

def numerical_leaves(value,path=()):
    if isinstance(value,dict):
        for key in sorted(value):yield from numerical_leaves(value[key],path+(str(key),))
    elif isinstance(value,(tuple,list)):
        for key,item in enumerate(value):yield from numerical_leaves(item,path+(str(key),))
    elif value is None:yield '/'.join(path),None
    else:yield '/'.join(path),np.asarray(value)

def parameter_abi(parameters):
    rows=[];coordinates=none=0
    for path,array in numerical_leaves(parameters):
        if array is None:none+=1;rows.append(dict(path=path,static_None=True));continue
        if array.dtype.kind not in 'f' or not np.isfinite(array).all():raise ValueError('Invalid parameter '+path)
        coordinates+=array.size;rows.append(dict(path=path,shape=list(array.shape),dtype=str(array.dtype)))
    if (len(rows)-none,coordinates,none)!=(300,64884481,2):raise ValueError('Parameter ABI differs')
    return dict(numeric_leaves=300,coordinates=coordinates,static_None=none,finite=True,
        descriptor_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())

def scope():
    return dict(status='authorized_preparation',steps=list(STEPS),arms=list(ARMS),selected_ordinals=list(ORDINALS),
        horizons=[7,30],windows_per_horizon=2,role='held_out_development',sealed_test_used=False,
        selection_rule='fixed manifest ordinals, before scientific predictions; one temporal and one joint per horizon',
        historical_method_development_visible=True,scientific_thresholds=None,production_adoption=False,
        CPU_seconds=300,GPU_seconds=1500,GPU_count=1,GPU_memory_limit_GiB=24,
        max_model_forward_compiles=2,model_gradient_compiles=0,training_updates=0,optimizer_executions=0,
        OMP_threads=4,OpenBLAS_threads=1,max_card_hours=1500/3600,paid_jobs=0,CNY=0,
        original_forward_path='admitted original field expert128/backbone source reuse',
        strict_failures_preserved=dict(candidate_gradient=978,candidate_state=287,reference_b_state=7))

def prepare_cpu():
    began=time.perf_counter();FAMILY.mkdir(parents=True,exist_ok=True)
    write(FAMILY/'scope.json',scope())
    manifest=json.loads(VALIDATION.read_text());panel=json.loads((VALIDATION.parent/manifest['validation_panel']).read_text())
    if manifest.get('status')!='passed' or manifest.get('sealed_test_used') is not False or manifest.get('teacher_data_regenerated') is not False or manifest.get('historical_v5_used') is not False:raise ValueError('Prepared validation provenance differs')
    if len(manifest['entries'])!=48 or manifest['validation_panel_sha256']!=panel['panel_sha256']:raise ValueError('Validation panel differs')
    records={int(r['ordinal']):r for r in manifest['entries']};selected=[]
    splits={'temporal':('train','validation'),'joint':('validation','validation')}
    for ordinal in ORDINALS:
        record=records[ordinal];entry=record['window_entry'];h=int(record['horizon_days']);start=int(entry['start_rows'][0]);name=record['slice_id']
        path=VALIDATION.parent/record['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Selected prepared asset missing or changed')
        selected.append(dict(artifact_id=ordinal,slice_id=name,length=h,component={7:'seven_day',30:'thirty_day'}[h],
            landpoint_id=entry['landpoint_id'],year=int(entry['year']),start_row=start,
            teacher_day_keys=[[entry['landpoint_id'],int(entry['year']),day] for day in range(start,start+h)],
            spatial_split=splits[name][0],temporal_split=splits[name][1],prepared_path=str(path),prepared_sha256=record['sha256']))
    write(FAMILY/'selection.json',dict(status='frozen_before_model_predictions',manifest=str(VALIDATION),
        panel_sha256=panel['panel_sha256'],rows=selected,rule=scope()['selection_rule'],sealed_test_used=False,
        historical_method_development_visible=True,spatial_only_slice_not_sampled=True))
    # Passive prepared records only; no runtime/model construction or evaluation.
    bundle=load(BUNDLE);payload=bundle['payloads'][0];daily=payload['daily'];cases=payload['cases']
    training_manifest=json.loads((ROOT/'runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json').read_text())
    record=next(r for r in training_manifest['entries'] if int(r['update'])==742)
    if bundle['artifact_ids'][0]!=742 or bundle['source_git_head']!=HEAD:raise ValueError('Training input identity differs')
    metadata=dict(record);days=dict(daily=np.asarray(daily.inputs.day_index).reshape(-1).astype(int).tolist(),
        T7=np.asarray(cases[7][2][0].day_index).reshape(-1).astype(int).tolist(),
        T30=np.asarray(cases[30][2][0].day_index).reshape(-1).astype(int).tolist())
    write(FAMILY/'training-overlap-inputs.json',dict(artifact_id=742,manifest_record=metadata,actual_training_day_indices=days,
        update_count=16,same_fixed_inputs_all_updates=True,selected_rows=selected))
    training_path=ROOT/'runtime/outputs/training/target-oriented-prepared-1000-a243029'/record['path']
    with gzip.open(training_path,'rb') as stream:artifact=pickle.load(stream)
    schedule=artifact.schedule_entry
    if int(schedule['update'])!=742 or int(schedule['runtime_year'])!=int(record['runtime_year']):raise ValueError('Training schedule identity differs')
    def flattened(component):
        return [(r['landpoint_id'],int(r['year']),int(start)) for r in schedule['components'][component] for start in r['start_rows']]
    training_keys=[];training_windows=[]
    daily_keys=flattened('one_day')
    if len(daily_keys)!=len(days['daily']):raise ValueError('Training daily row count differs')
    training_keys.extend(daily_keys)
    window_index=int(bundle['method']['configuration']['recurrent_window_index'])
    for horizon,component_name in ((7,'seven_day'),(30,'thirty_day')):
        point,year,start=flattened(component_name)[window_index]
        actual=days[f'T{horizon}']
        if len(actual)!=horizon or not np.all(np.diff(actual)==1):raise ValueError('Training recurrent ordering differs')
        keys=[(point,year,row) for row in range(start,start+horizon)]
        training_keys.extend(keys);training_windows.append(dict(component=component_name,landpoint_id=point,year=year,start_row=start,length=horizon,actual_day_indices=actual,teacher_row_keys=keys))
    training_keys=set(training_keys);training_points={k[0] for k in training_keys};training_years={k[1] for k in training_keys}
    audits=[]
    for row in selected:
        keys={tuple(k) for k in row['teacher_day_keys']};common=sorted(keys & training_keys)
        audits.append(dict(artifact_id=row['artifact_id'],landpoint_id=row['landpoint_id'],year=row['year'],
            same_training_location=row['landpoint_id'] in training_points,same_training_year=row['year'] in training_years,
            exact_teacher_day_overlap_count=len(common),exact_teacher_day_overlap=common,
            exact_training_window_match=any((w['landpoint_id'],w['year'],w['start_row'],w['length'])==(row['landpoint_id'],row['year'],row['start_row'],row['length']) for w in training_windows)))
    if any(a['exact_teacher_day_overlap_count'] or a['exact_training_window_match'] for a in audits):raise ValueError('Fixed held-out panel overlaps actual repeated training inputs')
    write(FAMILY/'training-overlap.json',dict(status='passed_no_exact_day_or_window_overlap',artifact_id=742,
        audited_updates=list(range(1,17)),training_manifest_record=metadata,training_daily_row_keys=daily_keys,
        training_daily_actual_day_indices=days['daily'],training_windows=training_windows,
        unique_training_day_count=len(training_keys),selected_window_audits=audits,
        historical_method_development_visible=True,sealed_independent_test=False,
        scope='Actual fixed artifact742 inputs repeated in this 16-update diagnostic; earlier exploratory checkpoint/data exposure is not certified absent'))
    del artifact,bundle,daily,cases,payload;gc.collect()
    parameters=[];expected=None
    for step,arm in [(0,'shared_initial')]+[(s,a) for s in STEPS for a in ARMS]:
        destination=FAMILY/'parameters'/f'step-{step:02d}-{arm}.pkl'
        source=INITIAL if step==0 else DIAG/f'shared-v4-learning-sensitivity-20261004/model/update-{step:02d}-{arm}.pkl'
        if destination.exists():
            previous=arrays_load(destination);p=previous['parameters'];abi=parameter_abi(p)
        else:
            saved=arrays_load(source)
            if saved.get('source_head')!=HEAD or saved.get('artifact_id')!=742:raise ValueError('Weight provenance differs')
            p=saved['result']['parameters'];abi=parameter_abi(p)
            optimizer_step=int(saved['result']['optimizer'].step)
            if optimizer_step!=step+2:raise ValueError('Weight optimizer step differs')
            save(destination,dict(parameters=p,source=str(source),source_head=HEAD,artifact_id=742,
                diagnostic_step=step,optimizer_step=optimizer_step,ABI=abi))
            del saved
        if expected is None:expected=abi['descriptor_sha256']
        if abi['descriptor_sha256']!=expected:raise ValueError('Weights tree shape/dtype differs')
        parameters.append(dict(step=step,arm=arm,path=str(destination),bytes=destination.stat().st_size,ABI=abi))
        del p;gc.collect()
    write(FAMILY/'CPU-preparation.json',dict(status='passed',CPU_node=socket.gethostname(),elapsed_seconds=time.perf_counter()-began,
        parameters=parameters,selected_windows=len(selected),model_calls=0,GPU_calls=0,optimizer_executions=0,
        Teacher_regeneration=0,raw_Teacher_reads=0,prepared_assets_read=True))
    print(json.dumps(dict(status='CPU_preparation_passed',parameters=len(parameters),selected_windows=len(selected),elapsed_seconds=time.perf_counter()-began)),flush=True)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result

class TrackedForward:
    def __init__(self,jax,function,report,emit):
        self.jax=jax;self.original=jax.jit(function);self.report=report;self.emit=emit;self.cache={};self.signatures={}
    def lower(self,*arguments):
        horizon=int(np.shape(arguments[3][0].day_index)[0]);leaves,tree=self.jax.tree_util.tree_flatten(arguments)
        signature=hashlib.sha256(repr((str(tree),[(np.shape(x),str(x.dtype) if hasattr(x,'dtype') else str(np.asarray(x).dtype),bool(getattr(x,'weak_type',False))) for x in leaves])).encode()).hexdigest()
        if horizon in self.signatures and self.signatures[horizon]!=signature:raise ValueError('More than one forward ABI per horizon')
        self.signatures[horizon]=signature
        owner=self
        class Pending:
            def compile(self):
                if horizon in owner.cache:return owner.cache[horizon]
                binary_path=FAMILY/'model'/f'T{horizon}.forward.executable.pkl'
                metadata_path=FAMILY/'model'/f'T{horizon}.forward.json'
                if binary_path.exists():
                    from jax.experimental.serialize_executable import deserialize_and_load
                    metadata=json.loads(metadata_path.read_text())
                    if metadata['signature_sha256']!=signature:raise ValueError('Saved forward ABI differs')
                    executable=deserialize_and_load(*load(binary_path))
                else:
                    ledger_path=FAMILY/'model/compile-budget.json'
                    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[]}
                    if horizon not in (7,30) or len(ledger['attempts'])>=2 or any(r['horizon']==horizon for r in ledger['attempts']):raise ValueError('Two model forward graph/attempt allowance exhausted')
                    record=dict(horizon=horizon,signature_sha256=signature,status='lower_started',start_UTC_seconds=time.time());ledger['attempts'].append(record);write(ledger_path,ledger)
                    owner.emit('forward_lowering',horizon=horizon)
                    began=time.perf_counter();lowered=owner.original.lower(*arguments);lower_seconds=time.perf_counter()-began
                    record.update(status='backend_started',lower_seconds=lower_seconds);write(ledger_path,ledger);owner.emit('forward_compiling',horizon=horizon,lower_seconds=lower_seconds)
                    began=time.perf_counter();executable=lowered.compile();backend_seconds=time.perf_counter()-began
                    from scripts.hpc.probe_one_day_batch import _compiled_memory
                    memory=_compiled_memory(executable)
                    if memory['estimated_live_bytes']>24*2**30:raise MemoryError('Compiled forward 24GiB gate')
                    from jax.experimental.serialize_executable import serialize
                    save(binary_path,serialize(executable))
                    metadata=dict(horizon=horizon,signature_sha256=signature,lower_seconds=lower_seconds,backend_seconds=backend_seconds,compiled_memory=memory,
                        returned_observations=True,canonical_original_path=True,source_head=HEAD)
                    write(metadata_path,metadata);record.update(status='saved',backend_seconds=backend_seconds);write(ledger_path,ledger)
                class Executable:
                    def __call__(self,*values):
                        identity=owner.report['current'];rawpath=FAMILY/'model/raw'/f"{identity['tag']}-T{horizon}-{identity['ordinal']}.pkl"
                        if rawpath.exists():return owner.jax.device_put(load(rawpath)['result'])
                        owner.report['model_calls_started']+=1;owner.emit('forward_call_started',horizon=horizon)
                        began=time.perf_counter();result=owner.jax.block_until_ready(executable(*values));wall=time.perf_counter()-began
                        memory=owner.jax.local_devices()[0].memory_stats() or {}
                        save(rawpath,dict(identity=identity.copy(),horizon=horizon,result=owner.jax.device_get(result),wall_seconds=wall,memory=memory))
                        owner.report['model_calls']+=1;owner.emit('forward_raw_saved',horizon=horizon)
                        if memory.get('peak_bytes_in_use',24*2**30+1)>24*2**30:raise MemoryError('Active GPU memory 24GiB gate')
                        return result
                    def memory_analysis(self):return executable.memory_analysis()
                owner.cache[horizon]=Executable();owner.report['forward_executables'][str(horizon)]=metadata
                owner.emit('forward_ready',horizon=horizon);return owner.cache[horizon]
        return Pending()

class JaxFacade:
    def __init__(self,jax,report,emit):self.actual=jax;self.report=report;self.emit=emit;self.forward=None
    def __getattr__(self,name):return getattr(self.actual,name)
    def jit(self,function,*args,**kwargs):
        if getattr(function,'__name__','')=='rollout':
            if self.forward is not None:raise ValueError('Only one public monitor rollout owner')
            self.forward=TrackedForward(self.actual,function,self.report,self.emit);return self.forward
        return self.actual.jit(function,*args,**kwargs)

def execute_gpu():
    began=time.perf_counter();modeldir=FAMILY/'model';modeldir.mkdir(exist_ok=True)
    report=dict(status='running_original_forward_science_validation',model_calls=0,model_calls_started=0,model_gradient_compiles=0,
        optimizer_executions=0,training_updates=0,forward_executables={},panels=[],scientific_acceptance=False,production_adoption=False)
    previous=modeldir/'model.json'
    if previous.exists():report=json.loads(previous.read_text());report['status']='running_original_forward_science_validation'
    def emit(stage,**values):
        report.update(stage=stage,process_elapsed_seconds=time.perf_counter()-began,**values);write(previous,report);print(stage,flush=True)
    timer=threading.Timer(float(os.environ['SCIENCE_PROCESS_REMAINING_SECONDS']),lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
        initialize_gpu_backend()
        import jax
        import jax.numpy as jnp
        from research.daily_coarse_graining.daily_operator import shared_process_head
        if jax.__version__!='0.4.38' or len(jax.local_devices())!=1 or jax.local_devices()[0].platform!='gpu':raise ValueError('Canonical single GPU JAX0.4.38 required')
        preflight=json.loads((FAMILY/'preflight.json').read_text())
        if not preflight.get('passed') or preflight.get('actual_backend')!='gpu':raise ValueError('Actual runtime preflight missing')
        context=load(CONTEXT);profile=json.loads((DIAG/'shared-split-profile-c599236.json').read_text());reference=json.loads(Path(profile['technical_update_report']).read_text())
        if context['identity']['source_git_head']!=HEAD or context['identity']['method']!=reference['method'] or context['identity']['artifact_id']!=742:raise ValueError('Prepared original context identity differs')
        owner=module(ROOT/'runtime/transfer/shared-combined-recurrent-20260926/probe_shared_expert_real_daily.py','scripts.hpc.science_original_execution_owner')
        adapter=owner.load_staged_adapter(ROOT/'runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py')
        backbone=owner.load_backbone_assembler(ROOT/'runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py')
        FreshMethodValidationPool=module(OVERLAY/'fresh_method_validation_pool.py','scripts.hpc.fresh_method_validation_pool').FreshMethodValidationPool
        public_monitor=module(OVERLAY/'shared_learning_monitor.py','scripts.hpc.shared_learning_monitor')
        overlap=json.loads((FAMILY/'training-overlap.json').read_text())
        if overlap['status']!='passed_no_exact_day_or_window_overlap' or any(a['exact_teacher_day_overlap_count'] or a['exact_training_window_match'] for a in overlap['selected_window_audits']):raise ValueError('Actual training overlap admission missing')
        selection=json.loads((FAMILY/'selection.json').read_text());rows=selection['rows'];pool=FreshMethodValidationPool(VALIDATION)
        if [r['artifact_id'] for r in rows]!=list(ORDINALS):raise ValueError('Frozen selection differs')
        public_rows={r['artifact_id']:r for r in pool.rows}
        for row in rows:
            if any(row[k]!=v for k,v in public_rows[row['artifact_id']].items()):raise ValueError('Selection no longer matches public prepared pool')
        original_validate=public_monitor._validate_rows
        def validate_subset(requested,role):
            if role!=public_monitor.HELD_OUT_ROLE or list(requested)!=rows:raise ValueError('Only declared four-row diagnostic subset allowed')
            return tuple(requested)
        public_monitor._validate_rows=validate_subset
        facade=JaxFacade(jax,report,emit);runtime=dict(context['runtime'],jax=facade)
        monitor=public_monitor.build_shared_learning_monitor(runtime,context['head'],context['template'],context['state_scales'],context['science_scales'])
        class Pool:
            def case(self,jax_arg,row):
                report['current']=dict(tag=report['tag'],ordinal=row['artifact_id'],step=report['step'],arm=report['arm']);emit('window_inputs')
                return pool.case(jax_arg,row)
        with owner.candidate_process_head_patches(shared_process_head,adapter,backbone):
            # The patch is the admitted ORIGINAL execution path, shared by all weights.
            for step in STEPS:
                for arm in ARMS:
                    tag=f'step-{step:02d}-{arm}';output=modeldir/(tag+'.science.json')
                    if output.exists():continue
                    report.update(tag=tag,step=step,arm=arm);emit('parameter_load')
                    p=arrays_load(FAMILY/'parameters'/(tag+'.pkl'))['parameters'];parameter_abi(p);p=jax.device_put(p)
                    result=monitor(p,Pool(),rows,panel_name=tag,split_role=public_monitor.HELD_OUT_ROLE)
                    result.update(diagnostic_step=step,arm=arm,original_execution_all_arms=True,selection=selection)
                    write(output,result);report['panels'].append(dict(step=step,arm=arm,path=str(output),physical_gate_passed=result['physical_gate_passed']));emit('panel_saved')
                    del p;gc.collect()
        public_monitor._validate_rows=original_validate
        expected=[(s,a) for s in STEPS for a in ARMS]
        for step,arm in expected:
            if not (modeldir/f'step-{step:02d}-{arm}.science.json').exists():raise ValueError('Panel coverage incomplete')
        report.update(status='completed_not_adopted',selected_window_evaluations=24,model_forward_compiles=len(json.loads((modeldir/'compile-budget.json').read_text())['attempts']),
            device_memory=jax.local_devices()[0].memory_stats(),strict_failures_preserved=scope()['strict_failures_preserved'])
        emit('completed');return 0
    except BaseException as error:
        emit('exception',status='runtime_or_physical_failure',error_type=type(error).__name__,error=str(error));raise
    finally:timer.cancel()

def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-cpu',action='store_true');group.add_argument('--execute-approved',action='store_true');group.add_argument('--scope-only',action='store_true')
    args=parser.parse_args()
    if args.scope_only:print(json.dumps(scope(),indent=2));return 0
    if args.prepare_cpu:prepare_cpu();return 0
    return execute_gpu()

if __name__=='__main__':raise SystemExit(main())
