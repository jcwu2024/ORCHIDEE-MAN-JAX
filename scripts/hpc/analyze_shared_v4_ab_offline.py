"""Saved-array V4 A/B evidence and CPU-only official component AdamW proposals.

No model evaluation, model AD compilation, GPU, Teacher IO or HLO read occurs.
State interval formulas follow analyze_shared_chunk_batch_state.py. AdamW uses
the real update0 checkpoint and the official make_update wrapper unchanged.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np

_local_checkout=Path(__file__).resolve().parents[2]
if (_local_checkout/'research').is_dir():sys.path.insert(0,str(_local_checkout))

RTOL,ATOL=1e-5,1e-6
COMMIT='c599236bcf57b745e9bafed678298010f4c3b58f'
DAILY='shared-chunk-vmap-daily-dual-v4-20261001'
T7='shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4'

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def load(path):
    with path.open('rb') as stream:return pickle.load(stream)

def entries(tree,path=()):
    if isinstance(tree,dict):
        yield path,'dict',tuple(sorted(tree))
        for key in sorted(tree):yield from entries(tree[key],path+(str(key),))
    elif isinstance(tree,(tuple,list)):
        yield path,'sequence',(type(tree).__name__,len(tree))
        for index,value in enumerate(tree):yield from entries(value,path+(str(index),))
    elif tree is None:yield path,'static_None',None
    else:yield path,'numeric',np.asarray(tree)

def compare(actual,reference,*,per_leaf=False):
    count=leaves=failed=static_none=nonzero=0
    delta2=reference2=actual2=0.; largest=0.; finite=True; rows=[]
    for left,right in zip(entries(actual),entries(reference),strict=True):
        path,kind,a=left; other,other_kind,b=right
        if path!=other or kind!=other_kind:raise ValueError('tree ABI/static None differs at '+str(path))
        if kind!='numeric':
            if a!=b:raise ValueError('container/static ABI differs at '+str(path))
            static_none+=kind=='static_None';continue
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('numeric shape/dtype ABI differs at '+str(path))
        if a.dtype.kind not in 'biufc':raise ValueError('unsupported nonnumeric saved leaf')
        finite=finite and bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b)))
        mask=~np.isclose(a,b,rtol=RTOL,atol=ATOL)
        difference=a.astype(np.float64)-b.astype(np.float64)
        local_failed=int(np.count_nonzero(mask)); local_max=float(np.max(np.abs(difference),initial=0))
        count+=a.size;leaves+=1;failed+=local_failed;nonzero+=int(np.count_nonzero(a))
        delta2+=float(np.sum(difference*difference,dtype=np.float64))
        reference2+=float(np.sum(np.square(b.astype(np.float64)),dtype=np.float64))
        actual2+=float(np.sum(np.square(a.astype(np.float64)),dtype=np.float64))
        largest=max(largest,local_max)
        if per_leaf:rows.append(dict(path='/'.join(path),shape=list(a.shape),dtype=str(a.dtype),
            coordinates=int(a.size),failed=local_failed,max_abs=local_max))
    result=dict(passed=finite and failed==0,finite=finite,numeric_leaves=leaves,
        static_None_nodes=static_none,coordinates=int(count),failed=failed,max_abs=largest,
        relative_L2=float(np.sqrt(delta2/max(reference2,1e-300))),
        actual_norm=float(np.sqrt(actual2)),actual_nonzero=nonzero)
    if per_leaf:result['per_leaf']=rows
    return result

def controls():
    check=compare({'a':np.ones(2,np.float32),'inactive':None},
                  {'a':np.ones(2,np.float32),'inactive':None})
    if check['coordinates']!=2 or check['numeric_leaves']!=1 or check['static_None_nodes']!=1 or not check['passed']:
        raise ValueError('static None positive control failed')
    rejected=[]
    for name,actual,reference in (
        ('static_None_mismatch',{'a':None},{'a':np.asarray(0.,np.float32)}),
        ('missing_static_None_key',{'a':np.ones(1,np.float32)},{'a':np.ones(1,np.float32),'b':None}),
        ('shape_mismatch',np.ones(2,np.float32),np.ones(1,np.float32))):
        try:compare(actual,reference)
        except ValueError:rejected.append(name)
        else:raise ValueError('ABI negative control accepted: '+name)
    if compare(np.asarray(1.001),np.asarray(1.))['passed']:raise ValueError('strict difference negative accepted')
    return dict(status='passed',static_None_positive=True,rejected_ABI_controls=rejected,
        strict_numeric_difference_rejected=True)

def intervals(original,candidate,a,b,source,daily):
    if original.shape!=((4,3854) if daily else (3854,)) or candidate.shape!=original.shape or a.shape!=b.shape or a.shape!=(7,)+original.shape:
        raise ValueError('saved complete state ABI differs')
    if not all(np.all(np.isfinite(x)) for x in (original,candidate,a,b)):raise ValueError('nonfinite saved state')
    initial=~np.isclose(candidate,original,rtol=RTOL,atol=ATOL)
    pairs=~np.isclose(b,a,rtol=RTOL,atol=ATOL)
    counts=[int(np.count_nonzero(x)) for x in pairs]
    recorded=[x['all_initial_state_gradients']['failed'] for x in source['full_derivative_pair_comparisons']] if daily else [x['failed'] for x in source['initial_state_pair_replay']]
    initial_count=source['comparisons']['all_initial_state_gradients' if daily else 'initial_state_gradient']['failed']
    if counts!=recorded or int(initial.sum())!=initial_count:raise ValueError('saved state arrays do not reproduce recorded strict gates')
    self_a=~np.isclose(a,original,rtol=RTOL,atol=ATOL);self_b=~np.isclose(b,candidate,rtol=RTOL,atol=ATOL)
    union=initial|np.any(pairs,axis=0);every=np.all(pairs,axis=0)
    lo_a,hi_a,lo_b,hi_b=a.min(0),a.max(0),b.min(0),b.max(0)
    gap=np.maximum(np.maximum(lo_b-hi_a,lo_a-hi_b),0)
    allowance=ATOL+RTOL*np.max(np.abs(a),axis=0)
    rows=[]
    for arr in np.argwhere(union):
        index=tuple(int(i) for i in arr)
        delta=np.abs(b[(slice(None),)+index]-a[(slice(None),)+index])
        rows.append(dict(index=list(index),initial_default=float(original[index]),initial_candidate=float(candidate[index]),
            initial_failed=bool(initial[index]),initial_abs_error=float(abs(candidate[index]-original[index])),
            initial_allowance=float(ATOL+RTOL*abs(original[index])),
            default_repeats=a[(slice(None),)+index].tolist(),candidate_repeats=b[(slice(None),)+index].tolist(),
            failed_pair_indices=np.flatnonzero(pairs[(slice(None),)+index]).tolist(),
            default_self_failed_indices=np.flatnonzero(self_a[(slice(None),)+index]).tolist(),
            candidate_self_failed_indices=np.flatnonzero(self_b[(slice(None),)+index]).tolist(),
            default_range=[float(lo_a[index]),float(hi_a[index])],candidate_range=[float(lo_b[index]),float(hi_b[index])],
            paired_abs_error_range=[float(delta.min()),float(delta.max())],range_gap=float(gap[index]),
            largest_allowance=float(allowance[index]),failed_every_pair=bool(every[index]),
            disjoint_ranges_beyond_allowance=bool(gap[index]>allowance[index]),
            both_arms_self_pass=not bool(np.any(self_a[(slice(None),)+index]) or np.any(self_b[(slice(None),)+index]))))
    return dict(initial_failed=int(initial.sum()),paired_failure_counts=counts,union_failed=int(union.sum()),
        failed_every_pair=int(every.sum()),persistent_disjoint_beyond_allowance=int(np.count_nonzero(every&(gap>allowance))),
        original_self_counts=[int(x.sum()) for x in self_a],candidate_self_counts=[int(x.sum()) for x in self_b],
        coordinates=rows)

def paths(root,remote=False):
    daily=root if remote else root/'shared-chunk-vmap-daily-dual-v4-20261001'
    seven=root if remote else root/'shared-chunk-vmap-recurrent-20261001'
    return dict(daily=dict(report=daily/(DAILY+'-candidate.json'),reference=daily/(DAILY+'-candidate.field-replay.pkl'),
        candidate=daily/(DAILY+'.candidate.pkl'),repeats=daily/(DAILY+'-candidate.state-replays.pkl')),
        T7=dict(report=seven/(T7+'.json'),reference=seven/(T7+'.reference-replay.pkl'),candidate=seven/(T7+'.candidate.pkl')))

def hard_counts(value):
    hard=value[0][1]['hard']
    return {name:int(np.sum(np.asarray(count),dtype=np.int64)) for name,count in hard.items()}

def local_analysis(args):
    records={}; manifest=[]
    write(args.output/'analyzer-controls.json',controls())
    for component,files in paths(args.root).items():
        report=json.loads(files['report'].read_text());reference=load(files['reference']);candidate=load(files['candidate'])
        for arm,artifact in (('reference',reference),('candidate',candidate)):
            manifest.append(dict(component=component,arm=arm,path=str(files[arm]),bytes=files[arm].stat().st_size,
                top_level_keys=sorted(artifact),contains_parameters='parameters' in artifact,contains_optimizer='optimizer' in artifact))
        numeric=compare(candidate['result'][1][0],reference['result'][1][0],per_leaf=True)
        if numeric['numeric_leaves']!=300 or numeric['coordinates']!=64884481:raise ValueError('full parameter ABI differs')
        write(args.output/(component+'-parameter-gradients.json'),numeric)
        state_original=np.asarray(reference['result'][1][1]);state_candidate=np.asarray(candidate['result'][1][1])
        if component=='daily':
            repeats=load(files['repeats']);a=np.stack(repeats['field']);b=np.stack(repeats['candidate'])
        else:
            prefix=files['report'].with_suffix('')
            a=np.stack([load(Path(str(prefix)+f'.default.repeat{i}.initial_state_vjp.pkl')) for i in range(7)])
            b=np.stack([load(Path(str(prefix)+f'.candidate.repeat{i}.initial_state_vjp.pkl')) for i in range(7)])
        state=intervals(state_original,state_candidate,a,b,report,component=='daily')
        write(args.output/(component+'-state-intervals.json'),state)
        original_hard=hard_counts(reference['result']);candidate_hard=hard_counts(candidate['result'])
        if len(candidate_hard)!=(11 if component=='daily' else 14) or any(original_hard.values()) or any(candidate_hard.values()):
            raise ValueError('raw physical hard counts differ from expected scope')
        timing=report['matched_warm_seconds'];original_times=timing['field' if component=='daily' else 'default'];new_times=timing['candidate']
        ratio=float(np.median(original_times)/np.median(new_times))
        records[component]=dict(source_report=str(files['report']),loss=compare(candidate['result'][0][0],reference['result'][0][0]),
            auxiliary=compare(candidate['result'][0][1],reference['result'][0][1]),parameter_gradient={k:v for k,v in numeric.items() if k!='per_leaf'},
            physical_counts=dict(reference=original_hard,candidate=candidate_hard),
            original_times=original_times,candidate_times=new_times,median_ratio=ratio,
            seven_pair_parameter_evidence=('all seven recorded full comparisons pass' if component=='daily' else 'not captured; seven repeats contain state only'),
            state_summary={k:v for k,v in state.items() if k!='coordinates'},
            state_gradients_enter_optimizer=False,internal_recurrent_state_adjoints_observed=False)
        if component=='daily':
            recorded=report['full_derivative_pair_comparisons']
            if len(recorded)!=7 or any(not r['all_parameter_gradients']['passed'] or not r['loss']['passed'] or not r['auxiliary']['passed'] for r in recorded):
                raise ValueError('daily saved seven-pair A records not passing')
        del reference,candidate;gc.collect()
    write(args.output/'local-evidence.json',dict(status='analyzed',rtol=RTOL,atol=ATOL,components=records,
        artifact_manifest=manifest,GPU_calls=0,model_AD_compiles=0,Teacher_reads=0,training_updates=0,
        retained_reference_adoption=False,full_update_admission=False))
    print(json.dumps(records,indent=2))

def optimizer_analysis(args):
    import jax
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
    from research.daily_coarse_graining.daily_operator.target_oriented_training import target_oriented_adamw_update
    import inspect
    import subprocess
    checkout=Path(os.environ['ORCHIDEE_REPO_ROOT'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()!=COMMIT:
        raise ValueError('official immutable checkout identity differs')
    if not all(Path(inspect.getsourcefile(f)).resolve().is_relative_to(checkout.resolve()) for f in (make_update,target_oriented_adamw_update)):
        raise ValueError('official optimizer source is outside declared checkout')
    if any(args.output.glob('*-official-adamw.pkl')):raise FileExistsError('preserve previous official proposals; no retry')
    if jax.default_backend()!='cpu':raise ValueError('offline optimizer requires CPU')
    jax.config.update('jax_enable_x64',True)
    reports=paths(args.root,remote=True)
    daily_report=json.loads(reports['daily']['report'].read_text());method=daily_report['identity']['method']
    checkpoint_path=Path(daily_report['identity']['initial_checkpoint'])
    checkpoint=load_learning_checkpoint(checkpoint_path,expected_method=method)
    if checkpoint['completed_updates']!=0 or int(np.asarray(checkpoint['optimizer'].step))!=0:raise ValueError('real update0 required')
    parameter_abi=compare(checkpoint['parameters'],checkpoint['parameters'])
    if parameter_abi['numeric_leaves']!=300 or parameter_abi['coordinates']!=64884481:raise ValueError('real parameter ABI differs')
    for name in ('first','second'):
        abi=compare(getattr(checkpoint['optimizer'],name),checkpoint['parameters'])
        if abi['coordinates']!=64884481 or abi['numeric_leaves']!=300:raise ValueError('optimizer moment ABI differs')
    # Model differentiation and objective execution are forbidden by this mode.
    def forbidden(*_args,**_kwargs):raise RuntimeError('model AD forbidden in saved-gradient AdamW diagnostic')
    for name in ('grad','value_and_grad','jvp','vjp','linearize','jacfwd','jacrev'):setattr(jax,name,forbidden)
    update=make_update(method['daily_protocol']);compiled=None;records={};began=time.perf_counter()
    parameters,optimizer=jax.device_put((checkpoint['parameters'],checkpoint['optimizer']))
    for component,files in reports.items():
        results={};original=None
        for arm in ('reference','candidate'):
            saved=load(files[arm]);identity=saved['identity']
            declared=identity.get('initial_checkpoint',identity.get('checkpoint'))
            if identity['method']!=method or declared!=str(checkpoint_path):raise ValueError('saved result/checkpoint method identity differs')
            gradient=saved['result'][1][0]
            abi=compare(gradient,checkpoint['parameters'])
            if abi['coordinates']!=64884481 or abi['numeric_leaves']!=300 or not abi['finite']:raise ValueError('complete saved finite gradient ABI differs')
            device_gradient=jax.device_put(gradient)
            if compiled is None:
                start=time.perf_counter();compiled=update.lower(parameters,device_gradient,optimizer).compile();compile_seconds=time.perf_counter()-start
            start=time.perf_counter();value=jax.device_get(jax.block_until_ready(compiled(parameters,device_gradient,optimizer)));seconds=time.perf_counter()-start
            updated,new_optimizer,norm,scale=value
            if int(np.asarray(new_optimizer.step))!=1:raise ValueError('official update did not advance0 to1')
            path=args.output/(component+'-'+arm+'-official-adamw.pkl')
            with path.open('wb') as stream:pickle.dump(dict(component=component,arm=arm,input_checkpoint=str(checkpoint_path),
                input_completed_updates=0,method=method,gradient_source=str(files[arm]),updated_parameters=updated,
                optimizer=new_optimizer,raw_gradient_norm=norm,clip_scale=scale),stream,protocol=pickle.HIGHEST_PROTOCOL)
            results[arm]=value
            del saved,gradient,device_gradient;gc.collect()
        checks=dict(updated_parameters=compare(results['candidate'][0],results['reference'][0],per_leaf=True),
            optimizer=compare(results['candidate'][1],results['reference'][1],per_leaf=True),
            norm_and_clip=compare(results['candidate'][2:],results['reference'][2:]))
        write(args.output/(component+'-official-adamw-comparison.json'),checks)
        records[component]=dict(passed=all(row['passed'] for row in checks.values()),
            comparisons={k:{key:v for key,v in row.items() if key!='per_leaf'} for k,row in checks.items()},
            result_paths=[str(args.output/(component+'-'+arm+'-official-adamw.pkl')) for arm in ('reference','candidate')],
            raw_norms=[float(results[arm][2]) for arm in ('reference','candidate')],
            clip_scales=[float(results[arm][3]) for arm in ('reference','candidate')])
        del results;gc.collect()
    provenance=[]
    for function in (make_update,target_oriented_adamw_update):
        lines,start=inspect.getsourcelines(function)
        provenance.append(dict(file=inspect.getsourcefile(function),function=function.__name__,start=start,end=start+len(lines)-1))
    report=dict(status='completed',backend='cpu',jax_version=jax.__version__,checkpoint=str(checkpoint_path),
        completed_updates=0,input_optimizer_step=0,result_optimizer_step=1,parameter_ABI=parameter_abi,
        official_optimizer=method['daily_protocol']['optimizer'],source_provenance=provenance,components=records,
        official_optimizer_helper_compiles=1,helper_compile_seconds=compile_seconds,elapsed_seconds=time.perf_counter()-began,
        scope='separate fixed-weight daily-micro4 and T7 saved-gradient component proposals',
        mixed_daily32_T7_T30_update=False,changing_weight_same_workload_admission=False,model_evaluations=0,
        model_AD_compiles=0,GPU_calls=0,Teacher_reads=0,training_updates=0,production_adoption=False)
    write(args.output/'official-adamw-summary.json',report);print(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('local','optimizer'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='local':local_analysis(args)
    else:
        try:optimizer_analysis(args)
        except BaseException as error:
            write(args.output/'official-adamw-error.json',dict(status='failed',error_type=type(error).__name__,
                error=str(error),GPU_calls=0,model_AD_compiles=0,training_updates=0));raise

if __name__=='__main__':main()
