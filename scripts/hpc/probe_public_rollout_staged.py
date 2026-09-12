"""Restartable real T2 AD comparison, preserving each expensive gradient."""
from __future__ import annotations

import gzip
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training
from scripts.hpc.probe_public_rollout import gradient_comparison, physical_checks


class StageStore:
    """Atomic probe artifacts; a completed reference survives later time limits."""

    def __init__(self, directory, identity):
        self.directory, self.identity = Path(directory), identity

    def read(self, name):
        path = self.directory/f'{name}.pkl'
        if not path.exists():
            return None
        with path.open('rb') as handle:
            record = pickle.load(handle)
        if record['identity'] != self.identity or record['stage'] != name:
            raise ValueError(f'saved {name} belongs to a different probe case')
        return record['result']

    def write(self, name, result, jax):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory/f'{name}.pkl'
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        try:
            with temporary.open('wb') as handle:
                pickle.dump({'identity':self.identity,'stage':name,'result':jax.device_get(result)},
                            handle,protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)


def gradient_stage(name, component, arguments, store, emit, *, memory_limit=18*1024**3):
    """Reuse a durable stage or execute exactly one differentiation method."""
    from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report

    cached = store.read(name)
    if cached is not None:
        emit('reuse_'+name,method=cached['summary'])
        return cached
    jax = component.jax
    metadata = {}
    if name == 'streamed':
        started = time.perf_counter()
        emit('streamed_started')
        value, gradient = component.streamed_value_and_grad(*arguments,
            callback=lambda event,values:emit('streamed_'+event,stream_progress=values))
        metadata.update(compiled_memory=component.reverse_memory,
                        reverse_compile_count=component.reverse_compile_count,
                        total_seconds=time.perf_counter()-started)
    else:
        loss = component.loss if name == 'reference' else component.checkpointed_loss
        # Keep the admitted direct AD graph and dynamic arguments unchanged.
        evaluator = jax.jit(jax.value_and_grad(loss,argnums=(0,1)))
        emit('compile_'+name)
        started = time.perf_counter()
        compiled = evaluator.lower(*arguments).compile()
        memory = compiled.memory_analysis()
        required = (memory.argument_size_in_bytes+memory.output_size_in_bytes
                    +memory.temp_size_in_bytes-memory.alias_size_in_bytes)
        metadata.update(compiled_bytes=int(required),compile_seconds=time.perf_counter()-started)
        emit('compiled_'+name,method=metadata)
        if required>memory_limit:
            raise RuntimeError(f'{name} exceeds the declared compiled-memory envelope')
        started = time.perf_counter()
        value,gradient = compiled(*arguments)
        jax.block_until_ready((value,gradient))
        metadata['evaluation_seconds'] = time.perf_counter()-started
    summary = {**metadata,'loss':float(value),
               'parameters':_tree_numeric_report(jax,gradient[0]),
               'initial_state':_tree_numeric_report(jax,gradient[1])}
    if not np.isfinite(float(value)) or not summary['parameters']['all_finite'] or not summary['initial_state']['all_finite']:
        raise RuntimeError(f'{name} produced a nonfinite value or gradient')
    result = {'value':float(value),'gradient':gradient,'summary':summary}
    # Persist before another compilation or comparison can consume the time limit.
    store.write(name,result,jax)
    emit('saved_'+name,method=summary)
    return result


def compare_stage(jax, actual, reference, *, labels=None):
    comparison = {
        'loss_agrees':bool(np.isclose(actual['value'],reference['value'],atol=1e-6,rtol=1e-5)),
        'parameters':gradient_comparison(jax,actual['gradient'][0],reference['gradient'][0]),
        'initial_state':gradient_comparison(jax,actual['gradient'][1],reference['gradient'][1],labels=labels),
    }
    comparison['passed'] = comparison['loss_agrees'] and comparison['parameters']['passed'] and comparison['initial_state']['passed']
    return comparison


def directional_stage(component, arguments, reference, streamed, store, physical, emit):
    """Preserve each rounded-direction JVP/FD result independently."""
    jax,jnp = component.jax,component.jnp
    parameters,initial,discrete,days,targets,context = arguments
    norm = reference['summary']['parameters']['l2_norm']
    if not np.isfinite(norm) or norm<=0:
        raise RuntimeError('reference parameter gradient is zero or nonfinite')
    direction = jax.tree_util.tree_map(lambda g,p:(jnp.asarray(g)/norm).astype(p.dtype),reference['gradient'][0],parameters)
    loss = jax.jit(component.loss)
    tangent = jax.jit(lambda p,d,s,z,day_inputs,target_values,loss_context:
        jax.jvp(lambda q:component.loss(q,s,z,day_inputs,target_values,loss_context),(p,),(d,))[1])
    rollout = jax.jit(component.rollout)
    _,baseline = rollout(parameters,initial,discrete,days)
    pairs = []
    for epsilon in (3e-4,1e-4,3e-5):
        key = f'derivative-{epsilon:.0e}'
        row = store.read(key)
        if row is None:
            emit('derivative_started',epsilon=epsilon)
            plus = jax.tree_util.tree_map(lambda p,d:(p+epsilon*d).astype(p.dtype),parameters,direction)
            minus = jax.tree_util.tree_map(lambda p,d:(p-epsilon*d).astype(p.dtype),parameters,direction)
            effective = jax.tree_util.tree_map(lambda a,b,p:((a.astype('float64')-b.astype('float64'))/(2*epsilon)).astype(p.dtype),plus,minus,parameters)
            fd = float((loss(plus,initial,discrete,days,targets,context)-loss(minus,initial,discrete,days,targets,context))/(2*epsilon))
            jvp = float(tangent(parameters,effective,initial,discrete,days,targets,context))
            reverse = float(sum(jnp.vdot(jnp.asarray(g,dtype='float64'),d.astype('float64')) for g,d in zip(
                jax.tree_util.tree_leaves(streamed['gradient'][0]),jax.tree_util.tree_leaves(effective),strict=True)))
            stable = True
            for p in (plus,minus):
                _,obs = rollout(p,initial,discrete,days)
                stable &= not any(physical(obs).values()) and np.array_equal(obs['defined'],baseline['defined'])
            row = {'epsilon':epsilon,'fd':fd,'jvp':jvp,'streamed_projection':reverse,'mask_physical_stable':bool(stable),
                   'passed':bool(stable and np.isclose(fd,reverse,atol=1e-6,rtol=.01) and np.isclose(jvp,reverse,atol=1e-6,rtol=.01))}
            store.write(key,row,jax)
        pairs.append(row)
        emit('derivative_saved',derivative_pairs=pairs)
    return {'passed':all(row['passed'] for row in pairs),'derivative_pairs':pairs}


def bootstrap_case(args, protocol):
    """Same artifact/window, binding, observations and loss as the admitted T2 graph."""
    runtime,head,template,scales,parameters,_,_ = training.bootstrap(args,protocol)
    jax = runtime['jax']
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.canonical_rollout import (
        make_canonical_step,
        make_science_probe_loss,
        prepared_retained_days,
    )
    from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss

    manifest = json.loads(args.prepared_manifest.read_text())
    record = next(v for v in manifest['entries'] if v['update']==322)
    with gzip.open(args.prepared_manifest.parent/record['path'],'rb') as handle:
        artifact = pickle.load(handle)
    window,horizon = artifact.batch.seven_day,args.horizon
    templates = jax.tree_util.tree_map(lambda v:v[1,:horizon],window.inputs)
    retained = prepared_retained_days(window.day_inputs,1,horizon)
    initial,discrete = templates.continuous_state[0],jax.tree_util.tree_map(lambda v:v[0],templates.discrete_state)
    targets = {'state':window.expected_next_state.other[1,:horizon],
               'defined':window.expected_next_state_defined.other[1,:horizon],
               'gpp':window.expected_fast_gpp[1,:horizon],
               'gpp_defined':window.expected_fast_gpp_defined[1,:horizon],
               'fraction':templates.pft.pft_fraction,'active':templates.pft.active_pft_mask}
    expected_discrete = jax.tree_util.tree_map(lambda v:v[1,:horizon],window.expected_next_discrete_state)
    context = {'stock_weight':jnp.array(1.,dtype=jnp.float64)}
    component = make_rollout_loss(jax,jnp,make_canonical_step(head,template),make_science_probe_loss(head.contract,scales),limit_bytes=18*1024**3)
    arguments = (parameters,initial,discrete,(templates,retained),targets,context)
    identity = {'schema_version':'public_rollout_stages_v2','model_protocol':protocol,
                'artifact_schedule':artifact.schedule_entry,'window_index':1,'horizon':horizon,
                'day_indices':np.asarray(templates.day_index).tolist(),
                'loss_version':'four_cumulative_fluxes_final_agb_bgb_active_mean_v1',
                'component_version':'full_state_fraction_teacher_masks_4410c41'}

    def physical(obs):
        return physical_checks(obs,targets,expected_discrete,templates,head.contract,scales)

    return runtime,head,component,arguments,identity,physical


def run(args):
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    if args.admission_only or args.cpu_forward_only:
        raise ValueError('select a --stage, or use --shape-only')
    started = time.perf_counter()
    report = {'status':'running','requested_stage':args.stage,'updates':0,
              'source_git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'stage_directory':str(args.stage_directory),'methods':{}}

    def emit(stage,**values):
        report.update(values,stage=stage,elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output,report)
        print(f'public_staged stage={stage} seconds={report["elapsed_seconds"]:.1f}',flush=True)

    try:
        emit('bootstrap')
        protocol = json.loads(args.protocol.read_text())
        runtime,head,component,arguments,identity,physical = bootstrap_case(args,protocol)
        jax = runtime['jax']
        store = StageStore(args.stage_directory,identity)
        report['identity'] = identity
        if args.shape_only:
            evaluator = jax.jit(jax.value_and_grad(component.loss,argnums=(0,1)))
            jax.eval_shape(evaluator,*arguments)
            emit('completed',status='passed',classification='unchanged_reference_gradient_shapes_only')
            return report
        parameters,initial,discrete,days,targets,context = arguments
        _,baseline = jax.jit(component.rollout)(parameters,initial,discrete,days)
        hard = physical(baseline)
        emit('forward_completed',hard=hard)
        if any(hard.values()):
            emit('stopped',status='rejected',reason='physical_forward',passed=False)
            return report
        if args.stage in ('core','reference'):
            reference = gradient_stage('reference',component,arguments,store,emit)
        else:
            reference = store.read('reference')
            if reference is None:
                raise ValueError('run the reference/core stage first')
        report['methods']['reference'] = reference['summary']
        current_loss = float(component.loss_fn(parameters,initial,baseline,targets,context))
        if not np.isclose(current_loss,reference['value'],atol=1e-6,rtol=1e-5):
            raise ValueError('saved reference does not reproduce current forward loss')
        if args.stage == 'reference':
            emit('completed',status='passed',passed=True,classification='reference_preserved_only')
            return report
        if args.stage in ('core','streamed','checkpointed'):
            name = 'checkpointed' if args.stage == 'checkpointed' else 'streamed'
            result = gradient_stage(name,component,arguments,store,emit)
            report['methods'][name] = result['summary']
            comparison = compare_stage(jax,result,reference,labels=head.contract.state_leaves)
            store.write(name+'-comparison',comparison,jax)
            emit('completed',status='passed' if comparison['passed'] else 'rejected',
                 passed=comparison['passed'],comparison=comparison,
                 classification=name+'_reference_comparison_only',device_memory=runtime['devices'][0].memory_stats())
        else:
            comparison,streamed = store.read('streamed-comparison'),store.read('streamed')
            if comparison is None or not comparison['passed'] or streamed is None:
                raise ValueError('directional checks require the passed streamed/reference comparison')
            checks = directional_stage(component,arguments,reference,streamed,store,physical,emit)
            store.write('derivatives',checks,jax)
            emit('completed',status='passed' if checks['passed'] else 'rejected',
                 **checks,classification='directional_checks_only',device_memory=runtime['devices'][0].memory_stats())
    except Exception as error:
        emit('failed',status='failed',error=f'{type(error).__name__}: {error}')
        raise
    return report


if __name__=='__main__':
    parser = training.parser()
    parser.set_defaults(protocol=training.ROOT/'manifests/coarse_graining/fresh_start_one_day_probe_v3.json')
    parser.add_argument('--stage',choices=('core','reference','streamed','derivatives','checkpointed'),default='core')
    parser.add_argument('--stage-directory',type=Path,required=True)
    parser.add_argument('--horizon',type=int,choices=(2,),default=2)
    run(parser.parse_args())
