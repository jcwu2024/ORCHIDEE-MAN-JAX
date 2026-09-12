"""Localize saved AD disagreement through replay observations and directional AD."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training
from scripts.hpc.probe_public_rollout_staged import StageStore, bootstrap_case, compare_stage


def project(jax, gradient, direction):
    return float(sum(np.vdot(np.asarray(g,dtype=np.float64),np.asarray(d,dtype=np.float64))
                     for g,d in zip(jax.tree_util.tree_leaves(gradient),jax.tree_util.tree_leaves(direction),strict=True)))


def disagreement_directions(jax, parameters, initial, reference, streamed):
    delta = jax.tree_util.tree_map(lambda a,b:np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64),
                                   streamed['gradient'],reference['gradient'])
    norms = [np.sqrt(project(jax,d,d)) for d in delta]
    zero_params = jax.tree_util.tree_map(lambda p:np.zeros(p.shape,dtype=p.dtype),parameters)
    directions = []
    if norms[0]>0:
        dp = jax.tree_util.tree_map(lambda d,p:(d/norms[0]).astype(p.dtype),delta[0],parameters)
        directions.append(('parameter_difference',(dp,np.zeros_like(initial))))
    if norms[1]>0:
        directions.append(('state_difference',(zero_params,(delta[1]/norms[1]).astype(initial.dtype))))
    return directions


def joint_jvp(component):
    jax = component.jax
    return jax.jit(lambda p,s,dp,ds,z,day,t,c:
        jax.jvp(lambda p1,s1:component.loss(p1,s1,z,day,t,c),(p,s),(dp,ds))[1])


def replay_difference(jax, replay, forward, *, labels):
    entries,nonfloating_mismatch = [],0
    left,lt = jax.tree_util.tree_flatten_with_path(replay)
    right,rt = jax.tree_util.tree_flatten_with_path(forward)
    if lt != rt:
        raise ValueError('VJP replay and saved forward observation structures differ')
    for (path,a),(_,b) in zip(left,right,strict=True):
        a,b = np.asarray(a),np.asarray(b)
        if not np.issubdtype(a.dtype,np.floating):
            nonfloating_mismatch += int(np.sum(a!=b))
            continue
        error = np.abs(a.astype(float)-b.astype(float))
        if np.any(error):
            entries.append({'path':str(path),'max_absolute':float(np.max(error)),
                            'error_l2':float(np.linalg.norm(error)),'different_coordinates':int(np.sum(error!=0.))})
    state_error = np.abs(np.asarray(replay['state'],dtype=float)-np.asarray(forward['state'],dtype=float))
    state_fields = []
    for leaf in labels:
        if leaf.start is not None:
            part = state_error[leaf.start:leaf.stop]
            if np.any(part):
                state_fields.append({'field':leaf.key,'max_absolute':float(np.max(part)),
                                     'error_l2':float(np.linalg.norm(part))})
    entries.sort(key=lambda r:r['max_absolute'],reverse=True)
    state_fields.sort(key=lambda r:r['max_absolute'],reverse=True)
    return {'nonfloating_mismatch':nonfloating_mismatch,'floating_differences':entries,
            'state_fields':state_fields,'exact_observations':not entries and nonfloating_mismatch==0}


def run(args):
    from scripts.hpc.probe_one_day_batch import _compiled_memory
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    report = {'status':'running','updates':0,'purpose':'discrepancy_diagnosis_not_admission',
              'source_git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'replay':[],'directions':[]}

    def emit(stage,**values):
        report.update(values,stage=stage,elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output,report)
        print(f'rollout_diagnosis stage={stage} seconds={report["elapsed_seconds"]:.1f}',flush=True)

    try:
        emit('bootstrap')
        protocol = json.loads(args.protocol.read_text())
        runtime,head,component,arguments,identity,physical = bootstrap_case(args,protocol)
        jax,jnp = component.jax,component.jnp
        original = StageStore(args.stage_directory,identity)
        reference,streamed = original.read('reference'),original.read('streamed')
        if reference is None or streamed is None:
            raise ValueError('diagnosis requires both saved full gradients')
        parameters,initial,discrete,days,targets,context = arguments
        report['case_identity'] = identity
        directions = disagreement_directions(jax,parameters,initial,reference,streamed)
        state_delta = np.asarray(streamed['gradient'][1])-np.asarray(reference['gradient'][1])
        state_index = int(np.argmax(np.abs(state_delta)))
        state_leaf = next(v for v in head.contract.state_leaves
                          if v.start is not None and v.start<=state_index<v.stop)
        raw_value = float(np.asarray(initial)[state_index])
        raw_mean = float(np.asarray(head.input_normalization.continuous_mean)[state_index])
        raw_scale = float(np.asarray(head.input_normalization.continuous_scale)[state_index])
        safe_scale = raw_scale if np.isfinite(raw_scale) and raw_scale>0 else 1.
        scale32 = np.float32(raw_scale)
        safe_scale32 = scale32 if np.isfinite(scale32) and scale32>0 else np.float32(1.)
        report['largest_state_coordinate'] = {'index':state_index,'field':state_leaf.key,
            'coordinate':list(np.unravel_index(state_index-state_leaf.start,state_leaf.shape)),
            'initial_value':raw_value,'mean':raw_mean,'scale':raw_scale,
            'standardized_float32_arithmetic':float((np.float32(raw_value)-np.float32(raw_mean))/safe_scale32),
            'standardized_float64_arithmetic':float((raw_value-raw_mean)/safe_scale)}
        for i,(name,(dp,ds)) in enumerate(directions):
            if name == 'state_difference':
                unit = np.zeros_like(ds)
                unit[state_index] = 1.
                directions[i] = ('largest_state_coordinate',(dp,unit))
        tangent = joint_jvp(component)
        if args.shape_only:
            if not directions:
                raise ValueError('no saved discrepancy direction to trace')
            dp,ds = jax.tree_util.tree_map(jnp.asarray,directions[0][1])
            jax.eval_shape(tangent,parameters,initial,dp,ds,discrete,days,targets,context)
            emit('completed',status='passed',classification='joint_parameter_state_jvp_shapes_only')
            return report
        rollout = jax.jit(component.rollout)
        _,baseline = rollout(parameters,initial,discrete,days)
        hard = physical(baseline)
        current_loss = float(component.loss_fn(parameters,initial,baseline,targets,context))
        if any(hard.values()) or not np.isclose(current_loss,reference['value'],atol=1e-6,rtol=1e-5):
            raise ValueError('diagnostic forward does not reproduce the admitted physical case')
        report.update(hard=hard,loss=current_loss)

        def observe(index,replay,forward):
            row = replay_difference(jax,replay,forward,labels=head.contract.state_leaves)
            report['replay'].append({'day':index+1,**row})
            emit('replay_observed')

        emit('replay_reverse')
        value,gradient = component.streamed_value_and_grad(*arguments,primal_callback=observe,
            callback=lambda event,data:emit('reverse_'+event,reverse_progress=data))
        repeated = {'value':float(value),'gradient':gradient}
        emit('replay_completed',repeat_streamed_comparison=compare_stage(jax,repeated,streamed,labels=head.contract.state_leaves))
        del gradient,repeated
        if not directions:
            emit('completed',status='completed',reason='no_saved_disagreement')
            return report
        # One combined JVP graph accepts either parameter or state tangents.
        first_dp,first_ds = jax.tree_util.tree_map(jnp.asarray,directions[0][1])
        tangent_args = (parameters,initial,first_dp,first_ds,discrete,days,targets,context)
        emit('compile_joint_jvp')
        jvp_executable = tangent.lower(*tangent_args).compile()
        memory = _compiled_memory(jvp_executable)
        emit('joint_jvp_compiled',jvp_memory=memory)
        if memory['estimated_live_bytes']>18*1024**3:
            emit('stopped',status='resource_limited',reason='joint_jvp_memory')
            return report
        loss = jax.jit(component.loss)
        for name,(dp_host,ds_host) in directions:
            dp,ds = jax.tree_util.tree_map(jnp.asarray,(dp_host,ds_host))
            direct_projection = project(jax,reference['gradient'],(dp_host,ds_host))
            streamed_projection = project(jax,streamed['gradient'],(dp_host,ds_host))
            nominal_jvp = float(jvp_executable(parameters,initial,dp,ds,discrete,days,targets,context))
            # Freeze scales from saved first-order sensitivity, not FD outcomes.
            sensitivity_center = .01*max(abs(current_loss),1e-6)/max(abs(direct_projection),abs(streamed_projection),1e-6)
            if name == 'largest_state_coordinate':
                center = max(8*abs(np.spacing(raw_value)),min(safe_scale*1e-4,sensitivity_center))
            else:
                center = min(1e-4,max(1e-10,sensitivity_center))
            row = {'name':name,'reference_projection':direct_projection,
                   'streamed_projection':streamed_projection,'nominal_jvp':nominal_jvp,
                   'fd_center_epsilon':center,'pairs':[]}
            report['directions'].append(row)
            emit('nominal_direction')
            for epsilon in (4*center,center,center/4):
                plus = jax.tree_util.tree_map(lambda p,d:(p+epsilon*d).astype(p.dtype),parameters,dp)
                minus = jax.tree_util.tree_map(lambda p,d:(p-epsilon*d).astype(p.dtype),parameters,dp)
                sp,sm = (initial+epsilon*ds).astype(initial.dtype),(initial-epsilon*ds).astype(initial.dtype)
                effective_dp = jax.tree_util.tree_map(lambda a,b,p:((a.astype('float64')-b.astype('float64'))/(2*epsilon)).astype(p.dtype),plus,minus,parameters)
                effective_ds = ((sp-sm)/(2*epsilon)).astype(initial.dtype)
                effective = jax.device_get((effective_dp,effective_ds))
                expected = project(jax,reference['gradient'],effective)
                actual = project(jax,streamed['gradient'],effective)
                jvp = float(jvp_executable(parameters,initial,effective_dp,effective_ds,discrete,days,targets,context))
                fp,fm = float(loss(plus,sp,discrete,days,targets,context)),float(loss(minus,sm,discrete,days,targets,context))
                fd = (fp-fm)/(2*epsilon)
                stable = True
                for p,s in ((plus,sp),(minus,sm)):
                    _,obs = rollout(p,s,discrete,days)
                    stable &= not any(physical(obs).values()) and np.array_equal(obs['defined'],baseline['defined'])
                row['pairs'].append({'epsilon':epsilon,'reference_projection':expected,
                    'streamed_projection':actual,'jvp':jvp,'fd':fd,'mask_physical_stable':bool(stable),
                    'reference_jvp_abs_error':abs(expected-jvp),'streamed_jvp_abs_error':abs(actual-jvp),
                    'reference_fd_abs_error':abs(expected-fd),'streamed_fd_abs_error':abs(actual-fd)})
                emit('direction_pair')
        emit('completed',status='completed',admission_claim=False,device_memory=runtime['devices'][0].memory_stats())
    except Exception as error:
        emit('failed',status='failed',error=f'{type(error).__name__}: {error}')
        raise
    return report


if __name__=='__main__':
    parser = training.parser()
    parser.set_defaults(protocol=training.ROOT/'manifests/coarse_graining/fresh_start_one_day_probe_v3.json')
    parser.add_argument('--stage-directory',type=Path,required=True)
    parser.add_argument('--horizon',type=int,choices=(2,),default=2)
    run(parser.parse_args())
