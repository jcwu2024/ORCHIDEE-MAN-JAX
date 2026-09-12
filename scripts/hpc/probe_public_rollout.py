"""Real-model reference/checkpointed/streamed gradient comparison; no training."""
from __future__ import annotations

import gzip
import json
import pickle
import subprocess
import time

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training


def gradient_comparison(jax, actual, expected, *, labels=None):
    rows, total_error, total_reference = [], 0., 0.
    a, tree = jax.tree_util.tree_flatten_with_path(actual)
    b, other = jax.tree_util.tree_flatten_with_path(expected)
    if tree != other:
        raise ValueError('gradient structures differ')
    for (path, left), (_, right) in zip(a,b,strict=True):
        left, right = np.asarray(left,dtype=float), np.asarray(right,dtype=float)
        if labels is None:
            parts = [(str(path), left, right)]
        else:
            parts = [(leaf.key,left[leaf.start:leaf.stop],right[leaf.start:leaf.stop])
                     for leaf in labels if leaf.start is not None]
        for name, left_part, right_part in parts:
            error = float(np.linalg.norm(left_part-right_part))
            reference = float(np.linalg.norm(right_part))
            rows.append({'field':name,'error_l2':error,'reference_l2':reference,
                         'passed':bool(np.isfinite(error) and error <=1e-6+1e-4*reference)})
            total_error += error*error
            total_reference += reference*reference
    rows.sort(key=lambda row:row['error_l2']/max(row['reference_l2'],1e-6),reverse=True)
    return {'passed':all(row['passed'] for row in rows), 'error_l2':np.sqrt(total_error),
            'reference_l2':np.sqrt(total_reference),'largest_relative_errors':rows[:12]}


def run(args):
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    started = time.perf_counter()
    report = {'status':'running','updates':0,'horizon':args.horizon,'methods':{},'comparisons':{}}

    def emit(stage, **values):
        report.update(values,stage=stage,elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output,report)
        print(f'public_rollout stage={stage} seconds={report["elapsed_seconds"]:.1f}',flush=True)

    try:
        emit('bootstrap')
        protocol = json.loads(args.protocol.read_text())
        report['model_protocol'] = protocol
        report['source_git_head'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        runtime,head,template,scales,parameters,_,_ = training.bootstrap(args,protocol)
        jax = runtime['jax']
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.canonical_rollout import (
            make_canonical_step,
            make_science_probe_loss,
            prepared_retained_days,
        )
        from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
        from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
        manifest = json.loads(args.prepared_manifest.read_text())
        record = next(v for v in manifest['entries'] if v['update']==322)
        with gzip.open(args.prepared_manifest.parent/record['path'],'rb') as handle:
            artifact = pickle.load(handle)
        window = artifact.batch.seven_day
        templates = jax.tree_util.tree_map(lambda v:v[1,:args.horizon],window.inputs)
        retained = prepared_retained_days(window.day_inputs,1,args.horizon)
        initial, discrete = templates.continuous_state[0], jax.tree_util.tree_map(lambda v:v[0],templates.discrete_state)
        days = (templates,retained)
        targets = {'state':window.expected_next_state.other[1,:args.horizon],
                   'defined':window.expected_next_state_defined.other[1,:args.horizon],
                   'gpp':window.expected_fast_gpp[1,:args.horizon],
                   'fraction':templates.pft.pft_fraction,'active':templates.pft.active_pft_mask}
        expected_discrete = jax.tree_util.tree_map(lambda v:v[1,:args.horizon],window.expected_next_discrete_state)
        context = {'stock_weight':jnp.array(1.,dtype=jnp.float64)}
        component = make_rollout_loss(jax,jnp,make_canonical_step(head,template),make_science_probe_loss(head.contract,scales))
        arguments = (parameters,initial,discrete,days,targets,context)
        reference_gradient = jax.jit(jax.value_and_grad(component.loss,argnums=(0,1)))
        checkpoint_gradient = jax.jit(jax.value_and_grad(component.checkpointed_loss,argnums=(0,1)))
        if args.shape_only:
            jax.eval_shape(reference_gradient,*arguments)
            jax.eval_shape(checkpoint_gradient,*arguments)
            emit('completed',status='passed',classification='canonical_full_gradient_shapes_only')
            return report
        rollout = jax.jit(component.rollout)
        _, baseline = rollout(parameters,initial,discrete,days)
        jax.block_until_ready(baseline)

        def physical(obs):
            nonfinite = int(np.sum(np.asarray(obs['defined']) & ~np.isfinite(np.asarray(obs['state']))))
            budgets = np.asarray(obs['budgets'])
            budget = int(np.sum(~np.isfinite(budgets) | (abs(budgets)>2e-12)))
            mismatch = int(np.sum((np.asarray(obs['defined']) != np.asarray(targets['defined']))
                                 & ~np.asarray(scales.state_declared_dynamic.other)[None,:]))
            discrete_mismatch = sum(int(np.sum(np.asarray(a)!=np.asarray(b))) for a,b in zip(
                jax.tree_util.tree_leaves(obs['discrete']),jax.tree_util.tree_leaves(expected_discrete),strict=True))
            gsmean = next(v for v in head.contract.state_leaves if v.key=='diffuco_previous_step_state.gsmean')
            bare = np.asarray(templates.pft.pft_traits)[:,:,0] > .5
            conductance = np.asarray(obs['state'])[:,gsmean.start:gsmean.stop].reshape((args.horizon,*gsmean.shape))
            return {'state_nonfinite':nonfinite,'budget_failure':budget,'mask_mismatch':mismatch,'discrete_mismatch':discrete_mismatch,
                    'gpp_nonfinite':int(np.sum(~np.isfinite(np.asarray(obs['gpp'])))),
                    'fraction_nonfinite':int(np.sum(~np.isfinite(np.asarray(obs['fraction'])))),
                    'bare_soil_gsmean_nonzero':int(np.sum(bare[:,None,:] & (conductance !=0.)))}

        hard = physical(baseline)
        emit('forward_completed',hard=hard)
        if any(hard.values()):
            emit('stopped',status='rejected',reason='physical_forward')
            return report
        gradients = {}
        for name, evaluator in (('direct',reference_gradient),('checkpointed',checkpoint_gradient)):
            emit('compile_'+name)
            compiled = evaluator.lower(*arguments).compile()
            memory = compiled.memory_analysis()
            required = memory.argument_size_in_bytes+memory.output_size_in_bytes+memory.temp_size_in_bytes-memory.alias_size_in_bytes
            report['methods'][name] = {'compiled_bytes':int(required)}
            emit('memory_'+name)
            if required>24*1024**3:
                emit('stopped',status='resource_limited',reason=name+'_reference_memory')
                return report
            value, gradient = compiled(*arguments)
            jax.block_until_ready(gradient)
            gradients[name] = gradient
            report['methods'][name].update(loss=float(value),parameters=_tree_numeric_report(jax,gradient[0]),
                                           initial_state=_tree_numeric_report(jax,gradient[1]))
            emit('completed_'+name)
            if not np.isfinite(float(value)) or not report['methods'][name]['parameters']['all_finite'] or not report['methods'][name]['initial_state']['all_finite']:
                emit('stopped',status='rejected',reason=name+'_nonfinite_gradient')
                return report
        emit('streamed')
        value, gradient = component.streamed_value_and_grad(*arguments,
            callback=lambda event,values:emit('streamed_'+event,stream_progress=values))
        gradients['streamed'] = gradient
        report['methods']['streamed'] = {'loss':float(value),'compiled_memory':component.reverse_memory,
                                         'reverse_compile_count':component.reverse_compile_count}
        for name in ('checkpointed','streamed'):
            report['comparisons'][name] = {
                'loss_agrees':bool(np.isclose(report['methods'][name]['loss'],report['methods']['direct']['loss'],atol=1e-6,rtol=1e-5)),
                'parameters':gradient_comparison(jax,gradients[name][0],gradients['direct'][0]),
                'initial_state':gradient_comparison(jax,gradients[name][1],gradients['direct'][1],labels=head.contract.state_leaves)}
        emit('gradient_comparisons')
        if not all(row['loss_agrees'] and row['parameters']['passed'] and row['initial_state']['passed'] for row in report['comparisons'].values()):
            emit('stopped',status='rejected',reason='gradient_comparison')
            return report
        norm = report['methods']['direct']['parameters']['l2_norm']
        if not np.isfinite(norm) or norm<=0:
            emit('stopped',status='rejected',reason='zero_or_nonfinite_reference_gradient')
            return report
        direction = jax.tree_util.tree_map(lambda g,p:(g/norm).astype(p.dtype),gradients['direct'][0],parameters)
        loss = jax.jit(component.loss)
        tangent = jax.jit(lambda p,d,s,z,day_inputs,target_values,loss_context:
            jax.jvp(lambda q:component.loss(q,s,z,day_inputs,target_values,loss_context),(p,),(d,))[1])
        pairs = []
        for epsilon in (3e-4,1e-4,3e-5):
            plus = jax.tree_util.tree_map(lambda p,d:(p+epsilon*d).astype(p.dtype),parameters,direction)
            minus = jax.tree_util.tree_map(lambda p,d:(p-epsilon*d).astype(p.dtype),parameters,direction)
            effective = jax.tree_util.tree_map(lambda a,b,p:((a.astype('float64')-b.astype('float64'))/(2*epsilon)).astype(p.dtype),plus,minus,parameters)
            fd = float((loss(plus,initial,discrete,days,targets,context)-loss(minus,initial,discrete,days,targets,context))/(2*epsilon))
            jvp = float(tangent(parameters,effective,initial,discrete,days,targets,context))
            reverse = float(sum(jnp.vdot(g.astype('float64'),d.astype('float64')) for g,d in zip(
                jax.tree_util.tree_leaves(gradients['streamed'][0]),jax.tree_util.tree_leaves(effective),strict=True)))
            stable = True
            for p in (plus,minus):
                _, obs = rollout(p,initial,discrete,days)
                stable &= not any(physical(obs).values()) and np.array_equal(obs['defined'],baseline['defined'])
            pairs.append({'epsilon':epsilon,'fd':fd,'jvp':jvp,'streamed_projection':reverse,'mask_physical_stable':bool(stable),
                          'passed':bool(stable and np.isclose(fd,reverse,atol=1e-6,rtol=.01) and np.isclose(jvp,reverse,atol=1e-6,rtol=.01))})
            emit('derivative_pair',derivative_pairs=pairs)
        emit('completed',status='completed',passed=all(v['passed'] for v in pairs),
             device_memory_stats=runtime['devices'][0].memory_stats())
    except Exception as exc:
        emit('failed',status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    return report


if __name__=='__main__':
    parser = training.parser()
    parser.add_argument('--horizon',type=int,choices=(1,2,3,7),default=2)
    run(parser.parse_args())
