"""Bounded native-checkpointed parameter AD at a frozen trained T7/T30 state."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training
from scripts.hpc.trained_screen_common import WindowPool, json_safe, load_model


def rademacher_direction(jax,parameters,seed):
    """Exact +/-1 tangents keep projected derivatives numerically resolvable."""
    generator = np.random.Generator(np.random.PCG64(seed))
    leaves,tree = jax.tree_util.tree_flatten(parameters)
    values = [(2*generator.integers(0,2,size=p.shape,dtype=np.int8)-1).astype(p.dtype) for p in leaves]
    return jax.tree_util.tree_unflatten(tree,values),sum(v.size for v in values)


def selected_window(selection,horizon):
    """Use panel order, without selecting cases by their model results."""
    if horizon not in (7,30):
        raise ValueError('trained AD horizon must be 7 or 30')
    return next(row for row in selection['windows'] if row['length']==horizon)


def recurrent_probe_loss(runtime, head, protocol_path):
    """Bind declared physical scales/weights to the shared recurrent scalar."""
    from research.daily_coarse_graining.daily_operator.fresh_start_training import _canopy_change_plan
    from research.daily_coarse_graining.daily_operator.recurrent_science_loss import (
        CANOPY_KEY,
        FLUX_KEYS,
        make_recurrent_science_loss,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import derive_long_horizon_scales
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        derive_target_oriented_acceptance_scales,
    )

    configuration = json.loads(protocol_path.read_text())
    if configuration['schema_version'] != 'recurrent_science_probe_v2':
        raise ValueError('unknown recurrent science probe protocol')
    gpp_scale = runtime['process_statistics'].outputs['daily.fast_gpp'].scale
    flux = derive_target_oriented_acceptance_scales(
        runtime['base_statistics'].arrays['next_state'].scale, gpp_scale, head.contract)
    delta = derive_long_horizon_scales(runtime['base_statistics'].arrays['state_delta'].scale,
                                     gpp_scale, head.contract)
    canopy = _canopy_change_plan(runtime, head.contract, {'canopy_carbon_stock_change':
        {'weight': 1., 'scale_source_pft_index': 13}})
    flux_scales = {key: flux.fluxes[key] for key in FLUX_KEYS}
    stock_scales = {key: delta.stocks[key] for key in
                    ('aboveground_biomass_carbon', 'belowground_biomass_carbon')}
    stock_scales[CANOPY_KEY] = canopy.scale
    import jax.numpy as jnp

    scalar = make_recurrent_science_loss(runtime['jax'], jnp, contract=head.contract,
        flux_scales=flux_scales, stock_daily_change_scales=stock_scales,
        weights=configuration['weights'])
    return scalar, {'configuration': configuration, 'flux_scales': flux_scales,
                    'stock_daily_change_scales': stock_scales}


def run(args):
    from scripts.hpc.probe_public_rollout_staged import StageStore
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    if args.admission_only or args.cpu_forward_only:
        raise ValueError('use --phase reverse/verify or --shape-only')
    started = time.perf_counter()
    report = {'status':'running','phase':args.phase,'updates':0,
              'source_git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'scope':f'trained_T{args.horizon}_parameter_directional_AD_only','directions':[]}

    def emit(stage,**values):
        report.update(values,stage=stage,elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output,json_safe(report))
        print(f'trained_ad stage={stage} seconds={report["elapsed_seconds"]:.1f}',flush=True)

    try:
        emit('bootstrap')
        runtime,head,template,scales,parameters,method = load_model(args)
        jax = runtime['jax']
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.canonical_rollout import (
            make_canonical_step,
            make_science_probe_loss,
        )
        from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
        from scripts.hpc.diagnose_public_rollout_discrepancy import project
        from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
        from scripts.hpc.probe_public_rollout import physical_checks
        from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report

        pool = WindowPool(args.selection,method['source_identity'])
        row = selected_window(pool.selection,args.horizon)
        initial,discrete,days,targets,expected_discrete = pool.case(jax,row)
        context = {'stock_weight':jnp.array(1.,dtype=jnp.float64)}
        recurrent = getattr(args, 'recurrent_protocol', None)
        if recurrent is None:
            loss_fn = make_science_probe_loss(head.contract,scales)
            loss_specification = None
        else:
            loss_fn, loss_specification = recurrent_probe_loss(runtime, head, recurrent)
            if (pool.selection.get('role') != loss_specification['configuration']['selection_role']
                    or args.horizon != loss_specification['configuration']['horizon']):
                raise ValueError('recurrent admission needs its declared TRAIN selection/horizon')
        component = make_rollout_loss(jax,jnp,make_canonical_step(head,template),loss_fn)
        arguments = (parameters,initial,discrete,days,targets,context)
        identity = {'method':method,'window':row,'probe':f'trained_T{args.horizon}_native_parameter_ad_v1',
                    'direction_seeds':[20260913,20260914],'direction_scale':'exact_per_coordinate_plus_minus_one'}
        if recurrent is not None:
            identity.update(probe='recurrent_science_v2_native_T7_parameter_ad',
                            loss_specification=loss_specification)
        store = StageStore(args.stage_directory,identity)
        native = jax.jit(jax.value_and_grad(component.checkpointed_loss,argnums=0))
        tangent = jax.jit(lambda p,d,s,z,day,t,c:
            jax.jvp(lambda v:component.checkpointed_loss(v,s,z,day,t,c),(p,),(d,)))
        direction,count = rademacher_direction(jax,parameters,20260913)
        direction = jax.tree_util.tree_map(jnp.asarray,direction)
        report.update(identity=identity,parameter_count=count)
        if args.shape_only:
            jax.eval_shape(native,*arguments)
            jax.eval_shape(tangent,parameters,direction,*arguments[1:])
            emit('completed',status='passed',classification=f'native_T{args.horizon}_parameter_grad_and_jvp_shapes_only')
            return report
        _,obs = jax.jit(component.rollout)(parameters,initial,discrete,days)
        hard = physical_checks(obs,targets,expected_discrete,days[0],head.contract,scales)
        if recurrent is not None:
            terms = jax.device_get(loss_fn.terms(*arguments[:2], obs, targets, context))
            for name in ('daily_flux','cumulative_flux','stock_endpoint','stock_trajectory'):
                for key, term in getattr(terms, name).items():
                    hard[f'{name}.{key}.invalid'] = int(term.invalid_prediction_count + term.invalid_target_count)
            report['recurrent_science_terms'] = json_safe({
                'total': terms.total, 'weighted_components': terms.weighted_components,
                **{name: {key: term._asdict() for key, term in getattr(terms, name).items()}
                   for name in ('daily_flux','cumulative_flux','stock_endpoint','stock_trajectory')}})
        emit('forward_completed',hard=hard)
        if any(hard.values()):
            emit('stopped',status='rejected',reason='physical_forward')
            return report
        if args.phase=='reverse':
            existing = store.read('native')
            if existing is not None:
                emit('completed',status='passed',result=existing['summary'],reused=True)
                return report
            emit('compile_native')
            before = time.perf_counter()
            executable = native.lower(*arguments).compile()
            memory = _compiled_memory(executable)
            emit('native_compiled',compiled_memory=memory,compile_seconds=time.perf_counter()-before)
            if memory['estimated_live_bytes']>18*1024**3:
                emit('stopped',status='resource_limited')
                return report
            value,gradient = executable(*arguments)
            jax.block_until_ready(gradient)
            numeric = _tree_numeric_report(jax,gradient)
            if not np.isfinite(float(value)) or not numeric['all_finite'] or numeric['l2_norm']<=0:
                emit('stopped',status='rejected',reason='invalid_native_gradient',gradient=numeric)
                return report
            result = {'value':float(value),'gradient':gradient,'summary':{'loss':float(value),'gradient':numeric,'memory':memory}}
            store.write('native',result,jax)
            emit('completed',status='passed',result=result['summary'],memory=_device_memory(runtime['devices'][0]))
            return report
        result = store.read('native')
        if result is None:
            raise ValueError('native reverse must finish and be saved before JVP verification')
        emit('compile_jvp')
        executable = tangent.lower(parameters,direction,*arguments[1:]).compile()
        memory = _compiled_memory(executable)
        emit('jvp_compiled',compiled_memory=memory)
        if memory['estimated_live_bytes']>18*1024**3:
            emit('stopped',status='resource_limited')
            return report
        for seed in identity['direction_seeds']:
            d,count = rademacher_direction(jax,parameters,seed)
            projection = project(jax,result['gradient'],d)
            primal,jvp = executable(parameters,jax.tree_util.tree_map(jnp.asarray,d),*arguments[1:])
            jvp,primal = float(jvp),float(primal)
            passed = bool(np.isfinite(jvp) and np.isclose(primal,result['value'],atol=1e-6,rtol=1e-5)
                          and np.isclose(jvp,projection,atol=1e-6,rtol=.01))
            report['directions'].append({'seed':seed,'norm_float64':float(np.sqrt(count)),
                'reverse_projection':projection,'jvp':jvp,'primal':primal,'absolute_error':abs(jvp-projection),
                'resolved_projection':max(abs(jvp),abs(projection))>1e-5,'passed':passed})
            emit('direction_completed')
        passed = all(r['passed'] and r['resolved_projection'] for r in report['directions'])
        emit('completed',status='passed' if passed else 'rejected',bounded_directional_gate_passed=passed,
             full_multi_day_admission=False,finite_difference_certified=False,memory=_device_memory(runtime['devices'][0]))
    except Exception as error:
        emit('failed',status='failed',error=f'{type(error).__name__}: {error}')
        raise
    return report


if __name__=='__main__':
    parser = training.parser()
    for name in ('selection','training-report','checkpoint','stage-directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=('reverse','verify'),required=True)
    parser.add_argument('--horizon',type=int,choices=(7,30),default=7)
    parser.add_argument('--recurrent-protocol',type=Path)
    parser.add_argument('--expected-update',type=int)
    parser.add_argument('--expected-pool-protocol',type=Path)
    run(parser.parse_args())
