"""Fresh full-network learning with daily, T7 and T30 means and exact recovery.

This runner composes admitted objectives; it does not implement an adjoint.
The optional integration budget uses the same update/monitor/checkpoint path
in a separate run identity and cannot become the 200-update scientific run.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import probe_feedback_loss_updates as admission
from scripts.hpc import run_fresh_start_one_day as prototype


def validate_configuration(config, model):
    if (config.get('schema_version') != 'mixed_learning_v1'
            or config.get('updates') != 200
            or config.get('milestones') != [0, 50, 100, 200]
            or config.get('outer_weights') != {'daily': 1., 'T7': .5, 'T30': .5}
            or config.get('sampling') != dict(seed=20260912, batch_size=64,
                microbatch_size=8, max_artifacts_per_batch=5, batch_packing='fixed_replay_pad')
            or model.get('seed') != 20260911
            or model.get('parameter_count') != 109464375
            or model.get('trainable_scope') != 'all_neural_parameters'
            or model['optimizer']['schedule'] != 'constant_200_updates'):
        raise ValueError('mixed learning differs from declared fresh 200-update protocol')


def checkpoint_path(output, index=None):
    suffix = 'recovery' if index is None else f'update-{index:06d}'
    return output.with_name(f'{output.stem}.{suffix}.checkpoint.pkl')


def trim_resume_report(report, completed):
    """A report can be ahead of an atomic checkpoint after an interrupted write."""
    report['updates'] = [r for r in report['updates'] if r['update'] <= completed]
    if [r['update'] for r in report['updates']] != list(range(1,completed+1)):
        raise ValueError('recovery report is missing completed update records')
    report['monitors'] = [r for r in report['monitors'] if r['update'] <= completed]
    report['checkpoints'] = [r for r in report.get('checkpoints',[]) if r['update'] <= completed]
    for key in tuple(report):
        if key.startswith('failed_') or key in ('error','monitor_update','next_update',
                'classification','checkpoint_forward_replay'):
            report.pop(key)
    report['status'] = 'running'
    report['completed_updates'] = completed
    return report


def newest_checkpoint(output, requested, completed):
    """Prefer a newer retained milestone over an older rolling recovery file."""
    available = [index for index in (0,50,100,200)
                 if index > completed and checkpoint_path(output,index).is_file()]
    return checkpoint_path(output,max(available)) if available else requested


def profile_source_method(training_report, config, model):
    """Bind a diagnostic checkpoint to the completed run without resuming it."""
    if (training_report.get('status') != 'completed'
            or training_report.get('completed_updates') != 200
            or training_report.get('configuration') != config
            or training_report.get('model_protocol') != model
            or training_report.get('scope') != 'bounded_200_update_learning'):
        raise ValueError('profile requires the completed declared 200-update run')
    return dict(implementation='mixed_learning_v1',configuration=config,
                model_protocol=model,scope='bounded_200_update_learning',
                integration_updates=None,source_git_head=training_report['source_git_head'])


def lowered_precision_summary(lowered):
    """Static matrix result types; counts are not execution time or FLOP shares."""
    import re
    from collections import Counter

    text = lowered.as_text()
    types = Counter()
    for line in text.splitlines():
        if 'stablehlo.dot_general ' in line or 'stablehlo.convolution ' in line:
            match = re.search(r'->\s*tensor<[^>]*?((?:bf|f)\d+)>',line)
            types[match.group(1) if match else 'unresolved'] += 1
    return dict(matrix_operation_result_types=dict(types),
                interpretation='static_operation_counts_not_runtime_or_FLOP_proportions')


def run(args):
    from research.daily_coarse_graining.daily_operator.canonical_rollout import make_canonical_step
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective, make_update
    from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
    from research.daily_coarse_graining.daily_operator.mixed_training_state import (
        load_mixed_training_checkpoint,
        save_mixed_training_checkpoint,
    )
    from research.daily_coarse_graining.daily_operator.one_day_training_state import make_gradient_accumulator
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from research.daily_coarse_graining.daily_operator.recurrent_science_loss import (
        WEIGHT_COMPONENTS,
        bind_recurrent_science_loss,
    )
    from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
    from research.daily_coarse_graining.daily_operator.target_oriented_training import initialize_target_oriented_adamw
    from scripts.hpc.mixed_learning_monitor import build_mixed_rollout_monitor
    from scripts.hpc.mixed_learning_plan import load_plan
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
    from scripts.hpc.probe_public_rollout import physical_checks
    from scripts.hpc.run_fresh_start_one_day_pool import host_peak_bytes
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.run_trained_forward_screen import feedback_diagnostics, feedback_hard_counts
    from scripts.hpc.trained_screen_common import PreparedWindowPool, WindowPool, json_safe

    if args.admission_only or args.cpu_forward_only:
        raise ValueError('use shape-only or integration-updates for bounded checks')
    profiling = args.profile_checkpoint is not None
    if profiling != (args.profile_training_report is not None):
        raise ValueError('profile checkpoint and source training report are required together')
    if profiling and (args.resume or args.integration_updates or args.shape_only):
        raise ValueError('profiling is a separate diagnostic, not resume or integration training')
    if args.profile_trace and not profiling:
        raise ValueError('trace is available only in diagnostic profiling mode')
    if args.compare_checked_recurrent and not profiling:
        raise ValueError('checked-recurrent admission requires fixed-checkpoint profiling')
    if args.checked_recurrent and profiling:
        raise ValueError('use compare-checked-recurrent for fixed-checkpoint admission')
    if args.output.exists() and not args.resume:
        raise FileExistsError('fresh run requires a new output; use --resume for recovery')
    if args.resume and not args.output.is_file():
        raise FileNotFoundError('resume requires the original report alongside its checkpoint')
    config = json.loads(args.mixed_protocol.read_text())
    model = json.loads(args.protocol.read_text())
    validate_configuration(config, model)
    if (args.mixed_protocol.parent/config['model_protocol']).resolve() != args.protocol.resolve():
        raise ValueError('CLI and mixed model protocols differ')
    limit = args.integration_updates or config['updates']
    scope = ('performance_profile_not_training' if profiling else
             'integration_only' if args.integration_updates else 'bounded_200_update_learning')
    started = time.perf_counter()
    current_head = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report = dict(schema_version='mixed_learning_result_v1', status='running',
        source_git_head=current_head,
        configuration=config, model_protocol=model, scope=scope, completed_updates=0,
        updates=[], monitors=[], compilation={}, checkpoints=[],
        scientific_acceptance=False, checkpoint_inherited=False)
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (previous['configuration'], previous['model_protocol'], previous['scope']) != (config,model,scope):
            raise ValueError('recovery report method mismatch')
        report = previous
        report['prior_elapsed_seconds'] = report.get('prior_elapsed_seconds',0.)+report.get('elapsed_seconds',0.)

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter()-started,
                      host_peak_bytes=host_peak_bytes())
        _atomic_json(args.output, json_safe(report))
        print(f'mixed_learning stage={stage} updates={report["completed_updates"]} '
              f'seconds={report["elapsed_seconds"]:.1f}', flush=True)

    try:
        emit('metadata')
        plan = load_plan(args, config)
        report['data_plan'] = plan['summary']
        identities = plan['identities']
        method = dict(implementation='mixed_learning_v1', configuration=config,
                      model_protocol=model, scope=scope, integration_updates=args.integration_updates,
                      source_git_head=current_head)
        if args.checked_recurrent:
            method['recurrent_execution'] = 'compact_checked_v1'
        fresh_identity = dict(seed=model['seed'],initialization=model['initialization'],
                              parameter_count=model['parameter_count'],inherited_weights=False)
        recovery_identity = dict(expected_method=method,expected_source_identities=identities,
            expected_schedule_identity=identities,expected_fresh_initialization_identity=fresh_identity)
        batches, schedule = plan['batches'], plan['schedule']
        train_pool = PreparedWindowPool(schedule['source_manifest'], cache_size=2)
        development = WindowPool(args.selection)
        daily_pool = PreparedOneDayPool(plan['daily_source'], cache_size=5)
        if (daily_pool.identity != plan['catalogue']['source_identity']
                or train_pool.identity != schedule['source_identity']):
            raise ValueError('readers and metadata sources differ')
        fixed_rows = batches[0].rows
        fixed_windows = [schedule['update_rows'][0][key] for key in ('T7','T30')]
        emit('bootstrap')
        runtime, head, template, scales, parameters, _, _ = prototype.bootstrap(args, model)
        jax = runtime['jax']
        import jax.numpy as jnp
        if sum(p.size for p in jax.tree_util.tree_leaves(parameters)) != model['parameter_count']:
            raise ValueError('actual full-network parameter count differs')
        graph = calibration._loss_protocol(model,
            canopy_source_pft_index=calibration._canopy_source_pft_index(runtime, head))
        objective = make_objective(runtime, head, template, scales, graph)
        base = calibration._base_weights(graph)
        daily_backward = calibration._make_evaluator(jax,jnp,objective,base)
        daily_forward = jax.jit(calibration._make_batch_objective(jax,jnp,objective,base))
        coefficients = jnp.asarray((1., model['canopy_carbon_stock_change']['weight'],
                                    model['loss']['litterpart_feedback_mse']),dtype=jnp.float64)
        recurrent_loss, loss_spec = bind_recurrent_science_loss(runtime,head,config['recurrent_weights'])
        component = make_rollout_loss(jax,jnp,make_canonical_step(head,template),recurrent_loss)
        backward = jax.jit(jax.value_and_grad(component.checkpointed_loss,argnums=0))
        forward = jax.jit(component.rollout)
        terms_fn = jax.jit(recurrent_loss.terms)
        context = {'stock_weight': jnp.asarray(1.,dtype=jnp.float64)}
        update = make_update(model)
        add, finish = make_gradient_accumulator(jax,jnp)
        combine = make_mixed_gradient(jax,jnp)
        # Reduce diagnostics on device instead of copying 1.3 GB state every update.
        finite = jax.jit(lambda tree: jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(tree)])))
        norm_fn = jax.jit(lambda tree: jnp.sqrt(sum(jnp.sum(jnp.square(v.astype(jnp.float64)))
                                                  for v in jax.tree_util.tree_leaves(tree))))
        report.update(recurrent_scales=loss_spec, device=str(runtime['devices'][0]),
            parameter_count=model['parameter_count'], memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'))

        def sample(rows):
            rows = admission._rows(rows)
            return daily_pool.batch([(int(r['artifact_id']),int(r['batch_row'])) for r in rows])

        first_daily = sample(fixed_rows[:8])
        cases = {h: train_pool.case(jax, schedule['update_rows'][0][f'T{h}']) for h in (7,30)}
        if args.shape_only:
            (_, _), dg = jax.eval_shape(daily_backward, parameters, first_daily, coefficients)
            for h,case in cases.items():
                _, rg = jax.eval_shape(backward,parameters,*case[:4],context)
                mixed = jax.eval_shape(combine,dg,rg,1.,1.)
                state = jax.eval_shape(initialize_target_oriented_adamw,parameters)
                jax.eval_shape(update,parameters,mixed,state)
            emit('completed',status='passed',classification='daily_T7_T30_optimizer_shapes')
            return report

        if profiling:
            source_report = json.loads(args.profile_training_report.read_text())
            profile_identity = dict(recovery_identity,
                expected_method=profile_source_method(source_report,config,model))
            saved = load_mixed_training_checkpoint(args.profile_checkpoint,**profile_identity)
            if saved['completed_updates'] != 200:
                raise ValueError('profile checkpoint must be at update200')
            del parameters
            parameters, optimizer = jax.device_put((saved['parameters'],saved['optimizer_state']))
            report.update(checkpoint_inherited=True,profile_source_checkpoint=str(args.profile_checkpoint),
                          profile_source_update=200,persistent_parameter_updates=0)
            del saved
        elif args.resume:
            saved = load_mixed_training_checkpoint(args.resume, **recovery_identity)
            resolved_resume = newest_checkpoint(args.output,args.resume,saved['completed_updates'])
            if resolved_resume != args.resume:
                del saved
                saved = load_mixed_training_checkpoint(resolved_resume, **recovery_identity)
            del parameters
            parameters, optimizer = jax.device_put((saved['parameters'],saved['optimizer_state']))
            trim_resume_report(report,saved['completed_updates'])
            report['resumed_from'] = str(resolved_resume)
            del saved
        else:
            optimizer = initialize_target_oriented_adamw(parameters)
        if report['completed_updates'] > limit:
            raise ValueError('recovery exceeds this execution budget')

        def save(index=None):
            began = time.perf_counter()
            count = report['completed_updates']
            path = checkpoint_path(args.output,index)
            save_mixed_training_checkpoint(path,jax=jax,parameters=parameters,optimizer_state=optimizer,
                completed_updates=count,daily_sampler_cursor={'epoch':0,'batch_offset':count},
                recurrent_schedule_cursor=count,method=method,
                source_identities=identities,schedule_identity=identities,
                fresh_initialization_identity=fresh_identity,replace=index is None)
            record = dict(update=count,path=str(path),bytes=path.stat().st_size,
                          seconds=time.perf_counter()-began)
            report['checkpoints'] = [r for r in report['checkpoints'] if r['path'] != str(path)] + [record]
            emit('checkpoint_saved')

        def daily_panel(rows):
            losses, aux_rows = [], []
            for i in range(0,len(rows),8):
                loss, aux = daily_forward(parameters,sample(rows[i:i+8]),coefficients)
                loss, aux = jax.device_get((loss,aux))
                losses.append(float(loss))
                aux_rows.append(aux)
            aux = admission._concatenate_auxiliary(jax,aux_rows)
            hard = calibration._hard_summary(aux['hard'],batch_size=len(rows))
            result = dict(mean_loss=float(np.mean(losses)),count=len(rows),hard=hard,
                diagnostics=calibration._diagnostics(jax,aux,batch_size=len(rows)),
                mean_terms=np.mean(np.asarray(aux['terms']),axis=0).tolist())
            if not np.all(np.isfinite(losses)) or not hard['all_zero']:
                report['failed_daily_panel'] = result
                raise ValueError('daily monitor physical or finite gate failed')
            return result

        monitor = build_mixed_rollout_monitor(runtime,head,template,scales,model,
                                              compiled_memory_limit_bytes=18*1024**3)

        def monitor_milestone(index):
            if any(r['update'] == index for r in report['monitors']):
                return
            emit('monitoring',monitor_update=index)
            began = time.perf_counter()
            record = dict(update=index, fixed_training_daily=daily_panel(fixed_rows),
                          excluded_train_daily=daily_panel(plan['monitor_rows']))
            for name,pool,rows,gates in (
                    ('fixed_training_recurrent',train_pool,fixed_windows,False),
                    ('development',development,development.selection['windows'],True)):
                emit('monitor_'+name,monitor_update=index)
                result = monitor(parameters,pool,rows,apply_science_gates=gates,
                                 development=gates,panel_name=name)
                record[name] = result
                if not result['physical_gate_passed'] or not result['metric_population_complete']:
                    report['failed_monitor'] = record
                    raise ValueError(name+' physical/metric population gate failed')
            record['seconds'] = time.perf_counter()-began
            report['monitors'].append(record)
            emit('monitor_completed',memory=_device_memory(runtime['devices'][0]))

        if profiling:
            emit('profile_ready')
        elif report['completed_updates'] == 0:
            if not args.resume:
                save(0)
            monitor_milestone(0)
        elif report['completed_updates'] in config['milestones']:
            monitor_milestone(report['completed_updates'])

        checked_lowerings = {}
        if args.compare_checked_recurrent or args.checked_recurrent:
            from scripts.hpc.checked_recurrent_execution import bind_checked_backward

            checked_backward, checked_names = bind_checked_backward(
                jax,jnp,component,recurrent_loss,runtime,head,scales,model)
            for horizon, case in cases.items():
                emit(f'checked_lower_T{horizon}')
                checked_lowerings[horizon] = checked_backward.lower(parameters,*case[:4],context,case[4])

        executables = {}
        for name,fn,arguments in [('T7',backward,(parameters,*cases[7][:4],context)),
                                  ('T30',backward,(parameters,*cases[30][:4],context)),
                                  ('daily',daily_backward,(parameters,first_daily,coefficients))]:
            emit('compile_'+name)
            began = time.perf_counter()
            lowered = (checked_lowerings.pop(int(name[1:]))
                       if args.checked_recurrent and name != 'daily' else fn.lower(*arguments))
            executable = lowered.compile()
            memory = _compiled_memory(executable)
            report['compilation'][name] = dict(seconds=time.perf_counter()-began,memory=memory)
            if profiling:
                report['compilation'][name]['precision'] = lowered_precision_summary(lowered)
            if memory['estimated_live_bytes'] > config['compiled_memory_limit_gib']*1024**3:
                raise MemoryError(name+' compiled live memory exceeds bound')
            executables[name] = executable
            emit('compiled_'+name)
        del cases, first_daily, arguments, lowered

        def validate_recurrent(case, obs):
            initial,discrete,days,targets,expected_discrete = case
            hard = physical_checks(obs,targets,expected_discrete,days[0],head.contract,scales)
            terms = terms_fn(parameters,initial,obs,targets,context)
            for section in WEIGHT_COMPONENTS:
                for key,term in getattr(terms,section).items():
                    hard[section+'.'+key+'.invalid'] = int(term.invalid_prediction_count+term.invalid_target_count)
                    hard[section+'.'+key+'.empty'] = int(term.count == 0)
            first = jax.tree_util.tree_map(lambda v:v[0],days[0])
            assembled = head.input_assembler.assemble(first)
            feedback = feedback_diagnostics(jax,jnp,runtime,head,model,initial,assembled,obs,targets,days[0])
            hard.update(feedback_hard_counts(feedback))
            if any(hard.values()) or not np.isfinite(float(terms.total)):
                report['failed_training_window'] = dict(hard=hard,loss=float(terms.total))
                raise ValueError('rotating TRAIN trajectory physical/finite gate failed')
            return hard

        def valid_recurrent(row):
            case = train_pool.case(jax,row)
            _, obs = forward(parameters,*case[:3])
            jax.block_until_ready(obs)
            try:
                hard = validate_recurrent(case,obs)
            except ValueError:
                if 'failed_training_window' in report:
                    report['failed_training_window']['window'] = row
                raise
            return case,hard

        if profiling:
            from scripts.hpc.mixed_learning_profiling import _parity, profile_components

            profile_arguments = dict(jax=jax,jnp=jnp,parameters=parameters,optimizer=optimizer,
                sample=sample,daily_rows=fixed_rows,
                recurrent_rows={row['length']:row for row in fixed_windows},window_pool=train_pool,
                daily_executable=executables['daily'],
                recurrent_executables={7:executables['T7'],30:executables['T30']},
                coefficients=coefficients,context=context,forward=forward,
                validate_recurrent=validate_recurrent,update=update,add=add,finish=finish,
                combine=combine,finite=finite,norm_fn=norm_fn,repeats=args.profile_repeats,
                trace_dir=args.output.with_suffix('.trace') if args.profile_trace else None,emit=emit)
            profile = profile_components(**profile_arguments)
            emit('original_profile_completed',profile=profile)
            if args.compare_checked_recurrent:
                from research.daily_coarse_graining.daily_operator.one_day_training_state import _atomic_pickle
                from scripts.hpc.checked_recurrent_execution import decode_counts
                from scripts.hpc.probe_public_rollout import gradient_comparison

                checked_executables, comparisons = {}, {}
                for row in fixed_windows:
                    horizon, key = row['length'], f"T{row['length']}"
                    case, old_hard = valid_recurrent(row)
                    value, old_gradient = executables[key](parameters,*case[:4],context)
                    old_value, reference = jax.device_get((value,old_gradient))
                    path = args.output.with_suffix(f'.{key}.reference.pkl')
                    _atomic_pickle(path,dict(value=old_value,gradient=reference,hard=old_hard,
                                             checkpoint=str(args.profile_checkpoint),window=row))
                    del value, old_gradient
                    emit('checked_compile_'+key,reference_gradient=str(path))
                    began = time.perf_counter()
                    executable = checked_lowerings.pop(horizon).compile()
                    memory = _compiled_memory(executable)
                    report['compilation']['checked_'+key] = dict(seconds=time.perf_counter()-began,memory=memory)
                    if memory['estimated_live_bytes'] > config['compiled_memory_limit_gib']*1024**3:
                        raise MemoryError('checked '+key+' compiled live memory exceeds bound')
                    (new_value, counts), gradient = executable(parameters,*case[:4],context,case[4])
                    new_value, counts = jax.device_get((new_value,counts))
                    hard = decode_counts(checked_names,counts)
                    comparison = gradient_comparison(jax,gradient,reference)
                    comparison.update(loss_equal=bool(np.isclose(new_value,old_value,atol=1e-6,rtol=1e-5)),
                        old_loss=float(old_value),new_loss=float(new_value),
                        physical_counts_equal=hard==old_hard,all_hard_zero=not any(hard.values()),
                        reference_gradient=str(path),hard=hard)
                    comparisons[key] = comparison
                    emit('checked_compared_'+key,checked_comparisons=comparisons)
                    if not (comparison['passed'] and comparison['loss_equal']
                            and comparison['physical_counts_equal'] and comparison['all_hard_zero']):
                        raise ValueError('checked '+key+' loss/gradient/physical comparison failed')
                    checked_executables[horizon] = executable
                    del reference,gradient,case
                checked_profile = profile_components(**profile_arguments,
                    checked_recurrent_executables=checked_executables,checked_counter_names=checked_names)
                parity = _parity(profile['baseline']['records'],checked_profile['baseline']['records'])
                if not parity['passed']:
                    raise ValueError('checked complete mixed update differs from original')
                before, after = profile['baseline']['median_seconds'], checked_profile['baseline']['median_seconds']
                emit('checked_admission_completed',checked_profile=checked_profile,
                     checked_mixed_parity=parity,checked_speedup=before/after,
                     checked_time_reduction=1.-after/before)
            emit('completed',status='completed',profile=profile,persistent_parameter_updates=0,
                 memory=_device_memory(runtime['devices'][0]),daily_reader=daily_pool.statistics())
            return report

        for index in range(report['completed_updates'],limit):
            emit('training',next_update=index+1)
            began = time.perf_counter()
            batch = batches[index]
            daily_total = jax.tree_util.tree_map(jnp.zeros_like,parameters)
            daily_loss, hard_totals = 0., {}
            for i in range(0,64,8):
                (value,aux), gradient = executables['daily'](parameters,sample(batch.rows[i:i+8]),coefficients)
                hard = calibration._hard_summary(jax.device_get(aux['hard']),batch_size=8)
                if not np.isfinite(float(value)) or not hard['all_zero']:
                    report['failed_training_daily'] = dict(update=index+1,microbatch=i//8,hard=hard)
                    raise ValueError('daily training physical/finite gate failed')
                for key,count in hard['totals'].items():
                    hard_totals[key] = hard_totals.get(key,0)+count
                daily_total = add(daily_total,gradient,8)
                daily_loss += float(value)/8
            daily_mean = finish(daily_total,64)
            del daily_total,gradient,aux
            component_norms = {'daily':float(norm_fn(daily_mean))}
            mixed = daily_mean
            del daily_mean
            losses, recurrent_hard = {}, {}
            for h in (7,30):
                key = f'T{h}'
                row = schedule['update_rows'][index][key]
                if args.checked_recurrent:
                    from scripts.hpc.checked_recurrent_execution import decode_counts

                    case = train_pool.case(jax,row)
                    (value,counts),gradient = executables[key](parameters,*case[:4],context,case[4])
                    value,counts = jax.device_get((value,counts))
                    hard = decode_counts(checked_names,counts)
                    if any(hard.values()) or not np.isfinite(value):
                        report['failed_training_window'] = dict(window=row,hard=hard,loss=float(value))
                        raise ValueError('checked TRAIN trajectory physical/finite gate failed')
                else:
                    case,hard = valid_recurrent(row)
                    value,gradient = executables[key](parameters,*case[:4],context)
                losses[key], recurrent_hard[key] = float(value),hard
                component_norms[key] = float(norm_fn(gradient))
                mixed = combine(mixed,gradient,1.,.5)
                jax.block_until_ready(mixed)
                del gradient,case
            if (not all(np.isfinite(v) and v>0 for v in component_norms.values())
                    or not np.all(np.isfinite(list(losses.values()))) or not bool(finite(mixed))):
                raise ValueError('component/mixed gradient is nonfinite or empty')
            proposed,state,norm,clip = update(parameters,mixed,optimizer)
            jax.block_until_ready((proposed,state,norm,clip))
            if (not bool(finite((proposed,state))) or not np.isfinite(float(norm))
                    or not 0<float(clip)<=1 or int(state.step)!=index+1):
                raise ValueError('complete AdamW proposal invalid')
            parameters,optimizer = proposed,state
            del mixed,proposed,state
            report['completed_updates'] = index+1
            report['updates'].append(dict(update=index+1,daily_loss=daily_loss,**losses,
                mixed_loss=daily_loss+.5*(losses['T7']+losses['T30']),gradient_norms=component_norms,
                norm=float(norm),clip=float(clip),daily_hard=hard_totals,recurrent_hard=recurrent_hard,
                seconds=time.perf_counter()-began))
            emit('optimizer_completed',memory=_device_memory(runtime['devices'][0]),
                 daily_reader=daily_pool.statistics())
            if index+1 in config['milestones']:
                save(index+1)
                monitor_milestone(index+1)
            elif (index+1)%config['rolling_recovery_interval']==0 or index+1==limit:
                save()
            if args.integration_updates and index+1==limit:
                monitor_milestone(index+1)
        # Replay persisted weights through the actual daily forward entry point.
        final_path = checkpoint_path(args.output,limit if limit in config['milestones'] else None)
        expected = float(daily_forward(parameters,sample(fixed_rows[:8]),coefficients)[0])
        saved = load_mixed_training_checkpoint(final_path,**recovery_identity)
        restored = jax.device_put(saved['parameters'])
        actual = float(daily_forward(restored,sample(fixed_rows[:8]),coefficients)[0])
        if actual != expected:
            raise ValueError('persisted checkpoint forward replay differs')
        del restored,saved
        emit('completed',status='completed',classification=scope,
             checkpoint_forward_replay=True,scientific_acceptance=False,
             memory=_device_memory(runtime['devices'][0]))
        return report
    except Exception as error:
        emit('failed',status='resource_limited' if isinstance(error,MemoryError) else 'failed',
             error=f'{type(error).__name__}: {error}')
        raise


def parser():
    result = prototype.parser()
    for name in ('mixed-protocol','catalogue-report','recurrent-schedule','selection'):
        result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--integration-updates',type=int,choices=(1,2))
    result.add_argument('--resume',type=Path)
    result.add_argument('--profile-checkpoint',type=Path)
    result.add_argument('--profile-training-report',type=Path)
    result.add_argument('--profile-repeats',type=int,choices=(3,5),default=3)
    result.add_argument('--profile-trace',action='store_true')
    result.add_argument('--compare-checked-recurrent',action='store_true')
    result.add_argument('--checked-recurrent',action='store_true')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
