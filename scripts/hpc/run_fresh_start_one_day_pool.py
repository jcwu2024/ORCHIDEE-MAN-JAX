"""Persistent fresh full-network one-day training on the existing broad pool."""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as feedback_calibration
from scripts.hpc import run_fresh_start_one_day as prototype

ROOT = Path(__file__).resolve().parents[2]
METHOD_IMPLEMENTATION_VERSION = 'fresh_one_day_pool_v1'


def host_peak_bytes():
    try:
        import resource
    except ImportError:
        return None
    # The execution platform is Linux, where ru_maxrss is expressed in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def pool_plan(catalogue, configuration):
    from research.daily_coarse_graining.daily_operator.one_day_sampling import DayKey, OneDayArtifact, OneDaySampler

    artifacts = [OneDayArtifact.from_schedule_entry(e['update'], e) for e in catalogue['one_day_entries']]
    monitor = catalogue['monitor_rows']
    excluded = {DayKey(r['landpoint_id'], r['year'], r['start_row']) for r in monitor}
    sampling = configuration['sampling']
    sampler = OneDaySampler(artifacts, seed=sampling['seed'], batch_size=sampling['batch_size'],
        microbatch_size=sampling['microbatch_size'], max_artifacts_per_batch=sampling['max_artifacts_per_batch'],
        excluded_days=excluded, batch_packing=sampling.get('batch_packing', 'artifact_grouped'))
    training_panel = [r.as_metadata() for r in next(sampler.iter_epoch(0)).rows]
    return sampler, {'training': training_panel, 'excluded_days': monitor}


def sampling_identity(catalogue):
    """Bind a recovery cursor to row meanings using already-loaded metadata only."""
    rows = [(entry['update'],[(r['landpoint_id'],r['year'],start)
             for r in entry['components']['one_day'] for start in r['start_rows']])
            for entry in sorted(catalogue['one_day_entries'],key=lambda e:e['update'])]
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def validate_recovery(saved, sampler, epochs):
    cursor = saved['cursor']
    if not 0 <= cursor['epoch'] < epochs:
        raise ValueError('checkpoint epoch is outside the declared run')
    counts = [sampler.batch_count(e) for e in range(cursor['epoch']+1)]
    if not 0 <= cursor['batch_offset'] <= counts[-1]:
        raise ValueError('checkpoint cursor is outside its epoch')
    expected = sum(counts[:-1])+cursor['batch_offset']
    if saved['completed_updates'] != expected or int(saved['optimizer_state'].step) != expected:
        raise ValueError('checkpoint update count, sampler cursor and Adam step disagree')


def hard_pass(hard):
    return all(np.all(np.asarray(value) == 0) for value in hard.values())


def fixed_panel_record(parameters, rows, pool, forward, jax):
    from research.daily_coarse_graining.daily_operator.one_day_monitoring import summarize_monitor

    physical, identities, losses, flux_losses, state_losses, process_losses = [], [], [], [], [], []
    for row in rows:
        batch = pool.batch([(row['artifact_id'], row['batch_row'])])
        sample = jax.tree_util.tree_map(lambda v: v[0], batch)
        loss, aux = jax.device_get(forward(parameters, sample))
        if not np.isfinite(loss) or not hard_pass(aux['hard']):
            raise ValueError(f"monitor physical/nonfinite failure at {row}: {aux['hard']}")
        identities.append({**row, 'prepared_day_index': int(np.asarray(sample.inputs.day_index))})
        physical.append(aux['physical'])
        losses.append(float(loss))
        flux_losses.append(float(aux['flux_loss']))
        state_losses.append(float(aux['state_loss']))
        process_losses.append(float(aux['process_loss']))
    return {'rows': identities, 'mean_loss': float(np.mean(losses)),
            'mean_flux_loss': float(np.mean(flux_losses)), 'mean_state_loss': float(np.mean(state_losses)),
            'mean_process_loss': float(np.mean(process_losses)),
            'metrics': summarize_monitor(physical), 'physical': physical}


def feedback_loss_plan(protocol):
    """Resolve the optional admitted dynamic feedback-loss coefficients."""
    canopy = protocol.get('canopy_carbon_stock_change')
    loss = protocol.get('loss', {})
    has_litterpart = 'litterpart_feedback_mse' in loss
    if canopy is None and not has_litterpart:
        return None
    if canopy is None or not has_litterpart:
        raise ValueError('canopy and litterpart feedback losses must be enabled together')
    if (not isinstance(canopy, dict) or not {'weight', 'scale_source_pft_index'} <= set(canopy)
            or set(canopy) - {'weight', 'scale_source_pft_index', 'penalty'}):
        raise ValueError('canopy feedback-loss protocol keys drift')
    canopy_weight = float(canopy['weight'])
    if canopy.get('penalty', 'mse') not in ('mse', 'pseudo_huber'):
        raise ValueError('unknown canopy feedback penalty')
    litterpart_weight = float(loss['litterpart_feedback_mse'])
    source_pft_index = int(canopy['scale_source_pft_index'])
    if source_pft_index != 13:
        raise ValueError('feedback-loss admission is limited to the PFT14 source index')
    if not all(np.isfinite(value) and value >= 0. for value in (canopy_weight, litterpart_weight)):
        raise ValueError('feedback-loss coefficients must be finite and nonnegative')
    coefficients = {'base': 1., 'canopy_stock_change': canopy_weight,
                    'litterpart_feedback': litterpart_weight}
    return {'coefficients': coefficients, 'scale_source_pft_index': source_pft_index,
            'gradient_graph': 'feedback_loss_gradient_calibration_v1_dynamic_batch_objective'}


def retained_checkpoint_updates(configuration):
    values = configuration.get('checkpoint', {}).get('retained_updates', [])
    if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError('checkpoint retained_updates must be a list of integers')
    if any(value <= 0 for value in values) or values != sorted(set(values)):
        raise ValueError('checkpoint retained_updates must be unique increasing positive updates')
    return tuple(values)


def retained_checkpoint_path(output, update):
    return output.with_name(f'{output.stem}.update-{update:06d}.checkpoint.pkl')


def host_tree_finite(jax, tree):
    leaves = jax.tree_util.tree_leaves(jax.device_get(tree))
    return bool(leaves) and all(np.all(np.isfinite(np.asarray(value))) for value in leaves)


def feedback_fixed_panel_record(parameters, rows, pool, forward, coefficients, jax):
    """Monitor the admitted objective in batch fours and reuse its diagnostics."""
    auxiliaries, identities, weighted_loss = [], [], 0.
    for start in range(0, len(rows), 4):
        selected = rows[start:start+4]
        batch = pool.batch([(row['artifact_id'], row['batch_row']) for row in selected])
        loss, auxiliary = forward(parameters, batch, coefficients)
        loss, auxiliary, day_indices = jax.device_get((loss, auxiliary, batch.inputs.day_index))
        if not np.isfinite(loss) or not hard_pass(auxiliary['hard']):
            raise ValueError(f'monitor physical/nonfinite failure at {selected}: {auxiliary["hard"]}')
        weighted_loss += float(loss)*len(selected)
        auxiliaries.append(auxiliary)
        identities.extend({**row, 'prepared_day_index': int(day_index)}
                          for row,day_index in zip(selected,np.asarray(day_indices),strict=True))
    combined = jax.tree_util.tree_map(lambda *values: np.concatenate(values, axis=0), *auxiliaries)
    diagnostics = feedback_calibration._diagnostics(jax, combined, batch_size=len(rows))
    terms = np.asarray(combined['terms'], dtype=np.float64)
    hard = feedback_calibration._hard_summary(combined['hard'], batch_size=len(rows))
    physical = [jax.tree_util.tree_map(lambda value,index=index: np.asarray(value)[index],
                combined['physical']) for index in range(len(rows))]
    return {'rows': identities, 'mean_loss': weighted_loss/len(rows),
            'mean_flux_loss': float(np.mean(np.asarray(combined['flux_loss'],dtype=np.float64))),
            'mean_state_loss': float(np.mean(np.asarray(combined['state_loss'],dtype=np.float64))),
            'mean_process_loss': float(np.mean(np.asarray(combined['process_loss'],dtype=np.float64))),
            'mean_base_loss': float(np.mean(terms[:,0])),
            'mean_canopy_stock_change_loss': float(np.mean(terms[:,1])),
            'mean_litterpart_feedback_loss': float(np.mean(terms[:,2])),
            'metrics': diagnostics['physical'], 'feedback_diagnostics': {
                'canopy_stock_change': diagnostics['canopy_stock_change'],
                'litterpart_feedback': diagnostics['litterpart_feedback']},
            'hard_counts': hard, 'physical': physical}


def run(args):
    from scripts.hpc.probe_one_day_batch import (
        _compiled_memory,
        _device_memory,
        _resolved_reference,
        gradient_evaluators,
    )
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    configuration = json.loads(args.pool_protocol.read_text())
    protocol = json.loads((ROOT/configuration['model_protocol']).read_text())
    feedback = feedback_loss_plan(protocol)
    retained_updates = retained_checkpoint_updates(configuration)
    if args.admission_only or args.cpu_forward_only:
        raise ValueError('use --shape-only or --smoke-updates 2 for this runner')
    if args.smoke_updates is not None and args.resume:
        raise ValueError('technical smoke starts fresh; it is not a checkpoint continuation')
    if args.output.exists() and not args.resume:
        raise FileExistsError('choose a new output path for a fresh run, or explicitly resume')
    previous = json.loads(args.output.read_text()) if args.resume and args.output.exists() else None
    if previous and (previous.get('configuration') != configuration or previous.get('model_protocol') != protocol):
        raise ValueError('resume report describes a different method')
    started = time.perf_counter()
    report = {'status': 'running', 'completed_updates': 0, 'monitors': [], 'updates': [],
              'retained_checkpoints': [],
              'source_git_head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'configuration': configuration, 'model_protocol': protocol,
              'feedback_loss': feedback,
              'execution_scope': 'technical_smoke' if args.smoke_updates else 'one_prepared_pool_epoch',
              'generalization_claim': False, 'long_horizon_acceptance': False}
    if previous:
        report['monitors'],report['updates'] = previous['monitors'],previous['updates']
        report['retained_checkpoints'] = previous.get('retained_checkpoints', [])
        report['completed_updates'] = previous['completed_updates']

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter()-started,
                      host_peak_bytes=host_peak_bytes())
        _atomic_json(args.output, report)
        print(f'one_day_pool stage={stage} updates={report["completed_updates"]} seconds={report["elapsed_seconds"]:.1f}',flush=True)

    try:
        emit('catalogue')
        catalogue_report = json.loads(args.catalogue_report.read_text())
        if catalogue_report['status'] != 'passed':
            raise ValueError('one-day catalogue is not accepted')
        catalogue_path = _resolved_reference(catalogue_report['catalogue'], relative_to=args.catalogue_report.resolve())
        with gzip.open(catalogue_path, 'rt', encoding='utf-8') as handle:
            catalogue = json.load(handle)
        sampler, panels = pool_plan(catalogue, configuration)
        total_updates = sum(sampler.batch_count(epoch) for epoch in range(configuration['sampling']['epochs']))
        if any(update > total_updates for update in retained_updates):
            raise ValueError('retained checkpoint update exceeds the declared training run')
        method = {'configuration': configuration, 'model_protocol': protocol,
                  'feedback_loss': feedback,
                  'source_identity': catalogue['source_identity'], 'monitor_rows': catalogue['monitor_rows'],
                  'sampling_rows_identity':sampling_identity(catalogue),
                  'implementation_version':METHOD_IMPLEMENTATION_VERSION,
                  'execution_scope': report['execution_scope']}
        report.update(method=method, pool=sampler.pool_metadata())
        emit('bootstrap')
        runtime, head, template, scales, parameters, _, _ = prototype.bootstrap(args, protocol)
        jax = runtime['jax']
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective, make_update
        from research.daily_coarse_graining.daily_operator.one_day_training_state import (
            load_training_checkpoint,
            make_gradient_accumulator,
            save_training_checkpoint,
        )
        from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
        from research.daily_coarse_graining.daily_operator.target_oriented_training import (
            initialize_target_oriented_adamw,
        )

        if sum(v.size for v in jax.tree_util.tree_leaves(parameters)) != protocol['parameter_count']:
            raise ValueError('fresh neural parameter count differs')
        source = _resolved_reference(catalogue['source_manifest'], relative_to=catalogue_path)
        pool = PreparedOneDayPool(source, cache_size=configuration['sampling']['max_artifacts_per_batch'])
        if pool.identity != catalogue['source_identity']:
            raise ValueError('catalogue and prepared data source differ')
        if feedback is None:
            objective = make_objective(runtime, head, template, scales, protocol)
            _, backward = gradient_evaluators(jax,jnp,objective)
            forward = jax.jit(objective)
            coefficients = None
        else:
            graph_protocol = feedback_calibration._loss_protocol(
                protocol, canopy_source_pft_index=feedback['scale_source_pft_index'])
            objective = make_objective(runtime, head, template, scales, graph_protocol)
            backward = feedback_calibration._make_evaluator(
                jax, jnp, objective, feedback_calibration._base_weights(graph_protocol))
            forward = jax.jit(feedback_calibration._make_batch_objective(
                jax, jnp, objective, feedback_calibration._base_weights(graph_protocol),
                include_base_components=True))
            coefficients = jnp.asarray([feedback['coefficients'][name]
                                        for name in feedback_calibration.LOSS_NAMES],dtype=jnp.float64)
            report['feedback_loss'].update(
                graph_protocol=graph_protocol,
                state_field_layout=[
                    {'key':leaf.key, 'axis_names':list(leaf.axis_names), 'shape':list(leaf.shape),
                     'selected_pft_indices':list(leaf.selected_pft_indices)}
                    for leaf in head.contract.state_leaves
                    if leaf.key in (feedback_calibration.CANOPY_STOCK_KEY,
                                    'slowproc_stomate_previous_step_state.litterpart')])
        update = make_update(protocol)
        add_gradient, mean_gradient = make_gradient_accumulator(jax,jnp)
        first = next(sampler.iter_epoch(0)).microbatches[0]
        samples = pool.batch([(r.artifact_id,r.batch_row) for r in first.rows])
        sample = jax.tree_util.tree_map(lambda v:v[0],samples)
        if args.shape_only:
            emit('trace_gradient_and_optimizer')
            if feedback is None:
                (_, _), gradients = jax.eval_shape(backward,parameters,samples)
                jax.eval_shape(forward,parameters,sample)
            else:
                (_, _), gradients = jax.eval_shape(backward,parameters,samples,coefficients)
                jax.eval_shape(forward,parameters,samples,coefficients)
            state_shape = jax.eval_shape(initialize_target_oriented_adamw, parameters)
            jax.eval_shape(update,parameters,gradients,state_shape)
            emit('completed',status='passed',classification='pool_gradient_optimizer_monitor_shapes_only')
            return report

        state = initialize_target_oriented_adamw(parameters)
        cursor = {'epoch':0,'batch_offset':0}
        if args.resume:
            saved = load_training_checkpoint(args.resume,expected_method=method)
            validate_recovery(saved,sampler,configuration['sampling']['epochs'])
            parameters, state = jax.device_put((saved['parameters'],saved['optimizer_state']))
            cursor = saved['cursor']
            report['completed_updates'] = saved['completed_updates']
            if previous and previous.get('method') == method:
                report['monitors'] = [r for r in previous['monitors'] if r['update']<=saved['completed_updates']]
                report['updates'] = [r for r in previous['updates'] if r['update']<=saved['completed_updates']]
            elif saved['completed_updates']:
                raise ValueError('resume needs its original report to preserve initial fixed-panel evidence')
            report['resumed_from'] = str(args.resume)
        if feedback is not None:
            emit('feedback_initial_finiteness_started')
            initial_parameters_finite = host_tree_finite(jax, parameters)
            initial_optimizer_finite = host_tree_finite(jax, state)
            report['feedback_initial_finiteness'] = {
                'parameters':initial_parameters_finite, 'optimizer':initial_optimizer_finite}
            emit('feedback_initial_finiteness_completed')
            if not initial_parameters_finite or not initial_optimizer_finite:
                raise ValueError('fresh or resumed feedback parameters or optimizer state are nonfinite')

        emit('compile_gradient')
        compile_started = time.perf_counter()
        if feedback is None:
            compiled = backward.lower(parameters,samples).compile()
        else:
            compiled = backward.lower(parameters,samples,coefficients).compile()
        memory = _compiled_memory(compiled)
        emit('gradient_compiled',compile_seconds=time.perf_counter()-compile_started,compiled_memory=memory)
        if memory['estimated_live_bytes'] > configuration['admission']['compiled_memory_limit_gib']*1024**3:
            emit('stopped',status='resource_limited',reason='compiled_gradient_memory')
            return report

        def monitor(iteration):
            emit('monitor_started',monitor_update=iteration)
            record = {'update':iteration, 'panels':{}}
            # Use the same forward definition for reports and the training loss.
            for name,rows in panels.items():
                if feedback is None:
                    record['panels'][name] = fixed_panel_record(parameters,rows,pool,forward,jax)
                else:
                    record['panels'][name] = feedback_fixed_panel_record(
                        parameters,rows,pool,forward,coefficients,jax)
            report['monitors'].append(record)
            emit('monitor_completed')

        if not report['monitors']:
            monitor(0)
        checkpoint_path = args.output.with_suffix('.checkpoint.pkl')

        def checkpoint():
            # Publish audit history first. A crash can then leave it ahead of
            # the atomic checkpoint; resume truncates it to the saved step.
            emit('checkpoint_preparing',checkpoint=str(checkpoint_path),cursor=cursor)
            save_training_checkpoint(checkpoint_path,jax=jax,parameters=parameters,optimizer_state=state,
                completed_updates=report['completed_updates'],cursor=cursor,method=method)
            report['checkpoint'] = str(checkpoint_path)

        def retain_checkpoint(update):
            path = retained_checkpoint_path(args.output, update)
            existing = next((record for record in report['retained_checkpoints']
                             if record['update'] == update), None)
            if existing is not None:
                return
            if path.exists():
                emit('retained_checkpoint_recovering',retained_checkpoint_update=update,
                     retained_checkpoint=str(path),cursor=cursor)
            else:
                if report['completed_updates'] != update:
                    raise ValueError('cannot reconstruct an older missing retained checkpoint')
                emit('retained_checkpoint_preparing',retained_checkpoint_update=update,
                     retained_checkpoint=str(path),cursor=cursor)
                save_training_checkpoint(path,jax=jax,parameters=parameters,optimizer_state=state,
                    completed_updates=report['completed_updates'],cursor=cursor,method=method)
            restored = load_training_checkpoint(path,expected_method=method)
            validate_recovery(restored,sampler,configuration['sampling']['epochs'])
            if restored['completed_updates'] != update:
                raise ValueError('retained checkpoint roundtrip update differs')
            record = {'update':update,'path':str(path),'cursor':restored['cursor'],
                      'optimizer_step':int(restored['optimizer_state'].step),
                      'roundtrip_validated':True}
            report['retained_checkpoints'].append(record)
            emit('retained_checkpoint_completed',retained_checkpoint_update=None,
                 retained_checkpoint=None)

        if not args.resume:
            checkpoint()
        else:
            for checkpoint_update in retained_updates:
                if checkpoint_update <= report['completed_updates']:
                    retain_checkpoint(checkpoint_update)

        # This compiled check is shared by gradient and optimizer proposals.
        finite = (jax.jit(lambda tree: jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(tree)])))
            if feedback is None else None)
        observed_days = set()
        observed_rows = 0
        for epoch in range(configuration['sampling']['epochs']):
            for batch in sampler.iter_epoch(epoch):
                if epoch < cursor['epoch'] or (epoch == cursor['epoch'] and batch.batch_index < cursor['batch_offset']):
                    observed_days.update(r.day_key for r in batch.rows)
                    observed_rows += batch.draw_count
                    continue
                if args.smoke_updates and report['completed_updates'] >= args.smoke_updates:
                    break
                began = time.perf_counter()
                report['attempted_batch'] = batch.as_metadata()
                total = jax.tree_util.tree_map(jnp.zeros_like,parameters)
                total_loss = 0.
                for microbatch in batch.microbatches:
                    data = pool.batch([(r.artifact_id,r.batch_row) for r in microbatch.rows])
                    # Preserve any remainder: only that smaller shape compiles separately.
                    evaluator = compiled if microbatch.draw_count == first.draw_count else backward
                    if feedback is None:
                        (loss,aux),gradient = evaluator(parameters,data)
                    else:
                        (loss,aux),gradient = evaluator(parameters,data,coefficients)
                    if not np.isfinite(float(loss)) or not hard_pass(aux['hard']):
                        emit('stopped',status='rejected',reason='training_forward',failing_rows=[r.as_metadata() for r in microbatch.rows])
                        return report
                    total = add_gradient(total,gradient,microbatch.draw_count)
                    total_loss += float(loss)*microbatch.draw_count
                gradient = mean_gradient(total,batch.draw_count)
                gradient_finite = (bool(finite(gradient)) if feedback is None
                                   else host_tree_finite(jax, gradient))
                if not gradient_finite:
                    emit('stopped',status='rejected',reason='nonfinite_gradient')
                    return report
                proposed, proposed_state, norm, clipped = update(parameters,gradient,state)
                proposal_finite = (bool(finite((proposed,proposed_state))) if feedback is None
                                   else host_tree_finite(jax, (proposed,proposed_state)))
                clip_scale_valid = (np.isfinite(float(clipped)) and 0. < float(clipped) <= 1.)
                if (not np.isfinite(float(norm)) or not proposal_finite or
                        (feedback is not None and not clip_scale_valid)):
                    emit('stopped',status='rejected',reason='nonfinite_optimizer_proposal')
                    return report
                parameters,state = proposed,proposed_state
                report['completed_updates'] += 1
                cursor = {'epoch':epoch,'batch_offset':batch.batch_index+1}
                observed_days.update(r.day_key for r in batch.rows)
                observed_rows += batch.draw_count
                count = report['completed_updates']
                report['updates'].append({'update':count,'epoch':epoch,'batch_index':batch.batch_index,
                    'mean_loss':total_loss/batch.draw_count,'gradient_norm':float(norm),'clip_scale':float(clipped),
                    'gradient_all_finite':gradient_finite,
                    'parameters_optimizer_all_finite':proposal_finite,
                    'clip_scale_valid':clip_scale_valid,
                    'seconds':time.perf_counter()-began,'row_draws':batch.draw_count})
                report['consumed_row_draws'],report['consumed_unique_teacher_days'] = observed_rows,len(observed_days)
                if count in configuration['monitoring']['updates']:
                    monitor(count)
                if count % configuration['checkpoint']['every_updates'] == 0 or count in configuration['monitoring']['updates']:
                    checkpoint()
                if count in retained_updates:
                    retain_checkpoint(count)
                if count % configuration['report_every_updates'] == 0 or count == 1:
                    emit('update_completed',cursor=cursor,reader=pool.statistics(),memory=_device_memory(runtime['devices'][0]))
            if args.smoke_updates and report['completed_updates'] >= args.smoke_updates:
                break
        if report['monitors'][-1]['update'] != report['completed_updates']:
            monitor(report['completed_updates'])
        checkpoint()
        if not args.smoke_updates and {record['update'] for record in report['retained_checkpoints']} != set(retained_updates):
            raise ValueError('declared retained checkpoints are incomplete')
        initial,last = report['monitors'][0]['panels'],report['monitors'][-1]['panels']
        learning = {name:{metric:bool(last[name][metric]<initial[name][metric])
                         for metric in ('mean_loss','mean_flux_loss')} for name in panels}
        emit('completed',status='completed',learning_screen=learning,
             learning_screen_passed=all(all(v.values()) for v in learning.values()),
             all_updates_finite=all(record.get('gradient_all_finite',True)
                                    and record.get('parameters_optimizer_all_finite',True)
                                    and record.get('clip_scale_valid',True)
                                    for record in report['updates']),
             technical_smoke_passed=True if args.smoke_updates else None,
             cursor=cursor,reader=pool.statistics(),memory=_device_memory(runtime['devices'][0]))
    except Exception as error:
        emit('failed',status='failed',error=f'{type(error).__name__}: {error}')
        raise
    return report


def parser():
    result = prototype.parser()
    result.description = __doc__
    result.add_argument('--pool-protocol',type=Path,default=ROOT/'manifests/coarse_graining/fresh_start_one_day_pool_v1.json')
    result.add_argument('--catalogue-report',type=Path,required=True)
    result.add_argument('--resume',type=Path)
    result.add_argument('--smoke-updates',type=int,choices=(2,))
    return result


if __name__ == '__main__':
    run(parser().parse_args())
