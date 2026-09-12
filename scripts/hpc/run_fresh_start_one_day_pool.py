"""Persistent fresh full-network one-day training on the existing broad pool."""
from __future__ import annotations

import gzip
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_fresh_start_one_day as prototype

ROOT = Path(__file__).resolve().parents[2]


def pool_plan(catalogue, configuration):
    from research.daily_coarse_graining.daily_operator.one_day_sampling import DayKey, OneDayArtifact, OneDaySampler

    artifacts = [OneDayArtifact.from_schedule_entry(e['update'], e) for e in catalogue['one_day_entries']]
    monitor = catalogue['monitor_rows']
    excluded = {DayKey(r['landpoint_id'], r['year'], r['start_row']) for r in monitor}
    sampling = configuration['sampling']
    sampler = OneDaySampler(artifacts, seed=sampling['seed'], batch_size=sampling['batch_size'],
        microbatch_size=sampling['microbatch_size'], max_artifacts_per_batch=sampling['max_artifacts_per_batch'],
        excluded_days=excluded)
    training_panel = [r.as_metadata() for r in next(sampler.iter_epoch(0)).rows]
    return sampler, {'training': training_panel, 'excluded_days': monitor}


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
              'source_git_head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'configuration': configuration, 'model_protocol': protocol,
              'execution_scope': 'technical_smoke' if args.smoke_updates else 'one_prepared_pool_epoch',
              'generalization_claim': False, 'long_horizon_acceptance': False}
    if previous:
        report['monitors'],report['updates'] = previous['monitors'],previous['updates']
        report['completed_updates'] = previous['completed_updates']

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter()-started)
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
        method = {'configuration': configuration, 'model_protocol': protocol,
                  'source_identity': catalogue['source_identity'], 'monitor_rows': catalogue['monitor_rows'],
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
        objective = make_objective(runtime, head, template, scales, protocol)
        _, backward = gradient_evaluators(jax,jnp,objective)
        forward = jax.jit(objective)
        update = make_update(protocol)
        add_gradient, mean_gradient = make_gradient_accumulator(jax,jnp)
        first = next(sampler.iter_epoch(0)).microbatches[0]
        samples = pool.batch([(r.artifact_id,r.batch_row) for r in first.rows])
        sample = jax.tree_util.tree_map(lambda v:v[0],samples)
        state = initialize_target_oriented_adamw(parameters)
        if args.shape_only:
            emit('trace_gradient_and_optimizer')
            (_, _), gradients = jax.eval_shape(backward,parameters,samples)
            jax.eval_shape(update,parameters,gradients,state)
            jax.eval_shape(forward,parameters,sample)
            emit('completed',status='passed',classification='pool_gradient_optimizer_monitor_shapes_only')
            return report

        cursor = {'epoch':0,'batch_offset':0}
        if args.resume:
            saved = load_training_checkpoint(args.resume,expected_method=method)
            parameters, state = jax.device_put((saved['parameters'],saved['optimizer_state']))
            cursor = saved['cursor']
            report['completed_updates'] = saved['completed_updates']
            if previous and previous.get('method') == method:
                report['monitors'] = [r for r in previous['monitors'] if r['update']<=saved['completed_updates']]
                report['updates'] = [r for r in previous['updates'] if r['update']<=saved['completed_updates']]
            elif saved['completed_updates']:
                raise ValueError('resume needs its original report to preserve initial fixed-panel evidence')
            report['resumed_from'] = str(args.resume)

        emit('compile_gradient')
        compile_started = time.perf_counter()
        compiled = backward.lower(parameters,samples).compile()
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
                record['panels'][name] = fixed_panel_record(parameters,rows,pool,forward,jax)
            report['monitors'].append(record)
            emit('monitor_completed')

        if not report['monitors']:
            monitor(0)
        checkpoint_path = args.output.with_suffix('.checkpoint.pkl')

        def checkpoint():
            save_training_checkpoint(checkpoint_path,jax=jax,parameters=parameters,optimizer_state=state,
                completed_updates=report['completed_updates'],cursor=cursor,method=method)
            report['checkpoint'] = str(checkpoint_path)

        if not args.resume:
            checkpoint()

        # This compiled check is shared by gradient and optimizer proposals.
        finite = jax.jit(lambda tree: jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(tree)])))
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
                    (loss,aux),gradient = evaluator(parameters,data)
                    if not np.isfinite(float(loss)) or not hard_pass(aux['hard']):
                        emit('stopped',status='rejected',reason='training_forward',failing_rows=[r.as_metadata() for r in microbatch.rows])
                        return report
                    total = add_gradient(total,gradient,microbatch.draw_count)
                    total_loss += float(loss)*microbatch.draw_count
                gradient = mean_gradient(total,batch.draw_count)
                if not bool(finite(gradient)):
                    emit('stopped',status='rejected',reason='nonfinite_gradient')
                    return report
                proposed, proposed_state, norm, clipped = update(parameters,gradient,state)
                if not np.isfinite(float(norm)) or not bool(finite((proposed,proposed_state))):
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
                    'seconds':time.perf_counter()-began,'row_draws':batch.draw_count})
                report['consumed_row_draws'],report['consumed_unique_teacher_days'] = observed_rows,len(observed_days)
                if count in configuration['monitoring']['updates']:
                    monitor(count)
                if count % configuration['checkpoint']['every_updates'] == 0 or count in configuration['monitoring']['updates']:
                    checkpoint()
                if count % configuration['report_every_updates'] == 0 or count == 1:
                    emit('update_completed',cursor=cursor,reader=pool.statistics(),memory=_device_memory(runtime['devices'][0]))
            if args.smoke_updates and report['completed_updates'] >= args.smoke_updates:
                break
        if report['monitors'][-1]['update'] != report['completed_updates']:
            monitor(report['completed_updates'])
        checkpoint()
        initial,last = report['monitors'][0]['panels'],report['monitors'][-1]['panels']
        learning = {name:{metric:bool(last[name][metric]<initial[name][metric])
                         for metric in ('mean_loss','mean_flux_loss')} for name in panels}
        emit('completed',status='completed',learning_screen=learning,
             learning_screen_passed=all(all(v.values()) for v in learning.values()),
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
