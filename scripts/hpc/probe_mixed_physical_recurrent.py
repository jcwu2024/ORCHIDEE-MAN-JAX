"""Bounded fresh full-network daily + native recurrent optimizer admission."""
from __future__ import annotations

import gc
import gzip
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import probe_feedback_loss_updates as admission
from scripts.hpc import run_fresh_start_one_day as prototype


def select_draws(sampler, selection, horizon=7):
    """Pair consecutive fixed32 draws and ordered TRAIN windows without outcome selection."""
    if selection.get('role') != 'train_only_bounded_recurrent_admission_not_validation':
        raise ValueError('mixed training requires the accepted TRAIN-only recurrent selection')
    if horizon not in (7, 30):
        raise ValueError('mixed recurrent horizon must be 7 or 30 days')
    batches = tuple(itertools.islice(sampler.iter_epoch(0), 4))
    if len(batches) != 4 or any(b.batch_index != i or b.draw_count != 32 for i, b in enumerate(batches)):
        raise ValueError('expected first four complete fixed32 epoch0 batches')
    rows = tuple(row for batch in batches for row in batch.rows)
    if len(rows) != 128 or len({(r.artifact_id, r.batch_row) for r in rows}) != 128:
        raise ValueError('mixed admission requires 128 distinct TRAIN examples')
    windows = tuple(row for row in selection['windows'] if row['length'] == horizon)[:4]
    keys = [tuple(key) for row in windows for key in row['teacher_day_keys']]
    if len(windows) != 4 or len(keys) != 4*horizon or len(set(keys)) != 4*horizon:
        raise ValueError(f'expected four distinct {horizon}-day TRAIN windows')
    return (rows[:64], rows[64:]), (windows[:2], windows[2:])


def validate_configuration(config):
    """Validate the bounded T7/T30 runner contract and optional T30 AD check."""
    horizon = config.get('horizon')
    if (config.get('schema_version') != 'mixed_physical_recurrent_admission_v1'
            or config.get('outer_weights') != {'daily': 1., 'recurrent': 1.}
            or (config.get('updates'), config.get('daily_batch'), config.get('daily_microbatch'),
                config.get('windows_per_update')) != (2, 64, 8, 2)
            or horizon not in (7, 30)):
        raise ValueError('runner supports declared two-update 64/micro8/two-window T7/T30 admission')
    directional = config.get('directional_check')
    if directional is not None:
        if (horizon != 30 or set(directional) != {'seeds', 'rtol', 'atol'}
                or tuple(directional['seeds']) != (20260913, 20260914)
                or directional['rtol'] != .01 or directional['atol'] != 1e-6):
            raise ValueError('optional directional check must be the declared two-direction T30 gate')
    return horizon, directional


def directional_record(primal, reverse, forward, reference, *, seed, rtol, atol):
    """Classify one declared native reverse/JVP comparison."""
    primal, reverse, forward, reference = map(float, (primal, reverse, forward, reference))
    resolved = max(abs(reverse), abs(forward)) > 1e-5
    primal_matches = bool(np.isclose(primal, reference, rtol=1e-5, atol=1e-6))
    passed = bool(all(np.isfinite((primal, reverse, forward))) and primal_matches
                  and np.isclose(reverse, forward, rtol=rtol, atol=atol))
    return dict(seed=seed, primal=primal, reverse_projection=reverse, jvp_projection=forward,
                absolute_error=abs(reverse-forward), relative_tolerance=rtol,
                absolute_tolerance=atol, resolved_projection=resolved,
                primal_matches_reverse_evaluation=primal_matches, passed=passed)


def run(args):
    from research.daily_coarse_graining.daily_operator.canonical_rollout import make_canonical_step
    from research.daily_coarse_graining.daily_operator.forward_screen_metrics import physical_window, summarize_windows
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective, make_update
    from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
    from research.daily_coarse_graining.daily_operator.one_day_training_state import make_gradient_accumulator
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from research.daily_coarse_graining.daily_operator.recurrent_science_loss import (
        WEIGHT_COMPONENTS,
        _accepted_science,
        bind_recurrent_science_loss,
        shared_science_trajectories,
    )
    from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
    from research.daily_coarse_graining.daily_operator.target_oriented_training import initialize_target_oriented_adamw
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory, _resolved_reference
    from scripts.hpc.probe_public_rollout import physical_checks
    from scripts.hpc.probe_public_rollout_staged import StageStore
    from scripts.hpc.run_fresh_start_one_day_pool import pool_plan
    from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.run_trained_forward_screen import feedback_diagnostics, feedback_hard_counts
    from scripts.hpc.trained_screen_common import WindowPool, json_safe

    started = time.perf_counter()
    report = dict(schema_version='mixed_physical_recurrent_admission_result_v1', status='running',
                  source_git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  completed_updates=0, updates=[], panels=[], compilation={},
                  checkpoint_inherited=False, checkpoint_written=False, scientific_acceptance=False)

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **values)
        _atomic_json(args.output, json_safe(report))
        print(f'mixed stage={stage} updates={report["completed_updates"]} '
              f'seconds={report["elapsed_seconds"]:.1f}', flush=True)

    try:
        if args.admission_only or args.cpu_forward_only:
            raise ValueError('use --shape-only, --forward-only, or full two-update execution')
        config = json.loads(args.mixed_protocol.read_text())
        horizon, directional_config = validate_configuration(config)
        model = json.loads(args.protocol.read_text())
        pool_config = json.loads(args.pool_protocol.read_text())
        daily_config = json.loads(args.admission_config.read_text())
        admission._validate_inputs(pool_config, model, {}, shape_only=args.shape_only,
                                   admission_configuration=daily_config)
        if _resolved_reference(config['model_protocol'], relative_to=args.mixed_protocol.resolve()) != args.protocol.resolve():
            raise ValueError('mixed and CLI model protocols differ')
        merge = json.loads(args.catalogue_report.read_text())
        if merge['status'] != 'passed':
            raise ValueError('expanded catalogue merge is not accepted')
        catalogue_path = _resolved_reference(merge['catalogue'], relative_to=args.catalogue_report.resolve())
        with gzip.open(catalogue_path, 'rt') as handle:
            catalogue = json.load(handle)
        sampler, _ = pool_plan(catalogue, pool_config)
        windows = WindowPool(args.selection)
        daily_groups, recurrent_groups = select_draws(sampler, windows.selection, horizon=horizon)
        monitor_rows = tuple(catalogue['monitor_rows'])
        if len(monitor_rows) != 64:
            raise ValueError('expected fixed64 excluded TRAIN monitor')
        report.update(configuration=config, model_protocol=model, horizon_days=horizon,
                      scope=f'fresh_daily_plus_T{horizon}_mixed_optimizer_admission',
                      directional_check={'requested': directional_config is not None,
                                         'status': 'pending' if directional_config is not None else 'not_requested'},
                      daily_rows=[[r.as_metadata() for r in rows] for rows in daily_groups],
                      recurrent_windows=recurrent_groups, monitor_rows=monitor_rows,
                      daily_source_identity=catalogue['source_identity'],
                      recurrent_source_identity=windows.selection['source_identity'])
        emit('bootstrap')
        runtime, head, template, scales, parameters, _, _ = prototype.bootstrap(args, model)
        jax = runtime['jax']
        import jax.numpy as jnp

        report.update(parameter_count=sum(p.size for p in jax.tree_util.tree_leaves(parameters)),
                      device=str(runtime['devices'][0]),
                      memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'))
        if report['parameter_count'] != model['parameter_count'] or model['parameter_count'] != 109464375:
            raise ValueError('actual parameter tree differs from declared full-network architecture')
        graph = calibration._loss_protocol(model, canopy_source_pft_index=calibration._canopy_source_pft_index(runtime, head))
        objective = make_objective(runtime, head, template, scales, graph)
        base_weights = calibration._base_weights(graph)
        daily_backward = calibration._make_evaluator(jax, jnp, objective, base_weights)
        daily_forward = jax.jit(calibration._make_batch_objective(jax, jnp, objective, base_weights))
        coefficients = jnp.asarray(admission._coefficients(daily_config), dtype=jnp.float64)
        recurrent_loss, loss_spec = bind_recurrent_science_loss(runtime, head, config['recurrent_weights'])
        report['recurrent_scales'] = loss_spec
        component = make_rollout_loss(jax, jnp, make_canonical_step(head, template), recurrent_loss)
        recurrent_backward = jax.jit(jax.value_and_grad(component.checkpointed_loss, argnums=0))
        recurrent_tangent = None
        if directional_config is not None:
            recurrent_tangent = jax.jit(lambda p, d, s, z, day, t, c:
                jax.jvp(lambda candidate: component.checkpointed_loss(candidate, s, z, day, t, c),
                        (p,), (d,)))
        recurrent_forward = jax.jit(component.rollout)
        recurrent_terms = jax.jit(recurrent_loss.terms)
        extract = jax.jit(lambda obs: _accepted_science(shared_science_trajectories(obs, head.contract)))
        context = {'stock_weight': jnp.asarray(1., dtype=jnp.float64)}
        update = make_update(model)
        add, finish = make_gradient_accumulator(jax, jnp)
        combine = make_mixed_gradient(jax, jnp)
        outer = config['outer_weights']
        source = _resolved_reference(catalogue['source_manifest'], relative_to=catalogue_path)
        daily_pool = PreparedOneDayPool(source, cache_size=4)
        if daily_pool.identity != catalogue['source_identity']:
            raise ValueError('daily reader and catalogue identities differ')

        def sample(rows):
            metadata = admission._rows(rows)
            return daily_pool.batch([(int(r['artifact_id']), int(r['batch_row'])) for r in metadata])

        first_daily = sample(daily_groups[0][:8])
        first_case = windows.case(jax, recurrent_groups[0][0])
        arguments = (parameters, *first_case[:4], context)
        if args.shape_only:
            (_, _), dg = jax.eval_shape(daily_backward, parameters, first_daily, coefficients)
            _, rg = jax.eval_shape(recurrent_backward, *arguments)
            if recurrent_tangent is not None:
                direction_shape = jax.tree_util.tree_map(jnp.zeros_like, parameters)
                jax.eval_shape(recurrent_tangent, parameters, direction_shape, *arguments[1:])
            mixed = jax.eval_shape(combine, dg, rg, outer['daily'], outer['recurrent'])
            state = jax.eval_shape(initialize_target_oriented_adamw, parameters)
            jax.eval_shape(update, parameters, mixed, state)
            _, obs = jax.eval_shape(recurrent_forward, parameters, *first_case[:3])
            jax.eval_shape(recurrent_terms, parameters, first_case[0], obs, first_case[3], context)
            jax.eval_shape(extract, obs)
            classification = ('actual_daily_T7_gradient_optimizer_shapes_only' if horizon == 7 else
                              'actual_daily_T30_gradient_optimizer_shapes_only')
            emit('completed', status='passed', classification=classification)
            return report

        def daily_panel(current, rows):
            losses, auxiliary_rows = [], []
            for i in range(0, len(rows), 8):
                value, auxiliary = daily_forward(current, sample(rows[i:i+8]), coefficients)
                jax.block_until_ready((value, auxiliary))
                losses.append(float(value))
                auxiliary_rows.append(auxiliary)
            aux = admission._concatenate_auxiliary(jax, auxiliary_rows)
            hard = calibration._hard_summary(aux['hard'], batch_size=len(rows))
            result = dict(mean_loss=float(np.mean(losses)), count=len(rows), hard=hard,
                          diagnostics=calibration._diagnostics(jax, aux, batch_size=len(rows)))
            if not np.all(np.isfinite(losses)) or not hard['all_zero']:
                raise ValueError('single-day forward physical/finite gate failed')
            return result

        def recurrent_panel(current, row):
            initial, discrete, days, targets, expected_discrete = windows.case(jax, row)
            _, obs = recurrent_forward(current, initial, discrete, days)
            jax.block_until_ready(obs)
            hard = physical_checks(obs, targets, expected_discrete, days[0], head.contract, scales)
            terms = recurrent_terms(current, initial, obs, targets, context)
            jax.block_until_ready(terms)
            for section in WEIGHT_COMPONENTS:
                for key, term in getattr(terms, section).items():
                    hard[section+'.'+key+'.invalid'] = int(term.invalid_prediction_count + term.invalid_target_count)
                    hard[section+'.'+key+'.empty'] = int(term.count == 0)
            first = jax.tree_util.tree_map(lambda v: v[0], days[0])
            assembled = head.input_assembler.assemble(first)
            feedback = feedback_diagnostics(jax, jnp, runtime, head, model, initial, assembled, obs, targets, days[0])
            hard.update(feedback_hard_counts(feedback))
            start_obs = {**{k: v[:1] for k, v in targets.items()},
                         'state': initial[None, :], 'defined': assembled.state_defined.other[None, :]}
            pf, ps = extract(obs)
            tf, ts = extract(targets)
            _, ss = extract(start_obs)
            physical = physical_window(pf, tf, ps, ts, ss)
            result = dict(window=row, loss=float(terms.total), hard=hard,
                          weighted_components=terms.weighted_components, physical=physical,
                          feedback=feedback)
            if any(hard.values()) or not np.isfinite(float(terms.total)):
                report['failed_window'] = result
                raise ValueError('recurrent forward physical/validity gate failed')
            return result

        def panels(current, index):
            emit('forward_panels', panel_update=index)
            began = time.perf_counter()
            daily = [daily_panel(current, rows) for rows in daily_groups]
            excluded = daily_panel(current, monitor_rows)
            recurrent = [recurrent_panel(current, row) for group in recurrent_groups for row in group]
            matched = [outer['daily']*daily[i]['mean_loss'] + outer['recurrent']*
                       np.mean([r['loss'] for r in recurrent[2*i:2*i+2]]) for i in range(2)]
            report['panels'].append(dict(update=index, daily_training=daily,
                excluded_train_monitor=excluded, recurrent_training=recurrent,
                matched_training_mixed_losses=matched,
                recurrent_physical_summary=summarize_windows([r['physical'] for r in recurrent]),
                seconds=time.perf_counter()-began))
            emit('forward_panels_completed', memory=_device_memory(runtime['devices'][0]))

        # All selected fresh trajectories must be physically valid before reverse compilation.
        panels(parameters, 0)
        if args.forward_only:
            classification = ('fresh_forward_only_no_updates' if horizon == 7 else
                              'fresh_daily_T30_forward_only_no_updates')
            emit('completed', status='passed', classification=classification)
            return report

        def compile_gradient(name, evaluator, values):
            emit('compile_'+name)
            began = time.perf_counter()
            executable = evaluator.lower(*values).compile()
            memory = _compiled_memory(executable)
            report['compilation'][name] = dict(seconds=time.perf_counter()-began, memory=memory)
            emit('compiled_'+name)
            if memory['estimated_live_bytes'] > config['compiled_memory_limit_gib'] * 1024**3:
                raise MemoryError(name+' exceeds declared compiled-memory envelope')
            return executable

        # Compile recurrent first; single-day compilation may reuse the micro8 cache.
        recurrent_executable = compile_gradient('recurrent', recurrent_backward, arguments)
        daily_executable = compile_gradient('daily', daily_backward, (parameters, first_daily, coefficients))
        optimizer = initialize_target_oriented_adamw(parameters)
        store = StageStore(args.output.with_suffix('.stages'), dict(configuration=config,
            source_git_head=report['source_git_head'], daily_rows=report['daily_rows'],
            recurrent_windows=json_safe(recurrent_groups)))

        def check_first_recurrent_gradient(value, gradient, case, row):
            """Persist the actual first-window reverse result before compiling its JVP."""
            from scripts.hpc.probe_trained_checkpointed_grad import rademacher_direction

            numeric = _tree_numeric_report(jax, gradient)
            if not np.isfinite(float(value)) or not numeric['all_finite'] or numeric['l2_norm'] <= 0:
                raise ValueError('first recurrent native gradient is nonfinite or empty')
            native = dict(value=float(value), gradient=gradient, summary=numeric,
                          window=json_safe(row))
            store.write('first_recurrent_native_gradient', native, jax)
            report['first_recurrent_native_gradient'] = dict(
                value=float(value), summary=numeric, window=json_safe(row),
                stage_file=str(store.directory/'first_recurrent_native_gradient.pkl'))
            emit('first_recurrent_native_gradient_saved')

            references, first_direction = [], None
            dot = jax.jit(lambda left, right: admission._tree_dot(jax, jnp, left, right))
            for index, seed in enumerate(directional_config['seeds']):
                began = time.perf_counter()
                host_direction, count = rademacher_direction(jax, parameters, seed)
                direction = jax.tree_util.tree_map(jnp.asarray, host_direction)
                projection = dot(gradient, direction)
                jax.block_until_ready(projection)
                references.append(dict(seed=seed, reverse_projection=float(projection),
                                       parameter_count=count, seconds=time.perf_counter()-began))
                if index == 0:
                    first_direction = direction
                del host_direction, direction, projection
            report['directional_check'].update(reverse_projections=references)
            emit('recurrent_reverse_projections_completed')

            emit('compile_recurrent_jvp')
            began = time.perf_counter()
            executable = recurrent_tangent.lower(parameters, first_direction, *case[:4], context).compile()
            memory = _compiled_memory(executable)
            report['compilation']['recurrent_jvp'] = dict(
                seconds=time.perf_counter()-began, memory=memory)
            emit('compiled_recurrent_jvp', memory=_device_memory(runtime['devices'][0]))
            if memory['estimated_live_bytes'] > config['compiled_memory_limit_gib'] * 1024**3:
                raise MemoryError('recurrent_jvp exceeds declared compiled-memory envelope')

            directions = []
            for index, reference in enumerate(references):
                direction = first_direction
                if index:
                    host_direction, _ = rademacher_direction(jax, parameters, reference['seed'])
                    direction = jax.tree_util.tree_map(jnp.asarray, host_direction)
                    del host_direction
                began = time.perf_counter()
                primal, tangent = executable(parameters, direction, *case[:4], context)
                jax.block_until_ready((primal, tangent))
                result = directional_record(
                    primal, reference['reverse_projection'], tangent, value,
                    seed=reference['seed'], rtol=directional_config['rtol'],
                    atol=directional_config['atol'])
                result['seconds'] = time.perf_counter()-began
                directions.append(result)
                report['directional_check'].update(directions=directions)
                emit('recurrent_direction_completed')
                if index:
                    del direction
            passed = all(row['passed'] and row['resolved_projection'] for row in directions)
            report['directional_check'].update(status='passed' if passed else 'rejected',
                                               bounded_directional_gate_passed=passed,
                                               memory_before_release=_device_memory(runtime['devices'][0]))
            report['bounded_directional_gate_passed'] = passed
            del executable, first_direction, dot
            gc.collect()
            report['directional_check']['memory_after_release'] = _device_memory(runtime['devices'][0])
            emit('recurrent_directional_check_completed')
            if not passed:
                raise ValueError('bounded recurrent directional check failed')

        for index in range(2):
            emit('update_gradient', update_index=index+1)
            began = time.perf_counter()
            daily_total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
            daily_loss, recurrent_value = 0., 0.
            for i in range(0, 64, 8):
                (value, aux), gradient = daily_executable(parameters, sample(daily_groups[index][i:i+8]), coefficients)
                daily_total = add(daily_total, gradient, 8)
                jax.block_until_ready((value, aux, daily_total))
                if not np.isfinite(float(value)) or not calibration._hard_summary(jax.device_get(aux['hard']), batch_size=8)['all_zero']:
                    raise ValueError('daily gradient primal physical/finite gate failed')
                daily_loss += float(value)/8
            daily_mean = finish(daily_total, 64)
            del daily_total, gradient, aux
            recurrent_total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
            for window_index, row in enumerate(recurrent_groups[index]):
                case = windows.case(jax, row)
                value, gradient = recurrent_executable(parameters, *case[:4], context)
                recurrent_total = add(recurrent_total, gradient, 1)
                jax.block_until_ready((value, recurrent_total))
                if not np.isfinite(float(value)):
                    raise ValueError('recurrent gradient primal nonfinite')
                if index == 0 and window_index == 0 and directional_config is not None:
                    check_first_recurrent_gradient(value, gradient, case, row)
                recurrent_value += float(value)/2
            recurrent_mean = finish(recurrent_total, 2)
            del recurrent_total, gradient
            gradient = combine(daily_mean, recurrent_mean, outer['daily'], outer['recurrent'])
            jax.block_until_ready(gradient)
            gradient_seconds = time.perf_counter()-began
            numeric = {name: _tree_numeric_report(jax, tree) for name, tree in
                       [('daily', daily_mean), ('recurrent', recurrent_mean), ('mixed', gradient)]}
            dot = float(admission._tree_dot(jax, jnp, daily_mean, recurrent_mean))
            if not all(v['all_finite'] and v['l2_norm'] > 0 for v in numeric.values()):
                raise ValueError('component or mixed gradient nonfinite/empty')
            if index == 0:
                store.write('first_mixed_gradient', dict(gradient=gradient, numeric=numeric,
                    daily_loss=daily_loss, recurrent_loss=recurrent_value), jax)
                emit('first_mixed_gradient_saved')
            del daily_mean, recurrent_mean
            proposed, proposed_state, norm, clip = update(parameters, gradient, optimizer)
            jax.block_until_ready((proposed, proposed_state, norm, clip))
            if (not admission._host_tree_finite(jax, proposed) or
                    not admission._host_tree_finite(jax, proposed_state) or
                    not np.isfinite(float(norm)) or not 0 < float(clip) <= 1):
                raise ValueError('complete AdamW update invalid')
            record = dict(update=index+1, daily_loss=daily_loss, recurrent_loss=recurrent_value,
                          mixed_loss=outer['daily']*daily_loss+outer['recurrent']*recurrent_value,
                          gradients=numeric, component_cosine=dot/(numeric['daily']['l2_norm']*numeric['recurrent']['l2_norm']),
                          norm=float(norm), clip=float(clip), gradient_seconds=gradient_seconds,
                          total_update_seconds=time.perf_counter()-began)
            parameters, optimizer = proposed, proposed_state
            del gradient, proposed, proposed_state
            report['updates'].append(record)
            report['completed_updates'] = index+1
            emit('optimizer_completed', memory=_device_memory(runtime['devices'][0]))
            panels(parameters, index+1)
            before = report['panels'][-2]['matched_training_mixed_losses'][index]
            after = report['panels'][-1]['matched_training_mixed_losses'][index]
            record.update(same_draw_forward_loss_before=before, same_draw_forward_loss_after=after,
                          same_draw_loss_decreased=bool(after < before))
        classification = ('two_mixed_updates_technically_admitted' if horizon == 7 else
                          'two_mixed_daily_T30_updates_technically_admitted')
        emit('completed', status='passed', classification=classification,
             all_hard_counts_zero=True, memory=_device_memory(runtime['devices'][0]),
             scientific_acceptance=False, daily_reader=daily_pool.statistics())
        return report
    except Exception as error:
        emit('failed', status='resource_limited' if isinstance(error, MemoryError) else 'failed',
             error=f'{type(error).__name__}: {error}')
        raise


def parser():
    result = prototype.parser()
    for name in ('mixed-protocol', 'pool-protocol', 'admission-config', 'catalogue-report', 'selection'):
        result.add_argument('--'+name, type=Path, required=True)
    result.add_argument('--forward-only', action='store_true')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
