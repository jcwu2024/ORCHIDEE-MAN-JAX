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
    epoch = config.get('schema_version') == 'mixed_learning_epoch_v1'
    if (config.get('schema_version') not in ('mixed_learning_v1', 'mixed_learning_epoch_v1')
            or config.get('updates') != (5415 if epoch else 200)
            or config.get('milestones') != ([0,200,1000,2500,4000,5415] if epoch else [0,50,100,200])
            or config.get('outer_weights') != {'daily': 1., 'T7': .5, 'T30': .5}
            or config.get('sampling') != dict(seed=20260912, batch_size=64,
                microbatch_size=8, max_artifacts_per_batch=5, batch_packing='fixed_replay_pad')
            or model.get('seed') != 20260911
            or model.get('parameter_count') != 109464375
            or model.get('trainable_scope') != 'all_neural_parameters'
            or model['optimizer']['schedule'] != ('constant_one_pool_pass' if epoch else 'constant_200_updates')):
        raise ValueError('mixed learning differs from declared fresh learning protocol')


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


def newest_checkpoint(output, requested, completed, milestones=(0,50,100,200)):
    """Prefer a newer retained milestone over an older rolling recovery file."""
    available = [index for index in milestones
                 if index > completed and checkpoint_path(output,index).is_file()]
    return checkpoint_path(output,max(available)) if available else requested


def reconcile_resume_checkpoint(report, path, completed):
    """Recover metadata if a process died after checkpoint rename, before emit."""
    path = Path(path).resolve()
    records = report.get('checkpoints', [])
    matching = [row for row in records if Path(row['path']).resolve() == path]
    if any(row['update'] == completed for row in matching):
        return
    report['checkpoints'] = [row for row in records
                             if Path(row['path']).resolve() != path] + [
        dict(update=completed, path=str(path), bytes=path.stat().st_size,
             recovered_metadata=True)]


def profile_source_method(training_report, config, model):
    """Bind a diagnostic checkpoint to the completed run without resuming it."""
    if (training_report.get('status') != 'completed'
            or training_report.get('completed_updates') != 200
            or training_report.get('configuration') != config
            or training_report.get('model_protocol') != model
            or training_report.get('scope') != 'bounded_200_update_learning'):
        raise ValueError('profile requires the completed declared 200-update run')
    method = dict(implementation='mixed_learning_v1',configuration=config,
                model_protocol=model,scope='bounded_200_update_learning',
                integration_updates=None,source_git_head=training_report['source_git_head'])
    if training_report.get('execution_policy') is not None:
        method.update(execution_policy=training_report['execution_policy'],
                      recurrent_execution=training_report['execution_policy']['recurrent_execution'])
    return method


def configure_execution(args):
    """Resolve the declared combined policy before bootstrap and checkpoint IO."""
    if args.execution_policy is None:
        if (getattr(args, 'dual_device_profile', False)
                or getattr(args, 'dual_device_training', False)):
            raise ValueError('dual-device execution requires the admitted execution policy')
        return None
    policy = json.loads(args.execution_policy.read_text())
    supported = dict(schema_version='mixed_learning_execution_v1',
        attention_score_dtype='fp32', recurrent_execution='split_checked_v1',
        source_family_pooling_dtype='legacy', physical_dtype='float64')
    if policy != supported:
        raise ValueError('unsupported mixed-learning execution policy')
    if args.compare_checked_recurrent and args.checked_reference_report is None:
        raise ValueError('combined admission requires the preserved legacy reference report')
    args.resolved_execution_policy = policy
    args.attention_score_dtype = policy['attention_score_dtype']
    args.separate_checked_validation = True
    args.checked_recurrent = not args.compare_checked_recurrent
    return policy


def validate_learning_budget_mode(args, *, profiling, execution_policy):
    """Fail closed before a fixed-checkpoint learning-budget diagnostic."""
    if not getattr(args, 'learning_budget_profile', False):
        return
    if not profiling:
        raise ValueError('learning-budget profiling requires fixed-checkpoint profiling')
    if execution_policy is None:
        raise ValueError('learning-budget profiling requires the admitted execution policy')
    conflicts = (
        args.profile_trace,
        args.precision_stage is not None,
        args.dual_device_profile,
        args.compare_checked_recurrent,
        args.checked_reference_report is not None,
        args.precision_reference_report is not None,
        args.precision_daily_reference is not None,
        args.learning_schedule is not None,
    )
    if any(conflicts):
        raise ValueError(
            'learning-budget profiling is separate from trace, precision, dual, '
            'comparison, and staged-training modes')
    if not args.checked_recurrent or not args.separate_checked_validation:
        raise ValueError('learning-budget profiling requires checked split execution')


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


def saved_gradient_path(report_path, report, name):
    recorded = report.get('checked_comparisons',{}).get(name,{}).get('reference_gradient')
    active = report.get('reference_gradient')
    if not recorded and active and Path(active).name.endswith(f'.{name}.reference.pkl'):
        recorded = active
    return Path(recorded) if recorded else report_path.with_suffix(f'.{name}.reference.pkl')


_PRECISION_STAGES = ('daily', 'T7', 'T30', 'full')


def precision_stage_components(stage):
    """Return the fixed update components admitted by a precision stage."""
    if stage not in _PRECISION_STAGES:
        raise ValueError('unsupported precision stage')
    return ('daily', 'T7', 'T30') if stage == 'full' else (stage,)


def precision_reference_path(report_path, report, component):
    """Resolve an immutable legacy full-gradient reference for one component."""
    if component not in ('daily', 'T7', 'T30'):
        raise ValueError('unsupported precision reference component')
    if component == 'daily':
        recorded = report.get('precision_comparisons', {}).get('daily', {}).get(
            'reference_gradient'
        )
        return Path(recorded) if recorded else None
    return saved_gradient_path(report_path, report, component)


def precision_reference_identity(component, *, checkpoint, rows=None, window=None):
    """Small, serializable identity for a cached fixed-panel gradient."""
    if component == 'daily':
        if rows is None or window is not None:
            raise ValueError('daily precision reference requires rows only')
        rows = admission._rows(rows)
    elif component in ('T7', 'T30'):
        if window is None or rows is not None:
            raise ValueError('recurrent precision reference requires one window only')
    else:
        raise ValueError('unsupported precision reference component')
    return dict(component=component, checkpoint=str(checkpoint), rows=rows, window=window)


def _mask_precision_auxiliary_rows(values, valid):
    """Zero undefined row values while retaining the validity mask itself."""
    result = {}
    valid = np.asarray(valid, dtype=np.bool_)
    for name, value in values.items():
        array = np.asarray(value)
        if (array.dtype.kind in 'fc' and array.ndim >= valid.ndim
                and array.shape[:valid.ndim] == valid.shape):
            mask = valid.reshape(valid.shape + (1,) * (array.ndim - valid.ndim))
            array = np.where(mask, array, 0.)
        result[name] = array
    return result


def masked_precision_daily_auxiliary(jax, auxiliary):
    """Keep daily loss terms plus only defined canopy/litter numeric entries."""
    host = jax.device_get({
        'terms': auxiliary['terms'],
        'physical': auxiliary['physical'],
        'canopy_stock_change': auxiliary['canopy_stock_change'],
        'litterpart_feedback': auxiliary['litterpart_feedback'],
    })
    canopy = host['canopy_stock_change']
    litter = host['litterpart_feedback']
    physical = {
        group: {
            name: _mask_precision_auxiliary_rows(field, np.asarray(field['defined']))
            for name, field in fields.items()
        }
        for group, fields in host['physical'].items()
    }
    return {
        'terms': np.asarray(host['terms']),
        'physical': physical,
        'canopy_stock_change': _mask_precision_auxiliary_rows(
            canopy, np.asarray(canopy['row_valid']) & np.asarray(canopy['scale_valid'])
        ),
        'litterpart_feedback': _mask_precision_auxiliary_rows(
            litter, np.asarray(litter['valid'])
        ),
    }


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
    from scripts.hpc.mixed_learning_reporting import MixedLearningReporter
    from scripts.hpc.mixed_learning_stages import load_learning_schedule
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
    from scripts.hpc.probe_public_rollout import physical_checks
    from scripts.hpc.run_fresh_start_one_day_pool import host_peak_bytes
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.run_trained_forward_screen import feedback_diagnostics, feedback_hard_counts
    from scripts.hpc.trained_screen_common import PreparedWindowPool, WindowPool, json_safe

    execution_policy = configure_execution(args)
    if args.admission_only or args.cpu_forward_only:
        raise ValueError('use shape-only or integration-updates for bounded checks')
    profiling = args.profile_checkpoint is not None
    if profiling != (args.profile_training_report is not None):
        raise ValueError('profile checkpoint and source training report are required together')
    validate_learning_budget_mode(
        args, profiling=profiling, execution_policy=execution_policy)
    if profiling and (args.resume or args.integration_updates or args.shape_only):
        raise ValueError('profiling is a separate diagnostic, not resume or integration training')
    if args.profile_trace and not profiling:
        raise ValueError('trace is available only in diagnostic profiling mode')
    if args.learning_schedule and profiling:
        raise ValueError('learning schedules are training methods, not fixed-checkpoint diagnostics')
    if args.dual_device_profile and not profiling:
        raise ValueError('dual-device profiling requires fixed-checkpoint profiling')
    if args.dual_device_training and profiling:
        raise ValueError('dual-device training is not a fixed-checkpoint diagnostic')
    if args.dual_device_training and args.dual_device_profile:
        raise ValueError('dual-device training and profiling are separate modes')
    if args.dual_device_profile and (args.profile_trace or args.precision_stage
                                     or args.compare_checked_recurrent
                                     or args.learning_budget_profile):
        raise ValueError('dual-device profiling is separate from legacy comparison diagnostics')
    if args.compare_checked_recurrent and not profiling:
        raise ValueError('checked-recurrent admission requires fixed-checkpoint profiling')
    if args.checked_reference_report and not args.compare_checked_recurrent:
        raise ValueError('reference reuse is only for checked-recurrent admission')
    if args.precision_stage and not profiling:
        raise ValueError('precision staging requires fixed-checkpoint profiling')
    if args.precision_reference_report and not args.precision_stage:
        raise ValueError('precision reference reuse requires a precision stage')
    if args.precision_daily_reference and not args.precision_stage:
        raise ValueError('daily precision reference reuse requires a precision stage')
    if args.precision_stage and args.compare_checked_recurrent:
        raise ValueError('precision staging and checked-recurrent admission are separate diagnostics')
    if args.separate_checked_validation and not (args.compare_checked_recurrent or args.checked_recurrent):
        raise ValueError('separate validation requires checked execution or comparison')
    if (args.checked_recurrent and profiling
            and not (args.dual_device_profile or args.learning_budget_profile)):
        raise ValueError('use compare-checked-recurrent for fixed-checkpoint admission')
    if args.checked_recurrent and args.shape_only:
        raise ValueError('checked execution is admitted by checked-profile, not the legacy shape-only path')
    if args.dual_device_training and (
            not args.checked_recurrent or not args.separate_checked_validation):
        raise ValueError('dual-device training requires admitted split checked execution')
    if args.output.exists() and not args.resume:
        raise FileExistsError('fresh run requires a new output; use --resume for recovery')
    if args.resume and not args.output.is_file():
        raise FileNotFoundError('resume requires the original report alongside its checkpoint')
    config = json.loads(args.mixed_protocol.read_text())
    model = json.loads(args.protocol.read_text())
    validate_configuration(config, model)
    if (args.mixed_protocol.parent/config['model_protocol']).resolve() != args.protocol.resolve():
        raise ValueError('CLI and mixed model protocols differ')
    learning_schedule = (None if args.learning_schedule is None else load_learning_schedule(
        args.learning_schedule,
        outer_weights=config['outer_weights'],
        available_updates=config['updates'],
    ))
    training_total = (config['updates'] if learning_schedule is None
                      else learning_schedule.total_updates)
    training_milestones = (tuple(config['milestones']) if learning_schedule is None
                           else learning_schedule.milestones)
    precision_stage = args.precision_stage
    precision_components = (() if precision_stage is None
                            else precision_stage_components(precision_stage))
    precision_candidate = getattr(args, 'attention_score_dtype', 'legacy') == 'fp32'
    if config['schema_version'] == 'mixed_learning_epoch_v1' and execution_policy is None:
        raise ValueError('one-pool-pass training requires the admitted execution policy')
    if precision_candidate and precision_stage is None and execution_policy is None:
        raise ValueError('FP32 Attention requires precision staging or a declared execution policy')
    if execution_policy is not None and precision_stage:
        raise ValueError('combined execution uses checked-profile, not isolated precision staging')
    if precision_candidate and precision_stage:
        needs_profile_reference = any(name in ('T7', 'T30') for name in precision_components)
        needs_daily_reference = 'daily' in precision_components
        if needs_profile_reference and args.precision_reference_report is None:
            raise ValueError('FP32 recurrent precision staging requires an immutable legacy profile report')
        if needs_daily_reference and args.precision_daily_reference is None:
            raise ValueError('FP32 daily precision staging requires an immutable legacy daily gradient')
    limit = args.integration_updates or training_total
    if limit > training_total:
        raise ValueError('integration budget exceeds the declared learning schedule')
    if args.stop_after_update is not None:
        if args.integration_updates is None:
            raise ValueError('stop-after-update is only for bounded integration recovery')
        if not 0 < args.stop_after_update < limit:
            raise ValueError('stop-after-update must be within and before the integration budget')
    execution_limit = args.stop_after_update or limit
    scope = ('performance_profile_not_training' if profiling else
             'integration_only' if args.integration_updates else
             'declared_learning_schedule' if learning_schedule is not None else
             'one_pool_pass_learning' if config['schema_version'] == 'mixed_learning_epoch_v1'
             else 'bounded_200_update_learning')
    started = time.perf_counter()
    current_head = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report = dict(schema_version='mixed_learning_result_v1', status='running',
        source_git_head=current_head,
        configuration=config, model_protocol=model, scope=scope, completed_updates=0,
        updates=[], monitors=[], compilation={}, checkpoints=[],
        scientific_acceptance=False, checkpoint_inherited=False)
    if execution_policy is not None:
        report['execution_policy'] = execution_policy
    if args.dual_device_training:
        from scripts.hpc.mixed_learning_dual_device import DUAL_DEVICE_TRAINING_LAYOUT

        report['device_execution'] = DUAL_DEVICE_TRAINING_LAYOUT
    if learning_schedule is not None:
        report['learning_schedule'] = learning_schedule.record
        report['learning_progress'] = learning_schedule.progress(0)
    if precision_stage:
        report.update(precision_study=dict(stage=precision_stage,
            candidate_attention_score_dtype=getattr(args, 'attention_score_dtype', 'legacy'),
            components=list(precision_components),
            recurrent_execution='legacy_original_for_attention_attribution',
            claim_scope='fixed loss/full-gradient/hard-counter parity; not full-state equivalence',
            retained_parameter_or_optimizer_state=False))
    resume_data_consumption = None
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (previous['configuration'], previous['model_protocol'], previous['scope']) != (config,model,scope):
            raise ValueError('recovery report method mismatch')
        if previous.get('execution_policy') != execution_policy:
            raise ValueError('recovery report execution policy mismatch')
        if previous.get('device_execution') != report.get('device_execution'):
            raise ValueError('recovery report device execution mismatch')
        if learning_schedule is not None:
            if previous.get('learning_schedule') != learning_schedule.record:
                raise ValueError('recovery report learning schedule mismatch')
            learning_schedule.validate_progress(
                previous.get('learning_progress'),
                completed_updates=previous['completed_updates'],
            )
            learning_schedule.validate_update_records(
                previous.get('updates'),
                completed_updates=previous['completed_updates'],
            )
            resume_data_consumption = previous.get('data_consumption')
        report = previous
        report['prior_elapsed_seconds'] = report.get('prior_elapsed_seconds',0.)+report.get('elapsed_seconds',0.)

    batches = None
    reporter = MixedLearningReporter(
        args.output, write_json=_atomic_json, json_safe=json_safe,
        interval=1 if profiling else getattr(args, 'report_interval', 25))
    report['reporting_policy'] = reporter.policy

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter()-started,
                      host_peak_bytes=host_peak_bytes())
        if learning_schedule is not None:
            progress = learning_schedule.progress(report['completed_updates'])
            report['learning_progress'] = progress
            if batches is not None:
                report['data_consumption'] = learning_schedule.data_consumption(
                    report['completed_updates'],batches)
            report['training_seconds'] = sum(
                row.get('seconds', 0.) for row in report['updates']
            )
            report['phase_training_seconds'] = {
                phase.name: sum(row.get('seconds', 0.) for row in report['updates']
                                if row.get('phase') == phase.name)
                for phase in learning_schedule.phases
            }
        reporter.publish(report)
        print(f'mixed_learning stage={stage} updates={report["completed_updates"]} '
              f'seconds={report["elapsed_seconds"]:.1f}', flush=True)

    dual_trainer = None
    try:
        emit('metadata')
        plan = load_plan(args, config)
        report['data_plan'] = plan['summary']
        identities = plan['identities']
        method = dict(implementation='mixed_learning_v1', configuration=config,
                      model_protocol=model, scope=scope, integration_updates=args.integration_updates,
                      source_git_head=current_head)
        if learning_schedule is not None:
            method['learning_schedule'] = learning_schedule.record
        if args.checked_recurrent:
            method['recurrent_execution'] = ('split_checked_v1' if args.separate_checked_validation
                                             else 'compact_checked_v1')
        if execution_policy is not None:
            method.update(execution_policy=execution_policy,
                          recurrent_execution=execution_policy['recurrent_execution'])
        if args.dual_device_training:
            method['device_execution'] = report['device_execution']
        fresh_identity = dict(seed=model['seed'],initialization=model['initialization'],
                              parameter_count=model['parameter_count'],inherited_weights=False)
        recovery_identity = dict(expected_method=method,expected_source_identities=identities,
            expected_schedule_identity=identities,expected_fresh_initialization_identity=fresh_identity)
        batches, schedule = plan['batches'], plan['schedule']
        recurrent_updates = (training_total if learning_schedule is None
                             else learning_schedule.recurrent_cursor(training_total))
        if training_total > len(batches) or recurrent_updates > len(schedule['update_rows']):
            raise ValueError('learning schedule exceeds the loaded daily/recurrent plan')
        if learning_schedule is not None and args.resume:
            learning_schedule.validate_data_consumption(
                resume_data_consumption,
                completed_updates=report['completed_updates'],
                batches=batches,
            )
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

        def primary_helper(function, *values, **keywords):
            if not args.dual_device_training:
                return function(*values, **keywords)
            with (jax.default_device(primary_device),
                  device_helper_cache(primary_device)):
                return function(*values, **keywords)

        dual_devices = None
        if args.dual_device_training:
            from scripts.hpc.checked_recurrent_execution import device_helper_cache

            dual_devices = tuple(jax.local_devices())
            if (len(dual_devices) != 2
                    or any(device.platform != 'gpu' for device in dual_devices)):
                raise ValueError('dual-device training requires exactly two visible GPUs')
            primary_device, recurrent_device = dual_devices
            with jax.default_device(primary_device), device_helper_cache(primary_device):
                first_daily = jax.device_put(sample(fixed_rows[:8]), primary_device)
            with jax.default_device(recurrent_device), device_helper_cache(recurrent_device):
                cases = {h: jax.device_put(train_pool.case(
                    jax, schedule['update_rows'][0][f'T{h}']), recurrent_device)
                    for h in (7,30)}
        else:
            first_daily = sample(fixed_rows[:8])
            cases = {h: train_pool.case(
                jax, schedule['update_rows'][0][f'T{h}']) for h in (7,30)}

        def execution_memory():
            if dual_devices is None:
                return _device_memory(runtime['devices'][0])
            return {str(device): _device_memory(device) for device in dual_devices}
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
            saved = load_mixed_training_checkpoint(
                args.profile_checkpoint, maximum_updates=training_total, **profile_identity
            )
            if saved['completed_updates'] != 200:
                raise ValueError('profile checkpoint must be at update200')
            del parameters
            parameters, optimizer = jax.device_put((saved['parameters'],saved['optimizer_state']))
            report.update(checkpoint_inherited=True,profile_source_checkpoint=str(args.profile_checkpoint),
                          profile_source_update=200,persistent_parameter_updates=0,
                          profile_source_method=profile_identity['expected_method'])
            del saved
        elif args.resume:
            saved = load_mixed_training_checkpoint(
                args.resume, maximum_updates=training_total, **recovery_identity
            )
            resolved_resume = newest_checkpoint(
                args.output,args.resume,saved['completed_updates'],training_milestones
            )
            if resolved_resume != args.resume:
                del saved
                saved = load_mixed_training_checkpoint(
                    resolved_resume, maximum_updates=training_total, **recovery_identity
                )
            del parameters
            parameters, optimizer = jax.device_put((saved['parameters'],saved['optimizer_state']))
            trim_resume_report(report,saved['completed_updates'])
            reconcile_resume_checkpoint(report,resolved_resume,saved['completed_updates'])
            report['resumed_from'] = str(resolved_resume)
            del saved
        else:
            optimizer = primary_helper(initialize_target_oriented_adamw, parameters)
        if args.dual_device_training:
            with (jax.default_device(primary_device),
                  device_helper_cache(primary_device)):
                parameters, optimizer, coefficients = jax.device_put(
                    (parameters, optimizer, coefficients), primary_device)
                jax.block_until_ready((parameters, optimizer, coefficients))
        if report['completed_updates'] > execution_limit:
            raise ValueError('recovery exceeds this execution budget')

        if args.dual_device_profile:
            from scripts.hpc.checked_recurrent_execution import bind_checked_backward
            from scripts.hpc.mixed_learning_dual_device import profile_dual_device

            emit('profile_ready')
            checked_backward, checked_names, checked_validator = bind_checked_backward(
                jax,jnp,component,recurrent_loss,runtime,head,scales,model,
                separate_validation=True)
            if checked_validator is None:
                raise ValueError('dual-device profiling requires separate checked validation')
            result = profile_dual_device(
                jax=jax,jnp=jnp,parameters=parameters,optimizer=optimizer,
                sample=sample,daily_rows=fixed_rows,
                recurrent_rows={h:schedule['update_rows'][0][f'T{h}'] for h in (7,30)},
                window_pool=train_pool,daily_backward=daily_backward,
                checked_backward=checked_backward,checked_validator=checked_validator,
                checked_counter_names=checked_names,coefficients=coefficients,
                context=context,update=update,repeats=args.profile_repeats,emit=emit,
                output=args.output,compiled_memory_limit_gib=config['compiled_memory_limit_gib'])
            emit('completed',status='completed',dual_device_profile=result,
                 persistent_parameter_updates=0)
            return report

        def save(index=None):
            # Persist all update records before the atomic checkpoint replacement.
            # Recovery may trim a report ahead of an old checkpoint, never infer
            # missing records for a newer checkpoint after an abrupt interruption.
            emit('checkpoint_saving')
            began = time.perf_counter()
            count = report['completed_updates']
            path = checkpoint_path(args.output,index)
            primary_helper(save_mixed_training_checkpoint,path,jax=jax,
                parameters=parameters,optimizer_state=optimizer,
                completed_updates=count,daily_sampler_cursor={'epoch':0,'batch_offset':count},
                recurrent_schedule_cursor=(count if learning_schedule is None
                                           else learning_schedule.recurrent_cursor(count)),
                method=method,maximum_updates=training_total,
                source_identities=identities,schedule_identity=identities,
                fresh_initialization_identity=fresh_identity,replace=index is None)
            record = dict(update=count,path=str(path),bytes=path.stat().st_size,
                          seconds=time.perf_counter()-began)
            report['checkpoints'] = [r for r in report['checkpoints'] if r['path'] != str(path)] + [record]
            emit('checkpoint_saved')

        def _daily_panel(rows):
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

        def daily_panel(rows):
            return primary_helper(_daily_panel, rows)

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
                result = primary_helper(
                    monitor, parameters,pool,rows,apply_science_gates=gates,
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
        elif report['completed_updates'] in training_milestones:
            monitor_milestone(report['completed_updates'])

        checked_lowerings = {}
        reference_report = None
        precision_reference_report = None
        precision_reference_paths = {}
        if args.checked_reference_report:
            reference_report = json.loads(args.checked_reference_report.read_text())
            if (reference_report.get('configuration') != config
                    or reference_report.get('model_protocol') != model
                    or reference_report.get('profile_source_checkpoint') != str(args.profile_checkpoint)
                    or reference_report.get('profile',{}).get('status') != 'completed'):
                raise ValueError('saved original profile does not match this diagnostic')
            report['original_reference_report'] = str(args.checked_reference_report)
        if args.precision_reference_report:
            precision_reference_report = json.loads(args.precision_reference_report.read_text())
            if (precision_reference_report.get('configuration') != config
                    or precision_reference_report.get('model_protocol') != model
                    or precision_reference_report.get('profile_source_checkpoint') != str(args.profile_checkpoint)
                    or precision_reference_report.get('status') != 'completed'):
                raise ValueError('precision reference does not match this fixed checkpoint')
            report['precision_reference_report'] = str(args.precision_reference_report)
        if precision_stage:
            for name in precision_components:
                path = (args.precision_daily_reference if name == 'daily'
                        and args.precision_daily_reference is not None else None)
                if path is None and precision_reference_report is not None:
                    path = precision_reference_path(
                        args.precision_reference_report, precision_reference_report, name
                    )
                if precision_candidate and (path is None or not path.is_file()):
                    raise FileNotFoundError('immutable legacy precision reference is unavailable for ' + name)
                if path is not None and not path.is_file():
                    raise FileNotFoundError('precision reference gradient is unavailable for ' + name)
                precision_reference_paths[name] = path
        if args.compare_checked_recurrent or args.checked_recurrent:
            from scripts.hpc.checked_recurrent_execution import bind_checked_backward, wrap_separate_checked

            checked_backward, checked_names, checked_validator = bind_checked_backward(
                jax,jnp,component,recurrent_loss,runtime,head,scales,model,
                separate_validation=args.separate_checked_validation)
            report['checked_validation_placement'] = ('separate_device_program'
                if checked_validator is not None else 'inline_aux')
            if not args.dual_device_training:
                for horizon, case in cases.items():
                    emit(f'checked_lower_T{horizon}')
                    checked_arguments = (parameters,*case[:4],context)
                    if checked_validator is None:
                        checked_arguments += (case[4],)
                    checked_lowerings[horizon] = checked_backward.lower(*checked_arguments)
                del checked_arguments

        executables, lowered = {}, None
        if args.dual_device_training:
            with (jax.default_device(recurrent_device),
                  device_helper_cache(recurrent_device)):
                peer_parameters = jax.device_put(parameters, recurrent_device)
                recurrent_context = jax.device_put(context, recurrent_device)
                jax.block_until_ready((peer_parameters, recurrent_context))
            compile_specs = [
                ('daily', daily_backward,
                 (parameters, first_daily, coefficients), primary_device),
                ('T7', checked_backward,
                 (peer_parameters, *cases[7][:4], recurrent_context), recurrent_device),
                ('T30', checked_backward,
                 (peer_parameters, *cases[30][:4], recurrent_context), recurrent_device),
            ]
            for name, fn, arguments, compile_device in compile_specs:
                emit('compile_'+name)
                began = time.perf_counter()
                # Raw AOT programs retain the admitted cache identities. Only
                # smaller device helpers are namespaced by device_helper_cache.
                with jax.default_device(compile_device):
                    lowered = fn.lower(*arguments)
                    executable = lowered.compile()
                memory = _compiled_memory(executable)
                report['compilation'][name] = dict(
                    seconds=time.perf_counter()-began, memory=memory,
                    device=str(compile_device))
                if memory['estimated_live_bytes'] > config['compiled_memory_limit_gib']*1024**3:
                    raise MemoryError(name+' compiled live memory exceeds bound')
                if name != 'daily':
                    with (jax.default_device(recurrent_device),
                          device_helper_cache(recurrent_device)):
                        executable = wrap_separate_checked(
                            jax, executable, checked_validator, device=recurrent_device)
                executables[name] = executable
                emit('compiled_'+name)
            del peer_parameters, recurrent_context, compile_specs
        else:
            for name,fn,arguments in [('T7',backward,(parameters,*cases[7][:4],context)),
                                      ('T30',backward,(parameters,*cases[30][:4],context)),
                                      ('daily',daily_backward,(parameters,first_daily,coefficients))]:
                if precision_stage and name not in precision_components:
                    continue
                if reference_report is not None:
                    # Compile missing references only when their horizon is reached.
                    continue
                emit('compile_'+name)
                began = time.perf_counter()
                lowered = (checked_lowerings.pop(int(name[1:]))
                           if args.checked_recurrent and name != 'daily' else fn.lower(*arguments))
                executable = lowered.compile()
                memory = _compiled_memory(executable)
                report['compilation'][name] = dict(seconds=time.perf_counter()-began,memory=memory)
                if profiling and not args.learning_budget_profile:
                    report['compilation'][name]['precision'] = lowered_precision_summary(lowered)
                if memory['estimated_live_bytes'] > config['compiled_memory_limit_gib']*1024**3:
                    raise MemoryError(name+' compiled live memory exceeds bound')
                if args.checked_recurrent and name != 'daily' and checked_validator is not None:
                    executable = wrap_separate_checked(jax,executable,checked_validator)
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

            if args.learning_budget_profile:
                from scripts.hpc.mixed_learning_profiling import profile_learning_budget

                budget_profile = profile_learning_budget(
                    jax=jax,jnp=jnp,parameters=parameters,optimizer=optimizer,
                    sample=sample,daily_rows=fixed_rows,
                    recurrent_rows={row['length']:row for row in fixed_windows},
                    window_pool=train_pool,daily_executable=executables['daily'],
                    recurrent_executables={7:executables['T7'],30:executables['T30']},
                    checked_counter_names=checked_names,coefficients=coefficients,
                    context=context,update=update,add=add,finish=finish,
                    combine=combine,finite=finite,norm_fn=norm_fn,
                    repeats=args.profile_repeats,emit=emit)
                emit('completed',status='completed',learning_budget_profile=budget_profile,
                     persistent_parameter_updates=0,
                     memory=_device_memory(runtime['devices'][0]),
                     daily_reader=daily_pool.statistics())
                return report

            if precision_stage:
                from research.daily_coarse_graining.daily_operator.one_day_training_state import _atomic_pickle
                from scripts.hpc.probe_public_rollout import gradient_comparison

                comparisons = {}
                for name in precision_components:
                    if name == 'daily':
                        identity = precision_reference_identity(
                            name, checkpoint=args.profile_checkpoint, rows=fixed_rows
                        )
                        daily_total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
                        daily_loss, daily_hard = 0., {}
                        daily_auxiliary_rows = []
                        began = time.perf_counter()
                        for start in range(0, 64, 8):
                            (value, auxiliary), gradient = executables[name](
                                parameters, sample(fixed_rows[start:start + 8]), coefficients
                            )
                            hard = calibration._hard_summary(
                                jax.device_get(auxiliary['hard']), batch_size=8
                            )
                            if not np.isfinite(float(value)) or not hard['all_zero']:
                                raise ValueError('daily precision stage failed physical/finite gates')
                            daily_auxiliary_rows.append(
                                masked_precision_daily_auxiliary(jax, auxiliary)
                            )
                            for hard_name, count in hard['totals'].items():
                                daily_hard[hard_name] = daily_hard.get(hard_name, 0) + count
                            daily_total = add(daily_total, gradient, 8)
                            daily_loss += float(value) / 8
                            del value, auxiliary, gradient, hard
                        value, gradient = daily_loss, finish(daily_total, 64)
                        jax.block_until_ready(gradient)
                        hard = daily_hard
                        daily_auxiliary = jax.tree_util.tree_map(
                            lambda *rows: np.concatenate([np.asarray(row) for row in rows], axis=0),
                            *daily_auxiliary_rows,
                        )
                        del daily_total
                    else:
                        row = next(row for row in fixed_windows if row['length'] == int(name[1:]))
                        identity = precision_reference_identity(
                            name, checkpoint=args.profile_checkpoint, window=row
                        )
                        case, hard = valid_recurrent(row)
                        began = time.perf_counter()
                        value, gradient = executables[name](parameters, *case[:4], context)
                        jax.block_until_ready(gradient)
                        value = float(value)
                        del case
                    execution_seconds = time.perf_counter() - began
                    path = precision_reference_paths[name]
                    if path is None and precision_candidate:
                        raise ValueError('FP32 precision stage cannot generate its own legacy reference')
                    if path is None:
                        path = args.output.with_suffix(f'.{name}.precision-reference.pkl')

                    if path.is_file():
                        import pickle

                        with path.open('rb') as handle:
                            saved_reference = pickle.load(handle)
                        saved_identity = saved_reference.get('identity')
                        if saved_identity is None:
                            # Checked recurrent references predate the generic precision cache.
                            if (name == 'daily' or saved_reference.get('checkpoint') != str(args.profile_checkpoint)
                                    or saved_reference.get('window') != identity['window']):
                                raise ValueError('precision reference belongs to a different fixed panel')
                        elif saved_identity != identity:
                            raise ValueError('precision reference identity differs')
                        old_value, reference, old_hard = (
                            saved_reference[key] for key in ('value', 'gradient', 'hard')
                        )
                        reference_auxiliary = saved_reference.get('daily_auxiliary')
                        del saved_reference
                        comparison = gradient_comparison(jax, gradient, reference)
                        if name == 'daily':
                            if reference_auxiliary is None:
                                raise ValueError('daily precision reference lacks auxiliary-term evidence')
                            auxiliary_comparison = gradient_comparison(
                                jax, daily_auxiliary, reference_auxiliary
                            )
                            comparison['daily_auxiliary_comparison'] = auxiliary_comparison
                        comparison.update(
                            loss_equal=bool(np.isclose(value, old_value, atol=1e-6, rtol=1e-5)),
                            old_loss=float(old_value), new_loss=float(value),
                            physical_counts_equal=hard == old_hard,
                            all_hard_zero=not any(hard.values()),
                            reference_gradient=str(path), hard=hard,
                            execution_seconds=execution_seconds,
                        )
                        candidate_path = args.output.with_suffix(f'.{name}.candidate-gradient.pkl')
                        candidate = dict(identity=identity, value=value,
                                         gradient=jax.device_get(gradient), hard=hard)
                        if name == 'daily':
                            candidate['daily_auxiliary'] = daily_auxiliary
                        _atomic_pickle(candidate_path, candidate)
                        comparison['candidate_gradient'] = str(candidate_path)
                        comparisons[name] = comparison
                        report['precision_comparisons'] = comparisons
                        emit('precision_compared_' + name, precision_comparisons=comparisons)
                        if not (comparison['passed'] and comparison['loss_equal']
                                and comparison['physical_counts_equal'] and comparison['all_hard_zero']
                                and (name != 'daily' or comparison['daily_auxiliary_comparison']['passed'])):
                            raise ValueError(name + ' precision loss/gradient/physical comparison failed')
                    else:
                        reference_payload = dict(identity=identity, value=value,
                                                 gradient=jax.device_get(gradient), hard=hard)
                        if name == 'daily':
                            reference_payload['daily_auxiliary'] = daily_auxiliary
                        _atomic_pickle(path, reference_payload)
                        comparison = dict(reference_gradient=str(path), baseline_captured=True,
                                          loss=float(value), hard=hard,
                                          all_hard_zero=not any(hard.values()),
                                          execution_seconds=execution_seconds)
                        if not comparison['all_hard_zero']:
                            raise ValueError(name + ' legacy precision reference has hard failures')
                        comparisons[name] = comparison
                        report['precision_comparisons'] = comparisons
                        emit('precision_compared_' + name, precision_comparisons=comparisons)
                    del gradient

                if precision_stage != 'full':
                    emit('completed', status='completed', persistent_parameter_updates=0,
                         memory=_device_memory(runtime['devices'][0]),
                         daily_reader=daily_pool.statistics())
                    return report

                if precision_reference_report is None:
                    raise ValueError('full precision timing requires the immutable legacy profile report')
                profile_arguments = dict(jax=jax,jnp=jnp,parameters=parameters,optimizer=optimizer,
                    sample=sample,daily_rows=fixed_rows,
                    recurrent_rows={row['length']:row for row in fixed_windows},window_pool=train_pool,
                    daily_executable=executables['daily'],
                    recurrent_executables={7:executables['T7'],30:executables['T30']},
                    coefficients=coefficients,context=context,forward=forward,
                    validate_recurrent=validate_recurrent,update=update,add=add,finish=finish,
                    combine=combine,finite=finite,norm_fn=norm_fn,repeats=args.profile_repeats,
                    trace_dir=None,emit=emit)
                profile = profile_components(**profile_arguments)
                legacy_profile = precision_reference_report.get('profile')
                if not isinstance(legacy_profile, dict) or legacy_profile.get('status') != 'completed':
                    raise ValueError('full precision timing reference lacks a completed legacy profile')
                parity = _parity(legacy_profile['baseline']['records'], profile['baseline']['records'])
                if not parity['passed']:
                    raise ValueError('full precision update differs from immutable legacy profile')
                before = legacy_profile['baseline']['median_seconds']
                after = profile['baseline']['median_seconds']
                emit('precision_full_profile_completed', profile=profile,
                     precision_full_parity=parity, legacy_baseline_seconds=before,
                     candidate_baseline_seconds=after, speedup=before / after,
                     time_reduction=1. - after / before,
                     same_process_timing_comparison=False,
                     timing_scope='legacy_original_recurrent_execution_not_checked_split')
                emit('completed', status='completed', profile=profile,
                     persistent_parameter_updates=0, memory=_device_memory(runtime['devices'][0]),
                     daily_reader=daily_pool.statistics())
                return report

            profile_arguments = dict(jax=jax,jnp=jnp,parameters=parameters,optimizer=optimizer,
                sample=sample,daily_rows=fixed_rows,
                recurrent_rows={row['length']:row for row in fixed_windows},window_pool=train_pool,
                daily_executable=executables.get('daily'),
                recurrent_executables={7:executables.get('T7'),30:executables.get('T30')},
                coefficients=coefficients,context=context,forward=forward,
                validate_recurrent=validate_recurrent,update=update,add=add,finish=finish,
                combine=combine,finite=finite,norm_fn=norm_fn,repeats=args.profile_repeats,
                trace_dir=args.output.with_suffix('.trace') if args.profile_trace else None,emit=emit)
            profile = (reference_report['profile'] if reference_report is not None else
                       profile_components(**profile_arguments))
            emit('original_profile_completed',profile=profile)
            if args.compare_checked_recurrent:
                from research.daily_coarse_graining.daily_operator.one_day_training_state import _atomic_pickle
                from scripts.hpc.checked_recurrent_execution import decode_counts
                from scripts.hpc.probe_public_rollout import gradient_comparison

                checked_executables, comparisons = {}, {}
                for row in fixed_windows:
                    horizon, key = row['length'], f"T{row['length']}"
                    path = (saved_gradient_path(args.checked_reference_report,reference_report,key)
                            if reference_report is not None else args.output.with_suffix(f'.{key}.reference.pkl'))
                    if reference_report is not None and path.is_file():
                        import pickle

                        with path.open('rb') as handle:
                            saved_reference = pickle.load(handle)
                        if (saved_reference['checkpoint'] != str(args.profile_checkpoint)
                                or saved_reference['window'] != row):
                            raise ValueError('saved gradient belongs to a different checkpoint/window')
                        old_value, reference, old_hard = (saved_reference[key] for key in ('value','gradient','hard'))
                        del saved_reference
                        case = train_pool.case(jax,row)
                    else:
                        case, old_hard = valid_recurrent(row)
                        if key not in executables:
                            emit('compile_reference_'+key)
                            began = time.perf_counter()
                            original = backward.lower(parameters,*case[:4],context).compile()
                            memory = _compiled_memory(original)
                            report['compilation'][key] = dict(seconds=time.perf_counter()-began,memory=memory)
                            if memory['estimated_live_bytes'] > config['compiled_memory_limit_gib']*1024**3:
                                raise MemoryError(key+' reference compiled live memory exceeds bound')
                            executables[key] = original
                            emit('compiled_reference_'+key)
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
                    emit('checked_compiled_'+key)
                    if checked_validator is not None:
                        executable = wrap_separate_checked(jax,executable,checked_validator,
                            emit=lambda stage, key=key: emit('checked_'+stage+'_'+key))
                    began = time.perf_counter()
                    (new_value, counts), gradient = executable(parameters,*case[:4],context,case[4])
                    jax.block_until_ready(gradient)
                    new_value, counts = jax.device_get((new_value,counts))
                    report['compilation']['checked_'+key]['first_call_seconds'] = time.perf_counter()-began
                    if checked_validator is not None:
                        report['compilation']['checked_'+key]['validation'] = dict(executable.statistics)
                    emit('checked_executed_'+key)
                    hard = decode_counts(checked_names,counts)
                    began = time.perf_counter()
                    comparison = gradient_comparison(jax,gradient,reference)
                    comparison['comparison_seconds'] = time.perf_counter()-began
                    comparison.update(loss_equal=bool(np.isclose(new_value,old_value,atol=1e-6,rtol=1e-5)),
                        old_loss=float(old_value),new_loss=float(new_value),
                        physical_counts_equal=hard==old_hard,all_hard_zero=not any(hard.values()),
                        reference_gradient=str(path),hard=hard)
                    candidate_path = args.output.with_suffix(f'.{key}.checked-gradient.pkl')
                    _atomic_pickle(candidate_path,dict(value=new_value,
                        gradient=jax.device_get(gradient),hard=hard,
                        checkpoint=str(args.profile_checkpoint),window=row,
                        execution_policy=execution_policy))
                    comparison['candidate_gradient'] = str(candidate_path)
                    comparisons[key] = comparison
                    emit('checked_compared_'+key,checked_comparisons=comparisons)
                    if not (comparison['passed'] and comparison['loss_equal']
                            and comparison['physical_counts_equal'] and comparison['all_hard_zero']):
                        raise ValueError('checked '+key+' loss/gradient/physical comparison failed')
                    checked_executables[horizon] = executable
                    del reference,gradient,case
                if profile_arguments['daily_executable'] is None:
                    emit('compile_daily_for_checked_replay')
                    began = time.perf_counter()
                    executable = daily_backward.lower(parameters,sample(fixed_rows[:8]),coefficients).compile()
                    memory = _compiled_memory(executable)
                    report['compilation']['daily'] = dict(seconds=time.perf_counter()-began,memory=memory)
                    if memory['estimated_live_bytes'] > config['compiled_memory_limit_gib']*1024**3:
                        raise MemoryError('daily compiled live memory exceeds bound')
                    profile_arguments['daily_executable'] = executable
                checked_profile = profile_components(**profile_arguments,
                    checked_recurrent_executables=checked_executables,checked_counter_names=checked_names)
                parity = _parity(profile['baseline']['records'],checked_profile['baseline']['records'])
                if not parity['passed']:
                    raise ValueError('checked complete mixed update differs from original')
                before, after = profile['baseline']['median_seconds'], checked_profile['baseline']['median_seconds']
                emit('checked_admission_completed',checked_profile=checked_profile,
                     checked_mixed_parity=parity,checked_speedup=before/after,
                     checked_time_reduction=1.-after/before,
                     checked_speed_admission_passed=bool(1.-after/before >= .05),
                     same_process_timing_comparison=reference_report is None)
                if execution_policy is not None and reference_report is not None:
                    prior_checked = reference_report.get('checked_profile',{}).get('baseline',{}).get('median_seconds')
                    if prior_checked is not None:
                        emit('combined_execution_completed',
                             legacy_checked_baseline_seconds=prior_checked,
                             combined_baseline_seconds=after,
                             incremental_time_reduction=1.-after/prior_checked)
            emit('completed',status='completed',profile=profile,persistent_parameter_updates=0,
                 memory=_device_memory(runtime['devices'][0]),daily_reader=daily_pool.statistics())
            return report

        if args.dual_device_training:
            from scripts.hpc.mixed_learning_dual_device import ContinuousDualDeviceTrainer

            def dual_daily_branch(local_parameters, batch):
                branch_started = time.perf_counter()
                total = jax.tree_util.tree_map(jnp.zeros_like, local_parameters)
                branch_loss, totals = 0., {}
                for start in range(0, 64, 8):
                    microbatch = jax.device_put(
                        sample(batch.rows[start:start+8]), primary_device)
                    (value, auxiliary), gradient = executables['daily'](
                        local_parameters, microbatch, coefficients)
                    hard = calibration._hard_summary(
                        jax.device_get(auxiliary['hard']), batch_size=8)
                    if not np.isfinite(float(value)) or not hard['all_zero']:
                        report['failed_training_daily'] = dict(
                            update=report['completed_updates']+1,
                            microbatch=start//8, hard=hard)
                        raise ValueError('daily training physical/finite gate failed')
                    for key, count in hard['totals'].items():
                        totals[key] = totals.get(key, 0)+count
                    total = add(total, gradient, 8)
                    branch_loss += float(value)/8
                gradient = finish(total, 64)
                return dict(gradient=gradient, loss=branch_loss,
                            norm=float(norm_fn(gradient)), hard=totals,
                            seconds=time.perf_counter()-branch_started)

            def dual_recurrent_branch(local_parameters, rows):
                branch_started = time.perf_counter()
                results = {}
                local_context = jax.device_put(context, recurrent_device)
                from scripts.hpc.checked_recurrent_execution import decode_counts

                for horizon in (7, 30):
                    started = time.perf_counter()
                    row = rows[horizon]
                    case = jax.device_put(train_pool.case(jax, row), recurrent_device)
                    (value, counts), gradient = executables[f'T{horizon}'](
                        local_parameters, *case[:4], local_context, case[4])
                    value, counts = jax.device_get((value, counts))
                    hard = decode_counts(checked_names, counts)
                    if any(hard.values()) or not np.isfinite(value):
                        report['failed_training_window'] = dict(
                            window=row, hard=hard, loss=float(value))
                        raise ValueError(
                            'checked TRAIN trajectory physical/finite gate failed')
                    results[horizon] = dict(
                        gradient=gradient, loss=float(value), hard=hard,
                        norm=float(norm_fn(gradient)),
                        seconds=time.perf_counter()-started)
                return results, time.perf_counter()-branch_started

            dual_trainer = ContinuousDualDeviceTrainer(
                jax=jax, jnp=jnp, parameters=parameters, optimizer=optimizer,
                daily_branch=dual_daily_branch,
                recurrent_branch=dual_recurrent_branch,
                combine=combine, update=update, finite=finite,
                devices=dual_devices)
            report['device_execution_runtime'] = dict(
                primary=str(primary_device), recurrent=str(recurrent_device),
                peer_initialized_from_update=report['completed_updates'])
            emit('dual_device_training_ready')

        for index in range(report['completed_updates'],execution_limit):
            emit('training',next_update=index+1)
            began = time.perf_counter()
            phase = (None if learning_schedule is None
                     else learning_schedule.phase_for_update(index+1))
            objective_name = 'mixed' if phase is None else phase.objective
            phase_weights = config['outer_weights'] if phase is None else phase.outer_weights
            batch = batches[index]
            recurrent_index = None
            if objective_name == 'mixed':
                recurrent_index = (index if learning_schedule is None
                                   else learning_schedule.recurrent_cursor(index))
            recurrent_rows = (None if recurrent_index is None else {
                horizon: schedule['update_rows'][recurrent_index][f'T{horizon}']
                for horizon in (7, 30)})
            if dual_trainer is not None:
                dual_record, dual_values = dual_trainer.step(
                    daily_payload=batch,
                    recurrent_payload=recurrent_rows,
                    weights=phase_weights,
                    expected_optimizer_step=index+1,
                )
                parameters, optimizer = (
                    dual_trainer.parameters, dual_trainer.optimizer)
                daily_loss = dual_record['losses']['daily']
                losses = {key: value for key, value in dual_record['losses'].items()
                          if key != 'daily'}
                component_norms = dual_record['gradient_norms']
                hard_totals = dual_record['hard']['daily']
                recurrent_hard = {key: value for key, value in dual_record['hard'].items()
                                  if key != 'daily'}
                norm, clip = (dual_record['combined_gradient_norm'],
                              dual_record['clip'])
                del dual_values
            else:
                daily_total = jax.tree_util.tree_map(jnp.zeros_like,parameters)
                daily_loss, hard_totals = 0., {}
                for i in range(0,64,8):
                    (value,aux), gradient = executables['daily'](
                        parameters,sample(batch.rows[i:i+8]),coefficients)
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
                if recurrent_rows is not None:
                    for h in (7,30):
                        key = f'T{h}'
                        row = recurrent_rows[h]
                        if args.checked_recurrent:
                            from scripts.hpc.checked_recurrent_execution import decode_counts

                            case = train_pool.case(jax,row)
                            (value,counts),gradient = executables[key](
                                parameters,*case[:4],context,case[4])
                            value,counts = jax.device_get((value,counts))
                            hard = decode_counts(checked_names,counts)
                            if any(hard.values()) or not np.isfinite(value):
                                report['failed_training_window'] = dict(
                                    window=row,hard=hard,loss=float(value))
                                raise ValueError(
                                    'checked TRAIN trajectory physical/finite gate failed')
                        else:
                            case,hard = valid_recurrent(row)
                            value,gradient = executables[key](parameters,*case[:4],context)
                        losses[key], recurrent_hard[key] = float(value),hard
                        component_norms[key] = float(norm_fn(gradient))
                        mixed = combine(mixed,gradient,1.,phase_weights[key])
                        jax.block_until_ready(mixed)
                        del gradient,case
                if (not all(np.isfinite(v) and v>0 for v in component_norms.values())
                        or not all(np.isfinite(v) for v in losses.values())
                        or not bool(finite(mixed))):
                    raise ValueError('component/mixed gradient is nonfinite or empty')
                proposed,state,norm,clip = update(parameters,mixed,optimizer)
                jax.block_until_ready((proposed,state,norm,clip))
                if (not bool(finite((proposed,state))) or not np.isfinite(float(norm))
                        or not 0<float(clip)<=1 or int(state.step)!=index+1):
                    raise ValueError('complete AdamW proposal invalid')
                parameters,optimizer = proposed,state
                del mixed,proposed,state
            report['completed_updates'] = index+1
            update_seconds = time.perf_counter()-began
            mixed_loss = phase_weights['daily']*daily_loss + sum(
                phase_weights[key]*value for key,value in losses.items())
            record = dict(update=index+1,daily_loss=daily_loss,**losses,
                mixed_loss=mixed_loss,gradient_norms=component_norms,norm=float(norm),
                clip=float(clip),daily_hard=hard_totals,recurrent_hard=recurrent_hard,
                seconds=update_seconds,daily_batch_index=index,
                recurrent_schedule_index=recurrent_index)
            if args.integration_updates:
                record['integration_data'] = dict(
                    daily_rows=admission._rows(batch.rows),
                    recurrent_rows={} if recurrent_rows is None else {
                        f'T{horizon}': recurrent_rows[horizon] for horizon in (7,30)})
            if dual_trainer is not None:
                record['device_timing'] = {
                    key: dual_record[key] for key in (
                        'daily_branch_seconds','recurrent_branch_seconds',
                        'branch_elapsed_seconds','merge_checks_optimizer_seconds',
                        'weight_synchronize_seconds','synchronized_parameter_bytes',
                        'peer_synchronized_optimizer_step')}
            if learning_schedule is not None:
                record.update(
                    phase=phase.name,objective=objective_name,daily_batch_index=index,
                    recurrent_schedule_index=recurrent_index,
                    recurrent_schedule_cursor_after=learning_schedule.recurrent_cursor(index+1),
                    training_cumulative_seconds=sum(
                        row.get('seconds',0.) for row in report['updates'])+update_seconds,
                )
            report['updates'].append(record)
            emit('optimizer_completed',memory=execution_memory(),
                  daily_reader=daily_pool.statistics())
            if index+1 in training_milestones:
                save(index+1)
                monitor_milestone(index+1)
            elif (index+1)%config['rolling_recovery_interval']==0 or index+1==execution_limit:
                save()
            if args.integration_updates and index+1==execution_limit:
                monitor_milestone(index+1)
        if dual_trainer is not None:
            dual_trainer.close()
            dual_trainer = None
        # Replay persisted weights through the actual daily forward entry point.
        final_path = checkpoint_path(
            args.output,execution_limit if execution_limit in training_milestones else None)
        if args.dual_device_training:
            with (jax.default_device(primary_device),
                  device_helper_cache(primary_device)):
                replay_batch = jax.device_put(sample(fixed_rows[:8]), primary_device)
                expected = float(daily_forward(parameters,replay_batch,coefficients)[0])
        else:
            expected = float(daily_forward(parameters,sample(fixed_rows[:8]),coefficients)[0])
        saved = load_mixed_training_checkpoint(
            final_path,maximum_updates=training_total,**recovery_identity)
        if args.dual_device_training:
            with (jax.default_device(primary_device),
                  device_helper_cache(primary_device)):
                restored = jax.device_put(saved['parameters'], primary_device)
                actual = float(daily_forward(restored,replay_batch,coefficients)[0])
        else:
            restored = jax.device_put(saved['parameters'])
            actual = float(daily_forward(restored,sample(fixed_rows[:8]),coefficients)[0])
        if actual != expected:
            raise ValueError('persisted checkpoint forward replay differs')
        del restored,saved
        stopped = execution_limit < limit
        emit('stopped' if stopped else 'completed',
             status='stopped_for_recovery' if stopped else 'completed',classification=scope,
             checkpoint_forward_replay=True,scientific_acceptance=False,
             memory=execution_memory())
        return report
    except Exception as error:
        if dual_trainer is not None:
            dual_trainer.close()
        emit('failed',status='resource_limited' if isinstance(error,MemoryError) else 'failed',
             error=f'{type(error).__name__}: {error}')
        raise


def parser():
    result = prototype.parser()
    for name in ('mixed-protocol','catalogue-report','recurrent-schedule','selection'):
        result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--integration-updates',type=int,choices=(1,2,3))
    result.add_argument('--stop-after-update',type=int,
                        help='integration-only recoverable early stop; not part of method identity')
    result.add_argument('--resume',type=Path)
    result.add_argument('--report-interval',type=int,default=25,
                        help='full report every N updates plus all checkpoints/monitors; '
                             'compact progress every event; 1 keeps eager reports')
    result.add_argument('--profile-checkpoint',type=Path)
    result.add_argument('--profile-training-report',type=Path)
    result.add_argument('--profile-repeats',type=int,choices=(3,5),default=3)
    result.add_argument('--profile-trace',action='store_true')
    result.add_argument('--compare-checked-recurrent',action='store_true')
    result.add_argument('--checked-recurrent',action='store_true')
    result.add_argument('--separate-checked-validation',action='store_true')
    result.add_argument('--checked-reference-report',type=Path)
    result.add_argument('--execution-policy',type=Path,
                        help='explicit neural precision and recurrent execution identity')
    result.add_argument('--learning-schedule',type=Path,
                        help='opt-in declared daily/mixed phase schedule')
    result.add_argument('--dual-device-profile',action='store_true',
                        help='fixed-checkpoint daily/recurrent two-device profile; never trains')
    result.add_argument('--dual-device-training',action='store_true',
                        help='continuous updates with daily/AdamW on GPU0 and recurrent on GPU1')
    result.add_argument('--learning-budget-profile',action='store_true',
                        help='fixed-checkpoint daily-only versus mixed warm timing; never trains')
    result.add_argument('--precision-stage', choices=_PRECISION_STAGES,
                        help='fixed-checkpoint legacy/FP32 component admission; never trains')
    result.add_argument('--precision-reference-report', type=Path,
                        help='immutable legacy precision-stage/profile result to reuse')
    result.add_argument('--precision-daily-reference', type=Path,
                        help='immutable legacy daily64 full-gradient cache')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
