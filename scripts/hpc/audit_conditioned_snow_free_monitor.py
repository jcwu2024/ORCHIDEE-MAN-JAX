"""Replay the admitted proposal once, then evaluate the fixed development monitor."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from scripts.hpc import audit_conditioned_attention_streamed_integration as integration
from scripts.hpc import run_target_oriented_coupled_gpp_direct_recalibration as recal

persistent, window = integration.persistent, integration.window
REFERENCE = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/conditioned-snow-free-integration-90f6403.json')
THIRTY = 'thirty_day_component_resolved_science'
DIRECT = 'direct_gpp_208_teacher_days'


def close(actual, expected):
    return bool(np.isclose(actual, expected, atol=1e-6, rtol=1e-5))


def active_sources(proposal):
    registry = proposal['direction']['source_registry']
    weights = proposal['direction']['selected_minimum_norm_direction']['normalized_simplex_weights']
    if set(registry) != set(weights) or not all(np.isfinite(v) and v >= 0 for v in weights.values()):
        raise ValueError('invalid recorded simplex')
    return [(name, weights[name]) for name in registry if weights[name] != 0.]


def combine_recorded(jax, jnp, values, proposal, units):
    """Preserve the original matrix shape/order and dtype-rounded update operations."""
    from jax.flatten_util import ravel_pytree
    registry = proposal['direction']['source_registry']
    weights = proposal['direction']['selected_minimum_norm_direction']['normalized_simplex_weights']
    if set(units) != {name for name, _ in active_sources(proposal)}:
        raise ValueError('replay must include every nonzero source exactly once')
    flat, unravel = ravel_pytree(values)
    zero = jnp.zeros_like(next(iter(units.values())))
    matrix = jnp.stack([units.get(name, zero) for name in registry])
    direction, norm, nonfinite = persistent._make_flat_direction(jax)(
        matrix, jnp.asarray([weights[name] for name in registry], dtype=jnp.float64))
    jax.block_until_ready((direction, norm, nonfinite))
    if int(nonfinite) or not close(float(norm), proposal['direction']['combined_gradient_norm']):
        raise RuntimeError('replayed combined direction norm differs from admitted proposal')
    if int(direction.size) != int(flat.size):
        raise ValueError('replayed direction does not fit parent parameters')
    proposed, nonfinite = persistent._make_dtype_preserving_direct_virtual_step(jax)(
        values, unravel(direction), norm,
        np.asarray(proposal['direction']['calibration']['full_parameter_delta_l2'], dtype=np.float64))
    if int(nonfinite):
        raise RuntimeError('nonfinite replayed candidate')
    candidate = persistent._scaled_parameters(jax, values, proposed, proposal['selection']['selected_scale'])
    jax.block_until_ready(candidate)
    return candidate, float(norm)


def reconstruct(args, bootstrap, values, proposal, *, emit):
    context = bootstrap['context']
    jax, jnp = context['runtime']['jax'], context['runtime']['jnp']
    active = active_sources(proposal)
    direct_names = [name for name, _ in active if name.startswith(DIRECT + '.artifact_')]
    if {name for name, _ in active} != set(direct_names) | {THIRTY}:
        raise ValueError('this replay supports the admitted direct plus thirty-day direction')
    flatten = persistent._make_flattened_unit_gradient(jax)
    measured = persistent._measure_direct_artifact_gradients(
        args, context, trainable=values, direct_steps=bootstrap['direct_steps'],
        updates=tuple(int(name.rsplit('_', 1)[1]) for name in direct_names),
        flatten_gradient=flatten, label='corrected_monitor_replay', expected_parameter_count=window.TRAINABLE_PARAMETER_COUNT)
    units = dict(zip(direct_names, measured['unit_gradients'], strict=True))
    records = []
    for name, gradient, loss in zip(direct_names, measured['gradient_reports'], measured['panel']['artifact_losses'], strict=True):
        records.append({'source': name, 'loss': loss['loss'], 'norm': gradient['l2_norm']})
    callbacks = integration.StreamedHorizonCallbacks(bootstrap, callback=lambda e, r: emit(e, {'progress': r}), compact_trace=True)
    attached = callbacks.attach()
    parameters = window.replace_conditioned_coupled_gpp_trainable(bootstrap['fixed_parameters'], values)
    summed = recal._zero_gradient(jax, jnp, values)
    losses, hard_counts = [], []
    group = proposal['train_only_panel']['artifact_groups'][0]
    for update in group:
        artifact = window._load_artifact(args, context, update)
        result, hard, gradient, _ = recal._horizon_gradient(
            jax, jnp, parameters, values, artifact.batch, bootstrap['steps'], attached['gradient_steps'],
            horizon=30, flux_basis=bootstrap['flux_basis'])
        summed = recal._add_gradient(jax, jnp, summed, gradient)
        losses.append(float(result.total))
        hard_counts.append(window._hard_failure_count(window._hard_host(hard)))
        emit('replay_artifact', {'replay_artifact': update, 'horizon_records': callbacks.records})
    gradient = recal._mean_gradient(jax, summed, len(group))
    unit, norm, nonfinite = flatten(gradient)
    if int(nonfinite):
        raise RuntimeError('nonfinite reconstructed thirty-day source')
    units[THIRTY] = unit
    records.append({'source': THIRTY, 'loss': float(np.mean(losses)), 'norm': float(norm)})
    for row in records:
        name = row['source']
        row['loss_agrees'] = close(row['loss'], proposal['direction']['constraint_losses'][name])
        row['norm_agrees'] = close(row['norm'], proposal['direction']['source_gradients'][name]['l2_norm'])
    passed = (all(r['loss_agrees'] and r['norm_agrees'] for r in records)
              and len(callbacks.records) == 32 and all(r['linearized_score_agrees'] for r in callbacks.records)
              and not any(hard_counts))
    replay = {'sources': records, 'horizon_records': callbacks.records, 'passed': passed,
              'reverse_compile_count': callbacks.reverse.compile_count, 'reverse_memory': callbacks.reverse.memory}
    emit('replay_sources_completed', {'reconstruction': replay})
    if not passed:
        return None, replay
    candidate, norm = combine_recorded(jax, jnp, values, proposal, units)
    replay['combined_norm'] = norm
    return candidate, replay


def verify_candidate(actual, expected, direct_groups, expected_groups):
    """Compare actual science, source losses and valid counts before monitor access."""
    checks = {'physical_hard_gate': actual['hard_gate_passed'] is True}
    for name, value in expected['source_mean_losses'].items():
        checks['source.' + name] = close(actual['source_mean_losses'][name], value)
    for horizon, summary in expected['acceptance'].items():
        for name, row in summary['keys'].items():
            current = actual['acceptance'][horizon]['keys'][name]
            for metric in ('normalized_rmse', 'normalized_signed_bias', 'evaluated_values'):
                checks[f'{horizon}.{name}.{metric}'] = (current[metric] == row[metric] if metric == 'evaluated_values'
                                                       else close(current[metric], row[metric]))
    expected_losses = {r['artifact_update']: r['after'] for g in expected_groups for r in g['artifact_changes']}
    actual_losses = {r['artifact_update']: r['loss'] for g in direct_groups for r in g['artifact_losses']}
    checks['direct_artifacts'] = set(expected_losses) == set(actual_losses)
    for update, loss in expected_losses.items():
        checks[f'direct.{update}'] = update in actual_losses and close(actual_losses[update], loss)
    return {'passed': all(checks.values()), 'checks': checks}


def paired_monitor(args, bootstrap, contract, baseline, candidate, *, emit):
    before = persistent._monitor(args, bootstrap, contract, trainable=baseline, label='corrected_monitor_baseline')
    emit('baseline_monitor_completed', {'monitor_before': before})
    after = persistent._monitor(args, bootstrap, contract, trainable=candidate, label='corrected_monitor_candidate')
    gate = persistent.monitor_transfer_gate(contract, before, after)
    return {'monitor_before': before, 'monitor_after': after, 'monitor_gate': gate,
            'baseline_absolute_science': persistent.absolute_science_gate(contract, before),
            'candidate_absolute_science': persistent.absolute_science_gate(contract, after),
            'passed': gate['passed']}


def run(args):
    started = time.perf_counter()
    report = {'schema_version': 'conditioned_snow_free_monitor_v1', 'status': 'initializing',
              'policy': {'warm_snow_free_boundary': True, 'accepted_recovery_updated': False,
                         'monitor_selects_candidate': False, 'independent_validation': False}}

    def emit(stage, updates):
        report.update(updates)
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        window._atomic_json(args.output, report)
        print(f'corrected_monitor stage={stage} elapsed={report["elapsed_seconds"]:.1f}s', flush=True)

    print('corrected_monitor stage=bootstrap', flush=True)
    try:
        reference = json.loads(args.integration_report.read_text())
        parent = json.loads((args.conditioned_run_dir / 'report.json').read_text())
        recovered = window.load_recovery(args.conditioned_run_dir / 'recovery.pkl', identity=parent['identity'],
                                         parameter_count=window.TRAINABLE_PARAMETER_COUNT)
        contract = persistent.load_conditioned_attention_persistent_training_contract(args.conditioned_persistent_contract)
        proposal = reference['proposal']
        panel = tuple(tuple(g) for g in proposal['train_only_panel']['artifact_groups'])
        if (reference['schema_version'] != 'conditioned_snow_free_integration_v1' or reference['status'] != 'completed'
                or not reference['admission']['passed'] or not reference['policy']['warm_snow_free_boundary']
                or reference['parent_identity'] != parent['identity'] or reference['logical_update'] != 4
                or recovered is None or recovered['completed_logical_update'] != 3
                or panel != contract.train_panels[contract.logical_updates.index(4)]
                or not set(contract.monitor_updates).isdisjoint(u for g in panel for u in g)):
            raise ValueError('requires admitted corrected update-4 report, its parent and fixed disjoint monitor')
        _, original = window._load_parent_state(args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())['contract_sha256'])
        bootstrap = window._conditioned_bootstrap(args, recovered=original, warm_snow_free_boundary=True,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract))
        jax = bootstrap['context']['runtime']['jax']
        baseline = jax.device_put(recovered['trainable'])
        emit('reconstruct_candidate', {'status': 'running', 'parent_identity': parent['identity'],
            'integration_report': str(args.integration_report), 'monitor_artifacts': list(contract.monitor_updates)})
        candidate, replay = reconstruct(args, bootstrap, baseline, proposal, emit=emit)
        if candidate is None:
            emit('completed', {'status': 'completed', 'passed': False, 'classification': 'proposal_replay_not_admitted'})
            return report
        parameters = window.replace_conditioned_coupled_gpp_trainable(bootstrap['fixed_parameters'], candidate)
        emit('train_replay_forward', {})
        science = persistent._evaluate_detailed_monitor(args, bootstrap['context'], parameters=parameters,
            trainable=candidate, steps=bootstrap['steps'], direct_steps=bootstrap['direct_steps'],
            state_steps=bootstrap['state_steps'], layout=bootstrap['layout'], updates=panel[0],
            flux_basis=bootstrap['flux_basis'], coefficients=bootstrap['coefficients'], label='corrected_candidate_replay')
        groups = persistent._evaluate_direct_groups(args, bootstrap['context'], trainable=candidate,
            direct_steps=bootstrap['direct_steps'], groups=panel, label='corrected_candidate_replay')
        selected = next(r for r in proposal['scale_evaluations'] if r['scale'] == proposal['selection']['selected_scale'])
        verified = verify_candidate(science, selected['train_only_science'], groups, selected['artifact_groups'])
        emit('train_replay_verified', {'candidate_replay': verified, 'train_science': science, 'reconstruction': replay})
        if not verified['passed']:
            emit('completed', {'status': 'completed', 'passed': False, 'classification': 'proposal_replay_not_admitted'})
            return report
        snapshot = args.output.with_suffix('.candidate.pkl')
        payload = {'schema_version': 'corrected_train_selected_candidate_v1', 'parent_identity': parent['identity'],
                   'integration_report': str(args.integration_report), 'warm_snow_free_boundary': True,
                   'parent_logical_update': 3, 'proposed_logical_update': 4, 'promotion_eligible': False,
                   'trainable': jax.device_get(candidate)}
        temporary = snapshot.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(snapshot)
        emit('candidate_snapshot_saved', {'candidate_snapshot': str(snapshot)})
        result = paired_monitor(args, bootstrap, contract, baseline, candidate, emit=emit)
        emit('completed', {**result, 'status': 'completed',
            'classification': 'corrected_monitor_transfer_passed' if result['passed'] else 'corrected_monitor_transfer_failed'})
    except Exception as exc:
        emit(report.get('stage', 'bootstrap'), {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    return report


def _parser():
    parser = window._parser()
    parser.add_argument('--integration-report', type=Path, default=REFERENCE)
    return parser


if __name__ == '__main__':
    run(_parser().parse_args())
