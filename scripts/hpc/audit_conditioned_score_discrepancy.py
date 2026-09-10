"""Localize one stopped update's forward score mismatch without training."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_conditioned_snow_free_continuation as training

window = training.window
integration = training.integration


def failed_case(report):
    records = report['rejected']['gradient_evidence']['records']
    failures = [r for r in records if not r['linearized_score_agrees']]
    if len(failures) != 1 or failures[0]['horizon'] != 30:
        raise ValueError('requires exactly one failed thirty-day score')
    failure = failures[0]
    ordinal = failure['ordinal'] - min(r['ordinal'] for r in records if r['horizon'] == 30)
    group = report['rejected']['record']['baseline']['artifact_groups'][0]['artifact_losses']
    return int(group[ordinal // 4]['artifact_update']), ordinal % 4, failure


def differences(jax, left, right):
    result = []
    left_items, structure = jax.tree_util.tree_flatten_with_path(left)
    right_items, other = jax.tree_util.tree_flatten_with_path(right)
    if structure != other:
        raise ValueError('comparison tree structure differs')
    for (path, a), (_, b) in zip(left_items, right_items, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        finite = np.isfinite(a) & np.isfinite(b)
        delta = np.subtract(a, b, out=np.zeros(a.shape, dtype=float), where=finite)
        result.append({'path': str(path), 'max_absolute': float(np.max(np.abs(delta))),
                       'finite_mask_mismatches': int(np.sum(np.isfinite(a) != np.isfinite(b)))})
    return sorted(result, key=lambda r: r['max_absolute'], reverse=True)


def localize_trace(args, bootstrap, parameters, values, emit):
    """Replay the first GPP jump using saved inputs, without any optimizer step."""
    jax = bootstrap['context']['runtime']['jax']
    contract = bootstrap['context']['runtime']['contract']
    scales = bootstrap['context']['scales']
    with args.trace_source.open('rb') as handle:
        saved = pickle.load(handle)
    micro, compact, total = saved['micro'], saved['compact'], saved['total']
    emit('full_trace')
    full, predicted = window._make_trace_evaluator(bootstrap)(parameters, micro)
    jax.block_until_ready(full)
    expected = saved['expected']
    stats = window.science_window_sufficient_statistics(
        predicted.fluxes, expected.fluxes, predicted.stocks, expected.stocks,
        flux_scales=scales.fluxes, stock_scales=scales.stocks, horizon=30, flux_basis=bootstrap['flux_basis'])
    score = float(window.linearized_science_window_loss(stats, total, flux_basis=bootstrap['flux_basis']))
    stats_diff = differences(jax, stats, saved['production_statistics'])
    a, b = np.asarray(compact.result.next_state.other), np.asarray(full.result.next_state.other)
    state_scale = np.asarray(bootstrap['context']['runtime']['base_statistics'].arrays['next_state'].scale)
    state_rows = []
    for day in range(10):
        valid = np.asarray(compact.result.next_state_defined.other[day]) & np.asarray(full.result.next_state_defined.other[day])
        delta = np.subtract(a[day], b[day], out=np.zeros(a.shape[1]), where=valid)
        rows = []
        for leaf in contract.state_leaves:
            if leaf.start is None or leaf.stop is None:
                continue
            part = np.abs(delta[leaf.start:leaf.stop])
            if not np.any(part):
                continue
            scale = state_scale[leaf.start:leaf.stop]
            normalized = np.divide(part, scale, out=np.zeros_like(part), where=np.isfinite(scale) & (scale > 0))
            rows.append({'field': leaf.key, 'start': int(leaf.start), 'stop': int(leaf.stop),
                         'max_absolute': float(np.max(part)), 'max_normalized': float(np.max(normalized))})
        rows.sort(key=lambda row: row['max_normalized'], reverse=True)
        state_rows.append({'day': day + 1, 'fields': rows})
    emit('day10_replay', full_trace_score=score, full_statistics_differences=stats_diff,
         daily_state_differences=state_rows,
         full_gpp=np.asarray(predicted.fluxes['gpp'].value).tolist(),
         compact_gpp=np.asarray(saved['compact_trajectory'].fluxes['gpp'].value).tolist())
    compact_states, compact_days = integration.step_audit.trace_days(jax, micro, compact)
    full_states, full_days = integration.step_audit.trace_days(jax, micro, full)
    step = jax.jit(integration.streamed.make_step(bootstrap))
    fixed = bootstrap['fixed_parameters']
    replay = {}
    for state_name, states in (('compact', compact_states), ('full', full_states)):
        for discrete_name, days in (('compact', compact_days), ('full', full_days)):
            result = step(values, states[9], fixed, days[9])
            jax.block_until_ready(result)
            replay[state_name + '_state_' + discrete_name + '_discrete'] = np.asarray(result['gpp']).tolist()
    interventions = []
    for row in state_rows[8]['fields'][:32]:
        state = np.asarray(compact_states[9]).copy()
        state[row['start']:row['stop']] = np.asarray(full_states[9])[row['start']:row['stop']]
        result = step(values, jax.device_put(state), fixed, compact_days[9])
        interventions.append({'field': row['field'], 'gpp': np.asarray(result['gpp']).tolist()})
    with args.output.with_suffix('.trace.pkl').open('wb') as handle:
        pickle.dump(jax.device_get({'full': full, 'micro': micro, 'total': total}), handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    emit('completed', status='completed', day10_replay=replay,
         day10_state_leaf_interventions=interventions, intervention_limit=32)


def run(args):
    started = time.perf_counter()
    source = args.diagnostic_source
    report = {'schema_version': 'conditioned_score_discrepancy_v1', 'status': 'running',
              'trained_updates': 0, 'source': str(source)}

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        window._atomic_json(args.output, report)
        print(f'score_discrepancy stage={stage} seconds={report["elapsed_seconds"]:.1f}', flush=True)

    emit('bootstrap')
    try:
        parent = json.loads((source / 'report.json').read_text())
        state = training.continuation.read_pickle(source / 'recovery.pkl')
        if state['completed_logical_update'] != parent['completed_logical_update']:
            raise ValueError('recovery and report endpoints differ')
        artifact_id, index, reference = failed_case(parent)
        emit('case_resolved', artifact=artifact_id, window=index, reference=reference,
             accepted_update=state['completed_logical_update'])
        _, original = window._load_parent_state(
            args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())['contract_sha256'])
        bootstrap = window._conditioned_bootstrap(training.bootstrap_args(args), recovered=original,
            warm_snow_free_boundary=True,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract))
        context = bootstrap['context']
        jax = context['runtime']['jax']
        scales = context['scales']
        parameters = window.replace_conditioned_coupled_gpp_trainable(
            bootstrap['fixed_parameters'], jax.device_put(state['trainable']))
        if args.trace_localization:
            localize_trace(args, bootstrap, parameters, jax.device_put(state['trainable']), emit)
            return report
        artifact = window._load_artifact(args, context, artifact_id)
        micros = [window._slice(artifact.batch.thirty_day, i, i + 1) for i in range(4)]
        emit('production_forward')
        outputs = [bootstrap['steps']['thirty_forward'](parameters, micro) for micro in micros]
        jax.block_until_ready(outputs)
        total = integration.step_audit.merge_statistics([v[0] for v in outputs])
        basis = bootstrap['flux_basis']

        def score(stats):
            return float(window.linearized_science_window_loss(stats, total, flux_basis=basis))

        micro = micros[index]
        production = score(outputs[index][0])
        emit('compact_trace', production_score=production,
             production_reproduced=bool(np.isclose(production, reference['production_linearized_score'], atol=1e-6, rtol=1e-5)),
             hard_failure_count=sum(window._hard_failure_count(window._hard_host(v[1])) for v in outputs))
        compact, compact_trajectory = integration.make_compact_trace_evaluator(bootstrap)(parameters, micro)
        observed, _ = integration.make_observation_gradient(bootstrap, 30)(
            integration.streamed.observations(compact), compact.result.next_state_defined.other,
            compact.active_pft_mask, total, micro)
        observed = float(observed)
        emit('production_trajectories', observation_score=observed,
             observation_reproduced=bool(np.isclose(observed, reference['observation_linearized_score'], atol=1e-6, rtol=1e-5)),
             score_agrees=bool(np.isclose(observed, production, atol=1e-6, rtol=1e-5)))
        evaluate = jax.jit(lambda p, m: bootstrap['steps']['window_trajectories'](p, m, rematerialize=False))
        production_trajectory, expected, hard = evaluate(parameters, micro)
        jax.block_until_ready(production_trajectory)

        def statistics(predicted):
            return window.science_window_sufficient_statistics(
                predicted.fluxes, expected.fluxes, predicted.stocks, expected.stocks,
                flux_scales=scales.fluxes, stock_scales=scales.stocks, horizon=30, flux_basis=basis)

        compact_stats = statistics(compact_trajectory)
        traced_stats = statistics(production_trajectory)
        # Save data for follow-up diagnosis, without another parameter checkpoint.
        payload = {'artifact': artifact_id, 'window': index, 'micro': micro, 'total': total,
                   'compact': compact, 'compact_trajectory': compact_trajectory,
                   'production_trajectory': production_trajectory, 'expected': expected,
                   'production_statistics': outputs[index][0]}
        with args.output.with_suffix('.trace.pkl').open('wb') as handle:
            pickle.dump(jax.device_get(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
        daily = []
        for category in ('fluxes', 'stocks'):
            for name, field in getattr(compact_trajectory, category).items():
                other = getattr(production_trajectory, category)[name]
                a, b = np.asarray(field.value), np.asarray(other.value)
                valid = np.asarray(field.defined) & np.asarray(other.defined)
                delta = np.subtract(a, b, out=np.zeros(a.shape, dtype=float), where=valid)
                daily.append({'category': category, 'name': name, 'compact': a.tolist(),
                              'production': b.tolist(), 'difference': delta.tolist(),
                              'defined_mismatches': int(np.sum(np.asarray(field.defined) != np.asarray(other.defined)))})
        emit('completed', status='completed', compact_postprocessed_score=score(compact_stats),
             production_trajectory_score=score(traced_stats),
             compact_statistics_differences=differences(jax, compact_stats, outputs[index][0]),
             traced_statistics_differences=differences(jax, traced_stats, outputs[index][0]), daily=daily)
    except Exception as exc:
        emit('failed', status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    return report


if __name__ == '__main__':
    parser = training._parser()
    parser.add_argument('--diagnostic-source', type=Path, default=Path(
        '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/conditioned-snow-free-continuation-v3-14991510'))
    parser.add_argument('--trace-localization', action='store_true')
    parser.add_argument('--trace-source', type=Path, default=Path(
        '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/conditioned-score-discrepancy-14991510.trace.pkl'))
    run(parser.parse_args())
