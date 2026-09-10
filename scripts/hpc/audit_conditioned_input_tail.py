"""Fixed-weight, train-only input-tail admission; no optimizer or weight writes."""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.daily_coarse_graining.daily_operator.input_conditioning import (
    CONTINUOUS_TAIL_THRESHOLD,
    CONTINUOUS_TAIL_VERSION,
)
from scripts.hpc import audit_conditioned_score_discrepancy as source
from scripts.hpc import audit_conditioned_snow_free_rollout as screen
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps

training, window, integration = source.training, source.window, source.integration
CASES = ((883, 30, 3), (322, 7, 1), (442, 30, 0))
ROOT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')


def with_tail(bootstrap, threshold):
    """Bind all forward consumers; old training helper closures must not be used."""
    context = bootstrap['context']
    normalization = bootstrap['head'].input_normalization._replace(continuous_tail_threshold=threshold)
    head = replace(bootstrap['head'], input_normalization=normalization)
    steps = _make_steps(context['runtime'], head, context['template'], context['scales'],
                        context['target_contract'], science_flux_basis=bootstrap['flux_basis'])
    return dict(bootstrap, head=head, steps=steps)


def within_regression_cap(candidate, reference):
    values = [(candidate['total'], reference['total'])]
    values.extend((candidate[group][name]['mse'], value['mse'])
                  for group in ('fluxes', 'stocks') for name, value in reference[group].items())
    return bool(all(np.isfinite(a) and a <= 1.05 * b + 1e-8 for a, b in values))


def physical_curves(predicted, expected):
    result = {}
    for category in ('fluxes', 'stocks'):
        result[category] = {}
        for name, field in getattr(predicted, category).items():
            truth = getattr(expected, category)[name]
            a, b = np.asarray(field.value), np.asarray(truth.value)
            valid = np.asarray(field.defined) & np.asarray(truth.defined)
            delta = np.where(valid, a - b, 0.)
            item = {'predicted': a.tolist(), 'teacher': b.tolist(),
                    'defined': valid.tolist(), 'finite': bool(np.all(np.isfinite(a[valid]))),
                    'rmse': float(np.sqrt(np.sum(delta ** 2) / max(int(valid.sum()), 1)))}
            if category == 'fluxes':
                item['cumulative_error'] = np.sum(delta, axis=0).tolist()
            else:
                # Net change over the returned endpoint series (day 1 to H),
                # not day-start to H. Label this explicitly in the report.
                item['endpoint_series_change_error'] = (delta[-1] - delta[0]).tolist()
                # Both trajectories share the Teacher day-start state, so
                # final stock error also measures full-window net-growth error.
                item['net_change_error_from_common_initial_state'] = delta[-1].tolist()
            result[category][name] = item
    return result


def doc_probes(bootstrap, values, emit):
    jax = bootstrap['context']['runtime']['jax']
    directory = ROOT / 'runtime/outputs/diagnostics'
    with (directory / 'conditioned-score-discrepancy-14991510.trace.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    with (directory / 'conditioned-score-day10-14991510.trace.pkl').open('rb') as handle:
        full = pickle.load(handle)['full']
    states, days = integration.step_audit.trace_days(jax, saved['micro'], saved['compact'])
    full_states, _ = integration.step_audit.trace_days(jax, saved['micro'], full)
    step = jax.jit(integration.streamed.make_step(bootstrap))
    fixed = bootstrap['fixed_parameters']
    emit('doc_day10_probe')
    original = np.asarray(step(values, states[9], fixed, days[9])['gpp'])
    result = {'compact_gpp': original.tolist(), 'interventions': []}
    for name, start, stop in (('DOC_leaf', 1327, 2223), ('DOC_coordinate_1785', 1785, 1786)):
        changed = np.asarray(states[9]).copy()
        changed[start:stop] = np.asarray(full_states[9])[start:stop]
        gpp = np.asarray(step(values, jax.device_put(changed), fixed, days[9])['gpp'])
        gap = float(np.max(abs(gpp - original)))
        result['interventions'].append({'name': name, 'gpp': gpp.tolist(),
                                        'max_difference': gap, 'passed': bool(np.isfinite(gap) and gap <= 1e-5)})
    return result


def run(args):
    started = time.perf_counter()
    report = {'schema_version': CONTINUOUS_TAIL_VERSION, 'status': 'running', 'cases': [],
              'threshold': CONTINUOUS_TAIL_THRESHOLD, 'training_updates': 0,
              'validation_accessed': False, 'source_update': 41}

    def emit(stage, **updates):
        report.update(updates, stage=stage, elapsed_seconds=time.perf_counter() - started)
        window._atomic_json(args.output, report)
        print(f'input_tail stage={stage} elapsed={report["elapsed_seconds"]:.1f}s', flush=True)

    try:
        emit('bootstrap')
        recovered = training.continuation.read_pickle(args.diagnostic_source / 'recovery.pkl')
        if recovered['completed_logical_update'] != 41:
            raise ValueError('fixed-weight admission requires accepted update 41')
        _, original = window._load_parent_state(
            args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())['contract_sha256'])
        bootstrap = window._conditioned_bootstrap(training.bootstrap_args(args), recovered=original,
            warm_snow_free_boundary=True,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract))
        jax = bootstrap['context']['runtime']['jax']
        values = jax.device_put(recovered['trainable'])
        parameters = window.replace_conditioned_coupled_gpp_trainable(bootstrap['fixed_parameters'], values)
        artifacts = {u: window._load_artifact(args, bootstrap['context'], u) for u in (883, 322, 442)}
        references = {}
        for arm, threshold in (('reference', None), ('tail', CONTINUOUS_TAIL_THRESHOLD)):
            current = with_tail(bootstrap, threshold)
            trace_forward = integration.make_compact_trace_evaluator(current)
            extract = {h: screen.make_trace_statistics(current, h) for h in (7, 30)}
            observe = {h: integration.make_observation_gradient(current, h) for h in (7, 30)}
            production_trace = jax.jit(lambda p, m, steps=current['steps']: steps['window_trajectories'](p, m, rematerialize=False))
            report.setdefault('doc_probes', {})[arm] = doc_probes(current, values, emit)
            for update, horizon, index in CASES:
                emit('production_forward', active_case=[arm, update, horizon, index])
                batch = getattr(artifacts[update].batch, 'seven_day' if horizon == 7 else 'thirty_day')
                micros = [window._slice(batch, i, i + 1) for i in range(len(batch.inputs.continuous_state))]
                forward = current['steps']['seven_forward' if horizon == 7 else 'thirty_forward']
                outputs = [forward(parameters, micro) for micro in micros]
                jax.block_until_ready(outputs)
                total = integration.step_audit.merge_statistics([v[0] for v in outputs])
                stats, hard = outputs[index]
                row = {'arm': arm, 'artifact': update, 'horizon': horizon, 'window': index,
                       'science': screen.science_summary(stats, bootstrap['flux_basis']),
                       'hard_failure_count': sum(window._hard_failure_count(window._hard_host(v[1])) for v in outputs)}
                report['cases'].append(row)
                emit('compact_trace')
                trace, predicted = trace_forward(parameters, micros[index])
                recorded, expected = extract[horizon](predicted, micros[index])
                production_predicted, _, _ = production_trace(parameters, micros[index])
                production_recorded, _ = extract[horizon](production_predicted, micros[index])
                masks_agree = all(np.array_equal(field.defined, getattr(production_predicted, category)[name].defined)
                    for category in ('fluxes', 'stocks') for name, field in getattr(predicted, category).items())
                production_trace_agrees = all(np.allclose(a, b, atol=1e-6, rtol=1e-5)
                    for a, b in zip(jax.tree_util.tree_leaves(stats), jax.tree_util.tree_leaves(production_recorded), strict=True))
                score = float(window.linearized_science_window_loss(stats, total, flux_basis=bootstrap['flux_basis']))
                observed, _ = observe[horizon](integration.streamed.observations(trace),
                    trace.result.next_state_defined.other, trace.active_pft_mask, total, micros[index])
                row.update(production_score=score, observation_score=float(observed),
                           score_agrees=bool(np.isclose(float(observed), score, atol=1e-6, rtol=1e-5)),
                           recorded_science=screen.science_summary(recorded, bootstrap['flux_basis']),
                           masks_agree=bool(masks_agree), production_trace_agrees=bool(production_trace_agrees),
                           trajectories=physical_curves(production_predicted, expected),
                           compact_trajectories=physical_curves(predicted, expected))
                if arm == 'reference':
                    references[(update, horizon, index)] = row
                else:
                    row['regression_cap_passed'] = within_regression_cap(row['science'], references[(update, horizon, index)]['science'])
                emit('case_completed')
            del current, trace_forward, extract, observe, production_trace
            jax.clear_caches()
        repaired = [r for r in report['cases'] if r['arm'] == 'tail']
        passed = len(repaired) == len(CASES) and all(r['score_agrees'] and r['hard_failure_count'] == 0
            and r['masks_agree'] and r['production_trace_agrees']
            and r['regression_cap_passed'] and all(v['finite'] for group in r['trajectories'].values() for v in group.values())
            for r in repaired) and all(r['passed'] for r in report['doc_probes']['tail']['interventions'])
        emit('completed', status='completed', forward_gate_passed=bool(passed),
             classification='forward_admitted_derivatives_pending' if passed else 'rejected_no_training')
    except Exception as exc:
        emit('failed', status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    return report


if __name__ == '__main__':
    parser = training._parser()
    parser.add_argument('--diagnostic-source', type=Path, default=ROOT /
        'runtime/outputs/training/conditioned-snow-free-continuation-v3-14991510')
    run(parser.parse_args())
