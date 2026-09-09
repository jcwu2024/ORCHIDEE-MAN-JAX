"""Ten corrected Attention updates, with train-selected pending recovery before monitoring."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from research.daily_coarse_graining.daily_operator import corrected_continuation as continuation
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_persistence import (
    _validate_parameter_tree,
    parameter_trees_identical,
)
from scripts.hpc import audit_conditioned_attention_streamed_integration as integration
from scripts.hpc.audit_conditioned_snow_free_monitor import close

persistent, window = integration.persistent, integration.window


def compare_monitor(actual, expected):
    checks = {'hard_gate': actual['hard_gate_passed'] is True}
    for name, value in expected['source_mean_losses'].items():
        checks['source.' + name] = close(actual['source_mean_losses'][name], value)
    for horizon, summary in expected['acceptance'].items():
        for name, row in summary['keys'].items():
            current = actual['acceptance'][horizon]['keys'][name]
            for metric in ('normalized_rmse', 'normalized_signed_bias', 'evaluated_values'):
                checks[f'{horizon}.{name}.{metric}'] = (current[metric] == row[metric] if metric == 'evaluated_values'
                                                       else close(current[metric], row[metric]))
    return {'passed': all(checks.values()), 'checks': checks}


def artifact_readability(args, context, contract):
    updates = sorted({u for p in contract.train_panels for g in p for u in g} | set(contract.monitor_updates))
    for update in updates:
        path = args.prepared_manifest.parent / context['prepared_records'][update]['path']
        with path.open('rb') as handle:
            if handle.read(2) != b'\x1f\x8b':
                raise ValueError(f'prepared artifact {update} is not a readable gzip file')
    return {'readable_artifacts': len(updates), 'artifact_updates': updates}


def bootstrap_args(args):
    result = copy.copy(args)
    # The historical initializer requires an unused report path; it writes no report.
    result.output = args.output.parent / 'bootstrap-context.json'
    return result


def run(args):
    started = time.perf_counter()
    contract = continuation.load_contract(args.continuation_contract)
    report = {'schema_version': 'conditioned_snow_free_continuation_report_v1', 'status': 'initializing',
              'plan': contract.plan, 'preflight_only': args.preflight_only,
              'independent_validation': False, 'promotion_eligible': False}

    def emit(stage, updates):
        report.update(updates)
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        window._atomic_json(args.output, report)
        print(f'corrected_continuation stage={stage} elapsed={report["elapsed_seconds"]:.1f}s', flush=True)

    try:
        snapshot = continuation.read_pickle(contract.plan['candidate_snapshot'])
        reference = json.loads(Path(contract.plan['monitor_reference']).read_text())
        if (snapshot['schema_version'] != 'corrected_train_selected_candidate_v1'
                or snapshot['proposed_logical_update'] != 4 or snapshot['warm_snow_free_boundary'] is not True
                or reference['status'] != 'completed' or reference['passed'] is not True
                or snapshot['parent_identity'] != reference['parent_identity']):
            raise ValueError('requires the admitted corrected candidate and its completed monitor')
        schema = _validate_parameter_tree(snapshot['trainable'], window.TRAINABLE_PARAMETER_COUNT)
        _, original = window._load_parent_state(args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())['contract_sha256'])
        print('corrected_continuation stage=bootstrap', flush=True)
        bootstrap = window._conditioned_bootstrap(bootstrap_args(args), recovered=original, warm_snow_free_boundary=True,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract))
        jax = bootstrap['context']['runtime']['jax']
        emit('artifacts_readable', {'status': 'running', 'parameter_schema': schema,
                                   'data_readability': artifact_readability(args, bootstrap['context'], contract)})
        directory = args.output.parent
        recovery = directory / 'recovery.pkl'
        state = None if args.preflight_only else continuation.read_state(recovery, contract.plan)
        if state is None:
            # Validate the actual saved tree, including host serialization, before any training.
            roundtrip = directory / 'candidate-reload.pkl'
            continuation.save_state(roundtrip, snapshot['trainable'])
            reloaded = continuation.read_pickle(roundtrip)
            if not parameter_trees_identical(snapshot['trainable'], reloaded):
                raise ValueError('candidate serialization changed parameters')
            values = jax.device_put(reloaded)
            baseline = persistent._monitor(args, bootstrap, contract, trainable=values, label='corrected_continuation_reload')
            comparison = compare_monitor(baseline, reference['monitor_after'])
            emit('baseline_verified', {'initial_monitor': baseline, 'baseline_comparison': comparison})
            if not comparison['passed']:
                raise RuntimeError('reloaded candidate does not reproduce its corrected monitor')
            if args.preflight_only:
                emit('completed', {'status': 'completed', 'passed': True, 'trained_updates': 0})
                return report
            state = {'schema_version': 'corrected_continuation_state_v1', 'plan': contract.plan,
                     'completed_logical_update': 4, 'trainable': jax.device_get(values),
                     'initial_monitor': baseline, 'records': [], 'monitors': [], 'stopped_reason': None}
            continuation.save_state(recovery, state)
        _validate_parameter_tree(state['trainable'], window.TRAINABLE_PARAMETER_COUNT)
        report.update(initial_monitor=state['initial_monitor'], records=state['records'], monitors=state['monitors'])
        callbacks = integration.StreamedHorizonCallbacks(bootstrap,
            callback=lambda event, row: emit(event, {'progress': row}), compact_trace=True)
        attached = callbacks.attach()
        kernels = dict(flatten_gradient=persistent._make_flattened_unit_gradient(jax),
                       gram_step=persistent._make_flat_gram(jax), combine_direction=persistent._make_flat_direction(jax),
                       direct_virtual_step=persistent._make_dtype_preserving_direct_virtual_step(jax))

        def propose(values, update, panel, registry):
            emit('train_update_start', {'logical_update': update})
            before_calls, before_records = dict(callbacks.calls), len(callbacks.records)
            result = dict(persistent._train_only_update(args, attached, contract, trainable=jax.device_put(values),
                panel=panel, source_registry=registry, logical_update=update, **kernels))
            calls = {h: callbacks.calls[h] - before_calls[h] for h in (7, 30)}
            records = callbacks.records[before_records:]
            admitted = calls == {7: 64, 30: 32} and len(records) == 96 and all(r['linearized_score_agrees'] for r in records)
            result.update(gradient_admitted=admitted, gradient_evidence={'passed': admitted,
                'calls': calls, 'records': records, 'reverse_compile_count': callbacks.reverse.compile_count})
            return result

        state = continuation.continue_training(contract, state, directory, propose=propose,
            monitor=lambda values, u: persistent._monitor(args, bootstrap, contract,
                trainable=jax.device_put(values), label=f'corrected_continuation_monitor_{u}'),
            transfer=lambda before, after: persistent.monitor_transfer_gate(contract, before, after),
            host=jax.device_get, emit=emit)
        final_monitor = state['monitors'][-1]['panel'] if state['monitors'] else state['initial_monitor']
        emit('completed', {'status': 'completed', 'completed_logical_update': state['completed_logical_update'],
            'trained_updates': len(state['records']), 'stopped_reason': state['stopped_reason'],
            'rejected': state.get('rejected'), 'records': state['records'], 'monitors': state['monitors'],
            'reverse_compile_count': callbacks.reverse.compile_count,
            'absolute_science': persistent.absolute_science_gate(contract, final_monitor)})
    except Exception as exc:
        emit(report.get('stage', 'bootstrap'), {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    return report


def _parser():
    parser = window._parser()
    parser.add_argument('--continuation-contract', type=Path, default=continuation.DEFAULT_PLAN)
    parser.add_argument('--preflight-only', action='store_true')
    return parser


if __name__ == '__main__':
    run(_parser().parse_args())
