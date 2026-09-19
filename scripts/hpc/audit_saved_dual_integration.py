"""Read-only CPU evidence recovery after a dual integration launcher timeout.

This compares saved states without marking an unfinished runner completed.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from research.daily_coarse_graining.daily_operator.mixed_training_state import load_mixed_training_checkpoint
from scripts.hpc.check_daily_training_runtime import _atomic_json
from scripts.hpc.probe_public_rollout import gradient_comparison
from scripts.hpc.run_dual_training_integration import compare_records


def audit(prefix, output):
    reports, states = {}, {}
    for arm in ('serial', 'dual', 'recovered'):
        path = prefix.with_name(prefix.name + '.' + arm + '.json')
        report = json.loads(path.read_text())
        if (report['completed_updates'] != 3 or report['scope'] != 'integration_only'
                or 3 not in [r['update'] for r in report['monitors']]):
            raise ValueError(arm + ' lacks three updates and the final monitor')
        checkpoint = path.with_suffix('.recovery.checkpoint.pkl')
        with checkpoint.open('rb') as handle:
            saved = pickle.load(handle)
        identity = {f'expected_{key}': saved[key] for key in (
            'method', 'source_identities', 'schedule_identity', 'fresh_initialization_identity')}
        del saved
        states[arm] = load_mixed_training_checkpoint(checkpoint, **identity)
        reports[arm] = report
        state = states[arm]
        if (state['completed_updates'] != 3 or state['recurrent_schedule_cursor'] != 3
                or state['daily_sampler_cursor'] != {'epoch': 0, 'batch_offset': 3}):
            raise ValueError(arm + ' checkpoint cursor mismatch')
    reference = states['serial']
    for arm, state in states.items():
        for key in ('source_identities', 'schedule_identity', 'fresh_initialization_identity'):
            if state[key] != reference[key]:
                raise ValueError('checkpoint identity differs: ' + key)
        method = {k: v for k, v in state['method'].items() if k != 'device_execution'}
        expected = {k: v for k, v in reference['method'].items() if k != 'device_execution'}
        if method != expected:
            raise ValueError('scientific method differs across saved states')
        for key in ('configuration', 'model_protocol', 'source_git_head', 'execution_policy', 'data_plan'):
            if reports[arm][key] != reports['serial'][key]:
                raise ValueError('report identity differs: ' + key)
    comparisons = {}
    for arm, baseline in (('dual', 'serial'), ('recovered', 'dual')):
        record = compare_records(reports[baseline]['updates'], reports[arm]['updates'])
        record['trees'] = {key: gradient_comparison(jax, states[arm][key], states[baseline][key])
                           for key in ('parameters', 'optimizer_state')}
        record['bitwise_equal'] = {
            key: all(np.array_equal(a, b) for a, b in zip(
                jax.tree_util.tree_leaves(states[arm][key]),
                jax.tree_util.tree_leaves(states[baseline][key]), strict=True))
            for key in ('parameters', 'optimizer_state')}
        record['passed'] = record['passed'] and all(r['passed'] for r in record['trees'].values())
        comparisons[arm + '_vs_' + baseline] = record
    data = [r['integration_data'] for r in reports['serial']['updates']]
    rotating = (all(data[i]['daily_rows'] != data[i-1]['daily_rows'] for i in (1, 2))
                and all(data[i]['recurrent_rows'][h] != data[i-1]['recurrent_rows'][h]
                        for h in ('T7', 'T30') for i in (1, 2)))
    resumed = bool(reports['recovered'].get('resumed_from'))
    passed = rotating and resumed and all(r['passed'] for r in comparisons.values())
    result = dict(schema_version='saved_dual_integration_audit_v1',
                  status='passed' if passed else 'failed', scope='saved_state_and_update_parity_only',
                  comparisons=comparisons, rotating_data=rotating, resumed_in_new_process=resumed,
                  runner_states={k: dict(status=r['status'], stage=r['stage'],
                      checkpoint_forward_replay=r.get('checkpoint_forward_replay', False))
                      for k, r in reports.items()},
                  original_job_completed=False, full_original_protocol_completed=False,
                  scientific_acceptance=False, new_optimizer_updates=0,
                  missing_evidence='recovered final checkpoint forward replay/completion marker',
                  timing_seconds={k: [u['seconds'] for u in r['updates']] for k, r in reports.items()})
    _atomic_json(output, result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise ValueError('saved-state comparison failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.prefix, args.output)
