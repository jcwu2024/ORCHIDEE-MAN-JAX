"""Bounded changing-data serial/dual/recovery admission through the real runner.

The coordinator never initializes JAX: each training leg owns its GPU process,
and full-state comparison runs afterwards on CPU. No new training recipe.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def integration_commands(output, common, devices, *, python=sys.executable):
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError('integration requires two distinct allocated GPU identifiers')
    reports = {arm: output.with_name(f'{output.stem}.{arm}.json')
               for arm in ('serial', 'dual', 'recovered')}
    base = [python, '-m', 'scripts.hpc.run_mixed_learning', *common,
            '--integration-updates', '3']
    legs = []
    for name, arm, extra in (
        ('serial', 'serial', []),
        ('dual', 'dual', ['--dual-device-training']),
        ('stop', 'recovered', ['--dual-device-training', '--stop-after-update', '2']),
        ('resume', 'recovered', ['--dual-device-training', '--resume',
                               str(reports['recovered'].with_suffix('.recovery.checkpoint.pkl'))]),
    ):
        legs.append(dict(name=name, gpu=devices[0] if arm == 'serial' else ','.join(devices),
                         command=[*base, '--output', str(reports[arm]), *extra]))
    return legs, reports


def compare_records(reference, candidate):
    """Same existing scalar tolerances, applied to corresponding changing steps."""
    if len(reference) != 3 or len(candidate) != 3:
        raise ValueError('three completed update records are required')
    rows = []
    for expected, actual in zip(reference, candidate, strict=True):
        scalars = ('daily_loss', 'T7', 'T30', 'mixed_loss', 'norm', 'clip')
        numeric = all(np.isclose(actual[k], expected[k], rtol=1e-5, atol=1e-6)
                      for k in scalars)
        numeric = numeric and actual['gradient_norms'].keys() == expected['gradient_norms'].keys()
        numeric = numeric and all(np.isclose(actual['gradient_norms'][k], v, rtol=1e-5, atol=1e-6)
                                  for k, v in expected['gradient_norms'].items())
        hard_equal = all(actual[k] == expected[k] for k in ('daily_hard', 'recurrent_hard'))
        hard_zero = (not any(actual['daily_hard'].values())
                     and not any(v for h in actual['recurrent_hard'].values() for v in h.values()))
        identities = ('update', 'daily_batch_index', 'recurrent_schedule_index', 'integration_data')
        same_data = all(k in actual and k in expected and actual[k] == expected[k] for k in identities)
        rows.append(dict(update=actual['update'], scalar_parity=bool(numeric),
                         physical_counts_equal=hard_equal, all_hard_zero=hard_zero,
                         same_data=same_data,
                         passed=bool(numeric and hard_equal and hard_zero and same_data)))
    return dict(passed=all(r['passed'] for r in rows), updates=rows)


def compare_runs(output, reports):
    import jax

    from research.daily_coarse_graining.daily_operator.mixed_training_state import load_mixed_training_checkpoint
    from scripts.hpc.probe_public_rollout import gradient_comparison

    results = {name: json.loads(path.read_text()) for name, path in reports.items()}
    for name, report in results.items():
        if (report['status'] != 'completed' or report['completed_updates'] != 3
                or report['scope'] != 'integration_only' or not report['checkpoint_forward_replay']):
            raise ValueError(name+' did not complete the bounded real training path')
    reference = results['serial']
    for report in results.values():
        for key in ('configuration', 'model_protocol', 'source_git_head', 'execution_policy', 'data_plan'):
            if report[key] != reference[key]:
                raise ValueError('integration scientific/data identity differs: '+key)
    # Load through the shared checkpoint verifier, obtaining its saved method
    # from trusted local run artifacts, then cross-check identities between arms.
    import pickle
    states = {}
    for name, path in reports.items():
        checkpoint = path.with_suffix('.recovery.checkpoint.pkl')
        with checkpoint.open('rb') as handle:
            identity = pickle.load(handle)
        expected = {f'expected_{k}': identity[k] for k in (
            'method', 'source_identities', 'schedule_identity', 'fresh_initialization_identity')}
        del identity
        states[name] = load_mixed_training_checkpoint(checkpoint, **expected)
    ref = states['serial']
    for name, state in states.items():
        if state['completed_updates'] != 3 or state['daily_sampler_cursor'] != {'epoch': 0, 'batch_offset': 3} or state['recurrent_schedule_cursor'] != 3:
            raise ValueError('saved data/optimizer cursor is inconsistent with three mixed updates')
        for key in ('source_identities', 'schedule_identity', 'fresh_initialization_identity'):
            if state[key] != ref[key]:
                raise ValueError('checkpoint identity differs: '+key)
        method = {k: v for k, v in state['method'].items() if k != 'device_execution'}
        baseline = {k: v for k, v in ref['method'].items() if k != 'device_execution'}
        if method != baseline:
            raise ValueError('checkpoint method differs beyond device execution')
    comparisons = {}
    for name, baseline in (('dual', 'serial'), ('recovered', 'dual')):
        comparison = compare_records(results[baseline]['updates'], results[name]['updates'])
        trees = {key: gradient_comparison(jax, states[name][key], states[baseline][key])
                 for key in ('parameters', 'optimizer_state')}
        comparison.update(trees=trees, passed=bool(comparison['passed'] and all(t['passed'] for t in trees.values())))
        comparisons[name+'_vs_'+baseline] = comparison
    data = [r['integration_data'] for r in reference['updates']]
    rotating_components = {
        'daily': all(data[i]['daily_rows'] != data[i-1]['daily_rows'] for i in range(1, 3)),
        **{key: all(data[i]['recurrent_rows'][key] != data[i-1]['recurrent_rows'][key]
                    for i in range(1, 3)) for key in ('T7', 'T30')},
    }
    rotating = all(rotating_components.values())
    resumed = bool(results['recovered'].get('resumed_from'))
    result = dict(schema_version='dual_training_integration_v1',
                  status='passed' if rotating and resumed and all(c['passed'] for c in comparisons.values()) else 'failed',
                  reports={k: str(v) for k, v in reports.items()}, comparisons=comparisons,
                  rotating_data=rotating, rotating_components=rotating_components,
                  resumed_in_new_process=resumed,
                  retained_updates_per_arm=3, scientific_acceptance=False,
                  timing_seconds={k: [u['seconds'] for u in v['updates']] for k, v in results.items()},
                  timing_interpretation='three changing batches including first-call costs; not sustained throughput')
    output.write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] != 'passed':
        raise ValueError('continuous dual integration admission failed; see '+str(output))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args, common = parser.parse_known_args(argv)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if args.compare_only:
        reports = {arm: args.output.with_name(f'{args.output.stem}.{arm}.json')
                   for arm in ('serial', 'dual', 'recovered')}
        compare_runs(args.output, reports)
        return
    reserved = ('--resume', '--integration-updates', '--dual-device-training', '--stop-after-update',
                '--profile-checkpoint', '--learning-schedule')
    if any(arg.split('=')[0] in reserved for arg in common):
        parser.error('integration coordinator owns the run budget, layout and recovery flags')
    legs, reports = integration_commands(args.output, common, devices)
    if args.dry_run:
        print(json.dumps(legs, indent=2))
        return
    if args.output.exists() or any(path.exists() for path in reports.values()):
        raise FileExistsError('integration requires fresh report paths')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'scripts.hpc.probe_dual_training_continuity',
                    '--backend', 'gpu', '--output', str(args.output.with_suffix('.runtime.json'))],
                   check=True)
    for leg in legs:
        print(f"integration leg={leg['name']} gpu={leg['gpu']}", flush=True)
        subprocess.run(leg['command'], env={**os.environ, 'CUDA_VISIBLE_DEVICES': leg['gpu']}, check=True)
    subprocess.run([sys.executable, '-m', __name__ if __name__ != '__main__' else
                    'scripts.hpc.run_dual_training_integration', '--output', str(args.output), '--compare-only'],
                   env={**os.environ, 'JAX_PLATFORMS': 'cpu'}, check=True)
    print(f'integration completed seconds={time.perf_counter()-began:.1f}', flush=True)


if __name__ == '__main__':
    main()
