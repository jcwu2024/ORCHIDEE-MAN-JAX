"""Export actual Teacher/parent/current curves on the existing independent panel; no training."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    _acceptance_trajectories,
    add_target_oriented_acceptance_statistics,
    derive_target_oriented_acceptance_scales,
    finalize_target_oriented_acceptance_statistics,
)
from scripts.hpc import run_conditioned_snow_free_continuation as runner
from scripts.hpc import run_target_oriented_structured_gpp_fresh_screen as fresh
from scripts.hpc.run_target_oriented_training import _validation_records
from scripts.hpc.successor_rollout_screen_prepared import load_prepared_rollout_artifact
from scripts.hpc.target_oriented_validation_prepared import load_target_oriented_validation_prepared

ROOT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
RUN = ROOT / 'runtime/outputs/training/conditioned-snow-free-continuation-v2-14984032'


def pack_trajectory(value):
    result = {}
    fluxes, stocks = _acceptance_trajectories(value.fluxes, value.stocks)
    for category, trajectories in (('fluxes', fluxes), ('stocks', stocks)):
        result[category] = {}
        for key, item in trajectories.items():
            values = np.asarray(item.value).reshape(-1)
            mask = np.asarray(item.defined, dtype=bool).reshape(-1)
            result[category][key] = [float(v) if valid and np.isfinite(v) else None
                                     for v, valid in zip(values, mask, strict=True)]
    return result


def run(args):
    started = time.perf_counter()
    contract = runner.continuation.load_contract(args.continuation_contract)
    state = runner.continuation.read_state(RUN / 'recovery.pkl', contract.plan)
    reference = json.loads((RUN / 'report.json').read_text())
    if state['completed_logical_update'] != 34 or not reference['planned_updates_completed']:
        raise ValueError('requires completed update-34 continuation')
    parent, _, _ = runner.starting_point(contract)
    _, original = runner.window._load_parent_state(
        args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
        persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())['contract_sha256'])
    print('curve_export stage=bootstrap', flush=True)
    bootstrap = runner.window._conditioned_bootstrap(runner.bootstrap_args(args), recovered=original,
        warm_snow_free_boundary=True,
        conditioning_contract=runner.window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract))
    context = bootstrap['context']
    runtime, jax = context['runtime'], context['runtime']['jax']
    path = Path(contract.plan['independent_validation']['prepared_manifest'])
    prepared, panel = load_target_oriented_validation_prepared(path,
        training_contract_sha256=context['training_contract'].sha256,
        target_contract_sha256=context['target_contract'].sha256, verify_artifacts=False)
    scales = derive_target_oriented_acceptance_scales(runtime['base_statistics'].arrays['next_state'].scale,
        runtime['process_statistics'].outputs['daily.fast_gpp'].scale, runtime['contract'])
    evaluate = fresh._make_fresh_evaluator(runtime, bootstrap['head'], context['template'], scales,
        context['scales'], include_trajectories=True)
    parameters = {label: runner.window.replace_conditioned_coupled_gpp_trainable(
        bootstrap['fixed_parameters'], jax.device_put(values))
        for label, values in (('update14', parent), ('update34', state['trainable']))}
    result = {'schema_version': 'conditioned_science_curves_v1', 'status': 'running',
        'source_job': 14984032, 'trained_updates': 0, 'selection': 'all 24 existing independent windows',
        'flux_unit': 'gC m-2 land day-1', 'stock_unit': 'gC m-2 land',
        'report_reproduction_checks': {}, 'windows': []}
    cells = {label: {} for label in parameters}
    for ordinal, (record, (slice_id, horizon, entry)) in enumerate(_validation_records(panel, prepared, mode='formal'), 1):
        artifact = load_prepared_rollout_artifact(path, record, slice_id=slice_id, window_entry=entry)
        row = {'ordinal': ordinal, 'slice': slice_id, 'horizon': horizon, 'entry': entry}
        for label, values in parameters.items():
            statistics, _, hard, predicted, expected = jax.device_get(
                evaluate(values, artifact.batch, artifact.day_inputs, horizon=horizon))
            row[label] = pack_trajectory(predicted)
            row['teacher'] = pack_trajectory(expected)
            row[label + '_hard_counts'] = {key: int(value) for key, value in zip(hard._fields, hard, strict=True)}
            key = (slice_id, horizon)
            cells[label][key] = (statistics if key not in cells[label] else
                add_target_oriented_acceptance_statistics(cells[label][key], statistics))
        result['windows'].append(row)
        runner.window._atomic_json(args.output, result)
        print(f'curve_export window={ordinal}/24 slice={slice_id} horizon={horizon}', flush=True)
    for label, source in (('update14', 'independent_before'), ('update34', 'independent_after')):
        for (slice_id, horizon), statistics in cells[label].items():
            actual = finalize_target_oriented_acceptance_statistics(statistics, horizon=horizon,
                target_contract=context['target_contract'])
            wanted = reference[source]['slices'][slice_id][str(horizon)]
            for key, item in wanted['keys'].items():
                for metric in ('normalized_rmse', 'normalized_signed_bias', 'evaluated_values'):
                    name = f'{label}.{slice_id}.{horizon}.{key}.{metric}'
                    result['report_reproduction_checks'][name] = bool(np.isclose(
                        actual['keys'][key][metric], item[metric], rtol=1e-5, atol=1e-7))
    result.update(status='completed', elapsed_seconds=time.perf_counter() - started,
                  report_reproduced=all(result['report_reproduction_checks'].values()))
    runner.window._atomic_json(args.output, result)
    if not result['report_reproduced']:
        raise RuntimeError('export does not reproduce independent acceptance report')
    print(f'curve_export completed seconds={result["elapsed_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    parser = runner._parser()
    parser.set_defaults(continuation_contract=runner.continuation.DEFAULT_PLAN.with_name('conditioned_snow_free_continuation_v2.json'))
    run(parser.parse_args())
