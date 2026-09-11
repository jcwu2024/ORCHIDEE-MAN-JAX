"""Localize completed fresh-start state-loss dominance; no parameter updates."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training


def run(args):
    from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text())
    reference = json.loads(args.probe_report.read_text())
    report = {'status': 'running', 'trained_updates': 0, 'reference': str(args.probe_report), 'cases': []}

    def emit(stage, **updates):
        report.update(updates, stage=stage, elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output, report)
        print(f'loss_balance stage={stage} seconds={report["elapsed_seconds"]:.1f}', flush=True)

    try:
        emit('bootstrap')
        runtime, head, template, scales, parameters, samples, metadata = training.bootstrap(args, protocol)
        jax = runtime['jax']
        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective
        from research.daily_coarse_graining.daily_operator.state_loss_attribution import (
            attribute_state_loss,
            state_field_names,
        )
        objective = make_objective(runtime, head, template, scales, protocol, state_diagnostics=True)
        if args.shape_only:
            for sample in samples:
                jax.eval_shape(objective, parameters, sample)
            emit('completed', status='passed', classification='diagnostic_objective_shapes_only')
            return report
        names = state_field_names(head.contract, runtime['adapter'].input_assembler)
        forward = jax.jit(objective)
        saved = {'field_names': names, 'samples': metadata, 'states': []}
        for arm in ('initial', 'update12'):
            if arm == 'update12':
                with args.probe_report.with_suffix('.checkpoint.pkl').open('rb') as handle:
                    recovered = pickle.load(handle)
                parameters = jax.device_put(recovered['parameters'])
                del recovered
            prior = reference['initial'] if arm == 'initial' else reference['monitors'][-1]['samples']
            for index, sample in enumerate(samples):
                emit('diagnostic_forward', active_case=[arm, index])
                value, aux = forward(parameters, sample)
                jax.block_until_ready(aux)
                diagnostic = jax.device_get(aux['state_diagnostics'])
                saved['states'].append({'arm': arm, 'sample': index, 'diagnostic': diagnostic})
                # Persist before CPU attribution, so a labeling issue never
                # requires another network execution.
                with args.output.with_suffix('.states.pkl').open('wb') as handle:
                    pickle.dump(saved, handle, protocol=pickle.HIGHEST_PROTOCOL)
                attribution = attribute_state_loss(diagnostic, names)
                agrees = bool(np.isclose(float(value), prior[index][0], atol=1e-6, rtol=1e-5))
                row = {'arm': arm, 'sample': index, 'loss': float(value), 'reference_loss': prior[index][0],
                       'score_agrees': agrees, 'hard': jax.device_get(aux['hard']), 'attribution': attribution}
                report['cases'].append(row)
                emit('case_completed')
                if not agrees or any(int(v) != 0 for v in row['hard'].values()):
                    raise ValueError('diagnostic forward differs from completed probe')
            if arm == 'initial':
                emit('original_gradient')
                ordinary = make_objective(runtime, head, template, scales, protocol)
                (value, _), gradient = jax.jit(jax.value_and_grad(ordinary, has_aux=True))(parameters, samples[0])
                numeric = _tree_numeric_report(jax, gradient)
                if not np.isclose(numeric['l2_norm'], reference['gradient']['l2_norm'], rtol=1e-4):
                    raise ValueError('original gradient norm does not reproduce')
                rows = []
                for index, field in enumerate(head.learned_schema.fields):
                    values = gradient['target_oriented']['state_heads'][index]
                    if values:
                        rows.append({'field': field.name, **_tree_numeric_report(jax, values)})
                rows.sort(key=lambda row: row['l2_norm'], reverse=True)
                emit('state_head_gradients', original_gradient=numeric, state_head_gradients=rows)
                del gradient
        emit('completed', status='completed', classification='localization_only_no_training')
    except Exception as exc:
        emit('failed', status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    return report


if __name__ == '__main__':
    parser = training.parser()
    parser.add_argument('--probe-report', type=Path, default=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/fresh-start-one-day-v2.json'))
    run(parser.parse_args())
