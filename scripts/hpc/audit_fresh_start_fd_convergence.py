"""Fixed-direction epsilon sweep with loss components and state-mask evidence."""
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
    protocol = json.loads(args.protocol.read_text())
    previous = json.loads(args.source_report.read_text())
    started = time.perf_counter()
    report = {'status': 'running', 'updates': 0, 'source_report': str(args.source_report), 'rows': []}

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output, report)
        print(f'fd_convergence stage={stage} seconds={report["elapsed_seconds"]:.1f}', flush=True)

    try:
        emit('bootstrap')
        runtime, head, template, scales, parameters, samples, metadata = training.bootstrap(args, protocol)
        jax = runtime['jax']
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.fd_convergence import (
            EPSILONS,
            comparison,
            mask_changes,
            plateau,
        )
        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective
        objective = make_objective(runtime, head, template, scales, protocol)
        diagnostic = make_objective(runtime, head, template, scales, protocol, state_diagnostics=True)
        sample = samples[0]
        if args.shape_only:
            jax.eval_shape(diagnostic, parameters, sample)
            emit('completed', status='passed', classification='diagnostic_shape_only')
            return report
        forward, detail = jax.jit(objective), jax.jit(diagnostic)
        emit('initial_gradient')
        (base_value, base_aux), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(parameters, sample)
        numeric = _tree_numeric_report(jax, gradient)
        if not numeric['all_finite'] or not np.isclose(numeric['l2_norm'], previous['gradient']['l2_norm'], rtol=1e-4):
            raise ValueError('original gradient does not reproduce')
        if not np.isclose(float(base_value), previous['initial'][0][0], atol=1e-6, rtol=1e-5):
            raise ValueError('initial loss does not reproduce')
        direction = jax.tree_util.tree_map(lambda g: (g/numeric['l2_norm']).astype(g.dtype), gradient)
        tangent = jax.jit(lambda p, d, s: jax.jvp(lambda v: objective(v, s)[0], (p,), (d,))[1])
        base_detailed, baseline = detail(parameters, sample)
        jax.block_until_ready(baseline)
        if not np.isclose(float(base_detailed), float(base_value), atol=1e-6, rtol=1e-5):
            raise ValueError('diagnostic output changes the baseline scalar')
        saved = {'metadata': metadata[0], 'baseline': jax.device_get(baseline['state_diagnostics']), 'endpoints': []}
        emit('baseline_ready', initial_loss=float(base_value), gradient=numeric,
             initial_components={k: float(base_aux[k]) for k in ('flux_loss', 'state_loss', 'process_loss')})
        weights = {'flux_loss': protocol['loss']['independent_daily_flux_mse'],
                   'state_loss': protocol['loss']['one_day_state_anchor'],
                   'process_loss': protocol['loss']['necessary_process_supervision']}

        def dot(left, right):
            return float(sum(jnp.vdot(a.astype('float64'), b.astype('float64')) for a, b in
                             zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True)))

        for epsilon in EPSILONS:
            endpoints, variants = [], []
            for sign in (1., -1.):
                varied = jax.tree_util.tree_map(lambda p, d: (p+sign*epsilon*d).astype(p.dtype), parameters, direction)
                value, aux = forward(varied, sample)
                detailed_value, details = detail(varied, sample)
                jax.block_until_ready(details)
                agrees = bool(np.isclose(float(value), float(detailed_value), atol=1e-6, rtol=1e-5))
                if not agrees:
                    raise ValueError('diagnostic output changes an endpoint scalar')
                changes = mask_changes(jax, baseline['state_diagnostics'], details['state_diagnostics'], head.contract)
                row = {'sign': sign, 'loss': float(value), 'hard': jax.device_get(aux['hard']),
                       'components': {k: float(aux[k]) for k in weights},
                       'dynamic_mismatch_count': int(aux['declared_dynamic_mask_mismatch']),
                       'masks': changes, 'diagnostic_scalar_agrees': agrees}
                endpoints.append(row)
                variants.append(varied)
                saved['endpoints'].append({'epsilon': epsilon, 'sign': sign,
                                          'diagnostic': jax.device_get(details['state_diagnostics'])})
            plus, minus = variants
            effective = jax.tree_util.tree_map(lambda a, b, p: ((a.astype('float64')-b.astype('float64'))/(2*epsilon)).astype(p.dtype), plus, minus, parameters)
            fd = (endpoints[0]['loss']-endpoints[1]['loss'])/(2*epsilon)
            reverse, jvp = dot(gradient, effective), float(tangent(parameters, effective, sample))
            result = {'epsilon': epsilon, 'finite_difference': fd, 'reverse_projection': reverse, 'jvp_projection': jvp,
                      'forward_slope': (endpoints[0]['loss']-float(base_value))/epsilon,
                      'backward_slope': (float(base_value)-endpoints[1]['loss'])/epsilon,
                      'effective_direction_norm': np.sqrt(dot(effective, effective)),
                      'component_slopes': {k: weights[k]*(endpoints[0]['components'][k]-endpoints[1]['components'][k])/(2*epsilon) for k in weights},
                      'endpoints': endpoints, 'mask_stable': all(v['masks']['stable'] for v in endpoints),
                      'physical_pass': all(all(int(n)==0 for n in v['hard'].values()) for v in endpoints),
                      **comparison(fd, reverse, jvp)}
            report['rows'].append(result)
            with args.output.with_suffix('.states.pkl').open('wb') as handle:
                pickle.dump(saved, handle, protocol=pickle.HIGHEST_PROTOCOL)
            emit('epsilon_completed', active_epsilon=epsilon)
        emit('completed', status='completed', plateau=plateau(report['rows']), classification='convergence_diagnosis_no_training')
    except Exception as exc:
        emit('failed', status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    return report


if __name__ == '__main__':
    parser = training.parser()
    parser.add_argument('--source-report', type=Path, default=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/fresh-start-gsmean-v2-free.json'))
    run(parser.parse_args())
