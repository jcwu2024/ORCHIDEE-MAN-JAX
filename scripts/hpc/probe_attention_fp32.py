"""Bounded attention-only FP64 promotion versus FP32 study; no model training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Active model widths/head counts, representative workloads rather than a
# sampled training-distribution histogram: (batch, query tokens, key tokens, width).
CASES = ((1, 32, 32, 256), (8, 32, 32, 256), (1, 128, 128, 384),
         (32, 1, 48, 384), (256, 1, 48, 384))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('explore_gpu', 'gpu', 'cpu'), default='explore_gpu')
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--quick', action='store_true')
    for name in ('dataset-manifest', 'acceptance-report', 'base-statistics',
                 'process-statistics', 'fast-statistics', 'prepared-manifest'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    from scripts.hpc.check_daily_training_runtime import _atomic_json

    started = time.perf_counter()
    report = dict(status='running', cases=[], teacher_data_read=False, parameter_updates=0,
                  scope='representative_attention_shapes_not_full_model_speed_or_science',
                  repeats=args.repeats, matrix_precision='highest_no_TF32_or_FP16', seed=20260915)

    def emit(stage):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output, report)
        print(json.dumps(dict(stage=stage, seconds=report['elapsed_seconds'])), flush=True)

    try:
        emit('initialize')
        if args.backend == 'explore_gpu':
            from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
            jax, devices = initialize_gpu_backend()
        else:
            import jax
            devices = jax.devices(args.backend)
        jax.config.update('jax_enable_x64', True)
        jax.config.update('jax_default_matmul_precision', 'highest')
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
            _attention,
            _linear_parameters,
        )
        from scripts.hpc.probe_one_day_batch import _compiled_memory
        from scripts.hpc.probe_public_rollout import gradient_comparison
        from scripts.hpc.run_mixed_learning import lowered_precision_summary

        report.update(jax_version=jax.__version__, device=str(devices[0]),
                      device_kind=devices[0].device_kind, backend=devices[0].platform)
        for batch, nq, nk, width in CASES[:1] if args.quick else CASES:
            keys = jax.random.split(jax.random.PRNGKey(report['seed']), 7)
            p = {name: _linear_parameters(key, width, width)
                 for name, key in zip(('query', 'key', 'value', 'output'), keys)}
            q = jax.random.normal(keys[4], (batch, nq, width), dtype=jnp.float32)
            kv = jax.random.normal(keys[5], (batch, nk, width), dtype=jnp.float32)
            weights = jax.random.normal(keys[6], (batch, nq, width), dtype=jnp.float32)
            mask = jnp.ones((batch, nq, nk), dtype=bool).at[0, 0, :].set(False)
            mask = mask.at[:, :, -1].set(False)
            row = dict(shape=dict(batch=batch, query_tokens=nq, key_tokens=nk, width=width, heads=8),
                       modes={}, comparisons={})
            report['cases'].append(row)
            executables, first = {}, {}
            for mode, dtype in (('legacy_fp64_scores', None), ('fp32_scores', jnp.float32)):
                def forward(p, q, kv, dtype=dtype):
                    return _attention(p, q, kv, heads=8, mask=mask, score_dtype=dtype)

                def loss(p, q, kv):
                    return jnp.mean(forward(p, q, kv) * weights)

                executables[mode] = {}
                row['modes'][mode] = {}
                for phase, fn in (('forward', jax.jit(forward)),
                                  ('value_and_grad', jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2))))):
                    emit('compile_'+mode+'_'+phase)
                    began = time.perf_counter()
                    lowered = fn.lower(p, q, kv)
                    precision = lowered_precision_summary(lowered)
                    compiled = lowered.compile()
                    row['modes'][mode][phase] = dict(compile_seconds=time.perf_counter()-began,
                        precision=precision, memory=_compiled_memory(compiled), samples_seconds=[])
                    executables[mode][phase] = compiled
                    result = jax.block_until_ready(compiled(p, q, kv))
                    first[mode, phase] = jax.device_get(result)

            reference = first['legacy_fp64_scores', 'forward']
            actual = first['fp32_scores', 'forward']
            row['comparisons']['output'] = dict(
                maximum_absolute_error=float(np.max(np.abs(actual-reference))),
                relative_l2=float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference), 1e-12)),
                passed=bool(np.allclose(actual, reference, atol=1e-6, rtol=1e-5)))
            row['nontrivial_reference'] = bool(np.linalg.norm(reference) > 0)
            old_loss, old_grad = first['legacy_fp64_scores', 'value_and_grad']
            new_loss, new_grad = first['fp32_scores', 'value_and_grad']
            row['comparisons']['gradient'] = gradient_comparison(jax, new_grad, old_grad)
            row['comparisons']['loss'] = dict(old=float(old_loss), new=float(new_loss),
                passed=bool(np.isclose(new_loss, old_loss, atol=1e-6, rtol=1e-5)))
            row['comparisons']['masked_query_gradient_zero'] = bool(np.all(new_grad[1][0, 0, :] == 0))
            row['comparisons']['masked_key_gradient_zero'] = bool(np.all(new_grad[2][:, -1, :] == 0))
            emit('compare_completed')
            # Alternate order after two additional warmups to reduce clock/order bias.
            for repeat in range(-2, args.repeats):
                modes = tuple(executables) if repeat % 2 else tuple(reversed(executables))
                for mode in modes:
                    for phase, fn in executables[mode].items():
                        began = time.perf_counter()
                        result = jax.block_until_ready(fn(p, q, kv))
                        seconds = time.perf_counter()-began
                        if repeat >= 0:
                            row['modes'][mode][phase]['samples_seconds'].append(seconds)
                        del result
            for mode in executables:
                for phase in executables[mode]:
                    measurement = row['modes'][mode][phase]
                    measurement['median_seconds'] = float(np.median(measurement['samples_seconds']))
            row['time_reduction'] = {
                phase: 1-row['modes']['fp32_scores'][phase]['median_seconds'] /
                row['modes']['legacy_fp64_scores'][phase]['median_seconds']
                for phase in ('forward', 'value_and_grad')}
            row['passed'] = bool(all(row['comparisons'][key]['passed'] for key in ('output', 'gradient', 'loss'))
                and row['nontrivial_reference']
                and row['comparisons']['masked_query_gradient_zero']
                and row['comparisons']['masked_key_gradient_zero'])
            emit('case_completed')
            del executables, first, compiled, lowered
        report.update(status='completed', primitive_numerical_passed=all(r['passed'] for r in report['cases']),
                      full_model_admission=False, gpu_timing=args.backend != 'cpu')
        emit('completed')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        emit('failed')
        raise


if __name__ == '__main__':
    main()
