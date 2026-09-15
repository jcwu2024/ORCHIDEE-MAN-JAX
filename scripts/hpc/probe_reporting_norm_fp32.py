"""Time reporting-only FP64/FP32 norms on preserved real parameter gradients."""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=Path, nargs='+')
    parser.add_argument('--backend', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--repeats', type=int, default=51)
    for name in ('dataset-manifest', 'acceptance-report', 'base-statistics',
                 'process-statistics', 'fast-statistics', 'prepared-manifest'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if bool(args.references) == args.synthetic or args.repeats < 3:
        parser.error('choose references or synthetic, and at least three repeats')
    from scripts.hpc.check_daily_training_runtime import _atomic_json, _initialize_runtime

    started = time.perf_counter()
    report = dict(status='running', cases=[], parameter_updates=0, teacher_data_read=False,
                  scope='reporting_norm_only_not_optimizer_or_full_update', repeats=args.repeats)

    def emit(stage):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output, report)
        print(f'reporting_norm stage={stage} seconds={report["elapsed_seconds"]:.2f}', flush=True)

    try:
        jax, jnp, devices = _initialize_runtime(args.backend)
        report.update(jax_version=jax.__version__, device_kind=devices[0].device_kind)
        for path in args.references or [None]:
            emit('load_reference')
            if path is None:
                host = dict(a=np.linspace(-5, 5, 10000, dtype=np.float32),
                            b=np.full((32, 32), 1e-7, dtype=np.float32))
            else:
                with path.open('rb') as handle:
                    saved = pickle.load(handle)
                host = saved['gradient']
                del saved
            tree = jax.device_put(host)
            leaves = jax.tree_util.tree_leaves(host)
            row = dict(reference=str(path), elements=sum(v.size for v in leaves),
                       input_dtypes=sorted({str(v.dtype) for v in leaves}), modes={})
            report['cases'].append(row)
            del host, leaves
            executables = {}
            for name, dtype in (('fp64', jnp.float64), ('fp32', jnp.float32)):
                # Exactly the reporting expression in run_mixed_learning; AdamW's
                # independent FP32 clipping norm is outside this comparison.
                fn = jax.jit(lambda values, dtype=dtype: jnp.sqrt(sum(
                    jnp.sum(jnp.square(v.astype(dtype)))
                    for v in jax.tree_util.tree_leaves(values))))
                emit('compile_'+name)
                began = time.perf_counter()
                executable = fn.lower(tree).compile()
                row['modes'][name] = dict(compile_seconds=time.perf_counter()-began,
                    value=float(executable(tree)), samples_seconds=[])
                executables[name] = executable
            old, new = (row['modes'][key]['value'] for key in ('fp64', 'fp32'))
            row.update(relative_error=abs(new-old)/max(abs(old), 1e-12),
                       passed=bool(np.isfinite(old) and np.isfinite(new)
                                   and np.isclose(new, old, rtol=1e-5, atol=1e-6)))
            until = time.perf_counter()+0.5
            while time.perf_counter() < until:
                for executable in executables.values():
                    jax.block_until_ready(executable(tree))
            for index in range(args.repeats):
                for name in (('fp64','fp32') if index % 2 else ('fp32','fp64')):
                    began = time.perf_counter()
                    jax.block_until_ready(executables[name](tree))
                    row['modes'][name]['samples_seconds'].append(time.perf_counter()-began)
            for result in row['modes'].values():
                result['median_seconds'] = float(np.median(result['samples_seconds']))
            row['time_reduction'] = 1-row['modes']['fp32']['median_seconds']/row['modes']['fp64']['median_seconds']
            emit('case_completed')
            del tree, executables, executable, fn
        report.update(status='completed', passed=all(row['passed'] for row in report['cases']))
        emit('completed')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        emit('failed')
        raise


if __name__ == '__main__':
    main()
