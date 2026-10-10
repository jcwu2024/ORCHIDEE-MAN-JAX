"""Bounded synthetic deterministic-index/native-checkpoint compatibility probe.

No Teacher, model parameters, training or scientific acceptance is involved.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous diagnostic')
    started = time.perf_counter()
    report = dict(status='running', stage='initializing', scope='synthetic_only',
                  xla_flags=os.environ.get('XLA_FLAGS'), rows=[], real_model_checked=False,
                  teacher_data_read=False, real_training_updates=0, paid_jobs=0,
                  gpu_count=1, process_budget_seconds=120)

    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(args.output)
        print(report['stage'], flush=True)

    def timed_out(_signum, _frame):
        raise TimeoutError('110-second internal synthetic bound')

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(110)
    save()
    try:
        if '--xla_gpu_deterministic_ops=true' not in report['xla_flags'].split():
            raise ValueError('exact deterministic flag required')
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
        jax, devices = initialize_gpu_backend()
        import jax.numpy as jnp
        if jax.__version__ != '0.4.38' or len(devices) != 1:
            raise ValueError('requires original JAX0.4.38 and one GPU')
        jax.config.update('jax_enable_x64', True)
        report.update(jax_version=jax.__version__, actual_backend=jax.default_backend(),
                      devices=[str(d) for d in devices], neural_primitive_dtype='float32',
                      physical_accumulation_dtype='float64')
        # Dynamic unsorted indices repeat across 97 buckets; no unique-index hint.
        rng = np.random.default_rng(20261002)
        indices = rng.integers(0, 97, size=4096, dtype=np.int32)
        updates = rng.normal(size=(4096, 16)).astype(np.float32)
        weights = rng.normal(size=(4096, 16)).astype(np.float32)
        initial = rng.normal(scale=.1, size=(97, 16)).astype(np.float32)
        days = rng.normal(scale=.03, size=(7, 4096, 16)).astype(np.float32)
        params = rng.normal(scale=.02, size=(4096, 16)).astype(np.float32)

        def scatter(values, index):
            return jnp.zeros((97, 16), jnp.float32).at[index].add(values)

        def gathered_loss(state, index, cotangent):
            return jnp.sum(jnp.sin(state[index])*cotangent, dtype=jnp.float64)

        def scan_loss(parameters, state, index, forcing):
            def step(carry, day):
                selected = jnp.tanh(carry[index]*parameters+day)
                combined = scatter(selected, index)/jnp.float32(43)
                following = jnp.tanh(carry*jnp.float32(.7)+combined)
                return following, jnp.sum(following*following, dtype=jnp.float64)
            _, values = jax.lax.scan(jax.checkpoint(step), state, forcing)
            return jnp.sum(values, dtype=jnp.float64)

        cases = [
            ('duplicate_scatter_add', scatter, (updates, indices)),
            ('gather_native_vjp', jax.value_and_grad(gathered_loss), (initial, indices, weights)),
            ('checkpoint_scan_parameter_state_vjp',
             jax.value_and_grad(scan_loss, argnums=(0, 1)), (params, initial, indices, days)),
        ]
        artifacts = {}
        for name, function, host_args in cases:
            row = dict(name=name, status='lowering', repeat_calls=8, indices_sorted=False,
                       unique_indices=False)
            report['rows'].append(row)
            report['stage'] = name+':lowering'
            save()
            try:
                device_args = jax.device_put(host_args)
                began = time.perf_counter()
                lowered = jax.jit(function).lower(*device_args)
                row['lower_seconds'] = time.perf_counter()-began
                stable = str(lowered.compiler_ir(dialect='stablehlo'))
                args.output.with_name(args.output.stem+'.'+name+'.stablehlo.txt').write_text(stable)
                row['stablehlo_scatter_count'] = stable.count('stablehlo.scatter')
                row['stablehlo_gather_count'] = stable.count('stablehlo.gather')
                row['status'] = 'compiling'
                report['stage'] = name+':compiling'
                save()
                began = time.perf_counter()
                executable = lowered.compile(compiler_options={
                    'xla_gpu_enable_llvm_module_compilation_parallelism': False})
                row['backend_seconds'] = time.perf_counter()-began
                first = None
                durations = []
                max_error = 0.
                equal = True
                for repeat in range(8):
                    began = time.perf_counter()
                    value = jax.device_get(jax.block_until_ready(executable(*device_args)))
                    durations.append(time.perf_counter()-began)
                    leaves = [np.asarray(v) for v in jax.tree_util.tree_leaves(value)]
                    if not all(np.isfinite(v).all() for v in leaves):
                        raise ValueError('nonfinite synthetic output')
                    if not any(np.any(v != 0) for v in leaves):
                        raise ValueError('vacuous all-zero synthetic result')
                    if first is None:
                        first = leaves
                        for i, leaf in enumerate(leaves):
                            artifacts[name+'.leaf'+str(i)] = leaf
                    else:
                        equal &= all(a.tobytes() == b.tobytes() for a, b in zip(first, leaves))
                        max_error = max(max_error, max(float(np.max(np.abs(a.astype(np.float64)-b)))
                                                      for a, b in zip(first, leaves)))
                row.update(status='passed' if equal else 'repeat_drift', finite=True,
                           bitwise_repeat_stable=bool(equal), max_repeat_abs_error=max_error,
                           repeat_seconds=durations, output_shapes=[list(v.shape) for v in first],
                           output_dtypes=[str(v.dtype) for v in first])
            except Exception as error:
                row.update(status='error', error_type=type(error).__name__, error=str(error))
                traceback.print_exc()
            report['stage'] = name+':'+row['status']
            report['device_memory'] = devices[0].memory_stats()
            save()
        np.savez(args.output.with_suffix('.outputs.npz'), **artifacts)
        report['status'] = 'passed' if all(r['status']=='passed' for r in report['rows']) else 'incompatible_or_unstable'
        report['stage'] = 'completed'
        save()
        return 0 if report['status']=='passed' else 1
    except BaseException as error:
        report.update(status='error', stage='exception', error_type=type(error).__name__, error=str(error))
        save()
        traceback.print_exc()
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
