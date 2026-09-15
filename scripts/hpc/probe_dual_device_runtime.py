"""Tiny two-device placement/update check with synthetic arrays only."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts.hpc.check_daily_training_runtime import _atomic_json, _tree_error
from scripts.hpc.mixed_learning_dual_device import split_components


def run(args):
    import jax
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
    from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        initialize_target_oriented_adamw,
    )

    started = time.perf_counter()
    jax.config.update('jax_enable_x64', True)
    devices = jax.local_devices(backend=args.backend)
    if len(devices) != 2:
        raise ValueError('probe requires exactly two visible devices')
    primary, secondary = devices
    rng = np.random.default_rng(20260915)
    parameters = jax.device_put(dict(matrix=rng.normal(0, .1, (8, 8)).astype('float32'),
                                    bias=np.zeros(8, 'float32')), primary)
    optimizer = initialize_target_oriented_adamw(parameters)
    data = {h: rng.normal(0, .2, (h, 8)).astype('float32') for h in (1, 7, 30)}

    def loss(parameters, days):
        def step(state, day):
            state = jnp.tanh(state@parameters['matrix']+parameters['bias']+day)
            return state, state
        _, states = jax.lax.scan(jax.checkpoint(step), jnp.zeros(8, jnp.float32), days)
        return jnp.mean(jnp.square(states-jnp.asarray(.1, jnp.float32)))

    backward = jax.jit(jax.value_and_grad(loss))
    executables = {}
    for device in devices:
        with jax.default_device(device):
            placed = jax.device_put(parameters, device)
            executables[device.id] = {
                h: backward.lower(placed, jax.device_put(data[h], device)).compile()
                for h in (1, 7, 30)
            }
    combine = make_mixed_gradient(jax, jnp)
    update = make_update(dict(optimizer=dict(learning_rate=1e-5, clip_norm=1.,
        beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=0.)))
    records = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for index in range(2):
            peer_parameters = jax.device_put(parameters, secondary)

            def daily():
                with jax.default_device(primary):
                    return executables[primary.id][1](parameters, jax.device_put(data[1], primary))

            def recurrent(device):
                weights = parameters if device == primary else peer_parameters
                with jax.default_device(device):
                    return tuple(executables[device.id][h](weights, jax.device_put(data[h], device))
                                 for h in (7, 30))

            outputs = []
            for dual in (False, True):
                day, windows = split_components(
                    daily, lambda: recurrent(secondary if dual else primary),
                    executor=executor if dual else None)
                with jax.default_device(primary):
                    gradient = day[1]
                    for _, recurrent_gradient in windows:
                        gradient = combine(gradient, jax.device_put(recurrent_gradient, primary), 1., .5)
                    proposal = update(parameters, gradient, optimizer)
                    jax.block_until_ready(proposal)
                outputs.append((gradient, proposal))
            error = _tree_error(jax, outputs[1], outputs[0], rtol=1e-5, atol=1e-6)
            parameters, optimizer, _, _ = outputs[1][1]
            synchronized = jax.device_put(parameters, secondary)
            jax.block_until_ready(synchronized)
            if any(leaf.devices() != {secondary} for leaf in jax.tree_util.tree_leaves(synchronized)):
                raise ValueError('updated weights did not reach the recurrent device')
            records.append(dict(update=index+1, optimizer_step=int(optimizer.step),
                                maximum_absolute_error=error))
    report = dict(schema_version='dual_device_runtime_probe_v1', status='passed',
                  devices=[str(d) for d in devices], jax_version=jax.__version__,
                  updates=records, elapsed_seconds=time.perf_counter()-started,
                  teacher_payload_reads=0, real_model_acceptance=False)
    _atomic_json(args.output, report)
    print(report, flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--backend', choices=('gpu', 'cpu'), default='gpu')
    # Shared runtime launcher supplies these paths; this synthetic check never reads them.
    for name in ('dataset-manifest', 'acceptance-report', 'base-statistics',
                 'process-statistics', 'fast-statistics', 'prepared-manifest'):
        result.add_argument('--'+name, type=Path)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
