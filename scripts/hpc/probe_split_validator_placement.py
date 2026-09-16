"""Tiny split-validator device-placement regression; no Teacher payload reads."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np

from scripts.hpc.check_daily_training_runtime import _atomic_json
from scripts.hpc.checked_recurrent_execution import _device_validator_cache, wrap_separate_checked
from scripts.hpc.probe_dual_device_runtime import parser


def probe(jax, jnp, devices, *, scope_raw_cache=False):
    """Exercise captured primary constants and lazy peer compilation in a thread."""
    primary, secondary = devices
    captured = jax.device_put(np.asarray([2., 3.], dtype=np.float64), primary)

    @jax.jit
    def raw(parameters, initial, discrete, days, targets, context):
        observations = parameters['gain'] * initial + jnp.sum(days[1], axis=0)
        value = jnp.sum(observations - targets) + context + discrete['phase']
        return (value, (observations, jnp.zeros(1, dtype=jnp.int64))), {'gain': initial}

    @jax.jit
    def validator(initial, observations, targets, expected, templates, counts):
        # Like real validation, close over constants initialized on primary.
        return jnp.stack((jnp.sum(observations * captured),
                          jnp.sum(expected - templates), jnp.sum(counts)))

    records = []

    def exercise(device, horizon, *, explicit):
        with jax.default_device(device):
            arguments = jax.device_put((
                {'gain': np.asarray([.5, .25])}, np.asarray([1., 2.]),
                {'phase': np.asarray(0, dtype=np.int32)},
                (np.zeros((horizon, 2)), np.ones((horizon, 2))),
                np.zeros(2), np.asarray(0.), np.zeros((horizon, 2)),
            ), device)
            # Only the forced-cache regression uses this: its synthetic raw
            # function is identical on both GPUs and hits the same JAX issue.
            # Isolate it so this harness tests the production validator wrapper.
            scope = _device_validator_cache(device) if scope_raw_cache else nullcontext()
            with scope:
                executable = raw.lower(*arguments[:-1]).compile()
            checked = wrap_separate_checked(
                jax, executable, validator, device=device if explicit else None)
            result = checked(*arguments)
            jax.block_until_ready(result)
            expected_counts = np.asarray([5. * horizon + 2.5, 0., 0.])
            np.testing.assert_allclose(np.asarray(result[0][1]), expected_counts,
                                       rtol=0., atol=0.)
            np.testing.assert_array_equal(np.asarray(result[1]['gain']), [1., 2.])
            if explicit:
                assert all(v.devices() == {device} for v in jax.tree_util.tree_leaves(result))
            changed = arguments[:-1] + (arguments[-1] + 1.,)
            changed_result = checked(*changed)
            jax.block_until_ready(changed_result)
            np.testing.assert_allclose(np.asarray(changed_result[0][1]),
                                       [expected_counts[0], 2. * horizon, 0.], rtol=0., atol=0.)
            assert checked.statistics['validator_compile_count'] == 1
            return dict(device=str(device), horizon=horizon, repeated_calls=2,
                        validator_compile_count=1, counts=expected_counts.tolist())

    # The negative control is diagnostic: runtime versions may differ. The
    # explicit path must work irrespective of whether the old bug reproduces.
    legacy_error = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            exercise(primary, 7, explicit=False)
            executor.submit(exercise, secondary, 7, explicit=False).result()
        except Exception as error:
            legacy_error = f'{type(error).__name__}: {error}'
        for horizon in (7, 30):
            records.append(exercise(primary, horizon, explicit=True))
            records.append(executor.submit(exercise, secondary, horizon, explicit=True).result())
    return dict(legacy_peer_error=legacy_error, explicit_device_checks=records)


def run(args):
    import jax
    import jax.numpy as jnp

    began = time.perf_counter()
    jax.config.update('jax_enable_x64', True)
    devices = jax.local_devices(backend=args.backend)
    if len(devices) != 2:
        raise ValueError('placement probe requires exactly two visible devices')
    result = probe(jax, jnp, devices)
    report = dict(schema_version='split_validator_placement_probe_v1', status='passed',
                  jax_version=jax.__version__, elapsed_seconds=time.perf_counter()-began,
                  teacher_payload_reads=0, real_model_acceptance=False, **result)
    _atomic_json(args.output, report)
    print(report, flush=True)
    return report


if __name__ == '__main__':
    run(parser().parse_args())
