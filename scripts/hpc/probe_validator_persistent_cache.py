"""Exercise tiny validators with real persistent-cache writes on two devices."""
from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts.hpc.check_daily_training_runtime import _atomic_json
from scripts.hpc.checked_recurrent_execution import _device_validator_cache
from scripts.hpc.probe_dual_device_runtime import parser
from scripts.hpc.probe_split_validator_placement import probe


def run(args):
    import jax
    import jax.numpy as jnp
    from jax._src import cache_key

    began = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Configure before ANY compilation. Never lower the production cache's
    # write threshold or delete its costly existing executables.
    cache = Path(tempfile.mkdtemp(prefix='validator-cache-', dir=args.output.parent))
    jax.config.update('jax_compilation_cache_dir', str(cache))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_enable_x64', True)
    devices = jax.local_devices(backend=args.backend)
    if len(devices) != 2:
        raise ValueError('cache probe requires exactly two visible devices')

    def validator(x):
        return jnp.stack((jnp.sum(jnp.sin(x)), jnp.sum(x > 0)))

    def evaluate(device, *, scoped):
        placement = jax.sharding.SingleDeviceSharding(device)
        arguments = jax.device_put(np.arange(64, dtype=np.float64), device)
        function = jax.jit(validator, in_shardings=placement, out_shardings=placement)
        with jax.default_device(device):
            if scoped:
                with _device_validator_cache(device):
                    executable = function.lower(arguments).compile()
            else:
                executable = function.lower(arguments).compile()
            result = executable(arguments)
            jax.block_until_ready(result)
        np.testing.assert_allclose(np.asarray(result), [np.sin(np.arange(64)).sum(), 63],
                                   rtol=1e-12, atol=1e-12)
        assert result.devices() == {device}
        return dict(device=str(device), result=np.asarray(result).tolist())

    baseline_hook = cache_key.custom_hook()
    with ThreadPoolExecutor(max_workers=1) as executor:
        primary = evaluate(devices[0], scoped=False)
        cache_files = sorted(p.name for p in cache.iterdir() if p.name.endswith('-cache'))
        if not cache_files:
            raise AssertionError('negative control did not create a persistent cache entry')
        try:
            executor.submit(evaluate, devices[1], scoped=False).result()
            legacy_error = None
        except Exception as error:
            legacy_error = f'{type(error).__name__}: {error}'
            if not ('Buffer passed to Execute()' in legacy_error
                    and 'replica is assigned' in legacy_error):
                raise
        fixed = []
        # Clear in-memory caches between rounds, preserving the files. Round 2
        # must deserialize the correctly isolated entries on each device.
        for repeat in range(2):
            jax.clear_caches()
            fixed.append(evaluate(devices[0], scoped=True))
            fixed.append(executor.submit(evaluate, devices[1], scoped=True).result())
        with _device_validator_cache(devices[0]):
            assert cache_key.custom_hook() != baseline_hook
            assert executor.submit(cache_key.custom_hook).result() == baseline_hook
        assert cache_key.custom_hook() == baseline_hook
    if args.require_reproduction and legacy_error is None:
        raise AssertionError('deployed regression must reproduce the old cache assignment error')
    print(dict(legacy_peer_error=legacy_error, fixed_device_checks=fixed), flush=True)
    split = probe(jax, jnp, devices, scope_raw_cache=True)
    report = dict(schema_version='validator_persistent_cache_probe_v1', status='passed',
                  jax_version=jax.__version__, elapsed_seconds=time.perf_counter()-began,
                  cache_directory=str(cache), initial_cache_entries=cache_files,
                  primary_control=primary, legacy_peer_error=legacy_error,
                  fixed_device_checks=fixed, thread_local_hook_restored=True,
                  split_wrapper=split, teacher_payload_reads=0, real_model_acceptance=False)
    _atomic_json(args.output, report)
    print(report, flush=True)
    return report


if __name__ == '__main__':
    argument_parser = parser()
    argument_parser.add_argument('--require-reproduction', action='store_true')
    run(argument_parser.parse_args())
