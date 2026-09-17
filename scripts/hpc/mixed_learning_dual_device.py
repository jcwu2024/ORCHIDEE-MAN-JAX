"""Two-device replay of the admitted mixed update; no derivative implementation."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc.checked_recurrent_execution import (
    decode_counts,
    device_helper_cache,
    wrap_separate_checked,
)
from scripts.hpc.mixed_learning_profiling import _parity, _summary

DUAL_DEVICE_TRAINING_LAYOUT = dict(
    schema_version='mixed_learning_dual_device_training_v1',
    authoritative_parameters='device_0',
    authoritative_optimizer='device_0',
    daily_backward='device_0',
    recurrent_backward='device_1',
    gradient_merge_and_adamw='device_0',
    peer_parameter_synchronization='after_every_update',
)


def split_components(daily, recurrent, *, executor=None):
    """Run independent pre-update branches, joining before any optimizer step."""
    if executor is None:
        return daily(), recurrent()
    future = executor.submit(recurrent)
    try:
        result = daily()
    except BaseException:
        # Drain launched work before callers release device buffers or retry.
        try:
            future.result()
        except BaseException:
            pass
        raise
    return result, future.result()


class ContinuousDualDeviceTrainer:
    """Execute changing daily/recurrent updates with one authoritative device.

    Branch callbacks receive ``(parameters, payload)`` and return dictionaries
    containing ``gradient``, ``loss``, ``norm``, ``hard`` and ``seconds``.
    The recurrent callback returns ``(rows_by_horizon, total_seconds)``. Raw
    model executables are compiled by the caller; this class namespaces only
    the smaller device-specific helpers and their executions.
    """

    def __init__(self, *, jax, jnp, parameters, optimizer, daily_branch,
                 recurrent_branch, combine, update, finite, devices=None,
                 require_gpus=True):
        selected = tuple(jax.local_devices() if devices is None else devices)
        if len(selected) != 2:
            raise ValueError('dual-device training requires exactly two devices')
        if require_gpus and any(device.platform != 'gpu' for device in selected):
            raise ValueError('dual-device training requires exactly two visible GPUs')
        self.jax, self.jnp = jax, jnp
        self.primary, self.secondary = selected
        self.daily_branch, self.recurrent_branch = daily_branch, recurrent_branch
        self.combine, self.update, self.finite = combine, update, finite
        with jax.default_device(self.primary), device_helper_cache(self.primary):
            self.parameters, self.optimizer = jax.device_put(
                (parameters, optimizer), self.primary)
            jax.block_until_ready((self.parameters, self.optimizer))
        with jax.default_device(self.secondary), device_helper_cache(self.secondary):
            self.peer_parameters = jax.device_put(self.parameters, self.secondary)
            jax.block_until_ready(self.peer_parameters)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='recurrent_device')

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _daily(self, payload):
        with (self.jax.default_device(self.primary),
              device_helper_cache(self.primary)):
            return self.daily_branch(self.parameters, payload)

    def _recurrent(self, payload):
        with (self.jax.default_device(self.secondary),
              device_helper_cache(self.secondary)):
            return self.recurrent_branch(self.peer_parameters, payload)

    def step(self, *, daily_payload, recurrent_payload=None, weights=None,
             expected_optimizer_step=None, persist=True):
        """Run one daily-only or mixed update and optionally retain its state."""
        if self._executor is None:
            raise RuntimeError('dual-device trainer is closed')
        began = time.perf_counter()
        if recurrent_payload is None:
            daily, recurrent, recurrent_seconds = self._daily(daily_payload), {}, 0.
        else:
            daily, (recurrent, recurrent_seconds) = split_components(
                lambda: self._daily(daily_payload),
                lambda: self._recurrent(recurrent_payload),
                executor=self._executor,
            )
        branch_elapsed = time.perf_counter()-began
        weights = {} if weights is None else weights
        merge_started = time.perf_counter()
        losses = dict(daily=float(daily['loss']))
        norms = dict(daily=float(daily['norm']))
        hard = dict(daily=daily['hard'])

        def any_hard(value):
            if isinstance(value, dict):
                return any(any_hard(item) for item in value.values())
            return bool(value)

        if any_hard(hard):
            raise ValueError('daily branch returned nonzero physical counters')
        with (self.jax.default_device(self.primary),
              device_helper_cache(self.primary)):
            mixed = daily['gradient']
            for horizon in (7, 30):
                if horizon not in recurrent:
                    continue
                row = recurrent[horizon]
                transferred = self.jax.device_put(row['gradient'], self.primary)
                mixed = self.combine(mixed, transferred, 1., weights[f'T{horizon}'])
                self.jax.block_until_ready(mixed)
                losses[f'T{horizon}'] = float(row['loss'])
                norms[f'T{horizon}'] = float(row['norm'])
                hard[f'T{horizon}'] = row['hard']
                if any_hard(row['hard']):
                    raise ValueError(
                        f'T{horizon} branch returned nonzero physical counters')
            if (not all(np.isfinite(value) and value > 0 for value in norms.values())
                    or not all(np.isfinite(value) for value in losses.values())
                    or not bool(self.finite(mixed))):
                raise ValueError('component/mixed gradient is nonfinite or empty')
            proposal, next_optimizer, gradient_norm, clip = self.update(
                self.parameters, mixed, self.optimizer)
            self.jax.block_until_ready(
                (proposal, next_optimizer, gradient_norm, clip))
            expected = (int(self.optimizer.step)+1 if expected_optimizer_step is None
                        else expected_optimizer_step)
            if (not bool(self.finite((proposal, next_optimizer)))
                    or not np.isfinite(float(gradient_norm))
                    or not 0 < float(clip) <= 1
                    or int(next_optimizer.step) != expected):
                raise ValueError('complete AdamW proposal invalid')
        merge_seconds = time.perf_counter()-merge_started
        sync_started = time.perf_counter()
        with (self.jax.default_device(self.secondary),
              device_helper_cache(self.secondary)):
            peer_proposal = self.jax.device_put(proposal, self.secondary)
            self.jax.block_until_ready(peer_proposal)
        synchronize_seconds = time.perf_counter()-sync_started
        if persist:
            self.parameters, self.optimizer = proposal, next_optimizer
            self.peer_parameters = peer_proposal
        mixed_loss = losses['daily'] + sum(
            weights[f'T{horizon}']*losses[f'T{horizon}']
            for horizon in (7, 30) if f'T{horizon}' in losses)
        record = dict(
            losses=losses,
            gradient_norms=norms,
            mixed_loss=mixed_loss,
            combined_gradient_norm=float(gradient_norm),
            clip=float(clip),
            optimizer_step=int(next_optimizer.step),
            all_hard_counts_zero=not any_hard(hard),
            combined_finite=True,
            proposal_finite=True,
            seconds=time.perf_counter()-began,
            daily_branch_seconds=float(daily['seconds']),
            recurrent_branch_seconds=float(recurrent_seconds),
            branch_elapsed_seconds=branch_elapsed,
            merge_checks_optimizer_seconds=merge_seconds,
            weight_synchronize_seconds=synchronize_seconds,
            synchronized_parameter_bytes=sum(
                int(leaf.size)*int(np.dtype(leaf.dtype).itemsize)
                for leaf in self.jax.tree_util.tree_leaves(peer_proposal)),
            peer_synchronized_optimizer_step=int(next_optimizer.step),
            hard=hard,
        )
        return record, dict(
            gradient=mixed, parameters=proposal, optimizer=next_optimizer,
            peer_parameters=peer_proposal)


def cost_comparison(serial_seconds, dual_seconds, *, cpu_cores=8):
    if not (np.isfinite(serial_seconds) and np.isfinite(dual_seconds)
            and serial_seconds > 0 and dual_seconds > 0):
        raise ValueError('timings must be finite and positive')
    speedup = serial_seconds / dual_seconds
    serial_rate, dual_rate = 2.20 + .07 * cpu_cores, 4.40 + .07 * cpu_cores
    return dict(speedup=speedup, time_reduction=1.-1./speedup,
                gpu_hour_ratio=2./speedup,
                estimated_total_charge_ratio=dual_rate/serial_rate/speedup,
                serial_yuan_per_hour=serial_rate, dual_yuan_per_hour=dual_rate,
                assumed_cpu_cores_per_run=cpu_cores,
                assumed_cpu_gpu_charges_additive=True)


def profile_dual_device(*, jax, jnp, parameters, optimizer, sample, daily_rows,
                        recurrent_rows, window_pool, daily_backward,
                        checked_backward, checked_validator, checked_counter_names,
                        coefficients, context, update, repeats, emit, output,
                        compiled_memory_limit_gib):
    """Measure serial/parallel D64+T7+T30, retaining only a diagnostic reference.

    GPU0 owns daily accumulation and AdamW. GPU1 owns the two recurrent
    evaluations. Gradients merge in exactly the serial order; updated weights
    are sent back to GPU1 inside each timed dual replay. All proposals discarded.
    """
    from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
    from research.daily_coarse_graining.daily_operator.one_day_training_state import (
        _atomic_pickle,
        make_gradient_accumulator,
    )
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
    from scripts.hpc.probe_public_rollout import gradient_comparison

    devices = jax.local_devices()
    if len(devices) != 2 or any(d.platform != 'gpu' for d in devices):
        raise ValueError('real-model dual profiling requires exactly two visible GPUs')
    if len(daily_rows) != 64 or repeats not in (3, 5) or checked_validator is None:
        raise ValueError('dual profile requires daily64, 3/5 repeats and checked split')
    primary, secondary = devices
    parameters, optimizer, coefficients = jax.device_put(
        (parameters, optimizer, coefficients), primary)
    parameters_secondary = jax.device_put(parameters, secondary)
    jax.block_until_ready(parameters_secondary)
    add, finish = make_gradient_accumulator(jax, jnp)
    combine = make_mixed_gradient(jax, jnp)
    finite = jax.jit(lambda tree: jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(tree)])))
    norm = jax.jit(lambda tree: jnp.sqrt(sum(
        jnp.sum(jnp.square(v.astype(jnp.float64)))
        for v in jax.tree_util.tree_leaves(tree))))
    compiles, checked = {}, {}

    def compile_function(name, function, arguments):
        emit('dual_compile_'+name)
        began = time.perf_counter()
        executable = function.lower(*arguments).compile()
        memory = _compiled_memory(executable)
        compiles[name] = dict(seconds=time.perf_counter()-began, memory=memory)
        emit('dual_compiled_'+name, dual_compilation=compiles)
        if memory['estimated_live_bytes'] > compiled_memory_limit_gib*1024**3:
            raise MemoryError(name+' exceeds admitted compiled memory bound')
        return executable

    with jax.default_device(primary):
        with device_helper_cache(primary):
            first = jax.device_put(sample(daily_rows[:8]), primary)
        # Preserve the admitted, expensive raw-program cache identities.
        daily_executable = compile_function('daily_gpu0', daily_backward,
                                             (parameters, first, coefficients))
        del first

    def compile_recurrent(device):
        local_parameters = parameters if device == primary else parameters_secondary
        results = {}
        with jax.default_device(device):
            for horizon in (7, 30):
                with device_helper_cache(device):
                    case = jax.device_put(window_pool.case(jax, recurrent_rows[horizon]), device)
                    local_context = jax.device_put(context, device)
                name = f'T{horizon}_gpu{device.id}'
                raw = compile_function(name, checked_backward,
                                       (local_parameters, *case[:4], local_context))
                results[horizon] = wrap_separate_checked(
                    jax, raw, checked_validator, device=device,
                    emit=lambda stage, name=name: emit('dual_'+name+'_'+stage))
                del case
        checked[device.id] = results

    def daily_branch():
        began = time.perf_counter()
        with jax.default_device(primary), device_helper_cache(primary):
            total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
            loss, counts = 0., {}
            for start in range(0, 64, 8):
                batch = jax.device_put(sample(daily_rows[start:start+8]), primary)
                (value, auxiliary), gradient = daily_executable(parameters, batch, coefficients)
                hard = calibration._hard_summary(jax.device_get(auxiliary['hard']), batch_size=8)
                if not np.isfinite(float(value)) or not hard['all_zero']:
                    raise ValueError('dual-profile daily physical/finite failure')
                for key, count in hard['totals'].items():
                    counts[key] = counts.get(key, 0)+count
                total = add(total, gradient, 8)
                loss += float(value)/8
                del batch, auxiliary, gradient
            gradient = finish(total, 64)
            gradient_norm = float(norm(gradient))
        return dict(gradient=gradient, loss=loss, norm=gradient_norm, hard=counts,
                    seconds=time.perf_counter()-began)

    def recurrent_branch(device):
        began = time.perf_counter()
        local_parameters = parameters if device == primary else parameters_secondary
        results = {}
        with jax.default_device(device), device_helper_cache(device):
            local_context = jax.device_put(context, device)
            for horizon in (7, 30):
                started = time.perf_counter()
                case = jax.device_put(window_pool.case(jax, recurrent_rows[horizon]), device)
                executable = checked[device.id][horizon]
                (value, counts), gradient = executable(
                    local_parameters, *case[:4], local_context, case[4])
                compiles[f'T{horizon}_gpu{device.id}']['validation'] = dict(executable.statistics)
                if (executable.statistics['validator_memory']['estimated_live_bytes']
                        > compiled_memory_limit_gib*1024**3):
                    raise MemoryError('dual-profile validator exceeds admitted memory bound')
                value, counts = jax.device_get((value, counts))
                hard = decode_counts(checked_counter_names, counts)
                if not np.isfinite(value) or any(hard.values()):
                    raise ValueError(f'dual-profile T{horizon} physical/finite failure')
                results[horizon] = dict(gradient=gradient, loss=float(value), hard=hard,
                                       norm=float(norm(gradient)),
                                       seconds=time.perf_counter()-started)
                del case
        return results, time.perf_counter()-began

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='recurrent_gpu')

    def replay(mode):
        began = time.perf_counter()
        dual = mode == 'dual'
        device = secondary if dual else primary
        if mode == 'daily_only':
            daily, recurrent, recurrent_seconds = daily_branch(), {}, 0.
        else:
            daily, (recurrent, recurrent_seconds) = split_components(
                daily_branch, lambda: recurrent_branch(device),
                executor=pool if dual else None)
        branch_elapsed = time.perf_counter()-began
        merge_started = time.perf_counter()
        losses, norms = dict(daily=daily['loss']), dict(daily=daily['norm'])
        with jax.default_device(primary), device_helper_cache(primary):
            mixed = daily.pop('gradient')
            for horizon in (7, 30):
                if horizon not in recurrent:
                    continue
                row = recurrent[horizon]
                transferred = jax.device_put(row.pop('gradient'), primary)
                mixed = combine(mixed, transferred, 1., .5)
                jax.block_until_ready(mixed)
                losses[f'T{horizon}'], norms[f'T{horizon}'] = row['loss'], row['norm']
                del transferred
            if not bool(finite(mixed)) or not all(np.isfinite(v) and v > 0 for v in norms.values()):
                raise ValueError('dual-profile component/mixed gradient invalid')
            proposal, next_optimizer, gradient_norm, clip = update(parameters, mixed, optimizer)
            jax.block_until_ready((proposal, next_optimizer, gradient_norm, clip))
            if (not bool(finite((proposal, next_optimizer))) or not np.isfinite(float(gradient_norm))
                    or not 0 < float(clip) <= 1 or int(next_optimizer.step) != int(optimizer.step)+1):
                raise ValueError('dual-profile AdamW proposal invalid')
        merge_seconds = time.perf_counter()-merge_started
        sync_started = time.perf_counter()
        if dual:
            synchronized = jax.device_put(proposal, secondary)
            jax.block_until_ready(synchronized)
            del synchronized
        synchronize_seconds = time.perf_counter()-sync_started if dual else 0.
        elapsed = time.perf_counter()-began
        record = dict(losses=losses, gradient_norms=norms,
            mixed_loss=losses['daily']+.5*(losses.get('T7', 0.)+losses.get('T30', 0.)),
            combined_gradient_norm=float(gradient_norm), clip=float(clip),
            all_hard_counts_zero=True, combined_finite=True, proposal_finite=True,
            optimizer_step=int(next_optimizer.step), seconds=elapsed,
            daily_branch_seconds=daily['seconds'], recurrent_branch_seconds=recurrent_seconds,
            branch_elapsed_seconds=branch_elapsed,
            merge_checks_optimizer_seconds=merge_seconds,
            weight_synchronize_seconds=synchronize_seconds,
            hard=dict(daily=daily['hard'], **{f'T{h}':r['hard'] for h, r in recurrent.items()}))
        # Caller explicitly releases these; timed repeats must not retain proposals.
        return record, dict(gradient=mixed, parameters=proposal, optimizer=next_optimizer)

    try:
        compile_recurrent(primary)
        emit('dual_serial_reference')
        reference_record, reference = replay('serial')
        reference_host = jax.device_get(reference)
        reference_path = output.with_suffix('.serial-reference.pkl')
        _atomic_pickle(reference_path, dict(record=reference_record, **reference_host))
        del reference
        emit('dual_serial_reference_saved', dual_serial_reference=str(reference_path))
        compile_recurrent(secondary)
        emit('dual_candidate_reference')
        dual_record, candidate = replay('dual')
        comparisons = {key: gradient_comparison(jax, value, reference_host[key])
                       for key, value in candidate.items()}
        del candidate, reference_host
        scalar = _parity([reference_record], [dual_record])
        hard_equal = reference_record['hard'] == dual_record['hard']
        comparison = dict(trees=comparisons, scalars=scalar, physical_counts_equal=hard_equal,
                          passed=bool(all(v['passed'] for v in comparisons.values())
                                      and scalar['passed'] and hard_equal))
        emit('dual_numerical_comparison', dual_comparison=comparison)
        if not comparison['passed']:
            raise ValueError('dual-device complete update differs from serial reference')
        warm_record, warm_values = replay('daily_only')
        del warm_record, warm_values
        records = {mode: [] for mode in ('serial', 'dual', 'daily_only')}
        for repeat in range(repeats):
            order = ('serial', 'dual') if repeat % 2 == 0 else ('dual', 'serial')
            for mode in (*order, 'daily_only'):
                emit('dual_timing_'+mode, dual_repeat=repeat+1)
                record, values = replay(mode)
                records[mode].append(record)
                del values
        parity = _parity(records['serial'], records['dual'])
        if not parity['passed']:
            raise ValueError('warmed dual timings lost numerical parity')
        timing = {mode: {**_summary([r['seconds'] for r in rows]), 'records': rows}
                  for mode, rows in records.items()}
        cost = cost_comparison(timing['serial']['median_seconds'], timing['dual']['median_seconds'])
        return dict(schema_version='mixed_learning_dual_device_v1', status='completed',
                    numerical_comparison=comparison, timing_parity=parity, timing=timing,
                    cost=cost, compilation=compiles, reference=str(reference_path),
                    devices=[str(d) for d in devices],
                    memory={str(d): _device_memory(d) for d in devices},
                    same_data_and_weights=True, includes_updated_weight_transfer=True,
                    includes_host_reads_and_checks=True, persistent_parameter_updates=0,
                    rotating_data_throughput_validated=False)
    finally:
        pool.shutdown(wait=True)
