"""Bind compact validation to the shared differentiable recurrent scalar."""
from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from threading import Lock

import numpy as np

_validator_cache_scope = ContextVar('orchidee_validator_cache_scope', default='')
_cache_hook_lock = Lock()
_cache_hook_installed = False


@contextmanager
def device_helper_cache(device):
    """Namespace device-bound helpers, preserving every ordinary cache key.

    JAX 0.4.38 can reuse GPU0's serialized single-device executable on GPU1.
    Its custom cache-key hook is read during compilation; ContextVar keeps the
    suffix local to this compilation thread, including during parallel replay.
    Merely disabling the cache locally is insufficient once is_cache_used has
    memoized the process-wide enabled state in that JAX release.
    """
    from jax._src import cache_key

    global _cache_hook_installed
    with _cache_hook_lock:
        if not _cache_hook_installed:
            previous = cache_key.custom_hook

            def scoped_hook():
                original = previous()
                scope = _validator_cache_scope.get()
                return original if not scope else original + '|' + scope

            cache_key.custom_hook = scoped_hook
            _cache_hook_installed = True
    # Retain the existing suffix to reuse already corrected validator entries.
    token = _validator_cache_scope.set(
        f'orchidee-split-validator-v1:{device.platform}:{device.process_index}:{device.id}')
    try:
        yield
    finally:
        _validator_cache_scope.reset(token)


# Compatibility for the earlier standalone validator regression probe.
_device_validator_cache = device_helper_cache


def bind_checked_backward(
    jax,
    jnp,
    component,
    recurrent_loss,
    runtime,
    head,
    scales,
    model,
    *,
    separate_validation=False,
):
    from research.daily_coarse_graining.daily_operator.fresh_start_training import (
        _canopy_change_plan,
        _litterpart_feedback_plan,
    )
    from research.daily_coarse_graining.daily_operator.recurrent_science_loss import (
        FLUX_KEYS,
        STOCK_KEYS,
        WEIGHT_COMPONENTS,
    )
    from research.daily_coarse_graining.daily_operator.recurrent_validation import (
        PHYSICAL_COUNT_NAMES,
        feedback_counts,
        physical_counts,
    )
    from scripts.hpc.run_trained_forward_screen import feedback_diagnostics

    validation_names = list(PHYSICAL_COUNT_NAMES)
    science_names = []
    for section in WEIGHT_COMPONENTS:
        keys = FLUX_KEYS if section in ('daily_flux', 'cumulative_flux') else STOCK_KEYS
        for key in keys:
            science_names.extend((f'{section}.{key}.invalid', f'{section}.{key}.empty'))
    if _canopy_change_plan(runtime, head.contract, model) is not None:
        validation_names.extend(('canopy_change_invalid', 'canopy_change_empty'))
    if _litterpart_feedback_plan(head, model['loss'],
            scaling=model.get('feedback_state_scaling')) is not None:
        validation_names.extend(('litterpart_feedback_invalid', 'litterpart_feedback_empty'))
    science_names = tuple(sorted(science_names))
    names = tuple(sorted((*validation_names, *science_names)))

    def science_counts(terms):
        hard = {}
        for section in WEIGHT_COMPONENTS:
            for key, term in getattr(terms, section).items():
                hard[section+'.'+key+'.invalid'] = (
                    term.invalid_prediction_count + term.invalid_target_count
                )
                hard[section+'.'+key+'.empty'] = term.count == 0
        if set(hard) != set(science_names):
            raise ValueError('compact recurrent science counter schema differs')
        return jnp.stack([
            jnp.asarray(hard[key], dtype=jnp.int64) for key in science_names
        ])

    def validation_counts(initial, observations, targets, expected_discrete, templates):
        hard = physical_counts(observations, targets, expected_discrete,
                               templates, head.contract, scales)
        first = jax.tree_util.tree_map(lambda value: value[0], templates)
        assembled = head.input_assembler.assemble(first)
        feedback = feedback_diagnostics(jax, jnp, runtime, head, model, initial,
                                        assembled, observations, targets, templates)
        hard.update(feedback_counts(feedback))
        if set(hard) != set(validation_names):
            raise ValueError('compact recurrent physical counter schema differs')
        return hard

    if separate_validation:
        def loss_with_aux(parameters, initial, observations, targets, context):
            terms = recurrent_loss.terms(parameters, initial, observations, targets, context)
            return terms.total, (observations, science_counts(terms))

        checked_loss = component.bind_checkpointed_loss_with_aux(loss_with_aux)

        def objective(parameters, initial, discrete, days, targets, context):
            return checked_loss(parameters, initial, discrete, days, targets, context)

        def validator(initial, observations, targets, expected_discrete, templates, counts):
            hard = validation_counts(
                initial, observations, targets, expected_discrete, templates
            )
            if counts.shape != (len(science_names),):
                raise ValueError('compact recurrent science counter vector shape differs')
            for index, key in enumerate(science_names):
                hard[key] = counts[index]
            if set(hard) != set(names):
                raise ValueError('compact recurrent validation counter schema differs')
            return jnp.stack([jnp.asarray(hard[key], dtype=jnp.int64) for key in names])

        backward = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))
        return backward, names, jax.jit(validator)

    def loss_with_aux(parameters, initial, observations, targets, context):
        templates = context['validation_templates']
        terms = recurrent_loss.terms(parameters, initial, observations, targets, context)
        hard = validation_counts(
            initial,
            observations,
            targets,
            context['validation_discrete'],
            templates,
        )
        compact_science = science_counts(terms)
        for index, key in enumerate(science_names):
            hard[key] = compact_science[index]
        if set(hard) != set(names):
            raise ValueError('compact recurrent validation counter schema differs')
        return terms.total, jnp.stack([
            jnp.asarray(hard[key], dtype=jnp.int64) for key in names
        ])

    checked_loss = component.bind_checkpointed_loss_with_aux(loss_with_aux)

    def objective(parameters, initial, discrete, days, targets, context, expected_discrete):
        return checked_loss(parameters, initial, discrete, days, targets,
                            dict(context, validation_templates=days[0],
                                 validation_discrete=expected_discrete))

    return jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True)), names, None


def _tree_bytes(jax, tree):
    return sum(
        int(leaf.size) * int(np.dtype(leaf.dtype).itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def wrap_separate_checked(jax, raw_executable, validator, emit=None, *, device=None):
    """Join a split backward and validator behind the checked seven-argument API."""
    from scripts.hpc.probe_one_day_batch import _compiled_memory

    if device is not None:
        # Captured validation constants can otherwise select the primary device
        # even when the raw backward and its outputs live on a peer device.
        placement = jax.sharding.SingleDeviceSharding(device)
        validator = jax.jit(validator, in_shardings=placement, out_shardings=placement)
    compiled_validator = None
    notify = emit if emit is not None else lambda stage: None
    statistics = {
        'observation_bytes': None,
        'raw_first_execution_seconds': None,
        'validator_compile_seconds': None,
        'validator_memory': None,
        'validator_compile_count': 0,
        'validator_cache_device': None if device is None else str(device),
    }

    def checked(parameters, initial, discrete, days, targets, context, expected_discrete):
        nonlocal compiled_validator
        arguments = (initial, discrete, days, targets, context, expected_discrete)
        initial, discrete, days, targets, context, expected_discrete = (
            jax.device_put(arguments) if device is None else jax.device_put(arguments, device)
        )
        first_execution = statistics['raw_first_execution_seconds'] is None
        began = time.perf_counter() if first_execution else None
        (value, (observations, science_counts)), gradient = raw_executable(
            parameters, initial, discrete, days, targets, context
        )
        if statistics['observation_bytes'] is None:
            statistics['observation_bytes'] = _tree_bytes(jax, observations)
        if first_execution:
            jax.block_until_ready((value, observations, science_counts, gradient))
            statistics['raw_first_execution_seconds'] = time.perf_counter() - began
            notify('raw_executed')
        if compiled_validator is None:
            notify('validator_compile')
            began = time.perf_counter()
            cache_scope = nullcontext() if device is None else _device_validator_cache(device)
            with cache_scope:
                compiled_validator = validator.lower(
                    initial, observations, targets, expected_discrete, days[0], science_counts
                ).compile()
            statistics['validator_compile_seconds'] = time.perf_counter() - began
            statistics['validator_memory'] = _compiled_memory(compiled_validator)
            statistics['validator_compile_count'] = 1
            notify('validator_compiled')
        counts = compiled_validator(
            initial, observations, targets, expected_discrete, days[0], science_counts
        )
        del observations, science_counts
        return (value, counts), gradient

    checked.statistics = statistics
    return checked


def decode_counts(names, values):
    """Decode the already transferred compact vector, retaining every gate."""
    array = np.asarray(values)
    if array.shape != (len(names),):
        raise ValueError('compact recurrent validation vector shape differs')
    return {key: int(value) for key, value in zip(names, array, strict=True)}
