"""Bind compact validation to the shared differentiable recurrent scalar."""
from __future__ import annotations

import numpy as np


def bind_checked_backward(jax, jnp, component, recurrent_loss, runtime, head, scales, model):
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

    names = list(PHYSICAL_COUNT_NAMES)
    for section in WEIGHT_COMPONENTS:
        keys = FLUX_KEYS if section in ('daily_flux', 'cumulative_flux') else STOCK_KEYS
        for key in keys:
            names.extend((f'{section}.{key}.invalid', f'{section}.{key}.empty'))
    if _canopy_change_plan(runtime, head.contract, model) is not None:
        names.extend(('canopy_change_invalid', 'canopy_change_empty'))
    if _litterpart_feedback_plan(head, model['loss'],
            scaling=model.get('feedback_state_scaling')) is not None:
        names.extend(('litterpart_feedback_invalid', 'litterpart_feedback_empty'))
    names = tuple(sorted(names))

    def loss_with_aux(parameters, initial, observations, targets, context):
        templates = context['validation_templates']
        # These are the identical shared terms used by the scalar-only route.
        terms = recurrent_loss.terms(parameters, initial, observations, targets, context)
        hard = physical_counts(observations, targets, context['validation_discrete'],
                               templates, head.contract, scales)
        for section in WEIGHT_COMPONENTS:
            for key, term in getattr(terms, section).items():
                hard[section+'.'+key+'.invalid'] = term.invalid_prediction_count + term.invalid_target_count
                hard[section+'.'+key+'.empty'] = term.count == 0
        first = jax.tree_util.tree_map(lambda value: value[0], templates)
        assembled = head.input_assembler.assemble(first)
        feedback = feedback_diagnostics(jax, jnp, runtime, head, model, initial,
                                        assembled, observations, targets, templates)
        hard.update(feedback_counts(feedback))
        if set(hard) != set(names):
            raise ValueError('compact recurrent validation counter schema differs')
        return terms.total, jnp.stack([jnp.asarray(hard[key], dtype=jnp.int64) for key in names])

    checked_loss = component.bind_checkpointed_loss_with_aux(loss_with_aux)

    def objective(parameters, initial, discrete, days, targets, context, expected_discrete):
        return checked_loss(parameters, initial, discrete, days, targets,
                            dict(context, validation_templates=days[0],
                                 validation_discrete=expected_discrete))

    return jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True)), names


def decode_counts(names, values):
    """Decode the already transferred compact vector, retaining every gate."""
    array = np.asarray(values)
    if array.shape != (len(names),):
        raise ValueError('compact recurrent validation vector shape differs')
    return {key: int(value) for key, value in zip(names, array, strict=True)}
