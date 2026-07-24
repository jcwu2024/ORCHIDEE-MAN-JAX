"""Differentiable recursive training path for the canonical daily operator."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

import jax
import jax.numpy as jnp

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelParameters,
    canonical_model_apply,
    masked_huber_loss,
)
from research.daily_coarse_graining.canonical_training import (
    FastDayTargetRepresentation,
    _defined_numeric_mask_compiled,
    _normalize_finite_compiled,
    prepare_canonical_inference_batch_compiled,
    restore_fast_day_inference_prediction_compiled,
)
from research.daily_coarse_graining.markov_dataset import TrainingStatistics


class CanonicalMultistepSequence(NamedTuple):
    """One contiguous Teacher trajectory with a fixed scan horizon."""

    forcing_native: Any
    parameters: Any
    landpoint_static: Any
    annual_conditions: Any
    year: Any
    day_index: Any
    teacher_fast_day_target: Any
    teacher_next_state: Any
    retained_tail_inputs: Any


class CanonicalRolloutCarry(NamedTuple):
    continuous_state: Any
    discrete_state: Mapping[str, Any]


class CanonicalRolloutSteps(NamedTuple):
    continuous_state: Any
    discrete_state: Mapping[str, Any]
    physical_fast_day_target: Any
    fast_day_loss: Any
    undefined_loss: Any
    state_loss: Any
    state_defined_mismatches: Any


class CanonicalMultistepResult(NamedTuple):
    loss: Any
    final_carry: CanonicalRolloutCarry
    steps: CanonicalRolloutSteps


RetainedTailTransition = Callable[
    [Any, Mapping[str, Any], Any, Any, Any, Any],
    tuple[Any, Mapping[str, Any]],
]


def _singleton_batch(day, state):
    return {
        "state": state[None, :],
        "forcing_native": day.forcing_native[None, ...],
        "parameters": day.parameters[None, ...],
        "landpoint_static": day.landpoint_static[None, ...],
        "annual_conditions": day.annual_conditions[None, ...],
        "year": jnp.asarray(day.year)[None],
        "day_index": jnp.asarray(day.day_index)[None],
    }


def canonical_multistep_rollout(
    parameters: CanonicalModelParameters,
    initial_state,
    initial_discrete_state: Mapping[str, Any],
    sequence: CanonicalMultistepSequence,
    *,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
    fast_day_weights,
    retained_tail_transition: RetainedTailTransition,
    state_loss_weight: float = 1.0,
    undefined_loss_weight: float = 0.1,
) -> CanonicalMultistepResult:
    """Roll out a 1/3/7-day chain and differentiate through every day boundary.

    ``retained_tail_transition`` owns the source-backed daily season/STOMATE
    update. It receives physical ``B_fast`` values and must return canonical
    ``S[d+1]`` without crossing a host/NumPy boundary.
    """

    state_weights = jnp.full(
        (jnp.asarray(initial_state).shape[-1],),
        1.0 / jnp.asarray(initial_state).shape[-1],
        dtype=jnp.float32,
    )
    dynamic_indices = jnp.asarray(
        representation.dynamic_undefined_indices, dtype=jnp.int32
    )

    def body(carry: CanonicalRolloutCarry, day):
        inference = prepare_canonical_inference_batch_compiled(
            _singleton_batch(day, carry.continuous_state),
            statistics,
            representation,
        )
        prediction = canonical_model_apply(parameters, inference.model_input)
        physical_fast_day = restore_fast_day_inference_prediction_compiled(
            prediction.normalized_fast_day_target,
            prediction.dynamic_undefined_flip_logits,
            inference,
            statistics,
            representation,
        )[0]
        next_state, next_discrete = retained_tail_transition(
            carry.continuous_state,
            carry.discrete_state,
            physical_fast_day,
            day.retained_tail_inputs,
            day.year,
            day.day_index,
        )

        normalized_fast_target, finite_fast_target = _normalize_finite_compiled(
            day.teacher_fast_day_target,
            statistics.arrays["fast_day_target"],
        )
        fast_day_loss = masked_huber_loss(
            prediction.normalized_fast_day_target,
            normalized_fast_target[None, :],
            finite_fast_target[None, :],
            fast_day_weights,
        )
        target_dynamic_undefined = jnp.take(
            ~_defined_numeric_mask_compiled(day.teacher_fast_day_target),
            dynamic_indices,
        )
        flip_target = jnp.logical_xor(
            inference.persistent_dynamic_undefined[0],
            target_dynamic_undefined,
        ).astype(jnp.float32)
        logits = prediction.dynamic_undefined_flip_logits[0]
        undefined_loss = jnp.mean(
            jax.nn.softplus(logits) - flip_target * logits
        )

        normalized_next, finite_next = _normalize_finite_compiled(
            next_state,
            statistics.arrays["state"],
        )
        normalized_teacher_next, finite_teacher_next = _normalize_finite_compiled(
            day.teacher_next_state,
            statistics.arrays["state"],
        )
        common = finite_next & finite_teacher_next
        state_loss = masked_huber_loss(
            normalized_next[None, :],
            normalized_teacher_next[None, :],
            common[None, :],
            state_weights,
        )
        next_carry = CanonicalRolloutCarry(next_state, next_discrete)
        outputs = CanonicalRolloutSteps(
            continuous_state=next_state,
            discrete_state=next_discrete,
            physical_fast_day_target=physical_fast_day,
            fast_day_loss=fast_day_loss,
            undefined_loss=undefined_loss,
            state_loss=state_loss,
            state_defined_mismatches=jnp.sum(finite_next != finite_teacher_next),
        )
        return next_carry, outputs

    final_carry, steps = jax.lax.scan(
        body,
        CanonicalRolloutCarry(initial_state, initial_discrete_state),
        sequence,
    )
    total = (
        steps.fast_day_loss
        + undefined_loss_weight * steps.undefined_loss
        + state_loss_weight * steps.state_loss
    )
    return CanonicalMultistepResult(
        loss=jnp.mean(total),
        final_carry=final_carry,
        steps=steps,
    )
