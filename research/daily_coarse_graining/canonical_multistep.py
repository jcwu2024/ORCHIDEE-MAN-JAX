"""Differentiable recursive training path for the canonical daily operator."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
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
    teacher_state: Any
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
    state_increment_loss: Any
    state_defined_mismatches: Any


class CanonicalMultistepResult(NamedTuple):
    loss: Any
    final_carry: CanonicalRolloutCarry
    steps: CanonicalRolloutSteps


RetainedTailTransition = Callable[
    [Any, Mapping[str, Any], Any, Any, Any, Any],
    tuple[Any, Mapping[str, Any]],
]


def canonical_pushforward_prefix(
    parameters: Any,
    initial_state,
    initial_discrete_state: Mapping[str, Any],
    sequence: CanonicalMultistepSequence,
    *,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
    retained_tail_transition: RetainedTailTransition,
    model_apply: Callable = canonical_model_apply,
) -> CanonicalRolloutCarry:
    """Roll out a model prefix and detach the terminal state from gradients."""

    def body(carry: CanonicalRolloutCarry, day):
        inference = prepare_canonical_inference_batch_compiled(
            _singleton_batch(day, carry.continuous_state),
            statistics,
            representation,
        )
        prediction = model_apply(parameters, inference.model_input)
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
        return CanonicalRolloutCarry(next_state, next_discrete), None

    final_carry, _ = jax.lax.scan(
        body,
        CanonicalRolloutCarry(initial_state, initial_discrete_state),
        sequence,
    )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, final_carry)


def sequence_from_markov_window(
    window: Mapping[str, Any],
    *,
    retained_tail_inputs,
) -> CanonicalMultistepSequence:
    """Adapt one verified contiguous shard window to the compiled scan."""

    required = (
        "forcing_native",
        "parameters",
        "landpoint_static",
        "annual_conditions",
        "year",
        "day_index",
        "state_trajectory",
        "teacher_fast_day_target",
        "teacher_next_state",
    )
    missing = tuple(name for name in required if name not in window)
    if missing:
        raise ValueError(f"Markov window is missing multistep fields {missing}")
    horizon = int(np.shape(window["day_index"])[0])
    per_day = tuple(name for name in required if name != "state_trajectory")
    if horizon < 1 or any(np.shape(window[name])[0] != horizon for name in per_day):
        raise ValueError("Markov window multistep fields have inconsistent horizons")
    if np.shape(window["state_trajectory"])[0] != horizon + 1:
        raise ValueError("Markov window state trajectory must include both boundaries")
    retained_leaves = jax.tree_util.tree_leaves(retained_tail_inputs)
    if any(np.shape(value)[0] != horizon for value in retained_leaves):
        raise ValueError("retained-tail inputs do not match the Markov window horizon")
    return CanonicalMultistepSequence(
        forcing_native=window["forcing_native"],
        parameters=window["parameters"],
        landpoint_static=window["landpoint_static"],
        annual_conditions=window["annual_conditions"],
        year=window["year"],
        day_index=window["day_index"],
        teacher_state=window["state_trajectory"][:-1],
        teacher_fast_day_target=window["teacher_fast_day_target"],
        teacher_next_state=window["teacher_next_state"],
        retained_tail_inputs=retained_tail_inputs,
    )


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
    parameters: Any,
    initial_state,
    initial_discrete_state: Mapping[str, Any],
    sequence: CanonicalMultistepSequence,
    *,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
    fast_day_weights,
    retained_tail_transition: RetainedTailTransition,
    state_weights=None,
    state_delta_scale=None,
    state_loss_weight: float = 1.0,
    state_increment_loss_weight: float = 0.0,
    undefined_loss_weight: float = 0.1,
    rematerialize: bool = False,
    model_apply: Callable = canonical_model_apply,
) -> CanonicalMultistepResult:
    """Roll out a fixed-horizon chain and differentiate through every boundary.

    ``retained_tail_transition`` owns the source-backed daily season/STOMATE
    update. It receives physical ``B_fast`` values and must return canonical
    ``S[d+1]`` without crossing a host/NumPy boundary.
    """

    state_width = jnp.asarray(initial_state).shape[-1]
    state_weights = (
        jnp.full((state_width,), 1.0 / state_width, dtype=jnp.float32)
        if state_weights is None
        else jnp.asarray(state_weights, dtype=jnp.float32)
    )
    if state_weights.shape != (state_width,):
        raise ValueError("state loss weights do not match canonical state width")
    use_increment_loss = state_increment_loss_weight != 0.0
    if use_increment_loss:
        if state_delta_scale is None:
            raise ValueError("state increment loss requires a state-delta scale")
        state_delta_scale = jnp.asarray(state_delta_scale)
        if state_delta_scale.shape != (state_width,):
            raise ValueError("state-delta scale does not match canonical state width")
    dynamic_indices = jnp.asarray(
        representation.dynamic_undefined_indices, dtype=jnp.int32
    )

    def body(carry: CanonicalRolloutCarry, day):
        inference = prepare_canonical_inference_batch_compiled(
            _singleton_batch(day, carry.continuous_state),
            statistics,
            representation,
        )
        prediction = model_apply(parameters, inference.model_input)
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
        if use_increment_loss:
            predicted_previous = carry.continuous_state
            teacher_previous = day.teacher_state
            predicted_delta = next_state - predicted_previous
            teacher_delta = day.teacher_next_state - teacher_previous
            increment_common = (
                _defined_numeric_mask_compiled(next_state)
                & _defined_numeric_mask_compiled(predicted_previous)
                & _defined_numeric_mask_compiled(day.teacher_next_state)
                & _defined_numeric_mask_compiled(teacher_previous)
            )
            normalized_predicted_delta = jnp.where(
                increment_common,
                predicted_delta / state_delta_scale,
                0.0,
            )
            normalized_teacher_delta = jnp.where(
                increment_common,
                teacher_delta / state_delta_scale,
                0.0,
            )
            state_increment_loss = masked_huber_loss(
                normalized_predicted_delta[None, :],
                normalized_teacher_delta[None, :],
                increment_common[None, :],
                state_weights,
            )
        else:
            state_increment_loss = jnp.asarray(0.0, dtype=state_loss.dtype)
        next_carry = CanonicalRolloutCarry(next_state, next_discrete)
        outputs = CanonicalRolloutSteps(
            continuous_state=next_state,
            discrete_state=next_discrete,
            physical_fast_day_target=physical_fast_day,
            fast_day_loss=fast_day_loss,
            undefined_loss=undefined_loss,
            state_loss=state_loss,
            state_increment_loss=state_increment_loss,
            state_defined_mismatches=jnp.sum(finite_next != finite_teacher_next),
        )
        return next_carry, outputs

    scan_body = jax.checkpoint(body) if rematerialize else body
    final_carry, steps = jax.lax.scan(
        scan_body,
        CanonicalRolloutCarry(initial_state, initial_discrete_state),
        sequence,
    )
    total = (
        steps.fast_day_loss
        + undefined_loss_weight * steps.undefined_loss
        + state_loss_weight * steps.state_loss
        + state_increment_loss_weight * steps.state_increment_loss
    )
    return CanonicalMultistepResult(
        loss=jnp.mean(total),
        final_carry=final_carry,
        steps=steps,
    )


def canonical_multistep_batch_loss(
    parameters: Any,
    initial_states,
    initial_discrete_states: Mapping[str, Any],
    sequences: CanonicalMultistepSequence,
    *,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
    fast_day_weights,
    retained_tail_transition: RetainedTailTransition,
    state_weights=None,
    state_delta_scale=None,
    state_loss_weight: float = 1.0,
    state_increment_loss_weight: float = 0.0,
    undefined_loss_weight: float = 0.1,
    rematerialize: bool = False,
    model_apply: Callable = canonical_model_apply,
):
    """Average recursive loss for a same-runtime batch of trajectory windows."""

    def one(initial_state, initial_discrete_state, sequence):
        return canonical_multistep_rollout(
            parameters,
            initial_state,
            initial_discrete_state,
            sequence,
            statistics=statistics,
            representation=representation,
            fast_day_weights=fast_day_weights,
            retained_tail_transition=retained_tail_transition,
            state_weights=state_weights,
            state_delta_scale=state_delta_scale,
            state_loss_weight=state_loss_weight,
            state_increment_loss_weight=state_increment_loss_weight,
            undefined_loss_weight=undefined_loss_weight,
            rematerialize=rematerialize,
            model_apply=model_apply,
        ).loss

    losses = jax.vmap(one)(
        initial_states,
        initial_discrete_states,
        sequences,
    )
    return jnp.mean(losses)
