"""Differentiable loss decomposition for the frozen rollout-stability screen."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
    canonical_model_apply,
    canonical_one_step_loss,
    masked_huber_loss,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
    canonical_multistep_rollout,
)
from research.daily_coarse_graining.canonical_state_objective import (
    StateProcessWeighting,
)
from research.daily_coarse_graining.canonical_training import (
    CanonicalTrainingBatch,
    FastDayTargetRepresentation,
    _defined_numeric_mask_compiled,
)
from research.daily_coarse_graining.markov_dataset import TrainingStatistics

SCIENCE_FIELDS = (
    "gpp_daily",
    "npp_daily",
    "resp_growth",
    "resp_maint",
    "resp_hetero",
    "biomass",
    "lai",
    "height",
    "litter",
    "carbon_32l",
    "DOC",
    "deepC_peat",
)
SCIENCE_TENDENCY_FIELDS = (
    "biomass",
    "lai",
    "height",
    "litter",
    "carbon_32l",
    "DOC",
    "deepC_peat",
)
SOURCE_NONNEGATIVE_STATE_FIELDS = ("carbon_32l", "DOC", "deepC_peat")


@dataclass(frozen=True)
class ScienceObjectiveLayout:
    field_masks: np.ndarray
    tendency_field_mask: np.ndarray
    nonnegative_state_mask: np.ndarray
    metadata: Mapping[str, Any]
    sha256: str


class RolloutStabilityComponents(NamedTuple):
    L_fast: Any
    L_next: Any
    L_rollout: Any
    L_bias: Any
    L_science: Any
    unexpected_defined_status_mismatches: Any
    declared_dynamic_status_mismatches: Any
    discrete_state_mismatches: Any
    nonfinite_defined_values: Any
    negative_source_nonnegative_carbon_stocks: Any


class RolloutStabilityLossResult(NamedTuple):
    loss: Any
    components: RolloutStabilityComponents


def canonical_fast_day_anchor_loss(
    parameters: Any,
    batch: CanonicalTrainingBatch,
    *,
    fast_day_weights,
    undefined_loss_weight: float = 0.1,
    model_apply=canonical_model_apply,
):
    """Evaluate the matched one-step anchor shared by both experiment arms."""

    return canonical_one_step_loss(
        parameters,
        batch.model_input,
        normalized_fast_day_target=batch.normalized_fast_day_target,
        fast_day_target_finite=batch.fast_day_target_finite,
        fast_day_target_weights=fast_day_weights,
        dynamic_undefined_flip_target=batch.dynamic_undefined_flip_target,
        undefined_loss_weight=undefined_loss_weight,
        model_apply=model_apply,
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def science_objective_layout_from_contract(
    contract_metadata: Mapping[str, Any],
) -> ScienceObjectiveLayout:
    """Map every named science field to its exact canonical state columns."""

    width = int(contract_metadata["continuous_state_width"])
    field_masks = np.zeros((len(SCIENCE_FIELDS), width), dtype=bool)
    leaves_by_field: dict[str, list[dict[str, Any]]] = {
        name: [] for name in SCIENCE_FIELDS
    }
    for leaf in contract_metadata["state_leaves"]:
        if leaf["start"] is None:
            continue
        path = tuple(str(value) for value in leaf["path"])
        if (
            str(leaf["component"]) != "slowproc_stomate_previous_step_state"
            or not path
            or path[0] not in leaves_by_field
        ):
            continue
        start = int(leaf["start"])
        stop = int(leaf["stop"])
        if start < 0 or stop <= start or stop > width:
            raise ValueError(f"invalid science state leaf [{start}:{stop}]")
        field_index = SCIENCE_FIELDS.index(path[0])
        if np.any(field_masks[field_index, start:stop]):
            raise ValueError(f"overlapping science state leaf {path[0]}")
        field_masks[field_index, start:stop] = True
        leaves_by_field[path[0]].append(
            {
                "key": ".".join((str(leaf["component"]), *path)),
                "start": start,
                "stop": stop,
            }
        )
    missing = tuple(name for name, leaves in leaves_by_field.items() if not leaves)
    if missing:
        raise ValueError(f"canonical state is missing science fields {missing}")

    tendency_field_mask = np.asarray(
        [name in SCIENCE_TENDENCY_FIELDS for name in SCIENCE_FIELDS],
        dtype=bool,
    )
    nonnegative_state_mask = np.zeros((width,), dtype=bool)
    for name in SOURCE_NONNEGATIVE_STATE_FIELDS:
        nonnegative_state_mask |= field_masks[SCIENCE_FIELDS.index(name)]
    metadata = {
        "schema_version": "rollout_stability_science_layout_v1",
        "state_width": width,
        "fields": [
            {
                "id": name,
                "tendency_supervision": bool(tendency_field_mask[index]),
                "source_nonnegative": name in SOURCE_NONNEGATIVE_STATE_FIELDS,
                "leaves": leaves_by_field[name],
                "width": int(np.count_nonzero(field_masks[index])),
            }
            for index, name in enumerate(SCIENCE_FIELDS)
        ],
    }
    return ScienceObjectiveLayout(
        field_masks=field_masks,
        tendency_field_mask=tendency_field_mask,
        nonnegative_state_mask=nonnegative_state_mask,
        metadata=metadata,
        sha256=_canonical_sha256(metadata),
    )


def _fieldwise_huber(error, finite, field_masks):
    """Average one masked normalized Huber value per named field."""

    error = jnp.asarray(error)
    finite = jnp.asarray(finite, dtype=bool)
    masks = jnp.asarray(field_masks, dtype=bool)
    absolute = jnp.abs(jnp.where(finite, error, 0.0))
    element = jnp.where(
        absolute <= 1.0,
        0.5 * absolute * absolute,
        absolute - 0.5,
    )
    selected = finite[..., None, :] & masks
    axes = tuple(range(selected.ndim - 2)) + (selected.ndim - 1,)
    numerator = jnp.sum(jnp.where(selected, element[..., None, :], 0.0), axis=axes)
    denominator = jnp.sum(selected, axis=axes)
    valid = denominator > 0
    values = jnp.where(valid, numerator / jnp.maximum(denominator, 1), 0.0)
    return jnp.sum(values) / jnp.maximum(jnp.sum(valid), 1)


def _discrete_mismatches(predicted, teacher) -> Any:
    if teacher is None:
        return jnp.asarray(0, dtype=jnp.int32)
    predicted_leaves, predicted_tree = jax.tree_util.tree_flatten(predicted)
    teacher_leaves, teacher_tree = jax.tree_util.tree_flatten(teacher)
    if predicted_tree != teacher_tree:
        raise ValueError("predicted and Teacher discrete-state trees differ")
    return sum(
        jnp.sum(jnp.asarray(left) != jnp.asarray(right))
        for left, right in zip(predicted_leaves, teacher_leaves, strict=True)
    )


def rollout_stability_components(
    parameters: Any,
    initial_states,
    initial_discrete_states: Mapping[str, Any],
    sequences: CanonicalMultistepSequence,
    *,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
    fast_day_weights,
    retained_tail_transition,
    process_weighting: StateProcessWeighting,
    state_delta_scale,
    science_layout: ScienceObjectiveLayout,
    teacher_next_discrete_states: Mapping[str, Any] | None = None,
    undefined_loss_weight: float = 0.1,
    rematerialize: bool = False,
    model_apply=canonical_model_apply,
) -> RolloutStabilityComponents:
    """Return separately differentiable frozen loss components and hard counts."""

    state_weights = jnp.asarray(process_weighting.weights, dtype=jnp.float32)
    delta_scale = jnp.asarray(state_delta_scale)

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
            state_loss_weight=1.0,
            undefined_loss_weight=undefined_loss_weight,
            rematerialize=rematerialize,
            model_apply=model_apply,
        )

    results = jax.vmap(one)(initial_states, initial_discrete_states, sequences)
    steps = results.steps
    fast = jnp.mean(
        steps.fast_day_loss + undefined_loss_weight * steps.undefined_loss
    )
    next_state = jnp.mean(steps.state_loss[:, 0])
    rollout = (
        jnp.mean(steps.state_loss[:, 1:])
        if steps.state_loss.shape[1] > 1
        else jnp.asarray(0.0, dtype=steps.state_loss.dtype)
    )

    predicted_previous = jnp.concatenate(
        (jnp.asarray(initial_states)[:, None, :], steps.continuous_state[:, :-1, :]),
        axis=1,
    )
    predicted_delta = steps.continuous_state - predicted_previous
    teacher_delta = sequences.teacher_next_state - sequences.teacher_state
    tendency_finite = (
        _defined_numeric_mask_compiled(steps.continuous_state)
        & _defined_numeric_mask_compiled(predicted_previous)
        & _defined_numeric_mask_compiled(sequences.teacher_next_state)
        & _defined_numeric_mask_compiled(sequences.teacher_state)
    )
    tendency_error = jnp.where(
        tendency_finite,
        (predicted_delta - teacher_delta) / delta_scale,
        0.0,
    )
    tendency_count = jnp.sum(tendency_finite, axis=(0, 1))
    mean_tendency_error = jnp.sum(tendency_error, axis=(0, 1)) / jnp.maximum(
        tendency_count, 1
    )
    bias = masked_huber_loss(
        mean_tendency_error[None, :],
        jnp.zeros_like(mean_tendency_error)[None, :],
        (tendency_count > 0)[None, :],
        state_weights,
    )

    state_scale = jnp.asarray(statistics.arrays["state"].scale)
    state_finite = (
        _defined_numeric_mask_compiled(steps.continuous_state)
        & _defined_numeric_mask_compiled(sequences.teacher_next_state)
    )
    state_error = jnp.where(
        state_finite,
        (steps.continuous_state - sequences.teacher_next_state) / state_scale,
        0.0,
    )
    science_state = _fieldwise_huber(
        state_error,
        state_finite,
        science_layout.field_masks,
    )
    science_tendency_masks = np.asarray(science_layout.field_masks).copy()
    science_tendency_masks[~science_layout.tendency_field_mask] = False
    science_tendency = _fieldwise_huber(
        tendency_error,
        tendency_finite,
        science_tendency_masks,
    )
    science = 0.5 * (science_state + science_tendency)

    teacher_defined = _defined_numeric_mask_compiled(sequences.teacher_next_state)
    predicted_defined = _defined_numeric_mask_compiled(steps.continuous_state)
    status_mismatch = predicted_defined != teacher_defined
    dynamic_fast_indices = np.asarray(
        representation.dynamic_undefined_indices,
        dtype=np.int32,
    )
    dynamic_state_indices = np.asarray(
        representation.state_indices,
        dtype=np.int32,
    )[dynamic_fast_indices]
    if np.any(dynamic_state_indices < 0):
        raise ValueError("dynamic undefined output is missing its state owner")
    dynamic_state_mask = jnp.zeros(
        (steps.continuous_state.shape[-1],),
        dtype=bool,
    ).at[jnp.asarray(dynamic_state_indices)].set(True)
    unexpected_defined_mismatches = jnp.sum(
        status_mismatch & ~dynamic_state_mask
    )
    declared_dynamic_mismatches = jnp.sum(
        status_mismatch & dynamic_state_mask
    )
    nonfinite_defined = jnp.sum(
        teacher_defined & ~jnp.isfinite(steps.continuous_state)
    )
    nonnegative_mask = jnp.asarray(science_layout.nonnegative_state_mask)
    negative_stocks = jnp.sum(
        predicted_defined
        & nonnegative_mask
        & (steps.continuous_state < 0.0)
    )
    discrete_mismatches = _discrete_mismatches(
        steps.discrete_state,
        teacher_next_discrete_states,
    )
    return RolloutStabilityComponents(
        L_fast=fast,
        L_next=next_state,
        L_rollout=rollout,
        L_bias=bias,
        L_science=science,
        unexpected_defined_status_mismatches=unexpected_defined_mismatches,
        declared_dynamic_status_mismatches=declared_dynamic_mismatches,
        discrete_state_mismatches=discrete_mismatches,
        nonfinite_defined_values=nonfinite_defined,
        negative_source_nonnegative_carbon_stocks=negative_stocks,
    )


def rollout_stability_loss(
    parameters: Any,
    initial_states,
    initial_discrete_states: Mapping[str, Any],
    sequences: CanonicalMultistepSequence,
    *,
    coefficients: Mapping[str, Any],
    **kwargs,
) -> RolloutStabilityLossResult:
    components = rollout_stability_components(
        parameters,
        initial_states,
        initial_discrete_states,
        sequences,
        **kwargs,
    )
    total = components.L_fast + sum(
        jnp.asarray(coefficients[name]) * getattr(components, name)
        for name in ("L_next", "L_rollout", "L_bias", "L_science")
    )
    return RolloutStabilityLossResult(total, components)


def rollout_stability_candidate_loss(
    parameters: Any,
    anchor_batch: CanonicalTrainingBatch,
    initial_states,
    initial_discrete_states: Mapping[str, Any],
    sequences: CanonicalMultistepSequence,
    *,
    coefficients: Mapping[str, Any],
    fast_day_weights,
    undefined_loss_weight: float = 0.1,
    model_apply=canonical_model_apply,
    **kwargs,
) -> RolloutStabilityLossResult:
    """Combine one matched anchor with only the rollout-specific treatment.

    The rollout window's own fast-day loss is deliberately not added. The
    control and candidate therefore consume exactly the same one-step anchor,
    while ``L_next`` through ``L_science`` are the candidate treatment.
    """

    anchor = canonical_fast_day_anchor_loss(
        parameters,
        anchor_batch,
        fast_day_weights=fast_day_weights,
        undefined_loss_weight=undefined_loss_weight,
        model_apply=model_apply,
    )
    components = rollout_stability_components(
        parameters,
        initial_states,
        initial_discrete_states,
        sequences,
        fast_day_weights=fast_day_weights,
        undefined_loss_weight=undefined_loss_weight,
        model_apply=model_apply,
        **kwargs,
    )._replace(L_fast=anchor)
    total = anchor + sum(
        jnp.asarray(coefficients[name]) * getattr(components, name)
        for name in ("L_next", "L_rollout", "L_bias", "L_science")
    )
    return RolloutStabilityLossResult(total, components)
