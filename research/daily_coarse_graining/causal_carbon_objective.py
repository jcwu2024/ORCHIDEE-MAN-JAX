"""Causal carbon-interface objective for the frozen-base daily adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalPrediction,
    masked_huber_loss,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
    canonical_multistep_rollout,
)
from research.daily_coarse_graining.canonical_training import (
    CanonicalTrainingBatch,
    FastDayTargetRepresentation,
    _defined_numeric_mask_compiled,
)
from research.daily_coarse_graining.markov_dataset import TrainingStatistics

PFT14_INDEX = 13

FAST_INTERFACE_FIELDS = (
    ("daily_interface", "gpp_daily"),
    ("daily_interface", "resp_maint_part"),
    ("ok_leak", "carbon_32l"),
    ("ok_leak", "deepC_peat"),
)
PRIMARY_NEXT_STATE_FIELDS = (
    "npp_daily",
    "resp_growth",
    "resp_maint",
    "biomass",
    "lai",
    "carbon_32l",
    "deepC_peat",
)
FLUX_BIAS_FIELDS = (
    "npp_daily",
    "resp_growth",
    "resp_maint",
)
STOCK_TENDENCY_FIELDS = (
    "biomass",
    "lai",
    "carbon_32l",
    "deepC_peat",
)
NO_REGRESSION_GUARD_FIELDS = ("litter", "DOC")
SOURCE_NONNEGATIVE_STATE_FIELDS = ("carbon_32l", "DOC", "deepC_peat")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _pft_indices(
    leaf: Mapping[str, Any],
    *,
    pft_index: int,
    context: str,
) -> tuple[int, ...]:
    axes = tuple(str(value) for value in leaf.get("axis_names", ()))
    if "nvm" not in axes:
        raise ValueError(f"{context} has no PFT axis")
    selected = tuple(int(value) for value in leaf.get("selected_pft_indices", ()))
    if pft_index not in selected:
        raise ValueError(f"{context} does not carry PFT{pft_index + 1}")
    shape = tuple(int(value) for value in leaf["shape"])
    start = int(leaf["start"])
    stop = int(leaf["stop"])
    if int(np.prod(shape, dtype=np.int64)) != stop - start:
        raise ValueError(f"{context} shape does not match its compact range")
    local = np.arange(stop - start, dtype=np.int32).reshape(shape)
    indices = np.take(
        local,
        selected.index(pft_index),
        axis=axes.index("nvm"),
    ).reshape(-1)
    return tuple(int(index + start) for index in indices)


def _equal_field_weights(field_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(field_masks, dtype=bool)
    if masks.ndim != 2 or masks.shape[0] < 1:
        raise ValueError("field masks must be a nonempty field-by-column matrix")
    if np.any(np.sum(masks, axis=1) == 0):
        raise ValueError("every declared field must own at least one column")
    if np.any(np.sum(masks, axis=0) > 1):
        raise ValueError("declared fields must not overlap")
    weights = np.zeros((masks.shape[1],), dtype=np.float32)
    for mask in masks:
        weights[mask] = 1.0 / (masks.shape[0] * int(np.sum(mask)))
    return weights


@dataclass(frozen=True)
class CausalCarbonObjectiveLayout:
    fast_field_ids: tuple[str, ...]
    fast_field_masks: np.ndarray
    primary_state_field_ids: tuple[str, ...]
    primary_state_field_masks: np.ndarray
    flux_bias_field_ids: tuple[str, ...]
    flux_bias_field_masks: np.ndarray
    stock_tendency_field_ids: tuple[str, ...]
    stock_tendency_field_masks: np.ndarray
    guard_field_ids: tuple[str, ...]
    guard_field_masks: np.ndarray
    nonnegative_state_mask: np.ndarray
    fast_weights: np.ndarray
    primary_state_weights: np.ndarray
    metadata: Mapping[str, Any]
    sha256: str


def _leaf_record(
    leaf: Mapping[str, Any],
    *,
    indices: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "key": str(
            leaf.get("key")
            or ".".join(
                (
                    str(leaf.get("component") or leaf.get("family")),
                    *(str(value) for value in leaf["path"]),
                )
            )
        ),
        "compact_range": [int(leaf["start"]), int(leaf["stop"])],
        "pft14_indices": list(indices),
        "axis_names": [str(value) for value in leaf.get("axis_names", ())],
        "selected_pft_indices": [int(value) for value in leaf.get("selected_pft_indices", ())],
    }


def causal_carbon_objective_layout_from_contract(
    contract_metadata: Mapping[str, Any],
    *,
    pft_index: int = PFT14_INDEX,
) -> CausalCarbonObjectiveLayout:
    """Resolve all trainable interface and downstream PFT14 columns."""

    fast_width = int(contract_metadata["fast_day_target_width"])
    state_width = int(contract_metadata["continuous_state_width"])

    fast_by_key = {}
    for leaf in contract_metadata["fast_day_target_leaves"]:
        path = tuple(str(value) for value in leaf.get("path", ()))
        if not path:
            continue
        key = (str(leaf["family"]), path[0])
        if key not in FAST_INTERFACE_FIELDS:
            continue
        if key in fast_by_key:
            raise ValueError(f"duplicate fast interface field {key}")
        fast_by_key[key] = leaf
    missing_fast = tuple(key for key in FAST_INTERFACE_FIELDS if key not in fast_by_key)
    if missing_fast:
        raise ValueError(f"missing causal fast interface fields {missing_fast}")

    state_names = tuple(
        dict.fromkeys(
            (
                *PRIMARY_NEXT_STATE_FIELDS,
                *FLUX_BIAS_FIELDS,
                *STOCK_TENDENCY_FIELDS,
                *NO_REGRESSION_GUARD_FIELDS,
                *SOURCE_NONNEGATIVE_STATE_FIELDS,
            )
        )
    )
    state_by_name = {}
    for leaf in contract_metadata["state_leaves"]:
        path = tuple(str(value) for value in leaf.get("path", ()))
        if (
            str(leaf.get("component")) != "slowproc_stomate_previous_step_state"
            or not path
            or path[0] not in state_names
        ):
            continue
        if path[0] in state_by_name:
            raise ValueError(f"duplicate causal state field {path[0]}")
        state_by_name[path[0]] = leaf
    missing_state = tuple(name for name in state_names if name not in state_by_name)
    if missing_state:
        raise ValueError(f"missing causal downstream state fields {missing_state}")

    fast_masks = np.zeros((len(FAST_INTERFACE_FIELDS), fast_width), dtype=bool)
    fast_records = []
    for field_index, key in enumerate(FAST_INTERFACE_FIELDS):
        leaf = fast_by_key[key]
        indices = _pft_indices(
            leaf,
            pft_index=pft_index,
            context=f"{key[0]}.{key[1]}",
        )
        fast_masks[field_index, np.asarray(indices, dtype=np.int32)] = True
        fast_records.append(
            {
                "id": f"{key[0]}.{key[1]}",
                **_leaf_record(leaf, indices=indices),
            }
        )

    state_indices = {}
    state_records = {}
    for name in state_names:
        leaf = state_by_name[name]
        indices = _pft_indices(
            leaf,
            pft_index=pft_index,
            context=f"slowproc.{name}",
        )
        state_indices[name] = indices
        state_records[name] = {
            "id": name,
            **_leaf_record(leaf, indices=indices),
        }

    def masks(names: tuple[str, ...]) -> np.ndarray:
        result = np.zeros((len(names), state_width), dtype=bool)
        for field_index, name in enumerate(names):
            result[
                field_index,
                np.asarray(state_indices[name], dtype=np.int32),
            ] = True
        return result

    primary_masks = masks(PRIMARY_NEXT_STATE_FIELDS)
    flux_masks = masks(FLUX_BIAS_FIELDS)
    stock_masks = masks(STOCK_TENDENCY_FIELDS)
    guard_masks = masks(NO_REGRESSION_GUARD_FIELDS)
    nonnegative_mask = np.zeros((state_width,), dtype=bool)
    for name in SOURCE_NONNEGATIVE_STATE_FIELDS:
        nonnegative_mask[np.asarray(state_indices[name], dtype=np.int32)] = True

    metadata = {
        "schema_version": "causal_carbon_objective_layout_v1",
        "pft_index": pft_index,
        "pft_label": f"PFT{pft_index + 1}",
        "fast_day_target_width": fast_width,
        "continuous_state_width": state_width,
        "fast_interface_fields": fast_records,
        "primary_next_state_fields": [state_records[name] for name in PRIMARY_NEXT_STATE_FIELDS],
        "flux_bias_fields": [state_records[name] for name in FLUX_BIAS_FIELDS],
        "stock_tendency_fields": [state_records[name] for name in STOCK_TENDENCY_FIELDS],
        "no_regression_guard_fields": [state_records[name] for name in NO_REGRESSION_GUARD_FIELDS],
        "source_nonnegative_state_fields": [state_records[name] for name in SOURCE_NONNEGATIVE_STATE_FIELDS],
        "day_end_gpp_state_supervision": False,
        "reason_day_end_gpp_state_excluded": (
            "same-day GPP enters retained STOMATE through B_fast and the "
            "cross-day gpp_daily state is reset at the daily boundary"
        ),
    }
    return CausalCarbonObjectiveLayout(
        fast_field_ids=tuple(f"{family}.{field}" for family, field in FAST_INTERFACE_FIELDS),
        fast_field_masks=fast_masks,
        primary_state_field_ids=PRIMARY_NEXT_STATE_FIELDS,
        primary_state_field_masks=primary_masks,
        flux_bias_field_ids=FLUX_BIAS_FIELDS,
        flux_bias_field_masks=flux_masks,
        stock_tendency_field_ids=STOCK_TENDENCY_FIELDS,
        stock_tendency_field_masks=stock_masks,
        guard_field_ids=NO_REGRESSION_GUARD_FIELDS,
        guard_field_masks=guard_masks,
        nonnegative_state_mask=nonnegative_mask,
        fast_weights=_equal_field_weights(fast_masks),
        primary_state_weights=_equal_field_weights(primary_masks),
        metadata=metadata,
        sha256=_canonical_sha256(metadata),
    )


def _fieldwise_huber(error, finite, field_masks):
    error = jnp.asarray(error)
    finite = jnp.asarray(finite, dtype=bool)
    masks = jnp.asarray(field_masks, dtype=bool)
    absolute = jnp.abs(jnp.where(finite, error, 0.0))
    element = jnp.where(absolute <= 1.0, 0.5 * absolute * absolute, absolute - 0.5)
    selected = finite[..., None, :] & masks
    reduction_axes = tuple(range(selected.ndim - 2)) + (selected.ndim - 1,)
    denominator = jnp.sum(selected, axis=reduction_axes)
    values = jnp.sum(
        jnp.where(selected, element[..., None, :], 0.0),
        axis=reduction_axes,
    ) / jnp.maximum(denominator, 1)
    valid = denominator > 0
    return jnp.sum(jnp.where(valid, values, 0.0)) / jnp.maximum(jnp.sum(valid), 1)


def _fieldwise_signed_bias_huber(error, finite, field_masks):
    error = jnp.asarray(error)
    finite = jnp.asarray(finite, dtype=bool)
    masks = jnp.asarray(field_masks, dtype=bool)
    selected = finite[..., None, :] & masks
    reduction_axes = tuple(range(selected.ndim - 2)) + (selected.ndim - 1,)
    denominator = jnp.sum(selected, axis=reduction_axes)
    mean_error = jnp.sum(
        jnp.where(selected, error[..., None, :], 0.0),
        axis=reduction_axes,
    ) / jnp.maximum(denominator, 1)
    absolute = jnp.abs(mean_error)
    values = jnp.where(absolute <= 1.0, 0.5 * absolute * absolute, absolute - 0.5)
    valid = denominator > 0
    return jnp.sum(jnp.where(valid, values, 0.0)) / jnp.maximum(jnp.sum(valid), 1)


class CausalCarbonStateLosses(NamedTuple):
    L_next: Any
    L_rollout: Any
    L_flux_bias: Any
    L_stock_tendency_bias: Any
    L_guard: Any


def causal_carbon_state_losses(
    predicted_states,
    initial_states,
    teacher_states,
    teacher_next_states,
    *,
    statistics: TrainingStatistics,
    state_delta_scale,
    layout: CausalCarbonObjectiveLayout,
) -> CausalCarbonStateLosses:
    """Separate state accuracy from signed flux and stock-tendency bias."""

    predicted_states = jnp.asarray(predicted_states)
    initial_states = jnp.asarray(initial_states)
    teacher_states = jnp.asarray(teacher_states)
    teacher_next_states = jnp.asarray(teacher_next_states)
    predicted_previous = jnp.concatenate(
        (initial_states[:, None, :], predicted_states[:, :-1, :]),
        axis=1,
    )
    state_finite = _defined_numeric_mask_compiled(predicted_states) & _defined_numeric_mask_compiled(
        teacher_next_states
    )
    state_error = jnp.where(
        state_finite,
        (predicted_states - teacher_next_states) / jnp.asarray(statistics.arrays["state"].scale),
        0.0,
    )
    next_state = _fieldwise_huber(
        state_error[:, :1],
        state_finite[:, :1],
        layout.primary_state_field_masks,
    )
    rollout = (
        _fieldwise_huber(
            state_error[:, 1:],
            state_finite[:, 1:],
            layout.primary_state_field_masks,
        )
        if predicted_states.shape[1] > 1
        else jnp.asarray(0.0, dtype=state_error.dtype)
    )
    flux_bias = _fieldwise_signed_bias_huber(
        state_error,
        state_finite,
        layout.flux_bias_field_masks,
    )

    tendency_finite = (
        _defined_numeric_mask_compiled(predicted_states)
        & _defined_numeric_mask_compiled(predicted_previous)
        & _defined_numeric_mask_compiled(teacher_next_states)
        & _defined_numeric_mask_compiled(teacher_states)
    )
    tendency_error = jnp.where(
        tendency_finite,
        ((predicted_states - predicted_previous) - (teacher_next_states - teacher_states))
        / jnp.asarray(state_delta_scale),
        0.0,
    )
    stock_tendency_bias = _fieldwise_signed_bias_huber(
        tendency_error,
        tendency_finite,
        layout.stock_tendency_field_masks,
    )
    guard = _fieldwise_huber(
        state_error,
        state_finite,
        layout.guard_field_masks,
    )
    return CausalCarbonStateLosses(
        L_next=next_state,
        L_rollout=rollout,
        L_flux_bias=flux_bias,
        L_stock_tendency_bias=stock_tendency_bias,
        L_guard=guard,
    )


class CausalCarbonComponents(NamedTuple):
    L_interface: Any
    L_next: Any
    L_rollout: Any
    L_flux_bias: Any
    L_stock_tendency_bias: Any
    L_guard: Any
    unexpected_defined_status_mismatches: Any
    declared_dynamic_status_mismatches: Any
    discrete_state_mismatches: Any
    nonfinite_defined_values: Any
    negative_source_nonnegative_carbon_stocks: Any


class CausalCarbonLossResult(NamedTuple):
    loss: Any
    components: CausalCarbonComponents


def causal_carbon_interface_loss(
    parameters,
    batch: CanonicalTrainingBatch,
    *,
    layout: CausalCarbonObjectiveLayout,
    model_apply,
):
    prediction: CanonicalPrediction = model_apply(parameters, batch.model_input)
    return masked_huber_loss(
        prediction.normalized_fast_day_target,
        batch.normalized_fast_day_target,
        batch.fast_day_target_finite,
        layout.fast_weights,
    )


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


def causal_carbon_rollout_components(
    parameters,
    initial_states,
    initial_discrete_states: Mapping[str, Any],
    sequences: CanonicalMultistepSequence,
    *,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
    retained_tail_transition,
    state_delta_scale,
    layout: CausalCarbonObjectiveLayout,
    teacher_next_discrete_states: Mapping[str, Any] | None = None,
    rematerialize: bool = False,
    model_apply,
) -> CausalCarbonComponents:
    """Evaluate causal interface and downstream PFT14 carbon behavior."""

    def one(initial_state, initial_discrete_state, sequence):
        return canonical_multistep_rollout(
            parameters,
            initial_state,
            initial_discrete_state,
            sequence,
            statistics=statistics,
            representation=representation,
            fast_day_weights=layout.fast_weights,
            retained_tail_transition=retained_tail_transition,
            state_weights=layout.primary_state_weights,
            undefined_loss_weight=0.0,
            rematerialize=rematerialize,
            model_apply=model_apply,
        )

    results = jax.vmap(one)(initial_states, initial_discrete_states, sequences)
    steps = results.steps
    state_losses = causal_carbon_state_losses(
        steps.continuous_state,
        initial_states,
        sequences.teacher_state,
        sequences.teacher_next_state,
        statistics=statistics,
        state_delta_scale=state_delta_scale,
        layout=layout,
    )

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
    dynamic_state_mask = (
        jnp.zeros(
            (steps.continuous_state.shape[-1],),
            dtype=bool,
        )
        .at[jnp.asarray(dynamic_state_indices)]
        .set(True)
    )
    return CausalCarbonComponents(
        L_interface=jnp.mean(steps.fast_day_loss),
        L_next=state_losses.L_next,
        L_rollout=state_losses.L_rollout,
        L_flux_bias=state_losses.L_flux_bias,
        L_stock_tendency_bias=state_losses.L_stock_tendency_bias,
        L_guard=state_losses.L_guard,
        unexpected_defined_status_mismatches=jnp.sum(status_mismatch & ~dynamic_state_mask),
        declared_dynamic_status_mismatches=jnp.sum(status_mismatch & dynamic_state_mask),
        discrete_state_mismatches=_discrete_mismatches(
            steps.discrete_state,
            teacher_next_discrete_states,
        ),
        nonfinite_defined_values=jnp.sum(teacher_defined & ~jnp.isfinite(steps.continuous_state)),
        negative_source_nonnegative_carbon_stocks=jnp.sum(
            predicted_defined & jnp.asarray(layout.nonnegative_state_mask) & (steps.continuous_state < 0.0)
        ),
    )


def causal_carbon_candidate_loss(
    parameters,
    anchor_batch: CanonicalTrainingBatch,
    initial_states,
    initial_discrete_states: Mapping[str, Any],
    sequences: CanonicalMultistepSequence,
    *,
    coefficients: Mapping[str, Any],
    layout: CausalCarbonObjectiveLayout,
    model_apply,
    **kwargs,
) -> CausalCarbonLossResult:
    """Use the matched interface anchor plus candidate-only causal treatment."""

    anchor = causal_carbon_interface_loss(
        parameters,
        anchor_batch,
        layout=layout,
        model_apply=model_apply,
    )
    components = causal_carbon_rollout_components(
        parameters,
        initial_states,
        initial_discrete_states,
        sequences,
        layout=layout,
        model_apply=model_apply,
        **kwargs,
    )._replace(L_interface=anchor)
    total = anchor + sum(
        jnp.asarray(coefficients[name]) * getattr(components, name)
        for name in (
            "L_next",
            "L_rollout",
            "L_flux_bias",
            "L_stock_tendency_bias",
        )
    )
    return CausalCarbonLossResult(loss=total, components=components)
