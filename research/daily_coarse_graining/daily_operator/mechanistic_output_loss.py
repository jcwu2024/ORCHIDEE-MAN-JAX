"""One-day scientific loss for post-update mechanistic outputs."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .mechanistic_output_contract import MechanisticOutputContract
from .mechanistic_output_reconstruction import (
    CANOPY_ENDPOINT_AUX,
    CANOPY_ENDPOINT_SCALE_AUX,
    CANOPY_STATUS_LOGITS_AUX,
)
from .neural_architecture import NeuralOutputSchema
from .strong_baseline_contract import StrongBaselineContract
from .successor_fast_owner_decoder import (
    RVEGET_LOGIT_KEY,
    training_representation_outputs,
)
from .successor_fast_owner_training import successor_fast_owner_loss
from .training_data import MaskedProcessTarget


class MechanisticOutputLossResult(NamedTuple):
    total: Any
    expert_mse: Mapping[str, Any]
    expert_label_count: Mapping[str, Any]
    label_mse: Mapping[str, Any]
    label_count: Mapping[str, Any]
    field_mse: Mapping[str, Any]
    field_count: Mapping[str, Any]
    defined_bce: Mapping[str, Any]
    defined_count: Mapping[str, Any]
    exact_diagnostic_mse: Mapping[str, Any]
    exact_diagnostic_count: Mapping[str, Any]


def masked_mechanistic_output_loss(
    predicted_complete: Mapping[str, Any],
    targets: Mapping[str, MaskedProcessTarget],
    *,
    inputs: Any,
    complete_schema: NeuralOutputSchema,
    learned_schema: NeuralOutputSchema,
    scales: Mapping[str, Any],
    architecture_contract: StrongBaselineContract,
    mechanism_contract: MechanisticOutputContract,
    auxiliary_outputs: Mapping[str, Any] | None = None,
) -> MechanisticOutputLossResult:
    """Optimize learned fields while reporting exact fields as hard diagnostics."""

    complete_names = {field.name for field in complete_schema.fields}
    learned_names = {field.name for field in learned_schema.fields}
    if set(predicted_complete) != complete_names or set(targets) != complete_names:
        raise ValueError("mechanistic loss requires one complete 181-field boundary")
    if complete_names - learned_names != set(mechanism_contract.exact_fields):
        raise ValueError("mechanistic loss schema partition drift")

    canopy_names = set(mechanism_contract.canopy_transfer_fields)
    ordinary_fields = tuple(field for field in learned_schema.fields if field.name not in canopy_names)
    ordinary_schema = NeuralOutputSchema(
        fields=ordinary_fields,
        labels=learned_schema.labels,
        active_pft_indices=learned_schema.active_pft_indices,
    )
    ordinary_names = {field.name for field in ordinary_fields}
    learned_targets = {name: targets[name] for name in ordinary_names}
    learned_scales = {name: scales[name] for name in ordinary_names if name != RVEGET_LOGIT_KEY}
    base = successor_fast_owner_loss(
        {name: predicted_complete[name] for name in ordinary_names},
        learned_targets,
        inputs=inputs,
        schema=ordinary_schema,
        scales=learned_scales,
    )

    storage_out = targets["inventory.water.qsintveg.storage.outgoing_fraction"]
    storage_in = targets["inventory.water.qsintveg.storage.incoming_amount"]
    debt_out = targets["inventory.water.qsintveg.debt.outgoing_fraction"]
    debt_in = targets["inventory.water.qsintveg.debt.incoming_amount"]
    canopy_shape = tuple(jnp.asarray(storage_out.value).shape)
    canopy_size = int(np.prod(canopy_shape, dtype=np.int64))
    start_water = jnp.asarray(inputs.state.water, dtype=jnp.float64)
    start_storage = start_water[:canopy_size].reshape(canopy_shape)
    start_debt = start_water[canopy_size : 2 * canopy_size].reshape(canopy_shape)
    target_storage = start_storage * (1.0 - jnp.asarray(storage_out.value)) + jnp.asarray(storage_in.value)
    target_debt = start_debt * (1.0 - jnp.asarray(debt_out.value)) + jnp.asarray(debt_in.value)
    target_endpoint = target_storage - target_debt

    auxiliary = {} if auxiliary_outputs is None else auxiliary_outputs
    if CANOPY_ENDPOINT_AUX in auxiliary:
        predicted_endpoint = jnp.asarray(auxiliary[CANOPY_ENDPOINT_AUX], dtype=jnp.float64)
        endpoint_scale = jnp.asarray(auxiliary[CANOPY_ENDPOINT_SCALE_AUX], dtype=jnp.float64)
        status_logits = jnp.asarray(auxiliary[CANOPY_STATUS_LOGITS_AUX], dtype=jnp.float64)
    else:
        predicted_storage = start_storage * (
            1.0 - jnp.asarray(predicted_complete["inventory.water.qsintveg.storage.outgoing_fraction"])
        ) + jnp.asarray(predicted_complete["inventory.water.qsintveg.storage.incoming_amount"])
        predicted_debt = start_debt * (
            1.0 - jnp.asarray(predicted_complete["inventory.water.qsintveg.debt.outgoing_fraction"])
        ) + jnp.asarray(predicted_complete["inventory.water.qsintveg.debt.incoming_amount"])
        predicted_endpoint = predicted_storage - predicted_debt
        endpoint_scale = jnp.asarray(
            scales[mechanism_contract.canopy_debt_scale_field],
            dtype=jnp.float64,
        )
        status_logits = jnp.zeros((*predicted_endpoint.shape, 3), dtype=jnp.float64)
    endpoint_scale = jnp.where(
        jnp.isfinite(endpoint_scale) & (endpoint_scale > 0.0),
        endpoint_scale,
        1.0,
    )
    eligible = (
        jnp.asarray(storage_out.defined, dtype=jnp.bool_)
        & jnp.asarray(storage_in.defined, dtype=jnp.bool_)
        & jnp.asarray(debt_out.defined, dtype=jnp.bool_)
        & jnp.asarray(debt_in.defined, dtype=jnp.bool_)
    )
    vegetation = (
        jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
        & (jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)[:, 0] < 0.5)
    )
    eligible &= vegetation[None, :]
    endpoint_count = jnp.sum(eligible, dtype=jnp.float64)
    endpoint_squared = ((predicted_endpoint - target_endpoint) / endpoint_scale) ** 2
    endpoint_mse = jnp.where(
        endpoint_count > 0.0,
        jnp.sum(jnp.where(eligible, endpoint_squared, 0.0), dtype=jnp.float64) / endpoint_count,
        0.0,
    )
    status_target = jnp.where(
        target_endpoint > 1.0e-10,
        0,
        jnp.where(target_endpoint < -1.0e-10, 2, 1),
    )
    status_log_probability = jax.nn.log_softmax(status_logits, axis=-1)
    status_element = -jnp.take_along_axis(
        status_log_probability,
        status_target[..., None],
        axis=-1,
    )[..., 0]
    status_count = endpoint_count
    status_ce = jnp.where(
        status_count > 0.0,
        jnp.sum(jnp.where(eligible, status_element, 0.0), dtype=jnp.float64) / status_count,
        0.0,
    )
    target_one_hot = jax.nn.one_hot(status_target, 3, dtype=jnp.float64)
    status_binary = (
        jnp.maximum(status_logits, 0.0)
        - status_logits * target_one_hot
        + jnp.log1p(jnp.exp(-jnp.abs(status_logits)))
    )

    label_mse = dict(base.label_loss)
    label_count = dict(base.label_count)
    canopy_labels = {
        learned_schema.labels[field.label_index]
        for field in learned_schema.fields
        if field.name in canopy_names
    }
    if len(canopy_labels) != 1:
        raise ValueError("canopy endpoint coordinate must have one loss owner")
    canopy_label = next(iter(canopy_labels))
    label_mse[canopy_label] = 0.5 * (endpoint_mse + status_ce)
    label_count[canopy_label] = endpoint_count + status_count

    field_mse = dict(base.value_loss.field_mse)
    field_count = dict(base.value_loss.field_count)
    field_mse[mechanism_contract.canopy_endpoint_field] = endpoint_mse
    field_count[mechanism_contract.canopy_endpoint_field] = endpoint_count
    for index, status in enumerate(("storage", "zero", "debt")):
        name = mechanism_contract.canopy_status_fields[status]
        class_loss = jnp.where(
            status_count > 0.0,
            jnp.sum(jnp.where(eligible, status_binary[..., index], 0.0), dtype=jnp.float64)
            / status_count,
            0.0,
        )
        field_mse[name] = class_loss
        field_count[name] = status_count

    ownership = architecture_contract.label_ownership
    unknown = sorted(set(label_mse) - set(ownership))
    if unknown:
        raise ValueError(f"mechanistic loss labels lack expert ownership: {unknown}")

    expert_mse: dict[str, Any] = {}
    expert_label_count: dict[str, Any] = {}
    for expert in architecture_contract.expert_names:
        labels = tuple(label for label in label_mse if ownership[label] == expert)
        if not labels:
            expert_label_count[expert] = jnp.asarray(0.0, dtype=jnp.float64)
            expert_mse[expert] = jnp.asarray(0.0, dtype=jnp.float64)
            continue
        active = jnp.stack(tuple(label_count[label] > 0.0 for label in labels))
        values = jnp.stack(tuple(label_mse[label] for label in labels))
        count = jnp.sum(active, dtype=jnp.float64)
        expert_label_count[expert] = count
        expert_mse[expert] = jnp.where(
            count > 0.0,
            jnp.sum(jnp.where(active, values, 0.0), dtype=jnp.float64) / count,
            0.0,
        )
    active_experts = jnp.stack(tuple(expert_label_count[expert] > 0.0 for expert in architecture_contract.expert_names))
    expert_values = jnp.stack(tuple(expert_mse[expert] for expert in architecture_contract.expert_names))
    active_count = jnp.sum(active_experts, dtype=jnp.float64)
    total = jnp.where(
        active_count > 0.0,
        jnp.sum(jnp.where(active_experts, expert_values, 0.0), dtype=jnp.float64) / active_count,
        0.0,
    )

    represented = training_representation_outputs(
        predicted_complete,
        inputs,
        complete_schema,
    )
    exact_mse: dict[str, Any] = {}
    exact_count: dict[str, Any] = {}
    for name in mechanism_contract.exact_fields:
        predicted = jnp.asarray(represented[name], dtype=jnp.float64)
        target = targets[name]
        wanted = jnp.asarray(target.value, dtype=jnp.float64)
        defined = jnp.asarray(target.defined, dtype=jnp.bool_)
        scale = jnp.asarray(scales[name], dtype=jnp.float64)
        scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
        selected = defined & jnp.isfinite(predicted) & jnp.isfinite(wanted)
        count = jnp.sum(selected, dtype=jnp.float64)
        squared = ((predicted - wanted) / scale) ** 2
        exact_count[name] = count
        exact_mse[name] = jnp.where(
            count > 0.0,
            jnp.sum(jnp.where(selected, squared, 0.0), dtype=jnp.float64) / count,
            0.0,
        )
    return MechanisticOutputLossResult(
        total=total,
        expert_mse=expert_mse,
        expert_label_count=expert_label_count,
        label_mse=label_mse,
        label_count=label_count,
        field_mse=field_mse,
        field_count=field_count,
        defined_bce=base.defined_bce,
        defined_count=base.defined_count,
        exact_diagnostic_mse=exact_mse,
        exact_diagnostic_count=exact_count,
    )


__all__ = ["MechanisticOutputLossResult", "masked_mechanistic_output_loss"]
