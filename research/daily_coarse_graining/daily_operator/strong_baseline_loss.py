"""Expert-balanced process fidelity loss for the approved strong baseline."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp

from .neural_architecture import NeuralOutputSchema
from .strong_baseline_contract import StrongBaselineContract
from .successor_fast_owner_training import successor_fast_owner_loss
from .training_data import MaskedProcessTarget


class StrongBaselineProcessLossResult(NamedTuple):
    total: Any
    expert_mse: Mapping[str, Any]
    expert_label_count: Mapping[str, Any]
    label_mse: Mapping[str, Any]
    label_count: Mapping[str, Any]
    field_mse: Mapping[str, Any]
    field_count: Mapping[str, Any]
    defined_bce: Mapping[str, Any]
    defined_count: Mapping[str, Any]


def masked_equal_expert_label_macro_mse(
    predicted: Mapping[str, Any],
    targets: Mapping[str, MaskedProcessTarget],
    *,
    inputs: Any,
    schema: NeuralOutputSchema,
    scales: Mapping[str, Any],
    contract: StrongBaselineContract,
) -> StrongBaselineProcessLossResult:
    """Average valid elements by label, labels by expert, then experts equally."""

    base = successor_fast_owner_loss(
        predicted,
        targets,
        inputs=inputs,
        schema=schema,
        scales=scales,
    )
    observed_labels = set(base.label_loss)
    ownership = contract.label_ownership
    if observed_labels != set(ownership):
        missing = sorted(set(ownership) - observed_labels)
        extra = sorted(observed_labels - set(ownership))
        raise ValueError(f"strong-baseline loss label registry drift; missing={missing}, extra={extra}")

    expert_mse: dict[str, Any] = {}
    expert_label_count: dict[str, Any] = {}
    for expert in contract.expert_names:
        labels = tuple(label for label, owner in ownership.items() if owner == expert)
        active = jnp.stack(tuple(base.label_count[label] > 0.0 for label in labels))
        values = jnp.stack(tuple(base.label_loss[label] for label in labels))
        count = jnp.sum(active, dtype=jnp.float64)
        expert_label_count[expert] = count
        expert_mse[expert] = jnp.where(
            count > 0.0,
            jnp.sum(jnp.where(active, values, 0.0), dtype=jnp.float64) / count,
            0.0,
        )

    active_experts = jnp.stack(tuple(expert_label_count[expert] > 0.0 for expert in contract.expert_names))
    expert_values = jnp.stack(tuple(expert_mse[expert] for expert in contract.expert_names))
    active_count = jnp.sum(active_experts, dtype=jnp.float64)
    total = jnp.where(
        active_count > 0.0,
        jnp.sum(jnp.where(active_experts, expert_values, 0.0), dtype=jnp.float64) / active_count,
        0.0,
    )
    return StrongBaselineProcessLossResult(
        total=total,
        expert_mse=expert_mse,
        expert_label_count=expert_label_count,
        label_mse=base.label_loss,
        label_count=base.label_count,
        field_mse=base.value_loss.field_mse,
        field_count=base.value_loss.field_count,
        defined_bce=base.defined_bce,
        defined_count=base.defined_count,
    )


__all__ = [
    "StrongBaselineProcessLossResult",
    "masked_equal_expert_label_macro_mse",
]
