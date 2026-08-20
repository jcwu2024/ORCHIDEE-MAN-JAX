"""Branch-balanced loss for the DOC status and magnitude candidate."""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from .coordinate_coupled_doc_attention import (
    DOC_INCOMING_FIELD,
    DOC_OUTGOING_FIELD,
)
from .doc_status_branch_magnitude_attention import (
    DOC_INCOMING_RAW_AUX,
    DOC_OUTGOING_RAW_AUX,
    DOC_STATUS_LOGITS_AUX,
    DocStatusBranchMagnitudeConfig,
)
from .mechanistic_output_contract import MechanisticOutputContract
from .mechanistic_output_loss import (
    MechanisticOutputLossResult,
    masked_mechanistic_output_loss,
)
from .neural_architecture import NeuralOutputNormalization, NeuralOutputSchema
from .strong_baseline_contract import StrongBaselineContract
from .training_data import MaskedProcessTarget

DOC_STATUS_MACRO_CE = "__doc_loss.status_macro_ce"
DOC_INCOMING_COORDINATE_LOSS = "__doc_loss.incoming_coordinate_smooth_l1"
DOC_OUTGOING_COORDINATE_LOSS = "__doc_loss.outgoing_coordinate_smooth_l1"
DOC_PHYSICAL_STATUS_MACRO_MSE = "__doc_loss.physical_status_macro_mse"
DOC_COMPOSITE_LOSS = "__doc_loss.composite"


def _masked_mean(values: Any, selected: Any) -> tuple[Any, Any]:
    mask = jnp.asarray(selected, dtype=jnp.bool_)
    count = jnp.sum(mask, dtype=jnp.float64)
    mean = jnp.where(
        count > 0.0,
        jnp.sum(jnp.where(mask, values, 0.0), dtype=jnp.float64) / count,
        0.0,
    )
    return mean, count


def _equal_group_macro(values: Any, groups: tuple[Any, ...]) -> tuple[Any, Any]:
    summaries = tuple(_masked_mean(values, group) for group in groups)
    means = jnp.stack(tuple(summary[0] for summary in summaries))
    counts = jnp.stack(tuple(summary[1] for summary in summaries))
    active = counts > 0.0
    active_count = jnp.sum(active, dtype=jnp.float64)
    macro = jnp.where(
        active_count > 0.0,
        jnp.sum(jnp.where(active, means, 0.0), dtype=jnp.float64) / active_count,
        0.0,
    )
    return macro, jnp.sum(counts, dtype=jnp.float64)


def _smooth_l1(residual: Any, delta: float) -> Any:
    magnitude = jnp.abs(residual)
    return jnp.where(
        magnitude <= delta,
        0.5 * residual * residual / delta,
        magnitude - 0.5 * delta,
    )


def _inverse_softplus(value: Any) -> Any:
    return value + jnp.log(-jnp.expm1(-value))


def _recompute_expert_macro(
    label_mse: Mapping[str, Any],
    label_count: Mapping[str, Any],
    contract: StrongBaselineContract,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    ownership = contract.label_ownership
    expert_mse = {}
    expert_label_count = {}
    for expert in contract.expert_names:
        labels = tuple(label for label in label_mse if ownership[label] == expert)
        active = jnp.stack(tuple(label_count[label] > 0.0 for label in labels))
        values = jnp.stack(tuple(label_mse[label] for label in labels))
        count = jnp.sum(active, dtype=jnp.float64)
        expert_label_count[expert] = count
        expert_mse[expert] = jnp.where(
            count > 0.0,
            jnp.sum(jnp.where(active, values, 0.0), dtype=jnp.float64) / count,
            0.0,
        )
    active_experts = jnp.stack(
        tuple(expert_label_count[name] > 0.0 for name in contract.expert_names)
    )
    values = jnp.stack(tuple(expert_mse[name] for name in contract.expert_names))
    count = jnp.sum(active_experts, dtype=jnp.float64)
    total = jnp.where(
        count > 0.0,
        jnp.sum(jnp.where(active_experts, values, 0.0), dtype=jnp.float64) / count,
        0.0,
    )
    return expert_mse, expert_label_count, total


def masked_doc_status_branch_magnitude_loss(
    predicted_complete: Mapping[str, Any],
    targets: Mapping[str, MaskedProcessTarget],
    *,
    inputs: Any,
    complete_schema: NeuralOutputSchema,
    learned_schema: NeuralOutputSchema,
    scales: Mapping[str, Any],
    output_normalization: NeuralOutputNormalization,
    architecture_contract: StrongBaselineContract,
    mechanism_contract: MechanisticOutputContract,
    auxiliary_outputs: Mapping[str, Any],
    config: DocStatusBranchMagnitudeConfig = DocStatusBranchMagnitudeConfig(),
) -> MechanisticOutputLossResult:
    """Replace pooled DOC endpoint loss with status and branch-balanced terms."""

    required_auxiliary = {
        DOC_STATUS_LOGITS_AUX,
        DOC_INCOMING_RAW_AUX,
        DOC_OUTGOING_RAW_AUX,
    }
    if not required_auxiliary <= set(auxiliary_outputs):
        missing = sorted(required_auxiliary - set(auxiliary_outputs))
        raise ValueError(f"DOC status-magnitude loss lacks auxiliary outputs: {missing}")
    base = masked_mechanistic_output_loss(
        predicted_complete,
        targets,
        inputs=inputs,
        complete_schema=complete_schema,
        learned_schema=learned_schema,
        scales=scales,
        architecture_contract=architecture_contract,
        mechanism_contract=mechanism_contract,
        auxiliary_outputs=auxiliary_outputs,
    )
    fields = {field.name: field for field in learned_schema.fields}
    incoming_field = fields[DOC_INCOMING_FIELD]
    outgoing_field = fields[DOC_OUTGOING_FIELD]
    incoming_label = learned_schema.labels[incoming_field.label_index]
    outgoing_label = learned_schema.labels[outgoing_field.label_index]
    if incoming_label != outgoing_label:
        raise ValueError("DOC endpoint fields must share one loss label")

    incoming_target = targets[DOC_INCOMING_FIELD]
    outgoing_target = targets[DOC_OUTGOING_FIELD]
    incoming = jnp.asarray(incoming_target.value, dtype=jnp.float64)
    outgoing = jnp.asarray(outgoing_target.value, dtype=jnp.float64)
    predicted_incoming = jnp.asarray(
        predicted_complete[DOC_INCOMING_FIELD],
        dtype=jnp.float64,
    )
    predicted_outgoing = jnp.asarray(
        predicted_complete[DOC_OUTGOING_FIELD],
        dtype=jnp.float64,
    )
    status_logits = jnp.asarray(
        auxiliary_outputs[DOC_STATUS_LOGITS_AUX],
        dtype=jnp.float64,
    )
    incoming_raw = jnp.asarray(
        auxiliary_outputs[DOC_INCOMING_RAW_AUX],
        dtype=jnp.float64,
    )
    outgoing_raw = jnp.asarray(
        auxiliary_outputs[DOC_OUTGOING_RAW_AUX],
        dtype=jnp.float64,
    )
    if status_logits.shape != (*incoming.shape, 3):
        raise ValueError("DOC status logits do not match endpoint coordinates")
    if any(
        value.shape != incoming.shape
        for value in (
            outgoing,
            predicted_incoming,
            predicted_outgoing,
            incoming_raw,
            outgoing_raw,
        )
    ):
        raise ValueError("DOC status-magnitude loss shape drift")

    incoming_scale = jnp.broadcast_to(
        jnp.asarray(scales[DOC_INCOMING_FIELD], dtype=jnp.float64),
        incoming.shape,
    )
    outgoing_scale = jnp.broadcast_to(
        jnp.asarray(scales[DOC_OUTGOING_FIELD], dtype=jnp.float64),
        outgoing.shape,
    )
    valid_scale = (
        jnp.isfinite(incoming_scale)
        & (incoming_scale > 0.0)
        & jnp.isfinite(outgoing_scale)
        & (outgoing_scale > 0.0)
    )
    eligible = (
        jnp.asarray(incoming_target.defined, dtype=jnp.bool_)
        & jnp.asarray(outgoing_target.defined, dtype=jnp.bool_)
        & jnp.isfinite(incoming)
        & jnp.isfinite(outgoing)
        & valid_scale
    )
    incoming_active = eligible & (incoming > config.meaningful_magnitude)
    outgoing_active = eligible & (outgoing > config.meaningful_magnitude)
    unchanged = eligible & ~incoming_active & ~outgoing_active
    status_target = jnp.where(incoming_active, 0, jnp.where(outgoing_active, 2, 1))
    status_log_probability = jax.nn.log_softmax(status_logits, axis=-1)
    status_element = -jnp.take_along_axis(
        status_log_probability,
        status_target[..., None],
        axis=-1,
    )[..., 0]
    status_macro, status_count = _equal_group_macro(
        status_element,
        (incoming_active, unchanged, outgoing_active),
    )

    incoming_output_scale = jnp.broadcast_to(
        jnp.asarray(
            output_normalization.scale[DOC_INCOMING_FIELD],
            dtype=jnp.float64,
        ),
        incoming.shape,
    )
    incoming_output_scale = jnp.where(
        jnp.isfinite(incoming_output_scale) & (incoming_output_scale > 0.0),
        incoming_output_scale,
        1.0,
    )
    incoming_output_mean = jnp.broadcast_to(
        jnp.asarray(
            output_normalization.mean[DOC_INCOMING_FIELD],
            dtype=jnp.float64,
        ),
        incoming.shape,
    )
    incoming_center_ratio = jnp.maximum(
        incoming_output_mean / incoming_output_scale,
        config.minimum_nonnegative_ratio,
    )
    incoming_baseline = _inverse_softplus(incoming_center_ratio)
    safe_incoming_ratio = jnp.where(
        incoming_active,
        incoming / incoming_output_scale,
        1.0,
    )
    incoming_target_raw = _inverse_softplus(safe_incoming_ratio) - incoming_baseline
    incoming_coordinate_element = _smooth_l1(
        incoming_raw - incoming_target_raw,
        config.smooth_l1_delta,
    )
    incoming_coordinate_loss, incoming_count = _masked_mean(
        incoming_coordinate_element,
        incoming_active,
    )

    outgoing_output_mean = jnp.broadcast_to(
        jnp.asarray(
            output_normalization.mean[DOC_OUTGOING_FIELD],
            dtype=jnp.float64,
        ),
        outgoing.shape,
    )
    outgoing_center = jnp.clip(
        outgoing_output_mean,
        config.minimum_unit_interval_center,
        1.0 - config.minimum_unit_interval_center,
    )
    outgoing_baseline = jnp.log(outgoing_center) - jnp.log1p(-outgoing_center)
    safe_outgoing = jnp.clip(
        outgoing,
        config.minimum_unit_interval_center,
        1.0 - config.minimum_unit_interval_center,
    )
    outgoing_target_raw = (
        jnp.log(safe_outgoing) - jnp.log1p(-safe_outgoing) - outgoing_baseline
    )
    outgoing_coordinate_element = _smooth_l1(
        outgoing_raw - outgoing_target_raw,
        config.smooth_l1_delta,
    )
    outgoing_coordinate_loss, outgoing_count = _masked_mean(
        outgoing_coordinate_element,
        outgoing_active,
    )

    physical_element = 0.5 * (
        ((predicted_incoming - incoming) / incoming_scale) ** 2
        + ((predicted_outgoing - outgoing) / outgoing_scale) ** 2
    )
    physical_macro, physical_count = _equal_group_macro(
        physical_element,
        (incoming_active, unchanged, outgoing_active),
    )
    doc_composite = 0.25 * (
        status_macro
        + incoming_coordinate_loss
        + outgoing_coordinate_loss
        + physical_macro
    )

    label_mse = dict(base.label_mse)
    label_count = dict(base.label_count)
    field_mse = dict(base.field_mse)
    field_count = dict(base.field_count)
    doc_sse = sum(
        field_mse[name] * field_count[name]
        for name in (DOC_INCOMING_FIELD, DOC_OUTGOING_FIELD)
    )
    doc_count = sum(
        field_count[name]
        for name in (DOC_INCOMING_FIELD, DOC_OUTGOING_FIELD)
    )
    other_count = jnp.maximum(label_count[incoming_label] - doc_count, 0.0)
    other_sse = jnp.maximum(
        label_mse[incoming_label] * label_count[incoming_label] - doc_sse,
        0.0,
    )
    other_mse = jnp.where(other_count > 0.0, other_sse / other_count, 0.0)
    label_mse[incoming_label] = jnp.where(
        other_count > 0.0,
        0.5 * (other_mse + doc_composite),
        doc_composite,
    )

    diagnostics = {
        DOC_STATUS_MACRO_CE: (status_macro, status_count),
        DOC_INCOMING_COORDINATE_LOSS: (
            incoming_coordinate_loss,
            incoming_count,
        ),
        DOC_OUTGOING_COORDINATE_LOSS: (
            outgoing_coordinate_loss,
            outgoing_count,
        ),
        DOC_PHYSICAL_STATUS_MACRO_MSE: (physical_macro, physical_count),
        DOC_COMPOSITE_LOSS: (doc_composite, status_count),
    }
    for name, (value, count) in diagnostics.items():
        field_mse[name] = value
        field_count[name] = count
    expert_mse, expert_label_count, total = _recompute_expert_macro(
        label_mse,
        label_count,
        architecture_contract,
    )
    return base._replace(
        total=total,
        expert_mse=expert_mse,
        expert_label_count=expert_label_count,
        label_mse=label_mse,
        field_mse=field_mse,
        field_count=field_count,
    )


__all__ = [
    "DOC_COMPOSITE_LOSS",
    "DOC_INCOMING_COORDINATE_LOSS",
    "DOC_OUTGOING_COORDINATE_LOSS",
    "DOC_PHYSICAL_STATUS_MACRO_MSE",
    "DOC_STATUS_MACRO_CE",
    "masked_doc_status_branch_magnitude_loss",
]
