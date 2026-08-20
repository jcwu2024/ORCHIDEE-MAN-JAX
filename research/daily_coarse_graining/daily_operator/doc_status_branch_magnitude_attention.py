"""Coordinate-aligned DOC status and branch-specific magnitude heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .coordinate_coupled_doc_attention import (
    DOC_INCOMING_FIELD,
    DOC_OUTGOING_FIELD,
    CoordinateCoupledDocTransitionConfig,
    _coordinate_coupled_litter_features,
    _day_start_doc,
    initialize_coordinate_coupled_doc_transition_parameters,
)
from .hierarchical_coordinate_attention import (
    LITTER_EXPERT,
    HierarchicalCoordinateAttentionConfig,
)
from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .output_transform import normalized_output_to_physical
from .parameter_routing import apply_fast_parameter_routes
from .process_derivations import NativeForcingFieldLayout
from .strong_baseline_architecture import (
    StrongBaselineConfig,
    StrongBaselineContext,
    StrongBaselineLayout,
    _field_raw_output,
    _linear,
    _linear_parameters,
    encode_strong_baseline_experts,
    strong_baseline_parameter_count,
)
from .strong_baseline_contract import StrongBaselineContract
from .types import AssembledDailyOperatorInput

CANDIDATE_NAME = "coordinate_coupled_doc_status_branch_magnitude_attention_v5"
DOC_STATUS_ORDER = ("incoming", "unchanged", "outgoing")
DOC_STATUS_LOGITS_AUX = "doc.status_logits"
DOC_STATUS_PROBABILITIES_AUX = "doc.status_probabilities"
DOC_INCOMING_RAW_AUX = "doc.incoming_raw_coordinate"
DOC_OUTGOING_RAW_AUX = "doc.outgoing_raw_coordinate"
DOC_INCOMING_MAGNITUDE_AUX = "doc.incoming_ungated_magnitude"
DOC_OUTGOING_MAGNITUDE_AUX = "doc.outgoing_ungated_magnitude"


@dataclass(frozen=True)
class DocStatusBranchMagnitudeConfig:
    status_prior: tuple[float, float, float] = (
        0.07485037284144427,
        0.8762663118131869,
        0.04888331534536892,
    )
    minimum_nonnegative_ratio: float = 0.05
    minimum_unit_interval_center: float = 1.0e-6
    meaningful_magnitude: float = 1.0e-12
    smooth_l1_delta: float = 1.0
    parameter_count_maximum: int = 62_000_000

    def validate(
        self,
        baseline: StrongBaselineConfig,
        coordinate: HierarchicalCoordinateAttentionConfig,
        parent: CoordinateCoupledDocTransitionConfig,
    ) -> None:
        parent.validate(baseline, coordinate)
        prior = np.asarray(self.status_prior, dtype=np.float64)
        if prior.shape != (3,) or np.any(~np.isfinite(prior)) or np.any(prior <= 0.0):
            raise ValueError("DOC status prior must contain three positive finite values")
        if not np.isclose(float(np.sum(prior)), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("DOC status prior must sum to one")
        if not 0.0 < self.minimum_nonnegative_ratio < 1.0:
            raise ValueError("DOC incoming center ratio must lie in (0, 1)")
        if not 0.0 < self.minimum_unit_interval_center < 0.5:
            raise ValueError("DOC outgoing center bound must lie in (0, 0.5)")
        if self.meaningful_magnitude < 0.0 or self.smooth_l1_delta <= 0.0:
            raise ValueError("DOC status-magnitude loss thresholds are invalid")


class DocStatusBranchMagnitudeOutput(NamedTuple):
    incoming: Any
    outgoing: Any
    status_logits: Any
    status_probabilities: Any
    incoming_raw: Any
    outgoing_raw: Any
    incoming_magnitude: Any
    outgoing_magnitude: Any


def initialize_doc_status_branch_magnitude_parameters(
    key: Any,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    inputs: AssembledDailyOperatorInput,
    contract: StrongBaselineContract,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    coordinate_config: HierarchicalCoordinateAttentionConfig = (
        HierarchicalCoordinateAttentionConfig()
    ),
    parent_config: CoordinateCoupledDocTransitionConfig = (
        CoordinateCoupledDocTransitionConfig()
    ),
    config: DocStatusBranchMagnitudeConfig = DocStatusBranchMagnitudeConfig(),
) -> Mapping[str, Any]:
    """Replace the v4 signed scalar with status and two magnitude heads."""

    config.validate(baseline_config, coordinate_config, parent_config)
    parameters = dict(
        initialize_coordinate_coupled_doc_transition_parameters(
            key,
            schema=schema,
            layout=layout,
            inputs=inputs,
            contract=contract,
            baseline_config=baseline_config,
            coordinate_config=coordinate_config,
            config=parent_config,
        )
    )
    hierarchical = dict(parameters["hierarchical_litter"])
    coupling = dict(hierarchical["doc_coupling"])
    del coupling["transition_head"]
    status_key, incoming_key, outgoing_key = jax.random.split(
        jax.random.fold_in(key, 20260821),
        3,
    )
    status_head = _linear_parameters(status_key, parent_config.decoder_width, 3)
    status_head["kernel"] = status_head["kernel"] * 1.0e-3
    status_head["bias"] = jnp.log(
        jnp.asarray(config.status_prior, dtype=jnp.float32)
    )
    incoming_head = _linear_parameters(
        incoming_key,
        parent_config.decoder_width,
        1,
    )
    incoming_head["kernel"] = incoming_head["kernel"] * 1.0e-3
    outgoing_head = _linear_parameters(
        outgoing_key,
        parent_config.decoder_width,
        1,
    )
    outgoing_head["kernel"] = outgoing_head["kernel"] * 1.0e-3
    coupling["status_head"] = status_head
    coupling["incoming_magnitude_head"] = incoming_head
    coupling["outgoing_magnitude_head"] = outgoing_head
    hierarchical["doc_coupling"] = coupling
    parameters["hierarchical_litter"] = hierarchical
    count = strong_baseline_parameter_count(parameters)
    if count > config.parameter_count_maximum:
        raise ValueError(
            f"DOC status-magnitude candidate has {count} parameters, "
            f"above {config.parameter_count_maximum}"
        )
    return parameters


def materialize_doc_status_branch_magnitudes(
    status_logits: Any,
    incoming_raw: Any,
    outgoing_raw: Any,
    active: Any,
    *,
    incoming_field: Any,
    outgoing_field: Any,
    normalization: NeuralOutputNormalization,
    config: DocStatusBranchMagnitudeConfig = DocStatusBranchMagnitudeConfig(),
) -> DocStatusBranchMagnitudeOutput:
    """Decode differentiable status gates and branch-specific magnitudes."""

    if incoming_field.shape != outgoing_field.shape:
        raise ValueError("DOC status-magnitude endpoint shapes drifted")
    if incoming_field.constraint != "nonnegative":
        raise ValueError("DOC incoming magnitude must be nonnegative")
    if outgoing_field.constraint != "unit_interval":
        raise ValueError("DOC outgoing magnitude must lie in the unit interval")
    if incoming_field.value_representation != "direct":
        raise ValueError("DOC incoming magnitude requires direct representation")
    if outgoing_field.value_representation != "direct":
        raise ValueError("DOC outgoing magnitude requires direct representation")

    size = int(np.prod(incoming_field.shape, dtype=np.int64))
    logits = jnp.asarray(status_logits, dtype=jnp.float64).reshape(size, 3)
    incoming_coordinate = jnp.asarray(incoming_raw, dtype=jnp.float64).reshape(size)
    outgoing_coordinate = jnp.asarray(outgoing_raw, dtype=jnp.float64).reshape(size)
    admitted = jnp.asarray(active, dtype=jnp.bool_).reshape(size)
    incoming_mean = jnp.asarray(
        normalization.mean[incoming_field.name],
        dtype=jnp.float64,
    ).reshape(size)
    incoming_scale = jnp.asarray(
        normalization.scale[incoming_field.name],
        dtype=jnp.float64,
    ).reshape(size)
    outgoing_mean = jnp.asarray(
        normalization.mean[outgoing_field.name],
        dtype=jnp.float64,
    ).reshape(size)
    incoming_scale = jnp.where(
        jnp.isfinite(incoming_scale) & (incoming_scale > 0.0),
        incoming_scale,
        1.0,
    )
    incoming_ratio = jnp.maximum(
        incoming_mean / incoming_scale,
        config.minimum_nonnegative_ratio,
    )
    incoming_baseline = incoming_ratio + jnp.log(-jnp.expm1(-incoming_ratio))
    incoming_magnitude = (
        jax.nn.softplus(incoming_baseline + incoming_coordinate) * incoming_scale
    )
    outgoing_center = jnp.clip(
        outgoing_mean,
        config.minimum_unit_interval_center,
        1.0 - config.minimum_unit_interval_center,
    )
    outgoing_baseline = jnp.log(outgoing_center) - jnp.log1p(-outgoing_center)
    outgoing_magnitude = jax.nn.sigmoid(outgoing_baseline + outgoing_coordinate)
    probabilities = jax.nn.softmax(logits, axis=-1)
    incoming = probabilities[:, 0] * incoming_magnitude
    outgoing = probabilities[:, 2] * outgoing_magnitude
    incoming = jnp.where(admitted, incoming, 0.0)
    outgoing = jnp.where(admitted, outgoing, 0.0)
    shape = incoming_field.shape
    return DocStatusBranchMagnitudeOutput(
        incoming=incoming.reshape(shape),
        outgoing=outgoing.reshape(shape),
        status_logits=logits.reshape((*shape, 3)),
        status_probabilities=probabilities.reshape((*shape, 3)),
        incoming_raw=incoming_coordinate.reshape(shape),
        outgoing_raw=outgoing_coordinate.reshape(shape),
        incoming_magnitude=incoming_magnitude.reshape(shape),
        outgoing_magnitude=outgoing_magnitude.reshape(shape),
    )


def doc_status_branch_magnitude_process_outputs(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    output_normalization: NeuralOutputNormalization,
    forcing_layout: NativeForcingFieldLayout,
    message_gate_scale: float = 1.0,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    coordinate_config: HierarchicalCoordinateAttentionConfig = (
        HierarchicalCoordinateAttentionConfig()
    ),
    parent_config: CoordinateCoupledDocTransitionConfig = (
        CoordinateCoupledDocTransitionConfig()
    ),
    config: DocStatusBranchMagnitudeConfig = DocStatusBranchMagnitudeConfig(),
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Decode v5 learned outputs plus DOC loss coordinates."""

    config.validate(baseline_config, coordinate_config, parent_config)
    expert_tokens = encode_strong_baseline_experts(
        parameters,
        context,
        layout=layout,
        message_gate_scale=message_gate_scale,
        config=baseline_config,
    )
    litter_raw, litter_active, alignment, hidden = (
        _coordinate_coupled_litter_features(
            parameters,
            expert_tokens,
            context,
            inputs,
            schema=schema,
            layout=layout,
            input_normalization=input_normalization,
            coordinate_config=coordinate_config,
        )
    )
    coupling = parameters["hierarchical_litter"]["doc_coupling"]
    status_logits = _linear(coupling["status_head"], hidden)
    incoming_raw = _linear(coupling["incoming_magnitude_head"], hidden)[:, 0]
    outgoing_raw = _linear(coupling["outgoing_magnitude_head"], hidden)[:, 0]
    _day_start, day_start_defined = _day_start_doc(
        inputs,
        layout,
        alignment.incoming_field_index,
    )
    joint_active = litter_active[alignment.incoming_field_index] & day_start_defined
    decoded = materialize_doc_status_branch_magnitudes(
        status_logits,
        incoming_raw,
        outgoing_raw,
        joint_active,
        incoming_field=schema.fields[alignment.incoming_field_index],
        outgoing_field=schema.fields[alignment.outgoing_field_index],
        normalization=output_normalization,
        config=config,
    )
    doc_outputs = {
        DOC_INCOMING_FIELD: decoded.incoming,
        DOC_OUTGOING_FIELD: decoded.outgoing,
    }

    outputs = {}
    for field_index, (field, ownership) in enumerate(
        zip(schema.fields, layout.ledger, strict=True)
    ):
        if field.name in doc_outputs:
            outputs[field.name] = doc_outputs[field.name]
            continue
        if ownership.expert == LITTER_EXPERT:
            raw = litter_raw[field_index]
            active = litter_active[field_index]
        else:
            raw, active = _field_raw_output(
                parameters,
                expert_tokens,
                context,
                inputs,
                field=field,
                ownership=ownership,
                field_index=field_index,
                layout=layout,
                normalization=input_normalization,
            )
        outputs[field.name] = normalized_output_to_physical(
            raw,
            active,
            field=field,
            normalization=output_normalization,
            inputs=inputs,
            minimum_nonnegative_ratio=(
                coordinate_config.minimum_nonnegative_ratio
                if ownership.expert == LITTER_EXPERT
                else 1.0e-12
            ),
            minimum_unit_interval_center=(
                coordinate_config.minimum_unit_interval_center
                if ownership.expert == LITTER_EXPERT
                else 1.0e-6
            ),
            rescale_unit_interval_delta=ownership.expert != LITTER_EXPERT,
        )
    routed = apply_fast_parameter_routes(outputs, inputs, layout, forcing_layout)
    auxiliary = {
        DOC_STATUS_LOGITS_AUX: decoded.status_logits,
        DOC_STATUS_PROBABILITIES_AUX: decoded.status_probabilities,
        DOC_INCOMING_RAW_AUX: decoded.incoming_raw,
        DOC_OUTGOING_RAW_AUX: decoded.outgoing_raw,
        DOC_INCOMING_MAGNITUDE_AUX: decoded.incoming_magnitude,
        DOC_OUTGOING_MAGNITUDE_AUX: decoded.outgoing_magnitude,
    }
    return routed, auxiliary


def doc_status_branch_magnitude_parameter_count(
    parameters: Mapping[str, Any],
) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "CANDIDATE_NAME",
    "DOC_INCOMING_MAGNITUDE_AUX",
    "DOC_INCOMING_RAW_AUX",
    "DOC_OUTGOING_MAGNITUDE_AUX",
    "DOC_OUTGOING_RAW_AUX",
    "DOC_STATUS_LOGITS_AUX",
    "DOC_STATUS_ORDER",
    "DOC_STATUS_PROBABILITIES_AUX",
    "DocStatusBranchMagnitudeConfig",
    "DocStatusBranchMagnitudeOutput",
    "doc_status_branch_magnitude_parameter_count",
    "doc_status_branch_magnitude_process_outputs",
    "initialize_doc_status_branch_magnitude_parameters",
    "materialize_doc_status_branch_magnitudes",
]
