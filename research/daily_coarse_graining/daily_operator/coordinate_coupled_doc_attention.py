"""Coordinate-aligned DOC transfers and one joint conservative endpoint head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .hierarchical_coordinate_attention import (
    BRANCH_NAMES,
    LITTER_EXPERT,
    HierarchicalCoordinateAttentionConfig,
    _coordinate_query,
    _masked_stage_pool,
    _source_family_bank,
    initialize_hierarchical_coordinate_attention_parameters,
    litter_coordinate_branch,
)
from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .output_transform import field_coordinates, normalized_output_to_physical
from .parameter_routing import apply_fast_parameter_routes
from .process_derivations import NativeForcingFieldLayout
from .strong_baseline_architecture import (
    StrongBaselineConfig,
    StrongBaselineContext,
    StrongBaselineLayout,
    _attention,
    _ffn,
    _field_raw_output,
    _layer_norm,
    _linear,
    _linear_parameters,
    _normal,
    encode_strong_baseline_experts,
    strong_baseline_field_context,
    strong_baseline_parameter_count,
)
from .strong_baseline_contract import StrongBaselineContract
from .types import AssembledDailyOperatorInput

CANDIDATE_NAME = "coordinate_coupled_doc_transition_attention_v4"
DOC_INCOMING_FIELD = "inventory.carbon.DOC.incoming_amount"
DOC_OUTGOING_FIELD = "inventory.carbon.DOC.outgoing_fraction"
DOC_ALIGNED_SOURCE_FIELDS = (
    "typed.ok_leak.doc_gross_decomposition",
    "typed.ok_leak.doc_flood_gross_decomposition",
    "typed.ok_leak.doc_free_to_adsorbed",
    "typed.ok_leak.doc_advective_interface",
    "typed.ok_leak.doc_diffusive_interface",
)


@dataclass(frozen=True)
class CoordinateCoupledDocTransitionConfig:
    d_model: int = 384
    decoder_width: int = 512
    minimum_inventory_ratio: float = 0.05
    parameter_count_maximum: int = 62_000_000

    def validate(
        self,
        baseline: StrongBaselineConfig,
        coordinate: HierarchicalCoordinateAttentionConfig,
    ) -> None:
        if self.d_model != baseline.d_model or self.d_model != coordinate.d_model:
            raise ValueError("DOC coupling width must match the coordinate backbone")
        if self.decoder_width != coordinate.decoder_width:
            raise ValueError("DOC joint decoder width must match the coordinate decoder")
        if not 0.0 < self.minimum_inventory_ratio < 1.0:
            raise ValueError("DOC inventory origin ratio must lie in (0, 1)")


class DocCoordinateAlignment(NamedTuple):
    incoming_field_index: int
    outgoing_field_index: int
    source_field_indices: tuple[int, ...]
    source_gather_indices: tuple[np.ndarray, ...]


def _field_index(schema: NeuralOutputSchema, name: str) -> int:
    matches = tuple(index for index, field in enumerate(schema.fields) if field.name == name)
    if len(matches) != 1:
        raise ValueError(f"DOC coupling requires exactly one field {name!r}")
    return matches[0]


def _semantic_coordinate(
    field: Any,
    axes: tuple[str, ...],
    coordinate: np.ndarray,
    active_pft_indices: tuple[int, ...],
) -> dict[str, int]:
    result = {}
    for axis_index, axis in enumerate(axes):
        value = int(coordinate[axis_index])
        if axis == "pft" and not field.full_pft_axis:
            if value >= len(active_pft_indices):
                raise ValueError("compact DOC PFT coordinate exceeds the active layout")
            value = int(active_pft_indices[value])
        result[axis] = value
    return result


def build_doc_coordinate_alignment(
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
) -> DocCoordinateAlignment:
    """Build exact full-PFT to compact-inventory gathers for five DOC fields."""

    incoming_index = _field_index(schema, DOC_INCOMING_FIELD)
    outgoing_index = _field_index(schema, DOC_OUTGOING_FIELD)
    incoming = schema.fields[incoming_index]
    outgoing = schema.fields[outgoing_index]
    incoming_axes = tuple(layout.ledger[incoming_index].axes)
    outgoing_axes = tuple(layout.ledger[outgoing_index].axes)
    if incoming.shape != outgoing.shape or incoming_axes != outgoing_axes:
        raise ValueError("joint DOC inventory endpoint coordinates drifted")
    expected_axes = (
        "landpoint",
        "pft",
        "deep_layer",
        "doc_form",
        "doc_pool",
        "element",
    )
    if incoming_axes != expected_axes:
        raise ValueError(f"DOC inventory axes drifted: {incoming_axes}")

    target_coordinates = field_coordinates(incoming.shape)
    target_semantics = tuple(
        _semantic_coordinate(
            incoming,
            incoming_axes,
            coordinate,
            layout.active_pft_indices,
        )
        for coordinate in target_coordinates
    )
    source_indices = tuple(_field_index(schema, name) for name in DOC_ALIGNED_SOURCE_FIELDS)
    gathers = []
    for source_index in source_indices:
        source = schema.fields[source_index]
        source_axes = tuple(layout.ledger[source_index].axes)
        if not set(source_axes).issubset(incoming_axes):
            raise ValueError(f"DOC aligned source {source.name!r} has unmatched axes {source_axes}")
        source_coordinates = field_coordinates(source.shape)
        lookup = {}
        for flat_index, coordinate in enumerate(source_coordinates):
            semantic = _semantic_coordinate(
                source,
                source_axes,
                coordinate,
                layout.active_pft_indices,
            )
            key = tuple(semantic[axis] for axis in source_axes)
            if key in lookup:
                raise ValueError(f"DOC source coordinate collision for {source.name!r}")
            lookup[key] = flat_index
        gather = []
        for semantic in target_semantics:
            key = tuple(semantic[axis] for axis in source_axes)
            if key not in lookup:
                raise ValueError(f"DOC inventory coordinate lacks aligned source {source.name!r}: {key}")
            gather.append(lookup[key])
        gathers.append(np.asarray(gather, dtype=np.int32))
    return DocCoordinateAlignment(
        incoming_field_index=incoming_index,
        outgoing_field_index=outgoing_index,
        source_field_indices=source_indices,
        source_gather_indices=tuple(gathers),
    )


def initialize_coordinate_coupled_doc_transition_parameters(
    key: Any,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    inputs: AssembledDailyOperatorInput,
    contract: StrongBaselineContract,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    coordinate_config: HierarchicalCoordinateAttentionConfig = (HierarchicalCoordinateAttentionConfig()),
    config: CoordinateCoupledDocTransitionConfig = (CoordinateCoupledDocTransitionConfig()),
) -> Mapping[str, Any]:
    """Initialize v4 from a fresh v3 tree and replace the two DOC endpoint heads."""

    config.validate(baseline_config, coordinate_config)
    alignment = build_doc_coordinate_alignment(schema=schema, layout=layout)
    parent_key, coupling_key = jax.random.split(key)
    parameters = dict(
        initialize_hierarchical_coordinate_attention_parameters(
            parent_key,
            schema=schema,
            layout=layout,
            inputs=inputs,
            contract=contract,
            baseline_config=baseline_config,
            config=coordinate_config,
        )
    )
    hierarchical = dict(parameters["hierarchical_litter"])
    field_heads = list(hierarchical["field_heads"])
    field_heads[alignment.incoming_field_index] = {}
    field_heads[alignment.outgoing_field_index] = {}
    hierarchical["field_heads"] = tuple(field_heads)

    keys = iter(jax.random.split(coupling_key, 16))
    transition_head = _linear_parameters(next(keys), config.decoder_width, 1)
    transition_head["kernel"] = transition_head["kernel"] * 1.0e-3
    hierarchical["doc_coupling"] = {
        "source_embedding": _normal(
            next(keys),
            (len(DOC_ALIGNED_SOURCE_FIELDS), config.d_model),
            config.d_model,
        ),
        "source_projection": tuple(
            _linear_parameters(next(keys), config.d_model, config.d_model) for _ in DOC_ALIGNED_SOURCE_FIELDS
        ),
        "scalar_input": _linear_parameters(next(keys), 1, config.d_model),
        "pair_projection": _linear_parameters(
            next(keys),
            2 * config.d_model,
            config.d_model,
        ),
        "pair_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float32),
        "pair_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float32),
        "transition_decoder": _linear_parameters(
            next(keys),
            config.d_model,
            config.decoder_width,
        ),
        "transition_head": transition_head,
    }
    parameters["hierarchical_litter"] = hierarchical
    count = strong_baseline_parameter_count(parameters)
    if count > config.parameter_count_maximum:
        raise ValueError(
            f"coordinate-coupled DOC candidate has {count} parameters, above {config.parameter_count_maximum}"
        )
    return parameters


def _coordinate_query_with_aligned_transfers(
    parameters: Mapping[str, Any],
    field_context: Any,
    family_tokens: Any,
    family_mask: Any,
    upstream_contexts: tuple[Any, ...],
    aligned_tokens: Any,
    aligned_mask: Any,
    *,
    branch_index: int,
    coordinate_config: HierarchicalCoordinateAttentionConfig,
) -> Any:
    size = field_context.features.shape[0]
    family_bank = jnp.broadcast_to(
        family_tokens[None, :, :],
        (size, family_tokens.shape[0], coordinate_config.d_model),
    )
    banks = [
        family_bank,
        field_context.axis_tokens,
        field_context.local_state_token[:, None, :],
        aligned_tokens,
    ]
    masks = [
        field_context.active[:, None] & family_mask[None, :],
        field_context.active[:, None] & field_context.axis_mask,
        field_context.local_state_mask[:, None],
        field_context.active[:, None] & aligned_mask,
    ]
    for upstream in upstream_contexts:
        banks.append(
            jnp.broadcast_to(
                upstream,
                (size, 1, coordinate_config.d_model),
            )
        )
        masks.append(field_context.active[:, None])
    bank = jnp.concatenate(tuple(banks), axis=1)
    mask = jnp.concatenate(tuple(masks), axis=1)
    branch = parameters["hierarchical_litter"]
    query = (field_context.features + branch["branch_embedding"][branch_index])[:, None, :]
    for block in branch["branch_blocks"][branch_index]:
        message = _attention(
            block["attention"],
            query,
            bank,
            heads=coordinate_config.attention_heads,
            mask=mask[:, None, :],
        )
        query = _layer_norm(
            query + message,
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        query = _layer_norm(
            query + _ffn(block["ffn"], query),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
    return jnp.where(field_context.active[:, None], query[:, 0, :], 0.0)


def _decode_coordinate_raw(
    parameters: Mapping[str, Any],
    encoded: Any,
    *,
    field_index: int,
    branch_index: int,
) -> Any:
    branch = parameters["hierarchical_litter"]
    hidden = jax.nn.gelu(_linear(branch["branch_decoders"][branch_index], encoded))
    return _linear(branch["field_heads"][field_index], hidden)[:, 0]


def _aligned_transfer_bank(
    parameters: Mapping[str, Any],
    encoded: Mapping[int, Any],
    raw: Mapping[int, Any],
    active: Mapping[int, Any],
    alignment: DocCoordinateAlignment,
) -> tuple[Any, Any]:
    coupling = parameters["hierarchical_litter"]["doc_coupling"]
    tokens = []
    masks = []
    for slot, (field_index, gather) in enumerate(
        zip(
            alignment.source_field_indices,
            alignment.source_gather_indices,
            strict=True,
        )
    ):
        gather_index = jnp.asarray(gather, dtype=jnp.int32)
        hidden = encoded[field_index][gather_index]
        scalar = raw[field_index][gather_index, None]
        token = _linear(coupling["source_projection"][slot], hidden)
        token += _linear(coupling["scalar_input"], scalar)
        token += coupling["source_embedding"][slot]
        tokens.append(token)
        masks.append(active[field_index][gather_index])
    return jnp.stack(tuple(tokens), axis=1), jnp.stack(tuple(masks), axis=1)


def _coordinate_coupled_litter_features(
    parameters: Mapping[str, Any],
    expert_tokens: Any,
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    coordinate_config: HierarchicalCoordinateAttentionConfig,
) -> tuple[dict[int, Any], dict[int, Any], DocCoordinateAlignment, Any]:
    expert_index = layout.expert_names.index(LITTER_EXPERT)
    family_tokens, family_mask = _source_family_bank(context, layout, expert_index)
    alignment = build_doc_coordinate_alignment(schema=schema, layout=layout)
    field_contexts = {
        field_index: strong_baseline_field_context(
            parameters,
            expert_tokens,
            context,
            inputs,
            field=field,
            ownership=ownership,
            field_index=field_index,
            layout=layout,
            normalization=input_normalization,
            include_expert_token=False,
        )
        for field_index, (field, ownership) in enumerate(zip(schema.fields, layout.ledger, strict=True))
        if ownership.expert == LITTER_EXPERT
    }
    branch_indices = {
        name: tuple(
            field_index
            for field_index in field_contexts
            if litter_coordinate_branch(schema.fields[field_index].name) == name
        )
        for name in BRANCH_NAMES
    }
    if any(not indices for indices in branch_indices.values()):
        raise ValueError("each coordinate-query branch requires at least one learned field")

    stage_contexts = []
    encoded: dict[int, Any] = {}
    raw: dict[int, Any] = {}
    active = {index: value.active for index, value in field_contexts.items()}
    joint_indices = {
        alignment.incoming_field_index,
        alignment.outgoing_field_index,
    }
    for branch_index, branch_name in enumerate(BRANCH_NAMES):
        aligned_tokens = None
        aligned_mask = None
        if branch_name == "inventory_transition":
            aligned_tokens, aligned_mask = _aligned_transfer_bank(
                parameters,
                encoded,
                raw,
                active,
                alignment,
            )
        current = {}
        for field_index in branch_indices[branch_name]:
            if field_index in joint_indices:
                current[field_index] = _coordinate_query_with_aligned_transfers(
                    parameters,
                    field_contexts[field_index],
                    family_tokens,
                    family_mask,
                    tuple(stage_contexts),
                    aligned_tokens,
                    aligned_mask,
                    branch_index=branch_index,
                    coordinate_config=coordinate_config,
                )
            else:
                current[field_index] = _coordinate_query(
                    parameters,
                    field_contexts[field_index],
                    family_tokens,
                    family_mask,
                    tuple(stage_contexts),
                    branch_index=branch_index,
                    config=coordinate_config,
                )
        stage_context = _masked_stage_pool(
            tuple(current.values()),
            tuple(field_contexts[index].active for index in current),
            coordinate_config.d_model,
        )
        stage_contexts.append(stage_context)
        branch = parameters["hierarchical_litter"]
        addition = jax.nn.sigmoid(branch["context_gate_logits"][branch_index]) * _linear(
            branch["context_projections"][branch_index],
            stage_context,
        )
        for field_index, value in current.items():
            encoded[field_index] = _layer_norm(
                value + addition,
                branch["context_norm_scale"][branch_index],
                branch["context_norm_bias"][branch_index],
            )
            if field_index not in joint_indices:
                raw[field_index] = _decode_coordinate_raw(
                    parameters,
                    encoded[field_index],
                    field_index=field_index,
                    branch_index=branch_index,
                )

    coupling = parameters["hierarchical_litter"]["doc_coupling"]
    incoming_encoded = encoded[alignment.incoming_field_index]
    outgoing_encoded = encoded[alignment.outgoing_field_index]
    pair = jnp.concatenate((incoming_encoded, outgoing_encoded), axis=1)
    joint = _layer_norm(
        _linear(coupling["pair_projection"], pair),
        coupling["pair_norm_scale"],
        coupling["pair_norm_bias"],
    )
    hidden = jax.nn.gelu(_linear(coupling["transition_decoder"], joint))
    active[alignment.incoming_field_index] &= active[alignment.outgoing_field_index]
    active[alignment.outgoing_field_index] = active[alignment.incoming_field_index]
    if set(raw) | joint_indices != set(field_contexts):
        raise ValueError("coordinate-coupled litter features did not cover every field")
    return raw, active, alignment, hidden


def _coordinate_coupled_litter_raw_outputs(
    parameters: Mapping[str, Any],
    expert_tokens: Any,
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    coordinate_config: HierarchicalCoordinateAttentionConfig,
) -> tuple[dict[int, Any], dict[int, Any], DocCoordinateAlignment]:
    raw, active, alignment, hidden = _coordinate_coupled_litter_features(
        parameters,
        expert_tokens,
        context,
        inputs,
        schema=schema,
        layout=layout,
        input_normalization=input_normalization,
        coordinate_config=coordinate_config,
    )
    coupling = parameters["hierarchical_litter"]["doc_coupling"]
    signed_raw = _linear(coupling["transition_head"], hidden)[:, 0]
    raw[alignment.incoming_field_index] = signed_raw
    raw[alignment.outgoing_field_index] = signed_raw
    if set(raw) != {
        field_index
        for field_index, ownership in enumerate(layout.ledger)
        if ownership.expert == LITTER_EXPERT
    }:
        raise ValueError("coordinate-coupled litter decoder did not cover every field")
    return raw, active, alignment


def _day_start_doc(
    inputs: AssembledDailyOperatorInput,
    layout: StrongBaselineLayout,
    field_index: int,
) -> tuple[Any, Any]:
    indices = np.asarray(layout.local_state_indices[field_index], dtype=np.int32)
    if indices.size == 0:
        raise ValueError("joint DOC endpoint lacks its day-start state owner")
    mapped = indices >= 0
    safe = np.maximum(indices, 0)
    values = jnp.asarray(inputs.state.other, dtype=jnp.float64)[safe]
    defined = (
        jnp.asarray(mapped, dtype=jnp.bool_)
        & jnp.asarray(inputs.state_defined.other, dtype=jnp.bool_)[safe]
        & jnp.isfinite(values)
    )
    return jnp.where(defined, jnp.maximum(values, 0.0), 0.0), defined


def materialize_joint_doc_inventory_transition(
    raw: Any,
    active: Any,
    day_start_doc: Any,
    *,
    incoming_field: Any,
    outgoing_field: Any,
    normalization: NeuralOutputNormalization,
    minimum_inventory_ratio: float = 0.05,
) -> tuple[Any, Any]:
    """Materialize one signed conservative transition as exclusive endpoints."""

    if incoming_field.shape != outgoing_field.shape:
        raise ValueError("joint DOC endpoint field shapes drifted")
    if incoming_field.constraint != "nonnegative" or outgoing_field.constraint != "unit_interval":
        raise ValueError("joint DOC endpoint constraints drifted")
    raw = jnp.asarray(raw, dtype=jnp.float64).reshape(-1)
    active = jnp.asarray(active, dtype=jnp.bool_).reshape(-1)
    day_start = jnp.asarray(day_start_doc, dtype=jnp.float64).reshape(-1)
    incoming_mean = jnp.asarray(
        normalization.mean[incoming_field.name],
        dtype=jnp.float64,
    ).reshape(-1)
    incoming_scale = jnp.asarray(
        normalization.scale[incoming_field.name],
        dtype=jnp.float64,
    ).reshape(-1)
    outgoing_mean = jnp.asarray(
        normalization.mean[outgoing_field.name],
        dtype=jnp.float64,
    ).reshape(-1)
    outgoing_scale = jnp.asarray(
        normalization.scale[outgoing_field.name],
        dtype=jnp.float64,
    ).reshape(-1)
    if any(
        value.shape != raw.shape
        for value in (
            active,
            day_start,
            incoming_mean,
            incoming_scale,
            outgoing_mean,
            outgoing_scale,
        )
    ):
        raise ValueError("joint DOC endpoint normalization does not match its coordinates")
    incoming_scale = jnp.where(
        jnp.isfinite(incoming_scale) & (incoming_scale > 0.0),
        incoming_scale,
        1.0,
    )
    outgoing_scale = jnp.where(
        jnp.isfinite(outgoing_scale) & (outgoing_scale > 0.0),
        outgoing_scale,
        1.0,
    )
    transition_scale = jnp.sqrt(
        incoming_scale * incoming_scale + (day_start * outgoing_scale) * (day_start * outgoing_scale)
    )
    transition_scale = jnp.where(
        jnp.isfinite(transition_scale) & (transition_scale > 0.0),
        transition_scale,
        1.0,
    )
    center_change = incoming_mean - day_start * outgoing_mean
    center_end = jnp.maximum(
        day_start + center_change,
        transition_scale * minimum_inventory_ratio,
    )
    ratio = jnp.maximum(center_end / transition_scale, minimum_inventory_ratio)
    baseline = ratio + jnp.log(-jnp.expm1(-ratio))
    day_end = jax.nn.softplus(baseline + raw) * transition_scale
    change = day_end - day_start
    incoming = jnp.maximum(change, 0.0)
    safe_start = jnp.where(day_start > 0.0, day_start, 1.0)
    outgoing = jnp.where(
        day_start > 0.0,
        jnp.clip(jnp.maximum(-change, 0.0) / safe_start, 0.0, 1.0),
        0.0,
    )
    incoming = jnp.where(active, incoming, 0.0).reshape(incoming_field.shape)
    outgoing = jnp.where(active, outgoing, 0.0).reshape(outgoing_field.shape)
    return incoming, outgoing


def coordinate_coupled_doc_transition_process_outputs(
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
    coordinate_config: HierarchicalCoordinateAttentionConfig = (HierarchicalCoordinateAttentionConfig()),
    config: CoordinateCoupledDocTransitionConfig = (CoordinateCoupledDocTransitionConfig()),
) -> Mapping[str, Any]:
    """Decode v4 while preserving every non-DOC and non-litter v3 output path."""

    config.validate(baseline_config, coordinate_config)
    expert_tokens = encode_strong_baseline_experts(
        parameters,
        context,
        layout=layout,
        message_gate_scale=message_gate_scale,
        config=baseline_config,
    )
    litter_raw, litter_active, alignment = _coordinate_coupled_litter_raw_outputs(
        parameters,
        expert_tokens,
        context,
        inputs,
        schema=schema,
        layout=layout,
        input_normalization=input_normalization,
        coordinate_config=coordinate_config,
    )
    day_start, day_start_defined = _day_start_doc(
        inputs,
        layout,
        alignment.incoming_field_index,
    )
    joint_active = litter_active[alignment.incoming_field_index] & day_start_defined
    incoming, outgoing = materialize_joint_doc_inventory_transition(
        litter_raw[alignment.incoming_field_index],
        joint_active,
        day_start,
        incoming_field=schema.fields[alignment.incoming_field_index],
        outgoing_field=schema.fields[alignment.outgoing_field_index],
        normalization=output_normalization,
        minimum_inventory_ratio=config.minimum_inventory_ratio,
    )
    joint_outputs = {
        DOC_INCOMING_FIELD: incoming,
        DOC_OUTGOING_FIELD: outgoing,
    }

    outputs = {}
    for field_index, (field, ownership) in enumerate(zip(schema.fields, layout.ledger, strict=True)):
        if field.name in joint_outputs:
            outputs[field.name] = joint_outputs[field.name]
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
                coordinate_config.minimum_nonnegative_ratio if ownership.expert == LITTER_EXPERT else 1.0e-12
            ),
            minimum_unit_interval_center=(
                coordinate_config.minimum_unit_interval_center if ownership.expert == LITTER_EXPERT else 1.0e-6
            ),
            rescale_unit_interval_delta=ownership.expert != LITTER_EXPERT,
        )
    return apply_fast_parameter_routes(outputs, inputs, layout, forcing_layout)


def coordinate_coupled_doc_transition_parameter_count(
    parameters: Mapping[str, Any],
) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "CANDIDATE_NAME",
    "DOC_ALIGNED_SOURCE_FIELDS",
    "DOC_INCOMING_FIELD",
    "DOC_OUTGOING_FIELD",
    "CoordinateCoupledDocTransitionConfig",
    "DocCoordinateAlignment",
    "build_doc_coordinate_alignment",
    "coordinate_coupled_doc_transition_parameter_count",
    "coordinate_coupled_doc_transition_process_outputs",
    "initialize_coordinate_coupled_doc_transition_parameters",
    "materialize_joint_doc_inventory_transition",
]
