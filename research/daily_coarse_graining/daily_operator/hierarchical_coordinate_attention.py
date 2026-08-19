"""Coordinate-query Attention for the coupled litter, POC, and DOC branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp

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
    _attention,
    _ffn,
    _field_raw_output,
    _layer_norm,
    _linear,
    _linear_parameters,
    _normal,
    _residual_block_parameters,
    encode_strong_baseline_experts,
    initialize_strong_baseline_parameters,
    strong_baseline_field_context,
    strong_baseline_parameter_count,
)
from .strong_baseline_contract import StrongBaselineContract
from .types import AssembledDailyOperatorInput

CANDIDATE_NAME = "hierarchical_coordinate_query_attention_v3"
LITTER_EXPERT = "litter_poc_doc"
BRANCH_NAMES = (
    "litter_composition",
    "doc_reaction",
    "doc_transport",
    "inventory_transition",
)

_COMPOSITION_FIELDS = frozenset(
    {
        "typed.ok_leak.litter_respiration",
        "typed.ok_leak.litter_flood_respiration",
        "typed.ok_leak.litter_to_doc",
        "typed.ok_leak.floodcarbon_input",
    }
)
_REACTION_FIELDS = frozenset(
    {
        "typed.ok_leak.poc_gross_decomposition",
        "typed.ok_leak.poc_flood_gross_decomposition",
        "typed.ok_leak.doc_gross_decomposition",
        "typed.ok_leak.doc_flood_gross_decomposition",
        "typed.ok_leak.doc_free_to_adsorbed",
    }
)


@dataclass(frozen=True)
class HierarchicalCoordinateAttentionConfig:
    d_model: int = 384
    attention_heads: int = 8
    ffn_width: int = 1536
    branch_blocks: int = 2
    decoder_width: int = 512
    minimum_nonnegative_ratio: float = 0.05
    minimum_unit_interval_center: float = 0.05
    parameter_count_maximum: int = 60_000_000

    def validate(self, baseline: StrongBaselineConfig) -> None:
        if self.d_model != baseline.d_model:
            raise ValueError("coordinate-query and baseline widths must match")
        if self.attention_heads != baseline.attention_heads:
            raise ValueError("coordinate-query and baseline head counts must match")
        if self.d_model % self.attention_heads:
            raise ValueError("coordinate-query width must divide its head count")
        if self.ffn_width != baseline.ffn_width:
            raise ValueError("coordinate-query FFN width must match the shared backbone")
        if self.branch_blocks != 2:
            raise ValueError("coordinate-query stages require two residual blocks")
        if self.decoder_width < self.d_model:
            raise ValueError("coordinate-query decoder must not narrow the coordinate state")
        if not 0.0 < self.minimum_nonnegative_ratio < 1.0:
            raise ValueError("coordinate-query nonnegative ratio must lie in (0, 1)")
        if not 0.0 < self.minimum_unit_interval_center < 0.5:
            raise ValueError("coordinate-query bounded center must lie in (0, 0.5)")


def litter_coordinate_branch(name: str) -> str:
    """Assign every admitted litter field to one fixed feed-forward stage."""

    if name.startswith("inventory.carbon."):
        return "inventory_transition"
    if name.startswith("endpoint.ok_leak_aux.") or name in _COMPOSITION_FIELDS:
        return "litter_composition"
    if name in _REACTION_FIELDS:
        return "doc_reaction"
    if name.startswith("typed.ok_leak."):
        return "doc_transport"
    raise ValueError(f"unclassified litter/POC/DOC field {name!r}")


def initialize_hierarchical_coordinate_attention_parameters(
    key: Any,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    inputs: AssembledDailyOperatorInput,
    contract: StrongBaselineContract,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    config: HierarchicalCoordinateAttentionConfig = HierarchicalCoordinateAttentionConfig(),
) -> Mapping[str, Any]:
    """Initialize a fresh candidate while preserving the rejected baseline tree."""

    config.validate(baseline_config)
    baseline_key, candidate_key = jax.random.split(key)
    parameters = dict(
        initialize_strong_baseline_parameters(
            baseline_key,
            schema=schema,
            layout=layout,
            inputs=inputs,
            contract=contract,
            config=baseline_config,
        )
    )
    if LITTER_EXPERT not in layout.expert_names:
        raise ValueError("coordinate-query candidate requires the litter expert")
    litter_expert_index = layout.expert_names.index(LITTER_EXPERT)
    litter_fields = tuple(
        index
        for index, ownership in enumerate(layout.ledger)
        if ownership.expert == LITTER_EXPERT
    )
    if len(litter_fields) != 42:
        raise ValueError("coordinate-query candidate requires the frozen 42-field litter boundary")
    litter_labels = {schema.fields[index].label_index for index in litter_fields}
    label_heads = list(parameters["label_heads"])
    for label_index in litter_labels:
        label_heads[label_index] = {}
    parameters["label_heads"] = tuple(label_heads)
    expert_blocks = list(parameters["expert_blocks"])
    expert_blocks[litter_expert_index] = ()
    parameters["expert_blocks"] = tuple(expert_blocks)

    keys = iter(jax.random.split(candidate_key, 128))
    branch_blocks = tuple(
        tuple(
            _residual_block_parameters(
                next(keys),
                config.d_model,
                config.ffn_width,
            )
            for _ in range(config.branch_blocks)
        )
        for _ in BRANCH_NAMES
    )
    branch_decoders = tuple(
        _linear_parameters(next(keys), config.d_model, config.decoder_width)
        for _ in BRANCH_NAMES
    )
    context_projections = tuple(
        _linear_parameters(next(keys), config.d_model, config.d_model)
        for _ in BRANCH_NAMES
    )
    field_heads = []
    for field_index in range(len(schema.fields)):
        if field_index not in litter_fields:
            field_heads.append({})
            continue
        head = _linear_parameters(next(keys), config.decoder_width, 1)
        head["kernel"] = head["kernel"] * 1.0e-3
        field_heads.append(head)
    parameters["hierarchical_litter"] = {
        "branch_embedding": _normal(
            next(keys),
            (len(BRANCH_NAMES), config.d_model),
            config.d_model,
        ),
        "branch_blocks": branch_blocks,
        "branch_decoders": branch_decoders,
        "context_projections": context_projections,
        "context_gate_logits": jnp.zeros((len(BRANCH_NAMES),), dtype=jnp.float32),
        "context_norm_scale": jnp.ones(
            (len(BRANCH_NAMES), config.d_model),
            dtype=jnp.float32,
        ),
        "context_norm_bias": jnp.zeros(
            (len(BRANCH_NAMES), config.d_model),
            dtype=jnp.float32,
        ),
        "field_heads": tuple(field_heads),
    }
    count = strong_baseline_parameter_count(parameters)
    if count > config.parameter_count_maximum:
        raise ValueError(
            f"coordinate-query candidate has {count} parameters, above "
            f"{config.parameter_count_maximum}"
        )
    return parameters


def _source_family_bank(
    context: StrongBaselineContext,
    layout: StrongBaselineLayout,
    expert_index: int,
) -> tuple[Any, Any]:
    family_count = len(layout.source_family_names)
    family = jnp.asarray(context.source_family_index, dtype=jnp.int32)
    selected = (
        family[None, :] == jnp.arange(family_count, dtype=jnp.int32)[:, None]
    )
    selected &= jnp.asarray(context.source_mask, dtype=jnp.bool_)[None, :]
    selected &= jnp.asarray(layout.source_allowed[expert_index], dtype=jnp.bool_)[:, None]
    count = jnp.sum(selected, axis=1, dtype=jnp.float32)
    tokens = jnp.sum(
        selected[:, :, None].astype(jnp.float64)
        * jnp.asarray(context.source_tokens, dtype=jnp.float64)[None, :, :],
        axis=1,
        dtype=jnp.float64,
    ) / jnp.maximum(count[:, None], 1.0)
    valid = count > 0.0
    return jnp.asarray(jnp.where(valid[:, None], tokens, 0.0), dtype=jnp.float32), valid


def _coordinate_query(
    parameters: Mapping[str, Any],
    field_context: Any,
    family_tokens: Any,
    family_mask: Any,
    upstream_contexts: tuple[Any, ...],
    *,
    branch_index: int,
    config: HierarchicalCoordinateAttentionConfig,
) -> Any:
    size = field_context.features.shape[0]
    family_bank = jnp.broadcast_to(
        family_tokens[None, :, :],
        (size, family_tokens.shape[0], config.d_model),
    )
    local_token = field_context.local_state_token[:, None, :]
    banks = [family_bank, field_context.axis_tokens, local_token]
    masks = [
        field_context.active[:, None] & family_mask[None, :],
        field_context.active[:, None] & field_context.axis_mask,
        field_context.local_state_mask[:, None],
    ]
    for upstream in upstream_contexts:
        banks.append(jnp.broadcast_to(upstream, (size, 1, config.d_model)))
        masks.append(field_context.active[:, None])
    bank = jnp.concatenate(tuple(banks), axis=1)
    mask = jnp.concatenate(tuple(masks), axis=1)
    branch = parameters["hierarchical_litter"]
    query = (
        field_context.features + branch["branch_embedding"][branch_index]
    )[:, None, :]
    for block in branch["branch_blocks"][branch_index]:
        message = _attention(
            block["attention"],
            query,
            bank,
            heads=config.attention_heads,
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
    query = query[:, 0, :]
    return jnp.where(field_context.active[:, None], query, 0.0)


def _masked_stage_pool(
    values: tuple[Any, ...],
    masks: tuple[Any, ...],
    width: int,
) -> Any:
    total = jnp.zeros((width,), dtype=jnp.float64)
    field_count = jnp.asarray(0.0, dtype=jnp.float64)
    for value, mask in zip(values, masks, strict=True):
        selected = jnp.asarray(mask, dtype=jnp.bool_)
        coordinate_count = jnp.sum(selected, dtype=jnp.float64)
        field_mean = jnp.sum(
            jnp.where(selected[:, None], jnp.asarray(value, dtype=jnp.float64), 0.0),
            axis=0,
            dtype=jnp.float64,
        ) / jnp.maximum(coordinate_count, 1.0)
        valid = coordinate_count > 0.0
        total += jnp.where(valid, field_mean, 0.0)
        field_count += valid.astype(jnp.float64)
    return jnp.asarray(total / jnp.maximum(field_count, 1.0), dtype=jnp.float32)


def _hierarchical_litter_raw_outputs(
    parameters: Mapping[str, Any],
    expert_tokens: Any,
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    config: HierarchicalCoordinateAttentionConfig,
) -> tuple[dict[int, Any], dict[int, Any]]:
    expert_index = layout.expert_names.index(LITTER_EXPERT)
    family_tokens, family_mask = _source_family_bank(context, layout, expert_index)
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
        for field_index, (field, ownership) in enumerate(
            zip(schema.fields, layout.ledger, strict=True)
        )
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
    for branch_index, branch_name in enumerate(BRANCH_NAMES):
        current = {}
        for field_index in branch_indices[branch_name]:
            current[field_index] = _coordinate_query(
                parameters,
                field_contexts[field_index],
                family_tokens,
                family_mask,
                tuple(stage_contexts),
                branch_index=branch_index,
                config=config,
            )
        stage_context = _masked_stage_pool(
            tuple(current.values()),
            tuple(field_contexts[index].active for index in current),
            config.d_model,
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

    raw = {}
    active = {}
    branch = parameters["hierarchical_litter"]
    for field_index, value in encoded.items():
        branch_name = litter_coordinate_branch(schema.fields[field_index].name)
        branch_index = BRANCH_NAMES.index(branch_name)
        hidden = jax.nn.gelu(_linear(branch["branch_decoders"][branch_index], value))
        raw[field_index] = _linear(branch["field_heads"][field_index], hidden)[:, 0]
        active[field_index] = field_contexts[field_index].active
    return raw, active


def hierarchical_coordinate_attention_process_outputs(
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
    config: HierarchicalCoordinateAttentionConfig = HierarchicalCoordinateAttentionConfig(),
) -> Mapping[str, Any]:
    """Decode all fields while replacing only the litter coordinate mapping."""

    config.validate(baseline_config)
    expert_tokens = encode_strong_baseline_experts(
        parameters,
        context,
        layout=layout,
        message_gate_scale=message_gate_scale,
        config=baseline_config,
    )
    litter_raw, litter_active = _hierarchical_litter_raw_outputs(
        parameters,
        expert_tokens,
        context,
        inputs,
        schema=schema,
        layout=layout,
        input_normalization=input_normalization,
        config=config,
    )
    outputs = {}
    for field_index, (field, ownership) in enumerate(
        zip(schema.fields, layout.ledger, strict=True)
    ):
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
                config.minimum_nonnegative_ratio
                if ownership.expert == LITTER_EXPERT
                else 1.0e-12
            ),
            minimum_unit_interval_center=(
                config.minimum_unit_interval_center
                if ownership.expert == LITTER_EXPERT
                else 1.0e-6
            ),
            rescale_unit_interval_delta=ownership.expert != LITTER_EXPERT,
        )
    return apply_fast_parameter_routes(outputs, inputs, layout, forcing_layout)


def hierarchical_coordinate_attention_parameter_count(
    parameters: Mapping[str, Any],
) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "BRANCH_NAMES",
    "CANDIDATE_NAME",
    "HierarchicalCoordinateAttentionConfig",
    "hierarchical_coordinate_attention_parameter_count",
    "hierarchical_coordinate_attention_process_outputs",
    "initialize_hierarchical_coordinate_attention_parameters",
    "litter_coordinate_branch",
]
