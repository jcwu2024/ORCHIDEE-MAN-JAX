"""Shared, ownership-masked Transformer for raw daily process coordinates.

This module owns neural representation learning only.  Physical output
transforms, explicit parameter factors, conservative updates, and the retained
tail stay outside this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .input_conditioning import standardize_continuous
from .neural_architecture import NeuralInputNormalization, NeuralOutputSchema
from .output_transform import field_coordinates
from .state_features import assemble_local_pft_features
from .strong_baseline_architecture import (
    NeuralPrecisionPolicy,
    StrongBaselineConfig,
    StrongBaselineLayout,
    _ffn,
    _layer_norm,
    _linear,
    _linear_parameters,
    _normal,
    _residual_block_parameters,
    _stable_head_l2_normalize,
    assemble_strong_baseline_context,
)
from .types import AssembledDailyOperatorInput

GPP_FIELD = "daily.fast_gpp"


@dataclass(frozen=True)
class SharedTransformerConfig:
    """Fixed first implementation of the simplified shared architecture."""

    width: int = 768
    attention_heads: int = 8
    ffn_width: int = 3072
    backbone_blocks: int = 8
    coordinate_blocks: int = 2
    readout_width: int = 256
    output_chunk_size: int = 64
    normalization_epsilon: float = 1.0e-6
    decoder_execution: str = "gather_reuse"

    # The audited canonical tokenizer remains narrower than the new backbone.
    input_width: int = 384
    input_attention_heads: int = 8
    input_ffn_width: int = 1536
    forcing_width: int = 256
    forcing_heads: int = 8
    forcing_ffn_width: int = 1024
    forcing_blocks: int = 2
    axial_blocks: int = 1

    def validate(self) -> None:
        if self.width <= 0 or self.input_width <= 0 or self.forcing_width <= 0:
            raise ValueError("shared Transformer widths must be positive")
        if self.width % self.attention_heads:
            raise ValueError("shared Transformer width must divide its head count")
        if self.input_width % self.input_attention_heads:
            raise ValueError("input width must divide its head count")
        if self.forcing_width % self.forcing_heads:
            raise ValueError("forcing width must divide its head count")
        if self.ffn_width != 4 * self.width:
            raise ValueError("shared Transformer FFN width must be four times model width")
        if self.input_ffn_width != 4 * self.input_width:
            raise ValueError("input FFN width must be four times input width")
        if self.forcing_ffn_width != 4 * self.forcing_width:
            raise ValueError("forcing FFN width must be four times forcing width")
        if self.backbone_blocks <= 0 or self.coordinate_blocks <= 0:
            raise ValueError("shared Transformer block counts must be positive")
        if self.readout_width <= 0 or self.output_chunk_size <= 0:
            raise ValueError("readout width and output chunk size must be positive")
        if not 0.0 < self.normalization_epsilon < 1.0:
            raise ValueError("normalization epsilon must lie in (0, 1)")
        if self.decoder_execution not in {"original", "gather_only", "gather_reuse"}:
            raise ValueError(
                "decoder execution must be original, gather_only, or gather_reuse"
            )


class SharedTransformerContext(NamedTuple):
    source_tokens: Any
    source_mask: Any
    source_family_index: Any
    expert_axis_tokens: Any
    expert_axis_mask: Any
    pft_tokens: Any
    pft_mask: Any
    process_tokens: Any
    process_source_mask: Any


def _tokenizer_config(config: SharedTransformerConfig) -> StrongBaselineConfig:
    return StrongBaselineConfig(
        d_model=config.input_width,
        attention_heads=config.input_attention_heads,
        ffn_width=config.input_ffn_width,
        expert_blocks=1,
        forcing_width=config.forcing_width,
        forcing_heads=config.forcing_heads,
        forcing_ffn_width=config.forcing_ffn_width,
        forcing_blocks=config.forcing_blocks,
        axial_blocks=config.axial_blocks,
        head_width=max(1, config.input_width // 2),
        graph_rounds=0,
        neural_precision=NeuralPrecisionPolicy(
            attention_scores_fp32=True,
            source_family_pooling_fp32=True,
            normalize_attention_qk=True,
            attention_normalization_epsilon=config.normalization_epsilon,
        ),
    )


def _initialize_tokenizer_parameters(
    keys: Any,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    inputs: AssembledDailyOperatorInput,
    config: SharedTransformerConfig,
) -> dict[str, Any]:
    next_key = iter(keys)
    pft_width = int(assemble_local_pft_features(inputs.pft).shape[-1])
    process_positions = int(np.max(layout.process_static_position_index)) + 1
    return {
        "forcing_input": _linear_parameters(next(next_key), 5, config.forcing_width),
        "forcing_blocks": tuple(
            _residual_block_parameters(next(next_key), config.forcing_width, config.forcing_ffn_width)
            for _ in range(config.forcing_blocks)
        ),
        "axial_blocks": tuple(
            _residual_block_parameters(next(next_key), config.input_width, config.input_ffn_width)
            for _ in range(config.axial_blocks)
        ),
        "forcing_to_model": _linear_parameters(next(next_key), config.forcing_width, config.input_width),
        "summary_input": _linear_parameters(next(next_key), 4, config.input_width),
        "pft_input": _linear_parameters(next(next_key), pft_width, config.input_width),
        "process_static_input": _linear_parameters(next(next_key), 1, config.input_width),
        "calendar_input": _linear_parameters(next(next_key), 4, config.input_width),
        "source_family_embedding": _normal(
            next(next_key), (len(layout.source_family_names), config.input_width), config.input_width
        ),
        "forcing_field_embedding": _normal(
            next(next_key), (layout.forcing_family_index.size, config.forcing_width), config.forcing_width
        ),
        "forcing_position_embedding": _normal(
            next(next_key), (int(np.asarray(inputs.forcing.values).shape[0]), config.forcing_width), config.forcing_width
        ),
        "state_field_embedding": _normal(
            next(next_key), (layout.state_field_pool.shape[0], config.input_width), config.input_width
        ),
        "discrete_state_field_embedding": _normal(
            next(next_key), (layout.discrete_state_field_pool.shape[0], config.input_width), config.input_width
        ),
        "static_field_embedding": _normal(
            next(next_key), (layout.static_field_pool.shape[0], config.input_width), config.input_width
        ),
        "annual_field_embedding": _normal(
            next(next_key), (layout.annual_field_pool.shape[0], config.input_width), config.input_width
        ),
        "process_static_field_embedding": _normal(
            next(next_key), (len(inputs.process_static._fields), config.input_width), config.input_width
        ),
        "process_static_position_embedding": _normal(
            next(next_key), (process_positions, config.input_width), config.input_width
        ),
        "expert_embedding": _normal(
            next(next_key), (len(layout.expert_names), config.input_width), config.input_width
        ),
        "axis_role_embedding": _normal(
            next(next_key), (len(layout.axis_names), config.input_width), config.input_width
        ),
        "axis_position_embedding": tuple(
            None
            if axis == "pft"
            else _normal(next(next_key), (extent, config.input_width), config.input_width)
            for axis, extent in zip(layout.axis_names, layout.axis_extents, strict=True)
        ),
    }


def initialize_shared_transformer_parameters(
    key: Any,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    inputs: AssembledDailyOperatorInput,
    config: SharedTransformerConfig = SharedTransformerConfig(),
    carbon_state_layout: Any | None = None,
) -> Mapping[str, Any]:
    """Initialize the complete architecture from one key without inherited weights."""

    config.validate()
    if tuple(field.name for field in schema.fields) != tuple(field.name for field in layout.ledger):
        raise ValueError("shared Transformer schema and layout drift")
    keys = iter(jax.random.split(key, 4096))
    tokenizer_keys = jax.random.split(next(keys), 256)
    tokenizer = _initialize_tokenizer_parameters(
        tokenizer_keys,
        schema=schema,
        layout=layout,
        inputs=inputs,
        config=config,
    )
    carbon = {}
    if carbon_state_layout is not None:
        carbon = {
            "field_embedding": _normal(
                next(keys), (len(carbon_state_layout.field_names), config.width), config.width
            ),
            "axis_role_embedding": _normal(
                next(keys), (len(carbon_state_layout.axis_role_names), config.width), config.width
            ),
            "axis_position_embedding": tuple(
                _normal(next(keys), (int(extent), config.width), config.width)
                for extent in carbon_state_layout.axis_role_extents
            ),
        }
    shared = {
        "input_projection": _linear_parameters(next(keys), config.input_width, config.width),
        "process_query": _normal(next(keys), (len(layout.expert_names), config.width), config.width),
        "process_embedding": _normal(next(keys), (len(layout.expert_names), config.width), config.width),
        "backbone_blocks": tuple(
            _residual_block_parameters(next(keys), config.width, config.ffn_width)
            for _ in range(config.backbone_blocks)
        ),
        "coordinate_blocks": tuple(
            _residual_block_parameters(next(keys), config.width, config.ffn_width)
            for _ in range(config.coordinate_blocks)
        ),
        "field_embedding": _normal(next(keys), (len(schema.fields), config.width), config.width),
        "label_embedding": _normal(next(keys), (len(schema.labels), config.width), config.width),
        "output_axis_role_embedding": _normal(
            next(keys), (len(layout.axis_names), config.width), config.width
        ),
        "output_axis_position_embedding": tuple(
            None
            if axis == "pft"
            else _normal(next(keys), (extent, config.width), config.width)
            for axis, extent in zip(layout.axis_names, layout.axis_extents, strict=True)
        ),
        "axis_slot_embedding": _normal(
            next(keys), (layout.max_output_rank, config.width), config.width
        ),
        "local_value_input": _linear_parameters(next(keys), 2, config.width),
        "readout_first": _linear_parameters(next(keys), config.width, config.readout_width),
        "readout_second": _linear_parameters(next(keys), config.readout_width, 1),
        "carbon": carbon,
    }
    return {"tokenizer": tokenizer, "shared_transformer": shared}


def shared_transformer_parameter_count(parameters: Mapping[str, Any]) -> int:
    return int(sum(int(np.prod(leaf.shape, dtype=np.int64)) for leaf in jax.tree_util.tree_leaves(parameters)))


def _normalized_attention(
    parameters: Mapping[str, Any],
    query: Any,
    key_value: Any,
    *,
    heads: int,
    mask: Any,
    normalization_epsilon: float,
) -> Any:
    projected_query = _linear(parameters["query"], query)
    projected_key = _linear(parameters["key"], key_value)
    projected_value = _linear(parameters["value"], key_value)
    width = int(projected_query.shape[-1])
    head_width = width // int(heads)
    head_query = projected_query.reshape(projected_query.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    head_key = projected_key.reshape(projected_key.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    head_value = projected_value.reshape(projected_value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    head_query = _stable_head_l2_normalize(head_query, normalization_epsilon)
    head_key = _stable_head_l2_normalize(head_key, normalization_epsilon)
    temperature = jnp.sqrt(jnp.asarray(head_width, dtype=jnp.float32))
    logits = jnp.clip(jnp.einsum("...hqd,...hkd->...hqk", head_query, head_key), -1.0, 1.0)
    logits = logits * temperature
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    head_mask = jnp.expand_dims(mask, axis=-3)
    row_valid = jnp.expand_dims(jnp.any(mask, axis=-1, keepdims=True), axis=-3)
    logits = jnp.where(head_mask, logits, -jnp.inf)
    weights = jax.nn.softmax(jnp.where(row_valid, logits, 0.0), axis=-1)
    weights = jnp.where(head_mask, weights, 0.0)
    result = jnp.einsum("...hqk,...hkd->...hqd", weights, head_value)
    result = result.swapaxes(-3, -2).reshape(result.shape[:-3] + (result.shape[-2], width))
    return _linear(parameters["output"], result)


def _run_blocks(
    query: Any,
    bank: Any,
    mask: Any,
    blocks: Any,
    config: SharedTransformerConfig,
) -> Any:
    for block in blocks:
        message = _normalized_attention(
            block["attention"],
            query,
            bank,
            heads=config.attention_heads,
            mask=mask,
            normalization_epsilon=config.normalization_epsilon,
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
    return query


class _DecoderBlockProjection(NamedTuple):
    family_key: Any
    family_value: Any
    process_key: Any
    process_value: Any
    carbon_key: Any | None
    carbon_value: Any | None


def _project_key_value(parameters: Mapping[str, Any], tokens: Any) -> tuple[Any, Any]:
    return _linear(parameters["key"], tokens), _linear(parameters["value"], tokens)


def _decoder_projection_cache(
    parameters: Mapping[str, Any],
    context: SharedTransformerContext,
    family_tokens: Any,
    *,
    carbon_tokens: Any | None,
) -> tuple[tuple[_DecoderBlockProjection, ...], ...]:
    """Project banks shared by coordinate queries once per expert and block."""

    blocks = parameters["shared_transformer"]["coordinate_blocks"]
    carbon_by_block = []
    for block in blocks:
        if carbon_tokens is None:
            carbon_by_block.append((None, None))
        else:
            carbon_by_block.append(
                _project_key_value(block["attention"], carbon_tokens)
            )

    result = []
    for expert_index in range(int(context.process_tokens.shape[0])):
        expert = []
        for block_index, block in enumerate(blocks):
            family_key, family_value = _project_key_value(
                block["attention"], family_tokens[expert_index]
            )
            process_key, process_value = _project_key_value(
                block["attention"], context.process_tokens[expert_index][None, :]
            )
            carbon_key, carbon_value = carbon_by_block[block_index]
            expert.append(
                _DecoderBlockProjection(
                    family_key=family_key,
                    family_value=family_value,
                    process_key=process_key,
                    process_value=process_value,
                    carbon_key=carbon_key,
                    carbon_value=carbon_value,
                )
            )
        result.append(tuple(expert))
    return tuple(result)


def _attention_from_projected_segments(
    parameters: Mapping[str, Any],
    query: Any,
    dynamic_tokens: Any,
    *,
    family_key: Any,
    family_value: Any,
    family_mask: Any,
    dynamic_mask: Any,
    process_key: Any,
    process_value: Any,
    process_mask: Any,
    carbon_key: Any | None,
    carbon_value: Any | None,
    carbon_mask: Any | None,
    heads: int,
    normalization_epsilon: float,
) -> Any:
    """Attend to shared and query-local banks with one normalization."""

    projected_query = _linear(parameters["query"], query)
    dynamic_key, dynamic_value = _project_key_value(parameters, dynamic_tokens)
    width = int(projected_query.shape[-1])
    head_width = width // int(heads)

    def per_head(value: Any) -> Any:
        shaped = value.reshape(value.shape[:-1] + (heads, head_width))
        return shaped.swapaxes(-3, -2)

    head_query = per_head(projected_query)
    head_dynamic_key = per_head(dynamic_key)
    head_dynamic_value = per_head(dynamic_value)
    head_family_key = per_head(family_key)
    head_family_value = per_head(family_value)
    head_process_key = per_head(process_key)
    head_process_value = per_head(process_value)
    head_query = _stable_head_l2_normalize(head_query, normalization_epsilon)
    head_dynamic_key = _stable_head_l2_normalize(head_dynamic_key, normalization_epsilon)
    head_family_key = _stable_head_l2_normalize(head_family_key, normalization_epsilon)
    head_process_key = _stable_head_l2_normalize(head_process_key, normalization_epsilon)

    temperature = jnp.sqrt(jnp.asarray(head_width, dtype=jnp.float32))

    def scores(key: Any, *, shared: bool) -> Any:
        equation = "...hqd,hkd->...hqk" if shared else "...hqd,...hkd->...hqk"
        return jnp.clip(jnp.einsum(equation, head_query, key), -1.0, 1.0) * temperature

    logits = [
        scores(head_family_key, shared=True),
        scores(head_dynamic_key, shared=False),
        scores(head_process_key, shared=True),
    ]
    masks = [family_mask, dynamic_mask, process_mask]
    head_carbon_value = None
    if carbon_key is not None and carbon_mask is not None:
        head_carbon_key = per_head(carbon_key)
        head_carbon_value = per_head(carbon_value)
        head_carbon_key = _stable_head_l2_normalize(head_carbon_key, normalization_epsilon)
        logits.append(scores(head_carbon_key, shared=True))
        masks.append(carbon_mask)

    combined_logits = jnp.concatenate(tuple(logits), axis=-1)
    combined_mask = jnp.concatenate(tuple(masks), axis=-1)
    head_mask = jnp.expand_dims(combined_mask, axis=-3)
    row_valid = jnp.expand_dims(jnp.any(combined_mask, axis=-1, keepdims=True), axis=-3)
    combined_logits = jnp.where(head_mask, combined_logits, -jnp.inf)
    weights = jax.nn.softmax(jnp.where(row_valid, combined_logits, 0.0), axis=-1)
    weights = jnp.where(head_mask, weights, 0.0)

    offsets = np.cumsum(
        (0, family_key.shape[-2], dynamic_key.shape[-2], process_key.shape[-2])
    )
    result = jnp.einsum(
        "...hqk,hkd->...hqd", weights[..., offsets[0] : offsets[1]], head_family_value
    )
    result += jnp.einsum(
        "...hqk,...hkd->...hqd",
        weights[..., offsets[1] : offsets[2]],
        head_dynamic_value,
    )
    result += jnp.einsum(
        "...hqk,hkd->...hqd", weights[..., offsets[2] : offsets[3]], head_process_value
    )
    if head_carbon_value is not None:
        result += jnp.einsum(
            "...hqk,hkd->...hqd", weights[..., offsets[3] :], head_carbon_value
        )
    result = result.swapaxes(-3, -2).reshape(result.shape[:-3] + (result.shape[-2], width))
    return _linear(parameters["output"], result)


def _run_coordinate_blocks_reusing_projections(
    query: Any,
    dynamic_tokens: Any,
    *,
    dynamic_mask: Any,
    family_mask: Any,
    process_mask: Any,
    carbon_mask: Any | None,
    blocks: Any,
    projections: tuple[_DecoderBlockProjection, ...],
    config: SharedTransformerConfig,
) -> Any:
    for block, projected in zip(blocks, projections, strict=True):
        message = _attention_from_projected_segments(
            block["attention"],
            query,
            dynamic_tokens,
            family_key=projected.family_key,
            family_value=projected.family_value,
            family_mask=family_mask,
            dynamic_mask=dynamic_mask,
            process_key=projected.process_key,
            process_value=projected.process_value,
            process_mask=process_mask,
            carbon_key=projected.carbon_key,
            carbon_value=projected.carbon_value,
            carbon_mask=carbon_mask,
            heads=config.attention_heads,
            normalization_epsilon=config.normalization_epsilon,
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
    return query


def assemble_shared_transformer_context(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    *,
    normalization: NeuralInputNormalization,
    layout: StrongBaselineLayout,
    config: SharedTransformerConfig = SharedTransformerConfig(),
) -> SharedTransformerContext:
    """Encode canonical inputs and run seven isolated streams with shared weights."""

    config.validate()
    base = assemble_strong_baseline_context(
        parameters["tokenizer"],
        inputs,
        normalization=normalization,
        layout=layout,
        config=_tokenizer_config(config),
    )
    shared = parameters["shared_transformer"]
    source_tokens = _linear(shared["input_projection"], base.source_tokens)
    axis_tokens = _linear(shared["input_projection"], base.expert_axis_tokens)
    pft_tokens = _linear(shared["input_projection"], base.pft_tokens)
    family = jnp.asarray(base.source_family_index, dtype=jnp.int32)
    allowed = jnp.asarray(layout.source_allowed, dtype=jnp.bool_)
    source_mask = allowed[:, family] & jnp.asarray(base.source_mask, dtype=jnp.bool_)[None, :]
    bank = jnp.concatenate(
        (
            jnp.broadcast_to(source_tokens[None, :, :], (len(layout.expert_names),) + source_tokens.shape),
            axis_tokens,
        ),
        axis=1,
    )
    bank_mask = jnp.concatenate((source_mask, jnp.asarray(base.expert_axis_mask, dtype=jnp.bool_)), axis=1)
    query = (shared["process_query"] + shared["process_embedding"])[:, None, :]
    query = _run_blocks(
        query,
        bank,
        bank_mask[:, None, :],
        shared["backbone_blocks"],
        config,
    )[:, 0, :]
    return SharedTransformerContext(
        source_tokens=source_tokens,
        source_mask=jnp.asarray(base.source_mask, dtype=jnp.bool_),
        source_family_index=family,
        expert_axis_tokens=axis_tokens,
        expert_axis_mask=jnp.asarray(base.expert_axis_mask, dtype=jnp.bool_),
        pft_tokens=pft_tokens,
        pft_mask=jnp.asarray(base.pft_mask, dtype=jnp.bool_),
        process_tokens=query,
        process_source_mask=source_mask,
    )


def _source_family_bank(
    context: SharedTransformerContext,
    layout: StrongBaselineLayout,
) -> tuple[Any, Any]:
    family_count = len(layout.source_family_names)
    family = jnp.asarray(context.source_family_index, dtype=jnp.int32)
    selected = family[None, None, :] == jnp.arange(family_count, dtype=jnp.int32)[None, :, None]
    selected &= context.process_source_mask[:, None, :]
    count = jnp.sum(selected, axis=2, dtype=jnp.float32)
    tokens = jnp.sum(
        selected[..., None].astype(jnp.float32) * context.source_tokens[None, None, :, :],
        axis=2,
        dtype=jnp.float32,
    ) / jnp.maximum(count[..., None], 1.0)
    valid = count > 0.0
    return jnp.where(valid[..., None], tokens, 0.0), valid


def _field_active_and_pft_owner(
    context: SharedTransformerContext,
    field: Any,
    ownership: Any,
    layout: StrongBaselineLayout,
) -> tuple[Any, Any, Any]:
    coordinates = jnp.asarray(field_coordinates(field.shape), dtype=jnp.int32)
    active = jnp.ones((coordinates.shape[0],), dtype=jnp.bool_)
    pft_owner = jnp.full((coordinates.shape[0],), -1, dtype=jnp.int32)
    for axis_index, axis_name in enumerate(ownership.axes):
        if axis_name != "pft":
            continue
        coordinate = coordinates[:, axis_index]
        if field.full_pft_axis:
            matches = coordinate[:, None] == jnp.asarray(layout.active_pft_indices, dtype=jnp.int32)[None, :]
            compact = jnp.argmax(matches, axis=1)
            active &= jnp.any(matches, axis=1)
        else:
            compact = coordinate
            active &= compact < context.pft_tokens.shape[0]
        safe = jnp.minimum(compact, context.pft_tokens.shape[0] - 1)
        active &= context.pft_mask[safe]
        pft_owner = safe
    return coordinates, active, pft_owner


def _static_field_coordinate_indices(
    field: Any,
    ownership: Any,
    layout: StrongBaselineLayout,
) -> np.ndarray:
    """Return only coordinates allowed by the declared static PFT layout."""

    coordinates = np.asarray(field_coordinates(field.shape), dtype=np.int32)
    eligible = np.ones((coordinates.shape[0],), dtype=np.bool_)
    for axis_index, axis_name in enumerate(ownership.axes):
        if axis_name == "pft" and field.full_pft_axis:
            eligible &= np.isin(
                coordinates[:, axis_index],
                np.asarray(layout.active_pft_indices, dtype=np.int32),
            )
    return np.flatnonzero(eligible).astype(np.int32, copy=False)


def shared_transformer_static_coordinate_indices(
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
) -> Mapping[str, np.ndarray]:
    """Return the layout-only decoder coordinates retained for every field."""

    if tuple(field.name for field in schema.fields) != tuple(field.name for field in layout.ledger):
        raise ValueError("shared Transformer schema and layout drift")
    return {
        field.name: _static_field_coordinate_indices(field, ownership, layout)
        for field, ownership in zip(schema.fields, layout.ledger, strict=True)
    }


def shared_transformer_output_masks(
    context: SharedTransformerContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
) -> Mapping[str, Any]:
    del inputs
    if tuple(field.name for field in schema.fields) != tuple(field.name for field in layout.ledger):
        raise ValueError("shared Transformer schema and layout drift")
    result = {}
    for field, ownership in zip(schema.fields, layout.ledger, strict=True):
        _coordinates, active, _owner = _field_active_and_pft_owner(context, field, ownership, layout)
        result[field.name] = active.reshape(field.shape)
    return result


def _normalized_local_state(
    inputs: AssembledDailyOperatorInput,
    normalization: NeuralInputNormalization,
    indices: np.ndarray,
) -> tuple[Any, Any]:
    if not indices.size:
        return jnp.zeros((0,), dtype=jnp.float32), jnp.zeros((0,), dtype=jnp.bool_)
    mapped = indices >= 0
    safe = np.maximum(indices, 0)
    safe_indices = jnp.asarray(safe, dtype=jnp.int32)
    defined = jnp.asarray(mapped, dtype=jnp.bool_) & jnp.asarray(inputs.state_defined.other, dtype=jnp.bool_)[safe_indices]
    mean = jnp.asarray(normalization.continuous_mean)[safe_indices]
    raw = jnp.asarray(inputs.state.other)[safe_indices]
    value = standardize_continuous(
        jnp.where(defined, raw, mean),
        mean,
        jnp.asarray(normalization.continuous_scale)[safe_indices],
        normalization.continuous_tail_threshold,
        coordinate_mask=(
            None
            if normalization.continuous_tail_mask is None
            else jnp.asarray(normalization.continuous_tail_mask)[safe_indices]
        ),
    )
    return jnp.asarray(value, dtype=jnp.float32), defined


def _field_coordinate_inputs(
    parameters: Mapping[str, Any],
    context: SharedTransformerContext,
    inputs: AssembledDailyOperatorInput,
    *,
    field: Any,
    ownership: Any,
    field_index: int,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    coordinate_indices: np.ndarray | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    shared = parameters["shared_transformer"]
    coordinates, active, pft_owner = _field_active_and_pft_owner(context, field, ownership, layout)
    if coordinate_indices is not None:
        selected = jnp.asarray(coordinate_indices, dtype=jnp.int32)
        coordinates = coordinates[selected]
        active = active[selected]
        pft_owner = pft_owner[selected]
    expert_index = int(layout.field_expert_index[field_index])
    features = jnp.broadcast_to(
        context.process_tokens[expert_index]
        + shared["field_embedding"][field_index]
        + shared["label_embedding"][field.label_index],
        (coordinates.shape[0], context.process_tokens.shape[-1]),
    )
    axis_tokens = []
    axis_masks = []
    for slot, axis_name in enumerate(ownership.axes):
        coordinate = coordinates[:, slot]
        canonical = layout.axis_names.index(axis_name)
        if axis_name == "pft":
            safe = jnp.maximum(pft_owner, 0)
            token = context.pft_tokens[safe]
            dynamic_coordinate = safe
        else:
            token = jnp.zeros_like(features)
            dynamic_coordinate = coordinate
        offset = int(layout.axis_offsets[canonical])
        dynamic_index = offset + jnp.minimum(dynamic_coordinate, layout.axis_extents[canonical] - 1)
        dynamic = context.expert_axis_tokens[expert_index, dynamic_index]
        dynamic_mask = context.expert_axis_mask[expert_index, dynamic_index]
        token = token + jnp.where(dynamic_mask[:, None], dynamic, 0.0)
        token = token + shared["output_axis_role_embedding"][canonical]
        position = shared["output_axis_position_embedding"][canonical]
        if position is not None:
            token = token + position[jnp.minimum(dynamic_coordinate, layout.axis_extents[canonical] - 1)]
        token = token + shared["axis_slot_embedding"][slot]
        token = jnp.where(active[:, None], token, 0.0)
        features = features + token
        if axis_name != "landpoint":
            axis_tokens.append(token)
            axis_masks.append(active & dynamic_mask)
    local_indices = np.asarray(layout.local_state_indices[field_index], dtype=np.int32).reshape(-1)
    if coordinate_indices is not None and local_indices.size:
        local_indices = local_indices[coordinate_indices]
    if local_indices.size:
        local, local_defined = _normalized_local_state(inputs, input_normalization, local_indices)
    else:
        local = jnp.zeros((coordinates.shape[0],), dtype=jnp.float32)
        local_defined = jnp.zeros((coordinates.shape[0],), dtype=jnp.bool_)
    local_token = _linear(
        shared["local_value_input"],
        jnp.stack((local, local_defined.astype(jnp.float32)), axis=1),
    )
    local_token = jnp.where(active[:, None], local_token, 0.0)
    features = features + local_token
    if axis_tokens:
        stacked_axis = jnp.stack(tuple(axis_tokens), axis=1)
        stacked_mask = jnp.stack(tuple(axis_masks), axis=1)
    else:
        stacked_axis = jnp.zeros((coordinates.shape[0], 0, config_width(parameters)), dtype=jnp.float32)
        stacked_mask = jnp.zeros((coordinates.shape[0], 0), dtype=jnp.bool_)
    return features, active, pft_owner, stacked_axis, stacked_mask, local_token, local_defined


def config_width(parameters: Mapping[str, Any]) -> int:
    return int(parameters["shared_transformer"]["process_query"].shape[-1])


def _validate_carbon_layout_parameters(
    carbon_parameters: Mapping[str, Any],
    carbon_state_layout: Any,
) -> None:
    field_count = len(carbon_state_layout.field_names)
    role_count = len(carbon_state_layout.axis_role_names)
    extents = tuple(int(value) for value in carbon_state_layout.axis_role_extents)
    if carbon_parameters["field_embedding"].shape[0] != field_count:
        raise ValueError("carbon state field layout drift")
    if carbon_parameters["axis_role_embedding"].shape[0] != role_count:
        raise ValueError("carbon state axis-role layout drift")
    position_embeddings = carbon_parameters["axis_position_embedding"]
    if len(position_embeddings) != role_count or tuple(
        int(embedding.shape[0]) for embedding in position_embeddings
    ) != extents:
        raise ValueError("carbon state axis-position layout drift")

    state_indices = np.asarray(carbon_state_layout.state_indices)
    field_index = np.asarray(carbon_state_layout.field_index)
    pft_owner = np.asarray(carbon_state_layout.pft_owner_index)
    roles = np.asarray(carbon_state_layout.axis_role_index)
    positions = np.asarray(carbon_state_layout.axis_position_index)
    coordinate_count = state_indices.size
    if field_index.shape != (coordinate_count,) or pft_owner.shape != (coordinate_count,):
        raise ValueError("carbon state coordinate layout drift")
    if roles.shape != positions.shape or roles.ndim != 2 or roles.shape[0] != coordinate_count:
        raise ValueError("carbon state coordinate-axis layout drift")
    if np.any(field_index < 0) or np.any(field_index >= field_count):
        raise ValueError("carbon state field index is out of range")
    if np.any(roles < -1) or np.any(roles >= role_count):
        raise ValueError("carbon state axis-role index is out of range")
    if np.any((roles < 0) != (positions < 0)):
        raise ValueError("carbon state axis role and position masks drift")
    for role, extent in enumerate(extents):
        selected = roles == role
        if np.any(positions[selected] >= extent):
            raise ValueError("carbon state axis position is out of range")


def _carbon_tokens(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    normalization: NeuralInputNormalization,
    carbon_state_layout: Any,
) -> tuple[Any, Any]:
    carbon_parameters = parameters["shared_transformer"]["carbon"]
    if not carbon_parameters:
        raise ValueError("carbon state layout was not supplied during shared Transformer initialization")
    _validate_carbon_layout_parameters(carbon_parameters, carbon_state_layout)
    state_indices_host = np.asarray(carbon_state_layout.state_indices)
    state_size = int(np.prod(inputs.state.other.shape, dtype=np.int64))
    if np.any(state_indices_host < 0) or np.any(state_indices_host >= state_size):
        raise ValueError("carbon state index is out of range")
    indices = jnp.asarray(carbon_state_layout.state_indices, dtype=jnp.int32)
    raw = jnp.asarray(inputs.state.other)[indices]
    mean = jnp.asarray(normalization.continuous_mean)[indices]
    scale = jnp.asarray(normalization.continuous_scale)[indices]
    defined = jnp.asarray(inputs.state_defined.other, dtype=jnp.bool_)[indices] & jnp.isfinite(raw)
    normalized = standardize_continuous(
        jnp.where(defined, raw, mean),
        mean,
        scale,
        normalization.continuous_tail_threshold,
        coordinate_mask=(
            None
            if normalization.continuous_tail_mask is None
            else jnp.asarray(normalization.continuous_tail_mask)[indices]
        ),
    )
    shared = parameters["shared_transformer"]
    tokens = _linear(
        shared["local_value_input"],
        jnp.stack((jnp.asarray(normalized, dtype=jnp.float32), defined.astype(jnp.float32)), axis=1),
    )
    tokens += carbon_parameters["field_embedding"][jnp.asarray(carbon_state_layout.field_index, dtype=jnp.int32)]
    roles = jnp.asarray(carbon_state_layout.axis_role_index, dtype=jnp.int32)
    positions = jnp.asarray(carbon_state_layout.axis_position_index, dtype=jnp.int32)
    valid_roles = roles >= 0
    safe_roles = jnp.maximum(roles, 0)
    tokens += jnp.sum(
        jnp.where(valid_roles[..., None], carbon_parameters["axis_role_embedding"][safe_roles], 0.0),
        axis=1,
    )
    for role, embedding in enumerate(carbon_parameters["axis_position_embedding"]):
        selected = roles == role
        safe_positions = jnp.where(selected, positions, 0)
        tokens += jnp.sum(jnp.where(selected[..., None], embedding[safe_positions], 0.0), axis=1)
    return jnp.where(defined[:, None], tokens, 0.0), defined


def _decode_field(
    parameters: Mapping[str, Any],
    context: SharedTransformerContext,
    inputs: AssembledDailyOperatorInput,
    family_tokens: Any,
    family_mask: Any,
    *,
    field: Any,
    ownership: Any,
    field_index: int,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    config: SharedTransformerConfig,
    carbon_tokens: Any | None,
    carbon_defined: Any | None,
    carbon_state_layout: Any | None,
    projection_cache: tuple[_DecoderBlockProjection, ...] | None,
) -> tuple[Any, Any]:
    gather = config.decoder_execution != "original"
    coordinate_indices = (
        _static_field_coordinate_indices(field, ownership, layout) if gather else None
    )
    _full_coordinates, full_active, _full_owner = _field_active_and_pft_owner(
        context, field, ownership, layout
    )
    (
        features,
        active,
        pft_owner,
        axis_tokens,
        axis_mask,
        local_token,
        local_defined,
    ) = _field_coordinate_inputs(
        parameters,
        context,
        inputs,
        field=field,
        ownership=ownership,
        field_index=field_index,
        layout=layout,
        input_normalization=input_normalization,
        coordinate_indices=coordinate_indices,
    )
    expert_index = int(layout.field_expert_index[field_index])
    size = int(features.shape[0])
    if size == 0:
        return (
            jnp.zeros(field.shape, dtype=jnp.float32),
            full_active.reshape(field.shape),
        )
    chunk = min(int(config.output_chunk_size), size)
    padded_size = ((size + chunk - 1) // chunk) * chunk
    pad = padded_size - size
    chunk_count = padded_size // chunk

    def padded(value: Any, trailing: tuple[int, ...]) -> Any:
        return jnp.pad(value, ((0, pad),) + tuple((0, 0) for _ in trailing))

    feature_chunks = padded(features, features.shape[1:]).reshape((chunk_count, chunk, config.width))
    active_chunks = jnp.pad(active, ((0, pad),)).reshape((chunk_count, chunk))
    owner_chunks = jnp.pad(pft_owner, ((0, pad),), constant_values=-1).reshape((chunk_count, chunk))
    axis_chunks = padded(axis_tokens, axis_tokens.shape[1:]).reshape(
        (chunk_count, chunk, axis_tokens.shape[1], config.width)
    )
    axis_mask_chunks = padded(axis_mask, axis_mask.shape[1:]).reshape(
        (chunk_count, chunk, axis_mask.shape[1])
    )
    local_chunks = padded(local_token, local_token.shape[1:]).reshape(
        (chunk_count, chunk, config.width)
    )
    local_mask_chunks = jnp.pad(local_defined & active, ((0, pad),)).reshape((chunk_count, chunk))
    shared = parameters["shared_transformer"]

    def decode_chunk(values: tuple[Any, ...]) -> Any:
        chunk_features, chunk_active, chunk_owner, chunk_axis, chunk_axis_mask, chunk_local, chunk_local_mask = values
        count = chunk_features.shape[0]
        family_valid = chunk_active[:, None] & family_mask[expert_index][None, :]
        raw_mask = None
        if field.name == GPP_FIELD and carbon_tokens is not None:
            carbon_owner = jnp.asarray(carbon_state_layout.pft_owner_index, dtype=jnp.int32)
            raw_mask = carbon_defined[None, :] & (
                (carbon_owner[None, :] == -1) | (carbon_owner[None, :] == chunk_owner[:, None])
            )
            raw_mask &= chunk_active[:, None]
        if projection_cache is None:
            families = jnp.broadcast_to(
                family_tokens[expert_index][None, :, :],
                (count, family_tokens.shape[1], config.width),
            )
            process = jnp.broadcast_to(
                context.process_tokens[expert_index], (count, 1, config.width)
            )
            bank = jnp.concatenate(
                (families, chunk_axis, chunk_local[:, None, :], process), axis=1
            )
            mask = jnp.concatenate(
                (
                    family_valid,
                    chunk_axis_mask & chunk_active[:, None],
                    chunk_local_mask[:, None],
                    chunk_active[:, None],
                ),
                axis=1,
            )
            if raw_mask is not None:
                raw_carbon = jnp.broadcast_to(
                    carbon_tokens[None, :, :],
                    (count, carbon_tokens.shape[0], config.width),
                )
                bank = jnp.concatenate((bank, raw_carbon), axis=1)
                mask = jnp.concatenate((mask, raw_mask), axis=1)
            encoded = _run_blocks(
                chunk_features[:, None, :],
                bank,
                mask[:, None, :],
                shared["coordinate_blocks"],
                config,
            )[:, 0, :]
        else:
            dynamic = jnp.concatenate((chunk_axis, chunk_local[:, None, :]), axis=1)
            dynamic_valid = jnp.concatenate(
                (
                    chunk_axis_mask & chunk_active[:, None],
                    chunk_local_mask[:, None],
                ),
                axis=1,
            )
            encoded = _run_coordinate_blocks_reusing_projections(
                chunk_features[:, None, :],
                dynamic,
                dynamic_mask=dynamic_valid[:, None, :],
                family_mask=family_valid[:, None, :],
                process_mask=chunk_active[:, None, None],
                carbon_mask=(
                    raw_mask[:, None, :]
                    if raw_mask is not None
                    else None
                ),
                blocks=shared["coordinate_blocks"],
                projections=projection_cache,
                config=config,
            )[:, 0, :]
        raw = _linear(
            shared["readout_second"],
            jax.nn.gelu(_linear(shared["readout_first"], encoded)),
        )[:, 0]
        return jnp.where(chunk_active, raw, 0.0)

    decoded = jax.lax.map(
        decode_chunk,
        (
            feature_chunks,
            active_chunks,
            owner_chunks,
            axis_chunks,
            axis_mask_chunks,
            local_chunks,
            local_mask_chunks,
        ),
    ).reshape(-1)[:size]
    if coordinate_indices is None:
        raw = decoded
    else:
        raw = jnp.zeros((int(np.prod(field.shape, dtype=np.int64)),), dtype=decoded.dtype)
        raw = raw.at[jnp.asarray(coordinate_indices, dtype=jnp.int32)].set(decoded)
    return raw.reshape(field.shape), full_active.reshape(field.shape)


def shared_transformer_raw_outputs(
    parameters: Mapping[str, Any],
    context: SharedTransformerContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    config: SharedTransformerConfig = SharedTransformerConfig(),
    carbon_state_layout: Any | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return unconstrained raw coordinates and exact applicability masks."""

    config.validate()
    if tuple(field.name for field in schema.fields) != tuple(field.name for field in layout.ledger):
        raise ValueError("shared Transformer schema and layout drift")
    family_tokens, family_mask = _source_family_bank(context, layout)
    raw_carbon = carbon_defined = None
    if carbon_state_layout is not None:
        raw_carbon, carbon_defined = _carbon_tokens(
            parameters,
            inputs,
            input_normalization,
            carbon_state_layout,
        )
    projection_cache = None
    if config.decoder_execution == "gather_reuse":
        projection_cache = _decoder_projection_cache(
            parameters,
            context,
            family_tokens,
            carbon_tokens=raw_carbon,
        )
    outputs = {}
    masks = {}
    for field_index, (field, ownership) in enumerate(zip(schema.fields, layout.ledger, strict=True)):
        raw, active = _decode_field(
            parameters,
            context,
            inputs,
            family_tokens,
            family_mask,
            field=field,
            ownership=ownership,
            field_index=field_index,
            layout=layout,
            input_normalization=input_normalization,
            config=config,
            carbon_tokens=raw_carbon,
            carbon_defined=carbon_defined,
            carbon_state_layout=carbon_state_layout,
            projection_cache=(
                None
                if projection_cache is None
                else projection_cache[int(layout.field_expert_index[field_index])]
            ),
        )
        outputs[field.name] = raw
        masks[field.name] = active
    return outputs, masks


__all__ = [
    "SharedTransformerConfig",
    "SharedTransformerContext",
    "assemble_shared_transformer_context",
    "initialize_shared_transformer_parameters",
    "shared_transformer_output_masks",
    "shared_transformer_parameter_count",
    "shared_transformer_raw_outputs",
    "shared_transformer_static_coordinate_indices",
]
