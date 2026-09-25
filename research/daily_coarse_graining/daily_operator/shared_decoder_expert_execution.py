"""Diagnostic expert-grouped execution adapter for shared raw outputs."""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from .shared_decoder_dense import DenseDecoderField
from .shared_decoder_macro_packed import expert_decode_fields
from .shared_transformer import (
    SharedTransformerConfig,
    _carbon_tokens,
    _decode_field,
    _decoder_dynamic_plan,
    _decoder_dynamic_tokens,
    _decoder_projection_cache,
    _field_active_and_pft_owner,
    _field_coordinate_inputs,
    _field_dynamic_projection_cache,
    _project_decoder_context,
    _source_family_bank,
    _static_field_coordinate_indices,
)


def shared_transformer_raw_outputs_expert128(
    parameters: Mapping[str, Any],
    context: Any,
    inputs: Any,
    *,
    schema: Any,
    layout: Any,
    input_normalization: Any,
    config: SharedTransformerConfig = SharedTransformerConfig(),
    carbon_state_layout: Any | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return canonical raw outputs with non-GPP rows grouped by expert.

    This is an opt-in diagnostic route. Canonical feature assembly, masks,
    context/family projections, and static eligible-coordinate restoration
    follow the shared-transformer path. GPP is decoded by the unchanged
    ``_decode_field`` implementation.
    """

    config.validate()
    if config.decoder_execution != "gather_dynamic_reuse":
        raise ValueError("expert128 execution requires gather_dynamic_reuse configuration")
    if tuple(field.name for field in schema.fields) != tuple(field.name for field in layout.ledger):
        raise ValueError("shared Transformer schema and layout drift")

    family_tokens, family_mask = _source_family_bank(context, layout)
    decoder_context, family_tokens = _project_decoder_context(
        parameters, context, family_tokens, family_mask,
    )
    raw_carbon = carbon_defined = None
    if carbon_state_layout is not None:
        raw_carbon, carbon_defined = _carbon_tokens(
            parameters, inputs, input_normalization, carbon_state_layout,
        )
    dynamic_plan = _decoder_dynamic_plan(schema=schema, layout=layout)
    expert_projections = _decoder_projection_cache(
        parameters,
        decoder_context,
        family_tokens,
        carbon_tokens=raw_carbon,
    )

    outputs_by_index: list[Any | None] = [None] * len(schema.fields)
    masks_by_index: list[Any | None] = [None] * len(schema.fields)
    packed_fields: list[DenseDecoderField] = []
    packed_experts: list[int] = []
    packed_indices: list[np.ndarray] = []
    packed_positions: list[int] = []

    for field_index, (field, ownership) in enumerate(zip(schema.fields, layout.ledger, strict=True)):
        expert_index = int(layout.field_expert_index[field_index])
        coordinate_indices = np.asarray(
            _static_field_coordinate_indices(field, ownership, layout), dtype=np.int32,
        )
        _coordinates, full_active, _owner = _field_active_and_pft_owner(
            decoder_context, field, ownership, layout,
        )
        indices = dynamic_plan.field_axis_bank_indices[field_index]
        local_indices = dynamic_plan.field_local_bank_indices[field_index]
        dynamic_tokens = _decoder_dynamic_tokens(
            parameters,
            decoder_context,
            inputs,
            input_normalization=input_normalization,
            layout=layout,
            plan=dynamic_plan,
            axis_bank_indices=indices,
            local_bank_indices=local_indices,
        )
        field_projections = _field_dynamic_projection_cache(
            parameters,
            expert_projections[expert_index],
            dynamic_tokens,
        )

        # Preserve the original GPP decoder path and its carbon-bank behavior.
        if field.name == "daily.fast_gpp":
            raw, active = _decode_field(
                parameters,
                decoder_context,
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
                projection_cache=field_projections,
                dynamic_projection_indices=(
                    dynamic_plan.field_axis_indices[field_index],
                    dynamic_plan.field_local_indices[field_index],
                ),
            )
            outputs_by_index[field_index] = raw
            masks_by_index[field_index] = active
            continue

        if coordinate_indices.size == 0:
            outputs_by_index[field_index] = jnp.zeros(field.shape, dtype=jnp.float32)
            masks_by_index[field_index] = full_active.reshape(field.shape)
            continue

        (
            features,
            active,
            _pft_owner,
            axis_tokens,
            axis_mask,
            local_token,
            local_defined,
        ) = _field_coordinate_inputs(
            parameters,
            decoder_context,
            inputs,
            field=field,
            ownership=ownership,
            field_index=field_index,
            layout=layout,
            input_normalization=input_normalization,
            coordinate_indices=coordinate_indices,
        )
        dynamic = jnp.concatenate((axis_tokens, local_token[:, None, :]), axis=1)
        dynamic_mask = jnp.concatenate(
            (axis_mask & active[:, None], (local_defined & active)[:, None]), axis=1,
        )[:, None, :]
        packed_fields.append(
            DenseDecoderField(
                query=features[:, None, :],
                dynamic_tokens=dynamic,
                dynamic_mask=dynamic_mask,
                family_mask=(active[:, None, None] & family_mask[expert_index][None, None, :]),
                process_mask=active[:, None, None],
                carbon_mask=None,
                projections=field_projections,
                dynamic_projection_indices=(
                    dynamic_plan.field_axis_indices[field_index],
                    dynamic_plan.field_local_indices[field_index],
                    active,
                ),
                output_active=active,
            )
        )
        packed_experts.append(expert_index)
        packed_indices.append(coordinate_indices)
        packed_positions.append(field_index)
        masks_by_index[field_index] = full_active.reshape(field.shape)

    if packed_fields:
        packed_outputs = expert_decode_fields(
            parameters["shared_transformer"],
            tuple(packed_fields),
            tuple(packed_experts),
            expert_projections,
            config,
            chunk_size=128,
        )
        for field_index, coordinate_indices, decoded in zip(
            packed_positions, packed_indices, packed_outputs, strict=True,
        ):
            field = schema.fields[field_index]
            raw = jnp.zeros((int(np.prod(field.shape, dtype=np.int64)),), dtype=decoded.dtype)
            raw = raw.at[jnp.asarray(coordinate_indices, dtype=jnp.int32)].set(decoded)
            outputs_by_index[field_index] = raw.reshape(field.shape)

    return (
        {field.name: outputs_by_index[index] for index, field in enumerate(schema.fields)},
        {field.name: masks_by_index[index] for index, field in enumerate(schema.fields)},
    )
