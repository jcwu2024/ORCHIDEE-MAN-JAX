"""Bounded block-major dense execution for shared decoder fields.

This prototype changes only the scheduling of row-independent dense stages.
Attention remains field-local so heterogeneous banks, masks, and carbon routes
retain the production shared-transformer semantics.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp

from .shared_transformer import (
    SharedTransformerConfig,
    _attention_from_projected_segments,
    _DecoderBlockProjection,
    _run_coordinate_blocks_reusing_projections,
)
from .strong_baseline_architecture import _ffn, _layer_norm, _linear


class DenseDecoderField(NamedTuple):
    """One field's already assembled coordinate-decoder payload.

    ``query`` has shape ``[Q, 1, W]`` and ``dynamic_tokens`` has shape
    ``[Q, A + 1, W]``. Masks retain their production attention shapes. The
    optional dynamic indices address projections made by
    ``_field_dynamic_projection_cache``.
    """

    query: Any
    dynamic_tokens: Any
    dynamic_mask: Any
    family_mask: Any
    process_mask: Any
    carbon_mask: Any | None
    projections: tuple[_DecoderBlockProjection, ...]
    dynamic_projection_indices: tuple[Any, Any, Any] | None
    output_active: Any


def _projected_dynamic(
    block: Mapping[str, Any],
    projected: _DecoderBlockProjection,
    indices: tuple[Any, Any, Any] | None,
) -> tuple[Any | None, Any | None]:
    if indices is None:
        return None, None
    axis_indices, local_indices, dynamic_active = indices
    dynamic_key = jnp.concatenate(
        (
            projected.axis_key[axis_indices],
            projected.local_key[local_indices, None, :],
        ),
        axis=1,
    )
    dynamic_value = jnp.concatenate(
        (
            projected.axis_value[axis_indices],
            projected.local_value[local_indices, None, :],
        ),
        axis=1,
    )
    dynamic_key = jnp.where(
        dynamic_active[:, None, None],
        dynamic_key,
        block["attention"]["key"]["bias"],
    )
    dynamic_value = jnp.where(
        dynamic_active[:, None, None],
        dynamic_value,
        block["attention"]["value"]["bias"],
    )
    return dynamic_key, dynamic_value


def _attention_message(
    field: DenseDecoderField,
    query: Any,
    block: Mapping[str, Any],
    projected: _DecoderBlockProjection,
    config: SharedTransformerConfig,
) -> Any:
    dynamic_key, dynamic_value = _projected_dynamic(
        block,
        projected,
        field.dynamic_projection_indices,
    )
    return _attention_from_projected_segments(
        block["attention"],
        query,
        field.dynamic_tokens,
        dynamic_key=dynamic_key,
        dynamic_value=dynamic_value,
        family_key=projected.family_key,
        family_value=projected.family_value,
        family_mask=field.family_mask,
        dynamic_mask=field.dynamic_mask,
        process_key=projected.process_key,
        process_value=projected.process_value,
        process_mask=field.process_mask,
        carbon_key=projected.carbon_key,
        carbon_value=projected.carbon_value,
        carbon_mask=field.carbon_mask,
        heads=config.attention_heads,
        normalization_epsilon=config.normalization_epsilon,
    )


def _map_dense_rows(function: Any, values: Any, chunk_size: int) -> Any:
    row_count = int(values.shape[0])
    if chunk_size <= 0 or chunk_size >= row_count:
        return function(values)
    padded_count = ((row_count + chunk_size - 1) // chunk_size) * chunk_size
    pad = padded_count - row_count
    padded = jnp.pad(values, ((0, pad),) + ((0, 0),) * (values.ndim - 1))
    chunks = padded.reshape((padded_count // chunk_size, chunk_size) + values.shape[1:])
    mapped = jax.lax.map(function, chunks)
    return mapped.reshape((padded_count,) + mapped.shape[2:])[:row_count]


def _readout(
    shared_parameters: Mapping[str, Any],
    encoded: Any,
    *,
    dense_chunk_size: int,
) -> Any:
    def apply(values: Any) -> Any:
        return _linear(
            shared_parameters["readout_second"],
            jax.nn.gelu(_linear(shared_parameters["readout_first"], values)),
        )

    return _map_dense_rows(apply, encoded, dense_chunk_size)[..., 0]


def _validate_fields(
    shared_parameters: Mapping[str, Any],
    fields: tuple[DenseDecoderField, ...],
) -> None:
    blocks = shared_parameters["coordinate_blocks"]
    if not fields:
        raise ValueError("at least one decoder field is required")
    for field in fields:
        if field.query.ndim != 3 or field.query.shape[1] != 1:
            raise ValueError("field query must have shape [Q, 1, W]")
        if len(field.projections) != len(blocks):
            raise ValueError("each field must provide one projection per coordinate block")
        if field.output_active.shape != (field.query.shape[0],):
            raise ValueError("field output_active must have shape [Q]")


def reference_decode_fields(
    shared_parameters: Mapping[str, Any],
    fields: tuple[DenseDecoderField, ...],
    config: SharedTransformerConfig,
) -> tuple[Any, ...]:
    """Decode fields through the current production per-field block path."""

    _validate_fields(shared_parameters, fields)
    outputs = []
    for field in fields:
        encoded = _run_coordinate_blocks_reusing_projections(
            field.query,
            field.dynamic_tokens,
            dynamic_mask=field.dynamic_mask,
            family_mask=field.family_mask,
            process_mask=field.process_mask,
            carbon_mask=field.carbon_mask,
            blocks=shared_parameters["coordinate_blocks"],
            projections=field.projections,
            config=config,
            dynamic_projection_indices=field.dynamic_projection_indices,
        )[:, 0, :]
        raw = _readout(shared_parameters, encoded, dense_chunk_size=0)
        outputs.append(jnp.where(field.output_active, raw, 0.0))
    return tuple(outputs)


def packed_decode_fields(
    shared_parameters: Mapping[str, Any],
    fields: tuple[DenseDecoderField, ...],
    config: SharedTransformerConfig,
    *,
    dense_chunk_size: int = 0,
) -> tuple[Any, ...]:
    """Decode fields with block-major packed normalization, FFN, and readout."""

    _validate_fields(shared_parameters, fields)
    if dense_chunk_size < 0:
        raise ValueError("dense_chunk_size must be nonnegative")
    row_counts = tuple(int(field.query.shape[0]) for field in fields)
    queries = tuple(field.query for field in fields)

    for block_index, block in enumerate(shared_parameters["coordinate_blocks"]):
        messages = tuple(
            _attention_message(
                field,
                query,
                block,
                field.projections[block_index],
                config,
            )
            for field, query in zip(fields, queries, strict=True)
        )
        packed = jnp.concatenate(
            tuple(query + message for query, message in zip(queries, messages, strict=True)),
            axis=0,
        )
        packed = _map_dense_rows(
            lambda values: _layer_norm(
                values,
                block["attention_norm_scale"],
                block["attention_norm_bias"],
            ),
            packed,
            dense_chunk_size,
        )
        packed = _map_dense_rows(
            lambda values: _layer_norm(
                values + _ffn(block["ffn"], values),
                block["ffn_norm_scale"],
                block["ffn_norm_bias"],
            ),
            packed,
            dense_chunk_size,
        )
        split_points = tuple(sum(row_counts[:index]) for index in range(1, len(row_counts)))
        queries = tuple(jnp.split(packed, split_points, axis=0))

    encoded = jnp.concatenate(tuple(query[:, 0, :] for query in queries), axis=0)
    packed_raw = _readout(
        shared_parameters,
        encoded,
        dense_chunk_size=dense_chunk_size,
    )
    raw_fields = jnp.split(
        packed_raw,
        tuple(sum(row_counts[:index]) for index in range(1, len(row_counts))),
        axis=0,
    )
    return tuple(
        jnp.where(field.output_active, raw, 0.0)
        for field, raw in zip(fields, raw_fields, strict=True)
    )
