"""Diagnostic same-expert decoder grouping with field-local K/V banks.

The reference retains one production-shaped chunk map per field. Grouped
execution uses one map across compatible fields and keeps every field's
dynamic projection bank and attention mask distinct. This is not wired into
the production model.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from .shared_decoder_dense import DenseDecoderField
from .shared_transformer import _DecoderBlockProjection, _run_coordinate_blocks_reusing_projections
from .strong_baseline_architecture import _linear


def _decode_chunked(
    shared: Mapping[str, Any],
    *,
    query: Any,
    dynamic: Any,
    dynamic_mask: Any,
    family_mask: Any,
    process_mask: Any,
    active: Any,
    axis_indices: Any,
    local_indices: Any,
    projections: tuple[_DecoderBlockProjection, ...],
    config: Any,
    chunk_size: int,
) -> Any:
    rows = int(query.shape[0])
    chunk = min(chunk_size, rows)
    padded_rows = ((rows + chunk - 1) // chunk) * chunk
    pad = padded_rows - rows

    def chunks(value: Any) -> Any:
        value = jnp.pad(value, ((0, pad),) + ((0, 0),) * (value.ndim - 1))
        return value.reshape((padded_rows // chunk, chunk) + value.shape[1:])

    values = tuple(
        chunks(value)
        for value in (
            query,
            dynamic,
            dynamic_mask,
            family_mask,
            process_mask,
            active,
            axis_indices,
            local_indices,
        )
    )

    def decode_chunk(items: tuple[Any, ...]) -> Any:
        (
            chunk_query,
            chunk_dynamic,
            chunk_dynamic_mask,
            chunk_family_mask,
            chunk_process_mask,
            chunk_active,
            chunk_axis_indices,
            chunk_local_indices,
        ) = items
        encoded = _run_coordinate_blocks_reusing_projections(
            chunk_query,
            chunk_dynamic,
            dynamic_mask=chunk_dynamic_mask,
            family_mask=chunk_family_mask,
            process_mask=chunk_process_mask,
            carbon_mask=None,
            blocks=shared["coordinate_blocks"],
            projections=projections,
            config=config,
            dynamic_projection_indices=(chunk_axis_indices, chunk_local_indices, chunk_active),
        )[:, 0, :]
        raw = _linear(
            shared["readout_second"],
            jax.nn.gelu(_linear(shared["readout_first"], encoded)),
        )[:, 0]
        return jnp.where(chunk_active, raw, 0.0)

    return jax.lax.map(decode_chunk, values).reshape(-1)[:rows]


def _validate_fields(shared: Mapping[str, Any], fields: tuple[DenseDecoderField, ...], config: Any) -> None:
    if not fields:
        raise ValueError("at least one field is required")
    first = fields[0]
    width = config.resolved_decoder_width
    rows = int(first.query.shape[0])
    dynamic_slots = int(first.dynamic_tokens.shape[1])
    family_count = int(first.family_mask.shape[-1])
    axis_bank = int(first.projections[0].axis_key.shape[0])
    local_bank = int(first.projections[0].local_key.shape[0])
    blocks = len(shared["coordinate_blocks"])
    if rows <= 0 or dynamic_slots <= 0 or axis_bank <= 0 or local_bank <= 0:
        raise ValueError("nonempty rows and dynamic banks are required")
    for field in fields:
        indices = field.dynamic_projection_indices
        if field.query.shape != (rows, 1, width) or field.dynamic_tokens.shape != (rows, dynamic_slots, width):
            raise ValueError("fields must have an identical query/dynamic shape")
        if field.family_mask.shape != (rows, 1, family_count):
            raise ValueError("family masks must share the expert bank shape")
        if field.dynamic_mask.shape != (rows, 1, dynamic_slots) or field.process_mask.shape != (rows, 1, 1):
            raise ValueError("dynamic/process mask shape mismatch")
        if field.output_active.shape != (rows,) or field.carbon_mask is not None:
            raise ValueError("active shape mismatch or unsupported carbon bank")
        if indices is None or indices[0].shape != (rows, dynamic_slots - 1) or indices[1].shape != (rows,):
            raise ValueError("field-local projection index shape mismatch")
        if len(field.projections) != blocks:
            raise ValueError("coordinate-block count mismatch")
        for projected in field.projections:
            if projected.axis_key.shape != (axis_bank, width) or projected.axis_value.shape != (axis_bank, width):
                raise ValueError("axis projection bank shape mismatch")
            if projected.local_key.shape != (local_bank, width) or projected.local_value.shape != (local_bank, width):
                raise ValueError("local projection bank shape mismatch")


def reference_decode_fields_chunked(
    shared: Mapping[str, Any], fields: tuple[DenseDecoderField, ...], config: Any, *, chunk_size: int
) -> tuple[Any, ...]:
    """Preserve separate field maps at the selected coordinate chunk size."""

    _validate_fields(shared, fields, config)
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    return tuple(
        _decode_chunked(
            shared,
            query=field.query,
            dynamic=field.dynamic_tokens,
            dynamic_mask=field.dynamic_mask,
            family_mask=field.family_mask,
            process_mask=field.process_mask,
            active=field.output_active,
            axis_indices=field.dynamic_projection_indices[0],
            local_indices=field.dynamic_projection_indices[1],
            projections=field.projections,
            config=config,
            chunk_size=chunk_size,
        )
        for field in fields
    )


def reference_decode_fields_chunk64(
    shared: Mapping[str, Any], fields: tuple[DenseDecoderField, ...], config: Any
) -> tuple[Any, ...]:
    """Preserve five separate field maps, each with production 64-row chunks."""

    return reference_decode_fields_chunked(shared, fields, config, chunk_size=64)


def grouped_decode_fields(
    shared: Mapping[str, Any],
    fields: tuple[DenseDecoderField, ...],
    config: Any,
    *,
    chunk_size: int,
) -> tuple[Any, ...]:
    """Pack same-expert fields; retain distinct dynamic banks and output slices."""

    _validate_fields(shared, fields, config)
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    rows = int(fields[0].query.shape[0])
    axis_bank = int(fields[0].projections[0].axis_key.shape[0])
    local_bank = int(fields[0].projections[0].local_key.shape[0])
    projections = []
    for block_index in range(len(shared["coordinate_blocks"])):
        first = fields[0].projections[block_index]
        projections.append(
            _DecoderBlockProjection(
                family_key=first.family_key,
                family_value=first.family_value,
                process_key=first.process_key,
                process_value=first.process_value,
                carbon_key=None,
                carbon_value=None,
                axis_key=jnp.concatenate(tuple(field.projections[block_index].axis_key for field in fields)),
                axis_value=jnp.concatenate(tuple(field.projections[block_index].axis_value for field in fields)),
                local_key=jnp.concatenate(tuple(field.projections[block_index].local_key for field in fields)),
                local_value=jnp.concatenate(tuple(field.projections[block_index].local_value for field in fields)),
            )
        )
    axis_indices = jnp.concatenate(
        tuple(field.dynamic_projection_indices[0] + index * axis_bank for index, field in enumerate(fields))
    )
    local_indices = jnp.concatenate(
        tuple(field.dynamic_projection_indices[1] + index * local_bank for index, field in enumerate(fields))
    )
    decoded = _decode_chunked(
        shared,
        query=jnp.concatenate(tuple(field.query for field in fields)),
        dynamic=jnp.concatenate(tuple(field.dynamic_tokens for field in fields)),
        dynamic_mask=jnp.concatenate(tuple(field.dynamic_mask for field in fields)),
        family_mask=jnp.concatenate(tuple(field.family_mask for field in fields)),
        process_mask=jnp.concatenate(tuple(field.process_mask for field in fields)),
        active=jnp.concatenate(tuple(field.output_active for field in fields)),
        axis_indices=axis_indices,
        local_indices=local_indices,
        projections=tuple(projections),
        config=config,
        chunk_size=chunk_size,
    )
    return tuple(decoded[index * rows : (index + 1) * rows] for index in range(len(fields)))
