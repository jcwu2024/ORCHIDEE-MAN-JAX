"""Diagnostic cross-expert packing of non-GPP coordinate-decoder rows.

Family/process banks are selected per row from already projected expert
caches. The production segmented-attention helper supplies the exact mask,
head normalization, softmax and value-sum arithmetic. No production route
imports this module.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from .shared_decoder_dense import DenseDecoderField
from .shared_transformer import (
    _attention_from_projected_segments,
    _DecoderBlockProjection,
    _run_coordinate_blocks_reusing_projections,
)
from .strong_baseline_architecture import _ffn, _layer_norm, _linear


def _decode_mapped_rows(
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
    """The production-shaped field-local chunk map used as the reference."""

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
            query, dynamic, dynamic_mask, family_mask, process_mask,
            active, axis_indices, local_indices,
        )
    )

    def decode_chunk(items: tuple[Any, ...]) -> Any:
        q, dyn, dyn_mask, fam_mask, proc_mask, on, axis_idx, local_idx = items
        encoded = _run_coordinate_blocks_reusing_projections(
            q,
            dyn,
            dynamic_mask=dyn_mask,
            family_mask=fam_mask,
            process_mask=proc_mask,
            carbon_mask=None,
            blocks=shared["coordinate_blocks"],
            projections=projections,
            config=config,
            dynamic_projection_indices=(axis_idx, local_idx, on),
        )[:, 0, :]
        raw = _linear(shared["readout_second"], jax.nn.gelu(_linear(shared["readout_first"], encoded)))[:, 0]
        return jnp.where(on, raw, 0.0)

    return jax.lax.map(decode_chunk, values).reshape(-1)[:rows]


def reference_decode_fields(
    shared: Mapping[str, Any],
    fields: tuple[DenseDecoderField, ...],
    config: Any,
) -> tuple[Any, ...]:
    """Preserve each field's own 64-row production decoder map."""

    _validate_fields(shared, fields, config)
    return tuple(
        _decode_mapped_rows(
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
            chunk_size=64,
        )
        for field in fields
    )


def _validate_fields(shared: Mapping[str, Any], fields: tuple[DenseDecoderField, ...], config: Any) -> None:
    if not fields:
        raise ValueError("at least one field is required")
    width = config.resolved_decoder_width
    blocks = len(shared["coordinate_blocks"])
    family_count = int(fields[0].family_mask.shape[-1])
    for field in fields:
        rows = int(field.query.shape[0])
        slots = int(field.dynamic_tokens.shape[1]) - 1
        indices = field.dynamic_projection_indices
        if rows <= 0 or slots < 0 or field.query.shape != (rows, 1, width):
            raise ValueError("field query shape mismatch")
        if field.dynamic_tokens.shape != (rows, slots + 1, width):
            raise ValueError("field dynamic-token shape mismatch")
        if field.family_mask.shape != (rows, 1, family_count):
            raise ValueError("field family-mask shape mismatch")
        if field.dynamic_mask.shape != (rows, 1, slots + 1) or field.process_mask.shape != (rows, 1, 1):
            raise ValueError("field dynamic/process-mask shape mismatch")
        if field.output_active.shape != (rows,) or field.carbon_mask is not None:
            raise ValueError("inactive-mask shape mismatch or unsupported carbon bank")
        if indices is None or indices[0].shape != (rows, slots) or indices[1].shape != (rows,):
            raise ValueError("field projection-index shape mismatch")
        if len(field.projections) != blocks:
            raise ValueError("field block-count mismatch")
        for projected in field.projections:
            if projected.axis_key.shape[-1] != width or projected.local_key.shape[-1] != width:
                raise ValueError("field projection-width mismatch")
            if projected.axis_key.shape != projected.axis_value.shape:
                raise ValueError("field axis K/V shape mismatch")
            if projected.local_key.shape != projected.local_value.shape or projected.local_key.shape[0] <= 0:
                raise ValueError("field local K/V shape mismatch")


def packed_decode_fields(
    shared: Mapping[str, Any],
    fields: tuple[DenseDecoderField, ...],
    expert_indices: tuple[int, ...],
    expert_projections: tuple[tuple[_DecoderBlockProjection, ...], ...],
    config: Any,
    *,
    chunk_size: int,
) -> tuple[Any, ...]:
    """Pack non-carbon rows while selecting family/process K/V per query."""

    _validate_fields(shared, fields, config)
    if len(fields) != len(expert_indices) or chunk_size <= 0:
        raise ValueError("one expert identity per field and positive chunk size required")
    if any(index < 0 or index >= len(expert_projections) for index in expert_indices):
        raise ValueError("expert index outside projected cache")
    blocks = shared["coordinate_blocks"]
    if any(len(item) != len(blocks) for item in expert_projections):
        raise ValueError("expert projection block count mismatch")
    max_slots = max(int(field.dynamic_tokens.shape[1]) - 1 for field in fields)
    sizes = tuple(int(field.query.shape[0]) for field in fields)
    row_experts = jnp.concatenate(
        tuple(jnp.full((count,), expert, dtype=jnp.int32) for count, expert in zip(sizes, expert_indices, strict=True))
    )

    def axis_and_local(field: DenseDecoderField, value: Any) -> Any:
        slots = int(field.dynamic_tokens.shape[1]) - 1
        axis = jnp.pad(value[:, :slots], ((0, 0), (0, max_slots - slots), (0, 0)))
        return jnp.concatenate((axis, value[:, slots : slots + 1]), axis=1)

    def padded_dynamic_mask(field: DenseDecoderField) -> Any:
        slots = int(field.dynamic_tokens.shape[1]) - 1
        mask = field.dynamic_mask[:, 0, :]
        axis = jnp.pad(mask[:, :slots], ((0, 0), (0, max_slots - slots)))
        return jnp.concatenate((axis, mask[:, slots : slots + 1]), axis=1)[:, None, :]

    query = jnp.concatenate(tuple(field.query for field in fields))
    dynamic = jnp.concatenate(tuple(axis_and_local(field, field.dynamic_tokens) for field in fields))
    dynamic_mask = jnp.concatenate(tuple(padded_dynamic_mask(field) for field in fields))
    family_mask = jnp.concatenate(tuple(field.family_mask for field in fields))
    process_mask = jnp.concatenate(tuple(field.process_mask for field in fields))
    active = jnp.concatenate(tuple(field.output_active for field in fields))
    axis_index_parts = []
    local_index_parts = []
    axis_offset = local_offset = 0
    for field in fields:
        slots = int(field.dynamic_tokens.shape[1]) - 1
        axis_index_parts.append(
            jnp.pad(
                field.dynamic_projection_indices[0] + axis_offset,
                ((0, 0), (0, max_slots - slots)),
            )
        )
        local_index_parts.append(field.dynamic_projection_indices[1] + local_offset)
        axis_offset += int(field.projections[0].axis_key.shape[0])
        local_offset += int(field.projections[0].local_key.shape[0])
    axis_indices = jnp.concatenate(tuple(axis_index_parts))
    local_indices = jnp.concatenate(tuple(local_index_parts))

    projected_blocks = []
    for block_index in range(len(blocks)):
        projected_blocks.append(
            _DecoderBlockProjection(
                family_key=None,
                family_value=None,
                process_key=None,
                process_value=None,
                carbon_key=None,
                carbon_value=None,
                axis_key=jnp.concatenate(tuple(field.projections[block_index].axis_key for field in fields)),
                axis_value=jnp.concatenate(tuple(field.projections[block_index].axis_value for field in fields)),
                local_key=jnp.concatenate(tuple(field.projections[block_index].local_key for field in fields)),
                local_value=jnp.concatenate(tuple(field.projections[block_index].local_value for field in fields)),
            )
        )

    rows = int(query.shape[0])
    chunk = min(chunk_size, rows)
    padded_rows = ((rows + chunk - 1) // chunk) * chunk
    pad = padded_rows - rows

    def chunks(value: Any) -> Any:
        value = jnp.pad(value, ((0, pad),) + ((0, 0),) * (value.ndim - 1))
        return value.reshape((padded_rows // chunk, chunk) + value.shape[1:])

    mapped = tuple(
        chunks(value)
        for value in (
            query, dynamic, dynamic_mask, family_mask, process_mask,
            active, axis_indices, local_indices, row_experts,
        )
    )

    def decode_chunk(items: tuple[Any, ...]) -> Any:
        q, dyn, dyn_mask, fam_mask, proc_mask, on, axis_idx, local_idx, expert_idx = items
        for block_index, block in enumerate(blocks):
            projected = projected_blocks[block_index]
            dynamic_key = jnp.concatenate(
                (projected.axis_key[axis_idx], projected.local_key[local_idx, None, :]), axis=1
            )
            dynamic_value = jnp.concatenate(
                (projected.axis_value[axis_idx], projected.local_value[local_idx, None, :]), axis=1
            )
            dynamic_key = jnp.where(on[:, None, None], dynamic_key, block["attention"]["key"]["bias"])
            dynamic_value = jnp.where(on[:, None, None], dynamic_value, block["attention"]["value"]["bias"])
            expert_bank = expert_projections
            family_key = jnp.stack(tuple(item[block_index].family_key for item in expert_bank))[expert_idx]
            family_value = jnp.stack(tuple(item[block_index].family_value for item in expert_bank))[expert_idx]
            process_key = jnp.stack(tuple(item[block_index].process_key for item in expert_bank))[expert_idx]
            process_value = jnp.stack(tuple(item[block_index].process_value for item in expert_bank))[expert_idx]

            def attend_one(row_q, row_dyn, row_key, row_value, row_family_key,
                           row_family_value, row_family_mask, row_dynamic_mask,
                           row_process_key, row_process_value, row_process_mask):
                return _attention_from_projected_segments(
                    block["attention"],
                    row_q,
                    row_dyn,
                    dynamic_key=row_key,
                    dynamic_value=row_value,
                    family_key=row_family_key,
                    family_value=row_family_value,
                    family_mask=row_family_mask,
                    dynamic_mask=row_dynamic_mask,
                    process_key=row_process_key,
                    process_value=row_process_value,
                    process_mask=row_process_mask,
                    carbon_key=None,
                    carbon_value=None,
                    carbon_mask=None,
                    heads=config.resolved_decoder_attention_heads,
                    normalization_epsilon=config.normalization_epsilon,
                )

            message = jax.vmap(attend_one)(
                q, dyn, dynamic_key, dynamic_value, family_key, family_value,
                fam_mask, dyn_mask, process_key, process_value, proc_mask,
            )
            q = _layer_norm(q + message, block["attention_norm_scale"], block["attention_norm_bias"])
            q = _layer_norm(q + _ffn(block["ffn"], q), block["ffn_norm_scale"], block["ffn_norm_bias"])
        encoded = q[:, 0, :]
        raw = _linear(shared["readout_second"], jax.nn.gelu(_linear(shared["readout_first"], encoded)))[:, 0]
        return jnp.where(on, raw, 0.0)

    decoded = jax.lax.map(decode_chunk, mapped).reshape(-1)[:rows]
    offsets = (0,) + tuple(sum(sizes[:index]) for index in range(1, len(sizes))) + (rows,)
    return tuple(decoded[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=True))


def expert_decode_fields(
    shared: Mapping[str, Any],
    fields: tuple[DenseDecoderField, ...],
    expert_indices: tuple[int, ...],
    expert_projections: tuple[tuple[_DecoderBlockProjection, ...], ...],
    config: Any,
    *,
    chunk_size: int,
) -> tuple[Any, ...]:
    """Map heterogeneous fields per expert through production attention."""

    _validate_fields(shared, fields, config)
    if len(fields) != len(expert_indices) or chunk_size <= 0:
        raise ValueError("one expert identity per field and positive chunk size required")
    if any(index < 0 or index >= len(expert_projections) for index in expert_indices):
        raise ValueError("expert index outside projected cache")
    blocks = shared["coordinate_blocks"]
    if any(len(item) != len(blocks) for item in expert_projections):
        raise ValueError("expert projection block count mismatch")

    outputs: list[Any] = [None] * len(fields)
    for expert in sorted(set(expert_indices)):
        positions = tuple(index for index, owner in enumerate(expert_indices) if owner == expert)
        group = tuple(fields[index] for index in positions)
        max_slots = max(int(field.dynamic_tokens.shape[1]) - 1 for field in group)
        sizes = tuple(int(field.query.shape[0]) for field in group)

        def padded_dynamic(field: DenseDecoderField) -> Any:
            slots = int(field.dynamic_tokens.shape[1]) - 1
            axis = jnp.pad(field.dynamic_tokens[:, :slots], ((0, 0), (0, max_slots - slots), (0, 0)))
            return jnp.concatenate((axis, field.dynamic_tokens[:, slots : slots + 1]), axis=1)

        def padded_mask(field: DenseDecoderField) -> Any:
            slots = int(field.dynamic_tokens.shape[1]) - 1
            mask = field.dynamic_mask[:, 0, :]
            axis = jnp.pad(mask[:, :slots], ((0, 0), (0, max_slots - slots)))
            return jnp.concatenate((axis, mask[:, slots : slots + 1]), axis=1)[:, None, :]

        axis_parts = []
        local_parts = []
        axis_offset = local_offset = 0
        for field in group:
            slots = int(field.dynamic_tokens.shape[1]) - 1
            axis_parts.append(jnp.pad(
                field.dynamic_projection_indices[0] + axis_offset,
                ((0, 0), (0, max_slots - slots)),
            ))
            local_parts.append(field.dynamic_projection_indices[1] + local_offset)
            axis_offset += int(field.projections[0].axis_key.shape[0])
            local_offset += int(field.projections[0].local_key.shape[0])

        projections = []
        for block_index, common in enumerate(expert_projections[expert]):
            projections.append(_DecoderBlockProjection(
                family_key=common.family_key,
                family_value=common.family_value,
                process_key=common.process_key,
                process_value=common.process_value,
                carbon_key=None,
                carbon_value=None,
                axis_key=jnp.concatenate(tuple(field.projections[block_index].axis_key for field in group)),
                axis_value=jnp.concatenate(tuple(field.projections[block_index].axis_value for field in group)),
                local_key=jnp.concatenate(tuple(field.projections[block_index].local_key for field in group)),
                local_value=jnp.concatenate(tuple(field.projections[block_index].local_value for field in group)),
            ))
        decoded = _decode_mapped_rows(
            shared,
            query=jnp.concatenate(tuple(field.query for field in group)),
            dynamic=jnp.concatenate(tuple(padded_dynamic(field) for field in group)),
            dynamic_mask=jnp.concatenate(tuple(padded_mask(field) for field in group)),
            family_mask=jnp.concatenate(tuple(field.family_mask for field in group)),
            process_mask=jnp.concatenate(tuple(field.process_mask for field in group)),
            active=jnp.concatenate(tuple(field.output_active for field in group)),
            axis_indices=jnp.concatenate(tuple(axis_parts)),
            local_indices=jnp.concatenate(tuple(local_parts)),
            projections=tuple(projections),
            config=config,
            chunk_size=chunk_size,
        )
        offset = 0
        for position, size in zip(positions, sizes, strict=True):
            outputs[position] = decoded[offset : offset + size]
            offset += size
    return tuple(outputs)
