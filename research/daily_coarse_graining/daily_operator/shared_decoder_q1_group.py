"""Diagnostic same-expert, exact-shape Q1 decoder grouping.

Only the 22-row decoder scheduling differs from the per-field reference.
Field-local dynamic projections and masks are supplied separately, and this
module is not connected to the production shared Transformer.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from .shared_decoder_dense import DenseDecoderField
from .shared_transformer import _DecoderBlockProjection, _run_coordinate_blocks_reusing_projections
from .strong_baseline_architecture import _linear


def grouped_decode_q1_fields(
    shared_parameters: Mapping[str, Any],
    fields: tuple[DenseDecoderField, ...],
    config: Any,
) -> tuple[Any, ...]:
    """Decode one same-expert Q1 group while preserving every field output."""

    if not fields:
        raise ValueError("at least one field is required")
    blocks = shared_parameters["coordinate_blocks"]
    width = config.resolved_decoder_width
    family_count = int(fields[0].family_mask.shape[-1])
    for field in fields:
        if field.query.shape != (1, 1, width):
            raise ValueError("each field must contain exactly one query")
        if field.dynamic_tokens.shape != (1, 1, width):
            raise ValueError("Q1 group requires one local token and no axis tokens")
        if field.family_mask.shape != (1, 1, family_count):
            raise ValueError("fields must share one family-bank shape")
        if field.dynamic_mask.shape != (1, 1, 1) or field.process_mask.shape != (1, 1, 1):
            raise ValueError("Q1 masks have incompatible shapes")
        if field.carbon_mask is not None or field.dynamic_projection_indices is None:
            raise ValueError("Q1 group requires field-local projection reuse without carbon")
        if field.output_active.shape != (1,) or len(field.projections) != len(blocks):
            raise ValueError("Q1 output or projection shape mismatch")
        axis_indices, local_indices, _active = field.dynamic_projection_indices
        if axis_indices.shape != (1, 0) or local_indices.shape != (1,):
            raise ValueError("Q1 projection indices must address only the local token")
        for projected in field.projections:
            if projected.axis_key.shape != (0, width) or projected.local_key.shape != (1, width):
                raise ValueError("field-local projected K shape mismatch")
            if projected.axis_value.shape != (0, width) or projected.local_value.shape != (1, width):
                raise ValueError("field-local projected V shape mismatch")

    query = jnp.concatenate(tuple(field.query for field in fields), axis=0)
    dynamic = jnp.concatenate(tuple(field.dynamic_tokens for field in fields), axis=0)
    active = jnp.concatenate(tuple(field.output_active for field in fields), axis=0)
    local_indices = jnp.arange(len(fields), dtype=jnp.int32)
    projections = []
    for block_index in range(len(blocks)):
        first = fields[0].projections[block_index]
        projections.append(
            _DecoderBlockProjection(
                family_key=first.family_key,
                family_value=first.family_value,
                process_key=first.process_key,
                process_value=first.process_value,
                carbon_key=None,
                carbon_value=None,
                axis_key=jnp.zeros((0, width), dtype=query.dtype),
                axis_value=jnp.zeros((0, width), dtype=query.dtype),
                local_key=jnp.concatenate(tuple(field.projections[block_index].local_key for field in fields)),
                local_value=jnp.concatenate(tuple(field.projections[block_index].local_value for field in fields)),
            )
        )
    encoded = _run_coordinate_blocks_reusing_projections(
        query,
        dynamic,
        dynamic_mask=jnp.concatenate(tuple(field.dynamic_mask for field in fields), axis=0),
        family_mask=jnp.concatenate(tuple(field.family_mask for field in fields), axis=0),
        process_mask=jnp.concatenate(tuple(field.process_mask for field in fields), axis=0),
        carbon_mask=None,
        blocks=blocks,
        projections=tuple(projections),
        config=config,
        dynamic_projection_indices=(jnp.zeros((len(fields), 0), dtype=jnp.int32), local_indices, active),
    )[:, 0, :]
    raw = _linear(shared_parameters["readout_second"], jax.nn.gelu(_linear(shared_parameters["readout_first"], encoded)))[:, 0]
    raw = jnp.where(active, raw, 0.0)
    return tuple(raw[index : index + 1] for index in range(len(fields)))
