"""Selected research decoder: shared wide expert latents, private narrow readers.

Neural representation only. The architecture contract is section 3 of
docs/research/daily_coarse_graining/shared_structural_speed_routes_20261004.md.
Physical process provenance and typed transforms retain their existing owners.
All caches below are live JAX values local to one sample/day invocation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .shared_transformer import (
    GPP_FIELD,
    SharedTransformerConfig,
    _carbon_tokens,
    _decoder_dynamic_plan,
    _decoder_dynamic_tokens,
    _DecoderBlockProjection,
    _field_active_and_pft_owner,
    _field_coordinate_inputs,
    _normalized_attention,
    _project_decoder_context,
    _project_key_value,
    _run_coordinate_blocks_reusing_projections,
    _source_family_bank,
    _static_field_coordinate_indices,
)
from .strong_baseline_architecture import (
    _attention_parameters,
    _ffn,
    _ffn_parameters,
    _layer_norm,
    _linear,
    _linear_parameters,
    _normal,
    _residual_block_parameters,
)

LATENT_SLOTS = 32
LATENT_LAYERS = 2
READER_WIDTH = 64
READER_HEADS = 8
READER_LAYERS = 2
READER_FFN_WIDTH = 256
EXPERT_CHUNK_SIZE = 128


def shared_latent_reader_config(base: SharedTransformerConfig | None = None) -> SharedTransformerConfig:
    """Explicit canonical prototype selection; current model defaults stay intact."""
    return replace(
        SharedTransformerConfig() if base is None else base,
        width=768, attention_heads=8, ffn_width=3072, backbone_blocks=8,
        decoder_width=384, decoder_ffn_width=1536, decoder_attention_heads=8,
        coordinate_blocks=2, readout_width=256, decoder_architecture="latent_reader",
        latent_reader_execution="field",
    )


def _latent_block_parameters(key: Any, width: int) -> Mapping[str, Any]:
    cross, self_key, ffn = jax.random.split(key, 3)
    result = {
        "cross_attention": _attention_parameters(cross, width),
        "self_attention": _attention_parameters(self_key, width),
        "ffn": _ffn_parameters(ffn, width, 4 * width),
    }
    for name in ("cross_attention", "self_attention", "ffn"):
        result[name + "_norm_scale"] = jnp.ones((width,), dtype=jnp.float32)
        result[name + "_norm_bias"] = jnp.zeros((width,), dtype=jnp.float32)
    return result


def initialize_latent_reader_parameters(keys: Any, *, config: SharedTransformerConfig) -> Mapping[str, Any]:
    """Add exactly the selected latent/reader parameter groups, freshly in FP32."""
    width = config.resolved_decoder_width
    return {
        "latent_slots": _normal(next(keys), (LATENT_SLOTS, width), width),
        "latent_blocks": tuple(_latent_block_parameters(next(keys), width) for _ in range(LATENT_LAYERS)),
        "reader_projection": _linear_parameters(next(keys), width, READER_WIDTH),
        "reader_blocks": tuple(
            _residual_block_parameters(next(keys), READER_WIDTH, READER_FFN_WIDTH)
            for _ in range(READER_LAYERS)
        ),
        "readout_first": _linear_parameters(next(keys), READER_WIDTH, config.readout_width),
        "readout_second": _linear_parameters(next(keys), config.readout_width, 1),
    }


def build_expert_latents(parameters, context, family_tokens, family_mask, *, config):
    """Run isolated expert activations using shared slots and layer parameters.

The only direct cross bank is legal pooled families plus this expert's process.
Axis, local state and raw carbon never enter this new bank directly.
"""
    shared = parameters["shared_transformer"]
    query = shared["latent_slots"][None, :, :] + context.process_tokens[:, None, :]
    bank = jnp.concatenate((family_tokens, context.process_tokens[:, None, :]), axis=1)
    valid = jnp.concatenate((family_mask, jnp.ones((bank.shape[0], 1), dtype=bool)), axis=1)
    cross_mask = jnp.broadcast_to(valid[:, None, :], (bank.shape[0], LATENT_SLOTS, bank.shape[1]))
    self_mask = jnp.ones((bank.shape[0], LATENT_SLOTS, LATENT_SLOTS), dtype=bool)
    for block in shared["latent_blocks"]:
        for name, value, mask in (
            ("cross_attention", bank, cross_mask),
            ("self_attention", None, self_mask),
        ):
            message = _normalized_attention(
                block[name], query, query if value is None else value,
                heads=config.resolved_decoder_attention_heads, mask=mask,
                normalization_epsilon=config.normalization_epsilon,
            )
            query = _layer_norm(query + message, block[name + "_norm_scale"], block[name + "_norm_bias"])
        query = _layer_norm(query + _ffn(block["ffn"], query), block["ffn_norm_scale"], block["ffn_norm_bias"])
    return query


def _reader_config(config):
    return replace(config, decoder_width=READER_WIDTH, decoder_ffn_width=READER_FFN_WIDTH,
                   decoder_attention_heads=READER_HEADS)


def reader_projection_cache(parameters, latent_tokens, process_tokens, carbon_tokens=None):
    """Narrow shared banks once, then K/V once per expert and reader layer."""
    shared = parameters["shared_transformer"]
    affine = shared["reader_projection"]
    latent = _linear(affine, latent_tokens)
    process = _linear(affine, process_tokens)
    carbon = None if carbon_tokens is None else _linear(affine, carbon_tokens)
    carbon_by_block = tuple(
        (None, None) if carbon is None else _project_key_value(block["attention"], carbon)
        for block in shared["reader_blocks"]
    )
    result = []
    for expert in range(process_tokens.shape[0]):
        layers = []
        for block, (carbon_key, carbon_value) in zip(shared["reader_blocks"], carbon_by_block, strict=True):
            key, value = _project_key_value(block["attention"], latent[expert])
            process_key, process_value = _project_key_value(block["attention"], process[expert, None, :])
            layers.append(_DecoderBlockProjection(
                key, value, process_key, process_value, carbon_key, carbon_value,
                None, None, None, None,
            ))
        result.append(tuple(layers))
    return tuple(result)


def reader_attention(parameters, query, *, latent_key, latent_value, latent_mask,
                     axis_local_key, axis_local_value, axis_local_mask,
                     process_key, process_value, process_mask,
                     carbon_key=None, carbon_value=None, carbon_mask=None,
                     normalization_epsilon=1.0e-6):
    """Auditable joint softmax over all legal shared and private segments.

Masks have shape ``query.shape[:-1] + (segment_keys,)``. For a singleton
query axis, ``query.shape[:-2] + (segment_keys,)`` is also accepted. Making
that axis explicit prevents a row mask from broadcasting as another query.
"""
    from .shared_transformer import _attention_from_projected_segments

    if query.shape[-1] != READER_WIDTH:
        raise ValueError("reader query width differs from the selected architecture")

    def explicit_mask(mask, key):
        mask = jnp.asarray(mask, jnp.bool_)
        full_shape = query.shape[:-1] + (key.shape[-2],)
        short_shape = query.shape[:-2] + (key.shape[-2],)
        if mask.shape == short_shape and query.shape[-2] == 1:
            mask = jnp.expand_dims(mask, axis=-2)
        if mask.shape != full_shape:
            raise ValueError(f"reader segment mask must have shape {full_shape}; got {mask.shape}")
        return mask

    latent_mask = explicit_mask(latent_mask, latent_key)
    axis_local_mask = explicit_mask(axis_local_mask, axis_local_key)
    process_mask = explicit_mask(process_mask, process_key)
    if carbon_key is not None and carbon_mask is not None:
        carbon_mask = explicit_mask(carbon_mask, carbon_key)
    return _attention_from_projected_segments(
        parameters, query, jnp.zeros_like(axis_local_key),
        dynamic_key=axis_local_key, dynamic_value=axis_local_value,
        family_key=latent_key, family_value=latent_value, family_mask=latent_mask,
        dynamic_mask=axis_local_mask,
        process_key=process_key, process_value=process_value, process_mask=process_mask,
        carbon_key=carbon_key, carbon_value=carbon_value, carbon_mask=carbon_mask,
        heads=READER_HEADS, normalization_epsilon=normalization_epsilon,
    )


def _field_reader_projections(parameters, projected, axis_tokens, local_tokens):
    shared = parameters["shared_transformer"]
    axis = _linear(shared["reader_projection"], axis_tokens)
    local = _linear(shared["reader_projection"], local_tokens)
    result = []
    for block, layer in zip(shared["reader_blocks"], projected, strict=True):
        axis_key, axis_value = _project_key_value(block["attention"], axis)
        local_key, local_value = _project_key_value(block["attention"], local)
        result.append(layer._replace(axis_key=axis_key, axis_value=axis_value,
                                     local_key=local_key, local_value=local_value))
    return tuple(result)


def _prepare_field(parameters, context, inputs, *, field, ownership, field_index,
                   layout, input_normalization, dynamic_plan, projected):
    selected = _static_field_coordinate_indices(field, ownership, layout)
    _, full_active, _ = _field_active_and_pft_owner(context, field, ownership, layout)
    features, active, owners, _axis, axis_mask, _local, local_defined = _field_coordinate_inputs(
        parameters, context, inputs, field=field, ownership=ownership,
        field_index=field_index, layout=layout, input_normalization=input_normalization,
        coordinate_indices=selected,
    )
    size = int(features.shape[0])
    if not size:
        return selected, full_active, None, None
    shared = parameters["shared_transformer"]
    features = _linear(shared["reader_projection"], features)
    axis_tokens, local_tokens = _decoder_dynamic_tokens(
        parameters, context, inputs, input_normalization=input_normalization,
        layout=layout, plan=dynamic_plan,
        axis_bank_indices=dynamic_plan.field_axis_bank_indices[field_index],
        local_bank_indices=dynamic_plan.field_local_bank_indices[field_index],
    )
    projected = _field_reader_projections(parameters, projected, axis_tokens, local_tokens)
    values = (
        features, active, owners, axis_mask, local_defined & active,
        jnp.asarray(dynamic_plan.field_axis_indices[field_index], jnp.int32),
        jnp.asarray(dynamic_plan.field_local_indices[field_index], jnp.int32),
    )
    return selected, full_active, values, projected


def _decode_reader_rows(parameters, values, projected, *, config, maximum_chunk,
                        uses_carbon, carbon_defined, carbon_state_layout):
    """Read independent coordinate rows; chunking introduces no row interaction."""
    shared = parameters["shared_transformer"]
    size = int(values[0].shape[0])
    chunk = min(maximum_chunk, size)
    padded_size = ((size + chunk - 1) // chunk) * chunk
    padding = padded_size - size
    chunks = padded_size // chunk

    def pad(value, fill=0):
        value = jnp.pad(value, ((0, padding),) + ((0, 0),) * (value.ndim - 1), constant_values=fill)
        return value.reshape((chunks, chunk) + value.shape[1:])

    values = tuple(pad(value, -1 if index == 2 else 0) for index, value in enumerate(values))
    reader_config = _reader_config(config)

    def decode_chunk(values):
        query, active, owner, axis_mask, local_mask, axis_indices, local_indices = values
        dynamic_mask = jnp.concatenate((axis_mask & active[:, None], local_mask[:, None]), axis=1)
        raw_mask = None
        if uses_carbon and carbon_defined is not None:
            carbon_owner = jnp.asarray(carbon_state_layout.pft_owner_index, jnp.int32)
            raw_mask = carbon_defined[None, :] & (
                (carbon_owner[None, :] == -1) | (carbon_owner[None, :] == owner[:, None]))
            raw_mask &= active[:, None]
        encoded = _run_coordinate_blocks_reusing_projections(
            query[:, None, :], jnp.zeros((chunk, dynamic_mask.shape[1], READER_WIDTH), jnp.float32),
            dynamic_mask=dynamic_mask[:, None, :],
            family_mask=jnp.broadcast_to(active[:, None, None], (chunk, 1, LATENT_SLOTS)),
            process_mask=active[:, None, None],
            carbon_mask=None if raw_mask is None else raw_mask[:, None, :],
            blocks=shared["reader_blocks"], projections=projected, config=reader_config,
            dynamic_projection_indices=(axis_indices, local_indices, active),
        )[:, 0, :]
        raw = _linear(shared["readout_second"],
                      jax.nn.gelu(_linear(shared["readout_first"], encoded)))[:, 0]
        return jnp.where(active, raw, 0.0)

    return jax.lax.map(decode_chunk, values).reshape(-1)[:size]


def _restore_field(field, selected, full_active, decoded):
    raw = jnp.zeros((int(np.prod(field.shape)),), jnp.float32).at[jnp.asarray(selected)].set(decoded)
    return raw.reshape(field.shape), full_active.reshape(field.shape)


def _decode_field(parameters, context, inputs, *, field, ownership, field_index,
                  layout, input_normalization, config, dynamic_plan, projected,
                  carbon_defined, carbon_state_layout):
    selected, full_active, values, projected = _prepare_field(
        parameters, context, inputs, field=field, ownership=ownership, field_index=field_index,
        layout=layout, input_normalization=input_normalization, dynamic_plan=dynamic_plan,
        projected=projected,
    )
    if values is None:
        return jnp.zeros(field.shape, jnp.float32), full_active.reshape(field.shape)
    decoded = _decode_reader_rows(
        parameters, values, projected, config=config, maximum_chunk=config.output_chunk_size,
        uses_carbon=field.name == GPP_FIELD, carbon_defined=carbon_defined,
        carbon_state_layout=carbon_state_layout,
    )
    return _restore_field(field, selected, full_active, decoded)


def _decode_expert_group(parameters, entries, *, config, uses_carbon,
                         carbon_defined, carbon_state_layout):
    """Pack fields of one expert, preserving field-private bank identities.

Only the row scheduling changes from the field reference. Each row still sees
its own axis/local K/V and one shared expert latent/process bank. Non-GPP rows
use the existing expert128 schedule; GPP keeps its separate configured chunk.
"""
    max_axes = max(entry[3][3].shape[1] for entry in entries)
    value_parts = [[] for _ in range(7)]
    axis_offset = local_offset = 0
    for _field, _selected, _active, values, projected in entries:
        query, active, owners, axis_mask, local_mask, axis_indices, local_indices = values
        missing = max_axes - axis_mask.shape[1]
        axis_mask = jnp.pad(axis_mask, ((0, 0), (0, missing)))
        # Masked extra slots gather a legal sentinel and never enter softmax.
        axis_indices = jnp.pad(axis_indices + axis_offset, ((0, 0), (0, missing)))
        current = (query, active, owners, axis_mask, local_mask,
                   axis_indices, local_indices + local_offset)
        for parts, value in zip(value_parts, current, strict=True):
            parts.append(value)
        axis_offset += projected[0].axis_key.shape[0]
        local_offset += projected[0].local_key.shape[0]
    values = tuple(jnp.concatenate(parts, axis=0) for parts in value_parts)
    projected_group = []
    for layer in range(READER_LAYERS):
        base = entries[0][4][layer]
        axis_keys = [entry[4][layer].axis_key for entry in entries]
        axis_values = [entry[4][layer].axis_value for entry in entries]
        # A zero sentinel also covers groups with no physical axis tokens.
        axis_keys.append(jnp.zeros((1, READER_WIDTH), jnp.float32))
        axis_values.append(jnp.zeros((1, READER_WIDTH), jnp.float32))
        projected_group.append(base._replace(
            axis_key=jnp.concatenate(axis_keys), axis_value=jnp.concatenate(axis_values),
            local_key=jnp.concatenate([entry[4][layer].local_key for entry in entries]),
            local_value=jnp.concatenate([entry[4][layer].local_value for entry in entries]),
        ))
    decoded = _decode_reader_rows(
        parameters, values, tuple(projected_group), config=config,
        maximum_chunk=config.output_chunk_size if uses_carbon else EXPERT_CHUNK_SIZE,
        uses_carbon=uses_carbon, carbon_defined=carbon_defined,
        carbon_state_layout=carbon_state_layout,
    )
    result = {}
    offset = 0
    for field, selected, full_active, values, _projected in entries:
        count = values[0].shape[0]
        result[field.name] = _restore_field(field, selected, full_active, decoded[offset:offset + count])
        offset += count
    return result


def shared_latent_reader_raw_outputs(parameters, context, inputs, *, schema, layout,
                                    input_normalization, config, carbon_state_layout=None,
                                    field_reference=False):
    """Full-shaped raw outputs and original masks at the public neural boundary."""
    config.validate()
    if config.decoder_architecture != "latent_reader":
        raise ValueError("latent reader requires its explicit architecture selection")
    if tuple(field.name for field in schema.fields) != tuple(field.name for field in layout.ledger):
        raise ValueError("shared Transformer schema and layout drift")
    families, family_mask = _source_family_bank(context, layout)
    context, families = _project_decoder_context(parameters, context, families, family_mask)
    latent = build_expert_latents(parameters, context, families, family_mask, config=config)
    carbon = carbon_defined = None
    if carbon_state_layout is not None:
        carbon, carbon_defined = _carbon_tokens(parameters, inputs, input_normalization, carbon_state_layout)
    projections = reader_projection_cache(parameters, latent, context.process_tokens, carbon)
    plan = _decoder_dynamic_plan(schema=schema, layout=layout)
    outputs, masks = {}, {}
    groups = {}
    for index, (field, ownership) in enumerate(zip(schema.fields, layout.ledger, strict=True)):
        expert = int(layout.field_expert_index[index])
        if field_reference:
            outputs[field.name], masks[field.name] = _decode_field(
                parameters, context, inputs, field=field, ownership=ownership, field_index=index,
                layout=layout, input_normalization=input_normalization, config=config,
                dynamic_plan=plan, projected=projections[expert],
                carbon_defined=carbon_defined, carbon_state_layout=carbon_state_layout,
            )
            continue
        selected, full_active, values, projected = _prepare_field(
            parameters, context, inputs, field=field, ownership=ownership, field_index=index,
            layout=layout, input_normalization=input_normalization, dynamic_plan=plan,
            projected=projections[expert],
        )
        if values is None:
            outputs[field.name] = jnp.zeros(field.shape, jnp.float32)
            masks[field.name] = full_active.reshape(field.shape)
        else:
            groups.setdefault((expert, field.name == GPP_FIELD), []).append(
                (field, selected, full_active, values, projected))
    for (_expert, uses_carbon), entries in groups.items():
        result = _decode_expert_group(
            parameters, entries, config=config, uses_carbon=uses_carbon,
            carbon_defined=carbon_defined, carbon_state_layout=carbon_state_layout,
        )
        for name, (raw, active) in result.items():
            outputs[name], masks[name] = raw, active
    return outputs, masks
