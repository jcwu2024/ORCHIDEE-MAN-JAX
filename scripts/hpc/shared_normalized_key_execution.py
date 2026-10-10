"""Opt-in FP32 normalized-key reuse; no process or production-default change.

Source: shared_transformer.py::_decoder_projection_cache (376-423),
_field_dynamic_projection_cache (426-448), _attention_from_projected_segments
(451-534), _run_coordinate_blocks_reusing_projections (537-607).
Only L2-normalization placement changes. Native JAX AD remains authoritative.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys

import jax
import jax.numpy as jnp
import numpy as np
from research.daily_coarse_graining.daily_operator import shared_transformer as st


def normalize_key(value, config):
    if value is None:
        return None
    heads = config.resolved_decoder_attention_heads
    shaped = value.reshape(value.shape[:-1] + (heads, value.shape[-1] // heads))
    return st._stable_head_l2_normalize(shaped, config.normalization_epsilon).reshape(value.shape)


def normalize_projections(projections, config, names=("family_key", "process_key", "carbon_key", "axis_key", "local_key")):
    return tuple(p._replace(**{name: normalize_key(getattr(p, name), config) for name in names})
                 for p in projections)


def attention(parameters, query, dynamic_tokens, *, dynamic_key, dynamic_value,
              family_key, family_value, family_mask, dynamic_mask, process_key,
              process_value, process_mask, carbon_key, carbon_value, carbon_mask,
              heads, normalization_epsilon):
    """Original segmented attention consuming keys already normalized per head."""
    projected_query = st._linear(parameters["query"], query)
    if dynamic_key is None or dynamic_value is None:
        raise ValueError("normalized-key adapter requires gathered dynamic projections")
    width = int(projected_query.shape[-1])
    head_width = width // heads

    def per_head(value):
        return value.reshape(value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)

    head_query = st._stable_head_l2_normalize(per_head(projected_query), normalization_epsilon)
    temperature = jnp.sqrt(jnp.asarray(head_width, dtype=jnp.float32))

    def scores(key, *, shared):
        equation = "...hqd,hkd->...hqk" if shared else "...hqd,...hkd->...hqk"
        return jnp.clip(jnp.einsum(equation, head_query, per_head(key)), -1.0, 1.0) * temperature

    logits = [scores(family_key, shared=True), scores(dynamic_key, shared=False), scores(process_key, shared=True)]
    masks = [family_mask, dynamic_mask, process_mask]
    if carbon_key is not None and carbon_mask is not None:
        logits.append(scores(carbon_key, shared=True))
        masks.append(carbon_mask)
    combined_logits = jnp.concatenate(tuple(logits), axis=-1)
    combined_mask = jnp.concatenate(tuple(masks), axis=-1)
    head_mask = jnp.expand_dims(combined_mask, axis=-3)
    row_valid = jnp.expand_dims(jnp.any(combined_mask, axis=-1, keepdims=True), axis=-3)
    combined_logits = jnp.where(head_mask, combined_logits, -jnp.inf)
    weights = jax.nn.softmax(jnp.where(row_valid, combined_logits, 0.0), axis=-1)
    weights = jnp.where(head_mask, weights, 0.0)
    offsets = np.cumsum((0, family_key.shape[-2], dynamic_key.shape[-2], process_key.shape[-2]))
    result = jnp.einsum("...hqk,hkd->...hqd", weights[..., offsets[0]:offsets[1]], per_head(family_value))
    result += jnp.einsum("...hqk,...hkd->...hqd", weights[..., offsets[1]:offsets[2]], per_head(dynamic_value))
    result += jnp.einsum("...hqk,hkd->...hqd", weights[..., offsets[2]:offsets[3]], per_head(process_value))
    if carbon_key is not None and carbon_mask is not None:
        result += jnp.einsum("...hqk,hkd->...hqd", weights[..., offsets[3]:], per_head(carbon_value))
    result = result.swapaxes(-3, -2).reshape(result.shape[:-3] + (result.shape[-2], width))
    return st._linear(parameters["output"], result)


def run_blocks(query, dynamic_tokens, *, dynamic_mask, family_mask, process_mask,
               carbon_mask, blocks, projections, config, dynamic_projection_indices=None):
    if dynamic_projection_indices is None:
        raise ValueError("normalized-key adapter requires gather_dynamic_reuse")
    axis_indices, local_indices, active = dynamic_projection_indices
    for block, projected in zip(blocks, projections, strict=True):
        key = jnp.concatenate((projected.axis_key[axis_indices], projected.local_key[local_indices, None, :]), axis=1)
        value = jnp.concatenate((projected.axis_value[axis_indices], projected.local_value[local_indices, None, :]), axis=1)
        # Inactive projected zero-token keys are the trainable bias, not zero.
        bias = normalize_key(block["attention"]["key"]["bias"], config)
        key = jnp.where(active[:, None, None], key, bias)
        value = jnp.where(active[:, None, None], value, block["attention"]["value"]["bias"])
        message = attention(block["attention"], query, dynamic_tokens, dynamic_key=key, dynamic_value=value,
                            family_key=projected.family_key, family_value=projected.family_value, family_mask=family_mask,
                            dynamic_mask=dynamic_mask, process_key=projected.process_key, process_value=projected.process_value,
                            process_mask=process_mask, carbon_key=projected.carbon_key, carbon_value=projected.carbon_value,
                            carbon_mask=carbon_mask, heads=config.resolved_decoder_attention_heads,
                            normalization_epsilon=config.normalization_epsilon)
        query = st._layer_norm(query + message, block["attention_norm_scale"], block["attention_norm_bias"])
        query = st._layer_norm(query + st._ffn(block["ffn"], query), block["ffn_norm_scale"], block["ffn_norm_bias"])
    return query


@contextmanager
def patches(config):
    """Patch loaded original/staged aliases during lowering only; restore all."""
    original_cache = st._decoder_projection_cache
    original_field = st._field_dynamic_projection_cache
    def cache(*args, **kwargs):
        return tuple(normalize_projections(expert, config, ("family_key", "process_key", "carbon_key"))
                     for expert in original_cache(*args, **kwargs))
    def field(*args, **kwargs):
        return normalize_projections(original_field(*args, **kwargs), config, ("axis_key", "local_key"))
    replacements = {"_decoder_projection_cache": cache, "_field_dynamic_projection_cache": field,
                    "_run_coordinate_blocks_reusing_projections": run_blocks}
    saved = []
    prefix = "research.daily_coarse_graining.daily_operator."
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(prefix) or module is None:
            continue
        for attribute, function in replacements.items():
            if hasattr(module, attribute):
                saved.append((module, attribute, getattr(module, attribute)))
                setattr(module, attribute, function)
    try:
        yield
    finally:
        for module, attribute, function in reversed(saved):
            setattr(module, attribute, function)
