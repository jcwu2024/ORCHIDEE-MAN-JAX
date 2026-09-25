"""Diagnostic backbone attention with common source K/V projected once.

This helper does not replace the production shared Transformer path. Source
tokens are identical across experts; expert axis tokens and attention masks
remain separate. Both parameter and input gradients use native JAX AD.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .shared_transformer import SharedTransformerConfig, _stable_head_l2_normalize
from .strong_baseline_architecture import _ffn, _layer_norm, _linear


def run_backbone_with_shared_source_projection(
    query: Any,
    source_tokens: Any,
    expert_axis_tokens: Any,
    mask: Any,
    blocks: Any,
    config: SharedTransformerConfig,
) -> Any:
    """Match ``_run_blocks`` on ``[source, expert-axis]`` banks.

    Shapes are ``query [..., E, 1, D]``, ``source [..., S, D]``,
    ``axis [..., E, A, D]`` and ``mask [..., E, 1, S+A]``. The source
    projection is shared only within one differentiable invocation.
    """

    width = int(query.shape[-1])
    heads = int(config.attention_heads)
    if width != config.width or width % heads:
        raise ValueError("query width must match the backbone width and heads")
    if source_tokens.shape[:-2] != expert_axis_tokens.shape[:-3]:
        raise ValueError("source and expert-axis batch dimensions differ")
    if query.shape[:-2] != expert_axis_tokens.shape[:-2]:
        raise ValueError("query and expert-axis dimensions differ")
    if source_tokens.shape[-1] != width or expert_axis_tokens.shape[-1] != width:
        raise ValueError("bank token width differs from query width")
    if mask.shape != query.shape[:-1] + (source_tokens.shape[-2] + expert_axis_tokens.shape[-2],):
        raise ValueError("attention mask shape differs from source/axis bank")

    head_width = width // heads
    temperature = jnp.sqrt(jnp.asarray(head_width, dtype=jnp.float32))

    def per_head(value: Any) -> Any:
        return value.reshape(value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)

    for block in blocks:
        attention = block["attention"]
        projected_query = _linear(attention["query"], query)
        source_key = _linear(attention["key"], source_tokens)
        source_value = _linear(attention["value"], source_tokens)
        axis_key = _linear(attention["key"], expert_axis_tokens)
        axis_value = _linear(attention["value"], expert_axis_tokens)
        source_shape = axis_key.shape[:-2] + source_key.shape[-2:]
        projected_key = jnp.concatenate(
            (jnp.broadcast_to(source_key[..., None, :, :], source_shape), axis_key), axis=-2
        )
        projected_value = jnp.concatenate(
            (jnp.broadcast_to(source_value[..., None, :, :], source_shape), axis_value), axis=-2
        )

        head_query = _stable_head_l2_normalize(per_head(projected_query), config.normalization_epsilon)
        head_key = _stable_head_l2_normalize(per_head(projected_key), config.normalization_epsilon)
        head_value = per_head(projected_value)
        logits = jnp.clip(jnp.einsum("...hqd,...hkd->...hqk", head_query, head_key), -1.0, 1.0)
        logits = logits * temperature
        head_mask = jnp.expand_dims(mask, axis=-3)
        row_valid = jnp.expand_dims(jnp.any(mask, axis=-1, keepdims=True), axis=-3)
        logits = jnp.where(head_mask, logits, -jnp.inf)
        weights = jax.nn.softmax(jnp.where(row_valid, logits, 0.0), axis=-1)
        weights = jnp.where(head_mask, weights, 0.0)
        result = jnp.einsum("...hqk,...hkd->...hqd", weights, head_value)
        result = result.swapaxes(-3, -2).reshape(result.shape[:-3] + (result.shape[-2], width))
        message = _linear(attention["output"], result)
        query = _layer_norm(query + message, block["attention_norm_scale"], block["attention_norm_bias"])
        query = _layer_norm(query + _ffn(block["ffn"], query), block["ffn_norm_scale"], block["ffn_norm_bias"])
    return query


__all__ = ["run_backbone_with_shared_source_projection"]
