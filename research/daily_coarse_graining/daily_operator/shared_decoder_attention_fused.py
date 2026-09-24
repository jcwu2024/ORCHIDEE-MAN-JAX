"""Diagnostic single-contraction variant of the shared decoder attention.

This is not connected to the production model. Shared K/V segments are
broadcast only within a coordinate chunk; the GPP carbon segment remains an
explicit optional input so its larger bank can be assessed separately.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from .shared_transformer import _project_key_value, _stable_head_l2_normalize
from .strong_baseline_architecture import _linear


def fused_attention_from_projected_segments(
    parameters: Mapping[str, Any],
    query: Any,
    dynamic_tokens: Any,
    *,
    dynamic_key: Any | None,
    dynamic_value: Any | None,
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
    """Use one score and one value contraction after field-local K/V projection."""

    projected_query = _linear(parameters["query"], query)
    if dynamic_key is None or dynamic_value is None:
        dynamic_key, dynamic_value = _project_key_value(parameters, dynamic_tokens)
    width = int(projected_query.shape[-1])
    head_width = width // int(heads)

    def per_head(value: Any) -> Any:
        shaped = value.reshape(value.shape[:-1] + (heads, head_width))
        return shaped.swapaxes(-3, -2)

    head_query = _stable_head_l2_normalize(per_head(projected_query), normalization_epsilon)
    head_dynamic_key = _stable_head_l2_normalize(per_head(dynamic_key), normalization_epsilon)
    head_family_key = _stable_head_l2_normalize(per_head(family_key), normalization_epsilon)
    head_process_key = _stable_head_l2_normalize(per_head(process_key), normalization_epsilon)
    dynamic_shape = head_dynamic_key.shape[:-2]

    def shared(value: Any) -> Any:
        return jnp.broadcast_to(value, dynamic_shape + value.shape[-2:])

    keys = [shared(head_family_key), head_dynamic_key, shared(head_process_key)]
    values = [shared(per_head(family_value)), per_head(dynamic_value), shared(per_head(process_value))]
    masks = [family_mask, dynamic_mask, process_mask]
    if carbon_key is not None and carbon_mask is not None:
        keys.append(shared(_stable_head_l2_normalize(per_head(carbon_key), normalization_epsilon)))
        values.append(shared(per_head(carbon_value)))
        masks.append(carbon_mask)

    combined_key = jnp.concatenate(tuple(keys), axis=-2)
    combined_value = jnp.concatenate(tuple(values), axis=-2)
    combined_mask = jnp.concatenate(tuple(masks), axis=-1)
    logits = jnp.clip(jnp.einsum("...hqd,...hkd->...hqk", head_query, combined_key), -1.0, 1.0)
    logits *= jnp.sqrt(jnp.asarray(head_width, dtype=jnp.float32))
    head_mask = jnp.expand_dims(combined_mask, axis=-3)
    row_valid = jnp.expand_dims(jnp.any(combined_mask, axis=-1, keepdims=True), axis=-3)
    logits = jnp.where(head_mask, logits, -jnp.inf)
    weights = jax.nn.softmax(jnp.where(row_valid, logits, 0.0), axis=-1)
    weights = jnp.where(head_mask, weights, 0.0)
    result = jnp.einsum("...hqk,...hkd->...hqd", weights, combined_value)
    result = result.swapaxes(-3, -2).reshape(result.shape[:-3] + (result.shape[-2], width))
    return _linear(parameters["output"], result)
