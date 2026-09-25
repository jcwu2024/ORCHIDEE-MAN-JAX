"""Diagnostic shared-source backbone execution for the canonical context.

The production assembler remains unchanged. This path reuses one source K/V
projection per block while preserving expert-specific axis tokens and masks.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from .neural_architecture import NeuralInputNormalization
from .shared_backbone_projection_reuse import run_backbone_with_shared_source_projection
from .shared_transformer import SharedTransformerConfig, SharedTransformerContext, _tokenizer_config
from .strong_baseline_architecture import StrongBaselineLayout, _linear, assemble_strong_baseline_context
from .types import AssembledDailyOperatorInput


def assemble_shared_transformer_context_with_source_reuse(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    *,
    normalization: NeuralInputNormalization,
    layout: StrongBaselineLayout,
    config: SharedTransformerConfig = SharedTransformerConfig(),
) -> SharedTransformerContext:
    """Build the canonical context with common source K/V projected once."""

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
    bank_mask = jnp.concatenate((source_mask, jnp.asarray(base.expert_axis_mask, dtype=jnp.bool_)), axis=1)
    query = (shared["process_query"] + shared["process_embedding"])[:, None, :]
    query = run_backbone_with_shared_source_projection(
        query,
        source_tokens,
        axis_tokens,
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


__all__ = ["assemble_shared_transformer_context_with_source_reuse"]
