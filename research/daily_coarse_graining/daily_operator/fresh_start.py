"""Checkpoint-independent construction of the consolidated daily Attention model."""
from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np

from .strong_baseline_architecture import NeuralPrecisionPolicy
from .target_oriented_attention import initialize_target_oriented_attention_parameters
from .target_oriented_coupled_gpp_carbon_feedback import initialize_coupled_gpp_carbon_feedback_parameters
from .target_oriented_coupled_gpp_conditioned_attention import initialize_coupled_gpp_conditioned_attention_parameters
from .target_oriented_distribution_aware_maintenance import initialize_distribution_aware_maintenance_parameters
from .target_oriented_process_head import (
    build_target_oriented_coupled_gpp_carbon_feedback_head,
    build_target_oriented_coupled_gpp_conditioned_attention_head,
    build_target_oriented_distribution_aware_maintenance_head,
    build_target_oriented_structured_gpp_head,
)
from .target_oriented_structured_gpp import initialize_structured_gpp_parameters


def build_fresh_head(
    parent_head,
    retained_tail,
    *,
    threshold=16.0,
    apply_gsmean_applicability=False,
    neural_precision: NeuralPrecisionPolicy = NeuralPrecisionPolicy(),
):
    """Construct the existing topology with exact static owners and DOC conditioning."""
    parent_head = replace(
        parent_head,
        config=replace(parent_head.config, neural_precision=neural_precision),
    )
    head = build_target_oriented_distribution_aware_maintenance_head(parent_head, retained_tail)
    head = build_target_oriented_structured_gpp_head(head)
    head = build_target_oriented_coupled_gpp_carbon_feedback_head(head)
    head = build_target_oriented_coupled_gpp_conditioned_attention_head(head)
    doc = next(v for v in head.contract.state_leaves if v.key == 'slowproc_stomate_previous_step_state.DOC')
    mask = np.zeros_like(head.input_normalization.continuous_mean, dtype=bool)
    mask[doc.start:doc.stop] = True
    normalization = head.input_normalization._replace(
        continuous_tail_threshold=threshold, continuous_tail_mask=mask)
    return replace(
        head,
        input_normalization=normalization,
        warm_snow_free_boundary=True,
        apply_gsmean_applicability=apply_gsmean_applicability,
    )


def initialize_fresh_parameters(head, inputs, *, seed):
    """Generate every neural weight from one seed; no inherited parameter argument."""
    target_key, maintenance_key, gpp_key, feedback_key = jax.random.split(jax.random.PRNGKey(seed), 4)
    parameters = initialize_target_oriented_attention_parameters(
        target_key, schema=head.learned_schema, layout=head.layout, inputs=inputs,
        architecture_contract=head.architecture_contract, target_contract=head.target_contract,
        baseline_config=head.config, config=head.target_config)
    parameters = initialize_distribution_aware_maintenance_parameters(
        parameters, maintenance_key, schema=head.learned_schema, target_contract=head.target_contract,
        distribution_contract=head.distribution_contract, attention_config=head.target_config,
        config=head.distribution_config)
    parameters = initialize_structured_gpp_parameters(
        parameters, gpp_key, schema=head.learned_schema, target_contract=head.target_contract,
        structured_contract=head.structured_contract, attention_config=head.target_config,
        config=head.structured_config)
    parameters = initialize_coupled_gpp_carbon_feedback_parameters(
        parameters, feedback_key, state_layout=head.carbon_state_layout,
        coupled_contract=head.coupled_contract, structured_contract=head.structured_contract,
        structured_config=head.structured_config, config=head.coupled_config)
    return initialize_coupled_gpp_conditioned_attention_parameters(
        parameters, conditioning_contract=head.conditioning_contract,
        structured_config=head.structured_config, config=head.conditioning_config)
