"""Shared Transformer at the canonical daily process/retained-physics boundary.

No new physical process is introduced here. Maintenance materialization reuses
stomate_resp.f90::maint_respiration, lines 234-375, through its audited JAX
adapter; inventories and the daily tail retain their existing source owners.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .mechanistic_process_head import MechanisticStrongBaselineProcessHead
from .output_applicability import apply_gsmean_output_applicability
from .output_transform import normalized_output_to_physical
from .parameter_routing import DEFAULT_PFT_PARAMETER_ROUTING, VCMAX25_FACTOR_SCALE
from .shared_transformer import (
    SharedTransformerConfig,
    assemble_shared_transformer_context,
    initialize_shared_transformer_parameters,
    shared_transformer_raw_outputs,
)
from .strong_baseline_architecture import NeuralPrecisionPolicy
from .target_oriented_contract import load_target_oriented_contract
from .target_oriented_coupled_gpp_carbon_feedback import (
    build_carbon_feedback_state_layout,
    load_coupled_gpp_carbon_feedback_contract,
)
from .target_oriented_source_shaped_maintenance import (
    MAINTENANCE_FIELD,
    compact_coeff_maint_zero_from_retained_tail,
    materialize_part_resolved_source_shaped_maintenance,
)

GPP_FIELD = "daily.fast_gpp"
OUTGOING_CENTER_FLOOR = 1.0e-8


def transform_shared_output(raw, active, *, field, normalization, inputs,
                            outgoing_center_floor=OUTGOING_CENTER_FLOOR,
                            thermal_transform="train_centered"):
    """Center outgoing fractions and thermal tendencies in frozen TRAIN units.

The floor is a numerical interior point, not a learned decay rate. Keeping
the residual logit trainable preserves the full (0,1) range and its gradient.
Thermal residuals have the TRAIN scale as their local derivative at raw=0,
while retaining the full signed-unit range. ``direct_tanh`` reproduces the
original shared admission; it must be explicit when reusing that gradient.
All other typed transformations retain their existing implementation.
"""
    if (thermal_transform == "train_centered" and field.name.startswith("thermal.")
            and field.value_representation == "direct" and field.constraint == "signed_unit"):
        center = jnp.clip(jnp.asarray(normalization.mean[field.name], jnp.float64).reshape(field.shape),
                          -1.0 + 1e-6, 1.0 - 1e-6)
        scale = jnp.asarray(normalization.scale[field.name], jnp.float64).reshape(field.shape)
        raw = jnp.asarray(raw, jnp.float64).reshape(field.shape)
        slope = jnp.maximum(1.0 - center * center, 1e-6)
        value = jnp.tanh(jnp.arctanh(center) + raw * scale / slope)
        return jnp.where(jnp.asarray(active).reshape(field.shape), value, 0.0)
    if (field.name.endswith(".outgoing_fraction")
            and field.value_representation == "direct"
            and field.constraint == "unit_interval"):
        center = jnp.clip(jnp.asarray(normalization.mean[field.name], jnp.float64).reshape(field.shape),
                          outgoing_center_floor, 1.0 - outgoing_center_floor)
        raw = jnp.asarray(raw, jnp.float64).reshape(field.shape)
        value = jax.nn.sigmoid(jnp.log(center) - jnp.log1p(-center) + raw)
        return jnp.where(jnp.asarray(active).reshape(field.shape), value, 0.0)
    return normalized_output_to_physical(
        raw, active, field=field, normalization=normalization, inputs=inputs)


def apply_shared_gpp_factor(environmental_response, inputs):
    """Preserve the explicit VCMAX factor and vegetation applicability once."""
    pft = inputs.pft
    response = jnp.asarray(environmental_response, jnp.float64)
    column = DEFAULT_PFT_PARAMETER_ROUTING.index("vcmax25")
    factor = jnp.maximum(jnp.asarray(pft.pft_parameters)[:, column], 0.0) / VCMAX25_FACTOR_SCALE
    active = (jnp.asarray(pft.active_pft_mask, dtype=bool)
              & (jnp.asarray(pft.pft_fraction) > 0.0)
              & ~(jnp.asarray(pft.pft_traits)[:, 0] > 0.5))
    return jnp.where(active, response.reshape(active.shape) * factor, 0.0).reshape(response.shape)


@dataclass(frozen=True, eq=False, kw_only=True)
class SharedTransformerProcessHead(MechanisticStrongBaselineProcessHead):
    """One shared neural tree; source-backed physical decoders stay explicit."""

    coeff_maint_zero: Any
    carbon_state_layout: Any
    target_contract: Any
    shared_config: SharedTransformerConfig = SharedTransformerConfig()
    outgoing_center_floor: float = OUTGOING_CENTER_FLOOR
    thermal_transform: str = "train_centered"
    apply_gsmean_applicability: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.thermal_transform not in {"direct_tanh", "train_centered"}:
            raise ValueError("unknown shared thermal output transform")
        if not 0.0 < self.outgoing_center_floor < 0.5:
            raise ValueError("outgoing fraction center floor must lie in (0,0.5)")
        if np.shape(self.coeff_maint_zero) != (len(self.layout.active_pft_indices), 12):
            raise ValueError("maintenance coefficient layout differs from the canonical PFT/part layout")
        for field in self.learned_schema.fields:
            if field.name.endswith(".outgoing_fraction"):
                center = np.asarray(self.output_normalization.mean[field.name])
                if not np.all(np.isfinite(center)) or np.any((center < 0.0) | (center > 1.0)):
                    raise ValueError(f"invalid TRAIN outgoing prior: {field.name}")

    def initialize(self, inputs, *, seed):
        return initialize_shared_transformer_parameters(
            jax.random.PRNGKey(seed), schema=self.learned_schema, layout=self.layout,
            inputs=inputs, config=self.shared_config, carbon_state_layout=self.carbon_state_layout)

    def assemble_context(self, parameters, inputs):
        return assemble_shared_transformer_context(
            parameters, inputs, normalization=self.input_normalization,
            layout=self.layout, config=self.shared_config)

    def learned_outputs(self, parameters: Mapping[str, Any], context, inputs):
        raw, active = shared_transformer_raw_outputs(
            parameters, context, inputs, schema=self.learned_schema, layout=self.layout,
            input_normalization=self.input_normalization, config=self.shared_config,
            carbon_state_layout=self.carbon_state_layout)
        outputs = {
            field.name: transform_shared_output(
                raw[field.name], active[field.name], field=field,
                normalization=self.output_normalization, inputs=inputs,
                outgoing_center_floor=self.outgoing_center_floor,
                thermal_transform=self.thermal_transform)
            for field in self.learned_schema.fields if field.name != MAINTENANCE_FIELD
        }
        outputs[GPP_FIELD] = apply_shared_gpp_factor(outputs[GPP_FIELD], inputs)
        n_pft = len(self.layout.active_pft_indices)
        correction = jnp.tanh(jnp.asarray(raw[MAINTENANCE_FIELD], jnp.float64)).reshape(n_pft, 12)
        outputs[MAINTENANCE_FIELD] = materialize_part_resolved_source_shaped_maintenance(
            inputs, outputs, self.coeff_maint_zero,
            self.reconstructor.static(inputs, "ext_coeff_vegetfrac"), correction,
            layout=self.layout, forcing_layout=self.forcing_layout)
        if self.apply_gsmean_applicability:
            outputs = apply_gsmean_output_applicability(outputs, inputs, schema=self.learned_schema)
        return outputs


def build_shared_process_head(parent_head, retained_tail, *, config=SharedTransformerConfig(),
                              thermal_transform="train_centered"):
    """Reuse canonical data/physics metadata, never the parent's neural weights."""
    kwargs = {field.name: getattr(parent_head, field.name)
              for field in fields(MechanisticStrongBaselineProcessHead) if field.init}
    normalization = parent_head.input_normalization
    doc = next(leaf for leaf in parent_head.contract.state_leaves
               if leaf.key == "slowproc_stomate_previous_step_state.DOC")
    doc_mask = np.zeros_like(normalization.continuous_mean, dtype=bool)
    doc_mask[doc.start:doc.stop] = True
    kwargs.update(
        input_normalization=normalization._replace(
            continuous_tail_threshold=16.0, continuous_tail_mask=doc_mask),
        config=replace(parent_head.config, neural_precision=NeuralPrecisionPolicy(
            attention_scores_fp32=True, source_family_pooling_fp32=True)),
        warm_snow_free_boundary=True,
        message_gate_scale=0.0,
    )
    carbon_layout = build_carbon_feedback_state_layout(
        parent_head.contract, baseline_layout=parent_head.layout,
        architecture_contract=parent_head.architecture_contract,
        coupled_contract=load_coupled_gpp_carbon_feedback_contract())
    return SharedTransformerProcessHead(
        **kwargs, shared_config=config, carbon_state_layout=carbon_layout,
        thermal_transform=thermal_transform,
        target_contract=load_target_oriented_contract(),
        coeff_maint_zero=compact_coeff_maint_zero_from_retained_tail(retained_tail))
