"""Mechanistic binding for the target-oriented Attention candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from .mechanistic_process_head import MechanisticStrongBaselineProcessHead
from .output_applicability import apply_gsmean_output_applicability
from .parameter_routing import DEFAULT_PFT_PARAMETER_ROUTING, VCMAX25_FACTOR_SCALE
from .strong_baseline_architecture import StrongBaselineContext
from .target_oriented_attention import (
    TargetOrientedAttentionConfig,
    _target_oriented_attention_decoded_outputs,
    target_oriented_attention_process_outputs,
)
from .target_oriented_contract import (
    TargetOrientedContract,
    resolve_target_output_hierarchy,
)
from .target_oriented_coupled_gpp_carbon_feedback import (
    CarbonFeedbackStateLayout,
    CoupledGPPCarbonFeedbackConfig,
    CoupledGPPCarbonFeedbackContract,
    build_carbon_feedback_state_layout,
    coupled_gpp_carbon_feedback_factorization,
    load_coupled_gpp_carbon_feedback_contract,
    target_oriented_coupled_gpp_carbon_feedback_process_outputs,
)
from .target_oriented_coupled_gpp_conditioned_attention import (
    CoupledGPPConditionedAttentionConfig,
    CoupledGPPConditionedAttentionContract,
    coupled_gpp_conditioned_attention_temperatures,
    fixed_reference_attention_temperatures,
    load_coupled_gpp_conditioned_attention_contract,
)
from .target_oriented_distribution_aware_maintenance import (
    DistributionAwareMaintenanceConfig,
    DistributionAwareMaintenanceContract,
    load_distribution_aware_maintenance_contract,
    target_oriented_distribution_aware_maintenance_process_outputs,
)
from .target_oriented_source_shaped_maintenance import (
    MAINTENANCE_FIELD,
    SourceShapedMaintenanceConfig,
    SourceShapedMaintenanceContract,
    compact_coeff_maint_zero_from_retained_tail,
    load_source_shaped_maintenance_contract,
    target_oriented_source_shaped_maintenance_process_outputs,
)
from .target_oriented_structured_gpp import (
    GPP_FIELD,
    StructuredGPPConfig,
    StructuredGPPContract,
    StructuredGPPStateLayout,
    build_structured_gpp_state_layout,
    load_structured_gpp_contract,
    target_oriented_structured_gpp_process_outputs,
)
from .types import AssembledDailyOperatorInput


@dataclass(frozen=True, eq=False, kw_only=True)
class MechanisticTargetOrientedAttentionProcessHead(MechanisticStrongBaselineProcessHead):
    """Decode learned fields with the target hierarchy at the stable boundary."""

    target_contract: TargetOrientedContract
    target_config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig()
    apply_gsmean_applicability: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.target_config.validate(self.config)
        resolve_target_output_hierarchy(
            self.learned_schema,
            self.complete_schema,
            self.target_contract,
        )

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return target_oriented_attention_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            target_contract=self.target_contract,
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            config=self.target_config,
        )

    def prediction_from_learned_outputs(
        self,
        learned_outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ):
        """Apply opt-in source applicability before physical reconstruction."""

        if self.apply_gsmean_applicability:
            learned_outputs = apply_gsmean_output_applicability(
                learned_outputs,
                inputs,
                schema=self.learned_schema,
            )
        return super().prediction_from_learned_outputs(learned_outputs, inputs)


@dataclass(frozen=True, eq=False, kw_only=True)
class MechanisticTargetOrientedSourceShapedMaintenanceProcessHead(MechanisticTargetOrientedAttentionProcessHead):
    """Replace only direct maintenance decoding with the source-shaped head."""

    source_contract: SourceShapedMaintenanceContract
    coeff_maint_zero: Any
    source_config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.source_config.validate(self.source_contract, self.target_config)
        coefficient = np.asarray(self.coeff_maint_zero)
        expected = (len(self.layout.active_pft_indices), 12)
        if coefficient.shape != expected or not np.all(np.isfinite(coefficient)):
            raise ValueError(f"source-shaped maintenance coefficients must be finite with shape {expected}")

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return target_oriented_source_shaped_maintenance_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            target_contract=self.target_contract,
            source_contract=self.source_contract,
            coeff_maint_zero=self.coeff_maint_zero,
            ext_coeff_vegetfrac=self.reconstructor.static(
                inputs,
                "ext_coeff_vegetfrac",
            ),
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            attention_config=self.target_config,
            config=self.source_config,
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead(MechanisticTargetOrientedAttentionProcessHead):
    """Replace direct maintenance decoding with the part-resolved adapter."""

    distribution_contract: DistributionAwareMaintenanceContract
    coeff_maint_zero: Any
    distribution_config: DistributionAwareMaintenanceConfig = DistributionAwareMaintenanceConfig()
    source_config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.distribution_config.validate(self.distribution_contract, self.target_config)
        coefficient = np.asarray(self.coeff_maint_zero)
        expected = (len(self.layout.active_pft_indices), 12)
        if coefficient.shape != expected or not np.all(np.isfinite(coefficient)):
            raise ValueError(f"distribution-aware maintenance coefficients must be finite with shape {expected}")

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return target_oriented_distribution_aware_maintenance_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            target_contract=self.target_contract,
            distribution_contract=self.distribution_contract,
            coeff_maint_zero=self.coeff_maint_zero,
            ext_coeff_vegetfrac=self.reconstructor.static(
                inputs,
                "ext_coeff_vegetfrac",
            ),
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            attention_config=self.target_config,
            config=self.distribution_config,
            source_config=self.source_config,
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class MechanisticTargetOrientedStructuredGPPProcessHead(
    MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead
):
    """Replace only daily GPP with the approved day-local specialist."""

    structured_contract: StructuredGPPContract
    structured_state_layout: StructuredGPPStateLayout
    structured_config: StructuredGPPConfig = StructuredGPPConfig()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.structured_config.validate(self.structured_contract, self.target_config)

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return target_oriented_structured_gpp_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            state_layout=self.structured_state_layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            target_contract=self.target_contract,
            distribution_contract=self.distribution_contract,
            coeff_maint_zero=self.coeff_maint_zero,
            ext_coeff_vegetfrac=self.reconstructor.static(
                inputs,
                "ext_coeff_vegetfrac",
            ),
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            attention_config=self.target_config,
            distribution_config=self.distribution_config,
            source_config=self.source_config,
            config=self.structured_config,
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class MechanisticTargetOrientedCoupledGPPCarbonFeedbackProcessHead(MechanisticTargetOrientedStructuredGPPProcessHead):
    """Condition only daily GPP on coordinate-preserving day-start carbon state."""

    coupled_contract: CoupledGPPCarbonFeedbackContract
    carbon_state_layout: CarbonFeedbackStateLayout
    coupled_config: CoupledGPPCarbonFeedbackConfig = CoupledGPPCarbonFeedbackConfig()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.coupled_config.validate(self.coupled_contract, self.structured_config)

    def potential_attention_temperatures(self, parameters: Mapping[str, Any]) -> Any | None:
        del parameters
        return None

    def potential_attention_normalization_epsilon(self) -> float:
        return 1.0e-6

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return target_oriented_coupled_gpp_carbon_feedback_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            structured_state_layout=self.structured_state_layout,
            carbon_state_layout=self.carbon_state_layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            target_contract=self.target_contract,
            distribution_contract=self.distribution_contract,
            coeff_maint_zero=self.coeff_maint_zero,
            ext_coeff_vegetfrac=self.reconstructor.static(
                inputs,
                "ext_coeff_vegetfrac",
            ),
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            attention_config=self.target_config,
            distribution_config=self.distribution_config,
            source_config=self.source_config,
            structured_config=self.structured_config,
            config=self.coupled_config,
            potential_attention_temperatures=self.potential_attention_temperatures(parameters),
            potential_attention_normalization_epsilon=self.potential_attention_normalization_epsilon(),
        )

    def gpp_factorization(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        """Return the existing routed GPP factors for diagnostics."""

        _outputs, hidden = _target_oriented_attention_decoded_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            target_contract=self.target_contract,
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            config=self.target_config,
            deferred_primary_fields=frozenset((GPP_FIELD, MAINTENANCE_FIELD)),
        )
        factors = coupled_gpp_carbon_feedback_factorization(
            parameters,
            context,
            hidden[GPP_FIELD],
            inputs,
            layout=self.layout,
            structured_state_layout=self.structured_state_layout,
            carbon_state_layout=self.carbon_state_layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            structured_config=self.structured_config,
            config=self.coupled_config,
            potential_attention_temperatures=self.potential_attention_temperatures(parameters),
            potential_attention_normalization_epsilon=self.potential_attention_normalization_epsilon(),
        )
        vcmax_index = DEFAULT_PFT_PARAMETER_ROUTING.index("vcmax25")
        vcmax_factor = (
            jnp.maximum(
                jnp.asarray(inputs.pft.pft_parameters[:, vcmax_index], dtype=jnp.float64),
                0.0,
            )
            / VCMAX25_FACTOR_SCALE
        )
        return {
            "potential": factors.potential,
            "limitation": factors.limitation,
            "correction": factors.correction,
            "normalization_scale": factors.normalization_scale,
            "pre_vcmax_gpp": factors.pre_vcmax_gpp,
            "vcmax25_factor": vcmax_factor,
            "routed_gpp": factors.pre_vcmax_gpp * vcmax_factor[None, :],
        }


@dataclass(frozen=True, eq=False, kw_only=True)
class MechanisticTargetOrientedCoupledGPPConditionedAttentionProcessHead(
    MechanisticTargetOrientedCoupledGPPCarbonFeedbackProcessHead
):
    """Bound Q/K-conditioned potential attention to the existing coupled head."""

    conditioning_contract: CoupledGPPConditionedAttentionContract
    conditioning_config: CoupledGPPConditionedAttentionConfig = CoupledGPPConditionedAttentionConfig()
    fixed_temperature_reference: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.conditioning_config.validate(self.conditioning_contract, self.structured_config)

    def potential_attention_temperatures(self, parameters: Mapping[str, Any]) -> Any:
        if self.fixed_temperature_reference:
            return fixed_reference_attention_temperatures(config=self.conditioning_config)
        return coupled_gpp_conditioned_attention_temperatures(
            parameters,
            config=self.conditioning_config,
        )

    def potential_attention_normalization_epsilon(self) -> float:
        return self.conditioning_config.normalization_epsilon


def build_target_oriented_source_shaped_maintenance_head(
    parent: MechanisticTargetOrientedAttentionProcessHead,
    retained_tail: Any,
    *,
    source_contract: SourceShapedMaintenanceContract | None = None,
    source_config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig(),
) -> MechanisticTargetOrientedSourceShapedMaintenanceProcessHead:
    """Bind the exact retained static coefficient owner to the new process head."""

    contract = source_contract or load_source_shaped_maintenance_contract()
    if retained_tail.static_owner.contract.sha256 != parent.contract.sha256:
        raise ValueError("source-shaped head and retained-tail Markov contracts drift")
    coefficient = compact_coeff_maint_zero_from_retained_tail(retained_tail)
    return MechanisticTargetOrientedSourceShapedMaintenanceProcessHead(
        contract=parent.contract,
        architecture_contract=parent.architecture_contract,
        mechanism_contract=parent.mechanism_contract,
        input_assembler=parent.input_assembler,
        complete_schema=parent.complete_schema,
        learned_schema=parent.learned_schema,
        layout=parent.layout,
        input_normalization=parent.input_normalization,
        output_normalization=parent.output_normalization,
        forcing_layout=parent.forcing_layout,
        forcing_indices=parent.forcing_indices,
        message_gate_scale=parent.message_gate_scale,
        config=parent.config,
        target_contract=parent.target_contract,
        target_config=parent.target_config,
        apply_gsmean_applicability=parent.apply_gsmean_applicability,
        source_contract=contract,
        coeff_maint_zero=coefficient,
        source_config=source_config,
    )


def build_target_oriented_distribution_aware_maintenance_head(
    parent: MechanisticTargetOrientedAttentionProcessHead,
    retained_tail: Any,
    *,
    distribution_contract: DistributionAwareMaintenanceContract | None = None,
    distribution_config: DistributionAwareMaintenanceConfig = DistributionAwareMaintenanceConfig(),
    source_config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig(),
) -> MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead:
    """Bind the exact retained static coefficient owner to the part-resolved head."""

    contract = distribution_contract or load_distribution_aware_maintenance_contract()
    if retained_tail.static_owner.contract.sha256 != parent.contract.sha256:
        raise ValueError("distribution-aware head and retained-tail Markov contracts drift")
    coefficient = compact_coeff_maint_zero_from_retained_tail(retained_tail)
    return MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead(
        contract=parent.contract,
        architecture_contract=parent.architecture_contract,
        mechanism_contract=parent.mechanism_contract,
        input_assembler=parent.input_assembler,
        complete_schema=parent.complete_schema,
        learned_schema=parent.learned_schema,
        layout=parent.layout,
        input_normalization=parent.input_normalization,
        output_normalization=parent.output_normalization,
        forcing_layout=parent.forcing_layout,
        forcing_indices=parent.forcing_indices,
        message_gate_scale=parent.message_gate_scale,
        config=parent.config,
        target_contract=parent.target_contract,
        target_config=parent.target_config,
        apply_gsmean_applicability=parent.apply_gsmean_applicability,
        distribution_contract=contract,
        coeff_maint_zero=coefficient,
        distribution_config=distribution_config,
        source_config=source_config,
    )


def build_target_oriented_structured_gpp_head(
    parent: MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead,
    *,
    structured_contract: StructuredGPPContract | None = None,
    structured_config: StructuredGPPConfig = StructuredGPPConfig(),
) -> MechanisticTargetOrientedStructuredGPPProcessHead:
    """Install the structured GPP boundary over the accepted maintenance parent."""

    contract = structured_contract or load_structured_gpp_contract()
    return MechanisticTargetOrientedStructuredGPPProcessHead(
        contract=parent.contract,
        architecture_contract=parent.architecture_contract,
        mechanism_contract=parent.mechanism_contract,
        input_assembler=parent.input_assembler,
        complete_schema=parent.complete_schema,
        learned_schema=parent.learned_schema,
        layout=parent.layout,
        input_normalization=parent.input_normalization,
        output_normalization=parent.output_normalization,
        forcing_layout=parent.forcing_layout,
        forcing_indices=parent.forcing_indices,
        message_gate_scale=parent.message_gate_scale,
        config=parent.config,
        target_contract=parent.target_contract,
        target_config=parent.target_config,
        apply_gsmean_applicability=parent.apply_gsmean_applicability,
        distribution_contract=parent.distribution_contract,
        coeff_maint_zero=parent.coeff_maint_zero,
        distribution_config=parent.distribution_config,
        source_config=parent.source_config,
        structured_contract=contract,
        structured_state_layout=build_structured_gpp_state_layout(parent.contract),
        structured_config=structured_config,
    )


def build_target_oriented_coupled_gpp_carbon_feedback_head(
    parent: MechanisticTargetOrientedStructuredGPPProcessHead,
    *,
    coupled_contract: CoupledGPPCarbonFeedbackContract | None = None,
    coupled_config: CoupledGPPCarbonFeedbackConfig = CoupledGPPCarbonFeedbackConfig(),
) -> MechanisticTargetOrientedCoupledGPPCarbonFeedbackProcessHead:
    """Install the approved carbon-coordinate adapter over structured GPP."""

    contract = coupled_contract or load_coupled_gpp_carbon_feedback_contract()
    carbon_layout = build_carbon_feedback_state_layout(
        parent.contract,
        baseline_layout=parent.layout,
        architecture_contract=parent.architecture_contract,
        coupled_contract=contract,
    )
    return MechanisticTargetOrientedCoupledGPPCarbonFeedbackProcessHead(
        contract=parent.contract,
        architecture_contract=parent.architecture_contract,
        mechanism_contract=parent.mechanism_contract,
        input_assembler=parent.input_assembler,
        complete_schema=parent.complete_schema,
        learned_schema=parent.learned_schema,
        layout=parent.layout,
        input_normalization=parent.input_normalization,
        output_normalization=parent.output_normalization,
        forcing_layout=parent.forcing_layout,
        forcing_indices=parent.forcing_indices,
        message_gate_scale=parent.message_gate_scale,
        config=parent.config,
        target_contract=parent.target_contract,
        target_config=parent.target_config,
        apply_gsmean_applicability=parent.apply_gsmean_applicability,
        distribution_contract=parent.distribution_contract,
        coeff_maint_zero=parent.coeff_maint_zero,
        distribution_config=parent.distribution_config,
        source_config=parent.source_config,
        structured_contract=parent.structured_contract,
        structured_state_layout=parent.structured_state_layout,
        structured_config=parent.structured_config,
        coupled_contract=contract,
        carbon_state_layout=carbon_layout,
        coupled_config=coupled_config,
    )


def build_target_oriented_coupled_gpp_conditioned_attention_head(
    parent: MechanisticTargetOrientedCoupledGPPCarbonFeedbackProcessHead,
    *,
    conditioning_contract: CoupledGPPConditionedAttentionContract | None = None,
    conditioning_config: CoupledGPPConditionedAttentionConfig = CoupledGPPConditionedAttentionConfig(),
    fixed_temperature_reference: bool = False,
) -> MechanisticTargetOrientedCoupledGPPConditionedAttentionProcessHead:
    """Install bounded cosine potential attention over the coupled GPP head."""

    contract = conditioning_contract or load_coupled_gpp_conditioned_attention_contract()
    return MechanisticTargetOrientedCoupledGPPConditionedAttentionProcessHead(
        contract=parent.contract,
        architecture_contract=parent.architecture_contract,
        mechanism_contract=parent.mechanism_contract,
        input_assembler=parent.input_assembler,
        complete_schema=parent.complete_schema,
        learned_schema=parent.learned_schema,
        layout=parent.layout,
        input_normalization=parent.input_normalization,
        output_normalization=parent.output_normalization,
        forcing_layout=parent.forcing_layout,
        forcing_indices=parent.forcing_indices,
        message_gate_scale=parent.message_gate_scale,
        config=parent.config,
        target_contract=parent.target_contract,
        target_config=parent.target_config,
        apply_gsmean_applicability=parent.apply_gsmean_applicability,
        distribution_contract=parent.distribution_contract,
        coeff_maint_zero=parent.coeff_maint_zero,
        distribution_config=parent.distribution_config,
        source_config=parent.source_config,
        structured_contract=parent.structured_contract,
        structured_state_layout=parent.structured_state_layout,
        structured_config=parent.structured_config,
        coupled_contract=parent.coupled_contract,
        carbon_state_layout=parent.carbon_state_layout,
        coupled_config=parent.coupled_config,
        conditioning_contract=contract,
        conditioning_config=conditioning_config,
        fixed_temperature_reference=fixed_temperature_reference,
    )


__all__ = [
    "MechanisticTargetOrientedAttentionProcessHead",
    "MechanisticTargetOrientedCoupledGPPConditionedAttentionProcessHead",
    "MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead",
    "MechanisticTargetOrientedCoupledGPPCarbonFeedbackProcessHead",
    "MechanisticTargetOrientedSourceShapedMaintenanceProcessHead",
    "MechanisticTargetOrientedStructuredGPPProcessHead",
    "build_target_oriented_distribution_aware_maintenance_head",
    "build_target_oriented_coupled_gpp_carbon_feedback_head",
    "build_target_oriented_coupled_gpp_conditioned_attention_head",
    "build_target_oriented_source_shaped_maintenance_head",
    "build_target_oriented_structured_gpp_head",
]
