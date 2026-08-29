"""Mechanistic binding for the target-oriented Attention candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .mechanistic_process_head import MechanisticStrongBaselineProcessHead
from .strong_baseline_architecture import StrongBaselineContext
from .target_oriented_attention import (
    TargetOrientedAttentionConfig,
    target_oriented_attention_process_outputs,
)
from .target_oriented_contract import (
    TargetOrientedContract,
    resolve_target_output_hierarchy,
)
from .target_oriented_distribution_aware_maintenance import (
    DistributionAwareMaintenanceConfig,
    DistributionAwareMaintenanceContract,
    load_distribution_aware_maintenance_contract,
    target_oriented_distribution_aware_maintenance_process_outputs,
)
from .target_oriented_source_shaped_maintenance import (
    SourceShapedMaintenanceConfig,
    SourceShapedMaintenanceContract,
    compact_coeff_maint_zero_from_retained_tail,
    load_source_shaped_maintenance_contract,
    target_oriented_source_shaped_maintenance_process_outputs,
)
from .target_oriented_structured_gpp import (
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
        distribution_contract=parent.distribution_contract,
        coeff_maint_zero=parent.coeff_maint_zero,
        distribution_config=parent.distribution_config,
        source_config=parent.source_config,
        structured_contract=contract,
        structured_state_layout=build_structured_gpp_state_layout(parent.contract),
        structured_config=structured_config,
    )


__all__ = [
    "MechanisticTargetOrientedAttentionProcessHead",
    "MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead",
    "MechanisticTargetOrientedSourceShapedMaintenanceProcessHead",
    "MechanisticTargetOrientedStructuredGPPProcessHead",
    "build_target_oriented_distribution_aware_maintenance_head",
    "build_target_oriented_source_shaped_maintenance_head",
    "build_target_oriented_structured_gpp_head",
]
