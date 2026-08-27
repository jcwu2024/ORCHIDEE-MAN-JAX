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
from .target_oriented_source_shaped_maintenance import (
    SourceShapedMaintenanceConfig,
    SourceShapedMaintenanceContract,
    compact_coeff_maint_zero_from_retained_tail,
    load_source_shaped_maintenance_contract,
    target_oriented_source_shaped_maintenance_process_outputs,
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


__all__ = [
    "MechanisticTargetOrientedAttentionProcessHead",
    "MechanisticTargetOrientedSourceShapedMaintenanceProcessHead",
    "build_target_oriented_source_shaped_maintenance_head",
]
