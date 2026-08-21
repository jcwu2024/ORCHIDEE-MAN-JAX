"""Mechanistic binding for the target-oriented Attention candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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


__all__ = ["MechanisticTargetOrientedAttentionProcessHead"]
