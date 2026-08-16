"""Formal process-head adapter for the approved strong baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from research.daily_coarse_graining.daily_markov_contract import DailyMarkovContract

from .input_assembly import CanonicalDailyOperatorInputAssembler
from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
    build_process_axis_layout,
)
from .neural_process_head import NeuralProcessHead
from .process_derivations import NativeForcingFieldLayout
from .strong_baseline_architecture import (
    StrongBaselineConfig,
    StrongBaselineContext,
    StrongBaselineLayout,
    assemble_strong_baseline_context,
    strong_baseline_process_outputs,
)
from .strong_baseline_contract import StrongBaselineContract
from .successor_fast_owner_decoder import validate_successor_output_normalization
from .types import AssembledDailyOperatorInput, ProcessPrediction


@dataclass(frozen=True, eq=False)
class StrongBaselineProcessHead:
    """Bind the seven-expert network to the accepted scientific mapping."""

    contract: DailyMarkovContract
    architecture_contract: StrongBaselineContract
    input_assembler: CanonicalDailyOperatorInputAssembler
    schema: NeuralOutputSchema
    layout: StrongBaselineLayout
    input_normalization: NeuralInputNormalization
    output_normalization: NeuralOutputNormalization
    forcing_layout: NativeForcingFieldLayout
    forcing_indices: Mapping[str, int]
    message_gate_scale: float = 1.0
    config: StrongBaselineConfig = StrongBaselineConfig()
    _named_output_adapter: NeuralProcessHead = field(init=False, repr=False)

    def __post_init__(self) -> None:
        identity = self.architecture_contract.payload["data_identity"]
        if self.contract.sha256 != self.input_assembler.contract.sha256:
            raise ValueError("strong-baseline head and canonical assembler contract drift")
        if self.contract.sha256 != identity["markov_contract_sha256"]:
            raise ValueError("strong-baseline architecture and Markov contracts drift")
        if tuple(self.schema.active_pft_indices) != tuple(self.contract.active_pft_indices):
            raise ValueError("strong-baseline schema and Markov PFT axes drift")
        if tuple(self.layout.ledger[index].name for index in range(len(self.schema.fields))) != tuple(
            field.name for field in self.schema.fields
        ):
            raise ValueError("strong-baseline layout and output schema drift")
        self.config.validate(self.architecture_contract)
        validate_successor_output_normalization(self.schema, self.output_normalization)
        adapter = NeuralProcessHead(
            arm="named_output_adapter_only",
            contract=self.contract,
            input_assembler=self.input_assembler,
            schema=self.schema,
            axis_layout=build_process_axis_layout(self.input_assembler),
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            forcing_indices=self.forcing_indices,
        )
        object.__setattr__(self, "_named_output_adapter", adapter)

    def assemble_context(
        self,
        parameters: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ) -> StrongBaselineContext:
        return assemble_strong_baseline_context(
            parameters,
            inputs,
            normalization=self.input_normalization,
            layout=self.layout,
            config=self.config,
        )

    def named_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return strong_baseline_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.schema,
            layout=self.layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            message_gate_scale=self.message_gate_scale,
            config=self.config,
        )

    def __call__(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        outputs = self.named_outputs(parameters, context, inputs)
        return self.prediction_from_named_outputs(outputs, inputs)

    def prediction_from_named_outputs(
        self,
        outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        return self._named_output_adapter.prediction_from_named_outputs(outputs, inputs)


__all__ = ["StrongBaselineProcessHead"]
