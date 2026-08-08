"""Formal process-head adapter for the source-routed successor pair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from research.daily_coarse_graining.daily_markov_contract import DailyMarkovContract

from .input_assembly import CanonicalDailyOperatorInputAssembler
from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .neural_process_head import NeuralProcessHead
from .process_derivations import NativeForcingFieldLayout
from .successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
    SOURCE_ROUTED_GRAPH_CONTROL,
    SuccessorArchitectureConfig,
    SuccessorProcessContext,
    SuccessorTokenLayout,
    assemble_successor_process_context,
    successor_process_outputs,
)
from .types import AssembledDailyOperatorInput, ProcessPrediction


@dataclass(frozen=True, eq=False)
class SuccessorProcessHead:
    """Bind a successor arm to the accepted conservative process boundary.

    The private adapter below is used only for the accepted mapping from 101
    named outputs to ``ProcessPrediction``. It never invokes the closed first
    candidate's encoder, parameter tree, or output decoder.
    """

    arm: str
    contract: DailyMarkovContract
    input_assembler: CanonicalDailyOperatorInputAssembler
    schema: NeuralOutputSchema
    token_layout: SuccessorTokenLayout
    input_normalization: NeuralInputNormalization
    output_normalization: NeuralOutputNormalization
    forcing_layout: NativeForcingFieldLayout
    forcing_indices: Mapping[str, int]
    config: SuccessorArchitectureConfig = SuccessorArchitectureConfig()
    _named_output_adapter: NeuralProcessHead = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.arm not in {SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL}:
            raise ValueError(f"unknown successor arm {self.arm!r}")
        if self.contract.sha256 != self.input_assembler.contract.sha256:
            raise ValueError("successor head and canonical assembler contract drift")
        if tuple(self.schema.active_pft_indices) != tuple(self.contract.active_pft_indices):
            raise ValueError("successor schema and Markov PFT axes drift")
        self.config.validate()
        adapter = NeuralProcessHead(
            arm="named_output_adapter_only",
            contract=self.contract,
            input_assembler=self.input_assembler,
            schema=self.schema,
            axis_layout=self.token_layout.axis_layout,
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
    ) -> SuccessorProcessContext:
        return assemble_successor_process_context(
            parameters,
            inputs,
            normalization=self.input_normalization,
            layout=self.token_layout,
            config=self.config,
        )

    def named_outputs(
        self,
        parameters: Mapping[str, Any],
        context: SuccessorProcessContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return successor_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.schema,
            layout=self.token_layout,
            normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            arm=self.arm,
            config=self.config,
        )

    def __call__(
        self,
        parameters: Mapping[str, Any],
        context: SuccessorProcessContext,
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        outputs = self.named_outputs(parameters, context, inputs)
        return self.prediction_from_named_outputs(outputs, inputs)

    def prediction_from_named_outputs(
        self,
        outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        """Apply only the accepted named-output scientific mapping."""

        return self._named_output_adapter.prediction_from_named_outputs(outputs, inputs)
