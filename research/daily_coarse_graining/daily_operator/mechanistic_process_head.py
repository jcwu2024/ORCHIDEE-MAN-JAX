"""Approved mechanistic-output process head over the strong Attention backbone."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import DailyMarkovContract

from .coordinate_coupled_doc_attention import (
    CoordinateCoupledDocTransitionConfig,
    coordinate_coupled_doc_transition_process_outputs,
)
from .doc_status_branch_magnitude_attention import (
    DocStatusBranchMagnitudeConfig,
    doc_status_branch_magnitude_process_outputs,
)
from .fast_owner_reconstruction import build_successor_fast_owner_materializer
from .hierarchical_coordinate_attention import (
    HierarchicalCoordinateAttentionConfig,
    hierarchical_coordinate_attention_process_outputs,
)
from .input_assembly import CanonicalDailyOperatorInputAssembler
from .mechanistic_output_contract import MechanisticOutputContract
from .mechanistic_output_reconstruction import DailyBoundaryReconstructor
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
    build_strong_baseline_layout,
    strong_baseline_process_outputs,
)
from .strong_baseline_contract import (
    StrongBaselineContract,
    resolve_strong_baseline_field_ledger,
)
from .successor_fast_owner_decoder import validate_successor_output_normalization
from .types import (
    AssembledDailyOperatorInput,
    ConservativeUpdateResult,
    ProcessPrediction,
)


def build_mechanistic_output_layout(
    *,
    adapter: Any,
    complete_schema: NeuralOutputSchema,
    learned_schema: NeuralOutputSchema,
    owner_contract: Any,
    architecture_contract: StrongBaselineContract,
    mechanism_contract: MechanisticOutputContract,
    inputs: AssembledDailyOperatorInput,
) -> StrongBaselineLayout:
    """Route the coupled endpoint group through one surface-energy expert."""

    complete_ledger = resolve_strong_baseline_field_ledger(
        architecture_contract,
        schema=complete_schema,
        adapter=adapter,
        owner_contract=owner_contract,
    )
    learned_names = {field.name for field in learned_schema.fields}
    learned_ledger = tuple(field for field in complete_ledger if field.name in learned_names)
    layout = build_strong_baseline_layout(
        adapter=adapter,
        schema=learned_schema,
        owner_contract=owner_contract,
        contract=architecture_contract,
        inputs=inputs,
        resolved_ledger=learned_ledger,
    )
    by_name = {field.name: index for index, field in enumerate(layout.ledger)}
    coupled = tuple(mechanism_contract.surface_energy_fields)
    missing = sorted(set(coupled) - set(by_name))
    if missing:
        raise ValueError(f"coupled surface-energy fields are absent: {missing}")
    target = layout.expert_names.index("surface_energy")
    source_experts = {int(layout.field_expert_index[by_name[name]]) for name in coupled}
    ledger = tuple(
        replace(field, expert="surface_energy") if field.name in set(coupled) else field for field in layout.ledger
    )
    field_expert_index = np.asarray(
        [layout.expert_names.index(field.expert) for field in ledger],
        dtype=np.int32,
    )

    def union_target(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value).copy()
        for source in source_experts:
            result[target] |= result[source]
        return result

    return replace(
        layout,
        ledger=ledger,
        field_expert_index=field_expert_index,
        state_axis_pool=union_target(layout.state_axis_pool),
        discrete_state_axis_pool=union_target(layout.discrete_state_axis_pool),
        static_axis_pool=union_target(layout.static_axis_pool),
        process_static_axis_pool=union_target(layout.process_static_axis_pool),
        source_allowed=union_target(layout.source_allowed),
    )


@dataclass(frozen=True, eq=False)
class MechanisticStrongBaselineProcessHead:
    """Emit primitives first, then materialize one complete daily boundary."""

    contract: DailyMarkovContract
    architecture_contract: StrongBaselineContract
    mechanism_contract: MechanisticOutputContract
    input_assembler: CanonicalDailyOperatorInputAssembler
    complete_schema: NeuralOutputSchema
    learned_schema: NeuralOutputSchema
    layout: StrongBaselineLayout
    input_normalization: NeuralInputNormalization
    output_normalization: NeuralOutputNormalization
    forcing_layout: NativeForcingFieldLayout
    forcing_indices: Mapping[str, int]
    message_gate_scale: float = 1.0
    config: StrongBaselineConfig = StrongBaselineConfig()
    _named_output_adapter: NeuralProcessHead = field(init=False, repr=False)
    _reconstructor: DailyBoundaryReconstructor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        identity = self.architecture_contract.payload["data_identity"]
        if self.contract.sha256 != self.input_assembler.contract.sha256:
            raise ValueError("mechanistic head and canonical assembler contract drift")
        if self.contract.sha256 != identity["markov_contract_sha256"]:
            raise ValueError("mechanistic architecture and Markov contracts drift")
        if tuple(self.learned_schema.active_pft_indices) != tuple(self.contract.active_pft_indices):
            raise ValueError("mechanistic schema and Markov PFT axes drift")
        if tuple(field.name for field in self.layout.ledger) != tuple(
            field.name for field in self.learned_schema.fields
        ):
            raise ValueError("mechanistic layout and learned schema drift")
        complete_names = {field.name for field in self.complete_schema.fields}
        learned_names = {field.name for field in self.learned_schema.fields}
        if complete_names - learned_names != set(self.mechanism_contract.exact_fields):
            raise ValueError("mechanistic complete/learned schema partition drift")
        self.config.validate(self.architecture_contract)
        validate_successor_output_normalization(
            self.complete_schema,
            self.output_normalization,
        )
        adapter = NeuralProcessHead(
            arm="mechanistic_named_output_adapter_only",
            contract=self.contract,
            input_assembler=self.input_assembler,
            schema=self.complete_schema,
            axis_layout=build_process_axis_layout(self.input_assembler),
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            forcing_indices=self.forcing_indices,
        )
        reconstructor = DailyBoundaryReconstructor(
            assembler=self.input_assembler,
            contract=self.mechanism_contract,
            forcing_indices=self.forcing_indices,
            output_scale=self.output_normalization.scale,
            fast_owner_materializer=build_successor_fast_owner_materializer(
                self.contract,
                forcing_layout=self.forcing_layout,
            ),
        )
        object.__setattr__(self, "_named_output_adapter", adapter)
        object.__setattr__(self, "_reconstructor", reconstructor)

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

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return strong_baseline_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
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
        learned = self.learned_outputs(parameters, context, inputs)
        return self.prediction_from_learned_outputs(learned, inputs)

    def prediction_from_learned_outputs(
        self,
        learned_outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        """Apply the approved mechanism to already-decoded learned coordinates."""

        primitive, auxiliary = self._reconstructor.pre_update_outputs(
            learned_outputs,
            inputs,
        )
        prediction = self._named_output_adapter.prediction_from_named_outputs(
            primitive,
            inputs,
        )
        return prediction._replace(auxiliary_outputs=auxiliary)

    @property
    def reconstructor(self) -> DailyBoundaryReconstructor:
        return self._reconstructor

    def materialize_post_update(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
        prediction: ProcessPrediction,
        fast_update: ConservativeUpdateResult,
    ) -> ProcessPrediction:
        del parameters, context
        complete = self._reconstructor.post_update_outputs(
            prediction,
            fast_update,
            inputs,
        )
        expected = {field.name for field in self.complete_schema.fields}
        if set(complete) != expected:
            missing = sorted(expected - set(complete))
            extra = sorted(set(complete) - expected)
            raise ValueError(f"mechanistic complete boundary drift; missing={missing}, extra={extra}")
        complete_prediction = self._named_output_adapter.prediction_from_named_outputs(
            complete,
            inputs,
        )
        return complete_prediction._replace(auxiliary_outputs=prediction.auxiliary_outputs)


@dataclass(frozen=True, eq=False)
class MechanisticHierarchicalCoordinateAttentionProcessHead(MechanisticStrongBaselineProcessHead):
    """Use coordinate-query Attention only for the litter/POC/DOC branch."""

    coordinate_config: HierarchicalCoordinateAttentionConfig = HierarchicalCoordinateAttentionConfig()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.coordinate_config.validate(self.config)

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return hierarchical_coordinate_attention_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            config=self.coordinate_config,
        )


@dataclass(frozen=True, eq=False)
class MechanisticCoordinateCoupledDocTransitionProcessHead(MechanisticHierarchicalCoordinateAttentionProcessHead):
    """Add coordinate-aligned DOC transfers and one joint inventory head."""

    doc_transition_config: CoordinateCoupledDocTransitionConfig = CoordinateCoupledDocTransitionConfig()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.doc_transition_config.validate(self.config, self.coordinate_config)

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        return coordinate_coupled_doc_transition_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            coordinate_config=self.coordinate_config,
            config=self.doc_transition_config,
        )


@dataclass(frozen=True, eq=False)
class MechanisticDocStatusBranchMagnitudeProcessHead(
    MechanisticCoordinateCoupledDocTransitionProcessHead
):
    """Use coordinate-aligned DOC status and branch-specific magnitude heads."""

    doc_status_config: DocStatusBranchMagnitudeConfig = (
        DocStatusBranchMagnitudeConfig()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.doc_status_config.validate(
            self.config,
            self.coordinate_config,
            self.doc_transition_config,
        )

    def learned_outputs_with_auxiliary(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        return doc_status_branch_magnitude_process_outputs(
            parameters,
            context,
            inputs,
            schema=self.learned_schema,
            layout=self.layout,
            input_normalization=self.input_normalization,
            output_normalization=self.output_normalization,
            forcing_layout=self.forcing_layout,
            message_gate_scale=self.message_gate_scale,
            baseline_config=self.config,
            coordinate_config=self.coordinate_config,
            parent_config=self.doc_transition_config,
            config=self.doc_status_config,
        )

    def learned_outputs(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> Mapping[str, Any]:
        learned, _auxiliary = self.learned_outputs_with_auxiliary(
            parameters,
            context,
            inputs,
        )
        return learned

    def __call__(
        self,
        parameters: Mapping[str, Any],
        context: StrongBaselineContext,
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        learned, doc_auxiliary = self.learned_outputs_with_auxiliary(
            parameters,
            context,
            inputs,
        )
        prediction = self.prediction_from_learned_outputs(learned, inputs)
        auxiliary = dict(prediction.auxiliary_outputs)
        auxiliary.update(doc_auxiliary)
        return prediction._replace(auxiliary_outputs=auxiliary)


__all__ = [
    "MechanisticCoordinateCoupledDocTransitionProcessHead",
    "MechanisticDocStatusBranchMagnitudeProcessHead",
    "MechanisticHierarchicalCoordinateAttentionProcessHead",
    "MechanisticStrongBaselineProcessHead",
    "build_mechanistic_output_layout",
]
