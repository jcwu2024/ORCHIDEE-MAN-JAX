"""Versioned model construction and checkpoint dispatch for daily operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from research.daily_coarse_graining.axis_process_coupled_daily_model import (
    AxisProcessCoupledModelSpec,
    axis_process_coupled_model_apply,
    axis_process_spec_from_contract,
    initialize_axis_process_coupled_model,
)
from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    canonical_model_apply,
    initialize_canonical_model,
)
from research.daily_coarse_graining.causal_carbon_adapter_daily_model import (
    CAUSAL_CARBON_ADAPTER_V1,
    CausalCarbonAdapterModelSpec,
    causal_carbon_adapter_model_apply,
    causal_carbon_adapter_spec_from_contract,
    initialize_causal_carbon_adapter,
)
from research.daily_coarse_graining.structured_canonical_daily_model import (
    StructuredCanonicalModelSpec,
    initialize_structured_canonical_model,
    structured_canonical_model_apply,
    structured_spec_from_contract,
)

CANONICAL_FLAT_V1 = "canonical_flat_v1"
STRUCTURED_PROCESS_FILM_V1 = "structured_process_film_v1"
AXIS_PROCESS_COUPLED_V1 = "axis_process_coupled_v1"
MODEL_ARCHITECTURES = (
    CANONICAL_FLAT_V1,
    STRUCTURED_PROCESS_FILM_V1,
    AXIS_PROCESS_COUPLED_V1,
    CAUSAL_CARBON_ADAPTER_V1,
)


@dataclass(frozen=True)
class DailyModelDefinition:
    architecture_id: str
    config: CanonicalModelConfig
    structured_spec: StructuredCanonicalModelSpec | None = None
    axis_process_spec: AxisProcessCoupledModelSpec | None = None
    causal_carbon_adapter_spec: CausalCarbonAdapterModelSpec | None = None

    def identity(self) -> dict[str, Any]:
        if self.architecture_id == CANONICAL_FLAT_V1:
            return {"id": CANONICAL_FLAT_V1}
        if self.architecture_id == STRUCTURED_PROCESS_FILM_V1:
            if self.structured_spec is None:
                raise ValueError("structured architecture is missing its static spec")
            return self.structured_spec.identity()
        if self.architecture_id == AXIS_PROCESS_COUPLED_V1:
            if self.axis_process_spec is None:
                raise ValueError("axis-process architecture is missing its static spec")
            return self.axis_process_spec.identity()
        if self.causal_carbon_adapter_spec is None:
            raise ValueError("causal carbon adapter is missing its static spec")
        return self.causal_carbon_adapter_spec.identity()

    def initialize(
        self,
        *,
        seed: int,
        canonical_parameters=None,
        parent_parameters=None,
    ):
        if self.architecture_id == CANONICAL_FLAT_V1:
            if parent_parameters is not None:
                raise ValueError("flat architecture cannot reuse parent parameters")
            if canonical_parameters is not None:
                return canonical_parameters
            return initialize_canonical_model(self.config, seed=seed)
        if self.architecture_id == STRUCTURED_PROCESS_FILM_V1:
            if parent_parameters is not None:
                raise ValueError("structured architecture cannot reuse parent parameters")
            if self.structured_spec is None:
                raise ValueError("structured architecture is missing its static spec")
            return initialize_structured_canonical_model(
                self.structured_spec,
                seed=seed,
                canonical_parameters=canonical_parameters,
            )
        if self.architecture_id == AXIS_PROCESS_COUPLED_V1:
            if canonical_parameters is not None or parent_parameters is not None:
                raise ValueError("axis-process architecture must initialize from scratch")
            if self.axis_process_spec is None:
                raise ValueError("axis-process architecture is missing its static spec")
            return initialize_axis_process_coupled_model(
                self.axis_process_spec,
                seed=seed,
            )
        if canonical_parameters is not None:
            raise ValueError("causal carbon adapter cannot reuse flat parameters")
        if parent_parameters is None:
            raise ValueError("causal carbon adapter requires frozen parent parameters")
        if self.causal_carbon_adapter_spec is None:
            raise ValueError("causal carbon adapter is missing its static spec")
        return initialize_causal_carbon_adapter(
            self.causal_carbon_adapter_spec,
            base_parameters=parent_parameters,
            seed=seed,
        )

    def apply(self, parameters, batch):
        if self.architecture_id == CANONICAL_FLAT_V1:
            return canonical_model_apply(parameters, batch)
        if self.architecture_id == STRUCTURED_PROCESS_FILM_V1:
            if self.structured_spec is None:
                raise ValueError("structured architecture is missing its static spec")
            return structured_canonical_model_apply(
                parameters,
                batch,
                self.structured_spec,
            )
        if self.architecture_id == AXIS_PROCESS_COUPLED_V1:
            if self.axis_process_spec is None:
                raise ValueError("axis-process architecture is missing its static spec")
            return axis_process_coupled_model_apply(
                parameters,
                batch,
                self.axis_process_spec,
            )
        if self.causal_carbon_adapter_spec is None:
            raise ValueError("causal carbon adapter is missing its static spec")
        return causal_carbon_adapter_model_apply(
            parameters,
            batch,
            self.causal_carbon_adapter_spec,
        )


def build_daily_model_definition(
    architecture_id: str,
    config: CanonicalModelConfig,
    contract_metadata: Mapping[str, Any],
) -> DailyModelDefinition:
    if architecture_id == CANONICAL_FLAT_V1:
        return DailyModelDefinition(architecture_id, config)
    if architecture_id == STRUCTURED_PROCESS_FILM_V1:
        return DailyModelDefinition(
            architecture_id,
            config,
            structured_spec_from_contract(config, contract_metadata),
        )
    if architecture_id == AXIS_PROCESS_COUPLED_V1:
        return DailyModelDefinition(
            architecture_id,
            config,
            axis_process_spec=axis_process_spec_from_contract(config, contract_metadata),
        )
    if architecture_id == CAUSAL_CARBON_ADAPTER_V1:
        base_spec = axis_process_spec_from_contract(config, contract_metadata)
        return DailyModelDefinition(
            architecture_id,
            config,
            causal_carbon_adapter_spec=causal_carbon_adapter_spec_from_contract(
                base_spec,
                contract_metadata,
            ),
        )
    raise ValueError(f"unsupported daily model architecture {architecture_id!r}")


def checkpoint_architecture_id(identity: Mapping[str, Any]) -> str:
    """Treat pre-registry checkpoints as their historical flat architecture."""

    architecture = identity.get("model_architecture")
    if architecture is None:
        return CANONICAL_FLAT_V1
    if not isinstance(architecture, Mapping) or "id" not in architecture:
        raise ValueError("checkpoint model architecture identity is malformed")
    architecture_id = str(architecture["id"])
    if architecture_id not in MODEL_ARCHITECTURES:
        raise ValueError(f"unsupported checkpoint architecture {architecture_id!r}")
    return architecture_id


def verify_checkpoint_architecture(
    identity: Mapping[str, Any],
    definition: DailyModelDefinition,
) -> None:
    observed_id = checkpoint_architecture_id(identity)
    if observed_id != definition.architecture_id:
        raise ValueError(
            f"checkpoint architecture does not match requested model: {observed_id} != {definition.architecture_id}"
        )
    observed = identity.get("model_architecture", {"id": CANONICAL_FLAT_V1})
    if observed != definition.identity():
        raise ValueError("checkpoint structured architecture identity has drifted")


def initialize_causal_adapter_from_parent(
    definition: DailyModelDefinition,
    *,
    parent_identity: Mapping[str, Any],
    parent_parameters,
    seed: int,
):
    """Verify an axis-process checkpoint before attaching a fresh adapter."""

    if definition.architecture_id != CAUSAL_CARBON_ADAPTER_V1:
        raise ValueError("parent conversion requires the causal carbon adapter")
    if definition.causal_carbon_adapter_spec is None:
        raise ValueError("causal carbon adapter is missing its static spec")
    parent_definition = DailyModelDefinition(
        AXIS_PROCESS_COUPLED_V1,
        definition.config,
        axis_process_spec=definition.causal_carbon_adapter_spec.base_spec,
    )
    verify_checkpoint_architecture(parent_identity, parent_definition)
    return definition.initialize(
        seed=seed,
        parent_parameters=parent_parameters,
    )
