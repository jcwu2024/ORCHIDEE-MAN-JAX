"""Versioned model construction and checkpoint dispatch for daily operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    canonical_model_apply,
    initialize_canonical_model,
)
from research.daily_coarse_graining.structured_canonical_daily_model import (
    StructuredCanonicalModelSpec,
    initialize_structured_canonical_model,
    structured_canonical_model_apply,
    structured_spec_from_contract,
)

CANONICAL_FLAT_V1 = "canonical_flat_v1"
STRUCTURED_PROCESS_FILM_V1 = "structured_process_film_v1"
MODEL_ARCHITECTURES = (CANONICAL_FLAT_V1, STRUCTURED_PROCESS_FILM_V1)


@dataclass(frozen=True)
class DailyModelDefinition:
    architecture_id: str
    config: CanonicalModelConfig
    structured_spec: StructuredCanonicalModelSpec | None = None

    def identity(self) -> dict[str, Any]:
        if self.architecture_id == CANONICAL_FLAT_V1:
            return {"id": CANONICAL_FLAT_V1}
        if self.structured_spec is None:
            raise ValueError("structured architecture is missing its static spec")
        return self.structured_spec.identity()

    def initialize(self, *, seed: int, canonical_parameters=None):
        if self.architecture_id == CANONICAL_FLAT_V1:
            if canonical_parameters is not None:
                return canonical_parameters
            return initialize_canonical_model(self.config, seed=seed)
        if self.structured_spec is None:
            raise ValueError("structured architecture is missing its static spec")
        return initialize_structured_canonical_model(
            self.structured_spec,
            seed=seed,
            canonical_parameters=canonical_parameters,
        )

    def apply(self, parameters, batch):
        if self.architecture_id == CANONICAL_FLAT_V1:
            return canonical_model_apply(parameters, batch)
        if self.structured_spec is None:
            raise ValueError("structured architecture is missing its static spec")
        return structured_canonical_model_apply(
            parameters,
            batch,
            self.structured_spec,
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
            "checkpoint architecture does not match requested model: "
            f"{observed_id} != {definition.architecture_id}"
        )
    observed = identity.get("model_architecture", {"id": CANONICAL_FLAT_V1})
    if observed != definition.identity():
        raise ValueError("checkpoint structured architecture identity has drifted")
