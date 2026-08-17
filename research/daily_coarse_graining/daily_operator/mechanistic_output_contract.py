"""Approved schema partition for the mechanistic-output strong baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .neural_architecture import NeuralOutputSchema

DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "manifests" / "coarse_graining" / "strong_baseline_mechanistic_output_v2.json"
)


@dataclass(frozen=True)
class MechanisticOutputContract:
    payload: Mapping[str, Any]
    source_path: Path

    @property
    def exact_fields(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.payload["exact_daily_boundary_fields"])

    @property
    def doc_field(self) -> str:
        return str(self.payload["doc_boundary_residual_field"])

    @property
    def surface_energy_fields(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.payload["surface_energy_fields"])

    @property
    def nroot_coordinate_field(self) -> str:
        return str(self.payload["hydrology_boundary"]["nroot_coordinate_field"])

    @property
    def canopy_endpoint_field(self) -> str:
        return str(self.payload["canopy_endpoint_coordinate"]["endpoint_residual_field"])

    @property
    def canopy_status_fields(self) -> Mapping[str, str]:
        return {
            str(status): str(name)
            for status, name in self.payload["canopy_endpoint_coordinate"]["status_logit_fields"].items()
        }

    @property
    def canopy_transfer_fields(self) -> tuple[str, ...]:
        return tuple(
            str(name)
            for name in self.payload["canopy_endpoint_coordinate"]["materialized_transfer_fields"]
        )

    @property
    def canopy_debt_scale_field(self) -> str:
        return str(self.payload["canopy_endpoint_coordinate"]["debt_scale_source_field"])


def load_mechanistic_output_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> MechanisticOutputContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "strong_baseline_mechanistic_output_v2":
        raise ValueError("mechanistic-output schema version drift")
    if payload.get("status") != "approved_for_implementation" or payload.get("implementation_authorized") is not True:
        raise ValueError("mechanistic-output implementation is not authorized")
    complete = payload.get("complete_boundary", {})
    learned = payload.get("learned_boundary", {})
    if (complete.get("field_count"), complete.get("scalar_count")) != (181, 39371):
        raise ValueError("mechanistic-output complete boundary drift")
    if (learned.get("field_count"), learned.get("scalar_count")) != (161, 39006):
        raise ValueError("mechanistic-output learned boundary drift")
    exact = tuple(str(name) for name in payload.get("exact_daily_boundary_fields", ()))
    if len(exact) != 20 or len(set(exact)) != 20:
        raise ValueError("mechanistic-output exact field inventory drift")
    canopy = payload.get("canopy_endpoint_coordinate", {})
    canopy_status = tuple(str(name) for name in canopy.get("status_logit_fields", {}).values())
    canopy_transfer = tuple(str(name) for name in canopy.get("materialized_transfer_fields", ()))
    named_groups = (
        exact,
        tuple(str(name) for name in payload.get("surface_energy_fields", ())),
        canopy_status,
        canopy_transfer,
    )
    if any(len(group) != len(set(group)) for group in named_groups):
        raise ValueError("mechanistic-output named group contains duplicates")
    policy = payload.get("data_policy", {})
    if any(bool(value) for value in policy.values()):
        raise ValueError("mechanistic-output contract opens a forbidden data path")
    return MechanisticOutputContract(payload=payload, source_path=source)


def build_mechanistic_learned_schema(
    complete_schema: NeuralOutputSchema,
    contract: MechanisticOutputContract,
) -> NeuralOutputSchema:
    """Remove exact heads and expose DOC as an unconstrained residual coordinate."""

    complete = {field.name: field for field in complete_schema.fields}
    declared = set(contract.exact_fields)
    missing = sorted(declared - set(complete))
    if missing:
        raise ValueError(f"mechanistic-output exact fields are absent: {missing}")
    fields = []
    for field in complete_schema.fields:
        if field.name in declared:
            continue
        if field.name == contract.doc_field:
            field = replace(field, constraint="logit", value_representation="direct")
        elif field.name == contract.nroot_coordinate_field:
            field = replace(field, constraint="logit")
        elif field.name in set(contract.canopy_transfer_fields):
            field = replace(field, constraint="logit")
        fields.append(field)
    learned = NeuralOutputSchema(
        fields=tuple(fields),
        labels=complete_schema.labels,
        active_pft_indices=complete_schema.active_pft_indices,
    )
    scalar_count = sum(int(np.prod(field.shape, dtype=np.int64)) for field in learned.fields)
    expected = contract.payload["learned_boundary"]
    if len(learned.fields) != int(expected["field_count"]) or scalar_count != int(expected["scalar_count"]):
        raise ValueError("mechanistic-output learned schema count drift")
    return learned


__all__ = [
    "MechanisticOutputContract",
    "build_mechanistic_learned_schema",
    "load_mechanistic_output_contract",
]
