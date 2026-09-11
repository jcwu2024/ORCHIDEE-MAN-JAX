"""Source-backed applicability masks for learned daily outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp

from research.daily_coarse_graining.daily_pft_interface import (
    DailyPFTInterfaceContract,
    load_daily_pft_interface_contract,
)

from .neural_architecture import NeuralOutputSchema, OutputFieldSpec
from .types import AssembledDailyOperatorInput

GSMEAN_OUTPUT_FIELD = "endpoint.fast_state.diffuco_previous_step_state.gsmean"
BARE_SOIL_TRAIT = "role_bare_soil"


@dataclass(frozen=True)
class GsmeanOutputApplicability:
    """Named trait and output routing for the gsmean applicability rule."""

    field_name: str
    bare_soil_trait_index: int


def build_gsmean_output_applicability(
    contract: DailyPFTInterfaceContract,
) -> GsmeanOutputApplicability:
    """Resolve bare-soil identity by trait name rather than PFT position."""

    matches = tuple(
        channel.index
        for channel in contract.trait_channels
        if channel.name == BARE_SOIL_TRAIT
    )
    if len(matches) != 1:
        raise ValueError(
            f"the PFT interface requires exactly one {BARE_SOIL_TRAIT!r} trait"
        )
    return GsmeanOutputApplicability(
        field_name=GSMEAN_OUTPUT_FIELD,
        bare_soil_trait_index=int(matches[0]),
    )


DEFAULT_GSMEAN_OUTPUT_APPLICABILITY = build_gsmean_output_applicability(
    load_daily_pft_interface_contract()
)


def _field_spec(schema: NeuralOutputSchema, field_name: str) -> OutputFieldSpec:
    matches = tuple(field for field in schema.fields if field.name == field_name)
    if len(matches) != 1:
        raise ValueError(f"learned schema requires exactly one {field_name!r} field")
    field = matches[0]
    if field.pft_axis is None:
        raise ValueError(f"{field_name!r} must declare its PFT axis")
    return field


def apply_gsmean_output_applicability(
    outputs: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    applicability: GsmeanOutputApplicability = DEFAULT_GSMEAN_OUTPUT_APPLICABILITY,
) -> Mapping[str, Any]:
    """Set bare-soil gsmean and its parameter derivative exactly to zero.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_trans_co2``
    initializes ``gsmean`` to zero at line 2289, iterates only over vegetation
    PFTs ``2:nvm`` from line 2299, and assigns vegetation values at line 2885.
    """

    if applicability.field_name not in outputs:
        raise ValueError(f"learned outputs omit {applicability.field_name!r}")
    field = _field_spec(schema, applicability.field_name)
    value = jnp.asarray(outputs[applicability.field_name])
    if value.shape != field.shape:
        raise ValueError(
            f"{applicability.field_name!r} shape {value.shape} does not match {field.shape}"
        )

    traits = jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)
    active = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
    if traits.ndim != 2:
        raise ValueError("pft_traits must have shape (n_pft, n_traits)")
    if active.shape != (traits.shape[0],):
        raise ValueError("active_pft_mask must match the PFT trait axis")
    if not 0 <= applicability.bare_soil_trait_index < traits.shape[1]:
        raise ValueError("bare-soil trait index is outside pft_traits")

    compact_bare = traits[:, applicability.bare_soil_trait_index] > 0.5
    pft_axis = int(field.pft_axis)
    if field.full_pft_axis:
        if len(schema.active_pft_indices) != traits.shape[0]:
            raise ValueError("full-PFT output mapping does not match compact PFT traits")
        if any(
            index < 0 or index >= value.shape[pft_axis]
            for index in schema.active_pft_indices
        ):
            raise ValueError("active PFT indices are outside the full output axis")
        bare = jnp.zeros((value.shape[pft_axis],), dtype=jnp.bool_).at[
            jnp.asarray(schema.active_pft_indices, dtype=jnp.int32)
        ].set(compact_bare)
    else:
        if value.shape[pft_axis] != traits.shape[0]:
            raise ValueError("compact gsmean PFT axis does not match pft_traits")
        bare = compact_bare

    broadcast_shape = [1] * value.ndim
    broadcast_shape[pft_axis] = bare.shape[0]
    corrected = dict(outputs)
    corrected[applicability.field_name] = jnp.where(
        bare.reshape(tuple(broadcast_shape)),
        jnp.zeros_like(value),
        value,
    )
    return corrected


__all__ = [
    "BARE_SOIL_TRAIT",
    "DEFAULT_GSMEAN_OUTPUT_APPLICABILITY",
    "GSMEAN_OUTPUT_FIELD",
    "GsmeanOutputApplicability",
    "apply_gsmean_output_applicability",
    "build_gsmean_output_applicability",
]
