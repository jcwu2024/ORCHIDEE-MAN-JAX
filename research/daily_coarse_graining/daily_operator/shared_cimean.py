"""Source-owned materialization for the shared ``cimean`` endpoint.

Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/diffuco.f90``
``diffuco_trans_co2`` lines 2284-2328 initializes ``cimean`` to ``Ca``;
lines 2341-2347 and 2884-2895 overwrite only active vegetation points.
The paper PFT1/bare-soil coordinate therefore has the exact endpoint
``cimean = Ca``.  ``Ca`` is supplied by the annual ``ATM_CO2`` condition,
whose JAX route is documented in ``dim2_driver.f90`` lines 708-719,
1007-1010, and 1111-1117.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from research.daily_coarse_graining.daily_markov_contract import DailyMarkovContract

from .neural_architecture import NeuralOutputSchema, OutputFieldSpec
from .output_applicability import DEFAULT_GSMEAN_OUTPUT_APPLICABILITY
from .types import AssembledDailyOperatorInput

CIMEAN_FIELD = "endpoint.fast_state.diffuco_previous_step_state.cimean"
BARE_SOIL_TRAIT = "role_bare_soil"


def _cimean_field(schema: NeuralOutputSchema) -> OutputFieldSpec:
    matches = tuple(field for field in schema.fields if field.name == CIMEAN_FIELD)
    if len(matches) != 1:
        raise ValueError(f"schema requires exactly one {CIMEAN_FIELD!r} field")
    field = matches[0]
    if field.pft_axis is None:
        raise ValueError(f"{CIMEAN_FIELD!r} must declare a PFT axis")
    return field


def _annual_ca(
    inputs: AssembledDailyOperatorInput,
    contract: DailyMarkovContract,
    field: OutputFieldSpec,
) -> Any:
    """Extract annual ``annual_co2_ppm`` by its contract leaf, then broadcast."""

    leaves = contract.static_conditions.annual_condition_leaves
    matches = tuple(leaf for leaf in leaves if str(leaf.name) == "annual_co2_ppm")
    if len(matches) != 1:
        raise ValueError("contract requires exactly one annual_co2_ppm condition leaf")
    leaf = matches[0]

    canonical = inputs.canonical_input
    annual = jnp.asarray(canonical.annual_conditions, dtype=jnp.float64).reshape(-1)
    start, stop = int(leaf.start), int(leaf.stop)
    if start < 0 or stop <= start or stop > annual.shape[0]:
        raise ValueError("annual_co2_ppm leaf is outside annual_conditions")
    ca = annual[start:stop].reshape(tuple(leaf.shape))
    pft_axis = int(field.pft_axis)
    expected_without_pft = field.shape[:pft_axis] + field.shape[pft_axis + 1 :]
    if tuple(ca.shape) == tuple(expected_without_pft):
        ca = jnp.expand_dims(ca, axis=pft_axis)
    elif ca.size == 1:
        ca = jnp.reshape(ca, (1,) * len(field.shape))
    try:
        return jnp.broadcast_to(ca, field.shape)
    except ValueError as error:
        raise ValueError(
            f"annual_co2_ppm shape {tuple(ca.shape)} cannot broadcast to cimean "
            f"shape {field.shape}"
        ) from error


def _mask_for_field(active_masks: Mapping[str, Any], field: OutputFieldSpec, shape: tuple[int, ...]) -> Any:
    if field.name not in active_masks:
        raise ValueError(f"active_masks omit {field.name!r}")
    mask = jnp.asarray(active_masks[field.name], dtype=jnp.bool_)
    if mask.shape == shape:
        return mask
    pft_axis = int(field.pft_axis)
    if mask.ndim == 1 and mask.shape[0] == shape[pft_axis]:
        view = [1] * len(shape)
        view[pft_axis] = mask.shape[0]
        return jnp.broadcast_to(mask.reshape(tuple(view)), shape)
    try:
        return jnp.broadcast_to(mask, shape)
    except ValueError as error:
        raise ValueError(
            f"active cimean mask shape {mask.shape} cannot broadcast to {shape}"
        ) from error


def _pft_mask(
    values: Any,
    field: OutputFieldSpec,
    schema: NeuralOutputSchema,
) -> Any:
    traits = jnp.asarray(values, dtype=jnp.float64)
    if traits.ndim != 2:
        raise ValueError("pft_traits must have shape (n_pft, n_traits)")
    trait_index = int(DEFAULT_GSMEAN_OUTPUT_APPLICABILITY.bare_soil_trait_index)
    bare_compact = traits[:, trait_index] > 0.5
    pft_axis = int(field.pft_axis)
    extent = field.shape[pft_axis]
    if field.full_pft_axis:
        indices = tuple(int(index) for index in schema.active_pft_indices)
        if len(indices) != traits.shape[0] or any(index < 0 or index >= extent for index in indices):
            raise ValueError("full-PFT cimean mapping does not match schema and traits")
        bare = jnp.zeros((extent,), dtype=jnp.bool_).at[jnp.asarray(indices)].set(bare_compact)
    else:
        if extent != traits.shape[0]:
            raise ValueError("compact cimean PFT axis does not match pft_traits")
        bare = bare_compact
    shape = [1] * len(field.shape)
    shape[pft_axis] = extent
    return bare.reshape(tuple(shape))


def apply_shared_cimean(
    outputs: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    contract: DailyMarkovContract,
    active_masks: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the exact bare-soil ``cimean = Ca`` source owner.

    ``outputs`` are physical endpoints after the normal day-start delta
    decoder.  The current-day endpoint is replaced with annual ``Ca`` only on
    active bare-soil coordinates.  The day-start state is deliberately not
    used here: a prior-year value at a cross-year handoff must remain part of
    the caller's ``Ca - day_start`` training representation.
    """

    field = _cimean_field(schema)
    if CIMEAN_FIELD not in outputs:
        raise ValueError(f"outputs omit {CIMEAN_FIELD!r}")
    value = jnp.asarray(outputs[CIMEAN_FIELD], dtype=jnp.float64)
    if value.shape != field.shape:
        raise ValueError(f"cimean output shape {value.shape} != schema shape {field.shape}")

    canonical = inputs.canonical_input
    active = _mask_for_field(active_masks, field, field.shape)
    bare = _pft_mask(canonical.pft.pft_traits, field, schema)
    ca = _annual_ca(inputs, contract, field)
    corrected = jnp.where(active, jnp.where(bare, ca, value), 0.0)
    result = dict(outputs)
    result[CIMEAN_FIELD] = corrected
    return result


__all__ = ["BARE_SOIL_TRAIT", "CIMEAN_FIELD", "apply_shared_cimean"]
