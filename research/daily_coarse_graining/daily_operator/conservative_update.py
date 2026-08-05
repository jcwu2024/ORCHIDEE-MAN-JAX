"""Pure-JAX one-step conservative water, carbon, and thermal update."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .types import (
    AssembledDailyOperatorInput,
    ConservativeDomainResult,
    ConservativeUpdateResult,
    InventoryPrediction,
    ProcessPrediction,
)


def split_signed_inventory(value: Any) -> tuple[Any, Any]:
    """Represent a source-defined signed carry as storage minus debt."""

    array = jnp.asarray(value, dtype=jnp.float64)
    return jnp.maximum(array, 0.0), jnp.maximum(-array, 0.0)


def combine_signed_inventory(storage: Any, debt: Any) -> Any:
    """Recombine nonnegative storage and debt components exactly."""

    return jnp.asarray(storage, dtype=jnp.float64) - jnp.asarray(debt, dtype=jnp.float64)


def apply_conservative_inventory_update(
    start: Any,
    prediction: InventoryPrediction,
) -> ConservativeDomainResult:
    """Apply one bounded inventory transition with exact internal cancellation.

    For each source inventory, one bounded outgoing amount is split between an
    external sink and normalized internal destinations. This construction is
    the inference-side form of the accepted Gate-C inventory identity; it does
    not inspect or encode a Teacher endpoint and never clips the resulting stock.
    """

    start = jnp.asarray(start, dtype=jnp.float64)
    if start.ndim != 1:
        raise ValueError("the E1 conservative updater expects one unbatched inventory vector")
    outgoing_fraction = jnp.asarray(prediction.outgoing_fraction, dtype=jnp.float64)
    external_input = jnp.asarray(prediction.external_input_amount, dtype=jnp.float64)
    external_share = jnp.asarray(prediction.external_output_share, dtype=jnp.float64)
    destination_shares = jnp.asarray(prediction.destination_shares, dtype=jnp.float64)
    defined = jnp.asarray(prediction.defined, dtype=jnp.bool_)
    n_inventory = start.shape[0]
    if outgoing_fraction.shape != (n_inventory,) or external_input.shape != (n_inventory,):
        raise ValueError("inventory prediction does not match the start vector")
    if external_share.shape != (n_inventory,) or destination_shares.shape != (n_inventory, n_inventory):
        raise ValueError("inventory share prediction has incompatible axes")
    safe_start = jnp.where(defined, start, jnp.zeros_like(start))
    outgoing_amount = safe_start * outgoing_fraction
    external_output = outgoing_amount * external_share
    internal_transfer = (
        outgoing_amount[:, None]
        * (1.0 - external_share[:, None])
        * destination_shares
    )
    incoming_internal = jnp.sum(internal_transfer, axis=0)
    next_inventory = safe_start - outgoing_amount + external_input + incoming_internal
    next_inventory = jnp.where(defined, next_inventory, start)
    initial_total = jnp.sum(jnp.where(defined, start, 0.0))
    final_total = jnp.sum(jnp.where(defined, next_inventory, 0.0))
    budget_residual = final_total - initial_total - jnp.sum(external_input) + jnp.sum(external_output)
    return ConservativeDomainResult(
        inventory=next_inventory,
        external_input=external_input,
        external_output=external_output,
        internal_transfer=internal_transfer,
        budget_residual=budget_residual,
        defined=defined,
    )


def apply_conservative_update(
    inputs: AssembledDailyOperatorInput,
    prediction: ProcessPrediction,
) -> ConservativeUpdateResult:
    """Apply one fast-process transition and preserve exact discrete state."""

    water = apply_conservative_inventory_update(inputs.state.water, prediction.water_inventory)
    carbon = apply_conservative_inventory_update(inputs.state.carbon, prediction.carbon_inventory)
    thermal_start = jnp.asarray(inputs.state.thermal, dtype=jnp.float64)
    thermal = thermal_start + prediction.thermal.normalized_tendency * prediction.thermal.bounds
    thermal = jnp.where(prediction.thermal.defined, thermal, thermal_start)
    next_state = inputs.state._replace(
        water=water.inventory,
        carbon=carbon.inventory,
        thermal=thermal,
    )
    return ConservativeUpdateResult(
        state=next_state,
        state_defined=inputs.state_defined,
        discrete_state=inputs.discrete_state,
        water=water,
        carbon=carbon,
        thermal_tendency=thermal - thermal_start,
    )


def validate_conservative_result(result: ConservativeUpdateResult, *, atol: float = 2.0e-12) -> None:
    """Host-side E1 admission checks for bounds, nonnegativity, and budgets."""

    for name, domain in {"water": result.water, "carbon": result.carbon}.items():
        inventory = np.asarray(domain.inventory)
        defined = np.asarray(domain.defined)
        if np.any(inventory[defined] < -atol):
            raise ValueError(f"{name} inventory became negative without a legal signed decomposition")
        if abs(float(np.asarray(domain.budget_residual))) > atol:
            raise ValueError(f"{name} conservative budget residual exceeds tolerance")
