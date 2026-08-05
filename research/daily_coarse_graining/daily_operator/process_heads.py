"""Constraint parameterizations and the minimal Gate-E1 process head.

The linear head in this module exists only to make the scientific IO boundary
executable and differentiable. It is explicitly not an accepted E2 network.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .process_derivations import (
    NativeForcingFieldLayout,
    derive_carbon_decomposition,
    derive_litter_input,
    integrate_native_precipitation,
)
from .state_features import assemble_local_pft_features
from .types import (
    AssembledDailyOperatorInput,
    CarbonFluxPrediction,
    DailyFluxPrediction,
    DailyOperatorParameters,
    DailyTemperatureContext,
    EnergyFluxPrediction,
    GrossDecompositionPrediction,
    InventoryPrediction,
    PFTProcessPrediction,
    ProcessPrediction,
    RawInventoryPrediction,
    RawThermalPrediction,
    RetainedProcessBoundary,
    SignedFluxPrediction,
    ThermalPrediction,
    WaterFluxPrediction,
)

SECONDS_PER_DAY = 86400.0
PFT_PARAMETER_VCMAX25 = 0
PFT_PARAMETER_MAINT_A = 1
PFT_PARAMETER_MAINT_B = 2
PFT_PARAMETER_MAINT_C = 3
PFT_TRAIT_BARE_SOIL = 0


def _linear(kernel: Any, bias: Any, context: Any) -> Any:
    return jnp.asarray(kernel, dtype=jnp.float64) @ context + jnp.asarray(bias, dtype=jnp.float64)


def _masked_softmax(logits: Any, mask: Any) -> Any:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    logits = jnp.asarray(logits, dtype=jnp.float64)
    masked = jnp.where(mask, logits, -jnp.inf)
    row_has_destination = jnp.any(mask, axis=-1, keepdims=True)
    safe = jnp.where(row_has_destination, masked, jnp.zeros_like(masked))
    shares = jax.nn.softmax(safe, axis=-1)
    return jnp.where(mask & row_has_destination, shares, jnp.zeros_like(shares))


def parameterize_inventory_prediction(
    raw: RawInventoryPrediction,
    *,
    duration_seconds: Any = SECONDS_PER_DAY,
) -> InventoryPrediction:
    """Map raw outputs to a bounded conservative inventory parameterization."""

    duration_days = jnp.asarray(duration_seconds, dtype=jnp.float64) / SECONDS_PER_DAY
    rate = jax.nn.softplus(jnp.asarray(raw.outgoing_rate_logits, dtype=jnp.float64))
    outgoing = -jnp.expm1(-rate * duration_days)
    incoming = jax.nn.softplus(jnp.asarray(raw.external_input_logits, dtype=jnp.float64)) * duration_days
    shares = _masked_softmax(raw.destination_logits, raw.destination_mask)
    has_destination = jnp.any(jnp.asarray(raw.destination_mask, dtype=jnp.bool_), axis=-1)
    external_share = jax.nn.sigmoid(jnp.asarray(raw.external_output_share_logits, dtype=jnp.float64))
    external_share = jnp.where(has_destination, external_share, jnp.ones_like(external_share))
    defined = jnp.asarray(raw.defined, dtype=jnp.bool_)
    return InventoryPrediction(
        outgoing_fraction=jnp.where(defined, outgoing, 0.0),
        external_input_amount=jnp.where(defined, incoming, 0.0),
        external_output_share=jnp.where(defined, external_share, 0.0),
        destination_shares=jnp.where(defined[:, None], shares, 0.0),
        defined=defined,
    )


def parameterize_thermal_prediction(raw: RawThermalPrediction) -> ThermalPrediction:
    """Map raw thermal outputs to declared per-field daily tendency bounds."""

    defined = jnp.asarray(raw.defined, dtype=jnp.bool_)
    normalized = jnp.tanh(jnp.asarray(raw.normalized_tendency_logits, dtype=jnp.float64))
    return ThermalPrediction(
        normalized_tendency=jnp.where(defined, normalized, 0.0),
        bounds=jnp.asarray(raw.bounds, dtype=jnp.float64),
        defined=defined,
    )


def _domain_head(parameters, context, defined) -> InventoryPrediction:
    n_inventory = int(jnp.asarray(parameters.outgoing_bias).shape[0])
    destination = jnp.einsum(
        "ijc,c->ij",
        jnp.asarray(parameters.destination_kernel, dtype=jnp.float64),
        context,
    ) + jnp.asarray(parameters.destination_bias, dtype=jnp.float64)
    destination_mask = ~jnp.eye(n_inventory, dtype=jnp.bool_)
    raw = RawInventoryPrediction(
        outgoing_rate_logits=_linear(parameters.outgoing_kernel, parameters.outgoing_bias, context),
        external_input_logits=_linear(parameters.input_kernel, parameters.input_bias, context),
        external_output_share_logits=_linear(
            parameters.external_share_kernel,
            parameters.external_share_bias,
            context,
        ),
        destination_logits=destination,
        destination_mask=destination_mask,
        defined=defined,
    )
    return parameterize_inventory_prediction(raw)


def _pft_process_head(parameters, context, inputs: AssembledDailyOperatorInput) -> PFTProcessPrediction:
    local = assemble_local_pft_features(inputs.pft)
    kernel = jnp.asarray(parameters.kernel, dtype=jnp.float64)
    bias = jnp.asarray(parameters.bias, dtype=jnp.float64)
    if kernel.ndim != 2 or kernel.shape[0] != 6 or kernel.shape[1] != local.shape[1] + context.shape[0]:
        raise ValueError("minimal PFT head requires six shared outputs and local+global features")
    raw = jnp.concatenate((local, jnp.broadcast_to(context, (local.shape[0], context.shape[0]))), axis=1) @ kernel.T
    raw = raw + bias
    active = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
    fraction = jnp.asarray(inputs.pft.pft_fraction, dtype=jnp.float64)
    traits = jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)
    bare = active & (fraction > 0.0) & (traits[:, PFT_TRAIT_BARE_SOIL] > 0.5)
    vegetation = active & (fraction > 0.0) & ~bare
    parameters_by_pft = jnp.asarray(inputs.pft.pft_parameters, dtype=jnp.float64)
    vcmax25 = parameters_by_pft[:, PFT_PARAMETER_VCMAX25]
    # The environmental temperature proxy is a bounded named head output. The
    # explicit polynomial is the frozen source rule from
    # pft_parameters.f90::config_stomate_pft_parameters, lines 4220-4242.
    temperature_celsius = 40.0 * jnp.tanh(raw[:, 5])
    slope = (
        parameters_by_pft[:, PFT_PARAMETER_MAINT_C]
        + parameters_by_pft[:, PFT_PARAMETER_MAINT_B] * temperature_celsius
        + parameters_by_pft[:, PFT_PARAMETER_MAINT_A] * temperature_celsius**2
    )
    gpp = jax.nn.softplus(raw[:, 0]) * jnp.maximum(vcmax25, 0.0)
    maintenance = jax.nn.softplus(raw[:, 1]) * jnp.maximum(slope, 0.0)
    transpiration = jax.nn.softplus(raw[:, 2])
    wet_evaporation = jax.nn.softplus(raw[:, 3])
    bare_evaporation = jax.nn.softplus(raw[:, 4])
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return PFTProcessPrediction(
        gpp=jnp.where(vegetation, gpp, zero),
        maintenance_respiration=jnp.where(vegetation, maintenance, zero),
        transpiration=jnp.where(vegetation, transpiration, zero),
        wet_canopy_evaporation=jnp.where(vegetation, wet_evaporation, zero),
        bare_soil_evaporation=jnp.where(bare, bare_evaporation, zero),
        vegetation_process_mask=vegetation,
        bare_soil_process_mask=bare,
    )


def _primary_flux_heads(
    parameters,
    context: Any,
    inputs: AssembledDailyOperatorInput,
    pft: PFTProcessPrediction,
    forcing_layout: NativeForcingFieldLayout,
) -> DailyFluxPrediction:
    """Predict only frozen missing labels and compose deterministic owners.

    Owner inventory: ``daily_flux_label_inventory_v1.json``. No process label
    is filled from an unrelated endpoint tendency or generic transfer mirror.
    """

    water_raw = _linear(parameters.water_kernel, parameters.water_bias, context)
    carbon_raw = _linear(parameters.carbon_kernel, parameters.carbon_bias, context)
    energy_raw = _linear(parameters.energy_kernel, parameters.energy_bias, context)
    if water_raw.shape != (10,) or carbon_raw.shape != (13,) or energy_raw.shape != (8,):
        raise ValueError("primary flux heads do not match the frozen Gate-C owner counts")
    water = jax.nn.softplus(water_raw)
    carbon = jax.nn.softplus(carbon_raw)
    energy_components = jax.nn.softplus(energy_raw)
    rain_input, snowfall_input = integrate_native_precipitation(inputs.forcing, forcing_layout)
    npts, n_pft, _, nelements = inputs.day_start_process_state.bm_to_litter.shape
    ndeep = inputs.day_start_process_state.lignin_struc_below.shape[-1]
    poc = GrossDecompositionPrediction(
        ordinary=jnp.full((npts, 3, nelements, n_pft, ndeep), carbon[2], dtype=jnp.float64),
        flooded=jnp.full((npts, 3, nelements, n_pft, ndeep), carbon[3], dtype=jnp.float64),
    )
    doc = GrossDecompositionPrediction(
        ordinary=jnp.full((npts, n_pft, ndeep, 7), carbon[4], dtype=jnp.float64),
        flooded=jnp.full((npts, n_pft, ndeep, 7), carbon[5], dtype=jnp.float64),
    )
    derived = derive_carbon_decomposition(poc, doc, inputs)

    def signed(positive_index: int, negative_index: int) -> SignedFluxPrediction:
        positive = energy_components[positive_index]
        negative = energy_components[negative_index]
        return SignedFluxPrediction(positive=positive, negative=negative, value=positive - negative)

    return DailyFluxPrediction(
        water=WaterFluxPrediction(
            rain_input=rain_input,
            snowfall_input=snowfall_input,
            canopy_precipitation_partition=water[0],
            canopy_to_ground=water[1],
            wet_canopy_evaporation=pft.wet_canopy_evaporation,
            transpiration=pft.transpiration,
            bare_soil_evaporation=pft.bare_soil_evaporation,
            snow_sublimation=water[2],
            flood_evaporation=water[3],
            snow_phase_and_melt_transfer=water[4],
            soil_infiltration=water[5],
            soil_vertical_transfer=water[6],
            runoff=water[7],
            drainage=water[8],
            routing_exchange=water[9],
        ),
        carbon=CarbonFluxPrediction(
            fast_gpp=pft.gpp,
            fast_maintenance_respiration=pft.maintenance_respiration,
            litter_input=derive_litter_input(inputs),
            litter_respiration=carbon[0],
            litter_to_doc=carbon[1],
            poc_gross_decomposition=poc,
            poc_respiration=derived.poc_respiration,
            poc_to_doc=derived.poc_to_doc,
            doc_gross_decomposition=doc,
            doc_to_poc=derived.doc_to_poc,
            doc_respiration=derived.doc_respiration,
            doc_external_input=carbon[6],
            doc_export=carbon[7],
            doc_free_adsorbed_equilibration=carbon[8],
            doc_vertical_water_transport=carbon[9],
            doc_vertical_diffusion=carbon[10],
            cryoturbation_redistribution=carbon[11],
            perma_peat_redistribution=carbon[12],
        ),
        energy=EnergyFluxPrediction(
            net_radiation_integral=signed(0, 1),
            sensible_heat_integral=energy_components[2],
            latent_heat_integral=energy_components[3],
            ground_heat_integral=signed(4, 5),
            phase_change_integral=signed(6, 7),
        ),
    )


def run_minimal_process_heads(
    parameters: DailyOperatorParameters,
    context: Any,
    inputs: AssembledDailyOperatorInput,
    *,
    thermal_bounds: Any,
    forcing_layout: NativeForcingFieldLayout,
) -> ProcessPrediction:
    """Execute the deterministic E1 plumbing head with typed constrained outputs."""

    water_defined = jnp.asarray(inputs.state_defined.water, dtype=jnp.bool_)
    carbon_defined = jnp.asarray(inputs.state_defined.carbon, dtype=jnp.bool_)
    thermal_defined = jnp.asarray(inputs.state_defined.thermal, dtype=jnp.bool_)
    if water_defined.ndim != 1 or carbon_defined.ndim != 1 or thermal_defined.ndim != 1:
        raise ValueError("the minimal E1 head currently expects one unbatched inventory vector per domain")
    water = _domain_head(parameters.water, context, water_defined)
    carbon = _domain_head(parameters.carbon, context, carbon_defined)
    thermal_raw = RawThermalPrediction(
        normalized_tendency_logits=_linear(parameters.thermal.kernel, parameters.thermal.bias, context),
        bounds=thermal_bounds,
        defined=thermal_defined,
    )
    thermal = parameterize_thermal_prediction(thermal_raw)
    pft = _pft_process_head(parameters.pft, context, inputs)
    compact = _linear(parameters.compact.kernel, parameters.compact.bias, context)
    if compact.size < 2:
        raise ValueError("compact head must predict tsurf_daily and at least one tsoil_daily value")
    temperature_context = DailyTemperatureContext(
        tsurf_daily=compact[0],
        tsoil_daily=compact[1:],
    )
    daily_fields = {
        "tsurf_daily": temperature_context.tsurf_daily,
        "tsoil_daily": temperature_context.tsoil_daily,
        "gpp_daily": pft.gpp,
        "resp_maint_part": pft.maintenance_respiration,
        "resp_maint_radia": jnp.sum(pft.maintenance_respiration),
        "flood_root_radia": jnp.zeros_like(pft.maintenance_respiration),
    }
    return ProcessPrediction(
        water_inventory=water,
        carbon_inventory=carbon,
        thermal=thermal,
        pft=pft,
        fluxes=_primary_flux_heads(
            parameters.fluxes,
            context,
            inputs,
            pft,
            forcing_layout,
        ),
        daily_temperature_context=temperature_context,
        retained_boundary=RetainedProcessBoundary(
            fast_state_fields={},
            final_entry_diagnostics={},
            daily_fields=daily_fields,
            ok_leak_result=None,
        ),
    )


PROCESS_LABEL_BINDINGS = {
    "water_inventory": (
        "water.canopy_inventory_endpoints",
        "water.soil_inventory_endpoints",
        "water.snow_inventory_endpoints",
        "water.flood_inventory_endpoints",
    ),
    "carbon_inventory": ("carbon.ok_leak_inventory_endpoints",),
    "thermal": (
        "energy.surface_temperature_tendency",
        "energy.soil_temperature_tendency",
        "energy.snow_thermal_tendency",
    ),
    **{
        f"fluxes.water.{name}": (f"water.{name}",)
        for name in WaterFluxPrediction._fields
    },
    **{
        f"fluxes.carbon.{name}": (f"carbon.{name}",)
        for name in CarbonFluxPrediction._fields
    },
    **{
        f"fluxes.energy.{name}": (f"energy.{name}",)
        for name in EnergyFluxPrediction._fields
    },
    "daily_temperature_context": ("energy.daily_temperature_context",),
}


def classify_differentiability_case(
    inputs: AssembledDailyOperatorInput,
    *,
    threshold_atol: float = 1.0e-8,
) -> dict[str, Any]:
    """Classify masks and process thresholds before interpreting gradients."""

    fraction = jnp.asarray(inputs.pft.pft_fraction)
    active = jnp.asarray(inputs.pft.active_pft_mask)
    bare_trait = jnp.asarray(inputs.pft.pft_traits)[:, PFT_TRAIT_BARE_SOIL]
    threshold_adjacent = (
        jnp.any(jnp.abs(fraction) <= threshold_atol)
        | jnp.any(jnp.abs(bare_trait - 0.5) <= threshold_atol)
    )
    inactive = jnp.any(~active)
    return {
        "smooth_active": ~(inactive | threshold_adjacent),
        "contains_inactive_slots": inactive,
        "threshold_adjacent": threshold_adjacent,
    }
