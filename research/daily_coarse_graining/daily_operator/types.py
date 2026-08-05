"""Stable PyTree types for the Gate-E1 true-daily operator boundary."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class NativeForcingInput(NamedTuple):
    """Variable-record native forcing with explicit interval metadata."""

    values: Any
    record_time_seconds: Any
    duration_seconds: Any
    predecessor_mask: Any
    record_mask: Any


class PFTAxisInput(NamedTuple):
    """The five numerical arrays admitted by the frozen Gate-B2 contract."""

    pft_state: Any
    pft_parameters: Any
    pft_traits: Any
    pft_fraction: Any
    active_pft_mask: Any


class StateFeatureGroups(NamedTuple):
    """Continuous state grouped without removing scientific axes.

    Inventory transitions operate on the final axis of ``water`` and
    ``carbon``. All leading axes, including PFT or soil axes, remain explicit.
    """

    canopy: Any
    soil: Any
    snow: Any
    water: Any
    carbon: Any
    thermal: Any
    other: Any


class StateDefinedMasks(NamedTuple):
    """Exact defined-status masks matching :class:`StateFeatureGroups`."""

    canopy: Any
    soil: Any
    snow: Any
    water: Any
    carbon: Any
    thermal: Any
    other: Any


class DayStartProcessState(NamedTuple):
    """Named views of day-start state required by exact in-graph formulas."""

    bm_to_litter: Any
    turnover_daily: Any
    lignin_struc_above: Any
    lignin_struc_below: Any


class ProcessStaticConditions(NamedTuple):
    """Named parameter/static inputs for retained source-backed formulas."""

    rprof: Any
    z_soil: Any
    litterfrac: Any
    cue: Any
    frac_carb: Any
    sro_bottom: Any


class StaticRegistryInput(NamedTuple):
    """Single-owner non-PFT static registry consumed by the assembler.

    ``context_features`` is the frozen named landpoint-static vector. Exact
    process views are derived from this vector plus the two source-owned
    arrays below; callers never provide ``ProcessStaticConditions`` directly.
    """

    context_features: Any
    rprof: Any
    z_soil: Any


class DailyOperatorInput(NamedTuple):
    """Only caller-owned inference inputs; all overlapping views are absent."""

    continuous_state: Any
    discrete_state: Mapping[str, Any]
    forcing: NativeForcingInput
    pft: PFTAxisInput
    static_registry: StaticRegistryInput
    annual_conditions: Any
    year: Any
    day_index: Any


class AssembledDailyOperatorInput(NamedTuple):
    """Assembler-owned views used inside the staged daily transition."""

    state: StateFeatureGroups
    state_defined: StateDefinedMasks
    discrete_state: Mapping[str, Any]
    forcing: NativeForcingInput
    pft: PFTAxisInput
    day_start_process_state: DayStartProcessState
    process_static: ProcessStaticConditions
    static_conditions: Any
    annual_conditions: Any
    year: Any
    day_index: Any
    canonical_input: DailyOperatorInput


class ForcingEncoderParameters(NamedTuple):
    """Parameters for the minimal order-sensitive native-forcing encoder."""

    value_scale: Any
    input_kernel: Any
    recurrent_kernel: Any
    bias: Any


class DomainHeadParameters(NamedTuple):
    """Raw linear parameters for one conservative inventory domain."""

    outgoing_kernel: Any
    outgoing_bias: Any
    input_kernel: Any
    input_bias: Any
    external_share_kernel: Any
    external_share_bias: Any
    destination_kernel: Any
    destination_bias: Any


class ThermalHeadParameters(NamedTuple):
    kernel: Any
    bias: Any


class FluxHeadParameters(NamedTuple):
    water_kernel: Any
    water_bias: Any
    carbon_kernel: Any
    carbon_bias: Any
    energy_kernel: Any
    energy_bias: Any


class PFTHeadParameters(NamedTuple):
    kernel: Any
    bias: Any


class CompactHeadParameters(NamedTuple):
    kernel: Any
    bias: Any


class DailyOperatorParameters(NamedTuple):
    """Placeholder E1 parameters, deliberately not a promoted architecture."""

    forcing: ForcingEncoderParameters
    water: DomainHeadParameters
    carbon: DomainHeadParameters
    thermal: ThermalHeadParameters
    fluxes: FluxHeadParameters
    pft: PFTHeadParameters
    compact: CompactHeadParameters


class RawInventoryPrediction(NamedTuple):
    outgoing_rate_logits: Any
    external_input_logits: Any
    external_output_share_logits: Any
    destination_logits: Any
    destination_mask: Any
    defined: Any


class InventoryPrediction(NamedTuple):
    outgoing_fraction: Any
    external_input_amount: Any
    external_output_share: Any
    destination_shares: Any
    defined: Any


class RawThermalPrediction(NamedTuple):
    normalized_tendency_logits: Any
    bounds: Any
    defined: Any


class ThermalPrediction(NamedTuple):
    normalized_tendency: Any
    bounds: Any
    defined: Any


class PFTProcessPrediction(NamedTuple):
    """Primary per-PFT fast-process outputs; weights are shared by slot."""

    gpp: Any
    maintenance_respiration: Any
    transpiration: Any
    wet_canopy_evaporation: Any
    bare_soil_evaporation: Any
    vegetation_process_mask: Any
    bare_soil_process_mask: Any


class LitterInputPrediction(NamedTuple):
    above: Any
    below: Any


class GrossDecompositionPrediction(NamedTuple):
    ordinary: Any
    flooded: Any


class DailyTemperatureContext(NamedTuple):
    tsurf_daily: Any
    tsoil_daily: Any


class WaterFluxPrediction(NamedTuple):
    rain_input: Any
    snowfall_input: Any
    canopy_precipitation_partition: Any
    canopy_to_ground: Any
    wet_canopy_evaporation: Any
    transpiration: Any
    bare_soil_evaporation: Any
    snow_sublimation: Any
    flood_evaporation: Any
    snow_phase_and_melt_transfer: Any
    soil_infiltration: Any
    soil_vertical_transfer: Any
    runoff: Any
    drainage: Any
    routing_exchange: Any


class CarbonFluxPrediction(NamedTuple):
    fast_gpp: Any
    fast_maintenance_respiration: Any
    litter_input: Any
    litter_respiration: Any
    litter_to_doc: Any
    poc_gross_decomposition: Any
    poc_respiration: Any
    poc_to_doc: Any
    doc_gross_decomposition: Any
    doc_to_poc: Any
    doc_respiration: Any
    doc_external_input: Any
    doc_export: Any
    doc_free_adsorbed_equilibration: Any
    doc_vertical_water_transport: Any
    doc_vertical_diffusion: Any
    cryoturbation_redistribution: Any
    perma_peat_redistribution: Any


class SignedFluxPrediction(NamedTuple):
    positive: Any
    negative: Any
    value: Any


class EnergyFluxPrediction(NamedTuple):
    net_radiation_integral: SignedFluxPrediction
    sensible_heat_integral: Any
    latent_heat_integral: Any
    ground_heat_integral: SignedFluxPrediction
    phase_change_integral: SignedFluxPrediction


class DailyFluxPrediction(NamedTuple):
    water: WaterFluxPrediction
    carbon: CarbonFluxPrediction
    energy: EnergyFluxPrediction


class RetainedProcessBoundary(NamedTuple):
    """Typed fast outputs consumed by the retained exact daily owner."""

    fast_state_fields: Mapping[str, Mapping[str, Any]]
    final_entry_diagnostics: Mapping[str, Any]
    daily_fields: Mapping[str, Any]
    ok_leak_result: Any


class ProcessPrediction(NamedTuple):
    water_inventory: InventoryPrediction
    carbon_inventory: InventoryPrediction
    thermal: ThermalPrediction
    pft: PFTProcessPrediction
    fluxes: DailyFluxPrediction
    daily_temperature_context: DailyTemperatureContext
    retained_boundary: RetainedProcessBoundary


class ConservativeDomainResult(NamedTuple):
    inventory: Any
    external_input: Any
    external_output: Any
    internal_transfer: Any
    budget_residual: Any
    defined: Any


class ConservativeUpdateResult(NamedTuple):
    state: StateFeatureGroups
    state_defined: StateDefinedMasks
    discrete_state: Mapping[str, Any]
    water: ConservativeDomainResult
    carbon: ConservativeDomainResult
    thermal_tendency: Any


class RetainedTailInput(NamedTuple):
    fast_update: ConservativeUpdateResult
    process_prediction: ProcessPrediction
    canonical_input: DailyOperatorInput
    pft_parameters: Any
    pft_traits: Any
    static_conditions: Any
    annual_conditions: Any
    year: Any
    day_index: Any


class RetainedTailResult(NamedTuple):
    state: StateFeatureGroups
    state_defined: StateDefinedMasks
    discrete_state: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


class DailyOperatorResult(NamedTuple):
    prediction: ProcessPrediction
    fast_update: ConservativeUpdateResult
    next_state: StateFeatureGroups
    next_state_defined: StateDefinedMasks
    next_discrete_state: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
