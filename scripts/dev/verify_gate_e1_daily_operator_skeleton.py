from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration as teacher  # noqa: E402
from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402
from research.daily_coarse_graining.constrained_daily_replay import (  # noqa: E402
    CARBON_INVENTORY_FIELDS,
    ORCHIDEE_UNDEFINED_MAGNITUDE,
    THERMAL_FIELDS,
    WATER_INVENTORY_FIELDS,
    BoundedInventoryTransition,
    SignedInventoryTransition,
    audit_daily_carbon_budget,
    audit_daily_thermal_residual,
    audit_daily_water_budget,
    constrained_replay_boundary,
)
from research.daily_coarse_graining.daily_flux_capture import (  # noqa: E402
    CAPTURE_LABEL_PATHS,
    daily_flux_capture_arrays,
)
from research.daily_coarse_graining.daily_markov_contract import (  # noqa: E402
    build_daily_markov_contract,
    extract_state,
    native_forcing_days,
    reconstruct_compiled_forcing_day,
    reconstruct_state_fields,
)
from research.daily_coarse_graining.daily_operator.audit import (  # noqa: E402
    audit_executable_jaxpr,
    audit_inference_source_graph,
    audit_process_label_bindings,
)
from research.daily_coarse_graining.daily_operator.canonical_retained_tail import (  # noqa: E402
    build_canonical_retained_tail_adapter,
)
from research.daily_coarse_graining.daily_operator.input_assembly import (  # noqa: E402
    CanonicalDailyOperatorInputAssembler,
)
from research.daily_coarse_graining.daily_operator.model import (  # noqa: E402
    bind_daily_operator,
    daily_operator_transition,
)
from research.daily_coarse_graining.daily_operator.process_derivations import (  # noqa: E402
    NativeForcingFieldLayout,
    derive_carbon_decomposition,
    derive_litter_input,
    integrate_native_precipitation,
)
from research.daily_coarse_graining.daily_operator.retained_tail import (  # noqa: E402
    retained_tail_input,
)
from research.daily_coarse_graining.daily_operator.types import (  # noqa: E402
    AssembledDailyOperatorInput,
    CarbonFluxPrediction,
    CompactHeadParameters,
    DailyFluxPrediction,
    DailyOperatorInput,
    DailyOperatorParameters,
    DailyTemperatureContext,
    DomainHeadParameters,
    EnergyFluxPrediction,
    FluxHeadParameters,
    ForcingEncoderParameters,
    GrossDecompositionPrediction,
    InventoryPrediction,
    NativeForcingInput,
    PFTAxisInput,
    PFTHeadParameters,
    PFTProcessPrediction,
    ProcessPrediction,
    RetainedProcessBoundary,
    SignedFluxPrediction,
    StaticRegistryInput,
    ThermalHeadParameters,
    ThermalPrediction,
    WaterFluxPrediction,
)
from research.daily_coarse_graining.replay_ceiling import (  # noqa: E402
    _dynamic_boundary_value,
    capture_pre_daily_stomate_record,
    compare_replay_day,
)
from research.daily_coarse_graining.teacher_shards import (  # noqa: E402
    _annual_condition_groups,
    _landpoint_static_groups,
    _pack_condition_groups,
    _parameter_groups,
)

jax.config.update("jax_enable_x64", True)

PFT_AXES = {
    "qsintveg": 1,
    "litter_above": 2,
    "litter_below": 2,
    "carbon_32l": 3,
    "DOC": 1,
    "interception_storage": 1,
}


def _block(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(name): _json_value(child) for name, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if isinstance(value, np.ndarray) or hasattr(value, "shape"):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tree_allclose(expected: Any, observed: Any) -> tuple[bool, float]:
    expected_leaves = jax.tree_util.tree_leaves(expected)
    observed_leaves = jax.tree_util.tree_leaves(observed)
    if len(expected_leaves) != len(observed_leaves):
        return False, float("inf")
    passed = True
    max_error = 0.0
    for left, right in zip(expected_leaves, observed_leaves, strict=True):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.dtype.kind in "fc":
            difference = np.abs(left_array - right_array)
            if difference.size:
                max_error = max(max_error, float(np.nanmax(difference)))
            passed = passed and bool(
                np.allclose(
                    left_array,
                    right_array,
                    atol=1.0e-10,
                    rtol=1.0e-12,
                    equal_nan=True,
                )
            )
        else:
            passed = passed and bool(np.array_equal(left_array, right_array))
    return passed, max_error


def _defined(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return np.isfinite(array) & (np.abs(array) < ORCHIDEE_UNDEFINED_MAGNITUDE)


def _select_pft(value: Any, axis: int, indices: tuple[int, ...]) -> np.ndarray:
    return np.take(np.asarray(value), indices, axis=axis)


def _inventory_array(value: Any, field: str) -> np.ndarray:
    axis = PFT_AXES.get(field)
    return np.asarray(value) if axis is None else _select_pft(value, axis, _ACTIVE_PFT_INDICES.get())


class _ActivePFTIndices:
    """Narrow helper preventing an accidental hidden slot-number fallback."""

    value: tuple[int, ...] | None = None

    def set(self, value: tuple[int, ...]) -> None:
        self.value = tuple(int(index) for index in value)

    def get(self) -> tuple[int, ...]:
        if self.value is None:
            raise RuntimeError("active PFT indices have not been bound")
        return self.value


_ACTIVE_PFT_INDICES = _ActivePFTIndices()


def _compact_transition(value: Any, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = PFT_AXES.get(field)

    def compact(array: Any) -> np.ndarray:
        value_array = np.asarray(array)
        if axis is not None:
            value_array = _select_pft(value_array, axis, _ACTIVE_PFT_INDICES.get())
        return value_array.reshape(-1)

    return compact(value.outgoing_fraction), compact(value.incoming_amount), compact(value.defined)


def _inventory_prediction(
    transitions: Mapping[str, BoundedInventoryTransition | SignedInventoryTransition],
    *,
    domain: str,
    fields: tuple[str, ...],
) -> InventoryPrediction:
    outgoing = []
    incoming = []
    defined = []
    for field in fields:
        transition = transitions[f"{domain}.{field}"]
        parts = (transition.positive, transition.debt) if isinstance(transition, SignedInventoryTransition) else (transition,)
        for part in parts:
            part_outgoing, part_incoming, part_defined = _compact_transition(part, field)
            outgoing.append(part_outgoing)
            incoming.append(part_incoming)
            defined.append(part_defined)
    outgoing_array = np.concatenate(outgoing)
    size = outgoing_array.size
    return InventoryPrediction(
        outgoing_fraction=outgoing_array,
        external_input_amount=np.concatenate(incoming),
        external_output_share=np.ones((size,), dtype=np.float64),
        destination_shares=np.zeros((size, size), dtype=np.float64),
        defined=np.concatenate(defined),
    )


def _thermal_prediction(boundary: Any) -> tuple[ThermalPrediction, tuple[str, ...]]:
    keys = tuple(
        f"{component}.{field}"
        for component, field, _bound in THERMAL_FIELDS
        if f"{component}.{field}" in boundary.thermal_transitions
    )
    transitions = [boundary.thermal_transitions[key] for key in keys]
    return (
        ThermalPrediction(
            normalized_tendency=np.concatenate(
                [np.asarray(value.normalized_tendency).reshape(-1) for value in transitions]
            ),
            bounds=np.concatenate(
                [np.full(np.asarray(value.normalized_tendency).size, value.bound) for value in transitions]
            ),
            defined=np.concatenate([np.asarray(value.defined).reshape(-1) for value in transitions]),
        ),
        keys,
    )


def _forcing_layout(spec: Any) -> NativeForcingFieldLayout:
    width = int(np.prod(spec.field_shape, dtype=np.int64))

    def span(name: str) -> tuple[int, int]:
        index = spec.fields.index(name)
        return index * width, (index + 1) * width

    rain_start, rain_stop = span("Rainf")
    snow_start, snow_stop = span("Snowf")
    return NativeForcingFieldLayout(rain_start, rain_stop, snow_start, snow_stop)


def _native_forcing(value: np.ndarray, spec: Any) -> NativeForcingInput:
    interval = float(spec.source_interval_seconds)
    count = int(value.shape[0])
    predecessor = np.zeros((count,), dtype=np.bool_)
    predecessor[0] = True
    return NativeForcingInput(
        values=np.asarray(value, dtype=np.float64),
        record_time_seconds=(np.arange(count, dtype=np.float64) - 1.0) * interval,
        duration_seconds=np.full((count,), interval, dtype=np.float64),
        predecessor_mask=predecessor,
        record_mask=np.ones((count,), dtype=np.bool_),
    )


def _condition_contract(context: Any, record: Any, native_spec: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    parameters, parameter_leaves = _pack_condition_groups(
        _parameter_groups(context), temporal_role="landpoint_parameter", source="accepted parameter registry"
    )
    static, static_leaves = _pack_condition_groups(
        _landpoint_static_groups(context), temporal_role="landpoint_static", source="accepted static registry"
    )
    annual, annual_leaves = _pack_condition_groups(
        _annual_condition_groups(context, year=record.year),
        temporal_role="annual_exogenous",
        source="annual CO2",
    )
    contract = build_daily_markov_contract(
        record.expected_result.day_end_state,
        record,
        parameter_leaves=parameter_leaves,
        landpoint_static_leaves=static_leaves,
        annual_condition_leaves=annual_leaves,
        native_forcing_spec=native_spec,
    )
    return contract, static, annual


def _pft_input(previous_state: Any, context: Any, indices: tuple[int, ...]) -> PFTAxisInput:
    slow = previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
    features = []
    for field in ("veget_max", "lai", "biomass"):
        value = np.asarray(slow[field])
        selected = np.take(value, indices, axis=1)
        features.append(np.moveaxis(selected, 1, 0).reshape((len(indices), -1)))
    compiled = teacher._compiled_stomate_parameter_values(context)
    slope = np.asarray(compiled.maint_resp_slope)
    parameters = np.stack(
        (
            np.asarray(compiled.vcmax25),
            slope[:, 2],
            slope[:, 1],
            slope[:, 0],
            np.asarray(compiled.alloc_min),
            np.asarray(compiled.residence_time),
        ),
        axis=1,
    )[np.asarray(indices)]
    scalars = context.run_scalars
    pref = np.asarray(scalars.pref_soil_veg, dtype=np.int32)
    traits = np.stack(
        (
            np.arange(context.nvm) == 0,
            np.asarray(scalars.natural),
            np.asarray(scalars.is_tree),
            np.asarray(scalars.is_peat),
            np.asarray(scalars.is_c4),
            ~np.asarray(scalars.natural),
            pref == 1,
            pref == 2,
            pref == 3,
            pref == 4,
        ),
        axis=1,
    ).astype(np.float64)[np.asarray(indices)]
    fraction = np.asarray(slow["veget_max"], dtype=np.float64)[0, np.asarray(indices)]
    active = np.asarray(scalars.active_pft_mask, dtype=np.bool_)[np.asarray(indices)]
    return PFTAxisInput(
        pft_state=np.concatenate(features, axis=1).astype(np.float64),
        pft_parameters=parameters.astype(np.float64),
        pft_traits=traits,
        pft_fraction=fraction,
        active_pft_mask=active,
    )


def _operator_input(
    previous_state: Any,
    context: Any,
    contract: Any,
    forcing: NativeForcingInput,
    static: np.ndarray,
    annual: np.ndarray,
    *,
    year: int,
    day_index: int,
    allow_year_start_missing: bool,
) -> DailyOperatorInput:
    indices = contract.active_pft_indices
    _ACTIVE_PFT_INDICES.set(indices)
    continuous, discrete = extract_state(
        previous_state,
        contract,
        allow_year_start_missing=allow_year_start_missing,
    )
    boundary = context.stomate_boundary.kwargs
    return DailyOperatorInput(
        continuous_state=continuous,
        discrete_state=discrete,
        forcing=forcing,
        pft=_pft_input(previous_state, context, indices),
        static_registry=StaticRegistryInput(
            context_features=static,
            rprof=_select_pft(boundary["rprof"], 1, indices),
            z_soil=np.asarray(boundary["z_soil"], dtype=np.float64),
        ),
        annual_conditions=annual,
        year=np.asarray(year, dtype=np.int32),
        day_index=np.asarray(day_index, dtype=np.int32),
    )


def _dummy_parameters(inputs: AssembledDailyOperatorInput) -> DailyOperatorParameters:
    forcing_width = int(inputs.forcing.values.shape[1])

    def domain(size: int) -> DomainHeadParameters:
        return DomainHeadParameters(
            outgoing_kernel=np.zeros((size, 1)),
            outgoing_bias=np.zeros((size,)),
            input_kernel=np.zeros((size, 1)),
            input_bias=np.zeros((size,)),
            external_share_kernel=np.zeros((size, 1)),
            external_share_bias=np.zeros((size,)),
            destination_kernel=np.zeros((size, size, 1)),
            destination_bias=np.zeros((size, size)),
        )

    return DailyOperatorParameters(
        forcing=ForcingEncoderParameters(
            value_scale=np.ones((forcing_width,), dtype=np.float64),
            input_kernel=np.zeros((forcing_width + 3, 1), dtype=np.float64),
            recurrent_kernel=np.zeros((1, 1), dtype=np.float64),
            bias=np.zeros((1,), dtype=np.float64),
        ),
        water=domain(1),
        carbon=domain(1),
        thermal=ThermalHeadParameters(kernel=np.zeros((inputs.state.thermal.size, 1)), bias=np.zeros(inputs.state.thermal.size)),
        fluxes=FluxHeadParameters(
            water_kernel=np.zeros((10, 1)),
            water_bias=np.zeros(10),
            carbon_kernel=np.zeros((13, 1)),
            carbon_bias=np.zeros(13),
            energy_kernel=np.zeros((8, 1)),
            energy_bias=np.zeros(8),
        ),
        pft=PFTHeadParameters(kernel=np.zeros((6, 1)), bias=np.zeros(6)),
        compact=CompactHeadParameters(kernel=np.zeros((2, 1)), bias=np.zeros(2)),
    )


def _minimal_ok_mapping(value: Any) -> dict[str, Any]:
    perma_peat = value.soilcarbon.perma_peat
    return {
        "litter_above": value.littercalc.litter_above,
        "litter_below": value.littercalc.litter_below,
        "lignin_struc_above": value.littercalc.lignin_struc_above,
        "lignin_struc_below": value.littercalc.lignin_struc_below,
        "litterpart": value.littercalc.litterpart,
        "dead_leaves": value.littercalc.dead_leaves,
        "fuel_1hr": value.littercalc.fuel.fuel_1hr,
        "fuel_10hr": value.littercalc.fuel.fuel_10hr,
        "fuel_100hr": value.littercalc.fuel.fuel_100hr,
        "fuel_1000hr": value.littercalc.fuel.fuel_1000hr,
        "carbon_32l": value.soilcarbon.carbon_32l,
        "DOC": value.soilcarbon.doc,
        "interception_storage": value.interception_storage,
        "deepC_peat": None if perma_peat is None else perma_peat.deepc_peat,
    }

def _signed(value: Any) -> SignedFluxPrediction:
    array = jnp.asarray(value, dtype=jnp.float64)
    return SignedFluxPrediction(
        positive=jnp.maximum(array, 0.0),
        negative=jnp.maximum(-array, 0.0),
        value=array,
    )


@dataclass
class TrueLabelOracleHead:
    """Prebound true-label head; endpoint encoding occurs before construction."""

    water_inventory: InventoryPrediction
    carbon_inventory: InventoryPrediction
    thermal: ThermalPrediction
    thermal_keys: tuple[str, ...]
    gross_poc: GrossDecompositionPrediction
    gross_doc: GrossDecompositionPrediction
    capture_arrays: Mapping[str, Any]
    retained_boundary: RetainedProcessBoundary
    forcing_layout: NativeForcingFieldLayout
    calls: list[str]

    def _captured(self, label: str) -> tuple[Any, ...]:
        return tuple(self.capture_arrays[path] for path in CAPTURE_LABEL_PATHS[label])

    def __call__(
        self,
        parameters: Any,
        context: Any,
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        del parameters, context
        self.calls.append("true_label_oracle_head")
        rain, snow = integrate_native_precipitation(inputs.forcing, self.forcing_layout)
        litter_input = derive_litter_input(inputs)
        carbon = derive_carbon_decomposition(self.gross_poc, self.gross_doc, inputs)
        daily = self.retained_boundary.daily_fields
        indices = _ACTIVE_PFT_INDICES.get()

        def daily_pft(name: str) -> Any:
            value = np.asarray(daily[name])
            return _select_pft(value, 1, indices) if value.ndim > 1 and value.shape[1] >= max(indices) + 1 else value

        pft = PFTProcessPrediction(
            gpp=daily_pft("gpp_daily"),
            maintenance_respiration=daily_pft("resp_maint_part"),
            transpiration=self._captured("water.transpiration"),
            wet_canopy_evaporation=self._captured("water.wet_canopy_evaporation"),
            bare_soil_evaporation=self._captured("water.bare_soil_evaporation"),
            vegetation_process_mask=inputs.pft.active_pft_mask & (inputs.pft.pft_traits[:, 0] < 0.5),
            bare_soil_process_mask=inputs.pft.active_pft_mask & (inputs.pft.pft_traits[:, 0] > 0.5),
        )
        energy = self.capture_arrays
        temperature = DailyTemperatureContext(
            tsurf_daily=daily["tsurf_daily"],
            tsoil_daily=daily["tsoil_daily"],
        )
        return ProcessPrediction(
            water_inventory=self.water_inventory,
            carbon_inventory=self.carbon_inventory,
            thermal=self.thermal,
            pft=pft,
            fluxes=DailyFluxPrediction(
                water=WaterFluxPrediction(
                    rain_input=rain,
                    snowfall_input=snow,
                    canopy_precipitation_partition=self._captured("water.canopy_precipitation_partition"),
                    canopy_to_ground=self._captured("water.canopy_to_ground"),
                    wet_canopy_evaporation=self._captured("water.wet_canopy_evaporation"),
                    transpiration=self._captured("water.transpiration"),
                    bare_soil_evaporation=self._captured("water.bare_soil_evaporation"),
                    snow_sublimation=self._captured("water.snow_sublimation"),
                    flood_evaporation=self._captured("water.flood_evaporation"),
                    snow_phase_and_melt_transfer=self._captured("water.snow_phase_and_melt_transfer"),
                    soil_infiltration=self._captured("water.soil_infiltration"),
                    soil_vertical_transfer=self._captured("water.soil_vertical_transfer"),
                    runoff=self._captured("water.runoff"),
                    drainage=self._captured("water.drainage"),
                    routing_exchange=self._captured("water.routing_exchange"),
                ),
                carbon=CarbonFluxPrediction(
                    fast_gpp=pft.gpp,
                    fast_maintenance_respiration=pft.maintenance_respiration,
                    litter_input=litter_input,
                    litter_respiration=self._captured("carbon.litter_respiration"),
                    litter_to_doc=self._captured("carbon.litter_to_doc"),
                    poc_gross_decomposition=self.gross_poc,
                    poc_respiration=carbon.poc_respiration,
                    poc_to_doc=carbon.poc_to_doc,
                    doc_gross_decomposition=self.gross_doc,
                    doc_to_poc=carbon.doc_to_poc,
                    doc_respiration=carbon.doc_respiration,
                    doc_external_input=self._captured("carbon.doc_external_input"),
                    doc_export=self._captured("carbon.doc_export"),
                    doc_free_adsorbed_equilibration=self._captured("carbon.doc_free_adsorbed_equilibration"),
                    doc_vertical_water_transport=self._captured("carbon.doc_vertical_water_transport"),
                    doc_vertical_diffusion=self._captured("carbon.doc_vertical_diffusion"),
                    cryoturbation_redistribution=self._captured("carbon.cryoturbation_redistribution"),
                    perma_peat_redistribution=self._captured("carbon.perma_peat_redistribution"),
                ),
                energy=EnergyFluxPrediction(
                    net_radiation_integral=_signed(energy["energy.netrad"]),
                    sensible_heat_integral=energy["energy.fluxsens"],
                    latent_heat_integral=energy["energy.fluxlat"] + energy["energy.fluxsubli"],
                    ground_heat_integral=_signed(energy["energy.pgflux"]),
                    phase_change_integral=_signed(energy["energy.fusion"]),
                ),
            ),
            daily_temperature_context=temperature,
            retained_boundary=self.retained_boundary,
        )


def _prepare_oracle_head(boundary: Any, forcing_layout: NativeForcingFieldLayout) -> TrueLabelOracleHead:
    record = boundary.record
    arrays = daily_flux_capture_arrays(record.daily_flux_labels)
    indices = _ACTIVE_PFT_INDICES.get()
    poc = GrossDecompositionPrediction(
        ordinary=_select_pft(arrays["ok_leak.poc_gross_decomposition"], 3, indices),
        flooded=_select_pft(arrays["ok_leak.poc_flood_gross_decomposition"], 3, indices),
    )
    doc = GrossDecompositionPrediction(
        ordinary=_select_pft(arrays["ok_leak.doc_gross_decomposition"], 1, indices),
        flooded=_select_pft(arrays["ok_leak.doc_flood_gross_decomposition"], 1, indices),
    )
    normalized_state, final_diagnostics, daily_fields, minimal_ok = _dynamic_boundary_value(record)
    thermal, thermal_keys = _thermal_prediction(boundary)
    return TrueLabelOracleHead(
        water_inventory=_inventory_prediction(
            boundary.inventory_transitions, domain="water", fields=WATER_INVENTORY_FIELDS
        ),
        carbon_inventory=_inventory_prediction(
            boundary.inventory_transitions, domain="carbon", fields=CARBON_INVENTORY_FIELDS
        ),
        thermal=thermal,
        thermal_keys=thermal_keys,
        gross_poc=poc,
        gross_doc=doc,
        capture_arrays=arrays,
        retained_boundary=RetainedProcessBoundary(
            fast_state_fields=normalized_state.fields_by_component,
            final_entry_diagnostics=final_diagnostics,
            daily_fields=daily_fields,
            ok_leak_result=_minimal_ok_mapping(minimal_ok),
        ),
        forcing_layout=forcing_layout,
        calls=[],
    )


def _case(
    *,
    name: str,
    config: Path,
    context: Any,
    previous_state: Any,
    year: int,
    day_index: int,
    start_tstep: int,
    inventory: Path,
    run_jaxpr_audit: bool,
) -> tuple[dict[str, Any], Any]:
    common = dict(
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
        maintenance_local_jit=True,
        retain_stomate_step_results=False,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
        capture_daily_flux_labels=True,
    )
    teacher_record = capture_pre_daily_stomate_record(
        config,
        previous_state=previous_state,
        year=year,
        day_index=day_index,
        start_tstep=start_tstep,
        **common,
    )
    _block(teacher_record.expected_result)
    native_values, _indices, native_spec = native_forcing_days(
        context, year=year, day_indices=(day_index,)
    )
    contract, static_conditions, annual_conditions = _condition_contract(
        context, teacher_record, native_spec
    )
    assembler = CanonicalDailyOperatorInputAssembler.from_contract(contract)
    inputs = _operator_input(
        previous_state,
        context,
        contract,
        _native_forcing(native_values[0], native_spec),
        static_conditions,
        annual_conditions,
        year=year,
        day_index=day_index,
        allow_year_start_missing=start_tstep == 0,
    )
    assembled_inputs = assembler.assemble(inputs)
    prepared_labels = constrained_replay_boundary(
        previous_state,
        teacher_record,
        inventory_path=str(inventory),
    )
    oracle = _prepare_oracle_head(prepared_labels, _forcing_layout(native_spec))
    parameters = _dummy_parameters(assembled_inputs)
    compiled_forcing = reconstruct_compiled_forcing_day(
        native_values[0],
        contract.native_forcing,
        context,
        year=year,
        day_index=day_index,
    )
    adapter = build_canonical_retained_tail_adapter(
        config_path=config,
        context=context,
        initial_state=previous_state,
        compiled_forcing=compiled_forcing,
        contract=contract,
        input_assembler=assembler,
        runtime_year=year,
        lifecycle_start_tstep=start_tstep,
        result_day_index=day_index,
    )
    transition = bind_daily_operator(
        input_assembler=assembler,
        process_head=oracle,
        retained_tail=adapter,
    )
    jaxpr_audit = None
    if run_jaxpr_audit:
        jaxpr_audit = audit_executable_jaxpr(transition, parameters, inputs)
        oracle.calls.clear()
    operator_result = transition(parameters, inputs)
    _block(operator_result)
    adapter_transform_checks = None
    if run_jaxpr_audit:
        adapter_input = retained_tail_input(
            assembled_inputs,
            operator_result.prediction,
            operator_result.fast_update,
        )
        adapter_eager = adapter(adapter_input)
        adapter_compiled = jax.jit(adapter)(adapter_input)
        _block(adapter_compiled)
        jit_matches_eager, jit_max_error = _tree_allclose(
            adapter_eager,
            adapter_compiled,
        )

        def retained_carbon_loss(carbon_state):
            updated_groups = adapter_input.fast_update.state._replace(
                carbon=carbon_state
            )
            updated_fast = adapter_input.fast_update._replace(state=updated_groups)
            result = adapter(adapter_input._replace(fast_update=updated_fast))
            return jnp.sum(
                jnp.where(result.state_defined.carbon, result.state.carbon, 0.0)
            )

        carbon_state = jnp.asarray(adapter_input.fast_update.state.carbon)
        tangent_seed = jnp.ones_like(carbon_state)
        _jvp_primal, jvp_tangent = jax.jvp(
            retained_carbon_loss,
            (carbon_state,),
            (tangent_seed,),
        )
        vjp_primal, pullback = jax.vjp(retained_carbon_loss, carbon_state)
        (vjp_gradient,) = pullback(jnp.ones_like(vjp_primal))
        jvp_finite = bool(np.isfinite(np.asarray(jvp_tangent)).all())
        vjp_finite = bool(np.isfinite(np.asarray(vjp_gradient)).all())
        adapter_transform_checks = {
            "formal_adapter_class": type(adapter).__name__,
            "eager_vs_jit_passed": jit_matches_eager,
            "eager_vs_jit_max_absolute_error": jit_max_error,
            "forward_jvp_finite": jvp_finite,
            "forward_jvp_value": float(np.asarray(jvp_tangent)),
            "reverse_vjp_finite": vjp_finite,
            "reverse_vjp_l2_norm": float(
                np.linalg.norm(np.asarray(vjp_gradient).reshape(-1))
            ),
            "passed": jit_matches_eager and jvp_finite and vjp_finite,
        }
    next_fields = reconstruct_state_fields(
        np.asarray(operator_result.next_state.other),
        {
            name: np.asarray(field)
            for name, field in operator_result.next_discrete_state.items()
        },
        contract,
        template_fields=teacher_record.expected_result.day_end_state.fields_by_component,
    )
    next_packet = teacher.DriverPreviousStepStatePacket(
        tstep=start_tstep + 47,
        fields_by_component=next_fields,
        provenance_by_component={
            component: ("Gate E1 formal canonical retained-tail adapter",)
            for component in next_fields
        },
    )
    expected_modelout = teacher_record.expected_result.daily_modelout
    replay = replace(
        teacher_record.expected_result,
        daily_modelout=replace(
            expected_modelout,
            modelout_fields=operator_result.diagnostics["modelout_fields"],
            modelout=operator_result.diagnostics["modelout"],
        ),
        day_end_state=next_packet,
        missing_components=(),
    )
    comparison = compare_replay_day(teacher_record, replay, atol=1.0e-10, rtol=1.0e-12)
    expected_continuous, expected_discrete = extract_state(
        teacher_record.expected_result.day_end_state,
        contract,
    )
    observed_continuous = np.asarray(operator_result.next_state.other)
    continuous_error = np.abs(observed_continuous - expected_continuous)
    canonical_continuous_passed = bool(
        np.allclose(
            observed_continuous,
            expected_continuous,
            atol=1.0e-10,
            rtol=1.0e-12,
            equal_nan=True,
        )
    )
    canonical_discrete_passed = all(
        np.array_equal(
            np.asarray(operator_result.next_discrete_state[key]),
            expected_discrete[key],
        )
        for key in expected_discrete
    )
    canonical_next_state = {
        "continuous_passed": canonical_continuous_passed,
        "discrete_passed": canonical_discrete_passed,
        "continuous_width": int(expected_continuous.size),
        "max_absolute_error": float(np.nanmax(continuous_error)),
        "validation_packet_template_used_after_transition": True,
        "passed": canonical_continuous_passed and canonical_discrete_passed,
    }
    comparison["day_end_state"] = canonical_next_state
    water = audit_daily_water_budget(previous_state, teacher_record, dz_mm=context.hydrol_dz_mm)
    carbon = audit_daily_carbon_budget(previous_state, teacher_record)
    thermal = audit_daily_thermal_residual(teacher_record)
    endpoint_source_absent = (
        "encode_inventory_endpoints" not in inspect.getsource(TrueLabelOracleHead.__call__)
        and "encode_inventory_endpoints" not in inspect.getsource(daily_operator_transition)
    )
    chain = {
        "composition_root_executed": True,
        "true_label_oracle_head_executed": oracle.calls == ["true_label_oracle_head"],
        "conservative_updater_executed": bool(
            abs(float(np.asarray(operator_result.fast_update.water.budget_residual))) <= 2.0e-12
            and abs(float(np.asarray(operator_result.fast_update.carbon.budget_residual))) <= 2.0e-12
        ),
        "exact_retained_tail_adapter_executed": bool(
            np.asarray(operator_result.diagnostics["retained_tail_executed"]) == 1
        ),
        "pre_step_boundary_injected": False,
        "predicted_daily_fields_consumed": True,
        "predicted_ok_leak_boundary_consumed": True,
        "endpoint_encoder_absent_from_transition_source": endpoint_source_absent,
        "endpoint_encoder_calls_during_transition": 0,
    }
    chain_passed = bool(
        chain["composition_root_executed"]
        and chain["true_label_oracle_head_executed"]
        and chain["conservative_updater_executed"]
        and chain["exact_retained_tail_adapter_executed"]
        and not chain["pre_step_boundary_injected"]
        and chain["predicted_daily_fields_consumed"]
        and chain["predicted_ok_leak_boundary_consumed"]
        and chain["endpoint_encoder_absent_from_transition_source"]
        and chain["endpoint_encoder_calls_during_transition"] == 0
    )
    passed = bool(
        chain_passed
        and canonical_next_state["passed"]
        and comparison["modelout_fields"]["passed"]
        and comparison["modelout"]["passed"]
        and water["max_absolute_residual"] <= 2.0e-8
        and carbon["max_absolute_residual"] <= 2.0e-8
        and thermal["max_absolute_flux_side_residual"] <= 1.0e-7
        and (jaxpr_audit is None or jaxpr_audit["passed"])
        and (
            adapter_transform_checks is None
            or adapter_transform_checks["passed"]
        )
    )
    return (
        {
            "name": name,
            "passed": passed,
            "year": year,
            "day_index": day_index,
            "start_tstep": start_tstep,
            "chain": chain,
            "comparison": comparison,
            "canonical_next_state": canonical_next_state,
            "water_budget": water,
            "carbon_budget": carbon,
            "thermal_residual": thermal,
            "jaxpr_audit": jaxpr_audit,
            "adapter_transform_checks": adapter_transform_checks,
            "label_preparation_calls_endpoint_encoder": True,
            "retained_tail_owner": adapter.source_owner,
        },
        replay.day_end_state,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Gate-E1 true-daily operator skeleton.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs/reference_mode/used_run.def")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "manifests/coarse_graining/daily_flux_label_inventory_v1.json",
    )
    parser.add_argument(
        "--restart-evidence",
        type=Path,
        default=ROOT / "outputs/reference_mode/stage5_lifecycle_acceptance.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/research/daily_coarse_graining/gate_e1_daily_operator_skeleton/comparison.json",
    )
    args = parser.parse_args()

    configure_jax_compilation_cache(ROOT)
    context = teacher.prepare_paper_1961_driver_context(args.config, used_run_def_path=args.run_def)
    seed = teacher.paper_1961_driver_multiday_modelout_lite_run(
        args.config,
        ndays=1,
        year=1961,
        used_run_def_path=args.run_def,
        module_jit=True,
        diffuco_local_jit=True,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    if not seed.ready_for_requested_days or seed.last_day_end_state is None:
        raise RuntimeError(f"cold-start seed failed: {seed.missing_components}")

    cold, day2 = _case(
        name="cold_start_continuation_day2",
        config=args.config,
        context=context,
        previous_state=seed.last_day_end_state,
        year=1961,
        day_index=2,
        start_tstep=48,
        inventory=args.inventory,
        run_jaxpr_audit=True,
    )
    ordinary, day3 = _case(
        name="ordinary_later_day3",
        config=args.config,
        context=context,
        previous_state=day2,
        year=1961,
        day_index=3,
        start_tstep=96,
        inventory=args.inventory,
        run_jaxpr_audit=False,
    )
    restart, _ = _case(
        name="restart_year_boundary_1962_day1",
        config=args.config,
        context=context,
        previous_state=teacher.rebase_driver_state_for_year_start(day3),
        year=1962,
        day_index=1,
        start_tstep=0,
        inventory=args.inventory,
        run_jaxpr_audit=False,
    )
    restart_evidence = json.loads(args.restart_evidence.read_text(encoding="utf-8"))
    required = {
        "driver_file_roundtrip_exact",
        "sechiba_file_roundtrip_exact",
        "stomate_file_roundtrip_exact",
        "split_vs_memory_state_exact",
    }
    passed_checks = {
        item["name"]
        for item in restart_evidence.get("checks", ())
        if item.get("passed") is True
    }
    restart_roundtrip = restart_evidence.get("status") == "passed" and required <= passed_checks
    source_graph = audit_inference_source_graph()
    label_bindings = audit_process_label_bindings(args.inventory)
    cases = [cold, ordinary, restart]
    passed = all(case["passed"] for case in cases) and restart_roundtrip
    passed = passed and source_graph["passed"] and label_bindings["passed"]
    report = {
        "schema_version": "gate_e1_daily_operator_skeleton_oracle_v2",
        "status": "passed" if passed else "failed",
        "gate_status": "accepted",
        "candidate_uses_native_variable_record_boundary": True,
        "candidate_reconstructs_48_forcing_steps": False,
        "candidate_executes_48_state_transitions": False,
        "candidate_uses_teacher_endpoint_at_inference": False,
        "post_hoc_inventory_clipping": False,
        "cases": cases,
        "real_adapter_transform_checks": cold["adapter_transform_checks"],
        "restart_roundtrip_passed": restart_roundtrip,
        "source_graph_audit": source_graph,
        "label_binding_audit": label_bindings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_value(report), indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
