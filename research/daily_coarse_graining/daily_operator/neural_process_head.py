"""Formal process-head adapter for the first neural architecture A/B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp

from research.daily_coarse_graining.constrained_daily_replay import (
    CARBON_INVENTORY_FIELDS,
    THERMAL_FIELDS,
    WATER_INVENTORY_FIELDS,
)
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    reconstruct_state_fields_compiled,
)

from .input_assembly import CanonicalDailyOperatorInputAssembler
from .neural_architecture import (
    NeuralArchitectureConfig,
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
    NeuralProcessContext,
    ProcessAxisLayout,
    assemble_neural_process_context,
    neural_process_outputs,
)
from .process_derivations import (
    NativeForcingFieldLayout,
    derive_carbon_decomposition,
    derive_end_of_day_driver_fields,
    derive_litter_input,
    derive_native_daily_deterministic_fields,
    integrate_native_precipitation,
)
from .types import (
    AssembledDailyOperatorInput,
    CarbonFluxPrediction,
    DailyFluxPrediction,
    DailyTemperatureContext,
    EnergyFluxPrediction,
    GrossDecompositionPrediction,
    InventoryPrediction,
    PFTProcessPrediction,
    ProcessPrediction,
    RetainedProcessBoundary,
    SignedFluxPrediction,
    ThermalPrediction,
    WaterFluxPrediction,
)


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _copy_mapping(child) if isinstance(child, Mapping) else child
        for name, child in value.items()
    }


def _ensure_landpoint_axis(value: Any, expected: Any) -> Any:
    array = jnp.asarray(value, dtype=jnp.float64)
    expected_shape = tuple(jnp.asarray(expected).shape)
    if array.shape == expected_shape:
        return array
    if (1,) + array.shape == expected_shape:
        return array[None, ...]
    return expected


def _signed(value: Any) -> SignedFluxPrediction:
    value = jnp.asarray(value, dtype=jnp.float64)
    positive = jnp.maximum(value, 0.0)
    negative = jnp.maximum(-value, 0.0)
    return SignedFluxPrediction(positive=positive, negative=negative, value=value)


@dataclass(frozen=True, eq=False)
class NeuralProcessHead:
    """Bind one frozen neural arm to the accepted Gate-E process boundary."""

    arm: str
    contract: DailyMarkovContract
    input_assembler: CanonicalDailyOperatorInputAssembler
    schema: NeuralOutputSchema
    axis_layout: ProcessAxisLayout
    input_normalization: NeuralInputNormalization
    output_normalization: NeuralOutputNormalization
    forcing_layout: NativeForcingFieldLayout
    forcing_indices: Mapping[str, int]
    config: NeuralArchitectureConfig = NeuralArchitectureConfig()

    def __post_init__(self) -> None:
        if self.contract.sha256 != self.input_assembler.contract.sha256:
            raise ValueError("neural head and canonical assembler contract drift")

    @property
    def active_pft_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.contract.active_pft_indices)

    def assemble_context(
        self,
        parameters: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ) -> NeuralProcessContext:
        return assemble_neural_process_context(
            parameters,
            inputs,
            normalization=self.input_normalization,
            axis_layout=self.axis_layout,
            config=self.config,
        )

    def _select_active_pft(self, value: Any, *, axis: int = 0) -> Any:
        return jnp.take(
            jnp.asarray(value, dtype=jnp.float64),
            jnp.asarray(self.active_pft_indices, dtype=jnp.int32),
            axis=axis,
        )

    @staticmethod
    def _inventory_prediction(
        outgoing: Any,
        incoming: Any,
        defined: Any,
    ) -> InventoryPrediction:
        outgoing = jnp.asarray(outgoing, dtype=jnp.float64).reshape(-1)
        incoming = jnp.asarray(incoming, dtype=jnp.float64).reshape(-1)
        defined = jnp.asarray(defined, dtype=jnp.bool_).reshape(-1)
        size = outgoing.size
        if incoming.shape != outgoing.shape or defined.shape != outgoing.shape:
            raise ValueError("neural inventory outputs do not match the canonical vector")
        # This is the admitted Gate-C endpoint representation: decreases are
        # external outgoing amounts and increases are external incoming
        # amounts. Explicit typed transfer labels remain available separately.
        return InventoryPrediction(
            outgoing_fraction=jnp.where(defined, outgoing, 0.0),
            external_input_amount=jnp.where(defined, incoming, 0.0),
            external_output_share=jnp.where(defined, 1.0, 0.0),
            destination_shares=jnp.zeros((size, size), dtype=jnp.float64),
            defined=defined,
        )

    def _water_inventory(self, outputs: Mapping[str, Any], inputs: Any) -> InventoryPrediction:
        outgoing = []
        incoming = []
        for field in WATER_INVENTORY_FIELDS:
            prefixes = (
                (f"inventory.water.{field}.storage", f"inventory.water.{field}.debt")
                if field == "qsintveg"
                else (f"inventory.water.{field}",)
            )
            for prefix in prefixes:
                outgoing.append(outputs[f"{prefix}.outgoing_fraction"].reshape(-1))
                incoming.append(outputs[f"{prefix}.incoming_amount"].reshape(-1))
        return self._inventory_prediction(
            jnp.concatenate(tuple(outgoing)),
            jnp.concatenate(tuple(incoming)),
            inputs.state_defined.water,
        )

    def _carbon_inventory(self, outputs: Mapping[str, Any], inputs: Any) -> InventoryPrediction:
        outgoing = []
        incoming = []
        for field in CARBON_INVENTORY_FIELDS:
            prefix = f"inventory.carbon.{field}"
            outgoing.append(outputs[f"{prefix}.outgoing_fraction"].reshape(-1))
            incoming.append(outputs[f"{prefix}.incoming_amount"].reshape(-1))
        return self._inventory_prediction(
            jnp.concatenate(tuple(outgoing)),
            jnp.concatenate(tuple(incoming)),
            inputs.state_defined.carbon,
        )

    def _thermal(self, outputs: Mapping[str, Any], inputs: Any) -> ThermalPrediction:
        tendencies = []
        bounds = []
        available = set(self.input_assembler.thermal_keys)
        for component, field, bound in THERMAL_FIELDS:
            if f"{component}.{field}" not in available:
                continue
            key = f"thermal.{component}.{field}.normalized_tendency"
            value = jnp.asarray(outputs[key], dtype=jnp.float64).reshape(-1)
            tendencies.append(value)
            bounds.append(jnp.full(value.shape, float(bound), dtype=jnp.float64))
        tendency = jnp.concatenate(tuple(tendencies))
        bound = jnp.concatenate(tuple(bounds))
        defined = jnp.asarray(inputs.state_defined.thermal, dtype=jnp.bool_)
        if tendency.shape != defined.shape or bound.shape != defined.shape:
            raise ValueError("neural thermal output does not match canonical thermal layout")
        return ThermalPrediction(
            normalized_tendency=jnp.where(defined, tendency, 0.0),
            bounds=bound,
            defined=defined,
        )

    def _pft_prediction(self, outputs: Mapping[str, Any], inputs: Any) -> PFTProcessPrediction:
        active = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
        fraction = jnp.asarray(inputs.pft.pft_fraction, dtype=jnp.float64)
        bare = active & (fraction > 0.0) & (jnp.asarray(inputs.pft.pft_traits)[:, 0] > 0.5)
        vegetation = active & (fraction > 0.0) & ~bare
        gpp = jnp.asarray(outputs["daily.fast_gpp"], dtype=jnp.float64)[0]
        maintenance = jnp.asarray(
            outputs["daily.fast_maintenance_respiration"],
            dtype=jnp.float64,
        )[0]
        transpiration = self._select_active_pft(outputs["typed.water.transpir"])
        wet = self._select_active_pft(outputs["typed.water.vevapwet"])
        bare_value = jnp.asarray(outputs["typed.water.vevapnu"], dtype=jnp.float64)
        return PFTProcessPrediction(
            gpp=jnp.where(vegetation, gpp, 0.0),
            maintenance_respiration=jnp.where(vegetation[:, None], maintenance, 0.0),
            transpiration=jnp.where(vegetation, transpiration, 0.0),
            wet_canopy_evaporation=jnp.where(vegetation, wet, 0.0),
            bare_soil_evaporation=jnp.where(bare, bare_value, 0.0),
            vegetation_process_mask=vegetation,
            bare_soil_process_mask=bare,
        )

    def _gross_decomposition(self, outputs: Mapping[str, Any]) -> tuple[Any, Any]:
        poc_ordinary = self._select_active_pft(
            outputs["typed.ok_leak.poc_gross_decomposition"],
            axis=2,
        )
        poc_flooded = self._select_active_pft(
            outputs["typed.ok_leak.poc_flood_gross_decomposition"],
            axis=2,
        )
        poc = GrossDecompositionPrediction(
            ordinary=poc_ordinary[None, ...],
            flooded=poc_flooded[None, ...],
        )
        doc = GrossDecompositionPrediction(
            ordinary=self._select_active_pft(
                outputs["typed.ok_leak.doc_gross_decomposition"],
                axis=0,
            )[None, ...],
            flooded=self._select_active_pft(
                outputs["typed.ok_leak.doc_flood_gross_decomposition"],
                axis=0,
            )[None, ...],
        )
        return poc, doc

    def _fluxes(
        self,
        outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
        pft: PFTProcessPrediction,
    ) -> DailyFluxPrediction:
        rain, snow = integrate_native_precipitation(inputs.forcing, self.forcing_layout)
        poc, doc = self._gross_decomposition(outputs)
        derived = derive_carbon_decomposition(poc, doc, inputs)
        water = WaterFluxPrediction(
            rain_input=rain,
            snowfall_input=snow,
            canopy_precipitation_partition={
                name: outputs[f"typed.water.{name}"]
                for name in ("precip2canopy", "precip2ground")
            },
            canopy_to_ground=outputs["typed.water.canopy2ground"],
            wet_canopy_evaporation=pft.wet_canopy_evaporation,
            transpiration={
                "transpir": outputs["typed.water.transpir"],
                "rootsink": outputs["typed.water.rootsink"],
            },
            bare_soil_evaporation={
                "vevapnu": outputs["typed.water.vevapnu"],
                "vevapnu_ns": outputs["typed.water.vevapnu_ns"],
            },
            snow_sublimation={
                "vevapsno": outputs["typed.water.vevapsno"],
                "subsnownobio": outputs["typed.water.subsnownobio"],
            },
            flood_evaporation=outputs["typed.water.vevapflo"],
            snow_phase_and_melt_transfer={
                name: outputs[f"typed.water.{name}"]
                for name in (
                    "snowmelt",
                    "snowmelt_from_maxmass",
                    "snow_melt_mass_by_layer",
                    "snow_refreeze_mass_by_layer",
                    "snow_liquid_percolation_by_interface",
                )
            },
            soil_infiltration={
                name: outputs[f"typed.water.{name}"]
                for name in ("soil_infiltration", "water2infilt")
            },
            soil_vertical_transfer=outputs["typed.water.wat_flux"],
            runoff={
                name: outputs[f"typed.water.{name}"]
                for name in ("runoff", "runoff_per_soil", "runoff2peat")
            },
            drainage={
                name: outputs[f"typed.water.{name}"]
                for name in ("drainage", "drainage_per_soil")
            },
            routing_exchange={
                name: outputs[f"typed.water.{name}"]
                for name in ("returnflow", "reinfiltration", "irrigation", "floodout")
            },
        )
        carbon = CarbonFluxPrediction(
            fast_gpp=pft.gpp,
            fast_maintenance_respiration=pft.maintenance_respiration,
            litter_input=derive_litter_input(inputs),
            litter_respiration={
                name: outputs[f"typed.ok_leak.{name}"]
                for name in ("litter_respiration", "litter_flood_respiration")
            },
            litter_to_doc={
                name: outputs[f"typed.ok_leak.{name}"]
                for name in ("litter_to_doc", "floodcarbon_input")
            },
            poc_gross_decomposition=poc,
            poc_respiration=derived.poc_respiration,
            poc_to_doc=derived.poc_to_doc,
            doc_gross_decomposition=doc,
            doc_to_poc=derived.doc_to_poc,
            doc_respiration=derived.doc_respiration,
            doc_external_input={
                name: outputs[f"typed.ok_leak.{name}"]
                for name in (
                    "doc_to_topsoil",
                    "doc_to_subsoil",
                    "doc_precip2ground",
                    "doc_precip2canopy",
                    "dry_dep_canopy",
                )
            },
            doc_export={
                name: outputs[f"typed.ok_leak.{name}"]
                for name in ("doc_run", "doc_drain", "doc_flood", "doc_run_2_peat")
            },
            doc_free_adsorbed_equilibration=outputs["typed.ok_leak.doc_free_to_adsorbed"],
            doc_vertical_water_transport=outputs["typed.ok_leak.doc_advective_interface"],
            doc_vertical_diffusion=outputs["typed.ok_leak.doc_diffusive_interface"],
            cryoturbation_redistribution={
                name: outputs[f"typed.ok_leak.{name}"]
                for name in (
                    "cryoturbation_carbon_transfer",
                    "cryoturbation_doc_transfer",
                    "cryoturbation_litter_transfer",
                )
            },
            perma_peat_redistribution=outputs["typed.ok_leak.perma_peat_carbon_transfer"],
        )
        energy = EnergyFluxPrediction(
            net_radiation_integral=_signed(outputs["typed.energy.netrad"]),
            sensible_heat_integral=outputs["typed.energy.fluxsens"],
            latent_heat_integral={
                name: outputs[f"typed.energy.{name}"]
                for name in ("fluxlat", "fluxsubli", "lareva", "larsub")
            },
            ground_heat_integral=_signed(outputs["typed.energy.pgflux"]),
            phase_change_integral=_signed(outputs["typed.energy.fusion"]),
        )
        return DailyFluxPrediction(water=water, carbon=carbon, energy=energy)

    def _forcing_value(self, inputs: Any, name: str, *, reducer: str = "mean") -> Any:
        index = self.forcing_indices.get(name)
        if index is None:
            return jnp.asarray(0.0, dtype=jnp.float64)
        values = jnp.asarray(inputs.forcing.values, dtype=jnp.float64)[:, int(index)]
        mask = jnp.asarray(inputs.forcing.record_mask, dtype=jnp.bool_) & ~jnp.asarray(
            inputs.forcing.predecessor_mask,
            dtype=jnp.bool_,
        )
        selected = jnp.where(mask, values, 0.0)
        count = jnp.maximum(jnp.sum(mask, dtype=jnp.float64), 1.0)
        if reducer == "max":
            return jnp.max(jnp.where(mask, values, -jnp.inf))
        if reducer == "min":
            return jnp.min(jnp.where(mask, values, jnp.inf))
        return jnp.sum(selected) / count

    def _static_condition(self, inputs: AssembledDailyOperatorInput, name: str) -> Any:
        matches = tuple(
            leaf
            for leaf in self.contract.static_conditions.landpoint_static_leaves
            if leaf.name == name
        )
        if len(matches) != 1:
            raise ValueError(f"static condition requires exactly one {name!r} leaf")
        leaf = matches[0]
        return jnp.asarray(inputs.static_conditions, dtype=jnp.float64)[
            leaf.start : leaf.stop
        ].reshape(leaf.shape)

    def _daily_fields(
        self,
        outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
        start_fields: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        hydrol = start_fields["hydrol_previous_step_state"]
        gpp = jnp.asarray(outputs["daily.fast_gpp"], dtype=jnp.float64)
        maintenance = jnp.asarray(outputs["daily.fast_maintenance_respiration"], dtype=jnp.float64)
        full_resp = jnp.zeros((1, 14), dtype=jnp.float64)
        full_resp = full_resp.at[:, jnp.asarray(self.active_pft_indices)].set(jnp.sum(maintenance, axis=-1))
        deterministic = derive_native_daily_deterministic_fields(
            inputs.forcing,
            self.forcing_layout,
            model_interval_seconds=float(
                self.contract.native_forcing.model_interval_seconds
            ),
        )
        soilhum = jnp.mean(jnp.asarray(hydrol["mc"], dtype=jnp.float64), axis=-1)
        return {
            "flood_root_radia": jnp.zeros((1, 14), dtype=jnp.float64),
            "gpp_daily": gpp,
            "humrel_daily": jnp.asarray(hydrol["humrel"], dtype=jnp.float64),
            "litterhum_daily": jnp.mean(soilhum[:, :1], axis=-1),
            "precip_daily": deterministic.precip_daily,
            "resp_maint_part": maintenance,
            "resp_maint_radia": full_resp,
            "snowfall_daily": deterministic.snowfall_daily,
            "snowmass_daily": jnp.asarray(hydrol["snow"], dtype=jnp.float64).reshape(1),
            "soilhum_daily": soilhum,
            "t2m_daily": deterministic.t2m_daily,
            "t2m_max_daily": deterministic.t2m_max_daily,
            "t2m_min_daily": deterministic.t2m_min_daily,
            "tmc_topgrass_daily": jnp.mean(soilhum[:, :1], axis=-1),
            "tsoil_daily": jnp.asarray(outputs["daily.temperature_context.tsoil"]),
            "tsurf_daily": jnp.asarray(outputs["daily.temperature_context.tsurf"]),
            "wspeed_daily": self._forcing_value(inputs, "Wind").reshape(1),
        }

    def _retained_boundary(
        self,
        outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ) -> RetainedProcessBoundary:
        fields = _copy_mapping(
            reconstruct_state_fields_compiled(
                inputs.canonical_input.continuous_state,
                inputs.canonical_input.discrete_state,
                self.contract,
            )
        )
        # The accepted v6 representation marks 2,754/2,855 B_fast columns as
        # exact persistence.  Patch only named dynamic fields captured by v6;
        # inventory and thermal fields are overwritten by the conservative
        # updater inside CanonicalRetainedTailAdapter.
        hydrol = fields["hydrol_previous_step_state"]
        for name in (
            "canopy2ground",
            "drainage",
            "drainage_per_soil",
            "precip2canopy",
            "precip2ground",
            "runoff",
            "runoff2peat",
            "runoff_per_soil",
            "subsnownobio",
            "wat_flux",
            "water2infilt",
        ):
            key = f"typed.water.{name}"
            if key in outputs and name in hydrol:
                hydrol[name] = _ensure_landpoint_axis(outputs[key], hydrol[name])
        for component in ("driver_previous_step_state", "enerbil_previous_step_state"):
            component_fields = fields.get(component, {})
            for name in ("fluxsens", "soilflx", "soilflx_pft"):
                key = f"typed.energy.{name}"
                if key in outputs and name in component_fields:
                    component_fields[name] = _ensure_landpoint_axis(outputs[key], component_fields[name])

        measurement_heights = self._static_condition(inputs, "measurement_heights")
        driver_fields = derive_end_of_day_driver_fields(
            inputs.forcing,
            self.forcing_indices,
            scalar_measurement_height=measurement_heights[0],
        )
        fields.setdefault("driver_previous_step_state", {}).update(
            driver_fields._asdict()
        )

        rain, snow = integrate_native_precipitation(inputs.forcing, self.forcing_layout)
        daily = self._daily_fields(outputs, inputs, fields)
        slow = fields["slowproc_stomate_previous_step_state"]
        ok_leak = {
            name: slow[name]
            for name in (
                "litter_above",
                "litter_below",
                "lignin_struc_above",
                "lignin_struc_below",
                "litterpart",
                "dead_leaves",
                "fuel_1hr",
                "fuel_10hr",
                "fuel_100hr",
                "fuel_1000hr",
                "carbon_32l",
                "DOC",
                "interception_storage",
                "deepC_peat",
            )
        }
        return RetainedProcessBoundary(
            fast_state_fields=fields,
            final_entry_diagnostics={
                "t2mdiag": derive_native_daily_deterministic_fields(
                    inputs.forcing,
                    self.forcing_layout,
                    model_interval_seconds=float(
                        self.contract.native_forcing.model_interval_seconds
                    ),
                ).t2mdiag,
                "temp_sol": daily["tsurf_daily"],
            },
            daily_fields=daily,
            ok_leak_result=ok_leak,
        )

    def __call__(
        self,
        parameters: Mapping[str, Any],
        context: NeuralProcessContext,
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        outputs = neural_process_outputs(
            parameters,
            context,
            schema=self.schema,
            normalization=self.output_normalization,
            arm=self.arm,
            config=self.config,
        )
        return self.prediction_from_named_outputs(outputs, inputs)

    def prediction_from_named_outputs(
        self,
        outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ) -> ProcessPrediction:
        """Adapt already-decoded named fields without invoking this network.

        The successor architecture reuses this accepted scientific mapping,
        not the closed first-pair encoder, parameter tree, or checkpoint.
        """

        pft = self._pft_prediction(outputs, inputs)
        temperature = DailyTemperatureContext(
            tsurf_daily=outputs["daily.temperature_context.tsurf"],
            tsoil_daily=outputs["daily.temperature_context.tsoil"],
        )
        return ProcessPrediction(
            water_inventory=self._water_inventory(outputs, inputs),
            carbon_inventory=self._carbon_inventory(outputs, inputs),
            thermal=self._thermal(outputs, inputs),
            pft=pft,
            fluxes=self._fluxes(outputs, inputs, pft),
            daily_temperature_context=temperature,
            retained_boundary=self._retained_boundary(outputs, inputs),
            named_outputs=outputs,
            auxiliary_outputs={},
        )
