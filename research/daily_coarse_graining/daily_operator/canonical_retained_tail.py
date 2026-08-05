"""Typed reusable adapter over the accepted canonical retained daily owner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from jax_orchidee.driver import orchestration as teacher
from jax_orchidee.stomate.reference import read_stomate_restart_season_state
from research.daily_coarse_graining.canonical_retained_tail import (
    CanonicalRetainedTailDayInputs,
    canonical_retained_tail_day_inputs,
    canonical_retained_tail_dynamic_transition,
)
from research.daily_coarse_graining.constrained_daily_replay import (
    CARBON_INVENTORY_FIELDS,
    WATER_INVENTORY_FIELDS,
)
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    _compiled_select_pft_axes,
)

from .input_assembly import PFT_AXES, CanonicalDailyOperatorInputAssembler
from .types import RetainedTailInput, RetainedTailResult


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for name in path:
        current = current[name]
    return current


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _copy_mapping(child) if isinstance(child, Mapping) else child
        for name, child in value.items()
    }


@dataclass(frozen=True)
class CanonicalRetainedTailStaticOwner:
    """Trace-static scientific ownership and lifecycle configuration."""

    config_path: Path
    context: Any
    contract: DailyMarkovContract
    input_assembler: CanonicalDailyOperatorInputAssembler
    mineral_imin: int
    mineral_imax: int
    daily_carbon_dispatch: Mapping[str, Any]
    metadata: Any
    season_provenance: Any
    runtime_year: int
    lifecycle_start_tstep: int
    result_day_index: int

    def __post_init__(self) -> None:
        start = int(self.lifecycle_start_tstep)
        if start < 0 or start % 48 != 0:
            raise ValueError("canonical retained-tail lifecycle must start on a day boundary")
        if self.input_assembler.contract.sha256 != self.contract.sha256:
            raise ValueError("retained-tail owner and input assembler contract drift")


def build_canonical_retained_tail_adapter(
    *,
    config_path: Path,
    context: Any,
    initial_state: Any,
    compiled_forcing: Any,
    contract: DailyMarkovContract,
    input_assembler: CanonicalDailyOperatorInputAssembler,
    runtime_year: int,
    lifecycle_start_tstep: int,
    result_day_index: int,
) -> CanonicalRetainedTailAdapter:
    """Bind accepted retained-owner configuration without script-local wiring."""

    steps_per_day = int(
        round(context.runtime.dt_stomate / context.runtime.dt_sechiba)
    )
    transition_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=initial_state,
        year=int(runtime_year),
        start=int(lifecycle_start_tstep),
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
        compiled_forcing_series=compiled_forcing,
    )
    tables = transition_inputs.hydrol_runtime_static_tables
    season = read_stomate_restart_season_state(
        context.first_step_restart_state.stomate_input
    )
    run_scalars = transition_inputs.day_start_bundle.run_scalars
    template = transition_inputs.compiled_base_payload_template
    metadata = teacher.DriverRuntimeStepMetadata(
        pref_soil_veg=run_scalars.pref_soil_veg,
        nstm=run_scalars.nstm,
        is_tree=run_scalars.is_tree,
        is_peat=run_scalars.is_peat,
        lalo=template.lalo,
        npts=template.kjpindex,
    )
    static = {
        "metadata": metadata,
        "stomate_parameter_values": teacher._compiled_stomate_parameter_values(
            context
        ),
        "hydrol_table_arrays": teacher._compiled_hydrol_table_arrays(tables),
        "landpoint_payload": teacher._compiled_landpoint_payload(template),
        "stomate_restart_template": context.first_step_restart_state.stomate,
        "stomate_season_values": {
            name: value
            for name, value in season._asdict().items()
            if name != "provenance"
        },
        "diffuco_parameter_values": teacher._compiled_diffuco_parameter_values(
            context
        ),
    }
    day_inputs = canonical_retained_tail_day_inputs(compiled_forcing, static)
    return CanonicalRetainedTailAdapter(
        static_owner=CanonicalRetainedTailStaticOwner(
            config_path=config_path,
            context=context,
            contract=contract,
            input_assembler=input_assembler,
            mineral_imin=int(tables.mineral.imin),
            mineral_imax=int(tables.mineral.imax),
            daily_carbon_dispatch=teacher._paper_daily_carbon_static_dispatch(
                context,
                initial_state,
            ),
            metadata=metadata,
            season_provenance=season.provenance,
            runtime_year=int(runtime_year),
            lifecycle_start_tstep=int(lifecycle_start_tstep),
            result_day_index=int(result_day_index),
        ),
        day_inputs=jax.tree_util.tree_map(jnp.asarray, day_inputs),
    )


@dataclass(frozen=True, eq=False)
class CanonicalRetainedTailAdapter:
    """Public typed adapter; historical anonymous targets remain internal.

    ``day_inputs`` contains only deterministic retained-owner arrays. The
    process API consumes :class:`RetainedTailInput`; it never accepts a
    Teacher endpoint, day-end target, or anonymous fast-day vector.
    """

    static_owner: CanonicalRetainedTailStaticOwner
    day_inputs: CanonicalRetainedTailDayInputs
    source_owner: str = (
        "accepted canonical season/STOMATE retained daily owner via typed Gate-E1 adapter"
    )

    def _dynamic_day_inputs(self, value: RetainedTailInput) -> Any:
        """Lift closed-over day arrays into the staged retained-owner graph."""

        token = jnp.sum(jnp.asarray(value.fast_update.state.other))

        def lift(leaf: Any) -> Any:
            array = jnp.asarray(leaf)
            return jnp.where(token == token, array, array)

        return jax.tree_util.tree_map(lift, self.day_inputs)

    def _select_pft(self, value: Any, axis: int) -> Any:
        return jnp.take(
            jnp.asarray(value),
            jnp.asarray(self.static_owner.contract.active_pft_indices, dtype=jnp.int32),
            axis=axis,
        )

    def _restore_pft(self, base: Any, compact: Any, axis: int) -> Any:
        result = jnp.asarray(base)
        source = jnp.asarray(compact)
        for compact_index, full_index in enumerate(
            self.static_owner.contract.active_pft_indices
        ):
            target_slice = [slice(None)] * result.ndim
            source_slice = [slice(None)] * source.ndim
            target_slice[axis] = int(full_index)
            source_slice[axis] = int(compact_index)
            result = result.at[tuple(target_slice)].set(source[tuple(source_slice)])
        return result

    def _unpack_water(self, vector: Any, base_fields: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(base_fields)
        values = jnp.asarray(vector, dtype=jnp.float64)
        cursor = 0
        n_active = len(self.static_owner.contract.active_pft_indices)
        for field in WATER_INVENTORY_FIELDS:
            base = jnp.asarray(base_fields[field])
            axis = PFT_AXES.get(field)
            compact_shape = list(base.shape)
            if axis is not None:
                compact_shape[axis] = n_active
            size = 1
            for extent in compact_shape:
                size *= int(extent)
            if field == "qsintveg":
                storage = values[cursor : cursor + size].reshape(compact_shape)
                cursor += size
                debt = values[cursor : cursor + size].reshape(compact_shape)
                cursor += size
                compact = storage - debt
            else:
                compact = values[cursor : cursor + size].reshape(compact_shape)
                cursor += size
            result[field] = (
                compact
                if axis is None
                else self._restore_pft(base, compact, axis)
            )
        if cursor != values.size:
            raise ValueError("water updater vector does not match retained owner layout")
        return result

    def _unpack_carbon(self, vector: Any, base_fields: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(base_fields)
        values = jnp.asarray(vector, dtype=jnp.float64)
        cursor = 0
        n_active = len(self.static_owner.contract.active_pft_indices)
        for field in CARBON_INVENTORY_FIELDS:
            full = jnp.asarray(base_fields[field])
            axis = PFT_AXES[field]
            compact_shape = list(full.shape)
            compact_shape[axis] = n_active
            size = 1
            for extent in compact_shape:
                size *= int(extent)
            compact = values[cursor : cursor + size].reshape(compact_shape)
            cursor += size
            result[field] = self._restore_pft(full, compact, axis)
        if cursor != values.size:
            raise ValueError("carbon updater vector does not match retained owner layout")
        return result

    def _unpack_thermal(
        self,
        vector: Any,
        base_fields: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        result = _copy_mapping(base_fields)
        values = jnp.asarray(vector, dtype=jnp.float64)
        cursor = 0
        for key in self.static_owner.input_assembler.thermal_keys:
            component, field = key.split(".", 1)
            shape = tuple(jnp.asarray(base_fields[component][field]).shape)
            size = 1
            for extent in shape:
                size *= int(extent)
            result[component][field] = values[cursor : cursor + size].reshape(shape)
            cursor += size
        if cursor != values.size:
            raise ValueError("thermal updater vector does not match retained owner layout")
        return result

    def _typed_fast_target(self, value: RetainedTailInput) -> Any:
        predicted = value.process_prediction.retained_boundary
        fields = self._unpack_thermal(
            value.fast_update.state.thermal,
            predicted.fast_state_fields,
        )
        hydrol = fields["hydrol_previous_step_state"]
        fields["hydrol_previous_step_state"] = self._unpack_water(
            value.fast_update.state.water,
            hydrol,
        )
        ok_leak = self._unpack_carbon(
            value.fast_update.state.carbon,
            predicted.ok_leak_result,
        )
        groups = {
            "daily_interface": predicted.daily_fields,
            "ok_leak": ok_leak,
            "final_diagnostics": predicted.final_entry_diagnostics,
        }
        pieces = []
        for leaf in self.static_owner.contract.fast_day_target_leaves:
            source = (
                _get_path(fields[leaf.component], leaf.path)
                if leaf.component is not None
                else _get_path(groups[leaf.family], leaf.path)
            )
            compact = _compiled_select_pft_axes(source, leaf)
            if compact.shape != leaf.shape:
                raise ValueError(
                    f"typed retained boundary shape drift for {leaf.key}: "
                    f"{compact.shape} != {leaf.shape}"
                )
            pieces.append(jnp.asarray(compact, dtype=jnp.float64).reshape(-1))
        return jnp.concatenate(tuple(pieces))

    def __call__(self, value: RetainedTailInput) -> RetainedTailResult:
        owner = self.static_owner
        internal_target = self._typed_fast_target(value)
        state_result, modelout_fields, modelout = canonical_retained_tail_dynamic_transition(
            value.fast_update.state.other,
            value.fast_update.discrete_state,
            internal_target,
            self._dynamic_day_inputs(value),
            value.canonical_input.year,
            value.canonical_input.day_index,
            config_path=owner.config_path,
            context=owner.context,
            contract=owner.contract,
            mineral_imin=int(owner.mineral_imin),
            mineral_imax=int(owner.mineral_imax),
            daily_carbon_dispatch=owner.daily_carbon_dispatch,
            metadata=owner.metadata,
            season_provenance=owner.season_provenance,
            runtime_year=int(owner.runtime_year),
            lifecycle_start_tstep=int(owner.lifecycle_start_tstep),
            result_day_index=int(owner.result_day_index),
            return_modelout=True,
        )
        next_continuous, next_discrete = state_result
        next_input = value.canonical_input._replace(
            continuous_state=next_continuous,
            discrete_state=next_discrete,
        )
        assembled = owner.input_assembler.assemble(next_input)
        return RetainedTailResult(
            state=assembled.state,
            state_defined=assembled.state_defined,
            discrete_state=next_discrete,
            diagnostics={
                "retained_tail_executed": jnp.asarray(1, dtype=jnp.int32),
                "modelout_fields": modelout_fields,
                "modelout": modelout,
            },
        )
