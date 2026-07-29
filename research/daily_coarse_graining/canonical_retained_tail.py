"""Pure-JAX handoff from canonical neural output to retained daily STOMATE."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.canonical_rollout import (
    _contract_finalize_fields,
    _projected_finalize_after_slowproc,
    _runtime_boundary,
)
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    _compiled_select_pft_axes,
    extract_state_compiled,
    reconstruct_fast_day_target_compiled,
    reconstruct_state_fields_compiled,
)
from research.daily_coarse_graining.gradient_training_ready import (
    deterministic_boundary_fields,
)


class CanonicalRetainedTailMetadataInputs(NamedTuple):
    """Numerical metadata that genuinely varies between compatible landpoints."""

    pref_soil_veg: Any
    lalo: Any


class CanonicalRetainedTailDayInputs(NamedTuple):
    """All landpoint-varying leaves consumed by one retained daily tail."""

    compiled_forcing: Any
    metadata: CanonicalRetainedTailMetadataInputs
    stomate_parameter_values: Any
    hydrol_table_arrays: Any
    landpoint_payload: Any
    stomate_restart_template: Any
    stomate_season_values: Any
    diffuco_parameter_values: Any


def canonical_retained_tail_day_inputs(
    compiled_forcing,
    static: Mapping[str, Any],
) -> CanonicalRetainedTailDayInputs:
    """Move every array-valued landpoint dependency into one stable PyTree."""

    return CanonicalRetainedTailDayInputs(
        compiled_forcing=compiled_forcing,
        metadata=CanonicalRetainedTailMetadataInputs(
            pref_soil_veg=static["metadata"].pref_soil_veg,
            lalo=static["metadata"].lalo,
        ),
        stomate_parameter_values=static["stomate_parameter_values"],
        hydrol_table_arrays=static["hydrol_table_arrays"],
        landpoint_payload=static["landpoint_payload"],
        stomate_restart_template=static["stomate_restart_template"],
        stomate_season_values=static["stomate_season_values"],
        diffuco_parameter_values=static["diffuco_parameter_values"],
    )


def _retained_tail_fast_target(
    physical_fast_day_target,
    compiled_forcing,
    contract: DailyMarkovContract,
    context,
):
    """Install exact forcing outputs and detach structurally inactive bare pools."""

    target = jnp.asarray(physical_fast_day_target, dtype=jnp.float64)
    deterministic = deterministic_boundary_fields(
        compiled_forcing,
        dt_sechiba=context.runtime.dt_sechiba,
        dt_stomate=context.runtime.dt_stomate,
    )
    for leaf in contract.fast_day_target_leaves:
        if leaf.path[0] in deterministic and leaf.family in {
            "daily_interface",
            "final_diagnostics",
        }:
            compact = _compiled_select_pft_axes(
                deterministic[leaf.path[0]], leaf
            ).reshape(-1)
            target = target.at[leaf.start : leaf.stop].set(compact)
        if (
            leaf.family == "daily_interface"
            and leaf.path == ("resp_maint_part",)
            and 0 in leaf.selected_pft_indices
        ):
            compact = target[leaf.start : leaf.stop].reshape(leaf.shape)
            pft_axis = leaf.axis_names.index("nvm")
            bare_source_index = leaf.selected_pft_indices.index(0)
            selected = [slice(None)] * compact.ndim
            selected[pft_axis] = bare_source_index
            selected = tuple(selected)
            compact = compact.at[selected].set(
                jax.lax.stop_gradient(compact[selected])
            )
            target = target.at[leaf.start : leaf.stop].set(compact.reshape(-1))
    return target


def canonical_retained_tail_dynamic_transition(
    continuous_state,
    discrete_state: Mapping[str, Any],
    physical_fast_day_target,
    day_inputs: CanonicalRetainedTailDayInputs,
    year,
    day_index,
    *,
    config_path: Path,
    context,
    contract: DailyMarkovContract,
    mineral_imin: int,
    mineral_imax: int,
    daily_carbon_dispatch: Mapping[str, Any],
    metadata: Any,
    season_provenance: Any,
    runtime_year: int,
):
    """Advance one canonical day with all landpoint-varying arrays explicit.

    The executable uses a normalized later-day lifecycle (`tstep=47`,
    `start_tstep=48`) exactly like the accepted dynamic replay scan. The
    scientific day number remains a dynamic input to season/STOMATE.
    """

    del year
    fields = reconstruct_state_fields_compiled(
        continuous_state,
        discrete_state,
        contract,
    )
    provenance = {
        component: ("canonical differentiable retained-tail transition",)
        for component in fields
    }
    previous_state = teacher.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=fields,
        provenance_by_component=provenance,
    )
    retained_target = _retained_tail_fast_target(
        physical_fast_day_target,
        day_inputs.compiled_forcing,
        contract,
        context,
    )
    boundary = reconstruct_fast_day_target_compiled(
        retained_target,
        contract.fast_day_target_leaves,
        template_fields=fields,
    )
    dynamic_metadata = teacher.DriverRuntimeStepMetadata(
        pref_soil_veg=day_inputs.metadata.pref_soil_veg,
        nstm=metadata.nstm,
        is_tree=metadata.is_tree,
        is_peat=metadata.is_peat,
        lalo=day_inputs.metadata.lalo,
        npts=metadata.npts,
    )
    transition, daily_fold, ok_leak = _runtime_boundary(
        boundary,
        metadata=dynamic_metadata,
        end_tstep=95,
    )
    ok_leak_updates = teacher._paper_half_hour_ok_leak_state_updates(ok_leak)
    replay_values = {
        "_paper_1961_later_day_half_hour_transition": transition,
        "_paper_later_day_daily_process_from_completed_entries": daily_fold,
        "_paper_later_day_daily_fold_prerequisite_gaps": (),
        "stomate_maintenance_respiration_parts_from_stacks": jnp.asarray(0.0),
        "_paper_half_hour_ok_leak_fold_from_entries": (
            ok_leak,
            ok_leak_updates,
        ),
    }
    contract_finalize_fields = _contract_finalize_fields(contract)
    with ExitStack() as stack:
        for name, value in replay_values.items():
            stack.enter_context(
                patch.object(
                    teacher,
                    name,
                    lambda *args, _value=value, **kwargs: _value,
                )
            )
        stack.enter_context(
            patch.object(
                teacher,
                "_sechiba_finalize_state_after_slowproc",
                lambda previous, slowproc: _projected_finalize_after_slowproc(
                    previous,
                    slowproc,
                    contract_fields=contract_finalize_fields,
                ),
            )
        )
        result = teacher.paper_1961_driver_later_day_runtime_result(
            config_path,
            previous_state=previous_state,
            day_index=2,
            year=runtime_year,
            start_tstep=48,
            used_run_def_path=context.run_def_path,
            prepared_context=context,
            module_jit=True,
            diffuco_local_jit=True,
            use_static_jit_daily_carbon=False,
            prebuild_day_payloads=True,
            use_compiled_sechiba_day=True,
            prebound_hydrol_runtime_static_tables=teacher.HydrolRuntimeStaticTables(
                mineral=teacher.MineralCWRRTables(
                    **day_inputs.hydrol_table_arrays._asdict(),
                    imin=mineral_imin,
                    imax=mineral_imax,
                ),
                peat=None,
            ),
            materialize_compiled_entries=False,
            outer_compiled_daily_carbon_dispatch=daily_carbon_dispatch,
            compiled_forcing_series=day_inputs.compiled_forcing,
            model_day_number=day_index,
            compiled_stomate_parameter_values=day_inputs.stomate_parameter_values,
            compiled_landpoint_payload=day_inputs.landpoint_payload,
            compiled_stomate_restart_template=day_inputs.stomate_restart_template,
            compiled_stomate_season_template=teacher.StomateRestartSeasonState(
                **day_inputs.stomate_season_values,
                provenance=season_provenance,
            ),
            compiled_diffuco_parameter_values=day_inputs.diffuco_parameter_values,
        )
    if result.day_end_state is None:
        raise RuntimeError(
            "retained daily tail did not produce day-end state: "
            f"{getattr(result, 'missing_components', ())}"
        )
    return extract_state_compiled(
        result.day_end_state.fields_by_component,
        contract,
    )


def canonical_retained_tail_transition(
    continuous_state,
    discrete_state: Mapping[str, Any],
    physical_fast_day_target,
    compiled_forcing,
    year,
    day_index,
    *,
    config_path: Path,
    context,
    contract: DailyMarkovContract,
    static: Mapping[str, Any],
    runtime_year: int,
):
    """Backward-compatible transition with a trace-static landpoint bundle."""

    return canonical_retained_tail_dynamic_transition(
        continuous_state,
        discrete_state,
        physical_fast_day_target,
        canonical_retained_tail_day_inputs(compiled_forcing, static),
        year,
        day_index,
        config_path=config_path,
        context=context,
        contract=contract,
        mineral_imin=int(static["mineral_imin"]),
        mineral_imax=int(static["mineral_imax"]),
        daily_carbon_dispatch=static["daily_carbon_dispatch"],
        metadata=static["metadata"],
        season_provenance=static["season"].provenance,
        runtime_year=runtime_year,
    )


def bind_canonical_retained_tail_dynamic_transition(
    *,
    config_path: Path,
    context,
    contract: DailyMarkovContract,
    trace_static: Mapping[str, Any],
    runtime_year: int,
):
    """Bind only shape/control identity; landpoint arrays remain call inputs."""

    def transition(
        continuous_state,
        discrete_state,
        physical_fast_day_target,
        day_inputs,
        year,
        day_index,
    ):
        return canonical_retained_tail_dynamic_transition(
            continuous_state,
            discrete_state,
            physical_fast_day_target,
            day_inputs,
            year,
            day_index,
            config_path=config_path,
            context=context,
            contract=contract,
            mineral_imin=int(trace_static["mineral_imin"]),
            mineral_imax=int(trace_static["mineral_imax"]),
            daily_carbon_dispatch=trace_static["daily_carbon_dispatch"],
            metadata=trace_static["metadata"],
            season_provenance=trace_static["season"].provenance,
            runtime_year=runtime_year,
        )

    return transition


def bind_canonical_retained_tail_transition(
    *,
    config_path: Path,
    context,
    contract: DailyMarkovContract,
    static: Mapping[str, Any],
    runtime_year: int,
):
    """Bind trace-static Teacher context to the multistep transition API."""

    def transition(
        continuous_state,
        discrete_state,
        physical_fast_day_target,
        compiled_forcing,
        year,
        day_index,
    ):
        return canonical_retained_tail_transition(
            continuous_state,
            discrete_state,
            physical_fast_day_target,
            compiled_forcing,
            year,
            day_index,
            config_path=config_path,
            context=context,
            contract=contract,
            static=static,
            runtime_year=runtime_year,
        )

    return transition
