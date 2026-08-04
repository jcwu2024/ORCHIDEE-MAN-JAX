"""Gate-D1 physical-parameter objectives over the canonical Teacher.

The wrapper reuses the production complete-day transition and only exposes
selected physical parameters and source-backed outputs to JAX transforms.  It
does not duplicate any ORCHIDEE scientific formula.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.physical_parameter_gradients import (
    GradientAcceptancePolicy,
    PhysicalGradientComparison,
    classify_physical_gradient_estimates,
)

jax.config.update("jax_enable_x64", True)


PFT14_INDEX = 13
STOMATE_PARAMETER_IDS = frozenset(
    {
        "alloc_min",
        "residence_time",
        "vcmax25",
        "maint_resp_slope_a",
        "maint_resp_slope_b",
        "maint_resp_slope_c",
    }
)
DIFFUCO_PARAMETER_IDS = frozenset({"g0"})
MODEL_OUTPUT_IDS = frozenset({"AGB_model", "BGB_model", "GPP_model", "NPP_model"})


class TeacherGradientOutputs(NamedTuple):
    modelout_fields: dict[str, object]
    modelout: object


@dataclass(frozen=True)
class TeacherGradientRuntime:
    """Fixed context plus dynamic parameter inputs for a complete-day scan."""

    transition: Callable
    initial_values: object
    forcing_days: object
    science_day_numbers: object
    stomate_parameters: teacher.DriverCompiledStomateParameterValues
    hydrol_table_arrays: object
    landpoint_payload: object
    stomate_restart_template: object
    stomate_season_values: object
    diffuco_parameters: teacher.DriverCompiledDiffucoParameterValues
    day_indices: tuple[int, ...]
    year: int


@dataclass(frozen=True)
class TeacherHalfHourGradientRuntime:
    """Prepared canonical later-day context for bounded cross-step AD checks."""

    config_path: Path
    context: object
    initial_state: object
    day_inputs: object
    baseline_parameters: teacher.DriverCompiledDiffucoParameterValues
    tstep: int
    year: int


@dataclass(frozen=True)
class TeacherGradientPairSpec:
    pair_id: str
    pair_class: str
    parameter_id: str
    output_id: str
    finite_difference_step: float
    output_day: int = -1


def _replace_stomate_parameter(
    values: teacher.DriverCompiledStomateParameterValues,
    parameter_id: str,
    scalar,
    *,
    pft_index: int = PFT14_INDEX,
) -> teacher.DriverCompiledStomateParameterValues:
    if parameter_id in {"alloc_min", "residence_time", "vcmax25"}:
        updated = jnp.asarray(getattr(values, parameter_id)).at[pft_index].set(scalar)
        return values._replace(**{parameter_id: updated})
    slope_columns = {
        "maint_resp_slope_c": 0,
        "maint_resp_slope_b": 1,
        "maint_resp_slope_a": 2,
    }
    if parameter_id in slope_columns:
        updated = jnp.asarray(values.maint_resp_slope).at[pft_index, slope_columns[parameter_id]].set(scalar)
        return values._replace(maint_resp_slope=updated)
    if parameter_id in DIFFUCO_PARAMETER_IDS:
        return values
    raise KeyError(f"unsupported STOMATE physical parameter {parameter_id!r}")


def _replace_diffuco_parameter(
    values: teacher.DriverCompiledDiffucoParameterValues,
    parameter_id: str,
    scalar,
) -> teacher.DriverCompiledDiffucoParameterValues:
    if parameter_id not in DIFFUCO_PARAMETER_IDS:
        return values
    trans_co2 = dict(values.trans_co2)
    trans_co2[parameter_id] = scalar
    return values._replace(trans_co2=trans_co2)


def parameter_reference_value(runtime: TeacherGradientRuntime, parameter_id: str) -> float:
    """Return the active PFT14/scalar value used by the prepared Teacher."""

    if parameter_id in {"alloc_min", "residence_time", "vcmax25"}:
        return float(np.asarray(getattr(runtime.stomate_parameters, parameter_id))[PFT14_INDEX])
    slope_columns = {
        "maint_resp_slope_c": 0,
        "maint_resp_slope_b": 1,
        "maint_resp_slope_a": 2,
    }
    if parameter_id in slope_columns:
        return float(
            np.asarray(runtime.stomate_parameters.maint_resp_slope)[PFT14_INDEX, slope_columns[parameter_id]]
        )
    if parameter_id in DIFFUCO_PARAMETER_IDS:
        return float(np.asarray(runtime.diffuco_parameters.trans_co2[parameter_id]))
    raise KeyError(f"unknown physical parameter {parameter_id!r}")


def _output_scalar(
    outputs: TeacherGradientOutputs,
    output_id: str,
    *,
    output_day: int,
    pft_index: int,
):
    if output_id in MODEL_OUTPUT_IDS:
        value = getattr(outputs.modelout, output_id)
    elif output_id in outputs.modelout_fields:
        value = outputs.modelout_fields[output_id]
    else:
        raise KeyError(f"unknown Teacher output {output_id!r}")
    array = jnp.asarray(value)
    if array.ndim < 3:
        raise ValueError(
            f"Teacher gradient output {output_id!r} must retain day, point, and PFT axes; got {array.shape}"
        )
    return array[output_day, 0, pft_index]


def physical_parameter_objective(
    runtime: TeacherGradientRuntime,
    *,
    parameter_id: str,
    output_id: str,
    output_day: int = -1,
    pft_index: int = PFT14_INDEX,
) -> Callable:
    """Build one scalar output-parameter objective over complete Teacher days."""

    if parameter_id not in STOMATE_PARAMETER_IDS | DIFFUCO_PARAMETER_IDS:
        raise KeyError(f"unknown physical parameter {parameter_id!r}")
    if output_day < -len(runtime.day_indices) or output_day >= len(runtime.day_indices):
        raise IndexError("output_day is outside the prepared Teacher horizon")

    def objective(scalar):
        stomate_parameters = _replace_stomate_parameter(
            runtime.stomate_parameters,
            parameter_id,
            scalar,
            pft_index=pft_index,
        )
        diffuco_parameters = _replace_diffuco_parameter(
            runtime.diffuco_parameters,
            parameter_id,
            scalar,
        )
        _, outputs = runtime.transition(
            runtime.initial_values,
            runtime.forcing_days,
            runtime.science_day_numbers,
            stomate_parameters,
            runtime.hydrol_table_arrays,
            runtime.landpoint_payload,
            runtime.stomate_restart_template,
            runtime.stomate_season_values,
            diffuco_parameters,
        )
        return _output_scalar(
            TeacherGradientOutputs(*outputs),
            output_id,
            output_day=output_day,
            pft_index=pft_index,
        )

    return objective


def physical_parameter_vector_objective(
    runtime: TeacherGradientRuntime,
    *,
    parameter_ids: tuple[str, ...],
    output_keys: tuple[tuple[str, int], ...],
    pft_index: int = PFT14_INDEX,
) -> Callable:
    """Build one vector objective so a single AD graph covers a pair matrix."""

    if len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("parameter_ids must be unique")
    if len(set(output_keys)) != len(output_keys):
        raise ValueError("output_keys must be unique")
    if not parameter_ids or not output_keys:
        raise ValueError("parameter_ids and output_keys must be non-empty")

    def objective(parameter_vector):
        stomate_parameters = runtime.stomate_parameters
        diffuco_parameters = runtime.diffuco_parameters
        for index, parameter_id in enumerate(parameter_ids):
            stomate_parameters = _replace_stomate_parameter(
                stomate_parameters,
                parameter_id,
                parameter_vector[index],
                pft_index=pft_index,
            )
            diffuco_parameters = _replace_diffuco_parameter(
                diffuco_parameters,
                parameter_id,
                parameter_vector[index],
            )
        _, outputs = runtime.transition(
            runtime.initial_values,
            runtime.forcing_days,
            runtime.science_day_numbers,
            stomate_parameters,
            runtime.hydrol_table_arrays,
            runtime.landpoint_payload,
            runtime.stomate_restart_template,
            runtime.stomate_season_values,
            diffuco_parameters,
        )
        typed = TeacherGradientOutputs(*outputs)
        return jnp.stack(
            [
                _output_scalar(
                    typed,
                    output_id,
                    output_day=output_day,
                    pft_index=pft_index,
                )
                for output_id, output_day in output_keys
            ]
        )

    return objective


def run_teacher_gradient_matrix(
    runtime: TeacherGradientRuntime,
    pair_specs: tuple[TeacherGradientPairSpec, ...],
    *,
    policy: GradientAcceptancePolicy | None = None,
) -> tuple[PhysicalGradientComparison, ...]:
    """Evaluate a declared pair matrix with one forward and one reverse graph."""

    if not pair_specs:
        raise ValueError("pair_specs must be non-empty")
    pair_ids = [spec.pair_id for spec in pair_specs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair IDs must be unique")
    parameter_ids = tuple(dict.fromkeys(spec.parameter_id for spec in pair_specs))
    output_keys = tuple(dict.fromkeys((spec.output_id, spec.output_day) for spec in pair_specs))
    parameter_index = {name: index for index, name in enumerate(parameter_ids)}
    output_index = {key: index for index, key in enumerate(output_keys)}
    baseline = jnp.asarray(
        [parameter_reference_value(runtime, name) for name in parameter_ids],
        dtype=jnp.float64,
    )
    objective = physical_parameter_vector_objective(
        runtime,
        parameter_ids=parameter_ids,
        output_keys=output_keys,
    )
    objective_jit = jax.jit(objective)
    primal = np.asarray(jax.device_get(objective_jit(baseline)), dtype=np.float64)
    def projected_jvp(parameters, tangent):
        value, derivative = jax.jvp(objective, (parameters,), (tangent,))
        return value, derivative

    def projected_vjp(parameters, cotangent):
        value, pullback = jax.vjp(objective, parameters)
        return value, pullback(cotangent)[0]

    projected_jvp_jit = jax.jit(projected_jvp)
    projected_vjp_jit = jax.jit(projected_vjp)
    forward_columns = []
    for index in range(len(parameter_ids)):
        tangent = jnp.zeros_like(baseline).at[index].set(1.0)
        _, derivative = projected_jvp_jit(baseline, tangent)
        forward_columns.append(np.asarray(jax.device_get(derivative), dtype=np.float64))
    forward = np.stack(forward_columns, axis=1)
    reverse_rows = []
    for index in range(len(output_keys)):
        cotangent = jnp.zeros((len(output_keys),), dtype=jnp.float64).at[index].set(1.0)
        _, derivative = projected_vjp_jit(baseline, cotangent)
        reverse_rows.append(np.asarray(jax.device_get(derivative), dtype=np.float64))
    reverse = np.stack(reverse_rows, axis=0)
    if forward.shape != (len(output_keys), len(parameter_ids)) or reverse.shape != forward.shape:
        raise RuntimeError(
            f"unexpected Teacher Jacobian shapes: forward={forward.shape}, reverse={reverse.shape}"
        )

    fd_by_parameter: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for parameter_id in parameter_ids:
        index = parameter_index[parameter_id]
        steps = {spec.finite_difference_step for spec in pair_specs if spec.parameter_id == parameter_id}
        if len(steps) != 1:
            raise ValueError(f"parameter {parameter_id!r} has inconsistent finite-difference steps")
        step = float(next(iter(steps)))
        direction = jnp.zeros_like(baseline).at[index].set(step)
        plus = np.asarray(jax.device_get(objective_jit(baseline + direction)), dtype=np.float64)
        minus = np.asarray(jax.device_get(objective_jit(baseline - direction)), dtype=np.float64)
        fd = (plus - minus) / (2.0 * step)
        half_direction = direction / 2.0
        plus_half = np.asarray(jax.device_get(objective_jit(baseline + half_direction)), dtype=np.float64)
        minus_half = np.asarray(jax.device_get(objective_jit(baseline - half_direction)), dtype=np.float64)
        fd_half = (plus_half - minus_half) / step
        fd_by_parameter[parameter_id] = (fd, fd_half)

    results = []
    for spec in pair_specs:
        row = output_index[(spec.output_id, spec.output_day)]
        column = parameter_index[spec.parameter_id]
        fd, fd_half = fd_by_parameter[spec.parameter_id]
        results.append(
            classify_physical_gradient_estimates(
                pair_id=spec.pair_id,
                pair_class=spec.pair_class,
                parameter_value=float(np.asarray(baseline[column])),
                finite_difference_step=spec.finite_difference_step,
                primal_value=float(primal[row]),
                forward_ad=float(forward[row, column]),
                reverse_ad=float(reverse[row, column]),
                central_difference=float(fd[row]),
                central_difference_half_step=float(fd_half[row]),
                policy=policy,
            )
        )
    return tuple(results)


def run_teacher_gradient_pair(
    runtime: TeacherGradientRuntime,
    *,
    pair_id: str,
    pair_class: str,
    parameter_id: str,
    output_id: str,
    finite_difference_step: float,
    output_day: int = -1,
    policy: GradientAcceptancePolicy | None = None,
) -> PhysicalGradientComparison:
    """Run memory-bounded JIT stages for one complete-Teacher pair."""

    objective = physical_parameter_objective(
        runtime,
        parameter_id=parameter_id,
        output_id=output_id,
        output_day=output_day,
    )
    parameter_value = parameter_reference_value(runtime, parameter_id)
    value = jnp.asarray(parameter_value, dtype=jnp.float64)
    step = jnp.asarray(finite_difference_step, dtype=jnp.float64)

    primal = jax.jit(objective)(value)
    forward = jax.jit(jax.jacfwd(objective))(value)
    reverse = jax.jit(jax.jacrev(objective))(value)

    def finite_difference(candidate, delta):
        return (objective(candidate + delta) - objective(candidate - delta)) / (2.0 * delta)

    finite_difference_jit = jax.jit(finite_difference)
    fd = finite_difference_jit(value, step)
    fd_half = finite_difference_jit(value, step / 2.0)
    primal, forward, reverse, fd, fd_half = jax.device_get(
        (primal, forward, reverse, fd, fd_half)
    )
    return classify_physical_gradient_estimates(
        pair_id=pair_id,
        pair_class=pair_class,
        parameter_value=parameter_value,
        finite_difference_step=finite_difference_step,
        primal_value=float(np.asarray(primal)),
        forward_ad=float(np.asarray(forward)),
        reverse_ad=float(np.asarray(reverse)),
        central_difference=float(np.asarray(fd)),
        central_difference_half_step=float(np.asarray(fd_half)),
        policy=policy,
    )


def _build_transition(
    *,
    config_path: Path,
    context,
    initial_state,
    runtime_spec,
    prebound_tables,
    daily_carbon_dispatch,
    season_provenance,
    state_tstep: int,
    metadata_day_index: int,
    metadata_year: int,
    start_tstep: int,
) -> Callable:
    mineral_imin = int(prebound_tables.mineral.imin)
    mineral_imax = int(prebound_tables.mineral.imax)

    def transition(
        values_by_component,
        forcing_days,
        science_day_numbers,
        stomate_parameters,
        hydrol_table_arrays,
        landpoint_payload,
        stomate_restart_template,
        stomate_season_values,
        diffuco_parameters,
    ):
        def body(current_values, inputs):
            forcing, science_day_number = inputs
            state = teacher.DriverFastStateBundle(
                tstep=state_tstep,
                values_by_component=current_values,
                spec=runtime_spec,
            )
            day = teacher.paper_1961_driver_later_day_runtime_result(
                config_path,
                previous_state=state,
                day_index=metadata_day_index,
                year=metadata_year,
                start_tstep=start_tstep,
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                module_jit=True,
                diffuco_local_jit=True,
                use_static_jit_daily_carbon=False,
                prebuild_day_payloads=True,
                use_compiled_sechiba_day=True,
                prebound_hydrol_runtime_static_tables=teacher.HydrolRuntimeStaticTables(
                    mineral=teacher.MineralCWRRTables(
                        **hydrol_table_arrays._asdict(),
                        imin=mineral_imin,
                        imax=mineral_imax,
                    ),
                    peat=None,
                ),
                materialize_compiled_entries=False,
                outer_compiled_daily_carbon_dispatch=daily_carbon_dispatch,
                compiled_forcing_series=forcing,
                model_day_number=science_day_number,
                compiled_stomate_parameter_values=stomate_parameters,
                compiled_landpoint_payload=landpoint_payload,
                compiled_stomate_restart_template=stomate_restart_template,
                compiled_stomate_season_template=teacher.StomateRestartSeasonState(
                    **stomate_season_values,
                    provenance=season_provenance,
                ),
                compiled_diffuco_parameter_values=diffuco_parameters,
            )
            packet = day.day_end_state
            next_values = tuple(
                tuple(
                    packet.fields_by_component[component][name]
                    for name in field_names
                )
                for component, field_names in zip(
                    runtime_spec.components,
                    runtime_spec.field_names_by_component,
                    strict=True,
                )
            )
            return next_values, (
                day.daily_modelout.modelout_fields,
                day.daily_modelout.modelout,
            )

        return jax.lax.scan(
            body,
            values_by_component,
            (forcing_days, science_day_numbers),
        )

    return transition


def prepare_later_day_gradient_runtime(
    config_path: str | Path,
    *,
    used_run_def_path: str | Path,
    seed_days: int = 1,
    horizon: int = 1,
    year: int = 1961,
) -> TeacherGradientRuntime:
    """Prepare real cold-continuation Teacher days for Gate-D1 transforms."""

    if seed_days < 1:
        raise ValueError("seed_days must be at least one")
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    config_path = Path(config_path)
    context = teacher.prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
    )
    seed = teacher.paper_1961_driver_multiday_modelout_lite_run(
        config_path,
        ndays=seed_days,
        year=year,
        used_run_def_path=used_run_def_path,
        module_jit=True,
        diffuco_local_jit=True,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    if not seed.ready_for_requested_days or seed.last_day_end_state is None:
        raise RuntimeError(f"Teacher seed failed: {seed.missing_components}")
    initial_state = seed.last_day_end_state
    initial = teacher.fast_state_from_previous_packet(initial_state)
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    first_day_index = seed_days + 1
    day_indices = tuple(range(first_day_index, first_day_index + horizon))
    first_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=initial_state,
        year=year,
        start=(first_day_index - 1) * steps_per_day,
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    prebound_tables = first_inputs.hydrol_runtime_static_tables
    daily_carbon_dispatch = teacher._paper_daily_carbon_static_dispatch(context, initial_state)
    season_state = context.first_step_restart_state.stomate_readstart.season_state
    season_values = {
        name: value
        for name, value in season_state._asdict().items()
        if name != "provenance"
    }
    forcing_days = jax.tree_util.tree_map(
        lambda *values: np.stack(values),
        *(
            teacher._paper_compiled_forcing_day(
                context,
                year=year,
                start_tstep=(day_index - 1) * steps_per_day,
                steps_per_stomate=steps_per_day,
            )
            for day_index in day_indices
        ),
    )
    transition = _build_transition(
        config_path=config_path,
        context=context,
        initial_state=initial_state,
        runtime_spec=initial.spec,
        prebound_tables=prebound_tables,
        daily_carbon_dispatch=daily_carbon_dispatch,
        season_provenance=season_state.provenance,
        state_tstep=47,
        metadata_day_index=2,
        metadata_year=1961,
        start_tstep=48,
    )
    return TeacherGradientRuntime(
        transition=transition,
        initial_values=initial.values_by_component,
        forcing_days=forcing_days,
        science_day_numbers=np.asarray(day_indices, dtype=np.int32),
        stomate_parameters=teacher._compiled_stomate_parameter_values(context),
        hydrol_table_arrays=teacher._compiled_hydrol_table_arrays(prebound_tables),
        landpoint_payload=teacher._compiled_landpoint_payload(first_inputs.compiled_base_payload_template),
        stomate_restart_template=context.first_step_restart_state.stomate,
        stomate_season_values=season_values,
        diffuco_parameters=teacher._compiled_diffuco_parameter_values(context),
        day_indices=day_indices,
        year=int(year),
    )


def prepare_restart_year_gradient_runtime(
    config_path: str | Path,
    *,
    used_run_def_path: str | Path,
    seed_days: int = 3,
    source_year: int = 1961,
    restart_year: int = 1962,
) -> TeacherGradientRuntime:
    """Prepare the first complete Teacher day after a yearly restart rebase."""

    if seed_days < 1:
        raise ValueError("seed_days must be at least one")
    config_path = Path(config_path)
    context = teacher.prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
    )
    seed = teacher.paper_1961_driver_multiday_modelout_lite_run(
        config_path,
        ndays=seed_days,
        year=source_year,
        used_run_def_path=used_run_def_path,
        module_jit=True,
        diffuco_local_jit=True,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    if not seed.ready_for_requested_days or seed.last_day_end_state is None:
        raise RuntimeError(f"Teacher restart seed failed: {seed.missing_components}")
    initial_state = teacher.rebase_driver_state_for_year_start(seed.last_day_end_state)
    initial = teacher.fast_state_from_previous_packet(initial_state)
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    first_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=initial_state,
        year=restart_year,
        start=0,
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    prebound_tables = first_inputs.hydrol_runtime_static_tables
    daily_carbon_dispatch = teacher._paper_daily_carbon_static_dispatch(context, initial_state)
    season_state = context.first_step_restart_state.stomate_readstart.season_state
    season_values = {
        name: value
        for name, value in season_state._asdict().items()
        if name != "provenance"
    }
    forcing_days = jax.tree_util.tree_map(
        lambda value: np.asarray(value)[None, ...],
        teacher._paper_compiled_forcing_day(
            context,
            year=restart_year,
            start_tstep=0,
            steps_per_stomate=steps_per_day,
        ),
    )
    transition = _build_transition(
        config_path=config_path,
        context=context,
        initial_state=initial_state,
        runtime_spec=initial.spec,
        prebound_tables=prebound_tables,
        daily_carbon_dispatch=daily_carbon_dispatch,
        season_provenance=season_state.provenance,
        state_tstep=-1,
        metadata_day_index=1,
        metadata_year=restart_year,
        start_tstep=0,
    )
    return TeacherGradientRuntime(
        transition=transition,
        initial_values=initial.values_by_component,
        forcing_days=forcing_days,
        science_day_numbers=np.asarray([1], dtype=np.int32),
        stomate_parameters=teacher._compiled_stomate_parameter_values(context),
        hydrol_table_arrays=teacher._compiled_hydrol_table_arrays(prebound_tables),
        landpoint_payload=teacher._compiled_landpoint_payload(first_inputs.compiled_base_payload_template),
        stomate_restart_template=context.first_step_restart_state.stomate,
        stomate_season_values=season_values,
        diffuco_parameters=teacher._compiled_diffuco_parameter_values(context),
        day_indices=(1,),
        year=int(restart_year),
    )


def prepare_later_day_half_hour_gradient_runtime(
    config_path: str | Path,
    *,
    used_run_def_path: str | Path,
    seed_days: int = 1,
    year: int = 1961,
) -> TeacherHalfHourGradientRuntime:
    """Prepare one real later-day context shared by bounded AD objectives."""

    if seed_days < 1:
        raise ValueError("seed_days must be at least one")
    config_path = Path(config_path)
    context = teacher.prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
    )
    seed = teacher.paper_1961_driver_multiday_modelout_lite_run(
        config_path,
        ndays=seed_days,
        year=year,
        used_run_def_path=used_run_def_path,
        module_jit=True,
        diffuco_local_jit=True,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    if not seed.ready_for_requested_days or seed.last_day_end_state is None:
        raise RuntimeError(f"Teacher seed failed: {seed.missing_components}")
    initial_state = seed.last_day_end_state
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    tstep = seed_days * steps_per_day
    day_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=initial_state,
        year=year,
        start=tstep,
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    if not day_inputs.half_hour_inputs:
        raise RuntimeError("single-step diagnosis requires prebuilt half-hour inputs")
    return TeacherHalfHourGradientRuntime(
        config_path=config_path,
        context=context,
        initial_state=initial_state,
        day_inputs=day_inputs,
        baseline_parameters=teacher._compiled_diffuco_parameter_values(context),
        tstep=tstep,
        year=int(year),
    )


def _advance_half_hour_gradient_runtime(
    runtime: TeacherHalfHourGradientRuntime,
    *,
    current_state,
    offset: int,
    parameter_values: teacher.DriverCompiledDiffucoParameterValues,
    module_jit: bool,
    diffuco_local_jit: bool,
):
    return teacher._paper_1961_next_step_runtime_result_compact(
        runtime.config_path,
        previous_state=current_state,
        year=runtime.year,
        tstep=runtime.tstep + offset,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        used_run_def_path=runtime.context.run_def_path,
        prepared_context=runtime.context,
        hydrol_static_template=runtime.day_inputs.hydrol_static_template,
        hydrol_runtime_static_tables=runtime.day_inputs.hydrol_runtime_static_tables,
        diffuco_day_static_cache=runtime.day_inputs.diffuco_day_static_cache,
        compiled_diffuco_static_kwargs=teacher._compiled_diffuco_static_kwargs(
            runtime.context,
            parameter_values,
        ),
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
        prebuilt_input=runtime.day_inputs.half_hour_inputs[offset],
    )


def prepare_later_day_single_step_objective(
    config_path: str | Path,
    *,
    used_run_def_path: str | Path,
    parameter_id: str,
    output_id: str,
    seed_days: int = 1,
    half_hour_steps: int = 1,
    year: int = 1961,
    module_jit: bool = False,
    diffuco_local_jit: bool = False,
    stop_gradient_between_steps: bool = False,
    return_fast_state: bool = False,
    state_output: tuple[str, str, tuple[int, ...]] | None = None,
) -> tuple[Callable, float]:
    """Build a real later-day half-hour-prefix scalar objective for AD diagnosis.

    The state is produced by the canonical complete-day Teacher.  The returned
    objective advances a declared prefix of the following day and sums one
    source-owned STOMATE-entry field without duplicating formulas.
    """

    if parameter_id not in DIFFUCO_PARAMETER_IDS:
        raise KeyError("single-step diagnosis currently supports DIFFUCO parameters")
    if not 1 <= half_hour_steps <= 48:
        raise ValueError("half_hour_steps must be between 1 and 48")
    if return_fast_state and state_output is not None:
        raise ValueError("return_fast_state and state_output are mutually exclusive")
    runtime = prepare_later_day_half_hour_gradient_runtime(
        config_path,
        used_run_def_path=used_run_def_path,
        seed_days=seed_days,
        year=year,
    )
    baseline = float(np.asarray(runtime.baseline_parameters.trans_co2[parameter_id]))

    def objective(scalar):
        parameter_values = _replace_diffuco_parameter(
            runtime.baseline_parameters,
            parameter_id,
            scalar,
        )
        current_state = runtime.initial_state
        total = jnp.asarray(0.0, dtype=jnp.float64)
        for offset in range(half_hour_steps):
            result = _advance_half_hour_gradient_runtime(
                runtime,
                current_state=current_state,
                offset=offset,
                parameter_values=parameter_values,
                module_jit=module_jit,
                diffuco_local_jit=diffuco_local_jit,
            )
            if not result.ok or result.entry_payload is None:
                raise RuntimeError(f"half-hour Teacher failed: {result.missing_components}")
            value = jnp.asarray(result.entry_payload[output_id])
            if value.ndim < 2:
                raise ValueError(f"entry output {output_id!r} must have point and PFT axes")
            total = total + value[0, PFT14_INDEX]
            current_state = result.next_state
            if stop_gradient_between_steps and offset + 1 < half_hour_steps:
                fast_state = teacher.fast_state_from_previous_packet(current_state)
                current_state = teacher.previous_packet_from_fast_state(
                    replace(
                        fast_state,
                        values_by_component=jax.tree_util.tree_map(
                            jax.lax.stop_gradient,
                            fast_state.values_by_component,
                        ),
                    )
                )
        if return_fast_state:
            return teacher.fast_state_from_previous_packet(current_state)
        if state_output is not None:
            component, field, index = state_output
            fast_state = teacher.fast_state_from_previous_packet(current_state)
            return jnp.asarray(fast_state.component_fields(component)[field])[index]
        return total

    return objective, baseline


def prepare_later_day_cross_step_state_objectives(
    config_path: str | Path,
    *,
    used_run_def_path: str | Path,
    parameter_id: str,
    output_id: str,
    seed_days: int = 1,
    year: int = 1961,
) -> tuple[Callable, Callable, object, float]:
    """Expose the exact first-state and second-output maps around one carry."""

    if parameter_id not in DIFFUCO_PARAMETER_IDS:
        raise KeyError("cross-step diagnosis currently supports DIFFUCO parameters")
    runtime = prepare_later_day_half_hour_gradient_runtime(
        config_path,
        used_run_def_path=used_run_def_path,
        seed_days=seed_days,
        year=year,
    )
    baseline = float(np.asarray(runtime.baseline_parameters.trans_co2[parameter_id]))

    def first_state(scalar):
        parameter_values = _replace_diffuco_parameter(
            runtime.baseline_parameters,
            parameter_id,
            scalar,
        )
        result = _advance_half_hour_gradient_runtime(
            runtime,
            current_state=runtime.initial_state,
            offset=0,
            parameter_values=parameter_values,
            module_jit=False,
            diffuco_local_jit=False,
        )
        if not result.ok:
            raise RuntimeError(f"first cross-step Teacher call failed: {result.missing_components}")
        return teacher.fast_state_from_previous_packet(result.next_state)

    baseline_state = first_state(jnp.asarray(baseline, dtype=jnp.float64))

    def second_output(values_by_component):
        current_state = teacher.previous_packet_from_fast_state(
            teacher.DriverFastStateBundle(
                tstep=runtime.tstep,
                values_by_component=values_by_component,
                spec=baseline_state.spec,
            )
        )
        result = _advance_half_hour_gradient_runtime(
            runtime,
            current_state=current_state,
            offset=1,
            parameter_values=runtime.baseline_parameters,
            module_jit=False,
            diffuco_local_jit=False,
        )
        if not result.ok or result.entry_payload is None:
            raise RuntimeError(f"second cross-step Teacher call failed: {result.missing_components}")
        value = jnp.asarray(result.entry_payload[output_id])
        return value[0, PFT14_INDEX]

    return first_state, second_output, baseline_state, baseline
