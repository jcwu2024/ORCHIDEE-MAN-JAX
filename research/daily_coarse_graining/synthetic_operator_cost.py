"""Cost gate for a dynamic, neural-network-shaped daily coarse operator.

The synthetic operator is deliberately untrained and has no scientific skill.
It exists only to measure whether a realistic array program plus the retained
Teacher tail can retain a material CPU speed advantage over the complete-day
Teacher.  Teacher targets are never operator inputs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, Sequence
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.boundary_adapter import (
    HYDROL_COMPONENT,
    NROOT_FIELD,
    packet_from_projected_values,
    project_packet_values,
    promote_runtime_state_spec,
)
from research.daily_coarse_graining.persistence_baseline import (
    COMMON_OK_LEAK_FIELDS,
    _forcing_days,
    _ok_leak_output_mapping,
    _persistence_daily_interface,
    _persistence_ok_leak_interface,
    _unstack,
)
from research.daily_coarse_graining.replay_ceiling import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_DEF,
    _block_until_ready,
    _load_state_cache,
    _minimal_daily_fold,
    _teacher_sequence,
    _time_call,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "performance"
    / "daily_coarse_graining"
    / "synthetic_operator_cost_gate.json"
)

FORCING_FIELDS = (
    "zlev",
    "zlevuv",
    "u",
    "v",
    "qair",
    "temp_air",
    "pb",
    "precip_rain",
    "precip_snow",
    "lwdown",
    "swdown",
    "ccanopy",
    "salinity",
    "tide_height",
)

COARSE_STATE_COMPONENTS = (
    "driver_previous_step_state",
    "diffuco_previous_step_state",
    "enerbil_previous_step_state",
    HYDROL_COMPONENT,
    "thermosoil_previous_step_state",
    "sechiba_finalize_state",
)


class SyntheticOperatorParameters(NamedTuple):
    forcing_weight: Any
    forcing_bias: Any
    state_weight: Any
    state_bias: Any
    hidden1_weight: Any
    hidden1_bias: Any
    hidden2_weight: Any
    hidden2_bias: Any
    hidden3_weight: Any
    hidden3_bias: Any
    output_weight: Any
    output_bias: Any


@dataclass(frozen=True)
class SyntheticExecutable:
    executable: Any
    block_size: int
    first_day: bool
    input_spec: Any
    output_spec: Any
    stomate_parameter_values: Any
    hydrol_table_arrays: Any
    landpoint_payload: Any
    stomate_restart_template: Any
    stomate_season_values: Any
    diffuco_parameter_values: Any

    def run(self, previous_state, forcing_days, day_numbers, operator_parameters):
        fast = teacher.fast_state_from_previous_packet(previous_state)
        if fast.spec != self.input_spec:
            raise ValueError("synthetic executable input packet does not match its fixed spec")
        return self.executable(
            fast.values_by_component,
            forcing_days,
            day_numbers,
            operator_parameters,
            self.stomate_parameter_values,
            self.hydrol_table_arrays,
            self.landpoint_payload,
            self.stomate_restart_template,
            self.stomate_season_values,
            self.diffuco_parameter_values,
        )


@dataclass(frozen=True)
class SyntheticSequenceResult:
    last_day_end_state: Any
    daily_outputs: tuple[tuple[Any, Any], ...]
    validation_checksums: tuple[Any, ...]


def _is_float_array(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return np.issubdtype(np.dtype(dtype), np.floating)


def _operator_output_width(state) -> int:
    state_heads = 0
    for component in COARSE_STATE_COMPONENTS:
        for name, value in state.fields_by_component[component].items():
            if component == HYDROL_COMPONENT and name == NROOT_FIELD:
                continue
            if _is_float_array(value):
                state_heads += 1
    state_heads += 1  # nroot is produced even when the year-start packet omits it.
    return state_heads + 17 + len(COMMON_OK_LEAK_FIELDS) + 1 + 2


def initialize_operator_parameters(output_width: int, *, seed: int = 20260721):
    """Create dynamic arrays representative of a small encoder/MLP operator."""

    rng = np.random.default_rng(seed)

    def weight(shape, scale):
        return rng.normal(0.0, scale, size=shape).astype(np.float64)

    return SyntheticOperatorParameters(
        forcing_weight=weight((len(FORCING_FIELDS), 64), 0.08),
        forcing_bias=weight((64,), 0.02),
        state_weight=weight((16, 64), 0.08),
        state_bias=weight((64,), 0.02),
        hidden1_weight=weight((192, 128), 0.05),
        hidden1_bias=weight((128,), 0.02),
        hidden2_weight=weight((128, 128), 0.05),
        hidden2_bias=weight((128,), 0.02),
        hidden3_weight=weight((128, 64), 0.05),
        hidden3_bias=weight((64,), 0.02),
        output_weight=weight((64, output_width), 0.04),
        output_bias=weight((output_width,), 0.01),
    )


def _forcing_matrix(forcing):
    columns = []
    for name in FORCING_FIELDS:
        value = jnp.asarray(getattr(forcing, name))
        columns.append(jnp.mean(value.reshape((value.shape[0], -1)), axis=1))
    matrix = jnp.stack(columns, axis=-1)
    return matrix / (1.0 + jnp.abs(matrix))


def _state_feature_vector(state):
    fields = state.fields_by_component
    sources = (
        fields[HYDROL_COMPONENT]["mc"],
        fields[HYDROL_COMPONENT]["mcl"],
        fields[HYDROL_COMPONENT]["us"],
        fields[HYDROL_COMPONENT]["snow"],
        fields["thermosoil_previous_step_state"]["ptn"],
        fields["diffuco_previous_step_state"]["gpp"],
        fields["enerbil_previous_step_state"]["temp_sol"],
        fields["slowproc_stomate_previous_step_state"]["biomass"],
    )
    features = []
    for value in sources:
        array = jnp.asarray(value, dtype=jnp.float64)
        scaled = array / (1.0 + jnp.abs(array))
        features.extend((jnp.mean(scaled), jnp.std(scaled)))
    return jnp.stack(features)


def _operator_head(state, forcing, parameters):
    forcing_encoded = jnp.tanh(
        _forcing_matrix(forcing) @ parameters.forcing_weight + parameters.forcing_bias
    )
    forcing_summary = jnp.concatenate(
        (jnp.mean(forcing_encoded, axis=0), jnp.max(forcing_encoded, axis=0))
    )
    state_encoded = jnp.tanh(
        _state_feature_vector(state) @ parameters.state_weight + parameters.state_bias
    )
    hidden = jnp.tanh(
        jnp.concatenate((forcing_summary, state_encoded)) @ parameters.hidden1_weight
        + parameters.hidden1_bias
    )
    hidden = jnp.tanh(hidden @ parameters.hidden2_weight + parameters.hidden2_bias)
    hidden = jnp.tanh(hidden @ parameters.hidden3_weight + parameters.hidden3_bias)
    return hidden @ parameters.output_weight + parameters.output_bias


def _residual(value, coefficient):
    array = jnp.asarray(value)
    return array + jnp.asarray(1.0e-6, dtype=array.dtype) * jnp.tanh(coefficient) * (
        jnp.abs(array) + jnp.asarray(1.0, dtype=array.dtype)
    )


def _synthetic_nroot(state, coefficient):
    """Produce dynamic root fractions from the current root-zone state."""

    us = jnp.asarray(state.fields_by_component[HYDROL_COMPONENT]["us"])
    rooted = jnp.mean(jnp.abs(us), axis=2)
    layer_signal = jnp.linspace(-1.0, 1.0, rooted.shape[-1], dtype=rooted.dtype)
    logits = rooted / (1.0 + rooted) + 0.05 * jnp.tanh(coefficient) * layer_signal
    nroot = jax.nn.softmax(logits, axis=-1)
    present = jnp.any(jnp.abs(us) > 0.0, axis=(2, 3))
    return jnp.where(present[:, :, None], nroot, 0.0)


def _perturb_state(state, head, cursor: int):
    output = {component: dict(fields) for component, fields in state.fields_by_component.items()}
    for component in COARSE_STATE_COMPONENTS:
        for name, value in state.fields_by_component[component].items():
            if component == HYDROL_COMPONENT and name == NROOT_FIELD:
                continue
            if _is_float_array(value):
                output[component][name] = _residual(value, head[cursor])
                cursor += 1
    output[HYDROL_COMPONENT][NROOT_FIELD] = _synthetic_nroot(state, head[cursor])
    cursor += 1
    return teacher.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=output,
        provenance_by_component=state.provenance_by_component,
    ), cursor


def _perturb_daily_fields(state, forcing, head, cursor: int, *, min_wind: float):
    base = _persistence_daily_interface(state, forcing, min_wind=min_wind)
    result = {}
    for name, value in base.items():
        result[name] = _residual(value, head[cursor])
        cursor += 1
    return result, cursor


def _perturb_ok_leak(state, head, cursor: int):
    base = _persistence_ok_leak_interface(state)

    def next_value(value):
        nonlocal cursor
        result = _residual(value, head[cursor])
        cursor += 1
        return result

    litter = base.littercalc
    soil = base.soilcarbon
    result = SimpleNamespace(
        littercalc=SimpleNamespace(
            litter_above=next_value(litter.litter_above),
            litter_below=next_value(litter.litter_below),
            lignin_struc_above=next_value(litter.lignin_struc_above),
            lignin_struc_below=next_value(litter.lignin_struc_below),
            litterpart=next_value(litter.litterpart),
            dead_leaves=next_value(litter.dead_leaves),
            fuel=SimpleNamespace(
                fuel_1hr=next_value(litter.fuel.fuel_1hr),
                fuel_10hr=next_value(litter.fuel.fuel_10hr),
                fuel_100hr=next_value(litter.fuel.fuel_100hr),
                fuel_1000hr=next_value(litter.fuel.fuel_1000hr),
            ),
        ),
        soilcarbon=SimpleNamespace(
            carbon_32l=next_value(soil.carbon_32l),
            doc=next_value(soil.doc),
            resp_hetero_soil=soil.resp_hetero_soil,
            perma_peat=SimpleNamespace(deepc_peat=next_value(soil.perma_peat.deepc_peat)),
        ),
        interception_storage=next_value(base.interception_storage),
    )
    return result, cursor


def _checksum_arrays(value):
    leaves = jax.tree_util.tree_leaves(value)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if np.issubdtype(np.dtype(array.dtype), np.number) or np.issubdtype(
            np.dtype(array.dtype), np.bool_
        ):
            finite = jnp.nan_to_num(
                jnp.asarray(array, dtype=jnp.float64),
                nan=0.0,
                posinf=1.0e6,
                neginf=-1.0e6,
            )
            total = total + jnp.mean(finite / (1.0 + jnp.abs(finite)))
    return total


def synthetic_boundary(state, forcing, metadata, parameters, *, min_wind: float):
    """Build the complete synthetic pre-daily-STOMATE boundary."""

    head = _operator_head(state, forcing, parameters)
    cursor = 0
    end_state, cursor = _perturb_state(state, head, cursor)
    daily_fields, cursor = _perturb_daily_fields(
        state, forcing, head, cursor, min_wind=min_wind
    )
    ok_leak, cursor = _perturb_ok_leak(state, head, cursor)
    final_t2m = _residual(jnp.asarray(forcing.temp_air)[-1], head[cursor])
    cursor += 1
    final_temp_sol = _residual(
        end_state.fields_by_component["enerbil_previous_step_state"]["temp_sol"],
        head[cursor],
    )
    cursor += 1
    if cursor != head.shape[0]:
        raise ValueError(f"synthetic head width mismatch: consumed {cursor}, got {head.shape[0]}")

    stempdiag = end_state.fields_by_component["thermosoil_previous_step_state"][
        "stempdiag"
    ]
    transition = teacher.DriverRuntimeDayStepTransition(
        completed_entry_payloads=(
            {
                "t2m": final_t2m,
                "stempdiag": stempdiag,
                "t2mdiag": final_t2m,
                "temp_sol": final_temp_sol,
            },
        ),
        current_state=end_state,
        first_step_metadata=metadata,
        stopped_at_tstep=None,
        state_gaps=(),
        compiled_entry_stacks={"t2m": final_t2m, "stempdiag": stempdiag},
    )
    checksum = (
        _checksum_arrays(
            {
                component: end_state.fields_by_component[component]
                for component in COARSE_STATE_COMPONENTS
            }
        )
        + _checksum_arrays(daily_fields)
        + _checksum_arrays(_ok_leak_output_mapping(ok_leak))
        + _checksum_arrays((final_t2m, final_temp_sol))
        + jnp.sum(jnp.tanh(head))
    )
    return transition, _minimal_daily_fold(daily_fields), ok_leak, checksum


def _tail_static_inputs(context, initial_state, sample_forcing, *, first_day: bool):
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    start_tstep = 0 if first_day else 48
    first_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=initial_state,
        year=1962,
        start=start_tstep,
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
        compiled_forcing_series=jax.tree_util.tree_map(lambda value: value[0], sample_forcing),
    )
    tables = first_inputs.hydrol_runtime_static_tables
    season = teacher.read_stomate_restart_season_state(
        context.first_step_restart_state.stomate_input
    )
    run_scalars = first_inputs.day_start_bundle.run_scalars
    template = first_inputs.compiled_base_payload_template
    metadata = teacher.DriverRuntimeStepMetadata(
        pref_soil_veg=run_scalars.pref_soil_veg,
        nstm=run_scalars.nstm,
        is_tree=run_scalars.is_tree,
        is_peat=run_scalars.is_peat,
        lalo=template.lalo,
        npts=template.kjpindex,
    )
    return {
        "tables": tables,
        "mineral_imin": int(tables.mineral.imin),
        "mineral_imax": int(tables.mineral.imax),
        "daily_carbon_dispatch": teacher._paper_daily_carbon_static_dispatch(
            context, initial_state
        ),
        "stomate_parameter_values": teacher._compiled_stomate_parameter_values(context),
        "landpoint_payload": teacher._compiled_landpoint_payload(template),
        "stomate_restart_template": context.first_step_restart_state.stomate,
        "season": season,
        "stomate_season_values": {
            name: value for name, value in season._asdict().items() if name != "provenance"
        },
        "diffuco_parameter_values": teacher._compiled_diffuco_parameter_values(context),
        "hydrol_table_arrays": teacher._compiled_hydrol_table_arrays(tables),
        "metadata": metadata,
        "start_tstep": start_tstep,
    }


def _compile_synthetic_executable(
    *,
    config_path: Path,
    context,
    initial_state,
    sample_forcing,
    sample_day_numbers,
    operator_parameters,
    first_day: bool,
):
    block_size = int(np.asarray(sample_day_numbers).shape[0])
    if block_size < 1 or (first_day and block_size != 1):
        raise ValueError("first-day synthetic executable must contain exactly one day")
    initial = teacher.fast_state_from_previous_packet(initial_state)
    static = _tail_static_inputs(context, initial_state, sample_forcing, first_day=first_day)
    if first_day:
        sample_boundary = synthetic_boundary(
            initial_state,
            jax.tree_util.tree_map(lambda value: value[0], sample_forcing),
            static["metadata"],
            operator_parameters,
            min_wind=context.min_wind,
        )[0]
        output_spec = promote_runtime_state_spec(
            initial_state, sample_boundary.current_state
        )
    else:
        if NROOT_FIELD not in initial_state.fields_by_component[HYDROL_COMPONENT]:
            raise ValueError("later-day synthetic executable requires produced hydrol.nroot")
        output_spec = initial.spec

    def transition(
        values_by_component,
        forcing_days,
        science_day_numbers,
        dynamic_operator_parameters,
        stomate_parameters,
        dynamic_hydrol_table_arrays,
        dynamic_landpoint_payload,
        dynamic_stomate_restart_template,
        dynamic_stomate_season_values,
        dynamic_diffuco_parameter_values,
    ):
        def body(current_values, inputs):
            forcing, science_day_number = inputs
            state = teacher.DriverFastStateBundle(
                tstep=-1 if first_day else 47,
                values_by_component=current_values,
                spec=initial.spec,
            )
            previous_packet = teacher.previous_packet_from_fast_state(state)
            transition_value, daily_fold, ok_leak_result, checksum = synthetic_boundary(
                previous_packet,
                forcing,
                static["metadata"],
                dynamic_operator_parameters,
                min_wind=context.min_wind,
            )
            ok_leak_updates = teacher._paper_half_hour_ok_leak_state_updates(ok_leak_result)
            replay_values = {
                "_paper_1961_later_day_half_hour_transition": transition_value,
                "_paper_later_day_daily_process_from_completed_entries": daily_fold,
                "_paper_later_day_daily_fold_prerequisite_gaps": (),
                "stomate_maintenance_respiration_parts_from_stacks": jnp.asarray(0.0),
                "_paper_half_hour_ok_leak_fold_from_entries": (
                    ok_leak_result,
                    ok_leak_updates,
                ),
            }
            with ExitStack() as stack:
                for name, value in replay_values.items():
                    stack.enter_context(
                        patch.object(
                            teacher,
                            name,
                            lambda *args, _value=value, **kwargs: _value,
                        )
                    )
                day = teacher.paper_1961_driver_later_day_runtime_result(
                    config_path,
                    previous_state=previous_packet,
                    day_index=1 if first_day else 2,
                    year=1962,
                    start_tstep=static["start_tstep"],
                    used_run_def_path=context.run_def_path,
                    prepared_context=context,
                    module_jit=True,
                    diffuco_local_jit=True,
                    use_static_jit_daily_carbon=False,
                    prebuild_day_payloads=True,
                    use_compiled_sechiba_day=True,
                    prebound_hydrol_runtime_static_tables=teacher.HydrolRuntimeStaticTables(
                        mineral=teacher.MineralCWRRTables(
                            **dynamic_hydrol_table_arrays._asdict(),
                            imin=static["mineral_imin"],
                            imax=static["mineral_imax"],
                        ),
                        peat=None,
                    ),
                    materialize_compiled_entries=False,
                    outer_compiled_daily_carbon_dispatch=static[
                        "daily_carbon_dispatch"
                    ],
                    compiled_forcing_series=forcing,
                    model_day_number=science_day_number,
                    compiled_stomate_parameter_values=stomate_parameters,
                    compiled_landpoint_payload=dynamic_landpoint_payload,
                    compiled_stomate_restart_template=dynamic_stomate_restart_template,
                    compiled_stomate_season_template=teacher.StomateRestartSeasonState(
                        **dynamic_stomate_season_values,
                        provenance=static["season"].provenance,
                    ),
                    compiled_diffuco_parameter_values=dynamic_diffuco_parameter_values,
                )
            return project_packet_values(day.day_end_state, output_spec), (
                day.daily_modelout.modelout_fields,
                day.daily_modelout.modelout,
                checksum,
            )

        if first_day:
            forcing = jax.tree_util.tree_map(lambda value: value[0], forcing_days)
            final_values, outputs = body(
                values_by_component, (forcing, science_day_numbers[0])
            )
            return final_values, jax.tree_util.tree_map(
                lambda value: jnp.expand_dims(value, axis=0), outputs
            )
        return jax.lax.scan(
            body,
            values_by_component,
            (forcing_days, science_day_numbers),
        )

    executable = jax.jit(transition).lower(
        initial.values_by_component,
        sample_forcing,
        sample_day_numbers,
        operator_parameters,
        static["stomate_parameter_values"],
        static["hydrol_table_arrays"],
        static["landpoint_payload"],
        static["stomate_restart_template"],
        static["stomate_season_values"],
        static["diffuco_parameter_values"],
    ).compile()
    return SyntheticExecutable(
        executable=executable,
        block_size=block_size,
        first_day=first_day,
        input_spec=initial.spec,
        output_spec=output_spec,
        stomate_parameter_values=static["stomate_parameter_values"],
        hydrol_table_arrays=static["hydrol_table_arrays"],
        landpoint_payload=static["landpoint_payload"],
        stomate_restart_template=static["stomate_restart_template"],
        stomate_season_values=static["stomate_season_values"],
        diffuco_parameter_values=static["diffuco_parameter_values"],
    )


def _run_synthetic_sequence(
    *, initial_year_end_state, segments, operator_parameters
) -> SyntheticSequenceResult:
    previous_state = teacher.rebase_driver_state_for_year_start(initial_year_end_state)
    daily_outputs = []
    checksums = []
    for executable, forcing_days, day_numbers in segments:
        final_values, stacked = executable.run(
            previous_state, forcing_days, day_numbers, operator_parameters
        )
        fields, modelout, checksum = stacked
        daily_outputs.extend(
            zip(
                _unstack(fields, executable.block_size),
                _unstack(modelout, executable.block_size),
                strict=True,
            )
        )
        checksums.extend(_unstack(checksum, executable.block_size))
        previous_state = packet_from_projected_values(
            values_by_component=final_values,
            spec=executable.output_spec,
            tstep=47,
        )
    _block_until_ready(previous_state.fields_by_component)
    _block_until_ready(tuple(daily_outputs))
    _block_until_ready(tuple(checksums))
    return SyntheticSequenceResult(
        last_day_end_state=previous_state,
        daily_outputs=tuple(daily_outputs),
        validation_checksums=tuple(checksums),
    )


def _device_packet(packet):
    return teacher.DriverPreviousStepStatePacket(
        tstep=packet.tstep,
        fields_by_component=jax.tree_util.tree_map(jax.device_put, packet.fields_by_component),
        provenance_by_component=packet.provenance_by_component,
    )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )


def _spread(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "population_stdev": statistics.pstdev(values),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.days) != 8:
        raise ValueError("the cost gate uses Day 1 plus one reusable seven-day block")
    cache = _load_state_cache(args.state_cache)
    host_initial_state = cache["state"]
    context = teacher.prepare_paper_1961_driver_context(
        args.config, used_run_def_path=args.run_def
    )

    preparation_started = time.perf_counter()
    forcing_day1 = _forcing_days(context, year=args.year, start_day=1, days=1)
    forcing_later = _forcing_days(context, year=args.year, start_day=2, days=7)
    day_number1 = np.asarray([1], dtype=np.int32)
    day_numbers_later = np.arange(2, 9, dtype=np.int32)
    rebased = teacher.rebase_driver_state_for_year_start(host_initial_state)
    output_width = _operator_output_width(rebased)
    parameters = initialize_operator_parameters(output_width, seed=args.seed)
    host_preparation_seconds = time.perf_counter() - preparation_started

    transfer_started = time.perf_counter()
    (
        forcing_day1,
        forcing_later,
        day_number1,
        day_numbers_later,
        parameters,
    ) = jax.device_put(
        (
            forcing_day1,
            forcing_later,
            day_number1,
            day_numbers_later,
            parameters,
        )
    )
    device_initial_state = _device_packet(host_initial_state)
    _block_until_ready(
        (
            forcing_day1,
            forcing_later,
            day_number1,
            day_numbers_later,
            parameters,
            device_initial_state.fields_by_component,
        )
    )
    host_to_device_seconds = time.perf_counter() - transfer_started

    first_compile_started = time.perf_counter()
    first_executable = _compile_synthetic_executable(
        config_path=args.config,
        context=context,
        initial_state=teacher.rebase_driver_state_for_year_start(device_initial_state),
        sample_forcing=forcing_day1,
        sample_day_numbers=day_number1,
        operator_parameters=parameters,
        first_day=True,
    )
    first_compile_seconds = time.perf_counter() - first_compile_started

    bootstrap_started = time.perf_counter()
    first_values, first_stacked = first_executable.run(
        teacher.rebase_driver_state_for_year_start(device_initial_state),
        forcing_day1,
        day_number1,
        parameters,
    )
    _block_until_ready((first_values, first_stacked))
    first_day_state = packet_from_projected_values(
        values_by_component=first_values,
        spec=first_executable.output_spec,
        tstep=47,
    )
    schema_bootstrap_seconds = time.perf_counter() - bootstrap_started

    later_compile_started = time.perf_counter()
    later_executable = _compile_synthetic_executable(
        config_path=args.config,
        context=context,
        initial_state=first_day_state,
        sample_forcing=forcing_later,
        sample_day_numbers=day_numbers_later,
        operator_parameters=parameters,
        first_day=False,
    )
    later_compile_seconds = time.perf_counter() - later_compile_started
    segments = (
        (first_executable, forcing_day1, day_number1),
        (later_executable, forcing_later, day_numbers_later),
    )

    def synthetic_callable():
        return _run_synthetic_sequence(
            initial_year_end_state=device_initial_state,
            segments=segments,
            operator_parameters=parameters,
        )

    synthetic_warm, synthetic_warm_seconds = _time_call(synthetic_callable)
    _, teacher_first_call_seconds = _time_call(
        lambda: _teacher_sequence(
            config_path=args.config,
            initial_year_end_state=host_initial_state,
            year=args.year,
            days=args.days,
            context=context,
            compiled_day_block_size=7,
        )
    )
    synthetic_times = []
    teacher_times = []
    paired_speedups = []
    for _ in range(args.repeats):
        _, teacher_seconds = _time_call(
            lambda: _teacher_sequence(
                config_path=args.config,
                initial_year_end_state=host_initial_state,
                year=args.year,
                days=args.days,
                context=context,
                compiled_day_block_size=7,
            )
        )
        _, synthetic_seconds = _time_call(synthetic_callable)
        teacher_times.append(teacher_seconds)
        synthetic_times.append(synthetic_seconds)
        paired_speedups.append(teacher_seconds / synthetic_seconds)

    teacher_median = statistics.median(teacher_times)
    synthetic_median = statistics.median(synthetic_times)
    median_speedup = teacher_median / synthetic_median
    stable_speedup = min(paired_speedups)
    nroot = synthetic_warm.last_day_end_state.fields_by_component[HYDROL_COMPONENT][
        NROOT_FIELD
    ]
    checksums = [float(np.asarray(value)) for value in synthetic_warm.validation_checksums]
    validation_finite = all(np.isfinite(value) for value in checksums)
    nroot_finite = bool(np.all(np.isfinite(np.asarray(nroot))))
    cost_gate_passed = stable_speedup >= 3.0 and validation_finite and nroot_finite
    decision = (
        "cost_gate_passed_tiny_supervised_pilot_may_be_proposed"
        if cost_gate_passed
        else "stop_daily_neural_route_cost_gate_failed"
    )
    return {
        "status": "completed",
        "decision": decision,
        "cost_gate_passed": cost_gate_passed,
        "neural_learnability": "inconclusive_not_tested",
        "teacher_git_commit": "7333b46c0b38650fb6c9250582876137831657b8",
        "experiment_git_head": _git_head(),
        "experiment_worktree_had_research_changes": _git_dirty(),
        "scope": "PFT14 point 001.0-071.0, synthetic operator cost only",
        "year": args.year,
        "days": args.days,
        "operator": {
            "inputs": ["dynamic_real_day_start_state", "dynamic_forcing_48", "dynamic_parameter_arrays"],
            "teacher_targets_as_inputs": False,
            "forcing_encoder": [len(FORCING_FIELDS), 64],
            "state_encoder": [16, 64],
            "mlp_layers": [[192, 128], [128, 128], [128, 64]],
            "output_head": [64, output_width],
            "parameter_count": int(
                sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(parameters))
            ),
            "output_head_width": output_width,
            "state_components_updated": list(COARSE_STATE_COMPONENTS),
            "daily_interface_fields": 17,
            "ok_leak_fields": list(COMMON_OK_LEAK_FIELDS) + ["deepC_peat"],
            "validation_reduction_blocked": True,
            "validation_checksums_finite": validation_finite,
            "validation_checksums": checksums,
        },
        "nroot_adapter": {
            "year_start_present": False,
            "runtime_present": True,
            "shape": list(np.asarray(nroot).shape),
            "finite": nroot_finite,
            "producer": "synthetic daily operator nroot head from dynamic day-start hydrol.us and dynamic parameters",
            "teacher_owner": "jax_orchidee.driver.orchestration._add_hydrol_to_previous_fields",
            "next_transition_consumer": "run_hydrol_first_step_module_from_precall(nroot_state=hydrol_state.nroot)",
            "retained_tail_consumer": "day-end state packet, restart/year handoff, and next-day fixed runtime state",
            "default_or_static_fallback_used": False,
        },
        "timing_seconds": {
            "host_preparation_not_in_hot_runtime": host_preparation_seconds,
            "host_to_device_not_in_hot_runtime": host_to_device_seconds,
            "synthetic_compile": {
                "first_day": first_compile_seconds,
                "later_day_block7": later_compile_seconds,
                "total": first_compile_seconds + later_compile_seconds,
            },
            "schema_bootstrap_execution_not_in_hot_runtime": schema_bootstrap_seconds,
            "warmup": {
                "synthetic": synthetic_warm_seconds,
                "teacher_first_call_compile_and_run": teacher_first_call_seconds,
            },
            "teacher_hot_runs": teacher_times,
            "synthetic_hot_runs": synthetic_times,
            "teacher_hot_statistics": _spread(teacher_times),
            "synthetic_hot_statistics": _spread(synthetic_times),
            "paired_speedups": paired_speedups,
            "median_speedup_vs_teacher": median_speedup,
            "stable_min_paired_speedup_vs_teacher": stable_speedup,
            "synthetic_seconds_per_day": synthetic_median / args.days,
            "synthetic_50year_extrapolated_seconds": synthetic_median / args.days * 365 * 50,
            "teacher_seconds_per_day": teacher_median / args.days,
            "teacher_50year_extrapolated_seconds": teacher_median / args.days * 365 * 50,
            "timed_disk_io_seconds": 0.0,
        },
        "comparison_contract": {
            "teacher": "complete Day 1 plus one reusable compiled seven-day complete-day block",
            "synthetic": "compiled Day 1 schema promotion plus one reusable compiled seven-day retained-tail block",
            "same_forcing_window": True,
            "same_initial_state": True,
            "same_retained_tail": True,
            "same_block_size": 7,
            "host_preparation_in_hot_runtime": False,
            "host_to_device_in_hot_runtime": False,
            "disk_io_in_hot_runtime": False,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
        },
        "limitations": [
            "This is a cost-only untrained synthetic operator and provides no learnability or accuracy evidence.",
            "The tiny residual amplitude keeps the retained tail numerically legal; it is not a physical approximation.",
            "All boundary arrays feed the retained tail or the blocked validation checksum, but checksum consumption is not a scientific constraint.",
            "Only one CPU, landpoint, year window, and PFT14 configuration were timed.",
            "No training, dataset generation, GPU work, or server work was performed.",
            "XLA logged algebraic-simplifier circular-loop warnings during first compilation; compilation completed and all hot runs completed normally.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-def", type=Path, default=DEFAULT_RUN_DEF)
    parser.add_argument("--year", type=int, default=1962)
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 3:
        raise ValueError("cost gate requires at least three hot timing repeats")
    payload = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "stable_speedup": payload["timing_seconds"][
                    "stable_min_paired_speedup_vs_teacher"
                ],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
