"""Free-rollout, non-neural persistence baseline for the daily coarse route.

This is a go/no-go experiment, not a candidate scientific model.  It advances
the retained Teacher tail from a boundary built only from the current dynamic
state and the current day's forcing.  No Teacher end-of-day target is consumed
by the baseline transition.
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
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.replay_ceiling import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_DEF,
    _block_until_ready,
    _capture_sequence,
    _compare_trees,
    _compile_dynamic_replay_executable,
    _dynamic_compiled_replay_sequence,
    _load_state_cache,
    _minimal_daily_fold,
    _ok_leak_with_restart_shape,
    _teacher_sequence,
    _time_call,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "performance"
    / "daily_coarse_graining"
    / "persistence_baseline_go_no_go.json"
)

COMMON_OK_LEAK_FIELDS = (
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
)


@dataclass(frozen=True)
class BaselineExecutable:
    executable: Any
    block_size: int
    first_day: bool
    state_spec: Any
    stomate_parameter_values: Any
    hydrol_table_arrays: Any
    landpoint_payload: Any
    stomate_restart_template: Any
    stomate_season_values: Any
    diffuco_parameter_values: Any

    def run(self, previous_state, forcing_days, day_numbers):
        values = teacher.fast_state_from_previous_packet(previous_state).values_by_component
        return self.executable(
            values,
            forcing_days,
            day_numbers,
            self.stomate_parameter_values,
            self.hydrol_table_arrays,
            self.landpoint_payload,
            self.stomate_restart_template,
            self.stomate_season_values,
            self.diffuco_parameter_values,
        )


@dataclass(frozen=True)
class BaselineSequenceResult:
    last_day_end_state: Any
    first_day_end_state: Any
    daily_outputs: tuple[tuple[Any, Any], ...]
    daily_interfaces: tuple[Mapping[str, Any], ...]
    ok_leak_interfaces: tuple[Mapping[str, Any], ...]


def _persistence_daily_interface(state, forcing, *, min_wind: float) -> dict[str, Any]:
    """Build a cheap forcing/carry baseline with the exact 17-field contract.

    Provenance for the reduction units and field meanings:
    ``jax_orchidee/stomate/daily.py::_stomate_daily_accumulator_compiled_core``.
    The physical proxies are explicitly persistence/diagnostic approximations;
    they do not claim Teacher process equivalence.
    """

    slow = state.fields_by_component["slowproc_stomate_previous_step_state"]
    hydrol = state.fields_by_component["hydrol_previous_step_state"]
    enerbil = state.fields_by_component["enerbil_previous_step_state"]
    therm = state.fields_by_component["thermosoil_previous_step_state"]
    accumulators = slow["daily_accumulators"]

    temp_air = jnp.asarray(forcing.temp_air).reshape((forcing.temp_air.shape[0], -1))
    precip_rain = jnp.asarray(forcing.precip_rain).reshape(
        (forcing.precip_rain.shape[0], -1)
    )
    precip_snow = jnp.asarray(forcing.precip_snow).reshape(
        (forcing.precip_snow.shape[0], -1)
    )
    wind = jnp.maximum(
        jnp.asarray(min_wind),
        jnp.sqrt(jnp.asarray(forcing.u) ** 2 + jnp.asarray(forcing.v) ** 2),
    ).reshape((forcing.u.shape[0], -1))
    soil_mc = jnp.asarray(hydrol["soil_mc"])
    previous_gpp = jnp.maximum(
        0.0,
        jnp.asarray(slow["npp_daily"])
        + jnp.asarray(slow["resp_maint"])
        + jnp.asarray(slow["resp_growth"]),
    )
    resp_maint_part = jnp.repeat(
        jnp.asarray(slow["resp_maint"])[:, :, None] / 12.0,
        12,
        axis=2,
    )
    return {
        "humrel_daily": jnp.asarray(hydrol["humrel"]),
        "litterhum_daily": jnp.mean(jnp.asarray(hydrol["humrel"]), axis=1),
        "t2m_daily": jnp.mean(temp_air, axis=0),
        "tsurf_daily": jnp.asarray(enerbil["temp_sol"]),
        "tsoil_daily": jnp.asarray(therm["stempdiag"]),
        "soilhum_daily": jnp.mean(soil_mc, axis=-1),
        "precip_daily": jnp.sum(precip_rain + precip_snow, axis=0),
        "gpp_daily": previous_gpp,
        "wspeed_daily": jnp.mean(wind, axis=0),
        "snowfall_daily": jnp.mean(precip_snow, axis=0),
        "snowmass_daily": jnp.asarray(hydrol["snow"]),
        "tmc_topgrass_daily": jnp.mean(soil_mc[:, :6, :], axis=(1, 2)),
        "t2m_min_daily": jnp.min(temp_air, axis=0),
        "t2m_max_daily": jnp.max(temp_air, axis=0),
        "resp_maint_part": resp_maint_part,
        "resp_maint_radia": jnp.asarray(accumulators["resp_maint_radia"]),
        "flood_root_radia": jnp.asarray(accumulators["flood_root_radia"]),
    }


def _persistence_ok_leak_interface(state):
    """Persist all common OK_LEAK pools and conditional ``deepC_peat``."""

    slow = state.fields_by_component["slowproc_stomate_previous_step_state"]
    minimal = SimpleNamespace(
        littercalc=SimpleNamespace(
            litter_above=slow["litter_above"],
            litter_below=slow["litter_below"],
            lignin_struc_above=slow["lignin_struc_above"],
            lignin_struc_below=slow["lignin_struc_below"],
            litterpart=slow["litterpart"],
            dead_leaves=slow["dead_leaves"],
            fuel=SimpleNamespace(
                fuel_1hr=slow["fuel_1hr"],
                fuel_10hr=slow["fuel_10hr"],
                fuel_100hr=slow["fuel_100hr"],
                fuel_1000hr=slow["fuel_1000hr"],
            ),
        ),
        soilcarbon=SimpleNamespace(
            carbon_32l=slow["carbon_32l"],
            doc=slow["DOC"],
            perma_peat=SimpleNamespace(deepc_peat=slow["deepC_peat"]),
        ),
        interception_storage=slow["interception_storage"],
    )
    return _ok_leak_with_restart_shape(minimal, state)


def _baseline_boundary(state, forcing, metadata, *, min_wind: float):
    """Return the complete pre-daily-STOMATE baseline boundary."""

    daily_fields = _persistence_daily_interface(state, forcing, min_wind=min_wind)
    ok_leak = _persistence_ok_leak_interface(state)
    final_t2m = jnp.asarray(forcing.temp_air)[-1]
    final_temp_sol = state.fields_by_component["enerbil_previous_step_state"]["temp_sol"]
    end_state = teacher.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=state.fields_by_component,
        provenance_by_component=state.provenance_by_component,
    )
    transition = teacher.DriverRuntimeDayStepTransition(
        completed_entry_payloads=(
            {
                "t2m": final_t2m,
                "stempdiag": state.fields_by_component["thermosoil_previous_step_state"][
                    "stempdiag"
                ],
                "t2mdiag": final_t2m,
                "temp_sol": final_temp_sol,
            },
        ),
        current_state=end_state,
        first_step_metadata=metadata,
        stopped_at_tstep=None,
        state_gaps=(),
        compiled_entry_stacks={
            "t2m": final_t2m,
            "stempdiag": state.fields_by_component["thermosoil_previous_step_state"][
                "stempdiag"
            ],
        },
    )
    return transition, _minimal_daily_fold(daily_fields), ok_leak


def _ok_leak_output_mapping(ok_leak) -> dict[str, Any]:
    result = teacher._paper_half_hour_ok_leak_state_updates(ok_leak)
    if ok_leak.soilcarbon.perma_peat is not None:
        result["deepC_peat"] = ok_leak.soilcarbon.perma_peat.deepc_peat
    return result


def _compile_baseline_executable(
    *,
    config_path: Path,
    context,
    initial_state,
    sample_forcing,
    sample_day_numbers,
    first_day: bool,
) -> BaselineExecutable:
    """Compile one real baseline scan; no Teacher target is an argument."""

    block_size = int(np.asarray(sample_day_numbers).shape[0])
    if block_size < 1 or (first_day and block_size != 1):
        raise ValueError("first-day baseline executable must contain exactly one day")
    initial = teacher.fast_state_from_previous_packet(initial_state)
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
    prebound_tables = first_inputs.hydrol_runtime_static_tables
    mineral_imin = int(prebound_tables.mineral.imin)
    mineral_imax = int(prebound_tables.mineral.imax)
    daily_carbon_dispatch = teacher._paper_daily_carbon_static_dispatch(context, initial_state)
    stomate_parameter_values = teacher._compiled_stomate_parameter_values(context)
    landpoint_payload = teacher._compiled_landpoint_payload(
        first_inputs.compiled_base_payload_template
    )
    stomate_restart_template = context.first_step_restart_state.stomate
    season_state = teacher.read_stomate_restart_season_state(
        context.first_step_restart_state.stomate_input
    )
    stomate_season_values = {
        name: value for name, value in season_state._asdict().items() if name != "provenance"
    }
    diffuco_parameter_values = teacher._compiled_diffuco_parameter_values(context)
    hydrol_table_arrays = teacher._compiled_hydrol_table_arrays(prebound_tables)
    template = first_inputs.compiled_base_payload_template
    run_scalars = first_inputs.day_start_bundle.run_scalars
    metadata = teacher.DriverRuntimeStepMetadata(
        pref_soil_veg=run_scalars.pref_soil_veg,
        nstm=run_scalars.nstm,
        is_tree=run_scalars.is_tree,
        is_peat=run_scalars.is_peat,
        lalo=template.lalo,
        npts=template.kjpindex,
    )

    def transition(
        values_by_component,
        forcing_days,
        science_day_numbers,
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
            transition_value, daily_fold, ok_leak_result = _baseline_boundary(
                previous_packet,
                forcing,
                metadata,
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
                            **dynamic_hydrol_table_arrays._asdict(),
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
                    compiled_landpoint_payload=dynamic_landpoint_payload,
                    compiled_stomate_restart_template=dynamic_stomate_restart_template,
                    compiled_stomate_season_template=teacher.StomateRestartSeasonState(
                        **dynamic_stomate_season_values,
                        provenance=season_state.provenance,
                    ),
                    compiled_diffuco_parameter_values=dynamic_diffuco_parameter_values,
                )
            packet = day.day_end_state
            next_values = tuple(
                tuple(packet.fields_by_component[component][name] for name in field_names)
                for component, field_names in zip(
                    initial.spec.components,
                    initial.spec.field_names_by_component,
                    strict=True,
                )
            )
            return next_values, (
                day.daily_modelout.modelout_fields,
                day.daily_modelout.modelout,
                daily_fold.daily_fields,
                _ok_leak_output_mapping(ok_leak_result),
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
        stomate_parameter_values,
        hydrol_table_arrays,
        landpoint_payload,
        stomate_restart_template,
        stomate_season_values,
        diffuco_parameter_values,
    ).compile()
    return BaselineExecutable(
        executable=executable,
        block_size=block_size,
        first_day=first_day,
        state_spec=initial.spec,
        stomate_parameter_values=stomate_parameter_values,
        hydrol_table_arrays=hydrol_table_arrays,
        landpoint_payload=landpoint_payload,
        stomate_restart_template=stomate_restart_template,
        stomate_season_values=stomate_season_values,
        diffuco_parameter_values=diffuco_parameter_values,
    )


def _forcing_days(context, *, year: int, start_day: int, days: int):
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    return jax.tree_util.tree_map(
        lambda *values: np.stack(values),
        *(
            teacher._paper_compiled_forcing_day(
                context,
                year=year,
                start_tstep=(day_index - 1) * steps_per_day,
                steps_per_stomate=steps_per_day,
            )
            for day_index in range(start_day, start_day + days)
        ),
    )


def _unstack(value, size: int) -> tuple[Any, ...]:
    return tuple(jax.tree_util.tree_map(lambda leaf: leaf[index], value) for index in range(size))


def _run_baseline_sequence(
    *,
    initial_year_end_state,
    segments: Sequence[tuple[BaselineExecutable, Any, Any]],
) -> BaselineSequenceResult:
    previous_state = teacher.rebase_driver_state_for_year_start(initial_year_end_state)
    daily_outputs: list[tuple[Any, Any]] = []
    daily_interfaces: list[Mapping[str, Any]] = []
    ok_leak_interfaces: list[Mapping[str, Any]] = []
    first_day_end_state = None
    for executable, forcing_days, day_numbers in segments:
        final_values, stacked = executable.run(previous_state, forcing_days, day_numbers)
        fields, modelout, daily, ok_leak = stacked
        daily_outputs.extend(zip(_unstack(fields, executable.block_size), _unstack(modelout, executable.block_size), strict=True))
        daily_interfaces.extend(_unstack(daily, executable.block_size))
        ok_leak_interfaces.extend(_unstack(ok_leak, executable.block_size))
        previous_state = teacher.previous_packet_from_fast_state(
            teacher.DriverFastStateBundle(
                tstep=47,
                values_by_component=final_values,
                spec=executable.state_spec,
            )
        )
        if first_day_end_state is None:
            first_day_end_state = previous_state
    _block_until_ready(previous_state.fields_by_component)
    _block_until_ready(tuple(daily_outputs))
    return BaselineSequenceResult(
        last_day_end_state=previous_state,
        first_day_end_state=first_day_end_state,
        daily_outputs=tuple(daily_outputs),
        daily_interfaces=tuple(daily_interfaces),
        ok_leak_interfaces=tuple(ok_leak_interfaces),
    )


def _relative_l2(expected, actual) -> float:
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    denominator = max(float(np.linalg.norm(expected_array.ravel())), 1.0e-30)
    return float(np.linalg.norm((actual_array - expected_array).ravel()) / denominator)


def _mapping_errors(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name in expected:
        if name not in actual:
            result[name] = {"missing": True}
            continue
        result[name] = {
            "relative_l2": _relative_l2(expected[name], actual[name]),
            "max_absolute_error": float(
                np.max(np.abs(np.asarray(actual[name]) - np.asarray(expected[name])))
            ),
        }
    return result


def _state_group_errors(expected_state, actual_state) -> dict[str, Any]:
    components = (
        "diffuco_previous_step_state",
        "enerbil_previous_step_state",
        "hydrol_previous_step_state",
        "thermosoil_previous_step_state",
    )
    result = {}
    for component in components:
        expected = expected_state.fields_by_component[component]
        actual = actual_state.fields_by_component[component]
        missing = tuple(name for name in expected if name not in actual)
        relative = [
            _relative_l2(expected[name], actual[name])
            for name in expected
            if name in actual
        ]
        result[component] = {
            "field_count": len(relative),
            "missing_fields": missing,
            "max_field_relative_l2": 1.0e300 if missing else max(relative),
            "median_field_relative_l2": statistics.median(relative),
        }
    return result


def _stock_summary(state) -> dict[str, float]:
    fields = state.fields_by_component
    hydrol = fields["hydrol_previous_step_state"]
    slow = fields["slowproc_stomate_previous_step_state"]

    def total(*values):
        return float(sum(np.sum(np.asarray(value, dtype=np.float64)) for value in values))

    return {
        "water_storage_proxy": total(
            hydrol["mc"],
            hydrol["mcl"],
            hydrol["qsintveg"],
            hydrol["snow"],
            hydrol["flood_res"],
        ),
        "carbon_stock_proxy": total(
            slow["biomass"],
            slow["litter_above"],
            slow["litter_below"],
            slow["carbon_32l"],
            slow["DOC"],
            slow["deepC_peat"],
        ),
    }


def _legality(state) -> dict[str, Any]:
    selected = {
        "hydrol.mc": state.fields_by_component["hydrol_previous_step_state"]["mc"],
        "hydrol.mcl": state.fields_by_component["hydrol_previous_step_state"]["mcl"],
        "hydrol.snow": state.fields_by_component["hydrol_previous_step_state"]["snow"],
        "slowproc.biomass": state.fields_by_component["slowproc_stomate_previous_step_state"][
            "biomass"
        ],
        "slowproc.carbon_32l": state.fields_by_component[
            "slowproc_stomate_previous_step_state"
        ]["carbon_32l"],
        "slowproc.DOC": state.fields_by_component["slowproc_stomate_previous_step_state"]["DOC"],
        "slowproc.deepC_peat": state.fields_by_component[
            "slowproc_stomate_previous_step_state"
        ]["deepC_peat"],
    }
    return {
        name: {
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(np.asarray(value)))),
            "negative_count": int(np.count_nonzero(np.asarray(value) < 0.0)),
        }
        for name, value in selected.items()
    }


def _modelout_errors(records, result: BaselineSequenceResult) -> dict[str, Any]:
    names = ("AGB_model", "BGB_model", "GPP_model", "NPP_model")

    def pft14(value) -> float:
        array = np.asarray(value, dtype=np.float64)
        if array.shape and array.shape[-1] >= 14:
            return float(np.mean(array[..., 13]))
        return float(np.mean(array))

    by_day = []
    for record, (_, actual) in zip(records, result.daily_outputs, strict=True):
        expected = record.expected_result.daily_modelout.modelout
        values = {}
        for name in names:
            expected_value = pft14(getattr(expected, name))
            actual_value = pft14(getattr(actual, name))
            values[name] = {
                "teacher": expected_value,
                "baseline": actual_value,
                "relative_error": abs(actual_value - expected_value)
                / max(abs(expected_value), 1.0e-30),
            }
        by_day.append({"day_index": record.day_index, **values})
    return {"by_day": by_day, "final_day": by_day[-1]}


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    cache = _load_state_cache(args.state_cache)
    initial_state = cache["state"]
    context = teacher.prepare_paper_1961_driver_context(
        args.config,
        used_run_def_path=args.run_def,
    )
    days = int(args.days)
    if days != 8:
        raise ValueError("the bounded go/no-go design uses Day 1 plus one reusable 7-day block")

    capture_started = time.perf_counter()
    records = _capture_sequence(
        config_path=args.config,
        initial_year_end_state=initial_state,
        year=args.year,
        days=days,
        context=context,
    )
    capture_seconds = time.perf_counter() - capture_started

    preparation_started = time.perf_counter()
    forcing_day1 = _forcing_days(context, year=args.year, start_day=1, days=1)
    forcing_later = _forcing_days(context, year=args.year, start_day=2, days=7)
    day_number1 = np.asarray([1], dtype=np.int32)
    day_numbers_later = np.arange(2, 9, dtype=np.int32)
    forcing_preparation_seconds = time.perf_counter() - preparation_started

    transfer_started = time.perf_counter()
    forcing_day1, forcing_later, day_number1, day_numbers_later = jax.device_put(
        (forcing_day1, forcing_later, day_number1, day_numbers_later)
    )
    _block_until_ready((forcing_day1, forcing_later, day_number1, day_numbers_later))
    host_to_device_seconds = time.perf_counter() - transfer_started

    baseline_compile_started = time.perf_counter()
    rebased = teacher.rebase_driver_state_for_year_start(initial_state)
    first_executable = _compile_baseline_executable(
        config_path=args.config,
        context=context,
        initial_state=rebased,
        sample_forcing=forcing_day1,
        sample_day_numbers=day_number1,
        first_day=True,
    )
    later_executable = _compile_baseline_executable(
        config_path=args.config,
        context=context,
        initial_state=rebased,
        sample_forcing=forcing_later,
        sample_day_numbers=day_numbers_later,
        first_day=False,
    )
    baseline_compile_seconds = time.perf_counter() - baseline_compile_started
    segments = (
        (first_executable, forcing_day1, day_number1),
        (later_executable, forcing_later, day_numbers_later),
    )

    replay_compile_started = time.perf_counter()
    replay_executable = _compile_dynamic_replay_executable(
        config_path=args.config,
        context=context,
        initial_state=records[0].expected_result.day_end_state,
        sample_records=records[1:],
    )
    replay_compile_seconds = time.perf_counter() - replay_compile_started

    def baseline_callable():
        return _run_baseline_sequence(
            initial_year_end_state=initial_state,
            segments=segments,
        )

    def replay_callable():
        return _dynamic_compiled_replay_sequence(
            config_path=args.config,
            initial_year_end_state=initial_state,
            records=records,
            context=context,
            executable=replay_executable,
        )

    baseline_warm, baseline_warm_seconds = _time_call(baseline_callable)
    replay_warm, replay_warm_seconds = _time_call(replay_callable)
    _, teacher_warm_seconds = _time_call(
        lambda: _teacher_sequence(
            config_path=args.config,
            initial_year_end_state=initial_state,
            year=args.year,
            days=days,
            context=context,
            compiled_day_block_size=7,
        )
    )

    baseline_times = []
    replay_times = []
    teacher_times = []
    for _ in range(args.repeats):
        _, teacher_seconds = _time_call(
            lambda: _teacher_sequence(
                config_path=args.config,
                initial_year_end_state=initial_state,
                year=args.year,
                days=days,
                context=context,
                compiled_day_block_size=7,
            )
        )
        _, replay_seconds = _time_call(replay_callable)
        _, baseline_seconds = _time_call(baseline_callable)
        teacher_times.append(teacher_seconds)
        replay_times.append(replay_seconds)
        baseline_times.append(baseline_seconds)

    teacher_median = statistics.median(teacher_times)
    replay_median = statistics.median(replay_times)
    baseline_median = statistics.median(baseline_times)

    final_teacher = records[-1].expected_result.day_end_state
    state_errors = _state_group_errors(final_teacher, baseline_warm.last_day_end_state)
    daily_errors = [
        {
            "day_index": record.day_index,
            "fields": _mapping_errors(record.daily_fold.daily_fields, actual),
        }
        for record, actual in zip(records, baseline_warm.daily_interfaces, strict=True)
    ]
    teacher_ok = []
    for record in records:
        values = dict(record.ok_leak_updates)
        if record.ok_leak_result.soilcarbon.perma_peat is not None:
            values["deepC_peat"] = record.ok_leak_result.soilcarbon.perma_peat.deepc_peat
        teacher_ok.append(values)
    ok_errors = [
        {
            "day_index": record.day_index,
            "fields": _mapping_errors(expected, actual),
        }
        for record, expected, actual in zip(
            records, teacher_ok, baseline_warm.ok_leak_interfaces, strict=True
        )
    ]
    initial_rebased = teacher.rebase_driver_state_for_year_start(initial_state)
    stocks = {
        "initial": _stock_summary(initial_rebased),
        "baseline_day1": _stock_summary(baseline_warm.first_day_end_state),
        "baseline_day8": _stock_summary(baseline_warm.last_day_end_state),
        "teacher_day8": _stock_summary(final_teacher),
    }
    total_precipitation = float(
        np.sum(np.asarray(forcing_day1.precip_rain))
        + np.sum(np.asarray(forcing_day1.precip_snow))
        + np.sum(np.asarray(forcing_later.precip_rain))
        + np.sum(np.asarray(forcing_later.precip_snow))
    )
    physical_changed = any(
        name in baseline_warm.last_day_end_state.fields_by_component[component]
        and not np.array_equal(
            np.asarray(initial_rebased.fields_by_component[component][name]),
            np.asarray(
                baseline_warm.last_day_end_state.fields_by_component[component][name]
            ),
        )
        for component in (
            "diffuco_previous_step_state",
            "enerbil_previous_step_state",
            "hydrol_previous_step_state",
            "thermosoil_previous_step_state",
        )
        for name in initial_rebased.fields_by_component[component]
    )
    legality = _legality(baseline_warm.last_day_end_state)
    illegal = any(
        counts["nonfinite_count"] or counts["negative_count"]
        for counts in legality.values()
    )
    max_physical_error = max(
        item["max_field_relative_l2"] for item in state_errors.values()
    )
    missing_physical_fields = {
        component: item["missing_fields"]
        for component, item in state_errors.items()
        if item["missing_fields"]
    }
    modelout_errors = _modelout_errors(records, baseline_warm)
    final_flux_error = max(
        modelout_errors["final_day"][name]["relative_error"]
        for name in ("GPP_model", "NPP_model")
    )
    water_storage_changed = not np.isclose(
        stocks["baseline_day8"]["water_storage_proxy"],
        stocks["initial"]["water_storage_proxy"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    forcing_response_failure = total_precipitation > 0.0 and not water_storage_changed
    baseline_passed = (
        teacher_median / baseline_median >= 3.0
        and not illegal
        and not missing_physical_fields
        and not forcing_response_failure
        and max_physical_error <= 0.25
        and final_flux_error <= 0.25
    )
    baseline_assessment = (
        "persistence_baseline_passed"
        if baseline_passed
        else "persistence_baseline_failed"
    )
    observations = []
    if teacher_median / baseline_median < 3.0:
        observations.append("real_baseline_speedup_below_3x")
    if illegal:
        observations.append("illegal_or_nonfinite_state")
    if missing_physical_fields:
        observations.append("baseline_day_end_schema_missing_teacher_physical_fields")
    if forcing_response_failure:
        observations.append("physical_state_did_not_respond_to_nonzero_daily_forcing")
    if max_physical_error > 0.25:
        observations.append("eight_day_physical_state_relative_error_exceeds_25_percent")
    if final_flux_error > 0.25:
        observations.append("eight_day_gpp_or_npp_relative_error_exceeds_25_percent")

    return {
        "status": "completed",
        "baseline_assessment": baseline_assessment,
        "baseline_failure_observations": observations,
        "neural_learnability": "neural_learnability_inconclusive",
        "route_decision": "large_scale_training_not_authorized",
        "teacher_git_commit": "7333b46c0b38650fb6c9250582876137831657b8",
        "experiment_git_head": _git_head(),
        "scope": "PFT14 point 001.0-071.0, free-rollout non-neural persistence baseline",
        "year": args.year,
        "days": days,
        "design_note": "Day 1 uses a target-free one-day executable; Days 2-8 use one reusable seven-day executable.",
        "baseline_inputs": ["free_rollout_day_start_dynamic_state", "current_day_forcing_48"],
        "forbidden_inputs_absent": ["teacher_day_end_state", "teacher_daily_interface", "teacher_ok_leak_target"],
        "timing_seconds": {
            "teacher_target_capture_not_in_runtime": capture_seconds,
            "forcing_preparation_not_in_hot_runtime": forcing_preparation_seconds,
            "host_to_device_not_in_hot_runtime": host_to_device_seconds,
            "baseline_compile_not_in_hot_runtime": baseline_compile_seconds,
            "replay_compile_not_in_hot_runtime": replay_compile_seconds,
            "warmup": {
                "teacher": teacher_warm_seconds,
                "zero_inference_replay": replay_warm_seconds,
                "baseline": baseline_warm_seconds,
            },
            "teacher_runs": teacher_times,
            "zero_inference_replay_runs": replay_times,
            "baseline_runs": baseline_times,
            "teacher_median": teacher_median,
            "zero_inference_replay_median": replay_median,
            "baseline_median": baseline_median,
            "baseline_speedup_vs_teacher": teacher_median / baseline_median,
            "baseline_seconds_per_day": baseline_median / days,
            "baseline_50year_extrapolated_seconds": baseline_median / days * 365 * 50,
            "zero_inference_replay_speedup_vs_teacher": teacher_median / replay_median,
            "timed_disk_io_seconds": 0.0,
        },
        "performance_interpretation": {
            "directly_comparable_same_boundary_ab": False,
            "zero_inference_replay": "dynamic_teacher_boundary_transport_and_reconstruction_measurement",
            "persistence_baseline": "lower_cost_reference_subject_to_xla_dead_code_elimination",
            "architectural_absolute_ceiling_established": False,
        },
        "evidence_scope": {
            "water_and_ok_leak_drift": "proves_persistence_baseline_failure_only",
            "hydrol_nroot_missing": "adapter_schema_defect_not_learnability_evidence",
            "daily_operator_learnability": "not_tested",
        },
        "accuracy": {
            "final_complete_state": _compare_trees(
                final_teacher.fields_by_component,
                baseline_warm.last_day_end_state.fields_by_component,
                atol=1.0e-8,
                rtol=1.0e-10,
            ),
            "final_physical_state_groups": state_errors,
            "daily_interface_by_day": daily_errors,
            "ok_leak_interface_by_day": ok_errors,
            "modelout_agb_bgb_gpp_npp": modelout_errors,
        },
        "legality_and_minimal_budgets": {
            "final_state_legality": legality,
            "stocks": stocks,
            "total_forcing_precipitation_over_window": total_precipitation,
            "physical_state_changed_from_initial": physical_changed,
            "water_storage_proxy_changed_from_initial": water_storage_changed,
            "forcing_response_failure": forcing_response_failure,
            "water_budget_warning": "Storage proxy is not unit-closed; unchanged physical storage under nonzero precipitation is used only as a hard response failure.",
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
        },
        "limitations": [
            "Persistence is intentionally the cheapest honest non-neural baseline, not a scientific approximation.",
            "Persistence-defined storage and OK_LEAK failures do not establish that a learned daily operator is infeasible.",
            "The missing hydrol.nroot field is an adapter/schema defect, not learnability evidence.",
            "Dynamic Teacher replay and persistence baseline do not execute an identical boundary transport/reconstruction path; their speedups are not a direct A/B comparison or an architectural ceiling.",
            "XLA may eliminate persisted state work, so the persistence timing is only a lower-cost reference for this implementation.",
            "The eight-day window scores this baseline only and does not test supervised learnability.",
            "Water and carbon stock summaries are legality/drift proxies, not frozen conservation equations.",
            "Teacher target capture is used only for post-run scoring and zero-inference replay, never as a baseline input.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-def", type=Path, default=DEFAULT_RUN_DEF)
    parser.add_argument("--year", type=int, default=1962)
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    payload = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "baseline_assessment": payload["baseline_assessment"],
                "neural_learnability": payload["neural_learnability"],
                "baseline_speedup": payload["timing_seconds"]["baseline_speedup_vs_teacher"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
