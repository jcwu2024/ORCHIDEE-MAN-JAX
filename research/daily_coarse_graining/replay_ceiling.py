"""In-memory pre-daily-STOMATE replay speed-ceiling probe.

The probe does not change the Teacher implementation. It temporarily captures
or injects existing orchestration return values around the fast-day boundary,
then lets the original season, daily carbon, modelout, and day-end writeback
tail execute normally.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from contextlib import ExitStack
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from jax_orchidee.stomate.daily import (
    StomateDailyAccumulationFold,
    StomateDailyProcessFold,
    StomateMaintenanceFold,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
DEFAULT_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
DEFAULT_OUTPUT = ROOT / "outputs" / "performance" / "daily_coarse_graining" / "pre_daily_stomate_replay.json"


@dataclass(frozen=True)
class PreDailyStomateReplayRecord:
    """Teacher values injected immediately before the retained daily tail."""

    year: int
    day_index: int
    start_tstep: int
    input_state_tstep: int
    half_hour_transition: Any
    daily_fold: Any
    pre_step_boundary: Any
    maintenance_resp_parts: Any
    ok_leak_result: Any
    ok_leak_updates: Mapping[str, Any]
    expected_result: Any
    daily_flux_labels: Any | None = None


@dataclass(frozen=True)
class CompiledReplayBlock:
    """One fixed-record replay tail compiled at the Teacher block level."""

    start_index: int
    records: tuple[PreDailyStomateReplayRecord, ...]
    executable: Any
    state_spec: Any
    block_forcing: Any
    block_day_numbers: Any
    stomate_parameter_values: Any
    hydrol_table_arrays: Any
    landpoint_payload: Any
    stomate_restart_template: Any
    stomate_season_values: Any
    diffuco_parameter_values: Any

    def run(self, previous_state):
        values = teacher.fast_state_from_previous_packet(previous_state).values_by_component
        return self.executable(
            values,
            self.block_forcing,
            self.block_day_numbers,
            self.stomate_parameter_values,
            self.hydrol_table_arrays,
            self.landpoint_payload,
            self.stomate_restart_template,
            self.stomate_season_values,
            self.diffuco_parameter_values,
        )


@dataclass(frozen=True)
class DynamicCompiledReplayExecutable:
    """Reusable retained-tail executable with canonical day-boundary inputs."""

    executable: Any
    block_size: int
    state_spec: Any
    boundary_spec: DynamicBoundaryNodeSpec
    stomate_parameter_values: Any
    hydrol_table_arrays: Any
    landpoint_payload: Any
    stomate_restart_template: Any
    stomate_season_values: Any
    diffuco_parameter_values: Any

    def run(self, previous_state, *, boundary_days, forcing_days, day_numbers):
        values = teacher.fast_state_from_previous_packet(previous_state).values_by_component
        return self.executable(
            values,
            forcing_days,
            day_numbers,
            boundary_days,
            self.stomate_parameter_values,
            self.hydrol_table_arrays,
            self.landpoint_payload,
            self.stomate_restart_template,
            self.stomate_season_values,
            self.diffuco_parameter_values,
        )


@dataclass(frozen=True)
class CompiledReplaySequenceResult:
    last_day_end_state: Any
    daily_outputs: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class DynamicBoundaryNodeSpec:
    """Static object graph used to rebuild a boundary from numeric leaves."""

    kind: str
    value: Any = None
    names: tuple[str, ...] = ()
    children: tuple["DynamicBoundaryNodeSpec", ...] = ()


def _is_dynamic_array(value: Any) -> bool:
    return isinstance(value, (jax.Array, np.ndarray, np.generic))


def _pack_dynamic_boundary(value: Any) -> tuple[DynamicBoundaryNodeSpec, tuple[Any, ...]]:
    """Separate a fixed Python object graph from its day-varying numeric arrays."""

    if _is_dynamic_array(value):
        return DynamicBoundaryNodeSpec(kind="dynamic"), (value,)
    if is_dataclass(value) and not isinstance(value, type):
        names = tuple(field.name for field in dataclass_fields(value))
        packed = tuple(_pack_dynamic_boundary(getattr(value, name)) for name in names)
        return (
            DynamicBoundaryNodeSpec(
                kind="dataclass",
                value=type(value),
                names=names,
                children=tuple(item[0] for item in packed),
            ),
            tuple(leaf for item in packed for leaf in item[1]),
        )
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        packed = tuple(_pack_dynamic_boundary(item) for item in value)
        return (
            DynamicBoundaryNodeSpec(
                kind="namedtuple",
                value=type(value),
                names=tuple(value._fields),
                children=tuple(item[0] for item in packed),
            ),
            tuple(leaf for item in packed for leaf in item[1]),
        )
    if isinstance(value, dict):
        names = tuple(value.keys())
        packed = tuple(_pack_dynamic_boundary(value[name]) for name in names)
        return (
            DynamicBoundaryNodeSpec(
                kind="dict",
                names=names,
                children=tuple(item[0] for item in packed),
            ),
            tuple(leaf for item in packed for leaf in item[1]),
        )
    if isinstance(value, SimpleNamespace):
        names = tuple(vars(value).keys())
        packed = tuple(_pack_dynamic_boundary(getattr(value, name)) for name in names)
        return (
            DynamicBoundaryNodeSpec(
                kind="namespace",
                value=type(value),
                names=names,
                children=tuple(item[0] for item in packed),
            ),
            tuple(leaf for item in packed for leaf in item[1]),
        )
    if isinstance(value, (tuple, list)):
        packed = tuple(_pack_dynamic_boundary(item) for item in value)
        return (
            DynamicBoundaryNodeSpec(
                kind="tuple" if isinstance(value, tuple) else "list",
                children=tuple(item[0] for item in packed),
            ),
            tuple(leaf for item in packed for leaf in item[1]),
        )
    return DynamicBoundaryNodeSpec(kind="static", value=value), ()


def _unpack_dynamic_boundary(spec: DynamicBoundaryNodeSpec, leaves: Sequence[Any]) -> Any:
    iterator = iter(leaves)

    def rebuild(node: DynamicBoundaryNodeSpec):
        if node.kind == "dynamic":
            return next(iterator)
        if node.kind == "static":
            return node.value
        values = tuple(rebuild(child) for child in node.children)
        if node.kind == "dataclass":
            return node.value(**dict(zip(node.names, values, strict=True)))
        if node.kind == "namedtuple":
            return node.value(*values)
        if node.kind == "dict":
            return dict(zip(node.names, values, strict=True))
        if node.kind == "namespace":
            return node.value(**dict(zip(node.names, values, strict=True)))
        if node.kind == "tuple":
            return values
        if node.kind == "list":
            return list(values)
        raise ValueError(f"Unknown dynamic boundary node kind: {node.kind}")

    result = rebuild(spec)
    try:
        next(iterator)
    except StopIteration:
        return result
    raise ValueError("Dynamic boundary has more leaves than its static spec")


_CAPTURED_OWNER_NAMES = (
    "_paper_1961_later_day_half_hour_transition",
    "_paper_later_day_daily_process_from_completed_entries",
    "_paper_ok_leak_pre_step_boundary_from_entry_state",
    "stomate_maintenance_respiration_parts_from_stacks",
    "_paper_half_hour_ok_leak_fold_from_entries",
)


def _capture_once(name: str, owner, captured: dict[str, Any]):
    def wrapper(*args, **kwargs):
        if name in captured:
            raise RuntimeError(f"Teacher boundary owner {name} was called more than once")
        result = owner(*args, **kwargs)
        captured[name] = result
        return result

    return wrapper


def capture_pre_daily_stomate_record(
    config_path: str | Path,
    *,
    previous_state,
    year: int,
    day_index: int,
    start_tstep: int,
    **day_kwargs,
) -> PreDailyStomateReplayRecord:
    """Run one compiled Teacher day and capture its fast-day return values."""

    if not bool(day_kwargs.get("use_compiled_sechiba_day", False)):
        raise ValueError("capture requires use_compiled_sechiba_day=True")
    captured: dict[str, Any] = {}
    originals = {name: getattr(teacher, name) for name in _CAPTURED_OWNER_NAMES}
    with ExitStack() as stack:
        for name, owner in originals.items():
            stack.enter_context(patch.object(teacher, name, _capture_once(name, owner, captured)))
        result = teacher.paper_1961_driver_later_day_runtime_result(
            config_path,
            previous_state=previous_state,
            year=int(year),
            day_index=int(day_index),
            start_tstep=int(start_tstep),
            **day_kwargs,
        )
    missing = tuple(name for name in _CAPTURED_OWNER_NAMES if name not in captured)
    if missing:
        daily_fold = captured.get(
            "_paper_later_day_daily_process_from_completed_entries"
        )
        accumulator = getattr(daily_fold, "accumulator", None)
        maintenance = getattr(daily_fold, "maintenance", None)
        details = {
            "missing_boundaries": missing,
            "runtime_missing_components": tuple(
                getattr(result, "missing_components", ())
            ),
            "daily_fold_missing_inputs": tuple(
                getattr(daily_fold, "missing_inputs", ())
            ),
            "accumulator_missing_inputs": tuple(
                getattr(accumulator, "missing_inputs", ())
            ),
            "maintenance_missing_inputs": tuple(
                getattr(maintenance, "missing_inputs", ())
            ),
            "accumulator_failed_step": getattr(accumulator, "failed_step", None),
            "maintenance_failed_step": getattr(maintenance, "failed_step", None),
        }
        raise RuntimeError(f"Teacher day did not reach replay boundaries: {details}")
    if not bool(getattr(result, "ready_for_day_end_state", True)):
        raise RuntimeError(f"Teacher capture day failed: {getattr(result, 'missing_components', ())}")
    ok_leak_fold = captured["_paper_half_hour_ok_leak_fold_from_entries"]
    if len(ok_leak_fold) not in {2, 3, 4}:
        raise RuntimeError(
            "Teacher replay capture received an unsupported OK_LEAK return contract"
        )
    ok_leak_result, ok_leak_updates = ok_leak_fold[:2]
    return PreDailyStomateReplayRecord(
        year=int(year),
        day_index=int(day_index),
        start_tstep=int(start_tstep),
        input_state_tstep=int(previous_state.tstep),
        half_hour_transition=captured["_paper_1961_later_day_half_hour_transition"],
        daily_fold=captured["_paper_later_day_daily_process_from_completed_entries"],
        pre_step_boundary=captured["_paper_ok_leak_pre_step_boundary_from_entry_state"],
        maintenance_resp_parts=captured["stomate_maintenance_respiration_parts_from_stacks"],
        ok_leak_result=ok_leak_result,
        ok_leak_updates=dict(ok_leak_updates),
        expected_result=result,
        daily_flux_labels=getattr(result, "daily_flux_labels", None),
    )


def replay_pre_daily_stomate_record(
    config_path: str | Path,
    *,
    previous_state,
    record: PreDailyStomateReplayRecord,
    **day_kwargs,
):
    """Inject one captured fast-day boundary and execute the original daily tail."""

    if int(previous_state.tstep) != int(record.input_state_tstep):
        raise ValueError(
            f"Replay input tstep {previous_state.tstep} does not match record {record.input_state_tstep}"
        )
    replay_values = {
        "_paper_1961_later_day_half_hour_transition": record.half_hour_transition,
        "_paper_later_day_daily_process_from_completed_entries": record.daily_fold,
        "_paper_ok_leak_pre_step_boundary_from_entry_state": record.pre_step_boundary,
        "stomate_maintenance_respiration_parts_from_stacks": record.maintenance_resp_parts,
        "_paper_half_hour_ok_leak_fold_from_entries": (
            record.ok_leak_result,
            dict(record.ok_leak_updates),
        ),
    }
    kwargs = dict(day_kwargs)
    kwargs.update(
        {
            "use_compiled_sechiba_day": True,
            "retain_stomate_step_results": False,
            "prebuild_day_payloads": False,
        }
    )
    with ExitStack() as stack:
        for name, value in replay_values.items():
            stack.enter_context(patch.object(teacher, name, lambda *args, _value=value, **kw: _value))
        result = teacher.paper_1961_driver_later_day_runtime_result(
            config_path,
            previous_state=previous_state,
            year=record.year,
            day_index=record.day_index,
            start_tstep=record.start_tstep,
            **kwargs,
        )
    if not bool(getattr(result, "ready_for_day_end_state", True)):
        raise RuntimeError(f"Replay day failed: {getattr(result, 'missing_components', ())}")
    return result


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _compare_trees(expected: Any, actual: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    if expected_tree != actual_tree:
        return {
            "passed": False,
            "schema_equal": False,
            "leaf_count": len(expected_leaves),
            "max_absolute_error": None,
            "max_relative_error": None,
            "failed_leaf_indices": [],
        }
    max_abs = 0.0
    max_rel = 0.0
    failed: list[int] = []
    for index, (expected_leaf, actual_leaf) in enumerate(zip(expected_leaves, actual_leaves, strict=True)):
        expected_array = np.asarray(expected_leaf)
        actual_array = np.asarray(actual_leaf)
        if expected_array.shape != actual_array.shape or expected_array.dtype.kind != actual_array.dtype.kind:
            failed.append(index)
            continue
        if expected_array.dtype.kind in "biuOSU":
            equal = np.array_equal(expected_array, actual_array)
        else:
            difference = np.abs(actual_array - expected_array)
            if difference.size:
                max_abs = max(max_abs, float(np.nanmax(difference)))
                denominator = np.maximum(np.abs(expected_array), np.finfo(np.float64).tiny)
                max_rel = max(max_rel, float(np.nanmax(difference / denominator)))
            equal = np.allclose(actual_array, expected_array, atol=atol, rtol=rtol, equal_nan=True)
        if not equal:
            failed.append(index)
    return {
        "passed": not failed,
        "schema_equal": True,
        "leaf_count": len(expected_leaves),
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "failed_leaf_indices": failed,
    }


def compare_replay_day(
    record: PreDailyStomateReplayRecord,
    replay_result,
    *,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-10,
) -> dict[str, Any]:
    """Compare replay and Teacher at complete day-end and modelout boundaries."""

    expected = record.expected_result
    return {
        "year": record.year,
        "day_index": record.day_index,
        "day_end_state": _compare_trees(
            expected.day_end_state.fields_by_component,
            replay_result.day_end_state.fields_by_component,
            atol=atol,
            rtol=rtol,
        ),
        "modelout_fields": _compare_trees(
            expected.daily_modelout.modelout_fields,
            replay_result.daily_modelout.modelout_fields,
            atol=atol,
            rtol=rtol,
        ),
        "modelout": _compare_trees(
            expected.daily_modelout.modelout,
            replay_result.daily_modelout.modelout,
            atol=atol,
            rtol=rtol,
        ),
    }


def _load_state_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(f"{path} is not a driver state cache")
    return payload


def _capture_sequence(
    *,
    config_path: Path,
    initial_year_end_state,
    year: int,
    days: int,
    context,
) -> tuple[PreDailyStomateReplayRecord, ...]:
    records: list[PreDailyStomateReplayRecord] = []
    previous_state = teacher.rebase_driver_state_for_year_start(initial_year_end_state)
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    common = {
        "used_run_def_path": context.run_def_path,
        "prepared_context": context,
        "module_jit": True,
        "diffuco_local_jit": True,
        "use_static_jit_daily_carbon": True,
        "use_compiled_sechiba_day": True,
        "retain_stomate_step_results": False,
        "prebuild_day_payloads": True,
    }
    for day_index in range(1, int(days) + 1):
        record = capture_pre_daily_stomate_record(
            config_path,
            previous_state=previous_state,
            year=year,
            day_index=day_index,
            start_tstep=(day_index - 1) * steps_per_day,
            **common,
        )
        _block_until_ready(record.expected_result)
        records.append(record)
        previous_state = record.expected_result.day_end_state
    return tuple(records)


def _replay_sequence(
    *,
    config_path: Path,
    initial_year_end_state,
    records: Sequence[PreDailyStomateReplayRecord],
    context,
):
    previous_state = teacher.rebase_driver_state_for_year_start(initial_year_end_state)
    results = []
    common = {
        "used_run_def_path": context.run_def_path,
        "prepared_context": context,
        "module_jit": True,
        "diffuco_local_jit": True,
        "use_static_jit_daily_carbon": True,
    }
    for record in records:
        result = replay_pre_daily_stomate_record(
            config_path,
            previous_state=previous_state,
            record=record,
            **common,
        )
        results.append(result)
        previous_state = result.day_end_state
    return tuple(results)


def _compile_replay_block(
    *,
    config_path: Path,
    context,
    initial_state,
    records: Sequence[PreDailyStomateReplayRecord],
    start_index: int,
) -> CompiledReplayBlock:
    """Compile an unrolled fixed-record tail block as an optimistic ceiling."""

    block_records = tuple(records)
    if not block_records:
        raise ValueError("compiled replay block requires at least one record")
    initial = teacher.fast_state_from_previous_packet(initial_state)
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    first_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=initial_state,
        year=block_records[0].year,
        start=block_records[0].start_tstep,
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    prebound_tables = first_inputs.hydrol_runtime_static_tables
    mineral_imin = int(prebound_tables.mineral.imin)
    mineral_imax = int(prebound_tables.mineral.imax)
    daily_carbon_dispatch = teacher._paper_daily_carbon_static_dispatch(context, initial_state)
    stomate_parameter_values = teacher._compiled_stomate_parameter_values(context)
    landpoint_payload = teacher._compiled_landpoint_payload(first_inputs.compiled_base_payload_template)
    stomate_restart_template = context.first_step_restart_state.stomate
    season_state = teacher.read_stomate_restart_season_state(context.first_step_restart_state.stomate_input)
    stomate_season_values = {
        name: value for name, value in season_state._asdict().items() if name != "provenance"
    }
    diffuco_parameter_values = teacher._compiled_diffuco_parameter_values(context)
    block_forcing = jax.tree_util.tree_map(
        lambda *values: np.stack(values),
        *(
            teacher._paper_compiled_forcing_day(
                context,
                year=record.year,
                start_tstep=record.start_tstep,
                steps_per_stomate=steps_per_day,
            )
            for record in block_records
        ),
    )
    block_day_numbers = np.asarray([record.day_index for record in block_records], dtype=np.int32)
    hydrol_table_arrays = teacher._compiled_hydrol_table_arrays(prebound_tables)

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
        current_values = values_by_component
        output_fields = []
        output_modelout = []
        for offset, record in enumerate(block_records):
            state = teacher.DriverFastStateBundle(
                tstep=47,
                values_by_component=current_values,
                spec=initial.spec,
            )
            forcing = jax.tree_util.tree_map(lambda value: value[offset], forcing_days)
            replay_values = {
                "_paper_1961_later_day_half_hour_transition": record.half_hour_transition,
                "_paper_later_day_daily_process_from_completed_entries": record.daily_fold,
                "_paper_ok_leak_pre_step_boundary_from_entry_state": record.pre_step_boundary,
                "stomate_maintenance_respiration_parts_from_stacks": record.maintenance_resp_parts,
                "_paper_half_hour_ok_leak_fold_from_entries": (
                    record.ok_leak_result,
                    dict(record.ok_leak_updates),
                ),
            }
            with ExitStack() as stack:
                for name, value in replay_values.items():
                    stack.enter_context(
                        patch.object(teacher, name, lambda *args, _value=value, **kwargs: _value)
                    )
                day = teacher.paper_1961_driver_later_day_runtime_result(
                    config_path,
                    previous_state=state,
                    day_index=2,
                    year=1961,
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
                            **dynamic_hydrol_table_arrays._asdict(),
                            imin=mineral_imin,
                            imax=mineral_imax,
                        ),
                        peat=None,
                    ),
                    materialize_compiled_entries=False,
                    outer_compiled_daily_carbon_dispatch=daily_carbon_dispatch,
                    compiled_forcing_series=forcing,
                    model_day_number=science_day_numbers[offset],
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
            current_values = tuple(
                tuple(packet.fields_by_component[component][name] for name in field_names)
                for component, field_names in zip(
                    initial.spec.components,
                    initial.spec.field_names_by_component,
                    strict=True,
                )
            )
            output_fields.append(day.daily_modelout.modelout_fields)
            output_modelout.append(day.daily_modelout.modelout)
        return current_values, (
            jax.tree_util.tree_map(lambda *values: jax.numpy.stack(values), *output_fields),
            jax.tree_util.tree_map(lambda *values: jax.numpy.stack(values), *output_modelout),
        )

    executable = jax.jit(transition).lower(
        initial.values_by_component,
        block_forcing,
        block_day_numbers,
        stomate_parameter_values,
        hydrol_table_arrays,
        landpoint_payload,
        stomate_restart_template,
        stomate_season_values,
        diffuco_parameter_values,
    ).compile()
    return CompiledReplayBlock(
        start_index=int(start_index),
        records=block_records,
        executable=executable,
        state_spec=initial.spec,
        block_forcing=block_forcing,
        block_day_numbers=block_day_numbers,
        stomate_parameter_values=stomate_parameter_values,
        hydrol_table_arrays=hydrol_table_arrays,
        landpoint_payload=landpoint_payload,
        stomate_restart_template=stomate_restart_template,
        stomate_season_values=stomate_season_values,
        diffuco_parameter_values=diffuco_parameter_values,
    )


def _compile_replay_blocks(
    *,
    config_path: Path,
    initial_year_end_state,
    records: Sequence[PreDailyStomateReplayRecord],
    context,
    block_size: int,
) -> tuple[CompiledReplayBlock, ...]:
    blocks = []
    start_index = 1
    while start_index + int(block_size) <= len(records):
        initial_state = records[start_index - 1].expected_result.day_end_state
        blocks.append(
            _compile_replay_block(
                config_path=config_path,
                context=context,
                initial_state=initial_state,
                records=records[start_index : start_index + int(block_size)],
                start_index=start_index,
            )
        )
        start_index += int(block_size)
    return tuple(blocks)


def _dynamic_boundary_value(record: PreDailyStomateReplayRecord) -> tuple[Any, ...]:
    """Return only day-varying values owned by the replaced fast-day operator.

    The retained Teacher reconstructs the pre-step OK_LEAK boundary from the
    current state and static context. Daily-fold scaffolding, duplicate
    OK_LEAK updates, and the full OK_LEAK process graph are deliberately not
    transported across this boundary.
    """

    transition = record.half_hour_transition
    final_payload = transition.completed_entry_payloads[-1]
    current_state = transition.current_state
    normalized_state = teacher.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=current_state.fields_by_component,
        provenance_by_component=current_state.provenance_by_component,
    )
    daily_fields = dict(record.daily_fold.daily_fields)
    ok_leak = record.ok_leak_result
    litter = ok_leak.littercalc
    soilcarbon = ok_leak.soilcarbon
    minimal_ok_leak = SimpleNamespace(
        littercalc=SimpleNamespace(
            litter_above=litter.litter_above,
            litter_below=litter.litter_below,
            lignin_struc_above=litter.lignin_struc_above,
            lignin_struc_below=litter.lignin_struc_below,
            litterpart=litter.litterpart,
            dead_leaves=litter.dead_leaves,
            fuel=SimpleNamespace(
                fuel_1hr=litter.fuel.fuel_1hr,
                fuel_10hr=litter.fuel.fuel_10hr,
                fuel_100hr=litter.fuel.fuel_100hr,
                fuel_1000hr=litter.fuel.fuel_1000hr,
            ),
        ),
        soilcarbon=SimpleNamespace(
            carbon_32l=soilcarbon.carbon_32l,
            doc=soilcarbon.doc,
            perma_peat=(
                None
                if soilcarbon.perma_peat is None
                else SimpleNamespace(deepc_peat=soilcarbon.perma_peat.deepc_peat)
            ),
        ),
        interception_storage=ok_leak.interception_storage,
    )
    return (
        normalized_state,
        {
            "t2mdiag": final_payload["t2mdiag"],
            "temp_sol": final_payload["temp_sol"],
        },
        daily_fields,
        minimal_ok_leak,
    )


def _minimal_daily_fold(daily_fields: Mapping[str, Any]):
    """Rebuild the successful fold shell consumed by the retained daily tail."""

    accumulator = StomateDailyAccumulationFold(
        fields={},
        step_results=(),
        missing_inputs=(),
        failed_step=None,
        provenance=(),
        notes=(),
    )
    maintenance = StomateMaintenanceFold(
        resp_maint_part=daily_fields["resp_maint_part"],
        step_results=(),
        resp_maint_radia=daily_fields["resp_maint_radia"],
        flood_root_radia=daily_fields["flood_root_radia"],
        missing_inputs=(),
        failed_step=None,
        provenance=(),
        notes=(),
    )
    return StomateDailyProcessFold(
        accumulator=accumulator,
        maintenance=maintenance,
        daily_fields=dict(daily_fields),
        missing_inputs=(),
        provenance=(),
        notes=(),
    )


def _ok_leak_with_restart_shape(ok_leak_result, end_state):
    """Add the non-predictive shape carrier needed by Teacher writeback."""

    soilcarbon = ok_leak_result.soilcarbon
    previous_resp_hetero = end_state.fields_by_component[
        "slowproc_stomate_previous_step_state"
    ]["resp_hetero"]
    return SimpleNamespace(
        littercalc=ok_leak_result.littercalc,
        soilcarbon=SimpleNamespace(
            carbon_32l=soilcarbon.carbon_32l,
            doc=soilcarbon.doc,
            resp_hetero_soil=previous_resp_hetero,
            perma_peat=soilcarbon.perma_peat,
        ),
        interception_storage=ok_leak_result.interception_storage,
    )


def _dynamic_boundary_inventory(record: PreDailyStomateReplayRecord) -> dict[str, Any]:
    """Summarize per-day dynamic transport without materializing a block."""

    family_names = (
        "sechiba_end_state",
        "final_entry_diagnostics",
        "daily_interface",
        "ok_leak_interface",
    )
    values = _dynamic_boundary_value(record)
    families = {}
    total_leaves = 0
    total_bytes = 0
    for name, value in zip(family_names, values, strict=True):
        _, leaves = _pack_dynamic_boundary(value)
        byte_count = sum(int(np.asarray(leaf).nbytes) for leaf in leaves)
        families[name] = {
            "dynamic_leaf_count": len(leaves),
            "bytes_per_day": byte_count,
        }
        total_leaves += len(leaves)
        total_bytes += byte_count
    return {
        "schema": "canonical_minimal_v0",
        "dynamic_leaf_count": total_leaves,
        "bytes_per_day": total_bytes,
        "families": families,
    }


def _dynamic_boundary_block(
    records: Sequence[PreDailyStomateReplayRecord],
    *,
    expected_spec: DynamicBoundaryNodeSpec | None = None,
) -> tuple[DynamicBoundaryNodeSpec, tuple[Any, ...]]:
    packed = tuple(_pack_dynamic_boundary(_dynamic_boundary_value(record)) for record in records)
    spec = packed[0][0]
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("Dynamic replay boundary does not match the compiled executable spec")
    if any(item[0] != spec for item in packed[1:]):
        raise ValueError("Dynamic replay boundary structure changes within a block")
    leaf_count = len(packed[0][1])
    if any(len(item[1]) != leaf_count for item in packed):
        raise ValueError("Dynamic replay boundary leaf count changes within a block")
    return spec, tuple(
        jax.numpy.stack([item[1][leaf_index] for item in packed])
        for leaf_index in range(leaf_count)
    )


def _dynamic_replay_block_inputs(
    *,
    context,
    records: Sequence[PreDailyStomateReplayRecord],
    boundary_spec: DynamicBoundaryNodeSpec,
) -> tuple[Any, Any, Any]:
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    forcing_days = jax.tree_util.tree_map(
        lambda *values: np.stack(values),
        *(
            teacher._paper_compiled_forcing_day(
                context,
                year=record.year,
                start_tstep=record.start_tstep,
                steps_per_stomate=steps_per_day,
            )
            for record in records
        ),
    )
    day_numbers = np.asarray([record.day_index for record in records], dtype=np.int32)
    _, boundary_days = _dynamic_boundary_block(records, expected_spec=boundary_spec)
    return forcing_days, day_numbers, boundary_days


def _compile_dynamic_replay_executable(
    *,
    config_path: Path,
    context,
    initial_state,
    sample_records: Sequence[PreDailyStomateReplayRecord],
) -> DynamicCompiledReplayExecutable:
    """Compile one reusable scan whose replay boundary arrays are dynamic inputs."""

    block_records = tuple(sample_records)
    if not block_records:
        raise ValueError("dynamic compiled replay requires at least one sample record")
    initial = teacher.fast_state_from_previous_packet(initial_state)
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    first_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=initial_state,
        year=block_records[0].year,
        start=block_records[0].start_tstep,
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    prebound_tables = first_inputs.hydrol_runtime_static_tables
    mineral_imin = int(prebound_tables.mineral.imin)
    mineral_imax = int(prebound_tables.mineral.imax)
    daily_carbon_dispatch = teacher._paper_daily_carbon_static_dispatch(context, initial_state)
    stomate_parameter_values = teacher._compiled_stomate_parameter_values(context)
    landpoint_payload = teacher._compiled_landpoint_payload(first_inputs.compiled_base_payload_template)
    stomate_restart_template = context.first_step_restart_state.stomate
    season_state = teacher.read_stomate_restart_season_state(context.first_step_restart_state.stomate_input)
    stomate_season_values = {
        name: value for name, value in season_state._asdict().items() if name != "provenance"
    }
    diffuco_parameter_values = teacher._compiled_diffuco_parameter_values(context)
    hydrol_table_arrays = teacher._compiled_hydrol_table_arrays(prebound_tables)
    static_metadata = block_records[0].half_hour_transition.first_step_metadata
    if static_metadata is None:
        raise ValueError("dynamic compiled replay requires first-step metadata")
    boundary_spec, sample_boundary_days = _dynamic_boundary_block(block_records)
    sample_forcing, sample_day_numbers, _ = _dynamic_replay_block_inputs(
        context=context,
        records=block_records,
        boundary_spec=boundary_spec,
    )

    def transition(
        values_by_component,
        forcing_days,
        science_day_numbers,
        boundary_days,
        stomate_parameters,
        dynamic_hydrol_table_arrays,
        dynamic_landpoint_payload,
        dynamic_stomate_restart_template,
        dynamic_stomate_season_values,
        dynamic_diffuco_parameter_values,
    ):
        def body(current_values, inputs):
            forcing, science_day_number, boundary_leaves = inputs
            state = teacher.DriverFastStateBundle(
                tstep=47,
                values_by_component=current_values,
                spec=initial.spec,
            )
            end_state, final_entry_payload, daily_fields, ok_leak_result = (
                _unpack_dynamic_boundary(boundary_spec, boundary_leaves)
            )
            dummy_t2m = jax.numpy.zeros_like(final_entry_payload["t2mdiag"])
            dummy_stempdiag = jax.numpy.zeros_like(final_entry_payload["t2mdiag"])
            transition_value = teacher.DriverRuntimeDayStepTransition(
                completed_entry_payloads=(
                    {
                        "t2m": dummy_t2m,
                        "stempdiag": dummy_stempdiag,
                        **final_entry_payload,
                    },
                ),
                current_state=end_state,
                first_step_metadata=static_metadata,
                stopped_at_tstep=None,
                state_gaps=(),
                compiled_entry_stacks={
                    "t2m": dummy_t2m,
                    "stempdiag": dummy_stempdiag,
                },
            )
            daily_fold = _minimal_daily_fold(daily_fields)
            ok_leak_result = _ok_leak_with_restart_shape(ok_leak_result, end_state)
            ok_leak_updates = teacher._paper_half_hour_ok_leak_state_updates(ok_leak_result)
            replay_values = {
                "_paper_1961_later_day_half_hour_transition": transition_value,
                "_paper_later_day_daily_process_from_completed_entries": daily_fold,
                "_paper_later_day_daily_fold_prerequisite_gaps": (),
                "stomate_maintenance_respiration_parts_from_stacks": jax.numpy.asarray(0.0),
                "_paper_half_hour_ok_leak_fold_from_entries": (
                    ok_leak_result,
                    ok_leak_updates,
                ),
            }
            with ExitStack() as stack:
                for name, value in replay_values.items():
                    stack.enter_context(
                        patch.object(teacher, name, lambda *args, _value=value, **kwargs: _value)
                    )
                day = teacher.paper_1961_driver_later_day_runtime_result(
                    config_path,
                    previous_state=state,
                    day_index=2,
                    year=1961,
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
            )

        return jax.lax.scan(
            body,
            values_by_component,
            (forcing_days, science_day_numbers, boundary_days),
        )

    executable = jax.jit(transition).lower(
        initial.values_by_component,
        sample_forcing,
        sample_day_numbers,
        sample_boundary_days,
        stomate_parameter_values,
        hydrol_table_arrays,
        landpoint_payload,
        stomate_restart_template,
        stomate_season_values,
        diffuco_parameter_values,
    ).compile()
    return DynamicCompiledReplayExecutable(
        executable=executable,
        block_size=len(block_records),
        state_spec=initial.spec,
        boundary_spec=boundary_spec,
        stomate_parameter_values=stomate_parameter_values,
        hydrol_table_arrays=hydrol_table_arrays,
        landpoint_payload=landpoint_payload,
        stomate_restart_template=stomate_restart_template,
        stomate_season_values=stomate_season_values,
        diffuco_parameter_values=diffuco_parameter_values,
    )


def _dynamic_compiled_replay_sequence(
    *,
    config_path: Path,
    initial_year_end_state,
    records: Sequence[PreDailyStomateReplayRecord],
    context,
    executable: DynamicCompiledReplayExecutable,
) -> CompiledReplaySequenceResult:
    previous_state = teacher.rebase_driver_state_for_year_start(initial_year_end_state)
    daily_outputs: list[tuple[Any, Any]] = []
    common = {
        "used_run_def_path": context.run_def_path,
        "prepared_context": context,
        "module_jit": True,
        "diffuco_local_jit": True,
        "use_static_jit_daily_carbon": True,
    }
    first = replay_pre_daily_stomate_record(
        config_path,
        previous_state=previous_state,
        record=records[0],
        **common,
    )
    previous_state = first.day_end_state
    daily_outputs.append((first.daily_modelout.modelout_fields, first.daily_modelout.modelout))
    next_index = 1
    while next_index + executable.block_size <= len(records):
        block_records = records[next_index : next_index + executable.block_size]
        forcing_days, day_numbers, boundary_days = _dynamic_replay_block_inputs(
            context=context,
            records=block_records,
            boundary_spec=executable.boundary_spec,
        )
        final_values, stacked_outputs = executable.run(
            previous_state,
            boundary_days=boundary_days,
            forcing_days=forcing_days,
            day_numbers=day_numbers,
        )
        stacked_fields, stacked_modelout = stacked_outputs
        for offset in range(executable.block_size):
            daily_outputs.append(
                (
                    jax.tree_util.tree_map(lambda value: value[offset], stacked_fields),
                    jax.tree_util.tree_map(lambda value: value[offset], stacked_modelout),
                )
            )
        previous_state = teacher.previous_packet_from_fast_state(
            teacher.DriverFastStateBundle(
                tstep=block_records[-1].expected_result.day_end_state.tstep,
                values_by_component=final_values,
                spec=executable.state_spec,
            )
        )
        next_index += executable.block_size
    while next_index < len(records):
        result = replay_pre_daily_stomate_record(
            config_path,
            previous_state=previous_state,
            record=records[next_index],
            **common,
        )
        previous_state = result.day_end_state
        daily_outputs.append((result.daily_modelout.modelout_fields, result.daily_modelout.modelout))
        next_index += 1
    _block_until_ready(previous_state.fields_by_component)
    _block_until_ready(tuple(daily_outputs))
    return CompiledReplaySequenceResult(
        last_day_end_state=previous_state,
        daily_outputs=tuple(daily_outputs),
    )


def _compiled_replay_sequence(
    *,
    config_path: Path,
    initial_year_end_state,
    records: Sequence[PreDailyStomateReplayRecord],
    context,
    blocks: Sequence[CompiledReplayBlock],
) -> CompiledReplaySequenceResult:
    previous_state = teacher.rebase_driver_state_for_year_start(initial_year_end_state)
    daily_outputs: list[tuple[Any, Any]] = []
    common = {
        "used_run_def_path": context.run_def_path,
        "prepared_context": context,
        "module_jit": True,
        "diffuco_local_jit": True,
        "use_static_jit_daily_carbon": True,
    }
    next_index = 0
    if records:
        first = replay_pre_daily_stomate_record(
            config_path,
            previous_state=previous_state,
            record=records[0],
            **common,
        )
        previous_state = first.day_end_state
        daily_outputs.append((first.daily_modelout.modelout_fields, first.daily_modelout.modelout))
        next_index = 1
    for block in blocks:
        if block.start_index != next_index:
            raise RuntimeError("compiled replay blocks are not contiguous")
        final_values, stacked_outputs = block.run(previous_state)
        stacked_fields, stacked_modelout = stacked_outputs
        for offset in range(len(block.records)):
            daily_outputs.append(
                (
                    jax.tree_util.tree_map(lambda value: value[offset], stacked_fields),
                    jax.tree_util.tree_map(lambda value: value[offset], stacked_modelout),
                )
            )
        next_index += len(block.records)
        previous_state = teacher.previous_packet_from_fast_state(
            teacher.DriverFastStateBundle(
                tstep=block.records[-1].expected_result.day_end_state.tstep,
                values_by_component=final_values,
                spec=block.state_spec,
            )
        )
    while next_index < len(records):
        result = replay_pre_daily_stomate_record(
            config_path,
            previous_state=previous_state,
            record=records[next_index],
            **common,
        )
        previous_state = result.day_end_state
        daily_outputs.append((result.daily_modelout.modelout_fields, result.daily_modelout.modelout))
        next_index += 1
    _block_until_ready(previous_state.fields_by_component)
    _block_until_ready(tuple(daily_outputs))
    return CompiledReplaySequenceResult(
        last_day_end_state=previous_state,
        daily_outputs=tuple(daily_outputs),
    )


def _teacher_sequence(
    *,
    config_path: Path,
    initial_year_end_state,
    year: int,
    days: int,
    context,
    compiled_day_block_size: int,
):
    return teacher.paper_1961_driver_restart_year_multiday_modelout_lite_run(
        config_path,
        previous_year_end_state=initial_year_end_state,
        ndays=int(days),
        year=int(year),
        used_run_def_path=context.run_def_path,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
        use_compiled_sechiba_day=True,
        compiled_day_block_size=int(compiled_day_block_size),
    )


def _time_call(callable_) -> tuple[Any, float]:
    started = time.perf_counter()
    result = callable_()
    _block_until_ready(result)
    return result, time.perf_counter() - started


def _compiled_replay_parity(
    records: Sequence[PreDailyStomateReplayRecord],
    result: CompiledReplaySequenceResult,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected_final = records[-1].expected_result.day_end_state.fields_by_component
    daily = []
    for record, (modelout_fields, modelout) in zip(records, result.daily_outputs, strict=True):
        daily.append(
            {
                "year": record.year,
                "day_index": record.day_index,
                "modelout_fields": _compare_trees(
                    record.expected_result.daily_modelout.modelout_fields,
                    modelout_fields,
                    atol=atol,
                    rtol=rtol,
                ),
                "modelout": _compare_trees(
                    record.expected_result.daily_modelout.modelout,
                    modelout,
                    atol=atol,
                    rtol=rtol,
                ),
            }
        )
    return {
        "final_day_end_state": _compare_trees(
            expected_final,
            result.last_day_end_state.fields_by_component,
            atol=atol,
            rtol=rtol,
        ),
        "daily": daily,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    cache_payload = _load_state_cache(args.state_cache)
    initial_state = cache_payload["state"]
    year = int(args.year if args.year is not None else int(cache_payload.get("end_year", 1961)) + 1)
    context = teacher.prepare_paper_1961_driver_context(
        args.config,
        used_run_def_path=args.run_def,
    )

    capture_started = time.perf_counter()
    records = _capture_sequence(
        config_path=args.config,
        initial_year_end_state=initial_state,
        year=year,
        days=args.days,
        context=context,
    )
    capture_seconds = time.perf_counter() - capture_started

    replay_compile_seconds = 0.0
    compiled_blocks: tuple[CompiledReplayBlock, ...] = ()
    dynamic_executable: DynamicCompiledReplayExecutable | None = None
    dynamic_boundary_inventory = None
    if bool(args.dynamic_replay_boundary_inputs):
        if int(args.compiled_replay_block_size) <= 0:
            raise ValueError("dynamic replay boundary inputs require compiled-replay-block-size")
        if len(records) < 1 + int(args.compiled_replay_block_size):
            raise ValueError("dynamic compiled replay requires Day 1 plus one complete replay block")
        dynamic_boundary_inventory = _dynamic_boundary_inventory(records[1])
        compile_started = time.perf_counter()
        dynamic_executable = _compile_dynamic_replay_executable(
            config_path=args.config,
            context=context,
            initial_state=records[0].expected_result.day_end_state,
            sample_records=records[1 : 1 + int(args.compiled_replay_block_size)],
        )
        replay_compile_seconds = time.perf_counter() - compile_started

        def replay_callable():
            return _dynamic_compiled_replay_sequence(
                config_path=args.config,
                initial_year_end_state=initial_state,
                records=records,
                context=context,
                executable=dynamic_executable,
            )

        replay_warm, replay_warm_seconds = _time_call(replay_callable)
        parity = _compiled_replay_parity(
            records,
            replay_warm,
            atol=args.atol,
            rtol=args.rtol,
        )
        parity_passed = parity["final_day_end_state"]["passed"] and all(
            item["modelout_fields"]["passed"] and item["modelout"]["passed"]
            for item in parity["daily"]
        )
    elif int(args.compiled_replay_block_size) > 0:
        compile_started = time.perf_counter()
        compiled_blocks = _compile_replay_blocks(
            config_path=args.config,
            initial_year_end_state=initial_state,
            records=records,
            context=context,
            block_size=args.compiled_replay_block_size,
        )
        replay_compile_seconds = time.perf_counter() - compile_started
        def replay_callable():
            return _compiled_replay_sequence(
                config_path=args.config,
                initial_year_end_state=initial_state,
                records=records,
                context=context,
                blocks=compiled_blocks,
            )

        replay_warm, replay_warm_seconds = _time_call(replay_callable)
        parity = _compiled_replay_parity(
            records,
            replay_warm,
            atol=args.atol,
            rtol=args.rtol,
        )
        parity_passed = parity["final_day_end_state"]["passed"] and all(
            item["modelout_fields"]["passed"] and item["modelout"]["passed"]
            for item in parity["daily"]
        )
    else:
        def replay_callable():
            return _replay_sequence(
                config_path=args.config,
                initial_year_end_state=initial_state,
                records=records,
                context=context,
            )

        replay_warm, replay_warm_seconds = _time_call(replay_callable)
        parity = [
            compare_replay_day(record, result, atol=args.atol, rtol=args.rtol)
            for record, result in zip(records, replay_warm, strict=True)
        ]
        parity_passed = all(
            all(item[name]["passed"] for name in ("day_end_state", "modelout_fields", "modelout"))
            for item in parity
        )
    if not parity_passed:
        raise RuntimeError("Replay parity failed; refusing to report a speed ceiling")

    _, teacher_warm_seconds = _time_call(
        lambda: _teacher_sequence(
            config_path=args.config,
            initial_year_end_state=initial_state,
            year=year,
            days=args.days,
            context=context,
            compiled_day_block_size=args.compiled_day_block_size,
        )
    )
    teacher_times: list[float] = []
    replay_times: list[float] = []
    for _ in range(args.repeats):
        _, teacher_seconds = _time_call(
            lambda: _teacher_sequence(
                config_path=args.config,
                initial_year_end_state=initial_state,
                year=year,
                days=args.days,
                context=context,
                compiled_day_block_size=args.compiled_day_block_size,
            )
        )
        _, replay_seconds = _time_call(replay_callable)
        teacher_times.append(teacher_seconds)
        replay_times.append(replay_seconds)

    teacher_median = statistics.median(teacher_times)
    replay_median = statistics.median(replay_times)
    return {
        "status": "passed",
        "scope": "research-only in-memory pre-daily-STOMATE replay",
        "teacher_git_commit": "7333b46c0b38650fb6c9250582876137831657b8",
        "teacher_label": "provisional_teacher",
        "config": str(args.config),
        "run_def": str(args.run_def),
        "state_cache": str(args.state_cache),
        "year": year,
        "days": int(args.days),
        "repeats": int(args.repeats),
        "teacher_compiled_day_block_size": int(args.compiled_day_block_size),
        "replay_mode": (
            "compiled_dynamic_boundary_scan"
            if bool(args.dynamic_replay_boundary_inputs)
            else "compiled_fixed_boundary_block"
            if int(args.compiled_replay_block_size) > 0
            else "python_day_loop"
        ),
        "replay_compiled_block_size": int(args.compiled_replay_block_size),
        "replay_compiled_block_count": (
            (len(records) - 1) // int(args.compiled_replay_block_size)
            if bool(args.dynamic_replay_boundary_inputs)
            else len(compiled_blocks)
        ),
        "replay_single_executable_reused": bool(args.dynamic_replay_boundary_inputs),
        "dynamic_boundary_inventory": dynamic_boundary_inventory,
        "replay_compile_seconds": replay_compile_seconds,
        "capture_seconds_not_timed_as_replay": capture_seconds,
        "warmup_seconds": {
            "teacher": teacher_warm_seconds,
            "replay": replay_warm_seconds,
        },
        "timing_seconds": {
            "teacher_runs": teacher_times,
            "replay_runs": replay_times,
            "teacher_median": teacher_median,
            "replay_median": replay_median,
            "median_speedup": teacher_median / replay_median,
            "teacher_seconds_per_day": teacher_median / args.days,
            "replay_seconds_per_day": replay_median / args.days,
        },
        "parity": parity,
        "limitations": [
            (
                "The canonical-minimal replay transports end-state packet arrays, 17 daily interface fields, two final-entry diagnostics, and only retained-tail OK_LEAK leaves; field-level canonical-owner deduplication remains a schema-freeze task."
                if bool(args.dynamic_replay_boundary_inputs)
                else "The Python and fixed-boundary probes inject full Teacher boundary objects; they do not define the final learnable state schema."
            ),
            (
                "Dynamic replay uses one reusable executable, but boundary generation and future learned-model inference remain excluded."
                if bool(args.dynamic_replay_boundary_inputs)
                else "Compiled replay records are compile-time constants and give an optimistic ceiling; a learned model would supply dynamic outputs."
                if int(args.compiled_replay_block_size) > 0
                else "Replay still executes Python day orchestration and boundary adapters around the retained daily tail."
            ),
            "Teacher data preparation and disk IO are excluded from replay timing.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-def", type=Path, default=DEFAULT_RUN_DEF)
    parser.add_argument("--year", type=int)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compiled-day-block-size", type=int, default=0)
    parser.add_argument("--compiled-replay-block-size", type=int, default=0)
    parser.add_argument("--dynamic-replay-boundary-inputs", action="store_true")
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.days < 1 or args.repeats < 1:
        raise ValueError("days and repeats must be positive")
    if args.compiled_day_block_size == 1 or args.compiled_day_block_size < 0:
        raise ValueError("compiled-day-block-size must be zero or at least two")
    if args.compiled_replay_block_size == 1 or args.compiled_replay_block_size < 0:
        raise ValueError("compiled-replay-block-size must be zero or at least two")
    payload = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "days": payload["days"],
                "median_speedup": payload["timing_seconds"]["median_speedup"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
