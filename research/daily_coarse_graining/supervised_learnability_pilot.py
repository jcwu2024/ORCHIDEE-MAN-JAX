"""Tiny supervised learnability gate for the complete daily coarse boundary.

This bounded CPU experiment uses Teacher targets only as labels.  It first
tests whether a structured full-element decoder can overfit twelve days, then
conditionally evaluates unseen consecutive days and a seven-day free rollout.
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
from research.daily_coarse_graining.boundary_adapter import project_packet_values
from research.daily_coarse_graining.daily_markov_contract import (
    extract_diagnostics,
    extract_fast_day_target,
    extract_state,
    make_compiled_training_output_projector,
)
from research.daily_coarse_graining.persistence_baseline import (
    COMMON_OK_LEAK_FIELDS,
    _legality,
    _ok_leak_output_mapping,
    _persistence_daily_interface,
    _persistence_ok_leak_interface,
)
from research.daily_coarse_graining.replay_ceiling import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_DEF,
    PreDailyStomateReplayRecord,
    _block_until_ready,
    _load_state_cache,
    _minimal_daily_fold,
    _time_call,
    capture_pre_daily_stomate_record,
)
from research.daily_coarse_graining.synthetic_operator_cost import (
    COARSE_STATE_COMPONENTS,
    FORCING_FIELDS,
    _synthetic_nroot,
    _tail_static_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "performance"
    / "daily_coarse_graining"
    / "tiny_supervised_learnability_pilot.json"
)
DEFAULT_SAMPLE_MANIFEST_OUTPUT = (
    ROOT
    / "outputs"
    / "performance"
    / "daily_coarse_graining"
    / "tiny_supervised_sample_manifest.json"
)

TRAIN_DAYS = 12
HOLDOUT_DAYS = 4
ROLLOUT_DAYS = 7
OVERFIT_THRESHOLD = 1.0e-6
HOLDOUT_OVERALL_THRESHOLD = 1.0
HOLDOUT_FAMILY_THRESHOLD = 2.0


@dataclass(frozen=True)
class BoundaryLeafSpec:
    family: str
    component: str | None
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int


@dataclass(frozen=True)
class BoundaryVectorSpec:
    leaves: tuple[BoundaryLeafSpec, ...]
    family_ranges: tuple[tuple[str, int, int], ...]
    total_size: int


@dataclass(frozen=True)
class PilotSample:
    year: int
    day_index: int
    day_start_state: Any
    forcing: Any
    record: Any
    reference: np.ndarray
    target: np.ndarray
    finite_mask: np.ndarray


class CompactTrainingCaptureBlock(NamedTuple):
    year: int
    day_indices: np.ndarray
    state_rows: np.ndarray
    discrete_rows: dict[str, np.ndarray]
    fast_day_targets: np.ndarray
    diagnostics: np.ndarray
    final_state: Any
    contract: Any | None = None
    seed_record: Any | None = None


class EncoderParameters(NamedTuple):
    state_chunk_weight: Any
    state_chunk_bias: Any
    state_chunk_position: Any
    forcing_input_weight: Any
    forcing_recurrent_weight: Any
    forcing_bias: Any
    fusion_weight: Any
    fusion_bias: Any
    fusion2_weight: Any
    fusion2_bias: Any


class LearnedParameters(NamedTuple):
    encoder: EncoderParameters
    decoder_weights: tuple[Any, ...]
    decoder_biases: tuple[Any, ...]


class LearnedModelBundle(NamedTuple):
    parameters: LearnedParameters
    state_mean: Any
    state_scale: Any
    forcing_mean: Any
    forcing_scale: Any
    target_mean: Any
    target_scale: Any


def _is_float(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return np.issubdtype(np.dtype(dtype), np.floating)


def _target_family(component: str) -> str:
    if component in {"diffuco_previous_step_state", "enerbil_previous_step_state"}:
        return "diffuco_enerbil"
    if component == "hydrol_previous_step_state":
        return "hydrol"
    if component == "thermosoil_previous_step_state":
        return "thermosoil"
    if component == "driver_previous_step_state":
        return "driver"
    if component == "sechiba_finalize_state":
        return "sechiba_finalize"
    raise ValueError(f"unknown coarse state component {component!r}")


def build_boundary_vector_spec(record) -> BoundaryVectorSpec:
    leaves = []
    cursor = 0

    def add(family: str, component: str | None, name: str, value: Any):
        nonlocal cursor
        array = np.asarray(value)
        size = int(array.size)
        leaves.append(
            BoundaryLeafSpec(
                family=family,
                component=component,
                name=name,
                shape=tuple(array.shape),
                start=cursor,
                stop=cursor + size,
            )
        )
        cursor += size

    end_fields = record.half_hour_transition.current_state.fields_by_component
    for component in COARSE_STATE_COMPONENTS:
        for name, value in end_fields[component].items():
            if _is_float(value):
                add(_target_family(component), component, name, value)
    for name, value in record.daily_fold.daily_fields.items():
        add("daily_interface", None, name, value)
    ok_values = dict(record.ok_leak_updates)
    if record.ok_leak_result.soilcarbon.perma_peat is not None:
        ok_values["deepC_peat"] = record.ok_leak_result.soilcarbon.perma_peat.deepc_peat
    for name in (*COMMON_OK_LEAK_FIELDS, "deepC_peat"):
        add("ok_leak", None, name, ok_values[name])
    final = record.half_hour_transition.completed_entry_payloads[-1]
    add("final_diagnostics", None, "t2mdiag", final["t2mdiag"])
    add("final_diagnostics", None, "temp_sol", final["temp_sol"])

    family_ranges = []
    for family in dict.fromkeys(leaf.family for leaf in leaves):
        selected = [leaf for leaf in leaves if leaf.family == family]
        family_ranges.append((family, selected[0].start, selected[-1].stop))
    return BoundaryVectorSpec(
        leaves=tuple(leaves),
        family_ranges=tuple(family_ranges),
        total_size=cursor,
    )


def _teacher_target_vector(record, spec: BoundaryVectorSpec) -> np.ndarray:
    end_fields = record.half_hour_transition.current_state.fields_by_component
    daily = record.daily_fold.daily_fields
    ok_values = dict(record.ok_leak_updates)
    if record.ok_leak_result.soilcarbon.perma_peat is not None:
        ok_values["deepC_peat"] = record.ok_leak_result.soilcarbon.perma_peat.deepc_peat
    final = record.half_hour_transition.completed_entry_payloads[-1]
    values = []
    for leaf in spec.leaves:
        if leaf.component is not None:
            value = end_fields[leaf.component][leaf.name]
        elif leaf.family == "daily_interface":
            value = daily[leaf.name]
        elif leaf.family == "ok_leak":
            value = ok_values[leaf.name]
        else:
            value = final[leaf.name]
        values.append(np.asarray(value, dtype=np.float64).reshape(-1))
    return np.concatenate(values)


def _ok_reference_mapping(state) -> dict[str, Any]:
    return _ok_leak_output_mapping(_persistence_ok_leak_interface(state))


def _reference_vector(state, forcing, spec: BoundaryVectorSpec, *, min_wind: float):
    daily = _persistence_daily_interface(state, forcing, min_wind=min_wind)
    ok_values = _ok_reference_mapping(state)
    values = []
    for leaf in spec.leaves:
        if leaf.component is not None:
            fields = state.fields_by_component[leaf.component]
            if leaf.name in fields:
                value = fields[leaf.name]
            elif leaf.component == "hydrol_previous_step_state" and leaf.name == "nroot":
                value = _synthetic_nroot(state, jnp.asarray(0.0))
            else:
                raise ValueError(
                    f"no target-free reference for {leaf.component}.{leaf.name}"
                )
        elif leaf.family == "daily_interface":
            value = daily[leaf.name]
        elif leaf.family == "ok_leak":
            value = ok_values[leaf.name]
        elif leaf.name == "t2mdiag":
            value = jnp.asarray(forcing.temp_air)[-1]
        else:
            value = state.fields_by_component["enerbil_previous_step_state"]["temp_sol"]
        values.append(jnp.asarray(value, dtype=jnp.float64).reshape(-1))
    return jnp.concatenate(values)


def _state_input_vector(state, spec: BoundaryVectorSpec):
    values = []
    for leaf in spec.leaves:
        if leaf.component is None:
            continue
        fields = state.fields_by_component[leaf.component]
        if leaf.name in fields:
            value = fields[leaf.name]
        elif leaf.component == "hydrol_previous_step_state" and leaf.name == "nroot":
            value = _synthetic_nroot(state, jnp.asarray(0.0))
        else:
            raise ValueError(f"state input is missing {leaf.component}.{leaf.name}")
        values.append(jnp.asarray(value, dtype=jnp.float64).reshape(-1))
    slow = state.fields_by_component["slowproc_stomate_previous_step_state"]
    for name in (
        "biomass",
        "npp_daily",
        "resp_maint",
        "resp_growth",
        *COMMON_OK_LEAK_FIELDS,
        "deepC_peat",
    ):
        values.append(jnp.asarray(slow[name], dtype=jnp.float64).reshape(-1))
    return jnp.concatenate(values)


def _forcing_matrix(forcing):
    columns = []
    for name in FORCING_FIELDS:
        value = jnp.asarray(getattr(forcing, name), dtype=jnp.float64)
        columns.append(value.reshape((value.shape[0], -1)))
    return jnp.concatenate(columns, axis=1)


def _safe_array(value):
    return np.nan_to_num(np.asarray(value, dtype=np.float64), nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _make_sample(*, state, forcing, record, spec, min_wind: float):
    target = _teacher_target_vector(record, spec)
    reference = np.asarray(_reference_vector(state, forcing, spec, min_wind=min_wind))
    finite_mask = np.isfinite(target) & np.isfinite(reference)
    return PilotSample(
        year=record.year,
        day_index=record.day_index,
        day_start_state=state,
        forcing=forcing,
        record=record,
        reference=_safe_array(reference),
        target=_safe_array(target),
        finite_mask=finite_mask,
    )


def _capture_days(
    *, config_path, context, previous_state, year: int, start_day: int, days: int
):
    records = []
    states = []
    forcings = []
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
    current = previous_state
    for day_index in range(start_day, start_day + days):
        forcing = teacher._paper_compiled_forcing_day(
            context,
            year=year,
            start_tstep=(day_index - 1) * steps_per_day,
            steps_per_stomate=steps_per_day,
        )
        record = capture_pre_daily_stomate_record(
            config_path,
            previous_state=current,
            year=year,
            day_index=day_index,
            start_tstep=(day_index - 1) * steps_per_day,
            compiled_forcing_series=forcing,
            **common,
        )
        _block_until_ready(record.expected_result)
        states.append(current)
        forcings.append(forcing)
        records.append(record)
        current = record.expected_result.day_end_state
    return tuple(states), tuple(forcings), tuple(records), current


def _indexed_tree(tree, index: int):
    return jax.tree_util.tree_map(lambda value: np.asarray(value[index]), tree)


def _compiled_training_record(
    *,
    year: int,
    day_index: int,
    day_start_state,
    day_end_state,
    boundary,
    boundary_state_spec,
    steps_per_day: int,
):
    end_tstep = day_index * steps_per_day - 1
    half_hour_state = teacher.previous_packet_from_fast_state(
        teacher.DriverFastStateBundle(
            tstep=end_tstep,
            values_by_component=boundary.half_hour_state_values,
            spec=boundary_state_spec,
        )
    )
    transition = SimpleNamespace(
        current_state=half_hour_state,
        completed_entry_payloads=(dict(boundary.final_diagnostics),),
    )
    ok_leak = SimpleNamespace(
        soilcarbon=SimpleNamespace(
            perma_peat=SimpleNamespace(deepc_peat=boundary.deepc_peat)
        )
    )
    return PreDailyStomateReplayRecord(
        year=int(year),
        day_index=int(day_index),
        start_tstep=(day_index - 1) * steps_per_day,
        input_state_tstep=int(day_start_state.tstep),
        half_hour_transition=transition,
        daily_fold=SimpleNamespace(daily_fields=dict(boundary.daily_fields)),
        pre_step_boundary=None,
        maintenance_resp_parts=None,
        ok_leak_result=ok_leak,
        ok_leak_updates=dict(boundary.ok_leak_updates),
        expected_result=SimpleNamespace(day_end_state=day_end_state),
    )


def _iter_capture_days_compiled_blocks(
    *,
    config_path,
    context,
    previous_state,
    year: int,
    start_day: int,
    days: int,
    block_size: int = 7,
    compact_contract_factory=None,
):
    """Yield one audited capture day and bounded compiled following-day blocks."""

    if start_day < 1:
        raise ValueError("compiled training capture start_day must be positive")
    if days < 1:
        return
    if block_size < 2:
        raise ValueError("compiled training capture block_size must be at least two")
    states, forcings, records, current = _capture_days(
        config_path=config_path,
        context=context,
        previous_state=previous_state,
        year=year,
        start_day=start_day,
        days=1,
    )
    boundary_state_spec = teacher.fast_state_from_previous_packet(
        records[0].half_hour_transition.current_state
    ).spec
    compact_contract = None
    if compact_contract_factory is None:
        yield states, forcings, records, current
    else:
        compact_contract = compact_contract_factory(states[0], records[0])
        state_row, discrete_row = extract_state(
            states[0], compact_contract, allow_year_start_missing=True
        )
        yield CompactTrainingCaptureBlock(
            year=int(year),
            day_indices=np.asarray([records[0].day_index], dtype=np.int32),
            state_rows=state_row[None, :],
            discrete_rows={name: value[None, ...] for name, value in discrete_row.items()},
            fast_day_targets=extract_fast_day_target(
                records[0], compact_contract.fast_day_target_leaves
            )[None, :],
            diagnostics=extract_diagnostics(records[0], compact_contract)[None, :],
            final_state=current,
            contract=compact_contract,
            seed_record=records[0],
        )
    if days == 1:
        return

    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    first_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=current,
        year=year,
        start=start_day * steps_per_day,
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    prebound_tables = first_inputs.hydrol_runtime_static_tables
    daily_carbon_dispatch = teacher._paper_daily_carbon_static_dispatch(
        context, current
    )
    stomate_parameters = teacher._compiled_stomate_parameter_values(context)
    landpoint_payload = teacher._compiled_landpoint_payload(
        first_inputs.compiled_base_payload_template
    )
    stomate_restart_template = context.first_step_restart_state.stomate
    stomate_season_values = {
        name: value
        for name, value in teacher.read_stomate_restart_season_state(
            context.first_step_restart_state.stomate_input
        )._asdict().items()
        if name != "provenance"
    }
    diffuco_parameters = teacher._compiled_diffuco_parameter_values(context)
    hydrol_arrays = teacher._compiled_hydrol_table_arrays(prebound_tables)
    executables = {}
    state_spec = None
    next_day = start_day + 1
    final_day = start_day + days - 1
    while next_day <= final_day:
        current_block_size = min(block_size, final_day - next_day + 1)
        block_days = tuple(range(next_day, next_day + current_block_size))
        forcing_days = tuple(
            teacher._paper_compiled_forcing_day(
                context,
                year=year,
                start_tstep=(day_index - 1) * steps_per_day,
                steps_per_stomate=steps_per_day,
            )
            for day_index in block_days
        )
        block_forcing = jax.tree_util.tree_map(
            lambda *values: np.stack(values), *forcing_days
        )
        block_day_numbers = np.asarray(block_days, dtype=np.int32)
        executable = executables.get(current_block_size)
        if executable is None:
            projector = None
            projector_key = None
            if compact_contract is not None:
                day_start_state_spec = teacher.fast_state_from_previous_packet(
                    current
                ).spec
                projector = make_compiled_training_output_projector(
                    compact_contract,
                    day_start_state_spec=day_start_state_spec,
                    boundary_state_spec=boundary_state_spec,
                )
                projector_key = compact_contract.sha256
            executable, state_spec = teacher._paper_compiled_later_day_block_executable(
                config_path,
                context=context,
                initial_state=current,
                year=year,
                block_forcing=block_forcing,
                block_day_numbers=block_day_numbers,
                prebound_hydrol_runtime_static_tables=prebound_tables,
                daily_carbon_dispatch=daily_carbon_dispatch,
                stomate_parameter_values=stomate_parameters,
                capture_pre_daily_training_boundaries=True,
                training_output_projector=projector,
                training_output_projector_key=projector_key,
            )
            executables[current_block_size] = executable
        initial_values = teacher.fast_state_from_previous_packet(
            current
        ).values_by_component
        final_values, stacked_outputs = executable(
            initial_values,
            block_forcing,
            block_day_numbers,
            stomate_parameters,
            hydrol_arrays,
            landpoint_payload,
            stomate_restart_template,
            stomate_season_values,
            diffuco_parameters,
        )
        stacked_outputs = jax.device_get(stacked_outputs)
        if compact_contract is not None:
            stacked_state, stacked_discrete, stacked_targets, stacked_diagnostics = (
                stacked_outputs
            )
            current = teacher.previous_packet_from_fast_state(
                teacher.DriverFastStateBundle(
                    tstep=(next_day + current_block_size - 1) * steps_per_day - 1,
                    values_by_component=final_values,
                    spec=state_spec,
                )
            )
            yield CompactTrainingCaptureBlock(
                year=int(year),
                day_indices=np.asarray(block_days, dtype=np.int32),
                state_rows=np.asarray(stacked_state),
                discrete_rows={
                    leaf.key: np.asarray(value)
                    for leaf, value in zip(
                        compact_contract.discrete_leaves,
                        stacked_discrete,
                        strict=True,
                    )
                },
                fast_day_targets=np.asarray(stacked_targets),
                diagnostics=np.asarray(stacked_diagnostics),
                final_state=current,
            )
            next_day += current_block_size
            continue
        stacked_boundaries, stacked_day_end_values = stacked_outputs
        block_states = []
        block_forcings = []
        block_records = []
        for offset in range(current_block_size):
            day_index = next_day + offset
            day_end_state = teacher.previous_packet_from_fast_state(
                teacher.DriverFastStateBundle(
                    tstep=day_index * steps_per_day - 1,
                    values_by_component=_indexed_tree(
                        stacked_day_end_values, offset
                    ),
                    spec=state_spec,
                )
            )
            boundary = _indexed_tree(stacked_boundaries, offset)
            forcing = _indexed_tree(block_forcing, offset)
            record = _compiled_training_record(
                year=year,
                day_index=day_index,
                day_start_state=current,
                day_end_state=day_end_state,
                boundary=boundary,
                boundary_state_spec=boundary_state_spec,
                steps_per_day=steps_per_day,
            )
            block_states.append(current)
            block_forcings.append(forcing)
            block_records.append(record)
            current = day_end_state
        current = teacher.previous_packet_from_fast_state(
            teacher.DriverFastStateBundle(
                tstep=(next_day + current_block_size - 1) * steps_per_day - 1,
                values_by_component=final_values,
                spec=state_spec,
            )
        )
        yield (
            tuple(block_states),
            tuple(block_forcings),
            tuple(block_records),
            current,
        )
        next_day += current_block_size


def _capture_days_compiled_blocks(
    *,
    config_path,
    context,
    previous_state,
    year: int,
    start_day: int,
    days: int,
    block_size: int = 7,
):
    """Compatibility collector for callers that require all captured records."""

    states = []
    forcings = []
    records = []
    current = previous_state
    for block_states, block_forcings, block_records, current in (
        _iter_capture_days_compiled_blocks(
            config_path=config_path,
            context=context,
            previous_state=previous_state,
            year=year,
            start_day=start_day,
            days=days,
            block_size=block_size,
        )
    ):
        states.extend(block_states)
        forcings.extend(block_forcings)
        records.extend(block_records)
    return tuple(states), tuple(forcings), tuple(records), current


def _normalization(values: np.ndarray):
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return mean, scale


def _initialize_encoder(state_size: int, forcing_channels: int, *, seed: int):
    rng = np.random.default_rng(seed)
    chunk_size = 64
    chunks = (state_size + chunk_size - 1) // chunk_size

    def normal(shape, scale):
        return rng.normal(0.0, scale, size=shape).astype(np.float64)

    return EncoderParameters(
        state_chunk_weight=normal((chunk_size, 32), 0.08),
        state_chunk_bias=normal((32,), 0.02),
        state_chunk_position=normal((chunks, 32), 0.04),
        forcing_input_weight=normal((forcing_channels, 32), 0.08),
        forcing_recurrent_weight=normal((32, 32), 0.05),
        forcing_bias=normal((32,), 0.02),
        fusion_weight=normal((128, 128), 0.06),
        fusion_bias=normal((128,), 0.02),
        fusion2_weight=normal((128, 128), 0.06),
        fusion2_bias=normal((128,), 0.02),
    )


def _latent(state_vector, forcing_matrix, encoder: EncoderParameters):
    chunk_size = encoder.state_chunk_weight.shape[0]
    padded_size = encoder.state_chunk_position.shape[0] * chunk_size
    padded = jnp.pad(state_vector, (0, padded_size - state_vector.shape[0]))
    chunks = padded.reshape((encoder.state_chunk_position.shape[0], chunk_size))
    state_tokens = jnp.tanh(
        chunks @ encoder.state_chunk_weight
        + encoder.state_chunk_bias
        + encoder.state_chunk_position
    )
    state_latent = jnp.concatenate(
        (jnp.mean(state_tokens, axis=0), jnp.max(state_tokens, axis=0))
    )

    def forcing_step(carry, value):
        hidden = jnp.tanh(
            value @ encoder.forcing_input_weight
            + carry @ encoder.forcing_recurrent_weight
            + encoder.forcing_bias
        )
        return hidden, hidden

    initial = jnp.zeros((encoder.forcing_bias.shape[0],), dtype=jnp.float64)
    last, sequence = jax.lax.scan(forcing_step, initial, forcing_matrix)
    forcing_latent = jnp.concatenate((last, jnp.mean(sequence, axis=0)))
    hidden = jnp.tanh(
        jnp.concatenate((state_latent, forcing_latent)) @ encoder.fusion_weight
        + encoder.fusion_bias
    )
    return jnp.tanh(hidden @ encoder.fusion2_weight + encoder.fusion2_bias)


def _normalized_inputs(sample, spec, normalization):
    state = _safe_array(_state_input_vector(sample.day_start_state, spec))
    forcing = _safe_array(_forcing_matrix(sample.forcing))
    state_mean, state_scale, forcing_mean, forcing_scale = normalization
    return (state - state_mean) / state_scale, (forcing - forcing_mean) / forcing_scale


def fit_tiny_model(samples: Sequence[PilotSample], spec: BoundaryVectorSpec, *, seed: int):
    state_inputs = np.stack(
        [_safe_array(_state_input_vector(sample.day_start_state, spec)) for sample in samples]
    )
    forcing_inputs = np.stack(
        [_safe_array(_forcing_matrix(sample.forcing)) for sample in samples]
    )
    deltas = np.stack([sample.target - sample.reference for sample in samples])
    masks = np.stack([sample.finite_mask for sample in samples])
    state_mean, state_scale = _normalization(state_inputs)
    forcing_mean, forcing_scale = _normalization(
        forcing_inputs.reshape((-1, forcing_inputs.shape[-1]))
    )
    target_mean, target_scale = _normalization(deltas)
    target_normalized = (deltas - target_mean) / target_scale
    target_normalized = np.where(masks, target_normalized, 0.0)

    encoder = _initialize_encoder(
        state_inputs.shape[1], forcing_inputs.shape[-1], seed=seed
    )
    latent_fn = jax.jit(
        jax.vmap(lambda state, forcing: _latent(state, forcing, encoder))
    )
    normalized_states = (state_inputs - state_mean) / state_scale
    normalized_forcing = (forcing_inputs - forcing_mean) / forcing_scale
    latents = np.asarray(latent_fn(normalized_states, normalized_forcing))
    design = np.concatenate((latents, np.ones((len(samples), 1))), axis=1)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design))
    coefficients = np.linalg.pinv(design, rcond=1.0e-12) @ target_normalized
    decoder_weights = []
    decoder_biases = []
    for _, start, stop in spec.family_ranges:
        decoder_weights.append(coefficients[:-1, start:stop])
        decoder_biases.append(coefficients[-1, start:stop])
    parameters = LearnedParameters(
        encoder=encoder,
        decoder_weights=tuple(decoder_weights),
        decoder_biases=tuple(decoder_biases),
    )
    bundle = LearnedModelBundle(
        parameters=parameters,
        state_mean=state_mean,
        state_scale=state_scale,
        forcing_mean=forcing_mean,
        forcing_scale=forcing_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    return bundle, {
        "design_rank": rank,
        "design_rows": len(samples),
        "design_condition_number": condition,
        "fitting_method": "train_only_svd_pseudoinverse_structured_linear_decoder",
    }


def predict_normalized(state, forcing, bundle: LearnedModelBundle, spec):
    state_vector = jnp.nan_to_num(
        _state_input_vector(state, spec), nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )
    forcing_matrix = jnp.nan_to_num(
        _forcing_matrix(forcing), nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )
    state_normalized = (state_vector - bundle.state_mean) / bundle.state_scale
    forcing_normalized = (forcing_matrix - bundle.forcing_mean) / bundle.forcing_scale
    latent = _latent(state_normalized, forcing_normalized, bundle.parameters.encoder)
    outputs = []
    for weight, bias in zip(
        bundle.parameters.decoder_weights,
        bundle.parameters.decoder_biases,
        strict=True,
    ):
        outputs.append(latent @ weight + bias)
    return jnp.concatenate(outputs)


def predict_boundary_vector(state, forcing, bundle, spec, *, min_wind: float):
    normalized = predict_normalized(state, forcing, bundle, spec)
    delta = bundle.target_mean + bundle.target_scale * normalized
    return _reference_vector(state, forcing, spec, min_wind=min_wind) + delta


def _family_metrics(samples, predictions, bundle, spec):
    targets = np.stack([sample.target for sample in samples])
    masks = np.stack([sample.finite_mask for sample in samples])
    raw_predictions = np.stack([np.asarray(value, dtype=np.float64) for value in predictions])
    predictions = np.stack([_safe_array(value) for value in predictions])
    target_normalized = (
        (targets - np.stack([sample.reference for sample in samples]))
        - np.asarray(bundle.target_mean)
    ) / np.asarray(bundle.target_scale)
    predicted_normalized = (
        (predictions - np.stack([sample.reference for sample in samples]))
        - np.asarray(bundle.target_mean)
    ) / np.asarray(bundle.target_scale)
    result = {}
    for family, start, stop in spec.family_ranges:
        mask = masks[:, start:stop]
        error_n = predicted_normalized[:, start:stop] - target_normalized[:, start:stop]
        error = predictions[:, start:stop] - targets[:, start:stop]
        selected_n = error_n[mask]
        selected = error[mask]
        selected_target = targets[:, start:stop][mask]
        result[family] = {
            "normalized_rmse": float(np.sqrt(np.mean(selected_n**2))),
            "max_absolute_error": float(np.max(np.abs(selected))),
            "max_relative_error": float(
                np.max(np.abs(selected) / np.maximum(np.abs(selected_target), 1.0e-12))
            ),
            "finite_elements": int(mask.sum()),
            "masked_nonfinite_elements": int(mask.size - mask.sum()),
            "predicted_nonfinite_count": int(
                np.count_nonzero(~np.isfinite(raw_predictions[:, start:stop]))
            ),
        }
    all_selected = np.concatenate(
        [
            (
                predicted_normalized[:, start:stop]
                - target_normalized[:, start:stop]
            )[masks[:, start:stop]]
            for _, start, stop in spec.family_ranges
        ]
    )
    result["overall"] = {
        "normalized_rmse": float(np.sqrt(np.mean(all_selected**2))),
        "max_absolute_error": float(np.max(np.abs(predictions - targets))),
    }
    return result


def _selected_field_metrics(samples, predictions, bundle, spec):
    selected_state = {
        "diffuco_previous_step_state": {"temp_sol", "qsurf", "gpp"},
        "enerbil_previous_step_state": {"temp_sol", "qsurf", "soilcap"},
        "hydrol_previous_step_state": {"mc", "mcl", "nroot", "soil_mc", "snow"},
        "thermosoil_previous_step_state": {"ptn", "stempdiag", "tdeep", "hsdeep"},
    }
    targets = np.stack([sample.target for sample in samples])
    references = np.stack([sample.reference for sample in samples])
    masks = np.stack([sample.finite_mask for sample in samples])
    predicted = np.stack([_safe_array(value) for value in predictions])
    target_mean = np.asarray(bundle.target_mean)
    target_scale = np.asarray(bundle.target_scale)
    result = {}
    for leaf in spec.leaves:
        include = leaf.family in {"daily_interface", "ok_leak", "final_diagnostics"}
        if leaf.component in selected_state:
            include = leaf.name in selected_state[leaf.component]
        if not include:
            continue
        key = (
            f"{leaf.component}.{leaf.name}"
            if leaf.component is not None
            else f"{leaf.family}.{leaf.name}"
        )
        slc = slice(leaf.start, leaf.stop)
        mask = masks[:, slc]
        error = predicted[:, slc] - targets[:, slc]
        normalized_error = (
            (predicted[:, slc] - references[:, slc] - target_mean[slc])
            / target_scale[slc]
            - (targets[:, slc] - references[:, slc] - target_mean[slc])
            / target_scale[slc]
        )
        actual_error = error[mask]
        actual_target = targets[:, slc][mask]
        result[key] = {
            "normalized_rmse": float(np.sqrt(np.mean(normalized_error[mask] ** 2))),
            "max_absolute_error": float(np.max(np.abs(actual_error))),
            "max_relative_error": float(
                np.max(
                    np.abs(actual_error)
                    / np.maximum(np.abs(actual_target), 1.0e-12)
                )
            ),
        }
    return result


def _unpack_prediction(vector, state, spec: BoundaryVectorSpec):
    fields = {
        component: dict(component_fields)
        for component, component_fields in state.fields_by_component.items()
    }
    daily = {}
    ok_values = {}
    final = {}
    for leaf in spec.leaves:
        value = jnp.asarray(vector[leaf.start : leaf.stop]).reshape(leaf.shape)
        if leaf.component is not None:
            fields[leaf.component][leaf.name] = value
        elif leaf.family == "daily_interface":
            daily[leaf.name] = value
        elif leaf.family == "ok_leak":
            ok_values[leaf.name] = value
        else:
            final[leaf.name] = value
    end_state = teacher.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=fields,
        provenance_by_component=state.provenance_by_component,
    )
    ok = SimpleNamespace(
        littercalc=SimpleNamespace(
            litter_above=ok_values["litter_above"],
            litter_below=ok_values["litter_below"],
            lignin_struc_above=ok_values["lignin_struc_above"],
            lignin_struc_below=ok_values["lignin_struc_below"],
            litterpart=ok_values["litterpart"],
            dead_leaves=ok_values["dead_leaves"],
            fuel=SimpleNamespace(
                fuel_1hr=ok_values["fuel_1hr"],
                fuel_10hr=ok_values["fuel_10hr"],
                fuel_100hr=ok_values["fuel_100hr"],
                fuel_1000hr=ok_values["fuel_1000hr"],
            ),
        ),
        soilcarbon=SimpleNamespace(
            carbon_32l=ok_values["carbon_32l"],
            doc=ok_values["DOC"],
            resp_hetero_soil=state.fields_by_component[
                "slowproc_stomate_previous_step_state"
            ]["resp_hetero"],
            perma_peat=SimpleNamespace(deepc_peat=ok_values["deepC_peat"]),
        ),
        interception_storage=ok_values["interception_storage"],
    )
    return end_state, daily, ok, final


def _stock_proxies(end_state, ok_leak):
    hydrol = end_state.fields_by_component["hydrol_previous_step_state"]
    water = sum(
        float(np.sum(np.asarray(hydrol[name], dtype=np.float64)))
        for name in ("mc", "mcl", "snow", "flood_res")
        if name in hydrol
    )
    ok_values = _ok_leak_output_mapping(ok_leak)
    carbon_names = (
        "litter_above",
        "litter_below",
        "dead_leaves",
        "carbon_32l",
        "DOC",
        "deepC_peat",
    )
    carbon = sum(
        float(np.sum(np.asarray(ok_values[name], dtype=np.float64)))
        for name in carbon_names
    )
    return {"water_storage_proxy": water, "ok_leak_carbon_stock_proxy": carbon}


def _holdout_boundary_diagnostics(samples, predictions, spec):
    days = []
    water_relative = []
    carbon_relative = []
    for sample, prediction in zip(samples, predictions, strict=True):
        predicted_state, predicted_daily, predicted_ok, _ = _unpack_prediction(
            prediction, sample.day_start_state, spec
        )
        teacher_state = sample.record.half_hour_transition.current_state
        teacher_ok = sample.record.ok_leak_result
        predicted_stocks = _stock_proxies(predicted_state, predicted_ok)
        teacher_stocks = _stock_proxies(teacher_state, teacher_ok)
        water_error = (
            predicted_stocks["water_storage_proxy"]
            - teacher_stocks["water_storage_proxy"]
        ) / max(abs(teacher_stocks["water_storage_proxy"]), 1.0e-12)
        carbon_error = (
            predicted_stocks["ok_leak_carbon_stock_proxy"]
            - teacher_stocks["ok_leak_carbon_stock_proxy"]
        ) / max(abs(teacher_stocks["ok_leak_carbon_stock_proxy"]), 1.0e-12)
        water_relative.append(water_error)
        carbon_relative.append(carbon_error)
        legality = _legality(predicted_state)
        days.append(
            {
                "day_index": sample.day_index,
                "state_legality": legality,
                "stocks": {
                    "predicted": predicted_stocks,
                    "teacher": teacher_stocks,
                    "water_relative_error": water_error,
                    "ok_leak_carbon_relative_error": carbon_error,
                },
                "agb_bgb_predriver": {
                    "biomass_carry_exact": bool(
                        np.array_equal(
                            np.asarray(
                                predicted_state.fields_by_component[
                                    "slowproc_stomate_previous_step_state"
                                ]["biomass"]
                            ),
                            np.asarray(
                                sample.day_start_state.fields_by_component[
                                    "slowproc_stomate_previous_step_state"
                                ]["biomass"]
                            ),
                        )
                    ),
                },
                "gpp_npp_predriver": {
                    "gpp_daily_finite": bool(
                        np.all(np.isfinite(np.asarray(predicted_daily["gpp_daily"])))
                    ),
                    "resp_maint_part_finite": bool(
                        np.all(
                            np.isfinite(np.asarray(predicted_daily["resp_maint_part"]))
                        )
                    ),
                },
            }
        )
    return {
        "days": days,
        "water_proxy_max_absolute_relative_error": float(
            np.max(np.abs(water_relative))
        ),
        "ok_leak_carbon_proxy_max_absolute_relative_error": float(
            np.max(np.abs(carbon_relative))
        ),
        "warning": "These are stock/legality proxies, not frozen unit-closed budget equations.",
    }


def _learned_boundary(state, forcing, metadata, bundle, spec, *, min_wind: float):
    normalized = predict_normalized(state, forcing, bundle, spec)
    vector = predict_boundary_vector(
        state, forcing, bundle, spec, min_wind=min_wind
    )
    end_state, daily, ok_leak, final = _unpack_prediction(vector, state, spec)
    stempdiag = end_state.fields_by_component["thermosoil_previous_step_state"][
        "stempdiag"
    ]
    transition = teacher.DriverRuntimeDayStepTransition(
        completed_entry_payloads=(
            {
                "t2m": final["t2mdiag"],
                "stempdiag": stempdiag,
                "t2mdiag": final["t2mdiag"],
                "temp_sol": final["temp_sol"],
            },
        ),
        current_state=end_state,
        first_step_metadata=metadata,
        stopped_at_tstep=None,
        state_gaps=(),
        compiled_entry_stacks={"t2m": final["t2mdiag"], "stempdiag": stempdiag},
    )
    checksum = jnp.sum(jnp.tanh(jnp.nan_to_num(normalized))) + jnp.sum(
        jnp.tanh(jnp.nan_to_num(vector))
    )
    return transition, _minimal_daily_fold(daily), ok_leak, checksum


def _compile_learned_one_day(
    *, config_path, context, initial_state, forcing_batch, day_numbers, bundle, spec
):
    initial = teacher.fast_state_from_previous_packet(initial_state)
    static = _tail_static_inputs(
        context, initial_state, forcing_batch, first_day=False
    )

    def transition(
        values_by_component,
        forcing_days,
        science_day_numbers,
        dynamic_bundle,
        stomate_parameters,
        dynamic_hydrol_table_arrays,
        dynamic_landpoint_payload,
        dynamic_stomate_restart_template,
        dynamic_stomate_season_values,
        dynamic_diffuco_parameter_values,
    ):
        forcing = jax.tree_util.tree_map(lambda value: value[0], forcing_days)
        state = teacher.DriverFastStateBundle(
            tstep=47,
            values_by_component=values_by_component,
            spec=initial.spec,
        )
        previous_packet = teacher.previous_packet_from_fast_state(state)
        transition_value, daily_fold, ok_leak_result, checksum = _learned_boundary(
            previous_packet,
            forcing,
            static["metadata"],
            dynamic_bundle,
            spec,
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
                day_index=2,
                year=1962,
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
                model_day_number=science_day_numbers[0],
                compiled_stomate_parameter_values=stomate_parameters,
                compiled_landpoint_payload=dynamic_landpoint_payload,
                compiled_stomate_restart_template=dynamic_stomate_restart_template,
                compiled_stomate_season_template=teacher.StomateRestartSeasonState(
                    **dynamic_stomate_season_values,
                    provenance=static["season"].provenance,
                ),
                compiled_diffuco_parameter_values=dynamic_diffuco_parameter_values,
            )
        return project_packet_values(day.day_end_state, initial.spec), (
            day.daily_modelout.modelout_fields,
            day.daily_modelout.modelout,
            checksum,
        )

    started = time.perf_counter()
    executable = jax.jit(transition).lower(
        initial.values_by_component,
        forcing_batch,
        day_numbers,
        bundle,
        static["stomate_parameter_values"],
        static["hydrol_table_arrays"],
        static["landpoint_payload"],
        static["stomate_restart_template"],
        static["stomate_season_values"],
        static["diffuco_parameter_values"],
    ).compile()
    return executable, initial, static, time.perf_counter() - started


def _runtime_benchmark(*, config_path, context, sample, bundle, spec, repeats: int):
    forcing_batch = jax.tree_util.tree_map(
        lambda value: jnp.expand_dims(jnp.asarray(value), axis=0), sample.forcing
    )
    day_numbers = jnp.asarray([sample.day_index], dtype=jnp.int32)
    executable, initial, static, compile_seconds = _compile_learned_one_day(
        config_path=config_path,
        context=context,
        initial_state=sample.day_start_state,
        forcing_batch=forcing_batch,
        day_numbers=day_numbers,
        bundle=bundle,
        spec=spec,
    )

    def call():
        return executable(
            initial.values_by_component,
            forcing_batch,
            day_numbers,
            bundle,
            static["stomate_parameter_values"],
            static["hydrol_table_arrays"],
            static["landpoint_payload"],
            static["stomate_restart_template"],
            static["stomate_season_values"],
            static["diffuco_parameter_values"],
        )

    warm, warm_seconds = _time_call(call)
    checksum = float(np.asarray(warm[1][2]))
    hot = []
    for _ in range(repeats):
        _, elapsed = _time_call(call)
        hot.append(elapsed)
    return {
        "scope": "trained_decoder_plus_retained_tail_one_step_only_after_holdout_failure",
        "compile_seconds": compile_seconds,
        "warm_seconds": warm_seconds,
        "hot_runs_seconds": hot,
        "hot_median_seconds": statistics.median(hot),
        "blocked_checksum": checksum,
        "blocked_checksum_finite": bool(np.isfinite(checksum)),
        "rollout_performed": False,
        "d0554d0_49x_reused": False,
    }


def _exact_discrete_metrics(samples, spec):
    mismatches = []
    checked = 0
    for sample in samples:
        target_fields = sample.record.half_hour_transition.current_state.fields_by_component
        for component in COARSE_STATE_COMPONENTS:
            for name, target in target_fields[component].items():
                if _is_float(target):
                    continue
                start = sample.day_start_state.fields_by_component[component][name]
                actual = np.asarray(start)
                checked += int(np.asarray(target).size)
                if not np.array_equal(actual, np.asarray(target)):
                    mismatches.append(
                        {
                            "day_index": sample.day_index,
                            "field": f"{component}.{name}",
                        }
                    )
    return {"checked_elements": checked, "mismatches": mismatches, "passed": not mismatches}


def _parameter_count(bundle: LearnedModelBundle) -> int:
    return int(
        sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(bundle.parameters))
    )


def _schema_inventory(spec):
    result = {}
    for family, start, stop in spec.family_ranges:
        result[family] = {
            "elements": stop - start,
            "leaves": sum(leaf.family == family for leaf in spec.leaves),
        }
    return result


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _sample_manifest(samples):
    return [
        {
            "year": sample.year,
            "day_index": sample.day_index,
            "input_state_tstep": sample.record.input_state_tstep,
            "target_role": "supervision_label_only",
        }
        for sample in samples
    ]


def run_experiment(args):
    cache = _load_state_cache(args.state_cache)
    initial_year_end = cache["state"]
    context = teacher.prepare_paper_1961_driver_context(
        args.config, used_run_def_path=args.run_def
    )
    initial_state = teacher.rebase_driver_state_for_year_start(initial_year_end)
    capture_started = time.perf_counter()
    states, forcings, records, current = _capture_days(
        config_path=args.config,
        context=context,
        previous_state=initial_state,
        year=args.year,
        start_day=1,
        days=TRAIN_DAYS,
    )
    train_capture_seconds = time.perf_counter() - capture_started
    spec = build_boundary_vector_spec(records[0])
    train_samples = tuple(
        _make_sample(
            state=state,
            forcing=forcing,
            record=record,
            spec=spec,
            min_wind=context.min_wind,
        )
        for state, forcing, record in zip(states, forcings, records, strict=True)
    )
    fit_started = time.perf_counter()
    bundle, fit_info = fit_tiny_model(train_samples, spec, seed=args.seed)
    fit_seconds = time.perf_counter() - fit_started
    train_predictions = [
        np.asarray(
            predict_boundary_vector(
                sample.day_start_state,
                sample.forcing,
                bundle,
                spec,
                min_wind=context.min_wind,
            )
        )
        for sample in train_samples
    ]
    train_metrics = _family_metrics(train_samples, train_predictions, bundle, spec)
    discrete_train = _exact_discrete_metrics(train_samples, spec)
    overfit_passed = (
        train_metrics["overall"]["normalized_rmse"] <= OVERFIT_THRESHOLD
        and discrete_train["passed"]
    )

    result = {
        "status": "completed",
        "decision": "tiny_overfit_failed_representation_inadequate",
        "teacher_commit": "7333b46c0b38650fb6c9250582876137831657b8",
        "experiment_git_head": _git_head(),
        "scope": "PFT14 point 001.0-071.0, local CPU tiny supervised boundary pilot",
        "constraints": {
            "teacher_targets_are_labels_only": True,
            "teacher_targets_as_model_inputs": False,
            "teacher_daily_rollout_correction": False,
            "server_or_gpu_used": False,
            "large_dataset_written": False,
        },
        "schema": {
            "continuous_elements": spec.total_size,
            "continuous_leaves": len(spec.leaves),
            "families": _schema_inventory(spec),
            "decoder_type": "full_element_structured_residual_decoder",
            "broadcast_coefficient_heads": False,
        },
        "model": {
            "parameter_count": _parameter_count(bundle),
            "latent_size": 128,
            "state_encoder": "element_standardization_plus_ordered_64-element_chunks",
            "forcing_encoder": "48-step recurrent temporal encoder",
            "decoder": "separate full dense head per process family",
            **fit_info,
        },
        "normalization": {
            "source": "training_days_only",
            "training_days": list(range(1, TRAIN_DAYS + 1)),
            "holdout_days_excluded": True,
        },
        "gates": {
            "tiny_overfit": {
                "threshold_normalized_rmse": OVERFIT_THRESHOLD,
                "passed": overfit_passed,
                "metrics": train_metrics,
                "exact_discrete": discrete_train,
            },
            "one_step_holdout": {"status": "not_run_overfit_failed"},
            "seven_day_free_rollout": {"status": "not_run_prior_gate_failed"},
        },
        "sample_manifest": {
            "train": _sample_manifest(train_samples),
            "holdout": [],
            "rollout": [],
            "binary_samples_committed": False,
        },
        "timing_seconds": {
            "teacher_train_capture": train_capture_seconds,
            "model_fit": fit_seconds,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
        },
        "limitations": [
            "The decoder is fitted by train-only SVD pseudoinverse to make the tiny overfit gate diagnostic and deterministic.",
            "Passing tiny overfit proves representation capacity only, not generalization or scientific usability.",
        ],
    }
    if not overfit_passed:
        return result

    holdout_capture_started = time.perf_counter()
    h_states, h_forcings, h_records, current = _capture_days(
        config_path=args.config,
        context=context,
        previous_state=current,
        year=args.year,
        start_day=TRAIN_DAYS + 1,
        days=HOLDOUT_DAYS,
    )
    holdout_capture_seconds = time.perf_counter() - holdout_capture_started
    holdout_samples = tuple(
        _make_sample(
            state=state,
            forcing=forcing,
            record=record,
            spec=spec,
            min_wind=context.min_wind,
        )
        for state, forcing, record in zip(h_states, h_forcings, h_records, strict=True)
    )
    holdout_predictions = [
        np.asarray(
            predict_boundary_vector(
                sample.day_start_state,
                sample.forcing,
                bundle,
                spec,
                min_wind=context.min_wind,
            )
        )
        for sample in holdout_samples
    ]
    holdout_metrics = _family_metrics(
        holdout_samples, holdout_predictions, bundle, spec
    )
    discrete_holdout = _exact_discrete_metrics(holdout_samples, spec)
    family_skill = all(
        metrics["normalized_rmse"] <= HOLDOUT_FAMILY_THRESHOLD
        for family, metrics in holdout_metrics.items()
        if family != "overall"
    )
    holdout_passed = (
        holdout_metrics["overall"]["normalized_rmse"] <= HOLDOUT_OVERALL_THRESHOLD
        and family_skill
        and discrete_holdout["passed"]
    )
    result["decision"] = (
        "one_step_skill_passed_rollout_failed"
        if holdout_passed
        else "tiny_overfit_passed_holdout_failed"
    )
    key_field_metrics = _selected_field_metrics(
        holdout_samples, holdout_predictions, bundle, spec
    )
    boundary_diagnostics = _holdout_boundary_diagnostics(
        holdout_samples, holdout_predictions, spec
    )
    result["gates"]["one_step_holdout"] = {
        "days": list(range(TRAIN_DAYS + 1, TRAIN_DAYS + HOLDOUT_DAYS + 1)),
        "overall_threshold": HOLDOUT_OVERALL_THRESHOLD,
        "family_threshold": HOLDOUT_FAMILY_THRESHOLD,
        "passed": holdout_passed,
        "metrics": holdout_metrics,
        "key_field_metrics": key_field_metrics,
        "exact_discrete": discrete_holdout,
        "boundary_diagnostics": boundary_diagnostics,
        "agb_bgb_gpp_npp_predrivers": {
            "agb_bgb_biomass_carry_exact_all_days": all(
                day["agb_bgb_predriver"]["biomass_carry_exact"]
                for day in boundary_diagnostics["days"]
            ),
            "gpp_daily": key_field_metrics["daily_interface.gpp_daily"],
            "npp_maintenance_driver": key_field_metrics[
                "daily_interface.resp_maint_part"
            ],
            "scope_note": "These are pre-daily-STOMATE drivers; retained Teacher computes final AGB/BGB/GPP/NPP modelout.",
        },
    }
    result["sample_manifest"]["holdout"] = _sample_manifest(holdout_samples)
    result["timing_seconds"]["teacher_holdout_capture"] = holdout_capture_seconds
    result["trained_runtime_cost"] = _runtime_benchmark(
        config_path=args.config,
        context=context,
        sample=holdout_samples[0],
        bundle=bundle,
        spec=spec,
        repeats=3,
    )
    if not holdout_passed:
        return result

    result["gates"]["seven_day_free_rollout"] = {
        "status": "implementation_required_after_one_step_pass"
    }
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-def", type=Path, default=DEFAULT_RUN_DEF)
    parser.add_argument("--year", type=int, default=1962)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-manifest-output",
        type=Path,
        default=DEFAULT_SAMPLE_MANIFEST_OUTPUT,
    )
    return parser


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    payload = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sample_manifest = {
        "schema_version": "tiny_supervised_daily_boundary_samples_v1",
        "teacher_commit": payload["teacher_commit"],
        "experiment_git_head": payload["experiment_git_head"],
        "normalization_source": payload["normalization"]["source"],
        "continuous_schema": payload["schema"],
        "samples": payload["sample_manifest"],
        "teacher_targets_are_labels_only": True,
        "binary_samples_committed": False,
    }
    args.sample_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.sample_manifest_output.write_text(
        json.dumps(sample_manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "overfit_rmse": payload["gates"]["tiny_overfit"]["metrics"][
                    "overall"
                ]["normalized_rmse"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
