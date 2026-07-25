"""Exact Teacher queries from model-visited canonical daily states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from jax_orchidee.sechiba.restart_lifecycle import SECHIBA_FINALIZE_SOURCE_FIELDS
from research.daily_coarse_graining.canonical_rollout import (
    _packet_from_canonical_state,
)
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    extract_fast_day_target,
    extract_state,
)
from research.daily_coarse_graining.supervised_learnability_pilot import (
    _capture_days,
)


@dataclass(frozen=True)
class TeacherReentryTemplates:
    """Restart-derived values that are overwritten before scientific use."""

    overwritten_finalize: Mapping[str, Any]
    daily_accumulators: Mapping[str, Any]
    steps_per_day: int


@dataclass(frozen=True)
class TeacherReentryResult:
    """One exact Teacher transition evaluated at a supplied canonical state."""

    fast_day_target: np.ndarray
    next_state: np.ndarray
    next_discrete_state: Mapping[str, np.ndarray]


def teacher_reentry_templates(context) -> TeacherReentryTemplates:
    """Build non-scientific packet templates from the landpoint restart."""

    initial_finalize = teacher.sechiba_finalize_source_state_from_restart(
        context.first_step_restart_state.sechiba_restart_state
    )
    overwritten_names = (
        SECHIBA_FINALIZE_SOURCE_FIELDS - teacher._SECHIBA_HALF_HOUR_CARRY_FIELDS
    )
    daily_accumulators = {
        name: value
        for name, value in teacher.read_stomate_daily_accumulator_state(
            context.first_step_restart_state.stomate_input
        )._asdict().items()
        if name != "provenance"
    }
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    return TeacherReentryTemplates(
        overwritten_finalize={name: initial_finalize[name] for name in overwritten_names},
        daily_accumulators=daily_accumulators,
        steps_per_day=steps_per_day,
    )


def teacher_reentry_packet(
    continuous,
    discrete: Mapping[str, np.ndarray],
    contract: DailyMarkovContract,
    *,
    tstep: int,
    templates: TeacherReentryTemplates,
):
    """Rebuild the complete packet required by the unprojected Teacher."""

    expected_template_names = (
        SECHIBA_FINALIZE_SOURCE_FIELDS - teacher._SECHIBA_HALF_HOUR_CARRY_FIELDS
    )
    if set(templates.overwritten_finalize) != expected_template_names:
        raise ValueError("Teacher re-entry overwrite template has the wrong fields")
    packet = _packet_from_canonical_state(
        continuous,
        discrete,
        contract,
        tstep=tstep,
        template_fields={
            "sechiba_finalize_state": dict(templates.overwritten_finalize),
            "slowproc_stomate_previous_step_state": {
                "daily_accumulators": dict(templates.daily_accumulators)
            },
        },
        require_complete_finalize=True,
    )
    finalize = packet.fields_by_component["sechiba_finalize_state"]
    hydrol = packet.fields_by_component["hydrol_previous_step_state"]
    if not np.array_equal(finalize["fwet_new"], hydrol["fwet_new"]):
        raise ValueError("Teacher re-entry fwet_new mirror did not come from current state")
    invalid = []
    for index, value in enumerate(jax.tree_util.tree_leaves(packet.fields_by_component)):
        dtype = np.asarray(value).dtype
        if dtype.kind not in "biufc":
            invalid.append((index, str(dtype)))
    if invalid:
        raise ValueError(
            f"Teacher re-entry packet contains non-numeric dynamic leaves: {invalid}"
        )
    accumulators = packet.fields_by_component[
        "slowproc_stomate_previous_step_state"
    ]["daily_accumulators"]
    if any(np.any(np.asarray(value)) for value in accumulators.values()):
        raise ValueError("Teacher re-entry daily accumulators were not reset to zero")
    return packet


def query_teacher_day(
    *,
    config_path: Path,
    context,
    templates: TeacherReentryTemplates,
    continuous,
    discrete: Mapping[str, np.ndarray],
    contract: DailyMarkovContract,
    year: int,
    day_index: int,
) -> TeacherReentryResult:
    """Execute the exact source-backed Teacher for one model-visited day."""

    packet = teacher_reentry_packet(
        continuous,
        discrete,
        contract,
        tstep=(int(day_index) - 1) * templates.steps_per_day - 1,
        templates=templates,
    )
    _, _, records, final_state = _capture_days(
        config_path=config_path,
        context=context,
        previous_state=packet,
        year=int(year),
        start_day=int(day_index),
        days=1,
    )
    if final_state is None or len(records) != 1:
        raise RuntimeError("Teacher re-entry did not produce one complete day")
    next_state, next_discrete = extract_state(final_state, contract)
    return TeacherReentryResult(
        fast_day_target=np.asarray(
            extract_fast_day_target(records[0], contract.fast_day_target_leaves)
        ),
        next_state=np.asarray(next_state),
        next_discrete_state={
            name: np.asarray(value) for name, value in next_discrete.items()
        },
    )
