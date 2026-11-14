"""Source-backed scalar and landpoint calendar assignments."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


SOURCE_CALENDAR_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 556-610",
)


class CalendarAssignments(NamedTuple):
    growth_day: jnp.ndarray
    peatC_ok: jnp.ndarray
    first_ts_year: jnp.ndarray
    last_ts_day: jnp.ndarray
    end_of_year: jnp.ndarray


def source_calendar_assignments(
    *,
    growth_day,
    peatC_ok,
    temp_growth,
    peatC,
    month,
    day,
    sec,
    dt_sechiba,
    active=True,
    dynpeat_pc=False,
    judge_pc=False,
    pc_lim=50.0,
) -> CalendarAssignments:
    """Apply ``slowproc_main`` calendar assignments to landpoint memories.

    Fortran provenance: ``slowproc.f90::slowproc_main`` lines 556-610.

    ``active=False`` represents initialization: it emits no calendar flags and
    leaves both mutable memories unchanged. January 1 midnight remains the
    previous year's final day; the year reset occurs at the next first active
    timestep.
    """
    growth_day = jnp.asarray(growth_day)
    peatC_ok = jnp.asarray(peatC_ok)
    temp_growth = jnp.asarray(temp_growth)
    peatC = jnp.asarray(peatC)
    if growth_day.ndim != 1 or peatC_ok.ndim != 1 or temp_growth.ndim != 1 or peatC.ndim != 1:
        raise ValueError("calendar memories and inputs must be landpoint vectors")
    if not (growth_day.shape == peatC_ok.shape == temp_growth.shape == peatC.shape):
        raise ValueError("calendar memories and inputs must have matching landpoint shapes")

    active = jnp.asarray(active, dtype=jnp.bool_)
    month = jnp.asarray(month)
    day = jnp.asarray(day)
    sec = jnp.asarray(sec)
    dt_sechiba = jnp.asarray(dt_sechiba)

    first_ts_year = active & (sec == dt_sechiba) & (month == 1) & (day == 1)
    jan_second_first_step = active & (sec == dt_sechiba) & (month == 1) & (day == 2)
    last_ts_day = active & (sec == 0)
    end_of_year = last_ts_day & (month == 1) & (day == 1)

    growth_day = jnp.where(first_ts_year, jnp.zeros_like(growth_day), growth_day)
    growth_day = jnp.where(
        last_ts_day,
        growth_day + jnp.where(temp_growth > 5.0, 1.0, 0.0),
        growth_day,
    )

    peatC_gate = jnp.where(
        jnp.asarray(dynpeat_pc, dtype=jnp.bool_),
        jnp.where(
            jnp.asarray(judge_pc, dtype=jnp.bool_),
            jnp.where(peatC >= jnp.asarray(pc_lim), jnp.ones_like(peatC_ok), jnp.zeros_like(peatC_ok)),
            peatC_ok,
        ),
        jnp.ones_like(peatC_ok),
    )
    peatC_ok = jnp.where(jan_second_first_step, peatC_gate, peatC_ok)

    return CalendarAssignments(
        growth_day=growth_day,
        peatC_ok=peatC_ok,
        first_ts_year=first_ts_year,
        last_ts_day=last_ts_day,
        end_of_year=end_of_year,
    )
