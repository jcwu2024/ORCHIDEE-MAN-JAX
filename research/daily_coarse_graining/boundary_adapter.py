"""Research-only adapters for the daily coarse runtime boundary."""

from __future__ import annotations

from typing import Any

from jax_orchidee.driver import orchestration as teacher

HYDROL_COMPONENT = "hydrol_previous_step_state"
NROOT_FIELD = "nroot"


def promote_runtime_state_spec(day_start_state, produced_end_state):
    """Promote a year-start packet to the schema produced by the day operator.

    ``rebase_driver_state_for_year_start`` deliberately removes ``nroot``
    because the Fortran restart does not own it.  The first HYDROL transition,
    or a coarse operator replacing that transition, must produce it before a
    fixed-shape later-day executable can be built.  This adapter accepts only
    that producer-driven promotion; it never manufactures a value.
    """

    start_fields = day_start_state.fields_by_component
    produced_fields = produced_end_state.fields_by_component
    if HYDROL_COMPONENT not in start_fields or HYDROL_COMPONENT not in produced_fields:
        raise ValueError("runtime state promotion requires the HYDROL component")
    if NROOT_FIELD in start_fields[HYDROL_COMPONENT]:
        raise ValueError("runtime state promotion is only valid after a year-start nroot drop")
    if NROOT_FIELD not in produced_fields[HYDROL_COMPONENT]:
        raise ValueError("the first-day dynamic producer did not emit hydrol.nroot")

    for component, fields in start_fields.items():
        if component not in produced_fields:
            raise ValueError(f"first-day producer dropped state component {component!r}")
        missing = tuple(name for name in fields if name not in produced_fields[component])
        if missing:
            raise ValueError(
                f"first-day producer dropped fields from {component!r}: {missing!r}"
            )

    return teacher.fast_state_from_previous_packet(produced_end_state).spec


def project_packet_values(packet, spec) -> tuple[tuple[Any, ...], ...]:
    """Project a packet onto a fixed spec, rejecting every missing field."""

    projected = []
    for component, field_names in zip(
        spec.components,
        spec.field_names_by_component,
        strict=True,
    ):
        if component not in packet.fields_by_component:
            raise ValueError(f"runtime packet is missing component {component!r}")
        fields = packet.fields_by_component[component]
        missing = tuple(name for name in field_names if name not in fields)
        if missing:
            raise ValueError(f"runtime packet is missing {component!r} fields: {missing!r}")
        projected.append(tuple(fields[name] for name in field_names))
    return tuple(projected)


def packet_from_projected_values(*, values_by_component, spec, tstep: int):
    """Rebuild a packet using the promoted runtime spec."""

    return teacher.previous_packet_from_fast_state(
        teacher.DriverFastStateBundle(
            tstep=tstep,
            values_by_component=values_by_component,
            spec=spec,
        )
    )
