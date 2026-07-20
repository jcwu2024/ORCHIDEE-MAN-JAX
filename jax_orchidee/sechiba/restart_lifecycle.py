"""Strict scientific-state transition for SECHIBA component finalizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from jax_orchidee.sechiba.restart_io import (
    SECHIBA_RESTART_COMPONENT_FIELDS,
    SECHIBA_RESTART_TO_SOURCE_NAMES,
)


SECHIBA_FINALIZE_SOURCE_FIELDS = frozenset(
    SECHIBA_RESTART_TO_SOURCE_NAMES.get(name, name)
    for fields in SECHIBA_RESTART_COMPONENT_FIELDS.values()
    for name in fields
)


@dataclass(frozen=True)
class SechibaFinalizeStateTransition:
    """One complete 93-field transition with explicit update/carry ownership."""

    fields: Mapping[str, object]
    updated_fields: tuple[str, ...]
    carried_fields: tuple[str, ...]
    provenance_by_field: Mapping[str, tuple[str, ...]]


def transition_sechiba_finalize_source_state(
    previous: Mapping[str, object],
    *,
    updates: Mapping[str, object],
    carried_fields: tuple[str, ...],
    update_provenance: Mapping[str, tuple[str, ...]],
    carry_provenance: Mapping[str, tuple[str, ...]],
) -> SechibaFinalizeStateTransition:
    """Apply a complete transition without implicit stale-state fallback.

    Every field must be either updated by its current process owner or named in
    ``carried_fields`` with source provenance proving its SAVE/static lifetime.
    This gate deliberately rejects ``{**previous, **updates}`` style merging.

    Fortran provenance: ``sechiba.f90::sechiba_main`` lines 893-1128 and
    ``sechiba.f90::sechiba_finalize`` lines 1800-1923.
    """

    previous_names = set(previous)
    update_names = set(updates)
    carry_names = set(carried_fields)
    expected = set(SECHIBA_FINALIZE_SOURCE_FIELDS)
    if previous_names != expected:
        raise ValueError(
            "previous SECHIBA finalize state mismatch: "
            f"missing={sorted(expected - previous_names)}, extra={sorted(previous_names - expected)}"
        )
    overlap = update_names & carry_names
    if overlap:
        raise ValueError(f"SECHIBA fields cannot be both updated and carried: {sorted(overlap)}")
    actual = update_names | carry_names
    if actual != expected:
        raise ValueError(
            "SECHIBA transition ownership mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    missing_update_provenance = update_names - set(update_provenance)
    missing_carry_provenance = carry_names - set(carry_provenance)
    if missing_update_provenance or missing_carry_provenance:
        raise ValueError(
            "SECHIBA transition provenance missing: "
            f"updates={sorted(missing_update_provenance)}, carries={sorted(missing_carry_provenance)}"
        )
    fields = {name: updates[name] for name in update_names}
    fields.update({name: previous[name] for name in carry_names})
    provenance = {
        **{name: tuple(update_provenance[name]) for name in update_names},
        **{name: tuple(carry_provenance[name]) for name in carry_names},
    }
    return SechibaFinalizeStateTransition(
        fields=fields,
        updated_fields=tuple(sorted(update_names)),
        carried_fields=tuple(sorted(carry_names)),
        provenance_by_field=provenance,
    )
