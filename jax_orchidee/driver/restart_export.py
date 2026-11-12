"""Strict conversion from a day-end packet to STOMATE restart reader states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from jax_orchidee.parameters.pft_catalog import PFTRunLayout
from jax_orchidee.stomate.reference import (
    StomateDailyAccumulatorState,
    StomateRestartEntryState,
    StomateRestartSeasonState,
)
from jax_orchidee.stomate.restart_io import (
    DEFAULT_STOMATE_RESTART_SCHEMA,
    StomateReadstartStates,
    StomateRestartPhysicalState,
    StomateRestartWriteReport,
    write_stomate_full_writerestart_states,
)


@dataclass(frozen=True)
class StomateRestartPacketMergeReport:
    entry_updates: tuple[str, ...]
    season_updates: tuple[str, ...]
    daily_updates: tuple[str, ...]
    packet_only_fields: tuple[str, ...]
    retained_base_fields: tuple[str, ...]
    derived_fields: tuple[str, ...] = ()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    fields = getattr(value, "fields", None)
    if isinstance(fields, Mapping):
        return fields
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        return asdict()
    raise TypeError(f"{name} must be a mapping or expose mapping-valued fields/_asdict")


def _replace_from_mapping(
    state: object, values: Mapping[str, object]
) -> tuple[object, tuple[str, ...]]:
    field_names = set(state._fields) - {"provenance"}
    updates = {name: values[name] for name in field_names.intersection(values)}
    return state._replace(**updates), tuple(sorted(updates))


def stomate_restart_states_from_day_end_packet(
    packet: object,
    *,
    base_entry: StomateRestartEntryState,
    base_season: StomateRestartSeasonState,
    base_daily: StomateDailyAccumulatorState,
) -> tuple[
    StomateRestartEntryState,
    StomateRestartSeasonState,
    StomateDailyAccumulatorState,
    StomateRestartPacketMergeReport,
]:
    """Merge source-backed day-end writes into complete restart state objects.

    Fields absent from the day-end packet retain their prior restart value,
    except reader convenience totals reconstructed from their actual pools.
    Daily accumulators must come from the explicit nested reset packet produced
    at ``stomate.f90::stomate_main`` lines 4944-5000. The top-level and nested
    ``gpp_daily`` values represent the same Fortran variable and must agree.
    """

    if not isinstance(base_entry, StomateRestartEntryState):
        raise TypeError("base_entry must be a StomateRestartEntryState")
    if not isinstance(base_season, StomateRestartSeasonState):
        raise TypeError("base_season must be a StomateRestartSeasonState")
    if not isinstance(base_daily, StomateDailyAccumulatorState):
        raise TypeError("base_daily must be a StomateDailyAccumulatorState")

    fields = _mapping(packet, "packet")
    if "daily_accumulators" not in fields:
        raise ValueError("day-end packet must contain explicit daily_accumulators")
    daily_values = _mapping(fields["daily_accumulators"], "daily_accumulators")
    if "gpp_daily" in fields and "gpp_daily" in daily_values:
        if not np.array_equal(
            np.asarray(fields["gpp_daily"]), np.asarray(daily_values["gpp_daily"])
        ):
            raise ValueError("top-level and daily_accumulators gpp_daily differ")

    entry, entry_updates = _replace_from_mapping(base_entry, fields)
    # Reader conveniences are recomputed from the actual serialized pools,
    # rather than retained from an unrelated base snapshot. Fortran
    # stomate_lpj.f90 1545-1548 and stomate.f90 stomate_init 1586-1589.
    derived = {
        "prod10_total": np.sum(np.asarray(entry.prod10), axis=(1, 2)),
        "prod100_total": np.sum(np.asarray(entry.prod100), axis=(1, 2)),
        "soilc_total": np.asarray(entry.deepC_a) + np.asarray(entry.deepC_s) + np.asarray(entry.deepC_p),
    }
    derived = {name: value for name, value in derived.items() if name not in fields}
    entry = entry._replace(**derived)
    season, season_updates = _replace_from_mapping(base_season, fields)
    daily, daily_updates = _replace_from_mapping(base_daily, daily_values)

    serializable = (
        set(base_entry._fields) | set(base_season._fields) | set(base_daily._fields)
    ) - {"provenance"}
    packet_only = tuple(sorted(set(fields) - serializable - {"daily_accumulators"}))
    report = StomateRestartPacketMergeReport(
        entry_updates=entry_updates,
        season_updates=season_updates,
        daily_updates=daily_updates,
        packet_only_fields=packet_only,
        retained_base_fields=tuple(
            sorted(serializable - set(fields) - set(daily_values) - set(derived))
        ),
        derived_fields=tuple(sorted(derived)),
    )
    return entry, season, daily, report


def stomate_full_restart_states_from_day_end_packet(
    packet: object,
    *,
    base_states: StomateReadstartStates,
) -> tuple[StomateReadstartStates, StomateRestartPacketMergeReport]:
    """Merge matching packet fields across every complete restart state.

    Unrepresented fields retain their individually declared readstart values.
    Fortran ``stomate_io.f90::writerestart`` writes remainder memory at
    2287-2289, 2334-2336, 2369-2371, 2675-2677, 2860-2862 and gas at
    2801-2812; the reader's state grouping does not change field ownership.
    The merge report describes the entry/season/daily conversion contract.
    """

    if not isinstance(base_states, StomateReadstartStates):
        raise TypeError("base_states must be StomateReadstartStates")
    fields = _mapping(packet, "packet")
    entry, season, daily, report = stomate_restart_states_from_day_end_packet(
        fields,
        base_entry=base_states.entry_state,
        base_season=base_states.season_state,
        base_daily=base_states.daily_state,
    )
    gas, _ = _replace_from_mapping(base_states.gas_state, fields)
    remainder, _ = _replace_from_mapping(base_states.remainder_state, fields)
    return StomateReadstartStates(
        entry_state=entry,
        season_state=season,
        daily_state=daily,
        gas_state=gas,
        remainder_state=remainder,
        report=base_states.report,
    ), report


def write_stomate_restart_from_day_end_packet(
    output_path: str | Path,
    packet: object,
    *,
    base_states: StomateReadstartStates,
    physical_state: StomateRestartPhysicalState,
    pft_layout: PFTRunLayout,
    schema_path: str | Path = DEFAULT_STOMATE_RESTART_SCHEMA,
) -> tuple[StomateRestartWriteReport, StomateRestartPacketMergeReport]:
    """Merge a production day-end packet and write a standalone restart.

    Every matching STOMATE field, including gas and remainder, is overwritten
    from the packet. Unrepresented fields retain their declared readstart
    values; absent convenience totals derive from the merged pools.

    Fortran provenance: ``stomate.f90::stomate_finalize`` lines 5445-5496 and
    ``stomate_io.f90::writerestart`` lines 1751-2944.
    """

    if not isinstance(base_states, StomateReadstartStates):
        raise TypeError("base_states must be StomateReadstartStates")
    if not isinstance(pft_layout, PFTRunLayout):
        raise TypeError("pft_layout must be a PFTRunLayout")
    merged, merge_report = stomate_full_restart_states_from_day_end_packet(
        packet, base_states=base_states,
    )
    write_report = write_stomate_full_writerestart_states(
        output_path,
        physical_state=physical_state,
        entry_state=merged.entry_state,
        season_state=merged.season_state,
        daily_state=merged.daily_state,
        gas_state=merged.gas_state,
        remainder_state=merged.remainder_state,
        schema_path=schema_path,
        pft_layout=pft_layout,
    )
    return write_report, merge_report
