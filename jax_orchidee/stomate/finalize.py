"""Source-ordered STOMATE finalization for the fixed paper PFT14 path.

NetCDF and other file operations remain outside this module.  The restart
state is handed to an explicit callback as the existing normalized reader /
writer contracts, while the source-inactive crop and forcing branches are
recorded rather than silently approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, TypeVar

import numpy as np

from jax_orchidee.stomate.reference import (
    StomateDailyAccumulatorState,
    StomateRestartEntryState,
    StomateRestartSeasonState,
)


STOMATE_FINALIZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_finalize lines 5440-5496: unconditional writerestart call",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_finalize lines 5499-5523: ANY(ok_LAIdev) crop-restart gate",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_finalize lines 5525-5533: ok_co2/allow_forcing_write/name general-forcing gate",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_finalize lines 5535-6132: carbon-forcing normalization, gather, serialization, and cleanup gate",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_finalize lines 6133-6205: permafrost-carbon normalization and serialization gate",
    "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::writerestart lines 1751-2944: restart serialization routine",
)


@dataclass(frozen=True)
class StomateFinalizeDimensions:
    """Dynamic dimensions for the paper path; PFT numbers are Fortran-based."""

    npts: int
    nvm: int
    active_pft_fortran: int = 14

    def validate(self) -> None:
        if self.npts < 1:
            raise ValueError("npts must be positive")
        if self.nvm < self.active_pft_fortran:
            raise ValueError(
                f"nvm={self.nvm} cannot contain active Fortran PFT {self.active_pft_fortran}"
            )


@dataclass(frozen=True)
class StomateFinalizePFT14Switches:
    """Structural switches fixed by the materialized 250919 paper run.

    ``allow_forcing_write`` is the source default from
    ``constantes_var.f90`` line 447.  It cannot activate output because the
    materialized forcing name is ``NONE``.  The carbon forcing switches are
    retained explicitly even though their enclosing file-name gates are off.
    """

    ok_laidev: bool | tuple[bool, ...] = False
    ok_co2: bool = True
    allow_forcing_write: bool = True
    ok_leak: bool = True
    ok_peat: bool = False
    stomate_forcing_name: str = "NONE"
    stomate_cforcing_name: str = "NONE"
    cforcing_permafrost_name: str = "NONE"


class StomateRestartWriteRequest(NamedTuple):
    """Pure-data boundary matching the existing combined restart writer."""

    entry_state: StomateRestartEntryState
    season_state: StomateRestartSeasonState
    daily_state: StomateDailyAccumulatorState
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_finalize lines 5440-5496",
        "jax_orchidee/stomate/restart_io.py::write_stomate_restart_states_from_template",
    )

    def as_writer_kwargs(self) -> dict[str, object]:
        """Return keyword arguments accepted by the template-backed writer."""

        return {
            "entry_state": self.entry_state,
            "season_state": self.season_state,
            "daily_state": self.daily_state,
        }


class StomateFinalizeIOBoundary(NamedTuple):
    """One file-I/O decision in Fortran source order."""

    name: str
    active: bool
    classification: str
    source_file: str
    source_subroutine: str
    line_span: tuple[int, int]
    gate: str
    disposition: str

    @property
    def provenance(self) -> str:
        """Return the exact source owner for audits and diagnostics."""

        start, end = self.line_span
        return f"{self.source_file}::{self.source_subroutine} lines {start}-{end}"


_WriteResult = TypeVar("_WriteResult")
RestartWriter = Callable[[StomateRestartWriteRequest], _WriteResult]


class StomateFinalizePFT14Result(NamedTuple):
    restart_request: StomateRestartWriteRequest
    restart_write_result: object
    io_boundaries: tuple[StomateFinalizeIOBoundary, ...]
    process_order: tuple[str, ...]
    provenance: tuple[str, ...] = STOMATE_FINALIZE_PROVENANCE


def _validate_switches(switches: StomateFinalizePFT14Switches, nvm: int) -> None:
    expected = StomateFinalizePFT14Switches()
    for name in (
        "ok_co2",
        "allow_forcing_write",
        "ok_leak",
        "ok_peat",
        "stomate_forcing_name",
        "stomate_cforcing_name",
        "cforcing_permafrost_name",
    ):
        actual = getattr(switches, name)
        wanted = getattr(expected, name)
        if actual != wanted:
            raise NotImplementedError(
                f"unsupported stomate_finalize paper branch: {name}={actual!r}; required {wanted!r}"
            )

    ok_laidev = switches.ok_laidev
    if isinstance(ok_laidev, bool):
        active = ok_laidev
    else:
        if len(ok_laidev) != nvm:
            raise ValueError(f"ok_laidev must contain nvm={nvm} entries")
        active = any(bool(value) for value in ok_laidev)
    if active:
        raise NotImplementedError(
            "unsupported stomate_finalize paper branch: ANY(ok_LAIdev) must be false (lines 5499-5523)"
        )


def _require_shape(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def stomate_finalize_pft14_explicit(
    *,
    dimensions: StomateFinalizeDimensions,
    entry_state: StomateRestartEntryState,
    season_state: StomateRestartSeasonState,
    daily_state: StomateDailyAccumulatorState,
    restart_writer: RestartWriter[_WriteResult],
    switches: StomateFinalizePFT14Switches = StomateFinalizePFT14Switches(),
) -> StomateFinalizePFT14Result:
    """Finalize the fixed paper path in Fortran source order.

    The callback is the sole side-effect boundary.  A caller can adapt
    :func:`write_stomate_restart_states_from_template` by passing a callback
    which supplies its template/output paths and expands
    ``request.as_writer_kwargs()``.  No path, restart value, or callback result
    is synthesized here.
    """

    dimensions.validate()
    _validate_switches(switches, dimensions.nvm)
    if not isinstance(entry_state, StomateRestartEntryState):
        raise TypeError("entry_state must be a StomateRestartEntryState")
    if not isinstance(season_state, StomateRestartSeasonState):
        raise TypeError("season_state must be a StomateRestartSeasonState")
    if not isinstance(daily_state, StomateDailyAccumulatorState):
        raise TypeError("daily_state must be a StomateDailyAccumulatorState")
    if not callable(restart_writer):
        raise TypeError("restart_writer must be callable")

    shape = (dimensions.npts, dimensions.nvm)
    _require_shape("entry_state.pft_present", entry_state.pft_present, shape)
    entry_gpp = _require_shape("entry_state.gpp_daily", entry_state.gpp_daily, shape)
    daily_gpp = _require_shape("daily_state.gpp_daily", daily_state.gpp_daily, shape)
    _require_shape("season_state.moiavail_month", season_state.moiavail_month, shape)
    if not np.array_equal(entry_gpp, daily_gpp):
        raise ValueError("shared restart field gpp_daily differs between entry_state and daily_state")
    request = StomateRestartWriteRequest(entry_state, season_state, daily_state)
    write_result = restart_writer(request)

    source_file = "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
    source_subroutine = "stomate_finalize"
    boundaries = (
        StomateFinalizeIOBoundary(
            "writerestart",
            True,
            "restart_serialization",
            source_file,
            source_subroutine,
            (5440, 5496),
            "unconditional",
            "current scientific state is handed unchanged to the restart-writer boundary",
        ),
        StomateFinalizeIOBoundary(
            "sticslai_io_writestart",
            False,
            "crop_restart_io",
            source_file,
            source_subroutine,
            (5499, 5523),
            "ANY(ok_LAIdev)",
            "inactive because every paper-run ok_LAIdev entry is false",
        ),
        StomateFinalizeIOBoundary(
            "forcing_write",
            False,
            "forcing_io",
            source_file,
            source_subroutine,
            (5525, 5533),
            "ok_co2 .AND. allow_forcing_write .AND. TRIM(stomate_forcing_name) /= 'NONE'",
            "inactive because the paper-run stomate_forcing_name is NONE",
        ),
        StomateFinalizeIOBoundary(
            "carbon_forcing_write",
            False,
            "forcing_io",
            source_file,
            source_subroutine,
            (5535, 6132),
            "TRIM(stomate_Cforcing_name) /= 'NONE'",
            "inactive because the paper-run stomate_Cforcing_name is NONE",
        ),
        StomateFinalizeIOBoundary(
            "permafrost_carbon_forcing_write",
            False,
            "forcing_io",
            source_file,
            source_subroutine,
            (6133, 6205),
            "TRIM(Cforcing_permafrost_name) /= 'NONE'",
            "inactive because the paper-run Cforcing_permafrost_name is NONE",
        ),
    )
    return StomateFinalizePFT14Result(
        request,
        write_result,
        boundaries,
        tuple(boundary.name for boundary in boundaries),
    )


stomate_finalize = stomate_finalize_pft14_explicit
StomateFinalizeResult = StomateFinalizePFT14Result
