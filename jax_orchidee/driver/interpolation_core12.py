"""Explicit-IO owners for ``interpweight_1D`` and ``interpweight_2D``.

The NetCDF and MPI calls in the Fortran procedures are represented by an
``InterpolationSource`` and an injected aggregation callback.  No source
field, mask, overlap, or parallel value is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

from jax_orchidee.driver.interpolation import (
    interpweight_calc_resolution_in,
    interpweight_masking_input1d,
    interpweight_masking_input2d,
    interpweight_masking_input3d,
    interpweight_masking_input4d,
    interpweight_modifying_input1d,
    interpweight_modifying_input2d,
    interpweight_modifying_input3d,
    interpweight_modifying_input4d,
    interpweight_provide_fractions1d,
    interpweight_provide_fractions2d,
)


INTERPOLATION_CORE12_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_1D lines 95-426, control flow 229-420 (41 arms)",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_2D lines 444-896, control flow 580-892 (71 arms)",
)


class FortranUndefinedVariableError(RuntimeError):
    """A source path reads an array that the Fortran procedure did not allocate."""


@dataclass(frozen=True)
class InterpolationSource:
    """Values supplied by the caller's IO layer, in Fortran dimension order."""

    values: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    mask_variable: np.ndarray | None = None
    variable_name: str | None = None


@dataclass(frozen=True)
class InterpolationTarget:
    lalo: np.ndarray
    resolution: np.ndarray
    neighbours: np.ndarray
    contfrac: np.ndarray


@dataclass(frozen=True)
class AggregateRequest:
    source_rank: int
    lalo: np.ndarray
    resolution: np.ndarray
    neighbours: np.ndarray
    contfrac: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    mask: np.ndarray | None
    max_resolution_lon: float
    max_resolution_lat: float
    callsign: str
    nbvmax: int


@dataclass(frozen=True)
class AggregatePacket:
    sub_index: np.ndarray
    sub_area: np.ndarray
    ok: bool


class InterpweightResult(NamedTuple):
    fractions: np.ndarray
    availability: np.ndarray
    mask: np.ndarray
    variable_use_types: np.ndarray
    varmin: float
    varmax: float
    nbvmax_attempts: tuple[int, ...]


Aggregate = Callable[[AggregateRequest], AggregatePacket]


def _target_arrays(target: InterpolationTarget) -> tuple[np.ndarray, ...]:
    lalo = np.asarray(target.lalo, dtype=np.float64)
    resolution = np.asarray(target.resolution, dtype=np.float64)
    neighbours = np.asarray(target.neighbours, dtype=np.int64)
    contfrac = np.asarray(target.contfrac, dtype=np.float64)
    if lalo.ndim != 2 or lalo.shape[1] != 2:
        raise ValueError("lalo must have shape (nbpt, 2)")
    if resolution.shape != lalo.shape:
        raise ValueError("resolution must have shape (nbpt, 2)")
    if neighbours.ndim != 2 or neighbours.shape[0] != lalo.shape[0]:
        raise ValueError("neighbours must have shape (nbpt, NbNeighb)")
    if contfrac.shape != (lalo.shape[0],):
        raise ValueError("contfrac must have shape (nbpt,)")
    return lalo.copy(), resolution.copy(), neighbours.copy(), contfrac.copy()


def _variable_types(variabletypes) -> tuple[np.ndarray, np.ndarray, float, float]:
    requested = np.asarray(variabletypes, dtype=np.float64)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("variabletypes must be a non-empty rank-1 array")
    if requested[0] == -1.0:
        used = np.arange(1, requested.size + 1, dtype=np.float64)
        return requested, used, 1.0, float(requested.size)
    return requested, requested.copy(), np.nan, np.nan


def _mask_from_variable(mask_variable, expected: tuple[int, ...]) -> np.ndarray:
    if mask_variable is None:
        raise FortranUndefinedVariableError("masktype='var' requires an explicit mask_variable read by IO")
    mask_values = np.asarray(mask_variable, dtype=np.float64)
    if mask_values.shape != expected:
        raise ValueError(f"mask_variable must have shape {expected}")
    return (mask_values > 0.0).astype(np.int32)


def _callsign(source: InterpolationSource) -> str:
    if source.variable_name is None or not source.variable_name.strip():
        raise ValueError("variable_name must be supplied explicitly for the aggregate callsign")
    return source.variable_name.strip() + " map"


def _initial_nbvmax(
    resolution: np.ndarray,
    max_resolution_lon: float,
    max_resolution_lat: float,
    *,
    source_resolution: np.ndarray | None,
) -> int:
    if max_resolution_lon != -1.0 and max_resolution_lat != -1.0:
        nix = int(np.max(resolution[:, 0]) / max_resolution_lon) + 2
        njx = int(np.max(resolution[:, 1]) / max_resolution_lat) + 2
    else:
        if source_resolution is None:
            raise FortranUndefinedVariableError(
                "interpweight_1D lines 324-328 pass rank-1 coordinates to a rank-2 resolution routine"
            )
        nix = int(np.max(resolution[:, 0]) / np.max(source_resolution[:, :, 0])) + 2
        njx = int(np.max(resolution[:, 1]) / np.max(source_resolution[:, :, 1])) + 2
    return max(nix * njx, 200)


def _aggregate_until_ok(
    aggregate: Aggregate,
    request: AggregateRequest,
    nbpt: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    width = request.nbvmax
    attempts: list[int] = []
    while True:
        attempts.append(width)
        packet = aggregate(AggregateRequest(**{**request.__dict__, "nbvmax": width}))
        area = np.asarray(packet.sub_area, dtype=np.float64)
        index = np.asarray(packet.sub_index, dtype=np.int64)
        expected_index = (nbpt, width) if request.source_rank == 1 else (nbpt, width, 2)
        if area.shape != (nbpt, width) or index.shape != expected_index:
            raise ValueError(
                f"aggregate packet must contain sub_area {(nbpt, width)} and sub_index {expected_index}"
            )
        if packet.ok:
            return index.copy(), area.copy(), tuple(attempts)
        width *= 2


def _scale_availability(availability, resolution, contfrac) -> np.ndarray:
    result = np.asarray(availability, dtype=np.float64).copy()
    for point in range(result.size):
        if result[point] != -1.0:
            # Deliberately no zero guard: this is the exact source expression.
            result[point] = result[point] / (resolution[point, 0] * resolution[point, 1]) / contfrac[point]
    return result


def interpweight_1d(
    source: InterpolationSource,
    target: InterpolationTarget,
    variabletypes,
    aggregate: Aggregate,
    *,
    varmin: float,
    varmax: float,
    noneg: bool,
    masktype: str,
    maskvalues,
    typefrac: str = "default",
    max_resolution_lon: float = -1.0,
    max_resolution_lat: float = -1.0,
) -> InterpweightResult:
    """Port of ``interpweight_1D`` lines 95-426 with explicit IO/aggregate state."""

    values = np.asarray(source.values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"interpweight_1D input number of dimensions {values.ndim} not ready")
    lon = np.asarray(source.longitude, dtype=np.float64)
    lat = np.asarray(source.latitude, dtype=np.float64)
    if lon.ndim != 1 or lat.ndim != 1 or lon.shape != values.shape or lat.shape != values.shape:
        raise FortranUndefinedVariableError("interpweight_1D requires explicit rank-1 lon/lat arrays of length iml")
    lalo, resolution, neighbours, contfrac = _target_arrays(target)
    if noneg:
        values = interpweight_modifying_input1d(values, 0.0)
    else:
        values = values.copy()
    if masktype == "var":
        mask = _mask_from_variable(source.mask_variable, values.shape)
    else:
        values, mask = interpweight_masking_input1d(
            values, np.zeros(values.shape, dtype=np.int32), masktype, maskvalues
        )
    _, used_types, default_min, default_max = _variable_types(variabletypes)
    if not np.isnan(default_min):
        varmin, varmax = default_min, default_max
    nbvmax = _initial_nbvmax(
        resolution, max_resolution_lon, max_resolution_lat, source_resolution=None
    )
    request = AggregateRequest(
        1, lalo, resolution, neighbours, contfrac, lon.copy(), lat.copy(), None,
        max_resolution_lon, max_resolution_lat, _callsign(source), nbvmax,
    )
    sub_index, sub_area, attempts = _aggregate_until_ok(aggregate, request, lalo.shape[0])
    weighted = interpweight_provide_fractions1d(
        values, sub_area, sub_index, used_types, tint=typefrac, vmin=varmin, vmax=varmax
    )
    return InterpweightResult(
        weighted.fractions,
        _scale_availability(weighted.availability, resolution, contfrac),
        mask,
        used_types,
        float(varmin),
        float(varmax),
        attempts,
    )


def _select_2d_values(values: np.ndarray, initime: int) -> tuple[np.ndarray, int]:
    rank = values.ndim
    if rank not in (2, 3, 4):
        raise ValueError(f"interpweight_2D input number of dimensions {rank} not ready")
    if rank == 4 and initime != -1:
        time = 0 if initime <= 0 else min(initime, values.shape[3]) - 1
        return values[:, :, :, time].copy(), 3
    return values.copy(), rank


def _coordinates_2d(source: InterpolationSource, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(source.longitude, dtype=np.float64)
    lat = np.asarray(source.latitude, dtype=np.float64)
    if lon.ndim == 1:
        if lon.shape != (shape[0],) or lat.ndim != 1 or lat.shape != (shape[1],):
            raise ValueError("axis coordinates must have shapes (iml,) and (jml,)")
        return np.repeat(lon[:, None], shape[1], axis=1), np.repeat(lat[None, :], shape[0], axis=0)
    if lon.ndim == 2:
        if lon.shape != shape or lat.ndim != 2 or lat.shape != shape:
            raise ValueError("full coordinates must both have shape (iml, jml)")
        return lon.copy(), lat.copy()
    raise ValueError("longitude/latitude dimensions must be both rank 1 or both rank 2")


def interpweight_2d(
    source: InterpolationSource,
    target: InterpolationTarget,
    variabletypes,
    aggregate: Aggregate,
    *,
    varmin: float,
    varmax: float,
    noneg: bool,
    masktype: str,
    maskvalues,
    initime: int = -1,
    typefrac: str = "default",
    max_resolution_lon: float = -1.0,
    max_resolution_lat: float = -1.0,
) -> InterpweightResult:
    """Port of ``interpweight_2D`` lines 444-896 with explicit IO/aggregate state."""

    original = np.asarray(source.values, dtype=np.float64)
    values, effective_rank = _select_2d_values(original, initime)
    lon, lat = _coordinates_2d(source, original.shape[:2])
    lalo, resolution, neighbours, contfrac = _target_arrays(target)
    modifiers = {2: interpweight_modifying_input2d, 3: interpweight_modifying_input3d, 4: interpweight_modifying_input4d}
    if noneg:
        values = modifiers[effective_rank](values, 0.0)
    if masktype == "var":
        mask = _mask_from_variable(source.mask_variable, original.shape[:2])
    else:
        maskers = {2: interpweight_masking_input2d, 3: interpweight_masking_input3d, 4: interpweight_masking_input4d}
        values, mask = maskers[effective_rank](
            values, np.zeros(original.shape[:2], dtype=np.int32), masktype, maskvalues
        )
    source_resolution = interpweight_calc_resolution_in(lon, lat)
    nbvmax = _initial_nbvmax(
        resolution, max_resolution_lon, max_resolution_lat, source_resolution=source_resolution
    )
    request = AggregateRequest(
        effective_rank, lalo, resolution, neighbours, contfrac, lon, lat, mask.copy(),
        max_resolution_lon, max_resolution_lat, _callsign(source), nbvmax,
    )
    sub_index, sub_area, attempts = _aggregate_until_ok(aggregate, request, lalo.shape[0])
    _, used_types, default_min, default_max = _variable_types(variabletypes)
    if not np.isnan(default_min):
        varmin, varmax = default_min, default_max
    if effective_rank != 2:
        raise FortranUndefinedVariableError(
            "interpweight_2D lines 663 and 854-856 unconditionally use invar2D, which is unallocated for rank 3/4 input"
        )
    weighted = interpweight_provide_fractions2d(
        values, sub_area, sub_index, used_types, tint=typefrac, vmin=varmin, vmax=varmax
    )
    return InterpweightResult(
        weighted.fractions,
        _scale_availability(weighted.availability, resolution, contfrac),
        mask,
        used_types,
        float(varmin),
        float(varmax),
        attempts,
    )
