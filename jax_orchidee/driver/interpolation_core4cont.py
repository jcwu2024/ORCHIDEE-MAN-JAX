"""Source-faithful routed owners for ``interpweight_4D`` and ``2Dcont``.

NetCDF, MPI, and logging are explicit outer boundaries. Polygon aggregation is
injected as the same request/packet callback used by ``interpolation_core12``;
resolution selection, initial capacity, retry routing, masks, interpolation,
and availability scaling remain owned here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from jax_orchidee.driver.interpolation import (
    interpweight_calc_resolution_in,
    interpweight_masking_input2d,
    interpweight_masking_input3d,
    interpweight_masking_input4d,
    interpweight_modifying_input2d,
    interpweight_modifying_input3d,
    interpweight_modifying_input4d,
    interpweight_provide_fractions4d,
    interpweight_provide_interpolation2d,
)
from jax_orchidee.driver.interpolation_core12 import (
    Aggregate,
    AggregateRequest,
    InterpolationSource,
    InterpolationTarget,
)


INTERPOLATION_CORE4CONT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_4D lines 1378-1806",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_2Dcont lines 1824-2233",
)

# ALLOCATE(mask) at line 1662 does not initialize cells.  A deterministic
# sentinel preserves that source state without pretending those cells are sea.
UNDEFINED_MASK = np.iinfo(np.int32).min


class FortranUndefinedInputError(RuntimeError):
    """The source would pass an unallocated rank-specific array downstream."""


class TimeSelection(NamedTuple):
    values: np.ndarray
    source_rank: int
    effective_rank: int
    selected_time_index: int | None
    time_values: np.ndarray | None
    calendar: str | None


class InterpolationCoreResult(NamedTuple):
    output: np.ndarray
    availability: np.ndarray
    source_values: np.ndarray
    mask: np.ndarray
    overlap_area: np.ndarray
    overlap_index: np.ndarray
    time_selection: TimeSelection
    nbvmax_attempts: tuple[int, ...]


class Interpweight4DResult(NamedTuple):
    output: np.ndarray
    availability: np.ndarray
    source_values: np.ndarray
    mask: np.ndarray
    overlap_area: np.ndarray
    overlap_index: np.ndarray
    time_selection: TimeSelection
    variable_types: np.ndarray
    varmin: np.ndarray
    varmax: np.ndarray
    nbvmax_attempts: tuple[int, ...]


def select_interpweight_input(
    values,
    *,
    initime: int = -1,
    time_values=None,
    calendar: str | None = None,
) -> TimeSelection:
    """Apply lines 1513-1595/1953-2027 to an already decoded array.

    ``initime`` is a Fortran one-based record number.  Calendar labels are
    carried as metadata only: ``flinget`` selects a contiguous positional
    record and never searches coordinate values.  For ``initime <= 0`` the
    first record is selected; values above ``tml`` select the last record.
    """

    array = np.asarray(values).copy()
    if array.ndim not in (2, 3, 4):
        raise ValueError(f"input variable must have rank 2, 3, or 4, got {array.ndim}")
    if isinstance(initime, bool) or not isinstance(initime, (int, np.integer)):
        raise TypeError("initime must be an integer")
    if calendar is not None and not isinstance(calendar, str):
        raise TypeError("calendar must be a string or None")

    coordinates = None
    if time_values is not None:
        coordinates = np.asarray(time_values).copy()
        if coordinates.ndim != 1:
            raise ValueError("time_values must be rank 1")
        if array.ndim != 4:
            raise ValueError("time_values are valid only for a rank-4 input")
        if coordinates.size != array.shape[3]:
            raise ValueError("time_values length must equal the fourth Fortran dimension")

    if array.ndim != 4 or int(initime) == -1:
        return TimeSelection(array, array.ndim, array.ndim, None, coordinates, calendar)

    tml = array.shape[3]
    if tml < 1:
        raise FortranUndefinedInputError(
            "interpweight rank-4 initime selection requires at least one time record"
        )
    requested = int(initime)
    selected = 1 if requested <= 0 else min(requested, tml)
    selected_coordinates = None if coordinates is None else coordinates[selected - 1 : selected].copy()
    return TimeSelection(
        array[:, :, :, selected - 1].copy(),
        4,
        3,
        selected,
        selected_coordinates,
        calendar,
    )


def _lon_lat_arrays(lon, lat, source_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.asarray(lon, dtype=np.float64)
    latitude = np.asarray(lat, dtype=np.float64)
    if longitude.ndim == latitude.ndim == 1:
        if longitude.shape != (source_shape[0],) or latitude.shape != (source_shape[1],):
            raise ValueError("rank-1 lon/lat lengths must match the first two source dimensions")
        longitude = np.broadcast_to(longitude[:, None], source_shape).copy()
        latitude = np.broadcast_to(latitude[None, :], source_shape).copy()
    elif longitude.ndim == latitude.ndim == 2:
        if longitude.shape != source_shape or latitude.shape != source_shape:
            raise ValueError(f"rank-2 lon/lat must both have shape {source_shape}")
        longitude = longitude.copy()
        latitude = latitude.copy()
    else:
        raise ValueError("lon and lat must both be rank 1 or both be rank 2")
    return longitude, latitude


def _target_arrays(resolution, contfrac, npt: int) -> tuple[np.ndarray, np.ndarray]:
    target_resolution = np.asarray(resolution, dtype=np.float64)
    land_fraction = np.asarray(contfrac, dtype=np.float64)
    if target_resolution.shape != (npt, 2):
        raise ValueError(f"resolution must have shape {(npt, 2)}")
    if land_fraction.shape != (npt,):
        raise ValueError(f"contfrac must have shape {(npt,)}")
    return target_resolution, land_fraction


def _routed_target_arrays(target: InterpolationTarget) -> tuple[np.ndarray, ...]:
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


def _callsign(source: InterpolationSource) -> str:
    if source.variable_name is None or not source.variable_name.strip():
        raise ValueError("variable_name must be supplied explicitly for the aggregate callsign")
    return source.variable_name.strip() + " map"


def _initial_4d_nbvmax(
    resolution: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    max_resolution_lon: float,
    max_resolution_lat: float,
) -> int:
    """Exact branch at lines 1701-1714, including the 200-cell floor."""

    if max_resolution_lon != -1.0 and max_resolution_lat != -1.0:
        nix = int(np.max(resolution[:, 0]) / max_resolution_lon) + 2
        njx = int(np.max(resolution[:, 1]) / max_resolution_lat) + 2
    else:
        source_resolution = interpweight_calc_resolution_in(lon, lat)
        nix = int(np.max(resolution[:, 0]) / np.max(source_resolution[:, :, 0])) + 2
        njx = int(np.max(resolution[:, 1]) / np.max(source_resolution[:, :, 1])) + 2
    return max(nix * njx, 200)


def _initial_2dcont_nbvmax(
    resolution: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> int:
    """Exact lines 2134-2150; unlike 4D, this source has no 200 floor."""

    source_resolution = interpweight_calc_resolution_in(lon, lat)
    nix = int(np.max(resolution[:, 0]) / np.max(source_resolution[:, :, 0])) + 2
    njx = int(np.max(resolution[:, 1]) / np.max(source_resolution[:, :, 1])) + 2
    return nix * njx


def _aggregate_until_ok(
    aggregate: Aggregate,
    request: AggregateRequest,
    nbpt: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    width = request.nbvmax
    attempts: list[int] = []
    while True:
        attempts.append(width)
        routed = AggregateRequest(**{**request.__dict__, "nbvmax": width})
        packet = aggregate(routed)
        areas = np.asarray(packet.sub_area, dtype=np.float64)
        indices = np.asarray(packet.sub_index, dtype=np.int64)
        if areas.shape != (nbpt, width) or indices.shape != (nbpt, width, 2):
            raise ValueError(
                "aggregate packet must contain sub_area "
                f"{(nbpt, width)} and sub_index {(nbpt, width, 2)}"
            )
        if packet.ok:
            return indices.copy(), areas.copy(), tuple(attempts)
        width *= 2


def _overlap_arrays(sub_area, sub_index) -> tuple[np.ndarray, np.ndarray]:
    areas = np.asarray(sub_area, dtype=np.float64).copy()
    indices = np.asarray(sub_index, dtype=np.int64).copy()
    if areas.ndim != 2:
        raise ValueError("sub_area must be rank 2")
    if indices.shape != areas.shape + (2,):
        raise ValueError(f"sub_index must have shape {areas.shape + (2,)}")
    return areas, indices


def _shared_mask(values: np.ndarray, masktype: str, maskvalues, maskvar, *, initialized: bool):
    initial = np.zeros(values.shape[:2], dtype=np.int32) if initialized else np.full(
        values.shape[:2], UNDEFINED_MASK, dtype=np.int32
    )
    if masktype.strip() == "var":
        if maskvar is None:
            raise ValueError("maskvar is required when masktype='var'")
        external = np.asarray(maskvar)
        if external.shape != values.shape[:2]:
            raise ValueError(f"maskvar must have shape {values.shape[:2]}")
        initial[external > 0.0] = 1
        return values.copy(), initial
    thresholds = np.asarray(maskvalues, dtype=values.dtype)
    if thresholds.shape != (3,):
        raise ValueError("maskvalues must have shape (3,)")
    if values.ndim == 2:
        return interpweight_masking_input2d(values, initial, masktype.strip(), thresholds)
    if values.ndim == 3:
        return interpweight_masking_input3d(values, initial, masktype.strip(), thresholds)
    return interpweight_masking_input4d(values, initial, masktype.strip(), thresholds)


def _masked_overlaps(
    areas: np.ndarray,
    indices: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain positive, mask==un candidates in source order and compact them."""

    filtered_area = np.zeros_like(areas)
    filtered_index = np.zeros_like(indices)
    for point in range(areas.shape[0]):
        write = 0
        for overlap in range(areas.shape[1]):
            area = areas[point, overlap]
            if area <= 0.0:
                break
            i = int(indices[point, overlap, 0]) - 1
            j = int(indices[point, overlap, 1]) - 1
            if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
                raise IndexError("sub_index contains an invalid Fortran one-based source index")
            if mask[i, j] == 1:
                filtered_area[point, write] = area
                filtered_index[point, write, :] = indices[point, overlap, :]
                write += 1
    return filtered_area, filtered_index


def _scale_availability(availability, resolution: np.ndarray, contfrac: np.ndarray) -> np.ndarray:
    result = np.asarray(availability, dtype=np.float64).copy()
    valid = result != -1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result[valid] = (
            result[valid]
            / (resolution[valid, 0] * resolution[valid, 1])
            / contfrac[valid]
        )
    return result


def _modified(values: np.ndarray, noneg: bool) -> np.ndarray:
    if not noneg:
        return values.copy()
    if values.ndim == 2:
        return interpweight_modifying_input2d(values, 0.0)
    if values.ndim == 3:
        return interpweight_modifying_input3d(values, 0.0)
    return interpweight_modifying_input4d(values, 0.0)


def interpweight_4d(
    input_values,
    *,
    variabletypes,
    varmin,
    varmax,
    lon,
    lat,
    sub_area,
    sub_index,
    resolution,
    contfrac,
    initime: int = -1,
    time_values=None,
    calendar: str | None = None,
    noneg: bool = False,
    masktype: str = "nomask",
    maskvalues=(0.0, 0.0, 0.0),
    maskvar=None,
    typefrac: str = "default",
) -> Interpweight4DResult:
    """Post-aggregate numerical core for lines 1647-1690 and 1747-1789."""

    selection = select_interpweight_input(
        input_values, initime=initime, time_values=time_values, calendar=calendar
    )
    if selection.effective_rank != 4:
        raise FortranUndefinedInputError(
            "interpweight_4D lines 1764-1766 pass unallocated invar4D unless the effective input rank is 4"
        )
    values = _modified(selection.values, bool(noneg))
    _lon_lat_arrays(lon, lat, values.shape[:2])
    areas, indices = _overlap_arrays(sub_area, sub_index)
    target_resolution, land_fraction = _target_arrays(resolution, contfrac, areas.shape[0])
    values, mask = _shared_mask(
        values, masktype, maskvalues, maskvar, initialized=False
    )
    areas, indices = _masked_overlaps(areas, indices, mask)

    types = np.asarray(variabletypes, dtype=values.dtype).copy()
    if types.ndim != 1 or types.size == 0:
        raise ValueError("variabletypes must be a non-empty rank-1 array")
    minimum = np.asarray(varmin, dtype=values.dtype).copy()
    maximum = np.asarray(varmax, dtype=values.dtype).copy()
    if minimum.shape != types.shape or maximum.shape != types.shape:
        raise ValueError("varmin and varmax must have the same shape as variabletypes")
    if types[0] == -1.0:
        types = np.arange(1, types.size + 1, dtype=values.dtype)
        minimum[...] = 1.0
        maximum[...] = float(types.size)

    fractions = interpweight_provide_fractions4d(
        values,
        areas,
        indices,
        types,
        tint=typefrac.strip(),
        zeroval=0.0,
        vmin=minimum,
        vmax=maximum,
        two=2.0,
    )
    availability = _scale_availability(
        fractions.availability, target_resolution, land_fraction
    )
    return Interpweight4DResult(
        fractions.fractions,
        availability,
        values,
        mask,
        areas,
        indices,
        selection,
        types,
        minimum,
        maximum,
        (),
    )


def interpweight_2dcont(
    input_values,
    *,
    lon,
    lat,
    sub_area,
    sub_index,
    resolution,
    contfrac,
    initime: int = -1,
    time_values=None,
    calendar: str | None = None,
    noneg: bool = False,
    masktype: str = "nomask",
    maskvalues=(0.0, 0.0, 0.0),
    maskvar=None,
    typefrac: str = "default",
    defaultvalue: float = 0.0,
    default_no_value: float = 1.0,
) -> InterpolationCoreResult:
    """Post-aggregate numerical core for lines 2082-2128 and 2186-2214."""

    selection = select_interpweight_input(
        input_values, initime=initime, time_values=time_values, calendar=calendar
    )
    if selection.effective_rank != 2:
        raise FortranUndefinedInputError(
            "interpweight_2Dcont lines 2189-2191 pass unallocated invar2D unless the effective input rank is 2"
        )
    values = _modified(selection.values, bool(noneg))
    _lon_lat_arrays(lon, lat, values.shape)
    areas, indices = _overlap_arrays(sub_area, sub_index)
    target_resolution, land_fraction = _target_arrays(resolution, contfrac, areas.shape[0])
    values, mask = _shared_mask(values, masktype, maskvalues, maskvar, initialized=True)
    areas, indices = _masked_overlaps(areas, indices, mask)
    interpolation = interpweight_provide_interpolation2d(
        values,
        areas,
        indices,
        tint=typefrac.strip(),
        zeroval=0.0,
        defaultval=defaultvalue,
        default_no_value=default_no_value,
    )
    availability = _scale_availability(
        interpolation.availability, target_resolution, land_fraction
    )
    return InterpolationCoreResult(
        interpolation.fractions,
        availability,
        values,
        mask,
        areas,
        indices,
        selection,
        (),
    )


def interpweight_4d_routed(
    source: InterpolationSource,
    target: InterpolationTarget,
    variabletypes,
    aggregate: Aggregate,
    *,
    varmin,
    varmax,
    initime: int = -1,
    time_values=None,
    calendar: str | None = None,
    noneg: bool = False,
    masktype: str = "nomask",
    maskvalues=(0.0, 0.0, 0.0),
    typefrac: str = "default",
    max_resolution_lon: float = -1.0,
    max_resolution_lat: float = -1.0,
) -> Interpweight4DResult:
    """Routed owner of ``interpweight_4D`` lines 1378-1806.

    The callback is the source ``aggregate_p`` boundary, not an overlap input:
    this owner computes and retries ``nbvmax`` before entering the numerical
    fraction core.
    """

    selection = select_interpweight_input(
        source.values, initime=initime, time_values=time_values, calendar=calendar
    )
    values = _modified(selection.values, bool(noneg))
    lon, lat = _lon_lat_arrays(source.longitude, source.latitude, values.shape[:2])
    lalo, resolution, neighbours, contfrac = _routed_target_arrays(target)
    values, mask = _shared_mask(
        values,
        masktype,
        maskvalues,
        source.mask_variable,
        initialized=False,
    )
    initial_width = _initial_4d_nbvmax(
        resolution,
        lon,
        lat,
        float(max_resolution_lon),
        float(max_resolution_lat),
    )
    request = AggregateRequest(
        selection.effective_rank,
        lalo,
        resolution,
        neighbours,
        contfrac,
        lon,
        lat,
        mask.copy(),
        float(max_resolution_lon),
        float(max_resolution_lat),
        _callsign(source),
        initial_width,
    )
    sub_index, sub_area, attempts = _aggregate_until_ok(
        aggregate, request, lalo.shape[0]
    )
    result = interpweight_4d(
        values,
        variabletypes=variabletypes,
        varmin=varmin,
        varmax=varmax,
        lon=lon,
        lat=lat,
        sub_area=sub_area,
        sub_index=sub_index,
        resolution=resolution,
        contfrac=contfrac,
        initime=-1,
        time_values=(selection.time_values if selection.effective_rank == 4 else None),
        calendar=selection.calendar,
        noneg=False,
        masktype="var",
        maskvar=(mask == 1).astype(np.float64),
        typefrac=typefrac,
    )
    return result._replace(nbvmax_attempts=attempts)


def interpweight_2dcont_routed(
    source: InterpolationSource,
    target: InterpolationTarget,
    aggregate: Aggregate,
    *,
    initime: int = -1,
    time_values=None,
    calendar: str | None = None,
    noneg: bool = False,
    masktype: str = "nomask",
    maskvalues=(0.0, 0.0, 0.0),
    typefrac: str = "default",
    defaultvalue: float = 0.0,
    default_no_value: float = 1.0,
) -> InterpolationCoreResult:
    """Routed owner of ``interpweight_2Dcont`` lines 1824-2233."""

    selection = select_interpweight_input(
        source.values, initime=initime, time_values=time_values, calendar=calendar
    )
    values = _modified(selection.values, bool(noneg))
    lon, lat = _lon_lat_arrays(source.longitude, source.latitude, values.shape[:2])
    lalo, resolution, neighbours, contfrac = _routed_target_arrays(target)
    values, mask = _shared_mask(
        values,
        masktype,
        maskvalues,
        source.mask_variable,
        initialized=True,
    )
    initial_width = _initial_2dcont_nbvmax(resolution, lon, lat)
    request = AggregateRequest(
        selection.effective_rank,
        lalo,
        resolution,
        neighbours,
        contfrac,
        lon,
        lat,
        mask.copy(),
        -1.0,
        -1.0,
        _callsign(source),
        initial_width,
    )
    sub_index, sub_area, attempts = _aggregate_until_ok(
        aggregate, request, lalo.shape[0]
    )
    result = interpweight_2dcont(
        values,
        lon=lon,
        lat=lat,
        sub_area=sub_area,
        sub_index=sub_index,
        resolution=resolution,
        contfrac=contfrac,
        initime=-1,
        calendar=selection.calendar,
        noneg=False,
        masktype="var",
        maskvar=(mask == 1).astype(np.float64),
        typefrac=typefrac,
        defaultvalue=defaultvalue,
        default_no_value=default_no_value,
    )
    return result._replace(nbvmax_attempts=attempts)
