"""Source-backed serial geometry state for ``src_global/grid.f90``.

This module owns the nine grid procedures in the batch-2 geometry ledger.
Parallel transport and projection formulas are infrastructure dependencies:
serial array contracts are implemented here, while MPI and ``module_llxy``
work must be supplied explicitly by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal, Protocol

import numpy as np

from jax_orchidee.driver.geometry_haversine import (
    haversine_laloarea,
    haversine_laloseglen,
    haversine_polyheadings,
    haversine_polyseglen,
    haversine_polysort,
    haversine_reglatlontoploy,
    haversine_regxytoploy,
    haversine_singlepointploy,
    haversine_xyarea,
)
from jax_orchidee.driver.geometry_llxy import ProjectionInfo


VAL_EXP = 999_999.0
UNSET_NEIGHBOUR = -999_999
UNSET_LAND_INDEX = -10_000_000


class AllocationContractError(RuntimeError):
    """Raised when a Fortran ALLOCATED guard cannot preserve array shape."""


class MPIInfrastructureBoundary(NotImplementedError):
    """Raised when grid scatter/gather requires a parallel runtime."""


class ProjectionInfrastructureBoundary(NotImplementedError):
    """Raised when ``module_llxy::latlon_to_ij`` has not been supplied."""


class LatLonToIJ(Protocol):
    """Projection adapter returning Fortran one-based fractional ``(ri, rj)``."""

    def latlon_to_ij(self, lat: float, lon: float) -> tuple[float, float]: ...


@dataclass(frozen=True)
class GlobalGridState:
    """Global dimensions and root arrays from ``grid_set/allocate_glo``."""

    iim: int
    jjm: int
    nbp_glo: int | None = None
    nb_segments: int = 0
    nb_neighbours: int = 0
    neighbours: np.ndarray | None = None
    headings: np.ndarray | None = None
    seglength: np.ndarray | None = None
    corners: np.ndarray | None = None
    area: np.ndarray | None = None
    resolution: np.ndarray | None = None
    lalo: np.ndarray | None = None
    contfrac: np.ndarray | None = None
    index: np.ndarray | None = None
    lon: np.ndarray | None = None
    lat: np.ndarray | None = None
    zlev: np.ndarray | None = None

    @property
    def allocated(self) -> bool:
        return self.neighbours is not None


@dataclass(frozen=True)
class PolygonList:
    """Land-vector polygon output of ``grid_topolylist``."""

    global_grid: bool
    lalo: np.ndarray
    corners: np.ndarray
    neighbours: np.ndarray
    headings: np.ndarray
    seglength: np.ndarray
    area: np.ndarray
    ilandindex: np.ndarray
    jlandindex: np.ndarray


@dataclass(frozen=True)
class GridState:
    """Local grid module state, including its allocated global owner."""

    global_state: GlobalGridState
    npts: int
    nb_segments: int
    nb_neighbours: int
    grid_type: str
    grid_name: str
    global_grid: bool
    lalo: np.ndarray
    neighbours: np.ndarray
    headings: np.ndarray
    seglength: np.ndarray
    corners: np.ndarray
    area: np.ndarray
    resolution: np.ndarray
    contfrac: np.ndarray
    ilandindex: np.ndarray
    jlandindex: np.ndarray


def _positive_size(value: int, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def grid_set_glo(
    arg_nbp_lon: int,
    arg_nbp_lat: int,
    arg_nbp_glo: int | None = None,
    *,
    state: GlobalGridState | None = None,
) -> GlobalGridState:
    """Set dimensions held on every process.

    Fortran provenance: ``grid.f90::grid_set_glo``, lines 344-353. Changing
    dimensions after ``grid_allocate_glo`` is rejected because retained arrays
    would violate the Fortran module's allocation contract.
    """

    iim = _positive_size(arg_nbp_lon, "arg_nbp_lon")
    jjm = _positive_size(arg_nbp_lat, "arg_nbp_lat")
    nbp_glo = (
        None if arg_nbp_glo is None else _positive_size(arg_nbp_glo, "arg_nbp_glo")
    )
    if state is None:
        return GlobalGridState(iim=iim, jjm=jjm, nbp_glo=nbp_glo)
    resolved_nbp = state.nbp_glo if nbp_glo is None else nbp_glo
    if state.allocated and (state.iim, state.jjm, state.nbp_glo) != (
        iim,
        jjm,
        resolved_nbp,
    ):
        raise AllocationContractError(
            "cannot change global dimensions after grid_allocate_glo"
        )
    return replace(state, iim=iim, jjm=jjm, nbp_glo=resolved_nbp)


def grid_allocate_glo(state: GlobalGridState, nbseg: int) -> GlobalGridState:
    """Allocate root grid arrays with exact Fortran dimensions.

    Fortran provenance: ``grid.f90::grid_allocate_glo``, lines 367-400.
    ``VAL_EXP`` is used as an explicit unpopulated marker; no scientific value
    is inferred during allocation.
    """

    if state.allocated:
        raise AllocationContractError("grid_allocate_glo arrays are already allocated")
    if state.nbp_glo is None:
        raise AllocationContractError("nbp_glo must be set before grid_allocate_glo")
    segments = (
        state.nb_segments if state.nb_segments >= 3 else _positive_size(nbseg, "nbseg")
    )
    if segments < 3:
        raise ValueError("nbseg must describe a polygon with at least three segments")
    neighbours = 2 * segments
    npts = state.nbp_glo
    return replace(
        state,
        nb_segments=segments,
        nb_neighbours=neighbours,
        neighbours=np.full((npts, neighbours), UNSET_NEIGHBOUR, dtype=np.int32),
        headings=np.full((npts, neighbours), VAL_EXP, dtype=np.float64),
        seglength=np.full((npts, segments), VAL_EXP, dtype=np.float64),
        corners=np.full((npts, segments, 2), VAL_EXP, dtype=np.float64),
        area=np.full(npts, VAL_EXP, dtype=np.float64),
        resolution=np.full((npts, 2), VAL_EXP, dtype=np.float64),
        lalo=np.full((npts, 2), VAL_EXP, dtype=np.float64),
        contfrac=np.full(npts, VAL_EXP, dtype=np.float64),
        index=np.full(npts, UNSET_LAND_INDEX, dtype=np.int32),
        lon=np.full((state.iim, state.jjm), VAL_EXP, dtype=np.float64),
        lat=np.full((state.iim, state.jjm), VAL_EXP, dtype=np.float64),
        zlev=np.full((state.iim, state.jjm), VAL_EXP, dtype=np.float64),
    )


def _grid_type(gtype: str, nbseg: int, isglobal: bool | None) -> tuple[str, bool]:
    lowered = str(gtype).lower()
    if "reglonlat" in lowered:
        if nbseg != 4:
            raise ValueError("RegLonLat grids require exactly four polygon segments")
        return "RegLonLat", True if isglobal is None else bool(isglobal)
    if "regxy" in lowered:
        if nbseg != 4:
            raise ValueError("RegXY grids require exactly four polygon segments")
        return "RegXY", False if isglobal is None else bool(isglobal)
    if "unstruct" in lowered:
        return "UnStruct", True if isglobal is None else bool(isglobal)
    raise ValueError("gtype must contain RegLonLat, RegXY, or UnStruct")


def grid_init(
    npts: int,
    nbseg: int,
    gtype: str,
    gname: str,
    *,
    global_state: GlobalGridState,
    isglobal: bool | None = None,
    existing: GridState | None = None,
) -> GridState:
    """Initialize local grid metadata and guarded arrays.

    Fortran provenance: ``grid.f90::grid_init``, lines 207-328. When
    ``existing`` is passed, this models every ``.NOT.ALLOCATED`` false arm:
    only an exactly compatible allocation may be retained.
    """

    npts = _positive_size(npts, "npts")
    nbseg = _positive_size(nbseg, "nbseg")
    canonical_type, global_grid = _grid_type(gtype, nbseg, isglobal)
    if global_state.nbp_glo is None:
        raise AllocationContractError("grid_set_glo must set nbp_glo before grid_init")
    if existing is not None:
        contract = (npts, nbseg, canonical_type, str(gname), global_state.nbp_glo)
        current = (
            existing.npts,
            existing.nb_segments,
            existing.grid_type,
            existing.grid_name,
            existing.ilandindex.size,
        )
        if current != contract:
            raise AllocationContractError(
                "existing grid_init arrays do not match the requested contract"
            )
        return replace(existing, global_grid=global_grid, global_state=global_state)
    nneigh = 2 * nbseg
    return GridState(
        global_state=global_state,
        npts=npts,
        nb_segments=nbseg,
        nb_neighbours=nneigh,
        grid_type=canonical_type,
        grid_name=str(gname),
        global_grid=global_grid,
        lalo=np.full((npts, 2), VAL_EXP, dtype=np.float64),
        neighbours=np.full((npts, nneigh), UNSET_NEIGHBOUR, dtype=np.int32),
        headings=np.full((npts, nneigh), VAL_EXP, dtype=np.float64),
        seglength=np.full((npts, nbseg), VAL_EXP, dtype=np.float64),
        corners=np.full((npts, nbseg, 2), VAL_EXP, dtype=np.float64),
        area=np.full(npts, VAL_EXP, dtype=np.float64),
        resolution=np.full((npts, 2), VAL_EXP, dtype=np.float64),
        contfrac=np.full(npts, VAL_EXP, dtype=np.float64),
        ilandindex=np.full(global_state.nbp_glo, UNSET_LAND_INDEX, dtype=np.int32),
        jlandindex=np.full(global_state.nbp_glo, UNSET_LAND_INDEX, dtype=np.int32),
    )


def grid_topolylist(
    gtype: str,
    nbseg: int,
    nland: int,
    iim: int,
    jjm: int,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    kindex: np.ndarray,
    *,
    projection: ProjectionInfo | None = None,
    ij_to_latlon: Callable[[float, float], tuple[float, float]] | None = None,
    dxwrf: np.ndarray | None = None,
    dywrf: np.ndarray | None = None,
) -> PolygonList:
    """Transform a regular grid into land-vector polygons.

    Fortran provenance: ``grid.f90::grid_topolylist``, lines 490-631. The
    RegLonLat and RegXY branches call the independently owned haversine
    kernels. RegXY requires explicit projection, one-based ``ij_to_latlon``,
    and projected ``dxwrf/dywrf`` inputs. The source rejects UnStruct before
    its generic segment/area fallback can be reached.
    """

    grid_type = str(gtype)
    if grid_type not in {"RegLonLat", "RegXY"} or int(nbseg) != 4:
        raise ValueError(
            "grid_topolylist source path requires RegLonLat or RegXY with nbseg=4"
        )
    lon = np.asarray(grid_lon, dtype=np.float64)
    lat = np.asarray(grid_lat, dtype=np.float64)
    index = np.asarray(kindex, dtype=np.int64)
    if lon.shape != (int(iim), int(jjm)) or lat.shape != lon.shape:
        raise ValueError("grid_lon/grid_lat must have Fortran shape (iim,jjm)")
    if index.shape != (int(nland),):
        raise ValueError("kindex must have shape (nland,)")
    if grid_type == "RegLonLat" and lon.shape[0] >= 2:
        lon_axis = lon[:, 0]
        delta = float(np.max(np.abs(np.diff(lon_axis))))
        minimum = float(np.min(lon_axis))
        maximum = float(np.max(lon_axis))
        global_grid = (
            minimum > 0.0
            and maximum > 180.0
            and minimum - delta / 2.0 <= 0.0
            and maximum + delta / 2.0 >= 360.0
        ) or (
            minimum < 0.0
            and maximum > 0.0
            and minimum - delta / 2.0 <= -180.0
            and maximum + delta / 2.0 >= 180.0
        )
    else:
        global_grid = False

    if int(nland) == 1:
        polygon = haversine_singlepointploy(lon, lat, nbseg=int(nbseg))
    elif grid_type == "RegLonLat":
        polygon = haversine_reglatlontoploy(
            lon, lat, index, global_grid=global_grid, nbseg=int(nbseg)
        )
    else:
        if projection is None or ij_to_latlon is None:
            raise ProjectionInfrastructureBoundary(
                "RegXY grid_topolylist requires projection and one-based ij_to_latlon"
            )
        polygon = haversine_regxytoploy(
            lon,
            lat,
            index,
            projection_code=int(projection.code),
            ij_to_latlon=ij_to_latlon,
            nbseg=int(nbseg),
        )

    corners = np.stack((polygon.lonpoly[:, 0::2], polygon.latpoly[:, 0::2]), axis=-1)
    headings = haversine_polyheadings(polygon.lonpoly, polygon.latpoly, polygon.center)
    sorted_lon, sorted_lat, headings, neighbours = haversine_polysort(
        polygon.lonpoly, polygon.latpoly, headings, polygon.neighb_loc
    )
    if grid_type == "RegLonLat":
        seglength = haversine_laloseglen(sorted_lon, sorted_lat, nbseg=int(nbseg))
        area = haversine_laloarea(seglength, nbseg=int(nbseg))
    else:
        if dxwrf is None or dywrf is None:
            raise ProjectionInfrastructureBoundary(
                "RegXY grid_topolylist requires dxwrf and dywrf"
            )
        seglength = haversine_polyseglen(sorted_lon, sorted_lat, nbseg=int(nbseg))
        area = haversine_xyarea(polygon.iorig, polygon.jorig, dxwrf, dywrf)
    lalo = np.column_stack((polygon.center[:, 1], polygon.center[:, 0]))
    return PolygonList(
        global_grid=global_grid,
        lalo=lalo,
        corners=corners,
        neighbours=neighbours,
        headings=headings,
        seglength=seglength,
        area=area,
        ilandindex=np.asarray(polygon.iorig, dtype=np.int32),
        jlandindex=np.asarray(polygon.jorig, dtype=np.int32),
    )


def grid_scatter(state: GridState, *, mpi_size: int = 1) -> GridState:
    """Expose populated global arrays as local arrays for one process.

    Fortran provenance: ``grid.f90::grid_scatter``, lines 663-718. A serial
    run has an identity scatter. Multi-rank partitioning/broadcast is an MPI
    infrastructure boundary and is never represented as a scientific kernel.
    """

    if int(mpi_size) != 1:
        raise MPIInfrastructureBoundary(
            "grid_scatter requires an MPI partition/scatter backend for mpi_size != 1"
        )
    global_state = state.global_state
    if not global_state.allocated or global_state.nbp_glo != state.npts:
        raise AllocationContractError(
            "serial grid_scatter requires allocated global arrays matching npts"
        )
    required = (
        global_state.lalo,
        global_state.neighbours,
        global_state.headings,
        global_state.seglength,
        global_state.corners,
        global_state.area,
        global_state.contfrac,
        global_state.resolution,
        global_state.index,
    )
    if any(value is None for value in required):
        raise AllocationContractError("global grid arrays are incomplete")
    index = np.asarray(global_state.index, dtype=np.int64)
    iland = state.ilandindex.copy()
    jland = state.jlandindex.copy()
    if np.max(iland) < 0 and np.max(jland) < 0:
        iland = ((index - 1) % global_state.iim + 1).astype(np.int32)
        jland = ((index - 1) // global_state.iim + 1).astype(np.int32)
    return replace(
        state,
        lalo=np.asarray(global_state.lalo).copy(),
        neighbours=np.asarray(global_state.neighbours).copy(),
        headings=np.asarray(global_state.headings).copy(),
        seglength=np.asarray(global_state.seglength).copy(),
        corners=np.asarray(global_state.corners).copy(),
        area=np.asarray(global_state.area).copy(),
        resolution=np.asarray(global_state.resolution).copy(),
        contfrac=np.asarray(global_state.contfrac).copy(),
        ilandindex=iland,
        jlandindex=jland,
    )


def grid_stuff(
    state: GridState,
    npts_glo: int,
    iim: int,
    jjm: int,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    kindex: np.ndarray,
    contfrac_tmp: np.ndarray | None = None,
    *,
    mpi_size: int = 1,
    projection: ProjectionInfo | None = None,
    ij_to_latlon: Callable[[float, float], tuple[float, float]] | None = None,
    dxwrf: np.ndarray | None = None,
    dywrf: np.ndarray | None = None,
) -> GridState:
    """Populate global polygon arrays and complete the serial grid workflow.

    Fortran provenance: ``grid.f90::grid_stuff``, lines 416-470. The root
    polygon work is numerical; scatter/broadcast is delegated to the explicit
    serial contract in :func:`grid_scatter`.
    """

    global_state = state.global_state
    if not global_state.allocated:
        raise AllocationContractError("grid_allocate_glo must run before grid_stuff")
    if (int(npts_glo), int(iim), int(jjm)) != (
        global_state.nbp_glo,
        global_state.iim,
        global_state.jjm,
    ):
        raise AllocationContractError(
            "grid_stuff dimensions do not match global allocation"
        )
    topology = grid_topolylist(
        state.grid_type,
        state.nb_segments,
        int(npts_glo),
        int(iim),
        int(jjm),
        grid_lon,
        grid_lat,
        kindex,
        projection=projection,
        ij_to_latlon=ij_to_latlon,
        dxwrf=dxwrf,
        dywrf=dywrf,
    )
    fractions = (
        np.asarray(global_state.contfrac).copy()
        if contfrac_tmp is None
        else np.asarray(contfrac_tmp, dtype=np.float64)
    )
    if fractions.shape != (int(npts_glo),):
        raise ValueError("contfrac_tmp must have shape (npts_glo,)")
    if np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("contfrac_tmp must preserve the [0,1] continent-fraction mask")
    resolution = np.column_stack(
        (
            (topology.seglength[:, 0] + topology.seglength[:, 2]) / 2.0,
            (topology.seglength[:, 1] + topology.seglength[:, 3]) / 2.0,
        )
    )
    populated = replace(
        global_state,
        neighbours=topology.neighbours.copy(),
        headings=topology.headings.copy(),
        seglength=topology.seglength.copy(),
        corners=topology.corners.copy(),
        area=topology.area.copy(),
        resolution=resolution,
        lalo=topology.lalo.copy(),
        contfrac=fractions.copy(),
        index=np.asarray(kindex, dtype=np.int32).copy(),
        lon=np.asarray(grid_lon, dtype=np.float64).copy(),
        lat=np.asarray(grid_lat, dtype=np.float64).copy(),
    )
    return grid_scatter(
        replace(
            state,
            global_state=populated,
            global_grid=topology.global_grid,
            ilandindex=topology.ilandindex.copy(),
            jlandindex=topology.jlandindex.copy(),
        ),
        mpi_size=mpi_size,
    )


def _convert_index_base(
    ri: np.ndarray | float,
    rj: np.ndarray | float,
    index_base: Literal["fortran", "python"],
) -> tuple[np.ndarray | float, np.ndarray | float]:
    if index_base == "fortran":
        return ri, rj
    if index_base == "python":
        return np.asarray(ri) - 1.0, np.asarray(rj) - 1.0
    raise ValueError("index_base must be 'fortran' or 'python'")


def grid_toij_scal(
    lon: float,
    lat: float,
    *,
    projection: LatLonToIJ | None,
    index_base: Literal["fortran", "python"] = "fortran",
) -> tuple[float, float]:
    """Convert one lon/lat pair to projection indices with explicit base.

    Fortran provenance: ``grid.f90::grid_toij_scal``, lines 1047-1063. The
    projection adapter is ``module_llxy::latlon_to_ij`` and must return
    Fortran one-based fractional coordinates. ``index_base='python'`` is the
    sole zero-based conversion point and subtracts exactly one from each axis.
    """

    if projection is None:
        raise ProjectionInfrastructureBoundary(
            "grid_toij requires an initialized module_llxy projection adapter"
        )
    ri, rj = projection.latlon_to_ij(float(lat), float(lon))
    converted_ri, converted_rj = _convert_index_base(float(ri), float(rj), index_base)
    return float(converted_ri), float(converted_rj)


def grid_toij_1d(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    projection: LatLonToIJ | None,
    index_base: Literal["fortran", "python"] = "fortran",
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 wrapper for ``grid_toij_scal`` (Fortran lines 1067-1089)."""

    lon_array = np.asarray(lon, dtype=np.float64)
    lat_array = np.asarray(lat, dtype=np.float64)
    if lon_array.ndim != 1 or lat_array.shape != lon_array.shape:
        raise ValueError("lon and lat must have the same rank-1 shape")
    pairs = [
        grid_toij_scal(lo, la, projection=projection, index_base=index_base)
        for lo, la in zip(lon_array, lat_array)
    ]
    if not pairs:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    ri, rj = np.asarray(pairs, dtype=np.float64).T
    return ri, rj


def grid_toij_2d(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    projection: LatLonToIJ | None,
    index_base: Literal["fortran", "python"] = "fortran",
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-2 wrapper for ``grid_toij_scal`` (Fortran lines 1093-1118)."""

    lon_array = np.asarray(lon, dtype=np.float64)
    lat_array = np.asarray(lat, dtype=np.float64)
    if lon_array.ndim != 2 or lat_array.shape != lon_array.shape:
        raise ValueError("lon and lat must have the same rank-2 shape")
    flat_ri, flat_rj = grid_toij_1d(
        lon_array.ravel(),
        lat_array.ravel(),
        projection=projection,
        index_base=index_base,
    )
    return flat_ri.reshape(lon_array.shape), flat_rj.reshape(lon_array.shape)
