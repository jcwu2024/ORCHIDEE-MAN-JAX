"""Source-routed HYDROL and THERMOSOIL completion owners.

The routines here preserve the statement order and one-based indexing of the
Fortran procedures. Physical restart persistence, NetCDF transport, and grid
overlap construction remain explicit callback boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, NamedTuple

import numpy as np

from jax_orchidee.driver.interpolation_core12 import (
    Aggregate,
    AggregatePacket,
    AggregateRequest,
    InterpolationTarget,
)


HYDRO_SUBGRID_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/hydro_subgrid.f90::hydro_subgrid_main lines 42-248",
)
READ_REFSOC_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::read_refSOCfile lines 3241-3405",
)
HYDROL_FINALIZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_finalize lines 1682-1797",
)
HYDROL_ROTATION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_rotation_update lines 3591-3807",
)


class HydroSubgridResult(NamedTuple):
    fsat: np.ndarray
    fwet: np.ndarray
    fwt1: np.ndarray
    fwt2: np.ndarray
    fwt3: np.ndarray
    fwt4: np.ndarray
    fsat_index_fortran: np.ndarray
    fwet_index_fortran: np.ndarray
    provenance: tuple[str, ...] = HYDRO_SUBGRID_PROVENANCE


def hydro_subgrid_main(
    *,
    tab_fsat,
    tab_wtop,
    humtot,
    profil_froz_hydro,
    tab_fwet,
    tab_wtop_wet,
    pd_top,
    ruu_ch,
    ti_min,
    ti_max,
    pas,
    dz,
    wtd_bornes=(0.06, 0.12, 0.18, 0.24),
    min_sechiba=1.0e-8,
) -> HydroSubgridResult:
    """Evaluate ``hydro_subgrid_main`` with the source's table/index order.

    ``wtd_bornes`` routes the four module constants ``WTD1_borne`` through
    ``WTD4_borne``; callers must supply alternate source-configured constants
    explicitly. Fortran ``MINLOC`` indices are exposed as one-based values.
    """

    tab_fsat = np.asarray(tab_fsat, dtype=np.float64)
    tab_wtop = np.asarray(tab_wtop, dtype=np.float64)
    tab_fwet = np.asarray(tab_fwet, dtype=np.float64)
    tab_wtop_wet = np.asarray(tab_wtop_wet, dtype=np.float64)
    if tab_fsat.ndim != 2:
        raise ValueError("TOPMODEL tables must be rank 2")
    shape = tab_fsat.shape
    if any(table.shape != shape for table in (tab_wtop, tab_fwet, tab_wtop_wet)):
        raise ValueError("all TOPMODEL tables must have the same shape")
    npts, width = shape
    if width != 1000:
        raise ValueError(
            "hydro_subgrid_main Fortran tables must have exactly 1000 columns"
        )

    def vector(name, value):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (npts,):
            raise ValueError(f"{name} must have shape {(npts,)}")
        return array

    humtot = vector("humtot", humtot)
    ruu_ch = vector("ruu_ch", ruu_ch)
    ti_min = vector("ti_min", ti_min)
    ti_max = vector("ti_max", ti_max)
    pas = vector("pas", pas)
    profile = np.asarray(profil_froz_hydro, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[0] != npts or profile.shape[1] < 9:
        raise ValueError(
            "profil_froz_hydro must have shape (npts, nslm) with nslm >= 9"
        )
    if dz.ndim != 1 or dz.size < 9 or np.sum(dz[:9]) == 0.0:
        raise ValueError("dz must contain nine layers with a nonzero total thickness")
    pd_top = float(pd_top)
    if pd_top == 0.0:
        raise ValueError("pd_top must be nonzero for the unconditional source division")
    bornes = np.asarray(wtd_bornes, dtype=np.float64)
    if bornes.shape != (4,):
        raise ValueError("wtd_bornes must contain WTD1_borne through WTD4_borne")

    frozen_fraction = np.sum(profile[:, :9] * dz[None, :9], axis=1) / np.sum(dz[:9])
    wtop = np.minimum(humtot / (pd_top * 1000.0), tab_wtop[:, 0])
    wtop_wet = np.minimum(humtot / (pd_top * 1000.0), tab_wtop_wet[:, 0])
    fsat_index = np.argmin(np.abs(tab_wtop - wtop[:, None]), axis=1)
    wet_index = np.argmin(np.abs(tab_wtop_wet - wtop_wet[:, None]), axis=1)

    valid = (
        (ti_max != -99.99)
        & (ruu_ch >= min_sechiba)
        & (pd_top >= min_sechiba)
        & (pas >= min_sechiba)
    )
    wt_indices = np.full((npts, 4), 999, dtype=np.int64)
    denominator = (ruu_ch * pd_top / 1000.0) * pas
    for level, borne in enumerate(bornes):
        raw = np.zeros(npts, dtype=np.float64)
        raw[valid] = wet_index[valid] + 1 - (4.0 * borne / pd_top) / denominator[valid]
        # Fortran INT truncates toward zero and the source indices are one-based.
        wt_indices[valid, level] = np.trunc(raw[valid]).astype(np.int64)
    wt_indices = np.maximum(wt_indices, 1)
    if np.any(wt_indices > width):
        raise ValueError(
            "source wetland index exceeds the declared TOPMODEL table width"
        )

    fsat = np.zeros(npts, dtype=np.float64)
    fwet = np.zeros(npts, dtype=np.float64)
    fractions = np.zeros((npts, 4), dtype=np.float64)
    has_values = np.count_nonzero(tab_fsat > 0.0, axis=1) > 0
    points = np.nonzero(has_values)[0]
    fsat[points] = tab_fsat[points, fsat_index[points]]
    fwet[points] = tab_fwet[points, wet_index[points]]
    cumulative = fwet.copy()
    for level in range(4):
        current = tab_fwet[np.arange(npts), wt_indices[:, level] - 1] - cumulative
        fractions[:, level] = np.where(has_values, current, 0.0)
        cumulative = cumulative + fractions[:, level]

    degenerate = tab_wtop[:, 0] == tab_wtop[:, 899]
    positive_water = humtot > 0.0
    scale = np.where(positive_water, 1.0 - frozen_fraction, 1.0)
    fsat = np.where(degenerate, 0.0, fsat) * scale
    fwet = np.where(degenerate, 0.0, fwet) * scale
    fractions = np.where(degenerate[:, None], 0.0, fractions) * scale[:, None]
    return HydroSubgridResult(
        fsat,
        fwet,
        fractions[:, 0],
        fractions[:, 1],
        fractions[:, 2],
        fractions[:, 3],
        fsat_index + 1,
        wet_index + 1,
    )


@dataclass(frozen=True)
class RefSocFileData:
    """Structured ``flinget`` result in Fortran ``(lon, lat, level)`` order."""

    longitude: np.ndarray
    latitude: np.ndarray
    mask: np.ndarray
    soil_organic_carbon: np.ndarray


class RefSocReadResult(NamedTuple):
    refsoc: np.ndarray
    nbvmax_attempts: tuple[int, ...]
    source: RefSocFileData
    provenance: tuple[str, ...] = READ_REFSOC_PROVENANCE


RefSocReader = Callable[[str | Path], RefSocFileData]


def read_refsoc_netcdf(path: str | Path) -> RefSocFileData:
    """Read the four variables selected by Fortran lines 3307-3310."""

    import xarray as xr

    with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as dataset:
        required = ("longitude", "latitude", "mask", "soil_organic_carbon")
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            raise ValueError(f"refSOC file is missing variables: {missing}")
        lon_var = dataset["longitude"]
        lat_var = dataset["latitude"]
        if lon_var.ndim != 1 or lat_var.ndim != 1:
            raise ValueError("longitude and latitude must be rank-1 variables")
        lon_dim, lat_dim = lon_var.dims[0], lat_var.dims[0]
        mask_var = dataset["mask"]
        soc_var = dataset["soil_organic_carbon"]
        if lon_dim not in mask_var.dims or lat_dim not in mask_var.dims:
            raise ValueError("mask must carry longitude and latitude dimensions")
        if lon_dim not in soc_var.dims or lat_dim not in soc_var.dims:
            raise ValueError(
                "soil_organic_carbon must carry longitude and latitude dimensions"
            )
        mask_extra = [dim for dim in mask_var.dims if dim not in (lon_dim, lat_dim)]
        soc_extra = [dim for dim in soc_var.dims if dim not in (lon_dim, lat_dim)]
        longitude = np.asarray(lon_var.values, dtype=np.float64)
        latitude = np.asarray(lat_var.values, dtype=np.float64)
        mask_values = np.asarray(
            mask_var.transpose(lon_dim, lat_dim, *mask_extra).values
        )
        soc_values = np.asarray(soc_var.transpose(lon_dim, lat_dim, *soc_extra).values)
    mask_values = np.squeeze(mask_values)
    soc_values = np.squeeze(soc_values)
    if soc_values.ndim == 2:
        soc_values = soc_values[:, :, None]
    return RefSocFileData(
        longitude,
        latitude,
        np.asarray(mask_values, dtype=np.float64),
        np.asarray(soc_values, dtype=np.float64),
    )


def read_refsocfile(
    *,
    target: InterpolationTarget,
    aggregate: Aggregate,
    source: RefSocFileData | None = None,
    path: str | Path | None = None,
    reader: RefSocReader | None = None,
) -> RefSocReadResult:
    """Read and area-average SOC following ``read_refSOCfile`` lines 3274-3403."""

    if source is None:
        if path is None:
            raise ValueError(
                "read_refsocfile requires structured source data or an actual path"
            )
        selected_reader = read_refsoc_netcdf if reader is None else reader
        source = selected_reader(path)
    elif path is not None or reader is not None:
        raise ValueError(
            "supply either source data or a path/reader boundary, not both"
        )
    if not isinstance(source, RefSocFileData):
        raise TypeError("refSOC reader must return RefSocFileData")
    if not callable(aggregate):
        raise TypeError("aggregate must be callable")

    lon = np.asarray(source.longitude, dtype=np.float64)
    lat = np.asarray(source.latitude, dtype=np.float64)
    mask_lu = np.asarray(source.mask, dtype=np.float64)
    values = np.asarray(source.soil_organic_carbon, dtype=np.float64)
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("longitude and latitude must be rank 1")
    if mask_lu.shape != (lon.size, lat.size):
        raise ValueError("mask must have shape (nlon, nlat)")
    if values.ndim != 3 or values.shape[:2] != mask_lu.shape:
        raise ValueError("soil_organic_carbon must have shape (nlon, nlat, nlevel)")

    lalo = np.asarray(target.lalo, dtype=np.float64)
    resolution = np.asarray(target.resolution, dtype=np.float64)
    neighbours = np.asarray(target.neighbours, dtype=np.int64)
    contfrac = np.asarray(target.contfrac, dtype=np.float64)
    if lalo.ndim != 2 or lalo.shape[1] != 2 or resolution.shape != lalo.shape:
        raise ValueError("target lalo and resolution must have shape (nbpt, 2)")
    nbpt = lalo.shape[0]
    if neighbours.ndim != 2 or neighbours.shape[0] != nbpt or contfrac.shape != (nbpt,):
        raise ValueError("target neighbours/contfrac dimensions do not match lalo")

    lon_rel = np.broadcast_to(lon[:, None], mask_lu.shape).copy()
    lat_rel = np.broadcast_to(lat[None, :], mask_lu.shape).copy()
    mask = (mask_lu > 0.0).astype(np.int32)
    width = 16
    attempts: list[int] = []
    while True:
        attempts.append(width)
        packet: AggregatePacket = aggregate(
            AggregateRequest(
                source_rank=2,
                lalo=lalo.copy(),
                resolution=resolution.copy(),
                neighbours=neighbours.copy(),
                contfrac=contfrac.copy(),
                longitude=lon_rel,
                latitude=lat_rel,
                mask=mask,
                max_resolution_lon=-1.0,
                max_resolution_lat=-1.0,
                callsign="soil organic carbon",
                nbvmax=width,
            )
        )
        areas = np.asarray(packet.sub_area, dtype=np.float64)
        indices = np.asarray(packet.sub_index, dtype=np.int64)
        if areas.shape != (nbpt, width) or indices.shape != (nbpt, width, 2):
            raise ValueError(
                "aggregate packet dimensions must match nbpt and current nbvmax"
            )
        if packet.ok:
            break
        width *= 2

    refsoc = np.zeros((nbpt, values.shape[2]), dtype=np.float64)
    for point in range(nbpt):
        count = int(np.count_nonzero(areas[point] > 0.0))
        if count == 0:
            continue
        selected_area = areas[point, :count]
        selected_index = indices[point, :count]
        if np.any(selected_area <= 0.0):
            raise ValueError(
                "aggregate_p positive overlaps must occupy the first fopt slots"
            )
        if np.any(selected_index < 1):
            raise ValueError("aggregate_p sub_index values must be Fortran one-based")
        ii = selected_index[:, 0] - 1
        jj = selected_index[:, 1] - 1
        if np.any(ii >= lon.size) or np.any(jj >= lat.size):
            raise ValueError("aggregate_p sub_index exceeds source grid dimensions")
        total_area = np.sum(selected_area)
        refsoc[point] = (
            np.sum(values[ii, jj, :] * selected_area[:, None], axis=0) / total_area
        )
    return RefSocReadResult(refsoc, tuple(attempts), source)


class HydrolRestartWriteRequest(NamedTuple):
    rest_id: int
    name: str
    kjit: int
    value: np.ndarray
    source_lines: tuple[int, int]


class HydrolFinalizeResult(NamedTuple):
    requests: tuple[HydrolRestartWriteRequest, ...]
    write_results: tuple[object, ...]
    explicit_snow_result: object | None
    provenance: tuple[str, ...] = HYDROL_FINALIZE_PROVENANCE


RestartWriter = Callable[[HydrolRestartWriteRequest], object]


def explicitsnow_finalize_restart_packet(
    *, snowrho, snowtemp, snowdz, snowheat, snowgrain
) -> dict[str, np.ndarray]:
    """Select all explicit-snow restart fields.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_finalize`` lines
    566-587.
    """

    packet = {
        name: np.asarray(value)
        for name, value in (
            ("snowrho", snowrho),
            ("snowtemp", snowtemp),
            ("snowdz", snowdz),
            ("snowheat", snowheat),
            ("snowgrain", snowgrain),
        )
    }
    shape = packet["snowrho"].shape
    if len(shape) != 2 or any(value.shape != shape for value in packet.values()):
        raise ValueError("explicit-snow finalize fields must share (npts,nsnow)")
    return packet


def hydrol_finalize(
    *,
    kjit: int,
    rest_id: int,
    state: Mapping[str, object],
    inputs: Mapping[str, object],
    restart_writer: RestartWriter,
    use_refsoc_hydrol=False,
    check_waterbal=False,
    ok_explicitsnow=False,
    explicitsnow_finalizer: Callable[[Mapping[str, object]], object] | None = None,
) -> HydrolFinalizeResult:
    """Route every ``restput_p`` and optional snow finalizer in source order."""

    if not callable(restart_writer):
        raise TypeError("restart_writer must be callable")
    ordered = [
        ("moistc", "mc", 1727),
        ("moistcl", "mcl", 1728),
        ("us", "us", 1730),
        ("free_drain_coef", "free_drain_coef", 1732),
        ("zwt_force", "zwt_force", 1733),
        ("water2infilt", "water2infilt", 1734),
        ("ae_ns", "ae_ns", 1735),
        ("vegstress", "vegstress", 1736),
        ("snow", "snow", 1737),
        ("snow_age", "snow_age", 1738),
        ("snow_nobio", "snow_nobio", 1739),
        ("snow_nobio_age", "snow_nobio_age", 1740),
        ("qsintveg", "qsintveg", 1741),
        ("evap_bare_lim_ns", "evap_bare_lim_ns", 1742),
        ("evap_bare_lim", "evap_bare_lim", 1743),
        ("resdist", "resdist", 1744),
        ("vegtot_old", "vegtot_old", 1745),
        ("drysoil_frac", "drysoil_frac", 1746),
        ("humrel", "humrel", 1747),
    ]
    if use_refsoc_hydrol:
        ordered.append(("refSOC_1d", "refSOC_1d", 1748))
    ordered.extend(
        [
            ("fwet_out", "fwet_out", 1754),
            ("run2peat", "run2peat", 1760),
            ("wt_ab", "wt_ab", 1763),
            ("wtp", "wtp", 1766),
            ("fwet_new", "fwet_new", 1769),
            ("liqwt_ratio", "liqwt_ratio", 1772),
            ("wt_ab_tide", "wt_ab_tide", 1776),
            ("run2man", "run2man", 1779),
        ]
    )
    if check_waterbal:
        ordered.append(("tot_water_beg", "tot_water_end", 1782))
    ordered.extend(
        [
            ("tot_watveg_beg", "tot_watveg_beg", 1786),
            ("tot_watsoil_beg", "tot_watsoil_beg", 1787),
            ("snow_beg", "snow_beg", 1788),
        ]
    )
    merged = {**state, **inputs}
    missing = [
        source_name for _, source_name, _ in ordered if source_name not in merged
    ]
    if missing:
        raise ValueError(f"hydrol_finalize missing source state: {missing}")
    requests = tuple(
        HydrolRestartWriteRequest(
            rest_id, output_name, kjit, np.asarray(merged[source_name]), (line, line)
        )
        for output_name, source_name, line in ordered
    )
    snow_payload = None
    if ok_explicitsnow:
        if explicitsnow_finalizer is None or not callable(explicitsnow_finalizer):
            raise ValueError(
                "ok_explicitsnow requires an explicit explicitsnow_finalizer callback"
            )
        snow_names = ("snowrho", "snowtemp", "snowdz", "snowheat", "snowgrain")
        missing_snow = [name for name in snow_names if name not in inputs]
        if missing_snow:
            raise ValueError(f"explicitsnow_finalize missing inputs: {missing_snow}")
        snow_payload = {
            "kjit": kjit,
            "rest_id": rest_id,
            **{name: inputs[name] for name in snow_names},
        }

    results = tuple(restart_writer(request) for request in requests)
    snow_result = (
        explicitsnow_finalizer(snow_payload) if snow_payload is not None else None
    )
    return HydrolFinalizeResult(requests, results, snow_result)


class HydrolRotationResult(NamedTuple):
    mc: np.ndarray
    water2infilt: np.ndarray
    qsintveg: np.ndarray
    tmc: np.ndarray
    humtot: np.ndarray
    resdist: np.ndarray
    rot_matrix_tile: np.ndarray
    maxfrac_new: np.ndarray
    provenance: tuple[str, ...] = HYDROL_ROTATION_PROVENANCE


def hydrol_rotation_update(
    *,
    ip_fortran: int,
    rot_matrix,
    old_veget_max,
    veget_max,
    soiltile,
    qsintveg,
    pref_soil_veg,
    mc,
    water2infilt,
    tmc,
    humtot,
    resdist,
    dz,
    min_sechiba=1.0e-8,
    check_cwrr=False,
    allowed_err=1.0e-8,
) -> HydrolRotationResult:
    """Apply one source point's vegetation-to-soil-tile water rotation."""

    rot = np.asarray(rot_matrix, dtype=np.float64)
    old = np.asarray(old_veget_max, dtype=np.float64)
    veget = np.asarray(veget_max, dtype=np.float64)
    soil = np.asarray(soiltile, dtype=np.float64)
    canopy = np.asarray(qsintveg, dtype=np.float64).copy()
    pref = np.asarray(pref_soil_veg, dtype=np.int64)
    mc_out = np.asarray(mc, dtype=np.float64).copy()
    infil = np.asarray(water2infilt, dtype=np.float64).copy()
    tmc_out = np.asarray(tmc, dtype=np.float64).copy()
    hum_out = np.asarray(humtot, dtype=np.float64).copy()
    res_out = np.asarray(resdist, dtype=np.float64).copy()
    dz = np.asarray(dz, dtype=np.float64)
    if veget.ndim != 2 or soil.ndim != 2 or canopy.shape != veget.shape:
        raise ValueError(
            "veget_max/qsintveg must be (npts,nvm) and soiltile must be rank 2"
        )
    npts, nvm = veget.shape
    nstm = soil.shape[1]
    if rot.shape != (nvm, nvm) or old.shape != (nvm,) or pref.shape != (nvm,):
        raise ValueError("rotation/PFT arrays do not match nvm")
    if not 1 <= int(ip_fortran) <= npts:
        raise ValueError("ip_fortran is outside the one-based landpoint range")
    ip = int(ip_fortran) - 1
    if mc_out.ndim != 3 or mc_out.shape[0] != npts or mc_out.shape[2] != nstm:
        raise ValueError("mc must have shape (npts,nslm,nstm)")
    nslm = mc_out.shape[1]
    if nslm < 2 or dz.shape != (nslm,):
        raise ValueError("dz must match mc layers and at least two layers are required")
    expected_tile = (npts, nstm)
    if any(value.shape != expected_tile for value in (infil, tmc_out, res_out)):
        raise ValueError("water2infilt, tmc, and resdist must match soiltile")
    if hum_out.shape != (npts,) or np.any(pref < 1) or np.any(pref > nstm):
        raise ValueError("humtot or one-based pref_soil_veg dimensions are invalid")
    if np.sum(rot) <= 0.0:
        raise RuntimeError("Fortran soil_upd is undefined when SUM(rot_matrix) <= 0")

    maxfrac = old.copy()
    maxfrac_new = old.copy()
    tile_rot = np.zeros((nstm, nstm), dtype=np.float64)
    for source_pft in range(nvm):
        for target_pft in range(nvm):
            amount = rot[source_pft, target_pft]
            if amount > 0.0:
                maxfrac_new[target_pft] += maxfrac[source_pft] * amount
                maxfrac_new[source_pft] -= maxfrac[source_pft] * amount
                source_tile = pref[source_pft] - 1
                target_tile = pref[target_pft] - 1
                tile_rot[source_tile, target_tile] += (
                    amount * maxfrac[source_pft] / res_out[ip, source_tile]
                )
                fraction = tile_rot[source_tile, target_tile]
                if fraction > 1.0 or fraction <= 0.0:
                    raise RuntimeError("soiltile error in hydrol_rotation")
    if np.sum(np.abs(maxfrac_new - veget[ip])) > min_sechiba:
        raise RuntimeError("hydrol_rotation: fraction conversion error")

    tmc_old = tmc_out[ip].copy() if check_cwrr else None
    canopy_old = canopy[ip].copy() if check_cwrr else None
    for pft in range(nvm):
        if maxfrac_new[pft] < min_sechiba and canopy[ip, pft] > 0.0:
            tile = pref[pft] - 1
            infil[ip, tile] += canopy[ip, pft] / maxfrac[pft]
            canopy[ip, pft] = 0.0

    mc_old = mc_out[ip].copy()
    infil_old = infil[ip].copy()
    for target_tile in range(nstm):
        if np.sum(tile_rot[:, target_tile]) > min_sechiba:
            incoming = tile_rot[:, target_tile] > min_sechiba
            mc_dilu = np.where(incoming[None, :], mc_old, 0.0)
            infil_dilu = np.where(incoming, infil_old, 0.0)
            mc_out[ip, :, target_tile] = (
                mc_old[:, target_tile]
                * res_out[ip, target_tile]
                * (1.0 - np.sum(tile_rot[target_tile, :]))
            )
            infil[ip, target_tile] = (
                infil_old[target_tile]
                * res_out[ip, target_tile]
                * (1.0 - np.sum(tile_rot[target_tile, :]))
            )
            for source_tile in range(nstm):
                weight = res_out[ip, source_tile] * tile_rot[source_tile, target_tile]
                mc_out[ip, :, target_tile] += weight * mc_dilu[:, source_tile]
                infil[ip, target_tile] += weight * infil_dilu[source_tile]
            if soil[ip, target_tile] <= 0.0:
                raise RuntimeError(
                    "hydrol_rotation_update: target tile has no proportion"
                )
            mc_out[ip, :, target_tile] /= soil[ip, target_tile]
            infil[ip, target_tile] /= soil[ip, target_tile]
    inactive = soil[ip] < min_sechiba
    infil[ip, inactive] = 0.0
    mc_out[ip, :, inactive] = 0.0

    for tile in range(nstm):
        value = dz[1] * (3.0 * mc_out[ip, 0, tile] + mc_out[ip, 1, tile]) / 8.0
        for layer in range(1, nslm - 1):
            value += (
                dz[layer]
                * (3.0 * mc_out[ip, layer, tile] + mc_out[ip, layer - 1, tile])
                / 8.0
            )
            value += (
                dz[layer + 1]
                * (3.0 * mc_out[ip, layer, tile] + mc_out[ip, layer + 1, tile])
                / 8.0
            )
        value += dz[-1] * (3.0 * mc_out[ip, -1, tile] + mc_out[ip, -2, tile]) / 8.0
        tmc_out[ip, tile] = value + infil[ip, tile]
    hum_out[ip] = np.sum(soil[ip] * tmc_out[ip])
    if check_cwrr:
        error = abs(
            np.sum(tmc_out[ip] * soil[ip])
            - np.sum(tmc_old * res_out[ip])
            + np.sum(canopy[ip])
            - np.sum(canopy_old)
        )
        if error > allowed_err:
            raise RuntimeError(
                f"hydrol_rotation_update water balance error {error} > {allowed_err}"
            )
    res_out[:, :] = soil
    return HydrolRotationResult(
        mc_out, infil, canopy, tmc_out, hum_out, res_out, tile_rot, maxfrac_new
    )
