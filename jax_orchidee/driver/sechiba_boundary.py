"""SECHIBA first-step boundary payload for the verified driver bundle.

The payload maps source-backed driver inputs to Fortran-facing names used by
`intersurf_initialize_2d` / `intersurf_main_2d`. Fields requiring exact trace
truth remain explicit missing metadata until trace adapters supply them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import jax.numpy as jnp
from jax import core as jax_core

from jax_orchidee.driver.bundle import DriverFirstStepBundle, DriverStepBundle
from jax_orchidee.driver.trace import StaticTraceFields


@dataclass(frozen=True)
class IntersurfFirstStepPayload:
    """Land-point payload for first-step SECHIBA boundary tests."""

    kjpindex: int
    nbindex: int
    kindex: np.ndarray
    lalo: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    contfrac: np.ndarray
    resolution: np.ndarray | None
    neighbours: np.ndarray | None
    corners: np.ndarray | None
    seglength: np.ndarray | None
    zlev: np.ndarray
    zlevuv: np.ndarray
    u: np.ndarray
    v: np.ndarray
    qair: np.ndarray
    temp_air: np.ndarray
    pb: np.ndarray
    precip_rain: np.ndarray
    precip_snow: np.ndarray
    lwdown: np.ndarray
    swdown: np.ndarray
    ccanopy: np.ndarray
    veget_max: np.ndarray
    veget: np.ndarray | None
    soiltile: np.ndarray
    njsc: np.ndarray | None
    clay_frac: np.ndarray | None
    sand_frac: np.ndarray | None
    silt_frac: np.ndarray | None
    bulk_dens: np.ndarray | None
    soil_ph: np.ndarray | None
    poor_soils: np.ndarray | None
    soilclass_sub_index: np.ndarray | None
    soilclass_sub_area: np.ndarray | None
    soilclass: np.ndarray | None
    salinity: np.ndarray | None
    tide_height: np.ndarray | None
    missing_fields: tuple[str, ...]


def _land_vector(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1)


def _driver_wind_at_scalar_height(wind, *, zlev, zlevuv, z0=0.1) -> np.ndarray:
    """Convert forcing wind level to SECHIBA scalar level.

    Fortran provenance: `dim2_driver.f90`, lines 796 and 1011-1020 initialize
    `z0(:,:)=0.1` and compute `for_u/for_v = wind *
    LOG(zlev_vec/z0)/LOG(zlevuv_vec/z0)` before the intersurf calls. After the
    first `intersurf_initialize_2d`, `dim2_driver.f90`, lines 1144-1180 may
    replace `z0` with restart/missing driver state and recompute the wind before
    `intersurf_main_2d`.
    """

    use_jax = any(isinstance(value, jax_core.Tracer) for value in (wind, zlev, zlevuv, z0))
    vector = (
        (lambda value: jnp.asarray(value, dtype=jnp.float64).reshape(-1))
        if use_jax
        else _land_vector
    )
    wind = vector(wind)
    zlev = vector(zlev)
    zlevuv = vector(zlevuv)
    z0 = vector(z0)
    if z0.size == 1 and wind.size != 1:
        z0 = jnp.full_like(wind, z0[0]) if use_jax else np.full_like(wind, float(z0[0]))
    if not (wind.shape == zlev.shape == zlevuv.shape == z0.shape):
        raise ValueError("wind, zlev, zlevuv, and z0 must share land-vector shape")
    log = jnp.log if use_jax else np.log
    return wind * log(zlev / z0) / log(zlevuv / z0)


def build_intersurf_first_step_payload(
    bundle: DriverFirstStepBundle | DriverStepBundle,
    *,
    static_trace_fields: StaticTraceFields | None = None,
    driver_z0_for_wind: np.ndarray | float | None = None,
    dt_sechiba: float = 1800.0,
) -> IntersurfFirstStepPayload:
    """Map a DriverFirstStepBundle to first-step Fortran-facing fields.

    Fortran provenance: `dim2_driver.f90`, lines 851-853 convert driver
    surface pressure from Pa to `for_psurf=pb/100` because SECHIBA expects hPa.
    Lines 796 and 1011-1020 convert wind from `zlevuv` to `zlev` using the
    initial driver `z0=0.1` before the first intersurf call. Lines 1144-1180
    then recompute the wind for `intersurf_main_2d` with restart/missing-state
    `z0`; callers that are packaging the post-initialize main-call boundary
    pass that value through `driver_z0_for_wind`. Lines 869-871, 1073-1075,
    and 1259-1261 pass precipitation to intersurf as `precip_* * dt`, i.e.
    water amount per SECHIBA step rather than the forcing-file rate.
    Lines 1108-1125 and 1293-1305 pass `kindex`, lon/lat, `for_contfrac`,
    `for_resolution`, first-level forcing fields, and `for_ccanopy` to
    `intersurf_initialize_2d` and `intersurf_main_2d`. `intersurf.f90`, lines
    113-152, 377-387, 458-500, and 640-648, forwards gathered land-point fields
    to SECHIBA. This function packages already verified bundle values only and
    does not compute `resolution`, `neighbours`, `soilclass`, salinity, tide,
    or process state.
    """

    domain = bundle.domain
    forcing = bundle.forcing
    restart = bundle.restart_anchors
    static_trace_fields = static_trace_fields or bundle.static_trace_fields or StaticTraceFields()
    wind_z0 = 0.1 if driver_z0_for_wind is None else driver_z0_for_wind
    dt_sechiba = float(dt_sechiba)

    missing = set(bundle.missing_geometry_fields)
    missing.update(bundle.missing_static_fields)

    njsc = static_trace_fields.njsc if static_trace_fields.njsc is not None else (None if restart is None else restart.njsc)
    clay_frac = static_trace_fields.clay_frac if static_trace_fields.clay_frac is not None else (None if restart is None else restart.clay_frac)
    sand_frac = static_trace_fields.sand_frac if static_trace_fields.sand_frac is not None else (None if restart is None else restart.sand_frac)
    silt_frac = static_trace_fields.silt_frac
    bulk_dens = static_trace_fields.bulk_dens if static_trace_fields.bulk_dens is not None else (None if restart is None else restart.bulk_dens)
    soil_ph = static_trace_fields.soil_ph if static_trace_fields.soil_ph is not None else (None if restart is None else restart.soil_ph)
    poor_soils = static_trace_fields.poor_soils if static_trace_fields.poor_soils is not None else (None if restart is None else restart.poor_soils)

    soilclass = static_trace_fields.soilclass
    salinity = static_trace_fields.salinity
    tide_height = static_trace_fields.tide_height
    for name, value in (
        ("soilclass", soilclass),
        ("njsc", njsc),
        ("clay_frac", clay_frac),
        ("sand_frac", sand_frac),
        ("silt_frac", silt_frac),
        ("bulk_dens", bulk_dens),
        ("soil_ph", soil_ph),
        ("poor_soils", poor_soils),
        ("soilclass_sub_index", static_trace_fields.soilclass_sub_index),
        ("soilclass_sub_area", static_trace_fields.soilclass_sub_area),
        ("salinity", salinity),
        ("tide_height", tide_height),
    ):
        if value is not None:
            missing.discard(name)
    if domain.resolution is not None:
        missing.discard("resolution")
    if domain.neighbours is not None:
        missing.discard("neighbours")
    if domain.corners is not None:
        missing.discard("corners")
    if domain.seglength is not None:
        missing.discard("seglength")
    if bundle.vegetation.veget is not None:
        missing.discard("veget")

    return IntersurfFirstStepPayload(
        kjpindex=domain.nbindex,
        nbindex=domain.nbindex,
        kindex=domain.kindex,
        lalo=domain.lalo,
        lon=domain.lon,
        lat=domain.lat,
        contfrac=domain.contfrac_land,
        resolution=domain.resolution,
        neighbours=domain.neighbours,
        corners=domain.corners,
        seglength=domain.seglength,
        zlev=_land_vector(forcing.zlev),
        zlevuv=_land_vector(forcing.zlevuv),
        u=_driver_wind_at_scalar_height(forcing.u, zlev=forcing.zlev, zlevuv=forcing.zlevuv, z0=wind_z0),
        v=_driver_wind_at_scalar_height(forcing.v, zlev=forcing.zlev, zlevuv=forcing.zlevuv, z0=wind_z0),
        qair=_land_vector(forcing.qair),
        temp_air=_land_vector(forcing.temp_air),
        pb=_land_vector(forcing.pb) / 100.0,
        precip_rain=_land_vector(forcing.precip_rain) * dt_sechiba,
        precip_snow=_land_vector(forcing.precip_snow) * dt_sechiba,
        lwdown=_land_vector(forcing.lwdown),
        swdown=_land_vector(forcing.swdown),
        ccanopy=bundle.ccanopy,
        veget_max=bundle.vegetation.veget_max,
        veget=bundle.vegetation.veget,
        soiltile=bundle.vegetation.soiltile,
        njsc=njsc,
        clay_frac=clay_frac,
        sand_frac=sand_frac,
        silt_frac=silt_frac,
        bulk_dens=bulk_dens,
        soil_ph=soil_ph,
        poor_soils=poor_soils,
        soilclass_sub_index=static_trace_fields.soilclass_sub_index,
        soilclass_sub_area=static_trace_fields.soilclass_sub_area,
        soilclass=soilclass,
        salinity=salinity,
        tide_height=tide_height,
        missing_fields=tuple(sorted(missing)),
    )


def intersurf_payload_with_driver_wind_z0(
    payload: IntersurfFirstStepPayload,
    *,
    forcing,
    driver_z0_for_wind: np.ndarray | float | None,
) -> IntersurfFirstStepPayload:
    """Return ``payload`` with wind recomputed for the current driver ``z0``.

    Fortran provenance is the same as ``build_intersurf_first_step_payload``:
    ``dim2_driver.f90`` lines 1144-1180 recompute ``for_u``/``for_v`` from
    the current driver roughness before the main intersurf call.  All other
    payload fields are unchanged and remain sourced from the prebuilt step
    boundary.
    """

    wind_z0 = 0.1 if driver_z0_for_wind is None else driver_z0_for_wind
    return replace(
        payload,
        u=_driver_wind_at_scalar_height(
            forcing.u,
            zlev=forcing.zlev,
            zlevuv=forcing.zlevuv,
            z0=wind_z0,
        ),
        v=_driver_wind_at_scalar_height(
            forcing.v,
            zlev=forcing.zlev,
            zlevuv=forcing.zlevuv,
            z0=wind_z0,
        ),
    )
