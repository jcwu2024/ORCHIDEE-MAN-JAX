"""Trace schema metadata for Driver/Grid exact-parity unlocks.

This module defines required trace groups and column names for future Fortran
instrumentation. It does not read traces, compute grid geometry, or infer
missing paper-case static inputs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceGroup:
    """Named trace group with required CSV columns."""

    name: str
    columns: tuple[str, ...]
    fortran_provenance: tuple[str, ...]
    consumer: str


DOMAIN_FORCING_COLUMNS = (
    "year",
    "tstep",
    "ik",
    "i_fortran",
    "j_fortran",
    "kindex",
    "nav_lon",
    "nav_lat",
    "lalo_lat",
    "lalo_lon",
    "contfrac",
    "area",
    "Tair",
    "PSurf",
    "Qair",
    "Wind_E",
    "Wind_N",
    "Rainf",
    "Snowf",
    "SWdown",
    "LWdown",
    "Height_Lev1",
    "Height_Levuv",
)

GRID_GEOMETRY_COLUMNS = (
    "ik",
    "kindex",
    "lalo_lat",
    "lalo_lon",
    "contfrac",
    "area",
    "resolution_x",
    "resolution_y",
    "neighbour_1",
    "neighbour_2",
    "neighbour_3",
    "neighbour_4",
    "neighbour_5",
    "neighbour_6",
    "neighbour_7",
    "neighbour_8",
    "corner_1_lon",
    "corner_1_lat",
    "corner_2_lon",
    "corner_2_lat",
    "corner_3_lon",
    "corner_3_lat",
    "corner_4_lon",
    "corner_4_lat",
    "seglength_1",
    "seglength_2",
    "seglength_3",
    "seglength_4",
)

SOIL_AGGREGATION_COLUMNS = (
    "ik",
    "fopt",
    "source_i",
    "source_j",
    "sub_area",
    "soiltext",
    "soil_ph_source",
    "poor_soils_source",
    "bulk_dens_source",
    "soilclass_1",
    "soilclass_2",
    "soilclass_3",
    "njsc",
    "clay_frac",
    "sand_frac",
    "bulk_dens",
    "soil_ph",
    "poor_soils",
)

BBOX_FIELD_COLUMNS = (
    "field_name",
    "ik",
    "time_index",
    "lon_low",
    "lon_up",
    "lat_low",
    "lat_up",
    "source_count",
    "fallback_nearest",
    "nearest_source_i",
    "nearest_source_j",
    "mean_value",
    "final_value",
)

INTERSURF_BOUNDARY_COLUMNS = (
    "call_name",
    "year",
    "tstep",
    "ik",
    "kindex",
    "lalo_lat",
    "lalo_lon",
    "contfrac",
    "resolution_x",
    "resolution_y",
    "temp_air",
    "pb",
    "qair",
    "u",
    "v",
    "precip_rain",
    "precip_snow",
    "swdown",
    "lwdown",
    "zlev",
    "ccanopy",
)


TRACE_GROUPS = (
    TraceGroup(
        name="domain_forcing_selected_point",
        columns=DOMAIN_FORCING_COLUMNS,
        fortran_provenance=(
            "dim2_driver.f90:415-435",
            "readdim2.f90:49-495,660-1687,1691-1869,1899-1968,1972-2051,2263-2341",
        ),
        consumer="jax_orchidee.driver.domain",
    ),
    TraceGroup(
        name="grid_geometry_after_grid_stuff",
        columns=GRID_GEOMETRY_COLUMNS,
        fortran_provenance=(
            "grid.f90:416-470,490-631,663-718",
            "haversine.f90:394-488,667-736,863-888",
        ),
        consumer="jax_orchidee.driver.domain and jax_orchidee.driver.static",
    ),
    TraceGroup(
        name="soil_aggregation",
        columns=SOIL_AGGREGATION_COLUMNS,
        fortran_provenance=(
            "slowproc.f90:4284-4925",
            "interpol_help.f90:46-466,830-889",
        ),
        consumer="jax_orchidee.driver.static",
    ),
    TraceGroup(
        name="salinity_bbox",
        columns=BBOX_FIELD_COLUMNS,
        fortran_provenance=("slowproc.f90:2494-2513,6300-6570",),
        consumer="jax_orchidee.driver.static.bbox_center_mean",
    ),
    TraceGroup(
        name="tide_bbox",
        columns=BBOX_FIELD_COLUMNS,
        fortran_provenance=("slowproc.f90:2515-2534,6029-6297",),
        consumer="jax_orchidee.driver.static.bbox_center_mean",
    ),
    TraceGroup(
        name="intersurf_first_step_boundary",
        columns=INTERSURF_BOUNDARY_COLUMNS,
        fortran_provenance=(
            "dim2_driver.f90:1108-1125,1293-1305",
            "intersurf.f90:113-152,377-387,458-500,640-648",
        ),
        consumer="jax_orchidee.driver.bundle",
    ),
)


def trace_group_names() -> tuple[str, ...]:
    """Return available trace group names.

    Fortran provenance: schema groups mirror the exact trace insertion points
    audited in `dim2_driver.f90`, `readdim2.f90`, `grid.f90`,
    `haversine.f90`, `slowproc.f90`, and `interpol_help.f90`; this function
    exposes metadata only and computes no model geometry.
    """

    return tuple(group.name for group in TRACE_GROUPS)


def required_columns(group_name: str) -> tuple[str, ...]:
    """Return required columns for one trace group.

    Fortran provenance: column groups correspond to the trace insertion spans
    listed in each `TraceGroup.fortran_provenance`. This helper is schema-only;
    it does not infer `resolution`, `neighbours`, bboxes, or overlaps.
    """

    for group in TRACE_GROUPS:
        if group.name == group_name:
            return group.columns
    raise KeyError(f"unknown trace group: {group_name}")


def trace_group(group_name: str) -> TraceGroup:
    """Return a trace group definition by name.

    Fortran provenance: this metadata maps future trace files to audited
    Fortran insertion spans and JAX consumers without computing static state.
    """

    for group in TRACE_GROUPS:
        if group.name == group_name:
            return group
    raise KeyError(f"unknown trace group: {group_name}")
