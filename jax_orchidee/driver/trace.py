"""Trace validation and explicit-truth adapters for Driver/Grid closure.

This module works only with trace rows supplied by the caller. It validates
columns and packages explicit trace truth for later driver/static tests; it
does not compute model geometry, infer bboxes, or mutate process state.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from jax_orchidee.driver.domain import DomainGrid
from jax_orchidee.driver.trace_schema import required_columns
from jax_orchidee.trace.fixed_format import read_records


TEXT_COLUMNS = {"field_name", "call_name"}


@dataclass(frozen=True)
class TraceValidationResult:
    """Column-level validation result for one trace group."""

    group_name: str
    row_count: int
    required_columns: tuple[str, ...]
    present_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    unknown_group: bool = False

    @property
    def is_valid(self) -> bool:
        return (not self.unknown_group) and not self.missing_columns


@dataclass(frozen=True)
class TraceTable:
    """Validated trace rows plus useful metadata."""

    group_name: str
    rows: tuple[dict[str, object], ...]
    validation: TraceValidationResult


@dataclass(frozen=True)
class StaticTraceFields:
    """Static fields extracted only from explicit trace rows."""

    soilclass: np.ndarray | None = None
    njsc: np.ndarray | None = None
    clay_frac: np.ndarray | None = None
    sand_frac: np.ndarray | None = None
    silt_frac: np.ndarray | None = None
    bulk_dens: np.ndarray | None = None
    soil_ph: np.ndarray | None = None
    poor_soils: np.ndarray | None = None
    soilclass_sub_index: np.ndarray | None = None
    soilclass_sub_area: np.ndarray | None = None
    salinity: np.ndarray | None = None
    tide_height: np.ndarray | None = None
    blocked_fields: tuple[str, ...] = ()


SERVER_1961_TRACE_FILENAMES = {
    "driver_forcing": "orchjax_driver_forcing_trace.txt",
    "grid": "orchjax_grid_trace.txt",
    "intersurf_boundary": "orchjax_intersurf_boundary_trace.txt",
    "intersurf_main": "orchjax_intersurf_main_trace.txt",
    "salinity": "orchjax_slowproc_read_annual_trace.txt",
    "tide": "orchjax_slowproc_read_data_trace.txt",
    "soil": "orchjax_slowproc_soilt_trace.txt",
}


def _as_rows(rows: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    return tuple(dict(row) for row in rows)


def validate_trace_rows(group_name: str, rows: Iterable[Mapping[str, object]]) -> TraceValidationResult:
    """Validate required columns for supplied trace rows.

    Fortran provenance: this validator mirrors insertion spans registered in
    `trace_schema.py` for `dim2_driver.f90`, `readdim2.f90`, `grid.f90`,
    `haversine.f90`, `slowproc.f90`, `interpol_help.f90`, and `intersurf.f90`.
    It validates metadata only and does not compute geometry or static fields.
    """

    materialized = _as_rows(rows)
    present: set[str] = set()
    for row in materialized:
        present.update(row.keys())
    try:
        required = required_columns(group_name)
    except KeyError:
        return TraceValidationResult(
            group_name=group_name,
            row_count=len(materialized),
            required_columns=(),
            present_columns=tuple(sorted(present)),
            missing_columns=(),
            unknown_group=True,
        )
    missing = tuple(column for column in required if column not in present)
    return TraceValidationResult(
        group_name=group_name,
        row_count=len(materialized),
        required_columns=required,
        present_columns=tuple(sorted(present)),
        missing_columns=missing,
    )


def coerce_trace_row_values(row: Mapping[str, object]) -> dict[str, object]:
    """Convert numeric-looking trace values while preserving text columns.

    Fortran provenance: future traces target numeric fields from the audited
    driver/static spans, but group labels such as `field_name` and `call_name`
    are textual. This helper performs schema-adjacent parsing only; it does
    not infer missing physics, geometry, or units.
    """

    out: dict[str, object] = {}
    for key, value in row.items():
        if key in TEXT_COLUMNS or value is None:
            out[key] = value
            continue
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            out[key] = value
            continue
        text = str(value).strip()
        if text == "":
            out[key] = value
            continue
        try:
            number = float(text)
        except ValueError:
            out[key] = value
        else:
            out[key] = int(number) if number.is_integer() else number
    return out


def read_trace_csv(path: str | Path, group_name: str, *, coerce_numeric: bool = True) -> TraceTable:
    """Read a caller-supplied CSV trace and validate its schema group.

    Fortran provenance: CSV columns are expected to follow the insertion plan
    for the audited Fortran spans in `docs/source_audits/driver_static_trace_
    instrumentation_plan.md`. The function reads and validates explicit trace
    rows only; it does not assume a trace exists or synthesize missing columns.
    """

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = tuple(coerce_trace_row_values(row) if coerce_numeric else dict(row) for row in reader)
    validation = validate_trace_rows(group_name, rows)
    return TraceTable(group_name=group_name, rows=rows, validation=validation)


def read_fixed_format_driver_forcing_trace(path: str | Path) -> tuple[dict[str, object], ...]:
    """Read the 2026-06-23 fixed-format first-forcing trace.

    Fortran provenance: consumes the local server trace inserted after
    `dim2_driver.f90` lines 415-435 and the `forcing_READ` active path through
    `readdim2.f90` lines 660-1869. The fixed-format record contains the
    selected first forcing point and copied `for_resolution`; it has no header
    and no forcing `Areas` field, so this parser does not synthesize area.
    """

    tokens = _trace_tokens(path)
    row_width = 21
    if len(tokens) % row_width != 0:
        raise ValueError(f"unexpected driver forcing trace token count: {len(tokens)}")

    rows: list[dict[str, object]] = []
    for offset in range(0, len(tokens), row_width):
        chunk = tokens[offset : offset + row_width]
        if chunk[0] != "first_forcing":
            raise ValueError(f"unexpected driver forcing record label: {chunk[0]}")
        rows.append(
            {
                "record_name": chunk[0],
                "tstep_fortran": _as_int(chunk[1]),
                "ik": _as_int(chunk[2]),
                "i_fortran": _as_int(chunk[3]),
                "j_fortran": _as_int(chunk[4]),
                "nav_lon": _as_float(chunk[5]),
                "nav_lat": _as_float(chunk[6]),
                "lalo_lon": _as_float(chunk[5]),
                "lalo_lat": _as_float(chunk[6]),
                "contfrac": _as_float(chunk[7]),
                "resolution_x": _as_float(chunk[8]),
                "resolution_y": _as_float(chunk[9]),
                "Tair": _as_float(chunk[10]),
                "PSurf": _as_float(chunk[11]),
                "Qair": _as_float(chunk[12]),
                "Wind_N": _as_float(chunk[13]),
                "Wind_E": _as_float(chunk[14]),
                "Rainf": _as_float(chunk[15]),
                "Snowf": _as_float(chunk[16]),
                "SWdown": _as_float(chunk[17]),
                "LWdown": _as_float(chunk[18]),
                "Height_Lev1": _as_float(chunk[19]),
                "Height_Levuv": _as_float(chunk[20]),
            }
        )
    return tuple(rows)


def read_fixed_format_intersurf_trace(path: str | Path) -> tuple[dict[str, object], ...]:
    """Read fixed-format `intersurf_initialize_2d` or `intersurf_main_2d` rows.

    Fortran provenance: consumes traces at `dim2_driver.f90` lines 1108-1125
    and 1293-1305, immediately before the calls into `intersurf.f90` lines
    113-152 and 458-500. These rows contain the gathered driver boundary
    values present in the trace file; they do not contain `zlev` or neighbours.
    """

    tokens = _trace_tokens(path)
    row_width = 17
    if len(tokens) % row_width != 0:
        raise ValueError(f"unexpected intersurf trace token count: {len(tokens)}")

    rows: list[dict[str, object]] = []
    for offset in range(0, len(tokens), row_width):
        chunk = tokens[offset : offset + row_width]
        rows.append(
            {
                "call_name": chunk[0],
                "tstep_fortran": _as_int(chunk[1]),
                "ik": _as_int(chunk[2]),
                "kindex": _as_int(chunk[3]),
                "lalo_lon": _as_float(chunk[4]),
                "lalo_lat": _as_float(chunk[5]),
                "contfrac": _as_float(chunk[6]),
                "temp_air": _as_float(chunk[7]),
                "pb": _as_float(chunk[8]),
                "qair": _as_float(chunk[9]),
                "u": _as_float(chunk[10]),
                "v": _as_float(chunk[11]),
                "precip_rain": _as_float(chunk[12]),
                "precip_snow": _as_float(chunk[13]),
                "swdown": _as_float(chunk[14]),
                "lwdown": _as_float(chunk[15]),
                "ccanopy": _as_float(chunk[16]),
            }
        )
    return tuple(rows)


def read_fixed_format_grid_trace(path: str | Path) -> tuple[dict[str, object], ...]:
    """Read fixed-format `grid_scatter` geometry rows from the server trace.

    Fortran provenance: consumes the trace after `grid.f90`, `grid_scatter`,
    lines 663-718. The row layout is the instrumented fixed-format payload:
    `ik`, final `area`, an unused placeholder value, resolution, neighbours,
    segment lengths, and corners. Values absent from the trace, such as `lalo`
    and `contfrac`, are not reconstructed here.
    """

    tokens = _trace_tokens(path)
    row_width = 26
    if len(tokens) % row_width != 0:
        raise ValueError(f"unexpected grid trace token count: {len(tokens)}")

    rows: list[dict[str, object]] = []
    for offset in range(0, len(tokens), row_width):
        chunk = tokens[offset : offset + row_width]
        if chunk[0] != "grid_scatter":
            raise ValueError(f"unexpected grid trace record label: {chunk[0]}")
        row: dict[str, object] = {
            "record_name": chunk[0],
            "ik": _as_int(chunk[1]),
            "area": _as_float(chunk[2]),
            "contfrac": _as_float(chunk[3]),
            "resolution_x": _as_float(chunk[4]),
            "resolution_y": _as_float(chunk[5]),
        }
        row.update({f"neighbour_{idx}": _as_int(chunk[5 + idx]) for idx in range(1, 9)})
        row.update({f"seglength_{idx}": _as_float(chunk[13 + idx]) for idx in range(1, 5)})
        corner_tokens = chunk[18:26]
        for idx in range(4):
            row[f"corner_{idx + 1}_lon"] = _as_float(corner_tokens[idx * 2])
            row[f"corner_{idx + 1}_lat"] = _as_float(corner_tokens[idx * 2 + 1])
        rows.append(row)
    return tuple(rows)


def read_fixed_format_soil_trace(path: str | Path) -> tuple[dict[str, object], ...]:
    """Read fixed-format `slowproc_soilt` overlap/final-field rows.

    Fortran provenance: consumes the trace inserted immediately before
    `slowproc.f90`, `slowproc_soilt`, lines 4897-4905 deallocate overlap
    arrays. The patch writes each positive `sub_area` row plus final
    `soilclass(ib,1:3)`, `clayfraction`, `sandfraction`, `siltfraction`,
    `bulk_density`, `soil_ph`, and `poor_soils`. For USDA, the trace's
    explicit overlap rows and `soiltext` values are sufficient to reconstruct
    all 12 soilclass fractions using `slowproc_soilt` lines 4853-4889.
    """

    fields = (
        "ik",
        "overlap_index",
        "soil_classif",
        "source_i",
        "source_j",
        "sub_area",
        "soiltext",
        "bulk_dens_source",
        "soil_ph_source",
        "poor_soils_source",
        "soilclass_1",
        "soilclass_2",
        "soilclass_3",
        "clay_frac",
        "sand_frac",
        "silt_frac",
        "bulk_dens",
        "soil_ph",
        "poor_soils",
    )
    rows: list[dict[str, object]] = []
    for record in read_records(path, tags="soil_overlap"):
        if len(record.values) != len(fields):
            raise ValueError(
                f"soil_overlap record at {record.path}:{record.start_line} has "
                f"{len(record.values)} values, expected {len(fields)}"
            )
        row = dict(zip(fields, record.values, strict=True))
        row["record_name"] = record.tag
        rows.append(row)
    return tuple(rows)


def read_fixed_format_bbox_trace(path: str | Path, *, group_name: str) -> tuple[dict[str, object], ...]:
    """Read salinity or tide fixed-format bbox/final-value traces.

    Fortran provenance: salinity rows target `slowproc.f90`,
    `slowproc_read_annual`, lines 6300-6570; tide rows target
    `slowproc_read_data`, lines 6029-6297. The fixed-format trace stores the
    final value once per annual field or per tide time index. Missing nearest
    source coordinates are kept at zero only when the traced fallback flag is
    false.
    """

    if group_name not in {"salinity_bbox", "tide_bbox"}:
        raise ValueError("group_name must be 'salinity_bbox' or 'tide_bbox'")
    tokens = _trace_tokens(path)
    row_width = 10 if group_name == "salinity_bbox" else 11
    if len(tokens) % row_width != 0:
        raise ValueError(f"unexpected {group_name} trace token count: {len(tokens)}")

    rows: list[dict[str, object]] = []
    for offset in range(0, len(tokens), row_width):
        chunk = tokens[offset : offset + row_width]
        field_name = chunk[0]
        if group_name == "salinity_bbox":
            ik = _as_int(chunk[1])
            time_index = 1
            value_offset = 2
        else:
            ik = _as_int(chunk[1])
            time_index = _as_int(chunk[2])
            value_offset = 3
        fallback = chunk[value_offset + 5]
        final_value = _as_float(chunk[value_offset + 7])
        rows.append(
            {
                "field_name": field_name,
                "ik": ik,
                "time_index": time_index,
                "lon_low": _as_float(chunk[value_offset]),
                "lon_up": _as_float(chunk[value_offset + 1]),
                "lat_low": _as_float(chunk[value_offset + 2]),
                "lat_up": _as_float(chunk[value_offset + 3]),
                "source_count": _as_int(chunk[value_offset + 4]),
                "fallback_nearest": fallback,
                "nearest_source_i": _as_int(chunk[value_offset + 6]),
                "nearest_source_j": 0,
                "mean_value": final_value,
                "final_value": final_value,
            }
        )
    return tuple(rows)


def read_static_fields_from_fixed_format_trace_dir(trace_dir: str | Path) -> StaticTraceFields:
    """Package soil, salinity, and tide static truth from 2026-06-23 traces.

    Fortran provenance: this is a convenience wrapper over fixed-format
    `slowproc_soilt`, `slowproc_read_annual`, and `slowproc_read_data` rows
    from `slowproc.f90` lines 2494-2534, 4284-4925, 6029-6297, and
    6300-6570. It closes only fields present in those traces and leaves
    vegetation fields blocked.
    """

    base = Path(trace_dir)
    soil_rows = read_fixed_format_soil_trace(base / SERVER_1961_TRACE_FILENAMES["soil"])
    salinity_rows = read_fixed_format_bbox_trace(
        base / SERVER_1961_TRACE_FILENAMES["salinity"],
        group_name="salinity_bbox",
    )
    tide_rows = read_fixed_format_bbox_trace(
        base / SERVER_1961_TRACE_FILENAMES["tide"],
        group_name="tide_bbox",
    )
    bbox_fields = static_fields_from_trace(salinity_rows=salinity_rows, tide_rows=tide_rows)
    soil_fields = static_soil_fields_from_fixed_format_trace(soil_rows)
    return StaticTraceFields(
        soilclass=soil_fields.soilclass,
        njsc=soil_fields.njsc,
        clay_frac=soil_fields.clay_frac,
        sand_frac=soil_fields.sand_frac,
        silt_frac=soil_fields.silt_frac,
        bulk_dens=soil_fields.bulk_dens,
        soil_ph=soil_fields.soil_ph,
        poor_soils=soil_fields.poor_soils,
        soilclass_sub_index=soil_fields.soilclass_sub_index,
        soilclass_sub_area=soil_fields.soilclass_sub_area,
        salinity=bbox_fields.salinity,
        tide_height=bbox_fields.tide_height,
        blocked_fields=tuple(soil_fields.blocked_fields + bbox_fields.blocked_fields),
    )


def apply_grid_geometry_trace_to_domain(domain: DomainGrid, rows: Iterable[Mapping[str, object]]) -> DomainGrid:
    """Return a DomainGrid copy populated from explicit grid-geometry trace.

    Fortran provenance: consumes trace columns from `grid.f90`, `grid_stuff`
    lines 416-470, `grid_topolylist` lines 490-631, `grid_scatter` lines
    663-718, and `haversine.f90` lines 394-488, 667-736, 863-888. The adapter
    copies traced `resolution`, `neighbours`, and `area`; it does not derive
    them from lon/lat or history diagnostics.
    """

    materialized = tuple(coerce_trace_row_values(row) for row in rows)
    validation = validate_trace_rows("grid_geometry_after_grid_stuff", materialized)
    if validation.unknown_group or validation.missing_columns:
        raise ValueError(f"invalid grid geometry trace columns: {validation.missing_columns}")
    if len(materialized) != domain.nbindex:
        raise ValueError("grid geometry trace row count must match domain.nbindex")

    resolution = np.asarray(
        [[float(row["resolution_x"]), float(row["resolution_y"])] for row in materialized],
        dtype=np.float64,
    )
    neighbours = np.asarray(
        [[int(row[f"neighbour_{idx}"]) for idx in range(1, 9)] for row in materialized],
        dtype=np.int32,
    )
    area = np.asarray([float(row["area"]) for row in materialized], dtype=np.float64)
    corners = np.asarray(
        [
            [[float(row[f"corner_{idx}_lon"]), float(row[f"corner_{idx}_lat"])] for idx in range(1, 5)]
            for row in materialized
        ],
        dtype=np.float64,
    )
    seglength = np.asarray(
        [[float(row[f"seglength_{idx}"]) for idx in range(1, 5)] for row in materialized],
        dtype=np.float64,
    )
    return replace(domain, resolution=resolution, neighbours=neighbours, area=area, corners=corners, seglength=seglength)


def apply_fixed_format_grid_trace_to_domain(domain: DomainGrid, path: str | Path) -> DomainGrid:
    """Populate domain geometry fields from explicit fixed-format grid trace.

    Fortran provenance: consumes `orchjax_grid_trace.txt` rows written after
    `grid.f90`, `grid_scatter`, lines 688-707 scatter neighbours, segment
    lengths, corners, area, and resolution. This adapter copies only fields
    represented by `DomainGrid` (`resolution`, `neighbours`, and final grid
    `area`) and does not derive geometry from forcing coordinates.
    """

    rows = read_fixed_format_grid_trace(path)
    if len(rows) != domain.nbindex:
        raise ValueError("fixed-format grid trace row count must match domain.nbindex")
    ordered = sorted(rows, key=lambda row: int(row["ik"]))
    resolution = np.asarray(
        [[float(row["resolution_x"]), float(row["resolution_y"])] for row in ordered],
        dtype=np.float64,
    )
    neighbours = np.asarray(
        [[int(row[f"neighbour_{idx}"]) for idx in range(1, 9)] for row in ordered],
        dtype=np.int32,
    )
    area = np.asarray([float(row["area"]) for row in ordered], dtype=np.float64)
    corners = np.asarray(
        [
            [[float(row[f"corner_{idx}_lon"]), float(row[f"corner_{idx}_lat"])] for idx in range(1, 5)]
            for row in ordered
        ],
        dtype=np.float64,
    )
    seglength = np.asarray([[float(row[f"seglength_{idx}"]) for idx in range(1, 5)] for row in ordered], dtype=np.float64)
    return replace(domain, resolution=resolution, neighbours=neighbours, area=area, corners=corners, seglength=seglength)


def static_soil_fields_from_fixed_format_trace(rows: Iterable[Mapping[str, object]]) -> StaticTraceFields:
    """Build explicit final soil fields from fixed-format `soil_overlap` rows.

    Fortran provenance: final scalar fields are copied directly from the trace
    written after `slowproc.f90`, `slowproc_soilt`, lines 4853-4889. For USDA
    soilclass, the trace contains every positive `sub_area`/`soiltext` overlap
    row; this function mirrors the Fortran accumulation and normalization in
    those lines, then derives `njsc` with `MAXLOC` as in lines 2007-2010 and
    2190-2192.
    """

    materialized = tuple(coerce_trace_row_values(row) for row in rows)
    if not materialized:
        return StaticTraceFields(
            blocked_fields=("soilclass", "njsc", "clay_frac", "sand_frac", "silt_frac", "bulk_dens", "soil_ph", "poor_soils")
        )

    by_ik: dict[int, list[dict[str, object]]] = {}
    for row in materialized:
        by_ik.setdefault(int(row["ik"]), []).append(row)

    soilclasses: list[np.ndarray] = []
    overlap_indices: list[list[tuple[int, int]]] = []
    overlap_areas: list[list[float]] = []
    njsc: list[int] = []
    clay: list[float] = []
    sand: list[float] = []
    silt: list[float] = []
    bulk: list[float] = []
    ph: list[float] = []
    poor: list[float] = []
    blocked: list[str] = []

    for ik in sorted(by_ik):
        group = sorted(by_ik[ik], key=lambda row: int(row["overlap_index"]))
        first = group[0]
        point_indices: list[tuple[int, int]] = []
        point_areas: list[float] = []
        classification = str(first["soil_classif"]).strip().lower()
        if classification == "usda":
            soilclass = np.zeros(12, dtype=np.float64)
            total_area = 0.0
            for row in group:
                area = float(row["sub_area"])
                if area <= 0.0:
                    continue
                point_indices.append((int(row["source_i"]), int(row["source_j"])))
                point_areas.append(area)
                source_class = int(round(float(row["soiltext"])))
                if source_class < 1 or source_class > 12:
                    raise ValueError(f"bad USDA soiltext class {source_class}")
                soilclass[source_class - 1] += area
                total_area += area
            if total_area <= 0.0:
                blocked.extend(["soilclass", "njsc"])
                continue
            soilclass = soilclass / total_area
            traced_first3 = np.asarray([float(first[f"soilclass_{idx}"]) for idx in range(1, 4)], dtype=np.float64)
            if not np.allclose(soilclass[:3], traced_first3):
                raise ValueError("reconstructed USDA soilclass disagrees with traced first three classes")
        elif classification in {"zobler", "fao"}:
            soilclass = np.asarray([float(first[f"soilclass_{idx}"]) for idx in range(1, 4)], dtype=np.float64)
        else:
            blocked.extend(["soilclass", "njsc"])
            continue

        soilclasses.append(soilclass)
        overlap_indices.append(point_indices)
        overlap_areas.append(point_areas)
        njsc.append(int(np.argmax(soilclass)) + 1)
        clay.append(float(first["clay_frac"]))
        sand.append(float(first["sand_frac"]))
        silt.append(float(first["silt_frac"]))
        bulk.append(float(first["bulk_dens"]))
        ph.append(float(first["soil_ph"]))
        poor.append(float(first["poor_soils"]))

    if not soilclasses:
        return StaticTraceFields(
            blocked_fields=tuple(blocked)
            or ("soilclass", "njsc", "clay_frac", "sand_frac", "silt_frac", "bulk_dens", "soil_ph", "poor_soils")
        )

    width = max((len(item) for item in overlap_indices), default=0)
    sub_index = np.zeros((len(overlap_indices), width, 2), dtype=np.int32)
    sub_area = np.zeros((len(overlap_areas), width), dtype=np.float64)
    for ib, (indices, areas) in enumerate(zip(overlap_indices, overlap_areas, strict=True)):
        for idx, ((source_i, source_j), area) in enumerate(zip(indices, areas, strict=True)):
            sub_index[ib, idx] = (source_i, source_j)
            sub_area[ib, idx] = area

    return StaticTraceFields(
        soilclass=np.asarray(soilclasses, dtype=np.float64),
        njsc=np.asarray(njsc, dtype=np.int32),
        clay_frac=np.asarray(clay, dtype=np.float64),
        sand_frac=np.asarray(sand, dtype=np.float64),
        silt_frac=np.asarray(silt, dtype=np.float64),
        bulk_dens=np.asarray(bulk, dtype=np.float64),
        soil_ph=np.asarray(ph, dtype=np.float64),
        poor_soils=np.asarray(poor, dtype=np.float64),
        soilclass_sub_index=sub_index,
        soilclass_sub_area=sub_area,
        blocked_fields=tuple(blocked),
    )


def _trace_tokens(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").split()


def _as_float(value: object) -> float:
    return float(str(value).replace("D", "E"))


def _as_int(value: object) -> int:
    return int(float(str(value).replace("D", "E")))


def static_fields_from_trace(
    *,
    soil_rows: Iterable[Mapping[str, object]] | None = None,
    salinity_rows: Iterable[Mapping[str, object]] | None = None,
    tide_rows: Iterable[Mapping[str, object]] | None = None,
) -> StaticTraceFields:
    """Package explicit final static fields from supplied trace rows.

    Fortran provenance: soil rows target `slowproc_soilt` lines 4284-4925 and
    `interpol_help.f90` lines 46-466/830-889; salinity rows target
    `slowproc_read_annual` lines 6300-6570; tide rows target
    `slowproc_read_data` lines 6029-6297. This adapter uses only final fields
    present in trace rows and reports blocked fields otherwise.
    """

    blocked: list[str] = []
    soilclass = njsc = clay_frac = sand_frac = silt_frac = bulk_dens = soil_ph = poor_soils = None
    soilclass_sub_index = soilclass_sub_area = None
    salinity = tide_height = None

    if soil_rows is None:
        blocked.extend(["soilclass", "njsc", "clay_frac", "sand_frac", "bulk_dens", "soil_ph", "poor_soils"])
    else:
        soil = tuple(coerce_trace_row_values(row) for row in soil_rows)
        validation = validate_trace_rows("soil_aggregation", soil)
        if not validation.is_valid:
            blocked.extend(["soilclass", "njsc", "clay_frac", "sand_frac", "bulk_dens", "soil_ph", "poor_soils"])
        else:
            soilclass = np.asarray(
                [[float(row["soilclass_1"]), float(row["soilclass_2"]), float(row["soilclass_3"])] for row in soil],
                dtype=np.float64,
            )
            njsc = np.asarray([int(row["njsc"]) for row in soil], dtype=np.int32)
            clay_frac = np.asarray([float(row["clay_frac"]) for row in soil], dtype=np.float64)
            sand_frac = np.asarray([float(row["sand_frac"]) for row in soil], dtype=np.float64)
            silt_frac = (
                np.asarray([float(row["silt_frac"]) for row in soil], dtype=np.float64)
                if "silt_frac" in soil[0]
                else None
            )
            bulk_dens = np.asarray([float(row["bulk_dens"]) for row in soil], dtype=np.float64)
            soil_ph = np.asarray([float(row["soil_ph"]) for row in soil], dtype=np.float64)
            poor_soils = np.asarray([float(row["poor_soils"]) for row in soil], dtype=np.float64)
            soilclass_sub_index = np.asarray(
                [[[int(row["source_i"]), int(row["source_j"])]] for row in soil],
                dtype=np.int32,
            )
            soilclass_sub_area = np.asarray([[float(row["sub_area"])] for row in soil], dtype=np.float64)

    if salinity_rows is None:
        blocked.append("salinity")
    else:
        salinity = _final_values_from_bbox_trace("salinity_bbox", salinity_rows)
        if salinity is None:
            blocked.append("salinity")

    if tide_rows is None:
        blocked.append("tide_height")
    else:
        tide_height = _final_values_from_bbox_trace("tide_bbox", tide_rows)
        if tide_height is None:
            blocked.append("tide_height")

    return StaticTraceFields(
        soilclass=soilclass,
        njsc=njsc,
        clay_frac=clay_frac,
        sand_frac=sand_frac,
        silt_frac=silt_frac,
        bulk_dens=bulk_dens,
        soil_ph=soil_ph,
        poor_soils=poor_soils,
        soilclass_sub_index=soilclass_sub_index,
        soilclass_sub_area=soilclass_sub_area,
        salinity=salinity,
        tide_height=tide_height,
        blocked_fields=tuple(blocked),
    )


def _final_values_from_bbox_trace(group_name: str, rows: Iterable[Mapping[str, object]]) -> np.ndarray | None:
    materialized = tuple(coerce_trace_row_values(row) for row in rows)
    validation = validate_trace_rows(group_name, materialized)
    if not validation.is_valid:
        return None
    by_ik: dict[int, list[tuple[int, float]]] = {}
    for row in materialized:
        ik = int(row["ik"])
        time_index = int(row["time_index"])
        by_ik.setdefault(ik, []).append((time_index, float(row["final_value"])))
    output: list[np.ndarray] = []
    for ik in sorted(by_ik):
        ordered = sorted(by_ik[ik], key=lambda item: item[0])
        output.append(np.asarray([value for _, value in ordered], dtype=np.float64))
    arr = np.asarray(output, dtype=np.float64)
    return arr[:, 0] if arr.shape[1] == 1 else arr
