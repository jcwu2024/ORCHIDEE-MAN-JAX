"""NetCDF dimension metadata owners for ``interpweight``.

The netCDF Fortran API reports dimension IDs in Fortran array order (the
fastest-varying dimension first).  Xarray exposes the file/CDL order, so that
order is reversed explicitly at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any, Mapping


INTERPWEIGHT_FILE_DIMS_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_get_var2dims_file lines 4081-4145",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_get_var3dims_file lines 4163-4229",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_get_var4dims_file lines 4247-4316",
)


class InterpolationFileMetadataError(ValueError):
    """Raised when NetCDF metadata cannot satisfy the Fortran contract."""


class InterpolationFileReadError(OSError):
    """Raised when a NetCDF file cannot be opened and inspected."""


@dataclass(frozen=True)
class CoordinateMetadata:
    """Metadata used to classify one variable dimension."""

    name: str
    dimensions: tuple[str, ...]
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class VariableDimensionMetadata:
    """A variable's file and netCDF-Fortran dimension contracts."""

    variable_name: str
    file_dimension_names: tuple[str, ...]
    file_dimension_sizes: tuple[int, ...]
    fortran_dimension_names: tuple[str, ...]
    fortran_dimension_sizes: tuple[int, ...]
    coordinates: tuple[CoordinateMetadata, ...]
    time_dimension: str | None
    time_fortran_axis: int | None
    level_dimension: str | None
    level_fortran_axis: int | None


_TIME_NAMES = frozenset({"t", "time", "times", "time_counter"})
_LEVEL_NAMES = frozenset(
    {
        "z",
        "lev",
        "level",
        "levels",
        "depth",
        "deptht",
        "altitude",
        "height",
        "soil_layer",
        "soil_layers",
    }
)
_VERTICAL_STANDARD_NAMES = frozenset(
    {
        "altitude",
        "depth",
        "height",
        "air_pressure",
        "model_level_number",
        "atmosphere_hybrid_sigma_pressure_coordinate",
        "ocean_sigma_coordinate",
    }
)


def _as_attributes(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InterpolationFileMetadataError(f"{context} attributes must be a mapping")
    return {str(key): item for key, item in value.items()}


def _as_dimensions(value: Any, *, context: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise InterpolationFileMetadataError(
            f"{context} dimensions must be a sequence of names"
        )
    dimensions = tuple(str(name) for name in value)
    if any(not name.strip() for name in dimensions):
        raise InterpolationFileMetadataError(f"{context} has an empty dimension name")
    if len(set(dimensions)) != len(dimensions):
        raise InterpolationFileMetadataError(f"{context} repeats a dimension name")
    return dimensions


def _as_dimension_sizes(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise InterpolationFileMetadataError(
            "metadata 'dimensions' must be a name-to-size mapping"
        )
    sizes: dict[str, int] = {}
    for raw_name, raw_size in value.items():
        name = str(raw_name)
        if not name.strip():
            raise InterpolationFileMetadataError(
                "metadata contains an empty dimension name"
            )
        if (
            isinstance(raw_size, bool)
            or not isinstance(raw_size, Integral)
            or raw_size < 0
        ):
            raise InterpolationFileMetadataError(
                f"dimension {name!r} size must be a non-negative integer"
            )
        sizes[name] = int(raw_size)
    return sizes


def _coordinate_kind(coordinate: CoordinateMetadata) -> str | None:
    attrs = {
        str(key).lower(): str(value).strip().lower()
        for key, value in coordinate.attributes.items()
    }
    name = coordinate.name.strip().lower()
    axis = attrs.get("axis", "")
    standard_name = attrs.get("standard_name", "")
    units = attrs.get("units", "")
    positive = attrs.get("positive", "")

    is_time = (
        axis == "t"
        or standard_name == "time"
        or " since " in f" {units} "
        or name in _TIME_NAMES
    )
    is_level = (
        axis == "z"
        or standard_name in _VERTICAL_STANDARD_NAMES
        or positive in {"up", "down"}
        or name in _LEVEL_NAMES
    )
    if is_time and is_level:
        raise InterpolationFileMetadataError(
            f"coordinate {coordinate.name!r} is classified as both time and level"
        )
    if is_time:
        return "time"
    if is_level:
        return "level"
    return None


def _metadata_from_mapping(
    metadata: Mapping[str, Any], varname: str
) -> VariableDimensionMetadata:
    sizes = _as_dimension_sizes(metadata.get("dimensions"))
    variables = metadata.get("variables")
    if not isinstance(variables, Mapping):
        raise InterpolationFileMetadataError("metadata 'variables' must be a mapping")
    if varname not in variables:
        raise InterpolationFileMetadataError(f"variable {varname!r} was not found")
    variable = variables[varname]
    if not isinstance(variable, Mapping):
        raise InterpolationFileMetadataError(
            f"variable {varname!r} metadata must be a mapping"
        )
    dimensions = _as_dimensions(
        variable.get("dimensions"), context=f"variable {varname!r}"
    )
    missing = [name for name in dimensions if name not in sizes]
    if missing:
        raise InterpolationFileMetadataError(
            f"variable {varname!r} references dimensions with no size: {missing}"
        )

    variable_attrs = _as_attributes(
        variable.get("attributes", variable.get("attrs")),
        context=f"variable {varname!r}",
    )
    coordinate_names = list(dimensions)
    auxiliary = variable_attrs.get("coordinates", "")
    if auxiliary:
        if not isinstance(auxiliary, str):
            raise InterpolationFileMetadataError(
                f"variable {varname!r} coordinates attribute must be a string"
            )
        coordinate_names.extend(auxiliary.split())

    coordinates: list[CoordinateMetadata] = []
    for name in dict.fromkeys(coordinate_names):
        raw = variables.get(name)
        if raw is None:
            if name in dimensions:
                coordinates.append(CoordinateMetadata(name, (name,), {}))
            continue
        if not isinstance(raw, Mapping):
            raise InterpolationFileMetadataError(
                f"coordinate {name!r} metadata must be a mapping"
            )
        coordinate_dimensions = _as_dimensions(
            raw.get("dimensions", ()), context=f"coordinate {name!r}"
        )
        coordinate_attrs = _as_attributes(
            raw.get("attributes", raw.get("attrs")), context=f"coordinate {name!r}"
        )
        coordinates.append(
            CoordinateMetadata(name, coordinate_dimensions, coordinate_attrs)
        )

    return _build_result(varname, dimensions, sizes, tuple(coordinates))


def _build_result(
    varname: str,
    file_dimensions: tuple[str, ...],
    sizes: Mapping[str, int],
    coordinates: tuple[CoordinateMetadata, ...],
) -> VariableDimensionMetadata:
    kinds: dict[str, list[str]] = {"time": [], "level": []}
    variable_dimensions = set(file_dimensions)
    for coordinate in coordinates:
        kind = _coordinate_kind(coordinate)
        if kind is None:
            continue
        attached = [
            name for name in coordinate.dimensions if name in variable_dimensions
        ]
        if coordinate.name in variable_dimensions:
            attached.append(coordinate.name)
        attached = list(dict.fromkeys(attached))
        if len(attached) != 1:
            raise InterpolationFileMetadataError(
                f"{kind} coordinate {coordinate.name!r} must identify exactly one variable dimension"
            )
        kinds[kind].append(attached[0])

    identified: dict[str, str | None] = {}
    for kind, matches in kinds.items():
        unique = tuple(dict.fromkeys(matches))
        if len(unique) > 1:
            raise InterpolationFileMetadataError(
                f"variable {varname!r} has multiple {kind} dimensions: {unique}"
            )
        identified[kind] = unique[0] if unique else None

    file_sizes = tuple(sizes[name] for name in file_dimensions)
    fortran_dimensions = tuple(reversed(file_dimensions))
    time_name = identified["time"]
    level_name = identified["level"]
    return VariableDimensionMetadata(
        variable_name=varname,
        file_dimension_names=file_dimensions,
        file_dimension_sizes=file_sizes,
        fortran_dimension_names=fortran_dimensions,
        fortran_dimension_sizes=tuple(reversed(file_sizes)),
        coordinates=coordinates,
        time_dimension=time_name,
        time_fortran_axis=(
            fortran_dimensions.index(time_name) + 1 if time_name else None
        ),
        level_dimension=level_name,
        level_fortran_axis=(
            fortran_dimensions.index(level_name) + 1 if level_name else None
        ),
    )


def read_variable_dimension_metadata(
    source: str | PathLike[str] | Mapping[str, Any], varname: str
) -> VariableDimensionMetadata:
    """Inspect one variable without reading its data values.

    Mapping input uses ``dimensions: {name: size}`` and ``variables`` entries
    with ``dimensions`` plus optional ``attributes``.  Path input is converted
    to that same metadata contract through xarray with CF decoding disabled.
    """

    if not isinstance(varname, str) or not varname.strip():
        raise InterpolationFileMetadataError("varname must be a non-empty string")
    if isinstance(source, Mapping):
        return _metadata_from_mapping(source, varname)

    path = Path(source)
    try:
        import xarray as xr

        with xr.open_dataset(path, decode_cf=False, cache=False) as dataset:
            metadata = {
                "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
                "variables": {
                    name: {
                        "dimensions": tuple(variable.dims),
                        "attributes": dict(variable.attrs),
                    }
                    for name, variable in dataset.variables.items()
                },
            }
            return _metadata_from_mapping(metadata, varname)
    except InterpolationFileMetadataError:
        raise
    except Exception as exc:
        raise InterpolationFileReadError(
            f"could not read NetCDF metadata from {str(path)!r}: {exc}"
        ) from exc


def _get_dims(
    source: str | PathLike[str] | Mapping[str, Any], varname: str, expected_rank: int
) -> tuple[int, ...]:
    metadata = read_variable_dimension_metadata(source, varname)
    actual_rank = len(metadata.fortran_dimension_sizes)
    if actual_rank != expected_rank:
        raise InterpolationFileMetadataError(
            f"variable {varname!r} has {actual_rank} dimensions, expected {expected_rank}"
        )
    return metadata.fortran_dimension_sizes


def interpweight_get_var2dims_file(
    source: str | PathLike[str] | Mapping[str, Any], varname: str
) -> tuple[int, int]:
    """Port of ``interpweight_get_var2dims_file`` lines 4081-4145."""

    result = _get_dims(source, varname, 2)
    return result[0], result[1]


def interpweight_get_var3dims_file(
    source: str | PathLike[str] | Mapping[str, Any], varname: str
) -> tuple[int, int, int]:
    """Port of ``interpweight_get_var3dims_file`` lines 4163-4229."""

    result = _get_dims(source, varname, 3)
    return result[0], result[1], result[2]


def interpweight_get_var4dims_file(
    source: str | PathLike[str] | Mapping[str, Any], varname: str
) -> tuple[int, int, int, int]:
    """Port of ``interpweight_get_var4dims_file`` lines 4247-4316."""

    result = _get_dims(source, varname, 4)
    return result[0], result[1], result[2], result[3]
