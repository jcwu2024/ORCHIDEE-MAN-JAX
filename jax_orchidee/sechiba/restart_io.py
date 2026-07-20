"""Complete SECHIBA restart state normalization and standalone serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset

from jax_orchidee.stomate.restart_io import (
    StomateRestartPhysicalState,
    StomateRestartWriteReport,
    create_stomate_restart_skeleton_from_schema,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECHIBA_RESTART_SCHEMA = (
    ROOT / "docs" / "source_audits" / "sechiba_restart_netcdf_schema.json"
)
PHYSICAL_RESTART_VARIABLES = frozenset(
    {"nav_lon", "nav_lat", "nav_lev", "time", "time_steps"}
)

SECHIBA_RESTART_COMPONENT_FIELDS = {
    "diffuco": frozenset({"rstruct", "cdrag_pft", "leaf_ci"}),
    "enerbil": frozenset(
        {
            "temp_sol", "temp_sol_pft", "qsurf", "evapot", "evapot_corr",
            "tsolrad", "evapora", "fluxlat", "fluxsens", "tempsolpot",
            "qsolpot",
        }
    ),
    "hydrol": frozenset(
        {
            "moistc", "moistcl", "us", "free_drain_coef", "zwt_force",
            "water2infilt", "ae_ns", "vegstress", "snow", "snow_age",
            "snow_nobio", "snow_nobio_age", "qsintveg", "evap_bare_lim_ns",
            "evap_bare_lim", "resdist", "vegtot_old", "drysoil_frac",
            "humrel", "fwet_out", "run2peat", "wt_ab", "wtp", "fwet_new",
            "liqwt_ratio", "wt_ab_tide", "run2man", "tot_watveg_beg",
            "tot_watsoil_beg", "snow_beg",
        }
    ),
    "explicit_snow": frozenset(
        {"snowrho", "snowtemp", "snowdz", "snowheat", "snowgrain"}
    ),
    "condveg": frozenset(
        {"z0m", "z0h", "roughheight", "roughheight_pft", "soilalbedo_bg"}
    ),
    "thermosoil": frozenset(
        {
            "ptn", "refSOC", "shum_ngrnd_prmlng", "shum_ngrnd_perma",
            "e_soil_lat", "cgrnd", "dgrnd", "gtemp", "soilcap",
            "soilcap_pft", "soilflx", "soilflx_pft", "cgrnd_snow",
            "dgrnd_snow", "lambda_snow",
        }
    ),
    "slowproc": frozenset(
        {
            "veget", "veget_max", "lai", "frac_nobio", "frac_age", "njsc",
            "reinf_slope", "clay_frac", "sand_frac", "height", "veget_year",
            "peatPET_last", "growth_day", "GSL", "peatPET_this",
            "precipitation_last", "precipitation_this", "summerpet_longterm",
            "summerp_longterm", "peatC", "peatC_ok", "soil_ph", "poor_soils",
            "bulk_dens",
        }
    ),
}

SLOWPROC_FINALIZE_ALIASES = {
    "clayfraction": "clay_frac",
    "sandfraction": "sand_frac",
    "peatPET_lastyear": "peatPET_last",
    "peatPET_thisyear": "peatPET_this",
    "precipitation_lastsummer": "precipitation_last",
    "precipitation_thissummer": "precipitation_this",
    "summerpet_long": "summerpet_longterm",
    "summerp_long": "summerp_longterm",
    "bulk_density": "bulk_dens",
}

SECHIBA_RESTART_TO_SOURCE_NAMES = {
    "cdrag_pft": "q_cdrag_pft",
    "tsolrad": "tsol_rad",
    "evapora": "vevapp",
    "tempsolpot": "temp_sol_pot",
    "qsolpot": "q_sol_pot",
    "moistc": "mc",
    "moistcl": "mcl",
    "soilalbedo_bg": "soilalb_bg",
    "refSOC": "refsoc",
    "shum_ngrnd_prmlng": "shum_ngrnd_permalong",
    **{restart_name: source_name for source_name, restart_name in SLOWPROC_FINALIZE_ALIASES.items()},
}

SECHIBA_COMPONENT_SOURCE_FILES = {
    "diffuco": "diffuco.f90",
    "enerbil": "enerbil.f90",
    "hydrol": "hydrol.f90",
    "explicit_snow": "explicitsnow.f90",
    "condveg": "condveg.f90",
    "thermosoil": "thermosoil.f90",
    "slowproc": "slowproc.f90",
}


@dataclass(frozen=True)
class SechibaRestartState:
    """All scientific variables in one SECHIBA restart file.

    Arrays use ``(npts, *Fortran_axes)``. IOIPSL stores the Fortran axes in
    reverse order before the physical ``y,x`` dimensions, so normalization and
    serialization are exact inverse transposes.
    """

    fields: Mapping[str, np.ndarray]
    provenance_by_field: Mapping[str, tuple[str, ...]]


def sechiba_finalize_source_state_from_restart(
    state: SechibaRestartState,
) -> dict[str, np.ndarray]:
    """Translate serialized restart names back to component-finalize names.

    This is a name normalization only. It neither fills missing fields nor
    changes values. The resulting mapping is the exact input contract of
    :func:`paper_sechiba_restart_state_from_finalize_state`.

    Fortran provenance: ``sechiba.f90::sechiba_initialize`` lines 561-679 and
    ``sechiba.f90::sechiba_finalize`` lines 1800-1923 route the same module
    state through component initialize/finalize calls.
    """

    if not isinstance(state, SechibaRestartState):
        raise TypeError("state must be a SechibaRestartState")
    source = {
        SECHIBA_RESTART_TO_SOURCE_NAMES.get(name, name): np.asarray(value)
        for name, value in state.fields.items()
    }
    expected = {
        SECHIBA_RESTART_TO_SOURCE_NAMES.get(name, name)
        for fields in SECHIBA_RESTART_COMPONENT_FIELDS.values()
        for name in fields
    }
    actual = set(source)
    if actual != expected:
        raise ValueError(
            "SECHIBA finalize source-state mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if len(source) != 93:
        raise ValueError(f"SECHIBA finalize source-state denominator must be 93, got {len(source)}")
    return source


def sechiba_restart_state_from_finalize_packets(
    *,
    diffuco: Mapping[str, object],
    enerbil: Mapping[str, object],
    hydrol: Mapping[str, object],
    explicit_snow: Mapping[str, object],
    condveg: Mapping[str, object],
    thermosoil: Mapping[str, object],
    slowproc: Mapping[str, object],
) -> SechibaRestartState:
    """Assemble the exact seven-owner ``sechiba_finalize`` restart state."""

    packets = {
        "diffuco": dict(diffuco),
        "enerbil": dict(enerbil),
        "hydrol": dict(hydrol),
        "explicit_snow": dict(explicit_snow),
        "condveg": dict(condveg),
        "thermosoil": dict(thermosoil),
        "slowproc": {
            SLOWPROC_FINALIZE_ALIASES.get(name, name): value
            for name, value in slowproc.items()
        },
    }
    fields: dict[str, np.ndarray] = {}
    provenance: dict[str, tuple[str, ...]] = {}
    for component, packet in packets.items():
        expected = SECHIBA_RESTART_COMPONENT_FIELDS[component]
        actual = set(packet)
        if actual != expected:
            raise ValueError(
                f"{component} finalize packet mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        overlap = set(fields) & actual
        if overlap:
            raise ValueError(f"SECHIBA finalize owners overlap: {sorted(overlap)}")
        for name, value in packet.items():
            fields[name] = np.asarray(value)
            provenance[name] = (
                "fortran_source/ORCHIDEE/src_sechiba/"
                f"{SECHIBA_COMPONENT_SOURCE_FILES[component]}::{component}_finalize",
            )
    if len(fields) != 93:
        raise ValueError(f"SECHIBA finalize denominator must be 93, got {len(fields)}")
    return SechibaRestartState(fields, provenance)


def _science_variable_names(dataset: Dataset) -> tuple[str, ...]:
    return tuple(name for name in dataset.variables if name not in PHYSICAL_RESTART_VARIABLES)


def _read_science_variable(variable: object) -> np.ndarray:
    dimensions = tuple(variable.dimensions)
    values = np.asarray(variable[:])
    if dimensions and dimensions[0] == "time":
        dimensions = dimensions[1:]
        values = values[0]
    if "y" not in dimensions or "x" not in dimensions:
        return values.reshape(-1)
    non_spatial = tuple(name for name in dimensions if name not in {"y", "x"})
    permutation = (
        dimensions.index("y"),
        dimensions.index("x"),
        *(dimensions.index(name) for name in reversed(non_spatial)),
    )
    ordered = np.transpose(values, permutation)
    return ordered.reshape((-1, *ordered.shape[2:]))


def _write_science_variable(variable: object, field: str, value: object) -> None:
    dimensions = tuple(variable.dimensions)
    target_dimensions = dimensions[1:] if dimensions and dimensions[0] == "time" else dimensions
    target_shape = tuple(variable.shape[1:] if dimensions and dimensions[0] == "time" else variable.shape)
    array = np.asarray(value)
    if "y" not in target_dimensions or "x" not in target_dimensions:
        if array.size != int(np.prod(target_shape)):
            raise ValueError(
                f"{field} normalized size must be {int(np.prod(target_shape))}, got {array.size}"
            )
        stored = array.reshape(target_shape)
    else:
        non_spatial = tuple(
            name for name in target_dimensions if name not in {"y", "x"}
        )
        normalized_shape = (
            int(variable.shape[target_dimensions.index("y") + 1])
            * int(variable.shape[target_dimensions.index("x") + 1]),
            *(int(variable.shape[target_dimensions.index(name) + 1]) for name in reversed(non_spatial)),
        )
        if array.shape != normalized_shape:
            raise ValueError(
                f"{field} normalized shape must be {normalized_shape}, got {array.shape}"
            )
        ny = int(variable.shape[target_dimensions.index("y") + 1])
        nx = int(variable.shape[target_dimensions.index("x") + 1])
        expanded = array.reshape((ny, nx, *normalized_shape[1:]))
        normalized_names = ("y", "x", *reversed(non_spatial))
        permutation = tuple(normalized_names.index(name) for name in target_dimensions)
        stored = np.transpose(expanded, permutation)
    if dimensions and dimensions[0] == "time":
        variable[0] = stored
    else:
        variable[:] = stored


def read_sechiba_restart_state(path: str | Path) -> SechibaRestartState:
    """Read all SECHIBA component-finalize state without defaults.

    Fortran provenance: ``sechiba.f90::sechiba_finalize`` lines 1800-1923 and
    component ``*_finalize`` restput calls routed by that subroutine.
    """

    with Dataset(path) as dataset:
        names = _science_variable_names(dataset)
        fields = {name: _read_science_variable(dataset.variables[name]) for name in names}
    provenance = {
        name: (
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_finalize lines 1800-1923",
            f"audited NetCDF variable {name} from component finalize restput_p",
        )
        for name in fields
    }
    return SechibaRestartState(fields, provenance)


def write_sechiba_restart_state(
    output_path: str | Path,
    *,
    state: SechibaRestartState,
    physical_state: StomateRestartPhysicalState,
    schema_path: str | Path = DEFAULT_SECHIBA_RESTART_SCHEMA,
) -> StomateRestartWriteReport:
    """Construct and populate every scientific SECHIBA restart variable."""

    if not isinstance(state, SechibaRestartState):
        raise TypeError("state must be SechibaRestartState")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    expected = set(schema["variables"]) - PHYSICAL_RESTART_VARIABLES
    actual = set(state.fields)
    if actual != expected:
        raise ValueError(
            "SECHIBA restart field coverage mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    output = create_stomate_restart_skeleton_from_schema(
        output_path,
        physical_state,
        schema_path=schema_path,
    )
    with Dataset(output, "r+") as dataset:
        for name in _science_variable_names(dataset):
            _write_science_variable(dataset.variables[name], name, state.fields[name])
    return StomateRestartWriteReport(
        output_path=output,
        written_fields=tuple(state.fields),
        validated_derived_fields=(),
        unsupported_fields=(),
    )
