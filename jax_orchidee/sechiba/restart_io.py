"""Complete SECHIBA restart state normalization and standalone serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset

from jax_orchidee.driver.restart import (
    SECHIBA_RESTART_PFT_AXES,
    remap_restart_fields_by_pft_id,
)
from jax_orchidee.parameters.pft_catalog import PFTRunLayout
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
SECHIBA_NORMALIZED_RESTART_PFT_AXES = {
    **SECHIBA_RESTART_PFT_AXES,
    "frac_age": 1,
}

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


def restore_sechiba_integer_source_fields(
    source: Mapping[str, object], *, njsc_class_count: int = 12,
) -> dict[str, np.ndarray]:
    """Restore INTEGER owners from the real-valued restart representation.

    Fortran slowproc.f90:122,210,1232,1251-1254,1655-1664 declares
    INTEGER owners and stores njsc as REAL. This supported codec accepts only
    exactly integral, finite values; it never rounds corrupt fractional input.
    njsc ranges from 1 to active nscm (USDA12/FAO3, control.f90:83-87;
    constantes_soil_var.f90:70-77). veget_year has no calendar range here:
    slowproc_init:1603-1644 leaves it undefined when IMPOSE_VEG is active.
    Both public source owners use a signed 32-bit integer representation.
    """
    restored = {name: np.asarray(value) for name, value in source.items()}
    limits = np.iinfo(np.int32)
    if njsc_class_count not in (3, 12):
        raise ValueError("njsc_class_count must be the supported active FAO3 or USDA12 count")
    for name in ("njsc", "veget_year"):
        if name not in restored:
            continue
        value = restored[name]
        if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite integer values")
        if not np.all(value == np.trunc(value)) or np.any(value < limits.min) or np.any(value > limits.max):
            raise ValueError(f"{name} must contain exactly integral signed 32-bit values")
        if name == "njsc" and (np.any(value < 1) or np.any(value > njsc_class_count)):
            raise ValueError(f"njsc must be in 1..{njsc_class_count}")
        restored[name] = value.astype(np.int32)
    return restored


def sechiba_finalize_source_state_from_restart(
    state: SechibaRestartState,
) -> dict[str, np.ndarray]:
    """Translate serialized restart names back to component-finalize names.

    This normalizes names and restores validated INTEGER owner dtypes without
    filling missing fields or changing values. The resulting mapping is the exact input contract of
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
    return restore_sechiba_integer_source_fields(source)


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


def read_sechiba_restart_state(
    path: str | Path,
    *,
    source_pft_layout: PFTRunLayout | None = None,
    target_pft_layout: PFTRunLayout | None = None,
) -> SechibaRestartState:
    """Read all SECHIBA component-finalize state without defaults.

    Fortran provenance: ``sechiba.f90::sechiba_finalize`` lines 1800-1923 and
    component ``*_finalize`` restput calls routed by that subroutine.
    """

    with Dataset(path) as dataset:
        names = _science_variable_names(dataset)
        fields = {name: _read_science_variable(dataset.variables[name]) for name in names}
    if (source_pft_layout is None) != (target_pft_layout is None):
        raise ValueError("source and target PFT layouts must be provided together")
    if source_pft_layout is not None and target_pft_layout is not None:
        fields = remap_restart_fields_by_pft_id(
            fields,
            source_pft_layout=source_pft_layout,
            target_pft_layout=target_pft_layout,
            pft_axes=SECHIBA_NORMALIZED_RESTART_PFT_AXES,
        )
    provenance = {
        name: (
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_finalize lines 1800-1923",
            f"audited NetCDF variable {name} from component finalize restput_p",
        )
        for name in fields
    }
    return SechibaRestartState(fields, provenance)


def read_hydrol_init_restart_state(
    path: str | Path,
    *,
    npts: int,
    nvm: int,
    nstm: int | None = None,
    nslm: int | None = None,
    source_pft_layout: PFTRunLayout | None = None,
    target_pft_layout: PFTRunLayout | None = None,
) -> dict[str, np.ndarray]:
    """Read the explicit HYDROL initialization mapping from an existing file.

    The input uses the same materialized land/PFT layout as the raw reader;
    ``npts,nvm`` bind that layout to the initialization owner. Present values
    are retained. Missing ``us`` (with explicit source soil/layer dimensions)
    and ``vegstress`` are materialized as the selected DIM2 val_exp so the
    shared owner can apply its per-layer and whole-local-array tests.
    An omitted key in a caller-created mapping carries no file-read context.

    Fortran provenance: hydrol.f90::hydrol_init2678/2821-2832/3045-3058;
    ioipsl_para.f90::restget_p_opp_r2d344-372; audited IOIPSL
    restcom.f90::restget_real1715-1736 and restget_opp_r2d1159-1168;
    stringop.f90::find_str80-113; mathelp.f90::ma_fugath_r111093-1123.
    These source rules bind the selected DIM2 default999999 and valid positive
    gather indices. The raw complete-state codec and writer keep their own
    field-coverage contracts.
    """
    if npts < 1 or nvm < 1:
        raise ValueError("HYDROL initialization dimensions must be positive")
    state = read_sechiba_restart_state(
        path, source_pft_layout=source_pft_layout, target_pft_layout=target_pft_layout,
    )
    return _hydrol_init_restart_mapping(state, npts=npts, nvm=nvm, nstm=nstm, nslm=nslm)


def _hydrol_init_restart_mapping(
    state: SechibaRestartState, *, npts: int, nvm: int,
    nstm: int | None = None, nslm: int | None = None,
) -> dict[str, np.ndarray]:
    """Adapt a real file read to hydrol_init's explicit restget boundary.

    Fortran provenance: hydrol_init2678/3045-3058 and the selected IOIPSL
    restget_real1715-1736 / restget_opp_r4d1270-1348 read contract.
    """
    from jax_orchidee.sechiba.hydrol import HYDROL_INIT_RESTART_EXPVAL

    names = SECHIBA_RESTART_COMPONENT_FIELDS["hydrol"] | SECHIBA_RESTART_COMPONENT_FIELDS["explicit_snow"]
    restart = {
        SECHIBA_RESTART_TO_SOURCE_NAMES.get(name, name): value
        for name, value in state.fields.items() if name in names
    }
    if "us" not in restart:
        if nstm is None or nslm is None or nstm < 1 or nslm < 1:
            raise ValueError("missing restart us requires explicit positive source nstm and nslm")
        restart["us"] = np.full((npts, nvm, nstm, nslm), HYDROL_INIT_RESTART_EXPVAL, dtype=np.float64)
    if "us" in restart and (restart["us"].ndim != 4 or restart["us"].shape[:2] != (npts, nvm)):
        raise ValueError("restart us must have leading dimensions (npts,nvm)")
    if nstm is not None and restart["us"].shape[2] != nstm:
        raise ValueError("restart us soil axis disagrees with source nstm")
    if nslm is not None and restart["us"].shape[3] != nslm:
        raise ValueError("restart us layer axis disagrees with source nslm")
    if "vegstress" not in restart:
        restart["vegstress"] = np.full((npts, nvm), HYDROL_INIT_RESTART_EXPVAL, dtype=np.float64)
    elif restart["vegstress"].shape != (npts, nvm):
        raise ValueError("restart vegstress must have shape (npts,nvm)")
    return restart


def read_sechiba_init_restart_state(
    path: str | Path,
    *,
    npts: int,
    nvm: int,
    pref_soil_veg,
    nstm: int | None = None,
    nslm: int | None = None,
    run_def_values: Mapping[str, object] | None = None,
    source_pft_layout: PFTRunLayout | None = None,
    target_pft_layout: PFTRunLayout | None = None,
) -> SechibaRestartState:
    """Read the paper file-initialization snapshot before its first main call.

    This boundary restores real file values, materializes missing US using
    explicit source dimensions, applies layer US_INIT and then the shared
    vegstress transition. It retains the initialized US state itself. Other
    component fields keep the raw reader's values and coverage; complete-state
    serialization stays strict.
    This is separate from restoring an already initialized runtime checkpoint.

    Fortran provenance: dim2_driver.f901111-1137, intersurf.f90377-387,
    sechiba.f90602/666, hydrol.f90::hydrol_init2678/2821-2832/3040-3058. The initialization
    dispatcher precedes the first main call, for both new cold/file runs.
    """
    from jax_orchidee.sechiba.hydrol import hydrol_init_us_state, hydrol_init_vegstress

    if npts < 1 or nvm < 1:
        raise ValueError("HYDROL initialization dimensions must be positive")
    raw = read_sechiba_restart_state(
        path, source_pft_layout=source_pft_layout, target_pft_layout=target_pft_layout,
    )
    restart = _hydrol_init_restart_mapping(raw, npts=npts, nvm=nvm, nstm=nstm, nslm=nslm)
    initialized_us = hydrol_init_us_state(us=restart["us"], run_def_values=run_def_values)
    stress = hydrol_init_vegstress(
        humrelv=initialized_us.humrelv,
        pref_soil_veg=pref_soil_veg, vegstress=restart["vegstress"],
    )
    fields = {**raw.fields, "us": np.asarray(initialized_us.us), "vegstress": np.asarray(stress)}
    provenance = dict(raw.provenance_by_field)
    provenance["us"] = (*provenance.get("us", ()),
        "hydrol.f90::hydrol_init2678/2821-2832, file restore then layer-local US_INIT")
    provenance["vegstress"] = (
        *provenance.get("vegstress", ()),
        "hydrol.f90::hydrol_init3039-3058, once at the paper file initialization boundary",
    )
    return SechibaRestartState(fields, provenance)


def write_sechiba_restart_state(
    output_path: str | Path,
    *,
    state: SechibaRestartState,
    physical_state: StomateRestartPhysicalState,
    schema_path: str | Path = DEFAULT_SECHIBA_RESTART_SCHEMA,
    pft_layout: PFTRunLayout | None = None,
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
        pft_layout=pft_layout,
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
