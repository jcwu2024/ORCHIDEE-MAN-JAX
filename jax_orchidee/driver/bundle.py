"""Phase 1F first-step driver/SECHIBA boundary bundle.

This module composes the source-backed driver readers implemented in Phases
1B-1E. It does not run SECHIBA, HYDROL, or STOMATE process code and it does not
invent grid/static geometry that is missing from local trace truth.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from jax_orchidee.driver.domain import (
    DomainGrid,
    ForcingStep,
    WaterTableSequences,
    load_case_config,
    read_annual_co2,
    read_domain_grid,
    read_forcing_first_step,
    read_forcing_model_step_cached,
    read_forcing_step,
    read_water_table_sequences,
)
from jax_orchidee.driver.init import ImposedVegetationState, RunScalars, initialize_imposed_vegetation_state, read_run_scalars
from jax_orchidee.driver.restart import SechibaStaticRestart, read_sechiba_static_restart, reference_restart_path
from jax_orchidee.driver.static import read_paper_salinity_tide_static_fields, read_paper_usda_soil_static_fields
from jax_orchidee.driver.trace import (
    SERVER_1961_TRACE_FILENAMES,
    StaticTraceFields,
    apply_fixed_format_grid_trace_to_domain,
    read_static_fields_from_fixed_format_trace_dir,
)


MISSING_GEOMETRY_FIELDS = (
    "resolution",
    "neighbours",
    "corners",
    "seglength",
)

MISSING_STATIC_FIELDS = (
    "soilclass",
    "soilclass_sub_index",
    "soilclass_sub_area",
    "njsc",
    "clay_frac",
    "sand_frac",
    "silt_frac",
    "bulk_dens",
    "soil_ph",
    "poor_soils",
    "salinity",
    "salinity_bbox",
    "tide_height",
    "tide_bbox",
    "veget",
)


@dataclass(frozen=True)
class DriverFirstStepBundle:
    """Verified first-step inputs at the driver/SECHIBA boundary."""

    year: int
    domain: DomainGrid
    forcing: ForcingStep
    run_scalars: RunScalars
    vegetation: ImposedVegetationState
    co2_ppm: float
    ccanopy: np.ndarray
    water_table: WaterTableSequences
    restart_anchors: SechibaStaticRestart | None
    static_trace_fields: StaticTraceFields
    missing_geometry_fields: tuple[str, ...]
    missing_static_fields: tuple[str, ...]


@dataclass(frozen=True)
class DriverStepBundle:
    """Verified inputs at one driver/SECHIBA timestep boundary."""

    year: int
    tstep: int
    domain: DomainGrid
    forcing: ForcingStep
    run_scalars: RunScalars
    vegetation: ImposedVegetationState
    co2_ppm: float
    ccanopy: np.ndarray
    water_table: WaterTableSequences
    restart_anchors: SechibaStaticRestart | None
    static_trace_fields: StaticTraceFields
    missing_geometry_fields: tuple[str, ...]
    missing_static_fields: tuple[str, ...]


def _missing_static_after_trace(static_trace_fields: StaticTraceFields) -> tuple[str, ...]:
    missing = set(MISSING_STATIC_FIELDS)
    for name, value in (
        ("soilclass", static_trace_fields.soilclass),
        ("njsc", static_trace_fields.njsc),
        ("clay_frac", static_trace_fields.clay_frac),
        ("sand_frac", static_trace_fields.sand_frac),
        ("silt_frac", static_trace_fields.silt_frac),
        ("bulk_dens", static_trace_fields.bulk_dens),
        ("soil_ph", static_trace_fields.soil_ph),
        ("poor_soils", static_trace_fields.poor_soils),
        ("soilclass_sub_index", static_trace_fields.soilclass_sub_index),
        ("soilclass_sub_area", static_trace_fields.soilclass_sub_area),
        ("salinity", static_trace_fields.salinity),
        ("tide_height", static_trace_fields.tide_height),
    ):
        if value is not None:
            missing.discard(name)
    if static_trace_fields.salinity is not None:
        missing.discard("salinity_bbox")
    if static_trace_fields.tide_height is not None:
        missing.discard("tide_bbox")
    return tuple(name for name in MISSING_STATIC_FIELDS if name in missing)


def _missing_static_after_trace_and_vegetation(
    static_trace_fields: StaticTraceFields,
    vegetation: ImposedVegetationState,
) -> tuple[str, ...]:
    """Return unresolved static fields after trace/local vegetation assembly."""

    missing = set(_missing_static_after_trace(static_trace_fields))
    if vegetation.veget is not None:
        missing.discard("veget")
    return tuple(name for name in MISSING_STATIC_FIELDS if name in missing)


def _static_fields_with_local_inputs(
    config_path: str | Path,
    *,
    domain: DomainGrid,
    static_fields: StaticTraceFields,
) -> StaticTraceFields:
    """Add local slowproc static inputs when exact geometry exists."""

    if (
        static_fields.soilclass is not None
        and static_fields.njsc is not None
        and static_fields.clay_frac is not None
        and static_fields.sand_frac is not None
        and static_fields.silt_frac is not None
        and static_fields.bulk_dens is not None
        and static_fields.soil_ph is not None
        and static_fields.poor_soils is not None
        and static_fields.soilclass_sub_index is not None
        and static_fields.soilclass_sub_area is not None
        and static_fields.salinity is not None
        and static_fields.tide_height is not None
    ):
        return static_fields
    needs_local = any(
        value is None
        for value in (
            static_fields.soilclass,
            static_fields.njsc,
            static_fields.clay_frac,
            static_fields.sand_frac,
            static_fields.silt_frac,
            static_fields.bulk_dens,
            static_fields.soil_ph,
            static_fields.poor_soils,
            static_fields.soilclass_sub_index,
            static_fields.soilclass_sub_area,
            static_fields.salinity,
            static_fields.tide_height,
        )
    )
    if domain.resolution is None:
        if needs_local:
            raise ValueError(
                "local slowproc static inputs require explicit driver resolution "
                "before reading soilclass/salinity/tide fields"
            )
        return static_fields
    soil_fields = None
    soil_overlap = None
    if (
        static_fields.soilclass is None
        or static_fields.njsc is None
        or static_fields.clay_frac is None
        or static_fields.sand_frac is None
        or static_fields.silt_frac is None
        or static_fields.bulk_dens is None
        or static_fields.soil_ph is None
        or static_fields.poor_soils is None
        or static_fields.soilclass_sub_index is None
        or static_fields.soilclass_sub_area is None
    ):
        soil_fields, soil_overlap = read_paper_usda_soil_static_fields(
            config_path,
            lalo=domain.lalo,
            resolution_m=domain.resolution,
        )
    salinity = static_fields.salinity
    tide_height = static_fields.tide_height
    if salinity is None or tide_height is None:
        local_salinity, local_tide, _ = read_paper_salinity_tide_static_fields(
            config_path,
            lalo=domain.lalo,
            resolution_m=domain.resolution,
        )
        if salinity is None:
            salinity = local_salinity
        if tide_height is None:
            tide_height = local_tide
    return StaticTraceFields(
        soilclass=static_fields.soilclass if static_fields.soilclass is not None else (None if soil_fields is None else soil_fields["soilclass"]),
        njsc=static_fields.njsc if static_fields.njsc is not None else (None if soil_fields is None else soil_fields["njsc"]),
        clay_frac=static_fields.clay_frac if static_fields.clay_frac is not None else (None if soil_fields is None else soil_fields["clay_frac"]),
        sand_frac=static_fields.sand_frac if static_fields.sand_frac is not None else (None if soil_fields is None else soil_fields["sand_frac"]),
        silt_frac=static_fields.silt_frac if static_fields.silt_frac is not None else (None if soil_fields is None else soil_fields["silt_frac"]),
        bulk_dens=static_fields.bulk_dens if static_fields.bulk_dens is not None else (None if soil_fields is None else soil_fields["bulk_dens"]),
        soil_ph=static_fields.soil_ph if static_fields.soil_ph is not None else (None if soil_fields is None else soil_fields["soil_ph"]),
        poor_soils=static_fields.poor_soils if static_fields.poor_soils is not None else (None if soil_fields is None else soil_fields["poor_soils"]),
        soilclass_sub_index=static_fields.soilclass_sub_index
        if static_fields.soilclass_sub_index is not None
        else (None if soil_overlap is None else soil_overlap["soilclass_sub_index"]),
        soilclass_sub_area=static_fields.soilclass_sub_area
        if static_fields.soilclass_sub_area is not None
        else (None if soil_overlap is None else soil_overlap["soilclass_sub_area"]),
        salinity=salinity,
        tide_height=tide_height,
        blocked_fields=static_fields.blocked_fields,
    )


def _missing_geometry_after_domain(domain: DomainGrid) -> tuple[str, ...]:
    missing = set(MISSING_GEOMETRY_FIELDS)
    if domain.resolution is not None:
        missing.discard("resolution")
    if domain.neighbours is not None:
        missing.discard("neighbours")
    if domain.corners is not None:
        missing.discard("corners")
    if domain.seglength is not None:
        missing.discard("seglength")
    return tuple(name for name in MISSING_GEOMETRY_FIELDS if name in missing)


def paper_forcing_split_from_config(config: dict[str, object], run_scalars: RunScalars) -> int:
    """Return the active paper-case forcing split count.

    Fortran provenance: `dim2_driver.f90`, lines 293-310 set
    `split = INT(dt_force/dt)` for non-weathergen forcing, where `dt_force`
    comes from forcing metadata and `dt` is `DT_SECHIBA`.
    """

    forcing_hours = float(config["time"]["forcing_step_hours"])  # type: ignore[index]
    dt_sechiba = float(getattr(run_scalars, "dt_sechiba", 1800.0))
    return int((forcing_hours * 3600.0) / dt_sechiba)


def paper_forcing_nb_spread(split: int) -> int:
    """Return active paper-case precipitation spreading count.

    Fortran provenance: `dim2_driver.f90`, lines 650-674. With 6-hour forcing
    and 1800 s SECHIBA, the default `SPRED_PREC` is half the split, i.e. 6.
    """

    split = int(split)
    if split < 1:
        raise ValueError("split must be positive")
    return min(int(0.5 * split), split) if split >= 6 else split


def ccanopy_from_co2(co2_ppm: float, npts: int) -> np.ndarray:
    """Build the land-point `ccanopy` vector from annual `ATM_CO2`.

    Fortran provenance: `fortran_source/ORCHIDEE/src_driver/dim2_driver.f90`,
    lines 708-719 read `ATM_CO2` and set `for_ccanopy(:,:)=atmco2`; lines
    1111-1117 pass `for_ccanopy` to `intersurf_initialize_2d`, and
    `intersurf.f90`, lines 113-152 and 377-387, forwards gathered `ccanopy`
    to `sechiba_initialize`.
    """

    if npts < 1:
        raise ValueError("npts must be positive")
    return np.full((npts,), float(co2_ppm), dtype=np.float64)


def load_paper_1961_first_step_bundle(
    config_path: str | Path,
    *,
    year: int = 1961,
    run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    domain_override: dict[str, float] | None = None,
) -> DriverFirstStepBundle:
    """Compose the verified active-path first-step driver bundle.

    Fortran provenance: this function composes source-backed readers only.
    Domain/forcing fields follow `dim2_driver.f90`, lines 147-175, 415-435,
    and 1108-1125 plus `readdim2.f90` audited `forcing_info`,
    `domain_size`, `forcing_landind`, and `forcing_just_read` paths. CO2
    follows `Job0_bio`, lines 334-335, and `dim2_driver.f90`, lines 708-719.
    Imposed PFT14 cover and soil tiles follow `slowproc.f90`, lines
    1881-1926 and `slowproc_veget`, lines 2820-2925. Restart anchors follow
    `slowproc.f90`, lines 1581-1601, 1655-1674, and 1791-1804.

    Missing grid/static fields are explicit metadata. No `resolution`,
    `neighbours`, `soilclass`, `salinity`, `tide_height`, or LAI-dependent
    `veget` values are filled without exact trace truth. When a
    `fixed_format_trace_dir` is supplied, supported geometry and static fields
    are read only from the audited fixed-format server traces.
    """

    domain = read_domain_grid(config_path, year=year, domain_override=domain_override)
    forcing = read_forcing_first_step(config_path, domain=domain, year=year)
    run_scalars = read_run_scalars(config_path, run_def_path=run_def_path)
    vegetation = initialize_imposed_vegetation_state(run_scalars, npts=domain.nbindex)
    co2_ppm = read_annual_co2(config_path, year)
    ccanopy = ccanopy_from_co2(co2_ppm, domain.nbindex)
    water_table = read_water_table_sequences(config_path)

    try:
        sechiba_start = (
            Path(reference_run_dir) / "sechiba_start.nc"
            if reference_run_dir is not None
            else reference_restart_path(config_path, "sechiba_start.nc")
        )
        restart_anchors = read_sechiba_static_restart(sechiba_start)
    except FileNotFoundError:
        restart_anchors = None

    if fixed_format_trace_dir is not None and static_trace_fields is not None:
        raise ValueError("pass either fixed_format_trace_dir or static_trace_fields, not both")
    if fixed_format_trace_dir is not None:
        trace_dir = Path(fixed_format_trace_dir)
        domain = apply_fixed_format_grid_trace_to_domain(domain, trace_dir / SERVER_1961_TRACE_FILENAMES["grid"])
        static_truth = read_static_fields_from_fixed_format_trace_dir(trace_dir)
    else:
        static_truth = static_trace_fields or StaticTraceFields()
    static_truth = _static_fields_with_local_inputs(
        config_path,
        domain=domain,
        static_fields=static_truth,
    )

    missing_geometry = _missing_geometry_after_domain(domain)
    missing_static = _missing_static_after_trace_and_vegetation(static_truth, vegetation)
    return DriverFirstStepBundle(
        year=year,
        domain=domain,
        forcing=forcing,
        run_scalars=run_scalars,
        vegetation=vegetation,
        co2_ppm=co2_ppm,
        ccanopy=ccanopy,
        water_table=water_table,
        restart_anchors=restart_anchors,
        static_trace_fields=static_truth,
        missing_geometry_fields=missing_geometry,
        missing_static_fields=missing_static,
    )


@lru_cache(maxsize=8)
def _load_paper_1961_first_step_bundle_cached(config_path: str, year: int) -> DriverFirstStepBundle:
    """Cache deterministic paper-case static/domain first-step assembly."""

    return load_paper_1961_first_step_bundle(config_path, year=year)


def load_paper_1961_step_bundle(
    config_path: str | Path,
    *,
    year: int = 1961,
    tstep: int = 0,
    run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    domain_override: dict[str, float] | None = None,
) -> DriverStepBundle:
    """Compose the verified active-path driver bundle for one forcing step.

    Fortran provenance matches ``load_paper_1961_first_step_bundle`` for the
    static/domain pieces and ``readdim2.f90::forcing_read_interpol`` lines
    660-1687 plus ``forcing_just_read`` lines 1691-1869 for the selected raw
    forcing slab. The function deliberately does not advance restart state or
    infer SECHIBA/STOMATE process state.
    """

    if (
        fixed_format_trace_dir is None
        and static_trace_fields is None
        and domain_override is None
        and run_def_path is None
        and reference_run_dir is None
    ):
        first = _load_paper_1961_first_step_bundle_cached(str(Path(config_path).resolve()), int(year))
    else:
        first = load_paper_1961_first_step_bundle(
            config_path,
            year=year,
            run_def_path=run_def_path,
            reference_run_dir=reference_run_dir,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            domain_override=domain_override,
        )
    config = load_case_config(config_path)
    split = paper_forcing_split_from_config(config, first.run_scalars)
    forcing = read_forcing_model_step_cached(
        config_path,
        domain=first.domain,
        year=year,
        model_tstep=tstep,
        split=split,
        nb_spread=paper_forcing_nb_spread(split),
    )
    return DriverStepBundle(
        year=year,
        tstep=int(tstep),
        domain=first.domain,
        forcing=forcing,
        run_scalars=first.run_scalars,
        vegetation=first.vegetation,
        co2_ppm=first.co2_ppm,
        ccanopy=first.ccanopy,
        water_table=first.water_table,
        restart_anchors=first.restart_anchors,
        static_trace_fields=first.static_trace_fields,
        missing_geometry_fields=first.missing_geometry_fields,
        missing_static_fields=first.missing_static_fields,
    )


def iter_paper_year_step_bundles(
    config_path: str | Path,
    *,
    year: int = 1961,
    run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    nsteps: int | None = None,
) -> Iterator[DriverStepBundle]:
    """Yield source-backed driver bundles in forcing-step order for one year.

    Fortran provenance: ``dim2_driver.f90`` lines 839-908 calls
    ``forcing_read`` during the time loop, and ``readdim2.f90`` lines 660-1687
    and 1691-1869 provide the ordered forcing slabs. This iterator only yields
    driver boundary inputs; it does not mutate restart state or run SECHIBA.
    """

    first = load_paper_1961_first_step_bundle(
        config_path,
        year=year,
        run_def_path=run_def_path,
        reference_run_dir=reference_run_dir,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
    )
    config = load_case_config(config_path)
    split = paper_forcing_split_from_config(config, first.run_scalars)
    total = int(nsteps) if nsteps is not None else int(config["drivers"]["atmospheric_forcing"]["dimensions"]["tstep"]) * split
    if total < 0:
        raise ValueError("nsteps must be non-negative")
    for tstep in range(total):
        forcing = read_forcing_model_step_cached(
            config_path,
            domain=first.domain,
            year=year,
            model_tstep=tstep,
            split=split,
            nb_spread=paper_forcing_nb_spread(split),
        )
        yield DriverStepBundle(
            year=year,
            tstep=tstep,
            domain=first.domain,
            forcing=forcing,
            run_scalars=first.run_scalars,
            vegetation=first.vegetation,
            co2_ppm=first.co2_ppm,
            ccanopy=first.ccanopy,
            water_table=first.water_table,
            restart_anchors=first.restart_anchors,
            static_trace_fields=first.static_trace_fields,
            missing_geometry_fields=first.missing_geometry_fields,
            missing_static_fields=first.missing_static_fields,
        )
