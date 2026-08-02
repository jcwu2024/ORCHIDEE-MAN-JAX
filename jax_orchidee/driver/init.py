"""Phase 1C active-path driver initialization readers.

This module covers only the paper-case active initialization path that was
audited for Phase 1C: run scalar parsing, imposed PFT cover, and soil-tile
aggregation. It deliberately does not synthesize soilclass maps or
salinity/tide grids without their exact source/trace paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jax_orchidee.driver.domain import load_case_config
from jax_orchidee.parameters.pft_catalog import (
    PFTRunLayout,
    build_pft_run_layout,
    load_pft_catalog,
    validate_active_capabilities,
)


DEFAULT_REFERENCE_RUN_DEF = Path(
    "fortran_run_scripts/paper_250919/"
    "sen_reference_arg2_1.0_001.0-071.0/I10/"
    "S2_63.206_0.0876_0.2019_50.658/"
    "run.def_63.206_0.0876_0.2019_50.658"
)
DEFAULT_USED_RUN_DEF = Path("outputs/server_1961_trace_full_20260623/run/used_run.def")

_NATURAL_MTC = np.asarray(
    [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, True],
    dtype=bool,
)
_PREF_SOIL_VEG_MTC = np.asarray([1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 3, 3, 4], dtype=np.int32)
_LEAF_TAB_MTC = np.asarray([4, 1, 1, 2, 1, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1], dtype=np.int32)
_IS_PEAT_MTC = np.asarray(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ],
    dtype=bool,
)
_IS_C4_MTC = np.asarray(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ],
    dtype=bool,
)
_Z0_OVER_HEIGHT_MTC = np.asarray(
    [0.0, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
    dtype=np.float64,
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _apply_indexed_overrides_for_ids(
    values: np.ndarray,
    fortran_pft_ids: tuple[int, ...],
    overrides: dict[str, Any],
    prefix: str,
    *,
    dtype=None,
) -> np.ndarray:
    """Apply one-based source overrides to stable-ID-selected execution rows."""

    result = np.asarray(values, dtype=dtype).copy()
    positions = {pft_id: index for index, pft_id in enumerate(fortran_pft_ids)}
    for key, value in overrides.items():
        if not key.startswith(prefix):
            continue
        pft_id = int(key.rsplit("__", 1)[1])
        if pft_id not in positions:
            continue
        position = positions[pft_id]
        if result.dtype == np.dtype(bool):
            result[position] = _parse_bool(value)
        else:
            result[position] = value
    return result


@dataclass(frozen=True)
class RunScalars:
    """Paper-case run scalars needed by imposed vegetation and soil tiles."""

    raw_run_def: dict[str, str]
    impose_veg: bool
    nvm: int
    nstm: int
    pft_layout: PFTRunLayout
    pft_ids: tuple[str, ...]
    fortran_pft_ids: np.ndarray
    active_pft_mask: np.ndarray
    pft_to_mtc: np.ndarray
    pref_soil_veg: np.ndarray
    natural: np.ndarray
    is_tree: np.ndarray
    is_peat: np.ndarray
    is_c4: np.ndarray
    sechiba_vegmax: np.ndarray
    ext_coeff_vegetfrac: np.ndarray
    slowproc_height: np.ndarray
    z0_over_height: np.ndarray
    ratio_z0m_z0h: np.ndarray
    nleafages: int
    read_lai: bool
    ok_stomate: bool


@dataclass(frozen=True)
class ImposedVegetationState:
    """Subset of slowproc vegetation initialization that is exact in Phase 1C."""

    veget_max: np.ndarray
    frac_nobio: np.ndarray
    totfrac_nobio: np.ndarray
    soiltile: np.ndarray
    veget: np.ndarray | None = None


def parse_run_def(path: str | Path) -> dict[str, str]:
    """Parse simple `KEY=VALUE` run.def entries.

    Fortran/run provenance: `fortran_run_scripts/paper_250919/Job0_bio`, lines
    55-87, 141-162, and 325-335, applies `remplace KEY VALUE` edits to run.def
    files. ORCHIDEE later consumes those keys through `getin_p`; this parser is
    intentionally limited to literal, non-comment `KEY=VALUE` lines and does
    not attempt to emulate the full IOIPSL/getin parser.
    """

    values: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.split("#", 1)[0].strip()
    return values


def parse_run_def_bool(value: str | bool) -> bool:
    """Parse a literal ORCHIDEE run.def boolean.

    Fortran/run provenance: scalar flags in ``run.def`` are consumed through
    ``getin_p`` booleans; for example ``control.f90`` lines 103-104 reads
    ``RIVER_ROUTING``. This helper only normalizes already materialized
    run-definition strings and refuses ambiguous values.
    """

    if isinstance(value, bool):
        return value
    token = str(value).strip().strip(".").upper()
    if token in {"T", "TRUE", "Y", "YES", "1"}:
        return True
    if token in {"F", "FALSE", "N", "NO", "0"}:
        return False
    raise ValueError(f"cannot parse run.def boolean value {value!r}")


def parse_run_def_float(values: dict[str, str], name: str) -> float:
    """Return a required numeric run.def scalar from ``parse_run_def`` output."""

    if name not in values:
        raise KeyError(name)
    return float(values[name])


def parse_run_def_int(values: dict[str, str], name: str) -> int:
    """Return a required integer run.def scalar from ``parse_run_def`` output."""

    return int(parse_run_def_float(values, name))


def parse_run_def_indexed_float(values: dict[str, str], name: str, index: int) -> float:
    """Return one ``NAME__00001`` style numeric run.def entry.

    Fortran provenance: indexed PFT parameters are consumed through repeated
    ``getin_p`` vector entries; the archived ``used_run.def`` materializes
    those entries as fixed-width one-based keys such as ``VCMAX_FIX__00014``.
    """

    return parse_run_def_float(values, f"{name}__{int(index):05d}")


def parse_run_def_indexed_bool(values: dict[str, str], name: str, index: int) -> bool:
    """Return one ``NAME__00001`` style boolean run.def entry."""

    key = f"{name}__{int(index):05d}"
    if key not in values:
        raise KeyError(key)
    return parse_run_def_bool(values[key])


def parse_run_def_indexed_vector(
    values: dict[str, str],
    name: str,
    count: int,
    *,
    dtype=np.float64,
) -> np.ndarray:
    """Return a one-based indexed run.def vector as a NumPy array."""

    return np.asarray(
        [parse_run_def_indexed_float(values, name, index) for index in range(1, int(count) + 1)],
        dtype=dtype,
    )


def parse_run_def_indexed_selection(
    values: dict[str, str],
    name: str,
    fortran_pft_ids: tuple[int, ...],
    *,
    dtype=np.float64,
) -> np.ndarray:
    """Read indexed source rows in a stable-ID-selected execution order."""

    return np.asarray(
        [parse_run_def_indexed_float(values, name, index) for index in fortran_pft_ids],
        dtype=dtype,
    )


def reference_run_def_path(config_path: str | Path) -> Path:
    """Resolve the local reference run.def file used for scalar inspection.

    Fortran/run provenance: `Job0_bio` writes per-case run.def files during the
    paper run; the YAML records the template path but not a dedicated
    `reference_run_def` key, so Phase 1C uses the audited local selected-case
    run.def path when no such key is present.
    """

    config = load_case_config(config_path)
    path = config.get("case", {}).get("reference_run_def")
    if path:
        return Path(path)
    return Path(config["paths"]["workspace_root"]) / DEFAULT_REFERENCE_RUN_DEF


def _default_used_run_def_path(config_path: str | Path) -> Path:
    """Resolve the audited run.def with materialized Fortran getin defaults."""

    config = load_case_config(config_path)
    root = Path(config["paths"]["workspace_root"])
    return root / DEFAULT_USED_RUN_DEF


def _values_with_materialized_defaults(
    config_path: str | Path,
    values: dict[str, str],
    required_keys: tuple[str, ...],
) -> dict[str, str]:
    """Fill missing materialized defaults from the audited ``used_run.def``.

    Fortran provenance: ``pft_parameters.f90`` reads indexed values through
    ``getin_p`` (`SLOWPROC_HEIGHT` lines 3002-3008 and
    `EXT_COEFF_VEGETFRAC` lines 3592-3598). The archived template run.def does
    not always materialize defaults; ``outputs/server_1961_trace_full_20260623``
    records the actual values emitted by the audited paper-case run.
    """

    missing = [key for key in required_keys if key not in values]
    if not missing:
        return values
    used_path = _default_used_run_def_path(config_path)
    if not used_path.exists():
        raise KeyError(f"run.def is missing {missing!r} and audited used_run.def was not found at {used_path}")
    materialized = parse_run_def(used_path)
    still_missing = [key for key in missing if key not in materialized]
    if still_missing:
        raise KeyError(f"audited used_run.def is missing required defaults {still_missing!r}")
    merged = dict(values)
    for key in missing:
        merged[key] = materialized[key]
    return merged


def read_run_scalars(config_path: str | Path, run_def_path: str | Path | None = None) -> RunScalars:
    """Read active paper-case vegetation and soil-tile scalars.

    Fortran provenance: `fortran_run_scripts/paper_250919/Job0_bio`, lines
    62-77, writes `IMPOSE_VEG` and `SECHIBA_VEGMAX__00001..00014`; lines
    141-162 writes `PREF_SOIL_VEG__00014=4` and `NSTM/NVM` for the active
    peat/tide path. `pft_parameters.f90`, lines 109-123, initializes and reads
    `PFT_TO_MTC`; lines 247-293 map MTC constants to PFT parameters; lines
    3011-3051 and 3433-3441 read `NATURAL`, `IS_C4`, `IS_PEAT`, and
    `PREF_SOIL_VEG`. `slowproc.f90`, subroutine `slowproc_initialize`, lines
    1586-1601 and 1881-1926, reads restart/imposed vegetation values.

    The selected local run.def file is parsed as a simple key-value source, but
    the active paper-case `remplace` overrides are taken from the audited YAML
    because the archived run.def in this repository is still template-like for
    several keys.
    """

    config = load_case_config(config_path)
    run_def = parse_run_def(run_def_path or reference_run_def_path(config_path))

    catalog_config = config["pft_catalog"]
    catalog = load_pft_catalog(catalog_config["path"])
    layout_id = str(catalog_config["layout_id"])
    if layout_id not in catalog.layouts:
        raise KeyError(f"configured PFT layout {layout_id!r} is absent from the catalog")
    catalog_entries = tuple(catalog.entries_by_id[pft_id] for pft_id in catalog.layouts[layout_id])
    fortran_pft_ids = tuple(entry.fortran_pft_id for entry in catalog_entries)
    nvm = len(catalog_entries)
    configured_nvm = int(config["structural_overrides"]["NVM"])
    if configured_nvm != nvm:
        raise ValueError(
            f"structural NVM={configured_nvm} disagrees with PFT layout {layout_id!r} length {nvm}"
        )
    nstm = int(config["structural_overrides"]["NSTM"])
    impose_veg = bool(config["run_def_flags"]["IMPOSE_VEG"])
    structural_overrides = config["structural_overrides"]

    vegmax_by_key = config["prescribed_pft_cover"]["sechiba_vegmax"]
    sechiba_vegmax = np.asarray(
        [float(vegmax_by_key[f"SECHIBA_VEGMAX__{pft_id:05d}"]) for pft_id in fortran_pft_ids],
        dtype=np.float64,
    )

    pft_layout = build_pft_run_layout(
        catalog,
        layout_id=layout_id,
        fractions=sechiba_vegmax,
    )
    validate_active_capabilities(catalog, pft_layout)

    pft_to_mtc = np.asarray([entry.mtc_id for entry in catalog_entries], dtype=np.int32)
    pft_to_mtc = _apply_indexed_overrides_for_ids(
        pft_to_mtc, fortran_pft_ids, structural_overrides, "PFT_TO_MTC__", dtype=np.int32
    )
    catalog_mtc = np.asarray([entry.mtc_id for entry in catalog_entries], dtype=np.int32)
    if not np.array_equal(pft_to_mtc, catalog_mtc):
        raise ValueError("PFT_TO_MTC overrides disagree with the selected source-backed catalog")
    if np.any(pft_to_mtc < 1) or np.any(pft_to_mtc > _NATURAL_MTC.shape[0]):
        raise ValueError("pft_to_mtc contains an MTC outside the Fortran constantes_mtc table")

    mtc_index = pft_to_mtc - 1
    pref_soil_veg = _PREF_SOIL_VEG_MTC[mtc_index].astype(np.int32)
    is_tree = (_LEAF_TAB_MTC[mtc_index] <= 2).astype(bool)
    natural = _NATURAL_MTC[mtc_index].astype(bool)
    is_peat = _IS_PEAT_MTC[mtc_index].astype(bool)
    is_c4 = _IS_C4_MTC[mtc_index].astype(bool)
    z0_over_height = _Z0_OVER_HEIGHT_MTC[mtc_index].astype(np.float64)

    pref_soil_veg = _apply_indexed_overrides_for_ids(
        pref_soil_veg, fortran_pft_ids, structural_overrides, "PREF_SOIL_VEG__", dtype=np.int32
    )
    natural = _apply_indexed_overrides_for_ids(
        natural, fortran_pft_ids, structural_overrides, "NATURAL__", dtype=bool
    )
    is_peat = _apply_indexed_overrides_for_ids(
        is_peat, fortran_pft_ids, structural_overrides, "IS_PEAT__", dtype=bool
    )
    is_c4 = _apply_indexed_overrides_for_ids(
        is_c4, fortran_pft_ids, structural_overrides, "IS_C4__", dtype=bool
    )
    if np.any(pref_soil_veg < 1) or np.any(pref_soil_veg > nstm):
        raise ValueError("pref_soil_veg contains a soil tile outside 1..NSTM")

    trait_arrays = {
        "natural": natural,
        "is_tree": is_tree,
        "is_peat": is_peat,
        "is_c4": is_c4,
        "pref_soil_veg": pref_soil_veg,
    }
    for position, entry in enumerate(catalog_entries):
        for trait_name, values in trait_arrays.items():
            if trait_name in entry.traits and values[position] != entry.traits[trait_name]:
                raise ValueError(
                    f"catalog trait {entry.pft_id}.{trait_name} disagrees with source-derived runtime value"
                )

    indexed_default_keys = tuple(
        [f"EXT_COEFF_VEGETFRAC__{pft_id:05d}" for pft_id in fortran_pft_ids]
        + [f"SLOWPROC_HEIGHT__{pft_id:05d}" for pft_id in fortran_pft_ids]
        + [f"RATIO_Z0M_Z0H__{pft_id:05d}" for pft_id in fortran_pft_ids]
    )
    run_def_with_defaults = _values_with_materialized_defaults(config_path, run_def, indexed_default_keys)
    ext_coeff_vegetfrac = parse_run_def_indexed_selection(
        run_def_with_defaults, "EXT_COEFF_VEGETFRAC", fortran_pft_ids
    )
    slowproc_height = parse_run_def_indexed_selection(
        run_def_with_defaults, "SLOWPROC_HEIGHT", fortran_pft_ids
    )
    ratio_z0m_z0h = parse_run_def_indexed_selection(
        run_def_with_defaults, "RATIO_Z0M_Z0H", fortran_pft_ids
    )
    nleafages = parse_run_def_int(run_def_with_defaults, "NLEAFAGES") if "NLEAFAGES" in run_def_with_defaults else 4
    read_lai = parse_run_def_bool(run_def_with_defaults.get("READ_LAI", "FALSE"))
    ok_stomate = parse_run_def_bool(run_def_with_defaults.get("STOMATE_OK_STOMATE", "TRUE"))

    return RunScalars(
        raw_run_def=run_def_with_defaults,
        impose_veg=impose_veg,
        nvm=nvm,
        nstm=nstm,
        pft_layout=pft_layout,
        pft_ids=pft_layout.pft_ids,
        fortran_pft_ids=np.asarray(fortran_pft_ids, dtype=np.int32),
        active_pft_mask=np.asarray(pft_layout.active_mask, dtype=bool),
        pft_to_mtc=pft_to_mtc,
        pref_soil_veg=pref_soil_veg,
        natural=natural,
        is_tree=is_tree,
        is_peat=is_peat,
        is_c4=is_c4,
        sechiba_vegmax=sechiba_vegmax,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        slowproc_height=slowproc_height,
        z0_over_height=z0_over_height,
        ratio_z0m_z0h=ratio_z0m_z0h,
        nleafages=nleafages,
        read_lai=read_lai,
        ok_stomate=ok_stomate,
    )


def initialize_imposed_veget_max(run_scalars: RunScalars, npts: int) -> np.ndarray:
    """Initialize `veget_max[npts,nvm]` from imposed `SECHIBA_VEGMAX`.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_initialize`, lines
    1586-1601 first attempts restart reads; lines 1881-1926 use the
    `IMPOSE_VEG` branch and call `setvar_p(veget_max, ..., 'SECHIBA_VEGMAX',
    ...)` when restart values are absent.
    """

    if not run_scalars.impose_veg:
        raise ValueError("Phase 1C implements only the active IMPOSE_VEG paper-case path")
    if npts < 1:
        raise ValueError("npts must be positive")
    return np.repeat(run_scalars.sechiba_vegmax[None, :], npts, axis=0)


def soil_tiles_from_veget_max(
    veget_max,
    pref_soil_veg,
    *,
    nstm: int,
    frac_nobio=None,
    min_vegfrac: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate PFT cover onto soil tiles with Fortran `slowproc_veget` logic.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_veget`, lines
    2820-2925. This implements the exact Phase 1C subset: small-fraction
    cleanup and normalization (2858-2875), `totfrac_nobio` (2899-2902), and
    `jst = pref_soil_veg(jv); soiltile(ji,jst)+=veget_max(ji,jv)` followed by
    division by `1 - totfrac_nobio` (2912-2921). It does not compute `veget`,
    which depends on LAI and `ext_coeff_vegetfrac` at lines 2893-2898.
    """

    veget_max = np.asarray(veget_max, dtype=np.float64).copy()
    pref_soil_veg = np.asarray(pref_soil_veg, dtype=np.int32)
    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape [npts,nvm]")
    npts, nvm = veget_max.shape
    if pref_soil_veg.shape != (nvm,):
        raise ValueError("pref_soil_veg must have shape [nvm]")
    if np.any(pref_soil_veg < 1) or np.any(pref_soil_veg > nstm):
        raise ValueError("pref_soil_veg contains a soil tile outside 1..nstm")

    if frac_nobio is None:
        frac_nobio_arr = np.zeros((npts, 1), dtype=np.float64)
    else:
        frac_nobio_arr = np.asarray(frac_nobio, dtype=np.float64).copy()
        if frac_nobio_arr.ndim != 2 or frac_nobio_arr.shape[0] != npts:
            raise ValueError("frac_nobio must have shape [npts,nnobio]")

    for ji in range(npts):
        if np.sum(frac_nobio_arr[ji, :]) < min_vegfrac:
            frac_nobio_arr[ji, :] = 0.0
        veget_max[ji, veget_max[ji, :] < min_vegfrac] = 0.0
        total = float(np.sum(frac_nobio_arr[ji, :]) + np.sum(veget_max[ji, :]))
        if total <= 0.0:
            raise ValueError("veget_max + frac_nobio sum must be positive")
        frac_nobio_arr[ji, :] /= total
        veget_max[ji, :] /= total

    totfrac_nobio = np.sum(frac_nobio_arr, axis=1)
    soiltile = np.zeros((npts, nstm), dtype=np.float64)
    for jv in range(nvm):
        jst = int(pref_soil_veg[jv]) - 1
        soiltile[:, jst] += veget_max[:, jv]
    for ji in range(npts):
        if totfrac_nobio[ji] < (1.0 - min_vegfrac):
            soiltile[ji, :] /= 1.0 - totfrac_nobio[ji]
    return soiltile, veget_max, totfrac_nobio


def initialize_imposed_vegetation_state(run_scalars: RunScalars, npts: int) -> ImposedVegetationState:
    """Initialize exact Phase 1C imposed vegetation state.

    Fortran provenance: `slowproc.f90`, lines 1881-1926 for imposed
    `veget_max`, and subroutine `slowproc_veget`, lines 2820-2925 for
    normalization, LAI-to-`veget` conversion, and soil-tile aggregation. The
    cold-start path follows `slowproc_init`, lines 2149-2172: before missing
    LAI is reset to zero for STOMATE, `slowproc_veget` sees `val_exp` LAI and
    present PFTs become full-cover `veget`.
    """

    from jax_orchidee.sechiba.slowproc import slowproc_cold_start_vegetation_entry_state

    veget_max = initialize_imposed_veget_max(run_scalars, npts)
    frac_nobio = np.zeros((npts, 1), dtype=np.float64)
    cold_start = slowproc_cold_start_vegetation_entry_state(
        veget_max=veget_max,
        frac_nobio=frac_nobio,
        pref_soil_veg=run_scalars.pref_soil_veg,
        ext_coeff_vegetfrac=run_scalars.ext_coeff_vegetfrac,
        height_presc=run_scalars.slowproc_height,
        nstm=run_scalars.nstm,
        nleafages=run_scalars.nleafages,
        read_lai=run_scalars.read_lai,
        ok_stomate=run_scalars.ok_stomate,
    )
    return ImposedVegetationState(
        veget_max=np.asarray(cold_start.vegetation.veget_max, dtype=np.float64),
        frac_nobio=np.asarray(cold_start.vegetation.frac_nobio, dtype=np.float64),
        totfrac_nobio=np.asarray(cold_start.vegetation.totfrac_nobio, dtype=np.float64),
        soiltile=np.asarray(cold_start.vegetation.soiltile, dtype=np.float64),
        veget=np.asarray(cold_start.vegetation.veget, dtype=np.float64),
    )
