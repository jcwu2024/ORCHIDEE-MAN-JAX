"""Reference restart readers for driver/static-state parity checks.

The routines in this module read exact fields from local Fortran restart files.
They are validation helpers, not source formulas: Phase 1D may compare JAX
driver/init state against these values, but static-state generation must still
follow the audited Fortran source path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from jax_orchidee.driver.domain import load_case_config
from jax_orchidee.parameters.pft_catalog import PFTRunLayout, remap_pft_axis

REFERENCE_CASE_OUTPUT = Path(
    "reference/case_001_071/OUT/orc_calibrate_250919_sen/"
    "arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658"
)

SECHIBA_STATIC_FIELDS = (
    "veget",
    "veget_max",
    "frac_nobio",
    "njsc",
    "clay_frac",
    "sand_frac",
    "bulk_dens",
    "soil_ph",
    "poor_soils",
    "wtp",
    "wt_ab_tide",
)

SECHIBA_RESTART_PFT_AXES = {
    "veget": 1,
    "veget_max": 1,
    "lai": 1,
    "height": 1,
    "frac_age": 2,
    "cdrag_pft": 1,
    "rstruct": 1,
    "leaf_ci": 1,
    "temp_sol_pft": 1,
    "us": 1,
    "qsintveg": 1,
    "vegstress": 1,
    "humrel": 1,
    "roughheight_pft": 1,
    "ptn": 2,
    "shum_ngrnd_prmlng": 2,
    "shum_ngrnd_perma": 2,
    "e_soil_lat": 1,
    "soilcap_pft": 1,
    "soilflx_pft": 1,
    "cgrnd": 2,
    "dgrnd": 2,
}


@dataclass(frozen=True)
class RestartFieldInventory:
    """Presence/absence summary for selected NetCDF restart variables."""

    path: Path
    present: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class SechibaStaticRestart:
    """Selected SECHIBA restart fields flattened to model-point axes."""

    path: Path
    veget: np.ndarray | None
    veget_max: np.ndarray | None
    frac_nobio: np.ndarray | None
    njsc: np.ndarray | None
    clay_frac: np.ndarray | None
    sand_frac: np.ndarray | None
    bulk_dens: np.ndarray | None
    soil_ph: np.ndarray | None
    poor_soils: np.ndarray | None
    wtp: np.ndarray | None
    wt_ab_tide: np.ndarray | None
    missing: tuple[str, ...]


def reference_case_output_dir(config_path: str | Path) -> Path:
    """Return the local reference case output directory.

    Fortran/run provenance: `fortran_run_scripts/paper_250919/Job0_bio`,
    lines 325-335, selects the paper-case run configuration and writes yearly
    restart outputs named in `configs/orchidee_man_250919.yaml`. This function
    only locates the local reference outputs used as parity truth.
    """

    config = load_case_config(config_path)
    return Path(config["paths"]["workspace_root"]) / REFERENCE_CASE_OUTPUT


def reference_restart_path(config_path: str | Path, filename: str) -> Path:
    """Resolve a named local reference restart file.

    Fortran/run provenance: `configs/orchidee_man_250919.yaml`
    `restart_and_outputs` names `driver_start.nc`, `driver_restart.nc`,
    `sechiba_start.nc`, and `sechiba_restart.nc`; this helper maps those names
    to the archived local reference case.
    """

    path = reference_case_output_dir(config_path) / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def inventory_restart_fields(path: str | Path, names: Iterable[str]) -> RestartFieldInventory:
    """List which requested variables exist in a NetCDF restart file.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_init`, lines
    1581-1601 reads `veget`, `veget_max`, and `frac_nobio`; lines 1655-1674
    read `njsc`, `clay_frac`, and `sand_frac`; lines 1791-1804 read
    `bulk_dens`, `soil_ph`, and `poor_soils`. `slowproc.f90`, lines
    1222-1239 and 1269-1271 write the corresponding SECHIBA restart fields.
    """

    wanted = tuple(names)
    with xr.open_dataset(path, decode_times=False) as ds:
        present = tuple(name for name in wanted if name in ds.variables)
    missing = tuple(name for name in wanted if name not in present)
    return RestartFieldInventory(path=Path(path), present=present, missing=missing)


def _drop_time(values: xr.DataArray) -> xr.DataArray:
    if "time" in values.dims:
        return values.isel(time=0)
    return values


def _flatten_restart_variable(values: xr.DataArray) -> np.ndarray:
    """Flatten known restart layouts to `[npts,...]`.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_init`, lines
    1581-1601 and 1655-1804, reads these fields with `restget_p(...,
    "gather", nbp_glo, index_g)`, producing `kjpindex`-major arrays. The
    local NetCDF restart stores the same fields on `time,y,x` plus optional
    class/PFT dimensions; this helper exposes the model-point axis explicitly.
    """

    values = _drop_time(values)
    dims = values.dims
    if dims == ("z_a", "y", "x"):
        arr = values.transpose("y", "x", "z_a").values
        return np.asarray(arr).reshape(-1, arr.shape[-1])
    if dims == ("z", "y", "x"):
        arr = values.transpose("y", "x", "z").values
        return np.asarray(arr).reshape(-1, arr.shape[-1])
    if dims == ("y", "x"):
        return np.asarray(values.transpose("y", "x").values).reshape(-1)
    if "y" in dims and "x" in dims:
        other_dims = tuple(dim for dim in dims if dim not in ("y", "x"))
        ordered = values.transpose("y", "x", *other_dims)
        arr = np.asarray(ordered.values)
        return arr.reshape((-1, *arr.shape[2:]))
    return np.asarray(values.values)


def remap_restart_fields_by_pft_id(
    fields: dict[str, np.ndarray],
    *,
    source_pft_layout: PFTRunLayout,
    target_pft_layout: PFTRunLayout,
    pft_axes: dict[str, int] = SECHIBA_RESTART_PFT_AXES,
) -> dict[str, np.ndarray]:
    """Remap declared restart PFT axes by stable identity."""

    return {
        name: remap_pft_axis(
            value,
            source=source_pft_layout,
            target=target_pft_layout,
            axis=pft_axes[name],
        )
        if name in pft_axes
        else np.asarray(value)
        for name, value in fields.items()
    }


def read_restart_fields(
    path: str | Path,
    names: Iterable[str],
    *,
    source_pft_layout: PFTRunLayout | None = None,
    target_pft_layout: PFTRunLayout | None = None,
) -> dict[str, np.ndarray]:
    """Read selected restart variables and normalize to model-point axes.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_init`, lines
    1581-1601, 1655-1674, and 1791-1804, reads the static fields used here.
    This helper deliberately raises on absent variables so callers cannot
    silently substitute fabricated restart truth.
    """

    names = tuple(names)
    out: dict[str, np.ndarray] = {}
    with xr.open_dataset(path, decode_times=False) as ds:
        missing = [name for name in names if name not in ds.variables]
        if missing:
            raise KeyError(f"{Path(path).name} is missing restart variables: {missing}")
        for name in names:
            out[name] = _flatten_restart_variable(ds[name])
    if (source_pft_layout is None) != (target_pft_layout is None):
        raise ValueError("source and target PFT layouts must be provided together")
    if source_pft_layout is not None and target_pft_layout is not None:
        out = remap_restart_fields_by_pft_id(
            out,
            source_pft_layout=source_pft_layout,
            target_pft_layout=target_pft_layout,
        )
    return out


def read_sechiba_static_restart(
    path: str | Path,
    *,
    source_pft_layout: PFTRunLayout | None = None,
    target_pft_layout: PFTRunLayout | None = None,
) -> SechibaStaticRestart:
    """Read exact local SECHIBA restart fields relevant to Phase 1D.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_init`, lines
    1581-1601 reads `veget`, `veget_max`, `frac_nobio`; lines 1655-1674 read
    `njsc`, `clay_frac`, `sand_frac`; lines 1791-1804 read `bulk_dens`,
    `soil_ph`, `poor_soils`. Restart write provenance is `slowproc.f90`,
    lines 1222-1239 and 1269-1271. `wtp` and `wt_ab_tide` are reference
    restart truth fields for later HYDROL alignment, not driver formulas.
    """

    inventory = inventory_restart_fields(path, SECHIBA_STATIC_FIELDS)
    fields = read_restart_fields(
        path,
        inventory.present,
        source_pft_layout=source_pft_layout,
        target_pft_layout=target_pft_layout,
    )

    def get(name: str) -> np.ndarray | None:
        return fields.get(name)

    njsc = get("njsc")
    if njsc is not None:
        njsc = np.rint(njsc).astype(np.int32)

    return SechibaStaticRestart(
        path=Path(path),
        veget=get("veget"),
        veget_max=get("veget_max"),
        frac_nobio=get("frac_nobio"),
        njsc=njsc,
        clay_frac=get("clay_frac"),
        sand_frac=get("sand_frac"),
        bulk_dens=get("bulk_dens"),
        soil_ph=get("soil_ph"),
        poor_soils=get("poor_soils"),
        wtp=get("wtp"),
        wt_ab_tide=get("wt_ab_tide"),
        missing=inventory.missing,
    )
