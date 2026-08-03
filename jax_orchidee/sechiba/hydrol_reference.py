"""HYDROL restart/reference readers for validation-boundary checks.

These helpers read explicitly selected local SECHIBA restart variables and
normalize axes for HYDROL diagnostics. They do not infer timestep state, run a
HYDROL process, or claim full `hydrol_soil` parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from jax_orchidee.driver.restart import (
    reference_restart_path,
    remap_restart_fields_by_pft_id,
)
from jax_orchidee.parameters.pft_catalog import PFTRunLayout


HYDROL_RESTART_FIELDS = (
    "njsc",
    "moistc",
    "moistcl",
    "free_drain_coef",
    "veget",
    "veget_max",
    "humrel",
    "vegstress",
    "wtp",
    "fwet_new",
    "wt_ab",
    "wt_ab_tide",
    "run2peat",
    "run2man",
    "zwt_force",
    "water2infilt",
    "ae_ns",
    "evap_bare_lim_ns",
    "us",
    "resdist",
)


@dataclass(frozen=True)
class HydrolRestartInventory:
    """Presence/absence summary for selected HYDROL restart variables."""

    path: Path
    present: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class HydrolRestartAnchors:
    """Selected local HYDROL restart fields on model-point axes."""

    path: Path
    njsc: np.ndarray | None
    moistc: np.ndarray | None
    moistcl: np.ndarray | None
    free_drain_coef: np.ndarray | None
    veget: np.ndarray | None
    veget_max: np.ndarray | None
    humrel: np.ndarray | None
    vegstress: np.ndarray | None
    wtp: np.ndarray | None
    fwet_new: np.ndarray | None
    wt_ab: np.ndarray | None
    wt_ab_tide: np.ndarray | None
    run2peat: np.ndarray | None
    run2man: np.ndarray | None
    zwt_force: np.ndarray | None
    water2infilt: np.ndarray | None
    ae_ns: np.ndarray | None
    evap_bare_lim_ns: np.ndarray | None
    us: np.ndarray | None
    resdist: np.ndarray | None
    missing: tuple[str, ...]


def hydrol_reference_restart_path(config_path: str | Path, filename: str = "sechiba_start.nc") -> Path:
    """Resolve a local SECHIBA restart file for HYDROL validation anchors.

    Reference/history reader validation boundary: path lookup reuses the
    audited driver restart location helper. HYDROL restart write provenance:
    `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`, subroutine
    `hydrol_finalize`, lines 1727-1747 writes `moistc`, `moistcl`,
    `vegstress`, and `humrel`.
    """

    return reference_restart_path(config_path, filename)


def inventory_hydrol_restart_fields(path: str | Path, names: Iterable[str]) -> HydrolRestartInventory:
    """List which requested HYDROL restart variables exist in a NetCDF file.

    Reference/history reader validation boundary. Restart write provenance:
    `hydrol.f90`, subroutine `hydrol_finalize`, lines 1727-1747 for HYDROL
    moisture/stress fields; related SECHIBA restart fields are local
    validation anchors only.
    """

    wanted = tuple(names)
    with xr.open_dataset(path, decode_times=False) as ds:
        present = tuple(name for name in wanted if name in ds.variables)
    return HydrolRestartInventory(
        path=Path(path),
        present=present,
        missing=tuple(name for name in wanted if name not in present),
    )


def _drop_time(values: xr.DataArray) -> xr.DataArray:
    if "time" in values.dims:
        return values.isel(time=0)
    return values


def _flatten_hydrol_restart_variable(values: xr.DataArray) -> np.ndarray:
    """Normalize known HYDROL restart layouts to `[npts,...]`.

    Reference/history reader validation boundary. Fortran restart gather/write
    stores model-point-major fields; local NetCDF uses `time,y,x` plus optional
    tile/PFT/layer dimensions. This helper only reorders axes.
    """

    values = _drop_time(values)
    dims = values.dims
    if dims == ("y", "x"):
        return np.asarray(values.transpose("y", "x").values).reshape(-1)
    if dims == ("z_a", "y", "x"):
        arr = np.asarray(values.transpose("y", "x", "z_a").values)
        return arr.reshape(-1, arr.shape[-1])
    if dims == ("z_c", "y", "x"):
        arr = np.asarray(values.transpose("y", "x", "z_c").values)
        return arr.reshape(-1, arr.shape[-1])
    if dims == ("l_c", "z_b", "y", "x"):
        arr = np.asarray(values.transpose("y", "x", "l_c", "z_b").values)
        return arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    if dims == ("m_a", "l_c", "z_a", "y", "x"):
        arr = np.asarray(values.transpose("y", "x", "z_a", "l_c", "m_a").values)
        return arr.reshape((-1, arr.shape[-3], arr.shape[-2], arr.shape[-1]))
    if "y" in dims and "x" in dims:
        other_dims = tuple(dim for dim in dims if dim not in ("y", "x"))
        arr = np.asarray(values.transpose("y", "x", *other_dims).values)
        return arr.reshape((-1, *arr.shape[2:]))
    return np.asarray(values.values)


def read_hydrol_restart_fields(
    path: str | Path,
    names: Iterable[str],
    *,
    source_pft_layout: PFTRunLayout | None = None,
    target_pft_layout: PFTRunLayout | None = None,
) -> dict[str, np.ndarray]:
    """Read explicitly selected HYDROL restart variables.

    Reference/history reader validation boundary. Missing variables raise
    rather than being fabricated; no process formulas or defaults are applied.
    """

    wanted = tuple(names)
    out: dict[str, np.ndarray] = {}
    with xr.open_dataset(path, decode_times=False) as ds:
        missing = [name for name in wanted if name not in ds.variables]
        if missing:
            raise KeyError(f"{Path(path).name} is missing HYDROL restart variables: {missing}")
        for name in wanted:
            out[name] = _flatten_hydrol_restart_variable(ds[name])
    if (source_pft_layout is None) != (target_pft_layout is None):
        raise ValueError("source and target PFT layouts must be provided together")
    if source_pft_layout is not None and target_pft_layout is not None:
        out = remap_restart_fields_by_pft_id(
            out,
            source_pft_layout=source_pft_layout,
            target_pft_layout=target_pft_layout,
        )
    return out


def read_hydrol_restart_anchors(
    path: str | Path,
    *,
    source_pft_layout: PFTRunLayout | None = None,
    target_pft_layout: PFTRunLayout | None = None,
) -> HydrolRestartAnchors:
    """Read local HYDROL restart anchors, without timestep parity claims.

    Reference/history reader validation boundary. Fortran restart write
    provenance: `hydrol.f90`, subroutine `hydrol_finalize`, lines 1727-1747.
    """

    inventory = inventory_hydrol_restart_fields(path, HYDROL_RESTART_FIELDS)
    fields = read_hydrol_restart_fields(
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

    return HydrolRestartAnchors(
        path=Path(path),
        njsc=njsc,
        moistc=get("moistc"),
        moistcl=get("moistcl"),
        free_drain_coef=get("free_drain_coef"),
        veget=get("veget"),
        veget_max=get("veget_max"),
        humrel=get("humrel"),
        vegstress=get("vegstress"),
        wtp=get("wtp"),
        fwet_new=get("fwet_new"),
        wt_ab=get("wt_ab"),
        wt_ab_tide=get("wt_ab_tide"),
        run2peat=get("run2peat"),
        run2man=get("run2man"),
        zwt_force=get("zwt_force"),
        water2infilt=get("water2infilt"),
        ae_ns=get("ae_ns"),
        evap_bare_lim_ns=get("evap_bare_lim_ns"),
        us=get("us"),
        resdist=get("resdist"),
        missing=inventory.missing,
    )
