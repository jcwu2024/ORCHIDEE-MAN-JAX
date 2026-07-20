"""Source-routed STICS crop-cycle initialization.

Fortran source truth: ``src_sticslai/Stics_init.f90::Stics_init`` lines
4-533.  The routine is an in-place Fortran state transition; this module
returns the corresponding immutable JAX writeback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SticsInitPftParameters:
    """PFT arrays consumed by ``Stics_init`` lines 380-394, 434, and 441-525."""

    ok_laidev: object
    sp_codlainet: object
    sp_stpltger: object
    sp_stamflax: object
    sp_stlaxsen: object
    sp_stsenlan: object
    sp_stlevdrp: object
    sp_stflodrp: object
    sp_stdrpmat: object
    sp_stdrpdes: object
    sp_stlevamf: object
    sp_nbox: object
    sp_vlaimax: object


class SticsInitResult(NamedTuple):
    """Complete state writeback from ``Stics_init``."""

    state: dict[str, jnp.ndarray]
    f_crop_init: bool
    f_crop_recycle: jnp.ndarray


# Lines 332-490: every ordinary (kjpindex,nvm) state assignment.
STICS_ZERO_REAL_FIELDS = (
    "nsendltams",
    "nsendltai",
    "nsenpfeuilverte",
    "nsendurvie",
    "nsenndurvie",
    "densiteequiv",
    "ssla",
    "pfeuilverte",
    "bsenlai",
    "zrac",
    "tcult",
    "udevair",
    "udevcult",
    "rfvi",
    "caljvc",
    "rfpi",
    "upvt",
    "utp",
    "somcour",
    "somcourdrp",
    "somcourutp",
    "tdevelop",
    "somtemp",
    "somcourfauche",
    "group",
    "densite",
    "densitelev",
    "densiteger",
    "somelong",
    "somger",
    "somtemphumec",
    "stpltlev",
    "stmatrec",
    "slai",
    "somfeuille",
    "pdlai",
    "reajust",
    "ulai",
    "pdulai",
    "efdensite",
    "tempeff",
    "deltai",
    "svmax",
    "laisen",
    "pdlaisen",
    "dltaisenat",
    "dltamsen",
    "dltaisen",
    "fgellev",
    "fstressgel",
    "durvieI",
    "durvie",
    "somsenreste",
    "shumrel",
    "mafeuiljaune",
    "msneojaune",
    "pdircarb",
    "ircarb",
    "nbgrains",
    "pgrain",
    "vitmoy",
    "nbgraingel",
    "pgraingel",
    "dltags",
    "ftempremp",
    "magrain",
    "pdmagrain",
    "pdsfruittot",
    "repracmax",
    "repracmin",
    "kreprac",
    "somtemprac",
    "urac",
    "reprac",
    "c_reserve",
    "c_leafb",
    "deltgrain",
)

STICS_ONE_REAL_FIELDS = (
    "tursla",
    "coeflev",
    "tustress",
    "swfac",
    "turfac",
    "senfac",
    "fgelflo",
)

STICS_ZERO_INTEGER_FIELDS = (
    "nplt",
    "nrec",
    "nlan",
    "ndrp",
    "nlev",
    "nger",
    "nflo",
    "nmat",
    "nlax",
    "ndebdes",
    "nbjhumec",
    "namf",
    "nbfeuille",
    "nstopfeuille",
    "nsen",
    "nsencour",
    "dernier_n",
    "ndebsen",
    "nbj0remp",
    "nstoprac",
    "gslen",
    "drylen",
)

STICS_FALSE_FIELDS = ("in_cycle", "etatvernal", "humectation", "gelee")
STICS_TRUE_FIELDS = ("f_sen_lai", "onarretesomcourdrp")

_PARAMETER_WRITES = {
    "stpltger": "sp_stpltger",
    "R_stamflax": "sp_stamflax",
    "R_stlaxsen": "sp_stlaxsen",
    "R_stsenlan": "sp_stsenlan",
    "R_stlevdrp": "sp_stlevdrp",
    "R_stflodrp": "sp_stflodrp",
    "R_stdrpmat": "sp_stdrpmat",
    "R_stdrpdes": "sp_stdrpdes",
    "R_stlevamf": "sp_stlevamf",
}

_SPECIAL_2D_FIELDS = (
    "nrecbutoir",
    "stlevflo",
    *_PARAMETER_WRITES,
)

_ALL_2D_FIELDS = (
    *STICS_ZERO_REAL_FIELDS,
    *STICS_ONE_REAL_FIELDS,
    *STICS_ZERO_INTEGER_FIELDS,
    *STICS_FALSE_FIELDS,
    *STICS_TRUE_FIELDS,
    *_SPECIAL_2D_FIELDS,
)


def _parameter_arrays(parameters: SticsInitPftParameters, nvm: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in parameters.__dataclass_fields__:
        dtype = bool if name == "ok_laidev" else None
        value = np.asarray(getattr(parameters, name), dtype=dtype)
        if value.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,), got {value.shape}")
        arrays[name] = value
    if not np.issubdtype(arrays["sp_codlainet"].dtype, np.integer):
        raise TypeError("sp_codlainet must be an integer PFT array")
    if not np.issubdtype(arrays["sp_nbox"].dtype, np.integer):
        raise TypeError("sp_nbox must be an integer PFT array")
    return arrays


def _masked_set(array: jnp.ndarray, mask: jnp.ndarray, value) -> jnp.ndarray:
    return jnp.where(mask, jnp.asarray(value, dtype=array.dtype), array)


def stics_init_source_routed(
    *,
    state: Mapping[str, object],
    f_crop_init: bool,
    f_crop_recycle,
    parameters: SticsInitPftParameters,
) -> SticsInitResult:
    """Execute ``Stics_init`` with source-owned selection and writeback.

    Provenance: ``Stics_init.f90::Stics_init`` lines 325-502 applies the
    ``f_crop_init OR f_crop_recycle(ip,j)`` mask and initializes crop state;
    lines 441-452 reset box histories for the global LAIdev/codlainet-3 arm;
    lines 492-497 reset codlainet-2 histories; lines 505-531 construct
    ``box_ulai`` from ``SP_nbox`` and ``SP_vlaimax``.  All arrays retain their
    caller-provided ``nvm`` dimension.  PFT14 selection, when needed, remains
    a caller-side zero-based index selection and does not determine shape.
    """

    recycle = jnp.asarray(f_crop_recycle, dtype=bool)
    if recycle.ndim != 2:
        raise ValueError("f_crop_recycle must have shape (kjpindex,nvm)")
    npts, nvm = recycle.shape
    params = _parameter_arrays(parameters, nvm)
    missing = sorted(set((*_ALL_2D_FIELDS, "v_dltams", "histgrowthset", "hist_sencourset", "hist_latestset", "doyhiststset", "box_ndays", "box_lai", "box_lairem", "box_tdev", "box_biom", "box_biomrem", "box_durage", "box_somsenbase", "box_ulai")) - set(state))
    if missing:
        raise ValueError(f"STICS initialization state is missing {tuple(missing)}")

    out = {name: jnp.asarray(value) for name, value in state.items()}
    for name in _ALL_2D_FIELDS:
        if out[name].shape != (npts, nvm):
            raise ValueError(f"{name} must have shape {(npts, nvm)}, got {out[name].shape}")

    if out["v_dltams"].shape != (npts, nvm, 60):
        raise ValueError("v_dltams must have source shape (kjpindex,nvm,60)")
    if out["histgrowthset"].shape != (npts, nvm, 300, 5):
        raise ValueError("histgrowthset must have source shape (kjpindex,nvm,300,5)")
    for name in ("hist_sencourset", "hist_latestset", "doyhiststset"):
        if out[name].shape != (npts, nvm):
            raise ValueError(f"{name} must have shape {(npts, nvm)}")

    box_ulai = out["box_ulai"]
    if box_ulai.ndim != 2 or box_ulai.shape[0] != nvm:
        raise ValueError("box_ulai must have shape (nvm,nboxmax)")
    nboxmax = box_ulai.shape[1]
    box_fields = (
        "box_ndays",
        "box_lai",
        "box_lairem",
        "box_tdev",
        "box_biom",
        "box_biomrem",
        "box_durage",
        "box_somsenbase",
    )
    for name in box_fields:
        if out[name].shape != (npts, nvm, nboxmax):
            raise ValueError(f"{name} must have shape {(npts, nvm, nboxmax)}")
    if np.any(params["sp_nbox"] < 0) or np.any(params["sp_nbox"] > nboxmax):
        raise ValueError("SP_nbox must remain within 0:nboxmax")

    selected = jnp.full((npts, nvm), bool(f_crop_init)) | recycle
    for name in STICS_ZERO_REAL_FIELDS + STICS_ZERO_INTEGER_FIELDS + STICS_FALSE_FIELDS:
        out[name] = _masked_set(out[name], selected, 0)
    for name in STICS_ONE_REAL_FIELDS + STICS_TRUE_FIELDS:
        out[name] = _masked_set(out[name], selected, 1)
    out["nrecbutoir"] = _masked_set(out["nrecbutoir"], selected, 999)

    for target, source in _PARAMETER_WRITES.items():
        values = jnp.asarray(params[source], dtype=out[target].dtype)[None, :]
        out[target] = _masked_set(out[target], selected, values)
    stlevflo = params["sp_stlevdrp"] - params["sp_stflodrp"]
    out["stlevflo"] = _masked_set(
        out["stlevflo"], selected, jnp.asarray(stlevflo)[None, :]
    )

    selected_3d = selected[:, :, None]
    out["v_dltams"] = _masked_set(out["v_dltams"], selected_3d, 0)
    any_laidev = bool(np.any(params["ok_laidev"]))
    reset_boxes = any_laidev and bool(np.any(params["sp_codlainet"] == 3))
    reset_history = any_laidev and bool(np.any(params["sp_codlainet"] == 2))
    if reset_boxes:
        for name in box_fields:
            out[name] = _masked_set(out[name], selected_3d, 0)
    if reset_history:
        out["histgrowthset"] = _masked_set(
            out["histgrowthset"], selected[:, :, None, None], 0
        )
        for name in ("hist_sencourset", "hist_latestset", "doyhiststset"):
            out[name] = _masked_set(out[name], selected, 0)

    recycle_out = jnp.where(selected, False, recycle)

    if reset_boxes:
        # Lines 505-525 reset all PFT rows, independently of the per-cell mask.
        box_ulai = jnp.zeros_like(box_ulai)
        for pft in range(nvm):
            nbox = int(params["sp_nbox"][pft])
            if nbox < 2:
                continue
            uinflex = params["sp_vlaimax"][pft]
            mid = int(np.floor(float(nbox) / 2.0))
            if mid < 1:
                mid = 1
            elif mid == nbox:
                mid = nbox - 1
            lower = 1.0 + np.arange(mid) * ((uinflex - 1.0) / float(mid))
            upper = uinflex + np.arange(nbox - mid) * (
                (3.0 - uinflex) / float(nbox - mid)
            )
            box_ulai = box_ulai.at[pft, :nbox].set(
                jnp.asarray(np.concatenate((lower, upper)), dtype=box_ulai.dtype)
            )
        out["box_ulai"] = box_ulai

    return SticsInitResult(out, bool(f_crop_init), recycle_out)
