"""Exact vertical geometry from ``vertical_soil_init``."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Mapping

import numpy as np


class VerticalSoilParameterError(ValueError):
    """Fatal or non-representable ``vertical_soil_init`` configuration."""


@dataclass(frozen=True)
class VerticalSoilGeometry:
    """Dynamically sized arrays allocated by the Fortran initializer."""

    depth_max_t: float
    depth_max_h: float
    depth_topthickness: float
    depth_cstthickness: float
    depth_geom: float
    ratio_geom_below: float
    nslm: int
    ngrnd: int
    znh: np.ndarray
    dnh: np.ndarray
    dlh: np.ndarray
    zlh: np.ndarray
    znt: np.ndarray
    dlt: np.ndarray
    zlt: np.ndarray


def _normalized(overrides: Mapping[str, object] | None) -> dict[str, object]:
    return {} if overrides is None else {str(key).strip().upper(): value for key, value in overrides.items()}


def _read(values: Mapping[str, object], key: str, default: float) -> float:
    if key not in values:
        return float(default)
    parsed = float(str(values[key]).strip().replace("D", "E").replace("d", "e"))
    if not np.isfinite(parsed):
        raise ValueError(f"{key} must be a finite Fortran real")
    return parsed


def vertical_soil_init(
    overrides: Mapping[str, object] | None = None,
    *,
    nblayermax: int = 500,
) -> VerticalSoilGeometry:
    """Read vertical parameters and construct source-identical dynamic grids.

    Fortran provenance: ``src_parameters/vertical_soil.f90``, subroutine
    ``vertical_soil_init``, lines 144-457. ``REFINEBOTTOM`` is hard-coded false
    at lines 225-236, so its unfinished body is intentionally unreachable.
    """

    values = _normalized(overrides)
    try:
        nblayermax = operator.index(nblayermax)
    except TypeError as exc:
        raise TypeError("nblayermax must be an integer") from exc
    if nblayermax < 3:
        raise VerticalSoilParameterError("nblayermax must provide the three temporary extension levels")

    # Lines 175-263: defaults are assigned immediately before getin_p.
    zmaxt = _read(values, "DEPTH_MAX_T", 38.0)
    zmaxh = _read(values, "DEPTH_MAX_H", 2.0)
    if zmaxh > zmaxt:
        raise VerticalSoilParameterError("DEPTH_MAX_H must not exceed DEPTH_MAX_T")
    top = _read(values, "DEPTH_TOPTHICK", 9.77517107e-4)
    cst = _read(values, "DEPTH_CSTTHICK", zmaxh)
    if cst != zmaxh and cst > zmaxh / 2.0:
        cst = zmaxh  # level-2 ipslerr correction at lines 219-223
    geom = _read(values, "DEPTH_GEOM", zmaxh)
    if geom < zmaxh:
        raise VerticalSoilParameterError("DEPTH_GEOM must not be shallower than DEPTH_MAX_H")
    ratio_below = _read(values, "RATIO_GEOM_BELOW", 1.05)

    # Arrays retain Fortran's one-based subscripts to keep MINLOC behavior exact.
    ztmp = np.zeros(nblayermax + 2, dtype=np.float64)
    dtmp = np.zeros(nblayermax + 2, dtype=np.float64)
    for i in range(1, nblayermax + 1):
        if ztmp[i] < cst:
            try:
                ztmp[i + 1] = top * 2.0 * (float(2**i) - 1.0)
            except OverflowError as exc:
                raise VerticalSoilParameterError("hydrology geometry does not close within nblayermax") from exc
            dtmp[i + 1] = ztmp[i + 1] - ztmp[i]
        else:
            ztmp[i + 1] = ztmp[i] + dtmp[i]
            dtmp[i + 1] = dtmp[i]

    # REFINEBOTTOM=.FALSE.; nbrefine remains one (lines 285-329).
    nslm = int(np.argmin(np.abs(ztmp[1 : nblayermax + 2] - zmaxh))) + 1
    if nslm < 2:
        raise VerticalSoilParameterError("vertical_soil_init produced fewer than two hydrology nodes")

    znh = ztmp[1 : nslm + 1].copy()
    dnh = dtmp[1 : nslm + 1].copy()
    znh[-1] = zmaxh
    dnh[-1] = ztmp[nslm] - ztmp[nslm - 1]
    dlh = np.empty(nslm, dtype=np.float64)
    dlh[:-1] = (dnh[:-1] + dnh[1:]) / 2.0
    dlh[-1] = dnh[-1] / 2.0

    ntmp = nslm
    ztmp.fill(0.0)
    ztmp[1] = top / 2.0
    ztmp[2 : ntmp + 1] = znh[1:ntmp]
    hh = dnh[ntmp - 1] / 2.0
    ztmp[ntmp] = ztmp[ntmp] - hh / 2.0
    ztmp[ntmp + 1] = ztmp[ntmp] + hh * 1.5
    ztmp[ntmp + 2] = ztmp[ntmp + 1] + hh * 2.0
    ntmp += 2
    for i in range(ntmp, nblayermax + 1):
        ratio = 1.0 if ztmp[i] < geom else ratio_below
        ztmp[i + 1] = ztmp[i] + ratio * (ztmp[i] - ztmp[i - 1])

    zint = np.zeros(nblayermax + 2, dtype=np.float64)
    zint[1] = top
    zint[2:nblayermax] = (ztmp[2:nblayermax] + ztmp[3 : nblayermax + 1]) / 2.0
    zint[nslm - 1] = (znh[nslm - 2] + znh[nslm - 1]) / 2.0
    zint[nslm] = znh[nslm - 1]
    zint[nblayermax] = ztmp[nblayermax] + (ztmp[nblayermax] - ztmp[nblayermax - 1]) / 2.0

    ngrnd = int(np.argmin(np.abs(zint[1 : nblayermax + 2] - zmaxt))) + 1
    if ngrnd < nslm or ngrnd < 2:
        raise VerticalSoilParameterError("thermal geometry cannot cover the hydrology grid")
    znt = ztmp[1 : ngrnd + 1].copy()
    zlt = zint[1 : ngrnd + 1].copy()
    dlt = np.empty(ngrnd, dtype=np.float64)
    dlt[0] = zint[1]
    dlt[1:] = zint[2 : ngrnd + 1] - zint[1:ngrnd]
    zlh = zlt[:nslm].copy()
    zlt[-1] = zmaxt
    dlt[-1] = zmaxt - zint[ngrnd - 1]

    return VerticalSoilGeometry(
        depth_max_t=zmaxt,
        depth_max_h=zmaxh,
        depth_topthickness=top,
        depth_cstthickness=cst,
        depth_geom=geom,
        ratio_geom_below=ratio_below,
        nslm=nslm,
        ngrnd=ngrnd,
        znh=znh,
        dnh=dnh,
        dlh=dlh,
        zlh=zlh,
        znt=znt,
        dlt=dlt,
        zlt=zlt,
    )


__all__ = ["VerticalSoilGeometry", "VerticalSoilParameterError", "vertical_soil_init"]
