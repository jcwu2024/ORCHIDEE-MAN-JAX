"""Concrete routing from ``interpweight`` requests to ``aggregate_p`` owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from jax_orchidee.driver.interpolation_aggregate import aggregate_2d_p, aggregate_vec_p
from jax_orchidee.driver.interpolation_core12 import AggregatePacket, AggregateRequest


INTERPOLATION_ROUTING_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_1D lines 344-371",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_2D lines 806-836",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_4D lines 1718-1744",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_2Dcont lines 2153-2182",
    "fortran_source/ORCHIDEE/src_global/interpol_help.f90::aggregate_vec_p lines 782-828",
    "fortran_source/ORCHIDEE/src_global/interpol_help.f90::aggregate_2d_p lines 830-889",
)


@dataclass(frozen=True)
class AggregateRouter:
    """Dispatch an ``AggregateRequest`` exactly as the Fortran generic does.

    Rank-one source coordinates select ``aggregate_vec_p``. All gridded source
    ranks share the same two-dimensional mask and select ``aggregate_2d_p``.
    RegXY-only projection state is explicit rather than inferred.
    """

    grid_type: str = "RegLonLat"
    size: int = 1
    backend: Any = None
    regxy_kwargs: Mapping[str, Any] | None = None

    def __call__(self, request: AggregateRequest) -> AggregatePacket:
        nbpt = int(np.asarray(request.lalo).shape[0])
        common = {
            "nbpt": nbpt,
            "lalo": request.lalo,
            "neighbours": request.neighbours,
            "resolution": request.resolution,
            "contfrac": request.contfrac,
            "callsign": request.callsign,
            "incmax": int(request.nbvmax),
        }
        transport = {
            "grid_type": self.grid_type,
            "size": int(self.size),
            "backend": self.backend,
            "regxy_kwargs": None if self.regxy_kwargs is None else dict(self.regxy_kwargs),
        }

        if int(request.source_rank) == 1:
            longitude = np.asarray(request.longitude)
            latitude = np.asarray(request.latitude)
            if longitude.ndim != 1 or latitude.shape != longitude.shape:
                raise ValueError("rank-1 aggregate requests require matching vector coordinates")
            result = aggregate_vec_p(
                **common,
                iml=longitude.size,
                lon_rel=longitude,
                lat_rel=latitude,
                resol_lon=float(request.max_resolution_lon),
                resol_lat=float(request.max_resolution_lat),
                **transport,
            )
        else:
            longitude = np.asarray(request.longitude)
            latitude = np.asarray(request.latitude)
            if longitude.ndim != 2 or latitude.shape != longitude.shape:
                raise ValueError("gridded aggregate requests require matching rank-2 coordinates")
            if request.mask is None:
                raise ValueError("gridded aggregate requests require the shared source mask")
            result = aggregate_2d_p(
                **common,
                iml=longitude.shape[0],
                jml=longitude.shape[1],
                lon_rel=longitude,
                lat_rel=latitude,
                mask=request.mask,
                **transport,
            )

        return AggregatePacket(
            sub_index=np.asarray(result.indinc),
            sub_area=np.asarray(result.areaoverlap),
            ok=bool(result.ok),
        )


__all__ = ["AggregateRouter", "INTERPOLATION_ROUTING_PROVENANCE"]
