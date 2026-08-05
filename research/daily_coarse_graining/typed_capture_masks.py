"""Source-capability masks for typed daily Teacher captures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.typed_sidecar import TypedSidecarContract


@dataclass(frozen=True)
class CapabilityMasks:
    active_pft: np.ndarray
    vegetation_pft: np.ndarray
    leak_carbon_pft: np.ndarray
    peat_pft: np.ndarray
    soil_tile: np.ndarray
    snow: bool
    interface: bool
    routing: bool
    peat: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "active_pft": self.active_pft.tolist(),
            "vegetation_pft": self.vegetation_pft.tolist(),
            "leak_carbon_pft": self.leak_carbon_pft.tolist(),
            "peat_pft": self.peat_pft.tolist(),
            "soil_tile": self.soil_tile.tolist(),
            "snow": self.snow,
            "interface": self.interface,
            "routing": self.routing,
            "peat": self.peat,
        }


def capability_masks_from_context(context) -> CapabilityMasks:
    active = np.asarray(context.run_scalars.active_pft_mask, dtype=np.bool_)
    entries = context.run_scalars.pft_layout.entries
    if len(entries) != active.size:
        raise ValueError("PFT capability layout does not match the active-PFT axis")
    vegetation = active & np.asarray(
        ["sechiba_vegetation" in entry.capabilities for entry in entries],
        dtype=np.bool_,
    )
    leak_carbon = active & np.asarray(
        ["leak_carbon" in entry.capabilities for entry in entries],
        dtype=np.bool_,
    )
    is_peat = np.asarray(context.run_scalars.is_peat, dtype=np.bool_)
    if is_peat.shape != active.shape:
        raise ValueError("PFT peat traits do not match the active-PFT axis")
    preference = np.asarray(context.run_scalars.pref_soil_veg, dtype=np.int32)
    soil_tile = np.zeros(int(context.run_scalars.nstm), dtype=np.bool_)
    selected = preference[active] - 1
    if selected.size and (np.min(selected) < 0 or np.max(selected) >= soil_tile.size):
        raise ValueError("active PFT soil-tile preference is outside the source layout")
    soil_tile[selected] = True
    perma_peat = str(context.run_def_values.get("PERMA_PEAT", "FALSE")).strip().upper()
    if perma_peat not in {"TRUE", "FALSE"}:
        raise ValueError("PERMA_PEAT must be an explicit logical value")
    return CapabilityMasks(
        active_pft=active,
        vegetation_pft=vegetation,
        leak_carbon_pft=leak_carbon,
        peat_pft=active & is_peat & leak_carbon,
        soil_tile=soil_tile,
        snow=bool(context.ok_explicitsnow),
        interface=True,
        routing=bool(context.river_routing),
        peat=perma_peat == "TRUE",
    )


def build_defined_masks(
    values: Mapping[str, np.ndarray],
    *,
    contract: TypedSidecarContract,
    capability: CapabilityMasks,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for field in contract.fields:
        value = np.asarray(values[field.path])
        expected_shape = (value.shape[0], *field.feature_shape)
        if value.shape != expected_shape or value.dtype != np.dtype(np.float64):
            raise ValueError(f"typed capture shape/dtype drift for {field.path}")
        defined = np.ones(value.shape, dtype=np.bool_)
        for component in field.mask_components:
            if component == "finite":
                defined &= np.isfinite(value) & (np.abs(value) < 0.5e20)
                continue
            if component in {"snow", "interface", "routing", "peat"}:
                defined &= bool(getattr(capability, component))
                continue
            axis_name = "soil_tile" if component == "soil_tile" else "pft"
            if axis_name not in field.axes:
                raise ValueError(f"mask component {component} has no matching axis for {field.path}")
            axis = field.axes.index(axis_name)
            source = np.asarray(getattr(capability, component), dtype=np.bool_)
            shape = [1] * value.ndim
            shape[axis] = source.size
            defined &= source.reshape(shape)
        result[field.path] = defined
    return result
