"""Contract-derived process, axis, and fast-target ownership metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.canonical_state_objective import (
    STATE_PROCESS_GROUPS,
    state_process_group,
)

TARGET_FAMILIES = (
    "driver",
    "diffuco_enerbil",
    "hydrol",
    "thermosoil",
    "daily_interface",
    "ok_leak",
    "final_diagnostics",
    "sechiba_finalize",
)

_AXIS_ROLES = {
    "npts": "landpoint",
    "nvm": "pft",
    "ngrnd": "soil_thermal_depth",
    "nslm": "soil_hydrology_depth",
    "ndeep": "deep_soil_depth",
    "nsnow": "snow_depth",
    "nstm": "soil_tile",
    "nnobio": "nonbiological_tile",
    "ncarb": "carbon_pool",
    "ndoc": "doc_pool",
    "npool": "soil_carbon_pool",
    "nlitt": "litter_pool",
    "nparts": "plant_part",
    "nelements": "chemical_element",
    "nlevs": "litter_level",
    "nage": "product_age",
    "nwp": "wood_product",
    "nleafages": "leaf_age",
    "nlai": "lai_class",
    "spectral_band_2": "spectral_band",
    "two": "pair",
    "npco2": "photosynthesis_parameter",
}
_CARBON_AXES = frozenset(
    {
        "ncarb",
        "ndoc",
        "npool",
        "nlitt",
        "nparts",
        "nelements",
        "nlevs",
        "nage",
        "nwp",
    }
)
_VERTICAL_AXES = frozenset({"ngrnd", "nslm", "ndeep", "nsnow"})
_MEMORY_AXES = frozenset({"nleafages", "nlai"})

_DAILY_INTERFACE_GROUP = {
    "flood_root_radia": "hydrology",
    "gpp_daily": "stomate_carbon_flux",
    "humrel_daily": "hydrology",
    "litterhum_daily": "stomate_litter_turnover",
    "precip_daily": "hydrology",
    "resp_maint_part": "stomate_carbon_flux",
    "resp_maint_radia": "thermal_energy",
    "snowfall_daily": "hydrology",
    "snowmass_daily": "hydrology",
    "soilhum_daily": "hydrology",
    "t2m_daily": "thermal_energy",
    "t2m_max_daily": "thermal_energy",
    "t2m_min_daily": "thermal_energy",
    "tmc_topgrass_daily": "thermal_energy",
    "tsoil_daily": "thermal_energy",
    "tsurf_daily": "thermal_energy",
    "wspeed_daily": "surface_exchange_finalize",
}


@dataclass(frozen=True)
class AxisPartition:
    encoder_kind: str
    indices: tuple[int, ...]
    leaf_keys: tuple[str, ...]


@dataclass(frozen=True)
class ProcessGroupLayout:
    id: str
    state_indices: tuple[int, ...]
    axis_partitions: tuple[AxisPartition, ...]


@dataclass(frozen=True)
class TargetFamilyLayout:
    id: str
    target_indices: tuple[int, ...]
    source_process_ids: tuple[str, ...]


@dataclass(frozen=True)
class DailyProcessAxisLayout:
    state_width: int
    target_width: int
    process_groups: tuple[ProcessGroupLayout, ...]
    target_families: tuple[TargetFamilyLayout, ...]
    metadata: Mapping[str, Any]
    sha256: str


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _leaf_key(item: Mapping[str, Any]) -> str:
    component = item.get("component")
    prefix = str(item["family"]) if component is None else str(component)
    return ".".join((prefix, *(str(value) for value in item["path"])))


def _axis_encoder_kind(item: Mapping[str, Any]) -> str:
    axes = tuple(str(value) for value in item.get("axis_names", ()))
    unknown = tuple(axis for axis in axes if axis not in _AXIS_ROLES)
    if unknown:
        raise ValueError(f"contract leaf has unknown axes {unknown}: {_leaf_key(item)}")
    axis_set = frozenset(axes)
    if axis_set & _CARBON_AXES:
        return "carbon_factorized"
    if axis_set & _VERTICAL_AXES:
        return "ordered_vertical"
    if axis_set & _MEMORY_AXES:
        return "ordered_memory"
    if "nvm" in axis_set:
        return "pft_shared"
    shape = tuple(int(value) for value in item["shape"])
    return "scalar_memory" if int(np.prod(shape, dtype=np.int64)) == 1 else "tensor"


def _target_leaf_process_group(item: Mapping[str, Any]) -> str:
    family = str(item["family"])
    component = item.get("component")
    field = str(item["path"][0])
    if component is not None:
        return state_process_group(str(component), field)
    if family == "ok_leak":
        return state_process_group("slowproc_stomate_previous_step_state", field)
    if family == "daily_interface":
        try:
            return _DAILY_INTERFACE_GROUP[field]
        except KeyError as error:
            raise ValueError(
                f"daily interface has no process owner for {field}"
            ) from error
    if family == "final_diagnostics":
        return "thermal_energy"
    raise ValueError(f"target leaf has no process owner: {_leaf_key(item)}")


def _validated_range(
    item: Mapping[str, Any],
    *,
    width: int,
    occupied: np.ndarray,
    label: str,
) -> tuple[int, int]:
    start = int(item["start"])
    stop = int(item["stop"])
    shape = tuple(int(value) for value in item["shape"])
    if (
        start < 0
        or stop <= start
        or stop > width
        or stop - start != int(np.prod(shape, dtype=np.int64))
        or np.any(occupied[start:stop])
    ):
        raise ValueError(f"invalid or overlapping {label} leaf [{start}:{stop}]")
    occupied[start:stop] = True
    return start, stop


def process_axis_layout_from_contract(
    contract_metadata: Mapping[str, Any],
) -> DailyProcessAxisLayout:
    """Build an exhaustive, hash-bound neural layout from Contract v5 metadata."""

    state_width = int(contract_metadata["continuous_state_width"])
    target_width = int(contract_metadata["fast_day_target_width"])
    state_occupied = np.zeros((state_width,), dtype=bool)
    target_occupied = np.zeros((target_width,), dtype=bool)
    state_assignments = {
        group: {} for group in STATE_PROCESS_GROUPS
    }
    state_leaves = []
    for item in contract_metadata["state_leaves"]:
        if item.get("start") is None:
            continue
        start, stop = _validated_range(
            item,
            width=state_width,
            occupied=state_occupied,
            label="state",
        )
        component = str(item["component"])
        field = str(item["path"][0])
        group = state_process_group(component, field)
        kind = _axis_encoder_kind(item)
        key = _leaf_key(item)
        bucket = state_assignments[group].setdefault(
            kind, {"indices": [], "leaf_keys": []}
        )
        bucket["indices"].extend(range(start, stop))
        bucket["leaf_keys"].append(key)
        state_leaves.append(
            {
                "key": key,
                "process_group": group,
                "encoder_kind": kind,
                "start": start,
                "stop": stop,
                "shape": [int(value) for value in item["shape"]],
                "axis_names": [str(value) for value in item.get("axis_names", ())],
                "axis_roles": [
                    _AXIS_ROLES[str(value)]
                    for value in item.get("axis_names", ())
                ],
            }
        )
    if not np.all(state_occupied):
        raise ValueError("process-axis layout does not cover continuous state")

    target_assignments = {family: [] for family in TARGET_FAMILIES}
    target_sources = {family: set() for family in TARGET_FAMILIES}
    target_leaves = []
    for item in contract_metadata["fast_day_target_leaves"]:
        family = str(item["family"])
        if family not in target_assignments:
            raise ValueError(f"unsupported fast-day target family {family!r}")
        start, stop = _validated_range(
            item,
            width=target_width,
            occupied=target_occupied,
            label="fast-day target",
        )
        group = _target_leaf_process_group(item)
        kind = _axis_encoder_kind(item)
        key = _leaf_key(item)
        target_assignments[family].extend(range(start, stop))
        target_sources[family].add(group)
        target_leaves.append(
            {
                "key": key,
                "family": family,
                "source_process_group": group,
                "encoder_kind": kind,
                "start": start,
                "stop": stop,
                "shape": [int(value) for value in item["shape"]],
                "axis_names": [str(value) for value in item.get("axis_names", ())],
                "axis_roles": [
                    _AXIS_ROLES[str(value)]
                    for value in item.get("axis_names", ())
                ],
            }
        )
    if not np.all(target_occupied):
        raise ValueError("process-axis layout does not cover fast-day target")

    process_groups = []
    process_metadata = []
    for group in STATE_PROCESS_GROUPS:
        partitions = state_assignments[group]
        if not partitions:
            raise ValueError(f"process-axis state group is empty: {group}")
        axis_partitions = tuple(
            AxisPartition(
                encoder_kind=kind,
                indices=tuple(int(value) for value in payload["indices"]),
                leaf_keys=tuple(str(value) for value in payload["leaf_keys"]),
            )
            for kind, payload in sorted(partitions.items())
        )
        state_indices = tuple(
            index for partition in axis_partitions for index in partition.indices
        )
        process_groups.append(
            ProcessGroupLayout(group, state_indices, axis_partitions)
        )
        process_metadata.append(
            {
                "id": group,
                "state_width": len(state_indices),
                "axis_partitions": [
                    {
                        "encoder_kind": partition.encoder_kind,
                        "width": len(partition.indices),
                        "leaf_keys": list(partition.leaf_keys),
                    }
                    for partition in axis_partitions
                ],
            }
        )

    target_families = []
    target_metadata = []
    for family in TARGET_FAMILIES:
        indices = tuple(int(value) for value in target_assignments[family])
        if not indices:
            raise ValueError(f"fast-day target family is empty: {family}")
        source_ids = tuple(
            group for group in STATE_PROCESS_GROUPS if group in target_sources[family]
        )
        if not source_ids:
            raise ValueError(f"fast-day target family has no process source: {family}")
        target_families.append(TargetFamilyLayout(family, indices, source_ids))
        target_metadata.append(
            {
                "id": family,
                "target_width": len(indices),
                "source_process_ids": list(source_ids),
            }
        )

    metadata = {
        "schema_version": "daily_process_axis_layout_v1",
        "continuous_state_width": state_width,
        "fast_day_target_width": target_width,
        "process_groups": process_metadata,
        "target_families": target_metadata,
        "state_leaves": state_leaves,
        "target_leaves": target_leaves,
    }
    return DailyProcessAxisLayout(
        state_width=state_width,
        target_width=target_width,
        process_groups=tuple(process_groups),
        target_families=tuple(target_families),
        metadata=metadata,
        sha256=_canonical_sha256(metadata),
    )
