"""Contract-driven state weighting for recursive daily training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.markov_dataset import TrainingStatistics

STATE_PROCESS_GROUPS = (
    "stomate_carbon_flux",
    "stomate_vegetation_structure",
    "stomate_litter_turnover",
    "stomate_carbon_storage",
    "stomate_environment_phenology_memory",
    "hydrology",
    "thermal_energy",
    "surface_exchange_finalize",
)

_STOMATE_LIVE_FIELDS = frozenset(
    {
        "age",
        "biomass",
        "height",
        "ind",
        "lai",
        "leaf_age",
        "leaf_frac",
        "sla_calc",
        "veget",
        "veget_lastlight",
        "veget_max",
    }
)
_STOMATE_FLUX_FIELDS = frozenset(
    {
        "bm_to_litter",
        "co2_to_bm",
        "gpp_daily",
        "gpp_week",
        "npp_daily",
        "npp_longterm",
        "resp_growth",
        "resp_hetero",
        "resp_maint",
        "resp_maint_part",
        "turnover_daily",
        "turnover_longterm",
        "turnover_time",
    }
)
_STOMATE_LITTER_PREFIXES = ("litter", "lignin")
_STOMATE_STORAGE_PREFIXES = (
    "carbon",
    "deepC",
    "flux",
    "fuel_",
    "prod",
    "soilc",
)
_STOMATE_STORAGE_FIELDS = frozenset({"DOC", "carb_mass_total"})


@dataclass(frozen=True)
class StateProcessWeighting:
    weights: np.ndarray
    metadata: Mapping[str, Any]
    sha256: str


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state_group(component: str, field: str) -> str:
    if component == "slowproc_stomate_previous_step_state":
        if field in _STOMATE_FLUX_FIELDS:
            return "stomate_carbon_flux"
        if field in _STOMATE_LIVE_FIELDS:
            return "stomate_vegetation_structure"
        if field == "dead_leaves" or field.startswith(_STOMATE_LITTER_PREFIXES):
            return "stomate_litter_turnover"
        if field in _STOMATE_STORAGE_FIELDS or field.startswith(
            _STOMATE_STORAGE_PREFIXES
        ):
            return "stomate_carbon_storage"
        return "stomate_environment_phenology_memory"
    if component == "hydrol_previous_step_state":
        return "hydrology"
    if component in {
        "enerbil_previous_step_state",
        "thermosoil_previous_step_state",
    }:
        return "thermal_energy"
    if component in {
        "diffuco_previous_step_state",
        "driver_previous_step_state",
        "sechiba_finalize_state",
    }:
        return "surface_exchange_finalize"
    raise ValueError(f"state objective has no process owner for {component}.{field}")


def state_process_weighting_from_contract(
    contract_metadata: Mapping[str, Any],
) -> StateProcessWeighting:
    """Give every declared state process group equal total nominal weight."""

    width = int(contract_metadata["continuous_state_width"])
    occupied = np.zeros((width,), dtype=bool)
    assignments: dict[str, list[dict[str, Any]]] = {
        group: [] for group in STATE_PROCESS_GROUPS
    }
    for leaf in contract_metadata["state_leaves"]:
        if leaf["start"] is None:
            continue
        start = int(leaf["start"])
        stop = int(leaf["stop"])
        if start < 0 or stop <= start or stop > width or np.any(occupied[start:stop]):
            raise ValueError(f"invalid or overlapping state leaf [{start}:{stop}]")
        path = tuple(str(value) for value in leaf["path"])
        if not path:
            raise ValueError("continuous state leaf has no field path")
        component = str(leaf["component"])
        group = _state_group(component, path[0])
        assignments[group].append(
            {
                "key": ".".join((component, *path)),
                "start": start,
                "stop": stop,
                "width": stop - start,
            }
        )
        occupied[start:stop] = True
    if not np.all(occupied):
        missing = np.flatnonzero(~occupied)
        raise ValueError(f"state process groups miss {missing.size} scalar values")
    empty = tuple(group for group, leaves in assignments.items() if not leaves)
    if empty:
        raise ValueError(f"state process groups are empty: {empty}")

    weights = np.zeros((width,), dtype=np.float32)
    groups = []
    for group in STATE_PROCESS_GROUPS:
        leaves = assignments[group]
        group_width = sum(int(leaf["width"]) for leaf in leaves)
        scalar_weight = 1.0 / (len(STATE_PROCESS_GROUPS) * group_width)
        for leaf in leaves:
            weights[leaf["start"] : leaf["stop"]] = scalar_weight
        groups.append(
            {
                "id": group,
                "width": group_width,
                "leaf_count": len(leaves),
                "nominal_weight_share": 1.0 / len(STATE_PROCESS_GROUPS),
                "leaves": leaves,
            }
        )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("state process weights must be finite and positive")
    metadata = {
        "schema_version": "canonical_state_process_weighting_v1",
        "policy": "equal_process_group_then_equal_scalar",
        "continuous_state_width": width,
        "groups": groups,
    }
    return StateProcessWeighting(
        weights=weights,
        metadata=metadata,
        sha256=_canonical_sha256(metadata),
    )


def stabilized_state_delta_scale(
    statistics: TrainingStatistics,
    *,
    floor_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bound daily-change normalization by a fraction of state variability."""

    if not 0.0 < floor_ratio <= 1.0:
        raise ValueError("state-delta floor ratio must be in (0, 1]")
    if "state" not in statistics.arrays or "state_delta" not in statistics.arrays:
        raise ValueError("training statistics require state and state_delta arrays")
    state_scale = np.asarray(statistics.arrays["state"].scale, dtype=np.float64)
    delta_scale = np.asarray(
        statistics.arrays["state_delta"].scale, dtype=np.float64
    )
    if state_scale.shape != delta_scale.shape:
        raise ValueError("state and state-delta statistics have different shapes")
    if (
        not np.all(np.isfinite(state_scale))
        or not np.all(np.isfinite(delta_scale))
        or np.any(state_scale <= 0.0)
        or np.any(delta_scale <= 0.0)
    ):
        raise ValueError("state and state-delta scales must be finite and positive")
    floor = floor_ratio * state_scale
    stabilized = np.maximum(delta_scale, floor)
    raised = delta_scale < floor
    audit = {
        "policy": "max_empirical_delta_scale_and_fraction_of_state_scale",
        "floor_ratio": floor_ratio,
        "state_width": int(state_scale.size),
        "raised_columns": int(np.count_nonzero(raised)),
        "empirical_min": float(np.min(delta_scale)),
        "stabilized_min": float(np.min(stabilized)),
        "stabilized_max": float(np.max(stabilized)),
    }
    return stabilized, audit
