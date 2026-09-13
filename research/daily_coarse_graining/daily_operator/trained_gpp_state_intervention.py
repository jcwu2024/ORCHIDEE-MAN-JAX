"""State partitions and metrics for trained GPP carry interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_structured_gpp import DIRECT_STATE_KEYS

DIRECT_GROUP_NAMES = (
    "direct_assim_param",
    "direct_temp_growth",
    "direct_humrel",
    "direct_qsintveg",
    "direct_control_salinity",
    "direct_control_inudate",
)
CANOPY_COVER_LEAF_KEYS = (
    "slowproc_stomate_previous_step_state.lai",
    "slowproc_stomate_previous_step_state.veget",
    "slowproc_stomate_previous_step_state.veget_max",
    "slowproc_stomate_previous_step_state.veget_lastlight",
    "slowproc_stomate_previous_step_state.leaf_age",
    "slowproc_stomate_previous_step_state.leaf_frac",
    "slowproc_stomate_previous_step_state.frac_age",
)
CANOPY_ORGANIC_CARBON_KEY = (
    "slowproc_stomate_previous_step_state.interception_storage"
)
DOC_INVENTORY_KEY = "slowproc_stomate_previous_step_state.DOC"
LITTER_INVENTORY_KEYS = (
    "slowproc_stomate_previous_step_state.litter",
    "slowproc_stomate_previous_step_state.litter_above",
    "slowproc_stomate_previous_step_state.litter_below",
)
LITTER_METADATA_KEYS = (
    "slowproc_stomate_previous_step_state.litterpart",
    "slowproc_stomate_previous_step_state.lignin_struc",
    "slowproc_stomate_previous_step_state.lignin_struc_above",
    "slowproc_stomate_previous_step_state.lignin_struc_below",
    "slowproc_stomate_previous_step_state.dead_leaves",
    "slowproc_stomate_previous_step_state.fuel_1hr",
    "slowproc_stomate_previous_step_state.fuel_10hr",
    "slowproc_stomate_previous_step_state.fuel_100hr",
    "slowproc_stomate_previous_step_state.fuel_1000hr",
)

DIFFUCO_COMPONENT = "diffuco_previous_step_state"
ENERBIL_COMPONENT = "enerbil_previous_step_state"
HYDROL_COMPONENT = "hydrol_previous_step_state"
THERMOSOIL_COMPONENT = "thermosoil_previous_step_state"
SLOWPROC_COMPONENT = "slowproc_stomate_previous_step_state"
DRIVER_COMPONENT = "driver_previous_step_state"


@dataclass(frozen=True)
class StateInterventionGroup:
    """One nonoverlapping set of continuous Markov-state coordinates."""

    name: str
    role: str
    state_keys: tuple[str, ...]
    mask: np.ndarray

    @property
    def coordinate_count(self) -> int:
        return int(np.count_nonzero(self.mask))


def build_gpp_state_intervention_groups(
    markov_contract: Any,
) -> tuple[StateInterventionGroup, ...]:
    """Partition all continuous state around direct and indirect GPP routes."""

    leaves = tuple(leaf for leaf in markov_contract.state_leaves if not leaf.discrete)
    by_key = {str(leaf.key): leaf for leaf in leaves}
    if len(by_key) != len(leaves):
        raise ValueError("continuous state leaf keys are not unique")
    required = (
        set(DIRECT_STATE_KEYS)
        | set(CANOPY_COVER_LEAF_KEYS)
        | set(LITTER_INVENTORY_KEYS)
        | set(LITTER_METADATA_KEYS)
        | {CANOPY_ORGANIC_CARBON_KEY, DOC_INVENTORY_KEY}
    )
    missing = sorted(required - set(by_key))
    if missing:
        raise ValueError(f"GPP intervention state owners are missing: {missing}")

    assignments: list[tuple[str, str, tuple[str, ...]]] = [
        (name, "direct_structured_gpp_input", (key,))
        for name, key in zip(DIRECT_GROUP_NAMES, DIRECT_STATE_KEYS, strict=True)
    ]
    assigned = set(DIRECT_STATE_KEYS)

    assignments.append(
        ("canopy_cover_leaf", "indirect_cover_and_leaf_context", CANOPY_COVER_LEAF_KEYS)
    )
    assigned.update(CANOPY_COVER_LEAF_KEYS)
    assignments.append(
        (
            "canopy_organic_carbon",
            "indirect_canopy_carbon_inventory",
            (CANOPY_ORGANIC_CARBON_KEY,),
        )
    )
    assigned.add(CANOPY_ORGANIC_CARBON_KEY)
    assignments.append(
        ("doc_inventory", "indirect_conditioned_doc_inventory", (DOC_INVENTORY_KEY,))
    )
    assigned.add(DOC_INVENTORY_KEY)
    assignments.append(
        ("litter_inventory", "indirect_litter_carbon_inventory", LITTER_INVENTORY_KEYS)
    )
    assigned.update(LITTER_INVENTORY_KEYS)
    assignments.append(
        ("litter_metadata", "indirect_litter_quality_and_fuel_context", LITTER_METADATA_KEYS)
    )
    assigned.update(LITTER_METADATA_KEYS)

    def unassigned_components(*components: str) -> tuple[str, ...]:
        selected = {
            leaf.key
            for leaf in leaves
            if leaf.component in components and leaf.key not in assigned
        }
        return tuple(leaf.key for leaf in leaves if leaf.key in selected)

    remaining_groups = (
        (
            "remaining_hydrology",
            "indirect_water_and_stress_context",
            unassigned_components(HYDROL_COMPONENT),
        ),
        (
            "energy_thermal",
            "indirect_energy_and_thermal_context",
            unassigned_components(ENERBIL_COMPONENT, THERMOSOIL_COMPONENT),
        ),
    )
    for name, role, keys in remaining_groups:
        if not keys:
            raise ValueError(f"GPP intervention group is empty: {name}")
        assignments.append((name, role, keys))
        assigned.update(keys)

    surface_boundary_keys = tuple(
        leaf.key
        for leaf in leaves
        if leaf.key not in assigned and leaf.component != SLOWPROC_COMPONENT
    )
    if not surface_boundary_keys:
        raise ValueError("remaining surface/driver/boundary intervention group is empty")
    assignments.append(
        (
            "remaining_surface_driver_boundary",
            "indirect_surface_driver_and_finalize_context",
            surface_boundary_keys,
        )
    )
    assigned.update(surface_boundary_keys)
    slow_keys = unassigned_components(SLOWPROC_COMPONENT)
    if not slow_keys:
        raise ValueError("remaining slow-process intervention group is empty")
    assignments.append(
        ("remaining_slow_process", "indirect_slow_process_context", slow_keys)
    )
    assigned.update(slow_keys)

    unassigned = sorted(set(by_key) - assigned)
    if unassigned:
        raise ValueError(f"continuous state owners are outside the intervention registry: {unassigned}")

    width = int(markov_contract.continuous_state_width)
    coverage = np.zeros((width,), dtype=np.int16)
    groups = []
    for name, role, keys in assignments:
        mask = np.zeros((width,), dtype=np.bool_)
        for key in keys:
            leaf = by_key[key]
            start, stop = leaf.start, leaf.stop
            if start is None or stop is None or not (0 <= start < stop <= width):
                raise ValueError(f"invalid continuous state slice for {key}")
            mask[start:stop] = True
        if not np.any(mask):
            raise ValueError(f"GPP intervention group has no coordinates: {name}")
        coverage += mask.astype(np.int16)
        groups.append(StateInterventionGroup(name, role, tuple(keys), mask))
    if np.any(coverage != 1):
        raise ValueError("GPP intervention groups do not partition continuous state exactly once")
    return tuple(groups)


def build_litter_detail_intervention_groups(markov_contract: Any) -> tuple[StateInterventionGroup, ...]:
    """Fixed follow-up contrasts; overlapping unions are intentional here."""
    original = {group.name: group for group in build_gpp_state_intervention_groups(markov_contract)}
    leaves = {leaf.key: leaf for leaf in markov_contract.state_leaves if not leaf.discrete}
    keys = LITTER_METADATA_KEYS
    selections = [(f"leaf_{key.rsplit('.', 1)[1]}", (key,)) for key in keys]
    selections.extend([
        ("litter_metadata", keys),
        ("doc_inventory", (DOC_INVENTORY_KEY,)),
        ("litter_metadata_and_doc", (*keys, DOC_INVENTORY_KEY)),
        ("lignin_family", tuple(key for key in keys if ".lignin_struc" in key)),
        ("fuel_family", tuple(key for key in keys if ".fuel_" in key)),
        ("outside_litter_and_doc", tuple(key for key in leaves if key not in {*keys, DOC_INVENTORY_KEY})),
    ])
    groups = []
    for name, selected in selections:
        mask = np.zeros(markov_contract.continuous_state_width, dtype=bool)
        for key in selected:
            leaf = leaves[key]
            mask[leaf.start:leaf.stop] = True
        groups.append(StateInterventionGroup(name, "followup_leaf_or_declared_union", tuple(selected), mask))
    # Same 15 groups / 32 arms as the admitted diagnostic: only input masks change.
    assert len(groups) == 15
    assert np.array_equal(groups[9].mask, original["litter_metadata"].mask)
    return tuple(groups)


def build_intervention_arm_states(
    free_state: Any,
    teacher_state: Any,
    groups: Sequence[StateInterventionGroup],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build matched rescue and reciprocal-damage states for one day."""

    free = np.asarray(free_state, dtype=np.float64)
    teacher = np.asarray(teacher_state, dtype=np.float64)
    if free.ndim != 1 or teacher.shape != free.shape:
        raise ValueError("free and Teacher intervention states must be equal-width vectors")
    names = ["control_free", "control_teacher"]
    states = [free, teacher]
    for group in groups:
        if group.mask.shape != free.shape:
            raise ValueError(f"state mask width differs for {group.name}")
        names.extend((f"rescue__{group.name}", f"damage__{group.name}"))
        states.extend(
            (
                np.where(group.mask, teacher, free),
                np.where(group.mask, free, teacher),
            )
        )
    return tuple(names), np.stack(states)


def series_error_metrics(
    predicted: Any,
    teacher: Any,
    *,
    predicted_defined: Any | None = None,
    teacher_defined: Any | None = None,
) -> Mapping[str, Any]:
    """Summarize one scalar daily series without filtering defined nonfinites."""

    predicted_array = np.asarray(predicted, dtype=np.float64).reshape(-1)
    teacher_array = np.asarray(teacher, dtype=np.float64).reshape(-1)
    if predicted_array.shape != teacher_array.shape:
        raise ValueError("predicted and Teacher series shapes differ")
    p_defined = (
        np.ones(predicted_array.shape, dtype=np.bool_)
        if predicted_defined is None
        else np.asarray(predicted_defined, dtype=np.bool_).reshape(-1)
    )
    t_defined = (
        np.ones(teacher_array.shape, dtype=np.bool_)
        if teacher_defined is None
        else np.asarray(teacher_defined, dtype=np.bool_).reshape(-1)
    )
    if p_defined.shape != predicted_array.shape or t_defined.shape != teacher_array.shape:
        raise ValueError("defined masks differ from their series")
    eligible = p_defined & t_defined
    nonfinite_defined = int(
        np.count_nonzero(eligible & (~np.isfinite(predicted_array) | ~np.isfinite(teacher_array)))
    )
    valid = eligible & np.isfinite(predicted_array) & np.isfinite(teacher_array)
    count = int(np.count_nonzero(valid))
    if count != int(np.count_nonzero(eligible)) or count == 0:
        return {
            "count": count,
            "eligible_count": int(np.count_nonzero(eligible)),
            "nonfinite_defined": nonfinite_defined,
            "mse": None,
            "rmse": None,
            "bias": None,
            "aggregate_bias_percent": None,
        }
    residual = predicted_array[valid] - teacher_array[valid]
    target_sum = float(np.sum(teacher_array[valid], dtype=np.float64))
    return {
        "count": count,
        "eligible_count": count,
        "nonfinite_defined": 0,
        "mse": float(np.mean(np.square(residual), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(residual), dtype=np.float64))),
        "bias": float(np.mean(residual, dtype=np.float64)),
        "aggregate_bias_percent": (
            100.0 * float(np.sum(residual, dtype=np.float64)) / target_sum
            if target_sum != 0.0
            else None
        ),
    }


def bidirectional_gap_metrics(
    free: Mapping[str, Any],
    teacher_reset: Mapping[str, Any],
    rescue: Mapping[str, Any],
    damage: Mapping[str, Any],
    *,
    minimum_gap: float = 1.0e-12,
    material_fraction: float = 0.10,
) -> Mapping[str, Any]:
    """Measure sufficiency and reciprocal necessity against the same MSE gap."""

    mses = {
        name: value.get("mse")
        for name, value in (
            ("free", free),
            ("teacher_reset", teacher_reset),
            ("rescue", rescue),
            ("damage", damage),
        )
    }
    if any(value is None or not np.isfinite(value) for value in mses.values()):
        return {
            "eligible": False,
            "free_to_teacher_mse_gap": None,
            "rescue_fraction": None,
            "damage_fraction": None,
            "bidirectionally_material": False,
        }
    gap = float(mses["free"] - mses["teacher_reset"])
    if gap <= minimum_gap:
        return {
            "eligible": False,
            "free_to_teacher_mse_gap": gap,
            "rescue_fraction": None,
            "damage_fraction": None,
            "bidirectionally_material": False,
        }
    rescue_fraction = float((mses["free"] - mses["rescue"]) / gap)
    damage_fraction = float((mses["damage"] - mses["teacher_reset"]) / gap)
    return {
        "eligible": True,
        "free_to_teacher_mse_gap": gap,
        "rescue_fraction": rescue_fraction,
        "damage_fraction": damage_fraction,
        "bidirectionally_material": bool(
            rescue_fraction >= material_fraction and damage_fraction >= material_fraction
        ),
    }


def group_report(
    groups: Sequence[StateInterventionGroup], markov_contract: Any
) -> list[Mapping[str, Any]]:
    """Attach runtime state slices and audited source references to each group."""

    leaves = {leaf.key: leaf for leaf in markov_contract.state_leaves if not leaf.discrete}
    return [
        {
            "name": group.name,
            "role": group.role,
            "coordinate_count": group.coordinate_count,
            "state_leaves": [
                {
                    "key": key,
                    "start": int(leaves[key].start),
                    "stop": int(leaves[key].stop),
                    "shape": list(leaves[key].shape),
                    "axis_names": list(leaves[key].axis_names),
                    "source_ref": leaves[key].source_ref,
                }
                for key in group.state_keys
            ],
        }
        for group in groups
    ]


__all__ = [
    "CANOPY_COVER_LEAF_KEYS",
    "CANOPY_ORGANIC_CARBON_KEY",
    "DIRECT_GROUP_NAMES",
    "DOC_INVENTORY_KEY",
    "LITTER_INVENTORY_KEYS",
    "LITTER_METADATA_KEYS",
    "StateInterventionGroup",
    "bidirectional_gap_metrics",
    "build_gpp_state_intervention_groups",
    "build_intervention_arm_states",
    "group_report",
    "series_error_metrics",
]
