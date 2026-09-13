from types import SimpleNamespace

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
    DIRECT_STATE_KEYS,
)
from research.daily_coarse_graining.daily_operator.trained_gpp_state_intervention import (
    CANOPY_COVER_LEAF_KEYS,
    CANOPY_ORGANIC_CARBON_KEY,
    DOC_INVENTORY_KEY,
    LITTER_INVENTORY_KEYS,
    LITTER_METADATA_KEYS,
    bidirectional_gap_metrics,
    build_gpp_state_intervention_groups,
    build_intervention_arm_states,
    series_error_metrics,
)
from scripts.hpc.diagnose_trained_gpp_state_interventions import _control_comparison, _load_archived_carries

COMPONENTS = {
    "diffuco_previous_step_state": ("control_salinity", "control_inudate", "gsmean"),
    "enerbil_previous_step_state": ("temp_sol",),
    "hydrol_previous_step_state": ("humrel", "qsintveg", "mc"),
    "thermosoil_previous_step_state": ("gtemp",),
    "slowproc_stomate_previous_step_state": (
        "assim_param",
        "temp_growth",
        "lai",
        "veget",
        "veget_max",
        "veget_lastlight",
        "leaf_age",
        "leaf_frac",
        "frac_age",
        "interception_storage",
        "DOC",
        "litter",
        "litter_above",
        "litter_below",
        "litterpart",
        "lignin_struc",
        "lignin_struc_above",
        "lignin_struc_below",
        "dead_leaves",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "biomass",
    ),
    "driver_previous_step_state": ("fluxsens",),
    "sechiba_finalize_state": ("leaf_ci",),
}


def _contract():
    leaves = []
    offset = 0
    for component, names in COMPONENTS.items():
        for name in names:
            leaves.append(
                SimpleNamespace(
                    component=component,
                    path=(name,),
                    key=f"{component}.{name}",
                    discrete=False,
                    start=offset,
                    stop=offset + 1,
                    shape=(1,),
                    axis_names=("npts",),
                    source_ref=f"source::{component}.{name}",
                )
            )
            offset += 1
    leaves.append(
        SimpleNamespace(
            component="driver_previous_step_state",
            path=("flag",),
            key="driver_previous_step_state.flag",
            discrete=True,
            start=None,
            stop=None,
        )
    )
    return SimpleNamespace(state_leaves=tuple(leaves), continuous_state_width=offset)


def test_intervention_groups_partition_all_continuous_coordinates_once():
    contract = _contract()
    groups = build_gpp_state_intervention_groups(contract)
    coverage = np.stack([group.mask for group in groups]).sum(axis=0)
    np.testing.assert_array_equal(coverage, np.ones(contract.continuous_state_width))
    by_name = {group.name: group for group in groups}
    assert len(DIRECT_STATE_KEYS) == 6
    assert by_name["canopy_organic_carbon"].state_keys == (
        CANOPY_ORGANIC_CARBON_KEY,
    )
    assert by_name["canopy_cover_leaf"].state_keys == CANOPY_COVER_LEAF_KEYS
    assert by_name["doc_inventory"].state_keys == (DOC_INVENTORY_KEY,)
    assert by_name["litter_inventory"].state_keys == LITTER_INVENTORY_KEYS
    assert by_name["litter_metadata"].state_keys == LITTER_METADATA_KEYS
    assert "hydrol_previous_step_state.mc" in by_name["remaining_hydrology"].state_keys
    assert "slowproc_stomate_previous_step_state.biomass" in (
        by_name["remaining_slow_process"].state_keys
    )
    assert "sechiba_finalize_state.leaf_ci" in (
        by_name["remaining_surface_driver_boundary"].state_keys
    )


def test_rescue_and_damage_arms_are_reciprocal_and_do_not_mutate_controls():
    groups = build_gpp_state_intervention_groups(_contract())
    free = np.arange(len(groups[0].mask), dtype=float)
    teacher = free + 100.0
    names, states = build_intervention_arm_states(free, teacher, groups)
    np.testing.assert_array_equal(states[0], free)
    np.testing.assert_array_equal(states[1], teacher)
    group = groups[2]
    rescue = states[names.index(f"rescue__{group.name}")]
    damage = states[names.index(f"damage__{group.name}")]
    np.testing.assert_array_equal(rescue[group.mask], teacher[group.mask])
    np.testing.assert_array_equal(rescue[~group.mask], free[~group.mask])
    np.testing.assert_array_equal(damage[group.mask], free[group.mask])
    np.testing.assert_array_equal(damage[~group.mask], teacher[~group.mask])


def test_bidirectional_gap_uses_one_free_to_teacher_mse_denominator():
    target = np.zeros(4)
    free = series_error_metrics(np.full(4, 2.0), target)
    teacher = series_error_metrics(np.zeros(4), target)
    rescue = series_error_metrics(np.full(4, 1.0), target)
    damage = series_error_metrics(np.full(4, np.sqrt(2.0)), target)
    result = bidirectional_gap_metrics(free, teacher, rescue, damage)
    assert result["eligible"]
    assert np.isclose(result["rescue_fraction"], 0.75)
    assert np.isclose(result["damage_fraction"], 0.5)
    assert result["bidirectionally_material"]


def test_defined_nonfinite_is_reported_without_silent_filtering():
    result = series_error_metrics(
        [1.0, np.nan],
        [1.0, 2.0],
        predicted_defined=[True, True],
        teacher_defined=[True, True],
    )
    assert result["eligible_count"] == 2
    assert result["count"] == 1
    assert result["nonfinite_defined"] == 1
    assert result["mse"] is None


def test_archived_free_carry_uses_previous_canonical_endpoint(tmp_path):
    path = tmp_path / "states.npz"
    free_state = np.arange(12, dtype=float).reshape(3, 4)
    teacher_start = free_state + 100.0
    np.savez(
        path,
        free_state=free_state,
        free_defined=np.ones_like(free_state, dtype=bool),
        teacher_state=teacher_start + 1.0,
        teacher_defined=np.ones_like(free_state, dtype=bool),
        teacher_start=teacher_start,
    )
    free_carry, teacher_carry, _ = _load_archived_carries(
        path, horizon=3, state_width=4
    )
    np.testing.assert_array_equal(teacher_carry, teacher_start)
    np.testing.assert_array_equal(free_carry[0], teacher_start[0])
    np.testing.assert_array_equal(free_carry[1:], free_state[:-1])


def test_control_replay_keeps_numeric_discrepancies_as_evidence():
    assert _control_comparison([1.0 + 1.0e-7], [1.0])["valid"]
    discrepancy = _control_comparison([1.1], [1.0])
    assert not discrepancy["valid"]
    assert np.isclose(discrepancy["maximum_absolute_error"], 0.1)
    assert not _control_comparison([np.nan], [1.0])["valid"]
