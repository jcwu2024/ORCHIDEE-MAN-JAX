from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_day5_crossed_attribution import (
    PARAMETER_FAMILY_COUNTS,
    STATE_COMPONENT_COORDINATE_COUNTS,
    classify_parameter_state_cross,
    classify_state_owner_controls,
    parameter_family_controls,
    resolve_crossed_gpp_state_components,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_state_feedback_attribution import (
    SOURCE_COMPONENTS,
)
from scripts.hpc import audit_target_oriented_coupled_gpp_day5_crossed_attribution as runner


@pytest.mark.parametrize(
    ("gpp_tc", "expected"),
    (
        (1.1, "state_path_dominant_for_day5_jump"),
        (2.0, "mixed_parameter_state_path_for_day5_jump"),
        (2.8, "state_path_not_explanatory_for_day5_jump"),
    ),
)
def test_parameter_state_cross_uses_frozen_matched_state_boundaries(gpp_tc, expected):
    result = classify_parameter_state_cross(
        gpp_cc=1.0,
        gpp_tc=gpp_tc,
        gpp_ct=2.7,
        gpp_tt=3.0,
    )
    assert result["classification"] == expected
    assert result["parameter_shapley_effect"] + result["state_shapley_effect"] == pytest.approx(2.0)
    assert result["shapley_additivity_residual"] == pytest.approx(0.0, abs=1.0e-15)


def test_state_owner_controls_report_necessity_and_sufficiency():
    names = tuple(STATE_COMPONENT_COORDINATE_COUNTS)
    matched = {name: 3.0 for name in names}
    transplanted = {name: 1.0 for name in names}
    matched[names[0]] = 1.2
    transplanted[names[0]] = 2.8

    result = classify_state_owner_controls(
        gpp_treatment_control_state=1.0,
        gpp_treatment_treatment_state=3.0,
        matched_owner_gpp=matched,
        transplanted_owner_gpp=transplanted,
        state_path_classification="state_path_dominant_for_day5_jump",
    )

    assert result["selected_components"] == [names[0]]
    assert result["components"][names[0]]["necessity_capture_fraction"] == pytest.approx(0.9)
    assert result["components"][names[0]]["sufficiency_fraction"] == pytest.approx(0.9)


def test_state_owner_controls_do_not_select_when_state_path_is_rejected():
    names = tuple(STATE_COMPONENT_COORDINATE_COUNTS)
    result = classify_state_owner_controls(
        gpp_treatment_control_state=1.0,
        gpp_treatment_treatment_state=3.0,
        matched_owner_gpp={name: 1.0 for name in names},
        transplanted_owner_gpp={name: 3.0 for name in names},
        state_path_classification="state_path_not_explanatory_for_day5_jump",
    )
    assert result["selected_components"] == []
    assert all(
        not record["individually_material"]
        for record in result["components"].values()
    )


def test_parameter_family_controls_resolve_interaction():
    families = tuple(PARAMETER_FAMILY_COUNTS)
    result = parameter_family_controls(
        control_gpp=1.0,
        treatment_gpp=3.0,
        treatment_family_only_gpp={families[0]: 2.6, families[1]: 1.2},
    )
    assert result["families"][families[0]]["signed_sufficiency_fraction"] == pytest.approx(0.8)
    assert result["families"][families[0]]["necessity_capture_fraction"] == pytest.approx(0.9)
    assert result["two_family_interaction"] == pytest.approx(0.2)
    assert result["interaction_fraction"] == pytest.approx(0.1)


def test_state_component_masks_are_exact_disjoint_and_gpp_admitted():
    leaves = []
    start = 0
    for name, component in SOURCE_COMPONENTS:
        count = STATE_COMPONENT_COORDINATE_COUNTS[name]
        leaves.append(
            SimpleNamespace(
                key=f"{component}.field",
                component=component,
                discrete=False,
                start=start,
                stop=start + count,
                shape=(count,),
                axis_names=("coordinate",),
            )
        )
        start += count
    markov = SimpleNamespace(state_leaves=tuple(leaves), continuous_state_width=start)
    layout = SimpleNamespace(
        expert_names=("canopy_gpp",),
        source_family_names=tuple(STATE_COMPONENT_COORDINATE_COUNTS),
        state_field_family_index=np.arange(len(leaves), dtype=np.int32),
        source_allowed=np.ones((1, len(leaves)), dtype=np.bool_),
    )

    result = resolve_crossed_gpp_state_components(markov, layout)

    assert tuple(result) == tuple(STATE_COMPONENT_COORDINATE_COUNTS)
    assert sum(record["state_coordinate_count"] for record in result.values()) == start
    assert all(record["canopy_gpp_admitted"] for record in result.values())


def test_tree_interpolation_preserves_endpoints_and_midpoint():
    first = {"value": jnp.asarray((1.0, 3.0), dtype=jnp.float32)}
    second = {"value": jnp.asarray((3.0, 7.0), dtype=jnp.float32)}
    assert runner._interpolate_tree(jax, jnp, first, second, 0.0) is first
    assert runner._interpolate_tree(jax, jnp, first, second, 1.0) is second
    np.testing.assert_array_equal(
        np.asarray(runner._interpolate_tree(jax, jnp, first, second, 0.5)["value"]),
        np.asarray((2.0, 5.0), dtype=np.float32),
    )


def test_crossed_runner_routes_every_required_input_and_has_no_paid_path():
    root = Path(__file__).resolve().parents[2]
    shared = (
        root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh"
    ).read_text(encoding="utf-8")
    wrapper = (
        root / "scripts/hpc/run_target_oriented_coupled_gpp_day5_crossed_attribution.sh"
    ).read_text(encoding="utf-8")
    required = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    assert sorted(option for option in required if option not in shared and option not in wrapper) == []
    assert "SNOWHEAT_CAUSAL_RESULT" in shared
    assert "SNOWHEAT_CAUSAL_RESULT" in wrapper
    assert "sbatch" not in wrapper
    assert "free gln01" in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-60m}" in wrapper
    source = inspect.getsource(runner.run)
    assert source.count("reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)") == 1
    assert source.index("resolve_crossed_gpp_state_components") < source.index(
        "reconstruct_localized_jump_parameters"
    )
    assert source.index("parameter-family count drift") < source.index(
        "reconstruct_localized_jump_parameters"
    )
    assert source.index("engineering_gate_failed") < source.index("_atomic_json(output, report)")
