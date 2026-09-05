from __future__ import annotations

import copy
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_structured_precision_attribution import (
    classify_precision_paths,
    classify_subtree_controls,
    float32_delta_inventory,
    float32_interpolation_membership,
)
from scripts.hpc import (
    audit_target_oriented_coupled_gpp_structured_precision_attribution as runner,
)


def _tree(first: np.ndarray, second: np.ndarray | None = None):
    output = {"first": {"weight": first}}
    if second is not None:
        output["second"] = (second,)
    return output


def test_float32_delta_inventory_counts_ulps_and_sign_crossing():
    base = np.asarray((1.0, -1.0, 0.0, 4.0), dtype=np.float32)
    one = np.nextafter(base[0], np.float32(np.inf), dtype=np.float32)
    two = np.nextafter(
        np.nextafter(base[1], np.float32(np.inf), dtype=np.float32),
        np.float32(np.inf),
        dtype=np.float32,
    )
    treatment = base.copy()
    treatment[0] = one
    treatment[1] = two
    treatment[2] = np.float32(-1.0e-45)

    report = float32_delta_inventory(
        _tree(base),
        _tree(treatment),
        groups=("first",),
    )

    summary = report["summary"]
    assert summary["parameter_count"] == 4
    assert summary["changed_coordinate_count"] == 3
    assert summary["one_ulp_changed_count"] == 2
    assert summary["two_ulp_changed_count"] == 1
    assert summary["more_than_two_ulp_changed_count"] == 0
    assert summary["sign_crossing_count"] == 1
    assert summary["maximum_ulp_distance"] == 2
    assert summary["ulp_count_conserved"] is True


def test_delta_inventory_rejects_non_float32_leaves():
    with pytest.raises(ValueError, match="float32"):
        float32_delta_inventory(
            _tree(np.asarray((1.0,), dtype=np.float64)),
            _tree(np.asarray((2.0,), dtype=np.float64)),
            groups=("first",),
        )


def test_interpolation_membership_conserves_changed_coordinates():
    control = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float32)
    treatment = np.asarray((2.0, 3.0, 3.0, 8.0), dtype=np.float32)
    realized = np.asarray((1.0, 3.0, 3.0, 6.0), dtype=np.float32)

    report = float32_interpolation_membership(
        _tree(control),
        _tree(treatment),
        _tree(realized),
        groups=("first",),
    )["summary"]

    assert report["endpoint_changed_coordinate_count"] == 3
    assert report["equals_control_changed_count"] == 1
    assert report["equals_treatment_changed_count"] == 1
    assert report["intermediate_changed_count"] == 1
    assert report["changed_membership_conserved"] is True
    assert report["unchanged_preserved"] is True


def test_registry_checks_accept_equivalent_mapping_insertion_orders():
    control = {
        "first": {"weight": np.asarray((1.0,), dtype=np.float32)},
        "second": {"weight": np.asarray((2.0,), dtype=np.float32)},
    }
    treatment = {
        "second": {"weight": np.asarray((3.0,), dtype=np.float32)},
        "first": {"weight": np.asarray((2.0,), dtype=np.float32)},
    }

    inventory = float32_delta_inventory(
        control,
        treatment,
        groups=("first", "second"),
    )

    assert tuple(inventory["groups"]) == ("first", "second")
    assert inventory["summary"]["changed_coordinate_count"] == 2


@pytest.mark.parametrize(
    ("float64_values", "expected"),
    (
        (
            (0.0, 0.25, 0.5, 0.75, 1.0),
            "float32_weight_quantization_dominant_for_sampled_jump",
        ),
        (
            (0.0, 0.0, 0.0, 0.0, 1.0),
            "structured_gpp_conditioning_persists_with_float64_weights",
        ),
        (
            (0.0, 0.0, 0.0, 0.0, 0.5),
            "mixed_or_inconclusive_precision_effect",
        ),
    ),
)
def test_precision_path_classification(float64_values, expected):
    result = classify_precision_paths(
        fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        production_gpp=(0.0, 0.0, 0.0, 0.0, 1.0),
        float64_weight_gpp=float64_values,
    )

    assert result["classification"] == expected


def test_subtree_controls_report_sufficiency_and_necessity():
    groups = ("first", "second")
    result = classify_subtree_controls(
        control_gpp=1.0,
        treatment_gpp=3.0,
        treatment_group_only_gpp={"first": 2.8, "second": 1.0},
        treatment_without_group_gpp={"first": 1.2, "second": 3.0},
        groups=groups,
    )

    assert result["selected_groups"] == ["first"]
    assert result["groups"]["first"]["signed_sufficiency_fraction"] == pytest.approx(0.9)
    assert result["groups"]["first"]["necessity_capture_fraction"] == pytest.approx(0.9)
    assert result["groups"]["second"]["signed_sufficiency_fraction"] == 0.0
    assert result["groups"]["second"]["necessity_capture_fraction"] == 0.0


def test_crossed_parent_identity_and_factor_metadata_are_exact():
    root = Path(__file__).resolve().parents[2]
    path = root / "outputs/diagnostics/target-oriented-coupled-gpp-day5-crossed-1c7dc9b.json"
    report, digest = runner._load_crossed_result(path)

    assert digest == runner.CROSSED_ATTRIBUTION_SHA256
    metadata = runner._parent_factor_metadata(report)

    assert metadata["vegetation_pft_index"] == 1
    assert metadata["control_gpp"] == pytest.approx(2.76733630416029)
    assert metadata["treatment_gpp"] == pytest.approx(4.869899723490635)
    assert len(metadata["control_gpp_trace"]) == 7

    drifted = copy.deepcopy(report)
    drifted["integrity"]["engineering_gate_passed"] = False
    with pytest.raises(ValueError, match="drift"):
        runner._validate_crossed_result(
            drifted,
            runner.CROSSED_ATTRIBUTION_SHA256,
        )


def test_mixed_subtree_replaces_only_selected_group(monkeypatch):
    monkeypatch.setattr(runner, "STRUCTURED_GPP_GROUPS", ("first", "second"))
    control = {"first": object(), "second": object()}
    treatment = {"second": object(), "first": object()}

    mixed = runner._mixed_subtree(control, treatment, ("second",))

    assert mixed["first"] is control["first"]
    assert mixed["second"] is treatment["second"]


def test_precision_interpolation_preserves_float32_endpoints_and_midpoint():
    first = {"value": jnp.asarray((1.0, 3.0), dtype=jnp.float32)}
    second = {"value": jnp.asarray((3.0, 7.0), dtype=jnp.float32)}

    start = runner._interpolate_tree(
        jax,
        jnp,
        first,
        second,
        0.0,
        dtype=jnp.float32,
    )
    midpoint = runner._interpolate_tree(
        jax,
        jnp,
        first,
        second,
        0.5,
        dtype=jnp.float32,
    )
    end = runner._interpolate_tree(
        jax,
        jnp,
        first,
        second,
        1.0,
        dtype=jnp.float32,
    )

    np.testing.assert_array_equal(np.asarray(start["value"]), np.asarray(first["value"]))
    np.testing.assert_array_equal(
        np.asarray(midpoint["value"]),
        np.asarray((2.0, 5.0), dtype=np.float32),
    )
    np.testing.assert_array_equal(np.asarray(end["value"]), np.asarray(second["value"]))


def test_policy_record_freezes_zero_training_boundary():
    policy = runner._policy_record()

    assert policy["logical_updates_applied"] == 0
    assert policy["historical_parameter_displacements_reconstructed_in_memory"] == 1
    assert all(
        value is False
        for key, value in policy.items()
        if key
        not in {
            "logical_updates_applied",
            "historical_parameter_displacements_reconstructed_in_memory",
            "forcing_records_reconstructed",
            "subdaily_state_scan_steps",
        }
    )
    assert policy["forcing_records_reconstructed"] == 0
    assert policy["subdaily_state_scan_steps"] == 0


def test_precision_runner_routes_inputs_and_resolves_metadata_before_reconstruction():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(encoding="utf-8")
    wrapper = (root / "scripts/hpc/run_target_oriented_coupled_gpp_structured_precision_attribution.sh").read_text(
        encoding="utf-8"
    )
    required = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }

    assert sorted(option for option in required if option not in shared and option not in wrapper) == []
    assert "CROSSED_ATTRIBUTION_RESULT" in shared
    assert "CROSSED_ATTRIBUTION_RESULT" in wrapper
    assert "sbatch" not in wrapper
    assert "free gln01" in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-60m}" in wrapper

    source = inspect.getsource(runner.run)
    reconstruction = "reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)"
    assert source.count(reconstruction) == 1
    assert source.index("_load_crossed_result") < source.index(reconstruction)
    assert source.index("_parent_factor_metadata") < source.index(reconstruction)
    assert source.index("_runtime_year_for_update") < source.index(reconstruction)
    assert source.index("_validate_initialized_structured_tree") < source.index(reconstruction)
    assert source.index("jax_enable_x64") < source.index(reconstruction)
    assert source.index("static_report =") < source.index(reconstruction)
    assert source.index("engineering_gate_failed") < source.index("_atomic_json(output, report)")
