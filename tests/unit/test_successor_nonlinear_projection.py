from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.nonlinear_projection import (
    classify_low_dimensional_nonlinear_projection,
    exact_violation_merit,
    solve_relinearized_projection_subproblem,
)
from scripts.hpc.probe_successor_parameter_geometry import (
    _compose_subspace_parameters,
    _median_effect_values_and_jacobian,
    _orthonormalize_parameter_directions,
)


def _gates() -> dict[str, float | int]:
    return {
        "maximum_group_component_relative_increase": 1e-6,
        "minimum_npp_median_relative_decrease": 1e-5,
        "minimum_agb_median_relative_decrease": 1e-5,
        "minimum_bgb_median_relative_decrease": 1e-5,
        "minimum_parameter_displacement_norm": 0.1,
        "maximum_parameter_displacement_norm": 1.0,
        "maintenance_normalized_rms_cap": 0.1,
        "maintenance_maximum_absolute_cap": 0.2,
        "hard_failure_count": 0,
        "nonfinite_value_count": 0,
    }


def _candidate(*, passing: bool) -> dict[str, object]:
    loss_change = -2e-5 if passing else 2e-3
    maintenance_rms = 0.05 if passing else 0.3
    return {
        "iteration": 1,
        "trial_fraction": 1.0,
        "loss_relative_changes": [[loss_change] * 4],
        "loss_summary": {
            "maximum_group_component_relative_increase": loss_change,
            "component_median_relative_changes": {
                "one_day_anchor": loss_change,
                "npp": -2e-5,
                "aboveground_biomass_tendency": -3e-5,
                "belowground_biomass_tendency": -4e-5,
            },
        },
        "maintenance_change": {
            "normalized_exact_rms": maintenance_rms,
            "maximum_absolute_exact_change": maintenance_rms,
        },
        "parameter_displacement_norm": 0.5,
        "hard_failure_count": 0,
        "nonfinite_value_count": 0,
    }


def test_nonlinear_projection_selects_first_exact_feasible_candidate():
    failed = _candidate(passing=False)
    passed = _candidate(passing=True)
    passed["iteration"] = 2
    passed["trial_fraction"] = 0.5
    result = classify_low_dimensional_nonlinear_projection(
        (failed, passed),
        evidence_valid=True,
        gates=_gates(),
        pass_action="screen",
        fail_action="stop",
    )

    assert result["status"] == "low_dimensional_nonlinear_projection_supported"
    assert result["selected_iteration"] == 2
    assert result["selected_trial_fraction"] == 0.5
    assert result["next_action"] == "screen"


def test_nonlinear_projection_merit_detects_failed_constraints():
    failed = exact_violation_merit(
        _candidate(passing=False),
        gates=_gates(),
        loss_scale=0.001,
        effect_scale=0.0001,
    )
    passed = exact_violation_merit(
        _candidate(passing=True),
        gates=_gates(),
        loss_scale=0.001,
        effect_scale=0.0001,
    )

    assert failed["maximum_normalized_violation"] > 0.0
    assert failed["violated_constraint_count"] > 0
    assert passed["maximum_normalized_violation"] == 0.0
    assert passed["violated_constraint_count"] == 0


def _subproblem(*, radius: float) -> dict[str, object]:
    return solve_relinearized_projection_subproblem(
        coordinates=np.asarray([0.4, 0.0]),
        loss_values=np.asarray([0.2]),
        loss_jacobian=np.asarray([[-1.0, 0.0]]),
        effect_values=np.asarray([-0.1, -0.1, -0.1]),
        effect_jacobian=np.zeros((3, 2)),
        maintenance_values=np.asarray([0.0, 0.0]),
        maintenance_jacobian=np.zeros((2, 2)),
        maintenance_rms=0.0,
        maintenance_rms_jacobian=np.zeros(2),
        raw_alignment=np.asarray([1.0, 0.0]),
        gates=_gates(),
        solver={
            "coordinate_trust_radius": radius,
            "maximum_inner_iterations": 100,
            "loss_violation_merit_scale": 1.0,
            "effect_violation_merit_scale": 1.0,
        },
    )


def test_relinearized_subproblem_finds_hard_minimum_norm_step():
    result = _subproblem(radius=0.5)

    assert result["phase"] == "hard_minimum_norm"
    assert result["success"] is True
    assert np.asarray(result["delta"])[0] >= 0.2 - 2e-6
    assert result["delta_norm"] <= 0.5 + 1e-8


def test_relinearized_subproblem_uses_bounded_slack_when_infeasible():
    result = _subproblem(radius=0.05)

    assert result["phase"] == "bounded_slack_restoration"
    assert result["success"] is True
    assert result["slack"] > 0.0
    assert result["delta_norm"] <= 0.05 + 1e-8


def test_parameter_basis_drops_dependent_direction_and_composes_coordinates():
    candidates = (
        ("x", {"value": jnp.asarray([2.0, 0.0, 0.0])}),
        ("duplicate_x", {"value": jnp.asarray([4.0, 0.0, 0.0])}),
        ("y", {"value": jnp.asarray([0.0, 3.0, 0.0])}),
    )
    basis = _orthonormalize_parameter_directions(
        jax,
        candidates,
        minimum_residual_fraction=1e-8,
    )
    composed = _compose_subspace_parameters(
        jax,
        flat_stopped=jnp.zeros(3),
        basis_matrix=basis["matrix"],
        unravel=basis["unravel"],
        coordinates=np.asarray([0.25, -0.5]),
    )

    assert basis["dimension"] == 2
    assert [record["kept"] for record in basis["records"]] == [True, False, True]
    np.testing.assert_allclose(composed["parameters"]["value"], [0.25, -0.5, 0.0])
    assert composed["displacement_norm"] == np.hypot(0.25, 0.5)


def test_median_effect_jacobian_uses_the_locally_active_middle_pair():
    values = np.zeros((10, 4), dtype=np.float64)
    values[:, 1] = np.arange(10, dtype=np.float64)
    values[:, 2] = np.arange(10, dtype=np.float64)[::-1]
    values[:, 3] = 2.0
    jacobian = np.zeros((10, 4, 2), dtype=np.float64)
    jacobian[:, 1, 0] = np.arange(10, dtype=np.float64)
    jacobian[:, 2, 1] = np.arange(10, dtype=np.float64)
    jacobian[:, 3, :] = 3.0

    medians, derivatives = _median_effect_values_and_jacobian(
        values.ravel(), jacobian.reshape(40, 2)
    )

    np.testing.assert_allclose(medians, [4.5, 4.5, 2.0])
    np.testing.assert_allclose(derivatives[0], [4.5, 0.0])
    np.testing.assert_allclose(derivatives[1], [0.0, 4.5])
    np.testing.assert_allclose(derivatives[2], [3.0, 3.0])


def test_relinearized_subproblem_accepts_full_maintenance_row_count():
    dimension = 8
    result = solve_relinearized_projection_subproblem(
        coordinates=np.asarray([0.4] + [0.0] * (dimension - 1)),
        loss_values=np.full(40, -0.01),
        loss_jacobian=np.zeros((40, dimension)),
        effect_values=np.full(3, -0.01),
        effect_jacobian=np.zeros((3, dimension)),
        maintenance_values=np.zeros(28800),
        maintenance_jacobian=np.zeros((28800, dimension)),
        maintenance_rms=0.0,
        maintenance_rms_jacobian=np.zeros(dimension),
        raw_alignment=np.asarray([1.0] + [0.0] * (dimension - 1)),
        gates=_gates(),
        solver={
            "coordinate_trust_radius": 0.05,
            "maximum_inner_iterations": 10,
            "loss_violation_merit_scale": 1.0,
            "effect_violation_merit_scale": 1.0,
        },
    )

    assert result["phase"] == "hard_minimum_norm"
    assert result["success"] is True
    assert result["delta_norm"] == 0.0
