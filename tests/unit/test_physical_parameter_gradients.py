from __future__ import annotations

import jax.numpy as jnp

from research.daily_coarse_graining.physical_parameter_gradients import (
    GradientAcceptancePolicy,
    classify_physical_gradient_estimates,
    compare_physical_parameter_gradient,
)


def test_smooth_active_gradient_compares_forward_reverse_and_central_difference():
    result = compare_physical_parameter_gradient(
        lambda value: jnp.exp(value) + 0.25 * value * value,
        pair_id="analytic.exp_quadratic",
        pair_class="smooth_active",
        parameter_value=0.7,
        finite_difference_step=1.0e-3,
    )

    assert result.passed
    assert result.all_finite
    assert result.forward_fd_relative_error is not None
    assert result.forward_fd_relative_error < 1.0e-8
    assert result.reverse_fd_relative_error is not None
    assert result.reverse_fd_relative_error < 1.0e-8


def test_inactive_pair_requires_all_methods_to_be_numerically_zero():
    result = compare_physical_parameter_gradient(
        lambda value: jnp.asarray(3.0, dtype=value.dtype),
        pair_id="analytic.inactive",
        pair_class="inactive",
        parameter_value=2.0,
        finite_difference_step=1.0e-3,
    )

    assert result.passed
    assert result.forward_ad == 0.0
    assert result.reverse_ad == 0.0
    assert result.richardson_difference == 0.0


def test_inactive_pair_rejects_a_real_derivative_even_when_it_is_small():
    result = compare_physical_parameter_gradient(
        lambda value: value * 1.0e-7,
        pair_id="analytic.false_inactive",
        pair_class="inactive",
        parameter_value=2.0,
        finite_difference_step=1.0e-3,
        policy=GradientAcceptancePolicy(inactive_absolute_tolerance=1.0e-9),
    )

    assert not result.passed
    assert "inactive" in result.decision_reason


def test_threshold_adjacent_pair_is_reported_without_claiming_smooth_equivalence():
    result = compare_physical_parameter_gradient(
        lambda value: jnp.where(value > 0.0, value, 0.0),
        pair_id="analytic.threshold",
        pair_class="threshold_adjacent",
        parameter_value=0.0,
        finite_difference_step=1.0e-4,
    )

    assert result.passed
    assert result.forward_ad != result.central_difference
    assert "without a smooth-gradient equivalence claim" in result.decision_reason


def test_precomputed_estimates_use_the_same_acceptance_policy():
    result = classify_physical_gradient_estimates(
        pair_id="precomputed.linear",
        pair_class="smooth_active",
        parameter_value=2.0,
        finite_difference_step=1.0e-4,
        primal_value=6.0,
        forward_ad=3.0,
        reverse_ad=3.0,
        central_difference=3.0,
        central_difference_half_step=3.0,
    )

    assert result.passed
    assert result.richardson_difference == 3.0
