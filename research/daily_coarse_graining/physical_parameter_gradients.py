"""Numerical acceptance primitives for physical-parameter gradients.

This module validates derivatives of scientific model outputs with respect to
physical parameters.  It deliberately contains no network-weight logic and no
ORCHIDEE process formulas; model-specific objectives are supplied by callers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


PairClass = Literal["smooth_active", "inactive", "threshold_adjacent"]


@dataclass(frozen=True)
class GradientAcceptancePolicy:
    """Numerical policy shared by every Gate-D physical-parameter pair."""

    relative_tolerance: float = 0.01
    absolute_tolerance: float = 1.0e-10
    inactive_absolute_tolerance: float = 1.0e-10
    finite_difference_stability_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.relative_tolerance <= 0.01:
            raise ValueError("relative_tolerance must be in (0, 0.01]")
        if self.absolute_tolerance < 0.0 or self.inactive_absolute_tolerance < 0.0:
            raise ValueError("absolute tolerances must be nonnegative")
        if self.finite_difference_stability_tolerance < 0.0:
            raise ValueError("finite-difference stability tolerance must be nonnegative")


@dataclass(frozen=True)
class PhysicalGradientComparison:
    """One output-parameter comparison across AD and finite differences."""

    pair_id: str
    pair_class: PairClass
    parameter_value: float
    finite_difference_step: float
    primal_value: float
    forward_ad: float
    reverse_ad: float
    central_difference: float
    central_difference_half_step: float
    richardson_difference: float
    forward_reverse_absolute_error: float
    forward_fd_absolute_error: float
    reverse_fd_absolute_error: float
    forward_fd_relative_error: float | None
    reverse_fd_relative_error: float | None
    finite_difference_stability_relative_error: float | None
    all_finite: bool
    passed: bool
    decision_reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _relative_error(actual: float, expected: float, *, absolute_floor: float) -> float | None:
    scale = max(abs(actual), abs(expected))
    if scale <= absolute_floor:
        return None
    return abs(actual - expected) / scale


def _central_difference(
    objective: Callable[[jnp.ndarray], jnp.ndarray],
    value: jnp.ndarray,
    step: float,
) -> jnp.ndarray:
    step_value = jnp.asarray(step, dtype=jnp.float64)
    return (objective(value + step_value) - objective(value - step_value)) / (2.0 * step_value)


def compare_physical_parameter_gradient(
    objective: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    pair_id: str,
    pair_class: PairClass,
    parameter_value: float,
    finite_difference_step: float,
    policy: GradientAcceptancePolicy | None = None,
) -> PhysicalGradientComparison:
    """Compare scalar forward/reverse AD with central finite differences.

    The half-step estimate and Richardson extrapolation expose unstable finite
    differences instead of silently selecting a favorable step size.
    """

    if pair_class not in {"smooth_active", "inactive", "threshold_adjacent"}:
        raise ValueError(f"unsupported pair class {pair_class!r}")
    if not np.isfinite(parameter_value):
        raise ValueError("parameter_value must be finite")
    if not np.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive and finite")

    accepted = policy or GradientAcceptancePolicy()
    value = jnp.asarray(parameter_value, dtype=jnp.float64)
    forward_fn = jax.jacfwd(objective)
    reverse_fn = jax.jacrev(objective)
    primal, forward, reverse, fd, fd_half = jax.device_get(
        (
            objective(value),
            forward_fn(value),
            reverse_fn(value),
            _central_difference(objective, value, finite_difference_step),
            _central_difference(objective, value, finite_difference_step / 2.0),
        )
    )
    return classify_physical_gradient_estimates(
        pair_id=pair_id,
        pair_class=pair_class,
        parameter_value=parameter_value,
        finite_difference_step=finite_difference_step,
        primal_value=float(np.asarray(primal)),
        forward_ad=float(np.asarray(forward)),
        reverse_ad=float(np.asarray(reverse)),
        central_difference=float(np.asarray(fd)),
        central_difference_half_step=float(np.asarray(fd_half)),
        policy=accepted,
    )


def classify_physical_gradient_estimates(
    *,
    pair_id: str,
    pair_class: PairClass,
    parameter_value: float,
    finite_difference_step: float,
    primal_value: float,
    forward_ad: float,
    reverse_ad: float,
    central_difference: float,
    central_difference_half_step: float,
    policy: GradientAcceptancePolicy | None = None,
) -> PhysicalGradientComparison:
    """Apply the shared policy to estimates produced by any execution plan."""

    if pair_class not in {"smooth_active", "inactive", "threshold_adjacent"}:
        raise ValueError(f"unsupported pair class {pair_class!r}")
    accepted = policy or GradientAcceptancePolicy()
    forward_value = float(forward_ad)
    reverse_value = float(reverse_ad)
    fd_value = float(central_difference)
    fd_half_value = float(central_difference_half_step)
    richardson_value = (4.0 * fd_half_value - fd_value) / 3.0
    values = np.asarray(
        [primal_value, forward_value, reverse_value, fd_value, fd_half_value, richardson_value],
        dtype=np.float64,
    )
    all_finite = bool(np.isfinite(values).all())

    forward_reverse_abs = abs(forward_value - reverse_value)
    forward_fd_abs = abs(forward_value - richardson_value)
    reverse_fd_abs = abs(reverse_value - richardson_value)
    forward_fd_rel = _relative_error(
        forward_value,
        richardson_value,
        absolute_floor=accepted.absolute_tolerance,
    )
    reverse_fd_rel = _relative_error(
        reverse_value,
        richardson_value,
        absolute_floor=accepted.absolute_tolerance,
    )
    fd_stability_rel = _relative_error(
        fd_value,
        fd_half_value,
        absolute_floor=accepted.absolute_tolerance,
    )

    if not all_finite:
        passed = False
        reason = "one or more primal/derivative estimates are non-finite"
    elif pair_class == "inactive":
        maximum = max(
            abs(forward_value),
            abs(reverse_value),
            abs(fd_value),
            abs(fd_half_value),
            abs(richardson_value),
        )
        passed = maximum <= accepted.inactive_absolute_tolerance
        reason = (
            "all derivative estimates are within the declared inactive absolute tolerance"
            if passed
            else "an inactive pair has a derivative above the declared absolute tolerance"
        )
    elif pair_class == "threshold_adjacent":
        passed = True
        reason = "finite threshold-adjacent estimates reported without a smooth-gradient equivalence claim"
    else:
        forward_ok = (
            forward_fd_abs <= accepted.absolute_tolerance
            if forward_fd_rel is None
            else forward_fd_rel <= accepted.relative_tolerance
        )
        reverse_ok = (
            reverse_fd_abs <= accepted.absolute_tolerance
            if reverse_fd_rel is None
            else reverse_fd_rel <= accepted.relative_tolerance
        )
        stability_ok = (
            abs(fd_value - fd_half_value) <= accepted.absolute_tolerance
            if fd_stability_rel is None
            else fd_stability_rel <= accepted.finite_difference_stability_tolerance
        )
        passed = bool(forward_ok and reverse_ok and stability_ok)
        reason = (
            "forward AD, reverse AD, and stable central differences satisfy the declared tolerance"
            if passed
            else "active-pair AD/finite-difference agreement or finite-difference stability failed"
        )

    return PhysicalGradientComparison(
        pair_id=pair_id,
        pair_class=pair_class,
        parameter_value=float(parameter_value),
        finite_difference_step=float(finite_difference_step),
        primal_value=primal_value,
        forward_ad=forward_value,
        reverse_ad=reverse_value,
        central_difference=fd_value,
        central_difference_half_step=fd_half_value,
        richardson_difference=richardson_value,
        forward_reverse_absolute_error=forward_reverse_abs,
        forward_fd_absolute_error=forward_fd_abs,
        reverse_fd_absolute_error=reverse_fd_abs,
        forward_fd_relative_error=forward_fd_rel,
        reverse_fd_relative_error=reverse_fd_rel,
        finite_difference_stability_relative_error=fd_stability_rel,
        all_finite=all_finite,
        passed=passed,
        decision_reason=reason,
    )
