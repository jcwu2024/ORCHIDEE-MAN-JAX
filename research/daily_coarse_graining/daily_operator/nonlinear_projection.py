"""Host-side bounded solver and acceptance for nonlinear parameter projection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def projection_candidate_checks(
    candidate: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> Mapping[str, bool]:
    """Evaluate the frozen exact finite-step gates for one candidate."""

    medians = candidate["loss_summary"]["component_median_relative_changes"]
    maintenance = candidate["maintenance_change"]
    displacement = float(candidate["parameter_displacement_norm"])
    return {
        "all_constraints_nonincreasing": float(
            candidate["loss_summary"]["maximum_group_component_relative_increase"]
        )
        <= float(gates["maximum_group_component_relative_increase"]),
        "npp_effect": float(medians["npp"])
        <= -float(gates["minimum_npp_median_relative_decrease"]),
        "agb_effect": float(medians["aboveground_biomass_tendency"])
        <= -float(gates["minimum_agb_median_relative_decrease"]),
        "bgb_effect": float(medians["belowground_biomass_tendency"])
        <= -float(gates["minimum_bgb_median_relative_decrease"]),
        "maintenance_rms_within_cap": float(
            maintenance["normalized_exact_rms"]
        )
        <= float(gates["maintenance_normalized_rms_cap"]),
        "maintenance_maximum_within_cap": float(
            maintenance["maximum_absolute_exact_change"]
        )
        <= float(gates["maintenance_maximum_absolute_cap"]),
        "minimum_displacement_retained": displacement
        >= float(gates["minimum_parameter_displacement_norm"]),
        "maximum_displacement_bounded": displacement
        <= float(gates["maximum_parameter_displacement_norm"]),
        "hard_failures_zero": int(candidate["hard_failure_count"])
        == int(gates["hard_failure_count"]),
        "nonfinite_values_zero": int(candidate["nonfinite_value_count"])
        == int(gates["nonfinite_value_count"]),
    }


def classify_low_dimensional_nonlinear_projection(
    candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_valid: bool,
    gates: Mapping[str, Any],
    pass_action: str,
    fail_action: str,
) -> Mapping[str, Any]:
    """Select the first exact-feasible iterate without retaining parameters."""

    records = []
    for candidate in candidates:
        checks = projection_candidate_checks(candidate, gates=gates)
        records.append(
            {
                "iteration": int(candidate["iteration"]),
                "trial_fraction": float(candidate["trial_fraction"]),
                "passed": all(checks.values()),
                "checks": checks,
            }
        )
    selected = next((record for record in records if record["passed"]), None)
    passed = bool(evidence_valid and selected is not None)
    return {
        "status": (
            "low_dimensional_nonlinear_projection_supported"
            if passed
            else "low_dimensional_nonlinear_projection_not_supported"
        ),
        "passed": passed,
        "evidence_valid": bool(evidence_valid),
        "candidate_checks": records,
        "selected_iteration": selected["iteration"] if selected else None,
        "selected_trial_fraction": (
            selected["trial_fraction"] if selected else None
        ),
        "next_action": pass_action if passed else fail_action,
    }


def exact_violation_merit(
    candidate: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
    loss_scale: float,
    effect_scale: float,
) -> Mapping[str, Any]:
    """Return a deterministic dimensionless merit for exact trial selection."""

    loss_changes = np.asarray(candidate["loss_relative_changes"], dtype=np.float64)
    medians = candidate["loss_summary"]["component_median_relative_changes"]
    maintenance = candidate["maintenance_change"]
    displacement = float(candidate["parameter_displacement_norm"])
    violations = [
        np.maximum(
            loss_changes
            - float(gates["maximum_group_component_relative_increase"]),
            0.0,
        ).ravel()
        / float(loss_scale)
    ]
    for label, gate in (
        ("npp", "minimum_npp_median_relative_decrease"),
        ("aboveground_biomass_tendency", "minimum_agb_median_relative_decrease"),
        ("belowground_biomass_tendency", "minimum_bgb_median_relative_decrease"),
    ):
        violations.append(
            np.asarray(
                [
                    max(
                        float(medians[label]) + float(gates[gate]),
                        0.0,
                    )
                    / float(effect_scale)
                ],
                dtype=np.float64,
            )
        )
    violations.extend(
        (
            np.asarray(
                [
                    max(
                        float(maintenance["normalized_exact_rms"])
                        - float(gates["maintenance_normalized_rms_cap"]),
                        0.0,
                    )
                    / float(gates["maintenance_normalized_rms_cap"])
                ]
            ),
            np.asarray(
                [
                    max(
                        float(maintenance["maximum_absolute_exact_change"])
                        - float(gates["maintenance_maximum_absolute_cap"]),
                        0.0,
                    )
                    / float(gates["maintenance_maximum_absolute_cap"])
                ]
            ),
            np.asarray(
                [
                    max(
                        float(gates["minimum_parameter_displacement_norm"])
                        - displacement,
                        0.0,
                    )
                    / float(gates["minimum_parameter_displacement_norm"]),
                    max(
                        displacement
                        - float(gates["maximum_parameter_displacement_norm"]),
                        0.0,
                    )
                    / float(gates["maximum_parameter_displacement_norm"]),
                ]
            ),
        )
    )
    values = np.concatenate(violations)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    positive = values[values > 0.0]
    return {
        "maximum_normalized_violation": float(
            np.max(values) if values.size else 0.0
        ),
        "mean_squared_normalized_violation": float(
            np.mean(values * values) if values.size else 0.0
        ),
        "violated_constraint_count": int(positive.size),
        "nonfinite_value_count": nonfinite,
    }


def _linearized_violation_system(
    *,
    coordinates: np.ndarray,
    loss_values: np.ndarray,
    loss_jacobian: np.ndarray,
    effect_values: np.ndarray,
    effect_jacobian: np.ndarray,
    maintenance_values: np.ndarray,
    maintenance_jacobian: np.ndarray,
    maintenance_rms: float,
    maintenance_rms_jacobian: np.ndarray,
    raw_alignment: np.ndarray,
    gates: Mapping[str, Any],
    loss_scale: float,
    effect_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build c + G delta <= 0 for all frozen linearized constraints."""

    dimension = coordinates.size
    rows = []
    offsets = []

    rows.append(loss_jacobian / loss_scale)
    offsets.append(
        (
            loss_values
            - float(gates["maximum_group_component_relative_increase"])
        )
        / loss_scale
    )
    effect_limits = -np.asarray(
        [
            gates["minimum_npp_median_relative_decrease"],
            gates["minimum_agb_median_relative_decrease"],
            gates["minimum_bgb_median_relative_decrease"],
        ],
        dtype=np.float64,
    )
    rows.append(effect_jacobian / effect_scale)
    offsets.append((effect_values - effect_limits) / effect_scale)

    rms_cap = float(gates["maintenance_normalized_rms_cap"])
    rows.append(maintenance_rms_jacobian.reshape(1, dimension) / rms_cap)
    offsets.append(np.asarray([(maintenance_rms - rms_cap) / rms_cap]))

    maximum_cap = float(gates["maintenance_maximum_absolute_cap"])
    rows.extend(
        (
            maintenance_jacobian / maximum_cap,
            -maintenance_jacobian / maximum_cap,
        )
    )
    offsets.extend(
        (
            (maintenance_values - maximum_cap) / maximum_cap,
            (-maintenance_values - maximum_cap) / maximum_cap,
        )
    )

    minimum_displacement = float(gates["minimum_parameter_displacement_norm"])
    progress = float(np.dot(raw_alignment, coordinates))
    rows.append((-raw_alignment / minimum_displacement).reshape(1, dimension))
    offsets.append(
        np.asarray([(minimum_displacement - progress) / minimum_displacement])
    )

    maximum_displacement = float(gates["maximum_parameter_displacement_norm"])
    displacement = float(np.linalg.norm(coordinates))
    displacement_gradient = (
        coordinates / displacement if displacement > 0.0 else raw_alignment
    )
    rows.append((displacement_gradient / maximum_displacement).reshape(1, dimension))
    offsets.append(
        np.asarray(
            [(displacement - maximum_displacement) / maximum_displacement]
        )
    )
    return np.concatenate(offsets), np.concatenate(rows, axis=0)


def solve_relinearized_projection_subproblem(
    *,
    coordinates: Sequence[float],
    loss_values: Any,
    loss_jacobian: Any,
    effect_values: Any,
    effect_jacobian: Any,
    maintenance_values: Any,
    maintenance_jacobian: Any,
    maintenance_rms: float,
    maintenance_rms_jacobian: Any,
    raw_alignment: Any,
    gates: Mapping[str, Any],
    solver: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Solve one bounded hard or slack-restoration SLSQP subproblem."""

    from scipy.optimize import Bounds, LinearConstraint, minimize

    current = np.asarray(coordinates, dtype=np.float64)
    dimension = current.size
    arrays = {
        "loss_values": np.asarray(loss_values, dtype=np.float64).reshape(-1),
        "loss_jacobian": np.asarray(loss_jacobian, dtype=np.float64),
        "effect_values": np.asarray(effect_values, dtype=np.float64).reshape(-1),
        "effect_jacobian": np.asarray(effect_jacobian, dtype=np.float64),
        "maintenance_values": np.asarray(
            maintenance_values, dtype=np.float64
        ).reshape(-1),
        "maintenance_jacobian": np.asarray(
            maintenance_jacobian, dtype=np.float64
        ),
        "maintenance_rms_jacobian": np.asarray(
            maintenance_rms_jacobian, dtype=np.float64
        ).reshape(-1),
        "raw_alignment": np.asarray(raw_alignment, dtype=np.float64).reshape(-1),
    }
    expected_shapes = {
        "loss_jacobian": (arrays["loss_values"].size, dimension),
        "effect_jacobian": (arrays["effect_values"].size, dimension),
        "maintenance_jacobian": (arrays["maintenance_values"].size, dimension),
        "maintenance_rms_jacobian": (dimension,),
        "raw_alignment": (dimension,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"nonlinear projection {name} shape drift")
    finite_values = (current, np.asarray(maintenance_rms), *arrays.values())
    if any(np.any(~np.isfinite(value)) for value in finite_values):
        raise ValueError("nonlinear projection subproblem contains nonfinite values")

    offsets, matrix = _linearized_violation_system(
        coordinates=current,
        maintenance_rms=float(maintenance_rms),
        gates=gates,
        loss_scale=float(solver["loss_violation_merit_scale"]),
        effect_scale=float(solver["effect_violation_merit_scale"]),
        **arrays,
    )
    radius = float(solver["coordinate_trust_radius"])
    maximum_iterations = int(solver["maximum_inner_iterations"])
    linear = LinearConstraint(matrix, -np.inf, -offsets)

    def trust(delta: np.ndarray) -> float:
        return radius * radius - float(np.dot(delta, delta))

    def trust_jacobian(delta: np.ndarray) -> np.ndarray:
        return -2.0 * delta

    hard = minimize(
        lambda delta: 0.5 * float(np.dot(delta, delta)),
        np.zeros(dimension, dtype=np.float64),
        jac=lambda delta: delta,
        constraints=(
            linear,
            {"type": "ineq", "fun": trust, "jac": trust_jacobian},
        ),
        method="SLSQP",
        options={"maxiter": maximum_iterations, "ftol": 1e-12, "disp": False},
    )
    hard_delta = np.asarray(hard.x, dtype=np.float64)
    hard_delta_norm = float(np.linalg.norm(hard_delta))
    if hard_delta_norm > radius and hard_delta_norm <= radius + 1e-8:
        hard_delta = hard_delta * (radius / hard_delta_norm)
        hard_delta_norm = radius
    hard_violation = float(np.max(offsets + matrix @ hard_delta))
    if (
        np.all(np.isfinite(hard_delta))
        and hard_violation <= 1e-8
        and hard_delta_norm <= radius + 1e-10
    ):
        return {
            "phase": "hard_minimum_norm",
            "success": True,
            "solver_success": bool(hard.success),
            "status": int(hard.status),
            "message": str(hard.message),
            "inner_iterations": int(hard.nit),
            "delta": hard_delta,
            "delta_norm": hard_delta_norm,
            "predicted_maximum_normalized_violation": hard_violation,
        }

    augmented = np.concatenate(
        (matrix, -np.ones((matrix.shape[0], 1), dtype=np.float64)), axis=1
    )
    initial_slack = max(float(np.max(offsets)), 0.0) + 1e-8
    initial = np.concatenate((np.zeros(dimension), np.asarray([initial_slack])))

    def slack_objective(values: np.ndarray) -> float:
        delta = values[:-1]
        return float(values[-1]) + 1e-3 * float(np.dot(delta, delta)) / (
            radius * radius
        )

    def slack_objective_jacobian(values: np.ndarray) -> np.ndarray:
        gradient = np.empty_like(values)
        gradient[:-1] = 2e-3 * values[:-1] / (radius * radius)
        gradient[-1] = 1.0
        return gradient

    def slack_trust(values: np.ndarray) -> float:
        return trust(values[:-1])

    def slack_trust_jacobian(values: np.ndarray) -> np.ndarray:
        return np.concatenate((trust_jacobian(values[:-1]), np.asarray([0.0])))

    restoration = minimize(
        slack_objective,
        initial,
        jac=slack_objective_jacobian,
        bounds=Bounds(
            np.concatenate((np.full(dimension, -np.inf), np.asarray([0.0]))),
            np.full(dimension + 1, np.inf),
        ),
        constraints=(
            LinearConstraint(augmented, -np.inf, -offsets),
            {
                "type": "ineq",
                "fun": slack_trust,
                "jac": slack_trust_jacobian,
            },
        ),
        method="SLSQP",
        options={"maxiter": maximum_iterations, "ftol": 1e-12, "disp": False},
    )
    delta = np.asarray(restoration.x[:-1], dtype=np.float64)
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > radius and delta_norm <= radius + 1e-8:
        delta = delta * (radius / delta_norm)
        delta_norm = radius
    slack = float(restoration.x[-1])
    predicted = float(np.max(offsets + matrix @ delta))
    operational_success = bool(
        np.all(np.isfinite(delta))
        and np.isfinite(slack)
        and slack >= -1e-10
        and delta_norm <= radius + 1e-8
        and predicted <= slack + 1e-8
    )
    return {
        "phase": "bounded_slack_restoration",
        "success": operational_success,
        "solver_success": bool(restoration.success),
        "status": int(restoration.status),
        "message": str(restoration.message),
        "inner_iterations": int(restoration.nit),
        "delta": delta,
        "delta_norm": delta_norm,
        "slack": slack,
        "predicted_maximum_normalized_violation": predicted,
        "hard_phase_status": int(hard.status),
        "hard_phase_message": str(hard.message),
    }
