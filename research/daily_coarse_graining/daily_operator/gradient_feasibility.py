"""Gradient-geometry diagnostics for bounded multi-objective training."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

GRADIENT_FEASIBILITY_LABELS = (
    "one_day_anchor",
    "npp",
    "aboveground_biomass_tendency",
    "belowground_biomass_tendency",
)
TRAINABLE_DIRECTION_NAMES = (
    "equal_unit",
    "minimum_norm_unit",
)
REALIZED_STEP_OPTIMIZERS = (
    "inherited_adamw",
    "reset_adamw",
)


def _project_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    positions = np.arange(1, ordered.size + 1, dtype=np.float64)
    active = ordered - cumulative / positions > 0.0
    index = int(np.nonzero(active)[0][-1])
    threshold = cumulative[index] / float(index + 1)
    return np.maximum(np.asarray(values, dtype=np.float64) - threshold, 0.0)


def _active_set_minimum_norm_weights(
    normalized_gram: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Solve the small simplex QP by enumerating its nonempty faces."""

    matrix = np.asarray(normalized_gram, dtype=np.float64)
    count = int(matrix.shape[0])
    best: np.ndarray | None = None
    best_squared_norm = np.inf
    for mask in range(1, 1 << count):
        active = np.asarray(
            [index for index in range(count) if mask & (1 << index)],
            dtype=np.int64,
        )
        face = matrix[np.ix_(active, active)]
        kkt = np.block(
            [
                [face, -np.ones((active.size, 1), dtype=np.float64)],
                [
                    np.ones((1, active.size), dtype=np.float64),
                    np.zeros((1, 1), dtype=np.float64),
                ],
            ]
        )
        right = np.concatenate(
            (np.zeros(active.size, dtype=np.float64), np.ones(1, dtype=np.float64))
        )
        solution, _residuals, _rank, _singular = np.linalg.lstsq(
            kkt,
            right,
            rcond=None,
        )
        if (
            np.max(np.abs(kkt @ solution - right)) > tolerance
            or np.any(solution[:-1] < -tolerance)
        ):
            continue
        weights = np.zeros(count, dtype=np.float64)
        weights[active] = np.maximum(solution[:-1], 0.0)
        total = float(np.sum(weights))
        if total <= 0.0:
            continue
        weights /= total
        gradient = matrix @ weights
        level = float(weights @ gradient)
        support = weights > tolerance
        if (
            np.any(np.abs(gradient[support] - level) > 10.0 * tolerance)
            or np.any(gradient[~support] < level - 10.0 * tolerance)
        ):
            continue
        squared_norm = float(weights @ matrix @ weights)
        if squared_norm < best_squared_norm:
            best = weights
            best_squared_norm = squared_norm
    if best is None:
        raise ValueError("minimum-norm active-set solver did not find a KKT point")
    return best


def _minimum_norm_simplex_weights(
    normalized_gram: np.ndarray,
    *,
    tolerance: float = 1.0e-13,
    maximum_iterations: int = 20_000,
) -> np.ndarray:
    count = int(normalized_gram.shape[0])
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    lipschitz = float(np.max(np.linalg.eigvalsh(normalized_gram)))
    if lipschitz <= 0.0:
        return weights
    step = 1.0 / lipschitz
    for _ in range(maximum_iterations):
        next_weights = _project_simplex(weights - step * (normalized_gram @ weights))
        if float(np.max(np.abs(next_weights - weights))) <= tolerance:
            return next_weights
        weights = next_weights
    return _active_set_minimum_norm_weights(normalized_gram)


def _direction_record(
    gram: np.ndarray,
    weights: np.ndarray,
    labels: Sequence[str],
) -> Mapping[str, Any]:
    squared_norm = float(weights @ gram @ weights)
    norm = float(np.sqrt(max(squared_norm, 0.0)))
    component_norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    directional = gram @ weights
    cosines = np.divide(
        directional,
        component_norms * norm,
        out=np.zeros_like(directional),
        where=(component_norms > 0.0) & (norm > 0.0),
    )
    return {
        "weights": {
            label: float(weights[index]) for index, label in enumerate(labels)
        },
        "direction_norm": norm,
        "normalized_directional_derivatives": {
            label: float(cosines[index]) for index, label in enumerate(labels)
        },
        "all_strictly_positive": bool(np.all(cosines > 0.0)),
    }


def gradient_direction_diagnostics(
    gram: Any,
    *,
    labels: Sequence[str] = GRADIENT_FEASIBILITY_LABELS,
) -> Mapping[str, Any]:
    """Compare fixed, equal-unit, and MGDA-style directions from a Gram matrix."""

    names = tuple(labels)
    matrix = np.asarray(gram, dtype=np.float64)
    if (
        matrix.shape != (len(names), len(names))
        or len(set(names)) != len(names)
        or np.any(~np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12)
        or np.any(np.diag(matrix) <= 0.0)
    ):
        raise ValueError("gradient feasibility Gram matrix is invalid")
    matrix = 0.5 * (matrix + matrix.T)
    norms = np.sqrt(np.diag(matrix))
    normalized = matrix / (norms[:, None] * norms[None, :])
    normalized = 0.5 * (normalized + normalized.T)
    np.fill_diagonal(normalized, 1.0)

    fixed_weights = np.ones(len(names), dtype=np.float64)
    equal_unit_weights = 1.0 / norms
    minimum_norm_weights = _minimum_norm_simplex_weights(normalized) / norms
    return {
        "gradient_norms": {
            label: float(norms[index]) for index, label in enumerate(names)
        },
        "pairwise_cosines": {
            left: {
                right: float(normalized[left_index, right_index])
                for right_index, right in enumerate(names)
            }
            for left_index, left in enumerate(names)
        },
        "directions": {
            "fixed_weight_primary_proxy": _direction_record(
                matrix,
                fixed_weights,
                names,
            ),
            "equal_unit": _direction_record(
                matrix,
                equal_unit_weights,
                names,
            ),
            "minimum_norm_unit": _direction_record(
                matrix,
                minimum_norm_weights,
                names,
            )
            | {
                "normalized_simplex_weights": {
                    label: float(minimum_norm_weights[index] * norms[index])
                    for index, label in enumerate(names)
                }
            },
        },
    }


def classify_gradient_feasibility(
    records: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    labels: Sequence[str] = GRADIENT_FEASIBILITY_LABELS,
) -> Mapping[str, Any]:
    names = tuple(labels)
    minimum_group_cosine = float(gates["minimum_group_component_cosine"])
    minimum_median_cosine = float(gates["minimum_component_median_cosine"])
    required_groups = int(gates["minimum_passing_groups"])
    if not records or required_groups < 1 or required_groups > len(records):
        raise ValueError("gradient feasibility group gate is invalid")

    directions = {}
    for direction_name in TRAINABLE_DIRECTION_NAMES:
        group_records = []
        for record in records:
            direction = record["diagnostics"]["directions"][direction_name]
            cosines = direction["normalized_directional_derivatives"]
            if tuple(cosines) != names:
                raise ValueError("gradient feasibility component registry drift")
            minimum = min(float(cosines[name]) for name in names)
            group_records.append(
                {
                    "group": int(record["group"]),
                    "minimum_component_cosine": minimum,
                    "passed": minimum >= minimum_group_cosine,
                }
            )
        medians = {
            name: float(
                np.median(
                    [
                        record["diagnostics"]["directions"][direction_name][
                            "normalized_directional_derivatives"
                        ][name]
                        for record in records
                    ]
                )
            )
            for name in names
        }
        passing_groups = sum(record["passed"] for record in group_records)
        passed = passing_groups >= required_groups and all(
            value >= minimum_median_cosine for value in medians.values()
        )
        directions[direction_name] = {
            "passed": passed,
            "passing_groups": passing_groups,
            "required_passing_groups": required_groups,
            "component_median_cosines": medians,
            "groups": group_records,
        }

    selected = next(
        (
            name
            for name in TRAINABLE_DIRECTION_NAMES
            if directions[name]["passed"]
        ),
        None,
    )
    return {
        "status": (
            "common_descent_direction_supported"
            if selected is not None
            else "common_descent_direction_not_supported"
        ),
        "passed": selected is not None,
        "selected_direction": selected,
        "directions": directions,
    }


def classify_realized_step_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    labels: Sequence[str] = GRADIENT_FEASIBILITY_LABELS,
) -> Mapping[str, Any]:
    """Classify whether AdamW realizes the selected raw descent direction."""

    names = tuple(labels)
    required_groups = int(gates["minimum_passing_groups"])
    total_groups = int(gates["total_groups"])
    if len(records) != total_groups or not 1 <= required_groups <= total_groups:
        raise ValueError("realized-step group gate is invalid")

    raw_threshold = float(gates["minimum_raw_component_cosine"])
    realized_threshold = float(gates["minimum_realized_component_cosine"])
    maximum_loss_increase = float(
        gates["maximum_virtual_component_relative_increase"]
    )
    raw_groups = sum(
        float(record["raw_direction"]["minimum_component_cosine"])
        >= raw_threshold
        for record in records
    )
    optimizers = {}
    for optimizer_name in REALIZED_STEP_OPTIMIZERS:
        direction_groups = 0
        virtual_loss_groups = 0
        group_records = []
        for record in records:
            optimizer = record["optimizers"][optimizer_name]
            cosines = optimizer["component_cosines"]
            relative_changes = optimizer["virtual_loss_relative_changes"]
            if tuple(cosines) != names or tuple(relative_changes) != names:
                raise ValueError("realized-step component registry drift")
            minimum_cosine = min(float(cosines[name]) for name in names)
            maximum_increase = max(float(relative_changes[name]) for name in names)
            direction_passed = minimum_cosine >= realized_threshold
            loss_passed = maximum_increase <= maximum_loss_increase
            direction_groups += int(direction_passed)
            virtual_loss_groups += int(loss_passed)
            group_records.append(
                {
                    "group": int(record["group"]),
                    "minimum_component_cosine": minimum_cosine,
                    "maximum_virtual_component_relative_increase": maximum_increase,
                    "direction_passed": direction_passed,
                    "virtual_losses_passed": loss_passed,
                }
            )
        optimizers[optimizer_name] = {
            "passed": direction_groups >= required_groups
            and virtual_loss_groups >= required_groups,
            "direction_passing_groups": direction_groups,
            "virtual_loss_passing_groups": virtual_loss_groups,
            "required_passing_groups": required_groups,
            "groups": group_records,
        }

    raw_passed = raw_groups >= required_groups
    inherited_passed = optimizers["inherited_adamw"]["passed"]
    reset_passed = optimizers["reset_adamw"]["passed"]
    if not raw_passed:
        status = "raw_common_descent_drifted"
        next_action = "stop_and_review_component_gradient_definition"
    elif inherited_passed:
        status = "inherited_adamw_handoff_supported"
        next_action = "stop_optimizer_changes_and_audit_batch_to_rollout_generalization"
    elif reset_passed:
        status = "inherited_adamw_moments_destroy_common_descent"
        next_action = "freeze_matched_reset_optimizer_smoke_before_training"
    else:
        status = "adamw_realization_not_supported"
        next_action = "design_non_adamw_direction_realization_before_training"
    return {
        "status": status,
        "raw_direction_passed": raw_passed,
        "raw_direction_passing_groups": raw_groups,
        "required_passing_groups": required_groups,
        "optimizers": optimizers,
        "next_action": next_action,
    }
