"""Milestone validation and selection for staged successor training."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .successor_rollout_screen import _finalize_rollout_statistics
from .successor_training import CUMULATIVE_FLUX_FIELDS, TERMINAL_STOCK_FIELDS

REQUIRED_VALIDATION_SLICES = ("temporal", "spatial", "joint")
REQUIRED_VALIDATION_HORIZONS = (7, 30)


def primary_metric_keys(primary_outputs: Sequence[str]) -> tuple[str, ...]:
    fluxes = set(CUMULATIVE_FLUX_FIELDS)
    stocks = set(TERMINAL_STOCK_FIELDS)
    keys = tuple(
        f"cumulative.{name}"
        if name in fluxes
        else f"terminal.{name}"
        if name in stocks
        else ""
        for name in primary_outputs
    )
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("successor staged primary-output registry drift")
    return keys


def finalize_staged_validation_statistics(
    statistics: Any,
    *,
    primary_outputs: Sequence[str],
) -> Mapping[str, Any]:
    return _finalize_rollout_statistics(
        statistics,
        primary_keys=primary_metric_keys(primary_outputs),
    )


def validation_primary_composite(
    arm_report: Mapping[str, Any],
    *,
    primary_outputs: Sequence[str],
) -> float:
    keys = primary_metric_keys(primary_outputs)
    squared = []
    for slice_id in REQUIRED_VALIDATION_SLICES:
        cell = arm_report["slices"][slice_id]
        for horizon in REQUIRED_VALIDATION_HORIZONS:
            metrics = cell["horizons"][str(horizon)]["keys"]
            squared.extend(float(metrics[key]["normalized_rmse"]) ** 2 for key in keys)
    result = float(np.sqrt(np.mean(np.asarray(squared, dtype=np.float64))))
    if not np.isfinite(result):
        raise ValueError("successor staged validation composite is nonfinite")
    return result


def validation_hard_failure_count(arm_report: Mapping[str, Any]) -> int:
    return sum(
        int(value)
        for slice_id in REQUIRED_VALIDATION_SLICES
        for value in arm_report["slices"][slice_id]["hard_counts"].values()
    )


def advance_staged_validation_state(
    previous: Mapping[str, Any] | None,
    *,
    milestone: int,
    arm_reports: Mapping[str, Mapping[str, Any]],
    contract: Any,
) -> Mapping[str, Any]:
    arms = tuple(contract.payload["matched_arms"])
    milestones = tuple(int(value) for value in contract.sampling["stage_milestones"])
    completed = () if previous is None else tuple(int(value) for value in previous["completed_milestones"])
    expected_position = len(completed)
    if (
        expected_position >= len(milestones)
        or int(milestone) != milestones[expected_position]
        or set(arm_reports) != set(arms)
    ):
        raise ValueError("successor staged milestone order drift")
    minimum_improvement = float(contract.payload["validation"]["minimum_relative_improvement"])
    previous_arms = {} if previous is None else previous["arms"]
    next_arms = {}
    milestone_arms = {}
    for arm in arms:
        composite = validation_primary_composite(
            arm_reports[arm],
            primary_outputs=contract.payload["validation"]["primary_outputs"],
        )
        hard_failures = validation_hard_failure_count(arm_reports[arm])
        prior = previous_arms.get(arm)
        prior_best = None if prior is None else float(prior["best_primary_composite"])
        improved = hard_failures == 0 and (
            prior_best is None
            or composite <= prior_best * (1.0 - minimum_improvement)
        )
        best = composite if improved else prior_best
        best_milestone = (
            int(milestone)
            if improved
            else None
            if prior is None
            else prior["best_milestone"]
        )
        misses = 0 if improved else (0 if prior is None else int(prior["consecutive_no_improvement"])) + 1
        next_arms[arm] = {
            "best_milestone": best_milestone,
            "best_primary_composite": best,
            "consecutive_no_improvement": misses,
        }
        milestone_arms[arm] = {
            "primary_composite": composite,
            "hard_failure_count": hard_failures,
            "improved": improved,
        }
    stop_for_hard_failure = any(
        milestone_arms[arm]["hard_failure_count"] > 0 for arm in arms
    )
    stop_for_plateau = all(
        int(next_arms[arm]["consecutive_no_improvement"]) >= 2 for arm in arms
    )
    stop_for_budget = int(milestone) == milestones[-1]
    record = {"milestone": int(milestone), "arms": milestone_arms}
    history = [] if previous is None else list(previous["history"])
    history.append(record)
    return {
        "schema_version": "successor_staged_validation_state_v1",
        "staged_training_contract_sha256": contract.sha256,
        "completed_milestones": [*completed, int(milestone)],
        "arms": next_arms,
        "history": history,
        "stop_training": stop_for_hard_failure or stop_for_plateau or stop_for_budget,
        "stop_reason": (
            "validation_hard_failure"
            if stop_for_hard_failure
            else "both_arms_plateaued_twice"
            if stop_for_plateau
            else "maximum_milestone_reached"
            if stop_for_budget
            else None
        ),
        "next_milestone": (
            None
            if stop_for_hard_failure or stop_for_plateau or stop_for_budget
            else milestones[expected_position + 1]
        ),
    }
